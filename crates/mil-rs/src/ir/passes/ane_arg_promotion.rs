//! Promote inline op arguments to named const ops for ANE compatibility.
//!
//! The ANE compiler requires certain op parameters (e.g., `axes` and
//! `keep_dims` for reduce ops, `axis` for softmax) to be passed as
//! references to named const ops, not as inline literal values.
//!
//! This pass scans for ops with inline `Value::Tensor`, `Value::Bool`,
//! `Value::Int`, or `Value::List` arguments in specific parameter slots
//! and promotes them to standalone `const` operations with
//! `Value::Reference` links.
//!
//! Must run **before** `AneLayoutPass` so that the promoted const ops
//! are properly handled (the layout pass skips const ops, preventing
//! corruption of parameter tensors like axis vectors).

use std::collections::HashMap;

use crate::error::Result;
use crate::ir::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::Value;

/// Promotes inline reduction/softmax arguments to named const ops.
pub struct AneArgPromotionPass;

/// Op types whose parameters need promotion.
const REDUCE_OPS: &[&str] = &[
    "reduce_mean",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_l2_norm",
    "reduce_l1_norm",
    "reduce_log_sum",
    "reduce_log_sum_exp",
    "reduce_sum_square",
    "reduce_prod",
];

/// Parameter names that need promotion for reduce ops.
const REDUCE_PARAMS: &[&str] = &["axes", "keep_dims"];

/// Parameter names that need promotion for softmax.
const SOFTMAX_PARAMS: &[&str] = &["axis"];

/// Parameter names that need promotion for layer_norm.
const LAYER_NORM_PARAMS: &[&str] = &["axes", "epsilon"];

impl Pass for AneArgPromotionPass {
    fn name(&self) -> &str {
        "ane-arg-promotion"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            promote_inline_args(&mut function.body.operations);
        }
        Ok(())
    }
}

/// Scan operations and promote inline args to const ops.
///
/// Also remaps reduce axes for ANE layout: in ANE's `[1,C,1,S]` layout,
/// the hidden dimension is axis 1 (C), not axis -1 (S). When a reduce op
/// has `axes=[-1]` and operates on a tensor where S=1 (seq_len=1
/// autoregressive decode), the reduction is a no-op. Remap to axes=[1]
/// to reduce over the channel (hidden) dimension instead.
fn promote_inline_args(operations: &mut Vec<Operation>) {
    // Collect promotions first (new const ops to insert, and references to patch).
    let mut const_ops: Vec<(usize, Operation)> = Vec::new();
    // Map: (op_index, param_name) → const output name
    let mut patches: Vec<(usize, String, String, bool)> = Vec::new(); // (idx, param, const_name, is_attr)

    for (i, op) in operations.iter().enumerate() {
        let params_to_check: &[&str] = if REDUCE_OPS.contains(&op.op_type.as_str()) {
            REDUCE_PARAMS
        } else if op.op_type == "softmax" {
            SOFTMAX_PARAMS
        } else if op.op_type == "layer_norm" {
            LAYER_NORM_PARAMS
        } else {
            continue;
        };

        for &param in params_to_check {
            // Check both inputs and attributes for the parameter.
            let (value, is_attr) = if let Some(v) = op.inputs.get(param) {
                (v, false)
            } else if let Some(v) = op.attributes.get(param) {
                (v, true)
            } else {
                continue;
            };

            // Only promote non-reference values (inline literals).
            if matches!(value, Value::Reference(_)) {
                continue;
            }

            // Remap axes for ANE layout: axis -1 → axis 1 when the
            // output tensor's last dim (S) is 1 (autoregressive decode).
            let value = if param == "axes" {
                remap_ane_reduce_axes(value, op)
            } else {
                value.clone()
            };

            let const_name = format!("{}_{}_const", op.name, param);
            let const_op = build_const_for_value(&const_name, &value);

            if let Some(const_op) = const_op {
                const_ops.push((i, const_op));
                patches.push((i, param.to_string(), const_name, is_attr));
            }
        }
    }

    // Insert const ops before their consumers (in reverse order to
    // maintain correct indices).
    // First, sort const_ops by insertion point.
    const_ops.sort_by_key(|(idx, _)| *idx);

    // Track how many insertions we've done before each index.
    let mut insertion_offsets: HashMap<usize, usize> = HashMap::new();

    for (offset, (target_idx, const_op)) in const_ops.into_iter().enumerate() {
        let insert_at = target_idx + offset;
        operations.insert(insert_at, const_op);
        insertion_offsets
            .entry(target_idx)
            .and_modify(|o| *o += 1)
            .or_insert(1);
    }

    // Apply patches: replace inline values with references.
    for (orig_idx, param, const_name, is_attr) in patches {
        // Compute the adjusted index after insertions.
        let extra: usize = insertion_offsets
            .iter()
            .filter(|&(&k, _)| k <= orig_idx)
            .map(|(_, v)| v)
            .sum();
        let actual_idx = orig_idx + extra;

        let op = &mut operations[actual_idx];
        let ref_val = Value::Reference(const_name);

        if is_attr {
            // Move from attributes to inputs (ANE expects named input refs).
            op.attributes.remove(&param);
            op.inputs.insert(param, ref_val);
        } else {
            op.inputs.insert(param, ref_val);
        }
    }
}

/// Remap reduce axes for ANE layout.
///
/// In ANE's `[1,C,1,S]` layout, the original hidden dimension (axis -1
/// in `[B,S,H]`) moves to axis 1 (C). When a reduce/norm op still has
/// `axes=[-1]`, it reduces over S instead of the hidden dimension.
/// This function remaps axis -1 to axis 1 for ANE-layout ops.
fn remap_ane_reduce_axes(value: &Value, _op: &Operation) -> Value {
    match value {
        Value::Tensor { data, shape, dtype }
            if *dtype == ScalarType::Int32 && shape.iter().product::<usize>() == 1 =>
        {
            if data.len() >= 4 {
                let axis = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                if axis == -1 || axis == 3 {
                    let new_data = 1i32.to_le_bytes().to_vec();
                    return Value::Tensor {
                        data: new_data,
                        shape: shape.clone(),
                        dtype: *dtype,
                    };
                }
            }
            value.clone()
        }
        Value::List(items)
            if items.len() == 1 && matches!(&items[0], Value::Int(n) if *n == -1 || *n == 3) =>
        {
            Value::List(vec![Value::Int(1)])
        }
        _ => value.clone(),
    }
}

/// Build a const op that produces the given value.
fn build_const_for_value(name: &str, value: &Value) -> Option<Operation> {
    let mut op = Operation::new("const", name).with_output(name);

    match value {
        Value::Tensor { data, shape, dtype } => {
            // Force 1-D shape — parameter tensors (axes, shapes) must
            // be 1-D vectors, regardless of any prior layout reshaping.
            let num_elements: usize = shape.iter().product();
            let flat_shape = vec![num_elements];
            op.inputs.insert(
                "val".to_string(),
                Value::Tensor {
                    data: data.clone(),
                    shape: flat_shape.clone(),
                    dtype: *dtype,
                },
            );
            op.output_types = vec![Some(TensorType::new(*dtype, flat_shape))];
        }
        Value::Bool(b) => {
            op.inputs.insert("val".to_string(), Value::Bool(*b));
            // Bool consts don't need output_types — the emitter handles them.
        }
        Value::Int(n) => {
            op.inputs.insert("val".to_string(), Value::Int(*n));
        }
        Value::Float(f) => {
            op.inputs.insert("val".to_string(), Value::Float(*f));
        }
        Value::List(items) if items.iter().all(|v| matches!(v, Value::Int(_))) => {
            // Convert int list to tensor.
            let ints: Vec<i64> = items
                .iter()
                .filter_map(|v| match v {
                    Value::Int(n) => Some(*n),
                    _ => None,
                })
                .collect();
            let data: Vec<u8> = ints
                .iter()
                .flat_map(|&v| (v as i32).to_le_bytes())
                .collect();
            let shape = vec![ints.len()];
            op.inputs.insert(
                "val".to_string(),
                Value::Tensor {
                    data,
                    shape: shape.clone(),
                    dtype: ScalarType::Int32,
                },
            );
            op.output_types = vec![Some(TensorType::new(ScalarType::Int32, shape))];
        }
        _ => return None,
    }

    Some(op)
}
