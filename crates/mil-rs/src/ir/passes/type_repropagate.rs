//! Post-optimization type re-propagation pass.
//!
//! Optimization passes (layout optimization, op splitting, etc.) insert new
//! operations without setting [`output_types`]. When the proto serializer
//! encounters an op without stored types, it falls back to inference — and
//! for shape-changing ops (conv, transpose, reshape, …) it emits unknown
//! dimensions, which can crash the CoreML runtime.
//!
//! This pass walks every operation in program order, builds a type map from
//! function inputs and earlier outputs, and fills in any missing
//! `output_types` entries.

use std::collections::HashMap;

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::shape_inference::{
    infer_concat_output, infer_conv_output, infer_pool_output, read_int_list,
};
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::Value;

/// Re-propagate output types for operations that are missing them.
///
/// Should be scheduled **after** all fusion and optimization passes so that
/// every newly-created op (transposes from layout optimization, tiles from
/// op splitting, etc.) gets concrete output types before serialization.
pub struct TypeRepropagationPass;

impl Pass for TypeRepropagationPass {
    fn name(&self) -> &str {
        "type-repropagation"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            let mut type_map: HashMap<String, TensorType> = HashMap::new();

            // Seed with function inputs.
            for (name, tt) in &function.inputs {
                type_map.insert(name.clone(), tt.clone());
            }

            for op in &mut function.body.operations {
                // Register any outputs that already have types.
                for (i, out_name) in op.outputs.iter().enumerate() {
                    if let Some(Some(tt)) = op.output_types.get(i) {
                        type_map.insert(out_name.clone(), tt.clone());
                    }
                }

                // For const ops, derive from val if output_types is missing.
                if op.op_type == "const" {
                    let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
                    if let Some(tt) = val.and_then(tensor_type_for_value) {
                        for out_name in &op.outputs {
                            type_map.entry(out_name.clone()).or_insert(tt.clone());
                        }
                    }
                    continue;
                }
                if op.op_type.starts_with("constexpr_") {
                    continue;
                }

                // Fill in missing output types.
                for (i, out_name) in op.outputs.iter().enumerate() {
                    if op.output_types.get(i).is_some_and(|ot| ot.is_some()) {
                        continue;
                    }

                    let inferred = infer_output_type(op, &type_map);
                    if let Some(tt) = inferred {
                        type_map.insert(out_name.clone(), tt.clone());
                        while op.output_types.len() <= i {
                            op.output_types.push(None);
                        }
                        op.output_types[i] = Some(tt);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Infer the output type for an operation based on its inputs and the type map.
fn infer_output_type(op: &Operation, type_map: &HashMap<String, TensorType>) -> Option<TensorType> {
    match op.op_type.as_str() {
        "transpose" => infer_transpose_output(op, type_map),
        "conv" => infer_conv_output(op, type_map),
        "max_pool" | "avg_pool" => infer_pool_output(op, type_map),
        "concat" => infer_concat_output(op, type_map),
        "tile" => infer_tile_output(op, type_map),
        "reduce_mean" | "reduce_sum" | "reduce_max" | "reduce_min" | "reduce_prod"
        | "reduce_l1" | "reduce_l2" => infer_reduce_output(op, type_map),
        "gather" => infer_gather_output(op, type_map),
        "reshape" => infer_reshape_output(op, type_map),
        "expand_dims" => infer_expand_dims_output(op, type_map),
        "squeeze" => infer_squeeze_output(op, type_map),
        _ => {
            // Element-wise / pass-through: output type matches primary input.
            resolve_primary_input(op, type_map)
        }
    }
}

/// Resolve the output type for element-wise / pass-through ops.
///
/// For binary ops (add, mul, real_div, …) the output rank equals the
/// maximum input rank due to broadcasting. We collect *all* resolved
/// input types and return the one with the highest rank.
fn resolve_primary_input(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<TensorType> {
    let resolve = |v: &Value| -> Option<&TensorType> {
        match v {
            Value::Reference(name) => type_map.get(name),
            Value::List(items) => items.iter().find_map(|item| {
                if let Value::Reference(name) = item {
                    type_map.get(name)
                } else {
                    None
                }
            }),
            _ => None,
        }
    };

    // Gather all resolvable input types, preferring high-priority keys first.
    let mut best: Option<TensorType> = None;
    for &key in &["x", "data", "input", "values"] {
        if let Some(tt) = op.inputs.get(key).and_then(resolve) {
            match &best {
                Some(b) if b.shape.len() >= tt.shape.len() => {}
                _ => best = Some(tt.clone()),
            }
        }
    }
    // Also consider non-primary inputs (e.g., "y" for binary ops).
    for tt in op.inputs.values().filter_map(resolve) {
        match &best {
            Some(b) if b.shape.len() >= tt.shape.len() => {}
            _ => best = Some(tt.clone()),
        }
    }
    best
}

/// Infer transpose output: permute the input dimensions.
fn infer_transpose_output(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<TensorType> {
    let in_tt = op.inputs.get("x").and_then(|v| {
        if let Value::Reference(n) = v {
            type_map.get(n)
        } else {
            None
        }
    })?;

    let perm = read_int_list(op.inputs.get("perm")?)?;

    if perm.len() != in_tt.shape.len() {
        return Some(in_tt.clone());
    }

    let permuted: Vec<Option<usize>> = perm
        .iter()
        .map(|&p| in_tt.shape.get(p as usize).copied().unwrap_or(None))
        .collect();

    Some(TensorType::with_dynamic_shape(in_tt.scalar_type, permuted))
}

/// Infer tile output: multiply input dimensions by reps.
fn infer_tile_output(op: &Operation, type_map: &HashMap<String, TensorType>) -> Option<TensorType> {
    let input_type = resolve_primary_input(op, type_map)?;

    // reps can be an inline tensor or a reference to a const op.
    let reps_value = op.inputs.get("reps")?;
    let reps = match reps_value {
        Value::Tensor { .. } => read_int_list(reps_value),
        Value::Reference(_) => {
            // reps is a reference — read from op.attributes as fallback,
            // or try to infer from the existing output type and input type.
            // For now, return the existing output type unmodified if we
            // can't resolve the reference.
            None
        }
        _ => None,
    };

    if let Some(reps) = reps {
        let mut out_shape = input_type.shape.clone();
        for (i, &rep) in reps.iter().enumerate() {
            if i < out_shape.len() {
                if let Some(d) = out_shape[i] {
                    out_shape[i] = Some(d * rep as usize);
                }
            }
        }
        Some(TensorType::with_dynamic_shape(
            input_type.scalar_type,
            out_shape,
        ))
    } else {
        // Can't resolve reps — return None to signal type inference failure.
        None
    }
}

/// Infer reduce op output: set reduced axes to 1 when keep_dims=true,
/// or remove them when keep_dims=false.
fn infer_reduce_output(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<TensorType> {
    let in_tt = op.inputs.get("x").and_then(|v| {
        if let Value::Reference(n) = v {
            type_map.get(n)
        } else {
            None
        }
    })?;

    // Read axes from inline tensor or reference.
    let axes_value = op.inputs.get("axes")?;
    let axes = match axes_value {
        Value::Tensor { .. } => read_int_list(axes_value)?,
        Value::Reference(_name) => {
            // Try to find the const op that defines this reference.
            // We don't have access to the const map here, so check type_map
            // for shape info. Fall back to input type if we can't resolve.
            return resolve_primary_input(op, type_map);
        }
        Value::List(items) => items
            .iter()
            .filter_map(|v| match v {
                Value::Int(i) => Some(*i),
                _ => None,
            })
            .collect(),
        _ => return resolve_primary_input(op, type_map),
    };

    let keep_dims = op
        .inputs
        .get("keep_dims")
        .or_else(|| op.attributes.get("keep_dims"))
        .and_then(|v| match v {
            Value::Bool(b) => Some(*b),
            Value::Reference(_) => Some(true), // default
            _ => None,
        })
        .unwrap_or(true);

    let ndim = in_tt.shape.len() as i64;
    let mut out_shape = in_tt.shape.clone();

    if keep_dims {
        for &axis in &axes {
            let a = if axis < 0 {
                (ndim + axis) as usize
            } else {
                axis as usize
            };
            if a < out_shape.len() {
                out_shape[a] = Some(1);
            }
        }
    } else {
        // Remove reduced axes (sort descending to avoid index shift).
        let mut sorted_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (ndim + a) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        sorted_axes.sort_unstable();
        sorted_axes.dedup();
        for &a in sorted_axes.iter().rev() {
            if a < out_shape.len() {
                out_shape.remove(a);
            }
        }
    }

    Some(TensorType::with_dynamic_shape(in_tt.scalar_type, out_shape))
}

/// Infer gather output: output_rank = rank(x) - 1 + rank(indices).
fn infer_gather_output(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<TensorType> {
    let resolve_ref = |v: &Value| -> Option<&TensorType> {
        if let Value::Reference(name) = v {
            type_map.get(name)
        } else {
            None
        }
    };
    let x_tt = op.inputs.get("x").and_then(resolve_ref)?;
    let idx_tt = op.inputs.get("indices").and_then(resolve_ref)?;
    let out_rank = x_tt.shape.len() - 1 + idx_tt.shape.len();
    Some(TensorType::with_dynamic_shape(
        x_tt.scalar_type,
        vec![None; out_rank],
    ))
}

/// Infer reshape output: rank = number of elements in the shape tensor.
fn infer_reshape_output(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<TensorType> {
    let in_tt = op.inputs.get("x").and_then(|v| {
        if let Value::Reference(n) = v {
            type_map.get(n)
        } else {
            None
        }
    })?;

    // shape can be an inline tensor, a list of ints, or a reference to a const.
    let shape_val = op.inputs.get("shape")?;
    let out_rank = match shape_val {
        Value::Tensor { shape, .. } => {
            // 1-D tensor: the first dim is the output rank.
            shape.first().copied()
        }
        Value::List(items) => Some(items.len()),
        Value::Reference(name) => {
            // Look up the shape const's type: it's a 1-D tensor whose first
            // dimension tells us how many output dims there are.
            let tt = type_map.get(name)?;
            tt.shape.first().and_then(|d| *d)
        }
        _ => None,
    }?;
    Some(TensorType::with_dynamic_shape(
        in_tt.scalar_type,
        vec![None; out_rank],
    ))
}

/// Infer expand_dims output: rank = rank(x) + number_of_axes.
fn infer_expand_dims_output(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<TensorType> {
    let in_tt = op.inputs.get("x").and_then(|v| {
        if let Value::Reference(n) = v {
            type_map.get(n)
        } else {
            None
        }
    })?;
    let num_axes = match op.inputs.get("axes").or_else(|| op.attributes.get("axes")) {
        Some(Value::List(items)) => items.len(),
        Some(Value::Int(_)) => 1,
        Some(Value::Tensor { shape, .. }) => shape.first().copied().unwrap_or(1),
        _ => return Some(in_tt.clone()),
    };
    Some(TensorType::with_dynamic_shape(
        in_tt.scalar_type,
        vec![None; in_tt.shape.len() + num_axes],
    ))
}

/// Infer squeeze output: rank = rank(x) - number_of_axes.
fn infer_squeeze_output(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<TensorType> {
    let in_tt = op.inputs.get("x").and_then(|v| {
        if let Value::Reference(n) = v {
            type_map.get(n)
        } else {
            None
        }
    })?;
    let num_axes = match op.inputs.get("axes").or_else(|| op.attributes.get("axes")) {
        Some(Value::List(items)) => items.len(),
        Some(Value::Int(_)) => 1,
        Some(Value::Tensor { shape, .. }) => shape.first().copied().unwrap_or(1),
        _ => return Some(in_tt.clone()),
    };
    let out_rank = in_tt.shape.len().saturating_sub(num_axes);
    Some(TensorType::with_dynamic_shape(
        in_tt.scalar_type,
        vec![None; out_rank],
    ))
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Extract a `TensorType` from a `Value` (for const ops).
fn tensor_type_for_value(value: &Value) -> Option<TensorType> {
    match value {
        Value::Tensor { shape, dtype, .. } => Some(TensorType::new(*dtype, shape.clone())),
        Value::List(items) if !items.is_empty() => {
            // Lists of ints are serialized as 1-D Int32 tensors.
            if items.iter().all(|v| matches!(v, Value::Int(_))) {
                Some(TensorType::new(ScalarType::Int32, vec![items.len()]))
            } else if items.iter().all(|v| matches!(v, Value::Float(_))) {
                Some(TensorType::new(ScalarType::Float32, vec![items.len()]))
            } else if items.iter().all(|v| matches!(v, Value::Bool(_))) {
                Some(TensorType::new(ScalarType::Bool, vec![items.len()]))
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::program::Function;
    use crate::ir::tensor::ScalarType;
    use crate::ir::types::TensorData;

    fn run_pass(program: &mut Program) {
        TypeRepropagationPass.run(program).unwrap();
    }

    #[test]
    fn fills_transpose_output_type() {
        let mut program = Program::new("1.0.0");
        let input_tt = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);
        let mut func = Function::new("main").with_input("image", input_tt);

        // transpose NCHW→NHWC with no output_types
        let perm_data: Vec<u8> = [0i32, 2, 3, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        func.body.add_op(
            Operation::new("transpose", "t0")
                .with_input("x", Value::Reference("image".into()))
                .with_input(
                    "perm",
                    Value::Tensor {
                        data: TensorData::Inline(perm_data),
                        shape: vec![4],
                        dtype: ScalarType::Int32,
                    },
                )
                .with_output("nhwc_out"),
        );
        func.body.outputs.push("nhwc_out".into());
        program.add_function(func);

        assert!(
            program.functions["main"].body.operations[0].output_types[0].is_none(),
            "should start with no output type"
        );

        run_pass(&mut program);

        let tt = program.functions["main"].body.operations[0]
            .output_types
            .first()
            .unwrap()
            .as_ref()
            .expect("should have output type");
        assert_eq!(
            tt.shape,
            vec![Some(1), Some(224), Some(224), Some(3)],
            "NCHW [1,3,224,224] → NHWC [1,224,224,3]"
        );
        assert_eq!(tt.scalar_type, ScalarType::Float32);
    }

    #[test]
    fn preserves_existing_output_types() {
        let mut program = Program::new("1.0.0");
        let input_tt = TensorType::new(ScalarType::Float32, vec![1, 10]);
        let mut func = Function::new("main").with_input("x", input_tt);

        let existing = TensorType::new(ScalarType::Float32, vec![1, 10]);
        let mut op = Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("x".into()))
            .with_output("relu_out");
        op.output_types.push(Some(existing.clone()));
        func.body.add_op(op);
        func.body.outputs.push("relu_out".into());
        program.add_function(func);

        run_pass(&mut program);

        let tt = program.functions["main"].body.operations[0].output_types[0]
            .as_ref()
            .unwrap();
        assert_eq!(tt.shape, vec![Some(1), Some(10)]);
    }

    #[test]
    fn fills_element_wise_op_missing_type() {
        let mut program = Program::new("1.0.0");
        let input_tt = TensorType::new(ScalarType::Float32, vec![1, 64, 56, 56]);
        let mut func = Function::new("main").with_input("x", input_tt);

        // relu with no output_types
        func.body.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("x".into()))
                .with_output("relu_out"),
        );
        func.body.outputs.push("relu_out".into());
        program.add_function(func);

        run_pass(&mut program);

        let tt = program.functions["main"].body.operations[0]
            .output_types
            .first()
            .unwrap()
            .as_ref()
            .expect("relu should get output type from input");
        assert_eq!(tt.shape, vec![Some(1), Some(64), Some(56), Some(56)]);
    }

    #[test]
    fn fills_conv_output_type() {
        let mut program = Program::new("1.0.0");
        let input_tt = TensorType::new(ScalarType::Float32, vec![1, 3, 8, 8]);
        let mut func = Function::new("main").with_input("x", input_tt);

        // weight const: [16, 3, 3, 3]
        let w_data = vec![0u8; 16 * 3 * 3 * 3 * 4];
        func.body.add_op(
            Operation::new("const", "w_const")
                .with_input(
                    "val",
                    Value::Tensor {
                        data: TensorData::Inline(w_data),
                        shape: vec![16, 3, 3, 3],
                        dtype: ScalarType::Float32,
                    },
                )
                .with_output("w"),
        );

        let pad_data: Vec<u8> = [0i32, 0, 0, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let stride_data: Vec<u8> = [1i32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
        let dil_data: Vec<u8> = [1i32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();

        // conv with no output_types
        func.body.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("x".into()))
                .with_input("weight", Value::Reference("w".into()))
                .with_input(
                    "pad",
                    Value::Tensor {
                        data: TensorData::Inline(pad_data),
                        shape: vec![4],
                        dtype: ScalarType::Int32,
                    },
                )
                .with_input(
                    "strides",
                    Value::Tensor {
                        data: TensorData::Inline(stride_data),
                        shape: vec![2],
                        dtype: ScalarType::Int32,
                    },
                )
                .with_input(
                    "dilations",
                    Value::Tensor {
                        data: TensorData::Inline(dil_data),
                        shape: vec![2],
                        dtype: ScalarType::Int32,
                    },
                )
                .with_output("conv_out"),
        );
        func.body.outputs.push("conv_out".into());
        program.add_function(func);

        run_pass(&mut program);

        let conv = &program.functions["main"].body.operations[1];
        let tt = conv
            .output_types
            .first()
            .unwrap()
            .as_ref()
            .expect("conv should get inferred output type");
        // Input [1,3,8,8] with 3x3 kernel, no padding, stride 1 → [1,16,6,6]
        assert_eq!(tt.shape, vec![Some(1), Some(16), Some(6), Some(6)]);
    }
}
