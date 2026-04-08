//! Shared shape / type inference helpers for MIL IR.
//!
//! These helpers compute output shapes for common operations (conv, pool, concat)
//! and are shared between the ONNX import path and the type re-propagation pass.

use std::collections::HashMap;

use crate::ir::operation::Operation;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::Value;

/// Read an integer list from a [`Value`].
///
/// Handles both `Value::List(Vec<Value::Int>)` (used by the ONNX import path) and
/// `Value::Tensor { Int32 data }` (used after constant folding).
pub(crate) fn read_int_list(value: &Value) -> Option<Vec<i64>> {
    match value {
        Value::List(items) => {
            let ints: Vec<i64> = items
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .collect();
            if ints.is_empty() { None } else { Some(ints) }
        }
        Value::Tensor {
            data,
            dtype: ScalarType::Int32,
            ..
        } => {
            let bytes = data.as_bytes()?;
            Some(
                bytes
                    .chunks_exact(4)
                    .map(|c: &[u8]| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect(),
            )
        }
        _ => None,
    }
}

/// Read an integer list from an operation input or attribute.
fn read_int_list_from_op(op: &Operation, name: &str) -> Option<Vec<i64>> {
    op.inputs
        .get(name)
        .or_else(|| op.attributes.get(name))
        .and_then(read_int_list)
}

/// Infer conv output shape: `[N, out_channels, out_h, out_w]`.
pub(crate) fn infer_conv_output(
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
    let w_tt = op.inputs.get("weight").and_then(|v| {
        if let Value::Reference(n) = v {
            type_map.get(n)
        } else {
            None
        }
    })?;

    if in_tt.shape.len() != 4 || w_tt.shape.len() != 4 {
        return Some(in_tt.clone());
    }

    let out_channels = w_tt.shape[0]?;
    let in_h = in_tt.shape[2]?;
    let in_w = in_tt.shape[3]?;
    let k_h = w_tt.shape[2]?;
    let k_w = w_tt.shape[3]?;

    let strides = read_int_list_from_op(op, "strides").unwrap_or_else(|| vec![1, 1]);
    let dilations = read_int_list_from_op(op, "dilations").unwrap_or_else(|| vec![1, 1]);
    let pads = read_int_list_from_op(op, "pad").unwrap_or_else(|| vec![0, 0, 0, 0]);

    let pad_top = *pads.first().unwrap_or(&0) as usize;
    let pad_bottom = *pads.get(1).unwrap_or(&0) as usize;
    let pad_left = *pads.get(2).unwrap_or(&0) as usize;
    let pad_right = *pads.get(3).unwrap_or(&0) as usize;
    let stride_h = *strides.first().unwrap_or(&1) as usize;
    let stride_w = *strides.get(1).unwrap_or(&1) as usize;
    let dil_h = *dilations.first().unwrap_or(&1) as usize;
    let dil_w = *dilations.get(1).unwrap_or(&1) as usize;

    // Check pad_type for "same" padding.
    let is_same = op
        .inputs
        .get("pad_type")
        .is_some_and(|v| matches!(v, Value::String(s) if s == "same"));

    let (out_h, out_w) = if is_same {
        (in_h.div_ceil(stride_h), in_w.div_ceil(stride_w))
    } else {
        let eff_kh = dil_h * (k_h - 1) + 1;
        let eff_kw = dil_w * (k_w - 1) + 1;
        (
            (in_h + pad_top + pad_bottom).saturating_sub(eff_kh) / stride_h + 1,
            (in_w + pad_left + pad_right).saturating_sub(eff_kw) / stride_w + 1,
        )
    };

    Some(TensorType::new(
        in_tt.scalar_type,
        vec![in_tt.shape[0].unwrap_or(1), out_channels, out_h, out_w],
    ))
}

/// Infer pool output shape: channels preserved, spatial dims shrink.
pub(crate) fn infer_pool_output(
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

    if in_tt.shape.len() != 4 {
        return Some(in_tt.clone());
    }

    let in_c = in_tt.shape[1]?;
    let in_h = in_tt.shape[2]?;
    let in_w = in_tt.shape[3]?;

    let is_global = op
        .attributes
        .get("global_pool")
        .is_some_and(|v| matches!(v, Value::Bool(true)));

    let kernels = if is_global {
        vec![in_h as i64, in_w as i64]
    } else {
        read_int_list_from_op(op, "kernel_sizes").unwrap_or_else(|| vec![1, 1])
    };
    let strides = read_int_list_from_op(op, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = read_int_list_from_op(op, "pad").unwrap_or_else(|| vec![0, 0, 0, 0]);

    let k_h = *kernels.first().unwrap_or(&1) as usize;
    let k_w = *kernels.get(1).unwrap_or(&1) as usize;
    let stride_h = *strides.first().unwrap_or(&1) as usize;
    let stride_w = *strides.get(1).unwrap_or(&1) as usize;
    let pad_top = *pads.first().unwrap_or(&0) as usize;
    let pad_bottom = *pads.get(1).unwrap_or(&0) as usize;
    let pad_left = *pads.get(2).unwrap_or(&0) as usize;
    let pad_right = *pads.get(3).unwrap_or(&0) as usize;

    let out_h = (in_h + pad_top + pad_bottom).saturating_sub(k_h) / stride_h + 1;
    let out_w = (in_w + pad_left + pad_right).saturating_sub(k_w) / stride_w + 1;

    Some(TensorType::new(
        in_tt.scalar_type,
        vec![in_tt.shape[0].unwrap_or(1), in_c, out_h, out_w],
    ))
}

/// Infer concat output: sum along concat axis, other dims preserved.
pub(crate) fn infer_concat_output(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<TensorType> {
    let values = op.inputs.get("values")?;
    let Value::List(items) = values else {
        return None;
    };

    let types: Vec<&TensorType> = items
        .iter()
        .filter_map(|v| {
            if let Value::Reference(name) = v {
                type_map.get(name)
            } else {
                None
            }
        })
        .collect();

    let first = types.first()?;
    let axis = op
        .inputs
        .get("axis")
        .or_else(|| op.attributes.get("axis"))
        .and_then(|v| {
            if let Value::Int(a) = v {
                Some(*a as usize)
            } else {
                None
            }
        })
        .unwrap_or(0);

    let mut out_shape = first.shape.clone();
    if axis < out_shape.len() {
        let axis_sum: usize = types
            .iter()
            .map(|tt| tt.shape.get(axis).copied().flatten().unwrap_or(0))
            .sum();
        out_shape[axis] = Some(axis_sum);
    }

    Some(TensorType::with_dynamic_shape(first.scalar_type, out_shape))
}
