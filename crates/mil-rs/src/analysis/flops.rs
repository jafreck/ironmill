//! FLOPs (floating-point operations) counter for MIL IR programs.
//!
//! Walks the MIL IR and counts multiply-accumulate operations per op type
//! to compute the total FLOPs for a forward pass.

use std::collections::HashMap;

use crate::analysis::build_type_map;
use crate::ir::{Function, Operation, Program, TensorType, Value};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Estimate total FLOPs for every function in a [`Program`].
pub fn estimate_program_flops(program: &Program) -> u64 {
    program
        .functions
        .values()
        .map(estimate_function_flops)
        .sum()
}

/// Estimate total FLOPs for a single [`Function`].
pub fn estimate_function_flops(func: &Function) -> u64 {
    let type_map = build_type_map(func);
    func.body
        .operations
        .iter()
        .map(|op| estimate_op_flops(op, &type_map))
        .sum()
}

/// Per-operation FLOPs breakdown for a [`Function`].
///
/// Returns a vec of `(op_name, op_type, flops)` tuples, one per operation.
pub fn per_op_flops(func: &Function) -> Vec<(String, String, u64)> {
    let type_map = build_type_map(func);
    func.body
        .operations
        .iter()
        .map(|op| {
            let flops = estimate_op_flops(op, &type_map);
            (op.name.clone(), op.op_type.clone(), flops)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Per-op dispatch
// ---------------------------------------------------------------------------

fn estimate_op_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    match op.op_type.as_str() {
        "conv" => estimate_conv_flops(op, type_map),
        "matmul" => estimate_matmul_flops(op, type_map),
        "batch_matmul" => estimate_batch_matmul_flops(op, type_map),
        "linear" => estimate_linear_flops(op, type_map),
        "relu" | "sigmoid" | "tanh" | "clip" | "sqrt" | "pow" | "cast" => {
            estimate_elementwise_flops(op, type_map)
        }
        "softmax" => estimate_elementwise_flops(op, type_map).saturating_mul(5),
        "batch_norm" | "layer_norm" => estimate_elementwise_flops(op, type_map).saturating_mul(4),
        "add" | "mul" | "sub" | "real_div" | "select" => estimate_elementwise_flops(op, type_map),
        "reduce_mean" => estimate_elementwise_flops(op, type_map),
        "max_pool" | "avg_pool" => estimate_pool_flops(op, type_map),
        "scaled_dot_product_attention" => estimate_attention_flops(op, type_map),
        // Memory/shape ops: no compute.
        "const" | "reshape" | "transpose" | "concat" | "squeeze" | "expand_dims" | "pad"
        | "split" | "slice_by_index" | "gather" => 0,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Op-specific estimators
// ---------------------------------------------------------------------------

/// Conv FLOPs = 2 × H_out × W_out × K_h × K_w × C_in_per_group × C_out.
fn estimate_conv_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    let weight_shape = resolve_input_shape(op, "weight", type_map);
    let (c_out, c_in_per_group, kh, kw) = match weight_shape {
        Some(ref s) if s.len() >= 4 => match (s[0], s[1], s[2], s[3]) {
            (Some(co), Some(ci), Some(kh), Some(kw)) => (co, ci, kh, kw),
            _ => return 0,
        },
        _ => return 0,
    };

    let (h_out, w_out) = output_spatial_dims(op, type_map).unwrap_or((0, 0));
    if h_out == 0 || w_out == 0 {
        return 0;
    }

    2u64.saturating_mul(h_out as u64)
        .saturating_mul(w_out as u64)
        .saturating_mul(kh as u64)
        .saturating_mul(kw as u64)
        .saturating_mul(c_in_per_group as u64)
        .saturating_mul(c_out as u64)
}

/// MatMul FLOPs = 2 × M × N × K.
fn estimate_matmul_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    let x_shape = resolve_input_shape(op, "x", type_map);
    let y_shape = resolve_input_shape(op, "y", type_map);

    let k = x_shape.as_ref().and_then(|s| s.last().copied()).flatten();
    let m = x_shape
        .as_ref()
        .and_then(|s| if s.len() >= 2 { s[s.len() - 2] } else { None });
    let n = y_shape.as_ref().and_then(|s| s.last().copied()).flatten();

    match (m, k, n) {
        (Some(m), Some(k), Some(n)) => 2u64
            .saturating_mul(m as u64)
            .saturating_mul(k as u64)
            .saturating_mul(n as u64),
        _ => 0,
    }
}

/// Batch MatMul FLOPs = 2 × B × M × N × K.
fn estimate_batch_matmul_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    let x_shape = resolve_input_shape(op, "x", type_map);
    let y_shape = resolve_input_shape(op, "y", type_map);

    let (batch, m, k) = match x_shape {
        Some(ref s) if s.len() >= 3 => {
            let b: u64 = s[..s.len() - 2]
                .iter()
                .filter_map(|d| d.map(|v| v as u64))
                .product();
            let m = s[s.len() - 2];
            let k = s[s.len() - 1];
            (b, m, k)
        }
        _ => return 0,
    };

    let n = y_shape.as_ref().and_then(|s| s.last().copied()).flatten();

    match (m, k, n) {
        (Some(m), Some(k), Some(n)) => 2u64
            .saturating_mul(batch)
            .saturating_mul(m as u64)
            .saturating_mul(n as u64)
            .saturating_mul(k as u64),
        _ => 0,
    }
}

/// Linear FLOPs = 2 × batch × in_features × out_features.
fn estimate_linear_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    let weight_shape = resolve_input_shape(op, "weight", type_map);
    let input_shape = resolve_input_shape(op, "x", type_map);

    let (out_features, in_features) = match weight_shape {
        Some(ref s) if s.len() >= 2 => match (s[0], s[1]) {
            (Some(o), Some(i)) => (o, i),
            _ => return 0,
        },
        _ => return 0,
    };

    let batch: usize = input_shape
        .as_ref()
        .map(|s| {
            s.iter()
                .rev()
                .skip(1)
                .filter_map(|d| *d)
                .product::<usize>()
                .max(1)
        })
        .unwrap_or(1);

    2u64.saturating_mul(batch as u64)
        .saturating_mul(in_features as u64)
        .saturating_mul(out_features as u64)
}

/// Element-wise FLOPs = number of output elements.
fn estimate_elementwise_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    if let Some(Some(ty)) = op.output_types.first() {
        return ty.shape.iter().filter_map(|d| *d).product::<usize>() as u64;
    }
    if let Some(shape) = resolve_input_shape(op, "x", type_map) {
        return shape.iter().filter_map(|d| *d).product::<usize>() as u64;
    }
    0
}

/// Pool FLOPs = output_elements × kernel_size.
fn estimate_pool_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    let output_elements = estimate_elementwise_flops(op, type_map);
    let kernel_size = get_attr_int_list(op, "kernel_sizes")
        .map(|ks| ks.iter().product::<i64>().unsigned_abs())
        .unwrap_or(1);
    output_elements.saturating_mul(kernel_size)
}

/// SDPA FLOPs = 4 × batch_heads × seq_len² × head_dim.
fn estimate_attention_flops(op: &Operation, type_map: &HashMap<String, TensorType>) -> u64 {
    let q_shape = resolve_input_shape(op, "query", type_map);
    match q_shape {
        Some(ref s) if s.len() >= 3 => {
            let dims: Vec<usize> = s.iter().filter_map(|d| *d).collect();
            if dims.len() >= 3 {
                let batch_heads: u64 = dims[..dims.len() - 2].iter().map(|&d| d as u64).product();
                let seq_len = dims[dims.len() - 2] as u64;
                let head_dim = dims[dims.len() - 1] as u64;
                4u64.saturating_mul(batch_heads)
                    .saturating_mul(seq_len)
                    .saturating_mul(seq_len)
                    .saturating_mul(head_dim)
            } else {
                0
            }
        }
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the shape of a named input to an operation via the type map.
fn resolve_input_shape(
    op: &Operation,
    input_name: &str,
    type_map: &HashMap<String, TensorType>,
) -> Option<Vec<Option<usize>>> {
    match op.inputs.get(input_name)? {
        Value::Reference(name) => type_map.get(name).map(|t| t.shape.clone()),
        Value::Tensor { shape, .. } => Some(shape.iter().map(|&d| Some(d)).collect()),
        _ => None,
    }
}

/// Extract an integer list from an operation attribute.
fn get_attr_int_list(op: &Operation, attr_name: &str) -> Option<Vec<i64>> {
    match op.attributes.get(attr_name)? {
        Value::List(items) => items
            .iter()
            .map(|v| match v {
                Value::Int(n) => Some(*n),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

/// Try to extract spatial output dimensions (H, W) for conv/pool ops.
fn output_spatial_dims(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<(usize, usize)> {
    if let Some(Some(ty)) = op.output_types.first() {
        if ty.shape.len() >= 4 {
            if let (Some(h), Some(w)) = (ty.shape[2], ty.shape[3]) {
                return Some((h, w));
            }
        }
    }
    // Fall back to input spatial dims (approximation ignoring stride/padding).
    let input_shape = resolve_input_shape(op, "x", type_map)?;
    if input_shape.len() >= 4 {
        if let (Some(h), Some(w)) = (input_shape[2], input_shape[3]) {
            return Some((h, w));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::TensorData;
    use crate::ir::{Block, Function, Operation, Program, ScalarType, TensorType};

    /// Build a minimal program with the given operations in a single function.
    fn make_program(ops: Vec<Operation>) -> Program {
        let mut func = Function::new("main");
        func.inputs.push((
            "input".into(),
            TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]),
        ));
        func.body = Block {
            operations: ops,
            outputs: vec!["out".into()],
        };
        let mut program = Program::new("1.0.0");
        program.add_function(func);
        program
    }

    #[test]
    fn test_conv_flops() {
        // Conv: 3×3 kernel, 64 input channels, 128 output channels, 56×56 output
        let mut conv = Operation::new("conv", "conv_0");
        conv.inputs.insert(
            "weight".into(),
            Value::Tensor {
                data: TensorData::Inline(vec![0u8; 4 * 128 * 64 * 3 * 3]),
                shape: vec![128, 64, 3, 3],
                dtype: ScalarType::Float32,
            },
        );
        conv.output_types = vec![Some(TensorType::new(
            ScalarType::Float32,
            vec![1, 128, 56, 56],
        ))];
        conv.outputs = vec!["conv_out".into()];

        let program = make_program(vec![conv]);
        // 2 * 56 * 56 * 3 * 3 * 64 * 128 = 924_844_032
        let expected: u64 = 2 * 56 * 56 * 3 * 3 * 64 * 128;
        assert_eq!(estimate_program_flops(&program), expected);
    }

    #[test]
    fn test_matmul_flops() {
        // matmul: x=[4, 128, 64], y=[4, 64, 256] → M=128, K=64, N=256
        let mut mm = Operation::new("matmul", "mm_0");
        mm.inputs.insert(
            "x".into(),
            Value::Tensor {
                data: TensorData::Inline(vec![]),
                shape: vec![4, 128, 64],
                dtype: ScalarType::Float32,
            },
        );
        mm.inputs.insert(
            "y".into(),
            Value::Tensor {
                data: TensorData::Inline(vec![]),
                shape: vec![4, 64, 256],
                dtype: ScalarType::Float32,
            },
        );
        mm.outputs = vec!["mm_out".into()];

        let program = make_program(vec![mm]);
        // 2 * 128 * 64 * 256 = 4_194_304
        let expected: u64 = 2 * 128 * 64 * 256;
        assert_eq!(estimate_program_flops(&program), expected);
    }

    #[test]
    fn test_linear_flops() {
        // linear: weight=[512, 256], input batch=8 → 2*8*256*512
        let mut linear = Operation::new("linear", "linear_0");
        linear.inputs.insert(
            "weight".into(),
            Value::Tensor {
                data: TensorData::Inline(vec![]),
                shape: vec![512, 256],
                dtype: ScalarType::Float32,
            },
        );
        linear.inputs.insert(
            "x".into(),
            Value::Tensor {
                data: TensorData::Inline(vec![]),
                shape: vec![8, 256],
                dtype: ScalarType::Float32,
            },
        );
        linear.outputs = vec!["linear_out".into()];

        let program = make_program(vec![linear]);
        let expected: u64 = 2 * 8 * 256 * 512;
        assert_eq!(estimate_program_flops(&program), expected);
    }

    #[test]
    fn test_elementwise_flops() {
        // relu with output type [1, 64, 32, 32] → 1*64*32*32 = 65536
        let mut relu = Operation::new("relu", "relu_0");
        relu.output_types = vec![Some(TensorType::new(
            ScalarType::Float32,
            vec![1, 64, 32, 32],
        ))];
        relu.outputs = vec!["relu_out".into()];

        let mut sigmoid = Operation::new("sigmoid", "sigmoid_0");
        sigmoid.output_types = vec![Some(TensorType::new(
            ScalarType::Float32,
            vec![1, 64, 32, 32],
        ))];
        sigmoid.outputs = vec!["sigmoid_out".into()];

        let program = make_program(vec![relu, sigmoid]);
        let elem_count: u64 = 1 * 64 * 32 * 32;
        assert_eq!(estimate_program_flops(&program), elem_count * 2);
    }

    #[test]
    fn test_zero_flops_for_shape_ops() {
        let ops = vec![
            Operation::new("const", "c0").with_output("c_out"),
            Operation::new("reshape", "r0").with_output("r_out"),
            Operation::new("transpose", "t0").with_output("t_out"),
        ];
        let program = make_program(ops);
        assert_eq!(estimate_program_flops(&program), 0);
    }

    #[test]
    fn test_attention_flops() {
        // SDPA: query shape [2, 8, 128, 64] → batch_heads=16, seq=128, head_dim=64
        let mut attn = Operation::new("scaled_dot_product_attention", "sdpa_0");
        attn.inputs.insert(
            "query".into(),
            Value::Tensor {
                data: TensorData::Inline(vec![]),
                shape: vec![2, 8, 128, 64],
                dtype: ScalarType::Float32,
            },
        );
        attn.outputs = vec!["attn_out".into()];

        let program = make_program(vec![attn]);
        // 4 * (2*8) * 128 * 128 * 64 = 4 * 16 * 16384 * 64 = 67_108_864
        let expected: u64 = 4 * 16 * 128 * 128 * 64;
        assert_eq!(estimate_program_flops(&program), expected);
    }

    #[test]
    fn test_program_total_flops() {
        // relu(65536) + conv(924_844_032) = 924_909_568
        let mut relu = Operation::new("relu", "relu_0");
        relu.output_types = vec![Some(TensorType::new(
            ScalarType::Float32,
            vec![1, 64, 32, 32],
        ))];
        relu.outputs = vec!["relu_out".into()];

        let mut conv = Operation::new("conv", "conv_0");
        conv.inputs.insert(
            "weight".into(),
            Value::Tensor {
                data: TensorData::Inline(vec![0u8; 4 * 128 * 64 * 3 * 3]),
                shape: vec![128, 64, 3, 3],
                dtype: ScalarType::Float32,
            },
        );
        conv.output_types = vec![Some(TensorType::new(
            ScalarType::Float32,
            vec![1, 128, 56, 56],
        ))];
        conv.outputs = vec!["conv_out".into()];

        let program = make_program(vec![relu, conv]);
        let relu_flops: u64 = 1 * 64 * 32 * 32;
        let conv_flops: u64 = 2 * 56 * 56 * 3 * 3 * 64 * 128;
        assert_eq!(estimate_program_flops(&program), relu_flops + conv_flops);
    }

    #[test]
    fn test_per_op_flops() {
        let mut relu = Operation::new("relu", "relu_0");
        relu.output_types = vec![Some(TensorType::new(
            ScalarType::Float32,
            vec![1, 64, 32, 32],
        ))];
        relu.outputs = vec!["relu_out".into()];

        let reshape = Operation::new("reshape", "reshape_0").with_output("reshape_out");

        let func = {
            let mut f = Function::new("main");
            f.inputs.push((
                "input".into(),
                TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]),
            ));
            f.body = Block {
                operations: vec![relu, reshape],
                outputs: vec!["reshape_out".into()],
            };
            f
        };

        let breakdown = per_op_flops(&func);
        assert_eq!(breakdown.len(), 2);

        assert_eq!(breakdown[0].0, "relu_0");
        assert_eq!(breakdown[0].1, "relu");
        assert_eq!(breakdown[0].2, 1 * 64 * 32 * 32);

        assert_eq!(breakdown[1].0, "reshape_0");
        assert_eq!(breakdown[1].1, "reshape");
        assert_eq!(breakdown[1].2, 0);
    }
}
