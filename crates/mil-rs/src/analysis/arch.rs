//! Model architecture detection from MIL IR programs.
//!
//! Walks a [`Program`]'s operations to extract transformer architecture
//! parameters (layer count, head counts, dimensions) without executing
//! the model.

use std::collections::{BTreeSet, HashMap};

use crate::ir::{Operation, Program, Value};

/// Detected transformer architecture parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArch {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (may differ from `num_heads` for GQA).
    pub num_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Hidden size (= num_heads × head_dim).
    pub hidden_size: usize,
    /// Vocabulary size (from the lm_head projection weight).
    pub vocab_size: usize,
}

/// Detect architecture parameters by walking the program's ops.
///
/// Looks for:
/// - Layer numbering in op names to count transformer layers.
/// - GQA-related ops to extract head counts and head_dim.
/// - Reshape ops with explicit head-count dimensions.
/// - The final linear/matmul projection weight shape for vocab_size.
///
/// Returns `None` if the program does not contain a recognizable
/// transformer architecture.
pub fn detect_model_arch(program: &Program) -> Option<ModelArch> {
    let func = program.main()?;
    let ops = &func.body.operations;
    if ops.is_empty() {
        return None;
    }

    let num_layers = detect_num_layers(ops)?;

    let (num_heads, num_kv_heads, head_dim) = detect_attention_params(ops)?;

    let hidden_size = num_heads * head_dim;

    let vocab_size = detect_vocab_size(ops).unwrap_or(0);

    Some(ModelArch {
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_size,
        vocab_size,
    })
}

/// Count transformer layers by finding unique layer numbers in op names.
fn detect_num_layers(ops: &[Operation]) -> Option<usize> {
    let mut layer_numbers = BTreeSet::new();

    for op in ops {
        if let Some(n) = extract_layer_number(&op.name) {
            layer_numbers.insert(n);
        }
    }

    if layer_numbers.is_empty() {
        return None;
    }

    // Layer count = max layer number + 1 (0-indexed).
    Some(layer_numbers.iter().last()? + 1)
}

/// Extract a layer number from an operation name.
///
/// Recognized patterns (anywhere in the name):
/// `layer_0_attn_q`, `layers.3.ffn.up`, `block_12_norm`,
/// `/model/layers.0/attn/...`, `model_layer_0_...`.
fn extract_layer_number(op_name: &str) -> Option<usize> {
    let lower = op_name.to_ascii_lowercase();
    // Search for layer/block patterns anywhere in the name (not just prefix),
    // since ONNX-converted ops often have a `model/layers.N/` prefix.
    for pattern in ["layers.", "layers_", "layer.", "layer_", "block.", "block_"] {
        if let Some(idx) = lower.find(pattern) {
            let rest = &lower[idx + pattern.len()..];
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !num_str.is_empty() {
                return num_str.parse().ok();
            }
        }
    }
    // Also match "layerN" without separator.
    if let Some(idx) = lower.find("layer") {
        let rest = &lower[idx + "layer".len()..];
        // Only if the next char is a digit (avoid matching "layers" → "s").
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !num_str.is_empty() {
            return num_str.parse().ok();
        }
    }
    None
}

/// Detect attention head counts and head_dim from ops.
///
/// Strategy:
/// 1. Look for reshape ops whose names contain "q_reshape" with explicit
///    head count in the shape (the GQA lowered pattern from ONNX conversion).
/// 2. Similarly look for "k_reshape" for kv_num_heads.
/// 3. Infer head_dim from Q/K/V projection weight shapes (inline or via const ops).
fn detect_attention_params(ops: &[Operation]) -> Option<(usize, usize, usize)> {
    let mut num_heads: Option<usize> = None;
    let mut num_kv_heads: Option<usize> = None;
    let mut head_dim: Option<usize> = None;

    // Build a shape map from output types for dimension inference.
    let mut shape_map: HashMap<String, Vec<Option<usize>>> = HashMap::new();
    for op in ops {
        for (out_name, out_ty) in op.outputs.iter().zip(op.output_types.iter()) {
            if let Some(ty) = out_ty {
                shape_map.insert(out_name.clone(), ty.shape.clone());
            }
        }
    }

    // Build a weight shape map from const ops (ONNX initializers become const ops).
    let mut const_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    for op in ops {
        if op.op_type == "const" {
            if let Some(Value::Tensor { shape, .. }) =
                op.inputs.get("val").or_else(|| op.attributes.get("val"))
            {
                for out_name in &op.outputs {
                    const_shapes.insert(out_name.clone(), shape.clone());
                }
            }
        }
    }

    for op in ops {
        let name_lower = op.name.to_ascii_lowercase();

        // Detect from reshape ops in the GQA lowered pattern.
        if op.op_type == "reshape" {
            if let Some(Value::List(dims)) = op.inputs.get("shape") {
                let int_dims: Vec<i64> = dims
                    .iter()
                    .filter_map(|v| match v {
                        Value::Int(i) => Some(*i),
                        _ => None,
                    })
                    .collect();

                // GQA reshape pattern: [0, 0, num_heads, -1]
                if int_dims.len() == 4 && int_dims[0] == 0 && int_dims[1] == 0 && int_dims[3] == -1
                {
                    let heads = int_dims[2] as usize;
                    if heads > 0 {
                        if name_lower.contains("_q_reshape") {
                            num_heads = Some(heads);
                        } else if name_lower.contains("_k_reshape")
                            || name_lower.contains("_kv_reshape")
                        {
                            num_kv_heads = Some(heads);
                        }
                    }
                }
            }
        }

        // Detect head_dim from Q projection weight shapes.
        if (op.op_type == "linear" || op.op_type == "matmul")
            && (name_lower.contains("attn_q")
                || name_lower.contains("q_proj")
                || name_lower.contains("attn/q"))
        {
            // Try inline tensor first.
            let weight_shape = op
                .inputs
                .get("weight")
                .or_else(|| op.inputs.get("y"))
                .or_else(|| op.attributes.get("weight"))
                .and_then(|v| match v {
                    Value::Tensor { shape, .. } => Some(shape.clone()),
                    // Follow reference to a const op.
                    Value::Reference(ref_name) => const_shapes.get(ref_name).cloned(),
                    _ => None,
                });

            if let Some(shape) = weight_shape {
                if shape.len() >= 2 {
                    // For linear: weight is [out_features, in_features] → shape[0].
                    // For matmul: y is [in_features, out_features] → shape[1].
                    // Respect transpose_y: if true, y is [out_features, in_features].
                    let out_features = if op.op_type == "matmul" {
                        let transpose_y = op
                            .attributes
                            .get("transpose_y")
                            .or_else(|| op.inputs.get("transpose_y"))
                            .and_then(|v| match v {
                                Value::Bool(b) => Some(*b),
                                _ => None,
                            })
                            .unwrap_or(false);
                        if transpose_y {
                            shape[0]
                        } else {
                            shape[shape.len() - 1]
                        }
                    } else {
                        shape[0]
                    };
                    if let Some(nh) = num_heads {
                        if nh > 0 && out_features > 0 && out_features % nh == 0 {
                            head_dim = Some(out_features / nh);
                        }
                    }
                }
            }
        }
    }

    // If we found num_heads from reshape but not kv_heads, default to MHA.
    let nh = num_heads?;
    let nkv = num_kv_heads.unwrap_or(nh);

    // If head_dim not found from weights, try to infer from output shapes
    // of the Q reshape op.
    if head_dim.is_none() {
        for op in ops {
            if op.op_type == "reshape" && op.name.to_ascii_lowercase().contains("_q_reshape") {
                if let Some(out_name) = op.outputs.first() {
                    if let Some(shape) = shape_map.get(out_name) {
                        // After reshape: [batch, seq, num_heads, head_dim]
                        if shape.len() == 4 {
                            if let Some(hd) = shape[3] {
                                head_dim = Some(hd);
                            }
                        }
                    }
                }
            }
        }
    }

    let hd = head_dim?;
    Some((nh, nkv, hd))
}

/// Detect vocab_size from the final linear/matmul projection.
///
/// Looks for the last `linear` or `matmul` op (typically `lm_head`) and
/// reads the output dimension from its weight shape (inline or const ref).
fn detect_vocab_size(ops: &[Operation]) -> Option<usize> {
    // Build const shape map for reference lookups.
    let mut const_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    for op in ops {
        if op.op_type == "const" {
            if let Some(Value::Tensor { shape, .. }) =
                op.inputs.get("val").or_else(|| op.attributes.get("val"))
            {
                for out_name in &op.outputs {
                    const_shapes.insert(out_name.clone(), shape.clone());
                }
            }
        }
    }

    // Walk ops in reverse to find the last linear/matmul.
    for op in ops.iter().rev() {
        if op.op_type == "linear" || op.op_type == "matmul" {
            let weight_shape = op
                .inputs
                .get("weight")
                .or_else(|| op.inputs.get("y"))
                .or_else(|| op.attributes.get("weight"))
                .and_then(|v| match v {
                    Value::Tensor { shape, .. } => Some(shape.clone()),
                    Value::Reference(ref_name) => const_shapes.get(ref_name).cloned(),
                    _ => None,
                });
            if let Some(shape) = weight_shape {
                if !shape.is_empty() {
                    return Some(shape[0]);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::TensorData;
    use crate::ir::{Block, Function, Operation, Program, Value};

    fn make_reshape_op(name: &str, shape_dims: &[i64]) -> Operation {
        Operation::new("reshape", name)
            .with_input("x", Value::Reference("input".into()))
            .with_input(
                "shape",
                Value::List(shape_dims.iter().map(|&d| Value::Int(d)).collect()),
            )
            .with_output(&format!("{name}_out"))
    }

    fn make_linear_op(name: &str, weight_shape: &[usize]) -> Operation {
        Operation::new("linear", name)
            .with_input("x", Value::Reference("input".into()))
            .with_input(
                "weight",
                Value::Tensor {
                    data: TensorData::Inline(vec![0u8; weight_shape.iter().product::<usize>() * 4]),
                    shape: weight_shape.to_vec(),
                    dtype: crate::ir::ScalarType::Float32,
                },
            )
            .with_output(&format!("{name}_out"))
    }

    fn make_program_from_ops(ops: Vec<Operation>) -> Program {
        let mut func = Function::new("main");
        func.body = Block {
            operations: ops,
            outputs: vec![],
        };
        let mut prog = Program::new("1.0");
        prog.add_function(func);
        prog
    }

    #[test]
    fn detect_layers_from_op_names() {
        let ops = vec![
            Operation::new("gather", "embed_tok"),
            Operation::new("linear", "layer_0_attn_q"),
            Operation::new("linear", "layer_0_ffn"),
            Operation::new("linear", "layer_1_attn_q"),
            Operation::new("linear", "layer_1_ffn"),
            Operation::new("layer_norm", "final_norm"),
        ];
        assert_eq!(detect_num_layers(&ops), Some(2));
    }

    #[test]
    fn detect_layers_28_layers() {
        let mut ops = vec![Operation::new("gather", "embed")];
        for i in 0..28 {
            ops.push(Operation::new("linear", &format!("layer_{i}_attn")));
        }
        assert_eq!(detect_num_layers(&ops), Some(28));
    }

    #[test]
    fn no_layers_returns_none() {
        let ops = vec![
            Operation::new("conv2d", "conv_a"),
            Operation::new("relu", "relu_a"),
        ];
        assert_eq!(detect_num_layers(&ops), None);
    }

    #[test]
    fn detect_gqa_heads_from_reshape() {
        // Simulates Qwen3-0.6B: 14 attention heads, 2 KV heads
        let ops = vec![
            make_reshape_op("gqa_0_q_reshape_op", &[0, 0, 14, -1]),
            make_reshape_op("gqa_0_k_reshape_op", &[0, 0, 2, -1]),
            make_linear_op("layer_0_attn_q", &[896, 896]),
        ];
        let (nh, nkv, hd) = detect_attention_params(&ops).unwrap();
        assert_eq!(nh, 14);
        assert_eq!(nkv, 2);
        assert_eq!(hd, 64); // 896 / 14
    }

    #[test]
    fn detect_mha_defaults_kv_heads() {
        // No K reshape → kv_heads defaults to num_heads
        let ops = vec![
            make_reshape_op("attn_q_reshape_op", &[0, 0, 8, -1]),
            make_linear_op("attn_q_proj", &[512, 512]),
        ];
        let (nh, nkv, hd) = detect_attention_params(&ops).unwrap();
        assert_eq!(nh, 8);
        assert_eq!(nkv, 8);
        assert_eq!(hd, 64);
    }

    #[test]
    fn detect_vocab_size_from_lm_head() {
        let ops = vec![
            Operation::new("linear", "layer_0_ffn"),
            make_linear_op("lm_head_proj", &[151936, 896]),
        ];
        assert_eq!(detect_vocab_size(&ops), Some(151936));
    }

    #[test]
    fn full_model_arch_detection() {
        let mut ops = Vec::new();
        // Embedding
        ops.push(Operation::new("gather", "embed_tok"));
        // GQA reshape ops (appear once in the converted model)
        ops.push(make_reshape_op("gqa_0_q_reshape_op", &[0, 0, 14, -1]));
        ops.push(make_reshape_op("gqa_0_k_reshape_op", &[0, 0, 2, -1]));
        // Q proj weight
        ops.push(make_linear_op("layer_0_attn_q", &[896, 896]));
        // 28 layers
        for i in 0..28 {
            ops.push(Operation::new("linear", &format!("layer_{i}_ffn")));
        }
        // LM head
        ops.push(make_linear_op("lm_head_proj", &[151936, 896]));

        let prog = make_program_from_ops(ops);
        let arch = detect_model_arch(&prog).unwrap();

        assert_eq!(arch.num_layers, 28);
        assert_eq!(arch.num_heads, 14);
        assert_eq!(arch.num_kv_heads, 2);
        assert_eq!(arch.head_dim, 64);
        assert_eq!(arch.hidden_size, 896);
        assert_eq!(arch.vocab_size, 151936);
    }

    #[test]
    fn non_transformer_returns_none() {
        let ops = vec![
            Operation::new("conv2d", "conv_a"),
            Operation::new("relu", "relu_a"),
        ];
        let prog = make_program_from_ops(ops);
        assert!(detect_model_arch(&prog).is_none());
    }

    #[test]
    fn detect_layers_onnx_naming() {
        // ONNX-converted models use names like /model/layers.0/attn/...
        let ops = vec![
            Operation::new("gather", "/model/embed_tokens/Gather"),
            Operation::new("matmul", "/model/layers.0/attn/q_proj/MatMul"),
            Operation::new("matmul", "/model/layers.0/mlp/gate_proj/MatMul"),
            Operation::new("matmul", "/model/layers.1/attn/q_proj/MatMul"),
            Operation::new("matmul", "/model/layers.27/mlp/down_proj/MatMul"),
        ];
        assert_eq!(detect_num_layers(&ops), Some(28));
    }

    #[test]
    fn detect_onnx_gqa_with_const_weights() {
        // Simulates ONNX-converted model: weight in const op, matmul references it.
        // ONNX matmul y input is [in_features, out_features] = [1024, 2048]
        // for Q projection: hidden_size=1024, num_heads=16, head_dim=128.
        let weight_const = Operation::new("const", "q_weight_const")
            .with_input(
                "val",
                Value::Tensor {
                    data: TensorData::Inline(vec![0u8; 1024 * 2048 * 4]),
                    shape: vec![1024, 2048],
                    dtype: crate::ir::ScalarType::Float32,
                },
            )
            .with_output("q_weight");

        let q_matmul = Operation::new("matmul", "/model/layers.0/attn/q_proj/MatMul")
            .with_input("x", Value::Reference("hidden".into()))
            .with_input("y", Value::Reference("q_weight".into()))
            .with_output("q_out");

        let ops = vec![
            weight_const,
            make_reshape_op(
                "/model/layers.0/attn/GroupQueryAttention_q_reshape_op",
                &[0, 0, 16, -1],
            ),
            make_reshape_op(
                "/model/layers.0/attn/GroupQueryAttention_k_reshape_op",
                &[0, 0, 8, -1],
            ),
            q_matmul,
        ];
        let (nh, nkv, hd) = detect_attention_params(&ops).unwrap();
        assert_eq!(nh, 16);
        assert_eq!(nkv, 8);
        assert_eq!(hd, 128); // 2048 / 16 = 128 (out_features is shape[1] for matmul)
    }
}
