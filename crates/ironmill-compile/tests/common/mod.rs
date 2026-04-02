//! Shared test helpers for integration tests.
//!
//! Provides builder functions for creating realistic test programs
//! and utility functions for validation and serialization.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mil_rs::ir::passes::tensor_utils::f32_slice_to_bytes;
use mil_rs::{
    Block, Function, Operation, PassPipeline, PipelineReport, Program, ScalarType, TensorType,
    Value, model_to_program, program_to_model,
};

/// CoreML spec version used for test conversions.
pub const SPEC_VERSION: i32 = 8;

/// Path to test fixture files.
pub fn fixture_path(name: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest)
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Build a minimal transformer-like program with N layers.
///
/// Each layer contains:
///   layer_norm → linear (Q) → linear (K) → linear (V) →
///   matmul (Q×K^T) → softmax → matmul (attn×V) → linear (proj) → add (residual)
///
/// Op names follow "layers.{i}.{op_type}" for name-based detection.
/// Input: single tensor "input" with shape [1, seq_len, hidden_dim].
pub fn build_transformer_program(n_layers: usize) -> Program {
    let seq_len = 32;
    let hidden_dim = 64;
    let input_ty = TensorType::new(ScalarType::Float32, vec![1, seq_len, hidden_dim]);

    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    let mut prev_output = "input".to_string();

    for i in 0..n_layers {
        let prefix = format!("layers.{i}");

        // layer_norm
        let ln_out = format!("{prefix}.layer_norm_out");
        block.add_op(
            Operation::new("layer_norm", &format!("{prefix}.layer_norm"))
                .with_input("x", Value::Reference(prev_output.clone()))
                .with_output(&ln_out),
        );

        // linear Q
        let q_weight = format!("{prefix}.q_weight");
        let q_out = format!("{prefix}.q_out");
        block.add_op(make_const_op(
            &q_weight,
            &[hidden_dim, hidden_dim],
            ScalarType::Float32,
        ));
        block.add_op(
            Operation::new("linear", &format!("{prefix}.attention.q_proj"))
                .with_input("x", Value::Reference(ln_out.clone()))
                .with_input("weight", Value::Reference(q_weight))
                .with_output(&q_out),
        );

        // linear K
        let k_weight = format!("{prefix}.k_weight");
        let k_out = format!("{prefix}.k_out");
        block.add_op(make_const_op(
            &k_weight,
            &[hidden_dim, hidden_dim],
            ScalarType::Float32,
        ));
        block.add_op(
            Operation::new("linear", &format!("{prefix}.attention.k_proj"))
                .with_input("x", Value::Reference(ln_out.clone()))
                .with_input("weight", Value::Reference(k_weight))
                .with_output(&k_out),
        );

        // linear V
        let v_weight = format!("{prefix}.v_weight");
        let v_out = format!("{prefix}.v_out");
        block.add_op(make_const_op(
            &v_weight,
            &[hidden_dim, hidden_dim],
            ScalarType::Float32,
        ));
        block.add_op(
            Operation::new("linear", &format!("{prefix}.attention.v_proj"))
                .with_input("x", Value::Reference(ln_out))
                .with_input("weight", Value::Reference(v_weight))
                .with_output(&v_out),
        );

        // matmul Q×K^T
        let qk_out = format!("{prefix}.qk_out");
        block.add_op(
            Operation::new("matmul", &format!("{prefix}.attention.qk_matmul"))
                .with_input("x", Value::Reference(q_out))
                .with_input("y", Value::Reference(k_out))
                .with_output(&qk_out),
        );

        // softmax
        let attn_out = format!("{prefix}.attn_weights");
        block.add_op(
            Operation::new("softmax", &format!("{prefix}.attention.softmax"))
                .with_input("x", Value::Reference(qk_out))
                .with_attr("axis", Value::Int(-1))
                .with_output(&attn_out),
        );

        // matmul attn×V
        let av_out = format!("{prefix}.av_out");
        block.add_op(
            Operation::new("matmul", &format!("{prefix}.attention.av_matmul"))
                .with_input("x", Value::Reference(attn_out))
                .with_input("y", Value::Reference(v_out))
                .with_output(&av_out),
        );

        // linear proj
        let proj_weight = format!("{prefix}.proj_weight");
        let proj_out = format!("{prefix}.proj_out");
        block.add_op(make_const_op(
            &proj_weight,
            &[hidden_dim, hidden_dim],
            ScalarType::Float32,
        ));
        block.add_op(
            Operation::new("linear", &format!("{prefix}.attention.out_proj"))
                .with_input("x", Value::Reference(av_out))
                .with_input("weight", Value::Reference(proj_weight))
                .with_output(&proj_out),
        );

        // residual add
        let add_out = format!("{prefix}.residual_out");
        block.add_op(
            Operation::new("add", &format!("{prefix}.residual_add"))
                .with_input("x", Value::Reference(proj_out))
                .with_input("y", Value::Reference(prev_output))
                .with_output(&add_out),
        );

        prev_output = add_out;
    }

    block.outputs.push(prev_output);
    program
}

/// Build a program with N conv ops (for NHWC/fusion testing).
///
/// Each conv has shape [1, H, W, C_in] → [1, H, W, C_out] with 3×3 kernel.
/// Optionally followed by batch_norm and relu ops for fusion testing.
pub fn build_conv_program(n_convs: usize) -> Program {
    let c_in = 3;
    let c_out = 16;
    let h = 8;
    let w = 8;
    let input_ty = TensorType::new(ScalarType::Float32, vec![1, c_in, h, w]);

    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    let mut prev_output = "input".to_string();

    for i in 0..n_convs {
        let cur_c_in = if i == 0 { c_in } else { c_out };

        // weight const
        let weight_name = format!("conv_{i}_weight");
        block.add_op(make_const_op(
            &weight_name,
            &[c_out, cur_c_in, 3, 3],
            ScalarType::Float32,
        ));

        // conv op
        let conv_out = format!("conv_{i}_out");
        block.add_op(
            Operation::new("conv", &format!("conv_{i}"))
                .with_input("x", Value::Reference(prev_output.clone()))
                .with_input("weight", Value::Reference(weight_name))
                .with_output(&conv_out),
        );

        // batch_norm op
        let bn_out = format!("bn_{i}_out");
        let bn_mean = format!("bn_{i}_mean");
        let bn_var = format!("bn_{i}_var");
        let bn_gamma = format!("bn_{i}_gamma");
        let bn_beta = format!("bn_{i}_beta");
        block.add_op(make_const_op(&bn_mean, &[c_out], ScalarType::Float32));
        block.add_op(make_const_op(&bn_var, &[c_out], ScalarType::Float32));
        block.add_op(make_const_op(&bn_gamma, &[c_out], ScalarType::Float32));
        block.add_op(make_const_op(&bn_beta, &[c_out], ScalarType::Float32));
        block.add_op(
            Operation::new("batch_norm", &format!("bn_{i}"))
                .with_input("x", Value::Reference(conv_out))
                .with_input("mean", Value::Reference(bn_mean))
                .with_input("variance", Value::Reference(bn_var))
                .with_input("gamma", Value::Reference(bn_gamma))
                .with_input("beta", Value::Reference(bn_beta))
                .with_output(&bn_out),
        );

        // relu op
        let relu_out = format!("relu_{i}_out");
        block.add_op(
            Operation::new("relu", &format!("relu_{i}"))
                .with_input("x", Value::Reference(bn_out))
                .with_output(&relu_out),
        );

        prev_output = relu_out;
    }

    block.outputs.push(prev_output);
    program
}

/// Build a program simulating MoE with N experts.
///
/// Structure: input → router linear → softmax → top-k select,
/// then N parallel expert branches (each: linear → gelu → linear),
/// then weighted combine → output.
/// Op names follow "expert_{i}.linear" pattern for name-based detection.
pub fn build_moe_program(n_experts: usize) -> Program {
    let hidden_dim = 64;
    let input_ty = TensorType::new(ScalarType::Float32, vec![1, 32, hidden_dim]);

    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;

    // Router: linear → softmax
    let router_weight_name = "gate.weight";
    block.add_op(make_const_op(
        router_weight_name,
        &[n_experts, hidden_dim],
        ScalarType::Float32,
    ));
    block.add_op(
        Operation::new("linear", "gate.linear")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference(router_weight_name.into()))
            .with_output("router_logits"),
    );
    block.add_op(
        Operation::new("softmax", "gate.softmax")
            .with_input("x", Value::Reference("router_logits".into()))
            .with_attr("axis", Value::Int(-1))
            .with_output("router_probs"),
    );

    // Expert branches
    let mut expert_outputs = Vec::new();
    for i in 0..n_experts {
        let prefix = format!("expert_{i}");

        // FFN: linear → gelu → linear
        let w1_name = format!("{prefix}.fc1.weight");
        block.add_op(make_const_op(
            &w1_name,
            &[hidden_dim * 4, hidden_dim],
            ScalarType::Float32,
        ));
        let fc1_out = format!("{prefix}.fc1_out");
        block.add_op(
            Operation::new("linear", &format!("{prefix}.linear"))
                .with_input("x", Value::Reference("input".into()))
                .with_input("weight", Value::Reference(w1_name))
                .with_output(&fc1_out),
        );

        let gelu_out = format!("{prefix}.gelu_out");
        block.add_op(
            Operation::new("gelu", &format!("{prefix}.gelu"))
                .with_input("x", Value::Reference(fc1_out))
                .with_output(&gelu_out),
        );

        let w2_name = format!("{prefix}.fc2.weight");
        block.add_op(make_const_op(
            &w2_name,
            &[hidden_dim, hidden_dim * 4],
            ScalarType::Float32,
        ));
        let fc2_out = format!("{prefix}.fc2_out");
        block.add_op(
            Operation::new("linear", &format!("{prefix}.fc2.linear"))
                .with_input("x", Value::Reference(gelu_out))
                .with_input("weight", Value::Reference(w2_name))
                .with_output(&fc2_out),
        );

        expert_outputs.push(fc2_out);
    }

    // Combine expert outputs with adds
    let mut combined = expert_outputs[0].clone();
    for (i, expert_out) in expert_outputs.iter().enumerate().skip(1) {
        let add_out = format!("combine_{i}");
        block.add_op(
            Operation::new("add", &format!("combine_add_{i}"))
                .with_input("x", Value::Reference(combined))
                .with_input("y", Value::Reference(expert_out.clone()))
                .with_output(&add_out),
        );
        combined = add_out;
    }

    block.outputs.push(combined);
    program
}

/// Build a program with KV cache inputs (autoregressive).
///
/// Inputs: "input_ids" [1, seq_len], "past_key_values.{0..n}.key" and
/// "past_key_values.{0..n}.value" with shape [1, heads, cache_len, head_dim].
/// Contains attention ops that consume the cache tensors.
pub fn build_autoregressive_program(max_seq_len: usize) -> Program {
    let heads = 4;
    let head_dim = 16;
    let hidden_dim = heads * head_dim;

    let input_ty = TensorType::with_dynamic_shape(
        ScalarType::Float32,
        vec![Some(1), None], // dynamic seq_len
    );

    let mut func = Function::new("main").with_input("input_ids", input_ty);

    // Add KV cache inputs with dynamic dims
    let n_cache_layers = 2;
    for layer in 0..n_cache_layers {
        let key_ty = TensorType::with_dynamic_shape(
            ScalarType::Float32,
            vec![Some(1), Some(heads), None, Some(head_dim)],
        );
        let value_ty = TensorType::with_dynamic_shape(
            ScalarType::Float32,
            vec![Some(1), Some(heads), None, Some(head_dim)],
        );
        func.inputs
            .push((format!("past_key_values.{layer}.key"), key_ty));
        func.inputs
            .push((format!("past_key_values.{layer}.value"), value_ty));
    }

    let mut program = Program::new("1");
    program.add_function(func);
    program.set_attribute("autoregressive", "true");

    let block = &mut program.functions.get_mut("main").unwrap().body;

    // Embedding (simplified)
    let embed_weight = "embedding_weight";
    block.add_op(make_const_op(
        embed_weight,
        &[1000, hidden_dim],
        ScalarType::Float32,
    ));
    block.add_op(
        Operation::new("gather", "embedding")
            .with_input("x", Value::Reference(embed_weight.into()))
            .with_input("indices", Value::Reference("input_ids".into()))
            .with_output("embed_out"),
    );

    let mut prev_output = "embed_out".to_string();
    for layer in 0..n_cache_layers {
        let prefix = format!("layers.{layer}");
        let key_input = format!("past_key_values.{layer}.key");
        let value_input = format!("past_key_values.{layer}.value");

        // Simplified attention using cache
        let attn_out = format!("{prefix}.attn_out");
        block.add_op(
            Operation::new("matmul", &format!("{prefix}.attention.qk"))
                .with_input("x", Value::Reference(prev_output.clone()))
                .with_input("y", Value::Reference(key_input))
                .with_output(&format!("{prefix}.qk_out")),
        );
        block.add_op(
            Operation::new("softmax", &format!("{prefix}.attention.softmax"))
                .with_input("x", Value::Reference(format!("{prefix}.qk_out")))
                .with_output(&format!("{prefix}.attn_weights")),
        );
        block.add_op(
            Operation::new("matmul", &format!("{prefix}.attention.av"))
                .with_input("x", Value::Reference(format!("{prefix}.attn_weights")))
                .with_input("y", Value::Reference(value_input))
                .with_output(&attn_out),
        );

        prev_output = attn_out;
    }

    block.outputs.push(prev_output);
    program
}

/// Build a program with RVQ codebook pattern (N codebooks).
///
/// Each codebook: const [K, D] → gather(indices) → chain of adds
/// to accumulate residuals. Final output is sum of all codebook lookups.
pub fn build_rvq_program(n_codebooks: usize) -> Program {
    let k = 256; // codebook size
    let d = 64; // embedding dim
    let input_ty = TensorType::new(ScalarType::Int32, vec![1, 32]);

    let func = Function::new("main").with_input("indices", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;

    let mut accumulated: Option<String> = None;

    for i in 0..n_codebooks {
        // Codebook const
        let cb_name = format!("codebook_{i}");
        block.add_op(make_const_op(&cb_name, &[k, d], ScalarType::Float32));

        // Gather
        let gather_out = format!("gather_{i}_out");
        block.add_op(
            Operation::new("gather", &format!("gather_{i}"))
                .with_input("x", Value::Reference(cb_name))
                .with_input("indices", Value::Reference("indices".into()))
                .with_output(&gather_out),
        );

        // Accumulate with add
        if let Some(prev) = accumulated {
            let add_out = format!("rvq_add_{i}");
            block.add_op(
                Operation::new("add", &format!("rvq_add_{i}"))
                    .with_input("x", Value::Reference(prev))
                    .with_input("y", Value::Reference(gather_out))
                    .with_output(&add_out),
            );
            accumulated = Some(add_out);
        } else {
            accumulated = Some(gather_out);
        }
    }

    block.outputs.push(accumulated.unwrap());
    program
}

/// Build a program with a single large const tensor of the given shape
/// and dtype, useful for quantization accuracy and size tests.
pub fn build_const_program(shape: &[usize], dtype: ScalarType) -> Program {
    let total_elements: usize = shape.iter().product();
    let data = match dtype {
        ScalarType::Float32 => {
            let values: Vec<f32> = (0..total_elements)
                .map(|i| ((i as f32 * 0.1) - (total_elements as f32 * 0.05)).sin())
                .collect();
            f32_slice_to_bytes(&values)
        }
        ScalarType::Float16 => {
            let values: Vec<f32> = (0..total_elements)
                .map(|i| ((i as f32 * 0.1) - (total_elements as f32 * 0.05)).sin())
                .collect();
            let half_values: Vec<u8> = values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect();
            half_values
        }
        _ => vec![0u8; total_elements * 4],
    };

    let input_ty = TensorType::new(ScalarType::Float32, vec![1]);
    let func = Function::new("main").with_input("dummy_input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(
        Operation::new("const", "weight")
            .with_attr(
                "val",
                Value::Tensor {
                    data,
                    shape: shape.to_vec(),
                    dtype,
                },
            )
            .with_output("weight_out"),
    );

    // Add an identity op to use the const as output
    block.add_op(
        Operation::new("identity", "output_identity")
            .with_input("x", Value::Reference("weight_out".into()))
            .with_output("output"),
    );

    block.outputs.push("output".into());
    program
}

/// Run default pipeline and return the report.
pub fn run_default_pipeline(program: &mut Program) -> PipelineReport {
    let pipeline = PassPipeline::new();
    pipeline
        .run(program)
        .expect("default pipeline should succeed")
}

/// Serialize program to Model proto and back, verify round-trip.
pub fn assert_serialization_roundtrip(program: &Program) {
    let model = program_to_model(program, SPEC_VERSION).expect("program_to_model should succeed");
    let roundtripped =
        model_to_program(&model).expect("model_to_program should succeed on roundtrip");

    // Verify function count matches
    assert_eq!(
        program.functions.len(),
        roundtripped.functions.len(),
        "round-trip should preserve function count"
    );

    // Verify op count matches for each function
    for (name, func) in &program.functions {
        let rt_func = roundtripped
            .functions
            .get(name)
            .unwrap_or_else(|| panic!("round-trip should preserve function '{name}'"));
        assert_eq!(
            func.body.operations.len(),
            rt_func.body.operations.len(),
            "round-trip should preserve op count in function '{name}'"
        );
    }
}

/// Serialize program to Model proto and return the serialized byte size.
pub fn serialized_size(program: &Program) -> usize {
    use prost::Message;
    let model = program_to_model(program, SPEC_VERSION).expect("program_to_model should succeed");
    model.encode_to_vec().len()
}

/// Run CLI command and return (success, stdout, stderr).
pub fn run_cli(args: &[&str]) -> (bool, String, String) {
    let output = std::process::Command::new("cargo")
        .args(["run", "-p", "ironmill-cli", "--quiet", "--"])
        .args(args)
        .output()
        .expect("failed to run CLI");

    let success = output.status.success();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (success, stdout, stderr)
}

/// Count total operations in a program.
pub fn count_ops(program: &Program) -> usize {
    program
        .functions
        .values()
        .map(|f| f.body.operations.len())
        .sum()
}

// ---- Internal helpers ----

/// Create a const op with randomly-seeded data.
pub fn make_const_op(name: &str, shape: &[usize], dtype: ScalarType) -> Operation {
    let total_elements: usize = shape.iter().product();
    let data = match dtype {
        ScalarType::Float32 => {
            let values: Vec<f32> = (0..total_elements)
                .map(|i| {
                    // Deterministic pseudo-random: use name hash + index for variety
                    let seed = name.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
                    let x = (seed as f32 + i as f32) * 0.01;
                    x.sin() * 0.5
                })
                .collect();
            f32_slice_to_bytes(&values)
        }
        ScalarType::Float16 => {
            let values: Vec<u8> = (0..total_elements)
                .flat_map(|i| {
                    let seed = name.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
                    let x = (seed as f32 + i as f32) * 0.01;
                    half::f16::from_f32(x.sin() * 0.5).to_le_bytes()
                })
                .collect();
            values
        }
        ScalarType::Int32 => {
            let values: Vec<u8> = (0..total_elements)
                .flat_map(|i| (i as i32).to_le_bytes())
                .collect();
            values
        }
        _ => vec![0u8; total_elements * std::mem::size_of::<f32>()],
    };

    Operation::new("const", name)
        .with_attr(
            "val",
            Value::Tensor {
                data,
                shape: shape.to_vec(),
                dtype,
            },
        )
        .with_output(&format!("{name}_out"))
}

/// Build a const tensor with specific FP32 values.
pub fn make_const_op_with_values(name: &str, shape: &[usize], values: &[f32]) -> Operation {
    let data = f32_slice_to_bytes(values);
    Operation::new("const", name)
        .with_attr(
            "val",
            Value::Tensor {
                data,
                shape: shape.to_vec(),
                dtype: ScalarType::Float32,
            },
        )
        .with_output(&format!("{name}_out"))
}
