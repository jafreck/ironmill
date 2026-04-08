//! Deterministic test model generator for quantization integration tests.
//!
//! Generates a minimal 2-layer LLaMA-like MIL [`Program`] with realistic
//! structure and shapes, along with calibration token data.
//!
//! # Architecture
//!
//! - `vocab_size`: 256, `hidden_dim`: 64, `intermediate_dim`: 128,
//!   `n_heads`: 2, `n_layers`: 2
//! - Each layer: RMSNorm → Q/K/V proj → O proj → residual →
//!   RMSNorm → gate/up proj → SiLU → mul → down proj → residual
//! - Final: RMSNorm → LM head
//!
//! All weights are deterministic (seeded RNG) so tests are reproducible.

use mil_rs::TensorData;
use mil_rs::ir::{Block, Function, Operation, Program, ScalarType, TensorType, Value};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

// ── Model hyper-parameters ──────────────────────────────────────────────

const VOCAB_SIZE: usize = 256;
const HIDDEN_DIM: usize = 64;
const INTERMEDIATE_DIM: usize = 128;
const N_LAYERS: usize = 2;
const SEQ_LEN: usize = 128;
const RNG_SEED: u64 = 0xDEAD_BEEF_CAFE;
const RMS_NORM_EPS: f64 = 1e-5;

// ── Public API ──────────────────────────────────────────────────────────

/// Create a minimal MIL Program that resembles a 2-layer LLaMA for testing
/// quantization passes.
///
/// The program contains:
/// - 21 `const` ops with FP32 tensor data (embedding + per-layer weights +
///   final norm + LM head)
/// - 15 `linear` ops consuming those weights
/// - 2D weight shapes suitable for per-group quantization (e.g. `[64, 64]`
///   with `group_size=32`)
/// - Deterministic weights generated from a seeded RNG
pub fn create_test_program() -> Program {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let mut block = Block::new();

    // ── Embedding ───────────────────────────────────────────────────
    let embed_weight = random_weight(&mut rng, &[VOCAB_SIZE, HIDDEN_DIM]);
    block.add_op(const_tensor_op(
        "embed_tokens_const",
        "embed_tokens_weight",
        &embed_weight,
        vec![VOCAB_SIZE, HIDDEN_DIM],
    ));
    block.add_op(
        Operation::new("gather", "embed_gather")
            .with_input("x", Value::Reference("embed_tokens_weight".into()))
            .with_input("indices", Value::Reference("input_ids".into()))
            .with_attr("axis", Value::Int(0))
            .with_output("embed_out"),
    );

    let mut prev_output = "embed_out".to_string();

    // ── Transformer layers ──────────────────────────────────────────
    for layer in 0..N_LAYERS {
        let p = format!("l{layer}");

        // Attention RMSNorm
        let attn_norm_out = emit_rms_norm(
            &mut block,
            &mut rng,
            &format!("{p}_attn_norm"),
            &prev_output,
            HIDDEN_DIM,
        );

        // Q / K / V / O projections
        let q_out = emit_linear(
            &mut block,
            &mut rng,
            &format!("{p}_q_proj"),
            &attn_norm_out,
            HIDDEN_DIM,
            HIDDEN_DIM,
        );
        let k_out = emit_linear(
            &mut block,
            &mut rng,
            &format!("{p}_k_proj"),
            &attn_norm_out,
            HIDDEN_DIM,
            HIDDEN_DIM,
        );
        let v_out = emit_linear(
            &mut block,
            &mut rng,
            &format!("{p}_v_proj"),
            &attn_norm_out,
            HIDDEN_DIM,
            HIDDEN_DIM,
        );

        // Simplified attention: matmul(q, k^T) then matmul(scores, v)
        let scores_out = format!("{p}_attn_scores");
        block.add_op(
            Operation::new("matmul", &format!("{p}_qk_matmul"))
                .with_input("x", Value::Reference(q_out))
                .with_input("y", Value::Reference(k_out))
                .with_attr("transpose_y", Value::Bool(true))
                .with_output(&scores_out),
        );
        let attn_out = format!("{p}_attn_out");
        block.add_op(
            Operation::new("matmul", &format!("{p}_sv_matmul"))
                .with_input("x", Value::Reference(scores_out))
                .with_input("y", Value::Reference(v_out))
                .with_output(&attn_out),
        );

        // O projection
        let o_out = emit_linear(
            &mut block,
            &mut rng,
            &format!("{p}_o_proj"),
            &attn_out,
            HIDDEN_DIM,
            HIDDEN_DIM,
        );

        // Residual add
        let attn_residual = format!("{p}_attn_residual");
        block.add_op(
            Operation::new("add", &format!("{p}_attn_add"))
                .with_input("x", Value::Reference(prev_output.clone()))
                .with_input("y", Value::Reference(o_out))
                .with_output(&attn_residual),
        );

        // FFN RMSNorm
        let ffn_norm_out = emit_rms_norm(
            &mut block,
            &mut rng,
            &format!("{p}_ffn_norm"),
            &attn_residual,
            HIDDEN_DIM,
        );

        // Gate projection → SiLU
        let gate_out = emit_linear(
            &mut block,
            &mut rng,
            &format!("{p}_gate_proj"),
            &ffn_norm_out,
            INTERMEDIATE_DIM,
            HIDDEN_DIM,
        );
        let gate_act = format!("{p}_gate_act");
        block.add_op(
            Operation::new("silu", &format!("{p}_silu"))
                .with_input("x", Value::Reference(gate_out))
                .with_output(&gate_act),
        );

        // Up projection
        let up_out = emit_linear(
            &mut block,
            &mut rng,
            &format!("{p}_up_proj"),
            &ffn_norm_out,
            INTERMEDIATE_DIM,
            HIDDEN_DIM,
        );

        // Element-wise mul (gate_act * up)
        let mlp_mul_out = format!("{p}_mlp_mul");
        block.add_op(
            Operation::new("mul", &format!("{p}_mlp_mul_op"))
                .with_input("x", Value::Reference(gate_act))
                .with_input("y", Value::Reference(up_out))
                .with_output(&mlp_mul_out),
        );

        // Down projection
        let down_out = emit_linear(
            &mut block,
            &mut rng,
            &format!("{p}_down_proj"),
            &mlp_mul_out,
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
        );

        // Residual add
        let ffn_residual = format!("{p}_ffn_residual");
        block.add_op(
            Operation::new("add", &format!("{p}_ffn_add"))
                .with_input("x", Value::Reference(attn_residual))
                .with_input("y", Value::Reference(down_out))
                .with_output(&ffn_residual),
        );

        prev_output = ffn_residual;
    }

    // ── Final norm + LM head ────────────────────────────────────────
    let final_norm_out =
        emit_rms_norm(&mut block, &mut rng, "final_norm", &prev_output, HIDDEN_DIM);
    let lm_head_out = emit_linear(
        &mut block,
        &mut rng,
        "lm_head",
        &final_norm_out,
        VOCAB_SIZE,
        HIDDEN_DIM,
    );

    block.outputs.push(lm_head_out);

    // Assemble function & program
    let input_ty = TensorType::new(ScalarType::Int32, vec![1, SEQ_LEN]);
    let func = Function::new("main").with_input("input_ids", input_ty);

    let mut program = Program::new("1.0.0");
    program.add_function(func);
    program.functions.get_mut("main").unwrap().body = block;

    program
}

/// Create a minimal calibration-compatible token dataset.
///
/// Returns `n_sequences` sequences of `seq_len` tokens, all in range
/// `[0, vocab_size)`. Uses a deterministic seeded RNG so results are
/// reproducible.
pub fn create_test_calibration_data(
    n_sequences: usize,
    seq_len: usize,
    seed: u64,
) -> Vec<Vec<u32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_sequences)
        .map(|_| {
            (0..seq_len)
                .map(|_| rng.random_range(0..VOCAB_SIZE as u32))
                .collect()
        })
        .collect()
}

/// Returns the vocabulary size used by [`create_test_program`].
pub fn vocab_size() -> usize {
    VOCAB_SIZE
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Build a `const` op carrying an FP32 tensor in `inputs["val"]`.
fn const_tensor_op(name: &str, output: &str, data: &[f32], shape: Vec<usize>) -> Operation {
    Operation::new("const", name)
        .with_input(
            "val",
            Value::Tensor {
                data: TensorData::Inline(f32_slice_to_bytes(data)),
                shape,
                dtype: ScalarType::Float32,
            },
        )
        .with_output(output)
}

/// Emit a const weight + linear op pair, returning the output name.
fn emit_linear(
    block: &mut Block,
    rng: &mut StdRng,
    prefix: &str,
    input: &str,
    out_features: usize,
    in_features: usize,
) -> String {
    let weight_name = format!("{prefix}_weight");
    let weight_const = format!("{prefix}_const");
    let out_name = format!("{prefix}_out");

    let weight_data = random_weight(rng, &[out_features, in_features]);
    block.add_op(const_tensor_op(
        &weight_const,
        &weight_name,
        &weight_data,
        vec![out_features, in_features],
    ));
    block.add_op(
        Operation::new("linear", &format!("{prefix}_op"))
            .with_input("x", Value::Reference(input.into()))
            .with_input("weight", Value::Reference(weight_name))
            .with_output(&out_name),
    );

    out_name
}

/// Emit a const norm-weight + rms_norm op pair, returning the output name.
fn emit_rms_norm(
    block: &mut Block,
    rng: &mut StdRng,
    prefix: &str,
    input: &str,
    dim: usize,
) -> String {
    let weight_name = format!("{prefix}_weight");
    let weight_const = format!("{prefix}_const");
    let out_name = format!("{prefix}_out");

    // Norm weights are typically initialized near 1.0
    let weight_data: Vec<f32> = (0..dim)
        .map(|_| 1.0 + rng.random_range(-0.01f32..0.01))
        .collect();
    block.add_op(const_tensor_op(
        &weight_const,
        &weight_name,
        &weight_data,
        vec![dim],
    ));
    block.add_op(
        Operation::new("rms_norm", &format!("{prefix}_op"))
            .with_input("x", Value::Reference(input.into()))
            .with_input("weight", Value::Reference(weight_name))
            .with_attr("epsilon", Value::Float(RMS_NORM_EPS))
            .with_output(&out_name),
    );

    out_name
}

/// Generate a weight tensor with values drawn uniformly from `[-0.1, 0.1]`.
///
/// This mimics the small-magnitude weights typical of Kaiming/Xavier
/// initialisation at small hidden sizes.
fn random_weight(rng: &mut StdRng, shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n).map(|_| rng.random_range(-0.1f32..0.1)).collect()
}

/// Convert an f32 slice to little-endian bytes.
fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}
