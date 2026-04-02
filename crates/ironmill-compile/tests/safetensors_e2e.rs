//! End-to-end integration test: SafeTensors → WeightProvider → template → MIL IR Program.
//!
//! Creates a synthetic LLaMA model on disk using the `safetensors` crate,
//! loads it through `SafeTensorsProvider`, and converts to a MIL IR program.
//! No network access required — runs entirely with generated data.

use std::fs;

use safetensors::Dtype;
use safetensors::tensor::TensorView;

use ironmill_compile::templates::weights_to_program;
use ironmill_compile::weights::safetensors::SafeTensorsProvider;
use ironmill_compile::weights::{Architecture, WeightProvider};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const HIDDEN: usize = 32;
const INTERMEDIATE: usize = 64;
const NUM_LAYERS: usize = 2;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8; // HIDDEN / NUM_HEADS
const VOCAB: usize = 100;
const MAX_POS: usize = 64;

fn config_json() -> String {
    serde_json::json!({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POS,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "head_dim": HEAD_DIM,
        "tie_word_embeddings": false
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Zero-filled FP16 byte buffer for a tensor with the given shape.
fn zeros_f16(shape: &[usize]) -> Vec<u8> {
    let n: usize = shape.iter().product();
    vec![0u8; n * 2]
}

/// Build all weight tensors for a tiny LLaMA model, returning name → (data, shape) pairs.
fn build_llama_tensors() -> Vec<(String, Vec<u8>, Vec<usize>)> {
    let mut tensors = Vec::new();

    let mut add = |name: &str, shape: &[usize]| {
        tensors.push((name.to_string(), zeros_f16(shape), shape.to_vec()));
    };

    add("model.embed_tokens.weight", &[VOCAB, HIDDEN]);

    for i in 0..NUM_LAYERS {
        let p = format!("model.layers.{i}");
        add(
            &format!("{p}.self_attn.q_proj.weight"),
            &[NUM_HEADS * HEAD_DIM, HIDDEN],
        );
        add(
            &format!("{p}.self_attn.k_proj.weight"),
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN],
        );
        add(
            &format!("{p}.self_attn.v_proj.weight"),
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN],
        );
        add(
            &format!("{p}.self_attn.o_proj.weight"),
            &[HIDDEN, NUM_HEADS * HEAD_DIM],
        );
        add(
            &format!("{p}.mlp.gate_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
        );
        add(&format!("{p}.mlp.up_proj.weight"), &[INTERMEDIATE, HIDDEN]);
        add(
            &format!("{p}.mlp.down_proj.weight"),
            &[HIDDEN, INTERMEDIATE],
        );
        add(&format!("{p}.input_layernorm.weight"), &[HIDDEN]);
        add(&format!("{p}.post_attention_layernorm.weight"), &[HIDDEN]);
    }

    add("model.norm.weight", &[HIDDEN]);
    add("lm_head.weight", &[VOCAB, HIDDEN]);

    tensors
}

/// Write a synthetic safetensors model directory into `dir`.
fn write_model_dir(dir: &std::path::Path) {
    // config.json
    fs::write(dir.join("config.json"), config_json()).unwrap();

    // Build tensors and serialize to a single .safetensors file.
    let raw = build_llama_tensors();

    // safetensors::serialize expects an iterable of (name, View).
    // TensorView::new(dtype, shape, &data) → Result<TensorView>
    let views: Vec<(String, TensorView<'_>)> = raw
        .iter()
        .map(|(name, data, shape)| {
            let tv = TensorView::new(Dtype::F16, shape.clone(), data).unwrap();
            (name.clone(), tv)
        })
        .collect();

    let bytes = safetensors::serialize(views, None).unwrap();
    fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn safetensors_provider_loads_config_correctly() {
    let dir = tempfile::tempdir().unwrap();
    write_model_dir(dir.path());

    let provider = SafeTensorsProvider::load(dir.path()).expect("load should succeed");
    let config = provider.config();

    assert_eq!(config.architecture, Architecture::Llama);
    assert_eq!(config.hidden_size, HIDDEN);
    assert_eq!(config.intermediate_size, INTERMEDIATE);
    assert_eq!(config.num_hidden_layers, NUM_LAYERS);
    assert_eq!(config.num_attention_heads, NUM_HEADS);
    assert_eq!(config.num_key_value_heads, NUM_KV_HEADS);
    assert_eq!(config.vocab_size, VOCAB);
    assert_eq!(config.head_dim, HEAD_DIM);
    assert!(!config.tie_word_embeddings);
}

#[test]
fn safetensors_provider_exposes_all_tensors() {
    let dir = tempfile::tempdir().unwrap();
    write_model_dir(dir.path());

    let provider = SafeTensorsProvider::load(dir.path()).unwrap();
    let names = provider.tensor_names();

    // 2 global + 9 per layer × 2 layers + lm_head = 2 + 18 + 1 = 21
    // embed_tokens + model.norm + 9*2 + lm_head
    assert!(
        names.len() >= 21,
        "expected at least 21 tensors, got {}",
        names.len()
    );
    assert!(provider.has_tensor("model.embed_tokens.weight"));
    assert!(provider.has_tensor("model.layers.0.self_attn.q_proj.weight"));
    assert!(provider.has_tensor("model.layers.1.mlp.down_proj.weight"));
    assert!(provider.has_tensor("model.norm.weight"));
    assert!(provider.has_tensor("lm_head.weight"));
}

#[test]
fn safetensors_weights_to_program_produces_valid_ir() {
    let dir = tempfile::tempdir().unwrap();
    write_model_dir(dir.path());

    let provider = SafeTensorsProvider::load(dir.path()).unwrap();

    let result = weights_to_program(&provider).expect("weights_to_program should succeed");

    // Program must have a "main" function.
    let main = result
        .program
        .main()
        .expect("program should have a main function");

    // Main function should have operations.
    assert!(
        !main.body.operations.is_empty(),
        "main function body should have operations"
    );

    // Main function should have at least one output.
    assert!(
        !main.body.outputs.is_empty(),
        "main function should have outputs"
    );

    // Collect op types for verification.
    let op_types: Vec<&str> = main
        .body
        .operations
        .iter()
        .map(|op| op.op_type.as_str())
        .collect();

    // Should contain const ops (weight tensors).
    assert!(
        op_types.contains(&"const"),
        "program should contain const ops for weights, got: {op_types:?}"
    );

    // Should contain linear ops (projections).
    assert!(
        op_types.contains(&"linear"),
        "program should contain linear ops, got: {op_types:?}"
    );

    // Should contain rms_norm ops.
    assert!(
        op_types.contains(&"rms_norm"),
        "program should contain rms_norm ops, got: {op_types:?}"
    );

    // Should contain add ops (residual connections).
    assert!(
        op_types.contains(&"add"),
        "program should contain add ops, got: {op_types:?}"
    );
}

#[test]
fn safetensors_program_is_marked_autoregressive() {
    let dir = tempfile::tempdir().unwrap();
    write_model_dir(dir.path());

    let provider = SafeTensorsProvider::load(dir.path()).unwrap();
    let result = weights_to_program(&provider).unwrap();

    assert!(
        result.program.is_autoregressive(),
        "LLaMA program should be marked autoregressive"
    );
}
