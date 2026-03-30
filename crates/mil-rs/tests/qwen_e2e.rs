//! End-to-end integration test: Qwen via SafeTensors → WeightProvider → template → MIL IR Program.
//!
//! Creates a synthetic Qwen2 model with attention bias tensors, loads it
//! through `SafeTensorsProvider`, and converts to a MIL IR program.
//! Verifies that bias handling in the Qwen template works end-to-end.

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
const HEAD_DIM: usize = 8;
const VOCAB: usize = 100;
const MAX_POS: usize = 64;

fn config_json() -> String {
    serde_json::json!({
        "model_type": "qwen2",
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

fn zeros_f16(shape: &[usize]) -> Vec<u8> {
    let n: usize = shape.iter().product();
    vec![0u8; n * 2]
}

/// Build weight tensors for a Qwen model including attention biases.
fn build_qwen_tensors() -> Vec<(String, Vec<u8>, Vec<usize>)> {
    let mut tensors = Vec::new();

    let mut add = |name: &str, shape: &[usize]| {
        tensors.push((name.to_string(), zeros_f16(shape), shape.to_vec()));
    };

    add("model.embed_tokens.weight", &[VOCAB, HIDDEN]);

    for i in 0..NUM_LAYERS {
        let p = format!("model.layers.{i}");

        // Weights (same as LLaMA)
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

        // Biases (Qwen-specific on Q/K/V)
        add(
            &format!("{p}.self_attn.q_proj.bias"),
            &[NUM_HEADS * HEAD_DIM],
        );
        add(
            &format!("{p}.self_attn.k_proj.bias"),
            &[NUM_KV_HEADS * HEAD_DIM],
        );
        add(
            &format!("{p}.self_attn.v_proj.bias"),
            &[NUM_KV_HEADS * HEAD_DIM],
        );

        // MLP
        add(
            &format!("{p}.mlp.gate_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
        );
        add(&format!("{p}.mlp.up_proj.weight"), &[INTERMEDIATE, HIDDEN]);
        add(
            &format!("{p}.mlp.down_proj.weight"),
            &[HIDDEN, INTERMEDIATE],
        );

        // Layer norms
        add(&format!("{p}.input_layernorm.weight"), &[HIDDEN]);
        add(&format!("{p}.post_attention_layernorm.weight"), &[HIDDEN]);
    }

    add("model.norm.weight", &[HIDDEN]);
    add("lm_head.weight", &[VOCAB, HIDDEN]);

    tensors
}

fn write_model_dir(dir: &std::path::Path) {
    fs::write(dir.join("config.json"), config_json()).unwrap();

    let raw = build_qwen_tensors();
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
fn qwen_provider_identifies_architecture() {
    let dir = tempfile::tempdir().unwrap();
    write_model_dir(dir.path());

    let provider = SafeTensorsProvider::load(dir.path()).expect("load should succeed");
    assert_eq!(provider.config().architecture, Architecture::Qwen);
}

#[test]
fn qwen_provider_has_bias_tensors() {
    let dir = tempfile::tempdir().unwrap();
    write_model_dir(dir.path());

    let provider = SafeTensorsProvider::load(dir.path()).unwrap();

    assert!(provider.has_tensor("model.layers.0.self_attn.q_proj.bias"));
    assert!(provider.has_tensor("model.layers.0.self_attn.k_proj.bias"));
    assert!(provider.has_tensor("model.layers.0.self_attn.v_proj.bias"));
    assert!(provider.has_tensor("model.layers.1.self_attn.q_proj.bias"));
}

#[test]
fn qwen_weights_to_program_produces_valid_ir() {
    let dir = tempfile::tempdir().unwrap();
    write_model_dir(dir.path());

    let provider = SafeTensorsProvider::load(dir.path()).unwrap();

    let result = weights_to_program(&provider).expect("weights_to_program should succeed");

    let main = result
        .program
        .main()
        .expect("program should have a main function");

    assert!(
        !main.body.operations.is_empty(),
        "main function body should have operations"
    );
    assert!(
        !main.body.outputs.is_empty(),
        "main function should have outputs"
    );

    let op_types: Vec<&str> = main
        .body
        .operations
        .iter()
        .map(|op| op.op_type.as_str())
        .collect();

    assert!(op_types.contains(&"const"));
    assert!(op_types.contains(&"linear"));
    assert!(op_types.contains(&"rms_norm"));
    assert!(op_types.contains(&"add"));
}

#[test]
fn qwen_program_has_bias_on_q_projection() {
    let dir = tempfile::tempdir().unwrap();
    write_model_dir(dir.path());

    let provider = SafeTensorsProvider::load(dir.path()).unwrap();
    let result = weights_to_program(&provider).unwrap();
    let main = result.program.main().unwrap();

    // Find Q projection linear op for layer 0.
    let q_proj = main
        .body
        .operations
        .iter()
        .find(|op| op.op_type == "linear" && op.name.contains("q_proj"));

    assert!(
        q_proj.is_some(),
        "should have a q_proj linear op, ops: {:?}",
        main.body
            .operations
            .iter()
            .map(|o| &o.name)
            .collect::<Vec<_>>()
    );
    let q_op = q_proj.unwrap();
    assert!(
        q_op.inputs.contains_key("bias"),
        "Qwen Q projection should have bias input, inputs: {:?}",
        q_op.inputs.keys().collect::<Vec<_>>()
    );
}

#[test]
fn qwen_program_is_marked_autoregressive() {
    let dir = tempfile::tempdir().unwrap();
    write_model_dir(dir.path());

    let provider = SafeTensorsProvider::load(dir.path()).unwrap();
    let result = weights_to_program(&provider).unwrap();

    assert!(
        result.program.is_autoregressive(),
        "Qwen program should be marked autoregressive"
    );
}
