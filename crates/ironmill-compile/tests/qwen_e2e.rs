//! End-to-end integration test: Qwen via SafeTensors → WeightProvider → template → MIL IR Program.
//!
//! Creates a synthetic Qwen2 model with attention bias tensors, loads it
//! through `SafeTensorsProvider`, and converts to a MIL IR program.
//! Verifies that bias handling in the Qwen template works end-to-end.

mod common;

use common::*;
use ironmill_compile::templates::weights_to_program;
use ironmill_compile::weights::safetensors::SafeTensorsProvider;
use ironmill_compile::weights::{Architecture, WeightProvider};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build weight tensors for a Qwen model: LLaMA base + attention biases.
fn build_qwen_tensors() -> Vec<(String, Vec<u8>, Vec<usize>)> {
    let mut tensors = build_llama_base_tensors();

    for i in 0..NUM_LAYERS {
        let p = format!("model.layers.{i}");
        tensors.push((
            format!("{p}.self_attn.q_proj.bias"),
            zeros_f16(&[NUM_HEADS * HEAD_DIM]),
            vec![NUM_HEADS * HEAD_DIM],
        ));
        tensors.push((
            format!("{p}.self_attn.k_proj.bias"),
            zeros_f16(&[NUM_KV_HEADS * HEAD_DIM]),
            vec![NUM_KV_HEADS * HEAD_DIM],
        ));
        tensors.push((
            format!("{p}.self_attn.v_proj.bias"),
            zeros_f16(&[NUM_KV_HEADS * HEAD_DIM]),
            vec![NUM_KV_HEADS * HEAD_DIM],
        ));
    }

    tensors
}

fn write_model_dir(dir: &std::path::Path) {
    let config = config_json("qwen2");
    let tensors = build_qwen_tensors();
    write_safetensors_model_dir(dir, &config, &tensors);
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

    assert_valid_llm_ir(&result.program);
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
