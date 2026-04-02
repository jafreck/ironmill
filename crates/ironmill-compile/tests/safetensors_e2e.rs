//! End-to-end integration test: SafeTensors → WeightProvider → template → MIL IR Program.
//!
//! Creates a synthetic LLaMA model on disk using the `safetensors` crate,
//! loads it through `SafeTensorsProvider`, and converts to a MIL IR program.
//! No network access required — runs entirely with generated data.

mod common;

use common::*;
use ironmill_compile::templates::weights_to_program;
use ironmill_compile::weights::safetensors::SafeTensorsProvider;
use ironmill_compile::weights::{Architecture, WeightProvider};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_model_dir(dir: &std::path::Path) {
    let config = config_json("llama");
    let tensors = build_llama_base_tensors();
    write_safetensors_model_dir(dir, &config, &tensors);
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

    assert_valid_llm_ir(&result.program);
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
