//! Runtime model information.
//!
//! [`ModelInfo`] provides a uniform view of a loaded model's properties
//! across all backends, populated from [`ModelConfig`] during engine loading.

use mil_rs::weights::{Architecture, ModelConfig};

/// Runtime information about a loaded model.
///
/// Populated during engine loading from [`ModelConfig`] and weight metadata.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model architecture (Llama, Qwen, Gemma, etc.).
    pub architecture: Architecture,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum context length supported by the model.
    pub max_context_len: usize,
    /// Quantization applied to weights.
    pub weight_quantization: String,
    /// EOS token IDs for this model.
    ///
    /// Sourced from the model's `generation_config.json` or `tokenizer_config.json`.
    pub eos_tokens: Vec<u32>,
    /// Number of parameters (approximate, in millions).
    pub param_count_m: f32,
    /// Whether the model uses grouped-query attention (num_kv_heads < num_heads).
    pub uses_gqa: bool,
    /// Whether the model uses multi-latent attention (DeepSeek-style).
    pub uses_mla: bool,
    /// Head dimension.
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (may differ from attention heads in GQA).
    pub num_kv_heads: usize,
}

impl ModelInfo {
    /// Create from a [`ModelConfig`], populating all fields.
    pub fn from_config(config: &ModelConfig) -> Self {
        let uses_gqa = config.num_key_value_heads < config.num_attention_heads;
        let uses_mla =
            config.extra.contains_key("mla_config") || config.extra.contains_key("q_lora_rank");
        let param_count_m = estimate_param_count(config);
        Self {
            architecture: config.architecture,
            num_layers: config.num_hidden_layers,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            max_context_len: config.max_position_embeddings,
            weight_quantization: String::from("fp16"),
            eos_tokens: Vec::new(),
            param_count_m,
            uses_gqa,
            uses_mla,
            head_dim: config.head_dim,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
        }
    }
}

fn estimate_param_count(config: &ModelConfig) -> f32 {
    let embed = config.vocab_size * config.hidden_size;
    let attn = config.num_hidden_layers * 4 * config.hidden_size * config.hidden_size;
    let ffn = config.num_hidden_layers * 3 * config.hidden_size * config.intermediate_size;
    (embed + attn + ffn) as f32 / 1e6
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ModelConfig {
        ModelConfig::new(Architecture::Llama)
            .with_hidden_size(4096)
            .with_intermediate_size(11008)
            .with_num_hidden_layers(32)
            .with_num_attention_heads(32)
            .with_num_key_value_heads(8)
            .with_head_dim(128)
            .with_vocab_size(32000)
            .with_max_position_embeddings(4096)
    }

    #[test]
    fn model_info_from_config_gqa() {
        let config = test_config();
        let info = ModelInfo::from_config(&config);
        assert!(info.uses_gqa);
        assert!(!info.uses_mla);
        assert_eq!(info.num_layers, 32);
        assert_eq!(info.hidden_size, 4096);
        assert_eq!(info.vocab_size, 32000);
        assert_eq!(info.num_kv_heads, 8);
        assert_eq!(info.num_attention_heads, 32);
        assert!(info.param_count_m > 0.0);
    }

    #[test]
    fn model_info_from_config_mla() {
        let mut config = test_config();
        config.extra.insert(
            "q_lora_rank".to_string(),
            serde_json::Value::Number(serde_json::Number::from(1536)),
        );
        let info = ModelInfo::from_config(&config);
        assert!(info.uses_mla);
    }

    #[test]
    fn model_info_from_config_no_gqa() {
        let mut config = test_config();
        config.num_key_value_heads = config.num_attention_heads;
        let info = ModelInfo::from_config(&config);
        assert!(!info.uses_gqa);
    }
}
