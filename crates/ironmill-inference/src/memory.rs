//! Memory estimation utilities.
//!
//! [`MemoryEstimator`] provides static methods to estimate GPU memory
//! requirements from a [`ModelConfig`] before loading. [`MemoryUsage`]
//! reports actual memory from a loaded engine via
//! [`InferenceEngine::memory_usage()`](crate::engine::InferenceEngine::memory_usage).

use mil_rs::weights::ModelConfig;

/// Memory information for a loaded engine.
///
/// Returned by [`InferenceEngine::memory_usage()`](crate::engine::InferenceEngine::memory_usage).
/// All sizes in bytes. Uses `u64` instead of `usize` to correctly
/// represent sizes > 4 GB on 32-bit targets.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total GPU memory consumed by model weights.
    pub weight_memory: u64,
    /// Current KV cache memory usage.
    pub kv_cache_memory: u64,
    /// Peak KV cache memory (maximum across all sequences).
    pub kv_cache_peak: u64,
    /// Estimated memory for temporary compute buffers (activations, etc.).
    pub scratch_memory: u64,
}

/// Memory estimation utilities (no engine required).
///
/// These are static methods that estimate memory requirements from config
/// alone, before loading. Useful for deciding whether a model will fit.
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// Estimate total weight memory for a model config + quantization level.
    pub fn weight_memory(config: &ModelConfig, quant: QuantLevel) -> u64 {
        let bytes_per_param: f64 = match quant {
            QuantLevel::Fp16 => 2.0,
            QuantLevel::Int8 => 1.0,
            QuantLevel::Int4 => 0.5,
            QuantLevel::Int2 => 0.25,
        };
        let params = Self::estimate_params(config);
        (params as f64 * bytes_per_param) as u64
    }

    /// Estimate KV cache memory for a given sequence length and batch size.
    pub fn kv_cache_memory(
        config: &ModelConfig,
        seq_len: usize,
        batch_size: usize,
        kv_quant: Option<KvQuantLevel>,
    ) -> u64 {
        let bytes_per_element: f64 = match kv_quant {
            None | Some(KvQuantLevel::None) => 2.0,        // FP16
            Some(KvQuantLevel::Int8) => 1.0,               // 8-bit
            Some(KvQuantLevel::TurboInt4) => 0.5,          // 4-bit
            Some(KvQuantLevel::TurboInt8) => 1.0,          // 8-bit turbo
        };
        // 2 (K+V) * layers * bytes_per_element * kv_heads * head_dim * seq_len * batch_size
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let layers = config.num_hidden_layers;
        (2.0 * layers as f64 * bytes_per_element * kv_heads as f64 * head_dim as f64
            * seq_len as f64 * batch_size as f64) as u64
    }

    /// Estimate maximum sequence length that fits in the given memory budget.
    pub fn max_seq_len_for_budget(
        config: &ModelConfig,
        budget_bytes: u64,
        batch_size: usize,
    ) -> usize {
        let weight_mem = Self::weight_memory(config, QuantLevel::Fp16);
        if budget_bytes <= weight_mem {
            return 0;
        }
        let remaining = budget_bytes - weight_mem;
        let per_token = Self::kv_cache_memory(config, 1, batch_size, None);
        if per_token == 0 {
            return usize::MAX;
        }
        (remaining / per_token) as usize
    }

    fn estimate_params(config: &ModelConfig) -> u64 {
        let embed = (config.vocab_size * config.hidden_size) as u64;
        let attn = (config.num_hidden_layers * 4 * config.hidden_size * config.hidden_size) as u64;
        let ffn =
            (config.num_hidden_layers * 3 * config.hidden_size * config.intermediate_size) as u64;
        embed + attn + ffn
    }
}

/// Quantization level for memory estimation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum QuantLevel {
    /// 16-bit floating point weights.
    Fp16,
    /// 8-bit integer quantized weights.
    Int8,
    /// 4-bit integer quantized weights.
    Int4,
    /// 2-bit integer quantized weights.
    Int2,
}

/// KV cache quantization level.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum KvQuantLevel {
    /// Full-precision KV cache (no quantization).
    None,
    /// 8-bit integer KV cache quantization.
    Int8,
    /// Turbo mode with 4-bit integer KV cache.
    TurboInt4,
    /// Turbo mode with 8-bit integer KV cache.
    TurboInt8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::weights::Architecture;

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
    fn weight_memory_scales_with_quant() {
        let config = test_config();
        let fp16 = MemoryEstimator::weight_memory(&config, QuantLevel::Fp16);
        let int8 = MemoryEstimator::weight_memory(&config, QuantLevel::Int8);
        let int4 = MemoryEstimator::weight_memory(&config, QuantLevel::Int4);
        assert_eq!(fp16, int8 * 2);
        assert_eq!(int8, int4 * 2);
        assert!(fp16 > 0);
    }

    #[test]
    fn kv_cache_memory_scales_with_seq_len() {
        let config = test_config();
        let mem1 = MemoryEstimator::kv_cache_memory(&config, 1, 1, None);
        let mem100 = MemoryEstimator::kv_cache_memory(&config, 100, 1, None);
        assert_eq!(mem100, mem1 * 100);
    }

    #[test]
    fn max_seq_len_zero_when_no_budget() {
        let config = test_config();
        assert_eq!(MemoryEstimator::max_seq_len_for_budget(&config, 0, 1), 0);
    }

    #[test]
    fn max_seq_len_reasonable() {
        let config = test_config();
        let budget = MemoryEstimator::weight_memory(&config, QuantLevel::Fp16)
            + MemoryEstimator::kv_cache_memory(&config, 1000, 1, None);
        let max_len = MemoryEstimator::max_seq_len_for_budget(&config, budget, 1);
        assert_eq!(max_len, 1000);
    }
}
