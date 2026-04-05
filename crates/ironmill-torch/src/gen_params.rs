//! Generation parameters for text generation.

use ironmill_core::model_info::ModelInfo;
use ironmill_inference::generate::GenerateRequest;
use ironmill_inference::sampling::SamplerConfig;

/// Controls sampling behaviour during text generation.
///
/// This is the user-facing knob set. Internally it maps to
/// [`SamplerConfig`] + [`GenerateRequest`] from `ironmill-inference`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GenParams {
    /// Sampling temperature. 0.0 = greedy (argmax).
    pub temperature: f32,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Top-p (nucleus) filtering threshold. 1.0 = disabled.
    pub top_p: f32,
    /// Top-k filtering count. 0 = disabled.
    pub top_k: usize,
    /// Min-p threshold. 0.0 = disabled.
    pub min_p: f32,
    /// Token IDs that signal end of generation.
    pub stop_tokens: Vec<u32>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: 512,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.0,
            stop_tokens: Vec::new(),
        }
    }
}

impl GenParams {
    /// Greedy decoding (temperature = 0).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Set the top-p (nucleus) filtering threshold.
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set the top-k filtering count.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set the min-p threshold.
    pub fn with_min_p(mut self, p: f32) -> Self {
        self.min_p = p;
        self
    }

    /// Set explicit stop tokens.
    pub fn with_stop_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.stop_tokens = tokens;
        self
    }

    /// Convert to the low-level inference request.
    pub(crate) fn to_generate_request(
        &self,
        prompt_tokens: Vec<u32>,
        _info: &ModelInfo,
    ) -> GenerateRequest {
        let sampler = SamplerConfig::default()
            .with_temperature(self.temperature)
            .with_top_p(self.top_p)
            .with_top_k(self.top_k)
            .with_min_p(self.min_p);

        let mut req = GenerateRequest::new(prompt_tokens)
            .with_sampler(sampler)
            .with_max_tokens(self.max_tokens);

        if !self.stop_tokens.is_empty() {
            req = req.with_stop_tokens(self.stop_tokens.clone());
        }

        req
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params() {
        let p = GenParams::default();
        assert!((p.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(p.max_tokens, 512);
        assert!((p.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(p.top_k, 0);
        assert!(p.stop_tokens.is_empty());
    }

    #[test]
    fn greedy_params() {
        let p = GenParams::greedy();
        assert!(p.temperature.abs() < f32::EPSILON);
    }

    #[test]
    fn builder_chain() {
        let p = GenParams::default()
            .with_temperature(0.5)
            .with_max_tokens(256)
            .with_top_p(0.95)
            .with_top_k(40)
            .with_min_p(0.05)
            .with_stop_tokens(vec![2]);
        assert!((p.temperature - 0.5).abs() < f32::EPSILON);
        assert_eq!(p.max_tokens, 256);
        assert!((p.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(p.top_k, 40);
        assert!((p.min_p - 0.05).abs() < f32::EPSILON);
        assert_eq!(p.stop_tokens, vec![2]);
    }

    #[test]
    fn to_generate_request_uses_sampler_config() {
        use mil_rs::weights::{Architecture, ModelConfig};

        let info = ironmill_core::model_info::ModelInfo::from_config(
            &ModelConfig::new(Architecture::Llama)
                .with_hidden_size(4096)
                .with_intermediate_size(11008)
                .with_num_hidden_layers(32)
                .with_num_attention_heads(32)
                .with_num_key_value_heads(8)
                .with_head_dim(128)
                .with_vocab_size(32000)
                .with_max_position_embeddings(4096),
        );
        let p = GenParams::default().with_max_tokens(100);
        let req = p.to_generate_request(vec![1, 2, 3], &info);
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.prompt_tokens, vec![1, 2, 3]);
    }
}
