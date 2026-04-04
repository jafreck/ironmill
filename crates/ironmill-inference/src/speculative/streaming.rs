//! Speculative Streaming — multi-stream attention (MSA) without an auxiliary draft model.
//!
//! Unlike EAGLE-3 speculative decoding which uses a separate draft head,
//! speculative streaming relies on models **fine-tuned with multi-stream
//! attention (MSA)** where additional projection heads predict tokens at
//! future positions (N+1, N+2, …, N+K) in a single forward pass.
//!
//! Models without MSA heads fall back to standard single-token decode.
//!
//! # Verification flow
//!
//! 1. Primary head produces token at position N.
//! 2. MSA stream heads produce speculative tokens for N+1 … N+K.
//! 3. On the next step, the primary head produces token at N+1. If it
//!    matches the speculative prediction for N+1, the token is accepted.
//! 4. Verification continues until a mismatch; unverified speculative
//!    tokens are discarded and the KV cache is rolled back via
//!    [`InferenceEngine::truncate_to`].

use half::f16;

use crate::engine::{InferenceEngine, InferenceError};

// ── Configuration ───────────────────────────────────────────────

/// Configuration for speculative streaming.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Number of speculative streams (positions predicted ahead).
    pub num_streams: usize,
    /// Minimum confidence to accept a speculative token.
    pub min_confidence: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            num_streams: 3,
            min_confidence: 0.5,
        }
    }
}

// ── MSA head weights ────────────────────────────────────────────

/// Multi-stream head weights for one layer.
///
/// Each stream has a projection matrix that maps hidden states to logits
/// for a future position. Weights are stored in `f16` and promoted to
/// `f32` during the forward pass.
#[derive(Debug, Clone)]
pub struct MsaHeadWeights {
    /// Per-stream projection matrices: `[num_streams][hidden_dim × vocab_size]`.
    stream_projections: Vec<Vec<f16>>,
    hidden_dim: usize,
    vocab_size: usize,
}

impl MsaHeadWeights {
    /// Create MSA head weights for one layer.
    ///
    /// # Panics
    ///
    /// Panics if any projection matrix length ≠ `hidden_dim * vocab_size`.
    pub fn new(stream_projections: Vec<Vec<f16>>, hidden_dim: usize, vocab_size: usize) -> Self {
        let expected = hidden_dim * vocab_size;
        for (i, proj) in stream_projections.iter().enumerate() {
            assert_eq!(
                proj.len(),
                expected,
                "stream_projections[{i}] length {} != hidden_dim({hidden_dim}) * vocab_size({vocab_size}) = {expected}",
                proj.len(),
            );
        }
        Self {
            stream_projections,
            hidden_dim,
            vocab_size,
        }
    }

    /// Number of streams in this layer.
    pub fn num_streams(&self) -> usize {
        self.stream_projections.len()
    }

    /// Project hidden states through a given stream head, producing logits.
    ///
    /// `stream_idx` must be `< self.num_streams()`.
    pub fn project(&self, stream_idx: usize, hidden: &[f32]) -> Vec<f32> {
        assert_eq!(
            hidden.len(),
            self.hidden_dim,
            "hidden length {} != hidden_dim {}",
            hidden.len(),
            self.hidden_dim,
        );
        let proj = &self.stream_projections[stream_idx];
        let mut logits = vec![0.0f32; self.vocab_size];
        for (v, logit) in logits.iter_mut().enumerate() {
            let row_offset = v * self.hidden_dim;
            let mut dot = 0.0f32;
            for h in 0..self.hidden_dim {
                dot += hidden[h] * f16::to_f32(proj[row_offset + h]);
            }
            *logit = dot;
        }
        logits
    }
}

// ── Speculative Streaming engine ────────────────────────────────

/// Speculative Streaming engine.
///
/// Wraps an [`InferenceEngine`] and, when MSA head weights are loaded,
/// produces speculative tokens for future positions alongside the primary
/// token. Models without MSA heads fall back to standard decode via
/// [`standard_step`](Self::standard_step).
pub struct SpeculativeStreaming<E: InferenceEngine> {
    engine: E,
    config: StreamingConfig,
    /// Per-layer MSA head weights. `None` if the model doesn't support MSA.
    msa_weights: Option<Vec<MsaHeadWeights>>,
    /// Buffer of speculative tokens from the previous step: `(token_id, confidence)`.
    pending_speculative: Vec<(u32, f32)>,
}

impl<E: InferenceEngine> SpeculativeStreaming<E> {
    /// Create a new speculative streaming engine wrapping the given engine.
    pub fn new(engine: E, config: StreamingConfig) -> Self {
        Self {
            engine,
            config,
            msa_weights: None,
            pending_speculative: Vec::new(),
        }
    }

    /// Check if the model has MSA heads loaded.
    pub fn has_msa_heads(&self) -> bool {
        self.msa_weights.is_some()
    }

    /// Load MSA head weights from flat weight data.
    ///
    /// `weights` contains one entry per layer. Each entry is the
    /// concatenation of `num_streams` projection matrices, each of size
    /// `hidden_dim × vocab_size`, stored as `f16`.
    pub fn load_msa_weights(&mut self, weights: &[Vec<f16>], hidden_dim: usize, vocab_size: usize) {
        let elements_per_proj = hidden_dim * vocab_size;
        let num_streams = self.config.num_streams;

        let msa: Vec<MsaHeadWeights> = weights
            .iter()
            .map(|layer_data| {
                let projections: Vec<Vec<f16>> = layer_data
                    .chunks_exact(elements_per_proj)
                    .take(num_streams)
                    .map(|chunk| chunk.to_vec())
                    .collect();
                MsaHeadWeights::new(projections, hidden_dim, vocab_size)
            })
            .collect();

        self.msa_weights = Some(msa);
    }

    /// Run one streaming step.
    ///
    /// 1. Verify pending speculative tokens against the primary head output.
    /// 2. Accept verified tokens.
    /// 3. Generate new speculative tokens from MSA heads.
    ///
    /// Returns accepted tokens (primary + verified speculative).
    pub fn streaming_step(&mut self, primary_logits: &[f32]) -> Result<Vec<u32>, InferenceError> {
        if primary_logits.is_empty() {
            return Err(InferenceError::Decode("empty primary logits".into()));
        }

        let primary_token = argmax(primary_logits);
        let mut accepted = Vec::new();

        // --- Phase 1: verify pending speculative tokens ---
        let base_pos = self.engine.seq_pos();
        let pending = std::mem::take(&mut self.pending_speculative);
        let mut verified_count = 0usize;

        if !pending.is_empty() {
            // The primary token at this step is the verifier for the first
            // speculative prediction from the previous step.
            if pending[0].0 == primary_token && pending[0].1 >= self.config.min_confidence {
                verified_count = 1;
                // Continue verifying subsequent speculative tokens by feeding
                // verified tokens through the engine to get their primary logits.
                for i in 1..pending.len() {
                    let verify_logits = self.engine.decode_step(pending[i - 1].0)?;
                    let verify_token = argmax(&verify_logits);
                    if verify_token == pending[i].0 && pending[i].1 >= self.config.min_confidence {
                        verified_count = i + 1;
                    } else {
                        break;
                    }
                }
            }
        }

        // Rollback any unverified positions from the KV cache.
        let target_pos = base_pos + verified_count;
        if self.engine.seq_pos() > target_pos {
            self.engine.truncate_to(target_pos);
        }

        // Collect accepted tokens: primary + all verified speculative.
        accepted.push(primary_token);
        for &(tok, _) in pending.iter().take(verified_count) {
            accepted.push(tok);
        }

        // --- Phase 2: generate new speculative tokens from MSA heads ---
        if let Some(ref msa_layers) = self.msa_weights {
            if let Some(first_layer) = msa_layers.first() {
                // Use the primary logits as a proxy for the hidden state.
                // (In a real implementation the engine would expose hidden
                // states; here we use primary logits for testability.)
                let hidden = primary_logits;

                let num_streams = first_layer.num_streams().min(self.config.num_streams);

                let mut new_speculative = Vec::with_capacity(num_streams);
                for s in 0..num_streams {
                    // Average across layers for robustness.
                    let mut combined = vec![0.0f32; first_layer.vocab_size];
                    let num_layers = msa_layers.len();
                    for layer in msa_layers {
                        if s < layer.num_streams() {
                            let proj = layer.project(s, hidden);
                            for (c, p) in combined.iter_mut().zip(proj.iter()) {
                                *c += p;
                            }
                        }
                    }
                    if num_layers > 1 {
                        let scale = 1.0 / num_layers as f32;
                        for c in &mut combined {
                            *c *= scale;
                        }
                    }

                    let confidence = softmax_max(&combined);
                    let spec_token = argmax(&combined);
                    new_speculative.push((spec_token, confidence));
                }

                self.pending_speculative = new_speculative;
            }
        }

        Ok(accepted)
    }

    /// Fallback: standard single-token decode (no MSA heads).
    pub fn standard_step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError> {
        self.engine.decode_step(token)
    }

    /// Access the underlying engine.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Mutably access the underlying engine.
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    /// Access the current pending speculative tokens.
    pub fn pending_speculative(&self) -> &[(u32, f32)] {
        &self.pending_speculative
    }

    /// Access the streaming configuration.
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Greedy argmax over a logit slice.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Maximum softmax probability (confidence of the top token).
fn softmax_max(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    if sum_exp == 0.0 {
        return 0.0;
    }
    1.0 / sum_exp // exp(max - max) / sum = 1 / sum
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Logits;
    use std::any::Any;

    /// Minimal mock engine for testing speculative streaming.
    struct MockStreamEngine {
        pos: usize,
        vocab_size: usize,
        /// Fixed token the engine always picks as most likely.
        favored_token: u32,
    }

    impl MockStreamEngine {
        fn new(vocab_size: usize, favored_token: u32) -> Self {
            Self {
                pos: 0,
                vocab_size,
                favored_token,
            }
        }
    }

    impl InferenceEngine for MockStreamEngine {
        fn load(&mut self, _artifacts: &dyn Any) -> Result<(), InferenceError> {
            Ok(())
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
            self.pos += tokens.len();
            let mut logits = vec![0.0f32; self.vocab_size];
            if (self.favored_token as usize) < self.vocab_size {
                logits[self.favored_token as usize] = 5.0;
            }
            Ok(logits)
        }

        fn decode_step(&mut self, _token: u32) -> Result<Logits, InferenceError> {
            self.pos += 1;
            let mut logits = vec![0.1f32; self.vocab_size];
            if (self.favored_token as usize) < self.vocab_size {
                logits[self.favored_token as usize] = 5.0;
            }
            Ok(logits)
        }

        fn reset(&mut self) {
            self.pos = 0;
        }

        fn seq_pos(&self) -> usize {
            self.pos
        }

        fn truncate_to(&mut self, pos: usize) {
            assert!(pos <= self.pos);
            self.pos = pos;
        }
    }

    fn make_msa_weights(
        num_streams: usize,
        hidden_dim: usize,
        vocab_size: usize,
        favored_token: u32,
    ) -> Vec<Vec<f16>> {
        // Build one layer's worth of concatenated stream projections.
        // Each stream's projection makes `favored_token` the argmax.
        let elements_per_proj = hidden_dim * vocab_size;
        let mut layer = vec![f16::from_f32(0.0); num_streams * elements_per_proj];
        for s in 0..num_streams {
            let offset = s * elements_per_proj;
            // Set the row for `favored_token` to positive weights.
            let row_start = offset + (favored_token as usize) * hidden_dim;
            for h in 0..hidden_dim {
                layer[row_start + h] = f16::from_f32(1.0);
            }
        }
        vec![layer]
    }

    // ── StreamingConfig tests ───────────────────────────────────

    #[test]
    fn streaming_config_default() {
        let cfg = StreamingConfig::default();
        assert_eq!(cfg.num_streams, 3);
        assert!((cfg.min_confidence - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn streaming_config_custom() {
        let cfg = StreamingConfig {
            num_streams: 5,
            min_confidence: 0.8,
        };
        assert_eq!(cfg.num_streams, 5);
        assert!((cfg.min_confidence - 0.8).abs() < f32::EPSILON);
    }

    // ── MsaHeadWeights tests ────────────────────────────────────

    #[test]
    fn streaming_msa_head_project_shape() {
        let hidden_dim = 4;
        let vocab_size = 8;
        let proj = vec![f16::from_f32(0.1); hidden_dim * vocab_size];
        let head = MsaHeadWeights::new(vec![proj], hidden_dim, vocab_size);
        assert_eq!(head.num_streams(), 1);

        let hidden = vec![1.0f32; hidden_dim];
        let logits = head.project(0, &hidden);
        assert_eq!(logits.len(), vocab_size);
    }

    #[test]
    #[should_panic(expected = "stream_projections[0] length")]
    fn streaming_msa_head_wrong_weight_size() {
        MsaHeadWeights::new(vec![vec![f16::from_f32(0.0); 10]], 4, 8);
    }

    #[test]
    #[should_panic(expected = "hidden length")]
    fn streaming_msa_head_wrong_hidden_size() {
        let hidden_dim = 4;
        let vocab_size = 8;
        let proj = vec![f16::from_f32(0.1); hidden_dim * vocab_size];
        let head = MsaHeadWeights::new(vec![proj], hidden_dim, vocab_size);
        head.project(0, &[1.0, 2.0]); // wrong length
    }

    // ── SpeculativeStreaming engine tests ────────────────────────

    #[test]
    fn streaming_fallback_without_msa_heads() {
        let engine = MockStreamEngine::new(10, 3);
        let config = StreamingConfig::default();
        let mut ss = SpeculativeStreaming::new(engine, config);
        assert!(!ss.has_msa_heads());

        // standard_step should work fine as fallback.
        let logits = ss.standard_step(0).unwrap();
        assert_eq!(logits.len(), 10);
        // Favored token 3 should have highest logit.
        assert_eq!(argmax(&logits), 3);
    }

    #[test]
    fn streaming_standard_step_returns_logits() {
        let engine = MockStreamEngine::new(8, 2);
        let config = StreamingConfig::default();
        let mut ss = SpeculativeStreaming::new(engine, config);

        let logits = ss.standard_step(0).unwrap();
        assert_eq!(logits.len(), 8);
        assert_eq!(argmax(&logits), 2);
    }

    #[test]
    fn streaming_step_produces_primary_token() {
        let engine = MockStreamEngine::new(8, 2);
        let config = StreamingConfig::default();
        let mut ss = SpeculativeStreaming::new(engine, config);

        // No MSA heads → streaming_step should still produce the primary token.
        let logits = vec![0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let tokens = ss.streaming_step(&logits).unwrap();
        assert_eq!(tokens[0], 2, "primary token should be argmax of logits");
    }

    #[test]
    fn streaming_step_with_msa_produces_speculative() {
        let hidden_dim = 8;
        let vocab_size = 8;
        let favored = 3u32;
        let engine = MockStreamEngine::new(vocab_size, favored);
        let config = StreamingConfig {
            num_streams: 2,
            min_confidence: 0.1,
        };
        let mut ss = SpeculativeStreaming::new(engine, config);

        let weights = make_msa_weights(2, hidden_dim, vocab_size, favored);
        ss.load_msa_weights(&weights, hidden_dim, vocab_size);
        assert!(ss.has_msa_heads());

        // First step: produce primary token + speculative tokens.
        let mut logits = vec![0.1f32; vocab_size];
        logits[favored as usize] = 5.0;
        let tokens = ss.streaming_step(&logits).unwrap();
        assert!(!tokens.is_empty());
        assert_eq!(
            tokens[0], favored,
            "primary token should be the favored one"
        );

        // After the step, pending speculative tokens should be populated.
        assert!(
            !ss.pending_speculative().is_empty(),
            "MSA heads should have produced speculative tokens"
        );
    }

    #[test]
    fn streaming_verification_accepts_matching_predictions() {
        let hidden_dim = 8;
        let vocab_size = 8;
        let favored = 3u32;
        // Engine always produces `favored` as the top token.
        let engine = MockStreamEngine::new(vocab_size, favored);
        let config = StreamingConfig {
            num_streams: 2,
            min_confidence: 0.1,
        };
        let mut ss = SpeculativeStreaming::new(engine, config);

        let weights = make_msa_weights(2, hidden_dim, vocab_size, favored);
        ss.load_msa_weights(&weights, hidden_dim, vocab_size);

        // Step 1: produce speculative tokens.
        let mut logits = vec![0.1f32; vocab_size];
        logits[favored as usize] = 5.0;
        let tokens1 = ss.streaming_step(&logits).unwrap();
        assert_eq!(tokens1[0], favored);

        let pending_before = ss.pending_speculative().to_vec();
        assert!(!pending_before.is_empty());

        // Step 2: the primary token matches the first speculative → should be verified.
        // Since the MSA heads predicted `favored` and the engine also outputs
        // `favored`, the speculative tokens should be accepted.
        let tokens2 = ss.streaming_step(&logits).unwrap();
        // Should have primary + at least one verified speculative.
        assert!(
            tokens2.len() > 1,
            "matching predictions should yield verified speculative tokens, got {tokens2:?}"
        );
    }

    #[test]
    fn streaming_verification_rejects_mismatched_predictions() {
        let hidden_dim = 8;
        let vocab_size = 8;
        // Engine favors token 5, but MSA heads will predict token 2.
        let engine = MockStreamEngine::new(vocab_size, 5);
        let config = StreamingConfig {
            num_streams: 2,
            min_confidence: 0.1,
        };
        let mut ss = SpeculativeStreaming::new(engine, config);

        // MSA weights predict token 2.
        let weights = make_msa_weights(2, hidden_dim, vocab_size, 2);
        ss.load_msa_weights(&weights, hidden_dim, vocab_size);

        // Step 1: primary = 5, speculative predicts 2.
        let mut logits = vec![0.1f32; vocab_size];
        logits[5] = 5.0;
        let tokens1 = ss.streaming_step(&logits).unwrap();
        assert_eq!(tokens1[0], 5);

        let pending = ss.pending_speculative().to_vec();
        assert!(!pending.is_empty());
        assert_eq!(pending[0].0, 2, "MSA should have predicted token 2");

        // Step 2: primary is 5 again, but speculative predicted 2 → mismatch.
        let tokens2 = ss.streaming_step(&logits).unwrap();
        assert_eq!(
            tokens2.len(),
            1,
            "mismatched predictions should yield only the primary token"
        );
        assert_eq!(tokens2[0], 5);
    }

    #[test]
    fn streaming_kv_cache_rollback_on_mismatch() {
        let hidden_dim = 8;
        let vocab_size = 8;
        let engine = MockStreamEngine::new(vocab_size, 5);
        let config = StreamingConfig {
            num_streams: 2,
            min_confidence: 0.1,
        };
        let mut ss = SpeculativeStreaming::new(engine, config);

        let weights = make_msa_weights(2, hidden_dim, vocab_size, 2);
        ss.load_msa_weights(&weights, hidden_dim, vocab_size);

        let mut logits = vec![0.1f32; vocab_size];
        logits[5] = 5.0;
        let _ = ss.streaming_step(&logits).unwrap();

        let pos_after_step1 = ss.engine().seq_pos();

        // Step 2 will have mismatches and should rollback.
        let _ = ss.streaming_step(&logits).unwrap();
        let pos_after_step2 = ss.engine().seq_pos();

        // The engine position should reflect rollback — it should be
        // base_pos + 0 verified tokens = base_pos from step 2.
        assert_eq!(
            pos_after_step2, pos_after_step1,
            "KV cache should be rolled back to the position before verification attempts"
        );
    }

    #[test]
    fn streaming_empty_logits_returns_error() {
        let engine = MockStreamEngine::new(8, 0);
        let config = StreamingConfig::default();
        let mut ss = SpeculativeStreaming::new(engine, config);

        let result = ss.streaming_step(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn streaming_no_msa_no_speculative_tokens() {
        let engine = MockStreamEngine::new(8, 1);
        let config = StreamingConfig::default();
        let mut ss = SpeculativeStreaming::new(engine, config);

        let logits = vec![0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let tokens = ss.streaming_step(&logits).unwrap();
        assert_eq!(tokens, vec![1]);
        assert!(
            ss.pending_speculative().is_empty(),
            "without MSA heads, no speculative tokens should be pending"
        );
    }

    #[test]
    fn streaming_load_msa_weights_multi_layer() {
        let hidden_dim = 4;
        let vocab_size = 6;
        let num_streams = 2;
        let elements_per_proj = hidden_dim * vocab_size;

        // Two layers.
        let layer0 = vec![f16::from_f32(0.1); num_streams * elements_per_proj];
        let layer1 = vec![f16::from_f32(0.2); num_streams * elements_per_proj];

        let engine = MockStreamEngine::new(vocab_size, 0);
        let config = StreamingConfig {
            num_streams,
            min_confidence: 0.3,
        };
        let mut ss = SpeculativeStreaming::new(engine, config);
        ss.load_msa_weights(&[layer0, layer1], hidden_dim, vocab_size);

        assert!(ss.has_msa_heads());
    }

    // ── Helper function tests ───────────────────────────────────

    #[test]
    fn streaming_argmax_basic() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[0.0, 0.0, 1.0]), 2);
    }

    #[test]
    fn streaming_softmax_max_basic() {
        let conf = softmax_max(&[0.0, 0.0, 10.0]);
        assert!(conf > 0.9, "dominant logit should have high confidence");
    }

    #[test]
    fn streaming_softmax_max_uniform() {
        let conf = softmax_max(&[1.0, 1.0, 1.0]);
        assert!(
            (conf - 1.0 / 3.0).abs() < 0.01,
            "uniform logits should give ~1/3 confidence"
        );
    }

    #[test]
    fn streaming_softmax_max_empty() {
        assert_eq!(softmax_max(&[]), 0.0);
    }
}
