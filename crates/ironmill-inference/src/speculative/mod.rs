//! EAGLE-3 / P-EAGLE speculative decoding.
//!
//! This module implements speculative decoding using an EAGLE-3 draft head
//! with P-EAGLE parallel candidate generation. The core idea: a lightweight
//! draft MLP proposes a tree of candidate continuations, which are verified
//! in batch against the target model using Leviathan et al. rejection sampling.
//!
//! # Architecture
//!
//! - [`config::SpecConfig`] — tuning knobs (depth, width, threshold).
//! - [`draft_head::DraftHead`] — the EAGLE-3 MLP that projects fused target-model
//!   hidden states into draft logits.
//! - [`tree::CandidateTree`] / [`tree::DraftCandidate`] — tree-structured
//!   candidate generation and verification.
//! - [`specbundle::load_spec_bundle`] — checkpoint loader for draft-head weights.
//! - [`SpeculativeEngine`] — top-level wrapper that orchestrates the
//!   draft → verify → accept/reject → correct loop.

pub mod config;
pub mod draft_head;
pub mod specbundle;
pub mod tree;
pub mod turbospec;

use std::path::Path;

use crate::engine::{InferenceEngine, InferenceError};
use crate::sampling::Sampler;
use crate::types::Logits;

pub use config::{SpecConfig, TurboSpecConfig};
pub use draft_head::DraftHead;
pub use tree::{CandidateTree, DraftCandidate};
pub use turbospec::TurboSpecController;

/// Speculative decoding engine wrapping a target [`InferenceEngine`].
///
/// When a [`DraftHead`] is loaded, [`speculative_step`](Self::speculative_step)
/// performs one full speculation round (draft → verify → accept/reject → correct).
/// Without a draft head, it falls back to standard single-token decode via
/// [`standard_step`](Self::standard_step).
pub struct SpeculativeEngine<E: InferenceEngine> {
    engine: E,
    draft_head: Option<DraftHead>,
    config: SpecConfig,
    sampler: Sampler,
    turbospec: Option<TurboSpecController>,
}

impl<E: InferenceEngine> SpeculativeEngine<E> {
    /// Create a new speculative engine wrapping the given target engine.
    pub fn new(engine: E, config: SpecConfig, sampler: Sampler) -> Self {
        Self {
            engine,
            draft_head: None,
            config,
            sampler,
            turbospec: None,
        }
    }

    /// Enable TurboSpec adaptive speculation control.
    pub fn with_turbospec(mut self, config: TurboSpecConfig) -> Self {
        self.turbospec = Some(TurboSpecController::new(config));
        self
    }

    /// Load a draft head from a SpecBundle checkpoint directory.
    pub fn load_draft_head(&mut self, path: &Path) -> Result<(), InferenceError> {
        let head = specbundle::load_spec_bundle(path)?;
        self.draft_head = Some(head);
        Ok(())
    }

    /// Run one speculation round: draft → verify → accept/reject → correct.
    ///
    /// Returns the accepted token IDs. If no draft head is loaded, falls back
    /// to [`standard_step`](Self::standard_step) and returns a single token.
    pub fn speculative_step(
        &mut self,
        last_token: u32,
        last_hidden: &[f32],
    ) -> Result<Vec<u32>, InferenceError> {
        // Apply TurboSpec-tuned config before drafting.
        if let Some(ref ts) = self.turbospec {
            self.config = ts.current_config();
        }

        let draft_head = match &self.draft_head {
            Some(h) => h,
            None => {
                let token = self.standard_step(last_token)?;
                return Ok(vec![token]);
            }
        };

        // 1. Build candidate tree from draft head.
        let vocab_size = draft_head.vocab_size;
        let tree = CandidateTree::build(
            &self.config,
            vocab_size,
            |_parent_token, _depth| draft_head.forward(last_hidden),
            last_token,
        );

        if tree.candidates.is_empty() {
            let token = self.standard_step(last_token)?;
            return Ok(vec![token]);
        }

        // Save position for rollback.
        let base_pos = self.engine.seq_pos();

        // 2. Prefill candidates through target model (DFS order with
        //    KV cache truncation between siblings).
        let target_logits_batch = self.prefill_candidates(&tree.candidates)?;

        // 3. Compute target log-probabilities for each candidate.
        let target_log_probs = compute_target_log_probs(&target_logits_batch, &tree.candidates);

        // 4. Verify via rejection sampling.
        let accepted_indices = tree.verify(&target_log_probs, &self.config);

        // 5. Collect accepted tokens.
        let accepted_tokens: Vec<u32> = accepted_indices
            .iter()
            .map(|&i| tree.candidates[i].token_id)
            .collect();

        // 5b. Feed results to TurboSpec controller.
        if let Some(ref mut ts) = self.turbospec {
            ts.observe(tree.candidates.len(), accepted_tokens.len());
        }

        // 6. Sample correction token from adjusted distribution.
        //    decode_step(token) returns logits predicting what comes *after*
        //    that token, so target_logits_batch[i] = p(·|prefix, d₀…dᵢ).
        //    After accepting [0…j] the correction token comes from
        //    target_logits_batch[last_accepted].
        let correction_logits = if accepted_indices.is_empty() {
            // All rejected — use the first position's target logits.
            target_logits_batch.first().cloned().unwrap_or_default()
        } else {
            let last_accepted = *accepted_indices.last().unwrap();
            target_logits_batch[last_accepted].clone()
        };

        let correction_token = if !correction_logits.is_empty() {
            self.sampler.sample(&mut correction_logits.clone())
        } else {
            // Shouldn't happen, but fallback.
            return Ok(accepted_tokens);
        };

        // 7. Rollback KV cache to last accepted position + 1.
        //    The correction token is returned for the caller to feed into the
        //    next decode step; it does not occupy a KV cache slot yet.
        let rollback_pos = base_pos + accepted_tokens.len();
        let current_pos = self.engine.seq_pos();
        if rollback_pos < current_pos {
            self.engine.truncate_to(rollback_pos);
        }

        // 8. Combine accepted tokens + correction token.
        let mut result = accepted_tokens;
        result.push(correction_token);
        Ok(result)
    }

    /// Fallback: standard single-token decode.
    pub fn standard_step(&mut self, token: u32) -> Result<u32, InferenceError> {
        let mut logits = self.engine.decode_step(token)?;
        let token = self.sampler.sample(&mut logits);
        Ok(token)
    }

    /// Access the underlying engine.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Mutably access the underlying engine.
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    /// Check whether a draft head is loaded.
    pub fn has_draft_head(&self) -> bool {
        self.draft_head.is_some()
    }

    /// Access the TurboSpec controller, if enabled.
    pub fn turbospec(&self) -> Option<&TurboSpecController> {
        self.turbospec.as_ref()
    }

    /// Prefill candidate tokens through the target model, collecting
    /// logits for each position.
    ///
    /// Candidates are processed in DFS order. Before each candidate the KV
    /// cache is truncated to `base_pos + depth` so siblings at the same
    /// depth never see each other — only their shared ancestor chain.
    /// The best (first) child at each level is processed last so the cache
    /// ends with the best-path tokens, keeping the post-verification
    /// rollback correct.
    fn prefill_candidates(
        &mut self,
        candidates: &[DraftCandidate],
    ) -> Result<Vec<Logits>, InferenceError> {
        let base_pos = self.engine.seq_pos();
        let n = candidates.len();
        let mut all_logits: Vec<Logits> = vec![Vec::new(); n];

        let order = dfs_order_best_last(candidates);

        for &idx in &order {
            let target_pos = base_pos + candidates[idx].depth;
            if self.engine.seq_pos() > target_pos {
                self.engine.truncate_to(target_pos);
            }
            let logits = self.engine.decode_step(candidates[idx].token_id)?;
            all_logits[idx] = logits;
        }

        Ok(all_logits)
    }
}

/// Compute target-model log-probabilities for each candidate token.
fn compute_target_log_probs(
    target_logits_batch: &[Logits],
    candidates: &[DraftCandidate],
) -> Vec<f32> {
    candidates
        .iter()
        .enumerate()
        .map(|(i, candidate)| {
            if i >= target_logits_batch.len() {
                return f32::NEG_INFINITY;
            }
            let logits = &target_logits_batch[i];
            let log_probs = log_softmax(logits);
            let tok = candidate.token_id as usize;
            if tok < log_probs.len() {
                log_probs[tok]
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

/// Numerically-stable log-softmax.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    logits.iter().map(|&l| l - max - log_sum_exp).collect()
}

/// DFS traversal order over candidates, processing the best (first) child
/// last among siblings.
///
/// `CandidateTree::build` always places the highest-probability child first
/// among siblings.  By pushing children onto the stack in forward order the
/// best child ends up at the bottom and is popped (processed) last.  This
/// guarantees that after the full traversal the KV cache contains exactly
/// the best root-to-leaf path, which is the path that `verify()` walks.
fn dfs_order_best_last(candidates: &[DraftCandidate]) -> Vec<usize> {
    let n = candidates.len();
    if n == 0 {
        return Vec::new();
    }

    let mut children: Vec<Vec<usize>> = vec![vec![]; n];
    let mut roots: Vec<usize> = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        match c.parent_idx {
            Some(p) => children[p].push(i),
            None => roots.push(i),
        }
    }

    // Push roots/children in forward order so the best (first) entry sits
    // at the bottom of the stack and is popped last.
    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<usize> = roots;
    while let Some(idx) = stack.pop() {
        order.push(idx);
        for &kid in children[idx].iter() {
            stack.push(kid);
        }
    }
    order
}

// ── Free function for engine.rs ──────────────────────────────────

/// Run one speculative decode round as a free function.
///
/// This is the entry point called from `engine.rs`. It delegates to
/// [`SpeculativeEngine::speculative_step`] if a draft head is available,
/// otherwise falls back to standard decode.
pub fn speculative_decode<E: InferenceEngine>(
    spec_engine: &mut SpeculativeEngine<E>,
    last_token: u32,
    last_hidden: &[f32],
) -> Result<Vec<u32>, InferenceError> {
    spec_engine.speculative_step(last_token, last_hidden)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplerConfig;
    use crate::types::Logits;
    use std::any::Any;

    /// Minimal mock engine for testing speculative decoding.
    struct MockEngine {
        pos: usize,
        vocab_size: usize,
    }

    impl MockEngine {
        fn new(vocab_size: usize) -> Self {
            Self { pos: 0, vocab_size }
        }
    }

    impl InferenceEngine for MockEngine {
        fn load(&mut self, _artifacts: &dyn Any) -> Result<(), InferenceError> {
            Ok(())
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
            self.pos += tokens.len();
            let mut logits = vec![0.0f32; self.vocab_size];
            if !logits.is_empty() {
                logits[0] = 1.0;
            }
            Ok(logits)
        }

        fn decode_step(&mut self, _token: u32) -> Result<Logits, InferenceError> {
            self.pos += 1;
            let mut logits = vec![0.1f32; self.vocab_size];
            // Make token 1 the most likely.
            if self.vocab_size > 1 {
                logits[1] = 2.0;
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

    #[test]
    fn speculative_engine_fallback_without_draft_head() {
        let engine = MockEngine::new(10);
        let config = SpecConfig::default();
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0, // greedy
            ..SamplerConfig::default()
        });

        let mut spec = SpeculativeEngine::new(engine, config, sampler);
        assert!(!spec.has_draft_head());

        let tokens = spec.speculative_step(0, &[]).unwrap();
        assert_eq!(tokens.len(), 1, "fallback should produce exactly one token");
        // Greedy should pick token 1 (highest logit).
        assert_eq!(tokens[0], 1);
    }

    #[test]
    fn speculative_engine_standard_step() {
        let engine = MockEngine::new(10);
        let config = SpecConfig::default();
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        });

        let mut spec = SpeculativeEngine::new(engine, config, sampler);
        let token = spec.standard_step(0).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn speculative_engine_with_draft_head() {
        let engine = MockEngine::new(8);
        let config = SpecConfig {
            max_draft_depth: 2,
            tree_width: 2,
            acceptance_threshold: 0.01,
        };
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        });

        let mut spec = SpeculativeEngine::new(engine, config, sampler);

        // Create a simple draft head.
        let hidden_dim = 4;
        let vocab_size = 8;
        let weights = vec![half::f16::from_f32(0.1); hidden_dim * vocab_size];
        let head = DraftHead::new(vec![weights], hidden_dim, vocab_size);
        spec.draft_head = Some(head);
        assert!(spec.has_draft_head());

        let hidden = vec![1.0f32; hidden_dim];
        let tokens = spec.speculative_step(0, &hidden).unwrap();
        // Should produce at least 1 token (the correction token).
        assert!(!tokens.is_empty(), "should produce at least one token");
    }

    #[test]
    fn speculative_engine_kv_cache_rollback() {
        let engine = MockEngine::new(8);
        let config = SpecConfig {
            max_draft_depth: 3,
            tree_width: 1,
            acceptance_threshold: 0.01,
        };
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        });

        let mut spec = SpeculativeEngine::new(engine, config, sampler);

        let hidden_dim = 4;
        let vocab_size = 8;
        let weights = vec![half::f16::from_f32(0.1); hidden_dim * vocab_size];
        let head = DraftHead::new(vec![weights], hidden_dim, vocab_size);
        spec.draft_head = Some(head);

        let initial_pos = spec.engine().seq_pos();
        let hidden = vec![1.0f32; hidden_dim];
        let tokens = spec.speculative_step(0, &hidden).unwrap();

        // After speculation, seq_pos should be base + accepted count.
        // The correction token is returned but not yet in the KV cache.
        let final_pos = spec.engine().seq_pos();
        let accepted_count = tokens.len().saturating_sub(1); // last is correction
        assert_eq!(
            final_pos,
            initial_pos + accepted_count,
            "KV cache should be rolled back to accepted position"
        );
    }

    #[test]
    fn speculative_free_function_delegates() {
        let engine = MockEngine::new(10);
        let config = SpecConfig::default();
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        });

        let mut spec = SpeculativeEngine::new(engine, config, sampler);
        let tokens = speculative_decode(&mut spec, 0, &[]).unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], 1);
    }

    #[test]
    fn speculative_compute_target_log_probs_basic() {
        let logits = vec![vec![1.0, 2.0, 3.0]];
        let candidates = vec![DraftCandidate {
            token_id: 2,
            log_prob: -0.5,
            parent_idx: None,
            depth: 0,
        }];
        let probs = compute_target_log_probs(&logits, &candidates);
        assert_eq!(probs.len(), 1);
        // Token 2 has logit 3.0; log_softmax([1,2,3])[2] should be the highest.
        assert!(probs[0] > -1.0, "token 2 should have high log-prob");
    }

    #[test]
    fn speculative_compute_target_log_probs_out_of_range() {
        let logits = vec![vec![1.0, 2.0]];
        let candidates = vec![DraftCandidate {
            token_id: 99, // out of range
            log_prob: -0.5,
            parent_idx: None,
            depth: 0,
        }];
        let probs = compute_target_log_probs(&logits, &candidates);
        assert_eq!(probs[0], f32::NEG_INFINITY);
    }

    #[test]
    fn turbospec_builder_enables_controller() {
        let engine = MockEngine::new(10);
        let config = SpecConfig::default();
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        });

        let spec = SpeculativeEngine::new(engine, config, sampler)
            .with_turbospec(TurboSpecConfig::default());
        assert!(spec.turbospec().is_some());
    }

    #[test]
    fn turbospec_integration_with_draft_head() {
        let engine = MockEngine::new(8);
        let config = SpecConfig {
            max_draft_depth: 2,
            tree_width: 2,
            acceptance_threshold: 0.01,
        };
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        });

        let mut spec =
            SpeculativeEngine::new(engine, config, sampler).with_turbospec(TurboSpecConfig {
                initial_depth: 2,
                min_depth: 1,
                max_depth: 8,
                ema_alpha: 0.5,
                depth_up_threshold: 0.8,
                depth_down_threshold: 0.4,
            });

        let hidden_dim = 4;
        let vocab_size = 8;
        let weights = vec![half::f16::from_f32(0.1); hidden_dim * vocab_size];
        let head = DraftHead::new(vec![weights], hidden_dim, vocab_size);
        spec.draft_head = Some(head);

        let hidden = vec![1.0f32; hidden_dim];
        // Run several speculation rounds.
        for _ in 0..10 {
            let _tokens = spec.speculative_step(0, &hidden).unwrap();
            // Reset engine position for next round.
            spec.engine_mut().reset();
        }

        let ts = spec.turbospec().unwrap();
        assert!(ts.total_proposed() > 0, "should have observed proposals");
    }

    #[test]
    fn turbospec_not_enabled_by_default() {
        let engine = MockEngine::new(10);
        let config = SpecConfig::default();
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        });

        let spec = SpeculativeEngine::new(engine, config, sampler);
        assert!(spec.turbospec().is_none());
    }
}
