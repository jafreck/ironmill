//! Token sampling and stopping criteria.
//!
//! Shared utilities for selecting the next token from logits,
//! independent of the inference backend.
//!
//! [`sample_token`] provides a simple temperature-only path used by the
//! autoregressive decode loop.  [`Sampler`] provides a full sampler chain
//! (repetition penalty → top-k → top-p → min-p → temperature → categorical
//! sample) matching the llama.cpp `llama_sampler` default ordering.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Legacy single-shot sampler (kept for backward compatibility)
// ---------------------------------------------------------------------------

/// Sample a token from logits.
///
/// - `temperature <= 0`: greedy (argmax).
/// - `temperature > 0`: temperature-scaled softmax sampling.
///
/// Returns `None` when `logits` is empty.
pub fn sample_token(logits: &[f32], temperature: f32) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }

    if temperature <= 0.0 {
        // Greedy: argmax. `total_cmp` gives deterministic ordering even with NaN logits.
        let (idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))?;
        return Some(idx as u32);
    }

    // Temperature-scaled softmax sampling.
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scaled: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_logit) / temperature).exp())
        .collect();
    let sum: f32 = scaled.iter().sum();
    let probs: Vec<f32> = scaled.iter().map(|&s| s / sum).collect();

    // Simple sampling using a pseudo-random threshold.
    // For deterministic benchmarks, callers use temperature=0 (greedy).
    let threshold = simple_random_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= threshold {
            return Some(i as u32);
        }
    }
    Some((logits.len() - 1) as u32)
}

/// Simple pseudo-random f32 in [0, 1) using a thread-local xorshift.
fn simple_random_f32() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0x1234_5678_9ABC_DEF0) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as u32 as f32) / (u32::MAX as f32)
    })
}

// ---------------------------------------------------------------------------
// EOS helpers
// ---------------------------------------------------------------------------

/// Common EOS token IDs across popular tokenizers.
///
/// - 2: LLaMA, Qwen
/// - 151643: Qwen3 (tiktoken-based)
/// - 128001: LLaMA-3
///
/// TODO: These should come from `ModelArchitecture` metadata so each model
/// declares its own EOS tokens rather than relying on a shared default.
pub const DEFAULT_EOS_TOKENS: &[u32] = &[2, 151643, 128001];

/// Check if a token ID is an end-of-sequence marker.
///
/// `eos_tokens` is the set of token IDs considered end-of-sequence.
/// Use [`DEFAULT_EOS_TOKENS`] when model-specific tokens are unavailable.
pub fn is_eos_token(token_id: u32, eos_tokens: &[u32]) -> bool {
    eos_tokens.contains(&token_id)
}

// ---------------------------------------------------------------------------
// Full sampler chain
// ---------------------------------------------------------------------------

/// Sampler configuration.
pub struct SamplerConfig {
    /// Temperature for logit scaling. 0.0 = greedy (argmax).
    pub temperature: f32,
    /// Min-p threshold. 0.0 = disabled.
    pub min_p: f32,
    /// Top-k filtering. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus) filtering. 1.0 = disabled.
    pub top_p: f32,
    /// Multiplicative repetition penalty. 1.0 = disabled.
    pub repeat_penalty: f32,
    /// Number of recent tokens to consider for repetition penalties.
    pub repeat_window: usize,
    /// Additive frequency penalty. 0.0 = disabled.
    pub frequency_penalty: f32,
    /// Additive presence penalty. 0.0 = disabled.
    pub presence_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.7,
            min_p: 0.05,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_window: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// Stateful sampler that tracks token history for repetition penalties.
///
/// Chain order (matching llama.cpp default):
/// repetition penalty → top-k → top-p → min-p → temperature → sample.
pub struct Sampler {
    config: SamplerConfig,
    recent_tokens: VecDeque<u32>,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            config,
            recent_tokens: VecDeque::new(),
        }
    }

    /// Apply the full sampler chain and return a sampled token ID.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        assert!(!logits.is_empty(), "logits must not be empty");

        // 1. Repetition / frequency / presence penalty.
        self.apply_repetition_penalty(logits);

        // 2–4. Truncation filters (operate before temperature scaling).
        apply_top_k(logits, self.config.top_k);
        apply_top_p(logits, self.config.top_p);
        apply_min_p(logits, self.config.min_p);

        // 5. Temperature → 6. Categorical sample (or greedy).
        let token = if self.config.temperature <= 0.0 {
            argmax(logits)
        } else {
            apply_temperature(logits, self.config.temperature);
            categorical_sample(logits)
        };

        // Track the token for future repetition penalties.
        self.recent_tokens.push_back(token);
        if self.recent_tokens.len() > self.config.repeat_window {
            self.recent_tokens.pop_front();
        }

        token
    }

    /// Reset token history (e.g., on conversation clear).
    pub fn reset(&mut self) {
        self.recent_tokens.clear();
    }

    // -- internals ----------------------------------------------------------

    fn apply_repetition_penalty(&self, logits: &mut [f32]) {
        if self.recent_tokens.is_empty() {
            return;
        }

        // Count occurrences for frequency penalty.
        let mut counts = std::collections::HashMap::<u32, u32>::new();
        for &tok in &self.recent_tokens {
            *counts.entry(tok).or_default() += 1;
        }

        let penalty = self.config.repeat_penalty;
        let freq = self.config.frequency_penalty;
        let pres = self.config.presence_penalty;

        for (&tok, &count) in &counts {
            let idx = tok as usize;
            if idx >= logits.len() {
                continue;
            }

            // Multiplicative repetition penalty (llama.cpp convention):
            // positive logits are divided, negative logits are multiplied.
            if penalty != 1.0 {
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }

            // Additive frequency penalty (proportional to count).
            logits[idx] -= freq * count as f32;

            // Additive presence penalty (flat if token appeared at all).
            logits[idx] -= pres;
        }
    }
}

// ---------------------------------------------------------------------------
// Individual sampler stages (free functions for testability)
// ---------------------------------------------------------------------------

/// Greedy argmax over logits.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Top-k: keep only the `k` highest logits; set the rest to -∞.
fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }

    // Find the k-th largest value via a partial sort using a min-heap of size k.
    let mut heap = std::collections::BinaryHeap::<std::cmp::Reverse<FloatOrd>>::new();
    for &v in logits.iter() {
        if v == f32::NEG_INFINITY {
            continue;
        }
        if heap.len() < k {
            heap.push(std::cmp::Reverse(FloatOrd(v)));
        } else if let Some(&std::cmp::Reverse(FloatOrd(min))) = heap.peek() {
            if v > min {
                heap.pop();
                heap.push(std::cmp::Reverse(FloatOrd(v)));
            }
        }
    }

    let threshold = heap.peek().map_or(f32::NEG_INFINITY, |r| r.0.0);
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Top-p (nucleus): keep the smallest set of tokens whose cumulative softmax
/// probability exceeds `p`; set the rest to -∞.
fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }

    let probs = softmax(logits);

    // Build (index, prob) pairs for non-neg-inf logits, sorted descending by prob.
    let mut sorted: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .filter(|(i, _)| logits[*i] != f32::NEG_INFINITY)
        .map(|(i, &pr)| (i, pr))
        .collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Walk until cumulative probability exceeds p.
    let mut cumulative = 0.0f32;
    let mut keep = std::collections::HashSet::new();
    for (idx, prob) in &sorted {
        keep.insert(*idx);
        cumulative += prob;
        if cumulative >= p {
            break;
        }
    }

    for (i, l) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Min-p: compute `threshold = min_p × max_prob`; zero out tokens with
/// `softmax(logit) < threshold` by setting their logit to -∞.
fn apply_min_p(logits: &mut [f32], min_p: f32) {
    if min_p <= 0.0 {
        return;
    }

    let probs = softmax(logits);
    let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;

    for (i, l) in logits.iter_mut().enumerate() {
        if probs[i] < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Divide logits by temperature (in-place).
fn apply_temperature(logits: &mut [f32], temperature: f32) {
    for l in logits.iter_mut() {
        if *l != f32::NEG_INFINITY {
            *l /= temperature;
        }
    }
}

/// Categorical sample from logits (applies softmax, then samples).
fn categorical_sample(logits: &[f32]) -> u32 {
    let probs = softmax(logits);
    let threshold = simple_random_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= threshold {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

/// Numerically-stable softmax returning a probability vector.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits
        .iter()
        .map(|&l| {
            if l == f32::NEG_INFINITY {
                0.0
            } else {
                (l - max_logit).exp()
            }
        })
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Newtype for total-ordering f32 in a BinaryHeap.
#[derive(Clone, Copy)]
struct FloatOrd(f32);

impl PartialEq for FloatOrd {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == std::cmp::Ordering::Equal
    }
}
impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- existing tests (preserved) ----------------------------------------

    #[test]
    fn greedy_sampling_picks_argmax() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(sample_token(&logits, 0.0), Some(3));
    }

    #[test]
    fn greedy_sampling_empty_logits() {
        assert_eq!(sample_token(&[], 0.0), None);
    }

    #[test]
    fn eos_detection() {
        assert!(is_eos_token(2, DEFAULT_EOS_TOKENS));
        assert!(is_eos_token(151643, DEFAULT_EOS_TOKENS));
        assert!(is_eos_token(128001, DEFAULT_EOS_TOKENS));
        assert!(!is_eos_token(42, DEFAULT_EOS_TOKENS));
    }

    #[test]
    fn eos_detection_custom_tokens() {
        let custom = &[99, 100];
        assert!(is_eos_token(99, custom));
        assert!(!is_eos_token(2, custom));
    }

    #[test]
    fn temperature_sampling_produces_valid_token() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let token = sample_token(&logits, 1.0).unwrap();
        assert!(token < 5);
    }

    // ---- SamplerConfig defaults -------------------------------------------

    #[test]
    fn sampler_config_defaults() {
        let c = SamplerConfig::default();
        assert_eq!(c.temperature, 0.7);
        assert_eq!(c.min_p, 0.05);
        assert_eq!(c.top_k, 0);
        assert_eq!(c.top_p, 1.0);
        assert_eq!(c.repeat_penalty, 1.0);
        assert_eq!(c.repeat_window, 64);
        assert_eq!(c.frequency_penalty, 0.0);
        assert_eq!(c.presence_penalty, 0.0);
    }

    // ---- greedy via Sampler -----------------------------------------------

    #[test]
    fn sampler_greedy_matches_legacy() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            min_p: 0.0,
            ..SamplerConfig::default()
        });
        let mut logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = sampler.sample(&mut logits);
        assert_eq!(token, 3);
    }

    // ---- disabled filters match legacy temperature sampling ---------------

    #[test]
    fn sampler_disabled_filters_valid_token() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 1.0,
            min_p: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            ..SamplerConfig::default()
        });
        for _ in 0..20 {
            let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let token = sampler.sample(&mut logits);
            assert!(token < 5, "token {token} out of range");
        }
    }

    // ---- top-k isolation --------------------------------------------------

    #[test]
    fn top_k_keeps_k_highest() {
        let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        apply_top_k(&mut logits, 2);
        // Top-2 are indices 1 (5.0) and 3 (4.0).
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[3], 4.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[4], f32::NEG_INFINITY);
    }

    #[test]
    fn top_k_disabled_when_zero() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_top_k(&mut logits, 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn top_k_larger_than_vocab_is_noop() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_top_k(&mut logits, 100);
        assert_eq!(logits, original);
    }

    // ---- top-p isolation --------------------------------------------------

    #[test]
    fn top_p_filters_low_probability_tokens() {
        // Token 0 has overwhelming probability; top-p=0.5 should keep only it.
        let mut logits = vec![10.0, 0.0, 0.0, 0.0];
        apply_top_p(&mut logits, 0.5);
        assert_eq!(logits[0], 10.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn top_p_disabled_when_one() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_top_p(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    // ---- min-p isolation --------------------------------------------------

    #[test]
    fn min_p_high_confidence_near_deterministic() {
        // One dominant logit: min-p should filter everything else.
        let mut logits = vec![10.0, 1.0, 1.0, 1.0, 1.0];
        apply_min_p(&mut logits, 0.1);
        // softmax([10,1,1,1,1]) ≈ [0.9987, 0.0003, …]; threshold = 0.1*0.9987 ≈ 0.0999.
        assert_eq!(logits[0], 10.0);
        for &l in &logits[1..] {
            assert_eq!(
                l,
                f32::NEG_INFINITY,
                "low-probability token should be filtered"
            );
        }
    }

    #[test]
    fn min_p_low_confidence_preserves_most() {
        // Uniform logits: all tokens have equal probability. min-p should keep all.
        let mut logits = vec![1.0; 10];
        apply_min_p(&mut logits, 0.05);
        for &l in &logits {
            assert_eq!(l, 1.0, "uniform logits should all survive min-p");
        }
    }

    #[test]
    fn min_p_disabled_when_zero() {
        let mut logits = vec![10.0, 1.0, 1.0];
        let original = logits.clone();
        apply_min_p(&mut logits, 0.0);
        assert_eq!(logits, original);
    }

    // ---- temperature isolation --------------------------------------------

    #[test]
    fn temperature_scales_logits() {
        let mut logits = vec![2.0, 4.0, f32::NEG_INFINITY];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    // ---- repetition penalty isolation -------------------------------------

    #[test]
    fn repetition_penalty_positive_logit() {
        let mut sampler = Sampler::new(SamplerConfig {
            repeat_penalty: 2.0,
            ..SamplerConfig::default()
        });
        sampler.recent_tokens.push_back(1);
        let mut logits = vec![0.0, 4.0, 2.0];
        sampler.apply_repetition_penalty(&mut logits);
        // Token 1: positive logit → divided by 2.0.
        assert!((logits[1] - 2.0).abs() < 1e-6);
        // Untouched tokens.
        assert!((logits[0] - 0.0).abs() < 1e-6);
        assert!((logits[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_negative_logit() {
        let mut sampler = Sampler::new(SamplerConfig {
            repeat_penalty: 2.0,
            ..SamplerConfig::default()
        });
        sampler.recent_tokens.push_back(0);
        let mut logits = vec![-3.0, 1.0];
        sampler.apply_repetition_penalty(&mut logits);
        // Token 0: negative logit → multiplied by 2.0.
        assert!((logits[0] - (-6.0)).abs() < 1e-6);
    }

    #[test]
    fn frequency_and_presence_penalty() {
        let mut sampler = Sampler::new(SamplerConfig {
            repeat_penalty: 1.0,
            frequency_penalty: 0.5,
            presence_penalty: 1.0,
            ..SamplerConfig::default()
        });
        // Token 2 appears 3 times.
        sampler.recent_tokens.push_back(2);
        sampler.recent_tokens.push_back(2);
        sampler.recent_tokens.push_back(2);
        let mut logits = vec![5.0, 5.0, 5.0];
        sampler.apply_repetition_penalty(&mut logits);
        // Token 2: 5.0 - 0.5*3 - 1.0 = 2.5
        assert!((logits[2] - 2.5).abs() < 1e-6);
        // Untouched.
        assert!((logits[0] - 5.0).abs() < 1e-6);
        assert!((logits[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn no_penalty_with_empty_history() {
        let sampler = Sampler::new(SamplerConfig {
            repeat_penalty: 2.0,
            frequency_penalty: 1.0,
            presence_penalty: 1.0,
            ..SamplerConfig::default()
        });
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        sampler.apply_repetition_penalty(&mut logits);
        assert_eq!(logits, original);
    }

    // ---- softmax helper ---------------------------------------------------

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_ignores_neg_inf() {
        let logits = vec![2.0, f32::NEG_INFINITY, 1.0];
        let probs = softmax(&logits);
        assert!((probs[1] - 0.0).abs() < 1e-6);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ---- reset ------------------------------------------------------------

    #[test]
    fn sampler_reset_clears_history() {
        let mut sampler = Sampler::new(SamplerConfig::default());
        sampler.recent_tokens.push_back(42);
        sampler.reset();
        assert!(sampler.recent_tokens.is_empty());
    }

    // ---- repeat_window cap ------------------------------------------------

    #[test]
    fn sampler_respects_repeat_window() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            min_p: 0.0,
            repeat_window: 3,
            ..SamplerConfig::default()
        });
        // Sample 5 tokens — history should never exceed 3.
        for _ in 0..5 {
            let mut logits = vec![1.0, 2.0, 3.0];
            sampler.sample(&mut logits);
        }
        assert!(sampler.recent_tokens.len() <= 3);
    }

    // ---- end-to-end: min-p with high-confidence logits --------------------

    #[test]
    fn sampler_min_p_high_confidence_deterministic() {
        // With one dominant logit and min-p enabled, sampling should always
        // pick the dominant token regardless of temperature.
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 1.0,
            min_p: 0.1,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_window: 64,
        });
        for _ in 0..50 {
            let mut logits = vec![10.0, 1.0, 1.0, 1.0, 1.0];
            let token = sampler.sample(&mut logits);
            assert_eq!(token, 0, "dominant token should always be selected");
        }
    }
}
