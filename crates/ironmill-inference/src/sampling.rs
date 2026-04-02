//! Token sampling and stopping criteria.
//!
//! Shared utilities for selecting the next token from logits,
//! independent of the inference backend.

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
