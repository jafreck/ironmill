//! Perplexity evaluation for LLM quantization quality.
//!
//! Measures how well a model predicts text by computing per-token
//! cross-entropy on a pre-tokenized corpus (WikiText-2). Lower
//! perplexity = better quality.

use anyhow::Result;
use std::path::Path;

/// A pre-tokenized evaluation dataset.
#[derive(serde::Deserialize)]
pub struct PerplexityDataset {
    pub name: String,
    pub model: String,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub num_sequences: usize,
    pub eos_token_id: Option<u32>,
    pub sequences: Vec<Vec<u32>>,
}

impl PerplexityDataset {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }
}

/// Result of perplexity evaluation for one optimization config.
pub struct PerplexityResult {
    pub config_name: String,
    pub perplexity: f64,
    pub avg_cross_entropy: f64,
    pub num_tokens_evaluated: usize,
    pub num_sequences: usize,
}

/// Compute cross-entropy for a single position.
///
/// Given the model's raw logits and the ground-truth next token,
/// computes `-log(softmax(logits)[target])`.
///
/// Uses log-sum-exp for numerical stability.
pub fn cross_entropy(logits: &[f32], target: u32) -> f64 {
    let target = target as usize;
    if target >= logits.len() {
        return (logits.len() as f64).ln();
    }

    // log-softmax = logit[target] - log(sum(exp(logits)))
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f64 = logits
        .iter()
        .map(|&x| ((x - max_logit) as f64).exp())
        .sum::<f64>()
        .ln()
        + max_logit as f64;

    let log_prob = logits[target] as f64 - log_sum_exp;
    -log_prob
}

/// Compute perplexity from a vector of per-token cross-entropy losses.
pub fn perplexity_from_losses(losses: &[f64]) -> f64 {
    if losses.is_empty() {
        return f64::INFINITY;
    }
    let avg_ce = losses.iter().sum::<f64>() / losses.len() as f64;
    avg_ce.exp()
}

/// A single sliding-window evaluation step.
///
/// Returned by [`sliding_window_schedule`] to describe which token range to
/// feed into the model and which positions to count losses for.
pub struct WindowStep {
    /// Start index into the full token stream.
    pub begin: usize,
    /// End index (exclusive) into the full token stream.
    pub end: usize,
    /// Index within the window where loss counting starts (inclusive).
    /// Positions before this are context-only (already counted by a
    /// previous window).
    pub loss_start: usize,
}

/// Compute the sliding-window schedule for a corpus of `total_tokens` tokens.
///
/// Uses the standard HuggingFace PPL methodology: windows of `max_length`
/// tokens are slid by `stride` positions. Only the last `stride` tokens
/// in each window contribute losses (except the first window, which
/// counts all tokens).
///
/// Returns a sequence of [`WindowStep`] descriptors.
pub fn sliding_window_schedule(
    total_tokens: usize,
    max_length: usize,
    stride: usize,
) -> Vec<WindowStep> {
    let mut steps = Vec::new();
    let mut prev_end: usize = 0;

    let mut begin: usize = 0;
    while begin < total_tokens {
        let end = (begin + max_length).min(total_tokens);
        if end <= 1 {
            break;
        }
        let trg_len = end - prev_end;
        // loss_start is the position within this window where we begin
        // counting losses. Positions before this were already evaluated
        // in the previous window.
        let window_len = end - begin;
        let loss_start = window_len.saturating_sub(trg_len);

        steps.push(WindowStep {
            begin,
            end,
            loss_start,
        });

        prev_end = end;
        if end == total_tokens {
            break;
        }
        begin += stride;
    }
    steps
}

/// Evaluate perplexity of a model on a pre-tokenized dataset.
///
/// Feeds tokens one at a time via `decode()`, collecting cross-entropy
/// at each position against the ground-truth next token.
#[cfg(feature = "ane-direct")]
pub fn evaluate_perplexity(
    inference: &mut ironmill_inference::ane::AneInference<
        ironmill_inference::ane::HardwareAneDevice,
    >,
    dataset: &PerplexityDataset,
    max_sequences: Option<usize>,
) -> Result<PerplexityResult> {
    let num_sequences = max_sequences
        .map(|m| m.min(dataset.num_sequences))
        .unwrap_or(dataset.num_sequences);

    let mut running_sum: f64 = 0.0;
    let mut loss_count: usize = 0;
    let start = std::time::Instant::now();

    for (seq_idx, sequence) in dataset.sequences.iter().take(num_sequences).enumerate() {
        inference.reset();

        for pos in 0..sequence.len() - 1 {
            let token = sequence[pos];
            let target = sequence[pos + 1];

            let logits = inference.decode(token)?;
            let ce = cross_entropy(&logits, target);
            running_sum += ce;
            loss_count += 1;

            if seq_idx == 0 && pos < 10 && std::env::var("IRONMILL_TRACE_CE").is_ok() {
                let argmax = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                eprintln!(
                    "  [ce] pos={} tok={} tgt={} argmax={} ce={:.2}",
                    pos, token, target, argmax, ce
                );
            }
        }

        if (seq_idx + 1) % 10 == 0 || seq_idx + 1 == num_sequences {
            let running_ppl = (running_sum / loss_count as f64).exp();
            let elapsed = start.elapsed().as_secs_f64();
            let tok_per_sec = loss_count as f64 / elapsed;
            eprintln!(
                "  [{}/{}] PPL: {:.2} ({} tokens, {:.1} tok/s)",
                seq_idx + 1,
                num_sequences,
                running_ppl,
                loss_count,
                tok_per_sec,
            );
        }
    }

    let avg_cross_entropy = if loss_count == 0 {
        f64::INFINITY
    } else {
        running_sum / loss_count as f64
    };
    let perplexity = avg_cross_entropy.exp();

    Ok(PerplexityResult {
        config_name: String::new(),
        perplexity,
        avg_cross_entropy,
        num_tokens_evaluated: loss_count,
        num_sequences,
    })
}

/// Format a perplexity result table for terminal output.
pub fn format_perplexity_table(results: &[PerplexityResult]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<25} {:>10} {:>10} {:>8} {:>10}\n",
        "Config", "PPL", "Avg CE", "Tokens", "Sequences"
    ));
    out.push_str(&format!("{}\n", "─".repeat(67)));

    let baseline_ppl = results.first().map(|r| r.perplexity);

    for r in results {
        let delta = match baseline_ppl {
            Some(base) if base > 0.0 => {
                let pct = (r.perplexity - base) / base * 100.0;
                if pct.abs() < 0.05 {
                    "—".to_string()
                } else {
                    format!("{:+.1}%", pct)
                }
            }
            _ => "—".to_string(),
        };
        out.push_str(&format!(
            "{:<25} {:>10.2} {:>10.4} {:>8} {:>10}  {}\n",
            r.config_name,
            r.perplexity,
            r.avg_cross_entropy,
            r.num_tokens_evaluated,
            r.num_sequences,
            delta,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_perfect_prediction() {
        let logits = vec![-10.0, -10.0, 100.0, -10.0];
        let ce = cross_entropy(&logits, 2);
        assert!(
            ce < 1e-6,
            "CE should be ~0 for perfect prediction, got {ce}"
        );
    }

    #[test]
    fn cross_entropy_uniform() {
        // Uniform logits over 4 tokens → CE = ln(4) ≈ 1.386
        let logits = vec![0.0; 4];
        let ce = cross_entropy(&logits, 0);
        assert!(
            (ce - (4.0_f64).ln()).abs() < 1e-6,
            "Expected ln(4), got {ce}"
        );
    }

    #[test]
    fn cross_entropy_out_of_range() {
        // Out-of-range target falls back to ln(vocab_size) — the cross-entropy
        // of a uniform distribution over the vocabulary.
        let logits = vec![1.0, 2.0, 3.0];
        let ce = cross_entropy(&logits, 100);
        let expected = (logits.len() as f64).ln();
        assert!(
            (ce - expected).abs() < 1e-10,
            "Out-of-range target should return ln(vocab_size), got {ce}"
        );
    }

    #[test]
    fn cross_entropy_strongly_wrong() {
        // Logits strongly favor token 0, but target is token 3
        let logits = vec![100.0, -10.0, -10.0, -10.0];
        let ce = cross_entropy(&logits, 3);
        assert!(
            ce > 100.0,
            "CE should be large for wrong prediction, got {ce}"
        );
    }

    #[test]
    fn perplexity_from_uniform() {
        // Uniform over V=32000 → PPL = 32000
        let ce = (32000.0_f64).ln();
        let losses = vec![ce; 100];
        let ppl = perplexity_from_losses(&losses);
        assert!(
            (ppl - 32000.0).abs() < 1.0,
            "Expected PPL ~32000, got {ppl}"
        );
    }

    #[test]
    fn perplexity_empty_losses() {
        assert!(perplexity_from_losses(&[]).is_infinite());
    }

    #[test]
    fn perplexity_perfect() {
        // Perfect predictions → CE ≈ 0 → PPL ≈ 1
        let losses = vec![0.0; 100];
        let ppl = perplexity_from_losses(&losses);
        assert!(
            (ppl - 1.0).abs() < 1e-6,
            "Perfect predictions should give PPL=1, got {ppl}"
        );
    }

    #[test]
    fn format_table_smoke() {
        let results = vec![
            PerplexityResult {
                config_name: "FP32 baseline".into(),
                perplexity: 12.41,
                avg_cross_entropy: 2.518,
                num_tokens_evaluated: 25600,
                num_sequences: 50,
            },
            PerplexityResult {
                config_name: "FP16".into(),
                perplexity: 12.42,
                avg_cross_entropy: 2.519,
                num_tokens_evaluated: 25600,
                num_sequences: 50,
            },
        ];
        let table = format_perplexity_table(&results);
        assert!(table.contains("FP32 baseline"));
        assert!(table.contains("FP16"));
        assert!(table.contains("12.41"));
    }

    #[test]
    fn dataset_deserialize() {
        let json = r#"{
            "name": "test",
            "model": "test-model",
            "vocab_size": 100,
            "seq_len": 4,
            "num_sequences": 2,
            "eos_token_id": 0,
            "sequences": [[1,2,3,4],[5,6,7,8]]
        }"#;
        let ds: PerplexityDataset = serde_json::from_str(json).unwrap();
        assert_eq!(ds.num_sequences, 2);
        assert_eq!(ds.sequences[0], vec![1, 2, 3, 4]);
    }

    #[test]
    fn sliding_window_full_stride() {
        // stride == max_length → no overlap, one window per chunk
        let steps = sliding_window_schedule(8, 4, 4);
        assert_eq!(steps.len(), 2);
        assert_eq!(
            (steps[0].begin, steps[0].end, steps[0].loss_start),
            (0, 4, 0)
        );
        assert_eq!(
            (steps[1].begin, steps[1].end, steps[1].loss_start),
            (4, 8, 0)
        );
    }

    #[test]
    fn sliding_window_overlapping() {
        // 10 tokens, window=6, stride=3
        // Window 0: [0,6), all tokens new, loss_start=0
        // Window 1: [3,9), new tokens=[6,9), loss_start=3
        // Window 2: [6,10), new tokens=[9,10), loss_start=3
        let steps = sliding_window_schedule(10, 6, 3);
        assert_eq!(steps.len(), 3);
        assert_eq!(
            (steps[0].begin, steps[0].end, steps[0].loss_start),
            (0, 6, 0)
        );
        assert_eq!(
            (steps[1].begin, steps[1].end, steps[1].loss_start),
            (3, 9, 3)
        );
        assert_eq!(
            (steps[2].begin, steps[2].end, steps[2].loss_start),
            (6, 10, 3)
        );
    }

    #[test]
    fn sliding_window_exact_fit() {
        // 2048 tokens, window=2048, stride=512 → first window covers all
        let steps = sliding_window_schedule(2048, 2048, 512);
        assert_eq!(steps.len(), 1);
        assert_eq!(
            (steps[0].begin, steps[0].end, steps[0].loss_start),
            (0, 2048, 0)
        );
    }

    #[test]
    fn sliding_window_standard_hf() {
        // 4096 tokens, window=2048, stride=512
        // Window 0: [0, 2048), loss_start=0 (2048 new tokens)
        // Window 1: [512, 2560), loss_start=1536 (512 new tokens)
        // ...continuing until end
        let steps = sliding_window_schedule(4096, 2048, 512);
        assert_eq!(steps[0].loss_start, 0);
        assert_eq!(steps[1].loss_start, 2048 - 512);
        // Total tokens counted should equal total - 1 (no target for last token)
        // Each counted token appears exactly once
        let total_counted: usize = steps
            .iter()
            .map(|s| {
                let wlen = s.end - s.begin;
                // positions loss_start..wlen-1 are counted
                wlen.saturating_sub(1).saturating_sub(s.loss_start)
            })
            .sum();
        assert_eq!(total_counted, 4095); // 4096 - 1
    }
}
