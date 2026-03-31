# Perplexity Benchmark Implementation

> **Status**: Not started
>
> **Prerequisite**: ~~`AneInference::decode()` must produce correct logits~~ ✅ FP16 attention implemented (`c632f05`)
>
> **Goal**: Measure perplexity on WikiText-2 for each quantization config
> and cross-reference with weight-level SNR to validate that ironmill's
> optimizations preserve model quality

## Background

Weight-level metrics (SNR, cosine similarity) tell us how close quantized
weights are to the originals, but not whether the model still works. A model
can have 44.7 dB SNR and still produce gibberish if errors accumulate across
28 transformer layers. Perplexity is the standard metric for evaluating LLM
quantization quality.

**Perplexity** = `exp(mean(cross_entropy_losses))` over a corpus. Lower is
better. For Qwen3-0.6B, FP32 baseline perplexity on WikiText-2 is typically
~12–13. A good INT8 quantization adds <1% to perplexity.

## Design

### Data flow

```
Pre-tokenized WikiText-2 fixture (JSON)
         │
         ▼
  ┌──────────────┐
  │ Load token    │  tests/fixtures/quality/wikitext2-qwen3.json
  │ sequences     │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ For each      │  AneInference::decode(token) → logits: Vec<f32>
  │ token in seq  │  No sampling — we need the full logit distribution
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Cross-entropy │  CE = -log(softmax(logits)[ground_truth_next_token])
  │ per position  │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Aggregate     │  PPL = exp(mean(all CE values))
  │ perplexity    │
  └──────────────┘
```

### What already exists

| Component | Location | Status |
|-----------|----------|--------|
| ANE inference loop | `ironmill-ane/src/inference.rs` | `decode()` returns `Vec<f32>` logits |
| Token sampling | `inference.rs::sample_token()` | Greedy argmax or temperature sampling |
| Weight fidelity bench | `ironmill-bench/src/quality.rs` | MSE/PSNR per tensor, wired to `--quality` |
| Quality plan | `docs/development/QUALITY_BENCHMARK_PLAN.md` | Phases 1–6 outlined |
| E2E bench | `ironmill-ane/examples/turboquant_e2e_bench.rs` | Token agreement only (no logit comparison) |

### What needs to be built

1. **Pre-tokenized dataset fixture** + preparation script
2. **Perplexity evaluation module** in `ironmill-bench`
3. **Cross-entropy math** (log-softmax + gather)
4. **Evaluation loop** that calls `decode()` without sampling
5. **CLI integration** via `ironmill-bench --perplexity`

## Implementation

### Step 1: Dataset preparation script

Create `scripts/prepare-quality-dataset.py`:

```python
"""Pre-tokenize WikiText-2 for perplexity evaluation.

Usage:
    pip install datasets transformers
    python scripts/prepare-quality-dataset.py
"""
import json
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Concatenate all text, split into fixed-length sequences
full_text = "\n\n".join([t for t in dataset["text"] if t.strip()])
tokens = tokenizer.encode(full_text)

SEQ_LEN = 512
sequences = []
for i in range(0, len(tokens) - SEQ_LEN, SEQ_LEN):
    sequences.append(tokens[i : i + SEQ_LEN])

# Cap at 500 sequences for tractable eval time
sequences = sequences[:500]

output = {
    "name": "WikiText-2 (Qwen3-0.6B tokenizer)",
    "model": "Qwen3-0.6B",
    "tokenizer": "Qwen/Qwen3-0.6B",
    "vocab_size": tokenizer.vocab_size,
    "seq_len": SEQ_LEN,
    "num_sequences": len(sequences),
    "eos_token_id": tokenizer.eos_token_id,
    "sequences": sequences,
}

out_path = "tests/fixtures/quality/wikitext2-qwen3.json"
with open(out_path, "w") as f:
    json.dump(output, f)

print(f"Wrote {len(sequences)} sequences ({SEQ_LEN} tokens each) to {out_path}")
```

Add to `scripts/download-fixtures.sh`:

```bash
if [ ! -f "$FIXTURE_DIR/quality/wikitext2-qwen3.json" ]; then
    echo "==> Pre-tokenizing WikiText-2 for Qwen3..."
    python3 scripts/prepare-quality-dataset.py
fi
```

### Step 2: Perplexity module

Create `crates/ironmill-bench/src/perplexity.rs`:

```rust
//! Perplexity evaluation for LLM quantization quality.

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
/// computes -log(softmax(logits)[target]).
///
/// Uses log-sum-exp for numerical stability.
pub fn cross_entropy(logits: &[f32], target: u32) -> f64 {
    let target = target as usize;
    if target >= logits.len() {
        return 0.0;
    }

    // log-softmax = logit[target] - log(sum(exp(logits)))
    // Use max subtraction for numerical stability.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_perfect_prediction() {
        // Logits strongly favoring token 2
        let logits = vec![-10.0, -10.0, 100.0, -10.0];
        let ce = cross_entropy(&logits, 2);
        assert!(ce < 1e-6, "CE should be ~0 for perfect prediction, got {ce}");
    }

    #[test]
    fn cross_entropy_uniform() {
        // Uniform logits over 4 tokens → CE = ln(4) ≈ 1.386
        let logits = vec![0.0; 4];
        let ce = cross_entropy(&logits, 0);
        assert!((ce - (4.0_f64).ln()).abs() < 1e-6);
    }

    #[test]
    fn perplexity_from_uniform() {
        // Uniform over V=32000 tokens → PPL = 32000
        let ce = (32000.0_f64).ln();
        let losses = vec![ce; 100];
        let ppl = perplexity_from_losses(&losses);
        assert!((ppl - 32000.0).abs() < 1.0);
    }
}
```

### Step 3: Evaluation loop

Add to `perplexity.rs` (requires `ironmill-ane` as dependency):

```rust
/// Evaluate perplexity of a compiled model on a dataset.
///
/// For each sequence, runs the model autoregressively and computes
/// cross-entropy at every position. The model's `decode()` must
/// return valid logits (requires real attention, not pass-through).
pub fn evaluate_perplexity(
    inference: &mut ironmill_ane::AneInference,
    dataset: &PerplexityDataset,
    max_sequences: Option<usize>,
) -> Result<PerplexityResult> {
    let num_sequences = max_sequences
        .map(|m| m.min(dataset.num_sequences))
        .unwrap_or(dataset.num_sequences);

    let mut all_losses = Vec::new();

    for (seq_idx, sequence) in dataset.sequences.iter().take(num_sequences).enumerate() {
        // Reset KV cache for each sequence.
        inference.reset();

        // Feed tokens one at a time, collecting CE at each position.
        // Token i predicts token i+1, so we evaluate positions 0..len-1.
        for pos in 0..sequence.len() - 1 {
            let token = sequence[pos];
            let target = sequence[pos + 1];

            let logits = inference.decode(token)?;
            let ce = cross_entropy(&logits, target);
            all_losses.push(ce);
        }

        if (seq_idx + 1) % 50 == 0 {
            let running_ppl = perplexity_from_losses(&all_losses);
            eprintln!(
                "  [{}/{}] running PPL: {:.2} ({} tokens)",
                seq_idx + 1,
                num_sequences,
                running_ppl,
                all_losses.len()
            );
        }
    }

    let perplexity = perplexity_from_losses(&all_losses);
    let avg_cross_entropy = all_losses.iter().sum::<f64>() / all_losses.len() as f64;

    Ok(PerplexityResult {
        config_name: String::new(),
        perplexity,
        avg_cross_entropy,
        num_tokens_evaluated: all_losses.len(),
        num_sequences,
    })
}
```

### Step 4: CLI integration

Add `--perplexity` flag to `ironmill-bench`:

```
ironmill-bench --perplexity \
    -m tests/fixtures/qwen3-0.6b.onnx \
    --dataset tests/fixtures/quality/wikitext2-qwen3.json
```

The flag triggers the perplexity pipeline instead of/in addition to the
latency benchmark. For each optimization config (baseline, FP16, INT8,
palettize-4, polar-4), it:

1. Compiles the model with that config
2. Creates an `AneInference` instance
3. Runs `evaluate_perplexity()`
4. Records the result

### Step 5: Cross-tabulated output

Combine with existing weight fidelity in a unified report:

```
Qwen3-0.6B — Quality Report (WikiText-2, 500 sequences)
═══════════════════════════════════════════════════════════════════
Optimization       SNR (dB)   Perplexity   Δ PPL     Size     Status
─────────────────  ─────────  ──────────   ───────   ──────   ──────
FP32 baseline      ∞          12.41        —         2.2 GB   —
FP16               78.3       12.41        +0.0%     1.1 GB   ✓ SAFE
INT8               44.7       12.56        +1.2%     578 MB   ✓ SAFE
Palettize 4-bit    25.1       14.12        +13.8%    ~300 MB  ⚠ WARN
PolarQuant 4-bit   —          12.91        +4.0%     ~350 MB  ✓ SAFE
═══════════════════════════════════════════════════════════════════
Thresholds: SAFE <5%, WARN 5–15%, FAIL >15%
```

## Performance considerations

### Evaluation time estimate

- 500 sequences × 512 tokens = **256,000 decode steps**
- At ~5 tok/s (current ANE throughput): **~14 hours per config**
- With 5 configs (baseline + 4 optimizations): **~70 hours total**

This is impractical for CI. Mitigations:

| Strategy | Sequences | Tokens | Time/config | Accuracy |
|----------|-----------|--------|-------------|----------|
| Full eval | 500 | 256K | ~14h | Gold standard |
| Reduced | 50 | 25.6K | ~1.4h | Good (±0.3 PPL) |
| Smoke test | 10 | 5.1K | ~17min | Rough (±1.0 PPL) |
| CI gate | 5 | 2.5K | ~8min | Directional only |

**Recommendation**: Default to 50 sequences. Use `--perplexity-sequences N`
to override. CI uses 5 sequences as a regression smoke test.

### Optimization: sliding window

Instead of resetting KV cache per sequence, use a **sliding window**
approach with stride < seq_len. This amortizes prefill cost and is the
standard approach in `lm-eval-harness`. A stride of 256 with seq_len 512
means each token has at least 256 tokens of context.

### Optimization: prefill batching

If `AneInference` supports batched prefill (feeding multiple prompt tokens
simultaneously rather than one-at-a-time decode), use it for the context
window. Only the final prediction at each position needs CE computed. This
could reduce evaluation time by 10–50× depending on prefill throughput.

## Validation

### Sanity checks before trusting results

1. **Baseline PPL in expected range**: Qwen3-0.6B FP32 on WikiText-2 should
   be ~12–13. If it's >20 or <8, something is wrong (bad tokenizer, broken
   attention, wrong dataset).

2. **FP16 ≈ FP32**: FP16 perplexity should be within 0.1% of FP32. If it's
   not, the FP16 pass has a bug.

3. **Monotonic degradation**: Generally, PPL should increase as quantization
   gets more aggressive: FP32 < FP16 ≤ INT8 < 4-bit. If INT8 is worse than
   4-bit, something is wrong with one of the passes.

4. **Determinism**: Same config + same dataset = same PPL. Run twice to
   confirm (no temperature sampling — greedy logit extraction only).

### Cross-validation with external tools

To validate ironmill's perplexity numbers:

1. Quantize Qwen3-0.6B with ironmill (e.g., INT8)
2. Export the quantized weights
3. Load into `llama.cpp` or HuggingFace
4. Run `lm-eval-harness` perplexity on the same WikiText-2 subset
5. Compare: results should be within ±0.5 PPL

## Related documents

- [TurboQuant E2E Inference](turboquant-e2e-inference.md) — current ANE
  inference status (attention not yet real)
- [ANE Inference Optimizations](ane-inference-optimizations.md) — throughput
  improvement roadmap
- [Benchmark Results](../BENCHMARK_RESULTS.md) — current weight-level SNR
  numbers

## Future: Classification Accuracy

**Models**: MobileNetV2 (ImageNet top-1/top-5), DistilBERT (SST-2 sentiment)

**Method**:
- Load pre-processed labeled inputs as binary tensor fixtures
- Run inference on each sample through the compiled CoreML model
- Compare predicted label against ground truth
- Report top-1 accuracy (and top-5 for ImageNet)

**Fixture format**:
```
tests/fixtures/quality/imagenet-500/
  inputs.bin     # 500 × [1, 3, 224, 224] float32 tensors (preprocessed)
  labels.json    # [281, 153, ...] ground truth class indices
  metadata.json  # { "model": "MobileNetV2", "task": "classification", "classes": 1000 }

tests/fixtures/quality/sst2-200/
  inputs.bin     # 200 × [1, 128] int32 tensors (tokenized)
  labels.json    # [0, 1, 1, 0, ...] ground truth labels
  metadata.json  # { "model": "DistilBERT", "task": "sentiment", "classes": 2 }
```

**Expected report**:
```
MobileNetV2 — Quality Impact
═══════════════════════════════════════════════════════════════════
Optimization       SNR (dB)   Top-1 Acc    Δ Acc     Size     Status
─────────────────  ─────────  ──────────   ───────   ──────   ──────
FP32 baseline      ∞          71.8%        —         16 MB    —
FP16               73.6       71.8%        +0.0%     8 MB     ✓ SAFE
INT8               42.7       71.2%        -0.8%     4 MB     ✓ SAFE
Palettize 4-bit    18.8       68.3%        -4.9%     ~2 MB    ⚠ WARN
═══════════════════════════════════════════════════════════════════
Thresholds: SAFE <2% drop, WARN 2–5%, FAIL >5%
```

**Challenges**:
- ImageNet fixtures cannot be redistributed (registration required) — download
  on demand via `scripts/download-fixtures.sh`, gitignored
- MobileNetV2 requires specific image preprocessing (resize, center crop,
  normalize with ImageNet mean/std)
- DistilBERT needs the same pre-tokenization approach as perplexity

**Priority**: After perplexity is validated. Classification accuracy is faster
to compute (~minutes, not hours) and easier to validate against published
baselines.

## Future: Word Error Rate (ASR)

**Models**: Whisper-tiny, Whisper-small (stretch)

**Method**:
- Load 50 pre-computed log-mel spectrograms from LibriSpeech test-clean
- Run encoder through CoreML to get audio features
- Greedy-decode output tokens to text
- Compute WER via Levenshtein distance on word sequences

**Challenges**:
- Whisper uses an encoder-decoder architecture — the encoder output feeds into
  autoregressive decoding, which is a separate inference path from LLM decode
- Audio preprocessing (80-bin log-mel spectrogram, 30s windows) is complex to
  implement in Rust — pre-computed spectrograms as fixtures avoids this
- WER computation: use the `strsim` crate for edit distance

**Recommended approach**: Implement in two phases:
1. Encoder-only quality (compare encoder output tensors before/after
   quantization — fast, no decoder needed)
2. Full pipeline WER with pre-computed mel spectrograms and a minimal greedy
   decoder

**Priority**: Lowest. Whisper encoder benchmarks already exist for latency;
WER adds correctness validation but requires the most new infrastructure.
