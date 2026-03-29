# Quality Benchmark Implementation Plan

## Problem

Ironmill's optimization pipeline applies transformations that trade precision
for speed and size — FP16 conversion, INT8 quantization, 4-bit palettization,
and PolarQuant rotation-based compression. Each of these changes model weights,
and the impact on model **output quality** is currently unmeasured.

### What Exists Today

**Weight-level fidelity** (`crates/ironmill-bench/src/quality.rs`):

- Compares original FP32 weights against quantized+reconstructed weights
- Metrics: MSE, PSNR (dB), compression ratio
- Per-tensor granularity — measures every const tensor ≥ 1024 elements
- Currently supports PolarQuant pass only
- Now wired into CLI via `ironmill-bench --quality`

This answers: "How close are the quantized weights to the originals?"

### What's Missing

**Task-level accuracy** — whether the model still produces correct outputs:

| Question | Metric | Why weight fidelity doesn't answer it |
|----------|--------|---------------------------------------|
| Does the LLM still generate coherent text? | Perplexity | Low MSE on individual weight tensors doesn't guarantee low accumulated error across 32+ transformer layers |
| Does the classifier still get the right answer? | Top-k accuracy | Error distribution matters — uniform small errors may be fine, but a few large errors in attention weights can flip predictions |
| Does the ASR model still transcribe correctly? | Word Error Rate | Quantization can shift the output distribution enough to change beam search decisions without large per-weight MSE |

The gap is real: a model can have 38 dB PSNR (weight SNR) and still show 3%
perplexity degradation, or conversely, 25 dB PSNR with negligible task impact
if the errors are in less-sensitive layers.

## Design

### Architecture

```
ironmill-bench --quality --model model.onnx
                   │
                   ▼
         ┌─────────────────┐
         │ Weight Fidelity  │ ← existing quality.rs
         │ (MSE, PSNR, CR)  │
         └────────┬─────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Task Accuracy    │ ← new: requires datasets + model-specific eval
         │ (PPL, Top-k, WER)│
         └────────┬─────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Cross-tabulated  │
         │ Quality Report   │
         └─────────────────┘
```

Weight fidelity runs on the MIL IR (fast, no datasets needed). Task accuracy
runs the compiled CoreML model on reference data (slower, needs fixtures).

### Task-Level Metrics

#### 1. Perplexity (Decoder / LLM models)

**Models**: Qwen3-0.6B, Llama-3.2-1B (future)

**Method**:
- Load a fixed dataset: 500 sequences from WikiText-2 (or similar)
- For each sequence, run the optimized model through CoreML
- Compute cross-entropy loss from the output logits vs. ground truth tokens
- Aggregate into perplexity: `exp(mean(cross_entropy_losses))`
- Compare against FP32 baseline perplexity

**Challenges**:
- CoreML logit extraction: `Model::extract_outputs()` already returns
  `Vec<OutputTensorData>` as f32, but the benchmark harness (`inference.rs`)
  currently discards prediction results — need to wire extraction into a new
  evaluation pipeline
- Tokenization: need a tokenizer to convert text → token IDs for input.
  Options: (a) pre-tokenized dataset bundled as fixture, (b) minimal BPE
  tokenizer in Rust, (c) shell out to Python tokenizer
- Autoregressive loop: each token prediction depends on prior context, so
  this must be sequential — can't batch like classification

**Recommended approach**: Pre-tokenize the dataset offline and bundle as a
fixture (JSON array of token ID sequences). This avoids runtime tokenizer
dependency and keeps the benchmark deterministic.

**Dataset fixture**: `tests/fixtures/quality/wikitext2-500.json`
```json
{
  "name": "WikiText-2 (500 samples)",
  "sequences": [
    [1234, 5678, ...],
    ...
  ]
}
```

**Download**: Add to `scripts/download-fixtures.sh`:
```bash
# Download and pre-tokenize WikiText-2 subset
python3 scripts/prepare-quality-dataset.py \
  --dataset wikitext-2 \
  --samples 500 \
  --tokenizer Qwen/Qwen3-0.6B \
  --output tests/fixtures/quality/wikitext2-500.json
```

#### 2. Top-k Accuracy (Encoder / Classification models)

**Models**: DistilBERT (SST-2), MobileNetV2 (ImageNet)

**Method**:
- Load a labeled dataset: 200 SST-2 samples (DistilBERT), 500 ImageNet-val
  samples (MobileNetV2)
- Run inference on each sample
- Compare predicted label against ground truth
- Report top-1 accuracy (and top-5 for ImageNet)

**Challenges**:
- DistilBERT needs tokenized text input — same approach as perplexity
  (pre-tokenized fixture)
- MobileNetV2 needs preprocessed images — bundle as raw float tensors or
  load from JPEG + normalize in Rust
- Label mapping: need class index → label for ImageNet

**Recommended approach**: Bundle pre-processed inputs as binary tensor files.
Each fixture is a directory:
```
tests/fixtures/quality/sst2-200/
  inputs.bin     # 200 × [1, 128] int32 tensors (tokenized)
  labels.json    # [0, 1, 1, 0, ...] ground truth labels
  metadata.json  # { "model": "DistilBERT", "task": "sentiment", "classes": 2 }

tests/fixtures/quality/imagenet-500/
  inputs.bin     # 500 × [1, 3, 224, 224] float32 tensors (preprocessed)
  labels.json    # [281, 153, ...] ground truth class indices
  metadata.json  # { "model": "MobileNetV2", "task": "classification", "classes": 1000 }
```

#### 3. Word Error Rate (ASR models)

**Models**: Whisper-tiny, Whisper-small (future)

**Method**:
- Load 50 audio samples from LibriSpeech test-clean
- Run encoder through CoreML to get audio features
- Decode output tokens to text (greedy or beam search)
- Compute WER against reference transcriptions

**Challenges**:
- Audio preprocessing: Whisper requires log-mel spectrogram (80 bins,
  30s windows). This is complex to implement in Rust.
- Decoder: Whisper uses an encoder-decoder architecture. The encoder
  output goes through autoregressive decoding.
- WER computation: standard Levenshtein distance on word sequences

**Recommended approach**: This is the most complex metric. Implement in
two phases:
1. **Phase 1**: Encoder-only timing (already benchmarked). Skip WER.
2. **Phase 2**: Full pipeline with pre-computed mel spectrograms as
   fixtures and a minimal greedy decoder.

### Quality Report Format

The output cross-references weight fidelity with task accuracy:

```
Qwen3-0.6B — Quality Impact
─────────────────────────────────────────────────────────────
Optimization       Weight PSNR  Perplexity   Δ PPL    Status
─────────────────  ──────────   ──────────   ──────   ──────
FP32 baseline      ∞            12.4         —        —
FP16               73.6 dB      12.4         +0.0%    ✓ SAFE
INT8               38.5 dB      12.8         +3.2%    ✓ SAFE
Palettize 4-bit    19.9 dB      14.1         +13.7%   ⚠ WARN
PolarQuant 4-bit   22.4 dB      12.9         +4.0%    ✓ SAFE

MobileNetV2 — Quality Impact
─────────────────────────────────────────────────────────────
Optimization       Weight PSNR  Top-1 Acc    Δ Acc    Status
─────────────────  ──────────   ──────────   ──────   ──────
FP32 baseline      ∞            71.8%        —        —
FP16               73.6 dB      71.8%        +0.0%    ✓ SAFE
INT8               38.5 dB      71.2%        -0.8%    ✓ SAFE
Palettize 4-bit    19.9 dB      68.3%        -4.9%    ⚠ WARN
```

**Thresholds** (configurable):
- Perplexity: >5% relative increase → WARN, >15% → FAIL
- Top-1 accuracy: >2% absolute drop → WARN, >5% → FAIL
- WER: >5% absolute increase → WARN, >15% → FAIL

## Implementation Order

### Phase 1 — Weight Fidelity Improvements (current PR)

- [x] Wire `quality.rs` into CLI via `--quality` flag
- [x] Add model-level quality summary (`QualitySummary`)
- [x] Add SAFE/WARN/RISK status based on worst-case PSNR
- [ ] Extend `measure_program_quality` with a pass selector to support FP16
      and INT8 (not just PolarQuant) — the comparison logic (MSE/PSNR) is
      identical; only the quantization pass and reconstruction path differ

### Phase 2 — Dataset Infrastructure & Inference Output Extraction

- [ ] Build a "real input → CoreML prediction → extract output" evaluation
      pipeline using the existing `Model::extract_outputs()` method — the
      current benchmark path (`inference.rs`) only times `predict()` and
      discards results; the new pipeline reuses `extract_outputs()` to obtain
      typed f32 tensors (logits, class probabilities) for metric computation
- [ ] Create `scripts/prepare-quality-dataset.py` for pre-tokenizing
      reference datasets
- [ ] Define fixture format for quality datasets
- [ ] Add dataset download to `scripts/download-fixtures.sh`
- [ ] Create `tests/fixtures/quality/` directory structure (`.gitignore`
      ImageNet fixtures — they are downloaded on demand, not committed)
- [ ] Smoke-test `extract_outputs()` → logit pipeline on Qwen3-0.6B with a
      single hand-crafted input to catch CoreML output-shape surprises early

### Phase 3 — Perplexity Measurement

- [ ] Add `perplexity.rs` module to `ironmill-bench`
- [ ] Implement cross-entropy computation from CoreML logit output
- [ ] Pre-tokenized dataset loading
- [ ] Autoregressive evaluation loop
- [ ] Baseline caching (FP32 perplexity computed once, reused)
- [ ] Unit tests for cross-entropy and perplexity aggregation

### Phase 4 — Classification Accuracy

- [ ] Add `accuracy.rs` module to `ironmill-bench`
- [ ] Top-1 and top-5 accuracy computation
- [ ] Support for DistilBERT (text classification) and MobileNetV2
      (image classification) fixture formats
- [ ] Pre-processed input loading from binary tensor files
- [ ] Unit tests for top-k accuracy computation

### Phase 5 — ASR Quality (stretch goal)

- [ ] Pre-computed mel spectrogram fixtures
- [ ] Minimal greedy decoder for Whisper output tokens
- [ ] WER computation using the `strsim` crate for Levenshtein distance on
      word sequences
- [ ] Integration with encoder benchmark results

### Phase 6 — Cross-Tabulated Report

- [ ] Combine weight fidelity and task accuracy in unified report
- [ ] JSON schema extension for quality metrics
- [ ] Regression tracking for quality metrics via new `QualityBaselineEntry`
      in a separate `~/.ironmill/quality-baselines/` directory — quality
      baselines are stored separately from performance baselines since they
      have different fields, comparison semantics (relative % for perplexity,
      absolute % for accuracy), and change drivers (optimization config vs.
      hardware)
- [ ] `--quality-threshold` CLI flag for configurable WARN/FAIL levels

## Dependencies

- **Pre-tokenized datasets**: Requires a one-time Python script run (HuggingFace
  `datasets` + `transformers` libraries) to prepare fixtures
- **CoreML inference**: Task accuracy runs through the compiled model, so it
  depends on the existing inference pipeline. `Model::extract_outputs()` already
  supports extracting f32 multi-array outputs; the benchmark harness needs a new
  evaluation pipeline that uses it instead of discarding prediction results
  (addressed in Phase 2).
- **No new Rust crate dependencies** for Phase 1–4 (dataset loading is just
  JSON + binary file I/O)
- **Phase 5** adds `strsim` for Levenshtein distance (WER computation) and may
  need an audio processing crate for mel spectrogram verification, but the
  recommended approach avoids the latter by pre-computing spectrograms

## Decisions

1. **Quality benchmarks do not block CI.** Report-only by default. Use
   `--quality-fail-on-regression` to opt in to CI gating.

2. **Dataset licensing determines fixture strategy.** Open-licensed datasets
   (WikiText-2 CC-BY-SA, SST-2, LibriSpeech CC-BY-4.0) are committed to the
   repo as pre-processed fixtures. ImageNet requires registration and cannot
   be redistributed — ImageNet fixtures are downloaded on demand via
   `scripts/download-fixtures.sh`, stored under `tests/fixtures/quality/imagenet-500/`,
   and excluded from version control via `.gitignore`.

3. **FP16/INT8 weight fidelity extends `measure_program_quality`.** The
   comparison logic (MSE/PSNR) is identical across quantization passes; only
   the pass applied and reconstruction path differ. A pass selector parameter
   avoids duplicating the measurement pipeline.

4. **Quality baselines are separate from performance baselines.** Stored under
   `~/.ironmill/quality-baselines/` with a `QualityBaselineEntry` schema
   distinct from `BaselineEntry`. Performance baselines change with hardware;
   quality baselines change with optimization config. Different lifecycle,
   different fields, different comparison semantics (relative % for perplexity,
   absolute % for accuracy).

## Open Questions

~~1. **Baseline management**: Should quality baselines be separate from
   performance baselines, or combined?~~ **Resolved** — separate. See Decision 4.
