# Quantization Quality Roadmap

## Current State

Qwen3.5-4B on M2 Max 64 GB:

| Config | PPL | ΔPPL vs FP16 | GPU MB |
|--------|-----|-------------|--------|
| FP16 baseline | 8.50 | — | 9,543 |
| FP16 + TQ-INT8 KV | 8.51 | +0.1% | 9,432 |
| INT4 naive | 9.39 | +10.5% | 2,898 |
| INT4 + AWQ (current) | 9.22 | +8.5% | 2,900 |
| INT4 + AWQ (target) | ~8.6 | +1-2% | ~2,900 |

The 70% memory reduction and 22× decode speedup are solid. The
quantization quality gap (+8.5% PPL) is 3-4× worse than state of the
art (+1-2%). This document outlines the improvements needed to close
that gap.

## P0: AWQ Alpha Grid Search (target: +4-5% PPL)

**Impact:** Largest single improvement. Typically halves the quality gap.

**Current:** Alpha is hardcoded at 0.5 for all layers and projections.

**Fix:** For each layer, search alpha ∈ {0.0, 0.05, 0.1, ..., 1.0} and
pick the value that minimizes the L2 reconstruction error:

```
error(alpha) = ||W·X - Q(W·diag(s^alpha))·diag(s^-alpha)·X||²
```

where `X` is the calibration activation matrix, `W` is the FP16 weight,
and `Q()` is the INT4 quantize-dequantize function.

**Implementation:**

1. During calibration, store the raw activation matrix `X` (already done —
   `awq_activations.json` has per-projection snapshots).
2. In `quantize_tensor_int4_awq()`, loop over alpha candidates:
   - Compute `s = (mag / max_mag)^alpha`
   - Scale weights: `W_scaled = W * diag(s)`
   - Quantize + dequantize: `W_q = dequant(quant(W_scaled))`
   - Compute error: `||W_q · diag(s⁻¹) · X - W · X||²`
   - Pick alpha with minimum error
3. This is CPU-only and adds ~5-10s to JIT quantization (negligible
   vs the current 30s load time).

**Files:** `crates/ironmill-compile/src/weights/quantized.rs`

## P1: GPTQ Second-Order Rounding (target: +1-3% PPL)

**Impact:** The biggest remaining gap after alpha search. This is what
separates "good" from "great" INT4 quantization.

**Current:** Round-to-nearest quantization after AWQ scaling.

**Fix:** Implement the GPTQ algorithm (Frantar et al., 2022):

1. Compute the Hessian approximation `H = 2 · X^T · X` from calibration
   activations (X is already captured).
2. For each row of the weight matrix:
   a. Cholesky-decompose the row's Hessian block
   b. Process columns in order of decreasing diagonal (most important first)
   c. Quantize the current column
   d. Distribute the rounding error across remaining columns:
      `W[:, j+1:] -= error * H[j, j+1:] / H[j, j]`
3. This produces INT4 weights that minimize the layer output error
   in a second-order sense (not just per-element rounding).

**Implementation:**

- Add `gptq_quantize_tensor()` to `quantized.rs` alongside
  `quantize_tensor_int4_awq()`
- Requires: Cholesky decomposition (use `nalgebra` crate or implement
  the 3-line blocked Cholesky from the GPTQ paper)
- Memory: ~O(K²) per row group for the Hessian block (K=hidden_size=2560,
  so ~25 MB — fits easily in CPU memory)
- Time: ~10-30s for Qwen3.5-4B on CPU (one Cholesky per row group per layer)

**Files:** `crates/ironmill-compile/src/weights/quantized.rs`

**Reference:** [GPTQ paper](https://arxiv.org/abs/2210.17323)

## P2: Mixed-Precision Sensitive Layers (target: +0.5-1% PPL)

**Impact:** Small but consistent improvement on top of GPTQ/AWQ.

**Current:** All layers quantized to INT4 uniformly.

**Fix:** Keep the most sensitive layers at higher precision (INT8 or FP16):

1. **First and last 1-2 layers:** These handle the embedding→hidden and
   hidden→logit transformations and are consistently the most sensitive
   to quantization in all LLM architectures.
2. **Embedding and lm_head:** The embedding is already INT4 (adds +0.05
   PPL). Keeping it at FP16 or INT8 would recover some quality.
3. **Sensitivity analysis:** During calibration, measure per-layer
   reconstruction error at INT4 vs INT8. Layers with >2× average error
   should stay at INT8.

**Implementation:**

- Add `sensitive_layers: Vec<usize>` to `AffineQuantConfig`
- In `QuantizedWeightProvider::tensor()`, check if the layer is sensitive
  and use INT8 instead of INT4
- The runtime already supports mixed INT4/INT8 (affine_matvec_int8 kernel
  exists)

**Files:** `crates/ironmill-compile/src/weights/quantized.rs`,
`crates/ironmill-bench/src/config.rs`

## P3: TQ-INT8 KV Cache Quality at Long Context

**Impact:** Minimal at short context (<0.1% PPL). Potentially measurable
at 32K+ context if accumulation errors compound.

**Current state:** TQ-INT8 adds +0.1% PPL at 2K context. Not measured at
longer contexts.

**Potential improvements:**

1. **Per-head adaptive codebook:** The current codebook is shared across
   all KV heads. Learning per-head codebooks from calibration data could
   reduce quantization error for heads with atypical value distributions.

2. **Outlier-aware quantization (already implemented):** The
   `turboquant_outlier_attention` kernel separates outlier and non-outlier
   channels with independent codebooks. Ensure this path is enabled for
   models where it helps.

3. **FP8 KV cache:** Apple M3+ GPUs have hardware FP8 support. An FP8 KV
   cache would give 2× compression (same as INT8) with better dynamic
   range and no codebook overhead. This requires:
   - New `fp8_cache_write` and `fp8_attention` kernels
   - Metal 3.1+ feature detection
   - Benchmark to verify the 2× bandwidth improvement materializes

**Files:** `crates/ironmill-inference/src/metal/shaders/turboquant.metal`,
`crates/ironmill-inference/src/metal/turboquant/`

## Priority Order

| Phase | Expected PPL | Effort | Dependencies |
|-------|-------------|--------|--------------|
| P0: Alpha grid search | +4-5% | 1 day | Calibration data (done) |
| P1: GPTQ rounding | +1-3% | 2-3 days | Hessian from P0 activations |
| P2: Mixed-precision | +0.5-1% | 0.5 day | P0 or P1 |
| P3: TQ-INT8 long ctx | +0.05% | 2-3 days | Long-context PPL benchmark |

P0 alone should bring INT4+AWQ from +8.5% to ~+4-5% PPL. Combined with
P1, the target is +1-3% — competitive with llama.cpp's Q4_K_M quality.

## Validation

After each phase, verify:
1. PPL on wikitext2-qwen35 (50 sequences, stride 512)
2. Decode throughput unchanged (AWQ/GPTQ are quantization-time only)
3. GPU memory unchanged (same INT4 format, just better-chosen values)

See [docs/benchmarks/qwen35-reproduction.md](../benchmarks/qwen35-reproduction.md)
for exact commands, config files, and baseline numbers to compare against.
The AWQ-INT4 + TQ-INT8 benchmark (Section 1) is the primary validation
target — run it before and after each phase to measure PPL delta.

```bash
# Calibrate
cargo run --release --example awq_calibrate --features metal -p ironmill-bench -- \
    models/Qwen3.5-4B tests/fixtures/quality/wikitext2-qwen35.json /tmp/awq_calib

# Benchmark (use the AWQ config from the reproduction guide)
cargo run --release -p ironmill-bench --features metal -- \
    --config /tmp/bench_awq.toml -b metal \
    -i 5 -w 2 -r 1 \
    --perplexity --perplexity-sequences 50 \
    --perplexity-dataset tests/fixtures/quality/wikitext2-qwen35.json
```
