# Qwen 3.5-4B Inference Performance Plan

## Current Baseline (INT4-TQ-INT8, M2 Max 64 GB)

| Metric | Value |
|--------|-------|
| Decode throughput | 17.8 tok/s (56.3 ms/tok) |
| PPL (WikiText-2) | 7.53 (+2.2% vs FP16) |
| GPU memory | 4,675 MB |
| Weight reads/token | 2.63 GB (INT4 packed) |
| Theoretical minimum | 6.6 ms/tok @ 400 GB/s |
| Bandwidth utilization | 47 GB/s of 400 GB/s (12%) |

The gap between measured (56.3 ms) and theoretical (6.6 ms) is **8.5×**. This
document identifies where the time goes and how to close the gap.

## Where the Time Goes

### Dispatch overhead (~40 ms)

Each decode step issues **~459 Metal compute dispatches** with a memory barrier
after each. At ~0.08–0.1 ms per dispatch/barrier cycle, this accounts for the
majority of the gap:

| Component | Dispatches |
|-----------|-----------|
| 24 GDN layers × 14 ops | 336 |
| 8 attention layers × 15 ops | 120 |
| Embedding + final norm + LM head | 3 |
| **Total** | **459** |

### Bandwidth-bound compute (~10 ms)

The actual weight data movement (2.63 GB at ~250 GB/s effective) takes only
~10 ms. The INT4 affine dequant kernel performs in-line unpacking during
matmul, adding modest ALU overhead.

### LM head projection (~3 ms)

The 248K × 2560 embedding table (tied as lm_head) is 1.27 GB at FP16,
read every token. This single projection is ~50% of the total weight
bandwidth despite being a single layer.

## Optimization Plan

### P0: Fuse GDN projections (target: −200 dispatches)

**Current:** Each GDN layer dispatches QKV, Z, A, B as 4 separate
`encode_projection` calls, each with its own memory barrier.

**Proposed:** Fuse into a single batched projection kernel that reads
`norm_out` once and writes all 4 outputs. Eliminates 3 dispatches + 3
barriers per GDN layer = **72 fewer dispatches**.

Similarly, fuse gate+up FFN projections into a single dispatch (both read
`norm_out`): saves 1 dispatch × 32 layers = **32 fewer dispatches**.

### P1: Fuse residual + norm + projection (target: −64 dispatches)

**Current:** End-of-layer is `fused_residual_rms_norm` → barrier → next
layer's Q/K/V projections. The norm output is written to `norm_out` then
immediately read by projections.

**Proposed:** A fused kernel that computes residual add + RMSNorm + starts
the first projection in a single dispatch. Eliminates the `norm_out`
intermediate write/read cycle.

### P2: Remove unnecessary memory barriers (target: −100 barriers)

**Current:** Memory barrier after every single dispatch, even between
independent operations (e.g., Q/K/V projections all read the same input
and write different outputs).

**Proposed:** Only emit barriers at true data dependency boundaries:
1. After all Q/K/V/gate projections complete (before attention/conv1d)
2. After attention (before O-proj)
3. After FFN down-proj (before residual add)

This cuts barriers from ~14/layer to ~4/layer = **~320 fewer barriers**.

### P3: Reduce GDN recurrent dispatch overhead (target: −48 dispatches)

**Current:** GDN conv1d and recurrent update are separate dispatches per
layer.

**Proposed:** Fuse conv1d + SiLU + recurrent update + output gate into a
single kernel per GDN layer. These are sequential but operate on small
buffers (qkv_dim × 1 for decode) — the dispatch overhead dominates the
actual compute.

### P4: Speculative decoding with EAGLE-3 (target: 2–3× throughput)

Orthogonal to kernel optimization. Uses a lightweight draft model to
predict multiple tokens, verified in parallel by the main model. The
benchmark harness already has `--specbundle` support.

## Expected Impact

| Optimization | Dispatches saved | Estimated speedup |
|-------------|-----------------|-------------------|
| P0: Fuse GDN + FFN projections | ~104 | −8 ms |
| P1: Fuse residual+norm+proj | ~64 | −5 ms |
| P2: Remove redundant barriers | ~320 | −15 ms |
| P3: Fuse GDN recurrent | ~48 | −4 ms |
| **Combined P0–P3** | **~536** | **~30 ms → ~26 ms/tok (38 tok/s)** |
| P4: EAGLE-3 speculation | N/A | 2–3× on top |

## Non-goals

- Further weight quantization (INT4 affine is the target precision)
- KV cache quantization changes (TQ-INT8 is fixed)
- Embedding/lm_head quantization
- CPU offloading

## Reproduce Current Baseline

```bash
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-optimization-matrix.toml -b metal \
  --baseline fp16-baseline -i 20 -w 5 -r 3 \
  --perplexity --perplexity-sequences 1 \
  --perplexity-dataset tests/fixtures/quality/wikitext2-qwen35.json
```

Hardware: Apple M2 Max, 64 GB unified memory.
