# Closing the MLX Gap: Metal Decode Optimization Plan

## Current State

Qwen3.5-4B INT4 gs=128 decode on M2 Max 64 GB:

| Framework | tok/s | ms/tok | Gap |
|-----------|-------|--------|-----|
| ironmill | 43.0 | 23.3 | baseline |
| MLX | 113.7 | 8.8 | 2.64× faster |

ironmill per-category GPU breakdown (22.2ms total):

| Category | GPU time | % |
|----------|---------|---|
| Projections (Q/K/V/O) | 10.48ms | 47.1% |
| FFN (gate+up+act+down) | 5.90ms | 26.5% |
| Attention | 4.59ms | 20.6% |
| Norms + residual | 0.35ms | 1.6% |
| LM head | 0.90ms | 4.0% |

## Gap Analysis

MLX source analysis reveals their QMV kernel (`qmv_fast_impl`) uses the
**identical architecture** to ironmill's `superblock_matvec_int4`: 64
threads, 2 simdgroups, 4 rows/SG, same pre-scale trick. The 2.64× gap
is **not in the kernel** — it's in the pipeline structure.

### What MLX does differently

1. **No GDN (linear attention) layers.** MLX's Qwen3.5-4B model uses
   standard attention for all 32 layers. ironmill uses 24 GDN + 8
   Standard. GDN involves a recurrent kernel + separate projection/core
   phases that may be less efficient than flash attention for decode.

2. **Flash-style fused attention.** MLX uses `steel_attention` (fused
   Q·K^T → softmax → ·V in one kernel with FP16 KV cache). ironmill
   uses TurboQuant INT8 KV cache with a separate attention kernel.

3. **Lazy eval command buffer batching.** MLX batches ~50 ops per
   command buffer (configurable per hardware). ironmill uses one command
   buffer per decode step. Both approaches show near-zero dispatch
   overhead within an encoder, so this is likely not a significant
   factor.

4. **No dead dispatches.** MLX has no equivalent to ironmill's wasted
   Q8 quantization dispatch on AWQ layers.

### What MLX does NOT do (that ironmill does)

- MLX does **not** fuse residual+norm (separate kernels).
- MLX does **not** fuse norm+projection (P1 fusion is ironmill-only).
- MLX does **not** fuse across layers.

ironmill's existing fusions (P1, fused_residual_norm, fused_ffn,
fused_embedding_norm) are more aggressive than MLX's. The gap must
come from the items above — primarily attention and GDN.

## Optimization Plan

### Priority 1: Eliminate dead Q8 dispatch
**Expected savings: ~0.5ms (eliminating 8 wasted dispatches + barriers)**

The Q8 input quantization runs on every Standard attention layer even
though `encode_projection_q8` ignores the Q8 data when AWQ is active
(`awq_scales.is_some()`). This wastes a dispatch + barrier per Standard
layer.

**Implementation:**
- In `pipeline.rs`, skip Q8 quantization when all projections have AWQ
  scales.
- Minimal code change: add `&& !has_awq` to the Q8 dispatch condition.

**Files:** `crates/ironmill-inference/src/metal/pipeline.rs:596-621`

### Priority 2: Profile GDN vs Standard attention cost
**Expected savings: determines whether GDN is a bottleneck**

ironmill uses 24 GDN layers + 8 Standard layers. MLX uses 32 Standard
layers with flash attention. The GDN recurrent kernel may be slower
than flash attention for single-token decode.

**Investigation:**
- Use `profile-metal` per-dispatch mode to measure per-layer GPU time
  for GDN vs Standard layers.
- Compare: GDN layer time (proj + recurrent + output) vs Standard layer
  time (proj + flash_attention + output).
- If GDN layers are significantly slower, investigate whether the
  Qwen3.5-4B model can use standard attention for all layers (the HF
  config may not require GDN).

**Files:** `crates/ironmill-inference/src/metal/pipeline.rs:510-577`,
`crates/ironmill-inference/src/metal/gdn.rs`

### Priority 3: Flash attention for decode
**Expected savings: potentially large (attention is 4.59ms = 21%)**

ironmill uses TurboQuant INT8 KV cache with a custom attention kernel.
MLX uses FP16 KV cache with fused flash attention (`steel_attention`).
The TurboQuant path involves dequantization overhead that may not pay
off for the model sizes where ironmill operates.

**Investigation:**
- Benchmark decode with FP16 KV cache (disable `kv_quant = "turbo-int8"`)
  to measure the raw attention kernel speed without TQ overhead.
- If FP16 KV attention is significantly faster, evaluate the memory
  tradeoff (2× KV cache size vs decode speed).
- Consider implementing flash-decode for the FP16 path if it doesn't
  exist.

**Files:** `crates/ironmill-inference/src/metal/shaders/attention/`,
`crates/ironmill-inference/src/metal/shaders/turboquant/`

### Priority 4: Fuse O-projection with post-attention norm
**Expected savings: ~0.3ms (eliminate attn_out → ffn_down DRAM round-trip)**

Current flow per layer:
```
attention → write attn_out → barrier → O-proj reads attn_out, writes ffn_down
  → barrier → fused_residual_norm reads ffn_down+hidden_state
```

A fused `attention_output_and_residual_norm` kernel could:
1. Read `attn_out` from the attention output
2. Compute O-projection inline
3. Add residual and compute RMSNorm
4. Write `norm_out` and `residual` directly

This eliminates the `attn_out` DRAM write (num_heads × head_dim × 2
bytes per layer = 2560×2 = 5KB, but 32 layers × 2 accesses = 320KB
total traffic eliminated).

**Prerequisite:** This requires the O-projection to be fused into the
same kernel as the residual+norm. The P1 kernel already demonstrates
this pattern (norm + projection). The new kernel would be O-proj +
residual + norm.

**Files:** `crates/ironmill-inference/src/metal/shaders/norm/superblock_fused_norm.metal`

### Priority 5: Fuse FFN down-projection with end-of-layer residual+norm
**Expected savings: ~0.3ms (eliminate ffn_down DRAM round-trip)**

Current flow:
```
fused_gate_up_act → write ffn_gate → barrier → down-proj reads ffn_gate,
  writes ffn_down → barrier → P1 fused reads ffn_down+residual
```

The `ffn_down` buffer is written by the down-projection and immediately
read by the P1 fused kernel. A fused `down_proj_residual_norm` kernel
could compute the down projection and immediately add residual + norm
without the DRAM round-trip for `ffn_down`.

This is more complex because the down-projection is a full matvec
(N=2560, K=10240) that needs threadgroup-level reduction, making fusion
with the subsequent norm non-trivial. The P1 kernel pattern could be
adapted (it already does norm + projection; this would be projection +
norm in reverse order).

**Files:** `crates/ironmill-inference/src/metal/shaders/quantized/`,
`crates/ironmill-inference/src/metal/shaders/norm/superblock_fused_norm.metal`

### Priority 6: Eliminate intermediate buffers via register passing
**Expected savings: ~1-2ms (eliminate norm_out DRAM traffic)**

The biggest remaining DRAM traffic is `norm_out`:
- Written by fused_residual_norm (or P1 kernel)
- Read by Q/K/V projections and FFN gate+up projections
- Size: 2560 × 2 = 5KB per write, but read 4-6 times per layer

The P1 fusion already eliminates one norm_out write+read for the first
projection. To eliminate more, the norm computation would need to be
fused into each projection kernel (each projection recomputes the norm
inline from `hidden_state`). This trades compute for memory bandwidth:

- Extra compute: 32 layers × 4 projections × RMSNorm(2560) ≈ negligible
- Saved memory: 32 layers × 5KB × 5 reads = 800KB saved

This requires each projection kernel to take `hidden_state` + norm
weights as input instead of `norm_out`, computing the norm inline
before the matvec. The norm is cheap (one reduction over hidden_size)
relative to the matvec.

**Files:** New shader variants needed. Significant shader complexity.

## Implementation Order

```
P1 → P2 → P3 → P4/P5 → P6
```

P1 is a trivial code fix. P2 is investigation-only (no code changes).
P3 may reveal that attention is the primary gap. P4-P6 are deeper
kernel fusion work that may not be needed if P1-P3 close the gap.

## Measurement Protocol

For each optimization, measure:
1. **Decode tok/s** (5 runs, drop outliers, report median)
2. **PPL** (must remain 9.42 ±0.01)
3. **Prefill tok/s** at 128 and 512 tokens

Use the existing benchmark infrastructure:
```bash
# Decode + PPL
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-decode-perf.toml --suite decode-ppl

# Prefill
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-decode-perf.toml --suite prefill

# MLX comparison
python3 scripts/mlx_decode_bench.py --tokens 200

# Per-category breakdown (requires profile-metal feature)
cargo run --release -p ironmill-bench --features metal,profile-metal -- \
  --config configs/qwen35-4b-decode-perf.toml --kernel-timing --suite decode
```

## Validated Non-Opportunities

From the profiling infrastructure work and 6 kernel optimization
attempts (see `metal-lazy-eval-optimization.md`):

- **Dispatch count reduction**: In-encoder overhead is near-zero on M2
  Max. Batching dispatches doesn't help.
- **Kernel architecture changes**: Both ironmill and MLX use identical
  8-row/64-thread matvec with pre-scale. No kernel-level improvement
  available.
- **Split-K / wider TGs**: GPU occupancy already saturated at K≤9216.
- **Dual-matrix fused shaders**: FFN BW advantage comes from large N,
  not dual-stream structure.
- **AMX-accelerated matvec**: 3× regression from barrier overhead.
