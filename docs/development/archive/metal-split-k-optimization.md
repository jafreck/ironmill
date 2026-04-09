# Metal Split-K Dispatch: Closing the MLX Gap

## Problem

Qwen 3.5-4B INT4 gs=128 decode on M2 Max 64 GB:

| Framework | Decode | BW Util | Key Technique |
|-----------|--------|---------|---------------|
| **ironmill** | **44 tok/s** | **~14%** | Separate arrays, 8 rows/TG, full-K per TG |
| MLX | 97 tok/s | ~50% | Separate arrays, split-K=8, 16 vals/thread |
| llama.cpp | 67 tok/s | ~35% | — |
| Theoretical | 152 tok/s | 100% | 400 GB/s / 1.02 GB per token |

After Phases B–D from the previous optimization round (see
`archive/metal-bandwidth-optimization.md`), ironmill moved from 38 → 44
tok/s. The remaining **2.2× gap to MLX** is dominated by two changes:

1. **Split-K dispatch** — MLX launches 8× more TGs by splitting the K
   dimension, achieving much better GPU occupancy and memory-latency hiding.
2. **Double values per thread** (Phase A) — MLX processes 16 elements per
   lane per iteration vs ironmill's 8, halving loop overhead.

This document covers the split-K implementation plan. Phase A (16
vals/thread) was previously attempted and reverted — it should be
re-attempted after split-K is in place.

## Root Cause: Low GPU Occupancy

### Current dispatch (full-K per TG)

```
Projection: N=2560, K=2560, INT4 gs=128

TGs launched:   ceil(2560 / 8) = 320
Threads/TG:     64 (2 simdgroups × 32 lanes)
Total threads:  20,480

M2 Max GPU:     ~40 compute units, each can run ≥2 TGs concurrently
Active TGs:     ~80 (hardware limit)
K loop iters:   320 per TG (K/8 words, stride 32)

Problem: 320 TGs with only ~80 active = 75% of TGs queued.
Each active TG must process ALL of K before yielding, creating
long-latency memory stalls with no other work to fill the gap.
```

### MLX dispatch (split-K=8)

```
Same projection: N=2560, K=2560

split_k = 8
TGs launched:   ceil(2560 / rows_per_tg) × 8 = 2560
K per TG:       2560 / 8 = 320 elements (40 words)
K loop iters:   ~5 per TG (40/8, strided by lanes)

Effect: 8× more TGs → GPU can schedule work to hide memory latency.
Each TG finishes quickly (5 iterations vs 320), allowing the scheduler
to overlap memory requests across many more concurrent TGs.
```

### Why this matters for bandwidth

The M2 Max has 400 GB/s peak bandwidth, but achieving it requires enough
concurrent memory requests to saturate the memory controller. With 320
TGs and long K loops, the GPU can't issue enough outstanding requests.
With 2560 TGs and short loops, the memory pipeline stays full.

---

## Implementation Plan

### Overview

Split-K requires two dispatches per projection:
1. **Partial matvec** — each TG computes a partial dot product over K/S
   elements, writing float32 partial sums to a scratch buffer.
2. **Reduction** — sum S partial results per output row, write final half
   result.

### Phase 1: Split-K for `superblock_matvec_int4` (Target: ~65 tok/s)

The core decode kernel. This single change should deliver the majority
of the occupancy benefit.

#### 1a. New shader: `split_k_matvec_int4`

Modify `superblock_matvec_int4` to accept K-slice parameters:

```metal
kernel void split_k_matvec_int4(
    device const half *A            [[buffer(0)]],   // [1, K]
    device const uchar *W           [[buffer(1)]],   // [N, K/2] contiguous
    device float *C_partial         [[buffer(2)]],   // [N, split_k] float32
    constant uint &N                [[buffer(3)]],
    constant uint &K                [[buffer(4)]],
    device const half *awq_scales   [[buffer(5)]],
    constant uint &has_awq          [[buffer(6)]],
    device const half *W_scales     [[buffer(7)]],   // [N, K/GS]
    device const half *W_zeros      [[buffer(8)]],   // [N, K/GS]
    constant uint &split_k          [[buffer(9)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint row_block = tgid / split_k;
    uint k_split   = tgid % split_k;
    uint base_row  = row_block * SB_ROWS_PER_TG + sgid * SB_ROWS_PER_SG;

    uint k_start = k_split * (K / split_k);
    uint k_end   = (k_split + 1) * (K / split_k);

    // Word range for this K-slice
    uint w_start = k_start / 8;
    uint w_end   = k_end / 8;

    float result[SB_ROWS_PER_SG] = {0};

    for (uint w = w_start + lane; w < w_end; w += 32) {
        // ... same inner loop as superblock_matvec_int4 ...
        // (pre-scaled input, 8 nibbles per word, scale * accum + bias * x_sum)
    }

    for (uint r = 0; r < SB_ROWS_PER_SG; r++) {
        result[r] = simd_sum(result[r]);
        if (lane == 0 && base_row + r < N) {
            // Write partial sum (float32, not half)
            C_partial[(base_row + r) * split_k + k_split] = result[r];
        }
    }
}
```

Key differences from the non-split kernel:
- Output is `float *C_partial` (not `half *C`) — avoids precision loss
  from intermediate half conversion.
- TG index encodes both row block and K-slice: `tgid / split_k`, `tgid % split_k`.
- K loop is bounded to `[k_start, k_end)` instead of `[0, K)`.
- Each TG writes one float per row to `C_partial[row * split_k + k_split]`.

#### 1b. New shader: `split_k_reduce`

```metal
kernel void split_k_reduce(
    device const float *C_partial   [[buffer(0)]],   // [N, split_k]
    device half *C                  [[buffer(1)]],   // [N]
    constant uint &N                [[buffer(2)]],
    constant uint &split_k          [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;

    float sum = 0.0f;
    for (uint s = 0; s < split_k; s++) {
        sum += C_partial[tid * split_k + s];
    }
    C[tid] = half(sum);
}
```

Dispatch: `(ceil(N/256), 1, 1)` TGs × `(256, 1, 1)` threads. Trivially
memory-bound, runs in <0.01ms.

#### 1c. Rust dispatch changes (`projection.rs`)

```rust
// Split-K decode path
let split_k: u32 = select_split_k(k);  // 8 for K ≤ 8192, 4 for K > 8192
let partial_buf = scratch.partial_sum_buffer(n, split_k);

// Dispatch 1: partial matvec
encoder.set_pipeline(pipeline_split_k);
// ... bind A, W, C_partial, N, K, awq, scales, zeros, split_k ...
let tg_count = n.div_ceil(8) * split_k as usize;
encoder.dispatch_threadgroups((tg_count, 1, 1), (64, 1, 1));

// Dispatch 2: reduction
encoder.set_pipeline(pipeline_reduce);
// ... bind C_partial, C, N, split_k ...
encoder.dispatch_threadgroups((n.div_ceil(256), 1, 1), (256, 1, 1));
```

#### 1d. Scratch buffer management

One scratch buffer per inference step, sized for the largest projection:

```
N_max = max(N across all projections) = typically 2560 or 4096
split_k_max = 8
scratch_size = N_max * split_k_max * 4 bytes = 82 KB (trivial)
```

Allocated once during engine init, reused for all projections within
a decode step (projections are sequential, not concurrent).

#### 1e. split_k selection heuristic

```rust
fn select_split_k(k: usize) -> u32 {
    // MLX uses split_k=8 for K ≤ 8192. We match that.
    // For very large K, fewer splits to keep per-TG work meaningful.
    if k <= 8192 { 8 } else { 4 }
}
```

### Phase 2: Split-K for all INT4 decode kernels

Extend split-K to the remaining INT4 decode kernels. Each needs a
split-K variant and the reduction dispatch.

| Kernel | File | Dispatch Helper |
|--------|------|-----------------|
| `superblock_batched_affine_matvec_int4` | `affine_batched.metal` | `encode_batched_affine_matvec_int4` |
| `superblock_gdn_batched_affine_matvec_int4` | `affine_batched.metal` | `encode_gdn_batched_affine_matvec_int4` |
| `superblock_fused_ffn_gate_up_act_int4` | `affine_fused.metal` | `encode_fused_ffn_gate_up_act_int4` |
| `superblock_affine_matvec_int4xq8` | `affine_fused.metal` | `encode_affine_matvec_int4xq8` |
| `superblock_fused_residual_norm_affine_matvec_int4` | `superblock_fused_norm.metal` | `encode_fused_residual_norm_affine_matvec_int4` |

**Note:** The fused kernels (FFN gate+up+act, residual+norm+matvec) are
more complex because they fuse activation/normalization with the matvec.
For split-K, the split only applies to the matvec portion — the
activation/norm must happen after the reduction.

This means fused split-K kernels may need to be "unfused" into:
1. Norm/residual dispatch (if applicable)
2. Split-K partial matvec
3. Reduction + activation

Whether this is a net win depends on whether the occupancy gain from
split-K exceeds the cost of losing fusion. Profile both paths.

### Phase 3: Re-attempt Phase A (16 vals/thread)

With split-K in place, each TG processes a shorter K range. Combined
with 16 vals/thread (4 × uint16 per lane), the inner loop becomes
very tight — potentially just 2-3 iterations for split_k=8, K=2560:

```
K per slice:        2560 / 8 = 320 elements
Elements per iter:  16 per lane × 32 lanes = 512
Iterations:         ceil(320 / 512) = 1 iteration per TG
```

This is the regime where Phase A + split-K compound: fewer iterations
per TG means less overhead per useful byte loaded.

**Phase A was reverted previously.** Before re-implementing, investigate
the revert cause (commit `1f88e62`).

---

## Expected Impact

| Change | Decode (tok/s) | Rationale |
|--------|---------------|-----------|
| Current baseline | 44 | Phases B+C+D |
| Phase 1: Split-K matvec_int4 | ~65 | 8× TGs → 2-3× occupancy |
| Phase 2: Split-K all decode | ~70 | Batched/fused kernels too |
| Phase 3: 16 vals/thread | ~85 | Halve iterations, compound with split-K |
| **Projected total** | **~85** | |
| MLX reference | 97 | Split-K + 16 vals + quad-group |

The remaining ~12% gap to MLX after all phases likely requires
quad-group parallelism and framework-level optimizations (lazy eval,
compute graph batching).

---

## Validation

### Correctness

Each phase must pass:
```bash
cargo check -p ironmill-inference
cargo test -p ironmill-inference --lib
```

### Numerical Equivalence

PPL must stay within ±0.05 of baseline (9.42):
```bash
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-decode-perf.toml
```

### Performance Thresholds

| Phase | Decode minimum | PPL |
|-------|---------------|-----|
| Phase 1: Split-K matvec | ≥ 55 tok/s | = baseline |
| Phase 2: Split-K all decode | ≥ 60 tok/s | = baseline |
| Phase 3: 16 vals/thread | ≥ 75 tok/s | = baseline |

---

## Files to Create/Modify

### New files
- `shaders/quantized/split_k_matvec.metal` — split-K partial matvec + reduction kernels
- `metal/ops/split_k.rs` — split-K dispatch helpers

### Modified files
- `metal/projection.rs` — route decode to split-K when enabled
- `metal/ops/mod.rs` — load split-K pipelines
- `metal/ops/quantized.rs` — split-K variants of batched/fused encode helpers
- `build.rs` — compile split-K metallib (per GS, like existing superblock)

### Existing infrastructure to reuse
- `shaders/common/simdgroup_reduce.metal` — `threadgroup_reduce_sum`
- `attention/flash_decode.metal` — prior art for split+reduce pattern

---

## Related Documents

- `archive/metal-bandwidth-optimization.md` — Previous optimization plan
  (Phases 1–4 + A–D). Contains root cause analysis, completed work, and
  occupancy sweep results.
- `decode-perf-investigation.md` — Detailed investigation brief.

## Hardware

Apple M2 Max, 64 GB unified memory, macOS 15.x.
Peak memory bandwidth: 400 GB/s. Metal GPU Family: Apple 8.

---

## Results (April 2025)

### Status: ❌ Split-K does not help on M2 Max for K ≤ 9216

Phase 1 was fully implemented and benchmarked. Two approaches tested:

| Approach | Decode | PPL | Δ vs baseline |
|----------|--------|-----|---------------|
| **Baseline** (no split-K) | **45 tok/s** | 9.42 | — |
| Two-dispatch (partial → barrier → reduce) | 44.9 tok/s | 9.42 | ≈ 0% |
| Wider TG (512 threads, intra-TG reduce) | 39.5 tok/s | 9.42 | **−14%** |

### Why split-K doesn't help

The original analysis assumed the M2 Max was occupancy-limited with 320 TGs.
In practice, the M2 Max has 38 CUs × 16 TGs/CU = **608 concurrent TGs**. With
only 320 TGs for a typical projection (N=2560), **all TGs are already active
simultaneously** — there is no queuing.

Split-K adds more TGs but the GPU is already saturated:
- **Two-dispatch:** The barrier + extra dispatch adds ~10-30 μs overhead per
  projection, negating any marginal occupancy benefit. Net: zero change.
- **Wider TGs:** 512 threads/TG limits concurrent TGs to 2 per CU (76 total),
  reducing inter-TG parallelism. Net: 14% regression.

### Corrected root cause analysis

The 2.2× gap to MLX is NOT primarily occupancy. The actual bottleneck
breakdown:

1. **Lazy evaluation / dispatch overhead** — MLX batches entire compute graphs
   before committing, amortizing per-dispatch overhead across many operations.
   ironmill dispatches each projection individually (5-7 per layer × 36
   layers = 180-250 dispatches per token). Each dispatch has ~5-20 μs
   scheduling overhead. See `metal-lazy-eval-optimization.md`.
2. **16 vals/thread (Phase A)** — MLX processes 16 elements per lane per
   iteration vs ironmill's 8, halving loop overhead and doubling memory
   throughput per instruction.
3. **Quad-group parallelism** — MLX uses quad-group operations for additional
   SIMD-level parallelism.

### What was committed

Split-K infrastructure committed but disabled (`select_split_k` returns 1):
- `shaders/quantized/split_k_matvec.metal` — wider-TG kernel with intra-TG
  reduction
- `ops/quantized.rs` — `SplitKPipelines` struct
- `ops/mod.rs` — pipeline compilation
- `projection.rs` — dispatch routing (gated by `select_split_k`)

Infrastructure may become relevant for:
- Larger models with K > 16384
- GPUs with fewer compute units where 320 TGs cause genuine queuing

### Phases 2–3: Not pursued

Given Phase 1 results, extending split-K to batched/fused kernels (Phase 2) and
16 vals/thread (Phase 3) would not improve decode throughput. Effort redirected
to lazy evaluation and dispatch overhead reduction.
