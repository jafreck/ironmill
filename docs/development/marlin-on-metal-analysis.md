# Marlin on Metal: Feasibility Analysis

**Context:** [Marlin](https://arxiv.org/abs/2408.11743) (Mixed-precision
Auto-Regressive LINear kernels) is a CUDA kernel for FP16×INT4 matrix
multiplication that achieves near-ideal 4× quantization speedup at batch sizes
up to 16–32 on NVIDIA GPUs. This document evaluates which of its techniques
can be ported to Apple Metal for ironmill's GPU backend.

> **Reference:** Frantar et al., "MARLIN: Mixed-Precision Auto-Regressive
> Parallel Inference on Large Language Models", arXiv 2408.11743 (2024).

---

## Marlin's Core Techniques

Marlin sustains close to theoretical peak bandwidth utilization by combining:

1. **Asynchronous global→shared memory pipeline** — `cp.async` loads data from
   global memory directly into shared memory without going through registers,
   with 4+ pipeline stages overlapping loads and tensor core compute.
2. **Tensor Core matmul** — WMMA/`mma` instructions for 16×16 FP16 tiles,
   consuming dequantized INT4 weights.
3. **Weight pre-shuffling** — offline layout transformation so INT4 data maps
   directly to `mma` fragment registers with no runtime shuffling.
4. **L2 cache eviction hints** — `cp.async.cg` (cache-global) policy for
   weights so they don't evict activations from L2; activations always fetched
   from L2 and reused in registers.
5. **Multiple warps per output tile** — instead of 1 warp = 1 tile, multiple
   warps collaborate on the same tile, boosting occupancy and latency hiding.
6. **Column-major dispatch** — threadblocks are assigned to output tiles in
   column-major order to maximize L2 activation reuse across blocks.

---

## Technique-by-Technique Metal Mapping

### Fully Portable

These techniques translate directly to Metal with no fundamental hardware gap.

| Marlin Technique | Metal Equivalent | Notes |
|---|---|---|
| Weight pre-shuffling | Offline layout transformation | Already using blocked `[N/8, K/8, 8, 8]` layout for dense matvec. Can extend to `simdgroup_matrix`-optimal layouts for INT4 |
| Multiple warps per tile | Multiple simdgroups per threadgroup | Currently using 8 simdgroups (256 threads/TG) in quantized kernels. Can tune collaboration pattern so simdgroups share an output tile |
| Register-level activation reuse | Identical semantics | Metal registers work the same way — load activations once, reuse across weight tiles |
| Conflict-free shared memory | Threadgroup memory banking | Same bank-conflict avoidance via padding/swizzle applies to Metal's threadgroup memory |
| Column-major dispatch | Threadgroup grid ordering | Dispatch threadgroups in column-major order for L2 activation locality. Controlled from the CPU side via `dispatchThreadgroups` |
| Fused dequant + matmul | Already implemented | `polarquant_*` and `affine_*` kernels already dequantize per-tile and accumulate inline |

### Partially Portable

These techniques have Metal equivalents that are less capable than their CUDA
counterparts.

#### `simdgroup_matrix_multiply_accumulate`

Metal's hardware matrix-multiply instruction (available M1+) is the equivalent
of CUDA tensor cores, but with important differences:

- **Tile size:** 8×8 on Metal vs 16×16 on CUDA. Four 8×8 tiles fit one 16×16
  tile, so the instruction must be issued 4× more often for the same work.
- **Input types:** FP16 and BF16 only — no native INT4 or INT8 input. Must
  dequantize INT4→FP16 before feeding to the hardware.
- **Current state in ironmill:** Not used at all. Quantized matmul kernels use
  scalar dot-product accumulation. **This is the single biggest optimization
  opportunity.**

#### Double-Buffered Pipeline

Manual double-buffering (ping-pong between two threadgroup memory buffers) is
straightforward in Metal:

```
// Pseudocode: double-buffered load/compute
load tile[0] into tg_buf[0]
threadgroup_barrier()
for each remaining tile t:
    load tile[t+1] into tg_buf[(t+1)%2]   // load next
    compute on tg_buf[t%2]                  // compute current
    threadgroup_barrier()
compute on tg_buf[last%2]
```

Without `cp.async`, the load still goes through registers (load→store→barrier
vs CUDA's direct DMA), so the overlap window is narrower. Still a clear win
over single-buffered.

#### Multi-Stage Pipeline (4+ stages)

Beyond double-buffering, Marlin uses 4+ pipeline stages. On Metal, going
beyond 2 stages has diminishing returns because:

- Each extra buffer doubles threadgroup memory pressure.
- Without hardware async, extra stages don't hide latency proportionally.
- Apple GPU hardware already interleaves many threadgroups to hide latency
  (occupancy-based latency hiding vs explicit pipelining).

**Recommendation:** Implement double-buffering (2 stages). Consider 3 stages
only if profiling shows threadgroup memory isn't the bottleneck.

### Not Portable

These techniques have no Metal equivalent and represent the fundamental
performance ceiling gap vs CUDA.

#### `cp.async` (Hardware Async Global→Shared Load)

This is Marlin's keystone technique. On CUDA, `cp.async` initiates a DMA
transfer from global to shared memory that proceeds independently of the
warp — the warp can immediately begin computing on previously loaded data.

**Metal has no equivalent.** Threadgroup memory loads require:
1. Thread reads from device memory into a register.
2. Thread writes from register into threadgroup memory.
3. All threads hit `threadgroup_barrier(mem_flags::mem_threadgroup)`.

This means load and compute cannot truly overlap within a threadgroup — they're
serialized around barriers. Apple GPUs compensate somewhat via high occupancy
(many threadgroups in flight hide each other's stalls), but the per-threadgroup
efficiency is lower.

**Impact:** ~10–15% throughput loss vs CUDA at the same batch size, growing
with batch size as compute density increases and latency hiding matters more.

#### L2 Cache Eviction Hints

CUDA's `cp.async.ca` (cache-all) vs `cp.async.cg` (cache-global) lets Marlin
tell the L2: "don't cache these weights — they're streamed once." This keeps
the L2 warm for activations that are reused across tiles.

**Metal has no cache hint instructions.** Weight data competes with activations
for L2 residency. Column-major dispatch ordering partially compensates (it
improves activation locality), but weights may still thrash the cache.

**Impact:** ~5% throughput loss, mostly visible at larger batch sizes where
activation reuse across tiles is critical.

#### Native INT4 Tensor Core Input

NVIDIA tensor cores (Ampere+) can directly consume INT4 data in certain
configurations. Metal's `simdgroup_matrix` requires FP16 input — you must
unpack 2 nibbles, subtract zero-point, multiply by scale, and cast to FP16
before the matrix instruction.

**Impact:** Extra register pressure and ALU work for dequantization. Partially
offset by Apple GPUs having generous register files.

---

## Expected Performance by Batch Size

| Batch Size | Marlin (CUDA, A100) | Projected Metal (M-series) | Gap Source |
|---|---|---|---|
| 1 (matvec) | ~4× theoretical | ~4× theoretical | Both bandwidth-bound; Metal may even win (no PCIe, unified memory) |
| 2–8 | ~3.8–3.9× | ~3.2–3.5× | No `cp.async`; 8×8 vs 16×16 tiles |
| 16 | ~3.7× | ~2.8–3.2× | Latency hiding gap widens |
| 32 | ~3.3× | ~2.2–2.8× | L2 thrashing without eviction hints |
| 64+ | ~2.5× | ~1.8–2.2× | Compute-bound regime; Metal has fewer FLOPS and no async pipeline |

> These are estimates based on architectural differences. Actual numbers depend
> on model dimensions, Apple Silicon generation, and kernel tuning.

---

## Actionable Improvements for ironmill

Ordered by expected impact, applicable to `affine_matmul_int4`,
`polarquant_matmul_int4`, and the dense `matvec` kernels.

### 1. Adopt `simdgroup_matrix_multiply_accumulate` (High Impact)

The current quantized matmul kernels use scalar dot-product accumulation in
threadgroup memory. Switching to hardware matrix-multiply would roughly 2×
throughput for batch>1.

**Approach:**
- Dequantize INT4→FP16 into registers (already done).
- Organize dequantized data into `simdgroup_matrix<half, 8, 8>` tiles.
- Call `simdgroup_matrix_multiply_accumulate` for the 8×8 FP16 matmul.
- Accumulate across K-dimension tiles in registers (FP32 accumulator).

**Tile geometry:** A reasonable starting point is 64×64 output tiles, each
computed by 8 simdgroups, with each simdgroup responsible for one 8×8 sub-tile,
iterating over K in steps of 8.

### 2. Double-Buffer Threadgroup Memory (Medium Impact)

Add ping-pong buffering so the next tile's weight data loads while the current
tile is being computed. Even without `cp.async`, this hides some of the load
latency behind barrier waits.

### 3. Optimize Weight Layout for simdgroup Access (Medium Impact)

Pre-shuffle INT4 packed data offline so that after unpacking, the FP16 values
land in the exact register layout expected by `simdgroup_matrix`. This avoids
runtime register shuffles and shared memory bank conflicts.

### 4. Column-Major Dispatch Ordering (Low–Medium Impact)

Dispatch threadgroups in column-major order (iterate over N first, then M) so
adjacent threadgroups share the same activation columns in L2. This is a
CPU-side change only — no kernel modifications needed.

### 5. Tune Threadgroup Dimensions (Low Impact)

Current kernels use 256 threads (8 simdgroups). Profile whether 512 threads
(16 simdgroups) improves occupancy and latency hiding on target hardware.
Trade-off: more threads = more threadgroup memory pressure = fewer concurrent
threadgroups.

---

## Apple Silicon Architectural Advantages

Not everything favors CUDA. Metal has structural advantages that partially
offset the missing techniques:

- **Unified memory** — no PCIe transfers, no staging buffers, no
  host↔device synchronization overhead. Weights are already in GPU-visible
  memory.
- **High occupancy latency hiding** — Apple GPUs schedule many threadgroups
  concurrently to hide memory latency, partially compensating for the lack of
  `cp.async`.
- **Lower dispatch overhead** — Metal command encoding is lightweight compared
  to CUDA kernel launches, which matters for the many small dispatches in a
  transformer layer.
- **Bandwidth/FLOP ratio** — Apple Silicon has a higher bytes-per-FLOP ratio
  than datacenter GPUs, meaning memory-bound kernels (which quantized matmul
  is at low batch sizes) are relatively faster.

---

## Summary

| | Portable | Partial | Not Portable |
|---|---|---|---|
| **Techniques** | Weight pre-shuffling, multi-simdgroup tiles, register reuse, bank-free TG memory, column-major dispatch, fused dequant | `simdgroup_matrix` (8×8 only, FP16 only), double-buffering (no hardware async), 2-stage pipeline | `cp.async`, L2 eviction hints, native INT4 tensor input |
| **Practical ceiling** | Batch=1: parity with Marlin | Batch=2–16: ~80% of Marlin | Batch=32+: ~60–70% of Marlin |

The highest-ROI change is adopting `simdgroup_matrix_multiply_accumulate` in
the quantized matmul kernels — it's the gap between ironmill's current scalar
accumulation and what the hardware can actually do, independent of any
CUDA-specific techniques.
