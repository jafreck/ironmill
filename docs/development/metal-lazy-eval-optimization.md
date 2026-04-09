# Metal Decode Performance: Profiling-First Optimization

## Problem

Qwen 3.5-4B INT4 gs=128 decode on M2 Max 64 GB:

| Framework | Decode | Dispatch Model |
|-----------|--------|----------------|
| **ironmill** | **45 tok/s** (22 ms/tok) | Eager: individual kernel dispatches |
| MLX | 97 tok/s (10.3 ms/tok) | Lazy: graph-batched, single eval() |
| llama.cpp | 67 tok/s (14.9 ms/tok) | Compute graph with op batching |

ironmill is **2.16× behind MLX** on identical quantization (INT4 gs=128).

### What we know

- **Split-K does not help.** GPU occupancy is already saturated on M2 Max
  for K ≤ 9216 (see `archive/metal-split-k-optimization.md`).
- **One command buffer, one encoder per token.** All ~430–580 dispatches
  are encoded onto a single `MTLComputeCommandEncoder`. No per-layer
  encoder creation.
- **PPL is correct.** ironmill PPL = 9.42, matching expected quality.

### What we don't know

Prior docs claim ~14% memory bandwidth utilization, computed as
`weight_bytes / (decode_time × peak_BW)` = 1.25 GB / (22 ms × 400 GB/s).
**This number is unreliable** because:

1. It assumes ALL 22 ms is spent loading weights. In reality, decode
   includes attention, norms, activations, dispatch overhead, and idle
   time between dispatches. If projections only account for 70% of wall
   time, actual weight bandwidth is higher (~20%).
2. It uses unobtainable peak bandwidth (400 GB/s). Achievable bandwidth
   on M2 Max is ~300–340 GB/s (75–85% of spec).
3. It ignores non-weight memory traffic (KV cache, activations, scale/zero
   buffers).
4. The "~50% utilization" attributed to MLX in prior docs is similarly
   unverified: 1.25 GB / (10.3 ms × 400 GB/s) = 30%, not 50%.

**Without profiling data, we cannot determine the actual bottleneck.**
Possible causes of the 2.16× gap include, but are not limited to:

- Per-dispatch overhead (encoding, pipeline switches, barriers)
- Intermediate buffer round-trips between fuse-able operations
- Weight memory access patterns (strided reads across rows)
- Arithmetic overhead in the hot kernel (AWQ divisions, pre-scaling)
- KV cache / attention inefficiency
- Framework overhead outside the GPU (commit/wait, CPU-side encoding time)

Any of these could dominate, and optimizing the wrong one wastes effort
(as the split-K investigation demonstrated).

---

## Phase 1: Profile and Measure (Required First Step)

Before implementing any optimization, we need per-kernel timing data to
identify the actual bottleneck. Without this, we are guessing.

### 1a. Per-dispatch GPU timing

Add Metal timestamp counters around each dispatch category to measure
actual GPU time per operation type.

Metal provides `MTLCounterSampleBuffer` with `MTLCommonCounterTimestamp`
for per-dispatch GPU timestamps. Alternatively, use separate command
buffers per operation category with `GPUStartTime`/`GPUEndTime` (coarser
but simpler).

Target output per decode step:

```
Projections (Q/K/V/O):   X.XX ms  (XX%)
FFN (gate+up+act+down):  X.XX ms  (XX%)
Attention:                X.XX ms  (XX%)
Norms + residual:         X.XX ms  (XX%)
KV cache write:           X.XX ms  (XX%)
RoPE + misc:              X.XX ms  (XX%)
Idle / overhead:          X.XX ms  (XX%)
Total:                    22.XX ms
```

This tells us WHERE time is spent. If projections are 80% of wall time,
optimizing projections matters. If they're 50%, the other 50% matters
equally.

### 1b. Memory bandwidth measurement

Use Metal's GPU performance counters (`MTLDevice.counterSets`) to measure
actual bytes read/written by the GPU per decode step. Compare against the
theoretical minimum (sum of weight + activation + KV cache bytes).

This tells us whether we're bandwidth-limited, compute-limited, or
latency-limited. Different answers lead to different optimizations:

| Bottleneck | Symptom | Optimization direction |
|------------|---------|----------------------|
| Memory BW saturated | Actual BW near achievable peak | Reduce bytes loaded (fusion, compression) |
| Memory BW underutilized | Actual BW well below peak | Fix access patterns, increase outstanding requests |
| Compute-bound | ALU utilization high, BW low | Reduce arithmetic (simpler dequant, fewer ops) |
| Latency-bound | Both low, idle time high | Reduce dispatches, fuse ops, remove barriers |

### 1c. Dispatch overhead measurement

Measure the gap between consecutive GPU dispatches — the time where the
GPU is idle between kernel completions and the next kernel start. This
directly quantifies dispatch/barrier overhead.

Approach: run the same model with a modified pipeline that replaces all
kernels with no-ops (immediately return), keeping the same dispatch
structure. The measured time is pure dispatch overhead.

### 1d. MLX comparison profiling

Profile MLX on the same model with Metal System Trace (Instruments) to
get ground-truth numbers for MLX's dispatch count, bandwidth utilization,
and per-kernel timing. This establishes the actual target, not estimated
from tok/s alone.

---

## Phase 2+: Optimize Based on Profiling Results

Implementation phases depend entirely on Phase 1 findings. Below are
candidate optimizations organized by which bottleneck they address.

### If latency-bound (high idle time between dispatches)

- **Barrier audit**: Remove unnecessary `memory_barrier_with_resources`
  calls between independent dispatches in `pipeline.rs` / `attention.rs`.
- **Fused QKV dispatch**: Replace 3 separate Q, K, V projections with a
  single batched dispatch (extends existing `batched_matvec_int4` pattern).
- **Cross-layer fusion**: Fuse FFN down-proj output with next layer's
  residual-add + RMS norm, eliminating inter-layer dispatch boundaries.

### If memory BW underutilized (access pattern issue)

- **Weight access pattern restructuring**: Profile cache hit rates per
  kernel. If strided row reads cause cache thrashing, consider transposed
  or tiled weight layouts.
- **Coalesced scale/zero access**: Interleave scale/zero data with weight
  data to improve spatial locality.
- **Prefetch hints**: Use Metal's `threadgroup_prefetch` or adjust TG
  size to match cache line boundaries.

### If memory BW saturated (need fewer bytes)

- **Intermediate buffer elimination**: Fuse adjacent kernels to avoid
  writing/reading intermediate activations through DRAM. Key pairs:
  residual + norm, SiLU + gate multiply, down-proj + residual.
- **Activation compression**: Quantize intermediate activations to INT8
  between layers (already done for input via Q8; extend to inter-layer).

### If compute-bound (high ALU utilization)

- **Simplify AWQ path**: AWQ adds 8 float divisions per iteration. If AWQ
  quality benefit is marginal, removing it could improve throughput.
- **Reduce per-iteration arithmetic**: The pre-scaled input trick
  eliminates shifts but adds 7 multiplications. Profile whether the
  net effect is positive on M2 Max.

---

## MLX Reference: What Lazy Eval Actually Does

For context, MLX's lazy evaluation provides (from `archive/mlx-backend.md`):

1. **Fewer dispatches** — fuses adjacent element-wise ops (residual add +
   RMS norm, SiLU + gate multiply) into single kernels.
2. **Eliminated intermediate traffic** — fused ops avoid DRAM round-trips
   for intermediate buffers.
3. **Cross-layer pipelining** — FFN output of layer N fuses with norm
   input of layer N+1.
4. **Single materialization point** — one `mlx_eval()` at the end,
   reducing framework overhead.

Not all of these require a full lazy eval runtime. Some (barrier removal,
batched QKV, fused residual+norm) can be done with hand-written kernels,
which ironmill already does in several places. The question is which ones
matter most — which Phase 1 profiling will answer.

---

## Validation

### Correctness

Each optimization phase must pass:
```bash
cargo build --features metal -p ironmill-inference
cargo test -p ironmill-inference --lib
```

### Numerical Equivalence

PPL must stay within ±0.05 of baseline (9.42):
```bash
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-decode-perf.toml
```

---

## Related Documents

- `archive/metal-split-k-optimization.md` — Split-K investigation (did
  not help; established that occupancy is not the bottleneck)
- `archive/metal-bandwidth-optimization.md` — Previous optimization phases
  (superblock layout, vectorized loads, multi-row TGs)
- `decode-perf-investigation.md` — Root cause analysis and ruled-out
  hypotheses
- `archive/mlx-backend.md` — MLX lazy evaluation architecture reference

## Hardware

Apple M2 Max, 64 GB unified memory, macOS 15.x.
Peak memory bandwidth: 400 GB/s (spec). Metal GPU Family: Apple 8.
