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

## Phase 1: Profile and Measure — COMPLETED

### 1a. Per-category GPU timing — DONE

**Implementation**: Added per-category command buffer profiling to
`run_pipeline_inner`. When `kernel_timing` is enabled for decode
(token_count == 1), the pipeline splits into separate command buffers
at each operation category boundary. Each command buffer's
`GPUStartTime`/`GPUEndTime` gives per-category GPU time.

Categories: `embed`, `proj`, `attn`, `ffn`, `norm`, `lm_head`.

Profiling mode disables P1 fusion and splits GDN encode into separate
projection/core phases to ensure accurate per-category attribution.

**Results** (Qwen3.5-4B INT4 AWQ gs=128, M2 Max 64 GB):

```
Projections (Q/K/V/O):   10.48ms  (47.1%)
FFN (gate+up+act+down):   5.90ms  (26.5%)
Attention:                 4.59ms  (20.6%)
Norms + residual:          0.35ms  ( 1.6%)
Embedding:                 0.01ms  ( 0.0%)
LM head:                   0.90ms  ( 4.0%)
─────────────────────────────────────────
Total GPU:                22.24ms
```

Note: Qwen3.5-4B has 24 GDN (linear_attention) + 8 Standard
(full_attention) layers. "Attention" includes GDN recurrent kernel +
O projection + residual+norm for GDN layers.

### 1b. Memory bandwidth measurement — COMPUTED ANALYTICALLY

MTLCounterSampleBuffer requires additional Metal API wrappers not yet
in ironmill-metal-sys. Bandwidth was computed from profiled GPU time and
known weight sizes.

| Category | Weight bytes (32 layers) | GPU time | Effective BW | % of peak (400 GB/s) |
|----------|--------------------------|----------|-------------|---------------------|
| Projections | ~526 MB | 10.48ms | **50.2 GB/s** | **12.6%** |
| FFN | ~1133 MB | 5.90ms | **192 GB/s** | **48.0%** |

**Key finding**: Projections achieve only 12.6% of peak bandwidth, while
FFN achieves 48.0%. The 3.8× gap is because:
- FFN uses `fused_ffn_gate_up_act_int4` which reads 2 weight matrices
  per threadgroup, doubling concurrent memory requests.
- Projections use individual 2-row matvec dispatches (1 weight matrix
  per threadgroup), limiting memory controller utilization.
- For K=2560 (hidden_size), each threadgroup reads only ~1280 bytes
  of weight data per output row — too little work to keep the memory
  pipeline saturated.

### 1c. Dispatch overhead — MEASURED

From profiling: Total GPU time = 22.24ms, baseline wall clock = 22.67ms.

**Dispatch overhead = 0.43ms (1.9% of wall clock).**

The pipeline is NOT latency-bound. The ~283 dispatches per decode step
add negligible overhead. Barriers and encoder switches are not the
bottleneck.

### 1d. MLX comparison profiling — SKIPPED

Requires running MLX with Metal System Trace (Instruments). Not
implementable via the ironmill benchmark infrastructure. The existing
tok/s comparison (45 vs 97 tok/s) sufficiently establishes the gap.

---

## Phase 2: Optimization Attempts — COMPLETED

### Diagnosis

The profiling data clearly identifies the bottleneck:

| Bottleneck | Symptom | Match? |
|------------|---------|--------|
| Memory BW saturated | Actual BW near achievable peak | ❌ (12.6% projection, 48% FFN) |
| **Memory BW underutilized** | **Actual BW well below peak** | **✅ Projections at 12.6%** |
| Compute-bound | ALU utilization high, BW low | ❌ (compute intensity 2.8 FLOP/byte, well below ridge point 34) |
| Latency-bound | Both low, idle time high | ❌ (overhead 1.9%) |

**Root cause**: Projection kernels underutilize memory bandwidth because
each threadgroup processes only one weight matrix. With K=2560, each TG
reads ~1280 bytes — insufficient to saturate the memory pipeline.

The FFN fused kernel proves that multi-matrix kernels (reading 2+ weight
streams per TG) achieve 3.8× higher bandwidth on the same hardware.

### Attempted optimizations

#### 1. Batched Q+gate / K+V using existing batched kernel — REJECTED

**Hypothesis**: Replacing 4 individual 2-row dispatches (Q, K, V, gate)
with 2 batched dispatches (Q+gate, K+V) using the existing
`batched_matvec_int4` kernel would reduce dispatch count and enable
memory interleaving.

**Result**: **32% regression** (30.0ms vs 22.7ms baseline).

**Why**: The batched kernel uses 32 threads/TG with 1 row/TG, while the
individual 2-row kernel uses 64 threads/TG with 2 rows/TG. The 2-row
kernel amortizes input vector reads across 2 output rows, making it
significantly more efficient per-row. The dispatch reduction does not
compensate for the per-TG efficiency loss.

#### 2. Individual 2-row dispatches for GDN (replacing batched) — REJECTED

**Hypothesis**: GDN layers use a batched 1-row INT4 kernel. Switching to
4 individual 2-row dispatches would improve per-TG efficiency.

**Result**: **Neutral to slightly worse** (23.2ms vs 22.7ms).

**Why**: The GDN batched kernel's single-dispatch benefit (reduced
scheduling overhead for 12,352 combined threadgroups) offsets the 2-row
kernel's per-TG advantage.

#### 3. Skip Q8 quantization for AWQ weights — REJECTED

**Finding**: The Q8 input quantization dispatch + barrier runs on every
Standard attention layer but the Q8 data is NEVER read when AWQ is
active (`awq_scales.is_some()` causes the Q8 code path to be skipped in
`encode_projection_q8`).

**Result**: Neutral (23.4ms vs 22.7ms). The wasted Q8 dispatch may
provide a minor cache-warming benefit for subsequent projections.

#### 4. Fused dual-matrix per-thread kernel (new shader) — REJECTED

**Hypothesis**: A new Metal shader (`superblock_fused_dual_matvec_int4`)
where each thread reads from both weight matrices per iteration (matching
the FFN fused kernel's per-thread dual-stream pattern) would improve
projection bandwidth. Two variants tested:

- **Simdgroup-split** (v1): 64 threads/TG, 2 simdgroups each handling 4
  rows from a different matrix. Dispatch: `max(ceil(N0/4), ceil(N1/4))`.
- **Per-thread dual-stream** (v2): 32 threads/TG, each thread reads from
  both matrices per iteration. Dispatch: `max(N0, N1)`.

Applied to both Standard (Q+gate, K+V) and GDN (QKV+Z) layers.

**Result**: **Neutral** (23.0-23.2ms vs 22.7ms baseline). PPL=9.42 ✓.

**Why**: The bottleneck at K=2560 is not about per-TG memory stream
count. The 8-row individual kernel already generates 4 independent row
addresses per thread (high address diversity), while the fused kernel
only generates 2 addresses from different buffers. Both saturate the
GPU's per-thread memory request queue similarly. The FFN fused kernel
achieves higher bandwidth because N=9216 gives 9216 TGs (sustained
memory pressure), not because of its dual-stream structure.

### Conclusion

**The 2.16× gap between ironmill and MLX is primarily explained by
projection kernel bandwidth underutilization (12.6% vs FFN's 48% BW).**
The root cause is the small K=2560 dimension: each thread reads only
~40 bytes per iteration, completing quickly relative to instruction
pipeline latency. This limits throughput regardless of how many memory
streams are active.

Four optimization approaches were tested and rejected:
1. Batched 1-row kernel — 32% regression (lost 2-row efficiency)
2. Individual 2-row for GDN — neutral (lost batching benefit)
3. Q8 skip for AWQ — neutral
4. **Fused dual-matrix shader — neutral** (per-thread dual-stream
   doesn't improve over 8-row's 4-address diversity at K=2560)

The key insight is that the FFN's 3.8× bandwidth advantage comes from
having more total threadgroups (N=9216 vs N≤4096), not from its
dual-stream structure. Projections with smaller N cannot sustain the
same memory pressure.

**Remaining optimization directions** (untested):
1. **AMX-accelerated matvec**: The existing `affine_matvec_int4_amx`
   kernel uses simdgroup matrix multiply (256 threads, 64 rows/TG,
   cooperative dequant to threadgroup memory). This was not benchmarked
   for the INT4 AWQ path. The wider memory loads of AMX operations may
   better saturate bandwidth for small K.
2. **Kernel launch pipelining**: Submit multiple command buffers
   concurrently instead of one sequential encoder. Metal can overlap
   kernel dispatch from different command buffers.
3. **Weight layout transposition**: Rearranging weight data so that
   threads read from contiguous cache lines across rows rather than
   within a row may improve L2 cache hit rates.

---

## Phase 3: Future Work

The profiling + optimization work eliminated several hypotheses about the
2.16× MLX gap. The remaining avenues require fundamentally different
kernel architectures, not just parameter tuning.

### Priority 1: AMX-accelerated INT4 projection path

The existing `affine_matvec_int4_amx` kernel (256 threads, 64 rows/TG,
cooperative dequant → threadgroup memory → simdgroup MMA) is not used
for INT4 AWQ projections. Testing this path for AWQ weights could unlock
wider memory loads via the Apple Matrix coprocessor. This is the lowest
risk / highest leverage change: the kernel already exists.

### Priority 2: Multi-row fused dual-stream kernel

Combine the 8-row kernel's address diversity with the fused kernel's
dual-matrix reads. Each simdgroup would process 4 rows from matrix A
AND 4 rows from matrix B simultaneously:
- 64 threads/TG, 2 simdgroups
- Each SG: 4 rows × 2 matrices = 8 addresses per thread per iteration
- This maximizes both per-thread address diversity and per-TG work

This requires a more complex shader with 8 accumulators per thread.

### Not recommended (tested and rejected)

- **Fused dual-matrix 32-thread kernel**: Matches FFN fused pattern but
  neutral at K=2560. The FFN's 3.8× BW advantage comes from N=9216
  (more TGs for sustained pressure), not from dual-stream structure.
- **Batched 1-row kernels for Standard layers**: 32% regression.
- **Individual 2-row for GDN**: Neutral (lost batching benefit).
- **Split-K**: Already tested separately — GPU occupancy saturated.
- **Barrier removal**: Overhead is only 1.9%.
- **Q8 skip for AWQ**: Neutral.

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

### Profiling infrastructure

Run with `--kernel-timing` to see per-category GPU breakdown:
```bash
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-decode-perf.toml --kernel-timing --suite decode
```

### Correctness

All changes pass:
```bash
cargo build --features metal -p ironmill-inference
cargo test -p ironmill-inference --lib
```

### Numerical Equivalence

PPL verified at 9.42 (±0.00 from baseline):
```bash
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-decode-perf.toml
```

### Performance

No regression from profiling infrastructure (code gated behind
`kernel_timing && token_count == 1`):

| Configuration | Decode (ms/tok) | tok/s | PPL |
|---------------|-----------------|-------|-----|
| Baseline | 22.67 | 44.1 | 9.42 |
| After profiling infra | 22.77 | 43.9 | 9.42 |

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
