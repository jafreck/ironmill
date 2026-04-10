# Closing the MLX Gap: Investigation & Optimization Plan

## Current State

Qwen3.5-4B INT4 gs=128 decode on M2 Max 64 GB:

| Framework | tok/s | ms/tok | Gap |
|-----------|-------|--------|-----|
| ironmill (TQ INT8 KV) | 43.0 | 23.3 | baseline |
| ironmill (FP16 KV) | 46.1 | 21.7 | +7.2% |
| MLX | 114.1 | 8.8 | 2.47–2.65× faster |

ironmill per-category GPU breakdown (22.6ms total, TQ INT8 KV):

| Category | GPU time | % |
|----------|---------|---|
| Projections (total) | 10.60ms | 47.0% |
|   ↳ GDN projections (24 layers) | 7.79ms | 0.32ms/layer |
|   ↳ Std projections QKV+O (8 layers) | 2.80ms | 0.35ms/layer |
| FFN (gate+up+act+down) | 5.96ms | 26.4% |
| Attention (total) | 4.72ms | 20.9% |
|   ↳ GDN recurrent+O+res+norm (24 layers) | 2.87ms | 0.12ms/layer |
|   ↳ Std attention (8 layers) | 1.86ms | 0.23ms/layer |
| Norms + residual | 0.36ms | 1.6% |
| LM head | 0.89ms | 3.9% |

CPU encoding overhead: 0.63ms (2.7% of wall clock). **The entire gap
to MLX is GPU-side compute time.**

## What We Know

1. **The matvec kernels are identical.** MLX's `qmv_fast_impl` and
   ironmill's `superblock_matvec_int4` use the same architecture: 64
   threads, 2 simdgroups, 4 rows/SG, same pre-scale trick. The gap is
   not in the hot kernel.

2. **In-encoder dispatch overhead is near-zero.** Measured at 0.43ms
   (1.9%). Reducing dispatch count does not help (validated by batched
   QKV experiment — see `metal-lazy-eval-optimization.md`).

3. **ironmill has more op fusion than MLX.** ironmill fuses
   residual+norm, embedding+norm, gate+up+activation, and
   norm+projection (P1). MLX does none of these fusions — it runs
   residual add, RMSNorm, and projections as separate kernels.

## What We Don't Know

These are the remaining open questions after investigations 1-5.

- **Where does MLX spend its 8.8ms?** We still have no per-category GPU
  breakdown for MLX. The ~13ms GPU gap could be concentrated in
  projections, FFN, or spread across all categories. Requires manual
  Instruments.app Metal System Trace (Investigation 3).

- **Why is ironmill's GPU compute 2.5× slower despite identical
  kernels?** The CPU overhead is negligible (0.63ms). The matvec
  kernels have identical architecture. The gap must be in either:
  (a) total weight data read (ironmill's GDN has 6 proj/layer vs 4),
  (b) intermediate buffer DRAM traffic, (c) barrier/sync overhead
  within the encoder, or (d) something about how MLX achieves better
  memory bandwidth utilization at the dispatch level.

- **How much DRAM traffic goes to intermediate buffers?** Each decode
  step writes and reads `norm_out`, `attn_out`, `ffn_gate`, `ffn_down`,
  etc. We know the sizes but haven't measured the actual bandwidth cost
  of these round-trips vs the weight-loading bottleneck.

## What We Now Know (from investigations 1-5)

4. **GDN layers are faster than Standard layers.** GDN costs 0.44ms/layer
   vs Standard 0.58ms/layer (24% faster). The mixed architecture saves
   3.36ms vs all-Standard. (Investigation 1)

5. **TurboQuant KV cache costs 7.2% decode throughput.** FP16 KV cache
   gives 46.1 tok/s vs 43.0 tok/s. The savings are entirely in Standard
   attention (9.8× faster per layer with FP16 KV). Memory cost is only
   +27.5 MB at seq_len=512. (Investigation 2)

6. **CPU encoding overhead is negligible.** 0.63ms (2.7% of wall clock).
   Lazy eval, buffer management, and dispatch scheduling optimizations
   would save <1ms. (Investigation 4)

7. **Dead Q8 dispatch removal has no effect.** Confirms near-zero
   in-encoder dispatch overhead. (Investigation 5)

## Investigation Plan

Each step is designed to answer a specific question with a measurable
test. No implementation work until the investigation identifies where
the gap actually is.

### Investigation 1: GDN layer cost
**Question:** Are GDN layers slower than Standard layers per-layer?

**Test:** Use `profile-metal` per-dispatch granularity to measure
per-layer GPU time. Compare:
- Average GDN layer time (proj + recurrent + residual/norm)
- Average Standard layer time (proj + attention + residual/norm)

**Success criterion:** If GDN layers are >1.5× slower per-layer than
Standard layers, GDN optimization (or replacement with standard
attention) is a high-priority target.

**How to run:**
```bash
cargo run --release -p ironmill-bench --features metal,profile-metal -- \
  --config configs/qwen35-4b-decode-perf.toml --kernel-timing --suite decode
```
The per-dispatch output labels each dispatch with its layer index and
category.

**Files:** `crates/ironmill-inference/src/metal/pipeline.rs:510-577`,
`crates/ironmill-inference/src/metal/gdn.rs`

**Result: GDN layers are FASTER than Standard layers. Not a bottleneck.**

Per-layer-type profiling (profile-metal with GDN/Std split categories):
```
Projections (total):              10.60ms
  GDN projections:                 7.79ms  (24 GDN layers, 0.32ms/layer)
  Std projections (QKV+O):         2.80ms  (8 Std layers, 0.35ms/layer)
    QKV: 2.53ms (0.32ms/layer)  O-proj: 0.27ms (0.03ms/layer)
Attention (total):                 4.72ms
  GDN recurrent+O+res+norm:        2.87ms  (24 GDN layers, 0.12ms/layer)
  Std attention:                   1.86ms  (8 Std layers, 0.23ms/layer)
```

Per-layer cost (excl. shared norm+FFN):
- **GDN: 0.44ms/layer** (proj 0.32 + recurrent+O+res+norm 0.12)
- **Standard: 0.58ms/layer** (QKV 0.32 + O-proj 0.03 + attn 0.23)
- GDN is **24% faster** per layer (0.76× Standard cost)

Hypothetical all-Standard (32 layers): 32×0.58 + 7.22 (shared) = 25.78ms
Actual mixed (24 GDN + 8 Std):        10.56 + 4.64 + 7.22 = 22.42ms
The mixed GDN/Standard architecture **saves 3.36ms** vs all-Standard.

**Conclusion:** GDN layers are efficient. The gap to MLX is not here.

### Investigation 2: TurboQuant KV cache overhead
**Question:** How much decode time does TurboQuant INT8 KV cache cost
vs FP16 KV cache?

**Test:** Run the same decode benchmark with `kv_quant = "none"` (FP16
KV cache) and compare tok/s. The config change is one line:
```toml
# In configs/qwen35-4b-decode-perf.toml:
kv_quant = "none"   # was: kv_quant = "turbo-int8"
```

**Success criterion:** If FP16 KV cache is >10% faster on decode, the
TurboQuant dequantization overhead is significant and worth optimizing
or making configurable per model size.

**Caveats:** FP16 KV cache uses 2× memory. For Qwen3.5-4B at
max_seq_len=512 this is small (~50MB → ~100MB), but for longer
sequences or larger models it matters.

**Files:** `configs/qwen35-4b-decode-perf.toml`,
`crates/ironmill-inference/src/metal/shaders/turboquant/`

**Result: FP16 KV cache is 7.2% faster. Below 10% threshold but significant.**

| Config | tok/s | ms/tok | GPU memory |
|--------|-------|--------|-----------|
| TurboQuant INT8 KV | 43.0 | 23.3 | 2789.2 MB |
| FP16 KV cache | 46.1 | 21.7 | 2816.7 MB |
| **Δ** | **+3.1 (+7.2%)** | **−1.6** | **+27.5 MB** |

Per-category GPU comparison (profile-metal):

| Category | TQ INT8 | FP16 | Δ |
|----------|---------|------|---|
| Projections | 10.60ms | 10.69ms | +0.09 (noise) |
| Std attention (8 layers) | 1.86ms (0.23ms/layer) | 0.19ms (0.02ms/layer) | **−1.67ms (9.8× faster)** |
| GDN attention (24 layers) | 2.87ms | 2.88ms | 0 (unaffected) |
| FFN | 5.96ms | 5.99ms | 0 (noise) |
| Total GPU | 22.55ms | 21.01ms | **−1.54ms** |

The savings are **100% concentrated in Standard attention**. TurboQuant
dequantization dominates the attention kernel — standard attention is
9.8× faster with FP16 KV cache (0.02ms vs 0.23ms per layer). GDN
layers are unaffected because they use recurrent state, not KV cache.

Memory cost: +27.5 MB at seq_len=512 (trivial for 4B model). At longer
sequences (4k+) the cost scales linearly: ~220 MB at 4096 tokens.

**Conclusion:** TurboQuant KV cache costs 7.2% decode throughput for
minimal memory savings at short sequences. Consider making KV
quantization configurable based on model size and sequence length, or
defaulting to FP16 KV for small models.

### Investigation 3: MLX per-category profiling
**Question:** Where does MLX spend its 8.8ms per token?

**Test:** Use Metal System Trace (Instruments.app) to capture MLX's GPU
timeline during decode. This requires:
1. Run `scripts/mlx_decode_bench.py --tokens 200` in background
2. Attach Instruments with Metal System Trace template
3. Capture ~5 seconds of decode
4. In the GPU timeline, identify the repeating per-token pattern
5. Measure time per kernel category (matmul, attention, elementwise)

Alternatively, use the `--trace` flag with `MTL_CAPTURE_ENABLED=1`:
```bash
MTL_CAPTURE_ENABLED=1 python3 scripts/mlx_decode_bench.py --tokens 20 --trace
open /tmp/mlx_trace.gputrace  # Opens in Instruments
```

**Success criterion:** Identify which category (projections, attention,
FFN, or something else) accounts for the majority of MLX's time. This
tells us where ironmill is losing.

**Partial result: MLX confirmed at 114.1 tok/s. Per-category breakdown requires manual Instruments trace.**

MLX comparison benchmark (200 tokens, INT4 gs=128):
```
MLX:      114.1 tok/s (8.77 ms/tok)
ironmill: 43.0 tok/s  (23.3 ms/tok)   — with TQ INT8 KV
ironmill: 46.1 tok/s  (21.7 ms/tok)   — with FP16 KV
Gap:      2.47× (FP16 KV) to 2.65× (TQ INT8)
```

Note: `--trace` Metal capture works but is extremely slow (~10×
overhead). Requires manual Instruments.app Metal System Trace for
accurate per-category breakdown. The key question — which category
accounts for the 13ms gap — remains unanswered without the trace.

**Important context from investigations 1-2:** ironmill's GPU-side
compute is 21-22ms per token. MLX's total is 8.8ms. The gap of
~13ms cannot be explained by TurboQuant alone (saves 1.5ms). The
remaining ~11.5ms gap is either in projections, FFN, or something
fundamental about how work is dispatched.

### Investigation 4: Lazy eval / graph-level optimization
**Question:** Does MLX's lazy evaluation model provide a structural
advantage beyond op fusion?

MLX defers GPU work via `eval()`, batching operations into command
buffers with a configurable threshold (~50 ops on M2 Max). This enables:
- **Buffer reuse**: Temporary buffers can be recycled across operations
  without CPU-side allocation
- **Allocation elision**: Buffers that are immediately consumed can be
  optimized away
- **Execution reordering**: Independent operations can be submitted in
  an optimal order

ironmill eagerly encodes all dispatches into a single command buffer
with pre-allocated intermediate buffers. We measured near-zero dispatch
overhead (0.43ms), which suggests the command buffer model isn't the
bottleneck. But lazy eval may provide wins we haven't measured.

**Test:** This is harder to test in isolation. Two approaches:
1. **Measure ironmill CPU-side encoding time**: Time the encode loop
   (everything between creating the command buffer and calling
   `commit()`). If encoding takes >1ms, lazy eval's deferred encoding
   could help.
2. **Measure buffer allocation overhead**: Profile `ironmill-bench`
   with Instruments Time Profiler to see if buffer management or Metal
   API calls take significant time outside the GPU.
3. **Compare dispatch counts**: Count MLX dispatches per decode step
   from the Metal System Trace (Investigation 3) vs ironmill's ~459
   dispatches. If MLX has dramatically fewer dispatches, lazy eval's
   fusion is eliminating work.

**Success criterion:** If CPU-side overhead >2ms, or if MLX uses
significantly fewer dispatches due to lazy fusion, lazy eval is worth
investigating further.

**Result: CPU encoding overhead is 0.63ms. Lazy eval would not help.**

Non-profiling kernel_timing measurement (single command buffer, no
per-category boundaries):
```
[kernel-timing] decode tokens=1 gpu=22.379ms wall=23.012ms cpu=0.634ms layers=32
[kernel-timing] prefill tokens=2 gpu=233.198ms wall=266.645ms cpu=33.447ms layers=32
```

| Phase | GPU time | Wall clock | CPU overhead | % |
|-------|---------|-----------|-------------|---|
| Decode (1 token) | 22.38ms | 23.01ms | **0.63ms** | 2.7% |
| Prefill (2 tokens) | 233.20ms | 266.65ms | 33.45ms | 12.6% |

The CPU encoding overhead for decode is **0.63ms** — well below the
2ms threshold. This means:
1. **CPU encoding is NOT a bottleneck.** 97.3% of wall clock time is
   GPU execution.
2. **Lazy eval's deferred encoding would save <1ms** — negligible
   compared to the 13ms GPU gap to MLX.
3. **Buffer allocation/reuse overhead is included** in this 0.63ms
   and is already trivial (ironmill pre-allocates all intermediates).

**Conclusion:** The entire performance gap to MLX is GPU-side compute
time (22.4ms vs 8.8ms). No CPU-side or framework-level optimization
(lazy eval, buffer management, dispatch scheduling) can address this.
The investigation must focus on why ironmill's GPU dispatches take
2.5× longer than MLX's for equivalent operations.

### Investigation 5: Eliminate dead Q8 dispatch
**Question:** Does removing the wasted Q8 quantization dispatch on AWQ
layers measurably improve decode?

The Q8 input quantization runs on every Standard attention layer even
though `encode_projection_q8` ignores the Q8 data when AWQ is active.
This is 8 wasted dispatches + barriers.

**Test:** Add `&& !has_awq` to the Q8 dispatch condition. Measure
decode tok/s before and after.

**Success criterion:** Any measurable improvement. Based on the batched
QKV finding (in-encoder dispatch overhead is near-zero), this likely has
negligible impact — but it's a trivial fix worth confirming.

**Files:** `crates/ironmill-inference/src/metal/pipeline.rs:596-621`

**Result: No measurable improvement. Fix applied anyway (code correctness).**

| Config | tok/s | Δ |
|--------|-------|---|
| Before (Q8 dispatched on AWQ) | 43.0 | baseline |
| After (Q8 skipped for AWQ) | 43.0 ±0.4 | 0% (noise) |

Profile-metal confirms total GPU time unchanged (22.54ms vs 22.55ms).
This confirms the existing finding: in-encoder dispatch overhead is
near-zero on M2 Max. Removing 8 idle dispatches + barriers has no
measurable effect.

The fix was applied (changed `has_affine_int4` to `has_non_awq_int4`
in pipeline.rs) as a correctness improvement — the Q8 dispatch was
dead code for AWQ models.

## Future Optimization Directions

These should only be pursued **after** the investigations above identify
the actual bottleneck. Impact estimates below are speculative until
validated by measurement.

### Op fusion: O-projection + residual + norm

Current flow per layer:
```
attention → write attn_out → barrier → O-proj reads attn_out, writes
  ffn_down → barrier → fused_residual_norm reads ffn_down+hidden_state
```

A fused kernel could eliminate the `attn_out` and `ffn_down` DRAM
round-trips. The P1 kernel already demonstrates the norm+projection
fusion pattern. This would be the reverse: projection+residual+norm.

**Prerequisite:** Investigation 3 must show that projection time is a
significant fraction of MLX's gap, not just ironmill's internal
breakdown.

### Op fusion: FFN down-proj + residual + norm

Same pattern: fuse the down-projection with the end-of-layer
residual+norm to eliminate the `ffn_down` buffer write/read.

**Prerequisite:** Same as above.

### Lazy eval / graph compilation

If Investigation 4 shows that MLX's lazy eval provides structural wins
(buffer reuse, fewer allocations, execution reordering), consider
implementing a lightweight graph-based dispatch model:
- Build a dispatch graph at pipeline setup time (not runtime)
- Pre-allocate buffers with lifetime analysis
- Encode all dispatches for one decode step in optimal order
- This is NOT a full lazy eval runtime — it's compile-time scheduling

**Prerequisite:** Investigation 4 must show measurable CPU-side or
buffer-management overhead.

**Result: Implemented. 33.1% barrier reduction. No measurable speed change.**

Compile-time decode graph implemented in `graph.rs` with:
- `DecodeGraph`: pre-compiled operation sequence from `LayerPlan`
- `BarrierTracker`: runtime dirty-tracking that eliminates redundant barriers
- Buffer lifetime analysis identifying aliasing opportunities

Graph analysis for Qwen3.5-4B (24 GDN + 8 Standard layers):
```
Operations:            195
Conservative barriers: 290
Optimized barriers:    194 (96 eliminated, 33.1% reduction)
Total barrier slots:   266
```

The barrier tracker skips barriers for buffers that haven't been written
since their last barrier. Main savings come from:
- `v_proj` in post-QK-norm barrier (8 Standard layers, no V-norm)
- `q_gate` not needing barrier until sigmoid_gate
- Various redundant re-barriers after fused operations

Benchmark (5 runs): 43.1, 43.9, 42.7, 42.8, 42.5 tok/s (median 42.8).
Baseline: 43.0 tok/s. **No measurable improvement (within noise).**

This confirms Investigation 4's finding: in-encoder barrier overhead is
near-zero on M2 Max. Even eliminating 33% of barriers (96 of 290) has
no measurable GPU time impact. The graph infrastructure remains valuable
as analysis tooling and documentation of the pipeline's data dependencies.

**Files:** `crates/ironmill-inference/src/metal/graph.rs`,
`crates/ironmill-inference/src/metal/pipeline.rs`

### Inline norm recomputation

Eliminate the `norm_out` buffer entirely by having each projection
kernel recompute the norm inline from `hidden_state`. Trades cheap
compute (RMSNorm over 2560 elements) for saved DRAM traffic.

**Prerequisite:** Must show that norm_out DRAM traffic is a
significant contributor to the gap (not currently measured).

## Measurement Protocol

For each optimization, measure:
1. **Decode tok/s** (5 runs, drop first run, report median of remaining 4)
2. **PPL** (must remain 9.42 ±0.01)
3. **Prefill tok/s** at 128 and 512 tokens (must not regress)

```bash
# Decode (5 runs)
for i in 1 2 3 4 5; do
  cargo run --release -p ironmill-bench --features metal -- \
    --config configs/qwen35-4b-decode-perf.toml --suite decode 2>&1 \
    | grep 'tok/s'
done

# PPL
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-decode-perf.toml --suite decode-ppl

# Prefill
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-decode-perf.toml --suite prefill

# MLX comparison
python3 scripts/mlx_decode_bench.py --tokens 200

# Per-category breakdown
cargo run --release -p ironmill-bench --features metal,profile-metal -- \
  --config configs/qwen35-4b-decode-perf.toml --kernel-timing --suite decode
```

## Validated Non-Opportunities

From the profiling infrastructure work and 6 kernel optimization
attempts (see `metal-lazy-eval-optimization.md`):

- **Dispatch count reduction**: In-encoder overhead is near-zero on M2
  Max. Batching dispatches doesn't help (validated by batched QKV
  experiment: 2-3× micro-bench speedup → 0% end-to-end change).
- **Kernel architecture changes**: Both ironmill and MLX use identical
  8-row/64-thread matvec with pre-scale. No kernel-level improvement
  available.
- **Split-K / wider TGs**: GPU occupancy already saturated at K≤9216.
- **Dual-matrix fused shaders**: FFN BW advantage comes from large N
  (more TGs), not dual-stream structure.
- **AMX-accelerated matvec**: 3× regression from barrier overhead.
- **Dead Q8 dispatch removal**: Removing wasted Q8 quantization on AWQ
  layers had no measurable impact (0% change). Confirms near-zero
  dispatch overhead. Fix applied anyway for code correctness.
- **Graph-compiled barrier optimization**: Compile-time decode graph
  eliminates 96 of 290 barriers (33.1% reduction) via dirty-tracking.
  No measurable speed change — confirms barriers have near-zero cost
  within a single Metal compute encoder on M2 Max.

## Confidence Summary

| Item | Confidence | Basis |
|------|-----------|-------|
| Kernels are identical (MLX vs ironmill) | **High** | Source code comparison |
| In-encoder dispatch overhead is negligible | **High** | Measured: 0.43ms (1.9%) |
| ironmill has more op fusion than MLX | **High** | Source code comparison |
| GDN layers may be slower than Standard | **Resolved: No** | GDN is 24% faster per-layer (0.44 vs 0.58ms) |
| TurboQuant KV cache may slow attention | **Resolved: Yes** | FP16 KV is 7.2% faster; Std attn 9.8× faster per-layer |
| MLX lazy eval provides structural advantage | **Resolved: No** | CPU overhead is 0.63ms (2.7% of wall clock) |
| DRAM intermediate traffic is significant | **Unknown** | Not measured |
| The gap is in projections/FFN | **Likely** | GPU time gap is 13.6ms; proj+FFN are 74% of ironmill's GPU time |
| The gap is in attention | **Unlikely** | Attention is only 4.7ms of ironmill's GPU time |
