# Closing the MLX Gap: Investigation & Optimization Plan

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

**We do not have a per-category breakdown for MLX.** We cannot attribute
the 14.5ms gap to specific categories without measuring MLX's GPU time
per operation.

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

These are open questions. Each must be answered by measurement, not
assumption.

- **Where does MLX spend its 8.8ms?** We have no per-category GPU
  breakdown for MLX. The 14.5ms gap could be concentrated in one
  category or spread across all of them.

- **Is GDN slower than standard attention for decode?** ironmill uses
  24 GDN (linear attention) + 8 Standard layers. MLX uses 32 Standard
  layers. GDN involves a recurrent kernel that may or may not be slower
  than flash attention for single-token decode. We don't know.

- **Does TurboQuant INT8 KV cache slow down attention?** ironmill
  compresses KV cache to INT8 via TurboQuant, adding dequantization
  overhead on every attention read. This saves memory but may cost
  decode speed. We haven't measured attention with FP16 KV cache.

- **Does MLX's lazy eval model provide a structural advantage?** MLX
  batches operations and defers GPU submission via `eval()`. ironmill
  eagerly encodes all dispatches into a single command buffer. Both
  achieve near-zero dispatch overhead, but MLX's lazy eval may enable
  framework-level optimizations (buffer reuse, allocation elision) that
  we haven't quantified.

- **How much DRAM traffic goes to intermediate buffers?** Each decode
  step writes and reads `norm_out`, `attn_out`, `ffn_gate`, `ffn_down`,
  etc. We know the sizes but haven't measured the actual bandwidth cost
  of these round-trips vs the weight-loading bottleneck.

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

## Confidence Summary

| Item | Confidence | Basis |
|------|-----------|-------|
| Kernels are identical (MLX vs ironmill) | **High** | Source code comparison |
| In-encoder dispatch overhead is negligible | **High** | Measured: 0.43ms (1.9%) |
| ironmill has more op fusion than MLX | **High** | Source code comparison |
| GDN layers may be slower than Standard | **Unknown** | Not measured |
| TurboQuant KV cache may slow attention | **Unknown** | Not measured |
| MLX lazy eval provides structural advantage | **Unknown** | Not measured |
| DRAM intermediate traffic is significant | **Unknown** | Not measured |
| The gap is in projections/FFN | **Unknown** | No MLX breakdown |
| The gap is in attention | **Unknown** | No MLX breakdown |
