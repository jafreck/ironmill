# Metal Lazy Evaluation: Reducing Dispatch Overhead

## Problem

Qwen 3.5-4B INT4 gs=128 decode on M2 Max 64 GB:

| Framework | Decode | BW Util | Dispatch Model |
|-----------|--------|---------|----------------|
| **ironmill** | **45 tok/s** | **~14%** | Eager: individual kernel dispatches |
| MLX | 97 tok/s | ~50% | Lazy: graph-batched, single eval() |
| llama.cpp | 67 tok/s | ~35% | Compute graph with op batching |
| Theoretical | 152 tok/s | 100% | 400 GB/s / 1.02 GB per token |

### Why split-K failed

Split-K dispatch was implemented and benchmarked
(see `archive/metal-split-k-optimization.md`). Result: **zero improvement**.

The M2 Max already achieves full GPU occupancy with the baseline dispatch
(38 CUs × 16 TGs/CU = 608 concurrent TGs vs ~320 needed). The bottleneck
is **not** occupancy — it's the overhead of dispatching hundreds of small
kernels per token.

### Current dispatch overhead

ironmill encodes all layer dispatches onto a single `MTLComputeCommandEncoder`
within one `MTLCommandBuffer` per token. No per-layer encoder creation.
Barriers (`memory_barrier_with_resources`) separate dependent stages.

Typical decode step for Qwen 3.5-4B (36 layers):

| Component | Dispatches/layer | Total (36 layers) |
|-----------|------------------|--------------------|
| Q8 input quantize | 1 | 36 |
| Q, K, V projections | 3 | 108 |
| QK norm + RoPE | 1–2 | 36–72 |
| KV cache write | 2 | 72 |
| Attention | 1–2 | 36–72 |
| O projection | 1 | 36 |
| Residual + norm | 1–3 | 36–108 |
| FFN (fused gate+up+act + down) | 2 | 72 |
| **Total per token** | **~12–16** | **~430–580** |

Each dispatch carries overhead:
- Metal command encoding: ~1–3 μs (set_pipeline, set_buffer × N, dispatch)
- GPU scheduling: ~5–15 μs (pipeline state switch, TG scheduling)
- Memory barriers: ~2–10 μs (cache flush/invalidate)

**Estimated overhead: 430 × 10 μs ≈ 4.3 ms per token** (19% of the 22 ms
decode latency). This is overhead where the GPU is NOT doing useful compute.

### How MLX avoids this

MLX uses lazy evaluation: operations are **recorded** (not executed) as they're
called. The entire compute graph is materialized only when `mlx_eval()` is
called. Benefits:

1. **Fewer dispatches** — MLX's compiler fuses adjacent element-wise ops
   (residual add + RMS norm, SiLU + gate multiply) into single kernels,
   reducing dispatch count per layer.
2. **Amortized scheduling** — one graph submission replaces hundreds of
   individual dispatches.
3. **Cross-layer pipelining** — the FFN output of layer N fuses with the
   norm input of layer N+1, eliminating an intermediate buffer read/write.
4. **Optimized memory** — the graph compiler can reuse intermediate buffers
   more aggressively since it sees the full dataflow.

---

## Approach: Dispatch Fusion Without Lazy Eval

Full lazy evaluation (recording a compute graph, compiling it, then
executing) is a large architectural change. Instead, ironmill can capture
most of the benefit through **dispatch fusion** — reducing the number of
Metal dispatches by merging adjacent operations.

### Principles

1. **Fuse adjacent kernels** that share input/output buffers into single
   dispatches, eliminating intermediate barriers and buffer traffic.
2. **Batch per-layer dispatches** where possible by encoding broader
   operations (e.g., QKV as one dispatch instead of three).
3. **Remove unnecessary barriers** — audit existing
   `memory_barrier_with_resources` calls and remove those between
   independent dispatches.
4. **Reduce pipeline state switches** — group dispatches by pipeline
   to minimize state changes.

### Non-goals

- Full compute graph / lazy tensor runtime (too large a change)
- Runtime kernel JIT / compilation (compile-time metallibs only)
- Changing the single-encoder-per-token architecture (already optimal)

---

## Implementation Plan

### Phase 1: Barrier Audit and Removal

**Target: 5–10% decode improvement**

Audit every `memory_barrier_with_resources` call in `pipeline.rs` and
`attention.rs`. Many barriers exist between dispatches that don't share
buffers or where the output of dispatch A is not the input of dispatch B.
Metal guarantees execution order within a command encoder — barriers are
only needed when a subsequent dispatch reads from a buffer that a prior
dispatch wrote to.

Barriers to audit:

| Location | Between | Likely needed? |
|----------|---------|----------------|
| After Q8 quantize | Q8 → Q/K/V projections | Yes (read-after-write) |
| After Q/K/V projections | Projections → RoPE | Yes |
| After RoPE | RoPE → cache write | Yes |
| After cache write | Cache → attention | Yes |
| After attention | Attention → O projection | Yes |
| After O projection | O proj → residual add | Yes |
| Post-attn norm → FFN | Norm → gate/up | Yes |
| After FFN gate+up+act | Fused → down proj | Yes |
| After FFN down | Down → residual add | Yes |
| **Between independent projections** | e.g., Q → K → V | **No** (independent) |
| **After residual → before next layer's norm** | Cross-layer | **Maybe not** |

Action: remove barriers between independent dispatches. Verify correctness
via PPL regression test.

### Phase 2: Fused QKV Projection

**Target: 5–10% decode improvement**

Replace 3 separate Q, K, V projection dispatches with a single fused
dispatch that computes all three in one kernel. This eliminates:
- 2 dispatch overheads
- 2 barriers
- Pipeline state is already the same (same kernel, different weights)

#### 2a. Concatenated QKV weight buffer

At weight loading time, concatenate Q, K, V weight matrices into a single
`[N_q + N_k + N_v, K]` buffer with concatenated scales/zeros. The fused
kernel processes all rows, writing Q/K/V outputs to their respective
buffers using row offset parameters.

This is similar to the existing `batched_matvec_int4` pattern (which
batches gate+up projections). Extend it to Q+K+V.

#### 2b. Fused QKV + RoPE + Q8 quantize

The ultimate fusion: a single dispatch that:
1. Quantizes input to Q8
2. Computes Q, K, V projections (INT4×Q8)
3. Applies RoPE to Q and K

This eliminates 4–5 dispatches per layer (Q8 + Q + K + V + RoPE → 1).
Requires a new combined kernel.

### Phase 3: Cross-Layer Buffer Fusion

**Target: 5–8% decode improvement**

Currently, each layer writes residual output to a buffer, then the next
layer reads it for normalization. If the residual add and the next layer's
RMS norm are fused (already partially done via `fused_residual_rms_norm`),
the intermediate buffer write/read is eliminated.

Audit which layers currently use the fused residual+norm path and extend
coverage to all applicable layers.

### Phase 4: FFN Down + Residual + Norm Fusion

**Target: 3–5% decode improvement**

Fuse the FFN down projection output with the residual add and the next
layer's input norm into a single operation:

```
output = RMSNorm(residual + down_proj(fused_ffn_output))
```

This is the "cross-layer fusion" that MLX achieves automatically. In
ironmill, it requires a new fused kernel that combines:
1. Matrix-vector multiply (down projection)
2. Residual addition
3. RMS normalization

The existing `fused_residual_norm_affine_matvec_int4` kernel already
does residual+norm+matvec for the attention path. A symmetric kernel
for the FFN→next-layer path would close this gap.

---

## Expected Impact

| Phase | Decode (tok/s) | Rationale |
|-------|---------------|-----------|
| Current baseline | 45 | Single encoder, ~430 dispatches/token |
| Phase 1: Barrier audit | ~48 | Remove ~30% unnecessary barriers |
| Phase 2: Fused QKV | ~53 | −2 dispatches/layer × 36 layers |
| Phase 3: Cross-layer buffer fusion | ~57 | Eliminate intermediate reads |
| Phase 4: FFN→norm fusion | ~60 | Cross-layer kernel fusion |
| **Projected total** | **~60** | |
| MLX reference | 97 | Full lazy eval + 16 vals/thread |

The remaining gap to MLX after dispatch fusion likely requires:
- **16 vals/thread** — doubles memory throughput per instruction
- **Full lazy evaluation** — enables arbitrary cross-op fusion
- **Quad-group parallelism** — SIMD-level optimization

---

## Validation

### Correctness

Each phase must pass:
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

### Performance

Measure per-phase impact:
```bash
# Quick decode-only benchmark
cargo run --release -p ironmill-bench --features metal -- \
  --config configs/qwen35-4b-decode-perf.toml
```

---

## Files to Modify

### Phase 1 (barrier audit)
- `metal/pipeline.rs` — remove unnecessary barriers

### Phase 2 (fused QKV)
- `shaders/quantized/affine_batched.metal` — new 3-way batched QKV kernel
- `metal/ops/quantized.rs` — QKV batched encode helper
- `metal/pipeline.rs` — route to fused QKV path
- `metal/weights.rs` — concatenated QKV weight loading

### Phase 3–4 (cross-layer fusion)
- `shaders/norm/superblock_fused_norm.metal` — extend fused norm kernels
- `metal/pipeline.rs` — cross-layer dispatch routing

### Existing infrastructure to reuse
- `superblock_batched_affine_matvec_int4` — prior art for batched projection
- `fused_residual_norm_affine_matvec_int4` — prior art for residual+norm+matvec
- `fused_residual_rms_norm` — fused residual+norm

---

## Related Documents

- `archive/metal-split-k-optimization.md` — Split-K investigation (did
  not help; established that occupancy is not the bottleneck)
- `archive/metal-bandwidth-optimization.md` — Previous optimization phases
  (superblock layout, vectorized loads, multi-row TGs)
- `decode-perf-investigation.md` — Root cause analysis identifying dispatch
  overhead as a key factor
- `archive/mlx-backend.md` — MLX lazy evaluation architecture reference

## Hardware

Apple M2 Max, 64 GB unified memory, macOS 15.x.
Peak memory bandwidth: 400 GB/s. Metal GPU Family: Apple 8.
