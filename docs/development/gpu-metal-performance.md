# Metal GPU Inference Performance Roadmap

**Goal:** Close the ~7–10× throughput gap with llama.cpp on Apple Silicon.

## Current Performance

| Model | ironmill | llama.cpp (Q8) | Gap |
|-------|----------|----------------|-----|
| Qwen3-1.7B (M2 Max, FP16) | 5.5 tok/s | ~40 tok/s (Q8) | ~7× |
| Qwen3-8B (M2 Max, FP16) | 2.2 tok/s | ~18 tok/s (Q8) | ~8× |

The gap is not a single bottleneck — it compounds across six architectural
differences between our MPS-based pipeline and llama.cpp's fully custom
Metal backend.

## Architecture Comparison

| Aspect | ironmill (current) | llama.cpp |
|--------|--------------------|-----------|
| Dense matmul | MPS framework | Custom tiled kernels |
| Op fusion | None (discrete dispatches) | RMSNorm+scale, gate+up+SiLU |
| Attention | Naive sequential scan | Flash/tiled attention |
| KV cache write (FP16) | CPU readback + scatter | GPU-only scatter kernel |
| Command buffers | Split mid-layer (FP16 path) | Single buffer, no mid-layer sync |
| Weight layout | Row-major, as-loaded | Pre-packed blocked format |

## Bottleneck Analysis

### 1. CPU-Side KV Cache Scatter (FP16 Path) — Critical

The FP16 KV path calls `wait_until_completed()` mid-layer, reads Q/K/V
projections back to CPU via `read_bytes()`, scatters them into cache buffers
on the CPU, then re-submits work to GPU. This happens **every layer, every
token**.

- **Files:** `inference.rs:901-908` (commit+wait), `inference.rs:910-942`
  (CPU readback)
- **Impact:** Kills GPU pipeline occupancy. Each `wait_until_completed()`
  drains the GPU command queue entirely.
- **Note:** The TurboQuant path already avoids this — it writes KV cache
  directly on GPU via `turboquant_cache_write` kernel.

### 2. MPS Matmul Overhead — High

Every linear projection (Q/K/V/O/gate/up/down = 7 per layer, plus the
LM head) goes through Apple's MPS `MPSMatrixMultiplication`. MPS adds
per-dispatch overhead: descriptor validation, internal buffer management,
and API-layer indirection that a custom kernel avoids.

For decode (batch=1), these are all matrix-vector products where a custom
SIMD-group kernel can be 2–4× faster than MPS due to better occupancy
control and elimination of framework overhead.

- **Files:** `inference.rs:361-427` (MPS cache build),
  `ops.rs:163-234` (encode helpers)

### 3. No Flash/Tiled Attention — High

`standard_attention` and `turboquant_attention` perform a sequential scan
over the KV cache:

```metal
for (uint p = 0; p < seq_len; p++) {
    // load K[p], compute dot product, online softmax update
    // load V[p], accumulate weighted sum
}
```

This is bandwidth-bound for long sequences: each cache position requires a
global memory load with no reuse in threadgroup memory. Flash attention
tiles cache blocks into threadgroup SRAM, reducing global memory traffic
by the tile factor.

- **Files:** `attention.metal:67-133`, `turboquant.metal:279-410`
- **Impact:** Scales poorly with sequence length. At seq_len=2048 on 8B,
  attention becomes a significant fraction of decode time.

### 4. No Operator Fusion — Medium

Each transformer layer dispatches 10+ separate kernel calls. The exact
count is model-dependent (e.g., Qwen3 adds Q/K norm steps), but a
typical layer looks like:

1. RMSNorm (input)
2. Q projection (MPS matmul)
3. K projection
4. V projection
5. Q norm (model-dependent)
6. K norm (model-dependent)
7. RoPE (Q)
8. RoPE (K)
9. KV cache write
10. Attention
11. O projection
12. Residual add
13. RMSNorm (post-attention)
14. Gate projection
15. Up projection
16. SiLU+gate
17. Down projection
18. Residual add

Each dispatch has kernel launch overhead and forces a pipeline flush
between ops. Fusing common sequences reduces dispatches:

- **RMSNorm + first matmul**: Eliminate the norm output buffer; compute
  norm inline during the matmul's input load.
- **Gate + Up + SiLU**: These three ops read/write the same buffers.
  A single kernel can compute `SiLU(gate(x)) * up(x)` with one read of x.
- **Residual add + RMSNorm**: The residual output feeds directly into the
  next norm; fuse to avoid a round-trip through global memory.

### 5. Weight Layout — Medium

Weights are stored row-major as loaded from SafeTensors. MPS handles
transposition internally, but custom kernels benefit from pre-packing
weights into blocked formats aligned to threadgroup tile sizes (e.g.,
32×32 or 64×64 blocks for SIMD-group matrix operations).

- **Files:** `weights.rs:208-212` (dense buffer creation)

### 6. No SIMD-Group Matrix Operations — Medium

Apple GPUs (M1+) support `simdgroup_matrix` types for 8×8 matrix tiles
in registers. llama.cpp uses these for matmul inner loops, achieving near
peak ALU throughput. Our custom kernels (quantized matmul, attention) use
scalar or basic SIMD reductions instead.

- **Files:** `quantized_matmul.metal:17-148` (scalar dot products in
  `polarquant_matvec_int4` / `polarquant_matmul_int4`)

## Optimization Roadmap

### Phase 1: Eliminate CPU Sync Points

**Target: 2× speedup on FP16 path, unblocks all other optimizations.**

| Task | Description | Files |
|------|-------------|-------|
| GPU-side KV scatter | Write a Metal kernel to scatter Q/K/V projections into KV cache directly on GPU, eliminating CPU readback | `inference.rs`, new `kv_scatter.metal` |
| Single command buffer | Remove mid-layer `commit()`/`wait_until_completed()`, encode all layers into one command buffer | `inference.rs:901-908` |
| Defer logit readback | Only read logits back to CPU after all layers complete | `inference.rs:1211-1235` |

### Phase 2: Custom Matmul Kernels

**Target: 2–3× speedup for decode (batch=1).**

| Task | Description | Files |
|------|-------------|-------|
| Custom FP16 matvec | Replace MPS with a SIMD-group kernel for M=1 (decode). Use `simdgroup_matrix_storage` for 8×8 tiles | New `matvec.metal`, replace MPS calls in `inference.rs` |
| Keep MPS for prefill | MPS is reasonable for M>1 (prefill). Switch dynamically based on token count | `inference.rs` dispatch logic |
| Weight pre-packing | At load time, transpose and tile weights into blocked format for the custom kernel | `weights.rs` |

### Phase 3: Flash Attention

**Target: Better scaling with sequence length, ~1.5× for long contexts.**

| Task | Description | Files |
|------|-------------|-------|
| Tiled KV loading | Load KV cache in tiles (e.g., 64 positions) into threadgroup memory before computing attention scores | `attention.metal` |
| Block-wise softmax | Maintain running max/sum across tiles (already have online softmax — extend to tile granularity) | `attention.metal` |
| Shared Q tile | Load Q once into threadgroup memory, reuse across all KV tiles | `attention.metal` |
| TurboQuant flash | Same tiling for quantized attention, with dequant happening in threadgroup memory | `turboquant.metal` |

### Phase 4: Operator Fusion

**Target: ~1.3× from reduced dispatch overhead and memory traffic.**

| Task | Description | Files |
|------|-------------|-------|
| Fused SiLU+gate | `silu_gate` in `activation.metal` is already a standalone kernel computing `silu(gate) * up`, but it's dispatched separately from the matmuls. True fusion would fold this into the gate+up matmul kernel to avoid the intermediate buffer writes | `activation.metal` |
| Fused residual+norm | Compute `residual = a + b` then RMSNorm in one kernel, avoiding a global memory round-trip | New fused kernel |
| Fused RoPE+attention | Apply RoPE to Q inside the attention kernel's Q load, eliminating the separate RoPE dispatch | `attention.metal` |

### Phase 5: Advanced Optimizations

**Target: Diminishing returns, but compounds with above.**

| Task | Description |
|------|-------------|
| Double-buffered command encoding | Overlap CPU-side encoding of layer N+1 with GPU execution of layer N |
| Async compute + copy | Use separate command queues for compute vs memory copies |
| Device capability detection | Probe GPU family at init, select optimal threadgroup sizes and kernel variants per chip |
| Speculative decode support | Batch multiple candidate tokens through prefill path |

## Expected Cumulative Impact

| Phase | Individual | Cumulative | Est. Qwen3-8B tok/s |
|-------|-----------|------------|---------------------|
| Current | — | — | 2.2 |
| Phase 1 | ~2× | 2× | ~4.5 |
| Phase 2 | ~2.5× | 5× | ~11 |
| Phase 3 | ~1.5× | 7.5× | ~16 |
| Phase 4 | ~1.3× | ~10× | ~22 |

These estimates assume decode (batch=1). Prefill performance has a
different profile (compute-bound rather than dispatch-overhead-bound).

## Key References

- llama.cpp Metal backend: [DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/5.2-metal-backend-(apple))
- Flash Attention: [Dao et al., 2022](https://arxiv.org/abs/2205.14135)
- Apple Metal Best Practices: [developer.apple.com](https://developer.apple.com/metal/best-practices/)
- SIMD-group matrix operations: [Metal Shading Language Spec §2.7](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
