# Metal GPU Backend

## Overview

A Metal-based GPU inference backend as an alternative to the ANE direct backend.
Uses a hybrid approach: MPS for optimized matrix operations (projections, lm_head)
and custom Metal compute shaders for TurboQuant quantization, attention, and
element-wise ops (RMSNorm, SiLU, RoPE). Supports INT8 KV cache compression via
the same TurboQuant algorithm used in the ANE path.

## Motivation

The ANE backend has hard constraints that limit optimization headroom:

- **~8 MB per-program weight budget** — forces model splitting, prevents
  cache-write fusion into pre_attn at Qwen3-0.6B scale.
- **Sequential token decode only** — ANE programs process one token at a time,
  no batch prefill.
- **IOSurface overhead** — each ANE sub-program requires IOSurface-backed I/O,
  adding allocation and copy cost between program boundaries.

A Metal GPU backend eliminates these constraints while reusing TurboQuant's
proven quantization scheme.

## Architecture

```
Decode step pipeline (per layer):
  1. rms_norm           [Metal kernel]
  2. Q/K/V projection   [MPS matmul]
  3. RoPE               [Metal kernel]
  4. cache-write         [Metal kernel]  fused: rotate → quantize → INT8 cache write
  5. attention           [Metal kernel]  fused: dequant → unrotate → QK → softmax → ×V
  6. output projection   [MPS matmul]
  7. residual add        [Metal kernel]
  8. rms_norm            [Metal kernel]
  9. gate/up projection  [MPS matmul ×2]
 10. silu + gate mul     [Metal kernel]
 11. down projection     [MPS matmul]
 12. residual add        [Metal kernel]
```

All ops are encoded into a single `MTLCommandBuffer` per token, committed once.
No per-op synchronization, no intermediate CPU readback.

### Crate Layout

```
crates/ironmill-metal-sys/              NEW — unsafe Metal FFI quarantine
├── src/
│   ├── lib.rs                          compile_error! on non-macOS
│   ├── device.rs                       MTLDevice, capability queries
│   ├── buffer.rs                       MTLBuffer (shared/private storage)
│   ├── command.rs                      command queue, command buffer, compute encoder
│   ├── pipeline.rs                     MTLComputePipelineState
│   ├── shader.rs                       .metal source → MTLLibrary → MTLFunction
│   ├── mps.rs                          MPSMatrixMultiplication wrapper
│   └── error.rs                        MetalSysError

crates/ironmill-inference/src/gpu/      NEW — GPU inference (safe code)
├── mod.rs
├── inference.rs                        GpuInference struct, decode/prefill/generate
├── config.rs                           GpuConfig
├── ops.rs                              kernel dispatch helpers
├── weights.rs                          SafeTensors/GGUF → MetalBuffer loading
├── shaders/                            Metal shader sources (include_str!)
│   ├── turboquant.metal
│   ├── normalization.metal
│   ├── activation.metal
│   ├── attention.metal
│   ├── rope.metal
│   ├── elementwise.metal
│   └── embedding.metal
└── turboquant/
    ├── mod.rs                          GpuTurboQuantModel
    └── cache.rs                        MTLBuffer-backed INT8 KV cache
```

### Key Components

| Component | Location | Description |
|---|---|---|
| Metal FFI | `ironmill-metal-sys/src/` | Safe wrappers for device, buffers, command submission |
| MPS matmul | `ironmill-metal-sys/src/mps.rs` | Apple-optimized FP16 matrix multiply |
| GPU decode loop | `ironmill-inference/src/gpu/inference.rs` | `GpuInference` — compile, decode, prefill, generate, reset |
| TQ cache write | `ironmill-inference/src/gpu/shaders/turboquant.metal` | Fused Hadamard rotation + beta-optimal quantize + INT8 write |
| TQ attention | `ironmill-inference/src/gpu/shaders/turboquant.metal` | Fused INT8 dequant + unrotation + attention |
| Standard ops | `ironmill-inference/src/gpu/shaders/*.metal` | RMSNorm, SiLU, RoPE, softmax, embedding, residual add |
| Weight loader | `ironmill-inference/src/gpu/weights.rs` | Reuses SafeTensorsProvider / GgufProvider from ironmill-compile |

## TurboQuant on GPU

Same quantization scheme as the ANE path — Hadamard rotation, beta-optimal
scalar INT8, optional QJL correction — but leverages GPU-specific advantages.

### Cache Write (fused kernel)

```
Input:  K_proj, V_proj  [num_kv_heads × head_dim]  FP16
        rotation_matrix [head_dim × head_dim]       FP16
Output: K_cache[layer][seq_pos], V_cache[layer][seq_pos]  INT8 (in-place)

Per KV-head, single dispatch:
  1. K_rotated = rotation_matrix @ K_head
  2. quantized = round(clip(K_rotated × inv_scale, -128, 127))
  3. write INT8 to cache at seq_pos offset
```

No separate program, no IOSurface copy. The quantized result is written
directly into the persistent cache buffer.

### Fused Attention (single kernel)

```
Input:  Q          [num_heads × head_dim]                    FP16
        K_cache    [num_kv_heads × max_seq_len × head_dim]   INT8
        V_cache    [num_kv_heads × max_seq_len × head_dim]   INT8
        rotation   [head_dim × head_dim]                     FP16
        deq_scale  scalar                                    F32
        seq_len    scalar                                    U32
Output: attn_out   [num_heads × head_dim]                    FP16

Per attention head (GQA-aware):
  for p in 0..seq_len:
    K_fp16    = cast(K_cache[kv_head][p]) × deq_scale
    K_unrot   = rotation^T @ K_fp16
    score[p]  = dot(Q_head, K_unrot) / √head_dim
  scores = softmax(scores[0..seq_len])
  for p in 0..seq_len:
    V_fp16    = cast(V_cache[kv_head][p]) × deq_scale
    output   += score[p] × V_fp16
```

Uses threadgroup shared memory for attention scores and partial output sums.
Sequence dimension is tiled for cache-friendly access.

### Comparison to ANE TurboQuant

| Aspect | ANE (current) | GPU (proposed) |
|---|---|---|
| Cache write | Separate ANE program + IOSurface copy | Single fused Metal kernel |
| Attention | Separate ANE program | Single fused Metal kernel |
| Memory | IOSurface-backed tensors | MTLBuffer (shared or private) |
| Cache update | `copy_column0_fp16_as_int8_to()` | Direct write in kernel |
| Matmul | ANE 1×1 conv | MPS matrix multiply |
| Per-program limit | ~8 MB weights | No limit |
| Prefill batching | Not feasible (sequential) | Possible (parallel tokens) |
| Kernel fusion | Limited by program boundaries | Cache-write fused, attention fused |

## Memory Model

Weights are loaded into `MTLBuffer` with shared storage mode (CPU-writable,
GPU-readable). Intermediate activations use private storage (GPU-only, faster).
KV cache buffers use shared storage for potential CPU inspection during
debugging but could be switched to private for production.

```
Weight loading:
  SafeTensors/GGUF → CPU Vec<f16> → MTLBuffer::newWithBytes (shared)

Per-token intermediates (reused across layers):
  hidden_state    [hidden_size]              FP16  private
  attn_out        [num_heads × head_dim]     FP16  private
  ffn_intermediate [intermediate_size]       FP16  private

KV cache (persistent across tokens):
  k_cache[layer]  [num_kv_heads × max_seq × head_dim]  INT8  shared
  v_cache[layer]  [num_kv_heads × max_seq × head_dim]  INT8  shared
```

## Key Differences from ANE Path

- **No MIL compilation.** The GPU backend does not compile MIL text into
  programs. It constructs a fixed decode loop from model weights and pre-compiled
  Metal kernels. The MIL IR is used only for weight extraction and architecture
  detection during model loading.

- **No model splitting.** ANE's per-program weight budget forces splitting into
  `pre_attn` / `post_attn` / `lm_head` sub-programs with IOSurface boundaries.
  The GPU path keeps all weights resident and dispatches kernels directly.

- **Single command buffer per token.** All layers' ops are encoded into one
  `MTLCommandBuffer`, avoiding per-op commit overhead and enabling the Metal
  driver to overlap compute across layers.

- **Prefill parallelism.** GPU can process multiple prompt tokens simultaneously
  in the prefill phase, unlike ANE's sequential single-token decode.

## Constraints & Considerations

- **macOS only** — Metal is Apple-platform-specific. `ironmill-metal-sys` uses
  `compile_error!` on non-macOS targets, matching `ironmill-ane-sys`.

- **FP16 compute** — all arithmetic in FP16 (matching ANE path). INT8 is storage
  only, cast to FP16 before math.

- **GQA support required** — attention kernels must handle grouped-query
  attention where `num_heads` is a multiple of `num_kv_heads`.

- **MPS matmul precision** — MPS uses hardware-optimal precision by default.
  Verify FP16 accumulation matches expected quality.

- **Weight formats** — initial support for SafeTensors (FP16) and GGUF. Future
  work: quantized weight formats (Q4_0, Q8_0) with dequantization in matmul
  kernels.

## Open Questions

1. **Threadgroup sizing** — optimal thread configuration for attention kernels
   depends on sequence length, head count, and Apple Silicon generation. Needs
   empirical tuning.
2. **Prefill strategy** — batch all prompt tokens in one matmul or chunk to
   limit memory? Depends on model size vs GPU memory.
3. **Weight quantization** — should the GPU path support INT4/INT8 weight
   compression (separate from KV cache TurboQuant)?
4. **Shared memory budget** — threadgroup memory for fused attention is limited
   to 32 KB on most Apple GPUs. Large sequence lengths may need tiling.

## References

- [TurboQuant](turboquant.md) — ANE KV cache compression (quantization algorithm)
- [ANE Inference](ane-inference.md) — ANE decode pipeline for comparison
- [ANE Constraints](ane-constraints.md) — hardware limits that motivate the GPU path
- [TurboQuant Research](../research/turboquant-analysis.md) — paper background
