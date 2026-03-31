# Metal GPU Backend

## Overview

A Metal-based GPU inference backend as an alternative to the ANE direct backend.
Uses a hybrid approach: MPS for optimized matrix operations (projections, lm_head)
and custom Metal compute shaders for TurboQuant quantization, attention, and
element-wise ops (RMSNorm, SiLU, RoPE). Supports INT8 KV cache compression via
the same TurboQuant algorithm used in the ANE path.

## Architecture

```
Full decode pipeline (single token):
  0. embedding lookup    [CPU]
  ─── per layer (×num_layers) ───
  1. rms_norm           [Metal kernel]
  2. Q/K/V projection   [MPS matmul]
  3. RoPE               [Metal kernel]
  4. cache-write         [Metal kernel]  fused: rotate → quantize → INT8 cache write
  5. attention           [Metal kernel]  fused: rotate Q → dequant K/V → QK → softmax → ×V → un-rotate
  6. output projection   [MPS matmul]
  7. residual add        [Metal kernel]
  8. rms_norm            [Metal kernel]
  9. gate/up projection  [MPS matmul ×2]
 10. silu + gate mul     [Metal kernel]
 11. down projection     [MPS matmul]
 12. residual add        [Metal kernel]
  ─── end per layer ───
 13. final rms_norm      [Metal kernel]
 14. lm_head projection  [MPS matmul]
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
├── inference.rs                        GpuInference: InferenceEngine impl
├── config.rs                           GpuConfig
├── error.rs                            GpuError (wraps MetalSysError)
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
| GPU decode loop | `ironmill-inference/src/gpu/inference.rs` | `GpuInference` implements `InferenceEngine` — load, prefill, decode_step, reset |
| TQ cache write | `ironmill-inference/src/gpu/shaders/turboquant.metal` | Fused Hadamard rotation + beta-optimal quantize + INT8 write |
| TQ attention | `ironmill-inference/src/gpu/shaders/turboquant.metal` | Fused INT8 dequant + unrotation + attention |
| Standard ops | `ironmill-inference/src/gpu/shaders/*.metal` | RMSNorm, SiLU, RoPE, softmax, embedding, residual add |
| Weight loader | `ironmill-inference/src/gpu/weights.rs` | Builds on SafeTensorsProvider / GgufProvider from ironmill-compile (extracts raw weight bytes via `WeightProvider::tensor()`, loads into MTLBuffers) |

### Integration

`GpuInference` implements the `InferenceEngine` trait from `ironmill-inference`.
The existing ANE path uses the separate `RuntimeBackend` / `RuntimeModel`
abstraction; `InferenceEngine` currently has no implementations. Integrating
`GpuInference` into `ironmill-bench` requires adding a `metal` arm to the
backend dispatch that constructs a `GpuInference` directly, bypassing the
`ComputeUnits` → `RuntimeBackend` path used by CoreML backends.

| Trait Method | GPU Implementation |
|---|---|
| `load` | Parse model config from weight file metadata, load weights into MTLBuffers, compile Metal shader source into pipeline states, allocate KV cache buffers |
| `prefill` | Batched forward pass over prompt tokens (see [Prefill Strategy](#prefill-strategy)) |
| `decode_step` | Single-token forward pass through full pipeline |
| `reset` | Clear KV cache positions, reset sequence counter |

Backend selection is via the `--backend` flag on `ironmill-bench`:

```
ironmill-bench --backend metal ...
ironmill-bench --backend ane ...    # CoreML ANE (default)
ironmill-bench --backend gpu ...    # CoreML CPU+GPU
```

The `metal` backend selects the direct Metal inference engine (`GpuInference`).
The existing `gpu` and `ane` values select CoreML compute units and are
unchanged.

`ironmill-cli` remains a compiler tool (`compile`, `inspect`, `validate`). The
GPU backend is an `InferenceEngine` implementation selected at runtime by
`ironmill-bench` and any downstream library consumers of `ironmill-inference`.
Auto-detection (e.g. falling back to GPU when ANE is unavailable) is future
work.

## TurboQuant on GPU

Same quantization scheme as the ANE path — Hadamard rotation, beta-optimal
scalar INT8, optional QJL correction — but leverages GPU-specific advantages.

### Cache Write (fused kernel)

```
Input:  K_proj, V_proj  [num_kv_heads × head_dim]  FP16
        rotation_matrix [head_dim × head_dim]       FP16
Output: K_cache[layer][seq_pos], V_cache[layer][seq_pos]  INT8 (in-place)

Per KV-head, single dispatch (same pipeline for both K and V):
  1. rotated   = rotation_matrix @ head_vec
  2. quantized = round(clip(rotated × inv_scale, -128, 127))
  3. write INT8 to cache at seq_pos offset
```

Both K and V are rotated before quantization — the Hadamard rotation
spreads energy across dimensions to improve scalar quantization quality.
The quantized result is written directly into the persistent cache buffer.

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
  Q_rot = rotation @ Q_head                          ← O(1) per token
  for p in 0..seq_len:
    K_fp16    = cast(K_cache[kv_head][p]) × deq_scale
    score[p]  = dot(Q_rot, K_fp16) / √head_dim       ← K stays in rotated space
  scores = softmax(scores[0..seq_len])
  for p in 0..seq_len:
    V_fp16    = cast(V_cache[kv_head][p]) × deq_scale
    output   += score[p] × V_fp16                     ← output in rotated V-space
  attn_out = output @ rotation                        ← O(1) un-rotation
```

Q is rotated to match K's rotated space (O(1) per token) rather than
un-rotating every cached K position (O(seq_len)). Since both K and V are
stored rotated, the weighted V sum is in rotated space and requires a
final un-rotation. This matches the ANE TurboQuant approach.

### Comparison to ANE TurboQuant

| Aspect | ANE (current) | GPU (proposed) |
|---|---|---|
| Cache write | Separate ANE program + IOSurface copy | Single fused Metal kernel (K and V) |
| Attention | Separate ANE program | Single fused Metal kernel |
| Memory | IOSurface-backed tensors | MTLBuffer (shared or private) |
| Cache update | `copy_column0_fp16_as_int8_to()` | Direct write in kernel |
| Matmul | ANE 1×1 conv | MPS matrix multiply |
| Per-program limit | Observed at ~8 MB weight scale (root cause undiagnosed) | None |
| Prefill batching | Not feasible (sequential) | Possible (parallel tokens) |
| Kernel fusion | Limited by program boundaries | Cache-write fused, attention fused |

## Prefill Strategy

During prefill, the GPU processes all prompt tokens in parallel. The same
pipeline stages and kernels apply, but with a `token_count` dimension
parameter: `token_count=1` for decode, `token_count=prompt_len` (or chunk
size) for prefill.

```
Prefill pipeline (token_count prompt tokens):
  0. embedding lookup    [CPU or Metal]  batch lookup for all tokens
  ─── per layer (×num_layers) ───
  1. rms_norm           [Metal kernel]  input: [token_count × hidden_size]
  2. Q/K/V projection   [MPS matmul]   [token_count × hidden] → [token_count × proj_dim]
  3. RoPE               [Metal kernel]  per-position rotation with offset indices
  4. cache-write         [Metal kernel]  write all token_count positions at once
  5. attention           [Metal kernel]  causal mask: position p attends to [0..p]
  6–12. (same as decode, all inputs [token_count × dim])
  ─── end per layer ───
 13. final rms_norm      [Metal kernel]
 14. lm_head projection  [MPS matmul]   only last position needed for next-token prediction
```

Key differences from decode:

- MPS matmuls operate on `[token_count × dim]` inputs instead of `[1 × dim]`
- Element-wise ops (RMSNorm, SiLU, RoPE, residual add) are trivially batched —
  they process all positions in parallel with no code changes
- Attention applies a causal mask so each position only attends to prior
  positions and itself
- Cache-write writes all `token_count` positions into the KV cache in a single
  dispatch
- lm_head logits are only needed for the final position; intermediate positions
  can be skipped as an optimization

For long prompts, prefill is chunked into fixed-size segments to bound peak
memory. Each chunk writes its positions to the KV cache before the next chunk
proceeds.

## Memory Model

Weights are loaded into `MTLBuffer` with shared storage mode for initial CPU→GPU
transfer, then blitted to private storage for higher GPU read bandwidth.
Intermediate activations use private storage (GPU-only, faster). KV cache
buffers use shared storage during development for CPU-side inspection; production
builds should use private storage.

```
Weight loading:
  SafeTensors/GGUF → CPU Vec<f16> → MTLBuffer::newWithBytes (shared) → blit to private

Per-token intermediates (reused across layers):
  hidden_state    [hidden_size]              FP16  private
  attn_out        [num_heads × head_dim]     FP16  private
  ffn_intermediate [intermediate_size]       FP16  private

KV cache (persistent across tokens):
  k_cache[layer]  [num_kv_heads × max_seq × head_dim]  INT8  shared
  v_cache[layer]  [num_kv_heads × max_seq × head_dim]  INT8  shared
```

## Key Differences from ANE Path

- **No MIL compilation.** The GPU backend does not compile MIL programs. During
  `load`, it parses the model architecture directly from SafeTensors/GGUF
  metadata, compiles Metal shader source into pipeline states, and loads weights
  into MTLBuffers. The MIL IR is not involved.

- **No model splitting.** ANE's per-program compilation limits force splitting
  into `pre_attn` / `post_attn` / `lm_head` sub-programs with IOSurface
  boundaries. The GPU path keeps all weights resident and dispatches kernels
  directly.

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

- **Warm-up cost** — Metal shader compilation and pipeline state creation happen
  eagerly during `load`. The Metal driver compiles shader source to GPU machine
  code on first load; subsequent runs benefit from Metal's on-disk shader cache.

- **Error handling** — `ironmill-metal-sys` defines `MetalSysError` for FFI
  failures. The safe inference layer defines `GpuError` in
  `ironmill-inference/src/gpu/error.rs`, wrapping `MetalSysError` and
  higher-level failures (weight loading, config parsing, shader compilation).

## Design Decisions

1. **Threadgroup sizing** — one threadgroup per attention head with `head_dim`
   threads (clamped to 64–256). Sequence dimension tiled in chunks of
   `min(seq_len, 32768 / (2 × head_dim))` to fit 32 KB threadgroup memory.
   Expose tile size as a `GpuConfig` parameter for per-chip-generation tuning.

2. **Prefill strategy** — full-prompt prefill (no chunking) for prompts up to
   `max_seq_len`. For longer sequences, chunk at `max_seq_len` boundaries.
   Chunk size configurable via `GpuConfig`. MPS matmul benefits from larger
   batch dimensions, so prefer large chunks when memory allows.

3. **Weight quantization** — FP16 weights only for initial implementation. Q8_0
   support (INT8 weights with per-tensor scale, dequant before matmul) is a
   follow-up. Q4_0 deferred — requires fused dequant-matmul kernels.

4. **Shared memory budget** — 32 KB per threadgroup on M1–M4. Tile K in chunks
   of 64 sequence positions (~16 KB for head_dim=128) to leave room for score
   accumulators and partial output sums.

## References

- [TurboQuant](turboquant.md) — ANE KV cache compression (quantization algorithm)
- [ANE Inference](ane-inference.md) — ANE decode pipeline for comparison
- [ANE Constraints](ane-constraints.md) — hardware limits that motivate the GPU path
- [TurboQuant Research](../research/turboquant-analysis.md) — paper background
