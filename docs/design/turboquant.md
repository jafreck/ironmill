# TurboQuant - ANE KV Cache Compression

> Consolidated from `turboquant-implementation.md` and `turboquant-zero-copy.md`.
> Original investigation docs archived in `docs/archive/`.

## Overview

TurboQuant compresses KV cache entries from FP16 to INT8 at runtime during
autoregressive decoding, halving cache memory and bandwidth. It uses Hadamard
rotation for incoherence, beta-optimal scalar quantization, and optional QJL
sign correction - all running on the ANE via hand-written MIL programs.

## Architecture

```
Decode step pipeline (per layer):
  1. pre_attn (ANE)    → Q, K, V projections via 1×1 conv
  2. cache-write (ANE) → Hadamard rotate K/V → quantize to INT8 → write IOSurface cache
  3. attention (ANE)    → dequantize cache → matmul Q×K → softmax → matmul ×V
  4. post_attn (ANE)    → FFN via 1×1 conv + SiLU activation
```

Each step is a separate ANE sub-program with IOSurface tensors passed between them.

### Key Components

| Component | Location | Description |
|---|---|---|
| MIL emitter | `ironmill-inference/src/ane/turboquant/mil_emitter.rs` | Generates cache-write and cache-read MIL programs |
| Quantizer | `ironmill-inference/src/ane/turboquant/quantize.rs` | Beta-optimal scalar quantization |
| Hadamard | `ironmill-inference/src/ane/turboquant/hadamard.rs` | Rotation matrices for incoherence |
| QJL | `ironmill-inference/src/ane/turboquant/qjl.rs` | Sign correction via Johnson-Lindenstrauss |
| Cache manager | `ironmill-inference/src/ane/turboquant/cache.rs` | IOSurface-backed INT8 KV cache |

## Implementation Status

### Implemented
- INT8 KV cache storage pipeline (quantize → cache write → cache read → dequantize)
- Hadamard rotation matrices
- Beta-optimal scalar quantization
- QJL sign correction
- Zero-copy IOSurface cache updates (`copy_column0_fp16_as_int8_to`)
- Direct IOSurface-to-IOSurface data path (no staging buffers)
- Q-rotation variant (rotate Q at read time instead of un-rotating K/V)
- Chunked lm_head on ANE

### Not Yet Validated
- **End-to-end correctness** - no perplexity or token-agreement tests exist yet.
  See `docs/development/QUALITY_BENCHMARK_PLAN.md`.

## Data Path (Zero-Copy)

Previous architecture required 7 CPU transfers, 2 format conversions, and 3
staging buffers per layer. The current architecture uses direct IOSurface copies:

| Step | Operation | Path |
|---|---|---|
| Cache write | `copy_column0_fp16_as_int8_to()` | IOSurface → IOSurface (direct) |
| Cache read | ANE `cast int8→fp16` in MIL program | IOSurface (in-place on ANE) |

Total per layer: 2 direct IOSurface copies, 0 staging buffers.

## Performance

Qwen3-0.6B on Apple Silicon:

| Config | Throughput | KV Cache Size |
|---|---|---|
| FP16 baseline | ~13–14 tok/s (128 tokens) | 29.0 MB |
| TurboQuant INT8 (Q-rotation) | ~12.3 tok/s | 14.5 MB |
| TurboQuant INT8 (K/V un-rotation) | ~11.6 tok/s | 14.5 MB |

Q-rotation is preferred: it avoids un-rotating all cached K/V entries and
rotates only the single Q vector at read time.

## ANE Op Requirements

All ops verified in [ANE Op Support Matrix](ane-op-support-matrix.md):
- Quantize chain: `mul` → `round` → `clip` → `cast fp16→int8`
- Dequantize chain: `cast int8→fp16` → `mul` → `add`
- Hadamard rotation: `matmul` with const rotation matrix
- Attention: `matmul` (Q×K), `softmax`, `matmul` (×V)
- Cache shapes: `reshape`, `slice_by_index`, `tile`

## Key Constraints

- **INT4 is NOT feasible** via the MIL text path - comprehensively rejected
  by ANE compiler. See [op matrix](ane-op-support-matrix.md#int4-comprehensively-rejected).
- **INT8 is a storage format, not compute** - all arithmetic must be in fp16.
  Cast to int8 for bandwidth, cast back for math.
- **Per-program weight budget** - ~8 MB fp16 per sub-program. Adding compute
  ops to programs at this limit causes compiler failure. See [ANE Constraints](ane-constraints.md).
- **Cache-write fusion into pre_attn is infeasible** at Qwen3-0.6B scale due
  to the per-program resource limit.

## Open Issues

1. **Correctness validation** - no perplexity or token-agreement benchmarks
   exist yet. See `docs/development/QUALITY_BENCHMARK_PLAN.md`.
2. **Prefill path** - batch processing of prompt tokens not yet implemented for
   TurboQuant (only single-token decode exists).

## References

- [ANE Op Support Matrix](ane-op-support-matrix.md) - verified op set
- [ANE Inference](ane-inference.md) - inference pipeline status
- [ANE Constraints](ane-constraints.md) - hardware limits
- [TurboQuant Research Analysis](../research/turboquant-analysis.md) - paper background
- [TurboQuant via Orion](../research/turboquant-ane-orion.md) - private API feasibility (not current path)
