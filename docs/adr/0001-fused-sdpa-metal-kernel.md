# ADR-0001: Fused Scaled Dot-Product Attention (SDPA) Metal Kernel

**Status:** Proposed  
**Date:** 2026-04-04

## Context

The Metal GPU inference backend dispatches attention as one threadgroup per (head, query_token), where each threadgroup iterates over KV cache tiles sequentially. This has two compounding problems at long sequence lengths:

1. **O(n²) threadgroup count** — at 8192 tokens × 16 heads = 131K threadgroups, each with variable work (1–256 KV tiles per query due to causal masking). The GPU cannot balance this load.

2. **Separated QKᵀ → softmax → V passes** — each KV tile is loaded from global memory twice (once for K, once for V), and intermediate scores are stored in threadgroup SRAM between phases.

Measured impact on Qwen3-0.6B prefill (Metal FP16 vs HuggingFace CPU FP32):

| Sequence length | Metal GPU | HF CPU | GPU vs CPU |
|-----------------|-----------|--------|------------|
| 1024            | 1287 tok/s | 1104 tok/s | **+17%** |
| 2048            | 778 tok/s | 1073 tok/s | −28% |
| 4096            | 418 tok/s | 995 tok/s | −58% |
| 8192            | 214 tok/s | 793 tok/s | −73% |

GPU throughput degrades ~6× from 1K→8K tokens while CPU degrades only ~1.4×. HF CPU uses PyTorch's `scaled_dot_product_attention`, which dispatches a fused FlashAttention kernel that tiles over both Q and KV dimensions in a single cache-friendly pass.

## Decision

Implement a fused SDPA Metal compute kernel that tiles over **both Q blocks and KV blocks** within a single threadgroup, matching the FlashAttention-2 tiling strategy adapted for Apple Silicon's SIMD group matrix hardware.

### Kernel design

**Grid:** `(num_heads, num_q_blocks)` threadgroups, where `num_q_blocks = ceil(seq_len / Q_BLOCK)`.

**Per threadgroup:**
```
for each KV tile (kv_start = 0 .. causal_limit .. KV_TILE):
    Load K[kv_start .. kv_start+KV_TILE] into SRAM           // one global read
    Compute S = Q_block × Kᵀ via simdgroup_matrix_multiply    // in SRAM
    Apply causal mask to S
    Online softmax update (per-query running max + sum)
    Load V[kv_start .. kv_start+KV_TILE] into SRAM           // reuse K memory
    Accumulate O += softmax_weights × V via simdgroup_matrix  // in SRAM
Normalize O by softmax denominators
Write O to global memory                                      // one global write
```

**Key properties:**
- K and V tiles loaded once from global memory per KV block (not per query)
- Q block stays in SRAM/registers across all KV iterations
- `simdgroup_matrix_multiply_accumulate` for both QKᵀ and weighted-V accumulation
- Online softmax — no materialized attention matrix
- Causal mask applied as −∞ scores before softmax (zero-cost in the exp)

### Tiling parameters (HEAD_DIM=128)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Q_BLOCK | 64 | Fills simdgroup matrix tiles (8 SGs × 8 rows) |
| KV_TILE | 32 | Matches SIMD width; fits K+V in SRAM |
| Threads/TG | 256 | 8 simdgroups for parallel Q×K dot products |

SRAM budget per threadgroup:
- Q block: 64 × 128 × 2 = 16 KB (half, persistent across KV loop)
- KV tile: 32 × 128 × 2 = 8 KB (half, aliased K then V)
- Output: 64 × 128 × 4 = 32 KB... **exceeds 32 KB limit**

Resolution options:
- **A)** Reduce Q_BLOCK to 32 → output = 16 KB, total = 40 KB... still tight. Store Q in half (8 KB) + output in half with float accumulator in registers. Total SRAM: 8 + 8 + per-query state ≈ 18 KB. ✓
- **B)** Use device memory for output accumulator (write-combine pattern). Q and KV in SRAM only. Requires atomic-free accumulation since each threadgroup owns its Q block exclusively.
- **C)** Use Metal's `setThreadgroupMemoryLength` to request exact SRAM budget at dispatch time, allowing runtime Q_BLOCK selection based on HEAD_DIM.

Recommended: Option A with Q_BLOCK=32, Q stored in half, output accumulated in float registers (each thread owns a fixed set of output elements across the Q block).

### Integration

- New kernel function: `fused_sdpa` in `attention.metal`
- New pipeline: `MetalPipelines::fused_sdpa`
- Dispatch from `encode_kv_cache_and_attention` when `token_count > 1` (prefill)
- Decode (token_count=1) continues using `standard_attention`
- Gated by `MetalConfig::use_fused_sdpa` (default: true for prefill)
- The existing `prefill_attention` and `prefill_attention_fa2` kernels remain as fallbacks

## Alternatives Considered

### 1. Keep current per-query-threadgroup kernel
- **Pro:** Simple, correct, fast at short sequences (≤1024 tokens)
- **Con:** O(n²) scaling, 73% slower than CPU at 8192 tokens
- **Verdict:** Unacceptable for production use with sequences >1K

### 2. FA2 multi-query kernel (already implemented, `use_fa2_prefill`)
- **Pro:** Better KV tile reuse, fewer threadgroups
- **Con:** Tested slower than per-query kernel for 0.6B model (500 vs 767 tok/s at 1024 tokens). Scores array and per-query softmax bookkeeping in SRAM adds overhead. V accumulation loop is triple-nested and serialized.
- **Verdict:** Wrong tiling granularity — tiles over KV but processes Q serially within SRAM. A proper fused kernel should tile over Q in simdgroup matrix tiles.

### 3. MPS `MPSGraph` attention
- **Pro:** Apple-optimized, handles FlashAttention internally
- **Con:** Requires MPSGraph framework integration (not wrapped in metal-sys). Black-box optimization; can't fuse with custom quantized KV cache paths. Would need separate code path from the compute-shader pipeline.
- **Verdict:** High integration cost, doesn't compose with TurboQuant

### 4. Use Metal Performance Shaders `MPSMatrixMultiplication` for QKᵀ
- **Pro:** Apple-optimized GEMM
- **Con:** Materializes the full `[seq_len × seq_len]` attention matrix in global memory (128 MB at 8192 tokens, FP16). Doesn't fuse softmax or V accumulation. Multiple kernel launches.
- **Verdict:** Memory-bound and unfusable; strictly worse than in-SRAM tiling

## Consequences

### Positive
- Prefill throughput at 8192 tokens expected to improve 3–5× (target: >800 tok/s, matching or exceeding CPU)
- Uniform work distribution across threadgroups — no causal-mask load imbalance
- Foundation for GQA-aware tiling (multiple Q heads share KV tile loads)
- Composes with TurboQuant: replace K/V tile loads with quantized cache reads in future variant

### Negative
- Complex kernel (~200 lines of Metal) with subtle correctness requirements (online softmax, causal masking, simdgroup matrix layout)
- HEAD_DIM-specific SRAM tuning — different Q_BLOCK for HEAD_DIM=64 vs 128 vs 256
- Decode path (token_count=1) doesn't benefit — keeps existing kernel
- Requires thorough numerical validation (PPL must match reference exactly)

### Neutral
- No API changes — internal kernel selection, transparent to callers
- Existing kernels remain as fallbacks for debugging or unsupported HEAD_DIM values

## Open Questions

1. **Register pressure:** Can each thread hold its share of the 32×128 float output accumulator in registers without spilling? Apple Silicon has 96 32-bit registers per thread — at 256 threads/TG with Q_BLOCK=32, each thread owns 32×128/256 = 16 float registers for output. Feasible.

2. **GQA optimization:** With num_kv_heads < num_heads, multiple Q heads share the same KV tile. The kernel should exploit this by loading KV once and processing all grouped Q heads. This changes the grid to `(num_kv_heads, num_q_blocks)` with an inner loop over the GQA group.

3. **Interaction with TurboQuant:** The quantized KV cache stores compressed data. A fused SDPA variant would dequantize KV tiles on-the-fly in SRAM. This is a follow-up kernel, not part of the initial implementation.

4. **Benchmark target:** Is matching HF CPU throughput sufficient, or should we target MLX-level performance (~2000+ tok/s for 0.6B at 8K)?
