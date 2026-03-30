# TurboQuant Zero-Copy Data Path

## Previous Architecture (CPU-heavy)

Each layer's TurboQuant attention required 7 CPU data transfers,
2 CPU format conversions, and 3 staging buffers:

```
pre_attn (ANE)
  │ outputs: Q [1,2048,1,32], K [1,1024,1,32], V [1,1024,1,32]
  │
  ├─ CPU: copy_column0_from(K → cw_k_staging)     ← memcpy #1
  ├─ CPU: copy_column0_from(V → cw_v_staging)     ← memcpy #2
  │
  ▼
cache-write MIL (ANE) — handcrafted
  │ inputs:  cw_k_staging, cw_v_staging, rotation_matrix
  │ ops:     reshape → matmul(rotate) → mul(scale) → round → clip → cast(int8) → cast(fp16)
  │ outputs: K_quant [1,kv_ch,1,32], V_quant [1,kv_ch,1,32]
  │
  ├─ CPU: read_column0_f16(K_quant)                ← memcpy #3
  ├─ CPU: convert FP16 → INT8 bytes                ← CPU compute
  ├─ CPU: write_bytes_at(k_cache, offset)           ← memcpy #4
  ├─ CPU: (same for V)                              ← memcpy #5
  │
  ├─ CPU: copy_column0_from(Q → attn_q_staging)    ← memcpy #6
  │
  ▼
attention MIL (ANE) — handcrafted (separate from FP16 emitter)
  │ inputs:  attn_q_staging, k_cache(INT8), v_cache(INT8), unrotation_matrix
  │ ops:     slice → cast(int8→fp16) → mul(deq) → sub(offset) → reshape → matmul(unrotate K)
  │          → matmul(unrotate V) → reshape → [GQA tile] → matmul(QK) → scale → softmax
  │          → matmul(AV) → reshape
  │ outputs: attn_out [1,q_ch,1,32]
  │
  ▼
post_attn (ANE)
```

**Per-layer overhead: 7 CPU data transfers, 2 FP16↔INT8 conversions, 3 staging buffers**
**Per-layer ANE: 2× O(seq_len × head_dim²) un-rotation matmuls on K and V caches**
**Per-token (29 layers): 203 CPU data transfers**

## Current Architecture (Implemented)

```
pre_attn (ANE)
  │ outputs: Q, K_proj, V_proj — allocated with tq_alloc_size
  │
  │ (no CPU copy — IOSurfaces passed directly via unified alloc size)
  │
  ▼
cache-write MIL (ANE) — handcrafted
  │ inputs:  K_proj IOSurface, V_proj IOSurface, rotation_matrix
  │ ops:     reshape → matmul(rotate) → mul(scale) → round → clip → cast(int8) → cast(fp16)
  │ outputs: K_quant [1,kv_ch,1,32], V_quant [1,kv_ch,1,32] as FP16
  │
  ├─ copy_column0_fp16_as_int8_to(K_quant → k_cache)  ← direct IOSurface copy
  ├─ copy_column0_fp16_as_int8_to(V_quant → v_cache)  ← direct IOSurface copy
  │   (single locked pass per tensor, zero heap allocations)
  │
  ▼
attention MIL (ANE) — unified emitter (AttentionMilConfig), Q-rotation
  │ inputs:  Q IOSurface, k_cache(INT8), v_cache(INT8), rotation_matrix
  │ ops:     matmul(rotate Q) ← O(head_dim²), constant per token
  │          → slice → cast(int8→fp16) → mul(deq_scale) → reshape
  │          → [GQA tile] → matmul(QK) → scale → softmax → matmul(AV)
  │          → matmul(un-rotate output) ← O(head_dim²), constant per token
  │          → reshape
  │ outputs: attn_out IOSurface
  │
  │ (no CPU copy — attn_out IOSurface passed directly to post_attn)
  │
  ▼
post_attn (ANE)
```

**Per-layer overhead: 2 direct IOSurface copies (K + V, zero-alloc), 0 staging buffers**
**Per-layer ANE: 2× O(head_dim²) rotation matmuls (constant, independent of seq_len)**
**Per-token (29 layers): 58 direct copies (vs 116 CPU ops before, vs 203 originally)**

## Status

| Item | Status | Impact |
|------|--------|--------|
| Staging buffer elimination | ✅ Done | −3 CPU copies/layer (87/token) |
| Unified attention MIL emitter | ✅ Done | Code dedup, no perf change |
| INT8 cache with ANE-inline cast | ✅ Done | 50% cache memory, +1 ANE cast op |
| Dequant chain simplification | ✅ Done | Removed `sub(offset)` + `cast` ops |
| Direct IOSurface copy (zero-alloc) | ✅ Done | −2 allocs/layer, −2 lock cycles/layer |
| Q-rotation (O(1) vs O(seq_len)) | ✅ Done | Eliminated 2× O(seq_len) matmuls |
| Merge cache-write into pre_attn | ❌ Future | Would eliminate 1 ANE eval call/layer |
| Direct cache write (IOSurface aliasing) | ❌ Infeasible | ANE can't write to IOSurface sub-regions |

## Changes Made

### Phase B+C: Unified Alloc + Staging Elimination + Unified MIL (Implemented)

- Single `tq_alloc_size` computed across all cache-write and attention
  input tensors. Pre-attn outputs allocated with this size via
  `min_output_alloc` parameter in `compile_and_load_sub()`.
- Staging buffers (`cw_k_staging`, `cw_v_staging`, `attn_q_staging`) removed.
- `step_attention()` passes `k_proj`, `v_proj`, `q` IOSurfaces directly
  to ANE eval calls.
- `emit_attention_mil` and `emit_fp16_attention_mil` merged into a single
  parameterized function via `AttentionMilConfig`. FP16 wrapper retained
  for backward compatibility.

### Direct IOSurface Copy (Implemented)

Replaced the `read_column0_f16` → `Vec<f16>` → convert → `Vec<u8>` →
`write_bytes_at` cache update path with `copy_column0_fp16_as_int8_to`,
a single-pass IOSurface-to-IOSurface strided copy with FP16→INT8
conversion.

Eliminates per-layer:
- 2 heap allocations (`Vec<f16>` + `Vec<u8>` per K and V)
- 2 extra lock/unlock cycles (combined read+write into 1 locked pass)

`KvCacheManager::update_cache_direct()` wraps this for the cache update
path, replacing the old `update_cache()` call in `step_attention()`.
QJL residual sign computation (when enabled) still requires a CPU
read-back from the cache.

### Q-Rotation: O(1) Per Token Instead of O(seq_len) (Implemented)

Replaced the two O(seq_len × head_dim²) un-rotation matmuls (which
un-rotated the entire K and V cache every token) with two O(head_dim²)
matmuls: rotate Q before the QK attention matmul, and un-rotate the
attention output after the AV matmul.

**Mathematical equivalence:**
- K is stored rotated in cache: `K_stored = R · K`
- Old approach: un-rotate K cache → `K_approx = R⁻¹ · K_dequant`, then `Q · K_approx^T`
- New approach: rotate Q → `Q_rot = R · Q`, then `Q_rot · K_dequant^T`
- Since R is orthogonal: `⟨R·Q, K_dequant⟩ = ⟨Q, R⁻¹·K_dequant⟩` (identical scores)
- For V: output is `softmax(scores) · (R·V)^T`, un-rotate via `attn_pre · R`

This is consistent with the TurboQuant paper's formulation — the paper
defines quantize/dequantize as abstract primitives that preserve inner
products. Where the inverse rotation is applied is an implementation
choice; the distortion bounds and quantization quality are identical.

The attention program now uses the **same rotation matrix** as the
cache-write program (not the inverse), eliminating the `unrotation_tensor`
field from `TurboQuantModel`.

### INT8 Cache with ANE-Inline Cast (Implemented)

The original spec proposed switching the KV cache from INT8 to FP16 to
avoid CPU format conversions (Phase A). This was rejected because it
doubles cache memory, negating TurboQuant's core benefit.

Instead, the cache remains INT8 (1 byte/element) and the attention MIL
program accepts INT8 function inputs directly. The `cast(int8→fp16)` op
runs on the ANE inline before dequantization. This was verified
empirically — ANE compiles and evaluates INT8 function inputs without
error, contrary to the earlier assumption in `ane-op-support-matrix.md`
that function inputs were limited to fp16/fp32/bool.

**Key correction:** `ane-op-support-matrix.md` stated that ANE rejects
INT8 function inputs. This is incorrect — INT8 function inputs compile
and eval successfully. The restriction applies only to INT8 function
*outputs* (ANE rejects those, which is why cache-write casts INT8→FP16
before outputting).

### Dequant Chain Simplification (Implemented)

The `sub(deq_offset)` op was removed from the dequant pipeline (offset
is always 0 for symmetric quantization). The old separate
`emit_dequantize_chain` function was inlined into the unified emitter.

## Remaining CPU Overhead

Per-layer, 2 direct IOSurface copy operations remain in `step_attention()`:

| # | Operation | Purpose |
|---|-----------|---------|
| 1 | `copy_column0_fp16_as_int8_to(cw_k → k_cache)` | Direct strided copy + conversion |
| 2 | `copy_column0_fp16_as_int8_to(cw_v → v_cache)` | Direct strided copy + conversion |

These are zero-allocation, single-locked-pass operations (vs the previous
4 separate CPU ops + 2 heap allocations per K/V pair).

When QJL is enabled, one additional CPU read per layer:
`k_cache.read_bytes_at()` to get quantized K values for residual sign computation,
plus `k_proj.read_f16()` to get original K values.

## Performance Results

### Qwen3-0.6B E2E (128 tokens, max_seq=512)

| Metric | FP16 Baseline | TQ (K/V un-rotation) | TQ (Q-rotation) |
|--------|---------------|----------------------|-----------------|
| Throughput | ~13–14 tok/s | 11.6 tok/s | 12.3 tok/s |
| KV cache size | 29.0 MB | 14.5 MB (50%) | 14.5 MB (50%) |
| CPU ops/layer | 0 | 2 (zero-alloc) | 2 (zero-alloc) |
| ANE rotation/layer | 0 | 2× O(seq_len) | 2× O(1) |

### Qwen3-0.6B E2E (512 tokens, max_seq=2048)

| Metric | FP16 Baseline | TQ (K/V un-rotation) | TQ (Q-rotation) |
|--------|---------------|----------------------|-----------------|
| Throughput | ~14–15 tok/s | 10.3 tok/s | 10.2 tok/s |
| KV cache size | 116.0 MB | 58.0 MB (50%) | 58.0 MB (50%) |

At long sequences, TQ throughput is limited by the extra ANE eval call
(cache-write program) and per-element cast + dequant ops on the full cache,
not by the rotation. The Q-rotation eliminated the O(seq_len) bottleneck
but the remaining fixed overhead still dominates for this small model.

## Performance Analysis: Why TQ Is Still Slower

TurboQuant adds these costs compared to FP16 baseline, per layer per token:

| Cost | Nature | Scaling |
|------|--------|---------|
| Cache-write ANE eval | Extra eval() call | **Fixed ~50μs** |
| `cast(int8→fp16)` on cache | On-chip ANE op | O(seq_len) |
| `mul(deq_scale)` on cache | On-chip ANE op | O(seq_len) |
| Q rotation matmul | On-chip ANE op | O(head_dim²) — **constant** |
| Output un-rotation matmul | On-chip ANE op | O(head_dim²) — **constant** |
| IOSurface copy K+V | CPU, zero-alloc | O(kv_channels) — **constant** |

The bandwidth savings from INT8 (reading half the cache bytes) must
exceed these combined costs for TQ to outperform FP16. For Qwen3-0.6B
(small KV cache: 8 KV heads × 64 dim = 512 channels), the cache is too
small to be bandwidth-bound even at seq_len=2048.

**Expected crossover conditions:**
- Larger model (more KV channels → bigger cache → more bandwidth savings)
- Longer context (seq_len=4096+ where cache read dominates)
- Both combined (e.g., 7B model at seq_len=4096: ~928 MB FP16 cache)

## Next Steps for Performance

### Priority 1: Merge cache-write into pre_attn

The cache-write program (rotate + quantize K/V) runs as a separate ANE
eval call per layer. This adds ~50μs × 29 layers = ~1.45ms per token of
fixed overhead. Fusing the rotation + quantization ops into the pre_attn
sub-program would eliminate 29 eval calls per token.

**Approach:** After the attention split, append the cache-write ops
(reshape → matmul(rotate) → scale → round → clip → cast) to each layer's
pre_attn sub-program. The pre_attn would then output Q (fp16) and
K_quant/V_quant (fp16 with INT8-range values) directly.

**Challenge:** The pre_attn sub-program is compiled from the model's
MIL IR via structural splitting. Injecting additional ops requires
modifying `compile_and_load_sub()` or the split pass to append the
cache-write pipeline to the pre_attn graph.

### Priority 2: Test with larger models

Qwen3-0.6B has a small KV cache (512 KV channels). The bandwidth savings
from INT8 are proportionally small. Testing with a 7B+ model (e.g.,
Qwen3-4B or Llama-7B) where the KV cache is 10–20× larger would better
demonstrate TQ's value at long sequences.

### Priority 3: INT4 packed in INT8

Store 2 quantized values per INT8 byte. Unpack with ANE ops. This would
halve cache memory again (4× vs FP16) with some additional ANE ops.
Native INT4 (Orion-style weight descriptor manipulation) would be
ideal but requires deep ANE compiler integration.

## Orion Comparison Notes

Investigation of [Orion](https://github.com/mechramc/Orion) (direct ANE
runtime, no TurboQuant) confirmed:

- Orion also uses CPU-side copies for KV cache updates — no project has
  achieved true zero-copy ANE→cache writes
- Orion does **CPU-side attention during decode** (not ANE attention),
  so its KV cache layout is unconstrained
- IOSurface sub-region aliasing (writing ANE output into a specific
  offset of a larger IOSurface) is not supported by the ANE runtime
- The shared pattern: IOSurface as flat byte buffer (`Width=bytes,
  Height=1, BytesPerElement=1`) with CPU↔ANE unified memory

## References

- `copy_column0_fp16_as_int8_to` — zero-alloc IOSurface-to-IOSurface copy
- `update_cache_direct` — direct cache update from ANE output tensors
- `emit_attention_mil` — unified attention MIL emitter (TQ + FP16)
- `AttentionMilConfig` — configuration struct for the unified emitter
- `emit_cache_write_mil` — cache-write MIL (rotate + quantize)
- `step_attention` in `turboquant.rs` — current data path
- `compile_and_load_sub` in `inference.rs` — `min_output_alloc` param
- FP16 decode path in `inference.rs` — zero-copy comparison target
- TurboQuant paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
