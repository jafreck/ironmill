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
  │ ops:     slice → cast(int8→fp16) → mul(deq) → sub(offset) → reshape → matmul(unrotate)
  │          → reshape → [GQA tile] → matmul(QK) → scale → softmax → matmul(AV) → reshape
  │ outputs: attn_out [1,q_ch,1,32]
  │
  ▼
post_attn (ANE)
```

**Per-layer overhead: 7 CPU data transfers, 2 FP16↔INT8 conversions, 3 staging buffers**
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
  ├─ CPU: read_column0_f16(K_quant)                ← read #1
  ├─ CPU: convert FP16 → INT8 bytes                ← compute
  ├─ CPU: write_bytes_at(k_cache, offset)           ← write #2
  ├─ CPU: (same for V)                              ← read #3, write #4
  │
  ▼
attention MIL (ANE) — unified emitter (AttentionMilConfig)
  │ inputs:  Q IOSurface, k_cache(INT8), v_cache(INT8), unrotation_matrix
  │ ops:     slice → cast(int8→fp16) → mul(deq_scale) → reshape → matmul(unrotate)
  │          → reshape → [GQA tile] → matmul(QK) → scale → softmax → matmul(AV) → reshape
  │ outputs: attn_out IOSurface
  │
  │ (no CPU copy — attn_out IOSurface passed directly to post_attn)
  │
  ▼
post_attn (ANE)
```

**Per-layer overhead: 4 CPU ops (2 reads + 2 writes for cache update), 0 staging buffers**
**Per-token (29 layers): 116 CPU ops (vs 203 before)**

## Status

| Item | Status | Impact |
|------|--------|--------|
| Staging buffer elimination | ✅ Done | −3 CPU copies/layer (87/token) |
| Unified attention MIL emitter | ✅ Done | Code dedup, no perf change |
| INT8 cache with ANE-inline cast | ✅ Done | 50% cache memory, +1 ANE cast op |
| Dequant chain simplification | ✅ Done | Removed `sub(offset)` + `cast` ops |
| Direct cache write (IOSurface aliasing) | ❌ Future | Would remove last 4 CPU ops/layer |

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

Per-layer, 4 CPU operations remain in `step_attention()`:

| # | Operation | Purpose |
|---|-----------|---------|
| 1 | `read_column0_f16(cw_k_output)` | Read quantized K from ANE output |
| 2 | `write_bytes_at(k_cache, offset)` | Write INT8 K to persistent cache |
| 3 | `read_column0_f16(cw_v_output)` | Read quantized V from ANE output |
| 4 | `write_bytes_at(v_cache, offset)` | Write INT8 V to persistent cache |

Plus a CPU FP16→INT8 conversion loop between reads and writes.

When QJL is enabled, one additional CPU read per layer:
`k_proj.read_f16()` to get original K values for residual sign computation.

### Phase D: Direct Cache Write (Future)

To eliminate the remaining 4 CPU ops, the cache-write MIL would need to
output directly into the correct offset of the persistent cache
IOSurface. This requires IOSurface sub-region aliasing (writing into
a sub-region of a larger IOSurface), which may not be supported by the
ANE runtime. Needs investigation.

## Performance Results

### Qwen3-0.6B E2E (128 tokens, max_seq=512)

| Metric | FP16 Baseline | TurboQuant INT8 |
|--------|---------------|-----------------|
| Throughput | 12.7 tok/s | 12.0 tok/s |
| KV cache size | 29.0 MB | 14.5 MB (50%) |
| CPU ops/layer | 0 | 4 |
| Staging buffers | 0 | 0 |

TurboQuant achieves 94% of FP16 baseline speed with 50% KV cache memory.

### Synthetic Benchmark (Qwen3-0.6B arch, dummy weights)

| Metric | Previous | Current |
|--------|----------|---------|
| CPU data transfers/layer | 7 | 4 |
| CPU format conversions/layer | 2 | 1 (FP16→INT8) |
| Staging buffers | 3 | 0 |
| KV cache (512 seq) | 14.5 MB | 14.5 MB |
| Per-layer latency | — | 366 μs |
| Per-token (28 layers) | — | 10.26 ms |

## References

- `emit_attention_mil` — unified attention MIL emitter (TQ + FP16)
- `AttentionMilConfig` — configuration struct for the unified emitter
- `emit_cache_write_mil` — cache-write MIL (rotate + quantize)
- `step_attention` in `turboquant.rs` — current data path
- `compile_and_load_sub` in `inference.rs` — `min_output_alloc` param
- FP16 decode path in `inference.rs` — zero-copy comparison target
