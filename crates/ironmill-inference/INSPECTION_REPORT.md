# ironmill-inference Deep Inspection Report

**Date:** 2026-04-07  
**Scope:** Full crate — 32K lines Rust + 6.5K lines Metal shaders  
**Compiler warnings (baseline):** 26

---

## Critical Bugs

### 1. GPU deadlock in `gdn_fused_decode` shader
**File:** `metal/shaders/gdn_recurrent.metal:520, 575, 583`  
**Dispatch:** `metal/ops.rs:1835-1839`

Threads where `tid >= v_head_dim` exit early via `return` at line 520, but the kernel has `threadgroup_barrier` calls at lines 575 and 583 that require *all* threads in the threadgroup to participate. The dispatch uses `max(channels_per_head, v_head_dim)` threads per threadgroup, so when `channels_per_head > v_head_dim`, the exited threads never reach the barriers → **GPU hang**.

**Fix:** Replace the early `return` with a conditional that lets all threads participate in barriers but only threads where `vi < v_head_dim` write results.

### 2. GDN hardcoded epsilon ignores model config
**File:** `metal/gdn.rs:314, 526`

Both `encode_gdn_prefill` and `encode_gdn_decode` pass `1e-6f32` to the recurrent kernel instead of the model's `rms_norm_eps`. Models with different epsilon values will produce incorrect outputs.

### 3. `fused_residual_norm` threadgroup buffer overflow
**File:** `metal/shaders/fused_residual_norm.metal:144`

```metal
threadgroup half tg_input[4096]; // supports hidden_size up to 4096
```

Hardcoded to 4096 elements with **no runtime guard**. Models with `hidden_size > 4096` will silently overflow the threadgroup buffer, causing memory corruption or GPU crash.

### 4. Global RoPE cache ignores `partial_rotary_factor`
**File:** `metal/loading.rs:103-106`

```rust
let (gc, gs) = build_rope_cache(&self.device, global_hd, global_hd, ...);
```

Passes `global_hd` for both `head_dim` and `rotary_dim`, so `global_cfg.partial_rotary_factor` is checked in the condition but never applied to the actual cache computation. Models with partial RoPE on global attention layers get incorrect positional encodings.

### 5. Grammar/stop termination drops the final token
**File:** `generate.rs:321-331`

When grammar completion or stop-token detection triggers, the sampled token has been pushed to `generated_tokens` but the iterator returns `Finished` without first emitting a `Token` event. The final valid token is lost from the output stream.

### 6. `load_weight_for_gather` can produce packed-only buffers
**File:** `weight_loading.rs:401-404`, `metal/weights.rs:742-759`

After CPU dequant, `Dense { buf: None, packed: Some(...) }` can be produced. Gather/embedding consumers need row-major data, so packed-only results break lookups.

---

## Bugs (Medium Severity)

### 7. Unsafe `Vec<u8>` → `&[u16]` cast has unsound alignment
**File:** `metal/pipeline.rs:970-978`

`Vec<u8>::as_ptr()` only guarantees 1-byte alignment, but the cast to `*const u16` requires 2-byte alignment. The SAFETY comment claims Metal readback alignment, but this is not enforced by the type system. Technically UB.

### 8. CPU conv1d in GDN skips current token
**File:** `metal/gdn.rs:648-667`

The conv1d loop uses `state_idx = k - (kernel_size - conv_state_width)`, where `conv_state_width = kernel_size - 1`. The newly appended sample at position `conv_state_width - 1` is included, but the indexing arithmetic means position `kernel_size - 1` (the current token's slot) maps to `state_idx = conv_state_width - 1` which is the *previous* token that was shifted into that position. The actual current value written at line 652 is never read by the kernel loop. Also panics if `kernel_size == 1` (underflow at `conv_state_width - 1 = -1`).

### 9. `prefill_with_cache` does not restore KV state
**File:** `engine.rs:124-145`

On a cache "hit", the engine resets and re-prefills the matched prefix instead of restoring cached KV activations. The `kv_slices` are fetched from the cache but never injected into the engine state — defeating the purpose of prefix caching.

### 10. `kv_cache_memory()` ignores KV quantization
**File:** `memory.rs:48-58`

The `_kv_quant` parameter is unused; memory estimation always assumes FP16 KV cache. Quantized KV cache models will have wildly overestimated memory requirements.

### 11. `calibrate_dac` batch_size=0 panics
**File:** `calibration/dataset.rs:85-87`

`iter_batches(0)` calls `chunks(0)`, which panics in the standard library.

### 12. Sampling `top_k` tie-breaking allows extra tokens
**File:** `sampling.rs:315-342`

`apply_top_k()` keeps *all* tokens tied at the k-th logit. When `top_k=1`, multiple tokens can survive, violating the contract.

### 13. `categorical_sample` silent fallback on empty distribution
**File:** `sampling.rs:408-420`

If filtering leaves no finite logits, the sampler silently falls back to the last token instead of returning an error. This masks bugs upstream (e.g., all logits filtered to -inf).

### 14. Grammar automaton panics on invalid token IDs
**File:** `grammar/automaton.rs:110-114`

`advance()` indexes `vocab[token_id as usize]` with no bounds check. Out-of-range token IDs from buggy tokenizers will panic.

### 15. Scheduler KV usage tracking is inconsistent
**File:** `serving/scheduler.rs:44-56, 104-115`

Pool allocations start with `used = 0` regardless of prompt length. `advance()` updates `pool.get_mut(id).used += 1` but never syncs `SequenceState.kv_allocation.used` or checks capacity overflow.

### 16. `d2quant_round_trip` and `offset_norm_weight` swallow errors
**File:** `metal/weights.rs:437-460, 473-487`

Read/write errors during weight transforms are silently ignored, so load-time quantization can fail without any indication.

### 17. MLA absorption uses unchecked integer division
**File:** `metal/mla/absorption.rs:66-77`

`hidden_size` is derived via division with only a `debug_assert!` guard. Release builds can silently use a truncated dimension.

### 18. MLA cache `advance_by` can overflow
**File:** `metal/mla/cache.rs:89-97`

Uses `self.seq_pos + count` without `checked_add`. Extremely long sequences could wrap.

### 19. Stale Gemma 4 state on model reload
**File:** `metal/loading.rs:65-75, 94-135`

`global_rope_*`, `global_head_dim`, `unit_norm_weight` are set during Gemma 4 loading but never cleared. Reloading a non-Gemma4 model can leave stale buffers.

### 20. BNF grammar silently drops malformed rules
**File:** `grammar/bnf.rs:151-162, 211-215`

Lines without `::=` are silently ignored. Rule bodies are not fully consumed, so trailing junk after a valid expression is accepted without error.

### 21. Grammar compiler allows duplicate rule names
**File:** `grammar/compiler.rs:76-89`

Later rules silently shadow earlier ones, leaving stale compiled data.

### 22. JSON schema conversion ignores `required`, `additionalProperties`
**File:** `grammar/json_schema.rs:138-213`

Object conversion ignores `required` fields, `additionalProperties`, and uniqueness constraints. Generated grammars accept invalid schemas.

### 23. Grammar mask `and_inplace` truncates silently
**File:** `grammar/mask.rs:68-71`

`and_inplace()` zips masks of different sizes and truncates to the shorter one instead of validating vocab size equality.

### 24. TurboQuant cache layout panics on edge cases
**File:** `turboquant/cache_layout.rs:67-74`

`next_power_of_two()` panics on zero-sized outlier/non-outlier splits. Layout hardcodes `/2` instead of using `outlier_bits`/`non_outlier_bits`.

### 25. TurboQuant outlier detection panics
**File:** `turboquant/outlier.rs:49-77`

Zero `out_features`/`head_dim` can panic, `n_outlier > head_dim` causes out-of-bounds slice, and code computes row norms despite docs saying column norms.

### 26. `NaN` in debug logging panics
**File:** `metal/pipeline.rs:120`

`partial_cmp(...).unwrap()` in logit debug logging panics on NaN values.

### 27. ANE tensor buffers silently truncate
**File:** `coreml/runtime.rs:49-63`, `ane/model.rs:536-563`

`chunks_exact()` silently truncates malformed tensor byte buffers; no length validation against shape/dtype.

---

## Unimplemented / Stub Code

| Location | Description |
|---|---|
| `jit.rs:75-96` | `with_int4()`, `with_fp16()`, `with_polar_quant()` — all no-op stubs that print warnings |
| `batch_runner.rs:63-89` | `submit()`, `step()`, `cancel()`, `has_pending()`, `active_count()` — all dummy/error returns |
| `metal/loading.rs:345-353` | `load_jit()` — always returns error "JIT loading not yet implemented" |
| `ane/decode.rs:792-805` | QK-norm weight loading — TODO stub |
| `ane/decode.rs:988-1010` | Metal↔ANE shared-event handoff — TODO stub |
| `ane/decode.rs:1034-1052` | Within-layer program chaining — TODO stub |
| `sampling.rs:85` | TODO comment about model-specific token metadata |

---

## Dead Code (Compiler-Confirmed + Manual)

| Location | Item |
|---|---|
| `metal/buffers.rs:252` | `MpsMatmulCache.layer_matmuls` — field never read |
| `metal/buffers.rs:255-264` | `LayerMatmuls` — all 8 fields never read |
| `metal/buffers.rs:452-487` | `read_buffer_f32`, `write_buffer_f32`, `read_weight_f32` — never called |
| `metal/calibration.rs:29, 895` | `calibrate_dac`, `prefill_calibration` — never called |
| `metal/engine.rs:145, 150` | `gpu_allocated_bytes`, `weights_mut` — never called |
| `metal/gdn.rs:43, 47` | `GdnState.gpu_raw_output`, `gpu_gdn_input` — fields never read |
| `metal/gdn.rs:562, 759, 771` | `run_gdn_layer_cpu`, `matvec`, `softplus` — never called |
| `metal/loading.rs:345` | `load_jit` — never called (also unimplemented) |
| `metal/mla/absorption.rs:145, 257` | `absorb_mla_weights`, `read_f16_buffer` — never called |
| `metal/mla/mod.rs:19` | Unused import `absorb_mla_weights` |
| `metal/pipeline.rs:33, 169, 202` | `prefill_all_logits`, `decode_step_with_hidden`, `last_hidden_state` — never called |
| `metal/plan.rs:21` | `AttentionKind::Gdn.gdn_index` — field never read |
| `metal/plan.rs:67-69` | `LayerPlan.has_post_ffn_norm/has_layer_scalar/has_ple` — never read |
| `metal/plan.rs:209` | `PlePlan.ple_hidden_size` — never read |
| `metal/plan.rs:214` | `ResidualStrategy` enum — never used |
| `metal/plan.rs:229-242` | `ModelPlan` — 10 fields never read |
| `metal/weights.rs:122-127` | `ProjectionMatmul::dense()` — never used |
| `metal/weights.rs:843-854` | `AffineQuantizedWeight.data_row_major` — never read |
| `dequant.rs:81-119, 185-226` | `dequant_affine_to_fp16`, `dequant_lut_to_fp16` — dead, `#[allow(dead_code)]` |
| `grammar/bnf.rs:205-208` | `Cursor::at_end()` — unused, `#[allow(dead_code)]` |
| Shader: `quantized_matmul.metal:14-16` | `TILE_M/N/K` — unused constants |
| Shader: `affine_matmul.metal:21-23` | `TILE_M/N/K` — unused constants |
| Shader: `attention.metal:19` | `SM` — unused constant |
| Shader: `fused_sdpa.metal:19` | `SM` — unused constant |

---

## Code Quality Issues

| Location | Issue |
|---|---|
| `metal/projection.rs:46-74` | `WeightBuffer::Dense` without packed buffer hard-panics instead of returning error |
| `metal/plan.rs:117-123` | GDN layer setup uses `expect()` — panics on config mismatch instead of error |
| `metal/engine.rs:150-153` | `weights_mut()` panics via `expect()` when called before `load()` |
| `metal/gdn.rs:56-75` | GDN state buffers allocated as `*4` bytes/element but read/written as FP16 (`*2`), doubling memory use |
| `metal/pipeline.rs:277-279` | `nkv`, `hd`, `inter` — computed but never used |
| `metal/pipeline.rs:328, 332` | `cmd_buf`, `enc` — needlessly marked `mut` |
| `metal/dequant.rs:231-245` | No `axis < shape.len()` validation; malformed metadata can panic |
| `shader_cache.rs:41-45, 68-72` | `max_size_bytes` never enforced; `key_path()` joins raw strings (potential path traversal) |
| `speculative/streaming.rs:151-165` | MSA weights silently truncated if malformed |
| `metal/ops.rs:1487-1490` | No guard that `num_q_heads % num_kv_heads == 0` or `num_kv_heads != 0` |

---

## Summary

| Category | Count |
|---|---|
| **Critical bugs** | 6 |
| **Medium bugs** | 21 |
| **Unimplemented stubs** | 7 |
| **Dead code items** | 30+ |
| **Code quality issues** | 10 |
| **Compiler warnings** | 26 |
