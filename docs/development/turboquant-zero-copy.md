# TurboQuant Zero-Copy Data Path

## Current Architecture (CPU-heavy)

Each layer's TurboQuant attention requires 6 CPU memcpy operations and
2 handcrafted MIL programs with staging buffers:

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
attention MIL (ANE) — handcrafted
  │ inputs:  attn_q_staging, k_cache(INT8), v_cache(INT8), unrotation_matrix
  │ ops:     slice → cast(int8→fp16) → mul(deq) → reshape → matmul(unrotate)
  │          → reshape → [GQA tile] → matmul(QK) → scale → softmax → matmul(AV) → reshape
  │ outputs: attn_out [1,q_ch,1,32]
  │
  ├─ CPU: read_column0_f16(attn_out)               ← memcpy #7
  │
  ▼
post_attn (ANE)
```

**Per-layer overhead: 7 CPU memcpys, 2 FP16↔INT8 conversions**
**Per-token: 29 × 7 = 203 CPU memcpys**

## Target Architecture (Zero-Copy)

Eliminate all CPU memcpys by passing IOSurfaces directly between ANE
programs, matching the FP16 baseline's data flow pattern:

```
pre_attn (ANE)
  │ outputs: Q [1,2048,1,32], K [1,1024,1,32], V [1,1024,1,32]
  │
  │ (no CPU copy — pre_attn output IOSurfaces passed directly)
  │
  ▼
cache-write MIL (ANE) — handcrafted, revised
  │ inputs:  pre_attn K output IOSurface, pre_attn V output IOSurface, rotation_matrix
  │ ops:     unchanged (reshape → rotate → quantize → clip → cast)
  │ output:  K_quant, V_quant as FP16 (values are integer-rounded, stored as FP16)
  │
  │ NOTE: output IOSurface IS the KV cache slice for this position.
  │       The MIL program writes to a pre-allocated output tensor that
  │       occupies the correct offset in the cache IOSurface.
  │       OR: CPU does one write_f16_at per tensor (2 writes, not 7).
  │
  ▼
attention MIL (ANE) — reuse emit_fp16_attention_mil pattern
  │ inputs:  Q (pre_attn output IOSurface), K_cache, V_cache
  │ ops:     slice_by_index → mul(deq_scale) → reshape → [GQA tile]
  │          → matmul(QK) → scale → softmax → matmul(AV) → reshape
  │ output:  attn_out IOSurface
  │
  │ (no CPU copy — attn_out IOSurface passed directly to post_attn)
  │
  ▼
post_attn (ANE)
```

**Per-layer overhead: ~2 CPU writes (cache update), 0 format conversions**
**Per-token: 29 × 2 = 58 CPU writes (vs 203 memcpys before)**

## Key Design Changes

### 1. FP16 Cache Format (Not INT8)

**Current:** KV cache stores raw INT8 bytes. CPU converts FP16→INT8
after cache-write ANE eval, and the attention MIL casts INT8→FP16
before dequantization.

**Proposed:** KV cache stores FP16 values that are integer-rounded
to the INT8 range [-128, 127]. The cache-write MIL already produces
these (it casts to INT8 then back to FP16 because ANE rejects INT8
outputs). Simply store the FP16 output directly.

**Memory impact:** Cache doubles from 1 byte/element (INT8) to
2 bytes/element (FP16). For Qwen3-0.6B with max_seq=512:
- Current: 14.5 MB (INT8)
- Proposed: 29.0 MB (FP16-quantized) — same as FP16 baseline

This trades memory for speed. The quantization still provides
quality benefits (rotation + Beta-optimal levels) even though storage
is FP16. If memory is critical, a future optimization can use the
ANE's dequantize op to convert INT8 cache → FP16 on-the-fly during
attention, but that adds complexity.

### 2. Uniform Alloc Sizes

The core ANE constraint: all input IOSurfaces in one eval must have
the same `alloc_size`. Currently, staging buffers exist solely to
match alloc sizes between pre_attn outputs and TQ program inputs.

**Fix:** Compute a global uniform alloc size across ALL programs
(pre_attn, cache-write, attention, post_attn) and use it for every
IOSurface. This is wasteful for small tensors (the rotation matrix
is `[1,1,64,64]` but gets a huge alloc) but eliminates all staging
copies.

Alternatively, keep per-program alloc sizes but ensure pre_attn
outputs match cache-write inputs. Since pre_attn output shapes are
model-determined (not configurable), the cache-write MIL's input
shapes must match. This requires emitting cache-write MIL with
shapes matching the actual pre_attn output shapes.

### 3. Attention MIL: Merge FP16 and TQ Patterns

The FP16 attention MIL (`emit_fp16_attention_mil`) and TurboQuant
attention MIL (`emit_attention_mil`) differ only in:
- FP16: reads FP16 cache directly
- TQ: reads INT8 cache → cast → mul(deq_scale) → sub(offset) → matmul(unrotate)

With FP16 cache format, the TQ attention path becomes:
- Read FP16-quantized cache → mul(deq_scale) → [skip unrotation if handled differently]

This can be a single parameterized function:
```rust
pub fn emit_attention_mil(
    config: AttentionMilConfig,
) -> String {
    // If dequant_scale is Some, add mul(deq_scale) after cache read
    // If unrotation is needed, add matmul(unrotation_matrix)
    // Core attention is identical: reshape → tile → QK → scale → softmax → AV → reshape
}
```

### 4. Cache Write: Direct to Cache IOSurface

**Current:** cache-write MIL outputs to a temporary IOSurface, CPU
reads it, converts to INT8, writes to persistent cache at offset.

**Proposed:** cache-write MIL outputs FP16-quantized values. The
decode loop writes these directly to the persistent FP16 cache
IOSurface at the correct position:

```rust
// After cache-write ANE eval produces K_quant, V_quant as FP16:
let k_quant_f16 = cw_k_output.read_f16()?;  // still 1 read
caches[layer].0.write_f16_at(seq_pos * kv_ch, &k_quant_f16)?;  // 1 write
// Total: 2 IOSurface operations per K (was 4: read + convert + write_bytes + staging)
```

Or even better: if the cache-write MIL can output directly into a
sub-region of the cache IOSurface (via shared IOSurface backing),
even these reads/writes are eliminated. This requires IOSurface
aliasing which may not be practical.

### 5. Q Passthrough (No Staging)

The Q projection from pre_attn goes directly to the attention MIL.
Both FP16 and TQ attention take Q as input. If alloc sizes match,
the pre_attn Q output IOSurface IS the attention Q input — zero copy.

## Implementation Phases

### Phase A: FP16 Cache + Eliminate INT8 Conversion
- Change `KvCacheManager` to store FP16 (ScalarType::Float16)
- Cache-write MIL output stays as-is (already FP16)
- Remove CPU INT8 conversion in `step_attention`
- Write FP16-quantized output directly to FP16 cache
- Update attention MIL to skip INT8→FP16 cast (already FP16)
- **Impact:** Removes 2 CPU format conversions per layer (58/token)

### Phase B: Eliminate Staging Copies
- Compute uniform alloc size across pre_attn + cache-write + attention
- Allocate pre_attn output tensors with this alloc size
- Pass pre_attn output IOSurfaces directly to cache-write MIL inputs
- Pass pre_attn Q output directly to attention MIL input
- Remove `cw_k_staging`, `cw_v_staging`, `attn_q_staging`
- **Impact:** Removes 3 copy_column0_from calls per layer (87/token)

### Phase C: Unify Attention MIL
- Merge `emit_attention_mil` and `emit_fp16_attention_mil` into
  a single parameterized function
- TQ path: `emit_attention_mil(dequant: Some(deq_scale), unrotate: Some(matrix))`
- FP16 path: `emit_attention_mil(dequant: None, unrotate: None)`
- Remove duplicated MIL generation code
- **Impact:** Code simplification, no performance change

### Phase D: Direct Cache Write (Future)
- Investigate IOSurface sub-region aliasing
- If possible: cache-write MIL outputs directly into cache IOSurface
  at the correct position offset
- Eliminates the remaining 2 read/write operations per layer
- **Impact:** True zero-copy — all data stays on IOSurfaces

## Performance Expectations

| Metric | Current | After A+B | After A+B+C |
|--------|---------|-----------|-------------|
| CPU memcpys/layer | 7 | 2 | 2 |
| Format conversions/layer | 2 | 0 | 0 |
| Staging buffers | 3 | 0 | 0 |
| KV cache size (512 seq) | 14.5 MB | 29.0 MB | 29.0 MB |
| Expected TQ tok/s | 12.0 | 14-16 | 14-16 |
| FP16 tok/s (unchanged) | 13.1 | 13.1 | 13.1 |

After Phases A+B, TurboQuant should match or exceed FP16 throughput
because the INT8-range quantized cache has the same memory footprint
(stored as FP16) but the attention computation operates on
pre-quantized values that may have better numerical properties for
the ANE's FP16 compute units.

## Tradeoffs

**Memory:** FP16 cache format doubles KV cache memory (14.5→29 MB).
TurboQuant's memory advantage is lost for KV cache, though the
quantization quality benefits (rotation + Beta-optimal) remain.

To recover the memory advantage, a future Phase E could:
- Store INT8 in the persistent cache (off-ANE)  
- Emit an attention MIL that casts INT8→FP16 inline (the existing
  `cast` + `mul(deq_scale)` ops, already eval-verified)
- This requires INT8 IOSurface inputs to the attention MIL, which
  works because the cache has many elements (S=max_seq_len ≥ 32,
  satisfying the C/S constraint)

## References

- `emit_fp16_attention_mil` — the FP16 attention MIL pattern to adopt
- `emit_attention_mil` — current TQ attention MIL with dequant/unrotation
- `emit_cache_write_mil` — current TQ cache-write MIL
- `step_attention` in `turboquant.rs` — current CPU-heavy data path
- FP16 decode path in `inference.rs` — the zero-copy target pattern

---

## Implementation Details (Phase A+B)

### Files to change

- `crates/ironmill-ane/src/turboquant.rs` — main changes
- `crates/ironmill-ane/src/turboquant_mil.rs` — attention MIL update
- `crates/ironmill-ane/src/inference.rs` — decode loop wiring

### turboquant.rs: KvCacheManager

**Current** (line 161): caches use `ScalarType::Int8`
```rust
k_caches.push(AneTensor::new_with_min_alloc(channels, config.max_seq_len, ScalarType::Int8, min_alloc)?);
v_caches.push(AneTensor::new_with_min_alloc(channels, config.max_seq_len, ScalarType::Int8, min_alloc)?);
```

**Change to** `ScalarType::Float16`:
```rust
k_caches.push(AneTensor::new_with_min_alloc(channels, config.max_seq_len, ScalarType::Float16, min_alloc)?);
v_caches.push(AneTensor::new_with_min_alloc(channels, config.max_seq_len, ScalarType::Float16, min_alloc)?);
```

### turboquant.rs: update_cache

**Current** (line 226-266): takes `k_quantized: &[u8]`, `v_quantized: &[u8]`
and calls `write_bytes_at`.

**Change to** take `&[f16]` and call `write_f16_at`:
```rust
pub fn update_cache(
    &mut self,
    layer: usize,
    k_quantized: &[f16],  // was &[u8]
    v_quantized: &[f16],  // was &[u8]
    k_original: Option<&[f16]>,
) -> Result<()> {
    // ...
    let elem_offset = self.seq_pos * token_elements;
    self.k_caches[layer].write_f16_at(elem_offset, k_quantized)?;  // was write_bytes_at
    self.v_caches[layer].write_f16_at(elem_offset, v_quantized)?;
    // ...
}
```

### turboquant.rs: TurboQuantModel::compile

**Current alloc sizes** (lines 452-465):
```rust
let cw_alloc_size = uniform_alloc_size(&[
    ([1, kv_ch, 1, MIN_IO_SEQ], ScalarType::Float16),  // K input
    ([1, kv_ch, 1, MIN_IO_SEQ], ScalarType::Float16),  // V input
    ([1, 1, head_dim, head_dim], ScalarType::Float16),  // rotation matrix
]);

let attn_alloc_size = uniform_alloc_size(&[
    ([1, q_ch, 1, MIN_IO_SEQ], ScalarType::Float16),               // Q
    ([1, kv_ch, 1, config.max_seq_len], ScalarType::Int8),          // K cache
    ([1, kv_ch, 1, config.max_seq_len], ScalarType::Int8),          // V cache
    ([1, 1, head_dim, head_dim], ScalarType::Float16),              // unrotation
]);
```

**Change** attn K/V cache to Float16:
```rust
let attn_alloc_size = uniform_alloc_size(&[
    ([1, q_ch, 1, MIN_IO_SEQ], ScalarType::Float16),
    ([1, kv_ch, 1, config.max_seq_len], ScalarType::Float16),  // was Int8
    ([1, kv_ch, 1, config.max_seq_len], ScalarType::Float16),  // was Int8
    ([1, 1, head_dim, head_dim], ScalarType::Float16),
]);
```

### turboquant.rs: step_attention

**Current flow** (lines 579-643):
1. `copy_column0_from(k_proj)` → staging          ← REMOVE
2. `copy_column0_from(v_proj)` → staging          ← REMOVE
3. ANE eval cache-write with staging inputs
4. `read_column0_f16` → convert to INT8 → `write_bytes_at`  ← SIMPLIFY
5. `copy_column0_from(q)` → staging               ← REMOVE
6. ANE eval attention with staging Q
7. Return attn_out

**New flow:**
1. ANE eval cache-write with pre_attn output IOSurfaces directly
   (requires matching alloc sizes — see Phase B)
2. `read_column0_f16` → `write_f16_at` (no INT8 conversion)
3. ANE eval attention with pre_attn Q IOSurface directly
4. Return attn_out

If alloc sizes don't match for Phase A (deferred to Phase B),
keep staging copies but remove INT8 conversion:
```rust
// Phase A: Keep staging, remove INT8 conversion
self.cw_k_staging.copy_column0_from(k_proj)?;
self.cw_v_staging.copy_column0_from(v_proj)?;

self.runtime.eval(&self.cache_write_program, ...)?;

// CHANGED: read FP16 directly, write FP16 to cache (no INT8 conversion)
let k_f16 = self.cw_k_output.read_column0_f16()?;
let v_f16 = self.cw_v_output.read_column0_f16()?;
self.cache.update_cache(layer, &k_f16, &v_f16, k_original.as_deref())?;
```

### turboquant_mil.rs: emit_attention_mil

**Current** (line 308-543): INT8 cache inputs with dequant chain:
```
a_input1: tensor<int8, [1, kv_ch, 1, max_seq_len]>  // K cache
a_input2: tensor<int8, [1, kv_ch, 1, max_seq_len]>  // V cache
```
Then: `slice_by_index → cast(int8→fp16) → mul(deq_scale) → sub(offset)`

**Change to** FP16 cache inputs with simpler dequant:
```
a_input1: tensor<fp16, [1, kv_ch, 1, max_seq_len]>  // K cache (FP16-quantized)
a_input2: tensor<fp16, [1, kv_ch, 1, max_seq_len]>  // V cache
```
Then: `slice_by_index → mul(deq_scale)` (no cast needed, no sub needed
since offset is 0 for symmetric quantization)

Specifically in `emit_dequantize_chain` (line 547-635):
- Remove the `cast(int8→fp16)` op (line 574-580)
- Remove the `sub(deq_offset)` op (lines 594-601) — offset is always 0
- Change input type from `int8` to `fp16`
- Change `sliced_shape` type from `int8` to `fp16`

### inference.rs: decode TurboQuant path

**Current** (lines 921-940): passes pre_attn output tensors to
`step_attention`:
```rust
let q = &layer.pre_attn.output_tensors[0];
let k_proj = if num_pre_outputs > 1 { &layer.pre_attn.output_tensors[1] } else { ... };
let v_proj = if num_pre_outputs > 2 { &layer.pre_attn.output_tensors[2] } else { ... };
let attn_tensor = tq.step_attention(layer_idx, q, k_proj, v_proj)?;
```

This wiring stays the same — `step_attention` receives the same
IOSurface references. The changes are internal to `step_attention`.

### QJL Impact

QJL (`enable_qjl`) computes `sign(K_original - K_dequantized)` in
`compute_qjl_signs` (line 318-350). This function:
1. Casts INT8 quantized K to f32: `(k_quantized[i] as i8) as f32`
2. Scales by `deq_scale`
3. Un-rotates
4. Compares against original K

**With FP16 cache:** Step 1 changes — instead of `(byte as i8) as f32`,
read the FP16 value directly: `k_quantized_f16[i].to_f32()`. The rest
stays the same. `update_cache` already receives `k_original` as
`Option<&[f16]>`.

Change `compute_qjl_signs` signature:
```rust
fn compute_qjl_signs(
    k_quantized: &[f16],  // was &[u8]
    k_original: &[f16],
    ...
```

### Tests to update

- `turboquant.rs` tests that create `KvCacheManager` and call
  `update_cache` with `&[u8]` → change to `&[f16]`
- `turboquant_mil.rs` tests that assert `int8` in attention MIL
  input shapes → change to `fp16`
- Any test asserting `ScalarType::Int8` for cache tensors → `Float16`

### Existing tests to verify unchanged

- `cache_write_mil_is_valid_program` — unchanged, cache-write still
  outputs FP16 (it always did)
- `attention_mil_gqa_uses_tile` — needs shape type update
- `turboquant_bench` example — should work with updated types
