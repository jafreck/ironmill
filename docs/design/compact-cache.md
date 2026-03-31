# Compact KV Cache: CPU-Quantized INT8 Cache for ANE Attention

## Motivation

TurboQuant's INT8 KV cache provides 50% memory savings but 3–30% throughput
**regression** on ANE (see `ane-constraints.md`). The regression is caused by
ANE's `cast(int8→fp16)` op inside the attention program - ANE dequantizes to
FP16 before compute, and the cast itself is pure overhead with no compensating
bandwidth savings.

Compact Cache takes a different approach: keep the ANE attention program
**identical to the FP16 baseline** and move quantize/dequantize to the CPU.

## Design

```
Per-token decode (per layer):

pre_attn (ANE) ── same as FP16 baseline
  │ outputs: Q [q_ch], K_proj [kv_ch], V_proj [kv_ch]
  │
  ▼
CPU: quantize K_proj → INT8, write to k_cache[seq_pos]    ← new
CPU: quantize V_proj → INT8, write to v_cache[seq_pos]    ← new
CPU: dequantize k_cache[0..seq_pos+1] → FP16 staging      ← new
CPU: dequantize v_cache[0..seq_pos+1] → FP16 staging      ← new
CPU: write Q to q_staging                                  ← same as FP16
  │
  ▼
attention (ANE) ── SAME FP16 program, SAME inputs/outputs
  │ inputs: q_staging(fp16), k_staging(fp16), v_staging(fp16)
  │ no cast ops, no dequant ops, no rotation
  │
  ▼
post_attn (ANE) ── same as FP16 baseline
```

### Comparison

| | FP16 Baseline | TurboQuant | Compact Cache |
|---|---|---|---|
| Cache dtype | FP16 | INT8 | INT8 |
| Cache memory | 100% | 50% | 50% |
| Attention program | FP16 | INT8+cast+dequant+rotation | FP16 (identical to baseline) |
| ANE evals/layer | 3 | 4 (extra cache-write) | 3 (same as baseline) |
| Quantize location | N/A | ANE (separate eval) | CPU (inline) |
| Dequantize location | N/A | ANE (inside attention) | CPU (before attention eval) |
| Extra ANE ops | None | cast, mul, rotation matmuls | None |
| Throughput vs FP16 | - | −3% to −30% | ~0% (hypothesis) |

## Quantization Strategy

### Phase 1: Simple Scalar Quantization

Per-channel min/max symmetric quantization:

```
scale = max(abs(values)) / 127.0
quantized = round(values / scale)          # FP16 → INT8
dequantized = quantized * scale            # INT8 → FP16
```

- No rotation needed
- ~0.4% max relative error for well-distributed values
- Cheap on CPU: vectorizable with NEON

### Phase 2 (optional): Hadamard Rotation

Apply randomized Hadamard rotation before quantization to reduce outlier
sensitivity. Uses existing `rotate_rows_hadamard` from `mil-rs/src/ir/passes/rotation.rs`.

```
rotated = R · K_proj                       # [head_dim × head_dim] × [head_dim]
quantized = round(rotated / scale)         # More uniform → less quantization error
dequantized = quantized * scale
unrotated = R⁻¹ · dequantized             # R is self-inverse for Hadamard
```

- Better quality at the cost of a CPU matmul per head per layer
- head_dim=128: 128×128 matmul = 16K MACs per head per token
- 8 KV heads × 16K = 128K MACs per layer - trivial on ARM

## Module Structure

```
crates/ironmill-inference/src/ane/
├── compact_cache/
│   ├── mod.rs           # CompactCacheConfig, pub exports
│   ├── cache.rs         # INT8 cache storage + CPU quantize/dequantize
│   └── manager.rs       # Per-layer cache lifecycle, staging tensor management
├── turboquant/          # (existing, retained)
├── decode.rs            # Add CompactCache branch alongside TQ and FP16
├── model.rs
├── runtime.rs
└── mod.rs               # Add: pub mod compact_cache;
```

### Key Types

```rust
/// Configuration for compact KV cache.
pub struct CompactCacheConfig {
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub num_layers: usize,
    /// Enable Hadamard rotation for better quantization quality.
    pub enable_rotation: bool,
    pub rotation_seed: u64,
}

/// Per-layer INT8 KV cache with CPU-side quantize/dequantize.
pub struct CompactCacheManager {
    /// INT8 cache storage: Vec<i8> per layer, per K/V.
    /// Layout: [kv_ch × max_seq_len] row-major.
    k_caches: Vec<Vec<i8>>,
    v_caches: Vec<Vec<i8>>,
    /// Per-layer, per-position scale factors.
    k_scales: Vec<Vec<f16>>,
    v_scales: Vec<Vec<f16>>,
    /// FP16 staging tensors for attention input (reused across layers).
    k_staging: AneTensor,
    v_staging: AneTensor,
    /// Current sequence position.
    seq_pos: usize,
    config: CompactCacheConfig,
}
```

### Core Operations

```rust
impl CompactCacheManager {
    /// Quantize and store K/V for the current token.
    ///
    /// CPU-side: compute per-token scale, quantize FP16 → INT8,
    /// write to cache at seq_pos.
    pub fn write_cache(&mut self, layer: usize, k: &[f16], v: &[f16]);

    /// Dequantize cache and write to FP16 staging tensors for ANE attention.
    ///
    /// CPU-side: read INT8 cache [0..seq_pos+1], dequantize → FP16,
    /// write to staging AneTensor IOSurfaces.
    pub fn prepare_attention(&mut self, layer: usize) -> Result<(&AneTensor, &AneTensor)>;

    /// Advance to next sequence position.
    pub fn advance_seq_pos(&mut self);

    /// Reset for new sequence.
    pub fn reset(&mut self);
}
```

## Integration in decode()

The compact cache branch in `decode()` would replace the TQ branch with:

```rust
let attn_out_data = if let Some(ref mut cc) = self.compact_cache {
    let q_data = read_f16_channels(&layer.pre_attn.output_tensors[0])?;
    let k_data = read_f16_channels(&layer.pre_attn.output_tensors[1])?;
    let v_data = read_f16_channels(&layer.pre_attn.output_tensors[2])?;

    // CPU: quantize and store K/V
    cc.write_cache(layer_idx, &k_data, &v_data);

    // CPU: dequantize cache → FP16 staging tensors
    let (k_staging, v_staging) = cc.prepare_attention(layer_idx)?;

    // ANE: standard FP16 attention (same compiled program as baseline)
    let q_staging = self.fp16_attn_q_staging.as_mut().unwrap();
    write_f16_padded(q_staging, &q_data)?;
    let out_staging = self.fp16_attn_out_staging.as_mut().unwrap();
    self.runtime.eval_raw(
        self.fp16_attn_compiled.as_ref().unwrap().as_raw_ptr(),
        &[q_staging, k_staging, v_staging],
        &mut [out_staging],
    )?;
    read_f16_channels(out_staging)?
}
```

## Performance Model

### CPU Cost (per layer, per token)

| Operation | Size | Estimated Cost |
|---|---|---|
| Quantize K (FP16→INT8) | kv_ch values | ~1 μs |
| Quantize V (FP16→INT8) | kv_ch values | ~1 μs |
| Dequantize K cache (INT8→FP16) | kv_ch × seq_len | ~10 μs @ seq=512, ~80 μs @ seq=4096 |
| Dequantize V cache (INT8→FP16) | kv_ch × seq_len | ~10 μs @ seq=512, ~80 μs @ seq=4096 |
| Write staging tensors | kv_ch × seq_len × 2 bytes | ~5 μs @ seq=512 |

**Total per layer:** ~25 μs @ seq=512, ~165 μs @ seq=4096
**Total per token (28 layers):** ~0.7 ms @ seq=512, ~4.6 ms @ seq=4096

### Comparison with TurboQuant ANE overhead

TurboQuant adds per layer:
- Cache-write eval: ~50 μs dispatch overhead
- `cast(int8→fp16)` + `mul(deq_scale)` on full cache: measured 7-30% of attention time

For 8B MHA @ 4096, the TQ INT8 attention measured 3811 μs vs FP16's 2681 μs -
a penalty of 1130 μs/layer. Compact Cache's estimated CPU cost of ~165 μs/layer
would be **~7× less overhead** at this scale.

### When Compact Cache is Worth It

The CPU dequant cost grows as O(seq_len × kv_ch × num_layers). It becomes
net-negative when this exceeds the memory savings benefit. For most practical
configurations (seq_len ≤ 8192, kv_ch ≤ 4096), the CPU cost is well under
the ANE attention latency and the memory savings justify the approach.

At very long contexts (seq_len > 16K), the O(seq_len) CPU dequant may become
the bottleneck. At that point, consider:
- Dequantizing only the most recent cache window (sliding window attention)
- Parallelizing dequant across CPU cores with rayon
- Keeping the most recent N tokens in FP16 and older tokens in INT8

## Testing Plan

1. Unit test: quantize → dequantize round-trip error vs reference
2. Integration: swap CompactCache into decode(), compare token outputs vs FP16
3. Benchmark: `cache_bandwidth_bench` with a CompactCache column
4. Quality: token agreement percentage vs FP16 baseline at various seq_len

## References

- `ane-constraints.md` - INT8 bandwidth finding (no throughput gain on ANE)
- `turboquant-zero-copy.md` - TurboQuant history and optimization attempts
- `cache_bandwidth_bench.rs` - INT8 vs FP16 attention benchmark
- `rotation.rs` - Hadamard rotation primitives for optional Phase 2
- Orion constraints doc - confirms CPU-side cache management pattern
