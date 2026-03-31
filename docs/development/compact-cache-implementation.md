# Compact Cache Implementation

> Implements `docs/design/compact-cache.md`.
> CPU-quantized INT8 KV cache with unmodified FP16 ANE attention.

## Overview

Add a `compact_cache` module to `crates/ironmill-inference/src/ane/` that stores
the KV cache as INT8 on the CPU side and dequantizes to FP16 staging tensors
before each ANE attention eval. The ANE attention program is the same FP16
program used by the baseline — no extra ANE ops.

## Files to Create

### `crates/ironmill-inference/src/ane/compact_cache/mod.rs`

```rust
pub mod cache;

pub use cache::{CompactCacheConfig, CompactCacheManager};
```

### `crates/ironmill-inference/src/ane/compact_cache/cache.rs`

Core implementation. Contains:

- `CompactCacheConfig` — configuration struct
- `CompactCacheManager` — per-layer INT8 storage, quantize/dequant, staging

## Files to Modify

### `crates/ironmill-inference/src/ane/mod.rs`

Add `pub mod compact_cache;` alongside existing `turboquant` module.

### `crates/ironmill-inference/src/ane/decode.rs`

Three changes:

1. **Add `compact_cache` field to `AneInference`:**
   ```rust
   compact_cache: Option<CompactCacheManager>,
   ```
   After the existing `turboquant` field (line 59).

2. **Add `CompactCacheConfig` parameter to `compile()`:**
   ```rust
   pub fn compile(
       program: &mil_rs::ir::Program,
       turbo_config: Option<TurboQuantConfig>,
       compact_config: Option<CompactCacheConfig>,  // new
   ) -> Result<Self> {
   ```
   Only one of `turbo_config` or `compact_config` should be `Some`.
   In the mode-selection block (~line 900), add a third branch that:
   - Compiles the FP16 attention program (same as baseline)
   - Allocates FP16 staging tensors (same as baseline)
   - Creates `CompactCacheManager` instead of FP16 `AneTensor` caches
   - Sets `compact_cache = Some(manager)`

3. **Add compact cache branch to `decode()` per-layer loop:**
   Insert between the TQ branch and the FP16 branch (~line 1025):
   ```rust
   } else if let Some(ref mut cc) = self.compact_cache {
       // read Q/K/V from pre_attn outputs
       // cc.write_cache(layer_idx, &k_data, &v_data)
       // (k_staging, v_staging) = cc.prepare_attention(layer_idx)?
       // eval_raw with FP16 attention program
       // read output
   } else if let Some(ref mut caches) = self.fp16_kv_caches {
   ```

## Implementation Details

### CompactCacheConfig

```rust
#[derive(Clone)]
pub struct CompactCacheConfig {
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
}
```

### CompactCacheManager

```rust
pub struct CompactCacheManager {
    config: CompactCacheConfig,
    /// INT8 cache: k_caches[layer][kv_ch * max_seq_len]
    k_caches: Vec<Vec<i8>>,
    v_caches: Vec<Vec<i8>>,
    /// Per-position scale: scales[layer][seq_pos] (one f32 per token per layer)
    k_scales: Vec<Vec<f32>>,
    v_scales: Vec<Vec<f32>>,
    /// Reusable FP16 staging for dequantized cache (shared across layers).
    k_staging: AneTensor,
    v_staging: AneTensor,
    /// Current sequence position.
    seq_pos: usize,
}
```

### write_cache()

Called once per layer per token. Quantizes K/V projections to INT8 and writes
to the cache at `seq_pos`.

```rust
pub fn write_cache(&mut self, layer: usize, k: &[f16], v: &[f16]) {
    let kv_ch = self.config.num_kv_heads * self.config.head_dim;

    // Symmetric per-token quantization: scale = max(|x|) / 127
    let k_max = k.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
    let k_scale = if k_max == 0.0 { 1.0 } else { k_max / 127.0 };
    let v_max = v.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
    let v_scale = if v_max == 0.0 { 1.0 } else { v_max / 127.0 };

    self.k_scales[layer].push(k_scale);
    self.v_scales[layer].push(v_scale);

    let offset = self.seq_pos * kv_ch;
    for (i, &val) in k.iter().enumerate() {
        self.k_caches[layer][offset + i] =
            (val.to_f32() / k_scale).round().clamp(-128.0, 127.0) as i8;
    }
    for (i, &val) in v.iter().enumerate() {
        self.v_caches[layer][offset + i] =
            (val.to_f32() / v_scale).round().clamp(-128.0, 127.0) as i8;
    }
}
```

### prepare_attention()

Called once per layer per token. Dequantizes the full cache [0..seq_pos+1]
into FP16 staging tensors for the ANE attention eval.

```rust
pub fn prepare_attention(
    &mut self,
    layer: usize,
) -> Result<(&AneTensor, &AneTensor)> {
    let kv_ch = self.config.num_kv_heads * self.config.head_dim;
    let active_len = self.seq_pos + 1;

    // Dequantize K cache → FP16
    let mut k_fp16 = vec![f16::ZERO; kv_ch * self.config.max_seq_len];
    for pos in 0..active_len {
        let scale = self.k_scales[layer][pos];
        let offset = pos * kv_ch;
        for ch in 0..kv_ch {
            // ANE layout: [1, C, 1, S] → element at (ch, pos) = ch * max_seq_len + pos
            let cache_idx = offset + ch;
            let staging_idx = ch * self.config.max_seq_len + pos;
            k_fp16[staging_idx] =
                f16::from_f32(self.k_caches[layer][cache_idx] as f32 * scale);
        }
    }
    self.k_staging.write_f16(&k_fp16)?;

    // Dequantize V cache → FP16 (same layout)
    let mut v_fp16 = vec![f16::ZERO; kv_ch * self.config.max_seq_len];
    for pos in 0..active_len {
        let scale = self.v_scales[layer][pos];
        let offset = pos * kv_ch;
        for ch in 0..kv_ch {
            let cache_idx = offset + ch;
            let staging_idx = ch * self.config.max_seq_len + pos;
            v_fp16[staging_idx] =
                f16::from_f32(self.v_caches[layer][cache_idx] as f32 * scale);
        }
    }
    self.v_staging.write_f16(&v_fp16)?;

    Ok((&self.k_staging, &self.v_staging))
}
```

**Note on layout:** The INT8 cache stores values row-major `[seq_pos, kv_ch]`
for cache-write locality. The ANE staging tensor uses `[1, C, 1, S]` layout
where element `(ch, pos)` is at index `ch * max_seq_len + pos`. The transpose
happens during dequant.

### Staging Tensor Allocation

The staging tensors must share the same `uniform_alloc_size` as Q and the
attention program's other inputs:

```rust
let attn_alloc = uniform_alloc_size(&[
    ([1, q_channels, 1, MIN_IO_SEQ], ScalarType::Float16),
    ([1, kv_channels, 1, max_seq_len], ScalarType::Float16),
    ([1, kv_channels, 1, max_seq_len], ScalarType::Float16),
]);

let k_staging = AneTensor::new_with_min_alloc(
    kv_channels, max_seq_len, ScalarType::Float16, attn_alloc,
)?;
let v_staging = AneTensor::new_with_min_alloc(
    kv_channels, max_seq_len, ScalarType::Float16, attn_alloc,
)?;
```

This is the same alloc pattern as the FP16 baseline caches, so the existing
FP16 attention program works without modification.

## Task Breakdown

### Task 1: Create CompactCacheManager

File: `crates/ironmill-inference/src/ane/compact_cache/cache.rs`

- `CompactCacheConfig` struct
- `CompactCacheManager::new(config, attn_alloc) -> Result<Self>`
- `write_cache(layer, k, v)` — quantize + store
- `prepare_attention(layer) -> Result<(&AneTensor, &AneTensor)>` — dequant + write staging
- `advance_seq_pos()`
- `reset()`
- `seq_len() -> usize`

Tests:
- Round-trip test: quantize → dequant → compare with original, check max error < 1%
- Multi-position test: write 10 positions, verify all positions dequant correctly
- Reset test: verify cache clears properly

### Task 2: Create mod.rs

File: `crates/ironmill-inference/src/ane/compact_cache/mod.rs`

- `pub mod cache;`
- `pub use cache::{CompactCacheConfig, CompactCacheManager};`

Register in `crates/ironmill-inference/src/ane/mod.rs`:
- Add `pub mod compact_cache;`

### Task 3: Integrate into AneInference

File: `crates/ironmill-inference/src/ane/decode.rs`

- Add `compact_cache: Option<CompactCacheManager>` field to `AneInference`
- Add `compact_config: Option<CompactCacheConfig>` parameter to `compile()`
- Add compact cache initialization in `compile()` mode-selection block:
  - Compile FP16 attention program (reuse existing code)
  - Allocate staging tensors (reuse existing code)
  - Create `CompactCacheManager::new(config, attn_alloc)`
- Add compact cache branch in `decode()` per-layer loop
- Call `cc.advance_seq_pos()` after all layers

Tests:
- Compile with compact_config, verify no panics
- Generate 10 tokens, verify output is non-zero

### Task 4: Add to E2E benchmark

File: `crates/ironmill-inference/examples/turboquant_e2e_bench.rs`

- Add a "Compact Cache" mode alongside FP16 and TQ
- Compare throughput and memory across all three

### Task 5: Add to cache bandwidth benchmark

File: `crates/ironmill-inference/examples/cache_bandwidth_bench.rs`

- Add a column showing compact cache latency
- Breakdown: CPU dequant time vs ANE attention time

## Build & Test Commands

```sh
# Build
cargo check -p ironmill-inference

# Unit tests
cargo test -p ironmill-inference compact_cache

# E2E benchmark
cargo run -p ironmill-inference --example turboquant_e2e_bench --release -- compact

# Cache bandwidth comparison
cargo run -p ironmill-inference --example cache_bandwidth_bench --release
```

## Success Criteria

1. Compact cache throughput is within 5% of FP16 baseline
2. KV cache memory is 50% of FP16 baseline
3. Token output agreement ≥ 95% vs FP16 at seq_len=512
4. No regression in FP16 or TQ paths
