# MLX Backend

## Overview

Add MLX as an additional GPU inference backend alongside the existing
Metal/MPS backend. MLX is Apple's array framework for Apple Silicon,
offering lazy evaluation with automatic kernel fusion and optimized
built-in operations. The MLX backend targets the same `InferenceEngine`
trait and reuses the same `WeightProvider` abstraction (from
`mil_rs::weights`), `ModelConfig`, TurboQuant algorithm, and sampling
infrastructure.

MLX does **not** replace the MPS backend. Both will coexist behind
separate feature flags.

## Motivation

The current Metal/MPS backend dispatches operations eagerly — each
compute kernel (RMSNorm, RoPE, SiLU, residual add) is a separate
`MTLComputeCommandEncoder` dispatch encoded onto a command buffer.
MPS matmuls are dispatched individually per projection. This works
well but leaves performance on the table:

- **No kernel fusion.** Adjacent element-wise ops (e.g. SiLU + gate
  multiply, RMSNorm + residual) are dispatched as separate kernels
  with intermediate buffer reads/writes.
- **Manual buffer management.** Every intermediate tensor is a
  pre-allocated `MetalBuffer`. The allocation strategy is static.
- **No graph-level optimization.** The dispatch order is hardcoded
  in `run_pipeline()`.

MLX addresses these through lazy evaluation — operations are recorded
into a computation graph and fused/optimized before dispatch. This can
yield measurable speedups on memory-bandwidth-bound workloads like LLM
decode.

## Current Architecture (MPS Backend)

```
Full decode pipeline (single token):
  0. embedding lookup    [Metal kernel]
  ─── per layer (×num_layers) ───
  1. rms_norm           [Metal kernel]
  2. Q/K/V projection   [MPS matmul]
  3. RoPE               [Metal kernel]
  4. cache-write         [Metal kernel]  fused: rotate → quantize → INT8 cache write
  5. attention           [Metal kernel]  fused: dequant K/V → QK → softmax → ×V
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

Key components:
- `ironmill-metal-sys` — safe Rust wrappers for Metal, MPS, command buffers, compute pipelines
- `gpu::ops` — `GpuPipelines` compiles and dispatches 14 kernel functions across 8 `.metal` shader files
- `gpu::turboquant` — INT4/INT8 KV cache with Hadamard rotation, per-head absmax scaling, outlier channel strategy
- `gpu::weights` — loads `WeightProvider` tensors into `MetalBuffer`s
- `gpu::inference` — `GpuInference` implements `InferenceEngine`, orchestrates the pipeline

All ops are encoded into a single `MTLCommandBuffer` per step. No
per-op synchronization, no intermediate CPU readback (except FP16 KV
cache path which requires a flush).

## Proposed Architecture (MLX Backend)

### Pipeline Mapping — FP16 KV Cache (Phase 2)

In the FP16 path, the entire pipeline is lazy with a single `eval()`
at the end. No intermediate synchronization required.

```
Full decode pipeline (single token):
  0. embedding lookup    [mlx_embedding]
  ─── per layer (×num_layers) ───
  1. rms_norm           [mlx_fast_rms_norm]
  2. Q/K/V projection   [mlx_matmul]              ← fused with rms_norm
  3. RoPE               [mlx_fast_rope]
  4. attention           [mlx_fast_sdpa]           ← built-in SDPA with KV cache
  5. output projection   [mlx_matmul]
  6. residual add        [mlx_add]                 ← fused with output proj
  7. rms_norm            [mlx_fast_rms_norm]
  8. gate/up projection  [mlx_matmul ×2]
  9. silu + gate mul     [mlx_silu + mlx_multiply] ← fused by lazy eval
 10. down projection     [mlx_matmul]
 11. residual add        [mlx_add]                 ← fused with down proj
  ─── end per layer ───
 12. final rms_norm      [mlx_fast_rms_norm]
 13. lm_head projection  [mlx_matmul]
 14. mlx_eval()          ← single materialization point
```

### Pipeline Mapping — TurboQuant INT4/INT8 KV Cache (Phase 3)

With TurboQuant, per-layer `eval()` calls are required to materialize
the quantized cache before attention reads it. This narrows the lazy
fusion window to two regions per layer:

```
Full decode pipeline (single token):
  0. embedding lookup    [mlx_embedding]
  ─── per layer (×num_layers) ───
  ┌─ lazy region A (norm + projections) ─────────────────────────┐
  │ 1. rms_norm           [mlx_fast_rms_norm]                    │
  │ 2. Q/K/V projection   [mlx_matmul]   ← fused with rms_norm  │
  │ 3. RoPE               [mlx_fast_rope]                        │
  └──────────────────────────────────────────────────────────────┘
  4. cache-write         [custom Metal kernel ×2]  K write + V write (separate dispatches)
     mlx_eval()          ← materialize cache + scales
  5. attention           [custom Metal kernel]     dequant + QK + softmax + ×V
  ┌─ lazy region B (FFN block) ──────────────────────────────────┐
  │ 6. output projection   [mlx_matmul]                          │
  │ 7. residual add        [mlx_add]          ← fused            │
  │ 8. rms_norm            [mlx_fast_rms_norm]                   │
  │ 9. gate/up projection  [mlx_matmul ×2]                      │
  │10. silu + gate mul     [mlx_silu + mlx_multiply] ← fused    │
  │11. down projection     [mlx_matmul]                          │
  │12. residual add        [mlx_add]          ← fused            │
  └──────────────────────────────────────────────────────────────┘
  ─── end per layer ───
 13. final rms_norm      [mlx_fast_rms_norm]
 14. lm_head projection  [mlx_matmul]
 15. mlx_eval()           ← materialize logits for sampling
```

The realistic fusion benefit with TurboQuant is:
- **Region A:** rms_norm + 3 matmuls + RoPE (5 ops fused)
- **Region B:** matmul + add + rms_norm + 2 matmuls + silu + mul + matmul + add (9 ops fused)
- **Cross-layer:** region B of layer N fuses with region A of layer N+1

The FFN block (region B) is where most fusion value lies — it
eliminates 4 intermediate buffer reads/writes per layer that the MPS
backend currently performs.

### Crate Layout

```
crates/
  ironmill-mlx-sys/                  NEW — unsafe FFI bindings to mlx-c
    Cargo.toml
    build.rs                         find/build mlx-c via CMake, generate bindings with bindgen
    src/
      lib.rs                         safe wrappers: MlxArray, MlxStream, MlxDevice
      array.rs                       array creation, data access, dtype mapping
      error.rs                       MlxSysError wrapping mlx-c error codes
      ops.rs                         matmul, rms_norm, rope, add, multiply, etc.
      metal_kernel.rs                custom Metal kernel registration + dispatch
      stream.rs                      stream management, eval, synchronize

  ironmill-inference/
    Cargo.toml                       add: ironmill-mlx-sys = { optional = true }
    src/
      mlx/                           NEW — MLX backend module
        mod.rs
        config.rs                    MlxConfig (analogous to GpuConfig)
        error.rs                     MlxError wrapping MlxSysError → InferenceError
        inference.rs                 MlxInference: impl InferenceEngine
        weights.rs                   WeightProvider → MlxArray loading (dense + quantized)
        quantized_matmul.rs          PolarQuant INT4/INT8 matmul kernels adapted for MLX
        turboquant/
          mod.rs                     MlxTurboQuantModel (rotation signs, codebooks, scales)
          cache.rs                   MlxKvCache (INT4/INT8 arrays + scale arrays + outlier split)
          kernels.rs                 adapted TurboQuant + outlier kernel source + dispatch
```

**Note on `ironmill-mlx-sys` build approach:** unlike `ironmill-metal-sys`
which uses bare `#[link(name = "Metal", kind = "framework")]` directives
(no `build.rs`), `ironmill-mlx-sys` requires a `build.rs` because mlx-c
is not a system framework — it must be built from source via CMake or
located via `MLX_DIR`. This is new build complexity not present in the
existing FFI crate.

### Feature Gating

```toml
# crates/ironmill-inference/Cargo.toml
[features]
metal = ["dep:ironmill-metal-sys"]
mlx = ["dep:ironmill-mlx-sys"]
```

```rust
// crates/ironmill-inference/src/lib.rs
#[cfg(feature = "metal")]
pub mod gpu;

#[cfg(feature = "mlx")]
pub mod mlx;
```

The `metal` and `mlx` features are independent. Both can be compiled
simultaneously, but only one backend should be initialized per process
to avoid GPU command queue contention and memory pool conflicts (see
[Risks](#risks-and-open-questions)).

### Error Handling

```rust
// ironmill-mlx-sys/src/error.rs
#[derive(Debug, thiserror::Error)]
pub enum MlxSysError {
    #[error("mlx-c error: {0}")]
    MlxC(String),
    #[error("kernel compilation failed: {0}")]
    KernelCompile(String),
    #[error("invalid dtype: {0}")]
    InvalidDtype(String),
}

// ironmill-inference/src/mlx/error.rs
#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    #[error("mlx sys: {0}")]
    Sys(#[from] MlxSysError),
    #[error("weight loading: {0}")]
    WeightLoading(String),
    #[error("shape mismatch: {0}")]
    Shape(String),
}

// Maps to InferenceError at the trait boundary:
impl From<MlxError> for InferenceError {
    fn from(e: MlxError) -> Self {
        InferenceError::Runtime(e.to_string())
    }
}
```

### Weight Loading

MLX arrays can be created from raw data pointers. The existing
`WeightProvider` trait (in `mil_rs::weights`) provides tensor data as
`Cow<[u8]>` with shape and dtype. `ModelConfig` provides the
architecture parameters (num_layers, head_dim, num_kv_heads, etc.)
needed to set up the pipeline.

**Lifetime constraint:** when the `WeightProvider` returns
`Cow::Borrowed` data (mmap'd SafeTensors), the source data must
outlive any `MlxArray` created from it. Since `mlx_array_new_data()`
does not guarantee a copy, the safe wrapper must either:
1. Always copy into MLX-owned memory (safest, slight overhead), or
2. Hold an `Arc` reference to the provider, ensuring the mmap lives
   as long as the weight arrays.

Option 1 is recommended for initial implementation. Weight loading is
a one-time cost and the copy is negligible compared to inference.

```rust
// mlx/weights.rs

/// Mirrors gpu::weights::WeightBuffer — projections may be dense or quantized.
pub enum MlxWeightBuffer {
    Dense(MlxArray),
    Quantized(MlxQuantizedWeight),
}

pub struct MlxQuantizedWeight {
    pub indices: MlxArray,
    pub lut: MlxArray,
    pub norms: MlxArray,
    pub n_bits: u8,
    pub shape: (usize, usize),
}

pub struct MlxLayerWeights {
    pub input_norm: MlxArray,
    pub q_proj: MlxWeightBuffer,
    pub k_proj: MlxWeightBuffer,
    pub v_proj: MlxWeightBuffer,
    pub o_proj: MlxWeightBuffer,
    pub post_attn_norm: MlxArray,
    pub gate_proj: MlxWeightBuffer,
    pub up_proj: MlxWeightBuffer,
    pub down_proj: MlxWeightBuffer,
    pub q_norm: Option<MlxArray>,
    pub k_norm: Option<MlxArray>,
}

pub struct MlxWeights {
    pub embedding: MlxArray,
    pub layers: Vec<MlxLayerWeights>,
    pub final_norm: MlxArray,
    pub lm_head: MlxArray,
    pub config: ModelConfig,
}

fn load_weight(stream: &MlxStream, provider: &dyn WeightProvider, name: &str)
    -> Result<MlxWeightBuffer, MlxError>
{
    let tensor = provider.tensor(name)
        .map_err(|e| MlxError::WeightLoading(format!("{name}: {e}")))?;
    match &tensor.quant_info {
        QuantizationInfo::None => {
            let arr = MlxArray::from_data_copy(
                &tensor.data, &tensor.shape, tensor.dtype.to_mlx(), stream,
            )?;
            Ok(MlxWeightBuffer::Dense(arr))
        }
        QuantizationInfo::LutToDense { n_bits, .. } => {
            // Load packed indices, LUT, and norms as separate MlxArrays.
            // Dispatched via adapted PolarQuant kernels at inference time.
            let (indices, lut, norms) = unpack_lut_quant(&tensor)?;
            Ok(MlxWeightBuffer::Quantized(MlxQuantizedWeight {
                indices: MlxArray::from_data_copy(&indices.data, &indices.shape, indices.dtype.to_mlx(), stream)?,
                lut: MlxArray::from_data_copy(&lut.data, &lut.shape, lut.dtype.to_mlx(), stream)?,
                norms: MlxArray::from_data_copy(&norms.data, &norms.shape, norms.dtype.to_mlx(), stream)?,
                n_bits: *n_bits,
                shape: (tensor.shape[0], tensor.shape[1]),
            }))
        }
        QuantizationInfo::AffineDequantize { .. } => {
            // CPU-side dequantize to dense, same as MPS backend
            let dense = dequant_affine(&tensor)?;
            let arr = MlxArray::from_data_copy(
                &dense.data, &dense.shape, dense.dtype.to_mlx(), stream,
            )?;
            Ok(MlxWeightBuffer::Dense(arr))
        }
    }
}
```

For quantized weights, matmul dispatch branches on the weight type:

```rust
fn matmul_or_quantized(
    input: &MlxArray, weight: &MlxWeightBuffer, output_name: &str, stream: &MlxStream,
) -> Result<MlxArray, MlxError> {
    match weight {
        MlxWeightBuffer::Dense(w) => Ok(mlx_matmul(input, w)),
        MlxWeightBuffer::Quantized(q) => {
            // Dispatch adapted PolarQuant kernel (INT4 or INT8, matvec or matmul)
            mlx_polarquant_matmul(input, q, stream)
        }
    }
}
```

The 4 PolarQuant kernel variants (`polarquant_matvec_int4`,
`polarquant_matmul_int4`, `polarquant_matvec_int8`,
`polarquant_matmul_int8`) are selected by `n_bits` and whether the
input is a single token (matvec) or multi-token (matmul), matching the
MPS backend's dispatch logic.

### Artifacts and InferenceEngine Entry Point

Following the MPS backend pattern (`GpuArtifacts`), the MLX backend
defines its own artifacts type for the type-erased `load()` interface:

```rust
// mlx/inference.rs
pub struct MlxArtifacts<'a> {
    pub weights: &'a dyn WeightProvider,
    pub config: MlxConfig,
}

impl InferenceEngine for MlxInference {
    fn load(&mut self, artifacts: &dyn Any) -> Result<(), InferenceError> {
        let mlx_artifacts = artifacts
            .downcast_ref::<MlxArtifacts<'_>>()
            .ok_or_else(|| InferenceError::Runtime(
                "MlxInference::load expects MlxArtifacts".into()
            ))?;
        self.weights = MlxWeights::load(&self.stream, mlx_artifacts.weights)?;
        self.config = mlx_artifacts.config.clone();
        // ...
        Ok(())
    }
}
```

### GQA (Grouped-Query Attention)

MLX's `mlx_fast_scaled_dot_product_attention` handles GQA natively
when Q and K/V have different head counts — it broadcasts K/V heads
automatically. No manual `repeat_interleave` tiling is needed for the
FP16 path.

For the TurboQuant custom attention kernel, GQA head expansion is
handled inside the kernel (same as the MPS backend), mapping each Q
head to its corresponding KV head via `q_head / gqa_group_size`.

### TurboQuant Integration

The MPS backend has 6 TurboQuant-related kernel functions:

1. `turboquant_cache_write` — standard cache write (Hadamard rotate → absmax → quantize → store)
2. `turboquant_attention` — standard attention (dequant K/V → scale → QK → softmax → ×V)
3. `turboquant_outlier_cache_write` — outlier channel split cache write
4. `turboquant_outlier_attention` — outlier channel split attention
5. QJL sign computation (within cache write)
6. QJL residual correction (post-attention)

All 6 must be ported. MLX's built-in quantization doesn't support
per-head-per-position dynamic scaling or Hadamard rotation. These
stay as custom Metal kernels.

**Kernel porting (not reuse):** the existing `.metal` shader source
cannot be used verbatim with `mlx_fast_metal_kernel()`. The MPS
backend kernels use explicit `[[buffer(N)]]` attribute bindings
(10 buffers for cache-write, 14 for attention), while MLX's
`metal_kernel` API passes inputs/outputs as ordered arrays with its
own generated function signature. The kernel *bodies* (math, indexing,
threadgroup shared memory usage) are portable, but the function
signatures and buffer binding conventions must be adapted.

Key porting concerns:
- `metal_kernel` may impose constraints on threadgroup shared memory
  declarations (TurboQuant attention uses up to 4×4096 bytes of
  threadgroup memory)
- Input/output count limits in the `metal_kernel` API (outlier
  attention may need even more buffers than standard attention's 14)
- The dispatch grid/threadgroup geometry must be specified through
  MLX's API rather than Metal's `dispatchThreadgroups`

These concerns should be validated in the Phase 2.5 spike before
committing to the full TurboQuant port.

#### Standard Cache (INT4/INT8)

```rust
// mlx/turboquant/cache.rs
pub struct MlxKvCache {
    // Standard path (INT4/INT8 without outlier split)
    k_caches: Vec<MlxArray>,      // per-layer [num_kv_heads, max_seq_len, head_dim] INT8/INT4
    v_caches: Vec<MlxArray>,      // same
    k_scales: Vec<MlxArray>,      // per-layer [num_kv_heads, max_seq_len] FP32
    v_scales: Vec<MlxArray>,      // same
    rotation_signs: MlxArray,     // [head_dim] FP32 — per-element ±1.0 signs

    // Outlier channel split (active when OutlierConfig is present)
    outlier: Option<MlxOutlierCache>,

    // QJL correction (optional, enabled via config)
    qjl_matrix: Option<MlxArray>,       // [head_dim, head_dim] FP16
    k_qjl_signs: Option<Vec<MlxArray>>, // per-layer [num_kv_heads, max_seq_len, head_dim/8]
    k_r_norms: Option<Vec<MlxArray>>,   // per-layer [num_kv_heads, max_seq_len]
}
```

#### Outlier Channel Split

The outlier strategy (Section 4.3 of the TurboQuant paper) separates
high-variance channels into a separate cache with independent
quantization. This uses dedicated kernels
(`turboquant_outlier_cache_write`, `turboquant_outlier_attention`)
that split channels by index before quantization.

```rust
pub struct MlxOutlierCache {
    // Outlier channels — quantized at potentially different bit width
    k_outlier_caches: Vec<MlxArray>,
    v_outlier_caches: Vec<MlxArray>,
    k_outlier_scales: Vec<MlxArray>,
    v_outlier_scales: Vec<MlxArray>,
    // Non-outlier channels
    k_non_outlier_caches: Vec<MlxArray>,
    v_non_outlier_caches: Vec<MlxArray>,
    k_non_outlier_scales: Vec<MlxArray>,
    v_non_outlier_scales: Vec<MlxArray>,
    // Shared config
    outlier_rotation_signs: MlxArray,
    non_outlier_rotation_signs: MlxArray,
    channel_indices: MlxArray,
}

pub struct OutlierConfig {
    pub outlier_channels: Vec<usize>,
    pub outlier_bits: u8,
    pub non_outlier_bits: u8,
}
```

When `OutlierConfig` is present, cache write dispatches the outlier
kernel variant instead of the standard one, and attention reads from
both outlier and non-outlier caches, combining results.

#### QJL Correction

QJL (Johnson-Lindenstrauss) residual correction runs as an additional
custom kernel after the attention output. It computes sign-based
sketches during cache write and applies a correction term during
attention to reduce quantization error. The QJL matrix is a
`[head_dim, head_dim]` Gaussian random projection, stored once per
model.

#### Cache Write Dispatch

```rust
impl MlxKvCache {
    pub fn write_k(&mut self, layer: usize, k: &MlxArray, pos: usize,
                   stream: &MlxStream) -> Result<(), MlxError>
    {
        if let Some(outlier) = &mut self.outlier {
            // Outlier path: split channels, quantize separately
            mlx_fast_metal_kernel(
                "turboquant_outlier_cache_write",
                &[k, &outlier.outlier_rotation_signs,
                  &outlier.non_outlier_rotation_signs, &outlier.channel_indices],
                &[&outlier.k_outlier_caches[layer], &outlier.k_outlier_scales[layer],
                  &outlier.k_non_outlier_caches[layer], &outlier.k_non_outlier_scales[layer]],
                TURBOQUANT_OUTLIER_CACHE_WRITE_SOURCE,
                grid, threadgroup, stream,
            )
        } else {
            // Standard path
            mlx_fast_metal_kernel(
                "turboquant_cache_write",
                &[k, &self.rotation_signs],
                &[&self.k_caches[layer], &self.k_scales[layer]],
                TURBOQUANT_CACHE_WRITE_SOURCE,
                grid, threadgroup, stream,
            )
        }
    }

    // write_v follows the same pattern
}
```

### Synchronization Strategy

MLX's lazy evaluation defers computation until `mlx_eval()` is called.
The TurboQuant decode pipeline needs per-layer synchronization:

1. **After cache write (step 4):** the quantized K/V caches **and
   their scale buffers** must be materialized before the attention
   kernel reads them.
2. **After lm_head (step 14):** logits must be materialized for
   CPU-side sampling.

```rust
fn run_pipeline(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
    let mut x = self.embedding_lookup(tokens);

    for layer in 0..self.num_layers {
        // ── Lazy region A: norm + projections + RoPE ──
        let normed = mlx_fast_rms_norm(&x, &self.weights.layers[layer].input_norm, eps);
        let q = mlx_matmul(&normed, &self.weights.layers[layer].q_proj);
        let k = mlx_matmul(&normed, &self.weights.layers[layer].k_proj);
        let v = mlx_matmul(&normed, &self.weights.layers[layer].v_proj);
        let (q, k) = mlx_fast_rope(&q, &k, self.seq_pos);

        // ── TurboQuant cache write — must materialize ──
        self.kv_cache.write_k(layer, &k, self.seq_pos, &self.stream)?;
        self.kv_cache.write_v(layer, &v, self.seq_pos, &self.stream)?;
        mlx_eval(&[
            &self.kv_cache.k_caches[layer], &self.kv_cache.v_caches[layer],
            &self.kv_cache.k_scales[layer], &self.kv_cache.v_scales[layer],
        ]);

        // ── TurboQuant attention (custom kernel) ──
        let attn_out = self.turboquant_attention(layer, &q)?;

        // ── Lazy region B: FFN block ──
        let proj = mlx_matmul(&attn_out, &self.weights.layers[layer].o_proj);
        x = mlx_add(&x, &proj);
        let normed2 = mlx_fast_rms_norm(&x, &self.weights.layers[layer].post_norm, eps);
        let gate = mlx_matmul(&normed2, &self.weights.layers[layer].gate_proj);
        let up = mlx_matmul(&normed2, &self.weights.layers[layer].up_proj);
        let ffn = mlx_multiply(&mlx_silu(&gate), &up);
        let down = mlx_matmul(&ffn, &self.weights.layers[layer].down_proj);
        x = mlx_add(&x, &down);
        // Region B stays lazy — fuses with region A of next layer
    }

    // ── Final: norm + lm_head ──
    let normed = mlx_fast_rms_norm(&x, &self.weights.final_norm, eps);
    let logits = mlx_matmul(&normed, &self.weights.lm_head);

    // Materialize logits for sampling
    mlx_eval(&[&logits]);
    Ok(logits.to_cpu())
}
```

### FP16 KV Cache Path

For non-TurboQuant inference (standard FP16 KV cache), the entire
pipeline can run without intermediate `eval()` calls. MLX's built-in
`mlx_fast_scaled_dot_product_attention` handles KV cache management
and GQA natively. This is the simplest integration path — no custom
kernels, no per-layer sync, maximum lazy fusion.

### Prefill Strategy

Prefill processes all prompt tokens at once (or in chunks) before
entering the single-token decode loop.

**FP16 path:** MLX's `mlx_fast_scaled_dot_product_attention` accepts
multi-token queries and applies causal masking internally. The entire
prefill can be expressed as a single lazy graph:

```rust
fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
    let chunk_size = self.config.prefill_chunk_size.unwrap_or(tokens.len());
    for chunk in tokens.chunks(chunk_size) {
        let x = self.run_pipeline(chunk)?;
        // run_pipeline handles multi-token Q with causal masking via SDPA
    }
    // Return logits for last token
}
```

Chunking is optional but recommended for long prompts to bound peak
memory. Each chunk's graph is evaluated independently.

**TurboQuant path:** prefill uses the same chunked approach, but each
chunk requires per-layer `eval()` for cache writes (same as decode).
The custom attention kernel must handle multi-token Q against the
growing KV cache. The MPS backend already supports this via its
`token_count` parameter — the adapted MLX kernel uses the same logic.

**`async_eval` opportunity:** MLX's `mlx_async_eval` can overlap
computation of chunk N+1's graph construction with chunk N's GPU
execution. This is explored in Phase 4.

### Memory Model

**MLX memory pool:** MLX manages its own GPU memory allocator. All
`MlxArray` data lives in MLX's pool, which grows on demand and caches
freed allocations. This differs from the MPS backend where every
`MetalBuffer` is explicitly allocated with a chosen `StorageMode`.

**Weight memory:** weights are copied into MLX-owned arrays during
`load()`. For a 1B-parameter FP16 model this is ~2GB. The copy is a
one-time cost at load time.

**Intermediate memory:** MLX automatically allocates and reuses
intermediate buffers during graph evaluation. No manual intermediate
buffer pre-allocation (unlike `IntermediateBuffers` in the MPS
backend). This simplifies the code but makes peak memory less
predictable.

**KV cache memory:** TurboQuant cache arrays are pre-allocated at
`max_seq_len` size (same as MPS backend). Scale buffers are
`[num_kv_heads × max_seq_len]` FP32 per layer.

**Dual-backend caution:** if both `metal` and `mlx` features are
compiled in, only one backend should be initialized per process. Both
frameworks submit work to the same GPU command queues and maintain
separate memory pools. Concurrent initialization wastes memory and
risks GPU scheduling contention.

### Benchmark Integration

The `ironmill-bench` harness supports backend selection via a unified
`--backend` flag. Currently the CLI uses `--backend` for CoreML compute
units (`cpu`, `gpu`, `ane`) and `--metal` as a separate boolean flag.
This should be consolidated into a single `--backend` flag that selects
any backend:

```toml
# ironmill-bench/Cargo.toml
[features]
default = []
ane-direct = ["dep:ironmill-iosurface"]
metal = ["ironmill-inference/metal"]
mlx = ["ironmill-inference/mlx"]
```

```rust
// ironmill-bench CLI — unified backend selection
--backend cpu      // CoreML CPU-only
--backend gpu      // CoreML CPU+GPU
--backend ane      // CoreML CPU+ANE
--backend coreml   // CoreML all compute units
--backend metal    // Metal/MPS backend
--backend mlx      // MLX backend (new)
```

Multiple backends can be specified to compare in a single run:
`--backend metal --backend mlx`.

**Benchmark success criteria:**
- **Phase 2 (FP16):** MLX decode tok/s ≥ MPS FP16 decode tok/s on
  same model/hardware. If MLX is slower, the backend is still worth
  maintaining if the code is significantly simpler.
- **Phase 3 (TurboQuant):** MLX TurboQuant tok/s within 10% of MPS
  TurboQuant tok/s. Numerical parity (max abs diff < 1e-3 on logits).
- **Phase 4 (optimized):** MLX decode tok/s > MPS decode tok/s by any
  margin, demonstrating fusion benefit.

Benchmark on: Qwen3-0.6B, LLaMA-3.2-1B. Compare FP16-vs-FP16 and
TurboQuant-vs-TurboQuant (not FP16 MLX vs TurboQuant MPS).

## Implementation Plan

### Phase 1: ironmill-mlx-sys

1. Set up crate with `build.rs` that finds or builds `mlx-c`
   - Support `MLX_DIR` env var for pre-built library path
   - Fall back to CMake build from vendored source
2. Generate raw bindings with `bindgen` for mlx-c headers
3. Safe `MlxArray` wrapper with `Drop` calling `mlx_free` (reference
   counting: `Clone` calls `mlx_retain`, `Drop` calls `mlx_free`)
4. Safe `MlxStream` and `MlxDevice` wrappers
5. `MlxSysError` error type wrapping mlx-c error codes
6. Wrap core ops: `matmul`, `add`, `multiply`, `reshape`, `transpose`
7. Wrap `mlx_fast_*`: `rms_norm`, `rope`, `scaled_dot_product_attention`
8. Wrap `mlx_fast_metal_kernel` for custom kernel dispatch
9. Wrap `mlx_eval` and `mlx_async_eval`
10. Unit tests: array creation, basic ops, eval round-trip

**Acceptance criteria:** can create arrays, run a matmul, eval, and
read results back from Rust. Reference counting is leak-free (test
with a tight loop).

### Phase 2: MLX Backend — FP16 KV Cache

1. Create `ironmill-inference/src/mlx/` module structure with
   `error.rs`, `config.rs`, `weights.rs`, `inference.rs`
2. Define `MlxArtifacts`, `MlxConfig`, `MlxError`
3. Implement `MlxWeights::load()` from `WeightProvider` (dense + quantized)
4. Implement `MlxInference` with `InferenceEngine` trait
5. Full decode pipeline using MLX built-in ops (dense weights) and
   adapted PolarQuant kernels (quantized weights)
6. FP16 KV cache using `mlx_fast_scaled_dot_product_attention`
7. Prefill with chunking support
8. Wire into `ironmill-bench --backend mlx`
9. Benchmark against MPS FP16 on Qwen3-0.6B, LLaMA-3.2-1B
10. Verify numerical parity with MPS FP16 output (max abs diff < 1e-3)

### Phase 2.5: Custom Metal Kernel Spike

Before committing to the full TurboQuant port, validate that MLX's
`metal_kernel` API supports the requirements:

1. Port the simplest custom kernel (`rms_norm` or `silu_gate`) to
   `metal_kernel` as a proof of concept
2. Verify threadgroup shared memory works (needed for attention)
3. Test with 14+ input buffers (needed for TurboQuant attention: 14,
   outlier variants may need more)
4. Measure dispatch overhead vs. direct Metal encoding
5. Document any API limitations or required kernel adaptations

**Go/no-go gate:** if `metal_kernel` cannot support threadgroup shared
memory or sufficient buffer count, evaluate alternatives:
- Use MLX's C++ custom op API (requires cxx bridge)
- Fall back to direct Metal dispatch for TurboQuant + PolarQuant
  kernels only (hybrid: MLX for standard ops, Metal for custom kernels)

### Phase 3: TurboQuant on MLX

1. Adapt `turboquant_cache_write` kernel body for `metal_kernel` API
   - Separate K and V into independent dispatches
   - Map `[[buffer(N)]]` bindings to ordered input/output arrays
   - Include QJL sign computation (QJL projection matrix, sign buffers,
     and R-norm buffers are part of the cache-write kernel interface)
2. Adapt `turboquant_attention` kernel body similarly
   - Include QJL residual correction (the `qjl_coeff * S·q_rot`
     correction term is computed inside the attention kernel)
3. Implement `MlxKvCache` with per-layer INT4/INT8 cache + FP32 scale
   arrays + QJL sign/norm buffers
4. Implement `MlxTurboQuantModel` (rotation signs, QJL matrix,
   codebooks, scales)
5. Wire TurboQuant into `MlxInference` decode loop with per-layer eval
6. Verify numerical parity with MPS TurboQuant output (INT8)
7. Adapt `turboquant_outlier_cache_write` kernel for `metal_kernel` API
8. Adapt `turboquant_outlier_attention` kernel similarly
9. Implement `MlxOutlierCache` and `OutlierConfig`
10. Verify numerical parity with MPS outlier path output (INT4)
11. Adapt PolarQuant matmul kernels (`polarquant_matvec_int4`,
    `polarquant_matmul_int4`, `polarquant_matvec_int8`,
    `polarquant_matmul_int8`) for `metal_kernel` API
12. Implement `MlxWeightBuffer::Quantized` dispatch with bit-width and
    matvec/matmul selection (matching MPS backend logic)
16. Benchmark INT4/INT8 path against MPS backend

### Phase 4: Optimization

1. Profile `eval()` placement — minimize sync points per layer
2. Experiment with `mlx_async_eval` for prefill chunk overlap
3. Benchmark prefill throughput (MLX batched matmul vs MPS)
4. Evaluate MLX's built-in quantized matmul as alternative to ported
   PolarQuant kernels for weight-only INT4/INT8
5. Profile cross-layer fusion (region B → region A)

## Risks and Open Questions

- **`metal_kernel` API limitations (HIGH):** the 10 TurboQuant +
  PolarQuant kernels use 10-14 buffer bindings, threadgroup shared
  memory (up to 4×4096 bytes), and custom dispatch geometries.
  The outlier variants may require even more buffers. `metal_kernel`
  may not support all of these. The Phase 2.5 spike is designed to
  validate this before committing engineering effort to Phase 3. If
  `metal_kernel` is insufficient, a hybrid approach (MLX for standard
  ops, direct Metal for TurboQuant/PolarQuant) remains viable.

- **mlx-c reference counting:** MLX C++ objects use reference counting
  exposed via `mlx_retain`/`mlx_free` in the C API. The safe Rust
  wrapper must implement `Clone` (retain) and `Drop` (free) correctly
  to avoid leaks or double-frees. This is a correctness-critical piece
  of `ironmill-mlx-sys` and should be tested with leak-detection tools
  (Instruments Leaks, `MallocStackLogging`).

- **Per-layer eval() overhead:** with TurboQuant, each layer requires
  an `mlx_eval()` to materialize the KV cache. This is a full GPU
  synchronization point. The realistic fusion window is limited to
  ~5 ops (region A) and ~9 ops (region B) per layer — meaningful but
  not as dramatic as the "single eval at the end" FP16 path. Profile
  to determine if per-layer sync overhead outweighs fusion savings.

- **mlx-c stability:** the C API is younger than the Python/C++ API.
  Breaking changes between mlx-c versions could require binding
  updates. Pin to a specific mlx release.

- **Custom kernel JIT overhead:** `metal_kernel` JIT-compiles on first
  call. Cache the compiled kernel handle to avoid per-step compilation
  latency. MLX does internal caching, but verify it works correctly
  for kernels called in a tight decode loop.

- **GPU framework contention:** both Metal/MPS and MLX submit work to
  the same GPU. Even with only one backend "active," initializing both
  frameworks allocates GPU resources (command queues, memory pools).
  Only one backend should be initialized per process.

- **Memory unpredictability:** MLX's automatic intermediate buffer
  management makes peak memory harder to predict than the MPS
  backend's pre-allocated `IntermediateBuffers`. Monitor memory usage
  during long sequence generation.

- **Build complexity:** mlx-c requires CMake + C++ toolchain. The
  `build.rs` must handle this gracefully or support a pre-built
  library path via `MLX_DIR`.

## Decision Record

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Integration level | New backend, not replacement | MPS backend is stable and well-tested |
| FFI target | mlx-c (C API) | Rust FFI to C is straightforward; avoids cxx bridge complexity |
| TurboQuant approach | Adapt all 6 TurboQuant + 4 PolarQuant kernel bodies for `metal_kernel` API | Kernel math is proven; signatures need rework for MLX dispatch model |
| Feature flag | Independent `mlx` feature | No coupling to `metal` feature |
| KV cache first pass | FP16 via MLX built-in SDPA | Simplest path to validate end-to-end correctness |
| Weight loading | Copy into MLX-owned memory | Avoids lifetime hazards with mmap'd `Cow::Borrowed` data |
| Custom kernel validation | Phase 2.5 spike before Phase 3 | De-risk `metal_kernel` API limitations early |
