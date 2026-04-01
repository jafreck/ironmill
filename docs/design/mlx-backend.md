# MLX Backend

## Overview

Add MLX as an additional GPU inference backend alongside the existing
Metal/MPS backend. MLX is Apple's array framework for Apple Silicon,
offering lazy evaluation with automatic kernel fusion and optimized
built-in operations. The MLX backend targets the same `InferenceEngine`
trait and reuses the same `WeightProvider` abstraction, TurboQuant
algorithm, and model configuration types.

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
- `gpu::ops` — `GpuPipelines` compiles and dispatches 8 custom `.metal` kernels
- `gpu::turboquant` — INT8 KV cache with Hadamard rotation, per-head absmax scaling
- `gpu::weights` — loads `WeightProvider` tensors into `MetalBuffer`s
- `gpu::inference` — `GpuInference` implements `InferenceEngine`, orchestrates the pipeline

All ops are encoded into a single `MTLCommandBuffer` per step. No
per-op synchronization, no intermediate CPU readback (except FP16 KV
cache path which requires a flush).

## Proposed Architecture (MLX Backend)

### Pipeline Mapping

```
Full decode pipeline (single token):
  0. embedding lookup    [mlx.embedding]
  ─── per layer (×num_layers) ───
  1. rms_norm           [mlx.fast.rms_norm]
  2. Q/K/V projection   [mlx.matmul]            ← fused with rms_norm by lazy eval
  3. RoPE               [mlx.fast.rope]
  4. cache-write         [custom Metal kernel]   ← TurboQuant rotate + quantize
  5. attention           [custom Metal kernel]   ← TurboQuant dequant + attention
  6. output projection   [mlx.matmul]
  7. residual add        [mlx.add]               ← fused by lazy eval
  8. rms_norm            [mlx.fast.rms_norm]
  9. gate/up projection  [mlx.matmul ×2]
 10. silu + gate mul     [mlx.multiply(mlx.sigmoid(gate) * gate, up)]  ← fused
 11. down projection     [mlx.matmul]
 12. residual add        [mlx.add]               ← fused by lazy eval
  ─── end per layer ───
 13. final rms_norm      [mlx.fast.rms_norm]
 14. lm_head projection  [mlx.matmul]
 15. mx.eval()           ← single materialization point
```

Steps 1-2, 7, 10, 12 become fusion candidates under MLX's lazy
evaluation. TurboQuant cache-write and attention (steps 4-5) remain
custom Metal kernels dispatched via `mx.fast.metal_kernel()`.

### Crate Layout

```
crates/
  ironmill-mlx-sys/                  NEW — FFI bindings to mlx-c
    Cargo.toml
    build.rs                         find/build mlx-c, bindgen
    src/
      lib.rs                         safe wrappers: MlxArray, MlxStream, MlxDevice
      array.rs                       array creation, data access, dtype mapping
      ops.rs                         matmul, rms_norm, rope, add, multiply, etc.
      metal_kernel.rs                custom Metal kernel registration + dispatch
      stream.rs                      stream management, eval, synchronize

  ironmill-inference/
    Cargo.toml                       add: ironmill-mlx-sys = { optional = true }
    src/
      mlx/                           NEW — MLX backend module
        mod.rs
        config.rs                    MlxConfig (analogous to GpuConfig)
        inference.rs                 MlxInference: impl InferenceEngine
        weights.rs                   WeightProvider → MlxArray loading
        turboquant/
          mod.rs                     MlxTurboQuantModel
          cache.rs                   MlxKvCache (INT8 arrays + scale arrays)
```

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

The `metal` and `mlx` features are independent. Both can be enabled
simultaneously — backend selection happens at runtime via the
`InferenceEngine` trait.

### Weight Loading

MLX arrays can be created from raw pointers without copying. The
existing `WeightProvider` trait (in `mil_rs::weights`) provides tensor
data as `Cow<[u8]>` with shape and dtype — this maps directly to
`mlx_array_new_data()`:

```rust
// mlx/weights.rs
fn load_weight(stream: &MlxStream, provider: &dyn WeightProvider, name: &str)
    -> Result<MlxArray, MlxError>
{
    let tensor = provider.tensor(name)?;
    // Zero-copy when data is mmap'd (Cow::Borrowed)
    MlxArray::from_data(&tensor.data, &tensor.shape, tensor.dtype.to_mlx(), stream)
}
```

### TurboQuant Integration

TurboQuant's core operations are:
1. **Cache write:** Hadamard rotate → absmax scale → quantize to INT8 → store
2. **Attention:** dequantize K/V → scale → QK matmul → causal mask → softmax → ×V

These are fused multi-step operations with custom memory layouts.
MLX's built-in quantization doesn't support per-head-per-position
dynamic scaling or Hadamard rotation. These stay as custom Metal
kernels.

The existing `.metal` shader source files (`turboquant.metal`,
`attention.metal`) can be reused via `mx.fast.metal_kernel()`:

```rust
// mlx/turboquant/cache.rs
pub struct MlxKvCache {
    k_caches: Vec<MlxArray>,      // [num_kv_heads, max_seq_len, head_dim] INT8
    v_caches: Vec<MlxArray>,      // same
    k_scales: Vec<MlxArray>,      // [num_kv_heads, max_seq_len] FP32
    v_scales: Vec<MlxArray>,      // same
    rotation_matrix: MlxArray,    // [head_dim, head_dim] FP16
}

impl MlxKvCache {
    pub fn write(&mut self, layer: usize, k: &MlxArray, v: &MlxArray, pos: usize,
                 stream: &MlxStream) -> Result<(), MlxError>
    {
        // Dispatch turboquant_cache_write custom kernel
        // Inputs: k, v, rotation_matrix, scale buffers
        // Outputs: quantized cache slices, updated scales
        mlx_fast_metal_kernel(
            "turboquant_cache_write",
            &[k, v, &self.rotation_matrix, &self.k_scales[layer]],
            &[&self.k_caches[layer]],
            TURBOQUANT_METAL_SOURCE,
            grid, threadgroup, stream,
        )
    }
}
```

### Synchronization Strategy

MLX's lazy evaluation defers computation until `mx.eval()` is called.
The decode pipeline needs two synchronization points:

1. **After cache write (step 4):** the quantized KV cache must be
   materialized before attention reads it, because attention indexes
   into the cache at specific positions.
2. **After lm_head (step 14):** logits must be materialized for
   CPU-side sampling.

Everything between these points benefits from lazy fusion:

```rust
fn run_pipeline(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
    let mut x = self.embedding_lookup(tokens);

    for layer in 0..self.num_layers {
        // Steps 1-3: norm → projections → RoPE (lazy, fused by MLX)
        let normed = mlx_fast_rms_norm(&x, &self.weights[layer].input_norm, eps);
        let q = mlx_matmul(&normed, &self.weights[layer].q_proj);
        let k = mlx_matmul(&normed, &self.weights[layer].k_proj);
        let v = mlx_matmul(&normed, &self.weights[layer].v_proj);
        let (q, k) = mlx_fast_rope(&q, &k, self.seq_pos);

        // Step 4: TurboQuant cache write — must materialize
        self.kv_cache.write(layer, &k, &v, self.seq_pos, &self.stream)?;
        mlx_eval(&[&self.kv_cache.k_caches[layer], &self.kv_cache.v_caches[layer]]);

        // Step 5: TurboQuant attention (custom kernel)
        let attn_out = self.turboquant_attention(layer, &q)?;

        // Steps 6-12: output proj → residual → FFN (lazy, fused by MLX)
        let proj = mlx_matmul(&attn_out, &self.weights[layer].o_proj);
        x = mlx_add(&x, &proj);
        let normed2 = mlx_fast_rms_norm(&x, &self.weights[layer].post_norm, eps);
        let gate = mlx_matmul(&normed2, &self.weights[layer].gate_proj);
        let up = mlx_matmul(&normed2, &self.weights[layer].up_proj);
        let ffn = mlx_silu_gate(&gate, &up);
        let down = mlx_matmul(&ffn, &self.weights[layer].down_proj);
        x = mlx_add(&x, &down);
    }

    // Steps 13-14: final norm → lm_head
    let normed = mlx_fast_rms_norm(&x, &self.final_norm, eps);
    let logits = mlx_matmul(&normed, &self.lm_head);

    // Materialize logits for sampling
    mlx_eval(&[&logits]);
    Ok(logits.to_cpu())
}
```

### FP16 KV Cache Path

For non-TurboQuant inference (standard FP16 KV cache), the entire
pipeline can run without intermediate `eval()` calls. MLX's built-in
`mx.fast.scaled_dot_product_attention` handles the KV cache natively.
This is the simplest integration path and a good starting point.

## Implementation Plan

### Phase 1: ironmill-mlx-sys

1. Set up crate with `build.rs` that finds or builds `mlx-c`
2. Generate bindings with `bindgen` for the mlx-c headers
3. Write safe Rust wrappers for: `MlxArray`, `MlxStream`, `MlxDevice`
4. Wrap core ops: `matmul`, `add`, `multiply`, `reshape`, `transpose`
5. Wrap `mlx_fast_*`: `rms_norm`, `rope`, `scaled_dot_product_attention`
6. Wrap `mlx_fast_metal_kernel` for custom kernel dispatch
7. Wrap `mlx_eval` and `mlx_async_eval`

### Phase 2: MLX Backend — FP16 KV Cache

1. Create `ironmill-inference/src/mlx/` module structure
2. Implement `MlxInference` with `InferenceEngine` trait
3. Weight loading via `WeightProvider → MlxArray`
4. Full decode pipeline using MLX built-in ops
5. FP16 KV cache using MLX's `scaled_dot_product_attention`
6. Benchmark against MPS backend on Qwen3-0.6B, LLaMA-3.2-1B

### Phase 3: TurboQuant on MLX

1. Port `turboquant_cache_write` kernel via `metal_kernel`
2. Port `turboquant_attention` kernel via `metal_kernel`
3. Implement `MlxKvCache` with INT8 arrays + scale tracking
4. Wire TurboQuant into `MlxInference` decode loop
5. Verify numerical parity with MPS TurboQuant output
6. Benchmark INT8 path against MPS backend

### Phase 4: Optimization

1. Profile `eval()` placement — minimize sync points
2. Experiment with `async_eval` for prefill chunking
3. Benchmark prefill throughput (MLX batched matmul vs MPS)
4. Evaluate MLX's quantized matmul for weight-only quantization

## Risks and Open Questions

- **mlx-c stability:** the C API is younger than the Python/C++ API.
  Breaking changes between mlx-c versions could require binding updates.
  Pin to a specific mlx release.

- **Custom kernel overhead:** `mx.fast.metal_kernel` JIT-compiles on
  first call. Cache the compiled kernel to avoid per-step compilation
  latency.

- **Synchronization overhead:** each `mx.eval()` call is a full GPU
  sync. If TurboQuant requires per-layer eval, this could negate
  fusion benefits. Profile carefully. Consider batching multiple
  layers between eval points if cache indexing allows it.

- **Memory management:** MLX manages its own memory pool. Running MLX
  alongside MPS (both features enabled) could cause memory pressure.
  Document that only one backend should be active at a time.

- **Build complexity:** mlx-c requires CMake + C++ toolchain. The
  `build.rs` must handle this gracefully or support a pre-built
  library path.

## Decision Record

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Integration level | New backend, not replacement | MPS backend is stable and well-tested |
| FFI target | mlx-c (C API) | Rust FFI to C is straightforward; avoids cxx bridge complexity |
| TurboQuant approach | Reuse existing `.metal` source | Proven kernels, numerical parity guaranteed |
| Feature flag | Independent `mlx` feature | No coupling to `metal` feature |
| KV cache first pass | FP16 via MLX built-in SDPA | Simplest path to validate end-to-end correctness |
