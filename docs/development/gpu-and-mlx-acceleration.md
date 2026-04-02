# GPU & MLX Acceleration — Implementation Spec

## Goal

Close the ~7–10× throughput gap with llama.cpp on Apple Silicon via two
parallel work streams:

1. **Metal performance** — eliminate CPU sync, custom kernels, flash
   attention, and operator fusion in the existing MPS backend.
2. **MLX backend** — add MLX as a second GPU backend with automatic
   kernel fusion via lazy evaluation.

Both streams share the same `InferenceEngine` trait, `WeightProvider`,
`ModelConfig`, TurboQuant algorithm, and sampling infrastructure.

## Current State

| Component | Status |
|-----------|--------|
| MPS backend (`gpu/`) | Working, 5.5 tok/s Qwen3-1.7B, 2.2 tok/s Qwen3-8B (M2 Max FP16) |
| TurboQuant INT4/INT8 KV cache | Working — Hadamard rotation, absmax, codebook, outlier, QJL |
| PolarQuant weight matmul | Working — 4 kernel variants (matvec/matmul × INT4/INT8) |
| MLX backend | Not started |
| FP16 KV cache path | Working but CPU-side scatter with mid-layer sync |
| Custom Metal kernels | 8 shader files, 15 kernel functions |
| Benchmark harness | `ironmill-bench` with `--backend` (CoreML) and `--metal` (separate flag) |

## Dependency Graph

```
MLX-SYS ──→ MLX-KERNEL-SPIKE ──┬──→ MLX-FP16 ──→ MLX-TURBOQUANT ──→ MLX-OPTIMIZE
                                │                       ↑
BENCH-UNIFY ────────────────────┘                       │
                                                        │
MTL-KV-SCATTER ───┬──→ MTL-CUSTOM-MATMUL ──┬──→ MTL-OP-FUSION (conditional)
                   │                        │
                   └──→ MTL-FLASH-ATTN ─────┴──→ (MLX-TURBOQUANT, cross-stream)
```

Parallelization windows:
- **Wave 1** (all parallel): `MLX-SYS`, `MTL-KV-SCATTER`, `BENCH-UNIFY`
- **Wave 2** (parallel pairs): `MTL-CUSTOM-MATMUL` ∥ `MTL-FLASH-ATTN`, `MLX-KERNEL-SPIKE`
- **Wave 3** (parallel): `MLX-FP16` ∥ `MTL-OP-FUSION`
- **Wave 4**: `MLX-TURBOQUANT`
- **Wave 5**: `MLX-OPTIMIZE`

---

## Tasks

### MLX-SYS: ironmill-mlx-sys crate

**Dependencies:** none

**Goal:** Safe Rust bindings to `mlx-c` (the C API for Apple's MLX
framework), following the same crate pattern as `ironmill-metal-sys`.

**Files to create:**
```
crates/ironmill-mlx-sys/
  Cargo.toml
  build.rs
  src/
    lib.rs
    array.rs
    device.rs
    stream.rs
    ops.rs
    fast_ops.rs
    metal_kernel.rs
    error.rs
```

**Files to modify:**
- `Cargo.toml` (workspace root) — add `"crates/ironmill-mlx-sys"` to
  `workspace.members`

**Implementation details:**

1. `build.rs`:
   - Check for `MLX_DIR` env var pointing to a pre-built mlx-c
     installation (headers + library).
   - If `MLX_DIR` is not set, build mlx-c from a vendored or fetched
     source via CMake (`cmake -B build -DMLX_BUILD_TESTS=OFF
     -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF`).
   - Use `bindgen` to generate raw FFI bindings from `mlx/c/mlx.h`.
   - Emit `cargo:rustc-link-lib=mlx` and appropriate link search paths.
   - Emit `cargo:rustc-link-lib=c++` for C++ runtime (mlx-c links
     against the C++ mlx library internally).

2. `array.rs` — `MlxArray` safe wrapper:
   ```rust
   pub struct MlxArray {
       raw: mlx_array,
   }

   impl Clone for MlxArray {
       fn clone(&self) -> Self {
           unsafe { mlx_retain(self.raw as _) };
           Self { raw: self.raw }
       }
   }

   impl Drop for MlxArray {
       fn drop(&mut self) {
           unsafe { mlx_free(self.raw as _) };
       }
   }
   ```
   Methods: `from_data_copy(data: &[u8], shape: &[usize], dtype, stream)`,
   `from_scalar(val, dtype, stream)`, `shape() -> Vec<usize>`,
   `dtype() -> MlxDtype`, `as_slice<T>() -> &[T]` (after eval),
   `ndim()`, `size()`, `item_size()`.

3. `device.rs` — `MlxDevice` wrapper: `default_gpu()`, `default_cpu()`.

4. `stream.rs` — `MlxStream` wrapper: `new(device)`, `default_gpu()`.
   Also wrap `mlx_eval(outputs: &[&MlxArray])` and
   `mlx_async_eval(outputs: &[&MlxArray])`.

5. `ops.rs` — wrap core operations. Each function takes `&MlxStream`:
   - `mlx_matmul(a, b) -> MlxArray`
   - `mlx_add(a, b) -> MlxArray`
   - `mlx_multiply(a, b) -> MlxArray`
   - `mlx_reshape(a, shape) -> MlxArray`
   - `mlx_transpose(a, axes) -> MlxArray`
   - `mlx_silu(a) -> MlxArray`
   - `mlx_slice(a, starts, stops, strides) -> MlxArray`
   - `mlx_expand_dims(a, axis) -> MlxArray`

6. `fast_ops.rs` — wrap `mlx_fast_*` optimized operations:
   - `mlx_fast_rms_norm(x, weight, eps) -> MlxArray`
   - `mlx_fast_rope(x, dims, traditional, base, scale, offset) -> MlxArray`
   - `mlx_fast_scaled_dot_product_attention(q, k, v, scale, mask) -> MlxArray`

7. `metal_kernel.rs` — wrap `mlx_fast_metal_kernel()`:
   ```rust
   pub fn mlx_fast_metal_kernel(
       name: &str,
       inputs: &[&MlxArray],
       outputs: &[&MlxArray],
       source: &str,
       grid: [usize; 3],
       threadgroup: [usize; 3],
       output_shapes: &[&[usize]],
       output_dtypes: &[MlxDtype],
       stream: &MlxStream,
   ) -> Result<Vec<MlxArray>, MlxSysError>;
   ```

8. `error.rs`:
   ```rust
   #[derive(Debug, thiserror::Error)]
   pub enum MlxSysError {
       #[error("mlx-c error: {0}")]
       MlxC(String),
       #[error("kernel compilation failed: {0}")]
       KernelCompile(String),
       #[error("invalid dtype: expected {expected}, got {got}")]
       InvalidDtype { expected: String, got: String },
       #[error("build error: {0}")]
       Build(String),
   }
   ```

9. `lib.rs` — re-export all public types and define `MlxDtype` enum
   mapping to mlx-c dtype constants. Map `mil_rs` dtype values to
   `MlxDtype` via a `to_mlx()` method or `From` impl.

**Acceptance criteria:**
- `cargo check -p ironmill-mlx-sys` passes (on macOS with mlx-c available).
- Unit test: create array from `&[f32]`, `mlx_matmul`, `mlx_eval`, read
  result back, verify correctness.
- Unit test: `Clone` + `Drop` in tight loop (1M iterations) does not leak
  (verify with Instruments or `MallocStackLogging`).
- Unit test: `mlx_fast_rms_norm` matches CPU reference within 1e-5.
- Unit test: `mlx_fast_metal_kernel` dispatches a trivial kernel (e.g.
  element-wise add) and returns correct results.

---

### MTL-KV-SCATTER: Eliminate CPU sync points

**Dependencies:** none

**Goal:** Remove the per-layer `wait_until_completed()` + CPU readback
in the FP16 KV cache path. This is the single largest bottleneck —
it drains the GPU pipeline every layer, every token.

**Files to create:**
- `crates/ironmill-inference/src/gpu/shaders/kv_scatter.metal`

**Files to modify:**
- `crates/ironmill-inference/src/gpu/inference.rs` — remove CPU
  readback loop (~lines 901–942), replace with GPU kernel dispatch;
  remove mid-layer `commit()`/`wait_until_completed()`; defer logit
  readback to after all layers complete.
- `crates/ironmill-inference/src/gpu/ops.rs` — add pipeline compilation
  and dispatch helper for `kv_scatter` kernel.

**Implementation details:**

1. **`kv_scatter.metal` kernel:**
   ```metal
   // Scatter Q/K/V projections into KV cache on GPU.
   // Input: projection buffer [token_count × num_kv_heads × head_dim] FP16
   // Output: cache buffer [num_kv_heads × max_seq_len × head_dim] FP16
   // Parameters: seq_pos, token_count, num_kv_heads, head_dim, max_seq_len
   kernel void kv_scatter(
       device const half* proj      [[buffer(0)]],
       device half* cache           [[buffer(1)]],
       constant uint& seq_pos       [[buffer(2)]],
       constant uint& token_count   [[buffer(3)]],
       constant uint& num_kv_heads  [[buffer(4)]],
       constant uint& head_dim      [[buffer(5)]],
       constant uint& max_seq_len   [[buffer(6)]],
       uint3 tid                    [[thread_position_in_grid]])
   {
       // tid.x = element within head_dim
       // tid.y = head index
       // tid.z = token index
       uint t = tid.z;
       uint head = tid.y;
       uint d = tid.x;
       if (t >= token_count || head >= num_kv_heads || d >= head_dim) return;

       uint src_idx = (t * num_kv_heads + head) * head_dim + d;
       uint dst_idx = (head * max_seq_len + (seq_pos + t)) * head_dim + d;
       cache[dst_idx] = proj[src_idx];
   }
   ```

2. **inference.rs changes:**
   - In the FP16 KV cache branch of the decode/prefill loop, replace
     the block at ~lines 901–942 (commit + wait + CPU readback + scatter
     loop) with two `kv_scatter` kernel dispatches (one for K, one for V).
   - Remove the mid-layer `cmd_buf.commit()` / `cmd_buf.wait_until_completed()`
     pair. The entire layer's work (projections + KV scatter + attention)
     stays on a single command buffer.
   - Move the final `cmd_buf.commit()` + `wait_until_completed()` +
     logit readback to after the last layer, just before sampling.
   - The command buffer is now allocated once before the layer loop and
     committed once after.

3. **ops.rs changes:**
   - Add `kv_scatter` to the shader source compilation in `GpuPipelines`.
   - Add `dispatch_kv_scatter(encoder, proj_buf, cache_buf, seq_pos,
     token_count, num_kv_heads, head_dim, max_seq_len)` helper.
   - Grid size: `(head_dim, num_kv_heads, token_count)`.
   - Threadgroup size: `(min(head_dim, 256), 1, 1)` — each thread
     copies one element.

**Acceptance criteria:**
- FP16 decode produces identical logits to the current CPU-scatter path
  (max abs diff < 1e-6 on FP16 values, i.e. bit-exact after rounding).
- No `wait_until_completed()` calls inside the layer loop for the FP16
  path (only one at the end for logit readback).
- `cargo test -p ironmill-inference` passes with `--features metal`.
- Benchmark: ≥1.5× tok/s improvement on Qwen3-1.7B FP16 (M2 Max).

---

### BENCH-UNIFY: Benchmark harness unification

**Dependencies:** none

**Goal:** Consolidate the `--backend` and `--metal` CLI flags into a
single `--backend` flag that selects any backend, including the future
MLX backend.

**Files to modify:**
- `crates/ironmill-bench/src/main.rs` — replace `--metal: bool` with
  `metal` and `mlx` as valid `--backend` values.
- `crates/ironmill-bench/Cargo.toml` — add `mlx` feature forwarding
  to `ironmill-inference/mlx`.

**Implementation details:**

1. Replace the current CLI args:
   ```rust
   // BEFORE:
   #[arg(short, long)]
   backend: Vec<String>,
   #[arg(long)]
   metal: bool,

   // AFTER:
   /// Backends to benchmark. May be specified multiple times.
   /// Values: coreml-cpu, coreml-gpu, coreml-ane, coreml-all, metal, mlx
   #[arg(short, long, value_delimiter = ',')]
   backend: Vec<Backend>,
   ```

2. Define the `Backend` enum:
   ```rust
   #[derive(Clone, Debug, PartialEq, clap::ValueEnum)]
   enum Backend {
       /// CoreML with CPU-only compute units
       CoremlCpu,
       /// CoreML with CPU + GPU compute units
       CoremlGpu,
       /// CoreML with CPU + ANE compute units
       CoremlAne,
       /// CoreML with all compute units
       CoremlAll,
       /// Direct Metal GPU backend (custom kernels + MPS)
       Metal,
       /// MLX GPU backend (lazy evaluation + automatic fusion)
       #[cfg(feature = "mlx")]
       Mlx,
   }
   ```

3. Route each variant:
   - `Backend::CoremlCpu` → existing CoreML path with `ComputeUnits::CpuOnly`.
   - `Backend::CoremlGpu` → existing CoreML path with `ComputeUnits::CpuAndGpu`.
   - `Backend::CoremlAne` → existing CoreML path with `ComputeUnits::CpuAndNeuralEngine`.
   - `Backend::CoremlAll` → existing CoreML path with `ComputeUnits::All`.
   - `Backend::Metal` → existing `GpuInference` path.
   - `Backend::Mlx` → future `MlxInference` path (behind
     `#[cfg(feature = "mlx")]`; compiles to an error message if the
     feature is disabled).

4. Cargo.toml features:
   ```toml
   [features]
   default = []
   ane-direct = ["dep:ironmill-iosurface"]
   metal = ["ironmill-inference/metal"]
   mlx = ["ironmill-inference/mlx"]
   ```

**Acceptance criteria:**
- `--backend metal` works identically to the old `--metal` flag.
- `--backend mlx` prints a clear error if the `mlx` feature is not
  enabled at compile time.
- `--backend metal --backend coreml-cpu` runs both backends and reports
  results for each.
- `--metal` flag is removed (breaking change is acceptable).
- `cargo check -p ironmill-bench --features metal` passes.

---

### MTL-CUSTOM-MATMUL: Custom FP16 matvec kernel + weight pre-packing

**Dependencies:** `MTL-KV-SCATTER`

Depends on `MTL-KV-SCATTER` because both modify `inference.rs` dispatch
logic. After KV-SCATTER lands, inference.rs has a clean single-command-
buffer structure that this task builds on.

**Goal:** Replace MPS `MPSMatrixMultiplication` with a custom SIMD-group
matvec kernel for decode (batch=1). Keep MPS for prefill (batch>1).
Pre-pack weights into blocked format at load time.

**Files to create:**
- `crates/ironmill-inference/src/gpu/shaders/matvec.metal`

**Files to modify:**
- `crates/ironmill-inference/src/gpu/inference.rs` — add dynamic
  dispatch: custom kernel for M=1, MPS for M>1.
- `crates/ironmill-inference/src/gpu/ops.rs` — add pipeline compilation
  and dispatch for `matvec` kernel.
- `crates/ironmill-inference/src/gpu/weights.rs` — add weight
  pre-packing into blocked format at load time.

**Implementation details:**

1. **`matvec.metal` kernel** — FP16 matrix-vector product using
   `simdgroup_matrix` 8×8 tiles:
   ```
   // y = x · W^T where x is [1, K] and W is [N, K]
   // Tiled: each simdgroup computes a 8-row output tile.
   // Load W in 8×8 blocks using simdgroup_matrix_storage.
   // Load x into threadgroup memory once, reuse across all N tiles.
   ```
   - Each threadgroup handles a tile of output rows (e.g. 64 rows).
   - Within the threadgroup, simdgroups (32 threads) process 8 output
     rows each using `simdgroup_matrix` for the K-dimension reduction.
   - x is loaded into threadgroup memory once per tile of K.
   - Weight layout assumption: blocked [N/8, K/8, 8, 8] FP16.

2. **Weight pre-packing** in `weights.rs`:
   - At load time, after creating the `MetalBuffer` for each dense
     weight, transpose and tile into [N/8, K/8, 8, 8] blocked format.
   - Store both the original (for MPS prefill) and packed (for custom
     matvec decode) buffers. The packed buffer is an additional
     allocation.
   - For quantized weights (PolarQuant), no packing change — they
     already use their own layout.

3. **Dynamic dispatch** in `inference.rs`:
   - If `token_count == 1` (decode), use the custom `matvec` kernel
     for all dense linear projections (Q, K, V, O, gate, up, down,
     lm_head).
   - If `token_count > 1` (prefill), use MPS `MPSMatrixMultiplication`
     with the original unpacked weights.
   - The dispatch decision is per-step, not per-model.

**Acceptance criteria:**
- Custom matvec produces results within 1e-3 of MPS on random inputs.
- Decode (M=1) uses the custom kernel; prefill (M>1) uses MPS.
- `cargo test -p ironmill-inference --features metal` passes.
- Benchmark: ≥2× decode tok/s improvement over post-KV-SCATTER baseline
  on Qwen3-1.7B FP16.

---

### MLX-FP16: MLX FP16 backend

**Dependencies:** `MLX-SYS`, `MLX-KERNEL-SPIKE`, `BENCH-UNIFY`

Depends on `MLX-KERNEL-SPIKE` because quantized weight matmuls dispatch
PolarQuant kernels via `mlx_fast_metal_kernel()`. The spike must confirm
this API works before committing to the full backend.

**Goal:** Implement a complete FP16 inference backend using MLX built-in
operations. No custom Metal kernels — pure MLX lazy evaluation with a
single `eval()` at the end.

**Files to create:**
```
crates/ironmill-inference/src/mlx/
  mod.rs
  config.rs
  error.rs
  inference.rs
  weights.rs
```

**Files to modify:**
- `crates/ironmill-inference/Cargo.toml` — add `mlx` feature with
  `dep:ironmill-mlx-sys`.
- `crates/ironmill-inference/src/lib.rs` — add
  `#[cfg(feature = "mlx")] pub mod mlx;`

**Implementation details:**

1. **`config.rs`:**
   ```rust
   pub struct MlxConfig {
       pub max_seq_len: usize,
       pub prefill_chunk_size: Option<usize>,
       // Fields below are added in MLX-TURBOQUANT:
       pub enable_turboquant: bool,
       pub rotation_seed: u64,
       pub n_bits: u8,   // 4 or 8
       pub outlier_config: Option<OutlierConfig>,
       pub qjl_config: Option<QjlConfig>,
   }
   ```
   Initially only `max_seq_len` and `prefill_chunk_size` are used.
   The TurboQuant fields default to `false`/`0`/`None` and are
   activated by `MLX-TURBOQUANT`.

2. **`error.rs`:**
   ```rust
   #[derive(Debug, thiserror::Error)]
   pub enum MlxError {
       #[error("mlx sys: {0}")]
       Sys(#[from] ironmill_mlx_sys::MlxSysError),
       #[error("weight loading: {0}")]
       WeightLoading(String),
       #[error("shape mismatch: expected {expected}, got {got}")]
       Shape { expected: String, got: String },
   }

   impl From<MlxError> for InferenceError {
       fn from(e: MlxError) -> Self {
           InferenceError::Runtime(e.to_string())
       }
   }
   ```

3. **`weights.rs`** — load from `WeightProvider` into `MlxArray`s:
   - `MlxWeightBuffer` enum: `Dense(MlxArray)` or
     `Quantized(MlxQuantizedWeight)`.
   - `MlxLayerWeights`: input_norm, q/k/v/o_proj, post_attn_norm,
     gate/up/down_proj, optional q_norm/k_norm.
   - `MlxWeights`: embedding, layers, final_norm, lm_head, config.
   - `load_weight()`: match on `QuantizationInfo`:
     - `None` → `from_data_copy` to dense `MlxArray`.
     - `LutToDense` → unpack into indices/lut/norms `MlxArray`s
       (dispatched via PolarQuant kernel at inference time).
     - `AffineDequantize` → CPU dequant to dense, then `from_data_copy`.
   - Always copy data into MLX-owned memory (option 1 from design doc).

4. **`inference.rs`** — `MlxInference: InferenceEngine`:

   ```rust
   pub struct MlxArtifacts<'a> {
       pub weights: &'a dyn WeightProvider,
       pub config: MlxConfig,
   }

   pub struct MlxInference {
       stream: MlxStream,
       weights: Option<MlxWeights>,
       config: MlxConfig,
       // FP16 KV cache managed by MLX's SDPA
       k_cache: Vec<Option<MlxArray>>,  // per-layer
       v_cache: Vec<Option<MlxArray>>,  // per-layer
       seq_pos: usize,
   }
   ```

   - `load()`: downcast to `MlxArtifacts`, call `MlxWeights::load()`,
     initialize empty KV cache vectors.
   - `decode_step(token)`: run full pipeline with lazy MLX ops:
     - Embedding lookup
     - Per-layer: rms_norm → Q/K/V matmul (dense or PolarQuant) →
       optional Q/K norm → RoPE → update KV cache →
       `mlx_fast_scaled_dot_product_attention` → O proj → residual →
       rms_norm → gate/up matmul → SiLU + gate multiply → down matmul
       → residual
     - Final rms_norm → lm_head matmul
     - Single `mlx_eval()` to materialize logits
     - Read logits back to CPU for sampling
   - `prefill(tokens)`: chunk tokens by `prefill_chunk_size`, run
     pipeline per chunk with multi-token Q and causal masking via SDPA.
   - `reset()`: clear KV caches and seq_pos.

   For quantized weight matmuls, dispatch via PolarQuant custom kernels
   using `mlx_fast_metal_kernel()` with adapted kernel source. For
   dense weights, use `mlx_matmul()`.

   FP16 KV cache: accumulate K/V per layer using MLX array concatenation
   or the SDPA cache mechanism. After each layer's SDPA call, update the
   stored K/V cache arrays. No custom kernels needed — MLX handles KV
   cache natively.

5. **`mod.rs`** — re-export `MlxInference`, `MlxArtifacts`, `MlxConfig`,
   `MlxError`.

**Acceptance criteria:**
- `cargo check -p ironmill-inference --features mlx` passes.
- `ironmill-bench --backend mlx` runs Qwen3-0.6B FP16 end-to-end and
  produces coherent text.
- Numerical parity with MPS FP16 output: max abs diff < 1e-3 on logits
  for the same prompt.
- No `mlx_eval()` calls inside the layer loop (single eval at the end).
- `cargo test -p ironmill-inference --features mlx` passes.

---

### MTL-FLASH-ATTN: Flash attention

**Dependencies:** `MTL-KV-SCATTER`

Depends on `MTL-KV-SCATTER` because the single-command-buffer refactor
must land first — flash attention assumes no mid-layer GPU sync.

**Goal:** Replace the sequential KV scan in `standard_attention` and
`turboquant_attention` with tiled flash attention. Load KV cache in
tiles into threadgroup SRAM to reduce global memory bandwidth.

**Files to modify:**
- `crates/ironmill-inference/src/gpu/shaders/attention.metal` — rewrite
  `standard_attention` kernel.
- `crates/ironmill-inference/src/gpu/shaders/turboquant.metal` — rewrite
  `turboquant_attention` and `turboquant_outlier_attention` kernels.
- `crates/ironmill-inference/src/gpu/ops.rs` — update dispatch
  parameters (threadgroup size, shared memory size).

**Implementation details:**

1. **Tiled attention algorithm** (for `standard_attention`):
   ```
   TILE_SIZE = 64  // KV positions per tile

   // Load Q [1, head_dim] into threadgroup memory once
   threadgroup float q_tile[head_dim];
   load_q_into_shared(q_tile, q_buf, head, head_dim);
   threadgroup_barrier();

   float running_max = -INFINITY;
   float running_sum = 0.0;
   float acc[head_dim] = {0};

   for (tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
       tile_end = min(tile_start + TILE_SIZE, seq_len);
       actual_tile = tile_end - tile_start;

       // Load K tile [TILE_SIZE, head_dim] into threadgroup memory
       threadgroup float k_tile[TILE_SIZE][head_dim];
       cooperative_load(k_tile, k_cache, head, tile_start, actual_tile);
       threadgroup_barrier();

       // Compute QK^T for this tile: scores[TILE_SIZE]
       float scores[TILE_SIZE];
       for (p = 0; p < actual_tile; p++) {
           scores[p] = dot(q_tile, k_tile[p]) * scale;
       }

       // Online softmax update with tile scores
       float tile_max = max(scores[0..actual_tile]);
       float old_max = running_max;
       running_max = max(running_max, tile_max);
       float correction = exp(old_max - running_max);
       running_sum = running_sum * correction;

       // Rescale accumulator for new max
       for (d = 0; d < head_dim; d++) acc[d] *= correction;

       // Load V tile and accumulate
       threadgroup float v_tile[TILE_SIZE][head_dim];
       cooperative_load(v_tile, v_cache, head, tile_start, actual_tile);
       threadgroup_barrier();

       for (p = 0; p < actual_tile; p++) {
           float w = exp(scores[p] - running_max);
           running_sum += w;
           for (d = 0; d < head_dim; d++) acc[d] += w * v_tile[p][d];
       }
   }

   // Final normalize
   for (d = 0; d < head_dim; d++) output[d] = acc[d] / running_sum;
   ```

2. **Threadgroup sizing:**
   - Threadgroup: `(head_dim, 1, 1)` — one threadgroup per head.
   - Shared memory: K and V tiles are never live simultaneously
     (K is consumed for scores before V is loaded), so they can alias
     the same threadgroup allocation. Total shared memory:
     `TILE_SIZE * head_dim * sizeof(type) + head_dim * sizeof(float)`
     (one KV tile + Q).
   - For TILE_SIZE=64, head_dim=128, half tiles:
     64×128×2 + 128×4 = 16,384 + 512 = 16,896 bytes (well within 32KB).
   - For TILE_SIZE=32, head_dim=128, float tiles:
     32×128×4 + 128×4 = 16,384 + 512 = 16,896 bytes.
   - If separate K/V tiles are preferred (simpler code), use half
     precision: `TILE_SIZE * head_dim * 2 * 2 + head_dim * 4`.
     TILE_SIZE=32: 32×128×2×2 + 128×4 = 16,384 + 512 = 16,896 bytes.
     TILE_SIZE=64: 64×128×2×2 + 128×4 = 32,768 + 512 = 33,280 bytes
     (exceeds 32KB; drop Q from shared or alias K/V tiles).
3. **TurboQuant flash attention** — same tiling strategy but with
   dequantization in threadgroup memory:
   - Load quantized K tile + scales into shared memory.
   - Dequantize in-place: `k_tile[p][d] = (float)k_quant[p][d] * scale[p]`.
   - Then proceed with standard tiled attention math.
   - Same for V tile.

4. **Dispatch changes in `ops.rs`:**
   - Update `dispatch_attention` to pass `TILE_SIZE` as a kernel
     constant or buffer parameter.
   - Set threadgroup memory size in the dispatch call.

**Acceptance criteria:**
- Flash `standard_attention` matches naive attention within 1e-4 on
  random inputs for seq_len = 1, 64, 512, 2048.
- Flash `turboquant_attention` matches naive TurboQuant attention within
  1e-3 (quantization adds noise).
- Flash `turboquant_outlier_attention` matches naive outlier attention.
- `cargo test -p ironmill-inference --features metal` passes.
- Benchmark: measurable improvement at seq_len ≥ 512 on Qwen3-8B.
  Target ~1.5× at seq_len=2048.

---

### MLX-KERNEL-SPIKE: MLX custom Metal kernel validation

**Dependencies:** `MLX-SYS`

**Goal:** Validate that MLX's `mlx_fast_metal_kernel()` API supports
the requirements for TurboQuant and PolarQuant kernels before
committing to the full port. This is a go/no-go gate.

**Files to create:**
- `crates/ironmill-inference/src/mlx/kernel_spike.rs` (test module,
  gated behind `#[cfg(test)]`)

**Implementation details:**

Test the following with `mlx_fast_metal_kernel()`:

1. **Basic kernel dispatch:** port `rms_norm` or `silu_gate` as a
   proof of concept. Verify output matches `mlx_fast_rms_norm`.

2. **Threadgroup shared memory:** write a kernel that declares
   `threadgroup float shared[4096]` (matching TurboQuant attention's
   requirement of up to 4×4096 bytes). Verify it compiles and runs.

3. **High buffer count:** dispatch a kernel with 14 input+output
   arrays (matching TurboQuant attention's 14 buffer bindings).
   Use dummy data. Verify all buffers are accessible.

4. **Even higher buffer count:** test with 18+ arrays for the outlier
   attention variant.

5. **Dispatch geometry:** test non-trivial grid sizes like
   `(head_dim, num_kv_heads, 1)` and verify thread position indexing.

6. **JIT compilation overhead:** time the first call vs. subsequent
   calls to verify MLX's internal kernel caching works. The first call
   may JIT-compile; subsequent calls should be fast.

**Acceptance criteria — go/no-go:**
- ✅ GO if all 6 tests pass: threadgroup shared memory works, ≥14
  buffers supported, dispatch geometry is flexible, JIT caching works.
- ❌ NO-GO if threadgroup shared memory or buffer count is limited.
  In that case, document the limitations and evaluate:
  - Use MLX's C++ custom op API via `cxx` bridge, or
  - Hybrid approach: MLX for standard ops, direct Metal dispatch for
    TurboQuant/PolarQuant kernels (dispatch via `ironmill-metal-sys`
    directly, feed results back to MLX via `MlxArray::from_data_copy`).

Document findings in the spike results. The go/no-go decision
determines whether `MLX-FP16` uses PolarQuant kernels via
`mlx_fast_metal_kernel` or falls back to CPU dequant, and whether
`MLX-TURBOQUANT` proceeds as designed or pivots to a hybrid approach.

---

### MLX-TURBOQUANT: MLX TurboQuant backend

**Dependencies:** `MLX-FP16`, `MLX-KERNEL-SPIKE`, `MTL-FLASH-ATTN`

Depends on `MTL-FLASH-ATTN` because the TurboQuant attention kernel
body should be ported *after* flash attention rewrites it — avoids
double-porting the kernel math. **Note:** this creates a cross-stream
dependency (MLX blocked on Metal). If `MTL-FLASH-ATTN` slips, consider
porting from the pre-flash kernels and updating when flash attention
lands, rather than blocking the entire MLX quantized path.

Depends on `MLX-KERNEL-SPIKE` for the go/no-go gate on the
`metal_kernel` API.

**Goal:** Port all TurboQuant and PolarQuant custom kernels to the MLX
backend, enabling INT4/INT8 KV cache inference via MLX.

**Files to create:**
```
crates/ironmill-inference/src/mlx/turboquant/
  mod.rs
  cache.rs
  kernels.rs
```

**Files to modify:**
- `crates/ironmill-inference/src/mlx/inference.rs` — add TurboQuant
  decode path with per-layer `eval()`.
- `crates/ironmill-inference/src/mlx/config.rs` — add TurboQuant
  fields to `MlxConfig`.
- `crates/ironmill-inference/src/mlx/weights.rs` — no changes expected
  (quantized weight loading already handled in MLX-FP16).

**Implementation details:**

1. **`cache.rs`** — `MlxKvCache` struct:
   ```rust
   pub struct MlxKvCache {
       // Standard INT4/INT8 cache
       k_caches: Vec<MlxArray>,      // [num_kv_heads, max_seq_len, head_dim]
       v_caches: Vec<MlxArray>,
       k_scales: Vec<MlxArray>,      // [num_kv_heads, max_seq_len]
       v_scales: Vec<MlxArray>,
       rotation_signs: MlxArray,     // [head_dim]

       // Outlier split (optional)
       outlier: Option<MlxOutlierCache>,

       // QJL correction (optional)
       qjl_matrix: Option<MlxArray>,
       k_qjl_signs: Option<Vec<MlxArray>>,
       k_r_norms: Option<Vec<MlxArray>>,
   }
   ```

2. **`kernels.rs`** — adapted kernel source strings:
   - Port the *body* of each Metal kernel from the MPS backend's
     `.metal` files, adapting function signatures from
     `[[buffer(N)]]` bindings to MLX's ordered input/output array
     convention.
   - Port from the *post-flash-attention* version of the kernels
     (i.e., after `MTL-FLASH-ATTN` lands).
   - Kernels to port:
     1. `turboquant_cache_write` (including QJL sign computation)
     2. `turboquant_attention` (including QJL residual correction)
     3. `turboquant_outlier_cache_write`
     4. `turboquant_outlier_attention`
     5. `polarquant_matvec_int4`
     6. `polarquant_matmul_int4`
     7. `polarquant_matvec_int8`
     8. `polarquant_matmul_int8`
   - Each kernel's source is stored as a `const &str` in this module.

3. **`mod.rs`** — `MlxTurboQuantModel`:
   ```rust
   pub struct MlxTurboQuantModel {
       pub rotation_signs: MlxArray,
       pub codebook: Vec<f32>,
       pub boundaries: Vec<f32>,
       pub n_bits: u8,
       pub outlier_config: Option<OutlierConfig>,
       pub qjl_config: Option<QjlConfig>,
   }
   ```
   - Initialize from `ModelConfig` + `MlxConfig` using the same
     codebook generation as the MPS backend (`codebook.rs`).

4. **Decode loop changes** in `inference.rs`:
   - When `config.enable_turboquant` is true, switch from FP16 SDPA
     path to TurboQuant path.
   - Per-layer structure becomes:
     ```
     Lazy region A: rms_norm + Q/K/V matmul + RoPE
     Cache write: dispatch turboquant_cache_write (or outlier variant)
     mlx_eval() — materialize cache + scales
     Attention: dispatch turboquant_attention (or outlier variant)
     Lazy region B: O proj + residual + rms_norm + gate/up + SiLU + down + residual
     (region B stays lazy, fuses with next layer's region A)
     ```
   - Final `mlx_eval()` after lm_head to materialize logits.

**Acceptance criteria:**
- INT8 TurboQuant decode produces logits within 1e-3 of MPS TurboQuant
  (same prompt, same rotation seed).
- INT4 outlier path produces logits within 1e-3 of MPS outlier path.
- QJL correction applies correctly (compare with/without QJL against
  MPS backend).
- PolarQuant INT4/INT8 weight matmul matches MPS within 1e-4.
- `cargo test -p ironmill-inference --features mlx` passes.
- `ironmill-bench --backend mlx` runs with `--turboquant` and produces
  coherent text.
- Per-layer `eval()` count equals `num_layers` (one per cache write) +
  1 (final logits).

---

### MTL-OP-FUSION: Metal operator fusion (conditional)

**Dependencies:** `MTL-FLASH-ATTN`, `MTL-CUSTOM-MATMUL`

**Conditional:** benchmark MLX vs. MPS after `MLX-FP16` completes. If
MLX with automatic fusion already matches or beats MPS, deprioritize
this task — MLX provides fusion for free via lazy evaluation.

**Goal:** Reduce per-layer kernel dispatch count by fusing common
operation sequences in the MPS backend.

**Files to create:**
- `crates/ironmill-inference/src/gpu/shaders/fused_residual_norm.metal`

**Files to modify:**
- `crates/ironmill-inference/src/gpu/shaders/activation.metal` — fuse
  gate+up matmul output with SiLU into the existing `silu_gate` kernel
  (eliminate intermediate buffer writes).
- `crates/ironmill-inference/src/gpu/shaders/attention.metal` — fold
  RoPE application into the attention kernel's Q load.
- `crates/ironmill-inference/src/gpu/ops.rs` — add fused kernel
  pipelines and dispatch helpers.
- `crates/ironmill-inference/src/gpu/inference.rs` — replace sequences
  of separate dispatches with fused kernel calls.

**Implementation details:**

1. **Fused residual + RMSNorm** (`fused_residual_norm.metal`):
   ```metal
   // Compute residual = a + b, then RMSNorm in one kernel.
   // Avoids writing residual to global memory then reading it back
   // for normalization.
   kernel void fused_residual_rms_norm(
       device const half* a,
       device const half* b,
       device const half* weight,
       device half* normed_output,
       device half* residual_output,  // also write residual for skip connection
       constant float& eps,
       constant uint& hidden_size,
       ...
   )
   ```
   - Compute `residual = a + b` in registers.
   - Compute RMS over the residual (parallel reduction in threadgroup).
   - Normalize and scale by weight.
   - Write both `normed_output` (for next matmul) and
     `residual_output` (for next skip connection).

2. **Fused SiLU+gate** — enhance `silu_gate` in `activation.metal`:
   - Currently `silu_gate` computes `silu(gate) * up` from two input
     buffers. This is already a fused kernel. The additional
     optimization is to fold this into the gate+up matmul output
     if using the custom matvec kernel. This requires the matvec
     kernel to support a fused activation function callback, which
     adds complexity. Consider whether the dispatch overhead savings
     justify this.

3. **RoPE in attention** — fold into `standard_attention`:
   - When loading Q in the attention kernel, apply RoPE rotation
     inline. This eliminates the separate RoPE kernel dispatch for Q.
   - K still needs separate RoPE before cache write (RoPE is applied
     before caching, not at attention time).
   - Net savings: 1 kernel dispatch per layer.

4. **Dispatch changes** in `inference.rs`:
   - Replace `residual_add` + `rms_norm` sequence with
     `fused_residual_rms_norm`.
   - Replace separate `rope(Q)` + `attention` with `attention` that
     applies RoPE internally (pass RoPE parameters to attention
     kernel).
   - Reduces per-layer dispatch count from ~18 to ~14.

**Acceptance criteria:**
- Fused kernels produce results within 1e-5 of the unfused sequence
  on random inputs.
- `cargo test -p ironmill-inference --features metal` passes.
- Per-layer kernel dispatch count decreases by ≥3.
- Benchmark: ≥1.2× tok/s improvement over post-flash-attention
  baseline.

---

### MLX-OPTIMIZE: MLX backend optimization

**Dependencies:** `MLX-TURBOQUANT`

**Goal:** Profile and optimize the MLX backend for maximum throughput.
This task is iterative and benchmark-driven.

**Files to modify:**
- `crates/ironmill-inference/src/mlx/inference.rs` — eval placement,
  async_eval, prefill optimization.

**Implementation details:**

1. **`eval()` placement optimization:**
   - Profile whether cross-layer lazy fusion (region B of layer N
     fusing with region A of layer N+1) actually fires in practice.
   - If per-layer `eval()` is too aggressive, explore batching: eval
     every 2-4 layers for the TurboQuant path (requires cache arrays
     for multiple layers to be written before any are read).
   - If single-eval-at-end for FP16 causes memory pressure on long
     sequences, add strategic mid-pipeline evals.

2. **`mlx_async_eval` for prefill:**
   - During prefill with chunking, use `mlx_async_eval` to overlap
     graph construction for chunk N+1 with GPU execution of chunk N.
   - Measure whether this improves prefill throughput.

3. **MLX built-in quantized matmul evaluation:**
   - Test whether MLX's native `mlx_quantized_matmul` (if available)
     is competitive with the ported PolarQuant kernels for
     weight-only quantization.
   - If competitive, switch to built-in and remove 4 custom kernel
     source strings.

4. **Memory profiling:**
   - Log peak MLX memory pool usage during generation.
   - Compare with MPS backend's pre-allocated memory.
   - If peak memory is excessive, add `mlx_metal_clear_cache()` calls
     at strategic points.

**Acceptance criteria:**
- Profile report comparing eval placement strategies (measurements, not
  just code changes).
- If `async_eval` improves prefill by ≥10%, ship it. Otherwise, leave
  synchronous.
- Final benchmark comparison: MLX vs. MPS on Qwen3-0.6B and
  LLaMA-3.2-1B for both FP16 and TurboQuant paths.

---

## Key Types Reference

These existing types are used across tasks. Agents should not redefine
them.

```rust
// crates/ironmill-inference/src/engine.rs
pub trait InferenceEngine {
    fn load(&mut self, artifacts: &dyn std::any::Any) -> Result<(), InferenceError>;
    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError>;
    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError>;
    fn reset(&mut self);
}

// crates/mil-rs/src/weights.rs
pub trait WeightProvider {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError>;
    fn tensor_names(&self) -> Vec<&str>;
    fn config(&self) -> &ModelConfig;
    fn has_tensor(&self, name: &str) -> bool;
}

pub struct ModelConfig {
    pub architecture: Architecture,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub extra: HashMap<String, serde_json::Value>,
}

// crates/ironmill-inference/src/gpu/config.rs
pub struct GpuConfig {
    pub max_seq_len: usize,
    pub attention_tile_size: Option<usize>,
    pub prefill_chunk_size: Option<usize>,
    pub enable_turboquant: bool,
    pub rotation_seed: u64,
    pub n_bits: u8,
    pub force_cpu_dequant: bool,
}

// crates/ironmill-inference/src/gpu/inference.rs
pub struct GpuArtifacts<'a> {
    pub weights: &'a dyn WeightProvider,
    pub config: GpuConfig,
}
```

## Metal Shader Files Reference

Existing shaders that tasks modify or port from:

| File | Kernels |
|------|---------|
| `normalization.metal` | `rms_norm` |
| `activation.metal` | `silu_gate` |
| `rope.metal` | `rope` |
| `elementwise.metal` | `residual_add`, `copy_buffer` |
| `embedding.metal` | `embedding_lookup` |
| `attention.metal` | `standard_attention` |
| `turboquant.metal` | `turboquant_cache_write`, `turboquant_attention`, `turboquant_outlier_cache_write`, `turboquant_outlier_attention` |
| `quantized_matmul.metal` | `polarquant_matvec_int4`, `polarquant_matmul_int4`, `polarquant_matvec_int8`, `polarquant_matmul_int8` |

## Build & Verification Commands

```bash
# Check workspace compiles
cargo check --workspace --all-features

# Check individual crates
cargo check -p ironmill-mlx-sys
cargo check -p ironmill-inference --features metal
cargo check -p ironmill-inference --features mlx
cargo check -p ironmill-bench --features metal

# Run tests
cargo test -p ironmill-mlx-sys
cargo test -p ironmill-inference --features metal
cargo test -p ironmill-inference --features mlx
cargo test -p ironmill-bench --features metal

# Benchmark (requires model weights)
cargo run -p ironmill-bench --features metal --release -- --backend metal
cargo run -p ironmill-bench --features mlx --release -- --backend mlx
```
