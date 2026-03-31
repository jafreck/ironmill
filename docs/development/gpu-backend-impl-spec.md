# Metal GPU Backend — Implementation Spec

Implementation spec derived from [gpu-backend.md](gpu-backend.md). Each task is
self-contained with file paths, type signatures, acceptance criteria, and
explicit dependencies on other tasks.

---

## Task 1 — `ironmill-metal-sys` crate scaffold

**Goal:** Create the FFI quarantine crate for Metal APIs, following the
`ironmill-ane-sys` / `ironmill-iosurface` pattern.

### Files to create

```
crates/ironmill-metal-sys/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── device.rs
│   ├── buffer.rs
│   ├── command.rs
│   ├── pipeline.rs
│   ├── shader.rs
│   ├── mps.rs
│   └── error.rs
```

### Cargo.toml

```toml
[package]
name = "ironmill-metal-sys"
version = "0.1.0"
edition = "2021"

[dependencies]
objc2 = "0.6"
objc2-foundation = { version = "0.3", features = ["NSError", "NSString", "NSArray"] }
objc2-metal = { version = "0.3", features = [
    "MTLDevice", "MTLBuffer", "MTLCommandQueue", "MTLCommandBuffer",
    "MTLComputeCommandEncoder", "MTLComputePipelineState", "MTLLibrary",
    "MTLFunction", "MTLResource",
] }
block2 = "0.6"
```

Add `"crates/ironmill-metal-sys"` to the workspace `members` list in the root
`Cargo.toml`.

### `lib.rs`

```rust
#[cfg(not(target_os = "macos"))]
compile_error!("ironmill-metal-sys only supports macOS");

mod buffer;
mod command;
mod device;
mod error;
mod mps;
mod pipeline;
mod shader;

pub use buffer::{MetalBuffer, StorageMode};
pub use command::{CommandQueue, CommandBuffer, ComputeEncoder};
pub use device::MetalDevice;
pub use error::MetalSysError;
pub use mps::MpsMatMul;
pub use pipeline::ComputePipeline;
pub use shader::ShaderLibrary;
```

### `error.rs`

```rust
#[derive(Debug, thiserror::Error)]
pub enum MetalSysError {
    #[error("no Metal device found")]
    NoDevice,
    #[error("buffer allocation failed: {0}")]
    BufferAllocFailed(String),
    #[error("shader compilation failed: {0}")]
    ShaderCompileFailed(String),
    #[error("pipeline creation failed: {0}")]
    PipelineCreationFailed(String),
    #[error("command buffer error: {0}")]
    CommandBufferError(String),
    #[error("MPS error: {0}")]
    MpsError(String),
    #[error("{0}")]
    Other(String),
}
```

### `device.rs`

```rust
pub struct MetalDevice { /* wraps objc2_metal::MTLDevice */ }

impl MetalDevice {
    /// Acquire the system default Metal device.
    pub fn system_default() -> Result<Self, MetalSysError>;

    /// Maximum threadgroup memory length (bytes). 32 KB on M1–M4.
    pub fn max_threadgroup_memory(&self) -> usize;

    /// Maximum threads per threadgroup.
    pub fn max_threads_per_threadgroup(&self) -> usize;

    /// Raw device pointer for downstream use (MPS, pipeline creation).
    pub(crate) fn raw(&self) -> /* &ProtocolObject<dyn MTLDevice> */;
}
```

### `buffer.rs`

```rust
#[derive(Debug, Clone, Copy)]
pub enum StorageMode { Shared, Private }

pub struct MetalBuffer { /* wraps retained MTLBuffer */ }

impl MetalBuffer {
    /// Allocate an empty buffer.
    pub fn new(device: &MetalDevice, size: usize, mode: StorageMode)
        -> Result<Self, MetalSysError>;

    /// Allocate and fill from CPU bytes. Mode must be Shared.
    pub fn from_bytes(device: &MetalDevice, data: &[u8], mode: StorageMode)
        -> Result<Self, MetalSysError>;

    pub fn length(&self) -> usize;
    pub fn storage_mode(&self) -> StorageMode;

    /// Read bytes from a Shared buffer into `dst`.
    pub fn read_bytes(&self, dst: &mut [u8]) -> Result<(), MetalSysError>;

    /// Write bytes into a Shared buffer from `src`.
    pub fn write_bytes(&self, src: &[u8]) -> Result<(), MetalSysError>;

    /// Raw pointer for encoder binding.
    pub(crate) fn raw(&self) -> /* &ProtocolObject<dyn MTLBuffer> */;
}
```

### `command.rs`

```rust
pub struct CommandQueue { /* retained MTLCommandQueue */ }

impl CommandQueue {
    pub fn new(device: &MetalDevice) -> Result<Self, MetalSysError>;
    pub fn command_buffer(&self) -> Result<CommandBuffer, MetalSysError>;
}

pub struct CommandBuffer { /* retained MTLCommandBuffer */ }

impl CommandBuffer {
    pub fn compute_encoder(&self) -> Result<ComputeEncoder, MetalSysError>;
    pub fn commit(&self);
    pub fn wait_until_completed(&self);
    pub fn status(&self) -> CommandBufferStatus;
    pub fn error(&self) -> Option<String>;
}

pub struct ComputeEncoder { /* retained MTLComputeCommandEncoder */ }

impl ComputeEncoder {
    pub fn set_pipeline(&self, pipeline: &ComputePipeline);
    pub fn set_buffer(&self, index: u32, buffer: &MetalBuffer, offset: u64);
    pub fn set_bytes(&self, index: u32, data: &[u8]);
    pub fn dispatch_threads(
        &self,
        grid_size: [u32; 3],
        threadgroup_size: [u32; 3],
    );
    pub fn dispatch_threadgroups(
        &self,
        threadgroup_count: [u32; 3],
        threads_per_threadgroup: [u32; 3],
    );
    pub fn end_encoding(&self);
}
```

### `pipeline.rs`

```rust
pub struct ComputePipeline { /* retained MTLComputePipelineState */ }

impl ComputePipeline {
    pub fn new(
        device: &MetalDevice,
        library: &ShaderLibrary,
        function_name: &str,
    ) -> Result<Self, MetalSysError>;

    pub fn max_total_threads_per_threadgroup(&self) -> usize;
    pub fn thread_execution_width(&self) -> usize;
}
```

### `shader.rs`

```rust
pub struct ShaderLibrary { /* retained MTLLibrary */ }

impl ShaderLibrary {
    /// Compile Metal shader source at runtime.
    pub fn from_source(device: &MetalDevice, source: &str)
        -> Result<Self, MetalSysError>;

    /// List available function names.
    pub fn function_names(&self) -> Vec<String>;
}
```

### `mps.rs`

```rust
pub struct MpsMatMul { /* retained MPSMatrixMultiplication */ }

impl MpsMatMul {
    /// Create an MPS FP16 matmul: C = alpha * A @ B + beta * C.
    pub fn new(
        device: &MetalDevice,
        transpose_left: bool,
        transpose_right: bool,
        result_rows: usize,
        result_columns: usize,
        interior_columns: usize,
    ) -> Result<Self, MetalSysError>;

    /// Encode the matmul into a command buffer.
    /// `a`, `b`, `c` are MetalBuffers in row-major FP16 layout.
    pub fn encode(
        &self,
        command_buffer: &CommandBuffer,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
    ) -> Result<(), MetalSysError>;
}
```

### Acceptance criteria

- `cargo check -p ironmill-metal-sys` succeeds on macOS.
- `cargo check -p ironmill-metal-sys --target x86_64-unknown-linux-gnu` fails
  with `compile_error!`.
- Unit tests: create device, allocate shared buffer, round-trip bytes.
- No `unsafe` escapes beyond `src/` — all public API is safe Rust.

### Dependencies

None — this is a leaf crate.

---

## Task 2 — Metal shader sources

**Goal:** Write the `.metal` compute shader source files that implement all
non-MPS ops in the decode pipeline.

### Files to create

```
crates/ironmill-inference/src/gpu/shaders/
├── embedding.metal
├── normalization.metal       # rms_norm
├── rope.metal                # rotary position embedding
├── activation.metal          # silu, gelu, residual add
├── elementwise.metal         # generic mul, add (if needed)
├── attention.metal           # fused TQ attention kernel
└── turboquant.metal          # fused TQ cache-write kernel
```

All shaders are embedded at compile time via `include_str!()`.

### Kernel signatures (Metal Shading Language)

#### `normalization.metal`

```metal
kernel void rms_norm(
    device const half*  input      [[buffer(0)]],   // [token_count × hidden_size]
    device half*        output     [[buffer(1)]],   // [token_count × hidden_size]
    device const half*  weight     [[buffer(2)]],   // [hidden_size]
    constant uint&      hidden_size [[buffer(3)]],
    constant uint&      token_count [[buffer(4)]],
    constant float&     eps        [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
);
```

Dispatch: one threadgroup per token, `hidden_size` threads per threadgroup.

#### `rope.metal`

```metal
kernel void rope_apply(
    device half*        qk         [[buffer(0)]],   // [token_count × dim]
    constant uint&      head_dim   [[buffer(1)]],
    constant uint&      num_heads  [[buffer(2)]],
    constant uint&      seq_offset [[buffer(3)]],   // position of first token
    constant float&     theta      [[buffer(4)]],
    constant uint&      token_count [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
);
```

Dispatch: total threads = `token_count × num_heads × (head_dim / 2)`.

#### `activation.metal`

```metal
// SiLU(gate) * up — fused gate activation
kernel void silu_gate_mul(
    device const half*  gate       [[buffer(0)]],   // [token_count × intermediate_size]
    device const half*  up         [[buffer(1)]],   // [token_count × intermediate_size]
    device half*        output     [[buffer(2)]],   // [token_count × intermediate_size]
    constant uint&      size       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
);

// Residual addition: output = a + b
kernel void residual_add(
    device const half*  a          [[buffer(0)]],
    device const half*  b          [[buffer(1)]],
    device half*        output     [[buffer(2)]],
    constant uint&      size       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
);
```

#### `turboquant.metal`

```metal
// Fused cache-write: rotate → quantize → INT8 write
kernel void tq_cache_write(
    device const half*   kv_proj        [[buffer(0)]],   // [num_kv_heads × head_dim] FP16
    device const half*   rotation       [[buffer(1)]],   // [head_dim × head_dim] FP16
    device char*         cache          [[buffer(2)]],   // [num_kv_heads × max_seq × head_dim] INT8
    constant uint&       head_dim       [[buffer(3)]],
    constant uint&       num_kv_heads   [[buffer(4)]],
    constant uint&       max_seq_len    [[buffer(5)]],
    constant uint&       seq_pos        [[buffer(6)]],
    constant float&      inv_scale      [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]    // (head_dim_idx, kv_head_idx)
);
```

Dispatch: grid `[head_dim, num_kv_heads]`, threadgroup `[head_dim, 1]`.

#### `attention.metal`

```metal
// Fused TQ attention: Q rotate → dequant K → QK scores → softmax → dequant V → weighted sum → un-rotate
kernel void tq_attention(
    device const half*   q             [[buffer(0)]],   // [num_heads × head_dim] FP16
    device const char*   k_cache       [[buffer(1)]],   // [num_kv_heads × max_seq × head_dim] INT8
    device const char*   v_cache       [[buffer(2)]],   // [num_kv_heads × max_seq × head_dim] INT8
    device const half*   rotation      [[buffer(3)]],   // [head_dim × head_dim] FP16
    device half*         output        [[buffer(4)]],   // [num_heads × head_dim] FP16
    constant uint&       head_dim      [[buffer(5)]],
    constant uint&       num_heads     [[buffer(6)]],
    constant uint&       num_kv_heads  [[buffer(7)]],
    constant uint&       max_seq_len   [[buffer(8)]],
    constant uint&       seq_len       [[buffer(9)]],   // current sequence length
    constant float&      deq_scale     [[buffer(10)]],
    constant uint&       token_count   [[buffer(11)]],  // 1 for decode, >1 for prefill
    uint2 tid [[thread_position_in_grid]]    // (thread_in_head, head_idx)
);
```

Dispatch: one threadgroup per attention head.

Threadgroup memory budget (32 KB):
- Score accumulator: `seq_len × sizeof(float)` — tiled in chunks of
  `min(seq_len, 32768 / (2 × head_dim))` positions per tile.
- Partial output: `head_dim × sizeof(float)`.

The kernel tiles over K positions in the sequence dimension:
1. For each tile of K positions: compute QK dot products, store to
   threadgroup `scores[]`.
2. After all tiles: two-pass softmax (max-subtract, exp-sum, normalize).
3. For each tile of V positions: accumulate `score[p] × V[p]` into partial
   output.
4. Un-rotate final output and write to device memory.

#### `embedding.metal`

```metal
kernel void embedding_lookup(
    device const half*   table      [[buffer(0)]],   // [vocab_size × hidden_size]
    device half*         output     [[buffer(1)]],   // [token_count × hidden_size]
    device const uint*   token_ids  [[buffer(2)]],   // [token_count]
    constant uint&       hidden_size [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]    // (hidden_dim_idx, token_idx)
);
```

### Acceptance criteria

- Each shader compiles without errors via `ShaderLibrary::from_source()`.
- Kernel function names match what `ops.rs` (Task 4) expects.
- All buffer indices are consistent between `.metal` signatures and Rust
  encoder binding code.
- Prefill kernels work with `token_count > 1` (same kernels, parameterized).

### Dependencies

Task 1 (for `ShaderLibrary::from_source` to validate compilation).

---

## Task 3 — GPU error types and config

**Goal:** Define `GpuError` and `GpuConfig` in the safe inference layer,
following the `AneError` / `AneConfig` pattern.

### Files to create

```
crates/ironmill-inference/src/gpu/
├── mod.rs
├── error.rs
└── config.rs
```

### `error.rs`

```rust
use ironmill_metal_sys::MetalSysError;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("Metal FFI error: {0}")]
    Metal(#[from] MetalSysError),
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),
    #[error("weight loading failed: {0}")]
    WeightLoading(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl From<GpuError> for crate::InferenceError {
    fn from(e: GpuError) -> Self {
        InferenceError::Runtime(e.to_string())
    }
}
```

### `config.rs`

```rust
pub struct GpuConfig {
    /// Maximum sequence length for KV cache allocation.
    pub max_seq_len: usize,
    /// Whether to use INT8 TurboQuant KV cache (vs FP16).
    pub enable_turboquant: bool,
    /// Attention tile size (sequence positions per tile). `None` = auto.
    pub attention_tile_size: Option<usize>,
    /// Prefill chunk size. `None` = full-prompt (no chunking).
    pub prefill_chunk_size: Option<usize>,
    /// Use shared storage mode for KV cache (enables CPU inspection).
    pub debug_cache: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            enable_turboquant: true,
            attention_tile_size: None,
            prefill_chunk_size: None,
            debug_cache: false,
        }
    }
}
```

### `mod.rs`

```rust
pub mod config;
pub mod error;
// Future tasks add:
// pub mod inference;
// pub mod ops;
// pub mod weights;
// pub mod turboquant;

pub use config::GpuConfig;
pub use error::GpuError;
```

Also add `pub mod gpu;` to `crates/ironmill-inference/src/lib.rs` (gated
behind `#[cfg(target_os = "macos")]`).

Add `ironmill-metal-sys` as a dependency of `ironmill-inference` in its
`Cargo.toml` (macOS-only via `[target.'cfg(target_os = "macos")'.dependencies]`).

### Acceptance criteria

- `cargo check -p ironmill-inference` succeeds.
- `GpuError` converts to `InferenceError` via `From`.
- `GpuConfig::default()` compiles and has sensible defaults.

### Dependencies

Task 1 (for `MetalSysError` in the `From` impl).

---

## Task 4 — Kernel dispatch helpers (`ops.rs`)

**Goal:** Safe Rust wrappers that compile shaders and encode kernel dispatches
into a `ComputeEncoder`. Each function handles buffer binding, threadgroup
sizing, and dispatch dimensions.

### File to create

```
crates/ironmill-inference/src/gpu/ops.rs
```

### Types

```rust
use ironmill_metal_sys::*;

/// All compiled pipelines for the decode/prefill pass.
pub struct GpuPipelines {
    pub rms_norm: ComputePipeline,
    pub rope: ComputePipeline,
    pub silu_gate_mul: ComputePipeline,
    pub residual_add: ComputePipeline,
    pub tq_cache_write: ComputePipeline,
    pub tq_attention: ComputePipeline,
    pub embedding: ComputePipeline,
}

impl GpuPipelines {
    /// Compile all shaders and create pipeline states.
    pub fn new(device: &MetalDevice) -> Result<Self, GpuError>;
}
```

### Dispatch functions

Each function takes a `&ComputeEncoder` plus the relevant buffers and
parameters, sets the pipeline, binds buffers, and dispatches:

```rust
pub fn encode_rms_norm(
    encoder: &ComputeEncoder,
    pipelines: &GpuPipelines,
    input: &MetalBuffer,
    output: &MetalBuffer,
    weight: &MetalBuffer,
    hidden_size: u32,
    token_count: u32,
    eps: f32,
);

pub fn encode_rope(
    encoder: &ComputeEncoder,
    pipelines: &GpuPipelines,
    qk: &MetalBuffer,
    head_dim: u32,
    num_heads: u32,
    seq_offset: u32,
    theta: f32,
    token_count: u32,
);

pub fn encode_silu_gate_mul(
    encoder: &ComputeEncoder,
    pipelines: &GpuPipelines,
    gate: &MetalBuffer,
    up: &MetalBuffer,
    output: &MetalBuffer,
    size: u32,
);

pub fn encode_residual_add(
    encoder: &ComputeEncoder,
    pipelines: &GpuPipelines,
    a: &MetalBuffer,
    b: &MetalBuffer,
    output: &MetalBuffer,
    size: u32,
);

pub fn encode_tq_cache_write(
    encoder: &ComputeEncoder,
    pipelines: &GpuPipelines,
    kv_proj: &MetalBuffer,
    rotation: &MetalBuffer,
    cache: &MetalBuffer,
    head_dim: u32,
    num_kv_heads: u32,
    max_seq_len: u32,
    seq_pos: u32,
    inv_scale: f32,
);

pub fn encode_tq_attention(
    encoder: &ComputeEncoder,
    pipelines: &GpuPipelines,
    q: &MetalBuffer,
    k_cache: &MetalBuffer,
    v_cache: &MetalBuffer,
    rotation: &MetalBuffer,
    output: &MetalBuffer,
    head_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    max_seq_len: u32,
    seq_len: u32,
    deq_scale: f32,
    token_count: u32,
);

pub fn encode_embedding(
    encoder: &ComputeEncoder,
    pipelines: &GpuPipelines,
    table: &MetalBuffer,
    output: &MetalBuffer,
    token_ids: &MetalBuffer,
    hidden_size: u32,
    token_count: u32,
);
```

### Threadgroup sizing rules

| Kernel | Grid size | Threadgroup size |
|---|---|---|
| `rms_norm` | `[hidden_size, token_count, 1]` | `[min(hidden_size, 256), 1, 1]` |
| `rope` | `[token_count × num_heads × head_dim/2, 1, 1]` | `[256, 1, 1]` |
| `silu_gate_mul` | `[size, 1, 1]` | `[256, 1, 1]` |
| `residual_add` | `[size, 1, 1]` | `[256, 1, 1]` |
| `tq_cache_write` | `[head_dim, num_kv_heads, 1]` | `[head_dim, 1, 1]` |
| `tq_attention` | `[head_dim, num_heads, 1]` | `[head_dim, 1, 1]` |
| `embedding` | `[hidden_size, token_count, 1]` | `[256, 1, 1]` |

### Acceptance criteria

- All `encode_*` functions compile.
- Threadgroup sizes do not exceed device max (queried from `MetalDevice`).
- No `unsafe` in this file — all safety is in `ironmill-metal-sys`.

### Dependencies

Task 1 (Metal FFI types), Task 2 (shader source for compilation).

---

## Task 5 — Weight loading (`weights.rs`)

**Goal:** Load model weights from SafeTensors or GGUF into `MetalBuffer`s,
organized per-layer for direct kernel binding.

### File to create

```
crates/ironmill-inference/src/gpu/weights.rs
```

### Types

```rust
use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};
use ironmill_compile::weights::{ModelConfig, WeightProvider};

/// Weights for a single transformer layer.
pub struct LayerWeights {
    pub q_proj: MetalBuffer,      // [num_heads × head_dim, hidden_size] FP16
    pub k_proj: MetalBuffer,      // [num_kv_heads × head_dim, hidden_size] FP16
    pub v_proj: MetalBuffer,      // [num_kv_heads × head_dim, hidden_size] FP16
    pub o_proj: MetalBuffer,      // [hidden_size, num_heads × head_dim] FP16
    pub gate_proj: MetalBuffer,   // [intermediate_size, hidden_size] FP16
    pub up_proj: MetalBuffer,     // [intermediate_size, hidden_size] FP16
    pub down_proj: MetalBuffer,   // [hidden_size, intermediate_size] FP16
    pub input_layernorm: MetalBuffer,     // [hidden_size] FP16
    pub post_attention_layernorm: MetalBuffer, // [hidden_size] FP16
}

/// All model weights loaded into Metal buffers.
pub struct GpuWeights {
    pub embed_tokens: MetalBuffer,   // [vocab_size × hidden_size] FP16
    pub layers: Vec<LayerWeights>,
    pub final_norm: MetalBuffer,     // [hidden_size] FP16
    pub lm_head: MetalBuffer,        // [vocab_size, hidden_size] FP16
    pub config: ModelConfig,
}

impl GpuWeights {
    /// Load all weights from a weight provider into Metal buffers.
    ///
    /// Strategy:
    /// 1. Call `provider.tensor(name)` to get CPU bytes.
    /// 2. Create shared `MetalBuffer` from bytes.
    /// 3. Blit to private storage for GPU read bandwidth.
    ///
    /// If `tie_word_embeddings` is true, `lm_head` aliases `embed_tokens`.
    pub fn load(
        device: &MetalDevice,
        queue: &CommandQueue,
        provider: &dyn WeightProvider,
    ) -> Result<Self, GpuError>;
}
```

### Weight name mapping

Standard HuggingFace naming convention:
```
model.embed_tokens.weight
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight
model.norm.weight
lm_head.weight
```

For GGUF, `GgufProvider` already normalizes names to this convention.

### Shared-to-private blit

After loading all weights into shared buffers, encode a single blit command
buffer that copies each shared buffer to a private-storage buffer, then
replace the shared buffer references. This is a one-time cost during `load`.

```rust
fn blit_to_private(
    device: &MetalDevice,
    queue: &CommandQueue,
    shared: &MetalBuffer,
) -> Result<MetalBuffer, GpuError>;
```

### Acceptance criteria

- Loads all weights for a Qwen3-0.6B model (24 layers) without error.
- Each `MetalBuffer` has the expected byte length:
  `rows × cols × sizeof(f16)`.
- Round-trip validation: read back one weight buffer and compare to source
  bytes (development/test only).

### Dependencies

Task 1 (MetalBuffer, MetalDevice, CommandQueue), Task 3 (GpuError).
Depends on `ironmill-compile` for `WeightProvider` and `ModelConfig`
(existing, no changes needed).

---

## Task 6 — GPU TurboQuant KV cache

**Goal:** INT8 KV cache backed by `MetalBuffer`s, with the same quantization
math as the ANE `KvCacheManager` but using GPU-native storage.

### Files to create

```
crates/ironmill-inference/src/gpu/turboquant/
├── mod.rs
└── cache.rs
```

### `cache.rs`

```rust
use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};

/// TurboQuant configuration (mirrors ANE TurboQuantConfig).
pub struct GpuTurboQuantConfig {
    pub max_seq_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub rotation_seed: u64,
}

/// GPU-backed INT8 KV cache with Hadamard rotation.
pub struct GpuKvCache {
    k_caches: Vec<MetalBuffer>,   // per-layer [num_kv_heads × max_seq × head_dim] INT8
    v_caches: Vec<MetalBuffer>,   // per-layer [num_kv_heads × max_seq × head_dim] INT8
    rotation: MetalBuffer,        // [head_dim × head_dim] FP16
    deq_scale: f32,
    inv_scale: f32,
    seq_pos: usize,
    config: GpuTurboQuantConfig,
}

impl GpuKvCache {
    /// Allocate all cache buffers and precompute rotation matrix.
    pub fn new(
        device: &MetalDevice,
        config: GpuTurboQuantConfig,
        debug: bool,  // shared vs private storage
    ) -> Result<Self, GpuError>;

    /// Current sequence position.
    pub fn seq_pos(&self) -> usize;

    /// Advance sequence position by `count` tokens.
    pub fn advance(&mut self, count: usize);

    /// Reset cache for new sequence.
    pub fn reset(&mut self);

    /// Get K cache buffer for a layer (for encoder binding).
    pub fn k_cache(&self, layer: usize) -> &MetalBuffer;

    /// Get V cache buffer for a layer.
    pub fn v_cache(&self, layer: usize) -> &MetalBuffer;

    /// Rotation matrix buffer.
    pub fn rotation(&self) -> &MetalBuffer;

    pub fn deq_scale(&self) -> f32;
    pub fn inv_scale(&self) -> f32;
    pub fn max_seq_len(&self) -> usize;
}
```

### `mod.rs`

```rust
pub mod cache;
pub use cache::{GpuKvCache, GpuTurboQuantConfig};
```

### Rotation matrix generation

Reuse the same Hadamard rotation generation as
`ironmill-inference/src/ane/turboquant/model.rs` — extract the rotation
matrix generation into a shared utility in
`ironmill-inference/src/turboquant_common.rs` (or duplicate for now and
refactor later). The matrix is a `[head_dim × head_dim]` FP16 Hadamard
matrix, seeded by `rotation_seed`.

### Quantization scale

Same beta-optimal scalar quantization as the ANE path:
- `deq_scale` = scale factor for INT8 → FP16 dequantization.
- `inv_scale` = `1.0 / deq_scale` for FP16 → INT8 quantization.
- These are precomputed from the Hadamard-rotated weight distribution.

### Acceptance criteria

- Allocate cache for 24-layer model with `max_seq_len=2048`,
  `num_kv_heads=8`, `head_dim=128`: total ~96 MB INT8.
- `seq_pos` advances correctly, `reset()` zeros the position counter.
- Rotation matrix matches ANE path output for same seed.

### Dependencies

Task 1 (MetalBuffer), Task 3 (GpuError).

---

## Task 7 — `GpuInference` engine

**Goal:** Implement the `InferenceEngine` trait for the Metal GPU backend.
This is the main decode loop — the GPU equivalent of `AneInference`.

### File to create

```
crates/ironmill-inference/src/gpu/inference.rs
```

### Type

```rust
use crate::{InferenceEngine, InferenceError, Logits};
use super::{GpuConfig, GpuError};
use super::ops::GpuPipelines;
use super::weights::GpuWeights;
use super::turboquant::GpuKvCache;
use ironmill_metal_sys::*;

pub struct GpuInference {
    device: MetalDevice,
    queue: CommandQueue,
    pipelines: GpuPipelines,
    weights: Option<GpuWeights>,
    kv_cache: Option<GpuKvCache>,
    config: GpuConfig,
    seq_pos: usize,

    // Reusable intermediate buffers (allocated once during load)
    hidden_buf: Option<MetalBuffer>,      // [hidden_size] FP16
    attn_out_buf: Option<MetalBuffer>,    // [num_heads × head_dim] FP16
    q_buf: Option<MetalBuffer>,           // [num_heads × head_dim] FP16
    k_buf: Option<MetalBuffer>,           // [num_kv_heads × head_dim] FP16
    v_buf: Option<MetalBuffer>,           // [num_kv_heads × head_dim] FP16
    gate_buf: Option<MetalBuffer>,        // [intermediate_size] FP16
    up_buf: Option<MetalBuffer>,          // [intermediate_size] FP16
    ffn_buf: Option<MetalBuffer>,         // [intermediate_size] FP16
    norm_buf: Option<MetalBuffer>,        // [hidden_size] FP16
    residual_buf: Option<MetalBuffer>,    // [hidden_size] FP16
    logits_buf: Option<MetalBuffer>,      // [vocab_size] FP16
}
```

### Construction

```rust
impl GpuInference {
    pub fn new(config: GpuConfig) -> Result<Self, GpuError> {
        let device = MetalDevice::system_default()?;
        let queue = CommandQueue::new(&device)?;
        let pipelines = GpuPipelines::new(&device)?;
        Ok(Self {
            device, queue, pipelines, config,
            weights: None, kv_cache: None, seq_pos: 0,
            hidden_buf: None, attn_out_buf: None,
            q_buf: None, k_buf: None, v_buf: None,
            gate_buf: None, up_buf: None, ffn_buf: None,
            norm_buf: None, residual_buf: None, logits_buf: None,
        })
    }
}
```

### `InferenceEngine` impl

```rust
impl InferenceEngine for GpuInference {
    fn load(&mut self, artifacts: &dyn std::any::Any) -> Result<(), InferenceError>;
    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError>;
    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError>;
    fn reset(&mut self);
}
```

#### `load`

The `artifacts` argument is downcast to a load request type:

```rust
/// Passed as `artifacts` to `GpuInference::load()`.
pub struct GpuLoadRequest<'a> {
    pub provider: &'a dyn WeightProvider,
}
```

Steps:
1. Downcast `artifacts` to `&GpuLoadRequest`.
2. `GpuWeights::load(device, queue, provider)` — loads all weights.
3. Allocate intermediate buffers based on `ModelConfig` dimensions.
4. Allocate `GpuKvCache` from config + model dimensions.
5. Store everything in `self`.

#### `decode_step`

Single-token forward pass. All ops encoded into one `CommandBuffer`:

```rust
fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError> {
    let w = self.weights.as_ref().ok_or(InferenceError::NotLoaded)?;
    let cache = self.kv_cache.as_mut().unwrap();
    let cfg = &w.config;

    let cb = self.queue.command_buffer()?;
    let enc = cb.compute_encoder()?;

    // 0. Embedding lookup (write token_id to a tiny shared buffer, dispatch)
    encode_embedding(&enc, &self.pipelines, &w.embed_tokens, hidden, token_ids_buf, ...);

    for i in 0..cfg.num_hidden_layers {
        let layer = &w.layers[i];

        // 1. RMSNorm (pre-attention)
        encode_rms_norm(&enc, ..., &layer.input_layernorm, ...);

        // 2. Q/K/V projections (MPS matmul)
        mps_q.encode(&cb, hidden, &layer.q_proj, q_buf)?;
        mps_k.encode(&cb, hidden, &layer.k_proj, k_buf)?;
        mps_v.encode(&cb, hidden, &layer.v_proj, v_buf)?;

        // 3. RoPE on Q and K
        encode_rope(&enc, ..., self.seq_pos, ...);

        // 4. Cache write (fused: rotate → quantize → INT8)
        encode_tq_cache_write(&enc, ..., cache.k_cache(i), ...);
        encode_tq_cache_write(&enc, ..., cache.v_cache(i), ...);

        // 5. Attention (fused: rotate Q → dequant K/V → scores → softmax → output)
        encode_tq_attention(&enc, ..., cache.k_cache(i), cache.v_cache(i), ...);

        // 6. Output projection (MPS matmul)
        mps_o.encode(&cb, attn_out, &layer.o_proj, hidden)?;

        // 7. Residual add
        encode_residual_add(&enc, ...);

        // 8. RMSNorm (post-attention)
        encode_rms_norm(&enc, ..., &layer.post_attention_layernorm, ...);

        // 9. Gate + up projections (MPS matmul ×2)
        mps_gate.encode(&cb, hidden, &layer.gate_proj, gate_buf)?;
        mps_up.encode(&cb, hidden, &layer.up_proj, up_buf)?;

        // 10. SiLU + gate mul
        encode_silu_gate_mul(&enc, ...);

        // 11. Down projection (MPS matmul)
        mps_down.encode(&cb, ffn_buf, &layer.down_proj, hidden)?;

        // 12. Residual add
        encode_residual_add(&enc, ...);
    }

    // 13. Final RMSNorm
    encode_rms_norm(&enc, ..., &w.final_norm, ...);

    // 14. lm_head (MPS matmul)
    mps_lm_head.encode(&cb, hidden, &w.lm_head, logits_buf)?;

    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    // Read logits from GPU
    cache.advance(1);
    self.seq_pos += 1;
    read_logits(logits_buf)
}
```

#### `prefill`

Same pipeline as `decode_step` but with `token_count = tokens.len()`:

1. All intermediate buffers are sized `[token_count × dim]` instead of
   `[1 × dim]`. Allocate temporary oversized buffers if needed, or
   re-allocate for each prefill call.
2. MPS matmuls operate on `[token_count × hidden]` inputs.
3. Element-wise kernels use `token_count` parameter.
4. Attention kernel applies causal mask.
5. Cache-write writes all `token_count` positions at once.
6. lm_head only needs the last token's hidden state. Slice or parameterize
   to avoid computing `vocab_size` logits for all positions.
7. If `config.prefill_chunk_size` is set and `tokens.len()` exceeds it,
   loop over chunks, each writing its KV positions before the next.

#### `reset`

```rust
fn reset(&mut self) {
    self.seq_pos = 0;
    if let Some(cache) = &mut self.kv_cache {
        cache.reset();
    }
}
```

### MPS matmul pre-creation

During `load`, pre-create `MpsMatMul` objects for each matmul shape. Shapes
are fixed per model, so they are created once:

```rust
struct MpsMatMuls {
    q_proj: MpsMatMul,     // [1, num_heads×head_dim] = [1, hidden] @ [hidden, num_heads×head_dim]
    k_proj: MpsMatMul,
    v_proj: MpsMatMul,
    o_proj: MpsMatMul,
    gate_proj: MpsMatMul,
    up_proj: MpsMatMul,
    down_proj: MpsMatMul,
    lm_head: MpsMatMul,
}
```

Note: for prefill, matmul dimensions change (`token_count` rows). Either
create separate `MpsMatMul` objects for prefill batch sizes, or re-create
per prefill call if MPS creation is cheap.

### Acceptance criteria

- `GpuInference::new()` succeeds on macOS with Metal GPU.
- `load()` accepts a `GpuLoadRequest` with a `SafeTensorsProvider`.
- `decode_step()` produces logits with `vocab_size` elements.
- `prefill()` processes multiple tokens and populates KV cache.
- `reset()` allows a fresh conversation.
- Total per-token latency: single `CommandBuffer` commit+wait.

### Dependencies

Task 1 (all Metal FFI), Task 2 (shaders compiled into pipelines),
Task 3 (GpuError, GpuConfig), Task 4 (encode_* helpers),
Task 5 (GpuWeights), Task 6 (GpuKvCache).

---

## Task 8 — Benchmark integration (`ironmill-bench`)

**Goal:** Add `--backend metal` to `ironmill-bench` so it uses `GpuInference`.

### Files to modify

```
crates/ironmill-bench/Cargo.toml          # add ironmill-metal-sys dep (macOS)
crates/ironmill-bench/src/main.rs         # add "metal" backend arm
crates/ironmill-bench/src/inference.rs    # add run_metal_inference function
```

### Changes

#### `main.rs` — backend dispatch

Add `"metal"` to the match on backend strings:

```rust
"metal" => {
    inference::run_metal_inference(&model_path, &opt, settings.iterations, settings.warmup)?;
}
```

#### `inference.rs` — new function

```rust
pub fn run_metal_inference(
    model_path: &Path,
    opt: &OptConfig,
    iterations: usize,
    warmup: usize,
) -> anyhow::Result<()> {
    use ironmill_compile::weights::{SafeTensorsProvider, GgufProvider};
    use ironmill_inference::gpu::{GpuConfig, GpuInference, GpuLoadRequest};
    use ironmill_inference::InferenceEngine;

    // 1. Open weight provider (SafeTensors or GGUF based on extension).
    let provider: Box<dyn WeightProvider> = if model_path.extension() == Some("gguf") {
        Box::new(GgufProvider::load(model_path)?)
    } else {
        Box::new(SafeTensorsProvider::load(model_path)?)
    };

    // 2. Build GpuConfig from OptConfig.
    let config = GpuConfig {
        max_seq_len: opt.max_seq_len.unwrap_or(2048),
        enable_turboquant: opt.kv_quant != KvQuantMode::None,
        ..Default::default()
    };

    // 3. Create engine and load.
    let mut engine = GpuInference::new(config)?;
    engine.load(&GpuLoadRequest { provider: provider.as_ref() })?;

    // 4. Run warmup + timed iterations using existing bench harness.
    // (Same pattern as run_ane_direct_inference)
    let prompt_tokens = vec![1u32; 4];  // minimal prompt
    for _ in 0..warmup {
        engine.prefill(&prompt_tokens)?;
        engine.decode_step(1)?;
        engine.reset();
    }
    // ... timed loop, print results ...
    Ok(())
}
```

### CLI help update

Update the `--backend` help text to list `metal` as an option:

```
--backend <BACKEND>  Inference backend: ane (default), gpu, cpu, metal
```

### Acceptance criteria

- `ironmill-bench --backend metal --model <path>` runs inference.
- Produces tok/s metrics comparable to existing backends.
- Existing `ane`/`gpu`/`cpu` backends are unaffected.

### Dependencies

Task 7 (GpuInference).

---

## Task 9 — Tests

**Goal:** Unit and integration tests for the GPU backend.

### Test files to create

```
crates/ironmill-metal-sys/tests/device_test.rs     # basic Metal device tests
crates/ironmill-inference/tests/gpu_inference.rs    # end-to-end GPU inference test
```

### `device_test.rs` — Metal FFI smoke tests

```rust
#[test]
fn test_system_default_device() {
    let device = MetalDevice::system_default().unwrap();
    assert!(device.max_threadgroup_memory() >= 32768);
}

#[test]
fn test_buffer_roundtrip() {
    let device = MetalDevice::system_default().unwrap();
    let data = vec![1u8; 1024];
    let buf = MetalBuffer::from_bytes(&device, &data, StorageMode::Shared).unwrap();
    let mut readback = vec![0u8; 1024];
    buf.read_bytes(&mut readback).unwrap();
    assert_eq!(data, readback);
}

#[test]
fn test_shader_compilation() {
    let device = MetalDevice::system_default().unwrap();
    let source = include_str!("../../ironmill-inference/src/gpu/shaders/normalization.metal");
    let lib = ShaderLibrary::from_source(&device, source).unwrap();
    assert!(lib.function_names().contains(&"rms_norm".to_string()));
}
```

### `gpu_inference.rs` — integration test

```rust
/// Requires a model file at the path specified by IRONMILL_TEST_MODEL env var.
#[test]
#[ignore]  // only run with: cargo test -- --ignored
fn test_gpu_decode_produces_logits() {
    let model_path = std::env::var("IRONMILL_TEST_MODEL")
        .expect("set IRONMILL_TEST_MODEL to a SafeTensors model directory");
    let provider = SafeTensorsProvider::load(Path::new(&model_path)).unwrap();
    let config = GpuConfig::default();
    let mut engine = GpuInference::new(config).unwrap();
    engine.load(&GpuLoadRequest { provider: &provider }).unwrap();

    let logits = engine.prefill(&[1, 2, 3]).unwrap();
    assert_eq!(logits.len(), provider.config().vocab_size);

    let logits = engine.decode_step(4).unwrap();
    assert_eq!(logits.len(), provider.config().vocab_size);

    engine.reset();
    // Can run again after reset.
    let logits = engine.prefill(&[1]).unwrap();
    assert_eq!(logits.len(), provider.config().vocab_size);
}
```

### Acceptance criteria

- `cargo test -p ironmill-metal-sys` passes (3 smoke tests).
- `cargo test -p ironmill-inference -- gpu` passes for non-ignored tests.
- `IRONMILL_TEST_MODEL=<path> cargo test -- --ignored gpu_decode`
  passes with a real model.

### Dependencies

All prior tasks.

---

## Dependency graph

```
Task 1 (metal-sys)
├── Task 2 (shaders) ──────────────┐
├── Task 3 (error/config) ─────────┤
│   ├── Task 4 (ops) ◄─ Task 2 ───┤
│   ├── Task 5 (weights) ──────────┤
│   └── Task 6 (kv cache) ─────────┤
│                                   ▼
│                         Task 7 (GpuInference)
│                                   │
│                                   ▼
│                         Task 8 (bench integration)
│                                   │
│                                   ▼
└──────────────────────── Task 9 (tests)
```

**Parallelizable:** Tasks 2, 3 can run in parallel after Task 1. Tasks 4, 5,
6 can run in parallel after Tasks 1+3. Task 7 requires 4+5+6. Tasks 8 and 9
follow Task 7.
