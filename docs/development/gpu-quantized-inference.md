# GPU Quantized Weight Inference — Implementation Spec

Implementation spec for PolarQuant and palettized weight support in the Metal
GPU backend. Each task is self-contained with file paths, type signatures,
acceptance criteria, and explicit dependencies.

Derived from investigation of the compiler passes, GPU weight loading path,
and TurboQuant INT4 implementation (which proves the custom-kernel pattern).

> **Prerequisites**: [PolarQuant Weight Quantization](../archive/polarquant-weight-quantization.md)
> (algorithm, compiler pass, ANE constraints),
> [Metal GPU Backend](../design/gpu-backend.md)

---

## Background

The Metal GPU backend loads all weights as dense FP16 via `GpuWeights::load()`,
which queries a `WeightProvider` by hardcoded HF-style names (e.g.,
`model.layers.0.self_attn.q_proj.weight`) and copies raw bytes into
`MTLBuffer`s. MPS matmul is hardcoded to `MPSDataTypeFloat16`.

The compiler's `PolarQuantPass` already produces quantized MIL IR
(`constexpr_lut_to_dense` + row norms `const` + `mul`), but the GPU backend
cannot consume it. The ONNX→MIL import sanitizes tensor names (dots → underscores),
so a name mapping layer is needed.

GPU TurboQuant already implements packed INT4 nibble storage in Metal shaders
(`turboquant.metal`), proving the custom-kernel pattern works end-to-end.

---

## Phase 1 — Load-time dequant (correctness baseline)

### Task 1 — Extend `WeightTensor` with quantization metadata

**Goal:** Add a `QuantizationInfo` enum so weight providers can communicate
quantized storage format without forcing immediate dequantization.

**File:** `crates/ironmill-compile/src/weights/mod.rs`

#### Types to add

```rust
/// Describes how a weight tensor is stored in compressed form.
#[derive(Debug, Clone)]
pub enum QuantizationInfo {
    /// Dense storage, no quantization. Existing behavior.
    None,
    /// PolarQuant / palettization: LUT + packed indices + row norms.
    /// Produced by `constexpr_lut_to_dense` in MIL IR.
    LutToDense {
        lut: Vec<u8>,               // [2^n_bits] reconstruction levels, raw bytes
        lut_dtype: ScalarType,       // Float16 or Float32
        indices: Vec<u8>,            // packed n-bit indices
        original_shape: Vec<usize>,  // shape before packing
        n_bits: u8,
        row_norms: Vec<u8>,          // [rows, 1] norms, raw bytes
        norms_dtype: ScalarType,
    },
    /// INT8 affine quantization: (quantized - zero_point) * scale.
    /// Produced by `constexpr_affine_dequantize` in MIL IR.
    AffineDequantize {
        scale: Vec<u8>,              // per-tensor or per-channel, raw bytes
        zero_point: Vec<u8>,
        scale_dtype: ScalarType,
        axis: Option<usize>,
    },
}
```

#### Changes to `WeightTensor`

```rust
pub struct WeightTensor<'a> {
    pub data: Cow<'a, [u8]>,
    pub shape: Vec<usize>,
    pub dtype: ScalarType,
    pub quant_info: QuantizationInfo,  // NEW
}
```

Update `WeightTensor::borrowed()` and `WeightTensor::owned()` to default
`quant_info` to `QuantizationInfo::None`.

#### Acceptance criteria

- `WeightTensor` has `quant_info` field, defaults to `None`.
- Existing `SafeTensorsProvider` and `GgufProvider` compile and pass tests
  without changes (they never set `quant_info`, so it stays `None`).
- `cargo check --workspace --all-features` passes.

**Dependencies:** none

---

### Task 2 — `MilWeightProvider`

**Goal:** A `WeightProvider` implementation that wraps a compiled
`mil_rs::ir::Program` and exposes quantized tensors by HF-style names.

**File:** new `crates/ironmill-compile/src/weights/mil_provider.rs`

#### Name mapping

Every `const` op carries an `onnx_name` attribute with the original ONNX
tensor name (set during import in `initializer_to_const()`). This attribute
survives both MIL name sanitization and PolarQuantPass mutation.

`MilWeightProvider` reads `onnx_name` directly — no reverse-sanitization:

```rust
pub struct MilWeightProvider {
    config: ModelConfig,
    /// HF name → extracted tensor data
    tensors: HashMap<String, ExtractedTensor>,
}

struct ExtractedTensor {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: ScalarType,
    quant_info: QuantizationInfo,
}
```

Construction walks the MIL Program's main function block:

1. For each `constexpr_lut_to_dense` op:
   - Extract `lut`, `indices`, `shape`, `polar_quant_seed` from op attributes
   - Read `onnx_name` attribute to get the HF tensor name
   - Find the associated `{op_name}_polar_norms` const op (row norms,
     output named `{original_output}_polar_norms`)
   - Store as `ExtractedTensor` with `QuantizationInfo::LutToDense`

2. For each `constexpr_affine_dequantize` op:
   - Extract `quantized_data`, `scale`, `zero_point`, `axis`
   - Read `onnx_name` attribute for HF tensor name
   - Store with `QuantizationInfo::AffineDequantize`

3. For each plain `const` op (not norms, not quantized):
   - Extract raw tensor data
   - Store with `QuantizationInfo::None`

#### `WeightProvider` impl

```rust
impl WeightProvider for MilWeightProvider {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError> {
        let t = self.tensors.get(name)
            .ok_or_else(|| MilError::TensorNotFound(name.to_string()))?;
        Ok(WeightTensor {
            data: Cow::Borrowed(&t.data),
            shape: t.shape.clone(),
            dtype: t.dtype,
            quant_info: t.quant_info.clone(),
        })
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    fn config(&self) -> &ModelConfig { &self.config }
}
```

#### Acceptance criteria

- `MilWeightProvider::new(program, config)` succeeds for a PolarQuant-compiled
  Program from a small ONNX model.
- `tensor_names()` returns HF-style names.
- `tensor("model.layers.0.self_attn.q_proj.weight")` returns a `WeightTensor`
  with `QuantizationInfo::LutToDense { n_bits: 4, .. }`.
- Unit test: round-trip a single weight through PolarQuantPass → extract →
  dequant → compare to original FP16 within tolerance.

**Dependencies:** Task 1

---

### Task 3 — CPU dequant functions

**Goal:** Implement dequantization from `QuantizationInfo` variants to FP16
byte arrays, for use in `GpuWeights::load()`.

**File:** new `crates/ironmill-inference/src/gpu/dequant.rs`

```rust
use ironmill_compile::weights::QuantizationInfo;

/// Dequantize a LUT-encoded tensor to FP16 bytes.
/// Unpacks n-bit indices, applies LUT lookup, multiplies by row norms.
pub fn dequant_lut_to_dense(
    indices: &[u8],
    lut: &[u8],
    lut_dtype: ScalarType,
    original_shape: &[usize],
    n_bits: u8,
    row_norms: &[u8],
    norms_dtype: ScalarType,
) -> Vec<u8> {
    // 1. Unpack n-bit indices from byte array
    // 2. For each index, look up the reconstruction level in lut
    // 3. Reshape to [rows, cols]
    // 4. Multiply each row by its norm
    // 5. Return as FP16 bytes
}

/// Dequantize an affine-quantized tensor to FP16 bytes.
pub fn dequant_affine(
    quantized_data: &[u8],
    scale: &[u8],
    zero_point: &[u8],
    scale_dtype: ScalarType,
    axis: Option<usize>,
    shape: &[usize],
) -> Vec<u8> {
    // (quantized - zero_point) * scale → FP16
}
```

#### Index unpacking detail

PolarQuantPass packs indices with `n_bits` per value into `u8` bytes,
LSB-first. For 4-bit: 2 values per byte — `lo | (hi << 4)` where the low
nibble holds the first value and the high nibble holds the second. This
matches GPU TurboQuant INT4 packing in `turboquant.metal`.

#### Acceptance criteria

- `dequant_lut_to_dense` correctly reconstructs FP16 weights from a
  PolarQuant-compressed tensor. Verify against the original FP16 within
  quantization tolerance (not bitwise exact — quantization is lossy).
- `dequant_affine` correctly reconstructs from INT8 affine representation.
- Unit tests with known inputs/outputs.

**Dependencies:** Task 1

---

### Task 4 — `GpuWeights::load()` quantization-aware path

**Goal:** When `WeightProvider` returns a tensor with non-`None`
`QuantizationInfo`, dequantize to FP16 on the CPU before creating the
Metal buffer.

**File:** `crates/ironmill-inference/src/gpu/weights.rs`

#### Changes

The existing `load_weight_buffer` helper:

```rust
fn load_weight_buffer(
    device: &MetalDevice,
    provider: &dyn WeightProvider,
    name: &str,
) -> Result<MetalBuffer, GpuError> {
    let tensor = provider.tensor(name)?;
    device.create_buffer_with_data(&tensor.data, StorageMode::Shared)
}
```

Becomes:

```rust
fn load_weight_buffer(
    device: &MetalDevice,
    provider: &dyn WeightProvider,
    name: &str,
) -> Result<MetalBuffer, GpuError> {
    let tensor = provider.tensor(name)?;
    let data = match &tensor.quant_info {
        QuantizationInfo::None => tensor.data.into_owned(),
        QuantizationInfo::LutToDense { lut, lut_dtype, indices,
            original_shape, n_bits, row_norms, norms_dtype } => {
            dequant_lut_to_dense(indices, lut, *lut_dtype,
                original_shape, *n_bits, row_norms, *norms_dtype)
        }
        QuantizationInfo::AffineDequantize { scale, zero_point,
            scale_dtype, axis } => {
            dequant_affine(&tensor.data, scale, zero_point,
                *scale_dtype, *axis, &tensor.shape)
        }
    };
    device.create_buffer_with_data(&data, StorageMode::Shared)
}
```

#### Acceptance criteria

- `GpuWeights::load(device, &mil_provider)` succeeds when `mil_provider`
  contains PolarQuant-compressed tensors.
- All Metal buffers contain valid FP16 data.
- Existing path (SafeTensors/GGUF with `QuantizationInfo::None`) unchanged.
- Tied embeddings (`config.tie_word_embeddings`) still work — `lm_head`
  loads from `model.embed_tokens.weight` when tied.

**Dependencies:** Task 2, Task 3

---

### Task 5 — Compile-for-GPU entry point

**Goal:** Provide a way to run PolarQuant passes on a model and get a
`MilWeightProvider` suitable for `GpuWeights::load()`.

**File:** new `crates/ironmill-compile/src/gpu/mod.rs`

This is deliberately minimal for Phase 1 — no bundle serialization yet.

```rust
use crate::weights::mil_provider::MilWeightProvider;
use mil_rs::ir::{Program, PassPipeline};

pub struct GpuCompileBuilder {
    input: PathBuf,
    n_bits: u8,
    min_elements: usize,
}

impl GpuCompileBuilder {
    pub fn new(input: impl Into<PathBuf>) -> Self;
    pub fn polar_quantize(mut self, n_bits: u8) -> Self;
    pub fn min_elements(mut self, min: usize) -> Self;

    /// Run import + PolarQuant passes, return a provider for GpuWeights.
    pub fn build(self) -> Result<MilWeightProvider, CompileError> {
        // 1. detect_format(input) → ONNX/SafeTensors/GGUF/MLPackage
        // 2. Import to Program (reuse CompileBuilder's import logic)
        // 3. Run PassPipeline::new().with_polar_quant(n_bits, min_elements)
        //    (includes PolarQuantPass + PolarRotationFusionPass)
        // 4. Wrap in MilWeightProvider
    }
}
```

#### Acceptance criteria

- `GpuCompileBuilder::new("model.onnx").polar_quantize(4).build()?`
  returns a `MilWeightProvider` with quantized tensors.
- Provider can be passed to `GpuWeights::load()` successfully.

**Dependencies:** Task 2

---

### Task 6 — E2E test

**Goal:** Validate the full pipeline: ONNX model → PolarQuant compile →
MilWeightProvider → GpuWeights::load → GPU inference → compare to FP16.

**File:** `crates/ironmill-inference/tests/gpu_polarquant_e2e.rs` or
extend existing GPU test module.

#### Test plan

1. Load a small model (SqueezNet or equivalent fixture)
2. Run FP16 GPU inference → record output logits
3. Run PolarQuant-4 compile → MilWeightProvider → GPU inference → record logits
4. Assert logits are close (tolerance: quantization noise, not bitwise)
5. Check that the loaded `MilWeightProvider` has `LutToDense` quant info
   for linear weight tensors and `None` for small tensors (below threshold)

#### Acceptance criteria

- Test passes with FP16-vs-PolarQuant output within reasonable tolerance.
- Confirms the full data path: compiler pass → provider → dequant → Metal buffer → MPS matmul → correct output.

**Dependencies:** Task 4, Task 5

---

## Phase 2 — GPU bundle format

### Task 7 — Define `.ironml-gpu` bundle format

**Goal:** A serialization format for compiled GPU models so they load
without recompilation.

**Files:** new `crates/ironmill-compile/src/gpu/bundle.rs`,
new `crates/ironmill-inference/src/gpu/bundle.rs`

#### Bundle layout

```
model.ironml-gpu/
├── manifest.json
└── weights/
    ├── layer_0_q_proj.bin     # packed indices
    ├── layer_0_q_proj.lut     # LUT values (raw bytes)
    ├── layer_0_q_proj.nrm     # row norms (raw bytes)
    ├── ...
    └── embed_tokens.bin       # dense FP16 (below min_elements)
```

#### `manifest.json` schema

```json
{
  "format_version": 1,
  "model_config": { },
  "quantization": {
    "method": "polarquant",
    "n_bits": 4,
    "seed": 42,
    "min_elements": 1024
  },
  "tensors": {
    "model.layers.0.self_attn.q_proj.weight": {
      "format": "lut_to_dense",
      "indices_file": "weights/layer_0_q_proj.bin",
      "lut_file": "weights/layer_0_q_proj.lut",
      "norms_file": "weights/layer_0_q_proj.nrm",
      "shape": [2048, 2048],
      "n_bits": 4,
      "dtype": "float16"
    },
    "model.embed_tokens.weight": {
      "format": "dense",
      "file": "weights/embed_tokens.bin",
      "shape": [151936, 896],
      "dtype": "float16"
    }
  }
}
```

Uses HF-style tensor names as keys (the canonical interface for `GpuWeights`).

#### Writer (`ironmill-compile/src/gpu/bundle.rs`)

```rust
pub fn write_gpu_bundle(
    provider: &MilWeightProvider,
    output_dir: impl AsRef<Path>,
) -> Result<(), CompileError>
```

#### Reader (`ironmill-inference/src/gpu/bundle.rs`)

Implements `WeightProvider` by memory-mapping the bundle files.

```rust
pub struct GpuBundleProvider { ... }

impl GpuBundleProvider {
    pub fn open(bundle_path: impl AsRef<Path>) -> Result<Self, GpuError>;
}

impl WeightProvider for GpuBundleProvider { ... }
```

#### Acceptance criteria

- Round-trip: `MilWeightProvider` → `write_gpu_bundle()` → `GpuBundleProvider::open()` → `GpuWeights::load()` produces identical Metal buffers.
- Bundle files are memory-mappable for fast loading.
- `manifest.json` is human-readable.

**Dependencies:** Task 6

---

## Phase 3 — In-VRAM quantized matmul

### Task 8 — Quantized matmul Metal kernels

**Goal:** Custom Metal compute shaders that read packed quantized weights
and compute matmul with inline dequantization. Keeps weights compressed in
GPU memory.

**File:** new `crates/ironmill-inference/src/gpu/shaders/quantized_matmul.metal`

#### Kernel: `polarquant_matmul`

```metal
// A = activations [M, K] (FP16, row-major)
// B = packed weights [N, K/pack_ratio] (UINT8)
// lut = [2^n_bits] (FP16)
// norms = [N] (FP16)
// C = output [M, N] (FP16)

kernel void polarquant_matmul_int4(
    device const half *A            [[buffer(0)]],
    device const uchar *B_packed    [[buffer(1)]],
    constant half *lut              [[buffer(2)]],
    device const half *norms        [[buffer(3)]],
    device half *C                  [[buffer(4)]],
    constant uint &M                [[buffer(5)]],
    constant uint &N                [[buffer(6)]],
    constant uint &K                [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint row = tid.y;  // output row (M dimension)
    uint col = tid.x;  // output col (N dimension)
    if (row >= M || col >= N) return;

    half acc = 0.0h;
    half norm = norms[col];
    for (uint k = 0; k < K; k += 2) {
        uchar packed = B_packed[col * (K / 2) + k / 2];
        half w0 = lut[packed & 0xF] * norm;
        half w1 = lut[packed >> 4] * norm;
        acc += A[row * K + k]     * w0;
        acc += A[row * K + k + 1] * w1;
    }
    C[row * N + col] = acc;
}
```

Production version will use threadgroup tiling and SIMD reductions.
Follow the patterns in `turboquant.metal` for threadgroup sizing.

An INT8 variant (`polarquant_matmul_int8`) is similar but reads one value
per byte with no nibble unpacking.

#### Acceptance criteria

- Kernel produces correct output vs. MPS FP16 matmul (within quantization tolerance).
- Benchmarked: memory savings and throughput vs. dense MPS path.
- Shader added to `GpuPipelines::compile()` in `gpu/ops.rs` via
  `include_str!` + `device.compile_shader_source()` (same pattern as
  existing shaders: normalization, activation, rope, turboquant, etc.).

**Dependencies:** Task 6

---

### Task 9 — Quantized weight buffers in `GpuWeights`

**Goal:** Store packed weights, LUTs, and norms as separate Metal buffers
instead of dequantizing to FP16.

**File:** `crates/ironmill-inference/src/gpu/weights.rs`

#### New types

```rust
pub enum WeightBuffer {
    /// Dense FP16 buffer for MPS matmul.
    Dense(MetalBuffer),
    /// Packed quantized buffer for custom kernel.
    Quantized(QuantizedWeight),
}

pub struct QuantizedWeight {
    pub indices: MetalBuffer,    // packed n-bit indices
    pub lut: MetalBuffer,        // [2^n_bits] reconstruction levels
    pub norms: MetalBuffer,      // [rows] row norms
    pub n_bits: u8,
    pub shape: (usize, usize),   // (out_features, in_features)
}
```

`LayerWeights` fields change from `MetalBuffer` to `WeightBuffer`.

#### Loading logic

When `QuantizationInfo::LutToDense` is present and Phase 3 kernels are
available, create a `WeightBuffer::Quantized` instead of dequantizing.
Small tensors (embeddings, norms, lm_head) remain `WeightBuffer::Dense`.

#### Acceptance criteria

- 7B model fits in ~4 GB VRAM with INT4 PolarQuant (vs ~14 GB FP16).
- `WeightBuffer::Dense` path unchanged for non-quantized tensors.

**Dependencies:** Task 8

---

### Task 10 — Matmul dispatch in `GpuInference`

**Goal:** Branch between MPS FP16 matmul and custom quantized kernel
per layer/projection.

**File:** `crates/ironmill-inference/src/gpu/inference.rs`

#### Changes

In the per-layer decode loop, replace direct MPS matmul calls:

```rust
// Before (FP16 only):
mps_matmul(&q_proj_buf, &input_buf, ...);

// After (dispatch by weight type):
match &layer.q_proj {
    WeightBuffer::Dense(buf) => {
        mps_matmul(buf, &input_buf, ...);
    }
    WeightBuffer::Quantized(q) => {
        encode_polarquant_matmul(encoder, &input_buf, q);
    }
}
```

#### Mixed precision strategy

| Tensor | Format | Rationale |
|--------|--------|-----------|
| `embed_tokens` | FP16 | Small relative to layers |
| `lm_head` | FP16 | Small, final output quality |
| `*_layernorm.weight` | FP16 | Tiny (hidden_size) |
| `q/k/v/o_proj` | Quantized | Large, bandwidth-bound |
| `gate/up/down_proj` | Quantized | Large, bandwidth-bound |

#### Acceptance criteria

- Mixed Dense/Quantized layers produce correct output.
- Decode loop runs without per-op CPU synchronization (single command buffer).
- Throughput ≥ FP16 MPS for bandwidth-bound models (7B+).

**Dependencies:** Task 9

---

## Phase 4 — CLI + benchmarking

### Task 11 — CLI `--target gpu` flag

**Goal:** `ironmill compile --target gpu --quantize polarquant-4` writes
a `.ironml-gpu` bundle.

**File:** `crates/ironmill-cli/src/compile.rs`

**Dependencies:** Task 7

---

### Task 12 — Bench integration

**Goal:** `ironmill bench model.ironml-gpu --backend gpu` reports memory,
tok/s, and perplexity for quantized GPU inference.

**File:** `crates/ironmill-bench/src/main.rs`

**Dependencies:** Task 10, Task 11

---

## Key design decisions

**Custom kernels over MPS quantized APIs.** Apple's `MPSMatrixMultiplication`
has no public quantized weight variant. Custom Metal compute shaders follow
the TurboQuant pattern already proven in the codebase.

**Phase 1 dequant-at-load is mandatory.** Validates correctness before
writing custom kernels. Catches bugs in LUT reconstruction, index packing,
row norm application, and the name mapping between MIL ops and HF tensors.

**Original ONNX names preserved.** `initializer_to_const()` stores the
original ONNX tensor name as an `onnx_name` attribute on every `const` op.
This attribute survives MIL name sanitization (which only renames outputs
and references, not string attribute values) and PolarQuantPass (which
only removes the `val` attribute). `MilWeightProvider` reads `onnx_name`
to map quantized ops back to HF tensor names — no fragile reverse-
sanitization needed.

**Separate `GpuCompileBuilder`.** The existing `CompileBuilder` is CoreML-
specific (writes `.mlpackage`, optionally compiles `.mlmodelc`). A separate
GPU builder avoids overloading the CoreML API surface.

**PolarQuant over GPTQ/AWQ.** No calibration data needed, no per-group
scales/zeros stored (just LUT + row norms), competitive quality. The
rotation-based approach maps to a simple LUT-indexed dequant in the shader.

## Interaction with TurboQuant

PolarQuant (weight compression) and TurboQuant (KV cache compression)
are orthogonal:

| Component | Method | Storage | Reduction |
|-----------|--------|---------|-----------|
| Weights | PolarQuant 4-bit | packed indices + LUT | 3.7× |
| KV cache | TurboQuant INT4 | packed nibbles | 4× |

Both can be active simultaneously. The matmul input (activations) and
output are FP16 regardless of weight format. TurboQuant operates on K/V
tensors after projection, not on weight buffers.

## Open questions

- **Rotation fusion on GPU:** Should `PolarRotationFusionPass` run for
  the GPU path? It cancels Hadamard rotations between sequential linear
  layers. Worth testing for throughput impact but not blocking for Phase 1.
- **Per-layer mixed precision:** Which layers benefit most from staying
  FP16? First/last layers and attention projections are candidates for
  higher precision. Needs empirical testing.
- **MPS fallback:** Should the GPU path fall back to MPS FP16 matmul
  when custom kernels are slower (small matrices)? Profile to determine
  crossover point.
