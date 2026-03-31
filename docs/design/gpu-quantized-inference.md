# GPU Quantized Weight Inference

> Design doc for PolarQuant and palettized weight support in the Metal GPU backend.
>
> **Status**: Planned — Phase 1 (load-time dequant) not yet started.
>
> **Prerequisites**: [PolarQuant Weight Quantization](polarquant-weight-quantization.md),
> [Metal GPU Backend](gpu-backend.md), [TurboQuant](turboquant.md)

## Problem

The Metal GPU backend loads all weights as dense FP16. A 7B-parameter model
requires ~14 GB of VRAM just for weights, and inference throughput is
memory-bandwidth-bound — each decode step reads every weight once for every
MPS matmul.

The ironmill compiler already produces quantized weight representations
(PolarQuant 4-bit, palettization, INT8 affine) via `mil-rs` passes, but
the GPU backend cannot consume them. Weights are dequantized to FP16 before
they ever reach a Metal buffer.

| Model | FP16 weights | 4-bit PolarQuant | VRAM savings |
|-------|-------------|-------------------|--------------|
| Qwen3-0.6B | 1.5 GB | ~400 MB | 3.7× |
| Qwen3-4B | 8.2 GB | ~2.2 GB | 3.7× |
| Llama-3-8B | 16 GB | ~4.3 GB | 3.7× |

Beyond memory, 4-bit weights reduce memory bandwidth per matmul by ~4×,
directly increasing tok/s for bandwidth-bound decode.

## Current State

### What exists

The compiler's quantization passes and the GPU's TurboQuant KV cache prove
that every layer of the stack — IR passes, Metal shaders, packed integer
buffers — already works. The gap is connecting them for weights.

**Compiler side (ready):**
- `PolarQuantPass` produces `constexpr_lut_to_dense` with LUT + packed
  indices + row norms. Backend-agnostic MIL IR. (`mil-rs/ir/passes/polar_quantize.rs`)
- `PalettizePass` produces the same op with k-means codebooks.
- `Int8QuantizePass` produces `constexpr_affine_dequantize` with
  scale/zero_point.
- `PolarRotationFusionPass` cancels adjacent Hadamard rotations.
- CLI: `--polar-quantize 4` already wired in `ironmill-cli` and
  `ironmill-bench`.

**GPU side (gap):**
- `GpuWeights::load()` copies raw bytes into `MTLBuffer`s. No dtype
  inspection, no quantization metadata. (`gpu/weights.rs`)
- `MpsMatrixMultiply` hardcodes `MPSDataTypeFloat16`. (`ironmill-metal-sys`)
- `WeightProvider` returns `(data, shape, dtype)` — no quantization info.
- GGUF provider dequantizes Q4_0/Q8_0 → FP16 at load time, discarding
  quant metadata. (`weights/gguf.rs`)

**GPU TurboQuant (proves the pattern):**
- INT4 packed nibble KV cache: write shader packs two 4-bit values per byte,
  attention shader unpacks with nibble extraction. (`gpu/shaders/turboquant.metal`)
- `GpuKvCache` sizes buffers by `n_bits`. (`gpu/turboquant/cache.rs`)
- Both INT4 and INT8 paths are fully implemented and benchmarked.

### ANE INT4 status (not applicable here)

ANE rejects `constexpr_lut_to_dense` via the private API — weights must
be FP16 at runtime. INT4 TurboQuant on ANE maps 4-bit quantization levels
into INT8 storage (no packed nibbles). These are ANE hardware limitations
that do not apply to the Metal GPU path.

## Architecture

### Data flow

```
                    ┌─────────────────────┐
                    │  ONNX / SafeTensors  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    mil-rs Program    │
                    │  (FP16 const ops)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   PolarQuantPass    │
                    │   + RotationFusion  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ constexpr_lut_to_   │
                    │ dense + row norms   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
   ┌──────────▼──────┐ ┌──────▼──────┐ ┌───────▼───────┐
   │  Phase 1:       │ │  Phase 2:   │ │  Phase 3:     │
   │  CPU dequant    │ │  GPU bundle │ │  In-VRAM      │
   │  → FP16 buffer  │ │  serialize  │ │  quantized    │
   │  → MPS matmul   │ │  + reload   │ │  matmul       │
   └─────────────────┘ └─────────────┘ └───────────────┘
```

### Phase 1: Load-time dequant (correctness baseline)

Dequantize on the CPU when creating Metal buffers. No VRAM savings, but
validates the full pipeline and establishes a correctness reference.

```
Compiled Program (with constexpr_lut_to_dense)
  → MilWeightProvider extracts LUT + indices + norms
  → CPU: unpack indices, apply LUT, multiply by row norms → FP16
  → create_buffer_with_data(fp16_bytes, Shared)
  → existing MPS matmul path (unchanged)
```

#### New abstractions

**`QuantizationInfo` enum** (in `ironmill-compile/src/weights/mod.rs`):

```rust
pub enum QuantizationInfo {
    None,
    LutToDense {
        lut: Vec<f16>,          // [2^n_bits] reconstruction levels
        indices: Vec<u8>,       // packed n-bit indices
        original_shape: Vec<usize>,
        n_bits: u8,
    },
    AffineDequantize {
        scale: Vec<f16>,        // per-tensor or per-channel
        zero_point: Vec<f16>,
        axis: Option<usize>,
    },
}
```

**`MilWeightProvider`** (new, in `ironmill-compile/src/weights/mil_provider.rs`):

Wraps a compiled `mil_rs::ir::Program`. Walks `constexpr_lut_to_dense` and
`constexpr_affine_dequantize` ops. Exposes each as a `WeightTensor` with
the appropriate `QuantizationInfo` variant. Maps MIL op output names to
canonical HuggingFace tensor names.

**`GpuWeights::load()` changes:**

```rust
match tensor.quant_info {
    QuantizationInfo::None => {
        // existing path: raw bytes → MTLBuffer
    }
    QuantizationInfo::LutToDense { lut, indices, shape, n_bits } => {
        let fp16 = dequant_lut_to_dense(&lut, &indices, &shape, n_bits);
        let fp16 = apply_row_norms(&fp16, &norms, &shape);
        device.create_buffer_with_data(&fp16, Shared)
    }
    QuantizationInfo::AffineDequantize { scale, zero_point, axis } => {
        let fp16 = dequant_affine(&tensor.data, &scale, &zero_point, axis);
        device.create_buffer_with_data(&fp16, Shared)
    }
}
```

### Phase 2: GPU bundle format

Serialize compiled quantized models so they don't require recompilation.

**`.ironml-gpu` bundle** (mirrors ANE `.ironml`):

```
model.ironml-gpu/
├── manifest.json          # model config, quant config, tensor index
└── weights/
    ├── layer_0_q_proj.bin # packed indices
    ├── layer_0_q_proj.lut # LUT values
    ├── layer_0_q_proj.nrm # row norms
    ├── ...
    └── embed_tokens.bin   # FP16 (below min_elements threshold)
```

`manifest.json`:
```json
{
  "format_version": 1,
  "model_config": { "...": "..." },
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
      "n_bits": 4
    },
    "model.embed_tokens.weight": {
      "format": "dense_fp16",
      "file": "weights/embed_tokens.bin",
      "shape": [151936, 896]
    }
  }
}
```

### Phase 3: In-VRAM quantized matmul

Keep weights packed in GPU memory. Custom Metal kernels dequantize inline
during matmul. This is the phase that delivers real memory and bandwidth
savings.

#### Kernel design

Follow the TurboQuant shader pattern: a single fused kernel that reads
packed weights, dequantizes per-element, and accumulates the matmul.

```metal
// Pseudocode for INT4 PolarQuant matmul
// A = activations [M, K] (FP16)
// B = packed weights [N, K/2] (UINT8, 2 values per byte)
// lut = [16] (FP16 reconstruction levels)
// norms = [N, 1] (FP16 row norms)
// C = output [M, N] (FP16)

kernel void polarquant_matmul(
    device const half *A,
    device const uchar *B_packed,
    device const half *lut,
    device const half *norms,
    device half *C,
    constant uint &M,
    constant uint &N,
    constant uint &K,
    uint2 tid [[thread_position_in_grid]])
{
    half acc = 0;
    for (uint k = 0; k < K; k += 2) {
        uchar packed = B_packed[row * (K/2) + k/2];
        half w0 = lut[packed & 0xF] * norms[row];
        half w1 = lut[packed >> 4] * norms[row];
        acc += A[col * K + k]     * w0;
        acc += A[col * K + k + 1] * w1;
    }
    C[col * N + row] = acc;
}
```

Actual implementation will use threadgroup tiling and SIMD reductions
matching the GPU's warp width.

#### Metal buffer layout for quantized weights

```rust
pub struct QuantizedLayerWeights {
    pub indices: MetalBuffer,    // packed n-bit indices [out, in/pack_ratio]
    pub lut: MetalBuffer,        // [2^n_bits] reconstruction levels
    pub norms: MetalBuffer,      // [out, 1] row norms
    pub n_bits: u8,
    pub shape: (usize, usize),   // [out_features, in_features]
}
```

#### Matmul dispatch

In `GpuInference`, branch per layer:

```rust
match &layer.q_proj {
    WeightBuffer::Dense(buf) => mps_matmul(buf, input),
    WeightBuffer::Quantized(q) => {
        encode_polarquant_matmul(encoder, input, q);
    }
}
```

Mixed precision: embeddings, final norm, and lm_head remain FP16 (small
relative to per-layer weights). Linear projections in transformer layers
use quantized matmul.

### Phase 4: CLI + benchmarking

```bash
# Compile for GPU with PolarQuant
ironmill compile model.onnx --target gpu --quantize polarquant-4

# Benchmark quantized GPU inference
ironmill bench model.ironml-gpu --backend gpu
```

## Performance Expectations

### Memory

| Component | FP16 | INT4 PolarQuant | Notes |
|-----------|------|-----------------|-------|
| Weights (7B) | 14 GB | ~3.8 GB | 3.7× reduction |
| KV cache (4K ctx) | 1 GB | 250 MB | INT4 TurboQuant (already done) |
| Activations | ~50 MB | ~50 MB | Unchanged |
| **Total** | **~15 GB** | **~4.1 GB** | Fits M1/M2 base (8 GB) |

### Throughput

Decode throughput for bandwidth-bound models should improve roughly
proportional to the bandwidth reduction. Reading 4-bit weights instead of
16-bit is a ~4× reduction in memory traffic per matmul. The dequant ALU
cost (LUT lookup + multiply by norm) is negligible compared to the memory
latency saved.

Prefill throughput is typically compute-bound, not bandwidth-bound, so
gains will be smaller.

## Design Decisions

**Custom kernels over MPS quantized APIs.** Apple's `MPSMatrixMultiplication`
does not expose a public quantized weight matmul. Custom Metal compute
shaders follow the proven TurboQuant pattern already in the codebase.

**PolarQuant over GPTQ/AWQ.** PolarQuant needs no calibration data, stores
no per-group scales/zeros (just a LUT + row norms), and achieves competitive
quality. The rotation-based approach maps cleanly to a simple LUT-indexed
dequant in the shader.

**Phase 1 dequant-at-load is mandatory.** Validates correctness before
investing in custom kernels. Catches bugs in LUT reconstruction, index
packing, row norm application, and name mapping between MIL ops and HF
tensor names.

**Bundle format mirrors ANE `.ironml`.** Consistent conventions, manifest
structure, and weight blob layout. Separate files per tensor for
memory-mapped loading.

## Interaction with TurboQuant

PolarQuant (weight compression) and TurboQuant (KV cache compression)
are orthogonal and composable:

```
Weights:    PolarQuant 4-bit → 3.7× smaller, 4× less bandwidth
KV cache:   TurboQuant INT4  → 4× smaller, 4× less bandwidth
Combined:   both active simultaneously, independent buffers
```

The GPU decode loop already dispatches TurboQuant vs. FP16 cache paths
based on config. Quantized weight matmul is an independent axis — the
matmul input (activations) and output format are FP16 regardless of
whether weights are quantized.

## Files

| Phase | New / Modified |
|-------|---------------|
| 1 | `ironmill-compile/src/weights/mod.rs` (extend `WeightTensor`), new `mil_provider.rs`, `ironmill-inference/src/gpu/weights.rs` |
| 2 | New `ironmill-compile/src/gpu/bundle.rs`, new `ironmill-inference/src/gpu/bundle.rs` |
| 3 | New `ironmill-inference/src/gpu/shaders/quantized_matmul.metal`, `gpu/inference.rs`, `gpu/ops.rs` |
| 4 | `ironmill-cli/src/compile.rs`, `ironmill-bench/src/main.rs` |

## References

- [PolarQuant Weight Quantization](polarquant-weight-quantization.md) — algorithm details, ANE constraints
- [Metal GPU Backend](gpu-backend.md) — GPU decode pipeline architecture
- [TurboQuant](turboquant.md) — KV cache compression (INT4/INT8)
- [PolarQuant paper](https://arxiv.org/abs/2502.02617) — Han et al., 2025
- `crates/mil-rs/src/ir/passes/polar_quantize.rs` — compiler pass
- `crates/ironmill-inference/src/gpu/shaders/turboquant.metal` — INT4 shader reference
