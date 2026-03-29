# Known Issues & Technical Debt

**Last updated**: 2026-03-29

Tracks shortcuts, workarounds, and unresolved issues. Grouped by severity.

---

## PolarQuant

### P0 — Correctness

#### Approximate un-rotation for non-power-of-two dimensions

The Hadamard rotation requires power-of-two dimensions. When a weight tensor
has a non-power-of-two inner dimension (e.g., 384, 1280), the pass pads,
rotates, then truncates indices back. The rotation fusion pass generates R_inv
as a `[cols, cols]` submatrix of the full `[padded_cols, padded_cols]` rotation.

This is lossy — truncation discards coordinates that participated in the
rotation. Quality impact depends on the padding ratio (384→512 is 25%;
1280→2048 is 37%).

**Fix**: Keep full padded representation for fallback layers. Store indices at
`padded_cols` width, emit matmul with full R_inv, then `slice_by_index` to trim.

**Files**: `polar_quantize.rs`, `polar_rotation_fusion.rs`

#### Inter-layer rotation fusion never fires on real models

The pairing algorithm traces `constexpr_lut_to_dense` → consumer linear op →
activation → next linear op, but doesn't follow through the norms `const` +
`mul` ops that PolarQuantPass inserts between the weight and consumer.

**Impact**: Every PolarQuant'd layer pays R_inv storage + matmul cost. On
Qwen3-0.6B this means 256 R_inv matrices instead of ~0 for paired layers.

**Fix**: Trace through `mul` (norms scaling) to find the actual consumer linear op.

**Files**: `polar_rotation_fusion.rs`

### P1 — Functionality

#### 3-bit PolarQuant blocked by CoreML LUT restriction

CoreML `constexpr_lut_to_dense` only accepts LUT sizes {2, 4, 16, 64, 256}.
3-bit (8 entries) is rejected. The pipeline accepts `n_bits=3` and quantizes
correctly, but CoreML compilation fails.

**Options**: Pad LUT to 16 entries (4-bit indices), route through `ane-direct`,
or reject `n_bits=3` at pipeline level.

#### R_inv storage not deduplicated

Layers with the same `(padded_cols, seed)` should share one R_inv const, but
a separate copy is emitted per tensor. On Qwen3-0.6B this is ~1GB of
redundant rotation matrices.

**Fix**: Deduplicate by `(padded_cols, seed)` key.

#### Quality benchmarks are stubs

`ironmill-bench/src/quality.rs` doesn't dequantize LUT values to compute
real MSE/PSNR. Only reports compression ratios.

#### `bench-size.sh` missing PolarQuant config

Size benchmark script doesn't include `--polar-quantize 4`.

---

## ONNX Conversion

### P0 — Correctness

#### Name sanitizer can alias distinct ops

`sanitize_mil_name` replaces all non-alphanumeric chars with `_`. Names like
`layer/0/conv` and `layer.0.conv` both become `layer_0_conv`. No uniqueness
check — could silently alias different ops.

**Fix**: Track seen names, append numeric suffix on collision.

**Files**: `onnx_graph.rs`

### P1 — Functionality

#### Transformer ops are opaque pass-throughs

`RotaryEmbedding` and `GroupQueryAttention` (ORT contrib ops) are mapped to
composite MIL ops that CoreML doesn't understand. Models convert through
ironmill but fail CoreML compilation.

**Fix**: Decompose into standard MIL ops (~200-400 lines each).

#### `gather` with int64 indices rejected by CoreML

ONNX uses int64 for indices; CoreML only accepts int32. Affects Whisper and
models with Shape → Gather patterns.

**Fix**: Insert `cast(dtype=int32)` before gather when indices are int64.

---

## CoreML Serialization

### P1 — Functionality

#### FP32 LUT round-trip deserializes as List, not Tensor

`constexpr_lut_to_dense` with FP32 LUT comes back as `Value::List(Float(...))`
after proto deserialization instead of `Value::Tensor`. Test was loosened to
accept both formats rather than fixing the deserializer.

**Files**: `convert/proto_to_ir.rs`

#### Row-norm ops lack explicit output types

The `const` and `mul` ops inserted for row norms rely on
`TypeRepropagationPass` to infer types. May produce incorrect shapes in
complex graphs.

**Fix**: Set `output_types` explicitly when creating norms/mul ops.

---

## Pre-existing Issues

#### SqueezeNet BNNS compilation warnings

`CreateBnnsGraphProgramFromMIL: input size and padding` warnings on CPU.
Model still runs via BNNS fallback. Indicates padding/pooling serialization issue.

#### ANE direct backend broken on current macOS

All configs fail with `_ANECompiler : ANECCompile() FAILED`. Compatibility
issue between private ANE APIs and current macOS version.
