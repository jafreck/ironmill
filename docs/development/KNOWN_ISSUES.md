# Known Issues & Technical Debt

**Last updated**: 2026-03-29

Tracks shortcuts, workarounds, and unresolved issues. Grouped by severity.

---

## Architecture

### Resolved — Unify runtime backends behind a common trait

The `ironmill-runtime` crate now provides `RuntimeBackend` and `RuntimeModel`
traits. Both `ironmill-coreml` and `ironmill-ane` implement them, and the
benchmark harness uses a single inference path over `&[Box<dyn RuntimeBackend>]`.

The legacy `Backend` enum and `compile_model_with_backend()` in
`mil-rs/src/compiler.rs` have been removed. The CLI now uses `compile_model`
directly (xcrun only).

---

## PolarQuant

### Resolved

#### Approximate un-rotation for non-power-of-two dimensions

Fixed. The quantize pass (`polar_quantize.rs`) now keeps the full padded
representation instead of truncating back to original columns. The rotation
fusion pass's `slice_by_index` handles trimming to original dimensions after
un-rotation.

#### Inter-layer rotation fusion fires on real models

Fixed. The pairing algorithm now traces through multiple intermediate
element-wise ops (add, mul, activation functions) up to 4 hops when looking
for producer-consumer linear pairs, matching common transformer graph patterns
like linear → activation → mul → linear.

#### 3-bit PolarQuant blocked by CoreML LUT restriction

CoreML `constexpr_lut_to_dense` only accepts LUT sizes {2, 4, 16, 64, 256}.
3-bit (8 entries) is rejected. The pipeline CLI and validation now correctly
advertise only `n_bits = 2 or 4`. The `PolarQuantPass` rejects unsupported
LUT sizes early.

---

## ONNX Conversion

### Resolved

#### Transformer ops decomposed into standard MIL ops

`RotaryEmbedding` and `GroupQueryAttention` (ORT contrib ops) are now
decomposed into standard MIL operations (split, mul, sub, add, concat,
gather, reshape, transpose, matmul, softmax, tile, identity). Models
using these ops will compile through CoreML without custom op errors.

---

## Pre-existing Issues

#### SqueezeNet BNNS compilation warnings

`CreateBnnsGraphProgramFromMIL: input size and padding` warnings on CPU.
Model still runs via BNNS fallback. Indicates padding/pooling serialization issue.

---

## Resolved

- **ANE direct backend broken on current macOS** — Working on macOS 26.4 / M2 Max. All e2e tests pass including compile, load, and predict. The issue was stale.
- **Unify runtime backends** — `ironmill-runtime` crate with `RuntimeBackend`/`RuntimeModel` traits; legacy `Backend` enum and `compile_model_with_backend` removed from `mil-rs/src/compiler.rs`
- **PolarQuant non-power-of-two truncation** — Quantize pass keeps full padded indices; fusion pass handles trimming via `slice_by_index`
- **Inter-layer rotation fusion pairing** — Pairing algorithm traces through multiple intermediate element-wise ops (up to 4 hops)
- **3-bit CLI messaging** — CLI and pipeline validation now correctly advertise `n_bits = 2 or 4` only
- **Transformer op decomposition** — `RotaryEmbedding` and `GroupQueryAttention` decomposed into standard MIL ops
- **R_inv storage not deduplicated** — Deduplicated by `(padded_cols, seed)` key in `polar_rotation_fusion.rs`
- **Quality benchmarks are stubs** — Real MSE/PSNR/dequantization implemented in `ironmill-bench/src/quality.rs`
- **`bench-size.sh` missing PolarQuant config** — Script now includes `--polar-quantize 4`
- **Name sanitizer can alias distinct ops** — `sanitize_block_names()` tracks seen names and appends numeric suffixes (test: `sanitize_block_names_deduplicates_collisions`)
- **`gather` with int64 indices rejected by CoreML** — `convert_gather()` inserts `cast(dtype=int32)` before gather
- **FP32 LUT round-trip deserializes as List, not Tensor** — `proto_to_ir.rs` reconstructs FP32 LUT tensors as `Value::Tensor`
- **Row-norm ops lack explicit output types** — `polar_quantize.rs` now sets `output_types` explicitly
