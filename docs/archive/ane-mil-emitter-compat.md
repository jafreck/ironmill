# ANE MIL Text Emitter Compatibility

> **Status:** Resolved
>
> **Blocked:** [TurboQuant E2E Inference](turboquant-e2e-inference.md) — layer sub-program compilation
>
> **Discovered during:** BLOBFILE investigation → E2E inference bringup

## Problem

`ir_to_mil_text.rs` emitted MIL text that the private ANE compiler
(`_ANECompiler`) rejected for real transformer sub-programs. Two
format issues were identified and fixed.

## Issues Found and Fixed

### Issue 1: `reshape` shape argument (fixed)

`format_value()` emitted `<tensor>` (Rust Debug format) instead of
a valid MIL tensor literal for `Value::Tensor` op arguments.

```mil
// Before (bug):
reshape(shape=<tensor>, x=input)

// After (fix):
reshape(x=input, shape=tensor<int32, [4]>([1,1024,1,1]))
```

**Fix:** `format_value()` now emits inline typed tensor literals via
`format_tensor_elements()`. Shapes are flattened to 1-D. Argument
ordering puts `x=` first.

### Issue 2: BLOBFILE weight references (unverified)

The original BLOBFILE investigation concluded that `compile_mil_text()`
fails on BLOBFILE references. However, this conclusion was reached
**before** the `emit_const_op` attribute lookup bug was fixed (the
weights were silently empty). The existing `AneModel::compile_and_load()`
E2E tests use BLOBFILE successfully with 4D weight shapes.

**Current status:** BLOBFILE with correct weight data and 4D type
declarations **has not been independently verified** for real
transformer-sized weights. The E2E compilation currently fails due
to an invalid reshape in the sub-program graph (see
[ANE Attention Split Investigation](ane-attention-split-investigation.md)),
which prevents reaching the BLOBFILE compilation path for valid programs.

**If BLOBFILE does fail** after the split issue is resolved, the fix
is to emit weights as function inputs (IOSurface parameters) instead
of BLOBFILE-backed const ops, matching the TurboQuant approach.

## Additional Fixes Applied

These were discovered and fixed while tracing the E2E compilation
failure. They apply to all ANE compilation, not just E2E inference:

1. **`emit_const_op` attribute lookup** — ONNX-converted const ops
   store weight tensors in `attributes["val"]`, not `inputs["val"]`.
   Fixed in `ir_to_mil_text.rs` to check both.
   *(This was the actual root cause of the BLOBFILE investigation.)*

2. **`reciprocal` decomposition** — ANE doesn't support `reciprocal`.
   Decomposed to `real_div(1, x)` in `AneInference` pre-compilation.

3. **Dynamic shape materialization** — ONNX models have dynamic `?`
   dimensions. Materialized to `1` before `AneLayoutPass` so the
   layout mapping sees static shapes.

4. **Weight const type declarations** — Reshaped to 4D in MIL text
   (`[1024]` → `[1,1024,1,1]`) via `to_ane_weight_shape()`.

5. **ANE layout dimension alignment** — Fixed 3D `[1,1,N]` → 4D
   mapping to put hidden_size in C (`[1,N,1,1]`) instead of S
   (`[1,1,1,N]`), aligning activations with weight shapes.

6. **Reshape shape sync** — `AneLayoutPass` now updates reshape ops'
   shape parameters to match the 4D output type.

## Files Changed

| File | Change |
|---|---|
| `crates/mil-rs/src/convert/ir_to_mil_text.rs` | `format_value` for tensors, arg ordering, weight 4D shapes |
| `crates/mil-rs/src/ir/passes/ane_layout.rs` | Reshape shape sync, 3D decode mapping, dynamic shape fix |
| `crates/ironmill-ane/src/inference.rs` | Pre-layout shape materialization, `reciprocal` decomposition |

## References

- [ANE BLOBFILE Investigation](ane-blobfile-investigation.md) — predecessor (root cause resolved: #1 above)
- [ANE Attention Split Investigation](ane-attention-split-investigation.md) — current blocker
- [TurboQuant E2E Inference](turboquant-e2e-inference.md)
- [ANE Op Support Matrix](../research/ane-op-support-matrix.md)
