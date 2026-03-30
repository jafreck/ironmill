# ANE MIL Text Emitter Compatibility

> **Status:** Open
>
> **Blocks:** [TurboQuant E2E Inference](turboquant-e2e-inference.md) — layer sub-program compilation
>
> **Discovered during:** BLOBFILE investigation → E2E inference bringup

## Problem

`ir_to_mil_text.rs` emits MIL text that the private ANE compiler
(`_ANECompiler`) rejects for real transformer sub-programs. The format
differences are specific and identified:

1. **`reshape` shape argument** — The emitter prints `reshape(shape=<tensor>, ...)`
   as a debug placeholder instead of emitting the actual shape value
   like `reshape(x=..., shape=tensor<int32, [4]>([1,1,1,128]))`.
   This is a bug in `format_value()` for `Value::Tensor` used as an
   op input (not a const val).

2. **BLOBFILE weight references** — `compile_mil_text()` fails on MIL
   text containing `BLOBFILE(path=..., offset=...)` const values. The
   working TurboQuant and ane_op_eval generators avoid BLOBFILE entirely
   by passing weights as function inputs (`a_inputN`). The general
   emitter needs the same approach: emit weights as function parameters
   instead of BLOBFILE-backed const ops.

## Root Cause

Two concrete format issues, identified by comparing working MIL text
(TurboQuant emitter, ane_op_eval examples) against failing MIL text
(ir_to_mil_text output for ONNX-converted layers):

### Issue 1: `reshape` shape argument (bug)

Working format (TurboQuant emitter):
```mil
tensor<fp16, [1,2,64,1]> reshaped = reshape(x=a_input0, shape=tensor<int32, [4]>([1,2,64,1]))[name=string("reshaped")];
```

Failing format (ir_to_mil_text):
```mil
tensor<fp16, [1,1,1,128]> out = reshape(shape=<tensor>, x=input)[name=string("out")];
```

`<tensor>` is the `Debug` format of a `Value::Tensor` variant, not
a valid MIL literal. `format_value()` needs to emit the tensor inline
(e.g., `tensor<int32, [4]>([1,1,1,128])`) when a tensor is used as
an op argument rather than a const val.

### Issue 2: BLOBFILE incompatibility (design)

Working format (TurboQuant / ane_op_eval):
```mil
func main<ios18>(tensor<fp16, [1,64,1,64]> a_input0, tensor<fp16, [1,64,1,64]> a_input2) {
    // weights are function inputs, populated via IOSurface before eval
```

Failing format (ir_to_mil_text):
```mil
tensor<fp16, [1024,1024]> weight = const()[name=string("weight"),
    val=tensor<fp16, [1024,1024]>(BLOBFILE(path=string("@model_path/weights/weight.bin"),
    offset=uint64(64)))];
```

The ANE compiler's `compile_mil_text()` API does not resolve BLOBFILE
references correctly, even when the weight dictionary is provided.
This was confirmed during the BLOBFILE investigation. The fix is to
emit weight tensors as function inputs and populate them via IOSurface
before eval, matching the TurboQuant approach.

## Approach

Two targeted fixes, not a broad survey:

### Fix 1 — `format_value()` for tensor op arguments

In `ir_to_mil_text.rs`, when a `Value::Tensor` appears as an op input
(not a const val), emit it as an inline typed tensor literal:

```rust
// Before (bug): prints Debug format
Value::Tensor { .. } => "<tensor>".to_string(),

// After: emit as MIL tensor literal
Value::Tensor { data, shape, dtype } => {
    // e.g., tensor<int32, [4]>([1, 1, 1, 128])
    format_inline_tensor(data, shape, dtype)
}
```

This only affects ops that take tensor-valued arguments inline (like
`reshape`'s `shape` parameter). Most ops use scalar or reference args.

### Fix 2 — Emit weights as function inputs instead of BLOBFILE

Change `emit_const_op` to emit large weight tensors as additional
function inputs (`a_inputN`) instead of BLOBFILE-backed const ops.
The caller pre-populates the IOSurface with weight data before eval.

This matches the working TurboQuant pattern. The weight data is the
same; only the delivery mechanism changes (IOSurface parameter vs
embedded const with BLOBFILE reference).

```
Before:
  func main(tensor<fp16, [1,1,1,1024]> a_input0) {
      tensor<fp16, [1024,1024]> w = const()[val=BLOBFILE(...)];
      y = matmul(x=a_input0, y=w);

After:
  func main(tensor<fp16, [1,1,1,1024]> a_input0,
            tensor<fp16, [1024,1024]> a_input1) {  // weight as input
      y = matmul(x=a_input0, y=a_input1);
```

## Files

| File | Role |
|---|---|
| `crates/mil-rs/src/convert/ir_to_mil_text.rs` | MIL text emitter (fix target) |
| `crates/ironmill-ane/src/turboquant_mil.rs` | Working MIL emitter (reference) |
| `crates/ironmill-ane/examples/ane_op_eval.rs` | Op-level ANE eval probes |
| `crates/ironmill-ane/examples/ane_op_probe.rs` | Op-level ANE compile probes |
| `docs/research/ane-op-support-matrix.md` | Known op support status |

## Key Insight

The working MIL text and failing MIL text use the **same dialect and
syntax** — same function signatures, same named argument format, same
const op format, same attribute syntax. The two concrete differences
are the `reshape` bug and BLOBFILE usage. This is not a broad format
incompatibility requiring systematic testing; it's two specific fixes.

## Fixes Already Applied (during investigation)

These were discovered and fixed while tracing the E2E compilation
failure. They apply to all ANE compilation, not just E2E inference:

1. **`emit_const_op` attribute lookup** — ONNX-converted const ops
   store weight tensors in `attributes["val"]`, not `inputs["val"]`.
   Fixed in `ir_to_mil_text.rs` to check both.
   *(This was the actual root cause of the BLOBFILE investigation.)*

2. **`reciprocal` decomposition** — ANE doesn't support `reciprocal`.
   Decomposed to `real_div(1, x)` in `AneInference` pre-compilation.

3. **Dynamic shape materialization** — ONNX models have dynamic `?`
   dimensions. Materialized to `1` for single-token decode shapes.

## References

- [ANE BLOBFILE Investigation](ane-blobfile-investigation.md) — predecessor (root cause resolved: #1 above)
- [TurboQuant E2E Inference](turboquant-e2e-inference.md) — blocked by this
- [ANE Op Support Matrix](../research/ane-op-support-matrix.md) — per-op status
