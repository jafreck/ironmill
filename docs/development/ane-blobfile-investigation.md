# ANE BLOBFILE Weight Investigation

> **Status:** Open
>
> **Discovered during:** TurboQuant implementation
>
> **Impact:** All ANE-direct MIL programs with embedded weight blobs

## Problem

`AneCompiler::compile_mil_text()` fails when MIL text contains BLOBFILE
weight references, even when the weight dictionary is correctly provided.

```mil
tensor<fp16, [1,1,64,64]> R = const()[name=string("R"),
    val=tensor<fp16, [1,1,64,64]>(BLOBFILE(
        path=string("@model_path/weights/R.bin"),
        offset=uint64(64)))];
```

The ANE compiler returns `_ANECompiler : ANECCompile() FAILED` with no
further error detail.

## Evidence

**Programs that fail:**
- Any MIL program with a `BLOBFILE(...)` const, regardless of the ops used
- Even a simple `add(x=input, y=BLOBFILE_weight)` fails

**Programs that pass (same ops, same shapes):**
- The same weight data delivered as a function input (`a_inputN`) works
- All 30 eval tests using inline scalar constants pass
- The full TurboQuant pipeline (30/30 checks) works with inline consts

## Workaround

Pass weight tensors as function inputs (IOSurface-backed `AneTensor`)
instead of embedding them as `const` ops with BLOBFILE references. The
caller populates the IOSurface with the weight data once at compile time.

TurboQuant uses this approach: rotation/un-rotation matrices are passed
as `a_input2` / `a_input3` function parameters.

## Root Cause Candidates

1. **Path format** — `@model_path/weights/name.bin` may not be resolved
   correctly by the in-memory compilation path. The `mlpackage` path
   compilation may handle it differently.

2. **Offset** — `offset=uint64(64)` may be wrong for the in-memory
   weight dictionary. The `compile_mil_text` implementation in
   `crates/mil-rs/src/ffi/ane.rs` builds an `NSDictionary` of weight
   blobs; the offset may need to be 0 for in-memory data.

3. **Weight dictionary encoding** — The NSData objects in the weight
   dictionary may need different formatting (e.g., including a 64-byte
   header before the payload to match the offset=64 convention).

4. **Weight name mapping** — The dictionary key format may not match
   what the MIL parser expects from the `path=string(...)` reference.

## Investigation Steps

1. Check the `compile_mil_text` implementation in `crates/mil-rs/src/ffi/ane.rs`
   for how the weight dictionary is constructed
2. Try `offset=uint64(0)` instead of `offset=uint64(64)`
3. Try weight path without `@model_path/` prefix
4. Add debug logging to capture the Objective-C error details from
   `_ANEInMemoryModelDescriptor`
5. Compare with the `ir_to_mil_text.rs` BLOBFILE emission path used
   by the standard `AneModel::compile_and_load` pipeline (which works)

## Files

- `crates/mil-rs/src/ffi/ane.rs` — `AneCompiler::compile_mil_text`
- `crates/mil-rs/src/convert/ir_to_mil_text.rs` — BLOBFILE emission for standard path
- `crates/ironmill-ane/src/turboquant_mil.rs` — workaround implementation
