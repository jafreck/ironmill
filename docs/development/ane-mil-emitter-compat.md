# ANE MIL Text Emitter Compatibility

> **Status:** Open
>
> **Blocks:** [TurboQuant E2E Inference](turboquant-e2e-inference.md) — layer sub-program compilation
>
> **Discovered during:** BLOBFILE investigation → E2E inference bringup

## Problem

`ir_to_mil_text.rs` emits MIL text that the private ANE compiler
(`_ANECompiler`) rejects for real transformer sub-programs, even when
all ops are individually ANE-supported.

The E2E inference pipeline (ONNX → MIL IR → passes → split → MIL text
→ ANE compile) successfully:
- loads the model and detects architecture
- splits layers at attention boundaries
- extracts weights as BLOBFILE entries
- materializes static shapes
- decomposes unsupported ops (`reciprocal` → `real_div`)

But `ANECCompile()` returns a generic failure with no error detail.

## Root Cause

The MIL text format produced by `emit_op()` in `ir_to_mil_text.rs`
doesn't match what the ANE compiler expects for certain argument
patterns. The emitter was built against simple test programs (add,
conv, linear) and hasn't been validated against the full op/argument
space that real ONNX-converted transformer layers produce.

Known format gaps:
1. **`reshape(shape=<tensor>)`** — The `shape` argument is emitted as
   a reference to a const tensor, but ANE may expect an inline int
   list like `shape=[1, 1, 1, 128]`.
2. **`reduce_mean(axes=tensor<int32,[1]>([-1]))`** — The axes format
   may need to be a plain int list, not a typed tensor literal.
3. **Argument ordering** — ANE's MIL parser may be sensitive to the
   order of named arguments (e.g., `x=` before `y=`), which `HashMap`
   iteration doesn't guarantee.
4. **Op name quoting** — Some op names from ONNX conversion contain
   dots and slashes that may need escaping or sanitization.

## Approach

Systematically test each op format against `ANECCompile()` by building
minimal MIL text programs with one op each, using the exact argument
formats from `ir_to_mil_text.rs`. This identifies which ops compile
and which need format fixes.

### Phase 1 — Minimal op compilation probes

For each op used in transformer layers, generate a minimal MIL program
and attempt ANE compilation. Record pass/fail and the exact MIL text.

**Ops to probe** (used in Qwen3-0.6B pre_attn/post_attn):

| Op | In support matrix | Used as | Priority |
|---|---|---|---|
| `add` | ✅ eval | RMSNorm, residual | High |
| `mul` | ✅ eval | RMSNorm, gating | High |
| `sub` | ✅ eval | RoPE (if kept) | Medium |
| `matmul` | ✅ eval | Q/K/V/O projection | High |
| `reduce_mean` | ✅ eval | RMSNorm | High |
| `sqrt` | ✅ eval | RMSNorm | High |
| `real_div` | ⚠️ compile | rsqrt decomposition | High |
| `reshape` | ✅ eval† | Head reshape | High |
| `softmax` | ✅ eval | Attention | Medium |
| `transpose` | ⚠️ compile | Head layout | Medium |
| `split` | ⚠️ compile | RoPE | Medium |
| `tile` | ✅ eval† | GQA head tiling | Medium |
| `silu` | ✅ eval | FFN activation | Medium |
| `cast` | ⚠️ compile | Type conversion | Low |

†Cannot compile standalone; only verified as intermediate ops.

### Phase 2 — Argument format normalization

For ops that fail, compare the emitted format against:
- The TurboQuant MIL emitter (`turboquant_mil.rs`) — which works
- The `ane_op_eval.rs` example — which also works
- coremltools' MIL text output for the same ops

Fix `emit_op()` argument formatting to match what ANE accepts.

### Phase 3 — Full layer compilation

Once individual ops pass, attempt full layer sub-program compilation.
If it still fails, bisect by removing ops until the minimal failing
combination is found.

## Files

| File | Role |
|---|---|
| `crates/mil-rs/src/convert/ir_to_mil_text.rs` | MIL text emitter (fix target) |
| `crates/ironmill-ane/src/turboquant_mil.rs` | Working MIL emitter (reference) |
| `crates/ironmill-ane/examples/ane_op_eval.rs` | Op-level ANE eval probes |
| `crates/ironmill-ane/examples/ane_op_probe.rs` | Op-level ANE compile probes |
| `docs/research/ane-op-support-matrix.md` | Known op support status |

## Key Insight

The TurboQuant MIL emitter (`turboquant_mil.rs`) works because it
generates MIL text by hand-crafting each op's format string — it knows
exactly what ANE accepts. The general-purpose `ir_to_mil_text.rs`
emitter uses generic formatting that doesn't account for ANE's
idiosyncratic MIL parser.

The fix path is to make `ir_to_mil_text.rs` emit ANE-compatible
format for each op, using `turboquant_mil.rs` and `ane_op_eval.rs`
as reference implementations for the correct syntax.

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
