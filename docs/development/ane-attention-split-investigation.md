# ANE Attention-Boundary Split — Structural Split Implementation Plan

> **Status:** Resolved (split) · In Progress (ANE compilation)
>
> **Blocks:** [TurboQuant E2E Inference](turboquant-e2e-inference.md) — layer sub-program compilation
>
> **Discovered during:** MIL emitter investigation → E2E inference bringup

## Problem

The attention-boundary split in `split_at_attention_boundary()` produces
sub-programs with **invalid reshape operations** — the reshape target
shape doesn't match the input element count. This affects any model with
ops between projections and attention (per-head norms, non-standard
reshape naming), not just Qwen3.

Example from `layer_1_pre_attn`:
```mil
// matmul output: 1024 elements
tensor<fp16, [1,1024,1,1]> k_proj = matmul(x=norm_out, y=k_weight)[...];

// reshape to 128 elements — INVALID (1024 ≠ 128)
tensor<fp16, [1,128,1,1]> k_reshaped = reshape(x=k_proj, shape=...)[...];
```

The ANE compiler correctly rejects this with `ANECCompile() FAILED`.

## Root Cause

The pre_attn sub-program only contains the **K projection** and
**K-norm**, not the full Q/K/V projection set. The reshape from 1024
to 128 dims is valid in the original model because it follows a K
projection that outputs `num_kv_heads × head_dim = 8 × 128 = 1024`
— then reshapes to `[batch, seq, num_kv_heads, head_dim]` before
per-head normalization. But in the split sub-program, the reshape
target was converted to a flat `[1, 128, 1, 1]` by the ANE layout
pass, which collapses the multi-head structure into a single-head
shape.

The attention-boundary split uses **op name heuristics** (substring
matching on `_q_reshape`, `_k_reshape`, etc.) to identify the
attention cluster. This is fundamentally fragile:

- Model-internal reshapes (e.g., K-norm reshape named
  `_attn_k_norm_Reshape_1`) get caught by `_k_reshape` substring
  matching even though they are normalization ops, not attention ops.
- Models with per-head K/V norms (e.g., Qwen3's
  `SimplifiedLayerNormalization`) have reshape → norm → reshape
  sequences between projections and attention. The boundary between
  "projection ops" and "attention ops" is not separable by name alone.
- Layer 0 has no pre_attn sub-program at all — every op was
  classified as an attention cluster op and stripped entirely.
- Layers 1–28 have pre_attn sub-programs with only the K projection
  path; Q and V projections were incorrectly stripped.

## Resolution — Structural Split

Replaced the name-based heuristic with a **data-flow graph traversal**
(`a016ae6`). The implementation:

- **`OpGraph`**: directed dependency graph built from `Value::Reference`
  edges in the flat op list. Provides `walk_forward()` / `walk_backward()`
  transitive traversal.
- **Q/K/V projection detection**: groups matmul/linear ops by shared
  non-const activation input. The group of 2–3 matmuls sharing the same
  norm output are the projections. This avoids false positives from FFN
  matmuls (which have different activation inputs).
- **O-projection detection**: first matmul/linear with a const weight
  reachable forward from softmax (excluding Q/K/V projections).
- **Split classification**:
  - Pre-attn = backward walk from projections' non-const data inputs
  - Post-attn = O-projection + forward walk + exclusively-feeding consts
  - Attention cluster = everything else (stripped)
- **Fallback**: legacy name heuristic if structural detection fails
  (e.g., layers without softmax).

Verified on Qwen3-0.6B: all 29 layers produce correct pre_attn and
post_attn sub-programs. Layer 0 (previously dropped) is now included.
Per-head K/V norm reshapes correctly stay in the attention cluster.

## Additional Fixes Discovered During E2E Verification

E2E testing with `turboquant_e2e_bench` on Qwen3-0.6B revealed that
the structural split was necessary but not sufficient — the sub-programs
also failed ANE compilation due to MIL emission issues. These were fixed
in `7accb97` and `f1f7b6b`:

### 1. AneArgPromotionPass (new pass)

The ANE compiler requires `reduce_mean`/`softmax`/`layer_norm` args
(`axes`, `keep_dims`, `epsilon`) to be named const references, not
inline literal values. The ONNX converter stores these as inline
`Value::Tensor`/`Value::Bool` in attributes. The new pass:

- Promotes inline args to standalone `const` ops with `Value::Reference`
- Remaps reduce axes from `-1` (original hidden dim) to `1` (ANE
  channel dim in `[1,C,1,S]` layout)
- Fixes downstream output types in the norm chain (reduce→add→pow
  all get `[1,1,1,S]` instead of `[1,C,1,S]`)

### 2. AneLayoutPass fixes

- Removed attribute reshaping — `axes`, `keep_dims` etc. are metadata
  parameters, not activation tensors. Reshaping them corrupts semantics.
- Skip reshaping integer tensor inputs (`Value::Tensor` with int dtype)
  — these are parameter metadata (axes vectors, shape arrays).

### 3. ONNX SimplifiedLayerNormalization → decomposed RMSNorm

Changed from `sqrt` → `reciprocal` (ANE-unsupported) to the
[Orion](https://github.com/mechramc/Orion)-verified pattern:

```
mul(x, x) → reduce_mean(axes=[1]) → add(eps) → pow(-0.5) → mul(x, rrms) → mul(weight)
```

Key: uses `axis=1` (channel dim in ANE layout), `pow(x, -0.5)` (eval-
verified, max_err=0.0004), and `reduce_mean` with const ref args.

### 4. MIL text emitter fixes

- Integer const tensors (axes, shapes) emitted **inline** instead of
  BLOBFILE — ANE expects inline for small parameter tensors.
- Reduce op output type inference: collapses the reduced axis dimension
  to 1 (was incorrectly copying the input shape).

## Current Status

| Sub-program | Status | Notes |
|---|---|---|
| `pre_attn` (all 29 layers) | ✅ Compiles | Structural split + RMSNorm fixes |
| `post_attn` (all 29 layers) | ✅ Compiles | matmul→conv + SiLU fusion |
| E2E inference | ❌ Eval fails | Runtime tensor buffer error — see next steps |

### Completed — post_attn ANE Compilation

The `post_attn` sub-program now compiles successfully on ANE:

1. **`matmul` → `conv` conversion** ✅: `AneMatmulToConvPass` converts
   `matmul(x, weight_const)` to 1×1 `conv(x, weight)` with weight
   transposition from `[Cin, Cout]` → `[Cout, Cin, 1, 1]`. Runs after
   the attention split (so structural splitter still finds matmul ops).
   **File:** `crates/mil-rs/src/ir/passes/ane_matmul_to_conv.rs`

2. **`sigmoid` + `mul` → `silu` fusion** ✅: Added to `OpSubstitutionPass`.
   Detects `mul(a, sigmoid(a))` in either input ordering and replaces
   with `silu(a)`. Both `sigmoid` (max_err=0.003) and `silu`
   (max_err=0.015) are eval-verified on ANE — fusion is an optimization,
   not a correctness requirement.
   **File:** `crates/mil-rs/src/ir/passes/op_substitute.rs`

### Next Steps — Runtime Inference

All 29 layers compile (5.99s), but `ANEProgramProcessRequestDirect()`
fails at eval time with `status=0x1d : statusType=0x9: Program Inference
error`. This is a **pre-existing** runtime error — identical with and
without the matmul→conv/SiLU changes (verified by reverting). The E5
binary loads but fails when processing the input request.

**Reproduce:** `cargo run -p ironmill-ane --example turboquant_e2e_bench --release`

**Where the error occurs:** `inference.rs` `decode()` method (line ~362
or ~446). The error propagates from `AneRuntime::eval()` in
`runtime.rs` (~line 340). The error message comes from Apple's
`_ANEInMemoryModel evaluateWithQoS:options:request:error:`.

**Control flow in decode():**
1. `layer.pre_attn.input_tensors[0].write_f16(&hidden)` — write input
2. `runtime.eval(&layer.pre_attn.loaded, ...)` — run pre_attn on ANE
3. Read Q/K/V from pre_attn outputs
4. Attention — in FP16 baseline, `fp16_attn` is `None` so Q is
   passed through as a **no-op** (line ~424: `q_data` returned directly)
5. `post_attn.input_tensors[0].write_f16(&attn_out_data)` — write
   attention output (which is actually just Q)
6. `runtime.eval(&post_attn.loaded, ...)` — run post_attn on ANE ← likely fails here

**The error does NOT tell us which eval call failed.** First debugging
step is to add error context identifying which layer/sub-program
fails (wrap each `runtime.eval()` in `decode()` with `.map_err()`
that adds the layer index and sub-program name).

**Key files:**
- `crates/ironmill-ane/src/inference.rs` — `decode()` (~line 345),
  `compile_and_load_sub()` (~line 540), I/O tensor preallocation
- `crates/ironmill-ane/src/runtime.rs` — `AneRuntime::eval()` (~line 240)
- `crates/ironmill-ane/src/tensor.rs` — `AneTensor`, `uniform_alloc_size`
- `crates/ironmill-ane/src/split.rs` — `SubProgram` I/O shape metadata

**Likely root causes (ordered by probability):**

1. **FP16 attention path is a no-op**: When `fp16_attn` is `None`,
   `decode()` returns raw Q data as the "attention output" (line ~424).
   Q shape is `[1, 2048, 1, 1]` (n_heads * head_dim = 32 * 64), but
   post_attn's first input (`a_input1` in the MIL) expects the
   concatenated attention output which should also be `[1, 2048, 1, 1]`.
   However, the pre_attn output tensor for Q may be allocated with
   different `alloc_size` than post_attn's input tensor, causing
   the IOSurface size check in the ANE framework to fail.

2. **Uniform alloc size mismatch**: `compile_and_load_sub()` computes
   `uniform_alloc_size()` separately for each sub-program. The ANE
   may require all IOSurfaces in a single eval call to match the
   exact buffer sizes the compiled model expects. Inspect the pre_attn
   output shapes vs post_attn input shapes — if they differ, their
   `uniform_alloc_size()` will differ, and the data copied from
   pre_attn output to post_attn input will be in a differently-sized
   IOSurface.

3. **Input count mismatch**: post_attn MIL has 2 inputs (`a_input0`
   = residual, `a_input1` = attention output), but `decode()` only
   fills `input_tensors[1]` conditionally (when `num_pre_outputs > 3`
   at line ~437). If this condition is false, post_attn gets only 1
   written input but the compiled model expects 2.

**Debugging approach:**
1. Add layer/sub-program identification to eval errors
2. Log the shape and alloc_size of every input/output tensor at eval
   time vs what the compiled MIL function signature declares
3. Try running a single-layer model (fewer layers = fewer compiles,
   faster iteration)
4. Compare against the working `e2e_compile_and_load_conv_linear` test
   in `lib.rs` which successfully compiles AND evaluates a conv program

## Implementation Tasks

### Task 1 — Build op dependency graph ✅

`OpGraph` struct with `forward`/`backward` adjacency lists built from
`Value::Reference` edges. Provides `walk_forward()` and `walk_backward()`
for transitive graph traversal.

**File:** `crates/ironmill-ane/src/split.rs`

### Task 2 — Identify anchor ops structurally ✅

- **Q/K/V projections**: grouped by shared non-const activation input
  (earliest group of 2+ matmul/linear ops sharing the same norm output).
- **O projection**: first matmul/linear with const weight reachable
  forward from softmax.
- **Softmax** as the attention core marker.

**File:** `crates/ironmill-ane/src/split.rs`

### Task 3 — Implement graph-based split ✅

Pre-attn = backward from projections' non-const inputs. Post-attn =
O-proj + forward + associated consts. Everything else = attention
cluster (stripped). Falls back to legacy name heuristic on failure.

**File:** `crates/ironmill-ane/src/split.rs`

### Task 4 — Tests for the structural split ✅

Five tests: standard GQA, Qwen3-like per-head norms, no-attention
fallback, layer-0 edge case, OpGraph unit test. All pass.

**File:** `crates/ironmill-ane/src/split.rs` (test module)

### Task 5 — ANE compilation fixes ✅

`AneArgPromotionPass`, `AneLayoutPass` metadata fixes, RMSNorm →
Orion-style decomposition, MIL emitter int tensor inline emission.
pre_attn compiles for all 29 Qwen3-0.6B layers.

**Files:** `crates/mil-rs/src/ir/passes/ane_arg_promotion.rs` (new),
`crates/mil-rs/src/ir/passes/ane_layout.rs`,
`crates/mil-rs/src/convert/onnx_to_mil.rs`,
`crates/mil-rs/src/convert/ir_to_mil_text.rs`

### Task 6 — E2E validation with AneInference::compile() 🔶

Both pre_attn and post_attn compile for all 29 layers. matmul→conv
(`AneMatmulToConvPass`) and SiLU fusion (`OpSubstitutionPass`) are
complete. Runtime inference fails with `ANEProgramProcessRequestDirect()
status=0x1d` — see "Next Steps — Runtime Inference" above.

**File:** `crates/ironmill-ane/src/inference.rs`

## Files

| File | Role |
|---|---|
| `crates/ironmill-ane/src/split.rs` | `split_at_attention_boundary()` — structural split with `OpGraph`. Called by `split_for_ane()`. |
| `crates/ironmill-ane/src/inference.rs` | `AneInference::compile()` → pass pipeline + `split_for_ane()` with `split_attention: true` |
| `crates/mil-rs/src/ir/passes/ane_arg_promotion.rs` | `AneArgPromotionPass` — promotes inline reduce/norm args to const refs, remaps axes |
| `crates/mil-rs/src/ir/passes/ane_layout.rs` | `AneLayoutPass` — reshapes tensors to `[1, C, 1, S]`. Fixed to skip metadata values. |
| `crates/mil-rs/src/convert/onnx_to_mil.rs` | ONNX → MIL converter. `SimplifiedLayerNormalization` → decomposed RMSNorm with axis=1. |
| `crates/mil-rs/src/convert/ir_to_mil_text.rs` | MIL text emitter. Int tensors inline, reduce output type inference. |
| `crates/mil-rs/src/ir/passes/op_substitute.rs` | `OpSubstitutionPass` — gelu substitution + rsqrt pattern detection. |

## References

- [ANE MIL Emitter Compatibility](ane-mil-emitter-compat.md) — predecessor investigation (resolved; `emit_const_op` fix in `2c9c10a`)
- [TurboQuant E2E Inference](turboquant-e2e-inference.md)
- [Orion project](https://github.com/mechramc/Orion) — reference ANE implementation; `orion_mil_rmsnorm` pattern used for RMSNorm decomposition
- [ANE Op Support Matrix](../research/ane-op-support-matrix.md) — empirically verified op support table
