# ANE Attention-Boundary Split — Structural Split Implementation Plan

> **Status:** Resolved — split works, runtime eval fixed, e2e inference running
>
> **Was blocking:** [TurboQuant E2E Inference](turboquant-e2e-inference.md) ✅ now running
>
> **Discovered during:** MIL emitter investigation → E2E inference bringup
>
> **Key fix:** ANE rejects IOSurface I/O tensors `[1, C, 1, S]` when
> C > ~768 and S < 32. Padding dim 3 to S=32 after AneLayoutPass
> resolved the 0x1d eval error. See "Next Steps — Runtime Inference"
> below for full root cause analysis.
>
> **Remaining:** Structural split falls back to name heuristic (softmax
> not found). FP16 attention sub-programs not yet compiled. Tracked in
> [ANE Inference Optimizations](ane-inference-optimizations.md).

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
without the matmul→conv/SiLU changes (verified by reverting).

**Reproduce:** `cargo run -p ironmill-ane --example turboquant_e2e_bench --release`

#### Root cause: ANE minimum I/O tensor size constraint

**Confirmed via isolation testing.** Even a trivial `mul(x,x)` program
fails at eval time when the function I/O tensor shape is `[1, C, 1, S]`
with C > ~768 and S < 32. The ANE compiler accepts these shapes, but
the hardware rejects them at eval. Empirical boundary:

- `C=768, S=1`: ✅ works
- `C=800, S=1`: ❌ fails (0x1d)
- `C=1024, S=1`: ❌ fails
- `C=1024, S=32`: ✅ works
- `C=768, S=33`: ❌ fails

This is NOT about IOSurface allocation size (tested with exact-fit and
padded — both same result). It's an ANE hardware constraint on how
function I/O buffers map to the ANE's on-chip memory.

**Evidence from external projects:**

- **Orion** (mechramc/Orion): Uses GPT-2 124M with `d_model=768` — below
  the threshold. Tests use shapes like `[1, 32, 1, 32]`.
- **maderix/ANE**: Documents "multi-input ANE requests cause 0x1d error"
  and packs ALL data (activations + weights) into a **single spatial
  input** tensor. For Qwen3-0.6B (DIM=1024, SEQ=256), the input is
  `[1, 1024, 1, 4352]` — always large spatial. They also use only
  1 input and 1 output per kernel.

**Fix (in progress):** After all passes (including AneLayoutPass which
converts to `[1,C,1,S]` format), pad dim 3 (S) to at least 32 for
all sub-program function I/O tensors. The MIL compiles with S=32
shapes; at decode time, write one token's data into column 0 and read
from column 0. The padding columns contain zeros which don't affect
correctness for elementwise or channel-reduction ops.

**Additional constraint — single input per eval call:** maderix/ANE
reports that multi-input ANE requests cause 0x1d error. Ironmill's
`post_attn` sub-programs have 2 inputs (attention output + residual).
After fixing the shape constraint, this may be the next blocker. The
workaround is to pack multiple inputs into the spatial dimension of
a single input tensor (matching how maderix and Orion handle this).

#### Debugging progress

1. ✅ Added `.map_err()` context → **layer 0 pre_attn** is the first
   eval that fails
2. ✅ Isolated to the MIL program itself — fails even when compiled and
   evaluated in complete isolation (no other programs loaded)
3. ✅ Narrowed to shape constraint: `[1, 1024, 1, 1]` fails, not op-
   specific (even `mul(x,x)` fails at this shape)
4. ✅ Padded S from 1 to 32 after AneLayoutPass in `inference.rs` —
   sub-programs compile and eval with `[1, 1024, 1, 32]` shapes
5. ✅ Updated `decode()` with scatter-write / gather-read helpers
   (`write_f16_padded`, `read_f16_channels`) for NCHW column-0 I/O
6. ✅ **FP16 baseline inference works end-to-end** — 128 tokens
   generated at 5.9 tok/s on Qwen3-0.6B (29 layers, ANE-only decode)

#### Remaining: TurboQuant uniform alloc size

TurboQuant's `step_attention` fails with "all input tensors must have
the same allocation size". TurboQuant compiles its own MIL sub-programs
(cache-write, attention) independently of `AneInference`. These programs
have inputs with different shapes (Q tensor vs cache tensor), producing
different-sized IOSurfaces that fail `validate_uniform_alloc` in
`runtime.eval()`. Fix: apply the same S-padding and uniform alloc logic
to TurboQuant's MIL programs in `turboquant.rs` / `turboquant_mil.rs`.
Note: the multi-input constraint from maderix/ANE may also apply here.

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
