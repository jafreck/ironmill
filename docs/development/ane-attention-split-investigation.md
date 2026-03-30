# ANE Attention-Boundary Split — Structural Split Implementation Plan

> **Status:** In Progress
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

## Approach — Structural Split

Replace the name-based heuristic with a **data-flow graph traversal**.
Instead of scanning op names, trace the computation graph to identify
structural boundaries:

- **Pre-attn:** all ops reachable backward from the Q/K/V projection
  matmul *inputs* — input norm, embedding lookups, and their weight
  const ops. These are ops that feed *into* the projections.
- **Post-attn:** all ops reachable forward from the attention output
  concat/reshape — O projection, residual add, FFN, and their weight
  const ops.
- **Attention cluster:** everything in between — the projections
  themselves, RoPE, per-head reshapes/norms, QK^T matmul, softmax,
  AV matmul, and the output reshape/transpose.

The key insight: the Q/K/V projection matmuls are the *first* ops in
the attention cluster (not pre-attn), and the O projection matmul is
the *first* op in post-attn. This draws a clean structural boundary
without relying on naming conventions.

## Implementation Tasks

### Task 1 — Build op dependency graph

Add a helper to `split.rs` that builds a directed graph from the
flat `&[Operation]` list. Each op's output names map to edges
connecting to ops that reference those outputs.

```rust
struct OpGraph {
    /// op index → set of op indices that consume this op's outputs
    forward: Vec<HashSet<usize>>,
    /// op index → set of op indices whose outputs this op consumes
    backward: Vec<HashSet<usize>>,
}
```

**File:** `crates/ironmill-ane/src/split.rs`

### Task 2 — Identify anchor ops structurally

Find the Q/K/V projection matmuls and the O projection matmul by
structure rather than name:

- **Q/K/V projections:** matmul ops whose one input is a weight
  const and the other traces back (transitively) to a layer norm op.
  There should be exactly 3 such matmuls at the start of attention
  (3 for GQA/MHA with separate Q, K, V projections; 2 for MQA with
  shared KV). For GQA models, the Q projection has a larger output
  dim than K/V projections.
- **O projection:** the matmul op whose input traces back to an
  attention-pattern op (softmax consumer's output → reshape →
  transpose → matmul) and whose output feeds into a residual add.
- **Attention core:** ops containing softmax, or matmuls between
  non-const tensors (QK^T and AV), which distinguish attention
  matmuls from projection matmuls (which always have a const weight).

Fall back to the existing name heuristic if structural detection
fails, logging a warning.

**File:** `crates/ironmill-ane/src/split.rs`

### Task 3 — Implement graph-based split

Replace the body of `split_at_attention_boundary()` with graph
traversal:

1. Build `OpGraph` from the op list.
2. Find anchor ops (Task 2).
3. Walk backward from Q/K/V projection inputs to collect pre-attn ops.
4. Walk forward from O projection output to collect post-attn ops.
5. Everything else is the attention cluster (stripped).

The existing function signature `(Vec<Operation>, Vec<Operation>)`
stays unchanged — callers (`split_for_ane`) are unaffected.

**File:** `crates/ironmill-ane/src/split.rs`

### Task 4 — Tests for the structural split

Add tests to `split.rs` covering:

- **Qwen3-like layer** (per-head K/V norms between projection and
  attention) — verifies that norm reshapes are correctly classified
  as attention cluster ops, not pre-attn.
- **Standard GQA layer** (no per-head norms) — regression test for
  the common case.
- **Layer with no attention pattern** — falls back to putting
  everything in pre-attn (existing behavior preserved).
- **Layer 0 edge case** — previously produced an empty pre-attn;
  the structural split should correctly identify pre-attn ops.

**File:** `crates/ironmill-ane/src/split.rs` (test module)

### Task 5 — BLOBFILE verification for large weights

Once the split produces valid sub-programs, verify BLOBFILE
compilation with transformer-sized weights (`[1, 1024, 1, 1024]` =
2 MB at fp16). The original BLOBFILE investigation concluded that references
fail, but that was before the `emit_const_op` attribute lookup bug
was fixed (`2c9c10a`). The `AneModel` E2E tests already use BLOBFILE
successfully with small 4D weights.

If BLOBFILE fails for large weights, fall back to emitting weights
as function inputs (IOSurface parameters), matching TurboQuant's
approach.

**File:** `crates/ironmill-ane/tests/` (new integration test)

### Task 6 — E2E validation with AneInference::compile()

Run the full `AneInference::compile()` pipeline on a Qwen3 model
(or equivalent with per-head norms) and verify:

- All layers produce both `pre_attn` and `post_attn` sub-programs
  (including layer 0).
- No `ANECCompile() FAILED` errors from invalid reshapes.
- Sub-program inputs/outputs have correct tensor types at boundaries.

**File:** `crates/ironmill-ane/src/inference.rs` (call chain: `compile()` → `split_for_ane()` → `split_at_attention_boundary()`)

## Files

| File | Role |
|---|---|
| `crates/ironmill-ane/src/split.rs` | `split_at_attention_boundary()` — primary change target. Called by `split_for_ane()`, which is the public entry point used by `AneInference::compile()`. |
| `crates/ironmill-ane/src/inference.rs` | `AneInference::compile()` → calls `split_for_ane()` with `split_attention: true` |
| `crates/mil-rs/src/ir/passes/ane_layout.rs` | `AneLayoutPass` — reshapes tensors to `[1, C, 1, S]`. Runs *before* the split. Interacts with reshape validity. |

## References

- [ANE MIL Emitter Compatibility](ane-mil-emitter-compat.md) — predecessor investigation (resolved; `emit_const_op` fix in `2c9c10a`)
- [TurboQuant E2E Inference](turboquant-e2e-inference.md)
- [Orion project](https://github.com/mechramc/Orion) — reference ANE implementation; uses reshape for attention layout transforms only
