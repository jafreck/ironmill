# ANE Attention-Boundary Split Investigation

> **Status:** Open
>
> **Blocks:** [TurboQuant E2E Inference](turboquant-e2e-inference.md) — layer sub-program compilation
>
> **Discovered during:** MIL emitter investigation → E2E inference bringup

## Problem

The attention-boundary split in `split_at_attention_boundary()` produces
sub-programs with **invalid reshape operations** — the reshape target
shape doesn't match the input element count.

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

The attention-boundary split uses **op name heuristics** to identify
the attention cluster. Ops between the first RoPE/reshape op and the
last attention op are stripped. But the model's internal reshapes
(like the K-norm reshape from `[1, 1024]` to `[1, 8, 128]`) may
share naming patterns with the attention reshape ops, causing them
to be incorrectly classified as part of the attention cluster and
stripped from the pre_attn sub-program.

Additionally, the Qwen3 model has per-head K/V norms
(SimplifiedLayerNormalization) that sit between the projection and
the attention. These require reshape → norm → reshape, and the
boundary between "projection ops" and "attention ops" is not cleanly
separable by name alone.

## Observations

1. **Layer 0 has no pre_attn sub-program** — all its ops were
   classified as attention cluster ops and stripped entirely.

2. **Layers 1–28 have pre_attn sub-programs** with only the K
   projection path (missing Q and V projections, which were stripped
   as attention ops).

3. The heuristic detects ops with names containing `_q_reshape`,
   `_k_reshape`, `_v_reshape` as attention cluster ops. But in Qwen3,
   the K-norm reshape is named `_attn_k_norm_Reshape_1` — the `_k_`
   prefix causes it to be caught by the `_k_reshape` pattern even
   though it's a normalization reshape, not an attention reshape.

## Approach

The name-based heuristic is fundamentally fragile for models with
per-head operations between projections and attention. Two options:

### Option A — Structural split (preferred)

Instead of splitting by op names, split by **data flow**:
- Pre-attn: all ops reachable backward from the Q/K/V projection
  outputs (norm, linear/matmul weight const ops).
- Post-attn: all ops reachable forward from the attention output
  (O projection, residual, FFN).
- Attention cluster: everything in between.

This requires tracing the computation graph, not just scanning names.

### Option B — Tighter name patterns

Make the attention cluster detection more specific:
- Only match exact GQA lowered patterns (`GroupQueryAttention_q_reshape_op`)
- Don't match model-internal reshapes (`_norm_Reshape`)
- Require the full GQA suffix, not substrings

This is less robust but simpler to implement.

### Option C — Skip the split for now

Run the entire layer as a single ANE program (no attention split).
This means the FP16 attention runs on ANE as part of the layer, and
TurboQuant can't slot in. Useful as a baseline to verify ANE
compilation works at all for these layers before attempting the split.

## BLOBFILE Verification Note

Once the split produces valid sub-programs, BLOBFILE compilation
must be verified independently. The original BLOBFILE investigation
concluded that BLOBFILE references fail, but that conclusion was
reached before the `emit_const_op` attribute lookup bug was fixed
(weights were silently empty). The existing `AneModel` E2E tests
use BLOBFILE successfully with small 4D weights. Whether it works
for transformer-sized weights (`[1, 1024, 1, 1024]` = 2 MB) has
not been tested.

If BLOBFILE fails for large weights, the fallback is to emit weights
as function inputs (IOSurface parameters), matching TurboQuant's
approach.

## Files

| File | Role |
|---|---|
| `crates/ironmill-ane/src/split.rs` | `split_at_attention_boundary()` — fix target |
| `crates/ironmill-ane/src/inference.rs` | `AneInference::compile()` — layer compilation loop |
| `crates/mil-rs/src/ir/passes/ane_layout.rs` | Shape transforms that interact with reshape validity |

## References

- [ANE MIL Emitter Compatibility](ane-mil-emitter-compat.md) — predecessor (resolved)
- [TurboQuant E2E Inference](turboquant-e2e-inference.md)
- [Orion project](https://github.com/mechramc/Orion) — reference ANE implementation; uses reshape for attention layout transforms only
