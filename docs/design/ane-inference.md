# ANE Inference — Design & Status

> Living design doc for the ANE direct inference pipeline.
>
> Last updated: 2026-03-31

## Architecture

Each transformer layer is split into two deep ANE sub-programs:

```
pre_attn (~15 ops, ~8 MB weights):
  input_norm → Q/K/V projections

  ↓ CPU: write K/V to cache, gather RoPE cos/sin

post_attn (~50 ops, ~22 MB weights):
  attention(Q, K_cache, V_cache) → O-proj → residual → post_norm → FFN
```

This 2-way split maximizes ANE graph depth (the ANE achieves 94% utilization
at 32+ ops per program) while enabling KV cache management on CPU between
the two dispatches.

### Why this split

- **ANE is optimized for deep graphs** — chaining 16-64 ops per program.
  Single-op dispatches waste ~70% capacity due to 0.095ms overhead.
  ([source](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615))
- **KV cache writes require a CPU boundary** — ANE programs are pure
  computation with no side effects. The cache must be updated between
  "produce K/V" and "consume KV cache."
- **Weight budget** — a full layer has ~30 MB of projection weights.
  Splitting at the attention boundary puts ~8 MB in pre_attn and ~22 MB
  in post_attn, each under the ~32 MB SRAM limit.

## Current Status

### What Works
- pre_attn sub-programs compile and run on ANE ✅
- Decode loop with per-layer execution
- CPU embedding lookup, CPU RoPE rotation
- Token sampling with temperature, greedy, and EOS detection
- Chunked lm_head on ANE
- Final RMSNorm applied before lm_head (CPU)
- Separate compile/load lifecycle (Orion-style loadWithQoS/unloadWithQoS)
- Perplexity benchmark (--perplexity flag) runs end-to-end

### What Doesn't Work
- **post_attn sub-programs fail to compile on ANE.** The MIL is valid
  (no duplicate names, no dangling references, correct shapes, 22 MB
  weights) but ANECCompile() rejects it. This is an ironmill bug,
  not a fundamental ANE limitation — see investigation notes below.
- Without post_attn, layers fall back to Q pass-through → PPL=inf

## Bugs Fixed During Investigation

| # | Bug | Fix |
|---|---|---|
| 1 | Missing final RMSNorm before lm_head | Added cpu_rms_norm() + extract_1d_weight() |
| 2 | Q/K/V output ordering assumed [Q,K,V] but split sorts lexicographically [K,Q,V] | Read index 1 as Q, 0 as K |
| 3 | Architecture detection ran on post-pass program (doubled dims) | Detect from original program |
| 4 | strip_gather_ops whitelist dropped verified ops (layer_norm, reduce_mean, etc.) | Narrowed to gather+split+concat+sub only |
| 5 | AneVariableNamingPass ran pre-split, creating z_output* name collisions | Moved to per-subprogram in compile_and_load_sub |
| 6 | Dangling function inputs after gather/split removal (20+ MB RoPE caches) | prune_unreferenced_inputs() + DCE |
| 7 | TypeRepropagationPass didn't handle tile output shapes | Added infer_tile_output + manual fix for Reference-valued reps |
| 8 | Output name deduplication missing in build_sub_program | Added collision detection + rename |

## Investigation: post_attn Compilation Failure

### What we know
- The emitted MIL is structurally valid (verified by script):
  - No duplicate variable definitions
  - No dangling references
  - All shapes consistent with S=32 padding
- 22 MB weights (5 conv projections: O + gate + up + down + norm)
- 4 function inputs, all ≤128 KB
- ~55 non-const ops including attention core + FFN
- pre_attn (same architecture, same passes, ~8 MB weights) compiles fine

### What we don't know
- **Why** ANECCompile rejects this specific program
- Whether it's the total weight size (22 MB is under 32 MB but close)
- Whether a specific op pattern in the combined attention+FFN triggers
  a compiler bug
- Whether splitting post_attn into attention + FFN (3-way) would compile

### Recommended next steps
1. **Bisect**: compile just the attention half of post_attn (no FFN) and
   just the FFN half (no attention) to isolate which part fails
2. **Reduce weights**: try with smaller intermediate_size to get under 16 MB
3. **Compare with maderix**: check if maderix's 64-op deep graphs include
   similar op patterns (matmul + softmax + conv + silu)

The failing MIL is dumped to /tmp/ironmill_debug_layer_0_post_attn.mil.

## References
- [ANE Op Support Matrix](ane-op-support-matrix.md) — 74 verified ops
- [ANE Constraints](ane-constraints.md) — hardware limits and diagnostics
- [TurboQuant Design](turboquant.md) — INT8 KV cache compression
