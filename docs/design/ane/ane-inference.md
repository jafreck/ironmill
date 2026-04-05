# ANE Inference — Design & Status

> Living design doc for the ANE direct inference pipeline.
>
> Last updated: 2026-03-31

## Architecture

Each transformer layer runs as three ANE dispatches:

```
pre_attn (~15 ops, ~8 MB weights):
  input_norm → Q/K/V projections

  ↓ CPU: write K/V to cache, RoPE rotation

attention (~15 ops, 0 weights):
  Q × K_cache^T → scale → softmax → × V_cache → output
  (hand-written MIL with correct 4D per-head shapes)

  ↓ CPU: (pass-through)

post_attn (~25 ops, ~16 MB weights):
  O-proj → residual add → post_norm → FFN (gate/up → SiLU → down)
```

### Why this architecture

- **ANE is optimized for deep graphs** — single-op dispatches waste ~70%
  capacity. Each sub-program has 15-25+ ops.
  ([source](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615))
- **KV cache writes require a CPU boundary** — ANE programs are pure
  computation with no side effects.
- **Weight budget** — each sub-program stays under ~32 MB SRAM.
  Attention has 0 weights (pure data-dependent computation).
- **AneLayoutPass is selective** — only transforms conv/linear ops to
  `[1,C,1,S]`. Attention ops keep their natural `[1,H,D,S]` multi-head
  shapes. This prevents the layout pass from corrupting attention reshapes.

## Current Status

### What Works
- **All 28 layers compile and execute on ANE** ✅
- 54 sub-programs compiled (pre_attn ×28 + post_attn ×28) via donor/patch
- Hand-written attention MIL compiles (shared across layers)
- 8.0 tok/s throughput on Qwen3-0.6B
- CPU embedding lookup, final RMSNorm, chunked lm_head
- Perplexity benchmark runs end-to-end with no runtime errors
- Separate compile/load lifecycle (Orion-style loadWithQoS/unloadWithQoS)

### What Doesn't Work
- **PPL = inf** — the Q/K/V data flow between ANE-layout pre_attn outputs
  (doubled channels, S=32 padded) and the hand-written attention inputs
  (real model dimensions) is incorrect. The truncation from 2048→1024
  channels discards data from ANE's interleaved layout rather than
  properly de-interleaving it.

### Remaining Blocker

The pre_attn outputs Q/K/V in ANE layout `[1, 2048, 1, 32]` where
channels are doubled by `AneLayoutPass`. The hand-written attention
expects `[1, 1024, 1, 32]` (real model dimensions). Simply truncating
the first 1024 elements produces garbage because ANE layout interleaves
data across the channel dimension.

**Fix needed:** Properly map ANE-layout output data to the attention
input format. Options:
1. De-interleave on CPU (reverse AneLayoutPass transform on the data)
2. Make the attention MIL accept ANE-layout inputs (adapt the reshapes)
3. Skip AneLayoutPass on pre_attn outputs that feed attention

## Bugs Fixed (this session)

| # | Bug | Fix | Impact |
|---|---|---|---|
| 1 | Missing final RMSNorm before lm_head | `cpu_rms_norm()` + `extract_1d_weight()` | CE 1548 → 908 |
| 2 | Q/K/V output ordering [K,Q,V] assumed [Q,K,V] | Read by lexicographic order | Data corruption |
| 3 | Arch detection on post-pass program (doubled dims) | Detect from original program + lm_head shape | Wrong head counts |
| 4 | `strip_gather_ops` whitelist dropped verified ops | Narrowed to gather+split+concat+sub | Broken attention graphs |
| 5 | `AneVariableNamingPass` pre-split caused `z_output*` collisions | Moved to per-subprogram | Invalid MIL |
| 6 | Dangling 20 MB RoPE cache function inputs | `prune_unreferenced_inputs()` + DCE | Compilation failure |
| 7 | `TypeRepropagationPass` missing tile shape inference | Added `infer_tile_output()` | Shape mismatches |
| 8 | Output name deduplication in `build_sub_program` | Collision detection + rename | Invalid MIL |
| 9 | `AneLayoutPass` flattened attention 4D shapes | Selective: only conv/linear + function I/O | Attention compilation |
| 10 | Tensor allocation used pre-layout shapes | Derive from post-layout processed program | 0x1d runtime errors |
| 11 | S≥32 padding missing in per-subprogram compile | `apply_min_seq_padding()` shared helper | 0x1d runtime errors |
| 12 | Attention merged into post_attn exceeded weight budget | 3-way split: separate attention sub-program | 22 MB > compile limit |

## Key Insights

### ANE Performance Architecture
- 19 TFLOPS FP16 true peak, 6.6 TFLOPS/W efficiency
- Deep graphs (16-64 ops) reach 94% utilization
- Conv 1×1 is ~3× faster than matmul for same FLOPs
- ~32 MB on-chip SRAM; exceeding it drops throughput ~30%
- INT8 saves bandwidth only, not compute (dequantized to FP16)
- 0.095ms dispatch overhead per eval

### AneLayoutPass Must Be Selective
The ANE conv engine requires `[1,C,1,S]` layout, but attention matmuls
need `[1,H,D,S]` multi-head format. Applying the layout pass blanket
corrupts attention reshapes `[1,16,64,S]` → `[1,16,1,64*S]`. The fix:
only transform conv/linear ops and function I/O, propagating through
elementwise/reduction ops that connect them.

### Attention Has Zero Weights
The attention computation (Q×K→softmax→×V) is purely data-dependent.
All weights are in the projections (Q/K/V/O in pre_attn/post_attn).
This makes attention naturally safe for the ANE weight budget — its
sub-program has 0 BLOBFILE entries.

## References
- [ANE Op Support Matrix](ane-op-support-matrix.md) — 74 verified ops
- [ANE Constraints](ane-constraints.md) — hardware limits and diagnostics
- [TurboQuant Design](turboquant.md) — INT8 KV cache compression
- [Inside the M4 ANE](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) — maderix benchmarks
