# Metal → ANE TurboQuant Adapter

> **Status:** Feasibility assessment
> **Scope:** Unifying TurboQuant across GPU (Metal), MLX, and ANE backends
> **Related:** [turboquant.md](turboquant.md), [ane-constraints.md](ane-constraints.md), [mlx-backend.md](mlx-backend.md)

## Problem

ironmill-inference maintains three independent TurboQuant implementations:

| Backend | Location | Execution model |
|---------|----------|-----------------|
| GPU (Metal) | `src/gpu/turboquant/` + `src/gpu/shaders/turboquant.metal` | Explicit Metal compute shaders |
| MLX | `src/mlx/turboquant/` | Custom Metal kernels via `mlx-c` |
| ANE | `src/ane/turboquant/` | Compiled MIL graphs via private ANE API |

The Metal and MLX backends share the same algorithms (MLX dispatches the same shader source through `metal_kernel()`). The ANE backend has **diverged algorithmically** from Metal, which is the source of truth for TurboQuant. This creates correctness risk and maintenance burden.

## ANE Execution Model

ANE does not execute Metal shaders. It accepts MIL (Machine Learning Intermediate Language) graphs compiled through Apple's private `_ANEInMemoryModel` API, backed by IOSurface tensors. The available op set is limited to standard tensor operations: `matmul`, `reshape`, `cast`, `mul`, `add`, `round`, `clip`, `softmax`, `slice_by_index`, `greater`, `select`, `tile`.

**Implications:** Any adapter must translate Metal's *algorithms* into MIL ops, not run Metal *code*.

## Divergence Inventory

| Aspect | GPU (Metal) | ANE (MIL) | Severity |
|--------|-------------|-----------|----------|
| Quantizer math | Lloyd-Max Gaussian codebooks (`turboquant/codebook.rs`) | Beta-optimal levels (`mil_rs::passes::beta_quantizer`) | **High** — different reconstruction quality |
| Cache bit-width | INT4 packed nibbles + INT8 | INT8 only | **High** — 2× cache memory on ANE |
| K/V asymmetry | K uses (b−1)-bit + QJL, V uses b-bit | Same scheme for both K and V | **Medium** |
| QJL correction | Random projection: `√(π/2)/d · ⟨Sq, sign(e)⟩` | Sign-based: `matmul(q_sign^T, residual)/√d` | **High** — different correction formula |
| Outlier channels | Full mixed-precision split with per-group codebooks | Not implemented | **Medium** |
| Cache-write | Codebook search → pack → per-position scales + QJL norms | Scalar round/clip/cast to INT8 | **High** — no codebook lookup |
| Attention dequant | Inline codebook dequant with scale/norm buffers | `cast(int8→fp16) × deq_scale` | **Medium** |

### Shared code today

The `turboquant/` module provides backend-agnostic math:

- `codebook.rs` — Lloyd-Max Gaussian codebook generation
- `rotation.rs` — Seeded Hadamard rotation signs + QJL matrix generation
- `outlier.rs` — Outlier-channel detection

GPU and MLX use these directly. ANE does **not** — it uses `mil_rs` passes (`beta_quantizer`, `rotation`) instead, which is the root of the algorithmic divergence.

## Adapter Approaches

### Approach A: Run Metal shaders on ANE

**Not feasible.** ANE is a dedicated neural engine with its own ISA. It accepts only compiled MIL graphs through a private API. No Metal shader dispatch path exists or could be created.

### Approach B: Rewrite ANE MIL emitter to match Metal's algorithms

**Partially feasible.** Most of Metal's TurboQuant math can be expressed in MIL ops:

| Metal algorithm | MIL expression | Feasible? |
|----------------|----------------|-----------|
| Lloyd-Max codebook quantization | Chain of `greater`/`select` ops (compare against boundaries, select levels) | ✅ Yes, but O(2^b) ops per level |
| K/V asymmetric bit allocation | Separate MIL programs for K and V | ✅ Yes |
| Random-projection QJL correction | `matmul` with QJL matrix + scaling | ✅ Yes |
| Per-position scale computation | `reduce_max` + `mul` | ✅ Yes |
| INT4 packed cache | Not expressible — ANE only supports INT8 | ❌ No |
| Online tiled softmax | MIL `softmax` is a single op; can't replicate tile-streaming numerics | ⚠️ Numerically close but not identical |
| Outlier channel splitting | Expressible but multiplies compile budget consumption | ⚠️ Constrained |

**Blockers:**
- INT4 is a permanent gap — ANE will always use more KV cache memory
- Codebook search as MIL compare/select chains is significantly heavier than Metal's loop, risking ANE performance regression
- ANE compile budget (~119 programs/process) limits how many additional programs outlier support would require

### Approach C: Shared algorithm spec with backend-specific emitters

**Recommended.** Define TurboQuant algorithms once, lower to backend-specific representations:

```
turboquant/          ← Shared algorithm definitions (codebook, rotation, QJL, outlier)
  ↓                     Already exists, used by GPU + MLX
gpu/turboquant/      ← Metal shader dispatch (existing)
mlx/turboquant/      ← MLX kernel dispatch (existing, mirrors Metal)
ane/turboquant/      ← MIL graph emission (rewrite to match Metal's math)
```

## Recommended Implementation Path

### Phase 1: Unify quantizer math

Rewrite `ane/turboquant/mil_emitter.rs` cache-write to use Lloyd-Max codebooks from the shared `turboquant/codebook.rs` instead of beta-optimal quantization. Express codebook lookup as MIL compare/select chains.

**Risk:** Performance regression from heavier MIL graphs. Benchmark against current beta-optimal path.

### Phase 2: Unify QJL correction

Replace ANE's sign-based QJL with the random-projection formula used by Metal. The QJL matrix is already generated in the shared module — the MIL emitter just needs to use it with the correct formula (`√(π/2)/d · ⟨Sq, sign(e)⟩`).

**Risk:** Low. The math maps cleanly to `matmul` + `mul` in MIL.

### Phase 3: Add K/V asymmetry

Emit separate cache-write programs for K (b−1 bits + QJL) and V (b bits, no QJL). This doubles the cache-write programs but aligns ANE with Metal's precision allocation.

**Risk:** Compile budget impact — each layer now needs 2 cache-write programs instead of 1.

### Phase 4: Evaluate outlier channels

Port outlier-channel detection and mixed-precision split to the MIL emitter. Each outlier/non-outlier group would need its own cache-write and attention programs.

**Risk:** High compile budget pressure. May not be viable on models with many layers. Should be evaluated empirically after phases 1–3.

## Permanent Gaps

These differences cannot be bridged and should be accepted as backend limitations:

- **INT4 cache** — Metal/MLX only. ANE stays INT8, using 2× the cache memory for the same sequence length.
- **Exact softmax numerics** — Metal uses online tiled softmax with running max/sum. MIL's `softmax` op uses ANE's internal implementation. Results will be numerically close but not bit-identical.
- **Performance tuning** — Metal allows threadgroup/tile size tuning. ANE graphs are compiled opaquely.

## Risk Summary

| Risk | Severity | Mitigation |
|------|----------|------------|
| ANE perf regression from codebook-as-MIL | Medium | Benchmark before/after; keep beta-optimal as fallback config |
| Compile budget exhaustion with K/V split + outliers | Medium | Track `remaining_budget()` at init; degrade gracefully |
| INT4 memory gap | Accepted | Document as backend limitation; INT8 still fits most deployment targets |
| Numerical divergence in softmax | Low | Validate perplexity impact with reference model |

## References

- [TurboQuant design](turboquant.md) — ANE-specific TurboQuant architecture
- [ANE constraints](ane-constraints.md) — Hardware/compiler limits
- [GPU backend](gpu-backend.md) — Metal compute architecture
- [MLX backend](mlx-backend.md) — MLX dispatch model
- `crates/ironmill-inference/src/gpu/shaders/turboquant.metal` — Metal source of truth
- `crates/ironmill-inference/src/ane/turboquant/mil_emitter.rs` — Current ANE implementation
- `crates/ironmill-inference/src/turboquant/` — Shared algorithm module
