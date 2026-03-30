# Academic Research Enhancements for Ironmill (2026)

Companion to `optimization-opportunities-2026.md`. This document catalogs
specific 2026 academic papers with concrete implementation paths as new
ironmill optimization passes or runtime features.

Ironmill currently has **34 compile-time optimization passes** and a runtime
KV-cache compression system (TurboQuant). The enhancements below are organized
by implementation category and prioritized by fit with ironmill's existing
architecture.

---

## 1. Quantization Passes (Compile-Time)

### 1a. SliderQuant — Layer-Sensitive Adaptive Bit-Width

**Paper:** SliderQuant: Accurate Post-Training Quantization for LLMs (ICLR 2026)
**Link:** <https://openreview.net/forum?id=YNqZqw4fLT>

**What it does:** Instead of applying a uniform bit-width across all layers,
SliderQuant assigns per-layer quantization granularity based on sensitivity
analysis. Shallow layers and deep layers are handled with more care. Strong
results for MoE architectures.

**How it maps to ironmill:** The existing `MixedPrecisionPass` already supports
per-layer config, but the user must specify it manually. A `SliderQuantPass`
would automate optimal bit-width assignment:

- Accept a calibration dataset (ironmill already supports `--calibration-dir`)
- Run forward passes to measure per-layer sensitivity (activation variance,
  Hessian diagonal approximation)
- Assign bit-widths that minimize total model quality loss under a memory budget
- Emit the result as a `MixedPrecisionPass` config or apply directly

**Effort:** Medium — requires calibration infrastructure but can reuse existing
per-layer quantization machinery in `pipeline.rs`.

---

### 1b. DuQuant — Dual Rotation + Permutation

**Paper:** DuQuant: Distributing Outliers via Dual Transformation
**Link:** <https://duquant.github.io/>

**What it does:** Extends rotation-based quantization (like ironmill's
PolarQuant) with an additional *permutation* step. Pure Hadamard rotation
smooths most outliers, but massive outliers concentrated in a few activation
columns persist. DuQuant adds a learned permutation that redistributes these
before rotation, achieving new SOTA for W4A4 quantization.

**How it maps to ironmill:** This is a natural extension of `PolarQuantPass` in
`crates/mil-rs/src/ir/passes/polar_quantize.rs`:

- Add an optional permutation matrix before the Hadamard rotation in
  `rotate_rows_hadamard()`
- The permutation can be precomputed from calibration data (column-wise
  activation magnitude sorting)
- `PolarRotationFusionPass` already handles rotation cancellation between
  layers — extend it to cancel permutation+rotation pairs

**Effort:** Low — the rotation infrastructure in `rotation.rs` and
`beta_quantizer.rs` already exists. The permutation is a column reorder applied
before rotation.

---

### 1c. CodeQuant — Unified Clustering + Rotation for MoE

**Paper:** CodeQuant: Unified Clustering and Quantization for Enhanced Outlier
Smoothing in Low-Precision Mixture-of-Experts
**Link:** <https://openreview.net/forum?id=ATpchFiBQi>

**What it does:** Combines learnable rotation (for activations) with adaptive
weight clustering to smooth and absorb outliers. Specifically designed for
Mixture-of-Experts architectures. 4.15× speedup with higher accuracy than
QuaRot or SmoothQuant under extreme quantization.

**How it maps to ironmill:** Complements the existing `PerExpertQuantPass` in
`crates/mil-rs/src/ir/passes/mixed_precision.rs`:

- Per-expert clustering: run k-means (ironmill already has `kmeans.rs`) on each
  expert's weights independently, producing per-expert codebooks
- Rotation: apply Hadamard rotation (reuse `rotation.rs`) to activations at
  expert boundaries
- Emit `constexpr_lut_to_dense` with expert-specific LUTs
- The `CodebookOptimizationPass` can be extended to handle per-expert codebooks

**Effort:** Medium — requires per-expert weight grouping logic but reuses
existing codebook and rotation infrastructure.

---

### 1d. LATMiX — Learnable Affine Transforms for Microscaling

**Paper:** LATMiX: Learnable Affine Transformations for Microscaling
Quantization
**Link:** <https://arxiv.org/abs/2602.17681>

**What it does:** Generalizes rotation-based quantization. Instead of a fixed
Hadamard matrix, LATMiX learns an arbitrary invertible affine transform
optimized for the target quantization format (MX / microscaling). Consistent
accuracy improvements for sub-4-bit quantization.

**How it maps to ironmill:** This would be a more advanced variant of
PolarQuant, where the transform matrix is learned rather than fixed:

- Requires a calibration/optimization step (gradient descent on the transform
  matrix) — heavier than PolarQuant's zero-shot approach
- The learned transform replaces the Hadamard in `polar_quantize.rs`
- `PolarRotationFusionPass` would need to handle arbitrary learned matrices
  instead of just Hadamard (no longer self-inverse — need explicit inverse)

**Effort:** High — requires gradient-based optimization during compilation.
Best suited as an offline tool that produces a transform matrix, which ironmill
then applies as a fixed pass.

---

### 1e. SpinOut — Outlier-Injected Rotation Training

**Paper:** SpinOut: Enhanced Rotation-Based Quantization for LLM by Outlier
Injection (2026)
**Link:** <http://esoc.hanyang.ac.kr/publications/2026/SpinOut_Enhanced_Rotation-Based_Quantization_for_LLM_by_Outlier_Injection.pdf>

**What it does:** During rotation matrix training, selectively injects
artificial outliers into "sensitive" layers. This improves the rotation's
ability to generalize and reduces quantization variance.

**How it maps to ironmill:** Could improve PolarQuant's fixed rotation seed
strategy:

- Use calibration data to identify sensitive layers
- Inject synthetic outliers during the rotation matrix search
- Produce per-layer rotation seeds or learned rotation matrices
- Apply as a preprocessing step before `PolarQuantPass`

**Effort:** Medium — primarily an offline calibration enhancement.

---

## 2. KV Cache & Decoding (Runtime / Compile-Time)

### 2a. QuantSpec — Apple's Self-Speculative Decoding

**Paper:** QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV
Cache (Apple Research, ICML 2025)
**Link:** <https://machinelearning.apple.com/research/quantspec>

**What it does:** Uses the same model architecture for both draft and verifier
(self-speculation), but with hierarchical 4-bit quantized KV cache. The draft
uses aggressively quantized KV, the verifier uses full-precision KV. >90%
draft acceptance rate, 2.5× speedup, ~1.3× memory savings over competitors.

**How it maps to ironmill:** This is the highest-ROI enhancement because it
comes from Apple Research and targets the exact same deployment stack:

- **Compile-time:** `KvCachePass` emits two KV cache variants per layer — a
  4-bit draft cache and a full-precision verifier cache. The model's CoreML
  stateful API exposes both.
- **Compile-time:** `ModelSplitPass` could emit draft/verifier `.mlpackage`
  pairs that share weights but differ in KV precision.
- **Runtime:** The inference loop alternates between draft (fast, quantized KV)
  and verify (accurate, full KV) phases, accepting or rejecting draft tokens.

**Effort:** Medium-High — requires changes to both `KvCachePass` and the
runtime inference loop. But Apple designed this for CoreML, so the API fit is
natural.

**Priority:** 🔥🔥🔥 Highest — Apple-native, designed for CoreML deployment.

---

### 2b. DapQ — Position-Aware KV Cache Eviction

**Paper:** Where Matters More Than What: Decoding-Aligned KV Cache Compression
via Position-aware Pseudo Queries (March 2026)
**Link:** <https://arxiv.org/abs/2603.11564>

**What it does:** Standard KV eviction decides which tokens to keep based on
attention scores during prefill. But prefill attention patterns don't predict
decode-time importance well. DapQ generates "pseudo queries" that simulate
future decoding positions, enabling much better eviction decisions. Achieves
near-lossless performance at 3% cache retention.

**How it maps to ironmill:** Extends `KvCachePass` with smarter eviction:

- At compile time, annotate KV cache ops with eviction metadata (window size,
  importance scoring function)
- The scoring function uses positional pseudo-queries — a small auxiliary
  computation graph emitted alongside the main attention
- At runtime, the eviction logic uses these scores to decide which KV entries
  to keep

**Effort:** Medium — the pseudo-query generation is a graph transformation,
but the eviction logic is runtime.

---

### 2c. FAFO — Draftless Speculative Decoding with Compressed KV

**Paper:** FAFO: Lossy KV Cache Compression for Lossless Inference
Acceleration via Draftless Fumble Decoding (ICLR 2026)
**Link:** <https://openreview.net/forum?id=oSk9tP5Mgs>

**What it does:** Merges speculative decoding with compressed KV caches in a
single-cache design. The model generates candidate continuations using a lossy
compressed KV cache, then verifies them losslessly in parallel. No separate
draft model needed. 1.2–2.7× speedup with nearly perfect output fidelity.

**How it maps to ironmill:** This could be a runtime enhancement for TurboQuant:

- TurboQuant already compresses KV caches at runtime (INT8 via rotation +
  quantization)
- FAFO adds a verify-then-accept loop: generate N tokens with compressed KV,
  verify all N in parallel with full-precision attention
- The compile-time component is minimal — emit the verification attention
  subgraph alongside the main model
- The runtime component manages the speculative loop

**Effort:** High — primarily a runtime architecture change. The compile-time
pass changes are small (emit verification subgraph).

---

### 2d. RotateKV — 2-Bit KV Cache Quantization

**Paper:** RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs
via Outlier-Aware Adaptive Rotations
**Link:** <https://huggingface.co/papers/2501.16383>

**What it does:** Extends rotation-based KV cache compression down to 2-bit
using per-head-group adaptive rotations. Different attention heads have
different outlier patterns, so RotateKV learns head-specific rotation matrices.

**How it maps to ironmill:** Natural extension of TurboQuant's `turbo-int8`:

- Add a `turbo-int2` mode to `crates/ironmill-ane/src/turboquant.rs`
- Replace the fixed Hadamard rotation with per-head-group rotations
- Use calibration data to determine which heads need adaptive rotations
- The MIL generation in `turboquant_mil.rs` would emit 2-bit
  quantize/dequantize ops

**Effort:** Low-Medium — TurboQuant's rotation + quantization pipeline already
exists. Main work is supporting 2-bit precision and per-head rotation matrices.

---

## 3. Pruning & Sparsity (Compile-Time)

### 3a. REAP — Router-Weighted Expert Pruning for MoE

**Paper:** REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression
(March 2026)
**Link:** <https://arxiv.org/abs/2510.13999>

**What it does:** For Mixture-of-Experts models, pruning entire experts is more
effective than merging them. REAP uses router gate values combined with expert
activation norms to rank experts. Achieves near-lossless 50% expert pruning on
models like Qwen3-Coder-480B.

**How it maps to ironmill:** A new `MoEExpertPruningPass`:

- Identify MoE router + expert subgraphs in the MIL IR
- Use router weights (available as constants in the graph) to compute expert
  importance scores
- Prune lowest-scoring experts by removing their subgraphs and adjusting router
  normalization
- Works naturally with `PerExpertQuantPass` — prune first, then quantize
  remaining experts

**Effort:** Medium — requires MoE subgraph detection (partially exists for
`PerExpertQuantPass`) and expert ranking logic.

---

### 3b. Sparse-BitNet — 1.58-Bit + N:M Structured Sparsity

**Paper:** Sparse-BitNet: 1.58-bit LLMs with N:M Sparsity (2026)
**Link:** <https://arxiv.org/abs/2603.05168>

**What it does:** Combines ultra-low-bit quantization (ternary / 1.58-bit
weights) with structured N:M sparsity (e.g., 2:4 — every group of 4 weights
has exactly 2 zeros). The structured pattern enables hardware acceleration.

**How it maps to ironmill:** CoreML does not currently expose N:M sparse
kernels, making this **speculative** for now:

- If Apple adds N:M sparsity support to CoreML/ANE, a `StructuredSparsityPass`
  could enforce N:M patterns on weight tensors
- Even without hardware support, the ternary quantization aspect could be
  implemented as a `TernaryQuantPass` — weights stored as {-1, 0, +1} with a
  per-channel scale factor, packed into 2-bit `constexpr_lut_to_dense`

**Effort:** Low (ternary quant only) to High (full N:M with hardware support).
**Status:** Speculative — blocked on CoreML N:M support.

---

## 4. Compiler Architecture

### 4a. TransFusion — Global Graph Fusion + Pipelined Scheduling

**Paper:** TransFusion: End-to-End Transformer Acceleration via Graph Fusion
and Pipelined Scheduling (MICRO 2025)
**Link:** <https://jnamaral.github.io/CDOL/papers/ZhangMICRO25.pdf>

**What it does:** Current fusion passes (including ironmill's) are *local* —
they pattern-match within a single layer (e.g., conv+bn+relu). TransFusion
introduces *global* fusion across layer boundaries with pipelined execution
scheduling. A DAG-based scheduler (DPipe) and Monte Carlo Tree Search-based
tile search (TileSeek) jointly optimize inter-layer and intra-layer fusions.
2.2× edge speedup.

**How it maps to ironmill:** This would be a significant rearchitecture of
`PassPipeline`:

- Today, ironmill's fusion passes (`op_fusion.rs`, `attention_fusion.rs`) run
  independently and locally
- TransFusion's approach would add a global scheduling pass *after* local
  fusion, reasoning about buffer reuse and execution overlap across layers
- The `LayerSchedulePass` and `ComputeUnitAnnotationPass` are steps in this
  direction but don't do cross-layer fusion

**Effort:** High — requires new scheduling infrastructure. Best approached
incrementally: start with cross-layer buffer reuse analysis, then add pipelined
scheduling.

---

## 5. Priority Matrix

Ranked by (impact × feasibility × alignment with ironmill's architecture):

| Rank | Enhancement | Type | Effort | Impact | Rationale |
|------|-------------|------|--------|--------|-----------|
| 1 | **QuantSpec** (§2a) | Compile + Runtime | Medium-High | 🔥🔥🔥 | Apple-native, designed for CoreML, 2.5× speedup |
| 2 | **DuQuant** (§1b) | Compile pass | Low | 🔥🔥 | Direct extension of PolarQuant, SOTA W4A4 |
| 3 | **SliderQuant** (§1a) | Compile pass | Medium | 🔥🔥 | Automates MixedPrecisionPass, strong for MoE |
| 4 | **DapQ** (§2b) | Compile + Runtime | Medium | 🔥🔥 | Better KV eviction, extends KvCachePass |
| 5 | **RotateKV** (§2d) | Runtime | Low-Medium | 🔥🔥 | Extends TurboQuant to 2-bit |
| 6 | **CodeQuant** (§1c) | Compile pass | Medium | 🔥🔥 | MoE-specific, reuses existing infrastructure |
| 7 | **REAP** (§3a) | Compile pass | Medium | 🔥🔥 | MoE expert pruning, 50% size reduction |
| 8 | **FAFO** (§2c) | Runtime | High | 🔥🔥 | Draftless speculation, single cache |
| 9 | **SpinOut** (§1e) | Compile pass | Medium | 🔥 | Improves PolarQuant rotation quality |
| 10 | **LATMiX** (§1d) | Compile pass | High | 🔥 | Learned transforms, sub-4-bit |
| 11 | **TransFusion** (§4a) | Architecture | High | 🔥🔥🔥 | Global fusion, but major rearchitecture |
| 12 | **Sparse-BitNet** (§3b) | Compile pass | High | 🔥 | Blocked on CoreML N:M support |

---

## 6. Relationship to Existing Passes

How these enhancements connect to ironmill's current 34 passes:

```
Existing Pass                    Enhancement
─────────────────────────────    ──────────────────────────
PolarQuantPass                 → DuQuant (add permutation step)
                               → SpinOut (improve rotation training)
                               → LATMiX (learned affine transform)

MixedPrecisionPass             → SliderQuant (auto bit-width assignment)
PerExpertQuantPass             → CodeQuant (per-expert clustering + rotation)

KvCachePass                    → QuantSpec (draft/verifier KV variants)
                               → DapQ (position-aware eviction)

CodebookOptimizationPass       → CodeQuant (per-expert codebooks)

ModelSplitPass                 → QuantSpec (draft/verifier split)

TurboQuant (runtime)           → RotateKV (2-bit KV cache)
                               → FAFO (draftless speculative loop)

(new)                          → MoEExpertPruningPass (REAP)
(new)                          → StructuredSparsityPass (Sparse-BitNet)

PassPipeline architecture      → TransFusion (global fusion + scheduling)
```

---

## 7. Key References

1. SliderQuant — <https://openreview.net/forum?id=YNqZqw4fLT> (ICLR 2026)
2. DuQuant — <https://duquant.github.io/> (NeurIPS 2025)
3. CodeQuant — <https://openreview.net/forum?id=ATpchFiBQi> (2026)
4. LATMiX — <https://arxiv.org/abs/2602.17681> (Feb 2026)
5. SpinOut — <http://esoc.hanyang.ac.kr/publications/2026/SpinOut_Enhanced_Rotation-Based_Quantization_for_LLM_by_Outlier_Injection.pdf> (2026)
6. QuantSpec — <https://machinelearning.apple.com/research/quantspec> (Apple, ICML 2025)
7. DapQ — <https://arxiv.org/abs/2603.11564> (March 2026)
8. FAFO — <https://openreview.net/forum?id=oSk9tP5Mgs> (ICLR 2026)
9. RotateKV — <https://huggingface.co/papers/2501.16383> (2025)
10. REAP — <https://arxiv.org/abs/2510.13999> (March 2026)
11. Sparse-BitNet — <https://arxiv.org/abs/2603.05168> (2026)
12. TransFusion — <https://jnamaral.github.io/CDOL/papers/ZhangMICRO25.pdf> (MICRO 2025)
