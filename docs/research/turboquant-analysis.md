# TurboQuant - Research Analysis

> **Paper:** [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://arxiv.org/abs/2504.19874)
> **Venue:** ICLR 2026
> **Authors:** Google Research (Kacham, Hadian, Han, Daliri, Gottesbüren, Jayaram)
> **Sub-papers:** [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026), [QJL](https://dl.acm.org/doi/10.1609/aaai.v39i24.34773) (AAAI)

## Summary

TurboQuant is a **data-oblivious vector quantization** method that achieves
near-optimal distortion at extreme bit-widths (2.5–3.5 bits per channel). It
requires no training, no codebook learning, and no calibration data - the
quantizer is derived purely from mathematical properties of random rotations in
high-dimensional space.

It combines two sub-techniques in a two-stage pipeline:

| Stage | Technique | Bits used | Role |
|-------|-----------|-----------|------|
| 1 | **PolarQuant** | Most (2–3.5) | Random rotation → concentrated Beta distribution on coordinates → optimal scalar quantizer per coordinate. Eliminates normalization overhead (no per-block scale/zero-point). |
| 2 | **QJL** | 1 | 1-bit Johnson-Lindenstrauss transform on the quantization residual. Removes inner-product estimation bias. Acts as a mathematical error-checker. |

### Key results

- **3.5 bits/channel:** absolute quality neutrality (zero accuracy loss) on KV
  cache quantization across LongBench, Needle-in-Haystack, ZeroSCROLLS, RULER,
  L-Eval benchmarks.
- **2.5 bits/channel:** marginal quality degradation.
- **6×+ KV memory reduction** with perfect downstream accuracy.
- **Up to 8× attention speedup** on H100 GPUs (4-bit TurboQuant vs 32-bit
  unquantized keys).
- **Superior recall** vs Product Quantization and RabbiQ in high-dimensional
  vector search, despite being data-oblivious (no codebook tuning).
- Within **≈2.7×** of the information-theoretic lower bound on distortion.

### Primary use cases

1. **KV cache compression** - reduce memory footprint of key-value caches in
   autoregressive LLM inference. This is a runtime/online application.
2. **Vector search** - compress high-dimensional embedding vectors for
   nearest-neighbor lookup. Dramatically speeds up index building.

---

## How TurboQuant works

### Stage 1 - PolarQuant

1. **Random rotation:** Apply a random orthogonal rotation R (e.g., randomized
   Hadamard transform) to the input vector.

2. **Distribution concentration:** After rotation, each coordinate of the
   vector follows a scaled Beta(½, (d-1)/2) distribution, where d is the
   vector dimension. This distribution is:
   - Tightly bounded (known support)
   - Highly concentrated (low variance)
   - Analytically computable

3. **Optimal scalar quantization:** Because the distribution is known, we can
   precompute the Lloyd-Max optimal quantization levels for any target
   bit-width. Apply this quantizer independently to each coordinate.

4. **No normalization overhead:** Traditional quantization stores per-block
   scale and zero-point in full precision, adding 1–2 extra bits per number.
   PolarQuant eliminates this because the distribution boundaries are fixed
   and known - no per-block metadata needed.

### Stage 2 - QJL (Quantized Johnson-Lindenstrauss)

1. **Compute residual:** After PolarQuant, compute the quantization error
   (residual = original - quantized).

2. **JL projection:** Apply a Johnson-Lindenstrauss random projection to the
   residual, reducing dimensionality while preserving distances.

3. **1-bit sign encoding:** Reduce each projected coordinate to a single sign
   bit (+1 or -1). This costs exactly 1 additional bit per coordinate.

4. **Unbiased estimation:** QJL uses a special estimator that combines the
   high-precision query with the low-precision data to produce unbiased
   inner-product estimates. This corrects the systematic bias that MSE-optimal
   quantizers introduce in inner product computation.

### Why two stages?

MSE-optimal scalar quantizers (PolarQuant) minimize reconstruction error but
introduce **bias** in inner product estimation - the expected value of the
quantized inner product doesn't equal the true inner product. QJL corrects this
bias using just 1 extra bit, making the combined estimator **unbiased**.

---

## Technical properties

| Property | Value |
|----------|-------|
| Data-oblivious | Yes - no training, no codebook learning |
| Online-capable | Yes - can quantize vectors as they arrive |
| Preprocessing | Random rotation (O(d log d) with Hadamard) |
| Storage overhead | Zero normalization overhead; only quantized values + shared rotation seed |
| Distortion bound | Within ≈2.7× of information-theoretic lower bound |
| Bit-widths tested | 2.5, 3, 3.5, 4 bits per channel |
| Models tested | Gemma, Mistral (open-source LLMs) |
| Hardware tested | H100 GPUs |

> **CoreML/ANE constraint:** ironmill's `constexpr_lut_to_dense` path only supports
> LUT sizes `{2, 4, 16, 64, 256}`, mapping to `n_bits ∈ {1, 2, 4, 6, 8}`. The paper's
> 3-bit and 3.5-bit results cannot be directly replicated on the CoreML path - 2-bit
> and 4-bit are the closest available. See `KNOWN_ISSUES.md` for details.

---

## Relevance to ironmill

### What aligns

| ironmill capability | TurboQuant relevance |
|--------------------|---------------------|
| KV cache awareness (`kv_cache_read`/`kv_cache_update`) | TurboQuant's primary target is KV cache compression |
| Attention fusion (`scaled_dot_product_attention`, `grouped_query_attention`) | TurboQuant operates on Q/K/V tensors within attention |
| INT8 affine quantization (`constexpr_affine_dequantize`) | PolarQuant's scalar quantizer is conceptually similar |
| Palettization (`constexpr_lut_to_dense`) | PolarQuant's precomputed levels map to a LUT |
| RVQ codebook fusion (`codebook_gather`) | TurboQuant's two-stage approach has RVQ-like structure |
| Pass infrastructure (`Pass` trait, `PassPipeline`) | New quantization methods plug in naturally |

### Fundamental tension

TurboQuant's flagship use case - **runtime KV cache compression** - is an
**inference-time** concern. ironmill operates at **compile/conversion time** and
does not control the inference loop. CoreML's runtime manages KV caches.

ironmill **cannot directly implement online KV cache quantization** without a
cooperating inference runtime.

However, PolarQuant's core algorithm (rotation + optimal scalar quantization)
**can** be applied to static weight tensors at conversion time, providing a new
quality/size tradeoff between INT8 (8 bits) and aggressive palettization (2–4
bits with k-means).

### What ironmill is missing

- Random rotation / Hadamard transform utilities
- Polar coordinate transforms
- Beta distribution optimal quantizer levels
- 1-bit quantization support
- Runtime quantization hooks

> **Update (March 2026):** PolarQuant static weight quantization has been
> implemented (see `polar_quantize.rs`, `beta_quantizer.rs`, `rotation.rs`).
> The ANE-direct backend has been probed for op support - see
> [ANE Op Support Matrix](../design/ane-op-support-matrix.md) for empirically verified
> op availability including eval-time correctness checks.

---

## Integration paths

Four paths were identified, ranked by feasibility:

| Path | Approach | Feasibility | Runtime changes? |
|------|----------|-------------|-----------------|
| **A** | PolarQuant-style static weight quantization | ✅ High | None |
| **B** | Model preparation for runtime TurboQuant | ⚠️ Medium | Requires custom runtime |
| **C** | Full TurboQuant for embedding/search models | ⚠️ Medium | Minimal |
| **D** | Rotation-aware codebook fusion | 🔻 Lower | None |

**Path A is recommended** as the starting point. See
[`docs/polar-quant-implementation-plan.md`](../polar-quant-implementation-plan.md)
for the detailed implementation plan.

### What ironmill cannot do

- **Online KV cache quantization** - ironmill doesn't run inference
- **Attention score correction via QJL** - runtime computation
- **Dynamic per-token quantization** - ironmill produces static graphs
- **GPU kernel integration** - TurboQuant's 8× H100 speedup requires custom
  CUDA kernels; CoreML/ANE is a different execution target

These require changes to the inference runtime (Apple's CoreML `MLModel` API,
which ironmill invokes via `objc2-core-ml` for benchmarking but does not
control), not to ironmill itself. CoreML internally dispatches to ANE, GPU, or
CPU based on the model's compute-unit annotations - ironmill has no direct
access to the inference loop.

---

## References

1. TurboQuant: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) - Daliri et al., ICLR 2026
2. PolarQuant: [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) - Han et al., AISTATS 2026
3. QJL: [doi:10.1609/aaai.v39i24.34773](https://dl.acm.org/doi/10.1609/aaai.v39i24.34773) - AAAI
4. Google Research Blog: [TurboQuant: Redefining AI Efficiency](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
