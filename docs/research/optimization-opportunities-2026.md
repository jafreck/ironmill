# Optimization Opportunities from Latest Research (March 2026)

This document maps cutting-edge AI research (inference optimization, model
recompilation, fine-tuning, and Apple Neural Engine advances) to concrete
optimization opportunities for ironmill.

---

## Current State of LLMs on Apple Neural Engine

Understanding what models actually run on ANE today sets the context for which
optimizations matter most.

### What runs fully on ANE today

| Model | Params | Approach | Throughput | Chip |
|-------|--------|----------|------------|------|
| Apple Intelligence Foundation Model | ~3B | 2-bit QAT, KV-cache sharing | Production quality | M1–M5 |
| GPT-2 (via Orion) | 124M | Direct ANE APIs (`_ANECompiler`) | 170+ tok/s | M4 Max |
| LLAMA variants (via ANEMLL) | 1–3B | Quantized CoreML, model splitting | ~55–120 GB/s bandwidth | M1–M5 |
| Qwen 2.5/3, Gemma 3, DeepSeek (via ANEMLL) | 1–3B | Quantized CoreML | Competitive with GPU | M3–M5 |

### What requires hybrid ANE/GPU/CPU execution

| Model | Params | Why hybrid |
|-------|--------|------------|
| Llama 3 8B | 8B | Memory exceeds ANE budget; large matmuls need GPU |
| Llama 3 70B+ | 70B+ | Far exceeds on-device memory; needs Private Cloud Compute |
| Any model >~4B | >4B | ANE memory/compute constraints require op splitting |

### Key hardware benchmarks (ANEMLL-bench)

| Chip | Inference time | Bandwidth | Notes |
|------|---------------|-----------|-------|
| M1 | 7.5 ms | ~55–61 GB/s | Baseline |
| M2 | 6.6–8.7 ms | ~62 GB/s | Marginal improvement |
| M3 | 6.9 ms | 63 GB/s | Similar to M2 |
| M3 Max | 3.98 ms | 120 GB/s | 2.2x bandwidth over M1 |
| M4 Max | 3.87 ms | ~119 GB/s | Comparable to M3 Max |
| M5 | 6.1 ms | 70 GB/s | Early results |

### Implications for ironmill

The sweet spot for ANE-optimized CoreML models today is **1–3B parameters with
aggressive quantization** (2-bit to 4-bit). Ironmill's conversion pipeline is
perfectly positioned to serve this niche — the optimizations below should
prioritize making sub-4B models run as fast as possible on ANE, while also
enabling hybrid execution hints for larger models.

---

## 1. Inference Optimization Research → Ironmill Opportunities

### 1a. KV Cache–Aware Graph Transforms

**Research context:** PagedAttention cuts wasted KV cache memory from 60–80%
down to ~4%, yielding 2–4x throughput. Token pruning and dynamic cache eviction
further reduce runtime memory. Apple's own 3B model uses KV-cache sharing
across attention heads.

**Ironmill opportunity:** Add a **KV cache layout pass** that, during attention
fusion, emits CoreML-compatible KV cache ops with static buffer sizing. Today
`attention_fusion.rs` fuses Q/K/V into `scaled_dot_product_attention` but
doesn't reason about cache layout. A new pass could:

- Insert explicit cache-management ops (ring buffers, sliding window) for
  autoregressive models
- Materialize static cache shapes to keep operations ANE-eligible
- Annotate cache tensors for NHWC layout where ANE benefits
- Support grouped-query attention (GQA) where fewer KV heads are shared across
  Q heads — critical for modern architectures like Llama 3

### 1b. Speculative Decoding–Friendly Model Splitting

**Research context:** Speculative decoding (draft + verifier) is now
production-standard in vLLM, SGLang, and TensorRT-LLM, achieving 2–3x speedup.
LongSpec extends this to long-context with constant-size KV caches. Batch
speculative decoding solves the ragged tensor problem for multi-sequence
serving.

**Ironmill opportunity:** Add a **model splitting pass** that can partition a
single model into draft/verifier variants at conversion time:

- Emit a smaller "draft" `.mlpackage` (first N layers or distilled subset)
  alongside the full model
- Ensure both share the same tokenizer metadata and I/O schema
- Could also emit the draft model with more aggressive quantization (e.g.,
  2-bit) while keeping the verifier at higher precision
- This is a conversion-time concern that fits ironmill's scope perfectly

### 1c. Advanced Quantization Schemes

**Research context:** FP8 is gold-standard for GPU inference. Block-scaled
FP4/INT4 with outlier management push further. AWQ (activation-aware
quantization) preserves quality by giving higher precision to channels important
to activations. Marlin kernels yield 2.5x speedup from kernel tuning alone.

**Ironmill opportunity:**

- **Mixed-precision quantization pass**: FP16/INT8/palettization are currently
  mutually exclusive (`pipeline.rs:75-122`). Research shows mixed-precision
  (attention in FP16, FFN in INT8/4-bit) preserves quality while maximizing
  throughput. Allow per-layer or per-op-type precision selection.
- **Per-channel quantization**: `int8_quantize.rs` uses per-tensor affine
  quantization. Per-channel is significantly more accurate for conv/linear.
- **Activation-aware quantization (AWQ-style)**: Weight channels important to
  activations get higher precision. Could be implemented as a
  calibration-informed pass using the existing calibration directory support.

---

## 2. Model Recompilation Research → Ironmill Opportunities

### 2a. Layer-Wise Compiler Parameter Tuning

**Research context:** Layer-wise optimization for NPUs yields ~43% latency
reduction over global search (ICCTA 2024). Greedy parameter propagation across
graph layers outperforms simulated annealing and other global metaheuristics.

**Ironmill opportunity:** The current `PassPipeline` applies each pass
globally. A **layer-aware pipeline scheduler** could:

- Group ops by "layer" (conv+bn+relu clusters, attention blocks, FFN blocks)
- Apply different quantization/fusion strategies per layer type
- Propagate optimal parameters (bit-width, palette size) from layer to layer
  based on sensitivity analysis
- Example: attention layers stay FP16 while FFN layers get 4-bit palettization

### 2b. Additional Fusion Patterns

**Research context:** FlashRNN achieves 50x speedup via I/O-aware multi-head
kernel fusion. Tile-based abstractions enable flexible kernel fusion across
hardware targets. The "Reasoning Compiler" (NeurIPS 2025) shows LLM-guided
optimization dramatically improves fusion discovery.

**Ironmill opportunity:** Expand fusion passes beyond the current set:

- **LayerNorm + Linear fusion** — very common in transformers, not currently
  fused
- **GELU + Linear fusion** — complement to current GELU expansion in
  `op_substitute.rs`
- **Residual add fusion** — skip connection + main branch into single op
- **Multi-head attention → single fused op** with explicit head count attribute
  for ANE
- **Grouped-query attention fusion** — fewer KV heads shared across Q heads

### 2c. Configurable Pipeline Search Space

**Research context:** The "Reasoning Compiler" (NeurIPS 2025) uses LLMs + MCTS
to guide compiler optimizations, dramatically improving sample efficiency for
complex graph rewrites.

**Ironmill opportunity (longer-term):** Make `PassPipeline` externally
configurable:

- Accept pass configurations from a JSON/TOML spec file
- Report detailed per-pass metrics (op counts, estimated latency, memory
  footprint)
- Enable A/B comparison of different pipeline configurations
- This enables external tools (or even LLM agents) to search over pipeline
  configurations

---

## 3. Fine-Tuning Research → Ironmill Opportunities

### 3a. LoRA Adapter Export Support

**Research context:** LoRA is the dominant PEFT method — 99%+ parameter
reduction, 95–98% accuracy retention. Adapter modules enable multi-task serving
from a single base model. Apple's Foundation Models framework now exposes LoRA
adapter APIs natively.

**Ironmill opportunity:** Add **LoRA-aware conversion**:

- Detect LoRA adapter weights in ONNX models (low-rank A/B matrix pairs
  attached to attention projections)
- Merge LoRA weights into base model at conversion time: `W_new = W + BA`
- Optionally emit separate adapter `.mlpackage` files for runtime adapter
  switching
- Support multiple adapter sets → multiple output packages from one base model
- Aligns with Apple's Foundation Models adapter API

### 3b. QLoRA / Sub-2-bit Weight Support

**Research context:** LowRA pushes quantization below 2 bits per parameter with
minimal accuracy loss. QLoRA enables 70B+ models to fine-tune on single 40GB
GPUs by quantizing base weights to 4-bit and training only LoRA adapters in
FP16.

**Ironmill opportunity:** Extend palettization and quantization:

- Current palettization supports 2/4/6/8-bit palettes (`palettize.rs`). Add
  **1-bit** (binary) support for specific layers
- Add **grouped quantization** where different weight groups within a single
  tensor use different codebooks
- Support importing pre-quantized weights from QLoRA/GPTQ/AWQ formats directly
  during ONNX conversion

### 3c. Updatable Model Support

**Research context:** On-device fine-tuning is increasingly viable. Orion
demonstrates 6.6 TFLOPS/W on M4 for ANE training. NeuralForge reverse-engineers
private APIs to enable on-device LLM fine-tuning.

**Ironmill opportunity:** `ir_to_proto.rs` currently hardcodes
`is_updatable: false`. CoreML supports updatable models with on-device
personalization:

- Allow marking specific layers as updatable during conversion
- Emit proper `UpdateDescription` protobuf with training inputs, loss layers,
  and optimizer config
- This keeps ironmill as a conversion tool but enables downstream on-device
  fine-tuning via CoreML's updatable model mechanism

---

## 4. Apple Neural Engine Research → Ironmill Opportunities

### 4a. Orion-Informed ANE Constraint Database

**Research context:** The Orion paper ([arxiv 2603.06728](https://arxiv.org/abs/2603.06728))
reverse-engineers ANE internals — documenting IR program restrictions, numerical
quirks, memory boundaries, and operator support that Apple hasn't publicly
documented. This is the first comprehensive characterization of ANE hardware
constraints.

**Ironmill opportunity:** Dramatically improve `validate.rs`:

- Current `is_ane_supported()` is a static op-type allowlist. Orion reveals
  that ANE support depends on **tensor shapes, data types, and memory
  alignment**, not just op type
- Add shape-aware validation: conv is ANE-eligible only if kernel ≤ certain
  dimensions, input channels aligned to 32/64
- Add memory alignment checks: ANE requires specific tensor stride patterns
- Report estimated ANE vs CPU/GPU execution split for the whole model
- Flag ops that are "technically supported" but perform poorly on ANE

### 4b. Direct ANE Compilation (Bypass xcrun)

**Research context:** Orion and NeuralForge demonstrate direct ANE compilation
via `_ANECompiler` private APIs. This enables custom optimization passes,
partial weight updates without full recompilation (delta compilation: 4.2s →
0.5s per step), and eliminates the cold-compilation bottleneck.

**Ironmill opportunity (experimental):**

- Today ironmill shells out to `xcrun coremlcompiler` (`compiler.rs:26-90`).
  This is a black box with no control over ANE-specific optimizations.
- Investigate wrapping `_ANECompiler` via Rust FFI (ObjC bridging) for direct
  ANE IR emission
- This would enable: incremental compilation, custom memory layout, and
  avoiding the cold-compilation bottleneck documented in `ane-research.md`
- **Risk:** relies on private APIs that could break between macOS versions.
  Should be an opt-in experimental feature behind a feature flag.

### 4c. Hybrid ANE/GPU Execution Hints

**Research context:** Disaggregated inference research
([paper](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)) shows
orchestrating GPU for dense matmuls and ANE for specialized kernels yields best
throughput on Apple Silicon. M3/M4 Max chips achieve 120 GB/s bandwidth when
properly utilizing both compute units.

**Ironmill opportunity:** Emit **compute unit annotations** in the CoreML model:

- CoreML supports per-operation compute unit preferences
- A new pass could analyze each op's shape/type and annotate whether ANE, GPU,
  or CPU is optimal based on the Orion constraint database
- Particularly useful for models in the 3–8B range where some ops inevitably
  fall back — explicit annotations are better than CoreML's heuristic placement

### 4d. NHWC Layout Fix (Unblock Existing Work)

**Code context:** Layout optimization (`layout_optimize.rs`) is already
implemented but **disabled** because transpose serialization causes CoreML
segfaults (`pipeline.rs:65-68`).

**Ironmill opportunity — highest ROI item:**

- ANE natively operates in NHWC; the current NCHW default causes implicit
  transposes at runtime
- Fix transpose serialization in the writer to unblock the layout pass
- This is pure engineering — no research uncertainty, just a serialization bug
- Once fixed, NHWC layout should become the default for ANE-targeted models

### 4e. Op Splitting for Large Models

**Research context:** ANEMLL implements operator splitting to decompose large
matrix multiplications into ANE-sized tiles, with 1GB iOS / 2GB macOS memory
budgets per ANE program.

**Ironmill opportunity:** Add an **op-splitting pass**:

- Decompose oversized linear/matmul ops into ANE-friendly tile dimensions
- Implement memory budget analysis per split subgraph
- Add transformer-specific splitting (multi-head attention distributed across
  ANE compute units)
- This is what enables 3B+ models to actually run on ANE rather than falling
  back to GPU

---

## 5. Mixture of Experts (MoE) on ANE → Ironmill Opportunities

### How ANE execution constrains MoE

MoE models (e.g., Mixtral 8x7B: 47B total params, ~13B active per token) are
attractive because they deliver large-model quality at a fraction of the compute
cost. However, ANE's execution model creates fundamental challenges:

- **ANE requires a fully compiled static graph.** There is no mechanism for
  dynamic per-token expert routing — the router's conditional dispatch has no
  ANE equivalent and falls back to CPU.
- **CoreML does not support true conditional execution.** The `if/else`
  branching that MoE gating needs is not ANE-accelerated.
- **Per-program memory budget.** ANE programs are limited to ~1GB (iOS) / ~2GB
  (macOS). You cannot point ANE at arbitrary weight pages in unified memory
  on-the-fly, even though unified memory avoids PCIe-style copies.

Despite sharing unified memory with CPU and GPU, ANE cannot dynamically "page
in" only the active experts per token. The entire compiled program must fit
within ANE's memory budget.

### Existing approaches in the ecosystem

| Approach | Mechanism | ANE? | Status |
|----------|-----------|------|--------|
| **Pre-split expert models** | Each expert as a separate `.mlpackage`; router on CPU dispatches winners to ANE | Yes | Feasible now |
| **CoreML multi-function models** | Experts bundled as separate functions; runtime calls the selected one | Partial | CoreML 8+ supports multi-function bundles |
| **Orion weight patching** | Patch active expert weights into compiled ANE program via IOSurface | Yes | Experimental; 8x faster than recompile; private APIs |
| **M1MoE (Metal)** | Stream expert weights from SSD; execute on GPU via Metal; router on CPU | No | Works today but bypasses ANE |
| **Apple RoE (Roster of Experts)** | Stochastic routing ensemble; 7B MoE matches 10.5B dense at 30% less compute | Research | Apple internal research (2026) |

### 5a. MoE-Aware Model Splitting Pass

**Ironmill opportunity:** Detect MoE architecture during ONNX → MIL IR
conversion and produce multiple output artifacts:

- Emit **shared layers** (embeddings, router/gating network, final
  normalization, LM head) as one `.mlpackage`
- Emit **each expert** as a separate `.mlpackage` with a standardized I/O
  schema so the runtime can dispatch to the correct one
- Generate a **manifest file** describing the model topology: which experts
  exist, input/output tensor names, router output mapping
- This turns a monolithic MoE ONNX model into a set of ANE-friendly submodels
  that a runtime can orchestrate

### 5b. Static Top-K Expert Fusion

**Ironmill opportunity:** For deployment scenarios where expert activation
patterns are known (via profiling or calibration data), merge only the
frequently-active experts into a single dense model:

- Accept a calibration dataset or activation frequency profile
- Identify the top-K most frequently activated experts
- Merge those K experts into a single dense `.mlpackage`, discarding the rest
- This trades model flexibility for ANE compatibility — the resulting model is
  a dense submodel of the original MoE that fits within ANE memory
- Particularly useful for domain-specific deployments where the same experts
  are always hot

### 5c. Per-Expert Quantization

**Ironmill opportunity:** Apply different compression levels to different
experts based on their importance:

- Hot experts (frequently activated) → FP16 or INT8 for maximum quality
- Cold experts (rarely activated) → 4-bit palettization or 2-bit quantization
- Shared layers (always active) → FP16 for stability
- This maximizes the total number of experts that fit in the ANE memory budget
  while preserving quality for the experts that matter most
- Could be driven by the same calibration data used for static top-K fusion

### 5d. Multi-Function CoreML Bundle Export

**Ironmill opportunity:** Leverage CoreML 8+'s multi-function model support to
bundle experts as separate functions within a single `.mlpackage`:

- Each expert becomes a named function in the CoreML ML Program
- The shared backbone is deduplicated across functions
- Runtime can call individual expert functions by name after routing
- This avoids the overhead of loading/unloading separate model files while
  staying within CoreML's supported API surface
- Requires extending `ir_to_proto.rs` to emit multi-function Program protos

---

## Priority Ranking

| Pri | Item | Impact | Effort | Risk |
|-----|------|--------|--------|------|
| **P0** | 4d. Fix NHWC layout pass | High | Low | Low |
| **P0** | 4a. Orion-informed ANE validation | High | Medium | Low |
| **P1** | 1c. Mixed-precision quantization | High | Medium | Low |
| **P1** | 2b. Additional fusion patterns | Medium | Medium | Low |
| **P1** | 3a. LoRA adapter merge at conversion | High | Medium | Low |
| **P1** | 1a. KV cache layout pass | High | Medium | Medium |
| **P2** | 4c. Compute unit annotations | Medium | Low | Low |
| **P2** | 3c. Updatable model export | Medium | Medium | Low |
| **P2** | 4e. Op splitting for large models | High | High | Medium |
| **P2** | 2a. Layer-wise pipeline scheduling | Medium | High | Medium |
| **P2** | 1b. Model splitting for spec. decoding | Medium | Medium | Medium |
| **P3** | 3b. Sub-2-bit / grouped quantization | Medium | High | Medium |
| **P2** | 5a. MoE-aware model splitting | High | High | Medium |
| **P2** | 5d. Multi-function CoreML bundle | Medium | Medium | Medium |
| **P3** | 5b. Static top-K expert fusion | Medium | High | Medium |
| **P3** | 5c. Per-expert quantization | Medium | Medium | Low |
| **P3** | 4b. Direct ANE compilation (FFI) | Very High | Very High | High |
| **P3** | 2c. Configurable pipeline search | Low-Med | Medium | Low |

---

## Key References

- [Orion: Characterizing and Programming Apple's Neural Engine for LLM Inference and Training](https://arxiv.org/abs/2603.06728) — March 2026
- [NeuralForge: On-device LLM fine-tuning via ANE private APIs](https://agent-wars.com/news/2026-03-13-neuralforge-on-device-llm-fine-tuning-on-mac-using-apple-neural-engine) — March 2026
- [Apple Intelligence Foundation Language Models Tech Report](https://arxiv.org/abs/2507.13575) — 2025
- [ANEMLL: Artificial Neural Engine ML Library](https://www.anemll.com/) & [Benchmarks](https://github.com/Anemll/anemll-bench)
- [Disaggregated LLM Inference on Apple Silicon](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)
- [Reasoning Compiler: LLM-Guided Optimizations (NeurIPS 2025)](https://arxiv.org/abs/2506.01374)
- [LongSpec: Long-Context Speculative Decoding](https://arxiv.org/abs/2502.17421)
- [LowRA: Sub-2-bit LoRA Fine-Tuning](https://arxiv.org/html/2502.08141)
- [Batch Speculative Decoding (ICLR 2026)](https://openreview.net/forum?id=eM51kSFkoG)
- [Layer-wise NPU Compiler Optimization (ICCTA 2024)](https://dl.acm.org/doi/fullHtml/10.1145/3674558.3674562)
- [LLM Inference Optimization 2026 (Zylos Research)](https://zylos.ai/research/2026-01-15-llm-inference-optimization)
- [Apple Roster of Experts (RoE): Hyper-Parallel MoE Inference](https://machinelearning.apple.com/research/roe) — 2026
- [M1MoE: MoE inference on Apple Silicon via Metal](https://github.com/koaWood/M1MoE)
- [Comprehensive Survey of Mixture-of-Experts (2025)](https://arxiv.org/abs/2503.07137)
