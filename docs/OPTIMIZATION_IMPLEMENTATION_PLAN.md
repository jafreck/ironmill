# Optimization Implementation Plan

Concrete implementation plan derived from
[optimization-opportunities-2026.md](research/optimization-opportunities-2026.md).
Picks up where [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) Phase 4 leaves
off, organized into four phases with clear deliverables and file-level task
breakdowns.

**Validation targets:**
- **Whisper Medium encoder** (769M params) — exercises all core passes, has
  established baselines via WhisperKit
- **Qwen3-TTS 0.6B** (multi-stage pipeline) — exercises autoregressive, KV
  cache, causal conv, RVQ codebooks; requires Phase 7 prerequisites

---

## Phase 5: ANE Reliability & Core Improvements

**Goal**: Fix the highest-ROI bug, upgrade ANE validation from a static
allowlist to shape-aware analysis, and unlock mixed-precision quantization.

**Ship**: `mil-rs` v0.4, `ironmill` v0.3 with accurate ANE reports and
mixed-precision support.

### 5.1 Fix NHWC layout pass (P0)

The layout optimization pass (`layout_optimize.rs`) is already implemented but
disabled in the default pipeline (`pipeline.rs:65-68`) because inserted
transposes cause CoreML runtime segfaults during serialization.

- [ ] Reproduce the segfault: write a minimal model with an explicit transpose,
  serialize via `ir_to_proto.rs`, load in CoreML
- [ ] Fix transpose serialization in `writer/mlpackage.rs` and/or
  `convert/ir_to_proto.rs` — likely an incorrect attribute mapping or missing
  dimension permutation in the protobuf output
- [ ] Re-enable `LayoutOptimizationPass` in `PassPipeline::new()` default
  ordering, after `op-substitution`
- [ ] Add integration test: ONNX model → NHWC-optimized `.mlpackage` →
  `xcrun coremlcompiler compile` succeeds
- [ ] Make NHWC the default for `--target cpu-and-ne`

**Files**: `ir/passes/layout_optimize.rs`, `ir/pipeline.rs`,
`convert/ir_to_proto.rs`, `writer/mlpackage.rs`

### 5.2 Orion-informed ANE validation (P0)

Current `validate.rs` uses a static op-type allowlist via
`is_ane_supported()`. Orion research shows ANE eligibility depends on tensor
shapes, data types, and memory alignment — not just op type.

- [ ] Add shape-aware validation rules:
  - Conv: kernel dimensions ≤ ANE limits, input channels aligned to 32/64
  - MatMul/Linear: inner dimension limits
  - Memory alignment: tensor strides must match ANE requirements
- [ ] Add per-op performance annotations: flag ops that are "technically
  supported" but perform poorly on ANE (e.g., large gather ops)
- [ ] Compute estimated ANE vs CPU/GPU execution split as a percentage of total
  compute, not just op count
- [ ] Emit machine-readable JSON report alongside human-readable text
  (`ironmill validate --format json`)
- [ ] Add `--format` flag to CLI `validate` subcommand

**Files**: `validate.rs`, `ironmill-cli/src/main.rs`

### 5.3 Mixed-precision quantization (P1)

FP16/INT8/palettization are currently mutually exclusive
(`pipeline.rs:75-122`). Research shows mixed-precision (e.g., attention in
FP16, FFN in INT8) preserves quality while maximizing throughput.

- [ ] Add per-layer precision configuration: accept a TOML/JSON spec mapping
  op names or op-type patterns to quantization modes
- [ ] Implement `MixedPrecisionPass` in `ir/passes/` that applies different
  quantization strategies per operation based on config
- [ ] Relax mutual exclusivity in `PassPipeline` when mixed-precision config
  is provided
- [ ] Upgrade `int8_quantize.rs` from per-tensor to per-channel affine
  quantization for conv/linear ops
- [ ] CLI: `ironmill compile model.onnx --quantize-config quant.toml`
- [ ] Add default mixed-precision profiles: `--quantize mixed-fp16-int8`
  (attention FP16, everything else INT8)

**Files**: `ir/passes/int8_quantize.rs`, `ir/passes/mixed_precision.rs` (new),
`ir/pipeline.rs`, `ironmill-cli/src/main.rs`

### 5.4 Additional fusion patterns (P1)

Expand fusion passes beyond the current conv+bn, conv+relu, linear+relu, and
attention patterns.

- [ ] **LayerNorm + Linear fusion** — very common in transformers, not
  currently fused. Add to `op_fusion.rs`
- [ ] **GELU + Linear fusion** — complement the existing GELU expansion in
  `op_substitute.rs`. Fuse the expanded GELU subgraph + following linear
- [ ] **Residual add fusion** — detect skip-connection pattern (main branch +
  identity branch → add) and emit a single fused op
- [ ] **Grouped-query attention (GQA) fusion** — extend `attention_fusion.rs`
  to detect GQA patterns where fewer KV heads are shared across Q heads
- [ ] Add benchmark cases to `benches/passes.rs` for each new fusion

**Files**: `ir/passes/op_fusion.rs`, `ir/passes/attention_fusion.rs`,
`ir/passes/op_substitute.rs`, `benches/passes.rs`

### 5.5 Compute unit annotations (P2)

CoreML supports per-operation compute unit preferences. Emit annotations
based on the improved ANE constraint database from 5.2.

- [ ] Add `compute_unit` attribute to `Operation` (enum: `Ane`, `Gpu`, `Cpu`,
  `Any`)
- [ ] Implement `ComputeUnitAnnotationPass` that uses Orion constraint data to
  assign preferred compute unit per op
- [ ] Serialize compute unit preferences in `ir_to_proto.rs`
- [ ] CLI: `ironmill compile model.onnx --annotate-compute-units`

**Files**: `ir/operation.rs`, `ir/passes/compute_unit.rs` (new),
`convert/ir_to_proto.rs`, `ironmill-cli/src/main.rs`

### Phase 5 validation

- Whisper Medium encoder: convert with NHWC + mixed-precision, validate 100%
  ANE eligibility, compare output size and weight fidelity against WhisperKit
  baseline
- Add Whisper benchmark cases to `benches/pipeline.rs`

---

## Phase 6: LLM & Autoregressive Support

**Goal**: Enable ironmill to convert autoregressive transformer models (LLMs,
TTS decoders) with proper KV cache handling and LoRA adapter support.

**Ship**: `mil-rs` v0.5, `ironmill` v0.4 with autoregressive model support.

### 6.1 KV cache layout pass (P1)

Current `attention_fusion.rs` fuses Q/K/V into
`scaled_dot_product_attention` but doesn't reason about cache layout.

- [ ] Detect autoregressive attention patterns: ops that consume and produce
  `past_key_values` tensors
- [ ] Insert explicit cache management ops: statically-sized ring buffers for
  KV cache with fixed max sequence length
- [ ] Materialize cache shapes to keep all ops ANE-eligible (builds on
  `shape_materialize.rs`)
- [ ] Annotate cache tensors for NHWC layout where ANE benefits (depends on
  5.1 NHWC fix)
- [ ] Support grouped-query attention (GQA) cache layout — fewer KV heads
  shared across Q heads

**Files**: `ir/passes/kv_cache.rs` (new), `ir/passes/attention_fusion.rs`,
`ir/passes/shape_materialize.rs`

**Depends on**: 5.1 (NHWC layout fix)

### 6.2 Autoregressive model support (P1)

Current passes assume static, feed-forward graphs. Autoregressive models
require special handling for KV cache state and dynamic sequence lengths.

- [ ] Detect autoregressive patterns in ONNX: `past_key_values` inputs/outputs,
  causal attention masks
- [ ] Extend `onnx_to_mil.rs` to handle autoregressive graph structure
- [ ] Insert KV cache management ops with statically-sized ring buffers
  (coordinates with 6.1)
- [ ] Materialize sequence length dimensions to fixed max values for ANE
  (extends `shape_materialize.rs` for autoregressive-specific patterns)
- [ ] Support CoreML stateful model export — persist KV cache state across
  inference calls via CoreML's state mechanism in `ir_to_proto.rs`
- [ ] Add `is_updatable` and state descriptor support to `ir_to_proto.rs`
  (currently hardcoded to `false`)

**Files**: `convert/onnx_to_mil.rs`, `convert/ir_to_proto.rs`,
`ir/passes/kv_cache.rs`, `ir/passes/shape_materialize.rs`

**Depends on**: 6.1 (KV cache pass)

### 6.3 LoRA adapter merge at conversion (P1)

LoRA is the dominant parameter-efficient fine-tuning method. Apple's
Foundation Models framework now exposes LoRA adapter APIs natively.

- [ ] Detect LoRA adapter weights in ONNX models: low-rank A/B matrix pairs
  attached to attention projection weights (naming convention:
  `*.lora_A.weight`, `*.lora_B.weight`)
- [ ] Implement weight merge: `W_new = W + alpha * B @ A` during ONNX → MIL
  conversion in `onnx_to_mil.rs`
- [ ] CLI: `ironmill compile model.onnx --merge-lora` (default: merge)
- [ ] Optional: `--emit-adapter` to produce a separate adapter `.mlpackage`
  for runtime adapter switching instead of merging
- [ ] Support multiple adapter sets → multiple output packages from one base
  model: `--adapter adapter1.safetensors --adapter adapter2.safetensors`

**Files**: `convert/onnx_to_mil.rs`, `convert/lora.rs` (new),
`ironmill-cli/src/main.rs`

### 6.4 Updatable model export (P2)

CoreML supports updatable models with on-device personalization.
`ir_to_proto.rs` currently hardcodes `is_updatable: false`.

- [ ] Allow marking specific layers as updatable during conversion via CLI flag
  or config: `--updatable-layers "layer1,layer2"`
- [ ] Emit `UpdateDescription` protobuf with training inputs, loss layer
  reference, and optimizer configuration
- [ ] Support basic on-device fine-tuning config: learning rate, epochs, loss
  function type

**Files**: `convert/ir_to_proto.rs`, `ironmill-cli/src/main.rs`

### Phase 6 validation

- Convert a small autoregressive model (GPT-2 124M or similar) end-to-end
  with KV cache, validate ANE eligibility of the result
- LoRA merge: convert a base model + LoRA adapter, compare numerics of merged
  output against PyTorch merged baseline

---

## Phase 7: Multi-Stage Pipelines & Large Model Support

**Goal**: Convert multi-stage model pipelines (TTS, encoder-decoder) and enable
models >3B params to run on ANE via op splitting.

**Ship**: `mil-rs` v0.6, `ironmill` v0.5 with pipeline conversion and op
splitting.

### 7.1 Multi-ONNX pipeline conversion (P2)

Support converting a set of related ONNX files into coordinated `.mlpackage`
outputs.

- [ ] Define pipeline manifest schema (TOML): stages, ONNX file paths, I/O
  connections between stages, per-stage optimization config
- [ ] Implement pipeline conversion: convert each stage independently, validate
  output tensor shapes/types from stage N match input expectations of stage N+1
- [ ] Apply per-stage optimization: e.g., aggressive quantization for codec
  decoders, FP16 for autoregressive LMs
- [ ] Emit a pipeline manifest alongside `.mlpackage` files for runtime
  orchestration
- [ ] CLI: `ironmill compile-pipeline pipeline.toml -o output_dir/`

**Files**: `convert/pipeline.rs` (new), `ironmill-cli/src/main.rs`

### 7.2 Op splitting for large models (P2)

ANEMLL implements operator splitting to decompose large matmuls into
ANE-sized tiles. ANE memory budgets: ~1GB iOS, ~2GB macOS.

- [ ] Implement memory budget analysis: estimate per-op memory footprint based
  on tensor shapes and dtypes
- [ ] Implement `OpSplittingPass`: decompose oversized linear/matmul ops into
  ANE-friendly tile dimensions when they exceed memory budget
- [ ] Support configurable memory budget via CLI:
  `--ane-memory-budget 1GB` (iOS) / `2GB` (macOS)
- [ ] Add transformer-specific splitting: multi-head attention distributed
  across tiles, with proper concat of results
- [ ] Validate that split subgraphs fit within budget and produce numerically
  equivalent results

**Files**: `ir/passes/op_split.rs` (new), `ir/pipeline.rs`,
`ironmill-cli/src/main.rs`

### 7.3 Causal convolution support (P2)

Streaming TTS and audio models use causal convolutions. Asymmetric left-only
padding patterns must be preserved through fusion.

- [ ] Detect causal padding patterns in ONNX (asymmetric left-only padding on
  conv ops)
- [ ] Verify these map to ANE-compatible conv ops after fusion passes
- [ ] Add fusion guard: prevent fusion patterns that would break causality
  (e.g., merging causal + non-causal convs)
- [ ] Add test cases with causal conv patterns

**Files**: `convert/onnx_to_mil.rs`, `ir/passes/op_fusion.rs`

### 7.4 Vector quantization / codebook ops (P2)

Neural audio codecs (Mimi, EnCodec) use Residual Vector Quantization (RVQ)
with learned codebooks. These appear in ONNX as gather/embedding patterns.

- [ ] Detect RVQ decode patterns in ONNX: codebook lookup → sum across
  quantization levels
- [ ] Map codebook gather ops to ANE-friendly static embedding tables with
  proper shaping
- [ ] Support multi-codebook models (8–16 codebooks summed)
- [ ] Optionally palettize codebook weights if memory-constrained

**Files**: `convert/onnx_to_mil.rs`, `ir/passes/` (codebook detection)

### 7.5 Speculative decoding model splitting (P2)

Partition a single model into draft/verifier variants at conversion time for
speculative decoding workflows.

- [ ] Implement `ModelSplitPass`: given a layer count N, emit a "draft"
  `.mlpackage` from the first N layers and a full "verifier" `.mlpackage`
- [ ] Ensure both share the same I/O schema and tokenizer metadata
- [ ] Support different quantization per variant: draft at 2-bit, verifier at
  FP16
- [ ] CLI: `ironmill compile model.onnx --split-draft-layers 6`

**Files**: `ir/passes/model_split.rs` (new), `ironmill-cli/src/main.rs`

### Phase 7 validation — Qwen3-TTS 0.6B

With phases 6 and 7 complete, Qwen3-TTS becomes convertible:

| Stage | Requires |
|-------|----------|
| Talker (28-layer autoregressive) | 6.1 KV cache, 6.2 autoregressive support |
| Code Predictor (5-layer parallel) | Core pipeline (already works) |
| Mimi Codec Decoder (RVQ + ConvNet) | 7.3 causal conv, 7.4 codebook ops |
| Speaker Encoder (ConvNet) | Core pipeline (already works) |

- Convert all four ONNX stages via `compile-pipeline`
- Validate per-stage ANE eligibility
- Measure total `.mlpackage` size vs original ONNX set
- First-packet latency target: <200ms on M3+

---

## Phase 8: MoE, Experimental Features & Long-Term

**Goal**: Support Mixture-of-Experts models on ANE, add experimental direct
ANE compilation, and enable external pipeline configuration.

**Ship**: `mil-rs` v0.7, `ironmill` v0.6 with MoE support and experimental
features.

### 8.1 MoE-aware model splitting (P2)

Detect MoE architecture during conversion and produce multiple ANE-friendly
output artifacts.

- [ ] Detect MoE patterns in ONNX: router/gating network → expert dispatch →
  weighted combination
- [ ] Emit shared layers (embeddings, router, final norm, LM head) as one
  `.mlpackage`
- [ ] Emit each expert as a separate `.mlpackage` with standardized I/O schema
- [ ] Generate a manifest describing topology: expert count, I/O tensor names,
  router output → expert mapping
- [ ] CLI: `ironmill compile model.onnx --moe-split`

**Files**: `convert/moe.rs` (new), `ironmill-cli/src/main.rs`

### 8.2 Multi-function CoreML bundle export (P2)

Leverage CoreML 8+ multi-function model support to bundle experts as separate
functions within a single `.mlpackage`.

- [ ] Extend `ir_to_proto.rs` to emit multi-function Program protos (currently
  assumes single `main` function)
- [ ] Each expert becomes a named function; shared backbone is deduplicated
- [ ] Runtime can call individual expert functions by name after routing
- [ ] CLI: `ironmill compile model.onnx --moe-bundle` (alternative to
  `--moe-split`)

**Files**: `convert/ir_to_proto.rs`, `ir/program.rs`

**Depends on**: 8.1 (MoE detection)

### 8.3 Per-expert quantization (P3)

Apply different compression levels to different experts based on activation
frequency.

- [ ] Accept calibration data or activation frequency profile
- [ ] Hot experts (frequently activated) → FP16 or INT8
- [ ] Cold experts (rarely activated) → 4-bit palettization or 2-bit
- [ ] Shared layers (always active) → FP16 for stability
- [ ] Reuses calibration infrastructure from `int8_quantize.rs`

**Files**: `ir/passes/mixed_precision.rs` (extends 5.3)

**Depends on**: 5.3 (mixed-precision), 8.1 (MoE detection)

### 8.4 Static top-K expert fusion (P3)

For domain-specific deployments where the same experts are always active,
merge the top-K into a single dense model.

- [ ] Accept calibration dataset or activation frequency profile
- [ ] Identify top-K most frequently activated experts
- [ ] Merge K experts into a single dense `.mlpackage`, discard the rest
- [ ] CLI: `ironmill compile model.onnx --moe-fuse-topk 2 --cal-data data/`

**Files**: `convert/moe.rs`

**Depends on**: 8.1 (MoE detection)

### 8.5 Sub-2-bit and grouped quantization (P3)

Push quantization further for maximum model compression.

- [ ] Add 1-bit (binary) palettization support for specific layers in
  `palettize.rs`
- [ ] Implement grouped quantization: different weight groups within a single
  tensor use different codebooks
- [ ] Support importing pre-quantized weights from QLoRA/GPTQ/AWQ formats
  during ONNX conversion

**Files**: `ir/passes/palettize.rs`, `convert/onnx_to_mil.rs`

### 8.6 Configurable pipeline search space (P3)

Make `PassPipeline` externally configurable for experimentation and A/B
testing.

- [ ] Accept pass configurations from a TOML spec file
- [ ] Report detailed per-pass metrics: op counts, estimated latency, memory
  footprint changes
- [ ] CLI: `ironmill compile model.onnx --pipeline-config pipeline.toml`
- [ ] Add `ironmill pipeline-report` subcommand for comparing configurations

**Files**: `ir/pipeline.rs`, `ironmill-cli/src/main.rs`

### 8.7 Layer-wise pipeline scheduling (P2)

Group ops by logical layer and apply different optimization strategies per
layer type.

- [ ] Detect layer boundaries: conv+bn+relu clusters, attention blocks, FFN
  blocks
- [ ] Allow per-layer-type quantization/fusion strategies in pipeline config
- [ ] Propagate optimal parameters (bit-width, palette size) layer-to-layer
  based on sensitivity

**Files**: `ir/pipeline.rs`, `ir/passes/mixed_precision.rs`

**Depends on**: 5.3 (mixed-precision), 8.6 (configurable pipeline)

### 8.8 Direct ANE compilation via FFI (P3, experimental)

Bypass `xcrun coremlcompiler` by wrapping `_ANECompiler` private APIs via
Rust FFI. Currently ironmill shells out to `xcrun` (`compiler.rs:26-90`).

- [ ] Investigate `_ANECompiler` ObjC API surface via Orion/NeuralForge
  documentation
- [ ] Implement Rust FFI wrapper behind `#[cfg(feature = "ane-direct")]`
  feature flag
- [ ] Enable incremental/delta compilation (4.2s → 0.5s per step)
- [ ] **Risk**: relies on private APIs — must be opt-in experimental feature,
  tested against specific macOS versions
- [ ] CLI: `ironmill compile model.onnx --backend ane-direct`

**Files**: `compiler.rs` (extend), `ffi/ane.rs` (new)

---

## Dependency Graph

```
Phase 5 (ANE Reliability)
├── 5.1 NHWC layout fix ←── no deps (P0, start here)
├── 5.2 ANE validation ←── no deps (P0, start here)
├── 5.3 Mixed-precision ←── no deps (P1)
├── 5.4 Fusion patterns ←── no deps (P1)
└── 5.5 Compute unit annotations ←── 5.2

Phase 6 (LLM Support)
├── 6.1 KV cache pass ←── 5.1
├── 6.2 Autoregressive support ←── 6.1
├── 6.3 LoRA merge ←── no deps (P1)
└── 6.4 Updatable models ←── no deps (P2)

Phase 7 (Pipelines & Large Models)
├── 7.1 Multi-ONNX pipeline ←── no deps (P2)
├── 7.2 Op splitting ←── 5.2
├── 7.3 Causal convolution ←── no deps (P2)
├── 7.4 Vector quantization ←── no deps (P2)
└── 7.5 Speculative decoding split ←── no deps (P2)

Phase 8 (MoE & Experimental)
├── 8.1 MoE splitting ←── no deps
├── 8.2 Multi-function bundle ←── 8.1
├── 8.3 Per-expert quantization ←── 5.3, 8.1
├── 8.4 Static top-K fusion ←── 8.1
├── 8.5 Sub-2-bit quantization ←── no deps
├── 8.6 Configurable pipeline ←── no deps
├── 8.7 Layer-wise scheduling ←── 5.3, 8.6
└── 8.8 Direct ANE compilation ←── no deps (experimental)
```

### Cross-phase critical path

```
5.1 NHWC fix → 6.1 KV cache → 6.2 Autoregressive → Qwen3-TTS validation
5.2 ANE validation → 5.5 Compute annotations
5.3 Mixed-precision → 8.3 Per-expert quant / 8.7 Layer-wise scheduling
```

---

## Success Criteria

| Phase | Done when... |
|-------|-------------|
| Phase 5 | Whisper Medium converts with NHWC layout enabled, validates 100% ANE-eligible, mixed-precision produces smaller model with <1% quality loss |
| Phase 6 | GPT-2 124M converts with KV cache and runs autoregressively via CoreML; LoRA adapter merge produces numerically correct output |
| Phase 7 | Qwen3-TTS 0.6B all four stages convert and validate; op splitting enables a 3B+ model to fit ANE memory budget |
| Phase 8 | A Mixtral-style MoE model splits into per-expert packages; configurable pipeline enables A/B comparison of optimization strategies |
