# Inference Improvements Implementation Plan

## Goal

Improve Ironmill's inference story by adopting the highest-value ideas from
`maderix/ANE`, `Orion`, and adjacent direct-ANE projects.

This plan is intentionally **not** a training plan. Ironmill is a Rust-native
compiler and runtime for Apple Silicon ML deployment:

- **Compiler**: ONNX/SafeTensors/GGUF → MIL IR → CoreML conversion,
  optimization passes, and `.mlmodelc` compilation
- **CoreML runtime**: stable production path via public CoreML APIs
  (`crates/ironmill-coreml/`)
- **ANE direct runtime**: experimental backend via private ANE APIs
  (`crates/ironmill-ane/`)

## Current Position

Ironmill already has a strong compiler and runtime foundation:

- ONNX → MIL → CoreML conversion and packaging
- SafeTensors and GGUF weight loading with direct pipeline integration
- ANE-specific optimization and validation passes
- CoreML runtime backend in `crates/ironmill-coreml/`
- experimental direct ANE runtime backend in `crates/ironmill-ane/`
- unified `RuntimeBackend`/`RuntimeModel` trait abstraction in
  `crates/ironmill-runtime/`
- a benchmark harness in `crates/ironmill-bench/`

The main remaining gap is that Ironmill's inference path is still mostly
structured as **load + predict**, while the strongest ANE projects get their
gains from controlling:

- stateful autoregressive inference
- prefill vs decode specialization
- shape bucketing
- routing across ANE / CPU / GPU
- dispatch-aware benchmarking and diagnostics

## Existing Scaffolding

The repo already contains useful starting points for this plan:

- `crates/mil-rs/src/ir/passes/kv_cache.rs`
- `crates/mil-rs/src/ir/passes/shape_materialize.rs`
- `crates/mil-rs/src/ir/passes/compute_unit.rs`
- `crates/mil-rs/src/ir/passes/model_split.rs`
- `crates/ironmill-coreml/src/lib.rs`
- `crates/ironmill-bench/src/inference.rs`
- `docs/development/TEST_SPEC.md`
- `docs/research/optimization-opportunities-2026.md`
- `crates/mil-rs/src/convert/templates/llama.rs` (prefill/decode split scaffolding)
- `crates/mil-rs/src/convert/pipeline.rs` (multi-stage pipeline with `pipeline.json` manifests)
- `crates/mil-rs/src/convert/moe.rs` (multi-artifact MoE splitting)
- `crates/ironmill-runtime/src/lib.rs` (unified `RuntimeBackend`/`RuntimeModel` traits)

These pieces suggest the right direction already exists in the codebase, but it
needs to be turned into a coherent inference implementation plan.

## Principles

### 1. Compiler Improvements Lift All Runtimes

Better artifacts, metadata, manifests, and compiler passes benefit both the
CoreML and ANE direct runtimes equally. Compiler-side work should be
prioritized when it produces gains across both backends.

### 2. CoreML First by Default

The public, stable runtime path should remain CoreML-based. The direct ANE
backend is valuable for research, profiling, and specialized workflows, but it
should stay feature-gated and non-default.

### 3. Treat LLM Inference as Stateful

Decoder-style inference is not a single `predict()` call repeated forever. It
depends on cache layout, sequence buckets, and different strategies for prompt
prefill vs token decode. Both runtimes need to support this.

### 4. Prefer Structured Metadata Over Hardcoded Behavior

Where possible, Ironmill should emit artifacts with routing decisions,
stage metadata, and shape constraints encoded declaratively - making both
runtimes and any downstream consumers smarter by default.

### 5. Copy Constraints and Tactics, Not Project Identity

Ironmill should learn from `maderix/ANE`, `Orion`, `Espresso`, and
`ane-infer`, but it should remain focused on its own compiler + runtime
architecture rather than replicating another project wholesale.

## Prerequisites

These items are not inference improvements themselves, but they block or
significantly affect multiple Tier 1 improvements. They should be addressed
first or in parallel with early Tier 1 work.

### Transformer Op Decomposition

`RotaryEmbedding` and `GroupQueryAttention` are currently emitted as opaque
custom MIL ops that CoreML cannot compile. Until these are decomposed into
standard MIL ops, stateful autoregressive export (#1) and prefill/decode
splitting (#2) cannot be validated end-to-end on real transformer models.

See: `KNOWN_ISSUES.md` - "Transformer ops are opaque pass-throughs"

### Remove Legacy Compiler Dispatch

The old `Backend` enum and `compile_model_with_backend()` in
`mil-rs/src/compiler.rs` should be removed now that `ironmill-runtime` provides
the trait-based `RuntimeBackend`/`RuntimeModel` abstraction. This cleanup
simplifies the benchmark overhaul (#4) and avoids confusion about which
backend abstraction to use.

## Tier 1: High-Confidence, High-Fit Tasks

### 1. Stateful Autoregressive Export and KV Cache Support

This is the most important inference improvement.

Ironmill should turn the existing cache-related scaffolding into a supported
pipeline for decoder-style models:

- detect autoregressive models from `past_key_values`-style inputs/outputs
- make `KvCachePass` part of a first-class LLM inference flow
- make `AutoregressiveShapeMaterializePass` part of export, not just test/spec
- emit stable cache and state descriptors during CoreML export
- support fixed `max_seq_length` and ring-buffer or sliding-window cache layouts

Why this matters:

- unlocks real decoder-style LLM deployment via both runtimes
- aligns with how modern ANE/CoreML LLM pipelines are actually used
- compiler-side improvements benefit both CoreML and ANE direct backends

### 2. Prefill / Decode Split With Shape Buckets

Ironmill should stop treating prompt prefill and token decode as one generic
inference path.

Recommended direction:

- export separate prefill and decode artifacts, or a manifest that describes
  both stages
- generate decode buckets for common fixed shapes instead of relying on one
  generic dynamic path
- add CLI and library affordances for selecting the correct artifact or bucket

Existing scaffolding:

- `crates/mil-rs/src/convert/templates/llama.rs` already emits separate
  `prefill` and `decode` functions with single-token decode paths when ANE
  mode is enabled
- this work should generalize that approach across architecture templates
  rather than starting from scratch

Why this matters:

- ANE performance is highly shape-sensitive
- `Orion` and `Espresso` both show that prefill and decode want different
  strategies
- bucketed export is a toolchain-friendly way to capture those wins

### 3. Better ANE-Aware Routing and Diagnostics

Ironmill already has ANE validation and compute-unit annotation. It should push
that further so users can make deployment decisions with confidence.

Recommended direction:

- incorporate more constraints and heuristics derived from `maderix/ANE` and
  `Orion`
- explain why an op or block should stay on ANE, CPU, or GPU
- surface specific warnings for known ANE pitfalls, such as causal SDPA
  decomposition, minimum decode bucket size, compile-budget pressure, and poor
  `lm_head` placement
- emit machine-readable routing metadata alongside human-readable reports

Why this matters:

- this is a direct extension of Ironmill's existing strengths
- it improves both CoreML and `ane-direct` workflows
- it avoids opaque "ANE compatibility %" reporting with no next step

### 4. Inference Benchmark Overhaul

Ironmill's benchmark harness should become inference-loop-aware instead of
single-call-latency-centric.

Recommended direction:

- measure cold vs warm load
- measure prefill vs decode separately
- report tokens per second, not just call latency
- track cache hit rate, memory footprint, and compile/load cost
- compare routing strategies and compute-unit selections

Why this matters:

- current benchmarks are useful, but they underrepresent the real costs that
  matter for modern on-device inference
- `Orion` demonstrates the value of measuring compile, reload, and dispatch
  overhead explicitly

## Tier 2: High-Value, More Involved Tasks

### 5. Hybrid Execution Planning

Ironmill should support hybrid execution planning across its runtimes.

Recommended direction:

- emit stage-level routing hints or manifests for ANE / CPU / GPU splits
- let large-model deployments keep certain stages off ANE when that is the
  better trade-off
- support package-level metadata that downstream runtimes can consume

Good candidates include:

- embeddings
- `lm_head`
- oversized matmuls
- stages that are memory-bound rather than ANE-friendly

Why this matters:

- larger models will increasingly need hybrid execution
- packaging-level guidance benefits both CoreML and ANE direct runtimes

### 6. Per-Layer Mixed Precision for Inference

Ironmill should expose more deliberate inference-time mixed-precision policies.

Recommended direction:

- allow attention blocks to remain FP16 where quality demands it
- allow FFN blocks to use INT8 or palettized weights where the trade-off is
  favorable
- make per-layer or per-stage precision selection a user-facing configuration
  rather than a one-off specialization

Why this matters:

- this is a practical deployment win
- it aligns with real-world model serving practice
- it builds on Ironmill's existing quantization and pass infrastructure

### 7. LoRA Hot-Swap at Runtime

Ironmill already supports static LoRA merging during compilation (ONNX and
SafeTensors paths), and the CLI has reserved `--adapter` and `--emit-adapter`
flags. This task extends that foundation into a runtime capability.

Recommended direction:

- implement `--emit-adapter` to package adapter weights separately from the
  base model
- implement `--adapter` to load external adapter weight sets at runtime
- add adapter selection to `RuntimeBackend`/`RuntimeModel` traits so both
  CoreML and ANE direct backends can apply adapters per session or request
- support loading multiple adapter sets and switching between them without
  recompilation

Existing scaffolding:

- `crates/mil-rs/src/convert/lora.rs` - format-agnostic LoRA detection and
  merge kernel
- `crates/mil-rs/src/convert/weights/safetensors.rs` - adapter discovery
  from `adapter_config.json`
- CLI reserved flags in `crates/ironmill-cli/src/main.rs`

Why this matters:

- adapter switching is a core deployment pattern for fine-tuned model serving
- static merge forces recompilation per adapter, which is impractical at scale
- the detection and merge foundations already exist; the gap is runtime plumbing

### 8. Multi-Procedure Pipeline Orchestration

Ironmill already emits multi-stage pipelines with `pipeline.json` manifests,
MoE multi-function bundles, and draft/verifier splits. This task adds a runtime
consumer for those artifacts.

Recommended direction:

- implement a pipeline orchestrator that reads `pipeline.json` and loads
  multiple compiled artifacts
- chain stage outputs to stage inputs automatically based on `depends_on`
  topology
- support state transfer conventions between stages (e.g., KV cache handoff)
- integrate with both CoreML and ANE direct backends

Existing scaffolding:

- `crates/mil-rs/src/convert/pipeline.rs` - stage topology, dependency
  ordering, I/O validation, and manifest emission
- `crates/mil-rs/src/convert/ir_to_proto.rs` - multi-function model bundling
  for MoE
- `crates/mil-rs/src/ir/passes/model_split.rs` - draft/verifier splitting

Why this matters:

- multi-stage execution is required for prefill/decode splits, MoE routing,
  and speculative decoding
- the compile-time packaging is already solid; the missing piece is runtime
  execution
- this directly enables Tier 3 items (multi-artifact packaging, speculative
  decoding)

### 9. Direct ANE Cache Hardening

Ironmill already has a `ProgramCache` for the experimental direct ANE backend.
That is a good idea and should be strengthened.

Recommended direction:

- improve persistent cache validation and reuse rules
- make cache behavior clearer in logs and reports
- handle multi-artifact and bucketed models cleanly
- expose compile-budget pressure in tools and diagnostics

Why this matters:

- compile budget and cache behavior are some of the most important practical
  constraints learned from `maderix/ANE` and `Orion`
- this is a focused improvement with a good risk/reward profile

## Tier 3: Valuable Later, Lower Priority

### 10. Multi-Artifact Packaging for Runtime Selection

Once bucketing and prefill/decode splitting exist, Ironmill should package
related artifacts together with selection metadata.

Examples:

- prefill model + decode model
- multiple decode buckets
- routing hints for ANE / CPU / GPU
- benchmark metadata for downstream selection

Existing scaffolding:

- `crates/mil-rs/src/convert/pipeline.rs` already writes one `.mlpackage` per
  stage with a `pipeline.json` manifest describing stage relationships
- `crates/mil-rs/src/convert/moe.rs` splits models into shared + expert
  artifacts with a manifest
- this item should unify and extend those mechanisms rather than creating a
  new packaging layer

This is a natural follow-on once the core inference improvements above land.

### 11. Limited Speculative Decoding Support at the Toolchain Layer

Ironmill may eventually want to support speculative-decoding-friendly packaging,
but only at the compiler and artifact level.

Recommended scope:

- emit draft/verifier-compatible artifacts
- expose metadata needed by a downstream scheduler
- avoid implementing a full speculative scheduling loop; keep focus on
  artifact generation and metadata

Existing scaffolding:

- `crates/mil-rs/src/ir/passes/model_split.rs` already implements
  draft/verifier splitting by detecting transformer layer boundaries and
  truncating the draft model

This should remain secondary to stateful export and bucketed decode support.

## Explicitly Deferred

The following ideas are worth understanding, but they are out of scope for
this plan:

- training-specific features
- delta compilation and weight hot-reload
- custom mega-kernel runtimes
- native INT4 direct-ANE execution as a mainline path
- turning Ironmill into an `Orion` replacement

Why these are deferred:

- they are orthogonal to the inference improvements in this plan
- several depend on fragile private APIs beyond what `ironmill-ane` already uses
- they carry significant maintenance risk relative to their value

## Implementation Order

### Phase 1

Focus on:

0. transformer op decomposition (prerequisite - unblocks #1 and #2 validation)
1. stateful autoregressive export and KV cache support
2. prefill/decode split with shape buckets
3. better ANE-aware routing and diagnostics
4. inference benchmark overhaul

### Phase 2

Focus on:

5. hybrid execution planning
6. per-layer mixed precision for inference
7. LoRA hot-swap at runtime
8. multi-procedure pipeline orchestration
9. direct ANE cache hardening

### Phase 3

Focus on:

10. multi-artifact packaging
11. toolchain-level speculative decoding support

## Success Criteria

This plan is succeeding when Ironmill can do all of the following:

- export a practical decoder-style model with explicit cache/state handling
- package prefill and decode artifacts with fixed-shape buckets
- load weights from ONNX, SafeTensors, and GGUF through a unified pipeline
- explain why each major block wants ANE, CPU, or GPU
- benchmark real inference loops rather than only generic prediction calls
- run stateful inference through both CoreML and ANE direct runtimes

## Related Documents

- `docs/research/optimization-opportunities-2026.md`
- `docs/research/ane-research.md`
- `docs/research/competitive-analysis.md`
- `docs/development/TEST_SPEC.md`
