# Inference Improvements Roadmap

## Goal

Improve Ironmill's inference story by adopting the highest-value ideas from
`maderix/ANE`, `Orion`, and adjacent direct-ANE projects without changing
Ironmill's identity.

This roadmap is intentionally **not** a training plan. Ironmill should remain a
Rust-native model conversion, optimization, validation, and deployment
toolchain with:

- **CoreML as the stable default path**
- **`ane-direct` as an experimental backend**
- **compiler and packaging improvements prioritized over bespoke runtime work**

## Current Position

Ironmill already has a strong compiler-side foundation:

- ONNX -> MIL -> CoreML conversion and packaging
- ANE-specific optimization and validation passes
- safe CoreML runtime bindings in `crates/ironmill-coreml/`
- an experimental direct ANE runtime in `crates/ironmill-ane/`
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

The repo already contains useful starting points for this work:

- `crates/mil-rs/src/ir/passes/kv_cache.rs`
- `crates/mil-rs/src/ir/passes/shape_materialize.rs`
- `crates/mil-rs/src/ir/passes/compute_unit.rs`
- `crates/mil-rs/src/ir/passes/model_split.rs`
- `crates/ironmill-coreml/src/lib.rs`
- `crates/ironmill-bench/src/inference.rs`
- `docs/TEST_SPEC.md`
- `docs/ane-direct-runtime-plan.md`
- `docs/research/optimization-opportunities-2026.md`

These pieces suggest the right direction already exists in the codebase, but it
needs to be turned into a coherent inference roadmap.

## Principles

### 1. Toolchain First, Runtime Second

Ironmill should prefer emitting better artifacts, metadata, manifests, and
compiler guidance rather than absorbing large amounts of custom runtime logic.

### 2. CoreML First by Default

The public, stable path should remain CoreML-based. The direct ANE backend is
valuable for research, profiling, and specialized workflows, but it should stay
feature-gated and non-default.

### 3. Treat LLM Inference as Stateful

Decoder-style inference is not a single `predict()` call repeated forever. It
depends on cache layout, sequence buckets, and different strategies for prompt
prefill vs token decode.

### 4. Prefer Packaging and Routing Metadata Over Runtime Tricks

Where possible, Ironmill should emit artifacts that make routing decisions
obvious to downstream runtimes instead of becoming a monolithic custom
inference engine.

### 5. Copy Constraints and Tactics, Not Project Identity

Ironmill should learn from `maderix/ANE`, `Orion`, `Espresso`, and
`ane-infer`, but it should not try to become another direct-ANE runtime
framework.

## Prioritized Improvements

## Tier 1: High-Confidence, High-Fit Improvements

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

- unlocks real decoder-style LLM deployment
- aligns with how modern ANE/CoreML LLM pipelines are actually used
- fits Ironmill's compiler identity better than inventing a new runtime

### 2. Prefill / Decode Split With Shape Buckets

Ironmill should stop treating prompt prefill and token decode as one generic
inference path.

Recommended direction:

- export separate prefill and decode artifacts, or a manifest that describes
  both stages
- generate decode buckets for common fixed shapes instead of relying on one
  generic dynamic path
- add CLI and library affordances for selecting the correct artifact or bucket

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

## Tier 2: High-Value, More Involved Improvements

### 5. Hybrid Execution Planning

Ironmill should support hybrid execution planning without becoming a full custom
hybrid runtime.

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
- packaging-level guidance is a better fit for Ironmill than a giant runtime
  scheduler

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

### 7. Direct ANE Cache Hardening

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

## Tier 3: Valuable Later, But Not Core for Now

### 8. Multi-Artifact Packaging for Runtime Selection

Once bucketing and prefill/decode splitting exist, Ironmill should package
related artifacts together with selection metadata.

Examples:

- prefill model + decode model
- multiple decode buckets
- routing hints for ANE / CPU / GPU
- benchmark metadata for downstream selection

This is a natural follow-on once the core inference improvements above land.

### 9. Limited Speculative Decoding Support at the Toolchain Layer

Ironmill may eventually want to support speculative-decoding-friendly packaging,
but only at the compiler and artifact level.

Recommended scope:

- emit draft/verifier-compatible artifacts
- expose metadata needed by a downstream scheduler
- avoid implementing a full speculative runtime loop inside Ironmill

This should remain secondary to stateful export and bucketed decode support.

## Explicitly Deferred

The following ideas are worth understanding, but they are poor fits for
Ironmill right now:

- training-specific features
- delta compilation and weight hot-reload
- LoRA hot-swap as a runtime feature
- multi-procedure chaining tricks
- custom mega-kernel runtimes
- native INT4 direct-ANE execution as a mainline path
- turning Ironmill into an `Orion` replacement

Why these are deferred:

- they are runtime-heavy rather than toolchain-heavy
- they depend on fragile private APIs
- they create significant maintenance risk
- they blur Ironmill's identity as a Rust-native compiler and deployment tool

## Suggested Execution Order

### Near Term

Focus on:

1. stateful autoregressive export and KV cache support
2. prefill/decode split with shape buckets
3. better ANE-aware routing and diagnostics
4. inference benchmark overhaul

### Mid Term

Focus on:

5. hybrid execution planning
6. per-layer mixed precision for inference
7. direct ANE cache hardening

### Later

Focus on:

8. multi-artifact packaging
9. toolchain-level speculative decoding support

## Success Criteria

This roadmap is succeeding when Ironmill can do all of the following:

- export a practical decoder-style model with explicit cache/state handling
- package prefill and decode artifacts with fixed-shape buckets
- explain why each major block wants ANE, CPU, or GPU
- benchmark real inference loops rather than only generic prediction calls
- support downstream runtimes without forcing Ironmill to become one

## Related Documents

- `docs/ane-direct-runtime-plan.md`
- `docs/research/optimization-opportunities-2026.md`
- `docs/research/ane-research.md`
- `docs/research/competitive-analysis.md`
- `docs/TEST_SPEC.md`
