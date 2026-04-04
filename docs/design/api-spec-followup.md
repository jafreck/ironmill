# API Specification — Implementation Followup

> **Branch**: `api-spec`
> **Date**: 2026-04-04
> **Status**: 13 commits, 134 files changed, +4231/−3173 lines

This document tracks what was implemented, what remains, and what needs
fixing before the `api-spec` branch can be merged.

---

## What Was Implemented

### Wave 0 — Foundation
- **mil-rs API hardening** (§2.1–§2.5, §6.4) — `#[non_exhaustive]` on 49
  public enums/structs, `ModelConfig` builder, `Pass` trait defaults,
  `PassPipeline::register_pass_factory()`, `ProgressSink` trait
- **ironmill-inference type hardening** (§4.1, §4.2, §4.7–§4.10) —
  `#[non_exhaustive]` on 29 types, `InferenceError::Runtime` boxed,
  `MetalConfig`/`SamplerConfig` builders, validation methods

### Wave 1 — Compile + Generation
- **ironmill-compile API hardening** (§3.1–§3.5) — `ModelComponent` enum,
  `StageQuantize` enum, structured `CompileError` variants,
  `TemplateOptions` Default, `BuildOutput` hardened
- **Generation API + ModelInfo + Memory** (§4.3, §4.11–§4.14) —
  `GenerateRequest`, `TokenStream` iterator, `CancellationToken`,
  `generate()`, `generate_with_callback()`, `ModelInfo::from_config()`,
  `MemoryEstimator`

### Wave 2 — CLI
- **CLI improvements** (§5.1–§5.7) — `QuantizeArg`/`TargetArg` ValueEnums,
  `LossFunctionArg`/`OptimizerArg`, stderr discipline, hard compiler errors

### Wave 3 — Traits + Abstractions
- **CompileTarget trait** (§3.6–§3.7) — `CompileTarget`, `CompileConfig`,
  `CompileOutput`, `ArtifactMetadata`, stub backends, partial re-export cleanup
- **InferenceEngine redesign** (§4.3–§4.4) — `load()` removed from trait,
  `Send` bound added, `RuntimeModel`/`RuntimeBackend` return `InferenceError`
- **Tokenizer abstraction** (§9) — `Tokenizer` trait, `ChatMessage`,
  `TokenizerError`, `hf-tokenizer` feature placeholder
- **CLI README** (§5.8) — All 5 commands documented with usage examples

### Wave 4 — Backends + High-Level API
- **ANE + CoreML wiring** (§4.5–§4.6) — `AneRuntimeModel::predict`,
  `AneDirectBackend::load`, `InferenceEngine for AneInference`,
  `CoremlRuntimeModel::predict` output extraction
- **Platform portability** (§6.1–§6.2) — `compile_error!` removed,
  backends feature-gated, `#![warn(missing_docs)]` on all library crates
- **ironmill-core high-level API** (§10) — `Device`, `Model`,
  `ModelBuilder`, `GenParams`, `TextOutput`, `TextChunk`, `ChatSession`
- **JIT compilation** (§11) — `TensorTransform` trait,
  `TransformPipeline`, `ShaderCache`

---

## Critical Issues (Fix Before Merge)

### C1: Integration tests don't compile

8 `E0639` (non-exhaustive struct construction), 6 `E0609`
(`MoeFuseResult` field access), 6 `E0308` (type mismatches) across
integration test crates. Tests construct structs with struct-literal
syntax that `#[non_exhaustive]` now blocks from outside the defining
crate. Fix by adding constructors/builders or updating test helpers.

### C2: 31 mil-rs lib test failures

`eval_quantize` assertion mismatches, `pipeline spinquant` validation
rejects valid configs, `reader/writer` I/O tests broken. These are
pre-existing in some cases but must be triaged — some may be regressions
from the `ModelConfig` builder changes or `#[non_exhaustive]` additions.

### C3: 7 ironmill-compile lib test failures

SafeTensors config validation tests fail, template warning tests broken.
Likely caused by `ModelConfig::new()` default values or
`TemplateOptions` changes.

### C4: macOS sys crates are non-optional dependencies

`ironmill-ane-sys`, `ironmill-iosurface`, `ironmill-coreml-sys` are
still unconditional deps in `Cargo.toml`. The feature-gated module
structure (§6.2) is in place, but the actual crate dependencies need
`optional = true` and wiring into the feature flags for true
cross-platform builds.

---

## Missing Items (Implement as Follow-Up)

### High Priority

| Spec Section | Item | Notes |
|---|---|---|
| §4.14 | `BatchRunner`, `SequenceHandle`, `BatchRunnerConfig`, `SchedulingPolicy` | Entirely missing. Large feature — managed batch inference loop. |
| §4.11 | `AsyncTokenStream`, `generate_async::spawn()` | Feature-gated behind `async` feature with `tokio` dependency. |
| §4.1 | `ElementType::BFloat16`, `ElementType::Int8` | Common dtypes needed for real-world usage. |
| §4.1 | `SequenceStatus::Waiting` | Missing variant. |
| §2.1 | `ComputeUnit::Cuda` | Trivial to add; needed for extensibility story. |

### Medium Priority

| Spec Section | Item | Notes |
|---|---|---|
| §10 | `Model::from_compiled()`, `Model::stream()`, `Model::chat()` | Requires wiring `ironmill-inference` as a dependency of `ironmill-core`. |
| §10 | `ChatSessionBuilder`, `ChatSession::say_stream()` | Builder pattern for chat + streaming response. |
| §10 | `Model::engine()`, `Model::tokenizer()`, `Model::info()` | Accessor methods for underlying components. |
| §11 | `MetalInference::load_jit()` | JIT loading integration point on MetalInference. |
| §5.7 | `--lenient` flag on validate command | Validation currently exits `Ok(())` on failure instead of non-zero. |
| §9.2 | `HfTokenizer` implementation | Requires `tokenizers` crate dependency. |
| §12.1 | Move `QuantLevel`, `KvQuantLevel`, `MemoryEstimator`, `ModelInfo` to `mil-rs` | Currently in `ironmill-inference`; spec says shared types belong in `mil-rs`. |

### Low Priority

| Spec Section | Item | Notes |
|---|---|---|
| §8.2 | `MetalInference::load(config, &artifacts)` one-step constructor | Still uses two-step `new()` + `load()`. Update migration guide or implement. |
| §3.6 | Remove `proto::specification::Model` and `tensor_utils` re-exports | Currently kept with TODO comments; used by `ironmill-compile-ffi`. |
| §4.11 | `GenerateError` return from `generate()` / `generate_with_callback()` | Currently returns `Result<_, InferenceError>`, not `Result<_, GenerateError>`. |

---

## Deviations from Spec (Intentional)

These are places where the implementation intentionally differs from the
specification. The deviations are considered improvements or pragmatic
choices.

| Location | Spec | Implementation | Rationale |
|---|---|---|---|
| §3.1 `LayerType` | `{Attention, Feedforward, Full}` | `{Conv, Attention, Ffn, Norm, Other}` | Richer; reflects actual model topologies |
| §3.1 `OpPrecision` | `{Float16, Float32}` | `{None, Fp16, Int8}` | Different semantics; `None` needed |
| §4.7 `with_fa2_prefill` | Takes no args | Takes `bool` | More flexible; can disable explicitly |
| §4.7 `with_cla` | Takes `Vec<usize>` | Takes `ClaConfig` | Type-safe; wraps the vector |
| §4.9 `SpecConfig` | `acceptance_threshold: f64` | `f32` | Matches existing field type |
| §10 `ChatSession` | Method named `say()` | Named `send()` | Naming preference |
| §10 `Model::generate` | Takes `GenParams` by value | Takes `&GenParams` by ref | Avoids unnecessary clone |
| §3.7 `CompileConfig` | Has `pipeline: PassPipeline` | Has `pipeline: Option<PassPipeline>` | `PassPipeline` isn't `Clone` |
| §3.7 `estimate_size` | Returns `MemoryEstimator::weight_memory(...)` | Returns `Ok(0)` | `MemoryEstimator` lives in inference crate |

---

## Test Plan

Before merging, all of the following must pass:

```bash
# Full workspace compilation
cargo check --workspace

# Unit tests
cargo test -p mil-rs --lib
cargo test -p ironmill-compile --lib
cargo test -p ironmill-inference --lib
cargo test -p ironmill-core --lib

# Integration tests
cargo test -p ironmill-compile --test '*'
cargo test -p mil-rs --test '*'

# Clippy
cargo clippy --workspace -- -D warnings
```

---

## Commit Log

```
0a6da1f feat(core): add high-level Model API, Device, GenParams, ChatSession
81fbedc feat(inference): add JIT compilation types and shader cache
020a13b feat(inference): platform portability, feature-gate backends, warn(missing_docs)
6b737dd feat(inference): wire up ANE and CoreML backend implementations
a118e08 docs: update README with full CLI command reference
e113587 feat(core): add Tokenizer trait, ChatMessage, TokenizerError
c88bd56 feat(inference): two-tier InferenceEngine, RuntimeModel typed errors
0ede6b7 feat(compile): add CompileTarget trait, ArtifactMetadata, re-export cleanup
5d92021 feat(cli): replace string flags with ValueEnums, stderr discipline, hard errors
419bf16 feat(inference): add generation API, ModelInfo, MemoryEstimator
3b283b7 feat(compile): add #[non_exhaustive], ModelComponent, StageQuantize, CompileError
6b2ad8d feat(inference): add #[non_exhaustive], error improvements, config builders
7c78a8a feat(mil-rs): add #[non_exhaustive], builders, Pass defaults, ProgressSink
```
