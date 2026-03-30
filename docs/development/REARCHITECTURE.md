# Ironmill Rearchitecture Plan

## Executive Summary

Ironmill has grown organically and its crate boundaries no longer reflect clean areas of
responsibility. `mil-rs` is a kitchen-sink crate that mixes a reusable MIL/CoreML library
with ANE-specific FFI, model templates, and build orchestration. `ironmill-ane` conflates
compilation (splitting, MIL emission, program compilation) with inference (tensor
management, decode loops, TurboQuant). Unsafe code is scattered across 310 sites in 5
crates with no quarantine boundary.

This document defines the target architecture: **`mil-rs` as an independent, publishable
MIL/CoreML library**, **`ironmill-compile` as the primary compilation crate**, and
**`ironmill-inference` as the primary inference crate**, each with focused subcrates for
distinct responsibilities.

---

## Target Crate Architecture

```
mil-rs                          (independent MIL/CoreML library — no ironmill dependency)
├── ir/                         (Program, Function, Block, Operation, Value, TensorType)
│   ├── pass.rs                 (Pass trait)
│   ├── pipeline.rs             (PassPipeline orchestration)
│   └── passes/                 (generic passes only)
├── proto/                      (CoreML + ONNX generated protobuf types)
├── convert/
│   ├── proto_to_ir.rs          (CoreML protobuf → MIL IR)
│   ├── ir_to_proto.rs          (MIL IR → CoreML protobuf)
│   ├── onnx_graph.rs           (ONNX graph → MIL IR)
│   └── onnx_to_mil.rs          (ONNX node → MIL operation)
├── reader/                     (mlmodel, mlpackage, onnx readers)
├── writer/                     (mlmodel, mlpackage writers)
├── analysis/                   (FLOPs, architecture analysis)
└── error.rs

ironmill-compile                (primary compilation crate)
├── ane/                        (ANE-specific compilation)
│   ├── passes/                 (ANE-specific IR passes)
│   ├── split.rs                (model → ANE-sized subprograms)
│   ├── packing.rs              (spatial I/O packing)
│   ├── mil_text.rs             (IR → ANE MIL text emission)
│   ├── blobfile.rs             (weight blob format)
│   ├── validate.rs             (ANE compatibility checking)
│   └── cache.rs                (compiled program disk cache)
├── coreml/                     (CoreML-specific compilation)
│   ├── compiler.rs             (xcrun coremlcompiler wrapper)
│   └── build_api.rs            (CompileBuilder for build.rs usage)
├── templates/                  (model architecture templates)
│   ├── llama.rs
│   ├── qwen.rs
│   └── gemma.rs
├── weights/                    (weight provider abstraction)
│   ├── safetensors.rs
│   └── gguf.rs
├── convert/                    (ironmill-specific conversions)
│   ├── lora.rs                 (LoRA merge/detection)
│   ├── moe.rs                  (MoE split/fuse)
│   └── pipeline.rs             (multi-model pipeline conversion)
└── c_api.rs                    (C FFI surface)

ironmill-inference              (primary inference crate)
├── engine.rs                   (InferenceEngine trait + shared logic)
├── sampling.rs                 (token sampling, stopping criteria)
├── ane/                        (ANE inference backend)
│   ├── runtime.rs              (program load/unload/eval lifecycle)
│   ├── decode.rs               (autoregressive decode loop)
│   └── turboquant/             (TurboQuant subsystem)
│       ├── config.rs
│       ├── kv_cache.rs
│       ├── mil_emitter.rs
│       └── model.rs
└── coreml/                     (CoreML inference backend)
    └── runtime.rs              (MLModel load/predict)

ironmill-ane-sys                (quarantined unsafe: ANE private API FFI)
├── objc.rs                     (ObjC runtime helpers: msgSend, selectors, etc.)
├── compiler.rs                 (ANE compilation: compile_mil_text, patch_weights)
├── runtime.rs                  (ANE execution: load/unload/eval programs)
└── error.rs                    (NSError extraction)

ironmill-iosurface              (quarantined unsafe: IOSurface tensor management)
├── surface.rs                  (IOSurface create/lock/release with safe API)
└── tensor.rs                   (AneTensor: typed read/write over IOSurface)

ironmill-coreml-sys             (quarantined unsafe: CoreML ObjC FFI)
└── model.rs                    (MLModel load/predict/extract with safe API)

ironmill-cli                    (thin CLI, delegates to workflow APIs)
ironmill-bench                  (benchmark harness — keeps 2 Mach API unsafe sites, see below)
```

### Dependency Graph

```
ironmill-cli
├── ironmill-compile
│   ├── mil-rs
│   ├── ironmill-ane-sys        (for ANE compilation)
│   └── ironmill-coreml-sys     (for xcrun integration, optional)
├── ironmill-inference
│   ├── mil-rs                  (for IR types in TurboQuant MIL emission)
│   ├── ironmill-ane-sys        (for ANE runtime)
│   ├── ironmill-iosurface      (for ANE tensor I/O)
│   └── ironmill-coreml-sys     (for CoreML runtime)
└── mil-rs                      (for IR types, readers)

ironmill-bench
├── ironmill-compile            (for model compilation)
├── ironmill-inference          (for runtime, replaces direct ironmill-coreml/ironmill-ane deps)
└── mil-rs                      (for IR types)

burn-coreml
├── ironmill-compile            (uses CompileBuilder for ONNX → CoreML conversion)
└── ironmill-coreml-sys         (uses Model::load/predict for inference)

candle-coreml
├── ironmill-compile            (uses CompileBuilder for ONNX → CoreML conversion)
└── ironmill-coreml-sys         (uses Model::load/predict for inference)
```

**Key rules:**
- `mil-rs` depends on nothing in ironmill. It is the leaf.
- `burn-coreml` and `candle-coreml` use a narrow API surface: `CompileBuilder`,
  `Quantization`, `TargetComputeUnit` from compile; `Model`, `PredictionInput`,
  `ComputeUnits` from coreml-sys.

### ironmill-runtime disposition

The existing `ironmill-runtime` crate defines `RuntimeBackend`, `RuntimeModel`,
`RuntimeTensor`, `ElementType`, and `InputFeatureDesc`. These traits and types are
consumed by both `ironmill-ane` and `ironmill-coreml` today.

**Change:** Absorb `ironmill-runtime` into `ironmill-inference`. The
`InferenceEngine` trait (defined below) supersedes `RuntimeBackend`/`RuntimeModel`.
The tensor and input-description types move into `ironmill-inference` as shared types.
Delete `ironmill-runtime` as a separate crate once migration is complete.

---

## What Moves Where

### mil-rs → stays in mil-rs (independent MIL library)

These are genuinely reusable for anyone working with Apple's MIL/CoreML format:

| Current location | Lines | Purpose |
|---|---|---|
| `ir/program.rs` | 213 | Program, Function, Block |
| `ir/operation.rs` | 98 | Operation (minus ComputeUnit — see below) |
| `ir/tensor.rs` | 63 | ScalarType, TensorType |
| `ir/types.rs` | 37 | Value enum |
| `ir/graph.rs` | 57 | Graph container |
| `ir/pass.rs` | 13 | Pass trait |
| `ir/pipeline.rs` | 1255 | PassPipeline (refactored — see below) |
| `proto/mod.rs` | + generated | CoreML + ONNX protobuf types |
| `convert/proto_to_ir.rs` | 684 | Protobuf → IR |
| `convert/ir_to_proto.rs` | 2251 | IR → Protobuf |
| `convert/onnx_graph.rs` | 1931 | ONNX graph → IR |
| `convert/onnx_to_mil.rs` | 2253 | ONNX node → IR operation |
| `reader/mlmodel.rs` | 173 | .mlmodel reader |
| `reader/mlpackage.rs` | 189 | .mlpackage reader |
| `reader/onnx.rs` | 187 | ONNX reader |
| `writer/mlmodel.rs` | 87 | .mlmodel writer |
| `writer/mlpackage.rs` | 208 | .mlpackage writer |
| `analysis/` | ~7+ | FLOPs analysis |
| `error.rs` | 36 | Error types |

**Generic passes that stay in mil-rs** (no backend assumptions):

| Pass file | Lines | Purpose |
|---|---|---|
| `dead_code.rs` | 169 | Dead code elimination |
| `identity_elim.rs` | 235 | Identity/no-op removal |
| `constant_fold.rs` | 276 | Constant folding |
| `bn_weight_fold.rs` | 868 | BatchNorm → Conv weight folding |
| `op_fusion.rs` | 976 | Conv+BN, Linear+ReLU, LayerNorm+Linear, etc. |
| `attention_fusion.rs` | 1340 | SDPA fusion |
| `fp16_quantize.rs` | 342 | FP16 quantization |
| `int8_quantize.rs` | 608 | INT8 quantization |
| `palettize.rs` | 1095 | Weight palettization |
| `kmeans.rs` | 253 | K-means helper |
| `layout_optimize.rs` | 524 | Transpose cancellation |
| `type_repropagate.rs` | 528 | Type repropagation |
| `shape_materialize.rs` | 440 | Shape materialization (both generic and AR) |
| `tensor_utils.rs` | 80 | Shared tensor math |
| `beta_quantizer.rs` | 239 | Beta-optimal quantizer |
| `rotation.rs` | 253 | Hadamard rotation utilities |
| `polar_quantize.rs` | 537 | PolarQuant |
| `polar_rotation_fusion.rs` | 843 | PolarQuant rotation cleanup |

### mil-rs → moves to ironmill-compile

These are ironmill-specific and don't belong in a general MIL library:

| Current location | Lines | Destination |
|---|---|---|
| `ffi/ane.rs` | 1169 | `ironmill-ane-sys` |
| `ffi/mod.rs` | 17 | `ironmill-ane-sys` |
| `validate.rs` | 1355 | `ironmill-compile::ane::validate` |
| `build_api.rs` | 474 | `ironmill-compile::coreml::build_api` |
| `compiler.rs` | 156 | `ironmill-compile::coreml::compiler` |
| `c_api.rs` | 673 | `ironmill-compile::c_api` |
| `convert/ir_to_mil_text.rs` | 980 | `ironmill-compile::ane::mil_text` |
| `convert/lora.rs` | 902 | `ironmill-compile::convert::lora` |
| `convert/moe.rs` | 1604 | `ironmill-compile::convert::moe` |
| `convert/pipeline.rs` | 1000 | `ironmill-compile::convert::pipeline` |
| `convert/templates/*` | ~3376 | `ironmill-compile::templates::*` |
| `convert/weights/*` | ~2211 | `ironmill-compile::weights::*` |
| `reader/gguf.rs` | 11 | `ironmill-compile::weights::gguf` |
| `reader/safetensors.rs` | 12 | `ironmill-compile::weights::safetensors` |

**Passes that move to ironmill-compile** (ANE/backend-specific):

| Pass file | Lines | Destination |
|---|---|---|
| `compute_unit.rs` | 182 | `ironmill-compile::ane::passes` |
| `kv_cache.rs` | 669 | `ironmill-compile::ane::passes` |
| `layer_schedule.rs` | 775 | `ironmill-compile::ane::passes` |
| `mixed_precision.rs` | 1340 | `ironmill-compile::ane::passes` |
| `op_substitute.rs` | 776 | `ironmill-compile::ane::passes` |
| `op_split.rs` | 1196 | `ironmill-compile::ane::passes` |
| `model_split.rs` | 420 | `ironmill-compile::ane::passes` |
| `codebook.rs` | 918 | `ironmill-compile::ane::passes` |

*(Plus the already feature-gated ANE passes: AneLayoutPass, AneArgPromotionPass,
AneMatmulToConvPass, AneVariableNamingPass, AneConcatEliminationPass,
AttentionDecomposePass)*

### ironmill-ane → splits into ironmill-compile + ironmill-inference

| Current location | Lines | Destination |
|---|---|---|
| `split.rs` | 2498 | `ironmill-compile::ane::split` |
| `packing.rs` | 465 | `ironmill-compile::ane::packing` |
| `blobfile.rs` | 211 | `ironmill-compile::ane::blobfile` |
| `cache.rs` | 419 | `ironmill-compile::ane::cache` |
| `runtime.rs` | 685 | `ironmill-ane-sys` (FFI) + `ironmill-inference::ane::runtime` (safe layer) |
| `program.rs` | 86 | `ironmill-ane-sys` (handle types) |
| `tensor.rs` | 819 | `ironmill-iosurface` |
| `inference.rs` | 1846 | `ironmill-inference::ane::decode` |
| `turboquant.rs` | 680 | `ironmill-inference::ane::turboquant::model` |
| `turboquant_mil.rs` | 1259 | `ironmill-inference::ane::turboquant::mil_emitter` |
| `lib.rs` | 1322 | Split: compile parts → `ironmill-compile`, runtime parts → `ironmill-inference` |

### ironmill-coreml → ironmill-coreml-sys + ironmill-inference::coreml

| Current location | Lines | Destination |
|---|---|---|
| `lib.rs` (FFI parts) | ~340 | `ironmill-coreml-sys` (safe wrappers over ObjC) |
| `lib.rs` (runtime bridge) | ~110 | `ironmill-inference::coreml::runtime` |

---

## Unsafe Quarantine Strategy

### Current state: 310 unsafe sites across 5 crates

| Crate | Sites | Primary concerns |
|---|---|---|
| `mil-rs` | 174 | ANE FFI (134), C API (38), mmap (2) |
| `ironmill-ane` | 105 | ObjC runtime (73), IOSurface (18), handle wrapping (14) |
| `ironmill-coreml` | 29 | ObjC/CoreML FFI |
| `ironmill-bench` | 2 | Mach API FFI |

### Target state: unsafe confined to 3 `-sys` crates + 1 exception

| Crate | Unsafe | Safe API surface |
|---|---|---|
| `ironmill-ane-sys` | ~200 | `AneCompiler::compile()`, `AneRuntime::load/eval/unload()`, ObjC helpers |
| `ironmill-iosurface` | ~20 | `AneTensor::new/read/write/lock()`, `IOSurface` lifecycle |
| `ironmill-coreml-sys` | ~30 | `CoreMlModel::load/predict/extract()` |

**Exception:** `ironmill-bench` retains 2 unsafe sites for Mach `task_info` FFI
(RSS measurement in `inference.rs:103-142`). This is benchmark-only instrumentation
behind `#[cfg(target_os = "macos")]` — too narrow and domain-specific to justify a
`-sys` crate. These 2 sites get `// SAFETY:` comments and
`#[allow(unsafe_code)]` on the function, with the rest of the crate using
`#![deny(unsafe_code)]`.

**All other crates become `#![forbid(unsafe_code)]`.**

### Unsafe hygiene rules for -sys crates

1. Every `unsafe` block gets a `// SAFETY:` comment explaining the invariant
2. `#![deny(unsafe_op_in_unsafe_fn)]` enforced in all -sys crates
3. Every public function is safe — unsafe is only internal
4. Duplicated ObjC FFI patterns (currently in both `mil-rs/ffi/ane.rs` and
   `ironmill-ane/runtime.rs`) consolidated into `ironmill-ane-sys::objc`
5. Shared helpers: `msg_send_retained()`, `with_iosurface_locked()`,
   `extract_nserror()` replace ad-hoc transmute patterns

---

## IR Cleanup

### Remove ComputeUnit from Operation

`ComputeUnit` (`Ane`/`Gpu`/`Cpu`/`Any`) is currently a field on every `Operation`
(`ir/operation.rs:7-23`). This is a backend scheduling concern that pollutes the
generic IR.

**Change:** Remove `compute_unit` from `Operation`. Backend passes in
`ironmill-compile` produce a side-table `HashMap<String, ComputeUnit>` consumed by
backend-specific lowering. The `ComputeUnitAnnotationPass` moves to
`ironmill-compile::ane::passes`.

### Make the pass pipeline extensible

The current pipeline (`ir/pipeline.rs`, 1255 lines) hardcodes pass ordering, has
special-case methods (`with_polar_quant()`, `without_fusion()`), and mixes generic
and backend-specific passes.

**Change:**
- `mil-rs` defines `PassPipeline` with a default set of generic passes
- Backend crates register additional passes via `pipeline.add_pass(pass)` or
  `pipeline.add_pass_after("type_repropagate", ane_layout_pass)`
- Pass ordering constraints are declared, not hardcoded
- Backend-specific pipeline configuration moves to `ironmill-compile`

---

## Model Architecture Decoupling

### Current problem

The split and inference code uses name-pattern heuristics to detect model structure:
- Layer names: regex on `layer_0`, `block_12`, `layers.3` (`split.rs:161-190`)
- Attention ops: substring matching on `_q_reshape`, `_k_reshape`, `RotaryEmbedding`
  (`split.rs:716-775`)
- Fused attention: hardcoded op names (`scaled_dot_product_attention`, `sdpa`,
  `GroupQueryAttention`) (`split.rs:633-714`)
- Q/K/V identification: channel size + name substrings (`inference.rs:1543-1594`)
- Head dim fallback to 64 (`inference.rs:1180-1184`)
- Hardcoded EOS tokens: `[2, 151643, 128001]` (`inference.rs:1427-1435`)

### Solution

Define a `ModelArchitecture` config that explicitly describes:

```rust
pub struct ModelArchitecture {
    pub layers: Vec<LayerDescriptor>,
    pub attention: AttentionConfig,
    pub embedding: EmbeddingConfig,
    pub lm_head: LmHeadConfig,
    pub eos_tokens: Vec<u32>,
}

pub struct AttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub style: AttentionStyle, // SDPA, GQA, MHA
}
```

- Derived from model config files (config.json, tokenizer_config.json) when available
- Fallback `ModelArchitecture::infer_from_program(program)` preserves current
  heuristics as opt-in, **gated behind `#[deprecated]`** to discourage long-term use
  and encourage explicit config adoption
- Split and inference engines consume this struct instead of guessing

---

## Inference Engine Trait

Define a shared interface in `ironmill-inference` that both backends implement:

```rust
pub trait InferenceEngine {
    fn load(&mut self, artifacts: &CompiledArtifacts) -> Result<()>;
    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits>;
    fn decode_step(&mut self, token: u32) -> Result<Logits>;
    fn reset(&mut self);
}
```

Shared concerns above this trait:
- Token sampling (temperature, top-k, top-p)
- Stopping criteria (EOS tokens, max length)
- Streaming output
- Batch management

These currently live inside `AneInference::decode()` and should be extracted.

---

## Hardcoded Constants → Configuration

| Constant | Current location | Target |
|---|---|---|
| `MIN_IO_SEQ = 32` | `turboquant_mil.rs:15-20` | `AneHardwareProfile` |
| `ANE_MIN_SURFACE_BYTES = 16384` | `tensor.rs:71-80` | `AneHardwareProfile` |
| LM head chunk cap `16384` | `inference.rs:163-190` | `ModelArchitecture` |
| Compile budget `119` | `cache.rs:12-13` | `AneHardwareProfile` |
| QoS level `21` | `runtime.rs:35-37` | `AneHardwareProfile` |
| EOS tokens `[2, 151643, 128001]` | `inference.rs:1427-1435` | `ModelArchitecture` |

```rust
pub struct AneHardwareProfile {
    pub min_io_seq: usize,          // 32
    pub min_surface_bytes: usize,   // 16384
    pub compile_budget: usize,      // 119
    pub qos_level: u32,             // 21
}

impl Default for AneHardwareProfile { /* current values */ }
```

---

## Error Handling Standardization

| Crate | Error type | Rule |
|---|---|---|
| `mil-rs` | `MilError` (thiserror) | Typed errors for IR/conversion/IO |
| `ironmill-ane-sys` | `AneSysError` (thiserror) | FFI failures, ObjC errors |
| `ironmill-iosurface` | `IOSurfaceError` (thiserror) | Alloc/lock/copy failures |
| `ironmill-compile` | `CompileError` (thiserror) | Pass/split/emit failures |
| `ironmill-inference` | `InferenceError` (thiserror) | Runtime/decode failures |
| `ironmill-cli` | `anyhow::Result` | Application boundary only |

Eliminate bare `.unwrap()` and `.expect()` in library crates.

---

## CLI Decoupling

The CLI currently imports and directly orchestrates pass pipelines, reader selection,
MoE split/fuse, speculative decoding, TurboQuant configuration, ANE artifact
preparation, and xcrun invocation across 1336 lines.

**Change:** Define high-level workflow APIs:

```rust
// In ironmill-compile
pub fn compile_to_coreml(input: &Path, options: CoreMlOptions) -> Result<Package>;
pub fn compile_to_ane(input: &Path, options: AneOptions) -> Result<AneArtifacts>;
pub fn compile_pipeline(manifest: &Path, options: PipelineOptions) -> Result<PipelineOutput>;
```

The CLI becomes a thin argument parser delegating to these functions.

---

## Migration Order

Each phase has acceptance criteria that must pass before proceeding to the next.
All verification is local (`cargo build`, `cargo test`, manual ANE smoke tests) —
CI runners lack ANE hardware.

### Phase 1: Quarantine unsafe (no behavior change)

1. Create `ironmill-ane-sys` — extract and consolidate ANE FFI from
   `mil-rs/src/ffi/ane.rs` and `ironmill-ane/src/runtime.rs`
2. Create `ironmill-iosurface` — extract IOSurface FFI from
   `ironmill-ane/src/tensor.rs`
3. Create `ironmill-coreml-sys` — extract CoreML FFI from
   `ironmill-coreml/src/lib.rs`
4. Add `// SAFETY:` comments to all remaining unsafe blocks
5. Add `#![forbid(unsafe_code)]` to all non-sys crates (except `ironmill-bench`,
   which uses `#![deny(unsafe_code)]` with a per-function `#[allow]` for the 2
   Mach `task_info` FFI sites)

**Intermediate dependency state:** During this phase, `ironmill-ane` depends on
the new `ironmill-ane-sys` while still containing compile and inference code that
will move later. Its `Cargo.toml` temporarily adds `ironmill-ane-sys` alongside
existing `mil-rs` (which loses its `ffi/` module). The `ironmill-ane` → `mil-rs`
dependency narrows to IR types and passes only; direct ANE FFI routes through
`ironmill-ane-sys`. Similarly, `ironmill-coreml` adds `ironmill-coreml-sys` and
its `lib.rs` shrinks to re-exports + safe orchestration over the new sys crate.

**Acceptance criteria:**
- `cargo build --workspace` succeeds
- `cargo test --workspace` passes (unit tests)
- Local ANE smoke test: compile + run a small model end-to-end on Apple Silicon
- `grep -r "unsafe" crates/{mil-rs,ironmill-ane,ironmill-coreml}/src/` returns 0 matches
- `-sys` crates have `#![deny(unsafe_op_in_unsafe_fn)]`

### Phase 2: Clean mil-rs (make it independent)

6. Remove `ComputeUnit` from `Operation` → side-table
7. Move ANE-specific passes out of `mil-rs/src/ir/passes/` (currently feature-gated)
8. Move `ffi/`, `validate.rs`, `c_api.rs`, `build_api.rs`, `compiler.rs` out
9. Move `convert/ir_to_mil_text.rs`, `convert/templates/`, `convert/weights/`,
   `convert/lora.rs`, `convert/moe.rs`, `convert/pipeline.rs` out
10. Move weight-format readers (`reader/gguf.rs`, `reader/safetensors.rs`) out
11. Verify `mil-rs` has zero ironmill dependencies and can build standalone
12. Refactor `PassPipeline` to be extensible (backend passes register externally)

**Acceptance criteria:**
- `mil-rs` builds with no workspace dependencies (`cargo build -p mil-rs`)
- `mil-rs` has no feature flags referencing ironmill or ANE
- `cargo test -p mil-rs` passes
- `cargo build --workspace` still succeeds
- Local ANE smoke test still passes

### Phase 3: Create ironmill-compile

13. Create `ironmill-compile` crate with all code extracted from mil-rs + ironmill-ane
    compilation paths
14. Move model splitting (`split.rs`), packing (`packing.rs`), blobfile writing from
    ironmill-ane
15. Move compile orchestration from `ironmill-ane/src/lib.rs` (AneModel::compile_and_load,
    CompiledArtifacts::prepare)
16. Define workflow APIs (compile_to_coreml, compile_to_ane, compile_pipeline)
17. Update burn-coreml and candle-coreml to use ironmill-compile. Both crates use a
    narrow builder API surface (`CompileBuilder::new()`, `.output()`, `.quantize()`,
    `.target()`, `.input_shape()`, `.palettize()`, `.compile()`, `.build()`). Migration
    is an import-path change from `mil_rs::build_api::*` to
    `ironmill_compile::coreml::build_api::*`, plus verifying that `BuildOutput` field
    names (`mlpackage`, `mlmodelc`) are preserved.

**Acceptance criteria:**
- `cargo build --workspace` succeeds
- `cargo test --workspace` passes
- `burn-coreml` and `candle-coreml` build and their tests pass
- Local ANE smoke test: compile path produces identical artifacts to pre-migration

### Phase 4: Create ironmill-inference

18. Define `InferenceEngine` trait
19. Absorb `ironmill-runtime` types (`RuntimeTensor`, `ElementType`,
    `InputFeatureDesc`) into `ironmill-inference` as shared types
20. Extract ANE decode loop from `ironmill-ane/src/inference.rs`
21. Extract TurboQuant into subcrate/submodule
22. Extract CoreML inference from `ironmill-coreml`
23. Extract sampling/stopping logic into shared module
24. Introduce `ModelArchitecture` config, migrate heuristics
25. Introduce `AneHardwareProfile`, migrate hardcoded constants
26. Delete `ironmill-runtime` crate (superseded by `ironmill-inference`)

**Acceptance criteria:**
- `cargo build --workspace` succeeds
- `cargo test --workspace` passes
- Local ANE smoke test: full prefill + decode loop produces identical output
- Local CoreML smoke test: CoreML inference path still works
- `ironmill-runtime` no longer in workspace members

### Phase 5: CLI and polish

27. Refactor CLI to use workflow APIs from ironmill-compile
28. Standardize error types across all crates
29. Delete `ironmill-ane` and `ironmill-coreml` crates (replaced by new structure)
30. Update all documentation

**Acceptance criteria:**
- `cargo build --workspace` succeeds with no warnings
- `cargo test --workspace` passes
- `cargo clippy --workspace` clean
- CLI produces identical output for all existing subcommands
- Local ANE + CoreML end-to-end smoke tests pass

---

## Testing Strategy

CI runners do not have ANE hardware. Testing is split into two tiers:

### CI (all platforms)
- `cargo build --workspace` — compilation check
- `cargo test --workspace` — unit tests (mocked hardware where needed)
- `cargo clippy --workspace` — lint
- `mil-rs` standalone build (`cargo build -p mil-rs`) — independence check

### Local (macOS with Apple Silicon)
- ANE smoke test: compile a small model → load → eval → verify output
- CoreML smoke test: ONNX → CoreML conversion → `Model::load` → predict
- Full decode loop: prefill + autoregressive decode with known input/output pair
- Benchmark regression: `ironmill-bench` latency within 5% of baseline

A `scripts/local-smoke-test.sh` script should be maintained that runs the local
tier. Contributors run this before submitting PRs that touch compilation or
inference paths.

---

## Line Count Summary

| Current crate | Lines | Fate |
|---|---|---|
| `mil-rs` | ~38,000 | ~17,000 stays (IR, proto, ONNX, readers/writers, generic passes); ~21,000 moves out |
| `ironmill-ane` | ~10,300 | ~4,500 → ironmill-compile; ~4,000 → ironmill-inference; ~1,800 → sys crates |
| `ironmill-coreml` | ~550 | ~340 → ironmill-coreml-sys; ~210 → ironmill-inference |
| `ironmill-runtime` | ~160 | Absorbed into ironmill-inference; crate deleted |
| `ironmill-cli` | ~1,336 | Refactored in place (shrinks significantly) |
| `ironmill-bench` | ~1,200 | Stays; deps updated to new crates; 2 unsafe sites retained |
| **New: ironmill-ane-sys** | ~2,000 | Consolidated unsafe FFI |
| **New: ironmill-iosurface** | ~850 | IOSurface safe wrappers |
| **New: ironmill-coreml-sys** | ~350 | CoreML safe wrappers |
| **New: ironmill-compile** | ~25,500 | ANE passes + split + templates + weights + build API |
| **New: ironmill-inference** | ~4,400 | Decode loops + TurboQuant + engine trait + runtime types |
