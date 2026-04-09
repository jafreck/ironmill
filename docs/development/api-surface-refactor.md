# API Surface Refactor — ironmill-compile & ironmill-inference

ironmill is unreleased. There should be no deprecated modules, no compatibility
shims, and no stub APIs in the public surface. This document specifies the
changes needed to make both crate APIs narrow, clean, and idiomatic.

---

## Part 1 — ironmill-compile

### 1.1 Delete the `mil` re-export façade

`src/lib.rs:33–112` re-exports ~45 symbols from `mil_rs`, including two
`#[deprecated]` sub-modules (`mil::proto`, `mil::passes`) that exist solely for
backward compatibility with ironmill-compile-ffi and ironmill-bench.

There is no backward compatibility to maintain. Delete the entire `pub mod mil { … }` block.

**Consumers to update:**

| Crate | Current import | Change to |
|---|---|---|
| ironmill-cli | `ironmill_compile::mil::{PassPipeline, Program, …}` | `mil_rs::{PassPipeline, Program, …}` |
| ironmill-bench | `ironmill_compile::mil::{Pass, Program, …}` | `mil_rs::{Pass, Program, …}` |
| ironmill-bench (quality.rs) | `ironmill_compile::mil::passes::{PolarQuantPass, tensor_utils}` | `mil_rs::ir::passes::{PolarQuantPass, tensor_utils}` |
| ironmill-bench (inference_e2e.rs) | `ironmill_compile::mil::passes::{…}` | `mil_rs::ir::passes::{…}` |
| ironmill-compile-ffi | `ironmill_compile::mil::proto::specification::Model` | `mil_rs::proto::specification::Model` |
| ironmill-compile-ffi | `ironmill_compile::mil::proto::onnx::ModelProto` | `mil_rs::proto::onnx::ModelProto` |
| ironmill-compile-ffi | `ironmill_compile::mil::{read_onnx, read_mlmodel, …}` | `mil_rs::{read_onnx, read_mlmodel, …}` |
| ironmill-compile (gpu doc comment) | `ironmill_compile::mil::PassPipeline` | `mil_rs::PassPipeline` |

Each consumer crate that doesn't already list `mil-rs` as a dependency needs
it added to its `Cargo.toml`. The workspace already has `mil-rs` as a workspace
dependency, so this is `mil-rs.workspace = true`.

After deletion, `ironmill-compile::mil` no longer exists. The `Granularity`,
`AffineQuantizePass`, and `BitWidth` re-exports currently at lines 98–99
are only used internally by the GPU compilation path — move those to
`use mil_rs::…` inside `src/gpu/mod.rs`.

The `mil::convert` re-exports of LoRA/MoE helpers are used by ironmill-cli.
Update ironmill-cli to import from `mil_rs::convert::lora` and
`mil_rs::convert::moe` directly.

### 1.2 Delete unimplemented `CompileTarget` stubs

`src/compile_target.rs:89–131` defines `MetalCompileTarget` and
`CoremlCompileTarget`, both of which return `Err("not yet implemented")`.

Delete both structs. Keep the `CompileTarget` trait and its supporting types
(`CompileConfig`, `CompileOutput`, `ArtifactMetadata`) — these are
well-designed and will be implemented when real backends land. Make the trait
and types `pub` but don't provide any implementations yet.

### 1.3 Narrow the ANE internal surface

The `ane` module exposes implementation details that only the bundle compiler
uses. No external consumer constructs a `CpuWeight`, calls
`replace_gather_with_inputs`, or manipulates a `ProgramKey` directly.

**Make `pub(crate)`:**

| Module | Items to hide |
|---|---|
| `ane::decode_compile` | Entire module — entry point is `ane::bundle::compile_decode_bundle` |
| `ane::packing` | Entire module — only called by bundle compilation |
| `ane::blobfile` | Entire module — only called by bundle save |
| `ane::cache` | `ProgramKey`, `make_key`, `disk_path_for`, `record_compilation`, `remaining_budget` |

**Keep public:**
- `ane::bundle::{compile_model_bundle, compile_decode_bundle, AneModelBundle, AneDecodeBundle, AneCompileConfig, AneDecodeConfig}`
- `ane::validate::{validate_ane_compatibility, ValidationReport, OpReport}`
- `ane::split::{split_for_ane, ModelSplit, SplitConfig}`
- `ane::passes::*` (pass structs implement `Pass`, consumers add them to pipelines)
- `ane::{AneCompileError, Result}`

For `ane::cache::ProgramCache` — keep `new`, `get`, `insert`, `contains`,
`len`, `is_empty` public. Hide the rest.

For `ane::split::SubProgram` — keep the struct public (bundle consumers need it)
but consider whether its fields need to be `pub` or can be accessed via methods.

### 1.4 Hide `ane::passes` internals

In `ane::passes::layer_schedule`:
- `DetectedLayer`, `LayerType`, `detect_layers` → `pub(crate)`. These are
  helpers for `LayerSchedulePass`, not consumer API.

### 1.5 Re-export entry points at crate root

After removing the `mil` façade, the crate root exports only modules. Add
targeted re-exports of the two primary entry points:

```rust
// src/lib.rs — after module declarations
pub use coreml::build_api::CompileBuilder;
pub use gpu::GpuCompileBuilder;
pub use error::{CompileError, Result};
pub use weights::{SafeTensorsProvider, GgufProvider, WeightProvider, ModelConfig};
pub use templates::weights_to_program;
```

This gives consumers a one-import path for the common workflow:
```rust
use ironmill_compile::{GpuCompileBuilder, SafeTensorsProvider};
```

### 1.6 Clean up `weights::quantized` constructors

`AffineQuantConfig` has 8 constructor methods that form a combinatorial
explosion (`int4`, `int4_awq`, `int4_awq_with_activations`,
`int4_awq_block`, `int4_gptq`, `int4_awq_gptq`, `int4_with_sensitive`).

Replace with a builder:
```rust
impl AffineQuantConfig {
    pub fn int4(group_size: usize) -> Self { … }

    pub fn with_awq(self, magnitudes: …, activations: …, token_count: usize) -> Self { … }
    pub fn with_gptq(self, hessian: …, block_size: usize, dampening: f32) -> Self { … }
    pub fn with_sensitive_layers(self, layers: HashSet<String>) -> Self { … }
    pub fn with_block_config(self, config: AwqBlockConfig) -> Self { … }
}
```

One base constructor, chainable refinements. The AWQ+GPTQ combination is
just `.with_awq(…).with_gptq(…)`.

---

## Part 2 — ironmill-inference

### 2.1 Make unfinished modules `pub(crate)`

Three modules are stubs or explicitly documented as future work:

| Module | Evidence | Action |
|---|---|---|
| `batch_runner` | All fields prefixed `_`, all methods return `Err("not yet implemented")` or hardcoded values | `pub(crate)` |
| `jit` | Doc comment says "Actual GPU kernel compilation is future work"; `with_int4`, `with_fp16`, `with_polar_quant` all return `Err(Unsupported)` | `pub(crate)` |
| `shader_cache` | Internal GPU plumbing, no external consumer | `pub(crate)` |
| `turboquant` | Internal quantization runtime, no external consumer | `pub(crate)` |

Remove corresponding `pub use` lines from `lib.rs` (line 43 for BatchRunner,
etc.).

### 2.2 Make internal infrastructure modules `pub(crate)`

These modules are implementation details of the engine, not consumer API:

| Module | Reason |
|---|---|
| `serving` (`KvPool`, `BatchScheduler`, `InferenceBatch`, `SequenceState`) | Engine-internal batch scheduling plumbing |
| `calibration` (`AwqActivationStore`, `GptqActivationStore`, `ActivationHook`) | Engine-internal calibration hooks |

If `calibration` types are needed by ironmill-bench's AWQ examples, expose a
narrow public surface: `pub use calibration::{CalibrationRunner, CalibrationDataset}`.
Hide the stores and hooks.

### 2.3 Delete the `model_info` re-export

`src/model_info.rs` is a single-line deprecated re-export:
```rust
#[deprecated(note = "Import from ironmill_core::model_info instead")]
pub use ironmill_core::model_info::ModelInfo;
```

Delete the file. Delete `pub mod model_info` and `pub use model_info::ModelInfo`
from `lib.rs`. Consumers already get `ModelInfo` from `ironmill_core`.

### 2.4 Reduce crate-root re-exports

The current `lib.rs` lines 40–67 re-export 50+ symbols at the crate root.
Replace with a focused set:

```rust
// Core engine
pub use engine::{InferenceEngine, BatchInferenceEngine, InferenceError, SequenceId};

// Generation (primary high-level API)
pub use generate::{
    GenerateRequest, GenerateEvent, GenerateResult, FinishReason,
    TokenStream, CancellationToken,
    generate, generate_with_callback,
};
#[cfg(feature = "async")]
pub use generate::generate_async;

// Sampling
pub use sampling::{Sampler, SamplerConfig};

// Types
pub use types::{Logits, ElementType, RuntimeTensor};

// Memory estimation
pub use memory::{MemoryEstimator, MemoryUsage, QuantLevel};
```

**Removed from root** (still accessible via module path):

| Symbol | Access via |
|---|---|
| `BatchRunner`, `BatchRunnerConfig`, `SchedulingPolicy`, `SequenceHandle` | Hidden (stub) |
| `KvCacheSlice`, `KvLayerSlice`, `LinearPrefixCache`, `LruPolicy`, `PrefixCache`, `RadixTree` | `ironmill_inference::cache::*` |
| `ConstrainedDecoder`, `prefill_with_cache` | `ironmill_inference::engine::*` |
| `CompiledGrammar`, `GrammarState`, `TokenMask` | `ironmill_inference::grammar::*` |
| `DraftCandidate`, `DraftHead`, `MsaHeadWeights`, `SpecConfig`, `SpeculativeEngine`, `SpeculativeStreaming`, `StreamingConfig`, `speculative_decode` | `ironmill_inference::speculative::*` |
| `DEFAULT_EOS_TOKENS`, `SamplingError`, `apply_token_mask`, `is_eos_token`, `sample_token` | `ironmill_inference::sampling::*` |
| `InputFeatureDesc`, `RuntimeBackend`, `RuntimeModel` | `ironmill_inference::types::*` |
| `KvQuantLevel` | `ironmill_inference::memory::*` |
| `GenerateError` | `ironmill_inference::generate::GenerateError` |
| `AneConfig`, `AneDirectBackend`, `AneModel`, `AneRuntimeModel` | `ironmill_inference::ane::model::*` |

### 2.5 Stabilize `coreml_runtime`

`src/lib.rs:78–186` defines `coreml_runtime` as `#[doc(hidden)]` but
`burn-coreml` and `candle-coreml` depend on it. It contains useful abstractions
(`CoreMlSession`, `SessionOutput`, `SessionInputDesc`).

Remove `#[doc(hidden)]`. The module is feature-gated behind
`#[cfg(all(feature = "coreml", target_os = "macos"))]` which is sufficient
access control. Add proper module-level documentation explaining this is the
shared CoreML inference session API.

### 2.6 Make `Logits` a newtype

`pub type Logits = Vec<f32>` in `types.rs` provides no type safety.

```rust
/// Raw (unnormalized) logit scores from model inference.
#[derive(Debug, Clone)]
pub struct Logits(Vec<f32>);

impl Logits {
    pub fn new(data: Vec<f32>) -> Self { Self(data) }
    pub fn into_inner(self) -> Vec<f32> { self.0 }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
}

impl std::ops::Deref for Logits {
    type Target = [f32];
    fn deref(&self) -> &[f32] { &self.0 }
}

impl std::ops::DerefMut for Logits {
    fn deref_mut(&mut self) -> &mut [f32] { &mut self.0 }
}
```

The `Deref`/`DerefMut` impls mean existing code using `&logits[..]` or
`logits.iter()` continues to work. Only code constructing `Logits` from a
raw `Vec<f32>` (engine implementations) needs to wrap with `Logits::new()`.

### 2.7 Make `SamplingError` use thiserror

`SamplingError` manually implements `Display` and `std::error::Error`. Every
other error type in the crate uses `thiserror`. Align it:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SamplingError {
    #[error("logits slice must not be empty")]
    EmptyLogits,
    #[error("no valid tokens remain after filtering")]
    EmptyDistribution,
}
```

---

## Part 3 — Cross-cutting

### 3.1 Add crate-level doc examples

Neither crate has a usage example in its top-level `//!` doc comment. Add
end-to-end examples showing the intended workflow.

**ironmill-compile `src/lib.rs`:**
```rust
//! # Example: compile SafeTensors → CoreML mlpackage
//!
//! ```no_run
//! use ironmill_compile::{CompileBuilder, SafeTensorsProvider};
//!
//! let provider = SafeTensorsProvider::load("model/")?;
//! let output = CompileBuilder::new("model.onnx")
//!     .quantize(ironmill_compile::coreml::build_api::Quantization::Fp16)
//!     .compile(true)
//!     .build()?;
//! # Ok::<(), ironmill_compile::CompileError>(())
//! ```
```

**ironmill-inference `src/lib.rs`:**
```rust
//! # Example: generate tokens
//!
//! ```ignore
//! use ironmill_inference::{GenerateRequest, TokenStream, CancellationToken};
//!
//! let cancel = CancellationToken::new();
//! let request = GenerateRequest::new(prompt_tokens);
//! let stream = TokenStream::new(&mut engine, request, &cancel);
//!
//! for event in stream {
//!     match event? {
//!         GenerateEvent::Token { token, .. } => print!("{token}"),
//!         GenerateEvent::Finished { .. } => break,
//!         _ => {}
//!     }
//! }
//! ```
```

### 3.2 `#[non_exhaustive]` sweep

Ensure every public struct with `pub` fields and every public enum has
`#[non_exhaustive]`. Candidates missing it:

- `ane::split::SplitConfig`
- `ane::decode_compile::CacheWriteConfig` (becomes `pub(crate)` per §1.3)
- `ane::blobfile::BlobEntry` (becomes `pub(crate)` per §1.3)
- `speculative::config::TurboSpecConfig`
- `serving::batch::InferenceBatch` (becomes `pub(crate)` per §2.2)
- `calibration::ChannelMagnitudes` (becomes `pub(crate)` per §2.2)

---

## Execution order

These changes form a dependency chain. Recommended order:

1. **ironmill-compile §1.1** — Delete `mil` façade, update all consumers to
   import from `mil_rs` directly. This is the largest diff but purely
   mechanical (find-and-replace import paths).

2. **ironmill-inference §2.1 + §2.2 + §2.3** — Hide stubs and internals,
   delete `model_info`. No consumer breakage since these types aren't used
   externally.

3. **ironmill-inference §2.4** — Reduce root re-exports. Consumers using
   module paths (`ironmill_inference::speculative::SpecConfig`) are unaffected;
   those using root imports need to add the module prefix.

4. **ironmill-compile §1.2 + §1.3 + §1.4 + §1.5** — Remove stubs, narrow
   ANE surface, add root re-exports.

5. **ironmill-inference §2.5 + §2.6 + §2.7** — Stabilize coreml_runtime,
   Logits newtype, SamplingError thiserror.

6. **Both §3.1 + §3.2** — Doc examples and `#[non_exhaustive]` sweep.

7. **ironmill-compile §1.6** — AffineQuantConfig builder (optional, lower
   priority).

Steps 1–4 can each be a separate commit. Steps 5–7 are independent of
each other and can be done in any order.

---

## Review Notes (2026-04-09)

Verified every claim in this document against the current codebase.
The proposal is well-structured and the overall direction is correct.
Below are corrections and issues that need resolution before implementation.

### Fix 1 — §1.4: `LayerType` must stay public

**Problem:** `LayerScheduleConfig::strategies` is typed
`HashMap<LayerType, OpPrecision>` and both are re-exported from
`ane::passes::mod.rs:30`. Hiding `LayerType` makes the public config
unusable.

**Fix:** Narrow the hiding to only the two truly-internal items:

```rust
// ane/passes/layer_schedule.rs

pub(crate) struct DetectedLayer { … }
pub(crate) fn detect_layers(ops: &[Operation]) -> Vec<DetectedLayer> { … }

// LayerType stays `pub` — it's part of LayerScheduleConfig's contract.
```

Update the §1.4 table to remove `LayerType` from the "hide" list.

---

### Fix 2 — §1.3: `ProgramKey` / `ProgramCache` visibility

**Problem:** `ProgramCache::get`, `insert`, `contains`, and `make_key`
all take or return `ProgramKey`. Hiding `ProgramKey` while keeping
`ProgramCache` public breaks the API.

**Fix:** Hide `ProgramKey` and absorb it into the cache methods. The key
is a trivial `(u64, u64)` pair — callers don't need to see it:

```rust
// ane/cache.rs

/// Opaque handle returned by `make_key`. Not constructable outside this module.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ProgramKey(pub(crate) u64, pub(crate) u64);

impl ProgramCache {
    /// Build a cache key from MIL text and weight data.
    pub fn make_key(mil_text: &str, weight_data: &[u8]) -> ProgramKey { … }

    pub fn get(&mut self, key: &ProgramKey) -> Option<&PathBuf> { … }
    pub fn insert(&mut self, key: ProgramKey, disk_path: PathBuf) { … }
    pub fn contains(&self, key: &ProgramKey) -> bool { … }
    pub fn disk_path_for(&self, key: &ProgramKey) -> Option<PathBuf> { … }

    // These become pub(crate):
    pub(crate) fn record_compilation(&mut self) { … }
    pub(crate) fn remaining_budget(&self) -> usize { … }
}
```

`ProgramKey` stays `pub` as a type (callers can hold it, pass it back),
but its fields become `pub(crate)` so it's opaque to external code.
`make_key` is the only constructor. `record_compilation` and
`remaining_budget` become `pub(crate)` as the proposal intended.

---

### Fix 3 — §1.2: `CompileTarget::estimate_size` sentinel

**Problem:** `Ok(0)` is indistinguishable from "zero-byte artifact".

**Fix:** Change the default return to `Option<usize>`:

```rust
fn estimate_size(
    &self,
    _source: &dyn mil_rs::weights::WeightProvider,
    _config: &CompileConfig,
) -> Result<Option<usize>, CompileError> {
    Ok(None) // default: unknown
}
```

Since the stubs are being deleted in this same step, no existing
implementations need updating — only the trait definition and any
future implementors.

---

### Fix 4 — §2.1: `turboquant` used by in-crate example

**Problem:** `examples/ane_op_eval.rs:1001` imports
`ironmill_inference::turboquant::codebook::lloyd_max_gaussian`. Rust
examples are separate compilation units and cannot access `pub(crate)`.

**Fix:** The turboquant functions used by the example
(`lloyd_max_gaussian`, `TurboQuantConfig`, `mil_emitter::*`) are
internal pipeline details being tested — this is a test masquerading
as an example. Move the affected test functions from `ane_op_eval.rs`
into `#[cfg(test)]` modules within the crate source (e.g.
`ane/turboquant/mod.rs`), where `pub(crate)` items are accessible.

Then root `turboquant` becomes `pub(crate)` cleanly. No re-exports,
no `#[doc(hidden)]`.

---

### Fix 5 — §2.2: calibration public surface too narrow

**Problem:** ironmill-bench's `awq_block_calibrate.rs` implements
`ActivationHook` (trait) and calls
`CalibratingEngine::prefill_with_hooks`. The proposal's narrow surface
(`CalibrationRunner`, `CalibrationDataset`) omits these.

**Fix:** Expand the public surface to include the traits:

```rust
// calibration/mod.rs  (after changes)
mod awq_store;
pub mod dataset;
mod gptq_store;
mod hook;
mod runner;

// Public API
pub use dataset::CalibrationDataset;
pub use hook::ActivationHook;
pub use runner::{CalibratingEngine, CalibrationRunner};

// These become pub(crate) — only used by runner/stores internally:
pub(crate) use awq_store::{AwqActivationStore, ChannelMagnitudes};
pub(crate) use gptq_store::{GptqActivationStore, HessianAccumulator};
pub(crate) use runner::{HessianHook, QuipHessianAccumulator};
```

The four public items (`CalibrationDataset`, `ActivationHook`,
`CalibratingEngine`, `CalibrationRunner`) are the minimal set needed
by ironmill-bench.

---

### Fix 6 — §1.1: missing consumers in migration table

**Problem:** The table omits `ironmill-bench/src/compiler.rs` (heavy
user, ~15 import sites) and `ironmill-bench/src/inference.rs`.

**Fix:** Add these rows to the §1.1 table:

| Crate | Current import | Change to |
|---|---|---|
| ironmill-bench (compiler.rs) | `ironmill_compile::mil::{Pass, Program, PassPipeline, PipelineReport, …}` | `mil_rs::{Pass, Program, PassPipeline, PipelineReport, …}` |
| ironmill-bench (inference.rs) | `ironmill_compile::mil::Program` | `mil_rs::Program` |

Note: ironmill-bench already has `mil-rs` as a workspace dependency
(`Cargo.toml:18`), so no Cargo.toml change needed for it. Only
ironmill-cli and ironmill-compile-ffi need `mil-rs.workspace = true`
added.

---

### Fix 7 — §3.2: `TurboSpecConfig` already annotated

**Problem:** The proposal lists `TurboSpecConfig` as missing
`#[non_exhaustive]`, but `speculative/config.rs:34` already has it.
`SpecConfig` at line 7 also already has it.

**Fix:** Remove `TurboSpecConfig` from the §3.2 candidates list.
The remaining genuine candidates after all other changes are:

- `ane::split::SplitConfig` — needs `#[non_exhaustive]` added

The others (`CacheWriteConfig`, `BlobEntry`, `InferenceBatch`,
`ChannelMagnitudes`) all become `pub(crate)` per earlier steps, so
`#[non_exhaustive]` is moot for them.

---

### Fix 8 — §2.5: missing `coreml_runtime` consumer

**Problem:** The §2.5 prose only mentions burn-coreml and candle-coreml.

**Fix:** Add ironmill-bench to the list of consumers:

- `ironmill-bench/src/inference.rs:5,193`
- `ironmill-bench/src/suites/coreml.rs:52`
- `ironmill-bench/tests/inference_e2e.rs:14`

These all import `CoreMlSession`, `SessionOutput`, or `ComputeUnits`
from `coreml_runtime`. They will benefit from the `#[doc(hidden)]`
removal — no import changes needed.

---

### Fix 9 — §1.2: stub error message references nonexistent type

**Problem:** `MetalCompileTarget`'s error says "Use GpuCompileTarget
for GPU bundle output" but `GpuCompileTarget` doesn't exist (the real
entry point is `GpuCompileBuilder`).

**Fix:** Moot — both stubs are being deleted. No action needed beyond
deletion.
