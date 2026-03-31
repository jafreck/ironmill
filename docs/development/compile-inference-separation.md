# Compile / Inference Separation

## Problem

`ironmill-inference` depends on `ironmill-compile` because the ANE-direct
backend does JIT compilation at model load time — running IR passes, splitting
programs, emitting MIL text, and writing weight blobs. The CoreML backend has
no such dependency. Compile and inference should be two totally separate
systems: compile produces artifacts, inference consumes them.

## Current State

Two code paths create the coupling:

**Simple path** (`AneModel::compile_and_load` in `ane/model.rs`):
Program → 5 passes → `split_for_ane` → `program_to_mil_text` → ANE compile →
load. Already has a `CompiledArtifacts::prepare()` API that the CLI uses.

**Decode path** (`AneInference::compile` in `ane/decode.rs`):
Program → architecture detection → autoregressive passes → gather replacement →
attention-aware split → per-sub-program packing/matmul→conv/DCE → cache-write
fusion → donor/patch compilation → full decode engine setup. ~2000 lines,
deeply intertwined with compile-crate types.

### What inference imports from compile

| Category | Items |
|----------|-------|
| Passes | `OpSubstitutionPass`, `AneLayoutPass`, `AttentionDecomposePass`, `AneConcatEliminationPass`, `AneVariableNamingPass`, `AneArgPromotionPass`, `AneMatmulToConvPass` |
| Split | `SplitConfig`, `split_for_ane`, `SubProgram`, `ModelSplit` |
| Emission | `MilTextConfig`, `program_to_mil_text`, `BlobFileWriter`, `WeightBlobEntry` |
| Packing | `InputPacking`, `pack_inputs`, `write_packed_inputs` |
| Cache | `ProgramCache`, `ProgramKey` |
| Types | `TensorDescriptor`, `AneCompileError` |

### Key observation

None of the decode-path passes require runtime information. They use
compile-time decisions (seq_len=1, TurboQuant config, split strategy). The
CLI already proves this — `CompiledArtifacts::prepare()` runs passes without
any ANE runtime. These are compile passes that run at load time because
there's no compile API that accepts "compile for autoregressive decode" as
a parameter.

## Approach

Create a clean artifact boundary: `ironmill-compile` produces self-contained
bundles that `ironmill-inference` can load and run without touching IR passes,
splitting, or MIL emission. Shared types go in `mil-rs` (the common
foundation).

## Phase 1: Move shared types to mil-rs

`TensorDescriptor` is `{ name: String, shape: [usize; 4], dtype: ScalarType }`
— generic enough for the foundation crate. `InputPacking` is
`{ offsets: Vec<usize>, sizes: Vec<usize> }` — also simple.

- Move `TensorDescriptor` from `ironmill-compile::ane` → `mil_rs::ir`
- Move `InputPacking` from `ironmill-compile::ane::packing` → `mil_rs::ir`
- Re-export from `ironmill-compile::ane` for backward compat
- Update all import sites

## Phase 2: Define artifact bundle types in ironmill-compile

Create `ironmill_compile::ane::bundle` module:

```rust
/// Compiled ANE model — ready for runtime loading.
pub struct AneModelBundle {
    pub sub_programs: Vec<SubProgramBundle>,
}

/// Single ANE sub-program artifact.
pub struct SubProgramBundle {
    pub name: String,
    pub mil_text: String,
    pub weight_blob: Vec<u8>,
    pub inputs: Vec<TensorDescriptor>,
    pub outputs: Vec<TensorDescriptor>,
    pub input_packing: Option<InputPacking>,
}

/// Compiled autoregressive decode model.
pub struct AneDecodeBundle {
    pub architecture: ModelArchitecture,
    pub rope_cos: Vec<u8>,
    pub rope_sin: Vec<u8>,
    pub embedding_weights: Vec<u8>,
    pub lm_head: LmHeadBundle,
    pub final_norm_weight: Option<Vec<u8>>,
    pub layers: Vec<LayerBundle>,
}

pub struct LayerBundle {
    pub index: usize,
    pub pre_attn: SubProgramBundle,
    pub post_attn: SubProgramBundle,
    pub fp16_attn: Option<SubProgramBundle>,
    pub cache_write_fused: bool,
}
```

Add serialization (save/load to disk) so artifacts can be cached or
distributed.

## Phase 3: Create high-level compilation APIs in ironmill-compile

Move all compilation orchestration into compile:

**Simple path:**
```rust
/// ironmill_compile::ane::bundle
pub fn compile_model_bundle(
    program: &Program,
    config: &AneCompileConfig,
) -> Result<AneModelBundle>
```
Absorbs the pass pipeline + split + MIL emission + packing from
`inference::ane::model::compile_and_load`.

**Decode path:**
```rust
pub fn compile_decode_bundle(
    program: &Program,
    config: &AneDecodeConfig,
) -> Result<AneDecodeBundle>
```
Absorbs the decode setup from `inference::ane::decode::compile`:
- Autoregressive shape materialization
- Gather replacement
- Attention-aware splitting
- Per-sub-program packing, matmul→conv, DCE
- Cache-write fusion (TurboQuant)
- CPU weight extraction (embedding, lm_head, final norm)
- Architecture detection and RoPE cache computation

The passes (`AutoregressiveShapeMaterializePass`, etc.) are MIL IR
transformations — they belong in compile even though they're
inference-mode-specific.

## Phase 4: Refactor ironmill-inference to consume bundles

Replace JIT compilation with bundle consumption:

**Simple path:**
```rust
impl AneModel {
    pub fn from_bundle(bundle: AneModelBundle, config: AneConfig) -> Result<Self>
}
```
Takes pre-compiled bundle → ANE `compile_mil_text` → load → allocate tensors.

**Decode path:**
```rust
impl AneInference {
    pub fn from_bundle(bundle: AneDecodeBundle, turbo: Option<TurboQuantConfig>) -> Result<Self>
}
```
Takes pre-compiled bundle → load sub-programs (with donor/patch optimization)
→ set up caches → ready for decode loop.

Remove `ironmill-compile` from inference's `Cargo.toml`. The convenience
wrappers (`compile_and_load`, `AneInference::compile`) move to CLI/bench
as orchestration code, or are removed.

## Phase 5: Update dependent crates and README

**ironmill-cli:** Switch from `CompiledArtifacts::prepare()` to
`compile_model_bundle()`. For decode, use `compile_decode_bundle()`.

**ironmill-bench:** ANE-direct benchmark calls compile API to get bundle,
then inference API to load/run. Bench orchestrates both crates.

**README Mermaid diagram:** Remove `inference --> compile` edge. Add
`inference --> mil` edge (already a real dependency, currently missing
from diagram).

Updated crate dependency graph after separation:

```
cli --> compile
cli -.-> inference

bench --> compile
bench --> inference
bench -.-> ios

burn --> compile
burn --> inference

candle --> compile
candle --> inference

compile --> mil
compile --> ios

inference --> mil
inference --> ane-sys
inference --> ios
inference --> coreml-sys

ios --> mil
```

## Risks

1. **Decode bundle complexity**: The decode setup is ~2000 lines. The bundle
   must capture everything the decode loop needs (architecture, RoPE, CPU
   weights, packing metadata, donor compatibility annotations).

2. **Donor/patch optimization**: The decode loop compiles the first layer
   normally then patches weights for subsequent layers. The bundle should
   annotate which layers share structure to enable this at load time.

3. **`write_packed_inputs` at runtime**: This function writes packed data
   into ANE tensors during decode. It uses `InputPacking` metadata (moving
   to mil-rs) but operates on `AneTensor` (from inference). The function
   itself should move to inference since it operates on runtime tensors.

4. **`ProgramCache`**: Currently in compile, used by inference for ANE
   compile budget tracking. Should move to inference or be reimplemented
   there (~200 lines) since it's a runtime concern.

5. **Testing**: Decode-path tests that build MIL programs and compile them
   would need to call both compile and inference APIs, so they'd move to
   integration tests or the bench crate.
