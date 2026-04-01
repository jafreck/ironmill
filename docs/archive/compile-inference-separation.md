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
load. `CompiledArtifacts::prepare()` (also in `ane/model.rs`) runs the pass
pipeline without ANE runtime, and the CLI uses it for the `--runtime ane-direct`
compile path.

**Decode path** (`AneInference::compile` in `ane/decode.rs`):
Program → architecture detection → autoregressive passes → gather replacement →
attention-aware split → per-sub-program packing/matmul→conv/DCE → cache-write
fusion → donor/patch compilation → full decode engine setup. ~730 lines
(lines 378–1106), deeply intertwined with compile-crate types.

### What inference imports from compile

| Category | Items |
|----------|-------|
| Passes | `OpSubstitutionPass`, `AneLayoutPass`, `AttentionDecomposePass`, `AneConcatEliminationPass`, `AneVariableNamingPass`, `AneArgPromotionPass`, `AneMatmulToConvPass` |
| Split | `SplitConfig`, `split_for_ane`, `SubProgram`¹ |
| Emission | `MilTextConfig`, `program_to_mil_text`, `BlobFileWriter` |
| Packing | `InputPacking`¹, `pack_inputs`¹, `write_packed_inputs`¹ |
| Cache | `ProgramCache`, `ProgramKey`¹ |
| Types | `TensorDescriptor`, `AneCompileError` |

¹ Used via fully-qualified paths rather than direct `use` imports.

### Key observation

None of the decode-path passes require runtime information. They use
compile-time decisions (seq_len=1, TurboQuant config, split strategy). The
CLI already proves this — `CompiledArtifacts::prepare()` runs passes without
any ANE runtime. These are compile passes that run at load time because
there's no compile API that accepts "compile for autoregressive decode" as
a parameter.

## Approach

Create a clean artifact boundary: `ironmill-compile` produces self-contained
`.ironml` bundles that `ironmill-inference` can load and run without touching
IR passes, splitting, or MIL emission. The `.ironml` directory format is the
contract between the two crates — no shared Rust types are required.

## Phase 1: Define `.ironml` format and bundle types

### Artifact format

A `.ironml` bundle is a directory containing compiled ANE artifacts ready for
runtime loading. The format follows the same directory-bundle pattern as
CoreML's `.mlpackage` — a JSON manifest with separate binary files for weights
and programs, enabling mmap-based loading and metadata updates without
re-serializing large binaries.

```
model.ironml/
├── manifest.json              # metadata, architecture, sub-program descriptors
├── programs/
│   ├── layer_0_pre_attn.mil   # compiled MIL text per sub-program
│   ├── layer_0_post_attn.mil
│   └── ...
├── weights/
│   ├── layer_0_pre_attn.bin   # weight blob per sub-program
│   ├── layer_0_post_attn.bin
│   └── ...
└── cpu_weights/               # decode-path only: CPU-side weights
    ├── embedding.bin
    ├── lm_head.bin
    ├── rope_cos.bin
    ├── rope_sin.bin
    └── final_norm.bin         # optional
```

The manifest contains:
- Format version (for forward compatibility)
- Model type (`simple` or `decode`)
- Architecture identifier (for decode bundles)
- Per-sub-program metadata: name, input/output tensor descriptors (name,
  shape as `[usize; 4]`, dtype), input packing offsets/sizes
- Per-layer metadata (for decode): layer index, cache-write-fused flag,
  donor compatibility annotations
- LM head configuration

### Bundle types in ironmill-compile

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
    pub donor_compatible: bool,
}
```

`TensorDescriptor` (`{ name, shape: [usize; 4], dtype }`) and `InputPacking`
(`{ offsets, sizes }`) stay in ironmill-compile — they are ANE-specific types
(the `[usize; 4]` shape encodes NCHW layout), not generic IR. Inference
defines its own equivalent types for deserialization from the manifest.

Add `AneModelBundle::save()` and `AneDecodeBundle::save()` to write the
`.ironml` directory format.

## Phase 2: Create high-level compilation APIs in ironmill-compile

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
Absorbs the decode setup from `inference::ane::decode::compile` (~730 lines):
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

## Phase 3: Refactor ironmill-inference to consume bundles

Replace JIT compilation with bundle consumption:

**Simple path:**
```rust
impl AneModel {
    pub fn from_bundle(bundle_path: &Path, config: AneConfig) -> Result<Self>
}
```
Reads `.ironml` directory → ANE `compile_mil_text` → load → allocate tensors.

**Decode path:**
```rust
impl AneInference {
    pub fn from_bundle(bundle_path: &Path, turbo: Option<TurboQuantConfig>) -> Result<Self>
}
```
Reads `.ironml` directory → load sub-programs (with donor/patch optimization)
→ set up caches → ready for decode loop.

Remove `ironmill-compile` from inference's `Cargo.toml`.

**Convenience wrappers** (`AneModel::compile_and_load`, `AneInference::compile`)
stay on inference behind an optional `compile` feature flag that re-adds the
`ironmill-compile` dependency. When enabled, these methods call
`compile_model_bundle()` / `compile_decode_bundle()` then `from_bundle()`
internally. Callers like `ironmill-bench` enable this feature for ergonomic
one-call usage. The core inference crate has no compile dependency.

## Phase 4: Update dependent crates and README

**ironmill-cli:** The CLI is a compiler — it has no inference/run command.
Switch the `--runtime ane-direct` path from `CompiledArtifacts::prepare()` to
`compile_model_bundle()` / `compile_decode_bundle()`, emitting `.ironml`
bundles. CLI compile remains always-available (not feature-gated), supporting
both `--runtime coreml` (`.mlpackage`) and `--runtime ane-direct` (`.ironml`).

**ironmill-bench:** ANE-direct benchmarks and perplexity evaluation use the
inference convenience wrappers (`AneInference::compile`, etc.) with the
`compile` feature enabled. Bench orchestrates both compile and inference
through the Rust API.

**burn-coreml / candle-coreml:** These crates use the CoreML path
(`CompileBuilder` + `Model::load`), not ANE-direct. No changes needed.

**README Mermaid diagram:** Remove `inference --> compile` edge. Add
`inference --> mil` edge (already a real dependency, currently missing
from diagram).

Updated crate dependency graph after separation:

```
cli --> compile

bench --> compile
bench --> inference
bench -.-> ironmill-iosurface

burn --> compile
burn --> inference

candle --> compile
candle --> inference

compile --> ironmill-core
compile --> mil
compile --> ironmill-iosurface

inference --> ironmill-core
inference --> mil
inference --> ane-sys
inference --> ironmill-iosurface
inference --> coreml-sys
inference -.-> compile           (optional, behind "compile" feature)
inference -.-> metal-sys         (optional, behind "metal" feature)

ironmill-core --> mil
ironmill-iosurface --> mil
```

## Risks

1. **Decode bundle complexity**: The decode setup is ~730 lines. The bundle
   must capture everything the decode loop needs (architecture, RoPE, CPU
   weights, packing metadata, donor compatibility annotations).

2. **Donor/patch optimization**: The decode loop compiles the first layer
   normally then patches weights for subsequent layers. The bundle annotates
   which layers share structure (`donor_compatible`) to enable this at load
   time.

3. **`write_packed_inputs` at runtime**: This function writes packed data
   into ANE tensors during decode. It uses `InputPacking` metadata but
   operates on `AneTensor` (from inference). The function should move to
   inference since it operates on runtime tensors. Inference defines its
   own `InputPacking` equivalent for deserialization from the manifest.

4. **`ProgramCache`**: Currently in compile (~230 lines), used by inference
   for ANE compile budget tracking. Should move to inference or be
   reimplemented there since it's a runtime concern.

5. **Testing**: Decode-path tests that build MIL programs and compile them
   would need to call both compile and inference APIs, so they'd move to
   integration tests or the bench crate.

6. **Format versioning**: The `.ironml` manifest needs a version field from
   day one. Bundle format changes must be backward-compatible or versioned
   to avoid breaking cached artifacts.
