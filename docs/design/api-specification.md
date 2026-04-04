# Ironmill Public API Specification

> **Status**: Proposal  
> **Scope**: mil-rs, ironmill-compile, ironmill-inference, ironmill-cli  
> **Goal**: Establish a stable, extensible public contract that supports new models, quantization methods, backends, and optimizations without breaking consumers.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [mil-rs: IR & Weight Abstraction](#2-mil-rs)
3. [ironmill-compile: Model Compilation](#3-ironmill-compile)
4. [ironmill-inference: Runtime Engine](#4-ironmill-inference)
5. [ironmill-cli: Command-Line Interface](#5-ironmill-cli)
6. [Cross-Cutting Concerns](#6-cross-cutting-concerns)
7. [Usage Examples](#7-usage-examples)
8. [Migration Guide](#8-migration-guide)
9. [Tokenizer Abstraction](#9-tokenizer-abstraction)
10. [ironmill-core: High-Level API](#10-ironmill-core-high-level-api)
11. [JIT Compilation](#11-jit-compilation)
12. [Design Decisions](#12-design-decisions)

---

## 1. Design Principles

### 1.1 Semver Stability

Every public enum **must** carry `#[non_exhaustive]`. Every public struct that
users construct **must** use one of:

- **Builder pattern** — for complex configuration with validation
- **`#[non_exhaustive]` + `Default`** — for simpler structs with field access
- **Private fields + accessor methods** — when invariants must be maintained

Output/report structs (read-only data returned to the user) may keep public
fields but still carry `#[non_exhaustive]`.

### 1.2 Type Safety

Replace stringly-typed APIs with enums and typed config structs wherever
possible. `HashMap<String, Value>` bags are acceptable only at the IR
serialization boundary (where they mirror the CoreML MIL spec).

### 1.3 Error Discipline

- Each crate defines its own error enum with `#[non_exhaustive]`.
- Error conversions preserve source errors via `#[source]` or boxing.
- Public APIs never return `Result<_, String>`.
- Public traits never return `anyhow::Result` — use crate-specific errors.

### 1.4 Output Discipline (CLI)

- **stdout**: Machine-readable output only (JSON, model data).
- **stderr**: Progress, diagnostics, warnings, notes.
- Non-zero exit on any failure, including compiler failures.

### 1.5 Trait Extensibility

All public traits include default method implementations for every method
except the core required set, so new methods can be added without breaking
external implementors.

### 1.6 Platform Portability

The core abstraction layer (traits, types, sampling, grammar, serving) **must**
be platform-agnostic. Hardware-specific backends (Metal, ANE, CoreML, and
future CUDA/Vulkan) are isolated behind feature flags and `#[cfg]` gates.

- The `ironmill-inference` crate **must not** have a top-level
  `compile_error!` for non-macOS. Platform-specific code lives in
  feature-gated modules.
- Core types (`InferenceEngine`, `RuntimeModel`, `Sampler`, `BatchScheduler`,
  etc.) must compile on any `target_os`.
- Adding a new hardware backend (e.g., CUDA) must not require modifying
  any existing backend module — only adding a new feature-gated module
  and extending `#[non_exhaustive]` enums.

### 1.7 Inference Architecture — Two Paths

SafeTensors, GGUF, and ONNX files are **weight storage formats** — they
contain raw tensor data (matrices, vectors) with no computation logic. They
are not executable. Running inference requires pairing those weights with an
execution engine that knows what operations to perform (matmul, RoPE,
softmax, etc.).

Ironmill provides two fundamentally different execution paths:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Path 1: Direct GPU (Metal today, CUDA tomorrow)                     │
│                                                                     │
│   SafeTensors ──→ WeightProvider ──→ GPU buffers ──→ Hand-written   │
│                        │                              Metal shaders │
│                        ↓                                            │
│                  (optional: quantization                             │
│                   passes on weight data)                             │
│                                                                     │
│   • No compilation step. No MIL IR. No CoreML.                      │
│   • Weights are loaded directly into Metal buffers.                  │
│   • Shaders are pre-compiled (metallib) or JIT-compiled.            │
│   • Time-to-first-token: seconds (weight loading only).             │
│   • This is MetalInference / CudaInference.                        │
├─────────────────────────────────────────────────────────────────────┤
│ Path 2: CoreML / ANE (Apple Neural Engine)                          │
│                                                                     │
│   SafeTensors ──→ MIL IR ──→ .mlpackage ──→ coremlcompiler         │
│                  (computation    (serialized     ──→ .mlmodelc      │
│                   graph)          protobuf)       (compiled for     │
│                                                    CPU/GPU/ANE)     │
│                                                                     │
│   • Requires a computation graph (MIL IR) because CoreML needs to   │
│     know the full graph topology to optimize for ANE/GPU/CPU.       │
│   • Compilation is slow (30s–minutes) but produces a distributable  │
│     artifact that Apple's runtime can execute on any Apple device.   │
│   • Required for ANE inference — the Neural Engine only runs        │
│     CoreML-compiled models.                                         │
│   • This is CoremlBackend / AneInference.                           │
├─────────────────────────────────────────────────────────────────────┤
│ Path 3: JIT (proposed — see §11)                                    │
│                                                                     │
│   SafeTensors ──→ WeightProvider ──→ GPU buffers ──→ JIT-compiled   │
│                        │                              Metal shaders │
│                        ↓                                            │
│                  (quantization applied                               │
│                   directly to tensors,                               │
│                   no MIL IR needed)                                  │
│                                                                     │
│   • Like Path 1 but with on-the-fly shader specialization.          │
│   • Quantization transforms operate on raw tensors, not MIL IR.     │
│   • Compiled shader pipelines are cached to disk.                   │
│   • Fastest iteration speed: change config → re-run immediately.    │
└─────────────────────────────────────────────────────────────────────┘
```

**Why does the current GPU path touch MIL IR at all?**

`GpuCompileBuilder` currently routes through MIL IR because the quantization
passes (`PassPipeline`) operate on MIL `Program` graphs. This is a historical
artifact — weight quantization (INT4, AWQ, PolarQuant) transforms individual
tensors and does not require a computation graph. The JIT path (§11) proposes
`TensorTransform` passes that operate directly on `WeightProvider` data,
eliminating the MIL IR dependency for GPU inference entirely.

**Which path should users choose?**

| Use case | Path | Why |
|----------|------|-----|
| GPU inference (dev/prod) | Direct GPU or JIT | Fastest. No compile step. |
| ANE inference | CoreML/ANE | ANE requires CoreML models. |
| Distributing a compiled model | CoreML | Produces .mlmodelc that runs anywhere on Apple. |
| Experimenting with quantization | JIT | No recompile on config changes. |
| Maximum decode throughput | Direct GPU (Metal) | Hand-tuned Metal shaders. |

---

## 2. mil-rs

### 2.1 Enums — Add `#[non_exhaustive]`

Every public enum gets `#[non_exhaustive]`. This is the single highest-impact
change — it makes every future variant addition non-breaking.

```rust
// ── ir/tensor.rs ──
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarType {
    Float16, Float32, Float64,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Bool,
}

// ── ir/operation.rs ──
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeUnit {
    Ane,
    Gpu,
    Cpu,
    Cuda,  // future: NVIDIA GPU
    Any,
}

// ── ir/types.rs ──
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Reference(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    List(Vec<Value>),
    Type(TensorType),
    Tensor { data: Vec<u8>, shape: Vec<usize>, dtype: ScalarType },
}

// ── weights.rs ──
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Llama, Qwen, Gemma,
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum QuantizationInfo {
    None,
    LutToDense { /* existing fields */ },
    AffineDequantize { /* existing fields */ },
}

// ── error.rs ──
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum MilError {
    #[error("undefined value: {0}")]
    UndefinedValue(String),
    #[error("type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
    #[error("unsupported operation: {0}")]
    UnsupportedOp(String),
    #[error("protobuf error: {0}")]
    Protobuf(String),
    #[error("invalid package: {0}")]
    InvalidPackage(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("validation error: {0}")]
    Validation(String),
}

// ── ir/pipeline.rs ──
#[non_exhaustive]
pub enum SpinQuantMethod {
    MinMax { group_size: usize },
    Awq { channel_magnitudes: HashMap<String, Vec<f32>>, group_size: usize },
    #[cfg(feature = "gptq")]
    Gptq { /* fields */ },
}

// ── convert/ir_to_proto.rs ──
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UpdateOptimizer { Sgd, Adam }

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossFunction { CategoricalCrossEntropy, MeanSquaredError }
```

### 2.2 Structs — `#[non_exhaustive]` on Constructible Types

Structs that users construct via struct-literal syntax need `#[non_exhaustive]`
so we can add fields. This means users must switch to builder/constructor
patterns. We provide `Default` + modifier methods or dedicated constructors.

```rust
// ── weights.rs ──
// ModelConfig has too many fields for struct literals — use a builder.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: Architecture,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub extra: HashMap<String, serde_json::Value>,
}

impl ModelConfig {
    pub fn new(architecture: Architecture) -> Self {
        Self {
            architecture,
            hidden_size: 0,
            intermediate_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            head_dim: 0,
            vocab_size: 0,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            extra: HashMap::new(),
        }
    }

    pub fn with_hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = size;
        self
    }
    // ... builder methods for each field ...
}

// ── ir/pipeline.rs ──
// Output structs — #[non_exhaustive] with public fields for reading.
#[non_exhaustive]
#[derive(Debug)]
pub struct PipelineReport {
    pub pass_results: Vec<PassResult>,
}

#[non_exhaustive]
#[derive(Debug)]
pub struct PassResult {
    pub name: String,
    pub ops_before: usize,
    pub ops_after: usize,
    pub flops_before: u64,
    pub flops_after: u64,
    pub memory_before: u64,
    pub memory_after: u64,
    pub elapsed: Duration,
}

// ── PipelineConfig / PassConfig — deserialized from TOML, keep fields public
#[non_exhaustive]
#[derive(Debug, Deserialize)]
pub struct PipelineConfig {
    pub passes: Vec<PassConfig>,
}

#[non_exhaustive]
#[derive(Debug, Deserialize)]
pub struct PassConfig {
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub params: HashMap<String, toml::Value>,
}

// ── SpinQuantConfig ──
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SpinQuantConfig {
    pub rotation_epochs: usize,
    pub bits: u8,
}

// ── convert types ──
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct UpdatableModelConfig {
    pub updatable_layers: Vec<String>,
    pub learning_rate: f64,
    pub epochs: i64,
    pub loss_function: LossFunction,
    pub optimizer: UpdateOptimizer,
}

#[non_exhaustive]
#[derive(Debug)]
pub struct ConversionResult {
    pub program: Program,
    pub warnings: Vec<String>,
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    pub merge_lora: bool,
    pub model_dir: Option<PathBuf>,
}
```

### 2.3 Pass Trait — Add Default Methods

```rust
pub trait Pass {
    /// Human-readable name for this pass.
    fn name(&self) -> &str;

    /// Apply the pass to the program, modifying it in place.
    fn run(&self, program: &mut Program) -> crate::error::Result<()>;

    /// Optional description of what this pass does.
    fn description(&self) -> &str { "" }

    /// Optional list of pass names this pass depends on.
    fn dependencies(&self) -> &[&str] { &[] }
}
```

### 2.4 Pass Registry — Open Extension Point

Add a user-facing pass resolver to `PassPipeline` so external passes can
participate in TOML config loading:

```rust
impl PassPipeline {
    /// Register a custom pass factory for TOML config loading.
    ///
    /// When `with_config()` or `from_config_str()` encounters a pass name
    /// not in the built-in registry, it calls registered resolvers in order.
    pub fn register_pass_factory(
        &mut self,
        factory: Box<dyn Fn(&str, &HashMap<String, toml::Value>) -> Option<Box<dyn Pass>>>,
    ) { /* ... */ }
}
```

### 2.5 Core IR Types — Unchanged

The core IR types (`Program`, `Function`, `Block`, `Operation`, `TensorType`,
`Graph`) keep their public fields and chainable constructors. These are the
MIL spec representation — their stringly-typed nature (`op_type: String`,
`inputs: HashMap<String, Value>`) is intentional and mirrors Apple's format.

Add `#[non_exhaustive]` to these structs for forward compatibility:

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Program { pub version: String, pub functions: IndexMap<String, Function>, pub attributes: HashMap<String, String> }

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Function { pub name: String, pub inputs: Vec<(String, TensorType)>, pub body: Block }

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Block { pub operations: Vec<Operation>, pub outputs: Vec<String> }

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Operation {
    pub op_type: String,
    pub name: String,
    pub inputs: HashMap<String, Value>,
    pub outputs: Vec<String>,
    pub output_types: Vec<Option<TensorType>>,
    pub attributes: HashMap<String, Value>,
    pub compute_unit: Option<ComputeUnit>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType { pub scalar_type: ScalarType, pub shape: Vec<Option<usize>> }
```

### 2.6 WeightProvider — Add Default for `has_tensor`

Already has a default. No changes needed. Keep as-is:

```rust
pub trait WeightProvider {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError>;
    fn tensor_names(&self) -> Vec<&str>;
    fn config(&self) -> &ModelConfig;
    fn has_tensor(&self, name: &str) -> bool { self.tensor(name).is_ok() }
}
```

---

## 3. ironmill-compile

### 3.1 Enums — Add `#[non_exhaustive]`

```rust
// ── coreml/build_api.rs ──
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quantization { #[default] None, Fp16, Int8 }

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TargetComputeUnit { CpuOnly, CpuAndGpu, CpuAndNeuralEngine, #[default] All }

// ── weights/gguf.rs ──
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
    Q2K, Q3K, Q4K, Q5K, Q6K,
    IQ2XXS, IQ2XS, IQ3XXS, IQ1S, IQ4NL, IQ3S, IQ2S, IQ4XS,
    I8, I16, I32, I64, F64, IQ1M, BF16,
}

// ── ane/passes ──
#[non_exhaustive]
pub enum LayerType { Attention, Feedforward, Full }

#[non_exhaustive]
pub enum OpPrecision { Float16, Float32 }

#[non_exhaustive]
pub enum ExpertQuantStrategy { /* variants */ }

// ── error.rs ──
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum CompileError {
    #[error(transparent)]
    Mil(#[from] MilError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("coremlcompiler not found at {path}")]
    CompilerNotAvailable { path: PathBuf },
    #[error("coremlcompiler failed with exit code {exit_code}: {stderr}")]
    CompilerFailed { exit_code: i32, stderr: String },
    #[error("{0}")]
    Other(String),
}
```

Note: `CompilerNotAvailable` and `CompilerFailed` are changed from `String`
wrappers to structured variants with useful fields.

### 3.2 Component Enum — Replace Magic Strings

```rust
/// Model component for targeted compilation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelComponent {
    /// Full model (default).
    Full,
    /// Embedding table only.
    Embeddings,
    /// Transformer layers only.
    Transformer,
    /// Language model head only.
    LmHead,
}

// Replace:
//   weights_to_program_component(provider, Some("embeddings"))
// With:
//   weights_to_program_component(provider, ModelComponent::Embeddings)

pub fn weights_to_program(provider: &dyn WeightProvider) -> Result<ConversionResult, MilError> {
    weights_to_program_component(provider, ModelComponent::Full)
}

pub fn weights_to_program_with_options(
    provider: &dyn WeightProvider,
    options: &TemplateOptions,
) -> Result<ConversionResult, MilError> { /* ... */ }

pub fn weights_to_program_component(
    provider: &dyn WeightProvider,
    component: ModelComponent,
) -> Result<ConversionResult, MilError> { /* ... */ }
```

### 3.3 TemplateOptions — `#[non_exhaustive]` + Default

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TemplateOptions {
    /// Generate ANE-compatible operations.
    pub ane: bool,
}

impl Default for TemplateOptions {
    fn default() -> Self { Self { ane: false } }
}
```

### 3.4 BuildOutput — `#[non_exhaustive]`

```rust
#[non_exhaustive]
#[derive(Debug)]
pub struct BuildOutput {
    pub mlpackage: PathBuf,
    pub mlmodelc: Option<PathBuf>,
    pub report: PipelineReport,
}
```

### 3.5 Pipeline Config — Typed Quantize Field

```rust
/// Quantization method for a pipeline stage.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StageQuantize {
    None,
    Fp16,
    Int8,
    Int4,
    Awq,
    Gptq,
    D2Quant,
    Palettize,
    PolarQuant,
    QuipSharp,
}

#[non_exhaustive]
#[derive(Debug, Clone, Deserialize)]
pub struct StageConfig {
    pub name: String,
    pub source: PathBuf,
    pub quantize: StageQuantize,
    pub component: Option<ModelComponent>,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

#[non_exhaustive]
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineManifest {
    pub name: String,
    pub stages: Vec<StageConfig>,
}
```

### 3.6 Re-export Cleanup

```rust
// lib.rs — tighten the `mil` façade
pub mod mil {
    // Keep: IR types, reader/writer, conversion helpers
    pub use mil_rs::{
        Program, Function, Block, Operation, Value, TensorType, ScalarType, ComputeUnit,
        Pass, PassPipeline, PipelineReport, PassResult,
        MilError,
        read_mlmodel, read_mlpackage, read_onnx,
        write_mlmodel, write_mlpackage,
        program_to_model, model_to_program,
        onnx_to_program, onnx_to_program_with_config,
        ConversionResult, ConversionConfig,
    };

    // Remove: proto::specification::Model, proto::onnx::ModelProto
    // Remove: passes::tensor_utils (internal)
    // If users need proto types, add a `proto` feature gate.
}

// Remove pass-through re-exports in convert/lora.rs and convert/moe.rs.
// Instead, re-export only what's needed:
pub mod convert {
    pub use mil_rs::convert::{
        LoraAdapter, detect_lora_adapters, merge_lora,
        MoeTopology, MoeSplitResult, detect_moe, split_moe,
    };
}
```

### 3.7 CompileTarget — Backend-Agnostic Compilation

The current compilation API is tightly coupled to CoreML (MIL IR →
mlpackage → mlmodelc). For Metal direct inference and future CUDA support,
compilation means something completely different. `CompileTarget` abstracts
over these backends:

```rust
/// A compilation target that transforms weights for a specific runtime.
///
/// Implementations handle the full pipeline from source weights to a
/// runtime-ready artifact. The artifact type varies by target:
/// - Metal direct → `.ironml-gpu` bundle (quantized weight tensors)
/// - CoreML → `.mlmodelc` (compiled CoreML model)
/// - CUDA → `.ironml-cuda` bundle (future)
pub trait CompileTarget {
    /// Name of this compilation target (for logging/diagnostics).
    fn name(&self) -> &str;

    /// Compile a model from source weights to a runtime-ready artifact.
    ///
    /// `progress` is called with status updates during compilation. Pass
    /// `&NullProgress` to suppress output.
    fn compile(
        &self,
        source: &dyn WeightProvider,
        config: &CompileConfig,
        progress: &dyn ProgressSink,
    ) -> Result<CompileOutput, CompileError>;

    /// Estimate the output artifact size without performing compilation.
    fn estimate_size(
        &self,
        source: &dyn WeightProvider,
        config: &CompileConfig,
    ) -> Result<usize, CompileError> {
        // Default: weight memory estimate
        Ok(MemoryEstimator::weight_memory(source.config(), config.quant_level()))
    }
}

/// Compilation configuration shared across all targets.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CompileConfig {
    /// Output directory or file path.
    pub output: PathBuf,
    /// Quantization pass pipeline.
    pub pipeline: PassPipeline,
    /// Target compute unit hint.
    pub compute_unit: ComputeUnit,
}

/// Output of a compilation step.
#[non_exhaustive]
#[derive(Debug)]
pub struct CompileOutput {
    /// Path to the compiled artifact.
    pub artifact: PathBuf,
    /// Pipeline report (passes run, timing, size changes).
    pub report: PipelineReport,
    /// Warnings generated during compilation.
    pub warnings: Vec<String>,
    /// Metadata written into the artifact bundle (see §12.3).
    pub metadata: ArtifactMetadata,
}

/// Metadata embedded in every compiled artifact for version checking.
///
/// Written as `metadata.json` at the bundle root. Loaders check
/// `format_version` on load and reject incompatible bundles with a
/// clear error message.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    /// Bundle format version. Incremented on breaking layout changes.
    pub format_version: u32,
    /// Semver of the ironmill build that produced this artifact.
    pub ironmill_version: String,
    /// Model architecture name.
    pub architecture: String,
    /// Quantization method applied.
    pub quantization: String,
}
```

Built-in targets:

```rust
/// Metal direct compilation: weights → quantized tensors → .ironml-gpu bundle.
///
/// No MIL IR or CoreML involved. Quantization passes operate directly on
/// weight tensors via `TensorTransform` (see §11 JIT).
pub struct MetalCompileTarget;

impl CompileTarget for MetalCompileTarget {
    fn name(&self) -> &str { "metal-direct" }
    fn compile(&self, source: &dyn WeightProvider, config: &CompileConfig,
               progress: &dyn ProgressSink) -> Result<CompileOutput, CompileError> {
        // 1. Apply quantization transforms to weight tensors
        // 2. Write .ironml-gpu bundle (weights + config.json + metadata)
        // No MIL IR. No coremlcompiler.
        todo!()
    }
}

/// CoreML compilation: weights → MIL IR → mlpackage → mlmodelc.
///
/// Required for ANE inference and CoreML distribution.
pub struct CoremlCompileTarget;

impl CompileTarget for CoremlCompileTarget {
    fn name(&self) -> &str { "coreml" }
    fn compile(&self, source: &dyn WeightProvider, config: &CompileConfig,
               progress: &dyn ProgressSink) -> Result<CompileOutput, CompileError> {
        // 1. Convert to MIL IR via weights_to_program()
        // 2. Run PassPipeline on the MIL Program
        // 3. Serialize to mlpackage
        // 4. Run coremlcompiler → mlmodelc
        todo!()
    }
}
```

---

## 4. ironmill-inference

### 4.1 Enums — Add `#[non_exhaustive]`

```rust
// ── types.rs ──
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    Float32,
    Float16,
    BFloat16,  // common on NVIDIA (Ampere+), future Apple support
    Int32,
    Int8,      // quantized inference on all platforms
    Float64,
}

// ── engine.rs ──
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("runtime error: {0}")]
    Runtime(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("model not loaded")]
    NotLoaded,
    #[error("sampling error: {0}")]
    Sampling(String),
    #[error("allocation error: {0}")]
    Allocation(String),
    #[error("sequence {0} not found")]
    SequenceNotFound(u64),
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

// ── lib.rs (AneError) ──
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum AneError { /* existing variants */ }

// ── metal/error.rs ──
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum MetalError { /* existing variants */ }

// ── serving/sequence.rs ──
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus { Waiting, Prefilling, Decoding, Complete }
```

### 4.2 InferenceError — Preserve Source Errors

The key change: `Runtime(String)` → `Runtime(Box<dyn Error + Send + Sync>)`.
This preserves the full error chain from Metal, ANE, or CoreML backends.

```rust
// ── metal/error.rs ──
impl From<MetalError> for InferenceError {
    fn from(e: MetalError) -> Self {
        // Preserve the structured error instead of .to_string()
        InferenceError::Runtime(Box::new(e))
    }
}

// Users can still display errors:
//   eprintln!("Error: {err}");
// And can downcast if needed:
//   if let InferenceError::Runtime(source) = &err {
//       if let Some(metal_err) = source.downcast_ref::<MetalError>() { ... }
//   }
```

### 4.3 InferenceEngine — Two-Tier Design

The `InferenceEngine` trait must serve two audiences:

1. **Generic callers** (application code that knows its backend at compile time)
   want type-safe `load()` with compile-time artifact checking.
2. **Dynamic callers** (the high-level `Model` API, batch schedulers, tests)
   need `dyn`-dispatched engines where the backend is chosen at runtime.

Trying to serve both with a single trait (associated types for one, object
safety for the other) creates an irreconcilable tension. The solution is
**two traits**: a low-level object-safe `InferenceEngine` for the runtime,
and type-safe `load()` as a **separate method on each backend struct**.

```rust
/// Core inference engine trait — object-safe, used everywhere.
///
/// This trait covers the hot path (prefill/decode) and is the primary
/// interface consumed by `TokenStream`, `BatchRunner`, `Model`, etc.
/// It is intentionally object-safe so it can be used as `dyn InferenceEngine`.
///
/// Loading is NOT part of this trait — each backend has its own typed
/// `load()` method because artifact types differ. Once loaded, all
/// backends present the same interface.
pub trait InferenceEngine: Send {
    /// Prefill: process all prompt tokens, populating the KV cache.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError>;

    /// Decode one token, returning logits for the next token prediction.
    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError>;

    /// Reset all state for a new conversation.
    fn reset(&mut self);

    /// Current sequence position (number of tokens in KV cache).
    fn seq_pos(&self) -> usize;

    /// Truncate KV cache to the given position.
    fn truncate_to(&mut self, pos: usize);

    /// Maximum sequence length this engine supports.
    fn max_seq_len(&self) -> usize { usize::MAX }

    /// Model info for this loaded engine (see §4.13).
    fn model_info(&self) -> &ModelInfo;
}
```

Each backend provides its own typed loading:

```rust
impl MetalInference {
    /// Load model from Metal-specific artifacts.
    /// Type-safe: compile-time guarantee that you pass MetalArtifacts.
    pub fn load(config: MetalConfig, artifacts: &MetalArtifacts<'_>)
        -> Result<Self, InferenceError> { /* ... */ }
}

impl MlxInference {
    pub fn load(config: MlxConfig, artifacts: &MlxArtifacts<'_>)
        -> Result<Self, InferenceError> { /* ... */ }
}

impl AneInference<D: AneDevice> {
    pub fn load(config: AneConfig, bundle_path: &Path, device: D)
        -> Result<Self, InferenceError> { /* ... */ }
}
```

**Why `Send` on the trait?** All backends hold GPU/accelerator state that
is `Send` (can be moved to another thread) but NOT `Sync` (can't be shared).
The `Send` bound enables the async adapter and thread-per-engine patterns.

**Dynamic dispatch works naturally:**

```rust
// Runtime backend selection
fn create_engine(device: Device, path: &Path) -> Result<Box<dyn InferenceEngine>, InferenceError> {
    match device {
        Device::Metal => {
            let artifacts = MetalBundleProvider::open(path)?;
            let engine = MetalInference::load(MetalConfig::default(), &artifacts)?;
            Ok(Box::new(engine))
        }
        Device::Ane => {
            let device = HardwareAneDevice::new()?;
            let engine = AneInference::load(AneConfig::default(), path, device)?;
            Ok(Box::new(engine))
        }
        // ...
    }
}

// TokenStream, generate(), etc. all take &mut dyn InferenceEngine
let mut engine: Box<dyn InferenceEngine> = create_engine(Device::Metal, path)?;
let stream = TokenStream::new(&mut *engine, request, &cancel);
```

**Migration from old `load(&dyn Any)` pattern:**

```rust
// BEFORE (runtime type check, panics on mismatch):
let mut engine = MetalInference::new(config)?;
engine.load(&artifacts as &dyn std::any::Any)?;

// AFTER (compile-time type check, no possible mismatch):
let engine = MetalInference::load(config, &artifacts)?;
```

### 4.4 RuntimeModel / RuntimeBackend — Typed Errors

Replace `anyhow::Result` with `InferenceError`:

```rust
pub trait RuntimeModel {
    fn input_description(&self) -> Vec<InputFeatureDesc>;
    fn predict(&self, inputs: &[RuntimeTensor]) -> Result<Vec<RuntimeTensor>, InferenceError>;
}

pub trait RuntimeBackend: Send + Sync {
    fn name(&self) -> &str;
    fn load(&self, model_path: &std::path::Path) -> Result<Box<dyn RuntimeModel>, InferenceError>;
}
```

### 4.5 Wire Up ANE Backend

`AneRuntimeModel::predict` and `AneDirectBackend::load` are currently stubs.
Implement them by delegating to the existing `AneModel` infrastructure:

```rust
// ── ane/model.rs ──
impl RuntimeModel for AneRuntimeModel {
    fn input_description(&self) -> Vec<InputFeatureDesc> {
        self.model.input_description()
    }

    fn predict(&self, inputs: &[RuntimeTensor]) -> Result<Vec<RuntimeTensor>, InferenceError> {
        // Convert RuntimeTensor → IOSurface-backed inputs
        // Call self.model.predict(...)
        // Convert outputs back to RuntimeTensor
        let io_inputs = self.convert_inputs(inputs)?;
        let io_outputs = self.model.predict(&io_inputs)
            .map_err(|e| InferenceError::Runtime(Box::new(e)))?;
        Ok(self.convert_outputs(&io_outputs))
    }
}

impl RuntimeBackend for AneDirectBackend {
    fn name(&self) -> &str { "ane-direct" }

    fn load(&self, model_path: &Path) -> Result<Box<dyn RuntimeModel>, InferenceError> {
        let device = HardwareAneDevice::new()
            .map_err(|e| InferenceError::Runtime(Box::new(e)))?;
        let config = AneConfig::default();
        let model = AneModel::from_bundle(model_path, device, &config)
            .map_err(|e| InferenceError::Runtime(Box::new(e)))?;
        Ok(Box::new(AneRuntimeModel { model }))
    }
}
```

Typed loading on the struct (not the trait — see §4.3):

```rust
impl<D: AneDevice> AneInference<D> {
    /// Load from an ANE decode bundle. Type-safe: compile-time guarantee
    /// that you pass the correct artifact type.
    pub fn load(config: AneConfig, bundle_path: &Path, device: D)
        -> Result<Self, InferenceError>
    {
        AneInference::from_bundle(bundle_path, device, &config)
            .map_err(|e| InferenceError::Runtime(Box::new(e)))
    }
}
```

Implement the object-safe `InferenceEngine` trait (no `load`, no associated types):

```rust
impl<D: AneDevice> InferenceEngine for AneInference<D> {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
        self.prefill(tokens)
            .map_err(|e| InferenceError::Runtime(Box::new(e)))
    }

    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError> {
        self.decode(token)
            .map_err(|e| InferenceError::Runtime(Box::new(e)))
    }

    fn reset(&mut self) { self.reset(); }
    fn seq_pos(&self) -> usize { self.seq_pos() }
    fn truncate_to(&mut self, pos: usize) { self.truncate_to(pos); }
    fn model_info(&self) -> &ModelInfo { &self.model_info }
}
```

### 4.6 Wire Up CoreML Backend

`CoremlRuntimeModel::predict` currently returns `Ok(vec![])`. Implement it
by extracting MLMultiArray outputs from the prediction:

```rust
// ── coreml/runtime.rs ──
impl RuntimeModel for CoremlRuntimeModel {
    fn input_description(&self) -> Vec<InputFeatureDesc> {
        self.model.input_descriptions()
            .into_iter()
            .map(|desc| InputFeatureDesc {
                name: desc.name.clone(),
                shape: desc.shape.clone(),
                dtype: multi_array_dtype_to_element_type(desc.data_type),
            })
            .collect()
    }

    fn predict(&self, inputs: &[RuntimeTensor]) -> Result<Vec<RuntimeTensor>, InferenceError> {
        // Build PredictionInput from RuntimeTensor slices
        let mut prediction_input = PredictionInput::new()
            .map_err(|e| InferenceError::Runtime(Box::new(e)))?;

        for tensor in inputs {
            let data_type = element_type_to_multi_array(tensor.dtype);
            match tensor.dtype {
                ElementType::Float32 => {
                    let f32_data: &[f32] = bytemuck::cast_slice(&tensor.data);
                    prediction_input
                        .add_multi_array(&tensor.name, &tensor.shape, data_type, f32_data)
                        .map_err(|e| InferenceError::Runtime(Box::new(e)))?;
                }
                ElementType::Float16 => {
                    prediction_input
                        .add_multi_array_raw(&tensor.name, &tensor.shape, data_type, &tensor.data)
                        .map_err(|e| InferenceError::Runtime(Box::new(e)))?;
                }
                _ => return Err(InferenceError::Decode(
                    format!("unsupported input dtype: {}", tensor.dtype)
                )),
            }
        }

        // Run prediction
        let output = self.model.predict(&prediction_input)
            .map_err(|e| InferenceError::Runtime(Box::new(e)))?;

        // Extract outputs
        let mut results = Vec::new();
        for extracted in output.extract_all() {
            results.push(RuntimeTensor {
                name: extracted.name.clone(),
                data: extracted.data.to_vec(),
                shape: extracted.shape.clone(),
                dtype: multi_array_dtype_to_element_type(extracted.data_type),
            });
        }
        Ok(results)
    }
}
```

### 4.7 MetalConfig — Builder Pattern

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MetalConfig {
    pub max_seq_len: usize,
    pub attention_tile_size: Option<usize>,
    pub prefill_chunk_size: Option<usize>,
    pub enable_turboquant: bool,
    pub rotation_seed: u64,
    pub n_bits: u8,
    pub force_cpu_dequant: bool,
    pub use_fa2_prefill: bool,
    pub fused_sdpa_tile_br: Option<usize>,
    pub fused_sdpa_tile_bc: Option<usize>,
    pub cla_config: Option<ClaConfig>,
    pub sliding_window: Option<SlidingWindowConfig>,
}

impl MetalConfig {
    /// Create a new config with sensible defaults.
    pub fn new() -> Self { Self::default() }

    pub fn with_max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len; self
    }

    pub fn with_turboquant(mut self, bits: u8) -> Self {
        self.enable_turboquant = true;
        self.n_bits = bits;
        self
    }

    pub fn with_fa2_prefill(mut self) -> Self {
        self.use_fa2_prefill = true; self
    }

    pub fn with_cla(mut self, anchor_layers: Vec<usize>) -> Self {
        self.cla_config = Some(ClaConfig { anchor_layers });
        self
    }

    pub fn with_sliding_window(mut self, window_size: usize, max_window_layers: usize) -> Self {
        self.sliding_window = Some(SlidingWindowConfig { window_size, max_window_layers });
        self
    }

    pub fn with_prefill_chunks(mut self, chunk_size: usize) -> Self {
        self.prefill_chunk_size = Some(chunk_size); self
    }

    /// Validate configuration. Returns an error if invalid.
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.max_seq_len == 0 {
            return Err(InferenceError::Decode("max_seq_len must be > 0".into()));
        }
        if self.n_bits != 4 && self.n_bits != 8 {
            return Err(InferenceError::Decode(format!("n_bits must be 4 or 8, got {}", self.n_bits)));
        }
        // ... sliding window, CLA validation ...
        Ok(())
    }
}
```

### 4.8 SamplerConfig — `#[non_exhaustive]` + Default

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub min_p: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            min_p: 0.0,
            top_k: 0,     // disabled
            top_p: 1.0,   // disabled
            repeat_penalty: 1.0,
            repeat_window: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, ..Default::default() }
    }

    pub fn with_temperature(mut self, t: f32) -> Self { self.temperature = t; self }
    pub fn with_top_p(mut self, p: f32) -> Self { self.top_p = p; self }
    pub fn with_top_k(mut self, k: usize) -> Self { self.top_k = k; self }
    pub fn with_min_p(mut self, p: f32) -> Self { self.min_p = p; self }
    pub fn with_repeat_penalty(mut self, penalty: f32, window: usize) -> Self {
        self.repeat_penalty = penalty;
        self.repeat_window = window;
        self
    }
}
```

### 4.9 Config Structs — `#[non_exhaustive]`

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AneConfig { pub max_programs: usize, pub cache_dir: Option<PathBuf>, /* ... */ }

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MlxConfig { pub max_seq_len: usize, /* ... */ }

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SpecConfig { pub max_draft_depth: usize, pub tree_width: usize, pub acceptance_threshold: f64 }

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct StreamingConfig { pub num_streams: usize, pub min_confidence: f64 }

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ClaConfig { pub anchor_layers: Vec<usize> }

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig { pub window_size: usize, pub max_window_layers: usize }

// Data structs — #[non_exhaustive] for forward compat
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RuntimeTensor { pub name: String, pub data: Vec<u8>, pub shape: Vec<usize>, pub dtype: ElementType }

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct InputFeatureDesc { pub name: String, pub shape: Vec<usize>, pub dtype: ElementType }

#[non_exhaustive]
pub struct KvAllocation { pub offset: usize, pub capacity: usize, pub used: usize }

#[non_exhaustive]
pub struct SequenceState { pub id: SequenceId, pub tokens: Vec<u32>, pub kv_allocation: KvAllocation, pub status: SequenceStatus }
```

### 4.10 Validation Methods — Return InferenceError

Replace all `Result<(), String>` with `Result<(), InferenceError>`:

```rust
impl MetalConfig {
    pub fn validate(&self) -> Result<(), InferenceError> { /* ... */ }
}
impl ClaConfig {
    pub fn validate(&self, num_layers: usize) -> Result<(), InferenceError> { /* ... */ }
}
impl MlxConfig {
    pub fn validate(&self) -> Result<(), InferenceError> { /* ... */ }
}
```

### 4.11 Generation API — Streaming & Async

The core inference loop (prefill → decode_step → sample → repeat) is low-level
and requires callers to manage sampling, stopping, grammar constraints, and
cancellation manually. The generation API provides a high-level, composable
layer that handles all of this.

#### Design goals

1. **Synchronous `Iterator`** as the primary interface — no async runtime
   dependency in the core crate.
2. **Cancellation** via a lightweight token, not runtime-specific channels.
3. **Grammar constraints** integrated directly — no separate wrapper needed.
4. **Callback-based alternative** using `ControlFlow` for inline streaming
   without allocating an iterator.
5. **Async adapter** behind an optional feature flag for server use cases.

#### GenerateRequest — Builder

All generation parameters in a single request object. This replaces the
pattern of users manually wiring sampler + stop logic + grammar.

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Prompt token IDs.
    pub prompt_tokens: Vec<u32>,
    /// Sampling configuration.
    pub sampler: SamplerConfig,
    /// Maximum number of tokens to generate (excluding prompt).
    pub max_tokens: usize,
    /// Token IDs that signal end of generation.
    pub stop_tokens: Vec<u32>,
    /// Optional grammar constraint.
    pub grammar: Option<Arc<CompiledGrammar>>,
}

impl GenerateRequest {
    /// Create a new request with the given prompt tokens.
    /// Uses default sampler, 256 max tokens, and DEFAULT_EOS_TOKENS.
    pub fn new(prompt_tokens: Vec<u32>) -> Self {
        Self {
            prompt_tokens,
            sampler: SamplerConfig::default(),
            max_tokens: 256,
            stop_tokens: Vec::new(), // empty = use model's EOS tokens from ModelInfo
            grammar: None,
        }
    }

    /// Explicitly set stop tokens. If empty (the default), the generation
    /// loop consults [`ModelInfo::eos_tokens()`] from the engine. This
    /// ensures model-specific EOS tokens (e.g. Llama's `[2]`, Qwen's
    /// `[151643, 151645]`) are used automatically without hardcoding.
    pub fn with_stop_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.stop_tokens = tokens; self
    }

    pub fn with_sampler(mut self, config: SamplerConfig) -> Self {
        self.sampler = config; self
    }

    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n; self
    }

    pub fn with_grammar(mut self, grammar: Arc<CompiledGrammar>) -> Self {
        self.grammar = Some(grammar); self
    }
}
```

#### GenerateEvent & FinishReason

```rust
/// Events emitted during generation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum GenerateEvent {
    /// Prompt processing (prefill) has completed.
    ///
    /// Emitted once, before the first `Token` event. Useful for progress
    /// reporting on long prompts — callers know prefill is happening.
    PromptProcessed {
        /// Number of prompt tokens processed.
        prompt_tokens: usize,
        /// Wall-clock time for the prefill step.
        elapsed: Duration,
    },
    /// A new token was generated.
    Token {
        /// The sampled token ID.
        token: u32,
        /// Log-probability of the sampled token.
        logprob: f32,
        /// Zero-based position within the generated output (not counting prompt).
        position: usize,
    },
    /// Generation has finished.
    Finished {
        reason: FinishReason,
        /// Number of tokens generated (excluding prompt).
        tokens_generated: usize,
        /// Number of prompt tokens processed.
        prompt_tokens: usize,
    },
}

/// Why generation stopped.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token (EOS) was generated.
    Stop,
    /// Reached the `max_tokens` limit.
    MaxTokens,
    /// Cancelled by the caller via [`CancellationToken`].
    Cancelled,
    /// Grammar reached an accepting state.
    GrammarComplete,
}
```

#### CancellationToken

A lightweight, cloneable handle for cooperative cancellation. No async
runtime dependency — just an `AtomicBool` behind an `Arc`.

```rust
/// Cooperative cancellation token.
///
/// Clone this token and share it with other threads or an async runtime.
/// Call [`cancel()`](Self::cancel) to signal the generation loop to stop.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self { cancelled: Arc::new(AtomicBool::new(false)) }
    }

    /// Signal cancellation. The next `TokenStream::next()` call will
    /// yield `GenerateEvent::Finished { reason: Cancelled, .. }`.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

impl Default for CancellationToken {
    fn default() -> Self { Self::new() }
}
```

#### TokenStream — Synchronous Iterator

The primary streaming interface. Wraps an engine and yields
`GenerateEvent` items. Handles prefill on first iteration, then decode
steps until a stop condition is met.

```rust
/// Streaming token generator.
///
/// Created via [`TokenStream::new()`]. Implements [`Iterator`] so it
/// works with `for` loops, `.take()`, `.map()`, `.collect()`, etc.
///
/// ```rust
/// let cancel = CancellationToken::new();
/// let stream = TokenStream::new(&mut engine, request, &cancel);
///
/// for event in stream {
///     match event? {
///         GenerateEvent::Token { token, .. } => print_token(token),
///         GenerateEvent::Finished { reason, .. } => break,
///     }
/// }
/// ```
pub struct TokenStream<'a> {
    engine: &'a mut dyn InferenceEngine,
    sampler: Sampler,
    request: GenerateRequest,
    cancel: &'a CancellationToken,
    grammar_state: Option<GrammarState>,
    // internal state
    position: usize,
    prefilled: bool,
    finished: bool,
    pending_finish: Option<FinishReason>,
    logits: Logits,
    effective_stop_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
}

impl<'a> TokenStream<'a> {
    /// Create a new token stream.
    ///
    /// Prefill is deferred to the first call to `next()`, so construction
    /// is cheap and infallible.
    pub fn new(
        engine: &'a mut dyn InferenceEngine,
        request: GenerateRequest,
        cancel: &'a CancellationToken,
    ) -> Self {
        let grammar_state = request.grammar.as_ref().map(|g| {
            GrammarState::new(Arc::clone(g))
        });
        let sampler = Sampler::new(request.sampler.clone());
        Self {
            engine,
            sampler,
            request,
            cancel,
            grammar_state,
            position: 0,
            prefilled: false,
            finished: false,
            pending_finish: None,
            logits: Vec::new(),
            generated_tokens: Vec::new(),
            effective_stop_tokens: Vec::new(),
        }
    }

    /// Tokens generated so far (useful for recovery after an error).
    ///
    /// When `decode_step()` fails mid-generation, the iterator yields
    /// `Some(Err(e))` then `None`. Call this method after the error to
    /// retrieve any tokens generated before the failure.
    pub fn tokens_so_far(&self) -> &[u32] { &self.generated_tokens }
}

impl Iterator for TokenStream<'_> {
    type Item = Result<GenerateEvent, InferenceError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            // Emit deferred Finished event (e.g., after max_tokens final Token).
            if let Some(reason) = self.pending_finish.take() {
                return Some(Ok(GenerateEvent::Finished {
                    reason,
                    tokens_generated: self.position,
                    prompt_tokens: self.request.prompt_tokens.len(),
                }));
            }
            return None;
        }

        // ── Prefill on first call ──
        if !self.prefilled {
            self.prefilled = true;
            let start = std::time::Instant::now();
            match self.engine.prefill(&self.request.prompt_tokens) {
                Ok(logits) => {
                    self.logits = logits;
                    // Resolve stop tokens: use explicit list, or fall back to model's EOS.
                    if self.request.stop_tokens.is_empty() {
                        self.effective_stop_tokens = self.engine
                            .model_info().eos_tokens.clone();
                    } else {
                        self.effective_stop_tokens = self.request.stop_tokens.clone();
                    }
                    return Some(Ok(GenerateEvent::PromptProcessed {
                        prompt_tokens: self.request.prompt_tokens.len(),
                        elapsed: start.elapsed(),
                    }));
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            }
        }

        // ── Check cancellation ──
        if self.cancel.is_cancelled() {
            self.finished = true;
            return Some(Ok(GenerateEvent::Finished {
                reason: FinishReason::Cancelled,
                tokens_generated: self.position,
                prompt_tokens: self.request.prompt_tokens.len(),
            }));
        }

        // ── Apply grammar mask (if constrained) ──
        if let Some(state) = &self.grammar_state {
            let mask = state.token_mask();
            crate::sampling::apply_token_mask(&mut self.logits, &mask);
        }

        // ── Sample ──
        let token = self.sampler.sample(&mut self.logits);
        let logprob = self.logits.get(token as usize)
            .copied().unwrap_or(f32::NEG_INFINITY);
        self.generated_tokens.push(token);

        // ── Advance grammar ──
        if let Some(state) = &mut self.grammar_state {
            state.advance(token);
            if state.is_complete() {
                self.finished = true;
                return Some(Ok(GenerateEvent::Finished {
                    reason: FinishReason::GrammarComplete,
                    tokens_generated: self.position + 1,
                    prompt_tokens: self.request.prompt_tokens.len(),
                }));
            }
        }

        // ── Check stop conditions ──
        if self.effective_stop_tokens.contains(&token) {
            self.finished = true;
            return Some(Ok(GenerateEvent::Finished {
                reason: FinishReason::Stop,
                tokens_generated: self.position,
                prompt_tokens: self.request.prompt_tokens.len(),
            }));
        }

        self.position += 1;

        if self.position >= self.request.max_tokens {
            self.finished = true;
            // Emit the final token. The *next* call to next() will
            // emit Finished { reason: MaxTokens } (see top of fn).
            self.pending_finish = Some(FinishReason::MaxTokens);
            return Some(Ok(GenerateEvent::Token {
                token, logprob, position: self.position - 1,
            }));
        }

        // ── Decode next ──
        match self.engine.decode_step(token) {
            Ok(logits) => self.logits = logits,
            Err(e) => {
                self.finished = true;
                return Some(Err(e));
            }
        }

        Some(Ok(GenerateEvent::Token {
            token,
            logprob,
            position: self.position - 1,
        }))
    }
}
```

#### Callback-Based Generation

For callers who prefer inline processing without an iterator. Uses
`std::ops::ControlFlow` for natural cancellation:

```rust
/// Run generation with a per-event callback.
///
/// The callback receives each [`GenerateEvent`] and returns:
/// - `ControlFlow::Continue(())` to keep generating
/// - `ControlFlow::Break(())` to cancel
///
/// ```rust
/// generate_with_callback(&mut engine, &request, |event| {
///     match &event {
///         GenerateEvent::Token { token, .. } => {
///             print!("{}", detokenize(*token));
///             std::io::stdout().flush().unwrap();
///             ControlFlow::Continue(())
///         }
///         GenerateEvent::Finished { .. } => ControlFlow::Break(()),
///     }
/// })?;
/// ```
pub fn generate_with_callback(
    engine: &mut dyn InferenceEngine,
    request: &GenerateRequest,
    mut on_event: impl FnMut(GenerateEvent) -> ControlFlow<(), ()>,
) -> Result<GenerateResult, InferenceError> {
    let cancel = CancellationToken::new();
    let stream = TokenStream::new(engine, request.clone(), &cancel);

    let mut tokens = Vec::new();
    let mut finish_reason = FinishReason::MaxTokens;

    for event in stream {
        let event = event?;
        match &event {
            GenerateEvent::PromptProcessed { .. } => {}
            GenerateEvent::Token { token, .. } => tokens.push(*token),
            GenerateEvent::Finished { reason, .. } => finish_reason = *reason,
            _ => {} // forward-compatible with future variants
        }
        if on_event(event) == ControlFlow::Break(()) {
            finish_reason = FinishReason::Cancelled;
            break;
        }
    }

    Ok(GenerateResult {
        tokens,
        finish_reason,
        prompt_tokens: request.prompt_tokens.len(),
    })
}
```

#### Convenience: Collect All Tokens

For non-streaming use cases where you just want the final output:

```rust
/// Run generation to completion and collect all output tokens.
///
/// ```rust
/// let result = generate(&mut engine, &request)?;
/// println!("Generated {} tokens, reason: {:?}", result.tokens.len(), result.finish_reason);
/// ```
pub fn generate(
    engine: &mut dyn InferenceEngine,
    request: &GenerateRequest,
) -> Result<GenerateResult, InferenceError> {
    let cancel = CancellationToken::new();
    let stream = TokenStream::new(engine, request.clone(), &cancel);

    let mut tokens = Vec::new();
    let mut finish_reason = FinishReason::MaxTokens;

    for event in stream {
        match event? {
            GenerateEvent::Token { token, .. } => tokens.push(token),
            GenerateEvent::Finished { reason, .. } => {
                finish_reason = reason;
                break;
            }
        }
    }

    Ok(GenerateResult {
        tokens,
        finish_reason,
        prompt_tokens: request.prompt_tokens.len(),
    })
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// Generated token IDs (excluding prompt).
    pub tokens: Vec<u32>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// Number of prompt tokens processed.
    pub prompt_tokens: usize,
}
```

#### Partial Error Recovery

When `decode_step` fails mid-generation (e.g., GPU OOM at token 450 of 512),
the tokens generated so far are valuable. `GenerateError` preserves them:

```rust
/// Generation error with partial results.
#[derive(Debug)]
pub struct GenerateError {
    /// The underlying inference error.
    pub source: InferenceError,
    /// Tokens generated before the error occurred.
    /// Empty if the error happened during prefill.
    pub partial_tokens: Vec<u32>,
    /// Number of prompt tokens processed (0 if prefill failed).
    pub prompt_tokens: usize,
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "generation failed after {} tokens: {}", self.partial_tokens.len(), self.source)
    }
}

impl std::error::Error for GenerateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
```

The `generate()` and `generate_with_callback()` functions return
`Result<GenerateResult, GenerateError>` instead of
`Result<GenerateResult, InferenceError>`, so callers can recover partial
output:

```rust
match generate(&mut engine, &request) {
    Ok(result) => println!("Done: {} tokens", result.tokens.len()),
    Err(e) => {
        eprintln!("Failed: {}", e.source);
        if !e.partial_tokens.is_empty() {
            eprintln!("Recovered {} tokens", e.partial_tokens.len());
            // Use partial_tokens instead of discarding all work
        }
    }
}
```

#### Async Streaming — Feature-Gated

For server and async use cases, an optional `async` feature provides
`AsyncTokenStream`. This is **runtime-agnostic** — it uses channels
internally and spawns the synchronous engine on a blocking thread.

The async adapter lives behind a feature flag so the core crate has
zero async runtime dependency:

```toml
# Cargo.toml
[features]
async = ["dep:tokio"]

[dependencies.tokio]
version = "1"
features = ["sync", "rt"]
optional = true
```

```rust
#[cfg(feature = "async")]
pub mod generate_async {
    use super::*;
    use tokio::sync::mpsc;

    /// Async streaming token generator.
    ///
    /// Created via [`spawn()`]. Runs the synchronous engine on a
    /// blocking thread and streams events through a channel.
    ///
    /// Implements [`tokio_stream::Stream`] if `tokio-stream` is available,
    /// otherwise use [`next()`](Self::next) directly.
    pub struct AsyncTokenStream {
        receiver: mpsc::Receiver<Result<GenerateEvent, InferenceError>>,
        cancel: CancellationToken,
    }

    impl AsyncTokenStream {
        /// Cancel the generation. The engine will stop after the current
        /// decode step completes.
        pub fn cancel(&self) {
            self.cancel.cancel();
        }

        /// Receive the next event, or `None` if the stream is finished.
        pub async fn next(&mut self) -> Option<Result<GenerateEvent, InferenceError>> {
            self.receiver.recv().await
        }
    }

    /// Spawn a generation task on a blocking thread and return an async stream.
    ///
    /// The engine is moved into the blocking thread. When the stream is dropped
    /// or cancelled, the engine is returned via the join handle.
    ///
    /// ```rust
    /// let mut stream = generate_async::spawn(engine, request, 32).await;
    ///
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         GenerateEvent::Token { token, .. } => {
    ///             send_sse(token).await;
    ///         }
    ///         GenerateEvent::Finished { .. } => break,
    ///     }
    /// }
    /// ```
    pub fn spawn(
        mut engine: Box<dyn InferenceEngine + Send>,
        request: GenerateRequest,
        buffer: usize,
    ) -> AsyncTokenStream {
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let (tx, rx) = mpsc::channel(buffer);

        tokio::task::spawn_blocking(move || {
            let stream = TokenStream::new(&mut *engine, request, &cancel_clone);
            for event in stream {
                if tx.blocking_send(event).is_err() {
                    // Receiver dropped — stop generating.
                    break;
                }
            }
        });

        AsyncTokenStream { receiver: rx, cancel }
    }
}
```

#### Thread Safety Notes

`InferenceEngine` implementations are generally `Send` but **not** `Sync`:
they hold mutable GPU/accelerator state that cannot be shared across threads.

For multi-threaded serving, the recommended patterns are:
- **One engine per worker thread** — the `BatchScheduler` manages scheduling
  across sequences, not across threads.
- **`Mutex<Box<dyn InferenceEngine + Send>>`** — for simple cases where
  contention is acceptable.
- **The async adapter** (`generate_async::spawn`) — moves the engine into a
  dedicated blocking thread and communicates via channels.

### 4.12 Memory Management

GPU memory is the primary constraint for on-device LLM inference. The API
must surface memory information so callers can make informed decisions about
model loading, sequence lengths, and batch sizes — catching OOM before it
happens, not after.

The core types (`MemoryUsage`, `MemoryEstimator`, `QuantLevel`,
`KvQuantLevel`) live in `mil-rs` so both `ironmill-compile` and
`ironmill-inference` can use them (see §12.1). `ironmill-inference`
re-exports them at the crate root.

```rust
/// Memory information for a loaded engine.
///
/// Returned by [`InferenceEngine::memory_info()`]. All sizes in bytes.
/// Uses `u64` instead of `usize` to correctly represent sizes > 4 GB
/// on 32-bit targets.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total GPU memory consumed by model weights.
    pub weight_memory: u64,
    /// Current KV cache memory usage.
    pub kv_cache_memory: u64,
    /// Peak KV cache memory (maximum across all sequences).
    pub kv_cache_peak: u64,
    /// Estimated memory for temporary compute buffers (activations, etc.).
    pub scratch_memory: u64,
}

/// Memory estimation utilities (no engine required).
///
/// These are static methods that estimate memory requirements from config
/// alone, before loading. Useful for deciding whether a model will fit.
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// Estimate total weight memory for a model config + quantization level.
    ///
    /// ```rust
    /// let est = MemoryEstimator::weight_memory(&config, QuantLevel::Int4);
    /// if est > available_gpu_memory() {
    ///     eprintln!("Model requires {:.1} GB, only {:.1} GB available",
    ///         est as f64 / 1e9, available_gpu_memory() as f64 / 1e9);
    /// }
    /// ```
    pub fn weight_memory(config: &ModelConfig, quant: QuantLevel) -> u64 { /* ... */ }

    /// Estimate KV cache memory for a given sequence length and batch size.
    pub fn kv_cache_memory(
        config: &ModelConfig,
        seq_len: usize,
        batch_size: usize,
        kv_quant: Option<KvQuantLevel>,
    ) -> u64 { /* ... */ }

    /// Estimate maximum sequence length that fits in the given memory budget.
    pub fn max_seq_len_for_budget(
        config: &ModelConfig,
        budget_bytes: u64,
        batch_size: usize,
    ) -> usize { /* ... */ }
}

/// Quantization level for memory estimation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum QuantLevel { Fp16, Int8, Int4, Int2 }

/// KV cache quantization level.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum KvQuantLevel { None, Int8, TurboInt4, TurboInt8 }
```

Add to `InferenceEngine`:

```rust
pub trait InferenceEngine: Send {
    // ... existing methods ...

    /// Current memory usage. Returns `None` if the backend doesn't track memory.
    fn memory_usage(&self) -> Option<MemoryUsage> { None }
}
```

### 4.13 Model Capabilities

After loading a model, callers need to query its properties without
inspecting the original config files. `ModelInfo` provides a uniform
view across all backends:

```rust
/// Runtime information about a loaded model.
///
/// Populated during engine loading from `ModelConfig` and weight metadata.
/// Available via [`InferenceEngine::model_info()`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model architecture (Llama, Qwen, Gemma, etc.).
    pub architecture: Architecture,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum context length supported by the model.
    pub max_context_len: usize,
    /// Quantization applied to weights.
    pub weight_quantization: String,
    /// EOS token IDs for this model.
    ///
    /// Sourced from the model's `generation_config.json` or `tokenizer_config.json`.
    /// Used by [`TokenStream`] when no explicit stop tokens are provided.
    pub eos_tokens: Vec<u32>,
    /// Number of parameters (approximate, in millions).
    pub param_count_m: f32,
    /// Whether the model uses grouped-query attention (num_kv_heads < num_heads).
    pub uses_gqa: bool,
    /// Whether the model uses multi-latent attention (DeepSeek-style).
    pub uses_mla: bool,
    /// Head dimension.
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (may differ from attention heads in GQA).
    pub num_kv_heads: usize,
}

impl ModelInfo {
    /// Create from a ModelConfig, populating all fields.
    pub fn from_config(config: &ModelConfig) -> Self { /* ... */ }
}
```

### 4.14 Batch Runner

The current batch inference API (§7.9) requires manual orchestration of
`BatchScheduler` + engine. `BatchRunner` composes them into a managed loop:

```rust
/// Managed batch inference loop.
///
/// Owns a `BatchScheduler` and drives an `InferenceEngine` in a
/// continuous-batching loop. Handles sequence lifecycle, scheduling,
/// and output collection.
///
/// ```rust
/// let mut runner = BatchRunner::new(engine, BatchRunnerConfig::default());
///
/// // Submit requests (non-blocking — they queue internally)
/// let handle1 = runner.submit(GenerateRequest::new(tokens1))?;
/// let handle2 = runner.submit(GenerateRequest::new(tokens2))?;
///
/// // Drive the loop (call from your event loop or dedicated thread)
/// while runner.has_pending() {
///     let events = runner.step()?;
///     for (handle, event) in events {
///         match event {
///             GenerateEvent::Token { token, .. } => { /* stream to client */ }
///             GenerateEvent::Finished { .. } => { /* notify client */ }
///             _ => {}
///         }
///     }
/// }
/// ```
pub struct BatchRunner {
    engine: Box<dyn BatchInferenceEngine>,
    scheduler: BatchScheduler,
    config: BatchRunnerConfig,
    sequences: HashMap<SequenceHandle, SequenceState>,
}

/// Handle to a submitted generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceHandle(u64);

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BatchRunnerConfig {
    /// Maximum KV cache pool size in bytes.
    pub kv_pool_size: usize,
    /// Maximum number of concurrent sequences.
    pub max_batch_size: usize,
    /// Scheduling policy.
    pub policy: SchedulingPolicy,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-come, first-served.
    Fcfs,
    /// Shortest sequence first (minimize latency for short prompts).
    ShortestFirst,
}

impl Default for BatchRunnerConfig {
    fn default() -> Self {
        Self {
            kv_pool_size: 1024 * 1024 * 1024, // 1 GB
            max_batch_size: 8,
            policy: SchedulingPolicy::Fcfs,
        }
    }
}

impl BatchRunner {
    pub fn new(engine: Box<dyn BatchInferenceEngine>, config: BatchRunnerConfig) -> Self { /* ... */ }

    /// Submit a new generation request. Returns a handle for tracking.
    pub fn submit(&mut self, request: GenerateRequest) -> Result<SequenceHandle, InferenceError> { /* ... */ }

    /// Run one step of the batch loop: select batch, prefill/decode, sample, emit events.
    pub fn step(&mut self) -> Result<Vec<(SequenceHandle, GenerateEvent)>, InferenceError> { /* ... */ }

    /// Cancel a specific sequence.
    pub fn cancel(&mut self, handle: SequenceHandle) { /* ... */ }

    /// Whether any sequences are still pending or in-progress.
    pub fn has_pending(&self) -> bool { /* ... */ }

    /// Number of active sequences.
    pub fn active_count(&self) -> usize { /* ... */ }
}
```

---

## 5. ironmill-cli

### 5.1 Quantize Flag — ValueEnum

Replace the free-form `--quantize` string with a proper clap `ValueEnum`:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum QuantizeArg {
    /// No quantization.
    #[default]
    None,
    /// FP16 quantization (weights and activations).
    Fp16,
    /// INT8 symmetric quantization.
    Int8,
    /// Mixed FP16/INT8 precision (predefined layer assignment).
    MixedFp16Int8,
    /// Activation-aware weight quantization (INT4, requires --cal-data).
    Awq,
    /// INT4 group quantization (group size 128).
    Int4,
    /// GPTQ optimal weight quantization (requires --cal-data).
    Gptq,
    /// D2Quant extreme low-bit (2-3 bit) quantization.
    D2quant,
}
```

### 5.2 Target Flag — ValueEnum

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum TargetArg {
    /// All available compute units on this platform.
    #[default]
    All,
    /// CPU only.
    CpuOnly,
    /// CPU and GPU (Metal on macOS).
    CpuAndGpu,
    /// CPU and Neural Engine (macOS only).
    CpuAndNe,
    /// GPU-only Metal backend (macOS only).
    Gpu,
    // Future variants (non-breaking due to #[non_exhaustive]):
    // Cuda,       — NVIDIA GPU
    // CudaAndCpu, — NVIDIA GPU + CPU
}
```

### 5.3 Loss / Optimizer — ValueEnum

```rust
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum LossFunctionArg {
    #[default]
    CrossEntropy,
    Mse,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum OptimizerArg {
    #[default]
    Sgd,
    Adam,
}
```

### 5.4 Remove Redundant Flags

```rust
// BEFORE (confusing):
//   --merge-lora       (default true)
//   --no-merge-lora    (conflicts_with merge_lora)

// AFTER (clear):
//   --no-merge-lora    (boolean flag, default false)

#[arg(long, help = "Disable automatic LoRA adapter merging")]
no_merge_lora: bool,

// Remove: --merge-lora entirely
```

### 5.5 Hide Unimplemented Flags

```rust
// Hide --emit-adapter and --adapter until implemented
#[arg(long, hide = true)]
emit_adapter: bool,

#[arg(long, hide = true, value_name = "PATH")]
adapters: Vec<PathBuf>,
```

### 5.6 Output Discipline — stderr for Diagnostics

All informational output moves to `stderr`:

```rust
// BEFORE:
println!("Converting {} model...", arch);
println!("  {} tensors loaded", count);

// AFTER:
eprintln!("Converting {} model...", arch);
eprintln!("  {} tensors loaded", count);
```

Structured output (JSON validation, pipeline manifests) stays on `stdout`.

Compiler failures become hard errors:

```rust
// BEFORE (compile_output):
//   Downgrades coremlcompiler failure to warning, exits 0

// AFTER:
fn compile_output(mlpackage: &Path, output_dir: &Path) -> Result<PathBuf> {
    let status = Command::new(compiler_path)
        .args([/* ... */])
        .status()
        .context("failed to run coremlcompiler")?;

    if !status.success() {
        bail!("coremlcompiler failed with exit code {}", status.code().unwrap_or(-1));
    }
    Ok(output_dir.join(/* ... */))
}
```

### 5.7 Validate — Exit Non-Zero on Analysis Failure

```rust
// BEFORE:
// If CoreML → MIL conversion fails for ANE analysis, prints note and returns Ok(())

// AFTER:
// Prints note to stderr AND returns non-zero exit unless --lenient is passed
#[arg(long, help = "Continue validation even if full analysis is not possible")]
lenient: bool,
```

### 5.8 Update README

Document all 5 commands:

```
COMMANDS:
  compile           Compile a model to CoreML, ANE, Metal, or CUDA format
  inspect           Print model structure and metadata
  validate          Validate model for target hardware compatibility
  compile-pipeline  Compile a multi-stage pipeline from a TOML manifest
  pipeline-report   Compare two pipeline configurations
```

### 5.9 Full CompileArgs (Target State)

```rust
#[derive(clap::Args)]
pub struct CompileArgs {
    /// Input model (ONNX, SafeTensors dir, GGUF, mlmodel, mlpackage).
    input: String,

    /// Output path.
    #[arg(short, long)]
    output: Option<String>,

    /// Target compute unit.
    #[arg(short, long, value_enum, default_value_t = TargetArg::All)]
    target: TargetArg,

    /// Quantization method.
    #[arg(short, long, value_enum, default_value_t = QuantizeArg::None)]
    quantize: QuantizeArg,

    /// Calibration data directory (required for AWQ, GPTQ, QuIP#).
    #[arg(long, value_name = "DIR")]
    cal_data: Option<PathBuf>,

    /// TOML quantization config for mixed-precision.
    #[arg(long, value_name = "PATH")]
    quantize_config: Option<PathBuf>,

    /// Palettization bit-width (2, 4, 6, or 8).
    #[arg(long, value_name = "BITS")]
    palettize: Option<u8>,

    /// PolarQuant bit-width.
    #[arg(long, value_name = "BITS")]
    polar_quantize: Option<u8>,

    /// Enable QuIP# quantization.
    #[arg(long)]
    quip_sharp: bool,

    /// Bit-width override for GPTQ/D2Quant.
    #[arg(long, value_name = "N")]
    bits: Option<u8>,

    /// Disable operator fusion passes.
    #[arg(long)]
    no_fusion: bool,

    /// Override input shapes (repeatable: NAME:D0xD1xD2).
    #[arg(long, value_name = "NAME:SHAPE")]
    input_shape: Vec<String>,

    /// Disable automatic LoRA merging.
    #[arg(long)]
    no_merge_lora: bool,

    // ── On-device training ──
    /// Layers to make updatable (comma-separated).
    #[arg(long, value_name = "LAYERS")]
    updatable_layers: Option<String>,
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f64,
    #[arg(long, default_value_t = 10)]
    epochs: i64,
    #[arg(long, value_enum, default_value_t = LossFunctionArg::CrossEntropy)]
    loss_function: LossFunctionArg,
    #[arg(long, value_enum, default_value_t = OptimizerArg::Sgd)]
    optimizer: OptimizerArg,

    // ── Model splitting ──
    #[arg(long, value_name = "N")]
    split_draft_layers: Option<usize>,
    #[arg(long)]
    moe_split: bool,
    #[arg(long, conflicts_with = "moe_split")]
    moe_bundle: bool,
    #[arg(long, value_name = "K")]
    moe_fuse_topk: Option<usize>,

    // ── Pipeline / TOML config ──
    #[arg(long, value_name = "PATH")]
    pipeline_config: Option<PathBuf>,

    // ── ANE-specific ──
    #[arg(long)]
    annotate_compute_units: bool,
    #[arg(long, value_name = "SIZE")]
    ane_memory_budget: Option<String>,
    #[arg(long, help = "Generate ANE-compatible operations")]
    ane: bool,

    // ── Runtime ──
    #[arg(long, value_enum, default_value_t = RuntimeArg::CoreMl)]
    runtime: RuntimeArg,
    #[arg(long, value_enum, default_value_t = KvQuantArg::None)]
    kv_quant: KvQuantArg,
    #[arg(long)]
    kv_quant_qjl: bool,
    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,
}
```

---

## 6. Cross-Cutting Concerns

### 6.1 Documentation Standards

Add `#![warn(missing_docs)]` to all three library crates (not `deny` initially
— use `warn` to adopt incrementally).

Every public item must have:
- A one-line summary
- For complex types: a `# Examples` section with runnable doctests

### 6.2 Crate Architecture — Platform-Portable Core

The current `ironmill-inference` crate has a top-level
`compile_error!("ironmill-inference only supports macOS")` which prevents
the crate from compiling on any other platform. This blocks future CUDA
support and makes the platform-agnostic core (traits, sampling, grammar,
serving) unavailable on Linux/Windows.

**Target architecture**: one crate, feature-gated backends.

```
ironmill-inference/
├── src/
│   ├── lib.rs              ← NO compile_error!, always compiles
│   ├── engine.rs           ← InferenceEngine trait (portable)
│   ├── types.rs            ← ElementType, RuntimeTensor, etc. (portable)
│   ├── sampling.rs         ← Sampler, SamplerConfig (portable)
│   ├── grammar/            ← Grammar, ConstrainedDecoder (portable)
│   ├── serving/            ← BatchScheduler, KvPool (portable)
│   ├── speculative/        ← SpeculativeEngine (portable)
│   ├── cache/              ← PrefixCache, RadixTree (portable)
│   ├── calibration/        ← (portable)
│   │
│   ├── metal/              ← #[cfg(feature = "metal")]
│   ├── ane/                ← #[cfg(feature = "ane")]
│   ├── coreml/             ← #[cfg(feature = "coreml")]
│   ├── mlx/                ← #[cfg(feature = "mlx")]
│   └── cuda/               ← #[cfg(feature = "cuda")] (future)
```

**Cargo.toml feature flags:**

```toml
[features]
default = ["metal", "coreml", "ane"]

# Platform-specific backends (each gated on target_os internally)
metal   = ["dep:ironmill-metal-sys"]
ane     = ["dep:ironmill-ane-sys", "dep:ironmill-iosurface"]
coreml  = ["dep:ironmill-coreml-sys"]
mlx     = ["dep:ironmill-mlx-sys"]

# Future backends
cuda    = ["dep:ironmill-cuda-sys"]
# vulkan = ["dep:ash"]
```

**Module gating pattern** (lib.rs):

```rust
// Core — always available, any platform
pub mod cache;
pub mod calibration;
pub mod engine;
pub mod grammar;
pub mod sampling;
pub mod serving;
pub mod speculative;
pub mod types;

// Platform-specific backends — feature + OS gated
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
#[cfg(all(feature = "ane", target_os = "macos"))]
pub mod ane;
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub mod coreml;
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub mod mlx;

// Future backends
#[cfg(feature = "cuda")]
pub mod cuda;
```

This means:
- `cargo check -p ironmill-inference --no-default-features` compiles on
  **any platform** (Linux, Windows, macOS) and gives you the full
  trait/type/sampling/grammar/serving API.
- `cargo check -p ironmill-inference` on macOS compiles with Metal + CoreML +
  ANE by default.
- A future `--features cuda` on Linux compiles the CUDA backend without
  touching any Apple code.

**What a future CUDA backend would look like:**

```rust
// src/cuda/mod.rs
#[cfg(feature = "cuda")]
pub mod config;
pub mod error;
pub mod inference;
pub mod weights;

// src/cuda/config.rs
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CudaConfig {
    pub max_seq_len: usize,
    pub device_id: usize,
    pub use_flash_attention: bool,
    pub use_tensor_cores: bool,
    pub memory_fraction: f32,
}

impl Default for CudaConfig { /* ... */ }

impl CudaConfig {
    pub fn new() -> Self { Self::default() }
    pub fn with_max_seq_len(mut self, len: usize) -> Self { self.max_seq_len = len; self }
    pub fn with_device(mut self, id: usize) -> Self { self.device_id = id; self }
    pub fn with_flash_attention(mut self) -> Self { self.use_flash_attention = true; self }
}

// src/cuda/inference.rs
pub struct CudaInference { /* ... */ }

pub struct CudaArtifacts<'a> {
    pub weights: &'a dyn WeightProvider,
    pub config: &'a CudaConfig,
}

impl CudaInference {
    /// Load model from CUDA-specific artifacts.
    /// Type-safe: compile-time guarantee you pass CudaArtifacts.
    pub fn load(config: CudaConfig, artifacts: &CudaArtifacts<'_>)
        -> Result<Self, InferenceError> { /* ... */ }
}

impl InferenceEngine for CudaInference {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> { /* ... */ }
    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError> { /* ... */ }
    fn reset(&mut self) { /* ... */ }
    fn seq_pos(&self) -> usize { /* ... */ }
    fn truncate_to(&mut self, pos: usize) { /* ... */ }
    fn model_info(&self) -> &ModelInfo { /* ... */ }
}
```

**No existing code changes required** — the CUDA backend is purely additive:
- New crate: `ironmill-cuda-sys` (CUDA/cuDNN FFI bindings)
- New module: `ironmill-inference/src/cuda/`
- New feature: `cuda` in `Cargo.toml`
- New variant: `ComputeUnit::Cuda` (non-breaking due to `#[non_exhaustive]`)
- New CLI flag value: `--target cuda` (non-breaking due to `#[non_exhaustive]` on `TargetArg`)

### 6.3 Crate Dependency Graph

```
ironmill-cli
  └─ ironmill-core (high-level API)
  │    ├─ ironmill-compile
  │    ├─ ironmill-inference
  │    └─ tokenizers (feature-gated)
  │
  └─ ironmill-compile (compilation API)
  │    └─ mil-rs (IR, passes, weights, shared types)
  │
  └─ ironmill-inference (runtime API)
       ├─ mil-rs (shared types: Architecture, QuantLevel, ModelInfo,
       │          MemoryEstimator, QuantizationInfo, ModelConfig)
       ├─ ironmill-metal-sys     [feature = "metal",  macOS only]
       ├─ ironmill-ane-sys       [feature = "ane",    macOS only]
       ├─ ironmill-coreml-sys    [feature = "coreml", macOS only]
       ├─ ironmill-mlx-sys       [feature = "mlx",    macOS only]
       └─ ironmill-cuda-sys      [feature = "cuda",   future]
```

Users should never need to depend on `mil-rs` directly for normal usage.
`ironmill-compile` re-exports the relevant IR types, and
`ironmill-inference` re-exports the relevant weight and estimation types.

### 6.4 Progress & Telemetry

All long-running operations (compilation, weight loading, prefill) must
support progress callbacks. This enables progress bars, logging, metrics,
and user-facing status updates without baking any specific UI into the core.

```rust
/// Sink for progress updates during long-running operations.
///
/// Implement this trait to receive progress events. The default
/// implementation is a no-op (silent).
///
/// ```rust
/// struct StderrProgress;
/// impl ProgressSink for StderrProgress {
///     fn on_stage(&self, name: &str) {
///         eprintln!("  → {name}...");
///     }
///     fn on_progress(&self, current: usize, total: usize, message: &str) {
///         eprintln!("  [{current}/{total}] {message}");
///     }
/// }
/// ```
pub trait ProgressSink: Send + Sync {
    /// A new named stage has started (e.g., "loading weights", "quantizing").
    fn on_stage(&self, _name: &str) {}

    /// Progress within the current stage.
    fn on_progress(&self, _current: usize, _total: usize, _message: &str) {}

    /// A stage has completed.
    fn on_stage_complete(&self, _name: &str, _elapsed: Duration) {}

    /// A non-fatal warning.
    fn on_warning(&self, _message: &str) {}
}

/// No-op progress sink (the default when none is provided).
pub struct NullProgress;
impl ProgressSink for NullProgress {}
```

Integration points:

```rust
// CompileTarget::compile() takes &dyn ProgressSink (see §3.7)

// Model loading (see §10)
impl Model {
    pub fn from_pretrained(path: impl AsRef<Path>) -> ModelBuilder {
        ModelBuilder { progress: Box::new(NullProgress), /* ... */ }
    }
}

impl ModelBuilder {
    pub fn with_progress(mut self, sink: impl ProgressSink + 'static) -> Self {
        self.progress = Box::new(sink); self
    }
}

// Engine loading
impl MetalInference {
    pub fn load_with_progress(
        config: MetalConfig,
        artifacts: &MetalArtifacts<'_>,
        progress: &dyn ProgressSink,
    ) -> Result<Self, InferenceError> { /* ... */ }
}
```

---

## 7. Usage Examples

### 7.1 Compile a SafeTensors Model (Library)

```rust
use ironmill_compile::{
    CompileBuilder, Quantization, TargetComputeUnit,
    weights::{SafeTensorsProvider, Architecture},
    templates::{weights_to_program, TemplateOptions},
};
use mil_rs::PassPipeline;

// Load weights
let provider = SafeTensorsProvider::load("./Qwen3-0.6B/")?;
println!("Architecture: {}", provider.config().architecture);

// Convert to MIL IR
let result = weights_to_program(&provider)?;
for w in &result.warnings {
    eprintln!("warning: {w}");
}

// Quantize + compile
let output = CompileBuilder::new("./Qwen3-0.6B/")
    .quantize(Quantization::Fp16)
    .target(TargetComputeUnit::CpuAndNeuralEngine)
    .output("./output.mlpackage")
    .compile()
    .build()?;

println!("Compiled: {}", output.mlpackage.display());
```

### 7.2 Compile with Custom Pass Pipeline

```rust
use ironmill_compile::{
    GpuCompileBuilder,
    weights::SafeTensorsProvider,
};
use mil_rs::PassPipeline;

let provider = SafeTensorsProvider::load("./model/")?;
let pipeline = PassPipeline::new()
    .with_fp16()?
    .without_fusion();

let weight_provider = GpuCompileBuilder::new("./model/")
    .with_pass_pipeline(pipeline)
    .build()?;
```

### 7.3 Run Inference with Metal Backend

```rust
use ironmill_inference::{
    InferenceEngine, SamplerConfig, Sampler,
    metal::{MetalInference, MetalConfig, MetalBundleProvider, MetalArtifacts},
};

// Configure
let config = MetalConfig::new()
    .with_max_seq_len(4096)
    .with_turboquant(8)
    .with_fa2_prefill();
config.validate()?;

// Load model
let mut engine = MetalInference::new(config)?;
let bundle = MetalBundleProvider::open("./model.ironml-gpu")?;
let artifacts = MetalArtifacts {
    weights: &bundle,
    config: &engine.config(),
};
engine.load(&artifacts)?;

// Run inference
let prompt_tokens = vec![1u32, 15043, 29892]; // "Hello,"
let mut logits = engine.prefill(&prompt_tokens)?;

let sampler = Sampler::new(
    SamplerConfig::default()
        .with_temperature(0.7)
        .with_top_p(0.9)
);

let mut output_tokens = Vec::new();
for _ in 0..100 {
    let token = sampler.sample(&logits).unwrap();
    if is_eos_token(token) { break; }
    output_tokens.push(token);
    logits = engine.decode_step(token)?;
}
```

### 7.4 Run Inference with CoreML Backend

```rust
use ironmill_inference::{
    RuntimeBackend, RuntimeModel,
    coreml::CoremlBackend,
    types::{RuntimeTensor, ElementType, build_dummy_inputs},
};
use ironmill_coreml_sys::ComputeUnits;

// Load compiled model
let backend = CoremlBackend::new(ComputeUnits::All);
let model = backend.load("./model.mlmodelc".as_ref())?;

// Inspect inputs
let inputs = model.input_description();
for input in &inputs {
    println!("{}: {:?} ({:?})", input.name, input.shape, input.dtype);
}

// Run prediction
let dummy = build_dummy_inputs(&inputs);
let outputs = model.predict(&dummy)?;
for out in &outputs {
    println!("output {}: {:?}", out.name, out.shape);
}
```

### 7.5 Run Inference with ANE Backend

```rust
use ironmill_inference::{
    RuntimeBackend, RuntimeModel,
    ane::{AneDirectBackend, AneConfig},
    types::build_dummy_inputs,
};

// Load ANE bundle
let backend = AneDirectBackend::new(AneConfig::default());
let model = backend.load("./model.ane-bundle".as_ref())?;

// Run prediction
let inputs = model.input_description();
let dummy = build_dummy_inputs(&inputs);
let outputs = model.predict(&dummy)?;
```

### 7.6 Autoregressive Decode with ANE

```rust
use ironmill_inference::{
    InferenceEngine, Sampler, SamplerConfig,
    ane::{AneInference, AneConfig, HardwareAneDevice},
};

let device = HardwareAneDevice::new()?;
let config = AneConfig::default();
let mut engine = AneInference::from_bundle("./model.ane-decode", device, &config)?;

let logits = engine.prefill(&prompt_tokens)?;

let sampler = Sampler::new(SamplerConfig::greedy());
let mut output = Vec::new();
let mut last_logits = logits;

for _ in 0..256 {
    let token = sampler.sample(&last_logits).unwrap();
    if is_eos_token(token) { break; }
    output.push(token);
    last_logits = engine.decode_step(token)?;
}
```

### 7.7 Grammar-Constrained Generation

```rust
use ironmill_inference::{
    InferenceEngine, ConstrainedDecoder, Sampler, SamplerConfig,
    grammar::{CompiledGrammar, GrammarState, json_schema_to_grammar},
};

// Compile a JSON schema grammar
let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}"#;
let bnf = json_schema_to_grammar(schema)?;
let grammar = CompiledGrammar::new(&bnf, vocab_size)?;
let state = GrammarState::new(&grammar, "root")?;

// Wrap engine with grammar constraints
let mut decoder = ConstrainedDecoder::new(&mut engine, state);
let sampler = Sampler::new(SamplerConfig::default().with_temperature(0.3));

let mut output_tokens = Vec::new();
let mut last_token = prompt_tokens.last().copied().unwrap();

loop {
    let logits = decoder.constrained_decode_step(last_token)?;
    let token = sampler.sample(&logits).unwrap();
    decoder.accept_token(token);

    if decoder.is_complete() { break; }
    output_tokens.push(token);
    last_token = token;
}
```

### 7.8 Speculative Decoding

```rust
use ironmill_inference::{
    InferenceEngine, Sampler, SamplerConfig,
    speculative::{SpeculativeEngine, SpecConfig, speculative_decode},
};

let spec_config = SpecConfig {
    max_draft_depth: 5,
    tree_width: 1,
    acceptance_threshold: 0.1,
};

let mut spec = SpeculativeEngine::new(engine, spec_config);
spec.load_draft_head(&draft_weights)?;

let mut logits = spec.engine_mut().prefill(&prompt_tokens)?;
let mut last_token = sample(&logits);
let mut last_hidden = vec![0.0f32; hidden_dim]; // from prefill

loop {
    let accepted = speculative_decode(&mut spec, last_token, &last_hidden)?;
    for &tok in &accepted {
        if is_eos_token(tok) { return Ok(()); }
        output.push(tok);
    }
    last_token = *accepted.last().unwrap();
}
```

### 7.9 Continuous Batching (Server)

```rust
use ironmill_inference::serving::{BatchScheduler, InferenceBatch};

let mut scheduler = BatchScheduler::new(
    1024 * 1024,  // KV pool size
    8,            // max batch size
);

// Add sequences
let id1 = scheduler.add_sequence(vec![1, 2, 3])?;
let id2 = scheduler.add_sequence(vec![4, 5, 6])?;

// Select batch and run
let batch_ids = scheduler.select_batch();
let batch = InferenceBatch::assemble(&scheduler, &batch_ids)?;

// After engine processes batch, advance each sequence
scheduler.advance(id1, next_token_1);
scheduler.advance(id2, next_token_2);

// When done
scheduler.complete_sequence(id1);
scheduler.remove_sequence(id1)?;
```

### 7.10 Build Custom MIL IR Program

```rust
use mil_rs::{
    Program, Function, Block, Operation, Value, TensorType, ScalarType,
    PassPipeline, write_mlpackage, program_to_model,
};

let mut block = Block::new();
block.add_op(
    Operation::new("linear", "fc1")
        .with_input("x", Value::Reference("input".into()))
        .with_input("weight", Value::Reference("w1".into()))
        .with_output("fc1_out"),
);
block.outputs = vec!["fc1_out".into()];

let func = Function::new("main")
    .with_input("input", TensorType::new(ScalarType::Float16, vec![1, 768]));

let mut program = Program::new("1.0");
program.add_function(Function { body: block, ..func });

// Run optimization passes
let pipeline = PassPipeline::new().with_fp16()?;
let report = pipeline.run(&mut program)?;
println!("Pipeline: {} passes in {:?}", report.pass_results.len(), report.total_elapsed());

// Serialize
let model = program_to_model(&program, 8)?;
write_mlpackage(&model, "./output.mlpackage")?;
```

### 7.11 Pipeline Compilation from TOML

TOML manifest (`pipeline.toml`):
```toml
name = "my-pipeline"

[[stages]]
name = "embeddings"
source = "./model/"
quantize = "fp16"
component = "embeddings"

[[stages]]
name = "transformer"
source = "./model/"
quantize = "int4"
component = "transformer"
depends_on = ["embeddings"]

[[stages]]
name = "lm-head"
source = "./model/"
quantize = "fp16"
component = "lm-head"
depends_on = ["transformer"]
```

CLI usage:
```bash
ironmill compile-pipeline pipeline.toml -o ./output/
```

### 7.12 CLI Common Workflows

```bash
# Basic FP16 compilation
ironmill compile ./model/ -q fp16 -o ./output.mlpackage

# INT4 quantization for GPU
ironmill compile ./model/ -q int4 --target gpu -o ./output.ironml-gpu

# AWQ with calibration data
ironmill compile ./model/ -q awq --cal-data ./calibration/ -o ./output.mlpackage

# ANE-targeted compilation
ironmill compile ./model/ --ane --target cpu-and-ne -o ./output.mlpackage

# Metal runtime with TurboQuant KV compression
ironmill compile ./model/ -q fp16 --target gpu --runtime ane-direct \
    --kv-quant turbo-int8 --max-seq-len 4096

# PolarQuant
ironmill compile ./model/ --polar-quantize 4 --target gpu

# Inspect model structure
ironmill inspect ./model.mlpackage

# Validate for ANE compatibility (JSON output)
ironmill validate ./model.mlpackage --format json

# Compare pipeline configurations
ironmill pipeline-report ./model.onnx --config-a baseline.toml --config-b optimized.toml
```

### 7.13 Streaming Token Generation

```rust
use ironmill_inference::{
    InferenceEngine, CancellationToken,
    generate::{GenerateRequest, GenerateEvent, TokenStream, FinishReason},
    sampling::SamplerConfig,
    metal::{MetalInference, MetalConfig},
};

// Set up engine (same as §7.3)
let config = MetalConfig::new().with_max_seq_len(4096);
let mut engine = MetalInference::new(config)?;
// ... load model ...

// Build request
let request = GenerateRequest::new(prompt_tokens)
    .with_sampler(SamplerConfig::default().with_temperature(0.7).with_top_p(0.9))
    .with_max_tokens(512)
    .with_stop_tokens(vec![2, 151643]);

// Stream tokens — cancel from another thread if needed
let cancel = CancellationToken::new();
let cancel_handle = cancel.clone();
// e.g. ctrlc::set_handler(move || cancel_handle.cancel());

let stream = TokenStream::new(&mut engine, request, &cancel);

for event in stream {
    match event? {
        GenerateEvent::Token { token, .. } => {
            print!("{}", detokenize(token));
        }
        GenerateEvent::Finished { reason, tokens_generated, .. } => {
            eprintln!("\n[done: {reason:?}, {tokens_generated} tokens]");
            break;
        }
    }
}
```

### 7.14 Callback-Based Streaming

```rust
use std::ops::ControlFlow;
use ironmill_inference::generate::{generate_with_callback, GenerateRequest, GenerateEvent};

let request = GenerateRequest::new(prompt_tokens)
    .with_max_tokens(100);

let result = generate_with_callback(&mut engine, &request, |event| {
    match &event {
        GenerateEvent::Token { token, .. } => {
            print!("{}", detokenize(*token));
            std::io::stdout().flush().unwrap();
            ControlFlow::Continue(())
        }
        GenerateEvent::Finished { .. } => ControlFlow::Break(()),
    }
})?;

println!("\nGenerated {} tokens", result.tokens.len());
```

### 7.15 Async Server Streaming (SSE)

```rust
use ironmill_inference::generate::{
    GenerateRequest, GenerateEvent,
    generate_async,  // requires feature = "async"
};

async fn handle_completion(engine: Box<dyn InferenceEngine + Send>, prompt: Vec<u32>) {
    let request = GenerateRequest::new(prompt)
        .with_sampler(SamplerConfig::default().with_temperature(0.8))
        .with_max_tokens(1024);

    let mut stream = generate_async::spawn(engine, request, 32);

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            GenerateEvent::Token { token, .. } => {
                send_sse_event(&format!("data: {}\n\n", detokenize(token))).await;
            }
            GenerateEvent::Finished { reason, .. } => {
                send_sse_event(&format!("data: [DONE] {reason:?}\n\n")).await;
                break;
            }
        }
    }
}
```

### 7.16 Non-Streaming (Collect All Tokens)

```rust
use ironmill_inference::generate::{generate, GenerateRequest};

let request = GenerateRequest::new(prompt_tokens)
    .with_sampler(SamplerConfig::greedy())
    .with_max_tokens(200);

let result = generate(&mut engine, &request)?;
println!("Output: {}", detokenize_all(&result.tokens));
println!("Finish reason: {:?}", result.finish_reason);
```

### 7.17 High-Level API — Minimal Example

```rust
use ironmill_core::{Model, GenParams};

let mut model = Model::from_pretrained("./Qwen3-0.6B/")
    .build()?;

let output = model.generate("What is Rust?", GenParams::default())?;
println!("{}", output.text);
```

### 7.18 High-Level API — Streaming

```rust
use ironmill_core::{Model, GenParams};

let mut model = Model::from_pretrained("./Qwen3-0.6B/")
    .device(Device::Metal)
    .max_seq_len(8192)
    .build()?;

for chunk in model.stream("Explain monads simply", GenParams::default())? {
    let chunk = chunk?;
    print!("{}", chunk.text);
    if chunk.finished {
        println!("\n[done: {:?}]", chunk.finish_reason.unwrap());
    }
}
```

### 7.19 Chat Session

```rust
use ironmill_core::{Model, GenParams};

let mut model = Model::from_pretrained("./Llama-3.2-3B/")
    .build()?;

let mut chat = model.chat()
    .system("You are a helpful, concise assistant.")
    .params(GenParams::default().with_temperature(0.8))
    .build();

let r1 = chat.say("What's the capital of France?")?;
println!("Assistant: {}", r1.text);

let r2 = chat.say("What about Germany?")?;
println!("Assistant: {}", r2.text);

// Context is maintained — the model knows we're talking about capitals
println!("History: {} turns", chat.history().len());
```

### 7.20 JIT Loading (No Compile Step)

```rust
use ironmill_inference::{
    metal::{MetalInference, MetalConfig},
};
use ironmill_compile::weights::SafeTensorsProvider;
use ironmill_core::jit::TransformPipeline;

// Load weights + apply INT4 quantization directly — no MIL IR
let provider = SafeTensorsProvider::load("./Qwen3-0.6B/")?;
let transforms = TransformPipeline::new().with_int4(128);
let config = MetalConfig::new().with_max_seq_len(4096);

let mut engine = MetalInference::load_jit(config, &provider, &transforms)?;

// Ready to run inference immediately
let logits = engine.prefill(&[1, 15043, 29892])?;
```

### 7.21 Progress Tracking

```rust
use ironmill_core::{Model, GenParams};
use ironmill_inference::ProgressSink;
use std::time::Duration;

struct CliProgress;
impl ProgressSink for CliProgress {
    fn on_stage(&self, name: &str) {
        eprintln!("⏳ {name}...");
    }
    fn on_progress(&self, current: usize, total: usize, msg: &str) {
        eprintln!("  [{current}/{total}] {msg}");
    }
    fn on_stage_complete(&self, name: &str, elapsed: Duration) {
        eprintln!("  ✓ {name} ({elapsed:.1?})");
    }
}

let mut model = Model::from_pretrained("./Qwen3-0.6B/")
    .with_progress(CliProgress)
    .build()?;
// Output:
// ⏳ detecting model format...
//   ✓ detecting model format (1.2ms)
// ⏳ loading weights...
//   [0/291] model.embed_tokens.weight
//   [291/291] lm_head.weight
//   ✓ loading weights (2.1s)
// ⏳ allocating KV cache...
//   ✓ allocating KV cache (45ms)
```

### 7.22 Memory Estimation

```rust
use ironmill_inference::{MemoryEstimator, QuantLevel, KvQuantLevel};
use ironmill_compile::weights::SafeTensorsProvider;

let provider = SafeTensorsProvider::load("./Llama-3.2-3B/")?;
let config = provider.config();

// Check if model fits before loading
let weight_mem = MemoryEstimator::weight_memory(config, QuantLevel::Int4);
let kv_mem = MemoryEstimator::kv_cache_memory(config, 4096, 1, None);
let total = weight_mem + kv_mem;

println!("Weight memory:  {:.1} GB", weight_mem as f64 / 1e9);
println!("KV cache (4K):  {:.1} GB", kv_mem as f64 / 1e9);
println!("Total estimate: {:.1} GB", total as f64 / 1e9);

// Find max sequence length for a 4 GB budget
let max_len = MemoryEstimator::max_seq_len_for_budget(config, 4_000_000_000, 1);
println!("Max seq len in 4 GB: {max_len}");
```

---

## 8. Migration Guide

### 8.1 For Consumers of mil-rs

**Pattern match on enums** — add a wildcard arm:
```rust
// BEFORE:
match scalar_type {
    ScalarType::Float16 => { ... }
    ScalarType::Float32 => { ... }
    // Compiles today, breaks when we add BFloat16
}

// AFTER:
match scalar_type {
    ScalarType::Float16 => { ... }
    ScalarType::Float32 => { ... }
    other => panic!("unsupported scalar type: {other:?}"),
}
```

**Struct construction** — use builders instead of literals:
```rust
// BEFORE:
let config = ModelConfig {
    architecture: Architecture::Llama,
    hidden_size: 4096,
    // ... 12 more fields ...
};

// AFTER:
let config = ModelConfig::new(Architecture::Llama)
    .with_hidden_size(4096)
    // ...
    ;
```

### 8.2 For Consumers of ironmill-inference

**InferenceEngine::load** — loading is now a method on each backend, not on the trait:
```rust
// BEFORE:
let mut engine = MetalInference::new(config)?;
engine.load(&artifacts as &dyn std::any::Any)?;

// AFTER:
let engine = MetalInference::load(config, &artifacts)?;
// Type-safe: compile-time guarantee you pass MetalArtifacts.
// Returns a ready-to-use engine (no two-step init).
```

**InferenceEngine trait** — `load()` is removed from the trait. The trait is
now object-safe (`dyn InferenceEngine` works). `model_info()` is added:
```rust
// BEFORE:
let engine: Box<dyn InferenceEngine> = /* impossible with associated types */;

// AFTER:
let engine: Box<dyn InferenceEngine> = Box::new(
    MetalInference::load(config, &artifacts)?
);
let info = engine.model_info();
println!("Architecture: {:?}", info.architecture);
```

**Error handling** — `InferenceError::Runtime` now boxes the source:
```rust
// BEFORE:
if let InferenceError::Runtime(msg) = &err {
    eprintln!("Runtime error: {msg}");
}

// AFTER:
if let InferenceError::Runtime(source) = &err {
    eprintln!("Runtime error: {source}");
    if let Some(metal_err) = source.downcast_ref::<MetalError>() {
        eprintln!("Metal details: {metal_err:?}");
    }
}
```

**GenerateEvent** — new `PromptProcessed` variant:
```rust
// BEFORE:
for event in stream {
    match event? {
        GenerateEvent::Token { .. } => { /* ... */ }
        GenerateEvent::Finished { .. } => break,
    }
}

// AFTER:
for event in stream {
    match event? {
        GenerateEvent::PromptProcessed { prompt_tokens, elapsed } => {
            eprintln!("Prefill: {prompt_tokens} tokens in {elapsed:.1?}");
        }
        GenerateEvent::Token { .. } => { /* ... */ }
        GenerateEvent::Finished { .. } => break,
    }
}
```

**Stop tokens** — now model-aware by default:
```rust
// BEFORE:
let request = GenerateRequest::new(tokens)
    .with_stop_tokens(vec![2]);  // had to know the model's EOS

// AFTER:
let request = GenerateRequest::new(tokens);
// Stop tokens are automatically pulled from ModelInfo::eos_tokens
// Only specify explicitly if you need to override the model's default.
```

**High-level API** (new, optional):
```rust
// BEFORE: ~25 lines to set up inference
let provider = SafeTensorsProvider::load("./model/")?;
let config = MetalConfig::new().with_max_seq_len(4096);
let mut engine = MetalInference::new(config)?;
engine.load(&artifacts)?;
let tokens = tokenize(&prompt);
// ... sampling loop ...

// AFTER: 2 lines
let mut model = Model::from_pretrained("./model/").build()?;
let output = model.generate("Hello!", GenParams::default())?;
```

**MetalConfig** — struct literal still works, but builder is preferred:
```rust
// Still compiles (because fields are pub + #[non_exhaustive] + Default):
let config = MetalConfig { max_seq_len: 4096, ..MetalConfig::default() };

// Preferred:
let config = MetalConfig::new().with_max_seq_len(4096);
```

### 8.3 For CLI Users

**Flag changes:**
| Old | New | Notes |
|-----|-----|-------|
| `--merge-lora` | *(removed)* | Merging is always the default |
| `-q none` | `-q none` | No change |
| `-q fp16` | `-q fp16` | No change |
| `-q mixed-fp16-int8` | `-q mixed-fp16-int8` | No change |
| `-t all` | `-t all` | No change |
| `-t cpu-and-ne` | `-t cpu-and-ne` | No change |
| `--loss-function cross-entropy` | `--loss-function cross-entropy` | Now validated by clap |
| `--optimizer sgd` | `--optimizer sgd` | Now validated by clap |

**Output changes:**
- Progress/status messages now go to stderr. Scripts that parse stdout
  will see cleaner output.
- Compiler failures are now hard errors (exit 1) instead of warnings.
  Add `|| true` to scripts that tolerated this.

---

## 9. Tokenizer Abstraction

The core inference API operates entirely on `Vec<u32>` token IDs. This is
correct for the low-level engine interface — tokenization is model-specific
and orthogonal to GPU dispatch. However, every practical consumer must pair
the engine with a tokenizer, and the current API provides no help with this.

### 9.1 Tokenizer Trait

```rust
/// Text ↔ token ID conversion.
///
/// This trait abstracts over tokenizer implementations (SentencePiece,
/// BPE, Unigram, etc.) so that the high-level API can tokenize/detokenize
/// without depending on a specific tokenizer library.
pub trait Tokenizer: Send + Sync {
    /// Encode text into token IDs.
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;

    /// Decode token IDs back to text.
    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError>;

    /// Decode a single token to its text representation.
    /// Returns empty string for special tokens.
    fn decode_token(&self, token: u32) -> Result<String, TokenizerError> {
        self.decode(&[token])
    }

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// EOS token IDs for this tokenizer/model.
    fn eos_tokens(&self) -> &[u32];

    /// BOS token ID, if the model uses one.
    fn bos_token(&self) -> Option<u32> { None }

    /// Apply the model's chat template to a conversation.
    /// Returns the formatted prompt string ready for encoding.
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String, TokenizerError> {
        // Default: concatenate messages with role prefixes.
        // Override for model-specific templates (Llama, Qwen, Gemma each differ).
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
        }
        Ok(prompt)
    }
}

/// A message in a conversation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role: "system", "user", "assistant", "tool".
    pub role: String,
    /// Message content.
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("tokenizer not loaded: {0}")]
    NotLoaded(String),
    #[error("encoding error: {0}")]
    Encode(String),
    #[error("decoding error: {0}")]
    Decode(String),
    #[error("chat template error: {0}")]
    Template(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

### 9.2 HuggingFace Tokenizer Adapter

The primary implementation wraps the `tokenizers` crate (HuggingFace's Rust
tokenizer library), which supports all common tokenizer types:

```rust
/// HuggingFace tokenizers adapter.
///
/// Loads `tokenizer.json` from a model directory. Supports SentencePiece,
/// BPE, Unigram, and WordPiece tokenizers.
///
/// ```rust
/// let tokenizer = HfTokenizer::from_model_dir("./Qwen3-0.6B/")?;
/// let tokens = tokenizer.encode("Hello, world!")?;
/// let text = tokenizer.decode(&tokens)?;
/// ```
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    eos_tokens: Vec<u32>,
    bos_token: Option<u32>,
    chat_template: Option<String>,
}

impl HfTokenizer {
    /// Load from a model directory containing `tokenizer.json`.
    ///
    /// Also reads `tokenizer_config.json` for EOS/BOS tokens and
    /// `generation_config.json` for chat template, if present.
    pub fn from_model_dir(path: impl AsRef<Path>) -> Result<Self, TokenizerError> { /* ... */ }

    /// Load from an explicit `tokenizer.json` path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> { /* ... */ }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.inner.encode(text, false)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| TokenizerError::Encode(e.to_string()))
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        self.inner.decode(tokens, true)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    fn vocab_size(&self) -> usize { self.inner.get_vocab_size(false) }
    fn eos_tokens(&self) -> &[u32] { &self.eos_tokens }
    fn bos_token(&self) -> Option<u32> { self.bos_token }

    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String, TokenizerError> {
        if let Some(template) = &self.chat_template {
            // Render Jinja2-style template (subset supported)
            render_chat_template(template, messages)
        } else {
            // Fallback to default implementation
            let mut prompt = String::new();
            for msg in messages {
                prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
            }
            Ok(prompt)
        }
    }
}
```

### 9.3 Dependency Isolation

The `tokenizers` crate is a heavy dependency (regex, unicode, serde). It
lives behind a feature flag:

```toml
[features]
default = ["tokenizer"]
tokenizer = ["dep:tokenizers"]
```

Users who want a different tokenizer implementation (e.g., `sentencepiece`,
a C++ binding, or a custom one) can disable the default feature and provide
their own `Tokenizer` impl.

### 9.4 Compilation Boundary

The `Tokenizer` trait and `HfTokenizer` live in `ironmill-core`, **not**
in `ironmill-compile`. Calibration-based quantization passes (AWQ, GPTQ)
accept pre-tokenized `&[Vec<u32>]` sequences — the caller is responsible
for tokenization. This keeps `ironmill-compile` free of tokenizer
dependencies and maintains a clean dependency graph (see §12.2).

---

## 10. ironmill-core: High-Level API

`ironmill-core` is the top-level entry point for users who want to run
inference without manually wiring weight providers, compilation targets,
engine backends, and tokenizers. It composes all the lower-level crates
into a single `Model` abstraction.

**Design principle**: the high-level API should make the common case trivial
and the advanced case possible. Users who need fine-grained control drop down
to the `ironmill-inference` / `ironmill-compile` APIs directly.

### 10.1 Device

```rust
/// Target compute device for inference.
///
/// Determines which backend the `Model` uses. `Auto` inspects the
/// current platform and selects the best available device.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Automatically select the best device for this platform.
    /// macOS: Metal GPU. Linux: CUDA (if available), else CPU.
    #[default]
    Auto,
    /// Metal GPU (macOS only).
    Metal,
    /// Apple Neural Engine via CoreML (macOS only).
    Ane,
    /// CoreML runtime (CPU/GPU/ANE per CoreML's scheduler).
    CoreMl,
    /// NVIDIA GPU via CUDA (Linux/Windows).
    Cuda,
    /// CPU-only inference.
    Cpu,
}

impl Device {
    /// Return all devices available on this platform.
    ///
    /// Uses compile-time feature gates and runtime hardware probing.
    /// `Cpu` is always available. `Auto` is always included (resolves
    /// at load time).
    ///
    /// ```rust
    /// let devices = Device::available();
    /// // macOS:            [Auto, Metal, Ane, CoreMl, Cpu]
    /// // Linux with CUDA:  [Auto, Cuda, Cpu]
    /// // Linux without:    [Auto, Cpu]
    /// ```
    pub fn available() -> Vec<Device> { /* ... */ }
}
```

### 10.2 Model — The Top-Level Abstraction

```rust
/// A loaded, ready-to-use language model.
///
/// `Model` owns everything needed for inference: the engine, tokenizer,
/// model info, and (optionally) the weight provider. It provides the
/// simplest possible interface for text generation.
///
/// # Quick Start
///
/// ```rust
/// use ironmill_core::{Model, GenParams};
///
/// let model = Model::from_pretrained("./Qwen3-0.6B/")?.build()?;
///
/// // Non-streaming
/// let output = model.generate("What is Rust?", GenParams::default())?;
/// println!("{}", output.text);
///
/// // Streaming
/// for chunk in model.stream("What is Rust?", GenParams::default())? {
///     print!("{}", chunk?.text);
/// }
/// ```
pub struct Model {
    engine: Box<dyn InferenceEngine>,
    tokenizer: Box<dyn Tokenizer>,
    info: ModelInfo,
}

impl Model {
    /// Start building a model from a pretrained model directory.
    ///
    /// The directory should contain:
    /// - Weight files (SafeTensors, GGUF)
    /// - `config.json` (model architecture)
    /// - `tokenizer.json` (tokenizer definition)
    /// - Optionally: `generation_config.json`, `tokenizer_config.json`
    ///
    /// The builder auto-detects architecture, selects the appropriate backend,
    /// and loads weights. No compilation step for the GPU-direct path.
    pub fn from_pretrained(path: impl AsRef<Path>) -> ModelBuilder {
        ModelBuilder::new(path.as_ref().to_path_buf(), ModelSource::Pretrained)
    }

    /// Load from a pre-compiled artifact (.ironml-gpu, .mlmodelc, etc.).
    ///
    /// Skips weight loading and compilation — the artifact is already
    /// in the target format. Still needs a tokenizer (from the original
    /// model directory or provided explicitly).
    pub fn from_compiled(path: impl AsRef<Path>) -> ModelBuilder {
        ModelBuilder::new(path.as_ref().to_path_buf(), ModelSource::Compiled)
    }

    /// Generate text (non-streaming). Returns the complete output.
    ///
    /// ```rust
    /// let output = model.generate("Explain monads", GenParams::default())?;
    /// println!("{}", output.text);
    /// println!("Generated {} tokens in {:?}", output.token_count, output.elapsed);
    /// ```
    pub fn generate(&mut self, prompt: &str, params: GenParams) -> Result<TextOutput, ModelError> {
        let tokens = self.tokenizer.encode(prompt)?;
        let request = params.to_generate_request(tokens, &self.info);
        let result = generate(&mut *self.engine, &request)?;
        let text = self.tokenizer.decode(&result.tokens)?;
        Ok(TextOutput {
            text,
            tokens: result.tokens,
            token_count: result.tokens.len(),
            prompt_token_count: result.prompt_tokens,
            finish_reason: result.finish_reason,
            elapsed: Duration::default(), // filled by timing wrapper
        })
    }

    /// Generate text with streaming output.
    ///
    /// Returns an iterator of text chunks. Each chunk contains one or
    /// more decoded tokens.
    pub fn stream<'a>(
        &'a mut self,
        prompt: &str,
        params: GenParams,
    ) -> Result<TextStream<'a>, ModelError> {
        let tokens = self.tokenizer.encode(prompt)?;
        let request = params.to_generate_request(tokens, &self.info);
        Ok(TextStream::new(&mut *self.engine, &self.tokenizer, request))
    }

    /// Start or continue a chat session.
    ///
    /// ```rust
    /// let mut chat = model.chat()
    ///     .system("You are a helpful assistant.")
    ///     .build();
    ///
    /// let response = chat.say("Hello!")?;
    /// println!("{}", response.text);
    ///
    /// let response = chat.say("Tell me more")?;
    /// println!("{}", response.text);
    /// ```
    pub fn chat(&mut self) -> ChatSessionBuilder<'_> {
        ChatSessionBuilder::new(self)
    }

    /// Access the underlying engine for low-level operations.
    pub fn engine(&self) -> &dyn InferenceEngine { &*self.engine }
    pub fn engine_mut(&mut self) -> &mut dyn InferenceEngine { &mut *self.engine }

    /// Access the tokenizer.
    pub fn tokenizer(&self) -> &dyn Tokenizer { &*self.tokenizer }

    /// Model information.
    pub fn info(&self) -> &ModelInfo { &self.info }
}

/// Generation parameters for the high-level API.
///
/// Simpler than `GenerateRequest` — no token IDs, no grammar.
/// Intentionally omits repetition penalties (`repeat_penalty`,
/// `frequency_penalty`, `presence_penalty`) for simplicity. Users who
/// need those should drop down to `GenerateRequest` + `SamplerConfig`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GenParams {
    pub temperature: f32,
    pub max_tokens: usize,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: 512,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.0,
        }
    }
}

impl GenParams {
    pub fn greedy() -> Self { Self { temperature: 0.0, ..Default::default() } }

    pub fn with_temperature(mut self, t: f32) -> Self { self.temperature = t; self }
    pub fn with_max_tokens(mut self, n: usize) -> Self { self.max_tokens = n; self }
    pub fn with_top_p(mut self, p: f32) -> Self { self.top_p = p; self }

    /// Convert to the low-level `GenerateRequest`.
    fn to_generate_request(&self, tokens: Vec<u32>, info: &ModelInfo) -> GenerateRequest {
        GenerateRequest::new(tokens)
            .with_sampler(
                SamplerConfig::default()
                    .with_temperature(self.temperature)
                    .with_top_p(self.top_p)
                    .with_top_k(self.top_k)
                    .with_min_p(self.min_p)
            )
            .with_max_tokens(self.max_tokens)
            .with_stop_tokens(info.eos_tokens.clone())
    }
}

/// Output from non-streaming generation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TextOutput {
    /// Generated text (decoded tokens).
    pub text: String,
    /// Raw token IDs.
    pub tokens: Vec<u32>,
    /// Number of generated tokens.
    pub token_count: usize,
    /// Number of prompt tokens.
    pub prompt_token_count: usize,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// Wall-clock generation time.
    pub elapsed: Duration,
}
```

### 10.3 ModelBuilder — Configuration & Loading

```rust
/// Builder for constructing a [`Model`].
///
/// Handles auto-detection of model format, architecture, and backend.
/// Supports explicit overrides for every auto-detected parameter.
pub struct ModelBuilder {
    path: PathBuf,
    source: ModelSource,
    device: Device,
    quantize: Option<QuantLevel>,
    max_seq_len: usize,
    tokenizer_path: Option<PathBuf>,
    progress: Box<dyn ProgressSink>,
}

enum ModelSource { Pretrained, Compiled }

impl ModelBuilder {
    fn new(path: PathBuf, source: ModelSource) -> Self {
        Self {
            path,
            source,
            device: Device::Auto,
            quantize: None,
            max_seq_len: 4096,
            tokenizer_path: None,
            progress: Box::new(NullProgress),
        }
    }

    /// Target device for inference.
    pub fn device(mut self, device: Device) -> Self { self.device = device; self }

    /// Apply weight quantization during loading.
    /// Only used for `from_pretrained()` — compiled artifacts are already quantized.
    pub fn quantize(mut self, level: QuantLevel) -> Self {
        self.quantize = Some(level); self
    }

    /// Maximum sequence length (context window).
    pub fn max_seq_len(mut self, len: usize) -> Self { self.max_seq_len = len; self }

    /// Explicit tokenizer path (overrides auto-detection from model dir).
    pub fn tokenizer(mut self, path: impl AsRef<Path>) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_path_buf()); self
    }

    /// Progress callback for loading status.
    pub fn with_progress(mut self, sink: impl ProgressSink + 'static) -> Self {
        self.progress = Box::new(sink); self
    }

    /// Build the model: detect format, load weights, create engine, load tokenizer.
    ///
    /// For `from_pretrained()` with `Device::Metal`:
    /// 1. Detect model format (SafeTensors/GGUF) and architecture
    /// 2. Load weights into WeightProvider
    /// 3. Apply quantization (if requested) directly to weight tensors
    /// 4. Create MetalInference engine + load weights into GPU
    /// 5. Load tokenizer from model directory
    ///
    /// No MIL IR. No CoreML. No compilation step.
    pub fn build(self) -> Result<Model, ModelError> {
        let device = self.resolve_device();
        self.progress.on_stage("detecting model format");
        let format = detect_model_format(&self.path)?;

        match self.source {
            ModelSource::Pretrained => self.build_from_pretrained(device, format),
            ModelSource::Compiled => self.build_from_compiled(device),
        }
    }

    fn resolve_device(&self) -> Device {
        match self.device {
            Device::Auto => {
                #[cfg(target_os = "macos")]
                { Device::Metal }
                #[cfg(not(target_os = "macos"))]
                { Device::Cpu }
            }
            other => other,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("inference error: {0}")]
    Inference(#[from] InferenceError),
    #[error("compile error: {0}")]
    Compile(#[from] CompileError),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("model not found: {0}")]
    NotFound(PathBuf),
    #[error("unsupported device {0:?} on this platform")]
    UnsupportedDevice(Device),
    #[error("{0}")]
    Other(String),
}
```

### 10.4 TextStream — Streaming with Detokenization

```rust
/// Streaming text output from generation.
///
/// Wraps [`TokenStream`] and detokenizes each token as it arrives.
/// Handles the complexity of multi-byte characters spanning token
/// boundaries.
pub struct TextStream<'a> {
    inner: TokenStream<'a>,
    tokenizer: &'a dyn Tokenizer,
    cancel: CancellationToken,
}

/// A chunk of streamed text.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// Decoded text for this chunk.
    pub text: String,
    /// Raw token ID.
    pub token: u32,
    /// Position in the generated sequence.
    pub position: usize,
    /// Whether this is the final chunk.
    pub finished: bool,
    /// Finish reason (only set on the final chunk).
    pub finish_reason: Option<FinishReason>,
}

impl Iterator for TextStream<'_> {
    type Item = Result<TextChunk, ModelError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next()? {
                Ok(GenerateEvent::PromptProcessed { .. }) => continue,
                Ok(GenerateEvent::Token { token, position, .. }) => {
                    let text = match self.tokenizer.decode_token(token) {
                        Ok(t) => t,
                        Err(e) => return Some(Err(e.into())),
                    };
                    return Some(Ok(TextChunk {
                        text, token, position, finished: false, finish_reason: None,
                    }));
                }
                Ok(GenerateEvent::Finished { reason, tokens_generated, .. }) => {
                    return Some(Ok(TextChunk {
                        text: String::new(),
                        token: 0,
                        position: tokens_generated,
                        finished: true,
                        finish_reason: Some(reason),
                    }));
                }
                Err(e) => return Some(Err(e.into())),
            }
        }
    }
}
```

### 10.5 ChatSession — Multi-Turn Conversations

```rust
/// Multi-turn conversation manager.
///
/// Tracks conversation history, applies chat templates, and manages
/// the token budget (truncating old turns when the context window fills).
///
/// ```rust
/// let mut chat = model.chat()
///     .system("You are a concise assistant.")
///     .max_context_tokens(4096)
///     .build();
///
/// let r1 = chat.say("What is 2+2?")?;
/// println!("Assistant: {}", r1.text);
///
/// let r2 = chat.say("And 3+3?")?;
/// println!("Assistant: {}", r2.text);
///
/// // Streaming
/// for chunk in chat.say_stream("Explain gravity")? {
///     print!("{}", chunk?.text);
/// }
///
/// // Access history
/// println!("Turns so far: {}", chat.history().len());
/// ```
pub struct ChatSession<'a> {
    model: &'a mut Model,
    history: Vec<ChatMessage>,
    system_prompt: Option<String>,
    params: GenParams,
    max_context_tokens: usize,
}

pub struct ChatSessionBuilder<'a> {
    model: &'a mut Model,
    system_prompt: Option<String>,
    params: GenParams,
    max_context_tokens: usize,
}

impl<'a> ChatSessionBuilder<'a> {
    fn new(model: &'a mut Model) -> Self {
        let max_ctx = model.info().max_context_len;
        Self {
            model,
            system_prompt: None,
            params: GenParams::default(),
            max_context_tokens: max_ctx,
        }
    }

    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into()); self
    }

    pub fn params(mut self, params: GenParams) -> Self {
        self.params = params; self
    }

    pub fn max_context_tokens(mut self, n: usize) -> Self {
        self.max_context_tokens = n; self
    }

    pub fn build(self) -> ChatSession<'a> {
        let mut history = Vec::new();
        if let Some(sys) = self.system_prompt {
            history.push(ChatMessage::system(sys));
        }
        ChatSession {
            model: self.model,
            history,
            system_prompt: None,
            params: self.params,
            max_context_tokens: self.max_context_tokens,
        }
    }
}

impl ChatSession<'_> {
    /// Send a message and get a response (non-streaming).
    pub fn say(&mut self, message: &str) -> Result<TextOutput, ModelError> {
        self.history.push(ChatMessage::user(message));

        // Truncate history if we exceed the context window
        let mut prompt = self.model.tokenizer().apply_chat_template(&self.history)?;
        let mut tokens = self.model.tokenizer().encode(&prompt)?;
        while tokens.len() > self.max_context_tokens && self.history.len() > 2 {
            // Remove oldest non-system message
            let remove_idx = if self.history[0].role == "system" { 1 } else { 0 };
            self.history.remove(remove_idx);
            // Re-encode with trimmed history
            prompt = self.model.tokenizer().apply_chat_template(&self.history)?;
            tokens = self.model.tokenizer().encode(&prompt)?;
        }

        self.model.engine_mut().reset();
        let output = self.model.generate(&prompt, self.params.clone())?;

        self.history.push(ChatMessage::assistant(&output.text));
        Ok(output)
    }

    /// Send a message and stream the response.
    ///
    /// The returned `StreamingChatResponse` must be consumed. When the
    /// stream completes (or is dropped), the assistant's response is
    /// automatically appended to conversation history.
    pub fn say_stream<'a>(&'a mut self, message: &str) -> Result<StreamingChatResponse<'a>, ModelError> {
        self.history.push(ChatMessage::user(message));
        let prompt = self.model.tokenizer().apply_chat_template(&self.history)?;
        self.model.engine_mut().reset();
        let stream = self.model.stream(&prompt, self.params.clone())?;
        Ok(StreamingChatResponse {
            stream,
            history: &mut self.history,
            collected_tokens: Vec::new(),
            tokenizer: self.model.tokenizer(),
        })
    }

    /// Access conversation history.
    pub fn history(&self) -> &[ChatMessage] { &self.history }

    /// Clear conversation history (keeps system prompt).
    pub fn clear(&mut self) {
        let system = self.history.iter()
            .find(|m| m.role == "system")
            .cloned();
        self.history.clear();
        if let Some(s) = system {
            self.history.push(s);
        }
        self.model.engine_mut().reset();
    }
}

/// Streaming chat response that commits the assistant turn to history
/// when the stream completes or is dropped.
pub struct StreamingChatResponse<'a> {
    stream: TextStream<'a>,
    history: &'a mut Vec<ChatMessage>,
    collected_tokens: Vec<u32>,
    tokenizer: &'a dyn Tokenizer,
}

impl Iterator for StreamingChatResponse<'_> {
    type Item = Result<TextChunk, ModelError>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = self.stream.next()?;
        if let Ok(ref c) = chunk {
            if !c.finished {
                self.collected_tokens.push(c.token);
            }
        }
        chunk
    }
}

impl Drop for StreamingChatResponse<'_> {
    fn drop(&mut self) {
        // Commit the assistant's response to history
        if !self.collected_tokens.is_empty() {
            if let Ok(text) = self.tokenizer.decode(&self.collected_tokens) {
                self.history.push(ChatMessage::assistant(text));
            }
        }
    }
}
```

### 10.6 Crate Structure

```
ironmill-core/
├── Cargo.toml
├── src/
│   ├── lib.rs          ← Model, Device, GenParams, TextOutput
│   ├── builder.rs      ← ModelBuilder, auto-detection logic
│   ├── chat.rs         ← ChatSession, ChatSessionBuilder
│   ├── stream.rs       ← TextStream, TextChunk
│   ├── tokenizer.rs    ← Tokenizer trait, HfTokenizer
│   └── error.rs        ← ModelError, TokenizerError
```

Dependencies:

```toml
[dependencies]
ironmill-inference = { path = "../ironmill-inference" }
ironmill-compile = { path = "../ironmill-compile", optional = true }
tokenizers = { version = "0.20", optional = true }

[features]
default = ["tokenizer", "compile"]
tokenizer = ["dep:tokenizers"]
compile = ["dep:ironmill-compile"]  # enables from_pretrained() with compilation
```

When `compile` is disabled, only `Model::from_compiled()` is available.
When `tokenizer` is disabled, users must provide their own `Tokenizer` impl.

---

## 11. JIT Compilation

### 11.1 Motivation

The current Metal inference path still routes through MIL IR (via
`GpuCompileBuilder`) even though the Metal backend never uses the MIL
computation graph — it only uses the quantized weight tensors. MIL IR is
an unnecessary intermediary when quantization can be applied directly to
weight data.

JIT compilation eliminates this indirection:

```
AOT (current):  SafeTensors → MIL IR Program → PassPipeline → MilWeightProvider → MetalInference
JIT (proposed): SafeTensors → WeightProvider → TensorTransforms → MetalInference
```

| | AOT (current) | JIT (proposed) |
|---|---|---|
| **Indirection** | Weights converted to MIL ops, then back to tensors | Direct tensor access |
| **Time-to-first-token** | Slow (build MIL graph, run passes, extract) | Fast (load + transform + go) |
| **Quantization target** | MIL `Operation` nodes | Raw weight tensors |
| **Disk artifacts** | `.ironml-gpu` bundle | None required (optional cache) |
| **Graph optimizations** | Available (op fusion, layout) | Not needed (hand-written shaders) |
| **CoreML compatibility** | Yes (can also produce .mlpackage) | No (Metal/CUDA only) |

### 11.2 TensorTransform — Direct Weight Quantization

`TensorTransform` replaces `Pass` for the JIT path. It operates on
individual weight tensors, not MIL program graphs:

```rust
/// A transform applied directly to weight tensors during loading.
///
/// Unlike [`Pass`] (which operates on a MIL `Program` graph),
/// `TensorTransform` operates on raw tensor data. This avoids the
/// overhead of converting weights to/from MIL IR representation.
pub trait TensorTransform: Send + Sync {
    /// Name of this transform (for logging).
    fn name(&self) -> &str;

    /// Transform a single weight tensor in-place.
    ///
    /// The transform receives the tensor name (e.g. "model.layers.0.self_attn.q_proj.weight"),
    /// the tensor data, and the model config. It can modify the data, change the
    /// dtype, or reshape — returning the transformed tensor.
    ///
    /// Return `None` to leave the tensor unchanged.
    fn transform(
        &self,
        name: &str,
        tensor: WeightTensor<'_>,
        config: &ModelConfig,
    ) -> Result<Option<TransformedTensor>, TransformError>;
}

/// Result of a tensor transform.
pub struct TransformedTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: ScalarType,
    /// Quantization metadata (for the engine to select the right kernel).
    pub quant_info: QuantizationInfo,
}

/// Pipeline of tensor transforms, applied during weight loading.
pub struct TransformPipeline {
    transforms: Vec<Box<dyn TensorTransform>>,
}

impl TransformPipeline {
    pub fn new() -> Self { Self { transforms: Vec::new() } }

    pub fn with_int4(mut self, group_size: usize) -> Self {
        self.transforms.push(Box::new(Int4AffineTransform { group_size }));
        self
    }

    pub fn with_fp16(mut self) -> Self {
        self.transforms.push(Box::new(Fp16CastTransform));
        self
    }

    pub fn with_polar_quant(mut self, bits: u8) -> Self {
        self.transforms.push(Box::new(PolarQuantTransform { bits }));
        self
    }

    /// Apply all transforms to a weight provider, yielding transformed tensors.
    ///
    /// This is the core JIT path: weights are loaded from disk, transformed
    /// (quantized/cast), and yielded one at a time for the engine to upload
    /// to GPU memory. No intermediate MIL IR is created.
    pub fn apply<'a>(
        &'a self,
        provider: &'a dyn WeightProvider,
        progress: &'a dyn ProgressSink,
    ) -> TransformIterator<'a> { /* ... */ }
}
```

### 11.3 JIT Engine Loading

The JIT path integrates with `MetalInference` via a new loading method:

```rust
impl MetalInference {
    /// Load model using the JIT path: weights are loaded, transformed, and
    /// uploaded to GPU in a single streaming pass.
    ///
    /// ```rust
    /// let provider = SafeTensorsProvider::load("./Qwen3-0.6B/")?;
    /// let transforms = TransformPipeline::new().with_fp16();
    /// let config = MetalConfig::new().with_max_seq_len(4096);
    ///
    /// let engine = MetalInference::load_jit(config, &provider, &transforms)?;
    /// // Ready to run inference — no compile step, no .ironml-gpu bundle.
    /// ```
    pub fn load_jit(
        config: MetalConfig,
        provider: &dyn WeightProvider,
        transforms: &TransformPipeline,
    ) -> Result<Self, InferenceError> {
        // 1. Create Metal device + command queue
        // 2. For each weight tensor:
        //    a. Apply transforms (quantize/cast)
        //    b. Upload to Metal buffer
        // 3. Allocate KV cache buffers
        // 4. Compile/load Metal shader pipelines (cached on disk)
        todo!()
    }
}
```

### 11.4 Shader Caching

JIT-compiled Metal shader pipelines are cached to avoid recompilation on
subsequent runs. The cache key includes:

- Model architecture + dimensions (hidden_size, num_layers, etc.)
- Quantization configuration
- Metal device capabilities (GPU family, feature set)
- Shader source hash

```rust
/// Persistent shader pipeline cache.
///
/// Stores compiled Metal pipeline state objects to disk, keyed by
/// a hash of the shader source + specialization constants. Uses LRU
/// eviction to prevent unbounded growth across model configurations.
pub struct ShaderCache {
    cache_dir: PathBuf,
    max_size_bytes: u64,
}

impl ShaderCache {
    /// Open or create a shader cache at the given directory.
    ///
    /// Default max size is 512 MB. Use [`with_max_size()`] to override.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, std::io::Error> { /* ... */ }

    /// Default cache location: `~/.cache/ironmill/shaders/`
    pub fn default_location() -> Result<Self, std::io::Error> { /* ... */ }

    /// Set the maximum cache size in bytes. Entries are evicted LRU
    /// when the cache exceeds this limit.
    pub fn with_max_size(mut self, bytes: u64) -> Self {
        self.max_size_bytes = bytes; self
    }

    /// Look up a cached pipeline binary.
    pub fn get(&self, key: &ShaderCacheKey) -> Option<Vec<u8>> { /* ... */ }

    /// Store a compiled pipeline binary. Evicts least-recently-used
    /// entries if the cache exceeds `max_size_bytes`.
    pub fn put(&self, key: &ShaderCacheKey, binary: &[u8]) -> Result<(), std::io::Error> { /* ... */ }

    /// Remove all cached entries.
    pub fn clear(&self) -> Result<(), std::io::Error> { /* ... */ }

    /// Current total size of cached entries in bytes.
    pub fn size(&self) -> u64 { /* ... */ }
}
```

### 11.5 When to Use JIT vs AOT

```
                    ┌──────────────────────────────────────────────────────┐
                    │              Which path should I use?                │
                    └──────────────────┬───────────────────────────────────┘
                                       │
                              ┌────────▼────────┐
                              │  Need ANE?       │
                              └────┬────────┬────┘
                                   │ yes    │ no
                              ┌────▼────┐   │
                              │ CoreML  │   │
                              │ (AOT)   │   │
                              └─────────┘   │
                                       ┌────▼────────────┐
                                       │ Need to ship a   │
                                       │ compiled model?   │
                                       └──┬──────────┬────┘
                                          │ yes      │ no
                                     ┌────▼────┐     │
                                     │ AOT     │     │
                                     │ (.ironml│     │
                                     │  -gpu)  │     │
                                     └─────────┘     │
                                                ┌────▼────┐
                                                │ JIT     │
                                                │ (direct)│
                                                └─────────┘
```

- **JIT**: Development, experimentation, rapid iteration, one-off runs.
  The high-level `Model::from_pretrained()` API uses JIT by default.
- **AOT**: Production deployment, distributing models to end users,
  reproducible builds. Use `ironmill compile` CLI or `CompileTarget` API.
- **CoreML AOT**: ANE inference, on-device training, integration with
  Apple's ML ecosystem.

---

## 12. Design Decisions

### 12.1 Shared Type Placement — mil-rs

`QuantLevel`, `KvQuantLevel`, `MemoryEstimator`, `ModelInfo`, and
`Architecture` live in `mil-rs`, not `ironmill-inference`. Both
`ironmill-compile` (via `CompileTarget::estimate_size()`) and
`ironmill-inference` need these types, and both already depend on
`mil-rs`. This keeps the dependency graph clean:

```
mil-rs (shared types: Architecture, QuantLevel, ModelInfo, MemoryEstimator)
  ↑                  ↑
ironmill-compile   ironmill-inference
  ↑                  ↑
  └──── ironmill-core ────┘
```

`ironmill-inference` re-exports them at the crate root for convenience.

### 12.2 Tokenizer Boundary — Callers Tokenize

The `Tokenizer` trait and `HfTokenizer` live in `ironmill-core`.
`ironmill-compile` does **not** depend on them. Calibration-based
quantization (AWQ, GPTQ) accepts `&[Vec<u32>]` — pre-tokenized
calibration sequences. The caller is responsible for tokenization.

This is the same boundary every compiler maintains: LLVM doesn't
parse source code, and ironmill-compile doesn't tokenize text.
The CLI handles tokenization when `--cal-data` points to raw text
files, bridging the gap for end users.

### 12.3 Compiled Artifact Versioning

Every compiled bundle (`.ironml-gpu`, `.ane-bundle`) includes a
`metadata.json` at the bundle root:

```json
{
    "format_version": 1,
    "ironmill_version": "0.3.0",
    "architecture": "qwen",
    "quantization": "int4-g128",
    "hidden_size": 1536,
    "num_layers": 28,
    "compile_config": { /* ... */ }
}
```

- `format_version` is an integer, incremented on any breaking change
  to the bundle layout or weight encoding. Loaders check this first
  and emit a clear error (e.g., "bundle format v2 requires ironmill
  >= 0.5.0, you have 0.3.0") instead of silently loading corrupt data.
- `ironmill_version` is informational — the semver of the build that
  produced the bundle.
- Remaining fields are diagnostic (logged on load, useful for debugging).
