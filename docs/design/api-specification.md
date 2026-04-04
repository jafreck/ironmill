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

### 4.3 InferenceEngine — Type-Safe Loading

Replace `&dyn Any` with an associated type:

```rust
/// Autoregressive inference engine.
pub trait InferenceEngine {
    /// Artifacts required to load a model (e.g. weights + config).
    type Artifacts;

    /// Load model artifacts.
    fn load(&mut self, artifacts: &Self::Artifacts) -> Result<(), InferenceError>;

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
}

// Backend implementations:
impl InferenceEngine for MetalInference {
    type Artifacts = MetalArtifacts<'_>;
    // ...
}

impl InferenceEngine for MlxInference {
    type Artifacts = MlxArtifacts<'_>;
    // ...
}
```

**Note**: This removes `dyn InferenceEngine` support (can't have associated
types with dynamic dispatch without boxing). If dynamic dispatch is needed,
provide a `DynInferenceEngine` wrapper:

```rust
/// Type-erased inference engine for dynamic dispatch.
pub struct DynInferenceEngine {
    inner: Box<dyn InferenceEngineObj>,
}

trait InferenceEngineObj {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError>;
    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError>;
    fn reset(&mut self);
    fn seq_pos(&self) -> usize;
    fn truncate_to(&mut self, pos: usize);
    fn max_seq_len(&self) -> usize;
}

impl<E: InferenceEngine> InferenceEngineObj for E { /* delegate */ }

impl DynInferenceEngine {
    pub fn new<E: InferenceEngine + 'static>(engine: E) -> Self { /* ... */ }
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> { /* ... */ }
    pub fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError> { /* ... */ }
    pub fn reset(&mut self) { /* ... */ }
    pub fn seq_pos(&self) -> usize { /* ... */ }
    pub fn truncate_to(&mut self, pos: usize) { /* ... */ }
}
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

Implement `InferenceEngine` for `AneInference`:

```rust
impl<D: AneDevice> InferenceEngine for AneInference<D> {
    type Artifacts = AneDecodeArtifacts;

    fn load(&mut self, artifacts: &AneDecodeArtifacts) -> Result<(), InferenceError> {
        // Load from decode bundle path
        *self = AneInference::from_bundle(&artifacts.bundle_path, /* ... */)
            .map_err(|e| InferenceError::Runtime(Box::new(e)))?;
        Ok(())
    }

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

impl InferenceEngine for CudaInference {
    type Artifacts = CudaArtifacts<'_>;

    fn load(&mut self, artifacts: &CudaArtifacts<'_>) -> Result<(), InferenceError> { /* ... */ }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> { /* ... */ }
    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError> { /* ... */ }
    fn reset(&mut self) { /* ... */ }
    fn seq_pos(&self) -> usize { /* ... */ }
    fn truncate_to(&mut self, pos: usize) { /* ... */ }
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
  └─ ironmill-compile (compilation API)
  │    └─ mil-rs (IR, passes, weights)
  └─ ironmill-inference (runtime API)
       ├─ mil-rs (shared weight types)
       ├─ ironmill-metal-sys     [feature = "metal",  macOS only]
       ├─ ironmill-ane-sys       [feature = "ane",    macOS only]
       ├─ ironmill-coreml-sys    [feature = "coreml", macOS only]
       ├─ ironmill-mlx-sys       [feature = "mlx",    macOS only]
       └─ ironmill-cuda-sys      [feature = "cuda",   future]
```

Users should never need to depend on `mil-rs` directly for normal usage.
`ironmill-compile` re-exports the relevant IR types, and
`ironmill-inference` re-exports the relevant weight types.

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

**InferenceEngine::load** — no more `&dyn Any`:
```rust
// BEFORE:
engine.load(&artifacts as &dyn std::any::Any)?;

// AFTER:
engine.load(&artifacts)?;  // type-checked at compile time
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
    // Optional: downcast to specific backend error
    if let Some(metal_err) = source.downcast_ref::<MetalError>() {
        eprintln!("Metal details: {metal_err:?}");
    }
}
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
