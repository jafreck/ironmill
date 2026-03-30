//! # mil-rs
//!
//! Read, write, and manipulate Apple CoreML models in Rust — no Python required.
//!
//! `mil-rs` provides a strongly-typed MIL (Model Intermediate Language)
//! intermediate representation alongside protobuf serialization for
//! Apple's CoreML on-disk formats (`.mlmodel` and `.mlpackage`).
//!
//! ## Quick start
//!
//! ```no_run
//! use mil_rs::{read_mlmodel, model_to_program, program_to_model, write_mlpackage};
//!
//! // Read a .mlmodel and convert to the typed IR
//! let model = read_mlmodel("input.mlmodel").unwrap();
//! let program = model_to_program(&model).unwrap();
//!
//! // Inspect the program
//! for (name, _func) in &program.functions {
//!     println!("function: {name}");
//! }
//!
//! // Convert back and write as .mlpackage
//! let out = program_to_model(&program, model.specification_version as i32).unwrap();
//! write_mlpackage(&out, "output.mlpackage").unwrap();
//! ```
//!
//! ## Modules
//!
//! | Module        | Description |
//! |---------------|-------------|
//! | [`build_api`] | High-level [`CompileBuilder`] for converting models in `build.rs` scripts |
//! | `c_api`       | C-compatible FFI (enable with `--features c-api`) |
//! | [`compiler`]  | Compile models to `.mlmodelc` via `xcrun coremlcompiler` |
//! | [`convert`]   | Bidirectional conversion between protobuf [`Model`] and IR [`Program`] |
//! | [`error`]     | [`MilError`] enum and [`Result`](error::Result) type alias |
//! | [`ir`]        | MIL intermediate representation — [`Program`], [`Function`], [`Block`], [`Operation`], [`Graph`], tensor types |
//! | [`proto`]     | Auto-generated protobuf types from Apple's CoreML `.proto` specification |
//! | [`reader`]    | Read `.mlmodel` and `.mlpackage` files into a protobuf [`Model`] |
//! | [`writer`]    | Write a protobuf [`Model`] to `.mlmodel` or `.mlpackage` |
//!
//! ## Supported model types
//!
//! The proto ↔ IR conversion supports **ML Program** models (CoreML spec v7+).
//! Legacy `NeuralNetwork` models can be read and written at the protobuf level
//! but cannot be converted to the MIL IR.
//!
//! ## C API
//!
//! A C-compatible FFI is available behind the `c-api` feature flag. Build with
//! `cargo build --features c-api` and generate the header with `cbindgen`.
//! See the `c_api` module for details and the [C API guide](../docs/C_API.md).
//!
//! ## Build-time compilation
//!
//! Use [`CompileBuilder`] in your `build.rs` to convert ONNX models to CoreML
//! at build time. See [`build_api`] for the full API and
//! [examples/build_rs_example.rs](../examples/build_rs_example.rs) for usage
//! patterns.

#![deny(unsafe_code)]

/// Analysis passes for MIL IR programs (FLOPs counting, etc.).
pub mod analysis;
/// High-level builder API for compiling ML models at build time.
///
/// Provides [`CompileBuilder`] for use in `build.rs` scripts.
pub mod build_api;
/// C-compatible FFI API for use from C, Swift, C++, Go, etc.
#[cfg(feature = "c-api")]
#[allow(unsafe_code)]
pub mod c_api;
/// Compile CoreML models to `.mlmodelc` via `xcrun coremlcompiler`.
pub mod compiler;
/// Bidirectional conversion between protobuf types, ONNX models, and the MIL IR.
pub mod convert;
/// Error types used throughout the crate.
pub mod error;
/// Direct ANE compilation via Objective-C FFI (experimental, private API).
#[cfg(feature = "ane-direct")]
pub mod ffi;
/// MIL Intermediate Representation — programs, functions, operations, and types.
pub mod ir;
/// Auto-generated protobuf types for CoreML and ONNX specifications.
pub mod proto;
/// Readers for `.mlmodel`, `.mlpackage`, and `.onnx` files.
pub mod reader;
/// ANE (Apple Neural Engine) compatibility validation.
pub mod validate;
/// Writers for `.mlmodel` and `.mlpackage` files.
pub mod writer;

// Re-export key types at crate root for convenience.

/// Compile a `.mlpackage` or `.mlmodel` to `.mlmodelc` using `xcrun coremlcompiler`.
pub use compiler::compile_model;
/// Check whether `xcrun coremlcompiler` is available on this system.
pub use compiler::is_compiler_available;

/// Configuration options for ONNX → MIL conversion.
pub use convert::ConversionConfig;
/// Convert a protobuf [`Model`] into a MIL IR [`Program`].
pub use convert::model_to_program;
/// Convert an ONNX [`proto::onnx::ModelProto`] into a MIL IR [`Program`].
pub use convert::onnx_to_program;
/// Convert an ONNX model with explicit configuration (e.g. LoRA merge control).
pub use convert::onnx_to_program_with_config;
/// Parse and convert a multi-ONNX pipeline from a TOML manifest.
pub use convert::pipeline::{PipelineManifest, convert_pipeline, parse_pipeline_manifest};
/// Convert a MIL IR [`Program`] back into a protobuf [`Model`].
pub use convert::program_to_model;
/// Convert a MoE split result into a single multi-function [`Model`].
pub use convert::program_to_multi_function_model;
/// Convert a MIL IR [`Program`] into an updatable protobuf [`Model`].
pub use convert::program_to_updatable_model;
/// Configuration for producing an updatable CoreML model.
pub use convert::{LossFunction, UpdatableModelConfig, UpdateOptimizer};

/// Options controlling ANE-aware template code generation.
pub use convert::TemplateOptions;
/// Build a MIL IR [`Program`] from weight tensors using architecture templates.
pub use convert::weights_to_program;
/// Build a MIL IR [`Program`] from weight tensors with explicit template options.
pub use convert::weights_to_program_with_options;
/// Weight provider abstraction for SafeTensors/GGUF formats.
pub use convert::{Architecture, ModelConfig, WeightProvider, WeightTensor};

/// Detect Mixture-of-Experts architecture in a MIL [`Program`].
pub use convert::moe::detect_moe;
/// Fuse top-K most frequently activated experts into a single dense program.
pub use convert::moe::fuse_top_k_experts;
/// Split a MoE program into shared layers and per-expert programs.
pub use convert::moe::split_moe;
pub use convert::moe::{
    ExpertFrequencyProfile, MoeFuseResult, MoeManifest, MoeSplitResult, MoeTopology,
};

/// Error type for all operations in this crate.
pub use error::MilError;

pub use ir::{
    Block, ComputeUnit, Function, Graph, MixedPrecisionConfig, MixedPrecisionPass, OpPrecision,
    OpSplittingPass, Operation, Pass, PassPipeline, PassResult, PipelineReport, Program,
    ScalarType, SplitResult, TensorType, Value,
};

/// Estimate total FLOPs for a MIL program's forward pass.
pub use analysis::flops::estimate_program_flops;

/// Result of ANE compatibility analysis.
pub use validate::ValidationReport;
/// Validate a MIL [`Program`] for Apple Neural Engine compatibility.
pub use validate::validate_ane_compatibility;
/// Serialize a [`ValidationReport`] to JSON.
pub use validate::validation_report_to_json;

#[cfg(not(doctest))]
pub use proto::specification::Model;

/// Read a GGUF model file into a [`GgufProvider`](convert::weights::gguf::GgufProvider).
pub use reader::read_gguf;
/// Read a `.mlmodel` file into a protobuf [`Model`].
pub use reader::read_mlmodel;
/// Read a `.mlpackage` directory into a protobuf [`Model`].
pub use reader::read_mlpackage;
/// Read an ONNX `.onnx` file into an ONNX [`proto::onnx::ModelProto`].
pub use reader::read_onnx;
/// Read an ONNX `.onnx` file, returning the model and its parent directory.
pub use reader::read_onnx_with_dir;
/// Read a SafeTensors model directory into a [`SafeTensorsProvider`].
pub use reader::read_safetensors;

/// SafeTensors weight provider for HuggingFace model directories.
pub use convert::SafeTensorsProvider;

/// Write a protobuf [`Model`] to a `.mlmodel` file.
pub use writer::write_mlmodel;
/// Write a protobuf [`Model`] to a `.mlpackage` directory.
pub use writer::write_mlpackage;

/// Builder for compiling ML models at build time.
pub use build_api::{BuildOutput, CompileBuilder, Quantization, TargetComputeUnit};
