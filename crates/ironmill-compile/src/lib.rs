//! Compilation pipeline for ironmill.
//!
//! This crate brings together ANE lowering passes, CoreML build-time compilation,
//! architecture templates, and weight providers. It depends on [`mil_rs`] for the
//! core MIL IR but owns all backend-specific compilation logic.
//!
//! # Example: compile ONNX → CoreML mlpackage
//!
//! ```no_run
//! use ironmill_compile::CompileBuilder;
//! use ironmill_compile::coreml::build_api::Quantization;
//!
//! let output = CompileBuilder::new("model.onnx")
//!     .quantize(Quantization::Fp16)
//!     .output("resources/model.mlpackage")
//!     .compile()
//!     .build()?;
//!
//! println!("wrote {}", output.mlpackage.display());
//! # Ok::<(), ironmill_compile::CompileError>(())
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]

/// Backend-agnostic compilation trait and supporting types.
pub mod compile_target;
/// Compilation error types.
pub mod error;

/// ANE (Apple Neural Engine) compilation: passes, splitting, packing, validation, and caching.
pub mod ane;
/// Model conversion utilities: LoRA merging, MoE splitting, and pipeline orchestration.
pub mod convert;
/// CoreML compilation via `xcrun coremlcompiler` and the high-level build API.
pub mod coreml;
/// GPU-targeted compilation: PolarQuant passes → MilWeightProvider.
pub mod gpu;
/// Architecture templates that construct MIL IR programs from weight tensors.
pub mod templates;
/// Weight providers for SafeTensors and GGUF formats.
pub mod weights;

// ── Convenience re-exports ────────────────────────────────────────────

pub use coreml::build_api::CompileBuilder;
pub use error::{CompileError, Result};
pub use gpu::GpuCompileBuilder;
pub use templates::weights_to_program;
pub use weights::{GgufProvider, ModelConfig, SafeTensorsProvider, WeightProvider};
