//! Compilation pipeline for ironmill.
//!
//! This crate brings together ANE lowering passes, CoreML build-time compilation,
//! architecture templates, and weight providers. It depends on [`mil_rs`] for the
//! core MIL IR but owns all backend-specific compilation logic.

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
