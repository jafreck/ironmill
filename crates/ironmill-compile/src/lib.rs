//! Compilation pipeline for ironmill.
//!
//! This crate brings together ANE lowering passes, CoreML build-time compilation,
//! architecture templates, and weight providers. It depends on [`mil_rs`] for the
//! core MIL IR but owns all backend-specific compilation logic.

#![deny(unsafe_code)]

/// Compilation error types.
pub mod error;

/// ANE (Apple Neural Engine) compilation: passes, splitting, packing, validation, and caching.
pub mod ane;
/// Model conversion utilities: LoRA merging, MoE splitting, and pipeline orchestration.
pub mod convert;
/// CoreML compilation via `xcrun coremlcompiler` and the high-level build API.
pub mod coreml;
/// Architecture templates that construct MIL IR programs from weight tensors.
pub mod templates;
/// Weight providers for SafeTensors and GGUF formats.
pub mod weights;

/// C-compatible FFI API (enable with `--features c-api`).
#[cfg(feature = "c-api")]
#[allow(unsafe_code)]
pub mod c_api;
