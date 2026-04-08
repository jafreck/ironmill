//! CoreML model export and inference for Burn.
//!
//! This crate provides two capabilities for Burn users:
//! - **Export**: Convert ONNX models (exported from Burn) to CoreML
//! - **Inference** (macOS only): Load and run CoreML models
//!
//! # Workflow
//!
//! 1. Export your Burn model to ONNX using Burn's built-in ONNX recorder
//! 2. Use [`export::export_to_coreml`] to convert ONNX → CoreML
//! 3. Use [`inference::CoreMlInference`] to run the model on macOS
//!
//! # Why no `burn-core` dependency?
//!
//! This crate intentionally does **not** depend on `burn-core` directly.
//! It provides standalone export/inference utilities that Burn users call
//! alongside their Burn workflow. This avoids heavy dependency coupling
//! and version lock-in — `burn-coreml` works with any version of Burn
//! that can export to ONNX.
//!
//! # Example
//!
//! ```no_run
//! use burn_coreml::export::{export_to_coreml, ExportOptions};
//!
//! let result = export_to_coreml("model.onnx", "model.mlpackage", ExportOptions::default())?;
//! println!("exported to {}", result.mlpackage.display());
//! # Ok::<(), anyhow::Error>(())
//! ```

#![forbid(unsafe_code)]

pub mod export;

#[cfg(target_os = "macos")]
pub mod inference;
