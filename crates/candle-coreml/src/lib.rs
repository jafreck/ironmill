//! Bridge between candle and Apple CoreML.
//!
//! This crate provides two capabilities:
//!
//! - **Conversion** (all platforms): Convert ONNX models to CoreML format
//!   using [`convert::convert_onnx`].
//! - **Runtime** (macOS only): Load and run CoreML models via
//!   [`runtime::CoreMlModel`], returning raw f32 data and shapes that can be
//!   wrapped in candle `Tensor` values by the caller.
//!
//! # Example
//!
//! ```no_run
//! use candle_coreml::convert::{convert_onnx, ConvertOptions};
//!
//! let result = convert_onnx("model.onnx", "model.mlpackage", ConvertOptions::default())?;
//! println!("converted to {}", result.mlpackage.display());
//! # Ok::<(), anyhow::Error>(())
//! ```

#![forbid(unsafe_code)]

pub mod convert;

#[cfg(target_os = "macos")]
pub mod runtime;
