//! ONNX → CoreML conversion helpers.
//!
//! Wraps [`ironmill_compile::coreml::build_api::convert`] in a candle-friendly API. Works on all
//! platforms — no macOS or Xcode required for conversion itself.
//!
//! The heavy lifting is done by [`ironmill_compile::coreml::build_api::convert`];
//! this module re-exports the shared types with candle-flavoured aliases.
//! See also: `burn-coreml::export` for the Burn equivalent.
//!
//! # Example
//!
//! ```no_run
//! use candle_coreml::convert::{convert_onnx, ConvertOptions};
//! use ironmill_compile::coreml::build_api::Quantization;
//!
//! let opts = ConvertOptions {
//!     quantization: Quantization::Fp16,
//!     ..Default::default()
//! };
//! let result = convert_onnx("model.onnx", "model.mlpackage", opts)?;
//! # Ok::<(), ironmill_compile::error::CompileError>(())
//! ```

use std::path::Path;

use ironmill_compile::coreml::build_api::convert_to_coreml;
pub use ironmill_compile::coreml::build_api::{Quantization, TargetComputeUnit};
pub use ironmill_compile::error::CompileError;

/// Options for ONNX → CoreML conversion.
///
/// Re-export of [`ironmill_compile::coreml::build_api::ConvertConfig`].
pub type ConvertOptions = ironmill_compile::coreml::build_api::ConvertConfig;

/// Result of a successful ONNX → CoreML conversion.
///
/// Re-export of [`ironmill_compile::coreml::build_api::ConvertOutput`].
pub type ConvertResult = ironmill_compile::coreml::build_api::ConvertOutput;

/// Convert an ONNX model to CoreML `.mlpackage` format.
///
/// Returns a [`ConvertResult`] containing the `.mlpackage` path and, if
/// compilation was requested via [`ConvertOptions::compile`], the `.mlmodelc` path.
///
/// # Errors
///
/// Returns an error if the input file does not exist, is not valid ONNX,
/// or if conversion fails.
///
/// # Example
///
/// ```no_run
/// use candle_coreml::convert::{convert_onnx, ConvertOptions};
///
/// let result = convert_onnx("model.onnx", "out.mlpackage", ConvertOptions::default())?;
/// println!("wrote {}", result.mlpackage.display());
/// # Ok::<(), CompileError>(())
/// ```
pub fn convert_onnx(
    onnx_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    options: ConvertOptions,
) -> Result<ConvertResult, CompileError> {
    convert_to_coreml(onnx_path.as_ref(), output_path.as_ref(), options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_options_default_values() {
        let opts = ConvertOptions::default();
        assert_eq!(opts.quantization, Quantization::None);
        assert!(opts.target.is_none());
        assert!(opts.input_shapes.is_empty());
        assert!(!opts.compile);
        assert!(opts.palettize_bits.is_none());
    }

    #[test]
    fn convert_nonexistent_onnx_returns_error() {
        let result = convert_onnx(
            "nonexistent_model.onnx",
            "out.mlpackage",
            ConvertOptions::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn convert_options_builder_pattern() {
        let opts = ConvertOptions {
            quantization: Quantization::Fp16,
            target: Some(TargetComputeUnit::CpuAndNeuralEngine),
            input_shapes: vec![("input".to_string(), vec![1, 3, 224, 224])],
            compile: true,
            palettize_bits: Some(4),
            ..Default::default()
        };
        assert_eq!(opts.quantization, Quantization::Fp16);
        assert_eq!(opts.target, Some(TargetComputeUnit::CpuAndNeuralEngine));
        assert_eq!(opts.input_shapes.len(), 1);
        assert!(opts.compile);
        assert_eq!(opts.palettize_bits, Some(4));
    }
}
