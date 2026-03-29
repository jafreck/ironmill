//! ONNX → CoreML conversion helpers.
//!
//! Wraps [`mil_rs::CompileBuilder`] in a candle-friendly API. Works on all
//! platforms — no macOS or Xcode required for conversion itself.
//!
//! # Example
//!
//! ```no_run
//! use candle_coreml::convert::{convert_onnx, ConvertOptions};
//! use mil_rs::Quantization;
//!
//! let opts = ConvertOptions {
//!     quantization: Quantization::Fp16,
//!     ..Default::default()
//! };
//! let path = convert_onnx("model.onnx", "model.mlpackage", opts)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::path::{Path, PathBuf};

use mil_rs::build_api::TargetComputeUnit;
use mil_rs::{CompileBuilder, Quantization};

/// Options for ONNX → CoreML conversion.
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    /// Quantization mode (default: None).
    pub quantization: Quantization,
    /// Target compute units (default: builder default, All).
    pub target: Option<TargetComputeUnit>,
    /// Fixed input shapes for ANE compatibility.
    pub input_shapes: Vec<(String, Vec<usize>)>,
    /// Also compile to `.mlmodelc` via `xcrun` (macOS only).
    pub compile: bool,
    /// Palettization bit-width (2, 4, 6, or 8).
    pub palettize_bits: Option<u8>,
}

/// Convert an ONNX model to CoreML `.mlpackage` format.
///
/// Returns the path to the created `.mlpackage` file.
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
/// let path = convert_onnx("model.onnx", "out.mlpackage", ConvertOptions::default())?;
/// println!("wrote {}", path.display());
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn convert_onnx(
    onnx_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    options: ConvertOptions,
) -> anyhow::Result<PathBuf> {
    let mut builder = CompileBuilder::new(onnx_path.as_ref())
        .output(output_path.as_ref())
        .quantize(options.quantization);

    if let Some(target) = options.target {
        builder = builder.target(target);
    }

    for (name, shape) in options.input_shapes {
        builder = builder.input_shape(name, shape);
    }

    if let Some(bits) = options.palettize_bits {
        builder = builder.palettize(bits);
    }

    if options.compile {
        builder = builder.compile();
    }

    let output = builder
        .build()
        .map_err(|e| anyhow::anyhow!("ONNX to CoreML conversion failed: {e}"))?;

    Ok(output.mlpackage)
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
        };
        assert_eq!(opts.quantization, Quantization::Fp16);
        assert_eq!(opts.target, Some(TargetComputeUnit::CpuAndNeuralEngine));
        assert_eq!(opts.input_shapes.len(), 1);
        assert!(opts.compile);
        assert_eq!(opts.palettize_bits, Some(4));
    }
}
