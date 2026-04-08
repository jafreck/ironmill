//! ONNX → CoreML export utilities.
//!
//! Burn models are exported to ONNX first (using Burn's built-in recorder),
//! then this module converts the ONNX file to a CoreML `.mlpackage`.
//!
//! The heavy lifting is done by [`ironmill_compile::coreml::build_api::convert`];
//! this module re-exports the shared types with Burn-flavoured aliases.
//! See also: `candle-coreml::convert` for the candle equivalent.

use std::path::Path;

use ironmill_compile::coreml::build_api::convert_to_coreml;
pub use ironmill_compile::coreml::build_api::{Quantization, TargetComputeUnit};

/// Options for exporting to CoreML.
///
/// Re-export of [`ironmill_compile::coreml::build_api::ConvertConfig`].
pub type ExportOptions = ironmill_compile::coreml::build_api::ConvertConfig;

/// Result of a successful export.
///
/// Re-export of [`ironmill_compile::coreml::build_api::ConvertOutput`].
pub type ExportResult = ironmill_compile::coreml::build_api::ConvertOutput;

/// Export a Burn model (via its ONNX representation) to CoreML.
///
/// Burn models can be exported to ONNX format first, then this function
/// converts the ONNX file to a CoreML `.mlpackage`.
///
/// # Arguments
///
/// * `onnx_path` — Path to the ONNX model file
/// * `output_path` — Desired output `.mlpackage` path
/// * `options` — Export configuration (quantization, target, etc.)
///
/// # Example
///
/// ```no_run
/// use burn_coreml::export::{export_to_coreml, ExportOptions};
///
/// let result = export_to_coreml(
///     "my_model.onnx",
///     "my_model.mlpackage",
///     ExportOptions::default(),
/// )?;
/// println!("exported to {}", result.mlpackage.display());
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn export_to_coreml(
    onnx_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    options: ExportOptions,
) -> anyhow::Result<ExportResult> {
    Ok(convert_to_coreml(
        onnx_path.as_ref(),
        output_path.as_ref(),
        options,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_options_default_values() {
        let opts = ExportOptions::default();
        assert_eq!(opts.quantization, Quantization::None);
        assert!(opts.target.is_none());
        assert!(opts.input_shapes.is_empty());
        assert!(!opts.compile);
        assert!(opts.palettize_bits.is_none());
        assert!(!opts.no_fusion);
    }

    #[test]
    fn export_nonexistent_onnx_returns_error() {
        let result = export_to_coreml(
            "does_not_exist.onnx",
            "output.mlpackage",
            ExportOptions::default(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("does not exist"),
            "error should mention missing file, got: {err}"
        );
    }

    #[test]
    fn export_options_with_all_fields() {
        let opts = ExportOptions {
            quantization: Quantization::Fp16,
            target: Some(TargetComputeUnit::CpuAndNeuralEngine),
            input_shapes: vec![("input".to_string(), vec![1, 3, 224, 224])],
            compile: true,
            palettize_bits: Some(4),
            no_fusion: true,
        };
        assert_eq!(opts.quantization, Quantization::Fp16);
        assert_eq!(opts.target, Some(TargetComputeUnit::CpuAndNeuralEngine));
        assert_eq!(opts.input_shapes.len(), 1);
        assert!(opts.compile);
        assert_eq!(opts.palettize_bits, Some(4));
        assert!(opts.no_fusion);
    }
}
