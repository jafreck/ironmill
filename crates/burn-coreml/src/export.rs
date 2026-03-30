//! ONNX → CoreML export utilities.
//!
//! Burn models are exported to ONNX first (using Burn's built-in recorder),
//! then this module converts the ONNX file to a CoreML `.mlpackage`.

use std::path::{Path, PathBuf};

use ironmill_compile::coreml::build_api::{CompileBuilder, Quantization, TargetComputeUnit};

/// Options for exporting to CoreML.
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    /// Quantization mode.
    pub quantization: Quantization,
    /// Target compute units.
    pub target: Option<TargetComputeUnit>,
    /// Fixed input shapes for ANE compatibility.
    ///
    /// Each entry is `(input_name, shape_dimensions)`.
    pub input_shapes: Vec<(String, Vec<usize>)>,
    /// Also compile to `.mlmodelc` via `xcrun` (macOS only).
    pub compile: bool,
    /// Palettization bit-width (2, 4, 6, or 8).
    pub palettize_bits: Option<u8>,
    /// Disable optimization/fusion passes.
    pub no_fusion: bool,
}

/// Result of a successful export.
#[derive(Debug)]
pub struct ExportResult {
    /// Path to the generated `.mlpackage`.
    pub mlpackage_path: PathBuf,
    /// Path to the compiled `.mlmodelc`, if compilation was requested and succeeded.
    pub mlmodelc_path: Option<PathBuf>,
}

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
/// println!("exported to {}", result.mlpackage_path.display());
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn export_to_coreml(
    onnx_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    options: ExportOptions,
) -> anyhow::Result<ExportResult> {
    let mut builder = CompileBuilder::new(onnx_path.as_ref()).output(output_path.as_ref());

    builder = builder.quantize(options.quantization);

    if let Some(target) = options.target {
        builder = builder.target(target);
    }

    for (name, shape) in options.input_shapes {
        builder = builder.input_shape(name, shape);
    }

    if let Some(bits) = options.palettize_bits {
        builder = builder.palettize(bits);
    }

    if options.no_fusion {
        builder = builder.no_fusion();
    }

    if options.compile {
        builder = builder.compile();
    }

    let output = builder.build()?;

    Ok(ExportResult {
        mlpackage_path: output.mlpackage,
        mlmodelc_path: output.mlmodelc,
    })
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
