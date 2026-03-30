//! High-level builder API for compiling ML models at build time.
//!
//! This module provides [`CompileBuilder`], a builder-pattern API that wraps the
//! full ONNX → CoreML conversion pipeline. It is designed for use in `build.rs`
//! scripts but works anywhere.
//!
//! # Example
//!
//! ```no_run
//! use mil_rs::build_api::{CompileBuilder, Quantization, TargetComputeUnit};
//!
//! let output = CompileBuilder::new("model.onnx")
//!     .quantize(Quantization::Fp16)
//!     .target(TargetComputeUnit::CpuAndNeuralEngine)
//!     .output("resources/model.mlpackage")
//!     .build()
//!     .expect("model compilation failed");
//!
//! println!("wrote {}", output.mlpackage.display());
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::coreml::compiler::{compile_model, is_compiler_available};
use mil_rs::convert::{
    ConversionConfig, model_to_program, onnx_to_program_with_config, program_to_model,
};
use mil_rs::error::{MilError, Result};
use mil_rs::ir::{PassPipeline, PipelineReport};
use mil_rs::reader::{read_mlmodel, read_mlpackage, read_onnx};
use mil_rs::writer::write_mlpackage;

/// Quantization mode for model conversion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Quantization {
    /// No quantization — keep original precision.
    #[default]
    None,
    /// Convert float32 weights and activations to float16.
    Fp16,
    /// Quantize weights to 8-bit integers.
    Int8,
}

/// Target compute unit configuration for the model at runtime.
///
/// This controls which hardware the model is eligible to run on.
/// Unlike [`crate::ir::ComputeUnit`] (which annotates individual operations),
/// this specifies the model-level deployment target.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TargetComputeUnit {
    /// Run on CPU only.
    CpuOnly,
    /// Run on CPU and GPU.
    CpuAndGpu,
    /// Run on CPU and Apple Neural Engine.
    CpuAndNeuralEngine,
    /// Run on all available compute units (default).
    #[default]
    All,
}

/// Builder for compiling ML models at build time.
///
/// Wraps the full conversion pipeline (read → convert → optimize → write) in a
/// convenient builder-pattern API suitable for `build.rs` scripts.
///
/// # Example
///
/// ```no_run
/// use mil_rs::build_api::{CompileBuilder, Quantization};
///
/// CompileBuilder::new("model.onnx")
///     .quantize(Quantization::Fp16)
///     .output("resources/model.mlpackage")
///     .build()
///     .expect("model compilation failed");
/// ```
pub struct CompileBuilder {
    input: PathBuf,
    output: Option<PathBuf>,
    quantization: Quantization,
    target: TargetComputeUnit,
    input_shapes: Vec<(String, Vec<usize>)>,
    palettize_bits: Option<u8>,
    no_fusion: bool,
    compile_mlmodelc: bool,
}

/// Output produced by [`CompileBuilder::build`].
#[derive(Debug)]
pub struct BuildOutput {
    /// Path to the generated `.mlpackage`.
    pub mlpackage: PathBuf,
    /// Path to the compiled `.mlmodelc`, if compilation was requested and succeeded.
    pub mlmodelc: Option<PathBuf>,
    /// Optimization pipeline report.
    pub report: PipelineReport,
}

impl CompileBuilder {
    /// Create a new builder with the given input model path.
    ///
    /// The input can be an `.onnx`, `.mlmodel`, `.mlpackage`, `.gguf` file,
    /// or a directory containing `config.json` (SafeTensors model).
    pub fn new(input: impl Into<PathBuf>) -> Self {
        Self {
            input: input.into(),
            output: None,
            quantization: Quantization::None,
            target: TargetComputeUnit::All,
            input_shapes: Vec::new(),
            palettize_bits: None,
            no_fusion: false,
            compile_mlmodelc: false,
        }
    }

    /// Set the output `.mlpackage` path.
    ///
    /// If not specified, the output path is derived from the input:
    /// - `model.onnx` → `model.mlpackage`
    /// - `model.mlmodel` → `model.mlpackage`
    /// - A directory → `<dirname>.mlpackage`
    pub fn output(mut self, path: impl Into<PathBuf>) -> Self {
        self.output = Some(path.into());
        self
    }

    /// Set the quantization mode.
    pub fn quantize(mut self, q: Quantization) -> Self {
        self.quantization = q;
        self
    }

    /// Set the target compute unit for the model.
    pub fn target(mut self, target: TargetComputeUnit) -> Self {
        self.target = target;
        self
    }

    /// Add a concrete shape constraint for a named input.
    ///
    /// This can be called multiple times to set shapes for different inputs.
    /// Shape materialization makes the model ANE-compatible by replacing
    /// dynamic dimensions with fixed values.
    pub fn input_shape(mut self, name: impl Into<String>, shape: Vec<usize>) -> Self {
        self.input_shapes.push((name.into(), shape));
        self
    }

    /// Set the weight palettization bit-width (2, 4, 6, or 8).
    pub fn palettize(mut self, bits: u8) -> Self {
        self.palettize_bits = Some(bits);
        self
    }

    /// Disable fusion and optimization passes.
    ///
    /// When set, only cleanup passes (DCE, identity elimination, constant folding)
    /// and any explicitly requested quantization/palettization passes will run.
    pub fn no_fusion(mut self) -> Self {
        self.no_fusion = true;
        self
    }

    /// Also compile the `.mlpackage` to `.mlmodelc` via `xcrun coremlcompiler`.
    ///
    /// This is a no-op on non-macOS platforms or when Xcode is not installed.
    pub fn compile(mut self) -> Self {
        self.compile_mlmodelc = true;
        self
    }

    /// Execute the full build pipeline.
    ///
    /// 1. Reads the input model
    /// 2. Converts to MIL IR
    /// 3. Runs optimization passes (quantization, fusion, etc.)
    /// 4. Writes the `.mlpackage`
    /// 5. Optionally compiles to `.mlmodelc`
    /// 6. Prints `cargo::rerun-if-changed` for `build.rs` integration
    pub fn build(self) -> Result<BuildOutput> {
        let input = &self.input;

        if !input.exists() {
            return Err(MilError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("input path does not exist: {}", input.display()),
            )));
        }

        // 1. Detect format and read/convert to Program
        let mut program = match detect_format(input) {
            InputFormat::Onnx => {
                let onnx_model = read_onnx(input)?;
                let config = ConversionConfig {
                    merge_lora: true,
                    model_dir: input.parent().map(|p| p.to_path_buf()),
                };
                let result = onnx_to_program_with_config(&onnx_model, &config)?;
                result.program
            }
            InputFormat::MlModel => {
                let model = read_mlmodel(input)?;
                model_to_program(&model)?
            }
            InputFormat::MlPackage => {
                let model = read_mlpackage(input)?;
                model_to_program(&model)?
            }
            InputFormat::SafeTensors => {
                let provider = crate::weights::safetensors::SafeTensorsProvider::load(input)?;
                let result = crate::templates::weights_to_program(&provider)?;
                result.program
            }
            InputFormat::Gguf => {
                let provider = crate::weights::gguf::GgufProvider::load(input)?;
                let result = crate::templates::weights_to_program(&provider)?;
                result.program
            }
            InputFormat::Unknown(ext) => {
                return Err(MilError::Validation(format!(
                    "unsupported input format '{ext}'. \
                     Expected .onnx, .mlmodel, .mlpackage, .gguf, .safetensors, \
                     or a directory containing config.json."
                )));
            }
        };

        // 2. Build the pass pipeline
        let mut pipeline = PassPipeline::new();

        if self.no_fusion {
            pipeline = pipeline.without_fusion();
        }

        // Add input shape materialization
        if !self.input_shapes.is_empty() {
            let shapes: HashMap<String, Vec<usize>> = self.input_shapes.into_iter().collect();
            pipeline = pipeline.with_shapes(shapes);
        }

        // Add quantization
        match self.quantization {
            Quantization::None => {}
            Quantization::Fp16 => {
                pipeline = pipeline.with_fp16()?;
            }
            Quantization::Int8 => {
                pipeline = pipeline.with_int8(None)?;
            }
        }

        // Add palettization
        if let Some(bits) = self.palettize_bits {
            pipeline = pipeline.with_palettize(bits)?;
        }

        // 3. Run the pipeline
        let report = pipeline.run(&mut program)?;

        // 4. Convert to Model proto (spec version 7 as default)
        let model = program_to_model(&program, 7)?;

        // 5. Determine output path
        let output_path = self.output.unwrap_or_else(|| default_output_path(input));

        // 6. Write .mlpackage
        write_mlpackage(&model, &output_path)?;

        // 7. Optionally compile to .mlmodelc
        let mlmodelc = if self.compile_mlmodelc && is_compiler_available() {
            let output_dir = output_path.parent().unwrap_or(Path::new("."));
            compile_model(&output_path, output_dir).ok()
        } else {
            None
        };

        // 8. Print cargo::rerun-if-changed for build.rs integration
        println!("cargo::rerun-if-changed={}", input.display());

        Ok(BuildOutput {
            mlpackage: output_path,
            mlmodelc,
            report,
        })
    }
}

// ── Input format detection ──────────────────────────────────────────────

enum InputFormat {
    Onnx,
    MlModel,
    MlPackage,
    SafeTensors,
    Gguf,
    Unknown(String),
}

fn detect_format(path: &Path) -> InputFormat {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext {
            "onnx" => return InputFormat::Onnx,
            "mlmodel" => return InputFormat::MlModel,
            "mlpackage" => return InputFormat::MlPackage,
            "gguf" => return InputFormat::Gguf,
            "safetensors" => return InputFormat::SafeTensors,
            _ => {}
        }
    }
    // A directory with config.json is a HuggingFace model directory.
    if path.is_dir() && path.join("config.json").exists() {
        return InputFormat::SafeTensors;
    }
    let ext_display = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("<none>")
        .to_string();
    InputFormat::Unknown(ext_display)
}

/// Derive a default `.mlpackage` output path from the input path.
fn default_output_path(input: &Path) -> PathBuf {
    if input.is_dir() {
        // Directory → <dirname>.mlpackage
        let dir_name = input
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model");
        input.with_file_name(format!("{dir_name}.mlpackage"))
    } else {
        // File → replace extension with .mlpackage
        input.with_extension("mlpackage")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_correct_defaults() {
        let builder = CompileBuilder::new("model.onnx");
        assert_eq!(builder.input, PathBuf::from("model.onnx"));
        assert_eq!(builder.quantization, Quantization::None);
        assert_eq!(builder.target, TargetComputeUnit::All);
        assert!(builder.input_shapes.is_empty());
        assert_eq!(builder.palettize_bits, None);
        assert!(!builder.no_fusion);
        assert!(!builder.compile_mlmodelc);
        assert!(builder.output.is_none());
    }

    #[test]
    fn builder_methods_chain() {
        let builder = CompileBuilder::new("model.onnx")
            .quantize(Quantization::Fp16)
            .target(TargetComputeUnit::CpuAndNeuralEngine)
            .output("out.mlpackage")
            .input_shape("input", vec![1, 3, 224, 224])
            .palettize(4)
            .no_fusion()
            .compile();

        assert_eq!(builder.input, PathBuf::from("model.onnx"));
        assert_eq!(builder.quantization, Quantization::Fp16);
        assert_eq!(builder.target, TargetComputeUnit::CpuAndNeuralEngine);
        assert_eq!(builder.output, Some(PathBuf::from("out.mlpackage")));
        assert_eq!(
            builder.input_shapes,
            vec![("input".to_string(), vec![1, 3, 224, 224])]
        );
        assert_eq!(builder.palettize_bits, Some(4));
        assert!(builder.no_fusion);
        assert!(builder.compile_mlmodelc);
    }

    #[test]
    fn quantization_default_is_none() {
        assert_eq!(Quantization::default(), Quantization::None);
    }

    #[test]
    fn target_default_is_all() {
        assert_eq!(TargetComputeUnit::default(), TargetComputeUnit::All);
    }

    #[test]
    fn default_output_path_from_onnx() {
        let path = PathBuf::from("model.onnx");
        assert_eq!(default_output_path(&path), PathBuf::from("model.mlpackage"));
    }

    #[test]
    fn default_output_path_from_mlmodel() {
        let path = PathBuf::from("model.mlmodel");
        assert_eq!(default_output_path(&path), PathBuf::from("model.mlpackage"));
    }

    #[test]
    fn default_output_path_with_parent_dir() {
        let path = PathBuf::from("models/my_model.onnx");
        assert_eq!(
            default_output_path(&path),
            PathBuf::from("models/my_model.mlpackage")
        );
    }

    #[test]
    fn build_nonexistent_input_returns_error() {
        let result = CompileBuilder::new("does_not_exist.onnx").build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("does not exist"),
            "error should mention missing file, got: {err}"
        );
    }

    #[test]
    fn multiple_input_shapes() {
        let builder = CompileBuilder::new("model.onnx")
            .input_shape("images", vec![1, 3, 224, 224])
            .input_shape("labels", vec![1, 1000]);

        assert_eq!(builder.input_shapes.len(), 2);
        assert_eq!(builder.input_shapes[0].0, "images");
        assert_eq!(builder.input_shapes[1].0, "labels");
    }

    #[test]
    fn detect_format_onnx() {
        assert!(matches!(
            detect_format(Path::new("m.onnx")),
            InputFormat::Onnx
        ));
    }

    #[test]
    fn detect_format_mlmodel() {
        assert!(matches!(
            detect_format(Path::new("m.mlmodel")),
            InputFormat::MlModel
        ));
    }

    #[test]
    fn detect_format_mlpackage() {
        assert!(matches!(
            detect_format(Path::new("m.mlpackage")),
            InputFormat::MlPackage
        ));
    }

    #[test]
    fn detect_format_gguf() {
        assert!(matches!(
            detect_format(Path::new("m.gguf")),
            InputFormat::Gguf
        ));
    }

    #[test]
    fn detect_format_unknown() {
        assert!(matches!(
            detect_format(Path::new("m.txt")),
            InputFormat::Unknown(_)
        ));
    }
}
