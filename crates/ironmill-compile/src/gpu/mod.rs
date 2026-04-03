//! GPU-targeted compilation: PolarQuant passes → [`MilWeightProvider`].
//!
//! This module provides [`GpuCompileBuilder`], a builder that imports a model,
//! runs PolarQuant quantization passes, and produces a [`MilWeightProvider`]
//! suitable for loading via `GpuWeights::load()`.
//!
//! Phase 1 — no bundle serialization; the provider is returned in-memory.

pub mod bundle;

use std::path::{Path, PathBuf};

use mil_rs::convert::{ConversionConfig, onnx_to_program_with_config};
use mil_rs::ir::passes::PolarQuantPass;
use mil_rs::ir::passes::TypeRepropagationPass;
use mil_rs::ir::{PassPipeline, Program};
use mil_rs::reader::read_onnx;

use crate::error::CompileError;
use crate::weights::{MilWeightProvider, ModelConfig, WeightProvider};

/// Builder for GPU-targeted model compilation.
///
/// Imports a model from disk, runs PolarQuant quantization passes on it, and
/// returns a [`MilWeightProvider`] that can feed quantized weights to the GPU
/// inference runtime.
///
/// # Example
///
/// ```no_run
/// use ironmill_compile::gpu::GpuCompileBuilder;
///
/// let provider = GpuCompileBuilder::new("model.onnx")
///     .polar_quantize(4)
///     .min_elements(2048)
///     .build()
///     .expect("GPU compile failed");
/// ```
pub struct GpuCompileBuilder {
    input: PathBuf,
    n_bits: u8,
    min_elements: usize,
    pipeline: Option<PassPipeline>,
}

impl GpuCompileBuilder {
    pub fn new(input: impl Into<PathBuf>) -> Self {
        Self {
            input: input.into(),
            n_bits: 4,
            min_elements: 1024,
            pipeline: None,
        }
    }

    /// Set a custom pass pipeline for quantization/optimization.
    ///
    /// When set, this pipeline replaces the default PolarQuant passes.
    /// The caller controls what quantization is applied (INT4, INT8, AWQ, etc.).
    pub fn with_pass_pipeline(mut self, pipeline: PassPipeline) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    /// Set the PolarQuant bit-width (must be 2 or 4).
    pub fn polar_quantize(mut self, n_bits: u8) -> Self {
        self.n_bits = n_bits;
        self
    }

    /// Set the minimum number of tensor elements required for quantization.
    ///
    /// Tensors with fewer elements than this threshold are left unquantized.
    pub fn min_elements(mut self, min: usize) -> Self {
        self.min_elements = min;
        self
    }

    /// Run import + PolarQuant passes and return a [`MilWeightProvider`].
    ///
    /// 1. Detect input format (ONNX / SafeTensors / GGUF)
    /// 2. Import to MIL IR [`Program`]
    /// 3. Run `PassPipeline` with PolarQuant passes
    /// 4. Extract a [`MilWeightProvider`] from the quantized program
    pub fn build(mut self) -> Result<MilWeightProvider, CompileError> {
        let input = &self.input;

        if !input.exists() {
            return Err(CompileError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("input path does not exist: {}", input.display()),
            )));
        }

        // 1. Detect format and import to Program + ModelConfig
        let format = detect_format(input);

        // If a custom pipeline is set, route everything through MIL IR.
        if let Some(pipeline) = self.pipeline.take() {
            return self.build_with_pipeline(input, format, pipeline);
        }

        // Legacy path: default PolarQuant behavior
        match format {
            InputFormat::Onnx => {
                // ONNX: import → run passes → extract via MilWeightProvider.
                let (mut program, config) = import_onnx(input)?;

                let mut pipeline = PassPipeline::new();
                let mut pq_pass = PolarQuantPass::new(self.n_bits);
                pq_pass.min_elements = self.min_elements;
                pipeline.add_pass(Box::new(pq_pass));
                pipeline.add_pass(Box::new(TypeRepropagationPass));
                let _report = pipeline.run(&mut program)?;

                let provider = MilWeightProvider::new(&program, config)?;
                Ok(provider)
            }
            InputFormat::SafeTensors | InputFormat::Gguf => {
                // SafeTensors/GGUF: load the provider directly and quantize
                // weight tensors in-memory. This avoids the template system
                // which is designed for CoreML/ANE and omits architecture-
                // specific tensors (q_norm, k_norm, etc.).
                let base_provider: Box<dyn WeightProvider> = match format {
                    InputFormat::SafeTensors => {
                        Box::new(crate::weights::SafeTensorsProvider::load(input)?)
                    }
                    InputFormat::Gguf => Box::new(crate::weights::GgufProvider::load(input)?),
                    _ => {
                        return Err(CompileError::Other(
                            "unexpected input format in SafeTensors/GGUF branch".into(),
                        ));
                    }
                };
                let config = base_provider.config().clone();

                let provider = MilWeightProvider::from_weight_provider(
                    base_provider.as_ref(),
                    config,
                    self.n_bits,
                    self.min_elements,
                )?;
                Ok(provider)
            }
            InputFormat::Unsupported(ext) => Err(CompileError::Other(format!(
                "unsupported input format '{ext}' for GPU compile. \
                 Expected .onnx, .safetensors, .gguf, or a directory containing config.json."
            ))),
        }
    }

    /// Unified build path: import to MIL IR → run pipeline → extract provider.
    fn build_with_pipeline(
        &self,
        input: &Path,
        format: InputFormat,
        pipeline: PassPipeline,
    ) -> Result<MilWeightProvider, CompileError> {
        let (mut program, config, base_provider) = match format {
            InputFormat::Onnx => {
                let (program, config) = import_onnx(input)?;
                (program, config, None)
            }
            InputFormat::SafeTensors | InputFormat::Gguf => {
                let base_provider: Box<dyn WeightProvider> = match format {
                    InputFormat::SafeTensors => {
                        Box::new(crate::weights::SafeTensorsProvider::load(input)?)
                    }
                    InputFormat::Gguf => Box::new(crate::weights::GgufProvider::load(input)?),
                    _ => unreachable!(),
                };
                let config = base_provider.config().clone();
                let result = crate::templates::weights_to_program(base_provider.as_ref())?;
                (result.program, config, Some(base_provider))
            }
            InputFormat::Unsupported(ext) => {
                return Err(CompileError::Other(format!(
                    "unsupported input format '{ext}' for GPU compile. \
                     Expected .onnx, .safetensors, .gguf, or a directory containing config.json."
                )));
            }
        };

        let _report = pipeline.run(&mut program)?;
        let mut provider = MilWeightProvider::new(&program, config)?;

        // Supplement any tensors the template system didn't emit
        if let Some(base) = base_provider.as_ref() {
            provider.supplement_from(base.as_ref())?;
        }

        Ok(provider)
    }
}

// ── Input format detection (GPU-specific subset) ────────────────────────

enum InputFormat {
    Onnx,
    SafeTensors,
    Gguf,
    Unsupported(String),
}

fn detect_format(path: &Path) -> InputFormat {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext {
            "onnx" => return InputFormat::Onnx,
            "gguf" => return InputFormat::Gguf,
            "safetensors" => return InputFormat::SafeTensors,
            _ => {}
        }
    }
    if path.is_dir() && path.join("config.json").exists() {
        return InputFormat::SafeTensors;
    }
    let ext_display = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("<none>")
        .to_string();
    InputFormat::Unsupported(ext_display)
}

// ── Format-specific importers ───────────────────────────────────────────

/// Import an ONNX model, returning the MIL program and a derived `ModelConfig`.
fn import_onnx(path: &Path) -> Result<(Program, ModelConfig), CompileError> {
    let mut onnx_model = read_onnx(path)?;
    let conv_config = ConversionConfig {
        merge_lora: true,
        model_dir: path.parent().map(|p| p.to_path_buf()),
    };
    let result = onnx_to_program_with_config(&mut onnx_model, &conv_config)?;

    // Attempt to load config.json from the same directory as the ONNX file.
    let config = if let Some(parent) = path.parent() {
        let config_path = parent.join("config.json");
        if config_path.exists() {
            crate::weights::safetensors::parse_hf_config(&config_path)?
        } else {
            derive_config_from_program(&result.program)?
        }
    } else {
        derive_config_from_program(&result.program)?
    };

    Ok((result.program, config))
}

/// Derive a minimal `ModelConfig` from the ONNX-imported program structure.
///
/// This is a best-effort fallback when no `config.json` is available alongside
/// the ONNX file. It inspects the program's main function to infer dimensions.
fn derive_config_from_program(program: &Program) -> Result<ModelConfig, CompileError> {
    use crate::weights::Architecture;
    use std::collections::HashMap;

    let function = program.main().ok_or_else(|| {
        CompileError::Other("program has no main function; cannot derive config".into())
    })?;

    // Count transformer layers by looking for repeated layer-prefixed ops.
    let mut max_layer: Option<usize> = None;
    let mut hidden_size: Option<usize> = None;

    for op in &function.body.operations {
        for output in &op.outputs {
            // Match patterns like "model.layers.N." or "/layers/N/"
            if let Some(idx) = extract_layer_index(output) {
                max_layer = Some(max_layer.map_or(idx, |m: usize| m.max(idx)));
            }
        }

        // Try to infer hidden_size from weight shapes.
        if hidden_size.is_none() {
            if let Some(shape) = op.output_types.first().and_then(|t| t.as_ref()) {
                let dims = &shape.shape;
                // A [hidden, hidden] matrix is a good signal for hidden_size.
                if dims.len() == 2 {
                    if let (Some(Some(d0)), Some(Some(d1))) = (dims.first(), dims.get(1)) {
                        if *d0 > 64 && *d0 == *d1 {
                            hidden_size = Some(*d0);
                        }
                    }
                }
            }
        }
    }

    let num_layers = max_layer.map_or(1, |m| m + 1);
    // Fallback hidden size: 4096 matches Llama-7B / Llama-2-7B.
    let h = hidden_size.unwrap_or(4096);
    let num_heads = (h / 128).max(1);

    Ok(ModelConfig {
        architecture: Architecture::Llama,
        hidden_size: h,
        intermediate_size: h * 4,
        num_hidden_layers: num_layers,
        num_attention_heads: num_heads,
        num_key_value_heads: num_heads,
        head_dim: ModelConfig::default_head_dim(h, num_heads),
        vocab_size: 32000,             // Llama-1/2 SentencePiece vocabulary size
        max_position_embeddings: 2048, // Llama-1 context length
        rms_norm_eps: 1e-5,            // Standard RMSNorm epsilon
        rope_theta: 10000.0,           // Default RoPE base frequency
        tie_word_embeddings: false,
        extra: HashMap::new(),
    })
}

/// Extract a layer index from an output name like "model.layers.5.self_attn..."
/// or "/layers/5/self_attn...".
fn extract_layer_index(name: &str) -> Option<usize> {
    // "model.layers.N." pattern
    if let Some(rest) = name.strip_prefix("model.layers.") {
        if let Some(dot_pos) = rest.find('.') {
            return rest[..dot_pos].parse().ok();
        }
    }
    // "/layers/N/" pattern (ONNX naming)
    for segment in name.split('/') {
        if !segment.is_empty()
            && segment.chars().all(|c| c.is_ascii_digit())
            && name.contains("/layers/")
        {
            return segment.parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_correct_defaults() {
        let builder = GpuCompileBuilder::new("model.onnx");
        assert_eq!(builder.input, PathBuf::from("model.onnx"));
        assert_eq!(builder.n_bits, 4);
        assert_eq!(builder.min_elements, 1024);
        assert!(builder.pipeline.is_none());
    }

    #[test]
    fn builder_methods_chain() {
        let builder = GpuCompileBuilder::new("model.onnx")
            .polar_quantize(2)
            .min_elements(2048);

        assert_eq!(builder.n_bits, 2);
        assert_eq!(builder.min_elements, 2048);
    }

    #[test]
    fn detect_format_onnx() {
        assert!(matches!(
            detect_format(Path::new("m.onnx")),
            InputFormat::Onnx
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
    fn detect_format_safetensors() {
        assert!(matches!(
            detect_format(Path::new("m.safetensors")),
            InputFormat::SafeTensors
        ));
    }

    #[test]
    fn detect_format_unsupported() {
        assert!(matches!(
            detect_format(Path::new("m.txt")),
            InputFormat::Unsupported(_)
        ));
    }

    #[test]
    fn build_nonexistent_input_returns_error() {
        let result = GpuCompileBuilder::new("does_not_exist.onnx").build();
        assert!(result.is_err());
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error for nonexistent input"),
        };
        assert!(
            err.contains("does not exist"),
            "error should mention missing file, got: {err}"
        );
    }

    #[test]
    fn extract_layer_index_dot_notation() {
        assert_eq!(
            extract_layer_index("model.layers.5.self_attn.q_proj"),
            Some(5)
        );
        assert_eq!(extract_layer_index("model.layers.0.mlp.gate"), Some(0));
        assert_eq!(extract_layer_index("model.layers.31.output"), Some(31));
    }

    #[test]
    fn extract_layer_index_slash_notation() {
        assert_eq!(extract_layer_index("/layers/3/self_attn"), Some(3));
    }

    #[test]
    fn extract_layer_index_no_match() {
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
    }
}
