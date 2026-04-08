//! GPU-targeted compilation: import → optimize → [`MilWeightProvider`].
//!
//! This module provides [`GpuCompileBuilder`], a builder that imports a model
//! from any supported format, runs an optimization/quantization [`PassPipeline`],
//! and produces a [`MilWeightProvider`] suitable for writing to a `.ironml-gpu`
//! bundle.
//!
//! All input formats (ONNX, SafeTensors, GGUF) follow the same pipeline:
//!
//! ```text
//! Input → MIL IR Program → PassPipeline → MilWeightProvider
//! ```
//!
//! Format-specific logic is limited to the initial import step. The pipeline
//! is format-agnostic — the caller decides what quantization to apply.

pub mod bundle;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use mil_rs::ScalarType;
use mil_rs::Value;
use mil_rs::convert::{ConversionConfig, onnx_to_program_with_config};
use mil_rs::ir::{PassPipeline, Program};
use mil_rs::reader::read_onnx;

use crate::error::CompileError;
use crate::weights::{MilWeightProvider, ModelConfig, WeightProvider};

/// Builder for GPU-targeted model compilation.
///
/// Imports a model from disk, runs an optimization/quantization pass pipeline,
/// and returns a [`MilWeightProvider`] that can feed weights to the GPU
/// inference runtime.
///
/// # Example
///
/// ```no_run
/// use ironmill_compile::gpu::GpuCompileBuilder;
/// use ironmill_compile::mil::PassPipeline;
///
/// // INT4 affine quantization
/// let pipeline = PassPipeline::new()
///     .with_int4(128)
///     .expect("INT4 config failed");
///
/// let provider = GpuCompileBuilder::new("model.safetensors")
///     .with_pass_pipeline(pipeline)
///     .build()
///     .expect("GPU compile failed");
/// ```
pub struct GpuCompileBuilder {
    input: PathBuf,
    pipeline: PassPipeline,
}

impl GpuCompileBuilder {
    /// Create a new builder targeting the given input model path.
    pub fn new(input: impl Into<PathBuf>) -> Self {
        Self {
            input: input.into(),
            pipeline: PassPipeline::new(),
        }
    }

    /// Set the pass pipeline for quantization/optimization.
    ///
    /// An empty pipeline (the default) produces unquantized FP16 output.
    /// The caller controls what quantization is applied by building the
    /// appropriate pipeline (INT4, INT8, AWQ, PolarQuant, etc.).
    pub fn with_pass_pipeline(mut self, pipeline: PassPipeline) -> Self {
        self.pipeline = pipeline;
        self
    }

    /// Import the model, run the pass pipeline, and return a [`MilWeightProvider`].
    ///
    /// 1. Detect input format (ONNX / SafeTensors / GGUF)
    /// 2. Import to MIL IR [`Program`]
    /// 3. Run the [`PassPipeline`]
    /// 4. Extract a [`MilWeightProvider`] from the optimized program
    pub fn build(self) -> Result<MilWeightProvider, CompileError> {
        let input = &self.input;

        if !input.exists() {
            return Err(CompileError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("input path does not exist: {}", input.display()),
            )));
        }

        let format = detect_format(input);

        // Import to MIL IR — format-specific, but produces a uniform Program.
        // For SafeTensors/GGUF, the provider is wrapped in Arc and attached to the
        // program so that passes can lazily resolve External tensor data on demand.
        let (mut program, config, supplement_tensors, _base_provider) = match format {
            InputFormat::Onnx => {
                let (program, config) = import_onnx(input)?;
                (program, config, HashMap::new(), None)
            }
            InputFormat::SafeTensors | InputFormat::Gguf => {
                let provider = load_weight_provider(input, &format)?;
                let config = provider.config().clone();

                // Wrap in Arc for shared ownership between program and builder.
                let provider_arc: Arc<dyn WeightProvider + Send + Sync> = Arc::from(provider);

                // Template emission creates External refs — no weight copy for large tensors.
                let result = crate::templates::weights_to_program(provider_arc.as_ref())?;
                let mut program = result.program;

                let supplements = collect_supplement_tensors(provider_arc.as_ref(), &program);

                // Attach provider for lazy resolution by passes.
                program.set_weight_provider(provider_arc.clone());

                (program, config, supplements, Some(provider_arc))
            }
            InputFormat::Unsupported(ext) => {
                return Err(CompileError::Other(format!(
                    "unsupported input format '{ext}' for GPU compile. \
                     Expected .onnx, .safetensors, .gguf, or a directory containing config.json."
                )));
            }
        };

        // Run the pass pipeline (may be empty for FP16 passthrough).
        // Quantization passes resolve External tensors on demand via weight_provider.
        let _report = self.pipeline.run(&mut program)?;

        // Extract weights from the optimized program.
        // MilWeightProvider resolves any remaining External tensors during extraction.
        let mut provider = MilWeightProvider::new(&mut program, config)?;

        // Inject any architecture-specific tensors that the template system
        // didn't emit (e.g. q_norm, k_norm).
        provider.apply_supplements(supplement_tensors);

        Ok(provider)
    }
}

// ── Input format detection ──────────────────────────────────────────────

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

/// Load a weight provider for SafeTensors or GGUF input.
fn load_weight_provider(
    input: &Path,
    format: &InputFormat,
) -> Result<Box<dyn WeightProvider + Send + Sync>, CompileError> {
    match format {
        InputFormat::SafeTensors => Ok(Box::new(crate::weights::SafeTensorsProvider::load(input)?)),
        InputFormat::Gguf => Ok(Box::new(crate::weights::GgufProvider::load(input)?)),
        _ => Err(CompileError::Other(
            "load_weight_provider called with non-weight format".into(),
        )),
    }
}

/// Collect tensors from the source provider that are NOT emitted into the MIL
/// program (e.g., `q_norm`, `k_norm` weight tensors used by some architectures
/// but referenced only at bundle-writing time, not as MIL const ops).
///
/// Returns a map of name → (bytes, shape, dtype) for later injection into the
/// `MilWeightProvider` via [`MilWeightProvider::apply_supplements`].
fn collect_supplement_tensors(
    provider: &dyn WeightProvider,
    program: &Program,
) -> HashMap<String, (Vec<u8>, Vec<usize>, ScalarType)> {
    // Build a set of tensor names that appear as const/constexpr ops in the program.
    let mut program_tensor_names: HashSet<String> = HashSet::new();
    for function in program.functions.values() {
        for op in &function.body.operations {
            if op.op_type == "const" || op.op_type.starts_with("constexpr_") {
                // Recover the HF-style name: prefer onnx_name, fall back to output name.
                let name = op
                    .attributes
                    .get("onnx_name")
                    .and_then(|v| match v {
                        Value::String(s) if !s.is_empty() => Some(s.clone()),
                        _ => None,
                    })
                    .or_else(|| op.outputs.first().cloned());
                if let Some(n) = name {
                    program_tensor_names.insert(n);
                }
            }
        }
    }

    provider
        .tensor_names()
        .iter()
        .filter(|name| !program_tensor_names.contains(**name))
        .filter_map(|name| match provider.tensor(name) {
            Ok(t) => Some((name.to_string(), (t.data.into_owned(), t.shape, t.dtype))),
            Err(e) => {
                eprintln!("Warning: failed to load supplement tensor '{name}': {e}");
                None
            }
        })
        .collect()
}

/// Import an ONNX model, returning the MIL program and a derived `ModelConfig`.
fn import_onnx(path: &Path) -> Result<(Program, ModelConfig), CompileError> {
    let mut onnx_model = read_onnx(path)?;
    let mut conv_config = ConversionConfig::default();
    conv_config.model_dir = path.parent().map(|p| p.to_path_buf());
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

    Ok(ModelConfig::new(Architecture::Llama)
        .with_hidden_size(h)
        .with_intermediate_size(h * 4)
        .with_num_hidden_layers(num_layers)
        .with_num_attention_heads(num_heads)
        .with_num_key_value_heads(num_heads)
        .with_head_dim(ModelConfig::default_head_dim(h, num_heads))
        .with_vocab_size(32000)
        .with_max_position_embeddings(2048)
        .with_rms_norm_eps(1e-5)
        .with_rope_theta(10000.0)
        .with_tie_word_embeddings(false)
        .with_extra(HashMap::new()))
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
    // "/layers/N/" pattern (ONNX naming) — match only the segment immediately after "/layers/"
    if let Some(rest) = name.split("/layers/").nth(1) {
        let segment = rest.split('/').next().unwrap_or("");
        if !segment.is_empty() && segment.chars().all(|c| c.is_ascii_digit()) {
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
    }

    #[test]
    fn builder_methods_chain() {
        let pipeline = PassPipeline::new();
        let builder = GpuCompileBuilder::new("model.onnx").with_pass_pipeline(pipeline);

        // Pipeline is set (not default empty) — just verify it doesn't panic.
        assert_eq!(builder.input, PathBuf::from("model.onnx"));
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
