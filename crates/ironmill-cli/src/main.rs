#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use ironmill_compile::ane::passes::ModelSplitPass;
use ironmill_compile::ane::validate::validate_ane_compatibility;
use ironmill_compile::ane::validate::{print_validation_report, validation_report_to_json};
use ironmill_compile::convert::pipeline::{convert_pipeline, parse_pipeline_manifest};
use ironmill_compile::coreml::compiler::{compile_model, is_compiler_available};
use ironmill_compile::gpu::GpuCompileBuilder;
use ironmill_compile::gpu::bundle::write_gpu_bundle;
use ironmill_compile::mil::PassPipeline;
use ironmill_compile::mil::PipelineReport;
use ironmill_compile::mil::{
    ConversionConfig, LossFunction, UpdatableModelConfig, UpdateOptimizer,
    onnx_to_program_with_config, program_to_model, program_to_multi_function_model,
    program_to_updatable_model, read_mlmodel, read_mlpackage, read_onnx, write_mlpackage,
};
use ironmill_compile::mil::{print_model_summary, print_onnx_summary};
use ironmill_compile::templates::{TemplateOptions, weights_to_program_with_options};
use ironmill_compile::weights::{GgufProvider, SafeTensorsProvider, WeightProvider};

/// CoreML specification version used for all model serialization.
///
/// Version 9 corresponds to iOS 18 / macOS 15. This must match the version
/// expected by `coremltools` and the on-device CoreML runtime.
const COREML_SPEC_VERSION: i32 = 9;

/// ironmill — Convert and optimize ML models for Apple's Neural Engine.
///
/// A Rust-native alternative to Python's coremltools. Converts ONNX models
/// to CoreML format with ANE-targeted optimizations, no Python required.
#[derive(Parser)]
#[command(name = "ironmill", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Runtime backend for inference.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum RuntimeArg {
    /// Use CoreML MLModel for inference (default, stable).
    #[value(name = "coreml")]
    CoreMl,
    /// Use ANE direct backend for inference (experimental).
    #[value(name = "ane-direct")]
    AneDirect,
}

/// Quantization mode for `--quantize`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum QuantizeArg {
    /// No quantization.
    #[default]
    None,
    /// FP16 quantization (weights and activations).
    Fp16,
    /// INT8 symmetric quantization.
    Int8,
    /// Mixed FP16/INT8 precision (predefined layer assignment).
    MixedFp16Int8,
    /// Activation-aware weight quantization (INT4, requires --cal-data).
    Awq,
    /// INT4 group quantization (group size 128).
    Int4,
    /// GPTQ optimal weight quantization (requires --cal-data).
    Gptq,
    /// D2Quant extreme low-bit (2-3 bit) quantization.
    D2quant,
}

/// Target compute units for `--target`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum TargetArg {
    /// All available compute units on this platform.
    #[default]
    All,
    /// CPU only.
    CpuOnly,
    /// CPU and GPU (Metal on macOS).
    CpuAndGpu,
    /// CPU and Neural Engine (macOS only).
    CpuAndNe,
    /// GPU-only Metal backend (macOS only).
    Gpu,
}

/// Loss function for on-device training.
#[derive(Debug, Clone, Copy, clap::ValueEnum, Default)]
pub enum LossFunctionArg {
    #[default]
    CrossEntropy,
    Mse,
}

/// Optimizer for on-device training.
#[derive(Debug, Clone, Copy, clap::ValueEnum, Default)]
pub enum OptimizerArg {
    #[default]
    Sgd,
    Adam,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum KvQuantArg {
    /// No KV cache quantization (default)
    #[value(name = "none")]
    None,
    /// TurboQuant INT8 KV cache compression
    #[value(name = "turbo-int8")]
    TurboInt8,
}

#[derive(clap::Args)]
struct CompileArgs {
    /// Path to the input model (.onnx, .mlmodel, or .mlpackage).
    input: String,

    /// Path for the output .mlpackage.
    #[arg(short, long)]
    output: Option<String>,

    /// Target compute units.
    #[arg(short, long, value_enum, default_value_t = TargetArg::All)]
    target: TargetArg,

    /// Quantization mode.
    #[arg(short, long, value_enum, default_value_t = QuantizeArg::None)]
    quantize: QuantizeArg,

    /// Calibration data directory (for int8, AWQ, or GPTQ quantization).
    #[arg(long = "cal-data", value_name = "DIR")]
    cal_data: Option<PathBuf>,

    /// Path to a TOML config for mixed-precision quantization.
    #[arg(long = "quantize-config", value_name = "PATH")]
    quantize_config: Option<PathBuf>,

    /// Weight palettization bit-width (2, 4, 6, or 8).
    #[arg(long, value_name = "BITS")]
    palettize: Option<u8>,

    /// PolarQuant weight quantization bit-width (2 or 4).
    #[arg(long = "polar-quantize", value_name = "BITS")]
    polar_quantize: Option<u8>,

    /// QuIP# (E8 lattice) 2-bit weight quantization. Requires --cal-data
    /// pointing to a directory for Hessian-guided calibration.
    #[arg(long = "quip-sharp")]
    quip_sharp: bool,

    /// Bit-width for quantization modes that accept it (e.g. d2quant: 2 or 3).
    #[arg(long, value_name = "N")]
    bits: Option<u8>,

    /// Disable fusion and optimization passes.
    #[arg(long)]
    no_fusion: bool,

    /// Set a concrete shape for a named input (for ANE compatibility).
    /// Format: "name:d0,d1,d2,...". May be repeated.
    #[arg(long = "input-shape", value_name = "NAME:SHAPE")]
    input_shapes: Vec<String>,

    /// Disable automatic LoRA adapter merging.
    #[arg(long = "no-merge-lora")]
    no_merge_lora: bool,

    /// Emit a separate adapter .mlpackage instead of merging LoRA weights.
    #[arg(long = "emit-adapter", hide = true)]
    emit_adapter: bool,

    /// Path to an external adapter file (e.g. .safetensors). May be repeated.
    #[arg(long = "adapter", hide = true, value_name = "PATH")]
    adapters: Vec<PathBuf>,

    /// Comma-separated list of layer names to mark as updatable for
    /// on-device training (e.g. "dense_0,dense_1").
    #[arg(long = "updatable-layers", value_name = "LAYERS")]
    updatable_layers: Option<String>,

    /// Learning rate for the on-device training optimizer.
    #[arg(long, default_value = "0.001")]
    learning_rate: f64,

    /// Number of training epochs for on-device fine-tuning.
    #[arg(long, default_value = "10")]
    epochs: i64,

    /// Loss function for on-device training.
    #[arg(long = "loss-function", value_enum, default_value_t = LossFunctionArg::CrossEntropy)]
    loss_function: LossFunctionArg,

    /// Optimizer for on-device training.
    #[arg(long = "optimizer", value_enum, default_value_t = OptimizerArg::Sgd)]
    optimizer_type: OptimizerArg,

    /// Split the model for speculative decoding: produce a draft model
    /// with the first N transformer layers and a full verifier model.
    /// Outputs: <name>-draft.mlpackage and <name>-verifier.mlpackage.
    #[arg(long = "split-draft-layers", value_name = "N")]
    split_draft_layers: Option<usize>,

    /// Detect MoE architecture and split into per-expert .mlpackage files.
    #[arg(long = "moe-split")]
    moe_split: bool,

    /// Detect MoE architecture and bundle experts as functions in a single .mlpackage.
    /// Uses CoreML 8+ multi-function model support.
    #[arg(long = "moe-bundle", conflicts_with = "moe_split")]
    moe_bundle: bool,

    /// Fuse the top-K most frequently activated MoE experts into a single
    /// dense .mlpackage, removing router overhead. Requires --cal-data
    /// pointing to a directory containing an `expert_profile.json` file,
    /// or a uniform profile is used when --cal-data is omitted.
    #[arg(long = "moe-fuse-topk", value_name = "K")]
    moe_fuse_topk: Option<usize>,

    /// Path to a TOML pipeline configuration file.
    /// When provided, overrides default pass selection and --no-fusion.
    #[arg(long = "pipeline-config", value_name = "PATH")]
    pipeline_config: Option<PathBuf>,

    /// Annotate each operation with its preferred compute unit (ANE, GPU,
    /// CPU, or Any) based on shape-aware ANE constraints.
    #[arg(long = "annotate-compute-units")]
    annotate_compute_units: bool,

    /// ANE memory budget per operation (e.g. "1GB", "2GB", "512MB").
    /// Operations exceeding this budget are split into smaller tiles.
    /// Default: 1GB (iOS). Use 2GB for macOS.
    #[arg(long = "ane-memory-budget", value_name = "SIZE")]
    ane_memory_budget: Option<String>,

    /// Emit ANE-optimized ops in weight-based templates (1×1 conv projections,
    /// decomposed RMSNorm, KV-cache state, prefill/decode split).
    #[arg(long)]
    ane: bool,

    /// Runtime backend for inference: "coreml" (default) or "ane-direct" (experimental).
    ///
    /// When set to "ane-direct", uses the direct ANE runtime instead of CoreML,
    /// bypassing MLModel for potentially lower latency. Requires --features ane-direct.
    #[arg(long, value_enum, default_value_t = RuntimeArg::CoreMl)]
    runtime: RuntimeArg,

    /// KV cache quantization strategy. Options: "none" (default), "turbo-int8".
    #[arg(long, value_enum, default_value = "none")]
    kv_quant: KvQuantArg,

    /// Enable QJL 1-bit bias correction for TurboQuant (requires --kv-quant turbo-int8).
    #[arg(long, default_value_t = false)]
    kv_quant_qjl: bool,

    /// Maximum sequence length for KV cache allocation (default: 2048).
    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert an ONNX model to CoreML format (.mlpackage).
    Compile(Box<CompileArgs>),

    /// Inspect a model and show its structure.
    Inspect {
        /// Path to a .mlmodel, .mlpackage, or .onnx model.
        input: String,
    },

    /// Validate whether a model is compatible with the Apple Neural Engine.
    Validate {
        /// Path to the model to validate.
        input: String,

        /// Output format: "text" (default) or "json".
        #[arg(long, default_value = "text")]
        format: String,

        /// Continue validation even if full analysis is not possible.
        #[arg(long)]
        lenient: bool,
    },

    /// Convert a multi-ONNX pipeline to coordinated .mlpackage outputs.
    CompilePipeline {
        /// Path to the pipeline manifest (.toml).
        manifest: String,

        /// Output directory for .mlpackage files and pipeline.json.
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Compare two pipeline configurations on a model and report metrics.
    PipelineReport {
        /// Path to the input ONNX model.
        input: String,

        /// Path to the first pipeline config (TOML).
        #[arg(long = "config-a", value_name = "PATH")]
        config_a: PathBuf,

        /// Path to the second pipeline config (TOML).
        #[arg(long = "config-b", value_name = "PATH")]
        config_b: PathBuf,
    },
}

/// Count total operations across all functions in a program.
fn count_ops(program: &ironmill_compile::mil::Program) -> usize {
    program
        .functions
        .values()
        .map(|f| f.body.operations.len())
        .sum()
}

/// Detected input format for the compile command.
enum InputFormat {
    Onnx,
    SafeTensors,
    Gguf,
    CoreMl(String),
    Unknown(String),
}

/// Detect the input format from a file path.
fn detect_input_format(path: &Path) -> InputFormat {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext {
            "onnx" => return InputFormat::Onnx,
            "gguf" => return InputFormat::Gguf,
            "mlmodel" | "mlpackage" => return InputFormat::CoreMl(ext.to_string()),
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

fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile(args) => {
            let merge_lora = !args.no_merge_lora;
            cmd_compile(
                &args.input,
                CompileOpts {
                    output: args.output,
                    target: args.target,
                    quantize: args.quantize,
                    cal_data: args.cal_data,
                    quantize_config: args.quantize_config,
                    palettize: args.palettize,
                    polar_quantize: args.polar_quantize,
                    quip_sharp: args.quip_sharp,
                    bits: args.bits,
                    no_fusion: args.no_fusion,
                    input_shapes: args.input_shapes,
                    merge_lora,
                    emit_adapter: args.emit_adapter,
                    adapters: args.adapters,
                    updatable_layers: args.updatable_layers,
                    learning_rate: args.learning_rate,
                    epochs: args.epochs,
                    loss_function: args.loss_function,
                    optimizer_type: args.optimizer_type,
                    split_draft_layers: args.split_draft_layers,
                    moe_split: args.moe_split,
                    moe_bundle: args.moe_bundle,
                    moe_fuse_topk: args.moe_fuse_topk,
                    pipeline_config: args.pipeline_config,
                    annotate_compute_units: args.annotate_compute_units,
                    ane_memory_budget: args.ane_memory_budget,
                    ane: args.ane,
                    runtime: args.runtime,
                    kv_quant: args.kv_quant,
                    kv_quant_qjl: args.kv_quant_qjl,
                    max_seq_len: args.max_seq_len,
                },
            )
        }
        Commands::Inspect { input } => cmd_inspect(&input),
        Commands::Validate {
            input,
            format,
            lenient,
        } => cmd_validate(&input, &format, lenient),
        Commands::CompilePipeline { manifest, output } => {
            cmd_compile_pipeline(&manifest, output.as_deref())
        }
        Commands::PipelineReport {
            input,
            config_a,
            config_b,
        } => cmd_pipeline_report(&input, &config_a, &config_b),
    }
}

/// Parse an `--input-shape` value like `"input:1,3,224,224"` into `(name, dims)`.
fn parse_input_shape(s: &str) -> Result<(String, Vec<usize>)> {
    let (name, dims_str) = s.split_once(':').ok_or_else(|| {
        anyhow::anyhow!("invalid --input-shape format: expected 'name:d0,d1,...', got '{s}'")
    })?;
    let dims: Vec<usize> = dims_str
        .split(',')
        .map(|d| {
            d.trim()
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("invalid dimension '{d}' in --input-shape '{s}'"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((name.to_string(), dims))
}

struct CompileOpts {
    output: Option<String>,
    target: TargetArg,
    quantize: QuantizeArg,
    cal_data: Option<PathBuf>,
    quantize_config: Option<PathBuf>,
    palettize: Option<u8>,
    polar_quantize: Option<u8>,
    quip_sharp: bool,
    bits: Option<u8>,
    no_fusion: bool,
    input_shapes: Vec<String>,
    merge_lora: bool,
    emit_adapter: bool,
    adapters: Vec<PathBuf>,
    updatable_layers: Option<String>,
    learning_rate: f64,
    epochs: i64,
    loss_function: LossFunctionArg,
    optimizer_type: OptimizerArg,
    split_draft_layers: Option<usize>,
    moe_split: bool,
    moe_bundle: bool,
    moe_fuse_topk: Option<usize>,
    pipeline_config: Option<PathBuf>,
    annotate_compute_units: bool,
    ane_memory_budget: Option<String>,
    ane: bool,
    runtime: RuntimeArg,
    kv_quant: KvQuantArg,
    kv_quant_qjl: bool,
    max_seq_len: usize,
}

fn cmd_compile(input: &str, opts: CompileOpts) -> Result<()> {
    let input_path = Path::new(input);
    if !input_path.exists() {
        bail!("Input file not found: {input}");
    }

    if opts.max_seq_len == 0 {
        bail!("--max-seq-len must be at least 1");
    }

    if opts.target == TargetArg::Gpu {
        return compile_for_gpu(input_path, &opts);
    }

    if !matches!(opts.target, TargetArg::All | TargetArg::CpuAndNe) {
        eprintln!(
            "Note: --target '{:?}' will be fully supported in Phase 3. Proceeding with default target.",
            opts.target
        );
    }

    // TurboQuant validation
    if opts.kv_quant == KvQuantArg::TurboInt8 && !matches!(opts.runtime, RuntimeArg::AneDirect) {
        bail!("--kv-quant turbo-int8 requires --runtime ane-direct");
    }
    if opts.kv_quant_qjl && opts.kv_quant != KvQuantArg::TurboInt8 {
        bail!("--kv-quant-qjl requires --kv-quant turbo-int8");
    }
    if opts.kv_quant == KvQuantArg::TurboInt8 {
        eprintln!(
            "TurboQuant: INT8 KV cache (max_seq_len={}{})",
            opts.max_seq_len,
            if opts.kv_quant_qjl {
                ", QJL enabled"
            } else {
                ""
            }
        );
    }

    match detect_input_format(input_path) {
        InputFormat::Onnx => compile_from_onnx(input_path, &opts),
        InputFormat::SafeTensors | InputFormat::Gguf => compile_from_weights(input_path, &opts),
        InputFormat::CoreMl(ext) => {
            if opts.moe_split || opts.moe_bundle || opts.moe_fuse_topk.is_some() {
                eprintln!(
                    "Note: --moe-split/--moe-bundle/--moe-fuse-topk is only supported for ONNX input. Ignoring."
                );
            }
            compile_coreml(input_path, &ext)
        }
        InputFormat::Unknown(ext) => {
            bail!(
                "Unrecognized input format (extension: '{ext}'). \
                 Supported formats: .onnx, .safetensors, .gguf, .mlmodel, .mlpackage, \
                 or a HuggingFace model directory containing config.json."
            );
        }
    }
}

/// GPU compilation path: builds a pass pipeline and writes a `.ironml-gpu` bundle.
fn compile_for_gpu(input_path: &Path, opts: &CompileOpts) -> Result<()> {
    let output = opts
        .output
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| input_path.with_extension("ironml-gpu"));

    // ── Regular (MIL IR) path ───────────────────────────────────────
    // Reject conflicting quantization flags.
    if opts.quantize != QuantizeArg::None && opts.polar_quantize.is_some() {
        anyhow::bail!("Cannot specify both --quantize and --polar-quantize. Use only one.");
    }
    if opts.quip_sharp && opts.quantize != QuantizeArg::None {
        anyhow::bail!("Cannot specify both --quantize and --quip-sharp. Use only one.");
    }
    if opts.quip_sharp && opts.polar_quantize.is_some() {
        anyhow::bail!("Cannot specify both --polar-quantize and --quip-sharp. Use only one.");
    }

    // Build the pipeline — handles all quantization methods uniformly.
    let pipeline = build_pass_pipeline(opts)?;

    let provider = GpuCompileBuilder::new(input_path)
        .with_pass_pipeline(pipeline)
        .build()
        .context("GPU compilation failed")?;

    write_gpu_bundle(&provider, &output).context("Failed to write GPU bundle")?;
    eprintln!("Wrote GPU bundle to {}", output.display());
    Ok(())
}

/// Build the optimization pass pipeline from CLI options.
fn build_pass_pipeline(opts: &CompileOpts) -> Result<PassPipeline> {
    if let Some(ref config_path) = opts.pipeline_config {
        return PassPipeline::with_config(config_path)
            .with_context(|| format!("Failed to load pipeline config: {}", config_path.display()));
    }

    let mut pipeline = PassPipeline::new();

    if opts.no_fusion {
        pipeline = pipeline.without_fusion();
    }

    // Parse and add input shapes
    if !opts.input_shapes.is_empty() {
        let mut shapes = HashMap::new();
        for raw in &opts.input_shapes {
            let (name, dims) = parse_input_shape(raw)?;
            shapes.insert(name, dims);
        }
        pipeline = pipeline.with_shapes(shapes);
    }

    // Add quantization. If a quantize-config file is provided it takes
    // precedence over the --quantize flag to avoid double quantization.
    if let Some(ref config_path) = opts.quantize_config {
        let pass = ironmill_compile::ane::passes::MixedPrecisionPass::from_config_file(config_path)
            .context("Failed to configure mixed-precision from config file")?;
        pipeline.add_pass(Box::new(pass));
    } else {
        match opts.quantize {
            QuantizeArg::Fp16 => {
                pipeline = pipeline
                    .with_fp16()
                    .context("Failed to configure FP16 quantization")?;
            }
            QuantizeArg::Int8 => {
                pipeline = pipeline
                    .with_int8(opts.cal_data.clone())
                    .context("Failed to configure INT8 quantization")?;
            }
            QuantizeArg::MixedFp16Int8 => {
                let config =
                    ironmill_compile::ane::passes::MixedPrecisionConfig::preset_fp16_int8();
                let pass = ironmill_compile::ane::passes::MixedPrecisionPass::new(config);
                pipeline.add_pass(Box::new(pass));
            }
            QuantizeArg::Awq => {
                let cal_dir = opts
                    .cal_data
                    .as_ref()
                    .context("AWQ quantization requires --cal-data pointing to a directory containing awq_magnitudes.json")?;
                let mag_path = cal_dir.join("awq_magnitudes.json");
                let json = std::fs::read_to_string(&mag_path).with_context(|| {
                    format!(
                        "Failed to read AWQ channel magnitudes from {}",
                        mag_path.display()
                    )
                })?;
                let channel_magnitudes: HashMap<String, Vec<f32>> = serde_json::from_str(&json)
                    .with_context(|| {
                        format!(
                            "Failed to parse AWQ channel magnitudes from {}",
                            mag_path.display()
                        )
                    })?;

                // Optionally load raw calibration activations for exact loss.
                let act_path = cal_dir.join("awq_activations.json");
                let tc_path = cal_dir.join("awq_token_count.json");
                let (cal_activations, cal_token_count) = if act_path.exists() && tc_path.exists() {
                    let act_json = std::fs::read_to_string(&act_path).with_context(|| {
                        format!("Failed to read AWQ activations from {}", act_path.display())
                    })?;
                    let activations: HashMap<String, Vec<f32>> = serde_json::from_str(&act_json)
                        .with_context(|| {
                            format!(
                                "Failed to parse AWQ activations from {}",
                                act_path.display()
                            )
                        })?;
                    let tc_json = std::fs::read_to_string(&tc_path).with_context(|| {
                        format!("Failed to read AWQ token count from {}", tc_path.display())
                    })?;
                    let token_count: usize = serde_json::from_str(&tc_json).with_context(|| {
                        format!("Failed to parse AWQ token count from {}", tc_path.display())
                    })?;
                    (activations, token_count)
                } else {
                    (HashMap::new(), 0)
                };

                pipeline = pipeline
                    .with_awq(channel_magnitudes, 128, cal_activations, cal_token_count)
                    .context("Failed to configure AWQ quantization")?;
            }
            QuantizeArg::Int4 => {
                let group_size = 128; // industry standard
                pipeline = pipeline
                    .with_int4(group_size)
                    .context("Failed to configure INT4 quantization")?;
            }
            QuantizeArg::Gptq => {
                let cal_dir = opts
                    .cal_data
                    .as_ref()
                    .context("GPTQ quantization requires --cal-data pointing to a directory containing gptq_hessians.json")?;
                let hessian_path = cal_dir.join("gptq_hessians.json");
                let json = std::fs::read_to_string(&hessian_path).with_context(|| {
                    format!(
                        "Failed to read GPTQ Hessian data from {}",
                        hessian_path.display()
                    )
                })?;
                let hessian_data: HashMap<String, (Vec<f32>, usize, usize)> =
                    serde_json::from_str(&json).with_context(|| {
                        format!(
                            "Failed to parse GPTQ Hessian data from {}",
                            hessian_path.display()
                        )
                    })?;
                let bits = opts.bits.unwrap_or(4);
                pipeline = pipeline
                    .with_gptq(hessian_data, bits, 128, 128, 0.01)
                    .context("Failed to configure GPTQ quantization")?;
            }
            QuantizeArg::D2quant => {
                let bits = opts.bits.unwrap_or(2);
                if bits != 2 && bits != 3 {
                    bail!("D2Quant --bits must be 2 or 3, got {bits}");
                }
                pipeline = pipeline
                    .with_d2quant(bits, 128, 0.99, None)
                    .context("Failed to configure D2Quant quantization")?;
            }
            QuantizeArg::None => {}
        }
    }

    // Add palettization
    if let Some(bits) = opts.palettize {
        pipeline = pipeline
            .with_palettize(bits)
            .context("Failed to configure palettization")?;
    }

    // Add PolarQuant
    if let Some(bits) = opts.polar_quantize {
        pipeline = pipeline
            .with_polar_quant(bits)
            .context("Failed to configure PolarQuant")?;
    }

    // Add QuIP# (E8 lattice 2-bit quantization)
    if opts.quip_sharp {
        if opts.cal_data.is_none() {
            bail!(
                "--quip-sharp requires --cal-data pointing to a calibration data directory \
                 for Hessian-guided rounding"
            );
        }
        pipeline = pipeline
            .with_quip_sharp(2, 42)
            .context("Failed to configure QuIP# quantization")?;
    }

    // Add compute unit annotations (should run last, after all transforms).
    if opts.annotate_compute_units {
        pipeline.add_pass(Box::new(
            ironmill_compile::ane::passes::ComputeUnitAnnotationPass,
        ));
    }

    // Add op splitting for ANE memory budget
    if let Some(ref budget_str) = opts.ane_memory_budget {
        let budget = ironmill_compile::ane::passes::op_split::parse_memory_size(budget_str)
            .map_err(|e| anyhow::anyhow!("invalid --ane-memory-budget: {e}"))?;
        pipeline.add_pass(Box::new(
            ironmill_compile::ane::passes::OpSplittingPass::new(budget),
        ));
        eprintln!("  ANE memory budget: {budget_str} ({budget} bytes)");
    }

    Ok(pipeline)
}

/// Derive a base stem from `--output` (stripping any extension) or from the
/// input filename. Used by modes that emit multiple output files.
fn resolve_output_stem(output: Option<&str>, input_path: &Path) -> String {
    if let Some(out) = output {
        let p = Path::new(out);
        p.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(out)
            .to_string()
    } else {
        input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string()
    }
}

/// Resolve a single output path from `--output` or `{input_stem}{default_suffix}`.
fn resolve_output_path(output: Option<&str>, input_path: &Path, default_suffix: &str) -> String {
    if let Some(out) = output {
        out.to_string()
    } else {
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        format!("{stem}{default_suffix}")
    }
}

/// Convert a MIL program to CoreML, write as .mlpackage, and print status.
fn write_coreml_package(
    program: &ironmill_compile::mil::Program,
    output_path: &str,
    label: &str,
) -> Result<()> {
    let model = program_to_model(program, COREML_SPEC_VERSION)
        .with_context(|| format!("Failed to convert {label} to CoreML"))?;
    eprintln!("  Writing {label}: {output_path}");
    write_mlpackage(&model, output_path)
        .with_context(|| format!("Failed to write {output_path}"))?;
    Ok(())
}

/// MoE split: write per-expert .mlpackage files and a manifest.
/// Returns `Ok(true)` if MoE was detected and handled.
fn emit_moe_split(
    program: &mut ironmill_compile::mil::Program,
    input_path: &Path,
    opts: &CompileOpts,
) -> Result<bool> {
    if !opts.moe_split {
        return Ok(false);
    }

    use ironmill_compile::convert::moe::{detect_moe, split_moe};

    let topology = match detect_moe(program) {
        Some(t) => t,
        None => {
            eprintln!(
                "Note: --moe-split specified but no MoE pattern detected. Producing single output."
            );
            return Ok(false);
        }
    };

    eprintln!(
        "MoE architecture detected: {} experts",
        topology.expert_count
    );
    let split_result = split_moe(program, &topology).context("failed to split MoE program")?;
    let stem = resolve_output_stem(opts.output.as_deref(), input_path);

    let shared_path = format!("{stem}-shared.mlpackage");
    write_coreml_package(&split_result.shared, &shared_path, "shared layers")?;

    for (i, expert) in split_result.experts.iter().enumerate() {
        let expert_path = format!("{stem}-expert-{i}.mlpackage");
        write_coreml_package(expert, &expert_path, &format!("expert {i}"))?;
    }

    // Write manifest
    let manifest_path = format!("{stem}-manifest.json");
    let manifest_json = serde_json::to_string_pretty(&split_result.manifest)
        .context("Failed to serialize MoE manifest")?;
    std::fs::write(&manifest_path, &manifest_json)
        .with_context(|| format!("Failed to write {manifest_path}"))?;
    eprintln!("  Wrote manifest: {manifest_path}");

    eprintln!();
    eprintln!("MoE split complete.");
    Ok(true)
}

/// MoE bundle: write a single multi-function .mlpackage with all experts.
/// Returns `Ok(true)` if MoE was detected and handled.
fn emit_moe_bundle(
    program: &mut ironmill_compile::mil::Program,
    input_path: &Path,
    opts: &CompileOpts,
) -> Result<bool> {
    if !opts.moe_bundle {
        return Ok(false);
    }

    use ironmill_compile::convert::moe::{detect_moe, split_moe};

    let topology = match detect_moe(program) {
        Some(t) => t,
        None => {
            eprintln!(
                "Note: --moe-bundle specified but no MoE pattern detected. Producing single output."
            );
            return Ok(false);
        }
    };

    eprintln!(
        "MoE architecture detected: {} experts",
        topology.expert_count
    );
    let split_result = split_moe(program, &topology).context("failed to split MoE program")?;

    let bundle_model = program_to_multi_function_model(&split_result, COREML_SPEC_VERSION)
        .context("Failed to convert MoE programs to multi-function CoreML model")?;

    let bundle_path = resolve_output_path(opts.output.as_deref(), input_path, "-bundle.mlpackage");
    eprintln!(
        "  Writing multi-function bundle ({} experts): {bundle_path}",
        topology.expert_count
    );
    write_mlpackage(&bundle_model, &bundle_path)
        .with_context(|| format!("Failed to write {bundle_path}"))?;

    // Write manifest alongside the bundle
    let manifest_path = format!(
        "{}-manifest.json",
        bundle_path
            .strip_suffix(".mlpackage")
            .unwrap_or(&bundle_path)
    );
    let manifest_json = serde_json::to_string_pretty(&split_result.manifest)
        .context("Failed to serialize MoE manifest")?;
    std::fs::write(&manifest_path, &manifest_json)
        .with_context(|| format!("Failed to write {manifest_path}"))?;
    eprintln!("  Wrote manifest: {manifest_path}");

    compile_output(&bundle_path);

    eprintln!();
    eprintln!("MoE bundle complete.");
    Ok(true)
}

/// MoE top-K fusion: fuse the most frequently activated experts into a dense model.
/// Returns `Ok(true)` if fusion was performed.
fn emit_topk_fusion(
    program: &mut ironmill_compile::mil::Program,
    input_path: &Path,
    opts: &CompileOpts,
) -> Result<bool> {
    let k = match opts.moe_fuse_topk {
        Some(k) => k,
        None => return Ok(false),
    };

    use ironmill_compile::convert::moe::{ExpertFrequencyProfile, fuse_top_k_experts};

    // Load or build frequency profile
    let profile = if let Some(ref cal_dir) = opts.cal_data {
        let profile_path = cal_dir.join("expert_profile.json");
        if profile_path.exists() {
            let json = std::fs::read_to_string(&profile_path)
                .with_context(|| format!("Failed to read {}", profile_path.display()))?;
            ExpertFrequencyProfile::from_json(&json)
                .with_context(|| format!("Failed to parse {}", profile_path.display()))?
        } else {
            eprintln!(
                "  Note: {} not found, using uniform profile.",
                profile_path.display()
            );
            let expert_count = ironmill_compile::convert::moe::detect_moe(program)
                .map(|t| t.expert_count)
                .unwrap_or(k);
            ExpertFrequencyProfile::uniform(expert_count)
        }
    } else {
        eprintln!("  Note: --cal-data not provided, using uniform profile.");
        let expert_count = ironmill_compile::convert::moe::detect_moe(program)
            .map(|t| t.expert_count)
            .unwrap_or(k);
        ExpertFrequencyProfile::uniform(expert_count)
    };

    let fuse_result = match fuse_top_k_experts(program, k, &profile)? {
        Some(r) => r,
        None => {
            eprintln!(
                "Note: --moe-fuse-topk specified but no MoE pattern detected. Producing single output."
            );
            return Ok(false);
        }
    };

    eprintln!(
        "MoE top-{k} fusion: kept experts {:?}, discarded {:?}",
        fuse_result.kept_expert_indices, fuse_result.discarded_expert_indices
    );
    eprintln!(
        "  Fused program: {} ops (was {} ops)",
        count_ops(&fuse_result.program),
        count_ops(program)
    );

    let output_path = resolve_output_path(opts.output.as_deref(), input_path, "-fused.mlpackage");
    write_coreml_package(&fuse_result.program, &output_path, "fused model")?;
    compile_output(&output_path);

    eprintln!();
    eprintln!("MoE top-{k} fusion complete.");
    Ok(true)
}

/// Speculative decoding split: produce draft and verifier .mlpackage files.
fn emit_speculative_split(
    program: &mut ironmill_compile::mil::Program,
    input_path: &Path,
    opts: &CompileOpts,
    n_layers: usize,
) -> Result<()> {
    let stem = resolve_output_stem(opts.output.as_deref(), input_path);
    let output_parent = opts
        .output
        .as_deref()
        .map(|o| Path::new(o).parent().unwrap_or(Path::new(".")));

    let build_path = |suffix: &str| -> String {
        match output_parent {
            Some(parent) => parent
                .join(format!("{stem}{suffix}"))
                .to_string_lossy()
                .into_owned(),
            None => format!("{stem}{suffix}"),
        }
    };

    let split_pass = ModelSplitPass::new(n_layers);
    let split_result = split_pass
        .split(program)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("Failed to split model for speculative decoding")?;

    let draft_path = build_path("-draft.mlpackage");
    write_coreml_package(
        &split_result.draft,
        &draft_path,
        &format!("draft model ({n_layers} layers)"),
    )?;

    let verifier_path = build_path("-verifier.mlpackage");
    write_coreml_package(
        &split_result.verifier,
        &verifier_path,
        "verifier model (full)",
    )?;

    compile_output(&draft_path);
    compile_output(&verifier_path);
    Ok(())
}

/// Standard CoreML output: produce a single .mlpackage (optionally updatable).
fn emit_coreml(
    program: &mut ironmill_compile::mil::Program,
    input_path: &Path,
    opts: &CompileOpts,
) -> Result<()> {
    let model = if let Some(ref layers) = opts.updatable_layers {
        let layer_names: Vec<String> = layers
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        anyhow::ensure!(
            opts.epochs > 0,
            "--epochs must be a positive integer, got {}",
            opts.epochs
        );
        anyhow::ensure!(
            opts.learning_rate > 0.0 && opts.learning_rate.is_finite(),
            "--learning-rate must be a positive finite number, got {}",
            opts.learning_rate
        );

        let loss_fn = match opts.loss_function {
            LossFunctionArg::CrossEntropy => LossFunction::CategoricalCrossEntropy,
            LossFunctionArg::Mse => LossFunction::MeanSquaredError,
        };
        let opt = match opts.optimizer_type {
            OptimizerArg::Adam => UpdateOptimizer::Adam,
            OptimizerArg::Sgd => UpdateOptimizer::Sgd,
        };

        let mut config = UpdatableModelConfig::default();
        config.updatable_layers = layer_names.clone();
        config.learning_rate = opts.learning_rate;
        config.epochs = opts.epochs;
        config.loss_function = loss_fn;
        config.optimizer = opt;

        eprintln!(
            "Updatable model: {} layer(s) marked for on-device training",
            layer_names.len()
        );
        program_to_updatable_model(program, COREML_SPEC_VERSION, &config)
            .context("Failed to convert MIL IR to updatable CoreML protobuf")?
    } else {
        program_to_model(program, COREML_SPEC_VERSION)
            .context("Failed to convert MIL IR to CoreML protobuf")?
    };

    let output_path = resolve_output_path(opts.output.as_deref(), input_path, ".mlpackage");
    eprintln!("Writing CoreML package: {output_path}");
    write_mlpackage(&model, &output_path)
        .with_context(|| format!("Failed to write mlpackage: {output_path}"))?;

    compile_output(&output_path);
    Ok(())
}

/// ANE-direct output: compile to a `.ironml` bundle for the direct ANE runtime.
fn emit_ane_direct(
    program: &mut ironmill_compile::mil::Program,
    input_path: &Path,
    opts: &CompileOpts,
) -> Result<()> {
    #[cfg(feature = "ane-direct")]
    {
        use ironmill_compile::ane::bundle::{AneCompileConfig, compile_model_bundle};

        let output_dir = resolve_output_path(opts.output.as_deref(), input_path, ".ironml");

        eprintln!("Compiling for ANE direct runtime → {output_dir}");

        let config = AneCompileConfig::default();
        let bundle = compile_model_bundle(program, &config)
            .context("ANE model bundle compilation failed")?;

        let output_path = std::path::Path::new(&output_dir);
        bundle
            .save(output_path)
            .context("Failed to save .ironml bundle")?;

        eprintln!(
            "  ✓ {} sub-program(s) written to {output_dir}",
            bundle.sub_programs.len(),
        );
    }
    #[cfg(not(feature = "ane-direct"))]
    {
        let _ = (program, input_path, opts);
        anyhow::bail!("ANE direct runtime requires --features ane-direct");
    }
    #[allow(unreachable_code)]
    Ok(())
}

/// Run the pass pipeline, dispatch to the appropriate output handler, and
/// print warnings. Shared by both ONNX and weights compilation paths.
fn compile_and_emit(
    program: &mut ironmill_compile::mil::Program,
    input_path: &Path,
    opts: &CompileOpts,
    warnings: Vec<String>,
) -> Result<()> {
    // Build and run the pass pipeline.
    let pipeline = build_pass_pipeline(opts)?;

    eprintln!("Running optimization passes...");
    let report = pipeline
        .run(program)
        .context("Optimization pipeline failed")?;

    print_pipeline_report(&report);

    // Materialize all External tensor refs (from lazy loading and
    // spill-after-quantize) before emission. Serialization paths
    // (CoreML protobuf, ANE blob packing) require inline byte data.
    if program.has_weight_provider() {
        program
            .materialize_all()
            .context("Failed to materialize external tensors before emission")?;
    }

    // MoE modes return early (no warnings / "Conversion complete" footer).
    if emit_moe_split(program, input_path, opts)?
        || emit_moe_bundle(program, input_path, opts)?
        || emit_topk_fusion(program, input_path, opts)?
    {
        return Ok(());
    }

    // Speculative split or standard single-model output.
    if let Some(n_layers) = opts.split_draft_layers {
        emit_speculative_split(program, input_path, opts, n_layers)?;
    } else {
        match opts.runtime {
            RuntimeArg::CoreMl => emit_coreml(program, input_path, opts)?,
            RuntimeArg::AneDirect => emit_ane_direct(program, input_path, opts)?,
        }
    }

    // Print warnings
    if !warnings.is_empty() {
        eprintln!();
        eprintln!("Warnings:");
        let unsupported: Vec<&str> = warnings.iter().map(|w| w.as_str()).collect();
        eprintln!(
            "  {} unsupported op(s) skipped: {:?}",
            unsupported.len(),
            unsupported
        );
    }

    eprintln!();
    eprintln!("Conversion complete.");
    Ok(())
}

/// Print a pipeline report summary.
fn print_pipeline_report(report: &PipelineReport) {
    for result in &report.pass_results {
        if result.ops_after < result.ops_before {
            let diff = result.ops_before - result.ops_after;
            let label = if diff == 1 { "op" } else { "ops" };
            eprintln!(
                "  {}: removed {diff} {label} ({:.2?})",
                result.name, result.elapsed
            );
        } else if result.ops_after > result.ops_before {
            let diff = result.ops_after - result.ops_before;
            let label = if diff == 1 { "op" } else { "ops" };
            eprintln!(
                "  {}: added {diff} {label} ({:.2?})",
                result.name, result.elapsed
            );
        } else {
            eprintln!("  {}: no changes ({:.2?})", result.name, result.elapsed);
        }
    }
    eprintln!("  Total pipeline time: {:.2?}", report.total_elapsed());
    eprintln!();
}

fn compile_from_onnx(input_path: &Path, opts: &CompileOpts) -> Result<()> {
    let input_display = input_path.display();

    // 1. Read ONNX model
    eprintln!("Reading ONNX model: {input_display}");
    let mut onnx_model = read_onnx(input_path)
        .with_context(|| format!("Failed to read ONNX model: {input_display}"))?;
    print_onnx_summary(&onnx_model);

    if opts.merge_lora {
        eprintln!("  LoRA merge: enabled (use --no-merge-lora to disable)");
    }
    if opts.emit_adapter {
        bail!(
            "--emit-adapter is not yet implemented. This flag is reserved for future adapter export support."
        );
    }
    if !opts.adapters.is_empty() {
        bail!(
            "--adapter is not yet implemented. This flag is reserved for future external adapter loading ({} path(s) provided).",
            opts.adapters.len()
        );
    }
    eprintln!();

    // 2. Convert to MIL IR
    eprintln!("Converting to CoreML MIL IR...");
    let mut config = ConversionConfig::default();
    config.merge_lora = opts.merge_lora;
    config.model_dir = input_path.parent().map(|p| p.to_path_buf());
    let result = onnx_to_program_with_config(&mut onnx_model, &config)
        .context("Failed to convert ONNX model to MIL IR")?;
    let mut program = result.program;
    let warnings = result.warnings;
    eprintln!(
        "  Converted: {} functions, {} ops",
        program.functions.len(),
        count_ops(&program)
    );
    eprintln!();

    // 3. Run shared pipeline + output
    compile_and_emit(&mut program, input_path, opts, warnings)
}

/// Compile a model from SafeTensors or GGUF weight files.
fn compile_from_weights(input_path: &Path, opts: &CompileOpts) -> Result<()> {
    let input_display = input_path.display();

    let is_gguf = input_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e == "gguf");

    // 1. Load weight provider (wrapped in Arc for program attachment)
    let provider: Arc<dyn WeightProvider + Send + Sync> = if is_gguf {
        eprintln!("Loading GGUF model: {input_display}");
        let p = GgufProvider::load(input_path)
            .with_context(|| format!("Failed to load GGUF model: {input_display}"))?;
        Arc::new(p)
    } else {
        // SafeTensors — resolve model directory (accepts a .safetensors file or directory)
        let model_dir = if input_path.is_dir() {
            input_path.to_path_buf()
        } else {
            input_path
                .parent()
                .ok_or_else(|| {
                    anyhow::anyhow!("Cannot determine model directory from: {input_display}")
                })?
                .to_path_buf()
        };
        eprintln!("Loading SafeTensors model: {}", model_dir.display());
        let p = SafeTensorsProvider::load(&model_dir).with_context(|| {
            format!("Failed to load SafeTensors model: {}", model_dir.display())
        })?;
        Arc::new(p)
    };

    eprintln!("  Architecture: {}", provider.config().architecture);
    eprintln!("  Parameters: {} tensors", provider.tensor_names().len());
    eprintln!();

    // 2. Build MIL program from architecture template
    let ane_mode = opts.ane || matches!(opts.runtime, RuntimeArg::AneDirect);
    let mut template_opts = TemplateOptions::default();
    template_opts.ane = ane_mode;

    eprintln!(
        "Building MIL program from {} template...",
        provider.config().architecture
    );
    let result = weights_to_program_with_options(provider.as_ref(), &template_opts)
        .context("Failed to build MIL program from weights")?;
    let mut program = result.program;
    let warnings = result.warnings;

    // Attach provider for lazy tensor resolution by passes and spill-after-quantize.
    program.set_weight_provider(provider);

    eprintln!(
        "  Built: {} functions, {} ops",
        program.functions.len(),
        count_ops(&program)
    );
    eprintln!();

    // 3. Run shared pipeline + output
    compile_and_emit(&mut program, input_path, opts, warnings)
}

/// Compile a .mlpackage using xcrun coremlcompiler.
///
/// This is a convenience step — the .mlpackage is already complete on disk.
/// coremlcompiler failures are logged as warnings since they may indicate
/// model issues that the on-device runtime would also encounter, but they
/// don't prevent the output file from being usable.
fn compile_output(output_path: &str) {
    if is_compiler_available() {
        eprintln!("Compiling with xcrun coremlcompiler...");
        let output_dir = Path::new(output_path).parent().unwrap_or(Path::new("."));
        let compiled = compile_model(output_path, output_dir)
            .with_context(|| format!("xcrun coremlcompiler failed for {output_path}"))?;
        eprintln!("Done: {}", compiled.display());
    } else {
        eprintln!("Note: xcrun coremlcompiler not found — skipping compilation step.");
        eprintln!("  The .mlpackage can be compiled on a Mac with Xcode installed.");
    }
}

fn compile_coreml(input_path: &Path, ext: &str) -> Result<()> {
    let input_display = input_path.display();
    eprintln!("Input is already CoreML format (.{ext}): {input_display}");

    if !is_compiler_available() {
        bail!("xcrun coremlcompiler is not available. Install Xcode to compile CoreML models.");
    }

    let output_dir = input_path.parent().unwrap_or(Path::new("."));
    eprintln!("Compiling with xcrun coremlcompiler...");
    let compiled = compile_model(input_path, output_dir)
        .with_context(|| format!("Failed to compile {input_display}"))?;
    eprintln!("Done: {}", compiled.display());

    Ok(())
}

fn cmd_inspect(input: &str) -> Result<()> {
    let input_path = Path::new(input);
    if !input_path.exists() {
        bail!("Input file not found: {input}");
    }

    let ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "onnx" => {
            let model = read_onnx(input_path)
                .with_context(|| format!("Failed to read ONNX model: {input}"))?;
            print_onnx_summary(&model);
        }
        "mlmodel" => {
            let model = read_mlmodel(input_path)
                .with_context(|| format!("Failed to read mlmodel: {input}"))?;
            print_model_summary(&model);
        }
        "mlpackage" => {
            let model = read_mlpackage(input_path)
                .with_context(|| format!("Failed to read mlpackage: {input}"))?;
            print_model_summary(&model);
        }
        _ => bail!("Unsupported format '.{ext}'. Expected .onnx, .mlmodel, or .mlpackage"),
    }

    Ok(())
}

fn cmd_validate(input: &str, format: &str, lenient: bool) -> Result<()> {
    let input_path = Path::new(input);
    if !input_path.exists() {
        bail!("Input file not found: {input}");
    }

    let ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let is_text = format == "text";

    if is_text {
        eprintln!("Validating: {input}");
        eprintln!();
    }

    let report = match ext.as_str() {
        "onnx" => {
            let mut onnx_model = read_onnx(input_path)
                .with_context(|| format!("Failed to read ONNX model: {input}"))?;
            if is_text {
                eprintln!("✓ ONNX model parsed successfully");
                eprintln!("  Converting to MIL IR...");
            }

            let result = onnx_to_program_with_config(&mut onnx_model, &ConversionConfig::default())
                .context("Failed to convert ONNX model to MIL IR")?;

            if is_text {
                let op_count = count_ops(&result.program);
                eprintln!(
                    "✓ Converted: {} functions, {} ops",
                    result.program.functions.len(),
                    op_count
                );

                if !result.warnings.is_empty() {
                    eprintln!(
                        "⚠ {} unsupported op(s) during conversion: {:?}",
                        result.warnings.len(),
                        result.warnings
                    );
                }

                eprintln!();
            }

            validate_ane_compatibility(&result.program)
        }
        "mlmodel" => {
            let model = read_mlmodel(input_path)
                .with_context(|| format!("Failed to read mlmodel: {input}"))?;
            if is_text {
                eprintln!("✓ CoreML model (.mlmodel) parsed successfully");
            }

            match ironmill_compile::mil::model_to_program(&model) {
                Ok(program) => {
                    if is_text {
                        eprintln!();
                    }
                    validate_ane_compatibility(&program)
                }
                Err(e) => {
                    if lenient {
                        eprintln!();
                        eprintln!("Note: Could not convert to MIL IR for ANE analysis: {e}");
                        eprintln!("  ANE validation requires an ML Program model (spec v7+).");
                        return Ok(());
                    } else {
                        eprintln!();
                        eprintln!("Error: Could not convert to MIL IR for ANE analysis: {e}");
                        eprintln!("  ANE validation requires an ML Program model (spec v7+).");
                        eprintln!("  Use --lenient to continue despite this failure.");
                        std::process::exit(1);
                    }
                }
            }
        }
        "mlpackage" => {
            let model = read_mlpackage(input_path)
                .with_context(|| format!("Failed to read mlpackage: {input}"))?;
            if is_text {
                eprintln!("✓ CoreML package (.mlpackage) parsed successfully");
            }

            match ironmill_compile::mil::model_to_program(&model) {
                Ok(program) => {
                    if is_text {
                        eprintln!();
                    }
                    validate_ane_compatibility(&program)
                }
                Err(e) => {
                    if lenient {
                        eprintln!();
                        eprintln!("Note: Could not convert to MIL IR for ANE analysis: {e}");
                        eprintln!("  ANE validation requires an ML Program model (spec v7+).");
                        return Ok(());
                    } else {
                        eprintln!();
                        eprintln!("Error: Could not convert to MIL IR for ANE analysis: {e}");
                        eprintln!("  ANE validation requires an ML Program model (spec v7+).");
                        eprintln!("  Use --lenient to continue despite this failure.");
                        std::process::exit(1);
                    }
                }
            }
        }
        _ => bail!("Unsupported format '.{ext}'. Expected .onnx, .mlmodel, or .mlpackage"),
    };

    match format {
        "text" => print_validation_report(&report),
        "json" => {
            let json = validation_report_to_json(&report)
                .context("Failed to serialize validation report")?;
            println!("{json}");
        }
        _ => bail!("Unsupported --format '{format}'. Expected 'text' or 'json'."),
    }

    Ok(())
}

fn cmd_compile_pipeline(manifest_path: &str, output: Option<&str>) -> Result<()> {
    let manifest_file = Path::new(manifest_path);
    if !manifest_file.exists() {
        bail!("Pipeline manifest not found: {manifest_path}");
    }

    eprintln!("Reading pipeline manifest: {manifest_path}");
    let toml_str = std::fs::read_to_string(manifest_file)
        .with_context(|| format!("Failed to read manifest: {manifest_path}"))?;

    let manifest =
        parse_pipeline_manifest(&toml_str).context("Failed to parse pipeline manifest")?;

    eprintln!(
        "Pipeline '{}': {} stage(s)",
        manifest.pipeline.name,
        manifest.stages.len()
    );
    for stage in &manifest.stages {
        let deps = if stage.depends_on.is_empty() {
            String::new()
        } else {
            format!(" (depends on: {})", stage.depends_on.join(", "))
        };
        let source = stage
            .onnx
            .as_deref()
            .or(stage.safetensors.as_deref())
            .or(stage.gguf.as_deref())
            .unwrap_or("unknown");
        eprintln!(
            "  - {} [{}] quantize={}{deps}",
            stage.name, source, stage.quantize
        );
    }
    eprintln!();

    let base_dir = manifest_file
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let output_dir = PathBuf::from(output.unwrap_or("."));

    let result = convert_pipeline(&manifest, &base_dir, &output_dir)
        .context("Pipeline conversion failed")?;

    // Print per-stage warnings.
    for (stage_name, warnings) in &result.stage_warnings {
        if !warnings.is_empty() {
            eprintln!(
                "Warnings for stage '{stage_name}': {} unsupported op(s): {:?}",
                warnings.len(),
                warnings
            );
        }
    }

    eprintln!(
        "Pipeline manifest written: {}",
        output_dir.join("pipeline.json").display()
    );
    eprintln!(
        "Pipeline conversion complete: {} stage(s) converted.",
        result.manifest.stages.len()
    );
    Ok(())
}

fn cmd_pipeline_report(input: &str, config_a: &Path, config_b: &Path) -> Result<()> {
    let input_path = Path::new(input);
    if !input_path.exists() {
        bail!("Input file not found: {input}");
    }

    let ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext != "onnx" {
        bail!("pipeline-report currently only supports ONNX models, got '.{ext}'");
    }

    eprintln!("Reading ONNX model: {input}");
    let mut onnx_model =
        read_onnx(input_path).with_context(|| format!("Failed to read ONNX model: {input}"))?;

    eprintln!("Converting to MIL IR...");
    let result_a =
        onnx_to_program_with_config(&mut onnx_model.clone(), &ConversionConfig::default())
            .context("Failed to convert ONNX model")?;
    let result_b = onnx_to_program_with_config(&mut onnx_model, &ConversionConfig::default())
        .context("Failed to convert ONNX model")?;
    let mut program_a = result_a.program;
    let mut program_b = result_b.program;

    eprintln!("Loading config A: {}", config_a.display());
    let pipeline_a = PassPipeline::with_config(config_a)
        .with_context(|| format!("Failed to load config A: {}", config_a.display()))?;
    eprintln!("  Passes: {:?}", pipeline_a.pass_names());

    eprintln!("Loading config B: {}", config_b.display());
    let pipeline_b = PassPipeline::with_config(config_b)
        .with_context(|| format!("Failed to load config B: {}", config_b.display()))?;
    eprintln!("  Passes: {:?}", pipeline_b.pass_names());
    eprintln!();

    eprintln!("Running pipeline A...");
    let report_a = pipeline_a
        .run(&mut program_a)
        .context("Pipeline A failed")?;

    eprintln!("Running pipeline B...");
    let report_b = pipeline_b
        .run(&mut program_b)
        .context("Pipeline B failed")?;

    eprintln!();
    println!("{}", PipelineReport::compare(&report_a, &report_b));

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}
