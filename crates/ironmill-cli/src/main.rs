use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use mil_rs::ir::ModelSplitPass;
use mil_rs::ir::PassPipeline;
use mil_rs::ir::PipelineReport;
use mil_rs::reader::{print_model_summary, print_onnx_summary};
use mil_rs::validate::{print_validation_report, validation_report_to_json};
#[allow(unused_imports)]
use mil_rs::{
    ConversionConfig, LossFunction, TemplateOptions, UpdatableModelConfig, UpdateOptimizer,
    compile_model, convert_pipeline, is_compiler_available, onnx_to_program_with_config,
    parse_pipeline_manifest, program_to_model, program_to_multi_function_model,
    program_to_updatable_model, read_mlmodel, read_mlpackage, read_onnx,
    validate_ane_compatibility, weights_to_program, weights_to_program_with_options,
    write_mlpackage,
};

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

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum KvQuantArg {
    /// No KV cache quantization (default)
    #[value(name = "none")]
    None,
    /// TurboQuant INT8 KV cache compression
    #[value(name = "turbo-int8")]
    TurboInt8,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Convert an ONNX model to CoreML format (.mlpackage).
    Compile {
        /// Path to the input model (.onnx, .mlmodel, or .mlpackage).
        input: String,

        /// Path for the output .mlpackage.
        #[arg(short, long)]
        output: Option<String>,

        /// Target compute units: "all", "cpu-only", "cpu-and-ne", "cpu-and-gpu".
        #[arg(short, long, default_value = "all")]
        target: String,

        /// Quantization mode: "none", "fp16", "int8", "mixed-fp16-int8".
        #[arg(short, long, default_value = "none")]
        quantize: String,

        /// Calibration data directory (for int8 quantization).
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

        /// Disable fusion and optimization passes.
        #[arg(long)]
        no_fusion: bool,

        /// Set a concrete shape for a named input (for ANE compatibility).
        /// Format: "name:d0,d1,d2,...". May be repeated.
        #[arg(long = "input-shape", value_name = "NAME:SHAPE")]
        input_shapes: Vec<String>,

        /// Merge LoRA adapter weights into the base model during conversion (default).
        /// LoRA A/B weight pairs found in the ONNX initializers are automatically
        /// detected and merged. Disable with --no-merge-lora.
        #[arg(long = "merge-lora", default_value_t = true, action = clap::ArgAction::SetTrue)]
        merge_lora: bool,

        /// Disable automatic LoRA adapter merging.
        #[arg(long = "no-merge-lora", conflicts_with = "merge_lora")]
        no_merge_lora: bool,

        /// Emit a separate adapter .mlpackage instead of merging LoRA weights.
        #[arg(long = "emit-adapter")]
        emit_adapter: bool,

        /// Path to an external adapter file (e.g. .safetensors). May be repeated.
        #[arg(long = "adapter", value_name = "PATH")]
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

        /// Loss function for on-device training: "cross-entropy" or "mse".
        #[arg(long = "loss-function", default_value = "cross-entropy")]
        loss_function: String,

        /// Optimizer for on-device training: "sgd" or "adam".
        #[arg(long = "optimizer", default_value = "sgd")]
        optimizer_type: String,

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
    },

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
fn count_ops(program: &mil_rs::ir::Program) -> usize {
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
        Commands::Compile {
            input,
            output,
            target,
            quantize,
            cal_data,
            quantize_config,
            palettize,
            polar_quantize,
            no_fusion,
            input_shapes,
            merge_lora,
            no_merge_lora,
            emit_adapter,
            adapters,
            updatable_layers,
            learning_rate,
            epochs,
            loss_function,
            optimizer_type,
            split_draft_layers,
            moe_split,
            moe_bundle,
            moe_fuse_topk,
            pipeline_config,
            annotate_compute_units,
            ane_memory_budget,
            ane,
            runtime,
            kv_quant,
            kv_quant_qjl,
            max_seq_len,
        } => cmd_compile(
            &input,
            CompileOpts {
                output,
                target,
                quantize,
                cal_data,
                quantize_config,
                palettize,
                polar_quantize,
                no_fusion,
                input_shapes,
                merge_lora: merge_lora && !no_merge_lora,
                emit_adapter,
                adapters,
                updatable_layers,
                learning_rate,
                epochs,
                loss_function,
                optimizer_type,
                split_draft_layers,
                moe_split,
                moe_bundle,
                moe_fuse_topk,
                pipeline_config,
                annotate_compute_units,
                ane_memory_budget,
                ane,
                runtime,
                kv_quant,
                kv_quant_qjl,
                max_seq_len,
            },
        ),
        Commands::Inspect { input } => cmd_inspect(&input),
        Commands::Validate { input, format } => cmd_validate(&input, &format),
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
    target: String,
    quantize: String,
    cal_data: Option<PathBuf>,
    quantize_config: Option<PathBuf>,
    palettize: Option<u8>,
    polar_quantize: Option<u8>,
    no_fusion: bool,
    input_shapes: Vec<String>,
    merge_lora: bool,
    emit_adapter: bool,
    adapters: Vec<PathBuf>,
    updatable_layers: Option<String>,
    learning_rate: f64,
    epochs: i64,
    loss_function: String,
    optimizer_type: String,
    split_draft_layers: Option<usize>,
    moe_split: bool,
    moe_bundle: bool,
    moe_fuse_topk: Option<usize>,
    pipeline_config: Option<PathBuf>,
    annotate_compute_units: bool,
    ane_memory_budget: Option<String>,
    #[allow(dead_code)] // Used in compile_from_weights once SafeTensorsProvider lands.
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

    if opts.target != "all" && opts.target != "cpu-and-ne" {
        println!(
            "Note: --target '{}' will be fully supported in Phase 3. Proceeding with default target.",
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
        println!(
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
                println!(
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

fn compile_from_onnx(input_path: &Path, opts: &CompileOpts) -> Result<()> {
    let input_display = input_path.display();

    // 1. Read ONNX model
    println!("Reading ONNX model: {input_display}");
    let onnx_model = read_onnx(input_path)
        .with_context(|| format!("Failed to read ONNX model: {input_display}"))?;
    print_onnx_summary(&onnx_model);

    if opts.merge_lora {
        println!("  LoRA merge: enabled (use --no-merge-lora to disable)");
    }
    if opts.emit_adapter {
        println!("  Note: --emit-adapter is reserved for future adapter export support.");
    }
    if !opts.adapters.is_empty() {
        println!(
            "  Note: --adapter is reserved for future external adapter loading ({} path(s) provided).",
            opts.adapters.len()
        );
    }
    println!();

    // 2. Convert to MIL IR
    println!("Converting to CoreML MIL IR...");
    let config = ConversionConfig {
        merge_lora: opts.merge_lora,
        model_dir: input_path.parent().map(|p| p.to_path_buf()),
    };
    let result = onnx_to_program_with_config(&onnx_model, &config)
        .context("Failed to convert ONNX model to MIL IR")?;
    let mut program = result.program;
    let warnings = result.warnings;
    println!(
        "  Converted: {} functions, {} ops",
        program.functions.len(),
        count_ops(&program)
    );
    println!();

    // 3. Build the pass pipeline
    let pipeline = if let Some(ref config_path) = opts.pipeline_config {
        PassPipeline::with_config(config_path)
            .with_context(|| format!("Failed to load pipeline config: {}", config_path.display()))?
    } else {
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
            pipeline = pipeline
                .with_mixed_precision(config_path)
                .context("Failed to configure mixed-precision from config file")?;
        } else {
            match opts.quantize.as_str() {
                "fp16" => {
                    pipeline = pipeline
                        .with_fp16()
                        .context("Failed to configure FP16 quantization")?;
                }
                "int8" => {
                    pipeline = pipeline
                        .with_int8(opts.cal_data.clone())
                        .context("Failed to configure INT8 quantization")?;
                }
                "mixed-fp16-int8" => {
                    let config = mil_rs::MixedPrecisionConfig::preset_fp16_int8();
                    pipeline = pipeline
                        .with_mixed_precision_config(config)
                        .context("Failed to configure mixed-precision quantization")?;
                }
                "none" => {}
                other => {
                    bail!(
                        "Unsupported quantization mode: '{other}'. Expected 'none', 'fp16', 'int8', or 'mixed-fp16-int8'."
                    )
                }
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

        // Add compute unit annotations (should run last, after all transforms).
        if opts.annotate_compute_units {
            pipeline = pipeline.with_compute_unit_annotations();
        }

        // Add op splitting for ANE memory budget
        if let Some(ref budget_str) = opts.ane_memory_budget {
            let budget = mil_rs::ir::passes::op_split::parse_memory_size(budget_str)
                .map_err(|e| anyhow::anyhow!("invalid --ane-memory-budget: {e}"))?;
            pipeline = pipeline.with_op_splitting(budget);
            println!("  ANE memory budget: {budget_str} ({budget} bytes)");
        }

        pipeline
    };

    // 4. Run the pipeline
    println!("Running optimization passes...");
    let report = pipeline
        .run(&mut program)
        .context("Optimization pipeline failed")?;

    for result in &report.pass_results {
        if result.ops_after < result.ops_before {
            let diff = result.ops_before - result.ops_after;
            let label = if diff == 1 { "op" } else { "ops" };
            println!(
                "  {}: removed {diff} {label} ({:.2?})",
                result.name, result.elapsed
            );
        } else if result.ops_after > result.ops_before {
            let diff = result.ops_after - result.ops_before;
            let label = if diff == 1 { "op" } else { "ops" };
            println!(
                "  {}: added {diff} {label} ({:.2?})",
                result.name, result.elapsed
            );
        } else {
            println!("  {}: no changes ({:.2?})", result.name, result.elapsed);
        }
    }
    println!("  Total pipeline time: {:.2?}", report.total_elapsed());
    println!();

    // 5. MoE split (if requested)
    if opts.moe_split {
        use mil_rs::{detect_moe, split_moe};

        if let Some(topology) = detect_moe(&program) {
            println!(
                "MoE architecture detected: {} experts",
                topology.expert_count
            );
            let split_result = split_moe(&program, &topology);

            let stem = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");

            // Write shared program
            let shared_model = program_to_model(&split_result.shared, 9)
                .context("Failed to convert shared program to CoreML")?;
            let shared_path = format!("{stem}-shared.mlpackage");
            println!("  Writing shared layers: {shared_path}");
            write_mlpackage(&shared_model, &shared_path)
                .with_context(|| format!("Failed to write {shared_path}"))?;

            // Write per-expert programs
            for (i, expert) in split_result.experts.iter().enumerate() {
                let expert_model = program_to_model(expert, 9)
                    .with_context(|| format!("Failed to convert expert {i} to CoreML"))?;
                let expert_path = format!("{stem}-expert-{i}.mlpackage");
                println!("  Writing expert {i}: {expert_path}");
                write_mlpackage(&expert_model, &expert_path)
                    .with_context(|| format!("Failed to write {expert_path}"))?;
            }

            // Write manifest
            let manifest_path = format!("{stem}-manifest.json");
            let manifest_json = serde_json::to_string_pretty(&split_result.manifest)
                .context("Failed to serialize MoE manifest")?;
            std::fs::write(&manifest_path, &manifest_json)
                .with_context(|| format!("Failed to write {manifest_path}"))?;
            println!("  Wrote manifest: {manifest_path}");

            println!();
            println!("MoE split complete.");
            return Ok(());
        } else {
            println!(
                "Note: --moe-split specified but no MoE pattern detected. Producing single output."
            );
        }
    }

    // 5b. MoE bundle (if requested) — single multi-function .mlpackage
    if opts.moe_bundle {
        use mil_rs::{detect_moe, split_moe};

        if let Some(topology) = detect_moe(&program) {
            println!(
                "MoE architecture detected: {} experts",
                topology.expert_count
            );
            let split_result = split_moe(&program, &topology);

            let stem = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");

            // Bundle all experts as functions in a single model.
            let bundle_model = program_to_multi_function_model(&split_result, 9)
                .context("Failed to convert MoE programs to multi-function CoreML model")?;

            let bundle_path = opts
                .output
                .clone()
                .unwrap_or_else(|| format!("{stem}-bundle.mlpackage"));
            println!(
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
            println!("  Wrote manifest: {manifest_path}");

            compile_output(&bundle_path);

            println!();
            println!("MoE bundle complete.");
            return Ok(());
        } else {
            println!(
                "Note: --moe-bundle specified but no MoE pattern detected. Producing single output."
            );
        }
    }

    // 5c. MoE top-K fusion (if requested)
    if let Some(k) = opts.moe_fuse_topk {
        use mil_rs::{ExpertFrequencyProfile, fuse_top_k_experts};

        // Load or build frequency profile
        let profile = if let Some(ref cal_dir) = opts.cal_data {
            let profile_path = cal_dir.join("expert_profile.json");
            if profile_path.exists() {
                let json = std::fs::read_to_string(&profile_path)
                    .with_context(|| format!("Failed to read {}", profile_path.display()))?;
                ExpertFrequencyProfile::from_json(&json)
                    .with_context(|| format!("Failed to parse {}", profile_path.display()))?
            } else {
                println!(
                    "  Note: {} not found, using uniform profile.",
                    profile_path.display()
                );
                // Detect expert count to build uniform profile
                let expert_count = mil_rs::detect_moe(&program)
                    .map(|t| t.expert_count)
                    .unwrap_or(k);
                ExpertFrequencyProfile::uniform(expert_count)
            }
        } else {
            println!("  Note: --cal-data not provided, using uniform profile.");
            let expert_count = mil_rs::detect_moe(&program)
                .map(|t| t.expert_count)
                .unwrap_or(k);
            ExpertFrequencyProfile::uniform(expert_count)
        };

        if let Some(fuse_result) = fuse_top_k_experts(&program, k, &profile) {
            println!(
                "MoE top-{k} fusion: kept experts {:?}, discarded {:?}",
                fuse_result.kept_expert_indices, fuse_result.discarded_expert_indices
            );
            println!(
                "  Fused program: {} ops (was {} ops)",
                count_ops(&fuse_result.program),
                count_ops(&program)
            );

            let stem = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");

            let output_path = opts
                .output
                .clone()
                .unwrap_or_else(|| format!("{stem}-fused.mlpackage"));
            let model = program_to_model(&fuse_result.program, 9)
                .context("Failed to convert fused program to CoreML")?;
            println!("  Writing fused model: {output_path}");
            write_mlpackage(&model, &output_path)
                .with_context(|| format!("Failed to write {output_path}"))?;

            compile_output(&output_path);

            println!();
            println!("MoE top-{k} fusion complete.");
            return Ok(());
        } else {
            println!(
                "Note: --moe-fuse-topk specified but no MoE pattern detected. Producing single output."
            );
        }
    }

    // 6. Handle model splitting for speculative decoding, or normal single-model output.
    if let Some(n_layers) = opts.split_draft_layers {
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");

        let split_pass = ModelSplitPass::new(n_layers);
        let split_result = split_pass
            .split(&program)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("Failed to split model for speculative decoding")?;

        // Write draft model
        let draft_path = opts
            .output
            .as_deref()
            .map(|o| {
                let p = Path::new(o);
                let s = p.file_stem().and_then(|s| s.to_str()).unwrap_or(stem);
                let parent = p.parent().unwrap_or(Path::new("."));
                parent
                    .join(format!("{s}-draft.mlpackage"))
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_else(|| format!("{stem}-draft.mlpackage"));

        let draft_model = program_to_model(&split_result.draft, 9)
            .context("Failed to convert draft MIL IR to CoreML protobuf")?;
        println!("Writing draft model ({n_layers} layers): {draft_path}");
        write_mlpackage(&draft_model, &draft_path)
            .with_context(|| format!("Failed to write draft mlpackage: {draft_path}"))?;

        // Write verifier model
        let verifier_path = opts
            .output
            .as_deref()
            .map(|o| {
                let p = Path::new(o);
                let s = p.file_stem().and_then(|s| s.to_str()).unwrap_or(stem);
                let parent = p.parent().unwrap_or(Path::new("."));
                parent
                    .join(format!("{s}-verifier.mlpackage"))
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_else(|| format!("{stem}-verifier.mlpackage"));

        let verifier_model = program_to_model(&split_result.verifier, 9)
            .context("Failed to convert verifier MIL IR to CoreML protobuf")?;
        println!("Writing verifier model (full): {verifier_path}");
        write_mlpackage(&verifier_model, &verifier_path)
            .with_context(|| format!("Failed to write verifier mlpackage: {verifier_path}"))?;

        // Compile both
        compile_output(&draft_path);
        compile_output(&verifier_path);
    } else {
        // Normal single-model output.
        match opts.runtime {
            RuntimeArg::CoreMl => {
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

                    let loss_fn = match opts.loss_function.as_str() {
                        "mse" | "mean-squared-error" => LossFunction::MeanSquaredError,
                        _ => LossFunction::CategoricalCrossEntropy,
                    };
                    let opt = match opts.optimizer_type.as_str() {
                        "adam" => UpdateOptimizer::Adam,
                        _ => UpdateOptimizer::Sgd,
                    };

                    let config = UpdatableModelConfig {
                        updatable_layers: layer_names.clone(),
                        learning_rate: opts.learning_rate,
                        epochs: opts.epochs,
                        loss_function: loss_fn,
                        optimizer: opt,
                    };

                    println!(
                        "Updatable model: {} layer(s) marked for on-device training",
                        layer_names.len()
                    );
                    program_to_updatable_model(&program, 9, &config)
                        .context("Failed to convert MIL IR to updatable CoreML protobuf")?
                } else {
                    program_to_model(&program, 9)
                        .context("Failed to convert MIL IR to CoreML protobuf")?
                };

                let output_path = opts.output.clone().unwrap_or_else(|| {
                    let stem = input_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("model");
                    format!("{stem}.mlpackage")
                });
                println!("Writing CoreML package: {output_path}");
                write_mlpackage(&model, &output_path)
                    .with_context(|| format!("Failed to write mlpackage: {output_path}"))?;

                compile_output(&output_path);
            }
            RuntimeArg::AneDirect => {
                #[cfg(feature = "ane-direct")]
                {
                    use ironmill_ane::CompiledArtifacts;

                    let output_dir = opts.output.clone().unwrap_or_else(|| {
                        let stem = input_path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("model");
                        format!("{stem}.ane")
                    });

                    println!("Compiling for ANE direct runtime → {output_dir}");

                    let artifacts = CompiledArtifacts::prepare(&program)
                        .context("ANE direct compilation failed")?;

                    let warnings = artifacts
                        .validate()
                        .context("ANE artifact validation failed")?;
                    for w in &warnings {
                        println!("  ⚠ {w}");
                    }

                    artifacts
                        .save(std::path::Path::new(&output_dir))
                        .context("failed to save ANE artifacts")?;

                    println!(
                        "  ✓ {} sub-program(s) written to {output_dir}",
                        artifacts.num_sub_programs(),
                    );
                }
                #[cfg(not(feature = "ane-direct"))]
                {
                    anyhow::bail!("ANE direct runtime requires --features ane-direct");
                }
            }
        }
    }

    // 9. Print warnings
    if !warnings.is_empty() {
        println!();
        println!("Warnings:");
        let unsupported: Vec<&str> = warnings.iter().map(|w| w.as_str()).collect();
        println!(
            "  {} unsupported op(s) skipped: {:?}",
            unsupported.len(),
            unsupported
        );
    }

    println!();
    println!("Conversion complete.");
    Ok(())
}

/// Compile a model from SafeTensors or GGUF weight files.
fn compile_from_weights(input_path: &Path, _opts: &CompileOpts) -> Result<()> {
    let input_display = input_path.display();

    let is_gguf = input_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e == "gguf");

    if is_gguf {
        bail!("GGUF format support is not yet implemented. Please use SafeTensors format.");
    }

    // 1. Load weights
    println!("Loading weights from: {input_display}");
    println!("  Note: SafeTensors provider is a stub — using template for graph construction.");
    println!();

    // TODO(milestone-2): Replace this stub with real SafeTensorsProvider.
    // For now, we construct the program using the template which will
    // emit warnings for any missing weight tensors.
    bail!(
        "SafeTensors weight loading is not yet implemented (Milestone 2). \
         The architecture template and CLI routing are ready — \
         provide a SafeTensorsProvider implementation to enable end-to-end conversion."
    );

    // When SafeTensorsProvider is implemented, the flow will be:
    //
    //   let provider = SafeTensorsProvider::load(input_path)?;
    //
    //   // Derive ANE template options from CLI flags.
    //   let ane_mode = _opts.ane
    //       || matches!(_opts.runtime, RuntimeArg::AneDirect);
    //   let template_opts = TemplateOptions { ane: ane_mode };
    //
    //   let result = weights_to_program_with_options(&provider, &template_opts)?;
    //   let mut program = result.program;
    //   let warnings = result.warnings;
    //
    //   // Build and run the same pass pipeline as ONNX:
    //   let pipeline = build_pass_pipeline(opts)?;
    //   let report = pipeline.run(&mut program)?;
    //   // ... write output ...
}

/// Compile a .mlpackage using xcrun coremlcompiler.
fn compile_output(output_path: &str) {
    if is_compiler_available() {
        println!("Compiling with xcrun coremlcompiler...");
        let output_dir = Path::new(output_path).parent().unwrap_or(Path::new("."));
        match compile_model(output_path, output_dir) {
            Ok(compiled) => println!("Done: {}", compiled.display()),
            Err(e) => println!("Warning: compilation failed: {e}"),
        }
    } else {
        println!("Note: xcrun coremlcompiler not found — skipping compilation step.");
        println!("  The .mlpackage can be compiled on a Mac with Xcode installed.");
    }
}

fn compile_coreml(input_path: &Path, ext: &str) -> Result<()> {
    let input_display = input_path.display();
    println!("Input is already CoreML format (.{ext}): {input_display}");

    if !is_compiler_available() {
        bail!("xcrun coremlcompiler is not available. Install Xcode to compile CoreML models.");
    }

    let output_dir = input_path.parent().unwrap_or(Path::new("."));
    println!("Compiling with xcrun coremlcompiler...");
    let compiled = compile_model(input_path, output_dir)
        .with_context(|| format!("Failed to compile {input_display}"))?;
    println!("Done: {}", compiled.display());

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

fn cmd_validate(input: &str, format: &str) -> Result<()> {
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
        println!("Validating: {input}");
        println!();
    }

    let report = match ext.as_str() {
        "onnx" => {
            let onnx_model = read_onnx(input_path)
                .with_context(|| format!("Failed to read ONNX model: {input}"))?;
            if is_text {
                println!("✓ ONNX model parsed successfully");
                println!("  Converting to MIL IR...");
            }

            let result = onnx_to_program_with_config(&onnx_model, &ConversionConfig::default())
                .context("Failed to convert ONNX model to MIL IR")?;

            if is_text {
                let op_count = count_ops(&result.program);
                println!(
                    "✓ Converted: {} functions, {} ops",
                    result.program.functions.len(),
                    op_count
                );

                if !result.warnings.is_empty() {
                    println!(
                        "⚠ {} unsupported op(s) during conversion: {:?}",
                        result.warnings.len(),
                        result.warnings
                    );
                }

                println!();
            }

            validate_ane_compatibility(&result.program)
        }
        "mlmodel" => {
            let model = read_mlmodel(input_path)
                .with_context(|| format!("Failed to read mlmodel: {input}"))?;
            if is_text {
                println!("✓ CoreML model (.mlmodel) parsed successfully");
            }

            match mil_rs::model_to_program(&model) {
                Ok(program) => {
                    if is_text {
                        println!();
                    }
                    validate_ane_compatibility(&program)
                }
                Err(e) => {
                    if is_text {
                        println!();
                        println!("Note: Could not convert to MIL IR for ANE analysis: {e}");
                        println!("  ANE validation requires an ML Program model (spec v7+).");
                    }
                    return Ok(());
                }
            }
        }
        "mlpackage" => {
            let model = read_mlpackage(input_path)
                .with_context(|| format!("Failed to read mlpackage: {input}"))?;
            if is_text {
                println!("✓ CoreML package (.mlpackage) parsed successfully");
            }

            match mil_rs::model_to_program(&model) {
                Ok(program) => {
                    if is_text {
                        println!();
                    }
                    validate_ane_compatibility(&program)
                }
                Err(e) => {
                    if is_text {
                        println!();
                        println!("Note: Could not convert to MIL IR for ANE analysis: {e}");
                        println!("  ANE validation requires an ML Program model (spec v7+).");
                    }
                    return Ok(());
                }
            }
        }
        _ => bail!("Unsupported format '.{ext}'. Expected .onnx, .mlmodel, or .mlpackage"),
    };

    match format {
        "text" => print_validation_report(&report),
        "json" => {
            let json = validation_report_to_json(&report);
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

    println!("Reading pipeline manifest: {manifest_path}");
    let toml_str = std::fs::read_to_string(manifest_file)
        .with_context(|| format!("Failed to read manifest: {manifest_path}"))?;

    let manifest =
        parse_pipeline_manifest(&toml_str).context("Failed to parse pipeline manifest")?;

    println!(
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
        println!(
            "  - {} [{}] quantize={}{deps}",
            stage.name, source, stage.quantize
        );
    }
    println!();

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
            println!(
                "Warnings for stage '{stage_name}': {} unsupported op(s): {:?}",
                warnings.len(),
                warnings
            );
        }
    }

    println!(
        "Pipeline manifest written: {}",
        output_dir.join("pipeline.json").display()
    );
    println!(
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

    println!("Reading ONNX model: {input}");
    let onnx_model =
        read_onnx(input_path).with_context(|| format!("Failed to read ONNX model: {input}"))?;

    println!("Converting to MIL IR...");
    let result_a = onnx_to_program_with_config(&onnx_model, &ConversionConfig::default())
        .context("Failed to convert ONNX model")?;
    let result_b = onnx_to_program_with_config(&onnx_model, &ConversionConfig::default())
        .context("Failed to convert ONNX model")?;
    let mut program_a = result_a.program;
    let mut program_b = result_b.program;

    println!("Loading config A: {}", config_a.display());
    let pipeline_a = PassPipeline::with_config(config_a)
        .with_context(|| format!("Failed to load config A: {}", config_a.display()))?;
    println!("  Passes: {:?}", pipeline_a.pass_names());

    println!("Loading config B: {}", config_b.display());
    let pipeline_b = PassPipeline::with_config(config_b)
        .with_context(|| format!("Failed to load config B: {}", config_b.display()))?;
    println!("  Passes: {:?}", pipeline_b.pass_names());
    println!();

    println!("Running pipeline A...");
    let report_a = pipeline_a
        .run(&mut program_a)
        .context("Pipeline A failed")?;

    println!("Running pipeline B...");
    let report_b = pipeline_b
        .run(&mut program_b)
        .context("Pipeline B failed")?;

    println!();
    println!("{}", PipelineReport::compare(&report_a, &report_b));

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}
