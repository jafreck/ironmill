use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use mil_rs::compiler::Backend;
use mil_rs::ir::PassPipeline;
use mil_rs::reader::{print_model_summary, print_onnx_summary};
use mil_rs::validate::print_validation_report;
use mil_rs::{
    ConversionConfig, LossFunction, UpdatableModelConfig, UpdateOptimizer, compile_model,
    compile_model_with_backend, is_compiler_available, onnx_to_program_with_config,
    program_to_model, program_to_updatable_model, read_mlmodel, read_mlpackage, read_onnx,
    validate_ane_compatibility, write_mlpackage,
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

/// Compilation backend selection for the CLI.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendArg {
    /// Use `xcrun coremlcompiler` (default, stable).
    Xcrun,
    /// Use direct ANE FFI (experimental, requires `ane-direct` feature).
    AneDirect,
}

impl From<BackendArg> for Backend {
    fn from(arg: BackendArg) -> Self {
        match arg {
            BackendArg::Xcrun => Backend::Xcrun,
            BackendArg::AneDirect => Backend::AneDirect,
        }
    }
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

        /// Quantization mode: "none", "fp16", "int8".
        #[arg(short, long, default_value = "none")]
        quantize: String,

        /// Calibration data directory (for int8 quantization).
        #[arg(long = "cal-data", value_name = "DIR")]
        cal_data: Option<PathBuf>,

        /// Weight palettization bit-width (2, 4, 6, or 8).
        #[arg(long, value_name = "BITS")]
        palettize: Option<u8>,

        /// Disable fusion and optimization passes.
        #[arg(long)]
        no_fusion: bool,

        /// Set a concrete shape for a named input (for ANE compatibility).
        /// Format: "name:d0,d1,d2,...". May be repeated.
        #[arg(long = "input-shape", value_name = "NAME:SHAPE")]
        input_shapes: Vec<String>,

        /// Compilation backend: "xcrun" (default) or "ane-direct" (experimental).
        ///
        /// The "ane-direct" backend calls Apple's private _ANECompiler class
        /// directly via FFI, bypassing xcrun. Requires the `ane-direct`
        /// feature flag at build time.
        #[arg(long, value_enum, default_value_t = BackendArg::Xcrun)]
        backend: BackendArg,

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

fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            input,
            output,
            target,
            quantize,
            cal_data,
            palettize,
            no_fusion,
            input_shapes,
            backend,
            merge_lora,
            no_merge_lora,
            emit_adapter,
            adapters,
            updatable_layers,
            learning_rate,
            epochs,
            loss_function,
            optimizer_type,
        } => cmd_compile(
            &input,
            CompileOpts {
                output,
                target,
                quantize,
                cal_data,
                palettize,
                no_fusion,
                input_shapes,
                backend: backend.into(),
                merge_lora: merge_lora && !no_merge_lora,
                emit_adapter,
                adapters,
                updatable_layers,
                learning_rate,
                epochs,
                loss_function,
                optimizer_type,
            },
        ),
        Commands::Inspect { input } => cmd_inspect(&input),
        Commands::Validate { input } => cmd_validate(&input),
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
    palettize: Option<u8>,
    no_fusion: bool,
    input_shapes: Vec<String>,
    backend: Backend,
    merge_lora: bool,
    emit_adapter: bool,
    adapters: Vec<PathBuf>,
    updatable_layers: Option<String>,
    learning_rate: f64,
    epochs: i64,
    loss_function: String,
    optimizer_type: String,
}

fn cmd_compile(input: &str, opts: CompileOpts) -> Result<()> {
    let input_path = Path::new(input);
    if !input_path.exists() {
        bail!("Input file not found: {input}");
    }

    let ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if opts.target != "all" {
        println!(
            "Note: --target '{}' will be fully supported in Phase 3. Proceeding with default target.",
            opts.target
        );
    }

    match ext.as_str() {
        "onnx" => compile_from_onnx(input_path, &opts),
        "mlmodel" | "mlpackage" => compile_coreml(input_path, &ext, opts.backend),
        _ => bail!("Unsupported input format '.{ext}'. Expected .onnx, .mlmodel, or .mlpackage"),
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

    // Add quantization
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
        "none" => {}
        other => {
            bail!("Unsupported quantization mode: '{other}'. Expected 'none', 'fp16', or 'int8'.")
        }
    }

    // Add palettization
    if let Some(bits) = opts.palettize {
        pipeline = pipeline
            .with_palettize(bits)
            .context("Failed to configure palettization")?;
    }

    // 4. Run the pipeline
    println!("Running optimization passes...");
    let report = pipeline
        .run(&mut program)
        .context("Optimization pipeline failed")?;

    for result in &report.pass_results {
        let diff = result.ops_before.saturating_sub(result.ops_after);
        if diff > 0 {
            let label = if diff == 1 { "op" } else { "ops" };
            println!("  {}: removed {diff} {label}", result.name);
        } else {
            println!("  {}: no changes", result.name);
        }
    }
    println!();

    // 6. Convert IR to proto
    let model = if let Some(ref layers) = opts.updatable_layers {
        let layer_names: Vec<String> = layers.split(',').map(|s| s.trim().to_string()).collect();

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
        program_to_model(&program, 9).context("Failed to convert MIL IR to CoreML protobuf")?
    };

    // 7. Write as .mlpackage
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

    // 8. Compile the model
    match opts.backend {
        Backend::AneDirect => {
            println!("Compiling with ANE direct backend...");
            let output_dir = Path::new(&output_path).parent().unwrap_or(Path::new("."));
            match compile_model_with_backend(&output_path, output_dir, Backend::AneDirect) {
                Ok(compiled) => println!("Done: {}", compiled.display()),
                Err(e) => println!("Warning: ANE direct compilation failed: {e}"),
            }
        }
        Backend::Xcrun => {
            if is_compiler_available() {
                println!("Compiling with xcrun coremlcompiler...");
                let output_dir = Path::new(&output_path).parent().unwrap_or(Path::new("."));
                match compile_model(&output_path, output_dir) {
                    Ok(compiled) => println!("Done: {}", compiled.display()),
                    Err(e) => println!("Warning: compilation failed: {e}"),
                }
            } else {
                println!("Note: xcrun coremlcompiler not found — skipping compilation step.");
                println!("  The .mlpackage can be compiled on a Mac with Xcode installed.");
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

fn compile_coreml(input_path: &Path, ext: &str, backend: Backend) -> Result<()> {
    let input_display = input_path.display();
    println!("Input is already CoreML format (.{ext}): {input_display}");

    let output_dir = input_path.parent().unwrap_or(Path::new("."));

    match backend {
        Backend::AneDirect => {
            println!("Compiling with ANE direct backend...");
            let compiled = compile_model_with_backend(input_path, output_dir, Backend::AneDirect)
                .with_context(|| format!("Failed to compile {input_display}"))?;
            println!("Done: {}", compiled.display());
        }
        Backend::Xcrun => {
            if !is_compiler_available() {
                bail!(
                    "xcrun coremlcompiler is not available. Install Xcode to compile CoreML models."
                );
            }
            println!("Compiling with xcrun coremlcompiler...");
            let compiled = compile_model(input_path, output_dir)
                .with_context(|| format!("Failed to compile {input_display}"))?;
            println!("Done: {}", compiled.display());
        }
    }

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

fn cmd_validate(input: &str) -> Result<()> {
    let input_path = Path::new(input);
    if !input_path.exists() {
        bail!("Input file not found: {input}");
    }

    let ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    println!("Validating: {input}");
    println!();

    match ext.as_str() {
        "onnx" => {
            let onnx_model = read_onnx(input_path)
                .with_context(|| format!("Failed to read ONNX model: {input}"))?;
            println!("✓ ONNX model parsed successfully");

            println!("  Converting to MIL IR...");
            let result = onnx_to_program_with_config(&onnx_model, &ConversionConfig::default())
                .context("Failed to convert ONNX model to MIL IR")?;

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
            let report = validate_ane_compatibility(&result.program);
            print_validation_report(&report);
        }
        "mlmodel" => {
            let model = read_mlmodel(input_path)
                .with_context(|| format!("Failed to read mlmodel: {input}"))?;
            println!("✓ CoreML model (.mlmodel) parsed successfully");

            match mil_rs::model_to_program(&model) {
                Ok(program) => {
                    println!();
                    let report = validate_ane_compatibility(&program);
                    print_validation_report(&report);
                }
                Err(e) => {
                    println!();
                    println!("Note: Could not convert to MIL IR for ANE analysis: {e}");
                    println!("  ANE validation requires an ML Program model (spec v7+).");
                }
            }
        }
        "mlpackage" => {
            let model = read_mlpackage(input_path)
                .with_context(|| format!("Failed to read mlpackage: {input}"))?;
            println!("✓ CoreML package (.mlpackage) parsed successfully");

            match mil_rs::model_to_program(&model) {
                Ok(program) => {
                    println!();
                    let report = validate_ane_compatibility(&program);
                    print_validation_report(&report);
                }
                Err(e) => {
                    println!();
                    println!("Note: Could not convert to MIL IR for ANE analysis: {e}");
                    println!("  ANE validation requires an ML Program model (spec v7+).");
                }
            }
        }
        _ => bail!("Unsupported format '.{ext}'. Expected .onnx, .mlmodel, or .mlpackage"),
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}
