use std::path::Path;
use std::process;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use mil_rs::ir::{
    ConstantFoldPass, DeadCodeEliminationPass, IdentityEliminationPass, Pass,
    ShapeMaterializePass,
};
use mil_rs::validate::print_validation_report;
use mil_rs::{
    compile_model, is_compiler_available, onnx_to_program, program_to_model, read_mlmodel,
    read_mlpackage, read_onnx, validate_ane_compatibility, write_mlpackage,
};
use mil_rs::reader::{print_model_summary, print_onnx_summary};

/// coreml-kit — Convert and optimize ML models for Apple's Neural Engine.
///
/// A Rust-native alternative to Python's coremltools. Converts ONNX models
/// to CoreML format with ANE-targeted optimizations, no Python required.
#[derive(Parser)]
#[command(name = "coreml-kit", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
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

        /// Set a concrete shape for a named input (for ANE compatibility).
        /// Format: "name:d0,d1,d2,...". May be repeated.
        #[arg(long = "input-shape", value_name = "NAME:SHAPE")]
        input_shapes: Vec<String>,
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
            input_shapes,
        } => cmd_compile(&input, output, &target, &quantize, &input_shapes),
        Commands::Inspect { input } => cmd_inspect(&input),
        Commands::Validate { input } => cmd_validate(&input),
    }
}

/// Parse an `--input-shape` value like `"input:1,3,224,224"` into `(name, dims)`.
fn parse_input_shape(s: &str) -> Result<(String, Vec<usize>)> {
    let (name, dims_str) = s
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("invalid --input-shape format: expected 'name:d0,d1,...', got '{s}'"))?;
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

fn cmd_compile(input: &str, output: Option<String>, target: &str, quantize: &str, input_shapes: &[String]) -> Result<()> {
    let input_path = Path::new(input);
    if !input_path.exists() {
        bail!("Input file not found: {input}");
    }

    let ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if target != "all" {
        println!("Note: --target '{target}' will be fully supported in Phase 3. Proceeding with default target.");
    }
    if quantize != "none" {
        println!("Note: --quantize '{quantize}' will be fully supported in Phase 3. Proceeding without quantization.");
    }

    match ext.as_str() {
        "onnx" => compile_from_onnx(input_path, output, input_shapes),
        "mlmodel" | "mlpackage" => compile_coreml(input_path, &ext),
        _ => bail!(
            "Unsupported input format '.{ext}'. Expected .onnx, .mlmodel, or .mlpackage"
        ),
    }
}

fn compile_from_onnx(input_path: &Path, output: Option<String>, input_shapes: &[String]) -> Result<()> {
    let input_display = input_path.display();

    // 1. Read ONNX model
    println!("Reading ONNX model: {input_display}");
    let onnx_model =
        read_onnx(input_path).with_context(|| format!("Failed to read ONNX model: {input_display}"))?;
    print_onnx_summary(&onnx_model);
    println!();

    // 2. Convert to MIL IR
    println!("Converting to CoreML MIL IR...");
    let result =
        onnx_to_program(&onnx_model).context("Failed to convert ONNX model to MIL IR")?;
    let mut program = result.program;
    let warnings = result.warnings;
    println!(
        "  Converted: {} functions, {} ops",
        program.functions.len(),
        count_ops(&program)
    );
    println!();

    // 3. Run shape materialization if --input-shape was provided.
    if !input_shapes.is_empty() {
        let mut shape_pass = ShapeMaterializePass::new();
        for raw in input_shapes {
            let (name, dims) = parse_input_shape(raw)?;
            shape_pass = shape_pass.with_shape(name, dims);
        }
        println!("Materializing input shapes...");
        shape_pass
            .run(&mut program)
            .context("Shape materialization failed")?;
        println!("  shape-materialization: done");
        println!();
    }

    // 4. Run optimization passes
    println!("Running optimization passes...");
    let passes: Vec<Box<dyn Pass>> = vec![
        Box::new(DeadCodeEliminationPass),
        Box::new(IdentityEliminationPass),
        Box::new(ConstantFoldPass),
    ];

    for pass in &passes {
        let before = count_ops(&program);
        pass.run(&mut program)
            .with_context(|| format!("Optimization pass '{}' failed", pass.name()))?;
        let after = count_ops(&program);
        let diff = before.saturating_sub(after);
        if diff > 0 {
            let label = if diff == 1 { "op" } else { "ops" };
            println!("  {}: removed {diff} {label}", pass.name());
        } else {
            println!("  {}: no changes", pass.name());
        }
    }
    println!();

    // 5. Convert IR to proto
    let model = program_to_model(&program, 7)
        .context("Failed to convert MIL IR to CoreML protobuf")?;

    // 6. Write as .mlpackage
    let output_path = output.unwrap_or_else(|| {
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        format!("{stem}.mlpackage")
    });
    println!("Writing CoreML package: {output_path}");
    write_mlpackage(&model, &output_path)
        .with_context(|| format!("Failed to write mlpackage: {output_path}"))?;

    // 7. Optionally compile with xcrun
    if is_compiler_available() {
        println!("Compiling with xcrun coremlcompiler...");
        let output_dir = Path::new(&output_path)
            .parent()
            .unwrap_or(Path::new("."));
        match compile_model(&output_path, output_dir) {
            Ok(compiled) => println!("Done: {}", compiled.display()),
            Err(e) => println!("Warning: compilation failed: {e}"),
        }
    } else {
        println!("Note: xcrun coremlcompiler not found — skipping compilation step.");
        println!("  The .mlpackage can be compiled on a Mac with Xcode installed.");
    }

    // 8. Print warnings
    if !warnings.is_empty() {
        println!();
        println!("Warnings:");
        let unsupported: Vec<&str> = warnings
            .iter()
            .map(|w| w.as_str())
            .collect();
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

fn compile_coreml(input_path: &Path, ext: &str) -> Result<()> {
    let input_display = input_path.display();
    println!("Input is already CoreML format (.{ext}): {input_display}");

    if !is_compiler_available() {
        bail!(
            "xcrun coremlcompiler is not available. Install Xcode to compile CoreML models."
        );
    }

    println!("Compiling with xcrun coremlcompiler...");
    let output_dir = input_path.parent().unwrap_or(Path::new("."));
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
        _ => bail!(
            "Unsupported format '.{ext}'. Expected .onnx, .mlmodel, or .mlpackage"
        ),
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
            let result = onnx_to_program(&onnx_model)
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
                    println!(
                        "Note: Could not convert to MIL IR for ANE analysis: {e}"
                    );
                    println!(
                        "  ANE validation requires an ML Program model (spec v7+)."
                    );
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
                    println!(
                        "Note: Could not convert to MIL IR for ANE analysis: {e}"
                    );
                    println!(
                        "  ANE validation requires an ML Program model (spec v7+)."
                    );
                }
            }
        }
        _ => bail!(
            "Unsupported format '.{ext}'. Expected .onnx, .mlmodel, or .mlpackage"
        ),
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}
