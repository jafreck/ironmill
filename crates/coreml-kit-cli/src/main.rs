use anyhow::Result;
use clap::{Parser, Subcommand};

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
        /// Path to the input ONNX model.
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
    },

    /// Inspect a CoreML model and show its structure.
    Inspect {
        /// Path to a .mlmodel, .mlpackage, or .mlmodelc.
        input: String,
    },

    /// Validate whether a model is compatible with the Apple Neural Engine.
    Validate {
        /// Path to the model to validate.
        input: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            input,
            output,
            target,
            quantize,
        } => {
            let output = output.unwrap_or_else(|| {
                let stem = std::path::Path::new(&input)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model");
                format!("{stem}.mlpackage")
            });
            println!("coreml-kit compile");
            println!("  input:    {input}");
            println!("  output:   {output}");
            println!("  target:   {target}");
            println!("  quantize: {quantize}");
            println!();
            println!("Not yet implemented — this is the scaffold.");
            println!("See docs/research/ for the project roadmap.");
        }
        Commands::Inspect { input } => {
            println!("coreml-kit inspect: {input}");
            println!("Not yet implemented.");
        }
        Commands::Validate { input } => {
            println!("coreml-kit validate: {input}");
            println!("Not yet implemented.");
        }
    }

    Ok(())
}
