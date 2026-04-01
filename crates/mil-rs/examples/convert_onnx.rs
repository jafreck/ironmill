//! Example: Convert an ONNX model to a CoreML `.mlpackage`.
//!
//! Usage:
//! ```bash
//! cargo run --example convert_onnx -- input.onnx output.mlpackage
//! ```

use mil_rs::ir::{ConstantFoldPass, DeadCodeEliminationPass, IdentityEliminationPass, Pass};
use mil_rs::{onnx_to_program, program_to_model, read_onnx, write_mlpackage};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).map_or("model.onnx", |s| s.as_str());
    let output = args.get(2).map_or("model.mlpackage", |s| s.as_str());

    // 1. Read the ONNX model.
    println!("Reading ONNX model: {input}");
    let mut onnx_model = read_onnx(input)?;

    // 2. Convert to the MIL IR.
    println!("Converting to MIL IR...");
    let result = onnx_to_program(&mut onnx_model)?;
    let mut program = result.program;

    if !result.warnings.is_empty() {
        println!("Warnings: {:?}", result.warnings);
    }

    // 3. Run optimization passes.
    let passes: Vec<Box<dyn Pass>> = vec![
        Box::new(DeadCodeEliminationPass),
        Box::new(IdentityEliminationPass),
        Box::new(ConstantFoldPass),
    ];
    for pass in &passes {
        pass.run(&mut program)?;
        println!("  ran pass: {}", pass.name());
    }

    // 4. Convert back to CoreML protobuf and write.
    let model = program_to_model(&program, 7)?;
    write_mlpackage(&model, output)?;
    println!("Wrote: {output}");

    Ok(())
}
