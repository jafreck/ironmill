//! Example: Read and inspect a CoreML model.
//!
//! Usage:
//! ```bash
//! cargo run --example inspect_model -- path/to/model.mlmodel
//! ```

use mil_rs::{model_to_program, read_mlmodel, reader::print_model_summary};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model.mlmodel".to_string());

    println!("Reading: {path}");
    let model = read_mlmodel(&path)?;
    print_model_summary(&model);

    // Convert to the MIL IR and print function/op counts.
    match model_to_program(&model) {
        Ok(program) => {
            println!("\nMIL IR summary:");
            for (name, func) in &program.functions {
                println!(
                    "  function '{}': {} inputs, {} ops",
                    name,
                    func.inputs.len(),
                    func.body.operations.len()
                );
            }
        }
        Err(e) => {
            println!("\nNote: could not convert to MIL IR: {e}");
            println!("  (Only ML Program models with spec v7+ are supported)");
        }
    }

    Ok(())
}
