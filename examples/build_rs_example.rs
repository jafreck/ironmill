//! Example: Using `mil_rs::CompileBuilder` in a `build.rs` script.
//!
//! Add this to your project's `build.rs` to automatically convert an
//! ONNX model to CoreML at build time.
//!
//! This file is illustrative — it shows the code you would put in your
//! own `build.rs`. It is not a runnable binary in this workspace.

// ── Basic usage ─────────────────────────────────────────────────────────
//
// Convert model.onnx to a .mlpackage with default settings:
//
// ```rust,no_run
// use mil_rs::CompileBuilder;
//
// fn main() {
//     CompileBuilder::new("model.onnx")
//         .output("resources/model.mlpackage")
//         .build()
//         .expect("model compilation failed");
// }
// ```

// ── With FP16 quantization ─────────────────────────────────────────────
//
// Reduce model size by converting weights and activations to float16:
//
// ```rust,no_run
// use mil_rs::{CompileBuilder, Quantization};
//
// fn main() {
//     CompileBuilder::new("model.onnx")
//         .quantize(Quantization::Fp16)
//         .output("resources/model.mlpackage")
//         .build()
//         .expect("FP16 compilation failed");
// }
// ```

// ── With input shapes for ANE ──────────────────────────────────────────
//
// Fix dynamic input dimensions so the model is eligible for the Apple
// Neural Engine. Multiple inputs can be shaped independently:
//
// ```rust,no_run
// use mil_rs::{CompileBuilder, TargetComputeUnit};
//
// fn main() {
//     CompileBuilder::new("model.onnx")
//         .target(TargetComputeUnit::CpuAndNeuralEngine)
//         .input_shape("images", vec![1, 3, 224, 224])
//         .input_shape("mask", vec![1, 1, 224, 224])
//         .output("resources/model.mlpackage")
//         .build()
//         .expect("ANE compilation failed");
// }
// ```

// ── With weight palettization ──────────────────────────────────────────
//
// Compress weights using k-means palettization (2, 4, 6, or 8 bits).
// This can be combined with FP16 quantization:
//
// ```rust,no_run
// use mil_rs::{CompileBuilder, Quantization};
//
// fn main() {
//     CompileBuilder::new("model.onnx")
//         .quantize(Quantization::Fp16)
//         .palettize(4)
//         .output("resources/model.mlpackage")
//         .build()
//         .expect("palettized compilation failed");
// }
// ```

// ── Full pipeline with .mlmodelc compilation ───────────────────────────
//
// Convert, optimize, write, and compile to .mlmodelc in one step.
// The `.compile()` call invokes `xcrun coremlcompiler` on macOS (no-op
// on other platforms).
//
// ```rust,no_run
// use mil_rs::{CompileBuilder, Quantization, TargetComputeUnit};
//
// fn main() {
//     let output = CompileBuilder::new("model.onnx")
//         .quantize(Quantization::Fp16)
//         .target(TargetComputeUnit::CpuAndNeuralEngine)
//         .input_shape("input", vec![1, 3, 224, 224])
//         .palettize(4)
//         .output("resources/model.mlpackage")
//         .compile()  // also produce .mlmodelc
//         .build()
//         .expect("full pipeline failed");
//
//     println!("mlpackage: {}", output.mlpackage.display());
//     if let Some(ref compiled) = output.mlmodelc {
//         println!("mlmodelc:  {}", compiled.display());
//     }
//     println!("passes run: {}", output.report.passes_run);
// }
// ```

fn main() {
    // This file is illustrative only. See the code examples above
    // for patterns to use in your own build.rs.
    println!("See the doc comments in this file for build.rs usage examples.");
}
