# burn-coreml

CoreML model export and inference for [Burn](https://burn.dev) via
[mil-rs](https://github.com/jafreck/ironmill).

## What this crate does

`burn-coreml` bridges the gap between Burn models and Apple's CoreML runtime:

- **Export** - Convert Burn models (via ONNX) to CoreML `.mlpackage` format
- **Inference** (macOS only) - Load and run CoreML models with f32 tensors

## Workflow

```text
Burn Model  →  ONNX  →  CoreML .mlpackage  →  .mlmodelc (compiled)
   (Burn's        (burn-coreml               (burn-coreml
    ONNX           export)                    inference)
    recorder)
```

1. Export your Burn model to ONNX using Burn's built-in ONNX recorder
2. Use `burn_coreml::export::export_to_coreml` to convert ONNX → CoreML
3. Optionally compile to `.mlmodelc` for fast loading
4. Use `burn_coreml::inference::CoreMlInference` to run inference on macOS

## Why no `burn-core` dependency?

This crate intentionally does **not** depend on `burn-core`. It provides
standalone export/inference utilities that work alongside any version of Burn.
This avoids:

- Heavy dependency coupling
- Version lock-in with Burn releases
- Long compile times from pulling in the full Burn dependency tree

Users simply export to ONNX (which Burn already supports) and then use this
crate for CoreML conversion and inference.

## Export example

```rust,no_run
use burn_coreml::export::{export_to_coreml, ExportOptions};
use ironmill_compile::coreml::build_api::{Quantization, TargetComputeUnit};

// Convert a Burn-exported ONNX model to CoreML
let result = export_to_coreml(
    "my_model.onnx",
    "my_model.mlpackage",
    ExportOptions {
        quantization: Quantization::Fp16,
        target: Some(TargetComputeUnit::CpuAndNeuralEngine),
        input_shapes: vec![("input".into(), vec![1, 3, 224, 224])],
        compile: true,
        ..Default::default()
    },
)?;

println!("exported to {}", result.mlpackage_path.display());
if let Some(compiled) = &result.mlmodelc_path {
    println!("compiled to {}", compiled.display());
}
# Ok::<(), anyhow::Error>(())
```

## Inference example (macOS)

```rust,no_run
use burn_coreml::inference::CoreMlInference;
use ironmill_coreml::ComputeUnits;

// Load a compiled CoreML model
let session = CoreMlInference::load("my_model.mlmodelc", ComputeUnits::All)?;

// Check input requirements
let inputs = session.input_description()?;
for input in &inputs {
    println!("input '{}': shape {:?}", input.name, input.shape);
}

// Run inference
let input_data = vec![0.0f32; 1 * 3 * 224 * 224];
let outputs = session.predict(&[
    ("input", &[1, 3, 224, 224], &input_data),
])?;

for output in &outputs {
    println!("output '{}': shape {:?}", output.name, output.shape);
}
# Ok::<(), anyhow::Error>(())
```

## License

Apache-2.0
