# candle-coreml

Bridge between [candle](https://github.com/huggingface/candle) and Apple
CoreML via [mil-rs](https://github.com/jafreck/ironmill).

## Features

- **Conversion** (all platforms): Convert ONNX models to CoreML `.mlpackage`
  format using the `mil-rs` pipeline.
- **Runtime** (macOS only): Load and run CoreML models, returning f32 data
  and shapes that you convert to candle `Tensor` values.

## Conversion example

```rust,no_run
use candle_coreml::convert::{convert_onnx, ConvertOptions};
use ironmill_compile::coreml::build_api::Quantization;

let opts = ConvertOptions {
    quantization: Quantization::Fp16,
    ..Default::default()
};
let path = convert_onnx("model.onnx", "model.mlpackage", opts)
    .expect("conversion failed");
```

## Runtime example (macOS)

```rust,ignore
use candle_coreml::runtime::{CoreMlModel, ComputeUnits};
use candle_core::{Device, Tensor};

// Load a compiled CoreML model
let model = CoreMlModel::load("model.mlmodelc", ComputeUnits::All)?;

// Run inference
let input_data = vec![0.0f32; 1 * 3 * 224 * 224];
let outputs = model.predict(&[("input", &[1, 3, 224, 224], &input_data)])?;

// Convert outputs to candle tensors
for out in &outputs {
    let tensor = Tensor::from_slice(&out.data, &out.shape, &Device::Cpu)?;
    println!("{}: {:?}", out.name, tensor.shape());
}
```

## Related crates

| Crate | Role |
|-------|------|
| [mil-rs](https://github.com/jafreck/ironmill/tree/main/crates/mil-rs) | CoreML protobuf + MIL IR + ONNX conversion |
| [ironmill-coreml](https://github.com/jafreck/ironmill/tree/main/crates/ironmill-coreml) | macOS CoreML runtime wrapper |
| [ironmill](https://github.com/jafreck/ironmill) | CLI tool for model conversion |
