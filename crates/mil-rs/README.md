# mil-rs

Read, write, and manipulate Apple CoreML models in Rust — no Python required.

`mil-rs` provides a strongly-typed MIL (Model Intermediate Language)
intermediate representation, ONNX-to-CoreML conversion, optimization passes
targeting the Apple Neural Engine, and protobuf serialization for Apple's
CoreML on-disk formats.

## Installation

```toml
[dependencies]
mil-rs = "0.1"
```

## Features

### Model I/O

- **`.mlmodel` reader/writer** — serialize and deserialize single-file CoreML
  models (protobuf binary)
- **`.mlpackage` reader/writer** — directory-based package format with
  `Manifest.json` handling
- **ONNX reader** — parse `.onnx` files into ONNX protobuf types

### MIL IR

Strongly-typed intermediate representation mirroring Apple's MIL:

- `Program` → `Function` → `Block` → `Operation`
- `TensorType` and `ScalarType` for shape/dtype tracking
- `Value` enum for constants, references, and tensor data
- `Graph` for standalone operation graphs

### ONNX → CoreML conversion

Convert ONNX models to CoreML's MIL IR, then serialize as `.mlpackage`:

- Automatic topological sort of ONNX nodes
- Initializer lowering to `const` operations
- Graceful handling of unsupported ops (warnings, not failures)

### Optimization passes

All passes implement the `Pass` trait and transform a `Program` in place:

| Pass | Description |
|------|-------------|
| `DeadCodeEliminationPass` | Remove operations whose outputs are unused |
| `IdentityEliminationPass` | Bypass identity/copy operations |
| `ConstantFoldPass` | Evaluate constant expressions at compile time |
| `ConvBatchNormFusionPass` | Fuse `conv` + `batch_norm` into a single `conv` |
| `ConvReluFusionPass` | Fuse `conv` + `relu` into a single `conv` |
| `LinearReluFusionPass` | Fuse `linear` + `relu` into a single `linear` |
| `Fp16QuantizePass` | Quantize Float32 constants and types to Float16 |
| `ShapeMaterializePass` | Replace dynamic input shapes with concrete dimensions |

### ANE validation

Check whether operations will run on the Apple Neural Engine or fall back to
CPU/GPU. The validator reports per-operation compatibility, warnings for
dynamic shapes and oversized dimensions, and an overall compatibility
percentage.

### Compiler integration

Shell out to `xcrun coremlcompiler` to compile `.mlpackage` / `.mlmodel` into
`.mlmodelc` bundles loadable at runtime on Apple platforms.

## Usage

### Read and inspect a CoreML model

```rust,no_run
use mil_rs::{read_mlmodel, reader::print_model_summary};

let model = read_mlmodel("model.mlmodel").unwrap();
print_model_summary(&model);
```

### Convert ONNX to CoreML

```rust,no_run
use mil_rs::{read_onnx, onnx_to_program, program_to_model, write_mlpackage};

let onnx = read_onnx("model.onnx").unwrap();
let result = onnx_to_program(&onnx).unwrap();
let model = program_to_model(&result.program, 7).unwrap();
write_mlpackage(&model, "model.mlpackage").unwrap();
```

### Run optimization passes

```rust,no_run
use mil_rs::ir::{Pass, DeadCodeEliminationPass, ConstantFoldPass, Fp16QuantizePass};
# use mil_rs::{read_onnx, onnx_to_program};
# let onnx = read_onnx("m.onnx").unwrap();
# let result = onnx_to_program(&onnx).unwrap();
# let mut program = result.program;

DeadCodeEliminationPass.run(&mut program).unwrap();
ConstantFoldPass.run(&mut program).unwrap();
Fp16QuantizePass.run(&mut program).unwrap();
```

### Validate ANE compatibility

```rust,no_run
use mil_rs::{validate_ane_compatibility, validate::print_validation_report};
# use mil_rs::{read_onnx, onnx_to_program};
# let onnx = read_onnx("m.onnx").unwrap();
# let result = onnx_to_program(&onnx).unwrap();

let report = validate_ane_compatibility(&result.program);
print_validation_report(&report);
println!("ANE compatibility: {:.1}%", report.compatibility_pct);
```

### Round-trip between formats

```rust,no_run
use mil_rs::{read_mlpackage, write_mlmodel};

let model = read_mlpackage("input.mlpackage").unwrap();
write_mlmodel(&model, "output.mlmodel").unwrap();
```

## Crate structure

| Module | Description |
|--------|-------------|
| [`reader`](https://docs.rs/mil-rs/latest/mil_rs/reader/) | Read `.mlmodel`, `.mlpackage`, and `.onnx` files |
| [`writer`](https://docs.rs/mil-rs/latest/mil_rs/writer/) | Write `.mlmodel` and `.mlpackage` files |
| [`convert`](https://docs.rs/mil-rs/latest/mil_rs/convert/) | ONNX → MIL, proto ↔ IR conversion |
| [`ir`](https://docs.rs/mil-rs/latest/mil_rs/ir/) | MIL intermediate representation and optimization passes |
| [`validate`](https://docs.rs/mil-rs/latest/mil_rs/validate/) | ANE compatibility analysis |
| [`compiler`](https://docs.rs/mil-rs/latest/mil_rs/compiler/) | `xcrun coremlcompiler` integration |
| [`proto`](https://docs.rs/mil-rs/latest/mil_rs/proto/) | Auto-generated protobuf types (CoreML + ONNX) |
| [`error`](https://docs.rs/mil-rs/latest/mil_rs/error/) | `MilError` enum and `Result` type alias |

## Supported model types

The proto ↔ IR conversion supports **ML Program** models (CoreML spec v7+).
Legacy `NeuralNetwork` models can be read and written at the protobuf level
but cannot be converted to the MIL IR.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.
