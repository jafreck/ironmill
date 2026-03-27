# mil-rs

Rust implementation of Apple's Model Intermediate Language (MIL) and CoreML
protobuf format. Read, write, and manipulate CoreML models without any Python
dependency.

## Features

- **`.mlmodel` reader/writer** — serialize and deserialize single-file CoreML
  models (protobuf binary)
- **`.mlpackage` reader/writer** — full directory-based package format with
  `Manifest.json` handling
- **MIL IR types** — strongly-typed intermediate representation mirroring
  Apple's MIL: `Program`, `Function`, `Block`, `Operation`, `Graph`
- **Proto ↔ IR conversion** — lossless round-trip between protobuf `Model` and
  the Rust IR

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mil-rs = "0.1"
```

## Usage

### Read a `.mlmodel` file

```rust,no_run
use mil_rs::read_mlmodel;

let model = read_mlmodel("model.mlmodel").unwrap();
println!("spec version: {}", model.specification_version);
```

### Read a `.mlpackage` directory

```rust,no_run
use mil_rs::read_mlpackage;

let model = read_mlpackage("model.mlpackage").unwrap();
println!("spec version: {}", model.specification_version);
```

### Convert between protobuf and IR

```rust,no_run
use mil_rs::{read_mlmodel, model_to_program, program_to_model, write_mlpackage};

// Read a protobuf model and convert to the typed IR
let model = read_mlmodel("input.mlmodel").unwrap();
let program = model_to_program(&model).unwrap();

// Inspect or transform the program...
println!("functions: {}", program.functions.len());

// Convert back to protobuf and write as .mlpackage
let output_model = program_to_model(&program, model.specification_version as i32).unwrap();
write_mlpackage(&output_model, "output.mlpackage").unwrap();
```

### Write a model to a different format

```rust,no_run
use mil_rs::{read_mlpackage, write_mlmodel};

let model = read_mlpackage("input.mlpackage").unwrap();
write_mlmodel(&model, "output.mlmodel").unwrap();
```

## Crate structure

| Module    | Description |
|-----------|-------------|
| `ir`      | MIL intermediate representation — `Program`, `Function`, `Block`, `Operation`, `Graph`, tensor types |
| `proto`   | Auto-generated protobuf types from Apple's CoreML `.proto` spec |
| `reader`  | Functions to read `.mlmodel` and `.mlpackage` files |
| `writer`  | Functions to write `.mlmodel` and `.mlpackage` files |
| `convert` | Bidirectional conversion between protobuf `Model` and IR `Program` |
| `error`   | `MilError` enum and `Result` type alias |

## Supported model types

Currently supports **ML Program** models (CoreML spec v7+). Legacy
`NeuralNetwork` models can be read/written at the protobuf level but cannot be
converted to the MIL IR.

## Roadmap

- **v0.1** (current) — Read/write CoreML `.mlmodel` and `.mlpackage`, MIL IR,
  proto ↔ IR conversion
- **v0.2** — ONNX → MIL conversion for common operators
- **v0.3** — Optimization passes (constant folding, op fusion)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.
