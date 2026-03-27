# coreml-kit

Rust-native tools for working with Apple CoreML models — no Python required.

## Crates

| Crate | Description | Status |
|-------|-------------|--------|
| [`mil-rs`](crates/mil-rs/) | Read, write, and manipulate CoreML `.mlmodel`/`.mlpackage` files; MIL IR types; proto ↔ IR conversion | **v0.1** — usable |
| [`coreml-kit-cli`](crates/coreml-kit-cli/) | CLI for ONNX → CoreML conversion and model inspection | scaffold (not yet functional) |

## Motivation

Apple's CoreML framework runs models on the Neural Engine (ANE), GPU, or CPU.
The ANE offers significant power savings and frees the GPU for rendering — but
the only tool to *create* CoreML models (`coremltools`) requires Python.

`coreml-kit` fills this gap for Rust.

## Using mil-rs as a library

```rust,no_run
use mil_rs::{read_mlmodel, model_to_program, write_mlpackage, program_to_model};

let model = read_mlmodel("input.mlmodel").unwrap();
let program = model_to_program(&model).unwrap();
// ... inspect or transform ...
let out = program_to_model(&program, model.specification_version as i32).unwrap();
write_mlpackage(&out, "output.mlpackage").unwrap();
```

## Building

```bash
cargo build --workspace
cargo test --workspace
```

## Roadmap

### Phase 1 — Foundation (current)
- [x] MIL IR data structures (graph, operations, types, tensors)
- [x] CoreML protobuf reader/writer (via `prost`)
- [x] `.mlmodel` and `.mlpackage` round-trip
- [x] Proto ↔ IR bidirectional conversion

### Phase 2 — Conversion
- [ ] ONNX → MIL converter (common ops)
- [ ] Basic optimization passes (constant folding, dead code elimination)
- [ ] CLI: `coreml-kit compile model.onnx`

### Phase 3 — ANE Optimization
- [ ] Op fusion passes (conv+bn+relu, etc.)
- [ ] FP16/INT8 quantization pipeline
- [ ] ANE compatibility validator

### Phase 4 — Ecosystem
- [ ] `candle` / `burn` integration
- [ ] C API via `cbindgen`

## Documentation

See [`docs/research/`](docs/research/) for background research:

- [ANE Gap Analysis](docs/research/ane-research.md)
- [Competitive Analysis](docs/research/competitive-analysis.md)
- [Integration Strategy](docs/research/integration-strategy.md)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
