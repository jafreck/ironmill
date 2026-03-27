# coreml-kit

A Rust-native CoreML model compiler and optimizer for Apple's Neural Engine.

**Convert ONNX models to CoreML format — no Python required.**

## The Problem

Apple's Neural Engine (ANE) delivers up to 38 TOPS of AI inference power on every
Apple Silicon device — but the only tool to prepare models for it (`coremltools`)
requires Python. This forces every Rust, C++, Swift, and Go developer through a
Python bottleneck just to deploy models to the most efficient AI accelerator in
consumer hardware.

Every other NPU vendor provides native tools:

| Vendor   | Tool             | Language |
|----------|------------------|----------|
| NVIDIA   | TensorRT         | C++      |
| Qualcomm | qairt-converter  | C++ CLI  |
| Intel    | OpenVINO         | C++      |
| Google   | IREE             | C++      |
| **Apple**| **coremltools**  | **Python only** |

`coreml-kit` fills this gap.

## Architecture

This project is a Cargo workspace with two crates:

- **`mil-rs`** — The foundation: MIL (Model Intermediate Language) IR, CoreML protobuf
  types, and graph manipulation primitives. The "serde of CoreML."
- **`coreml-kit-cli`** — The user-facing tool: ONNX → CoreML conversion with
  ANE-targeted optimizations.

```
ONNX model ──▶ mil-rs (IR) ──▶ optimize ──▶ CoreML .mlpackage ──▶ ANE
```

## Quick Start

```bash
# Install the CLI
cargo install coreml-kit-cli

# Convert an ONNX model to CoreML
coreml-kit compile model.onnx --target cpu-and-ne --quantize fp16

# Inspect a CoreML model
coreml-kit inspect model.mlpackage

# Validate ANE compatibility
coreml-kit validate model.mlpackage
```

### As a library (in `build.rs`)

```rust
// Cargo.toml: [build-dependencies] mil-rs = "0.1"
fn main() {
    // TODO: API not yet implemented
    // mil_rs::from_onnx("models/whisper.onnx")
    //     .optimize_for_ane()
    //     .save_mlpackage("resources/whisper.mlpackage")
    //     .unwrap();
}
```

## Roadmap

### Phase 1 — Foundation (current)
- [x] Project scaffold and workspace structure
- [ ] MIL IR data structures (graph, operations, types, tensors)
- [ ] CoreML protobuf reader/writer (via `prost`)
- [ ] Load and inspect pre-compiled `.mlmodelc` files

### Phase 2 — Conversion
- [ ] ONNX → MIL converter (top 50 ops)
- [ ] Basic optimization passes (constant folding, dead code elimination)
- [ ] `xcrun coremlcompiler` integration
- [ ] CLI: `coreml-kit compile model.onnx`

### Phase 3 — ANE Optimization
- [ ] Op fusion passes (conv+bn+relu, etc.)
- [ ] FP16/INT8 quantization pipeline
- [ ] Shape materialization for dynamic models
- [ ] ANE compatibility validator

### Phase 4 — Ecosystem
- [ ] `candle` integration crate
- [ ] `burn` backend
- [ ] C API via `cbindgen` for non-Rust consumers
- [ ] Benchmark suite (GPU vs ANE)

## Documentation

See [`docs/research/`](docs/research/) for the full research that led to this project:

- [ANE Gap Analysis](docs/research/ane-research.md) — The macOS AI landscape and what's missing
- [Competitive Analysis](docs/research/competitive-analysis.md) — Every competitor examined
- [Value Proposition](docs/research/value-proposition.md) — Why this matters for Rust and beyond

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
