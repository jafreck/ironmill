# Implementation Plan

## Overview

Build `mil-rs` (CoreML IR + protobuf foundation) and `ironmill` (CLI conversion tool)
in four phases. Each phase produces a shippable, useful artifact.

---

## Phase 1: CoreML Protobuf Foundation

**Goal**: Read and write CoreML `.mlmodel` / `.mlpackage` files from pure Rust.
**Ship**: `mil-rs` v0.1 on crates.io — the first Rust crate that can parse and emit CoreML models.

### 1.1 Vendor CoreML proto files
- Download all `.proto` files from `apple/coremltools/mlmodel/format/`
- Place in `crates/mil-rs/proto/`
- Key files: `Model.proto`, `MIL.proto`, `NeuralNetwork.proto`, `FeatureTypes.proto`,
  `DataStructures.proto`, `Parameters.proto`
- Non-essential model types (SVM, TreeEnsemble, GLM, etc.) can be included but aren't
  priority — they compile for free with prost

### 1.2 Set up prost code generation
- Add `build.rs` in `mil-rs` that compiles `.proto` → Rust types via `prost-build`
- Configure `prost-build` to generate into `src/proto/` (or use `OUT_DIR`)
- Verify generated types compile and are re-exported from `mil_rs::proto`

### 1.3 Implement `.mlmodel` reader
- Read a `.mlmodel` file (single protobuf binary)
- Deserialize into generated `Model` struct
- Print model metadata: inputs, outputs, model type, description

### 1.4 Implement `.mlpackage` reader
- Parse the `.mlpackage` directory structure (it's a folder, not a single file)
- Read `Manifest.json` to locate the model spec and weight files
- Load the protobuf spec + weight blobs separately
- Handle both old-format (`.mlmodel`) and new-format (`.mlpackage`)

### 1.5 Implement `.mlmodel` / `.mlpackage` writer
- Serialize a `Model` struct back to protobuf binary
- Write `.mlmodel` single-file format
- Write `.mlpackage` directory format (Manifest.json + spec + weights)
- Round-trip test: read → write → read → assert equal

### 1.6 MIL IR data structures
- Flesh out the IR types already scaffolded (Graph, Operation, TensorType, Value)
- Add: Function (a graph with a signature), Program (collection of functions)
- Add: Block (scope within a function, for control flow)
- Add: typed attributes matching MIL spec (e.g., `IntegerType`, `FloatType`, `TensorType`)
- These are Rust-native types, separate from the protobuf types — the protobuf types are
  for serialization, the IR types are for manipulation

### 1.7 Proto ↔ IR conversion
- `proto_to_ir()`: Convert protobuf `Model` → IR `Program`
- `ir_to_proto()`: Convert IR `Program` → protobuf `Model`
- Focus on MIL program models (the modern format), not legacy NeuralNetwork format
- Test with real `.mlmodel` files downloaded from Apple or HuggingFace

### 1.8 Tests and CI
- Unit tests for protobuf round-tripping
- Integration tests with real `.mlmodel` / `.mlpackage` files (small test models)
- GitHub Actions CI (macOS + Linux — protobuf generation works everywhere,
  runtime CoreML calls are macOS-only)
- `cargo doc` with examples

### 1.9 Publish `mil-rs` v0.1
- Minimal useful crate: read, inspect, and write CoreML model files from Rust
- Clear documentation and README
- Published to crates.io

---

## Phase 2: ONNX → CoreML Conversion

**Goal**: Convert ONNX models to CoreML format from Rust, no Python.
**Ship**: `ironmill` CLI that converts common ONNX models. `mil-rs` v0.2 with conversion API.

### 2.1 ONNX protobuf reader
- Add ONNX `.proto` files (from `onnx/onnx` repo) to the build
- Parse `.onnx` files into ONNX protobuf types
- Alternative: depend on existing `onnx-ir` or `tract-onnx` crate for ONNX parsing
  (evaluate trade-off: dependency vs. control)

### 2.2 ONNX → MIL op mapping (Tier 1: essential ops)
These cover ~80% of common models (vision + transformer inference):

| ONNX Op | MIL Op | Priority | Notes |
|---------|--------|----------|-------|
| Conv | conv | P0 | Core of all CNN models |
| MatMul | matmul | P0 | Attention, dense layers |
| Gemm | linear | P0 | Fully connected layers |
| Relu | relu | P0 | Most common activation |
| Add | add | P0 | Skip connections, bias |
| Mul | mul | P0 | Scaling, gating |
| Reshape | reshape | P0 | Ubiquitous shape manipulation |
| Transpose | transpose | P0 | Layout changes, attention |
| Softmax | softmax | P0 | Classification, attention |
| BatchNormalization | batch_norm | P0 | Normalization |
| MaxPool | max_pool | P0 | Downsampling |
| AveragePool | avg_pool | P0 | Downsampling |
| Concat | concat | P0 | Feature aggregation |
| Flatten | reshape | P0 | Maps to reshape |
| Sigmoid | sigmoid | P1 | Activation |
| Tanh | tanh | P1 | Activation |
| Clip | clip | P1 | ReLU6, clamping |
| Gather | gather | P1 | Embedding lookup |
| Unsqueeze | expand_dims | P1 | Shape manipulation |
| Squeeze | squeeze | P1 | Shape manipulation |
| Slice | slice_by_index | P1 | Tensor slicing |
| Pad | pad | P1 | Convolution padding |
| Resize | upsample_bilinear | P1 | Upsampling |
| ReduceMean | reduce_mean | P1 | Layer norm, pooling |
| LayerNormalization | layer_norm | P1 | Transformer norm |
| Cast | cast | P1 | Type conversion |
| Constant | const | P1 | Literal values |
| Shape | shape | P2 | Dynamic shape ops |
| Split | split | P2 | Tensor splitting |
| Where | select | P2 | Conditional |
| Pow | pow | P2 | Math |
| Sqrt | sqrt | P2 | Math |
| Div | real_div | P2 | Math |
| Sub | sub | P2 | Math |
| Erf | erf | P2 | GELU activation |
| ConvTranspose | conv_transpose | P2 | Decoder models |

### 2.3 Graph construction from ONNX
- Walk ONNX graph in topological order
- Map each ONNX node to MIL operations using the op table
- Handle ONNX initializers (weights) → MIL const ops
- Map ONNX inputs/outputs → MIL function signature
- Handle ONNX opset version differences (target opset 13+)

### 2.4 Basic optimization passes
- **Constant folding**: Evaluate ops with all-constant inputs at conversion time
- **Dead code elimination**: Remove ops whose outputs are unused
- **Identity elimination**: Remove no-op identity/reshape chains
- These are generic graph passes, not ANE-specific

### 2.5 `xcrun coremlcompiler` integration
- Shell out to `xcrun coremlcompiler compile model.mlpackage output_dir/`
- Detect Xcode/CommandLineTools installation
- Handle errors: missing xcrun, compilation failures
- Produce `.mlmodelc` directory ready for runtime loading

### 2.6 Wire up the CLI
- `ironmill compile model.onnx` → runs conversion + xcrun
- `ironmill inspect model.mlmodel` → reads and prints model structure
- `ironmill inspect model.onnx` → reads and prints ONNX structure
- Progress output showing which ops were mapped, any unsupported ops skipped

### 2.7 Validation test suite
- Convert models from ONNX Model Zoo (small ones: MNIST, SqueezeNet, MobileNet)
- Compare outputs: run original ONNX via `ort` and converted CoreML via `xcrun`
- Numerical accuracy comparison (tolerance for FP32→FP16 differences)
- CI: conversion tests on macOS, protobuf-only tests cross-platform

### 2.8 Publish
- `mil-rs` v0.2 with ONNX conversion API
- `ironmill` v0.1 CLI on crates.io
- Blog post / announcement

---

## Phase 3: ANE Optimization & Quantization

**Goal**: Models converted by ironmill actually run well on the ANE.
**Ship**: `mil-rs` v0.3 with optimization passes. `ironmill` v0.2 with `--quantize` and `--validate`.

### 3.1 ANE compatibility validator
- Analyze a MIL graph and report which ops will run on ANE vs fallback to CPU/GPU
- Check for known ANE constraints:
  - All input shapes must be static (no `None` dimensions)
  - Supported data types (FP16, INT8 only for most ops)
  - Max tensor sizes and memory limits
  - Supported op set (document which MIL ops ANE handles)
- Output: human-readable report + machine-readable JSON
- `ironmill validate model.mlpackage`

### 3.2 Shape materialization pass
- For models with dynamic shapes, insert concrete dimensions
- User provides target shapes via CLI: `--input-shape "input:1,3,224,224"`
- Replace dynamic dims in the IR with fixed values
- Required for ANE execution

### 3.3 FP16 quantization
- Convert FP32 weights and activations to FP16
- Straightforward truncation (no calibration needed for FP16)
- Update tensor types throughout the graph
- `ironmill compile model.onnx --quantize fp16`

### 3.4 Op fusion passes
- **Conv + BatchNorm fusion**: Fold BN parameters into conv weights (standard optimization)
- **Conv + Relu fusion**: Merge into single fused op
- **Linear + Relu fusion**: Same pattern for dense layers
- These are well-documented optimizations that improve both GPU and ANE performance

### 3.5 INT8 quantization (stretch)
- Post-training quantization with calibration data
- User provides a small calibration dataset
- Compute per-channel or per-tensor scale/zero-point
- More complex than FP16, may defer to Phase 4

### 3.6 Memory layout optimization
- Reorder tensor layouts for ANE's preferred format (channel-last for some ops)
- Insert transpose ops where needed
- This is heuristic-based and may require experimentation

### 3.7 Publish
- `mil-rs` v0.3
- `ironmill` v0.2
- Benchmark results: before/after optimization, GPU vs ANE

---

## Phase 4: Ecosystem Integration & Polish

**Goal**: Other Rust projects can depend on `mil-rs` for CoreML support.
**Ship**: Bridge crates, C API, comprehensive docs.

### 4.1 `candle-coreml` integration
- PR to existing `candle-coreml`: add optional `mil-rs` dependency
- New capability: convert a candle model graph → CoreML at build time
- Or: provide a helper that wraps `mil-rs` conversion for candle users

### 4.2 `burn-coreml` crate (new)
- Implement Burn's `Backend` trait using CoreML as the execution engine
- Model export: use `mil-rs` to convert Burn model → CoreML
- Inference: use CoreML runtime (via objc2) to execute

### 4.3 C API via cbindgen
- Expose key `mil-rs` functions with `extern "C"` + `#[no_mangle]`
- Generate C header with `cbindgen`
- Enables Swift, C++, Go, and other languages to use the converter
- Ship as a static library + header

### 4.4 `build.rs` API
- High-level API for use in `build.rs` scripts:
  ```rust
  ironmill::compile("model.onnx")
      .quantize(Fp16)
      .target(ComputeUnit::CpuAndNeuralEngine)
      .output("resources/model.mlmodelc")
      .build()?;
  ```
- Automatically calls xcrun if on macOS, errors clearly on other platforms

### 4.5 Benchmark suite
- Compare inference: Metal GPU (candle) vs CoreML+ANE (ironmill)
- Models: Whisper encoder, MobileNet, BERT-tiny
- Metrics: latency, throughput, power (if measurable)
- Publish results in docs

### 4.6 Documentation & examples
- Comprehensive rustdoc for all public APIs
- Example: convert and run Whisper
- Example: Tauri app with build.rs model conversion
- Example: candle integration
- Contributing guide for adding new ONNX op mappings

---

## Success Criteria

| Phase | Done when... |
|---|---|
| Phase 1 | Can read a real `.mlmodel`, inspect it, write it back, and the round-trip is lossless |
| Phase 2 | Can convert MobileNetV2 ONNX → CoreML, compile with xcrun, and get correct predictions |
| Phase 3 | Converted + optimized model runs on ANE (confirmed via Instruments) with FP16 quantization |
| Phase 4 | Another Rust project depends on `mil-rs` from crates.io and uses it successfully |
