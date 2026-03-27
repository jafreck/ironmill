# macOS AI Inference & the ANE: Gap Analysis for Rust Ecosystem

## The Landscape (March 2026)

### Hardware: Apple Neural Engine (ANE)
- **M4/M5 chips**: ~38 TOPS (INT8), ~19 TFLOPS (FP16 real-world)
- **Efficiency**: 6.6 TFLOPS/W — 80x more power-efficient than NVIDIA A100 for some workloads
- **Present on**: Every Apple Silicon Mac, iPhone, iPad, Vision Pro (~2B+ devices)

### The Problem: A Locked-Down Accelerator
The ANE is the most power-efficient AI accelerator in consumer hardware, but Apple treats it as a black box:

| What you can do | What you can't do |
|---|---|
| Submit a CoreML model and *hope* it runs on ANE | Guarantee ANE execution for any specific op |
| Use FP16/INT8 with fixed input shapes | Use dynamic shapes, custom ops, or train |
| Get fast vision/audio inference | Debug why your model fell back to GPU |
| Profile at the CoreML level | See ANE-specific bottlenecks or schedules |

### Current Frameworks

| Framework | Language | ANE? | Strengths | Weaknesses |
|---|---|---|---|---|
| **CoreML** | Swift/ObjC | Yes (opaque) | Only public ANE path | Black box, fixed shapes, inference-only |
| **MLX** | Python/C++ | No (GPU only) | Flexible, NumPy-like | Cannot use ANE at all |
| **ONNX Runtime** | C/Python/Rust(FFI) | Via CoreML EP | Cross-platform | CoreML EP is bolted-on, limited control |
| **Orion** | C/C++ | Yes (direct!) | Reverse-engineered private APIs, training | Fragile, may break on OS updates |
| **ANEMLL** | Python/Swift/C++ | Via CoreML | LLM-focused, open source | Python toolchain, CoreML constraints |

### Rust-Specific Ecosystem

| Crate | Status | What it does | What it doesn't do |
|---|---|---|---|
| `coreml-rs` | Barely maintained | Basic model loading | No async, poor API, requires Swift |
| `coreml-native` | Pre-1.0, low adoption | Safe CoreML bindings via objc2 | No model conversion, no ANE control |
| `candle` + `candle-coreml` | Active | ML framework + CoreML inference bridge | No ANE guarantees, no model conversion |
| `ort` (ONNX Runtime) | Active | ONNX inference | No native CoreML EP from Rust |
| `burn` | Active | ONNX import → Rust code | No CoreML output, no ANE path |

## The Critical Gap

**There is no Rust-native toolchain for preparing, optimizing, and deploying models to the ANE.**

Every Rust project wanting ANE acceleration today must:
1. **Drop into Python** to convert models via `coremltools` (no Rust alternative)
2. Use immature CoreML bindings with **zero control** over compute unit dispatch
3. Accept that CoreML may **silently fall back** to GPU/CPU with no diagnostics
4. Deal with the **MIL IR and CoreML protobuf format** entirely in Python

This Python dependency is the #1 friction point in the entire chain. It breaks Rust's value proposition of a self-contained, reproducible toolchain.

## Recommended Project: `coreml-compiler` — A Rust-native CoreML Model Compiler

### Vision
A foundational Rust crate that eliminates the Python dependency for CoreML/ANE deployment. It would be the "missing link" between Rust ML frameworks and Apple's hardware accelerators.

### Architecture

```
┌─────────────────────────────────────────────┐
│              coreml-compiler                │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  ONNX    │  │SafeTensor│  │  GGUF    │  │
│  │  Reader  │  │  Reader  │  │  Reader  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └──────────────┼──────────────┘       │
│                      ▼                      │
│            ┌─────────────────┐              │
│            │   MIL IR (Rust) │              │
│            │  Graph Builder  │              │
│            └────────┬────────┘              │
│                     ▼                       │
│            ┌─────────────────┐              │
│            │  Optimization   │              │
│            │    Passes       │              │
│            │ • Op fusion     │              │
│            │ • Shape fixing  │              │
│            │ • Quantization  │              │
│            │ • ANE targeting │              │
│            └────────┬────────┘              │
│                     ▼                       │
│            ┌─────────────────┐              │
│            │ CoreML Protobuf │              │
│            │    Emitter      │              │
│            │ (.mlpackage)    │              │
│            └────────┬────────┘              │
│                     ▼                       │
│            ┌─────────────────┐              │
│            │  Runtime        │              │
│            │ • Load .mlmodelc│              │
│            │ • Inference API │              │
│            │ • Device control│              │
│            └─────────────────┘              │
└─────────────────────────────────────────────┘
```

### Core Components

1. **MIL IR in Rust** — A typed, graph-based intermediate representation matching Apple's MIL spec. This is the linchpin: every model format converts *to* MIL, and MIL converts *to* CoreML protobuf.

2. **Model Format Readers** — Parse ONNX, SafeTensors, and GGUF into the MIL IR. Start with ONNX (broadest compatibility).

3. **ANE-Targeted Optimization Passes**:
   - **Op fusion**: Merge conv+bn+relu into fused ops the ANE handles natively
   - **Shape materialization**: Convert dynamic shapes to fixed (required for ANE)
   - **Quantization**: FP32→FP16/INT8 with calibration data support
   - **Memory layout**: Reorder tensors for ANE's expected NHWC layout
   - **Op substitution**: Replace unsupported ops with ANE-friendly equivalents

4. **CoreML Protobuf Emitter** — Generate `.mlmodel` / `.mlpackage` files. The CoreML format is protobuf-based and Apple's `.proto` schemas are published in `coremltools`.

5. **Compilation Bridge** — Shell out to `xcrun coremlcompiler` to produce `.mlmodelc` (this is unavoidable — Apple's compiler is closed-source, but the CLI tool is free on every Mac).

6. **Runtime Inference API** — Safe, async Rust API wrapping CoreML with explicit `ComputeUnit` selection (`.cpuAndNeuralEngine`, `.all`, `.cpuOnly`).

### Why This Has Wide Appeal

| Who benefits | How |
|---|---|
| **candle / burn users** | Drop-in CoreML backend for ANE acceleration |
| **whisper.cpp / llama.cpp Rust ports** | Eliminate Python model prep step entirely |
| **Tauri / desktop app developers** | Ship on-device AI features without Python |
| **iOS/macOS Rust developers** | Native Swift-free path to ANE from Rust |
| **CLI tool authors** | Build `cargo install`-able inference tools |
| **CI/CD pipelines** | Reproducible model compilation without Python env |

### Competitive Differentiation

| vs. what exists | Advantage |
|---|---|
| vs. `coremltools` (Python) | No Python dependency, embeddable in Rust apps |
| vs. `coreml-native` / `coreml-rs` | Full pipeline (convert + optimize + run), not just bindings |
| vs. Orion (direct ANE) | Uses public APIs — won't break on OS updates |
| vs. ONNX Runtime CoreML EP | Native Rust, deeper optimization, ANE-specific passes |

### Phased Roadmap

**Phase 1 — Foundation**
- MIL IR data structures in Rust
- CoreML protobuf reader/writer (using `prost`)
- Load and run pre-compiled `.mlmodelc` files from Rust
- Explicit compute unit selection

**Phase 2 — Conversion**
- ONNX → MIL converter (core ops: conv, matmul, relu, softmax, etc.)
- Basic optimization passes (constant folding, dead code elimination)
- Integration with `xcrun coremlcompiler`

**Phase 3 — ANE Optimization**
- Op fusion passes targeting ANE primitives
- FP16/INT8 quantization pipeline
- Shape materialization for dynamic models
- ANE compatibility validator ("will this model actually run on ANE?")

**Phase 4 — Ecosystem Integration**
- `candle-coreml-compiler` bridge crate
- `burn` backend
- CLI tool: `coreml-compile model.onnx --target ane --quantize fp16`
- Benchmark suite comparing GPU vs ANE inference

## Key Sources

- [Orion paper — Direct ANE programming](https://arxiv.org/html/2603.06728v1)
- [Disaggregated inference on Apple Silicon](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
- [ANEMLL project](https://github.com/Anemll/Anemll)
- [CoreML Tools (Apple)](https://apple.github.io/coremltools/docs-guides/)
- [coreml-native crate](https://github.com/robertelee78/coreml-native/)
- [whisper.cpp ANE issues](https://github.com/ggml-org/whisper.cpp/issues/3702)
- [Apple's on-device Llama research](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- [MLX benchmarks](https://towardsdatascience.com/how-fast-is-mlx-a-comprehensive-benchmark-on-8-apple-silicon-chips-and-4-cuda-gpus-378a0ae356a0/)
