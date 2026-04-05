# macOS AI Inference & the ANE: Gap Analysis for Rust Ecosystem

## The Landscape (March 2026)

### Hardware: Apple Neural Engine (ANE)
- **M4/M5 chips**: ~38 TOPS (INT8 marketing), ~19 TFLOPS (FP16 real-world)
- **Efficiency**: ~1.5-2x more power-efficient than the local Metal GPU for supported models
  (often cited as "80x vs A100" but that's a cherry-picked cross-class comparison)
- **Present on**: Every Apple Silicon Mac, iPhone, iPad, Vision Pro (~2B+ devices)

### What Actually Works Today

**Models CAN access the ANE.** This is not theoretical - CoreML routes supported operations
to the ANE through its public API. From Rust, the `coreml-native` crate can load compiled
CoreML models (`.mlmodelc`) and run inference with `ComputeUnits::CpuAndNeuralEngine`.
This works today.

The real constraints are:

| What works | What doesn't |
|---|---|
| CoreML routes supported ops to ANE automatically | You can't guarantee which ops run on ANE vs CPU |
| FP16/INT8 models with fixed input shapes run well | Dynamic shapes force fallback to CPU |
| `coreml-native` (Rust) loads and runs `.mlmodelc` files | No Rust tool can *create* `.mlmodelc` files |
| Xcode Instruments can profile ANE usage | No programmatic way to query "did this op use ANE?" |
| ANE uses 40-65% less power than Metal GPU | ANE speed is only ~20-33% faster (not a dramatic gap) |

### ANE vs Metal GPU: Honest Benchmarks

The ANE advantage is **real but modest** for speed, and **significant** for power:

| Workload | ANE | Metal GPU | Speed gap | Power gap |
|---|---|---|---|---|
| LLaMA 7B Q4 | ~80 tok/s | ~60 tok/s | ANE ~33% faster | ANE ~40% less power |
| Whisper encoder | ~3x vs CPU | ~2.5-3x vs CPU | Roughly equal | ANE ~50% less power |
| YOLOv8 INT8 | ~92 FPS | ~70 FPS | ANE ~30% faster | ANE ~65% less power |

**Caveat**: ANE has a multi-minute cold compilation penalty on first load. whisper.cpp
actually defaults the encoder to Metal GPU to avoid this. For interactive use, Metal GPU
often provides a better experience.

### When ANE Actually Matters

ANE's advantage is decisive in specific scenarios:
- **Battery-constrained devices** (iPhone, iPad, MacBook on battery)
- **Always-on background inference** (dictation, accessibility, health monitoring)
- **GPU-busy workloads** (games, video editing, AR) where inference must not compete for GPU
- **Thermal-constrained environments** (fanless devices, sustained workloads)

For a Rust developer running inference on a plugged-in Mac, Metal GPU via candle/burn
is often "good enough." The ANE story is strongest for shipping apps to end users.

### Current Frameworks

| Framework | Language | ANE? | Strengths | Weaknesses |
|---|---|---|---|---|
| **CoreML** | Swift/ObjC | Yes (opaque) | Only public ANE path | Black box scheduling, fixed shapes |
| **MLX** | Python/C++ | No (GPU only) | Flexible, NumPy-like | Cannot use ANE at all |
| **ONNX Runtime** | C/Python/Rust(FFI) | Via CoreML EP | Cross-platform | CoreML EP is bolted-on, limited control |
| **Orion** | C/C++ | Yes (direct!) | Reverse-engineered private APIs | Fragile, may break on OS updates |
| **ANEMLL** | Python/Swift/C++ | Via CoreML | LLM-focused, open source | Python toolchain required |

### Rust-Specific Ecosystem

| Crate | Status | What it does | What it doesn't do |
|---|---|---|---|
| `coreml-native` | Pre-1.0, low adoption | Load `.mlmodelc`, run inference with ANE | Create, convert, or optimize models |
| `coreml-rs` | Barely maintained | Basic model loading | No async, poor API, requires Swift |
| `candle` | Active, large community | ML framework, Metal GPU inference | No CoreML/ANE path |
| `burn` | Active | ONNX import, multi-backend | No CoreML output, no ANE path |
| `ort` (ONNX Runtime) | Active | ONNX inference | CoreML EP not exposed to Rust |
| `tract` | Production-used | Pure Rust ONNX inference | CPU-only on Apple |

## The Actual Gap

**The gap is in model conversion, not model execution.**

Rust can already *run* CoreML models on the ANE via `coreml-native`. What's missing is
the ability to *create* CoreML models without Python. Today's workflow:

```
Train (Python ✅) → Convert to .mlpackage (Python-only ❌) → Compile (xcrun ✅) → Run (Rust ✅)
                         ↑ THIS is the gap
```

Every Rust project wanting CoreML deployment must:
1. Install Python + `coremltools` + numpy + protobuf + torch (~1.5GB)
2. Write a Python conversion script
3. Run it to produce `.mlpackage`
4. Compile with `xcrun coremlcompiler`
5. Ship the `.mlmodelc` in their app

This Python dependency breaks Rust's self-contained toolchain promise. It's the same
problem for C++, Swift (without Xcode), and Go developers.

## Project: `ironmill` - Rust-native CoreML Model Conversion

### Vision
Eliminate the Python dependency in the CoreML model conversion pipeline. Not a
reimplementation of all of `coremltools` - a focused tool that converts common model
formats to CoreML and applies basic optimizations.

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

1. **MIL IR in Rust** - A typed, graph-based intermediate representation matching Apple's MIL spec. This is the linchpin: every model format converts *to* MIL, and MIL converts *to* CoreML protobuf.

2. **Model Format Readers** - Parse ONNX, SafeTensors, and GGUF into the MIL IR. Start with ONNX (broadest compatibility).

3. **ANE-Targeted Optimization Passes**:
   - **Op fusion**: Merge conv+bn+relu into fused ops the ANE handles natively
   - **Shape materialization**: Convert dynamic shapes to fixed (required for ANE)
   - **Quantization**: FP32→FP16/INT8 with calibration data support
   - **Memory layout**: Reorder tensors for ANE's expected NHWC layout
   - **Op substitution**: Replace unsupported ops with ANE-friendly equivalents

4. **CoreML Protobuf Emitter** - Generate `.mlmodel` / `.mlpackage` files. The CoreML format is protobuf-based and Apple's `.proto` schemas are published in `coremltools`.

5. **Compilation Bridge** - Shell out to `xcrun coremlcompiler` to produce `.mlmodelc` (this is unavoidable - Apple's compiler is closed-source, but the CLI tool is free on every Mac).

6. **Runtime Inference API** - Safe, async Rust API wrapping CoreML with explicit `ComputeUnit` selection (`.cpuAndNeuralEngine`, `.all`, `.cpuOnly`).

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
| vs. Orion (direct ANE) | Uses public APIs - won't break on OS updates |
| vs. ONNX Runtime CoreML EP | Native Rust, deeper optimization, ANE-specific passes |

### Phased Roadmap

**Phase 1 - Foundation**
- MIL IR data structures in Rust
- CoreML protobuf reader/writer (using `prost`)
- Load and run pre-compiled `.mlmodelc` files from Rust
- Explicit compute unit selection

**Phase 2 - Conversion**
- ONNX → MIL converter (core ops: conv, matmul, relu, softmax, etc.)
- Basic optimization passes (constant folding, dead code elimination)
- Integration with `xcrun coremlcompiler`

**Phase 3 - ANE Optimization**
- Op fusion passes targeting ANE primitives
- FP16/INT8 quantization pipeline
- Shape materialization for dynamic models
- ANE compatibility validator ("will this model actually run on ANE?")

**Phase 4 - Ecosystem Integration**
- `candle-coreml-compiler` bridge crate
- `burn` backend
- CLI tool: `coreml-compile model.onnx --target ane --quantize fp16`
- Benchmark suite comparing GPU vs ANE inference

## Key Sources

- [Orion paper - Direct ANE programming](https://arxiv.org/html/2603.06728v1)
- [Disaggregated inference on Apple Silicon](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
- [ANEMLL project](https://github.com/Anemll/Anemll)
- [CoreML Tools (Apple)](https://apple.github.io/coremltools/docs-guides/)
- [coreml-native crate](https://github.com/robertelee78/coreml-native/)
- [whisper.cpp ANE issues](https://github.com/ggml-org/whisper.cpp/issues/3702)
- [Apple's on-device Llama research](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- [MLX benchmarks](https://towardsdatascience.com/how-fast-is-mlx-a-comprehensive-benchmark-on-8-apple-silicon-chips-and-4-cuda-gpus-378a0ae356a0/)
