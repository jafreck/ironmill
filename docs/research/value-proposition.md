# Value Proposition: `mil-rs` / `coreml-kit`
## Training vs Inference, Ecosystem Fit, and the Rust AI Toolchain

---

## 1. Training or Inference?

**Primarily inference — but with a meaningful training/fine-tuning story.**

### The inference case (core value)
The ANE is fundamentally an inference accelerator. The entire value chain looks like:

```
Train (Python/GPU cloud)  →  Convert (???)  →  Deploy (device, ANE)
         ✅ solved              ❌ GAP             ✅ partially solved
```

Our project fills the "Convert" gap. Today, every Rust (or C++, or Swift, or Go) project
that wants to deploy a model to the ANE must:
1. Install Python + coremltools
2. Write a Python conversion script
3. Run it to produce .mlpackage
4. Compile with xcrun
5. Ship the .mlmodelc in their app

With `coreml-kit`, steps 1-4 collapse into:
```
coreml-kit compile model.onnx --target ane --quantize fp16
```
Or in Rust code:
```rust
let mil = coreml_kit::from_onnx("model.onnx")?;
let optimized = mil.optimize_for_ane()?;
optimized.save_mlpackage("model.mlpackage")?;
```

### The training/fine-tuning angle
CoreML's public API now supports on-device fine-tuning and personalization (as of
WWDC 2024+). Models can be updated with user data on-device without data leaving the
device. This means:

- **Updatable models** can be prepared with our tool (mark layers as updatable in the
  CoreML protobuf)
- **Fine-tuning runtime** is CoreML's job (we don't reimplement backprop)
- Our role: prepare models that *are structured for* on-device adaptation

So: **we enable inference deployment and fine-tuning preparation, not training from scratch.**
Training from scratch stays in Python/PyTorch where it belongs.

---

## 2. Where It Fits in the Wider AI Tooling Ecosystem

### The AI deployment pipeline (all languages)

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL LIFECYCLE                               │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │ TRAINING │    │  CONVERSION  │    │      DEPLOYMENT       │  │
│  │          │    │  & OPTIMIZE  │    │                       │  │
│  │ PyTorch  │    │              │    │  ┌─────────────────┐  │  │
│  │ JAX      │───▶│  coremltools │───▶│  │ CoreML Runtime  │  │  │
│  │ TF       │    │  (Python)    │    │  │ → ANE / GPU /CPU│  │  │
│  │          │    │              │    │  └─────────────────┘  │  │
│  │          │    │  TVM/IREE    │───▶│  ┌─────────────────┐  │  │
│  │          │    │  (C++/Python)│    │  │ Custom runtime  │  │  │
│  │          │    │              │    │  │ → GPU / CPU     │  │  │
│  │          │    │  OpenVINO    │───▶│  └─────────────────┘  │  │
│  │          │    │  (C++/Python)│    │  ┌─────────────────┐  │  │
│  │          │    │              │    │  │ ONNX Runtime    │  │  │
│  │          │    │  QAIRT       │───▶│  │ → various HW    │  │  │
│  │          │    │  (C++ CLI)   │    │  └─────────────────┘  │  │
│  └──────────┘    │              │    │                       │  │
│                  │  ┌─────────┐ │    │                       │  │
│                  │  │coreml-  │ │    │  All of the above     │  │
│                  │  │kit(Rust)│─│───▶│  but from Rust        │  │
│                  │  └─────────┘ │    │  without Python       │  │
│                  └──────────────┘    └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

The key insight: **every hardware accelerator vendor has a conversion tool —
except Apple requires you to use Python for theirs.** We fix this for the
non-Python world.

### Analogues in other ecosystems
| Vendor | Accelerator | Native conversion tool | Language |
|--------|------------|----------------------|----------|
| Qualcomm | Hexagon NPU | qairt-converter | C++ CLI |
| Intel | CPU/GPU/VPU | OpenVINO Model Optimizer | C++/CLI |
| Google | TPU/Edge TPU | IREE compiler | C++ |
| NVIDIA | GPU/DLA | TensorRT | C++ |
| **Apple** | **ANE** | **coremltools** | **Python only** ❌ |

We provide what every other vendor already has: a native tool.

---

## 3. Fit in the Rust AI Toolchain Specifically

The Rust AI ecosystem has matured around a clear pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                RUST AI TOOLCHAIN (2026)                      │
│                                                             │
│  TRAINING          FORMATS           INFERENCE              │
│  ┌──────┐          ┌──────┐          ┌──────────────────┐   │
│  │ burn │──export──▶│ ONNX │──load──▶│ ort (ONNX RT)    │   │
│  └──────┘          │      │         │ tract (pure Rust) │   │
│                    │      │         │ burn (own runtime) │   │
│  ┌────────┐        │      │         └──────────────────┘   │
│  │ candle │──uses──▶│ Safe │                                │
│  │        │        │Tensor│         ┌──────────────────┐   │
│  └────────┘        │      │──load──▶│ candle (Metal)    │   │
│                    └──────┘         └──────────────────┘   │
│                    ┌──────┐                                 │
│                    │ GGUF │──load──▶ llama.cpp (via FFI)    │
│                    └──────┘                                 │
│                                                             │
│  ════════════════════════════════════════════════════════    │
│  THE GAP: None of the above can target ANE.                 │
│  They all use Metal GPU or CPU on Apple Silicon.            │
│  ════════════════════════════════════════════════════════    │
│                                                             │
│  WITH coreml-kit:                                           │
│                                                             │
│  ┌──────┐   ┌──────┐   ┌───────────┐   ┌──────────────┐   │
│  │ ONNX │──▶│mil-rs│──▶│ coreml-kit│──▶│ CoreML + ANE │   │
│  │ Safe │   │(IR)  │   │(optimize) │   │ 38 TOPS      │   │
│  │ GGUF │   └──────┘   └───────────┘   │ 80x eff/watt │   │
│  └──────┘                               └──────────────┘   │
│                                                             │
│  burn ──export──▶ ONNX ──▶ coreml-kit ──▶ ANE ✅           │
│  candle ─export─▶ ONNX ──▶ coreml-kit ──▶ ANE ✅           │
│  tract ──reads──▶ ONNX ──▶ coreml-kit ──▶ ANE ✅           │
│  any framework ─▶ ONNX ──▶ coreml-kit ──▶ ANE ✅           │
└─────────────────────────────────────────────────────────────┘
```

### What each Rust project gains:

| Project | Today | With coreml-kit |
|---------|-------|-----------------|
| **candle** | Metal GPU on Mac, CPU elsewhere | +ANE (3-10x perf/watt for supported models) |
| **burn** | WGPU/Metal GPU, CPU | +ANE as a deployment target via ONNX export |
| **tract** | CPU only on Apple | +ANE acceleration path |
| **ort** | CoreML EP exists in C++ but not exposed to Rust | Native Rust CoreML path, no FFI gymnastics |
| **Tauri apps** | Ship Python to convert models, or pre-convert | `build.rs` integration, no Python in build |
| **murmur** (this repo) | whisper.cpp C bridge for CoreML | Pure Rust CoreML model loading + inference |
| **CLI tools** | Can't `cargo install` and target ANE | Self-contained, no external deps |

### The critical value for Rust specifically:

**Rust's promise is self-contained binaries with no runtime dependencies.**
Python in the model conversion pipeline breaks this promise completely.

A Tauri app developer today must:
1. Ensure Python 3.x is installed in CI
2. Install coremltools + numpy + protobuf + torch (1.5GB+)
3. Run conversion scripts
4. Hope the versions don't conflict

With coreml-kit, it's:
```toml
[build-dependencies]
coreml-kit = "0.1"
```
```rust
// build.rs
fn main() {
    coreml_kit::compile("models/whisper-small.onnx")
        .quantize(Fp16)
        .target(ComputeUnit::CpuAndNeuralEngine)
        .output("resources/whisper.mlmodelc")
        .build()
        .unwrap();
}
```

**That's the value: the entire conversion pipeline becomes a Rust build dependency.**

---

## 4. Who Specifically Would Use This?

### Immediate users (high confidence)
1. **whisper.cpp / murmur** — Rust speech-to-text projects that currently bridge to C
   for CoreML. Pure Rust path to ANE.
2. **Tauri AI apps** — Desktop apps shipping on-device models. No Python in CI.
3. **candle users on Mac** — HuggingFace ecosystem developers wanting ANE speedup.
4. **iOS/macOS Rust developers** — Growing community using Rust for Swift interop.

### Medium-term users (as ecosystem matures)
5. **burn** — Could add a CoreML backend using mil-rs as foundation.
6. **tract** — Could optionally target ANE for Apple platforms.
7. **Enterprise Rust shops** — Companies using Rust for performance-critical inference
   (fintech, real-time audio/video, robotics).

### Broader ecosystem impact
8. **C/C++ projects via FFI** — mil-rs could be called from C/C++ too, providing the
   first non-Python CoreML model generation for ANY native language.
9. **Swift projects** — Via swift-bridge or similar, Rust crate could replace coremltools
   in Xcode build pipelines.

---

## Summary

| Question | Answer |
|----------|--------|
| Training or inference? | **Inference deployment** + fine-tuning preparation. Training stays in Python. |
| Novel? | Yes — no non-Python CoreML model compiler exists in any language. |
| Wide ecosystem fit? | Fills the same role as TensorRT (NVIDIA), OpenVINO (Intel), QAIRT (Qualcomm) — but for Apple. |
| Rust-specific value? | Eliminates Python from the build pipeline, preserving Rust's self-contained binary promise. Unlocks ANE for every Rust ML framework. |
| Addressable audience? | Every Rust developer on Mac (large), every non-Python developer targeting ANE (very large), every Tauri/desktop AI app (growing fast). |
