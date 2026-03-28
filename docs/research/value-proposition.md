# Value Proposition: `mil-rs` / `ironmill`
## Training vs Inference, Ecosystem Fit, and the Rust AI Toolchain

---

## 1. Training or Inference?

**Inference deployment — specifically, model conversion for deployment.**

### What's actually missing
The ANE is accessible today. `coreml-native` (Rust) can load compiled CoreML models and
run inference with ANE acceleration. Candle and burn provide solid Metal GPU inference.
The Rust inference story is reasonable.

What's missing is the **conversion pipeline**:

```
Train (Python ✅)  →  Convert to CoreML (Python-only ❌)  →  Run on ANE (Rust ✅)
                           ↑ THIS is the gap
```

With `ironmill`, the conversion step becomes native Rust:
```
ironmill compile model.onnx --target ane --quantize fp16
```
Or in Rust code:
```rust
let mil = ironmill::from_onnx("model.onnx")?;
let optimized = mil.optimize_for_ane()?;
optimized.save_mlpackage("model.mlpackage")?;
```

### The training/fine-tuning angle
CoreML's public API now supports on-device fine-tuning and personalization (as of
WWDC 2024+). Our role would be to prepare models that are *structured for* on-device
adaptation (marking layers as updatable in the CoreML protobuf). We don't reimplement
backprop — training from scratch stays in Python/PyTorch where it belongs.

---

## 2. Where It Fits in the Wider AI Tooling Ecosystem

### The real problem in context
Every NPU vendor provides native model conversion tools — except Apple, which requires Python:

| Vendor | Accelerator | Native conversion tool | Language |
|--------|------------|----------------------|----------|
| Qualcomm | Hexagon NPU | qairt-converter | C++ CLI |
| Intel | CPU/GPU/VPU | OpenVINO Model Optimizer | C++/CLI |
| Google | TPU/Edge TPU | IREE compiler | C++ |
| NVIDIA | GPU/DLA | TensorRT | C++ |
| **Apple** | **ANE** | **coremltools** | **Python only** ❌ |

We provide what every other vendor already has: a native conversion tool.

### Honest scope
This is **not** "unlocking stranded hardware that nobody can use." CoreML + ANE works
today via Python conversion → xcrun compile → native runtime. The value is removing
Python from that pipeline — which matters most for:

- **CI/CD pipelines** that don't want a Python environment
- **Rust/C++ apps** that want self-contained build systems
- **Tauri/desktop apps** where `cargo build` should be sufficient
- **iOS/macOS Rust developers** who want `build.rs` integration

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
│  └────────┘        │      │──load──▶│ candle (Metal GPU)│   │
│                    └──────┘         └──────────────────┘   │
│                    ┌──────┐                                 │
│                    │ GGUF │──load──▶ llama.cpp (via FFI)    │
│                    └──────┘                                 │
│                                                             │
│  Metal GPU works well for most workloads today.             │
│  ANE adds: lower power, frees GPU, ~20-33% faster          │
│  for quantized models. Worth it for shipped apps.           │
│                                                             │
│  WITH ironmill:                                           │
│                                                             │
│  ┌──────┐   ┌──────┐   ┌───────────┐   ┌──────────────┐   │
│  │ ONNX │──▶│mil-rs│──▶│ ironmill│──▶│ CoreML + ANE │   │
│  │      │   │(IR)  │   │(convert)  │   │              │   │
│  └──────┘   └──────┘   └───────────┘   └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### What each Rust project gains:

| Project | Today (Metal GPU) | With ironmill (adds ANE option) |
|---------|-------------------|-----------------------------------|
| **candle** | Metal GPU works well | +ANE for lower power, frees GPU for rendering |
| **burn** | WGPU/Metal GPU, CPU | +CoreML as a deployment target via ONNX export |
| **tract** | CPU only on Apple | +CoreML/ANE acceleration path |
| **Tauri apps** | Must use Python to convert models | `build.rs` integration, no Python |
| **CLI tools** | Pre-convert models with Python | Self-contained `cargo install` |

### The critical Rust-specific value

**Rust's promise is self-contained binaries with no runtime dependencies.**
Python in the model conversion pipeline breaks this promise.

A Tauri app developer today must:
1. Ensure Python 3.x is installed in CI
2. Install coremltools + numpy + protobuf + torch (1.5GB+)
3. Run conversion scripts
4. Hope the versions don't conflict

With ironmill, it's:
```toml
[build-dependencies]
ironmill = "0.1"
```
```rust
// build.rs
fn main() {
    ironmill::compile("models/whisper-small.onnx")
        .quantize(Fp16)
        .target(ComputeUnit::CpuAndNeuralEngine)
        .output("resources/whisper.mlmodelc")
        .build()
        .unwrap();
}
```

---

## 4. Who Specifically Would Use This?

### Strongest use cases
1. **Tauri/desktop AI apps** — No Python in CI, self-contained builds
2. **iOS/macOS Rust developers** — `build.rs` model conversion
3. **CI/CD pipelines** — Reproducible model compilation without Python env
4. **CLI tool authors** — `cargo install`-able tools that include model conversion

### Moderate use cases
5. **candle/burn users** — ANE path for power-sensitive deployments (mobile, laptop)
6. **Embedded Rust on Apple** — IoT, kiosk, always-on inference

### Weaker use cases (Metal GPU is fine)
7. **Server-side Rust on Mac** — Plugged in, GPU available, power doesn't matter
8. **Dev/prototyping** — Metal GPU via candle is fast enough

---

## 5. Honest Assessment

| Claim | Reality |
|-------|---------|
| "Models can't access ANE from Rust" | **Wrong.** `coreml-native` does this today. |
| "ANE is 80x more efficient than GPU" | **Misleading.** That's vs data-center A100. Vs local Metal GPU it's ~1.5-2x. |
| "38 TOPS of stranded hardware" | **Overstated.** Real throughput is ~19 TFLOPS. ANE is used by CoreML automatically. |
| "No way to convert models without Python" | **True.** This is the real gap. |
| "Python breaks Rust's toolchain promise" | **True.** This matters for app developers and CI. |
| "ANE is faster than GPU" | **Sometimes.** ~20-33% for quantized models, but GPU avoids cold-compile penalty. |
| "ANE saves significant power" | **True.** 40-65% less power. Matters for battery and thermal. |
| "The GPU can do other work while ANE infers" | **True.** Important for games, AR, video editing. |

## Summary

| Question | Answer |
|----------|--------|
| Training or inference? | **Model conversion for inference deployment.** Training stays in Python. |
| What's truly missing? | Native (non-Python) model conversion to CoreML format. |
| Is ANE access broken? | **No.** Runtime access works. Conversion pipeline requires Python. |
| Is this worth building? | Yes — for the toolchain story and the shipped-app power/GPU-freedom benefits. Not for dramatic speed gains. |
| Addressable audience? | Rust app developers shipping on Apple platforms, CI/CD pipelines, anyone wanting Python-free CoreML conversion. |
