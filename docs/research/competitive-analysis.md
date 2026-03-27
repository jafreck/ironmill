# Competitive Analysis: Rust-native CoreML Model Compiler

## Executive Summary

**Is this truly novel?** Yes — with caveats. The *concept* of a model compiler targeting a
hardware accelerator is well-established (Qualcomm, Intel, Google all have them). What's novel
is that **no one has built one for CoreML/ANE outside of Python, in any language.** Not in C++,
not in Swift (meaningfully), not in Go, not in Rust. Apple created a Python-only bottleneck
that the entire non-Python world must route through.

---

## Tier 1: Direct Competitors (CoreML Model Generation)

### 1. Apple `coremltools` (Python) — THE incumbent
- **What it does**: Full pipeline: PyTorch/TF/ONNX → MIL IR → CoreML protobuf → .mlmodel/.mlpackage
- **Language**: Python only
- **Stars**: ~4.5k GitHub
- **Strengths**: Official, complete, MIL optimizer, quantization, maintained by Apple
- **Weaknesses**: Python-only, heavyweight dependency, impossible to embed in native apps
- **Verdict**: This is what we'd be reimplementing the output side of. It's the gold standard
  but its Python-only nature is the entire reason the gap exists.

### 2. SwiftCoreMLTools (Swift) — the only non-Python attempt
- **What it does**: Programmatically create CoreML neural network models in Swift
- **Language**: Swift
- **Stars**: ~162 GitHub
- **Strengths**: No Python dependency, uses CoreML protobuf directly
- **Weaknesses**:
  - Neural networks only (no ML programs, no MIL)
  - Cannot convert from ONNX/PyTorch/TF
  - No optimization passes
  - Single maintainer, not actively maintained
  - No ANE-specific optimizations
- **Verdict**: Proof of concept that died. Validates the idea is useful but shows the scope
  required is much larger than what one person built for NN-only models.

### 3. Apple Create ML (Xcode)
- **What it does**: GUI/Swift API for training simple models (classifiers, regressors)
- **Strengths**: Fully native, Apple-supported
- **Weaknesses**: Cannot convert external models, limited model types, requires Xcode
- **Verdict**: Not a model compiler — it's a training tool. Doesn't compete.

---

## Tier 2: Adjacent Competitors (Non-CoreML Model Compilers)

### 4. Qualcomm AI Engine Direct SDK (`qairt-converter`)
- **What it does**: ONNX/TF → .dlc format for Qualcomm NPU
- **Language**: C++ CLI tool (no Python required for conversion!)
- **Strengths**: Native CLI, full pipeline, quantization, profiling
- **Weaknesses**: Qualcomm-only hardware, proprietary, not open source
- **Verdict**: **This is the closest analogue to what we'd build, but for a different chip.**
  Qualcomm proved the model works: native CLI tool, no Python, targets their NPU.
  Apple has nothing equivalent. We'd fill that void for 2B Apple devices.

### 5. Intel OpenVINO
- **What it does**: ONNX/TF/PyTorch → optimized IR for Intel CPU/GPU/VPU
- **Language**: C++/Python
- **Strengths**: Cross-format, extensive optimization passes, INT8 quantization
- **Weaknesses**: Intel-only, no Apple hardware support
- **Verdict**: Another validated model — hardware vendor provides native compiler.
  Apple is the outlier that only provides Python tools.

### 6. Apache TVM
- **What it does**: Universal ML compiler framework, ONNX/TF → optimized code for many targets
- **Language**: C++/Python
- **Apple support**: Metal GPU backend exists; **no ANE/CoreML target backend**
- **Strengths**: Extremely flexible, many targets, auto-tuning
- **Weaknesses**: Complex, no CoreML output, no ANE path
- **Verdict**: Could theoretically add a CoreML backend but hasn't in 7+ years.
  Our project could complement TVM by being the CoreML emission layer.

### 7. Google IREE (via MLIR)
- **What it does**: MLIR-based end-to-end compiler → optimized binaries for CPU/GPU/NPU
- **Language**: C++
- **Apple support**: Metal backend (GPU); MPS dialect RFC exists; **no ANE target**
- **Strengths**: Modular, production-quality, strong community
- **Weaknesses**: No CoreML output, no ANE path, Rust not first-class
- **Verdict**: Like TVM — works at a lower level but doesn't bridge to CoreML.
  An MPS dialect is in RFC stage but still targets GPU, not ANE.

---

## Tier 3: Rust Ecosystem (Inference-only, no compilation)

### 8. `coreml-native` (Rust)
- **What it does**: Safe Rust bindings to CoreML runtime (load .mlmodelc, run inference)
- **Stars**: <50
- **Strengths**: Pure Rust, async, zero-copy tensors
- **Weaknesses**: Runtime only — cannot create, convert, or optimize models
- **Verdict**: Complementary, not competitive. Our project would use this (or something like it)
  as its runtime layer.

### 9. `candle` + `candle-coreml` (Rust, HuggingFace)
- **What it does**: ML framework + CoreML inference bridge
- **Strengths**: Active, large community, Metal GPU support
- **Weaknesses**: No model conversion to CoreML, no ANE optimization
- **Verdict**: Potential consumer of our project. Candle generates models but can't emit CoreML.

### 10. `burn` (Rust)
- **What it does**: ML framework with ONNX import, multiple backends
- **Strengths**: Growing, ONNX→Rust code generation
- **Weaknesses**: No CoreML output backend, no ANE path
- **Verdict**: Another potential consumer. Burn could add a CoreML backend using our crate.

### 11. `ort` / ONNX Runtime (Rust FFI)
- **What it does**: ONNX inference with execution providers (including CoreML EP)
- **Strengths**: Mature, cross-platform, CoreML EP exists in C++
- **Weaknesses**: CoreML EP is C++ not exposed to Rust, no model optimization/conversion
- **Verdict**: Alternative approach (keep model as ONNX, run via CoreML EP). But the EP has
  limited optimization and no Rust-native control.

### 12. `tract` (Rust)
- **What it does**: ONNX/TF inference engine, pure Rust
- **Strengths**: No C dependencies, production-used (Snips/Sonos)
- **Weaknesses**: CPU-only on Apple, no Metal/CoreML/ANE path
- **Verdict**: Could benefit from our project as an ANE acceleration path.

---

## Tier 4: Research / Reverse Engineering (Different approach entirely)

### 13. Orion (C/C++)
- **What it does**: Direct ANE programming via reverse-engineered private APIs
- **Strengths**: True hardware access, training support, ~19 TFLOPS
- **Weaknesses**: Private APIs — may break on any OS update, not App Store safe
- **Verdict**: Impressive research, terrible production strategy. Our project uses public
  APIs and is update-safe. Different risk/reward profile entirely.

### 14. ANEMLL (Python/Swift/C++)
- **What it does**: LLM optimization for CoreML/ANE deployment
- **Strengths**: Working LLM inference on ANE, open source
- **Weaknesses**: Python conversion pipeline, CoreML constraints apply
- **Verdict**: Uses coremltools internally. Would benefit from our project replacing
  its Python dependency.

---

## The Landscape Matrix

```
                    Can CREATE CoreML models?
                    Yes                         No
                ┌───────────────────────┬──────────────────────┐
   Python       │ coremltools (Apple)   │                      │
                │ onnx-coreml (archived)│                      │
                │ HF exporters          │                      │
                ├───────────────────────┼──────────────────────┤
   Swift        │ SwiftCoreMLTools (dead│ Create ML (training) │
                │   162⭐, NN-only)     │                      │
                ├───────────────────────┼──────────────────────┤
   C/C++        │                       │ TVM, IREE, Orion     │
                │      EMPTY            │ whisper.cpp, llama.cpp│
                ├───────────────────────┼──────────────────────┤
   Rust         │                       │ candle, burn, tract   │
                │      EMPTY            │ coreml-native, ort    │
                ├───────────────────────┼──────────────────────┤
   Go/Java/     │                       │ ONNX Runtime bindings │
   Other        │      EMPTY            │                      │
                └───────────────────────┴──────────────────────┘
```

**The entire "Can CREATE CoreML models" column outside Python is essentially empty.**
SwiftCoreMLTools is dead (162 stars, single maintainer, NN-only, no conversion).

---

## Key Technical Enabler

Apple publishes the CoreML protobuf schemas under BSD-3 license in
`apple/coremltools/mlmodel/format/`. This means:
- The `.proto` files can be compiled with `prost` (Rust protobuf) to generate Rust types
- The format is documented and stable
- No reverse engineering needed for the serialization layer

The hard part isn't the format — it's the **MIL IR semantics, optimization passes, and
op-mapping logic** that `coremltools` has built up over years.

---

## Honest Assessment: What's Novel vs. What's Hard

| Aspect | Novel? | Hard? | Notes |
|---|---|---|---|
| Non-Python CoreML model generation | ✅ Yes — no one has done this meaningfully | Medium | Protobuf schema is open; SwiftCoreMLTools proved partial feasibility |
| Rust specifically | ✅ Yes — zero Rust tools exist | Medium | Rust protobuf tooling is mature (prost) |
| ONNX → CoreML conversion | ❌ No — coremltools does this | Hard | 200+ op mappings, edge cases, numerical correctness |
| MIL IR implementation | ❌ No — coremltools has this | Hard | Graph IR with type system, passes infrastructure |
| ANE-targeted optimizations | Partially novel | Very hard | Apple's internal heuristics are undocumented; community knowledge is scattered |
| Full pipeline (convert + optimize + run) | ✅ Novel outside Python | Hard | Integration testing across model zoo is huge |

### Reality check on the problem statement

Some claims from the initial research were overstated:

| Original claim | Reality |
|---|---|
| "Models can't access ANE from Rust" | **Wrong.** `coreml-native` loads `.mlmodelc` and runs on ANE today. |
| "ANE is 80x more efficient" | **Misleading.** That's vs data-center A100. Vs local Metal GPU: ~1.5-2x. |
| "38 TOPS of stranded hardware" | **Overstated.** CoreML uses ANE automatically for supported ops. |
| "No way to convert models without Python" | **True.** This is the real, verified gap. |
| "Metal GPU is leaving performance on the table" | **Mostly false.** Metal GPU is within 20-33% of ANE for speed. ANE wins on power (40-65% less). |

The project is justified, but by the **toolchain argument** (Python-free conversion)
and the **shipped-app argument** (power efficiency, GPU freedom), not by dramatic
performance gaps.

---

## Risk Assessment

### Why this could fail:
1. **Scope creep**: coremltools has 200+ op mappings. Getting to 80% coverage is a multi-year effort.
2. **Apple changes formats**: New CoreML versions could break compatibility (mitigated: protobuf is stable).
3. **WWDC surprise**: Apple could release native Swift/C++ model tools (possible but they haven't in 8 years).
4. **Adoption**: Rust ML ecosystem is small; the audience might not be large enough.

### Why this could succeed:
1. **Clear gap**: Every search confirms "no non-Python converter exists."
2. **Precedent**: Qualcomm and Intel both offer native CLI tools — Apple is the outlier.
3. **Growing demand**: Rust desktop apps (Tauri), CLI tools, and on-device AI are booming.
4. **80/20 rule**: Supporting the top 30 ONNX ops covers most practical models.
5. **Protobuf is open**: The hardest part (format) is documented and stable.

---

## Refined Recommendation

Given this analysis, the project scope should be **narrower and more opinionated** than
originally proposed. Instead of a general-purpose "coremltools in Rust," consider:

### Option A: `coreml-kit` — Focused Model Compiler
**Target**: ONNX → CoreML with ANE optimization, CLI + library
**Scope**: Top 50 ONNX ops, FP16 quantization, basic op fusion
**Differentiator**: `cargo install coreml-kit && coreml-kit compile model.onnx`
**Audience**: Rust developers shipping on-device inference

### Option B: `mil-rs` — CoreML IR Library (Lower-level, wider impact)
**Target**: MIL IR data structures + CoreML protobuf reader/writer in Rust
**Scope**: Foundation crate that others build on (converters, optimizers, runtimes)
**Differentiator**: The "serde of CoreML" — doesn't do conversion itself but enables it
**Audience**: Anyone building CoreML tooling in Rust (or via FFI from C/C++/Swift)

### Option C: Hybrid — Start with B, build A on top
**Phase 1**: `mil-rs` crate (IR + protobuf, small, useful immediately)
**Phase 2**: `coreml-kit` CLI using `mil-rs` + ONNX reader + basic passes
**Phase 3**: ANE optimizations, quantization, ecosystem integrations

**Option C is strongest.** It creates immediate value (Phase 1 is useful on its own),
builds credibility, and the layered approach manages scope risk.
