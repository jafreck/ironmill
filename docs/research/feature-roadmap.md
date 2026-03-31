# Ironmill Feature Roadmap

A forward-looking plan for capabilities that make ironmill the definitive
Rust-native ML compiler for Apple Silicon. Organized into three horizons by
maturity and effort.

---

## Horizon 1 - Strengthen the Core

Deepen what ironmill already does well. These features build directly on
existing infrastructure with moderate effort.

### 1.1 Pass-by-Pass Pipeline Introspection

Make the optimization pipeline observable rather than a black box.

**Goal:** Every `ironmill compile` run can report exactly what each pass did.

```
$ ironmill compile model.onnx --verbose-passes

Pass  1/17  DeadCodeElimination      1247 → 1205 ops  (−42)
Pass  2/17  ConstantFold              1205 → 1187 ops  (−18 constants folded)
Pass  3/17  ConvBatchNormFusion       1187 → 1163 ops  (24 conv+bn fused)
Pass  4/17  AttentionFusion           1163 → 1091 ops  (9 attention blocks fused)
...
Total: 1247 → 891 ops · est. 2.3× speedup · 38% size reduction
```

Deliverables:
- Per-pass op count, memory estimate, and estimated latency delta
- `--dump-ir-after <pass-name>` to export intermediate MIL for debugging
- JSON-structured pass report for CI integration (`--pass-report pass_report.json`)

**Why this matters:** Every ML compiler today is opaque. Transparent
optimization builds trust and enables users to tune their pipeline config
based on data, not guesswork.

---

### 1.2 Quantization Quality Explorer

Automate the quality-vs-size-vs-speed tradeoff search that users currently do
manually.

```
$ ironmill explore model.onnx \
    --sweep fp16,int8,polar-4,polar-2,palettize-4 \
    --metric perplexity,size,latency \
    --calibration-dir ./cal_data

  Config         Size     Latency   Perplexity   Δ vs FP32
  ─────────────  ───────  ────────  ──────────   ─────────
  fp16           1.2 GB   4.2 ms   5.31         +0.02
  int8           680 MB   3.1 ms   5.44         +0.15
  polar-4        420 MB   3.4 ms   5.38         +0.09  ← Pareto optimal
  polar-2        240 MB   3.8 ms   5.92         +0.63
  palettize-4    380 MB   3.3 ms   5.41         +0.12

  Per-layer sensitivity:
  Layer 14 (self_attn.q_proj): high sensitivity - keep ≥ INT8
  Layer 27 (mlp.gate_proj):    low sensitivity  - safe at 2-bit
```

Deliverables:
- `ironmill explore` subcommand with `--sweep` and `--metric` flags
- Per-layer sensitivity report identifying precision-critical layers
- Pareto frontier visualization (text table, optional JSON/CSV export)
- Auto-generate optimal `--pipeline-config` JSON from sweep results

---

### 1.3 Graph Visualization Export

Add structured graph output to the existing `inspect` command.

```
$ ironmill inspect model.onnx --format dot > model.dot
$ ironmill inspect model.onnx --format mermaid
$ ironmill inspect model.onnx --format html --open
```

Deliverables:
- Graphviz DOT export with compute-unit color coding (ANE / GPU / CPU)
- Mermaid diagram for embedding in markdown docs
- Self-contained HTML viewer (single-file, no server) with:
  - zoom/pan navigation for large graphs
  - op metadata on hover (shape, dtype, quantization status)
  - before/after toggle showing graph pre- and post-optimization
- Highlight quantized vs full-precision ops with distinct node styles

---

### 1.4 ANE Pre-Flight Validator with Auto-Fix Suggestions

Extend the existing `validate` command from "pass/fail" to "here's how to
fix it."

```
$ ironmill validate model.mlpackage --format text

  ✗ Layer 14: matmul output shape [1, 32, 4096, 128] exceeds ANE tile limit
    → Suggestion: enable --op-split to partition into 2 tiles

  ✗ Layer 22: gather op not supported on ANE
    → Suggestion: will fall back to CPU (0.3ms overhead estimated)

  ✓ 27/29 layers fully ANE-eligible
  ✓ Estimated ANE utilization: 94%
```

Deliverables:
- Actionable fix suggestions for each failed validation check
- Estimated compute-unit map (which ops run on ANE / GPU / CPU)
- Estimated ANE utilization percentage
- Absorb documented ANE hardware constraints (SRAM limits, compile limits,
  concat rejection, SDPA mask behavior, IOSurface sizing, etc.)

---

## Horizon 2 - Expand the Platform

New capabilities that extend ironmill beyond a converter into a full model
engineering toolkit.

### 2.1 Model Surgery CLI

Enable model composition and decomposition as first-class operations.

```
$ ironmill split model.onnx \
    --draft-layers 0-11 --draft-output draft.mlpackage \
    --verifier-output verifier.mlpackage

$ ironmill merge encoder.onnx decoder.onnx \
    --output full_pipeline.mlpackage

$ ironmill extract model.onnx \
    --layers "model.layers.0-15" \
    --output first_half.mlpackage
```

Deliverables:
- **`split`** - partition a model into draft/verifier pairs for speculative
  decoding, or pre-attention/post-attention halves for ANE execution
- **`merge`** - combine multiple models into a single CoreML pipeline with
  defined I/O connections
- **`extract`** - pull a subgraph (by layer name or index range) into a
  standalone model
- **`swap`** - replace a subgraph (e.g., swap a classification head)
- Build on existing `ModelSplitPass` and `compile-pipeline` infrastructure

**Why this matters:** Model composition is manual and error-prone today.
Making it a CLI operation enables speculative decoding, ensemble deployment,
and modular model architectures without custom scripts.

---

### 2.2 Model Diff and Regression Tool

Compare two models structurally, numerically, and performance-wise.

```
$ ironmill diff model_v1.mlpackage model_v2.mlpackage

  Structural changes:
    + Added 2 layers (model.layers.28, model.layers.29)
    ~ Changed layer 14 quantization: FP16 → INT8
    − Removed 1 skip connection

  Weight changes:
    model.layers.0.self_attn.q_proj: max Δ = 0.003, mean Δ = 0.0001
    model.layers.14.mlp.gate_proj:   max Δ = 1.24  ← significant

  Performance (requires --benchmark):
    v1: 4.2 ms/token, 1.2 GB    v2: 3.1 ms/token, 680 MB
```

Deliverables:
- Structural diff: added / removed / changed ops and layers
- Weight diff: per-layer statistical summary (max, mean, histogram)
- Optional performance diff via benchmark harness
- Machine-readable JSON output for CI/CD regression gates

---

### 2.3 ONNX Export (Cross-Platform Compiler Mode)

*See [issue #3](https://github.com/jafreck/ironmill/issues/3) for full spec.*

Enable ironmill as a general-purpose optimization compiler, not just a
CoreML converter.

```
$ ironmill compile model.onnx --output optimized.onnx --format onnx
```

Deliverables:
- MIL → ONNX reverse mapper (`writer/onnx.rs`)
- CoreML-specific op decomposition (fused attention → matmul+softmax, LUT
  weights → dense)
- `--format` CLI flag (`coreml` default, `onnx` new)
- Round-trip numerical equivalence tests
- Enables deployment to ONNX Runtime, TensorRT, OpenVINO, etc.

---

### 2.4 Swift / Xcode Integration Package

Expand the C API into a native Swift developer experience.

```swift
import IronmillKit

let model = try Ironmill.compile(
    "model.onnx",
    quantization: .polar(bits: 4),
    computeUnits: .cpuAndNeuralEngine
)

let prediction = try model.predict(["input": inputTensor])
```

Deliverables:
- Swift package wrapping the C API with idiomatic Swift types
- Xcode Build Phase plugin: add an ONNX file to your project, get a compiled
  `.mlmodelc` at build time
- SPM (Swift Package Manager) distribution
- Go bindings for server-side CoreML compilation

**Why this matters:** coremltools requires Python. There is no way to compile
ML models from Swift or Xcode natively today. This makes ironmill the only
option for pure-Apple toolchain workflows.

---

## Horizon 3 - Define the Category

Ambitious capabilities that establish ironmill as a new kind of tool in the
ML ecosystem.

### 3.1 Metal Direct Backend

A runtime that executes ironmill's optimized IR directly on the GPU via Metal,
without CoreML.

```
$ ironmill compile model.onnx --runtime metal-direct
```

Deliverables:
- `crates/ironmill-metal/` - new crate implementing `RuntimeBackend` trait
- MIL IR → MPSGraph lowering for standard ops
- Custom Metal compute shaders for fused attention and quantized matmul
- Leverage Apple Silicon unified memory for near zero-copy
- Performance parity with or better than llama.cpp's Metal backend (ironmill
  has the compiler advantage - optimized graph + direct GPU)

**Why this matters:** CoreML is a black box that decides how to schedule ops.
Metal direct gives full control. Combined with ironmill's 34 optimization
passes, this is optimized-graph + direct-GPU - a combination nobody else has.

---

### 3.2 Adaptive Pipeline Autotuner

Use ironmill's benchmark harness to automatically discover the optimal
pipeline configuration for a given model + target hardware.

```
$ ironmill autotune model.onnx --target m4-max --budget 500mb

  Searched 847 configurations in 12 minutes
  Optimal: polar-4 + attention-fusion + layout-opt + ANE-schedule
  Speedup: 3.1× over default pipeline
  Saved: autotune_config.json
```

Deliverables:
- Exhaustive or guided search over pass combinations, quantization configs,
  and compute-unit assignments
- Hardware-aware cost model using benchmark telemetry
- Export the winning config as a reusable `--pipeline-config` JSON
- CI mode: verify that a model meets latency/size budgets

---

### 3.3 Live Model Profiler

Runtime profiling that maps performance data back to the model graph.

```
$ ironmill profile model.mlpackage --iterations 100

  Hotspot analysis:
  Layer 14 self_attn     1.8 ms  (43% of total)  - running on GPU (ANE fallback)
  Layer 22 mlp.down_proj 0.6 ms  (14% of total)  - running on ANE
  ...

  Recommendations:
  → Layer 14: shape [1,32,4096,128] causes ANE fallback. Split via --op-split
    to recover ~0.9 ms (estimated 22% speedup).
```

Deliverables:
- Per-op latency attribution using CoreML's profiling APIs
- Compute-unit actual assignment map (where did each op really run?)
- Automated optimization recommendations based on profiling results
- Flame-graph style visualization export

---

### 3.4 Federated Model Compilation

Compile and optimize models for a fleet of Apple devices with different
capabilities.

```
$ ironmill compile model.onnx \
    --targets m1,m2-pro,m4-max,a17-pro \
    --output-dir ./compiled_models/

  Generated:
    compiled_models/m1/model.mlmodelc         (INT8, 680 MB)
    compiled_models/m2_pro/model.mlmodelc     (polar-4, 420 MB)
    compiled_models/m4_max/model.mlmodelc     (FP16, 1.2 GB)
    compiled_models/a17_pro/model.mlmodelc    (INT8 + op-split, 680 MB)
    compiled_models/manifest.json
```

Deliverables:
- Per-target hardware profiles (ANE capabilities, memory budget, compute units)
- Automatic quantization and pass selection per target
- Manifest file for runtime target selection in apps
- Integration with App Store asset catalogs for on-demand model delivery

---

## Priority Summary

| Feature | Horizon | Effort | Impact | Builds On |
|---|---|---|---|---|
| Pass introspection (§1.1) | 1 | Low | 🔥🔥 | `PassPipeline` |
| Quantization explorer (§1.2) | 1 | Medium | 🔥🔥🔥 | `ironmill-bench` |
| Graph visualization (§1.3) | 1 | Low | 🔥🔥 | `inspect` CLI |
| ANE validator + auto-fix (§1.4) | 1 | Medium | 🔥🔥 | `validate` CLI |
| Model surgery CLI (§2.1) | 2 | Medium | 🔥🔥🔥 | `ModelSplitPass`, `compile-pipeline` |
| Model diff (§2.2) | 2 | Medium | 🔥🔥 | `pipeline-report` |
| ONNX export (§2.3) | 2 | High | 🔥🔥🔥 | Issue #3 |
| Swift/Xcode package (§2.4) | 2 | Medium | 🔥🔥🔥 | C API |
| Metal direct backend (§3.1) | 3 | High | 🔥🔥🔥 | `RuntimeBackend` trait |
| Adaptive autotuner (§3.2) | 3 | High | 🔥🔥🔥 | `ironmill-bench` |
| Live profiler (§3.3) | 3 | High | 🔥🔥 | `ironmill-bench` |
| Federated compilation (§3.4) | 3 | High | 🔥🔥 | `CompileBuilder` |
