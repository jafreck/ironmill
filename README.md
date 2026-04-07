<div align="center">

# ⚙️ ironmill

[![Build](https://github.com/jafreck/ironmill/actions/workflows/ci.yml/badge.svg)](https://github.com/jafreck/ironmill/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macOS_Apple_Silicon-lightgrey.svg)]()

Rust-native model compiler and inference runtime for Apple Silicon.

</div>

> [!WARNING]
> **ironmill is currently in alpha and under active development.** APIs, file
> formats, and CLI interfaces may change without notice. Not recommended for
> production use.

ironmill compiles ML models into optimized bundles and runs them on Apple
Silicon hardware. It supports multiple input formats (ONNX, SafeTensors,
GGUF, CoreML), applies optimization and quantization passes via an
intermediate representation ([MIL](https://apple.github.io/coremltools/docs-guides/source/mil-program.html)),
and produces artifacts for three inference backends: Metal (GPU/MPS), CoreML,
and an experimental direct-ANE backend built on reverse-engineered private APIs.

## Features

### Compiler

Imports models, applies optimization passes, and outputs backend-specific
bundles.

#### Model Import

- **ONNX** (`.onnx`) via MIL conversion
- **SafeTensors** directories with architecture auto-detection from `config.json`
- **GGUF** quantized model files
- **CoreML** (`.mlmodel` / `.mlpackage`) with full MIL round-trip

#### Architecture Templates

Built-in templates generate MIL programs from SafeTensors/GGUF weights for
supported architectures:

| Architecture | Variants | Notes |
|---|---|---|
| LLaMA | LLaMA 2/3, CodeLlama, Mistral | ANE lowering with static KV cache, 1×1 conv projections |
| Qwen | Qwen, Qwen 2/3 | Q/K/V biases, sliding-window metadata |
| Qwen 3.5 | Qwen 3.5 | GDN layers, partial rotary (25%), attention output gate |
| Gemma | Gemma 2/3/4 | Embedding scaling, PLE, MoE, KV-shared layers, logit softcapping |

#### General Optimization Passes

| Pass | Description |
|---|---|
| Dead code elimination | Removes unreachable ops from the graph |
| Constant folding | Evaluates compile-time constant expressions |
| Identity elimination | Strips no-op identity operations |
| Op fusion | Conv+BatchNorm, Conv+ReLU, Linear+ReLU, SDPA fusion |

#### Weight Quantization

| Method | Bits | Description |
|---|---|---|
| FP16 | 16 | Half-precision float conversion |
| INT8 | 8 | Symmetric weight-only quantization |
| INT4 | 4 | Affine per-group quantization |
| PolarQuant | 2/4 | LUT-based quantization with codebook indices |
| D2Quant | 2/3 | Dual-scale quantization (coarse + fine scales) |
| AWQ | 4 | Activation-aware quantization with calibration data |
| GPTQ | 4 | Post-training quantization with calibration (requires `gptq` feature) |
| QuIP# | 2 | E8 lattice quantization with Hadamard rotation |
| Palettization | 2/4/6/8 | Weight clustering with shared codebooks |

#### ANE Lowering Passes

| Pass | Description |
|---|---|
| `AneMatmulToConvPass` | Lowers matmul to 1×1 convolution for ANE execution |
| `AneLayoutPass` | Reshapes tensors to ANE-native `[1, C, 1, S]` layout |
| `OpSubstitutionPass` | Rewrites ops for ANE (GELU expansion, SiLU fusion, rsqrt rewrite) |
| `AttentionDecomposePass` | Decomposes SDPA into matmul/mask/softmax/matmul |
| `OpSplittingPass` | Tiles oversized matmul/linear/conv ops to fit ANE memory budget |
| `ModelSplitPass` | Splits model into draft/verifier programs for speculative decoding |
| `KvCachePass` | Inserts KV-cache read/update ops with static cache shapes |
| `ComputeUnitAnnotationPass` | Annotates ops with preferred compute unit (ANE/GPU/CPU) |
| `LayerSchedulePass` | Layer-type aware scheduling (conv/attention/ffn/norm) |
| `MixedPrecisionPass` | Per-op FP16/INT8 precision assignment |
| `PerExpertQuantPass` | Per-expert quantization for MoE models |
| `CodebookOptimizationPass` | Fuses RVQ codebook gather+sum, adds palettize hints |

#### Output Formats

- `.mlpackage` for CoreML (with optional `xcrun coremlcompiler` compilation)
- `.ironml` bundles for ANE Direct (MIL text + weight blobs)
- `.ironml-gpu` bundles for Metal GPU

### Inference Runtime

Three backends for running compiled models, each targeting different
hardware:

| | Metal GPU | CoreML | ANE Direct |
|---|:---:|:---:|:---:|
| Autoregressive decode | ✅ | — | ✅ |
| Custom compute kernels | ✅ | — | ✅ |
| INT8 KV cache (TurboQuant) | ✅ | — | ✅ |
| Hardware scheduling | ironmill | Apple | ironmill |
| Backing tech | Metal / MPS | CoreML (CPU, GPU, ANE) | ANE private API |

#### Metal GPU

Primary backend for LLM inference on Apple Silicon GPUs.

**Fused Compute Kernels:**

| Kernel | Description |
|---|---|
| `fused_residual_norm` | Residual add + RMSNorm in a single dispatch |
| `fused_residual_norm_matvec` | Residual + RMSNorm + matrix-vector projection |
| `fused_qk_norm_rope` | QK normalization + partial RoPE (supports Qwen 3.5 25% rotary) |
| `fused_embedding_norm` | Token embedding lookup + RMSNorm |
| `fused_sdpa` | FlashAttention-style SDPA with online softmax and conditional rescaling |
| `fused_softcap` | Logit softcapping (Gemma) |
| `gdn_fused_decode` | GDN conv1d + SiLU + recurrent update + output gate in one dispatch |
| `gdn_batched_matvec` | 4 GDN dense projections in a single dispatch |
| `batched_affine_matvec` | FFN gate + up projection fused for INT4/INT8 |

**Attention:**

- Standard FP16 single-step decode attention
- FlashAttention-2 prefill with tiled Q/KV and online softmax
- Sliding-window attention with ring-buffer KV cache
- CLA (Cross-Layer Attention) with anchor-layer KV reuse
- Per-layer head_dim and kv_head configuration (Gemma 4)

**Weight Quantization at Runtime:**

| Format | Description |
|---|---|
| FP16 | Half-precision with blocked layout for GPU cache efficiency |
| INT4 affine | Per-group quantization with optional AWQ activation scales |
| INT8 affine | Symmetric per-channel quantization |
| D2Quant | 3-bit dual-scale (coarse + fine) quantization |
| PolarQuant | LUT-based dequantization via codebook indices |
| QuIP# | E8 lattice quantization with Hadamard rotation |
| Q8 input | On-the-fly input activation quantization for INT4×Q8 decode |

**KV Cache Quantization ([TurboQuant](docs/design/turboquant.md)):**

- INT4 and INT8 cache compression with outlier-aware quantization
- QJL (Quantized Johnson-Lindenstrauss) random projection variant
- Fused quantize-on-write and dequantize-on-read shaders
- Sliding-window ring buffers with quantized storage

**Architecture-Specific Optimizations:**

| Architecture | Optimizations |
|---|---|
| Gemma 4 | Per-layer head_dim/kv_heads, global + sliding window layers, PLE, MoE routing, V-norm, logit softcapping |
| Qwen 3.5 | GDN (Gated Delta Network) linear attention layers, centered RMSNorm, attention output gate, partial rotary (25%) |
| DeepSeek V2/V3 | MLA (Multi-Latent Attention) with weight absorption, compressed KV cache (latent + RoPE key) |
| LLaMA/Qwen | Standard GQA attention, SiLU-gated FFN |

**Memory Optimizations:**

- Tied embedding reuse — shares `lm_head` weights with token embeddings
- Blocked weight packing — cache-friendly layout for GPU matvec
- Buffer compaction — drops redundant weight copies after load transforms
- Private vs shared GPU buffers — GPU-only storage for weights, CPU-visible for I/O
- On-demand activation buffer growth — allocates as needed during prefill
- Prefix-cache radix tree with LRU eviction

**Additional Features:**

- Speculative decoding with draft + verifier models
- Grammar-constrained decoding
- Batch inference runner
- Precompiled Metal shader library with runtime fallback for large head_dim

#### CoreML

Wraps Apple's CoreML runtime (`MLModel`). Loads compiled `.mlmodelc`
packages and delegates hardware scheduling (ANE/GPU/CPU) to Apple's
runtime. Supports model loading and prediction — no LLM-specific decode
loop or KV cache management.

#### ANE Direct *(experimental)*

Bypasses CoreML to talk directly to the Neural Engine using
reverse-engineered private APIs (`_ANECompiler`, `_ANEInMemoryModel`).
Loads pre-compiled `.ironml` bundles:

- IOSurface-backed tensor I/O for ANE-compatible memory layout
- [TurboQuant](docs/design/turboquant.md): INT8 KV cache compression with
  Hadamard rotation and on-ANE dequantization
- Autoregressive decode loop with ANE-accelerated lm_head via chunked conv1×1

## Usage

### CLI

```bash
cargo install --path crates/ironmill-cli
```

#### Commands

```
COMMANDS:
  compile           Compile a model to CoreML, ANE, or Metal format
  inspect           Print model structure and metadata
  validate          Validate model for target hardware compatibility
  compile-pipeline  Compile a multi-stage pipeline from a TOML manifest
  pipeline-report   Compare two pipeline configurations
```

#### `compile`

Convert an ONNX, SafeTensors, GGUF, or CoreML model to an optimized output format.

Key flags:

| Flag | Description |
|------|-------------|
| `-o, --output <PATH>` | Output path (default: derived from input) |
| `-t, --target <TARGET>` | Compute units: `all`, `cpu-only`, `cpu-and-gpu`, `cpu-and-ne`, `gpu` |
| `-q, --quantize <MODE>` | Quantization: `none`, `fp16`, `int8`, `mixed-fp16-int8`, `awq`, `int4`, `gptq`, `d2quant` |
| `--cal-data <DIR>` | Calibration data directory (for INT8, AWQ, or GPTQ) |
| `--polar-quantize <BITS>` | PolarQuant weight quantization (2 or 4 bit) |
| `--palettize <BITS>` | Weight palettization bit-width (2, 4, 6, or 8) |
| `--quip-sharp` | QuIP# (E8 lattice) 2-bit weight quantization |
| `--input-shape <NAME:SHAPE>` | Set concrete input shape for ANE (repeatable) |
| `--ane` | Emit ANE-optimized ops (1×1 conv projections, decomposed RMSNorm, etc.) |
| `--ane-memory-budget <SIZE>` | ANE memory budget per op (e.g. `1GB`, `512MB`) |
| `--runtime <BACKEND>` | Runtime backend: `coreml` (default), `ane-direct` (experimental) |
| `--kv-quant <MODE>` | KV cache quantization: `none`, `turbo-int8` |
| `--pipeline-config <PATH>` | TOML pipeline configuration (overrides default passes) |
| `--no-fusion` | Disable fusion and optimization passes |
| `--moe-split` | Split MoE model into per-expert `.mlpackage` files |
| `--moe-bundle` | Bundle MoE experts as functions in a single `.mlpackage` |
| `--split-draft-layers <N>` | Split model for speculative decoding (draft + verifier) |
| `--annotate-compute-units` | Annotate ops with preferred compute unit (ANE/GPU/CPU) |

#### `inspect`

Print model structure and metadata for `.onnx`, `.mlmodel`, or `.mlpackage` files.

#### `validate`

Check whether a model is compatible with the Apple Neural Engine.

| Flag | Description |
|------|-------------|
| `--format <FMT>` | Output format: `text` (default) or `json` |

#### `compile-pipeline`

Compile a multi-ONNX pipeline from a TOML manifest into coordinated `.mlpackage` outputs.

| Flag | Description |
|------|-------------|
| `-o, --output <DIR>` | Output directory for `.mlpackage` files and `pipeline.json` |

#### `pipeline-report`

Compare two pipeline configurations on a model and report metrics.

| Flag | Description |
|------|-------------|
| `--config-a <PATH>` | Path to the first pipeline config (TOML) |
| `--config-b <PATH>` | Path to the second pipeline config (TOML) |

#### Examples

```bash
# Basic CoreML conversion
ironmill compile model.onnx

# FP16 quantization with explicit output path
ironmill compile model.onnx -o output.mlpackage --quantize fp16

# Weight-only INT8 quantization
ironmill compile model.onnx --quantize int8

# Fixed input shapes for ANE compatibility
ironmill compile model.onnx --input-shape "input:1,3,224,224"

# 4-bit PolarQuant for Metal GPU backend
ironmill compile model.onnx --target gpu --polar-quantize 4

# ANE-optimized compile with TurboQuant KV cache
ironmill compile model.onnx --ane --kv-quant turbo-int8

# Compile a multi-stage pipeline
ironmill compile-pipeline pipeline.toml -o out/

# Compare two pipeline configs
ironmill pipeline-report model.onnx --config-a fast.toml --config-b accurate.toml

# Inspect model structure
ironmill inspect model.onnx

# Validate ANE compatibility (JSON output)
ironmill validate model.onnx --format json
```

### Rust API — High-Level (`ironmill-torch`)

`ironmill-torch` provides a PyTorch-level abstraction over the lower-level
compile and inference crates.

#### Quick Start

```rust
use ironmill_torch::{Model, GenParams};

let mut model = Model::from_pretrained("./Qwen3-0.6B/")
    .build()?;

let output = model.generate("What is Rust?", &GenParams::default())?;
println!("{}", output.text);
```

#### Model Loading

```rust
use ironmill_torch::{Model, Device};

// Load from a SafeTensors directory (auto-detects architecture)
let mut model = Model::from_pretrained("./model/")
    .device(Device::Metal)
    .max_seq_len(8192)
    .build()?;

// Or load from a pre-compiled .ironml-gpu bundle
let mut model = Model::from_compiled("model.ironml-gpu")
    .build()?;
```

#### Streaming Generation

```rust
use ironmill_torch::{Model, GenParams};

let mut model = Model::from_pretrained("./model/").build()?;
let params = GenParams::default()
    .with_temperature(0.8)
    .with_max_tokens(256)
    .with_top_p(0.95);

for chunk in model.stream("Once upon a time", &params)? {
    let chunk = chunk?;
    print!("{}", chunk.text);
}
```

#### Multi-Turn Chat

```rust
use ironmill_torch::{Model, GenParams};

let mut model = Model::from_pretrained("./model/").build()?;
let mut chat = model.chat()
    .system("You are a helpful assistant.")
    .params(GenParams::default().with_temperature(0.7))
    .build();

let reply = chat.send("What is Rust?")?;
println!("{}", reply.text);

let followup = chat.send("How does its borrow checker work?")?;
println!("{}", followup.text);

// Streaming chat
let stream = chat.send_stream("Tell me more")?;
let mut full_text = String::new();
for chunk in stream {
    let chunk = chunk?;
    full_text.push_str(&chunk.text);
    print!("{}", chunk.text);
}
chat.finish_stream(full_text);
```

#### GenParams

| Parameter | Default | Description |
|---|---|---|
| `temperature` | 0.7 | Sampling temperature (0.0 = greedy) |
| `max_tokens` | 512 | Maximum tokens to generate |
| `top_p` | 0.9 | Nucleus sampling threshold (1.0 = disabled) |
| `top_k` | 0 | Top-k filtering (0 = disabled) |
| `min_p` | 0.0 | Min-p threshold (0.0 = disabled) |
| `stop_tokens` | `[]` | Token IDs that signal end of generation |

### Rust API — Compilation

Compile an ONNX model to a CoreML package:

```rust
use ironmill_compile::coreml::CompileBuilder;

let output = CompileBuilder::new("model.onnx")
    .quantize(Quantization::Fp16)
    .compile()          // run xcrun coremlcompiler
    .build()?;

// output.mlpackage, output.mlmodelc
```

Compile for Metal GPU with PolarQuant weight quantization:

```rust
use ironmill_compile::gpu::GpuCompileBuilder;
use ironmill_compile::gpu::bundle::write_gpu_bundle;

let provider = GpuCompileBuilder::new("model.onnx")
    .polar_quantize(4)  // 4-bit PolarQuant
    .build()?;

write_gpu_bundle(&provider, "model.ironml-gpu")?;
```

### Rust API — Inference

**Metal GPU** — load a compiled bundle and run autoregressive decoding:

```rust
use ironmill_inference::gpu::{GpuConfig, GpuInference};
use ironmill_inference::gpu::bundle::GpuBundleProvider;
use ironmill_inference::InferenceEngine;

let provider = GpuBundleProvider::open("model.ironml-gpu")?;
let config = GpuConfig::default();

let mut engine = GpuInference::new(config.clone())?;
engine.load_weights(&provider, config)?;

let logits = engine.prefill(&[1, 2, 3])?;          // prompt
let next_logits = engine.decode_step(token_id)?;    // autoregressive
```

**CoreML** — load a compiled model and run prediction:

```rust
use ironmill_inference::coreml_runtime::{ComputeUnits, Model, PredictionInput};

let model = Model::load("Model.mlmodelc".as_ref(), ComputeUnits::All)?;

let mut input = PredictionInput::new();
input.add_multi_array("input", &[1, 3, 224, 224], dtype, &data)?;

let output = model.predict(&input)?;
```

**ANE Direct** — load a bundle and generate tokens:

```rust
use std::sync::Arc;
use ironmill_inference::ane::decode::AneInference;
use ironmill_inference::ane::HardwareAneDevice;

let device = Arc::new(HardwareAneDevice::new()?);
let mut model = AneInference::from_bundle(device, "model.ironml".as_ref(), None)?;

let tokens = model.generate(&[1, 2, 3], 128, 0.8)?;
```

### C API & Framework Bridges

- **High-level API:** [`ironmill-torch`](crates/ironmill-torch/) — PyTorch-style model loading, generation, streaming, and chat
- **C API:** stable C ABI for Swift, C++, Go, or any FFI language ([docs](docs/C_API.md))
- **[candle-coreml](crates/candle-coreml/):** ONNX→CoreML conversion + runtime for [candle](https://github.com/huggingface/candle)
- **[burn-coreml](crates/burn-coreml/):** export + inference bridge for [Burn](https://github.com/tracel-ai/burn)

## Architecture

### Compilation Pipeline

```mermaid
graph TD
    onnx["ONNX (.onnx)"]
    coreml_in["CoreML (.mlmodel/.mlpackage)"]
    safetensors["SafeTensors"]
    gguf["GGUF"]

    mil["MIL Program<br/><i>Model Intermediate Language</i>"]

    onnx -->|mil-rs| mil
    coreml_in -->|mil-rs| mil
    safetensors -->|"architecture template"| mil
    gguf -->|"architecture template"| mil

    compile["ironmill-compile"]
    mil --> compile

    passes["Optimization Passes<br/><i>DCE · Constant fold · Op fusion<br/>FP16 · INT8 · PolarQuant</i>"]
    compile --> passes

    proto["CoreML Proto"]
    passes --> proto

    pkg[".mlpackage"]
    proto --> pkg

    xcrun["xcrun coremlcompiler"]
    pkg -.->|optional| xcrun

    mlmodelc[".mlmodelc<br/><i>Compiled model</i>"]
    xcrun -.-> mlmodelc

    gpu_bundle[".ironml-gpu bundle<br/><i>Weights + manifest</i>"]
    passes --> gpu_bundle

    ane["ANE Lowering<br/><i>MatMul→Conv · Op substitution<br/>Layout · Model split</i>"]
    passes --> ane

    bundle[".ironml bundle<br/><i>MIL text + weight blobs</i>"]
    ane --> bundle
```

### Crate Structure

```mermaid
%%{init: {'flowchart': {'curve': 'basis'}}}%%
flowchart TD
    subgraph user["User-Facing"]
        direction LR
        cli["ironmill-cli"]
        bench["ironmill-bench"]
        torch["ironmill-torch"]
        burn["burn-coreml"]
        candle["candle-coreml"]
    end

    subgraph core["Core"]
        direction LR
        compile["ironmill-compile"]
        inference["ironmill-inference"]
        corelib["ironmill-core"]
    end

    subgraph sys["System Bindings"]
        direction LR
        ane["ironmill-ane-sys"]
        ios["ironmill-iosurface"]
        coremlsys["ironmill-coreml-sys"]
        metalsys["ironmill-metal-sys"]
    end

    subgraph found["Foundation"]
        mil["mil-rs"]
    end

    cli --> compile
    bench --> compile
    bench --> inference
    bench -. "ane-direct" .-> ios
    torch --> compile
    torch --> inference
    burn --> compile
    burn -. "macos" .-> inference
    candle --> compile
    candle -. "macos" .-> inference
    compile --> corelib
    compile --> mil
    compile --> ios
    inference --> corelib
    inference --> mil
    inference --> ane
    inference --> ios
    inference --> coremlsys
    inference -. "metal" .-> metalsys
    corelib --> mil
    ios --> mil
```

| Crate | Description |
|-------|-------------|
| [`mil-rs`](crates/mil-rs/) | MIL IR library: read/write CoreML models, ONNX conversion, proto↔IR, pass pipeline |
| [`ironmill-core`](crates/ironmill-core/) | Shared types: bundle schemas, weight traits, model configs, MIL text emitter |
| [`ironmill-compile`](crates/ironmill-compile/) | Compiler: optimization passes, CoreML/ANE/GPU build APIs, weight providers |
| [`ironmill-inference`](crates/ironmill-inference/) | Inference: Metal GPU, CoreML, and ANE Direct backends |
| [`ironmill-torch`](crates/ironmill-torch/) | High-level PyTorch-style API: model loading, generation, streaming, chat |
| [`ironmill-ane-sys`](crates/ironmill-ane-sys/) | FFI bindings for ANE private APIs (macOS) |
| [`ironmill-iosurface`](crates/ironmill-iosurface/) | IOSurface tensor management for ANE I/O (macOS) |
| [`ironmill-coreml-sys`](crates/ironmill-coreml-sys/) | CoreML runtime bindings via objc2 (macOS) |
| [`ironmill-metal-sys`](crates/ironmill-metal-sys/) | Metal and MPS framework bindings (macOS) |
| [`ironmill-cli`](crates/ironmill-cli/) | CLI: `compile`, `inspect`, `validate`, `compile-pipeline`, `pipeline-report` |
| [`ironmill-bench`](crates/ironmill-bench/) | Benchmarks: latency, power, perplexity |
| [`ironmill-compile-ffi`](crates/ironmill-compile-ffi/) | C FFI for the compilation pipeline |
| [`candle-coreml`](crates/candle-coreml/) | [candle](https://github.com/huggingface/candle) bridge: ONNX→CoreML + runtime |
| [`burn-coreml`](crates/burn-coreml/) | [Burn](https://github.com/tracel-ai/burn) bridge: export + inference |

## ANE Research & Related Projects

Building on [prior art](#ane-related-projects) in ANE reverse-engineering, ironmill
contributes reproducible eval-verified tests for MIL ops on Apple's Neural
Engine:

- **38 newly verified ops** (33 eval-verified, 5 compile-verified) not
  confirmed by any other open-source project
- **The epsilon discovery:** `rsqrt`, `log`, and `inverse` require an
  undocumented `epsilon` parameter; without it the compiler silently rejects
  them. Previously believed hardware-unsupported.
- **`layer_norm` on ANE:** other projects perform normalization on CPU
- **`erf` on ANE:** enables on-ANE GELU without tanh decomposition
- **Full INT8 pipeline:** `quantize`/`dequantize`/`cast` verified for
  end-to-end INT8 KV cache on ANE
- **Comparison + conditional ops:** all 6 comparison ops plus
  `select`/`logical_not` verified, enabling conditional logic on ANE

Every finding has a reproducible eval test in
[`ane_op_eval.rs`](crates/ironmill-inference/examples/ane_op_eval.rs).
See the full [ANE Op Support Matrix](docs/design/ane-op-support-matrix.md).

### ANE Related Projects

Open-source projects working with the ANE via private APIs:

- [maderix/ANE](https://github.com/maderix/ANE): ANE reverse-engineering, hardware characterization, transformer training proof-of-concept
- [mechramc/Orion](https://github.com/mechramc/Orion): ANE LLM training and inference runtime with graph IR compiler ([paper](https://arxiv.org/abs/2603.06728))
- [vipuldivyanshu92/ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt): GPT-style transformer training on ANE
- [hollance/neural-engine](https://github.com/hollance/neural-engine): Community documentation of ANE capabilities

## Rust ML Ecosystem

ironmill sits alongside a growing ecosystem of Rust-native ML frameworks.
The candle and Burn bridge crates let you use ironmill's Metal GPU/CoreML
backends from models built in those frameworks.

| Project | Focus |
|---------|-------|
| [candle](https://github.com/huggingface/candle) | Lightweight ML framework with GPU support (candle-coreml bridge in this repo) |
| [Burn](https://github.com/tracel-ai/burn) | Modular deep learning framework with multiple backends (burn-coreml bridge in this repo) |
| [tract](https://github.com/sonos/tract) | ONNX/NNEF inference engine for edge deployment |
| [ort](https://github.com/pykeio/ort) | Rust bindings for ONNX Runtime |
| [tch-rs](https://github.com/LaurentMazare/tch-rs) | Rust bindings for the PyTorch C++ API (libtorch) |
| [dfdx](https://github.com/coreylowman/dfdx) | Compile-time typed deep learning framework |
| [luminal](https://github.com/jafioti/luminal) | Graph-based ML framework with Metal support |

## Building from Source

```bash
git clone https://github.com/jafreck/ironmill.git
cd ironmill
cargo build --workspace
cargo test --workspace
```

Requires Rust 1.85+ (edition 2024).

## Documentation

- [C API](docs/C_API.md): building, linking, and calling from C/Swift/C++
- [ANE Op Support Matrix](docs/design/ane-op-support-matrix.md): verified ANE ops with eval tests
- [ANE Inference](docs/design/ane-inference.md): inference pipeline architecture
- [ANE Constraints](docs/design/ane-constraints.md): hardware limits and diagnostics
- [TurboQuant](docs/design/turboquant.md): INT8 KV cache compression design
- [Compact Cache](docs/design/compact-cache.md): cache memory optimization

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or <http://www.apache.org/licenses/LICENSE-2.0>).
