# Benchmark Results

**Date**: 2026-03-29
**Hardware**: M2 Max Apple Silicon Mac (arm64) 64GB
**Models**: MobileNetV2 (14M ONNX), SqueezeNet 1.1 (4.7M ONNX)
**Tool**: ironmill v0.1.0 (Rust, release build)

## Methodology

Four benchmark suites measure different aspects of ironmill's optimization pipeline:

1. **Compile-time performance** (Criterion) — measures how fast ironmill transforms model graphs in-memory. Each benchmark loads an ONNX model, converts to MIL IR, and runs the optimization pipeline. Timings are median of 100 samples after warmup.

2. **Model size** — compares `.mlpackage` output size across optimization configurations against the original ONNX file size.

3. **Numerical quality** — compares weight tensor values between baseline (unoptimized) and optimized models. FP16 values are converted back to FP32; INT8 values are dequantized via stored scale/zero-point; palettized values are reconstructed from LUT + indices. All comparisons are element-wise against the original FP32 weights.

4. **Inference latency** — end-to-end inference on Apple Silicon using the Rust `ironmill-bench` harness. Models are compiled to `.mlmodelc` via `xcrun coremlcompiler` and loaded with `MLModel`. Latencies are measured over 200 iterations after 20 warmup runs, across 3 independent runs. Significance tested with Welch's t-test against the FP32 baseline (`*** p<0.001, ** p<0.01, * p<0.05`).

---

## Inference Latency

End-to-end inference on Apple Silicon (arm64), measured with the Rust
`ironmill-bench` harness. Each configuration compiles the ONNX model with
ironmill's optimization pipeline, runs `coremlcompiler compile`, then loads the
`.mlmodelc` via CoreML's `MLModel` API and times 200 predictions after 20
warmup iterations across 3 runs.

### MobileNetV2 (14M ONNX, 155 nodes)

| Optimization | CPU | GPU (Metal) | ANE | ANE speedup | sig |
|---|---|---|---|---|---|
| FP32 baseline | 5.3ms | 1.5ms | 5.3ms | 1.0× | |
| Default (fusion) | 4.3ms | 1.5ms | 4.4ms | 1.2× | *** |
| **FP16** | **2.6ms** | **1.2ms** | **0.5ms** | **11.1×** | *** |
| INT8 | 4.4ms | 1.6ms | 4.3ms | 1.2× | *** |
| Palettize 4-bit | 4.4ms | 1.5ms | 4.4ms | 1.2× | *** |
| **PolarQuant 4-bit** | 4.5ms | 1.8ms | 4.9ms | 1.1× | *** |

### SqueezeNet 1.1 (4.7M ONNX, 66 nodes)

| Optimization | CPU | GPU (Metal) | ANE | ANE speedup | sig |
|---|---|---|---|---|---|
| FP32 baseline | 1.9ms | 1.5ms | 1.9ms | 1.0× | |
| Default (fusion) | 1.9ms | 1.5ms | 1.9ms | 1.0× | *** |
| **FP16** | 2.1ms | 1.3ms | **0.3ms** | **7.5×** | *** |
| INT8 | 1.9ms | 1.6ms | 1.9ms | 1.0× | * |
| Palettize 4-bit | 1.9ms | 1.5ms | 2.0ms | 0.9× | *** |
| **PolarQuant 4-bit** | 2.2ms | 1.7ms | 1.9ms | 1.0× | *** |

ANE speedup is measured as CPU FP32 baseline median ÷ ANE median for each
optimization. All values are median latency across 3 runs × 200 iterations.
Significance is Welch's t-test against the baseline (`*** p<0.001`).

### Key Findings

- **FP16 + ANE is the fastest configuration** — 7–11× faster than unoptimized
  CPU inference. The Neural Engine's FP16 data path is specifically optimized
  for half-precision arithmetic, which is why ironmill's FP16 quantization pass
  unlocks performance that other optimizations cannot.
- **PolarQuant 4-bit compiles and runs correctly on all backends** (CPU, GPU,
  ANE) via CoreML. On these CNN models, PolarQuant only quantizes layers with
  inner dimension ≥ 64 (the FC/classifier layers), so the latency impact is
  minimal. PolarQuant's primary value is on transformer architectures where
  most weight tensors qualify (see Qwen3 results below).
- **INT8 and palettization reduce model size but don't improve inference** on
  any backend — their value is in deployment size and load time, not throughput.
- **GPU (Metal) provides consistent ~1.5ms latency** across all optimizations,
  making it the most predictable backend for latency-sensitive applications.

---

## PolarQuant on Transformer Models

PolarQuant targets transformer architectures where weight tensors have large
inner dimensions (hidden_dim ≥ 64). On these models, the majority of weight
tensors are eligible for quantization.

### Qwen3-0.6B (decoder-only LLM, 1824 MIL ops after conversion)

| Metric | Value |
|---|---|
| Weight tensors quantized | 256 / 312 (82%) |
| Target bit-width | 4-bit |
| Rotation fusion ops added | Deduplicated (shared R_inv by seed+dims) |
| Pipeline time | ~28s |

PolarQuant quantizes 82% of Qwen3's weight tensors at 4-bit precision.
End-to-end inference benchmarks are not yet available because CoreML
compilation of transformer-specific ops (`rotary_embedding`,
`group_query_attention`) requires further serialization work.

### Whisper-tiny encoder (encoder-only audio, 562 MIL ops)

| Metric | Value |
|---|---|
| Weight tensors quantized | 29 |
| Target bit-width | 4-bit |

### Supported bit-widths

| Bit-width | CoreML LUT entries | Status |
|---|---|---|
| 4-bit | 16 | ✅ Supported |
| 2-bit | 4 | ✅ Supported |
| 3-bit | 8 | ❌ Rejected at pipeline level (CoreML requires LUT sizes in {2, 4, 16, 64, 256}) |

3-bit PolarQuant is mathematically functional but cannot be deployed through
CoreML's `constexpr_lut_to_dense` due to the LUT size restriction. A future
workaround could pad the 8-entry LUT to 16 entries (4-bit indices) at the
cost of wasted storage.

---

## Model Size Reduction

Output `.mlpackage` size compared to original ONNX file.

### MNIST (original: 28K)

| Configuration | Output Size | Reduction |
|---|---|---|
| No optimization (`--no-fusion`) | 32K | -4% |
| Default (always-on) | 32K | -4% |
| + FP16 | 20K | 41% |
| + INT8 | 16K | 59% |
| + Palettize 4-bit | 12K | 72% |
| + Palettize 6-bit | 16K | 62% |
| + FP16 + Palettize 4-bit | 12K | 73% |
| + PolarQuant 4-bit | 32K | -4% |

### SqueezeNet 1.1 (original: 4.7M)

| Configuration | Output Size | Reduction |
|---|---|---|
| No optimization (`--no-fusion`) | 4.7M | 0% |
| Default (always-on) | 4.7M | 0% |
| + FP16 | 2.4M | 49% |
| + INT8 | 1.2M | 73% |
| + Palettize 4-bit | 636K | 87% |
| + Palettize 6-bit | 948K | 80% |
| + FP16 + Palettize 4-bit | 632K | 87% |
| + PolarQuant 4-bit | 4.7M | 0% |

PolarQuant is a no-op on these CNN models (conv kernel inner dims are 1–3,
below the ≥ 64 threshold). The `+ PolarQuant 4-bit` rows confirm no size
change. PolarQuant's size benefits apply to transformer models where
linear weight tensors dominate the model size.

---

## Compile-Time Performance

### Per-Pass Execution Time (SqueezeNet 1.1, 119 ops)

Each pass is benchmarked independently on a fresh copy of the SqueezeNet IR.

| Pass | Median Time | Ops After |
|------|-------------|-----------|
| Dead code elimination | 404 µs | 8 |
| Constant folding | 276 µs | 119 |
| Conv-ReLU fusion | 159 µs | 93 |
| Layout optimization | 147 µs | 130 |
| Identity elimination | 122 µs | 119 |
| Conv-BN weight fold | 124 µs | 119 |
| Conv-BN fusion | 123 µs | 119 |
| Attention fusion | 122 µs | 119 |
| Op substitution | 119 µs | 119 |

Dead code elimination is the most impactful pass, removing 93% of ops (119 → 8). Conv-ReLU fusion eliminates 22% of ops by merging activation functions into convolutions.

### End-to-End Pipeline Throughput

Time to convert ONNX → optimized MIL IR with the full pass pipeline.

| Pipeline Configuration | MNIST | SqueezeNet 1.1 |
|---|---|---|
| Default (always-on passes) | 17 µs | 454 µs |
| Default + INT8 | 18 µs | 465 µs |
| Default + Palettize 4-bit | 17 µs | 467 µs |
| Default + FP16 | 21 µs | 705 µs |

FP16 quantization adds ~55% overhead due to float-to-half conversion of all weight tensors. INT8 and palettization add minimal overhead (<3%).

---

## Numerical Quality

Compares reconstructed weight values against original FP32 weights to quantify precision loss from each optimization.

### SqueezeNet 1.1

| Optimization | Max Error | MSE | Cosine Sim | SNR | Verdict |
|---|---|---|---|---|---|
| Default (always-on) | 0.0 | 0.0 | 1.000000 | ∞ | Lossless |
| FP16 | 0.000061 | 6.3e-11 | 1.000000 | 73.6 dB | Excellent |
| INT8 | 0.000788 | 2.1e-7 | 0.999929 | 38.5 dB | Good |
| Palettize 6-bit | 0.005889 | 2.4e-6 | 0.999175 | 27.8 dB | Acceptable |
| Palettize 4-bit | 0.098209 | 1.5e-5 | 0.994844 | 19.9 dB | Verify task accuracy |

### MNIST

| Optimization | Max Error | MSE | Cosine Sim | SNR | Verdict |
|---|---|---|---|---|---|
| Default (always-on) | 0.0 | 0.0 | 1.000000 | ∞ | Lossless |
| FP16 | 0.000410 | 1.5e-9 | 1.000000 | 73.5 dB | Excellent |
| INT8 | 0.003896 | 3.0e-6 | 0.999954 | 40.3 dB | Good |
| Palettize 6-bit | 0.014742 | 2.3e-5 | 0.999648 | 31.5 dB | Acceptable |
| Palettize 4-bit | 0.086620 | 3.7e-4 | 0.994240 | 19.4 dB | Verify task accuracy |

### Interpreting Quality Metrics

- **Cosine similarity ≥ 0.999**: Model behavior is effectively unchanged. Safe for production.
- **SNR ≥ 30 dB**: Good quality. Minor rounding artifacts, unlikely to affect task accuracy.
- **SNR 20–30 dB**: Acceptable for most tasks but should be validated on a representative dataset.
- **SNR < 20 dB**: Significant precision loss. Requires task-specific accuracy testing before deployment.

FP16 is nearly lossless (73+ dB SNR). INT8 provides strong compression with good fidelity (~40 dB). 4-bit palettization shows meaningful error that warrants task-level validation — recommended for size-constrained deployments where slight accuracy loss is acceptable.

---

## Running the Benchmarks

```bash
# Inference latency (cross-tabulated: optimizations × cpu/gpu/ane)
cargo run --release -p ironmill-bench

# With significance tests against baseline
cargo run --release -p ironmill-bench -- --baseline baseline

# With PolarQuant (custom config)
cargo run --release -p ironmill-bench -- --config bench.toml

# Single backend only (flat table format)
cargo run --release -p ironmill-bench -- -b all

# Custom model
cargo run --release -p ironmill-bench -- -m path/to/model.onnx

# JSON/CSV/Markdown output
cargo run --release -p ironmill-bench -- --output json
cargo run --release -p ironmill-bench -- --output csv
cargo run --release -p ironmill-bench -- --output markdown

# ANE direct runtime (experimental, bypass CoreML)
cargo run --release -p ironmill-bench --features ane-direct -- --ane-direct

# Compile-time performance (Criterion)
cargo bench

# Model size comparison
./scripts/bench-size.sh

# Numerical quality report
./scripts/bench-quality.sh
```
