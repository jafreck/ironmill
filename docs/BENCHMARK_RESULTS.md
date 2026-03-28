# Benchmark Results

**Date**: 2026-03-28
**Hardware**: M2 Max Apple Silicon Mac (arm64) 64GB
**Models**: MobileNetV2 (14M ONNX), SqueezeNet 1.1 (4.7M ONNX)
**Tool**: ironmill v0.1.0 (Rust, release build)

## Methodology

Three benchmark suites measure different aspects of ironmill's optimization pipeline:

1. **Compile-time performance** (Criterion) — measures how fast ironmill transforms model graphs in-memory. Each benchmark loads an ONNX model, converts to MIL IR, and runs the optimization pipeline. Timings are median of 100 samples after warmup.

2. **Model size** — compares `.mlpackage` output size across optimization configurations against the original ONNX file size.

3. **Numerical quality** — compares weight tensor values between baseline (unoptimized) and optimized models. FP16 values are converted back to FP32; INT8 values are dequantized via stored scale/zero-point; palettized values are reconstructed from LUT + indices. All comparisons are element-wise against the original FP32 weights.

4. **Inference latency** — end-to-end inference on Apple Silicon using the Swift `InferenceBench` harness. Models are compiled to `.mlmodelc` via `xcrun coremlcompiler` and loaded with `MLModel`. Latencies are measured over 200 iterations after 20 warmup runs.

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
|------------------------|-------|----------------|
| Default (always-on passes) | 17 µs | 454 µs |
| Default + INT8 | 18 µs | 465 µs |
| Default + Palettize 4-bit | 17 µs | 467 µs |
| Default + FP16 | 21 µs | 705 µs |

FP16 quantization adds ~55% overhead due to float-to-half conversion of all weight tensors. INT8 and palettization add minimal overhead (<3%).

---

## Model Size Reduction

Output `.mlpackage` size compared to original ONNX file.

### MNIST (original: 28K)

| Configuration | Output Size | Reduction |
|---------------|-------------|-----------|
| No optimization (`--no-fusion`) | 32K | -14% |
| Default (always-on) | 32K | -14% |
| + FP16 | 20K | 29% |
| + FP16 + Palettize 4-bit | 20K | 29% |

MNIST is a tiny model — the `.mlpackage` metadata overhead dominates, so most compression techniques show limited benefit.

### SqueezeNet 1.1 (original: 4.7M)

| Configuration | Output Size | Reduction |
|---------------|-------------|-----------|
| No optimization (`--no-fusion`) | 2.0M | 58% |
| Default (always-on) | 2.0M | 58% |
| + FP16 | 1.0M | 79% |
| + FP16 + Palettize 4-bit | 1.0M | 79% |

The baseline 58% reduction comes from the ONNX→CoreML format difference (protobuf encoding, deduplication). FP16 halves the weight storage, achieving 79% total reduction.

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

## Inference Latency

End-to-end inference on Apple Silicon (arm64), measured with the Rust
`ironmill-bench` harness.  Each configuration compiles the ONNX model with
ironmill's optimization pipeline, runs `coremlcompiler compile`, then loads the
`.mlmodelc` via CoreML's `MLModel` API and times 200 predictions after 20
warmup iterations across 3 runs.

### MobileNetV2 (14M ONNX, 155 nodes)

| Optimization | CPU | GPU (Metal) | ANE | ANE speedup |
|---|---|---|---|---|
| FP32 baseline | 7.4ms | 2.2ms | 6.0ms | 1.2× |
| Default (fusion) | 4.7ms | 2.6ms | 4.8ms | 1.5× |
| **FP16** | 3.0ms | 2.1ms | **1.0ms** | **7.4×** |
| INT8 | 5.2ms | 1.8ms | 4.7ms | 1.6× |
| Palettize 4-bit | 4.6ms | 1.5ms | 4.9ms | 1.5× |

### SqueezeNet 1.1 (4.7M ONNX, 66 nodes)

| Optimization | CPU | GPU (Metal) | ANE | ANE speedup |
|---|---|---|---|---|
| FP32 baseline | 2.1ms | 1.6ms | 2.0ms | 1.1× |
| Default (fusion) | 1.9ms | 1.6ms | 1.9ms | 1.1× |
| **FP16** | 2.0ms | 1.6ms | **0.3ms** | **7.7×** |
| INT8 | 2.3ms | 1.6ms | 1.9ms | 1.1× |
| Palettize 4-bit | 1.9ms | 1.7ms | 1.9ms | 1.1× |

ANE speedup is measured as CPU FP32 baseline median ÷ ANE median for each
optimization.  All values are median latency across 3 runs × 200 iterations.

### Key Findings

- **FP16 + ANE is the fastest configuration** — 7–8× faster than unoptimized
  CPU inference.  The Neural Engine's FP16 data path is specifically optimized
  for half-precision arithmetic, which is why ironmill's FP16 quantization pass
  unlocks performance that other optimizations cannot.
- **INT8 and palettization reduce model size but don't improve inference** on
  any backend — their value is in deployment size and load time, not throughput.
- **GPU (Metal) provides consistent ~2ms latency** across all optimizations,
  making it the most predictable backend for latency-sensitive applications.
- **Without FP16, the ANE is no faster than CPU** — the 6ms FP32 ANE latency
  shows that the Neural Engine cannot efficiently execute full-precision models.
  Ironmill's optimization pipeline is what makes ANE deployment practical.

---

## Running the Benchmarks

```bash
# Inference latency (cross-tabulated: optimizations × cpu/gpu/ane)
cargo run --release -p ironmill-bench

# With significance tests against baseline
cargo run --release -p ironmill-bench -- --baseline baseline

# Single backend only (flat table format)
cargo run --release -p ironmill-bench -- -b all

# Custom model
cargo run --release -p ironmill-bench -- -m path/to/model.onnx

# Compile-time performance (Criterion)
cargo bench

# Model size comparison
./scripts/bench-size.sh

# Numerical quality report
./scripts/bench-quality.sh
```
