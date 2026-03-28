# Benchmark Results

**Date**: 2026-03-28
**Hardware**: Apple Silicon Mac (arm64)
**Models**: MNIST (28K ONNX), SqueezeNet 1.1 (4.7M ONNX)
**Tool**: ironmill v0.1.0 (Rust, release build)

## Methodology

Three benchmark suites measure different aspects of ironmill's optimization pipeline:

1. **Compile-time performance** (Criterion) — measures how fast ironmill transforms model graphs in-memory. Each benchmark loads an ONNX model, converts to MIL IR, and runs the optimization pipeline. Timings are median of 100 samples after warmup.

2. **Model size** — compares `.mlpackage` output size across optimization configurations against the original ONNX file size.

3. **Numerical quality** — compares weight tensor values between baseline (unoptimized) and optimized models. FP16 values are converted back to FP32; INT8 values are dequantized via stored scale/zero-point; palettized values are reconstructed from LUT + indices. All comparisons are element-wise against the original FP32 weights.

Inference latency benchmarks require full Xcode installation and are not yet included.

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

## Running the Benchmarks

```bash
# Compile-time performance (Criterion)
cargo bench

# Model size comparison
./scripts/bench-size.sh

# Numerical quality report
./scripts/bench-quality.sh

# Inference latency (requires Xcode)
./scripts/bench-inference.sh
```
