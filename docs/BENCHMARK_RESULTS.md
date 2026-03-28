# Benchmark Results

**Date**: 2026-03-28
**Hardware**: Apple Silicon Mac (arm64)
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

End-to-end inference on Apple Silicon (arm64), measured with the Swift
`InferenceBench` harness.  Each configuration compiles the ONNX model with
`ironmill`, runs `coremlcompiler compile`, then loads the `.mlmodelc` via
`MLModel` and times 200 predictions after 20 warmup iterations.

### MobileNetV2 (14M ONNX, 155 nodes)

| Configuration | p50 | p95 | p99 | Load |
|---|---|---|---|---|
| No optimization (`--no-fusion`) | 1.9ms | 2.3ms | 2.4ms | 6.93s |
| Default (always-on) | 1.9ms | 2.4ms | 2.7ms | 7.99s |
| **+ FP16** | **515µs** | **555µs** | **595µs** | 9.95s |
| + INT8 | 1.8ms | 2.4ms | 2.6ms | 4.52s |
| + Palettize 4-bit | 1.9ms | 2.5ms | 2.9ms | 866ms |

FP16 delivers a **3.7× speedup** (1.9ms → 515µs) by enabling native half-precision
compute on the Neural Engine.  INT8 shows a modest improvement.  Palettize
reduces model load time significantly (866ms vs 7s) but doesn't improve
inference speed since weights are decompressed at load time.

### SqueezeNet 1.1 (4.7M ONNX, 66 nodes)

| Configuration | p50 | p95 | p99 | Load |
|---|---|---|---|---|
| No optimization (`--no-fusion`) | 1.2ms | 1.7ms | 1.9ms | 2.48s |
| Default (always-on) | 1.2ms | 1.6ms | 1.8ms | 682ms |
| **+ FP16** | **281µs** | **337µs** | **382µs** | 4.02s |
| + INT8 | 2.4ms | 7.1ms | 10.4ms | 1.22s |
| + Palettize 4-bit | 1.8ms | 2.5ms | 3.9ms | 563ms |

FP16 delivers a **4.3× speedup** (1.2ms → 281µs).  The always-on fusion
passes (conv-relu) cut model load time from 2.5s to 682ms by reducing the
number of ops the CoreML compiler must process.

### Backend Comparison: CPU vs GPU vs ANE

All results above use `.all` compute units (CoreML picks the fastest backend).
The table below isolates each backend on MobileNetV2 to show where the
speedups actually come from.

| Quantization | CPU | GPU (Metal) | ANE |
|---|---|---|---|
| FP32 | 5.7ms | 2.8ms | 2.4ms |
| **FP16** | 3.2ms | 2.3ms | **861µs** |
| INT8 | 5.6ms | 4.6ms | 2.1ms |
| Palettize 4-bit | 7.3ms | 2.5ms | 2.0ms |

- **ANE is fastest across all quantizations**, but FP16 unlocks its full
  potential — 2.8× faster than FP32 on the same ANE hardware (861µs vs 2.4ms).
- **GPU (Metal)** is ~2× faster than CPU at FP32, but FP16 barely helps it
  (2.3ms vs 2.8ms).
- **INT8 hurts GPU** (4.6ms vs 2.8ms FP32) — the dequantization overhead
  exceeds any compute savings.
- **Palettize** doesn't speed up inference on any backend — it primarily
  reduces model size and load time.

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
