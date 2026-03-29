# Benchmark Improvements Plan — Inference Performance & Per-Watt

## Goal

Extend ironmill's benchmark suite to measure the metrics that actually matter
for Apple Silicon ML deployment: throughput, energy efficiency, ANE utilization,
and cross-backend comparison. The current benchmarks are useful for compiler
development but don't capture the value of running optimized models on ANE
hardware.

This plan expands on item #4 ("Inference Benchmark Overhaul") from
`docs/INFERENCE_IMPROVEMENTS_PLAN.md` into a concrete implementation plan.

## Current State

Ironmill has four benchmark suites:

1. **Criterion microbenchmarks** (`crates/mil-rs/benches/`) — per-pass and
   full-pipeline timing on SqueezeNet and MNIST. Measures compiler throughput.
2. **Inference latency harness** (`crates/ironmill-bench/`) — end-to-end
   CoreML prediction timing with statistical analysis (mean, p95, p99,
   Welch's t-test). Reports latency in milliseconds.
3. **Model size** (`scripts/bench-size.sh`) — `.mlpackage` size across
   optimization configs.
4. **Numerical quality** (`scripts/bench-quality.sh`, `examples/quality_report.rs`)
   — MSE, cosine similarity, SNR for weight fidelity.

### What's Missing

**Performance gaps**: The inference harness measures **call latency** — how
long a single `model.predict()` takes. This is necessary but insufficient. It
doesn't answer:

- How many tokens per second for autoregressive models? (tok/s)
- What fraction of ANE peak TFLOPS am I achieving? (utilization)
- How much power does each backend consume? (per-watt efficiency)
- Where is the overhead — in dispatch, I/O, or compute? (profiling)
- Is this faster or slower than last week? (regression tracking)

**Quality gaps**: The quality benchmarks measure weight-level fidelity (MSE,
cosine similarity, SNR) — they verify that quantized weights are close to the
originals. But they don't answer:

- Does the model still produce correct outputs after optimization? (task accuracy)
- What is the perplexity impact of FP16 / INT8 / palettization? (LLM quality)
- Does top-k accuracy degrade on a reference dataset? (classification quality)
- What is the word error rate after quantization? (ASR quality)

**Model gaps**: The inference harness config lists Whisper-tiny-encoder,
DistilBERT, and Qwen3-0.6B, but current published results focus on CNN models
(MobileNetV2, SqueezeNet). Transformer models are benchmarked only for
compilation pipeline throughput (`weight_formats.rs`), not for inference
performance. The benchmarks don't demonstrate ironmill's value for the
workloads that matter most — LLM and transformer deployment.

## Model Matrix

Benchmarks are split into two categories: **compilation** (how fast ironmill
processes models) and **inference** (how fast the resulting models run on
hardware). Each category has different model requirements.

### Compilation Benchmarks

Compilation benchmarks measure ironmill's own performance — ONNX parsing, MIL
IR optimization, weight conversion, and CoreML packaging. These should cover a
range of model sizes to test scaling behavior.

| Model | Format | Size | Purpose | Status |
|-------|--------|------|---------|--------|
| MNIST | ONNX | 28K | Tiny baseline, pass overhead | ✅ Active |
| SqueezeNet 1.1 | ONNX | 4.7M | Small CNN, pass effectiveness | ✅ Active |
| Whisper-medium-encoder | ONNX | ~1.5GB | Large encoder, scaling test | ✅ Active |
| Qwen3-0.6B | SafeTensors | ~1.4GB | LLM weight loading + pipeline | ✅ Active |
| Qwen3-0.6B | GGUF Q8_0 | ~639MB | Quantized LLM weight loading | ✅ Active |

These are adequate. No changes needed.

### Inference Benchmarks

Inference benchmarks measure end-to-end model execution on Apple Silicon.
These should be **transformer-focused** — transformers are the primary
deployment target and the workload where ANE optimization matters most.

| Model | Arch | Params | Task | Purpose | Status |
|-------|------|--------|------|---------|--------|
| SqueezeNet 1.1 | CNN | 1.2M | Classification | Baseline, fast smoke test | ✅ Active |
| MobileNetV2 | CNN | 3.4M | Classification | CNN reference point | ✅ Active |
| DistilBERT | Encoder | 66M | Encoding | Encoder transformer | ⬜ In config, not published |
| Whisper-tiny-encoder | Encoder | 39M | ASR encoding | Audio transformer | ⬜ In config, not published |
| Qwen3-0.6B | Decoder | 596M | Text generation | LLM decode + prefill | ⬜ In config, blocked |

**Priority additions** (new fixtures needed):

| Model | Arch | Params | Task | Why |
|-------|------|--------|------|-----|
| Llama-3.2-1B | Decoder | 1.2B | Text generation | GQA, RoPE, mainstream LLM |
| Phi-3-mini | Decoder | 3.8B | Text generation | Tests SRAM pressure, larger LLM |
| Whisper-small | Encoder-decoder | 244M | ASR | Full encoder-decoder pipeline |

The CNN models (SqueezeNet, MobileNetV2) remain as fast smoke tests and
regression anchors. The transformer models are the primary benchmark targets.

### Inference Quality Benchmarks

Inference quality benchmarks verify that optimized models still produce correct
outputs. These complement weight-level fidelity metrics with **task-level
accuracy** measured on reference datasets.

| Model | Metric | Dataset | What it validates |
|-------|--------|---------|-------------------|
| Qwen3-0.6B | Perplexity | WikiText-2 (subset) | LLM generation quality after quantization |
| Llama-3.2-1B | Perplexity | WikiText-2 (subset) | LLM quality at scale |
| DistilBERT | Top-1 accuracy | SST-2 (subset) | Encoder classification after optimization |
| Whisper-tiny | WER | LibriSpeech test-clean (subset) | ASR accuracy after quantization |
| MobileNetV2 | Top-1 / Top-5 | ImageNet val (subset) | CNN classification baseline |

Subsets are used (e.g., 500 samples) to keep benchmark runtime practical while
still detecting quality regressions.

Quality benchmarks should produce a cross-tabulated report:

```
Qwen3-0.6B — Perplexity on WikiText-2 (500 samples)
──────────────────────────────────────────────────────
Optimization         Perplexity   Δ vs FP32   Weight SNR
───────────────────  ──────────   ─────────   ──────────
FP32 baseline        12.4         —           ∞
FP16                 12.4         +0.0%       73.6 dB
INT8                 12.8         +3.2%       38.5 dB
Palettize 4-bit      14.1         +13.7%      19.9 dB
PolarQuant 4-bit     12.9         +4.0%       —
```

This ties weight-level fidelity (SNR) to task-level impact (perplexity),
making it clear which optimizations are safe for deployment.

## Reference Projects

Two projects in the direct-ANE space have benchmark approaches worth learning
from.

### maderix/ANE

Hardware microbenchmarks focused on characterizing ANE silicon:

- **Peak TFLOPS** — measured via 128× conv 512ch matmuls, reported per
  precision (FP16 vs INT8 W8A8)
- **Sustained throughput** — TFLOPS over 5-second windows with utilization %
- **SRAM bandwidth** — probes the 32MB L2 SRAM cliff
- **Dispatch latency** — XPC + IOKit overhead per eval (~0.095ms on M4)
- **Cross-generation comparison** — M1 through M5 results in structured JSON
  with community submissions
- **Community results** — `benchmarks/community_results.json` with chip model,
  cores, RAM, macOS version, per-chip TFLOPS and training times

Key takeaway: structured machine-readable output enables community hardware
comparison. The cross-generation charts show performance trends that no single
benchmark run can.

### mechramc/Orion

Full LLM runtime with a 4-mode benchmark CLI:

- **`bench kernels`** — per-kernel ANE latency across 5 kernel types, with
  baseline save and >15% regression warnings
- **`bench inference`** — end-to-end tok/s, TFLOPS, and ANE utilization
- **`bench training`** — step breakdown (forward/backward/dW/Adam)
- **Regression tracking** — `--save-baseline` persists results; subsequent
  runs auto-compare and flag regressions

Metrics measured:
- TFLOPS (actual compute throughput)
- Tokens/sec (end-to-end inference speed)
- ANE utilization (time in `orion_eval` vs total wall time)
- Dispatch overhead (compile + eval scheduling cost)
- Memory RSS growth across compiles and swaps

Key takeaway: measuring utilization and dispatch overhead explains the gap
between peak ANE TFLOPS (~19) and observed throughput (~1.4). Without these
metrics, users assume ANE is slow when the bottleneck is actually dispatch.

### What Neither Project Measures

Neither maderix/ANE nor Orion measures **power consumption** or **per-watt
efficiency**. This is a differentiation opportunity for ironmill — macOS
`powermetrics` can sample CPU, GPU, and ANE power draw, enabling metrics like
inferences-per-watt and joules-per-inference that are uniquely relevant for
edge deployment.

## Improvements

### 1. FLOPs Counter for MIL IR

To report TFLOPS, we need to know how many floating-point operations a model
performs per forward pass.

Add a `flops` analysis module to `mil-rs` that walks the MIL IR and counts
multiply-accumulate operations per op type:

- `conv` — `2 × K_h × K_w × C_in × C_out × H_out × W_out`
- `matmul` / `linear` — `2 × M × N × K`
- `batch_matmul` — `2 × B × M × N × K`
- element-wise ops — count by output size
- attention composite — decompose into constituent matmuls

Expose as `program.total_flops() -> u64` in the public API.

Implementation location: `crates/mil-rs/src/analysis/flops.rs` (new module).

This is foundational — throughput metrics (#2) and backend comparison (#5)
depend on it.

### 2. Throughput Metrics

Extend `BenchResult` in `crates/ironmill-bench/src/stats.rs` with computed
throughput fields:

```rust
pub struct BenchResult {
    // Existing latency fields...
    pub mean: f64,
    pub stddev: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
    pub cv: f64,

    // New throughput fields
    pub inferences_per_sec: f64,      // 1000.0 / median_ms
    pub tflops: Option<f64>,          // model_flops / median_sec / 1e12
    pub tokens_per_sec: Option<f64>,  // for autoregressive models
    pub ttft_ms: Option<f64>,         // time to first token (prefill)
    pub decode_tok_per_sec: Option<f64>, // per-token decode throughput
}
```

For CNN models, `inferences_per_sec` and `tflops` are the primary metrics.

For transformer models, the inference loop must distinguish prefill from
decode. This is the most important throughput measurement:

- **TTFT** (time to first token) — latency of the prefill pass, measured
  after warmup. For encoder models (DistilBERT, Whisper encoder), this is
  the only relevant metric since there is no autoregressive decode.
- **Decode tok/s** — `1000 / decode_median_ms` for single-token decode
  steps in autoregressive models (Qwen3, Llama). This is the headline
  number for LLM inference performance.
- **Prompt processing tok/s** — `prompt_length / ttft_sec` for the prefill
  phase. Relevant for long-context workloads.

Detection: models with `past_key_values`-style inputs in the CoreML input
description are autoregressive. The benchmark loop should automatically
switch to prefill+decode measurement for these models.

### 3. Inference Quality Measurement

Add task-level accuracy benchmarks alongside the existing weight-level
fidelity metrics. This is critical for demonstrating that ironmill's
optimizations are deployment-safe on transformer models.

New module: `src/quality.rs` in `ironmill-bench` (extend the existing stub).

#### Perplexity (LLM models)

For decoder models (Qwen3-0.6B, Llama), compute perplexity on a reference
dataset:

- Load a small fixed dataset (e.g., 500 WikiText-2 sequences, bundled or
  downloaded by `scripts/download-fixtures.sh`)
- Run the optimized model through CoreML on each sequence
- Compute cross-entropy loss → perplexity
- Compare against the FP32 baseline perplexity

Report perplexity delta per optimization config. Flag when perplexity
degrades more than a configurable threshold (default: >5% relative increase).

#### Top-k Accuracy (Encoder / Classification models)

For encoder models (DistilBERT) and CNN models (MobileNetV2):

- Run inference on a small labeled dataset (e.g., 200 SST-2 samples for
  DistilBERT, 500 ImageNet-val samples for MobileNetV2)
- Compare predicted labels against ground truth
- Report top-1 accuracy (and top-5 for classification)
- Compare against FP32 baseline accuracy

#### Word Error Rate (ASR models)

For Whisper models:

- Run inference on a small audio dataset (e.g., 50 LibriSpeech test-clean
  samples)
- Decode output tokens to text
- Compute WER against reference transcriptions
- Compare against FP32 baseline WER

#### Quality Report Integration

Quality benchmarks should be opt-in via `--quality` flag (they require
datasets and take longer to run). Output format matches the throughput
reports:

```
ironmill-bench --quality --model qwen3-0.6b.onnx
```

Results should cross-reference weight fidelity and task accuracy:

```
Qwen3-0.6B — Quality Impact
─────────────────────────────────────────────────────────────
Optimization       Weight SNR   Perplexity   Δ PPL    Status
─────────────────  ──────────   ──────────   ──────   ──────
FP32 baseline      ∞            12.4         —        —
FP16               73.6 dB      12.4         +0.0%    ✓ SAFE
INT8               38.5 dB      12.8         +3.2%    ✓ SAFE
Palettize 4-bit    19.9 dB      14.1         +13.7%   ⚠ WARN
PolarQuant 4-bit   —            12.9         +4.0%    ✓ SAFE
```

### 4. Energy / Per-Watt Measurement

Add a `power.rs` module to `crates/ironmill-bench/` that wraps macOS
`powermetrics`:

```
sudo powermetrics --samplers cpu_power,gpu_power,ane_power \
  -i 100 -n <samples> --format plist
```

Parse plist output to extract per-component power draw in watts:

- CPU package power
- GPU power
- ANE power (when available)
- Combined package power

Derive efficiency metrics:

| Metric | Formula | Unit |
|--------|---------|------|
| Inferences per watt | inferences/sec ÷ package_watts | inf/W |
| Tokens per watt | tok/s ÷ package_watts | tok/W |
| TFLOPS per watt | tflops ÷ package_watts | TFLOPS/W |
| Joules per inference | package_watts × median_sec | J/inf |

The power sampler runs as a background process during the timed inference
loop. Sample at 100ms intervals for sufficient resolution without affecting
benchmark timing.

**Graceful degradation**: `powermetrics` requires root. When unavailable:

- skip energy metrics silently
- print a one-line note: `"Energy metrics unavailable (requires sudo)"`
- all other metrics report normally

**Idle vs loaded delta**: sample idle power for 2 seconds before the benchmark
starts. Report `delta_watts = loaded_watts - idle_watts` alongside absolute
power, so users can see the marginal cost of inference.

### 5. ANE Utilization and Dispatch Overhead

Measure the breakdown of wall time to explain the gap between peak ANE
throughput and observed throughput.

Instrument the inference path in `crates/ironmill-bench/src/inference.rs`:

```
total_time = load_time + warmup_time + measurement_time

measurement_time per iteration:
  = input_marshal_time    (build MLMultiArray / MLFeatureProvider)
  + predict_time          (model.predict() — includes CoreML dispatch)
  + output_read_time      (read results from MLMultiArray)
```

Report:

| Metric | What it measures |
|--------|------------------|
| ANE utilization % | `predict_time / total_iteration_time × 100` |
| Dispatch overhead ms | `total_iteration_time - predict_time` |
| Load time | Time to load `.mlmodelc` into CoreML |
| Compile time | Time for `coremlcompiler compile` (ONNX → mlmodelc) |

This directly addresses the observation from maderix/ANE that dispatch
overhead (~0.095ms XPC + IOKit) can dominate small-model inference, and
from Orion that ANE utilization is typically 4–9% of peak for full training
steps.

### 6. Backend Comparison Matrix

Run the same model across all available compute backends in a single
invocation and produce a side-by-side comparison.

Backends:

| Backend | CoreML `MLComputeUnits` |
|---------|-------------------------|
| CPU | `.cpuOnly` |
| GPU | `.cpuAndGPU` |
| ANE | `.cpuAndNeuralEngine` |
| All | `.all` (CoreML chooses) |

For each backend, collect: latency stats, throughput, energy (if available),
and utilization breakdown. Format as a comparison table:

```
MobileNetV2 + FP16 — Backend Comparison (M4 Max)
─────────────────────────────────────────────────────────────────────
Backend    Median    Inf/sec   TFLOPS   Watts   Inf/Watt   J/Inf
─────────  ──────    ───────   ──────   ─────   ────────   ─────
CPU        2.6ms     385       0.12     8.2W    46.9       0.021
GPU        1.2ms     833       0.26     12.1W   68.9       0.015
ANE        0.5ms     2000      0.62     3.8W    526.3      0.002  ←
All        0.5ms     2000      0.62     3.9W    512.8      0.002

← = best per-watt efficiency
```

This is the single most important output for demonstrating ironmill's value.
The per-watt column shows ANE's efficiency advantage — the story that raw
latency alone doesn't tell.

Implementation: loop over backends in `main.rs`, collect per-backend results,
add comparison formatter to `report.rs`.

### 7. Structured JSON Output

Add `--format json` to `ironmill-bench` CLI. Emit a stable, documented schema
for machine consumption:

```json
{
  "version": "1",
  "timestamp": "2026-03-29T22:00:00Z",
  "hardware": {
    "chip": "Apple M4 Max",
    "cores_cpu": 16,
    "cores_gpu": 40,
    "ne_cores": 16,
    "ram_gb": 64,
    "macos": "15.4"
  },
  "ironmill_version": "0.1.0",
  "results": [
    {
      "model": "MobileNetV2",
      "optimization": "fp16",
      "backend": "ane",
      "iterations": 200,
      "warmup": 20,
      "runs": 3,
      "latency": {
        "mean_ms": 0.52,
        "median_ms": 0.50,
        "p95_ms": 0.61,
        "p99_ms": 0.68,
        "stddev_ms": 0.04,
        "cv": 0.077
      },
      "throughput": {
        "inferences_per_sec": 2000,
        "tflops": 0.62
      },
      "energy": {
        "package_watts": 3.8,
        "ane_watts": 2.1,
        "inferences_per_watt": 526.3,
        "joules_per_inference": 0.002
      },
      "utilization": {
        "predict_pct": 94.2,
        "dispatch_overhead_ms": 0.03
      }
    }
  ]
}
```

Hardware detection: read chip model from `sysctl hw.model` and
`machdep.cpu.brand_string`, RAM from `sysctl hw.memsize`, macOS version from
`sw_vers`.

This enables:
- CI integration (diff JSON across commits)
- community hardware comparison (collect results across machines)
- downstream tooling (dashboards, charts)

### 8. Regression Tracking

Add baseline save/compare to `ironmill-bench`:

```bash
# Save current results as a named baseline
ironmill-bench --save-baseline v0.1.0

# Run again — auto-compares against most recent baseline
ironmill-bench
```

Baseline storage: `~/.ironmill/baselines/<name>.json` using the JSON schema
from #6.

Comparison logic:

| Change | Status | Meaning |
|--------|--------|---------|
| < 15% slower | OK | Within noise |
| 15–25% slower | WARN | Possible regression |
| > 25% slower | FAIL | Likely regression |
| Any faster | IMPROVED | |

Print a delta table on each run when a baseline exists:

```
Regression check vs baseline "v0.1.0"
──────────────────────────────────────────────────────
Model + Opt       Backend  Metric      Base   Now    Δ       Status
────────────────  ───────  ──────────  ─────  ─────  ──────  ──────
MobileNetV2 fp16  ANE      median_ms   0.50   0.48   -4.0%  ✓ OK
MobileNetV2 fp16  ANE      inf/sec     2000   2083   +4.2%  ↑ IMPROVED
SqueezeNet fp16   ANE      median_ms   0.30   0.38   +26.7% ✗ FAIL
```

Implementation: new module `src/baseline.rs` in `ironmill-bench`.

### 9. Memory Profiling

Track memory footprint across the inference lifecycle:

| Metric | When measured | How |
|--------|---------------|-----|
| RSS before load | Before `Model::load()` | `mach_task_basic_info` |
| RSS after load | After `Model::load()` | `mach_task_basic_info` |
| Peak RSS | During inference loop | Sample every 100 iterations |
| RSS growth | After N iterations | Final RSS - post-load RSS |
| Model file size | Before load | `std::fs::metadata` |

Report memory efficiency ratio: `model_file_size / runtime_rss_delta`.
Flag potential leaks when RSS growth > 10% over the measurement window.

macOS implementation: call `mach_task_basic_info` via the `mach2` crate
(already available on macOS) to read `resident_size`.

### 10. CLI Improvements

Extend `ironmill-bench` CLI to expose all new capabilities:

```
ironmill-bench [OPTIONS]

Options (existing):
  -c, --config <PATH>       TOML config file
  -m, --model <PATH>        Model path (repeatable)
  -i, --iterations <N>      Iterations [default: 200]
  -w, --warmup <N>          Warmup iterations [default: 20]
  -r, --runs <N>            Full runs [default: 3]
  -b, --backend <BACKEND>   Backend: cpu, gpu, ane, all [default: all]
  -o, --output <FORMAT>     Format: table, json, csv, markdown [default: table]
      --baseline <NAME>     Mark baseline for significance tests
      --alpha <FLOAT>       Significance level [default: 0.05]

Options (new):
      --backends <LIST>     Run across multiple backends: cpu,gpu,ane,all
      --power               Enable energy sampling (requires sudo)
      --quality             Run task-level accuracy benchmarks (requires datasets)
      --save-baseline <N>   Save results as named baseline
      --format <FMT>        Output format: table, json, markdown [default: table]

Subcommands (new):
  compare <A> <B>           Compare two saved baselines
  report                    Full report across all models in a directory
```

### 11. Results Documentation

Produce a `docs/INFERENCE_PERFORMANCE.md` with benchmark results that
demonstrate ironmill's value proposition for transformer deployment.
Structure inspired by Orion's `RESULTS.md`:

1. **Methodology** — iterations, warmup, runs, hardware, how power is sampled
2. **Transformer inference tables** — tok/s, TTFT, decode throughput, TFLOPS,
   energy for each model × optimization × backend
3. **Quality impact tables** — perplexity / accuracy / WER per optimization,
   cross-referenced with weight SNR
4. **Per-watt efficiency** — highlight ANE's advantage in tokens/watt for LLMs
5. **Backend comparison** — side-by-side CPU vs GPU vs ANE for each transformer
6. **Utilization analysis** — show where wall time goes (compute vs dispatch)
7. **CNN baselines** — MobileNetV2 and SqueezeNet as reference points
8. **Hardware specifications** — chip, cores, RAM, macOS version
9. **Reproducibility** — exact commands to regenerate all numbers

## Implementation Order

### Phase 1 — Foundations

1. **FLOPs counter** (`crates/mil-rs/src/analysis/flops.rs`)
   - Walk MIL IR, count MACs per op type
   - Expose `program.total_flops()`
   - Unit tests against known model FLOPs (SqueezeNet, MobileNetV2, Qwen3)

2. **Throughput metrics** (`crates/ironmill-bench/src/stats.rs`)
   - Add `tokens_per_sec`, `tflops`, `ttft_ms` to `BenchResult`
   - Prefill vs decode loop for autoregressive models
   - Update `report.rs` to display new fields

### Phase 2 — Quality and Hardware Measurement

3. **Inference quality** (`crates/ironmill-bench/src/quality.rs`)
   - Perplexity computation for LLM models (Qwen3, Llama)
   - Top-k accuracy for encoder/classification models
   - WER for ASR models (Whisper)
   - Cross-tabulated quality × optimization report
   - Dataset download via `scripts/download-fixtures.sh`

4. **Energy sampling** (`crates/ironmill-bench/src/power.rs`)
   - `powermetrics` wrapper with plist parsing
   - Background sampling during bench runs
   - Graceful fallback without root

5. **ANE utilization** (`crates/ironmill-bench/src/inference.rs`)
   - Instrument predict vs marshal vs total time
   - Report utilization % and dispatch overhead

6. **Memory profiling** (`crates/ironmill-bench/src/inference.rs`)
   - RSS sampling via `mach_task_basic_info`
   - Leak detection over measurement window

### Phase 3 — Comparison and Tracking

7. **Backend comparison** (`crates/ironmill-bench/src/report.rs`)
   - Multi-backend loop in `main.rs`
   - Side-by-side comparison table with best-per-metric highlighting

8. **JSON output** (`crates/ironmill-bench/src/report.rs`)
   - Stable schema with hardware detection
   - `--format json` CLI flag

9. **Regression tracking** (`crates/ironmill-bench/src/baseline.rs`)
   - Baseline save/load
   - Delta comparison with threshold-based status

### Phase 4 — Polish

10. **CLI improvements** — wire new flags and subcommands
11. **Results documentation** — run full benchmark suite, write up results

## Dependencies

```
flops-counter ──► throughput-metrics ──► backend-comparison ──► results-docs
inference-quality ──► results-docs
energy-metrics ──► backend-comparison
ane-utilization ──► backend-comparison
regression-tracking (independent)
json-output (independent)
memory-profiling (independent)
bench-cli depends on all metric items
results-docs depends on all metric items
```

## Success Criteria

This plan is succeeding when `ironmill-bench` can produce output like:

```
Qwen3-0.6B + FP16 — Inference Performance (M4 Max, macOS 15.4)
──────────────────────────────────────────────────────────────────────
Backend  TTFT     Decode tok/s  TFLOPS  Watts  Tok/Watt  Util%
───────  ──────   ────────────  ──────  ─────  ────────  ─────
CPU      48ms     21 tok/s      0.08    8.2W   2.6        —
GPU      22ms     45 tok/s      0.17    12.1W  3.7        —
ANE      8ms      125 tok/s     0.47    3.8W   32.9      91%
All      8ms      125 tok/s     0.47    3.9W   32.1      91%

Qwen3-0.6B — Quality Impact
─────────────────────────────────────────────────────────────
Optimization       Weight SNR   Perplexity   Δ PPL    Status
─────────────────  ──────────   ──────────   ──────   ──────
FP32 baseline      ∞            12.4         —        —
FP16               73.6 dB      12.4         +0.0%    ✓ SAFE
INT8               38.5 dB      12.8         +3.2%    ✓ SAFE
Palettize 4-bit    19.9 dB      14.1         +13.7%   ⚠ WARN
PolarQuant 4-bit   —            12.9         +4.0%    ✓ SAFE

Regression check vs "v0.1.0": 4/4 OK, 0 WARN, 0 FAIL
```

This output tells the full story: ANE is 6× faster than CPU at decoding,
uses 2× less power, delivers 13× better token efficiency per watt, and
FP16 quantization preserves model quality perfectly. That is the value
proposition that the current latency-only benchmarks miss.

## Related Documents

- `docs/BENCH_HARNESS_PLAN.md` — original benchmark harness architecture
- `docs/BENCHMARK_RESULTS.md` — current benchmark results
- `docs/INFERENCE_IMPROVEMENTS_PLAN.md` — broader inference improvements (this
  plan implements item #4)
- `docs/ane-direct-runtime-plan.md` — ANE direct backend plan
