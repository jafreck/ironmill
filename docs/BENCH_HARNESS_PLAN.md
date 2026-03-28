# Rust Inference Benchmark Harness — Plan

## Goal

Replace `scripts/bench-inference.sh` + `benches/swift/InferenceBench/` with a
single Rust binary that compiles models, runs inference via CoreML FFI, computes
statistics, and produces structured output.

## Architecture

```
crates/ironmill-bench/
├── Cargo.toml
└── src/
    ├── main.rs          # CLI entry point
    ├── config.rs        # Benchmark matrix definition
    ├── compiler.rs      # Model compilation (reuses mil-rs)
    ├── inference.rs     # CoreML inference via coreml-rs
    ├── stats.rs         # Statistics (mean, stddev, percentiles, significance)
    └── report.rs        # Output formatting (table, JSON, CSV, markdown)
```

## Dependencies

- `mil-rs` (workspace) — ONNX→mlpackage compilation pipeline
- `coreml-rs` — safe Rust CoreML bindings (MLModel, MLMultiArray, prediction)
- `clap` (workspace) — CLI args
- `serde` + `serde_json` (workspace) — structured output
- `statrs` — statistical tests (t-test, Mann-Whitney U)

## Benchmark Matrix

A benchmark run is the cartesian product of three dimensions:

```rust
struct BenchMatrix {
    models: Vec<ModelConfig>,       // ONNX paths + input shapes
    optimizations: Vec<OptConfig>,  // quantize, palettize, fusion on/off, per-pass
    backends: Vec<ComputeUnits>,    // cpu, gpu, ane, all
}

struct ModelConfig {
    name: String,
    path: PathBuf,
    input_shapes: Vec<(String, Vec<usize>)>,  // for --input-shape
}

struct OptConfig {
    name: String,
    quantize: Option<String>,       // "fp16", "int8"
    palettize: Option<u8>,          // 2, 4, 6, 8
    no_fusion: bool,
    disabled_passes: Vec<String>,   // e.g. ["conv-relu-fusion"] to isolate pass value
}
```

### Default matrix (matches current bench-inference.sh)
- Models: mobilenetv2.onnx, squeezenet1.1.onnx
- Optimizations: no-fusion, default, +fp16, +int8, +palettize-4
- Backends: all (ANE)

### Extended matrix (for per-pass analysis)
- Optimizations: each always-on pass disabled individually to measure its
  contribution. Requires a `--disabled-passes` CLI flag or config file.

## Config File (optional)

For complex matrices, support a TOML config:

```toml
[[model]]
name = "MobileNetV2"
path = "tests/fixtures/mobilenetv2.onnx"

[[model]]
name = "SqueezeNet"
path = "tests/fixtures/squeezenet1.1.onnx"

[[optimization]]
name = "baseline"
no_fusion = true

[[optimization]]
name = "default"

[[optimization]]
name = "fp16"
quantize = "fp16"

[[optimization]]
name = "int8"
quantize = "int8"

[[optimization]]
name = "palettize-4"
palettize = 4

[settings]
iterations = 200
warmup = 20
runs = 3                    # repeat full benchmark N times
backends = ["cpu", "gpu", "ane"]
```

## Inference via coreml-rs

```rust
use coreml_rs::{Model, BorrowedTensor, ComputeUnits};

fn run_inference(
    mlmodelc_path: &Path,
    compute_units: ComputeUnits,
    iterations: usize,
    warmup: usize,
) -> Vec<Duration> {
    let model = Model::load(mlmodelc_path, compute_units).unwrap();

    // Build dummy input from model description
    let desc = model.input_description();
    let input = build_dummy_input(&desc);

    // Warmup
    for _ in 0..warmup {
        model.predict(&input).unwrap();
    }

    // Timed runs
    (0..iterations).map(|_| {
        let start = Instant::now();
        model.predict(&input).unwrap();
        start.elapsed()
    }).collect()
}
```

## Statistics Module

```rust
struct BenchResult {
    config_label: String,
    latencies_ms: Vec<f64>,
    // Derived:
    mean: f64,
    stddev: f64,
    median: f64,
    p95: f64,
    p99: f64,
    min: f64,
    max: f64,
    cv: f64,              // coefficient of variation (stddev/mean)
}

/// Compare two results for statistical significance.
fn is_significant(a: &BenchResult, b: &BenchResult, alpha: f64) -> SignificanceResult {
    // Welch's t-test (unequal variances) or Mann-Whitney U
    // Returns: p-value, effect size, confidence interval
}
```

### Multi-run aggregation

When `runs > 1`, each run produces a `BenchResult`. Aggregate by:
1. Pool all latencies across runs (for percentiles)
2. Compute per-run means, then stddev-of-means (for run-to-run variance)
3. Report both within-run and between-run variance

## Compilation Pipeline

Reuse `mil-rs` directly — no subprocess:

```rust
use mil_rs::{read_onnx, onnx_to_program, PassPipeline, program_to_model, write_mlpackage};
use mil_rs::compiler::compile_model;

fn compile(model: &ModelConfig, opt: &OptConfig, work_dir: &Path) -> PathBuf {
    let onnx = read_onnx(&model.path).unwrap();
    let (mut program, _) = onnx_to_program(&onnx).unwrap();

    let mut pipeline = PassPipeline::default();
    if opt.no_fusion { pipeline.disable_fusion(); }
    if let Some(q) = &opt.quantize { pipeline.with_quantize(q); }
    if let Some(b) = opt.palettize { pipeline.with_palettize(b); }
    pipeline.run(&mut program).unwrap();

    let model_proto = program_to_model(&program, 7).unwrap();
    let mlpackage = work_dir.join("model.mlpackage");
    write_mlpackage(&model_proto, &mlpackage).unwrap();
    compile_model(&mlpackage, work_dir).unwrap()
}
```

## CLI Interface

```
ironmill-bench [OPTIONS]

Options:
  -c, --config <PATH>       TOML config file (default: built-in matrix)
  -m, --model <PATH>        Add model (repeatable, overrides config)
  -i, --iterations <N>      Iterations per measurement [default: 200]
  -w, --warmup <N>          Warmup iterations [default: 20]
  -r, --runs <N>            Number of full runs [default: 3]
  -b, --backend <BACKEND>   Backend filter: cpu, gpu, ane, all [default: all]
  -o, --output <FORMAT>     Output format: table, json, csv, markdown [default: table]
      --baseline <NAME>     Mark a config as baseline for significance tests
      --alpha <FLOAT>       Significance level [default: 0.05]
```

### Example usage

```bash
# Quick comparison
ironmill-bench -m tests/fixtures/mobilenetv2.onnx --runs 3

# Full matrix with all backends
ironmill-bench --config bench.toml -b cpu,gpu,ane -o markdown

# Per-pass ablation study
ironmill-bench -m tests/fixtures/mobilenetv2.onnx \
  --baseline "default" --alpha 0.01 -o json
```

## Output Formats

### Table (default)
```
MobileNetV2 — 3 runs × 200 iterations
──────────────────────────────────────────────────────────────
Configuration      CPU              GPU              ANE
                   mean±sd  p50     mean±sd  p50     mean±sd  p50
─────────────────  ───────  ──────  ───────  ──────  ───────  ──────
No optimization    5.6±0.3  5.7ms   2.7±0.2  2.8ms   2.3±0.4  2.4ms
Default            5.5±0.2  5.5ms   2.6±0.3  2.6ms   2.2±0.3  2.1ms
+ FP16             3.2±0.1  3.2ms   2.3±0.2  2.3ms   0.9±0.1  0.9ms  ***
+ INT8             5.5±0.2  5.5ms   3.8±0.9  4.6ms   2.1±0.1  2.1ms
+ Palettize 4-bit  6.1±0.8  5.0ms   2.4±0.2  2.5ms   3.1±1.5  2.0ms

*** p < 0.001 vs baseline (Welch's t-test)
```

### JSON (machine-readable)
Full structured output with all latencies, stats, and significance results.

### Markdown
Same as table but formatted for docs — can be piped directly into
BENCHMARK_RESULTS.md.

## Implementation Order

1. `stats.rs` — pure computation, easy to test
2. `config.rs` — matrix definition + TOML parsing
3. `compiler.rs` — thin wrapper around mil-rs pipeline
4. `inference.rs` — coreml-rs integration
5. `report.rs` — output formatting
6. `main.rs` — CLI wiring
7. Integration tests with fixture models

## Open Questions

- **coreml-rs maturity**: Verify it handles MLMultiArray creation and
  compute unit selection. If not, fall back to `objc2-core-ml` directly
  (more boilerplate but full control).
- **Per-pass ablation**: Requires `PassPipeline` to support disabling
  individual passes by name. Check if this API exists or needs adding.
- **Warm-start compilation**: Cache .mlmodelc across runs to separate
  compilation time from inference time in measurements.
