//! Criterion benchmarks for report formatting.
//!
//! Run: `cargo bench -p ironmill-bench --bench report_format`

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
struct FakeBenchResult {
    config_label: String,
    mean: f64,
    stddev: f64,
    median: f64,
    p95: f64,
    p99: f64,
    min: f64,
    max: f64,
    cv: f64,
    inferences_per_sec: f64,
}

fn make_rows(n: usize) -> Vec<FakeBenchResult> {
    (0..n)
        .map(|i| FakeBenchResult {
            config_label: format!("model-{i}/opt-{}/backend-{}", i % 5, i % 3),
            mean: 5.0 + (i as f64) * 0.1,
            stddev: 0.3 + (i as f64) * 0.01,
            median: 4.9 + (i as f64) * 0.1,
            p95: 5.5 + (i as f64) * 0.1,
            p99: 6.0 + (i as f64) * 0.1,
            min: 4.0 + (i as f64) * 0.1,
            max: 7.0 + (i as f64) * 0.1,
            cv: 0.06,
            inferences_per_sec: 200.0 - (i as f64),
        })
        .collect()
}

fn format_csv(rows: &[FakeBenchResult]) -> String {
    let mut out = String::with_capacity(rows.len() * 120);
    out.push_str("config,mean,stddev,median,p95,p99,min,max,cv,inf_per_sec\n");
    for r in rows {
        use std::fmt::Write;
        writeln!(
            out,
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.2}",
            r.config_label,
            r.mean,
            r.stddev,
            r.median,
            r.p95,
            r.p99,
            r.min,
            r.max,
            r.cv,
            r.inferences_per_sec
        )
        .unwrap();
    }
    out
}

fn format_json(rows: &[FakeBenchResult]) -> String {
    serde_json::to_string_pretty(rows).unwrap()
}

fn bench_format_csv(c: &mut Criterion) {
    let rows_10 = make_rows(10);
    let rows_100 = make_rows(100);

    let mut group = c.benchmark_group("format_csv");
    group.bench_function("10_rows", |b| b.iter(|| format_csv(black_box(&rows_10))));
    group.bench_function("100_rows", |b| b.iter(|| format_csv(black_box(&rows_100))));
    group.finish();
}

fn bench_format_json(c: &mut Criterion) {
    let rows_10 = make_rows(10);
    let rows_100 = make_rows(100);

    let mut group = c.benchmark_group("format_json");
    group.bench_function("10_rows", |b| b.iter(|| format_json(black_box(&rows_10))));
    group.bench_function("100_rows", |b| b.iter(|| format_json(black_box(&rows_100))));
    group.finish();
}

fn bench_config_parse(c: &mut Criterion) {
    let toml_content = r#"
[[model]]
name = "TestModel"
path = "test.onnx"

[[optimization]]
name = "baseline"
no_fusion = true

[[optimization]]
name = "fp16"
quantize = "fp16"

[[optimization]]
name = "int8"
quantize = "int8"

[[optimization]]
name = "polar-4"
polar_quantize = 4

[[optimization]]
name = "d2quant-2"
d2quant = 2
kv_quant = "turbo-int8"
max_seq_len = 4096

[settings]
iterations = 100
warmup = 10
runs = 5
backends = ["cpu", "gpu", "metal"]

[benchmarks]
suites = ["decode", "prefill"]
prefill_lengths = [128, 512, 1024, 2048]
quality = true
"#;

    c.bench_function("config_parse_toml", |b| {
        b.iter(|| {
            let _: toml::Value = toml::from_str(black_box(toml_content)).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_format_csv,
    bench_format_json,
    bench_config_parse
);
criterion_main!(benches);
