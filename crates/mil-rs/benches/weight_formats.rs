//! Criterion benchmarks for SafeTensors and GGUF model loading + template conversion.
//!
//! Uses real downloaded model files (Qwen3-0.6B) for meaningful performance
//! measurements. Download fixtures with `./scripts/download-fixtures.sh`.
//! Benchmarks are skipped gracefully when fixture files are absent.

use std::path::{Path, PathBuf};

use criterion::{Criterion, criterion_group, criterion_main};

use mil_rs::{PassPipeline, read_gguf, read_safetensors, weights_to_program};

fn fixture_path(name: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest)
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Qwen3-0.6B SafeTensors directory (~1.4GB) — only benchmarked when
/// the fixture directory is present with config.json + model.safetensors.
fn safetensors_fixture() -> Option<PathBuf> {
    let path = fixture_path("Qwen3-0.6B");
    if path.join("config.json").exists() && path.join("model.safetensors").exists() {
        Some(path)
    } else {
        None
    }
}

/// Qwen3-0.6B GGUF Q8_0 (~639MB) — only benchmarked when the fixture file
/// is present.
fn gguf_fixture() -> Option<PathBuf> {
    let path = fixture_path("Qwen3-0.6B-Q8_0.gguf");
    path.exists().then_some(path)
}

// ---------------------------------------------------------------------------
// SafeTensors benchmarks
// ---------------------------------------------------------------------------

fn bench_safetensors(c: &mut Criterion) {
    let Some(model_dir) = safetensors_fixture() else {
        eprintln!(
            "Skipping SafeTensors benchmarks — fixture not found. \
             Run ./scripts/download-fixtures.sh"
        );
        return;
    };

    let mut group = c.benchmark_group("safetensors");
    group.sample_size(10);

    group.bench_function("load", |b| {
        b.iter(|| {
            read_safetensors(&model_dir).expect("load safetensors");
        });
    });

    let provider = read_safetensors(&model_dir).expect("load");

    group.bench_function("to_program", |b| {
        b.iter(|| {
            weights_to_program(&provider).expect("template");
        });
    });

    group.bench_function("full_pipeline_fp16", |b| {
        b.iter(|| {
            let result = weights_to_program(&provider).expect("template");
            let mut prog = result.program;
            PassPipeline::new()
                .with_fp16()
                .expect("fp16")
                .run(&mut prog)
                .expect("pipeline");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// GGUF benchmarks
// ---------------------------------------------------------------------------

fn bench_gguf(c: &mut Criterion) {
    let Some(gguf_path) = gguf_fixture() else {
        eprintln!(
            "Skipping GGUF benchmarks — fixture not found. \
             Run ./scripts/download-fixtures.sh"
        );
        return;
    };

    let mut group = c.benchmark_group("gguf");
    group.sample_size(10);

    group.bench_function("load", |b| {
        b.iter(|| {
            read_gguf(&gguf_path).expect("load gguf");
        });
    });

    let provider = read_gguf(&gguf_path).expect("load");

    group.bench_function("to_program", |b| {
        b.iter(|| {
            weights_to_program(&provider).expect("template");
        });
    });

    group.bench_function("full_pipeline_fp16", |b| {
        b.iter(|| {
            let result = weights_to_program(&provider).expect("template");
            let mut prog = result.program;
            PassPipeline::new()
                .with_fp16()
                .expect("fp16")
                .run(&mut prog)
                .expect("pipeline");
        });
    });

    group.finish();
}

criterion_group!(benches, bench_safetensors, bench_gguf);
criterion_main!(benches);
