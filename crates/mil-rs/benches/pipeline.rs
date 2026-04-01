//! End-to-end ONNX → MIL IR → optimised pipeline benchmarks.

use std::path::{Path, PathBuf};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mil_rs::{PassPipeline, onnx_to_program, read_onnx};

fn fixture_path(name: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest)
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join(name)
}

const MODELS: &[&str] = &["mnist.onnx", "squeezenet1.1.onnx"];

/// Whisper Medium encoder (~1.5GB) — only included in benchmarks when
/// the fixture file is present. Download with `./scripts/download-fixtures.sh`.
fn whisper_fixture() -> Option<PathBuf> {
    let path = fixture_path("whisper-medium-encoder.onnx");
    path.exists().then_some(path)
}

fn bench_onnx_to_program(c: &mut Criterion) {
    let mut group = c.benchmark_group("onnx_to_program");
    for &model in MODELS {
        let mut onnx = read_onnx(fixture_path(model)).expect("read_onnx");
        group.bench_with_input(BenchmarkId::new("parse", model), &onnx, |b, onnx| {
            b.iter(|| {
                let mut model = onnx.clone();
                onnx_to_program(&mut model).expect("onnx_to_program")
            });
        });
    }
    group.finish();
}

fn bench_pipeline_default(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_default");
    for &model in MODELS {
        let mut onnx = read_onnx(fixture_path(model)).expect("read_onnx");
        let base = onnx_to_program(&mut onnx).expect("onnx_to_program");
        group.bench_with_input(BenchmarkId::new("run", model), &base.program, |b, prog| {
            b.iter(|| {
                let mut p = prog.clone();
                PassPipeline::new().run(&mut p).expect("pipeline");
            });
        });
    }
    group.finish();
}

fn bench_pipeline_fp16(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_fp16");
    for &model in MODELS {
        let mut onnx = read_onnx(fixture_path(model)).expect("read_onnx");
        let base = onnx_to_program(&mut onnx).expect("onnx_to_program");
        group.bench_with_input(BenchmarkId::new("run", model), &base.program, |b, prog| {
            b.iter(|| {
                let mut p = prog.clone();
                PassPipeline::new()
                    .with_fp16()
                    .expect("with_fp16")
                    .run(&mut p)
                    .expect("pipeline");
            });
        });
    }
    group.finish();
}

fn bench_pipeline_int8(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_int8");
    for &model in MODELS {
        let mut onnx = read_onnx(fixture_path(model)).expect("read_onnx");
        let base = onnx_to_program(&mut onnx).expect("onnx_to_program");
        group.bench_with_input(BenchmarkId::new("run", model), &base.program, |b, prog| {
            b.iter(|| {
                let mut p = prog.clone();
                PassPipeline::new()
                    .with_int8(None)
                    .expect("with_int8")
                    .run(&mut p)
                    .expect("pipeline");
            });
        });
    }
    group.finish();
}

fn bench_pipeline_palettize_4bit(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_palettize_4bit");
    for &model in MODELS {
        let mut onnx = read_onnx(fixture_path(model)).expect("read_onnx");
        let base = onnx_to_program(&mut onnx).expect("onnx_to_program");
        group.bench_with_input(BenchmarkId::new("run", model), &base.program, |b, prog| {
            b.iter(|| {
                let mut p = prog.clone();
                PassPipeline::new()
                    .with_palettize(4)
                    .expect("with_palettize")
                    .run(&mut p)
                    .expect("pipeline");
            });
        });
    }
    group.finish();
}

fn bench_whisper_pipeline(c: &mut Criterion) {
    let path = match whisper_fixture() {
        Some(p) => p,
        None => {
            eprintln!(
                "Skipping whisper benchmarks — fixture not found. Run ./scripts/download-fixtures.sh"
            );
            return;
        }
    };
    let mut onnx = read_onnx(&path).expect("read whisper onnx");
    let base = onnx_to_program(&mut onnx).expect("onnx_to_program whisper");

    let mut group = c.benchmark_group("whisper_medium_encoder");
    group.sample_size(10);

    group.bench_function("default", |b| {
        b.iter(|| {
            let mut p = base.program.clone();
            PassPipeline::new().run(&mut p).expect("pipeline");
        });
    });

    group.bench_function("fp16", |b| {
        b.iter(|| {
            let mut p = base.program.clone();
            PassPipeline::new()
                .with_fp16()
                .expect("with_fp16")
                .run(&mut p)
                .expect("pipeline");
        });
    });

    group.bench_function("int8", |b| {
        b.iter(|| {
            let mut p = base.program.clone();
            PassPipeline::new()
                .with_int8(None)
                .expect("with_int8")
                .run(&mut p)
                .expect("pipeline");
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_onnx_to_program,
    bench_pipeline_default,
    bench_pipeline_fp16,
    bench_pipeline_int8,
    bench_pipeline_palettize_4bit,
    bench_whisper_pipeline,
);
criterion_main!(benches);
