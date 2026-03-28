//! Per-pass microbenchmarks on squeezenet.

use std::path::{Path, PathBuf};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mil_rs::ir::passes::{
    AttentionFusionPass, ConstantFoldPass, ConvBatchNormFusionPass, ConvBatchNormWeightFoldPass,
    ConvReluFusionPass, DeadCodeEliminationPass, GeluLinearFusionPass, GqaFusionPass,
    IdentityEliminationPass, LayerNormLinearFusionPass, LayoutOptimizationPass, OpSubstitutionPass,
    ResidualAddFusionPass,
};
use mil_rs::{Pass, Program, onnx_to_program, read_onnx};

fn fixture_path(name: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest)
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn count_ops(program: &Program) -> usize {
    program
        .functions
        .values()
        .map(|f| f.body.operations.len())
        .sum()
}

fn base_program() -> Program {
    let onnx = read_onnx(fixture_path("squeezenet1.1.onnx")).expect("read squeezenet");
    onnx_to_program(&onnx).expect("onnx_to_program").program
}

fn bench_passes(c: &mut Criterion) {
    let base = base_program();

    let passes: Vec<Box<dyn Pass>> = vec![
        Box::new(DeadCodeEliminationPass),
        Box::new(IdentityEliminationPass),
        Box::new(ConstantFoldPass),
        Box::new(ConvBatchNormWeightFoldPass),
        Box::new(ConvBatchNormFusionPass),
        Box::new(ConvReluFusionPass),
        Box::new(LayerNormLinearFusionPass),
        Box::new(GeluLinearFusionPass),
        Box::new(ResidualAddFusionPass),
        Box::new(AttentionFusionPass),
        Box::new(GqaFusionPass),
        Box::new(OpSubstitutionPass),
        Box::new(LayoutOptimizationPass),
    ];

    let mut group = c.benchmark_group("passes");
    for pass in &passes {
        let ops_before = count_ops(&base);
        let mut probe = base.clone();
        pass.run(&mut probe).expect("pass probe run");
        let ops_after = count_ops(&probe);

        let label = format!("{} ({ops_before}→{ops_after} ops)", pass.name());
        group.bench_with_input(BenchmarkId::new("run", &label), &base, |b, prog| {
            b.iter(|| {
                let mut p = prog.clone();
                pass.run(&mut p).expect("pass");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_passes);
criterion_main!(benches);
