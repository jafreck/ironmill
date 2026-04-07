//! Criterion benchmarks for perplexity math.
//!
//! Run: `cargo bench -p ironmill-bench --bench perplexity_math`

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn cross_entropy(logits: &[f32], target: u32) -> f64 {
    let target = target as usize;
    if target >= logits.len() {
        return (logits.len() as f64).ln();
    }
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f64 = logits
        .iter()
        .map(|&x| ((x - max_logit) as f64).exp())
        .sum::<f64>()
        .ln()
        + max_logit as f64;
    let log_prob = logits[target] as f64 - log_sum_exp;
    -log_prob
}

fn perplexity_from_losses(losses: &[f64]) -> f64 {
    if losses.is_empty() {
        return f64::INFINITY;
    }
    let avg_ce = losses.iter().sum::<f64>() / losses.len() as f64;
    avg_ce.exp()
}

fn sliding_window_schedule(total_tokens: usize, max_length: usize, stride: usize) -> usize {
    let mut count = 0;
    let mut begin: usize = 0;
    let mut prev_end: usize = 0;
    while begin < total_tokens {
        let end = (begin + max_length).min(total_tokens);
        if end <= 1 {
            break;
        }
        let _ = end - prev_end;
        count += 1;
        prev_end = end;
        if end == total_tokens {
            break;
        }
        begin += stride;
    }
    count
}

fn bench_cross_entropy(c: &mut Criterion) {
    // Simulate logits for vocab_size=32000 (typical LLM)
    let logits: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.001) - 16.0).collect();

    let mut group = c.benchmark_group("cross_entropy");
    group.bench_function("vocab_32k", |b| {
        b.iter(|| cross_entropy(black_box(&logits), black_box(1500)))
    });

    let small_logits: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
    group.bench_function("vocab_128", |b| {
        b.iter(|| cross_entropy(black_box(&small_logits), black_box(42)))
    });
    group.finish();
}

fn bench_perplexity_from_losses(c: &mut Criterion) {
    let losses: Vec<f64> = (0..10000).map(|i| 2.0 + (i as f64) * 0.0001).collect();

    let mut group = c.benchmark_group("perplexity_from_losses");
    group.bench_function("10k_tokens", |b| {
        b.iter(|| perplexity_from_losses(black_box(&losses)))
    });
    group.finish();
}

fn bench_sliding_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("sliding_window_schedule");
    group.bench_function("4k_tokens_2k_window_512_stride", |b| {
        b.iter(|| sliding_window_schedule(black_box(4096), black_box(2048), black_box(512)))
    });
    group.bench_function("128k_tokens_2k_window_512_stride", |b| {
        b.iter(|| sliding_window_schedule(black_box(131072), black_box(2048), black_box(512)))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_cross_entropy,
    bench_perplexity_from_losses,
    bench_sliding_window
);
criterion_main!(benches);
