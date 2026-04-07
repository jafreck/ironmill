//! Criterion benchmarks for statistical computation.
//!
//! Run: `cargo bench -p ironmill-bench --bench stats`

use criterion::{Criterion, black_box, criterion_group, criterion_main};

// Re-export from the binary crate's lib-like modules via include.
// Since ironmill-bench is a [[bin]], we benchmark the logic by
// duplicating the pure functions here. These are stable, math-only
// functions unlikely to drift.

use statrs::distribution::{ContinuousCDF, StudentsT};

fn compute_stats(latencies: &[f64]) -> (f64, f64, f64) {
    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;
    let variance = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let stddev = variance.sqrt();
    let median = {
        let rank = 0.5 * (sorted.len() - 1) as f64;
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        let frac = rank - lower as f64;
        if lower == upper {
            sorted[lower]
        } else {
            sorted[lower] * (1.0 - frac) + sorted[upper] * frac
        }
    };
    (mean, stddev, median)
}

fn welch_t_test(a_mean: f64, a_stddev: f64, a_n: f64, b_mean: f64, b_stddev: f64, b_n: f64) -> f64 {
    let var_a = a_stddev.powi(2);
    let var_b = b_stddev.powi(2);
    let se = (var_a / a_n + var_b / b_n).sqrt();
    if se == 0.0 {
        return 1.0;
    }
    let t_stat = (a_mean - b_mean) / se;
    let num = (var_a / a_n + var_b / b_n).powi(2);
    let denom = (var_a / a_n).powi(2) / (a_n - 1.0) + (var_b / b_n).powi(2) / (b_n - 1.0);
    let df = num / denom;
    let dist = StudentsT::new(0.0, 1.0, df).unwrap();
    2.0 * (1.0 - dist.cdf(t_stat.abs()))
}

fn bench_compute_stats(c: &mut Criterion) {
    let latencies_100: Vec<f64> = (0..100).map(|i| 5.0 + (i as f64) * 0.01).collect();
    let latencies_1000: Vec<f64> = (0..1000).map(|i| 5.0 + (i as f64) * 0.001).collect();
    let latencies_10000: Vec<f64> = (0..10000).map(|i| 5.0 + (i as f64) * 0.0001).collect();

    let mut group = c.benchmark_group("compute_stats");
    group.bench_function("100_samples", |b| {
        b.iter(|| compute_stats(black_box(&latencies_100)))
    });
    group.bench_function("1000_samples", |b| {
        b.iter(|| compute_stats(black_box(&latencies_1000)))
    });
    group.bench_function("10000_samples", |b| {
        b.iter(|| compute_stats(black_box(&latencies_10000)))
    });
    group.finish();
}

fn bench_welch_t_test(c: &mut Criterion) {
    c.bench_function("welch_t_test", |b| {
        b.iter(|| {
            welch_t_test(
                black_box(5.0),
                black_box(0.3),
                black_box(100.0),
                black_box(10.0),
                black_box(0.4),
                black_box(100.0),
            )
        })
    });
}

criterion_group!(benches, bench_compute_stats, bench_welch_t_test);
criterion_main!(benches);
