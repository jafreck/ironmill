use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Results from a single benchmark measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub config_label: String,
    pub latencies_ms: Vec<f64>,
    pub mean: f64,
    pub stddev: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
    pub cv: f64,

    // Throughput fields
    /// Inferences per second (1000.0 / median_ms).
    #[serde(default)]
    pub inferences_per_sec: f64,
    /// Achieved TFLOPS (model_flops / median_sec / 1e12).
    #[serde(default)]
    pub tflops: Option<f64>,
    /// Tokens per second for autoregressive models.
    #[serde(default)]
    pub tokens_per_sec: Option<f64>,
    /// Time to first token in ms (prefill latency).
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    /// Per-token decode throughput (tok/s).
    #[serde(default)]
    pub decode_tok_per_sec: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceResult {
    pub significant: bool,
    pub p_value: f64,
    pub effect_size: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub method: String,
}

/// Aggregated results across multiple runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedResult {
    pub config_label: String,
    pub pooled: BenchResult,
    pub per_run_means: Vec<f64>,
    pub between_run_stddev: f64,
    pub runs: usize,
}

/// Compute statistics from a vector of latencies (in milliseconds).
pub fn compute_stats(label: &str, latencies: &[f64]) -> BenchResult {
    compute_stats_with_flops(label, latencies, None)
}

/// Compute statistics with optional FLOPs count for TFLOPS calculation.
pub fn compute_stats_with_flops(
    label: &str,
    latencies: &[f64],
    model_flops: Option<u64>,
) -> BenchResult {
    if latencies.is_empty() {
        return BenchResult {
            config_label: label.to_string(),
            latencies_ms: vec![],
            mean: 0.0,
            stddev: 0.0,
            median: 0.0,
            p95: 0.0,
            p99: 0.0,
            min: 0.0,
            max: 0.0,
            cv: 0.0,
            inferences_per_sec: 0.0,
            tflops: None,
            tokens_per_sec: None,
            ttft_ms: None,
            decode_tok_per_sec: None,
        };
    }

    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;

    let stddev = if sorted.len() > 1 {
        let variance = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt()
    } else {
        0.0
    };

    let median = percentile(&sorted, 0.5);
    let p95 = percentile(&sorted, 0.95);
    let p99 = percentile(&sorted, 0.99);
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let cv = if mean != 0.0 { stddev / mean } else { 0.0 };

    let inferences_per_sec = if median > 0.0 { 1000.0 / median } else { 0.0 };
    let tflops = model_flops.map(|flops| {
        let median_sec = median / 1000.0;
        if median_sec > 0.0 {
            flops as f64 / median_sec / 1e12
        } else {
            0.0
        }
    });

    BenchResult {
        config_label: label.to_string(),
        latencies_ms: latencies.to_vec(),
        mean,
        stddev,
        median,
        p95,
        p99,
        min,
        max,
        cv,
        inferences_per_sec,
        tflops,
        tokens_per_sec: None,
        ttft_ms: None,
        decode_tok_per_sec: None,
    }
}

/// Compare two results for statistical significance using Welch's t-test.
pub fn welch_t_test(a: &BenchResult, b: &BenchResult, alpha: f64) -> SignificanceResult {
    let n_a = a.latencies_ms.len() as f64;
    let n_b = b.latencies_ms.len() as f64;

    if n_a < 2.0 || n_b < 2.0 {
        return SignificanceResult {
            significant: false,
            p_value: 1.0,
            effect_size: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            method: "welch_t_test".to_string(),
        };
    }

    let var_a = a.stddev.powi(2);
    let var_b = b.stddev.powi(2);

    let se = (var_a / n_a + var_b / n_b).sqrt();

    if se == 0.0 {
        return SignificanceResult {
            significant: false,
            p_value: 1.0,
            effect_size: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            method: "welch_t_test".to_string(),
        };
    }

    let t_stat = (a.mean - b.mean) / se;

    // Welch-Satterthwaite degrees of freedom
    let num = (var_a / n_a + var_b / n_b).powi(2);
    let denom = (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
    let df = num / denom;

    // Two-tailed p-value
    let dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let p_value = 2.0 * (1.0 - dist.cdf(t_stat.abs()));

    // Cohen's d
    let pooled_stddev = ((var_a + var_b) / 2.0).sqrt();
    let effect_size = if pooled_stddev > 0.0 {
        (a.mean - b.mean) / pooled_stddev
    } else {
        0.0
    };

    // 95% CI for difference of means
    let t_crit = dist.inverse_cdf(1.0 - alpha / 2.0);
    let diff = a.mean - b.mean;
    let ci_lower = diff - t_crit * se;
    let ci_upper = diff + t_crit * se;

    SignificanceResult {
        significant: p_value < alpha,
        p_value,
        effect_size,
        ci_lower,
        ci_upper,
        method: "welch_t_test".to_string(),
    }
}

/// Aggregate multiple runs of the same configuration.
pub fn aggregate_runs(label: &str, runs: &[BenchResult]) -> AggregatedResult {
    let all_latencies: Vec<f64> = runs
        .iter()
        .flat_map(|r| r.latencies_ms.iter().copied())
        .collect();
    let pooled = compute_stats(label, &all_latencies);

    let per_run_means: Vec<f64> = runs.iter().map(|r| r.mean).collect();

    let between_run_stddev = if per_run_means.len() > 1 {
        let n = per_run_means.len() as f64;
        let mean = per_run_means.iter().sum::<f64>() / n;
        let variance = per_run_means
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt()
    } else {
        0.0
    };

    AggregatedResult {
        config_label: label.to_string(),
        pooled,
        per_run_means,
        between_run_stddev,
        runs: runs.len(),
    }
}

/// Compute a percentile (0.0 to 1.0) from sorted values.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let rank = p * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_stats_known_values() {
        let latencies = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = compute_stats("test", &latencies);

        assert_eq!(result.config_label, "test");
        assert!((result.mean - 5.0).abs() < 1e-10);
        assert_eq!(result.min, 2.0);
        assert_eq!(result.max, 9.0);
        assert!((result.median - 4.5).abs() < 1e-10);
        assert!(result.stddev > 0.0);
        assert!(result.cv > 0.0);
    }

    #[test]
    fn test_compute_stats_single_value() {
        let result = compute_stats("single", &[5.0]);
        assert!((result.mean - 5.0).abs() < 1e-10);
        assert!((result.stddev).abs() < 1e-10);
        assert!((result.median - 5.0).abs() < 1e-10);
        assert!((result.min - 5.0).abs() < 1e-10);
        assert!((result.max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_empty() {
        let result = compute_stats("empty", &[]);
        assert_eq!(result.mean, 0.0);
        assert_eq!(result.stddev, 0.0);
        assert!(result.latencies_ms.is_empty());
    }

    #[test]
    fn test_compute_stats_stddev() {
        // Known: [2, 4, 4, 4, 5, 5, 7, 9], mean=5, sample variance = 4.571..., stddev ≈ 2.138
        let latencies = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = compute_stats("test", &latencies);
        let expected_var: f64 = 32.0 / 7.0; // sum of squared deviations / (n-1)
        let expected_stddev = expected_var.sqrt();
        assert!((result.stddev - expected_stddev).abs() < 1e-10);
    }

    #[test]
    fn test_welch_t_test_different_distributions() {
        // Two clearly different distributions
        let a_latencies: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64) * 0.01).collect();
        let b_latencies: Vec<f64> = (0..100).map(|i| 20.0 + (i as f64) * 0.01).collect();

        let a = compute_stats("fast", &a_latencies);
        let b = compute_stats("slow", &b_latencies);

        let result = welch_t_test(&a, &b, 0.05);
        assert!(result.significant, "should detect significant difference");
        assert!(result.p_value < 0.05);
        assert!(
            result.effect_size.abs() > 0.5,
            "should have large effect size"
        );
        assert_eq!(result.method, "welch_t_test");
    }

    #[test]
    fn test_welch_t_test_identical_distributions() {
        let latencies: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64) * 0.01).collect();
        let a = compute_stats("a", &latencies);
        let b = compute_stats("b", &latencies);

        let result = welch_t_test(&a, &b, 0.05);
        assert!(
            !result.significant,
            "identical distributions should not be significant"
        );
        assert!(result.p_value >= 0.05);
    }

    #[test]
    fn test_welch_t_test_insufficient_data() {
        let a = compute_stats("a", &[5.0]);
        let b = compute_stats("b", &[10.0]);
        let result = welch_t_test(&a, &b, 0.05);
        assert!(!result.significant);
        assert_eq!(result.p_value, 1.0);
    }

    #[test]
    fn test_welch_t_test_confidence_interval() {
        let a_latencies: Vec<f64> = (0..50).map(|i| 5.0 + (i as f64) * 0.02).collect();
        let b_latencies: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64) * 0.02).collect();
        let a = compute_stats("a", &a_latencies);
        let b = compute_stats("b", &b_latencies);

        let result = welch_t_test(&a, &b, 0.05);
        // The CI should contain the actual difference (≈ -5.0)
        assert!(result.ci_lower < -4.0);
        assert!(result.ci_upper > -6.0);
    }

    #[test]
    fn test_aggregate_runs() {
        let run1 = compute_stats("cfg", &vec![5.0, 5.5, 6.0]);
        let run2 = compute_stats("cfg", &vec![5.1, 5.4, 5.9]);
        let run3 = compute_stats("cfg", &vec![5.2, 5.6, 6.1]);

        let agg = aggregate_runs("cfg", &[run1, run2, run3]);

        assert_eq!(agg.config_label, "cfg");
        assert_eq!(agg.runs, 3);
        assert_eq!(agg.per_run_means.len(), 3);
        assert_eq!(agg.pooled.latencies_ms.len(), 9);
        assert!(agg.between_run_stddev >= 0.0);
    }

    #[test]
    fn test_aggregate_single_run() {
        let run = compute_stats("cfg", &vec![5.0, 5.5, 6.0]);
        let agg = aggregate_runs("cfg", &[run]);
        assert_eq!(agg.runs, 1);
        assert_eq!(agg.between_run_stddev, 0.0);
    }

    #[test]
    fn test_percentile_basic() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&sorted, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile(&sorted, 0.5) - 3.0).abs() < 1e-10);
        assert!((percentile(&sorted, 1.0) - 5.0).abs() < 1e-10);
        assert!((percentile(&sorted, 0.25) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_interpolation() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0];
        // p=0.5 → rank = 0.5 * 3 = 1.5 → interpolate between index 1 and 2
        let p50 = percentile(&sorted, 0.5);
        assert!((p50 - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_cv_coefficient_of_variation() {
        let latencies = vec![10.0, 10.0, 10.0, 10.0];
        let result = compute_stats("uniform", &latencies);
        assert!((result.cv).abs() < 1e-10, "cv should be 0 for uniform data");

        let latencies2 = vec![1.0, 10.0, 1.0, 10.0];
        let result2 = compute_stats("varied", &latencies2);
        assert!(result2.cv > 0.0, "cv should be positive for varied data");
    }
}
