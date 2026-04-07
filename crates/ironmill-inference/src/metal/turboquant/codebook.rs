//! Re-export shared Lloyd-Max codebook from `crate::turboquant::codebook`,
//! plus per-head adaptive codebook training.

pub use crate::turboquant::codebook::*;

/// Per-head codebook configuration.
/// When `None`, all heads share a single codebook (current behavior).
/// When `Some`, each KV head has its own optimized codebook trained from
/// calibration data for that head's value distribution.
#[derive(Debug, Clone)]
pub struct PerHeadCodebooks {
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Per-head codebook tables, each `[2^bits]` entries.
    /// Outer vec: [num_kv_heads], inner vec: codebook entries for that head.
    pub codebooks: Vec<Vec<f32>>,
}

/// Train per-head codebooks from calibration KV cache snapshots.
///
/// `kv_snapshots` is `[num_kv_heads][sample_values]` — representative value
/// samples per head collected during calibration.
/// Uses Lloyd-Max iteration on each head independently.
pub fn train_per_head_codebooks(
    kv_snapshots: &[Vec<f32>],
    n_bits: u8,
    max_iterations: usize,
) -> PerHeadCodebooks {
    let num_kv_heads = kv_snapshots.len();
    let n_levels: usize = 1 << n_bits;

    let codebooks: Vec<Vec<f32>> = kv_snapshots
        .iter()
        .map(|samples| train_codebook_from_samples(samples, n_levels, max_iterations))
        .collect();

    PerHeadCodebooks {
        num_kv_heads,
        codebooks,
    }
}

/// Train a single codebook from empirical samples using Lloyd-Max iteration.
///
/// Unlike the Gaussian-assumption `lloyd_max_gaussian`, this operates on
/// actual calibration data, yielding codebooks adapted to each head's
/// real value distribution.
fn train_codebook_from_samples(
    samples: &[f32],
    n_levels: usize,
    max_iterations: usize,
) -> Vec<f32> {
    if samples.is_empty() || n_levels == 0 {
        return vec![0.0; n_levels];
    }

    // Filter out non-finite values (NaN, Inf) to prevent codebook corruption.
    let mut sorted: Vec<f32> = samples.iter().copied().filter(|s| s.is_finite()).collect();
    if sorted.is_empty() {
        return vec![0.0; n_levels];
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Initialize levels at uniformly spaced quantiles of the empirical distribution.
    let mut levels: Vec<f64> = (0..n_levels)
        .map(|i| {
            let frac = (i as f64 + 0.5) / n_levels as f64;
            let idx = ((frac * sorted.len() as f64) as usize).min(sorted.len() - 1);
            sorted[idx] as f64
        })
        .collect();

    // Lloyd-Max iterations on empirical data.
    for _ in 0..max_iterations {
        let boundaries = midpoints(&levels);

        // Compute centroids as mean of samples in each bin.
        let mut sums = vec![0.0f64; n_levels];
        let mut counts = vec![0usize; n_levels];

        for &s in samples {
            let val = s as f64;
            // Find the bin for this sample via boundary search.
            let bin = boundaries
                .iter()
                .position(|&b| val < b)
                .unwrap_or(n_levels - 1);
            sums[bin] += val;
            counts[bin] += 1;
        }

        let mut new_levels = Vec::with_capacity(n_levels);
        let mut max_delta = 0.0f64;
        for i in 0..n_levels {
            let centroid = if counts[i] > 0 {
                sums[i] / counts[i] as f64
            } else {
                levels[i]
            };
            max_delta = max_delta.max((centroid - levels[i]).abs());
            new_levels.push(centroid);
        }

        levels = new_levels;
        if max_delta < 1e-10 {
            break;
        }
    }

    levels.iter().map(|&v| v as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_head_codebooks_basic() {
        // Two heads with different distributions.
        let head0: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.01).collect();
        let head1: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.1).collect();

        let result = train_per_head_codebooks(&[head0, head1], 3, 100);
        assert_eq!(result.num_kv_heads, 2);
        assert_eq!(result.codebooks.len(), 2);
        assert_eq!(result.codebooks[0].len(), 8); // 2^3
        assert_eq!(result.codebooks[1].len(), 8);

        // Head 1 has wider distribution → wider codebook spread.
        let max0 = result.codebooks[0]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let max1 = result.codebooks[1]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max1 > max0,
            "wider distribution should produce wider codebook"
        );
    }

    #[test]
    fn per_head_codebooks_empty_samples() {
        let result = train_per_head_codebooks(&[vec![]], 2, 50);
        assert_eq!(result.num_kv_heads, 1);
        assert_eq!(result.codebooks[0].len(), 4);
    }
}
