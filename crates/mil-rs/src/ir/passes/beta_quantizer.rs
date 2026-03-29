//! Beta-optimal scalar quantizer using the Lloyd-Max algorithm.
//!
//! Computes optimal reconstruction levels for quantizing a
//! Beta(0.5, (dim-1)/2) random variable — the distribution of squared
//! cosine similarities in high-dimensional spaces.

use statrs::distribution::{Beta, ContinuousCDF};

/// Maximum number of Lloyd-Max iterations before giving up.
const MAX_ITERATIONS: usize = 200;

/// Convergence threshold for level updates.
const CONVERGENCE_EPS: f64 = 1e-10;

/// Returns the optimal reconstruction levels for quantizing a
/// Beta(0.5, (dim-1)/2) random variable to `n_bits` bits.
///
/// Returns `2^n_bits` levels as f32, sorted ascending.
/// These are the Lloyd-Max centroids for the Beta distribution.
pub fn beta_optimal_levels(dim: usize, n_bits: u8) -> Vec<f32> {
    assert!(
        dim >= 2,
        "dimension must be >= 2 for valid Beta distribution parameters"
    );
    let (alpha, beta_param) = distribution_params(dim);
    let dist = Beta::new(alpha, beta_param).expect("invalid Beta distribution parameters");
    // Auxiliary distribution with alpha+1 for the conditional mean formula.
    let dist_shifted =
        Beta::new(alpha + 1.0, beta_param).expect("invalid shifted Beta distribution parameters");
    let n_levels: usize = 1 << n_bits;

    // Initialize levels at uniformly spaced quantiles.
    let mut levels: Vec<f64> = (0..n_levels)
        .map(|i| {
            let p = (i as f64 + 0.5) / n_levels as f64;
            dist.inverse_cdf(p)
        })
        .collect();

    // Lloyd-Max iterations.
    let mean_ratio = alpha / (alpha + beta_param);
    for _ in 0..MAX_ITERATIONS {
        let boundaries = midpoints(&levels);
        let new_levels = compute_centroids(&dist, &dist_shifted, mean_ratio, &boundaries, n_levels);
        let max_delta = levels
            .iter()
            .zip(new_levels.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        levels = new_levels;
        if max_delta < CONVERGENCE_EPS {
            break;
        }
    }

    levels.iter().map(|&v| v as f32).collect()
}

/// Returns the decision boundaries (midpoints between adjacent levels).
/// Length: `2^n_bits - 1`.
pub fn beta_optimal_boundaries(dim: usize, n_bits: u8) -> Vec<f32> {
    let levels = beta_optimal_levels(dim, n_bits);
    levels.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
}

/// Quantize a single value to the nearest level index.
///
/// Uses binary search on the boundaries array. Returns `0` if `value`
/// is below the first boundary, `boundaries.len()` if above the last.
pub fn quantize_to_index(value: f32, boundaries: &[f32]) -> u8 {
    match boundaries
        .binary_search_by(|b| b.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Equal))
    {
        Ok(i) => i as u8,
        Err(i) => i as u8,
    }
}

// ── helpers ──────────────────────────────────────────────────────────

/// Beta distribution parameters for the squared-cosine-similarity
/// distribution in `dim` dimensions.
fn distribution_params(dim: usize) -> (f64, f64) {
    let alpha = 0.5;
    let beta_param = (dim as f64 - 1.0) / 2.0;
    (alpha, beta_param)
}

/// Compute midpoints between adjacent levels.
fn midpoints(levels: &[f64]) -> Vec<f64> {
    levels.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
}

/// Compute the conditional mean (centroid) within each bin defined by
/// `boundaries`, using an analytical formula.
///
/// For Beta(α, β), the conditional mean on [a, b] is:
///   E[X | a < X < b] = (α / (α+β)) · [F_s(b) - F_s(a)] / [F(b) - F(a)]
/// where F is the CDF of Beta(α, β) and F_s is the CDF of Beta(α+1, β).
fn compute_centroids(
    dist: &Beta,
    dist_shifted: &Beta,
    mean_ratio: f64,
    boundaries: &[f64],
    n_levels: usize,
) -> Vec<f64> {
    let mut levels = Vec::with_capacity(n_levels);
    for i in 0..n_levels {
        let lo = if i == 0 { 0.0 } else { boundaries[i - 1] };
        let hi = if i == n_levels - 1 {
            1.0
        } else {
            boundaries[i]
        };
        let centroid = conditional_mean(dist, dist_shifted, mean_ratio, lo, hi);
        levels.push(centroid);
    }
    levels
}

/// Compute E[X | lo < X < hi] for a Beta(α, β) distribution using the
/// analytical CDF-based formula, avoiding numerical integration of the
/// singular PDF near x = 0.
fn conditional_mean(dist: &Beta, dist_shifted: &Beta, mean_ratio: f64, lo: f64, hi: f64) -> f64 {
    if (hi - lo) < 1e-15 {
        return (lo + hi) / 2.0;
    }

    let prob = dist.cdf(hi) - dist.cdf(lo);
    if prob < 1e-30 {
        return (lo + hi) / 2.0;
    }

    let numerator = dist_shifted.cdf(hi) - dist_shifted.cdf(lo);
    mean_ratio * numerator / prob
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_sorted_ascending() {
        for dim in [3, 16, 64, 512] {
            for n_bits in [1, 2, 4] {
                let levels = beta_optimal_levels(dim, n_bits);
                for w in levels.windows(2) {
                    assert!(
                        w[0] < w[1],
                        "levels not strictly ascending for dim={dim}, n_bits={n_bits}: {w:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn levels_count_matches_bits() {
        for n_bits in 1..=4u8 {
            let levels = beta_optimal_levels(32, n_bits);
            assert_eq!(
                levels.len(),
                1 << n_bits,
                "expected {} levels for n_bits={n_bits}",
                1 << n_bits
            );
        }
    }

    #[test]
    fn levels_within_unit_interval() {
        for dim in [3, 16, 128] {
            let levels = beta_optimal_levels(dim, 4);
            for &l in &levels {
                assert!(
                    (0.0..=1.0).contains(&l),
                    "level {l} outside [0, 1] for dim={dim}"
                );
            }
        }
    }

    #[test]
    fn quantize_round_trip_bounded_error() {
        let dim = 64;
        let n_bits = 4;
        let levels = beta_optimal_levels(dim, n_bits);
        let boundaries = beta_optimal_boundaries(dim, n_bits);

        // Maximum bin width gives us the error bound.
        let mut max_bin_width: f32 = 0.0;
        for i in 0..levels.len() {
            let lo = if i == 0 { 0.0_f32 } else { boundaries[i - 1] };
            let hi = if i == levels.len() - 1 {
                1.0_f32
            } else {
                boundaries[i]
            };
            max_bin_width = max_bin_width.max(hi - lo);
        }

        // Test quantize → dequantize on each level's centroid.
        for &level in &levels {
            let idx = quantize_to_index(level, &boundaries) as usize;
            let reconstructed = levels[idx];
            let error = (level - reconstructed).abs();
            assert!(
                error <= max_bin_width / 2.0 + 1e-6,
                "round-trip error {error} exceeds half max bin width {} for value {level}",
                max_bin_width / 2.0
            );
        }
    }

    #[test]
    fn boundaries_count_correct() {
        for n_bits in 1..=4u8 {
            let boundaries = beta_optimal_boundaries(32, n_bits);
            assert_eq!(
                boundaries.len(),
                (1 << n_bits) - 1,
                "expected {} boundaries for n_bits={n_bits}",
                (1 << n_bits) - 1
            );
        }
    }

    #[test]
    fn quantize_edge_values() {
        let boundaries = beta_optimal_boundaries(64, 4);
        let n_levels = boundaries.len() + 1;

        // Value 0.0 should map to the first index.
        assert_eq!(quantize_to_index(0.0, &boundaries), 0);

        // Value 1.0 should map to the last index.
        assert_eq!(quantize_to_index(1.0, &boundaries), (n_levels - 1) as u8);
    }
}
