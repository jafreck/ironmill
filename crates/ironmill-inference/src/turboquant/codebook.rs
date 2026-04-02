//! Lloyd-Max optimal quantizer for N(0, 1/√d) distribution.
//!
//! Computes the data-independent codebook centroids and decision boundaries
//! used by TurboQuant for any head dimension `d`. After Hadamard rotation
//! of a unit-direction vector, each coordinate is approximately N(0, 1/√d).

use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Maximum number of Lloyd-Max iterations.
const MAX_ITERATIONS: usize = 200;

/// Convergence threshold for level updates.
const CONVERGENCE_EPS: f64 = 1e-10;

/// Compute the optimal Lloyd-Max codebook for N(0, 1/√d).
///
/// Returns `(levels, boundaries)` where:
/// - `levels` has `2^n_bits` centroids (f32, sorted ascending)
/// - `boundaries` has `2^n_bits - 1` decision thresholds (f32, sorted ascending)
pub fn lloyd_max_gaussian(dim: usize, n_bits: u8) -> (Vec<f32>, Vec<f32>) {
    let sigma = 1.0 / (dim as f64).sqrt();
    let n_levels: usize = 1 << n_bits;

    // Initialize levels at uniformly spaced quantiles of N(0, σ²).
    let mut levels: Vec<f64> = (0..n_levels)
        .map(|i| {
            let p = (i as f64 + 0.5) / n_levels as f64;
            sigma * normal_quantile(p)
        })
        .collect();

    // Lloyd-Max iterations.
    for _ in 0..MAX_ITERATIONS {
        let boundaries = midpoints(&levels);
        let new_levels = compute_centroids(sigma, &boundaries, n_levels);
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

    let boundaries = midpoints(&levels);
    (
        levels.iter().map(|&v| v as f32).collect(),
        boundaries.iter().map(|&v| v as f32).collect(),
    )
}

/// Compute midpoints between adjacent levels.
pub fn midpoints(levels: &[f64]) -> Vec<f64> {
    levels.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
}

/// Compute centroids (conditional means) for each bin of N(0, σ²).
///
/// E[X | a < X < b] for X ~ N(0, σ²) is:
///   σ² · (φ(a/σ) - φ(b/σ)) / (Φ(b/σ) - Φ(a/σ))
/// where φ is the standard normal PDF and Φ is the standard normal CDF.
pub fn compute_centroids(sigma: f64, boundaries: &[f64], n_levels: usize) -> Vec<f64> {
    let mut levels = Vec::with_capacity(n_levels);
    for i in 0..n_levels {
        let lo = if i == 0 {
            f64::NEG_INFINITY
        } else {
            boundaries[i - 1]
        };
        let hi = if i == n_levels - 1 {
            f64::INFINITY
        } else {
            boundaries[i]
        };
        levels.push(conditional_mean_gaussian(sigma, lo, hi));
    }
    levels
}

/// E[X | lo < X < hi] for X ~ N(0, σ²).
pub fn conditional_mean_gaussian(sigma: f64, lo: f64, hi: f64) -> f64 {
    let lo_z = lo / sigma;
    let hi_z = hi / sigma;

    let prob = normal_cdf(hi_z) - normal_cdf(lo_z);
    if prob < 1e-30 {
        return if lo.is_infinite() {
            hi
        } else if hi.is_infinite() {
            lo
        } else {
            (lo + hi) / 2.0
        };
    }

    // E[X | a < X < b] = σ · (φ(a/σ) - φ(b/σ)) / (Φ(b/σ) - Φ(a/σ))
    let numerator = normal_pdf(lo_z) - normal_pdf(hi_z);
    sigma * numerator / prob
}

/// Standard normal PDF: φ(x) = exp(-x²/2) / √(2π)
pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal CDF: Φ(x) using the erfc approximation.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * FRAC_1_SQRT_2))
}

/// Standard normal quantile (inverse CDF) via rational approximation.
pub fn normal_quantile(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let t = p - 0.5;
    if t.abs() <= 0.42 {
        // Central region: Rational approximation
        let r = t * t;
        let num = t
            * (((2.5066282746310002 * r + 18.6150006252966) * r + 41.39119773534996) * r
                + 25.44106049637689);
        let den = (((14.382029373388_f64 * r + 63.690613_f64) * r + 84.681_f64) * r + 40.304_f64)
            * r
            + 1.0;
        num / den
    } else {
        // Tail region
        let r = if t < 0.0 { p } else { 1.0 - p };
        let s = (-r.ln()).sqrt();
        let result = if s <= 5.0 {
            let ss = s - 1.6;
            let num = ((((((0.000124818987 * ss + 0.0227061845) * ss + 0.2728688) * ss
                + 1.3058437)
                * ss
                + 2.7580128)
                * ss
                + 2.2311569)
                * ss
                + 0.6462025)
                * ss
                + 0.001612;
            let den = ((((((0.000038746 * ss + 0.00745102) * ss + 0.09234118) * ss + 0.4710441)
                * ss
                + 1.0507501)
                * ss
                + 1.0)
                * ss
                + 0.2127601)
                * ss
                + 1.0;
            num / den
        } else {
            let ss = s - 5.0;
            (((((((0.0000000271 * ss + 0.00000342556) * ss + 0.000170717) * ss + 0.004531901)
                * ss
                + 0.06871337)
                * ss
                + 0.56419917)
                * ss
                + 1.9715909)
                * ss
                + 2.1538256)
                / (((((((0.0000000393 * ss + 0.00000489649) * ss + 0.000230147) * ss
                    + 0.005492439)
                    * ss
                    + 0.06353648)
                    * ss
                    + 0.3445453)
                    * ss
                    + 0.73741429)
                    * ss
                    + 1.0)
        };
        if t < 0.0 { -result } else { result }
    }
}

/// Error function approximation (Abramowitz and Stegun 7.1.28).
pub fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_is_sorted_and_symmetric() {
        for dim in [64, 128, 256] {
            let (levels, boundaries) = lloyd_max_gaussian(dim, 4);
            assert_eq!(levels.len(), 16);
            assert_eq!(boundaries.len(), 15);

            // Sorted ascending
            for w in levels.windows(2) {
                assert!(w[0] < w[1], "levels not sorted for dim={dim}");
            }
            for w in boundaries.windows(2) {
                assert!(w[0] < w[1], "boundaries not sorted for dim={dim}");
            }

            // Symmetric around zero (N(0, σ) is symmetric)
            for i in 0..8 {
                let lo = levels[i];
                let hi = levels[15 - i];
                assert!(
                    (lo + hi).abs() < 1e-5,
                    "levels not symmetric for dim={dim}: {lo} vs {hi}"
                );
            }
        }
    }

    #[test]
    fn codebook_scales_with_dim() {
        let (levels_64, _) = lloyd_max_gaussian(64, 4);
        let (levels_128, _) = lloyd_max_gaussian(128, 4);
        let (levels_256, _) = lloyd_max_gaussian(256, 4);

        // Larger dim → smaller spread (σ = 1/√d)
        let max_64 = levels_64.last().unwrap();
        let max_128 = levels_128.last().unwrap();
        let max_256 = levels_256.last().unwrap();
        assert!(max_64 > max_128, "d=64 should have wider spread than d=128");
        assert!(
            max_128 > max_256,
            "d=128 should have wider spread than d=256"
        );
    }

    #[test]
    fn three_bit_codebook() {
        let (levels, boundaries) = lloyd_max_gaussian(128, 3);
        assert_eq!(levels.len(), 8);
        assert_eq!(boundaries.len(), 7);
    }
}
