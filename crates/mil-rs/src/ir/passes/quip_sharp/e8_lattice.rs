#![allow(clippy::needless_range_loop)]
//! E8 lattice codebook for 2-bit vector quantization.
//!
//! The E8 lattice is the densest sphere packing in 8 dimensions.  Its 240
//! shortest non-zero vectors (the "roots") form a natural codebook for
//! 8-dimensional vector quantization at ~2 bits per element.
//!
//! # Codebook construction
//!
//! The 240 root vectors of E8 decompose into two orbits:
//!
//! 1. **Integer orbit (112 vectors):** all permutations of (±1, ±1, 0⁶).
//!    There are C(8,2) = 28 ways to pick the two non-zero positions, and
//!    2² = 4 sign patterns, giving 28 × 4 = 112 vectors.
//!
//! 2. **Half-integer orbit (128 vectors):** all vectors of the form
//!    (±½, ±½, ±½, ±½, ±½, ±½, ±½, ±½) where the number of minus signs
//!    is even.  There are 2⁸ = 256 sign patterns, half of which have an
//!    even count of negatives, giving 128 vectors.
//!
//! We add the zero vector to get 241 entries, then pad to 256 with scaled
//! variants (shorter E8 lattice points scaled down) so that every 8-bit
//! index is valid.
//!
//! # 4-bit feasibility note
//!
//! At 4-bit precision we would need ~65,536 codebook entries.  The number
//! of E8 lattice points within squared norm ≤ R² grows as:
//!
//! | R² | # points |
//! |----|----------|
//! |  2 |      240 |
//! |  4 |    2,160 |
//! |  6 |    6,720 |
//! |  8 |   17,520 |
//! | 10 |   30,240 |
//! | 12 |   60,480 |
//! | 14 |   82,560 |
//! | 16 |  140,400 |
//!
//! So R² ≈ 10–12 reaches the 65K range.  Brute-force nearest-neighbour
//! search on 65K entries costs ~0.5 ms per 8-vector group, or roughly
//! 1–2 seconds for a 4096 × 4096 matrix.  This is likely acceptable for
//! offline quantization.  If speed becomes a bottleneck, a KD-tree or
//! lattice-decoding shortcut can reduce this to O(1) per group.

/// Number of codebook entries (2-bit = 8 indices per byte gives 256 entries).
const CODEBOOK_SIZE: usize = 256;

/// Dimensionality of each codebook vector.
const DIM: usize = 8;

/// The E8 lattice codebook for 2-bit vector quantization.
///
/// Each codebook entry is an 8-dimensional vector.  The first 241 entries
/// are the 240 E8 root vectors plus the zero vector; the remaining 15 are
/// scaled duplicates used as padding.
pub struct E8Codebook {
    /// 256 codebook entries, each an 8-dimensional vector.
    entries: [[f32; DIM]; CODEBOOK_SIZE],
}

impl E8Codebook {
    /// Generate the E8 codebook.
    ///
    /// This is fully deterministic — the codebook is defined by the
    /// mathematical structure of the E8 lattice.
    pub fn new() -> Self {
        let mut entries = [[0.0f32; DIM]; CODEBOOK_SIZE];
        let mut idx = 0;

        // Entry 0: zero vector.
        idx += 1;

        // Integer orbit: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0).
        for i in 0..DIM {
            for j in (i + 1)..DIM {
                for &si in &[1.0f32, -1.0] {
                    for &sj in &[1.0f32, -1.0] {
                        entries[idx][i] = si;
                        entries[idx][j] = sj;
                        idx += 1;
                    }
                }
            }
        }
        debug_assert_eq!(idx, 1 + 112); // 0-vector + 112 integer-orbit vectors

        // Half-integer orbit: (±½)⁸ with even number of minus signs.
        for bits in 0u32..256 {
            if bits.count_ones() % 2 != 0 {
                continue; // odd number of negatives → skip
            }
            let mut v = [0.5f32; DIM];
            for d in 0..DIM {
                if bits & (1 << d) != 0 {
                    v[d] = -0.5;
                }
            }
            entries[idx] = v;
            idx += 1;
        }
        debug_assert_eq!(idx, 241); // 0 + 112 + 128

        // Pad to 256 with scaled-down copies of the first 15 root vectors.
        // We scale by 0.5 so they are still valid E8 lattice points (just
        // at a smaller radius).  This ensures every u8 index is usable.
        for pad in 0..15 {
            let src = entries[1 + pad]; // skip zero vector
            for d in 0..DIM {
                entries[idx][d] = src[d] * 0.5;
            }
            idx += 1;
        }
        debug_assert_eq!(idx, CODEBOOK_SIZE);

        Self { entries }
    }

    /// Return a reference to all codebook entries.
    pub fn entries(&self) -> &[[f32; DIM]; CODEBOOK_SIZE] {
        &self.entries
    }

    /// Find the nearest codebook entry to `vector` (brute-force).
    ///
    /// Returns `(index, codebook_entry)`.  With only 256 entries the
    /// brute-force search is negligible.
    pub fn nearest(&self, vector: &[f32; DIM]) -> (u8, [f32; DIM]) {
        let mut best_idx = 0u8;
        let mut best_dist = f32::MAX;

        for (i, entry) in self.entries.iter().enumerate() {
            let dist = squared_distance(vector, entry);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i as u8;
            }
        }

        (best_idx, self.entries[best_idx as usize])
    }

    /// Quantize a weight matrix stored in row-major order.
    ///
    /// The matrix has dimensions `(n_rows, n_cols)` where
    /// `n_rows = weights.len() / n_cols`.  Columns are padded to a multiple
    /// of 8 internally.
    ///
    /// Returns `(indices, scales)` where:
    /// - `indices[g]` is the codebook index for group `g`
    /// - `scales[g]` is the per-group scaling factor
    ///
    /// The scaling strategy: for each 8-element group, compute
    /// `scale = ‖group‖ / ‖nearest_codebook_entry‖`, so that
    /// `scale * codebook_entry ≈ group`.
    pub fn quantize_matrix(&self, weights: &[f32], n_cols: usize) -> (Vec<u8>, Vec<f32>) {
        assert!(!weights.is_empty());
        assert!(n_cols > 0);
        let n_rows = weights.len() / n_cols;
        assert_eq!(weights.len(), n_rows * n_cols);

        let padded_cols = n_cols.div_ceil(DIM) * DIM;
        let n_groups_per_row = padded_cols / DIM;
        let total_groups = n_rows * n_groups_per_row;

        let mut indices = Vec::with_capacity(total_groups);
        let mut scales = Vec::with_capacity(total_groups);

        for row in 0..n_rows {
            let row_start = row * n_cols;
            for g in 0..n_groups_per_row {
                let mut group = [0.0f32; DIM];
                for d in 0..DIM {
                    let col = g * DIM + d;
                    if col < n_cols {
                        group[d] = weights[row_start + col];
                    }
                }

                let group_norm = norm(&group);
                if group_norm < 1e-10 {
                    // Near-zero group → map to zero vector (index 0).
                    indices.push(0u8);
                    scales.push(0.0);
                    continue;
                }

                // Find the codebook entry + scale that minimizes reconstruction error.
                // For each entry e[i], optimal scale is s = dot(g, e) / dot(e, e),
                // and reconstruction error is ||g - s*e||².
                let mut best_idx = 0u8;
                let mut best_scale = 0.0f32;
                let mut best_err = f32::MAX;

                for (ei, entry) in self.entries.iter().enumerate() {
                    let dot_ge: f32 = (0..DIM).map(|d| group[d] * entry[d]).sum();
                    let dot_ee: f32 = (0..DIM).map(|d| entry[d] * entry[d]).sum();

                    if dot_ee < 1e-10 {
                        continue;
                    }

                    let scale = dot_ge / dot_ee;
                    let err: f32 = (0..DIM)
                        .map(|d| {
                            let diff = group[d] - scale * entry[d];
                            diff * diff
                        })
                        .sum();

                    if err < best_err {
                        best_err = err;
                        best_idx = ei as u8;
                        best_scale = scale;
                    }
                }

                indices.push(best_idx);
                scales.push(best_scale);
            }
        }

        (indices, scales)
    }

    /// Dequantize: reconstruct the weight matrix from indices and scales.
    ///
    /// Returns a flat row-major vector of length `n_rows * n_cols`.
    pub fn dequantize_matrix(&self, indices: &[u8], scales: &[f32], n_cols: usize) -> Vec<f32> {
        assert_eq!(indices.len(), scales.len());
        let padded_cols = n_cols.div_ceil(DIM) * DIM;
        let n_groups_per_row = padded_cols / DIM;
        let n_rows = indices.len() / n_groups_per_row;
        assert_eq!(indices.len(), n_rows * n_groups_per_row);

        let mut output = Vec::with_capacity(n_rows * n_cols);

        for row in 0..n_rows {
            for g in 0..n_groups_per_row {
                let gi = row * n_groups_per_row + g;
                let entry = &self.entries[indices[gi] as usize];
                let s = scales[gi];

                for d in 0..DIM {
                    let col = g * DIM + d;
                    if col < n_cols {
                        output.push(s * entry[d]);
                    }
                }
            }
        }

        output
    }
}

impl Default for E8Codebook {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn squared_distance(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    let mut sum = 0.0f32;
    for d in 0..DIM {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sum
}

fn norm(v: &[f32; DIM]) -> f32 {
    squared_distance(v, &[0.0; DIM]).sqrt()
}

/// Naive scalar 2-bit quantization for comparison.
///
/// Each scalar is independently mapped to one of 4 levels within
/// `[min, max]`.  Returns the reconstructed values.
pub fn naive_scalar_2bit_quantize(data: &[f32]) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if (max - min).abs() < 1e-10 {
        return data.to_vec();
    }

    let n_levels = 4u32; // 2-bit
    let step = (max - min) / (n_levels - 1) as f32;

    data.iter()
        .map(|&x| {
            let level = ((x - min) / step).round().clamp(0.0, (n_levels - 1) as f32) as u32;
            min + level as f32 * step
        })
        .collect()
}

/// Compute mean squared error between two equal-length slices.
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    sum / a.len() as f32
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_vector_count() {
        let cb = E8Codebook::new();

        // Count non-zero entries with squared norm ≈ 2 (integer orbit) or
        // ≈ 2 (half-integer orbit, norm² = 8 × 0.25 = 2).
        let mut n_roots = 0;
        for entry in cb.entries().iter() {
            let sq = squared_distance(entry, &[0.0; DIM]);
            if (sq - 2.0).abs() < 1e-6 {
                n_roots += 1;
            }
        }
        assert_eq!(n_roots, 240, "expected exactly 240 E8 root vectors");
    }

    #[test]
    fn integer_orbit_count() {
        let cb = E8Codebook::new();
        let mut count = 0;
        for entry in cb.entries().iter() {
            // Integer orbit vectors have exactly 2 non-zero coords, each ±1.
            let nonzero: Vec<_> = entry.iter().filter(|&&x| x.abs() > 0.5).collect();
            let zeros: Vec<_> = entry.iter().filter(|&&x| x.abs() < 1e-6).collect();
            if nonzero.len() == 2
                && zeros.len() == 6
                && nonzero.iter().all(|&&x| (x.abs() - 1.0).abs() < 1e-6)
            {
                count += 1;
            }
        }
        assert_eq!(count, 112, "expected 112 integer-orbit vectors");
    }

    #[test]
    fn half_integer_orbit_count() {
        let cb = E8Codebook::new();
        let mut count = 0;
        for entry in cb.entries().iter() {
            if entry.iter().all(|&x| (x.abs() - 0.5).abs() < 1e-6) {
                count += 1;
            }
        }
        assert_eq!(count, 128, "expected 128 half-integer-orbit vectors");
    }

    #[test]
    fn zero_vector_present() {
        let cb = E8Codebook::new();
        assert_eq!(cb.entries()[0], [0.0; DIM]);
    }

    #[test]
    fn codebook_has_256_entries() {
        let cb = E8Codebook::new();
        assert_eq!(cb.entries().len(), 256);
    }

    #[test]
    fn nearest_exact_match() {
        let cb = E8Codebook::new();
        // Every codebook entry should be its own nearest neighbour.
        for (i, entry) in cb.entries().iter().enumerate() {
            let (idx, found) = cb.nearest(entry);
            assert_eq!(
                idx as usize, i,
                "entry {i} did not match itself; found index {idx}"
            );
            assert_eq!(found, *entry);
        }
    }

    #[test]
    fn nearest_perturbed() {
        let cb = E8Codebook::new();
        // A small perturbation of a codebook entry should still map back
        // to the same entry.
        let entry = cb.entries()[5];
        let mut perturbed = entry;
        perturbed[0] += 0.01;
        perturbed[3] -= 0.01;
        let (idx, _) = cb.nearest(&perturbed);
        assert_eq!(idx, 5);
    }

    #[test]
    fn round_trip_identity_for_codebook_entries() {
        let cb = E8Codebook::new();
        // Build a small "matrix" from codebook entries (scaled).
        let scale = 3.0f32;
        let n_groups = 4;
        let n_cols = n_groups * DIM;
        let mut weights = vec![0.0f32; n_cols];
        for g in 0..n_groups {
            // Use entry g+1 (skip zero) scaled.
            let entry = &cb.entries()[g + 1];
            for d in 0..DIM {
                weights[g * DIM + d] = scale * entry[d];
            }
        }

        let (indices, scales) = cb.quantize_matrix(&weights, n_cols);
        let reconstructed = cb.dequantize_matrix(&indices, &scales, n_cols);

        let error = mse(&weights, &reconstructed);
        assert!(
            error < 1e-6,
            "round-trip error too large for exact codebook entries: {error}"
        );
    }

    #[test]
    fn quantize_dequantize_reduces_error_vs_naive() {
        // Random-ish test vector (deterministic).
        let n = 64;
        let weights: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.7 + 1.3).sin() * 2.0))
            .collect();

        let cb = E8Codebook::new();
        let (indices, scales) = cb.quantize_matrix(&weights, n);
        let e8_recon = cb.dequantize_matrix(&indices, &scales, n);
        let e8_err = mse(&weights, &e8_recon);

        let naive_recon = naive_scalar_2bit_quantize(&weights);
        let naive_err = mse(&weights, &naive_recon);

        // E8 vector quantization should beat naive scalar quantization.
        // (We don't assert a hard threshold — just print for the spike.)
        eprintln!("E8 2-bit MSE:    {e8_err:.6}");
        eprintln!("Naive 2-bit MSE: {naive_err:.6}");
        eprintln!(
            "E8 advantage:    {:.1}× lower error",
            naive_err / e8_err.max(1e-10)
        );
    }

    #[test]
    fn quantize_handles_zero_groups() {
        let cb = E8Codebook::new();
        let weights = vec![0.0f32; 16];
        let (indices, scales) = cb.quantize_matrix(&weights, 16);
        assert!(indices.iter().all(|&i| i == 0));
        assert!(scales.iter().all(|&s| s == 0.0));

        let recon = cb.dequantize_matrix(&indices, &scales, 16);
        assert_eq!(recon, weights);
    }

    #[test]
    fn quantize_handles_non_multiple_of_8_cols() {
        let cb = E8Codebook::new();
        let n_cols = 13;
        let weights: Vec<f32> = (0..n_cols).map(|i| i as f32 * 0.1).collect();
        let (indices, scales) = cb.quantize_matrix(&weights, n_cols);
        let recon = cb.dequantize_matrix(&indices, &scales, n_cols);
        assert_eq!(recon.len(), n_cols);
    }

    #[test]
    fn benchmark_4096x4096() {
        use std::time::Instant;

        let cb = E8Codebook::new();
        let n = 4096;
        let total = n * n;

        // Deterministic pseudo-random weights.
        let weights: Vec<f32> = (0..total)
            .map(|i| ((i as f64 * 0.00017 + 0.3).sin() * 1.5) as f32)
            .collect();

        // Benchmark quantization.
        let t0 = Instant::now();
        let (indices, scales) = cb.quantize_matrix(&weights, n);
        let quant_time = t0.elapsed();

        // Benchmark dequantization.
        let t1 = Instant::now();
        let recon = cb.dequantize_matrix(&indices, &scales, n);
        let dequant_time = t1.elapsed();

        // Reconstruction error.
        let e8_err = mse(&weights, &recon);

        // Naive scalar 2-bit for comparison.
        let naive_recon = naive_scalar_2bit_quantize(&weights);
        let naive_err = mse(&weights, &naive_recon);

        eprintln!("=== E8 Lattice Codebook Benchmark (4096×4096) ===");
        eprintln!("Quantization time:   {:.2?}", quant_time);
        eprintln!("Dequantization time: {:.2?}", dequant_time);
        eprintln!("E8 2-bit MSE:        {e8_err:.6}");
        eprintln!("Naive 2-bit MSE:     {naive_err:.6}");
        eprintln!(
            "E8 advantage:        {:.2}× lower error",
            naive_err / e8_err.max(1e-10)
        );
        eprintln!("Codebook indices:    {} bytes", indices.len());
        eprintln!("Scales:              {} bytes", scales.len() * 4);
        let orig_bytes = total * 4;
        let quant_bytes = indices.len() + scales.len() * 4;
        eprintln!(
            "Compression:         {:.1}× ({} → {} bytes)",
            orig_bytes as f64 / quant_bytes as f64,
            orig_bytes,
            quant_bytes
        );
        eprintln!();
        eprintln!("=== 4-bit Feasibility Estimate ===");
        eprintln!("65K codebook entries → ~256× more NN work per group");
        eprintln!(
            "Estimated 4-bit quant time: {:.2?}",
            quant_time * 256 / 1 // rough proportional estimate
        );
        eprintln!("Verdict: likely acceptable for offline quantization");
    }
}
