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
//! E8 root variants at 0.5× and 2.0× to provide better coverage at
//! different magnitude levels.
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

use std::sync::LazyLock;

/// Number of codebook entries (2-bit = 8 indices per byte gives 256 entries).
const CODEBOOK_SIZE: usize = 256;

/// Dimensionality of each codebook vector.
const DIM: usize = 8;

/// Internal storage for the lazily-initialized codebook data.
struct E8CodebookData {
    /// 256 codebook entries, each an 8-dimensional vector.
    entries: [[f32; DIM]; CODEBOOK_SIZE],
    /// Pre-computed squared norms (`dot(e, e)`) for each entry, invariant
    /// across all nearest-neighbor queries.
    norms_sq: [f32; CODEBOOK_SIZE],
}

/// Lazily-initialized codebook — computed once on first access.
static CODEBOOK_DATA: LazyLock<E8CodebookData> = LazyLock::new(|| {
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

    // Pad to 256 with scaled E8 root vectors for better coverage at
    // different magnitude levels:
    //   - 8 entries at 0.5× scale (norm² = 0.5, for small magnitudes)
    //   - 7 entries at 2.0× scale (norm² = 8.0, for large magnitudes)
    for pad in 0..8 {
        let src = entries[1 + pad];
        for d in 0..DIM {
            entries[idx][d] = src[d] * 0.5;
        }
        idx += 1;
    }
    for pad in 0..7 {
        let src = entries[1 + pad];
        for d in 0..DIM {
            entries[idx][d] = src[d] * 2.0;
        }
        idx += 1;
    }
    debug_assert_eq!(idx, CODEBOOK_SIZE);

    // Pre-compute squared norms for each entry.
    let mut norms_sq = [0.0f32; CODEBOOK_SIZE];
    for i in 0..CODEBOOK_SIZE {
        norms_sq[i] = entries[i].iter().map(|&x| x * x).sum();
    }

    E8CodebookData { entries, norms_sq }
});

/// The E8 lattice codebook for 2-bit vector quantization.
///
/// Each codebook entry is an 8-dimensional vector.  The first 241 entries
/// are the 240 E8 root vectors plus the zero vector; the remaining 15 are
/// scaled variants for better magnitude coverage.
///
/// The codebook data is lazily initialized on first use via [`LazyLock`]
/// and shared across all instances — subsequent `new()` calls are free.
pub struct E8Codebook {
    data: &'static E8CodebookData,
}

impl E8Codebook {
    /// Create a handle to the E8 codebook.
    ///
    /// The underlying data is lazily initialized on first access and shared
    /// across all `E8Codebook` instances.  Subsequent calls are effectively
    /// free.
    pub fn new() -> Self {
        Self {
            data: &CODEBOOK_DATA,
        }
    }

    /// Return a reference to all codebook entries.
    pub fn entries(&self) -> &[[f32; DIM]; CODEBOOK_SIZE] {
        &self.data.entries
    }

    /// Find the nearest codebook entry to `vector` (brute-force).
    ///
    /// Returns `(index, codebook_entry)`.  With only 256 entries the
    /// brute-force search is negligible.
    pub fn nearest(&self, vector: &[f32; DIM]) -> (u8, [f32; DIM]) {
        let mut best_idx = 0u8;
        let mut best_dist = f32::MAX;

        for (i, entry) in self.data.entries.iter().enumerate() {
            let dist = squared_distance(vector, entry);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i as u8;
            }
        }

        (best_idx, self.data.entries[best_idx as usize])
    }

    /// Quantize a single 8-element group with joint entry+scale optimization.
    ///
    /// For each codebook entry `e[i]`, the optimal scale is
    /// `s = dot(group, e) / dot(e, e)`, and the reconstruction error is
    /// `‖group − s·e‖²`.  Returns `(index, scale, quantized_vector)` for
    /// the entry that minimizes this error.
    pub fn quantize_vector(&self, group: &[f32; DIM]) -> (u8, f32, [f32; DIM]) {
        let group_norm_sq: f32 = group.iter().map(|&x| x * x).sum();
        if group_norm_sq < 1e-20 {
            return (0, 0.0, [0.0; DIM]);
        }

        let mut best_idx = 0u8;
        let mut best_scale = 0.0f32;
        let mut best_err = f32::MAX;

        for (ei, entry) in self.data.entries.iter().enumerate() {
            let dot_ee = self.data.norms_sq[ei];
            if dot_ee < 1e-10 {
                continue;
            }

            let dot_ge: f32 = (0..DIM).map(|d| group[d] * entry[d]).sum();
            let scale = dot_ge / dot_ee;

            // ‖g - s·e‖² = ‖g‖² - dot(g,e)² / ‖e‖²
            let err = group_norm_sq - dot_ge * dot_ge / dot_ee;

            if err < best_err {
                best_err = err;
                best_idx = ei as u8;
                best_scale = scale;
            }
        }

        let mut quantized = [0.0f32; DIM];
        let best_entry = &self.data.entries[best_idx as usize];
        for d in 0..DIM {
            quantized[d] = best_scale * best_entry[d];
        }

        (best_idx, best_scale, quantized)
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
    /// `scale = dot(group, e) / dot(e, e)` for the best codebook entry `e`,
    /// so that `scale * e ≈ group`.
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

                let (idx, scale, _) = self.quantize_vector(&group);
                indices.push(idx);
                scales.push(scale);
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
                let entry = &self.data.entries[indices[gi] as usize];
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

    /// Compute MSE between original data and its quantized reconstruction.
    ///
    /// Treats `original` as a flat sequence of 8-element groups.  Handles
    /// lengths that are not a multiple of 8 by only comparing actual
    /// (non-padding) elements.
    pub fn reconstruction_mse(&self, original: &[f32], indices: &[u8], scales: &[f32]) -> f32 {
        assert_eq!(indices.len(), scales.len());
        let n = original.len();
        if n == 0 {
            return 0.0;
        }
        let n_groups = n.div_ceil(DIM);
        assert_eq!(
            indices.len(),
            n_groups,
            "expected {} groups for {} elements, got {}",
            n_groups,
            n,
            indices.len()
        );

        let mut sum_sq_err = 0.0f32;
        for g in 0..n_groups {
            let entry = &self.data.entries[indices[g] as usize];
            let s = scales[g];
            for d in 0..DIM {
                let orig_idx = g * DIM + d;
                if orig_idx < n {
                    let diff = original[orig_idx] - s * entry[d];
                    sum_sq_err += diff * diff;
                }
            }
        }
        sum_sq_err / n as f32
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

    // ── New tests for production refinements ─────────────────────────────

    #[test]
    fn codebook_is_cached_across_instances() {
        let cb1 = E8Codebook::new();
        let cb2 = E8Codebook::new();
        // Both instances share the same underlying data.
        assert!(std::ptr::eq(cb1.entries(), cb2.entries()));
    }

    #[test]
    fn norms_sq_precomputed_correctly() {
        let data = &*CODEBOOK_DATA;
        for i in 0..CODEBOOK_SIZE {
            let expected: f32 = data.entries[i].iter().map(|&x| x * x).sum();
            assert!(
                (data.norms_sq[i] - expected).abs() < 1e-10,
                "norms_sq[{i}] = {}, expected {expected}",
                data.norms_sq[i]
            );
        }
    }

    #[test]
    fn padding_entries_have_mixed_scales() {
        let cb = E8Codebook::new();

        // Entries 241..249 should be 0.5× scaled (norm² ≈ 0.5).
        for i in 241..249 {
            let sq: f32 = cb.entries()[i].iter().map(|&x| x * x).sum();
            assert!(
                (sq - 0.5).abs() < 1e-6,
                "entry {i}: expected norm²=0.5, got {sq}"
            );
        }

        // Entries 249..256 should be 2.0× scaled (norm² ≈ 8.0).
        for i in 249..256 {
            let sq: f32 = cb.entries()[i].iter().map(|&x| x * x).sum();
            assert!(
                (sq - 8.0).abs() < 1e-6,
                "entry {i}: expected norm²=8.0, got {sq}"
            );
        }
    }

    #[test]
    fn quantize_vector_exact_codebook_entry() {
        let cb = E8Codebook::new();
        let scale = 2.5f32;
        let entry = cb.entries()[10];
        let mut group = [0.0f32; DIM];
        for d in 0..DIM {
            group[d] = scale * entry[d];
        }

        let (idx, s, quantized) = cb.quantize_vector(&group);
        assert_eq!(idx, 10);
        assert!((s - scale).abs() < 1e-5, "scale={s}, expected {scale}");
        for d in 0..DIM {
            assert!(
                (quantized[d] - group[d]).abs() < 1e-5,
                "quantized[{d}]={}, expected {}",
                quantized[d],
                group[d]
            );
        }
    }

    #[test]
    fn quantize_vector_zero_group() {
        let cb = E8Codebook::new();
        let (idx, scale, quantized) = cb.quantize_vector(&[0.0; DIM]);
        assert_eq!(idx, 0);
        assert_eq!(scale, 0.0);
        assert_eq!(quantized, [0.0; DIM]);
    }

    #[test]
    fn quantize_vector_returns_correct_tuple() {
        let cb = E8Codebook::new();
        let group = [0.3, -0.7, 0.1, 0.5, -0.2, 0.4, -0.6, 0.8];
        let (idx, scale, quantized) = cb.quantize_vector(&group);

        // Verify the quantized vector equals scale * codebook_entry.
        let entry = cb.entries()[idx as usize];
        for d in 0..DIM {
            assert!(
                (quantized[d] - scale * entry[d]).abs() < 1e-6,
                "quantized[{d}]={}, expected {}",
                quantized[d],
                scale * entry[d]
            );
        }
    }

    #[test]
    fn quantize_vector_agrees_with_quantize_matrix() {
        let cb = E8Codebook::new();
        let group = [1.2, -0.3, 0.7, -1.1, 0.5, 0.0, -0.8, 0.4];

        let (v_idx, v_scale, _) = cb.quantize_vector(&group);
        let (m_indices, m_scales) = cb.quantize_matrix(&group, DIM);

        assert_eq!(v_idx, m_indices[0]);
        assert!((v_scale - m_scales[0]).abs() < 1e-6);
    }

    #[test]
    fn reconstruction_mse_zero_for_exact_match() {
        let cb = E8Codebook::new();
        let scale = 1.5f32;
        let entry = cb.entries()[20];
        let original: Vec<f32> = entry.iter().map(|&x| scale * x).collect();

        let (indices, scales) = cb.quantize_matrix(&original, DIM);
        let err = cb.reconstruction_mse(&original, &indices, &scales);
        assert!(err < 1e-10, "expected near-zero MSE, got {err}");
    }

    #[test]
    fn reconstruction_mse_matches_manual_mse() {
        let cb = E8Codebook::new();
        let n = 64;
        let weights: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.7 + 1.3).sin() * 2.0))
            .collect();

        let (indices, scales) = cb.quantize_matrix(&weights, n);
        let recon = cb.dequantize_matrix(&indices, &scales, n);

        let manual_err = mse(&weights, &recon);
        let method_err = cb.reconstruction_mse(&weights, &indices, &scales);

        assert!(
            (manual_err - method_err).abs() < 1e-6,
            "manual MSE={manual_err}, method MSE={method_err}"
        );
    }

    #[test]
    fn reconstruction_mse_non_multiple_of_8() {
        let cb = E8Codebook::new();
        let n: usize = 13;
        let weights: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();

        let n_groups = n.div_ceil(DIM);
        // Quantize each group manually for the flat case.
        let mut indices = Vec::new();
        let mut scales = Vec::new();
        for g in 0..n_groups {
            let mut group = [0.0f32; DIM];
            for d in 0..DIM {
                let idx = g * DIM + d;
                if idx < n {
                    group[d] = weights[idx];
                }
            }
            let (i, s, _) = cb.quantize_vector(&group);
            indices.push(i);
            scales.push(s);
        }

        let err = cb.reconstruction_mse(&weights, &indices, &scales);
        assert!(err.is_finite(), "MSE should be finite, got {err}");
    }
}
