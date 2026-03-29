//! Randomized Hadamard rotation utilities for PolarQuant.
//!
//! Provides seeded, deterministic orthogonal rotations based on a
//! Walsh–Hadamard transform preceded by a random diagonal sign matrix.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Applies a seeded randomized Hadamard transform to a matrix in-place.
///
/// `data` is a row-major [rows, cols] matrix stored as `&mut [f32]`.
/// Each row is rotated independently. `cols` must be a power of two;
/// pad with zeros if necessary.
pub fn rotate_rows_hadamard(data: &mut [f32], rows: usize, cols: usize, seed: u64) {
    assert_eq!(data.len(), rows * cols, "data length must equal rows * cols");
    assert!(cols.is_power_of_two(), "cols must be a power of two");

    let signs = generate_signs(cols, seed);
    let scale = 1.0 / (cols as f32).sqrt();

    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];
        // Apply diagonal sign matrix D
        for (x, &s) in row.iter_mut().zip(signs.iter()) {
            *x *= s;
        }
        // In-place Walsh–Hadamard butterfly
        hadamard_butterfly(row);
        // Apply D again so that R = (1/√N)·D·H·D is symmetric and self-inverse
        for (x, &s) in row.iter_mut().zip(signs.iter()) {
            *x *= s;
        }
        // Normalize so the transform is orthogonal
        for x in row.iter_mut() {
            *x *= scale;
        }
    }
}

/// Applies the inverse (transpose) of the same rotation.
///
/// Since the normalized Hadamard matrix H is symmetric and orthogonal,
/// and the diagonal sign matrix D is self-inverse, the inverse is
/// identical to the forward transform.
pub fn unrotate_rows_hadamard(data: &mut [f32], rows: usize, cols: usize, seed: u64) {
    rotate_rows_hadamard(data, rows, cols, seed);
}

/// Pads `cols` up to the next power of two. Returns `(padded_data, padded_cols)`.
///
/// Each row is zero-padded on the right. If `cols` is already a power of
/// two the data is returned unchanged (copied into a new `Vec`).
pub fn pad_to_power_of_two(data: &[f32], rows: usize, cols: usize) -> (Vec<f32>, usize) {
    assert_eq!(data.len(), rows * cols, "data length must equal rows * cols");

    if cols == 0 {
        return (data.to_vec(), 0);
    }

    let padded_cols = cols.next_power_of_two();
    if padded_cols == cols {
        return (data.to_vec(), cols);
    }

    let mut out = vec![0.0f32; rows * padded_cols];
    for r in 0..rows {
        let src = &data[r * cols..(r + 1) * cols];
        let dst = &mut out[r * padded_cols..r * padded_cols + cols];
        dst.copy_from_slice(src);
    }
    (out, padded_cols)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Generate a vector of ±1 signs from a seeded PRNG.
fn generate_signs(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 }).collect()
}

/// Standard in-place Walsh–Hadamard butterfly (unnormalized).
///
/// Runs in O(N log N) for a row of length N.
fn hadamard_butterfly(row: &mut [f32]) {
    let n = row.len();
    let mut half = 1;
    while half < n {
        for i in (0..n).step_by(half * 2) {
            for j in i..i + half {
                let a = row[j];
                let b = row[j + half];
                row[j] = a + b;
                row[j + half] = a - b;
            }
        }
        half *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    /// R · R^T ≈ I (the rotation matrix is orthogonal).
    #[test]
    fn hadamard_is_orthogonal() {
        let n = 8;
        let seed = 42;

        // Build the rotation matrix by rotating each standard basis vector.
        let mut r_matrix = vec![0.0f32; n * n];
        for i in 0..n {
            let mut e = vec![0.0f32; n];
            e[i] = 1.0;
            rotate_rows_hadamard(&mut e, 1, n, seed);
            for j in 0..n {
                r_matrix[i * n + j] = e[j];
            }
        }

        // Compute R · R^T and compare with I.
        for i in 0..n {
            for j in 0..n {
                let dot: f32 = (0..n).map(|k| r_matrix[i * n + k] * r_matrix[j * n + k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < TOL,
                    "R·R^T[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    /// rotate then unrotate ≈ identity.
    #[test]
    fn hadamard_is_self_inverse() {
        let rows = 3;
        let cols = 16;
        let seed = 123;

        let original: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
        let mut data = original.clone();

        rotate_rows_hadamard(&mut data, rows, cols, seed);
        unrotate_rows_hadamard(&mut data, rows, cols, seed);

        for (a, b) in data.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < TOL,
                "mismatch: got {a}, expected {b}"
            );
        }
    }

    /// ‖Rx‖ ≈ ‖x‖ (orthogonal transforms preserve norms).
    #[test]
    fn hadamard_preserves_norm() {
        let cols = 32;
        let seed = 7;

        let mut data: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0).sin()).collect();
        let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        rotate_rows_hadamard(&mut data, 1, cols, seed);
        let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (norm_before - norm_after).abs() < TOL,
            "norm changed: {norm_before} -> {norm_after}"
        );
    }

    /// Padding works for various input sizes.
    #[test]
    fn pad_to_power_of_two_correct() {
        // Already a power of two → unchanged.
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let (padded, pc) = pad_to_power_of_two(&data, 1, 4);
        assert_eq!(pc, 4);
        assert_eq!(padded, data);

        // 3 → 4
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let (padded, pc) = pad_to_power_of_two(&data, 1, 3);
        assert_eq!(pc, 4);
        assert_eq!(padded, vec![1.0, 2.0, 3.0, 0.0]);

        // Multi-row: 2 rows × 5 cols → 2 rows × 8 cols
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (padded, pc) = pad_to_power_of_two(&data, 2, 5);
        assert_eq!(pc, 8);
        assert_eq!(padded.len(), 16);
        // First row padded
        assert_eq!(&padded[0..5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(&padded[5..8], &[0.0, 0.0, 0.0]);
        // Second row padded
        assert_eq!(&padded[8..13], &[6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_eq!(&padded[13..16], &[0.0, 0.0, 0.0]);

        // Single element → 1
        let data: Vec<f32> = vec![42.0];
        let (padded, pc) = pad_to_power_of_two(&data, 1, 1);
        assert_eq!(pc, 1);
        assert_eq!(padded, vec![42.0]);
    }

    /// Same seed produces identical output.
    #[test]
    fn rotate_deterministic_with_seed() {
        let cols = 16;
        let seed = 999;

        let original: Vec<f32> = (0..cols).map(|i| i as f32).collect();

        let mut a = original.clone();
        rotate_rows_hadamard(&mut a, 1, cols, seed);

        let mut b = original.clone();
        rotate_rows_hadamard(&mut b, 1, cols, seed);

        assert_eq!(a, b);
    }

    /// Different seeds produce different output.
    #[test]
    fn different_seeds_differ() {
        let cols = 16;

        let original: Vec<f32> = (0..cols).map(|i| i as f32 + 1.0).collect();

        let mut a = original.clone();
        rotate_rows_hadamard(&mut a, 1, cols, 1);

        let mut b = original.clone();
        rotate_rows_hadamard(&mut b, 1, cols, 2);

        assert_ne!(a, b);
    }
}
