//! Hessian computation utilities for GPTQ.
//!
//! Provides functions for finalising the accumulated X^T X Hessian,
//! Cholesky decomposition, and row-wise or full inverse computation.
//! All public items are gated behind the `gptq` feature flag.

// faer::prelude::* is used in tests for the `.cholesky()` convenience method.
#[cfg(test)]
use faer::prelude::*;

/// Finalize the Hessian from accumulated X^T X.
///
/// Applies scaling and diagonal dampening so the result is suitable for
/// Cholesky decomposition:
///
///   H = (2 / sample_count) · X^T X  +  dampening · mean(diag(H)) · I
///
/// Operates **in-place** on the row-major `xtx` buffer.
///
/// # Panics
///
/// Panics if `sample_count` is zero or if `xtx.len() != n_features²`.
pub fn finalize_hessian(xtx: &mut [f32], n_features: usize, sample_count: usize, dampening: f64) {
    assert!(
        sample_count > 0,
        "finalize_hessian: sample_count must be > 0"
    );
    assert_eq!(
        xtx.len(),
        n_features * n_features,
        "finalize_hessian: xtx length {} != n_features² {}",
        xtx.len(),
        n_features * n_features,
    );

    let n = n_features;
    let scale = 2.0 / sample_count as f64;

    // H = (2 / sample_count) · X^T X
    for val in xtx.iter_mut() {
        *val = (*val as f64 * scale) as f32;
    }

    // mean(diag(H))
    let diag_mean: f64 = (0..n).map(|i| xtx[i * n + i] as f64).sum::<f64>() / n as f64;

    // H[i,i] += dampening · mean(diag(H))
    let damp_val = (dampening * diag_mean) as f32;
    for i in 0..n {
        xtx[i * n + i] += damp_val;
    }
}

/// Cholesky-decompose a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor **L** (row-major, flat) such that
/// H = L · Lᵀ.  Internally uses `faer` in f64 for numerical stability.
///
/// # Errors
///
/// Returns `Err` if the matrix is not positive-definite.
pub fn cholesky_decompose(h: &[f32], n: usize) -> Result<Vec<f32>, String> {
    assert_eq!(
        h.len(),
        n * n,
        "cholesky_decompose: h length {} != n² {}",
        h.len(),
        n * n,
    );

    let mat = faer::Mat::<f64>::from_fn(n, n, |i, j| h[i * n + j] as f64);

    let cholesky = mat.cholesky(faer::Side::Lower).map_err(|e| {
        format!(
            "Cholesky decomposition failed: non-positive-definite minor at index {}",
            e.non_positive_definite_minor
        )
    })?;

    let l = cholesky.compute_l();

    let mut result = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = l.read(i, j) as f32;
        }
    }
    Ok(result)
}

/// Compute a single row of the inverse Hessian from the Cholesky factor.
///
/// GPTQ only needs one row at a time during the quantization loop.
/// Since H⁻¹ is symmetric, the *row*-th row equals the *row*-th column,
/// which is H⁻¹ · e_row.  We solve via forward/backward substitution:
///
///   L · y = e_row   (forward substitution)
///   Lᵀ · x = y      (backward substitution)
///
/// The result `x` is H⁻¹[row, :].
///
/// # Panics
///
/// Panics if `l.len() != n²` or `row >= n`.
pub fn cholesky_inverse_row(l: &[f32], n: usize, row: usize) -> Vec<f32> {
    assert_eq!(
        l.len(),
        n * n,
        "cholesky_inverse_row: l length {} != n² {}",
        l.len(),
        n * n,
    );
    assert!(row < n, "cholesky_inverse_row: row {row} >= n {n}");

    // Work in f64 for stability.
    let l64: Vec<f64> = l.iter().map(|&x| x as f64).collect();

    // e_row: unit vector with 1.0 at position `row`.
    let mut y = vec![0.0f64; n];
    y[row] = 1.0;

    // Forward substitution: L · y = e_row.
    // L is lower-triangular, row-major: L[i,j] = l64[i * n + j].
    for i in 0..n {
        let mut sum = y[i];
        for j in 0..i {
            sum -= l64[i * n + j] * y[j];
        }
        y[i] = sum / l64[i * n + i];
    }

    // Backward substitution: Lᵀ · x = y.
    // Lᵀ[i,j] = L[j,i] = l64[j * n + i].
    let mut x = y;
    for i in (0..n).rev() {
        let mut sum = x[i];
        for j in (i + 1)..n {
            sum -= l64[j * n + i] * x[j];
        }
        x[i] = sum / l64[i * n + i];
    }

    x.iter().map(|&v| v as f32).collect()
}

/// Compute the full inverse Hessian from the Cholesky factor.
///
/// Returns H⁻¹ as a row-major flat slice.  Suitable for small matrices
/// or testing; for large matrices prefer [`cholesky_inverse_row`].
///
/// # Panics
///
/// Panics if `l.len() != n²`.
pub fn cholesky_inverse(l: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(
        l.len(),
        n * n,
        "cholesky_inverse: l length {} != n² {}",
        l.len(),
        n * n,
    );

    let mut result = vec![0.0f32; n * n];
    for row in 0..n {
        let inv_row = cholesky_inverse_row(l, n, row);
        result[row * n..row * n + n].copy_from_slice(&inv_row);
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    /// Helper: check two flat matrices are approximately equal.
    fn assert_mat_approx(a: &[f32], b: &[f32], n: usize, tol: f32, label: &str) {
        assert_eq!(a.len(), b.len());
        for i in 0..n {
            for j in 0..n {
                let diff = (a[i * n + j] - b[i * n + j]).abs();
                assert!(
                    diff < tol,
                    "{label}: mismatch at ({i},{j}): {} vs {} (diff {diff})",
                    a[i * n + j],
                    b[i * n + j],
                );
            }
        }
    }

    /// Multiply two n×n row-major matrices.
    fn mat_mul(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; n * n];
        for i in 0..n {
            for k in 0..n {
                let a_ik = a[i * n + k];
                for j in 0..n {
                    c[i * n + j] += a_ik * b[k * n + j];
                }
            }
        }
        c
    }

    /// Transpose an n×n row-major matrix.
    fn transpose(m: &[f32], n: usize) -> Vec<f32> {
        let mut t = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                t[j * n + i] = m[i * n + j];
            }
        }
        t
    }

    /// Identity matrix.
    fn eye(n: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    // -- faer smoke test (preserved from skeleton) ----------------------------

    #[test]
    fn faer_cholesky_smoke() {
        let mat = faer::mat![[4.0_f64, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0],];
        let cholesky = mat.cholesky(faer::Side::Lower).unwrap();
        let l = cholesky.compute_l();
        let reconstructed = &l * l.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (reconstructed.read(i, j) - mat.read(i, j)).abs() < 1e-10,
                    "Cholesky reconstruction failed at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    // -- finalize_hessian tests -----------------------------------------------

    #[test]
    fn finalize_hessian_scaling_and_dampening() {
        // X = [[2, 3]]  →  X^T X = [[4, 6], [6, 9]]
        // sample_count = 1, dampening = 0.01
        //
        // H = 2/1 · [[4,6],[6,9]] = [[8,12],[12,18]]
        // mean(diag) = (8 + 18) / 2 = 13
        // damp = 0.01 * 13 = 0.13
        // H_final = [[8.13, 12], [12, 18.13]]
        let mut xtx = vec![4.0, 6.0, 6.0, 9.0];
        finalize_hessian(&mut xtx, 2, 1, 0.01);

        assert!((xtx[0] - 8.13).abs() < TOL, "H[0,0] = {}", xtx[0]);
        assert!((xtx[1] - 12.0).abs() < TOL, "H[0,1] = {}", xtx[1]);
        assert!((xtx[2] - 12.0).abs() < TOL, "H[1,0] = {}", xtx[2]);
        assert!((xtx[3] - 18.13).abs() < TOL, "H[1,1] = {}", xtx[3]);
    }

    #[test]
    fn finalize_hessian_sample_count_scaling() {
        // X^T X = [[10, 0], [0, 10]], sample_count = 5, dampening = 0
        // H = 2/5 · [[10,0],[0,10]] = [[4,0],[0,4]]
        let mut xtx = vec![10.0, 0.0, 0.0, 10.0];
        finalize_hessian(&mut xtx, 2, 5, 0.0);

        assert!((xtx[0] - 4.0).abs() < TOL);
        assert!((xtx[1]).abs() < TOL);
        assert!((xtx[2]).abs() < TOL);
        assert!((xtx[3] - 4.0).abs() < TOL);
    }

    #[test]
    fn finalize_hessian_zero_dampening() {
        let mut xtx = vec![4.0, 2.0, 2.0, 5.0];
        finalize_hessian(&mut xtx, 2, 2, 0.0);
        // H = 2/2 * xtx = xtx (no change from dampening)
        assert!((xtx[0] - 4.0).abs() < TOL);
        assert!((xtx[3] - 5.0).abs() < TOL);
    }

    #[test]
    #[should_panic(expected = "sample_count must be > 0")]
    fn finalize_hessian_zero_samples() {
        let mut xtx = vec![1.0; 4];
        finalize_hessian(&mut xtx, 2, 0, 0.01);
    }

    // -- cholesky_decompose tests ---------------------------------------------

    #[test]
    fn cholesky_decompose_3x3_spd() {
        // Known SPD matrix.
        #[rustfmt::skip]
        let h = vec![
            4.0, 2.0, 1.0,
            2.0, 5.0, 3.0,
            1.0, 3.0, 6.0,
        ];
        let l = cholesky_decompose(&h, 3).unwrap();

        // Verify L * L^T ≈ H.
        let lt = transpose(&l, 3);
        let reconstructed = mat_mul(&l, &lt, 3);
        assert_mat_approx(&reconstructed, &h, 3, TOL, "L*L^T vs H");

        // L should be lower-triangular: upper entries are zero.
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!(
                    l[i * 3 + j].abs() < TOL,
                    "L[{i},{j}] = {} should be zero",
                    l[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn cholesky_decompose_singular_fails() {
        // Singular matrix: rank 1.
        #[rustfmt::skip]
        let h = vec![
            1.0, 2.0,
            2.0, 4.0,
        ];
        let result = cholesky_decompose(&h, 2);
        assert!(result.is_err(), "singular matrix should fail Cholesky");
    }

    // -- cholesky_inverse_row tests -------------------------------------------

    #[test]
    fn cholesky_inverse_row_matches_full_inverse() {
        #[rustfmt::skip]
        let h = vec![
            4.0, 2.0, 1.0,
            2.0, 5.0, 3.0,
            1.0, 3.0, 6.0,
        ];
        let l = cholesky_decompose(&h, 3).unwrap();
        let full_inv = cholesky_inverse(&l, 3);

        for row in 0..3 {
            let single_row = cholesky_inverse_row(&l, 3, row);
            for j in 0..3 {
                let diff = (single_row[j] - full_inv[row * 3 + j]).abs();
                assert!(
                    diff < TOL,
                    "row {row} col {j}: single={} full={}",
                    single_row[j],
                    full_inv[row * 3 + j],
                );
            }
        }
    }

    // -- cholesky_inverse tests -----------------------------------------------

    #[test]
    fn cholesky_inverse_4x4_identity_product() {
        // H · H⁻¹ ≈ I for a 4×4 SPD matrix.
        #[rustfmt::skip]
        let h = vec![
            10.0,  2.0,  1.0,  0.5,
             2.0,  8.0,  1.5,  1.0,
             1.0,  1.5,  7.0,  2.0,
             0.5,  1.0,  2.0,  6.0,
        ];
        let l = cholesky_decompose(&h, 4).unwrap();
        let h_inv = cholesky_inverse(&l, 4);
        let product = mat_mul(&h, &h_inv, 4);
        let id = eye(4);
        assert_mat_approx(&product, &id, 4, 1e-3, "H * H^{-1} vs I (4×4)");
    }

    #[test]
    fn cholesky_inverse_8x8_identity_product() {
        // Diagonally dominant 8×8 → guaranteed SPD.
        let n = 8;
        let mut h = vec![0.0f32; n * n];
        for i in 0..n {
            h[i * n + i] = 10.0 + i as f32;
            for j in 0..n {
                if i != j {
                    h[i * n + j] = 1.0 / (1.0 + (i as f32 - j as f32).abs());
                }
            }
        }

        let l = cholesky_decompose(&h, n).unwrap();
        let h_inv = cholesky_inverse(&l, n);
        let product = mat_mul(&h, &h_inv, n);
        let id = eye(n);
        assert_mat_approx(&product, &id, n, 1e-3, "H * H^{-1} vs I (8×8)");
    }

    #[test]
    fn cholesky_inverse_symmetry() {
        #[rustfmt::skip]
        let h = vec![
            4.0, 2.0, 1.0,
            2.0, 5.0, 3.0,
            1.0, 3.0, 6.0,
        ];
        let l = cholesky_decompose(&h, 3).unwrap();
        let h_inv = cholesky_inverse(&l, 3);

        // H⁻¹ should be symmetric.
        for i in 0..3 {
            for j in 0..3 {
                let diff = (h_inv[i * 3 + j] - h_inv[j * 3 + i]).abs();
                assert!(
                    diff < TOL,
                    "H^{{-1}} not symmetric at ({i},{j}): {} vs {}",
                    h_inv[i * 3 + j],
                    h_inv[j * 3 + i],
                );
            }
        }
    }

    // -- dampened nearly-singular matrix succeeds ------------------------------

    #[test]
    fn dampened_singular_matrix_succeeds() {
        // Nearly singular: rank-1 + tiny diagonal → fails without dampening.
        let n = 3;
        // v = [1, 2, 3], H = v v^T  (rank 1, singular)
        #[rustfmt::skip]
        let mut xtx = vec![
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0,
        ];

        // Without dampening, Cholesky should fail.
        let undampened = xtx.clone();
        assert!(
            cholesky_decompose(&undampened, n).is_err(),
            "rank-1 matrix should fail Cholesky without dampening"
        );

        // With dampening applied through finalize_hessian, it should succeed.
        finalize_hessian(&mut xtx, n, 1, 0.01);
        let l = cholesky_decompose(&xtx, n).expect("dampened matrix should be positive definite");

        // Verify L * L^T ≈ dampened H.
        let lt = transpose(&l, n);
        let reconstructed = mat_mul(&l, &lt, n);
        assert_mat_approx(&reconstructed, &xtx, n, TOL, "dampened L*L^T vs H");

        // And the inverse should be computable.
        let h_inv = cholesky_inverse(&l, n);
        let product = mat_mul(&xtx, &h_inv, n);
        let id = eye(n);
        assert_mat_approx(&product, &id, n, 1e-2, "dampened H * H^{-1} vs I");
    }

    // -- numpy/scipy reference test -------------------------------------------

    #[test]
    fn cholesky_inverse_matches_numpy_reference() {
        // Reference computed with numpy:
        //   H = [[4, 2], [2, 5]]
        //   np.linalg.inv(H) = [[ 0.3125, -0.125], [-0.125, 0.25]]
        #[rustfmt::skip]
        let h = vec![4.0, 2.0, 2.0, 5.0];
        let l = cholesky_decompose(&h, 2).unwrap();
        let h_inv = cholesky_inverse(&l, 2);

        assert!((h_inv[0] - 0.3125).abs() < TOL, "H^-1[0,0]={}", h_inv[0]);
        assert!((h_inv[1] - (-0.125)).abs() < TOL, "H^-1[0,1]={}", h_inv[1]);
        assert!((h_inv[2] - (-0.125)).abs() < TOL, "H^-1[1,0]={}", h_inv[2]);
        assert!((h_inv[3] - 0.25).abs() < TOL, "H^-1[1,1]={}", h_inv[3]);
    }
}
