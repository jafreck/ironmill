//! Hessian computation utilities for GPTQ.

#[cfg(test)]
mod tests {
    #[test]
    fn faer_cholesky_smoke() {
        // Create a small positive definite matrix and verify Cholesky works
        let mat = faer::mat![[4.0_f64, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0],];
        let cholesky = mat.cholesky(faer::Side::Lower).unwrap();
        let l = cholesky.compute_l();
        // Verify L * L^T ≈ original matrix
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
}
