//! Cayley parameterization for orthogonal rotation matrices.
//!
//! The Cayley transform maps a skew-symmetric matrix `A` to an orthogonal
//! matrix `R = (I - A)(I + A)^{-1}`.  Because the domain is unconstrained
//! (any real vector encodes a valid skew-symmetric matrix), derivative-free
//! optimizers can freely explore the space while the output is guaranteed
//! orthogonal.

use faer::Mat;
use faer::linalg::solvers::Solve;

/// A rotation matrix parameterized via the Cayley transform.
///
/// Given a skew-symmetric matrix `A` (where `A^T = -A`), the Cayley transform
/// produces `R = (I - A)(I + A)^{-1}`, which is always orthogonal regardless
/// of the values in `A`.
///
/// Only the upper-triangular entries of `A` are stored (the lower triangle is
/// determined by skew-symmetry and the diagonal is zero).
pub struct CayleyRotation {
    /// Skew-symmetric parameters (upper triangle, `n*(n-1)/2` values).
    params: Vec<f64>,
    /// Matrix dimension.
    n: usize,
}

impl CayleyRotation {
    /// Create an identity rotation (`A = 0` → `R = I`).
    pub fn identity(n: usize) -> Self {
        let num_params = n * (n - 1) / 2;
        Self {
            params: vec![0.0; num_params],
            n,
        }
    }

    /// Create from a Hadamard-like initialization.
    ///
    /// Uses a deterministic seeded RNG to populate skew-symmetric parameters
    /// with small random values, producing a rotation close to identity but
    /// with enough variation to break symmetry during optimization.
    pub fn from_hadamard(n: usize, seed: u64) -> Self {
        let num_params = n * (n - 1) / 2;
        let mut params = Vec::with_capacity(num_params);

        // Simple deterministic PRNG (xorshift64) for reproducibility.
        let mut state = seed.wrapping_add(1);
        for _ in 0..num_params {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to small values in [-0.1, 0.1].
            let val = (state as f64 / u64::MAX as f64) * 0.2 - 0.1;
            params.push(val);
        }

        Self { params, n }
    }

    /// Build the full skew-symmetric matrix `A` from the stored parameters.
    fn build_skew_symmetric(&self) -> Mat<f64> {
        let n = self.n;
        let mut a = Mat::<f64>::zeros(n, n);
        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                a[(i, j)] = self.params[idx];
                a[(j, i)] = -self.params[idx];
                idx += 1;
            }
        }
        a
    }

    /// Get the current rotation matrix as a flat row-major `f32` array.
    ///
    /// Computes `R = (I - A)(I + A)^{-1}` by solving the linear system
    /// `(I + A) * R = (I - A)` via LU factorization.
    pub fn to_matrix(&self) -> Vec<f32> {
        let n = self.n;
        let a = self.build_skew_symmetric();
        let eye = Mat::<f64>::identity(n, n);

        // I + A
        let i_plus_a = Mat::from_fn(n, n, |i, j| eye[(i, j)] + a[(i, j)]);
        // I - A
        let i_minus_a = Mat::from_fn(n, n, |i, j| eye[(i, j)] - a[(i, j)]);

        // Solve (I + A) * R = (I - A)  →  R = (I + A)^{-1} * (I - A)
        let lu = i_plus_a.partial_piv_lu();
        let r = lu.solve(&i_minus_a);

        // Flatten to row-major f32.
        let mut out = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                out.push(r[(i, j)] as f32);
            }
        }
        out
    }

    /// Verify orthogonality: `||R * R^T - I||_F`.
    pub fn orthogonality_error(&self) -> f64 {
        let n = self.n;
        let flat = self.to_matrix();
        let r = Mat::from_fn(n, n, |i, j| flat[i * n + j] as f64);
        let rt = r.transpose();
        let rrt = &r * &rt;
        let eye = Mat::<f64>::identity(n, n);

        let mut frobenius_sq = 0.0;
        for i in 0..n {
            for j in 0..n {
                let diff = rrt[(i, j)] - eye[(i, j)];
                frobenius_sq += diff * diff;
            }
        }
        frobenius_sq.sqrt()
    }

    /// Number of free parameters.
    pub fn n_params(&self) -> usize {
        self.n * (self.n - 1) / 2
    }

    /// Direct access to the underlying parameters.
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Build a `CayleyRotation` from raw parameter values.
    pub fn from_params(n: usize, params: Vec<f64>) -> Self {
        assert_eq!(
            params.len(),
            n * (n - 1) / 2,
            "expected {} params for n={n}, got {}",
            n * (n - 1) / 2,
            params.len()
        );
        Self { params, n }
    }
}

/// Derivative-free optimizer for Cayley rotations.
///
/// Uses random search with an adaptive perturbation radius (simplified
/// CMA-ES).  At each iteration the optimizer generates `population_size`
/// random perturbations of the current best parameters, evaluates the loss
/// for each, and keeps the best candidate.  The perturbation radius (`sigma`)
/// is increased when improvements are found and decreased otherwise, loosely
/// following the 1/5-th success rule.
#[non_exhaustive]
pub struct CayleyOptimizer {
    /// Maximum number of optimization iterations (default 100).
    pub max_iterations: usize,
    /// Number of candidates evaluated per iteration (default 20).
    pub population_size: usize,
    /// Initial perturbation radius (default 0.1).
    pub initial_sigma: f64,
    /// Convergence threshold — stop if loss drops below this (default 1e-10).
    pub tolerance: f64,
}

impl Default for CayleyOptimizer {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            population_size: 20,
            initial_sigma: 0.1,
            tolerance: 1e-10,
        }
    }
}

impl CayleyOptimizer {
    /// Optimize a rotation to minimize the given loss function.
    ///
    /// `loss_fn` receives a rotation matrix (flat row-major `f32`, length
    /// `n*n`) and returns a scalar loss.  It will be called up to
    /// `population_size * max_iterations` times.
    pub fn optimize(
        &self,
        initial: &CayleyRotation,
        loss_fn: &mut dyn FnMut(&[f32]) -> f64,
    ) -> CayleyRotation {
        let n = initial.n;
        let _n_params = initial.n_params();

        let mut best_params = initial.params.clone();
        let mut best_loss =
            loss_fn(&CayleyRotation::from_params(n, best_params.clone()).to_matrix());

        let mut sigma = self.initial_sigma;
        // Simple deterministic PRNG (xorshift64).
        let mut rng_state: u64 = 42;

        for _iter in 0..self.max_iterations {
            if best_loss < self.tolerance {
                break;
            }

            let mut improved = false;

            for _pop in 0..self.population_size {
                // Generate a random perturbation.
                let mut candidate = best_params.clone();
                for p in candidate.iter_mut() {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    // Map to approximately normal via Box-Muller-like transform
                    // (using uniform as a simpler approximation).
                    let u = (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
                    *p += sigma * u;
                }

                let rotation = CayleyRotation::from_params(n, candidate.clone());
                let loss = loss_fn(&rotation.to_matrix());

                if loss < best_loss {
                    best_loss = loss;
                    best_params = candidate;
                    improved = true;
                }
            }

            // Adaptive sigma: grow on success, shrink on failure (1/5 rule).
            if improved {
                sigma *= 1.1;
            } else {
                sigma *= 0.9;
            }
        }

        CayleyRotation::from_params(n, best_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_produces_identity_matrix() {
        let rot = CayleyRotation::identity(4);
        let mat = rot.to_matrix();
        assert_eq!(mat.len(), 16);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0f32 } else { 0.0f32 };
                let actual = mat[i * 4 + j];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "mat[{i},{j}] = {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn random_rotation_is_orthogonal() {
        for seed in [1, 42, 123, 999] {
            for n in [3, 4, 8] {
                let rot = CayleyRotation::from_hadamard(n, seed);
                let err = rot.orthogonality_error();
                assert!(
                    err < 1e-5,
                    "orthogonality error for n={n} seed={seed}: {err}"
                );
            }
        }
    }

    #[test]
    fn optimizer_reduces_loss() {
        let n = 4;

        // Target: an arbitrary orthogonal matrix (a Cayley rotation with known params).
        let target_rot = CayleyRotation::from_hadamard(n, 7);
        let target_mat = target_rot.to_matrix();

        // Loss = ||R * target^T - I||_F^2
        // Minimized when R = target (so R * target^T = I, meaning R == target).
        let mut loss_fn = |r: &[f32]| -> f64 {
            let mut sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    // (R * target^T)[i,j] = sum_k R[i,k] * target[j,k]
                    let mut val = 0.0;
                    for k in 0..n {
                        val += r[i * n + k] as f64 * target_mat[j * n + k] as f64;
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    sum += (val - expected) * (val - expected);
                }
            }
            sum
        };

        let initial = CayleyRotation::identity(n);
        let initial_loss = loss_fn(&initial.to_matrix());

        let optimizer = CayleyOptimizer {
            max_iterations: 200,
            population_size: 40,
            initial_sigma: 0.05,
            tolerance: 1e-10,
        };

        let result = optimizer.optimize(&initial, &mut loss_fn);
        let final_loss = loss_fn(&result.to_matrix());

        assert!(
            final_loss < initial_loss,
            "optimizer did not reduce loss: initial={initial_loss}, final={final_loss}"
        );
    }

    #[test]
    fn optimized_rotation_is_orthogonal() {
        let n = 4;
        let target_rot = CayleyRotation::from_hadamard(n, 7);
        let target_mat = target_rot.to_matrix();

        let mut loss_fn = |r: &[f32]| -> f64 {
            let mut sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let mut val = 0.0;
                    for k in 0..n {
                        val += r[i * n + k] as f64 * target_mat[j * n + k] as f64;
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    sum += (val - expected) * (val - expected);
                }
            }
            sum
        };

        let initial = CayleyRotation::identity(n);
        let optimizer = CayleyOptimizer {
            max_iterations: 50,
            population_size: 20,
            initial_sigma: 0.05,
            tolerance: 1e-10,
        };

        let result = optimizer.optimize(&initial, &mut loss_fn);
        let err = result.orthogonality_error();
        assert!(
            err < 1e-5,
            "optimized rotation is not orthogonal: error = {err}"
        );
    }

    #[test]
    fn n_params_is_correct() {
        assert_eq!(CayleyRotation::identity(3).n_params(), 3);
        assert_eq!(CayleyRotation::identity(4).n_params(), 6);
        assert_eq!(CayleyRotation::identity(8).n_params(), 28);
    }
}
