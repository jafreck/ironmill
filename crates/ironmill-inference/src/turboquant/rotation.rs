//! Rotation sign and QJL random projection matrix generation for TurboQuant.
//!
//! These are backend-independent: the GPU backend consumes the raw `Vec<u8>`
//! little-endian bytes directly.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Generate the ±1 sign vector for the randomized Hadamard rotation.
///
/// Returns `dim` f32 values as little-endian bytes, matching
/// the sign generation in `mil-rs/src/ir/passes/rotation.rs`.
pub fn generate_rotation_signs(dim: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    let signs: Vec<f32> = (0..dim)
        .map(|_| {
            if rng.random_bool(0.5) {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect();
    signs.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

/// Generate the QJL random projection matrix S for residual correction.
///
/// Returns a [dim × dim] f32 matrix (row-major, little-endian bytes)
/// where each entry is drawn i.i.d. from N(0, 1) per Algorithm 2 line 3.
pub fn generate_qjl_matrix(dim: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);

    let n = dim * dim;
    let mut values = Vec::with_capacity(n);

    while values.len() < n {
        let u1: f64 = rng.random::<f64>().max(1e-300);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0_f64 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        values.push((r * theta.cos()) as f32);
        if values.len() < n {
            values.push((r * theta.sin()) as f32);
        }
    }

    values.iter().flat_map(|&v: &f32| v.to_le_bytes()).collect()
}
