//! TurboQuant quantized KV cache compression for MLX inference.
//!
//! Ports the GPU TurboQuant backend to MLX, dispatching custom Metal kernels
//! via [`ironmill_mlx_sys::metal_kernel`]. Supports INT4/INT8 quantization,
//! outlier-aware mixed-precision, and QJL residual correction.

pub mod cache;
pub mod kernels;

pub use cache::MlxKvCache;

use ironmill_mlx_sys::{MlxArray, MlxDtype, MlxStream};
use mil_rs::weights::ModelConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::config::MlxConfig;
use super::error::MlxError;

// ── Config types (redefined from gpu::turboquant to avoid coupling) ─

/// Outlier channel configuration for mixed-precision quantization.
#[derive(Debug, Clone)]
pub struct MlxOutlierConfig {
    /// Indices of outlier channels in the original KV dimension space.
    pub outlier_channels: Vec<usize>,
    /// Quantization bits for outlier channels (default: 4).
    pub outlier_bits: u8,
    /// Quantization bits for non-outlier channels (default: 3).
    pub non_outlier_bits: u8,
}

/// QJL random projection configuration.
#[derive(Debug, Clone)]
pub struct MlxQjlConfig {
    /// Random projection matrix seed (independent from rotation seed).
    pub seed: u64,
}

/// MLX arrays for the outlier channel state.
pub struct MlxOutlierModel {
    /// Channel indices: `[head_dim]` Uint32 — outlier first, then non-outlier.
    pub channel_indices: MlxArray,
    /// Rotation signs for outlier group: `[d_outlier_padded]` Float32.
    pub outlier_rotation_signs: MlxArray,
    /// Rotation signs for non-outlier group: `[d_non_padded]` Float32.
    pub non_outlier_rotation_signs: MlxArray,
    /// V codebook for outlier group.
    pub outlier_codebook: MlxArray,
    /// V boundaries for outlier group.
    pub outlier_boundaries: MlxArray,
    /// V codebook for non-outlier group.
    pub non_outlier_codebook: MlxArray,
    /// V boundaries for non-outlier group.
    pub non_outlier_boundaries: MlxArray,
    /// K codebook for outlier group: (outlier_bits-1)-bit.
    pub k_outlier_codebook: MlxArray,
    /// K boundaries for outlier group.
    pub k_outlier_boundaries: MlxArray,
    /// Number of K codebook levels for outlier group.
    pub k_outlier_n_levels: usize,
    /// K codebook for non-outlier group: (non_outlier_bits-1)-bit.
    pub k_non_outlier_codebook: MlxArray,
    /// K boundaries for non-outlier group.
    pub k_non_outlier_boundaries: MlxArray,
    /// Number of K codebook levels for non-outlier group.
    pub k_non_outlier_n_levels: usize,
    /// QJL matrix for outlier group: `[d_outlier_padded²]` Float32.
    pub outlier_qjl_matrix: MlxArray,
    /// QJL matrix for non-outlier group: `[d_non_padded²]` Float32.
    pub non_outlier_qjl_matrix: MlxArray,
    /// Number of V codebook levels for outlier group.
    pub outlier_n_levels: usize,
    /// Number of V codebook levels for non-outlier group.
    pub non_outlier_n_levels: usize,
}

/// Per-layer outlier cache arrays.
pub struct MlxOutlierCache {
    pub k_outlier_caches: Vec<MlxArray>,
    pub v_outlier_caches: Vec<MlxArray>,
    pub k_non_outlier_caches: Vec<MlxArray>,
    pub v_non_outlier_caches: Vec<MlxArray>,
    pub k_outlier_scales: Vec<MlxArray>,
    pub v_outlier_scales: Vec<MlxArray>,
    pub k_non_outlier_scales: Vec<MlxArray>,
    pub v_non_outlier_scales: Vec<MlxArray>,
    /// Outlier group K residual norms per layer.
    pub k_outlier_r_norms: Vec<MlxArray>,
    /// Non-outlier group K residual norms per layer.
    pub k_non_outlier_r_norms: Vec<MlxArray>,
}

// ── MlxTurboQuantModel ──────────────────────────────────────────

/// TurboQuant model state for MLX inference.
///
/// Per Algorithm 2 (TurboQuant_prod), K and V caches use different codebooks:
/// - K cache: (b-1)-bit MSE codebook + 1-bit QJL = b bits total
/// - V cache: b-bit MSE codebook, no QJL
pub struct MlxTurboQuantModel {
    /// Rotation sign vector `[head_dim]` Float32 (±1.0).
    pub rotation_signs: MlxArray,
    /// K cache codebook centroids (Rust-side): (b-1)-bit at 4-bit, b-bit at 8-bit.
    pub k_codebook: Vec<f32>,
    /// K codebook as MLX array `[k_n_levels]` Float32.
    pub k_codebook_arr: MlxArray,
    /// K boundaries (Rust-side).
    pub k_boundaries: Vec<f32>,
    /// K boundaries as MLX array.
    pub k_boundaries_arr: MlxArray,
    /// V cache codebook centroids (Rust-side): b-bit.
    pub v_codebook: Vec<f32>,
    /// V codebook as MLX array `[v_n_levels]` Float32.
    pub v_codebook_arr: MlxArray,
    /// V boundaries (Rust-side).
    pub v_boundaries: Vec<f32>,
    /// V boundaries as MLX array.
    pub v_boundaries_arr: MlxArray,
    /// Number of quantization bits (4 or 8).
    pub n_bits: u8,
    /// Outlier channel configuration (None = standard mode).
    pub outlier_config: Option<MlxOutlierConfig>,
    /// Outlier model state (None if no outlier config).
    pub outlier_model: Option<MlxOutlierModel>,
    /// QJL random projection matrix `[head_dim × head_dim]` Float32.
    pub qjl_matrix: Option<MlxArray>,
}

impl MlxTurboQuantModel {
    /// Initialize TurboQuant model state from model and MLX configs.
    ///
    /// Uses the same Lloyd-Max codebook generation as the GPU backend
    /// (see [`crate::gpu::turboquant::codebook`]).
    pub fn new(
        model_config: &ModelConfig,
        mlx_config: &MlxConfig,
        stream: &MlxStream,
    ) -> Result<Self, MlxError> {
        let head_dim = model_config.head_dim;
        let n_bits = mlx_config.n_bits;
        let rotation_seed = mlx_config.rotation_seed;

        assert!(
            head_dim.is_power_of_two(),
            "head_dim must be a power of two for Walsh-Hadamard butterfly"
        );
        assert!(
            head_dim <= 512,
            "head_dim must be <= 512 for TurboQuant kernel shared memory"
        );

        // Generate rotation signs
        let sign_data = generate_rotation_signs(head_dim, rotation_seed);
        let sign_bytes: Vec<u8> = sign_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let rotation_signs =
            MlxArray::from_data_copy(&sign_bytes, &[head_dim], MlxDtype::Float32, stream)?;

        // Generate K codebook: (b-1)-bit for INT4 (3-bit + 1-bit QJL), b-bit for INT8
        let k_bits = if n_bits == 4 { n_bits - 1 } else { n_bits };
        let (k_levels, k_bounds) = lloyd_max_gaussian(head_dim, k_bits);
        let k_codebook_bytes: Vec<u8> = k_levels.iter().flat_map(|v| v.to_le_bytes()).collect();
        let k_codebook_arr = MlxArray::from_data_copy(
            &k_codebook_bytes,
            &[k_levels.len()],
            MlxDtype::Float32,
            stream,
        )?;
        let k_boundaries_bytes: Vec<u8> = k_bounds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let k_boundaries_arr = MlxArray::from_data_copy(
            &k_boundaries_bytes,
            &[k_bounds.len()],
            MlxDtype::Float32,
            stream,
        )?;

        // Generate V codebook: b-bit (full precision, no QJL)
        let (v_levels, v_bounds) = lloyd_max_gaussian(head_dim, n_bits);
        let v_codebook_bytes: Vec<u8> = v_levels.iter().flat_map(|v| v.to_le_bytes()).collect();
        let v_codebook_arr = MlxArray::from_data_copy(
            &v_codebook_bytes,
            &[v_levels.len()],
            MlxDtype::Float32,
            stream,
        )?;
        let v_boundaries_bytes: Vec<u8> = v_bounds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let v_boundaries_arr = MlxArray::from_data_copy(
            &v_boundaries_bytes,
            &[v_bounds.len()],
            MlxDtype::Float32,
            stream,
        )?;

        // Generate QJL matrix (seed+1 for independence from rotation)
        let qjl_data = generate_qjl_matrix(head_dim, rotation_seed + 1);
        let qjl_bytes: Vec<u8> = qjl_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let qjl_matrix = MlxArray::from_data_copy(
            &qjl_bytes,
            &[head_dim * head_dim],
            MlxDtype::Float32,
            stream,
        )?;

        Ok(Self {
            rotation_signs,
            k_codebook: k_levels,
            k_codebook_arr,
            k_boundaries: k_bounds,
            k_boundaries_arr,
            v_codebook: v_levels,
            v_codebook_arr,
            v_boundaries: v_bounds,
            v_boundaries_arr,
            n_bits,
            outlier_config: None,
            outlier_model: None,
            qjl_matrix: Some(qjl_matrix),
        })
    }

    /// Initialize outlier channel state from an outlier config.
    pub fn init_outlier(
        &mut self,
        outlier_cfg: MlxOutlierConfig,
        head_dim: usize,
        rotation_seed: u64,
        stream: &MlxStream,
    ) -> Result<(), MlxError> {
        let n_outlier = outlier_cfg.outlier_channels.len();
        let n_non = head_dim - n_outlier;
        let d_outlier_padded = n_outlier.next_power_of_two();
        let d_non_padded = n_non.next_power_of_two();

        // Build index buffer: outlier indices first, then non-outlier
        let mut is_outlier = vec![false; head_dim];
        for &idx in &outlier_cfg.outlier_channels {
            is_outlier[idx] = true;
        }
        let mut all_indices: Vec<u32> = outlier_cfg
            .outlier_channels
            .iter()
            .map(|&i| i as u32)
            .collect();
        let non_outlier_indices: Vec<u32> = (0..head_dim)
            .filter(|&i| !is_outlier[i])
            .map(|i| i as u32)
            .collect();
        all_indices.extend_from_slice(&non_outlier_indices);

        let idx_bytes: Vec<u8> = all_indices.iter().flat_map(|v| v.to_le_bytes()).collect();
        let channel_indices =
            MlxArray::from_data_copy(&idx_bytes, &[all_indices.len()], MlxDtype::Uint32, stream)?;

        // Independent rotation signs
        let o_signs = generate_rotation_signs(d_outlier_padded, rotation_seed + 100);
        let o_signs_bytes: Vec<u8> = o_signs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let outlier_rotation_signs = MlxArray::from_data_copy(
            &o_signs_bytes,
            &[d_outlier_padded],
            MlxDtype::Float32,
            stream,
        )?;

        let n_signs = generate_rotation_signs(d_non_padded, rotation_seed + 200);
        let n_signs_bytes: Vec<u8> = n_signs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let non_outlier_rotation_signs =
            MlxArray::from_data_copy(&n_signs_bytes, &[d_non_padded], MlxDtype::Float32, stream)?;

        // Independent codebooks
        let (o_levels, o_bounds) = lloyd_max_gaussian(d_outlier_padded, outlier_cfg.outlier_bits);
        let o_cb_bytes: Vec<u8> = o_levels.iter().flat_map(|v| v.to_le_bytes()).collect();
        let outlier_codebook =
            MlxArray::from_data_copy(&o_cb_bytes, &[o_levels.len()], MlxDtype::Float32, stream)?;
        let o_bd_bytes: Vec<u8> = o_bounds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let outlier_boundaries =
            MlxArray::from_data_copy(&o_bd_bytes, &[o_bounds.len()], MlxDtype::Float32, stream)?;

        let (n_levels_vec, n_bounds) =
            lloyd_max_gaussian(d_non_padded, outlier_cfg.non_outlier_bits);
        let n_cb_bytes: Vec<u8> = n_levels_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
        let non_outlier_codebook = MlxArray::from_data_copy(
            &n_cb_bytes,
            &[n_levels_vec.len()],
            MlxDtype::Float32,
            stream,
        )?;
        let n_bd_bytes: Vec<u8> = n_bounds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let non_outlier_boundaries =
            MlxArray::from_data_copy(&n_bd_bytes, &[n_bounds.len()], MlxDtype::Float32, stream)?;

        // K codebooks: (b-1)-bit MSE codebook + 1-bit QJL = b bits total
        let (ko_levels, ko_bounds) =
            lloyd_max_gaussian(d_outlier_padded, outlier_cfg.outlier_bits - 1);
        let ko_cb_bytes: Vec<u8> = ko_levels.iter().flat_map(|v| v.to_le_bytes()).collect();
        let k_outlier_codebook =
            MlxArray::from_data_copy(&ko_cb_bytes, &[ko_levels.len()], MlxDtype::Float32, stream)?;
        let ko_bd_bytes: Vec<u8> = ko_bounds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let k_outlier_boundaries =
            MlxArray::from_data_copy(&ko_bd_bytes, &[ko_bounds.len()], MlxDtype::Float32, stream)?;
        let k_outlier_n_levels = ko_levels.len();

        let (kn_levels, kn_bounds) =
            lloyd_max_gaussian(d_non_padded, outlier_cfg.non_outlier_bits - 1);
        let kn_cb_bytes: Vec<u8> = kn_levels.iter().flat_map(|v| v.to_le_bytes()).collect();
        let k_non_outlier_codebook =
            MlxArray::from_data_copy(&kn_cb_bytes, &[kn_levels.len()], MlxDtype::Float32, stream)?;
        let kn_bd_bytes: Vec<u8> = kn_bounds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let k_non_outlier_boundaries =
            MlxArray::from_data_copy(&kn_bd_bytes, &[kn_bounds.len()], MlxDtype::Float32, stream)?;
        let k_non_outlier_n_levels = kn_levels.len();

        // QJL projection matrices (independent seeds)
        let o_qjl = generate_qjl_matrix(d_outlier_padded, rotation_seed + 300);
        let o_qjl_bytes: Vec<u8> = o_qjl.iter().flat_map(|v| v.to_le_bytes()).collect();
        let outlier_qjl_matrix = MlxArray::from_data_copy(
            &o_qjl_bytes,
            &[d_outlier_padded * d_outlier_padded],
            MlxDtype::Float32,
            stream,
        )?;

        let n_qjl = generate_qjl_matrix(d_non_padded, rotation_seed + 400);
        let n_qjl_bytes: Vec<u8> = n_qjl.iter().flat_map(|v| v.to_le_bytes()).collect();
        let non_outlier_qjl_matrix = MlxArray::from_data_copy(
            &n_qjl_bytes,
            &[d_non_padded * d_non_padded],
            MlxDtype::Float32,
            stream,
        )?;

        self.outlier_model = Some(MlxOutlierModel {
            channel_indices,
            outlier_rotation_signs,
            non_outlier_rotation_signs,
            outlier_codebook,
            outlier_boundaries,
            non_outlier_codebook,
            non_outlier_boundaries,
            k_outlier_codebook,
            k_outlier_boundaries,
            k_outlier_n_levels,
            k_non_outlier_codebook,
            k_non_outlier_boundaries,
            k_non_outlier_n_levels,
            outlier_qjl_matrix,
            non_outlier_qjl_matrix,
            outlier_n_levels: o_levels.len(),
            non_outlier_n_levels: n_levels_vec.len(),
        });
        self.outlier_config = Some(outlier_cfg);

        Ok(())
    }
}

// ── Codebook generation (reimplemented to avoid cross-crate dep) ─

/// Maximum Lloyd-Max iterations.
const MAX_ITERATIONS: usize = 200;
/// Convergence threshold.
const CONVERGENCE_EPS: f64 = 1e-10;

/// Lloyd-Max optimal codebook for N(0, 1/√d).
///
/// Returns `(levels, boundaries)` matching
/// [`crate::gpu::turboquant::codebook::lloyd_max_gaussian`].
fn lloyd_max_gaussian(dim: usize, n_bits: u8) -> (Vec<f32>, Vec<f32>) {
    let sigma = 1.0 / (dim as f64).sqrt();
    let n_levels: usize = 1 << n_bits;

    let mut levels: Vec<f64> = (0..n_levels)
        .map(|i| {
            let p = (i as f64 + 0.5) / n_levels as f64;
            sigma * normal_quantile(p)
        })
        .collect();

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

fn midpoints(levels: &[f64]) -> Vec<f64> {
    levels.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
}

fn compute_centroids(sigma: f64, boundaries: &[f64], n_levels: usize) -> Vec<f64> {
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

fn conditional_mean_gaussian(sigma: f64, lo: f64, hi: f64) -> f64 {
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
    let numerator = normal_pdf(lo_z) - normal_pdf(hi_z);
    sigma * numerator / prob
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let t = p - 0.5;
    if t.abs() <= 0.42 {
        let r = t * t;
        let num = t
            * (((2.5066282746310002 * r + 18.6150006252966) * r + 41.39119773534996) * r
                + 25.44106049637689);
        let den = (((14.382029373388_f64 * r + 63.690613_f64) * r + 84.681_f64) * r + 40.304_f64)
            * r
            + 1.0;
        num / den
    } else {
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

fn erf(x: f64) -> f64 {
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

// ── Sign / QJL generation (mirrors gpu::turboquant) ─────────────

/// Generate ±1 sign vector for randomized Hadamard rotation.
fn generate_rotation_signs(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim)
        .map(|_| if rng.r#gen_bool(0.5) { 1.0f32 } else { -1.0f32 })
        .collect()
}

/// Generate QJL random projection matrix `[dim × dim]` from N(0,1).
fn generate_qjl_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = dim * dim;
    let mut values = Vec::with_capacity(n);
    while values.len() < n {
        let u1: f64 = rng.r#gen::<f64>().max(1e-300);
        let u2: f64 = rng.r#gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        values.push((r * theta.cos()) as f32);
        if values.len() < n {
            values.push((r * theta.sin()) as f32);
        }
    }
    values
}
