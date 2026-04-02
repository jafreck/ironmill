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

use crate::turboquant::codebook::lloyd_max_gaussian;
use crate::turboquant::outlier::OutlierConfig;
use crate::turboquant::rotation::{generate_qjl_matrix, generate_rotation_signs};

use super::config::MlxConfig;
use super::error::MlxError;

// ── Config types ────────────────────────────────────────────────

/// Re-export shared OutlierConfig as MlxOutlierConfig for backward
/// compatibility with any code that uses the MLX-specific name.
pub type MlxOutlierConfig = OutlierConfig;

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
    /// Uses the shared Lloyd-Max codebook generation
    /// (see [`crate::turboquant::codebook`]).
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

        // Generate rotation signs (shared function returns le bytes)
        let sign_bytes = generate_rotation_signs(head_dim, rotation_seed);
        let rotation_signs =
            MlxArray::from_data_copy(&sign_bytes, &[head_dim], MlxDtype::Float32, stream)?;

        // K codebook: use full b-bit for both K and V.
        // The (b-1)-bit + 1-bit-QJL-sign approach (Algorithm 2) trades codebook
        // precision for inner-product correction, but with d projections in
        // d dimensions the estimator variance (π/(2d)) exceeds the quantization
        // error (1/d), making the correction counterproductive.
        let (k_levels, k_bounds) = lloyd_max_gaussian(head_dim, n_bits);
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
        let qjl_bytes = generate_qjl_matrix(head_dim, rotation_seed + 1);
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

        // Independent rotation signs (shared functions return le bytes directly)
        let o_signs_bytes = generate_rotation_signs(d_outlier_padded, rotation_seed + 100);
        let outlier_rotation_signs = MlxArray::from_data_copy(
            &o_signs_bytes,
            &[d_outlier_padded],
            MlxDtype::Float32,
            stream,
        )?;

        let n_signs_bytes = generate_rotation_signs(d_non_padded, rotation_seed + 200);
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

        // QJL projection matrices (independent seeds, shared functions return le bytes)
        let o_qjl_bytes = generate_qjl_matrix(d_outlier_padded, rotation_seed + 300);
        let outlier_qjl_matrix = MlxArray::from_data_copy(
            &o_qjl_bytes,
            &[d_outlier_padded * d_outlier_padded],
            MlxDtype::Float32,
            stream,
        )?;

        let n_qjl_bytes = generate_qjl_matrix(d_non_padded, rotation_seed + 400);
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
