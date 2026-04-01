//! TurboQuant quantized KV cache compression for GPU inference.
//!
//! Supports INT8 (n_bits=8), INT4 (n_bits=4), and outlier-aware
//! mixed-precision quantization (Section 4.3).

pub mod cache;
pub mod codebook;

pub use cache::GpuKvCache;

use ironmill_metal_sys::MetalDevice;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::error::GpuError;

/// TurboQuant model state for GPU inference.
///
/// Manages the Hadamard rotation signs and dequantization scale
/// used by the fused cache write and attention kernels.
pub struct GpuTurboQuantModel {
    /// Rotation sign vector [head_dim] float (±1.0).
    pub rotation_signs: ironmill_metal_sys::MetalBuffer,
    /// Inverse quantization scale for cache write.
    pub inv_scale: f32,
    /// Dequantization scale for attention.
    pub deq_scale: f32,
    /// Lloyd-Max codebook centroids [n_levels] float.
    pub codebook_buf: ironmill_metal_sys::MetalBuffer,
    /// Lloyd-Max decision boundaries [n_levels-1] float.
    pub boundaries_buf: ironmill_metal_sys::MetalBuffer,
    /// Number of codebook levels.
    pub n_levels: u32,
    /// QJL random projection matrix [head_dim × head_dim] float.
    pub qjl_matrix: ironmill_metal_sys::MetalBuffer,
    /// Outlier channel state (None = standard mode).
    pub outlier: Option<OutlierState>,
    /// Config.
    pub config: TurboQuantGpuConfig,
}

/// GPU buffers for the outlier channel strategy (Section 4.3).
///
/// Two independent TurboQuant instances run on channel subsets:
/// - Outlier channels: higher bit-width (e.g., 4-bit, 16 levels)
/// - Non-outlier channels: lower bit-width (e.g., 3-bit, 8 levels)
pub struct OutlierState {
    /// Channel indices buffer: [n_outlier + n_non_outlier] uint32.
    /// Outlier indices first, then non-outlier indices.
    pub channel_indices: ironmill_metal_sys::MetalBuffer,
    /// Rotation signs for outlier group: [d_outlier_padded] float.
    pub outlier_rotation_signs: ironmill_metal_sys::MetalBuffer,
    /// Rotation signs for non-outlier group: [d_non_padded] float.
    pub non_outlier_rotation_signs: ironmill_metal_sys::MetalBuffer,
    /// Codebook for outlier group: [outlier_n_levels] float.
    pub outlier_codebook: ironmill_metal_sys::MetalBuffer,
    /// Boundaries for outlier group: [outlier_n_levels - 1] float.
    pub outlier_boundaries: ironmill_metal_sys::MetalBuffer,
    /// Codebook for non-outlier group: [non_outlier_n_levels] float.
    pub non_outlier_codebook: ironmill_metal_sys::MetalBuffer,
    /// Boundaries for non-outlier group: [non_outlier_n_levels - 1] float.
    pub non_outlier_boundaries: ironmill_metal_sys::MetalBuffer,
    /// Number of outlier channels.
    pub n_outlier: u32,
    /// Outlier group dimension (padded to power of 2).
    pub d_outlier_padded: u32,
    /// Non-outlier group dimension (padded to power of 2).
    pub d_non_padded: u32,
    /// Number of codebook levels for outlier group.
    pub outlier_n_levels: u32,
    /// Number of codebook levels for non-outlier group.
    pub non_outlier_n_levels: u32,
}

/// Outlier channel configuration for mixed-precision quantization.
#[derive(Debug, Clone)]
pub struct OutlierConfig {
    /// Indices of outlier channels in the original KV dimension space.
    pub outlier_channels: Vec<usize>,
    /// Quantization bits for outlier channels (default: 4).
    pub outlier_bits: u8,
    /// Quantization bits for non-outlier channels (default: 3).
    pub non_outlier_bits: u8,
}

/// TurboQuant-specific configuration extracted from GpuConfig + model params.
#[derive(Debug, Clone)]
pub struct TurboQuantGpuConfig {
    pub n_bits: u8,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub rotation_seed: u64,
    /// Optional outlier channel configuration.
    pub outlier: Option<OutlierConfig>,
}

/// Generate the ±1 sign vector for the randomized Hadamard rotation.
///
/// Returns `dim` f32 values as little-endian bytes, matching
/// the sign generation in `mil-rs/src/ir/passes/rotation.rs`.
pub fn generate_rotation_signs(dim: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    let signs: Vec<f32> = (0..dim)
        .map(|_| if rng.r#gen_bool(0.5) { 1.0f32 } else { -1.0f32 })
        .collect();
    signs.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

/// Generate the QJL random projection matrix S for residual correction.
///
/// Returns a [dim × dim] f32 matrix (row-major, little-endian bytes)
/// where each entry is drawn i.i.d. from N(0, 1/dim). Uses Box-Muller.
pub fn generate_qjl_matrix(dim: usize, seed: u64) -> Vec<u8> {
    let sigma = 1.0 / (dim as f64).sqrt();
    let mut rng = StdRng::seed_from_u64(seed);

    let n = dim * dim;
    let mut values = Vec::with_capacity(n);

    while values.len() < n {
        let u1: f64 = rng.r#gen::<f64>().max(1e-300);
        let u2: f64 = rng.r#gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        values.push((r * theta.cos() * sigma) as f32);
        if values.len() < n {
            values.push((r * theta.sin() * sigma) as f32);
        }
    }

    values.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

fn create_f32_buffer(
    device: &MetalDevice,
    data: &[f32],
) -> Result<ironmill_metal_sys::MetalBuffer, GpuError> {
    let bytes: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    device
        .create_buffer_with_data(&bytes, ironmill_metal_sys::StorageMode::Shared)
        .map_err(GpuError::Metal)
}

fn create_u32_buffer(
    device: &MetalDevice,
    data: &[u32],
) -> Result<ironmill_metal_sys::MetalBuffer, GpuError> {
    let bytes: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    device
        .create_buffer_with_data(&bytes, ironmill_metal_sys::StorageMode::Shared)
        .map_err(GpuError::Metal)
}

impl GpuTurboQuantModel {
    /// Initialize TurboQuant with rotation signs and quantization scales.
    pub fn new(device: &MetalDevice, config: TurboQuantGpuConfig) -> Result<Self, GpuError> {
        assert!(
            config.head_dim.is_power_of_two(),
            "head_dim must be a power of two for Walsh-Hadamard butterfly"
        );

        let sign_bytes = generate_rotation_signs(config.head_dim, config.rotation_seed);
        let rotation_signs = device
            .create_buffer_with_data(&sign_bytes, ironmill_metal_sys::StorageMode::Shared)
            .map_err(GpuError::Metal)?;

        // Compute Lloyd-Max codebook for N(0, 1/√head_dim)
        let (levels, bounds) = codebook::lloyd_max_gaussian(config.head_dim, config.n_bits);
        let n_levels = levels.len() as u32;
        let codebook_buf = create_f32_buffer(device, &levels)?;
        let boundaries_buf = create_f32_buffer(device, &bounds)?;

        // QJL projection matrix (seed+1 for independence)
        let qjl_bytes = generate_qjl_matrix(config.head_dim, config.rotation_seed + 1);
        let qjl_matrix = device
            .create_buffer_with_data(&qjl_bytes, ironmill_metal_sys::StorageMode::Shared)
            .map_err(GpuError::Metal)?;

        // Outlier channel state
        let outlier = if let Some(ref outlier_cfg) = config.outlier {
            Some(Self::init_outlier(device, &config, outlier_cfg)?)
        } else {
            None
        };

        let max_abs = 32.0_f32;
        let inv_scale = 127.0 / max_abs;
        let deq_scale = max_abs / 127.0;

        Ok(Self {
            rotation_signs,
            inv_scale,
            deq_scale,
            codebook_buf,
            boundaries_buf,
            n_levels,
            qjl_matrix,
            outlier,
            config,
        })
    }

    /// Initialize outlier channel state: independent rotation signs and
    /// codebooks for outlier and non-outlier channel groups.
    fn init_outlier(
        device: &MetalDevice,
        config: &TurboQuantGpuConfig,
        outlier_cfg: &OutlierConfig,
    ) -> Result<OutlierState, GpuError> {
        let n_outlier = outlier_cfg.outlier_channels.len();
        let n_non = config.head_dim - n_outlier;
        let d_outlier_padded = n_outlier.next_power_of_two();
        let d_non_padded = n_non.next_power_of_two();

        // Build index buffers: outlier indices first, then non-outlier
        let mut is_outlier = vec![false; config.head_dim];
        for &idx in &outlier_cfg.outlier_channels {
            assert!(idx < config.head_dim, "outlier channel index out of range");
            is_outlier[idx] = true;
        }
        let outlier_indices: Vec<u32> = outlier_cfg
            .outlier_channels
            .iter()
            .map(|&i| i as u32)
            .collect();
        let non_outlier_indices: Vec<u32> = (0..config.head_dim)
            .filter(|&i| !is_outlier[i])
            .map(|i| i as u32)
            .collect();
        let mut all_indices = outlier_indices;
        all_indices.extend_from_slice(&non_outlier_indices);
        let channel_indices = create_u32_buffer(device, &all_indices)?;

        // Independent rotation signs (different seeds for each group)
        let outlier_sign_bytes =
            generate_rotation_signs(d_outlier_padded, config.rotation_seed + 100);
        let outlier_rotation_signs = device
            .create_buffer_with_data(&outlier_sign_bytes, ironmill_metal_sys::StorageMode::Shared)
            .map_err(GpuError::Metal)?;

        let non_outlier_sign_bytes =
            generate_rotation_signs(d_non_padded, config.rotation_seed + 200);
        let non_outlier_rotation_signs = device
            .create_buffer_with_data(
                &non_outlier_sign_bytes,
                ironmill_metal_sys::StorageMode::Shared,
            )
            .map_err(GpuError::Metal)?;

        // Independent codebooks: outlier at higher bits, non-outlier at lower bits
        let (o_levels, o_bounds) =
            codebook::lloyd_max_gaussian(d_outlier_padded, outlier_cfg.outlier_bits);
        let outlier_n_levels = o_levels.len() as u32;
        let outlier_codebook = create_f32_buffer(device, &o_levels)?;
        let outlier_boundaries = create_f32_buffer(device, &o_bounds)?;

        let (n_levels_vec, n_bounds) =
            codebook::lloyd_max_gaussian(d_non_padded, outlier_cfg.non_outlier_bits);
        let non_outlier_n_levels = n_levels_vec.len() as u32;
        let non_outlier_codebook = create_f32_buffer(device, &n_levels_vec)?;
        let non_outlier_boundaries = create_f32_buffer(device, &n_bounds)?;

        Ok(OutlierState {
            channel_indices,
            outlier_rotation_signs,
            non_outlier_rotation_signs,
            outlier_codebook,
            outlier_boundaries,
            non_outlier_codebook,
            non_outlier_boundaries,
            n_outlier: n_outlier as u32,
            d_outlier_padded: d_outlier_padded as u32,
            d_non_padded: d_non_padded as u32,
            outlier_n_levels,
            non_outlier_n_levels,
        })
    }
}
