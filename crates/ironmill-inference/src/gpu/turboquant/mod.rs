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
/// Per Algorithm 2 (TurboQuant_prod), K and V caches use different codebooks:
/// - K cache: (b-1)-bit MSE codebook + 1-bit QJL = b bits total (inner product)
/// - V cache: b-bit MSE codebook, no QJL (MSE reconstruction)
pub struct GpuTurboQuantModel {
    /// Rotation sign vector [head_dim] float (±1.0).
    pub rotation_signs: ironmill_metal_sys::MetalBuffer,
    /// Inverse quantization scale for cache write (INT8 path only).
    pub inv_scale: f32,
    /// Dequantization scale for attention (INT8 path only).
    pub deq_scale: f32,
    /// K cache codebook: (b-1)-bit Lloyd-Max centroids (for inner product via QJL).
    pub k_codebook_buf: ironmill_metal_sys::MetalBuffer,
    /// K cache boundaries: (b-1)-bit decision thresholds.
    pub k_boundaries_buf: ironmill_metal_sys::MetalBuffer,
    /// Number of K codebook levels (2^(b-1)).
    pub k_n_levels: u32,
    /// V cache codebook: b-bit Lloyd-Max centroids (for MSE reconstruction).
    pub v_codebook_buf: ironmill_metal_sys::MetalBuffer,
    /// V cache boundaries: b-bit decision thresholds.
    pub v_boundaries_buf: ironmill_metal_sys::MetalBuffer,
    /// Number of V codebook levels (2^b).
    pub v_n_levels: u32,
    /// QJL random projection matrix [head_dim × head_dim] float (K cache only).
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
    /// V codebook for outlier group: [outlier_n_levels] float.
    pub outlier_codebook: ironmill_metal_sys::MetalBuffer,
    /// V boundaries for outlier group: [outlier_n_levels - 1] float.
    pub outlier_boundaries: ironmill_metal_sys::MetalBuffer,
    /// V codebook for non-outlier group: [non_outlier_n_levels] float.
    pub non_outlier_codebook: ironmill_metal_sys::MetalBuffer,
    /// V boundaries for non-outlier group: [non_outlier_n_levels - 1] float.
    pub non_outlier_boundaries: ironmill_metal_sys::MetalBuffer,
    /// K codebook for outlier group: (outlier_bits-1)-bit.
    pub k_outlier_codebook: ironmill_metal_sys::MetalBuffer,
    /// K boundaries for outlier group.
    pub k_outlier_boundaries: ironmill_metal_sys::MetalBuffer,
    /// Number of K codebook levels for outlier group.
    pub k_outlier_n_levels: u32,
    /// K codebook for non-outlier group: (non_outlier_bits-1)-bit.
    pub k_non_outlier_codebook: ironmill_metal_sys::MetalBuffer,
    /// K boundaries for non-outlier group.
    pub k_non_outlier_boundaries: ironmill_metal_sys::MetalBuffer,
    /// Number of K codebook levels for non-outlier group.
    pub k_non_outlier_n_levels: u32,
    /// QJL matrix for outlier group: [d_outlier_padded²] float.
    pub outlier_qjl_matrix: ironmill_metal_sys::MetalBuffer,
    /// QJL matrix for non-outlier group: [d_non_padded²] float.
    pub non_outlier_qjl_matrix: ironmill_metal_sys::MetalBuffer,
    /// Number of outlier channels.
    pub n_outlier: u32,
    /// Outlier group dimension (padded to power of 2).
    pub d_outlier_padded: u32,
    /// Non-outlier group dimension (padded to power of 2).
    pub d_non_padded: u32,
    /// Number of V codebook levels for outlier group.
    pub outlier_n_levels: u32,
    /// Number of V codebook levels for non-outlier group.
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

impl OutlierConfig {
    /// Detect outlier channels from K/V projection weight column norms.
    ///
    /// Paper §4.3: channels with high variance after rotation are outliers.
    /// Since the Hadamard rotation is orthogonal, the post-rotation variance
    /// of channel `i` is proportional to the squared column norm of the
    /// projection weight matrix at that channel index. We select the top
    /// `n_outlier` channels by column norm across all layers.
    ///
    /// `weight_data` is a slice of (K weight, V weight) per layer, each as
    /// row-major f16 bytes with shape [out_features × in_features].
    /// `head_dim` is the per-head dimension (= out_features / num_kv_heads).
    /// `n_outlier` is how many channels to flag (paper uses d/4 = 32 for d=128).
    pub fn from_weight_norms(
        weight_data: &[(&[u8], &[u8])],
        out_features: usize,
        head_dim: usize,
        n_outlier: usize,
        outlier_bits: u8,
        non_outlier_bits: u8,
    ) -> Self {
        // Accumulate per-channel (per-head-dim-index) squared norms across
        // all KV heads and layers. Channel index is dim % head_dim.
        let mut channel_energy = vec![0.0f64; head_dim];

        for (k_bytes, v_bytes) in weight_data {
            for weight_bytes in [k_bytes, v_bytes] {
                // Weight is [out_features × in_features] stored as f16, row-major.
                // Column `j` contributes to channel `j % head_dim`.
                // The norm of column `j` indicates the energy of that output dimension.
                let n_rows = weight_bytes.len() / (out_features * 2);
                if n_rows == 0 {
                    continue;
                }
                for out_idx in 0..out_features {
                    let ch = out_idx % head_dim;
                    // Compute squared norm of row `out_idx` as a proxy
                    // (row norm = how much this output channel depends on inputs).
                    let row_start = out_idx * n_rows * 2;
                    let row_end = row_start + n_rows * 2;
                    if row_end > weight_bytes.len() {
                        continue;
                    }
                    let mut sq_sum = 0.0f64;
                    for i in (row_start..row_end).step_by(2) {
                        let val = half::f16::from_le_bytes([weight_bytes[i], weight_bytes[i + 1]])
                            .to_f64();
                        sq_sum += val * val;
                    }
                    channel_energy[ch] += sq_sum;
                }
            }
        }

        // Select top n_outlier channels by energy
        let mut indexed: Vec<(usize, f64)> = channel_energy.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut outlier_channels: Vec<usize> =
            indexed[..n_outlier].iter().map(|(i, _)| *i).collect();
        outlier_channels.sort(); // keep in ascending order for index buffer

        Self {
            outlier_channels,
            outlier_bits,
            non_outlier_bits,
        }
    }

    /// Create an outlier config with the default paper settings:
    /// top d/4 channels by weight energy, 4-bit outlier / 3-bit non-outlier.
    pub fn auto_from_weights(
        weight_data: &[(&[u8], &[u8])],
        out_features: usize,
        head_dim: usize,
    ) -> Self {
        Self::from_weight_norms(
            weight_data,
            out_features,
            head_dim,
            head_dim / 4, // d/4 outlier channels (paper default)
            4,            // 4-bit outlier
            3,            // 3-bit non-outlier
        )
    }
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
/// where each entry is drawn i.i.d. from N(0, 1) per Algorithm 2 line 3.
pub fn generate_qjl_matrix(dim: usize, seed: u64) -> Vec<u8> {
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
        if !config.head_dim.is_power_of_two() {
            return Err(GpuError::Config(
                "head_dim must be a power of two for Walsh-Hadamard butterfly".to_string(),
            ));
        }
        assert!(
            config.head_dim <= 512,
            "head_dim must be <= 512 for TurboQuant kernel shared memory"
        );

        let sign_bytes = generate_rotation_signs(config.head_dim, config.rotation_seed);
        let rotation_signs = device
            .create_buffer_with_data(&sign_bytes, ironmill_metal_sys::StorageMode::Shared)
            .map_err(GpuError::Metal)?;

        // Per Algorithm 2 (TurboQuant_prod): K cache uses (b-1)-bit MSE
        // codebook + 1-bit QJL = b bits total. V cache uses b-bit MSE codebook.
        // At INT8, QJL overhead is negligible so both use 8-bit.
        let (k_levels, k_bounds) = if config.n_bits == 4 {
            codebook::lloyd_max_gaussian(config.head_dim, config.n_bits - 1)
        } else {
            codebook::lloyd_max_gaussian(config.head_dim, config.n_bits)
        };
        let k_n_levels = k_levels.len() as u32;
        let k_codebook_buf = create_f32_buffer(device, &k_levels)?;
        let k_boundaries_buf = create_f32_buffer(device, &k_bounds)?;

        let (v_levels, v_bounds) = codebook::lloyd_max_gaussian(config.head_dim, config.n_bits);
        let v_n_levels = v_levels.len() as u32;
        let v_codebook_buf = create_f32_buffer(device, &v_levels)?;
        let v_boundaries_buf = create_f32_buffer(device, &v_bounds)?;

        // QJL projection matrix for K cache (seed+1 for independence)
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
            k_codebook_buf,
            k_boundaries_buf,
            k_n_levels,
            v_codebook_buf,
            v_boundaries_buf,
            v_n_levels,
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

        // Independent codebooks for V: outlier at higher bits, non-outlier at lower bits
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

        // K codebooks: (b-1)-bit MSE codebook + 1-bit QJL = b bits total
        let (ko_levels, ko_bounds) =
            codebook::lloyd_max_gaussian(d_outlier_padded, outlier_cfg.outlier_bits - 1);
        let k_outlier_n_levels = ko_levels.len() as u32;
        let k_outlier_codebook = create_f32_buffer(device, &ko_levels)?;
        let k_outlier_boundaries = create_f32_buffer(device, &ko_bounds)?;

        let (kn_levels, kn_bounds) =
            codebook::lloyd_max_gaussian(d_non_padded, outlier_cfg.non_outlier_bits - 1);
        let k_non_outlier_n_levels = kn_levels.len() as u32;
        let k_non_outlier_codebook = create_f32_buffer(device, &kn_levels)?;
        let k_non_outlier_boundaries = create_f32_buffer(device, &kn_bounds)?;

        // QJL projection matrices (independent seeds)
        let o_qjl_bytes = generate_qjl_matrix(d_outlier_padded, config.rotation_seed + 300);
        let outlier_qjl_matrix = device
            .create_buffer_with_data(&o_qjl_bytes, ironmill_metal_sys::StorageMode::Shared)
            .map_err(GpuError::Metal)?;

        let n_qjl_bytes = generate_qjl_matrix(d_non_padded, config.rotation_seed + 400);
        let non_outlier_qjl_matrix = device
            .create_buffer_with_data(&n_qjl_bytes, ironmill_metal_sys::StorageMode::Shared)
            .map_err(GpuError::Metal)?;

        Ok(OutlierState {
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
            n_outlier: n_outlier as u32,
            d_outlier_padded: d_outlier_padded as u32,
            d_non_padded: d_non_padded as u32,
            outlier_n_levels,
            non_outlier_n_levels,
        })
    }
}
