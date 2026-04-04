//! TurboQuant quantized KV cache compression for GPU inference.
//!
//! Supports INT8 (n_bits=8), INT4 (n_bits=4), and outlier-aware
//! mixed-precision quantization (Section 4.3).

pub mod cache;
pub mod codebook;

pub use crate::turboquant::outlier::OutlierConfig;
pub use cache::MetalKvCache;

use ironmill_metal_sys::MetalDevice;

use crate::turboquant::rotation::{generate_qjl_matrix, generate_rotation_signs};

use super::error::MetalError;

/// Per-dimension codebook, rotation, and QJL buffers for one unique head_dim.
///
/// When all layers share the same head_dim, the default codebooks on
/// [`MetalTurboQuantModel`] are used and this struct is not needed.
/// For heterogeneous models (e.g. Gemma 4 with sliding=256, global=512),
/// one `DimCodebooks` is stored per unique non-default head_dim.
pub struct DimCodebooks {
    /// Rotation sign vector [head_dim] float (±1.0).
    pub rotation_signs: ironmill_metal_sys::MetalBuffer,
    /// K cache codebook: (b-1)-bit Lloyd-Max centroids.
    pub k_codebook_buf: ironmill_metal_sys::MetalBuffer,
    /// K cache boundaries: (b-1)-bit decision thresholds.
    pub k_boundaries_buf: ironmill_metal_sys::MetalBuffer,
    /// Number of K codebook levels (2^(b-1)).
    pub k_n_levels: u32,
    /// V cache codebook: b-bit Lloyd-Max centroids.
    pub v_codebook_buf: ironmill_metal_sys::MetalBuffer,
    /// V cache boundaries: b-bit decision thresholds.
    pub v_boundaries_buf: ironmill_metal_sys::MetalBuffer,
    /// Number of V codebook levels (2^b).
    pub v_n_levels: u32,
    /// QJL random projection matrix [head_dim × head_dim] float.
    pub qjl_matrix: ironmill_metal_sys::MetalBuffer,
    /// The head_dim these codebooks were generated for.
    pub head_dim: usize,
}

/// TurboQuant model state for GPU inference.
///
/// Per Algorithm 2 (TurboQuant_prod), K and V caches use different codebooks:
/// - K cache: (b-1)-bit MSE codebook + 1-bit QJL = b bits total (inner product)
/// - V cache: b-bit MSE codebook, no QJL (MSE reconstruction)
pub struct MetalTurboQuantModel {
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
    pub config: TurboQuantMetalConfig,
    /// Per-unique-head_dim codebook sets for heterogeneous models.
    /// Empty when all layers use the default `config.head_dim`.
    pub dim_codebooks: Vec<DimCodebooks>,
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

/// Per-layer TurboQuant configuration for heterogeneous head dims (e.g. Gemma 4).
#[derive(Debug, Clone)]
pub struct TurboQuantLayerConfig {
    pub head_dim: usize,
    pub num_kv_heads: usize,
}

/// TurboQuant-specific configuration extracted from MetalConfig + model params.
#[derive(Debug, Clone)]
pub struct TurboQuantMetalConfig {
    /// Quantization bit width.
    pub n_bits: u8,
    /// Number of key-value attention heads.
    pub num_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Rotation seed for Hadamard transform.
    pub rotation_seed: u64,
    /// Optional outlier channel configuration.
    pub outlier: Option<OutlierConfig>,
    /// CLA anchor layers. None = all layers are anchors (standard behavior).
    pub anchor_layers: Option<Vec<usize>>,
    /// Per-layer sliding window sizes. `0` = full attention.
    /// When non-empty, SWA layers allocate smaller ring buffers.
    /// Length must equal `num_layers` (or be empty for no SWA).
    pub window_sizes: Vec<usize>,
    /// Per-layer configs for heterogeneous head dims (Gemma 4).
    /// When empty, all layers use the global head_dim/num_kv_heads.
    pub layer_configs: Vec<TurboQuantLayerConfig>,
}

fn create_f32_buffer(
    device: &MetalDevice,
    data: &[f32],
) -> Result<ironmill_metal_sys::MetalBuffer, MetalError> {
    let bytes: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    device
        .create_buffer_with_data(&bytes, ironmill_metal_sys::StorageMode::Shared)
        .map_err(MetalError::Metal)
}

fn create_u32_buffer(
    device: &MetalDevice,
    data: &[u32],
) -> Result<ironmill_metal_sys::MetalBuffer, MetalError> {
    let bytes: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    device
        .create_buffer_with_data(&bytes, ironmill_metal_sys::StorageMode::Shared)
        .map_err(MetalError::Metal)
}

impl MetalTurboQuantModel {
    /// Initialize TurboQuant with rotation signs and quantization scales.
    pub fn new(device: &MetalDevice, config: TurboQuantMetalConfig) -> Result<Self, MetalError> {
        if !config.head_dim.is_power_of_two() {
            return Err(MetalError::Config(
                "head_dim must be a power of two for Walsh-Hadamard butterfly".to_string(),
            ));
        }

        let sign_bytes = generate_rotation_signs(config.head_dim, config.rotation_seed);
        let rotation_signs = device
            .create_buffer_with_data(&sign_bytes, ironmill_metal_sys::StorageMode::Shared)
            .map_err(MetalError::Metal)?;

        // Algorithm 2 (TurboQuant_prod): K cache uses (b-1)-bit MSE codebook
        // + 1-bit QJL sign packed into the top bit of each element = b bits
        // total.  V cache uses the full b-bit MSE codebook (Algorithm 1).
        assert!(
            config.n_bits >= 2,
            "n_bits must be >= 2 for Algorithm 2 (need 1 codebook bit + 1 QJL bit)"
        );
        let k_bits = config.n_bits - 1;
        let (k_levels, k_bounds) = codebook::lloyd_max_gaussian(config.head_dim, k_bits);
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
            .map_err(MetalError::Metal)?;

        // Outlier channel state
        let outlier = if let Some(ref outlier_cfg) = config.outlier {
            Some(Self::init_outlier(device, &config, outlier_cfg)?)
        } else {
            None
        };

        // Per-unique-head_dim codebook sets for heterogeneous models.
        // Collect unique head_dims that differ from the default.
        let dim_codebooks = Self::init_dim_codebooks(device, &config)?;

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
            dim_codebooks,
        })
    }

    /// Build codebook sets for each unique non-default head_dim in `layer_configs`.
    fn init_dim_codebooks(
        device: &MetalDevice,
        config: &TurboQuantMetalConfig,
    ) -> Result<Vec<DimCodebooks>, MetalError> {
        if config.layer_configs.is_empty() {
            return Ok(Vec::new());
        }

        // Collect unique head_dims that differ from the default.
        let mut unique_dims: Vec<usize> = config
            .layer_configs
            .iter()
            .map(|lc| lc.head_dim)
            .filter(|&hd| hd != config.head_dim)
            .collect();
        unique_dims.sort_unstable();
        unique_dims.dedup();

        let k_bits = config.n_bits - 1;
        let mut codebooks = Vec::with_capacity(unique_dims.len());

        for &hd in &unique_dims {
            if !hd.is_power_of_two() {
                return Err(MetalError::Config(format!(
                    "per-layer head_dim {} must be a power of two for Walsh-Hadamard butterfly",
                    hd
                )));
            }

            let sign_bytes = generate_rotation_signs(hd, config.rotation_seed);
            let rotation_signs = device
                .create_buffer_with_data(&sign_bytes, ironmill_metal_sys::StorageMode::Shared)
                .map_err(MetalError::Metal)?;

            let (kl, kb) = codebook::lloyd_max_gaussian(hd, k_bits);
            let k_n_levels = kl.len() as u32;
            let k_codebook_buf = create_f32_buffer(device, &kl)?;
            let k_boundaries_buf = create_f32_buffer(device, &kb)?;

            let (vl, vb) = codebook::lloyd_max_gaussian(hd, config.n_bits);
            let v_n_levels = vl.len() as u32;
            let v_codebook_buf = create_f32_buffer(device, &vl)?;
            let v_boundaries_buf = create_f32_buffer(device, &vb)?;

            let qjl_bytes = generate_qjl_matrix(hd, config.rotation_seed + 1);
            let qjl_matrix = device
                .create_buffer_with_data(&qjl_bytes, ironmill_metal_sys::StorageMode::Shared)
                .map_err(MetalError::Metal)?;

            codebooks.push(DimCodebooks {
                rotation_signs,
                k_codebook_buf,
                k_boundaries_buf,
                k_n_levels,
                v_codebook_buf,
                v_boundaries_buf,
                v_n_levels,
                qjl_matrix,
                head_dim: hd,
            });
        }

        Ok(codebooks)
    }

    /// Look up the codebooks for a given layer. Returns `None` when the
    /// default (global) codebooks should be used (layer uses `config.head_dim`).
    pub fn codebooks_for_layer(&self, layer_idx: usize) -> Option<&DimCodebooks> {
        let lc = self.config.layer_configs.get(layer_idx)?;
        if lc.head_dim == self.config.head_dim {
            return None;
        }
        self.dim_codebooks
            .iter()
            .find(|dc| dc.head_dim == lc.head_dim)
    }

    /// Initialize outlier channel state: independent rotation signs and
    /// codebooks for outlier and non-outlier channel groups.
    fn init_outlier(
        device: &MetalDevice,
        config: &TurboQuantMetalConfig,
        outlier_cfg: &OutlierConfig,
    ) -> Result<OutlierState, MetalError> {
        assert!(
            outlier_cfg.outlier_bits >= 2,
            "outlier_bits must be >= 2 (K cache uses b-1 bits when b < 4, minimum 1-bit codebook)"
        );
        assert!(
            outlier_cfg.non_outlier_bits >= 2,
            "non_outlier_bits must be >= 2 (K cache uses b-1 bits when b < 4, minimum 1-bit codebook)"
        );

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
            .map_err(MetalError::Metal)?;

        let non_outlier_sign_bytes =
            generate_rotation_signs(d_non_padded, config.rotation_seed + 200);
        let non_outlier_rotation_signs = device
            .create_buffer_with_data(
                &non_outlier_sign_bytes,
                ironmill_metal_sys::StorageMode::Shared,
            )
            .map_err(MetalError::Metal)?;

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

        // K codebooks: Algorithm 2 always uses (b-1)-bit + 1-bit QJL
        let ko_bits = outlier_cfg.outlier_bits - 1;
        let (ko_levels, ko_bounds) = codebook::lloyd_max_gaussian(d_outlier_padded, ko_bits);
        let k_outlier_n_levels = ko_levels.len() as u32;
        let k_outlier_codebook = create_f32_buffer(device, &ko_levels)?;
        let k_outlier_boundaries = create_f32_buffer(device, &ko_bounds)?;

        let kn_bits = outlier_cfg.non_outlier_bits - 1;
        let (kn_levels, kn_bounds) = codebook::lloyd_max_gaussian(d_non_padded, kn_bits);
        let k_non_outlier_n_levels = kn_levels.len() as u32;
        let k_non_outlier_codebook = create_f32_buffer(device, &kn_levels)?;
        let k_non_outlier_boundaries = create_f32_buffer(device, &kn_bounds)?;

        // QJL projection matrices (independent seeds)
        let o_qjl_bytes = generate_qjl_matrix(d_outlier_padded, config.rotation_seed + 300);
        let outlier_qjl_matrix = device
            .create_buffer_with_data(&o_qjl_bytes, ironmill_metal_sys::StorageMode::Shared)
            .map_err(MetalError::Metal)?;

        let n_qjl_bytes = generate_qjl_matrix(d_non_padded, config.rotation_seed + 400);
        let non_outlier_qjl_matrix = device
            .create_buffer_with_data(&n_qjl_bytes, ironmill_metal_sys::StorageMode::Shared)
            .map_err(MetalError::Metal)?;

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
