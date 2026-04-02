//! MLX-resident quantized KV cache for TurboQuant inference.
//!
//! Mirrors [`crate::gpu::turboquant::cache::GpuKvCache`] but stores all
//! data as [`MlxArray`]s and dispatches cache-write / attention kernels
//! via [`metal_kernel`].

use ironmill_mlx_sys::{MlxArray, MlxDtype, MlxStream, metal_kernel};

use super::{MlxOutlierCache, MlxTurboQuantModel};
use crate::mlx::error::MlxError;

/// MLX-resident quantized KV cache for TurboQuant inference.
///
/// Each layer has separate K and V quantized cache arrays plus per-head
/// per-position dequantization scale arrays. Optional outlier split and
/// QJL correction arrays are allocated when enabled.
pub struct MlxKvCache {
    /// K cache per layer: `[num_kv_heads × max_seq_len × elements_per_pos]` Uint8.
    k_caches: Vec<MlxArray>,
    /// V cache per layer.
    v_caches: Vec<MlxArray>,
    /// Per-head per-position K dequantization scales per layer: `[num_kv_heads × max_seq_len]` Float32.
    k_scales: Vec<MlxArray>,
    /// Per-head per-position V dequantization scales per layer.
    v_scales: Vec<MlxArray>,
    /// Rotation sign vector `[head_dim]` Float32 (±1.0).
    rotation_signs: MlxArray,

    // ── Outlier split (optional) ──
    outlier: Option<MlxOutlierCache>,

    // ── QJL correction (optional) ──
    #[allow(dead_code)]
    qjl_matrix: Option<MlxArray>,
    /// Packed sign bits of QJL-projected residuals per K position.
    k_qjl_signs: Option<Vec<MlxArray>>,
    /// L2 norm of quantization residual per K position.
    k_r_norms: Option<Vec<MlxArray>>,

    // ── Dimensions ──
    num_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
    n_bits: u8,
    #[allow(dead_code)]
    num_layers: usize,
    seq_pos: usize,
}

/// Helper: create a zero-filled MlxArray with the given total byte count,
/// stored as Uint8. We use this for raw cache buffers.
fn zeros_uint8(size: usize, stream: &MlxStream) -> Result<MlxArray, MlxError> {
    let data = vec![0u8; size];
    Ok(MlxArray::from_data_copy(
        &data,
        &[size],
        MlxDtype::Uint8,
        stream,
    )?)
}

/// Helper: create a zero-filled Float32 array of `count` elements.
fn zeros_f32(count: usize, stream: &MlxStream) -> Result<MlxArray, MlxError> {
    let data = vec![0u8; count * 4];
    Ok(MlxArray::from_data_copy(
        &data,
        &[count],
        MlxDtype::Float32,
        stream,
    )?)
}

impl MlxKvCache {
    /// Allocate KV cache arrays for all layers.
    pub fn new(
        model: &MlxTurboQuantModel,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        num_layers: usize,
        stream: &MlxStream,
    ) -> Result<Self, MlxError> {
        let n_bits = model.n_bits;
        let elements_per_pos = if n_bits == 4 { head_dim / 2 } else { head_dim };
        let cache_size = num_kv_heads * max_seq_len * elements_per_pos;
        let scale_count = num_kv_heads * max_seq_len;

        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        let mut k_scales = Vec::with_capacity(num_layers);
        let mut v_scales = Vec::with_capacity(num_layers);

        // QJL arrays
        let qjl_signs_size = num_kv_heads * max_seq_len * (head_dim / 8);
        let mut k_qjl_signs = Vec::with_capacity(num_layers);
        let mut k_r_norms = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            k_caches.push(zeros_uint8(cache_size, stream)?);
            v_caches.push(zeros_uint8(cache_size, stream)?);
            k_scales.push(zeros_f32(scale_count, stream)?);
            v_scales.push(zeros_f32(scale_count, stream)?);
            k_qjl_signs.push(zeros_uint8(qjl_signs_size.max(1), stream)?);
            k_r_norms.push(zeros_f32(scale_count, stream)?);
        }

        // Outlier cache allocation
        let outlier = if let Some(ref outlier_cfg) = model.outlier_config {
            let d_o_padded = outlier_cfg.outlier_channels.len().next_power_of_two();
            let d_n = head_dim - outlier_cfg.outlier_channels.len();
            let d_n_padded = d_n.next_power_of_two();
            let o_cache_size = num_kv_heads * max_seq_len * (d_o_padded / 2);
            let n_cache_size = num_kv_heads * max_seq_len * (d_n_padded / 2);

            let mut k_outlier = Vec::with_capacity(num_layers);
            let mut v_outlier = Vec::with_capacity(num_layers);
            let mut k_non_outlier = Vec::with_capacity(num_layers);
            let mut v_non_outlier = Vec::with_capacity(num_layers);
            let mut k_outlier_scales = Vec::with_capacity(num_layers);
            let mut v_outlier_scales = Vec::with_capacity(num_layers);
            let mut k_non_outlier_scales = Vec::with_capacity(num_layers);
            let mut v_non_outlier_scales = Vec::with_capacity(num_layers);
            let mut k_outlier_r_norms = Vec::with_capacity(num_layers);
            let mut k_non_outlier_r_norms = Vec::with_capacity(num_layers);

            for _ in 0..num_layers {
                k_outlier.push(zeros_uint8(o_cache_size.max(1), stream)?);
                v_outlier.push(zeros_uint8(o_cache_size.max(1), stream)?);
                k_non_outlier.push(zeros_uint8(n_cache_size.max(1), stream)?);
                v_non_outlier.push(zeros_uint8(n_cache_size.max(1), stream)?);
                k_outlier_scales.push(zeros_f32(scale_count, stream)?);
                v_outlier_scales.push(zeros_f32(scale_count, stream)?);
                k_non_outlier_scales.push(zeros_f32(scale_count, stream)?);
                v_non_outlier_scales.push(zeros_f32(scale_count, stream)?);
                k_outlier_r_norms.push(zeros_f32(scale_count, stream)?);
                k_non_outlier_r_norms.push(zeros_f32(scale_count, stream)?);
            }

            Some(MlxOutlierCache {
                k_outlier_caches: k_outlier,
                v_outlier_caches: v_outlier,
                k_non_outlier_caches: k_non_outlier,
                v_non_outlier_caches: v_non_outlier,
                k_outlier_scales,
                v_outlier_scales,
                k_non_outlier_scales,
                v_non_outlier_scales,
                k_outlier_r_norms,
                k_non_outlier_r_norms,
            })
        } else {
            None
        };

        Ok(Self {
            k_caches,
            v_caches,
            k_scales,
            v_scales,
            rotation_signs: model.rotation_signs.clone(),
            outlier,
            qjl_matrix: model.qjl_matrix.clone(),
            k_qjl_signs: Some(k_qjl_signs),
            k_r_norms: Some(k_r_norms),
            num_kv_heads,
            max_seq_len,
            head_dim,
            n_bits,
            num_layers,
            seq_pos: 0,
        })
    }

    /// Dispatch the TurboQuant cache write kernel for one layer.
    ///
    /// `kv_proj` is the K or V projection output `[num_kv_heads × head_dim]` FP16.
    /// `is_k_cache` selects K (true) or V (false) cache.
    ///
    /// Returns a dummy output array whose eval() materialises the cache write.
    pub fn write_kv(
        &self,
        layer: usize,
        kv_proj: &MlxArray,
        is_k_cache: bool,
        model: &MlxTurboQuantModel,
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        let (cache, scales) = if is_k_cache {
            (&self.k_caches[layer], &self.k_scales[layer])
        } else {
            (&self.v_caches[layer], &self.v_scales[layer])
        };

        // Pack params into a uint32 array
        let params_data: Vec<u32> = vec![
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.max_seq_len as u32,
            self.seq_pos as u32,
            self.n_bits as u32,
            if is_k_cache {
                model.k_codebook.len() as u32
            } else {
                model.v_codebook.len() as u32
            },
            if is_k_cache { 1 } else { 0 },
        ];
        let params_bytes: Vec<u8> = params_data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let params_arr = MlxArray::from_data_copy(
            &params_bytes,
            &[params_data.len()],
            MlxDtype::Uint32,
            stream,
        )?;

        let qjl = model
            .qjl_matrix
            .as_ref()
            .expect("qjl_matrix must be set for TurboQuant");
        let qjl_signs = &self.k_qjl_signs.as_ref().unwrap()[layer];
        let r_norms = &self.k_r_norms.as_ref().unwrap()[layer];

        let (codebook_arr, boundaries_arr) = if is_k_cache {
            (&model.k_codebook_arr, &model.k_boundaries_arr)
        } else {
            (&model.v_codebook_arr, &model.v_boundaries_arr)
        };

        let inputs: Vec<&MlxArray> = vec![
            kv_proj,              // 0
            &self.rotation_signs, // 1
            cache,                // 2
            scales,               // 3
            codebook_arr,         // 4
            boundaries_arr,       // 5
            qjl,                  // 6
            qjl_signs,            // 7
            r_norms,              // 8
            &params_arr,          // 9
        ];

        let result = metal_kernel(
            "turboquant_cache_write",
            &inputs,
            &[],
            super::kernels::TURBOQUANT_CACHE_WRITE,
            [self.num_kv_heads, 1, 1],
            [self.head_dim.min(256), 1, 1],
            &[&[1]], // dummy output shape
            &[MlxDtype::Float32],
            stream,
        )?;

        result
            .into_iter()
            .next()
            .ok_or_else(|| MlxError::WeightLoading("cache write returned no outputs".into()))
    }

    /// Dispatch the TurboQuant attention kernel for one layer.
    ///
    /// `q` is the query tensor `[num_heads × head_dim]` FP16 (single token decode).
    /// Returns the attention output `[num_heads × head_dim]` FP16.
    pub fn attend(
        &self,
        layer: usize,
        q: &MlxArray,
        num_heads: usize,
        model: &MlxTurboQuantModel,
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        let params_data: Vec<u32> = vec![
            num_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.max_seq_len as u32,
            self.seq_pos as u32, // seq_len = number of cached positions
            self.n_bits as u32,
        ];
        let params_bytes: Vec<u8> = params_data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let params_arr = MlxArray::from_data_copy(
            &params_bytes,
            &[params_data.len()],
            MlxDtype::Uint32,
            stream,
        )?;

        let qjl = model
            .qjl_matrix
            .as_ref()
            .expect("qjl_matrix must be set for TurboQuant");
        let r_norms = &self.k_r_norms.as_ref().unwrap()[layer];

        let inputs: Vec<&MlxArray> = vec![
            q,                     // 0
            &self.k_caches[layer], // 1
            &self.v_caches[layer], // 2
            &self.rotation_signs,  // 3
            &self.k_scales[layer], // 4
            &self.v_scales[layer], // 5
            &model.k_codebook_arr, // 6: k_codebook ((b-1)-bit for QJL)
            &model.v_codebook_arr, // 7: v_codebook (b-bit for MSE)
            qjl,                   // 8
            r_norms,               // 9
            &params_arr,           // 10
        ];

        let output_size = num_heads * self.head_dim;
        let result = metal_kernel(
            "turboquant_attention",
            &inputs,
            &[],
            super::kernels::TURBOQUANT_ATTENTION,
            [num_heads, 1, 1],
            [self.head_dim.min(256), 1, 1],
            &[&[output_size]],
            &[MlxDtype::Float16],
            stream,
        )?;

        result
            .into_iter()
            .next()
            .ok_or_else(|| MlxError::WeightLoading("attention returned no outputs".into()))
    }

    /// Dispatch the outlier cache write kernel for one layer (K only).
    ///
    /// Uses K codebooks ((b-1)-bit + QJL) and writes residual norms.
    pub fn write_k_outlier(
        &self,
        layer: usize,
        kv_proj: &MlxArray,
        model: &MlxTurboQuantModel,
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        let outlier_cfg = model
            .outlier_config
            .as_ref()
            .expect("outlier_config must be set for outlier cache write");
        let outlier_cache = self
            .outlier
            .as_ref()
            .expect("outlier cache must be allocated");
        let outlier_model = model
            .outlier_model
            .as_ref()
            .expect("outlier_model must be set");

        let n_outlier = outlier_cfg.outlier_channels.len();
        let d_outlier_padded = n_outlier.next_power_of_two();
        let d_non = self.head_dim - n_outlier;
        let d_non_padded = d_non.next_power_of_two();

        let params_data: Vec<u32> = vec![
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.max_seq_len as u32,
            self.seq_pos as u32,
            n_outlier as u32,
            d_outlier_padded as u32,
            d_non_padded as u32,
            outlier_model.k_outlier_n_levels as u32,
            outlier_model.k_non_outlier_n_levels as u32,
            1, // is_k_cache
        ];
        let params_bytes: Vec<u8> = params_data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let params_arr = MlxArray::from_data_copy(
            &params_bytes,
            &[params_data.len()],
            MlxDtype::Uint32,
            stream,
        )?;

        let inputs: Vec<&MlxArray> = vec![
            kv_proj,                                     // 0
            &outlier_model.channel_indices,              // 1
            &outlier_cache.k_outlier_caches[layer],      // 2
            &outlier_cache.k_non_outlier_caches[layer],  // 3
            &outlier_model.outlier_rotation_signs,       // 4
            &outlier_model.non_outlier_rotation_signs,   // 5
            &outlier_model.k_outlier_codebook,           // 6
            &outlier_model.k_outlier_boundaries,         // 7
            &outlier_model.k_non_outlier_codebook,       // 8
            &outlier_model.k_non_outlier_boundaries,     // 9
            &outlier_cache.k_outlier_scales[layer],      // 10
            &outlier_cache.k_non_outlier_scales[layer],  // 11
            &outlier_model.outlier_qjl_matrix,           // 12
            &outlier_model.non_outlier_qjl_matrix,       // 13
            &outlier_cache.k_outlier_r_norms[layer],     // 14
            &outlier_cache.k_non_outlier_r_norms[layer], // 15
            &params_arr,                                 // 16
        ];

        let tg_size = d_outlier_padded.max(d_non_padded).min(256);

        let result = metal_kernel(
            "turboquant_outlier_cache_write",
            &inputs,
            &[],
            super::kernels::TURBOQUANT_OUTLIER_CACHE_WRITE,
            [self.num_kv_heads, 1, 1],
            [tg_size, 1, 1],
            &[&[1]],
            &[MlxDtype::Float32],
            stream,
        )?;

        result.into_iter().next().ok_or_else(|| {
            MlxError::WeightLoading("outlier K cache write returned no outputs".into())
        })
    }

    /// Dispatch the outlier cache write kernel for one layer (V only).
    ///
    /// Uses V codebooks (b-bit, no QJL).
    pub fn write_v_outlier(
        &self,
        layer: usize,
        kv_proj: &MlxArray,
        model: &MlxTurboQuantModel,
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        let outlier_cfg = model
            .outlier_config
            .as_ref()
            .expect("outlier_config must be set for outlier cache write");
        let outlier_cache = self
            .outlier
            .as_ref()
            .expect("outlier cache must be allocated");
        let outlier_model = model
            .outlier_model
            .as_ref()
            .expect("outlier_model must be set");

        let n_outlier = outlier_cfg.outlier_channels.len();
        let d_outlier_padded = n_outlier.next_power_of_two();
        let d_non = self.head_dim - n_outlier;
        let d_non_padded = d_non.next_power_of_two();

        let params_data: Vec<u32> = vec![
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.max_seq_len as u32,
            self.seq_pos as u32,
            n_outlier as u32,
            d_outlier_padded as u32,
            d_non_padded as u32,
            outlier_model.outlier_n_levels as u32,
            outlier_model.non_outlier_n_levels as u32,
            0, // is_k_cache
        ];
        let params_bytes: Vec<u8> = params_data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let params_arr = MlxArray::from_data_copy(
            &params_bytes,
            &[params_data.len()],
            MlxDtype::Uint32,
            stream,
        )?;

        let inputs: Vec<&MlxArray> = vec![
            kv_proj,                                     // 0
            &outlier_model.channel_indices,              // 1
            &outlier_cache.v_outlier_caches[layer],      // 2
            &outlier_cache.v_non_outlier_caches[layer],  // 3
            &outlier_model.outlier_rotation_signs,       // 4
            &outlier_model.non_outlier_rotation_signs,   // 5
            &outlier_model.outlier_codebook,             // 6
            &outlier_model.outlier_boundaries,           // 7
            &outlier_model.non_outlier_codebook,         // 8
            &outlier_model.non_outlier_boundaries,       // 9
            &outlier_cache.v_outlier_scales[layer],      // 10
            &outlier_cache.v_non_outlier_scales[layer],  // 11
            &outlier_model.outlier_qjl_matrix,           // 12
            &outlier_model.non_outlier_qjl_matrix,       // 13
            &outlier_cache.k_outlier_r_norms[layer],     // 14 (bound but unused)
            &outlier_cache.k_non_outlier_r_norms[layer], // 15 (bound but unused)
            &params_arr,                                 // 16
        ];

        let tg_size = d_outlier_padded.max(d_non_padded).min(256);

        let result = metal_kernel(
            "turboquant_outlier_cache_write",
            &inputs,
            &[],
            super::kernels::TURBOQUANT_OUTLIER_CACHE_WRITE,
            [self.num_kv_heads, 1, 1],
            [tg_size, 1, 1],
            &[&[1]],
            &[MlxDtype::Float32],
            stream,
        )?;

        result.into_iter().next().ok_or_else(|| {
            MlxError::WeightLoading("outlier V cache write returned no outputs".into())
        })
    }

    /// Dispatch the outlier attention kernel for one layer.
    pub fn attend_outlier(
        &self,
        layer: usize,
        q: &MlxArray,
        num_heads: usize,
        model: &MlxTurboQuantModel,
        stream: &MlxStream,
    ) -> Result<MlxArray, MlxError> {
        let outlier_cfg = model
            .outlier_config
            .as_ref()
            .expect("outlier_config must be set");
        let outlier_cache = self
            .outlier
            .as_ref()
            .expect("outlier cache must be allocated");
        let outlier_model = model
            .outlier_model
            .as_ref()
            .expect("outlier_model must be set");

        let n_outlier = outlier_cfg.outlier_channels.len();
        let d_outlier_padded = n_outlier.next_power_of_two();
        let d_non = self.head_dim - n_outlier;
        let d_non_padded = d_non.next_power_of_two();

        let params_data: Vec<u32> = vec![
            num_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.max_seq_len as u32,
            self.seq_pos as u32,
            n_outlier as u32,
            d_outlier_padded as u32,
            d_non_padded as u32,
        ];
        let params_bytes: Vec<u8> = params_data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let params_arr = MlxArray::from_data_copy(
            &params_bytes,
            &[params_data.len()],
            MlxDtype::Uint32,
            stream,
        )?;

        let inputs: Vec<&MlxArray> = vec![
            q,                                           // 0
            &outlier_cache.k_outlier_caches[layer],      // 1
            &outlier_cache.v_outlier_caches[layer],      // 2
            &outlier_cache.k_non_outlier_caches[layer],  // 3
            &outlier_cache.v_non_outlier_caches[layer],  // 4
            &outlier_model.channel_indices,              // 5
            &outlier_model.outlier_rotation_signs,       // 6
            &outlier_model.non_outlier_rotation_signs,   // 7
            &outlier_model.k_outlier_codebook,           // 8: K codebook
            &outlier_model.k_non_outlier_codebook,       // 9: K codebook
            &outlier_cache.k_outlier_scales[layer],      // 10
            &outlier_cache.v_outlier_scales[layer],      // 11
            &outlier_cache.k_non_outlier_scales[layer],  // 12
            &outlier_cache.v_non_outlier_scales[layer],  // 13
            &outlier_model.outlier_qjl_matrix,           // 14
            &outlier_model.non_outlier_qjl_matrix,       // 15
            &outlier_cache.k_outlier_r_norms[layer],     // 16
            &outlier_cache.k_non_outlier_r_norms[layer], // 17
            &outlier_model.outlier_codebook,             // 18: V codebook
            &outlier_model.non_outlier_codebook,         // 19: V codebook
            &params_arr,                                 // 20
        ];

        let output_size = num_heads * self.head_dim;
        let tg_size = d_outlier_padded.max(d_non_padded).min(256);

        let result = metal_kernel(
            "turboquant_outlier_attention",
            &inputs,
            &[],
            super::kernels::TURBOQUANT_OUTLIER_ATTENTION,
            [num_heads, 1, 1],
            [tg_size, 1, 1],
            &[&[output_size]],
            &[MlxDtype::Float16],
            stream,
        )?;

        result
            .into_iter()
            .next()
            .ok_or_else(|| MlxError::WeightLoading("outlier attention returned no outputs".into()))
    }

    /// Current sequence position.
    pub fn seq_pos(&self) -> usize {
        self.seq_pos
    }

    /// Advance sequence position by one token.
    pub fn advance(&mut self) {
        self.seq_pos += 1;
    }

    /// Advance by multiple positions (for prefill).
    pub fn advance_by(&mut self, count: usize) {
        self.seq_pos += count;
    }

    /// Current sequence length (number of cached tokens).
    pub fn seq_len(&self) -> usize {
        self.seq_pos
    }

    /// Reset cache for a new conversation.
    pub fn reset(&mut self) {
        self.seq_pos = 0;
    }
}
