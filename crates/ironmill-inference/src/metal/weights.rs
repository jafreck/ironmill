//! Weight loading from SafeTensors/GGUF into Metal buffers.

use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};
use mil_rs::ir::ScalarType;
use mil_rs::weights::{ModelConfig, QuantizationInfo, WeightProvider};

use super::dequant::{dequant_affine, dequant_lut_to_dense};
use super::error::MetalError;
use crate::weight_loading::{
    self, CpuDequant, LoadedLayer, WeightVisitor, dense_shape, dequant_tensor_to_dense,
};

/// A weight buffer that is either dense FP16 or packed quantized data.
#[non_exhaustive]
pub enum WeightBuffer {
    /// Dense FP16 buffer for MPS matmul, with an optional pre-packed blocked
    /// buffer for the custom matvec kernel (decode path, M=1).
    Dense {
        /// Original row-major [N, K] FP16 buffer (used by MPS for prefill).
        buf: MetalBuffer,
        /// Blocked [N/8, K/8, 8, 8] FP16 buffer for the custom matvec kernel.
        /// `None` if N or K is not a multiple of 8.
        packed: Option<MetalBuffer>,
    },
    /// Packed quantized buffer for custom kernel.
    Quantized(QuantizedWeight),
    /// INT4 affine-quantized weight kept packed on GPU.
    /// Dequantized to FP16 at inference time via a Metal compute kernel
    /// before feeding into MPS matmul.
    AffineQuantized(AffineQuantizedWeight),
}

impl WeightBuffer {
    /// Get the underlying buffer for MPS matmul (Dense only).
    /// Returns an error if called on a Quantized or AffineQuantized variant.
    pub fn as_dense(&self) -> anyhow::Result<&MetalBuffer> {
        match self {
            WeightBuffer::Dense { buf, .. } => Ok(buf),
            WeightBuffer::Quantized(_) | WeightBuffer::AffineQuantized(_) => Err(anyhow::anyhow!(
                "expected dense buffer, got quantized weight"
            )),
        }
    }

    /// Get the pre-packed blocked buffer for the custom matvec kernel.
    /// Returns `None` for Quantized/AffineQuantized weights or Dense weights
    /// where packing was not possible (dimensions not multiples of 8).
    pub fn packed_buf(&self) -> Option<&MetalBuffer> {
        match self {
            WeightBuffer::Dense { packed, .. } => packed.as_ref(),
            WeightBuffer::Quantized(_) | WeightBuffer::AffineQuantized(_) => None,
        }
    }
}

/// Packed quantized weight stored as separate Metal buffers for the custom
/// matmul kernel.
pub struct QuantizedWeight {
    /// Packed n-bit indices.
    pub indices: MetalBuffer,
    /// Reconstruction look-up table `[2^n_bits]`.
    pub lut: MetalBuffer,
    /// Per-row norms `[rows]`.
    pub norms: MetalBuffer,
    /// Bit-width of the palette indices (e.g. 2, 4).
    pub n_bits: u8,
    /// `(out_features, in_features)`.
    pub shape: (usize, usize),
}

/// Affine-quantized weight (INT4 or INT8) stored as packed bytes with
/// per-group scales and zero points on GPU. Dequantized inline during
/// matmul via fused Metal compute kernels.
pub struct AffineQuantizedWeight {
    /// Packed quantized data: INT4 = 2 values per byte, INT8 = 1 value per byte.
    pub data: MetalBuffer,
    /// Per-group FP16 scales.
    pub scales: MetalBuffer,
    /// Per-group FP16 zero points.
    pub zeros: MetalBuffer,
    /// Number of elements sharing one scale/zero pair.
    pub group_size: u32,
    /// Quantization bit width (4 or 8).
    pub bit_width: u8,
    /// `(out_features, in_features)` — logical element dimensions.
    pub shape: (usize, usize),
    /// Optional AWQ per-column channel scales [in_features] as FP16.
    /// When present, the kernel divides dequantized weights by these
    /// scales to compensate for activation-aware weight scaling.
    pub awq_scales: Option<MetalBuffer>,
}

/// Weights for a single transformer layer (type alias over shared layout).
pub type LayerWeights = LoadedLayer<MetalBuffer, WeightBuffer>;

/// All model weights loaded into Metal buffers, organized by layer.
pub struct MetalWeights {
    /// Embedding table [vocab_size × hidden_size] FP16.
    pub embedding: MetalBuffer,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm weight [hidden_size] FP16.
    pub final_norm: MetalBuffer,
    /// LM head weight [vocab_size × hidden_size] FP16.
    pub lm_head: MetalBuffer,
    /// Pre-packed LM head in blocked [N/8, K/8, 8, 8] FP16 (for custom matvec).
    pub lm_head_packed: Option<MetalBuffer>,
    /// Model configuration extracted from weight metadata.
    pub config: ModelConfig,
}

/// Backend-specific visitor that loads tensors into Metal buffers.
struct MetalVisitor<'a> {
    device: &'a MetalDevice,
    force_cpu_dequant: bool,
}

impl WeightVisitor for MetalVisitor<'_> {
    type Dense = MetalBuffer;
    type Weight = WeightBuffer;
    type Error = MetalError;

    fn load_dense(
        &self,
        provider: &dyn WeightProvider,
        name: &str,
    ) -> Result<MetalBuffer, MetalError> {
        load_dense_buffer(self.device, provider, name)
    }

    fn load_weight(
        &self,
        provider: &dyn WeightProvider,
        name: &str,
    ) -> Result<WeightBuffer, MetalError> {
        load_weight_buffer(self.device, provider, name, self.force_cpu_dequant)
    }
}

impl MetalWeights {
    /// Load model weights from a [`WeightProvider`] into Metal buffers.
    ///
    /// Weights are loaded into shared-mode buffers for CPU→GPU transfer.
    pub fn load(
        device: &MetalDevice,
        provider: &dyn WeightProvider,
        force_cpu_dequant: bool,
    ) -> Result<Self, MetalError> {
        let visitor = MetalVisitor {
            device,
            force_cpu_dequant,
        };
        let core = weight_loading::load_model_weights(&visitor, provider)?;
        let config = core.config;

        let lm_head = if config.tie_word_embeddings {
            load_dense_buffer(device, provider, "model.embed_tokens.weight")?
        } else {
            load_dense_buffer(device, provider, "lm_head.weight")?
        };

        let lm_head_packed =
            pack_weight_blocked(device, &lm_head, config.vocab_size, config.hidden_size)?;

        Ok(Self {
            embedding: core.embedding,
            layers: core.layers,
            final_norm: core.final_norm,
            lm_head,
            lm_head_packed,
            config,
        })
    }
}

/// Pack a row-major [N, K] FP16 weight buffer into blocked [N/8, K/8, 8, 8] format.
///
/// Returns `None` if N or K is not a multiple of 8.
fn pack_weight_blocked(
    device: &MetalDevice,
    original: &MetalBuffer,
    n: usize,
    k: usize,
) -> Result<Option<MetalBuffer>, MetalError> {
    if n % 8 != 0 || k % 8 != 0 {
        return Ok(None);
    }

    let total_bytes = n * k * 2; // FP16
    let mut src = vec![0u8; total_bytes];
    original
        .read_bytes(&mut src, 0)
        .map_err(MetalError::Metal)?;

    let mut dst = vec![0u8; total_bytes];
    let n_blocks = n / 8;
    let k_blocks = k / 8;

    for nb in 0..n_blocks {
        for kb in 0..k_blocks {
            for ri in 0..8u32 {
                for ki in 0..8u32 {
                    let src_row = nb * 8 + ri as usize;
                    let src_col = kb * 8 + ki as usize;
                    let src_offset = (src_row * k + src_col) * 2;

                    let dst_idx = ((nb * k_blocks + kb) * 8 + ri as usize) * 8 + ki as usize;
                    let dst_offset = dst_idx * 2;

                    dst[dst_offset] = src[src_offset];
                    dst[dst_offset + 1] = src[src_offset + 1];
                }
            }
        }
    }

    let buf = device
        .create_buffer_with_data(&dst, StorageMode::Shared)
        .map_err(MetalError::Metal)?;
    Ok(Some(buf))
}

/// CPU dequant operations for the Metal backend.
struct MetalDequantOps;

impl CpuDequant for MetalDequantOps {
    fn dequant_lut(
        indices: &[u8],
        lut: &[u8],
        lut_dtype: ScalarType,
        original_shape: &[usize],
        n_bits: u8,
        row_norms: &[u8],
        norms_dtype: ScalarType,
        polar_quant_seed: Option<u64>,
    ) -> anyhow::Result<Vec<u8>> {
        dequant_lut_to_dense(
            indices,
            lut,
            lut_dtype,
            original_shape,
            n_bits,
            row_norms,
            norms_dtype,
            polar_quant_seed,
        )
    }

    fn dequant_affine(
        data: &[u8],
        scale: &[u8],
        zero_point: &[u8],
        scale_dtype: ScalarType,
        zero_point_dtype: ScalarType,
        axis: Option<usize>,
        shape: &[usize],
        bit_width: u8,
        group_size: Option<usize>,
    ) -> anyhow::Result<Vec<u8>> {
        super::dequant::dequant_affine(
            data,
            scale,
            zero_point,
            scale_dtype,
            zero_point_dtype,
            axis,
            shape,
            bit_width,
            group_size,
        )
    }
}

/// Load a single weight tensor into a [`WeightBuffer`].
///
/// Returns [`WeightBuffer::Quantized`] for LUT-quantized tensors, keeping
/// packed indices, LUT, and norms as separate Metal buffers. Falls back to
/// dense FP16 for unquantized or affine-quantized tensors.
/// Dense weights are also pre-packed into blocked format for the custom
/// matvec kernel when dimensions are multiples of 8.
fn load_weight_buffer(
    device: &MetalDevice,
    provider: &dyn WeightProvider,
    name: &str,
    force_cpu_dequant: bool,
) -> Result<WeightBuffer, MetalError> {
    let tensor = provider
        .tensor(name)
        .map_err(|e| MetalError::WeightLoading(format!("{name}: {e}")))?;
    match &tensor.quant_info {
        QuantizationInfo::None => {
            let buf = device
                .create_buffer_with_data(&tensor.data, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            let (n, k) = dense_shape(&tensor.shape);
            let packed = pack_weight_blocked(device, &buf, n, k)?;
            Ok(WeightBuffer::Dense { buf, packed })
        }
        QuantizationInfo::LutToDense {
            lut,
            lut_dtype,
            indices,
            original_shape,
            n_bits,
            row_norms,
            norms_dtype,
            polar_quant_seed,
            quip_sharp_seed,
        } => {
            // QuIP# tensors always CPU-dequant for now (GPU kernel dispatch
            // is wired but inference.rs routing is not yet implemented).
            if quip_sharp_seed.is_some() || force_cpu_dequant {
                let data = if let Some(seed) = quip_sharp_seed {
                    super::dequant::dequant_quip_sharp(
                        indices,
                        lut,
                        *lut_dtype,
                        original_shape,
                        row_norms,
                        *norms_dtype,
                        *seed,
                    )?
                } else {
                    dequant_lut_to_dense(
                        indices,
                        lut,
                        *lut_dtype,
                        original_shape,
                        *n_bits,
                        row_norms,
                        *norms_dtype,
                        *polar_quant_seed,
                    )?
                };
                let buf = device
                    .create_buffer_with_data(&data, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                let (n, k) = dense_shape(original_shape);
                let packed = pack_weight_blocked(device, &buf, n, k)?;
                Ok(WeightBuffer::Dense { buf, packed })
            } else {
                let rows = original_shape[0];
                let cols = if original_shape.len() > 1 {
                    original_shape[1]
                } else {
                    1
                };
                let repacked = pack_quantized_blocked(indices, rows, cols, *n_bits as usize);
                let indices_buf = device
                    .create_buffer_with_data(&repacked, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                let lut_buf = device
                    .create_buffer_with_data(lut, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                let norms_buf = device
                    .create_buffer_with_data(row_norms, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                Ok(WeightBuffer::Quantized(QuantizedWeight {
                    indices: indices_buf,
                    lut: lut_buf,
                    norms: norms_buf,
                    n_bits: *n_bits,
                    shape: (rows, cols),
                }))
            }
        }
        QuantizationInfo::AffineDequantize {
            scale,
            zero_point,
            scale_dtype,
            zero_point_dtype,
            axis,
            bit_width,
            group_size,
            awq_scales,
            g_idx: _,
        } => {
            // INT4/INT8 with per-group quantization: keep packed on GPU and
            // dequantize inline during matmul via fused affine kernels.
            if (*bit_width == 4 || *bit_width == 8) && !force_cpu_dequant {
                let (n, k) = dense_shape(&tensor.shape);
                let total_elements = n * k;
                let gs = group_size.unwrap_or(total_elements);

                let repacked = pack_quantized_blocked(&tensor.data, n, k, *bit_width as usize);
                let data_buf = device
                    .create_buffer_with_data(&repacked, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;

                // Convert scales and zeros to FP16 bytes for the GPU.
                let scales_f16 = super::dequant::convert_params_to_f16(
                    scale,
                    *scale_dtype,
                    *axis,
                    &tensor.shape,
                    gs,
                )?;
                let zeros_f16 = super::dequant::convert_params_to_f16(
                    zero_point,
                    *zero_point_dtype,
                    *axis,
                    &tensor.shape,
                    gs,
                )?;

                let scales_buf = device
                    .create_buffer_with_data(&scales_f16, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                let zeros_buf = device
                    .create_buffer_with_data(&zeros_f16, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;

                // Upload AWQ per-column scales if present.
                let awq_buf = if let Some(awq_data) = awq_scales {
                    Some(
                        device
                            .create_buffer_with_data(awq_data, StorageMode::Shared)
                            .map_err(MetalError::Metal)?,
                    )
                } else {
                    None
                };

                return Ok(WeightBuffer::AffineQuantized(AffineQuantizedWeight {
                    data: data_buf,
                    scales: scales_buf,
                    zeros: zeros_buf,
                    group_size: gs as u32,
                    bit_width: *bit_width,
                    shape: (n, k),
                    awq_scales: awq_buf,
                }));
            }

            // Forced CPU dequant fallback.
            let data = dequant_affine(
                &tensor.data,
                scale,
                zero_point,
                *scale_dtype,
                *zero_point_dtype,
                *axis,
                &tensor.shape,
                *bit_width,
                *group_size,
            )?;
            let buf = device
                .create_buffer_with_data(&data, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            let (n, k) = dense_shape(&tensor.shape);
            let packed = pack_weight_blocked(device, &buf, n, k)?;
            Ok(WeightBuffer::Dense { buf, packed })
        }
        _ => Err(MetalError::WeightLoading(format!(
            "{name}: unsupported quantization format"
        ))),
    }
}

/// Load a single weight tensor into a dense FP16 [`MetalBuffer`].
///
/// Always dequantizes to FP16 — used for embeddings, norms, and lm_head
/// which are always consumed by MPS or element-wise kernels.
fn load_dense_buffer(
    device: &MetalDevice,
    provider: &dyn WeightProvider,
    name: &str,
) -> Result<MetalBuffer, MetalError> {
    let tensor = provider
        .tensor(name)
        .map_err(|e| MetalError::WeightLoading(format!("{name}: {e}")))?;
    let dense = dequant_tensor_to_dense::<MetalDequantOps>(&tensor)?;
    device
        .create_buffer_with_data(&dense.bytes, StorageMode::Shared)
        .map_err(MetalError::Metal)
}

/// Repack quantized weight bytes into blocked layout for simdgroup matmul.
///
/// Input layout:  row-major `[N, K_bytes_per_row]`
/// Output layout: `[N_blocks, K_blocks, TN_TILE, local_K_bytes]`
///
/// `N_blocks = ceil(N/64)`, `K_blocks = ceil(K/K_TILE)`
/// For INT4: `K_bytes_per_row = K/2`, `local_K_bytes = K_TILE/2 = 4`
/// For INT8: `K_bytes_per_row = K`,   `local_K_bytes = K_TILE = 8`
fn pack_quantized_blocked(data: &[u8], n: usize, k: usize, bit_width: usize) -> Vec<u8> {
    let k_tile: usize = 8;
    let n_tile: usize = 64;
    let n_blocks = n.div_ceil(n_tile);
    let k_blocks = k.div_ceil(k_tile);

    let elements_per_byte = 8 / bit_width; // 2 for INT4, 1 for INT8
    let bytes_per_row = k / elements_per_byte;
    let local_k_bytes = k_tile / elements_per_byte;
    let block_bytes = n_tile * local_k_bytes;

    let total_bytes = n_blocks * k_blocks * block_bytes;
    let mut out = vec![0u8; total_bytes];

    for nb in 0..n_blocks {
        for kb in 0..k_blocks {
            for n_local in 0..n_tile {
                let g_n = nb * n_tile + n_local;
                if g_n >= n {
                    continue;
                }
                for b in 0..local_k_bytes {
                    let src_byte = kb * local_k_bytes + b;
                    if src_byte >= bytes_per_row {
                        continue;
                    }
                    let src_offset = g_n * bytes_per_row + src_byte;
                    let dst_offset =
                        (nb * k_blocks + kb) * block_bytes + n_local * local_k_bytes + b;
                    out[dst_offset] = data[src_offset];
                }
            }
        }
    }
    out
}
