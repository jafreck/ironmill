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
    /// Dense FP16 buffer. Either a row-major buffer for gather/lookup access,
    /// or a pre-packed blocked buffer for the custom matmul kernel (or both
    /// when packing failed due to dimension constraints).
    Dense {
        /// Row-major [N, K] FP16 buffer. Used for embedding lookups and as
        /// fallback when blocked packing isn't possible. `None` when only
        /// the packed buffer is needed (matmul weights with valid dimensions).
        buf: Option<MetalBuffer>,
        /// Blocked [N/8, K/8, 8, 8] FP16 buffer for the custom matvec/matmul
        /// kernel. `None` if N or K is not a multiple of 8.
        packed: Option<MetalBuffer>,
    },
    /// Packed quantized buffer for custom kernel.
    Quantized(QuantizedWeight),
    /// INT4 affine-quantized weight kept packed on GPU.
    /// Dequantized to FP16 at inference time via a Metal compute kernel
    /// before feeding into MPS matmul.
    AffineQuantized(AffineQuantizedWeight),
    /// D2Quant dual-scale quantized weight kept packed on GPU.
    DualScaleQuantized(DualScaleQuantizedWeight),
}

impl WeightBuffer {
    /// Create an empty placeholder weight (no GPU memory).
    ///
    /// Used for GDN layer Q/K/V/O placeholders that are never dispatched.
    pub fn empty() -> Self {
        WeightBuffer::Dense {
            buf: None,
            packed: None,
        }
    }

    /// Returns `true` if this weight has no GPU buffers allocated.
    pub fn is_empty(&self) -> bool {
        matches!(
            self,
            WeightBuffer::Dense {
                buf: None,
                packed: None
            }
        )
    }

    /// Get the underlying row-major buffer (Dense only).
    /// Returns an error if the buffer is quantized or was dropped after packing.
    pub fn as_dense(&self) -> anyhow::Result<&MetalBuffer> {
        match self {
            WeightBuffer::Dense { buf: Some(b), .. } => Ok(b),
            WeightBuffer::Dense { buf: None, .. } => Err(anyhow::anyhow!(
                "dense row-major buffer was dropped after packing"
            )),
            WeightBuffer::Quantized(_)
            | WeightBuffer::AffineQuantized(_)
            | WeightBuffer::DualScaleQuantized(_) => Err(anyhow::anyhow!(
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
            WeightBuffer::Quantized(_)
            | WeightBuffer::AffineQuantized(_)
            | WeightBuffer::DualScaleQuantized(_) => None,
        }
    }

    /// Drop redundant buffers that are no longer needed after load-time
    /// transforms. Inference dispatches exclusively through the primary
    /// kernel path, so secondary layout copies are dead weight:
    ///
    /// - **Dense:** drops the row-major `buf` when a packed blocked copy exists.
    /// - **Other variants:** no-op (no redundant copies).
    pub fn compact(&mut self) {
        if let WeightBuffer::Dense {
            buf,
            packed: Some(_),
        } = self
        {
            *buf = None;
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
    /// Packed quantized data in blocked layout for matmul: [N/64, K/8, 64, BLK_K/2].
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

/// D2Quant dual-scale quantized weight stored as packed 3-bit data with
/// per-group dual-scale parameters on GPU. Dequantized inline during
/// matmul via fused Metal compute kernels.
pub struct DualScaleQuantizedWeight {
    /// Packed 3-bit (or 2-bit) quantized data, row-major.
    pub data: MetalBuffer,
    /// Per-group FP32 normal-partition scales.
    pub normal_scale: MetalBuffer,
    /// Per-group FP32 normal-partition zero points.
    pub normal_zero: MetalBuffer,
    /// Per-group FP32 outlier-partition scales.
    pub outlier_scale: MetalBuffer,
    /// Per-group FP32 outlier-partition zero points.
    pub outlier_zero: MetalBuffer,
    /// Packed outlier mask (1 bit per weight).
    pub outlier_mask: MetalBuffer,
    /// Group size for quantization parameters.
    pub group_size: u32,
    /// Quantization bit width (2 or 3).
    pub bit_width: u8,
    /// `(out_features, in_features)` — logical element dimensions.
    pub shape: (usize, usize),
}

/// Weights for a single transformer layer (type alias over shared layout).
pub type LayerWeights = LoadedLayer<MetalBuffer, WeightBuffer>;

/// All model weights loaded into Metal buffers, organized by layer.
pub struct MetalWeights {
    /// Embedding table [vocab_size × hidden_size] FP16 (for FP16 models).
    /// When INT4 quantized, this is used as a dequant scratch buffer and the
    /// actual data lives in `embedding_quantized`.
    pub embedding: MetalBuffer,
    /// INT4 quantized embedding table. When `Some`, the pipeline dispatches
    /// `affine_embedding_lookup_int4` to dequant on gather.
    pub embedding_quantized: Option<AffineQuantizedWeight>,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm weight [hidden_size] FP16.
    pub final_norm: MetalBuffer,
    /// LM head projection weight [vocab_size × hidden_size].
    /// May be dense FP16 (packed blocked), D2Quant, or other quantized format.
    pub lm_head: WeightBuffer,
    /// Model configuration extracted from weight metadata.
    pub config: ModelConfig,

    /// PLE embedding table `[vocab_size, num_layers * ple_hidden_size]` (Gemma 4).
    /// May be dense FP16 or D2Quant-compressed.
    pub ple_embed_tokens: Option<WeightBuffer>,
    /// PLE model projection weight (Gemma 4).
    pub ple_model_projection: Option<WeightBuffer>,
    /// PLE projection norm weight (Gemma 4).
    pub ple_projection_norm: Option<MetalBuffer>,
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
        load_weight_buffer(self.device, provider, name, self.force_cpu_dequant, true)
    }

    fn load_weight_for_gather(
        &self,
        provider: &dyn WeightProvider,
        name: &str,
    ) -> Result<WeightBuffer, MetalError> {
        load_weight_buffer(self.device, provider, name, self.force_cpu_dequant, false)
    }

    fn empty_weight(&self) -> WeightBuffer {
        WeightBuffer::empty()
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
        let mut core = weight_loading::load_model_weights(&visitor, provider)?;
        let config = core.config;

        // Qwen 3.5 centered RMSNorm: forward computes x * (1 + weight) / rms.
        // Weights are stored as residuals from 1.0 (near zero). Pre-add 1.0
        // at load time so the standard RMSNorm kernel (x * weight / rms) works.
        // Only applies to Qwen3_5RMSNorm (input/post-attn/final/QK norms).
        // Does NOT apply to GDN output gate norm (Qwen3_5RMSNormGated).
        if config.architecture == mil_rs::weights::Architecture::Qwen35 {
            for layer in &mut core.layers {
                offset_norm_weight(&layer.input_norm);
                offset_norm_weight(&layer.post_attn_norm);
                if let Some(ref qn) = layer.q_norm {
                    offset_norm_weight(qn);
                }
                if let Some(ref kn) = layer.k_norm {
                    offset_norm_weight(kn);
                }
            }
            offset_norm_weight(&core.final_norm);

            // Qwen 3.5 attn_output_gate: the q_proj weight [2*nh*hd, h] contains
            // interleaved Q and gate rows per head. Split into separate q_proj
            // [nh*hd, h] and attn_output_gate [nh*hd, h] so the Q and gate
            // projections can be dispatched independently.
            let has_output_gate = config
                .extra
                .get("attn_output_gate")
                .and_then(|v| v.as_bool())
                .unwrap_or_else(|| {
                    eprintln!(
                        "warning: Qwen 3.5 model missing 'attn_output_gate' config, defaulting to disabled"
                    );
                    false
                });
            if has_output_gate {
                let nh = config.num_attention_heads;
                let hd = config.head_dim;
                let h = config.hidden_size;
                for layer in &mut core.layers {
                    if layer.gdn_in_proj_qkv.is_some() {
                        continue;
                    }
                    let gate_weight = split_q_gate_weight(device, &mut layer.q_proj, nh, hd, h)?;
                    layer.attn_output_gate = Some(gate_weight);
                }
            }
        }

        // Try loading embedding as INT4 quantized. The QuantizedWeightProvider
        // wraps the original FP16 data with affine INT4 quantization metadata.
        // We load the raw tensor to get row-major packed data for gather
        // (not the blocked layout used for matmul).
        let embed_tensor = provider
            .tensor("model.embed_tokens.weight")
            .map_err(|e| MetalError::WeightLoading(format!("embed_tokens: {e}")))?;
        let (embedding, embedding_quantized) = match &embed_tensor.quant_info {
            mil_rs::weights::QuantizationInfo::AffineDequantize {
                scale,
                zero_point,
                scale_dtype,
                zero_point_dtype,
                axis,
                bit_width,
                group_size,
                ..
            } if *bit_width == 4 && !force_cpu_dequant => {
                let (n, k) = dense_shape(&embed_tensor.shape);
                let gs = group_size.unwrap_or(k);

                // Keep row-major layout for embedding gather (not blocked).
                let data_buf = device
                    .create_buffer_with_data(&embed_tensor.data, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;

                let scales_f16 = super::dequant::convert_params_to_f16(
                    scale,
                    *scale_dtype,
                    *axis,
                    &embed_tensor.shape,
                    gs,
                )?;
                let zeros_f16 = super::dequant::convert_params_to_f16(
                    zero_point,
                    *zero_point_dtype,
                    *axis,
                    &embed_tensor.shape,
                    gs,
                )?;

                let scales_buf = device
                    .create_buffer_with_data(&scales_f16, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                let zeros_buf = device
                    .create_buffer_with_data(&zeros_f16, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;

                let h = config.hidden_size;
                let scratch = device
                    .create_buffer((h * 2).max(16), StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                (
                    scratch,
                    Some(AffineQuantizedWeight {
                        data: data_buf, // row-major for embedding gather
                        scales: scales_buf,
                        zeros: zeros_buf,
                        group_size: gs as u32,
                        bit_width: *bit_width,
                        shape: (n, k),
                        awq_scales: None,
                    }),
                )
            }
            _ => {
                // FP16 or other format — dequant to dense.
                (core.embedding, None)
            }
        };

        let lm_head = if config.tie_word_embeddings {
            if embedding_quantized.is_some() {
                // Tied INT4 embedding: load lm_head as quantized weight
                // (same tensor, different layout — packed for matmul).
                load_weight_buffer(
                    device,
                    provider,
                    "model.embed_tokens.weight",
                    force_cpu_dequant,
                    true,
                )?
            } else {
                // Tied FP16 embedding: build packed lm_head from embedding data.
                let vocab = config.vocab_size;
                let h = config.hidden_size;
                let embed_bytes = vocab * h * 2;
                let mut raw = vec![0u8; embed_bytes];
                embedding
                    .read_bytes(&mut raw, 0)
                    .map_err(MetalError::Metal)?;
                if let Some(packed_data) = pack_bytes_blocked(&raw, vocab, h) {
                    let packed_buf = device
                        .create_buffer_with_data(&packed_data, StorageMode::Shared)
                        .map_err(MetalError::Metal)?;
                    WeightBuffer::Dense {
                        buf: None,
                        packed: Some(packed_buf),
                    }
                } else {
                    load_weight_buffer(
                        device,
                        provider,
                        "model.embed_tokens.weight",
                        force_cpu_dequant,
                        true,
                    )?
                }
            }
        } else {
            load_weight_buffer(device, provider, "lm_head.weight", force_cpu_dequant, true)?
        };

        Ok(Self {
            embedding,
            embedding_quantized,
            layers: core.layers,
            final_norm: core.final_norm,
            lm_head,
            config,
            ple_embed_tokens: core.ple_embed_tokens,
            ple_model_projection: core.ple_model_projection,
            ple_projection_norm: core.ple_projection_norm,
        })
    }

    /// Free redundant weight buffers across all layers and projections.
    ///
    /// After all load-time transforms (split_q_gate_weight, norm offsets,
    /// MLA absorption, D2Quant simulation) each weight only needs its
    /// primary dispatch layout. This calls [`WeightBuffer::compact`] on
    /// every weight to drop secondary copies:
    /// - Dense: drops row-major `buf` when a packed blocked copy exists
    pub fn compact(&mut self) {
        for layer in &mut self.layers {
            for wb in [
                &mut layer.q_proj,
                &mut layer.k_proj,
                &mut layer.v_proj,
                &mut layer.o_proj,
                &mut layer.gate_proj,
                &mut layer.up_proj,
                &mut layer.down_proj,
            ] {
                wb.compact();
            }
            if let Some(ref mut w) = layer.attn_output_gate {
                w.compact();
            }
            if let Some(ref mut w) = layer.gdn_in_proj_qkv {
                w.compact();
            }
            if let Some(ref mut w) = layer.gdn_in_proj_z {
                w.compact();
            }
            if let Some(ref mut w) = layer.gdn_in_proj_a {
                w.compact();
            }
            if let Some(ref mut w) = layer.gdn_in_proj_b {
                w.compact();
            }
            if let Some(ref mut w) = layer.gdn_out_proj {
                w.compact();
            }
            if let Some(ref mut w) = layer.router_weight {
                w.compact();
            }
            for w in &mut layer.expert_gate_projs {
                w.compact();
            }
            for w in &mut layer.expert_up_projs {
                w.compact();
            }
            for w in &mut layer.expert_down_projs {
                w.compact();
            }
            if let Some(ref mut w) = layer.ple_gate {
                w.compact();
            }
            if let Some(ref mut w) = layer.ple_projection {
                w.compact();
            }
        }
        self.lm_head.compact();
        if let Some(ref mut w) = self.ple_embed_tokens {
            w.compact();
        }
        if let Some(ref mut w) = self.ple_model_projection {
            w.compact();
        }
    }

    /// Apply D2Quant simulation in-place: quantize each weight buffer to
    /// `bits`-bit (2 or 3) using dual-scale quantization, then immediately
    /// dequantize back to FP16. This bakes the quantization error into the
    /// weights without requiring native GPU D2Quant kernels.
    ///
    /// Call after `load()` to measure the PPL impact of D2Quant quantization.
    pub fn apply_d2quant_simulation(&mut self, bits: u8) {
        for layer in &mut self.layers {
            for wb in [
                &mut layer.q_proj,
                &mut layer.k_proj,
                &mut layer.v_proj,
                &mut layer.o_proj,
                &mut layer.gate_proj,
                &mut layer.up_proj,
                &mut layer.down_proj,
            ] {
                d2quant_round_trip_weight_buffer(wb, bits);
            }
            if let Some(ref mut gdn_qkv) = layer.gdn_in_proj_qkv {
                d2quant_round_trip_weight_buffer(gdn_qkv, bits);
            }
            if let Some(ref mut gdn_z) = layer.gdn_in_proj_z {
                d2quant_round_trip_weight_buffer(gdn_z, bits);
            }
            if let Some(ref mut gdn_a) = layer.gdn_in_proj_a {
                d2quant_round_trip_weight_buffer(gdn_a, bits);
            }
            if let Some(ref mut gdn_b) = layer.gdn_in_proj_b {
                d2quant_round_trip_weight_buffer(gdn_b, bits);
            }
            if let Some(ref mut gdn_out) = layer.gdn_out_proj {
                d2quant_round_trip_weight_buffer(gdn_out, bits);
            }
            if let Some(ref mut gate) = layer.attn_output_gate {
                d2quant_round_trip_weight_buffer(gate, bits);
            }
        }
    }

    /// Temporarily replace a layer's projection weight buffer and return the original.
    pub(crate) fn swap_layer_weight(
        &mut self,
        layer_idx: usize,
        proj_name: &str,
        new_weight: WeightBuffer,
    ) -> Option<WeightBuffer> {
        if layer_idx >= self.layers.len() {
            return None;
        }
        let layer = &mut self.layers[layer_idx];
        let field = match proj_name {
            "q_proj" => &mut layer.q_proj,
            "k_proj" => &mut layer.k_proj,
            "v_proj" => &mut layer.v_proj,
            "o_proj" => &mut layer.o_proj,
            "gate_proj" => &mut layer.gate_proj,
            "up_proj" => &mut layer.up_proj,
            "down_proj" => &mut layer.down_proj,
            _ => return None,
        };
        Some(std::mem::replace(field, new_weight))
    }
}

/// Create a Dense (row-major only) FP16 [`WeightBuffer`] from f32 data on CPU.
pub(crate) fn create_dense_f16_buffer(
    device: &MetalDevice,
    data_f32: &[f32],
) -> Result<WeightBuffer, MetalError> {
    let f16_bytes: Vec<u8> = data_f32
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect();
    let buf = device
        .create_buffer_with_data(&f16_bytes, StorageMode::Shared)
        .map_err(MetalError::Metal)?;
    Ok(WeightBuffer::Dense {
        buf: Some(buf),
        packed: None,
    })
}

/// Create a Dense FP16 [`WeightBuffer`] with room for an `[n, k]` matrix.
///
/// Allocates both row-major (`buf`) and packed blocked (`packed`) buffers
/// so the weight is usable by `encode_projection` immediately.
/// Use [`update_dense_f16_data`] to populate it before dispatching.
pub(crate) fn create_dense_f16_buffer_sized(
    device: &MetalDevice,
    n: usize,
    k: usize,
) -> Result<WeightBuffer, MetalError> {
    let n_elements = n * k;
    let buf = device
        .create_buffer(n_elements * 2, StorageMode::Shared)
        .map_err(MetalError::Metal)?;
    let packed = if n % 8 == 0 && k % 8 == 0 {
        Some(
            device
                .create_buffer(n_elements * 2, StorageMode::Shared)
                .map_err(MetalError::Metal)?,
        )
    } else {
        None
    };
    Ok(WeightBuffer::Dense {
        buf: Some(buf),
        packed,
    })
}

/// Overwrite the contents of a Dense FP16 [`WeightBuffer`] with new f32 data.
///
/// Converts `data_f32` to FP16 on the CPU and writes into both the
/// row-major and packed blocked buffers (if present), avoiding new GPU
/// allocations.
///
/// `n` and `k` are the matrix dimensions (out_features, in_features),
/// required for blocked layout repacking.
///
/// # Errors
/// Returns an error if `wb` is not a Dense buffer, has no row-major buffer,
/// or the data is too large for the existing allocation.
pub(crate) fn update_dense_f16_data(
    wb: &WeightBuffer,
    data_f32: &[f32],
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    let (row_buf, packed_buf) = match wb {
        WeightBuffer::Dense {
            buf: Some(b),
            packed,
        } => (b, packed.as_ref()),
        _ => {
            return Err(MetalError::Other(anyhow::anyhow!(
                "update_dense_f16_data: expected Dense buffer with row-major allocation"
            )));
        }
    };
    let byte_len = data_f32.len() * 2;
    if byte_len > row_buf.length() {
        return Err(MetalError::BufferSizeMismatch {
            expected: byte_len,
            actual: row_buf.length(),
        });
    }
    let f16_bytes: Vec<u8> = data_f32
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect();
    row_buf
        .write_bytes(&f16_bytes, 0)
        .map_err(MetalError::Metal)?;

    // Update packed blocked buffer if present.
    if let Some(pb) = packed_buf {
        if let Some(packed_bytes) = pack_bytes_blocked(&f16_bytes, n, k) {
            pb.write_bytes(&packed_bytes, 0)
                .map_err(MetalError::Metal)?;
        }
    }
    Ok(())
}

/// Apply D2Quant quantize→dequantize round-trip to a single weight buffer.
///
/// Reads FP16 data from the buffer, converts to F32, runs dual-scale
/// quantization per 128-element group, dequantizes, converts back to FP16,
/// and writes the result. Dense buffers are modified in-place; quantized
/// buffers are skipped (they're already compressed).
fn d2quant_round_trip_weight_buffer(wb: &mut WeightBuffer, bits: u8) {
    use mil_rs::ir::passes::d2quant::dual_scale::{dual_scale_dequantize, dual_scale_quantize};

    let (dense_buf, packed_buf) = match wb {
        WeightBuffer::Dense { buf, packed } => (buf, packed),
        // Already quantized — skip.
        WeightBuffer::Quantized(_)
        | WeightBuffer::AffineQuantized(_)
        | WeightBuffer::DualScaleQuantized(_) => return,
    };

    // Helper: apply quantize→dequant round-trip to any FP16 Metal buffer.
    let apply_round_trip = |buf: &MetalBuffer| {
        let n = buf.length() / 2;
        let mut raw = vec![0u8; buf.length()];
        if let Err(e) = buf.read_bytes(&mut raw, 0) {
            eprintln!("d2quant round-trip: failed to read buffer: {e}");
            return;
        }
        let f32_weights: Vec<f32> = raw
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        let group_size = 128;
        let mut result = Vec::with_capacity(n);
        for group in f32_weights.chunks(group_size) {
            let (quantized, params) = dual_scale_quantize(group, bits, 0.99);
            let dequantized = dual_scale_dequantize(&quantized, &params, bits);
            result.extend_from_slice(&dequantized);
        }
        for i in 0..n {
            let fp16 = half::f16::from_f32(result[i]);
            let bytes = fp16.to_le_bytes();
            raw[i * 2] = bytes[0];
            raw[i * 2 + 1] = bytes[1];
        }
        if let Err(e) = buf.write_bytes(&raw, 0) {
            eprintln!("d2quant round-trip: failed to write buffer: {e}");
        }
    };

    if let Some(buf) = dense_buf {
        apply_round_trip(buf);
    }
    if let Some(packed) = packed_buf.as_ref() {
        apply_round_trip(packed);
    }
}

/// Add 1.0 to every FP16 element in a shared Metal buffer (in-place).
/// Used for Qwen 3.5 centered RMSNorm weight transform.
fn offset_norm_weight(buf: &MetalBuffer) {
    let n = buf.length() / 2;
    let mut data = vec![0u8; buf.length()];
    if let Err(e) = buf.read_bytes(&mut data, 0) {
        eprintln!("offset_norm_weight: failed to read buffer: {e}");
        return;
    }
    for i in 0..n {
        let off = i * 2;
        let val = half::f16::from_le_bytes([data[off], data[off + 1]]);
        let new_val = half::f16::from_f32(val.to_f32() + 1.0);
        let bytes = new_val.to_le_bytes();
        data[off] = bytes[0];
        data[off + 1] = bytes[1];
    }
    if let Err(e) = buf.write_bytes(&data, 0) {
        eprintln!("offset_norm_weight: failed to write buffer: {e}");
    }
}

/// Split q_proj [2*nh*hd, h] into Q [nh*hd, h] (kept in q_proj) and gate [nh*hd, h].
///
/// The original q_proj weight has interleaved Q and gate rows per head:
///   head 0: Q[0..hd], gate[0..hd], head 1: Q[0..hd], gate[0..hd], ...
///
/// Returns the gate WeightBuffer and overwrites q_proj with only the Q rows.
fn split_q_gate_weight(
    device: &MetalDevice,
    q_proj: &mut WeightBuffer,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
) -> Result<WeightBuffer, MetalError> {
    // Read the raw (unpacked) FP16 data from q_proj's buf.
    let full_n = 2 * num_heads * head_dim;
    let split_n = num_heads * head_dim;
    let k = hidden_size;
    let total_bytes = full_n * k * 2;

    let raw = match q_proj {
        &mut WeightBuffer::Dense { ref buf, .. } => {
            let buf = buf
                .as_ref()
                .ok_or_else(|| MetalError::WeightLoading("q_proj dense buffer is None".into()))?;
            let mut data = vec![0u8; total_bytes];
            buf.read_bytes(&mut data, 0).map_err(MetalError::Metal)?;
            data
        }
        _ => {
            return Err(MetalError::WeightLoading(
                "attn_output_gate requires dense q_proj weight".into(),
            ));
        }
    };

    // De-interleave: for each head, first head_dim rows are Q, next head_dim are gate.
    let q_bytes = split_n * k * 2;
    let mut q_data = vec![0u8; q_bytes];
    let mut gate_data = vec![0u8; q_bytes];

    for h in 0..num_heads {
        let src_q_start = h * 2 * head_dim;
        let src_g_start = h * 2 * head_dim + head_dim;
        let dst_start = h * head_dim;

        for r in 0..head_dim {
            let src_q_off = (src_q_start + r) * k * 2;
            let src_g_off = (src_g_start + r) * k * 2;
            let dst_off = (dst_start + r) * k * 2;
            let row_bytes = k * 2;

            q_data[dst_off..dst_off + row_bytes]
                .copy_from_slice(&raw[src_q_off..src_q_off + row_bytes]);
            gate_data[dst_off..dst_off + row_bytes]
                .copy_from_slice(&raw[src_g_off..src_g_off + row_bytes]);
        }
    }

    // Create new buffers and pack them.
    let q_buf = device
        .create_buffer_with_data(&q_data, StorageMode::Shared)
        .map_err(MetalError::Metal)?;
    let q_packed = pack_bytes_blocked(&q_data, split_n, k)
        .map(|packed_data| device.create_buffer_with_data(&packed_data, StorageMode::Shared))
        .transpose()
        .map_err(MetalError::Metal)?;
    *q_proj = WeightBuffer::Dense {
        buf: Some(q_buf),
        packed: q_packed,
    };

    let gate_buf = device
        .create_buffer_with_data(&gate_data, StorageMode::Shared)
        .map_err(MetalError::Metal)?;
    let gate_packed = pack_bytes_blocked(&gate_data, split_n, k)
        .map(|packed_data| device.create_buffer_with_data(&packed_data, StorageMode::Shared))
        .transpose()
        .map_err(MetalError::Metal)?;
    Ok(WeightBuffer::Dense {
        buf: Some(gate_buf),
        packed: gate_packed,
    })
}

/// Pack a row-major [N, K] FP16 **byte slice** into blocked [N/8, K/8, 8, 8] format.
///
/// Returns `None` if N or K is not a multiple of 8.
/// This operates entirely on CPU data — no GPU buffer roundtrip required.
fn pack_bytes_blocked(src: &[u8], n: usize, k: usize) -> Option<Vec<u8>> {
    if n % 8 != 0 || k % 8 != 0 {
        return None;
    }

    let total_bytes = n * k * 2; // FP16
    debug_assert_eq!(src.len(), total_bytes);

    let mut dst = vec![0u8; total_bytes];
    let n_blocks = n / 8;
    let k_blocks = k / 8;

    for nb in 0..n_blocks {
        for kb in 0..k_blocks {
            for ri in 0..8usize {
                let src_row = nb * 8 + ri;
                let dst_block_base = ((nb * k_blocks + kb) * 8 + ri) * 8;
                for ki in 0..8usize {
                    let src_col = kb * 8 + ki;
                    let src_offset = (src_row * k + src_col) * 2;
                    let dst_offset = (dst_block_base + ki) * 2;

                    dst[dst_offset] = src[src_offset];
                    dst[dst_offset + 1] = src[src_offset + 1];
                }
            }
        }
    }

    Some(dst)
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
/// When `pack_for_matmul` is true, dense weights are also pre-packed into
/// blocked format for the custom matvec kernel. Set to false for tensors
/// used only for gather/lookup (e.g. embedding tables).
fn load_weight_buffer(
    device: &MetalDevice,
    provider: &dyn WeightProvider,
    name: &str,
    force_cpu_dequant: bool,
    pack_for_matmul: bool,
) -> Result<WeightBuffer, MetalError> {
    let tensor = provider
        .tensor(name)
        .map_err(|e| MetalError::WeightLoading(format!("{name}: {e}")))?;
    match &tensor.quant_info {
        QuantizationInfo::None => {
            let (n, k) = dense_shape(&tensor.shape);
            if pack_for_matmul {
                if let Some(packed_data) = pack_bytes_blocked(&tensor.data, n, k) {
                    // Pack on CPU, upload once — no GPU→CPU readback.
                    let packed_buf = device
                        .create_buffer_with_data(&packed_data, StorageMode::Shared)
                        .map_err(MetalError::Metal)?;
                    let buf = device
                        .create_buffer_with_data(&tensor.data, StorageMode::Shared)
                        .map_err(MetalError::Metal)?;
                    return Ok(WeightBuffer::Dense {
                        buf: Some(buf),
                        packed: Some(packed_buf),
                    });
                }
            }
            // Fallback: dimensions not packable, or not needed for matmul.
            let buf = device
                .create_buffer_with_data(&tensor.data, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            Ok(WeightBuffer::Dense {
                buf: Some(buf),
                packed: None,
            })
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
                let (n, k) = dense_shape(original_shape);
                if pack_for_matmul {
                    if let Some(packed_data) = pack_bytes_blocked(&data, n, k) {
                        let packed_buf = device
                            .create_buffer_with_data(&packed_data, StorageMode::Shared)
                            .map_err(MetalError::Metal)?;
                        return Ok(WeightBuffer::Dense {
                            buf: None,
                            packed: Some(packed_buf),
                        });
                    }
                }
                // Gather/lookup or packing failed: always provide row-major buf.
                let buf = device
                    .create_buffer_with_data(&data, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                Ok(WeightBuffer::Dense {
                    buf: Some(buf),
                    packed: None,
                })
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
                let gs = match group_size {
                    Some(g) => *g,
                    None => {
                        eprintln!(
                            "warning: {name}: group_size metadata missing, defaulting to tensor size {}",
                            total_elements
                        );
                        total_elements
                    }
                };

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
            let (n, k) = dense_shape(&tensor.shape);
            if pack_for_matmul {
                if let Some(packed_data) = pack_bytes_blocked(&data, n, k) {
                    let packed_buf = device
                        .create_buffer_with_data(&packed_data, StorageMode::Shared)
                        .map_err(MetalError::Metal)?;
                    return Ok(WeightBuffer::Dense {
                        buf: None,
                        packed: Some(packed_buf),
                    });
                }
            }
            let buf = device
                .create_buffer_with_data(&data, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            Ok(WeightBuffer::Dense {
                buf: Some(buf),
                packed: None,
            })
        }
        QuantizationInfo::DualScaleDequantize {
            quantized_data,
            normal_scale,
            normal_zero,
            outlier_scale,
            outlier_zero,
            outlier_mask,
            original_shape,
            bit_width,
            group_size,
        } => {
            if force_cpu_dequant {
                let data = super::dequant::dequant_dual_scale(
                    quantized_data,
                    normal_scale,
                    normal_zero,
                    outlier_scale,
                    outlier_zero,
                    outlier_mask,
                    original_shape,
                    *bit_width,
                    *group_size,
                )?;
                let (n, k) = dense_shape(original_shape);
                if pack_for_matmul {
                    if let Some(packed_data) = pack_bytes_blocked(&data, n, k) {
                        let packed_buf = device
                            .create_buffer_with_data(&packed_data, StorageMode::Shared)
                            .map_err(MetalError::Metal)?;
                        return Ok(WeightBuffer::Dense {
                            buf: None,
                            packed: Some(packed_buf),
                        });
                    }
                }
                let buf = device
                    .create_buffer_with_data(&data, StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                return Ok(WeightBuffer::Dense {
                    buf: Some(buf),
                    packed: None,
                });
            }

            // Upload packed data + params to GPU — no CPU dequant needed.
            let data_buf = device
                .create_buffer_with_data(quantized_data, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            let ns_buf = device
                .create_buffer_with_data(normal_scale, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            let nz_buf = device
                .create_buffer_with_data(normal_zero, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            let os_buf = device
                .create_buffer_with_data(outlier_scale, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            let oz_buf = device
                .create_buffer_with_data(outlier_zero, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            let mask_buf = device
                .create_buffer_with_data(outlier_mask, StorageMode::Shared)
                .map_err(MetalError::Metal)?;
            let (n, k) = dense_shape(original_shape);
            Ok(WeightBuffer::DualScaleQuantized(DualScaleQuantizedWeight {
                data: data_buf,
                normal_scale: ns_buf,
                normal_zero: nz_buf,
                outlier_scale: os_buf,
                outlier_zero: oz_buf,
                outlier_mask: mask_buf,
                group_size: *group_size as u32,
                bit_width: *bit_width,
                shape: (n, k),
            }))
        }
        other => Err(MetalError::WeightLoading(format!(
            "{name}: unsupported quant_info variant: {other:?}"
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
