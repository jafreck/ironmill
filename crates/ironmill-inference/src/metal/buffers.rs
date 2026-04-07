//! Intermediate GPU buffers, MPS matmul caching, and buffer utilities.

use half::f16;
use ironmill_metal_sys::{
    MetalBuffer, MetalDevice, MpsMatrixMultiply, MpsMatrixMultiplyConfig, StorageMode,
};
use mil_rs::weights::ModelConfig;

use super::config::Gemma4Config;
use super::error::MetalError;
use super::weights::{MetalWeights, WeightBuffer};
use crate::engine::InferenceError;

// ── Intermediate activation buffers ─────────────────────────────

/// Group size for Q8 input quantization. Each group of elements shares
/// one FP32 scale factor. 128 matches llama.cpp's Q8_0 group size.
pub(crate) const Q8_GROUP_SIZE: usize = 128;

/// Reusable intermediate activation buffers.
pub(crate) struct IntermediateBuffers {
    pub(crate) hidden_state: MetalBuffer,
    pub(crate) attn_out: MetalBuffer,
    pub(crate) q_proj: MetalBuffer,
    pub(crate) k_proj: MetalBuffer,
    pub(crate) v_proj: MetalBuffer,
    /// Buffer for attention output gate values (Qwen3.5 attn_output_gate).
    pub(crate) q_gate: Option<MetalBuffer>,
    pub(crate) ffn_gate: MetalBuffer,
    pub(crate) ffn_up: MetalBuffer,
    pub(crate) ffn_down: MetalBuffer,
    pub(crate) residual: MetalBuffer,
    pub(crate) norm_out: MetalBuffer,
    pub(crate) logits: MetalBuffer,
    pub(crate) token_ids_buf: MetalBuffer,
    /// Second token IDs buffer for prefill pipelining — allows encoding
    /// the next chunk while the previous command buffer is still executing.
    pub(crate) token_ids_buf_b: MetalBuffer,
    /// PLE per-layer input buffer `[token_count, num_layers * ple_hidden_size]`.
    /// None when the model has no PLE.
    pub(crate) ple_per_layer_input: Option<MetalBuffer>,
    /// PLE scratch buffer for gate/projection intermediates `[token_count, ple_hidden_size]`.
    /// Reused across layers. None when the model has no PLE.
    pub(crate) ple_scratch: Option<MetalBuffer>,
    /// MoE router logits buffer `[token_count, num_experts]`. None when no MoE.
    pub(crate) moe_router_logits: Option<MetalBuffer>,
    /// MoE expert FFN gate scratch `[token_count, moe_intermediate_size]`. None when no MoE.
    pub(crate) moe_expert_gate: Option<MetalBuffer>,
    /// MoE expert FFN up scratch `[token_count, moe_intermediate_size]`. None when no MoE.
    pub(crate) moe_expert_up: Option<MetalBuffer>,
    /// MoE expert output accumulator `[num_experts, token_count, hidden_size]`. None when no MoE.
    pub(crate) moe_expert_outputs: Option<MetalBuffer>,
    /// MoE combined output `[token_count, hidden_size]`. None when no MoE.
    pub(crate) moe_combined: Option<MetalBuffer>,
    /// Q8-quantized input data `[max_tokens * hidden_size]` int8.
    /// Used for INT4×Q8 integer dot product in decode path.
    pub(crate) q8_data: MetalBuffer,
    /// Q8 per-group scale factors `[max_tokens * hidden_size / Q8_GROUP_SIZE]` float.
    pub(crate) q8_scales: MetalBuffer,
    /// FlashDecoding partial output `[max_splits × num_q_heads × head_dim]` float.
    pub(crate) flash_decode_partial_o: Option<MetalBuffer>,
    /// FlashDecoding partial softmax max `[max_splits × num_q_heads]` float.
    pub(crate) flash_decode_partial_max: Option<MetalBuffer>,
    /// FlashDecoding partial softmax sum `[max_splits × num_q_heads]` float.
    pub(crate) flash_decode_partial_sum: Option<MetalBuffer>,
    /// Maximum number of KV splits supported by current buffer allocation.
    pub(crate) flash_decode_max_splits: usize,
    /// Current maximum token capacity of these buffers.
    pub(crate) capacity: usize,
}

impl IntermediateBuffers {
    pub(crate) fn allocate(
        device: &MetalDevice,
        max_tokens: usize,
        mc: &ModelConfig,
        g4: Option<&Gemma4Config>,
    ) -> Result<Self, MetalError> {
        let h = mc.hidden_size;
        let nh = mc.num_attention_heads;
        let nkv = mc.num_key_value_heads;
        let hd = mc.head_dim;
        let inter = mc.intermediate_size;
        let vocab = mc.vocab_size;

        // For Gemma 4, use the maximum head_dim, num_kv_heads, and
        // intermediate_size across all layers so buffers are large enough
        // for both sliding (base) and global layers.
        let (hd, nkv, inter) = if let Some(g4) = g4 {
            let max_hd = g4
                .layer_configs
                .iter()
                .map(|lc| lc.head_dim)
                .max()
                .unwrap_or(hd);
            let max_nkv = g4
                .layer_configs
                .iter()
                .map(|lc| lc.num_kv_heads)
                .max()
                .unwrap_or(nkv);
            let max_inter = g4
                .layer_configs
                .iter()
                .map(|lc| lc.intermediate_size)
                .max()
                .unwrap_or(inter);
            (max_hd, max_nkv, max_inter)
        } else {
            (hd, nkv, inter)
        };

        let ple_hidden = mc
            .extra
            .get("hidden_size_per_layer_input")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let alloc_private = |size_elems: usize| -> Result<MetalBuffer, MetalError> {
            // GPU-only buffers use Private storage for ~10-15% bandwidth
            // improvement — no CPU coherency overhead.
            let bytes = (size_elems * 2).max(16);
            device
                .create_buffer(bytes, StorageMode::Private)
                .map_err(MetalError::Metal)
        };

        let alloc_shared = |size_elems: usize| -> Result<MetalBuffer, MetalError> {
            // Buffers that need CPU read/write use Shared storage.
            let bytes = (size_elems * 2).max(16);
            device
                .create_buffer(bytes, StorageMode::Shared)
                .map_err(MetalError::Metal)
        };

        let ple_per_layer_input = if ple_hidden > 0 {
            Some(alloc_private(
                max_tokens * mc.num_hidden_layers * ple_hidden,
            )?)
        } else {
            None
        };
        let ple_scratch = if ple_hidden > 0 {
            Some(alloc_private(max_tokens * ple_hidden)?)
        } else {
            None
        };

        let num_experts = mc
            .extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let moe_inter = mc
            .extra
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let has_moe = num_experts > 0 && moe_inter > 0;

        let moe_router_logits = if has_moe {
            Some(alloc_private(max_tokens * num_experts)?)
        } else {
            None
        };
        let moe_expert_gate = if has_moe {
            Some(alloc_private(max_tokens * moe_inter)?)
        } else {
            None
        };
        let moe_expert_up = if has_moe {
            Some(alloc_private(max_tokens * moe_inter)?)
        } else {
            None
        };
        let moe_expert_outputs = if has_moe {
            Some(alloc_private(num_experts * max_tokens * h)?)
        } else {
            None
        };
        let moe_combined = if has_moe {
            Some(alloc_private(max_tokens * h)?)
        } else {
            None
        };

        Ok(Self {
            hidden_state: alloc_private(max_tokens * h)?,
            attn_out: alloc_private(max_tokens * nh * hd)?,
            q_proj: alloc_private(max_tokens * nh * hd)?,
            k_proj: alloc_private(max_tokens * nkv * hd)?,
            v_proj: alloc_private(max_tokens * nkv * hd)?,
            q_gate: if mc
                .extra
                .get("attn_output_gate")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                Some(alloc_private(max_tokens * nh * hd)?)
            } else {
                None
            },
            ffn_gate: alloc_private(max_tokens * inter)?,
            ffn_up: alloc_private(max_tokens * inter)?,
            ffn_down: alloc_private(max_tokens * h)?,
            residual: alloc_private(max_tokens * h)?,
            norm_out: alloc_shared(max_tokens * h)?, // CPU reads in calibration
            logits: alloc_shared(max_tokens * vocab)?, // CPU reads logits
            token_ids_buf: device
                .create_buffer((max_tokens * 4).max(16), StorageMode::Shared) // CPU writes token IDs
                .map_err(MetalError::Metal)?,
            token_ids_buf_b: device
                .create_buffer((max_tokens * 4).max(16), StorageMode::Shared)
                .map_err(MetalError::Metal)?,
            ple_per_layer_input,
            ple_scratch,
            moe_router_logits,
            moe_expert_gate,
            moe_expert_up,
            moe_expert_outputs,
            moe_combined,
            // Q8 input quantization buffers: int8 data + float scales.
            // Sized for max_tokens × hidden_size (same as norm_out).
            q8_data: device
                .create_buffer((max_tokens * h).max(16), StorageMode::Private)
                .map_err(MetalError::Metal)?,
            q8_scales: device
                .create_buffer(
                    (max_tokens * h.div_ceil(Q8_GROUP_SIZE) * 4).max(16),
                    StorageMode::Private,
                )
                .map_err(MetalError::Metal)?,
            // FlashDecoding partial buffers: sized for max_splits × num_q_heads × head_dim.
            // Use 256 splits as initial max (covers up to ~256K context with 1K per split).
            flash_decode_partial_o: {
                let max_splits: usize = 256;
                let size = max_splits * nh * hd * 4; // float
                Some(
                    device
                        .create_buffer(size.max(16), StorageMode::Private)
                        .map_err(MetalError::Metal)?,
                )
            },
            flash_decode_partial_max: {
                let max_splits: usize = 256;
                let size = max_splits * nh * 4; // float
                Some(
                    device
                        .create_buffer(size.max(16), StorageMode::Private)
                        .map_err(MetalError::Metal)?,
                )
            },
            flash_decode_partial_sum: {
                let max_splits: usize = 256;
                let size = max_splits * nh * 4; // float
                Some(
                    device
                        .create_buffer(size.max(16), StorageMode::Private)
                        .map_err(MetalError::Metal)?,
                )
            },
            flash_decode_max_splits: 256,
            capacity: max_tokens,
        })
    }

    /// Grow buffers if `needed` exceeds current capacity. No-op otherwise.
    pub(crate) fn ensure_capacity(
        &mut self,
        device: &MetalDevice,
        needed: usize,
        mc: &ModelConfig,
        g4: Option<&Gemma4Config>,
    ) -> Result<(), MetalError> {
        if needed <= self.capacity {
            return Ok(());
        }
        *self = Self::allocate(device, needed, mc, g4)?;
        Ok(())
    }
}

// ── Projection matmul types ─────────────────────────────────────

/// Per-projection matmul state: Dense weights use MPS matrix multiplication
/// while Quantized weights use a custom compute-shader path and carry no MPS
/// object.
pub(crate) enum ProjectionMatmul {
    #[allow(dead_code)]
    Dense(MpsMatrixMultiply),
    Quantized,
}

impl ProjectionMatmul {
    /// Returns the MPS matmul for a Dense projection, or `None` for Quantized.
    #[allow(dead_code)]
    pub(crate) fn dense(&self) -> Option<&MpsMatrixMultiply> {
        match self {
            Self::Dense(m) => Some(m),
            Self::Quantized => None,
        }
    }
}

/// Cached MPS matmul instances for a given token count.
pub(crate) struct MpsMatmulCache {
    /// Token count these were built for.
    pub(crate) token_count: usize,
    /// Per-layer matmul instances: (q, k, v, o, gate, up, down).
    pub(crate) layer_matmuls: Vec<LayerMatmuls>,
}

pub(crate) struct LayerMatmuls {
    pub(crate) q: ProjectionMatmul,
    pub(crate) k: ProjectionMatmul,
    pub(crate) v: ProjectionMatmul,
    pub(crate) o: ProjectionMatmul,
    /// Gate projection matmul for attn_output_gate (Qwen3.5).
    pub(crate) q_gate: Option<ProjectionMatmul>,
    pub(crate) gate: ProjectionMatmul,
    pub(crate) up: ProjectionMatmul,
    pub(crate) down: ProjectionMatmul,
}

// ── RoPE cache builder ──────────────────────────────────────────

pub(crate) fn build_rope_cache(
    device: &MetalDevice,
    head_dim: usize,
    rotary_dim: usize,
    max_seq_len: usize,
    theta: f64,
    _partial_rotary_factor: f64,
) -> Result<(MetalBuffer, MetalBuffer), MetalError> {
    let half_dim = head_dim / 2;
    let rotary_half = rotary_dim / 2;
    let mut cos_data = vec![0u8; max_seq_len * half_dim * 2];
    let mut sin_data = vec![0u8; max_seq_len * half_dim * 2];

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let offset = (pos * half_dim + i) * 2;
            if i < rotary_half {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / rotary_dim as f64);
                let angle = pos as f64 * freq;
                let c = f16::from_f64(angle.cos());
                let s = f16::from_f64(angle.sin());
                cos_data[offset..offset + 2].copy_from_slice(&c.to_le_bytes());
                sin_data[offset..offset + 2].copy_from_slice(&s.to_le_bytes());
            } else {
                // Non-rotated dimensions: identity (cos=1, sin=0)
                let c = f16::from_f64(1.0);
                let s = f16::from_f64(0.0);
                cos_data[offset..offset + 2].copy_from_slice(&c.to_le_bytes());
                sin_data[offset..offset + 2].copy_from_slice(&s.to_le_bytes());
            }
        }
    }

    let cos_buf = device
        .create_buffer_with_data(&cos_data, StorageMode::Shared)
        .map_err(MetalError::Metal)?;
    let sin_buf = device
        .create_buffer_with_data(&sin_data, StorageMode::Shared)
        .map_err(MetalError::Metal)?;
    Ok((cos_buf, sin_buf))
}

// ── MPS matmul cache builder ────────────────────────────────────

pub(crate) fn build_matmul_cache(
    device: &MetalDevice,
    mc: &ModelConfig,
    g4: Option<&Gemma4Config>,
    weights: &MetalWeights,
    token_count: usize,
) -> Result<MpsMatmulCache, MetalError> {
    let h = mc.hidden_size;
    let nh = mc.num_attention_heads;
    let nkv = mc.num_key_value_heads;
    let hd = mc.head_dim;
    let inter = mc.intermediate_size;

    // Weight layout: [out_features, in_features] stored row-major FP16.
    // We compute: output = input × weight^T
    // MPS: result = left × right
    //   left  = input  [token_count × in_features]
    //   right = weight [out_features × in_features], transpose_right=true
    //   result = [token_count × out_features]
    let make_matmul =
        |rows: usize, cols: usize, inner: usize| -> Result<MpsMatrixMultiply, MetalError> {
            MpsMatrixMultiply::new(
                device,
                &MpsMatrixMultiplyConfig {
                    transpose_left: false,
                    transpose_right: true,   // weights are [out, in]
                    result_rows: rows,       // token_count
                    result_columns: cols,    // out_features
                    interior_columns: inner, // in_features
                    alpha: 1.0,
                    beta: 0.0,
                },
            )
            .map_err(MetalError::Metal)
        };

    // Only create MPS matmul instances for Dense weights; Quantized
    // projections use the custom compute kernel path instead.
    // Empty placeholders (GDN layer Q/K/V/O) produce Quantized stubs
    // that are never dispatched.
    let projection_matmul = |wb: &WeightBuffer,
                             rows: usize,
                             cols: usize,
                             inner: usize|
     -> Result<ProjectionMatmul, MetalError> {
        match wb {
            WeightBuffer::Dense {
                buf: None,
                packed: None,
            } => {
                // Empty placeholder (GDN layer) — no GPU memory, never dispatched.
                Ok(ProjectionMatmul::Quantized)
            }
            WeightBuffer::Dense { .. } => {
                Ok(ProjectionMatmul::Dense(make_matmul(rows, cols, inner)?))
            }
            WeightBuffer::Quantized(_) => Ok(ProjectionMatmul::Quantized),
            // AffineQuantized uses fused compute kernels — no MPS matmul needed.
            WeightBuffer::AffineQuantized(_) => Ok(ProjectionMatmul::Quantized),
            // DualScaleQuantized uses fused compute kernels — no MPS matmul needed.
            WeightBuffer::DualScaleQuantized(_) => Ok(ProjectionMatmul::Quantized),
        }
    };

    let mut layer_matmuls = Vec::with_capacity(mc.num_hidden_layers);
    for i in 0..mc.num_hidden_layers {
        let lw = &weights.layers[i];
        // Use per-layer head_dim/kv_heads/intermediate_size for Gemma 4.
        let (layer_hd, layer_nkv, layer_inter) = if let Some(g4) = g4 {
            let lc = &g4.layer_configs[i];
            (lc.head_dim, lc.num_kv_heads, lc.intermediate_size)
        } else {
            (hd, nkv, inter)
        };
        layer_matmuls.push(LayerMatmuls {
            q: projection_matmul(&lw.q_proj, token_count, nh * layer_hd, h)?,
            k: projection_matmul(&lw.k_proj, token_count, layer_nkv * layer_hd, h)?,
            v: projection_matmul(&lw.v_proj, token_count, layer_nkv * layer_hd, h)?,
            o: projection_matmul(&lw.o_proj, token_count, h, nh * layer_hd)?,
            q_gate: if let Some(ref gw) = lw.attn_output_gate {
                Some(projection_matmul(gw, token_count, nh * layer_hd, h)?)
            } else {
                None
            },
            gate: projection_matmul(&lw.gate_proj, token_count, layer_inter, h)?,
            up: projection_matmul(&lw.up_proj, token_count, layer_inter, h)?,
            down: projection_matmul(&lw.down_proj, token_count, h, layer_inter)?,
        });
    }

    Ok(MpsMatmulCache {
        token_count,
        layer_matmuls,
    })
}

// ── Byte → f16 conversion helper ───────────────────────────────

/// Reinterpret a raw byte slice as `&[f16]`.
///
/// # Panics
///
/// Panics if `bytes.len()` is not a multiple of 2 or if the pointer is not
/// 2-byte aligned. Both conditions are guaranteed for Metal buffer readbacks.
pub(crate) fn bytes_as_f16(bytes: &[u8]) -> &[f16] {
    let elem_size = std::mem::size_of::<f16>();
    assert!(
        bytes.len() % elem_size == 0,
        "byte slice length {} is not a multiple of f16 size ({})",
        bytes.len(),
        elem_size,
    );
    assert!(
        bytes.as_ptr() as usize % std::mem::align_of::<f16>() == 0,
        "byte slice pointer is not aligned for f16",
    );
    // SAFETY: We have verified length and alignment. `f16` is `#[repr(transparent)]`
    // over `u16` and has no invalid bit patterns.
    #[allow(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f16, bytes.len() / elem_size)
    }
}

// ── Helpers on ModelConfig ──────────────────────────────────────

pub(crate) trait ModelConfigExt {
    fn num_kv_heads(&self) -> usize;
}

impl ModelConfigExt for ModelConfig {
    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
    }
}

// ── Buffer read/write utilities ─────────────────────────────────

/// Read an FP16 Metal buffer into a Vec<f32>, converting from f16.
pub(crate) fn read_buffer_f32(
    buf: &MetalBuffer,
    num_elements: usize,
) -> Result<Vec<f32>, InferenceError> {
    let byte_count = num_elements * 2;
    let mut raw = vec![0u8; byte_count];
    buf.read_bytes(&mut raw, 0)
        .map_err(|e| InferenceError::runtime(e.to_string()))?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect())
}

/// Write a Vec<f32> into a shared Metal buffer as FP16.
pub(crate) fn write_buffer_f32(buf: &MetalBuffer, data: &[f32]) -> Result<(), InferenceError> {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|v| f16::from_f32(*v).to_le_bytes())
        .collect();
    buf.write_bytes(&bytes, 0)
        .map_err(|e| InferenceError::runtime(e.to_string()))
}

/// Read a weight buffer as dense FP32 values.
pub(crate) fn read_weight_f32(
    wb: &WeightBuffer,
    num_elements: usize,
) -> Result<Vec<f32>, InferenceError> {
    let buf = wb
        .as_dense()
        .map_err(|e| InferenceError::runtime(format!("GDN requires dense weights: {e}")))?;
    read_buffer_f32(buf, num_elements)
}
