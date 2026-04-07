//! Intermediate GPU buffers, MPS matmul caching, and buffer utilities.

use half::f16;
use ironmill_metal_sys::{MetalBuffer, MetalDevice, StorageMode};
use mil_rs::weights::ModelConfig;

use super::config::Gemma4Config;
use super::error::MetalError;
use super::weights::MetalWeights;

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
    /// FlashDecoding max hint from previous decode step `[num_q_heads]` float.
    /// Used as the initial softmax max estimate to avoid rescaling when the
    /// running max hasn't changed between adjacent tokens.
    pub(crate) flash_decode_max_hint: Option<MetalBuffer>,
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
            flash_decode_max_hint: {
                // Initialized to -INFINITY; reduce kernel writes actual max.
                let size = nh * 4; // float per Q head
                let buf = device
                    .create_buffer(size.max(16), StorageMode::Shared)
                    .map_err(MetalError::Metal)?;
                // Initialize to -INFINITY so first step has no bias.
                let neg_inf_bytes: Vec<u8> = (0..nh)
                    .flat_map(|_| f32::NEG_INFINITY.to_le_bytes())
                    .collect();
                buf.write_bytes(&neg_inf_bytes, 0)
                    .map_err(MetalError::Metal)?;
                Some(buf)
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

/// Cached MPS matmul instances for a given token count.
pub(crate) struct MpsMatmulCache {
    /// Token count these were built for.
    pub(crate) token_count: usize,
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
    _device: &MetalDevice,
    _mc: &ModelConfig,
    _g4: Option<&Gemma4Config>,
    _weights: &MetalWeights,
    token_count: usize,
) -> Result<MpsMatmulCache, MetalError> {
    Ok(MpsMatmulCache { token_count })
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
