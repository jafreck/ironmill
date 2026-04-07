//! Kernel dispatch helpers — compile shaders and dispatch compute operations.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer, MetalDevice};

use super::error::MetalError;

/// Maximum threads per threadgroup imposed by the Metal API.
const METAL_MAX_THREADS_PER_THREADGROUP: usize = 1024;

/// Default threadgroup width for 1-D element-wise dispatches.
///
/// 256 threads (8 SIMD-groups on Apple GPUs) strikes a good balance between
/// occupancy and register pressure for simple per-element kernels.
const DEFAULT_THREADGROUP_WIDTH: usize = 256;

/// Minimum threadgroup width for attention kernels.
///
/// The attention kernel processes QK^T positions in parallel across
/// simdgroups (one position per simdgroup). Using more threads than
/// head_dim improves occupancy and enables more parallel positions.
const ATTENTION_MIN_THREADGROUP_WIDTH: usize = 256;

/// Number of output rows processed by each threadgroup in the custom
/// FP16 matvec kernel. The shader is written for 8 SIMD-groups × 8 rows
/// each, giving 64 rows per threadgroup.
const MATVEC_ROWS_PER_THREADGROUP: usize = 64;

/// Whether to dispatch a linear projection as a memory-bandwidth-optimized
/// matvec (single-token decode) or a compute-optimized batched matmul
/// (multi-token prefill).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearKernelKind {
    /// Single-token decode: use matvec kernels optimized for memory bandwidth
    /// (wider threadgroups, coalesced reads, no shared-memory tiling).
    Matvec,
    /// Multi-token prefill: use batched matmul kernels that tile across M and N.
    Matmul,
}

impl LinearKernelKind {
    /// Select the optimal kernel variant based on token count.
    ///
    /// - `token_count == 1` → [`Matvec`](Self::Matvec) (decode phase, memory-bound)
    /// - `token_count > 1`  → [`Matmul`](Self::Matmul) (prefill phase, compute-bound)
    #[inline]
    pub fn for_token_count(token_count: usize) -> Self {
        if token_count == 1 {
            Self::Matvec
        } else {
            Self::Matmul
        }
    }

    /// Returns `true` when this is the decode (matvec) variant.
    #[inline]
    pub fn is_decode(self) -> bool {
        self == Self::Matvec
    }
}

/// All compiled Metal pipeline states for the Metal backend.
pub struct MetalPipelines {
    /// RMSNorm kernel.
    pub rms_norm: ComputePipeline,
    /// SiLU-gated activation kernel.
    pub silu_gate: ComputePipeline,
    /// GELU-gated activation kernel (Gemma 4).
    pub ffn_gelu_gate: ComputePipeline,
    /// Rotary positional embedding kernel.
    pub rope: ComputePipeline,
    /// Element-wise residual addition kernel.
    pub residual_add: ComputePipeline,
    /// Broadcast bias add: data[i] += bias[i % H]. Used for DAC correction.
    pub bias_add: ComputePipeline,
    /// Buffer copy kernel (element-wise half → half).
    pub copy_buffer: ComputePipeline,
    /// Token embedding lookup kernel.
    pub embedding_lookup: ComputePipeline,
    /// TurboQuant KV cache write kernel.
    pub turboquant_cache_write: ComputePipeline,
    /// TurboQuant attention kernel.
    pub turboquant_attention: ComputePipeline,
    /// TurboQuant outlier KV cache write kernel.
    pub turboquant_outlier_cache_write: ComputePipeline,
    /// TurboQuant outlier attention kernel.
    pub turboquant_outlier_attention: ComputePipeline,
    /// Standard FP16 attention (decode) kernel.
    pub standard_attention: ComputePipeline,
    /// Prefill attention kernel.
    pub prefill_attention: ComputePipeline,
    /// FlashAttention-2 prefill kernel.
    pub prefill_attention_fa2: ComputePipeline,
    /// Register-tiled FA2 prefill kernel (v2).
    pub prefill_attention_v2: ComputePipeline,
    /// PolarQuant INT4 matvec kernel.
    pub polarquant_matvec_int4: ComputePipeline,
    /// PolarQuant INT4 matmul kernel.
    pub polarquant_matmul_int4: ComputePipeline,
    /// PolarQuant INT8 matvec kernel.
    pub polarquant_matvec_int8: ComputePipeline,
    /// PolarQuant INT8 matmul kernel.
    pub polarquant_matmul_int8: ComputePipeline,
    /// Affine INT4 matvec kernel.
    pub affine_matvec_int4: ComputePipeline,
    /// Affine INT4 matmul kernel.
    pub affine_matmul_int4: ComputePipeline,
    /// Affine INT8 matvec kernel.
    pub affine_matvec_int8: ComputePipeline,
    /// Affine INT8 matmul kernel.
    pub affine_matmul_int8: ComputePipeline,
    /// KV scatter (cache write) kernel.
    pub kv_scatter: ComputePipeline,
    /// Dense FP16 matvec kernel.
    pub matvec: ComputePipeline,
    /// Dense FP16 matmul kernel.
    pub matmul: ComputePipeline,
    /// Fused residual-add + RMSNorm kernel.
    pub fused_residual_rms_norm: ComputePipeline,
    /// Fused QK normalization + RoPE kernel.
    pub fused_qk_norm_rope: ComputePipeline,
    /// Fused embedding lookup + first-layer norm kernel.
    pub fused_embedding_norm: ComputePipeline,
    /// INT4 dequantization kernel.
    pub int4_dequantize: ComputePipeline,
    /// Fused scaled dot-product attention kernel.
    /// `None` when the kernel exceeds threadgroup memory limits (head_dim ≥ 256).
    /// Decode uses FlashDecoding split+reduce instead.
    pub fused_sdpa: Option<ComputePipeline>,
    /// QuIP# matvec kernel.
    pub quip_sharp_matvec: ComputePipeline,
    /// QuIP# matmul kernel.
    pub quip_sharp_matmul: ComputePipeline,
    /// Fused softcapping kernel.
    pub fused_softcap: ComputePipeline,
    /// PLE GELU-gated activation kernel.
    pub ple_gelu_gate: ComputePipeline,
    /// PLE add-and-scale kernel.
    pub ple_add_scale: ComputePipeline,
    /// Element-wise buffer scaling kernel.
    pub scale_buffer: ComputePipeline,
    /// MoE router softmax kernel.
    pub moe_softmax: ComputePipeline,
    /// MoE expert GELU activation kernel.
    pub moe_gelu: ComputePipeline,
    /// MoE element-wise multiply kernel.
    pub moe_mul: ComputePipeline,
    /// MoE weighted expert combination kernel.
    pub moe_weighted_combine: ComputePipeline,
    /// D2Quant 3-bit matvec kernel.
    pub d2quant_matvec_3bit: ComputePipeline,
    /// D2Quant 3-bit matmul kernel.
    pub d2quant_matmul_3bit: ComputePipeline,
    /// D2Quant 3-bit embedding lookup kernel.
    pub d2quant_embedding_lookup_3bit: ComputePipeline,
    /// Sigmoid gating kernel (Qwen3.5 attn_output_gate).
    pub sigmoid_gate: ComputePipeline,
    /// GDN conv1d + SiLU kernel.
    pub gdn_conv1d_silu: ComputePipeline,
    /// GDN recurrent state update kernel.
    pub gdn_recurrent_update: ComputePipeline,
    /// GDN per-head output gate (RMSNorm + silu(z)) kernel.
    pub gdn_output_gate: ComputePipeline,
    /// GDN prefill batched conv1d + SiLU kernel (all tokens, one dispatch).
    pub gdn_prefill_conv1d_silu: ComputePipeline,
    /// GDN prefill batched recurrent + norm + gate kernel (all tokens, one dispatch).
    pub gdn_prefill_recurrent: ComputePipeline,
    /// Fused GDN decode kernel: conv1d+SiLU+recurrent+output_gate in one dispatch.
    pub gdn_fused_decode: ComputePipeline,
    /// Batched dense FP16 matvec for 4 GDN projections in one dispatch.
    pub gdn_batched_matvec: ComputePipeline,
    /// Batched affine INT4 matvec for FFN gate+up in one dispatch.
    pub batched_affine_matvec_int4: ComputePipeline,
    /// Batched affine INT4 matvec for 4 GDN projections in one dispatch.
    pub gdn_batched_affine_matvec_int4: ComputePipeline,
    /// AMX-accelerated INT4 matvec: dequant to threadgroup memory + simdgroup matrix multiply.
    pub affine_matvec_int4_amx: ComputePipeline,
    /// Row-major INT4 matvec for decode: ~32× better cache utilization than blocked layout.
    pub affine_matvec_int4_rowmajor: ComputePipeline,
    /// Fused FFN gate+up+activation for INT4 decode: gate+up dot products + SiLU/GELU inline.
    pub fused_ffn_gate_up_act_int4: ComputePipeline,
    /// Fused residual+RMSNorm+dense matvec in one dispatch.
    pub fused_residual_norm_matvec: ComputePipeline,
    /// Fused residual+RMSNorm+affine INT4 matvec in one dispatch.
    pub fused_residual_norm_affine_matvec_int4: ComputePipeline,
    /// Q8 input quantization kernel: FP16 → INT8 with per-group scales.
    pub quantize_input_q8: ComputePipeline,
    /// INT4×Q8 integer dot product matvec (decode path).
    pub affine_matvec_int4xq8: ComputePipeline,
    /// FlashDecoding split kernel: each threadgroup processes a KV slice.
    pub fused_sdpa_split: ComputePipeline,
    /// FlashDecoding reduce kernel: combines partial results across splits.
    pub fused_sdpa_reduce: ComputePipeline,
}

impl MetalPipelines {
    /// Load precompiled Metal shader libraries and create pipeline states.
    ///
    /// HEAD_DIM-independent shaders are loaded from precompiled `.metallib`
    /// binaries embedded at build time (~1ms). HEAD_DIM-dependent shaders
    /// (attention, turboquant) use precompiled variants for common head
    /// dimensions (64, 80, 128, 256) and fall back to runtime source
    /// compilation for uncommon values.
    pub fn compile(
        device: &MetalDevice,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<Self, MetalError> {
        // ── HEAD_DIM-independent shaders (precompiled) ──────────
        let norm_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/normalization.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let act_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/activation.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let rope_lib = device
            .load_library_from_data(include_bytes!(concat!(env!("OUT_DIR"), "/rope.metallib")))
            .map_err(MetalError::Metal)?;
        let elem_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/elementwise.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let embed_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/embedding.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let qmm_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/quantized_matmul.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let kv_scatter_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/kv_scatter.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let matvec_lib = device
            .load_library_from_data(include_bytes!(concat!(env!("OUT_DIR"), "/matvec.metallib")))
            .map_err(MetalError::Metal)?;
        let fused_rn_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/fused_residual_norm.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let fused_en_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/fused_embedding_norm.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let int4_dequant_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/int4_dequant.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let quip_sharp_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/quip_sharp.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let fused_softcap_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/fused_softcap.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let ple_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/ple_kernels.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let affine_mm_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/affine_matmul.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let d2quant_mm_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/d2quant_matmul.metallib"
            )))
            .map_err(MetalError::Metal)?;
        let gdn_lib = device
            .load_library_from_data(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/gdn_recurrent.metallib"
            )))
            .map_err(MetalError::Metal)?;

        // ── HEAD_DIM-dependent shaders (precompiled or fallback) ─
        let (attn_lib, tq_lib, sdpa_lib, fd_lib) =
            Self::load_head_dim_shaders(device, head_dim, rotary_dim)?;

        Ok(Self {
            rms_norm: device
                .create_compute_pipeline(
                    &norm_lib
                        .get_function("rms_norm")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            silu_gate: device
                .create_compute_pipeline(
                    &act_lib
                        .get_function("silu_gate")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            ffn_gelu_gate: device
                .create_compute_pipeline(
                    &act_lib
                        .get_function("gelu_gate")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            rope: device
                .create_compute_pipeline(&rope_lib.get_function("rope").map_err(MetalError::Metal)?)
                .map_err(MetalError::Metal)?,
            residual_add: device
                .create_compute_pipeline(
                    &elem_lib
                        .get_function("residual_add")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            bias_add: device
                .create_compute_pipeline(
                    &elem_lib
                        .get_function("bias_add")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            copy_buffer: device
                .create_compute_pipeline(
                    &elem_lib
                        .get_function("copy_buffer")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            sigmoid_gate: device
                .create_compute_pipeline(
                    &elem_lib
                        .get_function("sigmoid_gate_inplace")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            embedding_lookup: device
                .create_compute_pipeline(
                    &embed_lib
                        .get_function("embedding_lookup")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            turboquant_cache_write: device
                .create_compute_pipeline(
                    &tq_lib
                        .get_function("turboquant_cache_write")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            turboquant_attention: device
                .create_compute_pipeline(
                    &tq_lib
                        .get_function("turboquant_attention")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            turboquant_outlier_cache_write: device
                .create_compute_pipeline(
                    &tq_lib
                        .get_function("turboquant_outlier_cache_write")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            turboquant_outlier_attention: device
                .create_compute_pipeline(
                    &tq_lib
                        .get_function("turboquant_outlier_attention")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            standard_attention: device
                .create_compute_pipeline(
                    &attn_lib
                        .get_function("standard_attention")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            prefill_attention: device
                .create_compute_pipeline(
                    &attn_lib
                        .get_function("prefill_attention")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            prefill_attention_fa2: device
                .create_compute_pipeline(
                    &attn_lib
                        .get_function("prefill_attention_fa2")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            prefill_attention_v2: device
                .create_compute_pipeline(
                    &attn_lib
                        .get_function("prefill_attention_v2")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            polarquant_matvec_int4: device
                .create_compute_pipeline(
                    &qmm_lib
                        .get_function("polarquant_matvec_int4")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            polarquant_matmul_int4: device
                .create_compute_pipeline(
                    &qmm_lib
                        .get_function("polarquant_matmul_int4")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            polarquant_matvec_int8: device
                .create_compute_pipeline(
                    &qmm_lib
                        .get_function("polarquant_matvec_int8")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            polarquant_matmul_int8: device
                .create_compute_pipeline(
                    &qmm_lib
                        .get_function("polarquant_matmul_int8")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            affine_matvec_int4: device
                .create_compute_pipeline(
                    &affine_mm_lib
                        .get_function("affine_matvec_int4")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            affine_matmul_int4: device
                .create_compute_pipeline(
                    &affine_mm_lib
                        .get_function("affine_matmul_int4")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            affine_matvec_int8: device
                .create_compute_pipeline(
                    &affine_mm_lib
                        .get_function("affine_matvec_int8")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            affine_matmul_int8: device
                .create_compute_pipeline(
                    &affine_mm_lib
                        .get_function("affine_matmul_int8")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            kv_scatter: device
                .create_compute_pipeline(
                    &kv_scatter_lib
                        .get_function("kv_scatter")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            matvec: device
                .create_compute_pipeline(
                    &matvec_lib
                        .get_function("matvec")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            matmul: device
                .create_compute_pipeline(
                    &matvec_lib
                        .get_function("matmul")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            fused_residual_rms_norm: device
                .create_compute_pipeline(
                    &fused_rn_lib
                        .get_function("fused_residual_rms_norm")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            fused_qk_norm_rope: device
                .create_compute_pipeline(
                    &attn_lib
                        .get_function("fused_qk_norm_rope")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            fused_embedding_norm: device
                .create_compute_pipeline(
                    &fused_en_lib
                        .get_function("fused_embedding_norm")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            int4_dequantize: device
                .create_compute_pipeline(
                    &int4_dequant_lib
                        .get_function("int4_dequantize")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            fused_sdpa: sdpa_lib
                .get_function("fused_sdpa")
                .ok()
                .and_then(|f| device.create_compute_pipeline(&f).ok()),
            quip_sharp_matvec: device
                .create_compute_pipeline(
                    &quip_sharp_lib
                        .get_function("quip_sharp_matvec")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            quip_sharp_matmul: device
                .create_compute_pipeline(
                    &quip_sharp_lib
                        .get_function("quip_sharp_matmul")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            fused_softcap: device
                .create_compute_pipeline(
                    &fused_softcap_lib
                        .get_function("fused_softcap")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            ple_gelu_gate: device
                .create_compute_pipeline(
                    &ple_lib
                        .get_function("gelu_gate")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            ple_add_scale: device
                .create_compute_pipeline(
                    &ple_lib
                        .get_function("add_scale")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            scale_buffer: device
                .create_compute_pipeline(
                    &elem_lib
                        .get_function("scale_buffer")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            moe_softmax: {
                let lib = device
                    .load_library_from_data(include_bytes!(concat!(
                        env!("OUT_DIR"),
                        "/moe.metallib"
                    )))
                    .map_err(MetalError::Metal)?;
                device
                    .create_compute_pipeline(
                        &lib.get_function("moe_softmax").map_err(MetalError::Metal)?,
                    )
                    .map_err(MetalError::Metal)?
            },
            moe_gelu: {
                let lib = device
                    .load_library_from_data(include_bytes!(concat!(
                        env!("OUT_DIR"),
                        "/moe.metallib"
                    )))
                    .map_err(MetalError::Metal)?;
                device
                    .create_compute_pipeline(
                        &lib.get_function("moe_gelu").map_err(MetalError::Metal)?,
                    )
                    .map_err(MetalError::Metal)?
            },
            moe_mul: {
                let lib = device
                    .load_library_from_data(include_bytes!(concat!(
                        env!("OUT_DIR"),
                        "/moe.metallib"
                    )))
                    .map_err(MetalError::Metal)?;
                device
                    .create_compute_pipeline(
                        &lib.get_function("moe_mul").map_err(MetalError::Metal)?,
                    )
                    .map_err(MetalError::Metal)?
            },
            moe_weighted_combine: {
                let lib = device
                    .load_library_from_data(include_bytes!(concat!(
                        env!("OUT_DIR"),
                        "/moe.metallib"
                    )))
                    .map_err(MetalError::Metal)?;
                device
                    .create_compute_pipeline(
                        &lib.get_function("moe_weighted_combine")
                            .map_err(MetalError::Metal)?,
                    )
                    .map_err(MetalError::Metal)?
            },
            d2quant_matvec_3bit: device
                .create_compute_pipeline(
                    &d2quant_mm_lib
                        .get_function("d2quant_matvec_3bit")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            d2quant_matmul_3bit: device
                .create_compute_pipeline(
                    &d2quant_mm_lib
                        .get_function("d2quant_matmul_3bit")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            d2quant_embedding_lookup_3bit: device
                .create_compute_pipeline(
                    &d2quant_mm_lib
                        .get_function("d2quant_embedding_lookup_3bit")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            gdn_conv1d_silu: device
                .create_compute_pipeline(
                    &gdn_lib
                        .get_function("gdn_conv1d_silu")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            gdn_recurrent_update: device
                .create_compute_pipeline(
                    &gdn_lib
                        .get_function("gdn_recurrent_update")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            gdn_output_gate: device
                .create_compute_pipeline(
                    &gdn_lib
                        .get_function("gdn_output_gate")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            gdn_prefill_conv1d_silu: device
                .create_compute_pipeline(
                    &gdn_lib
                        .get_function("gdn_prefill_conv1d_silu")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            gdn_prefill_recurrent: device
                .create_compute_pipeline(
                    &gdn_lib
                        .get_function("gdn_prefill_recurrent")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            gdn_fused_decode: device
                .create_compute_pipeline(
                    &gdn_lib
                        .get_function("gdn_fused_decode")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            gdn_batched_matvec: device
                .create_compute_pipeline(
                    &matvec_lib
                        .get_function("gdn_batched_matvec")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            batched_affine_matvec_int4: device
                .create_compute_pipeline(
                    &affine_mm_lib
                        .get_function("batched_affine_matvec_int4")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            gdn_batched_affine_matvec_int4: device
                .create_compute_pipeline(
                    &affine_mm_lib
                        .get_function("gdn_batched_affine_matvec_int4")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            affine_matvec_int4_amx: device
                .create_compute_pipeline(
                    &affine_mm_lib
                        .get_function("affine_matvec_int4_amx")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            affine_matvec_int4_rowmajor: device
                .create_compute_pipeline(
                    &affine_mm_lib
                        .get_function("affine_matvec_int4_rowmajor")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            fused_ffn_gate_up_act_int4: device
                .create_compute_pipeline(
                    &affine_mm_lib
                        .get_function("fused_ffn_gate_up_act_int4")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            fused_residual_norm_matvec: device
                .create_compute_pipeline(
                    &fused_rn_lib
                        .get_function("fused_residual_norm_matvec")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            fused_residual_norm_affine_matvec_int4: device
                .create_compute_pipeline(
                    &fused_rn_lib
                        .get_function("fused_residual_norm_affine_matvec_int4")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            quantize_input_q8: device
                .create_compute_pipeline(
                    &elem_lib
                        .get_function("quantize_input_q8")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            affine_matvec_int4xq8: device
                .create_compute_pipeline(
                    &affine_mm_lib
                        .get_function("affine_matvec_int4xq8")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            fused_sdpa_split: device
                .create_compute_pipeline(
                    &fd_lib
                        .get_function("fused_sdpa_split")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
            fused_sdpa_reduce: device
                .create_compute_pipeline(
                    &fd_lib
                        .get_function("fused_sdpa_reduce")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
        })
    }

    /// Load attention, turboquant, fused SDPA, and FlashDecoding shaders for a specific HEAD_DIM.
    ///
    /// Uses precompiled `.metallib` for common values (64, 80, 128, 256, 512);
    /// falls back to runtime source compilation for uncommon dimensions.
    fn load_head_dim_shaders(
        device: &MetalDevice,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<
        (
            ironmill_metal_sys::ShaderLibrary,
            ironmill_metal_sys::ShaderLibrary,
            ironmill_metal_sys::ShaderLibrary,
            ironmill_metal_sys::ShaderLibrary,
        ),
        MetalError,
    > {
        // Macro to embed a precompiled metallib by head_dim.
        macro_rules! precompiled {
            (attn $hd:literal) => {
                include_bytes!(concat!(env!("OUT_DIR"), "/attention_hd", $hd, ".metallib"))
            };
            (tq $hd:literal) => {
                include_bytes!(concat!(env!("OUT_DIR"), "/turboquant_hd", $hd, ".metallib"))
            };
            (sdpa $hd:literal) => {
                include_bytes!(concat!(env!("OUT_DIR"), "/fused_sdpa_hd", $hd, ".metallib"))
            };
            (fd $hd:literal) => {
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/flash_decode_hd",
                    $hd,
                    ".metallib"
                ))
            };
        }

        let try_precompiled = |attn_data: &[u8],
                               tq_data: &[u8],
                               sdpa_data: &[u8],
                               fd_data: &[u8]|
         -> Result<
            (
                ironmill_metal_sys::ShaderLibrary,
                ironmill_metal_sys::ShaderLibrary,
                ironmill_metal_sys::ShaderLibrary,
                ironmill_metal_sys::ShaderLibrary,
            ),
            MetalError,
        > {
            let attn = device
                .load_library_from_data(attn_data)
                .map_err(MetalError::Metal)?;
            let tq = device
                .load_library_from_data(tq_data)
                .map_err(MetalError::Metal)?;
            let sdpa = device
                .load_library_from_data(sdpa_data)
                .map_err(MetalError::Metal)?;
            let fd = device
                .load_library_from_data(fd_data)
                .map_err(MetalError::Metal)?;
            Ok((attn, tq, sdpa, fd))
        };

        match head_dim {
            64 => try_precompiled(
                precompiled!(attn "64"),
                precompiled!(tq "64"),
                precompiled!(sdpa "64"),
                precompiled!(fd "64"),
            ),
            80 => try_precompiled(
                precompiled!(attn "80"),
                precompiled!(tq "80"),
                precompiled!(sdpa "80"),
                precompiled!(fd "80"),
            ),
            128 => try_precompiled(
                precompiled!(attn "128"),
                precompiled!(tq "128"),
                precompiled!(sdpa "128"),
                precompiled!(fd "128"),
            ),
            256 => {
                // Head dim 256: FA2 prefill kernel exceeds 32KB threadgroup limit
                // with Q_CHUNK=32. Compile at runtime with reduced tile size.
                let header = format!(
                    "#define HEAD_DIM 256\n#define HEAD_DIM_PACKED 128\n#define ROTARY_DIM {rotary_dim}\n"
                );
                let attn_src_raw = include_str!("shaders/attention.metal");
                let fused_qknr_src = include_str!("shaders/fused_qk_norm_rope.metal");
                let attn_src_patched = attn_src_raw
                    .replace(
                        "constant constexpr uint FA2_Q_CHUNK = 32;",
                        "constant constexpr uint FA2_Q_CHUNK = 8;",
                    )
                    .replace(
                        "constant constexpr uint FA2_KV_TILE = 32;",
                        "constant constexpr uint FA2_KV_TILE = 16;",
                    );
                let attn_src = format!("{header}{attn_src_patched}\n{fused_qknr_src}");
                let tq_helpers = include_str!("../shaders/turboquant_helpers.metal");
                let tq_src_raw = include_str!("shaders/turboquant.metal");
                let tq_src = format!("{header}{tq_helpers}\n{tq_src_raw}");
                // fused_sdpa exceeds 32KB threadgroup memory at head_dim=256.
                // Use an empty stub — decode uses FlashDecoding split+reduce instead.
                let sdpa = device
                    .compile_shader_source(&format!(
                        "{header}#include <metal_stdlib>\nusing namespace metal;\n"
                    ))
                    .map_err(MetalError::Metal)?;
                let fd_src_raw = include_str!("shaders/flash_decode.metal");
                let fd_src = format!("{header}#define SPLIT_BC 8\n{fd_src_raw}");

                let attn = device
                    .compile_shader_source(&attn_src)
                    .map_err(MetalError::Metal)?;
                let tq = device
                    .compile_shader_source(&tq_src)
                    .map_err(MetalError::Metal)?;
                // sdpa already compiled above as an empty stub
                let fd = device
                    .compile_shader_source(&fd_src)
                    .map_err(MetalError::Metal)?;
                Ok((attn, tq, sdpa, fd))
            }
            512 => try_precompiled(
                precompiled!(attn "512"),
                precompiled!(tq "512"),
                precompiled!(sdpa "512"),
                precompiled!(fd "512"),
            ),
            _ => {
                let header = format!(
                    "#define HEAD_DIM {head_dim}\n#define HEAD_DIM_PACKED {}\n#define ROTARY_DIM {rotary_dim}\n",
                    head_dim / 2
                );
                let attn_src_raw = include_str!("shaders/attention.metal");
                let fused_qk_src = include_str!("shaders/fused_qk_norm_rope.metal");
                let attn_src = format!("{header}{attn_src_raw}\n{fused_qk_src}");
                let tq_helpers = include_str!("../shaders/turboquant_helpers.metal");
                let tq_src_raw = include_str!("shaders/turboquant.metal");
                let tq_src = format!("{header}{tq_helpers}\n{tq_src_raw}");
                let sdpa_tile_defines = if head_dim >= 256 {
                    "#define SDPA_BR 4\n#define SDPA_BC 4\n"
                } else {
                    ""
                };
                let sdpa_src_raw = include_str!("shaders/fused_sdpa.metal");
                let sdpa_src = format!("{header}{sdpa_tile_defines}{sdpa_src_raw}");
                let fd_tile_defines = if head_dim >= 256 {
                    "#define SPLIT_BC 8\n"
                } else {
                    ""
                };
                let fd_src_raw = include_str!("shaders/flash_decode.metal");
                let fd_src = format!("{header}{fd_tile_defines}{fd_src_raw}");

                let attn = device
                    .compile_shader_source(&attn_src)
                    .map_err(MetalError::Metal)?;
                let tq = device
                    .compile_shader_source(&tq_src)
                    .map_err(MetalError::Metal)?;
                let sdpa = device
                    .compile_shader_source(&sdpa_src)
                    .map_err(MetalError::Metal)?;
                let fd = device
                    .compile_shader_source(&fd_src)
                    .map_err(MetalError::Metal)?;
                Ok((attn, tq, sdpa, fd))
            }
        }
    }

    // ── Phase-aware pipeline selection ───────────────────────────

    /// Select the FP16 dense linear pipeline (matvec or matmul) based on phase.
    #[inline]
    pub fn dense_linear_pipeline(&self, kind: LinearKernelKind) -> &ComputePipeline {
        match kind {
            LinearKernelKind::Matvec => &self.matvec,
            LinearKernelKind::Matmul => &self.matmul,
        }
    }

    /// Select the PolarQuant pipeline for a given bit-width and phase.
    #[inline]
    pub fn polarquant_pipeline(
        &self,
        n_bits: u32,
        kind: LinearKernelKind,
    ) -> Option<&ComputePipeline> {
        match (n_bits, kind) {
            (4, LinearKernelKind::Matvec) => Some(&self.polarquant_matvec_int4),
            (4, LinearKernelKind::Matmul) => Some(&self.polarquant_matmul_int4),
            (8, LinearKernelKind::Matvec) => Some(&self.polarquant_matvec_int8),
            (8, LinearKernelKind::Matmul) => Some(&self.polarquant_matmul_int8),
            _ => None,
        }
    }

    /// Select the affine-quantized pipeline for a given bit-width and phase.
    #[inline]
    pub fn affine_pipeline(
        &self,
        bit_width: u32,
        kind: LinearKernelKind,
    ) -> Option<&ComputePipeline> {
        match (bit_width, kind) {
            (4, LinearKernelKind::Matvec) => Some(&self.affine_matvec_int4),
            (4, LinearKernelKind::Matmul) => Some(&self.affine_matmul_int4),
            (8, LinearKernelKind::Matvec) => Some(&self.affine_matvec_int8),
            (8, LinearKernelKind::Matmul) => Some(&self.affine_matmul_int8),
            _ => None,
        }
    }

    /// Select the D2Quant dual-scale pipeline for a given bit-width and phase.
    #[inline]
    pub fn d2quant_pipeline(
        &self,
        bit_width: u32,
        kind: LinearKernelKind,
    ) -> Option<&ComputePipeline> {
        match (bit_width, kind) {
            (3, LinearKernelKind::Matvec) => Some(&self.d2quant_matvec_3bit),
            (3, LinearKernelKind::Matmul) => Some(&self.d2quant_matmul_3bit),
            _ => None,
        }
    }

    /// Select the QuIP# pipeline for the given phase.
    #[inline]
    pub fn quip_sharp_pipeline(&self, kind: LinearKernelKind) -> &ComputePipeline {
        match kind {
            LinearKernelKind::Matvec => &self.quip_sharp_matvec,
            LinearKernelKind::Matmul => &self.quip_sharp_matmul,
        }
    }
}

// ── Parameter structs ────────────────────────────────────────────

/// Parameters for [`encode_rms_norm`].
pub struct RmsNormParams<'a> {
    /// Input buffer.
    pub input: &'a MetalBuffer,
    /// Norm weight buffer.
    pub weight: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Hidden dimension size.
    pub hidden_size: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Epsilon for numerical stability.
    pub eps: f32,
}

/// Parameters for [`encode_rope`].
pub struct RopeParams<'a> {
    /// Q or K projection buffer (modified in-place).
    pub qk: &'a MetalBuffer,
    /// Cosine cache buffer.
    pub cos_cache: &'a MetalBuffer,
    /// Sine cache buffer.
    pub sin_cache: &'a MetalBuffer,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Sequence offset for position calculation.
    pub seq_offset: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
}

/// Parameters for [`encode_embedding_lookup`].
pub struct EmbeddingLookupParams<'a> {
    /// Token ID buffer.
    pub token_ids: &'a MetalBuffer,
    /// Embedding weight table buffer.
    pub embedding_table: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Hidden dimension size.
    pub hidden_size: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
}

/// Parameters for [`encode_d2quant_embedding_lookup`].
pub struct D2QuantEmbeddingLookupParams<'a> {
    /// Token ID buffer.
    pub token_ids: &'a MetalBuffer,
    /// D2Quant dual-scale quantized weight table.
    pub weight: &'a super::weights::DualScaleQuantizedWeight,
    /// Output buffer (FP16).
    pub output: &'a MetalBuffer,
    /// Hidden dimension size (K = num_layers * ple_hidden_size).
    pub hidden_size: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
}

/// Parameters for [`encode_standard_attention`].
pub struct StandardAttentionParams<'a> {
    /// Query projection buffer.
    pub q: &'a MetalBuffer,
    /// K cache buffer.
    pub k_cache: &'a MetalBuffer,
    /// V cache buffer.
    pub v_cache: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Number of query heads.
    pub num_heads: u32,
    /// Number of key-value heads.
    pub num_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Maximum sequence length (cache stride).
    pub max_seq_len: u32,
    /// Current sequence length.
    pub seq_len: u32,
}

/// Parameters for [`encode_prefill_attention`].
pub struct PrefillAttentionParams<'a> {
    /// Query projection buffer.
    pub q: &'a MetalBuffer,
    /// K cache buffer.
    pub k_cache: &'a MetalBuffer,
    /// V cache buffer.
    pub v_cache: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Number of query heads.
    pub num_heads: u32,
    /// Number of key-value heads.
    pub num_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Maximum sequence length (cache stride).
    pub max_seq_len: u32,
    /// Sequence offset (position of first token in batch).
    pub seq_offset: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Sliding window size (0 = full attention).
    pub window_size: u32,
    /// Attention scale factor (1/sqrt(head_dim) or 1.0 for QK-normed models).
    pub attn_scale: f32,
}

/// Parameters for [`encode_fused_sdpa`].
pub struct FusedSdpaParams<'a> {
    /// Query buffer.
    pub q: &'a MetalBuffer,
    /// Key buffer.
    pub k: &'a MetalBuffer,
    /// Value buffer.
    pub v: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Current sequence length.
    pub seq_len: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Number of query heads.
    pub num_q_heads: u32,
    /// Number of key-value heads.
    pub num_kv_heads: u32,
    /// Attention scale factor.
    pub scale: f32,
    /// Maximum sequence length (cache stride).
    pub max_seq_len: u32,
}

/// Parameters for [`encode_kv_scatter`].
pub struct KvScatterParams<'a> {
    /// Projection buffer to scatter into cache.
    pub proj: &'a MetalBuffer,
    /// KV cache buffer.
    pub cache: &'a MetalBuffer,
    /// Current sequence position.
    pub seq_pos: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
    /// Number of key-value heads.
    pub num_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Maximum sequence length (cache stride).
    pub max_seq_len: u32,
}

/// Parameters for [`encode_fused_residual_rms_norm`].
pub struct FusedResidualRmsNormParams<'a> {
    /// First input buffer for residual addition.
    pub a: &'a MetalBuffer,
    /// Second input buffer for residual addition.
    pub b: &'a MetalBuffer,
    /// Norm weight buffer.
    pub weight: &'a MetalBuffer,
    /// Normalized output buffer.
    pub normed_output: &'a MetalBuffer,
    /// Residual output buffer.
    pub residual_output: &'a MetalBuffer,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Hidden dimension size.
    pub hidden_size: u32,
    /// Number of tokens in the batch.
    pub token_count: u32,
}

// ── Dispatch helpers ─────────────────────────────────────────────
//
// These encode kernel dispatches into a ComputeEncoder without
// committing. All operations are batched into a single command
// buffer per token.

/// Encode an RMSNorm operation.
pub fn encode_rms_norm(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &RmsNormParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.input, 0, 0);
    encoder.set_buffer(params.weight, 0, 1);
    encoder.set_buffer(params.output, 0, 2);
    encoder.set_bytes(&params.hidden_size.to_le_bytes(), 3);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 4);
    encoder.set_bytes(&params.eps.to_le_bytes(), 5);
    // Cap threadgroup size to Metal's 1024-thread limit. The shader uses a
    // strided loop so it handles hidden_size > tg_size correctly.
    let tg_size = METAL_MAX_THREADS_PER_THREADGROUP.min(params.hidden_size as usize);
    encoder.dispatch_threadgroups((params.token_count as usize, 1, 1), (tg_size, 1, 1));
}

/// Encode SiLU-gated activation.
pub fn encode_silu_gate(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    gate: &MetalBuffer,
    up: &MetalBuffer,
    output: &MetalBuffer,
    size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(gate, 0, 0);
    encoder.set_buffer(up, 0, 1);
    encoder.set_buffer(output, 0, 2);
    encoder.set_bytes(&size.to_le_bytes(), 3);
    let threads = size as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode RoPE application.
pub fn encode_rope(encoder: &ComputeEncoder, pipeline: &ComputePipeline, params: &RopeParams<'_>) {
    debug_assert!(
        params.token_count == 0 || params.seq_offset.checked_add(params.token_count).is_some(),
        "rope: seq_offset + token_count overflows u32"
    );
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.qk, 0, 0);
    encoder.set_buffer(params.cos_cache, 0, 1);
    encoder.set_buffer(params.sin_cache, 0, 2);
    encoder.set_bytes(&params.num_heads.to_le_bytes(), 3);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 4);
    encoder.set_bytes(&params.seq_offset.to_le_bytes(), 5);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 6);
    let half_dim = params.head_dim / 2;
    encoder.dispatch_threads(
        (
            half_dim as usize,
            params.num_heads as usize,
            params.token_count as usize,
        ),
        ((half_dim as usize).min(DEFAULT_THREADGROUP_WIDTH), 1, 1),
    );
}

/// Encode residual add.
pub fn encode_residual_add(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    a: &MetalBuffer,
    b: &MetalBuffer,
    output: &MetalBuffer,
    size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(a, 0, 0);
    encoder.set_buffer(b, 0, 1);
    encoder.set_buffer(output, 0, 2);
    encoder.set_bytes(&size.to_le_bytes(), 3);
    let threads = size as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode broadcast bias add (D2Quant DAC):
/// `data[i] += bias[i % hidden_size]` for `total_size` elements.
///
/// Adds a per-channel correction bias `[hidden_size]` to a matrix
/// `[token_count × hidden_size]`, broadcasting across all tokens.
pub fn encode_bias_add(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    data: &MetalBuffer,
    bias: &MetalBuffer,
    hidden_size: u32,
    total_size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(data, 0, 0);
    encoder.set_buffer(bias, 0, 1);
    encoder.set_bytes(&hidden_size.to_le_bytes(), 2);
    encoder.set_bytes(&total_size.to_le_bytes(), 3);
    let threads = total_size as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode sigmoid gating (Qwen3.5 attn_output_gate):
/// `attn_out[i] *= sigmoid(gate[i])` for `size` elements.
pub fn encode_sigmoid_gate(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    attn_out: &MetalBuffer,
    gate: &MetalBuffer,
    size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(attn_out, 0, 0);
    encoder.set_buffer(gate, 0, 1);
    encoder.set_bytes(&size.to_le_bytes(), 2);
    let threads = size as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode in-place scalar multiply: data[i] *= scalar[0] for i in 0..count.
pub fn encode_scale_buffer(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    data: &MetalBuffer,
    scalar: &MetalBuffer,
    count: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(data, 0, 0);
    encoder.set_buffer(scalar, 0, 1);
    encoder.set_bytes(&count.to_le_bytes(), 2);
    let threads = count as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode buffer copy: `dst[i] = src[i]`.
pub fn encode_copy_buffer(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    src: &MetalBuffer,
    dst: &MetalBuffer,
    size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(src, 0, 0);
    encoder.set_buffer(dst, 0, 1);
    encoder.set_bytes(&size.to_le_bytes(), 2);
    let threads = size as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode embedding lookup.
pub fn encode_embedding_lookup(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &EmbeddingLookupParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.token_ids, 0, 0);
    encoder.set_buffer(params.embedding_table, 0, 1);
    encoder.set_buffer(params.output, 0, 2);
    encoder.set_bytes(&params.hidden_size.to_le_bytes(), 3);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 4);
    encoder.set_bytes(&params.vocab_size.to_le_bytes(), 5);
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(params.hidden_size as usize);
    encoder.dispatch_threadgroups(
        (
            (params.hidden_size as usize).div_ceil(tg_size),
            params.token_count as usize,
            1,
        ),
        (tg_size, 1, 1),
    );
}

/// Encode D2Quant 3-bit embedding lookup: gather rows from a packed
/// dual-scale quantized table and dequantize to FP16.
pub fn encode_d2quant_embedding_lookup(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &D2QuantEmbeddingLookupParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.token_ids, 0, 0);
    encoder.set_buffer(&params.weight.data, 0, 1);
    encoder.set_buffer(&params.weight.normal_scale, 0, 2);
    encoder.set_buffer(&params.weight.normal_zero, 0, 3);
    encoder.set_buffer(&params.weight.outlier_scale, 0, 4);
    encoder.set_buffer(&params.weight.outlier_zero, 0, 5);
    encoder.set_buffer(&params.weight.outlier_mask, 0, 6);
    encoder.set_buffer(params.output, 0, 7);
    encoder.set_bytes(&params.hidden_size.to_le_bytes(), 8);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 9);
    encoder.set_bytes(&params.vocab_size.to_le_bytes(), 10);
    encoder.set_bytes(&params.weight.group_size.to_le_bytes(), 11);
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(params.hidden_size as usize);
    encoder.dispatch_threadgroups(
        (
            (params.hidden_size as usize).div_ceil(tg_size),
            params.token_count as usize,
            1,
        ),
        (tg_size, 1, 1),
    );
}

/// Encode standard FP16 attention.
pub fn encode_standard_attention(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &StandardAttentionParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k_cache, 0, 1);
    encoder.set_buffer(params.v_cache, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.num_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 5);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 6);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 7);
    encoder.set_bytes(&params.seq_len.to_le_bytes(), 8);
    // HEAD_DIM is now exact — use at least ATTENTION_MIN_THREADGROUP_WIDTH
    // threads for better occupancy and parallel QK^T position processing.
    encoder.dispatch_threadgroups(
        (params.num_heads as usize, 1, 1),
        (
            ATTENTION_MIN_THREADGROUP_WIDTH
                .max(params.head_dim as usize)
                .min(METAL_MAX_THREADS_PER_THREADGROUP),
            1,
            1,
        ),
    );
}

/// Encode batched prefill attention — all query tokens in one dispatch.
pub fn encode_prefill_attention(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &PrefillAttentionParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k_cache, 0, 1);
    encoder.set_buffer(params.v_cache, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.num_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 5);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 6);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 7);
    encoder.set_bytes(&params.seq_offset.to_le_bytes(), 8);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 9);
    encoder.set_bytes(&params.window_size.to_le_bytes(), 10);
    encoder.set_bytes(&params.attn_scale.to_le_bytes(), 11);
    encoder.dispatch_threadgroups(
        (params.num_heads as usize, params.token_count as usize, 1),
        (
            ATTENTION_MIN_THREADGROUP_WIDTH
                .max(params.head_dim as usize)
                .min(METAL_MAX_THREADS_PER_THREADGROUP),
            1,
            1,
        ),
    );
}

/// FA2 Q-chunk sizes matching the compile-time constants in attention.metal.
fn fa2_q_chunk(head_dim: u32) -> u32 {
    if head_dim >= 512 {
        4
    } else if head_dim >= 256 {
        8
    } else {
        32
    }
}

/// Encode FlashAttention-2 style prefill attention.
///
/// Groups `FA2_Q_CHUNK` queries per threadgroup, loading each KV tile once
/// and reusing it across all queries. Better bandwidth utilization for
/// large models where KV tiles don't fit in L1 cache.
pub fn encode_fa2_prefill_attention(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &PrefillAttentionParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k_cache, 0, 1);
    encoder.set_buffer(params.v_cache, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.num_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 5);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 6);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 7);
    encoder.set_bytes(&params.seq_offset.to_le_bytes(), 8);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 9);
    encoder.set_bytes(&params.window_size.to_le_bytes(), 10);
    encoder.set_bytes(&params.attn_scale.to_le_bytes(), 11);
    let q_chunk = fa2_q_chunk(params.head_dim) as usize;
    let q_blocks = (params.token_count as usize).div_ceil(q_chunk);
    encoder.dispatch_threadgroups(
        (params.num_heads as usize, q_blocks, 1),
        (
            ATTENTION_MIN_THREADGROUP_WIDTH
                .max(params.head_dim as usize)
                .min(METAL_MAX_THREADS_PER_THREADGROUP),
            1,
            1,
        ),
    );
}

/// V2 query block size matching the compile-time constant in attention.metal.
fn v2_br(head_dim: u32) -> u32 {
    if head_dim >= 256 { 16 } else { 32 }
}

/// Encode register-tiled FA2 prefill attention (v2).
///
/// Combines cooperative KV tile loading (amortized across GQA group) with
/// register-based accumulation. Dispatches one threadgroup per
/// (kv_group, q_block), with one simdgroup per Q head in the group.
pub fn encode_v2_prefill_attention(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &PrefillAttentionParams<'_>,
) {
    assert!(params.num_kv_heads > 0, "num_kv_heads must be > 0");
    assert!(
        params.num_heads % params.num_kv_heads == 0,
        "num_heads must be divisible by num_kv_heads"
    );
    let heads_per_group = (params.num_heads / params.num_kv_heads) as usize;
    let num_kv_groups = params.num_kv_heads as usize;
    let br = v2_br(params.head_dim) as usize;
    let q_blocks = (params.token_count as usize).div_ceil(br);

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k_cache, 0, 1);
    encoder.set_buffer(params.v_cache, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.num_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 5);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 6);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 7);
    encoder.set_bytes(&params.seq_offset.to_le_bytes(), 8);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 9);
    encoder.set_bytes(&params.window_size.to_le_bytes(), 10);
    encoder.set_bytes(&params.attn_scale.to_le_bytes(), 11);
    let threads_per_tg = (heads_per_group * 32).min(METAL_MAX_THREADS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((num_kv_groups, q_blocks, 1), (threads_per_tg, 1, 1));
}

/// Default Q-block tile size (Br) for fused SDPA.
const FUSED_SDPA_DEFAULT_BR: usize = 32;

/// Encode fused scaled dot-product attention.
///
/// Handles both prefill (token_count > 1) and decode (token_count == 1).
/// Dispatches one threadgroup per (kv_head_group, q_block), with one
/// simdgroup per Q head within each group.
pub fn encode_fused_sdpa(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &FusedSdpaParams<'_>,
    tile_br: Option<usize>,
) {
    let br = tile_br.unwrap_or(FUSED_SDPA_DEFAULT_BR);
    let heads_per_group = (params.num_q_heads / params.num_kv_heads) as usize;

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k, 0, 1);
    encoder.set_buffer(params.v, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.seq_len.to_le_bytes(), 4);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 5);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 6);
    encoder.set_bytes(&params.num_q_heads.to_le_bytes(), 7);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 8);
    encoder.set_bytes(&params.scale.to_le_bytes(), 9);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 10);

    let num_kv_groups = params.num_kv_heads as usize;
    let q_blocks = (params.token_count as usize).div_ceil(br);
    // One simdgroup (32 threads) per Q head in the group.
    assert!(
        heads_per_group <= 32,
        "fused SDPA: GQA ratio {heads_per_group}:1 exceeds max 32 simdgroups per threadgroup"
    );
    let threads_per_tg = (heads_per_group * 32).min(METAL_MAX_THREADS_PER_THREADGROUP);

    encoder.dispatch_threadgroups((num_kv_groups, q_blocks, 1), (threads_per_tg, 1, 1));
}

/// Minimum sequence length to trigger FlashDecoding split.
/// Below this, the original single-pass kernel is used.
const FLASH_DECODE_MIN_SEQ: usize = 256;

/// KV positions per split — each split gets a chunk of the sequence.
/// 256 gives good granularity: at 16K context with 256 per split = 64 splits.
const FLASH_DECODE_KV_PER_SPLIT: usize = 256;

/// Encode FlashDecoding++: persistent split kernel with max hint, then reduce.
///
/// Improvements over basic FlashDecoding:
/// - Persistent threadgroups: launch fewer TGs that each loop over multiple
///   KV chunks, improving cache locality and reducing dispatch overhead.
/// - Unified max hint: passes the previous step's softmax max to the split
///   kernel so the O accumulator avoids rescaling when max hasn't changed.
/// - Reduce kernel writes back the global max for the next step's hint.
pub fn encode_flash_decode(
    encoder: &ComputeEncoder,
    split_pipeline: &ComputePipeline,
    reduce_pipeline: &ComputePipeline,
    sdpa_fallback: Option<&ComputePipeline>,
    params: &FusedSdpaParams<'_>,
    partial_o: &MetalBuffer,
    partial_max: &MetalBuffer,
    partial_sum: &MetalBuffer,
    max_hint: &MetalBuffer,
    max_splits: usize,
    gpu_max_threadgroups: usize,
) {
    let seq_len = params.seq_len as usize;

    // Short sequences: use single-pass if available, otherwise still split.
    if seq_len < FLASH_DECODE_MIN_SEQ {
        if let Some(sdpa) = sdpa_fallback {
            encode_fused_sdpa(encoder, sdpa, params, None);
            return;
        }
        // No fallback — use split+reduce even for short sequences.
    }

    let heads_per_group = (params.num_q_heads as usize) / (params.num_kv_heads as usize).max(1);
    let num_kv_groups = params.num_kv_heads as usize;
    let target_splits = (gpu_max_threadgroups / num_kv_groups.max(1)).max(2);
    let splits_from_seq = seq_len.div_ceil(FLASH_DECODE_KV_PER_SPLIT);
    let num_splits = target_splits.min(splits_from_seq).min(max_splits).max(2);

    if num_splits <= 1 {
        if let Some(sdpa) = sdpa_fallback {
            encode_fused_sdpa(encoder, sdpa, params, None);
            return;
        }
    }

    // Persistent split kernel: grid Y = min(num_splits, target_persistent_tgs).
    let persistent_tgs = num_splits.min(target_splits);
    let threads_per_tg = (heads_per_group * 32).min(METAL_MAX_THREADS_PER_THREADGROUP);

    encoder.set_pipeline(split_pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k, 0, 1);
    encoder.set_buffer(params.v, 0, 2);
    encoder.set_buffer(partial_o, 0, 3);
    encoder.set_buffer(partial_max, 0, 4);
    encoder.set_buffer(partial_sum, 0, 5);
    encoder.set_bytes(&params.seq_len.to_le_bytes(), 6);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 7);
    encoder.set_bytes(&params.num_q_heads.to_le_bytes(), 8);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 9);
    encoder.set_bytes(&params.scale.to_le_bytes(), 10);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 11);
    encoder.set_bytes(&(num_splits as u32).to_le_bytes(), 12);
    // Pass -INFINITY as initial max_hint (safe default; per-head hint via
    // buffer would be more accurate but requires an extra buffer binding).
    let neg_inf = f32::NEG_INFINITY;
    encoder.set_bytes(&neg_inf.to_le_bytes(), 13);
    encoder.dispatch_threadgroups((num_kv_groups, persistent_tgs, 1), (threads_per_tg, 1, 1));

    // Barrier between split and reduce.
    encoder.memory_barrier_with_resources(&[partial_o, partial_max, partial_sum]);

    // Reduce kernel: grid (num_q_heads, 1, 1)
    let num_q = params.num_q_heads as usize;
    encoder.set_pipeline(reduce_pipeline);
    encoder.set_buffer(partial_o, 0, 0);
    encoder.set_buffer(partial_max, 0, 1);
    encoder.set_buffer(partial_sum, 0, 2);
    encoder.set_buffer(params.output, 0, 3);
    encoder.set_bytes(&params.num_q_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 5);
    encoder.set_bytes(&(num_splits as u32).to_le_bytes(), 6);
    encoder.set_buffer(max_hint, 0, 7);
    encoder.dispatch_threadgroups((num_q, 1, 1), (32, 1, 1));
}

/// Encode KV scatter — copy projections into FP16 KV cache on GPU.
pub fn encode_kv_scatter(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &KvScatterParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.proj, 0, 0);
    encoder.set_buffer(params.cache, 0, 1);
    encoder.set_bytes(&params.seq_pos.to_le_bytes(), 2);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 3);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 4);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 5);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 6);
    let tg_x = (params.head_dim as usize).min(DEFAULT_THREADGROUP_WIDTH);
    encoder.dispatch_threads(
        (
            params.head_dim as usize,
            params.num_kv_heads as usize,
            params.token_count as usize,
        ),
        (tg_x, 1, 1),
    );
}

/// Encode a custom FP16 matvec: y = x · W^T for M=1 (decode).
///
/// Weights must be pre-packed into blocked [N/8, K/8, 8, 8] FP16 format.
/// Dispatch: one threadgroup per [`MATVEC_ROWS_PER_THREADGROUP`] output rows,
/// [`DEFAULT_THREADGROUP_WIDTH`] threads (8 simdgroups).
pub fn encode_matvec(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input: &MetalBuffer,
    weight_packed: &MetalBuffer,
    output: &MetalBuffer,
    n: u32,
    k: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(weight_packed, 0, 1);
    encoder.set_buffer(output, 0, 2);
    encoder.set_bytes(&n.to_le_bytes(), 3);
    encoder.set_bytes(&k.to_le_bytes(), 4);
    let tg_count = (n as usize).div_ceil(MATVEC_ROWS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (DEFAULT_THREADGROUP_WIDTH, 1, 1));
}

/// Encode a custom FP16 matmul: C = A · W^T for M>1 (prefill).
///
/// Weights must be pre-packed into blocked [N/8, K/8, 8, 8] FP16 format.
/// Dispatch: 2-D grid of threadgroups tiling M and N in 64-element blocks,
/// [`DEFAULT_THREADGROUP_WIDTH`] threads (8 simdgroups).
#[allow(clippy::too_many_arguments)]
pub fn encode_matmul(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input: &MetalBuffer,
    weight_packed: &MetalBuffer,
    output: &MetalBuffer,
    m: u32,
    n: u32,
    k: u32,
) {
    const TM_TILE: usize = 64;
    const TN_TILE: usize = 64;

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(weight_packed, 0, 1);
    encoder.set_buffer(output, 0, 2);
    encoder.set_bytes(&m.to_le_bytes(), 3);
    encoder.set_bytes(&n.to_le_bytes(), 4);
    encoder.set_bytes(&k.to_le_bytes(), 5);

    let tg_m = (m as usize).div_ceil(TM_TILE);
    let tg_n = (n as usize).div_ceil(TN_TILE);
    encoder.dispatch_threadgroups((tg_m, tg_n, 1), (DEFAULT_THREADGROUP_WIDTH, 1, 1));
}

/// Encode a fused residual-add + RMSNorm operation.
///
/// Computes `residual = a + b` and `normed = rms_norm(residual, weight)` in a
/// single kernel dispatch, avoiding the intermediate global-memory round-trip
/// that two separate dispatches would require.
pub fn encode_fused_residual_rms_norm(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &FusedResidualRmsNormParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.a, 0, 0);
    encoder.set_buffer(params.b, 0, 1);
    encoder.set_buffer(params.weight, 0, 2);
    encoder.set_buffer(params.normed_output, 0, 3);
    encoder.set_buffer(params.residual_output, 0, 4);
    encoder.set_bytes(&params.eps.to_le_bytes(), 5);
    encoder.set_bytes(&params.hidden_size.to_le_bytes(), 6);
    encoder.set_bytes(&params.token_count.to_le_bytes(), 7);
    let tg_size = METAL_MAX_THREADS_PER_THREADGROUP.min(params.hidden_size as usize);
    encoder.dispatch_threadgroups((params.token_count as usize, 1, 1), (tg_size, 1, 1));
}

// ── GDN kernel dispatches ────────────────────────────────────────

/// Encode GDN conv1d + SiLU.
///
/// Shifts conv_state, computes causal conv1d dot product, applies SiLU.
/// One thread per channel (qkv_dim threads).
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_conv1d_silu(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input_qkv: &MetalBuffer,
    conv_weight: &MetalBuffer,
    conv_state: &MetalBuffer,
    output: &MetalBuffer,
    qkv_dim: u32,
    kernel_size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input_qkv, 0, 0);
    encoder.set_buffer(conv_weight, 0, 1);
    encoder.set_buffer(conv_state, 0, 2);
    encoder.set_buffer(output, 0, 3);
    let params: [u32; 4] = [qkv_dim, kernel_size, kernel_size - 1, 0];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 4);
    let threads = qkv_dim as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode fused logit softcapping: `data[i] = softcap * tanh(data[i] / softcap)`.
pub fn encode_fused_softcap(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    data: &MetalBuffer,
    softcap: f32,
    count: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(data, 0, 0);
    encoder.set_bytes(&softcap.to_le_bytes(), 1);
    encoder.set_bytes(&count.to_le_bytes(), 2);
    let threads = count as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode GDN recurrent state update.
///
/// Per-head: compute gates, update state matrix S, compute o = S @ q.
/// One threadgroup per head, v_head_dim threads per threadgroup.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_recurrent_update(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    conv_out: &MetalBuffer,
    a_proj: &MetalBuffer,
    b_proj: &MetalBuffer,
    a_log: &MetalBuffer,
    dt_bias: &MetalBuffer,
    recurrent_state: &MetalBuffer,
    output: &MetalBuffer,
    key_dim: u32,
    value_dim: u32,
    num_v_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    num_k_heads: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(conv_out, 0, 0);
    encoder.set_buffer(a_proj, 0, 1);
    encoder.set_buffer(b_proj, 0, 2);
    encoder.set_buffer(a_log, 0, 3);
    encoder.set_buffer(dt_bias, 0, 4);
    encoder.set_buffer(recurrent_state, 0, 5);
    encoder.set_buffer(output, 0, 6);
    let params: [u32; 6] = [
        key_dim,
        value_dim,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        num_k_heads,
    ];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 7);
    let tg_size = (v_head_dim as usize).min(METAL_MAX_THREADS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((num_v_heads as usize, 1, 1), (tg_size, 1, 1));
}

/// Encode GDN output gating: per-head RMSNorm + silu(z) multiplication.
///
/// One threadgroup per head, v_head_dim threads per threadgroup.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_output_gate(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    raw_output: &MetalBuffer,
    z: &MetalBuffer,
    norm_weight: &MetalBuffer,
    output: &MetalBuffer,
    num_v_heads: u32,
    v_head_dim: u32,
    eps: f32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(raw_output, 0, 0);
    encoder.set_buffer(z, 0, 1);
    encoder.set_buffer(norm_weight, 0, 2);
    encoder.set_buffer(output, 0, 3);
    let params: [u32; 4] = [num_v_heads, v_head_dim, eps.to_bits(), 0];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 4);
    let tg_size = (v_head_dim as usize).min(METAL_MAX_THREADS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((num_v_heads as usize, 1, 1), (tg_size, 1, 1));
}

/// Encode fused GDN decode: conv1d+SiLU + recurrent update + output gate
/// in a single dispatch. Saves 2 dispatches + 2 barriers per GDN layer.
///
/// One threadgroup per value head. Threads cooperate on conv1d channels,
/// then each thread handles one row of the recurrent state matrix.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_fused_decode(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input_qkv: &MetalBuffer,
    conv_weight: &MetalBuffer,
    conv_state: &MetalBuffer,
    a_proj: &MetalBuffer,
    b_proj: &MetalBuffer,
    a_log: &MetalBuffer,
    dt_bias: &MetalBuffer,
    recurrent_state: &MetalBuffer,
    z_proj: &MetalBuffer,
    norm_weight: &MetalBuffer,
    output: &MetalBuffer,
    conv_out_scratch: &MetalBuffer,
    qkv_dim: u32,
    kernel_size: u32,
    key_dim: u32,
    value_dim: u32,
    num_v_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    num_k_heads: u32,
    eps: f32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input_qkv, 0, 0);
    encoder.set_buffer(conv_weight, 0, 1);
    encoder.set_buffer(conv_state, 0, 2);
    encoder.set_buffer(a_proj, 0, 3);
    encoder.set_buffer(b_proj, 0, 4);
    encoder.set_buffer(a_log, 0, 5);
    encoder.set_buffer(dt_bias, 0, 6);
    encoder.set_buffer(recurrent_state, 0, 7);
    encoder.set_buffer(z_proj, 0, 8);
    encoder.set_buffer(norm_weight, 0, 9);
    encoder.set_buffer(output, 0, 10);
    encoder.set_buffer(conv_out_scratch, 0, 11);
    let params: [u32; 10] = [
        qkv_dim,
        kernel_size,
        kernel_size - 1, // conv_state_len
        key_dim,
        value_dim,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        num_k_heads,
        eps.to_bits(),
    ];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 12);
    // Thread count: need enough for both conv1d (qkv_dim/num_v_heads channels)
    // and recurrent (v_head_dim rows). Use the larger of the two.
    let channels_per_head = qkv_dim.div_ceil(num_v_heads) as usize;
    let tg_size = channels_per_head
        .max(v_head_dim as usize)
        .min(METAL_MAX_THREADS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((num_v_heads as usize, 1, 1), (tg_size, 1, 1));
}

/// Encode batched dense FP16 matvec for 4 GDN projections in a single dispatch.
///
/// Computes y_i = x · W_i^T for i in {QKV, Z, A, B}. All share the same input x.
/// Saves 3 dispatches per GDN layer compared to 4 separate `encode_matvec` calls.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_batched_matvec(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input: &MetalBuffer,
    w_qkv: &MetalBuffer,
    w_z: &MetalBuffer,
    w_a: &MetalBuffer,
    w_b: &MetalBuffer,
    y_qkv: &MetalBuffer,
    y_z: &MetalBuffer,
    y_a: &MetalBuffer,
    y_b: &MetalBuffer,
    k: u32,
    n_qkv: u32,
    n_z: u32,
    n_a: u32,
    n_b: u32,
) {
    const ROWS_PER_TG: u32 = 64;
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(w_qkv, 0, 1);
    encoder.set_buffer(w_z, 0, 2);
    encoder.set_buffer(w_a, 0, 3);
    encoder.set_buffer(w_b, 0, 4);
    encoder.set_buffer(y_qkv, 0, 5);
    encoder.set_buffer(y_z, 0, 6);
    encoder.set_buffer(y_a, 0, 7);
    encoder.set_buffer(y_b, 0, 8);
    let params: [u32; 6] = [k, n_qkv, n_z, n_a, n_b, 0];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 9);
    let tg_count = n_qkv.div_ceil(ROWS_PER_TG)
        + n_z.div_ceil(ROWS_PER_TG)
        + n_a.div_ceil(ROWS_PER_TG)
        + n_b.div_ceil(ROWS_PER_TG);
    encoder.dispatch_threadgroups((tg_count as usize, 1, 1), (DEFAULT_THREADGROUP_WIDTH, 1, 1));
}

/// Encode batched affine INT4 matvec for FFN gate+up in a single dispatch.
///
/// Computes gate = x · W_gate^T and up = x · W_up^T concurrently.
/// Saves 1 dispatch per layer compared to 2 separate `affine_matvec_int4` calls.
#[allow(clippy::too_many_arguments)]
pub fn encode_batched_affine_matvec_int4(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input: &MetalBuffer,
    gate_weight: &super::weights::AffineQuantizedWeight,
    gate_output: &MetalBuffer,
    up_weight: &super::weights::AffineQuantizedWeight,
    up_output: &MetalBuffer,
    n_gate: u32,
    k: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&gate_weight.data, 0, 1);
    encoder.set_buffer(&gate_weight.scales, 0, 2);
    encoder.set_buffer(&gate_weight.zeros, 0, 3);
    encoder.set_buffer(gate_output, 0, 4);
    encoder.set_buffer(&up_weight.data, 0, 5);
    encoder.set_buffer(&up_weight.scales, 0, 6);
    encoder.set_buffer(&up_weight.zeros, 0, 7);
    encoder.set_buffer(up_output, 0, 8);
    encoder.set_bytes(&n_gate.to_le_bytes(), 9);
    encoder.set_bytes(&k.to_le_bytes(), 10);
    encoder.set_bytes(&gate_weight.group_size.to_le_bytes(), 11);
    // AWQ scales: use gate_weight's (shared between gate/up)
    if let Some(ref awq) = gate_weight.awq_scales {
        encoder.set_buffer(awq, 0, 12);
        encoder.set_bytes(&1u32.to_le_bytes(), 13);
    } else {
        encoder.set_buffer(&gate_weight.data, 0, 12); // dummy
        encoder.set_bytes(&0u32.to_le_bytes(), 13);
    }
    let tg_count = (2 * n_gate) as usize;
    encoder.dispatch_threadgroups((tg_count, 1, 1), (32, 1, 1));
}

/// Encode fused FFN gate+up+activation for INT4 decode.
///
/// Computes output[i] = activation(x · W_gate^T[i]) * (x · W_up^T[i]) in one dispatch.
/// Eliminates gate/up intermediate writes + the separate activation dispatch.
/// Saves 1 dispatch + 1 barrier per layer compared to batched_affine_matvec_int4 + silu_gate.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_ffn_gate_up_act_int4(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input: &MetalBuffer,
    gate_weight: &super::weights::AffineQuantizedWeight,
    up_weight: &super::weights::AffineQuantizedWeight,
    output: &MetalBuffer,
    n: u32,
    k: u32,
    use_gelu: bool,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&gate_weight.data, 0, 1);
    encoder.set_buffer(&gate_weight.scales, 0, 2);
    encoder.set_buffer(&gate_weight.zeros, 0, 3);
    encoder.set_buffer(&up_weight.data, 0, 4);
    encoder.set_buffer(&up_weight.scales, 0, 5);
    encoder.set_buffer(&up_weight.zeros, 0, 6);
    encoder.set_buffer(output, 0, 7);
    encoder.set_bytes(&n.to_le_bytes(), 8);
    encoder.set_bytes(&k.to_le_bytes(), 9);
    encoder.set_bytes(&gate_weight.group_size.to_le_bytes(), 10);
    if let Some(ref awq) = gate_weight.awq_scales {
        encoder.set_buffer(awq, 0, 11);
        encoder.set_bytes(&1u32.to_le_bytes(), 12);
    } else {
        encoder.set_buffer(&gate_weight.data, 0, 11); // dummy
        encoder.set_bytes(&0u32.to_le_bytes(), 12);
    }
    encoder.set_bytes(&(use_gelu as u32).to_le_bytes(), 13);
    encoder.dispatch_threadgroups((n as usize, 1, 1), (32, 1, 1));
}
///
/// Computes qkv = x·W0^T, z = x·W1^T, a = x·W2^T, b = x·W3^T concurrently.
/// Saves 3 dispatches per GDN layer compared to 4 separate `affine_matvec_int4` calls.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_batched_affine_matvec_int4(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input: &MetalBuffer,
    w0: &super::weights::AffineQuantizedWeight,
    out0: &MetalBuffer,
    n0: u32,
    w1: &super::weights::AffineQuantizedWeight,
    out1: &MetalBuffer,
    n1: u32,
    w2: &super::weights::AffineQuantizedWeight,
    out2: &MetalBuffer,
    n2: u32,
    w3: &super::weights::AffineQuantizedWeight,
    out3: &MetalBuffer,
    n3: u32,
    k: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&w0.data, 0, 1);
    encoder.set_buffer(&w0.scales, 0, 2);
    encoder.set_buffer(&w0.zeros, 0, 3);
    encoder.set_buffer(out0, 0, 4);
    encoder.set_buffer(&w1.data, 0, 5);
    encoder.set_buffer(&w1.scales, 0, 6);
    encoder.set_buffer(&w1.zeros, 0, 7);
    encoder.set_buffer(out1, 0, 8);
    encoder.set_buffer(&w2.data, 0, 9);
    encoder.set_buffer(&w2.scales, 0, 10);
    encoder.set_buffer(&w2.zeros, 0, 11);
    encoder.set_buffer(out2, 0, 12);
    encoder.set_buffer(&w3.data, 0, 13);
    encoder.set_buffer(&w3.scales, 0, 14);
    encoder.set_buffer(&w3.zeros, 0, 15);
    encoder.set_buffer(out3, 0, 16);
    let has_awq = w0.awq_scales.is_some() as u32;
    let params_words: [u32; 7] = [n0, n1, n2, n3, k, w0.group_size, has_awq];
    let params_bytes: Vec<u8> = params_words.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 17);
    if let Some(ref awq) = w0.awq_scales {
        encoder.set_buffer(awq, 0, 18);
    } else {
        encoder.set_buffer(&w0.data, 0, 18); // dummy
    }
    let tg_count = (n0 + n1 + n2 + n3) as usize;
    encoder.dispatch_threadgroups((tg_count, 1, 1), (32, 1, 1));
}

/// Encode fused residual + RMSNorm + dense FP16 matvec.
///
/// Replaces `encode_fused_residual_rms_norm` + barrier + `encode_matvec`.
/// Each threadgroup computes (a+b), derives rms_inv, and performs the
/// matvec using the normed input — all in one dispatch.
/// Threadgroup 0 writes `residual_output = a + b` and `normed_output = RMSNorm(a+b)`.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_residual_norm_matvec(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    a: &MetalBuffer,
    b: &MetalBuffer,
    norm_weight: &MetalBuffer,
    residual_output: &MetalBuffer,
    w_packed: &MetalBuffer,
    y: &MetalBuffer,
    normed_output: &MetalBuffer,
    k: u32,
    n: u32,
    eps: f32,
) {
    const ROWS_PER_TG: u32 = 64;
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(a, 0, 0);
    encoder.set_buffer(b, 0, 1);
    encoder.set_buffer(norm_weight, 0, 2);
    encoder.set_buffer(residual_output, 0, 3);
    encoder.set_buffer(w_packed, 0, 4);
    encoder.set_buffer(y, 0, 5);
    let params: [u32; 4] = [k, n, eps.to_bits(), 0];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 6);
    encoder.set_buffer(normed_output, 0, 7);
    let tg_count = n.div_ceil(ROWS_PER_TG);
    encoder.dispatch_threadgroups((tg_count as usize, 1, 1), (DEFAULT_THREADGROUP_WIDTH, 1, 1));
}

/// Encode fused residual + RMSNorm + affine INT4 matvec.
///
/// Same fusion as `encode_fused_residual_norm_matvec` but for INT4 weights.
/// One threadgroup per output row, 32 threads per group.
/// Also writes `normed_output` for subsequent projections.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_residual_norm_affine_matvec_int4(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    a: &MetalBuffer,
    b: &MetalBuffer,
    norm_weight: &MetalBuffer,
    residual_output: &MetalBuffer,
    weight: &super::weights::AffineQuantizedWeight,
    output: &MetalBuffer,
    normed_output: &MetalBuffer,
    n: u32,
    k: u32,
    eps: f32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(a, 0, 0);
    encoder.set_buffer(b, 0, 1);
    encoder.set_buffer(norm_weight, 0, 2);
    encoder.set_buffer(residual_output, 0, 3);
    encoder.set_buffer(&weight.data, 0, 4);
    encoder.set_buffer(&weight.scales, 0, 5);
    encoder.set_buffer(&weight.zeros, 0, 6);
    encoder.set_buffer(output, 0, 7);
    let params: [u32; 4] = [n, k, weight.group_size, eps.to_bits()];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 8);
    if let Some(ref awq) = weight.awq_scales {
        encoder.set_buffer(awq, 0, 9);
        encoder.set_bytes(&1u32.to_le_bytes(), 10);
    } else {
        encoder.set_buffer(&weight.data, 0, 9); // dummy
        encoder.set_bytes(&0u32.to_le_bytes(), 10);
    }
    encoder.set_buffer(normed_output, 0, 11);
    encoder.dispatch_threadgroups((n as usize, 1, 1), (32, 1, 1));
}

/// Encode GDN prefill batched conv1d + SiLU.
///
/// Processes ALL tokens sequentially per channel in a single dispatch.
/// One thread per channel (qkv_dim threads total).
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_prefill_conv1d_silu(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    all_qkv: &MetalBuffer,
    conv_weight: &MetalBuffer,
    conv_state: &MetalBuffer,
    all_conv_out: &MetalBuffer,
    qkv_dim: u32,
    kernel_size: u32,
    token_count: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(all_qkv, 0, 0);
    encoder.set_buffer(conv_weight, 0, 1);
    encoder.set_buffer(conv_state, 0, 2);
    encoder.set_buffer(all_conv_out, 0, 3);
    let params: [u32; 4] = [qkv_dim, kernel_size, kernel_size - 1, token_count];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 4);
    let threads = qkv_dim as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode GELU-gated element-wise multiply with strided input access:
///   `output[i] = gelu(gate[i]) * input_slice[token, layer_offset + elem]`.
///
/// The `input` buffer has row stride `input_stride` elements and each layer's
/// slice starts at column `input_offset`.
#[allow(clippy::too_many_arguments)]
pub fn encode_gelu_gate(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    gate: &MetalBuffer,
    input: &MetalBuffer,
    output: &MetalBuffer,
    ple_hidden: u32,
    token_count: u32,
    input_stride: u32,
    input_offset: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(gate, 0, 0);
    encoder.set_buffer(input, 0, 1);
    encoder.set_buffer(output, 0, 2);
    encoder.set_bytes(&ple_hidden.to_le_bytes(), 3);
    encoder.set_bytes(&token_count.to_le_bytes(), 4);
    encoder.set_bytes(&input_stride.to_le_bytes(), 5);
    encoder.set_bytes(&input_offset.to_le_bytes(), 6);
    let threads = (token_count * ple_hidden) as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode add-and-scale: `output[i] = (a[i] + b[i]) * scale`.
pub fn encode_add_scale(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    a: &MetalBuffer,
    b: &MetalBuffer,
    output: &MetalBuffer,
    size: u32,
    scale: f32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(a, 0, 0);
    encoder.set_buffer(b, 0, 1);
    encoder.set_buffer(output, 0, 2);
    encoder.set_bytes(&size.to_le_bytes(), 3);
    encoder.set_bytes(&scale.to_le_bytes(), 4);
    let threads = size as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

// ── MoE kernel dispatch helpers ─────────────────────────────────

/// Encode row-wise softmax on `[token_count, width]` shaped data (in-place).
pub fn encode_moe_softmax(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    data: &MetalBuffer,
    width: u32,
    token_count: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(data, 0, 0);
    encoder.set_bytes(&width.to_le_bytes(), 1);
    encoder.set_bytes(&token_count.to_le_bytes(), 2);
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(width as usize);
    encoder.dispatch_threadgroups((token_count as usize, 1, 1), (tg_size, 1, 1));
}

/// Encode in-place GELU activation.
pub fn encode_moe_gelu(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    data: &MetalBuffer,
    size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(data, 0, 0);
    encoder.set_bytes(&size.to_le_bytes(), 1);
    let threads = size as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode in-place element-wise multiply: `gate[i] *= up[i]`.
pub fn encode_moe_mul(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    gate: &MetalBuffer,
    up: &MetalBuffer,
    size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(gate, 0, 0);
    encoder.set_buffer(up, 0, 1);
    encoder.set_bytes(&size.to_le_bytes(), 2);
    let threads = size as usize;
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode MoE top-k weighted combine of expert outputs.
#[allow(clippy::too_many_arguments)]
pub fn encode_moe_weighted_combine(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    router_probs: &MetalBuffer,
    expert_outputs: &MetalBuffer,
    output: &MetalBuffer,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    token_count: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(router_probs, 0, 0);
    encoder.set_buffer(expert_outputs, 0, 1);
    encoder.set_buffer(output, 0, 2);
    encoder.set_bytes(&num_experts.to_le_bytes(), 3);
    encoder.set_bytes(&top_k.to_le_bytes(), 4);
    encoder.set_bytes(&hidden_size.to_le_bytes(), 5);
    encoder.set_bytes(&token_count.to_le_bytes(), 6);
    let tg_size = DEFAULT_THREADGROUP_WIDTH.min(hidden_size as usize);
    encoder.dispatch_threadgroups((token_count as usize, 1, 1), (tg_size, 1, 1));
}

/// Encode GDN prefill batched recurrent + RMSNorm + silu(z) gating.
///
/// Processes ALL tokens sequentially per head in a single dispatch.
/// One threadgroup per head, v_head_dim threads per threadgroup.
#[allow(clippy::too_many_arguments)]
pub fn encode_gdn_prefill_recurrent(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    all_conv_out: &MetalBuffer,
    all_a: &MetalBuffer,
    all_b: &MetalBuffer,
    a_log: &MetalBuffer,
    dt_bias: &MetalBuffer,
    norm_weight: &MetalBuffer,
    all_z: &MetalBuffer,
    recurrent_state: &MetalBuffer,
    all_output: &MetalBuffer,
    token_count: u32,
    qkv_dim: u32,
    key_dim: u32,
    value_dim: u32,
    num_v_heads: u32,
    k_head_dim: u32,
    v_head_dim: u32,
    eps: f32,
    num_k_heads: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(all_conv_out, 0, 0);
    encoder.set_buffer(all_a, 0, 1);
    encoder.set_buffer(all_b, 0, 2);
    encoder.set_buffer(a_log, 0, 3);
    encoder.set_buffer(dt_bias, 0, 4);
    encoder.set_buffer(norm_weight, 0, 5);
    encoder.set_buffer(all_z, 0, 6);
    encoder.set_buffer(recurrent_state, 0, 7);
    encoder.set_buffer(all_output, 0, 8);
    let params: [u32; 9] = [
        token_count,
        qkv_dim,
        key_dim,
        value_dim,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        eps.to_bits(),
        num_k_heads,
    ];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 9);
    let tg_size = (v_head_dim as usize).min(METAL_MAX_THREADS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((num_v_heads as usize, 1, 1), (tg_size, 1, 1));
}

// ── Q8 input quantization dispatch ──────────────────────────────

/// Encode Q8 input quantization: FP16 → INT8 with per-group scale factors.
///
/// One dispatch quantizes the full input vector. The result is reused by
/// all subsequent INT4×Q8 projections reading the same input.
pub fn encode_quantize_input_q8(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input: &MetalBuffer,
    q8_data: &MetalBuffer,
    q8_scales: &MetalBuffer,
    k: u32,
    group_size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(q8_data, 0, 1);
    encoder.set_buffer(q8_scales, 0, 2);
    encoder.set_bytes(&k.to_le_bytes(), 3);
    encoder.set_bytes(&group_size.to_le_bytes(), 4);
    let num_groups = k.div_ceil(group_size) as usize;
    let tg_size = (group_size as usize).min(METAL_MAX_THREADS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((num_groups, 1, 1), (tg_size, 1, 1));
}

/// Encode INT4×Q8 integer dot product matvec for decode.
///
/// Uses pre-quantized INT8 input and per-group scales instead of FP16 input.
/// The integer multiply-add inner loop is ~2× faster than float dequant.
pub fn encode_affine_matvec_int4xq8(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    q8_data: &MetalBuffer,
    q8_scales: &MetalBuffer,
    weight: &super::weights::AffineQuantizedWeight,
    output: &MetalBuffer,
    n: u32,
    k: u32,
    q8_group_size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(q8_data, 0, 0);
    encoder.set_buffer(q8_scales, 0, 1);
    encoder.set_buffer(&weight.data, 0, 2);
    encoder.set_buffer(&weight.scales, 0, 3);
    encoder.set_buffer(&weight.zeros, 0, 4);
    encoder.set_buffer(output, 0, 5);
    encoder.set_bytes(&n.to_le_bytes(), 6);
    encoder.set_bytes(&k.to_le_bytes(), 7);
    encoder.set_bytes(&weight.group_size.to_le_bytes(), 8);
    encoder.set_bytes(&q8_group_size.to_le_bytes(), 9);
    encoder.dispatch_threadgroups((n as usize, 1, 1), (32, 1, 1));
}
