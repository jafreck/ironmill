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
    /// Rotary positional embedding kernel.
    pub rope: ComputePipeline,
    /// Element-wise residual addition kernel.
    pub residual_add: ComputePipeline,
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
    pub fused_sdpa: ComputePipeline,
    /// QuIP# matvec kernel.
    pub quip_sharp_matvec: ComputePipeline,
    /// QuIP# matmul kernel.
    pub quip_sharp_matmul: ComputePipeline,
    pub fused_softcap: ComputePipeline,
    pub ple_gelu_gate: ComputePipeline,
    pub ple_add_scale: ComputePipeline,
    pub scale_buffer: ComputePipeline,
    pub copy_buffer: ComputePipeline,
    pub moe_softmax: ComputePipeline,
    pub moe_gelu: ComputePipeline,
    pub moe_mul: ComputePipeline,
    pub moe_weighted_combine: ComputePipeline,
}

impl MetalPipelines {
    /// Load precompiled Metal shader libraries and create pipeline states.
    ///
    /// HEAD_DIM-independent shaders are loaded from precompiled `.metallib`
    /// binaries embedded at build time (~1ms). HEAD_DIM-dependent shaders
    /// (attention, turboquant) use precompiled variants for common head
    /// dimensions (64, 80, 128, 256) and fall back to runtime source
    /// compilation for uncommon values.
    pub fn compile(device: &MetalDevice, head_dim: usize) -> Result<Self, MetalError> {
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

        // ── HEAD_DIM-dependent shaders (precompiled or fallback) ─
        let (attn_lib, tq_lib, sdpa_lib) = Self::load_head_dim_shaders(device, head_dim)?;

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
            fused_sdpa: device
                .create_compute_pipeline(
                    &sdpa_lib
                        .get_function("fused_sdpa")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
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
            copy_buffer: device
                .create_compute_pipeline(
                    &elem_lib
                        .get_function("copy_buffer")
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
        })
    }

    /// Load attention, turboquant, and fused SDPA shaders for a specific HEAD_DIM.
    ///
    /// Uses precompiled `.metallib` for common values (64, 80, 128, 256);
    /// falls back to runtime source compilation for uncommon dimensions.
    fn load_head_dim_shaders(
        device: &MetalDevice,
        head_dim: usize,
    ) -> Result<
        (
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
        }

        let try_precompiled = |attn_data: &[u8],
                               tq_data: &[u8],
                               sdpa_data: &[u8]|
         -> Result<
            (
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
            Ok((attn, tq, sdpa))
        };

        match head_dim {
            64 => try_precompiled(
                precompiled!(attn "64"),
                precompiled!(tq "64"),
                precompiled!(sdpa "64"),
            ),
            80 => try_precompiled(
                precompiled!(attn "80"),
                precompiled!(tq "80"),
                precompiled!(sdpa "80"),
            ),
            128 => try_precompiled(
                precompiled!(attn "128"),
                precompiled!(tq "128"),
                precompiled!(sdpa "128"),
            ),
            256 => try_precompiled(
                precompiled!(attn "256"),
                precompiled!(tq "256"),
                precompiled!(sdpa "256"),
            ),
            _ => {
                // Uncommon head_dim — fall back to runtime source compilation.
                let header = format!(
                    "#define HEAD_DIM {head_dim}\n#define HEAD_DIM_PACKED {}\n",
                    head_dim / 2
                );
                let attn_src_raw = include_str!("shaders/attention.metal");
                let attn_src = format!("{header}{attn_src_raw}");
                let tq_helpers = include_str!("../shaders/turboquant_helpers.metal");
                let tq_src_raw = include_str!("shaders/turboquant.metal");
                let tq_src = format!("{header}{tq_helpers}\n{tq_src_raw}");
                let sdpa_src_raw = include_str!("shaders/fused_sdpa.metal");
                let sdpa_src = format!("{header}{sdpa_src_raw}");

                let attn = device
                    .compile_shader_source(&attn_src)
                    .map_err(MetalError::Metal)?;
                let tq = device
                    .compile_shader_source(&tq_src)
                    .map_err(MetalError::Metal)?;
                let sdpa = device
                    .compile_shader_source(&sdpa_src)
                    .map_err(MetalError::Metal)?;
                Ok((attn, tq, sdpa))
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

/// Encode GELU-gated element-wise multiply with strided input access:
///   `output[i] = gelu(gate[i]) * input_slice[token, layer_offset + elem]`.
///
/// The `input` buffer has row stride `input_stride` elements and each layer's
/// slice starts at column `input_offset`.
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
