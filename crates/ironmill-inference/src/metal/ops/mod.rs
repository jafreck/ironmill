//! Kernel dispatch helpers — compile shaders and dispatch compute operations.

pub mod activation;
pub mod attention_ops;
pub mod elementwise;
pub mod embedding_ops;
pub mod fused;
pub mod gdn_ops;
pub mod kv;
pub mod library;
pub mod linear;
pub mod moe;
pub mod norm;
pub mod quantized;
pub mod rope;

pub use self::activation::*;
pub use self::attention_ops::*;
pub use self::elementwise::*;
pub use self::embedding_ops::*;
pub use self::fused::*;
pub use self::gdn_ops::*;
pub use self::kv::*;
pub use self::linear::*;
pub use self::moe::*;
pub use self::norm::*;
pub use self::quantized::*;
pub use self::rope::*;

use ironmill_metal_sys::{ComputePipeline, MetalDevice, ShaderLibrary};

use self::library::ShaderLibraries;
use super::error::MetalError;

// Type aliases for shader compilation return types to reduce tuple complexity.
type ElementwiseShaders = (
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
);
type AttentionShaders = (
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    Option<ComputePipeline>,
    ComputePipeline,
    ComputePipeline,
);
type QuantizedMatmulShaders = (
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
);
type AdvancedMatmulShaders = (
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
);
type FusedShaders = (
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
    ComputePipeline,
);

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
    /// Normalization pipelines.
    pub norm: NormPipelines,
    /// Activation pipelines.
    pub activation: ActivationPipelines,
    /// Attention pipelines.
    pub attention: AttentionPipelines,
    /// Dense FP16 linear pipelines.
    pub linear: LinearPipelines,
    /// Affine quantized pipelines.
    pub affine: AffinePipelines,
    /// PolarQuant pipelines.
    pub polarquant: PolarQuantPipelines,
    /// D2Quant pipelines.
    pub d2quant: D2QuantPipelines,
    /// QuIP# pipelines.
    pub quip: QuipPipelines,
    /// Elementwise pipelines.
    pub elementwise: ElementwisePipelines,
    /// Embedding pipelines.
    pub embedding: EmbeddingPipelines,
    /// KV cache pipelines.
    pub kv: KvPipelines,
    /// GDN recurrent pipelines.
    pub gdn: GdnPipelines,
    /// MoE pipelines.
    pub moe: MoePipelines,
    /// Rope pipelines.
    pub rope: RopePipelines,
    /// Fused operation pipelines.
    pub fused: FusedPipelines,
}

/// Create a compute pipeline from a shader library function.
fn make_pipeline(
    device: &MetalDevice,
    lib: &ShaderLibrary,
    name: &str,
) -> Result<ComputePipeline, MetalError> {
    device
        .create_compute_pipeline(&lib.get_function(name).map_err(MetalError::Metal)?)
        .map_err(MetalError::Metal)
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
        let libs = ShaderLibraries::load(device, head_dim, rotary_dim)?;

        let (rms_norm, silu_gate, ffn_gelu_gate, rope) =
            Self::compile_normalization_shaders(device, &libs)?;
        let (
            residual_add,
            bias_add,
            copy_buffer,
            sigmoid_gate,
            embedding_lookup,
            scale_buffer,
            quantize_input_q8,
            affine_embedding_lookup_int4,
        ) = Self::compile_elementwise_shaders(device, &libs)?;
        let (
            turboquant_cache_write,
            turboquant_attention,
            turboquant_outlier_cache_write,
            turboquant_outlier_attention,
        ) = Self::compile_turboquant_shaders(device, &libs)?;
        let (
            standard_attention,
            prefill_attention,
            prefill_attention_fa2,
            prefill_attention_v2,
            fused_qk_norm_rope,
            fused_sdpa,
            fused_sdpa_split,
            fused_sdpa_reduce,
        ) = Self::compile_attention_shaders(device, &libs)?;
        let (
            polarquant_matvec_int4,
            polarquant_matmul_int4,
            polarquant_matvec_int8,
            polarquant_matmul_int8,
            affine_matvec_int4,
            affine_matmul_int4,
            affine_matvec_int8,
            affine_matmul_int8,
        ) = Self::compile_quantized_matmul_shaders(device, &libs)?;
        let (kv_scatter, matvec, matmul, quip_sharp_matvec, quip_sharp_matmul) =
            Self::compile_dense_matmul_shaders(device, &libs)?;
        let (
            int4_dequantize,
            batched_affine_matvec_int4,
            gdn_batched_affine_matvec_int4,
            gdn_batched_matvec,
            affine_matvec_int4_amx,
            affine_matvec_int8_amx,
            d2quant_matvec_3bit_amx,
            fused_ffn_gate_up_act_int4,
            affine_matvec_int4xq8,
            affine_matvec_int4_2row,
            affine_matvec_int4_coalesced,
            sb_matvec_int4,
            sb_matmul_int4,
            sb_fused_ffn_gate_up_act_int4,
            sb_batched_matvec_int4,
            sb_gdn_batched_matvec_int4,
            sb_matvec_int4xq8,
        ) = Self::compile_advanced_matmul_shaders(device, &libs)?;
        let (
            fused_residual_rms_norm,
            fused_embedding_norm,
            fused_softcap,
            ple_gelu_gate,
            ple_add_scale,
            fused_residual_norm_matvec,
            fused_residual_norm_affine_matvec_int4,
            sb_fused_residual_norm_affine_matvec_int4,
        ) = Self::compile_fused_shaders(device, &libs)?;
        let (moe_softmax, moe_gelu, moe_mul, moe_weighted_combine) =
            Self::compile_moe_shaders(device)?;
        let (d2quant_matvec_3bit, d2quant_matmul_3bit, d2quant_embedding_lookup_3bit) =
            Self::compile_d2quant_shaders(device, &libs)?;
        let (
            gdn_conv1d_silu,
            gdn_recurrent_update,
            gdn_output_gate,
            gdn_prefill_conv1d_silu,
            gdn_prefill_recurrent,
            gdn_fused_decode,
        ) = Self::compile_gdn_shaders(device, &libs)?;

        Ok(Self {
            norm: NormPipelines {
                rms_norm,
                fused_residual_rms_norm,
                fused_embedding_norm,
                fused_qk_norm_rope,
                fused_softcap,
            },
            activation: ActivationPipelines {
                silu_gate,
                ffn_gelu_gate,
                ple_gelu_gate,
                ple_add_scale,
                sigmoid_gate,
            },
            attention: AttentionPipelines {
                standard_attention,
                prefill_attention,
                prefill_attention_fa2,
                prefill_attention_v2,
                fused_sdpa,
                fused_sdpa_split,
                fused_sdpa_reduce,
                turboquant_cache_write,
                turboquant_attention,
                turboquant_outlier_cache_write,
                turboquant_outlier_attention,
            },
            linear: LinearPipelines { matvec, matmul },
            affine: AffinePipelines {
                matvec_int4: affine_matvec_int4,
                matmul_int4: affine_matmul_int4,
                matvec_int8: affine_matvec_int8,
                matmul_int8: affine_matmul_int8,
                matvec_int4_amx: affine_matvec_int4_amx,
                matvec_int8_amx: affine_matvec_int8_amx,
                batched_matvec_int4: batched_affine_matvec_int4,
                gdn_batched_matvec_int4: gdn_batched_affine_matvec_int4,
                fused_ffn_gate_up_act_int4,
                matvec_int4xq8: affine_matvec_int4xq8,
                matvec_int4_2row: affine_matvec_int4_2row,
                matvec_int4_coalesced: affine_matvec_int4_coalesced,
                embedding_lookup_int4: affine_embedding_lookup_int4,
                sb_matvec_int4,
                sb_matmul_int4,
                sb_fused_ffn_gate_up_act_int4,
                sb_batched_matvec_int4,
                sb_gdn_batched_matvec_int4,
                sb_matvec_int4xq8,
            },
            polarquant: PolarQuantPipelines {
                matvec_int4: polarquant_matvec_int4,
                matmul_int4: polarquant_matmul_int4,
                matvec_int8: polarquant_matvec_int8,
                matmul_int8: polarquant_matmul_int8,
            },
            d2quant: D2QuantPipelines {
                matvec_3bit: d2quant_matvec_3bit,
                matmul_3bit: d2quant_matmul_3bit,
                embedding_lookup_3bit: d2quant_embedding_lookup_3bit,
                matvec_3bit_amx: d2quant_matvec_3bit_amx,
            },
            quip: QuipPipelines {
                matvec: quip_sharp_matvec,
                matmul: quip_sharp_matmul,
            },
            elementwise: ElementwisePipelines {
                residual_add,
                bias_add,
                copy_buffer,
                scale_buffer,
                quantize_input_q8,
            },
            embedding: EmbeddingPipelines { embedding_lookup },
            kv: KvPipelines { kv_scatter },
            gdn: GdnPipelines {
                conv1d_silu: gdn_conv1d_silu,
                recurrent_update: gdn_recurrent_update,
                output_gate: gdn_output_gate,
                prefill_conv1d_silu: gdn_prefill_conv1d_silu,
                prefill_recurrent: gdn_prefill_recurrent,
                fused_decode: gdn_fused_decode,
                batched_matvec: gdn_batched_matvec,
            },
            moe: MoePipelines {
                softmax: moe_softmax,
                gelu: moe_gelu,
                mul: moe_mul,
                weighted_combine: moe_weighted_combine,
            },
            rope: RopePipelines { rope },
            fused: FusedPipelines {
                residual_norm_matvec: fused_residual_norm_matvec,
                residual_norm_affine_matvec_int4: fused_residual_norm_affine_matvec_int4,
                int4_dequantize,
                sb_residual_norm_affine_matvec_int4: sb_fused_residual_norm_affine_matvec_int4,
            },
        })
    }

    /// Compile normalization and activation pipelines.
    fn compile_normalization_shaders(
        device: &MetalDevice,
        libs: &ShaderLibraries,
    ) -> Result<
        (
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
        ),
        MetalError,
    > {
        Ok((
            make_pipeline(device, &libs.norm, "rms_norm")?,
            make_pipeline(device, &libs.elem, "silu_gate")?,
            make_pipeline(device, &libs.elem, "gelu_gate")?,
            make_pipeline(device, &libs.elem, "rope")?,
        ))
    }

    /// Compile element-wise, embedding, and utility pipelines.
    fn compile_elementwise_shaders(
        device: &MetalDevice,
        libs: &ShaderLibraries,
    ) -> Result<ElementwiseShaders, MetalError> {
        Ok((
            make_pipeline(device, &libs.elem, "residual_add")?,
            make_pipeline(device, &libs.elem, "bias_add")?,
            make_pipeline(device, &libs.elem, "copy_buffer")?,
            make_pipeline(device, &libs.elem, "sigmoid_gate_inplace")?,
            make_pipeline(device, &libs.elem, "embedding_lookup")?,
            make_pipeline(device, &libs.elem, "scale_buffer")?,
            make_pipeline(device, &libs.quantized, "quantize_input_q8")?,
            make_pipeline(device, &libs.elem, "affine_embedding_lookup_int4")?,
        ))
    }

    /// Compile TurboQuant cache-write and attention pipelines.
    fn compile_turboquant_shaders(
        device: &MetalDevice,
        libs: &ShaderLibraries,
    ) -> Result<
        (
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
        ),
        MetalError,
    > {
        Ok((
            make_pipeline(device, &libs.tq, "turboquant_cache_write")?,
            make_pipeline(device, &libs.tq, "turboquant_attention")?,
            make_pipeline(device, &libs.tq, "turboquant_outlier_cache_write")?,
            make_pipeline(device, &libs.tq, "turboquant_outlier_attention")?,
        ))
    }

    /// Compile standard/prefill attention and fused SDPA pipelines.
    fn compile_attention_shaders(
        device: &MetalDevice,
        libs: &ShaderLibraries,
    ) -> Result<AttentionShaders, MetalError> {
        let fused_sdpa = libs
            .sdpa
            .get_function("fused_sdpa")
            .ok()
            .and_then(|f| device.create_compute_pipeline(&f).ok());
        Ok((
            make_pipeline(device, &libs.attn, "standard_attention")?,
            make_pipeline(device, &libs.attn, "prefill_attention")?,
            make_pipeline(device, &libs.attn, "prefill_attention_fa2")?,
            make_pipeline(device, &libs.attn, "prefill_attention_v2")?,
            make_pipeline(device, &libs.attn, "fused_qk_norm_rope")?,
            fused_sdpa,
            make_pipeline(device, &libs.fd, "fused_sdpa_split")?,
            make_pipeline(device, &libs.fd, "fused_sdpa_reduce")?,
        ))
    }

    /// Compile PolarQuant and Affine quantized matmul pipelines.
    fn compile_quantized_matmul_shaders(
        device: &MetalDevice,
        libs: &ShaderLibraries,
    ) -> Result<QuantizedMatmulShaders, MetalError> {
        Ok((
            make_pipeline(device, &libs.quantized, "polarquant_matvec_int4")?,
            make_pipeline(device, &libs.quantized, "polarquant_matmul_int4")?,
            make_pipeline(device, &libs.quantized, "polarquant_matvec_int8")?,
            make_pipeline(device, &libs.quantized, "polarquant_matmul_int8")?,
            make_pipeline(device, &libs.affine_mm, "affine_matvec_int4")?,
            make_pipeline(device, &libs.affine_mm, "affine_matmul_int4")?,
            make_pipeline(device, &libs.affine_mm, "affine_matvec_int8")?,
            make_pipeline(device, &libs.affine_mm, "affine_matmul_int8")?,
        ))
    }

    /// Compile dense FP16 matmul, KV scatter, and QuIP# pipelines.
    fn compile_dense_matmul_shaders(
        device: &MetalDevice,
        libs: &ShaderLibraries,
    ) -> Result<
        (
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
        ),
        MetalError,
    > {
        Ok((
            make_pipeline(device, &libs.kv_scatter, "kv_scatter")?,
            make_pipeline(device, &libs.matvec, "matvec")?,
            make_pipeline(device, &libs.matvec, "matmul")?,
            make_pipeline(device, &libs.quantized, "quip_sharp_matvec")?,
            make_pipeline(device, &libs.quantized, "quip_sharp_matmul")?,
        ))
    }

    /// Compile advanced matmul variants: batched, AMX, fused, and superblock.
    fn compile_advanced_matmul_shaders(
        device: &MetalDevice,
        libs: &ShaderLibraries,
    ) -> Result<AdvancedMatmulShaders, MetalError> {
        Ok((
            make_pipeline(device, &libs.quantized, "int4_dequantize")?,
            make_pipeline(device, &libs.affine_mm, "batched_affine_matvec_int4")?,
            make_pipeline(device, &libs.affine_mm, "gdn_batched_affine_matvec_int4")?,
            make_pipeline(device, &libs.matvec, "gdn_batched_matvec")?,
            make_pipeline(device, &libs.affine_mm, "affine_matvec_int4_amx")?,
            make_pipeline(device, &libs.affine_mm, "affine_matvec_int8_amx")?,
            make_pipeline(device, &libs.quantized, "d2quant_matvec_3bit_amx")?,
            make_pipeline(device, &libs.affine_mm, "fused_ffn_gate_up_act_int4")?,
            make_pipeline(device, &libs.affine_mm, "affine_matvec_int4xq8")?,
            make_pipeline(device, &libs.affine_mm, "affine_matvec_int4_2row")?,
            make_pipeline(device, &libs.affine_mm, "affine_matvec_int4_coalesced")?,
            // Superblock pipelines
            make_pipeline(device, &libs.affine_mm, "superblock_matvec_int4")?,
            make_pipeline(device, &libs.affine_mm, "superblock_matmul_int4")?,
            make_pipeline(
                device,
                &libs.affine_mm,
                "superblock_fused_ffn_gate_up_act_int4",
            )?,
            make_pipeline(
                device,
                &libs.affine_mm,
                "superblock_batched_affine_matvec_int4",
            )?,
            make_pipeline(
                device,
                &libs.affine_mm,
                "superblock_gdn_batched_affine_matvec_int4",
            )?,
            make_pipeline(device, &libs.affine_mm, "superblock_affine_matvec_int4xq8")?,
        ))
    }

    /// Compile fused residual-norm, embedding-norm, softcap, and PLE pipelines.
    fn compile_fused_shaders(
        device: &MetalDevice,
        libs: &ShaderLibraries,
    ) -> Result<FusedShaders, MetalError> {
        Ok((
            make_pipeline(device, &libs.norm, "fused_residual_rms_norm")?,
            make_pipeline(device, &libs.norm, "fused_embedding_norm")?,
            make_pipeline(device, &libs.norm, "fused_softcap")?,
            make_pipeline(device, &libs.ple, "ple_gelu_gate")?,
            make_pipeline(device, &libs.ple, "add_scale")?,
            make_pipeline(device, &libs.norm, "fused_residual_norm_matvec")?,
            make_pipeline(device, &libs.norm, "fused_residual_norm_affine_matvec_int4")?,
            make_pipeline(
                device,
                &libs.norm,
                "superblock_fused_residual_norm_affine_matvec_int4",
            )?,
        ))
    }

    /// Compile Mixture-of-Experts pipelines (loads dedicated metallib).
    fn compile_moe_shaders(
        device: &MetalDevice,
    ) -> Result<
        (
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
        ),
        MetalError,
    > {
        let moe_lib = device
            .load_library_from_data(include_bytes!(concat!(env!("OUT_DIR"), "/moe.metallib")))
            .map_err(MetalError::Metal)?;
        Ok((
            make_pipeline(device, &moe_lib, "moe_softmax")?,
            make_pipeline(device, &moe_lib, "moe_gelu")?,
            make_pipeline(device, &moe_lib, "moe_mul")?,
            make_pipeline(device, &moe_lib, "moe_weighted_combine")?,
        ))
    }

    /// Compile D2Quant 3-bit matmul and embedding pipelines.
    fn compile_d2quant_shaders(
        device: &MetalDevice,
        libs: &ShaderLibraries,
    ) -> Result<(ComputePipeline, ComputePipeline, ComputePipeline), MetalError> {
        Ok((
            make_pipeline(device, &libs.quantized, "d2quant_matvec_3bit")?,
            make_pipeline(device, &libs.quantized, "d2quant_matmul_3bit")?,
            make_pipeline(device, &libs.quantized, "d2quant_embedding_lookup_3bit")?,
        ))
    }

    /// Compile GDN recurrent pipelines.
    fn compile_gdn_shaders(
        device: &MetalDevice,
        libs: &ShaderLibraries,
    ) -> Result<
        (
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
            ComputePipeline,
        ),
        MetalError,
    > {
        Ok((
            make_pipeline(device, &libs.gdn, "gdn_conv1d_silu")?,
            make_pipeline(device, &libs.gdn, "gdn_recurrent_update")?,
            make_pipeline(device, &libs.gdn, "gdn_output_gate")?,
            make_pipeline(device, &libs.gdn, "gdn_prefill_conv1d_silu")?,
            make_pipeline(device, &libs.gdn, "gdn_prefill_recurrent")?,
            make_pipeline(device, &libs.gdn, "gdn_fused_decode")?,
        ))
    }

    // ── Phase-aware pipeline selection (delegating to sub-structs) ──

    /// Select the FP16 dense linear pipeline (matvec or matmul) based on phase.
    #[inline]
    pub fn dense_linear_pipeline(&self, kind: LinearKernelKind) -> &ComputePipeline {
        self.linear.for_kind(kind)
    }

    /// Select the PolarQuant pipeline for a given bit-width and phase.
    #[inline]
    pub fn polarquant_pipeline(
        &self,
        n_bits: u32,
        kind: LinearKernelKind,
    ) -> Option<&ComputePipeline> {
        self.polarquant.for_bits_and_kind(n_bits, kind)
    }

    /// Select the affine-quantized pipeline for a given bit-width and phase.
    #[inline]
    pub fn affine_pipeline(
        &self,
        bit_width: u32,
        kind: LinearKernelKind,
    ) -> Option<&ComputePipeline> {
        self.affine.for_bits_and_kind(bit_width, kind)
    }

    /// Select the D2Quant dual-scale pipeline for a given bit-width and phase.
    #[inline]
    pub fn d2quant_pipeline(
        &self,
        bit_width: u32,
        kind: LinearKernelKind,
    ) -> Option<&ComputePipeline> {
        self.d2quant.for_bits_and_kind(bit_width, kind)
    }

    /// Select the QuIP# pipeline for the given phase.
    #[inline]
    pub fn quip_sharp_pipeline(&self, kind: LinearKernelKind) -> &ComputePipeline {
        self.quip.for_kind(kind)
    }
}
