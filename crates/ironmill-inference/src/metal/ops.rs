//! Kernel dispatch helpers — compile shaders and dispatch compute operations.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer, MetalDevice};

use super::error::MetalError;

/// All compiled Metal pipeline states for the Metal backend.
pub struct MetalPipelines {
    pub rms_norm: ComputePipeline,
    pub silu_gate: ComputePipeline,
    pub rope: ComputePipeline,
    pub residual_add: ComputePipeline,
    pub embedding_lookup: ComputePipeline,
    pub turboquant_cache_write: ComputePipeline,
    pub turboquant_attention: ComputePipeline,
    pub turboquant_outlier_cache_write: ComputePipeline,
    pub turboquant_outlier_attention: ComputePipeline,
    pub standard_attention: ComputePipeline,
    pub polarquant_matvec_int4: ComputePipeline,
    pub polarquant_matmul_int4: ComputePipeline,
    pub polarquant_matvec_int8: ComputePipeline,
    pub polarquant_matmul_int8: ComputePipeline,
    pub kv_scatter: ComputePipeline,
    pub matvec: ComputePipeline,
    pub fused_residual_rms_norm: ComputePipeline,
}

impl MetalPipelines {
    /// Compile all Metal shaders and create pipeline states.
    ///
    /// `head_dim` is injected into TurboQuant and attention shaders via
    /// `#define HEAD_DIM` so shared memory is sized exactly.
    pub fn compile(device: &MetalDevice, head_dim: usize) -> Result<Self, MetalError> {
        let head_dim_header = format!(
            "#define HEAD_DIM {head_dim}\n#define HEAD_DIM_PACKED {}\n",
            head_dim / 2
        );

        let norm_src = include_str!("shaders/normalization.metal");
        let act_src = include_str!("shaders/activation.metal");
        let rope_src = include_str!("shaders/rope.metal");
        let elem_src = include_str!("shaders/elementwise.metal");
        let embed_src = include_str!("shaders/embedding.metal");
        let tq_src_raw = include_str!("shaders/turboquant.metal");
        let tq_src = format!("{head_dim_header}{tq_src_raw}");
        let attn_src_raw = include_str!("shaders/attention.metal");
        let attn_src = format!("{head_dim_header}{attn_src_raw}");
        let qmm_src = include_str!("shaders/quantized_matmul.metal");
        let kv_scatter_src = include_str!("shaders/kv_scatter.metal");
        let matvec_src = include_str!("shaders/matvec.metal");
        let fused_rn_src = include_str!("shaders/fused_residual_norm.metal");

        let norm_lib = device
            .compile_shader_source(norm_src)
            .map_err(MetalError::Metal)?;
        let act_lib = device
            .compile_shader_source(act_src)
            .map_err(MetalError::Metal)?;
        let rope_lib = device
            .compile_shader_source(rope_src)
            .map_err(MetalError::Metal)?;
        let elem_lib = device
            .compile_shader_source(elem_src)
            .map_err(MetalError::Metal)?;
        let embed_lib = device
            .compile_shader_source(embed_src)
            .map_err(MetalError::Metal)?;
        let tq_lib = device
            .compile_shader_source(&tq_src)
            .map_err(MetalError::Metal)?;
        let attn_lib = device
            .compile_shader_source(&attn_src)
            .map_err(MetalError::Metal)?;
        let qmm_lib = device
            .compile_shader_source(qmm_src)
            .map_err(MetalError::Metal)?;
        let kv_scatter_lib = device
            .compile_shader_source(kv_scatter_src)
            .map_err(MetalError::Metal)?;
        let matvec_lib = device
            .compile_shader_source(matvec_src)
            .map_err(MetalError::Metal)?;
        let fused_rn_lib = device
            .compile_shader_source(fused_rn_src)
            .map_err(MetalError::Metal)?;

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
            fused_residual_rms_norm: device
                .create_compute_pipeline(
                    &fused_rn_lib
                        .get_function("fused_residual_rms_norm")
                        .map_err(MetalError::Metal)?,
                )
                .map_err(MetalError::Metal)?,
        })
    }
}

// ── Parameter structs ────────────────────────────────────────────

/// Parameters for [`encode_rms_norm`].
pub struct RmsNormParams<'a> {
    pub input: &'a MetalBuffer,
    pub weight: &'a MetalBuffer,
    pub output: &'a MetalBuffer,
    pub hidden_size: u32,
    pub token_count: u32,
    pub eps: f32,
}

/// Parameters for [`encode_rope`].
pub struct RopeParams<'a> {
    pub qk: &'a MetalBuffer,
    pub cos_cache: &'a MetalBuffer,
    pub sin_cache: &'a MetalBuffer,
    pub num_heads: u32,
    pub head_dim: u32,
    pub seq_offset: u32,
    pub token_count: u32,
}

/// Parameters for [`encode_embedding_lookup`].
pub struct EmbeddingLookupParams<'a> {
    pub token_ids: &'a MetalBuffer,
    pub embedding_table: &'a MetalBuffer,
    pub output: &'a MetalBuffer,
    pub hidden_size: u32,
    pub token_count: u32,
    pub vocab_size: u32,
}

/// Parameters for [`encode_turboquant_cache_write`].
pub struct TurboquantCacheWriteParams<'a> {
    pub kv_proj: &'a MetalBuffer,
    pub rotation_signs: &'a MetalBuffer,
    pub cache: &'a MetalBuffer,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub seq_pos: u32,
    pub inv_scale: f32,
    pub n_bits: u32,
    pub scale_buf: &'a MetalBuffer,
    pub codebook: &'a MetalBuffer,
    pub boundaries: &'a MetalBuffer,
    pub n_levels: u32,
}

/// Parameters for [`encode_turboquant_attention`].
pub struct TurboquantAttentionParams<'a> {
    pub q: &'a MetalBuffer,
    pub k_cache: &'a MetalBuffer,
    pub v_cache: &'a MetalBuffer,
    pub rotation_signs: &'a MetalBuffer,
    pub output: &'a MetalBuffer,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub seq_len: u32,
    pub deq_scale: f32,
    pub n_bits: u32,
    pub k_scale_buf: &'a MetalBuffer,
    pub v_scale_buf: &'a MetalBuffer,
    pub codebook: &'a MetalBuffer,
}

/// Parameters for [`encode_standard_attention`].
pub struct StandardAttentionParams<'a> {
    pub q: &'a MetalBuffer,
    pub k_cache: &'a MetalBuffer,
    pub v_cache: &'a MetalBuffer,
    pub output: &'a MetalBuffer,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub seq_len: u32,
}

/// Parameters for [`encode_kv_scatter`].
pub struct KvScatterParams<'a> {
    pub proj: &'a MetalBuffer,
    pub cache: &'a MetalBuffer,
    pub seq_pos: u32,
    pub token_count: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
}

/// Parameters for [`encode_fused_residual_rms_norm`].
pub struct FusedResidualRmsNormParams<'a> {
    pub a: &'a MetalBuffer,
    pub b: &'a MetalBuffer,
    pub weight: &'a MetalBuffer,
    pub normed_output: &'a MetalBuffer,
    pub residual_output: &'a MetalBuffer,
    pub eps: f32,
    pub hidden_size: u32,
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
    let tg_size = 1024.min(params.hidden_size as usize);
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
    let tg_size = 256.min(threads);
    let tg_count = threads.div_ceil(tg_size);
    encoder.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
}

/// Encode RoPE application.
pub fn encode_rope(encoder: &ComputeEncoder, pipeline: &ComputePipeline, params: &RopeParams<'_>) {
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
        (half_dim.min(256) as usize, 1, 1),
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
    let tg_size = 256.min(threads);
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
    let tg_size = 256.min(params.hidden_size as usize);
    encoder.dispatch_threadgroups(
        (
            (params.hidden_size as usize).div_ceil(tg_size),
            params.token_count as usize,
            1,
        ),
        (tg_size, 1, 1),
    );
}

/// Encode TurboQuant cache write.
pub fn encode_turboquant_cache_write(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &TurboquantCacheWriteParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.kv_proj, 0, 0);
    encoder.set_buffer(params.rotation_signs, 0, 1);
    encoder.set_buffer(params.cache, 0, 2);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 3);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 4);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 5);
    encoder.set_bytes(&params.seq_pos.to_le_bytes(), 6);
    encoder.set_bytes(&params.inv_scale.to_le_bytes(), 7);
    encoder.set_bytes(&params.n_bits.to_le_bytes(), 8);
    encoder.set_buffer(params.scale_buf, 0, 9);
    encoder.set_buffer(params.codebook, 0, 10);
    encoder.set_buffer(params.boundaries, 0, 11);
    encoder.set_bytes(&params.n_levels.to_le_bytes(), 12);
    encoder.dispatch_threadgroups(
        (params.num_kv_heads as usize, 1, 1),
        (params.head_dim as usize, 1, 1),
    );
}

/// Encode TurboQuant attention.
pub fn encode_turboquant_attention(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &TurboquantAttentionParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.q, 0, 0);
    encoder.set_buffer(params.k_cache, 0, 1);
    encoder.set_buffer(params.v_cache, 0, 2);
    encoder.set_buffer(params.rotation_signs, 0, 3);
    encoder.set_buffer(params.output, 0, 4);
    encoder.set_bytes(&params.num_heads.to_le_bytes(), 5);
    encoder.set_bytes(&params.num_kv_heads.to_le_bytes(), 6);
    encoder.set_bytes(&params.head_dim.to_le_bytes(), 7);
    encoder.set_bytes(&params.max_seq_len.to_le_bytes(), 8);
    encoder.set_bytes(&params.seq_len.to_le_bytes(), 9);
    encoder.set_bytes(&params.deq_scale.to_le_bytes(), 10);
    encoder.set_bytes(&params.n_bits.to_le_bytes(), 11);
    encoder.set_buffer(params.k_scale_buf, 0, 12);
    encoder.set_buffer(params.v_scale_buf, 0, 13);
    encoder.set_buffer(params.codebook, 0, 14);
    // HEAD_DIM is now exact — clamp to Metal's 1024-thread limit only.
    encoder.dispatch_threadgroups(
        (params.num_heads as usize, 1, 1),
        ((params.head_dim as usize).min(1024), 1, 1),
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
    // HEAD_DIM is now exact — clamp to Metal's 1024-thread limit only.
    encoder.dispatch_threadgroups(
        (params.num_heads as usize, 1, 1),
        ((params.head_dim as usize).min(1024), 1, 1),
    );
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
    let tg_x = (params.head_dim as usize).min(256);
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
/// Dispatch: one threadgroup per 64 output rows, 256 threads (8 simdgroups).
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
    let rows_per_tg: usize = 64;
    let tg_count = (n as usize + rows_per_tg - 1) / rows_per_tg;
    encoder.dispatch_threadgroups((tg_count, 1, 1), (256, 1, 1));
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
    let tg_size = 1024.min(params.hidden_size as usize);
    encoder.dispatch_threadgroups((params.token_count as usize, 1, 1), (tg_size, 1, 1));
}
