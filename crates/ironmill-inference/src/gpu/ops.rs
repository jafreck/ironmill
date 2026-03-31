//! Kernel dispatch helpers — compile shaders and dispatch compute operations.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer, MetalDevice};

use super::error::GpuError;

/// All compiled Metal pipeline states for the GPU backend.
pub struct GpuPipelines {
    pub rms_norm: ComputePipeline,
    pub silu_gate: ComputePipeline,
    pub rope: ComputePipeline,
    pub residual_add: ComputePipeline,
    pub embedding_lookup: ComputePipeline,
    pub turboquant_cache_write: ComputePipeline,
    pub turboquant_attention: ComputePipeline,
    pub standard_attention: ComputePipeline,
}

impl GpuPipelines {
    /// Compile all Metal shaders and create pipeline states.
    pub fn compile(device: &MetalDevice) -> Result<Self, GpuError> {
        let norm_src = include_str!("shaders/normalization.metal");
        let act_src = include_str!("shaders/activation.metal");
        let rope_src = include_str!("shaders/rope.metal");
        let elem_src = include_str!("shaders/elementwise.metal");
        let embed_src = include_str!("shaders/embedding.metal");
        let tq_src = include_str!("shaders/turboquant.metal");
        let attn_src = include_str!("shaders/attention.metal");

        let norm_lib = device
            .compile_shader_source(norm_src)
            .map_err(GpuError::Metal)?;
        let act_lib = device
            .compile_shader_source(act_src)
            .map_err(GpuError::Metal)?;
        let rope_lib = device
            .compile_shader_source(rope_src)
            .map_err(GpuError::Metal)?;
        let elem_lib = device
            .compile_shader_source(elem_src)
            .map_err(GpuError::Metal)?;
        let embed_lib = device
            .compile_shader_source(embed_src)
            .map_err(GpuError::Metal)?;
        let tq_lib = device
            .compile_shader_source(tq_src)
            .map_err(GpuError::Metal)?;
        let attn_lib = device
            .compile_shader_source(attn_src)
            .map_err(GpuError::Metal)?;

        Ok(Self {
            rms_norm: device
                .create_compute_pipeline(
                    &norm_lib.get_function("rms_norm").map_err(GpuError::Metal)?,
                )
                .map_err(GpuError::Metal)?,
            silu_gate: device
                .create_compute_pipeline(
                    &act_lib.get_function("silu_gate").map_err(GpuError::Metal)?,
                )
                .map_err(GpuError::Metal)?,
            rope: device
                .create_compute_pipeline(&rope_lib.get_function("rope").map_err(GpuError::Metal)?)
                .map_err(GpuError::Metal)?,
            residual_add: device
                .create_compute_pipeline(
                    &elem_lib
                        .get_function("residual_add")
                        .map_err(GpuError::Metal)?,
                )
                .map_err(GpuError::Metal)?,
            embedding_lookup: device
                .create_compute_pipeline(
                    &embed_lib
                        .get_function("embedding_lookup")
                        .map_err(GpuError::Metal)?,
                )
                .map_err(GpuError::Metal)?,
            turboquant_cache_write: device
                .create_compute_pipeline(
                    &tq_lib
                        .get_function("turboquant_cache_write")
                        .map_err(GpuError::Metal)?,
                )
                .map_err(GpuError::Metal)?,
            turboquant_attention: device
                .create_compute_pipeline(
                    &tq_lib
                        .get_function("turboquant_attention")
                        .map_err(GpuError::Metal)?,
                )
                .map_err(GpuError::Metal)?,
            standard_attention: device
                .create_compute_pipeline(
                    &attn_lib
                        .get_function("standard_attention")
                        .map_err(GpuError::Metal)?,
                )
                .map_err(GpuError::Metal)?,
        })
    }
}

// ── Dispatch helpers ─────────────────────────────────────────────
//
// These encode kernel dispatches into a ComputeEncoder without
// committing. All operations are batched into a single command
// buffer per token.

/// Encode an RMSNorm operation.
#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    input: &MetalBuffer,
    weight: &MetalBuffer,
    output: &MetalBuffer,
    hidden_size: u32,
    token_count: u32,
    eps: f32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(weight, 0, 1);
    encoder.set_buffer(output, 0, 2);
    encoder.set_bytes(&hidden_size.to_le_bytes(), 3);
    encoder.set_bytes(&token_count.to_le_bytes(), 4);
    encoder.set_bytes(&eps.to_le_bytes(), 5);
    encoder.dispatch_threadgroups((token_count as usize, 1, 1), (hidden_size as usize, 1, 1));
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
#[allow(clippy::too_many_arguments)]
pub fn encode_rope(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    qk: &MetalBuffer,
    cos_cache: &MetalBuffer,
    sin_cache: &MetalBuffer,
    num_heads: u32,
    head_dim: u32,
    seq_offset: u32,
    token_count: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(qk, 0, 0);
    encoder.set_buffer(cos_cache, 0, 1);
    encoder.set_buffer(sin_cache, 0, 2);
    encoder.set_bytes(&num_heads.to_le_bytes(), 3);
    encoder.set_bytes(&head_dim.to_le_bytes(), 4);
    encoder.set_bytes(&seq_offset.to_le_bytes(), 5);
    encoder.set_bytes(&token_count.to_le_bytes(), 6);
    let half_dim = head_dim / 2;
    encoder.dispatch_threads(
        (half_dim as usize, num_heads as usize, token_count as usize),
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
#[allow(clippy::too_many_arguments)]
pub fn encode_embedding_lookup(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    token_ids: &MetalBuffer,
    embedding_table: &MetalBuffer,
    output: &MetalBuffer,
    hidden_size: u32,
    token_count: u32,
    vocab_size: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(token_ids, 0, 0);
    encoder.set_buffer(embedding_table, 0, 1);
    encoder.set_buffer(output, 0, 2);
    encoder.set_bytes(&hidden_size.to_le_bytes(), 3);
    encoder.set_bytes(&token_count.to_le_bytes(), 4);
    encoder.set_bytes(&vocab_size.to_le_bytes(), 5);
    let tg_size = 256.min(hidden_size as usize);
    encoder.dispatch_threadgroups(
        (
            (hidden_size as usize).div_ceil(tg_size),
            token_count as usize,
            1,
        ),
        (tg_size, 1, 1),
    );
}

/// Encode TurboQuant cache write.
#[allow(clippy::too_many_arguments)]
pub fn encode_turboquant_cache_write(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    kv_proj: &MetalBuffer,
    rotation_matrix: &MetalBuffer,
    cache: &MetalBuffer,
    num_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    seq_pos: u32,
    inv_scale: f32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(kv_proj, 0, 0);
    encoder.set_buffer(rotation_matrix, 0, 1);
    encoder.set_buffer(cache, 0, 2);
    encoder.set_bytes(&num_kv_heads.to_le_bytes(), 3);
    encoder.set_bytes(&head_dim.to_le_bytes(), 4);
    encoder.set_bytes(&max_seq_len.to_le_bytes(), 5);
    encoder.set_bytes(&seq_pos.to_le_bytes(), 6);
    encoder.set_bytes(&inv_scale.to_le_bytes(), 7);
    encoder.dispatch_threadgroups((num_kv_heads as usize, 1, 1), (head_dim as usize, 1, 1));
}

/// Encode TurboQuant attention.
#[allow(clippy::too_many_arguments)]
pub fn encode_turboquant_attention(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    q: &MetalBuffer,
    k_cache: &MetalBuffer,
    v_cache: &MetalBuffer,
    rotation_matrix: &MetalBuffer,
    output: &MetalBuffer,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    seq_len: u32,
    deq_scale: f32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(q, 0, 0);
    encoder.set_buffer(k_cache, 0, 1);
    encoder.set_buffer(v_cache, 0, 2);
    encoder.set_buffer(rotation_matrix, 0, 3);
    encoder.set_buffer(output, 0, 4);
    encoder.set_bytes(&num_heads.to_le_bytes(), 5);
    encoder.set_bytes(&num_kv_heads.to_le_bytes(), 6);
    encoder.set_bytes(&head_dim.to_le_bytes(), 7);
    encoder.set_bytes(&max_seq_len.to_le_bytes(), 8);
    encoder.set_bytes(&seq_len.to_le_bytes(), 9);
    encoder.set_bytes(&deq_scale.to_le_bytes(), 10);
    encoder.dispatch_threadgroups((num_heads as usize, 1, 1), (head_dim as usize, 1, 1));
}

/// Encode standard FP16 attention.
#[allow(clippy::too_many_arguments)]
pub fn encode_standard_attention(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    q: &MetalBuffer,
    k_cache: &MetalBuffer,
    v_cache: &MetalBuffer,
    output: &MetalBuffer,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    seq_len: u32,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(q, 0, 0);
    encoder.set_buffer(k_cache, 0, 1);
    encoder.set_buffer(v_cache, 0, 2);
    encoder.set_buffer(output, 0, 3);
    encoder.set_bytes(&num_heads.to_le_bytes(), 4);
    encoder.set_bytes(&num_kv_heads.to_le_bytes(), 5);
    encoder.set_bytes(&head_dim.to_le_bytes(), 6);
    encoder.set_bytes(&max_seq_len.to_le_bytes(), 7);
    encoder.set_bytes(&seq_len.to_le_bytes(), 8);
    encoder.dispatch_threadgroups((num_heads as usize, 1, 1), (head_dim as usize, 1, 1));
}
