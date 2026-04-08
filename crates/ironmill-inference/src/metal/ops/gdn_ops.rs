//! GDN recurrent pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::{DEFAULT_THREADGROUP_WIDTH, METAL_MAX_THREADS_PER_THREADGROUP};

/// GDN recurrent pipeline states.
pub struct GdnPipelines {
    /// GDN conv1d + SiLU kernel.
    pub conv1d_silu: ComputePipeline,
    /// GDN recurrent state update kernel.
    pub recurrent_update: ComputePipeline,
    /// GDN per-head output gate (RMSNorm + silu(z)) kernel.
    pub output_gate: ComputePipeline,
    /// GDN prefill batched conv1d + SiLU kernel (all tokens, one dispatch).
    pub prefill_conv1d_silu: ComputePipeline,
    /// GDN prefill batched recurrent + norm + gate kernel (all tokens, one dispatch).
    pub prefill_recurrent: ComputePipeline,
    /// Fused GDN decode kernel: conv1d+SiLU+recurrent+output_gate in one dispatch.
    pub fused_decode: ComputePipeline,
    /// Batched dense FP16 matvec for 4 GDN projections in one dispatch.
    pub batched_matvec: ComputePipeline,
}

// ── Parameter structs ────────────────────────────────────────────

/// Parameters for [`encode_gdn_recurrent_update`].
pub struct GdnRecurrentUpdateParams<'a> {
    /// Conv1d + SiLU output buffer.
    pub conv_out: &'a MetalBuffer,
    /// A projection buffer.
    pub a_proj: &'a MetalBuffer,
    /// B projection buffer.
    pub b_proj: &'a MetalBuffer,
    /// Log-space A parameter buffer.
    pub a_log: &'a MetalBuffer,
    /// Δt bias buffer.
    pub dt_bias: &'a MetalBuffer,
    /// Recurrent state matrix buffer (read/write).
    pub recurrent_state: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Total key dimension.
    pub key_dim: u32,
    /// Total value dimension.
    pub value_dim: u32,
    /// Number of value heads.
    pub num_v_heads: u32,
    /// Per-head key dimension.
    pub k_head_dim: u32,
    /// Per-head value dimension.
    pub v_head_dim: u32,
    /// Number of key heads.
    pub num_k_heads: u32,
}

/// Parameters for [`encode_gdn_fused_decode`].
pub struct GdnFusedDecodeParams<'a> {
    /// QKV input buffer.
    pub input_qkv: &'a MetalBuffer,
    /// Conv1d weight buffer.
    pub conv_weight: &'a MetalBuffer,
    /// Conv state buffer (read/write).
    pub conv_state: &'a MetalBuffer,
    /// A projection buffer.
    pub a_proj: &'a MetalBuffer,
    /// B projection buffer.
    pub b_proj: &'a MetalBuffer,
    /// Log-space A parameter buffer.
    pub a_log: &'a MetalBuffer,
    /// Δt bias buffer.
    pub dt_bias: &'a MetalBuffer,
    /// Recurrent state matrix buffer (read/write).
    pub recurrent_state: &'a MetalBuffer,
    /// Z (gate) projection buffer.
    pub z_proj: &'a MetalBuffer,
    /// RMSNorm weight buffer.
    pub norm_weight: &'a MetalBuffer,
    /// Output buffer.
    pub output: &'a MetalBuffer,
    /// Scratch buffer for intermediate conv1d output.
    pub conv_out_scratch: &'a MetalBuffer,
    /// Total QKV dimension.
    pub qkv_dim: u32,
    /// Conv1d kernel size.
    pub kernel_size: u32,
    /// Total key dimension.
    pub key_dim: u32,
    /// Total value dimension.
    pub value_dim: u32,
    /// Number of value heads.
    pub num_v_heads: u32,
    /// Per-head key dimension.
    pub k_head_dim: u32,
    /// Per-head value dimension.
    pub v_head_dim: u32,
    /// Number of key heads.
    pub num_k_heads: u32,
    /// RMSNorm epsilon.
    pub eps: f32,
}

/// Parameters for [`encode_gdn_batched_matvec`].
pub struct GdnBatchedMatvecParams<'a> {
    /// Shared input buffer.
    pub input: &'a MetalBuffer,
    /// QKV weight matrix (packed FP16).
    pub w_qkv: &'a MetalBuffer,
    /// Z weight matrix.
    pub w_z: &'a MetalBuffer,
    /// A weight matrix.
    pub w_a: &'a MetalBuffer,
    /// B weight matrix.
    pub w_b: &'a MetalBuffer,
    /// QKV output buffer.
    pub y_qkv: &'a MetalBuffer,
    /// Z output buffer.
    pub y_z: &'a MetalBuffer,
    /// A output buffer.
    pub y_a: &'a MetalBuffer,
    /// B output buffer.
    pub y_b: &'a MetalBuffer,
    /// Input dimension (hidden size).
    pub k: u32,
    /// QKV output dimension.
    pub n_qkv: u32,
    /// Z output dimension.
    pub n_z: u32,
    /// A output dimension.
    pub n_a: u32,
    /// B output dimension.
    pub n_b: u32,
}

/// Parameters for [`encode_gdn_prefill_recurrent`].
pub struct GdnPrefillRecurrentParams<'a> {
    /// All-token conv1d output buffer.
    pub all_conv_out: &'a MetalBuffer,
    /// All-token A projection buffer.
    pub all_a: &'a MetalBuffer,
    /// All-token B projection buffer.
    pub all_b: &'a MetalBuffer,
    /// Log-space A parameter buffer.
    pub a_log: &'a MetalBuffer,
    /// Δt bias buffer.
    pub dt_bias: &'a MetalBuffer,
    /// RMSNorm weight buffer.
    pub norm_weight: &'a MetalBuffer,
    /// All-token Z (gate) projection buffer.
    pub all_z: &'a MetalBuffer,
    /// Recurrent state matrix buffer (read/write).
    pub recurrent_state: &'a MetalBuffer,
    /// All-token output buffer.
    pub all_output: &'a MetalBuffer,
    /// Number of tokens in the prefill batch.
    pub token_count: u32,
    /// Total QKV dimension.
    pub qkv_dim: u32,
    /// Total key dimension.
    pub key_dim: u32,
    /// Total value dimension.
    pub value_dim: u32,
    /// Number of value heads.
    pub num_v_heads: u32,
    /// Per-head key dimension.
    pub k_head_dim: u32,
    /// Per-head value dimension.
    pub v_head_dim: u32,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Number of key heads.
    pub num_k_heads: u32,
}

// ── Dispatch helpers ─────────────────────────────────────────────

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

/// Encode GDN recurrent state update.
///
/// Per-head: compute gates, update state matrix S, compute o = S @ q.
/// One threadgroup per head, v_head_dim threads per threadgroup.
#[allow(dead_code)]
pub(crate) fn encode_gdn_recurrent_update(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &GdnRecurrentUpdateParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.conv_out, 0, 0);
    encoder.set_buffer(params.a_proj, 0, 1);
    encoder.set_buffer(params.b_proj, 0, 2);
    encoder.set_buffer(params.a_log, 0, 3);
    encoder.set_buffer(params.dt_bias, 0, 4);
    encoder.set_buffer(params.recurrent_state, 0, 5);
    encoder.set_buffer(params.output, 0, 6);
    let gpu_params: [u32; 6] = [
        params.key_dim,
        params.value_dim,
        params.num_v_heads,
        params.k_head_dim,
        params.v_head_dim,
        params.num_k_heads,
    ];
    let params_bytes: Vec<u8> = gpu_params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 7);
    let tg_size = (params.v_head_dim as usize).min(METAL_MAX_THREADS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((params.num_v_heads as usize, 1, 1), (tg_size, 1, 1));
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
pub(crate) fn encode_gdn_fused_decode(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &GdnFusedDecodeParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.input_qkv, 0, 0);
    encoder.set_buffer(params.conv_weight, 0, 1);
    encoder.set_buffer(params.conv_state, 0, 2);
    encoder.set_buffer(params.a_proj, 0, 3);
    encoder.set_buffer(params.b_proj, 0, 4);
    encoder.set_buffer(params.a_log, 0, 5);
    encoder.set_buffer(params.dt_bias, 0, 6);
    encoder.set_buffer(params.recurrent_state, 0, 7);
    encoder.set_buffer(params.z_proj, 0, 8);
    encoder.set_buffer(params.norm_weight, 0, 9);
    encoder.set_buffer(params.output, 0, 10);
    encoder.set_buffer(params.conv_out_scratch, 0, 11);
    let gpu_params: [u32; 10] = [
        params.qkv_dim,
        params.kernel_size,
        params.kernel_size - 1, // conv_state_len
        params.key_dim,
        params.value_dim,
        params.num_v_heads,
        params.k_head_dim,
        params.v_head_dim,
        params.num_k_heads,
        params.eps.to_bits(),
    ];
    let params_bytes: Vec<u8> = gpu_params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 12);
    // Thread count: need enough for both conv1d (qkv_dim/num_v_heads channels)
    // and recurrent (v_head_dim rows). Use the larger of the two.
    let channels_per_head = params.qkv_dim.div_ceil(params.num_v_heads) as usize;
    let tg_size = channels_per_head
        .max(params.v_head_dim as usize)
        .min(METAL_MAX_THREADS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((params.num_v_heads as usize, 1, 1), (tg_size, 1, 1));
}

/// Encode batched dense FP16 matvec for 4 GDN projections in a single dispatch.
///
/// Computes y_i = x · W_i^T for i in {QKV, Z, A, B}. All share the same input x.
/// Saves 3 dispatches per GDN layer compared to 4 separate `encode_matvec` calls.
pub(crate) fn encode_gdn_batched_matvec(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &GdnBatchedMatvecParams<'_>,
) {
    const ROWS_PER_TG: u32 = 64;
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.input, 0, 0);
    encoder.set_buffer(params.w_qkv, 0, 1);
    encoder.set_buffer(params.w_z, 0, 2);
    encoder.set_buffer(params.w_a, 0, 3);
    encoder.set_buffer(params.w_b, 0, 4);
    encoder.set_buffer(params.y_qkv, 0, 5);
    encoder.set_buffer(params.y_z, 0, 6);
    encoder.set_buffer(params.y_a, 0, 7);
    encoder.set_buffer(params.y_b, 0, 8);
    let gpu_params: [u32; 6] = [
        params.k,
        params.n_qkv,
        params.n_z,
        params.n_a,
        params.n_b,
        0,
    ];
    let params_bytes: Vec<u8> = gpu_params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 9);
    let tg_count = params.n_qkv.div_ceil(ROWS_PER_TG)
        + params.n_z.div_ceil(ROWS_PER_TG)
        + params.n_a.div_ceil(ROWS_PER_TG)
        + params.n_b.div_ceil(ROWS_PER_TG);
    encoder.dispatch_threadgroups((tg_count as usize, 1, 1), (DEFAULT_THREADGROUP_WIDTH, 1, 1));
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

/// Encode GDN prefill batched recurrent + RMSNorm + silu(z) gating.
///
/// Processes ALL tokens sequentially per head in a single dispatch.
/// One threadgroup per head, v_head_dim threads per threadgroup.
pub(crate) fn encode_gdn_prefill_recurrent(
    encoder: &ComputeEncoder,
    pipeline: &ComputePipeline,
    params: &GdnPrefillRecurrentParams<'_>,
) {
    encoder.set_pipeline(pipeline);
    encoder.set_buffer(params.all_conv_out, 0, 0);
    encoder.set_buffer(params.all_a, 0, 1);
    encoder.set_buffer(params.all_b, 0, 2);
    encoder.set_buffer(params.a_log, 0, 3);
    encoder.set_buffer(params.dt_bias, 0, 4);
    encoder.set_buffer(params.norm_weight, 0, 5);
    encoder.set_buffer(params.all_z, 0, 6);
    encoder.set_buffer(params.recurrent_state, 0, 7);
    encoder.set_buffer(params.all_output, 0, 8);
    let gpu_params: [u32; 9] = [
        params.token_count,
        params.qkv_dim,
        params.key_dim,
        params.value_dim,
        params.num_v_heads,
        params.k_head_dim,
        params.v_head_dim,
        params.eps.to_bits(),
        params.num_k_heads,
    ];
    let params_bytes: Vec<u8> = gpu_params.iter().flat_map(|v| v.to_le_bytes()).collect();
    encoder.set_bytes(&params_bytes, 9);
    let tg_size = (params.v_head_dim as usize).min(METAL_MAX_THREADS_PER_THREADGROUP);
    encoder.dispatch_threadgroups((params.num_v_heads as usize, 1, 1), (tg_size, 1, 1));
}
