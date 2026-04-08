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
