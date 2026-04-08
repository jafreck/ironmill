//! Mixture-of-Experts pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::DEFAULT_THREADGROUP_WIDTH;

/// MoE pipeline states.
pub struct MoePipelines {
    /// MoE router softmax kernel.
    pub softmax: ComputePipeline,
    /// MoE expert GELU activation kernel.
    pub gelu: ComputePipeline,
    /// MoE element-wise multiply kernel.
    pub mul: ComputePipeline,
    /// MoE weighted expert combination kernel.
    pub weighted_combine: ComputePipeline,
}

// ── Dispatch helpers ─────────────────────────────────────────────

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
