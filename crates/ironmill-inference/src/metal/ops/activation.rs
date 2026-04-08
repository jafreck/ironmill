//! Activation pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::DEFAULT_THREADGROUP_WIDTH;

/// Activation pipeline states.
pub struct ActivationPipelines {
    /// SiLU-gated activation kernel.
    pub silu_gate: ComputePipeline,
    /// GELU-gated activation kernel (Gemma 4).
    pub ffn_gelu_gate: ComputePipeline,
    /// PLE GELU-gated activation kernel.
    pub ple_gelu_gate: ComputePipeline,
    /// PLE add-and-scale kernel.
    pub ple_add_scale: ComputePipeline,
    /// Sigmoid gating kernel (Qwen3.5 attn_output_gate).
    pub sigmoid_gate: ComputePipeline,
}

// ── Dispatch helpers ─────────────────────────────────────────────

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
