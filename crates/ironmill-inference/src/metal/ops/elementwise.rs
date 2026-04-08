//! Elementwise pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::DEFAULT_THREADGROUP_WIDTH;

/// Elementwise pipeline states.
pub struct ElementwisePipelines {
    /// Element-wise residual addition kernel.
    pub residual_add: ComputePipeline,
    /// Broadcast bias add: data[i] += bias[i % H]. Used for DAC correction.
    pub bias_add: ComputePipeline,
    /// Buffer copy kernel (element-wise half → half).
    pub copy_buffer: ComputePipeline,
    /// Element-wise buffer scaling kernel.
    pub scale_buffer: ComputePipeline,
    /// Q8 input quantization kernel: FP16 → INT8 with per-group scales.
    pub quantize_input_q8: ComputePipeline,
}

// ── Dispatch helpers ─────────────────────────────────────────────

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
