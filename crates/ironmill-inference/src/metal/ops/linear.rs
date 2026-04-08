//! Dense FP16 linear pipeline states and dispatch helpers.

use ironmill_metal_sys::{ComputeEncoder, ComputePipeline, MetalBuffer};

use super::{DEFAULT_THREADGROUP_WIDTH, LinearKernelKind, MATVEC_ROWS_PER_THREADGROUP};

/// Dense FP16 linear pipeline states.
pub struct LinearPipelines {
    /// Dense FP16 matvec kernel.
    pub matvec: ComputePipeline,
    /// Dense FP16 matmul kernel.
    pub matmul: ComputePipeline,
}

impl LinearPipelines {
    /// Select the FP16 dense linear pipeline (matvec or matmul) based on phase.
    #[inline]
    pub fn for_kind(&self, kind: LinearKernelKind) -> &ComputePipeline {
        match kind {
            LinearKernelKind::Matvec => &self.matvec,
            LinearKernelKind::Matmul => &self.matmul,
        }
    }
}

// ── Dispatch helpers ─────────────────────────────────────────────

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
