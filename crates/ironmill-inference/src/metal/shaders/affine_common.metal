#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// Affine Quantized Matmul: Fused dequant + matmul for INT4/INT8
//
// Weights stay packed in GPU memory; affine dequantization happens inline
// during the dot-product:  w = (quantized - zero) * scale
//
// AWQ compensation (s^{-1}) is now fused into the preceding LayerNorm
// When AWQ scales are present (has_awq=1), the dequantized weight is
// divided by the per-column AWQ scale to compensate for activation-aware
// weight scaling:  w = (quantized - zero) * scale / awq_scale[col]
//
// Two paths per bit-width:
//   - matvec (M=1): one threadgroup per output row, SIMD reduction
//   - matmul (M>1): tiled GEMM with threadgroup-shared tiles
// ============================================================================

// ── Matmul tuning parameters ──
// N_SIMDGROUPS controls the threadgroup size: N_SIMDGROUPS * 32 threads.
// Increasing to 16 (512 threads) doubles output rows per threadgroup but
// increases register pressure and threadgroup memory, potentially reducing
// occupancy (fewer concurrent threadgroups per GPU core).
//
// Trade-off:
//   8 simdgroups (256 threads): lower register pressure, higher occupancy
//  16 simdgroups (512 threads): larger tiles, better arithmetic intensity
//
// Profile on target hardware to determine the optimum.
#include "common/matmul_tile_constants.h"
constant constexpr uint MATMUL_K_TILE  = 32;
constant constexpr uint K_BLOCKS       = MATMUL_K_TILE / 8;  // 4 MMA ops per K-tile

// ── Blocked-layout constants (must match pack_quantized_blocked) ─
constant constexpr uint BLK_N = 64;
constant constexpr uint BLK_K = 8;

// ── AMX decode constants (shared by all AMX matvec kernels) ──
constant constexpr uint AMX_ROWS_PER_TG = 64;
constant constexpr uint AMX_ROWS_PER_SG = 8;
constant constexpr uint AMX_SIMDGROUPS  = 8;
constant constexpr uint AMX_TILE_K      = 128;
constant constexpr uint AMX_TG_SIZE     = 256;

// ── Fused FFN tile constant ──
constant constexpr uint FUSED_FFN_TILE_K = 64;

// ── GDN batched params struct ──
struct GdnBatchedInt4Params {
    uint N0;           // qkv output dim
    uint N1;           // z output dim
    uint N2;           // a output dim
    uint N3;           // b output dim
    uint K;            // input dim (shared)
    uint group_size;
    uint has_awq;
};
