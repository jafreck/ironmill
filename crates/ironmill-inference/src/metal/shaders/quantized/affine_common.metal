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
constant constexpr uint N_SIMDGROUPS   = 8;
constant constexpr uint THREADS_PER_TG = N_SIMDGROUPS * 32;
constant constexpr uint TM_TILE        = N_SIMDGROUPS * 8;   // 64 for 8 SG, 128 for 16 SG
constant constexpr uint TN_TILE        = 64;
constant constexpr uint TN_STRIDE      = TN_TILE + 1;
constant constexpr uint MATMUL_K_TILE  = 32;
constant constexpr uint K_BLOCKS       = MATMUL_K_TILE / 8;  // 4 MMA ops per K-tile
constant constexpr uint TN_BLOCKS      = TN_TILE / 8;

// ── Blocked-layout constants (must match pack_quantized_blocked) ─
constant constexpr uint BLK_N = 64;
constant constexpr uint BLK_K = 8;

// ── Superblock layout constants ──
// Superblock = [scale:2B][zero:2B][data:group_size/elems_per_byte bytes]
// For INT4 gs=128: sb_bytes = 4 + 64 = 68
// For INT8 gs=128: sb_bytes = 4 + 128 = 132
//
// Layout: W[row * sb_stride + group * sb_bytes + offset]
// where sb_stride = num_groups * sb_bytes
constant constexpr uint SB_HEADER_BYTES = 4;  // 2B scale + 2B zero

// Superblock decode constants
constant constexpr uint SB_ROWS_PER_SG = 4;
constant constexpr uint SB_NUM_SIMDGROUPS = 2;
constant constexpr uint SB_ROWS_PER_TG = SB_NUM_SIMDGROUPS * SB_ROWS_PER_SG;  // 8

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
