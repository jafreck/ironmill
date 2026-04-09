#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── Group-size function constant ──
// Resolved at pipeline creation time via FunctionConstantValues.
// Enables compile-time: / GS → >> log2(GS), % GS → & (GS-1).
constant uint GS [[function_constant(0)]];

// ── Matmul tuning parameters ──
constant constexpr uint N_SIMDGROUPS   = 8;
constant constexpr uint THREADS_PER_TG = N_SIMDGROUPS * 32;
constant constexpr uint TM_TILE        = N_SIMDGROUPS * 8;
constant constexpr uint TN_TILE        = 64;
constant constexpr uint TN_STRIDE      = TN_TILE + 1;
constant constexpr uint MATMUL_K_TILE  = 32;
constant constexpr uint K_BLOCKS       = MATMUL_K_TILE / 8;
constant constexpr uint TN_BLOCKS      = TN_TILE / 8;

// ── Superblock layout constants ──
constant constexpr uint SB_HEADER_BYTES = 4;
constant constexpr uint SB_ROWS_PER_SG = 4;
constant constexpr uint SB_NUM_SIMDGROUPS = 2;
constant constexpr uint SB_ROWS_PER_TG = SB_NUM_SIMDGROUPS * SB_ROWS_PER_SG;

// ── Fused FFN tile constant ──
constant constexpr uint FUSED_FFN_TILE_K = 64;

// ── Blocked-layout constants (still needed for some address math) ──
constant constexpr uint BLK_N = 64;
constant constexpr uint BLK_K = 8;

// ── GDN batched params struct ──
struct GdnBatchedInt4Params {
    uint N0;
    uint N1;
    uint N2;
    uint N3;
    uint K;
    uint has_awq;
};
