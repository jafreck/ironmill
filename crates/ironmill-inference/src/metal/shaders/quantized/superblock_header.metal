// Superblock kernel header — prepended by build.rs with #define GS <value>
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// GS is defined by build.rs: #define GS 128 (or 32, 64, 256)
#ifndef GS
#error "GS must be defined by build.rs"
#endif

// ── Matmul tuning parameters ──
constant constexpr uint N_SIMDGROUPS   = 8;
constant constexpr uint THREADS_PER_TG = N_SIMDGROUPS * 32;
constant constexpr uint TM_TILE        = N_SIMDGROUPS * 8;
constant constexpr uint TN_TILE        = 64;
constant constexpr uint TN_STRIDE      = TN_TILE + 1;
constant constexpr uint MATMUL_K_TILE  = 32;
constant constexpr uint K_BLOCKS       = MATMUL_K_TILE / 8;
constant constexpr uint TN_BLOCKS      = TN_TILE / 8;

// ── Superblock layout constants (compile-time) ──
// Separate-array layout: data is contiguous (no inline headers).
// Scales and zeros are passed as separate buffer arguments.
constant constexpr uint SB_HEADER_BYTES = 0;
constant constexpr uint SB_BYTES_INT4 = GS / 2;
constant constexpr uint SB_BYTES_INT8 = GS;
constant constexpr uint SB_ROWS_PER_SG = 4;
constant constexpr uint SB_NUM_SIMDGROUPS = 2;
constant constexpr uint SB_ROWS_PER_TG = SB_NUM_SIMDGROUPS * SB_ROWS_PER_SG;

// ── Other shared constants ──
constant constexpr uint BLK_N = 64;
constant constexpr uint BLK_K = 8;
constant constexpr uint FUSED_FFN_TILE_K = 64;

// ── GDN batched params struct (no group_size — uses GS) ──
struct GdnBatchedInt4Params {
    uint N0;
    uint N1;
    uint N2;
    uint N3;
    uint K;
    uint has_awq;
};

// ── QKV batched params struct ──
struct QkvBatchedInt4Params {
    uint N_q;
    uint N_k;
    uint N_v;
    uint K;
    uint has_awq;
};
