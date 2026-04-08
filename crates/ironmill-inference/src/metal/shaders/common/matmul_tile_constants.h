#pragma once

// ── Shared matmul tile constants ──
// Used by affine, quantized, d2quant, and dense matmul kernels.
// Individual files define MATMUL_K_TILE and K_BLOCKS locally since
// these vary per quantization format.

constant constexpr uint N_SIMDGROUPS   = 8;
constant constexpr uint THREADS_PER_TG = N_SIMDGROUPS * 32;
constant constexpr uint TM_TILE        = N_SIMDGROUPS * 8;   // 64 for 8 SG
constant constexpr uint TN_TILE        = 64;
constant constexpr uint TN_STRIDE      = TN_TILE + 1;
constant constexpr uint TN_BLOCKS      = TN_TILE / 8;
