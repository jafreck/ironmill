//! Metal kernel source strings for TurboQuant and PolarQuant on MLX.
//!
//! Kernel logic lives in `.metal` files under `../shaders/` for proper
//! syntax highlighting, Metal linting and easier diffing.  This module
//! assembles the final source strings at compile time via
//! `concat!(include_str!(...))`.

// ── TurboQuant kernels ──────────────────────────────────────────
//
// Each TurboQuant kernel needs the shared helpers (hadamard rotation,
// cache addressing, quantized tile readers) prepended because Metal
// source passed to `metal_kernel()` cannot use `#include` to resolve
// sibling files.

/// TurboQuant cache write kernel source (helpers prepended).
///
/// MLX buffer order (inputs 0..10, output at index 10):
///   0: kv_proj [num_kv_heads × head_dim] half
///   1: rotation_signs [head_dim] float
///   2: cache (read/write — passed as input)
///   3: scale_buf (read/write — passed as input)
///   4: codebook [n_levels] float
///   5: boundaries [n_levels-1] float
///   6: qjl_matrix [head_dim × head_dim] float
///   7: qjl_signs_buf (read/write)
///   8: r_norms_buf (read/write)
///   9: params [7] uint32 — packed {num_kv_heads, head_dim, max_seq_len, seq_pos, n_bits, n_levels, is_k_cache}
///  Output 0: dummy [1] float (kernel writes to cache in-place)
pub const TURBOQUANT_CACHE_WRITE: &str = concat!(
    include_str!("../shaders/mlx_helpers.metal"),
    "\n",
    include_str!("../shaders/mlx_turboquant_cache_write.metal"),
);

/// Tiled flash attention with quantized KV cache and online softmax.
///
/// MLX buffer order:
///   0: q [num_heads × head_dim] half
///   1: k_cache (quantized)
///   2: v_cache (quantized)
///   3: rotation_signs [head_dim] float
///   4: k_scale_buf [num_kv_heads × max_seq_len] float
///   5: v_scale_buf [num_kv_heads × max_seq_len] float
///   6: k_codebook [n_levels] float
///   7: v_codebook [n_levels] float
///   8: qjl_matrix [head_dim × head_dim] float
///   9: k_r_norms [num_kv_heads × max_seq_len] float
///  10: params [6] uint32 — {num_heads, num_kv_heads, head_dim, max_seq_len, seq_len, n_bits}
///  Output 0: output [num_heads × head_dim] half
pub const TURBOQUANT_ATTENTION: &str = concat!(
    include_str!("../shaders/mlx_helpers.metal"),
    "\n",
    include_str!("../shaders/mlx_turboquant_attention.metal"),
);

/// Outlier-aware cache write: splits KV into outlier and non-outlier
/// channel groups, applies independent TurboQuant to each.
///
/// MLX buffer order:
///   0: kv_proj [num_kv_heads × head_dim] half
///   1: channel_indices [head_dim] uint
///   2–3: outlier_cache / non_outlier_cache (rw)
///   4–5: rotation signs (outlier / non-outlier)
///   6–9: codebooks & boundaries (outlier / non-outlier)
///  10–11: scale bufs (outlier / non-outlier)
///  12–13: qjl matrices
///  14–15: r_norms bufs
///  16: params [10] uint
///  Output 0: dummy [1] float
pub const TURBOQUANT_OUTLIER_CACHE_WRITE: &str = concat!(
    include_str!("../shaders/mlx_helpers.metal"),
    "\n",
    include_str!("../shaders/mlx_turboquant_outlier_cache_write.metal"),
);

/// Attention with dual-group quantized KV cache + QJL correction.
///
/// MLX buffer order:
///   0: q [num_heads × head_dim] half
///   1–4: K/V outlier/non-outlier caches
///   5: channel_indices [head_dim] uint
///   6–7: rotation signs
///   8–9: codebooks (K)
///  10–13: scale bufs (K/V × outlier/non-outlier)
///  14–15: qjl matrices
///  16–17: k_r_norms
///  18–19: codebooks (V)
///  20: params [8] uint
///  Output 0: output [num_heads × head_dim] half
pub const TURBOQUANT_OUTLIER_ATTENTION: &str = concat!(
    include_str!("../shaders/mlx_helpers.metal"),
    "\n",
    include_str!("../shaders/mlx_turboquant_outlier_attention.metal"),
);

// ── PolarQuant kernels ──────────────────────────────────────────
//
// All PolarQuant variants live in a single `.metal` file.  Each
// `metal_kernel()` call selects the appropriate entry point by name.

/// Combined PolarQuant kernel source (all 6 variants).
pub const POLARQUANT_SOURCE: &str = include_str!("../shaders/mlx_polarquant.metal");

/// PolarQuant INT4 matvec (decode path, M=1).
pub const POLARQUANT_MATVEC_INT4: &str = POLARQUANT_SOURCE;

/// PolarQuant INT4 tiled GEMM (prefill path, M>1).
pub const POLARQUANT_MATMUL_INT4: &str = POLARQUANT_SOURCE;

/// PolarQuant INT8 matvec (decode path, M=1).
pub const POLARQUANT_MATVEC_INT8: &str = POLARQUANT_SOURCE;

/// PolarQuant INT8 tiled GEMM (prefill path, M>1).
pub const POLARQUANT_MATMUL_INT8: &str = POLARQUANT_SOURCE;
