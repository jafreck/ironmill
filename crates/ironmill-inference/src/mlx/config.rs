//! Configuration for the MLX inference backend.

/// Configuration for the MLX inference backend.
///
/// Controls sequence length limits, prefill chunking, TurboQuant
/// quantized KV cache settings, and optimization knobs for eval
/// placement, async prefill, memory management, and profiling.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MlxConfig {
    /// Maximum sequence length for KV cache allocation.
    pub max_seq_len: usize,
    /// Prefill chunk size. `None` = process entire prompt at once.
    pub prefill_chunk_size: Option<usize>,
    /// Enable TurboQuant quantized KV cache compression.
    pub enable_turboquant: bool,
    /// TurboQuant rotation seed (must match between cache write and attention).
    pub rotation_seed: u64,
    /// Number of quantization bits for TurboQuant KV cache (4 or 8).
    pub n_bits: u8,

    // ── Optimization knobs (added by MLX-OPTIMIZE) ──────────────
    /// Eval interval for the TurboQuant path (reserved — currently unused).
    ///
    /// This field is retained for forward-compatibility but has no effect.
    /// The engine always calls `eval()` once per layer before `attend()` to
    /// ensure cache writes are materialized. A previous batched-eval
    /// optimization used this value to defer evals across multiple layers,
    /// but that caused `attend()` to read stale/uninitialized cache data
    /// when the interval was greater than 1.
    pub turboquant_eval_interval: usize,
    /// Use `async_eval()` for prefill chunk overlap.
    ///
    /// When `true` and `prefill_chunk_size` is set, the engine uses
    /// `async_eval()` for all chunks except the last, overlapping graph
    /// construction for chunk N+1 with GPU execution of chunk N.
    pub async_prefill: bool,
    /// Clear the Metal buffer cache on `reset()`.
    ///
    /// When `true`, `reset()` calls `metal_clear_cache()` to release
    /// pooled Metal buffers back to the system.
    pub clear_cache_on_reset: bool,
    /// Enable profiling instrumentation.
    ///
    /// When `true`, the inference loop tracks per-eval timing and
    /// eval call counts, logging them to stderr at the end of each
    /// `run_pipeline` invocation.
    pub profile: bool,
}

impl Default for MlxConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            prefill_chunk_size: None,
            enable_turboquant: false,
            rotation_seed: 42,
            n_bits: 8,
            turboquant_eval_interval: 1,
            async_prefill: false,
            clear_cache_on_reset: false,
            profile: false,
        }
    }
}

impl MlxConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), crate::engine::InferenceError> {
        if self.max_seq_len == 0 {
            return Err(crate::engine::InferenceError::runtime(
                "max_seq_len must be > 0",
            ));
        }
        if self.n_bits != 4 && self.n_bits != 8 {
            return Err(crate::engine::InferenceError::runtime(format!(
                "n_bits must be 4 or 8, got {}",
                self.n_bits
            )));
        }
        if let Some(chunk) = self.prefill_chunk_size {
            if chunk == 0 {
                return Err(crate::engine::InferenceError::runtime(
                    "prefill_chunk_size must be > 0",
                ));
            }
        }
        if self.turboquant_eval_interval == 0 {
            return Err(crate::engine::InferenceError::runtime(
                "turboquant_eval_interval must be >= 1",
            ));
        }
        Ok(())
    }
}
