//! Error types for the MLX inference backend.

use ironmill_mlx_sys::MlxSysError;

use crate::engine::InferenceError;

/// Errors from the MLX inference backend.
#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    /// Low-level MLX FFI error.
    #[error("mlx sys: {0}")]
    Sys(#[from] MlxSysError),

    /// Weight loading failed (missing tensor, wrong dtype, etc.).
    #[error("weight loading: {0}")]
    WeightLoading(String),

    /// Tensor shape mismatch.
    #[error("shape mismatch: expected {expected}, got {got}")]
    Shape { expected: String, got: String },

    /// Configuration error.
    #[error("config error: {0}")]
    Config(String),
}

impl From<MlxError> for InferenceError {
    fn from(e: MlxError) -> Self {
        InferenceError::Runtime(e.to_string())
    }
}
