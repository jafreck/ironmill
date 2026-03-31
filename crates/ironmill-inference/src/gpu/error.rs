//! Error types for the Metal GPU inference backend.

use ironmill_metal_sys::MetalSysError;

/// Errors from the GPU inference backend.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// Low-level Metal or MPS error.
    #[error("Metal error: {0}")]
    Metal(#[from] MetalSysError),

    /// Weight loading failed (missing tensor, wrong dtype, etc.).
    #[error("weight loading failed: {0}")]
    WeightLoading(String),

    /// Shader compilation failed.
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// Model architecture is not supported by the GPU backend.
    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Configuration error.
    #[error("config error: {0}")]
    Config(String),

    /// Buffer size mismatch between expected and actual.
    #[error("buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    /// A generic error from an underlying operation.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl From<GpuError> for crate::engine::InferenceError {
    fn from(e: GpuError) -> Self {
        crate::engine::InferenceError::Runtime(e.to_string())
    }
}
