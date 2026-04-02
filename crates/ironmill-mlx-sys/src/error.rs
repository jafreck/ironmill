//! Error types for MLX FFI operations.

/// Errors from low-level mlx-c FFI operations.
#[derive(Debug, thiserror::Error)]
pub enum MlxSysError {
    /// An error was returned by the mlx-c library.
    #[error("mlx-c error: {0}")]
    MlxC(String),

    /// A Metal kernel compilation failed.
    #[error("kernel compilation failed: {0}")]
    KernelCompile(String),

    /// An array had an unexpected dtype.
    #[error("invalid dtype: expected {expected}, got {got}")]
    InvalidDtype {
        /// The expected dtype name.
        expected: String,
        /// The actual dtype name.
        got: String,
    },

    /// A build/configuration error occurred.
    #[error("build error: {0}")]
    Build(String),
}
