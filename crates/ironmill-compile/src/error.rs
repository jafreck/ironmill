//! Error types for the ironmill-compile crate.

use std::path::PathBuf;

use mil_rs::error::MilError;

/// Errors from the ironmill compilation pipeline.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    /// An error from the MIL IR layer (parsing, validation, serialization).
    #[error(transparent)]
    Mil(#[from] MilError),

    /// I/O error during compilation (reading inputs, writing outputs).
    #[error("compile I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The xcrun compiler is not available at the given path.
    #[error("coremlcompiler not found at {path}")]
    CompilerNotAvailable {
        /// The path where the compiler was expected.
        path: PathBuf,
    },

    /// xcrun coremlcompiler returned a non-zero exit code.
    #[error("coremlcompiler failed with exit code {exit_code}: {stderr}")]
    CompilerFailed {
        /// The process exit code.
        exit_code: i32,
        /// Captured stderr output.
        stderr: String,
    },

    /// The requested quantization format is not supported.
    #[error("unsupported quantization: {0}")]
    UnsupportedQuantization(String),

    /// The model architecture is not supported.
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    /// An error occurred while loading or processing weight tensors.
    #[error("weight load error: {0}")]
    WeightLoadError(String),

    /// A generic compilation error.
    #[error("{0}")]
    Other(String),
}

/// Result type for compilation operations.
pub type Result<T> = std::result::Result<T, CompileError>;
