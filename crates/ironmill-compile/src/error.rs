//! Error types for the ironmill-compile crate.

use mil_rs::error::MilError;

/// Errors from the ironmill compilation pipeline.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    /// An error from the MIL IR layer (parsing, validation, serialization).
    #[error(transparent)]
    Mil(#[from] MilError),

    /// I/O error during compilation (reading inputs, writing outputs).
    #[error("compile I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The xcrun compiler is not available on this platform.
    #[error("coremlcompiler not available: {0}")]
    CompilerNotAvailable(String),

    /// xcrun coremlcompiler returned a non-zero exit code.
    #[error("coremlcompiler failed: {0}")]
    CompilerFailed(String),

    /// A generic compilation error.
    #[error("{0}")]
    Other(String),
}

/// Result type for compilation operations.
pub type Result<T> = std::result::Result<T, CompileError>;
