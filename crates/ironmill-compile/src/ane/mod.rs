//! ANE compilation pipeline: passes, splitting, packing, validation, and caching.

pub mod blobfile;
pub mod bundle;
pub mod cache;
pub mod decode_compile;
pub mod packing;
pub mod passes;
pub mod split;
pub mod validate;

/// Errors produced by the ANE compilation pipeline.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum AneCompileError {
    /// ANE compilation returned a non-zero status code.
    #[error("ANE compilation failed (status {status}): {context}")]
    CompileFailed {
        /// Non-zero status code returned by the ANE compiler.
        status: u32,
        /// Human-readable description of the failure.
        context: String,
    },
    /// ANE program evaluation returned a non-zero status code.
    #[error("ANE evaluation failed (status {status}): {context}")]
    EvalFailed {
        /// Non-zero status code returned by the ANE evaluator.
        status: u32,
        /// Human-readable description of the failure.
        context: String,
    },
    /// The per-process compile budget has been exhausted.
    #[error("compile budget exhausted: used {used} of {limit}")]
    BudgetExhausted {
        /// Number of compilations already performed.
        used: usize,
        /// Maximum compilations allowed per process.
        limit: usize,
    },
    /// I/O error during ANE bundle operations.
    #[error("ANE I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization/deserialization error.
    #[error("ANE serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    /// Error from the MIL IR layer.
    #[error(transparent)]
    Mil(#[from] mil_rs::error::MilError),
    /// Error from the compilation pipeline.
    #[error(transparent)]
    Compile(#[from] crate::error::CompileError),
    /// Opaque error forwarded from an underlying subsystem.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type for ANE compilation operations.
pub type Result<T> = std::result::Result<T, AneCompileError>;
