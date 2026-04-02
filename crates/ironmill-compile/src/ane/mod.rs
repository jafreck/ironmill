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
#[derive(Debug, thiserror::Error)]
pub enum AneCompileError {
    #[error("ANE compilation failed (status {status}): {context}")]
    CompileFailed { status: u32, context: String },
    #[error("ANE evaluation failed (status {status}): {context}")]
    EvalFailed { status: u32, context: String },
    #[error("IOSurface error: {0}")]
    SurfaceError(String),
    #[error("compile budget exhausted: used {used} of {limit}")]
    BudgetExhausted { used: usize, limit: usize },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<ironmill_iosurface::IOSurfaceError> for AneCompileError {
    fn from(e: ironmill_iosurface::IOSurfaceError) -> Self {
        AneCompileError::SurfaceError(e.to_string())
    }
}

/// Result type for ANE compilation operations.
pub type Result<T> = std::result::Result<T, AneCompileError>;
