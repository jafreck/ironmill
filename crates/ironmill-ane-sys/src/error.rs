//! Error types for ANE FFI operations.

/// Errors from low-level ANE FFI operations.
#[derive(Debug, thiserror::Error)]
pub enum AneSysError {
    /// The ANE compiler framework could not be loaded.
    #[error("ANE framework not available: {0}")]
    FrameworkNotFound(String),

    /// A required ObjC class was not found in the loaded framework.
    #[error("ANE class not found: {0}")]
    ClassNotFound(String),

    /// The compiler returned a runtime error.
    #[error("ANE compilation failed: {0}")]
    CompilationFailed(String),

    /// ANE evaluation failed.
    #[error("ANE eval failed (status {status:#x}): {context}")]
    EvalFailed { status: u32, context: String },

    /// An I/O error occurred while preparing inputs or reading outputs.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The input is invalid.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The per-process ANE compile budget has been exhausted.
    #[error("ANE compile budget exhausted ({count}/119 compilations)")]
    BudgetExhausted { count: usize },
}
