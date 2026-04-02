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

    /// Loading a compiled program into the ANE failed.
    #[error("ANE load failed: {0}")]
    LoadFailed(String),

    /// Unloading a program from the ANE failed.
    #[error("ANE unload failed: {0}")]
    UnloadFailed(String),

    /// The ANE device is not present or not reachable.
    #[error("ANE device not available")]
    DeviceNotAvailable,

    /// Mapping an IOSurface for ANE I/O failed.
    #[error("ANE IOSurface mapping failed: {0}")]
    IOSurfaceMappingFailed(String),

    /// A chaining (multi-segment) request failed.
    #[error("ANE chaining request failed: {0}")]
    ChainingFailed(String),

    /// An ANE request failed validation before submission.
    #[error("ANE request validation failed")]
    RequestValidationFailed,

    /// An ANE API call returned a null pointer unexpectedly.
    #[error("Null pointer returned from ANE API: {context}")]
    NullPointer { context: String },

    /// Setting a session hint on the ANE failed.
    #[error("ANE session hint failed: {0}")]
    SessionHintFailed(String),

    /// Creating an ANE program object failed.
    #[error("ANE program creation failed: {0}")]
    ProgramCreationFailed(String),
}
