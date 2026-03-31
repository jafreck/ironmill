//! Error types for Metal FFI operations.

/// Errors from low-level Metal and MPS FFI operations.
#[derive(Debug, thiserror::Error)]
pub enum MetalSysError {
    /// The Metal framework could not be loaded.
    #[error("Metal framework not available")]
    FrameworkNotFound,

    /// No Metal-capable GPU was found on this system.
    #[error("no Metal-capable GPU device found")]
    NoDevice,

    /// A Metal buffer allocation failed.
    #[error("Metal buffer allocation failed: {0}")]
    BufferAllocation(String),

    /// Shader source compilation failed.
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// Compute pipeline creation failed.
    #[error("pipeline creation failed: {0}")]
    PipelineCreation(String),

    /// A command buffer error occurred.
    #[error("command buffer error: {0}")]
    CommandBuffer(String),

    /// An MPS operation failed.
    #[error("MPS error: {0}")]
    Mps(String),

    /// An invalid argument was provided.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
