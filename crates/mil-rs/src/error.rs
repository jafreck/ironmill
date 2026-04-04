use thiserror::Error;

/// Errors that can occur when working with MIL IR or CoreML models.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MilError {
    /// An operation referenced a value that doesn't exist in the graph.
    #[error("undefined value: {0}")]
    UndefinedValue(String),

    /// A type mismatch was detected during graph construction or validation.
    #[error("type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    /// An unsupported ONNX operation was encountered during conversion.
    #[error("unsupported operation: {0}")]
    UnsupportedOp(String),

    /// An error occurred during protobuf serialization or deserialization.
    #[error("protobuf error: {0}")]
    Protobuf(String),

    /// An `.mlpackage` directory is malformed or missing required files.
    #[error("invalid package: {0}")]
    InvalidPackage(String),

    /// An I/O error occurred reading or writing model files.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// The model failed validation (e.g., cycles in the graph, dangling references).
    #[error("validation error: {0}")]
    Validation(String),
}

/// Convenience alias for `std::result::Result<T, MilError>`.
pub type Result<T> = std::result::Result<T, MilError>;
