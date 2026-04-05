//! Error types for the high-level model API.

use std::path::PathBuf;

use ironmill_compile::error::CompileError;
use ironmill_core::device::Device;
use ironmill_core::tokenizer::TokenizerError;
use ironmill_inference::engine::InferenceError;

/// Errors from the high-level model API.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum TorchError {
    /// An error from the inference engine.
    #[error("inference: {0}")]
    Inference(#[from] InferenceError),

    /// An error from the compilation pipeline.
    #[error("compile: {0}")]
    Compile(#[from] CompileError),

    /// An error from tokenization.
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),

    /// The model path does not exist or is unreadable.
    #[error("model not found: {0}")]
    NotFound(PathBuf),

    /// The requested device is not available on this platform.
    #[error("unsupported device {0:?} on this platform")]
    UnsupportedDevice(Device),

    /// The model format could not be detected or is unsupported.
    #[error("unknown model format at {0}")]
    UnknownFormat(PathBuf),
}
