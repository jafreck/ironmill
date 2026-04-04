//! Error types for the high-level Model API (§10.3).

use std::path::PathBuf;

use crate::device::Device;
use crate::tokenizer::TokenizerError;

/// Errors produced by the high-level model API.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("inference error: {0}")]
    Inference(String),

    #[error("compile error: {0}")]
    Compile(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("model not found: {0}")]
    NotFound(PathBuf),

    #[error("unsupported device {0:?} on this platform")]
    UnsupportedDevice(Device),

    #[error("{0}")]
    Other(String),
}
