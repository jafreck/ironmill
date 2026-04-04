//! Inference engine trait and shared logic.
//!
//! [`InferenceEngine`] provides a unified interface for autoregressive
//! inference across backends (ANE direct, CoreML, etc.).

use crate::types::Logits;

/// Errors from the inference engine.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    /// Runtime error (compilation, loading, execution).
    #[error("runtime error: {0}")]
    Runtime(String),

    /// Decode step error.
    #[error("decode error: {0}")]
    Decode(String),

    /// Model not loaded.
    #[error("model not loaded")]
    NotLoaded,

    /// Sampling error.
    #[error("sampling error: {0}")]
    Sampling(String),

    /// A generic error from an underlying operation.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Autoregressive inference engine.
///
/// Provides a backend-agnostic interface for loading models, running
/// prefill, and stepping through decode iterations.
pub trait InferenceEngine {
    /// Load model artifacts. The concrete type of `artifacts` depends
    /// on the backend.
    fn load(&mut self, artifacts: &dyn std::any::Any) -> Result<(), InferenceError>;

    /// Prefill: process all prompt tokens, populating the KV cache.
    /// Returns logits for the last prompt token.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError>;

    /// Decode one token, returning logits for the next token prediction.
    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError>;

    /// Reset all state for a new conversation.
    fn reset(&mut self);

    /// Current sequence position (number of tokens in KV cache).
    fn seq_pos(&self) -> usize;

    /// Truncate KV cache to the given position, discarding tokens after `pos`.
    fn truncate_to(&mut self, pos: usize);
}
