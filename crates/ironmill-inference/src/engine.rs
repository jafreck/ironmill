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

// ── Grammar-constrained decoding ─────────────────────────────────

use crate::grammar::{GrammarState, TokenMask};
use crate::sampling::apply_token_mask;

/// Grammar-constrained decode wrapper.
///
/// Wraps an [`InferenceEngine`] to apply grammar constraints during
/// generation. At each decode step the grammar's pushdown automaton
/// computes a [`TokenMask`] and invalid tokens are set to
/// [`f32::NEG_INFINITY`] before sampling.
///
/// # Usage
///
/// ```ignore
/// let mut decoder = ConstrainedDecoder::new(&mut engine, grammar_state);
/// loop {
///     let logits = decoder.constrained_decode_step(last_token)?;
///     let token = sample_token(&logits, temperature);
///     decoder.accept_token(token);
///     if decoder.is_complete() { break; }
/// }
/// ```
pub struct ConstrainedDecoder<'a, E: InferenceEngine> {
    engine: &'a mut E,
    grammar_state: GrammarState,
}

impl<'a, E: InferenceEngine> ConstrainedDecoder<'a, E> {
    /// Create a new constrained decoder wrapping the given engine.
    pub fn new(engine: &'a mut E, grammar_state: GrammarState) -> Self {
        Self {
            engine,
            grammar_state,
        }
    }

    /// Run one decode step with grammar constraints applied.
    ///
    /// Calls the underlying engine's [`InferenceEngine::decode_step`],
    /// then masks the returned logits so only grammar-valid tokens
    /// remain. The caller should sample from the masked logits and
    /// then call [`accept_token`](Self::accept_token).
    pub fn constrained_decode_step(&mut self, token: u32) -> Result<Logits, InferenceError> {
        let mut logits = self.engine.decode_step(token)?;
        let mask = self.grammar_state.token_mask();
        apply_token_mask(&mut logits, &mask);
        Ok(logits)
    }

    /// Notify the grammar that a token has been accepted.
    ///
    /// Advances the pushdown automaton so the next call to
    /// [`constrained_decode_step`](Self::constrained_decode_step)
    /// produces an updated mask.
    pub fn accept_token(&mut self, token_id: u32) {
        self.grammar_state.advance(token_id);
    }

    /// Check if the grammar is in an accepting state.
    pub fn is_complete(&self) -> bool {
        self.grammar_state.is_complete()
    }

    /// Compute the current token mask without decoding.
    pub fn current_mask(&self) -> TokenMask {
        self.grammar_state.token_mask()
    }

    /// Access the underlying grammar state.
    pub fn grammar_state(&self) -> &GrammarState {
        &self.grammar_state
    }
}
