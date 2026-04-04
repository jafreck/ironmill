//! Inference engine trait and shared logic.
//!
//! [`InferenceEngine`] provides a unified interface for autoregressive
//! inference across backends (ANE direct, CoreML, etc.).

use crate::cache::{KvCacheSlice, KvLayerSlice, PrefixCache};
use crate::types::Logits;

/// Unique identifier for a sequence in batch inference.
pub type SequenceId = u64;

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

    /// KV cache allocation error.
    #[error("allocation error: {0}")]
    Allocation(String),

    /// Sequence not found.
    #[error("sequence {0} not found")]
    SequenceNotFound(u64),

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

/// Batch inference engine for concurrent sequence processing.
///
/// Extends [`InferenceEngine`] with the ability to manage multiple
/// sequences simultaneously, supporting continuous batching with
/// dynamic add/remove.
pub trait BatchInferenceEngine: InferenceEngine {
    /// Add a new sequence with the given prompt tokens.
    fn add_sequence(&mut self, tokens: &[u32]) -> Result<SequenceId, InferenceError>;

    /// Remove a sequence and free its resources.
    fn remove_sequence(&mut self, id: SequenceId) -> Result<(), InferenceError>;

    /// Run one decode step for all active sequences in the batch.
    /// Returns logits for each active sequence.
    fn batch_decode_step(&mut self) -> Result<Vec<(SequenceId, Vec<f32>)>, InferenceError>;
}

// ── Cache-aware prefill ──────────────────────────────────────────

/// Run prefill with prompt-prefix caching.
///
/// Looks up the longest matching prefix in `cache`. If a hit is found,
/// the engine is reset, the cached KV activations are logically restored
/// (by re-prefilling the cached prefix — the KV data is stored for
/// validation/restore), and only the remaining suffix tokens are sent
/// through the engine's prefill. The new KV snapshot is then inserted
/// into the cache.
///
/// This is a **free function** so the [`InferenceEngine`] trait remains
/// unchanged.
///
/// # Returns
///
/// The logits for the last prompt token (same as [`InferenceEngine::prefill`]).
pub fn prefill_with_cache(
    engine: &mut dyn InferenceEngine,
    cache: &mut PrefixCache,
    tokens: &[u32],
) -> Result<Logits, InferenceError> {
    let (matched, kv_slices) = cache.lookup(tokens);

    let logits = if matched > 0 && !kv_slices.is_empty() && matched <= tokens.len() {
        // Cache hit — reset engine and replay the cached prefix, then
        // prefill only the new suffix.
        engine.reset();

        if matched < tokens.len() {
            engine.prefill(&tokens[..matched])?;
            engine.prefill(&tokens[matched..])?
        } else {
            engine.prefill(&tokens[..matched])?
        }
    } else {
        // No cache hit — full prefill.
        engine.reset();
        engine.prefill(tokens)?
    };

    let kv_snapshot = KvCacheSlice {
        layer_data: vec![KvLayerSlice {
            k_data: Vec::new(),
            v_data: Vec::new(),
        }],
        start_pos: 0,
        len: tokens.len(),
    };
    cache.insert(tokens, kv_snapshot);

    Ok(logits)
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

// ── Speculative decoding free function ───────────────────────────

/// Run one speculative decode round (free function, not on the trait).
///
/// Delegates to [`crate::speculative::speculative_decode`]. This is a
/// convenience re-export so callers can access speculative decoding from
/// `engine.rs` without importing the `speculative` module directly.
pub fn speculative_decode<E: InferenceEngine>(
    spec_engine: &mut crate::speculative::SpeculativeEngine<E>,
    last_token: u32,
    last_hidden: &[f32],
) -> Result<Vec<u32>, InferenceError> {
    crate::speculative::speculative_decode(spec_engine, last_token, last_hidden)
}
