//! High-level generation API — streaming, callbacks, and convenience helpers.
//!
//! The core inference loop (prefill → decode_step → sample → repeat) is
//! low-level and requires callers to manage sampling, stopping, grammar
//! constraints, and cancellation manually. This module provides a
//! composable layer that handles all of this.
//!
//! # Primary interface
//!
//! [`TokenStream`] is a synchronous [`Iterator`] that wraps an
//! [`InferenceEngine`] and yields [`GenerateEvent`] items. For simpler
//! use cases, [`generate()`] collects all tokens and
//! [`generate_with_callback()`] provides inline streaming.
//!
//! # Example
//!
//! ```ignore
//! use ironmill_inference::generate::*;
//!
//! let cancel = CancellationToken::new();
//! let request = GenerateRequest::new(prompt_tokens);
//! let stream = TokenStream::new(&mut engine, request, &cancel);
//!
//! for event in stream {
//!     match event? {
//!         GenerateEvent::Token { token, .. } => print_token(token),
//!         GenerateEvent::Finished { .. } => break,
//!         _ => {}
//!     }
//! }
//! ```

use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::engine::{InferenceEngine, InferenceError};
use crate::grammar::{CompiledGrammar, GrammarState};
use crate::sampling::{Sampler, SamplerConfig};
use crate::types::Logits;

// ── GenerateRequest ──────────────────────────────────────────────

/// All generation parameters in a single request object.
///
/// This replaces the pattern of manually wiring sampler + stop logic + grammar.
/// Use the builder methods to customise individual parameters.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Prompt token IDs.
    pub prompt_tokens: Vec<u32>,
    /// Sampling configuration.
    pub sampler: SamplerConfig,
    /// Maximum number of tokens to generate (excluding prompt).
    pub max_tokens: usize,
    /// Token IDs that signal end of generation.
    pub stop_tokens: Vec<u32>,
    /// Optional grammar constraint.
    pub grammar: Option<Arc<CompiledGrammar>>,
}

impl GenerateRequest {
    /// Create a new request with the given prompt tokens.
    ///
    /// Uses default sampler, 256 max tokens, and empty stop tokens
    /// (meaning the engine's [`ModelInfo::eos_tokens`](crate::model_info::ModelInfo::eos_tokens)
    /// will be used automatically).
    pub fn new(prompt_tokens: Vec<u32>) -> Self {
        Self {
            prompt_tokens,
            sampler: SamplerConfig::default(),
            max_tokens: 256,
            stop_tokens: Vec::new(),
            grammar: None,
        }
    }

    /// Explicitly set stop tokens. If empty (the default), the generation
    /// loop consults [`ModelInfo::eos_tokens`](crate::model_info::ModelInfo::eos_tokens)
    /// from the engine.
    pub fn with_stop_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.stop_tokens = tokens;
        self
    }

    /// Set the sampler configuration for token selection.
    pub fn with_sampler(mut self, config: SamplerConfig) -> Self {
        self.sampler = config;
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Set a compiled grammar to constrain generation output.
    pub fn with_grammar(mut self, grammar: Arc<CompiledGrammar>) -> Self {
        self.grammar = Some(grammar);
        self
    }
}

// ── GenerateEvent & FinishReason ─────────────────────────────────

/// Events emitted during generation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum GenerateEvent {
    /// Prompt processing (prefill) has completed.
    ///
    /// Emitted once, before the first `Token` event.
    PromptProcessed {
        /// Number of prompt tokens processed.
        prompt_tokens: usize,
        /// Wall-clock time for the prefill step.
        elapsed: Duration,
    },
    /// A new token was generated.
    Token {
        /// The sampled token ID.
        token: u32,
        /// Log-probability of the sampled token.
        logprob: f32,
        /// Zero-based position within the generated output (not counting prompt).
        position: usize,
    },
    /// Generation has finished.
    Finished {
        /// The reason generation stopped.
        reason: FinishReason,
        /// Number of tokens generated (excluding prompt).
        tokens_generated: usize,
        /// Number of prompt tokens processed.
        prompt_tokens: usize,
    },
}

/// Why generation stopped.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token (EOS) was generated.
    Stop,
    /// Reached the `max_tokens` limit.
    MaxTokens,
    /// Cancelled by the caller via [`CancellationToken`].
    Cancelled,
    /// Grammar reached an accepting state.
    GrammarComplete,
}

// ── CancellationToken ────────────────────────────────────────────

/// Cooperative cancellation token.
///
/// Clone this token and share it with other threads or an async runtime.
/// Call [`cancel()`](Self::cancel) to signal the generation loop to stop.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token in the non-cancelled state.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal cancellation. The next [`TokenStream::next()`] call will
    /// yield `GenerateEvent::Finished { reason: Cancelled, .. }`.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

// ── TokenStream ──────────────────────────────────────────────────

/// Streaming token generator.
///
/// Created via [`TokenStream::new()`]. Implements [`Iterator`] so it
/// works with `for` loops, `.take()`, `.map()`, `.collect()`, etc.
pub struct TokenStream<'a> {
    engine: &'a mut dyn InferenceEngine,
    sampler: Sampler,
    request: GenerateRequest,
    cancel: &'a CancellationToken,
    grammar_state: Option<GrammarState>,
    position: usize,
    prefilled: bool,
    finished: bool,
    pending_finish: Option<FinishReason>,
    logits: Logits,
    effective_stop_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
}

impl<'a> TokenStream<'a> {
    /// Create a new token stream.
    ///
    /// Prefill is deferred to the first call to `next()`, so construction
    /// is cheap and infallible.
    pub fn new(
        engine: &'a mut dyn InferenceEngine,
        request: GenerateRequest,
        cancel: &'a CancellationToken,
    ) -> Self {
        let grammar_state = request
            .grammar
            .as_ref()
            .map(|g| GrammarState::new(Arc::clone(g)));
        let sampler = Sampler::new(request.sampler.clone());
        Self {
            engine,
            sampler,
            request,
            cancel,
            grammar_state,
            position: 0,
            prefilled: false,
            finished: false,
            pending_finish: None,
            logits: Vec::new(),
            generated_tokens: Vec::new(),
            effective_stop_tokens: Vec::new(),
        }
    }

    /// Tokens generated so far (useful for recovery after an error).
    ///
    /// When `decode_step()` fails mid-generation, the iterator yields
    /// `Some(Err(e))` then `None`. Call this method after the error to
    /// retrieve any tokens generated before the failure.
    pub fn tokens_so_far(&self) -> &[u32] {
        &self.generated_tokens
    }
}

impl Iterator for TokenStream<'_> {
    type Item = Result<GenerateEvent, InferenceError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            // Emit deferred Finished event (e.g., after max_tokens final Token).
            if let Some(reason) = self.pending_finish.take() {
                return Some(Ok(GenerateEvent::Finished {
                    reason,
                    tokens_generated: self.position,
                    prompt_tokens: self.request.prompt_tokens.len(),
                }));
            }
            return None;
        }

        // ── Prefill on first call ──
        if !self.prefilled {
            self.prefilled = true;
            let start = Instant::now();
            match self.engine.prefill(&self.request.prompt_tokens) {
                Ok(logits) => {
                    self.logits = logits;
                    // Resolve stop tokens: use explicit list, or fall back to model's EOS.
                    if self.request.stop_tokens.is_empty() {
                        self.effective_stop_tokens = self.engine.model_info().eos_tokens.clone();
                    } else {
                        self.effective_stop_tokens = self.request.stop_tokens.clone();
                    }
                    return Some(Ok(GenerateEvent::PromptProcessed {
                        prompt_tokens: self.request.prompt_tokens.len(),
                        elapsed: start.elapsed(),
                    }));
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            }
        }

        // ── Check cancellation ──
        if self.cancel.is_cancelled() {
            self.finished = true;
            return Some(Ok(GenerateEvent::Finished {
                reason: FinishReason::Cancelled,
                tokens_generated: self.position,
                prompt_tokens: self.request.prompt_tokens.len(),
            }));
        }

        // ── Apply grammar mask (if constrained) ──
        if let Some(state) = &self.grammar_state {
            let mask = state.token_mask();
            crate::sampling::apply_token_mask(&mut self.logits, &mask);
        }

        // ── Sample ──
        let token = match self.sampler.sample(&mut self.logits) {
            Ok(t) => t,
            Err(e) => {
                self.finished = true;
                return Some(Err(InferenceError::Sampling(e.to_string())));
            }
        };
        let logprob = self
            .logits
            .get(token as usize)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        self.generated_tokens.push(token);

        // ── Advance grammar ──
        if let Some(state) = &mut self.grammar_state {
            state.advance(token);
            if state.is_complete() {
                self.finished = true;
                self.position += 1;
                self.pending_finish = Some(FinishReason::GrammarComplete);
                return Some(Ok(GenerateEvent::Token {
                    token,
                    logprob,
                    position: self.position - 1,
                }));
            }
        }

        // ── Check stop conditions ──
        if self.effective_stop_tokens.contains(&token) {
            self.finished = true;
            self.position += 1;
            self.pending_finish = Some(FinishReason::Stop);
            return Some(Ok(GenerateEvent::Token {
                token,
                logprob,
                position: self.position - 1,
            }));
        }

        self.position += 1;

        if self.position >= self.request.max_tokens {
            self.finished = true;
            // Emit the final token. The *next* call to next() will
            // emit Finished { reason: MaxTokens } (see top of fn).
            self.pending_finish = Some(FinishReason::MaxTokens);
            return Some(Ok(GenerateEvent::Token {
                token,
                logprob,
                position: self.position - 1,
            }));
        }

        // ── Decode next ──
        match self.engine.decode_step(token) {
            Ok(logits) => self.logits = logits,
            Err(e) => {
                self.finished = true;
                return Some(Err(e));
            }
        }

        Some(Ok(GenerateEvent::Token {
            token,
            logprob,
            position: self.position - 1,
        }))
    }
}

// ── GenerateResult ───────────────────────────────────────────────

/// Output from a completed generation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// Generated token IDs (excluding prompt).
    pub tokens: Vec<u32>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// Number of prompt tokens processed.
    pub prompt_tokens: usize,
}

// ── GenerateError ────────────────────────────────────────────────

/// Generation error with partial results.
#[derive(Debug)]
pub struct GenerateError {
    /// The underlying inference error.
    pub source: InferenceError,
    /// Tokens generated before the error occurred.
    /// Empty if the error happened during prefill.
    pub partial_tokens: Vec<u32>,
    /// Number of prompt tokens processed (0 if prefill failed).
    pub prompt_tokens: usize,
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "generation failed after {} tokens: {}",
            self.partial_tokens.len(),
            self.source
        )
    }
}

impl std::error::Error for GenerateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

// ── Convenience functions ────────────────────────────────────────

/// Run generation to completion and collect all output tokens.
pub fn generate(
    engine: &mut dyn InferenceEngine,
    request: &GenerateRequest,
) -> Result<GenerateResult, GenerateError> {
    let cancel = CancellationToken::new();
    let stream = TokenStream::new(engine, request.clone(), &cancel);

    let mut tokens = Vec::new();
    let mut finish_reason = FinishReason::MaxTokens;
    let prompt_len = request.prompt_tokens.len();

    for event in stream {
        match event {
            Ok(GenerateEvent::Token { token, .. }) => tokens.push(token),
            Ok(GenerateEvent::Finished { reason, .. }) => {
                finish_reason = reason;
                break;
            }
            Ok(_) => {} // PromptProcessed and future variants
            Err(e) => {
                return Err(GenerateError {
                    source: e,
                    partial_tokens: tokens,
                    prompt_tokens: prompt_len,
                });
            }
        }
    }

    Ok(GenerateResult {
        tokens,
        finish_reason,
        prompt_tokens: prompt_len,
    })
}

/// Run generation with a per-event callback.
///
/// The callback receives each [`GenerateEvent`] and returns:
/// - `ControlFlow::Continue(())` to keep generating
/// - `ControlFlow::Break(())` to cancel
pub fn generate_with_callback(
    engine: &mut dyn InferenceEngine,
    request: &GenerateRequest,
    mut on_event: impl FnMut(GenerateEvent) -> ControlFlow<(), ()>,
) -> Result<GenerateResult, GenerateError> {
    let cancel = CancellationToken::new();
    let stream = TokenStream::new(engine, request.clone(), &cancel);

    let mut tokens = Vec::new();
    let mut finish_reason = FinishReason::MaxTokens;
    let prompt_len = request.prompt_tokens.len();

    for event in stream {
        let event = match event {
            Ok(e) => e,
            Err(e) => {
                return Err(GenerateError {
                    source: e,
                    partial_tokens: tokens,
                    prompt_tokens: prompt_len,
                });
            }
        };
        match &event {
            GenerateEvent::PromptProcessed { .. } => {}
            GenerateEvent::Token { token, .. } => tokens.push(*token),
            GenerateEvent::Finished { reason, .. } => finish_reason = *reason,
        }
        if on_event(event) == ControlFlow::Break(()) {
            finish_reason = FinishReason::Cancelled;
            break;
        }
    }

    Ok(GenerateResult {
        tokens,
        finish_reason,
        prompt_tokens: prompt_len,
    })
}

#[cfg(feature = "async")]
pub mod generate_async {
    use super::*;
    use tokio::sync::mpsc;

    /// Async streaming token generator.
    ///
    /// Created via [`spawn()`]. Runs the synchronous engine on a
    /// blocking thread and streams events through a channel.
    pub struct AsyncTokenStream {
        receiver: mpsc::Receiver<Result<GenerateEvent, InferenceError>>,
        cancel: CancellationToken,
    }

    impl AsyncTokenStream {
        /// Cancel the generation.
        pub fn cancel(&self) {
            self.cancel.cancel();
        }

        /// Receive the next event, or `None` if the stream is finished.
        pub async fn next(&mut self) -> Option<Result<GenerateEvent, InferenceError>> {
            self.receiver.recv().await
        }
    }

    /// Spawn a generation task on a blocking thread and return an async stream.
    pub fn spawn(
        mut engine: Box<dyn InferenceEngine + Send>,
        request: GenerateRequest,
        buffer: usize,
    ) -> AsyncTokenStream {
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let (tx, rx) = mpsc::channel(buffer);

        tokio::task::spawn_blocking(move || {
            let stream = TokenStream::new(&mut *engine, request, &cancel_clone);
            for event in stream {
                if tx.blocking_send(event).is_err() {
                    break;
                }
            }
        });

        AsyncTokenStream {
            receiver: rx,
            cancel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::InferenceEngine;
    use crate::memory::MemoryUsage;
    use ironmill_core::model_info::ModelInfo;
    use mil_rs::weights::Architecture;

    /// Minimal mock engine for testing the generation API.
    struct MockGenEngine {
        pos: usize,
        vocab_size: usize,
        model_info: ModelInfo,
    }

    impl MockGenEngine {
        fn new(vocab_size: usize) -> Self {
            Self {
                pos: 0,
                vocab_size,
                model_info: ModelInfo {
                    architecture: Architecture::Llama,
                    num_layers: 1,
                    hidden_size: 64,
                    vocab_size,
                    max_context_len: 2048,
                    weight_quantization: String::from("fp16"),
                    eos_tokens: vec![2],
                    param_count_m: 1.0,
                    uses_gqa: false,
                    uses_mla: false,
                    head_dim: 64,
                    num_attention_heads: 1,
                    num_kv_heads: 1,
                },
            }
        }
    }

    impl InferenceEngine for MockGenEngine {
        fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
            self.pos += tokens.len();
            let mut logits = vec![0.0f32; self.vocab_size];
            // Make token 1 most likely.
            if self.vocab_size > 1 {
                logits[1] = 5.0;
            }
            Ok(logits)
        }

        fn decode_step(&mut self, _token: u32) -> Result<Logits, InferenceError> {
            self.pos += 1;
            let mut logits = vec![0.0f32; self.vocab_size];
            if self.vocab_size > 1 {
                logits[1] = 5.0;
            }
            Ok(logits)
        }

        fn reset(&mut self) {
            self.pos = 0;
        }

        fn seq_pos(&self) -> usize {
            self.pos
        }

        fn truncate_to(&mut self, pos: usize) {
            self.pos = pos;
        }

        fn model_info(&self) -> &ModelInfo {
            &self.model_info
        }
    }

    #[test]
    fn generate_basic() {
        let mut engine = MockGenEngine::new(10);
        let request = GenerateRequest::new(vec![0, 1, 2])
            .with_max_tokens(5)
            .with_sampler(SamplerConfig::greedy())
            .with_stop_tokens(vec![99]); // won't hit
        let result = generate(&mut engine, &request).unwrap();
        assert_eq!(result.tokens.len(), 5);
        assert_eq!(result.finish_reason, FinishReason::MaxTokens);
        assert_eq!(result.prompt_tokens, 3);
    }

    #[test]
    fn generate_stops_on_eos() {
        // EOS is token 2, and model_info.eos_tokens = [2].
        // Make token 2 most likely by using a custom engine.
        struct EosEngine {
            pos: usize,
            info: ModelInfo,
        }
        impl InferenceEngine for EosEngine {
            fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
                self.pos += tokens.len();
                let mut logits = vec![0.0f32; 10];
                logits[2] = 10.0; // EOS token
                Ok(logits)
            }
            fn decode_step(&mut self, _: u32) -> Result<Logits, InferenceError> {
                self.pos += 1;
                let mut logits = vec![0.0f32; 10];
                logits[2] = 10.0;
                Ok(logits)
            }
            fn reset(&mut self) {
                self.pos = 0;
            }
            fn seq_pos(&self) -> usize {
                self.pos
            }
            fn truncate_to(&mut self, pos: usize) {
                self.pos = pos;
            }
            fn model_info(&self) -> &ModelInfo {
                &self.info
            }
        }

        let mut engine = EosEngine {
            pos: 0,
            info: ModelInfo {
                architecture: Architecture::Llama,
                num_layers: 1,
                hidden_size: 64,
                vocab_size: 10,
                max_context_len: 2048,
                weight_quantization: String::from("fp16"),
                eos_tokens: vec![2],
                param_count_m: 1.0,
                uses_gqa: false,
                uses_mla: false,
                head_dim: 64,
                num_attention_heads: 1,
                num_kv_heads: 1,
            },
        };

        let request = GenerateRequest::new(vec![0, 1])
            .with_max_tokens(100)
            .with_sampler(SamplerConfig::greedy());
        let result = generate(&mut engine, &request).unwrap();
        assert_eq!(result.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn cancellation_token_works() {
        let cancel = CancellationToken::new();
        assert!(!cancel.is_cancelled());
        cancel.cancel();
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn token_stream_respects_cancellation() {
        let mut engine = MockGenEngine::new(10);
        let cancel = CancellationToken::new();
        cancel.cancel(); // cancel immediately

        let request = GenerateRequest::new(vec![0, 1])
            .with_max_tokens(100)
            .with_sampler(SamplerConfig::greedy())
            .with_stop_tokens(vec![99]);
        let stream = TokenStream::new(&mut engine, request, &cancel);

        let events: Vec<_> = stream.collect::<Vec<_>>();
        // Should get PromptProcessed then Cancelled.
        assert!(events.len() >= 2);
        match &events.last().unwrap() {
            Ok(GenerateEvent::Finished { reason, .. }) => {
                assert_eq!(*reason, FinishReason::Cancelled);
            }
            other => panic!("expected Finished(Cancelled), got {other:?}"),
        }
    }

    #[test]
    fn generate_with_callback_basic() {
        let mut engine = MockGenEngine::new(10);
        let request = GenerateRequest::new(vec![0])
            .with_max_tokens(3)
            .with_sampler(SamplerConfig::greedy())
            .with_stop_tokens(vec![99]);
        let mut seen_tokens = Vec::new();
        let result = generate_with_callback(&mut engine, &request, |event| {
            if let GenerateEvent::Token { token, .. } = event {
                seen_tokens.push(token);
            }
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(result.tokens.len(), 3);
        assert_eq!(seen_tokens.len(), 3);
    }

    #[test]
    fn generate_with_callback_early_cancel() {
        let mut engine = MockGenEngine::new(10);
        let request = GenerateRequest::new(vec![0])
            .with_max_tokens(100)
            .with_sampler(SamplerConfig::greedy())
            .with_stop_tokens(vec![99]);
        let mut count = 0;
        let result = generate_with_callback(&mut engine, &request, |event| {
            if matches!(event, GenerateEvent::Token { .. }) {
                count += 1;
                if count >= 2 {
                    return ControlFlow::Break(());
                }
            }
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(result.finish_reason, FinishReason::Cancelled);
        assert_eq!(result.tokens.len(), 2);
    }
}
