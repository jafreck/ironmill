//! Output types for text generation.

use std::time::Duration;

use ironmill_core::tokenizer::Tokenizer;
use ironmill_inference::generate::{CancellationToken, FinishReason, GenerateEvent, TokenStream};

use crate::error::TorchError;

/// The complete result of a non-streaming text generation call.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TextOutput {
    /// The generated text.
    pub text: String,
    /// The generated token IDs.
    pub tokens: Vec<u32>,
    /// Number of generated tokens.
    pub token_count: usize,
    /// Number of prompt tokens processed.
    pub prompt_token_count: usize,
    /// Why generation stopped.
    pub finish_reason: Option<FinishReason>,
    /// Wall-clock generation time.
    pub elapsed: Duration,
}

/// A single chunk emitted during streaming generation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// Decoded text for this token.
    pub text: String,
    /// The token ID.
    pub token: u32,
    /// Zero-based position in the generated output.
    pub position: usize,
    /// Whether generation has finished.
    pub finished: bool,
    /// Finish reason, if `finished` is true.
    pub finish_reason: Option<FinishReason>,
}

/// An iterator that yields [`TextChunk`]s during streaming generation.
///
/// Wraps the low-level [`TokenStream`] from `ironmill-inference` and
/// decodes each token into text using the model's tokenizer.
pub struct TextStream<'a> {
    inner: TokenStream<'a>,
    tokenizer: &'a dyn Tokenizer,
    cancel: CancellationToken,
}

impl<'a> TextStream<'a> {
    pub(crate) fn new(
        inner: TokenStream<'a>,
        tokenizer: &'a dyn Tokenizer,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            inner,
            tokenizer,
            cancel,
        }
    }

    /// Cancel generation. The next iteration will yield a final chunk
    /// with `finished: true` and `finish_reason: Some("Cancelled")`.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

impl Iterator for TextStream<'_> {
    type Item = Result<TextChunk, TorchError>;

    fn next(&mut self) -> Option<Self::Item> {
        let event = self.inner.next()?;
        match event {
            Ok(GenerateEvent::Token {
                token, position, ..
            }) => {
                let text = match self.tokenizer.decode(&[token]) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("token decode failed for token {token}: {e}");
                        "\u{FFFD}".to_string()
                    }
                };
                Some(Ok(TextChunk {
                    text,
                    token,
                    position,
                    finished: false,
                    finish_reason: None,
                }))
            }
            Ok(GenerateEvent::Finished { reason, .. }) => Some(Ok(TextChunk {
                text: String::new(),
                token: 0,
                position: 0,
                finished: true,
                finish_reason: Some(reason),
            })),
            Ok(GenerateEvent::PromptProcessed { .. }) => {
                // Prefill complete — skip to the first generated token.
                self.next()
            }
            // GenerateEvent is #[non_exhaustive]; handle future variants gracefully.
            Ok(_) => self.next(),
            Err(e) => Some(Err(e.into())),
        }
    }
}
