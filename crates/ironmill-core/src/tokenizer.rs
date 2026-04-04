//! Tokenizer abstraction (§9 of the API specification).
//!
//! Provides [`Tokenizer`] — a trait for text ↔ token-ID conversion —
//! along with [`ChatMessage`] and [`TokenizerError`].

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------

/// A single message in a chat-style prompt.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// TokenizerError
// ---------------------------------------------------------------------------

/// Errors that can occur during tokenization.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("tokenizer not loaded: {0}")]
    NotLoaded(String),

    #[error("encoding error: {0}")]
    Encode(String),

    #[error("decoding error: {0}")]
    Decode(String),

    #[error("chat template error: {0}")]
    Template(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Tokenizer trait
// ---------------------------------------------------------------------------

/// Text ↔ token ID conversion.
///
/// This trait abstracts over tokenizer implementations (SentencePiece,
/// tiktoken, HuggingFace `tokenizers`, etc.) so that higher-level code
/// can work with any model's vocabulary without hard-coding a backend.
pub trait Tokenizer: Send + Sync {
    /// Encode `text` into a sequence of token IDs.
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;

    /// Decode a sequence of token IDs back into text.
    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError>;

    /// Decode a single token ID. Default delegates to [`Self::decode`].
    fn decode_token(&self, token: u32) -> Result<String, TokenizerError> {
        self.decode(&[token])
    }

    /// Return the vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Return the end-of-sequence token ID(s).
    fn eos_tokens(&self) -> &[u32];

    /// Return the beginning-of-sequence token ID, if applicable.
    fn bos_token(&self) -> Option<u32> {
        None
    }

    /// Format `messages` into a prompt string using the model's chat
    /// template. The default implementation produces a simple
    /// `<|role|>\ncontent\n` format.
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String, TokenizerError> {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
        }
        Ok(prompt)
    }
}

// ---------------------------------------------------------------------------
// HuggingFace tokenizer adapter (§9.2) — gated behind `hf-tokenizer`
// ---------------------------------------------------------------------------

// NOTE: The `HfTokenizer` implementation wrapping the `tokenizers` crate is
// deferred until the `tokenizers` dependency is added to the workspace.  The
// feature flag `hf-tokenizer` will gate both the dependency and this module.
#[cfg(feature = "hf-tokenizer")]
pub mod hf {
    // Placeholder — actual implementation requires the `tokenizers` crate.
    // pub struct HfTokenizer { ... }
}
