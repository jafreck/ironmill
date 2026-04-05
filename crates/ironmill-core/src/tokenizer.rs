//! Tokenizer abstraction (§9 of the API specification).
//!
//! Provides [`Tokenizer`] — a trait for text ↔ token-ID conversion —
//! along with [`ChatMessage`], [`Role`], and [`TokenizerError`].

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

/// The role of a participant in a chat conversation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System-level instructions.
    System,
    /// Human user input.
    User,
    /// Model-generated response.
    Assistant,
    /// Tool / function-call result.
    Tool,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        f.write_str(s)
    }
}

impl FromStr for Role {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "tool" => Ok(Role::Tool),
            other => Err(format!("unknown role: {other}")),
        }
    }
}

impl From<&str> for Role {
    /// Convert a role string to a [`Role`], defaulting to [`Role::User`]
    /// for unrecognized values.
    fn from(s: &str) -> Self {
        s.parse().unwrap_or(Role::User)
    }
}

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------

/// A single message in a chat-style prompt.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// The role of the message author.
    pub role: Role,
    /// The text content of the message.
    pub content: String,
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
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
    /// Failed to encode text into tokens.
    #[error("encoding error: {0}")]
    Encode(String),

    /// Failed to decode tokens back into text.
    #[error("decoding error: {0}")]
    Decode(String),

    /// Failed to load a tokenizer from disk.
    #[error("load error: {0}")]
    Load(String),

    /// An I/O error occurred.
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

#[cfg(feature = "hf-tokenizer")]
mod hf_impl {
    use super::*;
    use std::path::Path;

    /// HuggingFace tokenizers adapter.
    ///
    /// Loads `tokenizer.json` from a model directory. Supports SentencePiece,
    /// BPE, Unigram, and WordPiece tokenizers.
    pub struct HfTokenizer {
        inner: tokenizers::Tokenizer,
        eos_tokens: Vec<u32>,
        bos_token: Option<u32>,
        chat_template: Option<String>,
    }

    impl HfTokenizer {
        /// Load from a model directory containing `tokenizer.json`.
        pub fn from_model_dir(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
            let dir = path.as_ref();
            let tokenizer_path = dir.join("tokenizer.json");
            let inner = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| TokenizerError::Load(format!("failed to load tokenizer.json: {e}")))?;

            let (eos_tokens, bos_token) = Self::read_token_config(dir);
            let chat_template = Self::read_chat_template(dir);

            Ok(Self {
                inner,
                eos_tokens,
                bos_token,
                chat_template,
            })
        }

        /// Load from an explicit `tokenizer.json` path.
        ///
        /// Unlike [`from_model_dir`](Self::from_model_dir), this does not read
        /// `tokenizer_config.json` — EOS/BOS tokens and the chat template
        /// will not be populated. Use this when you only have the raw
        /// tokenizer file without the surrounding model directory.
        pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
            let inner = tokenizers::Tokenizer::from_file(path.as_ref())
                .map_err(|e| TokenizerError::Load(format!("failed to load tokenizer: {e}")))?;
            Ok(Self {
                inner,
                eos_tokens: Vec::new(),
                bos_token: None,
                chat_template: None,
            })
        }

        /// Return the chat template string parsed from `tokenizer_config.json`,
        /// if one was present when the tokenizer was loaded.
        pub fn chat_template(&self) -> Option<&str> {
            self.chat_template.as_deref()
        }

        fn read_token_config(dir: &Path) -> (Vec<u32>, Option<u32>) {
            let config_path = dir.join("tokenizer_config.json");
            let Ok(data) = std::fs::read_to_string(&config_path) else {
                return (Vec::new(), None);
            };
            let Ok(config) = serde_json::from_str::<serde_json::Value>(&data) else {
                return (Vec::new(), None);
            };

            let eos = config
                .get("eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|id| vec![id as u32])
                .unwrap_or_default();
            let bos = config
                .get("bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|id| id as u32);
            (eos, bos)
        }

        fn read_chat_template(dir: &Path) -> Option<String> {
            let config_path = dir.join("tokenizer_config.json");
            let data = std::fs::read_to_string(&config_path).ok()?;
            let config: serde_json::Value = serde_json::from_str(&data).ok()?;
            config
                .get("chat_template")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        }
    }

    impl Tokenizer for HfTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
            self.inner
                .encode(text, false)
                .map(|enc| enc.get_ids().to_vec())
                .map_err(|e| TokenizerError::Encode(e.to_string()))
        }

        fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
            self.inner
                .decode(tokens, true)
                .map_err(|e| TokenizerError::Decode(e.to_string()))
        }

        fn vocab_size(&self) -> usize {
            self.inner.get_vocab_size(false)
        }
        fn eos_tokens(&self) -> &[u32] {
            &self.eos_tokens
        }
        fn bos_token(&self) -> Option<u32> {
            self.bos_token
        }

        fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String, TokenizerError> {
            let mut prompt = String::new();
            for msg in messages {
                prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
            }
            Ok(prompt)
        }
    }
}

#[cfg(feature = "hf-tokenizer")]
pub use hf_impl::HfTokenizer;
