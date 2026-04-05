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
    /// Role of the message sender (e.g., "system", "user", "assistant").
    pub role: String,
    /// Text content of the message.
    pub content: String,
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    /// Create an assistant message.
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
    /// The tokenizer has not been initialized or loaded.
    #[error("tokenizer not loaded: {0}")]
    NotLoaded(String),

    /// Failed to encode text into token IDs.
    #[error("encoding error: {0}")]
    Encode(String),

    /// Failed to decode token IDs back into text.
    #[error("decoding error: {0}")]
    Decode(String),

    /// Failed to apply or render the chat template.
    #[error("chat template error: {0}")]
    Template(String),

    /// Failed to load a tokenizer from disk.
    #[error("load error: {0}")]
    Load(String),

    /// An I/O error occurred while reading tokenizer files.
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

        /// Load from an explicit tokenizer.json path.
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
