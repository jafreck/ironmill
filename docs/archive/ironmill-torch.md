# ironmill-torch: High-Level Inference API

> **Status**: Proposal
> **Depends on**: `api-spec` branch (merged)
> **Goal**: Provide a PyTorch-level abstraction for model loading,
> generation, and chat — sitting on top of `ironmill-compile` and
> `ironmill-inference` without polluting the foundational `ironmill-core`
> crate.

---

## 1. Motivation

The `api-spec` branch introduced a high-level `Model` / `ChatSession` API
(spec §10) inside `ironmill-core`. This was architecturally wrong:

```
ironmill-core  ←── ironmill-compile
               ←── ironmill-inference
```

`ironmill-core` is foundational shared code. It cannot depend on
`ironmill-inference` (circular) or `ironmill-compile`, so the §10 types
ended up as `todo!()` stubs backed by `Box<dyn Any>` type erasure.

The fix is a new crate that sits **above** both:

```
ironmill-core       (shared types, traits, Tokenizer, ChatMessage, Device)
  ↑          ↑
  │          │
ironmill-compile   ironmill-inference
  ↑          ↑
  └────┬─────┘
  ironmill-torch   (NEW — high-level Model, ChatSession, GenParams)
       ↑
  ironmill-cli
```

`ironmill-torch` is the "batteries-included" crate for users who want a
simple API. `ironmill-compile` and `ironmill-inference` remain the
user-facing low-level primitives for power users.

---

## 2. Migration from ironmill-core

### 2.1 Types That Move to ironmill-torch

These types currently live in `ironmill-core` but belong in
`ironmill-torch` because they require both compilation and inference
crates to function:

| Type | Current location | Reason it moves |
|------|------------------|-----------------|
| `Model` | `core/src/model.rs` | Wraps `InferenceEngine` + `WeightProvider` |
| `ModelBuilder` | `core/src/model.rs` | Calls `MetalInference::load()`, `SafeTensorsProvider::load()` |
| `ChatSession` | `core/src/chat.rs` | Drives `Model::generate` / `Model::stream` |
| `ChatSessionBuilder` | `core/src/chat.rs` | Constructs `ChatSession` with engine binding |
| `TextOutput` | `core/src/text_output.rs` | Wraps `GenerateResult` from inference |
| `TextChunk` | `core/src/text_output.rs` | Wraps `GenerateEvent::Token` from inference |
| `TextStream` | `core/src/text_output.rs` | Wraps `TokenStream` from inference |
| `GenParams` | `core/src/gen_params.rs` | Maps to `SamplerConfig` + `GenerateRequest` |
| `ModelError` | `core/src/error.rs` | Wraps `InferenceError` + `CompileError` |

### 2.2 Types That Stay in ironmill-core

These types are genuinely shared and have no dependency on inference or
compile internals:

| Type | Location | Used by |
|------|----------|---------|
| `Device` | `core/src/device.rs` | compile, inference, torch, CLI |
| `Tokenizer` trait | `core/src/tokenizer.rs` | inference (for EOS tokens), torch (for encode/decode) |
| `HfTokenizer` | `core/src/tokenizer.rs` | torch (primary consumer) |
| `ChatMessage` | `core/src/tokenizer.rs` | torch (chat API), inference (serving) |
| `TokenizerError` | `core/src/tokenizer.rs` | anywhere tokenization happens |
| `ModelInfo` | `core/src/model_info.rs` | inference (engine metadata), torch (display) |
| `ane::*` | `core/src/ane/` | compile, inference (bundle formats) |
| `gpu::*` | `core/src/gpu/` | compile, inference (bundle formats) |
| `weights::*` | `core/src/weights.rs` | compile, inference (re-exports from mil-rs) |

### 2.3 Migration Steps

1. Create `crates/ironmill-torch/` with deps on `ironmill-core`,
   `ironmill-compile`, and `ironmill-inference`.
2. Move the 9 types listed in §2.1 from `ironmill-core` to
   `ironmill-torch`, rewriting their implementations to use concrete
   inference/compile types instead of `dyn Any`.
3. Remove the moved files from `ironmill-core` entirely. No deprecated
   re-exports needed — ironmill is unreleased, so there are no external
   consumers to maintain backward compatibility for. Prefer clean breaks
   over compatibility shims.
4. Update `ironmill-cli` to depend on `ironmill-torch` instead of
   importing these types from `ironmill-core`.

---

## 3. Crate Structure

```
crates/ironmill-torch/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── model.rs          # Model, ModelBuilder
    ├── chat.rs           # ChatSession, ChatSessionBuilder
    ├── gen_params.rs     # GenParams
    ├── text_output.rs    # TextOutput, TextChunk, TextStream
    └── error.rs          # TorchError
```

### 3.1 Cargo.toml

```toml
[package]
name = "ironmill-torch"
description = "High-level PyTorch-style model loading and inference API"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[features]
default = ["metal"]
metal = ["ironmill-inference/metal"]
ane = ["ironmill-inference/ane"]
coreml = ["ironmill-inference/coreml"]
hf-tokenizer = ["ironmill-core/hf-tokenizer"]
async = ["ironmill-inference/async"]

[dependencies]
ironmill-core = { workspace = true }
ironmill-compile = { workspace = true }
ironmill-inference = { workspace = true }
mil-rs = { workspace = true }
thiserror = { workspace = true }
```

### 3.2 lib.rs — Public API Surface

```rust
#![warn(missing_docs)]
//! High-level model loading and inference.
//!
//! `ironmill-torch` provides a PyTorch-level abstraction over the
//! lower-level `ironmill-compile` and `ironmill-inference` crates.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use ironmill_torch::{Model, GenParams};
//!
//! let mut model = Model::from_pretrained("./Qwen3-0.6B/")
//!     .build()?;
//!
//! let output = model.generate("What is Rust?", &GenParams::default())?;
//! println!("{}", output.text);
//! # Ok::<(), ironmill_torch::TorchError>(())
//! ```

mod chat;
mod error;
mod gen_params;
mod model;
mod text_output;

pub use chat::{ChatSession, ChatSessionBuilder};
pub use error::TorchError;
pub use gen_params::GenParams;
pub use model::{Model, ModelBuilder};
pub use text_output::{TextChunk, TextOutput, TextStream};

// Re-export commonly needed types from lower crates so users
// don't need to add them as direct dependencies.
pub use ironmill_core::{ChatMessage, Device, ModelInfo, Tokenizer};
```

---

## 4. Type Definitions

### 4.1 TorchError

```rust
use std::path::PathBuf;
use ironmill_core::{Device, TokenizerError};
use ironmill_compile::CompileError;
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
```

### 4.2 Model

```rust
use std::path::{Path, PathBuf};
use ironmill_core::{Device, ModelInfo, Tokenizer};
use ironmill_inference::engine::InferenceEngine;
use ironmill_inference::generate::{
    GenerateRequest, TokenStream, CancellationToken, GenerateEvent,
};
use mil_rs::ProgressSink;

use crate::chat::ChatSessionBuilder;
use crate::error::TorchError;
use crate::gen_params::GenParams;
use crate::text_output::{TextOutput, TextStream as TorchTextStream};

/// A loaded model ready for inference.
///
/// `Model` owns the inference engine and tokenizer. It provides
/// high-level methods for text generation, streaming, and chat.
///
/// # Example
///
/// ```rust,no_run
/// use ironmill_torch::{Model, GenParams, Device};
///
/// let mut model = Model::from_pretrained("./model/")
///     .device(Device::Metal)
///     .max_seq_len(8192)
///     .build()?;
///
/// let output = model.generate("Hello, world!", &GenParams::default())?;
/// println!("{}", output.text);
/// # Ok::<(), ironmill_torch::TorchError>(())
/// ```
pub struct Model {
    engine: Box<dyn InferenceEngine>,
    tokenizer: Box<dyn Tokenizer>,
    info: ModelInfo,
}

impl Model {
    /// Begin building a model from a pretrained weight directory.
    ///
    /// Supports SafeTensors directories. The builder auto-detects the
    /// model architecture from `config.json`.
    pub fn from_pretrained(path: impl AsRef<Path>) -> ModelBuilder {
        ModelBuilder::new(path.as_ref().to_path_buf(), ModelSource::Pretrained)
    }

    /// Begin building a model from a pre-compiled artifact.
    ///
    /// Supports `.ironml-gpu` bundles (Metal) and `.mlmodelc` (CoreML/ANE).
    pub fn from_compiled(path: impl AsRef<Path>) -> ModelBuilder {
        ModelBuilder::new(path.as_ref().to_path_buf(), ModelSource::Compiled)
    }

    /// Run text generation to completion (non-streaming).
    pub fn generate(
        &mut self,
        prompt: &str,
        params: &GenParams,
    ) -> Result<TextOutput, TorchError> {
        let prompt_tokens = self.tokenizer.encode(prompt)?;
        let request = params.to_generate_request(prompt_tokens, &self.info);
        let cancel = CancellationToken::new();
        let stream = TokenStream::new(&mut *self.engine, request, &cancel);

        let mut tokens = Vec::new();
        let mut finish_reason = String::from("max_tokens");
        let start = std::time::Instant::now();
        let prompt_token_count = prompt_tokens.len();

        for event in stream {
            match event? {
                GenerateEvent::Token { token, .. } => tokens.push(token),
                GenerateEvent::Finished { reason, .. } => {
                    finish_reason = format!("{reason:?}");
                    break;
                }
                _ => {}
            }
        }

        let text = self.tokenizer.decode(&tokens)?;
        Ok(TextOutput {
            text,
            token_count: tokens.len(),
            tokens,
            prompt_token_count,
            finish_reason,
            elapsed: start.elapsed(),
        })
    }

    /// Generate text with streaming output.
    ///
    /// Returns an iterator of [`TextChunk`](crate::TextChunk)s. Each
    /// chunk contains one decoded token. The iterator completes when
    /// generation finishes (EOS, max tokens, or cancellation).
    pub fn stream<'a>(
        &'a mut self,
        prompt: &str,
        params: &GenParams,
    ) -> Result<TorchTextStream<'a>, TorchError> {
        let prompt_tokens = self.tokenizer.encode(prompt)?;
        let request = params.to_generate_request(prompt_tokens, &self.info);
        let cancel = CancellationToken::new();
        let stream = TokenStream::new(&mut *self.engine, request, &cancel);
        Ok(TorchTextStream::new(stream, &*self.tokenizer, cancel))
    }

    /// Start a multi-turn chat session.
    pub fn chat(&mut self) -> ChatSessionBuilder<'_> {
        ChatSessionBuilder::new(self)
    }

    /// Access the underlying inference engine.
    pub fn engine(&self) -> &dyn InferenceEngine {
        &*self.engine
    }

    /// Access the underlying inference engine mutably.
    pub fn engine_mut(&mut self) -> &mut dyn InferenceEngine {
        &mut *self.engine
    }

    /// Access the tokenizer.
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        &*self.tokenizer
    }

    /// Model metadata.
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }
}
```

### 4.3 ModelBuilder

```rust
enum ModelSource {
    Pretrained,
    Compiled,
}

/// Builder for configuring and loading a [`Model`].
pub struct ModelBuilder {
    path: PathBuf,
    source: ModelSource,
    device: Device,
    max_seq_len: usize,
    tokenizer_path: Option<PathBuf>,
    progress: Box<dyn ProgressSink>,
}

impl ModelBuilder {
    fn new(path: PathBuf, source: ModelSource) -> Self {
        Self {
            path,
            source,
            device: Device::Auto,
            max_seq_len: 4096,
            tokenizer_path: None,
            progress: Box::new(mil_rs::NullProgress),
        }
    }

    /// Select the compute device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device; self
    }

    /// Set the maximum sequence length.
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len; self
    }

    /// Override the tokenizer path.
    pub fn tokenizer(mut self, path: impl AsRef<Path>) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_path_buf()); self
    }

    /// Provide a progress sink for loading feedback.
    pub fn with_progress(mut self, sink: impl ProgressSink + 'static) -> Self {
        self.progress = Box::new(sink); self
    }

    /// Load the model and return a ready-to-use [`Model`].
    ///
    /// This is where the actual engine initialization happens:
    /// - Detects model format from `config.json`
    /// - Loads tokenizer from the model directory (or override path)
    /// - For `from_pretrained`: loads weights via `SafeTensorsProvider`,
    ///   creates the appropriate engine (Metal, ANE, etc.)
    /// - For `from_compiled`: opens the artifact bundle and loads into
    ///   the matching engine
    pub fn build(self) -> Result<Model, TorchError> {
        let device = self.resolve_device();
        self.progress.on_stage("loading tokenizer");
        let tokenizer = self.load_tokenizer()?;

        self.progress.on_stage("loading model");
        let (engine, info) = match self.source {
            ModelSource::Pretrained => self.load_pretrained(device)?,
            ModelSource::Compiled => self.load_compiled(device)?,
        };

        Ok(Model { engine, tokenizer, info })
    }

    fn resolve_device(&self) -> Device {
        match self.device {
            Device::Auto => {
                #[cfg(target_os = "macos")]
                { Device::Metal }
                #[cfg(not(target_os = "macos"))]
                { Device::Cpu }
            }
            other => other,
        }
    }

    fn load_tokenizer(&self) -> Result<Box<dyn Tokenizer>, TorchError> {
        let tok_dir = self.tokenizer_path.as_deref()
            .unwrap_or(&self.path);
        // Prefer HfTokenizer if the feature is enabled
        #[cfg(feature = "hf-tokenizer")]
        {
            use ironmill_core::HfTokenizer;
            if tok_dir.join("tokenizer.json").exists() {
                let tok = HfTokenizer::from_model_dir(tok_dir)?;
                return Ok(Box::new(tok));
            }
        }
        Err(TorchError::NotFound(tok_dir.join("tokenizer.json")))
    }

    fn load_pretrained(
        &self,
        device: Device,
    ) -> Result<(Box<dyn InferenceEngine>, ModelInfo), TorchError> {
        // Implementation loads SafeTensorsProvider, creates MetalConfig/
        // AneConfig based on `device`, calls the appropriate backend's
        // load() method.
        match device {
            #[cfg(target_os = "macos")]
            Device::Metal => {
                use ironmill_compile::weights::SafeTensorsProvider;
                use ironmill_inference::metal::{MetalInference, MetalConfig};

                let provider = SafeTensorsProvider::load(&self.path)
                    .map_err(|e| TorchError::Compile(e.into()))?;
                let config = MetalConfig::new()
                    .with_max_seq_len(self.max_seq_len);
                let info = ModelInfo::from_config(provider.config());

                // Load weights into Metal engine
                let engine = MetalInference::load_from_provider(
                    config, &provider, &self.progress,
                ).map_err(TorchError::Inference)?;

                Ok((Box::new(engine), info))
            }
            _ => Err(TorchError::UnsupportedDevice(device)),
        }
    }

    fn load_compiled(
        &self,
        device: Device,
    ) -> Result<(Box<dyn InferenceEngine>, ModelInfo), TorchError> {
        match device {
            #[cfg(target_os = "macos")]
            Device::Metal => {
                use ironmill_inference::metal::{
                    MetalInference, MetalConfig, MetalBundleProvider,
                };

                let bundle = MetalBundleProvider::open(&self.path)
                    .map_err(|e| TorchError::Inference(e))?;
                let config = MetalConfig::new()
                    .with_max_seq_len(self.max_seq_len);
                let info = ModelInfo::from_config(bundle.config());

                let engine = MetalInference::load(config, &bundle)
                    .map_err(TorchError::Inference)?;

                Ok((Box::new(engine), info))
            }
            _ => Err(TorchError::UnsupportedDevice(device)),
        }
    }
}
```

### 4.4 GenParams

```rust
use ironmill_core::ModelInfo;
use ironmill_inference::generate::GenerateRequest;
use ironmill_inference::sampling::SamplerConfig;

/// Controls sampling behaviour during text generation.
///
/// This is the user-facing knob set. Internally it maps to
/// [`SamplerConfig`] + [`GenerateRequest`] from `ironmill-inference`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GenParams {
    pub temperature: f32,
    pub max_tokens: usize,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub stop_tokens: Vec<u32>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: 512,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.0,
            stop_tokens: Vec::new(),
        }
    }
}

impl GenParams {
    /// Greedy decoding (temperature = 0).
    pub fn greedy() -> Self {
        Self { temperature: 0.0, ..Default::default() }
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t; self
    }
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n; self
    }
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p; self
    }
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k; self
    }
    pub fn with_min_p(mut self, p: f32) -> Self {
        self.min_p = p; self
    }
    pub fn with_stop_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.stop_tokens = tokens; self
    }

    /// Convert to the low-level inference request.
    pub(crate) fn to_generate_request(
        &self,
        prompt_tokens: Vec<u32>,
        info: &ModelInfo,
    ) -> GenerateRequest {
        let sampler = SamplerConfig::default()
            .with_temperature(self.temperature)
            .with_top_p(self.top_p)
            .with_top_k(self.top_k)
            .with_min_p(self.min_p);

        let mut req = GenerateRequest::new(prompt_tokens)
            .with_sampler(sampler)
            .with_max_tokens(self.max_tokens);

        if !self.stop_tokens.is_empty() {
            req = req.with_stop_tokens(self.stop_tokens.clone());
        }
        // Otherwise GenerateRequest falls back to info.eos_tokens

        req
    }
}
```

### 4.5 ChatSession

```rust
use ironmill_core::ChatMessage;
use crate::error::TorchError;
use crate::gen_params::GenParams;
use crate::model::Model;
use crate::text_output::{TextOutput, TextStream};

/// A stateful multi-turn chat session bound to a [`Model`].
///
/// Maintains conversation history and applies the tokenizer's chat
/// template to format prompts.
///
/// # Example
///
/// ```rust,no_run
/// use ironmill_torch::{Model, GenParams};
///
/// let mut model = Model::from_pretrained("./model/").build()?;
/// let mut chat = model.chat()
///     .system("You are a helpful assistant.")
///     .build();
///
/// let reply = chat.send("What is Rust?")?;
/// println!("{}", reply.text);
/// # Ok::<(), ironmill_torch::TorchError>(())
/// ```
pub struct ChatSession<'m> {
    model: &'m mut Model,
    history: Vec<ChatMessage>,
    default_params: GenParams,
}

impl<'m> ChatSession<'m> {
    pub(crate) fn new(
        model: &'m mut Model,
        history: Vec<ChatMessage>,
        params: GenParams,
    ) -> Self {
        Self { model, history, default_params: params }
    }

    /// Send a user message and receive the assistant's reply.
    pub fn send(
        &mut self,
        message: impl Into<String>,
    ) -> Result<TextOutput, TorchError> {
        self.send_with_params(message, self.default_params.clone())
    }

    /// Send a user message with custom generation parameters.
    pub fn send_with_params(
        &mut self,
        message: impl Into<String>,
        params: GenParams,
    ) -> Result<TextOutput, TorchError> {
        let user_msg = ChatMessage::user(message);
        // Format the full conversation into a prompt
        let mut messages = self.history.clone();
        messages.push(user_msg.clone());
        let prompt = self.model.tokenizer().apply_chat_template(&messages)?;

        // Generate
        let output = self.model.generate(&prompt, &params)?;

        // Only append to history AFTER successful generation
        self.history.push(user_msg);
        self.history.push(ChatMessage::assistant(&output.text));

        Ok(output)
    }

    /// Send a message and stream the response.
    pub fn send_stream<'a>(
        &'a mut self,
        message: &str,
    ) -> Result<TextStream<'a>, TorchError> {
        let user_msg = ChatMessage::user(message);
        let mut messages = self.history.clone();
        messages.push(user_msg.clone());
        let prompt = self.model.tokenizer().apply_chat_template(&messages)?;

        // History is appended when the stream completes, via a callback
        // on TextStream's Drop or finish.
        self.history.push(user_msg);
        self.model.stream(&prompt, &self.default_params)
    }

    /// Return the conversation history.
    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// Clear the conversation history, keeping the system prompt.
    pub fn reset(&mut self) {
        let system = self.history.iter()
            .find(|m| m.role == "system")
            .cloned();
        self.history.clear();
        if let Some(sys) = system {
            self.history.push(sys);
        }
    }
}

/// Builder for [`ChatSession`].
pub struct ChatSessionBuilder<'a> {
    model: &'a mut Model,
    system_prompt: Option<String>,
    params: GenParams,
}

impl<'a> ChatSessionBuilder<'a> {
    pub(crate) fn new(model: &'a mut Model) -> Self {
        Self {
            model,
            system_prompt: None,
            params: GenParams::default(),
        }
    }

    /// Set the system prompt.
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into()); self
    }

    /// Set default generation parameters.
    pub fn params(mut self, params: GenParams) -> Self {
        self.params = params; self
    }

    /// Build the chat session.
    pub fn build(self) -> ChatSession<'a> {
        let mut history = Vec::new();
        if let Some(sys) = self.system_prompt {
            history.push(ChatMessage::system(sys));
        }
        ChatSession::new(self.model, history, self.params)
    }
}
```

### 4.6 TextOutput / TextStream

```rust
use std::time::Duration;
use ironmill_core::Tokenizer;
use ironmill_inference::engine::InferenceError;
use ironmill_inference::generate::{
    CancellationToken, GenerateEvent, TokenStream,
};
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
    /// Why generation stopped (e.g. "Stop", "MaxTokens").
    pub finish_reason: String,
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
    pub finish_reason: Option<String>,
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
        Self { inner, tokenizer, cancel }
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
            Ok(GenerateEvent::Token { token, position, .. }) => {
                let text = self.tokenizer.decode(&[token])
                    .unwrap_or_default();
                Some(Ok(TextChunk {
                    text,
                    token,
                    position,
                    finished: false,
                    finish_reason: None,
                }))
            }
            Ok(GenerateEvent::Finished { reason, .. }) => {
                Some(Ok(TextChunk {
                    text: String::new(),
                    token: 0,
                    position: 0,
                    finished: true,
                    finish_reason: Some(format!("{reason:?}")),
                }))
            }
            Ok(GenerateEvent::PromptProcessed { .. }) => {
                // Skip prefill events, advance to next
                self.next()
            }
            Err(e) => Some(Err(e.into())),
            _ => self.next(), // forward-compat
        }
    }
}
```

---

## 5. Usage Examples

### 5.1 Minimal Generation

```rust
use ironmill_torch::{Model, GenParams};

let mut model = Model::from_pretrained("./Qwen3-0.6B/").build()?;
let output = model.generate("What is Rust?", &GenParams::default())?;
println!("{}", output.text);
```

### 5.2 Streaming

```rust
use ironmill_torch::{Model, GenParams, Device};

let mut model = Model::from_pretrained("./Qwen3-0.6B/")
    .device(Device::Metal)
    .max_seq_len(8192)
    .build()?;

for chunk in model.stream("Explain monads simply", &GenParams::default())? {
    let chunk = chunk?;
    print!("{}", chunk.text);
    if chunk.finished {
        println!("\n[done: {}]", chunk.finish_reason.unwrap());
    }
}
```

### 5.3 Chat Session

```rust
use ironmill_torch::{Model, GenParams};

let mut model = Model::from_pretrained("./Llama-3.2-3B/").build()?;

let mut chat = model.chat()
    .system("You are a helpful, concise assistant.")
    .params(GenParams::default().with_temperature(0.8))
    .build();

let r1 = chat.send("What's the capital of France?")?;
println!("Assistant: {}", r1.text);

let r2 = chat.send("What about Germany?")?;
println!("Assistant: {}", r2.text);

println!("History: {} turns", chat.history().len());
```

### 5.4 Load Pre-Compiled Model

```rust
use ironmill_torch::{Model, GenParams, Device};

let mut model = Model::from_compiled("./model.ironml-gpu")
    .device(Device::Metal)
    .build()?;

let output = model.generate("Hello!", &GenParams::greedy())?;
```

### 5.5 Drop Down to Low-Level API

```rust
use ironmill_torch::Model;
use ironmill_inference::InferenceEngine;
use ironmill_inference::generate::{GenerateRequest, TokenStream, CancellationToken};
use ironmill_inference::sampling::SamplerConfig;

let mut model = Model::from_pretrained("./model/").build()?;

// Access the raw engine for low-level control
let engine = model.engine_mut();
let logits = engine.prefill(&[1, 15043, 29892])?;
// ... manual decode loop with custom sampling ...
```

### 5.6 Progress Tracking

```rust
use ironmill_torch::{Model, GenParams};
use mil_rs::ProgressSink;
use std::time::Duration;

struct CliProgress;
impl ProgressSink for CliProgress {
    fn on_stage(&self, name: &str) {
        eprintln!("⏳ {name}...");
    }
    fn on_stage_complete(&self, name: &str, elapsed: Duration) {
        eprintln!("  ✓ {name} ({elapsed:.1?})");
    }
}

let mut model = Model::from_pretrained("./model/")
    .with_progress(CliProgress)
    .build()?;
```

---

## 6. Design Decisions

### 6.1 Why a Separate Crate (Not ironmill-core)?

`ironmill-core` is the bottom of the dependency graph — shared types that
both `ironmill-compile` and `ironmill-inference` depend on. Putting `Model`
there would require `ironmill-core` to depend on `ironmill-inference`,
creating a cycle. A separate top-level crate breaks the cycle cleanly.

### 6.2 Why Not Put It in ironmill-cli?

The CLI is a binary crate. Library consumers (servers, apps, other tools)
need the high-level API too. A library crate that the CLI depends on is
the right factoring.

### 6.3 Why &mut self on generate/stream?

The underlying `InferenceEngine` requires `&mut self` for prefill/decode
(it mutates KV cache state). `Model::generate` therefore needs `&mut self`.
This prevents concurrent calls on the same model, which is correct — GPU
engines are not thread-safe. For server workloads, use `BatchRunner` from
`ironmill-inference` directly.

### 6.4 Error Wrapping Strategy

`TorchError` wraps `InferenceError` and `CompileError` via `#[from]`,
preserving the full error chain. Users who need the underlying error can
match on `TorchError::Inference(e)` and downcast `e` to `MetalError` etc.

### 6.5 GenParams vs SamplerConfig

`GenParams` is the user-facing surface — simple, obvious field names,
sensible defaults. `SamplerConfig` is the engine-facing config with
additional fields (repeat_penalty, frequency_penalty, etc.) that power
users access through `ironmill-inference` directly.

---

## 7. Open Questions

1. **Name**: `ironmill-torch` clearly communicates the abstraction level.
   Alternative: `ironmill-easy`, `ironmill-model`, `ironmill`.
2. **Chat template engine**: Should `apply_chat_template` support Jinja2
   templates (matching HuggingFace convention), or is the simple
   `<|role|>\ncontent\n` format sufficient for v1?
3. **Auto-compilation**: Should `from_pretrained` auto-compile (JIT path)
   when no pre-compiled artifact exists? Or require explicit compilation?
4. **Streaming chat**: `send_stream` appends the user message to history
   immediately, but the assistant response can only be appended when the
   stream completes. Should `TextStream` handle this via a completion
   callback, or should the caller be responsible?
