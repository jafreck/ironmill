//! Model loading and inference.

use std::path::{Path, PathBuf};

use ironmill_core::device::Device;
use ironmill_core::model_info::ModelInfo;
use ironmill_core::tokenizer::Tokenizer;
use ironmill_inference::engine::InferenceEngine;
use ironmill_inference::generate::{CancellationToken, GenerateEvent, TokenStream};
use mil_rs::{NullProgress, ProgressSink};

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
    // Stored here so `stream()` can return a `TextStream` that borrows it.
    cancel: CancellationToken,
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
    pub fn generate(&mut self, prompt: &str, params: &GenParams) -> Result<TextOutput, TorchError> {
        let prompt_tokens = self.tokenizer.encode(prompt)?;
        let prompt_token_count = prompt_tokens.len();
        let request = params.to_generate_request(prompt_tokens, &self.info);
        let cancel = CancellationToken::new();
        let stream = TokenStream::new(&mut *self.engine, request, &cancel);

        let mut tokens = Vec::new();
        let mut finish_reason = String::from("max_tokens");
        let start = std::time::Instant::now();

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
        // Reset the cancel token for this new stream.
        self.cancel = CancellationToken::new();
        let cancel_handle = self.cancel.clone();
        // Split-borrow self so we can pass engine, tokenizer, and cancel
        // to TokenStream / TorchTextStream without conflicting borrows.
        let Model {
            ref mut engine,
            ref tokenizer,
            ref cancel,
            ..
        } = *self;
        let stream = TokenStream::new(&mut **engine, request, cancel);
        Ok(TorchTextStream::new(stream, &**tokenizer, cancel_handle))
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

/// Where the model weights originate from.
enum ModelSource {
    /// A pretrained weight directory (SafeTensors).
    Pretrained,
    /// A pre-compiled artifact bundle.
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
            progress: Box::new(NullProgress),
        }
    }

    /// Select the compute device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the maximum sequence length.
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Override the tokenizer path.
    pub fn tokenizer(mut self, path: impl AsRef<Path>) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Provide a progress sink for loading feedback.
    pub fn with_progress(mut self, sink: impl ProgressSink + 'static) -> Self {
        self.progress = Box::new(sink);
        self
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

        Ok(Model {
            engine,
            tokenizer,
            info,
            cancel: CancellationToken::new(),
        })
    }

    fn resolve_device(&self) -> Device {
        match self.device {
            Device::Auto => {
                #[cfg(target_os = "macos")]
                {
                    Device::Metal
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Device::Cpu
                }
            }
            other => other,
        }
    }

    fn load_tokenizer(&self) -> Result<Box<dyn Tokenizer>, TorchError> {
        let tok_dir = self.tokenizer_path.as_deref().unwrap_or(&self.path);
        #[cfg(feature = "hf-tokenizer")]
        {
            use ironmill_core::tokenizer::HfTokenizer;
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
        match device {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal => {
                use ironmill_compile::weights::SafeTensorsProvider;
                use ironmill_inference::metal::{MetalConfig, MetalInference};

                let provider = SafeTensorsProvider::load(&self.path)?;
                let config = MetalConfig::new().with_max_seq_len(self.max_seq_len);
                let info = ModelInfo::from_config(provider.config());

                let mut engine =
                    MetalInference::new(config.clone()).map_err(InferenceError::from)?;
                engine.load_weights(&provider, config)?;

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
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal => {
                use ironmill_inference::metal::{MetalBundleProvider, MetalConfig, MetalInference};

                let bundle = MetalBundleProvider::open(&self.path).map_err(InferenceError::from)?;
                let config = MetalConfig::new().with_max_seq_len(self.max_seq_len);
                let info = ModelInfo::from_config(bundle.config());

                let mut engine =
                    MetalInference::new(config.clone()).map_err(InferenceError::from)?;
                engine.load_weights(&bundle, config)?;

                Ok((Box::new(engine), info))
            }
            _ => Err(TorchError::UnsupportedDevice(device)),
        }
    }
}

#[allow(unused_imports)]
use ironmill_inference::engine::InferenceError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_pretrained_returns_builder() {
        let builder = Model::from_pretrained("/tmp/test-model");
        assert_eq!(builder.max_seq_len, 4096);
    }

    #[test]
    fn from_compiled_returns_builder() {
        let builder = Model::from_compiled("/tmp/test.ironml-gpu");
        assert_eq!(builder.max_seq_len, 4096);
    }

    #[test]
    fn builder_chain() {
        let builder = Model::from_pretrained("/tmp/test-model")
            .device(Device::Metal)
            .max_seq_len(8192)
            .tokenizer("/tmp/tokenizer");
        assert_eq!(builder.max_seq_len, 8192);
        assert_eq!(builder.device, Device::Metal);
        assert!(builder.tokenizer_path.is_some());
    }

    #[test]
    fn resolve_device_auto() {
        let builder = Model::from_pretrained("/tmp/test");
        let resolved = builder.resolve_device();
        #[cfg(target_os = "macos")]
        assert_eq!(resolved, Device::Metal);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(resolved, Device::Cpu);
    }
}
