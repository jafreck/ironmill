//! Model loading and inference (§10.4).

use std::path::{Path, PathBuf};

use crate::chat::ChatSessionBuilder;
use crate::model_info::ModelInfo;
use crate::tokenizer::Tokenizer;
use crate::{Device, GenParams, ModelError, TextOutput};

/// A loaded model ready for inference.
pub struct Model {
    engine: Box<dyn std::any::Any + Send>,
    tokenizer: Box<dyn Tokenizer>,
    info: ModelInfo,
}

/// Where the model weights originate from.
enum ModelSource {
    /// A pretrained bundle on disk.
    Pretrained,
    /// A pre-compiled artifact.
    Compiled,
}

impl Model {
    /// Begin constructing a model from a pretrained bundle on disk.
    pub fn from_pretrained(path: impl AsRef<Path>) -> ModelBuilder {
        ModelBuilder::new(path.as_ref().to_path_buf(), ModelSource::Pretrained)
    }

    /// Load from a pre-compiled artifact.
    pub fn from_compiled(path: impl AsRef<Path>) -> ModelBuilder {
        ModelBuilder::new(path.as_ref().to_path_buf(), ModelSource::Compiled)
    }

    /// Run a single text-generation call (non-streaming).
    pub fn generate(&self, _prompt: &str, _params: &GenParams) -> Result<TextOutput, ModelError> {
        todo!("Model::generate requires ironmill-inference engine integration")
    }

    /// Generate text with streaming output.
    pub fn stream<'a>(
        &'a mut self,
        _prompt: &str,
        _params: &GenParams,
    ) -> Result<crate::text_output::TextStream<'a>, ModelError> {
        todo!("Model::stream requires ironmill-inference engine integration")
    }

    /// Start a chat session.
    pub fn chat(&mut self) -> ChatSessionBuilder<'_> {
        ChatSessionBuilder::new(self)
    }

    /// Access the underlying engine (type-erased).
    ///
    /// Downcast to `ironmill_inference::InferenceEngine` when that crate
    /// is available.
    pub fn engine(&self) -> &(dyn std::any::Any + Send) {
        &*self.engine
    }

    /// Access the underlying engine mutably (type-erased).
    ///
    /// Downcast to `ironmill_inference::InferenceEngine` when that crate
    /// is available.
    pub fn engine_mut(&mut self) -> &mut (dyn std::any::Any + Send) {
        &mut *self.engine
    }

    /// Access the tokenizer.
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        &*self.tokenizer
    }

    /// Model information.
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }
}

/// Builder for configuring and loading a [`Model`].
pub struct ModelBuilder {
    path: PathBuf,
    _source: ModelSource,
    device: Device,
    max_seq_len: usize,
    tokenizer_path: Option<PathBuf>,
}

impl ModelBuilder {
    fn new(path: PathBuf, source: ModelSource) -> Self {
        Self {
            path,
            _source: source,
            device: Device::Auto,
            max_seq_len: 4096,
            tokenizer_path: None,
        }
    }

    /// Select the compute device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the maximum sequence length the model will support.
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Override the tokenizer path (defaults to the one bundled with the model).
    pub fn tokenizer(mut self, path: impl AsRef<Path>) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Load and compile the model, returning a ready-to-use [`Model`].
    pub fn build(self) -> Result<Model, ModelError> {
        let _path = &self.path;
        let _seq_len = self.max_seq_len;
        let _tok = &self.tokenizer_path;
        todo!("Model::build requires ironmill-inference engine integration")
    }
}
