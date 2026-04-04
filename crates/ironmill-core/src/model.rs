//! Model loading and inference (§10.4).

use std::path::{Path, PathBuf};

use crate::{Device, GenParams, ModelError, TextOutput};

/// A loaded model ready for inference.
pub struct Model {
    _device: Device,
}

impl Model {
    /// Begin constructing a model from a pretrained bundle on disk.
    pub fn from_pretrained(path: impl AsRef<Path>) -> ModelBuilder {
        ModelBuilder::new(path.as_ref().to_path_buf())
    }

    /// Run a single text-generation call (non-streaming).
    pub fn generate(&self, _prompt: &str, _params: &GenParams) -> Result<TextOutput, ModelError> {
        todo!("Model::generate requires ironmill-inference engine integration")
    }
}

/// Builder for configuring and loading a [`Model`].
pub struct ModelBuilder {
    path: PathBuf,
    device: Device,
    max_seq_len: usize,
    tokenizer_path: Option<PathBuf>,
}

impl ModelBuilder {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
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
