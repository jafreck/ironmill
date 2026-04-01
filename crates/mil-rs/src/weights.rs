//! Weight provider abstraction layer.
//!
//! Provides a format-agnostic interface for loading model weights.
//! Format-specific providers (SafeTensors, GGUF) live in `ironmill-compile`.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::MilError;
use crate::ir::ScalarType;

/// Supported model architectures for template-based conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Llama,
    Qwen,
    Gemma,
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Architecture::Llama => write!(f, "llama"),
            Architecture::Qwen => write!(f, "qwen"),
            Architecture::Gemma => write!(f, "gemma"),
        }
    }
}

impl std::str::FromStr for Architecture {
    type Err = MilError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "llama" | "llama2" | "llama3" | "codellama" | "mistral" => Ok(Architecture::Llama),
            "qwen" | "qwen2" | "qwen3" => Ok(Architecture::Qwen),
            "gemma" | "gemma2" | "gemma3" => Ok(Architecture::Gemma),
            _ => Err(MilError::Validation(format!(
                "unsupported architecture: {s}"
            ))),
        }
    }
}

/// Architecture-agnostic model configuration extracted from
/// config.json (SafeTensors) or GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: Architecture,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    /// Architecture-specific parameters (e.g. Gemma sliding_window_size,
    /// Qwen attention bias flags). Templates access via architecture match.
    pub extra: HashMap<String, serde_json::Value>,
}

impl ModelConfig {
    /// Compute head_dim from hidden_size and num_attention_heads if not
    /// explicitly provided. This is the standard default for most architectures.
    pub fn default_head_dim(hidden_size: usize, num_attention_heads: usize) -> usize {
        hidden_size / num_attention_heads
    }
}

/// Borrowed view of a weight tensor. Uses `Cow` to allow zero-copy
/// mmap access from SafeTensors while still supporting owned data
/// from GGUF dequantization.
#[derive(Debug)]
pub struct WeightTensor<'a> {
    pub data: Cow<'a, [u8]>,
    pub shape: Vec<usize>,
    pub dtype: ScalarType,
}

impl<'a> WeightTensor<'a> {
    /// Create a new borrowed weight tensor.
    pub fn borrowed(data: &'a [u8], shape: Vec<usize>, dtype: ScalarType) -> Self {
        Self {
            data: Cow::Borrowed(data),
            shape,
            dtype,
        }
    }

    /// Create a new owned weight tensor.
    pub fn owned(data: Vec<u8>, shape: Vec<usize>, dtype: ScalarType) -> Self {
        Self {
            data: Cow::Owned(data),
            shape,
            dtype,
        }
    }

    /// Total number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Expected byte size based on shape and dtype.
    pub fn expected_byte_size(&self) -> usize {
        self.num_elements() * self.dtype.byte_size()
    }
}

/// Trait abstracting over weight storage formats.
/// Templates call this to get named weight tensors.
pub trait WeightProvider {
    /// Get a tensor by its canonical name (e.g. "model.layers.0.self_attn.q_proj.weight").
    /// Returns a borrowed view to avoid copying multi-GB tensors.
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError>;

    /// List all available tensor names.
    fn tensor_names(&self) -> Vec<&str>;

    /// Get model configuration.
    fn config(&self) -> &ModelConfig;

    /// Check whether a tensor with the given name exists.
    fn has_tensor(&self, name: &str) -> bool {
        self.tensor_names().contains(&name)
    }
}
