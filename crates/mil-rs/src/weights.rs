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

    /// Extract Cross-Layer Attention anchor layer indices from metadata.
    ///
    /// Looks for `"cla_anchor_layers"` in `extra` (a JSON array of layer indices).
    /// Returns `None` if the key is absent or not a valid array of unsigned integers.
    pub fn cla_anchor_layers(&self) -> Option<Vec<usize>> {
        let val = self.extra.get("cla_anchor_layers")?;
        let arr = val.as_array()?;
        let layers: Option<Vec<usize>> =
            arr.iter().map(|v| v.as_u64().map(|n| n as usize)).collect();
        let layers = layers?;
        if layers.is_empty() {
            return None;
        }
        Some(layers)
    }

    /// Extract Multi-Head Latent Attention (MLA) configuration from metadata.
    ///
    /// Looks for DeepSeek-V2/V3 MLA-specific fields (`kv_lora_rank`,
    /// `q_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`)
    /// in `extra`. Returns `None` if any required field is absent.
    pub fn mla_config(&self) -> Option<MlaConfig> {
        let kv_latent_dim = self.extra.get("kv_lora_rank")?.as_u64()? as usize;
        let q_latent_dim = self.extra.get("q_lora_rank")?.as_u64()? as usize;
        let qk_nope_head_dim = self.extra.get("qk_nope_head_dim")?.as_u64()? as usize;
        let qk_rope_head_dim = self.extra.get("qk_rope_head_dim")?.as_u64()? as usize;
        let v_head_dim = self.extra.get("v_head_dim")?.as_u64()? as usize;
        Some(MlaConfig {
            kv_latent_dim,
            q_latent_dim,
            num_heads: self.num_attention_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
        })
    }

    /// Extract sliding window size from HuggingFace config metadata.
    ///
    /// Looks for `"sliding_window"` in `extra`. Returns `None` if absent or null.
    pub fn sliding_window(&self) -> Option<usize> {
        let val = self.extra.get("sliding_window")?;
        val.as_u64().map(|n| n as usize)
    }

    /// Extract max_window_layers from HuggingFace config metadata.
    ///
    /// Layers `0..max_window_layers` use sliding window attention.
    /// Returns `None` if absent. Defaults should be handled by the caller
    /// (e.g. fall back to `num_hidden_layers` if absent but `sliding_window` is set).
    pub fn max_window_layers(&self) -> Option<usize> {
        let val = self.extra.get("max_window_layers")?;
        val.as_u64().map(|n| n as usize)
    }
}

/// Multi-Head Latent Attention (MLA) configuration.
///
/// Used by DeepSeek-V2/V3 models to compress the KV cache into a
/// low-dimensional latent space. Weight absorption at load time fuses
/// the up-projection weights into Q and O projections, eliminating
/// runtime decompression.
#[derive(Debug, Clone)]
pub struct MlaConfig {
    /// Compressed KV latent dimension (e.g., 512). Corresponds to
    /// `kv_lora_rank` in the HuggingFace config.
    pub kv_latent_dim: usize,
    /// Query latent dimension. Corresponds to `q_lora_rank`.
    pub q_latent_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Non-RoPE portion of the Q/K head dimension.
    pub qk_nope_head_dim: usize,
    /// RoPE-applied portion of the Q/K head dimension.
    pub qk_rope_head_dim: usize,
    /// Per-head value dimension.
    pub v_head_dim: usize,
}

/// Describes how a weight tensor is stored in compressed form.
#[derive(Debug, Clone)]
pub enum QuantizationInfo {
    /// Dense storage, no quantization. Existing behavior.
    None,
    /// PolarQuant / palettization: LUT + packed indices + row norms.
    /// Produced by `constexpr_lut_to_dense` in MIL IR.
    LutToDense {
        lut: Vec<u8>,
        lut_dtype: ScalarType,
        indices: Vec<u8>,
        original_shape: Vec<usize>,
        n_bits: u8,
        row_norms: Vec<u8>,
        norms_dtype: ScalarType,
        /// Hadamard rotation seed. When present, inverse rotation must be
        /// applied after LUT reconstruction to recover original weights.
        polar_quant_seed: Option<u64>,
        /// QuIP# Hadamard rotation seed. When present, the tensor uses E8
        /// lattice vector quantization and requires the E8 codebook +
        /// inverse Hadamard for dequantization.
        quip_sharp_seed: Option<u64>,
    },
    /// Affine quantization: (quantized - zero_point) * scale.
    /// Produced by `constexpr_affine_dequantize` in MIL IR.
    /// Supports INT4 and INT8 bit widths, and optional per-group granularity.
    AffineDequantize {
        scale: Vec<u8>,
        zero_point: Vec<u8>,
        scale_dtype: ScalarType,
        zero_point_dtype: ScalarType,
        axis: Option<usize>,
        /// Quantization bit width (4 or 8). Defaults to 8 for legacy models.
        bit_width: u8,
        /// Per-group size along the quantization axis. `None` means
        /// per-tensor or per-channel granularity.
        group_size: Option<usize>,
        /// Optional AWQ per-column channel scales. When present, the
        /// dequantized weight is divided by these scales to compensate
        /// for activation-aware weight scaling.
        awq_scales: Option<Vec<u8>>,
        /// Optional GPTQ column permutation index (act-order / desc_act).
        /// When present, weight columns were quantized in a permuted order
        /// and must be un-permuted during dequantization. Each entry maps
        /// a column index to its original position.
        g_idx: Option<Vec<u32>>,
    },
}

/// Borrowed view of a weight tensor. Uses `Cow` to allow zero-copy
/// mmap access from SafeTensors while still supporting owned data
/// from GGUF dequantization.
#[derive(Debug)]
pub struct WeightTensor<'a> {
    pub data: Cow<'a, [u8]>,
    pub shape: Vec<usize>,
    pub dtype: ScalarType,
    pub quant_info: QuantizationInfo,
}

impl<'a> WeightTensor<'a> {
    /// Create a new borrowed weight tensor.
    pub fn borrowed(data: &'a [u8], shape: Vec<usize>, dtype: ScalarType) -> Self {
        Self {
            data: Cow::Borrowed(data),
            shape,
            dtype,
            quant_info: QuantizationInfo::None,
        }
    }

    /// Create a new owned weight tensor.
    pub fn owned(data: Vec<u8>, shape: Vec<usize>, dtype: ScalarType) -> Self {
        Self {
            data: Cow::Owned(data),
            shape,
            dtype,
            quant_info: QuantizationInfo::None,
        }
    }

    /// Total number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Trait abstracting over weight storage formats.
/// Templates call this to get named weight tensors.
pub trait WeightProvider: Send + Sync {
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
