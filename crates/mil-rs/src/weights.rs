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
#[non_exhaustive]
pub enum Architecture {
    /// Meta's LLaMA family (includes Mistral, CodeLlama).
    Llama,
    /// Alibaba's Qwen family.
    Qwen,
    /// Google's Gemma family.
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
            "gemma" | "gemma2" | "gemma3" | "gemma4" | "gemma4_text" => Ok(Architecture::Gemma),
            _ => Err(MilError::Validation(format!(
                "unsupported architecture: {s}"
            ))),
        }
    }
}

/// Architecture-agnostic model configuration extracted from
/// config.json (SafeTensors) or GGUF metadata.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ModelConfig {
    /// Model architecture family.
    pub architecture: Architecture,
    /// Dimensionality of the hidden representations.
    pub hidden_size: usize,
    /// Dimensionality of the feed-forward intermediate layer.
    pub intermediate_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value attention heads (for grouped-query attention).
    pub num_key_value_heads: usize,
    /// Dimensionality of each attention head.
    pub head_dim: usize,
    /// Size of the token vocabulary.
    pub vocab_size: usize,
    /// Maximum sequence length supported by the positional embeddings.
    pub max_position_embeddings: usize,
    /// Epsilon for RMSNorm layers.
    pub rms_norm_eps: f64,
    /// Base frequency for rotary position embeddings.
    pub rope_theta: f64,
    /// Whether the input and output embedding weights are shared.
    pub tie_word_embeddings: bool,
    /// Architecture-specific parameters (e.g. Gemma sliding_window_size,
    /// Qwen attention bias flags). Templates access via architecture match.
    pub extra: HashMap<String, serde_json::Value>,
}

impl ModelConfig {
    /// Create a new `ModelConfig` with the given architecture and sensible defaults.
    ///
    /// All numeric fields default to `0`, booleans to `false`, and `extra`
    /// to an empty map. Use the `with_*()` builder methods to set fields.
    pub fn new(architecture: Architecture) -> Self {
        Self {
            architecture,
            hidden_size: 0,
            intermediate_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            head_dim: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            extra: HashMap::new(),
        }
    }

    /// Set the hidden size.
    pub fn with_hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    /// Set the intermediate size.
    pub fn with_intermediate_size(mut self, intermediate_size: usize) -> Self {
        self.intermediate_size = intermediate_size;
        self
    }

    /// Set the number of hidden layers.
    pub fn with_num_hidden_layers(mut self, num_hidden_layers: usize) -> Self {
        self.num_hidden_layers = num_hidden_layers;
        self
    }

    /// Set the number of attention heads.
    pub fn with_num_attention_heads(mut self, num_attention_heads: usize) -> Self {
        self.num_attention_heads = num_attention_heads;
        self
    }

    /// Set the number of key-value heads.
    pub fn with_num_key_value_heads(mut self, num_key_value_heads: usize) -> Self {
        self.num_key_value_heads = num_key_value_heads;
        self
    }

    /// Set the head dimension.
    pub fn with_head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = head_dim;
        self
    }

    /// Set the vocabulary size.
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Set the maximum position embeddings.
    pub fn with_max_position_embeddings(mut self, max_position_embeddings: usize) -> Self {
        self.max_position_embeddings = max_position_embeddings;
        self
    }

    /// Set the RMS norm epsilon.
    pub fn with_rms_norm_eps(mut self, rms_norm_eps: f64) -> Self {
        self.rms_norm_eps = rms_norm_eps;
        self
    }

    /// Set the RoPE theta.
    pub fn with_rope_theta(mut self, rope_theta: f64) -> Self {
        self.rope_theta = rope_theta;
        self
    }

    /// Set whether word embeddings are tied.
    pub fn with_tie_word_embeddings(mut self, tie_word_embeddings: bool) -> Self {
        self.tie_word_embeddings = tie_word_embeddings;
        self
    }

    /// Set extra architecture-specific parameters.
    pub fn with_extra(mut self, extra: HashMap<String, serde_json::Value>) -> Self {
        self.extra = extra;
        self
    }

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

    /// Extract per-layer attention types from Gemma 4 config.
    ///
    /// Returns `None` for non-Gemma-4 models. When present, each entry
    /// is `"sliding_attention"` or `"full_attention"`.
    pub fn layer_types(&self) -> Option<Vec<String>> {
        let val = self.extra.get("layer_types")?;
        let arr = val.as_array()?;
        let types: Option<Vec<String>> = arr.iter().map(|v| v.as_str().map(String::from)).collect();
        types
    }

    /// Get per-layer-type RoPE parameters from Gemma 4 config.
    ///
    /// Returns a map from layer type name to its RoPE configuration.
    /// `partial_rotary_factor` of 1.0 means full rotation (standard RoPE).
    pub fn rope_parameters(&self) -> Option<HashMap<String, RopeLayerConfig>> {
        let val = self.extra.get("rope_parameters")?;
        let obj = val.as_object()?;
        let mut result = HashMap::new();
        for (key, params) in obj {
            let theta = params
                .get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0);
            let partial_rotary_factor = params
                .get("partial_rotary_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            result.insert(
                key.clone(),
                RopeLayerConfig {
                    theta,
                    partial_rotary_factor,
                },
            );
        }
        Some(result)
    }

    /// Get the global head dim for Gemma 4 full-attention layers.
    /// Falls back to `self.head_dim` if not specified.
    pub fn global_head_dim(&self) -> usize {
        self.extra
            .get("global_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.head_dim)
    }

    /// Get the number of KV heads for global attention layers.
    /// Falls back to `self.num_key_value_heads` if not specified.
    pub fn num_global_key_value_heads(&self) -> usize {
        self.extra
            .get("num_global_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.num_key_value_heads)
    }
}

/// Per-layer-type RoPE configuration for Gemma 4.
#[derive(Debug, Clone)]
pub struct RopeLayerConfig {
    /// Base frequency for rotary position embeddings.
    pub theta: f64,
    /// Fraction of head dimensions that receive rotation (1.0 = all).
    pub partial_rotary_factor: f64,
}

/// Multi-Head Latent Attention (MLA) configuration.
///
/// Used by DeepSeek-V2/V3 models to compress the KV cache into a
/// low-dimensional latent space. Weight absorption at load time fuses
/// the up-projection weights into Q and O projections, eliminating
/// runtime decompression.
#[derive(Debug, Clone)]
#[non_exhaustive]
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

impl MlaConfig {
    /// Create a new MLA configuration.
    pub fn new(
        kv_latent_dim: usize,
        q_latent_dim: usize,
        num_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
    ) -> Self {
        Self {
            kv_latent_dim,
            q_latent_dim,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
        }
    }
}

/// Describes how a weight tensor is stored in compressed form.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum QuantizationInfo {
    /// Dense storage, no quantization. Existing behavior.
    None,
    /// PolarQuant / palettization: LUT + packed indices + row norms.
    /// Produced by `constexpr_lut_to_dense` in MIL IR.
    LutToDense {
        /// Raw bytes of the palette lookup table.
        lut: Vec<u8>,
        /// Scalar type of the LUT entries.
        lut_dtype: ScalarType,
        /// Packed index buffer referencing LUT entries.
        indices: Vec<u8>,
        /// Original unquantized tensor shape.
        original_shape: Vec<usize>,
        /// Number of bits per index (1, 2, 4, 6, or 8).
        n_bits: u8,
        /// Per-row norm values used for dequantization.
        row_norms: Vec<u8>,
        /// Scalar type of the row norm values.
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
        /// Raw bytes of the per-channel or per-group scale factors.
        scale: Vec<u8>,
        /// Raw bytes of the per-channel or per-group zero points.
        zero_point: Vec<u8>,
        /// Scalar type of the scale values.
        scale_dtype: ScalarType,
        /// Scalar type of the zero point values.
        zero_point_dtype: ScalarType,
        /// Axis along which quantization parameters are applied.
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
    /// D2Quant dual-scale quantization: each weight group has separate
    /// scale/zero for normal and outlier partitions, selected by a bit mask.
    /// Produced by `constexpr_dual_scale_dequantize` in MIL IR.
    DualScaleDequantize {
        /// Packed quantized data (2-bit or 3-bit packed into bytes).
        quantized_data: Vec<u8>,
        /// Per-group scale for the normal (low-magnitude) partition.
        normal_scale: Vec<u8>,
        /// Per-group zero point for the normal partition.
        normal_zero: Vec<u8>,
        /// Per-group scale for the outlier (high-magnitude) partition.
        outlier_scale: Vec<u8>,
        /// Per-group zero point for the outlier partition.
        outlier_zero: Vec<u8>,
        /// Packed outlier mask (1 bit per weight).
        outlier_mask: Vec<u8>,
        /// Original unquantized tensor shape.
        original_shape: Vec<usize>,
        /// Quantization bit width (2 or 3).
        bit_width: u8,
        /// Number of weights per group (typically 128).
        group_size: usize,
    },
}

/// Borrowed view of a weight tensor. Uses `Cow` to allow zero-copy
/// mmap access from SafeTensors while still supporting owned data
/// from GGUF dequantization.
#[derive(Debug)]
#[non_exhaustive]
pub struct WeightTensor<'a> {
    /// Raw weight data, borrowed (mmap) or owned (dequantized).
    pub data: Cow<'a, [u8]>,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: ScalarType,
    /// Quantization metadata, if any.
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

    /// Set quantization info on this tensor (builder-style).
    pub fn with_quant_info(mut self, quant_info: QuantizationInfo) -> Self {
        self.quant_info = quant_info;
        self
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

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Gemma,
            hidden_size: 1536,
            intermediate_size: 6144,
            num_hidden_layers: 6,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dim: 256,
            vocab_size: 262144,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            extra: HashMap::new(),
        }
    }

    #[test]
    fn architecture_from_str_gemma4() {
        assert_eq!(
            "gemma4".parse::<Architecture>().unwrap(),
            Architecture::Gemma
        );
        assert_eq!(
            "gemma4_text".parse::<Architecture>().unwrap(),
            Architecture::Gemma
        );
    }

    #[test]
    fn layer_types_extraction() {
        let mut config = minimal_config();
        config.extra.insert(
            "layer_types".into(),
            serde_json::json!([
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention"
            ]),
        );
        let lt = config.layer_types().unwrap();
        assert_eq!(lt.len(), 6);
        assert_eq!(lt[5], "full_attention");
    }

    #[test]
    fn layer_types_none_for_non_gemma4() {
        let config = minimal_config();
        assert!(config.layer_types().is_none());
    }

    #[test]
    fn rope_parameters_extraction() {
        let mut config = minimal_config();
        config.extra.insert("rope_parameters".into(), serde_json::json!({
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}
        }));
        let rp = config.rope_parameters().unwrap();
        let sliding = &rp["sliding_attention"];
        assert_eq!(sliding.theta, 10000.0);
        assert_eq!(sliding.partial_rotary_factor, 1.0);
        let global = &rp["full_attention"];
        assert_eq!(global.theta, 1000000.0);
        assert_eq!(global.partial_rotary_factor, 0.25);
    }

    #[test]
    fn global_head_dim_fallback() {
        let config = minimal_config();
        assert_eq!(config.global_head_dim(), 256);
    }

    #[test]
    fn global_head_dim_explicit() {
        let mut config = minimal_config();
        config
            .extra
            .insert("global_head_dim".into(), serde_json::json!(512));
        assert_eq!(config.global_head_dim(), 512);
    }

    #[test]
    fn num_global_kv_heads_fallback() {
        let config = minimal_config();
        assert_eq!(config.num_global_key_value_heads(), 1);
    }

    #[test]
    fn num_global_kv_heads_explicit() {
        let mut config = minimal_config();
        config
            .extra
            .insert("num_global_key_value_heads".into(), serde_json::json!(4));
        assert_eq!(config.num_global_key_value_heads(), 4);
    }
}
