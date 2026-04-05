//! ANE bundle manifest schema types.

use mil_rs::ir::ScalarType;
use serde::{Deserialize, Serialize};

/// Architecture metadata for decode bundles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleArchitecture {
    /// Size of the token vocabulary.
    pub vocab_size: usize,
    /// Token IDs that signal end-of-sequence.
    pub eos_tokens: Vec<u32>,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads (for grouped-query attention).
    pub num_kv_heads: usize,
    /// Dimensionality of each attention head.
    pub head_dim: usize,
    /// Hidden dimension of the model.
    pub hidden_size: usize,
    /// Base frequency for rotary position embeddings.
    pub rope_theta: f64,
    /// Maximum supported sequence length.
    pub max_seq_len: usize,
    /// Whether QK normalization is applied.
    pub qk_norm: bool,
}

/// Tensor descriptor for bundle manifest (serializable).
///
/// This is the bundle-format equivalent of the compile-crate's
/// [`TensorDescriptor`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleTensorDescriptor {
    /// Name identifying this tensor.
    pub name: String,
    /// Four-dimensional tensor shape (N, C, H, W).
    pub shape: [usize; 4],
    /// Element data type.
    pub dtype: ScalarType,
}

/// Input packing metadata for bundle manifest (serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleInputPacking {
    /// Spatial offset for each original input within the packed tensor.
    pub offsets: Vec<usize>,
    /// Spatial size (S dimension) of each original input.
    pub sizes: Vec<usize>,
}

/// Top-level manifest for `.ironml` bundles.
#[derive(Debug, Serialize, Deserialize)]
pub struct BundleManifest {
    /// Schema version of the bundle format.
    pub format_version: u32,
    /// Whether this is a simple or decode bundle.
    pub model_type: BundleModelType,
    /// Per-sub-program metadata (for simple bundles).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub sub_programs: Vec<SubProgramManifest>,
    /// Decode-specific metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<DecodeManifest>,
}

/// Discriminator for the kind of model stored in a bundle.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleModelType {
    /// A standalone model bundle with flat sub-programs.
    Simple,
    /// A decoder-style transformer bundle with layered structure.
    Decode,
}

/// Manifest entry for a single ANE sub-program.
#[derive(Debug, Serialize, Deserialize)]
pub struct SubProgramManifest {
    /// Identifier for this sub-program.
    pub name: String,
    /// Input tensor descriptors.
    pub inputs: Vec<BundleTensorDescriptor>,
    /// Output tensor descriptors.
    pub outputs: Vec<BundleTensorDescriptor>,
    /// Optional input packing metadata for spatially packed tensors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_packing: Option<BundleInputPacking>,
}

/// Decode-specific section of the bundle manifest.
#[derive(Debug, Serialize, Deserialize)]
pub struct DecodeManifest {
    /// Architecture hyperparameters for the decoder.
    pub architecture: BundleArchitecture,
    /// Language-model head configuration.
    pub lm_head: LmHeadManifest,
    /// Per-layer transformer manifests.
    pub layers: Vec<LayerManifest>,
}

/// Whether the LM head runs on ANE or CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LmHeadKind {
    /// LM head runs on the Apple Neural Engine.
    Ane,
    /// LM head runs on the CPU.
    Cpu,
}

/// Manifest entry for the LM head.
#[derive(Debug, Serialize, Deserialize)]
pub struct LmHeadManifest {
    /// Whether the LM head executes on ANE or CPU.
    pub kind: LmHeadKind,
    /// Vocabulary size for the output projection.
    pub vocab_size: usize,
    /// Hidden dimension fed into the head.
    pub hidden_size: usize,
    /// ANE chunk sub-program manifests (empty for CPU fallback).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub chunks: Vec<SubProgramManifest>,
}

/// Manifest entry for a single transformer layer.
#[derive(Debug, Serialize, Deserialize)]
pub struct LayerManifest {
    /// Zero-based index of this transformer layer.
    pub index: usize,
    /// Sub-program executed before the attention step.
    pub pre_attn: SubProgramManifest,
    /// Optional sub-program executed after the attention step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_attn: Option<SubProgramManifest>,
    /// Optional FP16 attention sub-program for higher precision.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp16_attn: Option<SubProgramManifest>,
    /// Whether KV-cache writes are fused into the attention kernel.
    pub cache_write_fused: bool,
    /// Whether this layer is compatible with donor-based weight sharing.
    pub donor_compatible: bool,
}

impl From<&super::TensorDescriptor> for BundleTensorDescriptor {
    fn from(td: &super::TensorDescriptor) -> Self {
        Self {
            name: td.name.clone(),
            shape: td.shape,
            dtype: td.dtype,
        }
    }
}
