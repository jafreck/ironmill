//! ANE bundle manifest schema types.

use serde::{Deserialize, Serialize};

/// Architecture metadata for decode bundles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleArchitecture {
    pub vocab_size: usize,
    pub eos_tokens: Vec<u32>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub rope_theta: f64,
    pub max_seq_len: usize,
    pub qk_norm: bool,
}

/// Tensor descriptor for bundle manifest (serializable).
///
/// This is the bundle-format equivalent of the compile-crate's
/// [`TensorDescriptor`]. Uses a string dtype for JSON compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleTensorDescriptor {
    pub name: String,
    pub shape: [usize; 4],
    /// Scalar type as a string, e.g. `"Float16"`, `"Float32"`.
    pub dtype: String,
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
    pub format_version: u32,
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
    Simple,
    Decode,
}

/// Manifest entry for a single ANE sub-program.
#[derive(Debug, Serialize, Deserialize)]
pub struct SubProgramManifest {
    pub name: String,
    pub inputs: Vec<BundleTensorDescriptor>,
    pub outputs: Vec<BundleTensorDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_packing: Option<BundleInputPacking>,
}

/// Decode-specific section of the bundle manifest.
#[derive(Debug, Serialize, Deserialize)]
pub struct DecodeManifest {
    pub architecture: BundleArchitecture,
    pub lm_head: LmHeadManifest,
    pub layers: Vec<LayerManifest>,
}

/// Manifest entry for the LM head.
#[derive(Debug, Serialize, Deserialize)]
pub struct LmHeadManifest {
    /// `"ane"` or `"cpu"`.
    pub kind: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// ANE chunk sub-program manifests (empty for CPU fallback).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub chunks: Vec<SubProgramManifest>,
}

/// Manifest entry for a single transformer layer.
#[derive(Debug, Serialize, Deserialize)]
pub struct LayerManifest {
    pub index: usize,
    pub pre_attn: SubProgramManifest,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_attn: Option<SubProgramManifest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fp16_attn: Option<SubProgramManifest>,
    pub cache_write_fused: bool,
    pub donor_compatible: bool,
}

impl From<&super::TensorDescriptor> for BundleTensorDescriptor {
    fn from(td: &super::TensorDescriptor) -> Self {
        Self {
            name: td.name.clone(),
            shape: td.shape,
            dtype: format!("{:?}", td.dtype),
        }
    }
}
