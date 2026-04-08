//! GPU bundle manifest schema types and I/O helpers.

use std::collections::HashMap;

use mil_rs::MilError;
use mil_rs::ir::ScalarType;
use mil_rs::weights::{Architecture, ModelConfig, QuantizationInfo, WeightTensor};
use serde::{Deserialize, Serialize};

/// Errors from GPU bundle manifest operations.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum GpuBundleError {
    /// Unsupported or unknown scalar type.
    #[error("{0}")]
    ScalarType(String),

    /// Invalid or missing manifest field.
    #[error("{0}")]
    Manifest(String),

    /// Error parsing model architecture.
    #[error("architecture parse error: {0}")]
    Architecture(#[source] mil_rs::MilError),
}

/// Result type for GPU bundle operations.
pub type Result<T> = std::result::Result<T, GpuBundleError>;

// ── Manifest types ──────────────────────────────────────────────────────

/// Top-level manifest stored as `manifest.json` inside the bundle.
#[derive(Debug, Serialize, Deserialize)]
pub struct GpuBundleManifest {
    /// Schema version of the GPU bundle format.
    pub format_version: u32,
    /// Serialized model configuration (architecture, dimensions, etc.).
    pub model_config: serde_json::Value,
    /// Global quantization parameters used to produce this bundle.
    pub quantization: QuantizationManifest,
    /// Per-tensor manifests keyed by tensor name.
    pub tensors: HashMap<String, TensorManifest>,
}

/// Global quantization parameters that produced the bundle.
#[derive(Debug, Serialize, Deserialize)]
pub struct QuantizationManifest {
    /// Quantization algorithm name (e.g., "polarquant", "affine").
    pub method: String,
    /// Bit width used for quantization.
    pub n_bits: u8,
    /// Random seed for reproducible quantization.
    pub seed: u32,
    /// Minimum tensor element count to qualify for quantization.
    pub min_elements: usize,
}

/// Per-tensor descriptor written into the manifest.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "format")]
pub enum TensorManifest {
    /// PolarQuant LUT-compressed tensor (indices + LUT + row norms).
    #[serde(rename = "lut_to_dense")]
    LutToDense {
        /// Path to the packed index file.
        indices_file: String,
        /// Path to the look-up table file.
        lut_file: String,
        /// Path to the per-row normalization file.
        norms_file: String,
        /// Original tensor shape before quantization.
        shape: Vec<usize>,
        /// Bit width of the quantized indices.
        n_bits: u8,
        /// Original element data type (e.g., "float16").
        dtype: String,
        /// Optional QuIP# Hadamard rotation seed. When present, the tensor
        /// uses E8 lattice vector quantization and requires the QuIP# dequant
        /// path instead of the PolarQuant path.
        #[serde(skip_serializing_if = "Option::is_none")]
        quip_sharp_seed: Option<u32>,
    },
    /// Dense (unquantized) tensor stored as raw bytes.
    #[serde(rename = "dense")]
    Dense {
        /// Path to the raw tensor data file.
        file: String,
        /// Tensor shape.
        shape: Vec<usize>,
        /// Element data type (e.g., "float32").
        dtype: String,
    },
    /// Affine-quantized tensor (packed INT4/INT8 data + scales + zero points).
    #[serde(rename = "affine_dequantize")]
    AffineDequantize {
        /// Path to the packed quantized weight data.
        quantized_data_file: String,
        /// Path to the per-group scale factors.
        scales_file: String,
        /// Path to the per-group zero-point values.
        zeros_file: String,
        /// Original tensor shape before quantization.
        shape: Vec<usize>,
        /// Quantization bit width (e.g., 4 or 8).
        bit_width: u8,
        /// Number of elements per quantization group.
        group_size: usize,
        /// Axis along which groups are formed.
        axis: i64,
        /// Original element data type.
        dtype: String,
        /// Optional per-column AWQ channel scales file. When present, the
        /// dequantized weight is divided by these scales to compensate for
        /// activation-aware weight scaling applied during quantization.
        #[serde(skip_serializing_if = "Option::is_none")]
        awq_scales_file: Option<String>,
        /// Optional group-index mapping file for GPTQ-style reordering.
        #[serde(skip_serializing_if = "Option::is_none")]
        g_idx_file: Option<String>,
        /// Data type of the scale parameters (e.g., "float16", "float32").
        #[serde(skip_serializing_if = "Option::is_none")]
        scale_dtype: Option<String>,
        /// Data type of the zero-point parameters (e.g., "float16", "float32").
        #[serde(skip_serializing_if = "Option::is_none")]
        zero_point_dtype: Option<String>,
    },
    /// D2Quant dual-scale quantized tensor.
    #[serde(rename = "dual_scale_dequantize")]
    DualScaleDequantize {
        /// Path to the packed quantized data file.
        quantized_data_file: String,
        /// Path to the normal-partition scale file (FP32).
        normal_scale_file: String,
        /// Path to the normal-partition zero point file (FP32).
        normal_zero_file: String,
        /// Path to the outlier-partition scale file (FP32).
        outlier_scale_file: String,
        /// Path to the outlier-partition zero point file (FP32).
        outlier_zero_file: String,
        /// Path to the packed outlier mask file.
        outlier_mask_file: String,
        /// Original tensor shape.
        shape: Vec<usize>,
        /// Quantization bit width (2 or 3).
        bit_width: u8,
        /// Number of weights per group.
        group_size: usize,
    },
}

// ── ScalarType string helpers ───────────────────────────────────────────

/// Convert a [`ScalarType`] to a stable string representation for the manifest.
pub fn scalar_type_to_str(dtype: ScalarType) -> Result<&'static str> {
    match dtype {
        ScalarType::Float16 => Ok("float16"),
        ScalarType::Float32 => Ok("float32"),
        ScalarType::Float64 => Ok("float64"),
        ScalarType::Int8 => Ok("int8"),
        ScalarType::Int16 => Ok("int16"),
        ScalarType::Int32 => Ok("int32"),
        ScalarType::Int64 => Ok("int64"),
        ScalarType::UInt8 => Ok("uint8"),
        ScalarType::UInt16 => Ok("uint16"),
        ScalarType::UInt32 => Ok("uint32"),
        ScalarType::UInt64 => Ok("uint64"),
        ScalarType::Bool => Ok("bool"),
        _ => Err(GpuBundleError::ScalarType(format!(
            "unsupported scalar type: {dtype:?}"
        ))),
    }
}

/// Parse a string representation back into a [`ScalarType`].
pub fn str_to_scalar_type(s: &str) -> Result<ScalarType> {
    match s {
        "float16" => Ok(ScalarType::Float16),
        "float32" => Ok(ScalarType::Float32),
        "float64" => Ok(ScalarType::Float64),
        "int8" => Ok(ScalarType::Int8),
        "int16" => Ok(ScalarType::Int16),
        "int32" => Ok(ScalarType::Int32),
        "int64" => Ok(ScalarType::Int64),
        "uint8" => Ok(ScalarType::UInt8),
        "uint16" => Ok(ScalarType::UInt16),
        "uint32" => Ok(ScalarType::UInt32),
        "uint64" => Ok(ScalarType::UInt64),
        "bool" => Ok(ScalarType::Bool),
        _ => Err(GpuBundleError::ScalarType(format!(
            "unknown scalar type: {s}"
        ))),
    }
}

// ── ModelConfig serialization ───────────────────────────────────────────

/// Serialize a [`ModelConfig`] to a JSON value for the manifest.
pub fn serialize_model_config(config: &ModelConfig) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "architecture".into(),
        serde_json::Value::String(config.architecture.to_string()),
    );
    map.insert("hidden_size".into(), serde_json::json!(config.hidden_size));
    map.insert(
        "intermediate_size".into(),
        serde_json::json!(config.intermediate_size),
    );
    map.insert(
        "num_hidden_layers".into(),
        serde_json::json!(config.num_hidden_layers),
    );
    map.insert(
        "num_attention_heads".into(),
        serde_json::json!(config.num_attention_heads),
    );
    map.insert(
        "num_key_value_heads".into(),
        serde_json::json!(config.num_key_value_heads),
    );
    map.insert("head_dim".into(), serde_json::json!(config.head_dim));
    map.insert("vocab_size".into(), serde_json::json!(config.vocab_size));
    map.insert(
        "max_position_embeddings".into(),
        serde_json::json!(config.max_position_embeddings),
    );
    map.insert(
        "rms_norm_eps".into(),
        serde_json::json!(config.rms_norm_eps),
    );
    map.insert("rope_theta".into(), serde_json::json!(config.rope_theta));
    map.insert(
        "tie_word_embeddings".into(),
        serde_json::json!(config.tie_word_embeddings),
    );
    if !config.extra.is_empty() {
        map.insert("extra".into(), serde_json::json!(config.extra));
    }
    serde_json::Value::Object(map)
}

/// Deserialize a [`ModelConfig`] from a JSON value read from the manifest.
pub fn deserialize_model_config(value: &serde_json::Value) -> Result<ModelConfig> {
    let obj = value
        .as_object()
        .ok_or_else(|| GpuBundleError::Manifest("model_config must be an object".into()))?;

    let get_str = |key: &str| -> Result<&str> {
        obj.get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| GpuBundleError::Manifest(format!("missing string field: {key}")))
    };

    let get_usize = |key: &str| -> Result<usize> {
        obj.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| GpuBundleError::Manifest(format!("missing integer field: {key}")))
    };

    let get_f64 = |key: &str| -> Result<f64> {
        obj.get(key)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| GpuBundleError::Manifest(format!("missing float field: {key}")))
    };

    let architecture: Architecture = get_str("architecture")?
        .parse()
        .map_err(GpuBundleError::Architecture)?;

    let extra = obj
        .get("extra")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();

    Ok(ModelConfig::new(architecture)
        .with_hidden_size(get_usize("hidden_size")?)
        .with_intermediate_size(get_usize("intermediate_size")?)
        .with_num_hidden_layers(get_usize("num_hidden_layers")?)
        .with_num_attention_heads(get_usize("num_attention_heads")?)
        .with_num_key_value_heads(get_usize("num_key_value_heads")?)
        .with_head_dim(get_usize("head_dim")?)
        .with_vocab_size(get_usize("vocab_size")?)
        .with_max_position_embeddings(get_usize("max_position_embeddings")?)
        .with_rms_norm_eps(get_f64("rms_norm_eps")?)
        .with_rope_theta(get_f64("rope_theta")?)
        .with_tie_word_embeddings(
            obj.get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        )
        .with_extra(extra))
}

// ── Tensor I/O helpers ──────────────────────────────────────────────────

/// Reconstruct a [`WeightTensor`] from a [`TensorManifest`] entry.
///
/// `read_file` is called for each file referenced by the manifest entry.
/// It should return the raw bytes of that file. Errors from the reader are
/// propagated as [`MilError::Validation`].
///
/// This function centralises the `TensorManifest → QuantizationInfo` dispatch
/// so that bundle readers don't need to duplicate variant-matching logic.
pub fn read_tensor_from_manifest<F>(
    desc: &TensorManifest,
    mut read_file: F,
) -> std::result::Result<WeightTensor<'static>, MilError>
where
    F: FnMut(&str) -> std::result::Result<Vec<u8>, MilError>,
{
    match desc {
        TensorManifest::LutToDense {
            indices_file,
            lut_file,
            norms_file,
            shape,
            n_bits,
            dtype,
            quip_sharp_seed,
        } => {
            let indices = read_file(indices_file)?;
            let lut = read_file(lut_file)?;
            let row_norms = read_file(norms_file)?;
            let dtype =
                str_to_scalar_type(dtype).map_err(|e| MilError::Validation(e.to_string()))?;

            Ok(
                WeightTensor::owned(Vec::new(), shape.clone(), dtype).with_quant_info(
                    QuantizationInfo::LutToDense {
                        lut,
                        lut_dtype: dtype,
                        indices,
                        original_shape: shape.clone(),
                        n_bits: *n_bits,
                        row_norms,
                        norms_dtype: dtype,
                        polar_quant_seed: None, // Bundle tensors are pre-unrotated
                        quip_sharp_seed: quip_sharp_seed.map(|s| s as u64),
                    },
                ),
            )
        }
        TensorManifest::Dense { file, shape, dtype } => {
            let data = read_file(file)?;
            let dtype =
                str_to_scalar_type(dtype).map_err(|e| MilError::Validation(e.to_string()))?;

            Ok(WeightTensor::owned(data, shape.clone(), dtype))
        }
        TensorManifest::AffineDequantize {
            quantized_data_file,
            scales_file,
            zeros_file,
            shape,
            bit_width,
            group_size,
            axis,
            dtype,
            awq_scales_file,
            scale_dtype,
            zero_point_dtype,
            ..
        } => {
            let quantized_data = read_file(quantized_data_file)?;
            let scale = read_file(scales_file)?;
            let zero_point = read_file(zeros_file)?;
            let awq_scales = match awq_scales_file {
                Some(f) => Some(read_file(f)?),
                None => None,
            };
            let dtype =
                str_to_scalar_type(dtype).map_err(|e| MilError::Validation(e.to_string()))?;

            let s_dtype = scale_dtype
                .as_ref()
                .map(|s| {
                    str_to_scalar_type(s)
                        .map_err(|e| MilError::Validation(format!("invalid scale_dtype: {e}")))
                })
                .transpose()?
                .unwrap_or_else(|| {
                    tracing::warn!("scale_dtype not specified in bundle, defaulting to Float16");
                    ScalarType::Float16
                });
            let zp_dtype = zero_point_dtype
                .as_ref()
                .map(|s| {
                    str_to_scalar_type(s)
                        .map_err(|e| MilError::Validation(format!("invalid zero_point_dtype: {e}")))
                })
                .transpose()?
                .unwrap_or_else(|| {
                    tracing::warn!(
                        "zero_point_dtype not specified in bundle, defaulting to Float16"
                    );
                    ScalarType::Float16
                });

            Ok(
                WeightTensor::owned(quantized_data, shape.clone(), dtype).with_quant_info(
                    QuantizationInfo::AffineDequantize {
                        scale,
                        zero_point,
                        scale_dtype: s_dtype,
                        zero_point_dtype: zp_dtype,
                        axis: Some(*axis as usize),
                        bit_width: *bit_width,
                        group_size: Some(*group_size),
                        awq_scales,
                        g_idx: None,
                    },
                ),
            )
        }
        TensorManifest::DualScaleDequantize {
            quantized_data_file,
            normal_scale_file,
            normal_zero_file,
            outlier_scale_file,
            outlier_zero_file,
            outlier_mask_file,
            shape,
            bit_width,
            group_size,
        } => {
            let quantized_data = read_file(quantized_data_file)?;
            let normal_scale = read_file(normal_scale_file)?;
            let normal_zero = read_file(normal_zero_file)?;
            let outlier_scale = read_file(outlier_scale_file)?;
            let outlier_zero = read_file(outlier_zero_file)?;
            let outlier_mask = read_file(outlier_mask_file)?;

            Ok(
                WeightTensor::owned(Vec::new(), shape.clone(), ScalarType::UInt8).with_quant_info(
                    QuantizationInfo::DualScaleDequantize {
                        quantized_data,
                        normal_scale,
                        normal_zero,
                        outlier_scale,
                        outlier_zero,
                        outlier_mask,
                        original_shape: shape.clone(),
                        bit_width: *bit_width,
                        group_size: *group_size,
                    },
                ),
            )
        }
    }
}
