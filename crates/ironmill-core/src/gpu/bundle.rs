//! GPU bundle manifest schema types.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use mil_rs::ir::ScalarType;
use mil_rs::weights::{Architecture, ModelConfig};
use serde::{Deserialize, Serialize};

// ── Manifest types ──────────────────────────────────────────────────────

/// Top-level manifest stored as `manifest.json` inside the bundle.
#[derive(Debug, Serialize, Deserialize)]
pub struct GpuBundleManifest {
    pub format_version: u32,
    pub model_config: serde_json::Value,
    pub quantization: QuantizationManifest,
    pub tensors: HashMap<String, TensorManifest>,
}

/// Global quantization parameters that produced the bundle.
#[derive(Debug, Serialize, Deserialize)]
pub struct QuantizationManifest {
    pub method: String,
    pub n_bits: u8,
    pub seed: u32,
    pub min_elements: usize,
}

/// Per-tensor descriptor written into the manifest.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "format")]
pub enum TensorManifest {
    /// PolarQuant LUT-compressed tensor (indices + LUT + row norms).
    #[serde(rename = "lut_to_dense")]
    LutToDense {
        indices_file: String,
        lut_file: String,
        norms_file: String,
        shape: Vec<usize>,
        n_bits: u8,
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
        file: String,
        shape: Vec<usize>,
        dtype: String,
    },
    /// Affine-quantized tensor (packed INT4/INT8 data + scales + zero points).
    #[serde(rename = "affine_dequantize")]
    AffineDequantize {
        quantized_data_file: String,
        scales_file: String,
        zeros_file: String,
        shape: Vec<usize>,
        bit_width: u8,
        group_size: usize,
        axis: i64,
        dtype: String,
        /// Optional per-column AWQ channel scales file. When present, the
        /// dequantized weight is divided by these scales to compensate for
        /// activation-aware weight scaling applied during quantization.
        #[serde(skip_serializing_if = "Option::is_none")]
        awq_scales_file: Option<String>,
    },
}

// ── ScalarType string helpers ───────────────────────────────────────────

/// Convert a [`ScalarType`] to a stable string representation for the manifest.
pub fn scalar_type_to_str(dtype: ScalarType) -> &'static str {
    match dtype {
        ScalarType::Float16 => "float16",
        ScalarType::Float32 => "float32",
        ScalarType::Float64 => "float64",
        ScalarType::Int8 => "int8",
        ScalarType::Int16 => "int16",
        ScalarType::Int32 => "int32",
        ScalarType::Int64 => "int64",
        ScalarType::UInt8 => "uint8",
        ScalarType::UInt16 => "uint16",
        ScalarType::UInt32 => "uint32",
        ScalarType::UInt64 => "uint64",
        ScalarType::Bool => "bool",
        _ => panic!("unsupported scalar type: {dtype:?}"),
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
        _ => Err(anyhow!("unknown scalar type: {s}")),
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
        .ok_or_else(|| anyhow!("model_config must be an object"))?;

    let get_str = |key: &str| -> Result<&str> {
        obj.get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing string field: {key}"))
    };

    let get_usize = |key: &str| -> Result<usize> {
        obj.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| anyhow!("missing integer field: {key}"))
    };

    let get_f64 = |key: &str| -> Result<f64> {
        obj.get(key)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing float field: {key}"))
    };

    let architecture: Architecture = get_str("architecture")?
        .parse()
        .map_err(|e: mil_rs::MilError| anyhow!(e.to_string()))?;

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
