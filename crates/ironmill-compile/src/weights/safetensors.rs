//! SafeTensors weight provider.
//!
//! Loads HuggingFace model directories containing `config.json` and one or
//! more `.safetensors` shard files. Tensor data is memory-mapped for
//! zero-copy access via `Cow::Borrowed`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::convert::lora::{self, LoraAdapter};
use mil_rs::MilError;
use mil_rs::error::Result;
use mil_rs::ir::ScalarType;

use super::{Architecture, ModelConfig, WeightProvider, WeightTensor};

/// Merged LoRA tensor data: (bytes, shape, dtype).
type MergedTensor = (Vec<u8>, Vec<usize>, ScalarType);

/// Location of a tensor within a set of sharded safetensors files.
///
/// Pre-parsed metadata is stored during `load()` so that `tensor()` can
/// slice the mmap directly without re-deserializing the SafeTensors header.
#[derive(Debug, Clone)]
struct TensorLocation {
    /// Index into `SafeTensorsProvider::mmaps`.
    shard_index: usize,
    /// Byte offset of the tensor data within the mmap.
    data_start: usize,
    /// Byte length of the tensor data.
    data_len: usize,
    /// Tensor shape.
    shape: Vec<usize>,
    /// MIL scalar type (BF16 is mapped to Float16).
    dtype: ScalarType,
    /// True when the underlying data is BF16 and must be converted to FP16.
    needs_bf16_conversion: bool,
}

/// Weight provider backed by memory-mapped SafeTensors files.
///
/// Tensors are borrowed directly from the mmap, avoiding multi-GB copies.
pub struct SafeTensorsProvider {
    config: ModelConfig,
    /// Memory-mapped safetensors file(s).
    mmaps: Vec<Mmap>,
    /// Maps tensor names to shard indices.
    tensor_index: HashMap<String, TensorLocation>,
    /// Merged LoRA weights stored as owned tensors when LoRA adapters are
    /// detected. Keyed by base tensor name.
    lora_merged: HashMap<String, MergedTensor>,
}

impl std::fmt::Debug for SafeTensorsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeTensorsProvider")
            .field("config", &self.config)
            .field("num_shards", &self.mmaps.len())
            .field("num_tensors", &self.tensor_index.len())
            .field("num_lora_merged", &self.lora_merged.len())
            .finish()
    }
}

impl SafeTensorsProvider {
    /// Load from a HuggingFace model directory containing:
    /// - `config.json`
    /// - `model.safetensors` (or sharded `model-00001-of-NNNNN.safetensors`, etc.)
    /// - optionally `adapter_config.json` for LoRA adapters
    #[allow(unsafe_code)]
    pub fn load(model_dir: &Path) -> Result<Self> {
        // --- Parse config.json ---
        let config_path = model_dir.join("config.json");
        let config = if config_path.exists() {
            parse_hf_config(&config_path)?
        } else {
            return Err(MilError::Validation(format!(
                "config.json not found in {}",
                model_dir.display()
            )));
        };

        // --- Discover shard files ---
        let shard_paths = discover_shard_files(model_dir)?;
        if shard_paths.is_empty() {
            return Err(MilError::Validation(format!(
                "no .safetensors files found in {}",
                model_dir.display()
            )));
        }

        // --- Memory-map each shard ---
        let mut mmaps = Vec::with_capacity(shard_paths.len());
        for path in &shard_paths {
            let file = fs::File::open(path)?;
            // SAFETY: The file is read-only and we hold the Mmap for the
            // lifetime of the provider. Concurrent modification by another
            // process would be a user error.
            let mmap = unsafe { Mmap::map(&file)? };
            mmaps.push(mmap);
        }

        // --- Build tensor index (pre-parse metadata to avoid re-deserializing) ---
        let mut tensor_index = HashMap::new();
        for (shard_idx, mmap) in mmaps.iter().enumerate() {
            let st = SafeTensors::deserialize(mmap).map_err(|e| {
                MilError::Validation(format!(
                    "failed to parse safetensors shard {}: {e}",
                    shard_paths[shard_idx].display()
                ))
            })?;
            let mmap_ptr = mmap.as_ptr() as usize;
            for (name, view) in st.tensors() {
                let data = view.data();
                let data_start = data.as_ptr() as usize - mmap_ptr;
                let data_len = data.len();
                let st_dtype = view.dtype();
                let needs_bf16 = st_dtype == safetensors::Dtype::BF16;
                let dtype = safetensors_dtype_to_scalar(st_dtype)?;
                tensor_index.insert(
                    name,
                    TensorLocation {
                        shard_index: shard_idx,
                        data_start,
                        data_len,
                        shape: view.shape().to_vec(),
                        dtype,
                        needs_bf16_conversion: needs_bf16,
                    },
                );
            }
        }

        // --- Detect and merge LoRA adapters ---
        let lora_merged = detect_and_merge_lora(&mmaps, &tensor_index, model_dir)?;

        Ok(Self {
            config,
            mmaps,
            tensor_index,
            lora_merged,
        })
    }
}

impl WeightProvider for SafeTensorsProvider {
    fn tensor(&self, name: &str) -> std::result::Result<WeightTensor<'_>, MilError> {
        // Check LoRA-merged tensors first.
        if let Some((data, shape, dtype)) = self.lora_merged.get(name) {
            return Ok(WeightTensor::borrowed(data, shape.clone(), *dtype));
        }

        let loc = self
            .tensor_index
            .get(name)
            .ok_or_else(|| MilError::Validation(format!("tensor not found: {name}")))?;

        let mmap = &self.mmaps[loc.shard_index];
        let data = &mmap[loc.data_start..loc.data_start + loc.data_len];

        if loc.needs_bf16_conversion {
            let converted = convert_bf16_to_f16(data);
            Ok(WeightTensor::owned(converted, loc.shape.clone(), loc.dtype))
        } else {
            Ok(WeightTensor::borrowed(data, loc.shape.clone(), loc.dtype))
        }
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.tensor_index.keys().map(|s| s.as_str()).collect()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

/// Parse a HuggingFace `config.json` into a [`ModelConfig`].
pub fn parse_hf_config(config_path: &Path) -> Result<ModelConfig> {
    let text = fs::read_to_string(config_path)?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| MilError::Validation(format!("invalid config.json: {e}")))?;

    let model_type = json
        .get("model_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| MilError::Validation("config.json missing 'model_type'".into()))?;

    // Gemma 4 multimodal wrapper — drill into text_config for the text decoder.
    let (model_type, json_root) = if model_type == "gemma4" {
        let text_config = json.get("text_config").ok_or_else(|| {
            MilError::Validation("gemma4 config.json missing 'text_config'".into())
        })?;
        let inner_type = text_config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("gemma4_text");
        (inner_type.to_string(), text_config.clone())
    } else {
        (model_type.to_string(), json.clone())
    };

    let architecture = Architecture::from_str(&model_type)?;

    let hidden_size = json_usize(&json_root, "hidden_size").ok_or_else(|| {
        MilError::Validation("missing required field 'hidden_size' in config.json".into())
    })?;
    let intermediate_size = json_usize(&json_root, "intermediate_size").ok_or_else(|| {
        MilError::Validation("missing required field 'intermediate_size' in config.json".into())
    })?;
    let num_hidden_layers = json_usize(&json_root, "num_hidden_layers").ok_or_else(|| {
        MilError::Validation("missing required field 'num_hidden_layers' in config.json".into())
    })?;
    let num_attention_heads = json_usize(&json_root, "num_attention_heads").ok_or_else(|| {
        MilError::Validation("missing required field 'num_attention_heads' in config.json".into())
    })?;
    let num_key_value_heads =
        json_usize(&json_root, "num_key_value_heads").unwrap_or(num_attention_heads);
    let head_dim = json_usize(&json_root, "head_dim")
        .unwrap_or_else(|| ModelConfig::default_head_dim(hidden_size, num_attention_heads));
    let vocab_size = json_usize(&json_root, "vocab_size").ok_or_else(|| {
        MilError::Validation("missing required field 'vocab_size' in config.json".into())
    })?;
    let max_position_embeddings = json_usize(&json_root, "max_position_embeddings").unwrap_or(0);
    let rms_norm_eps = json_f64(&json_root, "rms_norm_eps").unwrap_or(1e-6);
    let rope_theta = json_f64(&json_root, "rope_theta").unwrap_or(10000.0);
    let tie_word_embeddings = json_root
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Collect architecture-specific extra fields.
    let mut extra = HashMap::new();
    let known_keys: &[&str] = &[
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
        "tie_word_embeddings",
    ];
    if let Some(obj) = json_root.as_object() {
        for (k, v) in obj {
            if !known_keys.contains(&k.as_str()) {
                extra.insert(k.clone(), v.clone());
            }
        }
    }

    Ok(ModelConfig::new(architecture)
        .with_hidden_size(hidden_size)
        .with_intermediate_size(intermediate_size)
        .with_num_hidden_layers(num_hidden_layers)
        .with_num_attention_heads(num_attention_heads)
        .with_num_key_value_heads(num_key_value_heads)
        .with_head_dim(head_dim)
        .with_vocab_size(vocab_size)
        .with_max_position_embeddings(max_position_embeddings)
        .with_rms_norm_eps(rms_norm_eps)
        .with_rope_theta(rope_theta)
        .with_tie_word_embeddings(tie_word_embeddings)
        .with_extra(extra))
}

// ---------------------------------------------------------------------------
// Shard discovery
// ---------------------------------------------------------------------------

/// Discover safetensors shard files in a model directory.
///
/// Checks for `model.safetensors.index.json` first (sharded models),
/// then falls back to globbing `*.safetensors`.
fn discover_shard_files(model_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let text = fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            MilError::Validation(format!("invalid model.safetensors.index.json: {e}"))
        })?;

        // The weight_map maps tensor names to shard filenames.
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                MilError::Validation("model.safetensors.index.json missing 'weight_map'".into())
            })?;

        // Collect unique shard filenames, preserving order.
        let mut seen = std::collections::HashSet::new();
        let mut shard_files = Vec::new();
        for filename in weight_map.values() {
            if let Some(f) = filename.as_str() {
                if seen.insert(f.to_string()) {
                    let path = model_dir.join(f);
                    if !path.exists() {
                        return Err(MilError::Validation(format!(
                            "shard file not found: {}",
                            path.display()
                        )));
                    }
                    shard_files.push(path);
                }
            }
        }

        return Ok(shard_files);
    }

    // Fallback: glob for *.safetensors files.
    let mut files: Vec<std::path::PathBuf> = fs::read_dir(model_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    files.sort();
    Ok(files)
}

// ---------------------------------------------------------------------------
// LoRA detection & merge
// ---------------------------------------------------------------------------

/// Detect LoRA adapters in the loaded tensors and merge them into base weights.
fn detect_and_merge_lora(
    mmaps: &[Mmap],
    tensor_index: &HashMap<String, TensorLocation>,
    model_dir: &Path,
) -> Result<HashMap<String, MergedTensor>> {
    let mut merged = HashMap::new();

    // Check for LoRA indicators.
    let adapter_config_path = model_dir.join("adapter_config.json");
    let has_adapter_config = adapter_config_path.exists();
    let has_lora_tensors = tensor_index
        .keys()
        .any(|name| name.contains("lora_A") || name.contains("lora_B"));

    if !has_adapter_config && !has_lora_tensors {
        return Ok(merged);
    }

    // Read optional alpha from adapter_config.json.
    let global_alpha = if has_adapter_config {
        let text = fs::read_to_string(&adapter_config_path)?;
        let cfg: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| MilError::Validation(format!("invalid adapter_config.json: {e}")))?;
        cfg.get("lora_alpha").and_then(|v| v.as_f64())
    } else {
        None
    };

    // Collect LoRA A/B pairs.
    let lora_a_names: Vec<&str> = tensor_index
        .keys()
        .filter(|n| n.ends_with(".lora_A.weight"))
        .map(|s| s.as_str())
        .collect();

    for a_name in &lora_a_names {
        let prefix = match a_name.strip_suffix(".lora_A.weight") {
            Some(p) => p,
            None => continue,
        };
        let b_name = format!("{prefix}.lora_B.weight");
        let base_name = format!("{prefix}.weight");

        // Both A and B must exist.
        let a_loc = match tensor_index.get(*a_name) {
            Some(loc) => loc,
            None => continue,
        };
        let b_loc = match tensor_index.get(&b_name) {
            Some(loc) => loc,
            None => continue,
        };
        let base_loc = match tensor_index.get(&base_name) {
            Some(loc) => loc,
            None => continue,
        };

        // Read tensor views.
        let a_view = get_tensor_view(&mmaps[a_loc.shard_index], a_name)?;
        let b_view = get_tensor_view(&mmaps[b_loc.shard_index], &b_name)?;
        let base_view = get_tensor_view(&mmaps[base_loc.shard_index], &base_name)?;

        let a_dtype = a_view.dtype;
        let b_dtype = b_view.dtype;
        let base_dtype = base_view.dtype;

        if a_dtype != b_dtype || a_dtype != base_dtype {
            continue;
        }
        if a_view.shape.len() != 2 || b_view.shape.len() != 2 {
            continue;
        }

        // Try to read per-adapter alpha.
        let alpha_name = format!("{prefix}.lora_alpha");
        let alpha = if let Some(alpha_loc) = tensor_index.get(&alpha_name) {
            let alpha_view = get_tensor_view(&mmaps[alpha_loc.shard_index], &alpha_name)?;
            let numel: usize = alpha_view.shape.iter().product();
            if numel == 1 {
                let alpha_dtype = alpha_view.dtype;
                lora::scalar_from_bytes(&alpha_view.data, alpha_dtype)
            } else {
                global_alpha
            }
        } else {
            global_alpha
        };

        let adapter = LoraAdapter::new(
            base_name.clone(),
            a_view.data.to_vec(),
            [a_view.shape[0], a_view.shape[1]],
            b_view.data.to_vec(),
            [b_view.shape[0], b_view.shape[1]],
            a_dtype,
            alpha,
        );

        let mut base_data = base_view.data.to_vec();
        let base_shape = base_view.shape.clone();

        lora::merge_lora_weights(
            &mut base_data,
            &base_shape,
            &lora::LoraWeights::new(
                &adapter.a_data,
                &adapter.a_shape,
                &adapter.b_data,
                &adapter.b_shape,
                base_dtype,
                adapter.alpha,
            ),
        )?;

        merged.insert(base_name, (base_data, base_shape, base_dtype));
    }

    Ok(merged)
}

/// Lightweight tensor view extracted from a safetensors shard.
struct RawTensorView<'a> {
    data: std::borrow::Cow<'a, [u8]>,
    shape: Vec<usize>,
    dtype: ScalarType,
}

/// Get a tensor view from a memory-mapped safetensors shard.
///
/// BF16 data is automatically converted to FP16.
fn get_tensor_view<'a>(mmap: &'a Mmap, name: &str) -> Result<RawTensorView<'a>> {
    let st = SafeTensors::deserialize(mmap)
        .map_err(|e| MilError::Validation(format!("failed to deserialize shard: {e}")))?;
    let view = st
        .tensor(name)
        .map_err(|e| MilError::Validation(format!("tensor '{name}' not found in shard: {e}")))?;
    let st_dtype = view.dtype();
    let dtype = safetensors_dtype_to_scalar(st_dtype)?;
    let shape = view.shape().to_vec();
    if st_dtype == safetensors::Dtype::BF16 {
        let converted = convert_bf16_to_f16(view.data());
        Ok(RawTensorView {
            data: std::borrow::Cow::Owned(converted),
            shape,
            dtype,
        })
    } else {
        Ok(RawTensorView {
            data: std::borrow::Cow::Borrowed(view.data()),
            shape,
            dtype,
        })
    }
}

// ---------------------------------------------------------------------------
// Dtype conversion
// ---------------------------------------------------------------------------

/// Convert a safetensors [`Dtype`](safetensors::Dtype) to an ironmill
/// [`ScalarType`].
fn safetensors_dtype_to_scalar(dtype: safetensors::Dtype) -> Result<ScalarType> {
    use safetensors::Dtype;
    match dtype {
        Dtype::F16 => Ok(ScalarType::Float16),
        Dtype::BF16 => Ok(ScalarType::Float16), // data converted at read time
        Dtype::F32 => Ok(ScalarType::Float32),
        Dtype::F64 => Ok(ScalarType::Float64),
        Dtype::I8 => Ok(ScalarType::Int8),
        Dtype::I16 => Ok(ScalarType::Int16),
        Dtype::I32 => Ok(ScalarType::Int32),
        Dtype::I64 => Ok(ScalarType::Int64),
        Dtype::U8 => Ok(ScalarType::UInt8),
        Dtype::U16 => Ok(ScalarType::UInt16),
        Dtype::U32 => Ok(ScalarType::UInt32),
        Dtype::U64 => Ok(ScalarType::UInt64),
        Dtype::BOOL => Ok(ScalarType::Bool),
        other => Err(MilError::Validation(format!(
            "unsupported safetensors dtype: {other:?}"
        ))),
    }
}

/// Convert BF16 raw bytes to FP16 raw bytes.
///
/// Each pair of bytes is read as a bf16 value, converted through f32 to f16.
fn convert_bf16_to_f16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for chunk in data.chunks_exact(2) {
        let bf = half::bf16::from_le_bytes([chunk[0], chunk[1]]);
        let fp = half::f16::from_f32(bf.to_f32());
        out.extend_from_slice(&fp.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

fn json_usize(json: &serde_json::Value, key: &str) -> Option<usize> {
    json.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
}

fn json_f64(json: &serde_json::Value, key: &str) -> Option<f64> {
    json.get(key).and_then(|v| v.as_f64())
}

use std::str::FromStr;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_dtype_conversion() {
        use safetensors::Dtype;
        assert_eq!(
            safetensors_dtype_to_scalar(Dtype::F16).unwrap(),
            ScalarType::Float16
        );
        assert_eq!(
            safetensors_dtype_to_scalar(Dtype::BF16).unwrap(),
            ScalarType::Float16
        );
        assert_eq!(
            safetensors_dtype_to_scalar(Dtype::F32).unwrap(),
            ScalarType::Float32
        );
        assert_eq!(
            safetensors_dtype_to_scalar(Dtype::F64).unwrap(),
            ScalarType::Float64
        );
        assert_eq!(
            safetensors_dtype_to_scalar(Dtype::I32).unwrap(),
            ScalarType::Int32
        );
        assert_eq!(
            safetensors_dtype_to_scalar(Dtype::BOOL).unwrap(),
            ScalarType::Bool
        );
    }

    #[test]
    fn test_convert_bf16_to_f16() {
        // BF16 for 1.0: upper 16 bits of f32 1.0 = 0x3F80
        let bf16_one = half::bf16::from_f32(1.0);
        let bf16_neg = half::bf16::from_f32(-2.5);
        let mut raw = Vec::new();
        raw.extend_from_slice(&bf16_one.to_le_bytes());
        raw.extend_from_slice(&bf16_neg.to_le_bytes());

        let result = convert_bf16_to_f16(&raw);
        assert_eq!(result.len(), 4); // 2 × 2 bytes

        let h0 = half::f16::from_le_bytes([result[0], result[1]]);
        let h1 = half::f16::from_le_bytes([result[2], result[3]]);
        assert!((h0.to_f32() - 1.0).abs() < 0.01);
        assert!((h1.to_f32() - (-2.5)).abs() < 0.01);
    }

    #[test]
    fn test_load_bf16_tensor() {
        use safetensors::tensor::serialize;
        use std::collections::HashMap as StdHashMap;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "llama", "hidden_size": 4, "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 32000}"#,
        )
        .unwrap();

        // Create BF16 tensor data: two bf16 values [1.0, -0.5]
        let bf_vals: Vec<half::bf16> = vec![half::bf16::from_f32(1.0), half::bf16::from_f32(-0.5)];
        let data: Vec<u8> = bf_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape = vec![2];
        let mut tensors = StdHashMap::new();
        tensors.insert(
            "test.weight",
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, shape, &data).unwrap(),
        );
        let serialized = serialize(tensors, None).unwrap();
        std::fs::write(dir.path().join("model.safetensors"), &serialized).unwrap();

        let provider = SafeTensorsProvider::load(dir.path()).unwrap();
        let tensor = provider.tensor("test.weight").unwrap();
        assert_eq!(tensor.dtype, ScalarType::Float16);
        assert_eq!(tensor.shape, vec![2]);
        // Data should be FP16 now
        let h0 = half::f16::from_le_bytes([tensor.data[0], tensor.data[1]]);
        let h1 = half::f16::from_le_bytes([tensor.data[2], tensor.data[3]]);
        assert!((h0.to_f32() - 1.0).abs() < 0.01);
        assert!((h1.to_f32() - (-0.5)).abs() < 0.01);
    }

    #[test]
    fn test_parse_hf_config_minimal() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "vocab_size": 32000
            }"#,
        )
        .unwrap();

        let config = parse_hf_config(&config_path).unwrap();
        assert_eq!(config.architecture, Architecture::Llama);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.intermediate_size, 11008);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 32); // defaults to num_attention_heads
        assert_eq!(config.head_dim, 128); // 4096/32
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.rms_norm_eps, 1e-6); // default
        assert_eq!(config.rope_theta, 10000.0); // default
        assert!(!config.tie_word_embeddings); // default
    }

    #[test]
    fn test_parse_hf_config_with_all_fields() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{
                "model_type": "qwen2",
                "hidden_size": 2048,
                "intermediate_size": 5632,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": true,
                "use_sliding_window": false
            }"#,
        )
        .unwrap();

        let config = parse_hf_config(&config_path).unwrap();
        assert_eq!(config.architecture, Architecture::Qwen);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.rms_norm_eps, 1e-5);
        assert_eq!(config.rope_theta, 1000000.0);
        assert!(config.tie_word_embeddings);
        // Extra field captured.
        assert!(config.extra.contains_key("use_sliding_window"));
    }

    #[test]
    fn test_parse_hf_config_sliding_window() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{
                "model_type": "qwen3",
                "hidden_size": 2048,
                "intermediate_size": 5632,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "vocab_size": 151936,
                "sliding_window": 4096,
                "max_window_layers": 21
            }"#,
        )
        .unwrap();

        let config = parse_hf_config(&config_path).unwrap();
        assert_eq!(config.architecture, Architecture::Qwen);
        assert_eq!(config.num_hidden_layers, 28);
        // Sliding window fields captured in extra and accessible via helpers.
        assert_eq!(config.sliding_window(), Some(4096));
        assert_eq!(config.max_window_layers(), Some(21));
    }

    #[test]
    fn test_parse_hf_config_missing_model_type() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(&config_path, r#"{"hidden_size": 4096}"#).unwrap();

        let result = parse_hf_config(&config_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model_type"));
    }

    #[test]
    fn test_load_missing_config() {
        let dir = tempfile::tempdir().unwrap();
        let result = SafeTensorsProvider::load(dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("config.json"));
    }

    #[test]
    fn test_load_no_safetensors_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "llama", "hidden_size": 128, "intermediate_size": 512, "num_hidden_layers": 2, "num_attention_heads": 4, "vocab_size": 32000}"#,
        )
        .unwrap();

        let result = SafeTensorsProvider::load(dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no .safetensors"));
    }

    #[test]
    fn test_load_single_shard() {
        use safetensors::tensor::serialize;
        use std::collections::HashMap as StdHashMap;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "llama", "hidden_size": 4, "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 32000}"#,
        )
        .unwrap();

        // Create a minimal safetensors file with one tensor.
        let data: Vec<u8> = vec![0u8; 16]; // 4 float32s
        let shape = vec![2, 2];
        let mut tensors = StdHashMap::new();
        tensors.insert(
            "test.weight",
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, &data).unwrap(),
        );
        let serialized = serialize(tensors, None).unwrap();
        std::fs::write(dir.path().join("model.safetensors"), &serialized).unwrap();

        let provider = SafeTensorsProvider::load(dir.path()).unwrap();
        assert_eq!(provider.tensor_names().len(), 1);
        assert!(provider.has_tensor("test.weight"));

        let tensor = provider.tensor("test.weight").unwrap();
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.dtype, ScalarType::Float32);
        assert_eq!(tensor.data.len(), 16);
    }

    #[test]
    fn test_load_sharded_via_index() {
        use safetensors::tensor::serialize;
        use std::collections::HashMap as StdHashMap;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "gemma", "hidden_size": 4, "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 32000}"#,
        )
        .unwrap();

        // Create two shard files.
        let data1: Vec<u8> = vec![0u8; 8]; // 2 float32s
        let mut t1 = StdHashMap::new();
        t1.insert(
            "shard1.weight",
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &data1).unwrap(),
        );
        let s1 = serialize(t1, None).unwrap();
        std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), &s1).unwrap();

        let data2: Vec<u8> = vec![0u8; 16];
        let mut t2 = StdHashMap::new();
        t2.insert(
            "shard2.weight",
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &data2)
                .unwrap(),
        );
        let s2 = serialize(t2, None).unwrap();
        std::fs::write(dir.path().join("model-00002-of-00002.safetensors"), &s2).unwrap();

        // Index file.
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{
                "weight_map": {
                    "shard1.weight": "model-00001-of-00002.safetensors",
                    "shard2.weight": "model-00002-of-00002.safetensors"
                }
            }"#,
        )
        .unwrap();

        let provider = SafeTensorsProvider::load(dir.path()).unwrap();
        let mut names = provider.tensor_names();
        names.sort();
        assert_eq!(names, vec!["shard1.weight", "shard2.weight"]);

        let t = provider.tensor("shard2.weight").unwrap();
        assert_eq!(t.shape, vec![2, 2]);
    }

    #[test]
    fn test_parse_hf_config_gemma4_nested() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 2304,
                "intermediate_size": 9216,
                "num_hidden_layers": 30,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "vocab_size": 262144,
                "max_position_embeddings": 131072,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "layer_types": ["sliding_attention", "full_attention"],
                "rope_parameters": {
                    "sliding_attention": {"rope_theta": 10000.0},
                    "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
                }
            }
        }"#,
        )
        .unwrap();
        let config = parse_hf_config(&config_path).unwrap();
        assert_eq!(config.architecture, Architecture::Gemma);
        assert_eq!(config.hidden_size, 2304);
        assert_eq!(config.num_hidden_layers, 30);
        assert!(config.layer_types().is_some());
        assert!(config.rope_parameters().is_some());
    }

    #[test]
    fn test_parse_hf_config_gemma4_text_direct() {
        // Test direct gemma4_text config (not wrapped in multimodal)
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 35,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "vocab_size": 262144,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0
        }"#,
        )
        .unwrap();
        let config = parse_hf_config(&config_path).unwrap();
        assert_eq!(config.architecture, Architecture::Gemma);
        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.num_key_value_heads, 1);
    }
}
