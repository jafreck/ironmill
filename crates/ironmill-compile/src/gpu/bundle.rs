//! GPU bundle format (`.ironml-gpu`) writer.
//!
//! Serializes a compiled GPU model into a directory bundle so it can be loaded
//! without recompilation. The bundle contains a `manifest.json` plus raw weight
//! files under `weights/`.
//!
//! # Bundle layout
//!
//! ```text
//! model.ironml-gpu/
//! ├── manifest.json
//! └── weights/
//!     ├── <tensor>.bin    # packed indices or dense FP16
//!     ├── <tensor>.lut    # LUT values (raw bytes)
//!     └── <tensor>.nrm    # row norms (raw bytes)
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::error::CompileError;
use crate::weights::{QuantizationInfo, WeightProvider};

pub use ironmill_core::gpu::bundle::{
    GpuBundleManifest, QuantizationManifest, TensorManifest, deserialize_model_config,
    scalar_type_to_str, serialize_model_config, str_to_scalar_type,
};

// ── Filename helpers ────────────────────────────────────────────────────

/// Sanitize a tensor name for use as a filename component.
///
/// Replaces dots and slashes with underscores so the name is safe for all
/// filesystems.
fn sanitize_tensor_name(name: &str) -> String {
    name.replace(['.', '/'], "_")
}

// ── Writer ──────────────────────────────────────────────────────────────

/// Write a GPU bundle to `output_dir`.
///
/// Creates the directory structure, writes weight files, and generates
/// `manifest.json`. The `output_dir` should end with `.ironml-gpu` by
/// convention.
pub fn write_gpu_bundle(
    provider: &dyn WeightProvider,
    output_dir: impl AsRef<Path>,
) -> Result<(), CompileError> {
    let output_dir = output_dir.as_ref();
    let weights_dir = output_dir.join("weights");
    fs::create_dir_all(&weights_dir)?;

    let config = provider.config();
    let tensor_names = provider.tensor_names();

    let mut tensors = HashMap::new();
    let mut global_n_bits: u8 = 4;

    for name in &tensor_names {
        let tensor = provider.tensor(name)?;
        let sanitized = sanitize_tensor_name(name);

        match &tensor.quant_info {
            QuantizationInfo::LutToDense {
                lut,
                lut_dtype,
                indices,
                original_shape,
                n_bits,
                row_norms,
                ..
            } => {
                global_n_bits = *n_bits;

                let indices_file = format!("weights/{sanitized}.bin");
                let lut_file = format!("weights/{sanitized}.lut");
                let norms_file = format!("weights/{sanitized}.nrm");

                fs::write(output_dir.join(&indices_file), indices)?;
                fs::write(output_dir.join(&lut_file), lut)?;
                fs::write(output_dir.join(&norms_file), row_norms)?;

                tensors.insert(
                    name.to_string(),
                    TensorManifest::LutToDense {
                        indices_file,
                        lut_file,
                        norms_file,
                        shape: original_shape.clone(),
                        n_bits: *n_bits,
                        dtype: scalar_type_to_str(*lut_dtype).to_string(),
                    },
                );
            }
            QuantizationInfo::None => {
                let file = format!("weights/{sanitized}.bin");
                fs::write(output_dir.join(&file), &*tensor.data)?;

                tensors.insert(
                    name.to_string(),
                    TensorManifest::Dense {
                        file,
                        shape: tensor.shape.clone(),
                        dtype: scalar_type_to_str(tensor.dtype).to_string(),
                    },
                );
            }
            QuantizationInfo::AffineDequantize { .. } => {
                // AffineDequantize tensors are stored as dense for GPU bundles.
                let file = format!("weights/{sanitized}.bin");
                fs::write(output_dir.join(&file), &*tensor.data)?;

                tensors.insert(
                    name.to_string(),
                    TensorManifest::Dense {
                        file,
                        shape: tensor.shape.clone(),
                        dtype: scalar_type_to_str(tensor.dtype).to_string(),
                    },
                );
            }
        }
    }

    let manifest = GpuBundleManifest {
        format_version: 1,
        model_config: serialize_model_config(config),
        quantization: QuantizationManifest {
            method: "polarquant".to_string(),
            n_bits: global_n_bits,
            seed: 42,
            min_elements: 1024,
        },
        tensors,
    };

    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| CompileError::Other(format!("failed to serialize manifest: {e}")))?;
    fs::write(output_dir.join("manifest.json"), json)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::WeightTensor;
    use std::borrow::Cow;

    /// Minimal provider for testing the bundle writer.
    struct MockProvider {
        tensors: HashMap<String, OwnedTensor>,
        config: ModelConfig,
    }

    struct OwnedTensor {
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: ScalarType,
        quant_info: QuantizationInfo,
    }

    impl WeightProvider for MockProvider {
        fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, mil_rs::MilError> {
            let t = self
                .tensors
                .get(name)
                .ok_or_else(|| mil_rs::MilError::UndefinedValue(name.to_string()))?;
            Ok(WeightTensor {
                data: Cow::Borrowed(&t.data),
                shape: t.shape.clone(),
                dtype: t.dtype,
                quant_info: t.quant_info.clone(),
            })
        }

        fn tensor_names(&self) -> Vec<&str> {
            self.tensors.keys().map(|s| s.as_str()).collect()
        }

        fn config(&self) -> &ModelConfig {
            &self.config
        }
    }

    fn test_config() -> ModelConfig {
        use crate::weights::Architecture;
        ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 2048,
            intermediate_size: 5504,
            num_hidden_layers: 2,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            head_dim: 128,
            vocab_size: 32000,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            extra: HashMap::new(),
        }
    }

    fn mock_provider() -> MockProvider {
        let mut tensors = HashMap::new();

        // A quantized tensor (LutToDense)
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            OwnedTensor {
                data: vec![0xAA; 64],
                shape: vec![2048, 2048],
                dtype: ScalarType::Float16,
                quant_info: QuantizationInfo::LutToDense {
                    lut: vec![0x11; 32],
                    lut_dtype: ScalarType::Float16,
                    indices: vec![0xAA; 64],
                    original_shape: vec![2048, 2048],
                    n_bits: 4,
                    row_norms: vec![0x22; 16],
                    norms_dtype: ScalarType::Float16,
                },
            },
        );

        // A dense tensor
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            OwnedTensor {
                data: vec![0xBB; 128],
                shape: vec![32000, 2048],
                dtype: ScalarType::Float16,
                quant_info: QuantizationInfo::None,
            },
        );

        MockProvider {
            tensors,
            config: test_config(),
        }
    }

    #[test]
    fn write_creates_bundle_directory() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        let provider = mock_provider();
        write_gpu_bundle(&provider, &bundle_path).unwrap();

        assert!(bundle_path.join("manifest.json").exists());
        assert!(bundle_path.join("weights").is_dir());
    }

    #[test]
    fn write_creates_correct_weight_files() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        let provider = mock_provider();
        write_gpu_bundle(&provider, &bundle_path).unwrap();

        // Quantized tensor should produce .bin, .lut, .nrm
        let q_proj_base = "model_layers_0_self_attn_q_proj_weight";
        assert!(
            bundle_path
                .join(format!("weights/{q_proj_base}.bin"))
                .exists()
        );
        assert!(
            bundle_path
                .join(format!("weights/{q_proj_base}.lut"))
                .exists()
        );
        assert!(
            bundle_path
                .join(format!("weights/{q_proj_base}.nrm"))
                .exists()
        );

        // Dense tensor should produce .bin only
        let embed_base = "model_embed_tokens_weight";
        assert!(
            bundle_path
                .join(format!("weights/{embed_base}.bin"))
                .exists()
        );
    }

    #[test]
    fn write_correct_file_contents() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        let provider = mock_provider();
        write_gpu_bundle(&provider, &bundle_path).unwrap();

        let q_proj_base = "model_layers_0_self_attn_q_proj_weight";
        assert_eq!(
            fs::read(bundle_path.join(format!("weights/{q_proj_base}.bin"))).unwrap(),
            vec![0xAA; 64]
        );
        assert_eq!(
            fs::read(bundle_path.join(format!("weights/{q_proj_base}.lut"))).unwrap(),
            vec![0x11; 32]
        );
        assert_eq!(
            fs::read(bundle_path.join(format!("weights/{q_proj_base}.nrm"))).unwrap(),
            vec![0x22; 16]
        );

        let embed_base = "model_embed_tokens_weight";
        assert_eq!(
            fs::read(bundle_path.join(format!("weights/{embed_base}.bin"))).unwrap(),
            vec![0xBB; 128]
        );
    }

    #[test]
    fn manifest_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        let provider = mock_provider();
        write_gpu_bundle(&provider, &bundle_path).unwrap();

        let json = fs::read_to_string(bundle_path.join("manifest.json")).unwrap();
        let manifest: GpuBundleManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(manifest.format_version, 1);
        assert_eq!(manifest.quantization.method, "polarquant");
        assert_eq!(manifest.quantization.n_bits, 4);
        assert_eq!(manifest.tensors.len(), 2);

        // Verify the quantized tensor descriptor
        let q_desc = &manifest.tensors["model.layers.0.self_attn.q_proj.weight"];
        match q_desc {
            TensorManifest::LutToDense {
                n_bits,
                dtype,
                shape,
                ..
            } => {
                assert_eq!(*n_bits, 4);
                assert_eq!(dtype, "float16");
                assert_eq!(shape, &[2048, 2048]);
            }
            _ => panic!("expected LutToDense"),
        }

        // Verify the dense tensor descriptor
        let d_desc = &manifest.tensors["model.embed_tokens.weight"];
        match d_desc {
            TensorManifest::Dense { dtype, shape, .. } => {
                assert_eq!(dtype, "float16");
                assert_eq!(shape, &[32000, 2048]);
            }
            _ => panic!("expected Dense"),
        }
    }

    #[test]
    fn model_config_round_trips() {
        let config = test_config();
        let json_val = serialize_model_config(&config);
        let restored = deserialize_model_config(&json_val).unwrap();

        assert_eq!(restored.architecture, config.architecture);
        assert_eq!(restored.hidden_size, config.hidden_size);
        assert_eq!(restored.intermediate_size, config.intermediate_size);
        assert_eq!(restored.num_hidden_layers, config.num_hidden_layers);
        assert_eq!(restored.num_attention_heads, config.num_attention_heads);
        assert_eq!(restored.num_key_value_heads, config.num_key_value_heads);
        assert_eq!(restored.head_dim, config.head_dim);
        assert_eq!(restored.vocab_size, config.vocab_size);
        assert_eq!(
            restored.max_position_embeddings,
            config.max_position_embeddings
        );
        assert!((restored.rms_norm_eps - config.rms_norm_eps).abs() < f64::EPSILON);
        assert!((restored.rope_theta - config.rope_theta).abs() < f64::EPSILON);
        assert_eq!(restored.tie_word_embeddings, config.tie_word_embeddings);
    }

    #[test]
    fn scalar_type_round_trips() {
        let types = [
            ScalarType::Float16,
            ScalarType::Float32,
            ScalarType::Float64,
            ScalarType::Int8,
            ScalarType::Int16,
            ScalarType::Int32,
            ScalarType::Int64,
            ScalarType::UInt8,
            ScalarType::UInt16,
            ScalarType::UInt32,
            ScalarType::UInt64,
            ScalarType::Bool,
        ];
        for ty in types {
            let s = scalar_type_to_str(ty);
            let restored = str_to_scalar_type(s).unwrap();
            assert_eq!(restored, ty, "round-trip failed for {s}");
        }
    }

    #[test]
    fn sanitize_tensor_name_replaces_dots_and_slashes() {
        assert_eq!(
            sanitize_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            "model_layers_0_self_attn_q_proj_weight"
        );
        assert_eq!(sanitize_tensor_name("a/b.c"), "a_b_c");
    }
}
