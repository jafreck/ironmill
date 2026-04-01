//! GPU bundle format (`.ironml-gpu`) reader.
//!
//! Provides [`GpuBundleProvider`], a [`WeightProvider`] implementation that
//! loads tensors from a pre-compiled `.ironml-gpu` bundle directory. Weight
//! files are read from disk on each `tensor()` call and returned as owned data.

use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};

use ironmill_compile::gpu::bundle::{
    GpuBundleManifest, TensorManifest, deserialize_model_config, str_to_scalar_type,
};
use ironmill_compile::weights::{ModelConfig, QuantizationInfo, WeightProvider, WeightTensor};
use mil_rs::MilError;

use super::error::GpuError;

/// Weight provider backed by a `.ironml-gpu` bundle directory.
///
/// Reads `manifest.json` at open time and loads individual weight files from
/// disk on each [`WeightProvider::tensor`] call.
pub struct GpuBundleProvider {
    bundle_path: PathBuf,
    manifest: GpuBundleManifest,
    config: ModelConfig,
}

impl GpuBundleProvider {
    /// Open a `.ironml-gpu` bundle directory.
    ///
    /// Reads and validates `manifest.json`, then prepares for on-demand
    /// weight loading.
    pub fn open(bundle_path: impl AsRef<Path>) -> Result<Self, GpuError> {
        let bundle_path = bundle_path.as_ref().to_path_buf();

        let manifest_path = bundle_path.join("manifest.json");
        let manifest_json = fs::read_to_string(&manifest_path).map_err(|e| {
            GpuError::WeightLoading(format!(
                "failed to read manifest at {}: {e}",
                manifest_path.display()
            ))
        })?;

        let manifest: GpuBundleManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| GpuError::WeightLoading(format!("failed to parse manifest: {e}")))?;

        if manifest.format_version != 1 {
            return Err(GpuError::WeightLoading(format!(
                "unsupported bundle format version: {}",
                manifest.format_version
            )));
        }

        let config = deserialize_model_config(&manifest.model_config)
            .map_err(|e| GpuError::Config(e.to_string()))?;

        Ok(Self {
            bundle_path,
            manifest,
            config,
        })
    }

    /// Access the bundle manifest for quantization info and metadata.
    pub fn manifest(&self) -> &GpuBundleManifest {
        &self.manifest
    }

    /// Read a file relative to the bundle root.
    fn read_file(&self, relative_path: &str) -> Result<Vec<u8>, MilError> {
        fs::read(self.bundle_path.join(relative_path))
            .map_err(|e| MilError::Validation(format!("failed to read {relative_path}: {e}")))
    }
}

impl WeightProvider for GpuBundleProvider {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError> {
        let desc = self.manifest.tensors.get(name).ok_or_else(|| {
            MilError::UndefinedValue(format!("tensor not found in bundle: {name}"))
        })?;

        match desc {
            TensorManifest::LutToDense {
                indices_file,
                lut_file,
                norms_file,
                shape,
                n_bits,
                dtype,
            } => {
                let indices = self.read_file(indices_file)?;
                let lut = self.read_file(lut_file)?;
                let row_norms = self.read_file(norms_file)?;
                let dtype =
                    str_to_scalar_type(dtype).map_err(|e| MilError::Validation(e.to_string()))?;

                Ok(WeightTensor {
                    data: Cow::Owned(indices.clone()),
                    shape: shape.clone(),
                    dtype,
                    quant_info: QuantizationInfo::LutToDense {
                        lut,
                        lut_dtype: dtype,
                        indices,
                        original_shape: shape.clone(),
                        n_bits: *n_bits,
                        row_norms,
                        norms_dtype: dtype,
                    },
                })
            }
            TensorManifest::Dense { file, shape, dtype } => {
                let data = self.read_file(file)?;
                let dtype =
                    str_to_scalar_type(dtype).map_err(|e| MilError::Validation(e.to_string()))?;

                Ok(WeightTensor {
                    data: Cow::Owned(data),
                    shape: shape.clone(),
                    dtype,
                    quant_info: QuantizationInfo::None,
                })
            }
        }
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.manifest.tensors.keys().map(|s| s.as_str()).collect()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironmill_compile::gpu::bundle::write_gpu_bundle;
    use mil_rs::ir::ScalarType;
    use std::collections::HashMap;

    /// Minimal provider for round-trip testing.
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
        fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError> {
            let t = self
                .tensors
                .get(name)
                .ok_or_else(|| MilError::UndefinedValue(name.to_string()))?;
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
        use ironmill_compile::weights::Architecture;
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

        // Quantized tensor
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

        // Dense tensor
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
    fn round_trip_dense_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        let provider = mock_provider();
        write_gpu_bundle(&provider, &bundle_path).unwrap();

        let reader = GpuBundleProvider::open(&bundle_path).unwrap();
        let tensor = reader.tensor("model.embed_tokens.weight").unwrap();

        assert_eq!(tensor.data.as_ref(), &[0xBB; 128]);
        assert_eq!(tensor.shape, vec![32000, 2048]);
        assert_eq!(tensor.dtype, ScalarType::Float16);
        assert!(matches!(tensor.quant_info, QuantizationInfo::None));
    }

    #[test]
    fn round_trip_quantized_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        let provider = mock_provider();
        write_gpu_bundle(&provider, &bundle_path).unwrap();

        let reader = GpuBundleProvider::open(&bundle_path).unwrap();
        let tensor = reader
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        match &tensor.quant_info {
            QuantizationInfo::LutToDense {
                lut,
                lut_dtype,
                indices,
                original_shape,
                n_bits,
                row_norms,
                norms_dtype,
            } => {
                assert_eq!(indices, &vec![0xAA; 64]);
                assert_eq!(lut, &vec![0x11; 32]);
                assert_eq!(row_norms, &vec![0x22; 16]);
                assert_eq!(*lut_dtype, ScalarType::Float16);
                assert_eq!(*norms_dtype, ScalarType::Float16);
                assert_eq!(*n_bits, 4);
                assert_eq!(original_shape, &vec![2048, 2048]);
            }
            other => panic!("expected LutToDense, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_config() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        let provider = mock_provider();
        write_gpu_bundle(&provider, &bundle_path).unwrap();

        let reader = GpuBundleProvider::open(&bundle_path).unwrap();
        let config = reader.config();
        let original = provider.config();

        assert_eq!(config.architecture, original.architecture);
        assert_eq!(config.hidden_size, original.hidden_size);
        assert_eq!(config.num_hidden_layers, original.num_hidden_layers);
        assert_eq!(config.vocab_size, original.vocab_size);
    }

    #[test]
    fn round_trip_tensor_names() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        let provider = mock_provider();
        write_gpu_bundle(&provider, &bundle_path).unwrap();

        let reader = GpuBundleProvider::open(&bundle_path).unwrap();
        let mut names = reader.tensor_names();
        names.sort();

        let mut expected = provider.tensor_names();
        expected.sort();

        assert_eq!(names, expected);
    }

    #[test]
    fn open_missing_bundle_returns_error() {
        let result = GpuBundleProvider::open("/nonexistent/path.ironml-gpu");
        assert!(result.is_err());
    }

    #[test]
    fn tensor_not_found_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        let provider = mock_provider();
        write_gpu_bundle(&provider, &bundle_path).unwrap();

        let reader = GpuBundleProvider::open(&bundle_path).unwrap();
        let result = reader.tensor("nonexistent.tensor");
        assert!(result.is_err());
    }
}
