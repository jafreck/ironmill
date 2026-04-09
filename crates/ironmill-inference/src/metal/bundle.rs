//! GPU bundle format (`.ironml-gpu`) reader.
//!
//! Provides [`MetalBundleProvider`], a [`WeightProvider`] implementation that
//! loads tensors from a pre-compiled `.ironml-gpu` bundle directory. Weight
//! files are read from disk on each `tensor()` call and returned as owned data.

use std::fs;
use std::path::{Path, PathBuf};

use ironmill_core::gpu::bundle::{
    GpuBundleManifest, deserialize_model_config, read_tensor_from_manifest,
};
use ironmill_core::weights::{ModelConfig, WeightProvider, WeightTensor};
use mil_rs::MilError;

use super::error::MetalError;

/// Weight provider backed by a `.ironml-gpu` bundle directory.
///
/// Reads `manifest.json` at open time and loads individual weight files from
/// disk on each [`WeightProvider::tensor`] call.
pub struct MetalBundleProvider {
    bundle_path: PathBuf,
    manifest: GpuBundleManifest,
    config: ModelConfig,
}

impl MetalBundleProvider {
    /// Open a `.ironml-gpu` bundle directory.
    ///
    /// Reads and validates `manifest.json`, then prepares for on-demand
    /// weight loading.
    pub fn open(bundle_path: impl AsRef<Path>) -> Result<Self, MetalError> {
        let bundle_path = bundle_path.as_ref().to_path_buf();

        let manifest_path = bundle_path.join("manifest.json");
        let manifest_json = fs::read_to_string(&manifest_path).map_err(|e| {
            MetalError::WeightLoading(format!(
                "failed to read manifest at {}: {e}",
                manifest_path.display()
            ))
        })?;

        let manifest: GpuBundleManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| MetalError::WeightLoading(format!("failed to parse manifest: {e}")))?;

        if manifest.format_version != 1 {
            return Err(MetalError::WeightLoading(format!(
                "unsupported bundle format version: {}",
                manifest.format_version
            )));
        }

        let config = deserialize_model_config(&manifest.model_config)
            .map_err(|e| MetalError::Config(e.to_string()))?;

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

impl WeightProvider for MetalBundleProvider {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError> {
        let desc = self.manifest.tensors.get(name).ok_or_else(|| {
            MilError::UndefinedValue(format!("tensor not found in bundle: {name}"))
        })?;

        read_tensor_from_manifest(desc, |path| self.read_file(path))
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
    use mil_rs::ir::ScalarType;
    use mil_rs::weights::QuantizationInfo;

    /// Write a minimal `.ironml-gpu` bundle directory from raw components.
    fn write_test_bundle(
        bundle_path: &Path,
        config: &ModelConfig,
        tensors: &[(
            &str,            // canonical name
            &str,            // sanitized name (for files)
            &[u8],           // data
            Vec<usize>,      // shape
            &str,            // dtype string
            Option<TestLut>, // quantization info
        )],
    ) {
        let weights_dir = bundle_path.join("weights");
        fs::create_dir_all(&weights_dir).unwrap();

        let mut tensor_map = serde_json::Map::new();
        for (name, sanitized, data, shape, dtype, quant) in tensors {
            match quant {
                Some(lut) => {
                    fs::write(weights_dir.join(format!("{sanitized}.bin")), &lut.indices).unwrap();
                    fs::write(weights_dir.join(format!("{sanitized}.lut")), &lut.lut).unwrap();
                    fs::write(weights_dir.join(format!("{sanitized}.nrm")), &lut.norms).unwrap();
                    tensor_map.insert(
                        name.to_string(),
                        serde_json::json!({
                            "format": "lut_to_dense",
                            "indices_file": format!("weights/{sanitized}.bin"),
                            "lut_file": format!("weights/{sanitized}.lut"),
                            "norms_file": format!("weights/{sanitized}.nrm"),
                            "shape": shape,
                            "n_bits": lut.n_bits,
                            "dtype": dtype,
                        }),
                    );
                }
                None => {
                    fs::write(weights_dir.join(format!("{sanitized}.bin")), data).unwrap();
                    tensor_map.insert(
                        name.to_string(),
                        serde_json::json!({
                            "format": "dense",
                            "file": format!("weights/{sanitized}.bin"),
                            "shape": shape,
                            "dtype": dtype,
                        }),
                    );
                }
            }
        }

        let manifest = serde_json::json!({
            "format_version": 1,
            "model_config": {
                "architecture": format!("{:?}", config.architecture).to_lowercase(),
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "vocab_size": config.vocab_size,
                "max_position_embeddings": config.max_position_embeddings,
                "rms_norm_eps": config.rms_norm_eps,
                "rope_theta": config.rope_theta,
                "tie_word_embeddings": config.tie_word_embeddings,
                "extra": config.extra,
            },
            "quantization": {
                "method": "none",
                "n_bits": 16,
                "seed": 42,
                "min_elements": 1024,
            },
            "tensors": tensor_map,
        });

        fs::write(
            bundle_path.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    struct TestLut {
        indices: Vec<u8>,
        lut: Vec<u8>,
        norms: Vec<u8>,
        n_bits: u32,
    }

    fn test_config() -> ModelConfig {
        use ironmill_core::weights::Architecture;
        ModelConfig::new(Architecture::Llama)
            .with_hidden_size(2048)
            .with_intermediate_size(5504)
            .with_num_hidden_layers(2)
            .with_num_attention_heads(16)
            .with_num_key_value_heads(16)
            .with_head_dim(128)
            .with_vocab_size(32000)
            .with_max_position_embeddings(2048)
    }

    fn write_mock_bundle(bundle_path: &Path) {
        let config = test_config();
        write_test_bundle(
            bundle_path,
            &config,
            &[
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    "model_layers_0_self_attn_q_proj_weight",
                    &[], // data unused for quantized
                    vec![2048, 2048],
                    "float16",
                    Some(TestLut {
                        indices: vec![0xAA; 64],
                        lut: vec![0x11; 32],
                        norms: vec![0x22; 16],
                        n_bits: 4,
                    }),
                ),
                (
                    "model.embed_tokens.weight",
                    "model_embed_tokens_weight",
                    &[0xBB; 128],
                    vec![32000, 2048],
                    "float16",
                    None,
                ),
            ],
        );
    }

    #[test]
    fn round_trip_dense_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        write_mock_bundle(&bundle_path);

        let reader = MetalBundleProvider::open(&bundle_path).unwrap();
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

        write_mock_bundle(&bundle_path);

        let reader = MetalBundleProvider::open(&bundle_path).unwrap();
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
                ..
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

        write_mock_bundle(&bundle_path);

        let reader = MetalBundleProvider::open(&bundle_path).unwrap();
        let config = reader.config();
        let original = test_config();

        assert_eq!(config.architecture, original.architecture);
        assert_eq!(config.hidden_size, original.hidden_size);
        assert_eq!(config.num_hidden_layers, original.num_hidden_layers);
        assert_eq!(config.vocab_size, original.vocab_size);
    }

    #[test]
    fn round_trip_tensor_names() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        write_mock_bundle(&bundle_path);

        let reader = MetalBundleProvider::open(&bundle_path).unwrap();
        let mut names = reader.tensor_names();
        names.sort();

        let mut expected = vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ];
        expected.sort();

        assert_eq!(names, expected);
    }

    #[test]
    fn open_missing_bundle_returns_error() {
        let result = MetalBundleProvider::open("/nonexistent/path.ironml-gpu");
        assert!(result.is_err());
    }

    #[test]
    fn tensor_not_found_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        write_mock_bundle(&bundle_path);

        let reader = MetalBundleProvider::open(&bundle_path).unwrap();
        let result = reader.tensor("nonexistent.tensor");
        assert!(result.is_err());
    }

    struct TestAffine {
        quantized_data: Vec<u8>,
        scales: Vec<u8>,
        zeros: Vec<u8>,
        bit_width: u8,
        group_size: usize,
        axis: i64,
    }

    /// Write a bundle containing an affine-quantized tensor.
    fn write_test_bundle_with_affine(
        bundle_path: &Path,
        config: &ModelConfig,
        name: &str,
        sanitized: &str,
        shape: Vec<usize>,
        dtype: &str,
        affine: &TestAffine,
    ) {
        let weights_dir = bundle_path.join("weights");
        fs::create_dir_all(&weights_dir).unwrap();

        fs::write(
            weights_dir.join(format!("{sanitized}.qdata")),
            &affine.quantized_data,
        )
        .unwrap();
        fs::write(
            weights_dir.join(format!("{sanitized}.scale")),
            &affine.scales,
        )
        .unwrap();
        fs::write(
            weights_dir.join(format!("{sanitized}.zeros")),
            &affine.zeros,
        )
        .unwrap();

        let mut tensor_map = serde_json::Map::new();
        tensor_map.insert(
            name.to_string(),
            serde_json::json!({
                "format": "affine_dequantize",
                "quantized_data_file": format!("weights/{sanitized}.qdata"),
                "scales_file": format!("weights/{sanitized}.scale"),
                "zeros_file": format!("weights/{sanitized}.zeros"),
                "shape": shape,
                "bit_width": affine.bit_width,
                "group_size": affine.group_size,
                "axis": affine.axis,
                "dtype": dtype,
            }),
        );

        let manifest = serde_json::json!({
            "format_version": 1,
            "model_config": {
                "architecture": format!("{:?}", config.architecture).to_lowercase(),
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "vocab_size": config.vocab_size,
                "max_position_embeddings": config.max_position_embeddings,
                "rms_norm_eps": config.rms_norm_eps,
                "rope_theta": config.rope_theta,
                "tie_word_embeddings": config.tie_word_embeddings,
                "extra": config.extra,
            },
            "quantization": {
                "method": "none",
                "n_bits": 16,
                "seed": 42,
                "min_elements": 1024,
            },
            "tensors": tensor_map,
        });

        fs::write(
            bundle_path.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn round_trip_affine_quantized_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");

        let config = test_config();
        write_test_bundle_with_affine(
            &bundle_path,
            &config,
            "model.layers.0.mlp.gate_proj.weight",
            "model_layers_0_mlp_gate_proj_weight",
            vec![2048, 2048],
            "uint8",
            &TestAffine {
                quantized_data: vec![0xCC; 32],
                scales: vec![0xDD; 16],
                zeros: vec![0xEE; 16],
                bit_width: 4,
                group_size: 128,
                axis: 1,
            },
        );

        let reader = MetalBundleProvider::open(&bundle_path).unwrap();
        let tensor = reader
            .tensor("model.layers.0.mlp.gate_proj.weight")
            .unwrap();

        assert_eq!(tensor.dtype, ScalarType::UInt8);
        assert_eq!(tensor.data.as_ref(), &[0xCC; 32]);
        assert_eq!(tensor.shape, vec![2048, 2048]);

        match &tensor.quant_info {
            QuantizationInfo::AffineDequantize {
                scale,
                zero_point,
                scale_dtype,
                zero_point_dtype,
                axis,
                bit_width,
                group_size,
                ..
            } => {
                assert_eq!(scale, &vec![0xDD; 16]);
                assert_eq!(zero_point, &vec![0xEE; 16]);
                assert_eq!(*scale_dtype, ScalarType::Float16);
                assert_eq!(*zero_point_dtype, ScalarType::Float16);
                assert_eq!(*axis, Some(1));
                assert_eq!(*bit_width, 4);
                assert_eq!(*group_size, Some(128));
            }
            other => panic!("expected AffineDequantize, got {other:?}"),
        }
    }
}
