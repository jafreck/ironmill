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
//!     ├── <tensor>.bin     # packed indices or dense FP16
//!     ├── <tensor>.lut     # LUT values (raw bytes)
//!     ├── <tensor>.nrm     # row norms (raw bytes)
//!     ├── <tensor>.qdata   # packed quantized data (INT4/INT8)
//!     ├── <tensor>.scale   # per-group scales (FP16)
//!     └── <tensor>.zeros   # per-group zero points (FP16)
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use half::f16;

use crate::error::CompileError;
use crate::weights::{QuantizationInfo, WeightProvider};

use ironmill_core::gpu::bundle::{
    GpuBundleManifest, QuantizationManifest, TensorManifest, scalar_type_to_str,
    serialize_model_config,
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
    let mut global_n_bits: Option<u8> = None;
    let mut methods_seen = std::collections::HashSet::new();

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
                quip_sharp_seed,
                ..
            } => {
                methods_seen.insert("polarquant");

                match global_n_bits {
                    None => global_n_bits = Some(*n_bits),
                    Some(prev) if prev != *n_bits => {
                        eprintln!(
                            "Warning: tensor '{name}' has n_bits={} but global_n_bits is already {prev}",
                            n_bits
                        );
                    }
                    _ => {}
                }

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
                        quip_sharp_seed: quip_sharp_seed.map(|s| s as u32),
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
            QuantizationInfo::AffineDequantize {
                scale,
                zero_point,
                axis,
                bit_width,
                group_size,
                scale_dtype,
                zero_point_dtype,
                awq_scales,
                g_idx: _,
            } => {
                // Per-tensor/per-channel quantization (group_size=None) is not
                // supported in the GPU bundle format. Dequantize these tensors
                // back to FP16 and store them as Dense. These are typically
                // small tensors (RoPE tables, norm weights) where the size
                // impact is negligible.
                if group_size.is_none() || axis.is_none() {
                    let fp16_data = affine_dequantize_to_fp16(
                        &tensor.data,
                        scale,
                        zero_point,
                        *scale_dtype,
                        *zero_point_dtype,
                        *axis,
                        *bit_width,
                        &tensor.shape,
                    )?;
                    eprintln!(
                        "Note: tensor '{name}' has per-channel INT{bit_width} quantization; \
                         storing as FP16 in GPU bundle."
                    );

                    let file = format!("weights/{sanitized}.bin");
                    fs::write(output_dir.join(&file), &fp16_data)?;

                    tensors.insert(
                        name.to_string(),
                        TensorManifest::Dense {
                            file,
                            shape: tensor.shape.clone(),
                            dtype: "float16".to_string(),
                        },
                    );
                } else {
                    let group_size = group_size.ok_or_else(|| {
                        CompileError::Other(format!("tensor '{name}': missing group_size"))
                    })?;
                    let axis = axis.ok_or_else(|| {
                        CompileError::Other(format!("tensor '{name}': missing axis"))
                    })?;

                    methods_seen.insert("affine");

                    match global_n_bits {
                        None => global_n_bits = Some(*bit_width),
                        Some(prev) if prev != *bit_width => {
                            eprintln!(
                                "Warning: tensor '{name}' has bit_width={} but global_n_bits is already {prev}",
                                bit_width
                            );
                        }
                        _ => {}
                    }

                    let qdata_file = format!("weights/{sanitized}.qdata");
                    let scales_file = format!("weights/{sanitized}.scale");
                    let zeros_file = format!("weights/{sanitized}.zeros");

                    fs::write(output_dir.join(&qdata_file), &*tensor.data)?;

                    // Convert scales and zeros to FP16 for the GPU bundle
                    // (the bundle reader assumes FP16 for these parameters).
                    let scales_fp16 = convert_params_to_fp16(scale, *scale_dtype);
                    let zeros_fp16 = convert_params_to_fp16(zero_point, *zero_point_dtype);
                    fs::write(output_dir.join(&scales_file), &scales_fp16)?;
                    fs::write(output_dir.join(&zeros_file), &zeros_fp16)?;

                    let awq_scales_file = if let Some(awq) = awq_scales {
                        let awq_file = format!("weights/{sanitized}.awq");
                        fs::write(output_dir.join(&awq_file), awq)?;
                        Some(awq_file)
                    } else {
                        None
                    };

                    tensors.insert(
                        name.to_string(),
                        TensorManifest::AffineDequantize {
                            quantized_data_file: qdata_file,
                            scales_file,
                            zeros_file,
                            shape: tensor.shape.clone(),
                            bit_width: *bit_width,
                            group_size,
                            axis: axis as i64,
                            dtype: scalar_type_to_str(tensor.dtype).to_string(),
                            awq_scales_file,
                        },
                    );
                }
            }
            other => {
                return Err(CompileError::Other(format!(
                    "unsupported quantization info for tensor '{name}': {other:?}"
                )));
            }
        }
    }

    let method = match methods_seen.len() {
        0 => "none".to_string(),
        // length checked above
        1 => methods_seen
            .into_iter()
            .next()
            .expect("len is 1")
            .to_string(),
        _ => "mixed".to_string(),
    };

    let manifest = GpuBundleManifest {
        format_version: 1,
        model_config: serialize_model_config(config),
        quantization: QuantizationManifest {
            method,
            n_bits: global_n_bits.unwrap_or(4),
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

// ── Parameter dtype conversion helper ────────────────────────────────

/// Convert quantization parameters (scales or zeros) to FP16 bytes.
///
/// The GPU bundle format stores scales/zeros as FP16. This converts
/// from the source dtype (typically Float32 from the quantize pass).
fn convert_params_to_fp16(data: &[u8], dtype: mil_rs::ir::ScalarType) -> Vec<u8> {
    use mil_rs::ir::ScalarType;

    match dtype {
        ScalarType::Float16 => data.to_vec(),
        ScalarType::Float32 => data
            .chunks_exact(4)
            .flat_map(|c| {
                let val = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                f16::from_f32(val).to_le_bytes()
            })
            .collect(),
        ScalarType::UInt8 => data
            .iter()
            .flat_map(|&b| f16::from_f32(b as f32).to_le_bytes())
            .collect(),
        _ => data.to_vec(),
    }
}

// ── Affine dequantization helper ────────────────────────────────────────

/// Dequantize per-tensor/per-channel INT8 data back to FP16.
///
/// Used when writing GPU bundles for tensors that have per-channel
/// quantization (group_size=None), which the bundle format does not support.
/// These are typically small tensors (RoPE tables, norm weights).
#[allow(clippy::too_many_arguments)]
fn affine_dequantize_to_fp16(
    quantized_data: &[u8],
    scale_bytes: &[u8],
    zero_point_bytes: &[u8],
    scale_dtype: mil_rs::ir::ScalarType,
    zero_point_dtype: mil_rs::ir::ScalarType,
    axis: Option<usize>,
    bit_width: u8,
    shape: &[usize],
) -> Result<Vec<u8>, CompileError> {
    use mil_rs::ir::ScalarType;

    if bit_width != 8 {
        return Err(CompileError::Other(format!(
            "affine dequantize to FP16: only INT8 is supported, got {bit_width}-bit"
        )));
    }

    let num_elements: usize = shape.iter().product();

    // Read scales as f32
    let scales: Vec<f32> = match scale_dtype {
        ScalarType::Float16 => scale_bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        ScalarType::Float32 => scale_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => {
            return Err(CompileError::Other(format!(
                "unsupported scale dtype: {scale_dtype:?}"
            )));
        }
    };

    // Read zero points as f32
    let zeros: Vec<f32> = match zero_point_dtype {
        ScalarType::UInt8 => zero_point_bytes.iter().map(|&b| b as f32).collect(),
        ScalarType::Float16 => zero_point_bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        ScalarType::Float32 => zero_point_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        ScalarType::Int8 => zero_point_bytes.iter().map(|&b| b as i8 as f32).collect(),
        _ => {
            return Err(CompileError::Other(format!(
                "unsupported zero_point dtype: {zero_point_dtype:?}"
            )));
        }
    };

    let axis = axis.unwrap_or(0);

    // Compute stride along the quantization axis
    let stride: usize = shape[axis + 1..].iter().product();
    let dim_size = shape[axis];

    if scales.len() != dim_size || zeros.len() != dim_size {
        return Err(CompileError::Other(format!(
            "scale/zero_point length ({}/{}) doesn't match axis dimension ({dim_size})",
            scales.len(),
            zeros.len()
        )));
    }

    let mut fp16_out = Vec::with_capacity(num_elements * 2);
    for (i, &q_byte) in quantized_data.iter().enumerate().take(num_elements) {
        let channel_idx = (i / stride) % dim_size;
        let q_val = q_byte as f32;
        let dequantized = (q_val - zeros[channel_idx]) * scales[channel_idx];
        let h = f16::from_f32(dequantized);
        fp16_out.extend_from_slice(&h.to_le_bytes());
    }

    Ok(fp16_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{ModelConfig, WeightTensor};
    use ironmill_core::gpu::bundle::{deserialize_model_config, str_to_scalar_type};
    use mil_rs::ir::ScalarType;
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
                    polar_quant_seed: None,
                    quip_sharp_seed: None,
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

        // An affine-quantized tensor (AffineDequantize)
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            OwnedTensor {
                data: vec![0xCC; 32], // packed INT4 data
                shape: vec![2048, 2048],
                dtype: ScalarType::UInt8,
                quant_info: QuantizationInfo::AffineDequantize {
                    scale: vec![0xDD; 16],
                    zero_point: vec![0xEE; 16],
                    scale_dtype: ScalarType::Float16,
                    zero_point_dtype: ScalarType::Float16,
                    axis: Some(1),
                    bit_width: 4,
                    group_size: Some(128),
                    awq_scales: None,
                    g_idx: None,
                },
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
        assert_eq!(manifest.quantization.method, "mixed");
        assert_eq!(manifest.quantization.n_bits, 4);
        assert_eq!(manifest.tensors.len(), 3);

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
    fn write_affine_quantized_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.ironml-gpu");
        let provider = mock_provider();
        write_gpu_bundle(&provider, &bundle_path).unwrap();

        let base = "model_layers_0_mlp_gate_proj_weight";
        assert!(bundle_path.join(format!("weights/{base}.qdata")).exists());
        assert!(bundle_path.join(format!("weights/{base}.scale")).exists());
        assert!(bundle_path.join(format!("weights/{base}.zeros")).exists());

        assert_eq!(
            fs::read(bundle_path.join(format!("weights/{base}.qdata"))).unwrap(),
            vec![0xCC; 32]
        );
        assert_eq!(
            fs::read(bundle_path.join(format!("weights/{base}.scale"))).unwrap(),
            vec![0xDD; 16]
        );
        assert_eq!(
            fs::read(bundle_path.join(format!("weights/{base}.zeros"))).unwrap(),
            vec![0xEE; 16]
        );

        // Check manifest
        let json = fs::read_to_string(bundle_path.join("manifest.json")).unwrap();
        let manifest: GpuBundleManifest = serde_json::from_str(&json).unwrap();

        let desc = &manifest.tensors["model.layers.0.mlp.gate_proj.weight"];
        match desc {
            TensorManifest::AffineDequantize {
                quantized_data_file,
                scales_file,
                zeros_file,
                bit_width,
                group_size,
                axis,
                dtype,
                shape,
                ..
            } => {
                assert_eq!(
                    quantized_data_file,
                    "weights/model_layers_0_mlp_gate_proj_weight.qdata"
                );
                assert_eq!(
                    scales_file,
                    "weights/model_layers_0_mlp_gate_proj_weight.scale"
                );
                assert_eq!(
                    zeros_file,
                    "weights/model_layers_0_mlp_gate_proj_weight.zeros"
                );
                assert_eq!(*bit_width, 4);
                assert_eq!(*group_size, 128);
                assert_eq!(*axis, 1);
                assert_eq!(dtype, "uint8");
                assert_eq!(shape, &[2048, 2048]);
            }
            _ => panic!("expected AffineDequantize"),
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
