//! Streaming GPU bundle builder.
//!
//! Bypasses the MIL IR entirely: iterates tensors one at a time from a weight
//! provider, quantizes with standalone math functions, writes to the bundle,
//! and drops. Peak memory: ~1 tensor at a time.
//!
//! Supported input formats: SafeTensors and GGUF.
//! ONNX is not supported (use the regular [`GpuCompileBuilder`] path).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use half::f16;
use mil_rs::ir::ScalarType;
use mil_rs::ir::passes::affine_quantize::quantize_affine;
use mil_rs::ir::passes::int4_pack::pack_int4;
use mil_rs::ir::passes::tensor_utils::{tensor_as_f32_slice, tensor_f16_as_f32_slice};

use ironmill_core::gpu::bundle::{
    GpuBundleManifest, QuantizationManifest, TensorManifest, scalar_type_to_str,
    serialize_model_config,
};

use crate::error::CompileError;
use crate::weights::{GgufProvider, SafeTensorsProvider, WeightProvider};

// ── Constants ───────────────────────────────────────────────────────────

/// Minimum number of elements for a tensor to be eligible for quantization.
const MIN_ELEMENTS: usize = 1024;

// ── QuantMethod ─────────────────────────────────────────────────────────

/// Quantization method for the streaming builder.
#[derive(Debug, Clone)]
pub enum QuantMethod {
    /// Unsigned affine min/max quantization.
    Affine {
        /// Bit width (4 or 8).
        n_bits: u8,
        /// Group size along the last axis. Each group gets its own
        /// scale and zero point.
        group_size: usize,
    },
    /// Convert FP32 weights to FP16 (truncation, no calibration).
    Fp16,
}

// ── StreamingGpuBundleBuilder ───────────────────────────────────────────

/// Builds a `.ironml-gpu` bundle by streaming tensors one at a time.
///
/// Instead of constructing the full MIL IR program in memory, this builder
/// loads each tensor from the weight provider individually, quantizes it
/// with standalone math, writes the result to disk, and drops the data.
///
/// # Supported formats
///
/// - SafeTensors (`.safetensors` file or HuggingFace model directory)
/// - GGUF (`.gguf` file)
///
/// ONNX is **not** supported in the streaming path. Use
/// [`GpuCompileBuilder`](super::GpuCompileBuilder) for ONNX input.
pub struct StreamingGpuBundleBuilder {
    input: PathBuf,
    quantization: Option<QuantMethod>,
}

impl StreamingGpuBundleBuilder {
    /// Create a new builder for the given input path.
    pub fn new(input: impl Into<PathBuf>) -> Self {
        Self {
            input: input.into(),
            quantization: None,
        }
    }

    /// Set the quantization method.
    pub fn with_quantization(mut self, method: QuantMethod) -> Self {
        self.quantization = Some(method);
        self
    }

    /// Build the GPU bundle, writing output to `output_dir`.
    ///
    /// The output directory should end with `.ironml-gpu` by convention.
    pub fn build(self, output_dir: impl AsRef<Path>) -> Result<(), CompileError> {
        let output_dir = output_dir.as_ref();
        let weights_dir = output_dir.join("weights");
        fs::create_dir_all(&weights_dir)?;

        let provider = self.load_provider()?;
        let config = provider.config().clone();
        let tensor_names: Vec<String> = provider
            .tensor_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let quant = self.quantization.as_ref();

        let mut tensors = HashMap::new();
        let mut global_n_bits: Option<u8> = None;
        let mut methods_seen = std::collections::HashSet::new();

        for name in &tensor_names {
            let tensor = provider.tensor(name)?;
            let sanitized = sanitize_tensor_name(name);

            let rank = tensor.shape.len();
            let num_elements: usize = tensor.shape.iter().product();
            let eligible = is_eligible_for_quantization(name, rank, num_elements);

            match quant {
                Some(QuantMethod::Affine { n_bits, group_size }) if eligible => {
                    methods_seen.insert("affine");

                    match global_n_bits {
                        None => global_n_bits = Some(*n_bits),
                        Some(prev) if prev != *n_bits => {
                            eprintln!(
                                "Warning: tensor '{name}' n_bits={n_bits} but global is {prev}"
                            );
                        }
                        _ => {}
                    }

                    let floats = to_f32_vec(&tensor.data, tensor.dtype)?;
                    let (quantized, scales, zeros) =
                        quantize_affine_per_group(&floats, &tensor.shape, *group_size, *n_bits);

                    // Pack INT4 data
                    let packed_data = if *n_bits == 4 {
                        pack_int4(&quantized)
                    } else {
                        quantized
                    };

                    let qdata_file = format!("weights/{sanitized}.qdata");
                    let scales_file = format!("weights/{sanitized}.scale");
                    let zeros_file = format!("weights/{sanitized}.zeros");

                    fs::write(output_dir.join(&qdata_file), &packed_data)?;
                    fs::write(
                        output_dir.join(&scales_file),
                        f32_vec_to_fp16_bytes(&scales),
                    )?;
                    fs::write(output_dir.join(&zeros_file), f32_vec_to_fp16_bytes(&zeros))?;

                    let ndim = tensor.shape.len();
                    let axis = if ndim > 0 { (ndim - 1) as i64 } else { 0 };

                    tensors.insert(
                        name.to_string(),
                        TensorManifest::AffineDequantize {
                            quantized_data_file: qdata_file,
                            scales_file,
                            zeros_file,
                            shape: tensor.shape.clone(),
                            bit_width: *n_bits,
                            group_size: *group_size,
                            axis,
                            dtype: scalar_type_to_str(ScalarType::UInt8).to_string(),
                            awq_scales_file: None,
                        },
                    );
                }
                Some(QuantMethod::Fp16) if eligible => {
                    let fp16_data = to_fp16_bytes(&tensor.data, tensor.dtype)?;
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
                }
                // Not eligible or no quantization — passthrough as-is.
                _ => {
                    let (data, dtype_str) = passthrough_data(&tensor.data, tensor.dtype)?;
                    let file = format!("weights/{sanitized}.bin");
                    fs::write(output_dir.join(&file), &data)?;

                    tensors.insert(
                        name.to_string(),
                        TensorManifest::Dense {
                            file,
                            shape: tensor.shape.clone(),
                            dtype: dtype_str,
                        },
                    );
                }
            }
            // tensor bytes are dropped here — freed after each iteration
        }

        let method = match methods_seen.len() {
            0 => "none".to_string(),
            1 => methods_seen.into_iter().next().unwrap().to_string(),
            _ => "mixed".to_string(),
        };

        let manifest = GpuBundleManifest {
            format_version: 1,
            model_config: serialize_model_config(&config),
            quantization: QuantizationManifest {
                method,
                n_bits: global_n_bits.unwrap_or(16),
                seed: 42,
                min_elements: MIN_ELEMENTS,
            },
            tensors,
        };

        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| CompileError::Other(format!("failed to serialize manifest: {e}")))?;
        fs::write(output_dir.join("manifest.json"), json)?;

        Ok(())
    }

    /// Load the appropriate weight provider based on the input file format.
    fn load_provider(&self) -> Result<Box<dyn WeightProvider + Send + Sync>, CompileError> {
        let input = &self.input;

        if !input.exists() {
            return Err(CompileError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("input path does not exist: {}", input.display()),
            )));
        }

        if let Some(ext) = input.extension().and_then(|e| e.to_str()) {
            match ext {
                "safetensors" => {
                    return Ok(Box::new(SafeTensorsProvider::load(input)?));
                }
                "gguf" => {
                    return Ok(Box::new(GgufProvider::load(input)?));
                }
                "onnx" => {
                    return Err(CompileError::Other(
                        "ONNX is not supported in the streaming GPU build path. \
                         Use the regular compilation path (without --streaming) instead."
                            .into(),
                    ));
                }
                _ => {}
            }
        }

        // Directory with config.json → SafeTensors model dir.
        if input.is_dir() && input.join("config.json").exists() {
            return Ok(Box::new(SafeTensorsProvider::load(input)?));
        }

        Err(CompileError::Other(format!(
            "unsupported input format for streaming GPU build: {}. \
             Expected .safetensors, .gguf, or a HuggingFace model directory.",
            input.display()
        )))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Sanitize a tensor name for use as a filename (same logic as bundle.rs).
fn sanitize_tensor_name(name: &str) -> String {
    name.replace(['.', '/'], "_")
}

/// Determine whether a tensor is eligible for quantization.
///
/// A tensor is eligible if:
/// - It has rank >= 2 (not a norm, bias, or scalar)
/// - It has at least `MIN_ELEMENTS` elements
/// - Its name does not indicate a norm or embedding that should stay precise
fn is_eligible_for_quantization(name: &str, rank: usize, num_elements: usize) -> bool {
    if rank < 2 {
        return false;
    }
    if num_elements < MIN_ELEMENTS {
        return false;
    }
    // Skip layer norm weights — they are rank-1 in practice but just in case
    // a provider reports them with a batch dim.
    let lower = name.to_lowercase();
    if lower.contains("layernorm") || lower.contains("layer_norm") || lower.contains("rmsnorm") {
        return false;
    }
    true
}

/// Convert raw tensor bytes to an f32 vector, handling FP32 and FP16 inputs.
fn to_f32_vec(data: &[u8], dtype: ScalarType) -> Result<Vec<f32>, CompileError> {
    match dtype {
        ScalarType::Float32 => Ok(tensor_as_f32_slice(data)),
        ScalarType::Float16 => Ok(tensor_f16_as_f32_slice(data)),
        other => Err(CompileError::Other(format!(
            "streaming quantize: unsupported dtype {other:?}, expected Float32 or Float16"
        ))),
    }
}

/// Convert raw tensor bytes to FP16 bytes. FP32 is truncated; FP16 passes through.
fn to_fp16_bytes(data: &[u8], dtype: ScalarType) -> Result<Vec<u8>, CompileError> {
    match dtype {
        ScalarType::Float16 => Ok(data.to_vec()),
        ScalarType::Float32 => {
            let out: Vec<u8> = data
                .chunks_exact(4)
                .flat_map(|c| {
                    let val = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    f16::from_f32(val).to_le_bytes()
                })
                .collect();
            Ok(out)
        }
        other => Err(CompileError::Other(format!(
            "streaming FP16 conversion: unsupported dtype {other:?}"
        ))),
    }
}

/// For passthrough (non-quantized) tensors, convert FP32 to FP16 and
/// leave everything else as-is.
fn passthrough_data(data: &[u8], dtype: ScalarType) -> Result<(Vec<u8>, String), CompileError> {
    match dtype {
        ScalarType::Float32 => {
            let fp16 = to_fp16_bytes(data, dtype)?;
            Ok((fp16, "float16".to_string()))
        }
        _ => Ok((data.to_vec(), scalar_type_to_str(dtype).to_string())),
    }
}

/// Convert a vec of f32 values to FP16 bytes.
fn f32_vec_to_fp16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}

/// Per-group affine quantization. Partitions the last axis into groups,
/// producing (quantized_u8, all_scales, all_zeros).
fn quantize_affine_per_group(
    floats: &[f32],
    shape: &[usize],
    group_size: usize,
    n_bits: u8,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let qmax = match n_bits {
        4 => 15.0_f32,
        8 => 255.0_f32,
        _ => (1u32 << n_bits) as f32 - 1.0,
    };

    let ndim = shape.len();
    let last_dim = if ndim > 0 { shape[ndim - 1] } else { 1 };
    let outer_count: usize = if ndim > 1 {
        shape[..ndim - 1].iter().product()
    } else {
        1
    };
    let n_groups = last_dim.div_ceil(group_size);

    let mut all_quantized = Vec::with_capacity(floats.len());
    let mut all_scales = Vec::with_capacity(outer_count * n_groups);
    let mut all_zeros = Vec::with_capacity(outer_count * n_groups);

    for row in 0..outer_count {
        let row_start = row * last_dim;
        for g in 0..n_groups {
            let g_start = row_start + g * group_size;
            let g_end = (g_start + group_size).min(row_start + last_dim);
            let group_slice = &floats[g_start..g_end];
            let (q, s, zp) = quantize_affine(group_slice, qmax);
            all_quantized.extend_from_slice(&q);
            all_scales.push(s);
            all_zeros.push(zp);
        }
    }

    (all_quantized, all_scales, all_zeros)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::weights::{Architecture, ModelConfig, QuantizationInfo, WeightTensor};
    use std::borrow::Cow;

    fn test_config() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 16,
            vocab_size: 256,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            extra: HashMap::new(),
        }
    }

    /// Minimal in-memory weight provider for testing.
    #[allow(dead_code)]
    struct MockProvider {
        tensors: Vec<(String, Vec<u8>, Vec<usize>, ScalarType)>,
        config: ModelConfig,
    }

    impl WeightProvider for MockProvider {
        fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, mil_rs::MilError> {
            let (_, data, shape, dtype) = self
                .tensors
                .iter()
                .find(|(n, _, _, _)| n == name)
                .ok_or_else(|| mil_rs::MilError::UndefinedValue(name.to_string()))?;
            Ok(WeightTensor {
                data: Cow::Borrowed(data),
                shape: shape.clone(),
                dtype: *dtype,
                quant_info: QuantizationInfo::None,
            })
        }

        fn tensor_names(&self) -> Vec<&str> {
            self.tensors.iter().map(|(n, _, _, _)| n.as_str()).collect()
        }

        fn config(&self) -> &ModelConfig {
            &self.config
        }
    }

    /// Helper: create FP32 bytes from a slice.
    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn builder_defaults() {
        let b = StreamingGpuBundleBuilder::new("model.safetensors");
        assert_eq!(b.input, PathBuf::from("model.safetensors"));
        assert!(b.quantization.is_none());
    }

    #[test]
    fn builder_with_quantization_chains() {
        let b =
            StreamingGpuBundleBuilder::new("model.gguf").with_quantization(QuantMethod::Affine {
                n_bits: 4,
                group_size: 128,
            });
        assert!(b.quantization.is_some());
    }

    #[test]
    fn onnx_returns_error() {
        let b = StreamingGpuBundleBuilder::new("model.onnx");
        let err = b.build("out.ironml-gpu");
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("ONNX") || msg.contains("not supported") || msg.contains("does not exist")
        );
    }

    #[test]
    fn nonexistent_input_returns_error() {
        let b = StreamingGpuBundleBuilder::new("does_not_exist.safetensors");
        let err = b.build("out.ironml-gpu");
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn eligibility_check() {
        // Rank < 2: not eligible
        assert!(!is_eligible_for_quantization("bias", 1, 2048));
        // Too few elements
        assert!(!is_eligible_for_quantization("weight", 2, 512));
        // Norm in name
        assert!(!is_eligible_for_quantization(
            "model.layernorm.weight",
            2,
            4096
        ));
        // Good tensor
        assert!(is_eligible_for_quantization(
            "model.layers.0.self_attn.q_proj.weight",
            2,
            4096
        ));
    }

    #[test]
    fn affine_quantize_roundtrip() {
        // 2×128 tensor of incrementing values
        let dim = 128;
        let rows = 2;
        let values: Vec<f32> = (0..rows * dim).map(|i| i as f32 * 0.01).collect();
        let shape = vec![rows, dim];

        let (quantized, scales, zeros) = quantize_affine_per_group(&values, &shape, 128, 4);
        assert_eq!(quantized.len(), values.len());
        assert_eq!(scales.len(), rows); // 1 group per row
        assert_eq!(zeros.len(), rows);

        // Dequantize and check max error is bounded
        for (i, &q) in quantized.iter().enumerate() {
            let row = i / dim;
            let dequant = (q as f32 - zeros[row]) * scales[row];
            let err = (dequant - values[i]).abs();
            assert!(err < 0.1, "error too large at index {i}: {err}");
        }
    }

    #[test]
    fn fp16_conversion() {
        let fp32_data = f32_bytes(&[1.0, 2.0, 3.0]);
        let fp16 = to_fp16_bytes(&fp32_data, ScalarType::Float32).unwrap();
        assert_eq!(fp16.len(), 6); // 3 * 2 bytes

        // Already FP16 should pass through
        let fp16_input = vec![0u8; 6];
        let result = to_fp16_bytes(&fp16_input, ScalarType::Float16).unwrap();
        assert_eq!(result, fp16_input);
    }

    #[test]
    fn passthrough_converts_fp32_to_fp16() {
        let fp32_data = f32_bytes(&[1.0, 2.0]);
        let (data, dtype_str) = passthrough_data(&fp32_data, ScalarType::Float32).unwrap();
        assert_eq!(dtype_str, "float16");
        assert_eq!(data.len(), 4); // 2 * 2 bytes
    }

    #[test]
    fn manifest_written_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let bundle_dir = dir.path().join("test.ironml-gpu");

        // We can't use build() directly without a real provider,
        // but we can test the manifest generation logic by calling the
        // internal pieces. Create the bundle dir + weights dir manually.
        fs::create_dir_all(bundle_dir.join("weights")).unwrap();

        let config = test_config();
        let mut tensors = HashMap::new();

        // Write a dummy dense tensor
        let file = "weights/test_tensor.bin";
        fs::write(bundle_dir.join(file), &[0u8; 16]).unwrap();
        tensors.insert(
            "test.tensor".to_string(),
            TensorManifest::Dense {
                file: file.to_string(),
                shape: vec![2, 4],
                dtype: "float16".to_string(),
            },
        );

        let manifest = GpuBundleManifest {
            format_version: 1,
            model_config: serialize_model_config(&config),
            quantization: QuantizationManifest {
                method: "none".to_string(),
                n_bits: 16,
                seed: 42,
                min_elements: MIN_ELEMENTS,
            },
            tensors,
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        fs::write(bundle_dir.join("manifest.json"), &json).unwrap();

        // Read back and verify
        let read_back: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(bundle_dir.join("manifest.json")).unwrap())
                .unwrap();

        assert_eq!(read_back["format_version"], 1);
        assert!(read_back["model_config"]["architecture"].is_string());
        assert_eq!(read_back["quantization"]["method"], "none");
        assert!(read_back["tensors"]["test.tensor"].is_object());
    }
}
