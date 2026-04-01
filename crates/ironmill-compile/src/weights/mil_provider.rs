//! Weight provider backed by a compiled MIL IR [`Program`].
//!
//! [`MilWeightProvider`] walks the operations in a MIL program and extracts
//! quantized (and unquantized) weight tensors, exposing them by their original
//! HF-style names via the [`WeightProvider`] trait.

use std::borrow::Cow;
use std::collections::HashMap;

use mil_rs::{MilError, Operation, Program, ScalarType, Value};

use super::{ModelConfig, QuantizationInfo, WeightProvider, WeightTensor};

/// An extracted tensor together with its quantization metadata.
#[derive(Debug, Clone)]
struct ExtractedTensor {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: ScalarType,
    quant_info: QuantizationInfo,
}

/// A [`WeightProvider`] that reads weights from a compiled MIL IR [`Program`].
///
/// During construction every `constexpr_lut_to_dense`, `constexpr_affine_dequantize`,
/// and plain `const` op in the program's main function is inspected. The original
/// HF tensor name is recovered from the `onnx_name` attribute that the ONNX import
/// stage attaches to each initializer `const` op.
pub struct MilWeightProvider {
    tensors: HashMap<String, ExtractedTensor>,
    config: ModelConfig,
}

impl MilWeightProvider {
    /// Build a provider by walking the main function of `program`.
    pub fn new(program: &Program, config: ModelConfig) -> Result<Self, MilError> {
        let function = program
            .main()
            .ok_or_else(|| MilError::Validation("program has no main function".into()))?;

        let ops = &function.body.operations;

        // Pre-collect norms data keyed by the original output name prefix so we
        // can associate them with the corresponding constexpr_lut_to_dense op.
        // Norms ops are named `{original_output}_polar_norms` with a matching
        // output name.
        let mut norms_map: HashMap<String, (Vec<u8>, ScalarType)> = HashMap::new();
        for op in ops {
            if op.op_type != "const" {
                continue;
            }
            let output_name: &str = match op.outputs.first() {
                Some(n) => n.as_str(),
                None => continue,
            };
            if !output_name.ends_with("_polar_norms") {
                continue;
            }
            // Extract the tensor value from inputs or attributes.
            let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
            if let Some(Value::Tensor { data, dtype, .. }) = val {
                let prefix = &output_name[..output_name.len() - "_polar_norms".len()];
                norms_map.insert(prefix.to_string(), (data.clone(), *dtype));
            }
        }

        let mut tensors = HashMap::new();

        for op in ops {
            match op.op_type.as_str() {
                "constexpr_lut_to_dense" => {
                    let name = tensor_name_for_op(op);
                    let name = match name {
                        Some(n) => n,
                        None => continue,
                    };

                    let (lut_data, lut_dtype) = extract_tensor_attr(&op.attributes, "lut")?;
                    let (indices_data, indices_dtype) =
                        extract_tensor_attr(&op.attributes, "indices")?;
                    let (shape_bytes, _) = extract_tensor_attr(&op.attributes, "shape")?;

                    let original_shape = u32_bytes_to_usize_vec(&shape_bytes);

                    // Derive n_bits from the LUT element count.
                    let lut_elements = lut_data.len() / lut_dtype.byte_size();
                    let n_bits = (lut_elements as f64).log2() as u8;

                    // Look up the corresponding row norms.
                    let output_name: &str = op.outputs.first().map(String::as_str).unwrap_or("");
                    let (row_norms, norms_dtype) = norms_map
                        .get(output_name)
                        .cloned()
                        .unwrap_or_else(|| (Vec::new(), lut_dtype));

                    // The stored data is the packed indices (the primary payload
                    // consumers will unpack during GPU dispatch).
                    let extracted = ExtractedTensor {
                        data: indices_data.clone(),
                        shape: original_shape.clone(),
                        dtype: indices_dtype,
                        quant_info: QuantizationInfo::LutToDense {
                            lut: lut_data,
                            lut_dtype,
                            indices: indices_data,
                            original_shape,
                            n_bits,
                            row_norms,
                            norms_dtype,
                        },
                    };
                    tensors.insert(name, extracted);
                }

                "constexpr_affine_dequantize" => {
                    let name = tensor_name_for_op(op);
                    let name = match name {
                        Some(n) => n,
                        None => continue,
                    };

                    let (quantized_data, quantized_dtype) =
                        extract_tensor_attr(&op.attributes, "quantized_data")?;
                    let quantized_shape = extract_tensor_shape(&op.attributes, "quantized_data")?;

                    let (scale_data, scale_dtype) = extract_scale_or_zp(&op.attributes, "scale")?;
                    let (zp_data, zp_dtype) = extract_scale_or_zp(&op.attributes, "zero_point")?;

                    let axis = match op.attributes.get("axis") {
                        Some(Value::Int(v)) => Some(*v as usize),
                        _ => None,
                    };

                    let extracted = ExtractedTensor {
                        data: quantized_data,
                        shape: quantized_shape,
                        dtype: quantized_dtype,
                        quant_info: QuantizationInfo::AffineDequantize {
                            scale: scale_data,
                            zero_point: zp_data,
                            scale_dtype,
                            zero_point_dtype: zp_dtype,
                            axis,
                        },
                    };
                    tensors.insert(name, extracted);
                }

                "const" => {
                    // Skip norms ops — they are already consumed above.
                    let output_name: &str = op.outputs.first().map(String::as_str).unwrap_or("");
                    if output_name.ends_with("_polar_norms") {
                        continue;
                    }

                    let name = tensor_name_for_op(op);
                    let name = match name {
                        Some(n) => n,
                        None => continue,
                    };

                    let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
                    let val = match val {
                        Some(v) => v,
                        None => continue,
                    };

                    if let Value::Tensor { data, shape, dtype } = val {
                        let extracted = ExtractedTensor {
                            data: data.clone(),
                            shape: shape.clone(),
                            dtype: *dtype,
                            quant_info: QuantizationInfo::None,
                        };
                        tensors.insert(name, extracted);
                    }
                }

                _ => {}
            }
        }

        Ok(Self { tensors, config })
    }
}

impl WeightProvider for MilWeightProvider {
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

// ── Helpers ────────────────────────────────────────────────────────────────

/// Resolve the HF-style tensor name for an operation.
///
/// Prefers the `onnx_name` attribute set during ONNX import. Falls back to the
/// first output name when `onnx_name` is absent.
fn tensor_name_for_op(op: &Operation) -> Option<String> {
    if let Some(Value::String(n)) = op.attributes.get("onnx_name") {
        if !n.is_empty() {
            return Some(n.clone());
        }
    }
    // Fallback: use the MIL output name.
    op.outputs.first().cloned()
}

/// Extract a `Value::Tensor` attribute, returning its raw bytes and dtype.
fn extract_tensor_attr(
    attrs: &HashMap<String, Value>,
    key: &str,
) -> Result<(Vec<u8>, ScalarType), MilError> {
    match attrs.get(key) {
        Some(Value::Tensor { data, dtype, .. }) => Ok((data.clone(), *dtype)),
        _ => Err(MilError::UndefinedValue(format!(
            "missing or invalid tensor attribute '{key}'"
        ))),
    }
}

/// Extract the shape from a `Value::Tensor` attribute.
fn extract_tensor_shape(attrs: &HashMap<String, Value>, key: &str) -> Result<Vec<usize>, MilError> {
    match attrs.get(key) {
        Some(Value::Tensor { shape, .. }) => Ok(shape.clone()),
        _ => Err(MilError::UndefinedValue(format!(
            "missing or invalid tensor attribute '{key}'"
        ))),
    }
}

/// Extract scale or zero_point which may be a `Value::Float` (per-tensor) or
/// `Value::Tensor` (per-channel).
fn extract_scale_or_zp(
    attrs: &HashMap<String, Value>,
    key: &str,
) -> Result<(Vec<u8>, ScalarType), MilError> {
    match attrs.get(key) {
        Some(Value::Tensor { data, dtype, .. }) => Ok((data.clone(), *dtype)),
        Some(Value::Float(v)) => {
            let bytes = (*v as f32).to_le_bytes().to_vec();
            Ok((bytes, ScalarType::Float32))
        }
        _ => Err(MilError::UndefinedValue(format!(
            "missing or invalid attribute '{key}'"
        ))),
    }
}

/// Convert a byte buffer of little-endian `u32` values to `Vec<usize>`.
fn u32_bytes_to_usize_vec(bytes: &[u8]) -> Vec<usize> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::{Block, Function, Operation, Program, ScalarType, TensorType, Value};

    fn dummy_config() -> ModelConfig {
        ModelConfig {
            architecture: super::super::Architecture::Llama,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 32,
            vocab_size: 1000,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            extra: HashMap::new(),
        }
    }

    #[test]
    fn plain_const_extraction() {
        let data: Vec<u8> = vec![0u8; 16]; // 4 floats
        let mut op = Operation::new("const", "weight_0")
            .with_input(
                "val",
                Value::Tensor {
                    data: data.clone(),
                    shape: vec![2, 2],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output("weight_0_out");
        op.attributes.insert(
            "onnx_name".into(),
            Value::String("model.layers.0.weight".into()),
        );

        let mut func = Function::new("main");
        func.body.add_op(op);

        let mut program = Program::new("1.0.0");
        program.add_function(func);

        let provider = MilWeightProvider::new(&program, dummy_config()).unwrap();
        assert!(provider.has_tensor("model.layers.0.weight"));
        let t = provider.tensor("model.layers.0.weight").unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.dtype, ScalarType::Float32);
        assert!(matches!(t.quant_info, QuantizationInfo::None));
    }

    #[test]
    fn affine_dequantize_extraction() {
        let quantized = vec![1u8, 2, 3, 4, 5, 6];
        let scale = 0.5f32;
        let zp = 128.0f32;

        let mut op = Operation::new("constexpr_affine_dequantize", "aq_0");
        op.attributes.insert(
            "quantized_data".into(),
            Value::Tensor {
                data: quantized.clone(),
                shape: vec![2, 3],
                dtype: ScalarType::UInt8,
            },
        );
        op.attributes
            .insert("scale".into(), Value::Float(scale as f64));
        op.attributes
            .insert("zero_point".into(), Value::Float(zp as f64));
        op.attributes.insert("axis".into(), Value::Int(0));
        op.attributes.insert(
            "onnx_name".into(),
            Value::String("model.layers.0.q_proj.weight".into()),
        );
        op.outputs.push("aq_0_out".into());
        op.output_types
            .push(Some(TensorType::new(ScalarType::Float32, vec![2, 3])));

        let mut func = Function::new("main");
        func.body.add_op(op);

        let mut program = Program::new("1.0.0");
        program.add_function(func);

        let provider = MilWeightProvider::new(&program, dummy_config()).unwrap();
        let t = provider.tensor("model.layers.0.q_proj.weight").unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert!(matches!(
            t.quant_info,
            QuantizationInfo::AffineDequantize { axis: Some(0), .. }
        ));
    }

    #[test]
    fn lut_to_dense_extraction() {
        // Build a minimal constexpr_lut_to_dense op.
        let lut_data = vec![0u8; 4 * 4]; // 4 FP32 LUT entries → n_bits = 2
        let indices = vec![0u8; 8];
        let original_shape: Vec<u8> = [4u32, 8u32].iter().flat_map(|d| d.to_le_bytes()).collect();
        let norms_data = vec![0u8; 4 * 4]; // 4 FP32 norms

        let mut lut_op = Operation::new("constexpr_lut_to_dense", "pq_0");
        lut_op.attributes.insert(
            "lut".into(),
            Value::Tensor {
                data: lut_data.clone(),
                shape: vec![4],
                dtype: ScalarType::Float32,
            },
        );
        lut_op.attributes.insert(
            "indices".into(),
            Value::Tensor {
                data: indices.clone(),
                shape: vec![8],
                dtype: ScalarType::UInt8,
            },
        );
        lut_op.attributes.insert(
            "shape".into(),
            Value::Tensor {
                data: original_shape,
                shape: vec![2],
                dtype: ScalarType::UInt32,
            },
        );
        lut_op
            .attributes
            .insert("polar_quant_seed".into(), Value::Int(42));
        lut_op.attributes.insert(
            "onnx_name".into(),
            Value::String("model.embed.weight".into()),
        );
        lut_op.outputs.push("pq_0_out".into());
        lut_op
            .output_types
            .push(Some(TensorType::new(ScalarType::Float32, vec![4, 8])));

        // Norms const op.
        let mut norms_op = Operation::new("const", "pq_0_polar_norms")
            .with_input(
                "val",
                Value::Tensor {
                    data: norms_data.clone(),
                    shape: vec![4, 1],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output("pq_0_out_polar_norms");
        norms_op.output_types[0] = Some(TensorType::new(ScalarType::Float32, vec![4, 1]));

        let mut func = Function::new("main");
        func.body.add_op(lut_op);
        func.body.add_op(norms_op);

        let mut program = Program::new("1.0.0");
        program.add_function(func);

        let provider = MilWeightProvider::new(&program, dummy_config()).unwrap();
        let t = provider.tensor("model.embed.weight").unwrap();
        assert_eq!(t.shape, vec![4, 8]);
        match &t.quant_info {
            QuantizationInfo::LutToDense {
                n_bits,
                original_shape,
                row_norms,
                ..
            } => {
                assert_eq!(*n_bits, 2);
                assert_eq!(original_shape, &vec![4, 8]);
                assert_eq!(row_norms, &norms_data);
            }
            other => panic!("expected LutToDense, got {other:?}"),
        }
    }

    #[test]
    fn missing_tensor_returns_error() {
        let func = Function::new("main");
        let mut program = Program::new("1.0.0");
        program.add_function(func);

        let provider = MilWeightProvider::new(&program, dummy_config()).unwrap();
        assert!(provider.tensor("nonexistent").is_err());
    }

    #[test]
    fn fallback_to_output_name_when_no_onnx_name() {
        let data = vec![0u8; 4];
        let op = Operation::new("const", "some_const")
            .with_input(
                "val",
                Value::Tensor {
                    data: data.clone(),
                    shape: vec![1],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output("some_const_out");

        let mut func = Function::new("main");
        func.body.add_op(op);

        let mut program = Program::new("1.0.0");
        program.add_function(func);

        let provider = MilWeightProvider::new(&program, dummy_config()).unwrap();
        assert!(provider.has_tensor("some_const_out"));
    }
}
