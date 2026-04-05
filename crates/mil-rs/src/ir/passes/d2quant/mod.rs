//! D2Quant dual-scale sub-4-bit quantization pass.
//!
//! Partitions each weight group into "normal" and "outlier" subsets and
//! computes separate affine scale/zero-point pairs for each partition.
//! This yields lower quantization error than a single-scale approach at
//! the same bit-width (2 or 3 bits).

pub mod dac;
pub mod dual_scale;

use super::tensor_utils::{tensor_as_f32_slice, tensor_f16_as_f32_slice};
use crate::error::{MilError, Result};
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::{TensorData, Value};

use dual_scale::{dual_scale_quantize, pack_2bit, pack_3bit, pack_mask};

/// D2Quant dual-scale quantization pass.
///
/// Walks every `const` op with an FP32 tensor, processes it in groups of
/// `group_size`, and replaces each with a `constexpr_dual_scale_dequantize`
/// op carrying packed quantized data plus per-group dual-scale parameters.
#[non_exhaustive]
pub struct D2QuantPass {
    /// Bit-width: 2 or 3.
    pub bits: u8,
    /// Number of weights per group (typically 128).
    pub group_size: usize,
    /// Outlier percentile threshold (e.g. 0.99 → top 1 % are outliers).
    pub outlier_threshold: f32,
}

impl D2QuantPass {
    /// Create a pass with the given configuration.
    pub fn new(bits: u8, group_size: usize, outlier_threshold: f32) -> Self {
        assert!(bits == 2 || bits == 3, "bits must be 2 or 3");
        assert!(group_size > 0, "group_size must be > 0");
        Self {
            bits,
            group_size,
            outlier_threshold,
        }
    }

    /// Default 2-bit pass: group_size = 128, outlier_threshold = 0.99.
    pub fn two_bit() -> Self {
        Self::new(2, 128, 0.99)
    }

    /// Default 3-bit pass: group_size = 128, outlier_threshold = 0.99.
    pub fn three_bit() -> Self {
        Self::new(3, 128, 0.99)
    }
}

impl Pass for D2QuantPass {
    fn name(&self) -> &str {
        "d2quant"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        let provider = program.weight_provider.clone();
        let spill_index = program.spill_index.clone();
        let resolve = super::util::make_resolver(&provider, &spill_index);

        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                if op.op_type != "const" {
                    continue;
                }

                // val may live in inputs or attributes (ONNX puts it in attrs).
                // Accept Float32 and Float16 weight tensors.
                let is_quantizable_dtype = |v: Option<&Value>| -> bool {
                    matches!(
                        v,
                        Some(Value::Tensor {
                            dtype: ScalarType::Float32 | ScalarType::Float16,
                            ..
                        })
                    )
                };
                let in_inputs = is_quantizable_dtype(op.inputs.get("val"));
                let in_attrs = !in_inputs && is_quantizable_dtype(op.attributes.get("val"));

                if !in_inputs && !in_attrs {
                    continue;
                }

                let val = if in_inputs {
                    op.inputs
                        .remove("val")
                        .ok_or_else(|| MilError::Validation("missing val in inputs".into()))?
                } else {
                    op.attributes
                        .remove("val")
                        .ok_or_else(|| MilError::Validation("missing val in attributes".into()))?
                };

                if let Value::Tensor {
                    mut data,
                    shape,
                    dtype,
                } = val
                {
                    data.materialize_with(|key| resolve(key))?;
                    let raw = data.as_bytes().expect("tensor not materialized");

                    // Only quantize 2D weight matrices with sufficient size.
                    // Skip 1D vectors (norms, biases), embeddings loaded as
                    // dense, and very small tensors.
                    let total_elements: usize = shape.iter().product();
                    let is_weight_matrix = shape.len() == 2
                        && shape[0] >= 64
                        && shape[1] >= 64
                        && total_elements >= 4096;
                    if !is_weight_matrix {
                        // Re-insert the tensor and skip quantization.
                        let val = Value::Tensor { data, shape, dtype };
                        if in_inputs {
                            op.inputs.insert("val".to_string(), val);
                        } else {
                            op.attributes.insert("val".to_string(), val);
                        }
                        continue;
                    }

                    let floats = match dtype {
                        ScalarType::Float32 => tensor_as_f32_slice(raw),
                        ScalarType::Float16 => tensor_f16_as_f32_slice(raw),
                        _ => unreachable!("dtype already checked"),
                    };

                    // Partition along the last dimension per row so groups
                    // never span row (output-channel) boundaries.
                    let ndim = shape.len();
                    let last_dim = if ndim > 0 {
                        *shape
                            .last()
                            .ok_or_else(|| MilError::Validation("empty shape".into()))?
                    } else {
                        floats.len()
                    };
                    let outer_count: usize = if ndim > 1 {
                        shape[..ndim - 1].iter().product()
                    } else {
                        1
                    };
                    let n_groups_per_row = last_dim.div_ceil(self.group_size);
                    let total_groups = outer_count * n_groups_per_row;

                    // Accumulators across all groups.
                    let mut all_quantized_packed: Vec<u8> = Vec::new();
                    let mut all_normal_scale: Vec<f32> = Vec::new();
                    let mut all_normal_zero: Vec<f32> = Vec::new();
                    let mut all_outlier_scale: Vec<f32> = Vec::new();
                    let mut all_outlier_zero: Vec<f32> = Vec::new();
                    let mut all_mask_packed: Vec<u8> = Vec::new();

                    for row in 0..outer_count {
                        let row_start = row * last_dim;
                        for g in 0..n_groups_per_row {
                            let g_start = row_start + g * self.group_size;
                            let g_end = (g_start + self.group_size).min(row_start + last_dim);
                            let group = &floats[g_start..g_end];

                            let (quantized, params) =
                                dual_scale_quantize(group, self.bits, self.outlier_threshold);

                            // Pack quantized values.
                            let packed = match self.bits {
                                2 => pack_2bit(&quantized),
                                3 => pack_3bit(&quantized),
                                other => {
                                    return Err(MilError::Validation(format!(
                                        "unsupported bit width: {other}"
                                    )));
                                }
                            };
                            all_quantized_packed.extend_from_slice(&packed);

                            all_normal_scale.push(params.normal_scale);
                            all_normal_zero.push(params.normal_zero);
                            all_outlier_scale.push(params.outlier_scale);
                            all_outlier_zero.push(params.outlier_zero);
                            all_mask_packed.extend_from_slice(&pack_mask(&params.outlier_mask));
                        }
                    }

                    // Rewrite the op.
                    op.op_type = "constexpr_dual_scale_dequantize".to_string();
                    op.inputs.remove("val");
                    op.attributes.remove("val");

                    // Packed quantized data (raw bytes, shape = packed byte count).
                    op.attributes.insert(
                        "quantized_data".to_string(),
                        Value::Tensor {
                            data: TensorData::Inline(all_quantized_packed.clone()),
                            shape: vec![all_quantized_packed.len()],
                            dtype: ScalarType::UInt8,
                        },
                    );

                    // Per-group params stored as f32 tensors.
                    let f32_bytes = |vals: &[f32]| -> Vec<u8> {
                        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
                    };

                    op.attributes.insert(
                        "normal_scale".to_string(),
                        Value::Tensor {
                            data: TensorData::Inline(f32_bytes(&all_normal_scale)),
                            shape: vec![total_groups],
                            dtype: ScalarType::Float32,
                        },
                    );
                    op.attributes.insert(
                        "normal_zero".to_string(),
                        Value::Tensor {
                            data: TensorData::Inline(f32_bytes(&all_normal_zero)),
                            shape: vec![total_groups],
                            dtype: ScalarType::Float32,
                        },
                    );
                    op.attributes.insert(
                        "outlier_scale".to_string(),
                        Value::Tensor {
                            data: TensorData::Inline(f32_bytes(&all_outlier_scale)),
                            shape: vec![total_groups],
                            dtype: ScalarType::Float32,
                        },
                    );
                    op.attributes.insert(
                        "outlier_zero".to_string(),
                        Value::Tensor {
                            data: TensorData::Inline(f32_bytes(&all_outlier_zero)),
                            shape: vec![total_groups],
                            dtype: ScalarType::Float32,
                        },
                    );

                    // Packed outlier mask (1 bit per weight, shape = packed byte count).
                    op.attributes.insert(
                        "outlier_mask".to_string(),
                        Value::Tensor {
                            data: TensorData::Inline(all_mask_packed.clone()),
                            shape: vec![all_mask_packed.len()],
                            dtype: ScalarType::UInt8,
                        },
                    );

                    op.attributes
                        .insert("group_size".to_string(), Value::Int(self.group_size as i64));
                    op.attributes
                        .insert("bit_width".to_string(), Value::Int(self.bits as i64));

                    // Output type: the op dequantizes back to FP32 with the
                    // original shape.
                    let out_type = TensorType::new(ScalarType::Float32, shape);
                    if let Some(slot) = op.output_types.get_mut(0) {
                        *slot = Some(out_type);
                    } else {
                        op.output_types.push(Some(out_type));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::Function;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn const_tensor_op(name: &str, output: &str, value: Value) -> Operation {
        Operation::new("const", name)
            .with_input("val", value)
            .with_output(output)
    }

    #[test]
    fn pass_rewrites_const_to_dual_scale_dequantize() {
        let n = 64;
        let k = 128;
        let weights: Vec<f32> = (0..n * k).map(|i| (i as f32 - 4096.0) * 0.001).collect();
        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![n, k],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        D2QuantPass::new(2, 8, 0.99).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_dual_scale_dequantize");

        // Check required attributes are present.
        for key in &[
            "quantized_data",
            "normal_scale",
            "normal_zero",
            "outlier_scale",
            "outlier_zero",
            "outlier_mask",
            "group_size",
            "bit_width",
        ] {
            assert!(
                op.attributes.contains_key(*key),
                "missing attribute '{key}'"
            );
        }

        assert_eq!(op.attributes.get("group_size"), Some(&Value::Int(8)));
        assert_eq!(op.attributes.get("bit_width"), Some(&Value::Int(2)));
    }

    #[test]
    fn pass_preserves_non_const_ops() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("relu_out"),
        );
        func.body.outputs.push("relu_out".into());
        program.add_function(func);

        D2QuantPass::two_bit().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "relu");
    }

    #[test]
    fn pass_skips_non_fp32_tensors() {
        let tensor_val = Value::Tensor {
            data: TensorData::Inline(vec![0u8; 16]),
            shape: vec![16],
            dtype: ScalarType::UInt8,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op("w", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        D2QuantPass::two_bit().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "const", "non-FP32 const should be untouched");
    }

    #[test]
    fn pass_handles_val_in_attributes() {
        let n = 64;
        let k = 64;
        let weights: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.001).collect();
        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![n, k],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        let op = Operation::new("const", "w")
            .with_attr("val", tensor_val)
            .with_output("w_out");
        func.body.add_op(op);
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        D2QuantPass::three_bit().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_dual_scale_dequantize");
        assert_eq!(op.attributes.get("bit_width"), Some(&Value::Int(3)));
    }

    #[test]
    fn output_type_is_fp32_with_original_shape() {
        let n = 64;
        let k = 128;
        let weights: Vec<f32> = vec![0.0; n * k];
        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![n, k],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op("w", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        D2QuantPass::two_bit().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        let out_type = op.output_types[0].as_ref().expect("output type set");
        assert_eq!(out_type.scalar_type, ScalarType::Float32);
        assert_eq!(out_type.shape, vec![Some(n), Some(k)]);
    }
}
