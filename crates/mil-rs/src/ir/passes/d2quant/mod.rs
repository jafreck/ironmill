//! D2Quant dual-scale sub-4-bit quantization pass.
//!
//! Partitions each weight group into "normal" and "outlier" subsets and
//! computes separate affine scale/zero-point pairs for each partition.
//! This yields lower quantization error than a single-scale approach at
//! the same bit-width (2 or 3 bits).

pub mod dual_scale;

use super::tensor_utils::tensor_as_f32_slice;
use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::Value;

use dual_scale::{dual_scale_quantize, pack_2bit, pack_3bit, pack_mask};

/// D2Quant dual-scale quantization pass.
///
/// Walks every `const` op with an FP32 tensor, processes it in groups of
/// `group_size`, and replaces each with a `constexpr_dual_scale_dequantize`
/// op carrying packed quantized data plus per-group dual-scale parameters.
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
        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                if op.op_type != "const" {
                    continue;
                }

                // val may live in inputs or attributes (ONNX puts it in attrs).
                let in_inputs = matches!(
                    op.inputs.get("val"),
                    Some(Value::Tensor {
                        dtype: ScalarType::Float32,
                        ..
                    })
                );
                let in_attrs = !in_inputs
                    && matches!(
                        op.attributes.get("val"),
                        Some(Value::Tensor {
                            dtype: ScalarType::Float32,
                            ..
                        })
                    );

                if !in_inputs && !in_attrs {
                    continue;
                }

                let val = if in_inputs {
                    op.inputs.remove("val").unwrap()
                } else {
                    op.attributes.remove("val").unwrap()
                };

                if let Value::Tensor {
                    data,
                    shape,
                    dtype: _,
                } = val
                {
                    let floats = tensor_as_f32_slice(&data);
                    let total = floats.len();

                    // Accumulators across all groups.
                    let mut all_quantized_packed: Vec<u8> = Vec::new();
                    let mut all_normal_scale: Vec<f32> = Vec::new();
                    let mut all_normal_zero: Vec<f32> = Vec::new();
                    let mut all_outlier_scale: Vec<f32> = Vec::new();
                    let mut all_outlier_zero: Vec<f32> = Vec::new();
                    let mut all_mask_packed: Vec<u8> = Vec::new();

                    let num_groups = total.div_ceil(self.group_size);

                    for g in 0..num_groups {
                        let start = g * self.group_size;
                        let end = (start + self.group_size).min(total);
                        let group = &floats[start..end];

                        let (quantized, params) =
                            dual_scale_quantize(group, self.bits, self.outlier_threshold);

                        // Pack quantized values.
                        let packed = match self.bits {
                            2 => pack_2bit(&quantized),
                            3 => pack_3bit(&quantized),
                            _ => unreachable!(),
                        };
                        all_quantized_packed.extend_from_slice(&packed);

                        all_normal_scale.push(params.normal_scale);
                        all_normal_zero.push(params.normal_zero);
                        all_outlier_scale.push(params.outlier_scale);
                        all_outlier_zero.push(params.outlier_zero);
                        all_mask_packed.extend_from_slice(&pack_mask(&params.outlier_mask));
                    }

                    // Rewrite the op.
                    op.op_type = "constexpr_dual_scale_dequantize".to_string();
                    op.inputs.remove("val");
                    op.attributes.remove("val");

                    // Packed quantized data (raw bytes, shape = packed byte count).
                    op.attributes.insert(
                        "quantized_data".to_string(),
                        Value::Tensor {
                            data: all_quantized_packed.clone(),
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
                            data: f32_bytes(&all_normal_scale),
                            shape: vec![num_groups],
                            dtype: ScalarType::Float32,
                        },
                    );
                    op.attributes.insert(
                        "normal_zero".to_string(),
                        Value::Tensor {
                            data: f32_bytes(&all_normal_zero),
                            shape: vec![num_groups],
                            dtype: ScalarType::Float32,
                        },
                    );
                    op.attributes.insert(
                        "outlier_scale".to_string(),
                        Value::Tensor {
                            data: f32_bytes(&all_outlier_scale),
                            shape: vec![num_groups],
                            dtype: ScalarType::Float32,
                        },
                    );
                    op.attributes.insert(
                        "outlier_zero".to_string(),
                        Value::Tensor {
                            data: f32_bytes(&all_outlier_zero),
                            shape: vec![num_groups],
                            dtype: ScalarType::Float32,
                        },
                    );

                    // Packed outlier mask (1 bit per weight, shape = packed byte count).
                    op.attributes.insert(
                        "outlier_mask".to_string(),
                        Value::Tensor {
                            data: all_mask_packed.clone(),
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
        let weights: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let tensor_val = Value::Tensor {
            data: f32_bytes(&weights),
            shape: vec![16],
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
            data: vec![0u8; 16],
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
        let weights: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let tensor_val = Value::Tensor {
            data: f32_bytes(&weights),
            shape: vec![8],
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
        let weights: Vec<f32> = vec![0.0; 24];
        let tensor_val = Value::Tensor {
            data: f32_bytes(&weights),
            shape: vec![2, 3, 4],
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
        assert_eq!(out_type.shape, vec![Some(2), Some(3), Some(4)]);
    }
}
