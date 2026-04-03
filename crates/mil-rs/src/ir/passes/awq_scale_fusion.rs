//! AWQ scale fusion pass.
//!
//! After AWQ quantization, each `constexpr_affine_dequantize` op carries an
//! `"awq_channel_scales"` attribute — per-channel factors that must be applied
//! to activations at inference time to compensate for the weight scaling
//! (weights were divided by `s[c]`, so activations must be multiplied by `s[c]`).
//!
//! This pass eliminates that runtime cost in one of two ways:
//!
//! 1. **Fusion into LayerNorm / RMSNorm gamma** — if a norm op immediately
//!    precedes the linear op that consumes the dequantized weight, its gamma
//!    parameter is updated: `gamma_new[c] = gamma_old[c] * s[c]`.
//!    This has zero runtime cost.
//!
//! 2. **Explicit `mul` op** — when no fuseable norm is found, a `mul` op is
//!    inserted on the activation path before the linear/matmul op.

use super::tensor_utils::{f32_slice_to_bytes, tensor_as_f32_slice, tensor_f16_as_f32_slice};
use super::util::build_consumer_map;
use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::Value;

/// Fuses AWQ per-channel scaling factors into adjacent LayerNorm weights.
///
/// AWQ scales weights by 1/s[c] per channel, so activations must be scaled
/// by s[c] to compensate. If a LayerNorm immediately precedes the quantized
/// linear op, we can absorb the scale into LayerNorm's gamma parameter:
///   gamma_new[c] = gamma_old[c] * s[c]
/// This has zero runtime cost.
///
/// If no fuseable LayerNorm is found, emit an explicit element-wise mul op.
pub struct AwqScaleFusionPass;

impl Pass for AwqScaleFusionPass {
    fn name(&self) -> &str {
        "awq-scale-fusion"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            fuse_awq_scales(&mut function.body);
        }
        Ok(())
    }
}

/// Norm op types whose gamma parameter can absorb AWQ channel scales.
const FUSEABLE_NORM_OPS: &[&str] = &["layer_norm", "rms_norm"];

/// Walk a block and fuse or materialise AWQ channel scales.
fn fuse_awq_scales(block: &mut crate::ir::program::Block) {
    // We collect all the edits to apply, then apply them in a second pass to
    // avoid mutating the vec while iterating.

    // Phase 1: identify constexpr_affine_dequantize ops with AWQ scales and
    // their consuming linear/matmul ops.
    let consumer_map = build_consumer_map(&block.operations);

    struct FusionTarget {
        /// Index of the constexpr_affine_dequantize op carrying the scales.
        dequant_idx: usize,
        /// The AWQ channel scales vector.
        scales: Vec<f32>,
        /// Index of the linear/matmul op consuming the dequantized weight.
        linear_idx: usize,
        /// The input key on the linear op that takes the activation ("x" for
        /// linear, "x" or "y" for matmul).
        activation_input_key: String,
    }

    let mut targets: Vec<FusionTarget> = Vec::new();

    for (i, op) in block.operations.iter().enumerate() {
        if op.op_type != "constexpr_affine_dequantize" {
            continue;
        }
        let scales_val = match op.attributes.get("awq_channel_scales") {
            Some(v) => v,
            None => continue,
        };
        let scales = match scales_val {
            Value::Tensor {
                data,
                dtype: ScalarType::Float32,
                ..
            } => tensor_as_f32_slice(data),
            _ => continue,
        };

        // Find the consuming linear/matmul op.
        let dequant_output = match op.outputs.first() {
            Some(name) => name.clone(),
            None => continue,
        };

        let consumers = match consumer_map.get(&dequant_output) {
            Some(c) => c,
            None => continue,
        };

        for (consumer_idx, input_key) in consumers {
            let consumer_op = &block.operations[*consumer_idx];
            let activation_key = match consumer_op.op_type.as_str() {
                "linear" if input_key == "weight" => "x".to_string(),
                "matmul" if input_key == "y" => "x".to_string(),
                "matmul" if input_key == "x" => "y".to_string(),
                _ => continue,
            };

            targets.push(FusionTarget {
                dequant_idx: i,
                scales: scales.clone(),
                linear_idx: *consumer_idx,
                activation_input_key: activation_key,
            });
        }
    }

    // Phase 2: for each target, try to fuse into a preceding norm or insert mul.
    // We must track insertions that shift indices.
    let mut inserted: usize = 0;
    let mut processed_dequant_indices = Vec::new();

    for target in &targets {
        let linear_idx = target.linear_idx + inserted;
        let dequant_idx = target.dequant_idx + inserted;

        let linear_op = &block.operations[linear_idx];

        // Find the activation input reference.
        let act_ref = match linear_op.inputs.get(&target.activation_input_key) {
            Some(Value::Reference(name)) => name.clone(),
            _ => continue,
        };

        // Look for a preceding norm op that produces `act_ref`.
        let norm_idx = block.operations.iter().enumerate().find_map(|(idx, op)| {
            if FUSEABLE_NORM_OPS.contains(&op.op_type.as_str())
                && op.outputs.first().map(|s| s.as_str()) == Some(act_ref.as_str())
            {
                Some(idx)
            } else {
                None
            }
        });

        if let Some(norm_idx) = norm_idx {
            // Fuse: multiply gamma/weight by channel scales.
            let norm_op = &mut block.operations[norm_idx];
            let mut fused = false;

            // Template-generated norms use "weight", ONNX-converted use "gamma".
            let gamma_key = if norm_op.inputs.contains_key("weight") {
                "weight"
            } else {
                "gamma"
            };

            // Look for gamma/weight in inputs first (as a reference to a const op),
            // then in attributes.
            if let Some(Value::Reference(gamma_ref)) = norm_op.inputs.get(gamma_key).cloned() {
                // gamma is provided by a const op — find and update it.
                if let Some(gamma_op) = block
                    .operations
                    .iter_mut()
                    .find(|op| op.outputs.first().map(|s| s.as_str()) == Some(gamma_ref.as_str()))
                {
                    // The const op stores its value in inputs["val"] or attributes["val"].
                    let gamma_val = gamma_op
                        .inputs
                        .get("val")
                        .or_else(|| gamma_op.attributes.get("val"));
                    if let Some(Value::Tensor {
                        data, shape, dtype, ..
                    }) = gamma_val
                    {
                        let gamma_floats_opt = match *dtype {
                            ScalarType::Float32 => Some(tensor_as_f32_slice(data)),
                            ScalarType::Float16 => Some(tensor_f16_as_f32_slice(data)),
                            _ => None,
                        };
                        if let Some(mut gamma_floats) = gamma_floats_opt {
                            let n = gamma_floats.len().min(target.scales.len());
                            for (c, g) in gamma_floats.iter_mut().enumerate().take(n) {
                                *g *= target.scales[c];
                            }
                            // Store in original dtype to preserve compatibility.
                            let new_data = match *dtype {
                                ScalarType::Float16 => gamma_floats
                                    .iter()
                                    .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                                    .collect(),
                                _ => f32_slice_to_bytes(&gamma_floats),
                            };
                            let new_val = Value::Tensor {
                                data: new_data,
                                shape: shape.clone(),
                                dtype: *dtype,
                            };
                            if gamma_op.inputs.contains_key("val") {
                                gamma_op.inputs.insert("val".to_string(), new_val);
                            } else {
                                gamma_op.attributes.insert("val".to_string(), new_val);
                            }
                            fused = true;
                        }
                    }
                }
            } else if let Some(Value::Tensor {
                data, shape, dtype, ..
            }) = norm_op.inputs.get(gamma_key).cloned()
            {
                // gamma stored inline as a tensor value.
                let gamma_floats_opt = match dtype {
                    ScalarType::Float32 => Some(tensor_as_f32_slice(&data)),
                    ScalarType::Float16 => Some(tensor_f16_as_f32_slice(&data)),
                    _ => None,
                };
                if let Some(mut gamma_floats) = gamma_floats_opt {
                    let n = gamma_floats.len().min(target.scales.len());
                    for (c, g) in gamma_floats.iter_mut().enumerate().take(n) {
                        *g *= target.scales[c];
                    }
                    let new_data = match dtype {
                        ScalarType::Float16 => gamma_floats
                            .iter()
                            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                            .collect(),
                        _ => f32_slice_to_bytes(&gamma_floats),
                    };
                    norm_op.inputs.insert(
                        gamma_key.to_string(),
                        Value::Tensor {
                            data: new_data,
                            shape,
                            dtype,
                        },
                    );
                    fused = true;
                }
            }

            if fused {
                // Remove awq_channel_scales from the dequant op.
                processed_dequant_indices.push(dequant_idx);
            }
            // If fusion failed (gamma missing/wrong dtype), scales remain on
            // the op — a subsequent pass or manual intervention can handle it.
        } else {
            // No fuseable norm found — insert an explicit mul op.
            let num_channels = target.scales.len();
            let mul_name = format!("awq_scale_mul_{}", target.dequant_idx);
            let scale_const_name = format!("awq_scale_const_{}", target.dequant_idx);
            let scale_const_output = format!("{}_out", scale_const_name);
            let mul_output = format!("{}_out", mul_name);

            // Create the const op for the scale tensor.
            let scale_const_op = crate::ir::operation::Operation::new("const", &scale_const_name)
                .with_input(
                    "val",
                    Value::Tensor {
                        data: f32_slice_to_bytes(&target.scales),
                        shape: vec![num_channels],
                        dtype: ScalarType::Float32,
                    },
                )
                .with_output(&scale_const_output);

            // Create the mul op: output = activation * scales.
            let mul_out_type = TensorType::new(ScalarType::Float32, vec![num_channels]);
            let mut mul_op = crate::ir::operation::Operation::new("mul", &mul_name)
                .with_input("x", Value::Reference(act_ref.clone()))
                .with_input("y", Value::Reference(scale_const_output.clone()))
                .with_output(&mul_output);
            mul_op.output_types = vec![Some(mul_out_type)];

            // Insert const + mul just before the linear op.
            block.operations.insert(linear_idx, mul_op);
            block.operations.insert(linear_idx, scale_const_op);

            // Rewire the linear op's activation input to point at the mul output.
            let shifted_linear_idx = linear_idx + 2;
            block.operations[shifted_linear_idx].inputs.insert(
                target.activation_input_key.clone(),
                Value::Reference(mul_output),
            );

            inserted += 2;

            // Remove awq_channel_scales from the dequant op.
            let shifted_dequant_idx = dequant_idx;
            processed_dequant_indices.push(shifted_dequant_idx);
        }
    }

    // Phase 3: remove awq_channel_scales attributes from processed dequant ops.
    for idx in processed_dequant_indices {
        block.operations[idx]
            .attributes
            .remove("awq_channel_scales");
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::{Block, Function};

    /// Create FP32 tensor bytes from a slice.
    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Helper: build a `const` op.
    fn const_op(name: &str, output: &str, values: &[f32]) -> Operation {
        Operation::new("const", name)
            .with_input(
                "val",
                Value::Tensor {
                    data: f32_bytes(values),
                    shape: vec![values.len()],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output(output)
    }

    /// Helper: build a constexpr_affine_dequantize op with AWQ channel scales.
    fn dequant_op_with_scales(name: &str, output: &str, channel_scales: &[f32]) -> Operation {
        Operation::new("constexpr_affine_dequantize", name)
            .with_attr(
                "quantized_data",
                Value::Tensor {
                    data: vec![0u8; channel_scales.len() * 4],
                    shape: vec![channel_scales.len(), 4],
                    dtype: ScalarType::Float32,
                },
            )
            .with_attr("scale", Value::Float(1.0))
            .with_attr("zero_point", Value::Float(0.0))
            .with_attr("axis", Value::Int(0))
            .with_attr("group_size", Value::Int(4))
            .with_attr("bit_width", Value::Int(4))
            .with_attr(
                "awq_channel_scales",
                Value::Tensor {
                    data: f32_bytes(channel_scales),
                    shape: vec![channel_scales.len()],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output(output)
    }

    /// Wrap a block in a single-function program.
    fn program_with_block(block: Block) -> Program {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body = block;
        program.add_function(func);
        program
    }

    /// Get a ref to the main block's ops.
    fn block_ops(program: &Program) -> &[Operation] {
        &program.functions["main"].body.operations
    }

    // -----------------------------------------------------------------------
    // Fusion into LayerNorm gamma
    // -----------------------------------------------------------------------

    #[test]
    fn fuses_scales_into_layer_norm_gamma() {
        // gamma_const -> layer_norm -> linear <- dequant(awq_scales)
        let gamma_values = [1.0_f32, 2.0, 3.0, 4.0];
        let channel_scales = [2.0_f32, 0.5, 1.0, 3.0];

        let mut block = Block::new();
        block.add_op(const_op("gamma_const", "gamma_out", &gamma_values));
        block.add_op(
            Operation::new("layer_norm", "ln_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("gamma", Value::Reference("gamma_out".into()))
                .with_output("ln_out"),
        );
        block.add_op(dequant_op_with_scales(
            "dequant_0",
            "dequant_out",
            &channel_scales,
        ));
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("ln_out".into()))
                .with_input("weight", Value::Reference("dequant_out".into()))
                .with_output("linear_out"),
        );
        block.outputs.push("linear_out".into());

        let mut program = program_with_block(block);
        AwqScaleFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);

        // awq_channel_scales should be removed from the dequant op.
        let dequant = ops
            .iter()
            .find(|op| op.op_type == "constexpr_affine_dequantize")
            .unwrap();
        assert!(
            dequant.attributes.get("awq_channel_scales").is_none(),
            "awq_channel_scales should be removed after fusion"
        );

        // gamma_const should have been updated: gamma_new[c] = gamma_old[c] * s[c].
        let gamma_op = ops.iter().find(|op| op.name == "gamma_const").unwrap();
        let updated_gamma = match gamma_op.inputs.get("val") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data),
            _ => panic!("gamma_const should still have a tensor val"),
        };
        let expected: Vec<f32> = gamma_values
            .iter()
            .zip(channel_scales.iter())
            .map(|(g, s)| g * s)
            .collect();
        assert_eq!(updated_gamma, expected);

        // No mul ops should have been inserted.
        assert!(
            !ops.iter().any(|op| op.op_type == "mul"),
            "no mul op should be inserted when fusion succeeds"
        );
    }

    // -----------------------------------------------------------------------
    // Fusion into RMSNorm gamma
    // -----------------------------------------------------------------------

    #[test]
    fn fuses_scales_into_rms_norm_gamma() {
        let gamma_values = [1.0_f32, 1.0, 1.0];
        let channel_scales = [0.5_f32, 2.0, 1.5];

        let mut block = Block::new();
        block.add_op(const_op("gamma_const", "gamma_out", &gamma_values));
        block.add_op(
            Operation::new("rms_norm", "rmsn_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("gamma", Value::Reference("gamma_out".into()))
                .with_output("rmsn_out"),
        );
        block.add_op(dequant_op_with_scales(
            "dequant_0",
            "dequant_out",
            &channel_scales,
        ));
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("rmsn_out".into()))
                .with_input("weight", Value::Reference("dequant_out".into()))
                .with_output("linear_out"),
        );
        block.outputs.push("linear_out".into());

        let mut program = program_with_block(block);
        AwqScaleFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        let dequant = ops
            .iter()
            .find(|op| op.op_type == "constexpr_affine_dequantize")
            .unwrap();
        assert!(dequant.attributes.get("awq_channel_scales").is_none());

        let gamma_op = ops.iter().find(|op| op.name == "gamma_const").unwrap();
        let updated_gamma = match gamma_op.inputs.get("val") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data),
            _ => panic!("expected tensor"),
        };
        assert_eq!(updated_gamma, vec![0.5, 2.0, 1.5]);
    }

    // -----------------------------------------------------------------------
    // Fallback: insert explicit mul when no norm is found
    // -----------------------------------------------------------------------

    #[test]
    fn inserts_mul_when_no_norm_present() {
        // input -> linear <- dequant(awq_scales)
        let channel_scales = [2.0_f32, 3.0, 4.0];

        let mut block = Block::new();
        block.add_op(dequant_op_with_scales(
            "dequant_0",
            "dequant_out",
            &channel_scales,
        ));
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("weight", Value::Reference("dequant_out".into()))
                .with_output("linear_out"),
        );
        block.outputs.push("linear_out".into());

        let mut program = program_with_block(block);
        AwqScaleFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);

        // awq_channel_scales removed.
        let dequant = ops
            .iter()
            .find(|op| op.op_type == "constexpr_affine_dequantize")
            .unwrap();
        assert!(dequant.attributes.get("awq_channel_scales").is_none());

        // A const + mul pair should have been inserted.
        let mul_op = ops.iter().find(|op| op.op_type == "mul").unwrap();
        // mul's x input should be the original activation ref.
        assert_eq!(
            mul_op.inputs.get("x"),
            Some(&Value::Reference("input".into()))
        );

        // The mul's y input should reference a new const holding the scales.
        let scale_ref = match mul_op.inputs.get("y") {
            Some(Value::Reference(name)) => name.clone(),
            _ => panic!("mul y should be a reference"),
        };
        let scale_const = ops
            .iter()
            .find(|op| op.outputs.first().map(|s| s.as_str()) == Some(scale_ref.as_str()))
            .unwrap();
        let scale_vals = match scale_const.inputs.get("val") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data),
            _ => panic!("scale const should have tensor val"),
        };
        assert_eq!(scale_vals, channel_scales.to_vec());

        // The linear op's x input should now point to the mul output.
        let linear = ops.iter().find(|op| op.op_type == "linear").unwrap();
        let mul_output = mul_op.outputs.first().unwrap();
        assert_eq!(
            linear.inputs.get("x"),
            Some(&Value::Reference(mul_output.clone()))
        );
    }

    // -----------------------------------------------------------------------
    // No AWQ scales → pass is a no-op
    // -----------------------------------------------------------------------

    #[test]
    fn no_op_without_awq_scales() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("constexpr_affine_dequantize", "dequant_0")
                .with_attr(
                    "quantized_data",
                    Value::Tensor {
                        data: vec![0u8; 16],
                        shape: vec![4, 4],
                        dtype: ScalarType::Float32,
                    },
                )
                .with_attr("scale", Value::Float(1.0))
                .with_attr("zero_point", Value::Float(0.0))
                .with_attr("axis", Value::Int(0))
                .with_output("dequant_out"),
        );
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("weight", Value::Reference("dequant_out".into()))
                .with_output("linear_out"),
        );
        block.outputs.push("linear_out".into());

        let mut program = program_with_block(block);
        let ops_before = block_ops(&program).len();
        AwqScaleFusionPass.run(&mut program).unwrap();
        assert_eq!(block_ops(&program).len(), ops_before);
    }

    // -----------------------------------------------------------------------
    // Matmul consumer (weight on "y" input)
    // -----------------------------------------------------------------------

    #[test]
    fn inserts_mul_for_matmul_consumer() {
        let channel_scales = [1.5_f32, 2.5];

        let mut block = Block::new();
        block.add_op(dequant_op_with_scales(
            "dequant_0",
            "dequant_out",
            &channel_scales,
        ));
        block.add_op(
            Operation::new("matmul", "mm_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("y", Value::Reference("dequant_out".into()))
                .with_output("mm_out"),
        );
        block.outputs.push("mm_out".into());

        let mut program = program_with_block(block);
        AwqScaleFusionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        let dequant = ops
            .iter()
            .find(|op| op.op_type == "constexpr_affine_dequantize")
            .unwrap();
        assert!(dequant.attributes.get("awq_channel_scales").is_none());

        let mul_op = ops.iter().find(|op| op.op_type == "mul").unwrap();
        assert_eq!(
            mul_op.inputs.get("x"),
            Some(&Value::Reference("input".into()))
        );

        let matmul = ops.iter().find(|op| op.op_type == "matmul").unwrap();
        let mul_output = mul_op.outputs.first().unwrap();
        assert_eq!(
            matmul.inputs.get("x"),
            Some(&Value::Reference(mul_output.clone()))
        );
    }
}
