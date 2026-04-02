//! Conv+BatchNorm weight folding pass.
//!
//! Mathematically folds BatchNorm parameters into the preceding Conv's
//! weights and bias, completely eliminating the BN op. This is more
//! thorough than the graph-level `ConvBatchNormFusionPass` — it does the
//! actual arithmetic so the BN can be removed entirely.
//!
//! Given conv output `y = W·x + b` and batch_norm `z = γ(y − μ)/√(σ² + ε) + β`:
//!
//! ```text
//! W_folded = W × γ / √(σ² + ε)
//! b_folded = (b − μ) × γ / √(σ² + ε) + β
//! ```

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::{Block, Program};
use crate::ir::tensor::ScalarType;
use crate::ir::types::Value;

use super::replace_reference;
use super::tensor_utils::{f32_slice_to_bytes, tensor_as_f32_slice};
use super::util::is_single_consumer;

/// Folds BatchNorm parameters into Conv weights and bias.
///
/// W is 4-D `(Cout, Cin, kH, kW)` while BN params (γ, β, μ, σ²) are 1-D
/// `(Cout,)`. The folding broadcasts along the output-channel axis.
///
/// Only folds when all parameters are const tensors and the conv output has
/// a single consumer (the batch_norm). Must run *before*
/// [`ConvBatchNormFusionPass`](super::op_fusion::ConvBatchNormFusionPass).
pub struct ConvBatchNormWeightFoldPass;

impl Pass for ConvBatchNormWeightFoldPass {
    fn name(&self) -> &str {
        "conv-bn-weight-fold"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            fold_conv_bn_in_block(&mut function.body);
        }
        Ok(())
    }
}

// ---- Helpers ----------------------------------------------------------------

/// Resolve an op input to the referenced output name.
fn resolve_ref(op: &Operation, input_name: &str) -> Option<String> {
    match op.inputs.get(input_name) {
        Some(Value::Reference(name)) => Some(name.clone()),
        _ => None,
    }
}

/// Find the const op producing `output_name` and return its tensor value.
fn find_const_tensor<'a>(block: &'a Block, output_name: &str) -> Option<&'a Value> {
    block.operations.iter().find_map(|op| {
        if op.op_type == "const" && op.outputs.iter().any(|o| o == output_name) {
            match op.inputs.get("val").or_else(|| op.attributes.get("val")) {
                Some(val @ Value::Tensor { .. }) => Some(val),
                _ => None,
            }
        } else {
            None
        }
    })
}

/// Find the index of the const op producing `output_name`.
fn find_const_op_index(block: &Block, output_name: &str) -> Option<usize> {
    block
        .operations
        .iter()
        .position(|op| op.op_type == "const" && op.outputs.iter().any(|o| o == output_name))
}

/// Extract f32 data from a const tensor.
fn extract_f32_data(block: &Block, output_name: &str) -> Vec<f32> {
    match find_const_tensor(block, output_name) {
        Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data),
        _ => vec![],
    }
}

/// Extract f32 data and shape from a const tensor.
fn extract_f32_data_and_shape(block: &Block, output_name: &str) -> (Vec<f32>, Vec<usize>) {
    match find_const_tensor(block, output_name) {
        Some(Value::Tensor { data, shape, .. }) => (tensor_as_f32_slice(data), shape.clone()),
        _ => (vec![], vec![]),
    }
}

/// Overwrite the tensor data in a const op at `idx`.
fn update_const_tensor(block: &mut Block, idx: usize, new_data: Vec<u8>, new_shape: Vec<usize>) {
    if let Some(Value::Tensor { data, shape, .. }) = block.operations[idx].inputs.get_mut("val") {
        *data = new_data;
        *shape = new_shape;
    }
}

// ---- Core logic -------------------------------------------------------------

/// Process the block, folding one conv→batch_norm pair at a time.
fn fold_conv_bn_in_block(block: &mut Block) {
    while let Some((conv_idx, bn_idx)) = find_fold_candidate(block) {
        if !apply_fold(block, conv_idx, bn_idx) {
            // Mark BN as unfoldable to prevent infinite re-selection.
            block.operations[bn_idx]
                .attributes
                .insert("_skip_fold".into(), Value::Bool(true));
        }
    }
}

/// Scan for the first foldable conv→batch_norm pair.
fn find_fold_candidate(block: &Block) -> Option<(usize, usize)> {
    for (bi, bn_op) in block.operations.iter().enumerate() {
        if bn_op.op_type != "batch_norm" {
            continue;
        }
        if bn_op.attributes.get("_skip_fold") == Some(&Value::Bool(true)) {
            continue;
        }

        let bn_input = match bn_op.inputs.get("x") {
            Some(Value::Reference(name)) => name.clone(),
            _ => continue,
        };

        // Find the producing conv.
        let conv_pos = block.operations.iter().enumerate().find(|(_, op)| {
            op.op_type == "conv"
                && op.outputs.first().map(|s| s.as_str()) == Some(bn_input.as_str())
        });
        let (ci, _) = match conv_pos {
            Some(p) => p,
            None => continue,
        };

        // Conv output must have exactly one consumer (the BN).
        if !is_single_consumer(block, &bn_input, bi) {
            continue;
        }

        // All BN params must be const tensors.
        let all_bn_const = ["mean", "variance", "gamma", "beta"].iter().all(|p| {
            resolve_ref(&block.operations[bi], p)
                .and_then(|name| find_const_tensor(block, &name))
                .is_some()
        });
        if !all_bn_const {
            continue;
        }

        // Conv weight must be const.
        if resolve_ref(&block.operations[ci], "weight")
            .and_then(|name| find_const_tensor(block, &name))
            .is_none()
        {
            continue;
        }

        return Some((ci, bi));
    }
    None
}

/// Apply weight folding for a single conv→batch_norm pair and remove the BN op.
///
/// Returns `true` if the fold was applied, `false` if a validation check failed.
fn apply_fold(block: &mut Block, conv_idx: usize, bn_idx: usize) -> bool {
    // ---- Phase 1: extract all data (immutable) ----

    let epsilon = block.operations[bn_idx]
        .attributes
        .get("epsilon")
        .and_then(|v| match v {
            Value::Float(f) => Some(*f as f32),
            _ => None,
        })
        .unwrap_or(1e-5);

    let mean_ref = resolve_ref(&block.operations[bn_idx], "mean").unwrap();
    let var_ref = resolve_ref(&block.operations[bn_idx], "variance").unwrap();
    let gamma_ref = resolve_ref(&block.operations[bn_idx], "gamma").unwrap();
    let beta_ref = resolve_ref(&block.operations[bn_idx], "beta").unwrap();

    let mean = extract_f32_data(block, &mean_ref);
    let variance = extract_f32_data(block, &var_ref);
    let gamma = extract_f32_data(block, &gamma_ref);
    let beta = extract_f32_data(block, &beta_ref);
    let c_out = gamma.len();

    // All BN params must have the same length.
    if mean.len() != c_out || variance.len() != c_out || beta.len() != c_out {
        return false;
    }

    let weight_ref = resolve_ref(&block.operations[conv_idx], "weight").unwrap();
    let (weight_data, weight_shape) = extract_f32_data_and_shape(block, &weight_ref);

    // Weight shape[0] must match c_out.
    if weight_shape.is_empty() || weight_shape[0] != c_out {
        return false;
    }

    let elements_per_channel = weight_data.len() / c_out;

    let bias_ref = resolve_ref(&block.operations[conv_idx], "bias");
    let original_bias = match &bias_ref {
        Some(name) => {
            let b = extract_f32_data(block, name);
            // Existing bias must match c_out.
            if b.len() != c_out {
                return false;
            }
            b
        }
        None => vec![0.0f32; c_out],
    };

    let conv_name = block.operations[conv_idx].name.clone();
    let bn_output = block.operations[bn_idx].outputs[0].clone();
    let conv_output = block.operations[conv_idx].outputs[0].clone();

    // ---- Phase 2: compute folded values ----

    let mut folded_weight = weight_data;
    let mut folded_bias = vec![0.0f32; c_out];

    for c in 0..c_out {
        let scale = gamma[c] / (variance[c] + epsilon).sqrt();

        let start = c * elements_per_channel;
        let end = start + elements_per_channel;
        for w in &mut folded_weight[start..end] {
            *w *= scale;
        }

        folded_bias[c] = (original_bias[c] - mean[c]) * scale + beta[c];
    }

    // ---- Phase 3: mutate the block ----

    // Update weight const op.
    let weight_const_idx = find_const_op_index(block, &weight_ref).unwrap();
    update_const_tensor(
        block,
        weight_const_idx,
        f32_slice_to_bytes(&folded_weight),
        weight_shape,
    );

    // Update or create bias const op.
    // Track insertions so we can adjust bn_idx.
    let mut inserted = 0usize;

    match &bias_ref {
        Some(name) => {
            let bias_const_idx = find_const_op_index(block, name).unwrap();
            update_const_tensor(
                block,
                bias_const_idx,
                f32_slice_to_bytes(&folded_bias),
                vec![c_out],
            );
        }
        None => {
            // Create a new bias const op and wire it into the conv.
            let bias_out_name = format!("{}_folded_bias", conv_name);
            let bias_op = Operation::new("const", format!("{}_bias_const", conv_name))
                .with_input(
                    "val",
                    Value::Tensor {
                        data: f32_slice_to_bytes(&folded_bias),
                        shape: vec![c_out],
                        dtype: ScalarType::Float32,
                    },
                )
                .with_output(&bias_out_name);

            block.operations.insert(conv_idx, bias_op);
            inserted += 1;

            block.operations[conv_idx + inserted]
                .inputs
                .insert("bias".to_string(), Value::Reference(bias_out_name));
        }
    }

    // Rewire downstream references: BN output → conv output.
    replace_reference(block, &bn_output, &conv_output);

    // Remove the batch_norm op.
    block.operations.remove(bn_idx + inserted);

    // Mark the conv as folded so downstream passes (e.g. ConvBatchNormFusionPass) skip it.
    block.operations[conv_idx + inserted]
        .attributes
        .insert("bn_folded".into(), Value::Bool(true));

    true
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::program::Function;
    use crate::ir::tensor::ScalarType;

    /// Build a minimal program with a single "main" function.
    fn program_with_block(block: Block) -> Program {
        let mut func = Function::new("main");
        func.body = block;
        let mut program = Program::new("1.0.0");
        program.add_function(func);
        program
    }

    fn block_ops(program: &Program) -> &[Operation] {
        &program.functions["main"].body.operations
    }

    /// Helper: create a `const` op producing a tensor value.
    fn const_tensor_op(name: &str, output: &str, data: &[f32], shape: Vec<usize>) -> Operation {
        Operation::new("const", name)
            .with_input(
                "val",
                Value::Tensor {
                    data: f32_slice_to_bytes(data),
                    shape,
                    dtype: ScalarType::Float32,
                },
            )
            .with_output(output)
    }

    /// Build a complete conv→BN subgraph.
    ///
    /// Conv: Cout=2, Cin=1, kH=1, kW=1
    ///   W = [1.0, 2.0], b = [0.5, 0.5]
    ///
    /// BN: gamma=[1.0, 2.0], beta=[0.0, 1.0], mean=[0.0, 1.0], var=[1.0, 1.0], eps=0
    ///
    /// Expected folded:
    ///   scale = [1.0, 2.0]
    ///   W_folded = [1.0, 4.0]
    ///   b_folded = [0.5, 0.0]
    fn build_conv_bn_block() -> Block {
        let mut block = Block::new();

        // Const ops for conv
        block.add_op(const_tensor_op(
            "w_const",
            "w_val",
            &[1.0, 2.0],
            vec![2, 1, 1, 1],
        ));
        block.add_op(const_tensor_op("b_const", "b_val", &[0.5, 0.5], vec![2]));

        // Const ops for BN
        block.add_op(const_tensor_op(
            "gamma_const",
            "gamma_val",
            &[1.0, 2.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "beta_const",
            "beta_val",
            &[0.0, 1.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "mean_const",
            "mean_val",
            &[0.0, 1.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "var_const",
            "var_val",
            &[1.0, 1.0],
            vec![2],
        ));

        // Conv op
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("weight", Value::Reference("w_val".into()))
                .with_input("bias", Value::Reference("b_val".into()))
                .with_output("conv_out"),
        );

        // BatchNorm op
        block.add_op(
            Operation::new("batch_norm", "bn_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_input("gamma", Value::Reference("gamma_val".into()))
                .with_input("beta", Value::Reference("beta_val".into()))
                .with_input("mean", Value::Reference("mean_val".into()))
                .with_input("variance", Value::Reference("var_val".into()))
                .with_attr("epsilon", Value::Float(0.0))
                .with_output("bn_out"),
        );

        block.outputs.push("bn_out".into());
        block
    }

    // ---- Test 1: fold with known weights ------------------------------------

    #[test]
    fn fold_known_weights() {
        let block = build_conv_bn_block();
        let mut program = program_with_block(block);
        ConvBatchNormWeightFoldPass.run(&mut program).unwrap();

        // The BN op should be gone.
        let ops = block_ops(&program);
        assert!(
            !ops.iter().any(|op| op.op_type == "batch_norm"),
            "batch_norm should be removed"
        );

        // Find the weight const and verify folded values.
        let w_const = ops.iter().find(|op| op.name == "w_const").unwrap();
        if let Some(Value::Tensor { data, shape, .. }) = w_const.inputs.get("val") {
            let w = tensor_as_f32_slice(data);
            // scale = [1.0, 2.0], W_folded = [1.0*1.0, 2.0*2.0] = [1.0, 4.0]
            assert_eq!(w, vec![1.0, 4.0]);
            assert_eq!(*shape, vec![2, 1, 1, 1]);
        } else {
            panic!("expected tensor value in weight const");
        }

        // Find the bias const and verify folded values.
        let b_const = ops.iter().find(|op| op.name == "b_const").unwrap();
        if let Some(Value::Tensor { data, .. }) = b_const.inputs.get("val") {
            let b = tensor_as_f32_slice(data);
            // b_folded = [(0.5-0.0)*1.0+0.0, (0.5-1.0)*2.0+1.0] = [0.5, 0.0]
            assert_eq!(b, vec![0.5, 0.0]);
        } else {
            panic!("expected tensor value in bias const");
        }
    }

    // ---- Test 2: skip when BN params are not const --------------------------

    #[test]
    fn skip_non_const_bn_params() {
        let mut block = Block::new();

        block.add_op(const_tensor_op(
            "w_const",
            "w_val",
            &[1.0, 2.0],
            vec![2, 1, 1, 1],
        ));
        block.add_op(const_tensor_op("b_const", "b_val", &[0.5, 0.5], vec![2]));

        // Only gamma/beta/variance are const — mean is dynamic (not a const op).
        block.add_op(const_tensor_op(
            "gamma_const",
            "gamma_val",
            &[1.0, 2.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "beta_const",
            "beta_val",
            &[0.0, 1.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "var_const",
            "var_val",
            &[1.0, 1.0],
            vec![2],
        ));

        // mean_val comes from a non-const op (simulating a dynamic value).
        block.add_op(
            Operation::new("some_op", "dynamic_mean")
                .with_input("x", Value::Reference("input".into()))
                .with_output("mean_val"),
        );

        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("weight", Value::Reference("w_val".into()))
                .with_input("bias", Value::Reference("b_val".into()))
                .with_output("conv_out"),
        );

        block.add_op(
            Operation::new("batch_norm", "bn_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_input("gamma", Value::Reference("gamma_val".into()))
                .with_input("beta", Value::Reference("beta_val".into()))
                .with_input("mean", Value::Reference("mean_val".into()))
                .with_input("variance", Value::Reference("var_val".into()))
                .with_attr("epsilon", Value::Float(1e-5))
                .with_output("bn_out"),
        );
        block.outputs.push("bn_out".into());

        let mut program = program_with_block(block);
        let op_count_before = block_ops(&program).len();

        ConvBatchNormWeightFoldPass.run(&mut program).unwrap();

        // Nothing should be folded — all ops remain.
        assert_eq!(block_ops(&program).len(), op_count_before);
        assert!(
            block_ops(&program)
                .iter()
                .any(|op| op.op_type == "batch_norm")
        );
    }

    // ---- Test 3: skip when conv has multiple consumers ----------------------

    #[test]
    fn skip_multi_consumer() {
        let mut block = build_conv_bn_block();

        // Add a second consumer of conv_out.
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("relu_out"),
        );

        let mut program = program_with_block(block);
        ConvBatchNormWeightFoldPass.run(&mut program).unwrap();

        // batch_norm should still be present — no fold happened.
        assert!(
            block_ops(&program)
                .iter()
                .any(|op| op.op_type == "batch_norm"),
            "batch_norm should remain when conv has multiple consumers"
        );
    }

    // ---- Test 4: BN removed and outputs rewired -----------------------------

    #[test]
    fn bn_removed_and_outputs_rewired() {
        let mut block = build_conv_bn_block();

        // Add a downstream consumer of bn_out.
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("bn_out".into()))
                .with_output("relu_out"),
        );
        block.outputs = vec!["relu_out".into()];

        let mut program = program_with_block(block);
        ConvBatchNormWeightFoldPass.run(&mut program).unwrap();

        let ops = block_ops(&program);

        // BN should be gone.
        assert!(
            !ops.iter().any(|op| op.op_type == "batch_norm"),
            "batch_norm must be removed"
        );

        // relu should now reference conv_out (not bn_out).
        let relu = ops.iter().find(|op| op.op_type == "relu").unwrap();
        assert_eq!(
            relu.inputs.get("x"),
            Some(&Value::Reference("conv_out".into())),
            "downstream op should reference conv_out after fold"
        );
    }

    // ---- Test 5: fold without pre-existing bias ----------------------------

    #[test]
    fn fold_creates_bias_when_absent() {
        let mut block = Block::new();

        // Conv without a bias input.
        block.add_op(const_tensor_op(
            "w_const",
            "w_val",
            &[1.0, 2.0],
            vec![2, 1, 1, 1],
        ));

        block.add_op(const_tensor_op(
            "gamma_const",
            "gamma_val",
            &[1.0, 2.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "beta_const",
            "beta_val",
            &[0.0, 1.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "mean_const",
            "mean_val",
            &[0.0, 1.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "var_const",
            "var_val",
            &[1.0, 1.0],
            vec![2],
        ));

        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("weight", Value::Reference("w_val".into()))
                // no bias input
                .with_output("conv_out"),
        );

        block.add_op(
            Operation::new("batch_norm", "bn_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_input("gamma", Value::Reference("gamma_val".into()))
                .with_input("beta", Value::Reference("beta_val".into()))
                .with_input("mean", Value::Reference("mean_val".into()))
                .with_input("variance", Value::Reference("var_val".into()))
                .with_attr("epsilon", Value::Float(0.0))
                .with_output("bn_out"),
        );
        block.outputs.push("bn_out".into());

        let mut program = program_with_block(block);
        ConvBatchNormWeightFoldPass.run(&mut program).unwrap();

        let ops = block_ops(&program);

        // BN should be removed.
        assert!(!ops.iter().any(|op| op.op_type == "batch_norm"));

        // Conv should now have a bias input.
        let conv = ops.iter().find(|op| op.op_type == "conv").unwrap();
        assert!(
            conv.inputs.contains_key("bias"),
            "conv should have a bias input after fold"
        );

        // Verify the created bias values: (0 - mean) * scale + beta
        // b_folded = [(0-0)*1+0, (0-1)*2+1] = [0.0, -1.0]
        let bias_ref = match conv.inputs.get("bias") {
            Some(Value::Reference(name)) => name.clone(),
            _ => panic!("expected bias reference"),
        };
        let bias_const = ops
            .iter()
            .find(|op| op.op_type == "const" && op.outputs.iter().any(|o| o == &bias_ref))
            .unwrap();
        if let Some(Value::Tensor { data, .. }) = bias_const.inputs.get("val") {
            let b = tensor_as_f32_slice(data);
            assert_eq!(b, vec![0.0, -1.0]);
        } else {
            panic!("expected tensor value in created bias const");
        }
    }

    // ---- Test 6: skip when BN param lengths mismatch -----------------------

    #[test]
    fn skip_mismatched_bn_param_lengths() {
        let mut block = Block::new();

        block.add_op(const_tensor_op(
            "w_const",
            "w_val",
            &[1.0, 2.0],
            vec![2, 1, 1, 1],
        ));
        block.add_op(const_tensor_op("b_const", "b_val", &[0.5, 0.5], vec![2]));

        // gamma has 2 elements but mean has 3 — lengths don't match.
        block.add_op(const_tensor_op(
            "gamma_const",
            "gamma_val",
            &[1.0, 2.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "beta_const",
            "beta_val",
            &[0.0, 1.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "mean_const",
            "mean_val",
            &[0.0, 1.0, 2.0],
            vec![3],
        ));
        block.add_op(const_tensor_op(
            "var_const",
            "var_val",
            &[1.0, 1.0],
            vec![2],
        ));

        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("weight", Value::Reference("w_val".into()))
                .with_input("bias", Value::Reference("b_val".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("batch_norm", "bn_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_input("gamma", Value::Reference("gamma_val".into()))
                .with_input("beta", Value::Reference("beta_val".into()))
                .with_input("mean", Value::Reference("mean_val".into()))
                .with_input("variance", Value::Reference("var_val".into()))
                .with_attr("epsilon", Value::Float(0.0))
                .with_output("bn_out"),
        );
        block.outputs.push("bn_out".into());

        let mut program = program_with_block(block);
        ConvBatchNormWeightFoldPass.run(&mut program).unwrap();

        // No fold — batch_norm should still be present.
        assert!(
            block_ops(&program)
                .iter()
                .any(|op| op.op_type == "batch_norm"),
            "batch_norm should remain when BN param lengths mismatch"
        );
    }

    // ---- Test 7: skip when weight shape[0] != c_out ------------------------

    #[test]
    fn skip_weight_shape_mismatch() {
        let mut block = Block::new();

        // Weight has shape [3,1,1,1] (Cout=3) but BN params have length 2.
        block.add_op(const_tensor_op(
            "w_const",
            "w_val",
            &[1.0, 2.0, 3.0],
            vec![3, 1, 1, 1],
        ));
        block.add_op(const_tensor_op(
            "b_const",
            "b_val",
            &[0.5, 0.5, 0.5],
            vec![3],
        ));

        block.add_op(const_tensor_op(
            "gamma_const",
            "gamma_val",
            &[1.0, 2.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "beta_const",
            "beta_val",
            &[0.0, 1.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "mean_const",
            "mean_val",
            &[0.0, 1.0],
            vec![2],
        ));
        block.add_op(const_tensor_op(
            "var_const",
            "var_val",
            &[1.0, 1.0],
            vec![2],
        ));

        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("weight", Value::Reference("w_val".into()))
                .with_input("bias", Value::Reference("b_val".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("batch_norm", "bn_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_input("gamma", Value::Reference("gamma_val".into()))
                .with_input("beta", Value::Reference("beta_val".into()))
                .with_input("mean", Value::Reference("mean_val".into()))
                .with_input("variance", Value::Reference("var_val".into()))
                .with_attr("epsilon", Value::Float(0.0))
                .with_output("bn_out"),
        );
        block.outputs.push("bn_out".into());

        let mut program = program_with_block(block);
        ConvBatchNormWeightFoldPass.run(&mut program).unwrap();

        // No fold — batch_norm should still be present.
        assert!(
            block_ops(&program)
                .iter()
                .any(|op| op.op_type == "batch_norm"),
            "batch_norm should remain when weight shape[0] != c_out"
        );
    }

    // ---- Test 8: bn_folded attribute is set after fold ----------------------

    #[test]
    fn fold_sets_bn_folded_attribute() {
        let block = build_conv_bn_block();
        let mut program = program_with_block(block);
        ConvBatchNormWeightFoldPass.run(&mut program).unwrap();

        let conv = block_ops(&program)
            .iter()
            .find(|op| op.op_type == "conv")
            .unwrap();
        assert_eq!(
            conv.attributes.get("bn_folded"),
            Some(&Value::Bool(true)),
            "conv should be marked with bn_folded attribute"
        );
    }
}
