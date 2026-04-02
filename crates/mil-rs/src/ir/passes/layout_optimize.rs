//! Layout optimization pass — cancel redundant transpose pairs.
//!
//! ONNX models (especially those originating from TensorFlow) often contain
//! redundant transpose pairs at format boundaries (NHWC→NCHW→NHWC).  This pass
//! detects adjacent transpose pairs whose composed permutation is the identity
//! and removes both, rewiring downstream references to skip them.
//!
//! Note: MIL spatial ops (conv, pool, batch_norm) always validate assuming NCHW
//! layout.  The Apple Neural Engine handles internal layout conversion
//! transparently, so this pass does **not** insert NCHW→NHWC transposes around
//! spatial ops — doing so would violate the MIL contract and cause
//! `coremlcompiler` validation errors.

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::{Block, Program};
use crate::ir::types::Value;

use super::replace_reference;
use super::util::is_single_consumer;

/// Cancel redundant transpose pairs in the program.
///
/// Detects adjacent transpose pairs (A → B) where their composed permutation
/// is the identity `[0, 1, 2, …]`, removes both, and rewires downstream
/// references to A's original input.
pub struct LayoutOptimizationPass;

impl Pass for LayoutOptimizationPass {
    fn name(&self) -> &str {
        "layout-optimization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            cancel_inverse_transposes(&mut function.body);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Cancel inverse transpose pairs
// ---------------------------------------------------------------------------

/// Remove adjacent transpose pairs whose composed permutation is the identity.
///
/// Specifically, if `transpose_a` feeds directly into `transpose_b` and
/// `perm_a(perm_b) == [0, 1, 2, 3]`, both are redundant and can be removed,
/// with downstream references rewired to `transpose_a`'s input.
fn cancel_inverse_transposes(block: &mut Block) {
    let mut remove_indices: Vec<usize> = Vec::new();

    // Collect pairs to cancel.
    let mut pairs: Vec<(usize, usize, String)> = Vec::new(); // (a_idx, b_idx, a_input_name)

    for (bi, op_b) in block.operations.iter().enumerate() {
        if op_b.op_type != "transpose" {
            continue;
        }
        let b_input = match op_b.inputs.get("x") {
            Some(Value::Reference(name)) => name.clone(),
            _ => continue,
        };
        let perm_b = match get_perm(op_b) {
            Some(p) => p,
            None => continue,
        };

        // Find the producer transpose.
        let producer = block.operations.iter().enumerate().find(|(_, op)| {
            op.op_type == "transpose" && op.outputs.first().map(|s| s.as_str()) == Some(&b_input)
        });

        let (ai, op_a) = match producer {
            Some(pair) => pair,
            None => continue,
        };

        let perm_a = match get_perm(op_a) {
            Some(p) => p,
            None => continue,
        };

        let identity: Vec<usize> = (0..perm_a.len()).collect();
        if compose_perms(&perm_a, &perm_b).as_deref() == Some(&identity[..]) {
            // Check single-consumer: op_a's output should only be consumed by op_b.
            if is_single_consumer(block, &b_input, bi) {
                let a_input = match op_a.inputs.get("x") {
                    Some(Value::Reference(name)) => name.clone(),
                    _ => continue,
                };
                pairs.push((ai, bi, a_input));
            }
        }
    }

    for (ai, bi, a_input) in &pairs {
        let b_output = block.operations[*bi].outputs[0].clone();
        // Rewrite: everything that referenced b_output should now reference
        // a's original input.
        replace_reference(block, &b_output, a_input);
        remove_indices.push(*ai);
        remove_indices.push(*bi);
    }

    // Deduplicate and remove in reverse order.
    remove_indices.sort_unstable();
    remove_indices.dedup();
    for idx in remove_indices.into_iter().rev() {
        block.operations.remove(idx);
    }
}

/// Extract the permutation from a transpose op as a `Vec<usize>`.
fn get_perm(op: &Operation) -> Option<Vec<usize>> {
    let perm_val = op
        .inputs
        .get("perm")
        .or_else(|| op.attributes.get("perm"))?;
    match perm_val {
        Value::List(items) => {
            let mut perm = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::Int(n) if n >= &0 => perm.push(*n as usize),
                    _ => return None,
                }
            }
            Some(perm)
        }
        Value::Tensor {
            data,
            dtype: super::super::tensor::ScalarType::Int32,
            ..
        } => Some({
            let vals: Option<Vec<usize>> = data
                .chunks_exact(4)
                .map(|c| {
                    let v = i32::from_le_bytes(c.try_into().unwrap());
                    if v >= 0 { Some(v as usize) } else { None }
                })
                .collect();
            vals?
        }),
        _ => None,
    }
}

/// Compose two permutations: `result[i] = a[b[i]]`.
fn compose_perms(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    b.iter().map(|&bi| a.get(bi).copied()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::Function;

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

    /// Count ops of a given type.
    fn count_ops(program: &Program, op_type: &str) -> usize {
        block_ops(program)
            .iter()
            .filter(|op| op.op_type == op_type)
            .count()
    }

    /// Build a transpose [`Operation`] with the given perm stored as an i32
    /// tensor (mirrors the representation produced by the ONNX importer).
    fn make_transpose_op(name: &str, input: &str, output: &str, perm: &[i64]) -> Operation {
        use crate::ir::tensor::ScalarType;
        let perm_data: Vec<u8> = perm
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect();
        Operation::new("transpose", name)
            .with_input("x", Value::Reference(input.to_string()))
            .with_input(
                "perm",
                Value::Tensor {
                    data: perm_data,
                    shape: vec![perm.len()],
                    dtype: ScalarType::Int32,
                },
            )
            .with_output(output)
    }

    // ---- Inverse transpose pair gets cancelled -----------------------------

    #[test]
    fn inverse_pair_cancelled() {
        // transpose_a: [0,2,3,1] (NCHW→NHWC)
        // transpose_b: [0,3,1,2] (NHWC→NCHW)
        // Composed: identity → both should be removed.
        let mut block = Block::new();
        block.add_op(make_transpose_op("t_a", "input", "t_a_out", &[0, 2, 3, 1]));
        block.add_op(make_transpose_op(
            "t_b",
            "t_a_out",
            "t_b_out",
            &[0, 3, 1, 2],
        ));
        block.outputs.push("t_b_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        assert_eq!(count_ops(&program, "transpose"), 0);
        assert_eq!(block_ops(&program).len(), 0);
        // Block output should be rewired to the original input.
        assert_eq!(program.functions["main"].body.outputs, vec!["input"]);
    }

    // ---- Non-inverse pair is NOT cancelled ---------------------------------

    #[test]
    fn non_inverse_pair_not_cancelled() {
        // Two transposes whose composition is NOT the identity.
        let mut block = Block::new();
        block.add_op(make_transpose_op("t_a", "input", "t_a_out", &[0, 2, 3, 1]));
        block.add_op(make_transpose_op(
            "t_b",
            "t_a_out",
            "t_b_out",
            &[0, 2, 3, 1],
        ));
        block.outputs.push("t_b_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        // Both transposes should remain.
        assert_eq!(count_ops(&program, "transpose"), 2);
        assert_eq!(block_ops(&program).len(), 2);
    }

    // ---- No transposes: pass is a no-op ------------------------------------

    #[test]
    fn no_transposes_is_noop() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("relu_out"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("relu_out".into()))
                .with_output("softmax_out"),
        );
        block.outputs.push("softmax_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        assert_eq!(block_ops(&program).len(), 2);
        assert_eq!(count_ops(&program, "transpose"), 0);
    }

    // ---- Empty block: pass is a no-op --------------------------------------

    #[test]
    fn empty_block_is_noop() {
        let block = Block::new();
        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();
        assert_eq!(block_ops(&program).len(), 0);
    }

    // ---- Multi-consumer prevents cancellation ------------------------------

    #[test]
    fn multi_consumer_prevents_cancellation() {
        // transpose_a feeds both transpose_b (inverse) and another op.
        // Because t_a_out has multiple consumers, the pair must NOT cancel.
        let mut block = Block::new();
        block.add_op(make_transpose_op("t_a", "input", "t_a_out", &[0, 2, 3, 1]));
        block.add_op(make_transpose_op(
            "t_b",
            "t_a_out",
            "t_b_out",
            &[0, 3, 1, 2],
        ));
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("t_a_out".into()))
                .with_output("relu_out"),
        );
        block.outputs.push("t_b_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        // Both transposes should survive.
        assert_eq!(count_ops(&program, "transpose"), 2);
        assert_eq!(block_ops(&program).len(), 3);
    }

    // ---- Block output referencing intermediate prevents cancellation --------

    #[test]
    fn block_output_reference_prevents_cancellation() {
        // t_a_out is listed as a block output, so t_a is multi-consumed.
        let mut block = Block::new();
        block.add_op(make_transpose_op("t_a", "input", "t_a_out", &[0, 2, 3, 1]));
        block.add_op(make_transpose_op(
            "t_b",
            "t_a_out",
            "t_b_out",
            &[0, 3, 1, 2],
        ));
        block.outputs.push("t_a_out".into());
        block.outputs.push("t_b_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        assert_eq!(count_ops(&program, "transpose"), 2);
    }

    // ---- Downstream references rewired correctly ---------------------------

    #[test]
    fn downstream_references_rewired() {
        // input → t_a → t_b → relu (reads t_b_out)
        // After cancellation relu should read "input" directly.
        let mut block = Block::new();
        block.add_op(make_transpose_op("t_a", "input", "t_a_out", &[0, 2, 3, 1]));
        block.add_op(make_transpose_op(
            "t_b",
            "t_a_out",
            "t_b_out",
            &[0, 3, 1, 2],
        ));
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("t_b_out".into()))
                .with_output("relu_out"),
        );
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        assert_eq!(count_ops(&program, "transpose"), 0);
        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "relu");
        match ops[0].inputs.get("x") {
            Some(Value::Reference(name)) => assert_eq!(name, "input"),
            other => panic!("expected Reference(input), got {other:?}"),
        }
    }

    // ---- Multiple inverse pairs in sequence --------------------------------

    #[test]
    fn two_inverse_pairs_both_cancelled() {
        // input → t_a → t_b → relu → t_c → t_d → output
        // Both (t_a, t_b) and (t_c, t_d) are inverse pairs.
        let mut block = Block::new();
        block.add_op(make_transpose_op("t_a", "input", "t_a_out", &[0, 2, 3, 1]));
        block.add_op(make_transpose_op(
            "t_b",
            "t_a_out",
            "t_b_out",
            &[0, 3, 1, 2],
        ));
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("t_b_out".into()))
                .with_output("relu_out"),
        );
        block.add_op(make_transpose_op(
            "t_c",
            "relu_out",
            "t_c_out",
            &[0, 2, 3, 1],
        ));
        block.add_op(make_transpose_op(
            "t_d",
            "t_c_out",
            "t_d_out",
            &[0, 3, 1, 2],
        ));
        block.outputs.push("t_d_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        assert_eq!(count_ops(&program, "transpose"), 0);
        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "relu");
        // relu should read "input", block output should be "relu_out".
        match ops[0].inputs.get("x") {
            Some(Value::Reference(name)) => assert_eq!(name, "input"),
            other => panic!("expected Reference(input), got {other:?}"),
        }
        assert_eq!(program.functions["main"].body.outputs, vec!["relu_out"]);
    }

    // ---- Multiple functions are all processed ------------------------------

    #[test]
    fn processes_all_functions() {
        let mut block_a = Block::new();
        block_a.add_op(make_transpose_op("t_a1", "in_a", "mid_a", &[0, 2, 3, 1]));
        block_a.add_op(make_transpose_op("t_a2", "mid_a", "out_a", &[0, 3, 1, 2]));
        block_a.outputs.push("out_a".into());

        let mut block_b = Block::new();
        block_b.add_op(make_transpose_op("t_b1", "in_b", "mid_b", &[0, 2, 3, 1]));
        block_b.add_op(make_transpose_op("t_b2", "mid_b", "out_b", &[0, 3, 1, 2]));
        block_b.outputs.push("out_b".into());

        let mut func_a = Function::new("func_a");
        func_a.body = block_a;
        let mut func_b = Function::new("func_b");
        func_b.body = block_b;

        let mut program = Program::new("1.0.0");
        program.add_function(func_a);
        program.add_function(func_b);

        LayoutOptimizationPass.run(&mut program).unwrap();

        // Both functions should have their pairs cancelled.
        assert_eq!(program.functions["func_a"].body.operations.len(), 0);
        assert_eq!(program.functions["func_b"].body.operations.len(), 0);
    }

    // ---- Transpose with non-transpose producer is left alone ---------------

    #[test]
    fn single_transpose_not_removed() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("relu_out"),
        );
        block.add_op(make_transpose_op(
            "t_a",
            "relu_out",
            "t_a_out",
            &[0, 2, 3, 1],
        ));
        block.outputs.push("t_a_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        assert_eq!(count_ops(&program, "transpose"), 1);
        assert_eq!(block_ops(&program).len(), 2);
    }
}
