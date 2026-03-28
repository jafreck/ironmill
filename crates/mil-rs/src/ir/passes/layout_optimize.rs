//! Memory layout optimization pass (NCHW → NHWC).
//!
//! The Apple Neural Engine (ANE) prefers channel-last (NHWC) tensor layouts for
//! spatial operations such as convolution, pooling, and batch normalization.
//! This pass inserts `transpose` ops around those spatial ops to convert from
//! the ONNX-standard NCHW format and then cancels redundant transpose pairs in
//! linear chains.

use std::collections::HashMap;

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::{Block, Program};
use crate::ir::types::Value;

use super::replace_reference;

/// Tensor memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Batch, Channels, Height, Width (ONNX default).
    NCHW,
    /// Batch, Height, Width, Channels (ANE preferred).
    NHWC,
}

/// Permutation that converts NCHW → NHWC: `[0, 2, 3, 1]`.
const NCHW_TO_NHWC: [i64; 4] = [0, 2, 3, 1];

/// Permutation that converts NHWC → NCHW: `[0, 3, 1, 2]`.
const NHWC_TO_NCHW: [i64; 4] = [0, 3, 1, 2];

/// Op types that benefit from NHWC layout on the ANE.
const SPATIAL_OPS: &[&str] = &["conv", "pool", "batch_norm"];

/// Returns `true` if `op_type` is a 4-D spatial op that benefits from NHWC.
fn is_spatial_op(op_type: &str) -> bool {
    SPATIAL_OPS.contains(&op_type)
}

/// Reorder tensor layouts around 4-D spatial ops to ANE-preferred NHWC format,
/// then cancel redundant adjacent transpose pairs.
///
/// **Phase 1** targets linear chains:
/// 1. Insert `transpose [0,2,3,1]` before each spatial op's input (NCHW→NHWC).
/// 2. Insert `transpose [0,3,1,2]` after each spatial op's output (NHWC→NCHW).
/// 3. Cancel adjacent inverse transpose pairs.
/// 4. Propagate layout: if an op's output feeds only NHWC consumers, skip the
///    back-transpose.
pub struct LayoutOptimizationPass;

impl Pass for LayoutOptimizationPass {
    fn name(&self) -> &str {
        "layout-optimization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            let block = &mut function.body;

            // Quick check: bail out if there are no spatial ops at all.
            if !block.operations.iter().any(|op| is_spatial_op(&op.op_type)) {
                continue;
            }

            insert_transposes(block);
            cancel_inverse_transposes(block);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Phase 1: insert transpose pairs around spatial ops
// ---------------------------------------------------------------------------

/// Propagate layout through the block: insert NCHW→NHWC transposes before
/// spatial ops that receive NCHW inputs, let non-spatial (element-wise) ops
/// inherit their input layout transparently, and insert NHWC→NCHW transposes
/// only at block output boundaries where the value is still in NHWC.
fn insert_transposes(block: &mut Block) {
    let mut layout_map: HashMap<String, Layout> = HashMap::new();
    let mut counter: usize = 0;

    // Rebuild the operations list so we can splice transposes in.
    let original_ops: Vec<Operation> = std::mem::take(&mut block.operations);
    let mut new_ops: Vec<Operation> = Vec::with_capacity(original_ops.len() * 2);

    for op in original_ops {
        if is_spatial_op(&op.op_type) {
            // --- spatial op: wants NHWC input ---
            let mut new_op = op;

            if let Some(Value::Reference(input_name)) = new_op.inputs.get("x").cloned() {
                let current = layout_map.get(&input_name).copied().unwrap_or(Layout::NCHW);
                if current == Layout::NCHW {
                    let t_out = format!("_layout_nhwc_{counter}");
                    counter += 1;
                    let t_op = make_transpose(&t_out, &input_name, &NCHW_TO_NHWC, counter);
                    counter += 1;
                    new_ops.push(t_op);
                    new_op
                        .inputs
                        .insert("x".to_string(), Value::Reference(t_out));
                }
                // else: already NHWC — consume directly.
            }

            // Spatial op outputs are in NHWC.
            for out in &new_op.outputs {
                layout_map.insert(out.clone(), Layout::NHWC);
            }
            new_ops.push(new_op);
        } else {
            // --- non-spatial op: transparent — inherits input layout ---
            // Collect layouts for all reference inputs to detect mixed-layout
            // scenarios (e.g., add with one NHWC and one NCHW operand).
            let mut nhwc_inputs: Vec<String> = Vec::new();
            let mut nchw_inputs: Vec<String> = Vec::new();
            collect_ref_layouts(&op, &layout_map, &mut nhwc_inputs, &mut nchw_inputs);

            let mut op = op;

            if !nhwc_inputs.is_empty() && !nchw_inputs.is_empty() {
                // Mixed layouts — normalize minority inputs to match majority.
                let (targets, target_layout, perm) =
                    if nhwc_inputs.len() >= nchw_inputs.len() {
                        (nchw_inputs, Layout::NHWC, &NCHW_TO_NHWC)
                    } else {
                        (nhwc_inputs, Layout::NCHW, &NHWC_TO_NCHW)
                    };

                for input_name in &targets {
                    let t_out = format!("_layout_norm_{counter}");
                    counter += 1;
                    let t_op = make_transpose(&t_out, input_name, perm, counter);
                    counter += 1;
                    layout_map.insert(t_out.clone(), target_layout);
                    new_ops.push(t_op);

                    // Rewrite every occurrence of this input in the op.
                    for value in op.inputs.values_mut() {
                        rename_in_value(value, input_name, &t_out);
                    }
                }

                for out in &op.outputs {
                    layout_map.insert(out.clone(), target_layout);
                }
            } else {
                let input_layout = infer_input_layout(&op, &layout_map);
                for out in &op.outputs {
                    layout_map.insert(out.clone(), input_layout);
                }
            }

            new_ops.push(op);
        }
    }

    // At block output boundaries: if a block output is still NHWC, insert a
    // post-transpose so external consumers see NCHW.
    let mut post_ops: Vec<Operation> = Vec::new();
    for output_name in &block.outputs {
        if layout_map.get(output_name).copied() == Some(Layout::NHWC) {
            let internal = format!("_layout_nhwc_out_{counter}");
            counter += 1;
            // Rename the NHWC value everywhere in new_ops.
            rename_value_in_ops(&mut new_ops, output_name, &internal);
            // Transpose from internal (NHWC) → original output name (NCHW).
            let t_op = make_transpose(output_name, &internal, &NHWC_TO_NCHW, counter);
            counter += 1;
            post_ops.push(t_op);
        }
    }
    new_ops.extend(post_ops);

    block.operations = new_ops;
}

/// Determine the layout a transparent (non-spatial) op inherits from its inputs.
fn infer_input_layout(op: &Operation, layout_map: &HashMap<String, Layout>) -> Layout {
    // Prefer the canonical "x" input.
    if let Some(Value::Reference(name)) = op.inputs.get("x") {
        if let Some(&layout) = layout_map.get(name) {
            return layout;
        }
    }
    // Fall back to any reference input.
    for value in op.inputs.values() {
        if let Value::Reference(name) = value {
            if let Some(&layout) = layout_map.get(name) {
                return layout;
            }
        }
    }
    Layout::NCHW
}

/// Partition reference inputs of an op into NHWC and NCHW buckets based on
/// the current layout map.
fn collect_ref_layouts(
    op: &Operation,
    layout_map: &HashMap<String, Layout>,
    nhwc: &mut Vec<String>,
    nchw: &mut Vec<String>,
) {
    for value in op.inputs.values() {
        collect_ref_layouts_value(value, layout_map, nhwc, nchw);
    }
}

fn collect_ref_layouts_value(
    value: &Value,
    layout_map: &HashMap<String, Layout>,
    nhwc: &mut Vec<String>,
    nchw: &mut Vec<String>,
) {
    match value {
        Value::Reference(name) => match layout_map.get(name) {
            Some(&Layout::NHWC) => nhwc.push(name.clone()),
            Some(&Layout::NCHW) => nchw.push(name.clone()),
            None => nchw.push(name.clone()),
        },
        Value::List(items) => {
            for item in items {
                collect_ref_layouts_value(item, layout_map, nhwc, nchw);
            }
        }
        _ => {}
    }
}

/// Rename all occurrences of `old_name` (in outputs and reference inputs)
/// across the given ops list.
fn rename_value_in_ops(ops: &mut [Operation], old_name: &str, new_name: &str) {
    for op in ops.iter_mut() {
        for output in &mut op.outputs {
            if output == old_name {
                *output = new_name.to_string();
            }
        }
        for value in op.inputs.values_mut() {
            rename_in_value(value, old_name, new_name);
        }
    }
}

/// Recursively rename references inside a [`Value`].
fn rename_in_value(value: &mut Value, old_name: &str, new_name: &str) {
    match value {
        Value::Reference(name) if name == old_name => {
            *name = new_name.to_string();
        }
        Value::List(items) => {
            for item in items {
                rename_in_value(item, old_name, new_name);
            }
        }
        _ => {}
    }
}

/// Build a `transpose` [`Operation`].
fn make_transpose(output: &str, input: &str, perm: &[i64; 4], id: usize) -> Operation {
    Operation::new("transpose", format!("_layout_transpose_{id}"))
        .with_input("x", Value::Reference(input.to_string()))
        .with_input(
            "perm",
            Value::List(perm.iter().map(|&p| Value::Int(p)).collect()),
        )
        .with_output(output)
}

// ---------------------------------------------------------------------------
// Phase 2: cancel inverse transpose pairs
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
            op.op_type == "transpose"
                && op.outputs.first().map(|s| s.as_str()) == Some(&b_input)
        });

        let (ai, op_a) = match producer {
            Some(pair) => pair,
            None => continue,
        };

        let perm_a = match get_perm(op_a) {
            Some(p) => p,
            None => continue,
        };

        if compose_perms(&perm_a, &perm_b) == [0, 1, 2, 3] {
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
    let perm_val = op.inputs.get("perm").or_else(|| op.attributes.get("perm"))?;
    if let Value::List(items) = perm_val {
        let mut perm = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Value::Int(n) => perm.push(*n as usize),
                _ => return None,
            }
        }
        Some(perm)
    } else {
        None
    }
}

/// Compose two permutations: `result[i] = a[b[i]]`.
fn compose_perms(a: &[usize], b: &[usize]) -> Vec<usize> {
    b.iter().map(|&bi| a[bi]).collect()
}

/// Check if `value_name` is consumed only by the operation at `consumer_idx`.
fn is_single_consumer(block: &Block, value_name: &str, consumer_idx: usize) -> bool {
    for (idx, op) in block.operations.iter().enumerate() {
        if idx == consumer_idx {
            continue;
        }
        for input_val in op.inputs.values() {
            if references_name(input_val, value_name) {
                return false;
            }
        }
    }
    !block.outputs.contains(&value_name.to_string())
}

/// Returns `true` if `value` contains a reference to `name`.
fn references_name(value: &Value, name: &str) -> bool {
    match value {
        Value::Reference(n) => n == name,
        Value::List(items) => items.iter().any(|v| references_name(v, name)),
        _ => false,
    }
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

    /// Assert that a transpose op has the expected permutation.
    fn assert_perm(op: &Operation, expected: &[i64]) {
        assert_eq!(op.op_type, "transpose");
        let perm = op.inputs.get("perm").expect("transpose should have perm");
        let got: Vec<i64> = if let Value::List(items) = perm {
            items
                .iter()
                .map(|v| match v {
                    Value::Int(n) => *n,
                    _ => panic!("perm items should be Int"),
                })
                .collect()
        } else {
            panic!("perm should be a List");
        };
        assert_eq!(got, expected);
    }

    // ---- Single conv: verify transposes inserted ---------------------------

    #[test]
    fn single_conv_gets_transposes() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        block.outputs.push("conv_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        let ops = block_ops(&program);

        // Should be: pre-transpose, conv, post-transpose
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].op_type, "transpose");
        assert_perm(&ops[0], &[0, 2, 3, 1]);
        assert_eq!(ops[1].op_type, "conv");
        assert_eq!(ops[2].op_type, "transpose");
        assert_perm(&ops[2], &[0, 3, 1, 2]);

        // Block output should still be "conv_out" (produced by the post-transpose).
        assert_eq!(program.functions["main"].body.outputs, vec!["conv_out"]);
    }

    // ---- Conv→relu→conv chain: intermediate transposes cancelled -----------

    #[test]
    fn conv_relu_conv_chain_cancels_middle_transposes() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv0_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv0_out".into()))
                .with_output("relu_out"),
        );
        block.add_op(
            Operation::new("conv", "conv_1")
                .with_input("x", Value::Reference("relu_out".into()))
                .with_output("conv1_out"),
        );
        block.outputs.push("conv1_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        // The post-transpose of conv_0 (NHWC→NCHW) and pre-transpose of conv_1
        // (NCHW→NHWC) should cancel, leaving:
        //   pre-transpose, conv_0, relu, conv_1, post-transpose
        // The relu is a pass-through (no perm awareness) so it stays.
        // Total transpose count should be exactly 2 (one at entry, one at exit).
        let transpose_count = count_ops(&program, "transpose");
        assert_eq!(transpose_count, 2, "only boundary transposes should remain");

        // Block output should still work correctly.
        assert_eq!(program.functions["main"].body.outputs, vec!["conv1_out"]);
    }

    // ---- Non-4D ops: verify no transposes inserted -------------------------

    #[test]
    fn non_spatial_ops_unchanged() {
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

        // No spatial ops → pass is a no-op.
        assert_eq!(block_ops(&program).len(), 2);
        assert_eq!(count_ops(&program, "transpose"), 0);
    }

    // ---- Model with no spatial ops: pass is no-op --------------------------

    #[test]
    fn empty_block_is_noop() {
        let block = Block::new();
        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();
        assert_eq!(block_ops(&program).len(), 0);
    }

    #[test]
    fn linear_only_model_is_noop() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("linear", "linear_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("linear_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("linear_out".into()))
                .with_output("relu_out"),
        );
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        assert_eq!(block_ops(&program).len(), 2);
        assert_eq!(count_ops(&program, "transpose"), 0);
    }

    // ---- Pool op gets transposes -------------------------------------------

    #[test]
    fn pool_op_gets_transposes() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("pool", "pool_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("pool_out"),
        );
        block.outputs.push("pool_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        assert_eq!(block_ops(&program).len(), 3);
        assert_eq!(count_ops(&program, "transpose"), 2);
        assert_eq!(count_ops(&program, "pool"), 1);
    }

    // ---- batch_norm op gets transposes -------------------------------------

    #[test]
    fn batch_norm_op_gets_transposes() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("batch_norm", "bn_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("bn_out"),
        );
        block.outputs.push("bn_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        assert_eq!(block_ops(&program).len(), 3);
        assert_eq!(count_ops(&program, "transpose"), 2);
        assert_eq!(count_ops(&program, "batch_norm"), 1);
    }

    // ---- Downstream references stay valid ----------------------------------

    #[test]
    fn downstream_op_sees_correct_output() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("softmax", "softmax_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("softmax_out"),
        );
        block.outputs.push("softmax_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        // softmax should still receive "conv_out" (produced by the post-transpose).
        let ops = block_ops(&program);
        let softmax = ops.iter().find(|op| op.op_type == "softmax").unwrap();
        match softmax.inputs.get("x") {
            Some(Value::Reference(name)) => assert_eq!(name, "conv_out"),
            other => panic!("expected Reference(conv_out), got {other:?}"),
        }
    }

    // ---- Multiple functions ------------------------------------------------

    #[test]
    fn processes_all_functions() {
        let mut block_a = Block::new();
        block_a.add_op(
            Operation::new("conv", "conv_a")
                .with_input("x", Value::Reference("in_a".into()))
                .with_output("out_a"),
        );
        block_a.outputs.push("out_a".into());

        let mut block_b = Block::new();
        block_b.add_op(
            Operation::new("pool", "pool_b")
                .with_input("x", Value::Reference("in_b".into()))
                .with_output("out_b"),
        );
        block_b.outputs.push("out_b".into());

        let mut func_a = Function::new("func_a");
        func_a.body = block_a;
        let mut func_b = Function::new("func_b");
        func_b.body = block_b;

        let mut program = Program::new("1.0.0");
        program.add_function(func_a);
        program.add_function(func_b);

        LayoutOptimizationPass.run(&mut program).unwrap();

        // Both functions should have transposes.
        assert_eq!(program.functions["func_a"].body.operations.len(), 3);
        assert_eq!(program.functions["func_b"].body.operations.len(), 3);
    }

    // ---- Mixed-layout multi-input op: normalizing transpose inserted ------

    #[test]
    fn mixed_layout_add_inserts_normalizing_transpose() {
        // conv produces NHWC output; a direct (NCHW) input feeds the other
        // operand of `add`. The pass should insert a NCHW→NHWC transpose on
        // the NCHW input so both operands match.
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input_a".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_input("y", Value::Reference("input_b".into()))
                .with_output("add_out"),
        );
        block.outputs.push("add_out".into());

        let mut program = program_with_block(block);
        LayoutOptimizationPass.run(&mut program).unwrap();

        let ops = block_ops(&program);

        // There should be a normalizing NCHW→NHWC transpose whose output
        // feeds the add's "y" input.
        let add_op = ops.iter().find(|op| op.op_type == "add").unwrap();
        let y_input = match add_op.inputs.get("y") {
            Some(Value::Reference(name)) => name.clone(),
            other => panic!("expected Reference for y, got {other:?}"),
        };

        let norm_transpose = ops
            .iter()
            .find(|op| {
                op.op_type == "transpose"
                    && op.outputs.first().map(|s| s.as_str()) == Some(y_input.as_str())
            })
            .expect("normalizing transpose should exist for the NCHW input");

        assert_perm(norm_transpose, &[0, 2, 3, 1]); // NCHW→NHWC
    }
}
