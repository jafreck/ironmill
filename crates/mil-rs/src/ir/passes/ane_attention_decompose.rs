//! Attention decomposition pass for ANE.
//!
//! ANE's SDPA op ignores causal masks (constraint #6), so we must
//! decompose `scaled_dot_product_attention` into explicit
//! matmul → mask → softmax → matmul chains.

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::{Block, Program};
use crate::ir::types::Value;

use super::replace_reference;

/// Replace `scaled_dot_product_attention` with explicit matmul+mask+softmax+matmul.
pub struct AttentionDecomposePass;

impl Pass for AttentionDecomposePass {
    fn name(&self) -> &str {
        "ane-attention-decompose"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            decompose_in_block(&mut function.body);
        }
        Ok(())
    }
}

fn decompose_in_block(block: &mut Block) {
    let mut i = 0;
    while i < block.operations.len() {
        if block.operations[i].op_type != "scaled_dot_product_attention" {
            i += 1;
            continue;
        }

        let op = &block.operations[i];
        let new_ops = decompose_sdpa(op);

        let original_output = op.outputs.first().cloned().unwrap_or_default();
        let last_output = new_ops
            .last()
            .and_then(|o| o.outputs.first().cloned())
            .unwrap_or_default();

        block.operations.remove(i);
        let count = new_ops.len();
        for (j, new_op) in new_ops.into_iter().enumerate() {
            block.operations.insert(i + j, new_op);
        }

        if last_output != original_output {
            replace_reference(block, &original_output, &last_output);
        }
        i += count;
    }
}

/// Resolve Q, K, V, and optional mask inputs from the SDPA op.
fn get_input(op: &Operation, primary: &str, fallback: &str) -> Value {
    op.inputs
        .get(primary)
        .or_else(|| op.inputs.get(fallback))
        .cloned()
        .unwrap_or_else(|| Value::Reference("_unknown".into()))
}

fn get_mask(op: &Operation) -> Option<Value> {
    op.inputs
        .get("attn_mask")
        .or_else(|| op.inputs.get("mask"))
        .cloned()
}

/// Decompose a single SDPA op into matmul→scale→[mask→]softmax→matmul.
fn decompose_sdpa(op: &Operation) -> Vec<Operation> {
    let base = &op.name;
    let q = get_input(op, "query", "q");
    let k = get_input(op, "key", "k");
    let v = get_input(op, "value", "v");
    let mask = get_mask(op);

    let output_name = op
        .outputs
        .first()
        .cloned()
        .unwrap_or_else(|| format!("{base}_out"));

    let mut ops = Vec::new();

    // 1. scores = matmul(Q, K^T)
    let scores_name = format!("{base}_scores");
    ops.push(
        Operation::new("matmul", &scores_name)
            .with_input("x", q)
            .with_input("y", k)
            .with_attr("transpose_y", Value::Bool(true))
            .with_output(&scores_name),
    );

    // 2. scale_val = const(1.0 / sqrt(d_k))
    // Use d_k from the op's attributes if available, otherwise default to 64.
    let d_k: f64 = op
        .attributes
        .get("d_k")
        .and_then(|v| match v {
            Value::Int(n) => Some(*n as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        })
        .unwrap_or(64.0);
    let scale = 1.0 / d_k.sqrt();

    let scale_name = format!("{base}_scale_val");
    ops.push(
        Operation::new("const", &scale_name)
            .with_input("val", Value::Float(scale))
            .with_output(&scale_name),
    );

    // 3. scaled = mul(scores, scale_val)
    let scaled_name = format!("{base}_scaled");
    ops.push(
        Operation::new("mul", &scaled_name)
            .with_input("x", Value::Reference(scores_name))
            .with_input("y", Value::Reference(scale_name))
            .with_output(&scaled_name),
    );

    // Track the name to feed into softmax (may be scaled or masked).
    let softmax_input;

    // 4. Optionally apply mask: masked = add(scaled, mask)
    if let Some(mask_val) = mask {
        let masked_name = format!("{base}_masked");
        ops.push(
            Operation::new("add", &masked_name)
                .with_input("x", Value::Reference(scaled_name))
                .with_input("y", mask_val)
                .with_output(&masked_name),
        );
        softmax_input = masked_name;
    } else {
        softmax_input = scaled_name;
    }

    // 5. attn = softmax(input, axis=-1)
    let attn_name = format!("{base}_attn");
    ops.push(
        Operation::new("softmax", &attn_name)
            .with_input("x", Value::Reference(softmax_input))
            .with_attr("axis", Value::Int(-1))
            .with_output(&attn_name),
    );

    // 6. output = matmul(attn, V)
    ops.push(
        Operation::new("matmul", format!("{base}_out"))
            .with_input("x", Value::Reference(attn_name))
            .with_input("y", v)
            .with_output(&output_name),
    );

    ops
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn pass_name() {
        assert_eq!(AttentionDecomposePass.name(), "ane-attention-decompose");
    }

    #[test]
    fn ane_attention_decompose_with_mask() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("scaled_dot_product_attention", "sdpa_0")
                .with_input("query", Value::Reference("q".into()))
                .with_input("key", Value::Reference("k".into()))
                .with_input("value", Value::Reference("v".into()))
                .with_input("attn_mask", Value::Reference("mask".into()))
                .with_output("sdpa_out"),
        );
        block.outputs.push("sdpa_out".into());

        let mut program = program_with_block(block);
        AttentionDecomposePass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        let types: Vec<&str> = ops.iter().map(|o| o.op_type.as_str()).collect();
        // matmul, const, mul, add (mask), softmax, matmul
        assert_eq!(
            types,
            vec!["matmul", "const", "mul", "add", "softmax", "matmul"]
        );

        // Final output should be "sdpa_out".
        assert_eq!(ops.last().unwrap().outputs[0], "sdpa_out");
        assert_eq!(program.functions["main"].body.outputs, vec!["sdpa_out"]);
    }

    #[test]
    fn ane_attention_decompose_without_mask() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("scaled_dot_product_attention", "sdpa_0")
                .with_input("q", Value::Reference("q_in".into()))
                .with_input("k", Value::Reference("k_in".into()))
                .with_input("v", Value::Reference("v_in".into()))
                .with_output("sdpa_out"),
        );
        block.outputs.push("sdpa_out".into());

        let mut program = program_with_block(block);
        AttentionDecomposePass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        let types: Vec<&str> = ops.iter().map(|o| o.op_type.as_str()).collect();
        // No mask → no add step.
        assert_eq!(types, vec!["matmul", "const", "mul", "softmax", "matmul"]);
        assert_eq!(ops.last().unwrap().outputs[0], "sdpa_out");
    }

    #[test]
    fn ane_attention_decompose_downstream_refs() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("scaled_dot_product_attention", "sdpa_0")
                .with_input("query", Value::Reference("q".into()))
                .with_input("key", Value::Reference("k".into()))
                .with_input("value", Value::Reference("v".into()))
                .with_output("sdpa_out"),
        );
        block.add_op(
            Operation::new("add", "residual")
                .with_input("x", Value::Reference("sdpa_out".into()))
                .with_input("y", Value::Reference("skip".into()))
                .with_output("res_out"),
        );
        block.outputs.push("res_out".into());

        let mut program = program_with_block(block);
        AttentionDecomposePass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        let residual = ops.iter().find(|o| o.name == "residual").unwrap();
        match residual.inputs.get("x") {
            Some(Value::Reference(name)) => assert_eq!(name, "sdpa_out"),
            other => panic!("expected reference to sdpa_out, got {:?}", other),
        }
    }

    #[test]
    fn ane_attention_non_sdpa_untouched() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("matmul", "mm_0")
                .with_input("x", Value::Reference("a".into()))
                .with_input("y", Value::Reference("b".into()))
                .with_output("mm_out"),
        );
        block.outputs.push("mm_out".into());

        let mut program = program_with_block(block);
        AttentionDecomposePass.run(&mut program).unwrap();

        assert_eq!(block_ops(&program).len(), 1);
        assert_eq!(block_ops(&program)[0].op_type, "matmul");
    }
}
