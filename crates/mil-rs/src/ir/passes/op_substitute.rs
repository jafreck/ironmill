//! Op substitution pass — replace ANE-unsupported ops with equivalent
//! supported alternatives.
//!
//! Each substitution expands one unsupported op into one or more ops that
//! the Apple Neural Engine can execute natively, avoiding CPU fallback.

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::{Block, Program};
use crate::ir::types::Value;
use crate::validate::is_ane_supported;

use super::replace_reference;

/// Replaces ANE-unsupported ops with equivalent supported alternatives.
///
/// Substitution table:
///
/// | Unsupported Op | Replacement                        |
/// |----------------|------------------------------------|
/// | `gelu`         | tanh-based GELU approximation      |
pub struct OpSubstitutionPass;

impl Pass for OpSubstitutionPass {
    fn name(&self) -> &str {
        "op-substitution"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            substitute_in_block(&mut function.body);
        }
        Ok(())
    }
}

/// Walk the block and substitute unsupported ops in-place.
fn substitute_in_block(block: &mut Block) {
    let mut i = 0;
    while i < block.operations.len() {
        let op = &block.operations[i];

        // Only substitute ops the ANE can't run natively.
        if is_ane_supported(&op.op_type) {
            i += 1;
            continue;
        }

        let replacements = match op.op_type.as_str() {
            "gelu" => Some(substitute_gelu(op)),
            _ => None,
        };

        if let Some(new_ops) = replacements {
            if let (Some(original_output), Some(last_op)) =
                (op.outputs.first().cloned(), new_ops.last())
            {
                if let Some(last_output) = last_op.outputs.first() {
                    let last_output = last_output.clone();
                    // Remove the original op.
                    block.operations.remove(i);
                    // Splice in the replacements.
                    let count = new_ops.len();
                    for (j, new_op) in new_ops.into_iter().enumerate() {
                        block.operations.insert(i + j, new_op);
                    }
                    // Rewire downstream references from the original output to
                    // the last replacement op's output.
                    if last_output != original_output {
                        replace_reference(block, &original_output, &last_output);
                    }
                    i += count;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
}

// ---- Substitution functions -------------------------------------------------

/// Expand `gelu(x)` into a tanh-based approximation.
///
/// ```text
/// GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
/// ```
///
/// Expands to a chain of ANE-friendly ops:
///   1.  `mul`   — x² = x * x
///   2.  `mul`   — x³ = x² * x
///   3.  `const` — 0.044715
///   4.  `mul`   — 0.044715 * x³
///   5.  `add`   — x + 0.044715·x³
///   6.  `const` — √(2/π)
///   7.  `mul`   — √(2/π) · (x + 0.044715·x³)
///   8.  `tanh`  — tanh(...)
///   9.  `const` — 1.0
///   10. `add`   — 1 + tanh(...)
///   11. `mul`   — x · (1 + tanh(...))
///   12. `const` — 0.5
///   13. `mul`   — 0.5 · x · (1 + tanh(...))
fn substitute_gelu(op: &Operation) -> Vec<Operation> {
    let base = &op.name;
    let x_input = op
        .inputs
        .get("x")
        .cloned()
        .unwrap_or_else(|| Value::Reference("_unknown".into()));

    // Constants
    let coeff = 0.044715_f64;
    let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt(); // ≈ 0.7978845608

    // 1. x_sq = x * x
    let x_sq_name = format!("{base}_sub_0");
    let mul_x_sq = Operation::new("mul", &x_sq_name)
        .with_input("x", x_input.clone())
        .with_input("y", x_input.clone())
        .with_output(&x_sq_name);

    // 2. x_cu = x_sq * x
    let x_cu_name = format!("{base}_sub_1");
    let mul_x_cu = Operation::new("mul", &x_cu_name)
        .with_input("x", Value::Reference(x_sq_name.clone()))
        .with_input("y", x_input.clone())
        .with_output(&x_cu_name);

    // 3. const 0.044715
    let coeff_const_name = format!("{base}_sub_coeff");
    let coeff_const = Operation::new("const", &coeff_const_name)
        .with_input("val", Value::Float(coeff))
        .with_output(&coeff_const_name);

    // 4. coeff_x_cu = 0.044715 * x³
    let coeff_x_cu_name = format!("{base}_sub_2");
    let mul_coeff = Operation::new("mul", &coeff_x_cu_name)
        .with_input("x", Value::Reference(coeff_const_name.clone()))
        .with_input("y", Value::Reference(x_cu_name.clone()))
        .with_output(&coeff_x_cu_name);

    // 5. sum = x + 0.044715·x³
    let sum_name = format!("{base}_sub_3");
    let add_sum = Operation::new("add", &sum_name)
        .with_input("x", x_input.clone())
        .with_input("y", Value::Reference(coeff_x_cu_name.clone()))
        .with_output(&sum_name);

    // 6. const √(2/π)
    let scale_const_name = format!("{base}_sub_scale");
    let scale_const = Operation::new("const", &scale_const_name)
        .with_input("val", Value::Float(sqrt_2_over_pi))
        .with_output(&scale_const_name);

    // 7. scaled = √(2/π) · sum
    let scaled_name = format!("{base}_sub_4");
    let mul_scale = Operation::new("mul", &scaled_name)
        .with_input("x", Value::Reference(scale_const_name.clone()))
        .with_input("y", Value::Reference(sum_name.clone()))
        .with_output(&scaled_name);

    // 8. tanh(scaled)
    let tanh_name = format!("{base}_sub_5");
    let tanh_op = Operation::new("tanh", &tanh_name)
        .with_input("x", Value::Reference(scaled_name.clone()))
        .with_output(&tanh_name);

    // 9. const 1.0
    let one_const_name = format!("{base}_sub_one");
    let one_const = Operation::new("const", &one_const_name)
        .with_input("val", Value::Float(1.0))
        .with_output(&one_const_name);

    // 10. one_plus_tanh = 1 + tanh(...)
    let one_plus_tanh_name = format!("{base}_sub_6");
    let add_one = Operation::new("add", &one_plus_tanh_name)
        .with_input("x", Value::Reference(one_const_name.clone()))
        .with_input("y", Value::Reference(tanh_name.clone()))
        .with_output(&one_plus_tanh_name);

    // 11. x_times_inner = x · (1 + tanh(...))
    let x_times_inner_name = format!("{base}_sub_7");
    let mul_x_inner = Operation::new("mul", &x_times_inner_name)
        .with_input("x", x_input.clone())
        .with_input("y", Value::Reference(one_plus_tanh_name.clone()))
        .with_output(&x_times_inner_name);

    // 12. const 0.5
    let half_const_name = format!("{base}_sub_half");
    let half_const = Operation::new("const", &half_const_name)
        .with_input("val", Value::Float(0.5))
        .with_output(&half_const_name);

    // 13. result = 0.5 · x · (1 + tanh(...))  — final output takes the original op's output name
    let output_name = op
        .outputs
        .first()
        .cloned()
        .unwrap_or_else(|| format!("{base}_sub_8"));
    let mul_half = Operation::new("mul", format!("{base}_sub_8"))
        .with_input("x", Value::Reference(half_const_name.clone()))
        .with_input("y", Value::Reference(x_times_inner_name.clone()))
        .with_output(&output_name);

    vec![
        mul_x_sq,
        mul_x_cu,
        coeff_const,
        mul_coeff,
        add_sum,
        scale_const,
        mul_scale,
        tanh_op,
        one_const,
        add_one,
        mul_x_inner,
        half_const,
        mul_half,
    ]
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

    // ---- gelu substitution ---------------------------------------------------

    #[test]
    fn gelu_expanded_to_tanh_approximation() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("gelu", "gelu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("gelu_out"),
        );
        block.outputs.push("gelu_out".into());

        let mut program = program_with_block(block);
        OpSubstitutionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);

        // gelu expands to 13 ops:
        // mul, mul, const, mul, add, const, mul, tanh, const, add, mul, const, mul
        assert_eq!(
            ops.len(),
            13,
            "expected 13 replacement ops, got {}",
            ops.len()
        );

        // Verify op types in order.
        let types: Vec<&str> = ops.iter().map(|o| o.op_type.as_str()).collect();
        assert_eq!(
            types,
            vec![
                "mul", "mul", "const", "mul", "add", "const", "mul", "tanh", "const", "add", "mul",
                "const", "mul"
            ]
        );

        // The last op should produce the original output name.
        assert_eq!(ops.last().unwrap().outputs[0], "gelu_out");

        // Block output should still reference gelu_out.
        assert_eq!(program.functions["main"].body.outputs, vec!["gelu_out"]);
    }

    #[test]
    fn gelu_downstream_refs_rewired() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("gelu", "gelu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("gelu_out"),
        );
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("gelu_out".into()))
                .with_input("y", Value::Reference("input".into()))
                .with_output("add_out"),
        );
        block.outputs.push("add_out".into());

        let mut program = program_with_block(block);
        OpSubstitutionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        let add_op = ops.iter().find(|o| o.name == "add_0").unwrap();
        // The add op should still reference "gelu_out" (produced by the final mul).
        match add_op.inputs.get("x") {
            Some(Value::Reference(name)) => assert_eq!(name, "gelu_out"),
            other => panic!("expected reference to gelu_out, got {:?}", other),
        }
    }

    // ---- Already-supported ops: no substitution -----------------------------

    #[test]
    fn supported_op_not_substituted() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("conv_out"),
        );
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("relu_out"),
        );
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        OpSubstitutionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].op_type, "conv");
        assert_eq!(ops[1].op_type, "relu");
    }

    // ---- Unknown unsupported op: no substitution (no panic) -----------------

    #[test]
    fn unknown_unsupported_op_left_unchanged() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("exotic_op", "exotic_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("exotic_out"),
        );
        block.outputs.push("exotic_out".into());

        let mut program = program_with_block(block);
        OpSubstitutionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "exotic_op");
    }

    // ---- References resolve after substitution ------------------------------

    #[test]
    fn references_resolve_after_gelu_substitution() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("gelu", "gelu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("gelu_out"),
        );
        block.add_op(
            Operation::new("mul", "mul_final")
                .with_input("x", Value::Reference("input".into()))
                .with_input("y", Value::Reference("gelu_out".into()))
                .with_output("final_out"),
        );
        block.outputs.push("final_out".into());

        let mut program = program_with_block(block);
        OpSubstitutionPass.run(&mut program).unwrap();

        // Collect all defined output names.
        let ops = block_ops(&program);
        let defined: std::collections::HashSet<&str> = ops
            .iter()
            .flat_map(|op| op.outputs.iter().map(|s| s.as_str()))
            .collect();

        // Every reference in inputs should resolve to a defined output or
        // an external input ("input").
        for op in ops {
            for val in op.inputs.values() {
                if let Value::Reference(name) = val {
                    assert!(
                        defined.contains(name.as_str()) || name == "input",
                        "unresolved reference: {name}"
                    );
                }
            }
        }
    }
}
