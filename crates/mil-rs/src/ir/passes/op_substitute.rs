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

        // Pattern-based substitution: SiLU fusion.
        // Detect mul(a, sigmoid(a)) or mul(sigmoid(a), a) → silu(a).
        // Both sigmoid and silu are eval-verified on ANE; fusion reduces 2 ops → 1.
        if op.op_type == "mul" {
            if let Some((silu_op, sigmoid_idx)) = try_fuse_silu(block, i) {
                let original_output = op.outputs.first().cloned().unwrap();
                let silu_output = silu_op.outputs.first().cloned().unwrap();

                // Remove the mul op first.
                block.operations.remove(i);

                // Remove the sigmoid (adjust index for the removed mul).
                let adj_si = if sigmoid_idx > i {
                    sigmoid_idx - 1
                } else {
                    sigmoid_idx
                };
                block.operations.remove(adj_si);

                // Insert silu at the earlier of the two removed positions.
                let insert_at = adj_si.min(i);
                block.operations.insert(insert_at, silu_op);

                if silu_output != original_output {
                    replace_reference(block, &original_output, &silu_output);
                }
                i = insert_at + 1;
                continue;
            }
        }

        // Pattern-based substitution: rsqrt pattern.
        // Detect real_div(1.0, sqrt(x)) → pow(x, -0.5).
        // real_div is only compile-verified on ANE; pow(-0.5) is eval-verified.
        if op.op_type == "real_div" {
            if let Some(new_ops) = try_substitute_rsqrt(block, i) {
                let original_output = op.outputs.first().cloned().unwrap();
                let last_output = new_ops.last().unwrap().outputs.first().cloned().unwrap();
                // Find and remove the sqrt op that feeds this real_div.
                let sqrt_idx = find_sqrt_feeding_real_div(block, i);
                // Remove the real_div first (it's at index i).
                block.operations.remove(i);
                // If sqrt was before real_div, its index is still valid.
                if let Some(si) = sqrt_idx {
                    if si < i {
                        block.operations.remove(si);
                        // Adjust i since we removed an op before it.
                        let insert_at = si;
                        let count = new_ops.len();
                        for (j, new_op) in new_ops.into_iter().enumerate() {
                            block.operations.insert(insert_at + j, new_op);
                        }
                        if last_output != original_output {
                            replace_reference(block, &original_output, &last_output);
                        }
                        i = insert_at + count;
                    } else {
                        // sqrt was after real_div (unusual).
                        let count = new_ops.len();
                        for (j, new_op) in new_ops.into_iter().enumerate() {
                            block.operations.insert(i + j, new_op);
                        }
                        if last_output != original_output {
                            replace_reference(block, &original_output, &last_output);
                        }
                        i += count;
                    }
                } else {
                    // No sqrt found — just replace real_div with pow.
                    let count = new_ops.len();
                    for (j, new_op) in new_ops.into_iter().enumerate() {
                        block.operations.insert(i + j, new_op);
                    }
                    if last_output != original_output {
                        replace_reference(block, &original_output, &last_output);
                    }
                    i += count;
                }
                continue;
            }
        }

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

// ---- Pattern-based substitutions -------------------------------------------

/// Detect `mul(a, sigmoid(a))` or `mul(sigmoid(a), a)` and fuse to `silu(a)`.
///
/// Returns `(silu_op, sigmoid_index)` if the pattern matches and the sigmoid
/// output is only consumed by this mul (safe to remove).
fn try_fuse_silu(block: &Block, mul_idx: usize) -> Option<(Operation, usize)> {
    let op = &block.operations[mul_idx];
    if op.op_type != "mul" {
        return None;
    }

    let x_ref = match op.inputs.get("x") {
        Some(Value::Reference(name)) => name.clone(),
        _ => return None,
    };
    let y_ref = match op.inputs.get("y") {
        Some(Value::Reference(name)) => name.clone(),
        _ => return None,
    };

    // Try both orderings: mul(a, sigmoid(a)) and mul(sigmoid(a), a).
    let (sigmoid_output, raw_input) = if let Some(r) = match_sigmoid_input(block, &y_ref, &x_ref) {
        r
    } else if let Some(r) = match_sigmoid_input(block, &x_ref, &y_ref) {
        r
    } else {
        return None;
    };

    // Find sigmoid op index.
    let sigmoid_idx = block.operations.iter().position(|o| {
        o.op_type == "sigmoid" && o.outputs.first().map(|s| s.as_str()) == Some(&sigmoid_output)
    })?;

    // Only fuse if sigmoid is single-consumed (only used by this mul).
    let sigmoid_used_elsewhere = block.operations.iter().enumerate().any(|(idx, o)| {
        if idx == mul_idx {
            return false;
        }
        o.inputs
            .values()
            .any(|v| matches!(v, Value::Reference(name) if name == &sigmoid_output))
    }) || block.outputs.contains(&sigmoid_output);

    if sigmoid_used_elsewhere {
        return None;
    }

    // Preserve the mul's output name so downstream references work.
    let output_name = op
        .outputs
        .first()
        .cloned()
        .unwrap_or_else(|| format!("{}_silu", op.name));

    let silu_op = Operation::new("silu", format!("{}_silu", op.name))
        .with_input("x", Value::Reference(raw_input))
        .with_output(&output_name);

    Some((silu_op, sigmoid_idx))
}

/// Check if `candidate_sigmoid` is a sigmoid output whose input matches `candidate_raw`.
fn match_sigmoid_input(
    block: &Block,
    candidate_sigmoid: &str,
    candidate_raw: &str,
) -> Option<(String, String)> {
    let sigmoid_op = block.operations.iter().find(|o| {
        o.op_type == "sigmoid" && o.outputs.first().map(|s| s.as_str()) == Some(candidate_sigmoid)
    })?;

    match sigmoid_op.inputs.get("x") {
        Some(Value::Reference(name)) if name == candidate_raw => {
            Some((candidate_sigmoid.to_string(), candidate_raw.to_string()))
        }
        _ => None,
    }
}

/// Detect `real_div(scalar_1.0, sqrt_output)` and replace the sqrt + real_div
/// pair with `pow(sqrt_input, -0.5)`.
///
/// The ANE op support matrix shows:
/// - `real_div` → ⚠️ compile-only (not eval-verified)
/// - `pow(x, -0.5)` → ✅ eval-verified (max_err=0.0004)
fn try_substitute_rsqrt(block: &Block, real_div_idx: usize) -> Option<Vec<Operation>> {
    let op = &block.operations[real_div_idx];
    if op.op_type != "real_div" {
        return None;
    }

    // Check that x is a scalar 1.0.
    let x_is_one = match op.inputs.get("x") {
        Some(Value::Float(f)) => (*f - 1.0).abs() < 1e-6,
        Some(Value::Reference(name)) => {
            // Check if it references a const with value 1.0.
            block.operations.iter().any(|o| {
                o.op_type == "const"
                    && o.outputs.first().map(|s| s.as_str()) == Some(name.as_str())
                    && matches!(o.inputs.get("val").or(o.attributes.get("val")),
                        Some(Value::Float(f)) if (*f - 1.0).abs() < 1e-6)
            })
        }
        _ => false,
    };
    if !x_is_one {
        return None;
    }

    // Check that y references a sqrt op's output.
    let y_ref = match op.inputs.get("y") {
        Some(Value::Reference(name)) => name.clone(),
        _ => return None,
    };

    let sqrt_op = block
        .operations
        .iter()
        .find(|o| o.op_type == "sqrt" && o.outputs.first().map(|s| s.as_str()) == Some(&y_ref))?;

    // Get the sqrt's input (the value we want to compute pow(x, -0.5) on).
    let sqrt_input = sqrt_op.inputs.get("x")?.clone();

    let base = &op.name;
    let output_name = op
        .outputs
        .first()
        .cloned()
        .unwrap_or_else(|| format!("{base}_pow_out"));

    // const -0.5
    let nhalf_name = format!("{base}_nhalf");
    let nhalf_const = Operation::new("const", &nhalf_name)
        .with_input("val", Value::Float(-0.5))
        .with_output(&nhalf_name);

    // pow(sqrt_input, -0.5)
    let pow_op = Operation::new("pow", format!("{base}_pow"))
        .with_input("x", sqrt_input)
        .with_input("y", Value::Reference(nhalf_name.clone()))
        .with_output(&output_name);

    Some(vec![nhalf_const, pow_op])
}

/// Find the sqrt op that feeds the real_div at `real_div_idx`.
fn find_sqrt_feeding_real_div(block: &Block, real_div_idx: usize) -> Option<usize> {
    let op = &block.operations[real_div_idx];
    let y_ref = match op.inputs.get("y") {
        Some(Value::Reference(name)) => name.clone(),
        _ => return None,
    };
    block
        .operations
        .iter()
        .position(|o| o.op_type == "sqrt" && o.outputs.first().map(|s| s.as_str()) == Some(&y_ref))
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

    // ---- SiLU fusion --------------------------------------------------------

    #[test]
    fn silu_fusion_basic() {
        // sigmoid(x) → mul(x, sigmoid_out) → silu(x)
        let mut block = Block::new();
        block.add_op(
            Operation::new("sigmoid", "sig_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("sig_out"),
        );
        block.add_op(
            Operation::new("mul", "mul_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("y", Value::Reference("sig_out".into()))
                .with_output("mul_out"),
        );
        block.outputs.push("mul_out".into());

        let mut program = program_with_block(block);
        OpSubstitutionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1, "expected 1 silu op, got {}", ops.len());
        assert_eq!(ops[0].op_type, "silu");
        assert_eq!(ops[0].outputs[0], "mul_out");
        match ops[0].inputs.get("x") {
            Some(Value::Reference(name)) => assert_eq!(name, "input"),
            other => panic!("expected reference to input, got {:?}", other),
        }
    }

    #[test]
    fn silu_fusion_reversed_mul_inputs() {
        // mul(sigmoid(x), x) — sigmoid output as first arg
        let mut block = Block::new();
        block.add_op(
            Operation::new("sigmoid", "sig_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("sig_out"),
        );
        block.add_op(
            Operation::new("mul", "mul_0")
                .with_input("x", Value::Reference("sig_out".into()))
                .with_input("y", Value::Reference("input".into()))
                .with_output("mul_out"),
        );
        block.outputs.push("mul_out".into());

        let mut program = program_with_block(block);
        OpSubstitutionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "silu");
    }

    #[test]
    fn silu_fusion_preserves_downstream_refs() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("sigmoid", "sig_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("sig_out"),
        );
        block.add_op(
            Operation::new("mul", "mul_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("y", Value::Reference("sig_out".into()))
                .with_output("mul_out"),
        );
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("mul_out".into()))
                .with_input("y", Value::Reference("input".into()))
                .with_output("add_out"),
        );
        block.outputs.push("add_out".into());

        let mut program = program_with_block(block);
        OpSubstitutionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 2); // silu + add
        assert_eq!(ops[0].op_type, "silu");

        let add_op = &ops[1];
        match add_op.inputs.get("x") {
            Some(Value::Reference(name)) => assert_eq!(name, "mul_out"),
            other => panic!("expected reference to mul_out, got {:?}", other),
        }
    }

    #[test]
    fn silu_fusion_skipped_when_sigmoid_has_multiple_consumers() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("sigmoid", "sig_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("sig_out"),
        );
        block.add_op(
            Operation::new("mul", "mul_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("y", Value::Reference("sig_out".into()))
                .with_output("mul_out"),
        );
        // Another consumer of sig_out
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("sig_out".into()))
                .with_input("y", Value::Reference("input".into()))
                .with_output("add_out"),
        );
        block.outputs.push("mul_out".into());
        block.outputs.push("add_out".into());

        let mut program = program_with_block(block);
        OpSubstitutionPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        // Should NOT fuse — sigmoid is used by both mul and add
        assert!(
            ops.iter().any(|o| o.op_type == "sigmoid"),
            "sigmoid should remain when multi-consumed"
        );
        assert!(
            ops.iter().any(|o| o.op_type == "mul"),
            "mul should remain when fusion is skipped"
        );
    }
}
