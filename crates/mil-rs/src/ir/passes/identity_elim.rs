//! Identity elimination pass.

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::types::Value;

use super::replace_reference;

/// Removes identity/no-op operations and rewires their consumers.
///
/// Targets:
/// - Operations with `op_type == "identity"` — the single input is forwarded
///   directly to consumers of the output.
/// - `transpose` operations whose permutation is the identity `[0, 1, 2, …]`.
pub struct IdentityEliminationPass;

impl Pass for IdentityEliminationPass {
    fn name(&self) -> &str {
        "identity-elimination"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            let block = &mut function.body;

            // Gather (output_name, input_name) pairs for identity-like ops.
            let mut rewrites: Vec<(String, String)> = Vec::new();
            let mut remove_indices: Vec<usize> = Vec::new();

            for (idx, op) in block.operations.iter().enumerate() {
                if let Some((out, inp)) = detect_identity(op) {
                    rewrites.push((out, inp));
                    remove_indices.push(idx);
                }
            }

            // Apply rewrites: replace all references to the identity output
            // with the identity input.
            for (old_name, new_name) in &rewrites {
                replace_reference(block, old_name, new_name);
            }

            // Remove identity ops in reverse order to keep indices stable.
            for idx in remove_indices.into_iter().rev() {
                block.operations.remove(idx);
            }
        }
        Ok(())
    }
}

/// If `op` is an identity-like operation, return `(output_name, input_name)`.
fn detect_identity(op: &crate::ir::operation::Operation) -> Option<(String, String)> {
    let output = op.outputs.first()?;

    match op.op_type.as_str() {
        "identity" => {
            // The canonical input key is "x".
            let input_ref = op.inputs.get("x")?;
            if let Value::Reference(input_name) = input_ref {
                Some((output.clone(), input_name.clone()))
            } else {
                None
            }
        }
        "transpose" => {
            // Identity transpose has perm = [0, 1, 2, …].
            let perm = op
                .inputs
                .get("perm")
                .or_else(|| op.attributes.get("perm"))?;
            if is_identity_perm(perm) {
                let input_ref = op.inputs.get("x")?;
                if let Value::Reference(input_name) = input_ref {
                    Some((output.clone(), input_name.clone()))
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Returns `true` if `perm` is the identity permutation `[0, 1, 2, …]`.
fn is_identity_perm(perm: &Value) -> bool {
    if let Value::List(items) = perm {
        items.iter().enumerate().all(|(i, v)| match v {
            Value::Int(n) => *n as usize == i,
            _ => false,
        })
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::Function;

    #[test]
    fn removes_identity_and_rewires() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        let relu = Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("input".into()))
            .with_output("relu_out");
        let identity = Operation::new("identity", "id_0")
            .with_input("x", Value::Reference("relu_out".into()))
            .with_output("id_out");
        let softmax = Operation::new("softmax", "softmax_0")
            .with_input("x", Value::Reference("id_out".into()))
            .with_output("softmax_out");

        func.body.add_op(relu);
        func.body.add_op(identity);
        func.body.add_op(softmax);
        func.body.outputs.push("softmax_out".into());

        program.add_function(func);

        IdentityEliminationPass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].name, "relu_0");
        assert_eq!(ops[1].name, "softmax_0");
        // softmax should now reference relu_out directly
        if let Some(Value::Reference(name)) = ops[1].inputs.get("x") {
            assert_eq!(name, "relu_out");
        } else {
            panic!("expected Reference input on softmax");
        }
    }

    #[test]
    fn removes_trivial_transpose() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        let relu = Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("input".into()))
            .with_output("relu_out");
        let transpose = Operation::new("transpose", "transpose_0")
            .with_input("x", Value::Reference("relu_out".into()))
            .with_input(
                "perm",
                Value::List(vec![Value::Int(0), Value::Int(1), Value::Int(2)]),
            )
            .with_output("t_out");
        let softmax = Operation::new("softmax", "softmax_0")
            .with_input("x", Value::Reference("t_out".into()))
            .with_output("softmax_out");

        func.body.add_op(relu);
        func.body.add_op(transpose);
        func.body.add_op(softmax);
        func.body.outputs.push("softmax_out".into());

        program.add_function(func);

        IdentityEliminationPass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].name, "relu_0");
        assert_eq!(ops[1].name, "softmax_0");
        if let Some(Value::Reference(name)) = ops[1].inputs.get("x") {
            assert_eq!(name, "relu_out");
        } else {
            panic!("expected Reference input on softmax");
        }
    }

    #[test]
    fn keeps_non_trivial_transpose() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        let transpose = Operation::new("transpose", "transpose_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input(
                "perm",
                Value::List(vec![Value::Int(0), Value::Int(2), Value::Int(1)]),
            )
            .with_output("t_out");

        func.body.add_op(transpose);
        func.body.outputs.push("t_out".into());

        program.add_function(func);

        IdentityEliminationPass.run(&mut program).unwrap();

        assert_eq!(program.functions["main"].body.operations.len(), 1);
    }

    #[test]
    fn rewires_block_output() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        let relu = Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("input".into()))
            .with_output("relu_out");
        let identity = Operation::new("identity", "id_0")
            .with_input("x", Value::Reference("relu_out".into()))
            .with_output("id_out");

        func.body.add_op(relu);
        func.body.add_op(identity);
        func.body.outputs.push("id_out".into());

        program.add_function(func);

        IdentityEliminationPass.run(&mut program).unwrap();

        let block = &program.functions["main"].body;
        assert_eq!(block.operations.len(), 1);
        assert_eq!(block.outputs, vec!["relu_out"]);
    }
}
