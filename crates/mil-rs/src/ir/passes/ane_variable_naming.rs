//! ANE variable naming pass.
//!
//! ANE requires I/O IOSurfaces to be ordered alphabetically by their
//! MIL variable names (constraints #3, #13). This pass renames I/O
//! variables so alphabetical order matches the intended logical order.

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;

use super::replace_reference;

/// Rename I/O variables for alphabetical ordering compatibility with ANE.
///
/// Uses naming scheme: `a_input0`, `a_input1` for inputs and
/// `z_output0`, `z_output1` for outputs.
pub struct AneVariableNamingPass;

impl Pass for AneVariableNamingPass {
    fn name(&self) -> &str {
        "ane-variable-naming"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            // --- Phase 1: Rename inputs ---
            let input_renames: Vec<(String, String)> = function
                .inputs
                .iter()
                .enumerate()
                .map(|(i, (old_name, _ty))| (old_name.clone(), format!("a_input{i}")))
                .collect();

            for (old, new) in &input_renames {
                if old != new {
                    replace_reference(&mut function.body, old, new);
                }
            }

            // Update function input names.
            for (i, (old, _)) in input_renames.iter().enumerate() {
                if function.inputs[i].0 == *old {
                    function.inputs[i].0 = input_renames[i].1.clone();
                }
            }

            // --- Phase 2: Rename outputs ---
            let output_renames: Vec<(String, String)> = function
                .body
                .outputs
                .iter()
                .enumerate()
                .map(|(i, old_name)| (old_name.clone(), format!("z_output{i}")))
                .collect();

            for (old, new) in &output_renames {
                if old == new {
                    continue;
                }

                // Find the op that produces this output and rename it.
                for op in &mut function.body.operations {
                    for out in &mut op.outputs {
                        if out == old {
                            *out = new.clone();
                        }
                    }
                }

                // Update all downstream references (inputs of later ops).
                replace_reference(&mut function.body, old, new);
            }

            // Block outputs were already updated by replace_reference (it
            // updates block.outputs), but ensure they're correct.
            for (i, (_old, new)) in output_renames.iter().enumerate() {
                function.body.outputs[i] = new.clone();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::{Block, Function};
    use crate::ir::tensor::{ScalarType, TensorType};
    use crate::ir::types::Value;

    fn program_with_function(func: Function) -> Program {
        let mut program = Program::new("1.0.0");
        program.add_function(func);
        program
    }

    #[test]
    fn pass_name() {
        assert_eq!(AneVariableNamingPass.name(), "ane-variable-naming");
    }

    #[test]
    fn ane_variable_naming_renames_inputs() {
        let ty = TensorType::new(ScalarType::Float16, vec![1, 64, 1, 512]);
        let mut func = Function::new("main")
            .with_input("hidden_states", ty.clone())
            .with_input("attention_mask", ty.clone());

        let mut block = Block::new();
        block.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("hidden_states".into()))
                .with_input("y", Value::Reference("attention_mask".into()))
                .with_output("add_out"),
        );
        block.outputs.push("add_out".into());
        func.body = block;

        let mut program = program_with_function(func);
        AneVariableNamingPass.run(&mut program).unwrap();

        let f = &program.functions["main"];
        assert_eq!(f.inputs[0].0, "a_input0");
        assert_eq!(f.inputs[1].0, "a_input1");

        // References in ops should be updated.
        let add_op = &f.body.operations[0];
        assert_eq!(
            add_op.inputs.get("x"),
            Some(&Value::Reference("a_input0".into()))
        );
        assert_eq!(
            add_op.inputs.get("y"),
            Some(&Value::Reference("a_input1".into()))
        );
    }

    #[test]
    fn ane_variable_naming_renames_outputs() {
        let ty = TensorType::new(ScalarType::Float16, vec![1, 64, 1, 512]);
        let mut func = Function::new("main").with_input("x", ty);

        let mut block = Block::new();
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("x".into()))
                .with_output("hidden"),
        );
        block.add_op(
            Operation::new("sigmoid", "sig_0")
                .with_input("x", Value::Reference("hidden".into()))
                .with_output("logits"),
        );
        block.outputs.push("logits".into());
        func.body = block;

        let mut program = program_with_function(func);
        AneVariableNamingPass.run(&mut program).unwrap();

        let f = &program.functions["main"];
        // Output should be renamed.
        assert_eq!(f.body.outputs, vec!["z_output0"]);
        // The op producing the output should have its output renamed.
        let sig_op = f
            .body
            .operations
            .iter()
            .find(|o| o.name == "sig_0")
            .unwrap();
        assert_eq!(sig_op.outputs[0], "z_output0");
    }

    #[test]
    fn ane_variable_naming_multiple_outputs() {
        let ty = TensorType::new(ScalarType::Float16, vec![1, 64, 1, 512]);
        let mut func = Function::new("main").with_input("x", ty);

        let mut block = Block::new();
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("x".into()))
                .with_output("out_a"),
        );
        block.add_op(
            Operation::new("sigmoid", "sig_0")
                .with_input("x", Value::Reference("x".into()))
                .with_output("out_b"),
        );
        block.outputs.push("out_a".into());
        block.outputs.push("out_b".into());
        func.body = block;

        let mut program = program_with_function(func);
        AneVariableNamingPass.run(&mut program).unwrap();

        let f = &program.functions["main"];
        assert_eq!(f.body.outputs, vec!["z_output0", "z_output1"]);
        assert_eq!(f.inputs[0].0, "a_input0");
    }

    #[test]
    fn ane_variable_naming_no_inputs_no_crash() {
        let mut func = Function::new("main");
        let mut block = Block::new();
        block.add_op(
            Operation::new("const", "c0")
                .with_input("val", Value::Float(1.0))
                .with_output("one"),
        );
        block.outputs.push("one".into());
        func.body = block;

        let mut program = program_with_function(func);
        AneVariableNamingPass.run(&mut program).unwrap();

        let f = &program.functions["main"];
        assert!(f.inputs.is_empty());
        assert_eq!(f.body.outputs, vec!["z_output0"]);
    }

    #[test]
    fn ane_variable_naming_idempotent() {
        let ty = TensorType::new(ScalarType::Float16, vec![1, 64, 1, 512]);
        let mut func = Function::new("main").with_input("x", ty);

        let mut block = Block::new();
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("x".into()))
                .with_output("y"),
        );
        block.outputs.push("y".into());
        func.body = block;

        let mut program = program_with_function(func);
        AneVariableNamingPass.run(&mut program).unwrap();
        // Run again — should be stable.
        AneVariableNamingPass.run(&mut program).unwrap();

        let f = &program.functions["main"];
        assert_eq!(f.inputs[0].0, "a_input0");
        assert_eq!(f.body.outputs, vec!["z_output0"]);
    }
}
