//! ANE concat elimination pass.
//!
//! ANE does not support the `concat` op (constraint #1). Programs
//! targeting ANE must not contain concat operations — the model splitter
//! should produce concat-free sub-programs. This pass validates that
//! invariant and returns a clear error if any concat ops remain.

use crate::error::{MilError, Result};
use crate::ir::pass::Pass;
use crate::ir::program::Program;

/// Validate that no `concat` ops remain in an ANE-targeted program.
///
/// ANE does not support `concat` (constraint #1). The model splitter
/// should partition the graph so that each ANE sub-program is concat-free.
/// This pass acts as a validation gate — it returns an error listing any
/// remaining concat ops.
pub struct AneConcatEliminationPass;

impl Pass for AneConcatEliminationPass {
    fn name(&self) -> &str {
        "ane-concat-elimination"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        let mut concat_ops = Vec::new();

        for function in program.functions.values() {
            for op in &function.body.operations {
                if op.op_type == "concat" {
                    concat_ops.push(format!("{}::{}", function.name, op.name));
                }
            }
        }

        if concat_ops.is_empty() {
            Ok(())
        } else {
            Err(MilError::UnsupportedOp(format!(
                "ANE does not support concat (constraint #1). \
                 Found {} concat op(s) that must be eliminated before ANE execution: {}",
                concat_ops.len(),
                concat_ops.join(", ")
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::{Block, Function};
    use crate::ir::types::Value;

    fn program_with_block(block: Block) -> Program {
        let mut func = Function::new("main");
        func.body = block;
        let mut program = Program::new("1.0.0");
        program.add_function(func);
        program
    }

    #[test]
    fn pass_name() {
        assert_eq!(AneConcatEliminationPass.name(), "ane-concat-elimination");
    }

    #[test]
    fn ane_concat_elim_no_concat_passes() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("relu_out"),
        );
        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        assert!(AneConcatEliminationPass.run(&mut program).is_ok());
    }

    #[test]
    fn ane_concat_elim_detects_concat() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("concat", "cat_0")
                .with_input(
                    "values",
                    Value::List(vec![
                        Value::Reference("a".into()),
                        Value::Reference("b".into()),
                    ]),
                )
                .with_attr("axis", Value::Int(0))
                .with_output("cat_out"),
        );
        block.outputs.push("cat_out".into());

        let mut program = program_with_block(block);
        let result = AneConcatEliminationPass.run(&mut program);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("concat"), "error should mention concat: {msg}");
        assert!(msg.contains("cat_0"), "error should mention op name: {msg}");
    }

    #[test]
    fn ane_concat_elim_multiple_concats() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("concat", "cat_0")
                .with_input(
                    "values",
                    Value::List(vec![
                        Value::Reference("a".into()),
                        Value::Reference("b".into()),
                    ]),
                )
                .with_output("cat_0_out"),
        );
        block.add_op(
            Operation::new("concat", "cat_1")
                .with_input(
                    "values",
                    Value::List(vec![
                        Value::Reference("c".into()),
                        Value::Reference("d".into()),
                    ]),
                )
                .with_output("cat_1_out"),
        );
        block.outputs.push("cat_1_out".into());

        let mut program = program_with_block(block);
        let result = AneConcatEliminationPass.run(&mut program);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("2 concat op(s)"), "should report count: {msg}");
    }

    #[test]
    fn ane_concat_elim_empty_program_passes() {
        let block = Block::new();
        let mut program = program_with_block(block);
        assert!(AneConcatEliminationPass.run(&mut program).is_ok());
    }
}
