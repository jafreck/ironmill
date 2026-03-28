//! Dead code elimination pass.

use std::collections::HashSet;

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::{Block, Program};
use crate::ir::types::Value;

/// Removes operations whose outputs are not consumed by any other operation
/// or listed as block outputs.
///
/// The pass runs to a fixed point — it keeps removing dead ops until the
/// graph stabilises, since removing one dead op can make others dead.
pub struct DeadCodeEliminationPass;

impl Pass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "dead-code-elimination"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            eliminate_dead_code(&mut function.body);
        }
        Ok(())
    }
}

/// Collect all value names that appear as [`Value::Reference`] inside a value.
fn collect_refs_from_value(value: &Value, refs: &mut HashSet<String>) {
    match value {
        Value::Reference(name) => {
            refs.insert(name.clone());
        }
        Value::List(items) => {
            for item in items {
                collect_refs_from_value(item, refs);
            }
        }
        _ => {}
    }
}

/// Run dead-code elimination on a single block to a fixed point.
fn eliminate_dead_code(block: &mut Block) {
    loop {
        // 1. Gather every value name that is *used* — either by an op input or
        //    as a block output.
        let mut used: HashSet<String> = block.outputs.iter().cloned().collect();
        for op in &block.operations {
            for value in op.inputs.values() {
                collect_refs_from_value(value, &mut used);
            }
        }

        // 2. Remove ops whose outputs are ALL unused.
        let before = block.operations.len();
        block
            .operations
            .retain(|op| op.outputs.iter().any(|o| used.contains(o)));

        // 3. Stop when nothing was removed.
        if block.operations.len() == before {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;

    #[test]
    fn removes_unused_op() {
        let mut program = Program::new("1.0.0");
        let mut func = crate::ir::program::Function::new("main");

        // Two ops: relu_0 is consumed by softmax_0; add_0 is dead.
        let relu = Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("input".into()))
            .with_output("relu_out");
        let add = Operation::new("add", "add_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input("y", Value::Reference("input".into()))
            .with_output("add_out");
        let softmax = Operation::new("softmax", "softmax_0")
            .with_input("x", Value::Reference("relu_out".into()))
            .with_output("softmax_out");

        func.body.add_op(relu);
        func.body.add_op(add);
        func.body.add_op(softmax);
        func.body.outputs.push("softmax_out".into());

        program.add_function(func);

        DeadCodeEliminationPass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].name, "relu_0");
        assert_eq!(ops[1].name, "softmax_0");
    }

    #[test]
    fn keeps_all_when_everything_used() {
        let mut program = Program::new("1.0.0");
        let mut func = crate::ir::program::Function::new("main");

        let relu = Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("input".into()))
            .with_output("relu_out");

        func.body.add_op(relu);
        func.body.outputs.push("relu_out".into());

        program.add_function(func);

        DeadCodeEliminationPass.run(&mut program).unwrap();

        assert_eq!(program.functions["main"].body.operations.len(), 1);
    }

    #[test]
    fn cascading_removal() {
        let mut program = Program::new("1.0.0");
        let mut func = crate::ir::program::Function::new("main");

        // a -> b -> c (output). If we remove nothing initially, but suppose
        // we add a dead branch: a -> d (dead).
        let a = Operation::new("relu", "a")
            .with_input("x", Value::Reference("input".into()))
            .with_output("a_out");
        let b = Operation::new("relu", "b")
            .with_input("x", Value::Reference("a_out".into()))
            .with_output("b_out");
        // d only consumes b_out but d_out is unused → d is dead,
        // but b is still alive because its output feeds d. Once d is removed,
        // if b_out is no longer needed... but b_out is used by c as well.
        // Let's make a true cascade: d depends on a_out, e depends on d_out.
        // Neither d nor e are block outputs.
        let d = Operation::new("relu", "d")
            .with_input("x", Value::Reference("a_out".into()))
            .with_output("d_out");
        let e = Operation::new("relu", "e")
            .with_input("x", Value::Reference("d_out".into()))
            .with_output("e_out");

        func.body.add_op(a);
        func.body.add_op(d);
        func.body.add_op(e);
        func.body.add_op(b);
        func.body.outputs.push("b_out".into());

        program.add_function(func);

        DeadCodeEliminationPass.run(&mut program).unwrap();

        let names: Vec<&str> = program.functions["main"]
            .body
            .operations
            .iter()
            .map(|op| op.name.as_str())
            .collect();
        assert_eq!(names, vec!["a", "b"]);
    }
}
