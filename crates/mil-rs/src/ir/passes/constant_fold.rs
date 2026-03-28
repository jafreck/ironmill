//! Constant folding pass.

use std::collections::HashMap;

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::types::Value;

/// Evaluates operations with all-constant inputs at compile time.
///
/// Currently handles scalar arithmetic on `Int` and `Float` constants:
/// - `add`, `sub`, `mul`, `real_div`
///
/// The pass runs to a fixed point so that chains of constant arithmetic
/// are fully collapsed.
pub struct ConstantFoldPass;

impl Pass for ConstantFoldPass {
    fn name(&self) -> &str {
        "constant-folding"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            let block = &mut function.body;

            loop {
                // 1. Build a map from value name → constant Value for all `const` ops.
                let mut constants: HashMap<String, Value> = HashMap::new();
                for op in &block.operations {
                    if op.op_type == "const" {
                        if let Some(val) = op.inputs.get("val").or_else(|| op.attributes.get("val"))
                        {
                            for out in &op.outputs {
                                constants.insert(out.clone(), val.clone());
                            }
                        }
                    }
                }

                // 2. Find the first foldable op and fold it.
                let mut folded = false;
                for i in 0..block.operations.len() {
                    if let Some(replacement) = try_fold(&block.operations[i], &constants) {
                        block.operations[i] = replacement;
                        folded = true;
                        break;
                    }
                }

                if !folded {
                    break;
                }
            }
        }
        Ok(())
    }
}

/// Attempt to fold `op` into a `const` op. Returns `Some(replacement)` on success.
fn try_fold(op: &Operation, constants: &HashMap<String, Value>) -> Option<Operation> {
    match op.op_type.as_str() {
        "add" | "sub" | "mul" | "real_div" => fold_binary_arithmetic(op, constants),
        _ => None,
    }
}

/// Fold a binary arithmetic op with two scalar-constant operands.
fn fold_binary_arithmetic(
    op: &Operation,
    constants: &HashMap<String, Value>,
) -> Option<Operation> {
    let lhs = resolve_scalar(op.inputs.get("x")?, constants)?;
    let rhs = resolve_scalar(op.inputs.get("y")?, constants)?;
    let result = eval_binary(op.op_type.as_str(), lhs, rhs)?;

    let output_name = op.outputs.first()?;
    let mut replacement = Operation::new("const", &op.name);
    replacement.inputs.insert("val".into(), result);
    replacement.outputs.push(output_name.clone());
    Some(replacement)
}

/// A resolved scalar constant.
#[derive(Debug, Clone, Copy)]
enum Scalar {
    Int(i64),
    Float(f64),
}

/// Resolve a [`Value`] to a [`Scalar`], following references through `constants`.
fn resolve_scalar(value: &Value, constants: &HashMap<String, Value>) -> Option<Scalar> {
    match value {
        Value::Int(n) => Some(Scalar::Int(*n)),
        Value::Float(f) => Some(Scalar::Float(*f)),
        Value::Reference(name) => {
            let resolved = constants.get(name)?;
            match resolved {
                Value::Int(n) => Some(Scalar::Int(*n)),
                Value::Float(f) => Some(Scalar::Float(*f)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Evaluate a binary arithmetic operation on two scalars.
fn eval_binary(op_type: &str, lhs: Scalar, rhs: Scalar) -> Option<Value> {
    match (lhs, rhs) {
        (Scalar::Int(a), Scalar::Int(b)) => {
            let result = match op_type {
                "add" => a.checked_add(b)?,
                "sub" => a.checked_sub(b)?,
                "mul" => a.checked_mul(b)?,
                "real_div" => {
                    if b == 0 {
                        return None;
                    }
                    // Integer division via real_div produces a float.
                    return Some(Value::Float(a as f64 / b as f64));
                }
                _ => return None,
            };
            Some(Value::Int(result))
        }
        (Scalar::Float(a), Scalar::Float(b)) => {
            let result = match op_type {
                "add" => a + b,
                "sub" => a - b,
                "mul" => a * b,
                "real_div" => {
                    if b == 0.0 {
                        return None;
                    }
                    a / b
                }
                _ => return None,
            };
            Some(Value::Float(result))
        }
        // Mixed int/float: promote to float.
        (Scalar::Int(a), Scalar::Float(b)) => eval_binary(op_type, Scalar::Float(a as f64), Scalar::Float(b)),
        (Scalar::Float(a), Scalar::Int(b)) => eval_binary(op_type, Scalar::Float(a), Scalar::Float(b as f64)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::program::Function;

    /// Helper: build a `const` op with a scalar value.
    fn const_op(name: &str, output: &str, val: Value) -> Operation {
        Operation::new("const", name)
            .with_input("val", val)
            .with_output(output)
    }

    #[test]
    fn folds_int_add() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        func.body.add_op(const_op("c1", "two", Value::Int(2)));
        func.body.add_op(const_op("c2", "three", Value::Int(3)));
        func.body.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("two".into()))
                .with_input("y", Value::Reference("three".into()))
                .with_output("sum"),
        );
        func.body.outputs.push("sum".into());

        program.add_function(func);
        ConstantFoldPass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        assert_eq!(ops.len(), 3);
        // The add should now be a const.
        let folded = &ops[2];
        assert_eq!(folded.op_type, "const");
        assert_eq!(folded.outputs, vec!["sum"]);
        match folded.inputs.get("val") {
            Some(Value::Int(5)) => {} // correct
            other => panic!("expected Int(5), got {:?}", other),
        }
    }

    #[test]
    fn folds_float_mul() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        func.body
            .add_op(const_op("c1", "a", Value::Float(2.5)));
        func.body
            .add_op(const_op("c2", "b", Value::Float(4.0)));
        func.body.add_op(
            Operation::new("mul", "mul_0")
                .with_input("x", Value::Reference("a".into()))
                .with_input("y", Value::Reference("b".into()))
                .with_output("product"),
        );
        func.body.outputs.push("product".into());

        program.add_function(func);
        ConstantFoldPass.run(&mut program).unwrap();

        let folded = &program.functions["main"].body.operations[2];
        assert_eq!(folded.op_type, "const");
        match folded.inputs.get("val") {
            Some(Value::Float(v)) => assert!((v - 10.0).abs() < 1e-12),
            other => panic!("expected Float(10.0), got {:?}", other),
        }
    }

    #[test]
    fn does_not_fold_non_constant_inputs() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        func.body.add_op(const_op("c1", "two", Value::Int(2)));
        func.body.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("two".into()))
                .with_input("y", Value::Reference("runtime_val".into()))
                .with_output("result"),
        );
        func.body.outputs.push("result".into());

        program.add_function(func);
        ConstantFoldPass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        assert_eq!(ops[1].op_type, "add");
    }

    #[test]
    fn chains_constant_folds() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        // const(1) + const(2) = const(3), then const(3) * const(4) = const(12)
        func.body.add_op(const_op("c1", "one", Value::Int(1)));
        func.body.add_op(const_op("c2", "two", Value::Int(2)));
        func.body.add_op(const_op("c3", "four", Value::Int(4)));
        func.body.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("one".into()))
                .with_input("y", Value::Reference("two".into()))
                .with_output("three"),
        );
        func.body.add_op(
            Operation::new("mul", "mul_0")
                .with_input("x", Value::Reference("three".into()))
                .with_input("y", Value::Reference("four".into()))
                .with_output("result"),
        );
        func.body.outputs.push("result".into());

        program.add_function(func);
        ConstantFoldPass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        // add_0 is now const, mul_0 is now const
        let last = ops.last().unwrap();
        assert_eq!(last.op_type, "const");
        assert_eq!(last.outputs, vec!["result"]);
        match last.inputs.get("val") {
            Some(Value::Int(12)) => {} // correct
            other => panic!("expected Int(12), got {:?}", other),
        }
    }
}
