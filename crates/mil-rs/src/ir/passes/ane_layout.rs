//! ANE layout pass — reshape all tensors to `[1, C, 1, S]`.
//!
//! ANE requires this specific 4D layout for all tensors. This pass
//! transforms shape metadata so all tensor types conform to the ANE
//! convention. It does **not** change actual data layout — that is
//! handled by the MIL text emitter.

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::TensorType;
use crate::ir::types::Value;

/// Reshape all tensors in the program to ANE's required `[1, C, 1, S]` layout.
pub struct AneLayoutPass;

impl Pass for AneLayoutPass {
    fn name(&self) -> &str {
        "ane-layout"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            // Reshape function input types (activations).
            for (_name, ty) in &mut function.inputs {
                reshape_tensor_type(ty);
            }

            // Reshape operation output types (activations).
            // Skip reshaping const op tensor values — weights use their own
            // layout convention (e.g., [Cout, Cin, kH, kW] for conv).
            for op in &mut function.body.operations {
                for t in op.output_types.iter_mut().flatten() {
                    reshape_tensor_type(t);
                }

                // Only reshape non-const op input tensors.
                if op.op_type != "const" {
                    for value in op.inputs.values_mut() {
                        reshape_value_tensor(value);
                    }
                    for value in op.attributes.values_mut() {
                        reshape_value_tensor(value);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Reshape a `Value::Tensor`'s shape (or recurse into lists).
fn reshape_value_tensor(value: &mut Value) {
    match value {
        Value::Tensor { shape, .. } => {
            *shape = to_ane_shape_static(shape);
        }
        Value::List(items) => {
            for item in items {
                reshape_value_tensor(item);
            }
        }
        _ => {}
    }
}

/// Convert a static shape to ANE 4D `[1, C, 1, S]`.
fn to_ane_shape_static(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1, 1, 1],
        1 => vec![1, shape[0], 1, 1],
        2 => vec![1, shape[0], 1, shape[1]],
        3 => {
            let (b, m, n) = (shape[0], shape[1], shape[2]);
            if b == 1 {
                vec![1, m, 1, n]
            } else {
                vec![1, b * m, 1, n]
            }
        }
        4 => {
            let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            // Already in ANE layout?
            if b == 1 && h == 1 {
                vec![1, c, 1, w]
            } else {
                vec![1, c, 1, h * w]
            }
        }
        _ => {
            // 5D+ — collapse all dims after the second into the sequence dim.
            let c = shape[1];
            let s: usize = shape[2..].iter().product();
            vec![1, c, 1, s]
        }
    }
}

/// Reshape a `TensorType`'s shape to ANE 4D.
fn reshape_tensor_type(ty: &mut TensorType) {
    // Only reshape when all dims are static.
    if !ty.is_static() {
        // Best-effort: if rank != 4, pad/fold what we can.
        ty.shape = to_ane_shape_dynamic(&ty.shape);
        return;
    }

    let static_shape: Vec<usize> = ty.shape.iter().map(|d| d.unwrap()).collect();
    let new = to_ane_shape_static(&static_shape);
    ty.shape = new.into_iter().map(Some).collect();
}

/// Convert a potentially-dynamic shape to ANE 4D.
fn to_ane_shape_dynamic(shape: &[Option<usize>]) -> Vec<Option<usize>> {
    match shape.len() {
        0 => vec![Some(1), Some(1), Some(1), Some(1)],
        1 => vec![Some(1), shape[0], Some(1), Some(1)],
        2 => vec![Some(1), shape[0], Some(1), shape[1]],
        3 => {
            // If batch dim is known to be 1, straightforward.
            if shape[0] == Some(1) {
                vec![Some(1), shape[1], Some(1), shape[2]]
            } else {
                // Cannot statically compute B*M — keep channel as dynamic.
                vec![Some(1), None, Some(1), shape[2]]
            }
        }
        4 => {
            if shape[2] == Some(1) {
                // Already [_, C, 1, S].
                vec![Some(1), shape[1], Some(1), shape[3]]
            } else {
                // Cannot statically compute H*W — keep seq as dynamic.
                vec![Some(1), shape[1], Some(1), None]
            }
        }
        _ => {
            vec![Some(1), shape[1], Some(1), None]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::{Block, Function};
    use crate::ir::tensor::ScalarType;

    fn program_with_block(block: Block) -> Program {
        let mut func = Function::new("main");
        func.body = block;
        let mut program = Program::new("1.0.0");
        program.add_function(func);
        program
    }

    #[test]
    fn pass_name() {
        assert_eq!(AneLayoutPass.name(), "ane-layout");
    }

    #[test]
    fn ane_layout_1d_input() {
        let func =
            Function::new("main").with_input("x", TensorType::new(ScalarType::Float16, vec![128]));
        let mut program = Program::new("1.0.0");
        program.add_function(func);

        AneLayoutPass.run(&mut program).unwrap();

        let shape = &program.functions["main"].inputs[0].1.shape;
        assert_eq!(shape, &vec![Some(1), Some(128), Some(1), Some(1)]);
    }

    #[test]
    fn ane_layout_2d_input() {
        let func = Function::new("main")
            .with_input("x", TensorType::new(ScalarType::Float16, vec![64, 512]));
        let mut program = Program::new("1.0.0");
        program.add_function(func);

        AneLayoutPass.run(&mut program).unwrap();

        let shape = &program.functions["main"].inputs[0].1.shape;
        assert_eq!(shape, &vec![Some(1), Some(64), Some(1), Some(512)]);
    }

    #[test]
    fn ane_layout_3d_batch1() {
        let func = Function::new("main")
            .with_input("x", TensorType::new(ScalarType::Float16, vec![1, 64, 512]));
        let mut program = Program::new("1.0.0");
        program.add_function(func);

        AneLayoutPass.run(&mut program).unwrap();

        let shape = &program.functions["main"].inputs[0].1.shape;
        assert_eq!(shape, &vec![Some(1), Some(64), Some(1), Some(512)]);
    }

    #[test]
    fn ane_layout_3d_batch_gt1() {
        let func = Function::new("main")
            .with_input("x", TensorType::new(ScalarType::Float16, vec![2, 64, 512]));
        let mut program = Program::new("1.0.0");
        program.add_function(func);

        AneLayoutPass.run(&mut program).unwrap();

        let shape = &program.functions["main"].inputs[0].1.shape;
        assert_eq!(shape, &vec![Some(1), Some(128), Some(1), Some(512)]);
    }

    #[test]
    fn ane_layout_4d_spatial_flatten() {
        let func = Function::new("main")
            .with_input("x", TensorType::new(ScalarType::Float16, vec![1, 32, 7, 7]));
        let mut program = Program::new("1.0.0");
        program.add_function(func);

        AneLayoutPass.run(&mut program).unwrap();

        let shape = &program.functions["main"].inputs[0].1.shape;
        assert_eq!(shape, &vec![Some(1), Some(32), Some(1), Some(49)]);
    }

    #[test]
    fn ane_layout_already_4d_no_change() {
        let func = Function::new("main").with_input(
            "x",
            TensorType::new(ScalarType::Float16, vec![1, 64, 1, 512]),
        );
        let mut program = Program::new("1.0.0");
        program.add_function(func);

        AneLayoutPass.run(&mut program).unwrap();

        let shape = &program.functions["main"].inputs[0].1.shape;
        assert_eq!(shape, &vec![Some(1), Some(64), Some(1), Some(512)]);
    }

    #[test]
    fn ane_layout_op_output_types() {
        let mut block = Block::new();
        let mut op = Operation::new("relu", "relu_0").with_output("out");
        op.output_types = vec![Some(TensorType::new(ScalarType::Float16, vec![64, 512]))];
        block.add_op(op);
        block.outputs.push("out".into());

        let mut program = program_with_block(block);
        AneLayoutPass.run(&mut program).unwrap();

        let oty = program.functions["main"].body.operations[0].output_types[0]
            .as_ref()
            .unwrap();
        assert_eq!(oty.shape, vec![Some(1), Some(64), Some(1), Some(512)]);
    }

    #[test]
    fn ane_layout_const_tensor_shape_preserved() {
        // Const op tensor values (weights) should NOT be reshaped —
        // they use their own layout convention (e.g., [Cout, Cin, kH, kW]).
        let mut block = Block::new();
        let op = Operation::new("const", "w")
            .with_input(
                "val",
                Value::Tensor {
                    data: vec![0; 16],
                    shape: vec![4, 4],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output("w");
        block.add_op(op);
        block.outputs.push("w".into());

        let mut program = program_with_block(block);
        AneLayoutPass.run(&mut program).unwrap();

        match program.functions["main"].body.operations[0]
            .inputs
            .get("val")
        {
            Some(Value::Tensor { shape, .. }) => {
                // Shape should NOT be reshaped — preserved as-is.
                assert_eq!(shape, &vec![4, 4]);
            }
            other => panic!("expected Tensor value, got {:?}", other),
        }
    }
}
