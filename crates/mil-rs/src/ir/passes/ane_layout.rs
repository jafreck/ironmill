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

            // Fix reshape ops: update the shape parameter to match the
            // ANE 4D output type. The layout pass converts output types
            // to 4D but doesn't touch the shape const that reshape reads.
            for op in &mut function.body.operations {
                if op.op_type == "reshape" {
                    if let Some(Some(out_ty)) = op.output_types.first() {
                        let target_shape: Vec<usize> =
                            out_ty.shape.iter().map(|d| d.unwrap_or(1)).collect();
                        // Replace the shape input with an inline tensor
                        // containing the 4D target shape.
                        let n = target_shape.len();
                        let data: Vec<u8> = target_shape
                            .iter()
                            .flat_map(|&d| (d as i32).to_le_bytes())
                            .collect();
                        op.inputs.insert(
                            "shape".to_string(),
                            Value::Tensor {
                                data,
                                shape: vec![n],
                                dtype: crate::ir::ScalarType::Int32,
                            },
                        );
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
            if b == 1 && m == 1 {
                // [1, 1, N] — single-token decode: N is hidden_size, put in C.
                vec![1, n, 1, 1]
            } else if b == 1 {
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
            if shape[0] == Some(1) && shape[1] == Some(1) {
                // [1, 1, N] — single-token decode: N is hidden_size, put in C.
                vec![Some(1), shape[2], Some(1), Some(1)]
            } else if shape[0] == Some(1) {
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

    #[test]
    fn reshape_op_shape_updated_to_4d() {
        // A reshape from [1, 1024] to [1, 1, 128] should get its shape
        // input updated to [1, 1, 1, 128] to match the ANE 4D output type.
        let mut block = Block::new();

        let shape_data: Vec<u8> = [1i32, 1, 128]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let reshape_op = Operation::new("reshape", "reshape_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input(
                "shape",
                Value::Tensor {
                    data: shape_data,
                    shape: vec![3],
                    dtype: ScalarType::Int32,
                },
            )
            .with_output("reshaped");
        let mut reshape_op = reshape_op;
        reshape_op.output_types = vec![Some(TensorType::new(ScalarType::Float16, vec![1, 1, 128]))];

        block.add_op(reshape_op);
        block.outputs.push("reshaped".into());

        let input_ty = TensorType::new(ScalarType::Float16, vec![1, 1024]);
        let func = Function::new("main").with_input("input", input_ty);
        let mut func = func;
        func.body = block;
        let mut program = Program::new("1.0");
        program.add_function(func);

        AneLayoutPass.run(&mut program).unwrap();

        // The reshape shape input should now be 4D matching the output type.
        // [1, 1, 128] with b=1, m=1 maps to [1, 128, 1, 1] (hidden in C).
        let op = &program.functions["main"].body.operations[0];
        match op.inputs.get("shape") {
            Some(Value::Tensor { data, shape, dtype }) => {
                assert_eq!(*dtype, ScalarType::Int32);
                assert_eq!(shape, &vec![4]); // 1-D tensor with 4 elements
                let values: Vec<i32> = data
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                assert_eq!(values, vec![1, 128, 1, 1]);
            }
            other => panic!("expected Tensor shape input, got {:?}", other),
        }
    }
}
