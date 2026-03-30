//! ANE matmul→conv conversion pass.
//!
//! Converts `matmul(x, weight_const)` to `conv(x, weight)` with 1×1
//! convolution. ANE linear layers must use conv, not matmul — Orion's
//! benchmarks show 3× faster execution with conv on ANE.
//!
//! Weight is transposed from `[1, Cin, 1, Cout]` (ANE layout for matmul)
//! to `[Cout, Cin, 1, 1]` (standard conv weight format).
//!
//! This pass runs after `AneLayoutPass` (which converts to 4D ANE shapes)
//! and before the attention split.

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::{Block, Program};
use crate::ir::types::Value;

/// Converts `matmul(x, weight_const)` to 1×1 `conv` for ANE execution.
///
/// Only converts matmul ops where the `y` input references a const tensor
/// with shape `[1, Cin, 1, Cout]` (post-AneLayout). The weight data is
/// transposed to `[Cout, Cin, 1, 1]` and conv parameter consts are added.
pub struct AneMatmulToConvPass;

impl Pass for AneMatmulToConvPass {
    fn name(&self) -> &str {
        "ane-matmul-to-conv"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            convert_matmul_to_conv(&mut function.body);
        }
        Ok(())
    }
}

fn convert_matmul_to_conv(block: &mut Block) {
    let mut i = 0;
    while i < block.operations.len() {
        if block.operations[i].op_type != "matmul" {
            i += 1;
            continue;
        }

        // The weight must be a reference to a const tensor.
        let y_ref = match block.operations[i].inputs.get("y") {
            Some(Value::Reference(name)) => name.clone(),
            _ => {
                i += 1;
                continue;
            }
        };

        // Find the const op and extract weight info.
        // Weights may be in `inputs["val"]` (hand-built IR) or
        // `attributes["val"]` (ONNX converter).
        let weight_info = block.operations.iter().find_map(|o| {
            if o.op_type == "const" && o.outputs.first().map(|s| s.as_str()) == Some(&y_ref) {
                let val = o.inputs.get("val").or_else(|| o.attributes.get("val"));
                match val {
                    Some(Value::Tensor { data, shape, dtype }) => {
                        Some((data.clone(), shape.clone(), *dtype))
                    }
                    _ => None,
                }
            } else {
                None
            }
        });

        let (weight_data, weight_shape, dtype) = match weight_info {
            Some(info) => info,
            None => {
                i += 1;
                continue;
            }
        };

        // Verify weight shape and extract Cin/Cout.
        // Post-AneLayout, weight types show [1,Cin,1,Cout] but the raw tensor
        // data retains its original shape. Handle both 2D [Cin,Cout] and
        // 4D [1,Cin,1,Cout] formats.
        let (cin, cout) = match weight_shape.as_slice() {
            [cin, cout] => (*cin, *cout),
            [1, cin, 1, cout] => (*cin, *cout),
            _ => {
                i += 1;
                continue;
            }
        };
        let element_size = dtype.byte_size();

        // Transpose weight: [1, Cin, 1, Cout] → [Cout, Cin, 1, 1]
        let transposed = transpose_weight(&weight_data, cin, cout, element_size);

        // Capture matmul info before mutating.
        let base = block.operations[i].name.clone();
        let x_input = match block.operations[i].inputs.get("x").cloned() {
            Some(v) => v,
            None => {
                i += 1;
                continue;
            }
        };
        let output_name = match block.operations[i].outputs.first().cloned() {
            Some(v) => v,
            None => {
                i += 1;
                continue;
            }
        };
        let output_types = block.operations[i].output_types.clone();

        // Build replacement ops.
        let new_ops = build_conv_replacement(
            &base,
            x_input,
            transposed,
            cin,
            cout,
            dtype,
            &output_name,
            output_types,
        );
        let count = new_ops.len();

        // Remove matmul, insert conv + consts at same position.
        // The orphaned weight const will be cleaned up by DCE.
        block.operations.remove(i);
        for (j, new_op) in new_ops.into_iter().enumerate() {
            block.operations.insert(i + j, new_op);
        }

        i += count;
    }
}

/// Transpose weight data from row-major `[1, Cin, 1, Cout]` to `[Cout, Cin, 1, 1]`.
///
/// In the source layout, element `[0, c_in, 0, c_out]` is at byte offset
/// `(c_in * Cout + c_out) * element_size`. In the target layout, element
/// `[c_out, c_in, 0, 0]` is at `(c_out * Cin + c_in) * element_size`.
fn transpose_weight(data: &[u8], cin: usize, cout: usize, element_size: usize) -> Vec<u8> {
    let expected_len = cin * cout * element_size;
    if data.len() < expected_len {
        // Data too short — return as-is (will likely fail at compile time
        // with a clear error rather than panicking here).
        return data.to_vec();
    }

    let mut result = vec![0u8; expected_len];
    for c_in in 0..cin {
        for c_out in 0..cout {
            let src = (c_in * cout + c_out) * element_size;
            let dst = (c_out * cin + c_in) * element_size;
            result[dst..dst + element_size].copy_from_slice(&data[src..src + element_size]);
        }
    }
    result
}

/// Build the conv op and its parameter consts, matching Orion's `orion_mil_linear`.
#[allow(clippy::too_many_arguments)]
fn build_conv_replacement(
    base: &str,
    x_input: Value,
    weight_data: Vec<u8>,
    cin: usize,
    cout: usize,
    dtype: crate::ir::ScalarType,
    output_name: &str,
    output_types: Vec<Option<crate::ir::TensorType>>,
) -> Vec<Operation> {
    let pt_name = format!("{base}_conv_pt");
    let st_name = format!("{base}_conv_st");
    let pd_name = format!("{base}_conv_pd");
    let dl_name = format!("{base}_conv_dl");
    let gr_name = format!("{base}_conv_gr");
    let w_name = format!("{base}_conv_W");

    let pt = Operation::new("const", &pt_name)
        .with_input("val", Value::String("valid".into()))
        .with_output(&pt_name);

    let st = Operation::new("const", &st_name)
        .with_input("val", Value::List(vec![Value::Int(1), Value::Int(1)]))
        .with_output(&st_name);

    let pd = Operation::new("const", &pd_name)
        .with_input(
            "val",
            Value::List(vec![
                Value::Int(0),
                Value::Int(0),
                Value::Int(0),
                Value::Int(0),
            ]),
        )
        .with_output(&pd_name);

    let dl = Operation::new("const", &dl_name)
        .with_input("val", Value::List(vec![Value::Int(1), Value::Int(1)]))
        .with_output(&dl_name);

    let gr = Operation::new("const", &gr_name)
        .with_input("val", Value::Int(1))
        .with_output(&gr_name);

    let w = Operation::new("const", &w_name)
        .with_input(
            "val",
            Value::Tensor {
                data: weight_data,
                shape: vec![cout, cin, 1, 1],
                dtype,
            },
        )
        .with_output(&w_name);

    let mut conv = Operation::new("conv", format!("{base}_conv"))
        .with_input("x", x_input)
        .with_input("weight", Value::Reference(w_name.clone()))
        .with_input("pad_type", Value::Reference(pt_name.clone()))
        .with_input("strides", Value::Reference(st_name.clone()))
        .with_input("pad", Value::Reference(pd_name.clone()))
        .with_input("dilations", Value::Reference(dl_name.clone()))
        .with_input("groups", Value::Reference(gr_name.clone()))
        .with_output(output_name);
    conv.output_types = output_types;

    vec![pt, st, pd, dl, gr, w, conv]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ScalarType;
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
    fn matmul_with_const_weight_converted_to_conv() {
        let mut block = Block::new();

        // Weight const: [4, 8] (Cin=4, Cout=8) — 2D as ONNX stores it
        let weight_data = vec![0u8; 4 * 8 * 2]; // fp16
        let mut w_op = Operation::new("const", "W");
        w_op.attributes.insert(
            "val".into(),
            Value::Tensor {
                data: weight_data,
                shape: vec![4, 8],
                dtype: ScalarType::Float16,
            },
        );
        w_op.outputs.push("W".into());
        w_op.output_types.push(None);
        block.add_op(w_op);

        // matmul(x, W)
        block.add_op(
            Operation::new("matmul", "mm_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("y", Value::Reference("W".into()))
                .with_output("mm_out"),
        );

        block.outputs.push("mm_out".into());

        let mut program = program_with_block(block);
        AneMatmulToConvPass.run(&mut program).unwrap();

        let ops = block_ops(&program);

        // Original const W remains (orphaned, cleaned by DCE).
        // New ops: 6 conv param consts + 1 conv = 7 new + 1 old const = 8
        assert_eq!(
            ops.len(),
            8,
            "expected 8 ops (1 old const + 7 new), got {}",
            ops.len()
        );

        // The conv op should be last of the new ops.
        let conv_op = ops.iter().find(|o| o.op_type == "conv").unwrap();
        assert_eq!(conv_op.outputs[0], "mm_out");

        // Weight should be [Cout=8, Cin=4, 1, 1]
        let new_w = ops.iter().find(|o| o.name == "mm_0_conv_W").unwrap();
        match new_w.inputs.get("val") {
            Some(Value::Tensor { shape, .. }) => {
                assert_eq!(shape, &[8, 4, 1, 1]);
            }
            other => panic!("expected tensor, got {:?}", other),
        }

        // Block output should still be "mm_out".
        assert_eq!(program.functions["main"].body.outputs, vec!["mm_out"]);
    }

    #[test]
    fn matmul_without_const_weight_not_converted() {
        let mut block = Block::new();
        block.add_op(
            Operation::new("matmul", "mm_0")
                .with_input("x", Value::Reference("input_a".into()))
                .with_input("y", Value::Reference("input_b".into()))
                .with_output("mm_out"),
        );
        block.outputs.push("mm_out".into());

        let mut program = program_with_block(block);
        AneMatmulToConvPass.run(&mut program).unwrap();

        let ops = block_ops(&program);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, "matmul");
    }

    #[test]
    fn weight_transpose_correctness() {
        // 2×3 matrix (Cin=2, Cout=3), fp16 (2 bytes each)
        // Source [1,2,1,3]: row-major = [[a,b,c], [d,e,f]]
        //   (0,0)=a (0,1)=b (0,2)=c (1,0)=d (1,1)=e (1,2)=f
        // Target [3,2,1,1]: [[a,d], [b,e], [c,f]]
        //   (0,0)=a (0,1)=d (1,0)=b (1,1)=e (2,0)=c (2,1)=f
        let src: Vec<u8> = vec![
            1, 0, // a
            2, 0, // b
            3, 0, // c
            4, 0, // d
            5, 0, // e
            6, 0, // f
        ];
        let result = transpose_weight(&src, 2, 3, 2);
        let expected: Vec<u8> = vec![
            1, 0, // a
            4, 0, // d
            2, 0, // b
            5, 0, // e
            3, 0, // c
            6, 0, // f
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn downstream_refs_preserved_after_conversion() {
        let mut block = Block::new();

        let weight_data = vec![0u8; 4 * 8 * 2];
        let mut w_op = Operation::new("const", "W");
        w_op.attributes.insert(
            "val".into(),
            Value::Tensor {
                data: weight_data,
                shape: vec![4, 8],
                dtype: ScalarType::Float16,
            },
        );
        w_op.outputs.push("W".into());
        w_op.output_types.push(None);
        block.add_op(w_op);

        block.add_op(
            Operation::new("matmul", "mm_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("y", Value::Reference("W".into()))
                .with_output("mm_out"),
        );

        block.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("mm_out".into()))
                .with_output("relu_out"),
        );

        block.outputs.push("relu_out".into());

        let mut program = program_with_block(block);
        AneMatmulToConvPass.run(&mut program).unwrap();

        // relu should still reference "mm_out" which is now the conv output.
        let relu = block_ops(&program)
            .iter()
            .find(|o| o.op_type == "relu")
            .unwrap();
        match relu.inputs.get("x") {
            Some(Value::Reference(name)) => assert_eq!(name, "mm_out"),
            other => panic!("expected reference to mm_out, got {:?}", other),
        }
    }
}
