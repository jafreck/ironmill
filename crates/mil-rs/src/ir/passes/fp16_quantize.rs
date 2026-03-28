//! FP16 quantization pass.
//!
//! Converts FP32 weights and tensor types to FP16 throughout a MIL Program.
//! This is a straightforward truncation (no calibration data needed).

use half::f16;

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::ScalarType;
use crate::ir::types::Value;

/// Convert FP32 weights and activations to FP16.
///
/// Updates:
/// - `const` operations with FP32 tensor data → FP16 tensor data
/// - [`TensorType`](crate::ir::TensorType) annotations from Float32 → Float16
/// - Function input types from Float32 → Float16
pub struct Fp16QuantizePass;

impl Pass for Fp16QuantizePass {
    fn name(&self) -> &str {
        "fp16-quantization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            // Convert function input types.
            for (_name, ty) in &mut function.inputs {
                if ty.scalar_type == ScalarType::Float32 {
                    ty.scalar_type = ScalarType::Float16;
                }
            }

            // Convert operations in the function body.
            for op in &mut function.body.operations {
                for value in op.inputs.values_mut() {
                    quantize_value(value);
                }
                for value in op.attributes.values_mut() {
                    quantize_value(value);
                }
            }
        }
        Ok(())
    }
}

/// Recursively quantize a [`Value`] from FP32 to FP16 where applicable.
fn quantize_value(value: &mut Value) {
    match value {
        Value::Tensor {
            data,
            shape: _,
            dtype,
        } if *dtype == ScalarType::Float32 => {
            *data = fp32_to_fp16_bytes(data);
            *dtype = ScalarType::Float16;
        }
        Value::Type(ty) if ty.scalar_type == ScalarType::Float32 => {
            ty.scalar_type = ScalarType::Float16;
        }
        Value::List(items) => {
            for item in items {
                quantize_value(item);
            }
        }
        _ => {}
    }
}

/// Convert raw FP32 bytes to FP16 bytes using truncation.
///
/// Reads the input as little-endian `f32` values, converts each to `f16`
/// via [`f16::from_f32`], and returns the resulting bytes.
fn fp32_to_fp16_bytes(data: &[u8]) -> Vec<u8> {
    debug_assert!(
        data.len() % 4 == 0,
        "FP32 tensor data length must be a multiple of 4"
    );

    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(4) {
        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let h = f16::from_f32(f);
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::Function;
    use crate::ir::tensor::TensorType;

    /// Helper: build a `const` op with a tensor value.
    fn const_tensor_op(name: &str, output: &str, value: Value) -> Operation {
        Operation::new("const", name)
            .with_input("val", value)
            .with_output(output)
    }

    /// Create FP32 tensor bytes from a slice of f32 values.
    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn converts_fp32_const_to_fp16() {
        let fp32_data = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(fp32_data.len(), 16);

        let tensor_val = Value::Tensor {
            data: fp32_data,
            shape: vec![4],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        Fp16QuantizePass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        match op.inputs.get("val") {
            Some(Value::Tensor { data, shape, dtype }) => {
                assert_eq!(*dtype, ScalarType::Float16);
                assert_eq!(*shape, vec![4]);
                // FP16 is 2 bytes per element → 8 bytes total.
                assert_eq!(data.len(), 8);

                // Verify round-trip values.
                let values: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected FP16 Tensor, got {other:?}"),
        }
    }

    #[test]
    fn converts_function_inputs_from_fp32_to_fp16() {
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);

        let mut program = Program::new("1.0.0");
        let func = Function::new("main").with_input("image", input_ty);
        program.add_function(func);

        Fp16QuantizePass.run(&mut program).unwrap();

        let main = &program.functions["main"];
        assert_eq!(main.inputs[0].1.scalar_type, ScalarType::Float16);
        // Shape should be unchanged.
        assert_eq!(
            main.inputs[0].1.shape,
            vec![Some(1), Some(3), Some(224), Some(224)]
        );
    }

    #[test]
    fn leaves_non_float_ops_unchanged() {
        let int_val = Value::Tensor {
            data: vec![1, 0, 0, 0], // i32 = 1
            shape: vec![1],
            dtype: ScalarType::Int32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op("idx", "idx_out", int_val));
        // Also add a non-tensor const.
        func.body.add_op(
            Operation::new("const", "bool_c")
                .with_input("val", Value::Bool(true))
                .with_output("b_out"),
        );
        func.body.outputs.push("idx_out".into());
        program.add_function(func);

        Fp16QuantizePass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        // Int tensor should be untouched.
        match ops[0].inputs.get("val") {
            Some(Value::Tensor { dtype, data, .. }) => {
                assert_eq!(*dtype, ScalarType::Int32);
                assert_eq!(data.len(), 4);
            }
            other => panic!("expected Int32 Tensor, got {other:?}"),
        }
        // Bool const should be untouched.
        match ops[1].inputs.get("val") {
            Some(Value::Bool(true)) => {}
            other => panic!("expected Bool(true), got {other:?}"),
        }
    }

    #[test]
    fn does_not_double_convert_fp16() {
        let fp16_data: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        let original_data = fp16_data.clone();

        let tensor_val = Value::Tensor {
            data: fp16_data,
            shape: vec![2],
            dtype: ScalarType::Float16,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op("w", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        Fp16QuantizePass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        match op.inputs.get("val") {
            Some(Value::Tensor { data, dtype, .. }) => {
                assert_eq!(*dtype, ScalarType::Float16);
                assert_eq!(*data, original_data);
            }
            other => panic!("expected FP16 Tensor, got {other:?}"),
        }
    }

    #[test]
    fn converts_tensor_type_annotations_in_attributes() {
        let type_val = Value::Type(TensorType::new(ScalarType::Float32, vec![1, 10]));

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(
            Operation::new("cast", "cast_0")
                .with_attr("dtype", type_val)
                .with_input("x", Value::Reference("input".into()))
                .with_output("cast_out"),
        );
        func.body.outputs.push("cast_out".into());
        program.add_function(func);

        Fp16QuantizePass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        match op.attributes.get("dtype") {
            Some(Value::Type(ty)) => {
                assert_eq!(ty.scalar_type, ScalarType::Float16);
                assert_eq!(ty.shape, vec![Some(1), Some(10)]);
            }
            other => panic!("expected Type(Float16), got {other:?}"),
        }
    }
}
