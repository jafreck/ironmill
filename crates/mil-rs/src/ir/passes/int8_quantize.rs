//! INT8 post-training quantization pass.
//!
//! Quantizes FP32 const weight tensors to UINT8 using min/max range scaling.
//! Phase 1 implements weight-only quantization (no calibration data needed).
//!
//! For each FP32 const tensor the pass computes:
//!   scale      = (max - min) / 255
//!   zero_point = round(-min / scale)
//!   q[i]       = clamp(round(x[i] / scale) + zero_point, 0, 255)
//!
//! The quantized bytes replace the original data, the dtype becomes UInt8,
//! and `scale` / `zero_point` are stored as operation attributes so that
//! downstream consumers (or a dequantize step) can recover approximate FP32
//! values.

use std::path::PathBuf;

use super::tensor_utils::tensor_as_f32_slice;
use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::ScalarType;
use crate::ir::types::Value;

/// Quantization granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// One scale/zero-point pair for the entire tensor.
    PerTensor,
    /// One scale/zero-point pair per output channel (not yet implemented).
    PerChannel,
}

/// INT8 post-training quantization with optional calibration data.
///
/// When `calibration_dir` is `None` (the default), only weight-only
/// quantization is performed: const FP32 tensors are quantized in-place
/// using their own min/max range.
pub struct Int8QuantizePass {
    /// Directory containing calibration input tensors. `None` for weight-only.
    pub calibration_dir: Option<PathBuf>,
    /// Per-tensor or per-channel quantization.
    pub granularity: Granularity,
}

impl Int8QuantizePass {
    /// Create a weight-only per-tensor quantization pass.
    pub fn weight_only() -> Self {
        Self {
            calibration_dir: None,
            granularity: Granularity::PerTensor,
        }
    }

    /// Create a pass with explicit configuration.
    pub fn new(calibration_dir: Option<PathBuf>, granularity: Granularity) -> Self {
        Self {
            calibration_dir,
            granularity,
        }
    }
}

impl Pass for Int8QuantizePass {
    fn name(&self) -> &str {
        "int8-quantization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                // Only touch const ops whose "val" input is an FP32 tensor.
                if op.op_type != "const" {
                    continue;
                }

                let needs_quant = matches!(
                    op.inputs.get("val"),
                    Some(Value::Tensor { dtype: ScalarType::Float32, .. })
                );
                if !needs_quant {
                    continue;
                }

                // Take ownership of the tensor value to transform it.
                let val = op.inputs.remove("val").unwrap();
                if let Value::Tensor { data, shape, dtype: _ } = val {
                    let floats = tensor_as_f32_slice(&data);
                    let (quantized, scale, zero_point) = quantize_f32_to_uint8(&floats);

                    op.inputs.insert(
                        "val".to_string(),
                        Value::Tensor {
                            data: quantized,
                            shape,
                            dtype: ScalarType::UInt8,
                        },
                    );
                    op.attributes
                        .insert("scale".to_string(), Value::Float(scale as f64));
                    op.attributes
                        .insert("zero_point".to_string(), Value::Float(zero_point as f64));
                }
            }
        }
        Ok(())
    }
}

/// Quantize an f32 slice to UINT8 bytes using min/max affine quantization.
///
/// Returns `(quantized_bytes, scale, zero_point)`.
///
/// When all values are identical (min == max), scale is set to 1.0 and
/// zero_point is chosen so that the single value maps to a valid uint8.
fn quantize_f32_to_uint8(values: &[f32]) -> (Vec<u8>, f32, f32) {
    if values.is_empty() {
        return (Vec::new(), 1.0, 0.0);
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let (scale, zp_float) = if (max - min).abs() < f32::EPSILON {
        // Degenerate case: all values are the same.
        let zp = (-min).round();
        (1.0_f32, zp)
    } else {
        let s = (max - min) / 255.0;
        let zp = (-min / s).round();
        (s, zp)
    };

    let quantized: Vec<u8> = values
        .iter()
        .map(|&x| {
            let q = (x / scale + zp_float).round().clamp(0.0, 255.0);
            q as u8
        })
        .collect();

    (quantized, scale, zp_float)
}

/// Dequantize UINT8 values back to f32 using scale and zero_point.
///
/// `x_approx = (q - zero_point) * scale`
#[cfg(test)]
fn dequantize_uint8_to_f32(quantized: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    quantized
        .iter()
        .map(|&q| (q as f32 - zero_point) * scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::Function;

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
    fn quantizes_fp32_const_to_uint8_with_attrs() {
        let fp32_data = f32_bytes(&[0.0, 1.0, 2.0, 3.0]);
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

        Int8QuantizePass::weight_only().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        match op.inputs.get("val") {
            Some(Value::Tensor { data, shape, dtype }) => {
                assert_eq!(*dtype, ScalarType::UInt8);
                assert_eq!(*shape, vec![4]);
                assert_eq!(data.len(), 4); // 1 byte per element
                // All values should be in [0, 255]
                // All u8 values are inherently in [0, 255]; verify non-empty.
                assert!(!data.is_empty());
            }
            other => panic!("expected UInt8 Tensor, got {other:?}"),
        }

        // scale and zero_point attributes must exist.
        assert!(
            op.attributes.contains_key("scale"),
            "missing 'scale' attribute"
        );
        assert!(
            op.attributes.contains_key("zero_point"),
            "missing 'zero_point' attribute"
        );

        match op.attributes.get("scale") {
            Some(Value::Float(s)) => assert!(*s > 0.0, "scale must be positive"),
            other => panic!("expected Float scale, got {other:?}"),
        }
        match op.attributes.get("zero_point") {
            Some(Value::Float(_)) => {} // zero_point can be outside [0, 255]
            other => panic!("expected Float zero_point, got {other:?}"),
        }
    }

    #[test]
    fn quantized_values_within_uint8_range() {
        // Mix of negative and positive values.
        let vals = [-5.0_f32, -1.0, 0.0, 2.5, 10.0];
        let fp32_data = f32_bytes(&vals);
        let tensor_val = Value::Tensor {
            data: fp32_data,
            shape: vec![5],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("w", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        Int8QuantizePass::weight_only().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        if let Some(Value::Tensor { data, .. }) = op.inputs.get("val") {
            // u8 is inherently in [0, 255]; verify correct count.
            assert_eq!(data.len(), 5);
            // min input should map to 0, max to 255.
            assert_eq!(data[0], 0, "min value should map to 0");
            assert_eq!(data[4], 255, "max value should map to 255");
        } else {
            panic!("expected Tensor");
        }
    }

    #[test]
    fn round_trip_within_tolerance() {
        let original: Vec<f32> = vec![-3.0, -1.5, 0.0, 1.5, 3.0, 6.0];
        let (quantized, scale, zero_point) = quantize_f32_to_uint8(&original);
        let recovered = dequantize_uint8_to_f32(&quantized, scale, zero_point);

        // Tolerance: half a quantization step.
        let tol = scale / 2.0 + f32::EPSILON;
        for (orig, recov) in original.iter().zip(recovered.iter()) {
            let err = (orig - recov).abs();
            assert!(
                err <= tol,
                "round-trip error {err} exceeds tolerance {tol} for value {orig} (recovered {recov})"
            );
        }
    }

    #[test]
    fn edge_case_all_same_values() {
        let vals = [42.0_f32; 8];
        let fp32_data = f32_bytes(&vals);
        let tensor_val = Value::Tensor {
            data: fp32_data,
            shape: vec![8],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("w", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        Int8QuantizePass::weight_only().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        if let Some(Value::Tensor { data, dtype, .. }) = op.inputs.get("val") {
            assert_eq!(*dtype, ScalarType::UInt8);
            // All quantized values should be the same.
            let first = data[0];
            for &b in data {
                assert_eq!(b, first, "all-same input should produce all-same output");
            }
        } else {
            panic!("expected Tensor");
        }
    }

    #[test]
    fn edge_case_single_element() {
        let fp32_data = f32_bytes(&[7.5]);
        let tensor_val = Value::Tensor {
            data: fp32_data,
            shape: vec![1],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("w", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        Int8QuantizePass::weight_only().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        if let Some(Value::Tensor { data, dtype, shape }) = op.inputs.get("val") {
            assert_eq!(*dtype, ScalarType::UInt8);
            assert_eq!(*shape, vec![1]);
            assert_eq!(data.len(), 1);
        } else {
            panic!("expected Tensor");
        }
    }

    #[test]
    fn leaves_non_fp32_tensors_unchanged() {
        let int_val = Value::Tensor {
            data: vec![1, 0, 0, 0],
            shape: vec![1],
            dtype: ScalarType::Int32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("idx", "idx_out", int_val));
        func.body.outputs.push("idx_out".into());
        program.add_function(func);

        Int8QuantizePass::weight_only().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        match op.inputs.get("val") {
            Some(Value::Tensor { dtype, data, .. }) => {
                assert_eq!(*dtype, ScalarType::Int32);
                assert_eq!(data.len(), 4);
            }
            other => panic!("expected Int32 Tensor, got {other:?}"),
        }
    }

    #[test]
    fn leaves_non_const_ops_unchanged() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("relu_out"),
        );
        func.body.outputs.push("relu_out".into());
        program.add_function(func);

        Int8QuantizePass::weight_only().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "relu");
        assert!(op.attributes.is_empty());
    }

    #[test]
    fn negative_only_range_maps_min_to_0_max_to_255() {
        let vals = [-10.0_f32, -8.0, -5.0, -1.0];
        let (quantized, scale, zp) = quantize_f32_to_uint8(&vals);

        assert_eq!(quantized[0], 0, "min value should map to 0");
        assert_eq!(quantized[3], 255, "max value should map to 255");

        // Round-trip tolerance: half a quantization step.
        let recovered = dequantize_uint8_to_f32(&quantized, scale, zp);
        let tol = scale / 2.0 + f32::EPSILON;
        for (orig, recov) in vals.iter().zip(recovered.iter()) {
            let err = (orig - recov).abs();
            assert!(
                err <= tol,
                "round-trip error {err} exceeds tolerance {tol} for {orig} (recovered {recov})"
            );
        }
    }

    #[test]
    fn positive_only_range_maps_min_to_0_max_to_255() {
        let vals = [1.0_f32, 5.0, 8.0, 10.0];
        let (quantized, scale, zp) = quantize_f32_to_uint8(&vals);

        assert_eq!(quantized[0], 0, "min value should map to 0");
        assert_eq!(quantized[3], 255, "max value should map to 255");

        // Round-trip tolerance: half a quantization step.
        let recovered = dequantize_uint8_to_f32(&quantized, scale, zp);
        let tol = scale / 2.0 + f32::EPSILON;
        for (orig, recov) in vals.iter().zip(recovered.iter()) {
            let err = (orig - recov).abs();
            assert!(
                err <= tol,
                "round-trip error {err} exceeds tolerance {tol} for {orig} (recovered {recov})"
            );
        }
    }
}
