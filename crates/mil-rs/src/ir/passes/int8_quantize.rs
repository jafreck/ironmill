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
use crate::error::{MilError, Result};
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::{TensorData, Value};

/// Quantization granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
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
        let provider = program.weight_provider.clone();
        let spill_index = program.spill_index.clone();
        let resolve = super::util::make_resolver(&provider, &spill_index);

        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                if op.op_type != "const" {
                    continue;
                }

                // val may live in inputs or attributes depending on the
                // frontend (ONNX import puts it in attributes).
                let in_inputs = matches!(
                    op.inputs.get("val"),
                    Some(Value::Tensor {
                        dtype: ScalarType::Float32,
                        ..
                    })
                );
                let in_attrs = !in_inputs
                    && matches!(
                        op.attributes.get("val"),
                        Some(Value::Tensor {
                            dtype: ScalarType::Float32,
                            ..
                        })
                    );

                if !in_inputs && !in_attrs {
                    continue;
                }

                let val = if in_inputs {
                    op.inputs
                        .remove("val")
                        .ok_or_else(|| MilError::Validation("missing val in inputs".into()))?
                } else {
                    op.attributes
                        .remove("val")
                        .ok_or_else(|| MilError::Validation("missing val in attributes".into()))?
                };

                if let Value::Tensor {
                    mut data,
                    shape,
                    dtype: _,
                } = val
                {
                    data.materialize_with(|key| resolve(key))?;
                    let floats =
                        tensor_as_f32_slice(data.as_bytes().expect("tensor not materialized"));

                    let use_per_channel = self.granularity == Granularity::PerChannel
                        && shape.len() >= 2
                        && shape[0] > 1;

                    if use_per_channel {
                        let num_channels = shape[0];
                        let channel_size: usize = shape[1..].iter().product();
                        let mut all_quantized = Vec::with_capacity(floats.len());
                        let mut scales = Vec::with_capacity(num_channels);
                        let mut zero_points = Vec::with_capacity(num_channels);

                        for ch in 0..num_channels {
                            let start = ch * channel_size;
                            let end = start + channel_size;
                            let channel_slice = &floats[start..end];
                            let (q, s, zp) = quantize_f32_to_uint8(channel_slice);
                            all_quantized.extend_from_slice(&q);
                            scales.push(s);
                            zero_points.push(zp);
                        }

                        let quantized_val = Value::Tensor {
                            data: TensorData::Inline(all_quantized),
                            shape: shape.clone(),
                            dtype: ScalarType::UInt8,
                        };

                        op.op_type = "constexpr_affine_dequantize".to_string();
                        op.inputs.remove("val");
                        op.attributes.remove("val");
                        op.attributes
                            .insert("quantized_data".to_string(), quantized_val);

                        let scale_bytes: Vec<u8> =
                            scales.iter().flat_map(|s| s.to_le_bytes()).collect();
                        op.attributes.insert(
                            "scale".to_string(),
                            Value::Tensor {
                                data: TensorData::Inline(scale_bytes),
                                shape: vec![num_channels],
                                dtype: ScalarType::Float32,
                            },
                        );

                        let zp_bytes: Vec<u8> =
                            zero_points.iter().flat_map(|z| z.to_le_bytes()).collect();
                        op.attributes.insert(
                            "zero_point".to_string(),
                            Value::Tensor {
                                data: TensorData::Inline(zp_bytes),
                                shape: vec![num_channels],
                                dtype: ScalarType::Float32,
                            },
                        );
                        op.attributes.insert("axis".to_string(), Value::Int(0));

                        let out_type = TensorType::new(ScalarType::Float32, shape);
                        if let Some(slot) = op.output_types.get_mut(0) {
                            *slot = Some(out_type);
                        } else {
                            op.output_types.push(Some(out_type));
                        }
                    } else {
                        // Per-tensor quantization (original behavior).
                        let (quantized, scale, zero_point) = quantize_f32_to_uint8(&floats);

                        let quantized_val = Value::Tensor {
                            data: TensorData::Inline(quantized),
                            shape: shape.clone(),
                            dtype: ScalarType::UInt8,
                        };

                        op.op_type = "constexpr_affine_dequantize".to_string();
                        op.inputs.remove("val");
                        op.attributes.remove("val");
                        op.attributes
                            .insert("quantized_data".to_string(), quantized_val);
                        op.attributes
                            .insert("scale".to_string(), Value::Float(scale as f64));
                        op.attributes
                            .insert("zero_point".to_string(), Value::Float(zero_point as f64));
                        op.attributes.insert("axis".to_string(), Value::Int(0));

                        let out_type = TensorType::new(ScalarType::Float32, shape);
                        if let Some(slot) = op.output_types.get_mut(0) {
                            *slot = Some(out_type);
                        } else {
                            op.output_types.push(Some(out_type));
                        }
                    }

                    // INT8 quantization always uses 8-bit width.
                    op.attributes.insert("bit_width".to_string(), Value::Int(8));
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
    use crate::ir::passes::test_helpers::{const_tensor_op, f32_bytes};
    use crate::ir::program::Function;

    #[test]
    fn quantizes_fp32_const_to_uint8_with_attrs() {
        let fp32_data = f32_bytes(&[0.0, 1.0, 2.0, 3.0]);
        let tensor_val = Value::Tensor {
            data: TensorData::Inline(fp32_data),
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
        assert_eq!(op.op_type, "constexpr_affine_dequantize");
        match op.attributes.get("quantized_data") {
            Some(Value::Tensor { data, shape, dtype }) => {
                assert_eq!(*dtype, ScalarType::UInt8);
                assert_eq!(*shape, vec![4]);
                assert_eq!(data.byte_len(), 4);
                assert!(!data.as_bytes().expect("tensor not materialized").is_empty());
            }
            other => panic!("expected UInt8 Tensor, got {other:?}"),
        }

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
            Some(Value::Float(_)) => {}
            other => panic!("expected Float zero_point, got {other:?}"),
        }
    }

    #[test]
    fn quantized_values_within_uint8_range() {
        let vals = [-5.0_f32, -1.0, 0.0, 2.5, 10.0];
        let fp32_data = f32_bytes(&vals);
        let tensor_val = Value::Tensor {
            data: TensorData::Inline(fp32_data),
            shape: vec![5],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op("w", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        Int8QuantizePass::weight_only().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        if let Some(Value::Tensor { data, .. }) = op.attributes.get("quantized_data") {
            let data = data.as_bytes().expect("tensor not materialized");
            assert_eq!(data.len(), 5);
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
            data: TensorData::Inline(fp32_data),
            shape: vec![8],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op("w", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        Int8QuantizePass::weight_only().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        if let Some(Value::Tensor { data, dtype, .. }) = op.attributes.get("quantized_data") {
            assert_eq!(*dtype, ScalarType::UInt8);
            let data = data.as_bytes().expect("tensor not materialized");
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
            data: TensorData::Inline(fp32_data),
            shape: vec![1],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op("w", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        Int8QuantizePass::weight_only().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        if let Some(Value::Tensor { data, dtype, shape }) = op.attributes.get("quantized_data") {
            assert_eq!(*dtype, ScalarType::UInt8);
            assert_eq!(*shape, vec![1]);
            assert_eq!(data.byte_len(), 1);
        } else {
            panic!("expected Tensor");
        }
    }

    #[test]
    fn leaves_non_fp32_tensors_unchanged() {
        let int_val = Value::Tensor {
            data: TensorData::Inline(vec![1, 0, 0, 0]),
            shape: vec![1],
            dtype: ScalarType::Int32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op("idx", "idx_out", int_val));
        func.body.outputs.push("idx_out".into());
        program.add_function(func);

        Int8QuantizePass::weight_only().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        match op.inputs.get("val") {
            Some(Value::Tensor { dtype, data, .. }) => {
                assert_eq!(*dtype, ScalarType::Int32);
                assert_eq!(data.byte_len(), 4);
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
    fn per_channel_quantization_produces_per_channel_scales() {
        // 2 channels, 3 elements each → shape [2, 3]
        let channel_0 = [0.0_f32, 1.0, 2.0];
        let channel_1 = [-3.0_f32, 0.0, 6.0];
        let mut all_vals = Vec::new();
        all_vals.extend_from_slice(&channel_0);
        all_vals.extend_from_slice(&channel_1);
        let fp32_data = f32_bytes(&all_vals);

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(fp32_data),
            shape: vec![2, 3],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        Int8QuantizePass::new(None, Granularity::PerChannel)
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");

        // Scale should be a Tensor with shape [2] (one per channel).
        match op.attributes.get("scale") {
            Some(Value::Tensor { shape, dtype, data }) => {
                assert_eq!(*shape, vec![2]);
                assert_eq!(*dtype, ScalarType::Float32);
                assert_eq!(data.byte_len(), 8); // 2 floats * 4 bytes
            }
            other => panic!("expected per-channel scale Tensor, got {other:?}"),
        }

        // Zero-point should also be a Tensor with shape [2].
        match op.attributes.get("zero_point") {
            Some(Value::Tensor { shape, dtype, data }) => {
                assert_eq!(*shape, vec![2]);
                assert_eq!(*dtype, ScalarType::Float32);
                assert_eq!(data.byte_len(), 8);
            }
            other => panic!("expected per-channel zero_point Tensor, got {other:?}"),
        }

        // Quantized data should be 6 bytes (2 channels * 3 elements).
        match op.attributes.get("quantized_data") {
            Some(Value::Tensor { data, shape, dtype }) => {
                assert_eq!(*dtype, ScalarType::UInt8);
                assert_eq!(*shape, vec![2, 3]);
                assert_eq!(data.byte_len(), 6);
            }
            other => panic!("expected UInt8 Tensor, got {other:?}"),
        }
    }

    #[test]
    fn per_channel_falls_back_to_per_tensor_for_1d() {
        // 1D tensor: per-channel not applicable, should use per-tensor.
        let fp32_data = f32_bytes(&[1.0, 2.0, 3.0]);
        let tensor_val = Value::Tensor {
            data: TensorData::Inline(fp32_data),
            shape: vec![3],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        Int8QuantizePass::new(None, Granularity::PerChannel)
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");

        // Should fall back to per-tensor (scalar scale).
        match op.attributes.get("scale") {
            Some(Value::Float(s)) => assert!(*s > 0.0),
            other => panic!("expected per-tensor Float scale, got {other:?}"),
        }
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
