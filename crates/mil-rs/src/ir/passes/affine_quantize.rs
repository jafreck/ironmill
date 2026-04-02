//! Generalized affine weight quantization pass.
//!
//! Supports configurable bit width (INT4/INT8) and per-group quantization.
//! This is a superset of [`Int8QuantizePass`] — it can reproduce INT8 per-tensor
//! and per-channel behavior while also supporting INT4 and per-group granularity.
//!
//! Quantization formula (unsigned affine):
//!   scale      = (max - min) / qmax
//!   zero_point = round(-min / scale)
//!   q[i]       = clamp(round(x[i] / scale) + zero_point, 0, qmax)
//!   x_approx   = (q[i] - zero_point) * scale
//!
//! where `qmax = 2^bits - 1` (15 for INT4, 255 for INT8).

use super::int4_pack::pack_int4;
use super::tensor_utils::tensor_as_f32_slice;
use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::passes::int8_quantize::Granularity;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::Value;

/// Quantization bit width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitWidth {
    /// 4-bit unsigned (0–15).
    Four,
    /// 8-bit unsigned (0–255).
    Eight,
}

impl BitWidth {
    /// Maximum representable value for this bit width.
    fn qmax(self) -> f32 {
        match self {
            BitWidth::Four => 15.0,
            BitWidth::Eight => 255.0,
        }
    }

    /// Return the numeric bit width (4 or 8).
    pub fn as_u8(self) -> u8 {
        match self {
            BitWidth::Four => 4,
            BitWidth::Eight => 8,
        }
    }
}

/// Generalized affine weight quantization pass.
///
/// Quantizes FP32 const weight tensors using unsigned affine min/max scaling.
/// Supports INT4 and INT8 bit widths, per-tensor, per-channel, and per-group
/// granularity.
pub struct AffineQuantizePass {
    /// Quantization bit width.
    pub bits: BitWidth,
    /// Per-group size. `None` means per-tensor or per-channel (determined by
    /// `granularity`). `Some(g)` means each group of `g` elements along the
    /// quantization axis gets its own scale/zero_point.
    pub group_size: Option<usize>,
    /// Per-tensor or per-channel base granularity. When `group_size` is set,
    /// this field is ignored (per-group overrides it).
    pub granularity: Granularity,
}

impl AffineQuantizePass {
    /// INT4 per-tensor quantization.
    pub fn int4_per_tensor() -> Self {
        Self {
            bits: BitWidth::Four,
            group_size: None,
            granularity: Granularity::PerTensor,
        }
    }

    /// INT4 per-group quantization along the reduction axis.
    pub fn int4_per_group(group_size: usize) -> Self {
        Self {
            bits: BitWidth::Four,
            group_size: Some(group_size),
            granularity: Granularity::PerTensor, // ignored when group_size is set
        }
    }

    /// INT8 per-tensor quantization (matches `Int8QuantizePass::weight_only()`).
    pub fn int8_per_tensor() -> Self {
        Self {
            bits: BitWidth::Eight,
            group_size: None,
            granularity: Granularity::PerTensor,
        }
    }

    /// Full configuration constructor.
    pub fn new(bits: BitWidth, group_size: Option<usize>, granularity: Granularity) -> Self {
        Self {
            bits,
            group_size,
            granularity,
        }
    }
}

impl Pass for AffineQuantizePass {
    fn name(&self) -> &str {
        "affine-quantization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        let qmax = self.bits.qmax();
        let bit_width = self.bits.as_u8();

        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                if op.op_type != "const" {
                    continue;
                }

                // Locate the FP32 tensor value (may be in inputs or attributes).
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
                    op.inputs.remove("val").unwrap()
                } else {
                    op.attributes.remove("val").unwrap()
                };

                if let Value::Tensor {
                    data,
                    shape,
                    dtype: _,
                } = val
                {
                    let floats = tensor_as_f32_slice(&data);

                    if let Some(group_size) = self.group_size {
                        // --- Per-group quantization ---
                        // Quantize along the last axis, partitioning into groups.
                        emit_per_group(op, &floats, &shape, group_size, qmax);
                    } else {
                        let use_per_channel = self.granularity == Granularity::PerChannel
                            && shape.len() >= 2
                            && shape[0] > 1;

                        if use_per_channel {
                            emit_per_channel(op, &floats, &shape, qmax);
                        } else {
                            emit_per_tensor(op, &floats, &shape, qmax);
                        }
                    }

                    // Pack INT4 quantized data (2 values per byte).
                    if bit_width == 4 {
                        if let Some(Value::Tensor { data, .. }) =
                            op.attributes.get_mut("quantized_data")
                        {
                            *data = pack_int4(data);
                        }
                    }

                    // Store bit_width as an attribute for downstream consumers.
                    op.attributes
                        .insert("bit_width".to_string(), Value::Int(bit_width as i64));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Quantization core
// ---------------------------------------------------------------------------

/// Quantize an f32 slice to unsigned integers using min/max affine scaling.
///
/// Returns `(quantized_bytes, scale, zero_point)`.
/// Each quantized value is stored as a single `u8` (valid for both 4-bit and
/// 8-bit — 4-bit values simply stay in range 0..=15).
pub fn quantize_affine(values: &[f32], qmax: f32) -> (Vec<u8>, f32, f32) {
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
        let s = (max - min) / qmax;
        let zp = (-min / s).round();
        (s, zp)
    };

    let quantized: Vec<u8> = values
        .iter()
        .map(|&x| {
            let q = (x / scale + zp_float).round().clamp(0.0, qmax);
            q as u8
        })
        .collect();

    (quantized, scale, zp_float)
}

/// Dequantize unsigned integer values back to f32.
#[cfg(test)]
fn dequantize_affine(quantized: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    quantized
        .iter()
        .map(|&q| (q as f32 - zero_point) * scale)
        .collect()
}

// ---------------------------------------------------------------------------
// Emission helpers — write quantized data back into the Operation
// ---------------------------------------------------------------------------

/// Emit per-tensor quantization: single scalar scale and zero_point.
fn emit_per_tensor(
    op: &mut crate::ir::operation::Operation,
    floats: &[f32],
    shape: &[usize],
    qmax: f32,
) {
    let (quantized, scale, zero_point) = quantize_affine(floats, qmax);

    let quantized_val = Value::Tensor {
        data: quantized,
        shape: shape.to_vec(),
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

    let out_type = TensorType::new(ScalarType::Float32, shape.to_vec());
    if let Some(slot) = op.output_types.get_mut(0) {
        *slot = Some(out_type);
    } else {
        op.output_types.push(Some(out_type));
    }
}

/// Emit per-channel quantization: one scale/zero_point per output channel.
fn emit_per_channel(
    op: &mut crate::ir::operation::Operation,
    floats: &[f32],
    shape: &[usize],
    qmax: f32,
) {
    let num_channels = shape[0];
    let channel_size: usize = shape[1..].iter().product();
    let mut all_quantized = Vec::with_capacity(floats.len());
    let mut scales = Vec::with_capacity(num_channels);
    let mut zero_points = Vec::with_capacity(num_channels);

    for ch in 0..num_channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        let (q, s, zp) = quantize_affine(&floats[start..end], qmax);
        all_quantized.extend_from_slice(&q);
        scales.push(s);
        zero_points.push(zp);
    }

    let quantized_val = Value::Tensor {
        data: all_quantized,
        shape: shape.to_vec(),
        dtype: ScalarType::UInt8,
    };

    op.op_type = "constexpr_affine_dequantize".to_string();
    op.inputs.remove("val");
    op.attributes.remove("val");
    op.attributes
        .insert("quantized_data".to_string(), quantized_val);

    let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    op.attributes.insert(
        "scale".to_string(),
        Value::Tensor {
            data: scale_bytes,
            shape: vec![num_channels],
            dtype: ScalarType::Float32,
        },
    );

    let zp_bytes: Vec<u8> = zero_points.iter().flat_map(|z| z.to_le_bytes()).collect();
    op.attributes.insert(
        "zero_point".to_string(),
        Value::Tensor {
            data: zp_bytes,
            shape: vec![num_channels],
            dtype: ScalarType::Float32,
        },
    );
    op.attributes.insert("axis".to_string(), Value::Int(0));

    let out_type = TensorType::new(ScalarType::Float32, shape.to_vec());
    if let Some(slot) = op.output_types.get_mut(0) {
        *slot = Some(out_type);
    } else {
        op.output_types.push(Some(out_type));
    }
}

/// Emit per-group quantization.
///
/// The weight tensor is partitioned along the last axis into contiguous groups
/// of `group_size` elements. Each group gets its own scale and zero_point.
///
/// Output layout:
///   - `quantized_data`: same shape as input, dtype UInt8
///   - `scale`: shape `[..outer_dims, n_groups]`, dtype Float32
///   - `zero_point`: shape `[..outer_dims, n_groups]`, dtype Float32
///   - `axis`: last axis index
fn emit_per_group(
    op: &mut crate::ir::operation::Operation,
    floats: &[f32],
    shape: &[usize],
    group_size: usize,
    qmax: f32,
) {
    assert!(group_size > 0, "group_size must be positive");

    let ndim = shape.len();
    let last_dim = if ndim > 0 { shape[ndim - 1] } else { 1 };
    let outer_count: usize = if ndim > 1 {
        shape[..ndim - 1].iter().product()
    } else {
        1
    };
    let n_groups = last_dim.div_ceil(group_size);

    let mut all_quantized = Vec::with_capacity(floats.len());
    let mut all_scales = Vec::with_capacity(outer_count * n_groups);
    let mut all_zero_points = Vec::with_capacity(outer_count * n_groups);

    for row in 0..outer_count {
        let row_start = row * last_dim;
        for g in 0..n_groups {
            let g_start = row_start + g * group_size;
            let g_end = (g_start + group_size).min(row_start + last_dim);
            let group_slice = &floats[g_start..g_end];
            let (q, s, zp) = quantize_affine(group_slice, qmax);
            all_quantized.extend_from_slice(&q);
            all_scales.push(s);
            all_zero_points.push(zp);
        }
    }

    let quantized_val = Value::Tensor {
        data: all_quantized,
        shape: shape.to_vec(),
        dtype: ScalarType::UInt8,
    };

    // Scale/zero_point shape: replace last dim with n_groups.
    let mut param_shape = shape.to_vec();
    if let Some(last) = param_shape.last_mut() {
        *last = n_groups;
    }

    let scale_bytes: Vec<u8> = all_scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let zp_bytes: Vec<u8> = all_zero_points
        .iter()
        .flat_map(|z| z.to_le_bytes())
        .collect();

    let axis = if ndim > 0 { (ndim - 1) as i64 } else { 0 };

    op.op_type = "constexpr_affine_dequantize".to_string();
    op.inputs.remove("val");
    op.attributes.remove("val");
    op.attributes
        .insert("quantized_data".to_string(), quantized_val);
    op.attributes.insert(
        "scale".to_string(),
        Value::Tensor {
            data: scale_bytes,
            shape: param_shape.clone(),
            dtype: ScalarType::Float32,
        },
    );
    op.attributes.insert(
        "zero_point".to_string(),
        Value::Tensor {
            data: zp_bytes,
            shape: param_shape,
            dtype: ScalarType::Float32,
        },
    );
    op.attributes.insert("axis".to_string(), Value::Int(axis));

    // Store group_size as an attribute for downstream consumers.
    op.attributes
        .insert("group_size".to_string(), Value::Int(group_size as i64));

    let out_type = TensorType::new(ScalarType::Float32, shape.to_vec());
    if let Some(slot) = op.output_types.get_mut(0) {
        *slot = Some(out_type);
    } else {
        op.output_types.push(Some(out_type));
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::passes::int8_quantize::Int8QuantizePass;
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

    /// Build a single-const-op program for testing.
    fn make_program(values: &[f32], shape: Vec<usize>) -> Program {
        let tensor_val = Value::Tensor {
            data: f32_bytes(values),
            shape,
            dtype: ScalarType::Float32,
        };
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);
        program
    }

    /// Extract quantized bytes from the first op's `quantized_data` attribute.
    fn get_quantized_data(program: &Program) -> &[u8] {
        let op = &program.functions["main"].body.operations[0];
        match op.attributes.get("quantized_data") {
            Some(Value::Tensor { data, .. }) => data,
            other => panic!("expected quantized_data Tensor, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // INT4 per-tensor tests
    // -----------------------------------------------------------------------

    #[test]
    fn int4_per_tensor_basic() {
        use crate::ir::passes::int4_pack::unpack_int4;

        let values = [0.0_f32, 1.0, 2.0, 3.0];
        let mut program = make_program(&values, vec![4]);

        AffineQuantizePass::int4_per_tensor()
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");

        let data = get_quantized_data(&program);
        // 4 INT4 values packed into 2 bytes.
        assert_eq!(data.len(), 2);
        let unpacked = unpack_int4(data, 4);
        // min=0, max=3 → scale=3/15=0.2, zp=round(0/0.2)=0
        // q = [0, 5, 10, 15]
        assert_eq!(unpacked[0], 0);
        assert_eq!(unpacked[3], 15);

        // All unpacked values should be in [0, 15].
        for &b in &unpacked {
            assert!(b <= 15, "INT4 value {b} exceeds 15");
        }
    }

    #[test]
    fn int4_per_tensor_known_values() {
        // Carefully chosen so we can verify exact quantized values.
        // range: [-3, 12] → scale = 15/15 = 1.0, zp = round(3/1) = 3
        // q(x) = clamp(round(x/1.0) + 3, 0, 15)
        let values = [-3.0_f32, 0.0, 3.0, 6.0, 12.0];
        let (quantized, scale, zp) = quantize_affine(&values, 15.0);

        assert!(
            (scale - 1.0).abs() < 1e-6,
            "scale should be 1.0, got {scale}"
        );
        assert!((zp - 3.0).abs() < 1e-6, "zp should be 3, got {zp}");
        assert_eq!(quantized, vec![0, 3, 6, 9, 15]);
    }

    #[test]
    fn int4_per_tensor_round_trip() {
        let original = [-2.0_f32, -0.5, 0.0, 1.5, 3.0];
        let (quantized, scale, zero_point) = quantize_affine(&original, 15.0);
        let recovered = dequantize_affine(&quantized, scale, zero_point);

        let tol = scale / 2.0 + f32::EPSILON;
        for (orig, recov) in original.iter().zip(recovered.iter()) {
            let err = (orig - recov).abs();
            assert!(
                err <= tol,
                "INT4 round-trip error {err} > tol {tol} for {orig} (got {recov})"
            );
        }
    }

    #[test]
    fn int4_per_tensor_negative_range() {
        let values = [-10.0_f32, -8.0, -5.0, -1.0];
        let (quantized, scale, zp) = quantize_affine(&values, 15.0);

        assert_eq!(quantized[0], 0, "min maps to 0");
        assert_eq!(quantized[3], 15, "max maps to 15");

        let recovered = dequantize_affine(&quantized, scale, zp);
        let tol = scale / 2.0 + f32::EPSILON;
        for (orig, recov) in values.iter().zip(recovered.iter()) {
            assert!((orig - recov).abs() <= tol, "round-trip error for {orig}");
        }
    }

    // -----------------------------------------------------------------------
    // INT4 per-group tests
    // -----------------------------------------------------------------------

    #[test]
    fn int4_per_group_basic() {
        // Shape [2, 8], group_size=4 → 2 groups per row, 4 groups total.
        // Row 0: [0,1,2,3, 4,5,6,7]  → group0=[0,1,2,3], group1=[4,5,6,7]
        // Row 1: [-1,0,1,2, 10,20,30,40] → group0=[-1..2], group1=[10..40]
        #[rustfmt::skip]
        let values = [
            0.0, 1.0, 2.0, 3.0,   4.0, 5.0, 6.0, 7.0,
            -1.0, 0.0, 1.0, 2.0,  10.0, 20.0, 30.0, 40.0,
        ];
        let mut program = make_program(&values, vec![2, 8]);

        AffineQuantizePass::int4_per_group(4)
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");

        // Quantized data shape = [2, 8] (same as input), packed into 8 bytes.
        match op.attributes.get("quantized_data") {
            Some(Value::Tensor { shape, dtype, data }) => {
                use crate::ir::passes::int4_pack::unpack_int4;
                assert_eq!(*shape, vec![2, 8]);
                assert_eq!(*dtype, ScalarType::UInt8);
                assert_eq!(data.len(), 8);
                let unpacked = unpack_int4(data, 16);
                for &b in unpacked.iter() {
                    assert!(b <= 15, "INT4 value {b} exceeds 15");
                }
            }
            other => panic!("expected quantized_data Tensor, got {other:?}"),
        }

        // Scale shape = [2, 2] (2 rows, 2 groups per row).
        match op.attributes.get("scale") {
            Some(Value::Tensor { shape, dtype, data }) => {
                assert_eq!(*shape, vec![2, 2]);
                assert_eq!(*dtype, ScalarType::Float32);
                assert_eq!(data.len(), 4 * 4); // 4 float32s

                let scales: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                // Each group should have a positive scale.
                for s in &scales {
                    assert!(*s > 0.0, "scale should be positive, got {s}");
                }
                // Group 1 of row 1 (range [10,40]) should have a larger scale
                // than group 0 of row 0 (range [0,3]).
                assert!(
                    scales[3] > scales[0],
                    "wide-range group should have larger scale"
                );
            }
            other => panic!("expected scale Tensor, got {other:?}"),
        }

        // zero_point shape = [2, 2].
        match op.attributes.get("zero_point") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(*shape, vec![2, 2]);
                assert_eq!(*dtype, ScalarType::Float32);
            }
            other => panic!("expected zero_point Tensor, got {other:?}"),
        }

        // Axis should be last (1).
        match op.attributes.get("axis") {
            Some(Value::Int(a)) => assert_eq!(*a, 1),
            other => panic!("expected axis Int, got {other:?}"),
        }

        // group_size attribute.
        match op.attributes.get("group_size") {
            Some(Value::Int(g)) => assert_eq!(*g, 4),
            other => panic!("expected group_size Int, got {other:?}"),
        }
    }

    #[test]
    fn int4_per_group_verifies_per_group_scales() {
        // Two groups with very different ranges to verify independent quantization.
        // Group 0: [0, 0, 0, 15] → range [0,15], scale=1.0, zp=0
        // Group 1: [0, 0, 0, 1]  → range [0,1], scale=1/15≈0.0667, zp=0
        let values = [0.0_f32, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 1.0];
        let mut program = make_program(&values, vec![1, 8]);

        AffineQuantizePass::int4_per_group(4)
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        match op.attributes.get("scale") {
            Some(Value::Tensor { data, shape, .. }) => {
                assert_eq!(*shape, vec![1, 2]);
                let scales: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                assert_eq!(scales.len(), 2);
                // Group 0 scale should be ~1.0, group 1 scale should be ~0.0667.
                assert!(
                    (scales[0] - 1.0).abs() < 1e-5,
                    "group 0 scale ≈ 1.0, got {}",
                    scales[0]
                );
                assert!(
                    (scales[1] - 1.0 / 15.0).abs() < 1e-5,
                    "group 1 scale ≈ 0.0667, got {}",
                    scales[1]
                );
            }
            other => panic!("expected scale Tensor, got {other:?}"),
        }
    }

    #[test]
    fn int4_per_group_partial_last_group() {
        // Shape [1, 6], group_size=4 → group 0 has 4 elements, group 1 has 2.
        let values = [0.0_f32, 1.0, 2.0, 3.0, 10.0, 20.0];
        let mut program = make_program(&values, vec![1, 6]);

        AffineQuantizePass::int4_per_group(4)
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];

        // Scale shape = [1, 2] (two groups even though last is partial).
        match op.attributes.get("scale") {
            Some(Value::Tensor { shape, .. }) => {
                assert_eq!(*shape, vec![1, 2]);
            }
            other => panic!("expected scale Tensor, got {other:?}"),
        }

        // Quantized data should have 6 elements packed into 3 bytes.
        match op.attributes.get("quantized_data") {
            Some(Value::Tensor { data, shape, .. }) => {
                assert_eq!(*shape, vec![1, 6]);
                assert_eq!(data.len(), 3);
            }
            other => panic!("expected quantized_data Tensor, got {other:?}"),
        }
    }

    #[test]
    fn int4_per_group_1d_tensor() {
        // 1D tensor, group_size=2: [0, 10, 0, 5]
        let values = [0.0_f32, 10.0, 0.0, 5.0];
        let mut program = make_program(&values, vec![4]);

        AffineQuantizePass::int4_per_group(2)
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        match op.attributes.get("scale") {
            Some(Value::Tensor { shape, .. }) => {
                assert_eq!(*shape, vec![2]); // 2 groups
            }
            other => panic!("expected scale Tensor, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // INT8 per-tensor — verify matches Int8QuantizePass behavior
    // -----------------------------------------------------------------------

    #[test]
    fn int8_per_tensor_matches_existing_pass() {
        let values = [-5.0_f32, -1.0, 0.0, 2.5, 10.0];

        // Run the existing Int8QuantizePass.
        let mut program_old = make_program(&values, vec![5]);
        Int8QuantizePass::weight_only()
            .run(&mut program_old)
            .unwrap();

        // Run our new AffineQuantizePass in INT8 per-tensor mode.
        let mut program_new = make_program(&values, vec![5]);
        AffineQuantizePass::int8_per_tensor()
            .run(&mut program_new)
            .unwrap();

        let op_old = &program_old.functions["main"].body.operations[0];
        let op_new = &program_new.functions["main"].body.operations[0];

        // Both should produce constexpr_affine_dequantize.
        assert_eq!(op_old.op_type, op_new.op_type);

        // Quantized data should be identical.
        let data_old = get_quantized_data(&program_old);
        let data_new = get_quantized_data(&program_new);
        assert_eq!(data_old, data_new, "quantized bytes differ");

        // Scale and zero_point should match.
        match (
            op_old.attributes.get("scale"),
            op_new.attributes.get("scale"),
        ) {
            (Some(Value::Float(a)), Some(Value::Float(b))) => {
                assert!((a - b).abs() < 1e-10, "scales differ: {a} vs {b}");
            }
            (a, b) => panic!("scale mismatch: {a:?} vs {b:?}"),
        }
        match (
            op_old.attributes.get("zero_point"),
            op_new.attributes.get("zero_point"),
        ) {
            (Some(Value::Float(a)), Some(Value::Float(b))) => {
                assert!((a - b).abs() < 1e-10, "zero_points differ: {a} vs {b}");
            }
            (a, b) => panic!("zero_point mismatch: {a:?} vs {b:?}"),
        }
    }

    #[test]
    fn int8_per_tensor_round_trip() {
        let original = [-3.0_f32, -1.5, 0.0, 1.5, 3.0, 6.0];
        let (quantized, scale, zero_point) = quantize_affine(&original, 255.0);
        let recovered = dequantize_affine(&quantized, scale, zero_point);

        let tol = scale / 2.0 + f32::EPSILON;
        for (orig, recov) in original.iter().zip(recovered.iter()) {
            let err = (orig - recov).abs();
            assert!(
                err <= tol,
                "INT8 round-trip error {err} > tol {tol} for {orig} (got {recov})"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn edge_case_uniform_weights() {
        let values = [42.0_f32; 8];
        let mut program = make_program(&values, vec![8]);

        AffineQuantizePass::int4_per_tensor()
            .run(&mut program)
            .unwrap();

        let data = get_quantized_data(&program);
        let first = data[0];
        for &b in data {
            assert_eq!(b, first, "all-same input should produce all-same output");
            assert!(b <= 15, "INT4 value {b} exceeds 15");
        }
    }

    #[test]
    fn edge_case_uniform_weights_int8() {
        let values = [42.0_f32; 8];
        let mut program = make_program(&values, vec![8]);

        AffineQuantizePass::int8_per_tensor()
            .run(&mut program)
            .unwrap();

        let data = get_quantized_data(&program);
        let first = data[0];
        for &b in data {
            assert_eq!(b, first, "all-same input should produce all-same output");
        }
    }

    #[test]
    fn edge_case_single_weight() {
        let values = [7.5_f32];
        let mut program = make_program(&values, vec![1]);

        AffineQuantizePass::int4_per_tensor()
            .run(&mut program)
            .unwrap();

        let data = get_quantized_data(&program);
        assert_eq!(data.len(), 1);
        assert!(data[0] <= 15);
    }

    #[test]
    fn edge_case_group_size_larger_than_tensor() {
        // group_size=128 but tensor only has 4 elements → single group.
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let mut program = make_program(&values, vec![4]);

        AffineQuantizePass::int4_per_group(128)
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];

        // Should have 1 group.
        match op.attributes.get("scale") {
            Some(Value::Tensor { shape, .. }) => {
                assert_eq!(*shape, vec![1]);
            }
            other => panic!("expected scale Tensor, got {other:?}"),
        }

        let data = get_quantized_data(&program);
        assert_eq!(data.len(), 4);
        for &b in data {
            assert!(b <= 15);
        }
    }

    #[test]
    fn edge_case_group_size_equals_dim() {
        // group_size equals last dim → exactly 1 group per row = per-channel-like.
        let values = [0.0_f32, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0];
        let mut program = make_program(&values, vec![2, 4]);

        AffineQuantizePass::int4_per_group(4)
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        match op.attributes.get("scale") {
            Some(Value::Tensor { shape, .. }) => {
                assert_eq!(*shape, vec![2, 1]); // 1 group per row
            }
            other => panic!("expected scale Tensor, got {other:?}"),
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
        func.body.add_op(const_tensor_op("idx", "idx_out", int_val));
        func.body.outputs.push("idx_out".into());
        program.add_function(func);

        AffineQuantizePass::int4_per_tensor()
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "const");
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

        AffineQuantizePass::int4_per_tensor()
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "relu");
    }

    #[test]
    fn int4_per_group_round_trip_accuracy() {
        // Verify per-group quantization produces low round-trip error within each group.
        #[rustfmt::skip]
        let values = [
            // Group 0: small range
            0.0, 0.1, 0.2, 0.3,
            // Group 1: large range
            -10.0, 0.0, 5.0, 20.0,
        ];

        // Quantize per-group, group_size=4
        let group_size = 4;
        let qmax = 15.0;

        for g in 0..2 {
            let start = g * group_size;
            let end = start + group_size;
            let group = &values[start..end];
            let (quantized, scale, zp) = quantize_affine(group, qmax);
            let recovered = dequantize_affine(&quantized, scale, zp);

            let tol = scale / 2.0 + f32::EPSILON;
            for (orig, recov) in group.iter().zip(recovered.iter()) {
                let err = (orig - recov).abs();
                assert!(
                    err <= tol,
                    "group {g}: round-trip error {err} > tol {tol} for {orig}"
                );
            }
        }
    }

    #[test]
    fn int8_per_channel_matches_existing() {
        // Verify per-channel INT8 matches Int8QuantizePass per-channel.
        let channel_0 = [0.0_f32, 1.0, 2.0];
        let channel_1 = [-3.0_f32, 0.0, 6.0];
        let mut all_vals = Vec::new();
        all_vals.extend_from_slice(&channel_0);
        all_vals.extend_from_slice(&channel_1);

        let mut program_old = make_program(&all_vals, vec![2, 3]);
        Int8QuantizePass::new(None, Granularity::PerChannel)
            .run(&mut program_old)
            .unwrap();

        let mut program_new = make_program(&all_vals, vec![2, 3]);
        AffineQuantizePass::new(BitWidth::Eight, None, Granularity::PerChannel)
            .run(&mut program_new)
            .unwrap();

        let data_old = get_quantized_data(&program_old);
        let data_new = get_quantized_data(&program_new);
        assert_eq!(data_old, data_new, "per-channel quantized bytes differ");
    }

    #[test]
    fn quantize_affine_empty_slice() {
        let (q, s, zp) = quantize_affine(&[], 15.0);
        assert!(q.is_empty());
        assert_eq!(s, 1.0);
        assert_eq!(zp, 0.0);
    }

    // -----------------------------------------------------------------------
    // bit_width / group_size metadata tests
    // -----------------------------------------------------------------------

    #[test]
    fn int4_per_tensor_emits_bit_width_4() {
        let values = [0.0_f32, 1.0, 2.0, 3.0];
        let mut program = make_program(&values, vec![4]);

        AffineQuantizePass::int4_per_tensor()
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.attributes.get("bit_width"), Some(&Value::Int(4)));
        // No group_size for per-tensor.
        assert!(op.attributes.get("group_size").is_none());
    }

    #[test]
    fn int8_per_tensor_emits_bit_width_8() {
        let values = [0.0_f32, 1.0, 2.0, 3.0];
        let mut program = make_program(&values, vec![4]);

        AffineQuantizePass::int8_per_tensor()
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.attributes.get("bit_width"), Some(&Value::Int(8)));
        assert!(op.attributes.get("group_size").is_none());
    }

    #[test]
    fn int4_per_group_emits_bit_width_and_group_size() {
        let values = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut program = make_program(&values, vec![2, 4]);

        AffineQuantizePass::int4_per_group(2)
            .run(&mut program)
            .unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.attributes.get("bit_width"), Some(&Value::Int(4)));
        assert_eq!(op.attributes.get("group_size"), Some(&Value::Int(2)));
    }

    #[test]
    fn int8_pass_emits_bit_width_8() {
        let values = [0.0_f32, 1.0, 2.0, 3.0];
        let mut program = make_program(&values, vec![4]);

        Int8QuantizePass::weight_only().run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.attributes.get("bit_width"), Some(&Value::Int(8)));
    }

    #[test]
    fn int4_per_group_proto_round_trip() {
        use crate::convert::ir_to_proto::program_to_model;
        use crate::convert::proto_to_ir::model_to_program;

        let values = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut program = make_program(&values, vec![2, 4]);
        program.version = "1".to_string();

        AffineQuantizePass::int4_per_group(2)
            .run(&mut program)
            .unwrap();

        // Round-trip through proto.
        let model = program_to_model(&program, 7).unwrap();
        let recovered = model_to_program(&model).unwrap();

        let op = &recovered.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");
        assert_eq!(op.attributes.get("bit_width"), Some(&Value::Int(4)));
        assert_eq!(op.attributes.get("group_size"), Some(&Value::Int(2)));
    }

    #[test]
    fn legacy_int8_without_bit_width_defaults_correctly() {
        // Simulate a legacy constexpr_affine_dequantize op without bit_width
        // or group_size attributes (as would exist in models saved before
        // this feature was added).
        let quantized_data = Value::Tensor {
            data: vec![0, 128, 255, 64],
            shape: vec![4],
            dtype: ScalarType::UInt8,
        };

        let op = Operation::new("constexpr_affine_dequantize", "w_quant")
            .with_attr("quantized_data", quantized_data)
            .with_attr("scale", Value::Float(0.01))
            .with_attr("zero_point", Value::Float(128.0))
            .with_attr("axis", Value::Int(0))
            .with_output("w_quant");

        // No bit_width or group_size attributes — default extraction
        // should give bit_width=8, group_size=None.
        let bit_width = match op.attributes.get("bit_width") {
            Some(Value::Int(v)) => *v as u8,
            _ => 8,
        };
        let group_size = match op.attributes.get("group_size") {
            Some(Value::Int(v)) => Some(*v as usize),
            _ => None,
        };

        assert_eq!(bit_width, 8);
        assert_eq!(group_size, None);
    }
}
