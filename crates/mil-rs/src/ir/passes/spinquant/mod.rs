//! SpinQuant rotation learning via the Cayley parameterization.
//!
//! The Cayley transform guarantees that learned rotation matrices remain
//! orthogonal throughout optimization, which is the key insight behind
//! SpinQuant's approach to quantization-aware rotation learning.
//!
//! [`SpinQuantPass`] optimizes a per-weight rotation matrix to minimize
//! post-quantization MSE, then absorbs the learned rotation into the
//! weights before emitting `constexpr_affine_dequantize`.

pub mod cayley;

pub use cayley::{CayleyOptimizer, CayleyRotation};

use crate::error::{MilError, Result};
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::passes::rotation::pad_to_power_of_two;
use crate::ir::passes::tensor_utils::tensor_as_f32_slice;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::{TensorData, Value};

/// SpinQuant weight quantization pass.
///
/// For each eligible `const` weight tensor the pass:
/// 1. Initializes a [`CayleyRotation`] from Hadamard-like random parameters.
/// 2. Optimizes the rotation using [`CayleyOptimizer`] to minimize the MSE
///    between the rotated weights and their INT4 affine dequantization.
/// 3. Applies the learned rotation and quantizes the result.
/// 4. Emits `constexpr_affine_dequantize` with the quantized weights.
pub struct SpinQuantPass {
    /// Quantization bit width (typically 4).
    pub bits: u8,
    /// Number of elements per quantization group along the last axis.
    pub group_size: usize,
    /// Number of optimization iterations for the Cayley optimizer.
    pub rotation_epochs: usize,
    /// Seed for Hadamard initialization of the Cayley rotation.
    pub seed: u64,
    /// Skip tensors with fewer than this many elements.
    pub min_elements: usize,
}

impl SpinQuantPass {
    /// Default INT4 per-group SpinQuant configuration.
    pub fn new() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            rotation_epochs: 100,
            seed: 42,
            min_elements: 1024,
        }
    }
}

impl Pass for SpinQuantPass {
    fn name(&self) -> &str {
        "spin-quantization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        let qmax = (1u32 << self.bits) as f32 - 1.0;

        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                if op.op_type != "const" {
                    continue;
                }

                let eligible = extract_eligible(op, self.min_elements);
                let info = match eligible {
                    Some(t) => t,
                    None => continue,
                };

                // Need rank >= 2 for meaningful rotation along the last axis.
                if info.shape.len() < 2 {
                    continue;
                }

                let cols = *info
                    .shape
                    .last()
                    .ok_or_else(|| MilError::Validation("empty shape".into()))?;
                let rows: usize = info.shape[..info.shape.len() - 1].iter().product();

                // Pad last dimension to a power of two for the rotation.
                let (padded_data, padded_cols) = pad_to_power_of_two(&info.floats, rows, cols);

                // Step 1-2: Initialize CayleyRotation from Hadamard.
                let initial = CayleyRotation::from_hadamard(padded_cols, self.seed);

                // Step 3: Optimize the rotation to minimize quantization MSE.
                let optimizer = CayleyOptimizer {
                    max_iterations: self.rotation_epochs,
                    population_size: 20,
                    initial_sigma: 0.1,
                    tolerance: 1e-10,
                };

                let group_size = self.group_size;
                let padded_data_ref = &padded_data;

                let mut loss_fn = |rotation_matrix: &[f32]| -> f64 {
                    let rotated =
                        apply_rotation_cols(padded_data_ref, rows, padded_cols, rotation_matrix);
                    compute_quantization_mse(&rotated, rows, padded_cols, group_size, qmax)
                };

                let learned = optimizer.optimize(&initial, &mut loss_fn);

                // Step 4: Apply the learned rotation.
                let rotation_matrix = learned.to_matrix();
                let rotated =
                    apply_rotation_cols(&padded_data, rows, padded_cols, &rotation_matrix);

                // Trim back to original columns if padding was added.
                let final_weights = if padded_cols != cols {
                    trim_cols(&rotated, rows, padded_cols, cols)
                } else {
                    rotated
                };

                // Step 5: Quantize using per-group affine quantization.
                emit_affine_quantized(
                    op,
                    &final_weights,
                    &info.shape,
                    self.group_size,
                    qmax,
                    self.bits,
                );
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Eligibility
// ---------------------------------------------------------------------------

/// Extracted information about an eligible const tensor.
struct EligibleTensor {
    floats: Vec<f32>,
    shape: Vec<usize>,
}

/// Check whether an op is an eligible FP32 const tensor and extract its data.
fn extract_eligible(op: &Operation, min_elements: usize) -> Option<EligibleTensor> {
    if op.op_type != "const" {
        return None;
    }

    let val = op.inputs.get("val").or_else(|| op.attributes.get("val"))?;

    let (floats, shape) = match val {
        Value::Tensor {
            data,
            shape,
            dtype: ScalarType::Float32,
        } => (
            tensor_as_f32_slice(data.as_bytes().expect("tensor not materialized")),
            shape.clone(),
        ),
        _ => return None,
    };

    let numel: usize = shape.iter().product();
    if numel < min_elements {
        return None;
    }

    Some(EligibleTensor { floats, shape })
}

// ---------------------------------------------------------------------------
// Rotation helpers
// ---------------------------------------------------------------------------

/// Apply a column rotation: `rotated = data @ R^T`.
///
/// `data` is row-major `[rows, cols]`, `rotation` is row-major `[cols, cols]`.
/// For each row `w`, computes `w' = w @ R^T` (i.e. `w'[j] = Σ_k w[k] * R[j][k]`).
fn apply_rotation_cols(data: &[f32], rows: usize, cols: usize, rotation: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for j in 0..cols {
            let mut sum = 0.0f32;
            for k in 0..cols {
                // R^T[k, j] = R[j, k]
                sum += row[k] * rotation[j * cols + k];
            }
            dst[j] = sum;
        }
    }
    out
}

/// Trim padded columns back to the original width.
fn trim_cols(data: &[f32], rows: usize, padded_cols: usize, orig_cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * orig_cols);
    for r in 0..rows {
        out.extend_from_slice(&data[r * padded_cols..r * padded_cols + orig_cols]);
    }
    out
}

use crate::ir::passes::affine_quantize::quantize_affine;
use crate::ir::passes::int4_pack::pack_int4;

/// Dequantize unsigned affine integers back to f32.
fn dequantize_affine(quantized: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    quantized
        .iter()
        .map(|&q| (q as f32 - zero_point) * scale)
        .collect()
}

/// Compute per-group quantization MSE between original and dequantized values.
fn compute_quantization_mse(
    data: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    qmax: f32,
) -> f64 {
    let mut total_se = 0.0f64;
    let n = data.len();

    for r in 0..rows {
        let row_start = r * cols;
        let mut col = 0;
        while col < cols {
            let g_end = (col + group_size).min(cols);
            let group = &data[row_start + col..row_start + g_end];
            let (quantized, scale, zp) = quantize_affine(group, qmax);
            let dequantized = dequantize_affine(&quantized, scale, zp);
            for (i, &orig) in group.iter().enumerate() {
                let diff = orig as f64 - dequantized[i] as f64;
                total_se += diff * diff;
            }
            col = g_end;
        }
    }

    total_se / n as f64
}

// ---------------------------------------------------------------------------
// Op emission
// ---------------------------------------------------------------------------

/// Emit per-group affine quantization into the operation, rewriting it as
/// `constexpr_affine_dequantize`.
fn emit_affine_quantized(
    op: &mut Operation,
    floats: &[f32],
    shape: &[usize],
    group_size: usize,
    qmax: f32,
    bits: u8,
) {
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

    let packed_data = if bits == 4 {
        pack_int4(&all_quantized)
    } else {
        all_quantized
    };

    let quantized_val = Value::Tensor {
        data: TensorData::Inline(packed_data),
        shape: shape.to_vec(),
        dtype: ScalarType::UInt8,
    };

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
            data: TensorData::Inline(scale_bytes),
            shape: param_shape.clone(),
            dtype: ScalarType::Float32,
        },
    );
    op.attributes.insert(
        "zero_point".to_string(),
        Value::Tensor {
            data: TensorData::Inline(zp_bytes),
            shape: param_shape,
            dtype: ScalarType::Float32,
        },
    );
    op.attributes.insert("axis".to_string(), Value::Int(axis));
    op.attributes
        .insert("group_size".to_string(), Value::Int(group_size as i64));
    op.attributes
        .insert("bit_width".to_string(), Value::Int(bits as i64));

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
    use crate::ir::program::{Function, Program};

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn const_tensor_op(name: &str, output: &str, value: Value) -> Operation {
        Operation::new("const", name)
            .with_input("val", value)
            .with_output(output)
    }

    fn make_program(values: &[f32], shape: Vec<usize>) -> Program {
        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(values)),
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

    /// Deterministic seeded PRNG for generating test weight matrices.
    fn make_weight_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(1);
        (0..rows * cols)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn spinquant_produces_lower_mse_than_identity() {
        // Generate a weight matrix where rotation can help.
        let rows = 8;
        let cols = 64;
        let weights = make_weight_matrix(rows, cols, 123);
        let qmax = 15.0f32;
        let group_size = 32;

        // MSE with identity rotation (no rotation, just quantize).
        let identity_mse = compute_quantization_mse(&weights, rows, cols, group_size, qmax);

        // Run SpinQuant optimization.
        let initial = CayleyRotation::from_hadamard(cols, 42);
        let optimizer = CayleyOptimizer {
            max_iterations: 100,
            population_size: 20,
            initial_sigma: 0.1,
            tolerance: 1e-10,
        };

        let weights_ref = &weights;
        let mut loss_fn = |rotation_matrix: &[f32]| -> f64 {
            let rotated = apply_rotation_cols(weights_ref, rows, cols, rotation_matrix);
            compute_quantization_mse(&rotated, rows, cols, group_size, qmax)
        };

        let learned = optimizer.optimize(&initial, &mut loss_fn);
        let rotated = apply_rotation_cols(&weights, rows, cols, &learned.to_matrix());
        let optimized_mse = compute_quantization_mse(&rotated, rows, cols, group_size, qmax);

        assert!(
            optimized_mse <= identity_mse,
            "SpinQuant MSE ({optimized_mse}) should be <= identity MSE ({identity_mse})"
        );
    }

    #[test]
    fn rotation_is_orthogonal_after_optimization() {
        let cols = 16;
        let rows = 4;
        let weights = make_weight_matrix(rows, cols, 77);
        let qmax = 15.0f32;
        let group_size = 8;

        let initial = CayleyRotation::from_hadamard(cols, 42);
        let optimizer = CayleyOptimizer {
            max_iterations: 50,
            population_size: 20,
            initial_sigma: 0.1,
            tolerance: 1e-10,
        };

        let weights_ref = &weights;
        let mut loss_fn = |rotation_matrix: &[f32]| -> f64 {
            let rotated = apply_rotation_cols(weights_ref, rows, cols, rotation_matrix);
            compute_quantization_mse(&rotated, rows, cols, group_size, qmax)
        };

        let learned = optimizer.optimize(&initial, &mut loss_fn);
        let err = learned.orthogonality_error();
        assert!(
            err < 1e-5,
            "learned rotation is not orthogonal: error = {err}"
        );
    }

    #[test]
    fn pass_rewrites_const_op() {
        let rows = 16;
        let cols = 128;
        let weights = make_weight_matrix(rows, cols, 99);

        let mut program = make_program(&weights, vec![rows, cols]);

        let pass = SpinQuantPass {
            bits: 4,
            group_size: 128,
            rotation_epochs: 10,
            seed: 42,
            min_elements: 256,
        };

        pass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(
            op.op_type, "constexpr_affine_dequantize",
            "op should be rewritten to constexpr_affine_dequantize"
        );

        // Verify required attributes are present.
        assert!(
            op.attributes.contains_key("quantized_data"),
            "missing quantized_data"
        );
        assert!(op.attributes.contains_key("scale"), "missing scale");
        assert!(
            op.attributes.contains_key("zero_point"),
            "missing zero_point"
        );
        assert!(op.attributes.contains_key("axis"), "missing axis");
        assert!(
            op.attributes.contains_key("group_size"),
            "missing group_size"
        );
        assert!(op.attributes.contains_key("bit_width"), "missing bit_width");

        // Quantized data values should be in [0, 15] for INT4.
        match op.attributes.get("quantized_data") {
            Some(Value::Tensor { data, shape, dtype }) => {
                use crate::ir::passes::int4_pack::unpack_int4;
                assert_eq!(*dtype, ScalarType::UInt8);
                assert_eq!(*shape, vec![rows, cols]);
                let numel = rows * cols;
                let data = data.as_bytes().expect("tensor not materialized");
                assert_eq!(data.len(), numel.div_ceil(2));
                let unpacked = unpack_int4(data, numel);
                for &b in unpacked.iter() {
                    assert!(b <= 15, "INT4 value {b} exceeds 15");
                }
            }
            other => panic!("expected quantized_data Tensor, got {other:?}"),
        }

        // bit_width attribute should be 4.
        match op.attributes.get("bit_width") {
            Some(Value::Int(bw)) => assert_eq!(*bw, 4),
            other => panic!("expected bit_width Int(4), got {other:?}"),
        }
    }

    #[test]
    fn pass_skips_small_tensors() {
        let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let mut program = make_program(&weights, vec![8, 8]);

        let pass = SpinQuantPass {
            bits: 4,
            group_size: 128,
            rotation_epochs: 10,
            seed: 42,
            min_elements: 1024,
        };

        pass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "const", "small tensor should be left unchanged");
    }

    #[test]
    fn pass_skips_1d_tensors() {
        let weights: Vec<f32> = (0..2048).map(|i| i as f32 * 0.001).collect();
        let mut program = make_program(&weights, vec![2048]);

        let pass = SpinQuantPass::new();
        pass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "const", "1D tensor should be left unchanged");
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let values = [-2.0_f32, -0.5, 0.0, 1.5, 3.0];
        let qmax = 15.0;
        let (quantized, scale, zp) = quantize_affine(&values, qmax);
        let recovered = dequantize_affine(&quantized, scale, zp);

        let tol = scale / 2.0 + f32::EPSILON;
        for (orig, recov) in values.iter().zip(recovered.iter()) {
            let err = (orig - recov).abs();
            assert!(
                err <= tol,
                "round-trip error {err} > tol {tol} for {orig} (got {recov})"
            );
        }
    }
}
