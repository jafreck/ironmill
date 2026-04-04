//! GPTQ quantization pass — Hessian-guided weight quantization.
//!
//! Implements the GPTQ algorithm (Frantar et al., 2022): column-block
//! quantization with Hessian-based error compensation.  Each weight matrix
//! is quantized column-by-column (in blocks of `block_size`), and the
//! quantization error for each column is propagated to the remaining
//! unquantized columns using the inverse Hessian, minimising the overall
//! output reconstruction error.
//!
//! Gated behind the `gptq` feature flag.

pub mod hessian;

use std::collections::HashMap;

use crate::ir::passes::int4_pack::pack_int4;

use hessian::{cholesky_decompose, cholesky_inverse_factor, finalize_hessian};

use crate::error::{MilError, Result};
use crate::ir::pass::Pass;
use crate::ir::passes::tensor_utils::{f32_slice_to_bytes, tensor_as_f32_slice};
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::Value;

/// GPTQ weight quantization pass.
///
/// For each `const` op whose name appears in [`hessian_data`](Self::hessian_data),
/// the pass applies the GPTQ column-block algorithm to produce a quantized
/// weight matrix with per-group scales and zero points.  The op is then
/// rewritten to `constexpr_affine_dequantize`.
///
/// Weights whose names are **not** in `hessian_data` are left unchanged.
pub struct GptqQuantizePass {
    /// Quantization bit width (4 or 8).
    pub bits: u8,
    /// Per-group granularity along the last axis (typically 128).
    pub group_size: usize,
    /// Number of columns processed together in each GPTQ block (typically 128).
    pub block_size: usize,
    /// Diagonal dampening factor for the Hessian (typically 0.01).
    pub dampening: f64,
    /// Per-layer accumulated X^T X data.
    ///
    /// Keys are layer names (matching the `name` field of `const` ops).
    /// Values are `(xtx_flat, n_features, sample_count)` where `xtx_flat` is a
    /// row-major `n_features × n_features` matrix and `sample_count` is the
    /// number of calibration samples used to accumulate `X^T X`.
    pub hessian_data: HashMap<String, (Vec<f32>, usize, usize)>,
}

impl GptqQuantizePass {
    /// Create a pass with the given configuration and pre-collected Hessian data.
    pub fn new(
        bits: u8,
        group_size: usize,
        block_size: usize,
        dampening: f64,
        hessian_data: HashMap<String, (Vec<f32>, usize, usize)>,
    ) -> Self {
        assert!(bits == 4 || bits == 8, "bits must be 4 or 8, got {bits}");
        assert!(group_size > 0, "group_size must be > 0");
        assert!(block_size > 0, "block_size must be > 0");
        Self {
            bits,
            group_size,
            block_size,
            dampening,
            hessian_data,
        }
    }
}

impl Pass for GptqQuantizePass {
    fn name(&self) -> &str {
        "gptq-quantization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        let qmax = (1u32 << self.bits) - 1;
        let qmax_f = qmax as f32;

        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                if op.op_type != "const" {
                    continue;
                }

                // Only process ops that have Hessian data.
                // Look up by onnx_name first (canonical HF weight name), then
                // fall back to op.name (MIL-internal const name).
                let hessian_entry = self
                    .hessian_data
                    .get(
                        op.attributes
                            .get("onnx_name")
                            .and_then(|v| match v {
                                Value::String(s) if !s.is_empty() => Some(s.as_str()),
                                _ => None,
                            })
                            .unwrap_or(&op.name),
                    )
                    .or_else(|| self.hessian_data.get(&op.name));
                let (xtx_orig, n_features, sample_count) = match hessian_entry {
                    Some(v) => v,
                    None => continue,
                };

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
                    op.inputs
                        .remove("val")
                        .ok_or_else(|| MilError::Validation("missing val in inputs".into()))?
                } else {
                    op.attributes
                        .remove("val")
                        .ok_or_else(|| MilError::Validation("missing val in attributes".into()))?
                };

                if let Value::Tensor {
                    data,
                    shape,
                    dtype: _,
                } = val
                {
                    let floats = tensor_as_f32_slice(&data);

                    // W shape: [out_features, in_features].
                    assert!(
                        shape.len() >= 2,
                        "GPTQ expects at least 2-D weight, got shape {shape:?}"
                    );
                    let in_features = *shape
                        .last()
                        .ok_or_else(|| MilError::Validation("empty shape".into()))?;
                    let out_features: usize = shape[..shape.len() - 1].iter().product();

                    assert_eq!(
                        *n_features, in_features,
                        "Hessian n_features ({}) != weight in_features ({in_features})",
                        n_features
                    );

                    // Run the GPTQ algorithm.
                    let quantized = gptq_quantize_weight(
                        &floats,
                        out_features,
                        in_features,
                        xtx_orig,
                        *sample_count,
                        self.dampening,
                        self.block_size,
                        self.group_size,
                        qmax_f,
                    )?;

                    // Emit the constexpr_affine_dequantize op.
                    emit_gptq_result(op, &quantized, &shape, self.group_size, qmax_f, self.bits);
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GPTQ core algorithm
// ---------------------------------------------------------------------------

/// Result of GPTQ quantization for a single weight matrix.
struct GptqResult {
    /// Quantized weight values (u8), shape [out_features, in_features].
    quantized: Vec<u8>,
    /// Per-group scales, shape [out_features, n_groups].
    scales: Vec<f32>,
    /// Per-group zero points, shape [out_features, n_groups].
    zero_points: Vec<f32>,
}

/// Apply the GPTQ column-block quantization algorithm to a weight matrix.
///
/// W is row-major `[out_features, in_features]`.
/// Returns quantized weights with per-group scale/zero_point.
///
/// Follows the reference GPTQ implementation (Frantar et al., 2022):
///   1. Compute H = (2/n)·X^TX + dampening
///   2. Cholesky-decompose H → L
///   3. Compute U = upper Cholesky factor of H⁻¹ (H⁻¹ = UᵀU)
///   4. Process columns in blocks using U submatrices
///
/// The upper Cholesky factor U encodes the conditional/Schur-complement
/// structure: U[i,i] is the correct denominator for the OBQ update at
/// column i, and U[i, j>i] gives the correct propagation coefficients.
fn gptq_quantize_weight(
    w: &[f32],
    out_features: usize,
    in_features: usize,
    xtx: &[f32],
    sample_count: usize,
    dampening: f64,
    block_size: usize,
    group_size: usize,
    qmax: f32,
) -> Result<GptqResult> {
    // 1. Finalize Hessian: apply scaling (2 / sample_count) and dampening.
    let mut h = xtx.to_vec();
    finalize_hessian(&mut h, in_features, sample_count, dampening)?;

    // 2. Cholesky decompose H → L (H = L · Lᵀ).
    let l = cholesky_decompose(&h, in_features)?;

    // 3. Compute U = upper Cholesky factor of H⁻¹ (H⁻¹ = Uᵀ U).
    //    This is the matrix used in the reference GPTQ implementation.
    //    U[i,i] gives the correct conditional denominator for column i,
    //    and U[i,j] for j>i gives the correct propagation coefficients.
    let u = cholesky_inverse_factor(&l, in_features)?;

    // 4. Pre-compute per-group scales and zero_points from the ORIGINAL
    //    weights. The quantization grid is fixed; error compensation only
    //    adjusts which grid point each value maps to.
    let n_groups = in_features.div_ceil(group_size);
    let mut scales = vec![0.0f32; out_features * n_groups];
    let mut zero_points = vec![0.0f32; out_features * n_groups];

    for row in 0..out_features {
        for g in 0..n_groups {
            let g_start = g * group_size;
            let g_end = ((g + 1) * group_size).min(in_features);
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for k in g_start..g_end {
                let v = w[row * in_features + k];
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }
            let (scale, zp) = if (max_val - min_val).abs() < f32::EPSILON {
                (1.0f32, (-min_val).round())
            } else {
                let s = (max_val - min_val) / qmax;
                (s, (-min_val / s).round())
            };
            scales[row * n_groups + g] = scale;
            zero_points[row * n_groups + g] = zp;
        }
    }

    // 5. Working copy of W for in-place error compensation.
    let mut w_mut = w.to_vec();

    let mut q_out = vec![0u8; out_features * in_features];

    // Per-row error buffer for inter-block propagation.
    let mut err_buf = vec![vec![0.0f32; block_size]; out_features];

    // 6. Process columns in blocks using U submatrices.
    //    Within each block [col, col_end), use U_block = U[col:col_end, col:col_end].
    //    U_block[i,i] is the correct conditional denominator.
    //    U_block[i, j>i] gives the correct intra-block propagation coefficients.
    //    U[col:col_end, col_end:] gives the inter-block propagation coefficients.
    let mut col = 0;
    while col < in_features {
        let col_end = (col + block_size).min(in_features);
        let bsize = col_end - col;

        // Reset error buffer for this block.
        for row_buf in err_buf.iter_mut() {
            for v in row_buf[..bsize].iter_mut() {
                *v = 0.0;
            }
        }

        for j in col..col_end {
            let j_local = j - col;
            // Diagonal of U subblock: U[j, j].
            let u_jj = u[j * in_features + j];
            let group_idx = j / group_size;

            // Quantize column j for every output row.
            for row in 0..out_features {
                let scale = scales[row * n_groups + group_idx];
                let zp = zero_points[row * n_groups + group_idx];
                let w_val = w_mut[row * in_features + j];

                // Quantize.
                let q_val = (w_val / scale + zp).round().clamp(0.0, qmax);
                q_out[row * in_features + j] = q_val as u8;

                // Dequantize.
                let w_hat = (q_val - zp) * scale;

                // Scaled error: divide by U[j,j] (conditional std dev).
                let err = if u_jj.abs() > 1e-10 {
                    (w_val - w_hat) / u_jj
                } else {
                    0.0
                };
                err_buf[row][j_local] = err;

                // Propagate error to remaining columns within this block
                // using U[j, k] (intra-block coefficients from Cholesky factor).
                for k in (j + 1)..col_end {
                    w_mut[row * in_features + k] -= err * u[j * in_features + k];
                }
            }
        }

        // Inter-block lazy update: propagate this block's errors to all
        // columns beyond col_end using U[j, k] for k >= col_end.
        if col_end < in_features {
            for row in 0..out_features {
                for j_local in 0..bsize {
                    let err = err_buf[row][j_local];
                    if err.abs() < 1e-15 {
                        continue;
                    }
                    let j = col + j_local;
                    for k in col_end..in_features {
                        w_mut[row * in_features + k] -= err * u[j * in_features + k];
                    }
                }
            }
        }

        col = col_end;
    }

    Ok(GptqResult {
        quantized: q_out,
        scales,
        zero_points,
    })
}

/// Emit the `constexpr_affine_dequantize` op from GPTQ results.
fn emit_gptq_result(
    op: &mut crate::ir::operation::Operation,
    result: &GptqResult,
    shape: &[usize],
    group_size: usize,
    _qmax: f32,
    bits: u8,
) {
    let ndim = shape.len();
    let in_features = shape[ndim - 1];
    let n_groups = in_features.div_ceil(group_size);

    let quantized_data = if bits == 4 {
        pack_int4(&result.quantized)
    } else {
        result.quantized.clone()
    };

    let quantized_val = Value::Tensor {
        data: quantized_data,
        shape: shape.to_vec(),
        dtype: ScalarType::UInt8,
    };

    // Scale/zero_point shape: replace last dim with n_groups.
    let mut param_shape = shape.to_vec();
    if let Some(last) = param_shape.last_mut() {
        *last = n_groups;
    }

    let scale_bytes = f32_slice_to_bytes(&result.scales);
    let zp_bytes = f32_slice_to_bytes(&result.zero_points);
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
    fn make_program(name: &str, values: &[f32], shape: Vec<usize>) -> Program {
        let tensor_val = Value::Tensor {
            data: f32_bytes(values),
            shape,
            dtype: ScalarType::Float32,
        };
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op(name, "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);
        program
    }

    /// Create a symmetric positive-definite Hessian (identity * factor).
    fn identity_xtx(n: usize, factor: f32) -> Vec<f32> {
        let mut h = vec![0.0f32; n * n];
        for i in 0..n {
            h[i * n + i] = factor;
        }
        h
    }

    /// Create a diagonally-dominant SPD Hessian for realistic tests.
    fn spd_hessian(n: usize) -> Vec<f32> {
        let mut h = vec![0.0f32; n * n];
        for i in 0..n {
            h[i * n + i] = 10.0 + i as f32;
            for j in 0..n {
                if i != j {
                    h[i * n + j] = 1.0 / (1.0 + (i as f32 - j as f32).abs());
                }
            }
        }
        h
    }

    /// Compute reconstruction error: sum of squared differences.
    fn reconstruction_error(original: &[f32], reconstructed: &[f32]) -> f64 {
        original
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum()
    }

    /// Dequantize GPTQ result back to float for error comparison.
    fn dequantize_gptq(
        result: &GptqResult,
        out_features: usize,
        in_features: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let n_groups = in_features.div_ceil(group_size);
        let mut out = vec![0.0f32; out_features * in_features];
        for row in 0..out_features {
            for col in 0..in_features {
                let group = col / group_size;
                let scale = result.scales[row * n_groups + group];
                let zp = result.zero_points[row * n_groups + group];
                let q = result.quantized[row * in_features + col] as f32;
                out[row * in_features + col] = (q - zp) * scale;
            }
        }
        out
    }

    /// Simple min/max quantization for comparison (no error compensation).
    fn minmax_quantize(
        w: &[f32],
        out_features: usize,
        in_features: usize,
        group_size: usize,
        qmax: f32,
    ) -> GptqResult {
        let n_groups = in_features.div_ceil(group_size);
        let mut quantized = vec![0u8; out_features * in_features];
        let mut scales = vec![0.0f32; out_features * n_groups];
        let mut zero_points = vec![0.0f32; out_features * n_groups];

        for row in 0..out_features {
            for g in 0..n_groups {
                let g_start = g * group_size;
                let g_end = ((g + 1) * group_size).min(in_features);
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;
                for k in g_start..g_end {
                    let v = w[row * in_features + k];
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                }
                let (scale, zp) = if (max_val - min_val).abs() < f32::EPSILON {
                    (1.0, (-min_val).round())
                } else {
                    let s = (max_val - min_val) / qmax;
                    (s, (-min_val / s).round())
                };
                scales[row * n_groups + g] = scale;
                zero_points[row * n_groups + g] = zp;
                for k in g_start..g_end {
                    let v = w[row * in_features + k];
                    let q = (v / scale + zp).round().clamp(0.0, qmax) as u8;
                    quantized[row * in_features + k] = q;
                }
            }
        }

        GptqResult {
            quantized,
            scales,
            zero_points,
        }
    }

    // -----------------------------------------------------------------------
    // Test: GPTQ with identity Hessian reduces to standard quantization
    // -----------------------------------------------------------------------

    #[test]
    fn gptq_identity_hessian_matches_standard() {
        let out_features = 4;
        let in_features = 8;
        let group_size = 4;
        let qmax = 15.0f32;

        // Weights with varied values.
        let w: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32 - 16.0) * 0.5)
            .collect();

        // Identity Hessian: error compensation divides by 1.0 and H_inv[j,k]=0
        // for j!=k, so no error propagation occurs.
        let xtx = identity_xtx(in_features, 1.0);

        let gptq_result = gptq_quantize_weight(
            &w,
            out_features,
            in_features,
            &xtx,
            2,   // sample_count
            0.0, // no dampening (identity is already well-conditioned)
            128, // block_size > in_features → single block
            group_size,
            qmax,
        )
        .unwrap();

        let minmax_result = minmax_quantize(&w, out_features, in_features, group_size, qmax);

        // With identity Hessian, GPTQ quantized values should match min/max.
        // Note: dampening=0 with finalize_hessian(sample_count=2) gives
        //   H = (2/2)*I = I, and H_inv = I, cholesky(H_inv) = I.
        // So U[j,j]=1 and U[j,k]=0 for k!=j → no error propagation.
        assert_eq!(
            gptq_result.quantized, minmax_result.quantized,
            "GPTQ with identity Hessian should match standard quantization"
        );
    }

    // -----------------------------------------------------------------------
    // Test: GPTQ produces lower Hessian-weighted error than MinMax
    // -----------------------------------------------------------------------

    /// Compute Hessian-weighted reconstruction error: Σ_row  e_row^T H e_row
    /// where e_row = W[row, :] - dequant(Q)[row, :].
    fn hessian_weighted_error(
        original: &[f32],
        reconstructed: &[f32],
        h: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> f64 {
        let mut total = 0.0f64;
        for row in 0..out_features {
            let row_start = row * in_features;
            for i in 0..in_features {
                let ei = (original[row_start + i] - reconstructed[row_start + i]) as f64;
                for j in 0..in_features {
                    let ej = (original[row_start + j] - reconstructed[row_start + j]) as f64;
                    total += ei * h[i * in_features + j] as f64 * ej;
                }
            }
        }
        total
    }

    #[test]
    fn gptq_lower_error_than_minmax() {
        let out_features = 8;
        let in_features = 16;
        let group_size = 8;
        let qmax = 15.0f32;

        // Random-ish weights with varied distributions across rows.
        let w: Vec<f32> = (0..out_features * in_features)
            .map(|i| {
                let x = i as f32 * 0.17 + 0.3;
                (x.sin() * 5.0 + x.cos() * 3.0)
            })
            .collect();

        // Non-trivial SPD Hessian with off-diagonal correlations.
        let xtx = spd_hessian(in_features);

        // Finalize a copy of xtx to get the actual H used in error comparison.
        let mut h_for_eval = xtx.clone();
        finalize_hessian(&mut h_for_eval, in_features, 2, 0.01).unwrap();

        let gptq_result = gptq_quantize_weight(
            &w,
            out_features,
            in_features,
            &xtx,
            2,
            0.01,
            8,
            group_size,
            qmax,
        )
        .unwrap();

        let minmax_result = minmax_quantize(&w, out_features, in_features, group_size, qmax);

        let gptq_recon = dequantize_gptq(&gptq_result, out_features, in_features, group_size);
        let minmax_recon = dequantize_gptq(&minmax_result, out_features, in_features, group_size);

        let gptq_err =
            hessian_weighted_error(&w, &gptq_recon, &h_for_eval, out_features, in_features);
        let minmax_err =
            hessian_weighted_error(&w, &minmax_recon, &h_for_eval, out_features, in_features);

        assert!(
            gptq_err <= minmax_err + 1e-3,
            "GPTQ H-weighted error ({gptq_err:.6}) should be <= MinMax H-weighted error ({minmax_err:.6})"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Error compensation adjusts remaining columns
    // -----------------------------------------------------------------------

    #[test]
    fn gptq_error_compensation_adjusts_columns() {
        // Use a larger weight matrix so error accumulates enough to cross
        // rounding boundaries. Values are chosen so the first few columns'
        // quantization errors, amplified by strong Hessian correlations,
        // shift later columns noticeably.
        let out_features = 4;
        let in_features = 8;
        let group_size = 8;
        let qmax = 15.0f32;

        // Weights near quantization boundaries.
        #[rustfmt::skip]
        let w = vec![
            // row 0: values near 0.5*(scale) boundaries
            -3.0, -1.51, 0.49, 1.51, 2.49, 3.51, 4.49, 5.51,
            // row 1
            -2.51, -0.49, 0.51, 1.49, 2.51, 3.49, 4.51, 5.49,
            // row 2
            -3.49, -2.01, -0.49, 0.99, 2.01, 3.49, 4.99, 6.01,
            // row 3
            -1.01, 0.49, 1.99, 3.01, 3.99, 5.01, 5.99, 7.01,
        ];

        // Very strong off-diagonal correlations (near-Toeplitz structure).
        let n = in_features;
        let mut xtx = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                // Exponential decay with distance, but still quite strong.
                let dist = (i as f32 - j as f32).abs();
                xtx[i * n + j] = 10.0 * (-0.3 * dist).exp();
            }
        }

        let gptq_result = gptq_quantize_weight(
            &w,
            out_features,
            in_features,
            &xtx,
            2,
            0.01,
            128,
            group_size,
            qmax,
        )
        .unwrap();

        let minmax_result = minmax_quantize(&w, out_features, in_features, group_size, qmax);

        // With strong correlations and boundary weights, GPTQ should
        // produce at least one different quantized value.
        let any_different = gptq_result
            .quantized
            .iter()
            .zip(minmax_result.quantized.iter())
            .any(|(a, b)| a != b);

        assert!(
            any_different,
            "GPTQ should differ from MinMax when Hessian has strong off-diagonal terms.\n\
             Quantized GPTQ:   {:?}\n\
             Quantized MinMax: {:?}",
            gptq_result.quantized, minmax_result.quantized,
        );
    }

    // -----------------------------------------------------------------------
    // Test: Pass trait integration — rewrites op correctly
    // -----------------------------------------------------------------------

    #[test]
    fn gptq_pass_rewrites_const_op() {
        let out_features = 4;
        let in_features = 8;
        let group_size = 4;

        let w: Vec<f32> = (0..out_features * in_features)
            .map(|i| i as f32 * 0.3 - 2.0)
            .collect();

        let xtx = identity_xtx(in_features, 1.0);

        let mut hessian_data = HashMap::new();
        hessian_data.insert("weight".to_string(), (xtx, in_features, 2));

        let pass = GptqQuantizePass::new(4, group_size, 128, 0.0, hessian_data);

        let mut program = make_program("weight", &w, vec![out_features, in_features]);
        pass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");

        // Check quantized_data attribute exists with correct shape.
        match op.attributes.get("quantized_data") {
            Some(Value::Tensor { shape, dtype, data }) => {
                use crate::ir::passes::int4_pack::unpack_int4;
                assert_eq!(*shape, vec![out_features, in_features]);
                assert_eq!(*dtype, ScalarType::UInt8);
                let numel = out_features * in_features;
                assert_eq!(data.len(), numel.div_ceil(2));
                let unpacked = unpack_int4(data, numel);
                for &b in unpacked.iter() {
                    assert!(b <= 15, "INT4 value {b} exceeds 15");
                }
            }
            other => panic!("expected quantized_data Tensor, got {other:?}"),
        }

        // Check scale shape.
        let n_groups = in_features.div_ceil(group_size);
        match op.attributes.get("scale") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(*shape, vec![out_features, n_groups]);
                assert_eq!(*dtype, ScalarType::Float32);
            }
            other => panic!("expected scale Tensor, got {other:?}"),
        }

        // Check zero_point shape.
        match op.attributes.get("zero_point") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(*shape, vec![out_features, n_groups]);
                assert_eq!(*dtype, ScalarType::Float32);
            }
            other => panic!("expected zero_point Tensor, got {other:?}"),
        }

        // Check bit_width attribute.
        assert_eq!(op.attributes.get("bit_width"), Some(&Value::Int(4)));
    }

    // -----------------------------------------------------------------------
    // Test: Ops without Hessian data are not modified
    // -----------------------------------------------------------------------

    #[test]
    fn gptq_pass_skips_ops_without_hessian() {
        let w: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let pass = GptqQuantizePass::new(4, 4, 128, 0.01, HashMap::new());

        let mut program = make_program("no_hessian", &w, vec![2, 2]);
        pass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(
            op.op_type, "const",
            "op without Hessian should be unchanged"
        );
    }

    // -----------------------------------------------------------------------
    // Test: 8-bit GPTQ
    // -----------------------------------------------------------------------

    #[test]
    fn gptq_8bit_quantization() {
        let out_features = 4;
        let in_features = 8;
        let group_size = 8;
        let qmax = 255.0f32;

        let w: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32 - 16.0) * 0.1)
            .collect();

        let xtx = identity_xtx(in_features, 1.0);

        let result = gptq_quantize_weight(
            &w,
            out_features,
            in_features,
            &xtx,
            2,
            0.0,
            128,
            group_size,
            qmax,
        )
        .unwrap();

        // All quantized values should be in [0, 255].
        for &q in &result.quantized {
            assert!(q <= 255, "INT8 value exceeds 255");
        }

        // 8-bit should have very low reconstruction error.
        let recon = dequantize_gptq(&result, out_features, in_features, group_size);
        let err = reconstruction_error(&w, &recon);
        assert!(
            err < 0.1,
            "8-bit GPTQ should have very low reconstruction error, got {err}"
        );
    }
}
