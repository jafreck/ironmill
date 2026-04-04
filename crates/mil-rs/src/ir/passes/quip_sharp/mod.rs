//! QuIP# (Quantization with Incoherence Processing, Sharp) pass.
//!
//! Combines randomized Hadamard rotation with E8 lattice vector quantization
//! for high-quality 2-bit weight compression.  Each weight matrix is
//! normalised per-row, rotated, quantised using the E8 lattice codebook,
//! and stored as a `constexpr_lut_to_dense` op with row-norms rescaling.
//!
//! # References
//!
//! * Tseng et al., "QuIP#: Even Better LLM Quantization with Hadamard
//!   Incoherence and Lattice Codebooks", 2024.

pub mod e8_lattice;

pub use e8_lattice::E8Codebook;

use half::f16;

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::{TensorData, Value};

use super::rotation::{pad_to_power_of_two, rotate_rows_hadamard};
use super::tensor_utils::tensor_as_f32_slice;

/// E8 codebook group dimension.
const E8_DIM: usize = 8;

/// Number of E8 codebook entries (256 = 2^8).
const E8_CODEBOOK_SIZE: usize = 256;

/// QuIP# weight quantization pass.
///
/// For each eligible `const` op the pass:
/// 1. Extracts the weight matrix and normalises each row.
/// 2. Applies a seeded randomised Hadamard rotation.
/// 3. Quantises using E8 lattice vector quantization (8-dim groups).
/// 4. Replaces the op with `constexpr_lut_to_dense` + row-norm rescaling.
pub struct QuipSharpPass {
    /// Effective bits per weight (2 for the primary E8 codebook).
    pub bits: u8,
    /// Seed for the randomised Hadamard rotation (deterministic).
    pub seed: u64,
    /// Minimum number of tensor elements to apply quantisation.
    pub min_elements: usize,
}

impl QuipSharpPass {
    pub fn new(bits: u8) -> Self {
        Self {
            bits,
            seed: 42,
            min_elements: 256,
        }
    }

    /// LDLQ Hessian-guided rounding for a single layer's weight matrix.
    ///
    /// Given a per-layer Hessian `H ≈ X^T X / n` (where X is the input
    /// activation matrix from calibration), uses Cholesky decomposition to
    /// adaptively round weight values for lower quantization error.
    ///
    /// Currently delegates to the default nearest-neighbor E8 quantization
    /// path. A future implementation will perform the full LDLQ solve:
    ///   1. Compute L D L^T decomposition of H
    ///   2. Process columns in reverse order
    ///   3. Round each group using the E8 codebook with Hessian-weighted error
    ///   4. Propagate rounding error to remaining columns
    pub fn quantize_with_hessian(
        &self,
        weights: &[f32],
        rows: usize,
        cols: usize,
        _hessian: &[f32],
    ) -> bool {
        let shape = if rows == 1 {
            vec![cols]
        } else {
            vec![rows, cols]
        };
        let info = EligibleTensor {
            floats: weights.to_vec(),
            shape,
            original_output: String::new(),
            op_name: String::new(),
            original_dtype: ScalarType::Float32,
        };
        compute_quantization(&info, self.seed).is_some()
    }
}

impl Pass for QuipSharpPass {
    fn name(&self) -> &str {
        "quip-sharp"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            let mut insertions: Vec<(usize, Vec<Operation>)> = Vec::new();
            let mut replacements: Vec<(String, String)> = Vec::new();

            for i in 0..function.body.operations.len() {
                let eligible =
                    extract_eligible_tensor(&function.body.operations[i], self.min_elements);
                let info = match eligible {
                    Some(t) => t,
                    None => continue,
                };

                let quant = match compute_quantization(&info, self.seed) {
                    Some(q) => q,
                    None => continue,
                };

                rewrite_op_in_place(&mut function.body.operations[i], &quant, &info, self.seed);

                let (norms_op, mul_op, mul_output) = build_norm_mul_ops(&info, &quant);

                insertions.push((i, vec![norms_op, mul_op]));
                replacements.push((info.original_output.clone(), mul_output));
            }

            apply_insertions(&mut function.body.operations, &insertions);
            patch_references(&mut function.body, &replacements);
        }
        Ok(())
    }
}

// ── Phase helpers ───────────────────────────────────────────────────────────

/// Extracted information about an eligible const tensor.
struct EligibleTensor {
    floats: Vec<f32>,
    shape: Vec<usize>,
    original_output: String,
    op_name: String,
    original_dtype: ScalarType,
}

/// Results of E8 vector-quantizing a tensor.
struct QuantizedE8Tensor {
    /// Per-group codebook indices (one u8 per 8-element group).
    indices: Vec<u8>,
    /// Per-group scaling factors.
    scales: Vec<f32>,
    /// Per-row L2 norms (for rescaling after un-rotation).
    row_norms: Vec<f32>,
    /// Shape after column-padding to a power of two.
    padded_shape: Vec<usize>,
}

/// Phase 1: check whether an op is an eligible const tensor and extract its data.
fn extract_eligible_tensor(op: &Operation, min_elements: usize) -> Option<EligibleTensor> {
    if op.op_type != "const" {
        return None;
    }

    let val = op.inputs.get("val").or_else(|| op.attributes.get("val"))?;

    let (floats, shape, original_dtype) = match val {
        Value::Tensor {
            data,
            shape,
            dtype: dtype @ ScalarType::Float32,
        } => (
            tensor_as_f32_slice(data.as_bytes().expect("tensor not materialized")),
            shape.clone(),
            *dtype,
        ),
        Value::Tensor {
            data,
            shape,
            dtype: dtype @ ScalarType::Float16,
        } => (
            fp16_bytes_to_f32(data.as_bytes().expect("tensor not materialized")),
            shape.clone(),
            *dtype,
        ),
        _ => return None,
    };

    let numel: usize = shape.iter().product();
    if numel < min_elements {
        return None;
    }

    Some(EligibleTensor {
        floats,
        shape,
        original_output: op.outputs.first().cloned().unwrap_or_default(),
        op_name: op.name.clone(),
        original_dtype,
    })
}

/// Phase 2: normalise, rotate, and E8-quantise the tensor.
///
/// Returns `None` for tensors that don't meet shape requirements (rank < 2
/// or last dimension < 8).
fn compute_quantization(info: &EligibleTensor, seed: u64) -> Option<QuantizedE8Tensor> {
    let rank = info.shape.len();
    if rank < 2 {
        return None;
    }

    let cols = info.shape[rank - 1];
    let rows: usize = info.shape[..rank - 1].iter().product();

    // Need at least one full E8 group (8 elements per group).
    if cols < E8_DIM {
        return None;
    }

    let (mut padded_data, padded_cols) = pad_to_power_of_two(&info.floats, rows, cols);

    // Compute row norms and normalise each row to unit length.
    let mut row_norms = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &padded_data[r * padded_cols..(r + 1) * padded_cols];
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        row_norms[r] = norm;
        if norm > 0.0 {
            let row_mut = &mut padded_data[r * padded_cols..(r + 1) * padded_cols];
            for x in row_mut.iter_mut() {
                *x /= norm;
            }
        }
    }

    rotate_rows_hadamard(&mut padded_data, rows, padded_cols, seed);

    // E8 lattice vector quantization.
    let codebook = E8Codebook::new();
    let (indices, scales) = codebook.quantize_matrix(&padded_data, padded_cols);

    let mut padded_shape = info.shape.clone();
    // Safety: shape cloned from tensor, guaranteed non-empty.
    *padded_shape.last_mut().unwrap() = padded_cols;

    Some(QuantizedE8Tensor {
        indices,
        scales,
        row_norms,
        padded_shape,
    })
}

/// Phase 3: mutate the original const op into `constexpr_lut_to_dense`.
fn rewrite_op_in_place(
    op: &mut Operation,
    quant: &QuantizedE8Tensor,
    info: &EligibleTensor,
    seed: u64,
) {
    let codebook = E8Codebook::new();

    // LUT: all 256 codebook entries flattened into [256, 8].
    let lut_floats: Vec<f32> = codebook
        .entries()
        .iter()
        .flat_map(|entry| entry.iter().copied())
        .collect();

    let lut_data: Vec<u8> = match info.original_dtype {
        ScalarType::Float16 => lut_floats
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect(),
        _ => lut_floats.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    let lut_value = Value::Tensor {
        data: TensorData::Inline(lut_data),
        shape: vec![E8_CODEBOOK_SIZE, E8_DIM],
        dtype: info.original_dtype,
    };

    // Indices: per-group u8 codebook indices.
    let indices_value = Value::Tensor {
        data: TensorData::Inline(quant.indices.clone()),
        shape: vec![quant.indices.len()],
        dtype: ScalarType::UInt8,
    };

    // Original (unpadded) tensor shape.
    let shape_bytes: Vec<u8> = info
        .shape
        .iter()
        .flat_map(|&d| (d as u32).to_le_bytes())
        .collect();
    let shape_value = Value::Tensor {
        data: TensorData::Inline(shape_bytes),
        shape: vec![info.shape.len()],
        dtype: ScalarType::UInt32,
    };

    // Per-group scales for reconstructing scaled codebook entries.
    let scales_data: Vec<u8> = match info.original_dtype {
        ScalarType::Float16 => quant
            .scales
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect(),
        _ => quant.scales.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    let scales_value = Value::Tensor {
        data: TensorData::Inline(scales_data),
        shape: vec![quant.scales.len()],
        dtype: info.original_dtype,
    };

    op.op_type = "constexpr_lut_to_dense".to_string();
    op.inputs.remove("val");
    op.attributes.remove("val");
    op.attributes.insert("lut".to_string(), lut_value);
    op.attributes.insert("indices".to_string(), indices_value);
    op.attributes.insert("shape".to_string(), shape_value);
    op.attributes
        .insert("quip_sharp_scales".to_string(), scales_value);
    op.attributes
        .insert("quip_sharp_seed".to_string(), Value::Int(seed as i64));

    if !op.output_types.is_empty() {
        op.output_types[0] = Some(TensorType::new(
            info.original_dtype,
            quant.padded_shape.clone(),
        ));
    }
}

/// Phase 4: build the row-norms const and mul ops for rescaling.
fn build_norm_mul_ops(
    info: &EligibleTensor,
    quant: &QuantizedE8Tensor,
) -> (Operation, Operation, String) {
    let norms_output = format!("{}_quip_norms", info.original_output);
    let mul_output = format!("{}_quip_scaled", info.original_output);

    let norms_data: Vec<u8> = match info.original_dtype {
        ScalarType::Float16 => quant
            .row_norms
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect(),
        _ => quant
            .row_norms
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    };

    let mut norms_shape = quant.padded_shape.clone();
    // Safety: shape cloned from tensor, guaranteed non-empty.
    *norms_shape.last_mut().unwrap() = 1;

    let mut norms_op = Operation::new("const", format!("{}_quip_norms", info.op_name))
        .with_input(
            "val",
            Value::Tensor {
                data: TensorData::Inline(norms_data),
                shape: norms_shape.clone(),
                dtype: info.original_dtype,
            },
        )
        .with_output(&norms_output);
    norms_op.output_types[0] = Some(TensorType::new(info.original_dtype, norms_shape));

    let mut mul_op = Operation::new("mul", format!("{}_quip_mul", info.op_name))
        .with_input("x", Value::Reference(info.original_output.clone()))
        .with_input("y", Value::Reference(norms_output))
        .with_output(&mul_output);
    mul_op.output_types[0] = Some(TensorType::new(
        info.original_dtype,
        quant.padded_shape.clone(),
    ));

    (norms_op, mul_op, mul_output)
}

/// Insert new ops after their original positions, adjusting indices for
/// prior insertions (each insertion adds 2 ops).
fn apply_insertions(ops: &mut Vec<Operation>, insertions: &[(usize, Vec<Operation>)]) {
    for (offset, (idx, new_ops)) in insertions.iter().enumerate() {
        let insert_at = idx + 1 + offset * 2;
        for (j, op) in new_ops.iter().enumerate() {
            ops.insert(insert_at + j, op.clone());
        }
    }
}

/// Rewire downstream references from original const outputs to mul outputs,
/// then restore the mul op's own `x` input back to the original.
fn patch_references(body: &mut crate::ir::program::Block, replacements: &[(String, String)]) {
    for (old_name, new_name) in replacements {
        super::replace_reference(body, old_name, new_name);
        // The mul op's `x` input was also rewritten — fix it back.
        for op in &mut body.operations {
            if op.op_type == "mul" && op.outputs.iter().any(|o| o == new_name) {
                if let Some(Value::Reference(r)) = op.inputs.get_mut("x") {
                    if r == new_name {
                        *r = old_name.clone();
                    }
                }
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert raw FP16 little-endian bytes to `Vec<f32>`.
fn fp16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    debug_assert!(
        data.len() % 2 == 0,
        "FP16 tensor data length must be a multiple of 2"
    );
    data.chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::e8_lattice::{mse, naive_scalar_2bit_quantize};
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::passes::rotation::unrotate_rows_hadamard;
    use crate::ir::program::{Function, Program};
    use crate::ir::types::TensorData;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn const_tensor_op(name: &str, output: &str, value: Value) -> Operation {
        Operation::new("const", name)
            .with_input("val", value)
            .with_output(output)
    }

    #[test]
    fn quip_sharp_skips_small_tensors() {
        let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![8, 8],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op("w", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        // 64 elements < 256 min_elements → should be skipped.
        QuipSharpPass::new(2).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "const", "small tensor should be left unchanged");
    }

    #[test]
    fn quip_sharp_rewrites_const_to_lut_to_dense() {
        let rows = 32;
        let cols = 32;
        let numel = rows * cols; // 1024 >= 256 min_elements
        let weights: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.1).sin()).collect();

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![rows, cols],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        QuipSharpPass::new(2).run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;

        // Should have 3 ops: constexpr_lut_to_dense, const (norms), mul.
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].op_type, "constexpr_lut_to_dense");
        assert_eq!(ops[1].op_type, "const");
        assert_eq!(ops[2].op_type, "mul");

        // LUT should have shape [256, 8], matching input dtype.
        match ops[0].attributes.get("lut") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(shape, &[256, 8]);
                assert_eq!(*dtype, ScalarType::Float32);
            }
            other => panic!("expected LUT tensor, got {other:?}"),
        }

        // Indices should be UInt8.
        match ops[0].attributes.get("indices") {
            Some(Value::Tensor { dtype, .. }) => {
                assert_eq!(*dtype, ScalarType::UInt8);
            }
            other => panic!("expected indices tensor, got {other:?}"),
        }

        // Per-group scales stored.
        match ops[0].attributes.get("quip_sharp_scales") {
            Some(Value::Tensor { dtype, .. }) => {
                assert_eq!(*dtype, ScalarType::Float32);
            }
            other => panic!("expected scales tensor, got {other:?}"),
        }

        // Rotation seed stored.
        match ops[0].attributes.get("quip_sharp_seed") {
            Some(Value::Int(42)) => {}
            other => panic!("expected seed 42, got {other:?}"),
        }

        // Norms tensor shape is [rows, 1], matching input dtype.
        match ops[1].inputs.get("val") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(shape, &[rows, 1]);
                assert_eq!(*dtype, ScalarType::Float32);
            }
            other => panic!("expected norms tensor, got {other:?}"),
        }

        // Block output should point to the mul output.
        let block_outputs = &program.functions["main"].body.outputs;
        assert_eq!(block_outputs, &["w_out_quip_scaled"]);
    }

    #[test]
    fn quip_sharp_lower_mse_than_naive_2bit() {
        let rows = 32;
        let cols = 64;
        // Weights with varying per-row scales, mimicking real NN weights
        // where different output channels have different magnitudes.
        // Naive 2-bit uses a global min/max, wasting levels on small-norm
        // rows. QuIP# normalises per-row, so each row gets optimal treatment.
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let row = i / cols;
                let col = i % cols;
                let row_scale = 2.0f32.powi(row as i32 / 4);
                ((col as f32 * 0.3 + row as f32 * 0.7).sin() * row_scale)
            })
            .collect();

        let seed = 42u64;

        // Full QuIP# pipeline: normalise → rotate → E8 quantise → dequantise → un-rotate → rescale
        let (mut padded_data, padded_cols) = pad_to_power_of_two(&weights, rows, cols);

        let mut row_norms = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &padded_data[r * padded_cols..(r + 1) * padded_cols];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            row_norms[r] = norm;
            if norm > 0.0 {
                let row_mut = &mut padded_data[r * padded_cols..(r + 1) * padded_cols];
                for x in row_mut.iter_mut() {
                    *x /= norm;
                }
            }
        }

        rotate_rows_hadamard(&mut padded_data, rows, padded_cols, seed);

        let codebook = E8Codebook::new();
        let (indices, scales) = codebook.quantize_matrix(&padded_data, padded_cols);
        let mut reconstructed = codebook.dequantize_matrix(&indices, &scales, padded_cols);

        // Un-rotate.
        unrotate_rows_hadamard(&mut reconstructed, rows, padded_cols, seed);

        // Un-normalise (multiply back row norms).
        for r in 0..rows {
            let row = &mut reconstructed[r * padded_cols..(r + 1) * padded_cols];
            for x in row.iter_mut() {
                *x *= row_norms[r];
            }
        }

        // Strip padding columns to compare against original.
        let mut quip_recon = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            quip_recon.extend_from_slice(&reconstructed[r * padded_cols..r * padded_cols + cols]);
        }

        let quip_err = mse(&weights, &quip_recon);

        // Naive 2-bit scalar quantization for comparison.
        let naive_recon = naive_scalar_2bit_quantize(&weights);
        let naive_err = mse(&weights, &naive_recon);

        eprintln!("QuIP# MSE: {quip_err:.6}");
        eprintln!("Naive MSE: {naive_err:.6}");
        eprintln!(
            "QuIP# advantage: {:.1}× lower error",
            naive_err / quip_err.max(1e-10)
        );

        assert!(
            quip_err < naive_err,
            "QuIP# should beat naive scalar 2-bit: quip={quip_err:.6}, naive={naive_err:.6}"
        );
    }
}
