//! PolarQuant weight quantization pass.
//!
//! Compresses FP32/FP16 weight tensors using randomized Hadamard rotation
//! followed by Beta-optimal scalar quantization. Each weight matrix is
//! normalised per-row, rotated, quantised into a lookup table, and stored
//! as a `constexpr_lut_to_dense` op with an additional row-norms
//! multiplication to recover the original scale.

use half::f16;

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::ScalarType;
use crate::ir::types::Value;

use super::beta_quantizer::{beta_optimal_boundaries, beta_optimal_levels, quantize_to_index};
use super::rotation::{pad_to_power_of_two, rotate_rows_hadamard};
use super::tensor_utils::tensor_as_f32_slice;

/// PolarQuant weight quantization pass.
///
/// For each eligible `const` op the pass:
/// 1. Extracts the weight matrix and normalises each row.
/// 2. Applies a seeded randomised Hadamard rotation.
/// 3. Quantises using Beta-optimal levels/boundaries.
/// 4. Replaces the op with `constexpr_lut_to_dense` + row-norm rescaling.
pub struct PolarQuantPass {
    /// Number of bits per quantised index.
    pub n_bits: u8,
    /// Seed for the randomised Hadamard rotation (deterministic).
    pub seed: u64,
    /// Minimum number of tensor elements to apply quantisation.
    pub min_elements: usize,
}

impl PolarQuantPass {
    pub fn new(n_bits: u8) -> Self {
        Self {
            n_bits,
            seed: 42,
            min_elements: 1024,
        }
    }
}

impl Pass for PolarQuantPass {
    fn name(&self) -> &str {
        "polar-quantization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        let k = 1usize << self.n_bits;

        for function in program.functions.values_mut() {
            // Collected post-loop insertions: (original_op_index, new_ops_to_insert_after).
            let mut insertions: Vec<(usize, Vec<Operation>)> = Vec::new();
            // Collected reference replacements: (old_output, new_mul_output).
            let mut replacements: Vec<(String, String)> = Vec::new();

            for i in 0..function.body.operations.len() {
                // ── Phase 1: read-only extraction ──────────────────────────
                let (floats, shape, original_output, op_name) = {
                    let op = &function.body.operations[i];
                    if op.op_type != "const" {
                        continue;
                    }

                    let val = match op.inputs.get("val").or_else(|| op.attributes.get("val")) {
                        Some(v) => v,
                        None => continue,
                    };

                    let (floats, shape) = match val {
                        Value::Tensor {
                            data,
                            shape,
                            dtype: ScalarType::Float32,
                        } => (tensor_as_f32_slice(data), shape.clone()),
                        Value::Tensor {
                            data,
                            shape,
                            dtype: ScalarType::Float16,
                        } => (fp16_bytes_to_f32(data), shape.clone()),
                        _ => continue,
                    };

                    let numel: usize = shape.iter().product();
                    if numel < self.min_elements {
                        continue;
                    }

                    let original_output = op.outputs.first().cloned().unwrap_or_default();
                    let op_name = op.name.clone();
                    (floats, shape, original_output, op_name)
                };

                // ── Phase 2: compute quantisation ─────────────────────────
                let rank = shape.len();
                let (rows, cols) = if rank >= 2 {
                    let cols = shape[rank - 1];
                    let rows: usize = shape[..rank - 1].iter().product();
                    (rows, cols)
                } else {
                    (1usize, shape[0])
                };

                // The Beta distribution requires dim >= 2, and PolarQuant's
                // statistical guarantees are only meaningful for N >= 64.
                if cols < 64 {
                    continue;
                }

                // Pad columns to next power of two.
                let (mut padded_data, padded_cols) = pad_to_power_of_two(&floats, rows, cols);

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

                // Apply randomised Hadamard rotation.
                rotate_rows_hadamard(&mut padded_data, rows, padded_cols, self.seed);

                // Beta-optimal quantisation levels and boundaries.
                let levels = beta_optimal_levels(padded_cols, self.n_bits);
                let boundaries = beta_optimal_boundaries(padded_cols, self.n_bits);

                // Quantise every element to an index.
                let all_indices: Vec<usize> = padded_data
                    .iter()
                    .map(|&v| quantize_to_index(v, &boundaries) as usize)
                    .collect();

                // Truncate back to original cols (discard padded columns).
                let truncated_indices = if padded_cols > cols {
                    let mut trunc = Vec::with_capacity(rows * cols);
                    for r in 0..rows {
                        let start = r * padded_cols;
                        trunc.extend_from_slice(&all_indices[start..start + cols]);
                    }
                    trunc
                } else {
                    all_indices
                };

                // Pack indices at n_bits per element.
                let packed = pack_indices(&truncated_indices, self.n_bits);

                // LUT: Beta-optimal levels stored as Float16.
                let lut_data: Vec<u8> = levels
                    .iter()
                    .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                    .collect();
                let lut_value = Value::Tensor {
                    data: lut_data,
                    shape: vec![k],
                    dtype: ScalarType::Float16,
                };

                let indices_value = Value::Tensor {
                    data: packed.clone(),
                    shape: vec![packed.len()],
                    dtype: ScalarType::UInt8,
                };

                // Original shape as UInt32 tensor.
                let shape_bytes: Vec<u8> = shape
                    .iter()
                    .flat_map(|&d| (d as u32).to_le_bytes())
                    .collect();
                let shape_value = Value::Tensor {
                    data: shape_bytes,
                    shape: vec![shape.len()],
                    dtype: ScalarType::UInt32,
                };

                // ── Phase 3: mutate the original op ───────────────────────
                let op = &mut function.body.operations[i];
                op.op_type = "constexpr_lut_to_dense".to_string();
                op.inputs.remove("val");
                op.attributes.remove("val");
                op.attributes.insert("lut".to_string(), lut_value);
                op.attributes.insert("indices".to_string(), indices_value);
                op.attributes.insert("shape".to_string(), shape_value);
                op.attributes
                    .insert("polar_quant_seed".to_string(), Value::Int(self.seed as i64));

                // Update output type to Float16 to match the LUT dtype.
                // CoreML requires constexpr_lut_to_dense output dtype == LUT dtype.
                use crate::ir::tensor::TensorType;
                if !op.output_types.is_empty() {
                    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, shape.clone()));
                }

                // ── Phase 4: build row-norms const + mul ops ──────────────
                let norms_output = format!("{original_output}_polar_norms");
                let mul_output = format!("{original_output}_polar_scaled");

                let norms_data: Vec<u8> = row_norms
                    .iter()
                    .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                    .collect();

                let norms_op = Operation::new("const", format!("{op_name}_polar_norms"))
                    .with_input(
                        "val",
                        Value::Tensor {
                            data: norms_data,
                            shape: {
                                let mut s = shape.clone();
                                *s.last_mut().unwrap() = 1;
                                s
                            },
                            dtype: ScalarType::Float16,
                        },
                    )
                    .with_output(&norms_output);

                let mul_op = Operation::new("mul", format!("{op_name}_polar_mul"))
                    .with_input("x", Value::Reference(original_output.clone()))
                    .with_input("y", Value::Reference(norms_output))
                    .with_output(&mul_output);

                insertions.push((i, vec![norms_op, mul_op]));
                replacements.push((original_output, mul_output));
            }

            // Insert new ops after the loop (adjust indices for prior insertions).
            for (offset, (idx, new_ops)) in insertions.into_iter().enumerate() {
                let insert_at = idx + 1 + offset * 2;
                for (j, op) in new_ops.into_iter().enumerate() {
                    function.body.operations.insert(insert_at + j, op);
                }
            }

            // Rewire downstream references from the original const output to
            // the mul output. Then restore the mul op's own `x` input which
            // must still point to the original (dequantised) output.
            for (old_name, new_name) in &replacements {
                super::replace_reference(&mut function.body, old_name, new_name);
                // The mul op's `x` input was also rewritten — fix it back.
                for op in &mut function.body.operations {
                    if op.op_type == "mul" && op.outputs.iter().any(|o| o == new_name) {
                        if let Some(x_val) = op.inputs.get_mut("x") {
                            if matches!(x_val, Value::Reference(r) if r == new_name) {
                                *x_val = Value::Reference(old_name.clone());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
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

/// Pack assignment indices into n-bit packed bytes (MSB-first).
fn pack_indices(indices: &[usize], n_bits: u8) -> Vec<u8> {
    if n_bits == 8 {
        return indices.iter().map(|&i| i as u8).collect();
    }

    let mask = (1u16 << n_bits) - 1;
    let total_bits = indices.len() * n_bits as usize;
    let n_bytes = total_bits.div_ceil(8);
    let mut packed = vec![0u8; n_bytes];

    for (i, &idx) in indices.iter().enumerate() {
        let bit_offset = i * n_bits as usize;
        let byte_pos = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;
        let val = (idx as u16) & mask;

        let shifted = val << (16 - n_bits as usize - bit_in_byte);
        let [hi, lo] = shifted.to_be_bytes();
        packed[byte_pos] |= hi;
        if byte_pos + 1 < n_bytes {
            packed[byte_pos + 1] |= lo;
        }
    }

    packed
}

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

    #[test]
    fn skips_small_tensors() {
        let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let tensor_val = Value::Tensor {
            data: f32_bytes(&weights),
            shape: vec![8, 8],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(const_tensor_op("w", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        // 64 elements < 1024 min_elements → should be skipped.
        PolarQuantPass::new(4).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "const", "small tensor should be left unchanged");
    }

    #[test]
    fn quantises_large_tensor() {
        let rows = 64;
        let cols = 32;
        let numel = rows * cols; // 2048 >= 1024
        let weights: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.1).sin()).collect();

        let tensor_val = Value::Tensor {
            data: f32_bytes(&weights),
            shape: vec![rows, cols],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        PolarQuantPass::new(4).run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;

        // Should have 3 ops: constexpr_lut_to_dense, const (norms), mul.
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].op_type, "constexpr_lut_to_dense");
        assert_eq!(ops[1].op_type, "const");
        assert_eq!(ops[2].op_type, "mul");

        // LUT should have 16 entries (4-bit).
        match ops[0].attributes.get("lut") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(shape, &[16]);
                assert_eq!(*dtype, ScalarType::Float16);
            }
            other => panic!("expected LUT tensor, got {other:?}"),
        }

        // Rotation seed stored.
        match ops[0].attributes.get("polar_quant_seed") {
            Some(Value::Int(42)) => {}
            other => panic!("expected seed 42, got {other:?}"),
        }

        // Norms tensor shape is [rows, 1].
        match ops[1].inputs.get("val") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(shape, &[rows, 1]);
                assert_eq!(*dtype, ScalarType::Float16);
            }
            other => panic!("expected norms tensor, got {other:?}"),
        }

        // Block output should point to the mul output.
        let block_outputs = &program.functions["main"].body.outputs;
        assert_eq!(block_outputs, &["w_out_polar_scaled"]);
    }

    #[test]
    fn downstream_references_updated() {
        let rows = 32;
        let cols = 64;
        let numel = rows * cols;
        let weights: Vec<f32> = (0..numel).map(|i| (i as f32).cos()).collect();

        let tensor_val = Value::Tensor {
            data: f32_bytes(&weights),
            shape: vec![rows, cols],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        // Add a downstream op that references w_out.
        let relu = Operation::new("relu", "relu0")
            .with_input("x", Value::Reference("w_out".into()))
            .with_output("r_out");
        func.body.add_op(relu);
        func.body.outputs.push("r_out".into());
        program.add_function(func);

        PolarQuantPass::new(2).run(&mut program).unwrap();

        // The relu op should now reference the mul output.
        let ops = &program.functions["main"].body.operations;
        // ops: [constexpr_lut_to_dense, const(norms), mul, relu]
        assert_eq!(ops.len(), 4);
        let relu_op = &ops[3];
        assert_eq!(relu_op.op_type, "relu");
        match relu_op.inputs.get("x") {
            Some(Value::Reference(name)) => {
                assert_eq!(name, "w_out_polar_scaled");
            }
            other => panic!("expected reference to polar_scaled, got {other:?}"),
        }
    }
}
