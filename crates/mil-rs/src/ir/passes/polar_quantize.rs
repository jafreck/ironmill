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
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::{TensorData, Value};

use super::beta_quantizer::quantize_to_index;
use super::rotation::{pad_to_power_of_two, rotate_rows_hadamard};
use super::tensor_utils::tensor_as_f32_slice;

/// PolarQuant weight quantization pass.
///
/// For each eligible `const` op the pass:
/// 1. Extracts the weight matrix and normalises each row.
/// 2. Applies a seeded randomised Hadamard rotation.
/// 3. Quantises using Beta-optimal levels/boundaries.
/// 4. Replaces the op with `constexpr_lut_to_dense` + row-norm rescaling.
#[non_exhaustive]
pub struct PolarQuantPass {
    /// Number of bits per quantised index.
    pub n_bits: u8,
    /// Seed for the randomised Hadamard rotation (deterministic).
    pub seed: u64,
    /// Minimum number of tensor elements to apply quantisation.
    pub min_elements: usize,
}

impl PolarQuantPass {
    /// Create a new polar quantization pass with the given bit width.
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
        validate_lut_size(self.n_bits, k)?;
        let provider = program.weight_provider.clone();
        let spill_index = program.spill_index.clone();
        let resolve = super::util::make_resolver(&provider, &spill_index);

        for function in program.functions.values_mut() {
            let mut insertions: Vec<(usize, Vec<Operation>)> = Vec::new();
            let mut replacements: Vec<(String, String)> = Vec::new();

            for i in 0..function.body.operations.len() {
                // Materialize External tensors before extracting.
                if let Some(Value::Tensor { data, .. }) =
                    function.body.operations[i].inputs.get_mut("val")
                {
                    data.materialize_with(|key| resolve(key))?;
                }
                if let Some(Value::Tensor { data, .. }) =
                    function.body.operations[i].attributes.get_mut("val")
                {
                    data.materialize_with(|key| resolve(key))?;
                }

                let eligible =
                    extract_eligible_tensor(&function.body.operations[i], self.min_elements);
                let info = match eligible {
                    Some(t) => t,
                    None => continue,
                };

                let quant = match compute_quantization(&info, self.n_bits, self.seed) {
                    Some(q) => q,
                    None => continue,
                };

                rewrite_op_in_place(
                    &mut function.body.operations[i],
                    &quant,
                    &info,
                    k,
                    self.seed,
                );

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

/// Results of quantizing a tensor.
struct QuantizedTensor {
    packed: Vec<u8>,
    levels: Vec<f32>,
    row_norms: Vec<f32>,
    padded_shape: Vec<usize>,
}

/// Validate that the LUT size is accepted by CoreML.
fn validate_lut_size(n_bits: u8, k: usize) -> Result<()> {
    const VALID_LUT_SIZES: &[usize] = &[2, 4, 16, 64, 256];
    if !VALID_LUT_SIZES.contains(&k) {
        return Err(crate::error::MilError::Validation(format!(
            "PolarQuant: n_bits={} produces LUT size {} which is not supported by CoreML \
             (valid LUT sizes: 2, 4, 16, 64, 256 → n_bits ∈ {{1, 2, 4, 6, 8}})",
            n_bits, k
        )));
    }
    Ok(())
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

/// Phase 2: normalise, rotate, and quantise the tensor.
///
/// Returns `None` for tensors that don't meet shape requirements (rank < 2
/// or last dimension < 64).
fn compute_quantization(info: &EligibleTensor, n_bits: u8, seed: u64) -> Option<QuantizedTensor> {
    let rank = info.shape.len();
    // Skip 1D tensors (layer norms, biases).
    if rank < 2 {
        return None;
    }

    let cols = info.shape[rank - 1];
    let rows: usize = info.shape[..rank - 1].iter().product();

    // PolarQuant's statistical guarantees require N >= 64.
    if cols < 64 {
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

    // Symmetric absmax quantizer.
    let n_levels = 1usize << n_bits;
    let absmax = padded_data
        .iter()
        .fold(0.0f32, |m, &v| m.max(v.abs()))
        .max(1e-10);

    let step = 2.0 * absmax / n_levels as f32;
    let levels: Vec<f32> = (0..n_levels)
        .map(|i| -absmax + step * (i as f32 + 0.5))
        .collect();

    let boundaries: Vec<f32> = levels.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

    let all_indices: Vec<usize> = padded_data
        .iter()
        .map(|&v| {
            let clamped = v.clamp(-absmax, absmax);
            quantize_to_index(clamped, &boundaries) as usize
        })
        .collect();

    let packed = pack_indices(&all_indices, n_bits);

    let mut padded_shape = info.shape.clone();
    // Safety: shape cloned from tensor, guaranteed non-empty.
    *padded_shape.last_mut().unwrap() = padded_cols;

    Some(QuantizedTensor {
        packed,
        levels,
        row_norms,
        padded_shape,
    })
}

/// Phase 3: mutate the original const op into `constexpr_lut_to_dense`.
fn rewrite_op_in_place(
    op: &mut Operation,
    quant: &QuantizedTensor,
    info: &EligibleTensor,
    k: usize,
    seed: u64,
) {
    let lut_data: Vec<u8> = match info.original_dtype {
        ScalarType::Float16 => quant
            .levels
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect(),
        _ => quant.levels.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    let lut_value = Value::Tensor {
        data: TensorData::Inline(lut_data),
        shape: vec![k],
        dtype: info.original_dtype,
    };

    let indices_value = Value::Tensor {
        data: TensorData::Inline(quant.packed.clone()),
        shape: vec![quant.packed.len()],
        dtype: ScalarType::UInt8,
    };

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

    op.op_type = "constexpr_lut_to_dense".to_string();
    op.inputs.remove("val");
    op.attributes.remove("val");
    op.attributes.insert("lut".to_string(), lut_value);
    op.attributes.insert("indices".to_string(), indices_value);
    op.attributes.insert("shape".to_string(), shape_value);
    op.attributes
        .insert("polar_quant_seed".to_string(), Value::Int(seed as i64));

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
    quant: &QuantizedTensor,
) -> (Operation, Operation, String) {
    let norms_output = format!("{}_polar_norms", info.original_output);
    let mul_output = format!("{}_polar_scaled", info.original_output);

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

    let mut norms_op = Operation::new("const", format!("{}_polar_norms", info.op_name))
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

    let mut mul_op = Operation::new("mul", format!("{}_polar_mul", info.op_name))
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

/// Pack assignment indices into n-bit packed bytes (MSB-first).
pub fn pack_indices(indices: &[usize], n_bits: u8) -> Vec<u8> {
    if n_bits == 8 {
        return indices.iter().map(|&i| i as u8).collect();
    }

    let mask = (1u16 << n_bits) - 1;
    let total_bits = indices.len() * n_bits as usize;
    let n_bytes = total_bits.div_ceil(8);
    // Allocate one extra byte so the last value's lo byte always has a
    // valid destination, then truncate back to the true output size.
    let mut packed = vec![0u8; n_bytes + 1];

    for (i, &idx) in indices.iter().enumerate() {
        let bit_offset = i * n_bits as usize;
        let byte_pos = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;
        let val = (idx as u16) & mask;

        let shifted = val << (16 - n_bits as usize - bit_in_byte);
        let [hi, lo] = shifted.to_be_bytes();
        packed[byte_pos] |= hi;
        packed[byte_pos + 1] |= lo;
    }

    packed.truncate(n_bytes);
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
            data: TensorData::Inline(f32_bytes(&weights)),
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
        let cols = 64;
        let numel = rows * cols; // 4096 >= 1024, cols >= 64
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

        PolarQuantPass::new(4).run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;

        // Should have 3 ops: constexpr_lut_to_dense, const (norms), mul.
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].op_type, "constexpr_lut_to_dense");
        assert_eq!(ops[1].op_type, "const");
        assert_eq!(ops[2].op_type, "mul");

        // LUT should have 16 entries (4-bit), matching input dtype (Float32).
        match ops[0].attributes.get("lut") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(shape, &[16]);
                assert_eq!(*dtype, ScalarType::Float32);
            }
            other => panic!("expected LUT tensor, got {other:?}"),
        }

        // Rotation seed stored.
        match ops[0].attributes.get("polar_quant_seed") {
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
        assert_eq!(block_outputs, &["w_out_polar_scaled"]);
    }

    #[test]
    fn downstream_references_updated() {
        let rows = 32;
        let cols = 64;
        let numel = rows * cols;
        let weights: Vec<f32> = (0..numel).map(|i| (i as f32).cos()).collect();

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
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

    #[test]
    fn rejects_3_bit_lut_size() {
        let mut program = Program::new("1.0.0");
        let func = Function::new("main");
        program.add_function(func);

        let result = PolarQuantPass::new(3).run(&mut program);
        assert!(
            result.is_err(),
            "n_bits=3 should be rejected (LUT size 8 not valid)"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("LUT size 8"),
            "error should mention LUT size: {msg}"
        );
    }

    #[test]
    fn accepts_valid_bit_widths() {
        let mut program = Program::new("1.0.0");
        let func = Function::new("main");
        program.add_function(func);

        for bits in [1, 2, 4, 6, 8] {
            let result = PolarQuantPass::new(bits).run(&mut program);
            assert!(result.is_ok(), "n_bits={bits} should be accepted");
        }
    }

    #[test]
    fn norms_op_has_output_types() {
        let rows = 32;
        let cols = 64;
        let numel = rows * cols;
        let weights: Vec<f32> = (0..numel).map(|i| (i as f32).cos()).collect();

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

        PolarQuantPass::new(4).run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        // norms const op (index 1) should have output_types set
        let norms_op = &ops[1];
        assert_eq!(norms_op.op_type, "const");
        assert!(
            !norms_op.output_types.is_empty() && norms_op.output_types[0].is_some(),
            "norms const op should have explicit output type"
        );

        // mul op (index 2) should have output_types set
        let mul_op = &ops[2];
        assert_eq!(mul_op.op_type, "mul");
        assert!(
            !mul_op.output_types.is_empty() && mul_op.output_types[0].is_some(),
            "mul op should have explicit output type"
        );
    }
}
