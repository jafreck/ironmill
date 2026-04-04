//! Weight palettization pass.
//!
//! Compresses FP32/FP16 weight tensors via k-means clustering into lookup
//! tables, converting `const` ops into `constexpr_lut_to_dense` ops.

use half::f16;

use crate::error::{MilError, Result};
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::ScalarType;
use crate::ir::types::{TensorData, Value};

use super::kmeans::kmeans;
use super::tensor_utils::tensor_as_f32_slice;

/// Compress weights using k-means palettization.
///
/// Each weight tensor is clustered into 2^n_bits centroids. The tensor is
/// then stored as indices + a palette lookup table, reducing size by
/// 32/n_bits × (roughly 8× for 4-bit, ~5× for 6-bit).
pub struct PalettizePass {
    /// Number of bits per weight index (1, 2, 4, 6, or 8).
    n_bits: u8,
    /// Maximum k-means iterations.
    max_iter: usize,
}

impl PalettizePass {
    pub fn new(n_bits: u8) -> Self {
        assert!(
            matches!(n_bits, 1 | 2 | 4 | 6 | 8),
            "n_bits must be one of 1, 2, 4, 6, or 8"
        );
        Self {
            n_bits,
            max_iter: 100,
        }
    }
}

/// Compress weights using grouped k-means palettization.
///
/// Instead of one codebook per tensor, the output-channel dimension is split
/// into groups of `group_size` channels and each group gets its own
/// independent k-means clustering and codebook. This yields better accuracy
/// than a single global codebook at a modest increase in LUT storage.
pub struct GroupedPalettizePass {
    /// Number of bits per weight index (1, 2, 4, 6, or 8).
    n_bits: u8,
    /// Maximum k-means iterations per group.
    max_iter: usize,
    /// Number of output channels per group (default 128).
    group_size: usize,
}

impl GroupedPalettizePass {
    /// Create a grouped palettization pass.
    ///
    /// `group_size` is the number of output channels per group. The last
    /// group may be smaller if the output-channel dimension is not evenly
    /// divisible.
    pub fn new(n_bits: u8, group_size: usize) -> Self {
        assert!(
            matches!(n_bits, 1 | 2 | 4 | 6 | 8),
            "n_bits must be one of 1, 2, 4, 6, or 8"
        );
        assert!(group_size > 0, "group_size must be at least 1");
        Self {
            n_bits,
            max_iter: 100,
            group_size,
        }
    }
}

impl Pass for PalettizePass {
    fn name(&self) -> &str {
        "palettization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        let k = 1usize << self.n_bits; // 2^n_bits centroids

        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                if op.op_type != "const" {
                    continue;
                }

                // val may live in inputs or attributes depending on the
                // frontend (ONNX import puts it in attributes).
                let val = match op.inputs.get("val").or_else(|| op.attributes.get("val")) {
                    Some(v) => v,
                    None => continue,
                };

                let (floats, shape, original_dtype) = match val {
                    Value::Tensor {
                        data,
                        shape,
                        dtype: dtype @ (ScalarType::Float32 | ScalarType::Float16),
                    } => {
                        let bytes = data.as_bytes().expect("tensor not materialized");
                        let f = match dtype {
                            ScalarType::Float32 => tensor_as_f32_slice(bytes),
                            ScalarType::Float16 => fp16_bytes_to_f32(bytes)?,
                            other => {
                                return Err(MilError::TypeMismatch {
                                    expected: "Float32 or Float16".into(),
                                    actual: format!("{other:?}"),
                                });
                            }
                        };
                        (f, shape.clone(), *dtype)
                    }
                    _ => continue,
                };

                if floats.is_empty() {
                    continue;
                }

                let (centroids, assignments) = kmeans(&floats, k, self.max_iter)?;
                let lut_data: Vec<u8> = match original_dtype {
                    ScalarType::Float16 => centroids
                        .iter()
                        .flat_map(|&c| f16::from_f32(c).to_le_bytes())
                        .collect(),
                    _ => centroids.iter().flat_map(|c| c.to_le_bytes()).collect(),
                };

                let lut_value = Value::Tensor {
                    data: TensorData::Inline(lut_data),
                    shape: vec![k],
                    dtype: original_dtype,
                };

                // Pack indices into n-bit representation.
                let packed_indices = pack_indices(&assignments, self.n_bits);

                // CoreML expects the indices as a 1D (rank 1) tensor of
                // packed bytes; the output shape comes from the `shape` attr.
                let indices_value = Value::Tensor {
                    data: TensorData::Inline(packed_indices.clone()),
                    shape: vec![packed_indices.len()],
                    dtype: ScalarType::UInt8,
                };

                // Shape must be a UInt32 tensor per the CoreML spec for
                // constexpr_lut_to_dense.
                let shape_bytes: Vec<u8> = shape
                    .iter()
                    .flat_map(|&d| (d as u32).to_le_bytes())
                    .collect();
                let shape_value = Value::Tensor {
                    data: TensorData::Inline(shape_bytes),
                    shape: vec![shape.len()],
                    dtype: ScalarType::UInt32,
                };

                // Transform the op.
                op.op_type = "constexpr_lut_to_dense".to_string();
                op.inputs.remove("val");
                op.attributes.remove("val");
                op.attributes.insert("lut".to_string(), lut_value);
                op.attributes.insert("indices".to_string(), indices_value);
                op.attributes.insert("shape".to_string(), shape_value);
            }
        }
        Ok(())
    }
}

impl Pass for GroupedPalettizePass {
    fn name(&self) -> &str {
        "grouped-palettization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        let k = 1usize << self.n_bits;

        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                if op.op_type != "const" {
                    continue;
                }

                let val = match op.inputs.get("val").or_else(|| op.attributes.get("val")) {
                    Some(v) => v,
                    None => continue,
                };

                let (floats, shape, original_dtype) = match val {
                    Value::Tensor {
                        data,
                        shape,
                        dtype: dtype @ (ScalarType::Float32 | ScalarType::Float16),
                    } => {
                        let bytes = data.as_bytes().expect("tensor not materialized");
                        let f = match dtype {
                            ScalarType::Float32 => tensor_as_f32_slice(bytes),
                            ScalarType::Float16 => fp16_bytes_to_f32(bytes)?,
                            other => {
                                return Err(MilError::TypeMismatch {
                                    expected: "Float32 or Float16".into(),
                                    actual: format!("{other:?}"),
                                });
                            }
                        };
                        (f, shape.clone(), *dtype)
                    }
                    _ => continue,
                };

                if floats.is_empty() {
                    continue;
                }

                // Need at least 2-D tensor to group along the output channel
                // (first) dimension. Fall back to ungrouped for 1-D tensors.
                if shape.len() < 2 {
                    let (centroids, assignments) = kmeans(&floats, k, self.max_iter)?;
                    apply_lut_transform(
                        op,
                        &centroids,
                        &assignments,
                        &shape,
                        original_dtype,
                        self.n_bits,
                        k,
                    );
                    continue;
                }

                let out_channels = shape[0];
                let elements_per_channel: usize = shape[1..].iter().product();
                let n_groups = out_channels.div_ceil(self.group_size);

                // Guard: grouped palettization concatenates per-group LUTs into
                // a single index space. If n_groups * k > 256 the indices would
                // exceed the 8-bit packing limit — fall back to ungrouped.
                if n_groups * k > 256 {
                    let (centroids, assignments) = kmeans(&floats, k, self.max_iter)?;
                    apply_lut_transform(
                        op,
                        &centroids,
                        &assignments,
                        &shape,
                        original_dtype,
                        self.n_bits,
                        k,
                    );
                    continue;
                }

                // Each group gets independent clustering; concatenate LUTs and
                // remap indices so group g uses centroid range [g*k .. (g+1)*k).
                let mut all_lut_values: Vec<f32> = Vec::with_capacity(n_groups * k);
                let mut all_assignments: Vec<usize> = Vec::with_capacity(floats.len());

                for g in 0..n_groups {
                    let ch_start = g * self.group_size;
                    let ch_end = (ch_start + self.group_size).min(out_channels);
                    let elem_start = ch_start * elements_per_channel;
                    let elem_end = ch_end * elements_per_channel;
                    let group_data = &floats[elem_start..elem_end];

                    let (centroids, assignments) = kmeans(group_data, k, self.max_iter)?;

                    all_lut_values.extend_from_slice(&centroids);
                    // Offset indices so each group occupies its own range.
                    let base = g * k;
                    all_assignments.extend(assignments.iter().map(|&a| a + base));
                }

                let total_lut_entries = n_groups * k;

                let lut_data: Vec<u8> = match original_dtype {
                    ScalarType::Float16 => all_lut_values
                        .iter()
                        .flat_map(|&c| f16::from_f32(c).to_le_bytes())
                        .collect(),
                    _ => all_lut_values
                        .iter()
                        .flat_map(|c| c.to_le_bytes())
                        .collect(),
                };

                let lut_value = Value::Tensor {
                    data: TensorData::Inline(lut_data),
                    shape: vec![total_lut_entries],
                    dtype: original_dtype,
                };

                // Compute the bit-width required for the grouped index space.
                // Each index must address total_lut_entries values.
                let grouped_bits = bits_for(total_lut_entries);
                let packed_indices = pack_indices(&all_assignments, grouped_bits);

                let indices_value = Value::Tensor {
                    data: TensorData::Inline(packed_indices.clone()),
                    shape: vec![packed_indices.len()],
                    dtype: ScalarType::UInt8,
                };

                let shape_bytes: Vec<u8> = shape
                    .iter()
                    .flat_map(|&d| (d as u32).to_le_bytes())
                    .collect();
                let shape_value = Value::Tensor {
                    data: TensorData::Inline(shape_bytes),
                    shape: vec![shape.len()],
                    dtype: ScalarType::UInt32,
                };

                op.op_type = "constexpr_lut_to_dense".to_string();
                op.inputs.remove("val");
                op.attributes.remove("val");
                op.attributes.insert("lut".to_string(), lut_value);
                op.attributes.insert("indices".to_string(), indices_value);
                op.attributes.insert("shape".to_string(), shape_value);
                op.attributes
                    .insert("n_groups".to_string(), Value::Int(n_groups as i64));
                op.attributes
                    .insert("group_size".to_string(), Value::Int(self.group_size as i64));
            }
        }
        Ok(())
    }
}

/// Apply the standard LUT transform to an operation (shared helper).
fn apply_lut_transform(
    op: &mut crate::ir::operation::Operation,
    centroids: &[f32],
    assignments: &[usize],
    shape: &[usize],
    original_dtype: ScalarType,
    n_bits: u8,
    k: usize,
) {
    let lut_data: Vec<u8> = match original_dtype {
        ScalarType::Float16 => centroids
            .iter()
            .flat_map(|&c| f16::from_f32(c).to_le_bytes())
            .collect(),
        _ => centroids.iter().flat_map(|c| c.to_le_bytes()).collect(),
    };
    let lut_value = Value::Tensor {
        data: TensorData::Inline(lut_data),
        shape: vec![k],
        dtype: original_dtype,
    };
    let packed_indices = pack_indices(assignments, n_bits);
    let indices_value = Value::Tensor {
        data: TensorData::Inline(packed_indices.clone()),
        shape: vec![packed_indices.len()],
        dtype: ScalarType::UInt8,
    };
    let shape_bytes: Vec<u8> = shape
        .iter()
        .flat_map(|&d| (d as u32).to_le_bytes())
        .collect();
    let shape_value = Value::Tensor {
        data: TensorData::Inline(shape_bytes),
        shape: vec![shape.len()],
        dtype: ScalarType::UInt32,
    };
    op.op_type = "constexpr_lut_to_dense".to_string();
    op.inputs.remove("val");
    op.attributes.remove("val");
    op.attributes.insert("lut".to_string(), lut_value);
    op.attributes.insert("indices".to_string(), indices_value);
    op.attributes.insert("shape".to_string(), shape_value);
}

/// Compute the minimum number of bits required to index `n` entries.
fn bits_for(n: usize) -> u8 {
    if n <= 2 {
        return 1;
    }
    let b = (usize::BITS - (n - 1).leading_zeros()) as u8;
    // Round up to a supported pack width (1, 2, 4, 6, 8).
    match b {
        0..=1 => 1,
        2 => 2,
        3..=4 => 4,
        5..=6 => 6,
        7..=8 => 8,
        _ => 8,
    }
}

/// Convert raw FP16 little-endian bytes to `Vec<f32>`.
fn fp16_bytes_to_f32(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() % 2 != 0 {
        return Err(MilError::Validation(format!(
            "FP16 tensor data length must be a multiple of 2, got {}",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect())
}

/// Pack assignment indices into n-bit packed bytes.
///
/// Uses bit-level packing that works for all supported widths (1, 2, 4, 6, 8),
/// including widths that cause indices to span byte boundaries (e.g. 6-bit).
/// Bits are written MSB-first within each byte.
fn pack_indices(indices: &[usize], n_bits: u8) -> Vec<u8> {
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

        // Shift value so MSB aligns with bit_in_byte within a 16-bit window.
        let shifted = val << (16 - n_bits as usize - bit_in_byte);
        let [hi, lo] = shifted.to_be_bytes();
        packed[byte_pos] |= hi;
        packed[byte_pos + 1] |= lo;
    }

    packed.truncate(n_bytes);
    packed
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::Function;

    /// Create FP32 tensor bytes from a slice of f32 values.
    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Helper: build a `const` op with a tensor value.
    fn const_tensor_op(name: &str, output: &str, value: Value) -> Operation {
        Operation::new("const", name)
            .with_input("val", value)
            .with_output(output)
    }

    #[test]
    fn palettize_4bit_produces_16_centroids() {
        // 64 weights clustered around 4 groups
        let mut weights = Vec::new();
        for &center in &[0.0f32, 1.0, 2.0, 3.0] {
            for i in 0..16 {
                weights.push(center + i as f32 * 0.01);
            }
        }

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![64],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        PalettizePass::new(4).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_lut_to_dense");
        assert!(op.inputs.get("val").is_none());

        // LUT should have 16 centroids.
        match op.attributes.get("lut") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(*shape, vec![16]);
                assert_eq!(*dtype, ScalarType::Float32);
            }
            other => panic!("expected lut tensor, got {other:?}"),
        }

        // Indices: 64 values packed at 4 bits = 32 packed bytes.
        match op.attributes.get("indices") {
            Some(Value::Tensor { data, shape, .. }) => {
                assert_eq!(*shape, vec![32]); // packed byte count
                assert_eq!(data.byte_len(), 32); // 64 × 4 bits / 8
            }
            other => panic!("expected indices tensor, got {other:?}"),
        }

        // Shape attribute preserved as UInt32 tensor.
        match op.attributes.get("shape") {
            Some(Value::Tensor { data, shape, dtype }) => {
                assert_eq!(*dtype, ScalarType::UInt32);
                assert_eq!(*shape, vec![1]);
                let bytes = data.as_bytes().expect("tensor not materialized");
                let dim = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                assert_eq!(dim, 64);
            }
            other => panic!("expected shape tensor, got {other:?}"),
        }
    }

    #[test]
    fn palettize_2bit_produces_4_centroids() {
        let weights: Vec<f32> = (0..32).map(|i| (i % 4) as f32).collect();

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![32],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        PalettizePass::new(2).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_lut_to_dense");

        match op.attributes.get("lut") {
            Some(Value::Tensor { shape, .. }) => {
                assert_eq!(*shape, vec![4]);
            }
            other => panic!("expected lut with 4 centroids, got {other:?}"),
        }

        // 32 values × 2 bits = 8 packed bytes.
        match op.attributes.get("indices") {
            Some(Value::Tensor { data, shape, .. }) => {
                assert_eq!(*shape, vec![8]); // packed byte count
                assert_eq!(data.byte_len(), 8);
            }
            other => panic!("expected indices tensor, got {other:?}"),
        }
    }

    #[test]
    fn compression_ratio() {
        let n = 256;
        let weights: Vec<f32> = (0..n).map(|i| (i % 16) as f32).collect();
        let original_size = n * 4; // FP32 = 4 bytes each

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![n],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        PalettizePass::new(4).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        let lut_size = match op.attributes.get("lut") {
            Some(Value::Tensor { data, .. }) => data.byte_len(),
            _ => panic!("missing lut"),
        };
        let idx_size = match op.attributes.get("indices") {
            Some(Value::Tensor { data, .. }) => data.byte_len(),
            _ => panic!("missing indices"),
        };

        let compressed_size = lut_size + idx_size;
        assert!(
            compressed_size < original_size,
            "compressed {compressed_size} should be < original {original_size}"
        );
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

        PalettizePass::new(4).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "relu");
    }

    #[test]
    fn leaves_non_float_const_unchanged() {
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

        PalettizePass::new(4).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "const");
    }

    #[test]
    fn palettize_fp16_tensor() {
        let weights: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let fp16_data: Vec<u8> = weights
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(fp16_data),
            shape: vec![16],
            dtype: ScalarType::Float16,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        PalettizePass::new(4).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_lut_to_dense");

        // LUT should preserve FP16 dtype.
        match op.attributes.get("lut") {
            Some(Value::Tensor { dtype, shape, .. }) => {
                assert_eq!(*dtype, ScalarType::Float16);
                assert_eq!(*shape, vec![16]);
            }
            other => panic!("expected FP16 lut, got {other:?}"),
        }
    }

    #[test]
    fn pack_indices_4bit() {
        // Two indices per byte, MSB-first.
        let indices: Vec<usize> = vec![0, 1, 2, 3, 15, 14, 13, 12];
        let packed = pack_indices(&indices, 4);
        assert_eq!(packed.len(), 4);
        assert_eq!(packed[0], 0x01); // (0 << 4) | 1
        assert_eq!(packed[1], 0x23); // (2 << 4) | 3
        assert_eq!(packed[2], 0xFE); // (15 << 4) | 14
        assert_eq!(packed[3], 0xDC); // (13 << 4) | 12
    }

    #[test]
    fn pack_indices_2bit() {
        // Four indices per byte.
        let indices: Vec<usize> = vec![0, 1, 2, 3];
        let packed = pack_indices(&indices, 2);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0b00_01_10_11);
    }

    #[test]
    fn pack_indices_8bit() {
        let indices: Vec<usize> = vec![0, 127, 255];
        let packed = pack_indices(&indices, 8);
        assert_eq!(packed, vec![0, 127, 255]);
    }

    #[test]
    fn pack_indices_6bit() {
        // 4 indices × 6 bits = 24 bits = 3 bytes.
        // idx 0: 0b000001 (1), idx 1: 0b000010 (2), idx 2: 0b111111 (63), idx 3: 0b100000 (32)
        // Bit layout MSB-first:
        //   byte 0: 000001_00  => bits[0..6]=idx0, bits[6..8]=top 2 of idx1 => 0b00000100 = 0x04
        //   byte 1: 0010_1111  => bits[0..4]=bottom 4 of idx1, bits[4..8]=top 4 of idx2 => 0b00101111 = 0x2F
        //   byte 2: 11_100000  => bits[0..2]=bottom 2 of idx2, bits[2..8]=idx3 => 0b11100000 = 0xE0
        let indices: Vec<usize> = vec![1, 2, 63, 32];
        let packed = pack_indices(&indices, 6);
        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0], 0x04);
        assert_eq!(packed[1], 0x2F);
        assert_eq!(packed[2], 0xE0);
    }

    // ---- 1-bit palettization tests ----------------------------------------

    #[test]
    fn palettize_1bit_produces_2_centroids() {
        // 32 weights clearly split into two clusters.
        let mut weights = Vec::new();
        for _ in 0..16 {
            weights.push(-1.0f32);
        }
        for _ in 0..16 {
            weights.push(1.0f32);
        }

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![32],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        PalettizePass::new(1).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_lut_to_dense");

        // LUT should have 2 centroids.
        match op.attributes.get("lut") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(*shape, vec![2]);
                assert_eq!(*dtype, ScalarType::Float32);
            }
            other => panic!("expected lut with 2 centroids, got {other:?}"),
        }

        // 32 values × 1 bit = 4 packed bytes.
        match op.attributes.get("indices") {
            Some(Value::Tensor { data, shape, .. }) => {
                assert_eq!(*shape, vec![4]);
                assert_eq!(data.byte_len(), 4);
            }
            other => panic!("expected indices tensor, got {other:?}"),
        }
    }

    #[test]
    fn palettize_1bit_round_trip_accuracy() {
        // Weights near -1 and +1 should round-trip with minimal error.
        let weights: Vec<f32> = (0..64)
            .map(|i| {
                if i < 32 {
                    -1.0 + (i as f32) * 0.001
                } else {
                    1.0 + ((i - 32) as f32) * 0.001
                }
            })
            .collect();

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![64],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        PalettizePass::new(1).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_lut_to_dense");

        // Verify the LUT contains values close to the two cluster centers.
        match op.attributes.get("lut") {
            Some(Value::Tensor { data, shape, dtype }) => {
                assert_eq!(*shape, vec![2]);
                assert_eq!(*dtype, ScalarType::Float32);
                let bytes = data.as_bytes().expect("tensor not materialized");
                let c0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
                let c1 = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
                let (lo, hi) = if c0 < c1 { (c0, c1) } else { (c1, c0) };
                assert!(
                    (lo - (-1.0)).abs() < 0.1,
                    "low centroid {lo} should be near -1.0"
                );
                assert!(
                    (hi - 1.0).abs() < 0.1,
                    "high centroid {hi} should be near 1.0"
                );
            }
            other => panic!("expected lut, got {other:?}"),
        }
    }

    #[test]
    fn pack_indices_1bit() {
        // 8 indices × 1 bit = 1 byte.
        let indices: Vec<usize> = vec![1, 0, 1, 1, 0, 0, 1, 0];
        let packed = pack_indices(&indices, 1);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0b10110010);
    }

    // ---- Grouped palettization tests --------------------------------------

    #[test]
    fn grouped_palettize_2d_tensor() {
        // 4 output channels × 8 elements each = shape [4, 8].
        // Two groups of 2 channels each (group_size=2).
        let mut weights = Vec::new();
        // Group 0 (channels 0-1): values near 0.0 and 1.0
        for _ in 0..2 {
            for i in 0..8 {
                weights.push(if i < 4 { 0.0 } else { 1.0 });
            }
        }
        // Group 1 (channels 2-3): values near 10.0 and 20.0
        for _ in 0..2 {
            for i in 0..8 {
                weights.push(if i < 4 { 10.0 } else { 20.0 });
            }
        }

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![4, 8],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        GroupedPalettizePass::new(2, 2).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_lut_to_dense");

        // 2 groups × 4 centroids (2-bit) = 8 total LUT entries.
        match op.attributes.get("lut") {
            Some(Value::Tensor { shape, dtype, .. }) => {
                assert_eq!(*shape, vec![8]);
                assert_eq!(*dtype, ScalarType::Float32);
            }
            other => panic!("expected grouped lut, got {other:?}"),
        }

        // n_groups and group_size metadata.
        match op.attributes.get("n_groups") {
            Some(Value::Int(n)) => assert_eq!(*n, 2),
            other => panic!("expected n_groups=2, got {other:?}"),
        }
        match op.attributes.get("group_size") {
            Some(Value::Int(gs)) => assert_eq!(*gs, 2),
            other => panic!("expected group_size=2, got {other:?}"),
        }
    }

    #[test]
    fn grouped_palettize_falls_back_for_1d() {
        // 1-D tensor should still be palettized (ungrouped fallback).
        let weights: Vec<f32> = (0..16).map(|i| (i % 4) as f32).collect();

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![16],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        GroupedPalettizePass::new(2, 4).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_lut_to_dense");

        match op.attributes.get("lut") {
            Some(Value::Tensor { shape, .. }) => {
                assert_eq!(*shape, vec![4]);
            }
            other => panic!("expected lut with 4 centroids, got {other:?}"),
        }
    }

    #[test]
    fn grouped_palettize_uneven_groups() {
        // 3 output channels with group_size=2 ⇒ 2 groups (sizes 2 and 1).
        let mut weights = Vec::new();
        for ch in 0..3 {
            for i in 0..4 {
                weights.push(ch as f32 * 10.0 + i as f32);
            }
        }

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![3, 4],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        GroupedPalettizePass::new(2, 2).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_lut_to_dense");

        // 2 groups × 4 centroids = 8 LUT entries.
        match op.attributes.get("lut") {
            Some(Value::Tensor { shape, .. }) => {
                assert_eq!(*shape, vec![8]);
            }
            other => panic!("expected grouped lut with 8 entries, got {other:?}"),
        }

        match op.attributes.get("n_groups") {
            Some(Value::Int(n)) => assert_eq!(*n, 2),
            other => panic!("expected n_groups=2, got {other:?}"),
        }
    }

    #[test]
    fn grouped_palettize_round_trip_accuracy() {
        // Each group has distinct value ranges; grouped quantization should
        // yield lower error than a single global codebook would.
        let mut weights = Vec::new();
        // Group 0: values in [0, 1]
        for i in 0..32 {
            weights.push(i as f32 / 31.0);
        }
        // Group 1: values in [100, 101]
        for i in 0..32 {
            weights.push(100.0 + i as f32 / 31.0);
        }

        let original = weights.clone();
        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![2, 32],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        GroupedPalettizePass::new(4, 1).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];

        // Reconstruct approximate values from LUT + indices.
        let lut_data = match op.attributes.get("lut") {
            Some(Value::Tensor { data, .. }) => data.as_bytes().expect("tensor not materialized"),
            _ => panic!("missing lut"),
        };
        let lut: Vec<f32> = lut_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // For grouped quantization the indices span the full grouped range,
        // so we need the grouped_bits width to unpack.
        let n_groups = match op.attributes.get("n_groups") {
            Some(Value::Int(n)) => *n as usize,
            _ => 1,
        };
        let k = 1usize << 4; // 16
        let total_entries = n_groups * k;
        let grouped_bits = bits_for(total_entries);

        let idx_data = match op.attributes.get("indices") {
            Some(Value::Tensor { data, .. }) => data.as_bytes().expect("tensor not materialized"),
            _ => panic!("missing indices"),
        };
        let unpacked = unpack_indices(idx_data, original.len(), grouped_bits);

        let mut max_err: f32 = 0.0;
        for (i, &orig) in original.iter().enumerate() {
            let approx = lut[unpacked[i]];
            let err = (orig - approx).abs();
            if err > max_err {
                max_err = err;
            }
        }

        // 4-bit per-channel (group_size=1) should be very accurate.
        assert!(max_err < 0.1, "max round-trip error {max_err} exceeds 0.1");
    }

    /// Unpack n-bit packed indices (inverse of `pack_indices`).
    fn unpack_indices(packed: &[u8], count: usize, n_bits: u8) -> Vec<usize> {
        if n_bits == 8 {
            return packed.iter().map(|&b| b as usize).collect();
        }

        let mask = (1u16 << n_bits) - 1;
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let bit_offset = i * n_bits as usize;
            let byte_pos = bit_offset / 8;
            let bit_in_byte = bit_offset % 8;
            let window = if byte_pos + 1 < packed.len() {
                ((packed[byte_pos] as u16) << 8) | packed[byte_pos + 1] as u16
            } else {
                (packed[byte_pos] as u16) << 8
            };
            let val = (window >> (16 - n_bits as usize - bit_in_byte)) & mask;
            result.push(val as usize);
        }
        result
    }

    #[test]
    fn grouped_palettize_overflow_falls_back_to_ungrouped() {
        // 128 output channels with group_size=1 and 4-bit (k=16) ⇒
        // n_groups * k = 128 * 16 = 2048, which exceeds 256.
        // The pass should fall back to ungrouped palettization.
        let mut weights = Vec::new();
        for ch in 0..128 {
            for i in 0..4 {
                weights.push(ch as f32 + i as f32 * 0.1);
            }
        }

        let tensor_val = Value::Tensor {
            data: TensorData::Inline(f32_bytes(&weights)),
            shape: vec![128, 4],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body
            .add_op(const_tensor_op("weight", "w_out", tensor_val));
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        GroupedPalettizePass::new(4, 1).run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_lut_to_dense");

        // Fell back to ungrouped: LUT has exactly k=16 entries, no n_groups attr.
        match op.attributes.get("lut") {
            Some(Value::Tensor { shape, .. }) => {
                assert_eq!(*shape, vec![16], "should have ungrouped 16-entry LUT");
            }
            other => panic!("expected ungrouped lut, got {other:?}"),
        }
        assert!(
            op.attributes.get("n_groups").is_none(),
            "ungrouped fallback should not set n_groups"
        );
    }

    #[test]
    fn bits_for_coverage() {
        assert_eq!(bits_for(1), 1);
        assert_eq!(bits_for(2), 1);
        assert_eq!(bits_for(3), 2);
        assert_eq!(bits_for(4), 2);
        assert_eq!(bits_for(5), 4);
        assert_eq!(bits_for(16), 4);
        assert_eq!(bits_for(17), 6);
        assert_eq!(bits_for(64), 6);
        assert_eq!(bits_for(65), 8);
        assert_eq!(bits_for(256), 8);
    }
}
