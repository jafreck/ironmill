//! Weight palettization pass.
//!
//! Compresses FP32/FP16 weight tensors via k-means clustering into lookup
//! tables, converting `const` ops into `constexpr_lut_to_dense` ops.

use half::f16;

use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::ScalarType;
use crate::ir::types::Value;

use super::kmeans::kmeans;
use super::tensor_utils::tensor_as_f32_slice;

/// Compress weights using k-means palettization.
///
/// Each weight tensor is clustered into 2^n_bits centroids. The tensor is
/// then stored as indices + a palette lookup table, reducing size by
/// 32/n_bits × (roughly 8× for 4-bit, ~5× for 6-bit).
pub struct PalettizePass {
    /// Number of bits per weight index (2, 4, 6, or 8).
    n_bits: u8,
    /// Maximum k-means iterations.
    max_iter: usize,
}

impl PalettizePass {
    pub fn new(n_bits: u8) -> Self {
        assert!(
            matches!(n_bits, 2 | 4 | 6 | 8),
            "n_bits must be one of 2, 4, 6, or 8"
        );
        Self {
            n_bits,
            max_iter: 100,
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
                        let f = match dtype {
                            ScalarType::Float32 => tensor_as_f32_slice(data),
                            ScalarType::Float16 => fp16_bytes_to_f32(data),
                            _ => unreachable!(),
                        };
                        (f, shape.clone(), *dtype)
                    }
                    _ => continue,
                };

                if floats.is_empty() {
                    continue;
                }

                let (centroids, assignments) = kmeans(&floats, k, self.max_iter);

                // Build LUT tensor (centroids as f32 bytes).
                let lut_data: Vec<u8> = match original_dtype {
                    ScalarType::Float16 => centroids
                        .iter()
                        .flat_map(|&c| f16::from_f32(c).to_le_bytes())
                        .collect(),
                    _ => centroids.iter().flat_map(|c| c.to_le_bytes()).collect(),
                };

                let lut_value = Value::Tensor {
                    data: lut_data,
                    shape: vec![k],
                    dtype: original_dtype,
                };

                // Pack indices into n-bit representation.
                let packed_indices = pack_indices(&assignments, self.n_bits);

                // CoreML expects the indices as a 1D (rank 1) tensor of
                // packed bytes; the output shape comes from the `shape` attr.
                let indices_value = Value::Tensor {
                    data: packed_indices.clone(),
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
                    data: shape_bytes,
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

/// Pack assignment indices into n-bit packed bytes.
///
/// Uses bit-level packing that works for all supported widths (2, 4, 6, 8),
/// including widths that cause indices to span byte boundaries (e.g. 6-bit).
/// Bits are written MSB-first within each byte.
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

        // Shift value so MSB aligns with bit_in_byte within a 16-bit window.
        let shifted = val << (16 - n_bits as usize - bit_in_byte);
        let [hi, lo] = shifted.to_be_bytes();
        packed[byte_pos] |= hi;
        if byte_pos + 1 < n_bytes {
            packed[byte_pos + 1] |= lo;
        }
    }

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
            data: f32_bytes(&weights),
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
                assert_eq!(data.len(), 32); // 64 × 4 bits / 8
            }
            other => panic!("expected indices tensor, got {other:?}"),
        }

        // Shape attribute preserved as UInt32 tensor.
        match op.attributes.get("shape") {
            Some(Value::Tensor { data, shape, dtype }) => {
                assert_eq!(*dtype, ScalarType::UInt32);
                assert_eq!(*shape, vec![1]);
                let dim = u32::from_le_bytes(data[..4].try_into().unwrap());
                assert_eq!(dim, 64);
            }
            other => panic!("expected shape tensor, got {other:?}"),
        }
    }

    #[test]
    fn palettize_2bit_produces_4_centroids() {
        let weights: Vec<f32> = (0..32).map(|i| (i % 4) as f32).collect();

        let tensor_val = Value::Tensor {
            data: f32_bytes(&weights),
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
                assert_eq!(data.len(), 8);
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
            data: f32_bytes(&weights),
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
            Some(Value::Tensor { data, .. }) => data.len(),
            _ => panic!("missing lut"),
        };
        let idx_size = match op.attributes.get("indices") {
            Some(Value::Tensor { data, .. }) => data.len(),
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
            data: vec![1, 0, 0, 0],
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
            data: fp16_data,
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
}
