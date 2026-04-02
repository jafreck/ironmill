//! Quantization quality evaluation framework.
//!
//! Provides per-layer weight reconstruction error metrics (MSE, max error, SNR)
//! as a proxy for perplexity degradation. Compares original FP32 weights against
//! their dequantized counterparts after a quantization pass.
//!
//! This is NOT perplexity (which requires loaded-model inference on a GPU), but
//! per-layer weight reconstruction error — a metric that correlates well with
//! actual perplexity degradation.

use std::collections::HashMap;

use super::tensor_utils::tensor_as_f32_slice;
use crate::ir::operation::Operation;
use crate::ir::program::Program;
use crate::ir::tensor::ScalarType;
use crate::ir::types::Value;

/// Per-operation quantization quality metrics.
#[derive(Debug, Clone)]
pub struct QuantizationMetrics {
    /// Mean squared error between original and dequantized weights.
    pub mse: f64,
    /// Maximum absolute error across all elements.
    pub max_error: f64,
    /// Signal-to-noise ratio in decibels.
    pub snr_db: f64,
    /// Range of the original weight tensor (max − min).
    pub original_range: f64,
    /// Total number of elements in the weight tensor.
    pub n_elements: usize,
}

impl QuantizationMetrics {
    /// Pretty-print a summary table of metrics for all evaluated ops.
    pub fn summary(metrics: &HashMap<String, QuantizationMetrics>) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "{:<40} {:>10} {:>10} {:>8} {:>10} {:>10}",
            "Op", "MSE", "MaxErr", "SNR(dB)", "Range", "Elements"
        ));
        lines.push("-".repeat(92));

        let mut sorted: Vec<_> = metrics.iter().collect();
        sorted.sort_by_key(|(name, _)| (*name).clone());

        let mut total_mse = 0.0;
        let mut total_elements = 0usize;
        let mut min_snr = f64::INFINITY;

        for (name, m) in &sorted {
            lines.push(format!(
                "{:<40} {:>10.2e} {:>10.2e} {:>8.1} {:>10.4} {:>10}",
                truncate_name(name, 40),
                m.mse,
                m.max_error,
                m.snr_db,
                m.original_range,
                m.n_elements,
            ));
            total_mse += m.mse * m.n_elements as f64;
            total_elements += m.n_elements;
            if m.snr_db < min_snr {
                min_snr = m.snr_db;
            }
        }

        lines.push("-".repeat(92));
        if total_elements > 0 {
            let weighted_mse = total_mse / total_elements as f64;
            lines.push(format!(
                "Weighted MSE: {:.2e}  |  Min SNR: {:.1} dB  |  Ops: {}  |  Total elements: {}",
                weighted_mse,
                min_snr,
                sorted.len(),
                total_elements
            ));
        }

        lines.join("\n")
    }
}

/// Evaluate the quality of a quantization pass by measuring reconstruction error.
///
/// This is NOT perplexity (which requires inference), but per-layer weight
/// reconstruction MSE — a proxy that correlates with perplexity degradation.
pub struct QuantizationEvaluator;

impl QuantizationEvaluator {
    /// Compare original weights to quantized weights, computing per-op metrics.
    ///
    /// Returns a map of `op_name → QuantizationMetrics`.
    ///
    /// The evaluator:
    /// 1. Walks the original program's const ops and collects FP32 weight data
    ///    keyed by output name.
    /// 2. Walks the quantized program's ops and finds quantized const-expression
    ///    ops (`constexpr_affine_dequantize`, `constexpr_lut_to_dense`,
    ///    `constexpr_dual_scale_dequantize`).
    /// 3. Matches by output name, dequantizes back to FP32, and computes MSE,
    ///    max error, and SNR between original and reconstructed weights.
    pub fn evaluate(
        original: &Program,
        quantized: &Program,
    ) -> HashMap<String, QuantizationMetrics> {
        let mut results = HashMap::new();

        // Build a map of output_name → original FP32 data for const ops.
        let mut original_weights: HashMap<String, Vec<f32>> = HashMap::new();
        for function in original.functions.values() {
            for op in &function.body.operations {
                if op.op_type != "const" {
                    continue;
                }
                let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
                if let Some(Value::Tensor {
                    data,
                    dtype: ScalarType::Float32,
                    ..
                }) = val
                {
                    if let Some(output_name) = op.outputs.first() {
                        let floats = tensor_as_f32_slice(data);
                        original_weights.insert(output_name.clone(), floats);
                    }
                }
            }
        }

        // Walk quantized program and find dequantize ops.
        for function in quantized.functions.values() {
            for op in &function.body.operations {
                if !is_quantized_op(&op.op_type) {
                    continue;
                }

                let output_name = match op.outputs.first() {
                    Some(name) => name,
                    None => continue,
                };

                let original_floats = match original_weights.get(output_name) {
                    Some(f) => f,
                    None => continue,
                };

                let dequantized = match dequantize_op(op) {
                    Some(d) => d,
                    None => continue,
                };

                if original_floats.len() != dequantized.len() {
                    continue;
                }

                let metrics = compute_metrics(original_floats, &dequantized);
                results.insert(op.name.clone(), metrics);
            }
        }

        results
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn is_quantized_op(op_type: &str) -> bool {
    matches!(
        op_type,
        "constexpr_affine_dequantize"
            | "constexpr_lut_to_dense"
            | "constexpr_dual_scale_dequantize"
    )
}

fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("…{}", &name[name.len() - max_len + 1..])
    }
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

/// Dequantize a quantized const-expression op back to FP32.
fn dequantize_op(op: &Operation) -> Option<Vec<f32>> {
    match op.op_type.as_str() {
        "constexpr_affine_dequantize" => dequantize_affine_op(op),
        "constexpr_lut_to_dense" => dequantize_lut_op(op),
        "constexpr_dual_scale_dequantize" => dequantize_dual_scale_op(op),
        _ => None,
    }
}

/// Dequantize a `constexpr_affine_dequantize` op.
///
/// Handles per-tensor (scale/zp are scalar `Float`), per-channel (scale/zp
/// are 1-D `Tensor`), and per-group (scale/zp are `Tensor` with `group_size`
/// attribute).
fn dequantize_affine_op(op: &Operation) -> Option<Vec<f32>> {
    let (quantized_data, shape) = match op.attributes.get("quantized_data") {
        Some(Value::Tensor {
            data,
            shape,
            dtype: ScalarType::UInt8,
        }) => (data.as_slice(), shape.as_slice()),
        _ => return None,
    };

    let group_size = match op.attributes.get("group_size") {
        Some(Value::Int(g)) => Some(*g as usize),
        _ => None,
    };

    let scales = extract_f32_params(op.attributes.get("scale"))?;
    let zero_points = extract_f32_params(op.attributes.get("zero_point"))?;

    let n = quantized_data.len();
    let mut result = vec![0.0f32; n];

    if scales.len() == 1 && zero_points.len() == 1 {
        // Per-tensor: single scale and zero_point.
        let scale = scales[0];
        let zp = zero_points[0];
        for i in 0..n {
            result[i] = (quantized_data[i] as f32 - zp) * scale;
        }
    } else if let Some(gs) = group_size {
        // Per-group: scale/zp per group of `gs` elements along the last axis.
        let ndim = shape.len();
        let last_dim = if ndim > 0 { shape[ndim - 1] } else { 1 };
        let outer_count: usize = if ndim > 1 {
            shape[..ndim - 1].iter().product()
        } else {
            1
        };
        let n_groups = last_dim.div_ceil(gs);

        for row in 0..outer_count {
            for g in 0..n_groups {
                let g_start = row * last_dim + g * gs;
                let g_end = (g_start + gs).min(row * last_dim + last_dim);
                let param_idx = row * n_groups + g;
                let scale = scales[param_idx];
                let zp = zero_points[param_idx];
                for i in g_start..g_end {
                    result[i] = (quantized_data[i] as f32 - zp) * scale;
                }
            }
        }
    } else {
        // Per-channel: one scale/zp per output channel (axis 0).
        let num_channels = if !shape.is_empty() { shape[0] } else { 1 };
        let channel_size: usize = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            n / num_channels.max(1)
        };

        for ch in 0..num_channels {
            let scale = scales.get(ch).copied().unwrap_or(1.0);
            let zp = zero_points.get(ch).copied().unwrap_or(0.0);
            for i in 0..channel_size {
                let idx = ch * channel_size + i;
                if idx < n {
                    result[idx] = (quantized_data[idx] as f32 - zp) * scale;
                }
            }
        }
    }

    Some(result)
}

/// Extract f32 parameters from a `Value` that may be a scalar `Float` or a
/// `Float32` `Tensor`.
fn extract_f32_params(value: Option<&Value>) -> Option<Vec<f32>> {
    match value {
        Some(Value::Float(f)) => Some(vec![*f as f32]),
        Some(Value::Tensor {
            data,
            dtype: ScalarType::Float32,
            ..
        }) => Some(tensor_as_f32_slice(data)),
        _ => None,
    }
}

/// Dequantize a `constexpr_lut_to_dense` op (QuIP#-style codebook quantization).
///
/// Attributes:
/// - `lut`: f32 tensor of codebook entries
/// - `indices`: UInt8 tensor of packed indices
/// - `shape`: output tensor shape
/// - `quip_sharp_scales` (optional): f32 per-group scales
fn dequantize_lut_op(op: &Operation) -> Option<Vec<f32>> {
    let lut = extract_f32_params(op.attributes.get("lut"))?;

    let indices_bytes = match op.attributes.get("indices") {
        Some(Value::Tensor {
            data,
            dtype: ScalarType::UInt8,
            ..
        }) => data.as_slice(),
        _ => return None,
    };

    let shape = match op.attributes.get("shape") {
        Some(Value::Tensor {
            data,
            shape: s,
            dtype: ScalarType::Int32,
        }) => {
            let ints = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
                .collect::<Vec<_>>();
            if ints.is_empty() { s.clone() } else { ints }
        }
        Some(Value::List(items)) => items
            .iter()
            .filter_map(|v| match v {
                Value::Int(i) => Some(*i as usize),
                _ => None,
            })
            .collect(),
        _ => return None,
    };

    let n_elements: usize = shape.iter().product();
    if n_elements == 0 {
        return Some(vec![]);
    }

    // Determine bits per index from LUT size: lut has 2^n_bits entries.
    let n_bits = match lut.len() {
        0 => return None,
        n => {
            let mut bits = 0u32;
            let mut v = n;
            while v > 1 {
                v >>= 1;
                bits += 1;
            }
            if 1 << bits != n {
                return None; // LUT size not a power of 2
            }
            bits as usize
        }
    };

    // Unpack indices from bytes. Each byte may contain multiple indices
    // packed from LSB to MSB.
    let mask = (1u8 << n_bits) - 1;
    let indices_per_byte = 8 / n_bits;
    let mut unpacked = Vec::with_capacity(n_elements);
    for &byte in indices_bytes {
        for sub in 0..indices_per_byte {
            if unpacked.len() >= n_elements {
                break;
            }
            let idx = (byte >> (sub * n_bits)) & mask;
            unpacked.push(idx as usize);
        }
    }

    if unpacked.len() < n_elements {
        return None;
    }

    // Look up each index in the codebook.
    let mut result: Vec<f32> = unpacked[..n_elements]
        .iter()
        .map(|&idx| if idx < lut.len() { lut[idx] } else { 0.0 })
        .collect();

    // Apply optional QuIP# per-group scales.
    if let Some(scales) = extract_f32_params(op.attributes.get("quip_sharp_scales")) {
        if !scales.is_empty() {
            let group_size = n_elements / scales.len();
            if group_size > 0 {
                for (i, val) in result.iter_mut().enumerate() {
                    let g = i / group_size;
                    if g < scales.len() {
                        *val *= scales[g];
                    }
                }
            }
        }
    }

    Some(result)
}

/// Dequantize a `constexpr_dual_scale_dequantize` op (D2Quant-style).
///
/// Uses normal scale/zero for regular elements and outlier scale/zero for
/// elements flagged in the outlier mask.
///
/// Attributes:
/// - `quantized_data`: UInt8 packed data
/// - `normal_scale`, `normal_zero`: f32 per-group params for normal elements
/// - `outlier_scale`, `outlier_zero`: f32 per-group params for outliers
/// - `outlier_mask`: UInt8 bitmask
/// - `group_size`: int
/// - `bit_width`: int (default 4)
fn dequantize_dual_scale_op(op: &Operation) -> Option<Vec<f32>> {
    let quantized_bytes = match op.attributes.get("quantized_data") {
        Some(Value::Tensor {
            data,
            dtype: ScalarType::UInt8,
            ..
        }) => data.as_slice(),
        _ => return None,
    };

    let normal_scales = extract_f32_params(op.attributes.get("normal_scale"))?;
    let normal_zeros = extract_f32_params(op.attributes.get("normal_zero"))?;
    let outlier_scales = extract_f32_params(op.attributes.get("outlier_scale"))?;
    let outlier_zeros = extract_f32_params(op.attributes.get("outlier_zero"))?;

    let outlier_mask_bytes = match op.attributes.get("outlier_mask") {
        Some(Value::Tensor {
            data,
            dtype: ScalarType::UInt8,
            ..
        }) => data.as_slice(),
        _ => return None,
    };

    let group_size = match op.attributes.get("group_size") {
        Some(Value::Int(g)) => *g as usize,
        _ => return None,
    };

    let bit_width = match op.attributes.get("bit_width") {
        Some(Value::Int(b)) => *b as usize,
        _ => 4,
    };

    if bit_width == 0 || group_size == 0 {
        return None;
    }

    // Unpack quantized values from packed bytes.
    let mask = (1u32 << bit_width) - 1;
    let vals_per_byte = 8 / bit_width;
    let mut quant_vals = Vec::new();
    for &byte in quantized_bytes {
        for sub in 0..vals_per_byte {
            let q = ((byte as u32) >> (sub * bit_width)) & mask;
            quant_vals.push(q as f32);
        }
    }

    // Unpack outlier mask: 1 bit per element, packed LSB-first.
    let mut outlier_flags = Vec::new();
    for &byte in outlier_mask_bytes {
        for bit in 0..8 {
            outlier_flags.push((byte >> bit) & 1 != 0);
        }
    }

    let n_elements = quant_vals.len().min(outlier_flags.len());
    let mut result = vec![0.0f32; n_elements];

    for i in 0..n_elements {
        let g = i / group_size;
        let q = quant_vals[i];

        if i < outlier_flags.len() && outlier_flags[i] {
            let scale = outlier_scales.get(g).copied().unwrap_or(1.0);
            let zero = outlier_zeros.get(g).copied().unwrap_or(0.0);
            result[i] = (q - zero) * scale;
        } else {
            let scale = normal_scales.get(g).copied().unwrap_or(1.0);
            let zero = normal_zeros.get(g).copied().unwrap_or(0.0);
            result[i] = (q - zero) * scale;
        }
    }

    Some(result)
}

// ---------------------------------------------------------------------------
// Metrics computation
// ---------------------------------------------------------------------------

fn compute_metrics(original: &[f32], dequantized: &[f32]) -> QuantizationMetrics {
    let n = original.len();
    assert_eq!(n, dequantized.len());

    let mut sum_sq_error = 0.0_f64;
    let mut max_error = 0.0_f64;
    let mut sum_sq_signal = 0.0_f64;
    let mut orig_min = f64::INFINITY;
    let mut orig_max = f64::NEG_INFINITY;

    for i in 0..n {
        let o = original[i] as f64;
        let d = dequantized[i] as f64;
        let err = (o - d).abs();
        sum_sq_error += err * err;
        if err > max_error {
            max_error = err;
        }
        sum_sq_signal += o * o;
        if o < orig_min {
            orig_min = o;
        }
        if o > orig_max {
            orig_max = o;
        }
    }

    let mse = if n > 0 { sum_sq_error / n as f64 } else { 0.0 };
    let original_range = if n > 0 { orig_max - orig_min } else { 0.0 };

    let snr_db = if sum_sq_error > 0.0 {
        10.0 * (sum_sq_signal / sum_sq_error).log10()
    } else {
        f64::INFINITY
    };

    QuantizationMetrics {
        mse,
        max_error,
        snr_db,
        original_range,
        n_elements: n,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::pass::Pass;
    use crate::ir::passes::affine_quantize::AffineQuantizePass;
    use crate::ir::program::{Block, Function};
    use crate::ir::tensor::TensorType;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // -- Helpers ------------------------------------------------------------

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn const_tensor_op(name: &str, output: &str, data: &[f32], shape: Vec<usize>) -> Operation {
        Operation::new("const", name)
            .with_input(
                "val",
                Value::Tensor {
                    data: f32_bytes(data),
                    shape,
                    dtype: ScalarType::Float32,
                },
            )
            .with_output(output)
    }

    fn random_weight(rng: &mut StdRng, shape: &[usize]) -> Vec<f32> {
        let n: usize = shape.iter().product();
        (0..n).map(|_| rng.gen_range(-0.1f32..0.1)).collect()
    }

    /// Build a realistic multi-layer test program with 2D weight matrices
    /// suitable for per-group quantization with group_size=32.
    fn create_eval_test_program() -> Program {
        let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
        let mut block = Block::new();

        let hidden = 64;
        let intermediate = 128;
        let vocab = 256;

        // Embedding
        let data = random_weight(&mut rng, &[vocab, hidden]);
        block.add_op(const_tensor_op(
            "embed_const",
            "embed_weight",
            &data,
            vec![vocab, hidden],
        ));
        block.add_op(
            Operation::new("gather", "embed_gather")
                .with_input("x", Value::Reference("embed_weight".into()))
                .with_input("indices", Value::Reference("input_ids".into()))
                .with_output("embed_out"),
        );

        // Layer 0: q_proj, k_proj, v_proj, o_proj
        for name in &["l0_q_proj", "l0_k_proj", "l0_v_proj", "l0_o_proj"] {
            let w = random_weight(&mut rng, &[hidden, hidden]);
            let const_name = format!("{name}_const");
            let weight_name = format!("{name}_weight");
            let out_name = format!("{name}_out");
            block.add_op(const_tensor_op(
                &const_name,
                &weight_name,
                &w,
                vec![hidden, hidden],
            ));
            block.add_op(
                Operation::new("linear", &format!("{name}_op"))
                    .with_input("x", Value::Reference("embed_out".into()))
                    .with_input("weight", Value::Reference(weight_name))
                    .with_output(&out_name),
            );
        }

        // Layer 0: gate_proj, up_proj (hidden → intermediate)
        for name in &["l0_gate_proj", "l0_up_proj"] {
            let w = random_weight(&mut rng, &[intermediate, hidden]);
            let const_name = format!("{name}_const");
            let weight_name = format!("{name}_weight");
            let out_name = format!("{name}_out");
            block.add_op(const_tensor_op(
                &const_name,
                &weight_name,
                &w,
                vec![intermediate, hidden],
            ));
            block.add_op(
                Operation::new("linear", &format!("{name}_op"))
                    .with_input("x", Value::Reference("l0_o_proj_out".into()))
                    .with_input("weight", Value::Reference(weight_name))
                    .with_output(&out_name),
            );
        }

        // Layer 0: down_proj (intermediate → hidden)
        {
            let w = random_weight(&mut rng, &[hidden, intermediate]);
            block.add_op(const_tensor_op(
                "l0_down_proj_const",
                "l0_down_proj_weight",
                &w,
                vec![hidden, intermediate],
            ));
            block.add_op(
                Operation::new("linear", "l0_down_proj_op")
                    .with_input("x", Value::Reference("l0_gate_proj_out".into()))
                    .with_input("weight", Value::Reference("l0_down_proj_weight".into()))
                    .with_output("l0_down_proj_out"),
            );
        }

        // LM head
        let w = random_weight(&mut rng, &[vocab, hidden]);
        block.add_op(const_tensor_op(
            "lm_head_const",
            "lm_head_weight",
            &w,
            vec![vocab, hidden],
        ));
        block.add_op(
            Operation::new("linear", "lm_head_op")
                .with_input("x", Value::Reference("l0_down_proj_out".into()))
                .with_input("weight", Value::Reference("lm_head_weight".into()))
                .with_output("lm_head_out"),
        );

        block.outputs.push("lm_head_out".into());

        let input_ty = TensorType::new(ScalarType::Int32, vec![1, 128]);
        let func = Function::new("main").with_input("input_ids", input_ty);

        let mut program = Program::new("1.0.0");
        program.add_function(func);
        program.functions.get_mut("main").unwrap().body = block;
        program
    }

    // -- Unit tests ---------------------------------------------------------

    #[test]
    fn eval_quantize_per_tensor_int4() {
        let values: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let mut program = {
            let tensor_val = Value::Tensor {
                data: f32_bytes(&values),
                shape: vec![10, 10],
                dtype: ScalarType::Float32,
            };
            let mut p = Program::new("1.0.0");
            let mut func = Function::new("main");
            func.body.add_op(
                Operation::new("const", "w_const")
                    .with_input("val", tensor_val)
                    .with_output("w_out"),
            );
            func.body.outputs.push("w_out".into());
            p.add_function(func);
            p
        };

        let original = program.clone();
        AffineQuantizePass::int4_per_tensor()
            .run(&mut program)
            .unwrap();

        let metrics = QuantizationEvaluator::evaluate(&original, &program);
        assert_eq!(metrics.len(), 1);

        let m = metrics.values().next().unwrap();
        assert!(m.mse > 0.0, "MSE should be positive for lossy quantization");
        assert!(m.max_error > 0.0);
        assert!(
            m.snr_db > 15.0,
            "INT4 per-tensor SNR should be > 15 dB, got {}",
            m.snr_db
        );
        assert_eq!(m.n_elements, 100);
        assert!((m.original_range - 9.9).abs() < 0.01);
    }

    #[test]
    fn eval_quantize_per_group_int4() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let mut program = {
            let tensor_val = Value::Tensor {
                data: f32_bytes(&values),
                shape: vec![2, 32],
                dtype: ScalarType::Float32,
            };
            let mut p = Program::new("1.0.0");
            let mut func = Function::new("main");
            func.body.add_op(
                Operation::new("const", "w_const")
                    .with_input("val", tensor_val)
                    .with_output("w_out"),
            );
            func.body.outputs.push("w_out".into());
            p.add_function(func);
            p
        };

        let original = program.clone();
        AffineQuantizePass::int4_per_group(32)
            .run(&mut program)
            .unwrap();

        let metrics = QuantizationEvaluator::evaluate(&original, &program);
        assert_eq!(metrics.len(), 1);

        let m = metrics.values().next().unwrap();
        assert!(m.mse > 0.0);
        assert!(
            m.snr_db > 20.0,
            "INT4 per-group SNR should be > 20 dB, got {}",
            m.snr_db
        );
        assert_eq!(m.n_elements, 64);
    }

    #[test]
    fn eval_quantize_no_match_returns_empty() {
        let original = Program::new("1.0.0");
        let quantized = Program::new("1.0.0");
        let metrics = QuantizationEvaluator::evaluate(&original, &quantized);
        assert!(metrics.is_empty());
    }

    #[test]
    fn eval_quantize_skips_non_quantized_ops() {
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let tensor_val = Value::Tensor {
            data: f32_bytes(&values),
            shape: vec![4],
            dtype: ScalarType::Float32,
        };

        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(
            Operation::new("const", "w_const")
                .with_input("val", tensor_val)
                .with_output("w_out"),
        );
        func.body.outputs.push("w_out".into());
        program.add_function(func);

        // Both original and "quantized" have the same const ops (no quantization).
        let metrics = QuantizationEvaluator::evaluate(&program, &program);
        assert!(
            metrics.is_empty(),
            "const ops should not match as quantized"
        );
    }

    #[test]
    fn eval_quantize_metrics_summary_formatting() {
        let mut metrics = HashMap::new();
        metrics.insert(
            "layer_0_weight".to_string(),
            QuantizationMetrics {
                mse: 1.5e-5,
                max_error: 0.002,
                snr_db: 35.0,
                original_range: 0.2,
                n_elements: 4096,
            },
        );
        metrics.insert(
            "layer_1_weight".to_string(),
            QuantizationMetrics {
                mse: 2.0e-5,
                max_error: 0.003,
                snr_db: 30.0,
                original_range: 0.25,
                n_elements: 8192,
            },
        );

        let summary = QuantizationMetrics::summary(&metrics);
        assert!(summary.contains("layer_0_weight"));
        assert!(summary.contains("layer_1_weight"));
        assert!(summary.contains("Weighted MSE"));
        assert!(summary.contains("Min SNR"));
    }

    #[test]
    fn eval_quantize_perfect_reconstruction() {
        // All-zero tensor: quantization should be lossless.
        let values = vec![0.0_f32; 64];
        let mut program = {
            let tensor_val = Value::Tensor {
                data: f32_bytes(&values),
                shape: vec![8, 8],
                dtype: ScalarType::Float32,
            };
            let mut p = Program::new("1.0.0");
            let mut func = Function::new("main");
            func.body.add_op(
                Operation::new("const", "zero_const")
                    .with_input("val", tensor_val)
                    .with_output("zero_out"),
            );
            func.body.outputs.push("zero_out".into());
            p.add_function(func);
            p
        };

        let original = program.clone();
        AffineQuantizePass::int4_per_tensor()
            .run(&mut program)
            .unwrap();

        let metrics = QuantizationEvaluator::evaluate(&original, &program);
        assert_eq!(metrics.len(), 1);

        let m = metrics.values().next().unwrap();
        assert!(
            m.mse < 1e-30,
            "All-zero tensor should have near-zero MSE, got {}",
            m.mse
        );
    }

    // -- QuIP# (LUT) dequantization tests -----------------------------------

    #[test]
    fn eval_quantize_lut_to_dense_2bit() {
        // Build a program with an original FP32 const and a quantized LUT op.
        let lut = [0.0f32, 0.5, -0.5, 1.0]; // 4 entries = 2-bit
        // 8 elements packed into 2 bytes (4 indices per byte, 2 bits each)
        // Indices: [0, 1, 2, 3, 0, 1, 2, 3]
        // Byte 0: 0b_11_10_01_00 = 0xE4, Byte 1: 0b_11_10_01_00 = 0xE4
        let packed_indices: Vec<u8> = vec![0xE4, 0xE4];
        let expected: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, 0.0, 0.5, -0.5, 1.0];

        let original = {
            let mut p = Program::new("1.0.0");
            let mut func = Function::new("main");
            func.body.add_op(
                Operation::new("const", "w_const")
                    .with_input(
                        "val",
                        Value::Tensor {
                            data: f32_bytes(&expected),
                            shape: vec![8],
                            dtype: ScalarType::Float32,
                        },
                    )
                    .with_output("w_out"),
            );
            func.body.outputs.push("w_out".into());
            p.add_function(func);
            p
        };

        let quantized = {
            let mut p = Program::new("1.0.0");
            let mut func = Function::new("main");
            let mut op = Operation::new("constexpr_lut_to_dense", "w_lut");
            op.attributes.insert(
                "lut".into(),
                Value::Tensor {
                    data: f32_bytes(&lut),
                    shape: vec![4],
                    dtype: ScalarType::Float32,
                },
            );
            op.attributes.insert(
                "indices".into(),
                Value::Tensor {
                    data: packed_indices,
                    shape: vec![2],
                    dtype: ScalarType::UInt8,
                },
            );
            op.attributes
                .insert("shape".into(), Value::List(vec![Value::Int(8)]));
            op.outputs.push("w_out".into());
            func.body.add_op(op);
            func.body.outputs.push("w_out".into());
            p.add_function(func);
            p
        };

        let metrics = QuantizationEvaluator::evaluate(&original, &quantized);
        assert_eq!(metrics.len(), 1, "Should evaluate the LUT op");

        let m = metrics.values().next().unwrap();
        assert!(
            m.mse < 1e-10,
            "Perfect LUT reconstruction should have near-zero MSE, got {}",
            m.mse
        );
        assert_eq!(m.n_elements, 8);
    }

    #[test]
    fn eval_quantize_lut_with_quip_sharp_scales() {
        let lut = [0.0f32, 1.0, -1.0, 2.0]; // 2-bit
        // Indices: [1, 1, 3, 3] → pre-scale values: [1.0, 1.0, 2.0, 2.0]
        // Byte: 0b_11_11_01_01 = 0xF5
        let packed_indices: Vec<u8> = vec![0xF5];
        // 2 groups of 2, scales: [0.5, 2.0]
        // Expected: [1.0*0.5, 1.0*0.5, 2.0*2.0, 2.0*2.0] = [0.5, 0.5, 4.0, 4.0]
        let expected = [0.5f32, 0.5, 4.0, 4.0];

        let original = {
            let mut p = Program::new("1.0.0");
            let mut func = Function::new("main");
            func.body.add_op(
                Operation::new("const", "w_const")
                    .with_input(
                        "val",
                        Value::Tensor {
                            data: f32_bytes(&expected),
                            shape: vec![4],
                            dtype: ScalarType::Float32,
                        },
                    )
                    .with_output("w_out"),
            );
            func.body.outputs.push("w_out".into());
            p.add_function(func);
            p
        };

        let scales = [0.5f32, 2.0];
        let quantized = {
            let mut p = Program::new("1.0.0");
            let mut func = Function::new("main");
            let mut op = Operation::new("constexpr_lut_to_dense", "w_lut");
            op.attributes.insert(
                "lut".into(),
                Value::Tensor {
                    data: f32_bytes(&lut),
                    shape: vec![4],
                    dtype: ScalarType::Float32,
                },
            );
            op.attributes.insert(
                "indices".into(),
                Value::Tensor {
                    data: packed_indices,
                    shape: vec![1],
                    dtype: ScalarType::UInt8,
                },
            );
            op.attributes
                .insert("shape".into(), Value::List(vec![Value::Int(4)]));
            op.attributes.insert(
                "quip_sharp_scales".into(),
                Value::Tensor {
                    data: f32_bytes(&scales),
                    shape: vec![2],
                    dtype: ScalarType::Float32,
                },
            );
            op.outputs.push("w_out".into());
            func.body.add_op(op);
            func.body.outputs.push("w_out".into());
            p.add_function(func);
            p
        };

        let metrics = QuantizationEvaluator::evaluate(&original, &quantized);
        assert_eq!(metrics.len(), 1);
        let m = metrics.values().next().unwrap();
        assert!(
            m.mse < 1e-10,
            "QuIP# with scales should reconstruct exactly, got MSE {}",
            m.mse
        );
    }

    // -- D2Quant (dual-scale) dequantization tests ----------------------------

    #[test]
    fn eval_quantize_dual_scale_dequantize() {
        // 4 elements, group_size=4, 4-bit, all in one group.
        // Elements: q=[2, 5, 3, 7], outlier_mask bits: [0, 1, 0, 1]
        // Normal: scale=0.1, zero=1.0  →  (q-1)*0.1
        // Outlier: scale=0.5, zero=2.0  →  (q-2)*0.5
        // Expected:
        //   elem0: normal: (2-1)*0.1 = 0.1
        //   elem1: outlier: (5-2)*0.5 = 1.5
        //   elem2: normal: (3-1)*0.1 = 0.2
        //   elem3: outlier: (7-2)*0.5 = 2.5
        let expected = [0.1f32, 1.5, 0.2, 2.5];

        // Pack: 4 values in 4-bit = 2 bytes
        // Byte 0 low nibble: q[0]=2, high nibble: q[1]=5 → 0x52
        // Byte 1 low nibble: q[2]=3, high nibble: q[3]=7 → 0x73
        let quantized_data: Vec<u8> = vec![0x52, 0x73];

        // Outlier mask: bits [0,1,0,1] → byte 0b_0000_1010 = 0x0A
        let outlier_mask: Vec<u8> = vec![0x0A];

        let original = {
            let mut p = Program::new("1.0.0");
            let mut func = Function::new("main");
            func.body.add_op(
                Operation::new("const", "w_const")
                    .with_input(
                        "val",
                        Value::Tensor {
                            data: f32_bytes(&expected),
                            shape: vec![4],
                            dtype: ScalarType::Float32,
                        },
                    )
                    .with_output("w_out"),
            );
            func.body.outputs.push("w_out".into());
            p.add_function(func);
            p
        };

        let quantized = {
            let mut p = Program::new("1.0.0");
            let mut func = Function::new("main");
            let mut op = Operation::new("constexpr_dual_scale_dequantize", "w_d2q");
            op.attributes.insert(
                "quantized_data".into(),
                Value::Tensor {
                    data: quantized_data,
                    shape: vec![2],
                    dtype: ScalarType::UInt8,
                },
            );
            op.attributes.insert(
                "normal_scale".into(),
                Value::Tensor {
                    data: f32_bytes(&[0.1]),
                    shape: vec![1],
                    dtype: ScalarType::Float32,
                },
            );
            op.attributes.insert(
                "normal_zero".into(),
                Value::Tensor {
                    data: f32_bytes(&[1.0]),
                    shape: vec![1],
                    dtype: ScalarType::Float32,
                },
            );
            op.attributes.insert(
                "outlier_scale".into(),
                Value::Tensor {
                    data: f32_bytes(&[0.5]),
                    shape: vec![1],
                    dtype: ScalarType::Float32,
                },
            );
            op.attributes.insert(
                "outlier_zero".into(),
                Value::Tensor {
                    data: f32_bytes(&[2.0]),
                    shape: vec![1],
                    dtype: ScalarType::Float32,
                },
            );
            op.attributes.insert(
                "outlier_mask".into(),
                Value::Tensor {
                    data: outlier_mask,
                    shape: vec![1],
                    dtype: ScalarType::UInt8,
                },
            );
            op.attributes.insert("group_size".into(), Value::Int(4));
            op.attributes.insert("bit_width".into(), Value::Int(4));
            op.outputs.push("w_out".into());
            func.body.add_op(op);
            func.body.outputs.push("w_out".into());
            p.add_function(func);
            p
        };

        let metrics = QuantizationEvaluator::evaluate(&original, &quantized);
        assert_eq!(metrics.len(), 1, "Should evaluate the D2Quant op");

        let m = metrics.values().next().unwrap();
        assert!(
            m.mse < 1e-6,
            "D2Quant should reconstruct accurately, got MSE {}",
            m.mse
        );
        assert_eq!(m.n_elements, 4);
    }

    #[test]
    fn eval_quantize_dual_scale_no_outliers() {
        // All normal elements (outlier mask = 0x00).
        // 4 elements, group_size=4, 4-bit.
        // q=[0, 1, 2, 3], normal scale=2.0, zero=0.0
        // Expected: [0.0, 2.0, 4.0, 6.0]
        let expected = [0.0f32, 2.0, 4.0, 6.0];
        let quantized_data: Vec<u8> = vec![0x10, 0x32]; // [0,1], [2,3]
        let outlier_mask: Vec<u8> = vec![0x00]; // no outliers

        let original = {
            let mut p = Program::new("1.0.0");
            let mut func = Function::new("main");
            func.body.add_op(
                Operation::new("const", "w_const")
                    .with_input(
                        "val",
                        Value::Tensor {
                            data: f32_bytes(&expected),
                            shape: vec![4],
                            dtype: ScalarType::Float32,
                        },
                    )
                    .with_output("w_out"),
            );
            func.body.outputs.push("w_out".into());
            p.add_function(func);
            p
        };

        let quantized = {
            let mut p = Program::new("1.0.0");
            let mut func = Function::new("main");
            let mut op = Operation::new("constexpr_dual_scale_dequantize", "w_d2q");
            op.attributes.insert(
                "quantized_data".into(),
                Value::Tensor {
                    data: quantized_data,
                    shape: vec![2],
                    dtype: ScalarType::UInt8,
                },
            );
            op.attributes.insert(
                "normal_scale".into(),
                Value::Tensor {
                    data: f32_bytes(&[2.0]),
                    shape: vec![1],
                    dtype: ScalarType::Float32,
                },
            );
            op.attributes.insert(
                "normal_zero".into(),
                Value::Tensor {
                    data: f32_bytes(&[0.0]),
                    shape: vec![1],
                    dtype: ScalarType::Float32,
                },
            );
            op.attributes.insert(
                "outlier_scale".into(),
                Value::Tensor {
                    data: f32_bytes(&[1.0]),
                    shape: vec![1],
                    dtype: ScalarType::Float32,
                },
            );
            op.attributes.insert(
                "outlier_zero".into(),
                Value::Tensor {
                    data: f32_bytes(&[0.0]),
                    shape: vec![1],
                    dtype: ScalarType::Float32,
                },
            );
            op.attributes.insert(
                "outlier_mask".into(),
                Value::Tensor {
                    data: outlier_mask,
                    shape: vec![1],
                    dtype: ScalarType::UInt8,
                },
            );
            op.attributes.insert("group_size".into(), Value::Int(4));
            op.attributes.insert("bit_width".into(), Value::Int(4));
            op.outputs.push("w_out".into());
            func.body.add_op(op);
            func.body.outputs.push("w_out".into());
            p.add_function(func);
            p
        };

        let metrics = QuantizationEvaluator::evaluate(&original, &quantized);
        assert_eq!(metrics.len(), 1);
        let m = metrics.values().next().unwrap();
        assert!(
            m.mse < 1e-10,
            "All-normal D2Quant should reconstruct exactly, got MSE {}",
            m.mse
        );
    }

    // -- End-to-end quality evaluation test ---------------------------------

    #[test]
    fn eval_quantize_int4_group32_quality_gate() {
        let original = create_eval_test_program();
        let mut quantized = original.clone();

        AffineQuantizePass::int4_per_group(32)
            .run(&mut quantized)
            .unwrap();

        let metrics = QuantizationEvaluator::evaluate(&original, &quantized);
        assert!(
            !metrics.is_empty(),
            "Should have evaluated at least one quantized op"
        );

        // Print the summary table (visible with `cargo test -- --nocapture`).
        let summary = QuantizationMetrics::summary(&metrics);
        println!("\n=== INT4 group_size=32 Quantization Quality ===\n{summary}\n");

        // Assert that every op exceeds the quality gate.
        for (name, m) in &metrics {
            assert!(
                m.snr_db > 20.0,
                "Op '{name}' SNR {:.1} dB is below 20 dB quality gate",
                m.snr_db
            );
            assert!(
                m.mse < 1e-2,
                "Op '{name}' MSE {:.2e} is unreasonably high",
                m.mse
            );
        }

        // Verify we evaluated the expected number of weight ops.
        // 8 linear weight consts: 4 attn proj + gate + up + down + lm_head
        // 1 embedding const
        // = 9 total
        assert_eq!(
            metrics.len(),
            9,
            "Expected 9 quantized weight ops, got {}",
            metrics.len()
        );
    }
}
