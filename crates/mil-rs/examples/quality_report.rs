//! Numerical accuracy quality report.
//!
//! Compares weight tensor values between baseline (unoptimized) and optimized
//! MIL IR programs, reporting max error, MSE, cosine similarity, SNR, and
//! the percentage of values that changed for each optimization combo.
//!
//! Usage:
//! ```bash
//! cargo run --release --example quality_report -- tests/fixtures/squeezenet1.1.onnx
//! ```

use std::collections::HashMap;

use half::f16;
use mil_rs::ir::passes::tensor_utils::tensor_as_f32_slice;
use mil_rs::{
    ConversionConfig, PassPipeline, Program, ScalarType, Value, onnx_to_program_with_config,
    read_onnx_with_dir,
};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let input = args
        .get(1)
        .map_or("tests/fixtures/squeezenet1.1.onnx", |s| s.as_str());

    let model_name = std::path::Path::new(input)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    println!("Model: {model_name}\n");

    let (onnx_model, model_dir) = read_onnx_with_dir(input)?;
    let conversion_config = ConversionConfig {
        model_dir: Some(model_dir),
        ..Default::default()
    };

    // Each combo: (label, baseline pipeline builder, optimized pipeline builder).
    // The baseline uses the default always-on passes so that graph structure
    // matches, isolating just the quantization/palettization effect.
    let combos: Vec<(
        &str,
        Box<dyn Fn() -> anyhow::Result<PassPipeline>>,
        Box<dyn Fn() -> anyhow::Result<PassPipeline>>,
    )> = vec![
        (
            "Default pipeline (always-on)",
            Box::new(|| Ok(PassPipeline::new().without_fusion())),
            Box::new(|| Ok(PassPipeline::new())),
        ),
        (
            "FP16 Quantization",
            Box::new(|| Ok(PassPipeline::new())),
            Box::new(|| Ok(PassPipeline::new().with_fp16()?)),
        ),
        (
            "INT8 Quantization",
            Box::new(|| Ok(PassPipeline::new())),
            Box::new(|| Ok(PassPipeline::new().with_int8(None)?)),
        ),
        (
            "4-bit Palettization",
            Box::new(|| Ok(PassPipeline::new())),
            Box::new(|| Ok(PassPipeline::new().with_palettize(4)?)),
        ),
        (
            "6-bit Palettization",
            Box::new(|| Ok(PassPipeline::new())),
            Box::new(|| Ok(PassPipeline::new().with_palettize(6)?)),
        ),
    ];

    for (label, make_base_pipeline, make_opt_pipeline) in &combos {
        // Baseline: default passes only (same graph structure, no quantization).
        let result_base = onnx_to_program_with_config(&onnx_model, &conversion_config)?;
        let mut baseline = result_base.program;
        let base_pipeline = make_base_pipeline()?;
        base_pipeline.run(&mut baseline)?;

        // Optimized: default passes + the specific quantization/palettization.
        let result_opt = onnx_to_program_with_config(&onnx_model, &conversion_config)?;
        let mut optimized = result_opt.program;
        let opt_pipeline = make_opt_pipeline()?;
        let report = opt_pipeline.run(&mut optimized)?;

        let ops_before = report
            .pass_results
            .first()
            .map(|r| r.ops_before)
            .unwrap_or(0);
        let ops_after = report.pass_results.last().map(|r| r.ops_after).unwrap_or(0);

        // Collect const tensor values from both programs.
        let base_tensors = collect_const_tensors(&baseline);
        let opt_tensors = collect_const_tensors(&optimized);

        // Match by op name and compute metrics.
        let mut matched = 0u64;
        let mut total_max_abs_err = 0.0_f64;
        let mut total_mse = 0.0_f64;
        let mut total_cosine_num = 0.0_f64;
        let mut total_cosine_denom_a = 0.0_f64;
        let mut total_cosine_denom_b = 0.0_f64;
        let mut total_signal_power = 0.0_f64;
        let mut total_noise_power = 0.0_f64;
        let mut total_values = 0u64;
        let mut total_changed = 0u64;

        for (name, base_vals) in &base_tensors {
            let opt_vals = match opt_tensors.get(name) {
                Some(v) => v,
                None => continue,
            };

            if base_vals.is_empty() || opt_vals.is_empty() {
                continue;
            }

            // Use the shorter length in case shapes differ slightly.
            let len = base_vals.len().min(opt_vals.len());
            matched += 1;

            let mut max_err = 0.0_f64;
            let mut sum_sq_err = 0.0_f64;
            let mut dot = 0.0_f64;
            let mut norm_a = 0.0_f64;
            let mut norm_b = 0.0_f64;
            let mut sig_pow = 0.0_f64;
            let mut noise_pow = 0.0_f64;
            let mut changed = 0u64;

            for i in 0..len {
                let a = base_vals[i] as f64;
                let b = opt_vals[i] as f64;
                let diff = (a - b).abs();
                if diff > max_err {
                    max_err = diff;
                }
                sum_sq_err += diff * diff;
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
                sig_pow += a * a;
                noise_pow += diff * diff;
                if (a - b).abs() > f64::EPSILON {
                    changed += 1;
                }
            }

            if max_err > total_max_abs_err {
                total_max_abs_err = max_err;
            }
            total_mse += sum_sq_err;
            total_cosine_num += dot;
            total_cosine_denom_a += norm_a;
            total_cosine_denom_b += norm_b;
            total_signal_power += sig_pow;
            total_noise_power += noise_pow;
            total_values += len as u64;
            total_changed += changed;
        }

        // Compute aggregate metrics.
        let mse = if total_values > 0 {
            total_mse / total_values as f64
        } else {
            0.0
        };

        let cosine = if total_cosine_denom_a > 0.0 && total_cosine_denom_b > 0.0 {
            total_cosine_num / (total_cosine_denom_a.sqrt() * total_cosine_denom_b.sqrt())
        } else {
            1.0
        };

        let snr = if total_noise_power > 0.0 {
            10.0 * (total_signal_power / total_noise_power).log10()
        } else {
            f64::INFINITY
        };

        let pct_changed = if total_values > 0 {
            (total_changed as f64 / total_values as f64) * 100.0
        } else {
            0.0
        };

        // Print report table.
        println!("Optimization: {label}");
        println!("┌─────────────────────┬───────────────┐");
        println!("│ Metric              │ Value         │");
        println!("├─────────────────────┼───────────────┤");
        let matched_label = format!("{matched} matched");
        println!("│ Const tensors       │ {matched_label:<13} │");
        println!("│ Max absolute error  │ {:<13.6} │", total_max_abs_err);
        println!("│ Mean squared error  │ {:<13.6e} │", mse);
        println!("│ Cosine similarity   │ {cosine:<13.6} │");
        if snr.is_infinite() {
            println!("│ SNR                 │ {:<13} │", "∞ dB");
        } else {
            println!("│ SNR                 │ {:<12.1} dB │", snr);
        }
        println!("│ Values changed      │ {pct_changed:<12.1}% │");
        println!(
            "│ Ops before → after  │ {:<5} → {:<5} │",
            ops_before, ops_after
        );
        println!("└─────────────────────┴───────────────┘");
        println!();
    }

    Ok(())
}

/// Collect all const tensor values from a program, keyed by op name.
///
/// Handles three storage formats:
/// - Plain `const` ops with FP32/FP16 tensor data in inputs["val"]
/// - INT8 quantized `const` ops (UInt8 data + scale/zero_point attributes)
/// - Palettized `constexpr_lut_to_dense` ops (LUT + packed indices)
fn collect_const_tensors(program: &Program) -> HashMap<String, Vec<f32>> {
    let mut result = HashMap::new();

    for function in program.functions.values() {
        for op in &function.body.operations {
            if op.op_type == "constexpr_lut_to_dense" {
                if let Some(values) = reconstruct_palettized(op) {
                    result.insert(op.name.clone(), values);
                }
                continue;
            }

            if op.op_type != "const" {
                continue;
            }

            let val = match op.inputs.get("val").or_else(|| op.attributes.get("val")) {
                Some(v) => v,
                None => continue,
            };

            match val {
                Value::Tensor {
                    data,
                    dtype: ScalarType::Float32,
                    ..
                } => {
                    result.insert(op.name.clone(), tensor_as_f32_slice(data));
                }
                Value::Tensor {
                    data,
                    dtype: ScalarType::Float16,
                    ..
                } => {
                    result.insert(op.name.clone(), fp16_bytes_to_f32(data));
                }
                Value::Tensor {
                    data,
                    dtype: ScalarType::UInt8,
                    ..
                } => {
                    // INT8 quantized — dequantize using scale/zero_point.
                    let scale = match op.attributes.get("scale") {
                        Some(Value::Float(s)) => *s as f32,
                        _ => continue,
                    };
                    let zero_point = match op.attributes.get("zero_point") {
                        Some(Value::Float(zp)) => *zp as f32,
                        _ => continue,
                    };
                    let dequantized: Vec<f32> = data
                        .iter()
                        .map(|&q| (q as f32 - zero_point) * scale)
                        .collect();
                    result.insert(op.name.clone(), dequantized);
                }
                _ => {}
            }
        }
    }

    result
}

/// Reconstruct float values from a palettized `constexpr_lut_to_dense` op.
fn reconstruct_palettized(op: &mil_rs::Operation) -> Option<Vec<f32>> {
    let (lut_data, lut_dtype, k) = match op.attributes.get("lut") {
        Some(Value::Tensor { data, dtype, shape }) => (data, dtype, shape[0]),
        _ => return None,
    };

    // Decode LUT centroids to f32.
    let centroids: Vec<f32> = match lut_dtype {
        ScalarType::Float32 => tensor_as_f32_slice(lut_data),
        ScalarType::Float16 => fp16_bytes_to_f32(lut_data),
        _ => return None,
    };

    let (indices_data, n_values) = match op.attributes.get("indices") {
        Some(Value::Tensor { data, shape, .. }) => (data, shape[0]),
        _ => return None,
    };

    // Determine bit-width from LUT size.
    let n_bits = match k {
        4 => 2u8,
        16 => 4,
        64 => 6,
        256 => 8,
        _ => return None,
    };

    let assignments = unpack_indices(indices_data, n_values, n_bits);
    Some(
        assignments
            .iter()
            .map(|&idx| {
                if idx < centroids.len() {
                    centroids[idx]
                } else {
                    0.0
                }
            })
            .collect(),
    )
}

/// Unpack n-bit packed indices (MSB-first packing, matching `pack_indices`).
fn unpack_indices(packed: &[u8], count: usize, n_bits: u8) -> Vec<usize> {
    if n_bits == 8 {
        return packed.iter().take(count).map(|&b| b as usize).collect();
    }

    let mask = (1u16 << n_bits) - 1;
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let bit_offset = i * n_bits as usize;
        let byte_pos = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        // Read a 16-bit window from the packed bytes.
        let hi = packed.get(byte_pos).copied().unwrap_or(0) as u16;
        let lo = packed.get(byte_pos + 1).copied().unwrap_or(0) as u16;
        let window = (hi << 8) | lo;

        let shift = 16 - n_bits as usize - bit_in_byte;
        let idx = ((window >> shift) & mask) as usize;
        result.push(idx);
    }

    result
}

/// Convert raw FP16 little-endian bytes to `Vec<f32>`.
fn fp16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}
