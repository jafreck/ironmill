//! Quality benchmarks for quantization fidelity.
//!
//! Measures per-tensor MSE and PSNR for different quantization methods,
//! enabling automated tracking of weight fidelity impact from optimizations.

use half::f16;
use mil_rs::ir::passes::PolarQuantPass;
use mil_rs::ir::passes::tensor_utils::tensor_as_f32_slice;
use mil_rs::{Pass, Program, ScalarType, Value};

/// Result of a quality benchmark for one (model, method) pair.
#[derive(Debug, Clone)]
pub struct QualityResult {
    #[allow(dead_code)]
    pub model_name: String,
    pub method: String,
    pub bits: u8,
    pub mse: f64,
    pub psnr_db: f64,
    pub compression_ratio: f64,
}

/// Compute MSE between two f32 slices.
fn compute_mse(original: &[f32], reconstructed: &[f32]) -> f64 {
    assert_eq!(original.len(), reconstructed.len());
    let sum: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    sum / original.len() as f64
}

/// Compute PSNR in dB from MSE and signal range.
fn compute_psnr(mse: f64, max_val: f64) -> f64 {
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (max_val.powi(2) / mse).log10()
}

/// Unpack n-bit packed indices (MSB-first) back to individual index values.
fn unpack_indices(packed: &[u8], n_bits: u8, count: usize) -> Vec<usize> {
    if n_bits == 8 {
        return packed.iter().map(|&b| b as usize).collect();
    }

    let mask = (1u16 << n_bits) - 1;
    let mut indices = Vec::with_capacity(count);

    for i in 0..count {
        let bit_offset = i * n_bits as usize;
        let byte_pos = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        let raw = if byte_pos + 1 < packed.len() {
            ((packed[byte_pos] as u16) << 8) | (packed[byte_pos + 1] as u16)
        } else {
            (packed[byte_pos] as u16) << 8
        };

        let idx = (raw >> (16 - n_bits as usize - bit_in_byte)) & mask;
        indices.push(idx as usize);
    }

    indices
}

/// Reconstruct dequantized f32 values from a constexpr_lut_to_dense op.
fn reconstruct_from_lut_op(op: &mil_rs::Operation, original_shape: &[usize]) -> Option<Vec<f32>> {
    let lut = op.attributes.get("lut")?;
    let indices = op.attributes.get("indices")?;

    let (lut_data, lut_dtype) = match lut {
        Value::Tensor { data, dtype, .. } => (data, *dtype),
        _ => return None,
    };

    let lut_f32: Vec<f32> = match lut_dtype {
        ScalarType::Float16 => lut_data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        ScalarType::Float32 => tensor_as_f32_slice(lut_data).to_vec(),
        _ => return None,
    };

    let packed_bytes = match indices {
        Value::Tensor { data, .. } => data,
        _ => return None,
    };

    let numel: usize = original_shape.iter().product();
    let n_bits = match lut_f32.len() {
        2 => 1,
        4 => 2,
        16 => 4,
        64 => 6,
        256 => 8,
        _ => return None,
    };

    let unpacked = unpack_indices(packed_bytes, n_bits, numel);

    let reconstructed: Vec<f32> = unpacked
        .iter()
        .map(|&idx| lut_f32.get(idx).copied().unwrap_or(0.0))
        .collect();

    Some(reconstructed)
}

/// Run quality benchmarks on a program's const tensors.
///
/// For each const op with a large enough FP32 tensor:
/// 1. Save original weights
/// 2. Apply quantization pass
/// 3. Reconstruct dequantized weights from LUT + indices
/// 4. Compute MSE and PSNR vs original
pub fn measure_program_quality(program: &Program, method: &str, bits: u8) -> Vec<QualityResult> {
    let mut results = Vec::new();

    for function in program.functions.values() {
        for op in &function.body.operations {
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
                    shape,
                    dtype: ScalarType::Float32,
                } => {
                    let numel: usize = shape.iter().product();
                    if numel < 1024 {
                        continue;
                    }

                    let original = tensor_as_f32_slice(data).to_vec();
                    let max_val = original.iter().map(|v| v.abs()).fold(0.0f32, f32::max) as f64;

                    let mut test_program = program.clone();
                    let pass = PolarQuantPass::new(bits);
                    if pass.run(&mut test_program).is_ok() {
                        let original_bytes = numel * 4; // f32
                        let compressed_bytes =
                            (numel * bits as usize).div_ceil(8) + (1 << bits) * 2;

                        // Find the quantized op by matching the original op name.
                        let mut mse = 0.0;
                        let mut psnr_db = 0.0;

                        for qfn in test_program.functions.values() {
                            for qop in &qfn.body.operations {
                                if qop.op_type == "constexpr_lut_to_dense" && qop.name == op.name {
                                    if let Some(reconstructed) = reconstruct_from_lut_op(qop, shape)
                                    {
                                        // Row norms are factored out, so the
                                        // reconstructed values are the unit-norm
                                        // quantized weights. Apply norms for MSE.
                                        let rank = shape.len();
                                        let (_rows, cols) = if rank >= 2 {
                                            let c = shape[rank - 1];
                                            let r: usize = shape[..rank - 1].iter().product();
                                            (r, c)
                                        } else {
                                            (1, shape[0])
                                        };

                                        // Find corresponding norms const op.
                                        let norms_name = format!("{}_polar_norms", op.name);
                                        let norms = qfn
                                            .body
                                            .operations
                                            .iter()
                                            .find(|o| o.name == norms_name)
                                            .and_then(|o| {
                                                o.inputs
                                                    .get("val")
                                                    .or_else(|| o.attributes.get("val"))
                                            });

                                        let scaled: Vec<f32> = if let Some(Value::Tensor {
                                            data: norms_data,
                                            dtype: norms_dtype,
                                            ..
                                        }) = norms
                                        {
                                            let norms_f32: Vec<f32> = match norms_dtype {
                                                ScalarType::Float16 => norms_data
                                                    .chunks_exact(2)
                                                    .map(|c| {
                                                        f16::from_le_bytes([c[0], c[1]]).to_f32()
                                                    })
                                                    .collect(),
                                                _ => tensor_as_f32_slice(norms_data).to_vec(),
                                            };
                                            reconstructed
                                                .iter()
                                                .enumerate()
                                                .map(|(i, &v)| {
                                                    let row = i / cols;
                                                    let norm =
                                                        norms_f32.get(row).copied().unwrap_or(1.0);
                                                    v * norm
                                                })
                                                .collect()
                                        } else {
                                            reconstructed
                                        };

                                        mse = compute_mse(&original, &scaled);
                                        psnr_db = compute_psnr(mse, max_val);
                                    }
                                }
                            }
                        }

                        results.push(QualityResult {
                            model_name: op.name.clone(),
                            method: method.to_string(),
                            bits,
                            mse,
                            psnr_db,
                            compression_ratio: original_bytes as f64 / compressed_bytes as f64,
                        });
                    }
                }
                _ => continue,
            }
        }
    }

    results
}

/// Format quality results as a summary table string.
#[allow(dead_code)]
pub fn format_quality_table(results: &[QualityResult]) -> String {
    let mut out = String::new();
    out.push_str("| Tensor | Method | Bits | MSE | PSNR (dB) | Compression |\n");
    out.push_str("|--------|--------|------|-----|-----------|-------------|\n");
    for r in results {
        out.push_str(&format!(
            "| {} | {} | {} | {:.6} | {:.1} | {:.1}× |\n",
            r.model_name, r.method, r.bits, r.mse, r.psnr_db, r.compression_ratio
        ));
    }
    out
}

/// Aggregate quality results into a per-model summary.
#[derive(Debug, Clone)]
pub struct QualitySummary {
    pub model_name: String,
    pub method: String,
    pub bits: u8,
    #[allow(dead_code)]
    pub tensor_count: usize,
    pub avg_mse: f64,
    pub avg_psnr_db: f64,
    pub avg_compression_ratio: f64,
    pub worst_psnr_db: f64,
}

/// Compute a summary from per-tensor quality results.
pub fn summarize_quality(model_name: &str, results: &[QualityResult]) -> Option<QualitySummary> {
    if results.is_empty() {
        return None;
    }
    let n = results.len() as f64;
    let method = results[0].method.clone();
    let bits = results[0].bits;
    Some(QualitySummary {
        model_name: model_name.to_string(),
        method,
        bits,
        tensor_count: results.len(),
        avg_mse: results.iter().map(|r| r.mse).sum::<f64>() / n,
        avg_psnr_db: results.iter().map(|r| r.psnr_db).sum::<f64>() / n,
        avg_compression_ratio: results.iter().map(|r| r.compression_ratio).sum::<f64>() / n,
        worst_psnr_db: results
            .iter()
            .map(|r| r.psnr_db)
            .fold(f64::INFINITY, f64::min),
    })
}

/// Format a quality summary table across models and optimizations.
pub fn format_quality_summary(summaries: &[QualitySummary]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "Weight Fidelity Impact").unwrap();
    writeln!(out, "{}", "─".repeat(80)).unwrap();
    writeln!(
        out,
        "{:<18} {:<12} {:>6} {:>10} {:>12} {:>10} {:>10}",
        "Model", "Method", "Bits", "Avg MSE", "Avg PSNR", "Worst PSNR", "Compress"
    )
    .unwrap();
    writeln!(
        out,
        "{:<18} {:<12} {:>6} {:>10} {:>12} {:>10} {:>10}",
        "─────────────────",
        "───────────",
        "──────",
        "──────────",
        "────────────",
        "──────────",
        "──────────"
    )
    .unwrap();

    for s in summaries {
        let status = if s.worst_psnr_db > 30.0 {
            "✓ SAFE"
        } else if s.worst_psnr_db > 20.0 {
            "⚠ WARN"
        } else {
            "✗ RISK"
        };
        writeln!(
            out,
            "{:<18} {:<12} {:>6} {:>10.6} {:>10.1} dB {:>10.1} dB {:>8.1}×  {}",
            s.model_name,
            s.method,
            s.bits,
            s.avg_mse,
            s.avg_psnr_db,
            s.worst_psnr_db,
            s.avg_compression_ratio,
            status
        )
        .unwrap();
    }
    out
}
