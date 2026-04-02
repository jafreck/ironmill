//! Activation-Aware Weight Quantization (AWQ) pass.
//!
//! Uses per-channel activation magnitudes (from calibration data) to identify
//! salient channels and apply optimal per-channel scaling before quantization.
//! This preserves important weight channels more faithfully than uniform MinMax.
//!
//! Algorithm per linear op weight:
//!   1. Look up activation magnitudes for the layer
//!   2. Identify salient channels (top percentile by magnitude)
//!   3. Grid-search per-channel scale factors to minimize quantization MSE
//!   4. Apply scales, quantize with per-group affine, emit rewritten op
//!
//! The per-channel scales are stored as an `"awq_channel_scales"` attribute
//! on the op — Phase 2 Task 2.2 will fuse them into adjacent ops.

use std::collections::HashMap;

use super::tensor_utils::{f32_slice_to_bytes, tensor_as_f32_slice};
use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::Value;

/// AWQ quantization pass.
///
/// Requires pre-computed per-channel activation magnitudes (mean |x|) obtained
/// from calibration. The `channel_magnitudes` map is keyed by the operation
/// name of the const weight.
pub struct AwqQuantizePass {
    /// Quantization bit width (4 or 8).
    pub bits: u8,
    /// Group size for per-group quantization (typically 128).
    pub group_size: usize,
    /// Per-channel activation magnitudes: op name → `Vec<f32>` of mean |x|.
    pub channel_magnitudes: HashMap<String, Vec<f32>>,
    /// Number of candidate scale values to evaluate in grid search (default 20).
    pub grid_search_steps: usize,
    /// Fraction of channels (by magnitude rank) considered salient (default 0.99
    /// means the top 1% are salient).
    pub salient_percentile: f32,
}

impl AwqQuantizePass {
    /// Create a new AWQ pass with default grid search parameters.
    pub fn new(bits: u8, group_size: usize, channel_magnitudes: HashMap<String, Vec<f32>>) -> Self {
        assert!(bits == 4 || bits == 8, "AWQ only supports 4-bit or 8-bit");
        Self {
            bits,
            group_size,
            channel_magnitudes,
            grid_search_steps: 20,
            salient_percentile: 0.99,
        }
    }

    /// Override the grid search step count.
    pub fn with_grid_steps(mut self, steps: usize) -> Self {
        self.grid_search_steps = steps;
        self
    }

    /// Override the salient percentile threshold.
    pub fn with_salient_percentile(mut self, pct: f32) -> Self {
        self.salient_percentile = pct;
        self
    }
}

/// Maximum quantized value for a given bit width.
fn qmax(bits: u8) -> f32 {
    (1u32 << bits) as f32 - 1.0
}

impl Pass for AwqQuantizePass {
    fn name(&self) -> &str {
        "awq-quantization"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        let qmax = qmax(self.bits);

        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                if op.op_type != "const" {
                    continue;
                }

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
                    op.inputs.remove("val").unwrap()
                } else {
                    op.attributes.remove("val").unwrap()
                };

                if let Value::Tensor {
                    data,
                    shape,
                    dtype: _,
                } = val
                {
                    let floats = tensor_as_f32_slice(&data);

                    // Need at least 2-D to have channels (rows × columns).
                    if shape.len() < 2 {
                        emit_fallback_per_group(
                            op,
                            &floats,
                            &shape,
                            self.group_size,
                            qmax,
                            self.bits,
                        );
                        continue;
                    }

                    let magnitudes = self.channel_magnitudes.get(&op.name);

                    match magnitudes {
                        Some(mags) if !mags.is_empty() => {
                            let num_channels = shape[0];
                            let mags = if mags.len() >= num_channels {
                                &mags[..num_channels]
                            } else {
                                // Magnitude vector shorter than channel count —
                                // fall back to MinMax.
                                emit_fallback_per_group(
                                    op,
                                    &floats,
                                    &shape,
                                    self.group_size,
                                    qmax,
                                    self.bits,
                                );
                                continue;
                            };

                            // AWQ: compute optimal per-channel scales.
                            let channel_scales = compute_awq_scales(
                                &floats,
                                &shape,
                                mags,
                                self.grid_search_steps,
                                self.salient_percentile,
                                qmax,
                            );

                            // Apply scales and quantize.
                            let channel_size: usize = shape[1..].iter().product();
                            let mut scaled_weights = floats.clone();
                            for (ch, &s) in channel_scales.iter().enumerate() {
                                if s != 1.0 {
                                    let start = ch * channel_size;
                                    let end = start + channel_size;
                                    for w in &mut scaled_weights[start..end] {
                                        *w /= s;
                                    }
                                }
                            }

                            // Per-group quantization on the scaled weights.
                            emit_per_group_with_scales(
                                op,
                                &scaled_weights,
                                &shape,
                                self.group_size,
                                qmax,
                                &channel_scales,
                                self.bits,
                            );
                        }
                        _ => {
                            // No calibration data — fall back to standard MinMax.
                            emit_fallback_per_group(
                                op,
                                &floats,
                                &shape,
                                self.group_size,
                                qmax,
                                self.bits,
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AWQ scale computation
// ---------------------------------------------------------------------------

/// Compute optimal per-channel scales using grid search on salient channels.
///
/// For each salient channel, searches `grid_steps` candidate scales to find
/// the one that minimizes quantize-then-dequantize MSE for that channel's
/// weight values.
fn compute_awq_scales(
    floats: &[f32],
    shape: &[usize],
    magnitudes: &[f32],
    grid_steps: usize,
    salient_percentile: f32,
    qmax: f32,
) -> Vec<f32> {
    let num_channels = shape[0];
    let channel_size: usize = shape[1..].iter().product();

    // Determine which channels are salient.
    let salient_threshold = find_salient_threshold(magnitudes, salient_percentile);

    // Compute a reasonable max_scale from the magnitude range.
    let max_mag = magnitudes.iter().cloned().fold(0.0_f32, f32::max);
    let max_scale = if max_mag > 1e-8 { max_mag } else { 1.0 };

    let mut scales = vec![1.0_f32; num_channels];

    for ch in 0..num_channels {
        if magnitudes[ch] < salient_threshold {
            // Not salient — keep scale = 1.0.
            continue;
        }

        let start = ch * channel_size;
        let end = start + channel_size;
        let original = &floats[start..end];

        let mut best_scale = 1.0_f32;
        let mut best_mse = f32::INFINITY;

        let grid_steps = grid_steps.max(1);
        for step in 0..grid_steps {
            // Scale candidates linearly spaced in [0.1, max_scale].
            let t = step as f32 / (grid_steps - 1).max(1) as f32;
            let candidate = 0.1 + t * (max_scale - 0.1);

            if candidate <= 0.0 {
                continue;
            }

            let mse = compute_scaled_quantization_mse(original, candidate, qmax);

            if mse < best_mse {
                best_mse = mse;
                best_scale = candidate;
            }
        }

        scales[ch] = best_scale;
    }

    scales
}

/// Find the magnitude threshold above which channels are considered salient.
///
/// Channels with magnitude ≥ this threshold are in the top `(1 - percentile)`
/// fraction. E.g., percentile=0.99 means the top 1% are salient.
fn find_salient_threshold(magnitudes: &[f32], percentile: f32) -> f32 {
    if magnitudes.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = magnitudes.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Index at which salient channels begin.
    let idx = ((percentile * sorted.len() as f32) as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// Compute MSE of: scale weights by 1/s, quantize, dequantize, scale back by s.
fn compute_scaled_quantization_mse(original: &[f32], scale: f32, qmax: f32) -> f32 {
    if original.is_empty() {
        return 0.0;
    }

    let inv_scale = 1.0 / scale;

    // Scale down.
    let scaled: Vec<f32> = original.iter().map(|&w| w * inv_scale).collect();

    // Quantize then dequantize the scaled values.
    let (quantized, q_scale, q_zp) = quantize_affine(&scaled, qmax);
    let dequantized: Vec<f32> = quantized
        .iter()
        .map(|&q| (q as f32 - q_zp) * q_scale)
        .collect();

    // Scale back up and compute MSE vs. original.
    let mut sum_sq = 0.0_f32;
    for (i, &orig) in original.iter().enumerate() {
        let reconstructed = dequantized[i] * scale;
        let err = orig - reconstructed;
        sum_sq += err * err;
    }
    sum_sq / original.len() as f32
}

use super::affine_quantize::quantize_affine;
use super::int4_pack::pack_int4;

// ---------------------------------------------------------------------------
// Emission helpers
// ---------------------------------------------------------------------------

/// Emit per-group quantization with AWQ channel scales stored as attribute.
fn emit_per_group_with_scales(
    op: &mut crate::ir::operation::Operation,
    scaled_floats: &[f32],
    shape: &[usize],
    group_size: usize,
    qmax: f32,
    channel_scales: &[f32],
    bits: u8,
) {
    assert!(group_size > 0, "group_size must be positive");

    let ndim = shape.len();
    let last_dim = if ndim > 0 { shape[ndim - 1] } else { 1 };
    let outer_count: usize = if ndim > 1 {
        shape[..ndim - 1].iter().product()
    } else {
        1
    };
    let n_groups = last_dim.div_ceil(group_size);

    let mut all_quantized = Vec::with_capacity(scaled_floats.len());
    let mut all_scales = Vec::with_capacity(outer_count * n_groups);
    let mut all_zero_points = Vec::with_capacity(outer_count * n_groups);

    for row in 0..outer_count {
        let row_start = row * last_dim;
        for g in 0..n_groups {
            let g_start = row_start + g * group_size;
            let g_end = (g_start + group_size).min(row_start + last_dim);
            let group_slice = &scaled_floats[g_start..g_end];
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
        data: packed_data,
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

    // Store AWQ channel scales for downstream fusion (Phase 2 Task 2.2).
    let num_channels = channel_scales.len();
    op.attributes.insert(
        "awq_channel_scales".to_string(),
        Value::Tensor {
            data: f32_slice_to_bytes(channel_scales),
            shape: vec![num_channels],
            dtype: ScalarType::Float32,
        },
    );

    let out_type = TensorType::new(ScalarType::Float32, shape.to_vec());
    if let Some(slot) = op.output_types.get_mut(0) {
        *slot = Some(out_type);
    } else {
        op.output_types.push(Some(out_type));
    }
}

/// Fallback: standard per-group MinMax quantization (no AWQ scaling).
fn emit_fallback_per_group(
    op: &mut crate::ir::operation::Operation,
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
        data: packed_data,
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

    /// Compute MSE between original floats and dequantized reconstruction.
    fn reconstruction_mse(original: &[f32], program: &Program) -> f32 {
        use crate::ir::passes::int4_pack::unpack_int4;

        let op = &program.functions["main"].body.operations[0];
        let q_data_raw = match op.attributes.get("quantized_data") {
            Some(Value::Tensor { data, .. }) => data,
            _ => panic!("missing quantized_data"),
        };

        // Get bit_width to determine if data is packed INT4.
        let bit_width = match op.attributes.get("bit_width") {
            Some(Value::Int(b)) => *b as u8,
            _ => 8,
        };

        // Unpack INT4 data if needed.
        let q_data: Vec<u8> = if bit_width == 4 {
            unpack_int4(q_data_raw, original.len())
        } else {
            q_data_raw.to_vec()
        };

        // Get per-group scales and zero points.
        let scales = match op.attributes.get("scale") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data),
            Some(Value::Float(f)) => vec![*f as f32],
            _ => panic!("missing scale"),
        };
        let zero_points = match op.attributes.get("zero_point") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data),
            Some(Value::Float(f)) => vec![*f as f32],
            _ => panic!("missing zero_point"),
        };

        let group_size = match op.attributes.get("group_size") {
            Some(Value::Int(g)) => *g as usize,
            _ => original.len(),
        };

        // Get AWQ channel scales if present.
        let channel_scales = match op.attributes.get("awq_channel_scales") {
            Some(Value::Tensor { data, .. }) => Some(tensor_as_f32_slice(data)),
            _ => None,
        };

        // Determine shape to figure out layout.
        let (num_rows, row_len) = match op.attributes.get("quantized_data") {
            Some(Value::Tensor { shape, .. }) if shape.len() >= 2 => {
                let outer: usize = shape[..shape.len() - 1].iter().product();
                let last = *shape.last().unwrap();
                (outer, last)
            }
            Some(Value::Tensor { shape, .. }) => (1, shape.iter().product()),
            _ => (1, original.len()),
        };

        let n_groups = row_len.div_ceil(group_size);

        let mut sum_sq = 0.0_f32;
        for row in 0..num_rows {
            for g in 0..n_groups {
                let param_idx = row * n_groups + g;
                let s = scales[param_idx];
                let zp = zero_points[param_idx];

                let g_start = row * row_len + g * group_size;
                let g_end = (g_start + group_size).min(row * row_len + row_len);

                for i in g_start..g_end {
                    let mut reconstructed = (q_data[i] as f32 - zp) * s;
                    // If AWQ scales present, multiply back by channel scale.
                    if let Some(ref cs) = channel_scales {
                        let ch = i / row_len; // which row = which output channel
                        if ch < cs.len() {
                            reconstructed *= cs[ch];
                        }
                    }
                    let err = original[i] - reconstructed;
                    sum_sq += err * err;
                }
            }
        }

        sum_sq / original.len() as f32
    }

    // -----------------------------------------------------------------------
    // Test: AWQ produces lower MSE than MinMax on weights with varying
    // channel importance.
    // -----------------------------------------------------------------------

    #[test]
    fn awq_lower_mse_than_minmax() {
        // 4 channels × 8 columns.  Channels 0,1 have large magnitudes,
        // channels 2,3 have small values.  AWQ should protect channels 0,1.
        let mut weights = vec![0.0_f32; 4 * 8];
        // Channel 0: large dynamic range
        for i in 0..8 {
            weights[i] = (i as f32) * 10.0 - 35.0;
        }
        // Channel 1: large dynamic range, different pattern
        for i in 0..8 {
            weights[8 + i] = (i as f32).sin() * 20.0;
        }
        // Channel 2: small values
        for i in 0..8 {
            weights[16 + i] = (i as f32) * 0.01;
        }
        // Channel 3: small values
        for i in 0..8 {
            weights[24 + i] = (i as f32) * 0.02 - 0.05;
        }
        let original = weights.clone();

        // Magnitudes: channels 0,1 are far more important.
        let magnitudes = vec![50.0, 40.0, 0.5, 0.3];

        // --- MinMax baseline ---
        let mut minmax_prog = make_program("weight", &weights, vec![4, 8]);
        let minmax_pass = AwqQuantizePass::new(4, 8, HashMap::new());
        minmax_pass.run(&mut minmax_prog).unwrap();
        let minmax_mse = reconstruction_mse(&original, &minmax_prog);

        // --- AWQ ---
        let mut awq_mags = HashMap::new();
        awq_mags.insert("weight".to_string(), magnitudes);
        let mut awq_prog = make_program("weight", &weights, vec![4, 8]);
        let awq_pass = AwqQuantizePass::new(4, 8, awq_mags)
            .with_grid_steps(40)
            .with_salient_percentile(0.5);
        awq_pass.run(&mut awq_prog).unwrap();
        let awq_mse = reconstruction_mse(&original, &awq_prog);

        assert!(
            awq_mse <= minmax_mse,
            "AWQ MSE ({awq_mse}) should be <= MinMax MSE ({minmax_mse})"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Salient channels get non-trivial scales.
    // -----------------------------------------------------------------------

    #[test]
    fn salient_channels_get_nontrivial_scales() {
        let mut weights = vec![0.0_f32; 4 * 16];
        for i in 0..4 {
            for j in 0..16 {
                weights[i * 16 + j] = ((i * 16 + j) as f32) * 0.1 - 3.0;
            }
        }
        // Channels 0,1 salient; 2,3 not.
        let magnitudes = vec![100.0, 80.0, 1.0, 0.5];

        let mut mags = HashMap::new();
        mags.insert("w".to_string(), magnitudes);

        let mut prog = make_program("w", &weights, vec![4, 16]);
        let pass = AwqQuantizePass::new(4, 16, mags).with_salient_percentile(0.5);
        pass.run(&mut prog).unwrap();

        let op = &prog.functions["main"].body.operations[0];
        let scales = match op.attributes.get("awq_channel_scales") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data),
            _ => panic!("missing awq_channel_scales"),
        };

        assert_eq!(scales.len(), 4);

        // At least one salient channel should have a non-1.0 scale.
        let has_nontrivial = scales[0] != 1.0 || scales[1] != 1.0;
        assert!(
            has_nontrivial,
            "salient channels should have non-1.0 scales: {scales:?}"
        );

        // Non-salient channels should remain at 1.0.
        assert_eq!(scales[2], 1.0, "non-salient channel 2 should be 1.0");
        assert_eq!(scales[3], 1.0, "non-salient channel 3 should be 1.0");
    }

    // -----------------------------------------------------------------------
    // Test: Grid search selects scales that improve MSE.
    // -----------------------------------------------------------------------

    #[test]
    fn grid_search_improves_mse() {
        // Construct weights where a non-unity scale clearly helps.
        let mut weights = vec![0.0_f32; 2 * 32];
        // Channel 0: large outlier pattern.
        for j in 0..32 {
            weights[j] = if j < 4 { 100.0 } else { 0.1 * j as f32 };
        }
        // Channel 1: uniform small.
        for j in 0..32 {
            weights[32 + j] = 0.01 * j as f32;
        }
        let original = weights.clone();

        let magnitudes = vec![90.0, 1.0];

        // AWQ with grid search (all channels salient via percentile=0.0).
        let mut mags = HashMap::new();
        mags.insert("w".to_string(), magnitudes);

        let mut prog = make_program("w", &weights, vec![2, 32]);
        let pass = AwqQuantizePass::new(4, 32, mags)
            .with_grid_steps(50)
            .with_salient_percentile(0.0);
        pass.run(&mut prog).unwrap();
        let awq_mse = reconstruction_mse(&original, &prog);

        // Compare with scale=1.0 everywhere (plain MinMax).
        let mut plain_prog = make_program("w", &weights, vec![2, 32]);
        let plain_pass = AwqQuantizePass::new(4, 32, HashMap::new());
        plain_pass.run(&mut plain_prog).unwrap();
        let plain_mse = reconstruction_mse(&original, &plain_prog);

        assert!(
            awq_mse <= plain_mse,
            "Grid-searched AWQ MSE ({awq_mse}) should be <= plain MSE ({plain_mse})"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Fallback to MinMax when no calibration data.
    // -----------------------------------------------------------------------

    #[test]
    fn fallback_to_minmax_no_calibration() {
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let mut prog = make_program("weight", &values, vec![2, 8]);

        // Empty magnitudes → no calibration data.
        let pass = AwqQuantizePass::new(4, 8, HashMap::new());
        pass.run(&mut prog).unwrap();

        let op = &prog.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");

        // Should NOT have awq_channel_scales (fallback path).
        assert!(
            op.attributes.get("awq_channel_scales").is_none(),
            "fallback should not produce awq_channel_scales"
        );

        // Should still have quantized data.
        assert!(op.attributes.get("quantized_data").is_some());
        assert!(op.attributes.get("scale").is_some());
        assert!(op.attributes.get("zero_point").is_some());
    }

    // -----------------------------------------------------------------------
    // Test: 8-bit AWQ works.
    // -----------------------------------------------------------------------

    #[test]
    fn awq_8bit_quantization() {
        let weights: Vec<f32> = (0..32).map(|i| (i as f32) * 0.3 - 4.0).collect();
        let magnitudes = vec![10.0, 1.0, 0.5, 0.2];

        let mut mags = HashMap::new();
        mags.insert("w".to_string(), magnitudes);

        let mut prog = make_program("w", &weights, vec![4, 8]);
        let pass = AwqQuantizePass::new(8, 8, mags).with_salient_percentile(0.5);
        pass.run(&mut prog).unwrap();

        let op = &prog.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");
        assert!(op.attributes.get("awq_channel_scales").is_some());

        // All quantized values should be in [0, 255].
        let q_data = match op.attributes.get("quantized_data") {
            Some(Value::Tensor { data, .. }) => data,
            _ => panic!("missing quantized_data"),
        };
        for &b in q_data {
            assert!(b <= 255, "INT8 value {b} out of range");
        }
    }

    // -----------------------------------------------------------------------
    // Test: 1-D tensor falls back gracefully.
    // -----------------------------------------------------------------------

    #[test]
    fn one_dim_tensor_fallback() {
        let values = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut prog = make_program("bias", &values, vec![4]);

        let mut mags = HashMap::new();
        mags.insert("bias".to_string(), vec![5.0, 3.0, 1.0, 0.5]);

        let pass = AwqQuantizePass::new(4, 4, mags);
        pass.run(&mut prog).unwrap();

        let op = &prog.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");
        // 1-D goes through fallback, no AWQ scales.
        assert!(op.attributes.get("awq_channel_scales").is_none());
    }
}
