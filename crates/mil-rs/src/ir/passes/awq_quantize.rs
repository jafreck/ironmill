//! Activation-Aware Weight Quantization (AWQ) pass.
//!
//! Uses per-channel activation magnitudes (from calibration data) to identify
//! salient channels and apply optimal per-channel scaling before quantization.
//! This preserves important weight channels more faithfully than uniform MinMax.
//!
//! Algorithm per linear op weight (Equations 4–5, Lin et al. MLSys 2024):
//!   1. Look up per-input-channel activation magnitudes s_X for the layer
//!   2. Grid-search a single exponent α ∈ [0, 1] to minimize the
//!      activation-weighted quantization loss: s[c] = s_X[c]^α
//!   3. Apply scales, quantize with per-group affine, emit rewritten op
//!
//! The per-channel scales are stored as an `"awq_channel_scales"` attribute
//! on the op — Phase 2 Task 2.2 will fuse them into adjacent ops.

use std::collections::HashMap;

use super::tensor_utils::{f32_slice_to_bytes, tensor_as_f32_slice, tensor_f16_as_f32_slice};
use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::Value;

/// AWQ quantization pass.
///
/// Requires pre-computed per-channel activation magnitudes (mean |x|) obtained
/// from calibration. The `channel_magnitudes` map is keyed by the operation
/// name of the const weight. Optionally accepts raw calibration activations
/// to compute the paper's exact loss (Equation 4) instead of the mag²-weighted
/// MSE approximation.
pub struct AwqQuantizePass {
    /// Quantization bit width (4 or 8).
    pub bits: u8,
    /// Group size for per-group quantization (typically 128).
    pub group_size: usize,
    /// Per-channel activation magnitudes: op name → `Vec<f32>` of mean |x|.
    pub channel_magnitudes: HashMap<String, Vec<f32>>,
    /// Raw calibration activations: op name → flattened `[tokens, features]`.
    pub calibration_activations: HashMap<String, Vec<f32>>,
    /// Number of tokens in each calibration activation snapshot.
    pub calibration_token_count: usize,
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
            calibration_activations: HashMap::new(),
            calibration_token_count: 0,
            grid_search_steps: 20,
            salient_percentile: 0.99,
        }
    }

    /// Set raw calibration activations and token count for exact loss computation.
    pub fn with_calibration_activations(
        mut self,
        activations: HashMap<String, Vec<f32>>,
        token_count: usize,
    ) -> Self {
        self.calibration_activations = activations;
        self.calibration_token_count = token_count;
        self
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

        // Cache α per magnitude group: all projections sharing the same
        // norm receive identical magnitudes, so they must use the same α
        // (and thus the same scales) for correct norm-gamma fusion.
        let mut alpha_cache: HashMap<Vec<u8>, f32> = HashMap::new();

        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                if op.op_type != "const" {
                    continue;
                }

                // Locate a float tensor value (FP32 or FP16) in inputs or attributes.
                let is_float = |v: Option<&Value>| {
                    matches!(
                        v,
                        Some(Value::Tensor {
                            dtype: ScalarType::Float32 | ScalarType::Float16,
                            ..
                        })
                    )
                };

                let in_inputs = is_float(op.inputs.get("val"));
                let in_attrs = !in_inputs && is_float(op.attributes.get("val"));

                if !in_inputs && !in_attrs {
                    continue;
                }

                // Check eligibility before removing the value.
                let (numel, rank) = {
                    let val = if in_inputs {
                        op.inputs.get("val").unwrap()
                    } else {
                        op.attributes.get("val").unwrap()
                    };
                    if let Value::Tensor { shape, .. } = val {
                        (shape.iter().product::<usize>(), shape.len())
                    } else {
                        continue;
                    }
                };

                // Skip small tensors and 1D tensors (norms, biases).
                if numel < 1024 || rank < 2 {
                    continue;
                }

                let val = if in_inputs {
                    op.inputs.remove("val").unwrap()
                } else {
                    op.attributes.remove("val").unwrap()
                };

                if let Value::Tensor { data, shape, dtype } = val {
                    let floats = match dtype {
                        ScalarType::Float32 => tensor_as_f32_slice(&data),
                        ScalarType::Float16 => tensor_f16_as_f32_slice(&data),
                        _ => unreachable!(),
                    };

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
                            let in_features = *shape.last().unwrap_or(&1);
                            let mags = if mags.len() >= in_features {
                                &mags[..in_features]
                            } else {
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

                            // Key for magnitude group: all ops with same mags
                            // share a norm and must use the same α.
                            let mag_key: Vec<u8> =
                                mags.iter().flat_map(|v| v.to_le_bytes()).collect();

                            // Look up raw calibration activations for this op.
                            let cal_act = self.calibration_activations.get(&op.name);
                            let cal_tokens = self.calibration_token_count;

                            // Search α on first encounter, reuse for group.
                            // This ensures all projections sharing a norm use
                            // identical scales, making norm-gamma fusion correct.
                            let alpha = *alpha_cache.entry(mag_key).or_insert_with(|| {
                                search_alpha(
                                    &floats,
                                    &shape,
                                    mags,
                                    self.grid_search_steps,
                                    self.group_size,
                                    qmax,
                                    cal_act.map(|a| a.as_slice()),
                                    cal_tokens,
                                )
                            });

                            // s[c] = s_X[c]^α (Equation 5)
                            let channel_scales: Vec<f32> =
                                mags.iter().map(|&m| m.max(1e-8).powf(alpha)).collect();

                            // Apply scales per-column (input channel):
                            // W_scaled[:, c] = W[:, c] * s[c]
                            let out_features = shape[0];
                            let mut scaled_weights = floats.clone();
                            for row in 0..out_features {
                                for (col, &s) in channel_scales.iter().enumerate() {
                                    if s != 1.0 {
                                        scaled_weights[row * in_features + col] *= s;
                                    }
                                }
                            }

                            // Quantize the up-scaled weights. AWQ channel scales
                            // are stored on the op and written to the GPU bundle.
                            // The Metal kernel divides by s[c] during dequant.
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

/// Compute optimal per-input-channel scales using Equation 5 from the AWQ
/// paper (Lin et al., MLSys 2024).
///
/// Searches a single exponent α ∈ [0, 1] such that `s[c] = s_X[c]^α`
/// minimizes the quantization loss. When raw calibration activations are
/// available, uses the paper's exact loss (Equation 4):
///   L(s) = ||Q(W · diag(s)) · diag(s)^{-1} · X − W · X||²
/// Otherwise falls back to the mag²-weighted MSE approximation.
///
/// Because α is shared across all channels, every projection sharing the
/// same norm produces the same scale vector — enabling offline norm-gamma
/// fusion.
fn search_alpha(
    floats: &[f32],
    shape: &[usize],
    magnitudes: &[f32],
    grid_steps: usize,
    group_size: usize,
    qmax: f32,
    activations: Option<&[f32]>,
    token_count: usize,
) -> f32 {
    let out_features = shape[0];
    let in_features = *shape.last().unwrap_or(&1);
    let grid_steps = grid_steps.max(1);
    let n_groups = in_features.div_ceil(group_size);

    // Validate activations: must have the right shape [tokens, in_features].
    let use_exact = activations.is_some()
        && token_count > 0
        && activations.unwrap().len() == token_count * in_features;

    // Pre-compute W · X^T = [out_features, tokens] for the reference output.
    // X is stored as [tokens, in_features] row-major.
    let (ref_output, n_tokens) = if use_exact {
        let act = activations.unwrap();
        let n_tokens = token_count;
        // W [out, in] × X^T [in, tokens] → [out, tokens]
        let mut wx = vec![0.0f32; out_features * n_tokens];
        for row in 0..out_features {
            for t in 0..n_tokens {
                let mut dot = 0.0f32;
                for c in 0..in_features {
                    dot += floats[row * in_features + c] * act[t * in_features + c];
                }
                wx[row * n_tokens + t] = dot;
            }
        }
        (Some(wx), n_tokens)
    } else {
        (None, 0)
    };

    let mut best_alpha = 0.0_f32;
    let mut best_loss = f32::INFINITY;

    for step in 0..=grid_steps {
        let alpha = step as f32 / grid_steps as f32;

        // s[c] = s_X[c]^α  (Equation 5)
        let scales: Vec<f32> = magnitudes
            .iter()
            .map(|&m| m.max(1e-8).powf(alpha))
            .collect();

        // Build W_dq: quantize W·diag(s), then dequant and divide by s.
        // W_dq[row, col] = dequant(Q(W[row,col]*s[col])) / s[col]
        let mut w_dq = vec![0.0f32; out_features * in_features];
        for row in 0..out_features {
            for g in 0..n_groups {
                let g_start = g * group_size;
                let g_end = (g_start + group_size).min(in_features);

                let scaled_group: Vec<f32> = (g_start..g_end)
                    .map(|col| {
                        let s = if col < scales.len() { scales[col] } else { 1.0 };
                        floats[row * in_features + col] * s
                    })
                    .collect();

                let (quantized, q_scale, q_zp) = quantize_affine(&scaled_group, qmax);

                for (j, col) in (g_start..g_end).enumerate() {
                    let s = if col < scales.len() { scales[col] } else { 1.0 };
                    let dequant = (quantized[j] as f32 - q_zp) * q_scale;
                    w_dq[row * in_features + col] = dequant / s;
                }
            }
        }

        let total_loss = if use_exact {
            // Exact loss: L(s) = ||W_dq · X^T − W · X^T||²
            // = Σ_{row, token} (Σ_c (W_dq[row,c] - W[row,c]) · X[token,c])²
            let act = activations.unwrap();
            let ref_out = ref_output.as_ref().unwrap();
            let mut loss = 0.0f32;
            for row in 0..out_features {
                for t in 0..n_tokens {
                    let mut dq_dot = 0.0f32;
                    for c in 0..in_features {
                        dq_dot += w_dq[row * in_features + c] * act[t * in_features + c];
                    }
                    let err = dq_dot - ref_out[row * n_tokens + t];
                    loss += err * err;
                }
            }
            loss
        } else {
            // Fallback: mag²-weighted MSE approximation.
            let mut loss = 0.0f32;
            for row in 0..out_features {
                for col in 0..in_features {
                    let mag = if col < magnitudes.len() {
                        magnitudes[col]
                    } else {
                        0.0
                    };
                    let importance = mag * mag;
                    let err = floats[row * in_features + col] - w_dq[row * in_features + col];
                    loss += importance * err * err;
                }
            }
            loss
        };

        if total_loss < best_loss {
            best_loss = total_loss;
            best_alpha = alpha;
        }
    }

    best_alpha
}

use super::affine_quantize::quantize_affine;
use super::int4_pack::pack_int4;

// ---------------------------------------------------------------------------
// Emission helpers
// ---------------------------------------------------------------------------

/// Emit per-group quantization with AWQ per-column channel scales.
///
/// Weights were scaled UP by s[c] per input column before quantization.
/// The compensation (dividing activations by s[c]) is handled by the
/// AwqScaleFusionPass, which absorbs 1/s[c] into the preceding norm gamma.
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

    // Store AWQ channel scales for downstream fusion.
    // The fusion pass divides norm gamma by these scales to compensate.
    let num_scales = channel_scales.len();
    op.attributes.insert(
        "awq_channel_scales".to_string(),
        Value::Tensor {
            data: f32_slice_to_bytes(channel_scales),
            shape: vec![num_scales],
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
                    // AWQ scales are per-input-channel (column). Undo the
                    // pre-quantization scale-up: reconstructed /= s[col].
                    if let Some(ref cs) = channel_scales {
                        let col = i - row * row_len; // position within the row
                        if col < cs.len() {
                            reconstructed /= cs[col];
                        }
                    }
                    let err = original[i] - reconstructed;
                    sum_sq += err * err;
                }
            }
        }

        sum_sq / original.len() as f32
    }

    /// Like `reconstruction_mse`, but each column's error is weighted by
    /// `magnitudes[col]²`, matching the AWQ loss (Equation 4 approximation).
    fn weighted_reconstruction_mse(original: &[f32], program: &Program, magnitudes: &[f32]) -> f32 {
        use crate::ir::passes::int4_pack::unpack_int4;

        let op = &program.functions["main"].body.operations[0];
        let q_data_raw = match op.attributes.get("quantized_data") {
            Some(Value::Tensor { data, .. }) => data,
            _ => panic!("missing quantized_data"),
        };
        let bit_width = match op.attributes.get("bit_width") {
            Some(Value::Int(b)) => *b as u8,
            _ => 8,
        };
        let q_data: Vec<u8> = if bit_width == 4 {
            unpack_int4(q_data_raw, original.len())
        } else {
            q_data_raw.to_vec()
        };
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
        let channel_scales = match op.attributes.get("awq_channel_scales") {
            Some(Value::Tensor { data, .. }) => Some(tensor_as_f32_slice(data)),
            _ => None,
        };
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

        let mut weighted_sum = 0.0_f32;
        for row in 0..num_rows {
            for g in 0..n_groups {
                let param_idx = row * n_groups + g;
                let s = scales[param_idx];
                let zp = zero_points[param_idx];
                let g_start = row * row_len + g * group_size;
                let g_end = (g_start + group_size).min(row * row_len + row_len);

                for i in g_start..g_end {
                    let col = i - row * row_len;
                    let mut reconstructed = (q_data[i] as f32 - zp) * s;
                    if let Some(ref cs) = channel_scales {
                        if col < cs.len() {
                            reconstructed /= cs[col];
                        }
                    }
                    let mag = if col < magnitudes.len() {
                        magnitudes[col]
                    } else {
                        0.0
                    };
                    let err = original[i] - reconstructed;
                    weighted_sum += mag * mag * err * err;
                }
            }
        }
        weighted_sum / original.len() as f32
    }

    // -----------------------------------------------------------------------
    // Test: AWQ produces lower MSE than MinMax on weights with varying
    // channel importance.
    // -----------------------------------------------------------------------

    #[test]
    fn awq_lower_mse_than_minmax() {
        // 8 rows × 128 cols. Some input channels are important (large
        // activation magnitudes) and carry outlier weights that benefit from
        // more quantization range.
        //
        // Classic AWQ scenario: important channels carry small weights while
        // unimportant channels carry large weights in the same quantization
        // groups. AWQ re-distributes quantization range toward the important
        // channels.
        let out = 8;
        let inp = 128;
        let group_sz = 32;
        let mut weights = vec![0.0_f32; out * inp];
        for row in 0..out {
            for col in 0..inp {
                if col % 4 == 0 {
                    // Important channels: small weights, high activation mag.
                    weights[row * inp + col] = 0.1 * (row as f32 - 3.5);
                } else {
                    // Unimportant channels: large weights, low activation mag.
                    weights[row * inp + col] = 50.0 * ((col as f32) / inp as f32 - 0.5);
                }
            }
        }
        let original = weights.clone();

        // Per-input-channel magnitudes.
        let magnitudes: Vec<f32> = (0..inp)
            .map(|c| if c % 4 == 0 { 100.0 } else { 0.1 })
            .collect();

        // --- MinMax baseline ---
        let mut minmax_prog = make_program("weight", &weights, vec![out, inp]);
        let minmax_pass = AwqQuantizePass::new(4, group_sz, HashMap::new());
        minmax_pass.run(&mut minmax_prog).unwrap();
        let minmax_wmse = weighted_reconstruction_mse(&original, &minmax_prog, &magnitudes);

        // --- AWQ ---
        let mut awq_mags = HashMap::new();
        awq_mags.insert("weight".to_string(), magnitudes.clone());
        let mut awq_prog = make_program("weight", &weights, vec![out, inp]);
        let awq_pass = AwqQuantizePass::new(4, group_sz, awq_mags).with_grid_steps(40);
        awq_pass.run(&mut awq_prog).unwrap();
        let awq_wmse = weighted_reconstruction_mse(&original, &awq_prog, &magnitudes);

        // AWQ optimizes the activation-weighted MSE; α=0 equals MinMax,
        // so the weighted MSE can only match or improve.
        assert!(
            awq_wmse <= minmax_wmse,
            "AWQ weighted MSE ({awq_wmse}) should be <= MinMax weighted MSE ({minmax_wmse})"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Equation 5 α-search produces scales that vary with magnitude.
    // -----------------------------------------------------------------------

    #[test]
    fn alpha_search_produces_magnitude_dependent_scales() {
        let out = 8;
        let inp = 128;
        let group_sz = 32;
        let mut weights = vec![0.0_f32; out * inp];
        for row in 0..out {
            for col in 0..inp {
                if col % 4 == 0 {
                    weights[row * inp + col] = 0.1 * (row as f32 - 3.5);
                } else {
                    weights[row * inp + col] = 50.0 * ((col as f32) / inp as f32 - 0.5);
                }
            }
        }

        // Clear magnitude variation: every 4th channel very high, rest low.
        let magnitudes: Vec<f32> = (0..inp)
            .map(|c| if c % 4 == 0 { 100.0 } else { 0.5 })
            .collect();

        let mut mags = HashMap::new();
        mags.insert("w".to_string(), magnitudes);

        let mut prog = make_program("w", &weights, vec![out, inp]);
        let pass = AwqQuantizePass::new(4, group_sz, mags).with_grid_steps(20);
        pass.run(&mut prog).unwrap();

        let op = &prog.functions["main"].body.operations[0];
        let scales = match op.attributes.get("awq_channel_scales") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data),
            _ => panic!("missing awq_channel_scales"),
        };

        assert_eq!(scales.len(), inp);

        // With Eq. 5 (s[c] = s_X[c]^α), high-magnitude channels should
        // have strictly larger scales than low-magnitude channels.
        let high_mag_scale = scales[0]; // s_X = 100.0
        let low_mag_scale = scales[1]; // s_X = 0.5
        assert!(
            high_mag_scale > low_mag_scale,
            "high-mag channel should get larger scale: {high_mag_scale} vs {low_mag_scale}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Grid search selects scales that improve MSE.
    // -----------------------------------------------------------------------

    #[test]
    fn grid_search_improves_mse() {
        // Construct weights where non-uniform scaling clearly helps.
        let out = 8;
        let inp = 128;
        let mut weights = vec![0.0_f32; out * inp];
        for row in 0..out {
            for col in 0..inp {
                // First 8 columns have outlier values, rest are small.
                weights[row * inp + col] = if col < 8 {
                    100.0 - (col as f32) * 10.0
                } else {
                    0.1 * col as f32
                };
            }
        }
        let original = weights.clone();

        // High magnitudes on the outlier columns.
        let magnitudes: Vec<f32> = (0..inp).map(|c| if c < 8 { 90.0 } else { 1.0 }).collect();

        // AWQ with grid search.
        let mut mags = HashMap::new();
        mags.insert("w".to_string(), magnitudes);

        let mut prog = make_program("w", &weights, vec![out, inp]);
        let pass = AwqQuantizePass::new(4, inp, mags).with_grid_steps(50);
        pass.run(&mut prog).unwrap();
        let awq_mse = reconstruction_mse(&original, &prog);

        // Compare with plain MinMax (no calibration data).
        let mut plain_prog = make_program("w", &weights, vec![out, inp]);
        let plain_pass = AwqQuantizePass::new(4, inp, HashMap::new());
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
        let out = 8;
        let inp = 128;
        let values: Vec<f32> = (0..(out * inp)).map(|i| i as f32 * 0.5).collect();
        let mut prog = make_program("weight", &values, vec![out, inp]);

        // Empty magnitudes → no calibration data.
        let pass = AwqQuantizePass::new(4, inp, HashMap::new());
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
        let out = 8;
        let inp = 128;
        let weights: Vec<f32> = (0..(out * inp)).map(|i| (i as f32) * 0.3 - 4.0).collect();
        let magnitudes: Vec<f32> = (0..inp).map(|c| if c < 32 { 10.0 } else { 0.5 }).collect();

        let mut mags = HashMap::new();
        mags.insert("w".to_string(), magnitudes);

        let mut prog = make_program("w", &weights, vec![out, inp]);
        let pass = AwqQuantizePass::new(8, inp, mags);
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
    // Test: 1-D tensor is left untouched by the pass.
    // -----------------------------------------------------------------------

    #[test]
    fn one_dim_tensor_skipped() {
        let values: Vec<f32> = (0..2048).map(|i| i as f32 * 0.01).collect();
        let mut prog = make_program("bias", &values, vec![2048]);

        let mut mags = HashMap::new();
        mags.insert("bias".to_string(), vec![5.0; 2048]);

        let pass = AwqQuantizePass::new(4, 128, mags);
        pass.run(&mut prog).unwrap();

        // 1-D tensors (rank < 2) are skipped — op stays "const".
        let op = &prog.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "const");
    }
}
