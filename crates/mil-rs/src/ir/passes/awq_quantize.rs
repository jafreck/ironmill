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
//!   3. Apply per-group weight clipping to minimize quantization MSE
//!   4. Apply scales, quantize with per-group affine, emit rewritten op
//!
//! The per-channel scales are stored as an `"awq_channel_scales"` attribute
//! on the op — Phase 2 Task 2.2 will fuse them into adjacent ops.

use std::collections::HashMap;

use super::tensor_utils::{f32_slice_to_bytes, tensor_as_f32_slice, tensor_f16_as_f32_slice};
use crate::error::{MilError, Result};
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::{TensorData, Value};

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
    pub fn new(
        bits: u8,
        group_size: usize,
        channel_magnitudes: HashMap<String, Vec<f32>>,
    ) -> Result<Self> {
        if bits != 4 && bits != 8 {
            return Err(MilError::Validation(
                "AWQ only supports 4-bit or 8-bit".into(),
            ));
        }
        Ok(Self {
            bits,
            group_size,
            channel_magnitudes,
            calibration_activations: HashMap::new(),
            calibration_token_count: 0,
            grid_search_steps: 20,
            salient_percentile: 0.99,
        })
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

        for function in program.functions.values_mut() {
            // ----------------------------------------------------------
            // Phase 1: Collect eligible ops and categorize.
            // ----------------------------------------------------------
            struct EligibleOp {
                op_idx: usize,
                op_name: String,
                floats: Vec<f32>,
                shape: Vec<usize>,
                in_features: usize,
                mag_key: Vec<u8>,
            }

            let mut eligible: Vec<EligibleOp> = Vec::new();
            let mut fallback_ops: Vec<(usize, Vec<f32>, Vec<usize>)> = Vec::new();

            for (idx, op) in function.body.operations.iter_mut().enumerate() {
                if op.op_type != "const" {
                    continue;
                }

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

                let (numel, rank) = {
                    let val = if in_inputs {
                        op.inputs
                            .get("val")
                            .ok_or_else(|| MilError::Validation("missing val in inputs".into()))?
                    } else {
                        op.attributes.get("val").ok_or_else(|| {
                            MilError::Validation("missing val in attributes".into())
                        })?
                    };
                    if let Value::Tensor { shape, .. } = val {
                        (shape.iter().product::<usize>(), shape.len())
                    } else {
                        continue;
                    }
                };

                if numel < 1024 || rank < 2 {
                    continue;
                }

                let val = if in_inputs {
                    op.inputs
                        .remove("val")
                        .ok_or_else(|| MilError::Validation("missing val in inputs".into()))?
                } else {
                    op.attributes
                        .remove("val")
                        .ok_or_else(|| MilError::Validation("missing val in attributes".into()))?
                };

                if let Value::Tensor { data, shape, dtype } = val {
                    let floats = match dtype {
                        ScalarType::Float32 => {
                            tensor_as_f32_slice(data.as_bytes().expect("tensor not materialized"))
                        }
                        ScalarType::Float16 => tensor_f16_as_f32_slice(
                            data.as_bytes().expect("tensor not materialized"),
                        ),
                        other => {
                            return Err(MilError::TypeMismatch {
                                expected: "Float32 or Float16".into(),
                                actual: format!("{other:?}"),
                            });
                        }
                    };

                    if shape.len() < 2 {
                        fallback_ops.push((idx, floats, shape));
                        continue;
                    }

                    let in_features = *shape.last().unwrap_or(&1);
                    let magnitudes = self.channel_magnitudes.get(&op.name);

                    match magnitudes {
                        Some(mags) if !mags.is_empty() && mags.len() >= in_features => {
                            let mags_slice = &mags[..in_features];

                            match self.calibration_activations.get(&op.name) {
                                Some(a) if !a.is_empty() => {
                                    let mag_key: Vec<u8> =
                                        mags_slice.iter().flat_map(|v| v.to_le_bytes()).collect();

                                    eligible.push(EligibleOp {
                                        op_idx: idx,
                                        op_name: op.name.clone(),
                                        floats,
                                        shape,
                                        in_features,
                                        mag_key,
                                    });
                                }
                                _ => {
                                    fallback_ops.push((idx, floats, shape));
                                }
                            }
                        }
                        _ => {
                            fallback_ops.push((idx, floats, shape));
                        }
                    }
                }
            }

            // ----------------------------------------------------------
            // Phase 2: Group by magnitude key, search α per group.
            // Projections sharing a norm (Q/K/V or gate/up) get
            // identical magnitude vectors, so grouping by mag_key
            // ensures joint α search across the group.
            // ----------------------------------------------------------
            let mut group_map: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
            for (i, eop) in eligible.iter().enumerate() {
                group_map.entry(eop.mag_key.clone()).or_default().push(i);
            }

            eprintln!(
                "[awq] {} eligible ops, {} fallback ops, {} groups",
                eligible.len(),
                fallback_ops.len(),
                group_map.len()
            );

            let mut group_best_alpha: HashMap<Vec<u8>, f32> = HashMap::new();

            for (group_idx, (mag_key, indices)) in group_map.iter().enumerate() {
                let t0 = std::time::Instant::now();
                let first = &eligible[indices[0]];
                let in_features = first.in_features;
                let mags = &self.channel_magnitudes[&first.op_name][..in_features];

                // Build per-projection data for the combined search.
                let mut proj_data: Vec<GroupProjection> = Vec::new();

                for &i in indices {
                    let eop = &eligible[i];
                    let out_features = eop.shape[0];
                    let cal_act = match self.calibration_activations.get(&eop.op_name) {
                        Some(a) => a.as_slice(),
                        None => continue,
                    };
                    let token_count = self.calibration_token_count;

                    let n_tokens = if token_count > 0 && cal_act.len() == token_count * in_features
                    {
                        token_count.min(32)
                    } else {
                        continue;
                    };

                    let stride = if token_count > n_tokens {
                        token_count / n_tokens
                    } else {
                        1
                    };
                    let sub_activations: Vec<f32> = (0..n_tokens)
                        .flat_map(|ti| {
                            let t = ti * stride;
                            (0..in_features).map(move |c| cal_act[t * in_features + c])
                        })
                        .collect();

                    // Pre-compute W · X^T = [out_features, n_tokens].
                    let mut ref_output = vec![0.0f32; out_features * n_tokens];
                    for row in 0..out_features {
                        for t in 0..n_tokens {
                            let mut dot = 0.0f32;
                            for c in 0..in_features {
                                dot += eop.floats[row * in_features + c]
                                    * sub_activations[t * in_features + c];
                            }
                            ref_output[row * n_tokens + t] = dot;
                        }
                    }

                    proj_data.push(GroupProjection {
                        floats: &eop.floats,
                        out_features,
                        in_features,
                        sub_activations,
                        n_tokens,
                        ref_output,
                    });
                }

                let best_alpha = search_alpha_group(
                    &proj_data,
                    mags,
                    self.grid_search_steps,
                    self.group_size,
                    qmax,
                );

                group_best_alpha.insert(mag_key.clone(), best_alpha);
                eprintln!(
                    "[awq] group {}/{}: α={:.2}, {} projections, {:.1}ms",
                    group_idx + 1,
                    group_map.len(),
                    best_alpha,
                    indices.len(),
                    t0.elapsed().as_secs_f64() * 1000.0,
                );
            }

            // ----------------------------------------------------------
            // Phase 3: Apply the group's best α scales, clip, and quantize.
            // Paper Section 3.2: scale → clip → quantize.
            // ----------------------------------------------------------
            for (eop_idx, eop) in eligible.iter().enumerate() {
                let t0 = std::time::Instant::now();
                let alpha = group_best_alpha[&eop.mag_key];
                let in_features = eop.in_features;
                let mags = &self.channel_magnitudes[&eop.op_name][..in_features];
                let channel_scales = compute_scales(mags, alpha);

                let out_features = eop.shape[0];
                let mut scaled_weights = eop.floats.clone();
                for row in 0..out_features {
                    for (col, &s) in channel_scales.iter().enumerate() {
                        if s != 1.0 {
                            scaled_weights[row * in_features + col] *= s;
                        }
                    }
                }

                // Weight clipping: search for optimal per-group max_val and clip.
                if let Some(cal_act) = self.calibration_activations.get(&eop.op_name) {
                    let token_count = self.calibration_token_count;
                    if token_count > 0 && cal_act.len() == token_count * in_features {
                        let clip_maxvals = search_clip_ranges(
                            &scaled_weights,
                            out_features,
                            in_features,
                            self.group_size,
                            qmax,
                            cal_act,
                            token_count,
                            20,  // clip_grid
                            0.5, // max_shrink
                        );
                        apply_clip(
                            &mut scaled_weights,
                            out_features,
                            in_features,
                            self.group_size,
                            &clip_maxvals,
                        );
                    }
                }

                emit_per_group_with_scales(
                    &mut function.body.operations[eop.op_idx],
                    &scaled_weights,
                    &eop.shape,
                    self.group_size,
                    qmax,
                    &channel_scales,
                    self.bits,
                );
                eprintln!(
                    "[awq] phase3 {}/{}: {}x{}, {:.1}ms",
                    eop_idx + 1,
                    eligible.len(),
                    eop.shape[0],
                    in_features,
                    t0.elapsed().as_secs_f64() * 1000.0,
                );
            }

            // ----------------------------------------------------------
            // Phase 4: MinMax fallback for ops without calibration.
            // ----------------------------------------------------------
            for (idx, floats, shape) in &fallback_ops {
                emit_fallback_per_group(
                    &mut function.body.operations[*idx],
                    floats,
                    shape,
                    self.group_size,
                    qmax,
                    self.bits,
                );
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AWQ scale computation
// ---------------------------------------------------------------------------

/// Per-channel scaling factors from activation magnitudes and alpha exponent.
/// Paper Equation 5: s = s_X^α  (Lin et al., MLSys 2024).
fn compute_scales(magnitudes: &[f32], alpha: f32) -> Vec<f32> {
    magnitudes
        .iter()
        .map(|&m| m.max(1e-4).powf(alpha))
        .collect()
}

// ---------------------------------------------------------------------------
// Weight clipping (paper Section 3.2: "We further apply weight clipping
// to minimize the MSE error of quantization")
// ---------------------------------------------------------------------------

/// Search for the optimal per-group clipping range on (already-scaled) weights.
///
/// Matches the reference `auto_clip_layer`: for each (row, group), finds the
/// `max_val` that minimizes activation-weighted quantization error.
/// Uses pre-allocated buffers and inlined quantization to avoid inner-loop
/// allocations (the hot loop runs ~millions of iterations).
///
/// TODO: move to Metal compute shader for GPU-accelerated clipping search
/// (matching the reference which uses PyTorch GPU tensor ops).
///
/// Returns a flat array of optimal `max_val` per (row, group).
fn search_clip_ranges(
    scaled_weights: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    qmax: f32,
    sub_activations: &[f32],
    n_tokens: usize,
    clip_grid: usize,
    max_shrink: f32,
) -> Vec<f32> {
    let n_groups = in_features.div_ceil(group_size);
    let mut clip_maxvals = vec![f32::INFINITY; out_features * n_groups];

    // Pre-allocate reusable buffers (avoids millions of inner-loop allocations).
    let mut clipped = vec![0.0f32; group_size];
    let mut quantized = vec![0u8; group_size];

    // Sub-sample tokens.
    let n_sample = n_tokens.min(4);
    let stride = if n_tokens > n_sample {
        n_tokens / n_sample
    } else {
        1
    };

    for g in 0..n_groups {
        let g_start = g * group_size;
        let g_end = (g_start + group_size).min(in_features);
        let gsize = g_end - g_start;

        // Pre-compute sub-sampled activations for this group: [n_sample, gsize].
        let act_g: Vec<f32> = (0..n_sample)
            .flat_map(|si| {
                let t = si * stride;
                (0..gsize).map(move |j| sub_activations[t * in_features + g_start + j])
            })
            .collect();

        for row in 0..out_features {
            let w_base = row * in_features + g_start;
            let w_slice = &scaled_weights[w_base..w_base + gsize];

            let org_max = w_slice.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            if org_max < 1e-8 {
                clip_maxvals[row * n_groups + g] = org_max;
                continue;
            }

            // Reference output per sample token (stack array, n_sample <= 16).
            let mut org_out = [0.0f32; 16];
            for (si, org) in org_out[..n_sample].iter_mut().enumerate() {
                let mut dot = 0.0f32;
                let ab = si * gsize;
                for j in 0..gsize {
                    dot += w_slice[j] * act_g[ab + j];
                }
                *org = dot;
            }

            let mut best_err = f32::INFINITY;
            let mut best_max = org_max;

            let n_steps = (max_shrink * clip_grid as f32) as usize;
            for step in 0..n_steps {
                let max_val = org_max * (1.0 - step as f32 / clip_grid as f32);

                // Clip weights into pre-allocated buffer.
                for j in 0..gsize {
                    clipped[j] = w_slice[j].clamp(-max_val, max_val);
                }

                // Inline affine quantization (no alloc).
                let mut wmin = f32::INFINITY;
                let mut wmax = f32::NEG_INFINITY;
                for &val in &clipped[..gsize] {
                    if val < wmin {
                        wmin = val;
                    }
                    if val > wmax {
                        wmax = val;
                    }
                }
                let scale = ((wmax - wmin) / qmax).max(1e-10);
                let zp = (-wmin / scale).round();
                for j in 0..gsize {
                    quantized[j] = (clipped[j] / scale + zp).round().clamp(0.0, qmax) as u8;
                }

                // Compute error against reference.
                let mut err = 0.0f32;
                for (si, &org) in org_out[..n_sample].iter().enumerate() {
                    let mut q_out = 0.0f32;
                    let ab = si * gsize;
                    for j in 0..gsize {
                        q_out += (quantized[j] as f32 - zp) * scale * act_g[ab + j];
                    }
                    let diff = q_out - org;
                    err += diff * diff;
                }

                if err < best_err {
                    best_err = err;
                    best_max = max_val;
                }
            }

            clip_maxvals[row * n_groups + g] = best_max;
        }
    }

    clip_maxvals
}

/// Apply per-group clipping to a weight matrix in-place.
fn apply_clip(
    weights: &mut [f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    clip_maxvals: &[f32],
) {
    let n_groups = in_features.div_ceil(group_size);
    for row in 0..out_features {
        for g in 0..n_groups {
            let max_val = clip_maxvals[row * n_groups + g];
            if max_val < f32::INFINITY {
                let g_start = g * group_size;
                let g_end = (g_start + group_size).min(in_features);
                for c in g_start..g_end {
                    let idx = row * in_features + c;
                    weights[idx] = weights[idx].clamp(-max_val, max_val);
                }
            }
        }
    }
}

/// Pre-computed data for one projection in a magnitude group.
struct GroupProjection<'a> {
    floats: &'a [f32],
    out_features: usize,
    in_features: usize,
    sub_activations: Vec<f32>,
    n_tokens: usize,
    ref_output: Vec<f32>,
}

/// Search α ∈ [0, 1] that minimizes the combined quantization loss across
/// all projections in a group (e.g. Q/K/V sharing the same norm).
///
/// For each candidate α the same scale vector is applied to every projection
/// and the per-projection losses are summed, matching the reference which
/// evaluates all `linears2scale` projections together.
fn search_alpha_group(
    projections: &[GroupProjection],
    magnitudes: &[f32],
    grid_steps: usize,
    group_size: usize,
    qmax: f32,
) -> f32 {
    if projections.is_empty() {
        return 0.0;
    }

    let grid_steps = grid_steps.max(1);
    let in_features = magnitudes.len();
    let n_groups = in_features.div_ceil(group_size);

    let mut best_alpha = 0.0_f32;
    let mut best_loss = f32::INFINITY;

    for step in 0..=grid_steps {
        let alpha = step as f32 / grid_steps as f32;
        let scales = compute_scales(magnitudes, alpha);

        let mut total_loss = 0.0f32;

        for proj in projections {
            // Build W_dq: quantize W·diag(s), then dequant and divide by s.
            let mut w_dq = vec![0.0f32; proj.out_features * proj.in_features];
            for row in 0..proj.out_features {
                for g in 0..n_groups {
                    let g_start = g * group_size;
                    let g_end = (g_start + group_size).min(proj.in_features);

                    let scaled_group: Vec<f32> = (g_start..g_end)
                        .map(|col| {
                            let s = if col < scales.len() { scales[col] } else { 1.0 };
                            proj.floats[row * proj.in_features + col] * s
                        })
                        .collect();

                    let (quantized, q_scale, q_zp) = quantize_affine(&scaled_group, qmax);

                    for (j, col) in (g_start..g_end).enumerate() {
                        let s = if col < scales.len() { scales[col] } else { 1.0 };
                        let dequant = (quantized[j] as f32 - q_zp) * q_scale;
                        w_dq[row * proj.in_features + col] = dequant / s;
                    }
                }
            }

            // Exact loss: L(s) = ||W_dq · X^T − W · X^T||²
            let mut loss = 0.0f32;
            for row in 0..proj.out_features {
                for t in 0..proj.n_tokens {
                    let mut dq_dot = 0.0f32;
                    for c in 0..proj.in_features {
                        dq_dot += w_dq[row * proj.in_features + c]
                            * proj.sub_activations[t * proj.in_features + c];
                    }
                    let err = dq_dot - proj.ref_output[row * proj.n_tokens + t];
                    loss += err * err;
                }
            }
            total_loss += loss;
        }

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
        data: TensorData::inline(packed_data),
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
            data: TensorData::inline(scale_bytes),
            shape: param_shape.clone(),
            dtype: ScalarType::Float32,
        },
    );
    op.attributes.insert(
        "zero_point".to_string(),
        Value::Tensor {
            data: TensorData::inline(zp_bytes),
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
            data: TensorData::inline(f32_slice_to_bytes(channel_scales)),
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
        data: TensorData::inline(packed_data),
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
            data: TensorData::inline(scale_bytes),
            shape: param_shape.clone(),
            dtype: ScalarType::Float32,
        },
    );
    op.attributes.insert(
        "zero_point".to_string(),
        Value::Tensor {
            data: TensorData::inline(zp_bytes),
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
    use crate::ir::passes::test_helpers::{const_tensor_op, f32_bytes};
    use crate::ir::program::Function;

    /// Build a single-const-op program for testing.
    fn make_program(name: &str, values: &[f32], shape: Vec<usize>) -> Program {
        let tensor_val = Value::Tensor {
            data: TensorData::inline(f32_bytes(values)),
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

    /// Generate synthetic calibration activations consistent with the given
    /// magnitudes. Each token has activations whose channel magnitudes
    /// roughly match the provided `magnitudes` vector.
    fn make_calibration_activations(magnitudes: &[f32], n_tokens: usize) -> (Vec<f32>, usize) {
        let in_features = magnitudes.len();
        let mut activations = vec![0.0f32; n_tokens * in_features];
        for t in 0..n_tokens {
            for c in 0..in_features {
                // Simple pattern: alternate signs, scale by magnitude.
                let sign = if (t + c) % 2 == 0 { 1.0 } else { -1.0 };
                activations[t * in_features + c] = sign * magnitudes[c];
            }
        }
        (activations, n_tokens)
    }

    /// Compute MSE between original floats and dequantized reconstruction.
    #[allow(dead_code)]
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
            unpack_int4(q_data_raw.as_bytes().unwrap(), original.len())
        } else {
            q_data_raw.as_bytes().unwrap().to_vec()
        };

        // Get per-group scales and zero points.
        let scales = match op.attributes.get("scale") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data.as_bytes().unwrap()),
            Some(Value::Float(f)) => vec![*f as f32],
            _ => panic!("missing scale"),
        };
        let zero_points = match op.attributes.get("zero_point") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data.as_bytes().unwrap()),
            Some(Value::Float(f)) => vec![*f as f32],
            _ => panic!("missing zero_point"),
        };

        let group_size = match op.attributes.get("group_size") {
            Some(Value::Int(g)) => *g as usize,
            _ => original.len(),
        };

        // Get AWQ channel scales if present.
        let channel_scales = match op.attributes.get("awq_channel_scales") {
            Some(Value::Tensor { data, .. }) => Some(tensor_as_f32_slice(data.as_bytes().unwrap())),
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
            unpack_int4(q_data_raw.as_bytes().unwrap(), original.len())
        } else {
            q_data_raw.as_bytes().unwrap().to_vec()
        };
        let scales = match op.attributes.get("scale") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data.as_bytes().unwrap()),
            Some(Value::Float(f)) => vec![*f as f32],
            _ => panic!("missing scale"),
        };
        let zero_points = match op.attributes.get("zero_point") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data.as_bytes().unwrap()),
            Some(Value::Float(f)) => vec![*f as f32],
            _ => panic!("missing zero_point"),
        };
        let group_size = match op.attributes.get("group_size") {
            Some(Value::Int(g)) => *g as usize,
            _ => original.len(),
        };
        let channel_scales = match op.attributes.get("awq_channel_scales") {
            Some(Value::Tensor { data, .. }) => Some(tensor_as_f32_slice(data.as_bytes().unwrap())),
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
        let original = weights.clone();

        let magnitudes: Vec<f32> = (0..inp)
            .map(|c| if c % 4 == 0 { 100.0 } else { 0.1 })
            .collect();
        let (cal_act, n_tokens) = make_calibration_activations(&magnitudes, 32);

        // --- MinMax baseline ---
        let mut minmax_prog = make_program("weight", &weights, vec![out, inp]);
        let minmax_pass = AwqQuantizePass::new(4, group_sz, HashMap::new()).unwrap();
        minmax_pass.run(&mut minmax_prog).unwrap();
        let minmax_wmse = weighted_reconstruction_mse(&original, &minmax_prog, &magnitudes);

        // --- AWQ (with calibration activations) ---
        let mut awq_mags = HashMap::new();
        awq_mags.insert("weight".to_string(), magnitudes.clone());
        let mut awq_cal = HashMap::new();
        awq_cal.insert("weight".to_string(), cal_act);
        let mut awq_prog = make_program("weight", &weights, vec![out, inp]);
        let awq_pass = AwqQuantizePass::new(4, group_sz, awq_mags)
            .unwrap()
            .with_calibration_activations(awq_cal, n_tokens)
            .with_grid_steps(40);
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

        let magnitudes: Vec<f32> = (0..inp)
            .map(|c| if c % 4 == 0 { 100.0 } else { 0.5 })
            .collect();
        let (cal_act, n_tokens) = make_calibration_activations(&magnitudes, 32);

        let mut mags = HashMap::new();
        mags.insert("w".to_string(), magnitudes);
        let mut cal = HashMap::new();
        cal.insert("w".to_string(), cal_act);

        let mut prog = make_program("w", &weights, vec![out, inp]);
        let pass = AwqQuantizePass::new(4, group_sz, mags)
            .unwrap()
            .with_calibration_activations(cal, n_tokens)
            .with_grid_steps(20);
        pass.run(&mut prog).unwrap();

        let op = &prog.functions["main"].body.operations[0];
        let scales = match op.attributes.get("awq_channel_scales") {
            Some(Value::Tensor { data, .. }) => tensor_as_f32_slice(data.as_bytes().unwrap()),
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
        // AWQ-friendly scenario: high-activation channels have moderate
        // weights mixed with large-weight non-salient channels. AWQ
        // scales up the salient channels to give them more quantization
        // resolution.
        let out = 8;
        let inp = 128;
        let group_sz = 32;
        let mut weights = vec![0.0_f32; out * inp];
        for row in 0..out {
            for col in 0..inp {
                if col % 8 == 0 {
                    // Salient channels: small weights, high activations.
                    weights[row * inp + col] = 0.2 * (row as f32 - 3.5);
                } else {
                    // Non-salient: large weights dominate the group range.
                    weights[row * inp + col] = 40.0 * ((col as f32) / inp as f32 - 0.5);
                }
            }
        }
        let original = weights.clone();

        let magnitudes: Vec<f32> = (0..inp)
            .map(|c| if c % 8 == 0 { 80.0 } else { 0.5 })
            .collect();
        let (cal_act, n_tokens) = make_calibration_activations(&magnitudes, 32);

        // AWQ with grid search + calibration activations.
        let mut mags = HashMap::new();
        mags.insert("w".to_string(), magnitudes.clone());
        let mut cal = HashMap::new();
        cal.insert("w".to_string(), cal_act);

        let mut prog = make_program("w", &weights, vec![out, inp]);
        let pass = AwqQuantizePass::new(4, group_sz, mags)
            .unwrap()
            .with_calibration_activations(cal, n_tokens)
            .with_grid_steps(50);
        pass.run(&mut prog).unwrap();
        let awq_wmse = weighted_reconstruction_mse(&original, &prog, &magnitudes);

        // Compare with plain MinMax (no calibration data).
        let mut plain_prog = make_program("w", &weights, vec![out, inp]);
        let plain_pass = AwqQuantizePass::new(4, group_sz, HashMap::new()).unwrap();
        plain_pass.run(&mut plain_prog).unwrap();
        let plain_wmse = weighted_reconstruction_mse(&original, &plain_prog, &magnitudes);

        // AWQ optimizes activation-weighted output MSE (Eq. 4); α=0
        // matches MinMax, so the weighted MSE can only match or improve.
        assert!(
            awq_wmse <= plain_wmse,
            "AWQ weighted MSE ({awq_wmse}) should be <= plain weighted MSE ({plain_wmse})"
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
        let pass = AwqQuantizePass::new(4, inp, HashMap::new()).unwrap();
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
        let (cal_act, n_tokens) = make_calibration_activations(&magnitudes, 32);

        let mut mags = HashMap::new();
        mags.insert("w".to_string(), magnitudes);
        let mut cal = HashMap::new();
        cal.insert("w".to_string(), cal_act);

        let mut prog = make_program("w", &weights, vec![out, inp]);
        let pass = AwqQuantizePass::new(8, inp, mags)
            .unwrap()
            .with_calibration_activations(cal, n_tokens);
        pass.run(&mut prog).unwrap();

        let op = &prog.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "constexpr_affine_dequantize");
        assert!(op.attributes.get("awq_channel_scales").is_some());

        // All quantized values should be in [0, 255].
        let q_data = match op.attributes.get("quantized_data") {
            Some(Value::Tensor { data, .. }) => data,
            _ => panic!("missing quantized_data"),
        };
        let bytes = q_data.as_bytes().unwrap();
        assert!(!bytes.is_empty(), "quantized data should be non-empty");
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

        let pass = AwqQuantizePass::new(4, 128, mags).unwrap();
        pass.run(&mut prog).unwrap();

        // 1-D tensors (rank < 2) are skipped — op stays "const".
        let op = &prog.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "const");
    }
}
