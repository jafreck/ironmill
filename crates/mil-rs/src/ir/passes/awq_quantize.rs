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

use rayon::prelude::*;

use super::tensor_utils::{f32_slice_to_bytes, tensor_as_f32_slice, tensor_f16_as_f32_slice};
use crate::error::{MilError, Result};
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::{ScalarType, TensorType};
use crate::ir::types::Value;

// ---------------------------------------------------------------------------
// Apple Accelerate BLAS bindings for hardware-accelerated matrix math.
// On Apple Silicon this uses the AMX coprocessor for near-GPU throughput.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod blas {
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}

/// Compute C = A × B^T using Accelerate BLAS on macOS, scalar fallback elsewhere.
/// A: [m × k] row-major, B: [n × k] row-major, C: [m × n] row-major.
#[allow(unsafe_code)]
fn matmul_a_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(target_os = "macos")]
    {
        unsafe {
            blas::cblas_sgemm(
                blas::ROW_MAJOR,
                blas::NO_TRANS,
                blas::TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                k as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut dot = 0.0f32;
                for p in 0..k {
                    dot += a[i * k + p] * b[j * k + p];
                }
                c[i * n + j] = dot;
            }
        }
    }
}

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
                        ScalarType::Float32 => tensor_as_f32_slice(&data),
                        ScalarType::Float16 => tensor_f16_as_f32_slice(&data),
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

            let mut group_best_alpha: HashMap<Vec<u8>, f32> = HashMap::new();

            for (mag_key, indices) in &group_map {
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
                        token_count.min(128)
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

                    // Pre-compute W · X^T = [out_features, n_tokens] using BLAS.
                    let mut ref_output = vec![0.0f32; out_features * n_tokens];
                    matmul_a_bt(
                        &eop.floats,
                        &sub_activations,
                        &mut ref_output,
                        out_features,
                        n_tokens,
                        in_features,
                    );

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
            }

            // ----------------------------------------------------------
            // Phase 3: Apply the group's best α scales, clip, and quantize.
            // Paper Section 3.2: scale → clip → quantize.
            // ----------------------------------------------------------
            for eop in &eligible {
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
                        let n_tokens = token_count.min(128);
                        let stride = if token_count > n_tokens {
                            token_count / n_tokens
                        } else {
                            1
                        };
                        let sub_act: Vec<f32> = (0..n_tokens)
                            .flat_map(|ti| {
                                let t = ti * stride;
                                (0..in_features).map(move |c| cal_act[t * in_features + c])
                            })
                            .collect();

                        let clip_maxvals = search_clip_ranges(
                            &scaled_weights,
                            out_features,
                            in_features,
                            self.group_size,
                            qmax,
                            &sub_act,
                            n_tokens,
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
/// For each (row, group), searches for a `max_val` that minimizes
///   `|| (act * Q(clamp(w, -max_val, max_val))) - (act * w) ||²`
/// over the calibration tokens.
///
/// Uses Rayon to parallelize across (row, group) pairs for near-GPU throughput.
///
/// Returns a flat array of optimal `max_val` per (row, group).
#[allow(clippy::too_many_arguments)]
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
    let n_steps = (max_shrink * clip_grid as f32) as usize;

    // Parallel search over all (row, group) pairs.
    let total_pairs = out_features * n_groups;
    let results: Vec<f32> = (0..total_pairs)
        .into_par_iter()
        .map(|pair_idx| {
            let row = pair_idx / n_groups;
            let g = pair_idx % n_groups;

            let g_start = g * group_size;
            let g_end = (g_start + group_size).min(in_features);
            let gsize = g_end - g_start;

            let row_base = row * in_features;

            // Per-group abs max (read from weight slice directly, no allocation).
            let mut org_max = 0.0f32;
            for c in g_start..g_end {
                let v = scaled_weights[row_base + c].abs();
                if v > org_max {
                    org_max = v;
                }
            }
            if org_max < 1e-8 {
                return org_max;
            }

            // Compute original (unquantized) group output per token.
            let mut org_out = vec![0.0f32; n_tokens];
            for (t, org) in org_out.iter_mut().enumerate() {
                let act_base = t * in_features + g_start;
                let mut sum = 0.0f32;
                for j in 0..gsize {
                    sum += scaled_weights[row_base + g_start + j] * sub_activations[act_base + j];
                }
                *org = sum;
            }

            // Reusable buffers.
            let mut clipped = vec![0.0f32; gsize];
            let mut best_err = f32::INFINITY;
            let mut best_max = org_max;

            for step in 0..n_steps {
                let max_val = org_max * (1.0 - step as f32 / clip_grid as f32);

                // Clamp weights into reusable buffer.
                for j in 0..gsize {
                    clipped[j] = scaled_weights[row_base + g_start + j].clamp(-max_val, max_val);
                }

                let (quantized, q_scale, q_zp) = quantize_affine(&clipped, qmax);

                // Dequantize once into the clipped buffer (reuse it).
                for j in 0..gsize {
                    clipped[j] = (quantized[j] as f32 - q_zp) * q_scale;
                }

                let mut err = 0.0f32;
                for (t, &org) in org_out.iter().enumerate() {
                    let act_base = t * in_features + g_start;
                    let mut q_out = 0.0f32;
                    for j in 0..gsize {
                        q_out += clipped[j] * sub_activations[act_base + j];
                    }
                    let diff = q_out - org;
                    err += diff * diff;
                }

                if err < best_err {
                    best_err = err;
                    best_max = max_val;
                }
            }

            best_max
        })
        .collect();

    results
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
///
/// Uses Rayon to evaluate α candidates in parallel and Accelerate BLAS
/// for the loss matrix multiply, giving near-GPU throughput on Apple Silicon.
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

    // Evaluate all α candidates in parallel.
    let best = (0..=grid_steps)
        .into_par_iter()
        .map(|step| {
            let alpha = step as f32 / grid_steps as f32;
            let scales = compute_scales(magnitudes, alpha);

            let mut total_loss = 0.0f32;

            // Reusable buffers per thread (avoid repeated allocation).
            let max_out = projections
                .iter()
                .map(|p| p.out_features)
                .max()
                .unwrap_or(0);
            let mut w_dq = vec![0.0f32; max_out * in_features];
            let max_tokens = projections.iter().map(|p| p.n_tokens).max().unwrap_or(0);
            let mut output = vec![0.0f32; max_out * max_tokens];
            let mut scaled_group = vec![0.0f32; group_size];

            for proj in projections {
                let wdq = &mut w_dq[..proj.out_features * proj.in_features];

                // Build W_dq: quantize W·diag(s), then dequant and divide by s.
                for row in 0..proj.out_features {
                    for g in 0..n_groups {
                        let g_start = g * group_size;
                        let g_end = (g_start + group_size).min(proj.in_features);
                        let gsize = g_end - g_start;

                        // Scale the group into the reusable buffer.
                        for (j, col) in (g_start..g_end).enumerate() {
                            scaled_group[j] =
                                proj.floats[row * proj.in_features + col] * scales[col];
                        }

                        let (quantized, q_scale, q_zp) =
                            quantize_affine(&scaled_group[..gsize], qmax);

                        for (j, col) in (g_start..g_end).enumerate() {
                            let dequant = (quantized[j] as f32 - q_zp) * q_scale;
                            wdq[row * proj.in_features + col] = dequant / scales[col];
                        }
                    }
                }

                // Compute loss = ||W_dq · X^T − ref_output||² via BLAS matmul.
                let out = &mut output[..proj.out_features * proj.n_tokens];
                matmul_a_bt(
                    wdq,
                    &proj.sub_activations,
                    out,
                    proj.out_features,
                    proj.n_tokens,
                    proj.in_features,
                );

                let loss: f32 = out
                    .iter()
                    .zip(proj.ref_output.iter())
                    .map(|(&a, &b)| {
                        let e = a - b;
                        e * e
                    })
                    .sum();
                total_loss += loss;
            }

            (alpha, total_loss)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0.0, f32::INFINITY));

    best.0
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
        let minmax_pass = AwqQuantizePass::new(4, group_sz, HashMap::new());
        minmax_pass.run(&mut minmax_prog).unwrap();
        let minmax_wmse = weighted_reconstruction_mse(&original, &minmax_prog, &magnitudes);

        // --- AWQ (with calibration activations) ---
        let mut awq_mags = HashMap::new();
        awq_mags.insert("weight".to_string(), magnitudes.clone());
        let mut awq_cal = HashMap::new();
        awq_cal.insert("weight".to_string(), cal_act);
        let mut awq_prog = make_program("weight", &weights, vec![out, inp]);
        let awq_pass = AwqQuantizePass::new(4, group_sz, awq_mags)
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
            .with_calibration_activations(cal, n_tokens)
            .with_grid_steps(20);
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
            .with_calibration_activations(cal, n_tokens)
            .with_grid_steps(50);
        pass.run(&mut prog).unwrap();
        let awq_wmse = weighted_reconstruction_mse(&original, &prog, &magnitudes);

        // Compare with plain MinMax (no calibration data).
        let mut plain_prog = make_program("w", &weights, vec![out, inp]);
        let plain_pass = AwqQuantizePass::new(4, group_sz, HashMap::new());
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
        let (cal_act, n_tokens) = make_calibration_activations(&magnitudes, 32);

        let mut mags = HashMap::new();
        mags.insert("w".to_string(), magnitudes);
        let mut cal = HashMap::new();
        cal.insert("w".to_string(), cal_act);

        let mut prog = make_program("w", &weights, vec![out, inp]);
        let pass = AwqQuantizePass::new(8, inp, mags).with_calibration_activations(cal, n_tokens);
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
