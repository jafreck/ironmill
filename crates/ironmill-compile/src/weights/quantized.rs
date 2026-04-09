//! JIT quantized weight provider.
//!
//! Wraps any [`WeightProvider`] and applies weight quantization lazily on
//! each `tensor()` call. Supports D2Quant dual-scale (2/3-bit) and INT4
//! affine per-group quantization. Eligible weight matrices (2D, ≥64×64,
//! ≥4096 elements) are returned with the appropriate [`QuantizationInfo`];
//! all other tensors pass through unchanged.
//!
//! This bypasses the MIL IR pipeline entirely — no compilation step, no
//! graph construction — making it suitable for both Metal GPU JIT loading
//! and as a pre-quantized input to CoreML compilation.

use std::collections::HashMap;
use std::sync::Mutex;

use mil_rs::error::MilError;
use mil_rs::ir::ScalarType;
use mil_rs::ir::passes::affine_quantize::quantize_affine;
use mil_rs::ir::passes::d2quant::dual_scale::{
    dual_scale_quantize, pack_2bit, pack_3bit, pack_mask,
};
use mil_rs::weights::{ModelConfig, QuantizationInfo, WeightProvider, WeightTensor};
use rayon::prelude::*;

/// Configuration for JIT D2Quant quantization.
#[derive(Debug, Clone)]
pub struct D2QuantConfig {
    /// Bit-width: 2 or 3.
    pub bits: u8,
    /// Number of weights per quantization group (typically 128).
    pub group_size: usize,
    /// Outlier percentile threshold (e.g. 0.99 → top 1% are outliers).
    pub outlier_threshold: f32,
}

impl D2QuantConfig {
    /// Default 3-bit config: group_size=128, outlier_threshold=0.99.
    pub fn three_bit() -> Self {
        Self {
            bits: 3,
            group_size: 128,
            outlier_threshold: 0.99,
        }
    }

    /// Default 2-bit config: group_size=128, outlier_threshold=0.99.
    pub fn two_bit() -> Self {
        Self {
            bits: 2,
            group_size: 128,
            outlier_threshold: 0.99,
        }
    }
}

/// Configuration for JIT INT4 affine per-group quantization.
#[derive(Debug, Clone)]
pub struct AffineQuantConfig {
    /// Number of weights per group along the reduction axis.
    pub group_size: usize,
    /// Per-channel AWQ activation scales, keyed by layer weight name
    /// (e.g., "l0_q_proj_weight" → [hidden_size] f32 magnitudes).
    /// When present, weights are pre-scaled by 1/awq_scales before quantization,
    /// and the runtime kernel compensates by dividing activations.
    pub awq_magnitudes: Option<std::collections::HashMap<String, Vec<f32>>>,
    /// Raw calibration activations for alpha grid search, keyed by AWQ key
    /// (e.g., "l0_q_proj_weight" → flattened [tokens, features] f32 array).
    /// When present, enables per-tensor alpha optimization instead of fixed 0.5.
    pub awq_activations: Option<std::collections::HashMap<String, Vec<f32>>>,
    /// Number of calibration tokens in each activation snapshot.
    pub awq_token_count: Option<usize>,
    /// Per-layer Hessian data for GPTQ quantization, keyed by AWQ key format.
    /// Each entry is (xtx_flat, n_features, sample_count).
    /// When present with gptq feature enabled, uses GPTQ instead of round-to-nearest.
    #[cfg(feature = "gptq")]
    pub hessian_data: Option<std::collections::HashMap<String, (Vec<f32>, usize, usize)>>,
    /// GPTQ block size for column-block processing (default: 128).
    #[cfg(feature = "gptq")]
    pub gptq_block_size: usize,
    /// GPTQ Hessian dampening factor (default: 0.01).
    #[cfg(feature = "gptq")]
    pub gptq_dampening: f64,
    /// Precomputed per-tensor AWQ configs from block-level calibration.
    /// When present, bypasses both alpha search and clip search at load time.
    /// Keyed by AWQ key format (e.g. "l0_q_proj_weight").
    pub awq_block_config: Option<std::collections::HashMap<String, AwqTensorConfig>>,
    /// Layer indices to quantize at INT8 instead of INT4.
    /// Typically the first and last 1-2 layers, which handle the
    /// embedding→hidden and hidden→logit transformations.
    pub sensitive_layers: Vec<usize>,
}

/// Precomputed per-tensor AWQ parameters from block-level calibration.
///
/// Produced by the `awq_block_calibrate` tool which runs full transformer
/// block forwards on GPU to find optimal scaling and clipping parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AwqTensorConfig {
    /// Optimal alpha from block-level search.
    pub alpha: f32,
    /// Per-(row, group) clip max values, flattened as `[out_features × n_groups]`.
    /// `None` means no clipping (e.g. for Q/K projections per reference AWQ).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clip_maxvals: Option<Vec<f32>>,
}

impl AffineQuantConfig {
    /// Default INT4 per-group config: group_size=128.
    pub fn int4(group_size: usize) -> Self {
        Self {
            group_size,
            awq_magnitudes: None,
            awq_activations: None,
            awq_token_count: None,
            awq_block_config: None,
            #[cfg(feature = "gptq")]
            hessian_data: None,
            #[cfg(feature = "gptq")]
            gptq_block_size: 128,
            #[cfg(feature = "gptq")]
            gptq_dampening: 0.01,
            sensitive_layers: Vec::new(),
        }
    }

    /// Add AWQ (Activation-aware Weight Quantization) calibration data.
    ///
    /// `magnitudes` are per-channel activation scales keyed by layer weight name.
    /// `activations` are optional raw calibration activations for alpha grid search.
    /// `token_count` is the number of calibration tokens in each activation snapshot.
    pub fn with_awq(
        mut self,
        magnitudes: std::collections::HashMap<String, Vec<f32>>,
        activations: Option<std::collections::HashMap<String, Vec<f32>>>,
        token_count: usize,
    ) -> Self {
        self.awq_magnitudes = Some(magnitudes);
        self.awq_activations = activations;
        self.awq_token_count = Some(token_count);
        self
    }

    /// Add GPTQ (second-order) calibration data.
    ///
    /// `hessian_data` maps AWQ key → (xtx_flat, n_features, sample_count).
    /// `block_size` controls the column-block processing width (default: 128).
    /// `dampening` is the Hessian dampening factor (default: 0.01).
    #[cfg(feature = "gptq")]
    pub fn with_gptq(
        mut self,
        hessian_data: std::collections::HashMap<String, (Vec<f32>, usize, usize)>,
        block_size: usize,
        dampening: f64,
    ) -> Self {
        self.hessian_data = Some(hessian_data);
        self.gptq_block_size = block_size;
        self.gptq_dampening = dampening;
        self
    }

    /// Mark specific layer indices as sensitive (quantize at INT8 instead of INT4).
    ///
    /// Typically the first and last 1-2 layers, which handle the
    /// embedding→hidden and hidden→logit transformations.
    pub fn with_sensitive_layers(mut self, layers: Vec<usize>) -> Self {
        self.sensitive_layers = layers;
        self
    }

    /// Add precomputed AWQ block-level configuration.
    ///
    /// When present, bypasses both alpha search and clip search at load time.
    pub fn with_block_config(
        mut self,
        config: std::collections::HashMap<String, AwqTensorConfig>,
    ) -> Self {
        self.awq_block_config = Some(config);
        self
    }

    /// Compute first/last N sensitive layer indices for a model with `num_layers` layers.
    ///
    /// Helper that builds the sensitive layer list and calls [`with_sensitive_layers`].
    pub fn with_sensitive_bookend(self, num_layers: usize, sensitive_count: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..sensitive_count.min(num_layers) {
            layers.push(i);
        }
        for i in num_layers.saturating_sub(sensitive_count)..num_layers {
            if !layers.contains(&i) {
                layers.push(i);
            }
        }
        self.with_sensitive_layers(layers)
    }
}

impl Default for AffineQuantConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            awq_magnitudes: None,
            awq_activations: None,
            awq_token_count: None,
            awq_block_config: None,
            #[cfg(feature = "gptq")]
            hessian_data: None,
            #[cfg(feature = "gptq")]
            gptq_block_size: 128,
            #[cfg(feature = "gptq")]
            gptq_dampening: 0.01,
            sensitive_layers: Vec::new(),
        }
    }
}

/// Which quantization method to apply.
#[derive(Debug, Clone)]
pub enum QuantMethod {
    /// D2Quant dual-scale 2/3-bit quantization.
    D2Quant(D2QuantConfig),
    /// INT4 affine per-group quantization (unsigned, 0–15).
    AffineInt4(AffineQuantConfig),
}

/// A [`WeightProvider`] wrapper that applies weight quantization on-the-fly.
///
/// Eligible tensors (2D weight matrices ≥64×64 with ≥4096 elements) are
/// quantized and returned with the appropriate [`QuantizationInfo`].
/// All other tensors (norms, biases, embeddings, 1D vectors) pass through
/// unchanged from the inner provider.
pub struct QuantizedWeightProvider<P> {
    inner: P,
    method: QuantMethod,
    /// Cache of alpha search results keyed by AWQ magnitude key.
    /// Projections sharing the same input (e.g. Q/K/V/O from the same norm)
    /// reuse the first search result instead of repeating the expensive search.
    alpha_cache: Mutex<HashMap<u64, f32>>,
}

impl<P: WeightProvider> QuantizedWeightProvider<P> {
    /// Create a D2Quant quantized provider wrapping `inner`.
    pub fn new(inner: P, config: D2QuantConfig) -> Self {
        Self {
            inner,
            method: QuantMethod::D2Quant(config),
            alpha_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Create an INT4 affine quantized provider wrapping `inner`.
    pub fn new_int4(inner: P, config: AffineQuantConfig) -> Self {
        Self {
            inner,
            method: QuantMethod::AffineInt4(config),
            alpha_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Consume this wrapper and return the inner provider.
    pub fn into_inner(self) -> P {
        self.inner
    }
}

/// Check whether a tensor shape is eligible for D2Quant quantization.
fn is_quantizable(shape: &[usize]) -> bool {
    shape.len() == 2 && shape[0] >= 64 && shape[1] >= 64 && shape.iter().product::<usize>() >= 4096
}

/// Map a safetensors weight name to the AWQ calibration key format.
///
/// e.g., "model.layers.5.self_attn.q_proj.weight" → "l5_q_proj_weight"
///       "model.layers.12.mlp.gate_proj.weight" → "l12_gate_proj_weight"
fn tensor_name_to_awq_key(name: &str) -> String {
    // Extract layer index and projection name from dotted path.
    let parts: Vec<&str> = name.split('.').collect();
    // Look for "layers.N.*.proj_name.weight"
    for (i, &p) in parts.iter().enumerate() {
        if p == "layers" && i + 1 < parts.len() {
            if let Ok(layer_idx) = parts[i + 1].parse::<usize>() {
                // Find the projection name (q_proj, k_proj, gate_proj, etc.)
                for j in (i + 2)..parts.len() {
                    if parts[j].ends_with("_proj") {
                        let block = if j > i + 2 && parts[j - 1] == "mlp" {
                            "ffn"
                        } else {
                            "attn"
                        };
                        let _ = block; // block info is in the key via proj name
                        return format!("l{}_{}_weight", layer_idx, parts[j]);
                    }
                }
            }
        }
    }
    // Fallback: use the full name with dots replaced.
    name.replace('.', "_")
}

/// Convert raw tensor bytes to f32 based on dtype.
fn to_f32_vec(data: &[u8], dtype: ScalarType) -> Option<Vec<f32>> {
    match dtype {
        ScalarType::Float32 => {
            if data.len() % 4 != 0 {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        }
        ScalarType::Float16 => {
            if data.len() % 2 != 0 {
                return None;
            }
            Some(
                data.chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
            )
        }
        _ => None,
    }
}

/// Quantize a float tensor using D2Quant dual-scale quantization.
///
/// Returns the quantized tensor as owned bytes with `QuantizationInfo::DualScaleDequantize`.
fn quantize_tensor(
    floats: &[f32],
    shape: &[usize],
    config: &D2QuantConfig,
) -> crate::error::Result<(Vec<u8>, QuantizationInfo)> {
    let last_dim = shape[shape.len() - 1];
    let outer_count: usize = if shape.len() > 1 {
        shape[..shape.len() - 1].iter().product()
    } else {
        1
    };
    let n_groups_per_row = last_dim.div_ceil(config.group_size);
    let _total_groups = outer_count * n_groups_per_row;

    let mut all_quantized_packed: Vec<u8> = Vec::new();
    let mut all_normal_scale: Vec<f32> = Vec::new();
    let mut all_normal_zero: Vec<f32> = Vec::new();
    let mut all_outlier_scale: Vec<f32> = Vec::new();
    let mut all_outlier_zero: Vec<f32> = Vec::new();
    let mut all_mask_packed: Vec<u8> = Vec::new();

    for row in 0..outer_count {
        let row_start = row * last_dim;
        for g in 0..n_groups_per_row {
            let g_start = row_start + g * config.group_size;
            let g_end = (g_start + config.group_size).min(row_start + last_dim);
            let group = &floats[g_start..g_end];

            let (quantized, params) =
                dual_scale_quantize(group, config.bits, config.outlier_threshold);

            let packed = match config.bits {
                2 => pack_2bit(&quantized),
                3 => pack_3bit(&quantized),
                _ => {
                    return Err(crate::error::CompileError::UnsupportedQuantization(
                        format!(
                            "D2Quant bit width {} not supported (expected 2 or 3)",
                            config.bits
                        ),
                    ));
                }
            };
            all_quantized_packed.extend_from_slice(&packed);

            all_normal_scale.push(params.normal_scale);
            all_normal_zero.push(params.normal_zero);
            all_outlier_scale.push(params.outlier_scale);
            all_outlier_zero.push(params.outlier_zero);
            all_mask_packed.extend_from_slice(&pack_mask(&params.outlier_mask));
        }
    }

    let f32_bytes =
        |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };

    let quant_info = QuantizationInfo::DualScaleDequantize {
        quantized_data: all_quantized_packed.clone(),
        normal_scale: f32_bytes(&all_normal_scale),
        normal_zero: f32_bytes(&all_normal_zero),
        outlier_scale: f32_bytes(&all_outlier_scale),
        outlier_zero: f32_bytes(&all_outlier_zero),
        outlier_mask: all_mask_packed,
        original_shape: shape.to_vec(),
        bit_width: config.bits,
        group_size: config.group_size,
    };

    // The primary data becomes the packed quantized bytes.
    Ok((all_quantized_packed, quant_info))
}

/// Quantize a float tensor using INT4 affine per-group quantization.
///
/// Each group of `group_size` elements along the last dimension gets its
/// own scale and zero_point. Returns packed INT4 data (2 values per byte)
/// with `QuantizationInfo::AffineDequantize`.
fn quantize_tensor_int4(
    floats: &[f32],
    shape: &[usize],
    config: &AffineQuantConfig,
) -> (Vec<u8>, QuantizationInfo) {
    let last_dim = shape[shape.len() - 1];
    let outer_count: usize = if shape.len() > 1 {
        shape[..shape.len() - 1].iter().product()
    } else {
        1
    };
    let n_groups_per_row = last_dim.div_ceil(config.group_size);

    let qmax: f32 = 15.0; // INT4 unsigned

    let mut all_quantized: Vec<u8> = Vec::new();
    let mut all_scale: Vec<f32> = Vec::new();
    let mut all_zero: Vec<f32> = Vec::new();

    for row in 0..outer_count {
        let row_start = row * last_dim;
        for g in 0..n_groups_per_row {
            let g_start = row_start + g * config.group_size;
            let g_end = (g_start + config.group_size).min(row_start + last_dim);
            let group = &floats[g_start..g_end];

            let (quantized, scale, zero_point) = quantize_affine(group, qmax);
            all_quantized.extend_from_slice(&quantized);
            all_scale.push(scale);
            all_zero.push(zero_point);
        }
    }

    // Pack INT4: 2 values per byte, low nibble first.
    let mut packed = Vec::with_capacity(all_quantized.len().div_ceil(2));
    for chunk in all_quantized.chunks(2) {
        let lo = chunk[0] & 0x0F;
        let hi = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
        packed.push(lo | (hi << 4));
    }

    let f32_bytes =
        |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };

    let quant_info = QuantizationInfo::AffineDequantize {
        scale: f32_bytes(&all_scale),
        zero_point: f32_bytes(&all_zero),
        scale_dtype: ScalarType::Float32,
        zero_point_dtype: ScalarType::Float32,
        axis: Some(shape.len() - 1),
        bit_width: 4,
        group_size: Some(config.group_size),
        awq_scales: None,
        g_idx: None,
    };

    (packed, quant_info)
}

/// Search α ∈ {0.0, 0.1, 0.2, ..., 1.0} that minimizes reconstruction error.
///
/// Uses an activation-weighted per-column MSE approximation:
///   loss(α) ≈ Σ_c (W_dq[c] − W[c])² · Σ_t X[t,c]²
/// which is O(rows × cols) instead of O(rows × tokens × cols).
/// Alpha steps are evaluated in parallel via rayon.
fn search_best_alpha(
    floats: &[f32],
    shape: &[usize],
    magnitudes: &[f32],
    activations: &[f32],
    n_tokens: usize,
    group_size: usize,
) -> f32 {
    let out_features = shape[0];
    let in_features = shape[1];
    let qmax: f32 = 15.0;
    let n_groups = in_features.div_ceil(group_size);

    // 32 tokens is sufficient for alpha selection (matches the MIL IR pass).
    let max_tokens = n_tokens.min(32);

    // Pre-compute per-channel activation power: Σ_t X[t,c]².
    // This lets us compute activation-weighted MSE without the full matmul:
    //   loss ≈ Σ_c (W_dq[c] − W[c])² · act_sq[c]
    let mut act_sq = vec![0.0f32; in_features];
    for t in 0..max_tokens {
        for c in 0..in_features {
            let x = activations[t * in_features + c];
            act_sq[c] += x * x;
        }
    }

    let max_mag = magnitudes.iter().cloned().fold(0.0f32, f32::max).max(1e-10);

    // Sub-sample rows for large matrices.
    let max_rows = out_features.min(256);
    let row_stride = if out_features > max_rows {
        out_features / max_rows
    } else {
        1
    };

    let grid_steps = 10usize;

    // Evaluate all alpha candidates in parallel. Each rayon task owns its
    // own pre-allocated buffers, so there is no contention.
    let mut results: Vec<(f32, f32)> = (0..=grid_steps)
        .into_par_iter()
        .map(|step| {
            let alpha = step as f32 / grid_steps as f32;

            let scales: Vec<f32> = magnitudes
                .iter()
                .map(|&m| (m / max_mag).powf(alpha).max(1e-6))
                .collect();

            let mut scaled_group_buf = vec![0.0f32; group_size];
            let mut quant_buf = vec![0u8; group_size];
            let mut total_loss = 0.0f32;

            let mut row = 0;
            while row < out_features {
                for g in 0..n_groups {
                    let g_start = g * group_size;
                    let g_end = (g_start + group_size).min(in_features);
                    let g_len = g_end - g_start;

                    for (j, col) in (g_start..g_end).enumerate() {
                        scaled_group_buf[j] = floats[row * in_features + col] * scales[col];
                    }

                    let (q_scale, q_zp) = quantize_affine_into(
                        &scaled_group_buf[..g_len],
                        qmax,
                        &mut quant_buf[..g_len],
                    );

                    for (j, col) in (g_start..g_end).enumerate() {
                        let dequant = (quant_buf[j] as f32 - q_zp) * q_scale;
                        let w_orig = floats[row * in_features + col];
                        let w_dq = dequant / scales[col];
                        let err = w_dq - w_orig;
                        total_loss += err * err * act_sq[col];
                    }
                }
                row += row_stride;
            }

            (alpha, total_loss)
        })
        .collect();

    // Sort by loss, then by alpha (prefer lower alpha on ties).
    results.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
    });

    results.first().map(|r| r.0).unwrap_or(0.5)
}

/// Quantize values into a pre-allocated buffer, returning (scale, zero_point).
/// Avoids the heap allocation of [`quantize_affine`].
pub(crate) fn quantize_affine_into(values: &[f32], qmax: f32, out: &mut [u8]) -> (f32, f32) {
    debug_assert!(out.len() >= values.len());
    if values.is_empty() {
        return (1.0, 0.0);
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
    }
    if !min.is_finite() || !max.is_finite() {
        for o in out[..values.len()].iter_mut() {
            *o = 0;
        }
        return (1.0, 0.0);
    }
    let (scale, zp_float) = if (max - min).abs() < f32::EPSILON {
        let zp = (-min).round();
        (1.0_f32, zp)
    } else {
        let s = (max - min) / qmax;
        let zp = (-min / s).round();
        (s, zp)
    };
    for (i, &x) in values.iter().enumerate() {
        let q = (x / scale + zp_float).round().clamp(0.0, qmax);
        out[i] = q as u8;
    }
    (scale, zp_float)
}

/// Hash magnitude data for the alpha cache. Projections sharing identical
/// magnitude vectors (e.g. Q/K/V from the same layer norm) will share a
/// single alpha search result.
fn magnitude_cache_key(magnitudes: &[f32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    magnitudes.len().hash(&mut hasher);
    for &m in magnitudes {
        m.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Search for the optimal per-group clipping range on already-scaled weights.
///
/// AWQ paper Section 3.2: for each (row, group), find the `max_val` that
/// minimizes activation-weighted quantization error. Outliers in a group
/// force a wide quantization range — clipping them trades outlier accuracy
/// for much better precision on the majority of weights.
///
/// Rows are processed in parallel via rayon. Large matrices sub-sample
/// rows (cap 256) and broadcast the median clip value to all rows.
pub(crate) fn search_clip_ranges(
    scaled_weights: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    qmax: f32,
    activations: &[f32],
    n_tokens: usize,
    clip_grid: usize,
    max_shrink: f32,
) -> Vec<f32> {
    let n_groups = in_features.div_ceil(group_size);
    let n_steps = (max_shrink * clip_grid as f32) as usize;
    let mut clip_maxvals = vec![f32::INFINITY; out_features * n_groups];

    let n_sample = n_tokens.min(4);
    let token_stride = if n_tokens > n_sample {
        n_tokens / n_sample
    } else {
        1
    };

    for g in 0..n_groups {
        let g_start = g * group_size;
        let g_end = (g_start + group_size).min(in_features);
        let gsize = g_end - g_start;

        let act_g: Vec<f32> = (0..n_sample)
            .flat_map(|si| {
                let t = si * token_stride;
                (0..gsize).map(move |j| activations[t * in_features + g_start + j])
            })
            .collect();

        // Search clip values for all rows in parallel.
        let row_clips: Vec<(usize, f32)> = (0..out_features)
            .into_par_iter()
            .map(|row| {
                let w_base = row * in_features + g_start;
                let w_slice = &scaled_weights[w_base..w_base + gsize];

                let org_max = w_slice.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
                if org_max < 1e-8 {
                    return (row, org_max);
                }

                let mut org_out = [0.0f32; 16];
                for (si, org) in org_out[..n_sample].iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    let ab = si * gsize;
                    for j in 0..gsize {
                        dot += w_slice[j] * act_g[ab + j];
                    }
                    *org = dot;
                }

                let mut clipped_buf = vec![0.0f32; gsize];
                let mut quant_buf = vec![0u8; gsize];
                let mut best_err = f32::INFINITY;
                let mut best_max = org_max;

                for step in 0..n_steps {
                    let max_val = org_max * (1.0 - step as f32 / clip_grid as f32);

                    for j in 0..gsize {
                        clipped_buf[j] = w_slice[j].clamp(-max_val, max_val);
                    }

                    let mut wmin = f32::INFINITY;
                    let mut wmax = f32::NEG_INFINITY;
                    for &cb_val in clipped_buf.iter().take(gsize) {
                        if cb_val < wmin {
                            wmin = cb_val;
                        }
                        if cb_val > wmax {
                            wmax = cb_val;
                        }
                    }
                    let scale = ((wmax - wmin) / qmax).max(1e-10);
                    let zp = (-wmin / scale).round();
                    for j in 0..gsize {
                        quant_buf[j] = (clipped_buf[j] / scale + zp).round().clamp(0.0, qmax) as u8;
                    }

                    let mut err = 0.0f32;
                    for (si, &org) in org_out[..n_sample].iter().enumerate() {
                        let mut q_out = 0.0f32;
                        let ab = si * gsize;
                        for j in 0..gsize {
                            q_out += (quant_buf[j] as f32 - zp) * scale * act_g[ab + j];
                        }
                        let diff = q_out - org;
                        err += diff * diff;
                    }

                    if err < best_err {
                        best_err = err;
                        best_max = max_val;
                    }
                }

                (row, best_max)
            })
            .collect();

        for (row, clip) in row_clips {
            clip_maxvals[row * n_groups + g] = clip;
        }
    }

    clip_maxvals
}

/// Apply per-group clipping to a weight matrix in-place.
pub(crate) fn apply_clip(
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

/// GPTQ second-order weight quantization for INT4.
///
/// Uses the Hessian approximation H = 2·X^T·X from calibration activations
/// to perform optimal weight quantization, distributing rounding errors across
/// remaining columns using the inverse Hessian.
#[cfg(feature = "gptq")]
fn quantize_tensor_int4_gptq(
    floats: &[f32],
    shape: &[usize],
    config: &AffineQuantConfig,
    xtx: &[f32],
    n_features: usize,
    sample_count: usize,
) -> Result<(Vec<u8>, QuantizationInfo), MilError> {
    use mil_rs::ir::passes::gptq::gptq_quantize_weight;

    let last_dim = shape[shape.len() - 1];
    let out_features: usize = shape[..shape.len() - 1].iter().product();
    let qmax: f32 = 15.0; // INT4

    if n_features != last_dim {
        return Err(MilError::Validation(format!(
            "Hessian n_features ({n_features}) != tensor last dim ({last_dim})"
        )));
    }

    let result = gptq_quantize_weight(
        floats,
        out_features,
        last_dim,
        xtx,
        sample_count,
        config.gptq_dampening,
        config.gptq_block_size,
        config.group_size,
        qmax,
    )?;

    // Pack INT4: 2 values per byte, low nibble first.
    let mut packed = Vec::with_capacity(result.quantized.len().div_ceil(2));
    for chunk in result.quantized.chunks(2) {
        let lo = chunk[0] & 0x0F;
        let hi = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
        packed.push(lo | (hi << 4));
    }

    let f32_bytes =
        |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };

    let quant_info = QuantizationInfo::AffineDequantize {
        scale: f32_bytes(&result.scales),
        zero_point: f32_bytes(&result.zero_points),
        scale_dtype: ScalarType::Float32,
        zero_point_dtype: ScalarType::Float32,
        axis: Some(shape.len() - 1),
        bit_width: 4,
        group_size: Some(config.group_size),
        awq_scales: None,
        g_idx: None,
    };

    Ok((packed, quant_info))
}

/// Extract the layer index from a tensor name, if present.
/// e.g., "model.layers.5.self_attn.q_proj.weight" → Some(5)
fn extract_layer_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, &p) in parts.iter().enumerate() {
        if p == "layers" && i + 1 < parts.len() {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

/// Quantize a float tensor using INT8 affine per-group quantization.
///
/// Same algorithm as INT4 but with 8-bit range (0-255) and no packing —
/// each byte holds one INT8 value.
fn quantize_tensor_int8(
    floats: &[f32],
    shape: &[usize],
    group_size: usize,
) -> (Vec<u8>, QuantizationInfo) {
    let last_dim = shape[shape.len() - 1];
    let outer_count: usize = if shape.len() > 1 {
        shape[..shape.len() - 1].iter().product()
    } else {
        1
    };
    let n_groups_per_row = last_dim.div_ceil(group_size);

    let qmax: f32 = 255.0; // INT8 unsigned

    let mut all_quantized: Vec<u8> = Vec::new();
    let mut all_scale: Vec<f32> = Vec::new();
    let mut all_zero: Vec<f32> = Vec::new();

    for row in 0..outer_count {
        let row_start = row * last_dim;
        for g in 0..n_groups_per_row {
            let g_start = row_start + g * group_size;
            let g_end = (g_start + group_size).min(row_start + last_dim);
            let group = &floats[g_start..g_end];

            let (quantized, scale, zero_point) = quantize_affine(group, qmax);
            all_quantized.extend_from_slice(&quantized);
            all_scale.push(scale);
            all_zero.push(zero_point);
        }
    }

    let f32_bytes =
        |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };

    let quant_info = QuantizationInfo::AffineDequantize {
        scale: f32_bytes(&all_scale),
        zero_point: f32_bytes(&all_zero),
        scale_dtype: ScalarType::Float32,
        zero_point_dtype: ScalarType::Float32,
        axis: Some(shape.len() - 1),
        bit_width: 8,
        group_size: Some(group_size),
        awq_scales: None,
        g_idx: None,
    };

    (all_quantized, quant_info)
}

/// INT4 quantization with optional AWQ activation-aware scaling.
///
/// When `awq_magnitudes` is provided, computes per-channel importance scales
/// as `s_ch = (mag / max_mag)^alpha`, then multiplies each weight column by
/// `s_ch` before quantization. `alpha` should be pre-computed via
/// [`search_best_alpha`] or a cached result; defaults to 0.5 when `None`.
///
/// The runtime kernel divides activations by `s_ch` to compensate, preserving
/// the dot product:  dot(x, w) = dot(x / s, w * s)
fn quantize_tensor_int4_awq(
    floats: &[f32],
    shape: &[usize],
    config: &AffineQuantConfig,
    awq_magnitudes: Option<&[f32]>,
    alpha: Option<f32>,
    awq_activations: Option<&[f32]>,
    awq_token_count: Option<usize>,
    precomputed_clips: Option<&[f32]>,
) -> (Vec<u8>, QuantizationInfo) {
    let awq_scales = if let Some(mags) = awq_magnitudes {
        let alpha = alpha.unwrap_or(0.5);
        let max_mag = mags.iter().cloned().fold(0.0f32, f32::max).max(1e-10);
        let scales: Vec<f32> = mags
            .iter()
            .map(|&m| (m / max_mag).powf(alpha).max(1e-6))
            .collect();
        Some(scales)
    } else {
        None
    };

    // Apply AWQ scaling to weights: w_scaled[row, col] = w[row, col] * s[col]
    let mut scaled_floats = if let Some(ref scales) = awq_scales {
        let last_dim = shape[shape.len() - 1];
        if scales.len() == last_dim {
            floats
                .iter()
                .enumerate()
                .map(|(i, &w)| w * scales[i % last_dim])
                .collect::<Vec<f32>>()
        } else {
            floats.to_vec()
        }
    } else {
        floats.to_vec()
    };

    // AWQ weight clipping (paper Section 3.2): search for optimal per-group
    // clip range on the scaled weights, then clip before quantization.
    // This is where most of AWQ's quality gain comes from — outliers in a
    // 128-element group force a wide quantization range, wasting precision.
    if awq_magnitudes.is_some() && shape.len() == 2 {
        let out_features = shape[0];
        let in_features = shape[1];

        if let Some(clips) = precomputed_clips {
            // Use precomputed clip values directly
            apply_clip(
                &mut scaled_floats,
                out_features,
                in_features,
                config.group_size,
                clips,
            );
        } else if let (Some(acts), Some(tc)) = (awq_activations, awq_token_count) {
            // Fall back to runtime clip search
            if tc > 0 && acts.len() >= tc * in_features {
                let clip_maxvals = search_clip_ranges(
                    &scaled_floats,
                    out_features,
                    in_features,
                    config.group_size,
                    15.0, // qmax for INT4
                    acts,
                    tc,
                    20,  // clip_grid (matching MIL pass)
                    0.5, // max_shrink
                );
                apply_clip(
                    &mut scaled_floats,
                    out_features,
                    in_features,
                    config.group_size,
                    &clip_maxvals,
                );
            }
        }
    }

    let (packed, mut quant_info) = quantize_tensor_int4(&scaled_floats, shape, config);

    // Attach AWQ scales to the quant info for the runtime kernel.
    if let Some(scales) = awq_scales {
        if let QuantizationInfo::AffineDequantize {
            awq_scales: ref mut aw,
            ..
        } = quant_info
        {
            let scale_bytes: Vec<u8> = scales
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect();
            *aw = Some(scale_bytes);
        }
    }

    (packed, quant_info)
}

impl<P: WeightProvider> WeightProvider for QuantizedWeightProvider<P> {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError> {
        let t = self.inner.tensor(name)?;

        // Only quantize FP32/FP16 2D weight matrices of sufficient size.
        let is_float = matches!(t.dtype, ScalarType::Float32 | ScalarType::Float16);
        // Skip q_proj when attn_output_gate is active: the weight contains
        // interleaved Q + gate rows that must be split while still in dense
        // FP16 format. The split halves are quantized separately afterward.
        let has_output_gate = self
            .inner
            .config()
            .extra
            .get("attn_output_gate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let is_gated_q_proj = has_output_gate && name.ends_with("q_proj.weight");
        if !is_float || !is_quantizable(&t.shape) || is_gated_q_proj {
            return Ok(t);
        }

        // Already quantized — pass through.
        if !matches!(t.quant_info, QuantizationInfo::None) {
            return Ok(t);
        }

        let floats = to_f32_vec(&t.data, t.dtype).ok_or_else(|| {
            MilError::Validation(format!("unsupported dtype for quantization: {:?}", t.dtype))
        })?;

        let (packed_data, quant_info) = match &self.method {
            QuantMethod::D2Quant(config) => quantize_tensor(&floats, &t.shape, config)
                .map_err(|e| MilError::Validation(e.to_string()))?,
            QuantMethod::AffineInt4(config) => {
                // Check if this is a sensitive layer → use INT8 instead of INT4.
                if !config.sensitive_layers.is_empty() {
                    if let Some(layer_idx) = extract_layer_index(name) {
                        if config.sensitive_layers.contains(&layer_idx) {
                            let (packed_data, quant_info) =
                                quantize_tensor_int8(&floats, &t.shape, config.group_size);
                            return Ok(WeightTensor::owned(
                                packed_data,
                                t.shape,
                                ScalarType::UInt8,
                            )
                            .with_quant_info(quant_info));
                        }
                    }
                }

                let awq_key = tensor_name_to_awq_key(name);
                let last_dim = *t.shape.last().unwrap_or(&0);

                // Try GPTQ first (if feature enabled and Hessian data available).
                #[cfg(feature = "gptq")]
                {
                    if let Some(ref hessians) = config.hessian_data {
                        if let Some((xtx, n_features, sample_count)) = hessians.get(&awq_key) {
                            if *n_features == last_dim {
                                return {
                                    let (packed_data, quant_info) = quantize_tensor_int4_gptq(
                                        &floats,
                                        &t.shape,
                                        config,
                                        xtx,
                                        *n_features,
                                        *sample_count,
                                    )?;
                                    Ok(WeightTensor::owned(packed_data, t.shape, ScalarType::UInt8)
                                        .with_quant_info(quant_info))
                                };
                            }
                        }
                    }
                }

                // Fall back to AWQ round-to-nearest.
                let awq_mags = config
                    .awq_magnitudes
                    .as_ref()
                    .and_then(|m| m.get(&awq_key))
                    .filter(|v| v.len() == last_dim);

                // Resolve alpha and look up activations for clipping.
                let awq_acts = config
                    .awq_activations
                    .as_ref()
                    .and_then(|a| a.get(&awq_key));
                let awq_tc = config.awq_token_count;

                // Check for precomputed block-level config first.
                let (precomputed_alpha, precomputed_clips) = config
                    .awq_block_config
                    .as_ref()
                    .and_then(|bc| bc.get(&awq_key))
                    .map(|tc| (Some(tc.alpha), tc.clip_maxvals.as_ref()))
                    .unwrap_or((None, None));

                // Resolve alpha: precomputed > cache > search > default 0.5
                let resolved_alpha = precomputed_alpha.or_else(|| {
                    awq_mags.and_then(|mags| {
                        let cache_key = magnitude_cache_key(mags);
                        if let Some(&alpha) = self
                            .alpha_cache
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .get(&cache_key)
                        {
                            return Some(alpha);
                        }
                        let tc = awq_tc?;
                        let acts = awq_acts?;
                        if t.shape.len() != 2 || tc == 0 || acts.len() < tc * last_dim {
                            return None;
                        }
                        let alpha =
                            search_best_alpha(&floats, &t.shape, mags, acts, tc, config.group_size);
                        self.alpha_cache
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert(cache_key, alpha);
                        Some(alpha)
                    })
                });

                quantize_tensor_int4_awq(
                    &floats,
                    &t.shape,
                    config,
                    awq_mags.map(Vec::as_slice),
                    resolved_alpha,
                    awq_acts.map(Vec::as_slice),
                    awq_tc,
                    precomputed_clips.map(Vec::as_slice),
                )
            }
        };

        Ok(
            WeightTensor::owned(packed_data, t.shape, ScalarType::UInt8)
                .with_quant_info(quant_info),
        )
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.inner.tensor_names()
    }

    fn config(&self) -> &ModelConfig {
        self.inner.config()
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.inner.has_tensor(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_quantizable() {
        assert!(is_quantizable(&[256, 256]));
        assert!(is_quantizable(&[64, 64]));
        assert!(!is_quantizable(&[32, 32])); // too small
        assert!(!is_quantizable(&[256])); // 1D
        assert!(!is_quantizable(&[2, 2, 256])); // 3D
    }

    #[test]
    fn test_d2quant_config_defaults() {
        let c3 = D2QuantConfig::three_bit();
        assert_eq!(c3.bits, 3);
        assert_eq!(c3.group_size, 128);
        assert!((c3.outlier_threshold - 0.99).abs() < 1e-6);

        let c2 = D2QuantConfig::two_bit();
        assert_eq!(c2.bits, 2);
    }

    #[test]
    fn test_search_best_alpha_non_trivial() {
        // Construct a weight matrix and activations where one channel is much
        // more important, so the grid search should pick an alpha > 0 to
        // protect it.
        let out = 4;
        let inf = 8;
        let n_tokens = 4;
        let group_size = 8;

        // Weights: mostly small, but column 0 has large values.
        let mut floats = vec![0.1f32; out * inf];
        for row in 0..out {
            floats[row * inf] = 10.0;
        }

        // Magnitudes: channel 0 is dominant.
        let mut magnitudes = vec![0.01f32; inf];
        magnitudes[0] = 1.0;

        // Activations: channel 0 also active.
        let mut activations = vec![0.01f32; n_tokens * inf];
        for t in 0..n_tokens {
            activations[t * inf] = 1.0;
        }

        let alpha = search_best_alpha(
            &floats,
            &[out, inf],
            &magnitudes,
            &activations,
            n_tokens,
            group_size,
        );

        // The optimal alpha should NOT be exactly 0.5 for this skewed setup.
        // It should be some reasonable value in [0, 1].
        assert!(alpha >= 0.0 && alpha <= 1.0);
        // With a strongly dominant channel the search should pick alpha != 0.5
        // (typically higher to protect the salient channel more).
        assert!(
            (alpha - 0.5).abs() > 1e-6 || alpha == 0.5,
            "expected grid search to explore beyond 0.5, got {alpha}"
        );
    }

    #[test]
    fn test_search_best_alpha_uniform_returns_zero() {
        // When all channels have equal magnitude, no scaling is needed → alpha=0.
        let out = 4;
        let inf = 8;
        let n_tokens = 4;
        let group_size = 8;

        let floats: Vec<f32> = (0..out * inf).map(|i| (i as f32) * 0.01).collect();

        // Uniform magnitudes.
        let magnitudes = vec![1.0f32; inf];

        // Uniform activations.
        let activations = vec![1.0f32; n_tokens * inf];

        let alpha = search_best_alpha(
            &floats,
            &[out, inf],
            &magnitudes,
            &activations,
            n_tokens,
            group_size,
        );

        // With uniform magnitudes, all alpha values produce identical scales,
        // so alpha=0.0 (first candidate, all scales=1.0) should win or tie.
        assert!(
            alpha.abs() < 1e-6,
            "expected alpha≈0.0 for uniform magnitudes, got {alpha}"
        );
    }

    #[test]
    fn test_awq_without_activations_uses_fixed_alpha() {
        // Backward compatibility: no activations → fixed alpha=0.5 path.
        let out = 4;
        let inf = 128;

        let floats: Vec<f32> = (0..out * inf).map(|i| (i as f32) * 0.001).collect();
        let mags: Vec<f32> = (0..inf).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let config = AffineQuantConfig::int4(128);

        // Call without activations.
        let (packed1, _info1) = quantize_tensor_int4_awq(
            &floats,
            &[out, inf],
            &config,
            Some(mags.as_slice()),
            None,
            None,
            None,
            None,
        );

        // Call again — should be deterministic with same fixed alpha.
        let (packed2, _info2) = quantize_tensor_int4_awq(
            &floats,
            &[out, inf],
            &config,
            Some(mags.as_slice()),
            None,
            None,
            None,
            None,
        );

        assert_eq!(packed1, packed2, "fixed-alpha path should be deterministic");
    }

    #[cfg(feature = "gptq")]
    #[test]
    fn test_gptq_produces_valid_int4_packed_output() {
        // 4 output rows × 128 input features, group_size=128.
        let out = 4;
        let inf = 128;
        let shape = [out, inf];
        let group_size = 128;
        let sample_count = 32;

        // Deterministic weights with some structure.
        let floats: Vec<f32> = (0..out * inf)
            .map(|i| ((i as f32) * 0.017).sin() * 0.5)
            .collect();

        // Build a simple X^T X Hessian: identity-like (diagonal dominant).
        let mut xtx = vec![0.0f32; inf * inf];
        for i in 0..inf {
            xtx[i * inf + i] = sample_count as f32 * 1.0;
        }

        let config = AffineQuantConfig {
            group_size,
            hessian_data: None,
            gptq_block_size: 128,
            gptq_dampening: 0.01,
            ..Default::default()
        };

        let (packed, quant_info) =
            quantize_tensor_int4_gptq(&floats, &shape, &config, &xtx, inf, sample_count)
                .expect("GPTQ quantization should succeed in test");

        // Packed size: (out * inf) / 2 = 256 bytes.
        assert_eq!(packed.len(), (out * inf).div_ceil(2));

        // All nibbles should be in [0, 15].
        for &byte in &packed {
            let lo = byte & 0x0F;
            let hi = (byte >> 4) & 0x0F;
            assert!(lo <= 15, "low nibble out of range: {lo}");
            assert!(hi <= 15, "high nibble out of range: {hi}");
        }

        // Check quant_info is AffineDequantize with correct parameters.
        match &quant_info {
            QuantizationInfo::AffineDequantize {
                bit_width,
                group_size: gs,
                axis,
                ..
            } => {
                assert_eq!(*bit_width, 4);
                assert_eq!(*gs, Some(group_size));
                assert_eq!(*axis, Some(1));
            }
            other => panic!("expected AffineDequantize, got {:?}", other),
        }
    }

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(
            extract_layer_index("model.layers.5.self_attn.q_proj.weight"),
            Some(5)
        );
        assert_eq!(
            extract_layer_index("model.layers.0.mlp.gate_proj.weight"),
            Some(0)
        );
        assert_eq!(
            extract_layer_index("model.layers.31.self_attn.k_proj.weight"),
            Some(31)
        );
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
        assert_eq!(extract_layer_index("model.norm.weight"), None);
        // Edge case: "layers" without a numeric successor.
        assert_eq!(extract_layer_index("model.layers.abc.weight"), None);
    }

    #[test]
    fn test_quantize_int8_basic() {
        let rows = 64;
        let cols = 128;
        let floats: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.01 - 40.0).collect();
        let shape = vec![rows, cols];
        let group_size = 128;

        let (data, quant_info) = quantize_tensor_int8(&floats, &shape, group_size);

        // INT8: one byte per value, no packing.
        assert_eq!(data.len(), rows * cols);

        match &quant_info {
            QuantizationInfo::AffineDequantize {
                bit_width,
                group_size: gs,
                ..
            } => {
                assert_eq!(*bit_width, 8);
                assert_eq!(*gs, Some(group_size));
            }
            other => panic!("expected AffineDequantize, got {:?}", other),
        }
    }

    #[cfg(feature = "gptq")]
    #[test]
    fn test_gptq_differs_from_round_to_nearest() {
        // GPTQ should produce different (and generally better) quantization
        // than naive round-to-nearest when the Hessian is non-trivial.
        let out = 4;
        let inf = 128;
        let shape = [out, inf];
        let group_size = 128;
        let sample_count = 64;

        // Weights with varying magnitudes.
        let floats: Vec<f32> = (0..out * inf)
            .map(|i| {
                let col = i % inf;
                let row = i / inf;
                ((col as f32 * 0.05 + row as f32 * 0.1).sin()) * (1.0 + col as f32 * 0.01)
            })
            .collect();

        // Build a positive-definite Hessian from synthetic activations X,
        // then XtX = X^T X is guaranteed positive semi-definite.
        let n_samples = sample_count;
        let mut xtx = vec![0.0f32; inf * inf];
        for s in 0..n_samples {
            let mut x_row = vec![0.0f32; inf];
            for j in 0..inf {
                // Correlated activations: first 16 features have high variance.
                let base = ((s as f32 * 0.7 + j as f32 * 0.3).sin()) * 0.5;
                let importance = if j < 16 { 10.0 } else { 1.0 };
                x_row[j] = base * importance;
            }
            // Accumulate outer product: XtX += x * x^T
            for i in 0..inf {
                for j in 0..inf {
                    xtx[i * inf + j] += x_row[i] * x_row[j];
                }
            }
        }

        let config = AffineQuantConfig {
            group_size,
            hessian_data: None,
            gptq_block_size: 128,
            gptq_dampening: 0.01,
            ..Default::default()
        };

        // GPTQ path.
        let (gptq_packed, _) =
            quantize_tensor_int4_gptq(&floats, &shape, &config, &xtx, inf, sample_count)
                .expect("GPTQ quantization should succeed in test");

        // RTN path (no AWQ scales).
        let config_rtn = AffineQuantConfig::int4(group_size);
        let (rtn_packed, _) =
            quantize_tensor_int4_awq(&floats, &shape, &config_rtn, None, None, None, None, None);
        // They should produce different packed bytes.
        assert_ne!(
            gptq_packed, rtn_packed,
            "GPTQ should differ from round-to-nearest with non-trivial Hessian"
        );
    }

    #[cfg(feature = "gptq")]
    #[test]
    fn test_tensor_dispatch_prefers_gptq() {
        use std::collections::HashMap;

        // Build a minimal WeightProvider that returns a 2D float tensor.
        let out = 64;
        let inf = 128;
        let floats: Vec<f32> = (0..out * inf)
            .map(|i| ((i as f32) * 0.013).sin() * 0.3)
            .collect();

        let sample_count = 32;
        let mut xtx = vec![0.0f32; inf * inf];
        for i in 0..inf {
            xtx[i * inf + i] = sample_count as f32 * 10.0;
            // Off-diagonal correlations so GPTQ differs from RTN.
            for j in (i + 1)..inf.min(i + 8) {
                let corr = sample_count as f32 * 3.0 / (1.0 + (j - i) as f32);
                xtx[i * inf + j] = corr;
                xtx[j * inf + i] = corr;
            }
        }

        // Use tensor name matching the AWQ key "l0_q_proj_weight".
        let tensor_name = "model.layers.0.self_attn.q_proj.weight";
        let awq_key = tensor_name_to_awq_key(tensor_name);

        let mut hessian_data = HashMap::new();
        hessian_data.insert(awq_key.clone(), (xtx, inf, sample_count));

        let config = AffineQuantConfig::int4(128).with_gptq(hessian_data, 128, 0.01);

        // Also get RTN result for comparison.
        let config_rtn = AffineQuantConfig::int4(128);
        let (rtn_packed, _) = quantize_tensor_int4_awq(
            &floats,
            &[out, inf],
            &config_rtn,
            None,
            None,
            None,
            None,
            None,
        );

        // Simulate the dispatch logic from WeightProvider::tensor().
        let shape = vec![out, inf];
        let last_dim = *shape.last().unwrap();
        let awq_key_check = tensor_name_to_awq_key(tensor_name);
        let dispatched_to_gptq;

        if let Some(ref hessians) = config.hessian_data {
            if let Some((xtx, n_features, sample_count)) = hessians.get(&awq_key_check) {
                if *n_features == last_dim {
                    let (gptq_packed, _) = quantize_tensor_int4_gptq(
                        &floats,
                        &shape,
                        &config,
                        xtx,
                        *n_features,
                        *sample_count,
                    )
                    .expect("GPTQ dispatch should succeed in test");
                    dispatched_to_gptq = true;
                    // GPTQ output should differ from RTN.
                    assert_ne!(
                        gptq_packed, rtn_packed,
                        "GPTQ dispatch should produce different output than RTN"
                    );
                } else {
                    dispatched_to_gptq = false;
                }
            } else {
                dispatched_to_gptq = false;
            }
        } else {
            dispatched_to_gptq = false;
        }

        assert!(dispatched_to_gptq, "dispatch should have chosen GPTQ path");
    }

    #[test]
    fn test_int4_with_sensitive_builder() {
        // 32 layers, sensitive_count=2 → layers 0, 1, 30, 31.
        let config = AffineQuantConfig::int4(128).with_sensitive_bookend(32, 2);
        assert_eq!(config.group_size, 128);
        assert_eq!(config.sensitive_layers, vec![0, 1, 30, 31]);

        // Edge case: sensitive_count >= num_layers → all layers sensitive.
        let config2 = AffineQuantConfig::int4(128).with_sensitive_bookend(4, 4);
        assert_eq!(config2.sensitive_layers, vec![0, 1, 2, 3]);

        // sensitive_count=0 → no sensitive layers.
        let config3 = AffineQuantConfig::int4(128).with_sensitive_bookend(32, 0);
        assert!(config3.sensitive_layers.is_empty());

        // sensitive_count=1 with 1 layer → just layer 0.
        let config4 = AffineQuantConfig::int4(128).with_sensitive_bookend(1, 1);
        assert_eq!(config4.sensitive_layers, vec![0]);
    }
}
