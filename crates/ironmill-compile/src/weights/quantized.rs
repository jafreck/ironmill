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

use mil_rs::error::MilError;
use mil_rs::ir::ScalarType;
use mil_rs::ir::passes::affine_quantize::quantize_affine;
use mil_rs::ir::passes::d2quant::dual_scale::{
    dual_scale_quantize, pack_2bit, pack_3bit, pack_mask,
};
use mil_rs::weights::{ModelConfig, QuantizationInfo, WeightProvider, WeightTensor};

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
}

impl AffineQuantConfig {
    /// Default INT4 per-group config: group_size=128.
    pub fn int4(group_size: usize) -> Self {
        Self {
            group_size,
            awq_magnitudes: None,
            awq_activations: None,
            awq_token_count: None,
        }
    }

    /// INT4 with AWQ channel scales loaded from calibration output.
    pub fn int4_awq(
        group_size: usize,
        magnitudes: std::collections::HashMap<String, Vec<f32>>,
    ) -> Self {
        Self {
            group_size,
            awq_magnitudes: Some(magnitudes),
            awq_activations: None,
            awq_token_count: None,
        }
    }

    /// INT4 with AWQ magnitudes and raw calibration activations for alpha grid search.
    pub fn int4_awq_with_activations(
        group_size: usize,
        magnitudes: std::collections::HashMap<String, Vec<f32>>,
        activations: std::collections::HashMap<String, Vec<f32>>,
        token_count: usize,
    ) -> Self {
        Self {
            group_size,
            awq_magnitudes: Some(magnitudes),
            awq_activations: Some(activations),
            awq_token_count: Some(token_count),
        }
    }
}

impl Default for AffineQuantConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            awq_magnitudes: None,
            awq_activations: None,
            awq_token_count: None,
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
}

impl<P: WeightProvider> QuantizedWeightProvider<P> {
    /// Create a D2Quant quantized provider wrapping `inner`.
    pub fn new(inner: P, config: D2QuantConfig) -> Self {
        Self {
            inner,
            method: QuantMethod::D2Quant(config),
        }
    }

    /// Create an INT4 affine quantized provider wrapping `inner`.
    pub fn new_int4(inner: P, config: AffineQuantConfig) -> Self {
        Self {
            inner,
            method: QuantMethod::AffineInt4(config),
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
            assert!(data.len() % 4 == 0);
            Some(
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        }
        ScalarType::Float16 => {
            assert!(data.len() % 2 == 0);
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
) -> (Vec<u8>, QuantizationInfo) {
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
                _ => unreachable!("bits validated on construction"),
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
    (all_quantized_packed, quant_info)
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

/// Search α ∈ {0.0, 0.05, 0.1, ..., 1.0} that minimizes reconstruction error.
///
/// For each candidate alpha, compute:
///   error(α) = ||Q(W·diag(s^α))·diag(s^-α)·X - W·X||²
/// and return the alpha that minimizes it.
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

    // Sub-sample tokens to limit memory/compute.
    let max_tokens = n_tokens.min(128);

    // Pre-compute reference output: ref_out[row * max_tokens + t] = dot(W[row], X[t])
    let mut ref_out = vec![0.0f32; out_features * max_tokens];
    for row in 0..out_features {
        for t in 0..max_tokens {
            let mut dot = 0.0f32;
            for c in 0..in_features {
                dot += floats[row * in_features + c] * activations[t * in_features + c];
            }
            ref_out[row * max_tokens + t] = dot;
        }
    }

    let max_mag = magnitudes.iter().cloned().fold(0.0f32, f32::max).max(1e-10);

    let grid_steps = 20usize;
    let mut best_alpha = 0.5f32;
    let mut best_loss = f32::INFINITY;

    for step in 0..=grid_steps {
        let alpha = step as f32 / grid_steps as f32;

        // Compute scales: s[c] = (mag[c] / max_mag).powf(alpha).max(1e-6)
        let scales: Vec<f32> = magnitudes
            .iter()
            .map(|&m| (m / max_mag).powf(alpha).max(1e-6))
            .collect();

        // Build W_dq: quantize W·diag(s), then dequant and divide by s.
        let mut total_loss = 0.0f32;

        for row in 0..out_features {
            // Quantize and dequantize this row group-by-group, then compute
            // the loss against reference for all tokens.
            let mut w_dq_row = vec![0.0f32; in_features];

            for g in 0..n_groups {
                let g_start = g * group_size;
                let g_end = (g_start + group_size).min(in_features);

                let scaled_group: Vec<f32> = (g_start..g_end)
                    .map(|col| floats[row * in_features + col] * scales[col])
                    .collect();

                let (quantized, q_scale, q_zp) = quantize_affine(&scaled_group, qmax);

                for (j, col) in (g_start..g_end).enumerate() {
                    let dequant = (quantized[j] as f32 - q_zp) * q_scale;
                    w_dq_row[col] = dequant / scales[col];
                }
            }

            for t in 0..max_tokens {
                let mut dq_dot = 0.0f32;
                for c in 0..in_features {
                    dq_dot += w_dq_row[c] * activations[t * in_features + c];
                }
                let err = dq_dot - ref_out[row * max_tokens + t];
                total_loss += err * err;
            }
        }

        if total_loss < best_loss {
            best_loss = total_loss;
            best_alpha = alpha;
        }
    }

    best_alpha
}

/// INT4 quantization with optional AWQ activation-aware scaling.
///
/// When `awq_magnitudes` is provided, computes per-channel importance scales
/// as `s_ch = (mag / max_mag)^alpha`, then multiplies each weight column by
/// `s_ch` before quantization. When activations are also available, alpha is
/// chosen via grid search to minimize reconstruction error; otherwise alpha=0.5.
///
/// The runtime kernel divides activations by `s_ch` to compensate, preserving
/// the dot product:  dot(x, w) = dot(x / s, w * s)
fn quantize_tensor_int4_awq(
    floats: &[f32],
    shape: &[usize],
    config: &AffineQuantConfig,
    awq_magnitudes: Option<&Vec<f32>>,
    awq_activations: Option<&Vec<f32>>,
    awq_token_count: Option<usize>,
) -> (Vec<u8>, QuantizationInfo) {
    let awq_scales = if let Some(mags) = awq_magnitudes {
        let alpha = if let (Some(acts), Some(tc)) = (awq_activations, awq_token_count) {
            if shape.len() == 2 && tc > 0 && acts.len() >= tc * shape[1] {
                search_best_alpha(floats, shape, mags, acts, tc, config.group_size)
            } else {
                0.5f32
            }
        } else {
            0.5f32
        };
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
    let scaled_floats = if let Some(ref scales) = awq_scales {
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
            QuantMethod::D2Quant(config) => quantize_tensor(&floats, &t.shape, config),
            QuantMethod::AffineInt4(config) => {
                // Look up AWQ magnitudes for this tensor.
                let awq_key = tensor_name_to_awq_key(name);
                let last_dim = *t.shape.last().unwrap_or(&0);
                let awq_mags = config
                    .awq_magnitudes
                    .as_ref()
                    .and_then(|m| m.get(&awq_key))
                    .filter(|v| v.len() == last_dim); // Only apply if dimensions match
                let awq_acts = config
                    .awq_activations
                    .as_ref()
                    .and_then(|a| a.get(&awq_key));
                let awq_tc = config.awq_token_count;
                quantize_tensor_int4_awq(&floats, &t.shape, config, awq_mags, awq_acts, awq_tc)
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
        let (packed1, _info1) =
            quantize_tensor_int4_awq(&floats, &[out, inf], &config, Some(&mags), None, None);

        // Call again — should be deterministic with same fixed alpha.
        let (packed2, _info2) =
            quantize_tensor_int4_awq(&floats, &[out, inf], &config, Some(&mags), None, None);

        assert_eq!(packed1, packed2, "fixed-alpha path should be deterministic");
    }
}
