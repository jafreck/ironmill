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
}

impl AffineQuantConfig {
    /// Default INT4 per-group config: group_size=128.
    pub fn int4(group_size: usize) -> Self {
        Self { group_size }
    }
}

impl Default for AffineQuantConfig {
    fn default() -> Self {
        Self { group_size: 128 }
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
            QuantMethod::AffineInt4(config) => quantize_tensor_int4(&floats, &t.shape, config),
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
}
