//! LoRA adapter weight merging during ONNX → MIL conversion.
//!
//! Detects low-rank adapter (LoRA) weight pairs in ONNX model initializers
//! and merges them into base weights before const lowering. This produces a
//! single MIL program with baked-in adapter weights, removing the need for
//! runtime adapter application.
//!
//! ## LoRA convention
//!
//! Adapter initializers follow the naming pattern:
//!
//! - `<prefix>.lora_A.weight` — shape `(rank, in_features)`
//! - `<prefix>.lora_B.weight` — shape `(out_features, rank)`
//! - `<prefix>.lora_alpha` (optional scalar or 1-element tensor)
//!
//! The merge formula is:
//!
//! ```text
//! W_new = W + (alpha / rank) * B @ A
//! ```
//!
//! When `alpha` is absent the scaling factor defaults to `1.0` (i.e. `alpha = rank`).

use half::f16;

use crate::error::{MilError, Result};
use crate::ir::ScalarType;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A detected LoRA adapter pair ready for merging.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// The base weight name this adapter targets (e.g. `model.layer.weight`).
    pub base_name: String,
    /// Flat row-major data for the A matrix (rank × in_features).
    pub a_data: Vec<u8>,
    /// Shape of A: `[rank, in_features]`.
    pub a_shape: [usize; 2],
    /// Flat row-major data for the B matrix (out_features × rank).
    pub b_data: Vec<u8>,
    /// Shape of B: `[out_features, rank]`.
    pub b_shape: [usize; 2],
    /// Element data type shared by A, B, and the base weight.
    pub dtype: ScalarType,
    /// Scaling factor `alpha`. When `None`, defaults to `rank` (scale = 1.0).
    pub alpha: Option<f64>,
}

/// Scan ONNX initializer names and group LoRA A/B pairs.
///
/// Returns a list of [`LoraAdapter`]s found. Initializer tensors whose names
/// do not follow LoRA conventions are ignored.
pub fn detect_lora_adapters(
    initializers: &[(String, Vec<u8>, Vec<usize>, ScalarType)],
) -> Vec<LoraAdapter> {
    use std::collections::HashMap;

    type InitEntry = (String, Vec<u8>, Vec<usize>, ScalarType);

    // Index everything by name for fast lookup.
    let by_name: HashMap<&str, &InitEntry> =
        initializers.iter().map(|t| (t.0.as_str(), t)).collect();

    // Collect unique prefixes from `*.lora_A.weight` names.
    let mut adapters = Vec::new();
    for (name, a_data, a_shape, a_dtype) in initializers {
        let prefix = match name.strip_suffix(".lora_A.weight") {
            Some(p) => p,
            None => continue,
        };

        let b_key = format!("{prefix}.lora_B.weight");
        let (_, b_data, b_shape, b_dtype) = match by_name.get(b_key.as_str()) {
            Some(t) => *t,
            None => continue,
        };

        if a_dtype != b_dtype {
            continue;
        }
        if a_shape.len() != 2 || b_shape.len() != 2 {
            continue;
        }

        // Look for an alpha scalar.
        let alpha_key = format!("{prefix}.lora_alpha");
        let alpha = by_name
            .get(alpha_key.as_str())
            .and_then(|(_, data, shape, dtype)| {
                let numel: usize = shape.iter().product();
                if numel != 1 {
                    return None;
                }
                scalar_from_bytes(data, *dtype)
            });

        // Derive the base weight name. Common conventions:
        //   model.layers.0.self_attn.q_proj.lora_A.weight  →  model.layers.0.self_attn.q_proj.weight
        let base_name = format!("{prefix}.weight");

        adapters.push(LoraAdapter {
            base_name,
            a_data: a_data.clone(),
            a_shape: [a_shape[0], a_shape[1]],
            b_data: b_data.clone(),
            b_shape: [b_shape[0], b_shape[1]],
            dtype: *a_dtype,
            alpha,
        });
    }

    adapters
}

/// Merge a LoRA adapter into a base weight tensor in-place.
///
/// Applies `W_new = W + (alpha / rank) * B @ A` element-wise in the base
/// weight's dtype.
///
/// # Errors
///
/// Returns [`MilError::Validation`] if shapes are incompatible or dtypes
/// are unsupported.
pub fn merge_lora(
    base_data: &mut Vec<u8>,
    base_shape: &[usize],
    base_dtype: ScalarType,
    adapter: &LoraAdapter,
) -> Result<()> {
    if base_shape.len() != 2 {
        return Err(MilError::Validation(format!(
            "LoRA merge requires 2-D base weight, got shape {:?} for '{}'",
            base_shape, adapter.base_name
        )));
    }

    let [out_features, in_features] = [base_shape[0], base_shape[1]];
    let [rank, a_in] = adapter.a_shape;
    let [b_out, b_rank] = adapter.b_shape;

    if a_in != in_features {
        return Err(MilError::Validation(format!(
            "LoRA A in_features ({a_in}) != base in_features ({in_features}) for '{}'",
            adapter.base_name
        )));
    }
    if b_out != out_features {
        return Err(MilError::Validation(format!(
            "LoRA B out_features ({b_out}) != base out_features ({out_features}) for '{}'",
            adapter.base_name
        )));
    }
    if b_rank != rank {
        return Err(MilError::Validation(format!(
            "LoRA rank mismatch: A has rank {rank}, B has rank {b_rank} for '{}'",
            adapter.base_name
        )));
    }

    if base_dtype != adapter.dtype {
        return Err(MilError::Validation(format!(
            "dtype mismatch: base is {:?}, adapter is {:?} for '{}'",
            base_dtype, adapter.dtype, adapter.base_name
        )));
    }

    let scale = adapter.alpha.unwrap_or(rank as f64) / rank as f64;

    // Validate buffer sizes before accessing raw bytes.
    let dtype_size = match base_dtype {
        ScalarType::Float32 => 4,
        ScalarType::Float16 => 2,
        other => {
            return Err(MilError::Validation(format!(
                "LoRA merge unsupported for dtype {:?} on '{}'",
                other, adapter.base_name
            )));
        }
    };

    let expected_base_bytes = out_features * in_features * dtype_size;
    if base_data.len() < expected_base_bytes {
        return Err(MilError::Validation(format!(
            "base weight buffer too small: got {} bytes, expected {} for '{}'",
            base_data.len(),
            expected_base_bytes,
            adapter.base_name
        )));
    }

    let expected_a_bytes = rank * a_in * dtype_size;
    if adapter.a_data.len() < expected_a_bytes {
        return Err(MilError::Validation(format!(
            "LoRA A buffer too small: got {} bytes, expected {} for '{}'",
            adapter.a_data.len(),
            expected_a_bytes,
            adapter.base_name
        )));
    }

    let expected_b_bytes = b_out * b_rank * dtype_size;
    if adapter.b_data.len() < expected_b_bytes {
        return Err(MilError::Validation(format!(
            "LoRA B buffer too small: got {} bytes, expected {} for '{}'",
            adapter.b_data.len(),
            expected_b_bytes,
            adapter.base_name
        )));
    }

    match base_dtype {
        ScalarType::Float32 => merge_f32(base_data, out_features, in_features, adapter, scale),
        ScalarType::Float16 => merge_f16(base_data, out_features, in_features, adapter, scale),
        other => {
            return Err(MilError::Validation(format!(
                "LoRA merge unsupported for dtype {:?} on '{}'",
                other, adapter.base_name
            )));
        }
    }

    Ok(())
}

/// Returns the set of initializer names that belong to LoRA adapters and
/// should be removed after merging.
pub fn lora_initializer_names(adapters: &[LoraAdapter]) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for adapter in adapters {
        // Reverse-engineer the prefix from base_name.
        let prefix = adapter
            .base_name
            .strip_suffix(".weight")
            .unwrap_or(&adapter.base_name);
        names.insert(format!("{prefix}.lora_A.weight"));
        names.insert(format!("{prefix}.lora_B.weight"));
        names.insert(format!("{prefix}.lora_alpha"));
    }
    names
}

/// LoRA adapter weight matrices for merging into base weights.
pub struct LoraWeights<'a> {
    /// LoRA A matrix bytes (row-major, `[rank, in_features]`).
    pub lora_a: &'a [u8],
    /// Shape of A: `[rank, in_features]`.
    pub lora_a_shape: &'a [usize],
    /// LoRA B matrix bytes (row-major, `[out_features, rank]`).
    pub lora_b: &'a [u8],
    /// Shape of B: `[out_features, rank]`.
    pub lora_b_shape: &'a [usize],
    /// Element data type (must be `Float32` or `Float16`).
    pub dtype: ScalarType,
    /// Optional scaling factor. When `None`, defaults to `rank` (scale = 1.0).
    pub alpha: Option<f64>,
}

/// Format-agnostic LoRA merge kernel.
///
/// Applies `W_new = W + (alpha / rank) * B @ A` to the base weight buffer
/// in-place. This function operates on raw byte buffers and does not depend
/// on any specific model format (ONNX, SafeTensors, etc.).
pub fn merge_lora_weights(
    base: &mut Vec<u8>,
    base_shape: &[usize],
    lora: &LoraWeights<'_>,
) -> Result<()> {
    if base_shape.len() != 2 {
        return Err(MilError::Validation(format!(
            "LoRA merge requires 2-D base weight, got shape {base_shape:?}"
        )));
    }
    if lora.lora_a_shape.len() != 2 || lora.lora_b_shape.len() != 2 {
        return Err(MilError::Validation("LoRA A and B must be 2-D".into()));
    }

    let out_features = base_shape[0];
    let in_features = base_shape[1];
    let rank = lora.lora_a_shape[0];
    let a_in = lora.lora_a_shape[1];
    let b_out = lora.lora_b_shape[0];
    let b_rank = lora.lora_b_shape[1];

    if a_in != in_features {
        return Err(MilError::Validation(format!(
            "LoRA A in_features ({a_in}) != base in_features ({in_features})"
        )));
    }
    if b_out != out_features {
        return Err(MilError::Validation(format!(
            "LoRA B out_features ({b_out}) != base out_features ({out_features})"
        )));
    }
    if b_rank != rank {
        return Err(MilError::Validation(format!(
            "LoRA rank mismatch: A has rank {rank}, B has rank {b_rank}"
        )));
    }

    let scale = lora.alpha.unwrap_or(rank as f64) / rank as f64;

    let dtype_size = match lora.dtype {
        ScalarType::Float32 => 4,
        ScalarType::Float16 => 2,
        other => {
            return Err(MilError::Validation(format!(
                "LoRA merge unsupported for dtype {other:?}"
            )));
        }
    };

    let expected_base = out_features * in_features * dtype_size;
    if base.len() < expected_base {
        return Err(MilError::Validation(format!(
            "base buffer too small: got {} bytes, expected {expected_base}",
            base.len()
        )));
    }
    let expected_a = rank * a_in * dtype_size;
    if lora.lora_a.len() < expected_a {
        return Err(MilError::Validation(format!(
            "LoRA A buffer too small: got {} bytes, expected {expected_a}",
            lora.lora_a.len()
        )));
    }
    let expected_b = b_out * b_rank * dtype_size;
    if lora.lora_b.len() < expected_b {
        return Err(MilError::Validation(format!(
            "LoRA B buffer too small: got {} bytes, expected {expected_b}",
            lora.lora_b.len()
        )));
    }

    match lora.dtype {
        ScalarType::Float32 => {
            merge_f32_raw(
                base,
                out_features,
                in_features,
                lora.lora_a,
                lora.lora_b,
                rank,
                scale as f32,
            );
        }
        ScalarType::Float16 => {
            merge_f16_raw(
                base,
                out_features,
                in_features,
                lora.lora_a,
                lora.lora_b,
                rank,
                scale as f32,
            );
        }
        _ => unreachable!(),
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Read a single scalar from raw bytes.
pub fn scalar_from_bytes(data: &[u8], dtype: ScalarType) -> Option<f64> {
    match dtype {
        ScalarType::Float32 if data.len() >= 4 => {
            Some(f32::from_le_bytes(data[..4].try_into().ok()?) as f64)
        }
        ScalarType::Float64 if data.len() >= 8 => {
            Some(f64::from_le_bytes(data[..8].try_into().ok()?))
        }
        ScalarType::Float16 if data.len() >= 2 => {
            let bits = u16::from_le_bytes(data[..2].try_into().ok()?);
            Some(f16::from_bits(bits).to_f64())
        }
        ScalarType::Int32 if data.len() >= 4 => {
            Some(i32::from_le_bytes(data[..4].try_into().ok()?) as f64)
        }
        ScalarType::Int64 if data.len() >= 8 => {
            Some(i64::from_le_bytes(data[..8].try_into().ok()?) as f64)
        }
        _ => None,
    }
}

/// Merge in fp32: W += scale * B @ A.
fn merge_f32(
    base_data: &mut Vec<u8>,
    out_features: usize,
    in_features: usize,
    adapter: &LoraAdapter,
    scale: f64,
) {
    let rank = adapter.a_shape[0];
    let scale = scale as f32;

    let a = bytes_to_f32(&adapter.a_data);
    let b = bytes_to_f32(&adapter.b_data);
    let mut w = bytes_to_f32(base_data);

    // W[i, j] += scale * sum_r(B[i, r] * A[r, j])
    for i in 0..out_features {
        for j in 0..in_features {
            let mut dot = 0.0f32;
            for r in 0..rank {
                dot += b[i * rank + r] * a[r * in_features + j];
            }
            w[i * in_features + j] += scale * dot;
        }
    }

    *base_data = f32_to_bytes(&w);
}

/// Merge in fp16: promote to f32 for the matmul, then store back as fp16.
fn merge_f16(
    base_data: &mut Vec<u8>,
    out_features: usize,
    in_features: usize,
    adapter: &LoraAdapter,
    scale: f64,
) {
    let rank = adapter.a_shape[0];
    let scale = scale as f32;

    let a = bytes_to_f16_as_f32(&adapter.a_data);
    let b = bytes_to_f16_as_f32(&adapter.b_data);
    let mut w = bytes_to_f16_as_f32(base_data);

    for i in 0..out_features {
        for j in 0..in_features {
            let mut dot = 0.0f32;
            for r in 0..rank {
                dot += b[i * rank + r] * a[r * in_features + j];
            }
            w[i * in_features + j] += scale * dot;
        }
    }

    *base_data = f32_to_f16_bytes(&w);
}

/// Format-agnostic fp32 merge: W += scale * B @ A, operating on raw byte slices.
fn merge_f32_raw(
    base_data: &mut Vec<u8>,
    out_features: usize,
    in_features: usize,
    lora_a: &[u8],
    lora_b: &[u8],
    rank: usize,
    scale: f32,
) {
    let a = bytes_to_f32(lora_a);
    let b = bytes_to_f32(lora_b);
    let mut w = bytes_to_f32(base_data);

    for i in 0..out_features {
        for j in 0..in_features {
            let mut dot = 0.0f32;
            for r in 0..rank {
                dot += b[i * rank + r] * a[r * in_features + j];
            }
            w[i * in_features + j] += scale * dot;
        }
    }

    *base_data = f32_to_bytes(&w);
}

/// Format-agnostic fp16 merge: W += scale * B @ A, operating on raw byte slices.
fn merge_f16_raw(
    base_data: &mut Vec<u8>,
    out_features: usize,
    in_features: usize,
    lora_a: &[u8],
    lora_b: &[u8],
    rank: usize,
    scale: f32,
) {
    let a = bytes_to_f16_as_f32(lora_a);
    let b = bytes_to_f16_as_f32(lora_b);
    let mut w = bytes_to_f16_as_f32(base_data);

    for i in 0..out_features {
        for j in 0..in_features {
            let mut dot = 0.0f32;
            for r in 0..rank {
                dot += b[i * rank + r] * a[r * in_features + j];
            }
            w[i * in_features + j] += scale * dot;
        }
    }

    *base_data = f32_to_f16_bytes(&w);
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn f32_to_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_f16_as_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes(c.try_into().unwrap());
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| f16::from_f32(*v).to_le_bytes())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build raw f32 bytes from a slice.
    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        f32_to_bytes(vals)
    }

    /// Build raw f16 bytes from f32 values.
    fn f16_bytes(vals: &[f32]) -> Vec<u8> {
        f32_to_f16_bytes(vals)
    }

    /// Read f32 values back from raw bytes.
    fn read_f32(data: &[u8]) -> Vec<f32> {
        bytes_to_f32(data)
    }

    /// Read f16 values (as f32) back from raw bytes.
    fn read_f16(data: &[u8]) -> Vec<f32> {
        bytes_to_f16_as_f32(data)
    }

    // -----------------------------------------------------------------------
    // Weight merge math
    // -----------------------------------------------------------------------

    #[test]
    fn merge_identity_scale() {
        // W = [[1, 2], [3, 4]]  (2×2, fp32)
        // A = [[1, 0], [0, 1]]  (2×2, rank=2)
        // B = [[1, 0], [0, 1]]  (2×2)
        // alpha = 2.0, rank = 2 → scale = 1.0
        // W_new = W + 1.0 * I @ I = W + I = [[2, 2], [3, 5]]
        let mut base = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let adapter = LoraAdapter {
            base_name: "test.weight".into(),
            a_data: f32_bytes(&[1.0, 0.0, 0.0, 1.0]),
            a_shape: [2, 2],
            b_data: f32_bytes(&[1.0, 0.0, 0.0, 1.0]),
            b_shape: [2, 2],
            dtype: ScalarType::Float32,
            alpha: Some(2.0),
        };

        merge_lora(&mut base, &[2, 2], ScalarType::Float32, &adapter).unwrap();
        let result = read_f32(&base);
        assert_eq!(result, vec![2.0, 2.0, 3.0, 5.0]);
    }

    #[test]
    fn merge_with_alpha_scaling() {
        // W = zeros(2×3), A = ones(1×3), B = ones(2×1)
        // alpha = 4.0, rank = 1 → scale = 4.0
        // B@A = [[1,1,1],[1,1,1]]
        // W_new = 0 + 4 * [[1,1,1],[1,1,1]] = [[4,4,4],[4,4,4]]
        let mut base = f32_bytes(&[0.0; 6]);
        let adapter = LoraAdapter {
            base_name: "test.weight".into(),
            a_data: f32_bytes(&[1.0, 1.0, 1.0]),
            a_shape: [1, 3],
            b_data: f32_bytes(&[1.0, 1.0]),
            b_shape: [2, 1],
            dtype: ScalarType::Float32,
            alpha: Some(4.0),
        };

        merge_lora(&mut base, &[2, 3], ScalarType::Float32, &adapter).unwrap();
        let result = read_f32(&base);
        assert_eq!(result, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn merge_default_alpha() {
        // When alpha is None, scale = rank / rank = 1.0
        // W = zeros(2×2), A = [[2, 0]], B = [[1], [1]]
        // rank=1, scale=1.0
        // B@A = [[2, 0], [2, 0]]
        let mut base = f32_bytes(&[0.0; 4]);
        let adapter = LoraAdapter {
            base_name: "test.weight".into(),
            a_data: f32_bytes(&[2.0, 0.0]),
            a_shape: [1, 2],
            b_data: f32_bytes(&[1.0, 1.0]),
            b_shape: [2, 1],
            dtype: ScalarType::Float32,
            alpha: None,
        };

        merge_lora(&mut base, &[2, 2], ScalarType::Float32, &adapter).unwrap();
        let result = read_f32(&base);
        assert_eq!(result, vec![2.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn merge_fp16() {
        // Same as identity test but in fp16.
        let mut base = f16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let adapter = LoraAdapter {
            base_name: "test.weight".into(),
            a_data: f16_bytes(&[1.0, 0.0, 0.0, 1.0]),
            a_shape: [2, 2],
            b_data: f16_bytes(&[1.0, 0.0, 0.0, 1.0]),
            b_shape: [2, 2],
            dtype: ScalarType::Float16,
            alpha: Some(2.0),
        };

        merge_lora(&mut base, &[2, 2], ScalarType::Float16, &adapter).unwrap();
        let result = read_f16(&base);
        assert_eq!(result, vec![2.0, 2.0, 3.0, 5.0]);
    }

    #[test]
    fn merge_non_square() {
        // W = zeros(3×4), A = ones(2×4), B = ones(3×2)
        // alpha = 2.0, rank = 2, scale = 1.0
        // B@A = [[2,2,2,2],[2,2,2,2],[2,2,2,2]]
        let mut base = f32_bytes(&[0.0; 12]);
        let adapter = LoraAdapter {
            base_name: "test.weight".into(),
            a_data: f32_bytes(&[1.0; 8]),
            a_shape: [2, 4],
            b_data: f32_bytes(&[1.0; 6]),
            b_shape: [3, 2],
            dtype: ScalarType::Float32,
            alpha: Some(2.0),
        };

        merge_lora(&mut base, &[3, 4], ScalarType::Float32, &adapter).unwrap();
        let result = read_f32(&base);
        assert_eq!(result, vec![2.0; 12]);
    }

    #[test]
    fn merge_shape_mismatch_a() {
        let mut base = f32_bytes(&[0.0; 4]);
        let adapter = LoraAdapter {
            base_name: "test.weight".into(),
            a_data: f32_bytes(&[1.0; 3]),
            a_shape: [1, 3], // in_features=3 but base is 2×2
            b_data: f32_bytes(&[1.0; 2]),
            b_shape: [2, 1],
            dtype: ScalarType::Float32,
            alpha: None,
        };

        let err = merge_lora(&mut base, &[2, 2], ScalarType::Float32, &adapter);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("in_features"));
    }

    #[test]
    fn merge_shape_mismatch_b() {
        let mut base = f32_bytes(&[0.0; 4]);
        let adapter = LoraAdapter {
            base_name: "test.weight".into(),
            a_data: f32_bytes(&[1.0; 2]),
            a_shape: [1, 2],
            b_data: f32_bytes(&[1.0; 3]),
            b_shape: [3, 1], // out_features=3 but base is 2×2
            dtype: ScalarType::Float32,
            alpha: None,
        };

        let err = merge_lora(&mut base, &[2, 2], ScalarType::Float32, &adapter);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("out_features"));
    }

    #[test]
    fn merge_rank_mismatch() {
        let mut base = f32_bytes(&[0.0; 4]);
        let adapter = LoraAdapter {
            base_name: "test.weight".into(),
            a_data: f32_bytes(&[1.0; 4]),
            a_shape: [2, 2],
            b_data: f32_bytes(&[1.0; 6]),
            b_shape: [2, 3], // rank=3 vs A's rank=2
            dtype: ScalarType::Float32,
            alpha: None,
        };

        let err = merge_lora(&mut base, &[2, 2], ScalarType::Float32, &adapter);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("rank mismatch"));
    }

    #[test]
    fn merge_3d_base_rejected() {
        let mut base = f32_bytes(&[0.0; 8]);
        let adapter = LoraAdapter {
            base_name: "test.weight".into(),
            a_data: f32_bytes(&[1.0; 2]),
            a_shape: [1, 2],
            b_data: f32_bytes(&[1.0; 2]),
            b_shape: [2, 1],
            dtype: ScalarType::Float32,
            alpha: None,
        };

        let err = merge_lora(&mut base, &[2, 2, 2], ScalarType::Float32, &adapter);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("2-D"));
    }

    #[test]
    fn merge_dtype_mismatch() {
        let mut base = f32_bytes(&[0.0; 4]);
        let adapter = LoraAdapter {
            base_name: "test.weight".into(),
            a_data: f16_bytes(&[1.0; 2]),
            a_shape: [1, 2],
            b_data: f16_bytes(&[1.0; 2]),
            b_shape: [2, 1],
            dtype: ScalarType::Float16,
            alpha: None,
        };

        let err = merge_lora(&mut base, &[2, 2], ScalarType::Float32, &adapter);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("dtype mismatch"));
    }

    // -----------------------------------------------------------------------
    // LoRA detection from initializer names
    // -----------------------------------------------------------------------

    #[test]
    fn detect_lora_pair() {
        let inits = vec![
            (
                "model.attn.q_proj.lora_A.weight".into(),
                f32_bytes(&[1.0; 6]),
                vec![2, 3],
                ScalarType::Float32,
            ),
            (
                "model.attn.q_proj.lora_B.weight".into(),
                f32_bytes(&[1.0; 8]),
                vec![4, 2],
                ScalarType::Float32,
            ),
            (
                "model.attn.q_proj.weight".into(),
                f32_bytes(&[0.0; 12]),
                vec![4, 3],
                ScalarType::Float32,
            ),
        ];

        let adapters = detect_lora_adapters(&inits);
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].base_name, "model.attn.q_proj.weight");
        assert_eq!(adapters[0].a_shape, [2, 3]);
        assert_eq!(adapters[0].b_shape, [4, 2]);
        assert!(adapters[0].alpha.is_none());
    }

    #[test]
    fn detect_with_alpha() {
        let inits = vec![
            (
                "layer.lora_A.weight".into(),
                f32_bytes(&[1.0; 2]),
                vec![1, 2],
                ScalarType::Float32,
            ),
            (
                "layer.lora_B.weight".into(),
                f32_bytes(&[1.0; 2]),
                vec![2, 1],
                ScalarType::Float32,
            ),
            (
                "layer.lora_alpha".into(),
                f32_bytes(&[8.0]),
                vec![1],
                ScalarType::Float32,
            ),
        ];

        let adapters = detect_lora_adapters(&inits);
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].alpha, Some(8.0));
    }

    #[test]
    fn detect_skips_unpaired() {
        let inits = vec![(
            "layer.lora_A.weight".into(),
            f32_bytes(&[1.0; 2]),
            vec![1, 2],
            ScalarType::Float32,
        )];

        let adapters = detect_lora_adapters(&inits);
        assert!(adapters.is_empty(), "unpaired A should be skipped");
    }

    #[test]
    fn detect_skips_dtype_mismatch() {
        let inits = vec![
            (
                "layer.lora_A.weight".into(),
                f32_bytes(&[1.0; 2]),
                vec![1, 2],
                ScalarType::Float32,
            ),
            (
                "layer.lora_B.weight".into(),
                f16_bytes(&[1.0; 2]),
                vec![2, 1],
                ScalarType::Float16,
            ),
        ];

        let adapters = detect_lora_adapters(&inits);
        assert!(adapters.is_empty(), "dtype mismatch should be skipped");
    }

    #[test]
    fn detect_multiple_adapters() {
        let inits = vec![
            (
                "q.lora_A.weight".into(),
                f32_bytes(&[1.0; 2]),
                vec![1, 2],
                ScalarType::Float32,
            ),
            (
                "q.lora_B.weight".into(),
                f32_bytes(&[1.0; 2]),
                vec![2, 1],
                ScalarType::Float32,
            ),
            (
                "v.lora_A.weight".into(),
                f32_bytes(&[1.0; 4]),
                vec![2, 2],
                ScalarType::Float32,
            ),
            (
                "v.lora_B.weight".into(),
                f32_bytes(&[1.0; 6]),
                vec![3, 2],
                ScalarType::Float32,
            ),
        ];

        let adapters = detect_lora_adapters(&inits);
        assert_eq!(adapters.len(), 2);
    }

    #[test]
    fn lora_names_to_remove() {
        let adapters = vec![LoraAdapter {
            base_name: "model.layer.weight".into(),
            a_data: vec![],
            a_shape: [1, 2],
            b_data: vec![],
            b_shape: [2, 1],
            dtype: ScalarType::Float32,
            alpha: Some(4.0),
        }];

        let names = lora_initializer_names(&adapters);
        assert!(names.contains("model.layer.lora_A.weight"));
        assert!(names.contains("model.layer.lora_B.weight"));
        assert!(names.contains("model.layer.lora_alpha"));
    }
}
