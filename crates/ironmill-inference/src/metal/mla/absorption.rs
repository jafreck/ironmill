//! Weight absorption for Multi-Head Latent Attention.
//!
//! MLA models (DeepSeek-V2/V3) store compressed KV latents and use
//! up-projection weights to reconstruct K and V at inference time.
//! Weight absorption fuses these up-projections into the Q and O
//! weights at model load time, so the inference loop can work directly
//! with the compressed latents — no runtime up-projection needed.
//!
//! Per-head operations:
//!
//!   W_q_absorbed_h = W_uk_h^T · W_q_nope_h
//!     shape: [kv_latent_dim, hidden_size]
//!
//!   W_o_absorbed_h = W_o_h · W_uv_h
//!     shape: [hidden_size, kv_latent_dim]
//!
//! The absorbed weights replace the original Q and O projections. The Q
//! rope portion is unaffected and handled separately during inference.

use half::f16;
use ironmill_metal_sys::{MetalDevice, StorageMode};
use mil_rs::weights::{MlaConfig, WeightProvider};

use crate::metal::error::MetalError;
use crate::metal::weights::{MetalWeights, WeightBuffer};

/// Perform weight absorption for MLA layers.
///
/// Fuses the KV up-projection weights into Q and O projections:
///   - Q absorption: eliminates runtime K up-projection by pre-multiplying
///     the nope portion of Q with W_uk^T.
///   - O absorption: eliminates runtime V up-projection by pre-multiplying
///     W_o with W_uv.
///
/// After absorption, attention operates directly on compressed latent
/// vectors (dimension `kv_latent_dim`) instead of full K/V vectors.
///
/// # Arguments
///
/// * `w_q`    — Q projection weights `[num_heads * qk_dim, hidden_size]`
///   where `qk_dim = qk_nope_head_dim + qk_rope_head_dim`.
/// * `w_uk`   — K up-projection weights `[num_heads * qk_nope_head_dim, kv_latent_dim]`.
/// * `w_o`    — Output projection weights `[hidden_size, num_heads * v_head_dim]`.
/// * `w_uv`   — V up-projection weights `[num_heads * v_head_dim, kv_latent_dim]`.
/// * `config` — MLA configuration with dimension parameters.
///
/// # Returns
///
/// `(w_q_absorbed, w_o_absorbed)`:
/// * `w_q_absorbed` — `[num_heads * kv_latent_dim, hidden_size]` FP16.
/// * `w_o_absorbed` — `[hidden_size, num_heads * kv_latent_dim]` FP16.
pub fn absorb_weights(
    w_q: &[f16],
    w_uk: &[f16],
    w_o: &[f16],
    w_uv: &[f16],
    config: &MlaConfig,
) -> (Vec<f16>, Vec<f16>) {
    let nh = config.num_heads;
    let nope = config.qk_nope_head_dim;
    let rope = config.qk_rope_head_dim;
    let qk_dim = nope + rope;
    let kv_lat = config.kv_latent_dim;
    let v_dim = config.v_head_dim;

    // Derive hidden_size from the Q weight shape:
    // w_q is [num_heads * qk_dim, hidden_size]
    let hidden_size = w_q.len() / (nh * qk_dim);
    assert_eq!(
        w_q.len(),
        nh * qk_dim * hidden_size,
        "w_q size mismatch: expected {} * {} * {}, got {}",
        nh,
        qk_dim,
        hidden_size,
        w_q.len()
    );

    // ── Q absorption ───────────────────────────────────────────
    // Per head h:
    //   W_q_nope_h: [nope, hidden_size]  (rows from w_q)
    //   W_uk_h:     [nope, kv_lat]       (rows from w_uk)
    //   W_q_absorbed_h = W_uk_h^T · W_q_nope_h → [kv_lat, hidden_size]
    let mut w_q_absorbed = vec![f16::ZERO; nh * kv_lat * hidden_size];

    for h in 0..nh {
        let q_nope_offset = h * qk_dim; // first row of nope for head h
        let uk_offset = h * nope; // first row of uk for head h
        let out_offset = h * kv_lat; // first row of absorbed for head h

        // W_uk_h^T · W_q_nope_h = [kv_lat, hidden_size]
        for lat_row in 0..kv_lat {
            for col in 0..hidden_size {
                let mut acc = 0.0f32;
                for k in 0..nope {
                    // W_uk_h^T[lat_row, k] = W_uk_h[k, lat_row]
                    let uk_val = w_uk[(uk_offset + k) * kv_lat + lat_row].to_f32();
                    // W_q_nope_h[k, col]
                    let q_val = w_q[(q_nope_offset + k) * hidden_size + col].to_f32();
                    acc += uk_val * q_val;
                }
                w_q_absorbed[(out_offset + lat_row) * hidden_size + col] = f16::from_f32(acc);
            }
        }
    }

    // ── O absorption ───────────────────────────────────────────
    // Per head h:
    //   W_o_h:  [hidden_size, v_dim]  (columns from w_o)
    //   W_uv_h: [v_dim, kv_lat]      (rows from w_uv)
    //   W_o_absorbed_h = W_o_h · W_uv_h → [hidden_size, kv_lat]
    let o_in_features = nh * v_dim;
    let mut w_o_absorbed = vec![f16::ZERO; hidden_size * nh * kv_lat];

    for h in 0..nh {
        let o_col_offset = h * v_dim; // first column of w_o for head h
        let uv_offset = h * v_dim; // first row of w_uv for head h
        let out_col_offset = h * kv_lat; // first column of absorbed for head h

        // W_o_h · W_uv_h = [hidden_size, kv_lat]
        for row in 0..hidden_size {
            for lat_col in 0..kv_lat {
                let mut acc = 0.0f32;
                for k in 0..v_dim {
                    // W_o_h[row, k] = W_o[row, o_col_offset + k]
                    let o_val = w_o[row * o_in_features + o_col_offset + k].to_f32();
                    // W_uv_h[k, lat_col]
                    let uv_val = w_uv[(uv_offset + k) * kv_lat + lat_col].to_f32();
                    acc += o_val * uv_val;
                }
                w_o_absorbed[row * (nh * kv_lat) + out_col_offset + lat_col] = f16::from_f32(acc);
            }
        }
    }

    (w_q_absorbed, w_o_absorbed)
}

/// Absorb MLA up-projection weights into Q and O projections at load time.
///
/// For each transformer layer, reads W_uk and W_uv from the weight provider,
/// reads the current Q and O weights from the Metal buffers, performs the
/// absorption matrix multiplies, and replaces the Q and O Metal buffers with
/// the absorbed versions.
#[allow(dead_code)]
pub(crate) fn absorb_mla_weights(
    device: &MetalDevice,
    weights: &mut MetalWeights,
    mla: &MlaConfig,
    hidden_size: usize,
    provider: &dyn WeightProvider,
) -> Result<(), MetalError> {
    let num_layers = weights.layers.len();
    let qk_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim;
    let q_total = mla.num_heads * qk_dim * hidden_size;
    let o_total = hidden_size * mla.num_heads * mla.v_head_dim;
    let uk_total = mla.num_heads * mla.qk_nope_head_dim * mla.kv_latent_dim;
    let uv_total = mla.num_heads * mla.v_head_dim * mla.kv_latent_dim;

    for layer_idx in 0..num_layers {
        let prefix = format!("model.layers.{layer_idx}");

        // Load W_uk and W_uv from the provider.
        let uk_name = format!("{prefix}.self_attn.kv_b_proj.weight");
        let _uv_name = uk_name.clone(); // In DeepSeek, kv_b_proj contains both UK and UV

        // Read current Q and O weights from Metal buffers.
        let q_bytes = read_f16_buffer(&weights.layers[layer_idx].q_proj, q_total)?;
        let o_bytes = read_f16_buffer(&weights.layers[layer_idx].o_proj, o_total)?;

        // Convert to f16 slices.
        let w_q: Vec<f16> = q_bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]))
            .collect();
        let w_o: Vec<f16> = o_bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]))
            .collect();

        // Load up-projection weights. These may be combined in kv_b_proj;
        // the first num_heads * qk_nope_head_dim rows are W_uk, the next
        // num_heads * v_head_dim rows are W_uv.
        let (w_uk, w_uv) = if provider.has_tensor(&uk_name) {
            let tensor = provider
                .tensor(&uk_name)
                .map_err(|e| MetalError::WeightLoading(format!("{uk_name}: {e}")))?;
            let all_f16: Vec<f16> = tensor
                .data
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]))
                .collect();
            // Split: first uk_total elements are W_uk, next uv_total are W_uv.
            if all_f16.len() >= uk_total + uv_total {
                (
                    all_f16[..uk_total].to_vec(),
                    all_f16[uk_total..uk_total + uv_total].to_vec(),
                )
            } else {
                // Fallback: separate tensors
                if all_f16.len() < uk_total {
                    return Err(MetalError::WeightLoading(format!(
                        "kv_b_proj tensor too small: expected {} elements, got {}",
                        uk_total,
                        all_f16.len()
                    )));
                }
                let w_uk_vec = all_f16[..uk_total].to_vec();
                let uv_tensor = provider
                    .tensor(&format!("{prefix}.self_attn.v_b_proj.weight"))
                    .map_err(|e| MetalError::WeightLoading(format!("v_b_proj: {e}")))?;
                let w_uv_vec: Vec<f16> = uv_tensor
                    .data
                    .chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]))
                    .collect();
                (w_uk_vec, w_uv_vec)
            }
        } else {
            return Err(MetalError::WeightLoading(format!(
                "MLA up-projection weight not found: {uk_name}"
            )));
        };

        // Perform absorption.
        let (q_absorbed, o_absorbed) = absorb_weights(&w_q, &w_uk, &w_o, &w_uv, mla);

        // Create new Metal buffers with absorbed weights.
        let q_absorbed_bytes: Vec<u8> = q_absorbed.iter().flat_map(|v| v.to_le_bytes()).collect();
        let o_absorbed_bytes: Vec<u8> = o_absorbed.iter().flat_map(|v| v.to_le_bytes()).collect();

        let q_buf = device
            .create_buffer_with_data(&q_absorbed_bytes, StorageMode::Shared)
            .map_err(MetalError::Metal)?;
        let o_buf = device
            .create_buffer_with_data(&o_absorbed_bytes, StorageMode::Shared)
            .map_err(MetalError::Metal)?;

        // Replace the Q and O projection weights with absorbed versions.
        weights.layers[layer_idx].q_proj = WeightBuffer::Dense {
            buf: Some(q_buf),
            packed: None, // Absorption changes dimensions; re-packing can be added later.
        };
        weights.layers[layer_idx].o_proj = WeightBuffer::Dense {
            buf: Some(o_buf),
            packed: None,
        };
    }

    Ok(())
}

/// Read FP16 data from a WeightBuffer, returning raw bytes.
///
/// For Dense weights, reads directly from the underlying Metal buffer.
/// Returns an error for quantized weights (MLA absorption requires dense
/// weights as input).
#[allow(dead_code)]
fn read_f16_buffer(weight: &WeightBuffer, num_elements: usize) -> Result<Vec<u8>, MetalError> {
    let buf = weight.as_dense().map_err(|e| {
        MetalError::WeightLoading(format!(
            "MLA absorption requires dense weights, got quantized: {e}"
        ))
    })?;
    let byte_count = num_elements * 2;
    let mut data = vec![0u8; byte_count];
    buf.read_bytes(&mut data, 0).map_err(MetalError::Metal)?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small-scale absorption test verifying the matrix product is correct.
    ///
    /// Uses a 2-head model with tiny dimensions so the expected results
    /// can be verified by hand.
    #[test]
    fn absorb_weights_small() {
        // Config: 2 heads, nope=2, rope=1, kv_latent=3, v_dim=2
        let config = MlaConfig::new(3, 0, 2, 2, 1, 2);

        let hidden_size = 4;
        let qk_dim = 3; // nope + rope
        let nh = 2;
        let nope = 2;
        let kv_lat = 3;
        let v_dim = 2;

        // w_q: [nh * qk_dim, hidden_size] = [6, 4]
        // Fill with identity-like values for easy verification.
        let mut w_q = vec![f16::ZERO; nh * qk_dim * hidden_size];
        for i in 0..w_q.len() {
            w_q[i] = f16::from_f32((i + 1) as f32 * 0.1);
        }

        // w_uk: [nh * nope, kv_lat] = [4, 3]
        let mut w_uk = vec![f16::ZERO; nh * nope * kv_lat];
        for i in 0..w_uk.len() {
            w_uk[i] = f16::from_f32((i + 1) as f32 * 0.01);
        }

        // w_o: [hidden_size, nh * v_dim] = [4, 4]
        let mut w_o = vec![f16::ZERO; hidden_size * nh * v_dim];
        for i in 0..w_o.len() {
            w_o[i] = f16::from_f32((i + 1) as f32 * 0.05);
        }

        // w_uv: [nh * v_dim, kv_lat] = [4, 3]
        let mut w_uv = vec![f16::ZERO; nh * v_dim * kv_lat];
        for i in 0..w_uv.len() {
            w_uv[i] = f16::from_f32((i + 1) as f32 * 0.02);
        }

        let (q_abs, o_abs) = absorb_weights(&w_q, &w_uk, &w_o, &w_uv, &config);

        // Check output shapes.
        assert_eq!(q_abs.len(), nh * kv_lat * hidden_size); // [6, 4]
        assert_eq!(o_abs.len(), hidden_size * nh * kv_lat); // [4, 6]

        // Verify Q absorption for head 0 by manual computation:
        // W_q_nope_h0: rows 0..2 of w_q → [2, 4]
        // W_uk_h0: rows 0..2 of w_uk → [2, 3]
        // W_q_absorbed_h0 = W_uk_h0^T · W_q_nope_h0 → [3, 4]
        //
        // W_uk_h0^T is [3, 2] = transpose of rows 0..2 of [4, 3]
        // W_q_nope_h0 is [2, 4] = rows 0..1 of [6, 4]
        //
        // Result[0, 0] = uk_h0^T[0,0] * q_nope[0,0] + uk_h0^T[0,1] * q_nope[1,0]
        //              = w_uk[0*3+0] * w_q[0*4+0] + w_uk[1*3+0] * w_q[1*4+0]
        //              = 0.01 * 0.1 + 0.04 * 0.5
        //              = 0.001 + 0.020 = 0.021
        let expected = 0.01f32 * 0.1 + 0.04 * 0.5;
        let actual = q_abs[0].to_f32();
        assert!(
            (actual - expected).abs() < 0.01,
            "Q absorption mismatch: expected {expected}, got {actual}"
        );

        // Verify non-zero output (sanity check).
        assert!(
            q_abs.iter().any(|v| v.to_f32().abs() > 1e-6),
            "Q absorbed weights are all zero"
        );
        assert!(
            o_abs.iter().any(|v| v.to_f32().abs() > 1e-6),
            "O absorbed weights are all zero"
        );
    }

    /// Verify that absorption with identity-like up-projection preserves
    /// the original weight values (modulo dimension reshaping).
    #[test]
    fn absorb_with_identity_uk() {
        // 1 head, nope=2, rope=0, kv_lat=2, v_dim=2
        // When kv_lat == nope and W_uk is identity, the absorbed Q should
        // equal the original Q nope portion.
        let config = MlaConfig::new(2, 0, 1, 2, 0, 2);

        let hidden_size = 3;

        // w_q: [2, 3] — the nope portion (no rope)
        let w_q: Vec<f16> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();

        // w_uk: [2, 2] — identity matrix
        let w_uk: Vec<f16> = [1.0, 0.0, 0.0, 1.0]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();

        // w_o and w_uv are irrelevant for Q test
        let w_o: Vec<f16> = vec![f16::from_f32(1.0); hidden_size * 2];
        let w_uv: Vec<f16> = [1.0, 0.0, 0.0, 1.0]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();

        let (q_abs, _o_abs) = absorb_weights(&w_q, &w_uk, &w_o, &w_uv, &config);

        // W_uk^T · W_q = I · W_q = W_q
        assert_eq!(q_abs.len(), 2 * 3);
        for i in 0..6 {
            let expected = w_q[i].to_f32();
            let actual = q_abs[i].to_f32();
            assert!(
                (actual - expected).abs() < 0.01,
                "element {i}: expected {expected}, got {actual}"
            );
        }
    }
}
