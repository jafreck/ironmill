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
use mil_rs::weights::MlaConfig;

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
    debug_assert_eq!(
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
        let config = MlaConfig {
            kv_latent_dim: 3,
            q_latent_dim: 0, // unused in absorption
            num_heads: 2,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 1,
            v_head_dim: 2,
        };

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
        let config = MlaConfig {
            kv_latent_dim: 2,
            q_latent_dim: 0,
            num_heads: 1,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 0,
            v_head_dim: 2,
        };

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
