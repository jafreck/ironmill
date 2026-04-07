//! QK-norm, RoPE, KV cache scatter, and attention encoding.

use ironmill_metal_sys::{ComputeEncoder, MetalBuffer};

use super::buffers::IntermediateBuffers;
use super::kv_cache::Fp16KvCache;
use super::ops;
use super::turboquant::{MetalKvCache, MetalTurboQuantModel};
use super::weights::WeightBuffer;
use crate::engine::InferenceError;

pub(crate) fn encode_qk_norm_and_rope(
    enc: &ComputeEncoder,
    pipelines: &super::ops::MetalPipelines,
    bufs: &IntermediateBuffers,
    q_norm: Option<&MetalBuffer>,
    k_norm: Option<&MetalBuffer>,
    rope_cos: &MetalBuffer,
    rope_sin: &MetalBuffer,
    nh: u32,
    nkv: u32,
    hd: u32,
    seq_pos: usize,
    token_count: usize,
    eps: f32,
) -> Result<(), InferenceError> {
    if let (Some(q_norm_w), Some(k_norm_w)) = (q_norm, k_norm) {
        // Fused path: one dispatch does RMSNorm + RoPE for both Q and K.
        enc.set_pipeline(&pipelines.fused_qk_norm_rope);
        enc.set_buffer(&bufs.q_proj, 0, 0);
        enc.set_buffer(&bufs.k_proj, 0, 1);
        enc.set_buffer(q_norm_w, 0, 2);
        enc.set_buffer(k_norm_w, 0, 3);
        enc.set_buffer(rope_cos, 0, 4);
        enc.set_buffer(rope_sin, 0, 5);
        enc.set_bytes(&nh.to_le_bytes(), 6);
        enc.set_bytes(&nkv.to_le_bytes(), 7);
        enc.set_bytes(&hd.to_le_bytes(), 8);
        enc.set_bytes(&(seq_pos as u32).to_le_bytes(), 9);
        enc.set_bytes(&(token_count as u32).to_le_bytes(), 10);
        enc.set_bytes(&eps.to_le_bytes(), 11);
        let tg_size = (hd as usize).min(1024);
        enc.dispatch_threadgroups(((nh + nkv) as usize, token_count, 1), (tg_size, 1, 1));
    } else {
        // No QK-norm — just RoPE.
        ops::encode_rope(
            enc,
            &pipelines.rope,
            &ops::RopeParams {
                qk: &bufs.q_proj,
                cos_cache: rope_cos,
                sin_cache: rope_sin,
                num_heads: nh,
                head_dim: hd,
                seq_offset: seq_pos as u32,
                token_count: token_count as u32,
            },
        );
        ops::encode_rope(
            enc,
            &pipelines.rope,
            &ops::RopeParams {
                qk: &bufs.k_proj,
                cos_cache: rope_cos,
                sin_cache: rope_sin,
                num_heads: nkv,
                head_dim: hd,
                seq_offset: seq_pos as u32,
                token_count: token_count as u32,
            },
        );
    }

    Ok(())
}

/// Encode KV cache write and attention dispatch.
///
/// Handles TurboQuant (outlier and standard) and FP16 KV cache paths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_kv_cache_and_attention(
    enc: &ComputeEncoder,
    pipelines: &super::ops::MetalPipelines,
    bufs: &IntermediateBuffers,
    turboquant: Option<&MetalTurboQuantModel>,
    kv_cache: Option<&MetalKvCache>,
    fp16_kv_cache: Option<&Fp16KvCache>,
    max_seq_len: usize,
    n_bits: usize,
    layer_idx: usize,
    seq_pos: usize,
    token_count: usize,
    nh: u32,
    nkv: u32,
    hd: u32,
    enable_tq: bool,
    is_anchor: bool,
    window_size: usize,
    attn_scale: f32,
    gpu_max_threadgroups: usize,
) -> Result<(), InferenceError> {
    // For SWA layers, use window_size as the buffer stride and ring-buffer
    // write position. For full-attention layers, use the global max_seq_len.
    let effective_max = if window_size > 0 {
        window_size
    } else {
        max_seq_len
    };
    let ring_seq_pos = if window_size > 0 {
        seq_pos % window_size
    } else {
        seq_pos
    };
    // For attention: how many valid cache entries exist before the current batch.
    let attn_seq_pos = if window_size > 0 {
        let total = seq_pos + token_count;
        let effective = total.min(window_size);
        // Clamp so base_seq + token_count never exceeds the physical buffer.
        effective
            .saturating_sub(token_count)
            .min(effective_max.saturating_sub(token_count))
    } else {
        seq_pos
    };
    let max_seq = effective_max as u32;
    let n_bits = n_bits as u32;

    if enable_tq {
        let tq = turboquant.ok_or_else(|| {
            InferenceError::runtime("turboquant must be initialized when enable_tq is true")
        })?;
        let kv = kv_cache.ok_or_else(|| {
            InferenceError::runtime("kv_cache must be initialized when enable_tq is true")
        })?;

        if let Some(ref outlier) = tq.outlier {
            // ── Outlier channel strategy dispatch ──
            let ((k_o_cache, v_o_cache), (k_n_cache, v_n_cache)) =
                kv.layer_outlier_caches(layer_idx);
            let ((k_o_scale, v_o_scale), (k_n_scale, v_n_scale)) =
                kv.layer_outlier_scales(layer_idx);
            let (k_o_r_norms, k_n_r_norms) = kv.layer_outlier_r_norms(layer_idx);
            let tg_size = std::cmp::max(
                outlier.d_outlier_padded as usize,
                outlier.d_non_padded as usize,
            );

            let cache_write_pos = ring_seq_pos as u32;
            let attn_base_seq = attn_seq_pos as u32;

            // CLA: only anchor layers write to the KV cache.
            if is_anchor {
                // K cache: (b-1)-bit codebook + QJL — batched over all tokens
                enc.set_pipeline(&pipelines.turboquant_outlier_cache_write);
                enc.set_buffer(&bufs.k_proj, 0, 0);
                enc.set_buffer(&outlier.channel_indices, 0, 1);
                enc.set_buffer(k_o_cache, 0, 2);
                enc.set_buffer(k_n_cache, 0, 3);
                enc.set_buffer(&outlier.outlier_rotation_signs, 0, 4);
                enc.set_buffer(&outlier.non_outlier_rotation_signs, 0, 5);
                enc.set_buffer(&outlier.k_outlier_codebook, 0, 6);
                enc.set_buffer(&outlier.k_outlier_boundaries, 0, 7);
                enc.set_buffer(&outlier.k_non_outlier_codebook, 0, 8);
                enc.set_buffer(&outlier.k_non_outlier_boundaries, 0, 9);
                enc.set_buffer(k_o_scale, 0, 10);
                enc.set_buffer(k_n_scale, 0, 11);
                enc.set_bytes(&nkv.to_le_bytes(), 12);
                enc.set_bytes(&hd.to_le_bytes(), 13);
                enc.set_bytes(&max_seq.to_le_bytes(), 14);
                enc.set_bytes(&cache_write_pos.to_le_bytes(), 15);
                enc.set_bytes(&outlier.n_outlier.to_le_bytes(), 16);
                enc.set_bytes(&outlier.d_outlier_padded.to_le_bytes(), 17);
                enc.set_bytes(&outlier.d_non_padded.to_le_bytes(), 18);
                enc.set_bytes(&outlier.k_outlier_n_levels.to_le_bytes(), 19);
                enc.set_bytes(&outlier.k_non_outlier_n_levels.to_le_bytes(), 20);
                enc.set_bytes(&1u32.to_le_bytes(), 21); // is_k_cache = 1
                enc.set_buffer(&outlier.outlier_qjl_matrix, 0, 22);
                enc.set_buffer(&outlier.non_outlier_qjl_matrix, 0, 23);
                enc.set_buffer(k_o_r_norms, 0, 24);
                enc.set_buffer(k_n_r_norms, 0, 25);
                enc.dispatch_threadgroups(
                    (nkv as usize, token_count, 1),
                    (tg_size.min(1024), 1, 1),
                );

                // V cache: b-bit codebook, no QJL — batched over all tokens
                enc.set_pipeline(&pipelines.turboquant_outlier_cache_write);
                enc.set_buffer(&bufs.v_proj, 0, 0);
                enc.set_buffer(&outlier.channel_indices, 0, 1);
                enc.set_buffer(v_o_cache, 0, 2);
                enc.set_buffer(v_n_cache, 0, 3);
                enc.set_buffer(&outlier.outlier_rotation_signs, 0, 4);
                enc.set_buffer(&outlier.non_outlier_rotation_signs, 0, 5);
                enc.set_buffer(&outlier.outlier_codebook, 0, 6);
                enc.set_buffer(&outlier.outlier_boundaries, 0, 7);
                enc.set_buffer(&outlier.non_outlier_codebook, 0, 8);
                enc.set_buffer(&outlier.non_outlier_boundaries, 0, 9);
                enc.set_buffer(v_o_scale, 0, 10);
                enc.set_buffer(v_n_scale, 0, 11);
                enc.set_bytes(&nkv.to_le_bytes(), 12);
                enc.set_bytes(&hd.to_le_bytes(), 13);
                enc.set_bytes(&max_seq.to_le_bytes(), 14);
                enc.set_bytes(&cache_write_pos.to_le_bytes(), 15);
                enc.set_bytes(&outlier.n_outlier.to_le_bytes(), 16);
                enc.set_bytes(&outlier.d_outlier_padded.to_le_bytes(), 17);
                enc.set_bytes(&outlier.d_non_padded.to_le_bytes(), 18);
                enc.set_bytes(&outlier.outlier_n_levels.to_le_bytes(), 19);
                enc.set_bytes(&outlier.non_outlier_n_levels.to_le_bytes(), 20);
                enc.set_bytes(&0u32.to_le_bytes(), 21); // is_k_cache = 0
                enc.set_buffer(&outlier.outlier_qjl_matrix, 0, 22);
                enc.set_buffer(&outlier.non_outlier_qjl_matrix, 0, 23);
                enc.set_buffer(k_o_r_norms, 0, 24);
                enc.set_buffer(k_n_r_norms, 0, 25);
                enc.dispatch_threadgroups(
                    (nkv as usize, token_count, 1),
                    (tg_size.min(1024), 1, 1),
                );
            } // end is_anchor

            // Outlier attention — batched over all tokens
            enc.set_pipeline(&pipelines.turboquant_outlier_attention);
            enc.set_buffer(&bufs.q_proj, 0, 0);
            enc.set_buffer(k_o_cache, 0, 1);
            enc.set_buffer(v_o_cache, 0, 2);
            enc.set_buffer(k_n_cache, 0, 3);
            enc.set_buffer(v_n_cache, 0, 4);
            enc.set_buffer(&outlier.channel_indices, 0, 5);
            enc.set_buffer(&outlier.outlier_rotation_signs, 0, 6);
            enc.set_buffer(&outlier.non_outlier_rotation_signs, 0, 7);
            enc.set_buffer(&outlier.k_outlier_codebook, 0, 8);
            enc.set_buffer(&outlier.k_non_outlier_codebook, 0, 9);
            enc.set_buffer(&bufs.attn_out, 0, 10);
            enc.set_buffer(k_o_scale, 0, 11);
            enc.set_buffer(v_o_scale, 0, 12);
            enc.set_buffer(k_n_scale, 0, 13);
            enc.set_buffer(v_n_scale, 0, 14);
            enc.set_bytes(&nh.to_le_bytes(), 15);
            enc.set_bytes(&nkv.to_le_bytes(), 16);
            enc.set_bytes(&hd.to_le_bytes(), 17);
            enc.set_bytes(&max_seq.to_le_bytes(), 18);
            enc.set_bytes(&attn_base_seq.to_le_bytes(), 19);
            enc.set_bytes(&outlier.n_outlier.to_le_bytes(), 20);
            enc.set_bytes(&outlier.d_outlier_padded.to_le_bytes(), 21);
            enc.set_bytes(&outlier.d_non_padded.to_le_bytes(), 22);
            enc.set_buffer(&outlier.outlier_qjl_matrix, 0, 23);
            enc.set_buffer(&outlier.non_outlier_qjl_matrix, 0, 24);
            enc.set_buffer(k_o_r_norms, 0, 25);
            enc.set_buffer(k_n_r_norms, 0, 26);
            enc.set_buffer(&outlier.outlier_codebook, 0, 27);
            enc.set_buffer(&outlier.non_outlier_codebook, 0, 28);
            enc.set_bytes(&outlier.k_outlier_n_levels.to_le_bytes(), 29);
            enc.set_bytes(&attn_scale.to_le_bytes(), 30);
            enc.dispatch_threadgroups(
                (nh as usize, token_count, 1),
                (256_usize.max(tg_size).min(1024), 1, 1),
            );
        } else {
            // ── Standard TurboQuant dispatch ──
            let (k_cache, v_cache) = kv.layer_caches(layer_idx);
            let (k_scale, v_scale) = kv.layer_scales(layer_idx);
            let (k_qjl_signs, k_r_norms) = kv.layer_k_qjl(layer_idx);

            // Select per-layer codebooks if this layer uses a non-default head_dim.
            // When codebooks_for_layer() returns None, the layer uses the same
            // head_dim as the global config, so global codebooks are correct.
            // If layer_configs exists but has no matching DimCodebooks entry,
            // the global codebooks are used as a fallback — this is expected
            // for layers that share the default head_dim.
            let dim_cb = tq.codebooks_for_layer(layer_idx);
            if dim_cb.is_none() && !tq.config.layer_configs.is_empty() {
                if let Some(lc) = tq.config.layer_configs.get(layer_idx) {
                    if lc.head_dim != tq.config.head_dim {
                        eprintln!(
                            "Warning: layer {} has head_dim {} but no per-layer codebooks found; \
                             falling back to global codebooks (head_dim={}). \
                             Attention quality may be degraded.",
                            layer_idx, lc.head_dim, tq.config.head_dim
                        );
                    }
                }
            }
            let rotation_signs = dim_cb
                .map(|dc| &dc.rotation_signs)
                .unwrap_or(&tq.rotation_signs);
            let k_codebook = dim_cb
                .map(|dc| &dc.k_codebook_buf)
                .unwrap_or(&tq.k_codebook_buf);
            let k_boundaries = dim_cb
                .map(|dc| &dc.k_boundaries_buf)
                .unwrap_or(&tq.k_boundaries_buf);
            let k_n_levels_val = dim_cb.map(|dc| dc.k_n_levels).unwrap_or(tq.k_n_levels);
            let v_codebook = dim_cb
                .map(|dc| &dc.v_codebook_buf)
                .unwrap_or(&tq.v_codebook_buf);
            let v_boundaries = dim_cb
                .map(|dc| &dc.v_boundaries_buf)
                .unwrap_or(&tq.v_boundaries_buf);
            let v_n_levels_val = dim_cb.map(|dc| dc.v_n_levels).unwrap_or(tq.v_n_levels);
            let qjl_matrix = dim_cb.map(|dc| &dc.qjl_matrix).unwrap_or(&tq.qjl_matrix);

            let cache_write_pos = ring_seq_pos as u32;
            let attn_base_seq = attn_seq_pos as u32;

            // CLA: only anchor layers write to the KV cache.
            if is_anchor {
                // K cache write — batched over all tokens
                enc.set_pipeline(&pipelines.turboquant_cache_write);
                enc.set_buffer(&bufs.k_proj, 0, 0);
                enc.set_buffer(rotation_signs, 0, 1);
                enc.set_buffer(k_cache, 0, 2);
                enc.set_bytes(&nkv.to_le_bytes(), 3);
                enc.set_bytes(&hd.to_le_bytes(), 4);
                enc.set_bytes(&max_seq.to_le_bytes(), 5);
                enc.set_bytes(&cache_write_pos.to_le_bytes(), 6);
                enc.set_bytes(&tq.inv_scale.to_le_bytes(), 7);
                enc.set_bytes(&n_bits.to_le_bytes(), 8);
                enc.set_buffer(k_scale, 0, 9);
                enc.set_buffer(k_codebook, 0, 10);
                enc.set_buffer(k_boundaries, 0, 11);
                enc.set_bytes(&k_n_levels_val.to_le_bytes(), 12);
                enc.set_buffer(qjl_matrix, 0, 13);
                enc.set_buffer(k_qjl_signs, 0, 14);
                enc.set_buffer(k_r_norms, 0, 15);
                enc.set_bytes(&1u32.to_le_bytes(), 16);
                enc.dispatch_threadgroups(
                    (nkv as usize, token_count, 1),
                    ((hd as usize).min(1024), 1, 1),
                );

                // V cache write — batched over all tokens
                enc.set_pipeline(&pipelines.turboquant_cache_write);
                enc.set_buffer(&bufs.v_proj, 0, 0);
                enc.set_buffer(rotation_signs, 0, 1);
                enc.set_buffer(v_cache, 0, 2);
                enc.set_bytes(&nkv.to_le_bytes(), 3);
                enc.set_bytes(&hd.to_le_bytes(), 4);
                enc.set_bytes(&max_seq.to_le_bytes(), 5);
                enc.set_bytes(&cache_write_pos.to_le_bytes(), 6);
                enc.set_bytes(&tq.inv_scale.to_le_bytes(), 7);
                enc.set_bytes(&n_bits.to_le_bytes(), 8);
                enc.set_buffer(v_scale, 0, 9);
                enc.set_buffer(v_codebook, 0, 10);
                enc.set_buffer(v_boundaries, 0, 11);
                enc.set_bytes(&v_n_levels_val.to_le_bytes(), 12);
                enc.set_buffer(qjl_matrix, 0, 13);
                enc.set_buffer(k_qjl_signs, 0, 14);
                enc.set_buffer(k_r_norms, 0, 15);
                enc.set_bytes(&0u32.to_le_bytes(), 16);
                enc.dispatch_threadgroups(
                    (nkv as usize, token_count, 1),
                    ((hd as usize).min(1024), 1, 1),
                );
            } // end is_anchor

            // TurboQuant attention — batched over all tokens
            // (always runs, reading from the anchor's KV buffer)
            enc.set_pipeline(&pipelines.turboquant_attention);
            enc.set_buffer(&bufs.q_proj, 0, 0);
            enc.set_buffer(k_cache, 0, 1);
            enc.set_buffer(v_cache, 0, 2);
            enc.set_buffer(rotation_signs, 0, 3);
            enc.set_buffer(&bufs.attn_out, 0, 4);
            enc.set_bytes(&nh.to_le_bytes(), 5);
            enc.set_bytes(&nkv.to_le_bytes(), 6);
            enc.set_bytes(&hd.to_le_bytes(), 7);
            enc.set_bytes(&max_seq.to_le_bytes(), 8);
            enc.set_bytes(&attn_base_seq.to_le_bytes(), 9);
            enc.set_bytes(&tq.deq_scale.to_le_bytes(), 10);
            enc.set_bytes(&n_bits.to_le_bytes(), 11);
            enc.set_buffer(k_scale, 0, 12);
            enc.set_buffer(v_scale, 0, 13);
            enc.set_buffer(k_codebook, 0, 14);
            enc.set_buffer(v_codebook, 0, 15);
            enc.set_buffer(qjl_matrix, 0, 16);
            enc.set_buffer(k_r_norms, 0, 17);
            enc.set_bytes(&k_n_levels_val.to_le_bytes(), 18);
            enc.set_bytes(&attn_scale.to_le_bytes(), 19);
            enc.dispatch_threadgroups(
                (nh as usize, token_count, 1),
                (256_usize.max(hd as usize).min(1024), 1, 1),
            );
        }
    } else {
        // FP16 KV cache path — scatter projections into cache on GPU.
        let fp16_kv = fp16_kv_cache.ok_or_else(|| {
            InferenceError::runtime("fp16_kv_cache must be initialized for FP16 KV cache path")
        })?;
        let (k_cache, v_cache) = fp16_kv.layer_caches(layer_idx);

        // CLA: only anchor layers write to the KV cache.
        if is_anchor {
            // Scatter K and V projections into their caches entirely on GPU.
            // Ring buffer: kv_scatter.metal handles modular write via % max_seq_len.
            ops::encode_kv_scatter(
                enc,
                &pipelines.kv_scatter,
                &ops::KvScatterParams {
                    proj: &bufs.k_proj,
                    cache: k_cache,
                    seq_pos: ring_seq_pos as u32,
                    token_count: token_count as u32,
                    num_kv_heads: nkv,
                    head_dim: hd,
                    max_seq_len: max_seq,
                },
            );
            ops::encode_kv_scatter(
                enc,
                &pipelines.kv_scatter,
                &ops::KvScatterParams {
                    proj: &bufs.v_proj,
                    cache: v_cache,
                    seq_pos: ring_seq_pos as u32,
                    token_count: token_count as u32,
                    num_kv_heads: nkv,
                    head_dim: hd,
                    max_seq_len: max_seq,
                },
            );
        } // end is_anchor

        // FP16 attention — V2 register-tiled kernel for prefill,
        // fused SDPA for decode.
        //
        // V2 combines cooperative KV tile loading (amortized across GQA
        // group via threadgroup memory) with register-based accumulators.
        // Benchmarks show V2 dominates fused SDPA at all prefill lengths.
        if token_count > 1 {
            let seq_offset = attn_seq_pos as u32;
            let window = if window_size > 0 {
                window_size as u32
            } else {
                0
            };
            ops::encode_v2_prefill_attention(
                enc,
                &pipelines.prefill_attention_v2,
                &ops::PrefillAttentionParams {
                    q: &bufs.q_proj,
                    k_cache,
                    v_cache,
                    output: &bufs.attn_out,
                    num_heads: nh,
                    num_kv_heads: nkv,
                    head_dim: hd,
                    max_seq_len: max_seq,
                    seq_offset,
                    token_count: token_count as u32,
                    window_size: window,
                    attn_scale,
                },
            );
        } else {
            let total_seq_len = (attn_seq_pos + token_count) as u32;
            let sdpa_params = ops::FusedSdpaParams {
                q: &bufs.q_proj,
                k: k_cache,
                v: v_cache,
                output: &bufs.attn_out,
                seq_len: total_seq_len,
                token_count: token_count as u32,
                head_dim: hd,
                num_q_heads: nh,
                num_kv_heads: nkv,
                scale: attn_scale,
                max_seq_len: max_seq,
            };
            if let (Some(po), Some(pm), Some(ps), Some(mh)) = (
                bufs.flash_decode_partial_o.as_ref(),
                bufs.flash_decode_partial_max.as_ref(),
                bufs.flash_decode_partial_sum.as_ref(),
                bufs.flash_decode_max_hint.as_ref(),
            ) {
                ops::encode_flash_decode(
                    enc,
                    &pipelines.fused_sdpa_split,
                    &pipelines.fused_sdpa_reduce,
                    pipelines.fused_sdpa.as_ref(),
                    &sdpa_params,
                    po,
                    pm,
                    ps,
                    mh,
                    bufs.flash_decode_max_splits,
                    gpu_max_threadgroups,
                );
            } else if let Some(ref sdpa) = pipelines.fused_sdpa {
                ops::encode_fused_sdpa(enc, sdpa, &sdpa_params, None);
            } else {
                // Neither FlashDecoding buffers nor fused_sdpa available.
                // This shouldn't happen in practice — buffers are always allocated.
                return Err(InferenceError::runtime(
                    "no attention kernel available: FlashDecoding buffers and fused_sdpa both unavailable",
                ));
            }
        }
    }

    Ok(())
}

/// Encode end-of-layer residual: fused with next layer's norm, or standalone for the last layer.
///
/// When `next_first_proj` is provided and decode (token_count==1), the residual+norm
/// is fused into the first projection of the next layer, saving 1 dispatch + 1 barrier.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_end_of_layer_residual(
    enc: &ComputeEncoder,
    pipelines: &super::ops::MetalPipelines,
    bufs: &IntermediateBuffers,
    next_input_norm: Option<&MetalBuffer>,
    next_first_proj: Option<(&WeightBuffer, &MetalBuffer, usize)>, // (weight, output_buf, out_features)
    h: usize,
    token_count: usize,
    eps: f32,
) -> Result<bool, InferenceError> {
    if let Some(norm_weight) = next_input_norm {
        // Try fused residual+norm+projection for decode path
        if token_count == 1 {
            if let Some((proj_weight, proj_output, out_features)) = next_first_proj {
                // Dense weights with packed buffer: fuse with dense matvec
                if let Some(packed) = proj_weight.packed_buf() {
                    ops::encode_fused_residual_norm_matvec(
                        enc,
                        &pipelines.fused_residual_norm_matvec,
                        &bufs.residual,
                        &bufs.ffn_down,
                        norm_weight,
                        &bufs.hidden_state,
                        packed,
                        proj_output,
                        &bufs.norm_out,
                        h as u32,
                        out_features as u32,
                        eps,
                    );
                    return Ok(true); // fused: caller should skip first projection
                }
                // Affine INT4: fuse with affine matvec
                if let WeightBuffer::AffineQuantized(aq) = proj_weight {
                    if aq.bit_width == 4 {
                        ops::encode_fused_residual_norm_affine_matvec_int4(
                            enc,
                            &pipelines.fused_residual_norm_affine_matvec_int4,
                            &bufs.residual,
                            &bufs.ffn_down,
                            norm_weight,
                            &bufs.hidden_state,
                            aq,
                            proj_output,
                            &bufs.norm_out,
                            out_features as u32,
                            h as u32,
                            eps,
                        );
                        return Ok(true); // fused: caller should skip first projection
                    }
                }
            }
        }

        // Fallback: standard fused residual + norm (writes norm_out for separate projection)
        ops::encode_fused_residual_rms_norm(
            enc,
            &pipelines.fused_residual_rms_norm,
            &ops::FusedResidualRmsNormParams {
                a: &bufs.residual,
                b: &bufs.ffn_down,
                weight: norm_weight,
                normed_output: &bufs.norm_out,
                residual_output: &bufs.hidden_state,
                eps,
                hidden_size: h as u32,
                token_count: token_count as u32,
            },
        );
    } else {
        ops::encode_residual_add(
            enc,
            &pipelines.residual_add,
            &bufs.residual,
            &bufs.ffn_down,
            &bufs.hidden_state,
            (token_count * h) as u32,
        );
    }
    Ok(false) // not fused: caller should dispatch projection normally
}
