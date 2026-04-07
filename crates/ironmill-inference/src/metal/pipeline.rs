//! Core decode pipeline — run_pipeline, run_pipeline_inner, prefill_all_logits.

use half::f16;
use ironmill_metal_sys::{CommandBufferStatus, MetalBuffer};
use mil_rs::weights::Architecture;

use super::attention::{
    encode_end_of_layer_residual, encode_kv_cache_and_attention, encode_qk_norm_and_rope,
};
use super::buffers::{ModelConfigExt, Q8_GROUP_SIZE, build_matmul_cache};
use super::engine::MetalInference;
use super::ffn::{encode_ffn_block, encode_moe_block};
use super::gdn::{encode_gdn_decode, encode_gdn_prefill};
use super::ops;
use super::plan::{AttentionKind, RopeTable};
use super::ple;
use super::projection::{Q8Input, encode_projection, encode_projection_q8};
use super::weights::WeightBuffer;
use crate::engine::InferenceError;
use crate::types::Logits;

impl MetalInference {
    // ── Core decode pipeline ────────────────────────────────────

    /// Prefill all tokens and return logits for **every** position.
    ///
    /// Unlike `prefill()` (which returns only the last position's logits),
    /// this reads back the full `[token_count × vocab_size]` logits tensor.
    /// Tokens are processed in chunks matching the intermediate buffer size
    /// so arbitrarily long sequences work without OOM.
    ///
    /// Used for efficient perplexity evaluation.
    pub fn prefill_all_logits(&mut self, token_ids: &[u32]) -> Result<Vec<Logits>, InferenceError> {
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .clone();
        let vocab = mc.vocab_size;
        let n = token_ids.len();

        // Run full pipeline — buffers grow on demand inside run_pipeline_inner.
        // Skip the built-in last-token readback; we read ALL positions below.
        self.run_pipeline_inner(token_ids, true, false, false)?;

        let bufs = self
            .intermediate_buffers
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;

        let total_bytes = n * vocab * 2;
        let mut fp16_buf = vec![0u8; total_bytes];
        bufs.logits
            .read_bytes(&mut fp16_buf, 0)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        let all_logits: Vec<Logits> = (0..n)
            .map(|t| {
                let offset = t * vocab * 2;
                fp16_buf[offset..offset + vocab * 2]
                    .chunks_exact(2)
                    .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect()
            })
            .collect();

        #[cfg(debug_assertions)]
        {
            // DEBUG: save logits to disk for comparison with HF
            if std::env::var("IRONMILL_SAVE_LOGITS").is_ok() {
                for &pos in &[3usize, 20, 40] {
                    if pos < n {
                        let l = &all_logits[pos];
                        let path = format!("/tmp/im_logits_pos{pos}.bin");
                        let bytes: Vec<u8> = l.iter().flat_map(|v| v.to_le_bytes()).collect();
                        std::fs::write(&path, &bytes).ok();
                        let target = if pos + 1 < n {
                            token_ids[pos + 1] as usize
                        } else {
                            0
                        };
                        let max_val = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mean: f32 = l.iter().sum::<f32>() / l.len() as f32;
                        let log_sum_exp: f64 = l
                            .iter()
                            .map(|&x| ((x - max_val) as f64).exp())
                            .sum::<f64>()
                            .ln()
                            + max_val as f64;
                        let ce = if target < l.len() {
                            -(l[target] as f64 - log_sum_exp)
                        } else {
                            0.0
                        };
                        eprintln!(
                            "  [SAVE] pos={pos} target={target} ce={ce:.4} max={max_val:.3} l[target]={:.3} mean={mean:.3} std={:.3} len={}",
                            if target < l.len() { l[target] } else { 0.0 },
                            {
                                let m = mean;
                                (l.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / l.len() as f32)
                                    .sqrt()
                            },
                            l.len()
                        );
                    }
                }
            }

            // DEBUG: print per-position CE
            if std::env::var("IRONMILL_DEBUG_LOGITS").is_ok() {
                let mut debug_total_ce = 0.0f64;
                let mut debug_count = 0usize;
                for t in 0..n.saturating_sub(1) {
                    let l = &all_logits[t];
                    let target = token_ids[t + 1] as usize;
                    let max_val = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let argmax = l
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let log_sum_exp: f64 = l
                        .iter()
                        .map(|&x| ((x - max_val) as f64).exp())
                        .sum::<f64>()
                        .ln()
                        + max_val as f64;
                    let ce = if target < l.len() {
                        -(l[target] as f64 - log_sum_exp)
                    } else {
                        0.0
                    };
                    debug_total_ce += ce;
                    debug_count += 1;
                    if t < 20 || t >= n - 3 {
                        eprintln!(
                            "  [im] pos={:>2} argmax={:>6} max={:>7.3} ce={:>7.3}",
                            t, argmax, max_val, ce
                        );
                    }
                }
                if debug_count > 0 {
                    let avg = debug_total_ce / debug_count as f64;
                    eprintln!(
                        "  [im] DEBUG PPL from logits: {:.2} (avg CE={:.4}, n={})",
                        avg.exp(),
                        avg,
                        debug_count
                    );
                }
            }
        }

        Ok(all_logits)
    }

    /// Run the transformer decode pipeline for `token_count` tokens.
    /// Returns logits for the last token position.
    pub(crate) fn run_pipeline(&mut self, token_ids: &[u32]) -> Result<Logits, InferenceError> {
        self.run_pipeline_inner(token_ids, false, false, false)
    }

    /// Decode one token, returning logits AND the final hidden state (FP32).
    ///
    /// The hidden state is the output of the final RMSNorm (before the LM head
    /// projection), read back from `norm_out`. Used by EAGLE-3 speculative
    /// decoding where the draft head needs the target model's hidden state.
    pub(crate) fn decode_step_with_hidden(
        &mut self,
        token: u32,
    ) -> Result<(Logits, Vec<f32>), InferenceError> {
        let logits = self.run_pipeline(&[token])?;
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;
        let h = mc.hidden_size;
        let bufs = self
            .intermediate_buffers
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;

        // Read back the final normed hidden state (FP16 → FP32).
        let mut data = vec![0u8; h * 2];
        bufs.norm_out
            .read_bytes(&mut data, 0)
            .map_err(|e| InferenceError::runtime(format!("hidden state readback: {e}")))?;
        let hidden: Vec<f32> = data
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();

        Ok((logits, hidden))
    }

    /// Read the hidden state from the last decode/prefill step.
    ///
    /// Returns the content of `norm_out` (final RMSNorm output before LM head)
    /// as FP32. Call after `decode_step()`, `prefill()`, or `speculative_step()`
    /// to get the hidden state without re-running the pipeline.
    pub fn last_hidden_state(&self) -> Result<Vec<f32>, InferenceError> {
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;
        let h = mc.hidden_size;
        let bufs = self
            .intermediate_buffers
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;
        let mut data = vec![0u8; h * 2];
        bufs.norm_out
            .read_bytes(&mut data, 0)
            .map_err(|e| InferenceError::runtime(format!("hidden state readback: {e}")))?;
        Ok(data
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect())
    }

    pub(crate) fn run_pipeline_inner(
        &mut self,
        token_ids: &[u32],
        skip_logits: bool,
        use_alt_token_buf: bool,
        skip_wait: bool,
    ) -> Result<Logits, InferenceError> {
        let weights = self.weights.as_ref().ok_or(InferenceError::NotLoaded)?;
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .clone();

        // Grow intermediate buffers on demand for larger prefill batches.
        self.intermediate_buffers
            .as_mut()
            .ok_or(InferenceError::NotLoaded)?
            .ensure_capacity(
                &self.device,
                token_ids.len(),
                &mc,
                self.gemma4_config.as_ref(),
            )
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        // Grow GDN scratch buffers for prefill (token_count > 1).
        if let Some(ref mut gdn) = self.gdn_state {
            gdn.ensure_scratch_capacity(&self.device, token_ids.len(), mc.hidden_size)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
        }

        let bufs = self
            .intermediate_buffers
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;
        let rope_cos = self.rope_cos.as_ref().ok_or(InferenceError::NotLoaded)?;
        let rope_sin = self.rope_sin.as_ref().ok_or(InferenceError::NotLoaded)?;

        let token_count = token_ids.len();
        let seq_pos = self.seq_pos;

        // Guard: ensure tokens fit within the KV cache.
        if seq_pos
            .checked_add(token_count)
            .is_none_or(|end| end > self.config.max_seq_len)
        {
            return Err(InferenceError::runtime(format!(
                "sequence position {} + token count {} exceeds max_seq_len {}",
                seq_pos, token_count, self.config.max_seq_len,
            )));
        }

        let h = mc.hidden_size;
        let nh = mc.num_attention_heads as u32;
        let _nkv = mc.num_kv_heads() as u32;
        let _hd = mc.head_dim as u32;
        let _inter = mc.intermediate_size;
        let vocab = mc.vocab_size;
        let eps = mc.rms_norm_eps as f32;
        let enable_tq = self.config.enable_turboquant && self.turboquant.is_some();

        // Use the pre-built single-token matmul cache for decode steps;
        // only rebuild the general cache for prefill (token_count > 1).
        let _matmuls = if token_count == 1 {
            self.decode_matmuls_t1
                .as_ref()
                .ok_or(InferenceError::NotLoaded)?
        } else {
            let need_rebuild = self
                .decode_matmuls
                .as_ref()
                .is_none_or(|c| c.token_count != token_count);
            if need_rebuild {
                let cache = build_matmul_cache(
                    &self.device,
                    &mc,
                    self.gemma4_config.as_ref(),
                    weights,
                    token_count,
                )
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
                self.decode_matmuls = Some(cache);
            }
            self.decode_matmuls
                .as_ref()
                .ok_or_else(|| InferenceError::runtime("decode_matmuls not populated"))?
        };

        // Write token IDs to GPU buffer (reuse persistent buffer).
        // When pipelining prefill chunks, alternate between two token ID
        // buffers so the CPU can write the next chunk while the GPU still
        // reads the previous one.
        self.token_bytes_buf.clear();
        self.token_bytes_buf
            .extend(token_ids.iter().flat_map(|t| t.to_le_bytes()));
        let active_token_buf = if use_alt_token_buf {
            &bufs.token_ids_buf_b
        } else {
            &bufs.token_ids_buf
        };
        active_token_buf
            .write_bytes(&self.token_bytes_buf, 0)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        // Create command buffer and single shared compute encoder.
        let cmd_buf = self
            .queue
            .command_buffer()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        let enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        // Step 0: Fused embedding lookup + first-layer RMSNorm.
        // Writes both hidden_state (raw embedding for residual) and
        // norm_out (normalized for first layer's projections).
        // Gemma models scale embeddings by sqrt(hidden_size).
        let embed_scale = if let Some(ref plan) = self.model_plan {
            plan.embed_scale
        } else if mc.architecture == Architecture::Gemma {
            (h as f32).sqrt()
        } else {
            1.0
        };
        {
            let lw0 = &weights.layers[0];
            let pipelines = self.pipelines()?;
            enc.set_pipeline(&pipelines.fused_embedding_norm);
            enc.set_buffer(active_token_buf, 0, 0);
            enc.set_buffer(&weights.embedding, 0, 1);
            enc.set_buffer(&lw0.input_norm, 0, 2);
            enc.set_buffer(&bufs.norm_out, 0, 3);
            enc.set_buffer(&bufs.hidden_state, 0, 4);
            enc.set_bytes(&(h as u32).to_le_bytes(), 5);
            enc.set_bytes(&(token_count as u32).to_le_bytes(), 6);
            enc.set_bytes(&(vocab as u32).to_le_bytes(), 7);
            enc.set_bytes(&eps.to_le_bytes(), 8);
            enc.set_bytes(&embed_scale.to_le_bytes(), 9);
            let tg_size = h.min(1024);
            enc.dispatch_threadgroups((token_count, 1, 1), (tg_size, 1, 1));
        }
        enc.memory_barrier_with_resources(&[&bufs.norm_out, &bufs.hidden_state]);

        // PLE model-level computation: compute per-layer embeddings before the layer loop.
        // Result lives in ple_per_layer_input for the duration of all layers.
        ple::encode_ple_model_level(
            &enc,
            self.pipelines()?,
            &self.device,
            weights,
            bufs,
            active_token_buf,
            &mc,
            self.gemma4_config.as_ref(),
            token_count,
            h,
            vocab,
            eps,
            true,
        )?;

        // Per-layer processing.
        //
        // norm_out already contains the first layer's input-norm result
        // (from the fused embedding+norm above). Subsequent layers receive
        // their input norm from the previous layer's fused end-of-layer dispatch.

        // P1 fusion tracking: when the previous layer's end-of-layer kernel
        // fused residual+norm+first_projection, the current layer should skip
        // its first projection (Q for Standard, QKV for GDN).
        let mut skip_first_proj = false;

        for layer_idx in 0..mc.num_hidden_layers {
            let lw = &weights.layers[layer_idx];
            let plan = &self.layer_plans[layer_idx];

            // Per-layer dims from plan (resolved at load time).
            let layer_hd = plan.head_dim;
            let layer_nkv = plan.num_kv_heads;
            let layer_window = plan.window_size;
            let layer_inter = plan.intermediate_size;

            // norm_out already contains the input-norm result:
            //   • layer 0: computed by the standalone dispatch above
            //   • layer 1+: produced by the previous layer's fused end-of-layer kernel

            // Check if we should skip the first projection (pre-computed by P1 fusion).
            let p1_skip = skip_first_proj;
            skip_first_proj = false;

            // Steps 3-5: Q/K/V projections — dispatch by weight type.
            // These are independent (all read norm_out, write to different buffers).

            match &plan.attention {
                AttentionKind::Gdn { gdn_index: _ } => {
                    // ── GDN (linear-attention) layer — GPU path ─────
                    // Ensure scratch capacity for prefill before taking immutable borrows.
                    if token_count > 1 {
                        let gdn_mut = self
                            .gdn_state
                            .as_mut()
                            .ok_or_else(|| InferenceError::runtime("GDN state not initialized"))?;
                        gdn_mut
                            .ensure_scratch_capacity(&self.device, token_count, h)
                            .map_err(|e| InferenceError::runtime(e.to_string()))?;
                    }
                    let pipelines = self.pipelines()?;
                    let gdn = self
                        .gdn_state
                        .as_ref()
                        .ok_or_else(|| InferenceError::runtime("GDN state not initialized"))?;
                    let gdn_idx = gdn.gdn_index_for_layer(layer_idx).ok_or_else(|| {
                        InferenceError::runtime(format!("layer {layer_idx} is not a GDN layer"))
                    })?;
                    if token_count > 1 {
                        encode_gdn_prefill(
                            &enc,
                            bufs,
                            gdn,
                            lw,
                            pipelines,
                            gdn_idx,
                            token_count,
                            h,
                            eps,
                        )?;
                    } else {
                        encode_gdn_decode(
                            &enc, bufs, gdn, lw, pipelines, gdn_idx, h, eps, p1_skip,
                        )?;
                    }
                    enc.memory_barrier_with_resources(&[
                        &bufs.norm_out,
                        &bufs.residual,
                        &bufs.hidden_state,
                    ]);
                }
                AttentionKind::Standard {
                    has_output_gate,
                    has_v_norm,
                } => {
                    // Standard attention layer
                    let default_pipelines = self.pipelines()?;
                    let pipelines = if plan.use_global_pipelines {
                        self.global_pipelines.as_ref().unwrap_or(default_pipelines)
                    } else {
                        default_pipelines
                    };

                    let qkv_out_features = mc.num_attention_heads * layer_hd as usize;
                    let kv_out_features = layer_nkv as usize * layer_hd as usize;

                    // Q8 input quantization for decode: quantize norm_out once,
                    // reuse for all affine INT4 projections (Q, K, V, gate).
                    let q8_input = if token_count == 1 {
                        // Check if any projection uses affine INT4 (common case).
                        let has_affine_int4 = matches!(&lw.q_proj, WeightBuffer::AffineQuantized(aq) if aq.bit_width == 4)
                            || matches!(&lw.k_proj, WeightBuffer::AffineQuantized(aq) if aq.bit_width == 4);
                        if has_affine_int4 {
                            let pipelines_ref = self.pipelines()?;
                            ops::encode_quantize_input_q8(
                                &enc,
                                &pipelines_ref.quantize_input_q8,
                                &bufs.norm_out,
                                &bufs.q8_data,
                                &bufs.q8_scales,
                                h as u32,
                                Q8_GROUP_SIZE as u32,
                            );
                            enc.memory_barrier_with_resources(&[&bufs.q8_data, &bufs.q8_scales]);
                            Some(Q8Input {
                                data: &bufs.q8_data,
                                scales: &bufs.q8_scales,
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // Build the projection list: skip Q if P1 fusion already computed it.
                    let mut projections: Vec<(&WeightBuffer, &MetalBuffer, usize)> = Vec::new();
                    if !p1_skip {
                        projections.push((&lw.q_proj, &bufs.q_proj, qkv_out_features));
                    }
                    projections.push((&lw.k_proj, &bufs.k_proj, kv_out_features));
                    projections.push((&lw.v_proj, &bufs.v_proj, kv_out_features));
                    for (weight, output_buf, out_features) in &projections {
                        encode_projection_q8(
                            &enc,
                            &bufs.norm_out,
                            weight,
                            output_buf,
                            pipelines,
                            token_count,
                            *out_features,
                            h,
                            q8_input.as_ref(),
                        )?;
                    }

                    // Qwen3.5 attn_output_gate: dispatch gate projection alongside Q/K/V
                    // (reads norm_out, writes q_gate — independent of Q/K/V outputs).
                    // The gate result is only consumed later by sigmoid_gate after attention,
                    // so no separate barrier is needed here.
                    if let (Some(gate_w), Some(gate_buf)) = (&lw.attn_output_gate, &bufs.q_gate) {
                        encode_projection_q8(
                            &enc,
                            &bufs.norm_out,
                            gate_w,
                            gate_buf,
                            pipelines,
                            token_count,
                            qkv_out_features,
                            h,
                            q8_input.as_ref(),
                        )?;
                    }
                    // Barrier for Q/K/V/gate projection outputs.
                    {
                        let mut barrier_bufs: Vec<&MetalBuffer> =
                            vec![&bufs.q_proj, &bufs.k_proj, &bufs.v_proj];
                        if let Some(ref gate_buf) = bufs.q_gate {
                            barrier_bufs.push(gate_buf);
                        }
                        enc.memory_barrier_with_resources(&barrier_bufs);
                    }

                    // Gemma 4 V-norm: scale-free RMSNorm on V projections.
                    if *has_v_norm {
                        if let Some(ref unit_w) = self.unit_norm_weight {
                            ops::encode_rms_norm(
                                &enc,
                                &pipelines.rms_norm,
                                &ops::RmsNormParams {
                                    input: &bufs.v_proj,
                                    weight: unit_w,
                                    output: &bufs.v_proj,
                                    hidden_size: layer_hd,
                                    token_count: (token_count * layer_nkv as usize) as u32,
                                    eps,
                                },
                            );
                            // No barrier needed: next dispatch (QK norm+RoPE) reads
                            // q_proj/k_proj, not v_proj. v_proj is covered by the
                            // barrier after QK norm before KV cache scatter.
                        }
                    }

                    // Step 6: QK normalization + RoPE
                    let (layer_rope_cos, layer_rope_sin) = match plan.rope_table {
                        RopeTable::Global => (
                            self.global_rope_cos.as_ref().unwrap_or(rope_cos),
                            self.global_rope_sin.as_ref().unwrap_or(rope_sin),
                        ),
                        RopeTable::Default => (rope_cos, rope_sin),
                    };
                    encode_qk_norm_and_rope(
                        &enc,
                        pipelines,
                        bufs,
                        lw.q_norm.as_ref(),
                        lw.k_norm.as_ref(),
                        layer_rope_cos,
                        layer_rope_sin,
                        nh,
                        layer_nkv,
                        layer_hd,
                        seq_pos,
                        token_count,
                        eps,
                    )?;
                    // Barrier includes v_proj for KV cache (covers V-norm write if present).
                    enc.memory_barrier_with_resources(&[&bufs.q_proj, &bufs.k_proj, &bufs.v_proj]);

                    // Steps 7-8: KV cache write + attention
                    let attn_scale = plan.attn_scale;

                    encode_kv_cache_and_attention(
                        &enc,
                        pipelines,
                        bufs,
                        self.turboquant.as_ref(),
                        self.kv_cache.as_ref(),
                        self.fp16_kv_cache.as_ref(),
                        self.config.max_seq_len,
                        self.config.n_bits as usize,
                        plan.kv_cache_layer,
                        seq_pos,
                        token_count,
                        nh,
                        layer_nkv,
                        layer_hd,
                        enable_tq,
                        plan.kv_anchor,
                        layer_window,
                        attn_scale,
                        self.gpu_max_threadgroups,
                    )?;
                    enc.memory_barrier_with_resources(&[&bufs.attn_out]);

                    // Qwen3.5 attn_output_gate: apply sigmoid(gate) to attention output.
                    if *has_output_gate {
                        if let Some(gate_buf) = &bufs.q_gate {
                            let gate_size =
                                (token_count * mc.num_attention_heads * layer_hd as usize) as u32;
                            ops::encode_sigmoid_gate(
                                &enc,
                                &pipelines.sigmoid_gate,
                                &bufs.attn_out,
                                gate_buf,
                                gate_size,
                            );
                            enc.memory_barrier_with_resources(&[&bufs.attn_out]);
                        }
                    }

                    // Step 9: Output projection
                    let attn_out_features = mc.num_attention_heads * layer_hd as usize;
                    encode_projection(
                        &enc,
                        &bufs.attn_out,
                        &lw.o_proj,
                        &bufs.ffn_down,
                        pipelines,
                        token_count,
                        h,
                        attn_out_features,
                    )?;
                    enc.memory_barrier_with_resources(&[&bufs.ffn_down]);

                    // Step 10-11: Residual add + post-attention RMSNorm
                    if let Some(pre_ffn) = &lw.pre_ffn_norm {
                        // Gemma 4: post_attention_layernorm before residual add
                        ops::encode_rms_norm(
                            &enc,
                            &pipelines.rms_norm,
                            &ops::RmsNormParams {
                                input: &bufs.ffn_down,
                                weight: &lw.post_attn_norm,
                                output: &bufs.ffn_down,
                                hidden_size: h as u32,
                                token_count: token_count as u32,
                                eps,
                            },
                        );
                        enc.memory_barrier_with_resources(&[&bufs.ffn_down]);

                        ops::encode_residual_add(
                            &enc,
                            &pipelines.residual_add,
                            &bufs.hidden_state,
                            &bufs.ffn_down,
                            &bufs.residual,
                            (token_count * h) as u32,
                        );
                        enc.memory_barrier_with_resources(&[&bufs.residual]);

                        ops::encode_rms_norm(
                            &enc,
                            &pipelines.rms_norm,
                            &ops::RmsNormParams {
                                input: &bufs.residual,
                                weight: pre_ffn,
                                output: &bufs.norm_out,
                                hidden_size: h as u32,
                                token_count: token_count as u32,
                                eps,
                            },
                        );
                        enc.memory_barrier_with_resources(&[&bufs.norm_out]);
                    } else {
                        // Standard pre-norm transformer: fused residual + norm
                        ops::encode_fused_residual_rms_norm(
                            &enc,
                            &pipelines.fused_residual_rms_norm,
                            &ops::FusedResidualRmsNormParams {
                                a: &bufs.hidden_state,
                                b: &bufs.ffn_down,
                                weight: &lw.post_attn_norm,
                                normed_output: &bufs.norm_out,
                                residual_output: &bufs.residual,
                                eps,
                                hidden_size: h as u32,
                                token_count: token_count as u32,
                            },
                        );
                        enc.memory_barrier_with_resources(&[&bufs.norm_out, &bufs.residual]);
                    }
                } // end AttentionKind::Standard
            } // end match plan.attention

            // D2Quant DAC: add per-layer correction bias to post-attention
            // norm output to compensate for quantization-induced mean shift.
            if let Some(ref dac) = self.dac_biases {
                if layer_idx < dac.len() {
                    let pipelines = self.pipelines()?;
                    ops::encode_bias_add(
                        &enc,
                        &pipelines.bias_add,
                        &bufs.norm_out,
                        &dac[layer_idx],
                        h as u32,
                        (token_count * h) as u32,
                    );
                    enc.memory_barrier_with_resources(&[&bufs.norm_out]);
                }
            }

            // Get pipelines for the remaining steps (FFN, MoE, PLE, etc.)
            let default_pipelines = self.pipelines()?;
            let pipelines = if plan.use_global_pipelines {
                self.global_pipelines.as_ref().unwrap_or(default_pipelines)
            } else {
                default_pipelines
            };

            // Steps 12-15: FFN block (gate + up + activation + down)
            let use_gelu = plan.use_gelu;
            encode_ffn_block(
                &enc,
                pipelines,
                bufs,
                lw,
                h,
                layer_inter,
                token_count,
                use_gelu,
            )?;
            enc.memory_barrier_with_resources(&[&bufs.ffn_down]);

            // MoE block: when enabled, dense MLP output is combined
            // with MoE expert outputs via router → expert FFNs → weighted combine.
            // Must run BEFORE post_ffn_norm so the norm applies to the combined output.
            if let Some(ref moe) = plan.moe {
                encode_moe_block(
                    &enc,
                    pipelines,
                    bufs,
                    lw,
                    h,
                    moe.moe_intermediate_size,
                    token_count,
                    moe.num_experts,
                    moe.top_k,
                )?;
            }

            // Post-feedforward layernorm (Gemma 4).
            if let Some(ref post_ffn) = lw.post_ffn_norm {
                ops::encode_rms_norm(
                    &enc,
                    &pipelines.rms_norm,
                    &ops::RmsNormParams {
                        input: &bufs.ffn_down,
                        weight: post_ffn,
                        output: &bufs.ffn_gate,
                        hidden_size: h as u32,
                        token_count: token_count as u32,
                        eps,
                    },
                );
                enc.memory_barrier_with_resources(&[&bufs.ffn_gate]);
                ops::encode_copy_buffer(
                    &enc,
                    &pipelines.copy_buffer,
                    &bufs.ffn_gate,
                    &bufs.ffn_down,
                    (token_count * h) as u32,
                );
                enc.memory_barrier_with_resources(&[&bufs.ffn_down]);
            }

            // PLE per-layer: gate → GELU → multiply → project → norm → residual add.
            // When PLE is active, we split the fused residual+norm into separate steps
            // so PLE can be inserted between the FFN residual add and next layer's norm.
            let next_input_norm = if layer_idx + 1 < mc.num_hidden_layers {
                Some(&weights.layers[layer_idx + 1].input_norm)
            } else {
                None
            };
            let ple_applied = ple::encode_ple_per_layer(
                &enc,
                self.pipelines()?,
                bufs,
                lw,
                next_input_norm,
                self.gemma4_config.as_ref(),
                mc.num_hidden_layers,
                layer_idx,
                token_count,
                h,
                eps,
            )?;

            if !ple_applied {
                // Step 16: Residual add + layer_scalar + next layer's input norm.
                if let Some(scalar) = &lw.layer_scalar {
                    // Can't use fused residual+norm: need to insert layer_scalar
                    // between the residual add and the next-layer norm.
                    // HF: hidden_states = residual + hidden_states; hidden_states *= layer_scalar
                    ops::encode_residual_add(
                        &enc,
                        &pipelines.residual_add,
                        &bufs.residual,
                        &bufs.ffn_down,
                        &bufs.hidden_state,
                        (token_count * h) as u32,
                    );
                    enc.memory_barrier_with_resources(&[&bufs.hidden_state]);
                    ops::encode_scale_buffer(
                        &enc,
                        &pipelines.scale_buffer,
                        &bufs.hidden_state,
                        scalar,
                        (token_count * h) as u32,
                    );
                    enc.memory_barrier_with_resources(&[&bufs.hidden_state]);
                    if layer_idx + 1 < mc.num_hidden_layers {
                        let next_norm = &weights.layers[layer_idx + 1].input_norm;
                        ops::encode_rms_norm(
                            &enc,
                            &pipelines.rms_norm,
                            &ops::RmsNormParams {
                                input: &bufs.hidden_state,
                                weight: next_norm,
                                output: &bufs.norm_out,
                                hidden_size: h as u32,
                                token_count: token_count as u32,
                                eps,
                            },
                        );
                    }
                } else {
                    // No layer_scalar: use fused residual + norm for efficiency.
                    // For decode, attempt P1 fusion with next layer's first projection.
                    let next_norm = if layer_idx + 1 < mc.num_hidden_layers {
                        Some(&weights.layers[layer_idx + 1].input_norm)
                    } else {
                        None
                    };

                    // Determine next layer's first projection for P1 fusion.
                    let next_first_proj = if token_count == 1
                        && layer_idx + 1 < mc.num_hidden_layers
                        && next_norm.is_some()
                    {
                        let next_plan = &self.layer_plans[layer_idx + 1];
                        let next_lw = &weights.layers[layer_idx + 1];
                        match &next_plan.attention {
                            AttentionKind::Standard { .. } => {
                                // Fuse with Q projection (first of Q/K/V)
                                let qkv_out = mc.num_attention_heads * next_plan.head_dim as usize;
                                Some((&next_lw.q_proj, &bufs.q_proj, qkv_out))
                            }
                            AttentionKind::Gdn { .. } => {
                                // Fuse with QKV projection (first of QKV/Z/A/B)
                                let gdn = self.gdn_state.as_ref().ok_or_else(|| {
                                    InferenceError::runtime("GDN state not initialized")
                                })?;
                                let qkv_dim = gdn.config.qkv_dim;
                                Some((
                                    next_lw.gdn_in_proj_qkv.as_ref().unwrap(),
                                    &gdn.gpu_temp_qkv,
                                    qkv_dim,
                                ))
                            }
                        }
                    } else {
                        None
                    };

                    let fused = encode_end_of_layer_residual(
                        &enc,
                        pipelines,
                        bufs,
                        next_norm,
                        next_first_proj,
                        h,
                        token_count,
                        eps,
                    )?;

                    // If P1 fusion was used, the next layer's first projection is already
                    // computed. Signal the next iteration to skip it.
                    skip_first_proj = fused;
                }
                enc.memory_barrier_with_resources(&[&bufs.hidden_state, &bufs.norm_out]);
            }
        }
        ops::encode_rms_norm(
            &enc,
            &self.pipelines()?.rms_norm,
            &ops::RmsNormParams {
                input: &bufs.hidden_state,
                weight: &weights.final_norm,
                output: &bufs.norm_out,
                hidden_size: h as u32,
                token_count: token_count as u32,
                eps,
            },
        );
        enc.memory_barrier_with_resources(&[&bufs.norm_out]);

        // Step 18: LM head projection — dispatches through encode_projection
        // which handles Dense (packed blocked), D2Quant, affine INT4, etc.
        {
            let pipelines = self.pipelines()?;
            encode_projection(
                &enc,
                &bufs.norm_out,
                &weights.lm_head,
                &bufs.logits,
                pipelines,
                token_count,
                vocab,
                h,
            )?;
            enc.end_encoding();
        }

        // Gemma 4: final logit softcapping (softcap * tanh(logits / softcap)).
        let softcap = self
            .model_plan
            .as_ref()
            .and_then(|p| p.final_logit_softcapping)
            .or_else(|| {
                self.gemma4_config
                    .as_ref()
                    .and_then(|g| g.final_logit_softcapping)
            });
        if let Some(softcap) = softcap {
            let pipelines = self.pipelines()?;
            let sc_enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let count = (token_count * vocab) as u32;
            ops::encode_fused_softcap(
                &sc_enc,
                &pipelines.fused_softcap,
                &bufs.logits,
                softcap,
                count,
            );
            sc_enc.end_encoding();
        }

        // Step 19: Commit and optionally wait.
        // When pipelining prefill chunks, skip_wait allows the GPU to execute
        // this command buffer while the CPU encodes the next chunk.
        cmd_buf.commit();
        if !skip_wait {
            cmd_buf.wait_until_completed();

            if cmd_buf.status() == CommandBufferStatus::Error {
                return Err(InferenceError::Decode(
                    "Metal command buffer execution failed".into(),
                ));
            }
        }

        // Step 20: Read logits for the last token position → Vec<f32>.
        // When skip_logits is set (non-last prefill chunks), skip the
        // expensive GPU readback + f16→f32 conversion since the logits
        // are immediately discarded.
        let logits: Vec<f32> = if skip_logits {
            Vec::new()
        } else {
            let last_token_offset = (token_count - 1) * vocab * 2; // FP16 offset in bytes
            let logits_byte_count = vocab * 2;
            self.logits_fp16_buf.resize(logits_byte_count, 0);
            bufs.logits
                .read_bytes(&mut self.logits_fp16_buf, last_token_offset)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;

            // SIMD-accelerated FP16→FP32 via half crate's platform-optimized batch conversion.
            // On Apple Silicon this uses NEON vcvt instructions (~8× faster than scalar).
            use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
            let u16_slice: Vec<u16> = self
                .logits_fp16_buf
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let f16_slice: &[f16] = u16_slice.reinterpret_cast();
            let mut logits_f32 = vec![0.0f32; f16_slice.len()];
            f16_slice.convert_to_f32_slice(&mut logits_f32);
            logits_f32
        };

        self.seq_pos += token_count;
        if enable_tq {
            if let Some(kv) = self.kv_cache.as_mut() {
                kv.advance_by(token_count)?;
            }
        } else if let Some(fp16_kv) = self.fp16_kv_cache.as_mut() {
            fp16_kv.seq_pos += token_count;
        }
        if let Some(mla_kv) = self.mla_kv_cache.as_mut() {
            mla_kv
                .advance_by(token_count)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
        }

        Ok(logits)
    }
}
