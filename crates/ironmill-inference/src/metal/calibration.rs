//! Calibration pipeline, DAC, and activation hooks.

use std::time::Instant;

use half::f16;
use ironmill_metal_sys::{CommandBufferStatus, StorageMode};
use mil_rs::weights::Architecture;

use super::attention::{
    KvCacheAndAttentionParams, QkNormAndRopeParams, encode_end_of_layer_residual,
    encode_kv_cache_and_attention, encode_qk_norm_and_rope,
};
use super::buffers::{ModelConfigExt, build_matmul_cache, bytes_as_f16};
use super::engine::MetalInference;
use super::ffn::{encode_ffn_block, encode_moe_block};
use super::gdn::encode_gdn_prefill;
use super::ops;
use super::ple;
use super::projection::encode_projection;
use crate::calibration::ActivationHook;
use crate::engine::{InferenceEngine, InferenceError};
use crate::types::Logits;

/// Callback invoked per layer with `(layer_index, projection_name, raw_bytes, n_features)`.
type CalibrationCallback<'a> = dyn FnMut(usize, &str, &[u8], usize) + 'a;

impl MetalInference {
    // ── Calibration-mode pipeline ───────────────────────────────

    /// Run the transformer decode pipeline with per-layer command buffer
    /// commits, enabling CPU readback of intermediate activation buffers.
    ///
    /// This is ~10-50× slower than [`run_pipeline`] but allows collecting
    /// the input activations to each linear projection for calibration-based
    /// quantization (AWQ, GPTQ, SmoothQuant).
    ///
    /// The `layer_callback` is invoked after each transformer layer with:
    /// - `layer_index`: the 0-based layer number
    /// - `projection_name`: one of `"attn_norm"`, `"ffn_norm"`, `"o_proj_input"`,
    ///   `"down_proj_input"`, or `"block_output"`
    /// - `raw_bytes`: the FP16 activation data (token_count × n_features × 2 bytes)
    /// - `n_features`: number of features per token for this capture point
    ///
    /// Returns the same logits as [`run_pipeline`].
    pub(crate) fn run_pipeline_calibration(
        &mut self,
        token_ids: &[u32],
        layer_callback: &mut CalibrationCallback<'_>,
        stop_after_layer: Option<usize>,
    ) -> Result<Logits, InferenceError> {
        let total_start = Instant::now();

        let weights = self.weights.as_ref().ok_or(InferenceError::NotLoaded)?;
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .clone();

        // Grow intermediate buffers on demand for larger calibration batches.
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
        let nkv = mc.num_kv_heads() as u32;
        let hd = mc.head_dim as u32;
        let inter = mc.intermediate_size;
        let vocab = mc.vocab_size;
        let eps = mc.rms_norm_eps as f32;
        let enable_tq = self.config.enable_turboquant && self.turboquant.is_some();

        // Build or reuse MPS matmul cache for this token count.
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
        let _matmuls = self
            .decode_matmuls
            .as_ref()
            .ok_or_else(|| InferenceError::runtime("decode_matmuls not populated"))?;

        // Write token IDs to GPU buffer.
        let token_bytes: Vec<u8> = token_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        bufs.token_ids_buf
            .write_bytes(&token_bytes, 0)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        // Reusable readback buffer for norm_out: token_count × hidden_size × 2 bytes (FP16).
        let norm_readback_bytes = token_count * h * 2;

        // Calibration readback scratch for non-hidden-size activations (attn_out, ffn_gate).
        // Shared-mode so CPU can read back GPU results.
        let attn_out_features = mc.num_attention_heads * mc.head_dim;
        let max_readback_features = h.max(attn_out_features).max(mc.intermediate_size);
        let calib_scratch_bytes = token_count * max_readback_features * 2;
        let calib_scratch = self
            .device
            .create_buffer(calib_scratch_bytes.max(16), StorageMode::Shared)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        // Timing accumulators.
        let mut gpu_time_ms = 0.0f64;
        let mut readback_time_ms = 0.0f64;

        // ── Phase 0: Embedding lookup ───────────────────────────
        // Gemma models scale embeddings by sqrt(hidden_size).
        let embed_scale: f32 = if mc.architecture == Architecture::Gemma {
            (h as f32).sqrt()
        } else {
            1.0
        };
        let cmd_buf = self
            .queue
            .command_buffer()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        {
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            ops::encode_embedding_lookup(
                &enc,
                &self.pipelines()?.embedding.embedding_lookup,
                &ops::EmbeddingLookupParams {
                    token_ids: &bufs.token_ids_buf,
                    embedding_table: &weights.embedding,
                    output: &bufs.hidden_state,
                    hidden_size: h as u32,
                    token_count: token_count as u32,
                    vocab_size: vocab as u32,
                },
            );
            // Gemma embedding scaling: multiply hidden_state by sqrt(hidden_size).
            if embed_scale != 1.0 {
                enc.memory_barrier_buffers();
                let scale_half = f16::from_f32(embed_scale);
                let scale_buf = self
                    .device
                    .create_buffer(2, StorageMode::Shared)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                scale_buf
                    .write_bytes(&scale_half.to_le_bytes(), 0)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                ops::encode_scale_buffer(
                    &enc,
                    &self.pipelines()?.elementwise.scale_buffer,
                    &bufs.hidden_state,
                    &scale_buf,
                    (token_count * h) as u32,
                );
            }
            enc.end_encoding();
        }

        // Input norm for the first layer (standalone).
        {
            let lw0 = &weights.layers[0];
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            ops::encode_rms_norm(
                &enc,
                &self.pipelines()?.norm.rms_norm,
                &ops::RmsNormParams {
                    input: &bufs.hidden_state,
                    weight: &lw0.input_norm,
                    output: &bufs.norm_out,
                    hidden_size: h as u32,
                    token_count: token_count as u32,
                    eps,
                },
            );

            // PLE model-level: compute per-layer embeddings.
            ple::encode_ple_model_level(
                &enc,
                self.pipelines()?,
                &ple::PleModelLevelParams {
                    device: &self.device,
                    weights,
                    bufs,
                    token_ids_buf: &bufs.token_ids_buf,
                    mc: &mc,
                    gemma4_config: self.gemma4_config.as_ref(),
                    token_count,
                    h,
                    vocab,
                    eps,
                    apply_scaling: false,
                },
            )?;

            enc.end_encoding();
        }

        // Commit embedding + first-layer norm so norm_out is readable.
        let t0 = Instant::now();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        gpu_time_ms += t0.elapsed().as_secs_f64() * 1000.0;
        if cmd_buf.status() == CommandBufferStatus::Error {
            return Err(InferenceError::Decode(
                "Metal command buffer failed (embedding phase)".into(),
            ));
        }

        // ── Per-layer processing with per-layer commit ──────────
        for layer_idx in 0..mc.num_hidden_layers {
            let lw = &weights.layers[layer_idx];

            // Gemma 4: use per-layer config if available.
            let (layer_hd, layer_nkv, layer_window, layer_inter) =
                if let Some(ref g4) = self.gemma4_config {
                    let lc = &g4.layer_configs[layer_idx];
                    (
                        lc.head_dim as u32,
                        lc.num_kv_heads as u32,
                        lc.window_size,
                        lc.intermediate_size,
                    )
                } else {
                    (hd, nkv, self.config.layer_window_size(layer_idx), inter)
                };

            // ── Capture attn_norm activation (input to Q/K/V projections) ──
            // norm_out is now committed and safe to read.
            {
                let rb_start = Instant::now();
                // Allocate as u16 to guarantee 2-byte alignment for f16 reinterpret.
                let mut readback_u16 = vec![0u16; norm_readback_bytes / 2];
                let readback = bytemuck::cast_slice_mut::<u16, u8>(&mut readback_u16);
                bufs.norm_out
                    .read_bytes(readback, 0)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                readback_time_ms += rb_start.elapsed().as_secs_f64() * 1000.0;
                layer_callback(layer_idx, "attn_norm", readback, h);
            }

            // Create new command buffer for this layer's attention block.
            let is_gdn = lw.gdn_in_proj_qkv.is_some();

            // GDN layers need scratch capacity grown before taking immutable borrows.
            if is_gdn {
                let gdn_mut = self
                    .gdn_state
                    .as_mut()
                    .ok_or_else(|| InferenceError::runtime("GDN state not initialized"))?;
                gdn_mut
                    .ensure_scratch_capacity(&self.device, token_count, h)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
            }

            let mut cmd_buf = self
                .queue
                .command_buffer()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let mut enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let default_pipelines = self.pipelines()?;

            // Select the pipeline set matching this layer's HEAD_DIM.
            let pipelines = if self.global_head_dim > 0
                && layer_hd as usize == self.global_head_dim
                && self.global_head_dim != mc.head_dim
            {
                self.global_pipelines.as_ref().unwrap_or(default_pipelines)
            } else {
                default_pipelines
            };

            // Steps 3-9 + residual/norm: branch on GDN vs standard attention.
            if is_gdn {
                // ── GDN (linear-attention) layer ─────────────────
                // Calibration always uses prefill (token_count > 1).
                let gdn = self
                    .gdn_state
                    .as_ref()
                    .ok_or_else(|| InferenceError::runtime("GDN state not initialized"))?;
                let gdn_idx = gdn.gdn_index_for_layer(layer_idx).ok_or_else(|| {
                    InferenceError::runtime(format!("layer {layer_idx} is not a GDN layer"))
                })?;
                encode_gdn_prefill(&enc, bufs, gdn, lw, pipelines, gdn_idx, token_count, h, eps)?;
                enc.memory_barrier_buffers();
            } else {
                // ── Standard attention layer ─────────────────────
                let qkv_out_features = mc.num_attention_heads * layer_hd as usize;
                let kv_out_features = layer_nkv as usize * layer_hd as usize;
                for (weight, output_buf, out_features) in [
                    (&lw.q_proj, &bufs.q_proj, qkv_out_features),
                    (&lw.k_proj, &bufs.k_proj, kv_out_features),
                    (&lw.v_proj, &bufs.v_proj, kv_out_features),
                ] {
                    encode_projection(
                        &enc,
                        &bufs.norm_out,
                        weight,
                        output_buf,
                        pipelines,
                        token_count,
                        out_features,
                        h,
                    )?;
                }
                enc.memory_barrier_buffers();

                // Qwen3.5 attn_output_gate: compute gate projection.
                if let (Some(gate_w), Some(gate_buf)) = (&lw.attn_output_gate, &bufs.q_gate) {
                    encode_projection(
                        &enc,
                        &bufs.norm_out,
                        gate_w,
                        gate_buf,
                        pipelines,
                        token_count,
                        qkv_out_features,
                        h,
                    )?;
                    enc.memory_barrier_buffers();
                }

                // Step 6: QK normalization (Qwen3) + RoPE
                let (layer_rope_cos, layer_rope_sin) = if let Some(ref g4) = self.gemma4_config {
                    let lc = &g4.layer_configs[layer_idx];
                    if lc.is_global {
                        (
                            self.global_rope_cos.as_ref().unwrap_or(rope_cos),
                            self.global_rope_sin.as_ref().unwrap_or(rope_sin),
                        )
                    } else {
                        (rope_cos, rope_sin)
                    }
                } else {
                    (rope_cos, rope_sin)
                };
                encode_qk_norm_and_rope(
                    &enc,
                    pipelines,
                    &QkNormAndRopeParams {
                        bufs,
                        q_norm: lw.q_norm.as_ref(),
                        k_norm: lw.k_norm.as_ref(),
                        rope_cos: layer_rope_cos,
                        rope_sin: layer_rope_sin,
                        nh,
                        nkv: layer_nkv,
                        hd: layer_hd,
                        seq_pos,
                        token_count,
                        eps,
                    },
                )?;
                enc.memory_barrier_buffers();

                // Steps 7-8: KV cache write + attention
                let is_anchor = self
                    .config
                    .cla_config
                    .as_ref()
                    .is_none_or(|cla| cla.is_anchor(layer_idx));

                // Gemma 4 KV shared layers: shared layers skip KV writes and
                // read from their anchor layer's cache instead.
                let (is_anchor, kv_cache_layer) = if let Some(ref g4) = self.gemma4_config {
                    if let Some(anchor) = g4.layer_configs[layer_idx].kv_anchor {
                        (false, anchor)
                    } else {
                        (is_anchor, layer_idx)
                    }
                } else {
                    (is_anchor, layer_idx)
                };

                let attn_scale = mc.attn_scale();

                encode_kv_cache_and_attention(
                    &enc,
                    pipelines,
                    &KvCacheAndAttentionParams {
                        bufs,
                        turboquant: self.turboquant.as_ref(),
                        kv_cache: self.kv_cache.as_ref(),
                        fp16_kv_cache: self.fp16_kv_cache.as_ref(),
                        max_seq_len: self.config.max_seq_len,
                        n_bits: self.config.n_bits as usize,
                        layer_idx: kv_cache_layer,
                        seq_pos,
                        token_count,
                        nh,
                        nkv: layer_nkv,
                        hd: layer_hd,
                        enable_tq,
                        is_anchor,
                        window_size: layer_window,
                        attn_scale,
                        gpu_max_threadgroups: self.gpu_max_threadgroups,
                    },
                )?;
                enc.memory_barrier_buffers();

                // Qwen3.5 attn_output_gate: apply sigmoid(gate) to attention output.
                if let Some(gate_buf) = &bufs.q_gate {
                    let gate_size = (token_count * mc.num_attention_heads * mc.head_dim) as u32;
                    ops::encode_sigmoid_gate(
                        &enc,
                        &pipelines.activation.sigmoid_gate,
                        &bufs.attn_out,
                        gate_buf,
                        gate_size,
                    );
                    enc.memory_barrier_buffers();
                }

                // ── Capture O_proj input (attn_out) ──
                // Copy attn_out (Private) → calib_scratch (Shared) for CPU readback.
                {
                    let attn_out_features_local = mc.num_attention_heads * layer_hd as usize;
                    let copy_elems = (token_count * attn_out_features_local) as u32;
                    ops::encode_copy_buffer(
                        &enc,
                        &pipelines.elementwise.copy_buffer,
                        &bufs.attn_out,
                        &calib_scratch,
                        copy_elems,
                    );
                    enc.end_encoding();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();

                    let rb_bytes = token_count * attn_out_features_local * 2;
                    let rb_start = Instant::now();
                    let mut rb_u16 = vec![0u16; rb_bytes / 2];
                    let rb = bytemuck::cast_slice_mut::<u16, u8>(&mut rb_u16);
                    calib_scratch
                        .read_bytes(rb, 0)
                        .map_err(|e| InferenceError::runtime(e.to_string()))?;
                    readback_time_ms += rb_start.elapsed().as_secs_f64() * 1000.0;
                    layer_callback(layer_idx, "o_proj_input", rb, attn_out_features_local);

                    // New command buffer to continue.
                    cmd_buf = self
                        .queue
                        .command_buffer()
                        .map_err(|e| InferenceError::runtime(e.to_string()))?;
                    enc = cmd_buf
                        .compute_encoder()
                        .map_err(|e| InferenceError::runtime(e.to_string()))?;
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
                enc.memory_barrier_buffers();

                // Step 10-11: Residual add + post-attention RMSNorm
                if let Some(pre_ffn) = &lw.pre_ffn_norm {
                    // Gemma 4: post_attention_layernorm on attn output before residual
                    ops::encode_rms_norm(
                        &enc,
                        &pipelines.norm.rms_norm,
                        &ops::RmsNormParams {
                            input: &bufs.ffn_down,
                            weight: &lw.post_attn_norm,
                            output: &bufs.ffn_down,
                            hidden_size: h as u32,
                            token_count: token_count as u32,
                            eps,
                        },
                    );
                    enc.memory_barrier_buffers();
                    ops::encode_residual_add(
                        &enc,
                        &pipelines.elementwise.residual_add,
                        &bufs.hidden_state,
                        &bufs.ffn_down,
                        &bufs.residual,
                        (token_count * h) as u32,
                    );
                    enc.memory_barrier_buffers();
                    ops::encode_rms_norm(
                        &enc,
                        &pipelines.norm.rms_norm,
                        &ops::RmsNormParams {
                            input: &bufs.residual,
                            weight: pre_ffn,
                            output: &bufs.norm_out,
                            hidden_size: h as u32,
                            token_count: token_count as u32,
                            eps,
                        },
                    );
                } else {
                    ops::encode_fused_residual_rms_norm(
                        &enc,
                        &pipelines.norm.fused_residual_rms_norm,
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
                }
            }

            enc.end_encoding();

            // ── Mid-layer commit: attention block done, norm_out has FFN input ──
            let t0 = Instant::now();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
            gpu_time_ms += t0.elapsed().as_secs_f64() * 1000.0;
            if cmd_buf.status() == CommandBufferStatus::Error {
                return Err(InferenceError::Decode(format!(
                    "Metal command buffer failed (layer {layer_idx} attention phase)"
                )));
            }

            // ── Capture ffn_norm activation (input to gate/up/down projections) ──
            {
                let rb_start = Instant::now();
                let mut readback_u16 = vec![0u16; norm_readback_bytes / 2];
                let readback = bytemuck::cast_slice_mut::<u16, u8>(&mut readback_u16);
                bufs.norm_out
                    .read_bytes(readback, 0)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                readback_time_ms += rb_start.elapsed().as_secs_f64() * 1000.0;
                layer_callback(layer_idx, "ffn_norm", readback, h);
            }

            // ── New command buffer for FFN block ────────────────
            let mut cmd_buf = self
                .queue
                .command_buffer()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let mut enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let pipelines = self.pipelines()?;

            // Steps 12-15: FFN block (gate + up + activation + down)
            let use_gelu = mc.use_gelu();
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

            // ── Capture down_proj input (ffn_gate = activated gate*up) ──
            // ffn_gate still holds the SiLU/GELU-gated intermediate from encode_ffn_block.
            // Copy to calib_scratch for CPU readback before any post-FFN ops.
            {
                let copy_elems = (token_count * layer_inter) as u32;
                ops::encode_copy_buffer(
                    &enc,
                    &pipelines.elementwise.copy_buffer,
                    &bufs.ffn_gate,
                    &calib_scratch,
                    copy_elems,
                );
                enc.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();

                let rb_bytes = token_count * layer_inter * 2;
                let rb_start = Instant::now();
                let mut rb_u16 = vec![0u16; rb_bytes / 2];
                let rb = bytemuck::cast_slice_mut::<u16, u8>(&mut rb_u16);
                calib_scratch
                    .read_bytes(rb, 0)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                readback_time_ms += rb_start.elapsed().as_secs_f64() * 1000.0;
                layer_callback(layer_idx, "down_proj_input", rb, layer_inter);

                // New command buffer.
                cmd_buf = self
                    .queue
                    .command_buffer()
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
            }
            enc.memory_barrier_buffers();

            // MoE block (Gemma 4 26B): when enabled, dense MLP output is combined
            // with MoE expert outputs via router → expert FFNs → weighted combine.
            // Must run BEFORE post_ffn_norm so the norm applies to the combined output.
            if let Some(ref g4) = self.gemma4_config {
                let lc = &g4.layer_configs[layer_idx];
                if lc.enable_moe && g4.num_experts > 0 && !lw.expert_gate_projs.is_empty() {
                    encode_moe_block(
                        &enc,
                        pipelines,
                        bufs,
                        lw,
                        h,
                        g4.moe_intermediate_size,
                        token_count,
                        g4.num_experts,
                        g4.top_k_experts,
                    )?;
                }
            }

            // Gemma 4: apply post-feedforward layernorm to MLP output (after MoE combine).
            if let Some(ref post_ffn) = lw.post_ffn_norm {
                ops::encode_rms_norm(
                    &enc,
                    &pipelines.norm.rms_norm,
                    &ops::RmsNormParams {
                        input: &bufs.ffn_down,
                        weight: post_ffn,
                        output: &bufs.ffn_gate,
                        hidden_size: h as u32,
                        token_count: token_count as u32,
                        eps,
                    },
                );
                enc.memory_barrier_buffers();
                ops::encode_copy_buffer(
                    &enc,
                    &pipelines.elementwise.copy_buffer,
                    &bufs.ffn_gate,
                    &bufs.ffn_down,
                    (token_count * h) as u32,
                );
                enc.memory_barrier_buffers();
            }

            // PLE per-layer dispatch (same logic as run_pipeline_inner).
            let next_input_norm = if layer_idx + 1 < mc.num_hidden_layers {
                Some(&weights.layers[layer_idx + 1].input_norm)
            } else {
                None
            };
            let ple_applied = ple::encode_ple_per_layer(
                &enc,
                pipelines,
                &ple::PlePerLayerParams {
                    bufs,
                    lw,
                    next_input_norm,
                    gemma4_config: self.gemma4_config.as_ref(),
                    num_hidden_layers: mc.num_hidden_layers,
                    layer_idx,
                    token_count,
                    h,
                    eps,
                },
            )?;

            if !ple_applied {
                // Step 16: Residual add + layer_scalar + next layer's input norm.
                if let Some(scalar) = &lw.layer_scalar {
                    ops::encode_residual_add(
                        &enc,
                        &pipelines.elementwise.residual_add,
                        &bufs.residual,
                        &bufs.ffn_down,
                        &bufs.hidden_state,
                        (token_count * h) as u32,
                    );
                    enc.memory_barrier_buffers();
                    ops::encode_scale_buffer(
                        &enc,
                        &pipelines.elementwise.scale_buffer,
                        &bufs.hidden_state,
                        scalar,
                        (token_count * h) as u32,
                    );
                    enc.memory_barrier_buffers();
                    if layer_idx + 1 < mc.num_hidden_layers {
                        let next_norm = &weights.layers[layer_idx + 1].input_norm;
                        ops::encode_rms_norm(
                            &enc,
                            &pipelines.norm.rms_norm,
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
                    let next_norm = if layer_idx + 1 < mc.num_hidden_layers {
                        Some(&weights.layers[layer_idx + 1].input_norm)
                    } else {
                        None
                    };
                    encode_end_of_layer_residual(
                        &enc,
                        pipelines,
                        bufs,
                        next_norm,
                        None, // no P1 fusion in calibration pipeline
                        h,
                        token_count,
                        eps,
                    )?;
                }
            }
            enc.end_encoding();

            // ── End-of-layer commit ─────────────────────────────
            // This makes norm_out readable for the next iteration's attn_norm capture
            // (or for final norm / LM head on the last layer).
            let t0 = Instant::now();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
            gpu_time_ms += t0.elapsed().as_secs_f64() * 1000.0;
            if cmd_buf.status() == CommandBufferStatus::Error {
                return Err(InferenceError::Decode(format!(
                    "Metal command buffer failed (layer {layer_idx} FFN phase)"
                )));
            }

            // ── Capture block output (full hidden state after layer) ──
            // hidden_state is in Private storage — copy to Shared scratch first.
            {
                let copy_cmd = self
                    .queue
                    .command_buffer()
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                let copy_enc = copy_cmd
                    .compute_encoder()
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                ops::encode_copy_buffer(
                    &copy_enc,
                    &pipelines.elementwise.copy_buffer,
                    &bufs.hidden_state,
                    &calib_scratch,
                    (token_count * h) as u32,
                );
                copy_enc.end_encoding();
                copy_cmd.commit();
                copy_cmd.wait_until_completed();

                let rb_start = Instant::now();
                let mut readback_u16 = vec![0u16; norm_readback_bytes / 2];
                let readback = bytemuck::cast_slice_mut::<u16, u8>(&mut readback_u16);
                calib_scratch
                    .read_bytes(readback, 0)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                readback_time_ms += rb_start.elapsed().as_secs_f64() * 1000.0;
                layer_callback(layer_idx, "block_output", readback, h);
            }

            // ── Stop after specified layer (skip final norm + LM head) ──
            if stop_after_layer == Some(layer_idx) {
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
                return Ok(Vec::new());
            }
        }

        // ── Final norm + LM head ────────────────────────────────
        let cmd_buf = self
            .queue
            .command_buffer()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        // Step 17: Final RMSNorm
        {
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            ops::encode_rms_norm(
                &enc,
                &self.pipelines()?.norm.rms_norm,
                &ops::RmsNormParams {
                    input: &bufs.hidden_state,
                    weight: &weights.final_norm,
                    output: &bufs.norm_out,
                    hidden_size: h as u32,
                    token_count: token_count as u32,
                    eps,
                },
            );
            enc.end_encoding();
        }

        // Step 18: LM head projection.
        {
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
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

        // Final commit + wait.
        let t0 = Instant::now();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        gpu_time_ms += t0.elapsed().as_secs_f64() * 1000.0;

        if cmd_buf.status() == CommandBufferStatus::Error {
            return Err(InferenceError::Decode(
                "Metal command buffer failed (final norm + LM head)".into(),
            ));
        }

        // Read logits for the last token.
        let last_token_offset = (token_count - 1) * vocab * 2;
        let logits_byte_count = vocab * 2;
        let mut logits_fp16 = vec![0u8; logits_byte_count];
        bufs.logits
            .read_bytes(&mut logits_fp16, last_token_offset)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        let logits: Vec<f32> = logits_fp16
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f16::from_bits(bits).to_f32()
            })
            .collect();

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let num_layers = mc.num_hidden_layers;
        // 3 captures per layer: attn_norm + ffn_norm + block_output
        let captures_per_layer = 3;
        let total_captures = num_layers * captures_per_layer;
        // 3 command buffers per layer (embedding, attn, ffn) + 1 final
        let total_commits = 1 + num_layers * 2 + 1;
        tracing::info!(
            num_layers,
            total_captures,
            total_commits,
            gpu_ms = format_args!("{gpu_time_ms:.1}"),
            readback_ms = format_args!("{readback_time_ms:.1}"),
            total_ms = format_args!("{total_ms:.1}"),
            "calibration pass complete"
        );

        // Advance sequence position.
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

    /// Prefill with calibration hooks — captures activation inputs to every
    /// linear projection at each transformer layer.
    ///
    /// This is the calibration-mode equivalent of [`prefill`]. It processes
    /// all tokens in a single chunk (no chunking — calibration sequences are
    /// typically short).
    pub(crate) fn prefill_calibration(
        &mut self,
        tokens: &[u32],
        layer_callback: &mut CalibrationCallback<'_>,
    ) -> Result<Logits, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::Decode("empty calibration tokens".into()));
        }
        self.run_pipeline_calibration(tokens, layer_callback, None)
    }

    /// Run the full forward pass with [`ActivationHook`] callbacks.
    ///
    /// Wraps [`run_pipeline_calibration`](Self::run_pipeline_calibration),
    /// converting the raw byte readbacks to typed `&[f16]` slices and
    /// forwarding them to the hook. Uses the per-capture-point `n_features`
    /// from the callback (hidden_size for norms, num_heads×head_dim for
    /// O_proj, intermediate_size for down_proj).
    pub(crate) fn run_pipeline_with_hooks(
        &mut self,
        token_ids: &[u32],
        hooks: &mut dyn ActivationHook,
    ) -> Result<Logits, InferenceError> {
        self.run_pipeline_calibration(
            token_ids,
            &mut |layer, name, raw_bytes, n_features| {
                let f16_data = bytes_as_f16(raw_bytes);
                hooks.on_linear_input(layer, name, f16_data, n_features);
            },
            None,
        )
    }

    /// Prefill with [`ActivationHook`] callbacks — the calibration-mode
    /// equivalent of [`prefill`](InferenceEngine::prefill).
    ///
    /// Processes all tokens in a single chunk (calibration sequences are
    /// typically short) and invokes the hook for every linear-input
    /// readback.
    pub(crate) fn prefill_with_hooks(
        &mut self,
        tokens: &[u32],
        hooks: &mut dyn ActivationHook,
    ) -> Result<Logits, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::Decode("empty calibration tokens".into()));
        }
        self.run_pipeline_with_hooks(tokens, hooks)
    }

    /// Run a single transformer layer's forward pass (attention + FFN +
    /// residual), assuming `hidden_state` and `norm_out` are already
    /// populated by the caller.
    fn run_pipeline_calibration_from_layer(
        &mut self,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<(), InferenceError> {
        self.prepare_calibration_buffers(layer_idx, token_count)?;
        self.encode_calibration_attention_phase(layer_idx, token_count)?;
        self.encode_calibration_ffn_phase(layer_idx, token_count)?;
        self.advance_calibration_state(token_count)?;
        Ok(())
    }

    /// Ensure intermediate and scratch buffers are sized for the current
    /// token count and rebuild the matmul cache if necessary.
    fn prepare_calibration_buffers(
        &mut self,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<(), InferenceError> {
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .clone();

        self.intermediate_buffers
            .as_mut()
            .ok_or(InferenceError::NotLoaded)?
            .ensure_capacity(&self.device, token_count, &mc, self.gemma4_config.as_ref())
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        // Build or reuse MPS matmul cache for this token count.
        let need_rebuild = self
            .decode_matmuls
            .as_ref()
            .is_none_or(|c| c.token_count != token_count);
        if need_rebuild {
            let weights = self.weights.as_ref().ok_or(InferenceError::NotLoaded)?;
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

        // GDN layers need scratch capacity grown before taking immutable borrows.
        let is_gdn = self
            .weights
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .layers[layer_idx]
            .gdn_in_proj_qkv
            .is_some();
        if is_gdn {
            let h = mc.hidden_size;
            let gdn_mut = self
                .gdn_state
                .as_mut()
                .ok_or_else(|| InferenceError::runtime("GDN state not initialized"))?;
            gdn_mut
                .ensure_scratch_capacity(&self.device, token_count, h)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
        }

        Ok(())
    }

    /// Encode the attention phase: Q/K/V projections, QK-norm + RoPE,
    /// KV-cache write + attention, optional gates, output projection,
    /// and residual-add + post-attention norm.
    fn encode_calibration_attention_phase(
        &mut self,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<(), InferenceError> {
        let weights = self.weights.as_ref().ok_or(InferenceError::NotLoaded)?;
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .clone();
        let bufs = self
            .intermediate_buffers
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;
        let rope_cos = self.rope_cos.as_ref().ok_or(InferenceError::NotLoaded)?;
        let rope_sin = self.rope_sin.as_ref().ok_or(InferenceError::NotLoaded)?;
        let seq_pos = self.seq_pos;

        let h = mc.hidden_size;
        let nh = mc.num_attention_heads as u32;
        let nkv = mc.num_kv_heads() as u32;
        let hd = mc.head_dim as u32;
        let eps = mc.rms_norm_eps as f32;
        let enable_tq = self.config.enable_turboquant && self.turboquant.is_some();

        let lw = &weights.layers[layer_idx];

        // Gemma 4: use per-layer config if available.
        let (layer_hd, layer_nkv, layer_window, _layer_inter) =
            if let Some(ref g4) = self.gemma4_config {
                let lc = &g4.layer_configs[layer_idx];
                (
                    lc.head_dim as u32,
                    lc.num_kv_heads as u32,
                    lc.window_size,
                    lc.intermediate_size,
                )
            } else {
                (
                    hd,
                    nkv,
                    self.config.layer_window_size(layer_idx),
                    mc.intermediate_size,
                )
            };

        let is_gdn = lw.gdn_in_proj_qkv.is_some();

        let cmd_buf = self
            .queue
            .command_buffer()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        let enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        let default_pipelines = self.pipelines()?;

        let pipelines = if self.global_head_dim > 0
            && layer_hd as usize == self.global_head_dim
            && self.global_head_dim != mc.head_dim
        {
            self.global_pipelines.as_ref().unwrap_or(default_pipelines)
        } else {
            default_pipelines
        };

        if is_gdn {
            let gdn = self
                .gdn_state
                .as_ref()
                .ok_or_else(|| InferenceError::runtime("GDN state not initialized"))?;
            let gdn_idx = gdn.gdn_index_for_layer(layer_idx).ok_or_else(|| {
                InferenceError::runtime(format!("layer {layer_idx} is not a GDN layer"))
            })?;
            encode_gdn_prefill(&enc, bufs, gdn, lw, pipelines, gdn_idx, token_count, h, eps)?;
            enc.memory_barrier_buffers();
        } else {
            // Standard attention: Q/K/V projections.
            let qkv_out_features = mc.num_attention_heads * layer_hd as usize;
            let kv_out_features = layer_nkv as usize * layer_hd as usize;
            for (weight, output_buf, out_features) in [
                (&lw.q_proj, &bufs.q_proj, qkv_out_features),
                (&lw.k_proj, &bufs.k_proj, kv_out_features),
                (&lw.v_proj, &bufs.v_proj, kv_out_features),
            ] {
                encode_projection(
                    &enc,
                    &bufs.norm_out,
                    weight,
                    output_buf,
                    pipelines,
                    token_count,
                    out_features,
                    h,
                )?;
            }
            enc.memory_barrier_buffers();

            // Qwen3.5 attn_output_gate.
            if let (Some(gate_w), Some(gate_buf)) = (&lw.attn_output_gate, &bufs.q_gate) {
                encode_projection(
                    &enc,
                    &bufs.norm_out,
                    gate_w,
                    gate_buf,
                    pipelines,
                    token_count,
                    qkv_out_features,
                    h,
                )?;
                enc.memory_barrier_buffers();
            }

            // QK normalization + RoPE.
            let (layer_rope_cos, layer_rope_sin) = if let Some(ref g4) = self.gemma4_config {
                let lc = &g4.layer_configs[layer_idx];
                if lc.is_global {
                    (
                        self.global_rope_cos.as_ref().unwrap_or(rope_cos),
                        self.global_rope_sin.as_ref().unwrap_or(rope_sin),
                    )
                } else {
                    (rope_cos, rope_sin)
                }
            } else {
                (rope_cos, rope_sin)
            };
            encode_qk_norm_and_rope(
                &enc,
                pipelines,
                &QkNormAndRopeParams {
                    bufs,
                    q_norm: lw.q_norm.as_ref(),
                    k_norm: lw.k_norm.as_ref(),
                    rope_cos: layer_rope_cos,
                    rope_sin: layer_rope_sin,
                    nh,
                    nkv: layer_nkv,
                    hd: layer_hd,
                    seq_pos,
                    token_count,
                    eps,
                },
            )?;
            enc.memory_barrier_buffers();

            // KV cache write + attention.
            let is_anchor = self
                .config
                .cla_config
                .as_ref()
                .is_none_or(|cla| cla.is_anchor(layer_idx));

            let (is_anchor, kv_cache_layer) = if let Some(ref g4) = self.gemma4_config {
                if let Some(anchor) = g4.layer_configs[layer_idx].kv_anchor {
                    (false, anchor)
                } else {
                    (is_anchor, layer_idx)
                }
            } else {
                (is_anchor, layer_idx)
            };

            let attn_scale = mc.attn_scale();

            encode_kv_cache_and_attention(
                &enc,
                pipelines,
                &KvCacheAndAttentionParams {
                    bufs,
                    turboquant: self.turboquant.as_ref(),
                    kv_cache: self.kv_cache.as_ref(),
                    fp16_kv_cache: self.fp16_kv_cache.as_ref(),
                    max_seq_len: self.config.max_seq_len,
                    n_bits: self.config.n_bits as usize,
                    layer_idx: kv_cache_layer,
                    seq_pos,
                    token_count,
                    nh,
                    nkv: layer_nkv,
                    hd: layer_hd,
                    enable_tq,
                    is_anchor,
                    window_size: layer_window,
                    attn_scale,
                    gpu_max_threadgroups: self.gpu_max_threadgroups,
                },
            )?;
            enc.memory_barrier_buffers();

            // Qwen3.5 attn_output_gate: apply sigmoid(gate).
            if let Some(gate_buf) = &bufs.q_gate {
                let gate_size = (token_count * mc.num_attention_heads * mc.head_dim) as u32;
                ops::encode_sigmoid_gate(
                    &enc,
                    &pipelines.activation.sigmoid_gate,
                    &bufs.attn_out,
                    gate_buf,
                    gate_size,
                );
                enc.memory_barrier_buffers();
            }

            // Output projection.
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
            enc.memory_barrier_buffers();

            // Residual add + post-attention RMSNorm.
            if let Some(pre_ffn) = &lw.pre_ffn_norm {
                ops::encode_rms_norm(
                    &enc,
                    &pipelines.norm.rms_norm,
                    &ops::RmsNormParams {
                        input: &bufs.ffn_down,
                        weight: &lw.post_attn_norm,
                        output: &bufs.ffn_down,
                        hidden_size: h as u32,
                        token_count: token_count as u32,
                        eps,
                    },
                );
                enc.memory_barrier_buffers();
                ops::encode_residual_add(
                    &enc,
                    &pipelines.elementwise.residual_add,
                    &bufs.hidden_state,
                    &bufs.ffn_down,
                    &bufs.residual,
                    (token_count * h) as u32,
                );
                enc.memory_barrier_buffers();
                ops::encode_rms_norm(
                    &enc,
                    &pipelines.norm.rms_norm,
                    &ops::RmsNormParams {
                        input: &bufs.residual,
                        weight: pre_ffn,
                        output: &bufs.norm_out,
                        hidden_size: h as u32,
                        token_count: token_count as u32,
                        eps,
                    },
                );
            } else {
                ops::encode_fused_residual_rms_norm(
                    &enc,
                    &pipelines.norm.fused_residual_rms_norm,
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
            }
        }

        enc.end_encoding();

        // Mid-layer commit: attention block done.
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        if cmd_buf.status() == CommandBufferStatus::Error {
            return Err(InferenceError::Decode(format!(
                "Metal command buffer failed (layer {layer_idx} attention phase)"
            )));
        }

        Ok(())
    }

    /// Encode the FFN phase: feed-forward block, optional MoE, post-FFN
    /// norm, PLE dispatch, and end-of-layer residual.
    fn encode_calibration_ffn_phase(
        &mut self,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<(), InferenceError> {
        let weights = self.weights.as_ref().ok_or(InferenceError::NotLoaded)?;
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .clone();
        let bufs = self
            .intermediate_buffers
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;

        let h = mc.hidden_size;
        let inter = mc.intermediate_size;
        let eps = mc.rms_norm_eps as f32;

        let lw = &weights.layers[layer_idx];
        let layer_inter = if let Some(ref g4) = self.gemma4_config {
            g4.layer_configs[layer_idx].intermediate_size
        } else {
            inter
        };

        let cmd_buf = self
            .queue
            .command_buffer()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        let enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        let pipelines = self.pipelines()?;

        let use_gelu = mc.use_gelu();
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
        enc.memory_barrier_buffers();

        // MoE block (Gemma 4).
        if let Some(ref g4) = self.gemma4_config {
            let lc = &g4.layer_configs[layer_idx];
            if lc.enable_moe && g4.num_experts > 0 && !lw.expert_gate_projs.is_empty() {
                encode_moe_block(
                    &enc,
                    pipelines,
                    bufs,
                    lw,
                    h,
                    g4.moe_intermediate_size,
                    token_count,
                    g4.num_experts,
                    g4.top_k_experts,
                )?;
            }
        }

        // Post-FFN norm (Gemma 4).
        if let Some(ref post_ffn) = lw.post_ffn_norm {
            ops::encode_rms_norm(
                &enc,
                &pipelines.norm.rms_norm,
                &ops::RmsNormParams {
                    input: &bufs.ffn_down,
                    weight: post_ffn,
                    output: &bufs.ffn_gate,
                    hidden_size: h as u32,
                    token_count: token_count as u32,
                    eps,
                },
            );
            enc.memory_barrier_buffers();
            ops::encode_copy_buffer(
                &enc,
                &pipelines.elementwise.copy_buffer,
                &bufs.ffn_gate,
                &bufs.ffn_down,
                (token_count * h) as u32,
            );
            enc.memory_barrier_buffers();
        }

        // PLE per-layer dispatch.
        let next_input_norm = if layer_idx + 1 < mc.num_hidden_layers {
            Some(&weights.layers[layer_idx + 1].input_norm)
        } else {
            None
        };
        let ple_applied = ple::encode_ple_per_layer(
            &enc,
            pipelines,
            &ple::PlePerLayerParams {
                bufs,
                lw,
                next_input_norm,
                gemma4_config: self.gemma4_config.as_ref(),
                num_hidden_layers: mc.num_hidden_layers,
                layer_idx,
                token_count,
                h,
                eps,
            },
        )?;

        if !ple_applied {
            if let Some(scalar) = &lw.layer_scalar {
                ops::encode_residual_add(
                    &enc,
                    &pipelines.elementwise.residual_add,
                    &bufs.residual,
                    &bufs.ffn_down,
                    &bufs.hidden_state,
                    (token_count * h) as u32,
                );
                enc.memory_barrier_buffers();
                ops::encode_scale_buffer(
                    &enc,
                    &pipelines.elementwise.scale_buffer,
                    &bufs.hidden_state,
                    scalar,
                    (token_count * h) as u32,
                );
                enc.memory_barrier_buffers();
                if layer_idx + 1 < mc.num_hidden_layers {
                    let next_norm = &weights.layers[layer_idx + 1].input_norm;
                    ops::encode_rms_norm(
                        &enc,
                        &pipelines.norm.rms_norm,
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
                let next_norm = if layer_idx + 1 < mc.num_hidden_layers {
                    Some(&weights.layers[layer_idx + 1].input_norm)
                } else {
                    None
                };
                encode_end_of_layer_residual(
                    &enc,
                    pipelines,
                    bufs,
                    next_norm,
                    None,
                    h,
                    token_count,
                    eps,
                )?;
            }
        }
        enc.end_encoding();

        // End-of-layer commit.
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        if cmd_buf.status() == CommandBufferStatus::Error {
            return Err(InferenceError::Decode(format!(
                "Metal command buffer failed (layer {layer_idx} FFN phase)"
            )));
        }

        Ok(())
    }

    /// Advance sequence position and KV cache state after a layer completes.
    fn advance_calibration_state(&mut self, token_count: usize) -> Result<(), InferenceError> {
        let enable_tq = self.config.enable_turboquant && self.turboquant.is_some();

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

        Ok(())
    }
}

// ── GPU calibration trait ─────────────────────────────────────────

/// GPU-specific calibration operations on a loaded inference engine.
///
/// Provides the building blocks for block-level AWQ alpha search,
/// clip calibration, and DAC bias computation. These methods are
/// deliberately separated from the core [`InferenceEngine`] trait so
/// that production inference code never sees weight-swapping or
/// single-layer evaluation in its API surface.
///
/// Import this trait to access calibration methods on [`MetalInference`]:
/// ```ignore
/// use ironmill_inference::metal::GpuCalibrationEngine;
/// engine.calibrate_dac(&fp_provider, &tokens)?;
/// engine.run_single_layer(layer_idx, &input, token_count)?;
/// ```
pub trait GpuCalibrationEngine {
    /// The weight buffer type used by this engine.
    type WeightBuffer;

    /// Run DAC (Deviation-Aware Correction) calibration.
    ///
    /// Computes per-layer bias vectors that compensate for quantization-
    /// induced activation drift. The biases are stored on the engine
    /// and applied automatically during subsequent inference.
    ///
    /// Reference: D²Quant (arXiv:2602.02546) §3.3, Algorithm 1.
    fn calibrate_dac(
        &mut self,
        fp_provider: &dyn mil_rs::weights::WeightProvider,
        calibration_tokens: &[u32],
    ) -> Result<(), InferenceError>;

    /// Run a single transformer layer forward on GPU.
    ///
    /// Writes `input_hidden_state` (FP16 bytes) to the hidden_state
    /// buffer, runs one layer, and returns the output hidden state.
    /// Safe to call repeatedly for the same layer without accumulating
    /// KV cache state.
    fn run_single_layer(
        &mut self,
        layer_idx: usize,
        input_hidden_state: &[u8],
        token_count: usize,
    ) -> Result<Vec<u8>, InferenceError>;

    /// Swap a layer's projection weight, returning the original.
    ///
    /// `proj_name` is one of: `"q_proj"`, `"k_proj"`, `"v_proj"`,
    /// `"o_proj"`, `"gate_proj"`, `"up_proj"`, `"down_proj"`.
    fn swap_layer_weight(
        &mut self,
        layer_idx: usize,
        proj_name: &str,
        new_weight: Self::WeightBuffer,
    ) -> Option<Self::WeightBuffer>;

    /// Create a Dense FP16 GPU buffer from CPU f32 data.
    fn create_dense_f16_buffer(
        &self,
        data_f32: &[f32],
    ) -> Result<Self::WeightBuffer, InferenceError>;

    /// Create an empty Dense FP16 GPU buffer for an `[n, k]` weight matrix.
    ///
    /// Allocates both row-major and packed blocked buffers so the weight
    /// is immediately usable by projection kernels. Use
    /// [`update_weight_buffer_f16`] to populate it before dispatching.
    fn create_dense_f16_buffer_sized(
        &self,
        n: usize,
        k: usize,
    ) -> Result<Self::WeightBuffer, InferenceError>;
}

/// Overwrite the contents of a Dense FP16 [`WeightBuffer`](super::weights::WeightBuffer)
/// with new f32 data, avoiding a GPU allocation.
///
/// Updates both the row-major and packed blocked buffers. `n` and `k`
/// are the matrix dimensions (out_features, in_features).
///
/// This is a free function rather than a trait method because it
/// operates on the buffer alone, without engine state.
pub fn update_weight_buffer_f16(
    wb: &super::weights::WeightBuffer,
    data_f32: &[f32],
    n: usize,
    k: usize,
) -> Result<(), InferenceError> {
    super::weights::update_dense_f16_data(wb, data_f32, n, k)
        .map_err(|e| InferenceError::runtime(e.to_string()))
}

impl GpuCalibrationEngine for MetalInference {
    type WeightBuffer = super::weights::WeightBuffer;

    fn calibrate_dac(
        &mut self,
        fp_provider: &dyn mil_rs::weights::WeightProvider,
        calibration_tokens: &[u32],
    ) -> Result<(), InferenceError> {
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .clone();
        let h = mc.hidden_size;
        let num_layers = mc.num_hidden_layers;
        let token_count = calibration_tokens.len();

        if token_count == 0 {
            return Ok(());
        }

        // 1. Run FP16 reference model to capture post-attention norm outputs.
        let fp_norms = {
            let mut fp_engine = MetalInference::new(self.config.clone())
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            fp_engine
                .load_weights(fp_provider, self.config.clone())
                .map_err(|e| InferenceError::runtime(format!("DAC FP16 load: {e}")))?;

            let mut norms: Vec<Vec<f32>> = vec![vec![0.0f32; h]; num_layers];
            fp_engine.prefill_calibration(
                calibration_tokens,
                &mut |layer, name, data, _n_features| {
                    if name == "ffn_norm" && layer < num_layers {
                        let f16_vals: Vec<f32> = data
                            .chunks_exact(2)
                            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                            .collect();
                        for ch in 0..h {
                            let sum: f32 = (0..token_count).map(|t| f16_vals[t * h + ch]).sum();
                            norms[layer][ch] = sum / token_count as f32;
                        }
                    }
                },
            )?;
            norms
        };

        // 2. Run quantized (self) model to capture post-attention norm outputs.
        let q_norms = {
            let mut norms: Vec<Vec<f32>> = vec![vec![0.0f32; h]; num_layers];
            self.prefill_calibration(calibration_tokens, &mut |layer, name, data, _n_features| {
                if name == "ffn_norm" && layer < num_layers {
                    let f16_vals: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect();
                    for ch in 0..h {
                        let sum: f32 = (0..token_count).map(|t| f16_vals[t * h + ch]).sum();
                        norms[layer][ch] = sum / token_count as f32;
                    }
                }
            })?;
            norms
        };

        // 3. Compute per-layer correction bias: μ = mean(Y_fp) - mean(Y_q)
        let mut biases = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let bias_f16: Vec<u8> = (0..h)
                .flat_map(|ch| {
                    let deviation = fp_norms[layer][ch] - q_norms[layer][ch];
                    half::f16::from_f32(deviation).to_le_bytes()
                })
                .collect();
            let buf = self
                .device
                .create_buffer_with_data(&bias_f16, StorageMode::Shared)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            biases.push(buf);
        }

        self.dac_biases = Some(biases);
        self.reset();
        Ok(())
    }

    fn run_single_layer(
        &mut self,
        layer_idx: usize,
        input_hidden_state: &[u8],
        token_count: usize,
    ) -> Result<Vec<u8>, InferenceError> {
        self.seq_pos = 0;

        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .clone();
        self.intermediate_buffers
            .as_mut()
            .ok_or(InferenceError::NotLoaded)?
            .ensure_capacity(&self.device, token_count, &mc, self.gemma4_config.as_ref())
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        let h = mc.hidden_size;
        let io_bytes = token_count * h * 2; // FP16

        // Lazily allocate (or grow) reusable staging buffers.
        let need_alloc = self
            .calib_scratch
            .as_ref()
            .is_none_or(|s| s.capacity < io_bytes);
        if need_alloc {
            let alloc_bytes = io_bytes.max(16);
            let staging_in = self
                .device
                .create_buffer(alloc_bytes, StorageMode::Shared)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let readback = self
                .device
                .create_buffer(alloc_bytes, StorageMode::Shared)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.calib_scratch = Some(super::engine::CalibScratchBuffers {
                staging_in,
                readback,
                capacity: alloc_bytes,
            });
        }
        let scratch = self.calib_scratch.as_ref().unwrap();

        // Write input data to the reusable staging buffer.
        scratch
            .staging_in
            .write_bytes(input_hidden_state, 0)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        // Commit 1: copy-in + input norm (merged).
        let bufs = self
            .intermediate_buffers
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;
        let weights = self.weights.as_ref().ok_or(InferenceError::NotLoaded)?;
        let eps = mc.rms_norm_eps as f32;
        let lw = &weights.layers[layer_idx];

        let cmd = self
            .queue
            .command_buffer()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        {
            let enc = cmd
                .compute_encoder()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            ops::encode_copy_buffer(
                &enc,
                &self.pipelines()?.elementwise.copy_buffer,
                &scratch.staging_in,
                &bufs.hidden_state,
                (token_count * h) as u32,
            );
            enc.memory_barrier_buffers();
            ops::encode_rms_norm(
                &enc,
                &self.pipelines()?.norm.rms_norm,
                &ops::RmsNormParams {
                    input: &bufs.hidden_state,
                    weight: &lw.input_norm,
                    output: &bufs.norm_out,
                    hidden_size: h as u32,
                    token_count: token_count as u32,
                    eps,
                },
            );
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();

        // Commit 2+3: attention phase + FFN phase (inside the pipeline helper).
        self.run_pipeline_calibration_from_layer(layer_idx, token_count)?;

        // Readback: copy hidden_state (Private) → reusable readback (Shared).
        self.seq_pos = 0;
        let bufs = self
            .intermediate_buffers
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;
        let scratch = self.calib_scratch.as_ref().unwrap();
        {
            let cmd = self
                .queue
                .command_buffer()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let enc = cmd
                .compute_encoder()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            ops::encode_copy_buffer(
                &enc,
                &self.pipelines()?.elementwise.copy_buffer,
                &bufs.hidden_state,
                &scratch.readback,
                (token_count * h) as u32,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let mut output = vec![0u8; io_bytes];
        scratch
            .readback
            .read_bytes(&mut output, 0)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        Ok(output)
    }

    fn swap_layer_weight(
        &mut self,
        layer_idx: usize,
        proj_name: &str,
        new_weight: super::weights::WeightBuffer,
    ) -> Option<super::weights::WeightBuffer> {
        self.weights
            .as_mut()?
            .swap_layer_weight(layer_idx, proj_name, new_weight)
    }

    fn create_dense_f16_buffer(
        &self,
        data_f32: &[f32],
    ) -> Result<super::weights::WeightBuffer, InferenceError> {
        super::weights::create_dense_f16_buffer(&self.device, data_f32)
            .map_err(|e| InferenceError::runtime(e.to_string()))
    }

    fn create_dense_f16_buffer_sized(
        &self,
        n: usize,
        k: usize,
    ) -> Result<super::weights::WeightBuffer, InferenceError> {
        super::weights::create_dense_f16_buffer_sized(&self.device, n, k)
            .map_err(|e| InferenceError::runtime(e.to_string()))
    }
}
