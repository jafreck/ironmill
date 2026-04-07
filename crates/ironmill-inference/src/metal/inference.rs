//! Calibration and hook-based pipeline helpers.
//!
//! The [`MetalInference`] struct and trait impls live in [`super::engine`].
//! Weight loading and initialization live in [`super::loading`].
//! This module contains calibration-mode pipelines and activation-hook
//! bridges.

use std::time::Instant;

use half::f16;
use ironmill_metal_sys::{
    CommandBufferStatus, ComputeEncoder, MetalBuffer, MetalDevice, StorageMode,
};
use mil_rs::weights::{Architecture, ModelConfig, WeightProvider};

use super::attention::{
    encode_end_of_layer_residual, encode_kv_cache_and_attention, encode_qk_norm_and_rope,
};
use super::buffers::{
    IntermediateBuffers, ModelConfigExt, MpsMatmulCache,
    build_matmul_cache, build_rope_cache, bytes_as_f16,
};
use super::config::Gemma4Config;
use super::config::MetalConfig;
use super::engine::{MetalArtifacts, MetalInference};
use super::error::MetalError;
use super::ffn::{encode_ffn_block, encode_moe_block};
use super::gdn::{GdnState, encode_gdn_prefill};
use super::kv_cache::Fp16KvCache;
use super::mla::{MlaConfig, MlaKvCache};
use super::ops;
use super::ops::LinearKernelKind;
use super::plan::{LayerPlan, ModelPlan};
use super::ple;
use super::projection::encode_projection;
use super::turboquant::{
    MetalKvCache, MetalTurboQuantModel, OutlierConfig, TurboQuantLayerConfig, TurboQuantMetalConfig,
};
use super::weights::{
    AffineQuantizedWeight, DualScaleQuantizedWeight, MetalWeights, QuantizedWeight,
    WeightBuffer,
};
use crate::calibration::ActivationHook;
use crate::engine::{InferenceEngine, InferenceError};
use crate::types::Logits;
use ironmill_core::model_info::ModelInfo;

// ── Matmul tile dimensions — must match Metal shader constants ──
const MATMUL_TM_TILE: usize = 64;
const MATMUL_TN_TILE: usize = 64;
const MATMUL_THREADS_PER_TG: usize = 256;


impl MetalInference {
    // ── DAC (Deviation-Aware Correction) ────────────────────────

    /// Calibrate D2Quant Deviation-Aware Correction biases.
    ///
    /// Runs calibration tokens through a full-precision reference model and
    /// the current (quantized) model, computing the per-channel mean
    /// deviation at each layer's post-attention LayerNorm output. The
    /// resulting bias vectors are stored and applied during inference to
    /// compensate for quantization-induced activation drift.
    ///
    /// Reference: D²Quant (arXiv:2602.02546) §3.3, Algorithm 1 lines 3–10.
    pub fn calibrate_dac(
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
            fp_engine.prefill_calibration(calibration_tokens, &mut |layer, name, data| {
                if name == "ffn_norm" && layer < num_layers {
                    // data is FP16 bytes: [token_count × hidden_size]
                    let f16_vals: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect();
                    // Mean across tokens for each channel
                    for ch in 0..h {
                        let sum: f32 = (0..token_count).map(|t| f16_vals[t * h + ch]).sum();
                        norms[layer][ch] = sum / token_count as f32;
                    }
                }
            })?;
            norms
        };
        // FP16 engine is dropped here, freeing GPU memory.

        // 2. Run quantized (self) model to capture post-attention norm outputs.
        let q_norms = {
            let mut norms: Vec<Vec<f32>> = vec![vec![0.0f32; h]; num_layers];
            self.prefill_calibration(calibration_tokens, &mut |layer, name, data| {
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
    /// - `projection_name`: one of `"attn_norm"` (input to Q/K/V/O projections)
    ///   or `"ffn_norm"` (input to gate/up/down projections)
    /// - `raw_bytes`: the FP16 activation data (token_count × hidden_size × 2 bytes)
    ///
    /// Returns the same logits as [`run_pipeline`].
    pub fn run_pipeline_calibration(
        &mut self,
        token_ids: &[u32],
        layer_callback: &mut dyn FnMut(usize, &str, &[u8]),
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
                &self.pipelines()?.embedding_lookup,
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
                    &self.pipelines()?.scale_buffer,
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
                &self.pipelines()?.rms_norm,
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
                &self.device,
                weights,
                bufs,
                &bufs.token_ids_buf,
                &mc,
                self.gemma4_config.as_ref(),
                token_count,
                h,
                vocab,
                eps,
                false,
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
                #[allow(unsafe_code)]
                let readback = unsafe {
                    std::slice::from_raw_parts_mut(
                        readback_u16.as_mut_ptr() as *mut u8,
                        norm_readback_bytes,
                    )
                };
                bufs.norm_out
                    .read_bytes(readback, 0)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                readback_time_ms += rb_start.elapsed().as_secs_f64() * 1000.0;
                layer_callback(layer_idx, "attn_norm", readback);
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

            let cmd_buf = self
                .queue
                .command_buffer()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let enc = cmd_buf
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
                    bufs,
                    self.turboquant.as_ref(),
                    self.kv_cache.as_ref(),
                    self.fp16_kv_cache.as_ref(),
                    self.config.max_seq_len,
                    self.config.n_bits as usize,
                    kv_cache_layer,
                    seq_pos,
                    token_count,
                    nh,
                    layer_nkv,
                    layer_hd,
                    enable_tq,
                    is_anchor,
                    layer_window,
                    attn_scale,
                )?;
                enc.memory_barrier_buffers();

                // Qwen3.5 attn_output_gate: apply sigmoid(gate) to attention output.
                if let Some(gate_buf) = &bufs.q_gate {
                    let gate_size = (token_count * mc.num_attention_heads * mc.head_dim) as u32;
                    ops::encode_sigmoid_gate(
                        &enc,
                        &pipelines.sigmoid_gate,
                        &bufs.attn_out,
                        gate_buf,
                        gate_size,
                    );
                    enc.memory_barrier_buffers();
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
                    enc.memory_barrier_buffers();
                    ops::encode_residual_add(
                        &enc,
                        &pipelines.residual_add,
                        &bufs.hidden_state,
                        &bufs.ffn_down,
                        &bufs.residual,
                        (token_count * h) as u32,
                    );
                    enc.memory_barrier_buffers();
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
                } else {
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
                #[allow(unsafe_code)]
                let readback = unsafe {
                    std::slice::from_raw_parts_mut(
                        readback_u16.as_mut_ptr() as *mut u8,
                        norm_readback_bytes,
                    )
                };
                bufs.norm_out
                    .read_bytes(readback, 0)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                readback_time_ms += rb_start.elapsed().as_secs_f64() * 1000.0;
                layer_callback(layer_idx, "ffn_norm", readback);
            }

            // ── New command buffer for FFN block ────────────────
            let cmd_buf = self
                .queue
                .command_buffer()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let enc = cmd_buf
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
                enc.memory_barrier_buffers();
                ops::encode_copy_buffer(
                    &enc,
                    &pipelines.copy_buffer,
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
                    ops::encode_residual_add(
                        &enc,
                        &pipelines.residual_add,
                        &bufs.residual,
                        &bufs.ffn_down,
                        &bufs.hidden_state,
                        (token_count * h) as u32,
                    );
                    enc.memory_barrier_buffers();
                    ops::encode_scale_buffer(
                        &enc,
                        &pipelines.scale_buffer,
                        &bufs.hidden_state,
                        scalar,
                        (token_count * h) as u32,
                    );
                    enc.memory_barrier_buffers();
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
        // 2 captures per layer: attn_norm + ffn_norm
        let captures_per_layer = 2;
        let total_captures = num_layers * captures_per_layer;
        // 3 command buffers per layer (embedding, attn, ffn) + 1 final
        let total_commits = 1 + num_layers * 2 + 1;
        eprintln!(
            "[calibration] {num_layers} layers, {total_captures} captures, \
             {total_commits} commits | GPU: {gpu_time_ms:.1}ms, \
             readback: {readback_time_ms:.1}ms, total: {total_ms:.1}ms"
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
    pub fn prefill_calibration(
        &mut self,
        tokens: &[u32],
        layer_callback: &mut dyn FnMut(usize, &str, &[u8]),
    ) -> Result<Logits, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::Decode("empty calibration tokens".into()));
        }
        self.run_pipeline_calibration(tokens, layer_callback)
    }

    /// Run the full forward pass with [`ActivationHook`] callbacks.
    ///
    /// Wraps [`run_pipeline_calibration`](Self::run_pipeline_calibration),
    /// converting the raw byte readbacks to typed `&[f16]` slices and
    /// forwarding them to the hook. `n_features` is the model's
    /// `hidden_size` (both `attn_norm` and `ffn_norm` outputs have this
    /// dimensionality).
    pub fn run_pipeline_with_hooks(
        &mut self,
        token_ids: &[u32],
        hooks: &mut dyn ActivationHook,
    ) -> Result<Logits, InferenceError> {
        let hidden_size = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .hidden_size;

        self.run_pipeline_calibration(token_ids, &mut |layer, name, raw_bytes| {
            let f16_data = bytes_as_f16(raw_bytes);
            hooks.on_linear_input(layer, name, f16_data, hidden_size);
        })
    }

    /// Prefill with [`ActivationHook`] callbacks — the calibration-mode
    /// equivalent of [`prefill`](InferenceEngine::prefill).
    ///
    /// Processes all tokens in a single chunk (calibration sequences are
    /// typically short) and invokes the hook for every linear-input
    /// readback.
    pub fn prefill_with_hooks(
        &mut self,
        tokens: &[u32],
        hooks: &mut dyn ActivationHook,
    ) -> Result<Logits, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::Decode("empty calibration tokens".into()));
        }
        self.run_pipeline_with_hooks(tokens, hooks)
    }

}

// ── Calibration tests ──────────────────────────────────────────
//
// These tests require a real Metal device (macOS only) and a loaded model.
// They are gated behind `#[cfg(test)]` and will be skipped in CI without
// Metal hardware. For local validation, run:
//   cargo test -p ironmill-inference --features metal -- calibration
//
// Without a model fixture these serve as compile-time validation of the
// calibration API surface.  The `_api_surface` test verifies the method
// signatures and callback types compile correctly.

#[cfg(test)]
mod calibration_tests {
    use super::*;

    /// Verify that the calibration method signature compiles and the
    /// callback type is properly accepted as a trait object.
    #[test]
    fn calibration_api_surface_compiles() {
        // This test validates at compile time that:
        // 1. run_pipeline_calibration accepts &mut dyn FnMut(usize, &str, &[u8])
        // 2. prefill_calibration accepts the same callback type
        // 3. Both return Result<Logits, InferenceError>
        //
        // We cannot run inference without a loaded model, but we can
        // verify the type signatures are correct.

        fn _assert_method_exists(engine: &mut MetalInference) {
            let mut count = 0usize;
            let mut callback = |layer: usize, name: &str, data: &[u8]| {
                let _ = (layer, name, data);
                count += 1;
            };
            // These calls would fail at runtime (no model loaded), but they
            // prove the API compiles.
            let _ = engine.run_pipeline_calibration(&[1, 2, 3], &mut callback);
            let _ = engine.prefill_calibration(&[1, 2, 3], &mut callback);
        }

        // Just verify the function compiles — don't actually call it.
        let _ = _assert_method_exists;
    }

    /// Verify that run_pipeline_with_hooks and prefill_with_hooks signatures
    /// compile and accept `&mut dyn ActivationHook`.
    #[test]
    fn hook_bridge_api_surface_compiles() {
        use crate::calibration::{ActivationHook, AwqActivationStore};

        fn _assert_hook_methods(engine: &mut MetalInference) {
            let mut store = AwqActivationStore::new();
            // These would fail at runtime (no model loaded) but prove
            // the API surface compiles.
            let _ = engine.run_pipeline_with_hooks(&[1, 2, 3], &mut store);
            let _ = engine.prefill_with_hooks(&[1, 2, 3], &mut store);
        }

        let _ = _assert_hook_methods;
    }

    /// Verify that `MetalInference` implements `CalibratingEngine` and the
    /// trait methods compile with the expected signatures.
    #[test]
    fn calibrating_engine_impl_compiles() {
        use crate::calibration::{AwqActivationStore, CalibratingEngine};

        fn _assert_calibrating_engine(engine: &mut MetalInference) {
            let mut store = AwqActivationStore::new();
            let _ = CalibratingEngine::prefill_with_hooks(engine, &[1, 2, 3], &mut store);
            CalibratingEngine::reset(engine);
        }

        let _ = _assert_calibrating_engine;
    }

    /// Verify that `bytes_as_f16` correctly reinterprets raw bytes.
    #[test]
    fn bytes_as_f16_roundtrip() {
        let values = [f16::from_f32(1.0), f16::from_f32(-2.5), f16::from_f32(0.0)];
        // Serialize to bytes in native byte order.
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_bits().to_ne_bytes())
            .collect();

        let converted = bytes_as_f16(&bytes);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0], f16::from_f32(1.0));
        assert_eq!(converted[1], f16::from_f32(-2.5));
        assert_eq!(converted[2], f16::from_f32(0.0));
    }

    /// Verify that `bytes_as_f16` panics on an odd-length byte slice.
    #[test]
    #[should_panic(expected = "not a multiple of f16 size")]
    fn bytes_as_f16_rejects_odd_length() {
        bytes_as_f16(&[0u8; 3]);
    }

    /// Verify that a closure capturing mutable state works as the callback.
    #[test]
    fn calibration_callback_captures_state() {
        // Simulate what a real calibration consumer would do: accumulate
        // activation statistics across layers.
        struct ActivationStats {
            captures: Vec<(usize, String, usize)>, // (layer, name, byte_count)
        }

        let mut stats = ActivationStats {
            captures: Vec::new(),
        };

        // Build a callback that captures &mut stats.
        let mut callback = |layer: usize, name: &str, data: &[u8]| {
            stats.captures.push((layer, name.to_string(), data.len()));
        };

        // Simulate the callback being invoked as it would be during calibration.
        // 2 layers × 2 captures per layer = 4 invocations.
        let hidden_size = 128;
        let token_count = 8;
        let fake_data = vec![0u8; token_count * hidden_size * 2]; // FP16

        for layer in 0..2 {
            callback(layer, "attn_norm", &fake_data);
            callback(layer, "ffn_norm", &fake_data);
        }

        assert_eq!(stats.captures.len(), 4);
        assert_eq!(
            stats.captures[0],
            (0, "attn_norm".to_string(), fake_data.len())
        );
        assert_eq!(
            stats.captures[1],
            (0, "ffn_norm".to_string(), fake_data.len())
        );
        assert_eq!(
            stats.captures[2],
            (1, "attn_norm".to_string(), fake_data.len())
        );
        assert_eq!(
            stats.captures[3],
            (1, "ffn_norm".to_string(), fake_data.len())
        );

        // Verify expected byte size: token_count × hidden_size × 2 (FP16)
        for (_, _, byte_count) in &stats.captures {
            assert_eq!(*byte_count, token_count * hidden_size * 2);
        }
    }

    /// Verify the INT4 dequant shader compiles on the current Metal device.
    #[test]
    fn int4_dequant_shader_compiles_on_device() {
        use ironmill_metal_sys::MetalDevice;

        let device = match MetalDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Skipping — no Metal device available");
                return;
            }
        };

        let src = include_str!("shaders/int4_dequant.metal");
        let lib = device
            .compile_shader_source(src)
            .expect("int4_dequant.metal should compile");
        let func = lib
            .get_function("int4_dequantize")
            .expect("int4_dequantize function should exist");
        let _pipeline = device
            .create_compute_pipeline(&func)
            .expect("should create compute pipeline");
    }
}

// ── FA2 prefill attention correctness tests ────────────────────
//
// These tests verify that the FlashAttention-2 prefill kernel produces
// the same output as the fused SDPA kernel for the same inputs.
// Requires a Metal GPU.

#[cfg(test)]
mod fa2_prefill_tests {
    use half::f16;
    use ironmill_metal_sys::{MetalDevice, StorageMode};

    /// Create a Metal buffer filled with FP16 data from f32 values.
    fn make_fp16_buffer(device: &MetalDevice, data: &[f32]) -> ironmill_metal_sys::MetalBuffer {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        device
            .create_buffer_with_data(&bytes, StorageMode::Shared)
            .expect("create buffer")
    }

    /// Read FP16 buffer back as f32 values.
    fn read_fp16_buffer(buf: &ironmill_metal_sys::MetalBuffer, count: usize) -> Vec<f32> {
        let byte_count = count * 2;
        let mut bytes = vec![0u8; byte_count];
        buf.read_bytes(&mut bytes, 0).expect("read_bytes");
        bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }

    /// CPU reference: causal scaled dot-product attention.
    ///
    /// Q: [token_count, num_q_heads, head_dim]
    /// K/V cache: [num_kv_heads, max_seq_len, head_dim] (filled up to seq_offset + token_count)
    fn cpu_attention(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        seq_offset: usize,
        token_count: usize,
        scale: f32,
    ) -> Vec<f32> {
        let heads_per_group = num_q_heads / num_kv_heads;
        let mut output = vec![0.0f32; token_count * num_q_heads * head_dim];

        for t in 0..token_count {
            let causal_len = seq_offset + t + 1;
            for h in 0..num_q_heads {
                let kv_h = h / heads_per_group;
                let q_base = (t * num_q_heads + h) * head_dim;

                // Compute QK^T scores
                let mut scores = vec![-f32::INFINITY; causal_len];
                for p in 0..causal_len {
                    let k_base = (kv_h * max_seq_len + p) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k_cache[k_base + d];
                    }
                    scores[p] = dot * scale;
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }

                // Weighted sum of V
                let o_base = (t * num_q_heads + h) * head_dim;
                for p in 0..causal_len {
                    let v_base = (kv_h * max_seq_len + p) * head_dim;
                    for d in 0..head_dim {
                        output[o_base + d] += scores[p] * v_cache[v_base + d];
                    }
                }
            }
        }
        output
    }

    /// Generate deterministic pseudo-random f32 values in [-1, 1].
    fn pseudo_random(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Fill KV cache positions 0..token_count from flat fill arrays.
    fn fill_kv_cache(
        cache: &mut [f32],
        fill: &[f32],
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        token_count: usize,
    ) {
        for kv_h in 0..num_kv_heads {
            for t in 0..token_count {
                for d in 0..head_dim {
                    cache[kv_h * max_seq_len * head_dim + t * head_dim + d] =
                        fill[kv_h * token_count * head_dim + t * head_dim + d];
                }
            }
        }
    }

    /// Verify FA2 prefill produces the same output as fused SDPA.
    ///
    /// Uses head_dim=128 (precompiled shaders), 4 Q heads, 2 KV heads,
    /// 8 tokens of prefill.
    #[test]
    fn fa2_matches_fused_sdpa() {
        let device = match MetalDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP: no Metal device");
                return;
            }
        };
        let queue = device.create_command_queue().expect("command queue");

        let head_dim = 128usize;
        let num_q_heads = 4u32;
        let num_kv_heads = 2u32;
        let token_count = 8usize;
        let max_seq_len = 64usize;
        let seq_offset = 0usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_data = pseudo_random(42, token_count * num_q_heads as usize * head_dim);
        let mut k_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        let mut v_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        fill_kv_cache(
            &mut k_data,
            &pseudo_random(123, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );
        fill_kv_cache(
            &mut v_data,
            &pseudo_random(456, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );

        let expected = cpu_attention(
            &q_data,
            &k_data,
            &v_data,
            num_q_heads as usize,
            num_kv_heads as usize,
            head_dim,
            max_seq_len,
            seq_offset,
            token_count,
            scale,
        );

        let q_buf = make_fp16_buffer(&device, &q_data);
        let k_buf = make_fp16_buffer(&device, &k_data);
        let v_buf = make_fp16_buffer(&device, &v_data);
        let output_size = token_count * num_q_heads as usize * head_dim;
        let output_fa2 = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("output fa2");
        let output_sdpa = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("output sdpa");

        let pipelines = super::super::ops::MetalPipelines::compile(&device, head_dim, head_dim)
            .expect("compile pipelines");

        // --- Dispatch FA2 ---
        {
            let cmd = queue.command_buffer().expect("cmd");
            let enc = cmd.compute_encoder().expect("enc");
            super::super::ops::encode_fa2_prefill_attention(
                &enc,
                &pipelines.prefill_attention_fa2,
                &super::super::ops::PrefillAttentionParams {
                    q: &q_buf,
                    k_cache: &k_buf,
                    v_cache: &v_buf,
                    output: &output_fa2,
                    num_heads: num_q_heads,
                    num_kv_heads,
                    head_dim: head_dim as u32,
                    max_seq_len: max_seq_len as u32,
                    seq_offset: seq_offset as u32,
                    token_count: token_count as u32,
                    window_size: 0,
                    attn_scale: scale,
                },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // --- Dispatch fused SDPA ---
        {
            let cmd = queue.command_buffer().expect("cmd");
            let enc = cmd.compute_encoder().expect("enc");
            let total_seq = (seq_offset + token_count) as u32;
            super::super::ops::encode_fused_sdpa(
                &enc,
                &pipelines.fused_sdpa,
                &super::super::ops::FusedSdpaParams {
                    q: &q_buf,
                    k: &k_buf,
                    v: &v_buf,
                    output: &output_sdpa,
                    seq_len: total_seq,
                    token_count: token_count as u32,
                    head_dim: head_dim as u32,
                    num_q_heads,
                    num_kv_heads,
                    scale,
                    max_seq_len: max_seq_len as u32,
                },
                None,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let fa2_result = read_fp16_buffer(&output_fa2, output_size);
        let sdpa_result = read_fp16_buffer(&output_sdpa, output_size);

        let mut max_diff_fa2_sdpa = 0.0f32;
        let mut max_diff_fa2_cpu = 0.0f32;
        let mut max_diff_sdpa_cpu = 0.0f32;
        for i in 0..output_size {
            max_diff_fa2_sdpa = max_diff_fa2_sdpa.max((fa2_result[i] - sdpa_result[i]).abs());
            max_diff_fa2_cpu = max_diff_fa2_cpu.max((fa2_result[i] - expected[i]).abs());
            max_diff_sdpa_cpu = max_diff_sdpa_cpu.max((sdpa_result[i] - expected[i]).abs());
        }

        println!("FA2 vs SDPA max diff:  {max_diff_fa2_sdpa:.6}");
        println!("FA2 vs CPU  max diff:  {max_diff_fa2_cpu:.6}");
        println!("SDPA vs CPU max diff:  {max_diff_sdpa_cpu:.6}");

        // FP16 accumulation error: tolerate up to 0.05 for head_dim=128
        assert!(
            max_diff_fa2_sdpa < 0.05,
            "FA2 vs SDPA diverged: {max_diff_fa2_sdpa}"
        );
        assert!(
            max_diff_fa2_cpu < 0.1,
            "FA2 vs CPU diverged: {max_diff_fa2_cpu}"
        );
        assert!(
            max_diff_sdpa_cpu < 0.1,
            "SDPA vs CPU diverged: {max_diff_sdpa_cpu}"
        );
    }

    /// Verify FA2 handles GQA (grouped-query attention) correctly:
    /// multiple Q heads share the same KV head.
    #[test]
    fn fa2_gqa_correctness() {
        let device = match MetalDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP: no Metal device");
                return;
            }
        };
        let queue = device.create_command_queue().expect("command queue");

        let head_dim = 64usize;
        let num_q_heads = 8u32;
        let num_kv_heads = 2u32;
        let token_count = 4usize;
        let max_seq_len = 32usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_data = pseudo_random(77, token_count * num_q_heads as usize * head_dim);
        let mut k_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        let mut v_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        fill_kv_cache(
            &mut k_data,
            &pseudo_random(88, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );
        fill_kv_cache(
            &mut v_data,
            &pseudo_random(99, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );

        let expected = cpu_attention(
            &q_data,
            &k_data,
            &v_data,
            num_q_heads as usize,
            num_kv_heads as usize,
            head_dim,
            max_seq_len,
            0,
            token_count,
            scale,
        );

        let q_buf = make_fp16_buffer(&device, &q_data);
        let k_buf = make_fp16_buffer(&device, &k_data);
        let v_buf = make_fp16_buffer(&device, &v_data);
        let output_size = token_count * num_q_heads as usize * head_dim;
        let output_buf = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("out");

        let pipelines = super::super::ops::MetalPipelines::compile(&device, head_dim, head_dim)
            .expect("compile");

        let cmd = queue.command_buffer().expect("cmd");
        let enc = cmd.compute_encoder().expect("enc");
        super::super::ops::encode_fa2_prefill_attention(
            &enc,
            &pipelines.prefill_attention_fa2,
            &super::super::ops::PrefillAttentionParams {
                q: &q_buf,
                k_cache: &k_buf,
                v_cache: &v_buf,
                output: &output_buf,
                num_heads: num_q_heads,
                num_kv_heads,
                head_dim: head_dim as u32,
                max_seq_len: max_seq_len as u32,
                seq_offset: 0,
                token_count: token_count as u32,
                window_size: 0,
                attn_scale: scale,
            },
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = read_fp16_buffer(&output_buf, output_size);
        let mut max_diff = 0.0f32;
        for i in 0..output_size {
            max_diff = max_diff.max((result[i] - expected[i]).abs());
        }
        println!("FA2 GQA (8:2) vs CPU max diff: {max_diff:.6}");
        assert!(max_diff < 0.1, "FA2 GQA diverged: max_diff={max_diff}");
    }

    /// Verify FA2 with attn_scale=1.0 (QK-normed models like Gemma 4).
    #[test]
    fn fa2_unit_attn_scale() {
        let device = match MetalDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP: no Metal device");
                return;
            }
        };
        let queue = device.create_command_queue().expect("command queue");

        let head_dim = 128usize;
        let num_q_heads = 2u32;
        let num_kv_heads = 2u32;
        let token_count = 4usize;
        let max_seq_len = 16usize;
        let scale = 1.0f32;

        // Use small values so softmax doesn't saturate with scale=1.0
        let q_data: Vec<f32> = pseudo_random(111, token_count * num_q_heads as usize * head_dim)
            .iter()
            .map(|&v| v * 0.1)
            .collect();
        let mut k_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        let mut v_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        let k_fill: Vec<f32> = pseudo_random(222, num_kv_heads as usize * token_count * head_dim)
            .iter()
            .map(|&v| v * 0.1)
            .collect();
        fill_kv_cache(
            &mut k_data,
            &k_fill,
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );
        fill_kv_cache(
            &mut v_data,
            &pseudo_random(333, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );

        let expected = cpu_attention(
            &q_data,
            &k_data,
            &v_data,
            num_q_heads as usize,
            num_kv_heads as usize,
            head_dim,
            max_seq_len,
            0,
            token_count,
            scale,
        );

        let q_buf = make_fp16_buffer(&device, &q_data);
        let k_buf = make_fp16_buffer(&device, &k_data);
        let v_buf = make_fp16_buffer(&device, &v_data);
        let output_size = token_count * num_q_heads as usize * head_dim;
        let output_buf = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("out");

        let pipelines = super::super::ops::MetalPipelines::compile(&device, head_dim, head_dim)
            .expect("compile");

        let cmd = queue.command_buffer().expect("cmd");
        let enc = cmd.compute_encoder().expect("enc");
        super::super::ops::encode_fa2_prefill_attention(
            &enc,
            &pipelines.prefill_attention_fa2,
            &super::super::ops::PrefillAttentionParams {
                q: &q_buf,
                k_cache: &k_buf,
                v_cache: &v_buf,
                output: &output_buf,
                num_heads: num_q_heads,
                num_kv_heads,
                head_dim: head_dim as u32,
                max_seq_len: max_seq_len as u32,
                seq_offset: 0,
                token_count: token_count as u32,
                window_size: 0,
                attn_scale: scale,
            },
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = read_fp16_buffer(&output_buf, output_size);
        let mut max_diff = 0.0f32;
        for i in 0..output_size {
            max_diff = max_diff.max((result[i] - expected[i]).abs());
        }
        println!("FA2 scale=1.0 vs CPU max diff: {max_diff:.6}");
        assert!(
            max_diff < 0.1,
            "FA2 scale=1.0 diverged: max_diff={max_diff}"
        );
    }

    /// Verify v2 register-tiled prefill produces the same output as
    /// fused SDPA and the original FA2 kernel.
    #[test]
    fn v2_matches_fa2_and_sdpa() {
        let device = match MetalDevice::system_default() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP: no Metal device");
                return;
            }
        };
        let queue = device.create_command_queue().expect("command queue");

        let head_dim = 128usize;
        let num_q_heads = 4u32;
        let num_kv_heads = 2u32;
        let token_count = 16usize;
        let max_seq_len = 64usize;
        let seq_offset = 0usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_data = pseudo_random(42, token_count * num_q_heads as usize * head_dim);
        let mut k_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        let mut v_data = vec![0.0f32; num_kv_heads as usize * max_seq_len * head_dim];
        fill_kv_cache(
            &mut k_data,
            &pseudo_random(123, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );
        fill_kv_cache(
            &mut v_data,
            &pseudo_random(456, num_kv_heads as usize * token_count * head_dim),
            num_kv_heads as usize,
            max_seq_len,
            head_dim,
            token_count,
        );

        let expected = cpu_attention(
            &q_data,
            &k_data,
            &v_data,
            num_q_heads as usize,
            num_kv_heads as usize,
            head_dim,
            max_seq_len,
            seq_offset,
            token_count,
            scale,
        );

        let q_buf = make_fp16_buffer(&device, &q_data);
        let k_buf = make_fp16_buffer(&device, &k_data);
        let v_buf = make_fp16_buffer(&device, &v_data);
        let output_size = token_count * num_q_heads as usize * head_dim;

        let pipelines = super::super::ops::MetalPipelines::compile(&device, head_dim, head_dim)
            .expect("compile pipelines");

        // --- Dispatch v2 ---
        let output_v2 = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("out");
        {
            let cmd = queue.command_buffer().expect("cmd");
            let enc = cmd.compute_encoder().expect("enc");
            super::super::ops::encode_v2_prefill_attention(
                &enc,
                &pipelines.prefill_attention_v2,
                &super::super::ops::PrefillAttentionParams {
                    q: &q_buf,
                    k_cache: &k_buf,
                    v_cache: &v_buf,
                    output: &output_v2,
                    num_heads: num_q_heads,
                    num_kv_heads,
                    head_dim: head_dim as u32,
                    max_seq_len: max_seq_len as u32,
                    seq_offset: seq_offset as u32,
                    token_count: token_count as u32,
                    window_size: 0,
                    attn_scale: scale,
                },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // --- Dispatch FA2 (original) ---
        let output_fa2 = device
            .create_buffer(output_size * 2, StorageMode::Shared)
            .expect("out");
        {
            let cmd = queue.command_buffer().expect("cmd");
            let enc = cmd.compute_encoder().expect("enc");
            super::super::ops::encode_fa2_prefill_attention(
                &enc,
                &pipelines.prefill_attention_fa2,
                &super::super::ops::PrefillAttentionParams {
                    q: &q_buf,
                    k_cache: &k_buf,
                    v_cache: &v_buf,
                    output: &output_fa2,
                    num_heads: num_q_heads,
                    num_kv_heads,
                    head_dim: head_dim as u32,
                    max_seq_len: max_seq_len as u32,
                    seq_offset: seq_offset as u32,
                    token_count: token_count as u32,
                    window_size: 0,
                    attn_scale: scale,
                },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let v2_result = read_fp16_buffer(&output_v2, output_size);
        let fa2_result = read_fp16_buffer(&output_fa2, output_size);

        let mut max_v2_fa2 = 0.0f32;
        let mut max_v2_cpu = 0.0f32;
        for i in 0..output_size {
            max_v2_fa2 = max_v2_fa2.max((v2_result[i] - fa2_result[i]).abs());
            max_v2_cpu = max_v2_cpu.max((v2_result[i] - expected[i]).abs());
        }

        println!("V2 vs FA2 max diff:  {max_v2_fa2:.6}");
        println!("V2 vs CPU max diff:  {max_v2_cpu:.6}");

        assert!(max_v2_fa2 < 0.05, "V2 vs FA2 diverged: {max_v2_fa2}");
        assert!(max_v2_cpu < 0.1, "V2 vs CPU diverged: {max_v2_cpu}");
    }
}
