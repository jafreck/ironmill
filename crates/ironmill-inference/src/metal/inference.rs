//! GPU inference engine implementing the [`InferenceEngine`] trait.
//!
//! Runs the full LLaMA-family transformer decode pipeline on Metal:
//!   - MPS `MPSMatrixMultiplication` for linear projections
//!   - Custom Metal compute shaders for RMSNorm, RoPE, SiLU, residual add,
//!     embedding lookup, and attention
//!   - Optional TurboQuant INT8 KV cache compression

use std::any::Any;
use std::time::Instant;

use half::f16;
use ironmill_metal_sys::{
    CommandBufferStatus, MetalBuffer, MetalDevice, MpsMatrix, MpsMatrixMultiply,
    MpsMatrixMultiplyConfig, StorageMode,
};
use mil_rs::weights::{ModelConfig, WeightProvider};

use super::config::MetalConfig;
use super::error::MetalError;
use super::ops;
use super::turboquant::{MetalKvCache, MetalTurboQuantModel, OutlierConfig, TurboQuantMetalConfig};
use super::weights::{LayerWeights, MetalWeights, QuantizedWeight, WeightBuffer};
use crate::calibration::ActivationHook;
use crate::engine::{InferenceEngine, InferenceError};
use crate::types::Logits;

// ── Public artifacts type for load() ────────────────────────────

/// Artifacts passed to [`MetalInference::load`] via the type-erased
/// [`InferenceEngine`] interface.
pub struct MetalArtifacts<'a> {
    pub weights: &'a dyn WeightProvider,
    pub config: MetalConfig,
}

// ── Helper structs ──────────────────────────────────────────────

/// FP16 KV cache (when TurboQuant is disabled).
struct Fp16KvCache {
    /// K caches per layer: `[num_kv_heads × max_seq × head_dim]` FP16.
    k_caches: Vec<MetalBuffer>,
    /// V caches per layer.
    v_caches: Vec<MetalBuffer>,
    seq_pos: usize,
}

impl Fp16KvCache {
    fn new(
        device: &MetalDevice,
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Result<Self, MetalError> {
        let size_bytes = num_kv_heads * max_seq_len * head_dim * 2; // FP16
        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_caches.push(
                device
                    .create_buffer(size_bytes, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
            v_caches.push(
                device
                    .create_buffer(size_bytes, StorageMode::Shared)
                    .map_err(MetalError::Metal)?,
            );
        }
        Ok(Self {
            k_caches,
            v_caches,
            seq_pos: 0,
        })
    }

    fn layer_caches(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        (&self.k_caches[layer], &self.v_caches[layer])
    }

    fn reset(&mut self) {
        self.seq_pos = 0;
    }
}

/// Reusable intermediate activation buffers.
struct IntermediateBuffers {
    hidden_state: MetalBuffer,
    attn_out: MetalBuffer,
    q_proj: MetalBuffer,
    k_proj: MetalBuffer,
    v_proj: MetalBuffer,
    ffn_gate: MetalBuffer,
    ffn_up: MetalBuffer,
    ffn_down: MetalBuffer,
    residual: MetalBuffer,
    norm_out: MetalBuffer,
    logits: MetalBuffer,
    token_ids_buf: MetalBuffer,
}

impl IntermediateBuffers {
    fn allocate(
        device: &MetalDevice,
        max_tokens: usize,
        mc: &ModelConfig,
    ) -> Result<Self, MetalError> {
        let h = mc.hidden_size;
        let nh = mc.num_attention_heads;
        let nkv = mc.num_key_value_heads;
        let hd = mc.head_dim;
        let inter = mc.intermediate_size;
        let vocab = mc.vocab_size;

        let alloc = |size_elems: usize| -> Result<MetalBuffer, MetalError> {
            // FP16 = 2 bytes per element; minimum 16 bytes for Metal
            let bytes = (size_elems * 2).max(16);
            device
                .create_buffer(bytes, StorageMode::Shared)
                .map_err(MetalError::Metal)
        };

        Ok(Self {
            hidden_state: alloc(max_tokens * h)?,
            attn_out: alloc(max_tokens * nh * hd)?,
            q_proj: alloc(max_tokens * nh * hd)?,
            k_proj: alloc(max_tokens * nkv * hd)?,
            v_proj: alloc(max_tokens * nkv * hd)?,
            ffn_gate: alloc(max_tokens * inter)?,
            ffn_up: alloc(max_tokens * inter)?,
            ffn_down: alloc(max_tokens * h)?,
            residual: alloc(max_tokens * h)?,
            norm_out: alloc(max_tokens * h)?,
            logits: alloc(max_tokens * vocab)?,
            token_ids_buf: device
                .create_buffer((max_tokens * 4).max(16), StorageMode::Shared)
                .map_err(MetalError::Metal)?,
        })
    }
}

/// Per-projection matmul state: Dense weights use MPS matrix multiplication
/// while Quantized weights use a custom compute-shader path and carry no MPS
/// object.
enum ProjectionMatmul {
    Dense(MpsMatrixMultiply),
    Quantized,
}

impl ProjectionMatmul {
    /// Returns the MPS matmul for a Dense projection.
    ///
    /// # Panics
    ///
    /// Panics if called on a `Quantized` projection — callers must only invoke
    /// this inside a [`WeightBuffer::Dense`] branch.
    fn dense(&self) -> &MpsMatrixMultiply {
        match self {
            Self::Dense(m) => m,
            Self::Quantized => unreachable!(
                "dense() called on Quantized projection — weight type / matmul mismatch"
            ),
        }
    }
}

/// Cached MPS matmul instances for a given token count.
struct MpsMatmulCache {
    /// Token count these were built for.
    token_count: usize,
    /// Per-layer matmul instances: (q, k, v, o, gate, up, down).
    layer_matmuls: Vec<LayerMatmuls>,
    /// LM head matmul.
    lm_head: MpsMatrixMultiply,
}

struct LayerMatmuls {
    q: ProjectionMatmul,
    k: ProjectionMatmul,
    v: ProjectionMatmul,
    o: ProjectionMatmul,
    gate: ProjectionMatmul,
    up: ProjectionMatmul,
    down: ProjectionMatmul,
}

// ── MetalInference ────────────────────────────────────────────────

/// Metal GPU inference engine.
///
/// Implements the full transformer decode pipeline using Metal compute
/// shaders for element-wise ops and MPS for matrix multiplication.
pub struct MetalInference {
    device: MetalDevice,
    queue: ironmill_metal_sys::CommandQueue,
    pipelines: Option<super::ops::MetalPipelines>,
    weights: Option<MetalWeights>,
    turboquant: Option<MetalTurboQuantModel>,
    kv_cache: Option<MetalKvCache>,
    fp16_kv_cache: Option<Fp16KvCache>,
    intermediate_buffers: Option<IntermediateBuffers>,
    rope_cos: Option<MetalBuffer>,
    rope_sin: Option<MetalBuffer>,
    /// MPS matmul cache for prefill (variable token_count > 1).
    decode_matmuls: Option<MpsMatmulCache>,
    /// MPS matmul cache for single-token decode (token_count=1), preserved
    /// across prefill→decode transitions so it never needs rebuilding.
    decode_matmuls_t1: Option<MpsMatmulCache>,
    config: MetalConfig,
    model_config: Option<ModelConfig>,
    /// Pre-allocated buffer for FP16 logits readback.
    logits_fp16_buf: Vec<u8>,
    /// Pre-allocated buffer for serializing token IDs to GPU.
    token_bytes_buf: Vec<u8>,
    seq_pos: usize,
}

impl MetalInference {
    /// Access compiled pipelines — panics if `load()` hasn't been called yet.
    fn pipelines(&self) -> &super::ops::MetalPipelines {
        self.pipelines
            .as_ref()
            .expect("pipelines not compiled — call load() first")
    }

    /// Create a new GPU inference engine (device + queue + shader pipelines).
    pub fn new(config: MetalConfig) -> Result<Self, MetalError> {
        config.validate().map_err(|e| MetalError::Config(e))?;
        let device = MetalDevice::system_default().map_err(MetalError::Metal)?;
        let queue = device.create_command_queue().map_err(MetalError::Metal)?;
        // Pipelines are compiled in load() once head_dim is known.
        Ok(Self {
            device,
            queue,
            pipelines: None,
            weights: None,
            turboquant: None,
            kv_cache: None,
            fp16_kv_cache: None,
            intermediate_buffers: None,
            rope_cos: None,
            rope_sin: None,
            decode_matmuls: None,
            decode_matmuls_t1: None,
            config,
            model_config: None,
            logits_fp16_buf: Vec::new(),
            token_bytes_buf: Vec::new(),
            seq_pos: 0,
        })
    }

    /// Load model weights directly from a [`WeightProvider`], bypassing
    /// the type-erased [`InferenceEngine::load`] interface.
    pub fn load_weights(
        &mut self,
        provider: &dyn mil_rs::weights::WeightProvider,
        config: MetalConfig,
    ) -> Result<(), InferenceError> {
        self.config = config;

        let weights = MetalWeights::load(&self.device, provider, self.config.force_cpu_dequant)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        let mc = &weights.config;
        self.model_config = Some(mc.clone());

        // Pre-allocate logits readback buffer (vocab × 2 bytes for FP16).
        self.logits_fp16_buf.resize(mc.vocab_size * 2, 0);

        // Compile Metal shader pipelines with the model's head_dim.
        let pipelines = super::ops::MetalPipelines::compile(&self.device, mc.head_dim)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        self.pipelines = Some(pipelines);

        let max_prefill = self.config.prefill_chunk_size.unwrap_or(512).max(1);
        let bufs = IntermediateBuffers::allocate(&self.device, max_prefill, mc)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        self.intermediate_buffers = Some(bufs);

        let (cos, sin) = Self::build_rope_cache(
            &self.device,
            mc.head_dim,
            self.config.max_seq_len,
            mc.rope_theta,
        )
        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        self.rope_cos = Some(cos);
        self.rope_sin = Some(sin);

        let decode_cache_t1 = Self::build_matmul_cache(&self.device, mc, &weights, 1)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        self.decode_matmuls_t1 = Some(decode_cache_t1);
        self.decode_matmuls = None;

        if self.config.enable_turboquant {
            // Outlier channel strategy (§4.3) is disabled for now.
            // The mixed-precision split (4-bit outlier / 3-bit non-outlier)
            // combined with QJL sign bit-stealing produces worse quality
            // than uniform 4-bit codebook quantization for both K and V.
            let outlier_cfg: Option<OutlierConfig> = None;

            let tq_config = TurboQuantMetalConfig {
                n_bits: self.config.n_bits,
                num_kv_heads: mc.num_key_value_heads,
                head_dim: mc.head_dim,
                max_seq_len: self.config.max_seq_len,
                num_layers: mc.num_hidden_layers,
                rotation_seed: self.config.rotation_seed,
                outlier: outlier_cfg,
            };
            let tq_model = MetalTurboQuantModel::new(&self.device, tq_config.clone())
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let kv_cache = MetalKvCache::new(&self.device, &tq_config)
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            self.turboquant = Some(tq_model);
            self.kv_cache = Some(kv_cache);
            self.fp16_kv_cache = None;
        } else {
            let fp16_kv = Fp16KvCache::new(
                &self.device,
                mc.num_hidden_layers,
                mc.num_key_value_heads,
                self.config.max_seq_len,
                mc.head_dim,
            )
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            self.fp16_kv_cache = Some(fp16_kv);
            self.turboquant = None;
            self.kv_cache = None;
        }

        self.weights = Some(weights);
        self.seq_pos = 0;
        Ok(())
    }

    // ── Memory query ─────────────────────────────────────────────

    /// Returns the current Metal device allocation size in bytes.
    pub fn gpu_allocated_bytes(&self) -> usize {
        self.device.current_allocated_size()
    }

    // ── RoPE cache ──────────────────────────────────────────────

    fn build_rope_cache(
        device: &MetalDevice,
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
    ) -> Result<(MetalBuffer, MetalBuffer), MetalError> {
        let half_dim = head_dim / 2;
        let mut cos_data = vec![0u8; max_seq_len * half_dim * 2];
        let mut sin_data = vec![0u8; max_seq_len * half_dim * 2];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                let c = f16::from_f64(angle.cos());
                let s = f16::from_f64(angle.sin());
                let offset = (pos * half_dim + i) * 2;
                cos_data[offset..offset + 2].copy_from_slice(&c.to_le_bytes());
                sin_data[offset..offset + 2].copy_from_slice(&s.to_le_bytes());
            }
        }

        let cos_buf = device
            .create_buffer_with_data(&cos_data, StorageMode::Shared)
            .map_err(MetalError::Metal)?;
        let sin_buf = device
            .create_buffer_with_data(&sin_data, StorageMode::Shared)
            .map_err(MetalError::Metal)?;
        Ok((cos_buf, sin_buf))
    }

    // ── MPS matmul cache ────────────────────────────────────────

    fn build_matmul_cache(
        device: &MetalDevice,
        mc: &ModelConfig,
        weights: &MetalWeights,
        token_count: usize,
    ) -> Result<MpsMatmulCache, MetalError> {
        let h = mc.hidden_size;
        let nh = mc.num_attention_heads;
        let nkv = mc.num_key_value_heads;
        let hd = mc.head_dim;
        let inter = mc.intermediate_size;
        let vocab = mc.vocab_size;

        // Weight layout: [out_features, in_features] stored row-major FP16.
        // We compute: output = input × weight^T
        // MPS: result = left × right
        //   left  = input  [token_count × in_features]
        //   right = weight [out_features × in_features], transpose_right=true
        //   result = [token_count × out_features]
        let make_matmul =
            |rows: usize, cols: usize, inner: usize| -> Result<MpsMatrixMultiply, MetalError> {
                MpsMatrixMultiply::new(
                    device,
                    &MpsMatrixMultiplyConfig {
                        transpose_left: false,
                        transpose_right: true,   // weights are [out, in]
                        result_rows: rows,       // token_count
                        result_columns: cols,    // out_features
                        interior_columns: inner, // in_features
                        alpha: 1.0,
                        beta: 0.0,
                    },
                )
                .map_err(MetalError::Metal)
            };

        // Only create MPS matmul instances for Dense weights; Quantized
        // projections use the custom compute kernel path instead.
        let projection_matmul = |wb: &WeightBuffer,
                                 rows: usize,
                                 cols: usize,
                                 inner: usize|
         -> Result<ProjectionMatmul, MetalError> {
            match wb {
                WeightBuffer::Dense { .. } => {
                    Ok(ProjectionMatmul::Dense(make_matmul(rows, cols, inner)?))
                }
                WeightBuffer::Quantized(_) => Ok(ProjectionMatmul::Quantized),
                // AffineQuantized gets a Dense matmul — the dequant kernel
                // produces FP16 weights consumed by MPS.
                WeightBuffer::AffineQuantized(_) => {
                    Ok(ProjectionMatmul::Dense(make_matmul(rows, cols, inner)?))
                }
            }
        };

        let mut layer_matmuls = Vec::with_capacity(mc.num_hidden_layers);
        for i in 0..mc.num_hidden_layers {
            let lw = &weights.layers[i];
            layer_matmuls.push(LayerMatmuls {
                q: projection_matmul(&lw.q_proj, token_count, nh * hd, h)?,
                k: projection_matmul(&lw.k_proj, token_count, nkv * hd, h)?,
                v: projection_matmul(&lw.v_proj, token_count, nkv * hd, h)?,
                o: projection_matmul(&lw.o_proj, token_count, h, nh * hd)?,
                gate: projection_matmul(&lw.gate_proj, token_count, inter, h)?,
                up: projection_matmul(&lw.up_proj, token_count, inter, h)?,
                down: projection_matmul(&lw.down_proj, token_count, h, inter)?,
            });
        }

        let lm_head = make_matmul(token_count, vocab, h)?;

        Ok(MpsMatmulCache {
            token_count,
            layer_matmuls,
            lm_head,
        })
    }

    // ── Core decode pipeline ────────────────────────────────────

    /// Run the transformer decode pipeline for `token_count` tokens.
    /// Returns logits for the last token position.
    fn run_pipeline(&mut self, token_ids: &[u32]) -> Result<Logits, InferenceError> {
        let weights = self.weights.as_ref().ok_or(InferenceError::NotLoaded)?;
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;
        let bufs = self
            .intermediate_buffers
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;
        let rope_cos = self.rope_cos.as_ref().ok_or(InferenceError::NotLoaded)?;
        let rope_sin = self.rope_sin.as_ref().ok_or(InferenceError::NotLoaded)?;

        let token_count = token_ids.len();
        let seq_pos = self.seq_pos;
        let h = mc.hidden_size;
        let nh = mc.num_attention_heads as u32;
        let nkv = mc.num_kv_heads() as u32;
        let hd = mc.head_dim as u32;
        let inter = mc.intermediate_size;
        let vocab = mc.vocab_size;
        let eps = mc.rms_norm_eps as f32;
        let enable_tq = self.config.enable_turboquant && self.turboquant.is_some();

        // Use the pre-built single-token matmul cache for decode steps;
        // only rebuild the general cache for prefill (token_count > 1).
        let matmuls = if token_count == 1 {
            self.decode_matmuls_t1
                .as_ref()
                .ok_or(InferenceError::NotLoaded)?
        } else {
            let need_rebuild = self
                .decode_matmuls
                .as_ref()
                .is_none_or(|c| c.token_count != token_count);
            if need_rebuild {
                let cache = Self::build_matmul_cache(&self.device, mc, weights, token_count)
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                self.decode_matmuls = Some(cache);
            }
            self.decode_matmuls
                .as_ref()
                .expect("decode_matmuls must be populated")
        };

        // Write token IDs to GPU buffer (reuse persistent buffer).
        self.token_bytes_buf.clear();
        self.token_bytes_buf
            .extend(token_ids.iter().flat_map(|t| t.to_le_bytes()));
        bufs.token_ids_buf
            .write_bytes(&self.token_bytes_buf, 0)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        // Create command buffer.
        let cmd_buf = self
            .queue
            .command_buffer()
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        // Step 0: Embedding lookup via compute encoder.
        {
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;

            ops::encode_embedding_lookup(
                &enc,
                &self.pipelines().embedding_lookup,
                &ops::EmbeddingLookupParams {
                    token_ids: &bufs.token_ids_buf,
                    embedding_table: &weights.embedding,
                    output: &bufs.hidden_state,
                    hidden_size: h as u32,
                    token_count: token_count as u32,
                    vocab_size: vocab as u32,
                },
            );
            enc.end_encoding();
        }

        // Per-layer processing.
        //
        // Operator fusion: the fused_residual_rms_norm kernel combines a
        // residual add with RMSNorm in a single dispatch, cutting per-layer
        // kernel launches by two.  To enable the second fusion (end-of-layer
        // residual + next layer's input norm), we hoist the very first layer's
        // input norm out of the loop.  Subsequent layers receive their input
        // norm as part of the previous layer's fused end-of-layer dispatch.

        // Input norm for the first layer (standalone — no preceding residual to fuse).
        {
            let lw0 = &weights.layers[0];
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            ops::encode_rms_norm(
                &enc,
                &self.pipelines().rms_norm,
                &ops::RmsNormParams {
                    input: &bufs.hidden_state,
                    weight: &lw0.input_norm,
                    output: &bufs.norm_out,
                    hidden_size: h as u32,
                    token_count: token_count as u32,
                    eps,
                },
            );
            enc.end_encoding();
        }

        for layer_idx in 0..mc.num_hidden_layers {
            let lw = &weights.layers[layer_idx];
            let lm = &matmuls.layer_matmuls[layer_idx];
            let pipelines = self.pipelines();

            // norm_out already contains the input-norm result:
            //   • layer 0: computed by the standalone dispatch above
            //   • layer 1+: produced by the previous layer's fused end-of-layer kernel

            // Steps 3-5: Q/K/V projections — dispatch by weight type.
            let row_bytes_h = h * 2; // FP16
            let row_bytes_qo = (mc.num_attention_heads * mc.head_dim) * 2;
            let row_bytes_kv = (mc.num_kv_heads() * mc.head_dim) * 2;

            let norm_mat = MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;

            // Q / K / V projections
            let qkv_out_features = mc.num_attention_heads * mc.head_dim;
            let kv_out_features = mc.num_kv_heads() * mc.head_dim;
            for (weight, output_buf, matmul, out_features, row_bytes_out) in [
                (
                    &lw.q_proj,
                    &bufs.q_proj,
                    &lm.q,
                    qkv_out_features,
                    row_bytes_qo,
                ),
                (
                    &lw.k_proj,
                    &bufs.k_proj,
                    &lm.k,
                    kv_out_features,
                    row_bytes_kv,
                ),
                (
                    &lw.v_proj,
                    &bufs.v_proj,
                    &lm.v,
                    kv_out_features,
                    row_bytes_kv,
                ),
            ] {
                encode_projection(
                    &cmd_buf,
                    &bufs.norm_out,
                    &norm_mat,
                    weight,
                    output_buf,
                    matmul,
                    pipelines,
                    token_count,
                    out_features,
                    h,
                    row_bytes_h,
                    row_bytes_out,
                )?;
            }

            // Step 6: QK normalization (Qwen3) + RoPE
            encode_qk_norm_and_rope(
                &cmd_buf,
                pipelines,
                bufs,
                lw.q_norm.as_ref(),
                lw.k_norm.as_ref(),
                rope_cos,
                rope_sin,
                nh,
                nkv,
                hd,
                seq_pos,
                token_count,
                eps,
            )?;

            // Steps 7-8: KV cache write + attention
            encode_kv_cache_and_attention(
                &cmd_buf,
                pipelines,
                bufs,
                self.turboquant.as_ref(),
                self.kv_cache.as_ref(),
                self.fp16_kv_cache.as_ref(),
                self.config.max_seq_len,
                self.config.n_bits,
                layer_idx,
                seq_pos,
                token_count,
                nh,
                nkv,
                hd,
                enable_tq,
            )?;

            // Step 9: Output projection
            let attn_mat = MpsMatrix::from_buffer(
                &bufs.attn_out,
                token_count,
                mc.num_attention_heads * mc.head_dim,
                row_bytes_qo,
            )
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            encode_projection(
                &cmd_buf,
                &bufs.attn_out,
                &attn_mat,
                &lw.o_proj,
                &bufs.ffn_down,
                &lm.o,
                pipelines,
                token_count,
                h,
                mc.num_attention_heads * mc.head_dim,
                row_bytes_qo,
                row_bytes_h,
            )?;

            // Step 10-11 (fused): Residual add + post-attention RMSNorm
            {
                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                ops::encode_fused_residual_rms_norm(
                    &enc,
                    &pipelines.fused_residual_rms_norm,
                    &ops::FusedResidualRmsNormParams {
                        a: &bufs.hidden_state,           // a: skip connection (pre-norm hidden)
                        b: &bufs.ffn_down,               // b: o_proj output
                        weight: &lw.post_attn_norm,      // weight: post-attention norm
                        normed_output: &bufs.norm_out,   // normed output → MLP input
                        residual_output: &bufs.residual, // residual output → next skip connection
                        eps,
                        hidden_size: h as u32,
                        token_count: token_count as u32,
                    },
                );
                enc.end_encoding();
            }

            // Steps 12-15: FFN block (gate + up + SiLU + down)
            encode_ffn_block(&cmd_buf, pipelines, bufs, lw, lm, h, inter, token_count)?;

            // Step 16: Residual add + next layer's input norm (or standalone for last layer)
            let next_norm = if layer_idx + 1 < mc.num_hidden_layers {
                Some(&weights.layers[layer_idx + 1].input_norm)
            } else {
                None
            };
            encode_end_of_layer_residual(
                &cmd_buf,
                pipelines,
                bufs,
                next_norm,
                h,
                token_count,
                eps,
            )?;
        }

        // Step 17: Final RMSNorm
        {
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            ops::encode_rms_norm(
                &enc,
                &self.pipelines().rms_norm,
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

        // Step 18: LM head matmul
        let row_bytes_h = h * 2;
        let row_bytes_vocab = vocab * 2;
        if token_count == 1 {
            if let Some(ref packed_buf) = weights.lm_head_packed {
                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                ops::encode_matvec(
                    &enc,
                    &self.pipelines().matvec,
                    &bufs.norm_out,
                    packed_buf,
                    &bufs.logits,
                    vocab as u32,
                    h as u32,
                );
                enc.end_encoding();
            } else {
                let final_norm_mat =
                    MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                let lm_head_weight_mat =
                    MpsMatrix::from_buffer(&weights.lm_head, vocab, h, row_bytes_h)
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                let logits_mat =
                    MpsMatrix::from_buffer(&bufs.logits, token_count, vocab, row_bytes_vocab)
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                matmuls
                    .lm_head
                    .encode(&cmd_buf, &final_norm_mat, &lm_head_weight_mat, &logits_mat);
            }
        } else {
            let final_norm_mat =
                MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let lm_head_weight_mat =
                MpsMatrix::from_buffer(&weights.lm_head, vocab, h, row_bytes_h)
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let logits_mat =
                MpsMatrix::from_buffer(&bufs.logits, token_count, vocab, row_bytes_vocab)
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            matmuls
                .lm_head
                .encode(&cmd_buf, &final_norm_mat, &lm_head_weight_mat, &logits_mat);
        }

        // Step 19: Commit and wait.
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        if cmd_buf.status() == CommandBufferStatus::Error {
            return Err(InferenceError::Decode(
                "Metal command buffer execution failed".into(),
            ));
        }

        // Step 20: Read logits for the last token position → Vec<f32>.
        let last_token_offset = (token_count - 1) * vocab * 2; // FP16 offset in bytes
        let logits_byte_count = vocab * 2;
        self.logits_fp16_buf.resize(logits_byte_count, 0);
        bufs.logits
            .read_bytes(&mut self.logits_fp16_buf, last_token_offset)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        let logits: Vec<f32> = self
            .logits_fp16_buf
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f16::from_bits(bits).to_f32()
            })
            .collect();

        // Advance sequence position.
        self.seq_pos += token_count;
        if enable_tq {
            if let Some(kv) = self.kv_cache.as_mut() {
                kv.advance_by(token_count)?;
            }
        } else if let Some(fp16_kv) = self.fp16_kv_cache.as_mut() {
            fp16_kv.seq_pos += token_count;
        }

        Ok(logits)
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
        let bufs = self
            .intermediate_buffers
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;
        let rope_cos = self.rope_cos.as_ref().ok_or(InferenceError::NotLoaded)?;
        let rope_sin = self.rope_sin.as_ref().ok_or(InferenceError::NotLoaded)?;

        let token_count = token_ids.len();
        let seq_pos = self.seq_pos;
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
            let cache = Self::build_matmul_cache(&self.device, &mc, weights, token_count)
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            self.decode_matmuls = Some(cache);
        }
        let matmuls = self
            .decode_matmuls
            .as_ref()
            .expect("decode_matmuls must be populated");

        // Write token IDs to GPU buffer.
        let token_bytes: Vec<u8> = token_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        bufs.token_ids_buf
            .write_bytes(&token_bytes, 0)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        // Reusable readback buffer for norm_out: token_count × hidden_size × 2 bytes (FP16).
        let norm_readback_bytes = token_count * h * 2;

        // Timing accumulators.
        let mut gpu_time_ms = 0.0f64;
        let mut readback_time_ms = 0.0f64;

        // ── Phase 0: Embedding lookup ───────────────────────────
        let cmd_buf = self
            .queue
            .command_buffer()
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        {
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            ops::encode_embedding_lookup(
                &enc,
                &self.pipelines().embedding_lookup,
                &ops::EmbeddingLookupParams {
                    token_ids: &bufs.token_ids_buf,
                    embedding_table: &weights.embedding,
                    output: &bufs.hidden_state,
                    hidden_size: h as u32,
                    token_count: token_count as u32,
                    vocab_size: vocab as u32,
                },
            );
            enc.end_encoding();
        }

        // Input norm for the first layer (standalone).
        {
            let lw0 = &weights.layers[0];
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            ops::encode_rms_norm(
                &enc,
                &self.pipelines().rms_norm,
                &ops::RmsNormParams {
                    input: &bufs.hidden_state,
                    weight: &lw0.input_norm,
                    output: &bufs.norm_out,
                    hidden_size: h as u32,
                    token_count: token_count as u32,
                    eps,
                },
            );
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
            let lm = &matmuls.layer_matmuls[layer_idx];

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
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                readback_time_ms += rb_start.elapsed().as_secs_f64() * 1000.0;
                layer_callback(layer_idx, "attn_norm", readback);
            }

            // Create new command buffer for this layer's attention block.
            let cmd_buf = self
                .queue
                .command_buffer()
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let pipelines = self.pipelines();

            // Steps 3-5: Q/K/V projections
            let row_bytes_h = h * 2;
            let row_bytes_qo = (mc.num_attention_heads * mc.head_dim) * 2;
            let row_bytes_kv = (mc.num_kv_heads() * mc.head_dim) * 2;

            let norm_mat = MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;

            let qkv_out_features = mc.num_attention_heads * mc.head_dim;
            let kv_out_features = mc.num_kv_heads() * mc.head_dim;
            for (weight, output_buf, matmul, out_features, row_bytes_out) in [
                (
                    &lw.q_proj,
                    &bufs.q_proj,
                    &lm.q,
                    qkv_out_features,
                    row_bytes_qo,
                ),
                (
                    &lw.k_proj,
                    &bufs.k_proj,
                    &lm.k,
                    kv_out_features,
                    row_bytes_kv,
                ),
                (
                    &lw.v_proj,
                    &bufs.v_proj,
                    &lm.v,
                    kv_out_features,
                    row_bytes_kv,
                ),
            ] {
                encode_projection(
                    &cmd_buf,
                    &bufs.norm_out,
                    &norm_mat,
                    weight,
                    output_buf,
                    matmul,
                    pipelines,
                    token_count,
                    out_features,
                    h,
                    row_bytes_h,
                    row_bytes_out,
                )?;
            }

            // Step 6: QK normalization (Qwen3) + RoPE
            encode_qk_norm_and_rope(
                &cmd_buf,
                pipelines,
                bufs,
                lw.q_norm.as_ref(),
                lw.k_norm.as_ref(),
                rope_cos,
                rope_sin,
                nh,
                nkv,
                hd,
                seq_pos,
                token_count,
                eps,
            )?;

            // Steps 7-8: KV cache write + attention
            encode_kv_cache_and_attention(
                &cmd_buf,
                pipelines,
                bufs,
                self.turboquant.as_ref(),
                self.kv_cache.as_ref(),
                self.fp16_kv_cache.as_ref(),
                self.config.max_seq_len,
                self.config.n_bits,
                layer_idx,
                seq_pos,
                token_count,
                nh,
                nkv,
                hd,
                enable_tq,
            )?;

            // Step 9: Output projection
            let attn_mat = MpsMatrix::from_buffer(
                &bufs.attn_out,
                token_count,
                mc.num_attention_heads * mc.head_dim,
                row_bytes_qo,
            )
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            encode_projection(
                &cmd_buf,
                &bufs.attn_out,
                &attn_mat,
                &lw.o_proj,
                &bufs.ffn_down,
                &lm.o,
                pipelines,
                token_count,
                h,
                mc.num_attention_heads * mc.head_dim,
                row_bytes_qo,
                row_bytes_h,
            )?;

            // Step 10-11 (fused): Residual add + post-attention RMSNorm
            {
                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
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
                enc.end_encoding();
            }

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
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                readback_time_ms += rb_start.elapsed().as_secs_f64() * 1000.0;
                layer_callback(layer_idx, "ffn_norm", readback);
            }

            // ── New command buffer for FFN block ────────────────
            let cmd_buf = self
                .queue
                .command_buffer()
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let pipelines = self.pipelines();

            // Steps 12-15: FFN block (gate + up + SiLU + down)
            encode_ffn_block(&cmd_buf, pipelines, bufs, lw, lm, h, inter, token_count)?;

            // Step 16: Residual add + next layer's input norm (or standalone for last layer)
            let next_norm = if layer_idx + 1 < mc.num_hidden_layers {
                Some(&weights.layers[layer_idx + 1].input_norm)
            } else {
                None
            };
            encode_end_of_layer_residual(
                &cmd_buf,
                pipelines,
                bufs,
                next_norm,
                h,
                token_count,
                eps,
            )?;

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
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        // Step 17: Final RMSNorm
        {
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            ops::encode_rms_norm(
                &enc,
                &self.pipelines().rms_norm,
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

        // Step 18: LM head matmul
        let row_bytes_h = h * 2;
        let row_bytes_vocab = vocab * 2;
        if token_count == 1 {
            if let Some(ref packed_buf) = weights.lm_head_packed {
                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                ops::encode_matvec(
                    &enc,
                    &self.pipelines().matvec,
                    &bufs.norm_out,
                    packed_buf,
                    &bufs.logits,
                    vocab as u32,
                    h as u32,
                );
                enc.end_encoding();
            } else {
                let final_norm_mat =
                    MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                let lm_head_weight_mat =
                    MpsMatrix::from_buffer(&weights.lm_head, vocab, h, row_bytes_h)
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                let logits_mat =
                    MpsMatrix::from_buffer(&bufs.logits, token_count, vocab, row_bytes_vocab)
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                matmuls
                    .lm_head
                    .encode(&cmd_buf, &final_norm_mat, &lm_head_weight_mat, &logits_mat);
            }
        } else {
            let final_norm_mat =
                MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let lm_head_weight_mat =
                MpsMatrix::from_buffer(&weights.lm_head, vocab, h, row_bytes_h)
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let logits_mat =
                MpsMatrix::from_buffer(&bufs.logits, token_count, vocab, row_bytes_vocab)
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            matmuls
                .lm_head
                .encode(&cmd_buf, &final_norm_mat, &lm_head_weight_mat, &logits_mat);
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
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

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

// ── Byte → f16 conversion helper ───────────────────────────────

/// Reinterpret a raw byte slice as `&[f16]`.
///
/// # Panics
///
/// Panics if `bytes.len()` is not a multiple of 2 or if the pointer is not
/// 2-byte aligned. Both conditions are guaranteed for Metal buffer readbacks.
fn bytes_as_f16(bytes: &[u8]) -> &[f16] {
    let elem_size = std::mem::size_of::<f16>();
    assert!(
        bytes.len() % elem_size == 0,
        "byte slice length {} is not a multiple of f16 size ({})",
        bytes.len(),
        elem_size,
    );
    assert!(
        bytes.as_ptr() as usize % std::mem::align_of::<f16>() == 0,
        "byte slice pointer is not aligned for f16",
    );
    // SAFETY: We have verified length and alignment. `f16` is `#[repr(transparent)]`
    // over `u16` and has no invalid bit patterns.
    #[allow(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f16, bytes.len() / elem_size)
    }
}

// ── Helpers on ModelConfig ──────────────────────────────────────

trait ModelConfigExt {
    fn num_kv_heads(&self) -> usize;
}

impl ModelConfigExt for ModelConfig {
    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
    }
}

// ── Projection dispatch helper ──────────────────────────────────

/// Encode a single Q/K/V-style linear projection.
///
/// Handles all weight representations:
/// - **Dense + packed (token_count == 1):** custom blocked matvec kernel.
/// - **Dense (general):** MPS matrix multiplication.
/// - **Quantized:** PolarQuant compute kernel.
fn encode_projection(
    cmd_buf: &ironmill_metal_sys::CommandBuffer,
    input_buf: &MetalBuffer,
    input_mat: &MpsMatrix,
    weight: &WeightBuffer,
    output_buf: &MetalBuffer,
    matmul: &ProjectionMatmul,
    pipelines: &super::ops::MetalPipelines,
    token_count: usize,
    out_features: usize,
    in_features: usize,
    row_bytes_in: usize,
    row_bytes_out: usize,
) -> Result<(), InferenceError> {
    match weight {
        WeightBuffer::Dense { buf, packed } => {
            if token_count == 1 {
                if let Some(packed_buf) = packed {
                    let enc = cmd_buf
                        .compute_encoder()
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    ops::encode_matvec(
                        &enc,
                        &pipelines.matvec,
                        input_buf,
                        packed_buf,
                        output_buf,
                        out_features as u32,
                        in_features as u32,
                    );
                    enc.end_encoding();
                    return Ok(());
                }
            }
            // MPS matmul path: multi-token, or single-token without packed buf.
            let weight_mat = MpsMatrix::from_buffer(buf, out_features, in_features, row_bytes_in)
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let result_mat =
                MpsMatrix::from_buffer(output_buf, token_count, out_features, row_bytes_out)
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            matmul
                .dense()
                .encode(cmd_buf, input_mat, &weight_mat, &result_mat);
        }
        WeightBuffer::Quantized(q) => {
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            encode_polarquant_matmul(&enc, input_buf, q, output_buf, pipelines, token_count)?;
            enc.end_encoding();
        }
        WeightBuffer::AffineQuantized(aq) => {
            // Step 1: Dequantize INT4 → FP16 via compute kernel.
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            ops::encode_int4_dequantize(&enc, &pipelines.int4_dequantize, aq);
            enc.end_encoding();

            // Step 2: MPS matmul using the dequantized FP16 weight buffer.
            let (n, k) = aq.shape;
            let weight_mat = MpsMatrix::from_buffer(&aq.dequant_buf, n, k, k * 2)
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let result_mat = MpsMatrix::from_buffer(output_buf, token_count, n, row_bytes_out)
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            matmul
                .dense()
                .encode(cmd_buf, input_mat, &weight_mat, &result_mat);
        }
    }
    Ok(())
}

// ── PolarQuant kernel dispatch ──────────────────────────────────

/// Encode a PolarQuant quantized matmul or matvec via compute kernel.
///
/// Dispatches to the correct Metal kernel based on `n_bits` and token count:
/// - m=1: matvec (one threadgroup per output row)
/// - m>1: tiled matmul
fn encode_polarquant_matmul(
    encoder: &ironmill_metal_sys::ComputeEncoder,
    input: &MetalBuffer,
    weight: &QuantizedWeight,
    output: &MetalBuffer,
    pipelines: &super::ops::MetalPipelines,
    m: usize,
) -> Result<(), InferenceError> {
    let (n, k) = weight.shape; // (out_features, in_features)

    let pipeline = match (weight.n_bits, m) {
        (4, 1) => &pipelines.polarquant_matvec_int4,
        (4, _) => &pipelines.polarquant_matmul_int4,
        (8, 1) => &pipelines.polarquant_matvec_int8,
        (8, _) => &pipelines.polarquant_matmul_int8,
        _ => {
            return Err(InferenceError::Runtime(format!(
                "unsupported n_bits: {}",
                weight.n_bits
            )));
        }
    };

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&weight.indices, 0, 1);
    encoder.set_buffer(&weight.lut, 0, 2);
    encoder.set_buffer(&weight.norms, 0, 3);
    encoder.set_buffer(output, 0, 4);

    if m == 1 {
        // matvec: one threadgroup per output row
        encoder.set_bytes(&(n as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 6);
        let threads_per_group = 32; // SIMD width
        encoder.dispatch_threadgroups((n, 1, 1), (threads_per_group, 1, 1));
    } else {
        // matmul: tiled
        encoder.set_bytes(&(m as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(n as u32).to_le_bytes(), 6);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 7);
        let tile_m = 8;
        let tile_n = 32;
        encoder.dispatch_threadgroups(
            ((n + tile_n - 1) / tile_n, (m + tile_m - 1) / tile_m, 1),
            (tile_n, tile_m, 1),
        );
    }
    Ok(())
}

// ── Shared decode helpers ──────────────────────────────────────

/// Encode QK normalization (Qwen3) and RoPE for Q and K projections.
fn encode_qk_norm_and_rope(
    cmd_buf: &ironmill_metal_sys::CommandBuffer,
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
        let enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        ops::encode_rms_norm(
            &enc,
            &pipelines.rms_norm,
            &ops::RmsNormParams {
                input: &bufs.q_proj,
                weight: q_norm_w,
                output: &bufs.q_proj,
                hidden_size: hd,
                token_count: token_count as u32 * nh,
                eps,
            },
        );
        ops::encode_rms_norm(
            &enc,
            &pipelines.rms_norm,
            &ops::RmsNormParams {
                input: &bufs.k_proj,
                weight: k_norm_w,
                output: &bufs.k_proj,
                hidden_size: hd,
                token_count: token_count as u32 * nkv,
                eps,
            },
        );
        enc.end_encoding();
    }

    {
        let enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        ops::encode_rope(
            &enc,
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
            &enc,
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
        enc.end_encoding();
    }

    Ok(())
}

/// Encode KV cache write and attention dispatch.
///
/// Handles TurboQuant (outlier and standard) and FP16 KV cache paths.
fn encode_kv_cache_and_attention(
    cmd_buf: &ironmill_metal_sys::CommandBuffer,
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
) -> Result<(), InferenceError> {
    let max_seq = max_seq_len as u32;
    let n_bits = n_bits as u32;

    if enable_tq {
        let tq = turboquant.expect("turboquant must be initialized when enable_tq is true");
        let kv = kv_cache.expect("kv_cache must be initialized when enable_tq is true");

        let enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        if let Some(ref outlier) = tq.outlier {
            // ── Outlier channel strategy dispatch ──
            let ((k_o_cache, v_o_cache), (k_n_cache, v_n_cache)) =
                kv.layer_outlier_caches(layer_idx);
            let ((k_o_scale, v_o_scale), (k_n_scale, v_n_scale)) =
                kv.layer_outlier_scales(layer_idx);
            let (k_o_r_norms, k_n_r_norms) = kv.layer_outlier_r_norms(layer_idx);
            let kv_head_stride_bytes = (nkv as usize) * (hd as usize) * 2;
            let tg_size = std::cmp::max(
                outlier.d_outlier_padded as usize,
                outlier.d_non_padded as usize,
            );

            for t in 0..token_count {
                let token_offset = t * kv_head_stride_bytes;
                // K cache: (b-1)-bit codebook + QJL
                enc.set_pipeline(&pipelines.turboquant_outlier_cache_write);
                enc.set_buffer(&bufs.k_proj, token_offset, 0);
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
                enc.set_bytes(&((seq_pos + t) as u32).to_le_bytes(), 15);
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
                enc.dispatch_threadgroups((nkv as usize, 1, 1), (tg_size.min(1024), 1, 1));
                // V cache: b-bit codebook, no QJL
                enc.set_pipeline(&pipelines.turboquant_outlier_cache_write);
                enc.set_buffer(&bufs.v_proj, token_offset, 0);
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
                enc.set_bytes(&((seq_pos + t) as u32).to_le_bytes(), 15);
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
                enc.dispatch_threadgroups((nkv as usize, 1, 1), (tg_size.min(1024), 1, 1));
            }

            let q_head_stride_bytes = (nh as usize) * (hd as usize) * 2;
            for t in 0..token_count {
                let q_offset = t * q_head_stride_bytes;
                let attn_out_offset = t * q_head_stride_bytes;
                let current_seq_len = (seq_pos + t + 1) as u32;
                enc.set_pipeline(&pipelines.turboquant_outlier_attention);
                enc.set_buffer(&bufs.q_proj, q_offset, 0);
                enc.set_buffer(k_o_cache, 0, 1);
                enc.set_buffer(v_o_cache, 0, 2);
                enc.set_buffer(k_n_cache, 0, 3);
                enc.set_buffer(v_n_cache, 0, 4);
                enc.set_buffer(&outlier.channel_indices, 0, 5);
                enc.set_buffer(&outlier.outlier_rotation_signs, 0, 6);
                enc.set_buffer(&outlier.non_outlier_rotation_signs, 0, 7);
                enc.set_buffer(&outlier.k_outlier_codebook, 0, 8);
                enc.set_buffer(&outlier.k_non_outlier_codebook, 0, 9);
                enc.set_buffer(&bufs.attn_out, attn_out_offset, 10);
                enc.set_buffer(k_o_scale, 0, 11);
                enc.set_buffer(v_o_scale, 0, 12);
                enc.set_buffer(k_n_scale, 0, 13);
                enc.set_buffer(v_n_scale, 0, 14);
                enc.set_bytes(&nh.to_le_bytes(), 15);
                enc.set_bytes(&nkv.to_le_bytes(), 16);
                enc.set_bytes(&hd.to_le_bytes(), 17);
                enc.set_bytes(&max_seq.to_le_bytes(), 18);
                enc.set_bytes(&current_seq_len.to_le_bytes(), 19);
                enc.set_bytes(&outlier.n_outlier.to_le_bytes(), 20);
                enc.set_bytes(&outlier.d_outlier_padded.to_le_bytes(), 21);
                enc.set_bytes(&outlier.d_non_padded.to_le_bytes(), 22);
                enc.set_buffer(&outlier.outlier_qjl_matrix, 0, 23);
                enc.set_buffer(&outlier.non_outlier_qjl_matrix, 0, 24);
                enc.set_buffer(k_o_r_norms, 0, 25);
                enc.set_buffer(k_n_r_norms, 0, 26);
                enc.set_buffer(&outlier.outlier_codebook, 0, 27);
                enc.set_buffer(&outlier.non_outlier_codebook, 0, 28);
                enc.dispatch_threadgroups((nh as usize, 1, 1), (tg_size.min(1024), 1, 1));
            }
        } else {
            // ── Standard TurboQuant dispatch ──
            let (k_cache, v_cache) = kv.layer_caches(layer_idx);
            let (k_scale, v_scale) = kv.layer_scales(layer_idx);
            let (k_qjl_signs, k_r_norms) = kv.layer_k_qjl(layer_idx);

            let kv_head_stride_bytes = (nkv as usize) * (hd as usize) * 2;
            for t in 0..token_count {
                let token_offset = t * kv_head_stride_bytes;
                // K cache write: (b-1)-bit codebook + QJL
                enc.set_pipeline(&pipelines.turboquant_cache_write);
                enc.set_buffer(&bufs.k_proj, token_offset, 0);
                enc.set_buffer(&tq.rotation_signs, 0, 1);
                enc.set_buffer(k_cache, 0, 2);
                enc.set_bytes(&nkv.to_le_bytes(), 3);
                enc.set_bytes(&hd.to_le_bytes(), 4);
                enc.set_bytes(&max_seq.to_le_bytes(), 5);
                enc.set_bytes(&((seq_pos + t) as u32).to_le_bytes(), 6);
                enc.set_bytes(&tq.inv_scale.to_le_bytes(), 7);
                enc.set_bytes(&n_bits.to_le_bytes(), 8);
                enc.set_buffer(k_scale, 0, 9);
                enc.set_buffer(&tq.k_codebook_buf, 0, 10);
                enc.set_buffer(&tq.k_boundaries_buf, 0, 11);
                enc.set_bytes(&tq.k_n_levels.to_le_bytes(), 12);
                enc.set_buffer(&tq.qjl_matrix, 0, 13);
                enc.set_buffer(k_qjl_signs, 0, 14);
                enc.set_buffer(k_r_norms, 0, 15);
                enc.set_bytes(&1u32.to_le_bytes(), 16);
                enc.dispatch_threadgroups((nkv as usize, 1, 1), ((hd as usize).min(1024), 1, 1));
                // V cache write: b-bit codebook, no QJL
                enc.set_pipeline(&pipelines.turboquant_cache_write);
                enc.set_buffer(&bufs.v_proj, token_offset, 0);
                enc.set_buffer(&tq.rotation_signs, 0, 1);
                enc.set_buffer(v_cache, 0, 2);
                enc.set_bytes(&nkv.to_le_bytes(), 3);
                enc.set_bytes(&hd.to_le_bytes(), 4);
                enc.set_bytes(&max_seq.to_le_bytes(), 5);
                enc.set_bytes(&((seq_pos + t) as u32).to_le_bytes(), 6);
                enc.set_bytes(&tq.inv_scale.to_le_bytes(), 7);
                enc.set_bytes(&n_bits.to_le_bytes(), 8);
                enc.set_buffer(v_scale, 0, 9);
                enc.set_buffer(&tq.v_codebook_buf, 0, 10);
                enc.set_buffer(&tq.v_boundaries_buf, 0, 11);
                enc.set_bytes(&tq.v_n_levels.to_le_bytes(), 12);
                enc.set_buffer(&tq.qjl_matrix, 0, 13);
                enc.set_buffer(k_qjl_signs, 0, 14);
                enc.set_buffer(k_r_norms, 0, 15);
                enc.set_bytes(&0u32.to_le_bytes(), 16);
                enc.dispatch_threadgroups((nkv as usize, 1, 1), ((hd as usize).min(1024), 1, 1));
            }

            let q_head_stride_bytes = (nh as usize) * (hd as usize) * 2;
            for t in 0..token_count {
                let q_offset = t * q_head_stride_bytes;
                let attn_out_offset = t * q_head_stride_bytes;
                let current_seq_len = (seq_pos + t + 1) as u32;
                enc.set_pipeline(&pipelines.turboquant_attention);
                enc.set_buffer(&bufs.q_proj, q_offset, 0);
                enc.set_buffer(k_cache, 0, 1);
                enc.set_buffer(v_cache, 0, 2);
                enc.set_buffer(&tq.rotation_signs, 0, 3);
                enc.set_buffer(&bufs.attn_out, attn_out_offset, 4);
                enc.set_bytes(&nh.to_le_bytes(), 5);
                enc.set_bytes(&nkv.to_le_bytes(), 6);
                enc.set_bytes(&hd.to_le_bytes(), 7);
                enc.set_bytes(&max_seq.to_le_bytes(), 8);
                enc.set_bytes(&current_seq_len.to_le_bytes(), 9);
                enc.set_bytes(&tq.deq_scale.to_le_bytes(), 10);
                enc.set_bytes(&n_bits.to_le_bytes(), 11);
                enc.set_buffer(k_scale, 0, 12);
                enc.set_buffer(v_scale, 0, 13);
                enc.set_buffer(&tq.k_codebook_buf, 0, 14);
                enc.set_buffer(&tq.v_codebook_buf, 0, 15);
                enc.set_buffer(&tq.qjl_matrix, 0, 16);
                enc.set_buffer(k_r_norms, 0, 17);
                enc.dispatch_threadgroups((nh as usize, 1, 1), ((hd as usize).min(1024), 1, 1));
            }
        }
        enc.end_encoding();
    } else {
        // FP16 KV cache path — scatter projections into cache on GPU.
        let fp16_kv =
            fp16_kv_cache.expect("fp16_kv_cache must be initialized for FP16 KV cache path");
        let (k_cache, v_cache) = fp16_kv.layer_caches(layer_idx);

        let enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        // Scatter K and V projections into their caches entirely on GPU.
        ops::encode_kv_scatter(
            &enc,
            &pipelines.kv_scatter,
            &ops::KvScatterParams {
                proj: &bufs.k_proj,
                cache: k_cache,
                seq_pos: seq_pos as u32,
                token_count: token_count as u32,
                num_kv_heads: nkv,
                head_dim: hd,
                max_seq_len: max_seq,
            },
        );
        ops::encode_kv_scatter(
            &enc,
            &pipelines.kv_scatter,
            &ops::KvScatterParams {
                proj: &bufs.v_proj,
                cache: v_cache,
                seq_pos: seq_pos as u32,
                token_count: token_count as u32,
                num_kv_heads: nkv,
                head_dim: hd,
                max_seq_len: max_seq,
            },
        );

        // Standard attention — loop over tokens with causal
        // masking so each token attends only to 0..seq_pos+t+1.
        let q_head_stride_bytes = (nh as usize) * (hd as usize) * 2;
        for t in 0..token_count {
            let q_offset = t * q_head_stride_bytes;
            let attn_out_offset = t * q_head_stride_bytes;
            let current_seq_len = (seq_pos + t + 1) as u32;
            enc.set_pipeline(&pipelines.standard_attention);
            enc.set_buffer(&bufs.q_proj, q_offset, 0);
            enc.set_buffer(k_cache, 0, 1);
            enc.set_buffer(v_cache, 0, 2);
            enc.set_buffer(&bufs.attn_out, attn_out_offset, 3);
            enc.set_bytes(&nh.to_le_bytes(), 4);
            enc.set_bytes(&nkv.to_le_bytes(), 5);
            enc.set_bytes(&hd.to_le_bytes(), 6);
            enc.set_bytes(&max_seq.to_le_bytes(), 7);
            enc.set_bytes(&current_seq_len.to_le_bytes(), 8);
            enc.dispatch_threadgroups((nh as usize, 1, 1), ((hd as usize).min(1024), 1, 1));
        }
        enc.end_encoding();
    }

    Ok(())
}

/// Encode the FFN block: gate + up projections, SiLU activation, and down projection.
fn encode_ffn_block(
    cmd_buf: &ironmill_metal_sys::CommandBuffer,
    pipelines: &super::ops::MetalPipelines,
    bufs: &IntermediateBuffers,
    lw: &LayerWeights,
    lm: &LayerMatmuls,
    h: usize,
    inter: usize,
    token_count: usize,
) -> Result<(), InferenceError> {
    let row_bytes_h = h * 2;
    let row_bytes_inter = inter * 2;

    let norm_mat = MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
        .map_err(|e| InferenceError::Runtime(e.to_string()))?;

    // Gate projection
    encode_projection(
        cmd_buf,
        &bufs.norm_out,
        &norm_mat,
        &lw.gate_proj,
        &bufs.ffn_gate,
        &lm.gate,
        pipelines,
        token_count,
        inter,
        h,
        row_bytes_h,
        row_bytes_inter,
    )?;

    // Up projection
    encode_projection(
        cmd_buf,
        &bufs.norm_out,
        &norm_mat,
        &lw.up_proj,
        &bufs.ffn_up,
        &lm.up,
        pipelines,
        token_count,
        inter,
        h,
        row_bytes_h,
        row_bytes_inter,
    )?;

    // SiLU gate
    {
        let enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        ops::encode_silu_gate(
            &enc,
            &pipelines.silu_gate,
            &bufs.ffn_gate,
            &bufs.ffn_up,
            &bufs.ffn_gate, // in-place output into gate buffer
            (token_count * inter) as u32,
        );
        enc.end_encoding();
    }

    // Down projection
    let gate_mat = MpsMatrix::from_buffer(&bufs.ffn_gate, token_count, inter, row_bytes_inter)
        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
    encode_projection(
        cmd_buf,
        &bufs.ffn_gate,
        &gate_mat,
        &lw.down_proj,
        &bufs.ffn_down,
        &lm.down,
        pipelines,
        token_count,
        h,
        inter,
        row_bytes_inter,
        row_bytes_h,
    )?;

    Ok(())
}

/// Encode end-of-layer residual: fused with next layer's norm, or standalone for the last layer.
fn encode_end_of_layer_residual(
    cmd_buf: &ironmill_metal_sys::CommandBuffer,
    pipelines: &super::ops::MetalPipelines,
    bufs: &IntermediateBuffers,
    next_input_norm: Option<&MetalBuffer>,
    h: usize,
    token_count: usize,
    eps: f32,
) -> Result<(), InferenceError> {
    if let Some(norm_weight) = next_input_norm {
        let enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        ops::encode_fused_residual_rms_norm(
            &enc,
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
        enc.end_encoding();
    } else {
        let enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        ops::encode_residual_add(
            &enc,
            &pipelines.residual_add,
            &bufs.residual,
            &bufs.ffn_down,
            &bufs.hidden_state,
            (token_count * h) as u32,
        );
        enc.end_encoding();
    }
    Ok(())
}

impl InferenceEngine for MetalInference {
    fn load(&mut self, artifacts: &dyn Any) -> Result<(), InferenceError> {
        let gpu_artifacts = artifacts
            .downcast_ref::<MetalArtifacts<'_>>()
            .ok_or_else(|| {
                InferenceError::Runtime("MetalInference::load expects MetalArtifacts".into())
            })?;

        self.config = gpu_artifacts.config.clone();

        // Load weights into Metal buffers.
        let weights = MetalWeights::load(
            &self.device,
            gpu_artifacts.weights,
            self.config.force_cpu_dequant,
        )
        .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        let mc = &weights.config;
        self.model_config = Some(mc.clone());

        // Pre-allocate logits readback buffer (vocab × 2 bytes for FP16).
        self.logits_fp16_buf.resize(mc.vocab_size * 2, 0);

        // Compile Metal shader pipelines with the model's head_dim so
        // shared memory is sized exactly via #define HEAD_DIM.
        let pipelines = super::ops::MetalPipelines::compile(&self.device, mc.head_dim)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        self.pipelines = Some(pipelines);

        // Allocate intermediate buffers (sized for single-token decode;
        // prefill will rebuild the matmul cache for larger token counts).
        let max_prefill = self.config.prefill_chunk_size.unwrap_or(512).max(1);
        let bufs = IntermediateBuffers::allocate(&self.device, max_prefill, mc)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        self.intermediate_buffers = Some(bufs);

        // Build RoPE cos/sin caches.
        let (cos, sin) = Self::build_rope_cache(
            &self.device,
            mc.head_dim,
            self.config.max_seq_len,
            mc.rope_theta,
        )
        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        self.rope_cos = Some(cos);
        self.rope_sin = Some(sin);

        // Build MPS matmul cache for single-token decode.
        let decode_cache_t1 = Self::build_matmul_cache(&self.device, mc, &weights, 1)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        self.decode_matmuls_t1 = Some(decode_cache_t1);
        self.decode_matmuls = None;

        // Initialize KV cache.
        if self.config.enable_turboquant {
            let tq_config = TurboQuantMetalConfig {
                n_bits: self.config.n_bits,
                num_kv_heads: mc.num_key_value_heads,
                head_dim: mc.head_dim,
                max_seq_len: self.config.max_seq_len,
                num_layers: mc.num_hidden_layers,
                rotation_seed: self.config.rotation_seed,
                outlier: None,
            };
            let tq_model = MetalTurboQuantModel::new(&self.device, tq_config.clone())
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let kv_cache = MetalKvCache::new(&self.device, &tq_config)
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            self.turboquant = Some(tq_model);
            self.kv_cache = Some(kv_cache);
            self.fp16_kv_cache = None;
        } else {
            let fp16_kv = Fp16KvCache::new(
                &self.device,
                mc.num_hidden_layers,
                mc.num_key_value_heads,
                self.config.max_seq_len,
                mc.head_dim,
            )
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            self.fp16_kv_cache = Some(fp16_kv);
            self.turboquant = None;
            self.kv_cache = None;
        }

        self.weights = Some(weights);
        self.seq_pos = 0;
        Ok(())
    }

    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError> {
        self.run_pipeline(&[token])
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::Decode("empty prefill tokens".into()));
        }

        let chunk_size = self.config.prefill_chunk_size.unwrap_or(tokens.len());
        let chunk_size = chunk_size.max(1);

        let mut last_logits = None;
        for chunk in tokens.chunks(chunk_size) {
            last_logits = Some(self.run_pipeline(chunk)?);
        }

        last_logits.ok_or_else(|| InferenceError::Decode("no chunks processed".into()))
    }

    fn reset(&mut self) {
        self.seq_pos = 0;
        if let Some(kv) = self.kv_cache.as_mut() {
            kv.reset();
        }
        if let Some(fp16_kv) = self.fp16_kv_cache.as_mut() {
            fp16_kv.reset();
        }
    }
}

impl crate::calibration::CalibratingEngine for MetalInference {
    fn prefill_with_hooks(
        &mut self,
        tokens: &[u32],
        hooks: &mut dyn ActivationHook,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.prefill_with_hooks(tokens, hooks)?;
        Ok(())
    }

    fn reset(&mut self) {
        InferenceEngine::reset(self);
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
