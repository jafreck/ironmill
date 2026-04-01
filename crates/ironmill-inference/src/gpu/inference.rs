//! GPU inference engine implementing the [`InferenceEngine`] trait.
//!
//! Runs the full LLaMA-family transformer decode pipeline on Metal:
//!   - MPS `MPSMatrixMultiplication` for linear projections
//!   - Custom Metal compute shaders for RMSNorm, RoPE, SiLU, residual add,
//!     embedding lookup, and attention
//!   - Optional TurboQuant INT8 KV cache compression

use std::any::Any;

use half::f16;
use ironmill_metal_sys::{
    CommandBufferStatus, MetalBuffer, MetalDevice, MpsMatrix, MpsMatrixMultiply, StorageMode,
};
use mil_rs::weights::{ModelConfig, WeightProvider};

use super::config::GpuConfig;
use super::error::GpuError;
use super::ops;
use super::turboquant::{GpuKvCache, GpuTurboQuantModel, OutlierConfig, TurboQuantGpuConfig};
use super::weights::{GpuWeights, QuantizedWeight, WeightBuffer};
use crate::engine::{InferenceEngine, InferenceError};
use crate::types::Logits;

// ── Public artifacts type for load() ────────────────────────────

/// Artifacts passed to [`GpuInference::load`] via the type-erased
/// [`InferenceEngine`] interface.
pub struct GpuArtifacts<'a> {
    pub weights: &'a dyn WeightProvider,
    pub config: GpuConfig,
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
    ) -> Result<Self, GpuError> {
        let size_bytes = num_kv_heads * max_seq_len * head_dim * 2; // FP16
        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_caches.push(
                device
                    .create_buffer(size_bytes, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
            );
            v_caches.push(
                device
                    .create_buffer(size_bytes, StorageMode::Shared)
                    .map_err(GpuError::Metal)?,
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
    ) -> Result<Self, GpuError> {
        let h = mc.hidden_size;
        let nh = mc.num_attention_heads;
        let nkv = mc.num_key_value_heads;
        let hd = mc.head_dim;
        let inter = mc.intermediate_size;
        let vocab = mc.vocab_size;

        let alloc = |size_elems: usize| -> Result<MetalBuffer, GpuError> {
            // FP16 = 2 bytes per element; minimum 16 bytes for Metal
            let bytes = (size_elems * 2).max(16);
            device
                .create_buffer(bytes, StorageMode::Shared)
                .map_err(GpuError::Metal)
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
                .map_err(GpuError::Metal)?,
        })
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
    q: Option<MpsMatrixMultiply>,
    k: Option<MpsMatrixMultiply>,
    v: Option<MpsMatrixMultiply>,
    o: Option<MpsMatrixMultiply>,
    gate: Option<MpsMatrixMultiply>,
    up: Option<MpsMatrixMultiply>,
    down: Option<MpsMatrixMultiply>,
}

// ── GpuInference ────────────────────────────────────────────────

/// Metal GPU inference engine.
///
/// Implements the full transformer decode pipeline using Metal compute
/// shaders for element-wise ops and MPS for matrix multiplication.
pub struct GpuInference {
    device: MetalDevice,
    queue: ironmill_metal_sys::CommandQueue,
    pipelines: super::ops::GpuPipelines,
    weights: Option<GpuWeights>,
    turboquant: Option<GpuTurboQuantModel>,
    kv_cache: Option<GpuKvCache>,
    fp16_kv_cache: Option<Fp16KvCache>,
    intermediate_buffers: Option<IntermediateBuffers>,
    rope_cos: Option<MetalBuffer>,
    rope_sin: Option<MetalBuffer>,
    /// MPS matmul cache for decode (token_count=1).
    decode_matmuls: Option<MpsMatmulCache>,
    config: GpuConfig,
    model_config: Option<ModelConfig>,
    seq_pos: usize,
}

impl GpuInference {
    /// Create a new GPU inference engine (device + queue + shader pipelines).
    pub fn new(config: GpuConfig) -> Result<Self, GpuError> {
        config.validate().map_err(|e| GpuError::Config(e))?;
        let device = MetalDevice::system_default().map_err(GpuError::Metal)?;
        let queue = device.create_command_queue().map_err(GpuError::Metal)?;
        let pipelines = super::ops::GpuPipelines::compile(&device)?;
        Ok(Self {
            device,
            queue,
            pipelines,
            weights: None,
            turboquant: None,
            kv_cache: None,
            fp16_kv_cache: None,
            intermediate_buffers: None,
            rope_cos: None,
            rope_sin: None,
            decode_matmuls: None,
            config,
            model_config: None,
            seq_pos: 0,
        })
    }

    /// Load model weights directly from a [`WeightProvider`], bypassing
    /// the type-erased [`InferenceEngine::load`] interface.
    pub fn load_weights(
        &mut self,
        provider: &dyn mil_rs::weights::WeightProvider,
        config: GpuConfig,
    ) -> Result<(), InferenceError> {
        self.config = config;

        let weights = GpuWeights::load(&self.device, provider, self.config.force_cpu_dequant)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        let mc = &weights.config;
        self.model_config = Some(mc.clone());

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

        let decode_cache = Self::build_matmul_cache(&self.device, mc, &weights, 1)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        self.decode_matmuls = Some(decode_cache);

        if self.config.enable_turboquant {
            // Detect outlier channels from K/V weight column norms (§4.3).
            // Only enabled for INT4 where the quality gap benefits from it.
            let outlier_cfg = if self.config.n_bits == 4 {
                let weight_data: Vec<(Vec<u8>, Vec<u8>)> = weights
                    .layers
                    .iter()
                    .filter_map(|lw| {
                        if let (
                            crate::gpu::weights::WeightBuffer::Dense(k_buf),
                            crate::gpu::weights::WeightBuffer::Dense(v_buf),
                        ) = (&lw.k_proj, &lw.v_proj)
                        {
                            let mut k_data = vec![0u8; k_buf.length()];
                            let mut v_data = vec![0u8; v_buf.length()];
                            k_buf.read_bytes(&mut k_data, 0).ok()?;
                            v_buf.read_bytes(&mut v_data, 0).ok()?;
                            Some((k_data, v_data))
                        } else {
                            None
                        }
                    })
                    .collect();
                if !weight_data.is_empty() {
                    let refs: Vec<(&[u8], &[u8])> = weight_data
                        .iter()
                        .map(|(k, v)| (k.as_slice(), v.as_slice()))
                        .collect();
                    let out_features = mc.num_key_value_heads * mc.head_dim;
                    Some(OutlierConfig::auto_from_weights(
                        &refs,
                        out_features,
                        mc.head_dim,
                    ))
                } else {
                    None
                }
            } else {
                None
            };

            let tq_config = TurboQuantGpuConfig {
                n_bits: self.config.n_bits,
                num_kv_heads: mc.num_key_value_heads,
                head_dim: mc.head_dim,
                max_seq_len: self.config.max_seq_len,
                num_layers: mc.num_hidden_layers,
                rotation_seed: self.config.rotation_seed,
                outlier: outlier_cfg,
            };
            let tq_model = GpuTurboQuantModel::new(&self.device, tq_config.clone())
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let kv_cache = GpuKvCache::new(&self.device, &tq_config)
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
    ) -> Result<(MetalBuffer, MetalBuffer), GpuError> {
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
            .map_err(GpuError::Metal)?;
        let sin_buf = device
            .create_buffer_with_data(&sin_data, StorageMode::Shared)
            .map_err(GpuError::Metal)?;
        Ok((cos_buf, sin_buf))
    }

    // ── MPS matmul cache ────────────────────────────────────────

    fn build_matmul_cache(
        device: &MetalDevice,
        mc: &ModelConfig,
        weights: &GpuWeights,
        token_count: usize,
    ) -> Result<MpsMatmulCache, GpuError> {
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
            |rows: usize, cols: usize, inner: usize| -> Result<MpsMatrixMultiply, GpuError> {
                MpsMatrixMultiply::new(
                    device, false, // transpose_left
                    true,  // transpose_right (weights are [out, in])
                    rows,  // result_rows = token_count
                    cols,  // result_columns = out_features
                    inner, // interior_columns = in_features
                    1.0,   // alpha
                    0.0,   // beta
                )
                .map_err(GpuError::Metal)
            };

        // Only create MPS matmul instances for Dense weights; Quantized
        // projections use the custom compute kernel path instead.
        let dense_matmul = |wb: &WeightBuffer,
                            rows: usize,
                            cols: usize,
                            inner: usize|
         -> Result<Option<MpsMatrixMultiply>, GpuError> {
            match wb {
                WeightBuffer::Dense(_) => Ok(Some(make_matmul(rows, cols, inner)?)),
                WeightBuffer::Quantized(_) => Ok(None),
            }
        };

        let mut layer_matmuls = Vec::with_capacity(mc.num_hidden_layers);
        for i in 0..mc.num_hidden_layers {
            let lw = &weights.layers[i];
            layer_matmuls.push(LayerMatmuls {
                q: dense_matmul(&lw.q_proj, token_count, nh * hd, h)?,
                k: dense_matmul(&lw.k_proj, token_count, nkv * hd, h)?,
                v: dense_matmul(&lw.v_proj, token_count, nkv * hd, h)?,
                o: dense_matmul(&lw.o_proj, token_count, h, nh * hd)?,
                gate: dense_matmul(&lw.gate_proj, token_count, inter, h)?,
                up: dense_matmul(&lw.up_proj, token_count, inter, h)?,
                down: dense_matmul(&lw.down_proj, token_count, h, inter)?,
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
        let matmuls = self.decode_matmuls.as_ref().unwrap();

        // Write token IDs to GPU buffer.
        let token_bytes: Vec<u8> = token_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        bufs.token_ids_buf
            .write_bytes(&token_bytes, 0)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        // Create command buffer.
        let mut cmd_buf = self
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
                &self.pipelines.embedding_lookup,
                &bufs.token_ids_buf,
                &weights.embedding,
                &bufs.hidden_state,
                h as u32,
                token_count as u32,
                vocab as u32,
            );
            enc.end_encoding();
        }

        // Per-layer processing.
        for layer_idx in 0..mc.num_hidden_layers {
            let lw = &weights.layers[layer_idx];
            let lm = &matmuls.layer_matmuls[layer_idx];

            // Copy hidden_state → residual before layernorm.
            // We use residual_add with a zero source to copy, but actually
            // we just need to swap buffer roles. Since we can't swap,
            // we encode a copy via residual_add(hidden, zero) — but that's
            // wasteful. Instead, use the pattern: norm(hidden→norm_out),
            // then at residual add time, add hidden (original) + proj_out.
            // So `residual` holds the pre-norm hidden state.
            //
            // Actually: we copy hidden → residual via a residual_add with
            // the input as source and a zero-add. Simpler: just track that
            // the residual IS the hidden_state buffer at this point.
            //
            // The approach: after norm, we still have the pre-norm hidden_state.
            // We only overwrite hidden_state at the residual add step.
            // So the residual connection reads from hidden_state (pre-norm value).

            // Step 2: RMSNorm (input layernorm)
            {
                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                ops::encode_rms_norm(
                    &enc,
                    &self.pipelines.rms_norm,
                    &bufs.hidden_state,
                    &lw.input_norm,
                    &bufs.norm_out,
                    h as u32,
                    token_count as u32,
                    eps,
                );
                enc.end_encoding();
            }

            // Steps 3-5: Q/K/V projections — dispatch by weight type.
            // Dense weights use MPS matmul (encodes directly onto cmd_buf).
            // Quantized weights use a compute encoder with the PolarQuant kernel.
            let row_bytes_h = h * 2; // FP16
            let row_bytes_qo = (mc.num_attention_heads * mc.head_dim) * 2;
            let row_bytes_kv = (mc.num_kv_heads() * mc.head_dim) * 2;
            let row_bytes_inter = inter * 2;

            let norm_mat = MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;

            // Q projection
            match &lw.q_proj {
                WeightBuffer::Dense(buf) => {
                    let q_weight_mat = MpsMatrix::from_buffer(
                        buf,
                        mc.num_attention_heads * mc.head_dim,
                        h,
                        row_bytes_h,
                    )
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    let q_result_mat = MpsMatrix::from_buffer(
                        &bufs.q_proj,
                        token_count,
                        mc.num_attention_heads * mc.head_dim,
                        row_bytes_qo,
                    )
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    lm.q.as_ref().unwrap().encode(
                        &cmd_buf,
                        &norm_mat,
                        &q_weight_mat,
                        &q_result_mat,
                    );
                }
                WeightBuffer::Quantized(q) => {
                    let enc = cmd_buf
                        .compute_encoder()
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    encode_polarquant_matmul(
                        &enc,
                        &bufs.norm_out,
                        q,
                        &bufs.q_proj,
                        &self.pipelines,
                        token_count,
                    );
                    enc.end_encoding();
                }
            }

            // K projection
            match &lw.k_proj {
                WeightBuffer::Dense(buf) => {
                    let k_weight_mat = MpsMatrix::from_buffer(
                        buf,
                        mc.num_kv_heads() * mc.head_dim,
                        h,
                        row_bytes_h,
                    )
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    let k_result_mat = MpsMatrix::from_buffer(
                        &bufs.k_proj,
                        token_count,
                        mc.num_kv_heads() * mc.head_dim,
                        row_bytes_kv,
                    )
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    lm.k.as_ref().unwrap().encode(
                        &cmd_buf,
                        &norm_mat,
                        &k_weight_mat,
                        &k_result_mat,
                    );
                }
                WeightBuffer::Quantized(q) => {
                    let enc = cmd_buf
                        .compute_encoder()
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    encode_polarquant_matmul(
                        &enc,
                        &bufs.norm_out,
                        q,
                        &bufs.k_proj,
                        &self.pipelines,
                        token_count,
                    );
                    enc.end_encoding();
                }
            }

            // V projection
            match &lw.v_proj {
                WeightBuffer::Dense(buf) => {
                    let v_weight_mat = MpsMatrix::from_buffer(
                        buf,
                        mc.num_kv_heads() * mc.head_dim,
                        h,
                        row_bytes_h,
                    )
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    let v_result_mat = MpsMatrix::from_buffer(
                        &bufs.v_proj,
                        token_count,
                        mc.num_kv_heads() * mc.head_dim,
                        row_bytes_kv,
                    )
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    lm.v.as_ref().unwrap().encode(
                        &cmd_buf,
                        &norm_mat,
                        &v_weight_mat,
                        &v_result_mat,
                    );
                }
                WeightBuffer::Quantized(q) => {
                    let enc = cmd_buf
                        .compute_encoder()
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    encode_polarquant_matmul(
                        &enc,
                        &bufs.norm_out,
                        q,
                        &bufs.v_proj,
                        &self.pipelines,
                        token_count,
                    );
                    enc.end_encoding();
                }
            }

            // QK normalization (Qwen3): per-head RMSNorm on Q and K before RoPE.
            if let (Some(q_norm_w), Some(k_norm_w)) = (&lw.q_norm, &lw.k_norm) {
                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                ops::encode_rms_norm(
                    &enc,
                    &self.pipelines.rms_norm,
                    &bufs.q_proj,
                    q_norm_w,
                    &bufs.q_proj,
                    hd,
                    token_count as u32 * nh,
                    eps,
                );
                ops::encode_rms_norm(
                    &enc,
                    &self.pipelines.rms_norm,
                    &bufs.k_proj,
                    k_norm_w,
                    &bufs.k_proj,
                    hd,
                    token_count as u32 * nkv,
                    eps,
                );
                enc.end_encoding();
            }

            // Step 6: RoPE on Q and K
            {
                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                ops::encode_rope(
                    &enc,
                    &self.pipelines.rope,
                    &bufs.q_proj,
                    rope_cos,
                    rope_sin,
                    nh,
                    hd,
                    seq_pos as u32,
                    token_count as u32,
                );
                ops::encode_rope(
                    &enc,
                    &self.pipelines.rope,
                    &bufs.k_proj,
                    rope_cos,
                    rope_sin,
                    nkv,
                    hd,
                    seq_pos as u32,
                    token_count as u32,
                );
                enc.end_encoding();
            }

            // Steps 7-8: Cache write + attention
            if enable_tq {
                let tq = self.turboquant.as_ref().unwrap();
                let kv = self.kv_cache.as_ref().unwrap();
                let max_seq = self.config.max_seq_len as u32;
                let n_bits = self.config.n_bits as u32;

                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;

                if let Some(ref outlier) = tq.outlier {
                    // ── Outlier channel strategy dispatch ──
                    let ((k_o_cache, v_o_cache), (k_n_cache, v_n_cache)) =
                        kv.layer_outlier_caches(layer_idx);
                    let ((k_o_scale, v_o_scale), (k_n_scale, v_n_scale)) =
                        kv.layer_outlier_scales(layer_idx);
                    let kv_head_stride_bytes = (nkv as usize) * (hd as usize) * 2;
                    let tg_size = std::cmp::max(
                        outlier.d_outlier_padded as usize,
                        outlier.d_non_padded as usize,
                    );

                    for t in 0..token_count {
                        let token_offset = t * kv_head_stride_bytes;
                        for (proj_buf, o_cache, n_cache, o_scale, n_scale) in [
                            (&bufs.k_proj, k_o_cache, k_n_cache, k_o_scale, k_n_scale),
                            (&bufs.v_proj, v_o_cache, v_n_cache, v_o_scale, v_n_scale),
                        ] {
                            enc.set_pipeline(&self.pipelines.turboquant_outlier_cache_write);
                            enc.set_buffer(proj_buf, token_offset, 0);
                            enc.set_buffer(&outlier.channel_indices, 0, 1);
                            enc.set_buffer(o_cache, 0, 2);
                            enc.set_buffer(n_cache, 0, 3);
                            enc.set_buffer(&outlier.outlier_rotation_signs, 0, 4);
                            enc.set_buffer(&outlier.non_outlier_rotation_signs, 0, 5);
                            enc.set_buffer(&outlier.outlier_codebook, 0, 6);
                            enc.set_buffer(&outlier.outlier_boundaries, 0, 7);
                            enc.set_buffer(&outlier.non_outlier_codebook, 0, 8);
                            enc.set_buffer(&outlier.non_outlier_boundaries, 0, 9);
                            enc.set_buffer(o_scale, 0, 10);
                            enc.set_buffer(n_scale, 0, 11);
                            enc.set_bytes(&nkv.to_le_bytes(), 12);
                            enc.set_bytes(&hd.to_le_bytes(), 13);
                            enc.set_bytes(&max_seq.to_le_bytes(), 14);
                            enc.set_bytes(&((seq_pos + t) as u32).to_le_bytes(), 15);
                            enc.set_bytes(&outlier.n_outlier.to_le_bytes(), 16);
                            enc.set_bytes(&outlier.d_outlier_padded.to_le_bytes(), 17);
                            enc.set_bytes(&outlier.d_non_padded.to_le_bytes(), 18);
                            enc.set_bytes(&outlier.outlier_n_levels.to_le_bytes(), 19);
                            enc.set_bytes(&outlier.non_outlier_n_levels.to_le_bytes(), 20);
                            enc.dispatch_threadgroups((nkv as usize, 1, 1), (tg_size, 1, 1));
                        }
                    }

                    let q_head_stride_bytes = (nh as usize) * (hd as usize) * 2;
                    for t in 0..token_count {
                        let q_offset = t * q_head_stride_bytes;
                        let attn_out_offset = t * q_head_stride_bytes;
                        let current_seq_len = (seq_pos + t + 1) as u32;
                        enc.set_pipeline(&self.pipelines.turboquant_outlier_attention);
                        enc.set_buffer(&bufs.q_proj, q_offset, 0);
                        enc.set_buffer(k_o_cache, 0, 1);
                        enc.set_buffer(v_o_cache, 0, 2);
                        enc.set_buffer(k_n_cache, 0, 3);
                        enc.set_buffer(v_n_cache, 0, 4);
                        enc.set_buffer(&outlier.channel_indices, 0, 5);
                        enc.set_buffer(&outlier.outlier_rotation_signs, 0, 6);
                        enc.set_buffer(&outlier.non_outlier_rotation_signs, 0, 7);
                        enc.set_buffer(&outlier.outlier_codebook, 0, 8);
                        enc.set_buffer(&outlier.non_outlier_codebook, 0, 9);
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
                        enc.dispatch_threadgroups((nh as usize, 1, 1), (tg_size, 1, 1));
                    }
                } else {
                    // ── Standard TurboQuant dispatch ──
                    let (k_cache, v_cache) = kv.layer_caches(layer_idx);
                    let (k_scale, v_scale) = kv.layer_scales(layer_idx);
                    let (k_qjl_signs, k_r_norms) = kv.layer_k_qjl(layer_idx);

                    let kv_head_stride_bytes = (nkv as usize) * (hd as usize) * 2;
                    for t in 0..token_count {
                        let token_offset = t * kv_head_stride_bytes;
                        // K cache write (with QJL)
                        enc.set_pipeline(&self.pipelines.turboquant_cache_write);
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
                        enc.set_buffer(&tq.codebook_buf, 0, 10);
                        enc.set_buffer(&tq.boundaries_buf, 0, 11);
                        enc.set_bytes(&tq.n_levels.to_le_bytes(), 12);
                        enc.set_buffer(&tq.qjl_matrix, 0, 13);
                        enc.set_buffer(k_qjl_signs, 0, 14);
                        enc.set_buffer(k_r_norms, 0, 15);
                        enc.set_bytes(&1u32.to_le_bytes(), 16);
                        enc.dispatch_threadgroups((nkv as usize, 1, 1), (hd as usize, 1, 1));
                        // V cache write (no QJL)
                        enc.set_pipeline(&self.pipelines.turboquant_cache_write);
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
                        enc.set_buffer(&tq.codebook_buf, 0, 10);
                        enc.set_buffer(&tq.boundaries_buf, 0, 11);
                        enc.set_bytes(&tq.n_levels.to_le_bytes(), 12);
                        enc.set_buffer(&tq.qjl_matrix, 0, 13);
                        enc.set_buffer(k_qjl_signs, 0, 14);
                        enc.set_buffer(k_r_norms, 0, 15);
                        enc.set_bytes(&0u32.to_le_bytes(), 16);
                        enc.dispatch_threadgroups((nkv as usize, 1, 1), (hd as usize, 1, 1));
                    }

                    let q_head_stride_bytes = (nh as usize) * (hd as usize) * 2;
                    for t in 0..token_count {
                        let q_offset = t * q_head_stride_bytes;
                        let attn_out_offset = t * q_head_stride_bytes;
                        let current_seq_len = (seq_pos + t + 1) as u32;
                        enc.set_pipeline(&self.pipelines.turboquant_attention);
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
                        enc.set_buffer(&tq.codebook_buf, 0, 14);
                        enc.set_buffer(&tq.qjl_matrix, 0, 15);
                        enc.set_buffer(k_qjl_signs, 0, 16);
                        enc.set_buffer(k_r_norms, 0, 17);
                        enc.dispatch_threadgroups((nh as usize, 1, 1), (hd as usize, 1, 1));
                    }
                }
                enc.end_encoding();
            } else {
                // FP16 KV cache path
                let fp16_kv = self.fp16_kv_cache.as_ref().unwrap();
                let (k_cache, v_cache) = fp16_kv.layer_caches(layer_idx);
                let max_seq = self.config.max_seq_len as u32;

                // Must commit current command buffer so MPS matmul results
                // are available for CPU-side KV cache scatter.
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
                cmd_buf = self
                    .queue
                    .command_buffer()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;

                // Write K/V into FP16 cache at seq_pos via buffer copy.
                // Each K/V projection is [token_count × num_kv_heads × head_dim] FP16.
                // Cache layout: [num_kv_heads × max_seq × head_dim] FP16.
                // For simplicity, write via CPU side since buffers are Shared.
                let kv_size = mc.num_kv_heads() * mc.head_dim * token_count * 2;
                let mut k_data = vec![0u8; kv_size];
                let mut v_data = vec![0u8; kv_size];
                bufs.k_proj
                    .read_bytes(&mut k_data, 0)
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                bufs.v_proj
                    .read_bytes(&mut v_data, 0)
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;

                // Write token-by-token into the cache (interleaving heads).
                // Projection layout: [token × nkv × hd] row-major.
                // Cache layout: [nkv × max_seq × hd] row-major.
                for t in 0..token_count {
                    for head in 0..mc.num_kv_heads() {
                        let src_off =
                            (t * mc.num_kv_heads() * mc.head_dim + head * mc.head_dim) * 2;
                        let dst_off = (head * self.config.max_seq_len * mc.head_dim
                            + (seq_pos + t) * mc.head_dim)
                            * 2;
                        let len = mc.head_dim * 2;
                        k_cache
                            .write_bytes(&k_data[src_off..src_off + len], dst_off)
                            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                        v_cache
                            .write_bytes(&v_data[src_off..src_off + len], dst_off)
                            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    }
                }

                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                // Standard attention — loop over tokens with causal
                // masking so each token attends only to 0..seq_pos+t+1.
                let q_head_stride_bytes = (nh as usize) * (hd as usize) * 2;
                for t in 0..token_count {
                    let q_offset = t * q_head_stride_bytes;
                    let attn_out_offset = t * q_head_stride_bytes;
                    let current_seq_len = (seq_pos + t + 1) as u32;
                    enc.set_pipeline(&self.pipelines.standard_attention);
                    enc.set_buffer(&bufs.q_proj, q_offset, 0);
                    enc.set_buffer(k_cache, 0, 1);
                    enc.set_buffer(v_cache, 0, 2);
                    enc.set_buffer(&bufs.attn_out, attn_out_offset, 3);
                    enc.set_bytes(&nh.to_le_bytes(), 4);
                    enc.set_bytes(&nkv.to_le_bytes(), 5);
                    enc.set_bytes(&hd.to_le_bytes(), 6);
                    enc.set_bytes(&max_seq.to_le_bytes(), 7);
                    enc.set_bytes(&current_seq_len.to_le_bytes(), 8);
                    enc.dispatch_threadgroups((nh as usize, 1, 1), (hd as usize, 1, 1));
                }
                enc.end_encoding();
            }

            // Step 9: Output projection
            match &lw.o_proj {
                WeightBuffer::Dense(buf) => {
                    let attn_mat = MpsMatrix::from_buffer(
                        &bufs.attn_out,
                        token_count,
                        mc.num_attention_heads * mc.head_dim,
                        row_bytes_qo,
                    )
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    let o_weight_mat = MpsMatrix::from_buffer(
                        buf,
                        h,
                        mc.num_attention_heads * mc.head_dim,
                        row_bytes_qo,
                    )
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    let ffn_down_mat_for_o =
                        MpsMatrix::from_buffer(&bufs.ffn_down, token_count, h, row_bytes_h)
                            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    lm.o.as_ref().unwrap().encode(
                        &cmd_buf,
                        &attn_mat,
                        &o_weight_mat,
                        &ffn_down_mat_for_o,
                    );
                }
                WeightBuffer::Quantized(q) => {
                    let enc = cmd_buf
                        .compute_encoder()
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    encode_polarquant_matmul(
                        &enc,
                        &bufs.attn_out,
                        q,
                        &bufs.ffn_down,
                        &self.pipelines,
                        token_count,
                    );
                    enc.end_encoding();
                }
            }

            // Step 10: Residual add — hidden = hidden_state + o_proj_out
            // hidden_state still contains the pre-norm value (our residual).
            {
                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                ops::encode_residual_add(
                    &enc,
                    &self.pipelines.residual_add,
                    &bufs.hidden_state,
                    &bufs.ffn_down, // o_proj output was written here
                    &bufs.residual,
                    (token_count * h) as u32,
                );
                enc.end_encoding();
            }

            // Step 11: RMSNorm (post-attention norm)
            {
                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                ops::encode_rms_norm(
                    &enc,
                    &self.pipelines.rms_norm,
                    &bufs.residual,
                    &lw.post_attn_norm,
                    &bufs.norm_out,
                    h as u32,
                    token_count as u32,
                    eps,
                );
                enc.end_encoding();
            }

            // Steps 12-13: Gate and Up projections
            let norm_mat2 = MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;

            // Gate projection
            match &lw.gate_proj {
                WeightBuffer::Dense(buf) => {
                    let gate_weight_mat = MpsMatrix::from_buffer(buf, inter, h, row_bytes_h)
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    let gate_result_mat =
                        MpsMatrix::from_buffer(&bufs.ffn_gate, token_count, inter, row_bytes_inter)
                            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    lm.gate.as_ref().unwrap().encode(
                        &cmd_buf,
                        &norm_mat2,
                        &gate_weight_mat,
                        &gate_result_mat,
                    );
                }
                WeightBuffer::Quantized(q) => {
                    let enc = cmd_buf
                        .compute_encoder()
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    encode_polarquant_matmul(
                        &enc,
                        &bufs.norm_out,
                        q,
                        &bufs.ffn_gate,
                        &self.pipelines,
                        token_count,
                    );
                    enc.end_encoding();
                }
            }

            // Up projection
            match &lw.up_proj {
                WeightBuffer::Dense(buf) => {
                    let up_weight_mat = MpsMatrix::from_buffer(buf, inter, h, row_bytes_h)
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    let up_result_mat =
                        MpsMatrix::from_buffer(&bufs.ffn_up, token_count, inter, row_bytes_inter)
                            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    lm.up.as_ref().unwrap().encode(
                        &cmd_buf,
                        &norm_mat2,
                        &up_weight_mat,
                        &up_result_mat,
                    );
                }
                WeightBuffer::Quantized(q) => {
                    let enc = cmd_buf
                        .compute_encoder()
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    encode_polarquant_matmul(
                        &enc,
                        &bufs.norm_out,
                        q,
                        &bufs.ffn_up,
                        &self.pipelines,
                        token_count,
                    );
                    enc.end_encoding();
                }
            }

            // Step 14: SiLU gate
            {
                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                ops::encode_silu_gate(
                    &enc,
                    &self.pipelines.silu_gate,
                    &bufs.ffn_gate,
                    &bufs.ffn_up,
                    &bufs.ffn_gate, // in-place output into gate buffer
                    (token_count * inter) as u32,
                );
                enc.end_encoding();
            }

            // Step 15: Down projection
            match &lw.down_proj {
                WeightBuffer::Dense(buf) => {
                    let gate_out_mat =
                        MpsMatrix::from_buffer(&bufs.ffn_gate, token_count, inter, row_bytes_inter)
                            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    let down_weight_mat = MpsMatrix::from_buffer(buf, h, inter, row_bytes_inter)
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    let down_result_mat =
                        MpsMatrix::from_buffer(&bufs.ffn_down, token_count, h, row_bytes_h)
                            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    lm.down.as_ref().unwrap().encode(
                        &cmd_buf,
                        &gate_out_mat,
                        &down_weight_mat,
                        &down_result_mat,
                    );
                }
                WeightBuffer::Quantized(q) => {
                    let enc = cmd_buf
                        .compute_encoder()
                        .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                    encode_polarquant_matmul(
                        &enc,
                        &bufs.ffn_gate,
                        q,
                        &bufs.ffn_down,
                        &self.pipelines,
                        token_count,
                    );
                    enc.end_encoding();
                }
            }

            // Step 16: Residual add — hidden = residual + ffn_down
            {
                let enc = cmd_buf
                    .compute_encoder()
                    .map_err(|e| InferenceError::Runtime(e.to_string()))?;
                ops::encode_residual_add(
                    &enc,
                    &self.pipelines.residual_add,
                    &bufs.residual,
                    &bufs.ffn_down,
                    &bufs.hidden_state, // write back to hidden_state for next layer
                    (token_count * h) as u32,
                );
                enc.end_encoding();
            }
        }

        // Step 17: Final RMSNorm
        {
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            ops::encode_rms_norm(
                &enc,
                &self.pipelines.rms_norm,
                &bufs.hidden_state,
                &weights.final_norm,
                &bufs.norm_out,
                h as u32,
                token_count as u32,
                eps,
            );
            enc.end_encoding();
        }

        // Step 18: LM head matmul
        let row_bytes_h = h * 2;
        let row_bytes_vocab = vocab * 2;
        let final_norm_mat = MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        let lm_head_weight_mat = MpsMatrix::from_buffer(&weights.lm_head, vocab, h, row_bytes_h)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        let logits_mat = MpsMatrix::from_buffer(&bufs.logits, token_count, vocab, row_bytes_vocab)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        matmuls
            .lm_head
            .encode(&cmd_buf, &final_norm_mat, &lm_head_weight_mat, &logits_mat);

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

        // Advance sequence position.
        self.seq_pos += token_count;
        if enable_tq {
            if let Some(kv) = self.kv_cache.as_mut() {
                kv.advance_by(token_count);
            }
        } else if let Some(fp16_kv) = self.fp16_kv_cache.as_mut() {
            fp16_kv.seq_pos += token_count;
        }

        Ok(logits)
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
    pipelines: &super::ops::GpuPipelines,
    m: usize,
) {
    let (n, k) = weight.shape; // (out_features, in_features)

    let pipeline = match (weight.n_bits, m) {
        (4, 1) => &pipelines.polarquant_matvec_int4,
        (4, _) => &pipelines.polarquant_matmul_int4,
        (8, 1) => &pipelines.polarquant_matvec_int8,
        (8, _) => &pipelines.polarquant_matmul_int8,
        _ => panic!("unsupported n_bits: {}", weight.n_bits),
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
}

// ── InferenceEngine implementation ──────────────────────────────

impl InferenceEngine for GpuInference {
    fn load(&mut self, artifacts: &dyn Any) -> Result<(), InferenceError> {
        let gpu_artifacts = artifacts
            .downcast_ref::<GpuArtifacts<'_>>()
            .ok_or_else(|| {
                InferenceError::Runtime("GpuInference::load expects GpuArtifacts".into())
            })?;

        self.config = gpu_artifacts.config.clone();

        // Load weights into Metal buffers.
        let weights = GpuWeights::load(
            &self.device,
            gpu_artifacts.weights,
            self.config.force_cpu_dequant,
        )
        .map_err(|e| InferenceError::Runtime(e.to_string()))?;

        let mc = &weights.config;
        self.model_config = Some(mc.clone());

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
        let decode_cache = Self::build_matmul_cache(&self.device, mc, &weights, 1)
            .map_err(|e| InferenceError::Runtime(e.to_string()))?;
        self.decode_matmuls = Some(decode_cache);

        // Initialize KV cache.
        if self.config.enable_turboquant {
            let tq_config = TurboQuantGpuConfig {
                n_bits: self.config.n_bits,
                num_kv_heads: mc.num_key_value_heads,
                head_dim: mc.head_dim,
                max_seq_len: self.config.max_seq_len,
                num_layers: mc.num_hidden_layers,
                rotation_seed: self.config.rotation_seed,
                outlier: None,
            };
            let tq_model = GpuTurboQuantModel::new(&self.device, tq_config.clone())
                .map_err(|e| InferenceError::Runtime(e.to_string()))?;
            let kv_cache = GpuKvCache::new(&self.device, &tq_config)
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
