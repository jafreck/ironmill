//! GPU inference engine implementing the [`InferenceEngine`] trait.
//!
//! Runs the full LLaMA-family transformer decode pipeline on Metal:
//!   - MPS `MPSMatrixMultiplication` for linear projections
//!   - Custom Metal compute shaders for RMSNorm, RoPE, SiLU, residual add,
//!     embedding lookup, and attention
//!   - Optional TurboQuant INT8 KV cache compression

use std::time::Instant;

use half::f16;
use ironmill_metal_sys::{
    CommandBufferStatus, ComputeEncoder, MetalBuffer, MetalDevice, MpsMatrix, MpsMatrixMultiply,
    MpsMatrixMultiplyConfig, StorageMode,
};
use mil_rs::weights::{ModelConfig, WeightProvider};

use super::config::MetalConfig;
use super::error::MetalError;
use super::mla::{MlaConfig, MlaKvCache};
use super::ops;
use super::ops::LinearKernelKind;
use super::turboquant::{MetalKvCache, MetalTurboQuantModel, OutlierConfig, TurboQuantMetalConfig};
use super::weights::{
    AffineQuantizedWeight, LayerWeights, MetalWeights, QuantizedWeight, WeightBuffer,
};
use crate::calibration::ActivationHook;
use crate::engine::{InferenceEngine, InferenceError};
use crate::types::Logits;
use ironmill_core::model_info::ModelInfo;

// ── Matmul tile dimensions — must match Metal shader constants ──
const MATMUL_TM_TILE: usize = 64;
const MATMUL_TN_TILE: usize = 64;
const MATMUL_THREADS_PER_TG: usize = 256;

// ── Public artifacts type for load() ────────────────────────────

/// Artifacts passed to [`MetalInference::load`] via the type-erased
/// [`InferenceEngine`] interface.
pub struct MetalArtifacts<'a> {
    /// Weight provider for loading model tensors.
    pub weights: &'a dyn WeightProvider,
    /// Metal backend configuration.
    pub config: MetalConfig,
}

// ── Helper structs ──────────────────────────────────────────────

/// FP16 KV cache (when TurboQuant is disabled).
struct Fp16KvCache {
    /// K caches per buffer slot: `[num_kv_heads × effective_max_seq × head_dim]` FP16.
    k_caches: Vec<MetalBuffer>,
    /// V caches per buffer slot.
    v_caches: Vec<MetalBuffer>,
    seq_pos: usize,
    /// Global max sequence length (for full-attention layers).
    _max_seq_len: usize,
    /// CLA anchor layers. When set, only anchor layers have physical buffers.
    anchor_layers: Option<Vec<usize>>,
    /// Per-buffer sliding window sizes. `0` = full attention for that buffer.
    _window_sizes: Vec<usize>,
}

impl Fp16KvCache {
    fn new(
        device: &MetalDevice,
        num_layers: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        anchor_layers: Option<Vec<usize>>,
        layer_window_sizes: &[usize],
    ) -> Result<Self, MetalError> {
        let num_buffers = if let Some(ref anchors) = anchor_layers {
            anchors.len()
        } else {
            num_layers
        };

        let per_buffer_window: Vec<usize> = (0..num_buffers)
            .map(|buf_idx| {
                let layer = if let Some(ref anchors) = anchor_layers {
                    anchors[buf_idx]
                } else {
                    buf_idx
                };
                layer_window_sizes.get(layer).copied().unwrap_or(0)
            })
            .collect();

        let mut k_caches = Vec::with_capacity(num_buffers);
        let mut v_caches = Vec::with_capacity(num_buffers);
        for ws in &per_buffer_window {
            let effective_seq = if *ws > 0 { *ws } else { max_seq_len };
            let size_bytes = num_kv_heads * effective_seq * head_dim * 2; // FP16
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
            _max_seq_len: max_seq_len,
            anchor_layers,
            _window_sizes: per_buffer_window,
        })
    }

    /// Map a layer index to its physical buffer index.
    fn kv_buffer_for_layer(&self, layer: usize) -> usize {
        match self.anchor_layers {
            Some(ref anchors) => match anchors.binary_search(&layer) {
                Ok(idx) => idx,
                Err(idx) => idx.saturating_sub(1),
            },
            None => layer,
        }
    }

    /// Returns true if `layer` is an anchor layer (or CLA is not configured).
    fn _is_anchor_layer(&self, layer: usize) -> bool {
        match self.anchor_layers {
            Some(ref anchors) => anchors.binary_search(&layer).is_ok(),
            None => true,
        }
    }

    fn layer_caches(&self, layer: usize) -> (&MetalBuffer, &MetalBuffer) {
        let idx = self.kv_buffer_for_layer(layer);
        (&self.k_caches[idx], &self.v_caches[idx])
    }

    /// Write position for ring buffer. For SWA layers, wraps at window_size.
    fn _ring_pos(&self, layer: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        let ws = self._window_sizes.get(idx).copied().unwrap_or(0);
        if ws > 0 {
            self.seq_pos % ws
        } else {
            self.seq_pos
        }
    }

    /// Effective max_seq_len for a layer's buffer.
    fn _effective_max_seq_len(&self, layer: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        let ws = self._window_sizes.get(idx).copied().unwrap_or(0);
        if ws > 0 { ws } else { self._max_seq_len }
    }

    /// Effective seq_len for attention (capped at window_size for SWA layers).
    fn _effective_seq_len(&self, layer: usize, total_tokens: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        let ws = self._window_sizes.get(idx).copied().unwrap_or(0);
        if ws > 0 {
            total_tokens.min(ws)
        } else {
            total_tokens
        }
    }

    /// Window size for a layer. 0 = full attention.
    fn _window_size_for_layer(&self, layer: usize) -> usize {
        let idx = self.kv_buffer_for_layer(layer);
        self._window_sizes.get(idx).copied().unwrap_or(0)
    }

    fn reset(&mut self) {
        self.seq_pos = 0;
    }

    fn _seq_pos(&self) -> usize {
        self.seq_pos
    }

    fn truncate_to(&mut self, pos: usize) {
        assert!(pos <= self.seq_pos);
        self.seq_pos = pos;
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
    /// Second token IDs buffer for prefill pipelining — allows encoding
    /// the next chunk while the previous command buffer is still executing.
    _token_ids_buf_b: MetalBuffer,
    /// Current maximum token capacity of these buffers.
    capacity: usize,
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

        let alloc_private = |size_elems: usize| -> Result<MetalBuffer, MetalError> {
            // GPU-only buffers use Private storage for ~10-15% bandwidth
            // improvement — no CPU coherency overhead.
            let bytes = (size_elems * 2).max(16);
            device
                .create_buffer(bytes, StorageMode::Private)
                .map_err(MetalError::Metal)
        };

        let alloc_shared = |size_elems: usize| -> Result<MetalBuffer, MetalError> {
            // Buffers that need CPU read/write use Shared storage.
            let bytes = (size_elems * 2).max(16);
            device
                .create_buffer(bytes, StorageMode::Shared)
                .map_err(MetalError::Metal)
        };

        Ok(Self {
            hidden_state: alloc_private(max_tokens * h)?,
            attn_out: alloc_private(max_tokens * nh * hd)?,
            q_proj: alloc_private(max_tokens * nh * hd)?,
            k_proj: alloc_private(max_tokens * nkv * hd)?,
            v_proj: alloc_private(max_tokens * nkv * hd)?,
            ffn_gate: alloc_private(max_tokens * inter)?,
            ffn_up: alloc_private(max_tokens * inter)?,
            ffn_down: alloc_private(max_tokens * h)?,
            residual: alloc_private(max_tokens * h)?,
            norm_out: alloc_shared(max_tokens * h)?, // CPU reads in calibration
            logits: alloc_shared(max_tokens * vocab)?, // CPU reads logits
            token_ids_buf: device
                .create_buffer((max_tokens * 4).max(16), StorageMode::Shared) // CPU writes token IDs
                .map_err(MetalError::Metal)?,
            _token_ids_buf_b: device
                .create_buffer((max_tokens * 4).max(16), StorageMode::Shared)
                .map_err(MetalError::Metal)?,
            capacity: max_tokens,
        })
    }

    /// Grow buffers if `needed` exceeds current capacity. No-op otherwise.
    fn ensure_capacity(
        &mut self,
        device: &MetalDevice,
        needed: usize,
        mc: &ModelConfig,
    ) -> Result<(), MetalError> {
        if needed <= self.capacity {
            return Ok(());
        }
        *self = Self::allocate(device, needed, mc)?;
        Ok(())
    }
}

/// Per-projection matmul state: Dense weights use MPS matrix multiplication
/// while Quantized weights use a custom compute-shader path and carry no MPS
/// object.
enum ProjectionMatmul {
    #[allow(dead_code)]
    Dense(MpsMatrixMultiply),
    Quantized,
}

impl ProjectionMatmul {
    /// Returns the MPS matmul for a Dense projection, or `None` for Quantized.
    #[allow(dead_code)]
    fn dense(&self) -> Option<&MpsMatrixMultiply> {
        match self {
            Self::Dense(m) => Some(m),
            Self::Quantized => None,
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
    /// MLA configuration (set when the model uses Multi-Head Latent Attention).
    mla_config: Option<MlaConfig>,
    /// MLA compressed KV cache (used instead of fp16_kv_cache for MLA models).
    mla_kv_cache: Option<MlaKvCache>,
    /// Pre-allocated buffer for FP16 logits readback.
    logits_fp16_buf: Vec<u8>,
    /// Pre-allocated buffer for serializing token IDs to GPU.
    token_bytes_buf: Vec<u8>,
    seq_pos: usize,
    /// Cached model info, populated during `load()`.
    model_info: Option<ModelInfo>,
}

impl MetalInference {
    /// Access compiled pipelines — returns `NotLoaded` if `load()` hasn't been called yet.
    fn pipelines(&self) -> Result<&super::ops::MetalPipelines, InferenceError> {
        self.pipelines.as_ref().ok_or(InferenceError::NotLoaded)
    }

    /// Create a new GPU inference engine (device + queue + shader pipelines).
    pub fn new(config: MetalConfig) -> Result<Self, MetalError> {
        config
            .validate()
            .map_err(|e| MetalError::Config(e.to_string()))?;
        if config.use_fa2_prefill {
            eprintln!(
                "Warning: FA2 prefill is not yet implemented; \
                 falling back to standard attention."
            );
        }
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
            mla_config: None,
            mla_kv_cache: None,
            logits_fp16_buf: Vec::new(),
            token_bytes_buf: Vec::new(),
            seq_pos: 0,
            model_info: None,
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

        let mut weights = MetalWeights::load(&self.device, provider, self.config.force_cpu_dequant)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        let mc = weights.config.clone();
        self.model_config = Some(mc.clone());
        self.model_info = Some(ModelInfo::from_config(&mc));

        // Pre-allocate logits readback buffer (vocab × 2 bytes for FP16).
        self.logits_fp16_buf.resize(mc.vocab_size * 2, 0);

        // Compile Metal shader pipelines with the model's head_dim.
        let pipelines = super::ops::MetalPipelines::compile(&self.device, mc.head_dim)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.pipelines = Some(pipelines);

        let bufs = IntermediateBuffers::allocate(&self.device, 1, &mc)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.intermediate_buffers = Some(bufs);

        let (cos, sin) = Self::build_rope_cache(
            &self.device,
            mc.head_dim,
            self.config.max_seq_len,
            mc.rope_theta,
        )
        .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.rope_cos = Some(cos);
        self.rope_sin = Some(sin);

        let decode_cache_t1 = Self::build_matmul_cache(&self.device, &mc, &weights, 1)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.decode_matmuls_t1 = Some(decode_cache_t1);
        self.decode_matmuls = None;

        // Resolve CLA anchor layers: explicit config takes priority, then
        // fall back to model metadata. Validate against num_layers.
        let cla_anchors = self
            .config
            .cla_config
            .as_ref()
            .map(|c| c.anchor_layers.clone())
            .or_else(|| mc.cla_anchor_layers());
        if let Some(ref anchors) = cla_anchors {
            let cla = super::config::ClaConfig {
                anchor_layers: anchors.clone(),
            };
            cla.validate(mc.num_hidden_layers)?;
        }
        // Back-fill cla_config so is_anchor checks during inference see the
        // metadata-derived anchors, not the absent user config.
        if self.config.cla_config.is_none() {
            if let Some(ref anchors) = cla_anchors {
                self.config.cla_config = Some(super::config::ClaConfig {
                    anchor_layers: anchors.clone(),
                });
            }
        }

        // ── MLA detection and weight absorption ─────────────────
        let mla_cfg = mc.mla_config();
        if let Some(ref mla) = mla_cfg {
            // Perform weight absorption: fuse W_uk into Q and W_uv into O
            // at load time so inference works directly on compressed latents.
            absorb_mla_weights(&self.device, &mut weights, mla, mc.hidden_size, provider)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;

            // Create MLA compressed KV cache.
            let mla_cache = MlaKvCache::new(
                &self.device,
                mla,
                mc.num_hidden_layers,
                self.config.max_seq_len,
            )
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.mla_kv_cache = Some(mla_cache);
        } else {
            self.mla_kv_cache = None;
        }
        self.mla_config = mla_cfg;

        // Resolve sliding window config: explicit config takes priority,
        // then fall back to model metadata (sliding_window + max_window_layers).
        if self.config.sliding_window.is_none() {
            if let Some(ws) = mc.sliding_window() {
                let mwl = mc.max_window_layers().unwrap_or(mc.num_hidden_layers);
                self.config.sliding_window = Some(super::config::SlidingWindowConfig {
                    window_size: ws,
                    max_window_layers: mwl,
                });
            }
        }

        // Build per-layer window sizes for buffer allocation.
        let layer_window_sizes: Vec<usize> = (0..mc.num_hidden_layers)
            .map(|l| self.config.layer_window_size(l))
            .collect();

        if self.config.enable_turboquant {
            // Algorithm selection:
            // - b >= 4: Algorithm 1 (full b-bit codebook for K and V, no QJL).
            //   Standard path only — outlier separation not needed.
            // - b < 4: Algorithm 2 ((b-1)-bit K codebook + QJL). Outlier
            //   channel strategy (§4.3) auto-detects high-energy channels
            //   for independent quantization per group.
            let use_qjl = self.config.n_bits < 4;
            let outlier_cfg: Option<OutlierConfig> = if use_qjl {
                let mut weight_data: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
                for layer in 0..mc.num_hidden_layers {
                    let prefix = format!("model.layers.{layer}");
                    let k_name = format!("{prefix}.self_attn.k_proj.weight");
                    let v_name = format!("{prefix}.self_attn.v_proj.weight");
                    if let (Ok(k_t), Ok(v_t)) = (provider.tensor(&k_name), provider.tensor(&v_name))
                    {
                        weight_data.push((k_t.data.to_vec(), v_t.data.to_vec()));
                    }
                }
                if !weight_data.is_empty() {
                    let refs: Vec<(&[u8], &[u8])> = weight_data
                        .iter()
                        .map(|(k, v)| (k.as_slice(), v.as_slice()))
                        .collect();
                    let out_features = mc.num_key_value_heads * mc.head_dim;
                    let n_outlier = mc.head_dim / 4;
                    Some(OutlierConfig::from_weight_norms(
                        &refs,
                        out_features,
                        mc.head_dim,
                        n_outlier,
                        self.config.n_bits,
                        self.config.n_bits,
                    ))
                } else {
                    None
                }
            } else {
                None
            };

            let tq_config = TurboQuantMetalConfig {
                n_bits: self.config.n_bits,
                num_kv_heads: mc.num_key_value_heads,
                head_dim: mc.head_dim,
                max_seq_len: self.config.max_seq_len,
                num_layers: mc.num_hidden_layers,
                rotation_seed: self.config.rotation_seed,
                outlier: outlier_cfg,
                anchor_layers: cla_anchors.clone(),
                window_sizes: layer_window_sizes.clone(),
            };
            let tq_model = MetalTurboQuantModel::new(&self.device, tq_config.clone())
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let kv_cache = MetalKvCache::new(&self.device, &tq_config)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
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
                cla_anchors.clone(),
                &layer_window_sizes,
            )
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.fp16_kv_cache = Some(fp16_kv);
            self.turboquant = None;
            self.kv_cache = None;
        }

        self.weights = Some(weights);
        self.seq_pos = 0;
        Ok(())
    }

    /// Load model using the JIT path: weights are loaded, transformed, and
    /// uploaded to GPU in a single streaming pass.
    ///
    /// No compilation step, no MIL IR, no `.ironml-gpu` bundle.
    pub fn load_jit(
        _config: MetalConfig,
        _provider: &dyn mil_rs::weights::WeightProvider,
        _transforms: &crate::jit::TransformPipeline,
    ) -> Result<Self, crate::engine::InferenceError> {
        Err(crate::engine::InferenceError::Other(anyhow::anyhow!(
            "JIT loading not yet implemented"
        )))
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
                // AffineQuantized uses fused compute kernels — no MPS matmul needed.
                WeightBuffer::AffineQuantized(_) => Ok(ProjectionMatmul::Quantized),
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
        self.run_pipeline_inner(token_ids, true)?;

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

        Ok(all_logits)
    }

    /// Run the transformer decode pipeline for `token_count` tokens.
    /// Returns logits for the last token position.
    fn run_pipeline(&mut self, token_ids: &[u32]) -> Result<Logits, InferenceError> {
        self.run_pipeline_inner(token_ids, false)
    }

    /// Run the pipeline without reading logits — used for non-last prefill
    /// chunks where logits are immediately discarded.
    fn run_pipeline_no_logits(&mut self, token_ids: &[u32]) -> Result<(), InferenceError> {
        self.run_pipeline_inner(token_ids, true)?;
        Ok(())
    }

    fn run_pipeline_inner(
        &mut self,
        token_ids: &[u32],
        skip_logits: bool,
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
            .ensure_capacity(&self.device, token_ids.len(), &mc)
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
                let cache = Self::build_matmul_cache(&self.device, &mc, weights, token_count)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                self.decode_matmuls = Some(cache);
            }
            self.decode_matmuls
                .as_ref()
                .ok_or_else(|| InferenceError::runtime("decode_matmuls not populated"))?
        };

        // Write token IDs to GPU buffer (reuse persistent buffer).
        self.token_bytes_buf.clear();
        self.token_bytes_buf
            .extend(token_ids.iter().flat_map(|t| t.to_le_bytes()));
        bufs.token_ids_buf
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
        {
            let lw0 = &weights.layers[0];
            let pipelines = self.pipelines()?;
            enc.set_pipeline(&pipelines.fused_embedding_norm);
            enc.set_buffer(&bufs.token_ids_buf, 0, 0);
            enc.set_buffer(&weights.embedding, 0, 1);
            enc.set_buffer(&lw0.input_norm, 0, 2);
            enc.set_buffer(&bufs.norm_out, 0, 3);
            enc.set_buffer(&bufs.hidden_state, 0, 4);
            enc.set_bytes(&(h as u32).to_le_bytes(), 5);
            enc.set_bytes(&(token_count as u32).to_le_bytes(), 6);
            enc.set_bytes(&(vocab as u32).to_le_bytes(), 7);
            enc.set_bytes(&eps.to_le_bytes(), 8);
            let tg_size = h.min(1024);
            enc.dispatch_threadgroups((token_count, 1, 1), (tg_size, 1, 1));
        }
        enc.memory_barrier_buffers();

        // Per-layer processing.
        //
        // norm_out already contains the first layer's input-norm result
        // (from the fused embedding+norm above). Subsequent layers receive
        // their input norm from the previous layer's fused end-of-layer dispatch.

        for layer_idx in 0..mc.num_hidden_layers {
            let lw = &weights.layers[layer_idx];
            let lm = &matmuls.layer_matmuls[layer_idx];
            let pipelines = self.pipelines()?;

            // norm_out already contains the input-norm result:
            //   • layer 0: computed by the standalone dispatch above
            //   • layer 1+: produced by the previous layer's fused end-of-layer kernel

            // Steps 3-5: Q/K/V projections — dispatch by weight type.
            // These are independent (all read norm_out, write to different buffers).
            let row_bytes_h = h * 2; // FP16
            let row_bytes_qo = (mc.num_attention_heads * mc.head_dim) * 2;
            let row_bytes_kv = (mc.num_kv_heads() * mc.head_dim) * 2;

            let norm_mat = MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;

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
                    &enc,
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
            enc.memory_barrier_buffers();

            // Step 6: QK normalization (Qwen3) + RoPE
            encode_qk_norm_and_rope(
                &enc,
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
            enc.memory_barrier_buffers();

            // Steps 7-8: KV cache write + attention
            let is_anchor = self
                .config
                .cla_config
                .as_ref()
                .is_none_or(|cla| cla.is_anchor(layer_idx));
            encode_kv_cache_and_attention(
                &enc,
                pipelines,
                bufs,
                self.turboquant.as_ref(),
                self.kv_cache.as_ref(),
                self.fp16_kv_cache.as_ref(),
                self.config.max_seq_len,
                self.config.n_bits as usize,
                layer_idx,
                seq_pos,
                token_count,
                nh,
                nkv,
                hd,
                enable_tq,
                self.config.use_fa2_prefill,
                is_anchor,
                self.config.layer_window_size(layer_idx),
            )?;
            enc.memory_barrier_buffers();

            // Step 9: Output projection
            let attn_mat = MpsMatrix::from_buffer(
                &bufs.attn_out,
                token_count,
                mc.num_attention_heads * mc.head_dim,
                row_bytes_qo,
            )
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
            encode_projection(
                &enc,
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
            enc.memory_barrier_buffers();

            // Step 10-11 (fused): Residual add + post-attention RMSNorm
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
            enc.memory_barrier_buffers();

            // Steps 12-15: FFN block (gate + up + SiLU + down)
            encode_ffn_block(&enc, pipelines, bufs, lw, lm, h, inter, token_count)?;
            enc.memory_barrier_buffers();

            // Step 16: Residual add + next layer's input norm (or standalone for last layer)
            let next_norm = if layer_idx + 1 < mc.num_hidden_layers {
                Some(&weights.layers[layer_idx + 1].input_norm)
            } else {
                None
            };
            encode_end_of_layer_residual(&enc, pipelines, bufs, next_norm, h, token_count, eps)?;
            enc.memory_barrier_buffers();
        }

        // Step 17: Final RMSNorm
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
        enc.memory_barrier_buffers();

        // Step 18: LM head — select kernel variant based on prefill/decode phase.
        // Prefer custom matvec/matmul when packed weights exist; fall back
        // to MPS only when there is no packed buffer.
        let row_bytes_h = h * 2;
        let row_bytes_vocab = vocab * 2;
        if let Some(ref packed_buf) = weights.lm_head_packed {
            let pipelines = self.pipelines()?;
            let lm_kind = LinearKernelKind::for_token_count(token_count);
            let pipeline = pipelines.dense_linear_pipeline(lm_kind);
            if lm_kind.is_decode() {
                ops::encode_matvec(
                    &enc,
                    pipeline,
                    &bufs.norm_out,
                    packed_buf,
                    &bufs.logits,
                    vocab as u32,
                    h as u32,
                );
            } else {
                ops::encode_matmul(
                    &enc,
                    pipeline,
                    &bufs.norm_out,
                    packed_buf,
                    &bufs.logits,
                    token_count as u32,
                    vocab as u32,
                    h as u32,
                );
            }
            enc.end_encoding();
        } else {
            // MPS fallback — end shared encoder so MPS can create its own.
            enc.end_encoding();
            let final_norm_mat =
                MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let lm_head_weight_mat =
                MpsMatrix::from_buffer(&weights.lm_head, vocab, h, row_bytes_h)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let logits_mat =
                MpsMatrix::from_buffer(&bufs.logits, token_count, vocab, row_bytes_vocab)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
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

            self.logits_fp16_buf
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f16::from_bits(bits).to_f32()
                })
                .collect()
        };

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
            .ensure_capacity(&self.device, token_ids.len(), &mc)
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
            let cache = Self::build_matmul_cache(&self.device, &mc, weights, token_count)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.decode_matmuls = Some(cache);
        }
        let matmuls = self
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
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                readback_time_ms += rb_start.elapsed().as_secs_f64() * 1000.0;
                layer_callback(layer_idx, "attn_norm", readback);
            }

            // Create new command buffer for this layer's attention block.
            let cmd_buf = self
                .queue
                .command_buffer()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let pipelines = self.pipelines()?;

            // Steps 3-5: Q/K/V projections
            let row_bytes_h = h * 2;
            let row_bytes_qo = (mc.num_attention_heads * mc.head_dim) * 2;
            let row_bytes_kv = (mc.num_kv_heads() * mc.head_dim) * 2;

            let norm_mat = MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;

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
                    &enc,
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
            enc.memory_barrier_buffers();

            // Step 6: QK normalization (Qwen3) + RoPE
            encode_qk_norm_and_rope(
                &enc,
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
            enc.memory_barrier_buffers();

            // Steps 7-8: KV cache write + attention
            let is_anchor = self
                .config
                .cla_config
                .as_ref()
                .is_none_or(|cla| cla.is_anchor(layer_idx));
            encode_kv_cache_and_attention(
                &enc,
                pipelines,
                bufs,
                self.turboquant.as_ref(),
                self.kv_cache.as_ref(),
                self.fp16_kv_cache.as_ref(),
                self.config.max_seq_len,
                self.config.n_bits as usize,
                layer_idx,
                seq_pos,
                token_count,
                nh,
                nkv,
                hd,
                enable_tq,
                self.config.use_fa2_prefill,
                is_anchor,
                self.config.layer_window_size(layer_idx),
            )?;
            enc.memory_barrier_buffers();

            // Step 9: Output projection
            let attn_mat = MpsMatrix::from_buffer(
                &bufs.attn_out,
                token_count,
                mc.num_attention_heads * mc.head_dim,
                row_bytes_qo,
            )
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
            encode_projection(
                &enc,
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
            enc.memory_barrier_buffers();

            // Step 10-11 (fused): Residual add + post-attention RMSNorm
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

            // Steps 12-15: FFN block (gate + up + SiLU + down)
            encode_ffn_block(&enc, pipelines, bufs, lw, lm, h, inter, token_count)?;
            enc.memory_barrier_buffers();

            // Step 16: Residual add + next layer's input norm (or standalone for last layer)
            let next_norm = if layer_idx + 1 < mc.num_hidden_layers {
                Some(&weights.layers[layer_idx + 1].input_norm)
            } else {
                None
            };
            encode_end_of_layer_residual(&enc, pipelines, bufs, next_norm, h, token_count, eps)?;
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

        // Step 18: LM head — select kernel variant based on prefill/decode phase.
        // Prefer custom matvec/matmul when packed weights exist; fall back
        // to MPS only when there is no packed buffer.
        let row_bytes_h = h * 2;
        let row_bytes_vocab = vocab * 2;
        if let Some(ref packed_buf) = weights.lm_head_packed {
            let enc = cmd_buf
                .compute_encoder()
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let pipelines = self.pipelines()?;
            let lm_kind = LinearKernelKind::for_token_count(token_count);
            let pipeline = pipelines.dense_linear_pipeline(lm_kind);
            if lm_kind.is_decode() {
                ops::encode_matvec(
                    &enc,
                    pipeline,
                    &bufs.norm_out,
                    packed_buf,
                    &bufs.logits,
                    vocab as u32,
                    h as u32,
                );
            } else {
                ops::encode_matmul(
                    &enc,
                    pipeline,
                    &bufs.norm_out,
                    packed_buf,
                    &bufs.logits,
                    token_count as u32,
                    vocab as u32,
                    h as u32,
                );
            }
            enc.end_encoding();
        } else {
            // MPS fallback — only reached when packed weights are absent.
            let final_norm_mat =
                MpsMatrix::from_buffer(&bufs.norm_out, token_count, h, row_bytes_h)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let lm_head_weight_mat =
                MpsMatrix::from_buffer(&weights.lm_head, vocab, h, row_bytes_h)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let logits_mat =
                MpsMatrix::from_buffer(&bufs.logits, token_count, vocab, row_bytes_vocab)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
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

    /// Load model artifacts into this engine.
    ///
    /// This is a typed method on the backend struct rather than on the
    /// [`InferenceEngine`] trait, keeping the trait object-safe (§4.3).
    pub fn load(&mut self, artifacts: &MetalArtifacts<'_>) -> Result<(), InferenceError> {
        self.config = artifacts.config.clone();

        // Load weights into Metal buffers.
        let mut weights = MetalWeights::load(
            &self.device,
            artifacts.weights,
            self.config.force_cpu_dequant,
        )
        .map_err(|e| InferenceError::runtime(e.to_string()))?;

        let mc = weights.config.clone();
        self.model_config = Some(mc.clone());
        self.model_info = Some(ModelInfo::from_config(&mc));

        // Pre-allocate logits readback buffer (vocab × 2 bytes for FP16).
        self.logits_fp16_buf.resize(mc.vocab_size * 2, 0);

        // Compile Metal shader pipelines with the model's head_dim so
        // shared memory is sized exactly via #define HEAD_DIM.
        let pipelines = super::ops::MetalPipelines::compile(&self.device, mc.head_dim)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.pipelines = Some(pipelines);

        // Allocate intermediate buffers (start at 1 token; run_pipeline_inner
        // grows them on demand for larger prefill batches).
        let bufs = IntermediateBuffers::allocate(&self.device, 1, &mc)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.intermediate_buffers = Some(bufs);

        // Build RoPE cos/sin caches.
        let (cos, sin) = Self::build_rope_cache(
            &self.device,
            mc.head_dim,
            self.config.max_seq_len,
            mc.rope_theta,
        )
        .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.rope_cos = Some(cos);
        self.rope_sin = Some(sin);

        // Build MPS matmul cache for single-token decode.
        let decode_cache_t1 = Self::build_matmul_cache(&self.device, &mc, &weights, 1)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.decode_matmuls_t1 = Some(decode_cache_t1);
        self.decode_matmuls = None;

        // Resolve CLA anchor layers.
        let cla_anchors = self
            .config
            .cla_config
            .as_ref()
            .map(|c| c.anchor_layers.clone())
            .or_else(|| mc.cla_anchor_layers());
        if let Some(ref anchors) = cla_anchors {
            let cla = super::config::ClaConfig {
                anchor_layers: anchors.clone(),
            };
            cla.validate(mc.num_hidden_layers)?;
        }
        // Back-fill cla_config so is_anchor checks during inference see the
        // metadata-derived anchors, not the absent user config.
        if self.config.cla_config.is_none() {
            if let Some(ref anchors) = cla_anchors {
                self.config.cla_config = Some(super::config::ClaConfig {
                    anchor_layers: anchors.clone(),
                });
            }
        }

        // ── MLA detection and weight absorption ─────────────────
        let mla_cfg = mc.mla_config();
        if let Some(ref mla) = mla_cfg {
            absorb_mla_weights(
                &self.device,
                &mut weights,
                mla,
                mc.hidden_size,
                artifacts.weights,
            )
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

            let mla_cache = MlaKvCache::new(
                &self.device,
                mla,
                mc.num_hidden_layers,
                self.config.max_seq_len,
            )
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.mla_kv_cache = Some(mla_cache);
        } else {
            self.mla_kv_cache = None;
        }
        self.mla_config = mla_cfg;

        // Resolve sliding window config from model metadata.
        if self.config.sliding_window.is_none() {
            if let Some(ws) = mc.sliding_window() {
                let mwl = mc.max_window_layers().unwrap_or(mc.num_hidden_layers);
                self.config.sliding_window = Some(super::config::SlidingWindowConfig {
                    window_size: ws,
                    max_window_layers: mwl,
                });
            }
        }

        let layer_window_sizes: Vec<usize> = (0..mc.num_hidden_layers)
            .map(|l| self.config.layer_window_size(l))
            .collect();

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
                anchor_layers: cla_anchors.clone(),
                window_sizes: layer_window_sizes.clone(),
            };
            let tq_model = MetalTurboQuantModel::new(&self.device, tq_config.clone())
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let kv_cache = MetalKvCache::new(&self.device, &tq_config)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
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
                cla_anchors.clone(),
                &layer_window_sizes,
            )
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.fp16_kv_cache = Some(fp16_kv);
            self.turboquant = None;
            self.kv_cache = None;
        }

        self.weights = Some(weights);
        self.seq_pos = 0;
        Ok(())
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
/// Automatically selects the optimal kernel variant based on `token_count`:
/// - **Decode (token_count == 1):** memory-bandwidth-optimized matvec kernels.
/// - **Prefill (token_count > 1):** compute-optimized batched matmul kernels.
///
/// This applies uniformly across all weight representations (Dense, PolarQuant,
/// AffineQuantized).
#[allow(clippy::too_many_arguments)]
fn encode_projection(
    enc: &ComputeEncoder,
    input_buf: &MetalBuffer,
    _input_mat: &MpsMatrix,
    weight: &WeightBuffer,
    output_buf: &MetalBuffer,
    _matmul: &ProjectionMatmul,
    pipelines: &super::ops::MetalPipelines,
    token_count: usize,
    out_features: usize,
    in_features: usize,
    _row_bytes_in: usize,
    _row_bytes_out: usize,
) -> Result<(), InferenceError> {
    let kernel_kind = LinearKernelKind::for_token_count(token_count);

    match weight {
        WeightBuffer::Dense { buf: _, packed } => {
            if let Some(packed_buf) = packed {
                let pipeline = pipelines.dense_linear_pipeline(kernel_kind);
                if kernel_kind.is_decode() {
                    ops::encode_matvec(
                        enc,
                        pipeline,
                        input_buf,
                        packed_buf,
                        output_buf,
                        out_features as u32,
                        in_features as u32,
                    );
                } else {
                    ops::encode_matmul(
                        enc,
                        pipeline,
                        input_buf,
                        packed_buf,
                        output_buf,
                        token_count as u32,
                        out_features as u32,
                        in_features as u32,
                    );
                }
                return Ok(());
            }
            panic!("MPS dense fallback not supported in single-encoder mode");
        }
        WeightBuffer::Quantized(q) => {
            encode_polarquant_projection(
                enc,
                input_buf,
                q,
                output_buf,
                pipelines,
                kernel_kind,
                token_count,
            )?;
        }
        WeightBuffer::AffineQuantized(aq) => {
            encode_affine_projection(
                enc,
                input_buf,
                aq,
                output_buf,
                pipelines,
                kernel_kind,
                token_count,
            )?;
        }
    }
    Ok(())
}

// ── PolarQuant kernel dispatch ──────────────────────────────────

/// Encode a PolarQuant quantized projection via compute kernel.
///
/// Dispatches to the correct Metal kernel based on `n_bits` and kernel kind:
/// - Matvec (`LinearKernelKind::Matvec`): one threadgroup per output row
/// - Matmul (`LinearKernelKind::Matmul`): tiled matmul with `token_count` rows
fn encode_polarquant_projection(
    encoder: &ironmill_metal_sys::ComputeEncoder,
    input: &MetalBuffer,
    weight: &QuantizedWeight,
    output: &MetalBuffer,
    pipelines: &super::ops::MetalPipelines,
    kind: LinearKernelKind,
    token_count: usize,
) -> Result<(), InferenceError> {
    let (n, k) = weight.shape; // (out_features, in_features)

    let pipeline = pipelines
        .polarquant_pipeline(weight.n_bits.into(), kind)
        .ok_or_else(|| InferenceError::runtime(format!("unsupported n_bits: {}", weight.n_bits)))?;

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&weight.indices, 0, 1);
    encoder.set_buffer(&weight.lut, 0, 2);
    encoder.set_buffer(&weight.norms, 0, 3);
    encoder.set_buffer(output, 0, 4);

    if kind.is_decode() {
        // matvec: one threadgroup per output row
        encoder.set_bytes(&(n as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 6);
        let threads_per_group = 32; // SIMD width
        encoder.dispatch_threadgroups((n, 1, 1), (threads_per_group, 1, 1));
    } else {
        // matmul: simdgroup tiled (column-major dispatch)
        encoder.set_bytes(&(token_count as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(n as u32).to_le_bytes(), 6);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 7);
        encoder.dispatch_threadgroups(
            (
                token_count.div_ceil(MATMUL_TM_TILE),
                n.div_ceil(MATMUL_TN_TILE),
                1,
            ),
            (MATMUL_THREADS_PER_TG, 1, 1),
        );
    }
    Ok(())
}

/// Encode a fused affine quantized projection via compute kernel.
///
/// Dispatches to the correct Metal kernel based on `bit_width` and kernel kind:
/// - Matvec (`LinearKernelKind::Matvec`): one threadgroup per output row
/// - Matmul (`LinearKernelKind::Matmul`): tiled matmul with `token_count` rows
fn encode_affine_projection(
    encoder: &ironmill_metal_sys::ComputeEncoder,
    input: &MetalBuffer,
    weight: &AffineQuantizedWeight,
    output: &MetalBuffer,
    pipelines: &super::ops::MetalPipelines,
    kind: LinearKernelKind,
    token_count: usize,
) -> Result<(), InferenceError> {
    let (n, k) = weight.shape;

    let pipeline = pipelines
        .affine_pipeline(weight.bit_width.into(), kind)
        .ok_or_else(|| {
            InferenceError::runtime(format!(
                "unsupported affine bit_width: {}",
                weight.bit_width
            ))
        })?;

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&weight.data, 0, 1);
    encoder.set_buffer(&weight.scales, 0, 2);
    encoder.set_buffer(&weight.zeros, 0, 3);
    encoder.set_buffer(output, 0, 4);

    if kind.is_decode() {
        encoder.set_bytes(&(n as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 6);
        encoder.set_bytes(&weight.group_size.to_le_bytes(), 7);
        // AWQ scales: buffer 8 = scales data, buffer 9 = has_awq flag
        let has_awq: u32 = if weight.awq_scales.is_some() { 1 } else { 0 };
        if let Some(ref awq_buf) = weight.awq_scales {
            encoder.set_buffer(awq_buf, 0, 8);
        } else {
            // Bind the data buffer as a dummy (won't be read when has_awq=0).
            encoder.set_buffer(&weight.data, 0, 8);
        }
        encoder.set_bytes(&has_awq.to_le_bytes(), 9);
        let threads_per_group = 32;
        encoder.dispatch_threadgroups((n, 1, 1), (threads_per_group, 1, 1));
    } else {
        encoder.set_bytes(&(token_count as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(n as u32).to_le_bytes(), 6);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 7);
        encoder.set_bytes(&weight.group_size.to_le_bytes(), 8);
        let has_awq: u32 = if weight.awq_scales.is_some() { 1 } else { 0 };
        if let Some(ref awq_buf) = weight.awq_scales {
            encoder.set_buffer(awq_buf, 0, 9);
        } else {
            encoder.set_buffer(&weight.data, 0, 9);
        }
        encoder.set_bytes(&has_awq.to_le_bytes(), 10);
        encoder.dispatch_threadgroups(
            (
                token_count.div_ceil(MATMUL_TM_TILE),
                n.div_ceil(MATMUL_TN_TILE),
                1,
            ),
            (MATMUL_THREADS_PER_TG, 1, 1),
        );
    }
    Ok(())
}

// ── Shared decode helpers ──────────────────────────────────────

/// Encode QK normalization (Qwen3) and RoPE for Q and K projections.
///
/// When QK-norm weights are present, uses a single fused kernel that
/// normalizes and rotates each head in one pass. Otherwise falls back
/// to standalone RoPE dispatches.
#[allow(clippy::too_many_arguments)]
fn encode_qk_norm_and_rope(
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
fn encode_kv_cache_and_attention(
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
    use_fa2: bool,
    is_anchor: bool,
    window_size: usize,
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
    let _ = use_fa2;

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
            enc.set_bytes(&outlier.k_non_outlier_n_levels.to_le_bytes(), 30);
            enc.dispatch_threadgroups(
                (nh as usize, token_count, 1),
                (256_usize.max(tg_size).min(1024), 1, 1),
            );
        } else {
            // ── Standard TurboQuant dispatch ──
            let (k_cache, v_cache) = kv.layer_caches(layer_idx);
            let (k_scale, v_scale) = kv.layer_scales(layer_idx);
            let (k_qjl_signs, k_r_norms) = kv.layer_k_qjl(layer_idx);

            let cache_write_pos = ring_seq_pos as u32;
            let attn_base_seq = attn_seq_pos as u32;

            // CLA: only anchor layers write to the KV cache.
            if is_anchor {
                // K cache write — batched over all tokens
                enc.set_pipeline(&pipelines.turboquant_cache_write);
                enc.set_buffer(&bufs.k_proj, 0, 0);
                enc.set_buffer(&tq.rotation_signs, 0, 1);
                enc.set_buffer(k_cache, 0, 2);
                enc.set_bytes(&nkv.to_le_bytes(), 3);
                enc.set_bytes(&hd.to_le_bytes(), 4);
                enc.set_bytes(&max_seq.to_le_bytes(), 5);
                enc.set_bytes(&cache_write_pos.to_le_bytes(), 6);
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
                enc.dispatch_threadgroups(
                    (nkv as usize, token_count, 1),
                    ((hd as usize).min(1024), 1, 1),
                );

                // V cache write — batched over all tokens
                enc.set_pipeline(&pipelines.turboquant_cache_write);
                enc.set_buffer(&bufs.v_proj, 0, 0);
                enc.set_buffer(&tq.rotation_signs, 0, 1);
                enc.set_buffer(v_cache, 0, 2);
                enc.set_bytes(&nkv.to_le_bytes(), 3);
                enc.set_bytes(&hd.to_le_bytes(), 4);
                enc.set_bytes(&max_seq.to_le_bytes(), 5);
                enc.set_bytes(&cache_write_pos.to_le_bytes(), 6);
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
            enc.set_buffer(&tq.rotation_signs, 0, 3);
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
            enc.set_buffer(&tq.k_codebook_buf, 0, 14);
            enc.set_buffer(&tq.v_codebook_buf, 0, 15);
            enc.set_buffer(&tq.qjl_matrix, 0, 16);
            enc.set_buffer(k_r_norms, 0, 17);
            enc.set_bytes(&tq.k_n_levels.to_le_bytes(), 18);
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

        // FP16 attention — use fused SDPA for both prefill and decode.
        // The kernel reads directly from the KV cache (contiguous layout).
        let total_seq_len = (attn_seq_pos + token_count) as u32;
        ops::encode_fused_sdpa(
            enc,
            &pipelines.fused_sdpa,
            &ops::FusedSdpaParams {
                q: &bufs.q_proj,
                k: k_cache,
                v: v_cache,
                output: &bufs.attn_out,
                seq_len: total_seq_len,
                token_count: token_count as u32,
                head_dim: hd,
                num_q_heads: nh,
                num_kv_heads: nkv,
                scale: 1.0 / (hd as f32).sqrt(),
                max_seq_len: max_seq,
            },
            None,
        );
    }

    Ok(())
}

/// Encode the FFN block: gate + up projections, SiLU activation, and down projection.
#[allow(clippy::too_many_arguments)]
fn encode_ffn_block(
    enc: &ComputeEncoder,
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
        .map_err(|e| InferenceError::runtime(e.to_string()))?;

    // Gate projection
    encode_projection(
        enc,
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
        enc,
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

    enc.memory_barrier_buffers();

    // SiLU gate
    ops::encode_silu_gate(
        enc,
        &pipelines.silu_gate,
        &bufs.ffn_gate,
        &bufs.ffn_up,
        &bufs.ffn_gate, // in-place output into gate buffer
        (token_count * inter) as u32,
    );

    enc.memory_barrier_buffers();

    // Down projection
    let gate_mat = MpsMatrix::from_buffer(&bufs.ffn_gate, token_count, inter, row_bytes_inter)
        .map_err(|e| InferenceError::runtime(e.to_string()))?;
    encode_projection(
        enc,
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
    enc: &ComputeEncoder,
    pipelines: &super::ops::MetalPipelines,
    bufs: &IntermediateBuffers,
    next_input_norm: Option<&MetalBuffer>,
    h: usize,
    token_count: usize,
    eps: f32,
) -> Result<(), InferenceError> {
    if let Some(norm_weight) = next_input_norm {
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
    Ok(())
}

// ── MLA weight absorption helper ───────────────────────────────

/// Absorb MLA up-projection weights into Q and O projections at load time.
///
/// For each transformer layer, reads W_uk and W_uv from the weight provider,
/// reads the current Q and O weights from the Metal buffers, performs the
/// absorption matrix multiplies, and replaces the Q and O Metal buffers with
/// the absorbed versions.
fn absorb_mla_weights(
    device: &MetalDevice,
    weights: &mut MetalWeights,
    mla: &super::mla::MlaConfig,
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
        let (q_absorbed, o_absorbed) = super::mla::absorb_weights(&w_q, &w_uk, &w_o, &w_uv, mla);

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
            buf: q_buf,
            packed: None, // Absorption changes dimensions; re-packing can be added later.
        };
        weights.layers[layer_idx].o_proj = WeightBuffer::Dense {
            buf: o_buf,
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

impl InferenceEngine for MetalInference {
    fn decode_step(&mut self, token: u32) -> Result<Logits, InferenceError> {
        self.run_pipeline(&[token])
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Logits, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::Decode("empty prefill tokens".into()));
        }

        // With dynamic buffer resizing, process all tokens in one shot
        // unless the caller explicitly set a chunk size.
        let chunk_size = self.config.prefill_chunk_size.unwrap_or(tokens.len());
        let chunk_size = chunk_size.max(1);

        // SWA: prefill chunks must not exceed the smallest window_size so
        // that ring-buffer writes and attention reads stay in bounds.
        let chunk_size = if let Some(ref mc) = self.model_config {
            let min_ws = (0..mc.num_hidden_layers)
                .map(|l| self.config.layer_window_size(l))
                .filter(|&ws| ws > 0)
                .min();
            match min_ws {
                Some(ws) => chunk_size.min(ws),
                None => chunk_size,
            }
        } else {
            chunk_size
        };

        let chunks: Vec<&[u32]> = tokens.chunks(chunk_size).collect();
        let n_chunks = chunks.len();

        for chunk in &chunks[..n_chunks - 1] {
            self.run_pipeline_no_logits(chunk)?;
        }

        self.run_pipeline(chunks[n_chunks - 1])
    }

    fn reset(&mut self) {
        self.seq_pos = 0;
        if let Some(kv) = self.kv_cache.as_mut() {
            kv.reset();
        }
        if let Some(fp16_kv) = self.fp16_kv_cache.as_mut() {
            fp16_kv.reset();
        }
        if let Some(mla_kv) = self.mla_kv_cache.as_mut() {
            mla_kv.reset();
        }
    }

    fn seq_pos(&self) -> usize {
        self.seq_pos
    }

    fn truncate_to(&mut self, pos: usize) {
        assert!(pos <= self.seq_pos);
        self.seq_pos = pos;
        if let Some(kv) = self.kv_cache.as_mut() {
            kv.truncate_to(pos);
        }
        if let Some(fp16_kv) = self.fp16_kv_cache.as_mut() {
            fp16_kv.truncate_to(pos);
        }
        if let Some(mla_kv) = self.mla_kv_cache.as_mut() {
            mla_kv.truncate_to(pos);
        }
    }

    fn model_info(&self) -> &ModelInfo {
        self.model_info
            .as_ref()
            .expect("model_info() called before load()")
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
