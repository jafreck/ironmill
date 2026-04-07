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
    CommandBufferStatus, ComputeEncoder, MetalBuffer, MetalDevice, StorageMode,
};
use mil_rs::weights::{Architecture, ModelConfig, WeightProvider};

use super::attention::{
    encode_end_of_layer_residual, encode_kv_cache_and_attention, encode_qk_norm_and_rope,
};
use super::buffers::{
    IntermediateBuffers, ModelConfigExt, MpsMatmulCache,
    build_matmul_cache, build_rope_cache, bytes_as_f16, read_buffer_f32, read_weight_f32,
    write_buffer_f32,
};
use super::config::Gemma4Config;
use super::config::MetalConfig;
use super::error::MetalError;
use super::ffn::{encode_ffn_block, encode_moe_block};
use super::gdn::{GdnState, encode_gdn_decode, encode_gdn_prefill};
use super::kv_cache::Fp16KvCache;
use super::mla::{MlaConfig, MlaKvCache};
use super::ops;
use super::ops::LinearKernelKind;
use super::plan::{AttentionKind, LayerPlan, RopeTable};
use super::ple;
use super::turboquant::{
    MetalKvCache, MetalTurboQuantModel, OutlierConfig, TurboQuantLayerConfig, TurboQuantMetalConfig,
};
use super::weights::{
    AffineQuantizedWeight, DualScaleQuantizedWeight, LayerWeights, MetalWeights, QuantizedWeight,
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

// ── Public artifacts type for load() ────────────────────────────

/// Artifacts passed to [`MetalInference::load`] via the type-erased
/// [`InferenceEngine`] interface.
pub struct MetalArtifacts<'a> {
    /// Weight provider for loading model tensors.
    pub weights: &'a dyn WeightProvider,
    /// Metal backend configuration.
    pub config: MetalConfig,
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
    /// Second pipeline set for global layers with a different HEAD_DIM (Gemma 4).
    global_pipelines: Option<super::ops::MetalPipelines>,
    /// The head_dim that `global_pipelines` was compiled for (0 if not set).
    global_head_dim: usize,
    weights: Option<MetalWeights>,
    turboquant: Option<MetalTurboQuantModel>,
    kv_cache: Option<MetalKvCache>,
    fp16_kv_cache: Option<Fp16KvCache>,
    intermediate_buffers: Option<IntermediateBuffers>,
    rope_cos: Option<MetalBuffer>,
    rope_sin: Option<MetalBuffer>,
    /// RoPE tables for global (full_attention) layers with different theta.
    global_rope_cos: Option<MetalBuffer>,
    global_rope_sin: Option<MetalBuffer>,
    /// Unit-weight buffer for scale-free RMSNorm (e.g., Gemma 4 V-norm).
    /// Contains [1.0; max_head_dim] in FP16.
    unit_norm_weight: Option<MetalBuffer>,
    /// MPS matmul cache for prefill (variable token_count > 1).
    decode_matmuls: Option<MpsMatmulCache>,
    /// MPS matmul cache for single-token decode (token_count=1), preserved
    /// across prefill→decode transitions so it never needs rebuilding.
    decode_matmuls_t1: Option<MpsMatmulCache>,
    config: MetalConfig,
    model_config: Option<ModelConfig>,
    /// MLA configuration (set when the model uses Multi-Head Latent Attention).
    mla_config: Option<MlaConfig>,
    /// Gemma 4 per-layer attention configuration.
    gemma4_config: Option<Gemma4Config>,
    /// MLA compressed KV cache (used instead of fp16_kv_cache for MLA models).
    mla_kv_cache: Option<MlaKvCache>,
    /// Pre-allocated buffer for FP16 logits readback.
    logits_fp16_buf: Vec<u8>,
    /// Pre-allocated buffer for serializing token IDs to GPU.
    token_bytes_buf: Vec<u8>,
    seq_pos: usize,
    /// Cached model info, populated during `load()`.
    model_info: Option<ModelInfo>,
    /// GDN (Gated Delta Network) recurrent state for linear-attention layers.
    gdn_state: Option<GdnState>,
    /// Per-layer execution plan, resolved at load time.
    layer_plans: Vec<LayerPlan>,
    /// D2Quant Deviation-Aware Correction (DAC) per-layer bias vectors.
    /// Each buffer is `[hidden_size]` FP16. Added to post-attention LayerNorm
    /// output to compensate for quantization-induced mean shift.
    dac_biases: Option<Vec<MetalBuffer>>,
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
        let device = MetalDevice::system_default().map_err(MetalError::Metal)?;
        let queue = device.create_command_queue().map_err(MetalError::Metal)?;
        // Pipelines are compiled in load() once head_dim is known.
        Ok(Self {
            device,
            queue,
            pipelines: None,
            global_pipelines: None,
            global_head_dim: 0,
            weights: None,
            turboquant: None,
            kv_cache: None,
            fp16_kv_cache: None,
            intermediate_buffers: None,
            rope_cos: None,
            rope_sin: None,
            global_rope_cos: None,
            global_rope_sin: None,
            unit_norm_weight: None,
            decode_matmuls: None,
            decode_matmuls_t1: None,
            config,
            model_config: None,
            mla_config: None,
            gemma4_config: None,
            mla_kv_cache: None,
            logits_fp16_buf: Vec::new(),
            token_bytes_buf: Vec::new(),
            seq_pos: 0,
            model_info: None,
            gdn_state: None,
            layer_plans: Vec::new(),
            dac_biases: None,
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
        self.gemma4_config = Gemma4Config::from_model_config(&mc);

        // Pre-allocate logits readback buffer (vocab × 2 bytes for FP16).
        self.logits_fp16_buf.resize(mc.vocab_size * 2, 0);

        // Compute rotary_dim from partial_rotary_factor (defaults to head_dim).
        let partial_rotary_factor = mc
            .extra
            .get("rope_parameters")
            .and_then(|rp| rp.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .or_else(|| {
                mc.extra
                    .get("partial_rotary_factor")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);
        let rotary_dim = (mc.head_dim as f64 * partial_rotary_factor) as usize;

        // Compile Metal shader pipelines with the model's head_dim and rotary_dim.
        let pipelines = super::ops::MetalPipelines::compile(&self.device, mc.head_dim, rotary_dim)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.pipelines = Some(pipelines);

        // If Gemma 4 global layers use a different head_dim, compile a second
        // pipeline set with that HEAD_DIM for correct shader dispatch.
        if let Some(ref g4) = self.gemma4_config {
            if g4.global_head_dim != mc.head_dim {
                let global_pipelines = super::ops::MetalPipelines::compile(
                    &self.device,
                    g4.global_head_dim,
                    g4.global_head_dim,
                )
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
                self.global_head_dim = g4.global_head_dim;
                self.global_pipelines = Some(global_pipelines);
            }
        }

        let bufs = IntermediateBuffers::allocate(&self.device, 1, &mc, self.gemma4_config.as_ref())
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.intermediate_buffers = Some(bufs);

        let (cos, sin) = build_rope_cache(
            &self.device,
            mc.head_dim,
            rotary_dim,
            self.config.max_seq_len,
            mc.rope_theta,
            1.0,
        )
        .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.rope_cos = Some(cos);
        self.rope_sin = Some(sin);

        // Build global-layer RoPE tables if Gemma 4 uses a different theta.
        if let Some(ref g4) = self.gemma4_config {
            if let Some(rp) = mc.rope_parameters() {
                if let Some(global_cfg) = rp.get("full_attention") {
                    let global_hd = g4.global_head_dim;
                    if global_hd != mc.head_dim
                        || global_cfg.theta != mc.rope_theta
                        || global_cfg.partial_rotary_factor != 1.0
                    {
                        let (gc, gs) = build_rope_cache(
                            &self.device,
                            global_hd,
                            global_hd,
                            self.config.max_seq_len,
                            global_cfg.theta,
                            global_cfg.partial_rotary_factor,
                        )
                        .map_err(|e| InferenceError::runtime(e.to_string()))?;
                        self.global_rope_cos = Some(gc);
                        self.global_rope_sin = Some(gs);
                    }
                }
            }

            // Allocate unit-weight buffer for scale-free V-norm.
            let max_hd = g4
                .layer_configs
                .iter()
                .map(|lc| lc.head_dim)
                .max()
                .unwrap_or(0);
            if max_hd > 0 {
                let unit_data: Vec<u8> = (0..max_hd)
                    .flat_map(|_| f16::from_f64(1.0).to_le_bytes())
                    .collect();
                let buf = self
                    .device
                    .create_buffer_with_data(&unit_data, StorageMode::Shared)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                self.unit_norm_weight = Some(buf);
            }
        }

        let decode_cache_t1 =
            build_matmul_cache(&self.device, &mc, self.gemma4_config.as_ref(), &weights, 1)
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

            let tq_layer_configs: Vec<TurboQuantLayerConfig> =
                if let Some(ref g4) = self.gemma4_config {
                    g4.layer_configs
                        .iter()
                        .map(|lc| TurboQuantLayerConfig {
                            head_dim: lc.head_dim,
                            num_kv_heads: lc.num_kv_heads,
                        })
                        .collect()
                } else {
                    Vec::new()
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
                layer_configs: tq_layer_configs,
            };
            let tq_model = MetalTurboQuantModel::new(&self.device, tq_config.clone())
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let kv_cache = MetalKvCache::new(&self.device, &tq_config)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.turboquant = Some(tq_model);
            self.kv_cache = Some(kv_cache);
            self.fp16_kv_cache = None;
        } else {
            let per_layer_dims: Option<Vec<(usize, usize)>> =
                self.gemma4_config.as_ref().map(|g4| {
                    g4.layer_configs
                        .iter()
                        .map(|lc| (lc.num_kv_heads, lc.head_dim))
                        .collect()
                });
            let fp16_kv = Fp16KvCache::new(
                &self.device,
                mc.num_hidden_layers,
                mc.num_key_value_heads,
                self.config.max_seq_len,
                mc.head_dim,
                cla_anchors.clone(),
                &layer_window_sizes,
                per_layer_dims.as_deref(),
            )
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.fp16_kv_cache = Some(fp16_kv);
            self.turboquant = None;
            self.kv_cache = None;
        }

        // ── GDN state allocation ────────────────────────────────
        let gdn_cfg = super::config::GdnModelConfig::from_model_config(&mc);
        if let Some(ref cfg) = gdn_cfg {
            let gdn = GdnState::new(&self.device, cfg, mc.hidden_size)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.gdn_state = Some(gdn);
        } else {
            self.gdn_state = None;
        }

        // Free redundant row-major buffers now that all load-time transforms
        // (split_q_gate_weight, norm offsets, MLA absorption) are complete.
        // This typically halves GPU memory for dense FP16 models.
        weights.drop_dense_row_major();

        self.weights = Some(weights);

        // ── Build per-layer execution plans ───────────────────────
        self.layer_plans = LayerPlan::build(
            &mc,
            self.gemma4_config.as_ref(),
            gdn_cfg.as_ref(),
            self.config.cla_config.as_ref(),
            self.weights.as_ref().unwrap(),
        );

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

    /// Mutable access to loaded weights for post-load transforms (e.g. D2Quant simulation).
    pub fn weights_mut(&mut self) -> &mut MetalWeights {
        self.weights
            .as_mut()
            .expect("weights_mut() called before load()")
    }

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
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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

        Ok(all_logits)
    }

    /// Run the transformer decode pipeline for `token_count` tokens.
    /// Returns logits for the last token position.
    fn run_pipeline(&mut self, token_ids: &[u32]) -> Result<Logits, InferenceError> {
        self.run_pipeline_inner(token_ids, false, false, false)
    }

    /// Decode one token, returning logits AND the final hidden state (FP32).
    ///
    /// The hidden state is the output of the final RMSNorm (before the LM head
    /// projection), read back from `norm_out`. Used by EAGLE-3 speculative
    /// decoding where the draft head needs the target model's hidden state.
    pub fn decode_step_with_hidden(
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

    fn run_pipeline_inner(
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
        let nkv = mc.num_kv_heads() as u32;
        let hd = mc.head_dim as u32;
        let inter = mc.intermediate_size;
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
        let mut cmd_buf = self
            .queue
            .command_buffer()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        let mut enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;

        // Step 0: Fused embedding lookup + first-layer RMSNorm.
        // Writes both hidden_state (raw embedding for residual) and
        // norm_out (normalized for first layer's projections).
        // Gemma models scale embeddings by sqrt(hidden_size).
        let embed_scale: f32 = if mc.architecture == Architecture::Gemma {
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
        enc.memory_barrier_buffers();

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
                    enc.memory_barrier_buffers();
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

                    // Build the projection list: skip Q if P1 fusion already computed it.
                    let mut projections: Vec<(&WeightBuffer, &MetalBuffer, usize)> = Vec::new();
                    if !p1_skip {
                        projections.push((&lw.q_proj, &bufs.q_proj, qkv_out_features));
                    }
                    projections.push((&lw.k_proj, &bufs.k_proj, kv_out_features));
                    projections.push((&lw.v_proj, &bufs.v_proj, kv_out_features));
                    for (weight, output_buf, out_features) in projections {
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

                    // Qwen3.5 attn_output_gate: dispatch gate projection alongside Q/K/V
                    // (reads norm_out, writes q_gate — independent of Q/K/V outputs).
                    // The gate result is only consumed later by sigmoid_gate after attention,
                    // so no separate barrier is needed here.
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
                    }
                    enc.memory_barrier_buffers();

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
                            enc.memory_barrier_buffers();
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
                    enc.memory_barrier_buffers();

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
                    )?;
                    enc.memory_barrier_buffers();

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
                            enc.memory_barrier_buffers();
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
                    enc.memory_barrier_buffers();

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
                        enc.memory_barrier_buffers();
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
                        enc.memory_barrier_buffers();
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
                    enc.memory_barrier_buffers();
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
            enc.memory_barrier_buffers();

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
                enc.memory_barrier_buffers();
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
        enc.memory_barrier_buffers();

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
        if let Some(ref g4) = self.gemma4_config {
            if let Some(softcap) = g4.final_logit_softcapping {
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
            let u16_slice: &[u16] = {
                let ptr = self.logits_fp16_buf.as_ptr() as *const u16;
                let len = self.logits_fp16_buf.len() / 2;
                // SAFETY: logits_fp16_buf is aligned to 2 bytes (from Metal buffer readback)
                // and len is always even (vocab * 2 bytes). The lifetime is bounded by self.
                #[allow(unsafe_code)]
                unsafe {
                    std::slice::from_raw_parts(ptr, len)
                }
            };
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
        self.gemma4_config = Gemma4Config::from_model_config(&mc);
        self.model_info = Some(ModelInfo::from_config(&mc));

        // Pre-allocate logits readback buffer (vocab × 2 bytes for FP16).
        self.logits_fp16_buf.resize(mc.vocab_size * 2, 0);

        // Compute rotary_dim from partial_rotary_factor (defaults to head_dim).
        let partial_rotary_factor = mc
            .extra
            .get("rope_parameters")
            .and_then(|rp| rp.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .or_else(|| {
                mc.extra
                    .get("partial_rotary_factor")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);
        let rotary_dim = (mc.head_dim as f64 * partial_rotary_factor) as usize;

        // Compile Metal shader pipelines with the model's head_dim so
        // shared memory is sized exactly via #define HEAD_DIM.
        let pipelines = super::ops::MetalPipelines::compile(&self.device, mc.head_dim, rotary_dim)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.pipelines = Some(pipelines);

        // Allocate intermediate buffers (start at 1 token; run_pipeline_inner
        // grows them on demand for larger prefill batches).
        let bufs = IntermediateBuffers::allocate(&self.device, 1, &mc, self.gemma4_config.as_ref())
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.intermediate_buffers = Some(bufs);

        // Build RoPE cos/sin caches.
        let (cos, sin) = build_rope_cache(
            &self.device,
            mc.head_dim,
            rotary_dim,
            self.config.max_seq_len,
            mc.rope_theta,
            1.0,
        )
        .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.rope_cos = Some(cos);
        self.rope_sin = Some(sin);

        // Build global-layer RoPE tables if Gemma 4 uses a different theta.
        if let Some(ref g4) = self.gemma4_config {
            if let Some(rp) = mc.rope_parameters() {
                if let Some(global_cfg) = rp.get("full_attention") {
                    let global_hd = g4.global_head_dim;
                    if global_hd != mc.head_dim
                        || global_cfg.theta != mc.rope_theta
                        || global_cfg.partial_rotary_factor != 1.0
                    {
                        let (gc, gs) = build_rope_cache(
                            &self.device,
                            global_hd,
                            global_hd,
                            self.config.max_seq_len,
                            global_cfg.theta,
                            global_cfg.partial_rotary_factor,
                        )
                        .map_err(|e| InferenceError::runtime(e.to_string()))?;
                        self.global_rope_cos = Some(gc);
                        self.global_rope_sin = Some(gs);
                    }
                }
            }

            // Allocate unit-weight buffer for scale-free V-norm.
            let max_hd = g4
                .layer_configs
                .iter()
                .map(|lc| lc.head_dim)
                .max()
                .unwrap_or(0);
            if max_hd > 0 {
                let unit_data: Vec<u8> = (0..max_hd)
                    .flat_map(|_| f16::from_f64(1.0).to_le_bytes())
                    .collect();
                let buf = self
                    .device
                    .create_buffer_with_data(&unit_data, StorageMode::Shared)
                    .map_err(|e| InferenceError::runtime(e.to_string()))?;
                self.unit_norm_weight = Some(buf);
            }
        }

        // Build MPS matmul cache for single-token decode.
        let decode_cache_t1 =
            build_matmul_cache(&self.device, &mc, self.gemma4_config.as_ref(), &weights, 1)
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
            let tq_layer_configs: Vec<TurboQuantLayerConfig> =
                if let Some(ref g4) = self.gemma4_config {
                    g4.layer_configs
                        .iter()
                        .map(|lc| TurboQuantLayerConfig {
                            head_dim: lc.head_dim,
                            num_kv_heads: lc.num_kv_heads,
                        })
                        .collect()
                } else {
                    Vec::new()
                };

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
                layer_configs: tq_layer_configs,
            };
            let tq_model = MetalTurboQuantModel::new(&self.device, tq_config.clone())
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            let kv_cache = MetalKvCache::new(&self.device, &tq_config)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.turboquant = Some(tq_model);
            self.kv_cache = Some(kv_cache);
            self.fp16_kv_cache = None;
        } else {
            let per_layer_dims: Option<Vec<(usize, usize)>> =
                self.gemma4_config.as_ref().map(|g4| {
                    g4.layer_configs
                        .iter()
                        .map(|lc| (lc.num_kv_heads, lc.head_dim))
                        .collect()
                });
            let fp16_kv = Fp16KvCache::new(
                &self.device,
                mc.num_hidden_layers,
                mc.num_key_value_heads,
                self.config.max_seq_len,
                mc.head_dim,
                cla_anchors.clone(),
                &layer_window_sizes,
                per_layer_dims.as_deref(),
            )
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.fp16_kv_cache = Some(fp16_kv);
            self.turboquant = None;
            self.kv_cache = None;
        }

        // ── GDN state allocation ────────────────────────────────
        let gdn_cfg = super::config::GdnModelConfig::from_model_config(&mc);
        if let Some(ref cfg) = gdn_cfg {
            let gdn = GdnState::new(&self.device, cfg, mc.hidden_size)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
            self.gdn_state = Some(gdn);
        } else {
            self.gdn_state = None;
        }

        weights.drop_dense_row_major();

        self.weights = Some(weights);

        // ── Build per-layer execution plans ───────────────────────
        self.layer_plans = LayerPlan::build(
            &mc,
            self.gemma4_config.as_ref(),
            gdn_cfg.as_ref(),
            self.config.cla_config.as_ref(),
            self.weights.as_ref().unwrap(),
        );

        self.seq_pos = 0;
        Ok(())
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
pub(crate) fn encode_projection(
    enc: &ComputeEncoder,
    input_buf: &MetalBuffer,
    weight: &WeightBuffer,
    output_buf: &MetalBuffer,
    pipelines: &super::ops::MetalPipelines,
    token_count: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(), InferenceError> {
    let kernel_kind = LinearKernelKind::for_token_count(token_count);

    match weight {
        WeightBuffer::Dense {
            buf: None,
            packed: None,
        } => {
            // Empty placeholder (e.g., GDN layer Q/K/V/O) — skip dispatch.
            return Ok(());
        }
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
        WeightBuffer::DualScaleQuantized(dq) => {
            encode_d2quant_projection(
                enc,
                input_buf,
                dq,
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
        // Use row-major kernel for INT4 decode when available (32× better cache utilization)
        let use_rowmajor = weight.bit_width == 4 && weight.data_row_major.is_some();
        if use_rowmajor {
            encoder.set_pipeline(&pipelines.affine_matvec_int4_rowmajor);
            encoder.set_buffer(weight.data_row_major.as_ref().unwrap(), 0, 1);
        }
        encoder.set_bytes(&(n as u32).to_le_bytes(), 5);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 6);
        encoder.set_bytes(&weight.group_size.to_le_bytes(), 7);
        let has_awq: u32 = if weight.awq_scales.is_some() { 1 } else { 0 };
        if let Some(ref awq_buf) = weight.awq_scales {
            encoder.set_buffer(awq_buf, 0, 8);
        } else {
            encoder.set_buffer(&weight.data, 0, 8);
        }
        encoder.set_bytes(&has_awq.to_le_bytes(), 9);
        // Both row-major and blocked scalar use same dispatch: 1 row per TG, 32 threads
        encoder.dispatch_threadgroups((n, 1, 1), (32, 1, 1));
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

// ── D2Quant kernel dispatch ────────────────────────────────────

/// Encode a fused D2Quant dual-scale quantized projection via compute kernel.
///
/// Dispatches to the correct Metal kernel based on `bit_width` and kernel kind:
/// - Matvec (`LinearKernelKind::Matvec`): one threadgroup per output row
/// - Matmul (`LinearKernelKind::Matmul`): tiled matmul with `token_count` rows
#[allow(clippy::too_many_arguments)]
fn encode_d2quant_projection(
    encoder: &ironmill_metal_sys::ComputeEncoder,
    input: &MetalBuffer,
    weight: &DualScaleQuantizedWeight,
    output: &MetalBuffer,
    pipelines: &super::ops::MetalPipelines,
    kind: LinearKernelKind,
    token_count: usize,
) -> Result<(), InferenceError> {
    let (n, k) = weight.shape;

    let pipeline = pipelines
        .d2quant_pipeline(weight.bit_width.into(), kind)
        .ok_or_else(|| {
            InferenceError::runtime(format!(
                "unsupported d2quant bit_width: {}",
                weight.bit_width
            ))
        })?;

    encoder.set_pipeline(pipeline);
    encoder.set_buffer(input, 0, 0);
    encoder.set_buffer(&weight.data, 0, 1);
    encoder.set_buffer(&weight.normal_scale, 0, 2);
    encoder.set_buffer(&weight.normal_zero, 0, 3);
    encoder.set_buffer(&weight.outlier_scale, 0, 4);
    encoder.set_buffer(&weight.outlier_zero, 0, 5);
    encoder.set_buffer(&weight.outlier_mask, 0, 6);
    encoder.set_buffer(output, 0, 7);

    if kind.is_decode() {
        encoder.set_bytes(&(n as u32).to_le_bytes(), 8);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 9);
        encoder.set_bytes(&weight.group_size.to_le_bytes(), 10);
        let threads_per_group = 32;
        encoder.dispatch_threadgroups((n, 1, 1), (threads_per_group, 1, 1));
    } else {
        encoder.set_bytes(&(token_count as u32).to_le_bytes(), 8);
        encoder.set_bytes(&(n as u32).to_le_bytes(), 9);
        encoder.set_bytes(&(k as u32).to_le_bytes(), 10);
        encoder.set_bytes(&weight.group_size.to_le_bytes(), 11);
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

        if n_chunks <= 1 {
            // Single chunk — no pipelining needed.
            return self.run_pipeline(chunks[0]);
        }

        // Pipelined chunked prefill: submit non-last chunks with
        // skip_wait=true so the GPU executes while the CPU encodes the
        // next chunk. Alternate token ID buffers to avoid write-after-read
        // hazards on the Shared buffer.
        for (i, chunk) in chunks[..n_chunks - 1].iter().enumerate() {
            let use_alt = i % 2 != 0;
            self.run_pipeline_inner(chunk, true, use_alt, true)?;
        }

        // Last chunk: use the opposite buffer parity, wait for completion,
        // and read back logits.
        let last_alt = (n_chunks - 1) % 2 != 0;
        self.run_pipeline_inner(chunks[n_chunks - 1], false, last_alt, false)
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
        if let Some(gdn) = self.gdn_state.as_ref() {
            gdn.reset();
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
