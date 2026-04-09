//! Weight loading and resource initialization.

use std::time::Instant;

use half::f16;
use ironmill_metal_sys::StorageMode;
use mil_rs::weights::WeightProvider;

use super::buffers::{IntermediateBuffers, build_matmul_cache, build_rope_cache};
use super::config::Gemma4Config;
use super::config::MetalConfig;
use super::engine::{MetalArtifacts, MetalInference};
use super::error::MetalError;
use super::gdn::GdnState;
use super::kv_cache::Fp16KvCache;
use super::mla::MlaKvCache;
use super::plan::{LayerPlan, ModelPlan};
use super::turboquant::{
    MetalKvCache, MetalTurboQuantModel, OutlierConfig, OutlierQuantConfig, TurboQuantLayerConfig,
    TurboQuantMetalConfig,
};
use super::weights::{MetalWeights, WeightBuffer};
use crate::engine::InferenceError;
use ironmill_core::model_info::ModelInfo;

impl MetalInference {
    /// Shared model bootstrapping logic called by both [`load_weights`] and
    /// [`load`] after weights have been loaded and `self.config` has been set.
    ///
    /// `compute_outliers` controls whether TurboQuant outlier detection runs
    /// (requires reading K/V projection tensors from the provider).
    fn init_model_state(
        &mut self,
        mut weights: MetalWeights,
        provider: &dyn WeightProvider,
        compute_outliers: bool,
    ) -> Result<(), InferenceError> {
        let rotary_dim = self.load_model_config(&weights)?;

        let t0 = Instant::now();
        self.init_pipelines(rotary_dim)?;
        let t_pipelines = t0.elapsed();

        let t1 = Instant::now();
        self.init_weight_buffers(rotary_dim, &weights)?;
        let t_buffers = t1.elapsed();

        let t2 = Instant::now();
        self.init_kv_cache(&mut weights, provider, compute_outliers)?;
        let t_kv = t2.elapsed();

        let t3 = Instant::now();
        self.finalize_model_state(weights)?;
        let t_finalize = t3.elapsed();

        eprintln!(
            "  load: pipelines={:.0}ms  buffers={:.0}ms  kv_cache={:.0}ms  finalize={:.0}ms  total={:.0}ms",
            t_pipelines.as_secs_f64() * 1000.0,
            t_buffers.as_secs_f64() * 1000.0,
            t_kv.as_secs_f64() * 1000.0,
            t_finalize.as_secs_f64() * 1000.0,
            t0.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(())
    }

    /// Reset stale state, load model config and metadata, compute rotary_dim.
    fn load_model_config(&mut self, weights: &MetalWeights) -> Result<usize, InferenceError> {
        // Reset Gemma 4-specific state to avoid stale buffers from a
        // previously loaded model.
        self.global_head_dim = 0;
        self.global_pipelines = None;
        self.global_rope_cos = None;
        self.global_rope_sin = None;
        self.unit_norm_weight = None;
        self.gemma4_config = None;

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
            .unwrap_or_else(|| {
                tracing::warn!(
                    "partial_rotary_factor not specified, defaulting to 1.0 (full-head RoPE)"
                );
                1.0
            });
        let rotary_dim = (mc.head_dim as f64 * partial_rotary_factor) as usize;
        Ok(rotary_dim)
    }

    /// Compile Metal shader pipelines (main + optional global for Gemma 4).
    fn init_pipelines(&mut self, rotary_dim: usize) -> Result<(), InferenceError> {
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?;

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

        Ok(())
    }

    /// Allocate intermediate buffers, RoPE caches, and the decode matmul cache.
    fn init_weight_buffers(
        &mut self,
        rotary_dim: usize,
        weights: &MetalWeights,
    ) -> Result<(), InferenceError> {
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .clone();

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
                        let global_rotary_dim =
                            (global_hd as f64 * global_cfg.partial_rotary_factor) as usize;
                        let (gc, gs) = build_rope_cache(
                            &self.device,
                            global_hd,
                            global_rotary_dim,
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
            build_matmul_cache(&self.device, &mc, self.gemma4_config.as_ref(), weights, 1)
                .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.decode_matmuls_t1 = Some(decode_cache_t1);
        self.decode_matmuls = None;

        Ok(())
    }

    /// Resolve CLA anchors, MLA, sliding window, and build the KV cache
    /// (TurboQuant or FP16).
    fn init_kv_cache(
        &mut self,
        weights: &mut MetalWeights,
        provider: &dyn WeightProvider,
        compute_outliers: bool,
    ) -> Result<(), InferenceError> {
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .clone();

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
            absorb_mla_weights(&self.device, weights, mla, mc.hidden_size, provider)
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
            let outlier_cfg: Option<OutlierConfig> = if compute_outliers {
                let use_qjl = self.config.n_bits < 4;
                if use_qjl {
                    let mut weight_data: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
                    for layer in 0..mc.num_hidden_layers {
                        let prefix = format!("model.layers.{layer}");
                        let k_name = format!("{prefix}.self_attn.k_proj.weight");
                        let v_name = format!("{prefix}.self_attn.v_proj.weight");
                        if let (Ok(k_t), Ok(v_t)) =
                            (provider.tensor(&k_name), provider.tensor(&v_name))
                        {
                            weight_data.push((k_t.data.to_vec(), v_t.data.to_vec()));
                        } else {
                            tracing::warn!(
                                layer = layer,
                                "TurboQuant outlier config: skipping layer (missing k_proj or v_proj tensor)"
                            );
                        }
                    }
                    if weight_data.len() == mc.num_hidden_layers {
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
                    } else if weight_data.is_empty() {
                        return Err(InferenceError::runtime(
                            "TurboQuant outlier config: all layers failed to load k_proj/v_proj tensors"
                                .to_string(),
                        ));
                    } else {
                        tracing::warn!(
                            loaded = weight_data.len(),
                            total = mc.num_hidden_layers,
                            "TurboQuant outlier config: partial layer data, proceeding"
                        );
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
                    }
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
                            per_head_k_codebooks: None,
                            per_head_v_codebooks: None,
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
                outlier_config: OutlierQuantConfig::default(),
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

        Ok(())
    }

    /// Allocate GDN state, compact weights, build execution plans, reset seq_pos.
    fn finalize_model_state(&mut self, mut weights: MetalWeights) -> Result<(), InferenceError> {
        let mc = self
            .model_config
            .as_ref()
            .ok_or(InferenceError::NotLoaded)?
            .clone();

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
        weights.compact();

        self.weights = Some(weights);

        // ── Build per-layer execution plans ───────────────────────
        self.layer_plans = LayerPlan::build(
            &mc,
            self.gemma4_config.as_ref(),
            gdn_cfg.as_ref(),
            self.config.cla_config.as_ref(),
            self.weights.as_ref().unwrap(),
        )?;

        // ── Build model-level execution plan ──────────────────────
        self.model_plan = Some(ModelPlan::build(
            &mc,
            &self.config,
            self.gemma4_config.as_ref(),
            self.layer_plans.clone(),
        ));

        self.seq_pos = 0;
        Ok(())
    }

    /// Load model weights directly from a [`WeightProvider`], bypassing
    /// the type-erased [`InferenceEngine::load`] interface.
    pub fn load_weights(
        &mut self,
        provider: &dyn mil_rs::weights::WeightProvider,
        config: MetalConfig,
    ) -> Result<(), InferenceError> {
        self.config = config;

        let t_weight_start = Instant::now();
        let weights = MetalWeights::load(&self.device, provider, self.config.force_cpu_dequant)
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        eprintln!(
            "  load: weight_loading={:.0}ms",
            t_weight_start.elapsed().as_secs_f64() * 1000.0,
        );

        self.init_model_state(weights, provider, true)
    }

    /// Load model from pre-built [`MetalArtifacts`].
    pub fn load(&mut self, artifacts: &MetalArtifacts<'_>) -> Result<(), InferenceError> {
        self.config = artifacts.config.clone();

        let t_weight_start = Instant::now();
        let weights = MetalWeights::load(
            &self.device,
            artifacts.weights,
            self.config.force_cpu_dequant,
        )
        .map_err(|e| InferenceError::runtime(e.to_string()))?;
        eprintln!(
            "  load: weight_loading={:.0}ms",
            t_weight_start.elapsed().as_secs_f64() * 1000.0,
        );

        self.init_model_state(weights, artifacts.weights, false)
    }
}

// ── MLA weight absorption helper ───────────────────────────────

/// Absorb MLA up-projection weights into Q and O projections at load time.
///
/// For each transformer layer, reads W_uk and W_uv from the weight provider,
/// reads the current Q and O weights from the Metal buffers, performs the
/// absorption matrix multiplies, and replaces the Q and O Metal buffers with
/// the absorbed versions.
fn absorb_mla_weights(
    device: &ironmill_metal_sys::MetalDevice,
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
        let w_q: Vec<half::f16> = q_bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
            .collect();
        let w_o: Vec<half::f16> = o_bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
            .collect();

        // Load up-projection weights. These may be combined in kv_b_proj;
        // the first num_heads * qk_nope_head_dim rows are W_uk, the next
        // num_heads * v_head_dim rows are W_uv.
        let (w_uk, w_uv) = if provider.has_tensor(&uk_name) {
            let tensor = provider
                .tensor(&uk_name)
                .map_err(|e| MetalError::WeightLoading(format!("{uk_name}: {e}")))?;
            let all_f16: Vec<half::f16> = tensor
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
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
                let w_uv_vec: Vec<half::f16> = uv_tensor
                    .data
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
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
