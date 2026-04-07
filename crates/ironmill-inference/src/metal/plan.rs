//! Pre-computed per-layer and model-level execution plans.

use mil_rs::weights::{Architecture, ModelConfig};

use super::config::{ClaConfig, GdnModelConfig, Gemma4Config, MetalConfig};
use super::weights::MetalWeights;

/// How the attention block is computed for this layer.
#[derive(Clone, Debug)]
pub(crate) enum AttentionKind {
    /// Standard multi-head / grouped-query attention with KV cache.
    Standard {
        /// Whether this layer has an output gate (Qwen 3.5 `attn_output_gate`).
        has_output_gate: bool,
        /// Whether to apply scale-free V-norm (Gemma 4).
        has_v_norm: bool,
    },
    /// GDN (Gated Delta Network) linear attention with recurrent state.
    Gdn {
        /// Index into `GdnState::layers` for this layer's conv/recurrent state.
        gdn_index: usize,
    },
}

/// Which RoPE cos/sin table to use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RopeTable {
    /// Default model-level tables (e.g. sliding-window layers).
    Default,
    /// Global tables with different theta/rotary_factor (e.g. Gemma 4 full-attention).
    Global,
}

/// MoE configuration for a single layer.
#[derive(Clone, Debug)]
pub(crate) struct MoeLayerConfig {
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
    pub(crate) moe_intermediate_size: usize,
}

/// Pre-computed per-layer config — eliminates runtime architecture checks.
#[derive(Clone, Debug)]
pub(crate) struct LayerPlan {
    // ── Dimensions (may vary per layer for Gemma 4) ──
    pub(crate) head_dim: u32,
    pub(crate) num_kv_heads: u32,
    pub(crate) window_size: usize,
    pub(crate) intermediate_size: usize,

    // ── Attention ──
    pub(crate) attention: AttentionKind,
    pub(crate) attn_scale: f32,
    pub(crate) rope_table: RopeTable,
    /// Whether this is a KV-cache anchor layer (CLA).
    pub(crate) kv_anchor: bool,
    /// Which KV cache slot this layer uses (= layer_idx unless CLA remaps it).
    pub(crate) kv_cache_layer: usize,
    /// Whether to use the global pipeline set (different HEAD_DIM compile).
    pub(crate) use_global_pipelines: bool,

    // ── FFN ──
    pub(crate) use_gelu: bool,
    pub(crate) moe: Option<MoeLayerConfig>,

    // ── Post-layer ──
    pub(crate) has_post_ffn_norm: bool,
    pub(crate) has_layer_scalar: bool,
    pub(crate) has_ple: bool,
}

impl LayerPlan {
    /// Build per-layer plans from model config + optional architecture configs.
    pub(crate) fn build(
        mc: &ModelConfig,
        g4: Option<&Gemma4Config>,
        gdn_cfg: Option<&GdnModelConfig>,
        cla: Option<&ClaConfig>,
        weights: &MetalWeights,
    ) -> Vec<Self> {
        let has_output_gate = mc
            .extra
            .get("attn_output_gate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let has_v_norm = g4.is_some();

        let ple_enabled = g4.is_some_and(|g| g.ple_hidden_size > 0);

        (0..mc.num_hidden_layers)
            .map(|i| {
                let lw = &weights.layers[i];
                let is_gdn = lw.gdn_in_proj_qkv.is_some();

                // Per-layer dims (Gemma 4 varies head_dim/kv_heads per layer)
                let (hd, nkv, ws, inter) = if let Some(g4) = g4 {
                    let lc = &g4.layer_configs[i];
                    (
                        lc.head_dim as u32,
                        lc.num_kv_heads as u32,
                        lc.window_size,
                        lc.intermediate_size,
                    )
                } else {
                    (
                        mc.head_dim as u32,
                        mc.num_key_value_heads as u32,
                        mc.extra
                            .get("sliding_window")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize,
                        mc.intermediate_size,
                    )
                };

                // Attention kind
                let attention = if is_gdn {
                    let gdn = gdn_cfg.expect("GDN layer but no GDN config");
                    let gdn_index = gdn
                        .gdn_layer_indices
                        .iter()
                        .position(|&l| l == i)
                        .expect("GDN layer not in gdn_layer_indices");
                    AttentionKind::Gdn { gdn_index }
                } else {
                    AttentionKind::Standard {
                        has_output_gate,
                        has_v_norm,
                    }
                };

                // RoPE table selection
                let rope_table = if let Some(g4) = g4 {
                    if g4.layer_configs[i].is_global {
                        RopeTable::Global
                    } else {
                        RopeTable::Default
                    }
                } else {
                    RopeTable::Default
                };

                // KV anchor / CLA mapping
                let is_anchor_default = cla.is_none_or(|c| c.is_anchor(i));

                let (kv_anchor, kv_cache_layer) = if let Some(g4) = g4 {
                    // Gemma 4: kv_anchor field overrides CLA mapping.
                    // kv_anchor = Some(target_layer) means "share that layer's KV cache".
                    // kv_anchor = None means this IS an anchor layer (has its own cache).
                    if let Some(anchor) = g4.layer_configs[i].kv_anchor {
                        (false, anchor)
                    } else {
                        (is_anchor_default, i)
                    }
                } else {
                    (is_anchor_default, i)
                };

                // Global pipelines (different HEAD_DIM compile)
                let use_global_pipelines = if let Some(g4) = g4 {
                    g4.layer_configs[i].is_global && g4.global_head_dim != mc.head_dim
                } else {
                    false
                };

                // MoE
                let moe = if let Some(g4) = g4 {
                    let lc = &g4.layer_configs[i];
                    if lc.enable_moe && g4.num_experts > 0 && !lw.expert_gate_projs.is_empty() {
                        Some(MoeLayerConfig {
                            num_experts: g4.num_experts,
                            top_k: g4.top_k_experts,
                            moe_intermediate_size: g4.moe_intermediate_size,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };

                LayerPlan {
                    head_dim: hd,
                    num_kv_heads: nkv,
                    window_size: ws,
                    intermediate_size: inter,
                    attention,
                    attn_scale: mc.attn_scale(),
                    rope_table,
                    kv_anchor,
                    kv_cache_layer,
                    use_global_pipelines,
                    use_gelu: mc.use_gelu(),
                    moe,
                    has_post_ffn_norm: lw.post_ffn_norm.is_some(),
                    has_layer_scalar: lw.layer_scalar.is_some(),
                    has_ple: ple_enabled && lw.ple_gate.is_some(),
                }
            })
            .collect()
    }
}

// ── Model-level execution plan ─────────────────────────────────

/// PLE (Per-Layer Embedding) configuration for the model.
#[derive(Clone, Debug)]
pub(crate) struct PlePlan {
    pub(crate) ple_hidden_size: usize,
}

/// Strategy for residual connections around the attention block.
#[derive(Clone, Debug)]
pub(crate) enum ResidualStrategy {
    /// Standard pre-norm: fused residual + RMSNorm in one dispatch.
    Fused,
    /// Split residual add and norm (e.g. when layer_scalar is present).
    SplitWithScale,
    /// Gemma 4 post-attention norm: norm before residual, then separate pre-FFN norm.
    GemmaPostAttnNorm,
}

/// Pre-computed model-level execution plan.
///
/// Captures all architecture-specific decisions that would otherwise require
/// querying `ModelConfig`, `MetalConfig`, and `Gemma4Config` at every
/// pipeline iteration.
#[derive(Clone, Debug)]
pub(crate) struct ModelPlan {
    pub(crate) hidden_size: usize,
    pub(crate) num_attention_heads: u32,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f32,
    /// 1.0 for most models, sqrt(hidden_size) for Gemma.
    pub(crate) embed_scale: f32,
    pub(crate) layers: Vec<LayerPlan>,
    pub(crate) final_logit_softcapping: Option<f32>,
    pub(crate) enable_turboquant: bool,
    pub(crate) max_seq_len: usize,
    pub(crate) tq_n_bits: usize,
    pub(crate) ple: Option<PlePlan>,
    pub(crate) has_dac: bool,
}

impl ModelPlan {
    /// Build a model-level plan from config objects and pre-built layer plans.
    pub(crate) fn build(
        mc: &ModelConfig,
        config: &MetalConfig,
        g4: Option<&Gemma4Config>,
        layers: Vec<LayerPlan>,
    ) -> Self {
        let embed_scale = if mc.architecture == Architecture::Gemma {
            (mc.hidden_size as f32).sqrt()
        } else {
            1.0
        };

        let final_logit_softcapping = g4.and_then(|g| g.final_logit_softcapping);

        let ple = g4
            .filter(|g| g.ple_hidden_size > 0)
            .map(|g| PlePlan {
                ple_hidden_size: g.ple_hidden_size,
            });

        Self {
            hidden_size: mc.hidden_size,
            num_attention_heads: mc.num_attention_heads as u32,
            vocab_size: mc.vocab_size,
            rms_norm_eps: mc.rms_norm_eps as f32,
            embed_scale,
            layers,
            final_logit_softcapping,
            enable_turboquant: config.enable_turboquant,
            max_seq_len: config.max_seq_len,
            tq_n_bits: config.n_bits as usize,
            ple,
            has_dac: false, // Set after DAC calibration.
        }
    }
}
