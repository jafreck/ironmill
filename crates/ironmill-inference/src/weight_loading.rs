//! Shared model‐layout enumeration and weight‐loading driver used by both
//! the Metal and MLX backends.
//!
//! Each backend implements [`WeightVisitor`] to materialize tensors into its
//! own buffer types; [`load_model_weights`] walks the standard LLaMA/Qwen
//! layout and calls the visitor for every tensor.

use mil_rs::weights::{ModelConfig, WeightProvider};

// ── Generic layer / model structs ───────────────────────────────

/// Weights for a single transformer layer, generic over backend types.
///
/// `D` is the *dense* type (always‐dequantized, e.g. norm weights) and
/// `W` is the *weight buffer* type (may remain quantized).
pub struct LoadedLayer<D, W> {
    /// Input layernorm weight `[hidden_size]`.
    pub input_norm: D,
    /// Q projection `[num_heads × head_dim, hidden_size]`.
    pub q_proj: W,
    /// K projection `[num_kv_heads × head_dim, hidden_size]`.
    pub k_proj: W,
    /// V projection `[num_kv_heads × head_dim, hidden_size]`.
    pub v_proj: W,
    /// Output projection `[hidden_size, num_heads × head_dim]`.
    pub o_proj: W,
    /// Post‐attention layernorm weight `[hidden_size]`.
    pub post_attn_norm: D,
    /// Gate projection `[intermediate_size, hidden_size]`.
    pub gate_proj: W,
    /// Up projection `[intermediate_size, hidden_size]`.
    pub up_proj: W,
    /// Down projection `[hidden_size, intermediate_size]`.
    pub down_proj: W,
    /// Optional Q normalization weight `[head_dim]` (Qwen3 QK norm).
    pub q_norm: Option<D>,
    /// Optional K normalization weight `[head_dim]` (Qwen3 QK norm).
    pub k_norm: Option<D>,
}

/// Core model weights returned by [`load_model_weights`].
///
/// Does **not** include `lm_head` because the two backends handle it
/// differently (Metal always dequantizes; MLX may keep it quantized).
pub struct LoadedModelCore<D, W> {
    /// Embedding table `[vocab_size, hidden_size]`.
    pub embedding: D,
    /// Per‐layer weights.
    pub layers: Vec<LoadedLayer<D, W>>,
    /// Final RMSNorm weight `[hidden_size]`.
    pub final_norm: D,
    /// Model configuration extracted from weight metadata.
    pub config: ModelConfig,
}

// ── Visitor trait ────────────────────────────────────────────────

/// Backends implement this to turn raw tensors into device‐specific buffers.
pub trait WeightVisitor {
    /// Dense (always‐dequantized) type — used for norms, embeddings, etc.
    type Dense;
    /// Weight buffer type — may stay in a quantized representation.
    type Weight;
    /// Backend error type.
    type Error;

    /// Load a tensor and fully dequantize it to a dense representation.
    fn load_dense(
        &self,
        provider: &dyn WeightProvider,
        name: &str,
    ) -> Result<Self::Dense, Self::Error>;

    /// Load a tensor, potentially keeping its quantized representation.
    fn load_weight(
        &self,
        provider: &dyn WeightProvider,
        name: &str,
    ) -> Result<Self::Weight, Self::Error>;
}

// ── Layout driver ───────────────────────────────────────────────

/// Walk the standard transformer model layout and load every tensor via the
/// supplied [`WeightVisitor`].
pub fn load_model_weights<V: WeightVisitor>(
    visitor: &V,
    provider: &dyn WeightProvider,
) -> Result<LoadedModelCore<V::Dense, V::Weight>, V::Error> {
    let config = provider.config().clone();
    let num_layers = config.num_hidden_layers;

    let embedding = visitor.load_dense(provider, "model.embed_tokens.weight")?;

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let prefix = format!("model.layers.{i}");

        let q_norm_name = format!("{prefix}.self_attn.q_norm.weight");
        let k_norm_name = format!("{prefix}.self_attn.k_norm.weight");

        layers.push(LoadedLayer {
            input_norm: visitor
                .load_dense(provider, &format!("{prefix}.input_layernorm.weight"))?,
            q_proj: visitor.load_weight(provider, &format!("{prefix}.self_attn.q_proj.weight"))?,
            k_proj: visitor.load_weight(provider, &format!("{prefix}.self_attn.k_proj.weight"))?,
            v_proj: visitor.load_weight(provider, &format!("{prefix}.self_attn.v_proj.weight"))?,
            o_proj: visitor.load_weight(provider, &format!("{prefix}.self_attn.o_proj.weight"))?,
            post_attn_norm: visitor.load_dense(
                provider,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            gate_proj: visitor.load_weight(provider, &format!("{prefix}.mlp.gate_proj.weight"))?,
            up_proj: visitor.load_weight(provider, &format!("{prefix}.mlp.up_proj.weight"))?,
            down_proj: visitor.load_weight(provider, &format!("{prefix}.mlp.down_proj.weight"))?,
            q_norm: if provider.has_tensor(&q_norm_name) {
                Some(visitor.load_dense(provider, &q_norm_name)?)
            } else {
                None
            },
            k_norm: if provider.has_tensor(&k_norm_name) {
                Some(visitor.load_dense(provider, &k_norm_name)?)
            } else {
                None
            },
        });
    }

    let final_norm = visitor.load_dense(provider, "model.norm.weight")?;

    Ok(LoadedModelCore {
        embedding,
        layers,
        final_norm,
        config,
    })
}
