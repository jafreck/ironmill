//! Shared model‐layout enumeration and weight‐loading driver used by the
//! Metal backend.
//!
//! Each backend implements [`WeightVisitor`] to materialize tensors into its
//! own buffer types; [`load_model_weights`] walks the standard LLaMA/Qwen
//! layout and calls the visitor for every tensor.

use std::borrow::Cow;

use mil_rs::ir::ScalarType;
use mil_rs::weights::{ModelConfig, QuantizationInfo, WeightProvider, WeightTensor};

use crate::dequant::dequant_affine_with_g_idx;

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
    /// Pre-FFN RMSNorm weight (Gemma 4 `pre_feedforward_layernorm`).
    pub pre_ffn_norm: Option<D>,
    /// Post-FFN RMSNorm weight (Gemma 4 `post_feedforward_layernorm`).
    pub post_ffn_norm: Option<D>,
    /// MoE router weight `[num_experts, hidden_size]`.
    pub router_weight: Option<W>,
    /// MoE expert gate projections `[moe_intermediate_size, hidden_size]` per expert.
    pub expert_gate_projs: Vec<W>,
    /// MoE expert up projections `[moe_intermediate_size, hidden_size]` per expert.
    pub expert_up_projs: Vec<W>,
    /// MoE expert down projections `[hidden_size, moe_intermediate_size]` per expert.
    pub expert_down_projs: Vec<W>,
    /// PLE gate weight `[ple_hidden_size, hidden_size]` (Gemma 4).
    pub ple_gate: Option<W>,
    /// PLE projection weight `[hidden_size, ple_hidden_size]` (Gemma 4).
    pub ple_projection: Option<W>,
    /// PLE post-norm weight `[hidden_size]` (Gemma 4).
    pub ple_post_norm: Option<D>,
    /// Per-layer output scalar (Gemma 4 `layer_scalar`).
    pub layer_scalar: Option<D>,
}

/// Core model weights returned by [`load_model_weights`].
///
/// Does **not** include `lm_head` because backends may handle it
/// differently.
pub struct LoadedModelCore<D, W> {
    /// Embedding table `[vocab_size, hidden_size]`.
    pub embedding: D,
    /// Per‐layer weights.
    pub layers: Vec<LoadedLayer<D, W>>,
    /// Final RMSNorm weight `[hidden_size]`.
    pub final_norm: D,
    /// Model configuration extracted from weight metadata.
    pub config: ModelConfig,

    /// PLE embedding table `[vocab_size, num_layers * ple_hidden_size]` (Gemma 4).
    pub ple_embed_tokens: Option<D>,
    /// PLE model projection weight `[num_layers * ple_hidden_size, hidden_size]` (Gemma 4).
    pub ple_model_projection: Option<W>,
    /// PLE projection norm weight `[num_layers * ple_hidden_size]` (Gemma 4).
    pub ple_projection_norm: Option<D>,
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
            pre_ffn_norm: {
                let name = format!("{prefix}.pre_feedforward_layernorm.weight");
                if provider.has_tensor(&name) {
                    Some(visitor.load_dense(provider, &name)?)
                } else {
                    None
                }
            },
            post_ffn_norm: {
                let name = format!("{prefix}.post_feedforward_layernorm.weight");
                if provider.has_tensor(&name) {
                    Some(visitor.load_dense(provider, &name)?)
                } else {
                    None
                }
            },
            router_weight: {
                let name = format!("{prefix}.mlp.router.weight");
                if provider.has_tensor(&name) {
                    Some(visitor.load_weight(provider, &name)?)
                } else {
                    None
                }
            },
            expert_gate_projs: {
                let mut projs = Vec::new();
                for e in 0.. {
                    let name = format!("{prefix}.mlp.experts.{e}.gate_proj.weight");
                    if !provider.has_tensor(&name) {
                        break;
                    }
                    projs.push(visitor.load_weight(provider, &name)?);
                }
                projs
            },
            expert_up_projs: {
                let mut projs = Vec::new();
                for e in 0.. {
                    let name = format!("{prefix}.mlp.experts.{e}.up_proj.weight");
                    if !provider.has_tensor(&name) {
                        break;
                    }
                    projs.push(visitor.load_weight(provider, &name)?);
                }
                projs
            },
            expert_down_projs: {
                let mut projs = Vec::new();
                for e in 0.. {
                    let name = format!("{prefix}.mlp.experts.{e}.down_proj.weight");
                    if !provider.has_tensor(&name) {
                        break;
                    }
                    projs.push(visitor.load_weight(provider, &name)?);
                }
                projs
            },
            ple_gate: {
                let name = format!("{prefix}.per_layer_input_gate.weight");
                if provider.has_tensor(&name) {
                    Some(visitor.load_weight(provider, &name)?)
                } else {
                    None
                }
            },
            ple_projection: {
                let name = format!("{prefix}.per_layer_projection.weight");
                if provider.has_tensor(&name) {
                    Some(visitor.load_weight(provider, &name)?)
                } else {
                    None
                }
            },
            ple_post_norm: {
                let name = format!("{prefix}.post_per_layer_input_norm.weight");
                if provider.has_tensor(&name) {
                    Some(visitor.load_dense(provider, &name)?)
                } else {
                    None
                }
            },
            layer_scalar: {
                let name = format!("{prefix}.layer_scalar");
                if provider.has_tensor(&name) {
                    Some(visitor.load_dense(provider, &name)?)
                } else {
                    None
                }
            },
        });
    }

    let final_norm = visitor.load_dense(provider, "model.norm.weight")?;

    // PLE model-level weights (Gemma 4).
    let ple_embed_name = "model.embed_tokens_per_layer.weight";
    let ple_embed_tokens = if provider.has_tensor(ple_embed_name) {
        Some(visitor.load_dense(provider, ple_embed_name)?)
    } else {
        None
    };
    let ple_proj_name = "model.per_layer_model_projection.weight";
    let ple_model_projection = if provider.has_tensor(ple_proj_name) {
        Some(visitor.load_weight(provider, ple_proj_name)?)
    } else {
        None
    };
    let ple_norm_name = "model.per_layer_projection_norm.weight";
    let ple_projection_norm = if provider.has_tensor(ple_norm_name) {
        Some(visitor.load_dense(provider, ple_norm_name)?)
    } else {
        None
    };

    Ok(LoadedModelCore {
        embedding,
        layers,
        final_norm,
        config,
        ple_embed_tokens,
        ple_model_projection,
        ple_projection_norm,
    })
}

// ── Shared helpers ──────────────────────────────────────────────

/// Extract `(rows, cols)` from a weight shape `[N, K]`.
///
/// For 1-D tensors, returns `(1, N)` — treating the weight as a single row.
pub(crate) fn dense_shape(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        1 => (1, shape[0]),
        _ => (shape[0], shape[1]),
    }
}

// ── CPU dequant dispatch ────────────────────────────────────────

/// Dense tensor data after optional CPU dequantization.
///
/// For unquantized tensors the raw bytes are borrowed via [`Cow::Borrowed`].
/// For quantized tensors the data is dequantized to FP16 and owned.
pub struct DenseData<'a> {
    /// Byte data — either borrowed (unquantized) or owned (dequantized FP16).
    pub bytes: Cow<'a, [u8]>,
    /// Tensor shape.
    #[allow(dead_code)]
    pub shape: &'a [usize],
    /// Element data type — original for unquantized, `Float16` for dequantized.
    #[allow(dead_code)]
    pub dtype: ScalarType,
}

/// Backend-specific CPU dequantization operations.
///
/// Each backend implements this to select its preferred routines for
/// LUT-palettized and affine-quantized formats.
#[allow(clippy::too_many_arguments)]
pub trait CpuDequant {
    /// Dequantize a LUT-encoded tensor to FP16 bytes.
    fn dequant_lut(
        indices: &[u8],
        lut: &[u8],
        lut_dtype: ScalarType,
        original_shape: &[usize],
        n_bits: u8,
        row_norms: &[u8],
        norms_dtype: ScalarType,
        polar_quant_seed: Option<u64>,
    ) -> anyhow::Result<Vec<u8>>;

    /// Dequantize an affine-quantized tensor to FP16 bytes.
    fn dequant_affine(
        data: &[u8],
        scale: &[u8],
        zero_point: &[u8],
        scale_dtype: ScalarType,
        zero_point_dtype: ScalarType,
        axis: Option<usize>,
        shape: &[usize],
        bit_width: u8,
        group_size: Option<usize>,
    ) -> anyhow::Result<Vec<u8>>;
}

/// Dequantize a [`WeightTensor`] to a dense representation using the
/// backend-specific routines from [`CpuDequant`].
///
/// Matches on [`QuantizationInfo`] and dispatches to the appropriate
/// dequantization method, returning borrowed raw bytes for unquantized
/// tensors or owned FP16 bytes for quantized tensors.
pub fn dequant_tensor_to_dense<'a, D: CpuDequant>(
    tensor: &'a WeightTensor<'a>,
) -> anyhow::Result<DenseData<'a>> {
    match &tensor.quant_info {
        QuantizationInfo::None => Ok(DenseData {
            bytes: Cow::Borrowed(&tensor.data),
            shape: &tensor.shape,
            dtype: tensor.dtype,
        }),
        QuantizationInfo::LutToDense {
            lut,
            lut_dtype,
            indices,
            original_shape,
            n_bits,
            row_norms,
            norms_dtype,
            polar_quant_seed,
            quip_sharp_seed,
        } => {
            if let Some(seed) = quip_sharp_seed {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    let data = crate::metal::dequant::dequant_quip_sharp(
                        indices,
                        lut,
                        *lut_dtype,
                        original_shape,
                        row_norms,
                        *norms_dtype,
                        *seed,
                    )?;
                    return Ok(DenseData {
                        bytes: Cow::Owned(data),
                        shape: original_shape,
                        dtype: ScalarType::Float16,
                    });
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    let _ = seed;
                    anyhow::bail!("QuIP# dequantization requires Metal backend");
                }
            }
            let data = D::dequant_lut(
                indices,
                lut,
                *lut_dtype,
                original_shape,
                *n_bits,
                row_norms,
                *norms_dtype,
                *polar_quant_seed,
            )?;
            Ok(DenseData {
                bytes: Cow::Owned(data),
                shape: original_shape,
                dtype: ScalarType::Float16,
            })
        }
        QuantizationInfo::AffineDequantize {
            scale,
            zero_point,
            scale_dtype,
            zero_point_dtype,
            axis,
            bit_width,
            group_size,
            g_idx,
            ..
        } => {
            let data = if g_idx.is_some() {
                // GPTQ act-order: use g_idx for group mapping instead of
                // the default col/group_size. Re-dequantize with correct
                // per-column group indices.
                dequant_affine_with_g_idx(
                    &tensor.data,
                    scale,
                    zero_point,
                    *scale_dtype,
                    *zero_point_dtype,
                    &tensor.shape,
                    *bit_width,
                    // Safety: guarded by `if g_idx.is_some()` above.
                    g_idx.as_deref().expect("g_idx checked above"),
                )?
            } else {
                D::dequant_affine(
                    &tensor.data,
                    scale,
                    zero_point,
                    *scale_dtype,
                    *zero_point_dtype,
                    *axis,
                    &tensor.shape,
                    *bit_width,
                    *group_size,
                )?
            };

            Ok(DenseData {
                bytes: Cow::Owned(data),
                shape: &tensor.shape,
                dtype: ScalarType::Float16,
            })
        }
        other => anyhow::bail!("unsupported quant_info variant: {other:?}"),
    }
}
