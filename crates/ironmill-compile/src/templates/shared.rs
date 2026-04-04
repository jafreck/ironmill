//! Shared helper functions for architecture templates.
//!
//! These primitives are used by all architecture templates (LLaMA, Qwen, Gemma)
//! to emit common MIL IR patterns: weight constants, linear projections,
//! normalization, RoPE, GQA expansion, etc.

use crate::weights::{ModelConfig, WeightProvider};
use mil_rs::MilError;
use mil_rs::ir::{Block, Operation, ScalarType, TensorData, Value};

/// Immutable context shared across layer-level template emission.
///
/// Groups the read-only state that is threaded through every layer helper,
/// keeping function signatures short.
pub(super) struct LayerContext<'a> {
    pub provider: &'a dyn WeightProvider,
    pub config: &'a ModelConfig,
    pub layer_idx: usize,
    pub rope_cos: &'a str,
    pub rope_sin: &'a str,
}

// ---------------------------------------------------------------------------
// Weight / constant emission
// ---------------------------------------------------------------------------

/// Tensors smaller than this are stored inline to avoid indirection overhead.
const INLINE_THRESHOLD_BYTES: usize = 4096;

/// Emit a weight tensor as a `const` op in the block.
///
/// If the weight is missing, a placeholder const is emitted and a warning
/// is pushed so the graph remains structurally valid.
pub(super) fn emit_weight_const(
    block: &mut Block,
    provider: &dyn WeightProvider,
    weight_name: &str,
    const_name: &str,
    _warnings: &mut Vec<String>,
) -> Result<(), MilError> {
    match provider.tensor(weight_name) {
        Ok(tensor) => {
            let data = if tensor.data.len() <= INLINE_THRESHOLD_BYTES {
                // Small tensors (norms, biases): inline to avoid indirection.
                TensorData::inline(tensor.data.into_owned())
            } else {
                // Large tensors (projections, embeddings): defer loading.
                TensorData::external(weight_name.to_string(), tensor.data.len())
            };
            let op = Operation::new("const", const_name)
                .with_attr(
                    "val",
                    Value::Tensor {
                        data,
                        shape: tensor.shape.clone(),
                        dtype: tensor.dtype,
                    },
                )
                .with_attr("onnx_name", Value::String(weight_name.to_string()))
                .with_output(const_name);
            block.add_op(op);
            Ok(())
        }
        Err(e) => {
            _warnings.push(format!("missing weight '{weight_name}': {e}"));
            let data = TensorData::inline(vec![0u8; 4]);
            let op = Operation::new("const", const_name)
                .with_attr(
                    "val",
                    Value::Tensor {
                        data,
                        shape: vec![1],
                        dtype: ScalarType::Float32,
                    },
                )
                .with_attr("onnx_name", Value::String(weight_name.to_string()))
                .with_output(const_name);
            block.add_op(op);
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Linear projection
// ---------------------------------------------------------------------------

/// Emit a `linear` op (weight projection). Loads weight (and optional bias)
/// from the provider using HuggingFace naming convention.
pub(super) fn emit_linear(
    block: &mut Block,
    provider: &dyn WeightProvider,
    weight_prefix: &str,
    input: &str,
    op_prefix: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let weight_name = format!("{weight_prefix}.weight");
    let weight_const = format!("{op_prefix}_weight");
    emit_weight_const(block, provider, &weight_name, &weight_const, warnings)?;

    let bias_name = format!("{weight_prefix}.bias");
    let has_bias = provider.has_tensor(&bias_name);
    if has_bias {
        let bias_const = format!("{op_prefix}_bias");
        emit_weight_const(block, provider, &bias_name, &bias_const, warnings)?;
    }

    let out_name = format!("{op_prefix}_out");
    let mut op = Operation::new("linear", format!("{op_prefix}_op"))
        .with_input("x", Value::Reference(input.into()))
        .with_input("weight", Value::Reference(weight_const));

    if has_bias {
        op = op.with_input("bias", Value::Reference(format!("{op_prefix}_bias")));
    }

    op = op.with_output(&out_name);
    block.add_op(op);

    Ok(out_name)
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Emit an `rms_norm` op with its weight loaded from the provider.
pub(super) fn emit_rms_norm(
    block: &mut Block,
    provider: &dyn WeightProvider,
    config: &ModelConfig,
    weight_prefix: &str,
    input: &str,
    op_prefix: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let weight_name = format!("{weight_prefix}.weight");
    let weight_const = format!("{op_prefix}_weight");
    emit_weight_const(block, provider, &weight_name, &weight_const, warnings)?;

    let out_name = format!("{op_prefix}_out");
    let op = Operation::new("rms_norm", format!("{op_prefix}_op"))
        .with_input("x", Value::Reference(input.into()))
        .with_input("weight", Value::Reference(weight_const))
        .with_attr("epsilon", Value::Float(config.rms_norm_eps))
        .with_output(&out_name);
    block.add_op(op);

    Ok(out_name)
}

// ---------------------------------------------------------------------------
// Residual connection
// ---------------------------------------------------------------------------

/// Emit an `add` op for residual connections.
pub(super) fn emit_residual_add(block: &mut Block, x: &str, y: &str, out_name: &str) -> String {
    let op = Operation::new("add", format!("{out_name}_op"))
        .with_input("x", Value::Reference(x.into()))
        .with_input("y", Value::Reference(y.into()))
        .with_output(out_name);
    block.add_op(op);
    out_name.to_string()
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

/// Emit a `reshape` op.
pub(super) fn emit_reshape(block: &mut Block, input: &str, op_name: &str, shape: &[i64]) -> String {
    let shape_const_name = format!("{op_name}_shape");
    let shape_val = Value::List(shape.iter().map(|&d| Value::Int(d)).collect());
    let shape_op = Operation::new("const", &shape_const_name)
        .with_attr("val", shape_val)
        .with_output(&shape_const_name);
    block.add_op(shape_op);

    let out_name = format!("{op_name}_out");
    let op = Operation::new("reshape", format!("{op_name}_op"))
        .with_input("x", Value::Reference(input.into()))
        .with_input("shape", Value::Reference(shape_const_name))
        .with_output(&out_name);
    block.add_op(op);
    out_name
}

/// Emit a `transpose` op with the given permutation.
pub(super) fn emit_transpose(
    block: &mut Block,
    input: &str,
    op_name: &str,
    perm: &[i64],
) -> String {
    let perm_val = Value::List(perm.iter().map(|&d| Value::Int(d)).collect());

    let out_name = format!("{op_name}_out");
    let op = Operation::new("transpose", format!("{op_name}_op"))
        .with_input("x", Value::Reference(input.into()))
        .with_attr("perm", perm_val)
        .with_output(&out_name);
    block.add_op(op);
    out_name
}

// ---------------------------------------------------------------------------
// Rotary Position Embeddings (RoPE)
// ---------------------------------------------------------------------------

/// Precompute cos/sin frequency tables as const tensors for RoPE.
/// Returns `(cos_table_name, sin_table_name)` with shape
/// `[max_position_embeddings, head_dim / 2]` each.
pub(super) fn emit_rope_tables(block: &mut Block, config: &ModelConfig) -> (String, String) {
    let head_dim = config.head_dim;
    let max_pos = config.max_position_embeddings;
    let theta = config.rope_theta;
    let half_dim = head_dim / 2;

    let mut cos_bytes = Vec::with_capacity(max_pos * half_dim * 4);
    let mut sin_bytes = Vec::with_capacity(max_pos * half_dim * 4);

    for t in 0..max_pos {
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = t as f64 * freq;
            cos_bytes.extend_from_slice(&(angle.cos() as f32).to_le_bytes());
            sin_bytes.extend_from_slice(&(angle.sin() as f32).to_le_bytes());
        }
    }

    let cos_name = "rope_cos_table".to_string();
    let sin_name = "rope_sin_table".to_string();

    let cos_op = Operation::new("const", &cos_name)
        .with_attr(
            "val",
            Value::Tensor {
                data: TensorData::inline(cos_bytes),
                shape: vec![max_pos, half_dim],
                dtype: ScalarType::Float32,
            },
        )
        .with_output(&cos_name);
    block.add_op(cos_op);

    let sin_op = Operation::new("const", &sin_name)
        .with_attr(
            "val",
            Value::Tensor {
                data: TensorData::inline(sin_bytes),
                shape: vec![max_pos, half_dim],
                dtype: ScalarType::Float32,
            },
        )
        .with_output(&sin_name);
    block.add_op(sin_op);

    (cos_name, sin_name)
}

/// Apply rotary position embeddings to Q and K tensors.
/// Both inputs have shape `[batch, heads, seq, head_dim]`.
/// Returns `(q_roped, k_roped)` with the same shapes.
pub(super) fn emit_rotary_embedding(
    block: &mut Block,
    config: &ModelConfig,
    q_name: &str,
    k_name: &str,
    layer_idx: usize,
    cos_table: &str,
    sin_table: &str,
) -> (String, String) {
    let prefix = format!("l{layer_idx}_rope");
    let half_dim = config.head_dim / 2;

    // Gather cos/sin using position_ids
    let cos_gathered = {
        let out_name = format!("{prefix}_cos_gathered");
        let op = Operation::new("gather", format!("{prefix}_cos_gather_op"))
            .with_input("x", Value::Reference(cos_table.into()))
            .with_input("indices", Value::Reference("position_ids".into()))
            .with_attr("axis", Value::Int(0))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    let sin_gathered = {
        let out_name = format!("{prefix}_sin_gathered");
        let op = Operation::new("gather", format!("{prefix}_sin_gather_op"))
            .with_input("x", Value::Reference(sin_table.into()))
            .with_input("indices", Value::Reference("position_ids".into()))
            .with_attr("axis", Value::Int(0))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Reshape to [batch, 1, seq, half_dim] for broadcasting over heads
    let cos_reshaped = emit_reshape(
        block,
        &cos_gathered,
        &format!("{prefix}_cos_reshape"),
        &[-1, 1, -1, half_dim as i64],
    );

    let sin_reshaped = emit_reshape(
        block,
        &sin_gathered,
        &format!("{prefix}_sin_reshape"),
        &[-1, 1, -1, half_dim as i64],
    );

    let q_rope = emit_rope_apply(
        block,
        q_name,
        &cos_reshaped,
        &sin_reshaped,
        &format!("{prefix}_q"),
    );
    let k_rope = emit_rope_apply(
        block,
        k_name,
        &cos_reshaped,
        &sin_reshaped,
        &format!("{prefix}_k"),
    );

    (q_rope, k_rope)
}

/// Apply the RoPE rotation to a single tensor of shape `[batch, heads, seq, head_dim]`.
pub(super) fn emit_rope_apply(
    block: &mut Block,
    input: &str,
    cos: &str,
    sin: &str,
    prefix: &str,
) -> String {
    // Split into first and second halves along head_dim
    let half1 = format!("{prefix}_half1");
    let half2 = format!("{prefix}_half2");
    let split_op = Operation::new("split", format!("{prefix}_split_op"))
        .with_input("x", Value::Reference(input.into()))
        .with_attr("num_splits", Value::Int(2))
        .with_attr("axis", Value::Int(-1))
        .with_output(&half1)
        .with_output(&half2);
    block.add_op(split_op);

    // half1 * cos
    let h1_cos = {
        let out_name = format!("{prefix}_h1_cos");
        let op = Operation::new("mul", format!("{prefix}_h1_cos_op"))
            .with_input("x", Value::Reference(half1.clone()))
            .with_input("y", Value::Reference(cos.into()))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // half2 * sin
    let h2_sin = {
        let out_name = format!("{prefix}_h2_sin");
        let op = Operation::new("mul", format!("{prefix}_h2_sin_op"))
            .with_input("x", Value::Reference(half2.clone()))
            .with_input("y", Value::Reference(sin.into()))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // half1 * sin
    let h1_sin = {
        let out_name = format!("{prefix}_h1_sin");
        let op = Operation::new("mul", format!("{prefix}_h1_sin_op"))
            .with_input("x", Value::Reference(half1))
            .with_input("y", Value::Reference(sin.into()))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // half2 * cos
    let h2_cos = {
        let out_name = format!("{prefix}_h2_cos");
        let op = Operation::new("mul", format!("{prefix}_h2_cos_op"))
            .with_input("x", Value::Reference(half2))
            .with_input("y", Value::Reference(cos.into()))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // part1 = half1 * cos - half2 * sin
    let part1 = {
        let out_name = format!("{prefix}_rope_part1");
        let op = Operation::new("sub", format!("{prefix}_rope_sub_op"))
            .with_input("x", Value::Reference(h1_cos))
            .with_input("y", Value::Reference(h2_sin))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // part2 = half1 * sin + half2 * cos
    let part2 = {
        let out_name = format!("{prefix}_rope_part2");
        let op = Operation::new("add", format!("{prefix}_rope_add_op"))
            .with_input("x", Value::Reference(h1_sin))
            .with_input("y", Value::Reference(h2_cos))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Concatenate back along head_dim
    {
        let out_name = format!("{prefix}_rope_out");
        let op = Operation::new("concat", format!("{prefix}_rope_concat_op"))
            .with_input(
                "values",
                Value::List(vec![Value::Reference(part1), Value::Reference(part2)]),
            )
            .with_attr("axis", Value::Int(-1))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    }
}

// ---------------------------------------------------------------------------
// Grouped Query Attention (GQA) head expansion
// ---------------------------------------------------------------------------

/// Expand KV heads to match the number of attention heads for GQA.
/// Input shape: `[batch, num_kv_heads, seq, head_dim]`.
/// Output shape: `[batch, num_attention_heads, seq, head_dim]`.
pub(super) fn emit_gqa_expand(
    block: &mut Block,
    input: &str,
    n_rep: usize,
    config: &ModelConfig,
    layer_idx: usize,
    name: &str,
) -> String {
    let prefix = format!("l{layer_idx}_{name}_gqa");

    // Reshape to [batch, num_kv_heads, 1, seq, head_dim]
    let reshaped_5d = emit_reshape(
        block,
        input,
        &format!("{prefix}_reshape5d"),
        &[
            -1,
            config.num_key_value_heads as i64,
            1,
            -1,
            config.head_dim as i64,
        ],
    );

    // Tile along dim 2 by n_rep
    let tiled = {
        let reps_const = format!("{prefix}_tile_reps");
        let reps_val = Value::List(vec![
            Value::Int(1),
            Value::Int(1),
            Value::Int(n_rep as i64),
            Value::Int(1),
            Value::Int(1),
        ]);
        let op = Operation::new("const", &reps_const)
            .with_attr("val", reps_val)
            .with_output(&reps_const);
        block.add_op(op);

        let out_name = format!("{prefix}_tiled");
        let op = Operation::new("tile", format!("{prefix}_tile_op"))
            .with_input("x", Value::Reference(reshaped_5d))
            .with_input("reps", Value::Reference(reps_const))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Reshape back to [batch, num_attention_heads, seq, head_dim]
    emit_reshape(
        block,
        &tiled,
        &format!("{prefix}_reshape4d"),
        &[
            -1,
            config.num_attention_heads as i64,
            -1,
            config.head_dim as i64,
        ],
    )
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Emit the token embedding lookup: const weight + gather.
pub(super) fn emit_embedding(
    block: &mut Block,
    provider: &dyn WeightProvider,
    _config: &ModelConfig,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let weight_name = "model.embed_tokens.weight";
    let const_name = "embed_tokens_weight";
    emit_weight_const(block, provider, weight_name, const_name, warnings)?;

    let out_name = "embed_out".to_string();
    let gather = Operation::new("gather", "embed_gather")
        .with_input("x", Value::Reference(const_name.into()))
        .with_input("indices", Value::Reference("input_ids".into()))
        .with_attr("axis", Value::Int(0))
        .with_output(&out_name);
    block.add_op(gather);

    Ok(out_name)
}

// ---------------------------------------------------------------------------
// LM head
// ---------------------------------------------------------------------------

/// Emit the language model head projection.
///
/// If `tie_word_embeddings` is true, reuses the embedding weight that was
/// already emitted as `embed_tokens_weight`.
pub(super) fn emit_lm_head(
    block: &mut Block,
    provider: &dyn WeightProvider,
    config: &ModelConfig,
    input: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    if config.tie_word_embeddings {
        let out_name = "lm_head_out".to_string();
        let op = Operation::new("linear", "lm_head_op")
            .with_input("x", Value::Reference(input.into()))
            .with_input("weight", Value::Reference("embed_tokens_weight".into()))
            .with_output(&out_name);
        block.add_op(op);
        Ok(out_name)
    } else {
        emit_linear(block, provider, "lm_head", input, "lm_head", warnings)
    }
}

// ---------------------------------------------------------------------------
// Attention (standard LLaMA-style, reused by Qwen/Gemma)
// ---------------------------------------------------------------------------

/// Emit the standard self-attention block with Q/K/V projections, RoPE, GQA,
/// scaled dot-product attention, and output projection.
pub(super) fn emit_attention(
    block: &mut Block,
    ctx: &LayerContext<'_>,
    input: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{}.self_attn", ctx.layer_idx);

    // Q/K/V projections
    let q = emit_linear(
        block,
        ctx.provider,
        &format!("{prefix}.q_proj"),
        input,
        &format!("l{}_q_proj", ctx.layer_idx),
        warnings,
    )?;
    let k = emit_linear(
        block,
        ctx.provider,
        &format!("{prefix}.k_proj"),
        input,
        &format!("l{}_k_proj", ctx.layer_idx),
        warnings,
    )?;
    let v = emit_linear(
        block,
        ctx.provider,
        &format!("{prefix}.v_proj"),
        input,
        &format!("l{}_v_proj", ctx.layer_idx),
        warnings,
    )?;

    emit_attention_core(block, ctx, &prefix, &q, &k, &v, warnings)
}

/// Core attention computation shared across architectures.
/// Takes already-projected Q, K, V and emits reshape, RoPE, GQA, scaled dot-product, output projection.
pub(super) fn emit_attention_core(
    block: &mut Block,
    ctx: &LayerContext<'_>,
    prefix: &str,
    q: &str,
    k: &str,
    v: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let layer_idx = ctx.layer_idx;
    let config = ctx.config;

    // Reshape Q for multi-head: [batch, seq, hidden] -> [batch, seq, n_heads, head_dim]
    let q_reshaped = emit_reshape(
        block,
        q,
        &format!("l{layer_idx}_q_reshape"),
        &[
            -1,
            -1,
            config.num_attention_heads as i64,
            config.head_dim as i64,
        ],
    );

    let k_reshaped = emit_reshape(
        block,
        k,
        &format!("l{layer_idx}_k_reshape"),
        &[
            -1,
            -1,
            config.num_key_value_heads as i64,
            config.head_dim as i64,
        ],
    );

    let v_reshaped = emit_reshape(
        block,
        v,
        &format!("l{layer_idx}_v_reshape"),
        &[
            -1,
            -1,
            config.num_key_value_heads as i64,
            config.head_dim as i64,
        ],
    );

    // Transpose to [batch, n_heads, seq, head_dim]
    let q_t = emit_transpose(
        block,
        &q_reshaped,
        &format!("l{layer_idx}_q_transpose"),
        &[0, 2, 1, 3],
    );
    let k_t = emit_transpose(
        block,
        &k_reshaped,
        &format!("l{layer_idx}_k_transpose"),
        &[0, 2, 1, 3],
    );
    let v_t = emit_transpose(
        block,
        &v_reshaped,
        &format!("l{layer_idx}_v_transpose"),
        &[0, 2, 1, 3],
    );

    // Apply RoPE to Q and K
    let (q_roped, k_roped) = emit_rotary_embedding(
        block,
        config,
        &q_t,
        &k_t,
        layer_idx,
        ctx.rope_cos,
        ctx.rope_sin,
    );

    // GQA: expand K and V heads when using grouped query attention
    let (k_expanded, v_expanded) = if config.num_attention_heads != config.num_key_value_heads {
        if config.num_attention_heads % config.num_key_value_heads != 0 {
            return Err(MilError::Validation(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                config.num_attention_heads, config.num_key_value_heads
            )));
        }
        let n_rep = config.num_attention_heads / config.num_key_value_heads;
        let k_exp = emit_gqa_expand(block, &k_roped, n_rep, config, layer_idx, "k");
        let v_exp = emit_gqa_expand(block, &v_t, n_rep, config, layer_idx, "v");
        (k_exp, v_exp)
    } else {
        (k_roped, v_t)
    };

    // Transpose K for matmul: [batch, n_heads, head_dim, seq]
    let k_t2 = emit_transpose(
        block,
        &k_expanded,
        &format!("l{layer_idx}_k_transpose2"),
        &[0, 1, 3, 2],
    );

    // QK^T matmul
    let qk = {
        let out_name = format!("l{layer_idx}_qk_matmul");
        let op = Operation::new("matmul", format!("l{layer_idx}_qk_matmul_op"))
            .with_input("x", Value::Reference(q_roped))
            .with_input("y", Value::Reference(k_t2))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Scale by 1/sqrt(head_dim)
    let scale = {
        let scale_val = 1.0 / (config.head_dim as f64).sqrt();
        let scale_const = format!("l{layer_idx}_attn_scale_const");
        let op = Operation::new("const", &scale_const)
            .with_attr("val", Value::Float(scale_val))
            .with_output(&scale_const);
        block.add_op(op);

        let out_name = format!("l{layer_idx}_attn_scaled");
        let op = Operation::new("mul", format!("l{layer_idx}_attn_scale_op"))
            .with_input("x", Value::Reference(qk))
            .with_input("y", Value::Reference(scale_const))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Apply causal mask
    let masked = {
        let out_name = format!("l{layer_idx}_attn_masked");
        let op = Operation::new("add", format!("l{layer_idx}_attn_mask_op"))
            .with_input("x", Value::Reference(scale))
            .with_input("y", Value::Reference("causal_mask".into()))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Softmax
    let attn_weights = {
        let out_name = format!("l{layer_idx}_attn_weights");
        let op = Operation::new("softmax", format!("l{layer_idx}_softmax_op"))
            .with_input("x", Value::Reference(masked))
            .with_attr("axis", Value::Int(-1))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Attention @ V
    let attn_output = {
        let out_name = format!("l{layer_idx}_attn_output");
        let op = Operation::new("matmul", format!("l{layer_idx}_av_matmul_op"))
            .with_input("x", Value::Reference(attn_weights))
            .with_input("y", Value::Reference(v_expanded))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Transpose back: [batch, n_heads, seq, head_dim] -> [batch, seq, n_heads, head_dim]
    let attn_transposed = emit_transpose(
        block,
        &attn_output,
        &format!("l{layer_idx}_attn_out_transpose"),
        &[0, 2, 1, 3],
    );

    // Reshape to [batch, seq, hidden]
    let attn_flat = emit_reshape(
        block,
        &attn_transposed,
        &format!("l{layer_idx}_attn_out_reshape"),
        &[-1, -1, config.hidden_size as i64],
    );

    // Output projection
    let o_proj = emit_linear(
        block,
        ctx.provider,
        &format!("{prefix}.o_proj"),
        &attn_flat,
        &format!("l{layer_idx}_o_proj"),
        warnings,
    )?;

    Ok(o_proj)
}

// ---------------------------------------------------------------------------
// MLP (SiLU variant — used by LLaMA, Qwen)
// ---------------------------------------------------------------------------

/// Emit the SwiGLU MLP: gate + up + silu + down.
pub(super) fn emit_mlp_silu(
    block: &mut Block,
    provider: &dyn WeightProvider,
    _config: &ModelConfig,
    layer_idx: usize,
    input: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{layer_idx}.mlp");

    let gate = emit_linear(
        block,
        provider,
        &format!("{prefix}.gate_proj"),
        input,
        &format!("l{layer_idx}_gate_proj"),
        warnings,
    )?;

    let up = emit_linear(
        block,
        provider,
        &format!("{prefix}.up_proj"),
        input,
        &format!("l{layer_idx}_up_proj"),
        warnings,
    )?;

    // SiLU activation on gate
    let gate_act = {
        let out_name = format!("l{layer_idx}_gate_silu");
        let op = Operation::new("silu", format!("l{layer_idx}_silu_op"))
            .with_input("x", Value::Reference(gate))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Element-wise multiply gate_act * up
    let mlp_hidden = {
        let out_name = format!("l{layer_idx}_mlp_hidden");
        let op = Operation::new("mul", format!("l{layer_idx}_mlp_mul_op"))
            .with_input("x", Value::Reference(gate_act))
            .with_input("y", Value::Reference(up))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    let down = emit_linear(
        block,
        provider,
        &format!("{prefix}.down_proj"),
        &mlp_hidden,
        &format!("l{layer_idx}_down_proj"),
        warnings,
    )?;

    Ok(down)
}

/// Emit the GeGLU MLP: gate + up + gelu + down (used by Gemma).
pub(super) fn emit_mlp_gelu(
    block: &mut Block,
    provider: &dyn WeightProvider,
    _config: &ModelConfig,
    layer_idx: usize,
    input: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{layer_idx}.mlp");

    let gate = emit_linear(
        block,
        provider,
        &format!("{prefix}.gate_proj"),
        input,
        &format!("l{layer_idx}_gate_proj"),
        warnings,
    )?;

    let up = emit_linear(
        block,
        provider,
        &format!("{prefix}.up_proj"),
        input,
        &format!("l{layer_idx}_up_proj"),
        warnings,
    )?;

    // GELU activation on gate
    let gate_act = {
        let out_name = format!("l{layer_idx}_gate_gelu");
        let op = Operation::new("gelu", format!("l{layer_idx}_gelu_op"))
            .with_input("x", Value::Reference(gate))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Element-wise multiply gate_act * up
    let mlp_hidden = {
        let out_name = format!("l{layer_idx}_mlp_hidden");
        let op = Operation::new("mul", format!("l{layer_idx}_mlp_mul_op"))
            .with_input("x", Value::Reference(gate_act))
            .with_input("y", Value::Reference(up))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    let down = emit_linear(
        block,
        provider,
        &format!("{prefix}.down_proj"),
        &mlp_hidden,
        &format!("l{layer_idx}_down_proj"),
        warnings,
    )?;

    Ok(down)
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Minimal stub provider for testing template construction.
#[cfg(test)]
pub(super) struct StubProvider {
    pub config: ModelConfig,
    pub tensors: std::collections::HashMap<String, Vec<u8>>,
}

#[cfg(test)]
impl StubProvider {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            tensors: std::collections::HashMap::new(),
        }
    }

    pub fn with_tensor(mut self, name: &str, shape: &[usize], dtype: ScalarType) -> Self {
        let byte_size: usize = shape.iter().product::<usize>() * dtype.byte_size();
        self.tensors.insert(name.to_string(), vec![0u8; byte_size]);
        self
    }

    /// Populate all standard LLaMA weight names for the given config.
    pub fn with_llama_weights(mut self) -> Self {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let n_heads = self.config.num_attention_heads;
        let n_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let vocab = self.config.vocab_size;
        let num_layers = self.config.num_hidden_layers;
        let tie = self.config.tie_word_embeddings;

        self = self.with_tensor(
            "model.embed_tokens.weight",
            &[vocab, h],
            ScalarType::Float16,
        );

        for l in 0..num_layers {
            let p = format!("model.layers.{l}");
            self = self.with_tensor(
                &format!("{p}.self_attn.q_proj.weight"),
                &[n_heads * head_dim, h],
                ScalarType::Float16,
            );
            self = self.with_tensor(
                &format!("{p}.self_attn.k_proj.weight"),
                &[n_kv_heads * head_dim, h],
                ScalarType::Float16,
            );
            self = self.with_tensor(
                &format!("{p}.self_attn.v_proj.weight"),
                &[n_kv_heads * head_dim, h],
                ScalarType::Float16,
            );
            self = self.with_tensor(
                &format!("{p}.self_attn.o_proj.weight"),
                &[h, n_heads * head_dim],
                ScalarType::Float16,
            );
            self = self.with_tensor(
                &format!("{p}.input_layernorm.weight"),
                &[h],
                ScalarType::Float16,
            );
            self = self.with_tensor(
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
                ScalarType::Float16,
            );
            self = self.with_tensor(
                &format!("{p}.mlp.gate_proj.weight"),
                &[inter, h],
                ScalarType::Float16,
            );
            self = self.with_tensor(
                &format!("{p}.mlp.up_proj.weight"),
                &[inter, h],
                ScalarType::Float16,
            );
            self = self.with_tensor(
                &format!("{p}.mlp.down_proj.weight"),
                &[h, inter],
                ScalarType::Float16,
            );
        }

        self = self.with_tensor("model.norm.weight", &[h], ScalarType::Float16);
        if !tie {
            self = self.with_tensor("lm_head.weight", &[vocab, h], ScalarType::Float16);
        }

        self
    }

    /// Add attention bias tensors for all layers (Qwen-style).
    pub fn with_attention_biases(mut self) -> Self {
        let _h = self.config.hidden_size;
        let n_heads = self.config.num_attention_heads;
        let n_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let num_layers = self.config.num_hidden_layers;

        for l in 0..num_layers {
            let p = format!("model.layers.{l}");
            self = self.with_tensor(
                &format!("{p}.self_attn.q_proj.bias"),
                &[n_heads * head_dim],
                ScalarType::Float16,
            );
            self = self.with_tensor(
                &format!("{p}.self_attn.k_proj.bias"),
                &[n_kv_heads * head_dim],
                ScalarType::Float16,
            );
            self = self.with_tensor(
                &format!("{p}.self_attn.v_proj.bias"),
                &[n_kv_heads * head_dim],
                ScalarType::Float16,
            );
        }

        self
    }
}

#[cfg(test)]
impl WeightProvider for StubProvider {
    fn tensor(&self, name: &str) -> Result<crate::weights::WeightTensor<'_>, MilError> {
        let data = self
            .tensors
            .get(name)
            .ok_or_else(|| MilError::Validation(format!("tensor not found: {name}")))?;
        let elem_size = ScalarType::Float16.byte_size();
        let num_elements = data.len() / elem_size;
        Ok(crate::weights::WeightTensor::borrowed(
            data,
            vec![num_elements],
            ScalarType::Float16,
        ))
    }

    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

/// Create a tiny model config suitable for unit tests.
#[cfg(test)]
pub(super) fn tiny_llama_config() -> ModelConfig {
    use crate::weights::Architecture;
    ModelConfig {
        architecture: Architecture::Llama,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        head_dim: 16,
        vocab_size: 256,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        tie_word_embeddings: false,
        extra: std::collections::HashMap::new(),
    }
}
