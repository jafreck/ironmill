//! LLaMA architecture template.
//!
//! Builds a MIL IR [`Program`] for the LLaMA model family (LLaMA 2/3,
//! CodeLlama, Mistral, etc.) from weight tensors provided by a
//! [`WeightProvider`].
//!
//! The generated graph uses high-level ops (`linear`, `rms_norm`, `silu`)
//! that downstream optimization passes can lower to ANE-friendly forms.
//!
//! When [`TemplateOptions::ane`] is set, the template emits ANE-optimized ops
//! directly: 1×1 convolutions instead of linear projections, decomposed
//! RMSNorm, static KV-cache state inputs, and a prefill/decode function split.

use crate::templates::TemplateOptions;
use crate::weights::{ModelConfig, WeightProvider};
use mil_rs::MilError;
use mil_rs::convert::onnx_graph::ConversionResult;
use mil_rs::ir::{Block, Function, Operation, Program, ScalarType, TensorType, Value};

use super::shared::{
    LayerContext, emit_embedding, emit_gqa_expand, emit_linear, emit_reshape, emit_residual_add,
    emit_rms_norm, emit_rope_tables, emit_rotary_embedding, emit_transpose, emit_weight_const,
};

/// Build a complete MIL [`Program`] for a LLaMA-family model.
///
/// When `options.ane` is true, emits two functions (`prefill` and `decode`)
/// with ANE-optimized ops (1×1 conv projections, decomposed RMSNorm,
/// static KV-cache state). Otherwise emits a single `main` function with
/// high-level ops.
pub fn build_program(
    provider: &dyn WeightProvider,
    options: &TemplateOptions,
) -> Result<ConversionResult, MilError> {
    let config = provider.config().clone();
    let mut warnings: Vec<String> = Vec::new();
    let mut program = Program::new("1.0.0");

    if options.ane {
        // ANE mode: emit prefill (dynamic seq_len) and decode (seq_len=1).
        let prefill = build_function(
            "prefill",
            provider,
            &config,
            options,
            FunctionMode::Prefill,
            &mut warnings,
        )?;
        let decode = build_function(
            "decode",
            provider,
            &config,
            options,
            FunctionMode::Decode,
            &mut warnings,
        )?;
        program.add_function(prefill);
        program.add_function(decode);
    } else {
        let main = build_function(
            "main",
            provider,
            &config,
            options,
            FunctionMode::Standard,
            &mut warnings,
        )?;
        program.add_function(main);
    }

    program.set_attribute("autoregressive", "true");
    Ok(ConversionResult { program, warnings })
}

/// Controls the sequence-length handling and KV-cache behaviour of a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FunctionMode {
    /// Standard mode — single `main` function, no KV-cache, dynamic seq_len.
    Standard,
    /// ANE prefill — dynamic seq_len, builds KV-cache from scratch.
    Prefill,
    /// ANE decode — seq_len = 1, reads/updates KV-cache per token.
    Decode,
}

/// Build a single LLaMA function for the given mode.
fn build_function(
    func_name: &str,
    provider: &dyn WeightProvider,
    config: &ModelConfig,
    options: &TemplateOptions,
    mode: FunctionMode,
    warnings: &mut Vec<String>,
) -> Result<Function, MilError> {
    let seq_len: Option<usize> = match mode {
        FunctionMode::Decode => Some(1),
        _ => None, // dynamic
    };
    let batch: Option<usize> = Some(1);
    let ane = options.ane;

    // Build the function with typed inputs.
    let input_ids_ty = TensorType::with_dynamic_shape(ScalarType::Int32, vec![batch, seq_len]);
    let position_ids_ty = TensorType::with_dynamic_shape(ScalarType::Int32, vec![batch, seq_len]);
    let causal_mask_ty =
        TensorType::with_dynamic_shape(ScalarType::Float16, vec![batch, seq_len, seq_len]);

    let mut func = Function::new(func_name)
        .with_input("input_ids", input_ids_ty)
        .with_input("position_ids", position_ids_ty)
        .with_input("causal_mask", causal_mask_ty);

    // KV-cache state inputs (ANE modes only).
    if ane {
        let kv_cache_ty = TensorType::with_dynamic_shape(
            ScalarType::Float16,
            vec![
                batch,
                Some(config.num_key_value_heads),
                Some(config.max_position_embeddings),
                Some(config.head_dim),
            ],
        );
        for layer_idx in 0..config.num_hidden_layers {
            func = func
                .with_input(format!("k_cache_{layer_idx}"), kv_cache_ty.clone())
                .with_input(format!("v_cache_{layer_idx}"), kv_cache_ty.clone());
        }
    }

    let block = &mut func.body;

    // Embedding lookup: const weight + gather.
    let embed_out = emit_embedding(block, provider, config, warnings)?;

    // Precompute RoPE cos/sin tables (shared across all layers).
    let (rope_cos, rope_sin) = emit_rope_tables(block, config);

    // Transformer layers.
    let mut hidden = embed_out;
    for layer_idx in 0..config.num_hidden_layers {
        let ctx = LayerContext {
            provider,
            config,
            layer_idx,
            rope_cos: &rope_cos,
            rope_sin: &rope_sin,
        };
        hidden = emit_transformer_layer(block, &ctx, &hidden, warnings, ane)?;
    }

    // Final RMSNorm.
    let normed = if ane {
        emit_rms_norm_ane(
            block,
            provider,
            config,
            "model.norm",
            &hidden,
            "final_norm",
            warnings,
        )?
    } else {
        emit_rms_norm(
            block,
            provider,
            config,
            "model.norm",
            &hidden,
            "final_norm",
            warnings,
        )?
    };

    // LM head projection.
    let logits = emit_lm_head(block, provider, config, &normed, warnings, ane)?;

    block.outputs.push(logits);

    // In ANE mode, output updated KV-cache states.
    if ane {
        for layer_idx in 0..config.num_hidden_layers {
            block.outputs.push(format!("k_cache_{layer_idx}_updated"));
            block.outputs.push(format!("v_cache_{layer_idx}_updated"));
        }
    }

    Ok(func)
}

// ---------------------------------------------------------------------------
// Transformer layer
// ---------------------------------------------------------------------------

fn emit_transformer_layer(
    block: &mut Block,
    ctx: &LayerContext<'_>,
    input: &str,
    warnings: &mut Vec<String>,
    ane: bool,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{}", ctx.layer_idx);

    // 1. Input RMSNorm
    let normed_attn = emit_norm(
        block,
        ctx,
        &format!("{prefix}.input_layernorm"),
        input,
        &format!("l{}_input_norm", ctx.layer_idx),
        warnings,
        ane,
    )?;

    // 2. Self-attention
    let attn_out = emit_attention(block, ctx, &normed_attn, warnings, ane)?;

    // 3. Residual add (input + attn_out)
    let post_attn = emit_residual_add(
        block,
        input,
        &attn_out,
        &format!("l{}_post_attn_residual", ctx.layer_idx),
    );

    // 4. Post-attention RMSNorm
    let normed_mlp = emit_norm(
        block,
        ctx,
        &format!("{prefix}.post_attention_layernorm"),
        &post_attn,
        &format!("l{}_post_attn_norm", ctx.layer_idx),
        warnings,
        ane,
    )?;

    // 5. MLP
    let mlp_out = emit_mlp(
        block,
        ctx.provider,
        ctx.config,
        ctx.layer_idx,
        &normed_mlp,
        warnings,
        ane,
    )?;

    // 6. Residual add (post_attn + mlp_out)
    let layer_out = emit_residual_add(
        block,
        &post_attn,
        &mlp_out,
        &format!("l{}_output", ctx.layer_idx),
    );

    Ok(layer_out)
}

// ---------------------------------------------------------------------------
// Self-attention
// ---------------------------------------------------------------------------

fn emit_attention(
    block: &mut Block,
    ctx: &LayerContext<'_>,
    input: &str,
    warnings: &mut Vec<String>,
    ane: bool,
) -> Result<String, MilError> {
    let layer_idx = ctx.layer_idx;
    let config = ctx.config;
    let provider = ctx.provider;
    let prefix = format!("model.layers.{layer_idx}.self_attn");

    // Q/K/V projections
    let q = emit_projection(
        block,
        provider,
        &format!("{prefix}.q_proj"),
        input,
        &format!("l{layer_idx}_q_proj"),
        warnings,
        ane,
    )?;
    let k = emit_projection(
        block,
        provider,
        &format!("{prefix}.k_proj"),
        input,
        &format!("l{layer_idx}_k_proj"),
        warnings,
        ane,
    )?;
    let v = emit_projection(
        block,
        provider,
        &format!("{prefix}.v_proj"),
        input,
        &format!("l{layer_idx}_v_proj"),
        warnings,
        ane,
    )?;

    // Reshape Q for multi-head: [batch, seq, hidden] -> [batch, seq, n_heads, head_dim]
    let q_reshaped = emit_reshape(
        block,
        &q,
        &format!("l{layer_idx}_q_reshape"),
        &[
            -1,
            -1,
            config.num_attention_heads as i64,
            config.head_dim as i64,
        ],
    );

    // Reshape K for multi-head (may use GQA with num_key_value_heads)
    let k_reshaped = emit_reshape(
        block,
        &k,
        &format!("l{layer_idx}_k_reshape"),
        &[
            -1,
            -1,
            config.num_key_value_heads as i64,
            config.head_dim as i64,
        ],
    );

    // Reshape V
    let v_reshaped = emit_reshape(
        block,
        &v,
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

    // KV-cache update (ANE mode): write new K/V into the cache, read full
    // cache for attention.  In standard mode, skip cache entirely.
    let (k_for_attn, v_for_attn) = if ane {
        let (k_cached, v_cached) = emit_kv_cache_update(block, config, layer_idx, &k_roped, &v_t);
        (k_cached, v_cached)
    } else {
        (k_roped.clone(), v_t)
    };

    // GQA: expand K and V heads to match num_attention_heads when using grouped query attention
    let (k_expanded, v_expanded) = if config.num_attention_heads != config.num_key_value_heads {
        if config.num_attention_heads % config.num_key_value_heads != 0 {
            return Err(MilError::Validation(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                config.num_attention_heads, config.num_key_value_heads
            )));
        }
        let n_rep = config.num_attention_heads / config.num_key_value_heads;
        let k_exp = emit_gqa_expand(block, &k_for_attn, n_rep, config, layer_idx, "k");
        let v_exp = emit_gqa_expand(block, &v_for_attn, n_rep, config, layer_idx, "v");
        (k_exp, v_exp)
    } else {
        (k_for_attn, v_for_attn)
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
    let o_proj = emit_projection(
        block,
        provider,
        &format!("{prefix}.o_proj"),
        &attn_flat,
        &format!("l{layer_idx}_o_proj"),
        warnings,
        ane,
    )?;

    Ok(o_proj)
}

// ---------------------------------------------------------------------------
// MLP (gate + up + silu + down)
// ---------------------------------------------------------------------------

fn emit_mlp(
    block: &mut Block,
    provider: &dyn WeightProvider,
    _config: &ModelConfig,
    layer_idx: usize,
    input: &str,
    warnings: &mut Vec<String>,
    ane: bool,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{layer_idx}.mlp");

    // Gate projection
    let gate = emit_projection(
        block,
        provider,
        &format!("{prefix}.gate_proj"),
        input,
        &format!("l{layer_idx}_gate_proj"),
        warnings,
        ane,
    )?;

    // Up projection
    let up = emit_projection(
        block,
        provider,
        &format!("{prefix}.up_proj"),
        input,
        &format!("l{layer_idx}_up_proj"),
        warnings,
        ane,
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

    // Down projection
    let down = emit_projection(
        block,
        provider,
        &format!("{prefix}.down_proj"),
        &mlp_hidden,
        &format!("l{layer_idx}_down_proj"),
        warnings,
        ane,
    )?;

    Ok(down)
}

// ---------------------------------------------------------------------------
// LM head
// ---------------------------------------------------------------------------

fn emit_lm_head(
    block: &mut Block,
    provider: &dyn WeightProvider,
    config: &ModelConfig,
    input: &str,
    warnings: &mut Vec<String>,
    ane: bool,
) -> Result<String, MilError> {
    if config.tie_word_embeddings {
        if ane {
            // ANE path: 1×1 conv reusing the embedding weight.
            let weight_4d = emit_reshape(
                block,
                "embed_tokens_weight",
                "lm_head_weight_4d",
                &[0, 0, 1, 1],
            );
            let input_t = emit_transpose(block, input, "lm_head_ane_transpose_in", &[0, 2, 1]);
            let input_4d = emit_reshape(block, &input_t, "lm_head_ane_reshape_in", &[0, 0, 0, 1]);

            let conv_out_name = "lm_head_conv_out".to_string();
            let conv_op = Operation::new("conv", "lm_head_conv_op")
                .with_input("x", Value::Reference(input_4d))
                .with_input("weight", Value::Reference(weight_4d))
                .with_output(&conv_out_name);
            block.add_op(conv_op);

            let squeezed = {
                let out_name = "lm_head_ane_squeeze".to_string();
                let op = Operation::new("squeeze", "lm_head_ane_squeeze_op")
                    .with_input("x", Value::Reference(conv_out_name))
                    .with_attr("axes", Value::List(vec![Value::Int(3)]))
                    .with_output(&out_name);
                block.add_op(op);
                out_name
            };

            let out = emit_transpose(block, &squeezed, "lm_head_ane_transpose_out", &[0, 2, 1]);
            Ok(out)
        } else {
            // Reuse embedding weight (already emitted as embed_tokens_weight).
            let out_name = "lm_head_out".to_string();
            let op = Operation::new("linear", "lm_head_op")
                .with_input("x", Value::Reference(input.into()))
                .with_input("weight", Value::Reference("embed_tokens_weight".into()))
                .with_output(&out_name);
            block.add_op(op);
            Ok(out_name)
        }
    } else {
        emit_projection(block, provider, "lm_head", input, "lm_head", warnings, ane)
    }
}

// ---------------------------------------------------------------------------
// ANE-optimized helpers
// ---------------------------------------------------------------------------

/// Dispatch to either [`emit_linear`] or [`emit_projection_ane`] based on the
/// `ane` flag.
fn emit_projection(
    block: &mut Block,
    provider: &dyn WeightProvider,
    weight_prefix: &str,
    input: &str,
    op_prefix: &str,
    warnings: &mut Vec<String>,
    ane: bool,
) -> Result<String, MilError> {
    if ane {
        emit_projection_ane(block, provider, weight_prefix, input, op_prefix, warnings)
    } else {
        emit_linear(block, provider, weight_prefix, input, op_prefix, warnings)
    }
}

/// ANE-optimized projection: emit a 1×1 `conv` instead of `linear`.
///
/// Reshapes the weight from `[out, in]` → `[out, in, 1, 1]`, transposes the
/// input from `[batch, seq, hidden]` → `[batch, hidden, seq, 1]` (NCHW),
/// runs a `conv` op, then reshapes back to `[batch, seq, out]`.
fn emit_projection_ane(
    block: &mut Block,
    provider: &dyn WeightProvider,
    weight_prefix: &str,
    input: &str,
    op_prefix: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    // Load weight [out_features, in_features]
    let weight_name = format!("{weight_prefix}.weight");
    let weight_const = format!("{op_prefix}_weight");
    emit_weight_const(block, provider, &weight_name, &weight_const, warnings)?;

    // Reshape weight: [out, in] → [out, in, 1, 1]
    let weight_4d = emit_reshape(
        block,
        &weight_const,
        &format!("{op_prefix}_weight_4d"),
        &[0, 0, 1, 1],
    );

    // Optional bias
    let bias_name = format!("{weight_prefix}.bias");
    let has_bias = provider.has_tensor(&bias_name);
    if has_bias {
        let bias_const = format!("{op_prefix}_bias");
        emit_weight_const(block, provider, &bias_name, &bias_const, warnings)?;
    }

    // Transpose input: [batch, seq, hidden] → [batch, hidden, seq]
    let input_t = emit_transpose(
        block,
        input,
        &format!("{op_prefix}_ane_transpose_in"),
        &[0, 2, 1],
    );

    // Expand to 4-D: [batch, hidden, seq] → [batch, hidden, seq, 1]
    let input_4d = emit_reshape(
        block,
        &input_t,
        &format!("{op_prefix}_ane_reshape_in"),
        &[0, 0, 0, 1],
    );

    // 1×1 convolution
    let conv_out = format!("{op_prefix}_conv_out");
    let mut conv_op = Operation::new("conv", format!("{op_prefix}_conv_op"))
        .with_input("x", Value::Reference(input_4d))
        .with_input("weight", Value::Reference(weight_4d));

    if has_bias {
        conv_op = conv_op.with_input("bias", Value::Reference(format!("{op_prefix}_bias")));
    }

    conv_op = conv_op.with_output(&conv_out);
    block.add_op(conv_op);

    // Squeeze trailing dim: [batch, out, seq, 1] → [batch, out, seq]
    let squeezed = {
        let out_name = format!("{op_prefix}_ane_squeeze");
        let op = Operation::new("squeeze", format!("{op_prefix}_ane_squeeze_op"))
            .with_input("x", Value::Reference(conv_out))
            .with_attr("axes", Value::List(vec![Value::Int(3)]))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Transpose back: [batch, out, seq] → [batch, seq, out]
    let out = emit_transpose(
        block,
        &squeezed,
        &format!("{op_prefix}_ane_transpose_out"),
        &[0, 2, 1],
    );

    Ok(out)
}

/// Dispatch to either high-level [`emit_rms_norm`] or the ANE-decomposed form.
fn emit_norm(
    block: &mut Block,
    ctx: &LayerContext<'_>,
    weight_prefix: &str,
    input: &str,
    op_prefix: &str,
    warnings: &mut Vec<String>,
    ane: bool,
) -> Result<String, MilError> {
    if ane {
        emit_rms_norm_ane(
            block,
            ctx.provider,
            ctx.config,
            weight_prefix,
            input,
            op_prefix,
            warnings,
        )
    } else {
        emit_rms_norm(
            block,
            ctx.provider,
            ctx.config,
            weight_prefix,
            input,
            op_prefix,
            warnings,
        )
    }
}

/// ANE-decomposed RMSNorm: `x → concat([x, −x]) → layer_norm → slice`.
///
/// The ANE does not natively support RMSNorm but handles LayerNorm well.
/// Concatenating `[x, −x]` produces a zero-mean tensor whose variance equals
/// `mean(x²)`, so a standard LayerNorm on the doubled tensor is equivalent
/// to RMSNorm on the original. We slice the first half to recover the result.
fn emit_rms_norm_ane(
    block: &mut Block,
    provider: &dyn WeightProvider,
    config: &ModelConfig,
    weight_prefix: &str,
    input: &str,
    op_prefix: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    // Load RMSNorm weight (gamma)
    let weight_name = format!("{weight_prefix}.weight");
    let weight_const = format!("{op_prefix}_weight");
    emit_weight_const(block, provider, &weight_name, &weight_const, warnings)?;

    // neg_x = x * −1
    let neg_const = format!("{op_prefix}_neg_const");
    let neg_const_op = Operation::new("const", &neg_const)
        .with_attr("val", Value::Float(-1.0))
        .with_output(&neg_const);
    block.add_op(neg_const_op);

    let neg_x = format!("{op_prefix}_neg_x");
    let neg_mul = Operation::new("mul", format!("{op_prefix}_neg_mul_op"))
        .with_input("x", Value::Reference(input.into()))
        .with_input("y", Value::Reference(neg_const.clone()))
        .with_output(&neg_x);
    block.add_op(neg_mul);

    // concat([x, −x], axis = −1) → shape doubles along last dim
    let concat_out = format!("{op_prefix}_concat_out");
    let concat_op = Operation::new("concat", format!("{op_prefix}_concat_op"))
        .with_input(
            "values",
            Value::List(vec![
                Value::Reference(input.into()),
                Value::Reference(neg_x),
            ]),
        )
        .with_attr("axis", Value::Int(-1))
        .with_output(&concat_out);
    block.add_op(concat_op);

    // Double the weight: concat([weight, weight], axis = 0)
    let weight_doubled = format!("{op_prefix}_weight_doubled");
    let weight_double_op = Operation::new("concat", format!("{op_prefix}_weight_double_op"))
        .with_input(
            "values",
            Value::List(vec![
                Value::Reference(weight_const.clone()),
                Value::Reference(weight_const),
            ]),
        )
        .with_attr("axis", Value::Int(0))
        .with_output(&weight_doubled);
    block.add_op(weight_double_op);

    // layer_norm over last axis with doubled gamma, no beta
    let axes_const = format!("{op_prefix}_norm_axes");
    let axes_op = Operation::new("const", &axes_const)
        .with_attr("val", Value::List(vec![Value::Int(-1)]))
        .with_output(&axes_const);
    block.add_op(axes_op);

    let norm_out = format!("{op_prefix}_layer_norm_out");
    let norm_op = Operation::new("layer_norm", format!("{op_prefix}_layer_norm_op"))
        .with_input("x", Value::Reference(concat_out))
        .with_input("axes", Value::Reference(axes_const))
        .with_input("gamma", Value::Reference(weight_doubled))
        .with_attr("epsilon", Value::Float(config.rms_norm_eps))
        .with_output(&norm_out);
    block.add_op(norm_op);

    // Slice first half along last axis: [0 .. hidden_size]
    let begin_const = format!("{op_prefix}_slice_begin");
    let begin_op = Operation::new("const", &begin_const)
        .with_attr(
            "val",
            Value::List(vec![Value::Int(0), Value::Int(0), Value::Int(0)]),
        )
        .with_output(&begin_const);
    block.add_op(begin_op);

    let end_const = format!("{op_prefix}_slice_end");
    let end_op = Operation::new("const", &end_const)
        .with_attr(
            "val",
            Value::List(vec![
                Value::Int(0),
                Value::Int(0),
                Value::Int(config.hidden_size as i64),
            ]),
        )
        .with_output(&end_const);
    block.add_op(end_op);

    let out_name = format!("{op_prefix}_out");
    let slice_op = Operation::new("slice_by_index", format!("{op_prefix}_slice_op"))
        .with_input("x", Value::Reference(norm_out))
        .with_input("begin", Value::Reference(begin_const))
        .with_input("end", Value::Reference(end_const))
        .with_attr(
            "begin_mask",
            Value::List(vec![
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(true),
            ]),
        )
        .with_attr(
            "end_mask",
            Value::List(vec![
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(false),
            ]),
        )
        .with_output(&out_name);
    block.add_op(slice_op);

    Ok(out_name)
}

/// Emit KV-cache update ops for a single layer.
///
/// Reads the current cache state from function inputs `k_cache_{layer}` /
/// `v_cache_{layer}`, concatenates the new K/V along the sequence axis,
/// slices to `max_position_embeddings` to bound memory, and writes the
/// updated cache as `k_cache_{layer}_updated` / `v_cache_{layer}_updated`.
///
/// Returns `(k_cached, v_cached)` — the full K and V tensors to use for
/// attention.
fn emit_kv_cache_update(
    block: &mut Block,
    config: &ModelConfig,
    layer_idx: usize,
    k_new: &str,
    v_new: &str,
) -> (String, String) {
    let max_seq = config.max_position_embeddings as i64;

    let k_cache_in = format!("k_cache_{layer_idx}");
    let v_cache_in = format!("v_cache_{layer_idx}");

    // Reshape position_ids from [batch, seq] to [batch, 1, seq, 1] so that
    // indices rank matches the 4-D cache tensors and broadcasts across
    // kv_heads and head_dim dimensions.
    let pos_ids_4d = emit_reshape(
        block,
        "position_ids",
        &format!("l{layer_idx}_pos_ids_reshape"),
        &[0, 1, -1, 1],
    );

    // scatter new K into cache using position_ids
    // K_new shape: [batch, kv_heads, seq_new, head_dim]
    // k_cache shape: [batch, kv_heads, max_seq, head_dim]
    // We use scatter_along_axis on axis=2 to place new keys at position_ids.
    let k_updated = {
        let out_name = format!("k_cache_{layer_idx}_updated");
        let op = Operation::new("scatter", format!("l{layer_idx}_k_cache_scatter_op"))
            .with_input("data", Value::Reference(k_cache_in))
            .with_input("indices", Value::Reference(pos_ids_4d.clone()))
            .with_input("updates", Value::Reference(k_new.into()))
            .with_attr("axis", Value::Int(2))
            .with_attr("mode", Value::String("update".into()))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    let v_updated = {
        let out_name = format!("v_cache_{layer_idx}_updated");
        let op = Operation::new("scatter", format!("l{layer_idx}_v_cache_scatter_op"))
            .with_input("data", Value::Reference(v_cache_in))
            .with_input("indices", Value::Reference(pos_ids_4d))
            .with_input("updates", Value::Reference(v_new.into()))
            .with_attr("axis", Value::Int(2))
            .with_attr("mode", Value::String("update".into()))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Read the full updated cache for attention. We slice [0..max_seq] to
    // give downstream ops a bounded shape hint, even though the cache is
    // already max_seq in size. This is a no-op at runtime but aids ANE
    // shape inference.
    let k_sliced = emit_cache_slice(block, &k_updated, layer_idx, "k", max_seq);
    let v_sliced = emit_cache_slice(block, &v_updated, layer_idx, "v", max_seq);

    (k_sliced, v_sliced)
}

/// Slice a KV-cache tensor along the sequence axis (dim 2) to `[0..max_seq]`.
fn emit_cache_slice(
    block: &mut Block,
    input: &str,
    layer_idx: usize,
    kv: &str,
    max_seq: i64,
) -> String {
    let prefix = format!("l{layer_idx}_{kv}_cache_slice");

    let begin_const = format!("{prefix}_begin");
    let begin_op = Operation::new("const", &begin_const)
        .with_attr(
            "val",
            Value::List(vec![
                Value::Int(0),
                Value::Int(0),
                Value::Int(0),
                Value::Int(0),
            ]),
        )
        .with_output(&begin_const);
    block.add_op(begin_op);

    let end_const = format!("{prefix}_end");
    let end_op = Operation::new("const", &end_const)
        .with_attr(
            "val",
            Value::List(vec![
                Value::Int(0),
                Value::Int(0),
                Value::Int(max_seq),
                Value::Int(0),
            ]),
        )
        .with_output(&end_const);
    block.add_op(end_op);

    let out_name = format!("{prefix}_out");
    let slice_op = Operation::new("slice_by_index", format!("{prefix}_op"))
        .with_input("x", Value::Reference(input.into()))
        .with_input("begin", Value::Reference(begin_const))
        .with_input("end", Value::Reference(end_const))
        .with_attr(
            "begin_mask",
            Value::List(vec![
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(true),
            ]),
        )
        .with_attr(
            "end_mask",
            Value::List(vec![
                Value::Bool(true),
                Value::Bool(true),
                Value::Bool(false),
                Value::Bool(true),
            ]),
        )
        .with_output(&out_name);
    block.add_op(slice_op);

    out_name
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::templates::shared::{StubProvider, tiny_llama_config};
    use mil_rs::ir::Value;

    #[test]
    fn build_program_succeeds_with_all_weights() {
        let config = tiny_llama_config();
        let provider = StubProvider::new(config).with_llama_weights();
        let opts = TemplateOptions::default();

        let result = build_program(&provider, &opts).expect("build_program should succeed");
        assert_eq!(result.program.functions.len(), 1);
        assert!(
            result.warnings.is_empty(),
            "unexpected warnings: {:?}",
            result.warnings
        );

        let main = result.program.main().expect("should have main function");
        assert_eq!(main.inputs.len(), 3);
        assert!(!main.body.operations.is_empty());
        assert_eq!(main.body.outputs.len(), 1);
    }

    #[test]
    fn build_program_with_tied_embeddings() {
        let mut config = tiny_llama_config();
        config.tie_word_embeddings = true;
        let provider = StubProvider::new(config).with_llama_weights();
        let opts = TemplateOptions::default();

        let result = build_program(&provider, &opts).expect("build_program should succeed");
        let main = result.program.main().unwrap();

        // Should have lm_head_out referencing embed_tokens_weight
        let lm_head_op = main
            .body
            .operations
            .iter()
            .find(|op| op.name == "lm_head_op");
        assert!(lm_head_op.is_some(), "should have lm_head_op");
        let lm_head = lm_head_op.unwrap();
        assert_eq!(
            lm_head.inputs.get("weight"),
            Some(&Value::Reference("embed_tokens_weight".into()))
        );
    }

    #[test]
    fn build_program_warns_on_missing_weights() {
        let config = tiny_llama_config();
        // Provide no tensors — everything will be missing.
        let provider = StubProvider::new(config);
        let opts = TemplateOptions::default();

        let result = build_program(&provider, &opts)
            .expect("build_program should still succeed structurally");
        assert!(
            !result.warnings.is_empty(),
            "should have warnings for missing weights"
        );
        // The program should still be structurally valid.
        let main = result.program.main().unwrap();
        assert!(!main.body.operations.is_empty());
    }

    #[test]
    fn program_is_marked_autoregressive() {
        let config = tiny_llama_config();
        let provider = StubProvider::new(config).with_llama_weights();
        let opts = TemplateOptions::default();
        let result = build_program(&provider, &opts).unwrap();
        assert!(result.program.is_autoregressive());
    }

    #[test]
    fn build_program_with_gqa() {
        let mut config = tiny_llama_config();
        // 4 attention heads, 2 KV heads → GQA with n_rep=2
        config.num_key_value_heads = 2;
        let provider = StubProvider::new(config).with_llama_weights();
        let opts = TemplateOptions::default();

        let result = build_program(&provider, &opts).expect("GQA build should succeed");
        assert!(
            result.warnings.is_empty(),
            "unexpected warnings: {:?}",
            result.warnings
        );

        let main = result.program.main().unwrap();
        // Should contain tile ops for GQA expansion
        let tile_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.op_type == "tile")
            .collect();
        assert!(
            !tile_ops.is_empty(),
            "GQA should emit tile ops for KV head expansion"
        );
    }

    #[test]
    fn build_program_emits_rope() {
        let config = tiny_llama_config();
        let provider = StubProvider::new(config).with_llama_weights();
        let opts = TemplateOptions::default();

        let result = build_program(&provider, &opts).expect("build should succeed");
        let main = result.program.main().unwrap();

        // Should have RoPE const tables
        let has_cos_table = main
            .body
            .operations
            .iter()
            .any(|op| op.name == "rope_cos_table");
        let has_sin_table = main
            .body
            .operations
            .iter()
            .any(|op| op.name == "rope_sin_table");
        assert!(has_cos_table, "should emit rope_cos_table const");
        assert!(has_sin_table, "should emit rope_sin_table const");

        // Should have RoPE gather and concat ops per layer
        let rope_concat_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.name.contains("rope") && op.op_type == "concat")
            .collect();
        // 2 layers × 2 (Q and K) = 4 concat ops
        assert_eq!(
            rope_concat_ops.len(),
            4,
            "expected 4 RoPE concat ops (2 layers × Q+K), got {}",
            rope_concat_ops.len()
        );
    }

    // -------------------------------------------------------------------
    // ANE-mode tests
    // -------------------------------------------------------------------

    fn ane_opts() -> TemplateOptions {
        TemplateOptions { ane: true }
    }

    #[test]
    fn ane_emits_prefill_and_decode_functions() {
        let config = tiny_llama_config();
        let provider = StubProvider::new(config).with_llama_weights();

        let result = build_program(&provider, &ane_opts()).expect("ANE build should succeed");
        assert_eq!(
            result.program.functions.len(),
            2,
            "ANE mode should produce two functions"
        );
        assert!(
            result.program.functions.contains_key("prefill"),
            "should have prefill function"
        );
        assert!(
            result.program.functions.contains_key("decode"),
            "should have decode function"
        );
    }

    #[test]
    fn ane_decode_has_static_seq_len_1() {
        let config = tiny_llama_config();
        let provider = StubProvider::new(config).with_llama_weights();

        let result = build_program(&provider, &ane_opts()).unwrap();
        let decode = &result.program.functions["decode"];

        let (_, input_ids_ty) = &decode.inputs[0];
        assert_eq!(
            input_ids_ty.shape,
            vec![Some(1), Some(1)],
            "decode input_ids should have seq_len=1"
        );
    }

    #[test]
    fn ane_emits_conv_instead_of_linear() {
        let config = tiny_llama_config();
        let provider = StubProvider::new(config).with_llama_weights();

        let result = build_program(&provider, &ane_opts()).unwrap();
        let prefill = &result.program.functions["prefill"];

        let conv_ops: Vec<_> = prefill
            .body
            .operations
            .iter()
            .filter(|op| op.op_type == "conv")
            .collect();
        assert!(
            !conv_ops.is_empty(),
            "ANE mode should emit conv ops for projections"
        );

        // Should NOT have any linear ops (projections are all conv now)
        let linear_ops: Vec<_> = prefill
            .body
            .operations
            .iter()
            .filter(|op| op.op_type == "linear")
            .collect();
        assert!(
            linear_ops.is_empty(),
            "ANE mode should not emit linear ops, got {} linear ops",
            linear_ops.len()
        );
    }

    #[test]
    fn ane_emits_decomposed_rms_norm() {
        let config = tiny_llama_config();
        let provider = StubProvider::new(config).with_llama_weights();

        let result = build_program(&provider, &ane_opts()).unwrap();
        let prefill = &result.program.functions["prefill"];

        // Should have layer_norm ops (decomposed RMSNorm)
        let layer_norm_ops: Vec<_> = prefill
            .body
            .operations
            .iter()
            .filter(|op| op.op_type == "layer_norm")
            .collect();
        assert!(
            !layer_norm_ops.is_empty(),
            "ANE mode should use layer_norm for decomposed RMSNorm"
        );

        // Should NOT have any high-level rms_norm ops
        let rms_norm_ops: Vec<_> = prefill
            .body
            .operations
            .iter()
            .filter(|op| op.op_type == "rms_norm")
            .collect();
        assert!(
            rms_norm_ops.is_empty(),
            "ANE mode should not have rms_norm ops"
        );
    }

    #[test]
    fn ane_has_kv_cache_inputs() {
        let config = tiny_llama_config();
        let provider = StubProvider::new(config.clone()).with_llama_weights();

        let result = build_program(&provider, &ane_opts()).unwrap();
        let decode = &result.program.functions["decode"];

        // Each layer should have k_cache and v_cache inputs
        for layer_idx in 0..config.num_hidden_layers {
            let has_k = decode
                .inputs
                .iter()
                .any(|(name, _)| name == &format!("k_cache_{layer_idx}"));
            let has_v = decode
                .inputs
                .iter()
                .any(|(name, _)| name == &format!("v_cache_{layer_idx}"));
            assert!(has_k, "missing k_cache_{layer_idx} input");
            assert!(has_v, "missing v_cache_{layer_idx} input");
        }
    }

    #[test]
    fn ane_outputs_updated_kv_cache() {
        let config = tiny_llama_config();
        let provider = StubProvider::new(config.clone()).with_llama_weights();

        let result = build_program(&provider, &ane_opts()).unwrap();
        let prefill = &result.program.functions["prefill"];

        // Should output logits + 2 cache tensors per layer
        let expected_outputs = 1 + 2 * config.num_hidden_layers;
        assert_eq!(
            prefill.body.outputs.len(),
            expected_outputs,
            "prefill should output logits + KV cache updates"
        );
    }

    #[test]
    fn ane_program_is_autoregressive() {
        let config = tiny_llama_config();
        let provider = StubProvider::new(config).with_llama_weights();
        let result = build_program(&provider, &ane_opts()).unwrap();
        assert!(result.program.is_autoregressive());
    }
}
