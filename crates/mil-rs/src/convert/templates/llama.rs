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

use crate::MilError;
use crate::convert::onnx_graph::ConversionResult;
use crate::convert::templates::TemplateOptions;
use crate::convert::weights::{ModelConfig, WeightProvider};
use crate::ir::{Block, Function, Operation, Program, ScalarType, TensorType, Value};

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
#[allow(clippy::too_many_arguments)]
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
        hidden = emit_transformer_layer(
            block, provider, config, layer_idx, &hidden, &rope_cos, &rope_sin, warnings, ane,
        )?;
    }

    // Final RMSNorm.
    let normed = emit_norm(
        block,
        provider,
        config,
        "model.norm",
        &hidden,
        "final_norm",
        warnings,
        ane,
    )?;

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
// Embedding
// ---------------------------------------------------------------------------

fn emit_embedding(
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
// Transformer layer
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_transformer_layer(
    block: &mut Block,
    provider: &dyn WeightProvider,
    config: &ModelConfig,
    layer_idx: usize,
    input: &str,
    rope_cos: &str,
    rope_sin: &str,
    warnings: &mut Vec<String>,
    ane: bool,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{layer_idx}");

    // 1. Input RMSNorm
    let normed_attn = emit_norm(
        block,
        provider,
        config,
        &format!("{prefix}.input_layernorm"),
        input,
        &format!("l{layer_idx}_input_norm"),
        warnings,
        ane,
    )?;

    // 2. Self-attention
    let attn_out = emit_attention(
        block,
        provider,
        config,
        layer_idx,
        &normed_attn,
        rope_cos,
        rope_sin,
        warnings,
        ane,
    )?;

    // 3. Residual add (input + attn_out)
    let post_attn = emit_residual_add(
        block,
        input,
        &attn_out,
        &format!("l{layer_idx}_post_attn_residual"),
    );

    // 4. Post-attention RMSNorm
    let normed_mlp = emit_norm(
        block,
        provider,
        config,
        &format!("{prefix}.post_attention_layernorm"),
        &post_attn,
        &format!("l{layer_idx}_post_attn_norm"),
        warnings,
        ane,
    )?;

    // 5. MLP
    let mlp_out = emit_mlp(
        block,
        provider,
        config,
        layer_idx,
        &normed_mlp,
        warnings,
        ane,
    )?;

    // 6. Residual add (post_attn + mlp_out)
    let layer_out = emit_residual_add(block, &post_attn, &mlp_out, &format!("l{layer_idx}_output"));

    Ok(layer_out)
}

// ---------------------------------------------------------------------------
// Self-attention
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_attention(
    block: &mut Block,
    provider: &dyn WeightProvider,
    config: &ModelConfig,
    layer_idx: usize,
    input: &str,
    rope_cos: &str,
    rope_sin: &str,
    warnings: &mut Vec<String>,
    ane: bool,
) -> Result<String, MilError> {
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
    let (q_roped, k_roped) =
        emit_rotary_embedding(block, config, &q_t, &k_t, layer_idx, rope_cos, rope_sin);

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
// Primitive helpers
// ---------------------------------------------------------------------------

/// Emit a weight tensor as a `const` op in the block.
fn emit_weight_const(
    block: &mut Block,
    provider: &dyn WeightProvider,
    weight_name: &str,
    const_name: &str,
    warnings: &mut Vec<String>,
) -> Result<(), MilError> {
    match provider.tensor(weight_name) {
        Ok(tensor) => {
            let op = Operation::new("const", const_name)
                .with_attr(
                    "val",
                    Value::Tensor {
                        data: tensor.data.into_owned(),
                        shape: tensor.shape.clone(),
                        dtype: tensor.dtype,
                    },
                )
                .with_output(const_name);
            block.add_op(op);
            Ok(())
        }
        Err(e) => {
            warnings.push(format!("missing weight {weight_name}: {e}"));
            // Emit a placeholder const so the graph remains structurally valid.
            let op = Operation::new("const", const_name)
                .with_attr(
                    "val",
                    Value::Tensor {
                        data: Vec::new(),
                        shape: vec![0],
                        dtype: ScalarType::Float16,
                    },
                )
                .with_output(const_name);
            block.add_op(op);
            Ok(())
        }
    }
}

/// Emit a `linear` op (weight projection). Loads weight (and optional bias)
/// from the provider using HuggingFace naming convention.
fn emit_linear(
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

/// Emit an `rms_norm` op with its weight loaded from the provider.
fn emit_rms_norm(
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
#[allow(clippy::too_many_arguments)]
fn emit_norm(
    block: &mut Block,
    provider: &dyn WeightProvider,
    config: &ModelConfig,
    weight_prefix: &str,
    input: &str,
    op_prefix: &str,
    warnings: &mut Vec<String>,
    ane: bool,
) -> Result<String, MilError> {
    if ane {
        emit_rms_norm_ane(
            block,
            provider,
            config,
            weight_prefix,
            input,
            op_prefix,
            warnings,
        )
    } else {
        emit_rms_norm(
            block,
            provider,
            config,
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
#[allow(clippy::too_many_arguments)]
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

/// Emit an `add` op for residual connections.
fn emit_residual_add(block: &mut Block, x: &str, y: &str, out_name: &str) -> String {
    let op = Operation::new("add", format!("{out_name}_op"))
        .with_input("x", Value::Reference(x.into()))
        .with_input("y", Value::Reference(y.into()))
        .with_output(out_name);
    block.add_op(op);
    out_name.to_string()
}

/// Emit a `reshape` op.
fn emit_reshape(block: &mut Block, input: &str, op_name: &str, shape: &[i64]) -> String {
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
fn emit_transpose(block: &mut Block, input: &str, op_name: &str, perm: &[i64]) -> String {
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
fn emit_rope_tables(block: &mut Block, config: &ModelConfig) -> (String, String) {
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
                data: cos_bytes,
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
                data: sin_bytes,
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
fn emit_rotary_embedding(
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

    // Gather cos/sin using position_ids: [max_pos, half_dim] @ [batch, seq] → [batch, seq, half_dim]
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
fn emit_rope_apply(block: &mut Block, input: &str, cos: &str, sin: &str, prefix: &str) -> String {
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
fn emit_gqa_expand(
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

    // Tile along dim 2 by n_rep → [batch, num_kv_heads, n_rep, seq, head_dim]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::weights::{Architecture, WeightTensor};
    use std::collections::HashMap;

    /// Minimal stub provider for testing template construction.
    struct StubProvider {
        config: ModelConfig,
        tensors: HashMap<String, Vec<u8>>,
    }

    impl StubProvider {
        fn new(config: ModelConfig) -> Self {
            Self {
                config,
                tensors: HashMap::new(),
            }
        }

        fn with_tensor(mut self, name: &str, shape: &[usize], dtype: ScalarType) -> Self {
            let byte_size: usize = shape.iter().product::<usize>() * dtype.byte_size();
            self.tensors.insert(name.to_string(), vec![0u8; byte_size]);
            self
        }

        /// Populate all standard LLaMA weight names for the given config.
        fn with_llama_weights(mut self) -> Self {
            let h = self.config.hidden_size;
            let inter = self.config.intermediate_size;
            let n_heads = self.config.num_attention_heads;
            let n_kv_heads = self.config.num_key_value_heads;
            let head_dim = self.config.head_dim;
            let vocab = self.config.vocab_size;
            let num_layers = self.config.num_hidden_layers;
            let tie = self.config.tie_word_embeddings;

            // Embedding
            self = self.with_tensor(
                "model.embed_tokens.weight",
                &[vocab, h],
                ScalarType::Float16,
            );

            for l in 0..num_layers {
                let p = format!("model.layers.{l}");
                // Attention
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
                // Norms
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
                // MLP
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

            // Final norm
            self = self.with_tensor("model.norm.weight", &[h], ScalarType::Float16);
            // LM head
            if !tie {
                self = self.with_tensor("lm_head.weight", &[vocab, h], ScalarType::Float16);
            }

            self
        }
    }

    impl WeightProvider for StubProvider {
        fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError> {
            let data = self
                .tensors
                .get(name)
                .ok_or_else(|| MilError::Validation(format!("tensor not found: {name}")))?;
            // Infer a flat shape from data length (sufficient for structural tests).
            let elem_size = ScalarType::Float16.byte_size();
            let num_elements = data.len() / elem_size;
            Ok(WeightTensor::borrowed(
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

    fn tiny_config() -> ModelConfig {
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
            extra: HashMap::new(),
        }
    }

    #[test]
    fn build_program_succeeds_with_all_weights() {
        let config = tiny_config();
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
        let mut config = tiny_config();
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
        let config = tiny_config();
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
        let config = tiny_config();
        let provider = StubProvider::new(config).with_llama_weights();
        let opts = TemplateOptions::default();
        let result = build_program(&provider, &opts).unwrap();
        assert!(result.program.is_autoregressive());
    }

    #[test]
    fn build_program_with_gqa() {
        let mut config = tiny_config();
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
        let config = tiny_config();
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
        let config = tiny_config();
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
        let config = tiny_config();
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
        let config = tiny_config();
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
        let config = tiny_config();
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
        let config = tiny_config();
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
        let config = tiny_config();
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
        let config = tiny_config();
        let provider = StubProvider::new(config).with_llama_weights();
        let result = build_program(&provider, &ane_opts()).unwrap();
        assert!(result.program.is_autoregressive());
    }
}
