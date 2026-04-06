//! Qwen 3.5 architecture template.
//!
//! Supports the Qwen 3.5 model family which uses alternating Gated Delta
//! Network (GDN) layers and full attention layers.

use crate::weights::{ModelConfig, WeightProvider};
use mil_rs::MilError;
use mil_rs::convert::onnx_graph::ConversionResult;
use mil_rs::ir::{Block, Function, Operation, Program, ScalarType, TensorData, TensorType, Value};

use super::shared::{
    LayerContext, emit_embedding, emit_linear, emit_lm_head, emit_mlp_silu_coreml, emit_reshape,
    emit_residual_add, emit_rope_apply, emit_rope_tables, emit_transpose, emit_weight_const,
};

/// Build a complete MIL [`Program`] for a Qwen 3.5-family model.
pub fn build_program(provider: &dyn WeightProvider) -> Result<ConversionResult, MilError> {
    let config = provider.config().clone();
    let mut warnings: Vec<String> = Vec::new();

    let _seq_len: Option<usize> = None; // dynamic
    let _batch: Option<usize> = Some(1);

    let layer_types = parse_layer_types(&config);

    // Build the main function with fixed-size inputs.
    // CoreML E5RT requires concrete shapes for type inference through
    // rank-changing ops (expand_dims, reshape). Using seq_len=128 for now.
    let fixed_seq = 128;
    let input_ids_ty = TensorType::new(ScalarType::Int32, vec![1, fixed_seq]);
    let position_ids_ty = TensorType::new(ScalarType::Int32, vec![1, fixed_seq]);
    let causal_mask_ty = TensorType::new(ScalarType::Float16, vec![1, fixed_seq, fixed_seq]);

    let mut func = Function::new("main")
        .with_input("input_ids", input_ids_ty)
        .with_input("position_ids", position_ids_ty)
        .with_input("causal_mask", causal_mask_ty);

    let block = &mut func.body;

    // Embedding lookup.
    let embed_out = emit_embedding(block, provider, &config, &mut warnings)?;

    // Parse partial_rotary_factor (Qwen 3.5 only applies RoPE to a fraction of head_dim).
    // May be at top level or nested inside rope_parameters.
    let partial_rotary_factor = config
        .extra
        .get("partial_rotary_factor")
        .and_then(|v| v.as_f64())
        .or_else(|| {
            config
                .extra
                .get("rope_parameters")
                .and_then(|rp| rp.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
        })
        .unwrap_or(1.0);
    let rotary_dim = (config.head_dim as f64 * partial_rotary_factor) as usize;

    // Precompute RoPE cos/sin tables (used by full attention layers).
    // With partial_rotary_factor, tables are sized to rotary_dim, not full head_dim.
    let (rope_cos, rope_sin) = if rotary_dim < config.head_dim {
        emit_partial_rope_tables(block, &config, rotary_dim)
    } else {
        emit_rope_tables(block, &config)
    };

    // Parse GDN-specific config from extras.
    let gdn_cfg = GdnConfig::from_model_config(&config)?;

    let attn_output_gate = config
        .extra
        .get("attn_output_gate")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Transformer layers.
    let mut hidden = embed_out;
    for layer_idx in 0..config.num_hidden_layers {
        let layer_type = layer_types
            .get(layer_idx)
            .map(|s| s.as_str())
            .unwrap_or("full_attention");

        match layer_type {
            "linear_attention" => {
                hidden = emit_gdn_layer(
                    block,
                    provider,
                    &config,
                    &gdn_cfg,
                    layer_idx,
                    &hidden,
                    attn_output_gate,
                    &mut warnings,
                )?;
            }
            _ => {
                let ctx = LayerContext {
                    provider,
                    config: &config,
                    layer_idx,
                    rope_cos: &rope_cos,
                    rope_sin: &rope_sin,
                    layer_type: None,
                    effective_head_dim: config.head_dim,
                    effective_num_kv_heads: config.num_key_value_heads,
                };
                hidden = emit_qwen35_full_attention_layer(
                    block,
                    &ctx,
                    &hidden,
                    rotary_dim,
                    &mut warnings,
                )?;
            }
        }
    }

    // Final RMSNorm.
    let normed = emit_rms_norm_decomposed(
        block,
        provider,
        &config,
        "model.norm",
        &hidden,
        "final_norm",
        3,
        &mut warnings,
    )?;

    // LM head projection.
    let logits = emit_lm_head(block, provider, &config, &normed, &mut warnings)?;

    block.outputs.push(logits);

    let mut program = Program::new("1");
    program.add_function(func);
    program.set_attribute("autoregressive", "true");

    Ok(ConversionResult::new(program, warnings))
}

// ---------------------------------------------------------------------------
// Layer type parsing
// ---------------------------------------------------------------------------

fn parse_layer_types(config: &ModelConfig) -> Vec<String> {
    config
        .extra
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_else(|| vec!["full_attention".to_string(); config.num_hidden_layers])
}

// ---------------------------------------------------------------------------
// GDN config
// ---------------------------------------------------------------------------

struct GdnConfig {
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_conv_kernel_dim: usize,
}

impl GdnConfig {
    fn from_model_config(config: &ModelConfig) -> Result<Self, MilError> {
        let get = |key: &str| -> Result<usize, MilError> {
            config
                .extra
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| {
                    MilError::Validation(format!(
                        "Qwen 3.5 config missing required GDN parameter: {key}"
                    ))
                })
        };
        Ok(Self {
            linear_key_head_dim: get("linear_key_head_dim")?,
            linear_value_head_dim: get("linear_value_head_dim")?,
            linear_num_key_heads: get("linear_num_key_heads")?,
            linear_num_value_heads: get("linear_num_value_heads")?,
            linear_conv_kernel_dim: get("linear_conv_kernel_dim")?,
        })
    }

    fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    fn qkv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

// ---------------------------------------------------------------------------
// Full attention layer (Qwen 3.5 variant with output gate)
// ---------------------------------------------------------------------------

fn emit_qwen35_full_attention_layer(
    block: &mut Block,
    ctx: &LayerContext<'_>,
    input: &str,
    rotary_dim: usize,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{}", ctx.layer_idx);
    let li = ctx.layer_idx;
    let config = ctx.config;

    // 1. Input RMSNorm
    let normed_attn = emit_rms_norm_decomposed(
        block,
        ctx.provider,
        ctx.config,
        &format!("{prefix}.input_layernorm"),
        input,
        &format!("l{li}_input_norm"),
        3,
        warnings,
    )?;

    // 2. Q/K/V projections (no bias for Qwen 3.5)
    // Qwen 3.5 q_proj outputs [num_heads * head_dim * 2] — first half is Q,
    // second half is an attention output gate (applied as sigmoid(gate) * attn).
    let attn_prefix = format!("{prefix}.self_attn");
    let q_gate_proj = emit_linear(
        block,
        ctx.provider,
        &format!("{attn_prefix}.q_proj"),
        &normed_attn,
        &format!("l{li}_q_proj"),
        warnings,
    )?;
    let k = emit_linear(
        block,
        ctx.provider,
        &format!("{attn_prefix}.k_proj"),
        &normed_attn,
        &format!("l{li}_k_proj"),
        warnings,
    )?;
    let v = emit_linear(
        block,
        ctx.provider,
        &format!("{attn_prefix}.v_proj"),
        &normed_attn,
        &format!("l{li}_v_proj"),
        warnings,
    )?;

    // Split q_proj output: first half = Q, second half = gate
    let q_dim = (config.num_attention_heads * config.head_dim) as i64;
    let (q, attn_gate) = {
        let q_name = format!("l{li}_q_split_q");
        let gate_name = format!("l{li}_q_split_gate");
        let op = Operation::new("split", format!("l{li}_q_gate_split_op"))
            .with_input("x", Value::Reference(q_gate_proj))
            .with_attr(
                "split_sizes",
                Value::List(vec![Value::Int(q_dim), Value::Int(q_dim)]),
            )
            .with_attr("axis", Value::Int(-1))
            .with_output(&q_name)
            .with_output(&gate_name);
        block.add_op(op);
        (q_name, gate_name)
    };

    // 3. Reshape Q/K/V to [B, T, num_heads, head_dim]
    let q_reshaped = emit_reshape(
        block,
        &q,
        &format!("l{li}_q_reshape"),
        &[
            0,
            -1,
            config.num_attention_heads as i64,
            config.head_dim as i64,
        ],
    );
    let k_reshaped = emit_reshape(
        block,
        &k,
        &format!("l{li}_k_reshape"),
        &[
            0,
            -1,
            config.num_key_value_heads as i64,
            config.head_dim as i64,
        ],
    );
    let v_reshaped = emit_reshape(
        block,
        &v,
        &format!("l{li}_v_reshape"),
        &[
            0,
            -1,
            config.num_key_value_heads as i64,
            config.head_dim as i64,
        ],
    );

    // 4. QK norms (RMSNorm along head_dim, the last dim)
    let q_normed = emit_rms_norm_decomposed(
        block,
        ctx.provider,
        ctx.config,
        &format!("{attn_prefix}.q_norm"),
        &q_reshaped,
        &format!("l{li}_q_norm"),
        4,
        warnings,
    )?;
    let k_normed = emit_rms_norm_decomposed(
        block,
        ctx.provider,
        ctx.config,
        &format!("{attn_prefix}.k_norm"),
        &k_reshaped,
        &format!("l{li}_k_norm"),
        4,
        warnings,
    )?;

    // 5. Transpose to [B, heads, T, head_dim]
    let q_t = emit_transpose(
        block,
        &q_normed,
        &format!("l{li}_q_transpose"),
        &[0, 2, 1, 3],
    );
    let k_t = emit_transpose(
        block,
        &k_normed,
        &format!("l{li}_k_transpose"),
        &[0, 2, 1, 3],
    );
    let v_t = emit_transpose(
        block,
        &v_reshaped,
        &format!("l{li}_v_transpose"),
        &[0, 2, 1, 3],
    );

    // 6. Apply RoPE (with partial rotary factor support)
    let (q_roped, k_roped) = if rotary_dim < config.head_dim {
        emit_partial_rotary_embedding(
            block,
            &q_t,
            &k_t,
            li,
            ctx.rope_cos,
            ctx.rope_sin,
            rotary_dim,
            config.head_dim,
        )
    } else {
        emit_rotary_embedding_inline(
            block,
            &q_t,
            &k_t,
            li,
            ctx.rope_cos,
            ctx.rope_sin,
            config.head_dim,
        )
    };

    // 7. GQA: expand K and V heads when using grouped query attention
    let (k_expanded, v_expanded) = if config.num_attention_heads != config.num_key_value_heads {
        if config.num_attention_heads % config.num_key_value_heads != 0 {
            return Err(MilError::Validation(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                config.num_attention_heads, config.num_key_value_heads
            )));
        }
        let n_rep = config.num_attention_heads / config.num_key_value_heads;
        let k_exp = emit_gqa_expand_coreml(block, &k_roped, n_rep, config, li, "k");
        let v_exp = emit_gqa_expand_coreml(block, &v_t, n_rep, config, li, "v");
        (k_exp, v_exp)
    } else {
        (k_roped, v_t)
    };

    // 8. QK^T matmul
    let k_t2 = emit_transpose(
        block,
        &k_expanded,
        &format!("l{li}_k_transpose2"),
        &[0, 1, 3, 2],
    );
    let qk = {
        let out_name = format!("l{li}_qk_matmul");
        let op = Operation::new("matmul", format!("l{li}_qk_matmul_op"))
            .with_input("x", Value::Reference(q_roped))
            .with_input("y", Value::Reference(k_t2))
            .with_attr("transpose_x", Value::Bool(false))
            .with_attr("transpose_y", Value::Bool(false))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Scale by 1/sqrt(head_dim)
    let scale = {
        let scale_val = 1.0 / (config.head_dim as f64).sqrt();
        let scale_const = format!("l{li}_attn_scale_const");
        let op = Operation::new("const", &scale_const)
            .with_attr(
                "val",
                Value::Tensor {
                    data: TensorData::inline(half::f16::from_f64(scale_val).to_le_bytes().to_vec()),
                    shape: vec![1, 1, 1, 1],
                    dtype: ScalarType::Float16,
                },
            )
            .with_output(&scale_const);
        block.add_op(op);

        let out_name = format!("l{li}_attn_scaled");
        let op = Operation::new("mul", format!("l{li}_attn_scale_op"))
            .with_input("x", Value::Reference(qk))
            .with_input("y", Value::Reference(scale_const))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Apply causal mask — expand [B, T, T] to [B, 1, T, T] for head dim broadcast
    let mask_4d = {
        let out_name = format!("l{li}_causal_mask_4d");
        let op = Operation::new("expand_dims", format!("l{li}_causal_mask_expand_op"))
            .with_input("x", Value::Reference("causal_mask".into()))
            .with_attr("axes", Value::List(vec![Value::Int(1)]))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };
    let masked = {
        let out_name = format!("l{li}_attn_masked");
        let op = Operation::new("add", format!("l{li}_attn_mask_op"))
            .with_input("x", Value::Reference(scale))
            .with_input("y", Value::Reference(mask_4d))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Softmax
    let attn_weights = {
        let out_name = format!("l{li}_attn_weights");
        let op = Operation::new("softmax", format!("l{li}_softmax_op"))
            .with_input("x", Value::Reference(masked))
            .with_attr("axis", Value::Int(-1))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Attention @ V
    let attn_output = {
        let out_name = format!("l{li}_attn_output");
        let op = Operation::new("matmul", format!("l{li}_av_matmul_op"))
            .with_input("x", Value::Reference(attn_weights))
            .with_input("y", Value::Reference(v_expanded))
            .with_attr("transpose_x", Value::Bool(false))
            .with_attr("transpose_y", Value::Bool(false))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // 9. Transpose back and reshape to [B, T, num_heads * head_dim]
    let attn_transposed = emit_transpose(
        block,
        &attn_output,
        &format!("l{li}_attn_out_transpose"),
        &[0, 2, 1, 3],
    );
    let attn_dim = (config.num_attention_heads * config.head_dim) as i64;
    let attn_flat = emit_reshape(
        block,
        &attn_transposed,
        &format!("l{li}_attn_out_reshape"),
        &[1, -1, attn_dim],
    );

    // 10. Apply attention output gate: attn_output * sigmoid(gate)
    let gate_act = emit_unary(
        block,
        "sigmoid",
        &attn_gate,
        &format!("l{li}_attn_gate_sigmoid"),
    );
    let gated_attn = {
        let out_name = format!("l{li}_attn_gated");
        let op = Operation::new("mul", format!("l{li}_attn_gate_mul_op"))
            .with_input("x", Value::Reference(attn_flat))
            .with_input("y", Value::Reference(gate_act))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // 11. Output projection
    let o_proj = emit_linear(
        block,
        ctx.provider,
        &format!("{attn_prefix}.o_proj"),
        &gated_attn,
        &format!("l{li}_o_proj"),
        warnings,
    )?;

    // 12. Residual add
    let post_attn = emit_residual_add(block, input, &o_proj, &format!("l{li}_post_attn_residual"));

    // 13. Post-attention RMSNorm
    let normed_mlp = emit_rms_norm_decomposed(
        block,
        ctx.provider,
        ctx.config,
        &format!("{prefix}.post_attention_layernorm"),
        &post_attn,
        &format!("l{li}_post_attn_norm"),
        3,
        warnings,
    )?;

    // 14. MLP (SwiGLU)
    let mlp_out = emit_mlp_silu_coreml(block, ctx.provider, ctx.config, li, &normed_mlp, warnings)?;

    // 15. Residual add
    let layer_out = emit_residual_add(block, &post_attn, &mlp_out, &format!("l{li}_output"));

    Ok(layer_out)
}

// ---------------------------------------------------------------------------
// GDN (Gated Delta Network) layer
// ---------------------------------------------------------------------------

fn emit_gdn_layer(
    block: &mut Block,
    provider: &dyn WeightProvider,
    config: &ModelConfig,
    gdn: &GdnConfig,
    layer_idx: usize,
    input: &str,
    attn_output_gate: bool,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{layer_idx}");
    let attn_prefix = format!("{prefix}.linear_attn");
    let li = layer_idx;

    // 1. Input RMSNorm
    let normed = emit_rms_norm_decomposed(
        block,
        provider,
        config,
        &format!("{prefix}.input_layernorm"),
        input,
        &format!("l{li}_input_norm"),
        3,
        warnings,
    )?;

    // 2. Projections
    let qkv = emit_linear(
        block,
        provider,
        &format!("{attn_prefix}.in_proj_qkv"),
        &normed,
        &format!("l{li}_in_proj_qkv"),
        warnings,
    )?;

    let z = emit_linear(
        block,
        provider,
        &format!("{attn_prefix}.in_proj_z"),
        &normed,
        &format!("l{li}_in_proj_z"),
        warnings,
    )?;

    let a = emit_linear(
        block,
        provider,
        &format!("{attn_prefix}.in_proj_a"),
        &normed,
        &format!("l{li}_in_proj_a"),
        warnings,
    )?;

    let b = emit_linear(
        block,
        provider,
        &format!("{attn_prefix}.in_proj_b"),
        &normed,
        &format!("l{li}_in_proj_b"),
        warnings,
    )?;

    // 3. Transpose QKV to [B, qkv_dim, T] for depthwise conv1d
    let qkv_chw = emit_transpose(block, &qkv, &format!("l{li}_qkv_to_chw"), &[0, 2, 1]);

    // 4. Depthwise causal conv1d
    let conv_weight_name = format!("{attn_prefix}.conv1d.weight");
    let conv_weight_const = format!("l{li}_conv1d_weight");
    emit_weight_const(block, provider, &conv_weight_name, &conv_weight_const, warnings)?;

    let conv_bias_name = format!("{attn_prefix}.conv1d.bias");
    let conv_bias_const = format!("l{li}_conv1d_bias");
    let has_conv_bias = provider.has_tensor(&conv_bias_name);
    if has_conv_bias {
        emit_weight_const(block, provider, &conv_bias_name, &conv_bias_const, warnings)?;
    }

    let conv_out = {
        let out_name = format!("l{li}_conv1d_out");
        let kernel = gdn.linear_conv_kernel_dim;
        let strides_data: Vec<u8> = [1i32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let dilations_data: Vec<u8> = [1i32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut op = Operation::new("conv", format!("l{li}_conv1d_op"))
            .with_input("x", Value::Reference(qkv_chw))
            .with_input("weight", Value::Reference(conv_weight_const))
            .with_attr("groups", Value::Int(gdn.qkv_dim() as i64))
            .with_attr("pad_type", Value::String("custom".into()))
            .with_attr(
                "pad",
                Value::List(vec![Value::Int(kernel as i64 - 1), Value::Int(0)]),
            )
            .with_attr(
                "strides",
                Value::Tensor {
                    data: TensorData::Inline(strides_data),
                    shape: vec![1],
                    dtype: ScalarType::Int32,
                },
            )
            .with_attr(
                "dilations",
                Value::Tensor {
                    data: TensorData::Inline(dilations_data),
                    shape: vec![1],
                    dtype: ScalarType::Int32,
                },
            )
            .with_output(&out_name);
        if has_conv_bias {
            op = op.with_input("bias", Value::Reference(conv_bias_const));
        }
        block.add_op(op);
        out_name
    };

    // Transpose back to [B, T, qkv_dim]
    let conv_out_bts = emit_transpose(block, &conv_out, &format!("l{li}_conv_to_btc"), &[0, 2, 1]);

    // 5. SiLU activation on qkv
    let qkv_act = emit_silu(block, &conv_out_bts, &format!("l{li}_qkv_silu"));

    // 6. Split into q, k, v
    let key_dim = gdn.key_dim();
    let value_dim = gdn.value_dim();
    let (gdn_q, gdn_k, gdn_v) = {
        let q_name = format!("l{li}_gdn_q");
        let k_name = format!("l{li}_gdn_k");
        let v_name = format!("l{li}_gdn_v");
        let op = Operation::new("split", format!("l{li}_qkv_split_op"))
            .with_input("x", Value::Reference(qkv_act))
            .with_attr(
                "split_sizes",
                Value::List(vec![
                    Value::Int(key_dim as i64),
                    Value::Int(key_dim as i64),
                    Value::Int(value_dim as i64),
                ]),
            )
            .with_attr("axis", Value::Int(-1))
            .with_output(&q_name)
            .with_output(&k_name)
            .with_output(&v_name);
        block.add_op(op);
        (q_name, k_name, v_name)
    };

    // 7. Reshape q, k to [B, T, num_k_heads, k_head_dim], v to [B, T, num_v_heads, v_head_dim]
    let q_4d = emit_reshape(
        block,
        &gdn_q,
        &format!("l{li}_gdn_q_reshape"),
        &[
            0,
            -1,
            gdn.linear_num_key_heads as i64,
            gdn.linear_key_head_dim as i64,
        ],
    );
    let k_4d = emit_reshape(
        block,
        &gdn_k,
        &format!("l{li}_gdn_k_reshape"),
        &[
            0,
            -1,
            gdn.linear_num_key_heads as i64,
            gdn.linear_key_head_dim as i64,
        ],
    );
    let v_4d = emit_reshape(
        block,
        &gdn_v,
        &format!("l{li}_gdn_v_reshape"),
        &[
            0,
            -1,
            gdn.linear_num_value_heads as i64,
            gdn.linear_value_head_dim as i64,
        ],
    );

    // 8. Transpose to [B, heads, T, dim]
    let q_t = emit_transpose(block, &q_4d, &format!("l{li}_gdn_q_t"), &[0, 2, 1, 3]);
    let k_t = emit_transpose(block, &k_4d, &format!("l{li}_gdn_k_t"), &[0, 2, 1, 3]);
    let v_t = emit_transpose(block, &v_4d, &format!("l{li}_gdn_v_t"), &[0, 2, 1, 3]);

    // 9. Compute gates
    // beta = sigmoid(b) → [B, T, num_v_heads]
    let beta = emit_unary(block, "sigmoid", &b, &format!("l{li}_beta"));
    // Reshape beta for broadcasting: [B, num_v_heads, 1, T]
    let beta_reshaped = emit_reshape(
        block,
        &beta,
        &format!("l{li}_beta_reshape"),
        &[1, -1, gdn.linear_num_value_heads as i64],
    );
    let beta_t = emit_transpose(
        block,
        &beta_reshaped,
        &format!("l{li}_beta_transpose"),
        &[0, 2, 1],
    );
    // beta is now [B, num_v_heads, T], expand to [B, num_v_heads, 1, T]
    let beta_4d = {
        let out_name = format!("l{li}_beta_4d");
        let op = Operation::new("expand_dims", format!("l{li}_beta_expand_op"))
            .with_input("x", Value::Reference(beta_t))
            .with_attr("axes", Value::List(vec![Value::Int(2)]))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // A_log and dt_bias as const tensors — reshape to rank 3 for CoreML broadcast
    let a_log_name = format!("{attn_prefix}.A_log");
    let a_log_const = format!("l{li}_A_log");
    emit_weight_const(block, provider, &a_log_name, &a_log_const, warnings)?;
    let a_log_3d = emit_reshape(
        block,
        &a_log_const,
        &format!("l{li}_A_log_3d"),
        &[1, 1, gdn.linear_num_value_heads as i64],
    );

    let dt_bias_name = format!("{attn_prefix}.dt_bias");
    let dt_bias_const = format!("l{li}_dt_bias");
    emit_weight_const(block, provider, &dt_bias_name, &dt_bias_const, warnings)?;
    let dt_bias_3d = emit_reshape(
        block,
        &dt_bias_const,
        &format!("l{li}_dt_bias_3d"),
        &[1, 1, gdn.linear_num_value_heads as i64],
    );

    // A = -exp(A_log) — all rank 3
    let a_exp = emit_unary(block, "exp", &a_log_3d, &format!("l{li}_a_exp"));
    let neg_one = {
        let name = format!("l{li}_neg_one");
        let op = Operation::new("const", &name)
            .with_attr(
                "val",
                Value::Tensor {
                    data: TensorData::inline(half::f16::from_f64(-1.0).to_le_bytes().to_vec()),
                    shape: vec![1, 1, 1],
                    dtype: ScalarType::Float16,
                },
            )
            .with_output(&name);
        block.add_op(op);
        name
    };
    let neg_a = {
        let out_name = format!("l{li}_neg_a");
        let op = Operation::new("mul", format!("l{li}_neg_a_op"))
            .with_input("x", Value::Reference(a_exp))
            .with_input("y", Value::Reference(neg_one))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // dt = softplus(a_proj + dt_bias) where softplus(x) = log(1 + exp(x))
    let a_plus_bias = {
        let out_name = format!("l{li}_a_plus_bias");
        let op = Operation::new("add", format!("l{li}_a_plus_bias_op"))
            .with_input("x", Value::Reference(a.clone()))
            .with_input("y", Value::Reference(dt_bias_3d))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };
    // softplus: log(1 + exp(x))
    let sp_exp = emit_unary(block, "exp", &a_plus_bias, &format!("l{li}_sp_exp"));
    let one_const = {
        let name = format!("l{li}_one");
        let op = Operation::new("const", &name)
            .with_attr(
                "val",
                Value::Tensor {
                    data: TensorData::inline(half::f16::from_f64(1.0).to_le_bytes().to_vec()),
                    shape: vec![1, 1, 1],
                    dtype: ScalarType::Float16,
                },
            )
            .with_output(&name);
        block.add_op(op);
        name
    };
    let sp_add = {
        let out_name = format!("l{li}_sp_add");
        let op = Operation::new("add", format!("l{li}_sp_add_op"))
            .with_input("x", Value::Reference(sp_exp))
            .with_input("y", Value::Reference(one_const))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };
    let dt = {
        let out_name = format!("l{li}_dt");
        // CoreML log epsilon must be scalar FP16 (rank 0)
        let eps_name = format!("l{li}_log_eps");
        let eps_op = Operation::new("const", &eps_name)
            .with_attr(
                "val",
                Value::Tensor {
                    data: TensorData::inline(half::f16::from_f64(1e-7).to_le_bytes().to_vec()),
                    shape: vec![],
                    dtype: ScalarType::Float16,
                },
            )
            .with_output(&eps_name);
        block.add_op(eps_op);
        let op = Operation::new("log", format!("l{li}_log_op"))
            .with_input("x", Value::Reference(sp_add))
            .with_input("epsilon", Value::Reference(eps_name))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // g = A * dt → [B, T, num_v_heads]
    let g = {
        let out_name = format!("l{li}_g");
        let op = Operation::new("mul", format!("l{li}_g_op"))
            .with_input("x", Value::Reference(neg_a))
            .with_input("y", Value::Reference(dt))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // 10. Quadratic attention formulation
    // Transpose g to [B, num_v_heads, T]
    let g_reshape = emit_reshape(
        block,
        &g,
        &format!("l{li}_g_reshape"),
        &[1, -1, gdn.linear_num_value_heads as i64],
    );
    let g_t = emit_transpose(block, &g_reshape, &format!("l{li}_g_transpose"), &[0, 2, 1]);

    // cumsum along time axis (axis=2 in [B, H, T])
    let cumg = {
        let out_name = format!("l{li}_cumg");
        let op = Operation::new("cumsum", format!("l{li}_cumsum_op"))
            .with_input("x", Value::Reference(g_t))
            .with_attr("axis", Value::Int(2))
            .with_attr("reverse", Value::Bool(false))
            .with_attr("exclusive", Value::Bool(false))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // decay_row = cumg.unsqueeze(-1) → [B, H, T, 1]
    let decay_row = {
        let out_name = format!("l{li}_decay_row");
        let op = Operation::new("expand_dims", format!("l{li}_decay_row_op"))
            .with_input("x", Value::Reference(cumg.clone()))
            .with_attr("axes", Value::List(vec![Value::Int(-1)]))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // decay_col = cumg.unsqueeze(-2) → [B, H, 1, T]
    let decay_col = {
        let out_name = format!("l{li}_decay_col");
        let op = Operation::new("expand_dims", format!("l{li}_decay_col_op"))
            .with_input("x", Value::Reference(cumg))
            .with_attr("axes", Value::List(vec![Value::Int(-2)]))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // decay = exp(decay_row - decay_col) → [B, H, T, T]
    let decay_diff = {
        let out_name = format!("l{li}_decay_diff");
        let op = Operation::new("sub", format!("l{li}_decay_diff_op"))
            .with_input("x", Value::Reference(decay_row))
            .with_input("y", Value::Reference(decay_col))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };
    let decay = emit_unary(block, "exp", &decay_diff, &format!("l{li}_decay"));

    // scores = matmul(q, k^T) → [B, H, T, T]
    let k_transposed = emit_transpose(
        block,
        &k_t,
        &format!("l{li}_gdn_k_transpose2"),
        &[0, 1, 3, 2],
    );
    let scores = {
        let out_name = format!("l{li}_gdn_scores");
        let op = Operation::new("matmul", format!("l{li}_gdn_scores_op"))
            .with_input("x", Value::Reference(q_t))
            .with_input("y", Value::Reference(k_transposed))
            .with_attr("transpose_x", Value::Bool(false))
            .with_attr("transpose_y", Value::Bool(false))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // attn = scores * decay * beta_broadcast
    let scores_decay = {
        let out_name = format!("l{li}_scores_decay");
        let op = Operation::new("mul", format!("l{li}_scores_decay_op"))
            .with_input("x", Value::Reference(scores))
            .with_input("y", Value::Reference(decay))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };
    let scores_gated = {
        let out_name = format!("l{li}_scores_gated");
        let op = Operation::new("mul", format!("l{li}_scores_gated_op"))
            .with_input("x", Value::Reference(scores_decay))
            .with_input("y", Value::Reference(beta_4d))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Apply causal mask (multiply by lower-triangular mask from input)
    // causal_mask is [B, T, T]; expand to [B, 1, T, T] for head broadcast
    let mask_4d = {
        let out_name = format!("l{li}_mask_4d");
        let op = Operation::new("expand_dims", format!("l{li}_mask_expand_op"))
            .with_input("x", Value::Reference("causal_mask".into()))
            .with_attr("axes", Value::List(vec![Value::Int(1)]))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };
    let attn_masked = {
        let out_name = format!("l{li}_gdn_attn_masked");
        let op = Operation::new("mul", format!("l{li}_gdn_mask_op"))
            .with_input("x", Value::Reference(scores_gated))
            .with_input("y", Value::Reference(mask_4d))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // output = matmul(attn, v) → [B, H, T, D_v]
    let gdn_output = {
        let out_name = format!("l{li}_gdn_av_out");
        let op = Operation::new("matmul", format!("l{li}_gdn_av_op"))
            .with_input("x", Value::Reference(attn_masked))
            .with_input("y", Value::Reference(v_t))
            .with_attr("transpose_x", Value::Bool(false))
            .with_attr("transpose_y", Value::Bool(false))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // 11. Transpose back to [B, T, H, D_v] (keep per-head layout for gated norm)
    let gdn_out_t = emit_transpose(
        block,
        &gdn_output,
        &format!("l{li}_gdn_out_transpose"),
        &[0, 2, 1, 3],
    );

    // 12. Gated RMSNorm: norm(attn_out) * weight * silu(z)
    // Applied per head_v_dim — weight shape is [head_v_dim], not [value_dim]
    // gdn_out_t is [B, T, H, D_v], z is [B, T, value_dim] = [B, T, H * D_v]
    let z_per_head = emit_reshape(
        block,
        &z,
        &format!("l{li}_z_per_head"),
        &[
            1,
            -1,
            gdn.linear_num_value_heads as i64,
            gdn.linear_value_head_dim as i64,
        ],
    );

    // Apply RMSNorm to gdn_out_t on last dim (head_v_dim), input_rank=4
    let gdn_normed = if attn_output_gate {
        emit_rms_norm_decomposed(
            block,
            provider,
            config,
            &format!("model.layers.{li}.linear_attn.norm"),
            &gdn_out_t,
            &format!("l{li}_o_norm"),
            4,
            warnings,
        )?
    } else {
        gdn_out_t
    };

    // silu(z) gating
    let z_act = emit_silu(block, &z_per_head, &format!("l{li}_z_silu"));
    let gdn_gated_per_head = {
        let out_name = format!("l{li}_gdn_gated_per_head");
        let op = Operation::new("mul", format!("l{li}_gdn_gated_op"))
            .with_input("x", Value::Reference(gdn_normed))
            .with_input("y", Value::Reference(z_act))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Reshape back to [B, T, value_dim]
    let gdn_flat = emit_reshape(
        block,
        &gdn_gated_per_head,
        &format!("l{li}_gdn_out_reshape"),
        &[1, -1, value_dim as i64],
    );

    // 13. Output projection
    let out_proj = emit_linear(
        block,
        provider,
        &format!("{attn_prefix}.out_proj"),
        &gdn_flat,
        &format!("l{li}_gdn_out_proj"),
        warnings,
    )?;

    // 14. Residual add
    let post_attn = emit_residual_add(
        block,
        input,
        &out_proj,
        &format!("l{li}_post_attn_residual"),
    );

    // 15. Post-attention RMSNorm
    let normed_mlp = emit_rms_norm_decomposed(
        block,
        provider,
        config,
        &format!("{prefix}.post_attention_layernorm"),
        &post_attn,
        &format!("l{li}_post_attn_norm"),
        3,
        warnings,
    )?;

    // 16. MLP (SwiGLU)
    let mlp_out = emit_mlp_silu_coreml(block, provider, config, li, &normed_mlp, warnings)?;

    // 17. Residual add
    let layer_out = emit_residual_add(block, &post_attn, &mlp_out, &format!("l{li}_output"));

    Ok(layer_out)
}

// ---------------------------------------------------------------------------
// CoreML-compatible RMSNorm decomposition
// ---------------------------------------------------------------------------

/// Decompose RMSNorm into basic CoreML-compatible MIL ops:
/// `x / sqrt(mean(x² + eps)) * weight`
///
/// `input_rank` specifies the rank of the input tensor so the norm weight
/// can be reshaped to match (CoreML requires same-rank tensors for mul).
fn emit_rms_norm_decomposed(
    block: &mut Block,
    provider: &dyn WeightProvider,
    config: &ModelConfig,
    weight_prefix: &str,
    input: &str,
    op_prefix: &str,
    input_rank: usize,
    _warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    // Load weight and emit with correct shape for broadcasting.
    // CoreML requires same-rank tensors for mul, so we reshape [H] → [1,1,H] etc.
    let weight_name = format!("{weight_prefix}.weight");
    let weight_const = format!("{op_prefix}_weight");
    {
        let tensor = provider
            .tensor(&weight_name)
            .map_err(|e| MilError::Validation(format!("missing weight '{weight_name}': {e}")))?;
        let data = if tensor.data.len() <= 4096 {
            TensorData::inline(tensor.data.into_owned())
        } else {
            TensorData::external(weight_name.clone(), tensor.data.len())
        };
        // Build shape with leading 1s: [H] → [1,...,1,H] to match input_rank
        let mut shape = vec![1usize; input_rank.saturating_sub(1)];
        if let Some(&last) = tensor.shape.last() {
            shape.push(last);
        }
        let op = Operation::new("const", &weight_const)
            .with_attr(
                "val",
                Value::Tensor {
                    data,
                    shape,
                    dtype: tensor.dtype,
                },
            )
            .with_attr("onnx_name", Value::String(weight_name))
            .with_output(&weight_const);
        block.add_op(op);
    }

    // x² = x * x
    let x_sq = {
        let out = format!("{op_prefix}_sq");
        let op = Operation::new("mul", format!("{op_prefix}_sq_op"))
            .with_input("x", Value::Reference(input.into()))
            .with_input("y", Value::Reference(input.into()))
            .with_output(&out);
        block.add_op(op);
        out
    };

    // Add epsilon to x² BEFORE reduce_mean (avoids rank mismatch in CoreML).
    // mean(x² + eps) = mean(x²) + eps, mathematically equivalent.
    let eps_const = {
        let name = format!("{op_prefix}_eps");
        let eps_f16 = half::f16::from_f32(config.rms_norm_eps as f32);
        let eps_shape = vec![1usize; input_rank]; // match input rank for broadcast
        let op = Operation::new("const", &name)
            .with_attr(
                "val",
                Value::Tensor {
                    data: TensorData::inline(eps_f16.to_le_bytes().to_vec()),
                    shape: eps_shape,
                    dtype: ScalarType::Float16,
                },
            )
            .with_output(&name);
        block.add_op(op);
        name
    };
    let x_sq_eps = {
        let out = format!("{op_prefix}_sq_eps");
        let op = Operation::new("add", format!("{op_prefix}_sq_eps_op"))
            .with_input("x", Value::Reference(x_sq))
            .with_input("y", Value::Reference(eps_const))
            .with_output(&out);
        block.add_op(op);
        out
    };

    // mean(x² + eps, axis=-1, keepdim=true)
    let mean_sq = {
        let out = format!("{op_prefix}_mean_sq");
        let op = Operation::new("reduce_mean", format!("{op_prefix}_mean_sq_op"))
            .with_input("x", Value::Reference(x_sq_eps))
            .with_attr("axes", Value::List(vec![Value::Int(-1)]))
            .with_attr("keep_dims", Value::Bool(true))
            .with_output(&out);
        block.add_op(op);
        out
    };

    // sqrt then reciprocal
    let rms = {
        let out = format!("{op_prefix}_sqrt");
        let op = Operation::new("sqrt", format!("{op_prefix}_sqrt_op"))
            .with_input("x", Value::Reference(mean_sq))
            .with_output(&out);
        block.add_op(op);
        out
    };
    // rrms = x / sqrt(mean(x² + eps))  (use real_div directly: x / rms)
    let x_normed = {
        let out = format!("{op_prefix}_normed");
        let op = Operation::new("real_div", format!("{op_prefix}_rdiv_op"))
            .with_input("x", Value::Reference(input.into()))
            .with_input("y", Value::Reference(rms))
            .with_output(&out);
        block.add_op(op);
        out
    };

    // * weight (scale)
    let out_name = format!("{op_prefix}_out");
    let op = Operation::new("mul", format!("{op_prefix}_scale_op"))
        .with_input("x", Value::Reference(x_normed))
        .with_input("y", Value::Reference(weight_const))
        .with_output(&out_name);
    block.add_op(op);

    Ok(out_name)
}

// ---------------------------------------------------------------------------
// CoreML-compatible GQA expansion (single -1 dimension)
// ---------------------------------------------------------------------------

/// Expand KV heads for GQA. Uses batch=1 to avoid multiple -1 dims in reshape.
fn emit_gqa_expand_coreml(
    block: &mut Block,
    input: &str,
    n_rep: usize,
    config: &ModelConfig,
    layer_idx: usize,
    name: &str,
) -> String {
    let prefix = format!("l{layer_idx}_{name}_gqa");

    // Reshape to [1, num_kv_heads, 1, seq, head_dim]
    let reshaped_5d = emit_reshape(
        block,
        input,
        &format!("{prefix}_reshape5d"),
        &[
            1,
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

    // Reshape back to [1, num_attention_heads, seq, head_dim]
    emit_reshape(
        block,
        &tiled,
        &format!("{prefix}_reshape4d"),
        &[
            1,
            config.num_attention_heads as i64,
            -1,
            config.head_dim as i64,
        ],
    )
}

// ---------------------------------------------------------------------------
// CoreML-compatible SiLU decomposition
// ---------------------------------------------------------------------------

/// Decompose SiLU into `x * sigmoid(x)` (CoreML MIL has no native `silu` op).
fn emit_silu(block: &mut Block, input: &str, out_name: &str) -> String {
    let sig = {
        let sig_name = format!("{out_name}_sigmoid");
        let op = Operation::new("sigmoid", format!("{out_name}_sigmoid_op"))
            .with_input("x", Value::Reference(input.into()))
            .with_output(&sig_name);
        block.add_op(op);
        sig_name
    };

    let op = Operation::new("mul", format!("{out_name}_op"))
        .with_input("x", Value::Reference(input.into()))
        .with_input("y", Value::Reference(sig))
        .with_output(out_name);
    block.add_op(op);
    out_name.to_string()
}

// ---------------------------------------------------------------------------
// Helper: emit a single unary op
// ---------------------------------------------------------------------------

fn emit_unary(block: &mut Block, op_type: &str, input: &str, out_name: &str) -> String {
    let op = Operation::new(op_type, format!("{out_name}_op"))
        .with_input("x", Value::Reference(input.into()))
        .with_output(out_name);
    block.add_op(op);
    out_name.to_string()
}

// ---------------------------------------------------------------------------
// Partial RoPE helpers
// ---------------------------------------------------------------------------

/// Emit RoPE cos/sin tables sized to `rotary_dim` (< head_dim for partial rotary).
fn emit_partial_rope_tables(
    block: &mut Block,
    config: &ModelConfig,
    rotary_dim: usize,
) -> (String, String) {
    // Cap max_pos to avoid huge inline tables (Qwen3.5 has max_pos=262144)
    let max_pos = config.max_position_embeddings.min(4096);
    let theta = config.rope_theta;
    let half_dim = rotary_dim / 2;

    let mut cos_bytes = Vec::with_capacity(max_pos * half_dim * 2);
    let mut sin_bytes = Vec::with_capacity(max_pos * half_dim * 2);

    for t in 0..max_pos {
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / rotary_dim as f64);
            let angle = t as f64 * freq;
            cos_bytes.extend_from_slice(&half::f16::from_f64(angle.cos()).to_le_bytes());
            sin_bytes.extend_from_slice(&half::f16::from_f64(angle.sin()).to_le_bytes());
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
                dtype: ScalarType::Float16,
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
                dtype: ScalarType::Float16,
            },
        )
        .with_output(&sin_name);
    block.add_op(sin_op);

    (cos_name, sin_name)
}

/// Apply RoPE to only the first `rotary_dim` dimensions, leaving the rest unchanged.
/// Input Q/K have shape [B, heads, T, head_dim].
fn emit_partial_rotary_embedding(
    block: &mut Block,
    q_name: &str,
    k_name: &str,
    layer_idx: usize,
    cos_table: &str,
    sin_table: &str,
    rotary_dim: usize,
    head_dim: usize,
) -> (String, String) {
    let prefix = format!("l{layer_idx}_rope");
    let half_dim = rotary_dim / 2;
    let pass_through_dim = head_dim - rotary_dim;

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

    // Reshape to [B, 1, T, half_dim] for broadcasting over heads
    let cos_reshaped = emit_reshape(
        block,
        &cos_gathered,
        &format!("{prefix}_cos_reshape"),
        &[1, 1, -1, half_dim as i64],
    );
    let sin_reshaped = emit_reshape(
        block,
        &sin_gathered,
        &format!("{prefix}_sin_reshape"),
        &[1, 1, -1, half_dim as i64],
    );

    let q_out = emit_partial_rope_single(
        block,
        q_name,
        &cos_reshaped,
        &sin_reshaped,
        &format!("{prefix}_q"),
        rotary_dim,
        pass_through_dim,
    );
    let k_out = emit_partial_rope_single(
        block,
        k_name,
        &cos_reshaped,
        &sin_reshaped,
        &format!("{prefix}_k"),
        rotary_dim,
        pass_through_dim,
    );

    (q_out, k_out)
}

/// Apply partial RoPE to a single tensor [B, heads, T, head_dim].
fn emit_partial_rope_single(
    block: &mut Block,
    input: &str,
    cos: &str,
    sin: &str,
    prefix: &str,
    rotary_dim: usize,
    pass_through_dim: usize,
) -> String {
    // Split into rotary and pass-through portions
    let rot_part = format!("{prefix}_rot_part");
    let pass_part = format!("{prefix}_pass_part");
    let split_op = Operation::new("split", format!("{prefix}_partial_split_op"))
        .with_input("x", Value::Reference(input.into()))
        .with_attr(
            "split_sizes",
            Value::List(vec![
                Value::Int(rotary_dim as i64),
                Value::Int(pass_through_dim as i64),
            ]),
        )
        .with_attr("axis", Value::Int(-1))
        .with_output(&rot_part)
        .with_output(&pass_part);
    block.add_op(split_op);

    // Apply standard RoPE rotation to the rotary portion
    let roped = emit_rope_apply(block, &rot_part, cos, sin, prefix);

    // Concatenate rotated + pass-through
    let out_name = format!("{prefix}_partial_rope_out");
    let concat_op = Operation::new("concat", format!("{prefix}_partial_concat_op"))
        .with_input(
            "values",
            Value::List(vec![Value::Reference(roped), Value::Reference(pass_part)]),
        )
        .with_attr("axis", Value::Int(-1))
        .with_attr("interleave", Value::Bool(false))
        .with_output(&out_name);
    block.add_op(concat_op);
    out_name
}

/// Apply full RoPE to Q and K (inline, without using shared emit_rotary_embedding
/// which depends on config.head_dim for reshaping).
fn emit_rotary_embedding_inline(
    block: &mut Block,
    q_name: &str,
    k_name: &str,
    layer_idx: usize,
    cos_table: &str,
    sin_table: &str,
    head_dim: usize,
) -> (String, String) {
    let prefix = format!("l{layer_idx}_rope");
    let half_dim = head_dim / 2;

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

    let cos_reshaped = emit_reshape(
        block,
        &cos_gathered,
        &format!("{prefix}_cos_reshape"),
        &[1, 1, -1, half_dim as i64],
    );
    let sin_reshaped = emit_reshape(
        block,
        &sin_gathered,
        &format!("{prefix}_sin_reshape"),
        &[1, 1, -1, half_dim as i64],
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::templates::shared::StubProvider;
    use crate::weights::Architecture;

    fn tiny_qwen35_config() -> ModelConfig {
        let mut config = ModelConfig::new(Architecture::Qwen35)
            .with_hidden_size(64)
            .with_intermediate_size(128)
            .with_num_hidden_layers(4) // 3 GDN + 1 full attn
            .with_num_attention_heads(4)
            .with_num_key_value_heads(2)
            .with_head_dim(16)
            .with_vocab_size(256)
            .with_max_position_embeddings(128)
            .with_rms_norm_eps(1e-5)
            .with_rope_theta(10000000.0)
            .with_tie_word_embeddings(true);
        config.extra.insert(
            "layer_types".into(),
            serde_json::json!([
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ]),
        );
        config
            .extra
            .insert("linear_key_head_dim".into(), serde_json::json!(8));
        config
            .extra
            .insert("linear_value_head_dim".into(), serde_json::json!(8));
        config
            .extra
            .insert("linear_num_key_heads".into(), serde_json::json!(4));
        config
            .extra
            .insert("linear_num_value_heads".into(), serde_json::json!(4));
        config
            .extra
            .insert("linear_conv_kernel_dim".into(), serde_json::json!(4));
        config
            .extra
            .insert("attn_output_gate".into(), serde_json::json!(true));
        config
            .extra
            .insert("partial_rotary_factor".into(), serde_json::json!(0.25));
        config
    }

    /// Populate weights for the Qwen 3.5 model (both GDN and full attention layers).
    fn with_qwen35_weights(provider: StubProvider) -> StubProvider {
        let config = provider.config.clone();
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let vocab = config.vocab_size;

        let layer_types = parse_layer_types(&config);

        let gdn = GdnConfig::from_model_config(&config).expect("GDN config should parse");
        let value_dim = gdn.value_dim();
        let qkv_dim = gdn.qkv_dim();
        let num_v_heads = gdn.linear_num_value_heads;
        let conv_kernel = gdn.linear_conv_kernel_dim;

        let attn_output_gate = config
            .extra
            .get("attn_output_gate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mut p = provider.with_tensor(
            "model.embed_tokens.weight",
            &[vocab, h],
            ScalarType::Float16,
        );

        for (l, lt) in layer_types.iter().enumerate() {
            let prefix = format!("model.layers.{l}");
            // Shared: input/post-attention norms, MLP
            p = p.with_tensor(
                &format!("{prefix}.input_layernorm.weight"),
                &[h],
                ScalarType::Float16,
            );
            p = p.with_tensor(
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[h],
                ScalarType::Float16,
            );
            p = p.with_tensor(
                &format!("{prefix}.mlp.gate_proj.weight"),
                &[inter, h],
                ScalarType::Float16,
            );
            p = p.with_tensor(
                &format!("{prefix}.mlp.up_proj.weight"),
                &[inter, h],
                ScalarType::Float16,
            );
            p = p.with_tensor(
                &format!("{prefix}.mlp.down_proj.weight"),
                &[h, inter],
                ScalarType::Float16,
            );

            if lt == "linear_attention" {
                // GDN-specific weights — uses `linear_attn` prefix
                let ap = format!("{prefix}.linear_attn");
                p = p.with_tensor(
                    &format!("{ap}.in_proj_qkv.weight"),
                    &[qkv_dim, h],
                    ScalarType::Float16,
                );
                p = p.with_tensor(
                    &format!("{ap}.in_proj_z.weight"),
                    &[value_dim, h],
                    ScalarType::Float16,
                );
                p = p.with_tensor(
                    &format!("{ap}.in_proj_a.weight"),
                    &[num_v_heads, h],
                    ScalarType::Float16,
                );
                p = p.with_tensor(
                    &format!("{ap}.in_proj_b.weight"),
                    &[num_v_heads, h],
                    ScalarType::Float16,
                );
                p = p.with_tensor(
                    &format!("{ap}.conv1d.weight"),
                    &[qkv_dim, 1, conv_kernel],
                    ScalarType::Float16,
                );
                // No conv1d.bias — Qwen 3.5 HF weights don't include it
                p = p.with_tensor(&format!("{ap}.A_log"), &[num_v_heads], ScalarType::Float16);
                p = p.with_tensor(
                    &format!("{ap}.dt_bias"),
                    &[num_v_heads],
                    ScalarType::Float16,
                );
                p = p.with_tensor(
                    &format!("{ap}.out_proj.weight"),
                    &[h, value_dim],
                    ScalarType::Float16,
                );
                if attn_output_gate {
                    // Output gate norm is per head_v_dim, not per hidden_size
                    p = p.with_tensor(
                        &format!("{ap}.norm.weight"),
                        &[gdn.linear_value_head_dim],
                        ScalarType::Float16,
                    );
                }
            } else {
                // Full attention weights — no bias, with QK norms
                // q_proj packs both Q and attention gate (2x head_dim per head)
                let ap = format!("{prefix}.self_attn");
                p = p.with_tensor(
                    &format!("{ap}.q_proj.weight"),
                    &[n_heads * head_dim * 2, h],
                    ScalarType::Float16,
                );
                p = p.with_tensor(
                    &format!("{ap}.k_proj.weight"),
                    &[n_kv_heads * head_dim, h],
                    ScalarType::Float16,
                );
                p = p.with_tensor(
                    &format!("{ap}.v_proj.weight"),
                    &[n_kv_heads * head_dim, h],
                    ScalarType::Float16,
                );
                p = p.with_tensor(
                    &format!("{ap}.o_proj.weight"),
                    &[h, n_heads * head_dim],
                    ScalarType::Float16,
                );
                // QK norms
                p = p.with_tensor(
                    &format!("{ap}.q_norm.weight"),
                    &[head_dim],
                    ScalarType::Float16,
                );
                p = p.with_tensor(
                    &format!("{ap}.k_norm.weight"),
                    &[head_dim],
                    ScalarType::Float16,
                );
                // No output gate norm for full attention layers
            }
        }

        p = p.with_tensor("model.norm.weight", &[h], ScalarType::Float16);
        // tie_word_embeddings=true so no separate lm_head weight needed.
        if !config.tie_word_embeddings {
            p = p.with_tensor("lm_head.weight", &[vocab, h], ScalarType::Float16);
        }

        p
    }

    #[test]
    fn build_program_succeeds_with_all_weights() {
        let config = tiny_qwen35_config();
        let provider = with_qwen35_weights(StubProvider::new(config));

        let result = build_program(&provider).expect("build_program should succeed");
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
    fn build_program_with_gdn_layers() {
        let config = tiny_qwen35_config();
        let provider = with_qwen35_weights(StubProvider::new(config));

        let result = build_program(&provider).expect("build_program should succeed");
        let main = result.program.main().unwrap();

        // Verify GDN-specific ops are present (conv for layer 0)
        let has_conv = main
            .body
            .operations
            .iter()
            .any(|op| op.name == "l0_conv1d_op");
        assert!(has_conv, "should emit conv1d op for GDN layer 0");

        // Verify cumsum ops are present for GDN layers
        let has_cumsum = main
            .body
            .operations
            .iter()
            .any(|op| op.name == "l0_cumsum_op");
        assert!(has_cumsum, "should emit cumsum op for GDN layer 0");

        // Verify full attention ops for layer 3
        let has_softmax = main
            .body
            .operations
            .iter()
            .any(|op| op.name == "l3_softmax_op");
        assert!(
            has_softmax,
            "should emit softmax op for full attention layer 3"
        );
    }

    #[test]
    fn build_program_with_custom_layer_types() {
        // All full-attention layers
        let mut config = tiny_qwen35_config();
        config.extra.insert(
            "layer_types".into(),
            serde_json::json!([
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention"
            ]),
        );
        let provider = with_qwen35_weights(StubProvider::new(config));

        let result = build_program(&provider).expect("build_program should succeed");
        let main = result.program.main().unwrap();

        // No GDN ops should be present
        let has_conv = main.body.operations.iter().any(|op| op.op_type == "conv");
        assert!(!has_conv, "no conv ops for all-full-attention config");
    }

    #[test]
    fn build_program_errors_on_missing_weights() {
        let config = tiny_qwen35_config();
        let provider = StubProvider::new(config);

        let result = build_program(&provider);
        assert!(
            result.is_err(),
            "build_program should fail when required weights are missing"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("missing required weight"),
            "error should mention missing weight, got: {err}"
        );
    }

    #[test]
    fn program_is_marked_autoregressive() {
        let config = tiny_qwen35_config();
        let provider = with_qwen35_weights(StubProvider::new(config));
        let result = build_program(&provider).unwrap();
        assert!(result.program.is_autoregressive());
    }
}
