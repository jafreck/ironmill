//! Gemma architecture template.
//!
//! Builds a MIL IR [`Program`] for the Gemma model family (Gemma, Gemma 2,
//! Gemma 3, Gemma 4) from weight tensors provided by a [`WeightProvider`].
//!
//! Key differences from LLaMA:
//! - GELU activation in MLP instead of SiLU
//! - Embedding output is normalized by multiplying by sqrt(hidden_size)
//! - May use sliding window attention on alternating layers
//! - Uses RMSNorm (same as LLaMA/Qwen)
//!
//! Gemma 4 extensions:
//! - Per-layer attention types (sliding vs global) with different RoPE and head dims
//! - Per-Layer Embeddings (PLE)
//! - KV shared layers with anchor mapping
//! - Double-wide MLP on shared layers
//! - Final logit softcapping

use crate::weights::WeightProvider;
use mil_rs::MilError;
use mil_rs::convert::onnx_graph::ConversionResult;
use mil_rs::ir::{Block, Function, Operation, Program, ScalarType, TensorType, Value};

use super::moe;
use super::shared::{
    LayerContext, emit_attention, emit_embedding, emit_gather, emit_linear, emit_lm_head,
    emit_mlp_gelu, emit_residual_add, emit_rms_norm, emit_rope_tables,
    emit_rope_tables_with_params,
};

/// Build a complete MIL [`Program`] for a Gemma-family model.
pub fn build_program(provider: &dyn WeightProvider) -> Result<ConversionResult, MilError> {
    let config = provider.config().clone();
    let mut warnings: Vec<String> = Vec::new();

    let seq_len: Option<usize> = None; // dynamic
    let batch: Option<usize> = Some(1);

    // Check for sliding window configuration.
    let sliding_window: Option<usize> = config.sliding_window();

    if sliding_window.is_some() {
        warnings.push(
            "sliding_window attention on alternating layers is noted but mask handling is caller-provided"
                .into(),
        );
    }

    // Gemma 4: check for per-layer attention types
    let layer_types = config.layer_types();
    let rope_params = config.rope_parameters();

    // KV shared layers: precompute anchor mapping
    let num_kv_shared = config
        .extra
        .get("num_kv_shared_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let first_shared_idx = config.num_hidden_layers.saturating_sub(num_kv_shared);
    let mut kv_anchor: Vec<Option<usize>> = vec![None; config.num_hidden_layers];
    if num_kv_shared > 0 {
        if let Some(ref lts) = layer_types {
            let prev = &lts[..first_shared_idx];
            for layer_idx in first_shared_idx..config.num_hidden_layers {
                let lt = &lts[layer_idx];
                kv_anchor[layer_idx] = prev.iter().rposition(|t| t == lt);
            }
        }
    }

    // PLE config
    let ple_hidden_size = config
        .extra
        .get("hidden_size_per_layer_input")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let ple_vocab_size = config
        .extra
        .get("vocab_size_per_layer_input")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Double-wide MLP config
    let use_double_wide_mlp = config
        .extra
        .get("use_double_wide_mlp")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Final logit softcapping
    let final_logit_softcapping = config
        .extra
        .get("final_logit_softcapping")
        .and_then(|v| v.as_f64());

    // MoE config
    let enable_moe = config
        .extra
        .get("enable_moe_block")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let num_experts = config
        .extra
        .get("num_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let top_k_experts = config
        .extra
        .get("top_k_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Build the main function with typed inputs.
    let input_ids_ty = TensorType::with_dynamic_shape(ScalarType::Int32, vec![batch, seq_len]);
    let position_ids_ty = TensorType::with_dynamic_shape(ScalarType::Int32, vec![batch, seq_len]);
    let causal_mask_ty =
        TensorType::with_dynamic_shape(ScalarType::Float16, vec![batch, seq_len, seq_len]);

    let mut func = Function::new("main")
        .with_input("input_ids", input_ids_ty)
        .with_input("position_ids", position_ids_ty)
        .with_input("causal_mask", causal_mask_ty);

    let block = &mut func.body;

    // Embedding lookup.
    let embed_out = emit_embedding(block, provider, &config, &mut warnings)?;

    // Gemma-specific: normalize embeddings by multiplying by sqrt(hidden_size).
    let embed_normed = emit_embedding_norm(block, &config, &embed_out);

    // Emit RoPE tables — Gemma 4 needs separate tables per attention type
    let (default_cos, default_sin, global_cos, global_sin) = if let Some(ref rp) = rope_params {
        let (sc, ss) = if let Some(cfg) = rp.get("sliding_attention") {
            emit_rope_tables_with_params(
                block,
                config.head_dim,
                config.max_position_embeddings,
                cfg.theta,
                cfg.partial_rotary_factor,
                "rope_sliding",
            )
        } else {
            emit_rope_tables(block, &config)
        };
        let global_head_dim = config.global_head_dim();
        let (gc, gs) = if let Some(cfg) = rp.get("full_attention") {
            emit_rope_tables_with_params(
                block,
                global_head_dim,
                config.max_position_embeddings,
                cfg.theta,
                cfg.partial_rotary_factor,
                "rope_global",
            )
        } else {
            (sc.clone(), ss.clone())
        };
        (sc, ss, Some(gc), Some(gs))
    } else {
        let (c, s) = emit_rope_tables(block, &config);
        (c, s, None, None)
    };

    // PLE model-level computation (before layer loop)
    let per_layer_inputs = if ple_hidden_size > 0 && ple_vocab_size > 0 {
        // 1. Gather from embed_tokens_per_layer using input_ids
        let ple_embed = emit_gather(
            block,
            provider,
            "model.embed_tokens_per_layer",
            "input_ids",
            "ple_embed",
            &mut warnings,
        )?;

        // 2. Project inputs_embeds via per_layer_model_projection
        let ple_proj = emit_linear(
            block,
            provider,
            "model.per_layer_model_projection",
            &embed_normed,
            "ple_proj",
            &mut warnings,
        )?;

        // 3. Apply per_layer_projection_norm (RMSNorm)
        let ple_proj_normed = emit_rms_norm(
            block,
            provider,
            &config,
            "model.per_layer_projection_norm",
            &ple_proj,
            "ple_proj_norm",
            &mut warnings,
        )?;

        // 4. Sum: embed + proj
        let ple_sum = emit_residual_add(block, &ple_embed, &ple_proj_normed, "ple_sum");

        // 5. Scale by 2^(-0.5) = 0.7071067811865476
        let scale_const = "ple_scale_const".to_string();
        let op = Operation::new("const", &scale_const)
            .with_attr("val", Value::Float(std::f64::consts::FRAC_1_SQRT_2))
            .with_output(&scale_const);
        block.add_op(op);

        let ple_scaled = "ple_scaled".to_string();
        let op = Operation::new("mul", "ple_scale_op")
            .with_input("x", Value::Reference(ple_sum))
            .with_input("y", Value::Reference(scale_const))
            .with_output(&ple_scaled);
        block.add_op(op);

        Some(ple_scaled)
    } else {
        None
    };

    // Transformer layers.
    let mut hidden = embed_normed;
    for layer_idx in 0..config.num_hidden_layers {
        let lt = layer_types.as_ref().map(|lts| lts[layer_idx].as_str());
        let is_global = lt == Some("full_attention");

        // Select per-layer RoPE tables
        let (eff_cos, eff_sin) = if is_global {
            (
                global_cos.as_deref().unwrap_or(&default_cos),
                global_sin.as_deref().unwrap_or(&default_sin),
            )
        } else {
            (default_cos.as_str(), default_sin.as_str())
        };
        let eff_head_dim = if is_global {
            config.global_head_dim()
        } else {
            config.head_dim
        };
        let eff_num_kv_heads = if is_global {
            config.num_global_key_value_heads()
        } else {
            config.num_key_value_heads
        };

        // KV shared layers: skip K/V projection for shared layers
        let is_kv_shared = num_kv_shared > 0 && layer_idx >= first_shared_idx;

        // Double-wide MLP: layers in the shared region use 2x intermediate_size
        let _effective_intermediate = if use_double_wide_mlp && is_kv_shared {
            config.intermediate_size * 2
        } else {
            config.intermediate_size
        };

        let ctx = LayerContext {
            provider,
            config: &config,
            layer_idx,
            rope_cos: eff_cos,
            rope_sin: eff_sin,
            layer_type: lt,
            effective_head_dim: eff_head_dim,
            effective_num_kv_heads: eff_num_kv_heads,
        };

        // Per-layer PLE input slice (if PLE is active)
        let ple_layer_input = per_layer_inputs.as_ref().map(|pli| {
            let slice_name = format!("ple_layer_{layer_idx}_slice");
            let start = layer_idx * ple_hidden_size;
            let end = start + ple_hidden_size;
            let op = Operation::new("slice_by_index", format!("ple_layer_{layer_idx}_slice_op"))
                .with_input("x", Value::Reference(pli.clone()))
                .with_attr(
                    "begin",
                    Value::List(vec![Value::Int(0), Value::Int(0), Value::Int(start as i64)]),
                )
                .with_attr(
                    "end",
                    Value::List(vec![Value::Int(-1), Value::Int(-1), Value::Int(end as i64)]),
                )
                .with_output(&slice_name);
            block.add_op(op);
            slice_name
        });

        hidden = emit_gemma4_transformer_layer(
            block,
            &ctx,
            &hidden,
            kv_anchor[layer_idx],
            ple_layer_input.as_deref(),
            ple_hidden_size,
            enable_moe,
            num_experts,
            top_k_experts,
            &mut warnings,
        )?;
    }

    // Final RMSNorm.
    let normed = emit_rms_norm(
        block,
        provider,
        &config,
        "model.norm",
        &hidden,
        "final_norm",
        &mut warnings,
    )?;

    // LM head projection.
    let logits = emit_lm_head(block, provider, &config, &normed, &mut warnings)?;

    // Final logit softcapping: logits = softcap * tanh(logits / softcap)
    let final_output = if let Some(softcap) = final_logit_softcapping {
        let cap_const = "softcap_const".to_string();
        let op = Operation::new("const", &cap_const)
            .with_attr("val", Value::Float(softcap))
            .with_output(&cap_const);
        block.add_op(op);

        let inv_cap_const = "inv_softcap_const".to_string();
        let op = Operation::new("const", &inv_cap_const)
            .with_attr("val", Value::Float(1.0 / softcap))
            .with_output(&inv_cap_const);
        block.add_op(op);

        // logits / softcap
        let scaled = "logits_prescale".to_string();
        let op = Operation::new("mul", "softcap_prescale_op")
            .with_input("x", Value::Reference(logits))
            .with_input("y", Value::Reference(inv_cap_const))
            .with_output(&scaled);
        block.add_op(op);

        // tanh
        let tanh_out = "logits_tanh".to_string();
        let op = Operation::new("tanh", "softcap_tanh_op")
            .with_input("x", Value::Reference(scaled))
            .with_output(&tanh_out);
        block.add_op(op);

        // * softcap
        let capped = "logits_capped".to_string();
        let op = Operation::new("mul", "softcap_scale_op")
            .with_input("x", Value::Reference(tanh_out))
            .with_input("y", Value::Reference(cap_const))
            .with_output(&capped);
        block.add_op(op);

        capped
    } else {
        logits
    };

    block.outputs.push(final_output);

    let mut program = Program::new("1");
    program.add_function(func);
    program.set_attribute("autoregressive", "true");

    if let Some(sw) = sliding_window {
        program.set_attribute("sliding_window", sw.to_string());
    }

    // Emit layer_types as program attribute for inference
    if let Some(ref lts) = layer_types {
        let types_str = lts.join(",");
        program.set_attribute("layer_types", types_str);
    }

    Ok(ConversionResult::new(program, warnings))
}

// ---------------------------------------------------------------------------
// Gemma-specific embedding normalization
// ---------------------------------------------------------------------------

/// Multiply embedding output by sqrt(hidden_size) as Gemma prescribes.
fn emit_embedding_norm(
    block: &mut Block,
    config: &crate::weights::ModelConfig,
    input: &str,
) -> String {
    let scale_val = (config.hidden_size as f64).sqrt();
    let scale_const = "embed_norm_scale".to_string();
    let op = Operation::new("const", &scale_const)
        .with_attr("val", Value::Float(scale_val))
        .with_output(&scale_const);
    block.add_op(op);

    let out_name = "embed_normed".to_string();
    let op = Operation::new("mul", "embed_norm_op")
        .with_input("x", Value::Reference(input.into()))
        .with_input("y", Value::Reference(scale_const))
        .with_output(&out_name);
    block.add_op(op);
    out_name
}

// ---------------------------------------------------------------------------
// Gemma transformer layer
// ---------------------------------------------------------------------------

/// Emit a Gemma transformer layer with Gemma 4 feature support.
///
/// Handles standard Gemma layers and Gemma 4 extensions (KV sharing,
/// PLE, double-wide MLP). When Gemma 4 features are not configured,
/// behaves identically to the original Gemma layer.
#[allow(clippy::too_many_arguments)]
fn emit_gemma4_transformer_layer(
    block: &mut Block,
    ctx: &LayerContext<'_>,
    input: &str,
    _kv_anchor: Option<usize>,
    ple_input: Option<&str>,
    _ple_hidden_size: usize,
    enable_moe: bool,
    num_experts: usize,
    top_k_experts: usize,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let layer_idx = ctx.layer_idx;
    let prefix = format!("model.layers.{layer_idx}");

    // 1. Input RMSNorm
    let normed_attn = emit_rms_norm(
        block,
        ctx.provider,
        ctx.config,
        &format!("{prefix}.input_layernorm"),
        input,
        &format!("l{layer_idx}_input_norm"),
        warnings,
    )?;

    // 2. Self-attention
    // KV reuse from anchor layers is an inference-time concern, not IR-level.
    let attn_out = emit_attention(block, ctx, &normed_attn, warnings)?;

    // 3. Residual add
    let post_attn = emit_residual_add(
        block,
        input,
        &attn_out,
        &format!("l{layer_idx}_post_attn_residual"),
    );

    // 4. Post-attention RMSNorm
    let normed_mlp = emit_rms_norm(
        block,
        ctx.provider,
        ctx.config,
        &format!("{prefix}.post_attention_layernorm"),
        &post_attn,
        &format!("l{layer_idx}_post_attn_norm"),
        warnings,
    )?;

    // 5. MLP with GELU activation
    // Double-wide MLP: the weight tensors already have the correct shape,
    // so emit_mlp_gelu loads whatever shape the provider has.
    let mlp_out = emit_mlp_gelu(
        block,
        ctx.provider,
        ctx.config,
        layer_idx,
        &normed_mlp,
        warnings,
    )?;

    // 6. MoE block (parallel with dense MLP, outputs are summed) or standard residual
    let post_mlp = if enable_moe && num_experts > 0 {
        // 5b. MoE output from the same normed input
        let moe_out = moe::emit_moe_block(
            block,
            ctx.provider,
            layer_idx,
            &normed_mlp,
            num_experts,
            top_k_experts,
            warnings,
        )?;

        // Sum dense MLP + MoE outputs
        let combined = emit_residual_add(
            block,
            &mlp_out,
            &moe_out,
            &format!("l{layer_idx}_moe_combined"),
        );

        // Residual add with post-attention state
        emit_residual_add(
            block,
            &post_attn,
            &combined,
            &format!("l{layer_idx}_output"),
        )
    } else {
        // Standard: residual add with MLP output only
        emit_residual_add(block, &post_attn, &mlp_out, &format!("l{layer_idx}_output"))
    };

    // 7. Per-Layer Embedding (PLE) — applied AFTER the FFN block
    let layer_out = if let Some(ple_slice) = ple_input {
        // Gate: linear [hidden_size → ple_hidden_size]
        let gate = emit_linear(
            block,
            ctx.provider,
            &format!("{prefix}.per_layer_input_gate"),
            &post_mlp,
            &format!("l{layer_idx}_ple_gate"),
            warnings,
        )?;

        // Activation (gelu)
        let gate_act = {
            let out_name = format!("l{layer_idx}_ple_gelu");
            let op = Operation::new("gelu", format!("l{layer_idx}_ple_gelu_op"))
                .with_input("x", Value::Reference(gate))
                .with_output(&out_name);
            block.add_op(op);
            out_name
        };

        // Element-wise multiply with per_layer_input slice
        let gated = {
            let out_name = format!("l{layer_idx}_ple_gated");
            let op = Operation::new("mul", format!("l{layer_idx}_ple_mul_op"))
                .with_input("x", Value::Reference(gate_act))
                .with_input("y", Value::Reference(ple_slice.to_string()))
                .with_output(&out_name);
            block.add_op(op);
            out_name
        };

        // Project back: linear [ple_hidden_size → hidden_size]
        let projected = emit_linear(
            block,
            ctx.provider,
            &format!("{prefix}.per_layer_projection"),
            &gated,
            &format!("l{layer_idx}_ple_proj"),
            warnings,
        )?;

        // Post-PLE RMSNorm
        let ple_normed = emit_rms_norm(
            block,
            ctx.provider,
            ctx.config,
            &format!("{prefix}.post_per_layer_input_norm"),
            &projected,
            &format!("l{layer_idx}_ple_norm"),
            warnings,
        )?;

        // Residual add
        emit_residual_add(
            block,
            &post_mlp,
            &ple_normed,
            &format!("l{layer_idx}_ple_output"),
        )
    } else {
        post_mlp
    };

    Ok(layer_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::templates::shared::{StubProvider, tiny_llama_config};
    use crate::weights::Architecture;

    fn tiny_gemma_config() -> crate::weights::ModelConfig {
        let mut config = tiny_llama_config();
        config.architecture = Architecture::Gemma;
        config
    }

    #[test]
    fn build_program_succeeds_with_all_weights() {
        let config = tiny_gemma_config();
        let provider = StubProvider::new(config).with_llama_weights();

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
    fn build_program_uses_gelu_activation() {
        let config = tiny_gemma_config();
        let provider = StubProvider::new(config).with_llama_weights();

        let result = build_program(&provider).expect("build_program should succeed");
        let main = result.program.main().unwrap();

        // Should have gelu ops, not silu
        let gelu_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.op_type == "gelu")
            .collect();
        let silu_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.op_type == "silu")
            .collect();

        assert!(
            !gelu_ops.is_empty(),
            "Gemma should use GELU activation in MLP"
        );
        assert!(
            silu_ops.is_empty(),
            "Gemma should not use SiLU activation in MLP"
        );
    }

    #[test]
    fn build_program_normalizes_embeddings() {
        let config = tiny_gemma_config();
        let provider = StubProvider::new(config).with_llama_weights();

        let result = build_program(&provider).expect("build_program should succeed");
        let main = result.program.main().unwrap();

        // Should have embedding normalization scale const
        let has_embed_norm = main
            .body
            .operations
            .iter()
            .any(|op| op.name == "embed_norm_scale");
        assert!(
            has_embed_norm,
            "Gemma should emit embedding normalization scale"
        );

        // Should have the multiply op
        let has_embed_norm_op = main
            .body
            .operations
            .iter()
            .any(|op| op.name == "embed_norm_op");
        assert!(
            has_embed_norm_op,
            "Gemma should emit embedding normalization multiply"
        );
    }

    #[test]
    fn build_program_with_sliding_window() {
        let mut config = tiny_gemma_config();
        config
            .extra
            .insert("sliding_window".into(), serde_json::json!(4096));
        let provider = StubProvider::new(config).with_llama_weights();

        let result = build_program(&provider).expect("build_program should succeed");
        // sliding_window should be recorded as a program attribute for the runtime.
        assert_eq!(
            result.program.attributes.get("sliding_window"),
            Some(&"4096".to_string())
        );
    }

    #[test]
    fn program_is_marked_autoregressive() {
        let config = tiny_gemma_config();
        let provider = StubProvider::new(config).with_llama_weights();
        let result = build_program(&provider).unwrap();
        assert!(result.program.is_autoregressive());
    }

    #[test]
    fn build_program_errors_on_missing_weights() {
        let config = tiny_gemma_config();
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
    fn build_program_with_gqa() {
        let mut config = tiny_gemma_config();
        config.num_key_value_heads = 2;
        let provider = StubProvider::new(config).with_llama_weights();

        let result = build_program(&provider).expect("GQA build should succeed");
        let main = result.program.main().unwrap();
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
    fn build_program_with_tied_embeddings() {
        let mut config = tiny_gemma_config();
        config.tie_word_embeddings = true;
        let provider = StubProvider::new(config).with_llama_weights();

        let result = build_program(&provider).expect("build_program should succeed");
        let main = result.program.main().unwrap();

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
    fn build_program_gemma4_with_moe() {
        let mut config = tiny_gemma_config();
        config.num_hidden_layers = 2;
        config
            .extra
            .insert("enable_moe_block".into(), serde_json::json!(true));
        config
            .extra
            .insert("num_experts".into(), serde_json::json!(4));
        config
            .extra
            .insert("top_k_experts".into(), serde_json::json!(2));

        let moe_inter = 64; // moe_intermediate_size
        let h = config.hidden_size;
        let mut provider = StubProvider::new(config).with_llama_weights();
        // Add MoE weights for each layer
        for l in 0..2 {
            let p = format!("model.layers.{l}.mlp");
            provider =
                provider.with_tensor(&format!("{p}.router.weight"), &[4, h], ScalarType::Float16);
            for e in 0..4 {
                provider = provider
                    .with_tensor(
                        &format!("{p}.experts.{e}.gate_proj.weight"),
                        &[moe_inter, h],
                        ScalarType::Float16,
                    )
                    .with_tensor(
                        &format!("{p}.experts.{e}.up_proj.weight"),
                        &[moe_inter, h],
                        ScalarType::Float16,
                    )
                    .with_tensor(
                        &format!("{p}.experts.{e}.down_proj.weight"),
                        &[h, moe_inter],
                        ScalarType::Float16,
                    );
            }
        }

        let result = build_program(&provider).expect("MoE build should succeed");
        let main = result.program.main().unwrap();
        // Should have MoE-related ops
        let moe_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.name.contains("moe_"))
            .collect();
        assert!(!moe_ops.is_empty(), "MoE build should emit MoE ops");
        // Should also have standard MLP ops (dense + MoE in parallel)
        let gelu_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.op_type == "gelu")
            .collect();
        assert!(
            !gelu_ops.is_empty(),
            "MoE layers should still emit dense MLP gelu ops"
        );
    }

    #[test]
    fn build_program_gemma4_with_layer_types() {
        let mut config = tiny_gemma_config();
        config.extra.insert(
            "layer_types".into(),
            serde_json::json!(["sliding_attention", "full_attention"]),
        );
        config.extra.insert(
            "rope_parameters".into(),
            serde_json::json!({
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}
            }),
        );
        config.num_hidden_layers = 2;
        let provider = StubProvider::new(config).with_llama_weights();
        let result = build_program(&provider).expect("Gemma 4 build should succeed");
        assert_eq!(result.program.functions.len(), 1);
        assert_eq!(
            result.program.attributes.get("layer_types"),
            Some(&"sliding_attention,full_attention".to_string())
        );
    }

    #[test]
    fn build_program_gemma4_with_softcapping() {
        let mut config = tiny_gemma_config();
        config
            .extra
            .insert("final_logit_softcapping".into(), serde_json::json!(30.0));
        let provider = StubProvider::new(config).with_llama_weights();
        let result = build_program(&provider).expect("softcapping build should succeed");
        let main = result.program.main().unwrap();
        let has_tanh = main.body.operations.iter().any(|op| op.op_type == "tanh");
        assert!(has_tanh, "softcapping should emit tanh op");
    }

    #[test]
    fn build_program_gemma4_model_type_accepted() {
        use std::str::FromStr;
        assert!(Architecture::from_str("gemma4").is_ok());
        assert!(Architecture::from_str("gemma4_text").is_ok());
    }

    #[test]
    fn build_program_gemma4_31b_dense() {
        // 31B: no PLE, no K=V, no KV shared layers — simplest Gemma 4 variant
        let mut config = tiny_gemma_config();
        config.hidden_size = 128;
        config.intermediate_size = 512;
        config.num_attention_heads = 8;
        config.num_key_value_heads = 4;
        config.head_dim = 16;
        config.num_hidden_layers = 4;
        // Per-layer attention types (every 5th is full_attention)
        config.extra.insert(
            "layer_types".into(),
            serde_json::json!([
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention"
            ]),
        );
        config.extra.insert(
            "rope_parameters".into(),
            serde_json::json!({
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}
            }),
        );
        config
            .extra
            .insert("sliding_window".into(), serde_json::json!(4096));
        // 31B does NOT use PLE, K=V, KV sharing, or double-wide MLP
        config
            .extra
            .insert("hidden_size_per_layer_input".into(), serde_json::json!(0));
        config
            .extra
            .insert("attention_k_eq_v".into(), serde_json::json!(false));
        config
            .extra
            .insert("num_kv_shared_layers".into(), serde_json::json!(0));
        config
            .extra
            .insert("use_double_wide_mlp".into(), serde_json::json!(false));

        let provider = StubProvider::new(config).with_llama_weights();
        let result = build_program(&provider).expect("31B build should succeed");
        assert_eq!(result.program.functions.len(), 1);

        // Should have per-layer attention types
        assert!(result.program.attributes.get("layer_types").is_some());
        // Should NOT have PLE-related ops
        let main = result.program.main().unwrap();
        let ple_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.name.contains("ple_"))
            .collect();
        assert!(ple_ops.is_empty(), "31B should not emit PLE ops");
    }

    #[test]
    fn build_program_gemma4_e4b() {
        // E4B: same architecture as E2B — PLE, KV sharing, double-wide MLP
        let mut config = tiny_gemma_config();
        config.hidden_size = 128;
        config.intermediate_size = 256;
        config.num_attention_heads = 8;
        config.num_key_value_heads = 1; // MQA
        config.head_dim = 16;
        config.num_hidden_layers = 6;
        // Per-layer attention types
        config.extra.insert(
            "layer_types".into(),
            serde_json::json!([
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention"
            ]),
        );
        config.extra.insert(
            "rope_parameters".into(),
            serde_json::json!({
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}
            }),
        );
        config
            .extra
            .insert("sliding_window".into(), serde_json::json!(512));
        // E4B uses PLE
        config
            .extra
            .insert("hidden_size_per_layer_input".into(), serde_json::json!(32));
        config
            .extra
            .insert("vocab_size_per_layer_input".into(), serde_json::json!(256));
        // E4B uses KV sharing (last 4 of 6 layers)
        config
            .extra
            .insert("num_kv_shared_layers".into(), serde_json::json!(4));
        // E4B uses double-wide MLP on shared layers
        config
            .extra
            .insert("use_double_wide_mlp".into(), serde_json::json!(true));
        config
            .extra
            .insert("attention_k_eq_v".into(), serde_json::json!(false));
        config
            .extra
            .insert("final_logit_softcapping".into(), serde_json::json!(30.0));

        // Build weights with PLE model-level tensors
        let mut provider = StubProvider::new(config).with_llama_weights();
        // Add model-level PLE weights
        provider = provider
            .with_tensor(
                "model.embed_tokens_per_layer.weight",
                &[256, 6 * 32],
                ScalarType::Float16,
            )
            .with_tensor(
                "model.per_layer_model_projection.weight",
                &[6 * 32, 128],
                ScalarType::Float16,
            )
            .with_tensor(
                "model.per_layer_projection_norm.weight",
                &[32],
                ScalarType::Float16,
            );
        // Add per-layer PLE weights
        for l in 0..6 {
            let p = format!("model.layers.{l}");
            provider = provider
                .with_tensor(
                    &format!("{p}.per_layer_input_gate.weight"),
                    &[32, 128],
                    ScalarType::Float16,
                )
                .with_tensor(
                    &format!("{p}.per_layer_projection.weight"),
                    &[128, 32],
                    ScalarType::Float16,
                )
                .with_tensor(
                    &format!("{p}.post_per_layer_input_norm.weight"),
                    &[128],
                    ScalarType::Float16,
                );
        }

        let result = build_program(&provider).expect("E4B build should succeed");
        assert_eq!(result.program.functions.len(), 1);

        let main = result.program.main().unwrap();
        // Should have PLE ops
        let ple_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.name.contains("ple_"))
            .collect();
        assert!(!ple_ops.is_empty(), "E4B should emit PLE ops");

        // Should have softcapping
        let has_tanh = main.body.operations.iter().any(|op| op.op_type == "tanh");
        assert!(has_tanh, "E4B should emit softcapping tanh");
    }

    #[test]
    fn build_program_gemma4_no_layer_types_is_gemma3() {
        // When no layer_types are set, it's a standard Gemma (1/2/3) — backward compat
        let config = tiny_gemma_config();
        let provider = StubProvider::new(config).with_llama_weights();
        let result = build_program(&provider).expect("Gemma 3 build should succeed");
        assert!(
            result.program.attributes.get("layer_types").is_none(),
            "Gemma 3 should not emit layer_types attribute"
        );
    }

    #[test]
    fn build_program_gemma4_global_head_dim_differs() {
        // Test with global_head_dim != head_dim (e.g. global_head_dim=32 vs head_dim=16)
        let mut config = tiny_gemma_config();
        config.num_hidden_layers = 2;
        config.extra.insert(
            "layer_types".into(),
            serde_json::json!(["sliding_attention", "full_attention"]),
        );
        config.extra.insert(
            "rope_parameters".into(),
            serde_json::json!({
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5}
            }),
        );
        config
            .extra
            .insert("global_head_dim".into(), serde_json::json!(32));
        config
            .extra
            .insert("num_global_key_value_heads".into(), serde_json::json!(2));

        // Need custom weights for the global layer with different head_dim
        let mut provider = StubProvider::new(config.clone()).with_llama_weights();
        // Override the global layer's attention weights with correct dimensions
        let p = "model.layers.1";
        provider = provider
            .with_tensor(
                &format!("{p}.self_attn.q_proj.weight"),
                &[config.num_attention_heads * 32, config.hidden_size],
                ScalarType::Float16,
            )
            .with_tensor(
                &format!("{p}.self_attn.k_proj.weight"),
                &[2 * 32, config.hidden_size],
                ScalarType::Float16,
            )
            .with_tensor(
                &format!("{p}.self_attn.v_proj.weight"),
                &[2 * 32, config.hidden_size],
                ScalarType::Float16,
            )
            .with_tensor(
                &format!("{p}.self_attn.o_proj.weight"),
                &[config.hidden_size, config.num_attention_heads * 32],
                ScalarType::Float16,
            );

        let result = build_program(&provider).expect("heterogeneous head_dim build should succeed");
        assert_eq!(result.program.functions.len(), 1);
    }

    #[test]
    #[ignore] // Requires E2B checkpoint download: set GEMMA4_E2B_PATH
    fn gemma4_e2b_compilation() {
        use crate::weights::safetensors::SafeTensorsProvider;
        use std::path::Path;

        let model_path = std::env::var("GEMMA4_E2B_PATH")
            .expect("Set GEMMA4_E2B_PATH to Gemma 4 E2B checkpoint directory");
        let provider = SafeTensorsProvider::load(Path::new(&model_path))
            .expect("E2B checkpoint loading should succeed");
        let result = build_program(&provider).expect("E2B compilation should succeed");
        assert_eq!(result.program.functions.len(), 1);
        let main = result.program.main().unwrap();
        assert!(!main.body.operations.is_empty());
        // Verify no unexpected warnings
        for w in &result.warnings {
            eprintln!("  warning: {w}");
        }
    }

    #[test]
    fn build_program_gemma4_with_ple() {
        let mut config = tiny_gemma_config();
        config
            .extra
            .insert("hidden_size_per_layer_input".into(), serde_json::json!(8));
        config
            .extra
            .insert("vocab_size_per_layer_input".into(), serde_json::json!(256));
        config.num_hidden_layers = 2;
        let provider = StubProvider::new(config)
            .with_llama_weights()
            .with_ple_weights(8, 256);
        let result = build_program(&provider).expect("Gemma 4 PLE build should succeed");
        assert_eq!(result.program.functions.len(), 1);
        // Verify PLE ops were emitted
        let main = result.program.main().unwrap();
        let ple_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.name.contains("ple_"))
            .collect();
        assert!(!ple_ops.is_empty(), "PLE build should emit PLE ops");
    }

    #[test]
    fn build_program_gemma4_with_kv_sharing() {
        let mut config = tiny_gemma_config();
        config
            .extra
            .insert("attention_k_eq_v".into(), serde_json::json!(true));
        config.extra.insert(
            "layer_types".into(),
            serde_json::json!(["sliding_attention", "full_attention"]),
        );
        config.extra.insert(
            "rope_parameters".into(),
            serde_json::json!({
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {"rope_theta": 1000000.0}
            }),
        );
        config.num_hidden_layers = 2;
        let provider = StubProvider::new(config).with_llama_weights();
        let result = build_program(&provider).expect("Gemma 4 K=V build should succeed");
        // K=V sharing: global layer (layer 1) should NOT emit v_proj ops
        let main = result.program.main().unwrap();
        let l1_v_proj_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.name.contains("l1_v_proj"))
            .collect();
        assert!(
            l1_v_proj_ops.is_empty(),
            "K=V sharing should skip V projection on full_attention layers"
        );
        // But sliding layer (layer 0) SHOULD have v_proj
        let l0_v_proj_ops: Vec<_> = main
            .body
            .operations
            .iter()
            .filter(|op| op.name.contains("l0_v_proj"))
            .collect();
        assert!(
            !l0_v_proj_ops.is_empty(),
            "sliding_attention layers should still emit V projection"
        );
    }

    #[test]
    fn build_program_gemma4_with_kv_shared_layers() {
        let mut config = tiny_gemma_config();
        // 4 layers, last 2 are KV-shared
        config
            .extra
            .insert("num_kv_shared_layers".into(), serde_json::json!(2));
        config.extra.insert(
            "layer_types".into(),
            serde_json::json!([
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ]),
        );
        config.extra.insert(
            "rope_parameters".into(),
            serde_json::json!({
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {"rope_theta": 1000000.0}
            }),
        );
        config.num_hidden_layers = 4;
        let provider = StubProvider::new(config).with_llama_weights();
        let result =
            build_program(&provider).expect("Gemma 4 KV shared layers build should succeed");
        assert_eq!(result.program.functions.len(), 1);
    }
}
