//! Qwen architecture template.
//!
//! Builds a MIL IR [`Program`] for the Qwen model family (Qwen, Qwen2, Qwen3)
//! from weight tensors provided by a [`WeightProvider`].
//!
//! Key differences from LLaMA:
//! - Attention Q/K/V projections have bias terms
//! - Uses SwiGLU activation (same gate+up+silu+down pattern as LLaMA)
//! - May use sliding window attention (configured via `extra["sliding_window"]`)

use crate::MilError;
use crate::convert::onnx_graph::ConversionResult;
use crate::convert::weights::WeightProvider;
use crate::ir::{Function, Program, ScalarType, TensorType};

use super::shared::{
    emit_attention_core, emit_embedding, emit_linear, emit_lm_head, emit_mlp_silu,
    emit_residual_add, emit_rms_norm, emit_rope_tables,
};

/// Build a complete MIL [`Program`] for a Qwen-family model.
pub fn build_program(provider: &dyn WeightProvider) -> Result<ConversionResult, MilError> {
    let config = provider.config().clone();
    let mut warnings: Vec<String> = Vec::new();

    let seq_len: Option<usize> = None; // dynamic
    let batch: Option<usize> = Some(1);

    // Check for sliding window configuration.
    let sliding_window: Option<usize> = config
        .extra
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    if sliding_window.is_some() {
        warnings
            .push("sliding_window attention is noted but mask handling is caller-provided".into());
    }

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

    // Precompute RoPE cos/sin tables.
    let (rope_cos, rope_sin) = emit_rope_tables(block, &config);

    // Transformer layers.
    let mut hidden = embed_out;
    for layer_idx in 0..config.num_hidden_layers {
        hidden = emit_qwen_transformer_layer(
            block,
            provider,
            &config,
            layer_idx,
            &hidden,
            &rope_cos,
            &rope_sin,
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

    block.outputs.push(logits);

    let mut program = Program::new("1.0.0");
    program.add_function(func);
    program.set_attribute("autoregressive", "true");

    // Record sliding window in program attributes if configured.
    if let Some(sw) = sliding_window {
        program.set_attribute("sliding_window", sw.to_string());
    }

    Ok(ConversionResult { program, warnings })
}

// ---------------------------------------------------------------------------
// Qwen transformer layer
// ---------------------------------------------------------------------------

/// Emit a Qwen transformer layer. Same structure as LLaMA but uses
/// Qwen-specific attention (with bias on Q/K/V projections).
#[allow(clippy::too_many_arguments)]
fn emit_qwen_transformer_layer(
    block: &mut crate::ir::Block,
    provider: &dyn WeightProvider,
    config: &crate::convert::weights::ModelConfig,
    layer_idx: usize,
    input: &str,
    rope_cos: &str,
    rope_sin: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{layer_idx}");

    // 1. Input RMSNorm
    let normed_attn = emit_rms_norm(
        block,
        provider,
        config,
        &format!("{prefix}.input_layernorm"),
        input,
        &format!("l{layer_idx}_input_norm"),
        warnings,
    )?;

    // 2. Qwen attention (with bias)
    let attn_out = emit_qwen_attention(
        block,
        provider,
        config,
        layer_idx,
        &normed_attn,
        rope_cos,
        rope_sin,
        warnings,
    )?;

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
        provider,
        config,
        &format!("{prefix}.post_attention_layernorm"),
        &post_attn,
        &format!("l{layer_idx}_post_attn_norm"),
        warnings,
    )?;

    // 5. MLP (SwiGLU, same as LLaMA)
    let mlp_out = emit_mlp_silu(block, provider, config, layer_idx, &normed_mlp, warnings)?;

    // 6. Residual add
    let layer_out = emit_residual_add(block, &post_attn, &mlp_out, &format!("l{layer_idx}_output"));

    Ok(layer_out)
}

// ---------------------------------------------------------------------------
// Qwen attention (with bias on Q/K/V projections)
// ---------------------------------------------------------------------------

/// Emit Qwen attention. Like LLaMA attention but Q/K/V projections include
/// bias terms when present in the weight provider.
#[allow(clippy::too_many_arguments)]
fn emit_qwen_attention(
    block: &mut crate::ir::Block,
    provider: &dyn WeightProvider,
    config: &crate::convert::weights::ModelConfig,
    layer_idx: usize,
    input: &str,
    rope_cos: &str,
    rope_sin: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{layer_idx}.self_attn");

    // Q/K/V projections — emit_linear automatically picks up bias when present.
    let q = emit_linear(
        block,
        provider,
        &format!("{prefix}.q_proj"),
        input,
        &format!("l{layer_idx}_q_proj"),
        warnings,
    )?;
    let k = emit_linear(
        block,
        provider,
        &format!("{prefix}.k_proj"),
        input,
        &format!("l{layer_idx}_k_proj"),
        warnings,
    )?;
    let v = emit_linear(
        block,
        provider,
        &format!("{prefix}.v_proj"),
        input,
        &format!("l{layer_idx}_v_proj"),
        warnings,
    )?;

    // Delegate to shared attention core (reshape, RoPE, GQA, dot-product, o_proj)
    emit_attention_core(
        block, config, layer_idx, &prefix, &q, &k, &v, rope_cos, rope_sin, provider, warnings,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::templates::shared::{StubProvider, tiny_llama_config};
    use crate::convert::weights::Architecture;

    fn tiny_qwen_config() -> crate::convert::weights::ModelConfig {
        let mut config = tiny_llama_config();
        config.architecture = Architecture::Qwen;
        config
    }

    #[test]
    fn build_program_succeeds_with_all_weights() {
        let config = tiny_qwen_config();
        let provider = StubProvider::new(config)
            .with_llama_weights()
            .with_attention_biases();

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
    fn build_program_emits_bias_on_projections() {
        let config = tiny_qwen_config();
        let provider = StubProvider::new(config)
            .with_llama_weights()
            .with_attention_biases();

        let result = build_program(&provider).expect("build_program should succeed");
        let main = result.program.main().unwrap();

        // Q projection should have a bias input
        let q_proj_op = main
            .body
            .operations
            .iter()
            .find(|op| op.name == "l0_q_proj_op");
        assert!(q_proj_op.is_some(), "should have l0_q_proj_op");
        let q_op = q_proj_op.unwrap();
        assert!(
            q_op.inputs.contains_key("bias"),
            "Q projection should have bias input"
        );
    }

    #[test]
    fn build_program_without_bias_still_works() {
        let config = tiny_qwen_config();
        // No attention biases — should still work (linear just won't have bias input)
        let provider = StubProvider::new(config).with_llama_weights();

        let result = build_program(&provider).expect("build_program should succeed");
        let main = result.program.main().unwrap();

        let q_proj_op = main
            .body
            .operations
            .iter()
            .find(|op| op.name == "l0_q_proj_op");
        assert!(q_proj_op.is_some());
        let q_op = q_proj_op.unwrap();
        assert!(
            !q_op.inputs.contains_key("bias"),
            "Q projection should not have bias when none provided"
        );
    }

    #[test]
    fn build_program_with_sliding_window() {
        let mut config = tiny_qwen_config();
        config
            .extra
            .insert("sliding_window".into(), serde_json::json!(4096));
        let provider = StubProvider::new(config)
            .with_llama_weights()
            .with_attention_biases();

        let result = build_program(&provider).expect("build_program should succeed");
        // sliding_window should be noted in warnings
        assert!(
            result.warnings.iter().any(|w| w.contains("sliding_window")),
            "should have sliding_window note"
        );
        // sliding_window attribute should be set on program
        assert_eq!(
            result.program.attributes.get("sliding_window"),
            Some(&"4096".to_string())
        );
    }

    #[test]
    fn program_is_marked_autoregressive() {
        let config = tiny_qwen_config();
        let provider = StubProvider::new(config)
            .with_llama_weights()
            .with_attention_biases();
        let result = build_program(&provider).unwrap();
        assert!(result.program.is_autoregressive());
    }

    #[test]
    fn build_program_warns_on_missing_weights() {
        let config = tiny_qwen_config();
        let provider = StubProvider::new(config);

        let result =
            build_program(&provider).expect("build_program should still succeed structurally");
        assert!(
            !result.warnings.is_empty(),
            "should have warnings for missing weights"
        );
        let main = result.program.main().unwrap();
        assert!(!main.body.operations.is_empty());
    }

    #[test]
    fn build_program_with_gqa() {
        let mut config = tiny_qwen_config();
        config.num_key_value_heads = 2;
        let provider = StubProvider::new(config)
            .with_llama_weights()
            .with_attention_biases();

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
}
