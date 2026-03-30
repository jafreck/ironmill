//! Gemma architecture template.
//!
//! Builds a MIL IR [`Program`] for the Gemma model family (Gemma, Gemma 2,
//! Gemma 3) from weight tensors provided by a [`WeightProvider`].
//!
//! Key differences from LLaMA:
//! - GELU activation in MLP instead of SiLU
//! - Embedding output is normalized by multiplying by sqrt(hidden_size)
//! - May use sliding window attention on alternating layers
//! - Uses RMSNorm (same as LLaMA/Qwen)

use crate::weights::WeightProvider;
use mil_rs::MilError;
use mil_rs::convert::onnx_graph::ConversionResult;
use mil_rs::ir::{Block, Function, Operation, Program, ScalarType, TensorType, Value};

use super::shared::{
    emit_attention, emit_embedding, emit_lm_head, emit_mlp_gelu, emit_residual_add, emit_rms_norm,
    emit_rope_tables,
};

/// Build a complete MIL [`Program`] for a Gemma-family model.
pub fn build_program(provider: &dyn WeightProvider) -> Result<ConversionResult, MilError> {
    let config = provider.config().clone();
    let mut warnings: Vec<String> = Vec::new();

    let seq_len: Option<usize> = None; // dynamic
    let batch: Option<usize> = Some(1);

    // Check for sliding window configuration (alternating layers).
    let sliding_window: Option<usize> = config
        .extra
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    if sliding_window.is_some() {
        warnings.push(
            "sliding_window attention on alternating layers is noted but mask handling is caller-provided"
                .into(),
        );
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

    // Gemma-specific: normalize embeddings by multiplying by sqrt(hidden_size).
    let embed_normed = emit_embedding_norm(block, &config, &embed_out);

    // Precompute RoPE cos/sin tables.
    let (rope_cos, rope_sin) = emit_rope_tables(block, &config);

    // Transformer layers.
    let mut hidden = embed_normed;
    for layer_idx in 0..config.num_hidden_layers {
        hidden = emit_gemma_transformer_layer(
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

    if let Some(sw) = sliding_window {
        program.set_attribute("sliding_window", sw.to_string());
    }

    Ok(ConversionResult { program, warnings })
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

/// Emit a Gemma transformer layer.
/// Same structure as LLaMA but uses GELU activation in MLP.
#[allow(clippy::too_many_arguments)]
fn emit_gemma_transformer_layer(
    block: &mut Block,
    provider: &dyn WeightProvider,
    config: &crate::weights::ModelConfig,
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

    // 2. Self-attention (standard LLaMA-style, no bias)
    let attn_out = emit_attention(
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

    // 5. MLP with GELU activation (Gemma-specific)
    let mlp_out = emit_mlp_gelu(block, provider, config, layer_idx, &normed_mlp, warnings)?;

    // 6. Residual add
    let layer_out = emit_residual_add(block, &post_attn, &mlp_out, &format!("l{layer_idx}_output"));

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
        assert!(
            result.warnings.iter().any(|w| w.contains("sliding_window")),
            "should have sliding_window note"
        );
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
    fn build_program_warns_on_missing_weights() {
        let config = tiny_gemma_config();
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
}
