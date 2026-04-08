//! Architecture templates that construct MIL IR programs from weight tensors.
//!
//! Each template knows how to build a complete MIL [`Program`] for a specific
//! model architecture (LLaMA, Qwen, Gemma, etc.) given a [`WeightProvider`]
//! that supplies the model configuration and weight tensors.

pub mod config;
pub mod gemma;
pub mod llama;
pub(crate) mod moe;
pub mod qwen;
pub mod qwen35;
pub(crate) mod shared;

use crate::weights::{Architecture, WeightProvider};
use mil_rs::MilError;
use mil_rs::convert::onnx_graph::ConversionResult;
use serde::Deserialize;

/// Model component for targeted compilation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelComponent {
    /// Full model (default).
    Full,
    /// Embedding table only.
    Embeddings,
    /// Transformer layers only.
    Transformer,
    /// Language model head only.
    LmHead,
}

/// Options controlling MIL template generation for a model.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct TemplateOptions {
    /// When true, emit ANE-optimized ops:
    /// - Conv2d (1×1) instead of Linear for projections
    /// - Decomposed RMSNorm via concat + layer_norm + slice
    /// - Static KV-cache state inputs per layer
    /// - Prefill / decode function split
    pub ane: bool,
}

/// Build a MIL IR [`Program`] from a weight provider by dispatching to the
/// appropriate architecture template.
///
/// Uses default options (no ANE lowering). For ANE-optimized output, use
/// [`weights_to_program_with_options`].
pub fn weights_to_program(provider: &dyn WeightProvider) -> Result<ConversionResult, MilError> {
    weights_to_program_with_options(provider, &TemplateOptions::default())
}

/// Build a MIL IR [`Program`] from a weight provider with explicit template options.
pub fn weights_to_program_with_options(
    provider: &dyn WeightProvider,
    options: &TemplateOptions,
) -> Result<ConversionResult, MilError> {
    match provider.config().architecture {
        Architecture::Llama => llama::build_program(provider, options),
        Architecture::Qwen => {
            if options.ane {
                return Err(MilError::Validation(
                    "ANE lowering is not yet supported for Qwen. \
                     Only LLaMA models support ANE compilation."
                        .into(),
                ));
            }
            qwen::build_program(provider)
        }
        Architecture::Gemma => {
            if options.ane {
                return Err(MilError::Validation(
                    "ANE lowering is not yet supported for Gemma. \
                     Only LLaMA models support ANE compilation."
                        .into(),
                ));
            }
            gemma::build_program(provider)
        }
        Architecture::Qwen35 => {
            if options.ane {
                return Err(MilError::Validation(
                    "ANE lowering is not yet supported for Qwen 3.5. \
                     Only LLaMA models support ANE compilation."
                        .into(),
                ));
            }
            qwen35::build_program(provider)
        }
        _ => Err(MilError::Validation(format!(
            "unsupported architecture: {:?}",
            provider.config().architecture
        ))),
    }
}

/// Build a MIL IR [`Program`] for a specific component of the model.
///
/// When `component` is [`ModelComponent::Full`], the full model is built.
/// Otherwise, only the requested component is emitted.
pub fn weights_to_program_component(
    provider: &dyn WeightProvider,
    component: ModelComponent,
) -> Result<ConversionResult, MilError> {
    match component {
        ModelComponent::Full => weights_to_program(provider),
        _ => {
            // Build the full program, then extract the requested component.
            let full = weights_to_program(provider)?;
            extract_component(full, component, provider.config())
        }
    }
}

/// Extract a component subgraph from a full model program.
///
/// This is a lightweight filter approach: we tag ops with component
/// membership and keep only those belonging to the requested component.
/// For v1, we use a naming-convention heuristic to classify ops.
fn extract_component(
    result: ConversionResult,
    component: ModelComponent,
    config: &crate::weights::ModelConfig,
) -> Result<ConversionResult, MilError> {
    use mil_rs::ir::{Function, Program};

    let program = result.program;
    let func = program
        .main()
        .ok_or_else(|| MilError::Validation("no main function in program".into()))?;

    let ops = &func.body.operations;

    let keep: Vec<bool> = ops
        .iter()
        .map(|op| {
            let name = &op.name;
            match component {
                ModelComponent::Embeddings => {
                    name.contains("embed_tokens")
                        || name == "embed_gather"
                        || name == "embed_out"
                        || name.starts_with("embed_norm")
                }
                ModelComponent::Transformer => {
                    (name.starts_with("l")
                        && name.chars().nth(1).is_some_and(|c| c.is_ascii_digit()))
                        && !name.starts_with("lm_head")
                        || name.starts_with("rope_")
                }
                ModelComponent::LmHead => {
                    name.starts_with("final_norm") || name.starts_with("lm_head")
                }
                _ => true,
            }
        })
        .collect();

    // Warn about heuristic decisions that may break dataflow.
    if component == ModelComponent::Transformer {
        let has_embed_out = ops.iter().any(|op| op.name == "embed_out");
        if has_embed_out {
            tracing::warn!(
                "extract_component(Transformer): dropping 'embed_out' op. \
                 The Transformer component will not include the embedding output projection. \
                 This may break dataflow if downstream components depend on it."
            );
        }
    }

    let filtered_ops: Vec<_> = ops
        .iter()
        .zip(keep.iter())
        .filter(|(_, k)| **k)
        .map(|(op, _)| op.clone())
        .collect();

    let mut new_func = Function::new("main");
    // Preserve inputs for embedding/transformer, outputs for lm_head/full
    if matches!(
        component,
        ModelComponent::Embeddings | ModelComponent::Transformer
    ) {
        for (name, ty) in &func.inputs {
            new_func = new_func.with_input(name, ty.clone());
        }
    }
    if component == ModelComponent::LmHead {
        // Note: LmHead fabricates a fixed hidden-state input with shape
        // [1, max_position_embeddings, hidden_size]. This is a heuristic;
        // the actual hidden-state shape depends on the transformer output
        // and may differ for non-standard sequence lengths.
        tracing::warn!(
            "extract_component(LmHead): fabricating hidden-state input with fixed shape \
             [1, {}, {}]. Actual shape may differ at runtime.",
            config.max_position_embeddings,
            config.hidden_size
        );
        use mil_rs::ir::{ScalarType, TensorType};
        let hidden_ty = TensorType::new(
            ScalarType::Float16,
            vec![1, config.max_position_embeddings, config.hidden_size],
        );
        new_func = new_func.with_input("hidden_states", hidden_ty);
    }

    for op in filtered_ops {
        new_func.body.add_op(op);
    }

    // Set output to the last op's output
    if let Some(last_op) = new_func.body.operations.last() {
        if let Some(out) = last_op.outputs.first() {
            new_func.body.outputs.push(out.clone());
        }
    }

    let mut new_program = Program::new("1");
    new_program.add_function(new_func);
    if program.is_autoregressive() {
        new_program.set_attribute("autoregressive", "true");
    }

    let component_str = match component {
        ModelComponent::Embeddings => "embeddings",
        ModelComponent::Transformer => "transformer",
        ModelComponent::LmHead => "lm_head",
        _ => "full",
    };
    new_program.set_attribute("component", component_str);

    Ok(ConversionResult::new(new_program, result.warnings))
}
