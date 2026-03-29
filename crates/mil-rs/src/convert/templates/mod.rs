//! Architecture templates that construct MIL IR programs from weight tensors.
//!
//! Each template knows how to build a complete MIL [`Program`] for a specific
//! model architecture (LLaMA, Qwen, Gemma, etc.) given a [`WeightProvider`]
//! that supplies the model configuration and weight tensors.

pub mod config;
pub mod gemma;
pub mod llama;
pub mod qwen;
pub(crate) mod shared;

use crate::MilError;
use crate::convert::onnx_graph::ConversionResult;
use crate::convert::weights::{Architecture, WeightProvider};

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
        Architecture::Qwen => qwen::build_program(provider),
        Architecture::Gemma => gemma::build_program(provider),
    }
}

/// Build a MIL IR [`Program`] for a specific component of the model.
///
/// When `component` is `None`, the full model is built. When specified,
/// only the requested component is emitted:
/// - `"embeddings"` — token embedding layer only
/// - `"transformer"` — transformer blocks (layers 0..N)
/// - `"lm_head"` — final norm + language model head
pub fn weights_to_program_component(
    provider: &dyn WeightProvider,
    component: Option<&str>,
) -> Result<ConversionResult, MilError> {
    match component {
        None => weights_to_program(provider),
        Some(comp) => {
            // Build the full program, then extract the requested component.
            let full = weights_to_program(provider)?;
            extract_component(full, comp, provider.config())
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
    component: &str,
    config: &crate::convert::weights::ModelConfig,
) -> Result<ConversionResult, MilError> {
    use crate::ir::{Function, Program};

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
                "embeddings" => {
                    name.contains("embed_tokens")
                        || name == "embed_gather"
                        || name == "embed_out"
                        || name.starts_with("embed_norm")
                }
                "transformer" => {
                    name.starts_with("l") && !name.starts_with("lm_head")
                        || name.starts_with("rope_")
                }
                "lm_head" => name.starts_with("final_norm") || name.starts_with("lm_head"),
                _ => true,
            }
        })
        .collect();

    let filtered_ops: Vec<_> = ops
        .iter()
        .zip(keep.iter())
        .filter(|(_, k)| **k)
        .map(|(op, _)| op.clone())
        .collect();

    let mut new_func = Function::new("main");
    // Preserve inputs for embedding/transformer, outputs for lm_head/full
    if component == "embeddings" || component == "transformer" {
        for (name, ty) in &func.inputs {
            new_func = new_func.with_input(name, ty.clone());
        }
    }
    if component == "lm_head" {
        // lm_head takes hidden state as input (Float16, [batch, seq, hidden_size])
        use crate::ir::{ScalarType, TensorType};
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

    let mut new_program = Program::new("1.0.0");
    new_program.add_function(new_func);
    if program.is_autoregressive() {
        new_program.set_attribute("autoregressive", "true");
    }
    new_program.set_attribute("component", component);

    Ok(ConversionResult {
        program: new_program,
        warnings: result.warnings,
    })
}
