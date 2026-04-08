//! Mixture-of-Experts (MoE) template helpers.
//!
//! Used by the Gemma 4 MoE variant (26B). Each MoE layer runs a router
//! to select top-k experts, evaluates all expert MLPs (dense evaluation),
//! and combines outputs with router-weighted summation.

use crate::weights::WeightProvider;
use mil_rs::MilError;
use mil_rs::ir::{Block, Operation, Value};

use super::shared::{emit_linear, emit_softmax};

/// Emit a complete MoE block for a single layer.
///
/// Architecture:
/// 1. Router: linear [hidden_size → num_experts] → softmax → top-k selection
/// 2. Expert FFN: all num_experts MLPs evaluated (dense approach)
/// 3. Combine: weighted sum of expert outputs using router probabilities
///
/// Dense evaluation strategy: every expert is computed and the outputs are
/// combined using the full router probability vector. This avoids the need
/// for a `topk` op in the MIL IR and simplifies the graph at the cost of
/// evaluating all experts (acceptable for CoreML static graphs).
///
/// Returns the MoE output tensor name.
pub(super) fn emit_moe_block(
    block: &mut Block,
    provider: &dyn WeightProvider,
    layer_idx: usize,
    input: &str,
    num_experts: usize,
    _top_k: usize,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let prefix = format!("model.layers.{layer_idx}.mlp");

    // 1. Router: linear projection → softmax
    let router_logits = emit_linear(
        block,
        provider,
        &format!("{prefix}.router"),
        input,
        &format!("l{layer_idx}_moe_router"),
        warnings,
    )?;

    let router_probs = emit_softmax(block, &router_logits, &format!("l{layer_idx}_moe_softmax"));

    // 2. Expert FFN evaluation (dense — all experts)
    let mut expert_outputs = Vec::with_capacity(num_experts);
    for expert_idx in 0..num_experts {
        let expert_prefix = format!("{prefix}.experts.{expert_idx}");
        let expert_out = emit_expert_ffn(
            block,
            provider,
            layer_idx,
            expert_idx,
            &expert_prefix,
            input,
            warnings,
        )?;
        expert_outputs.push(expert_out);
    }

    // 3. Weighted combine via iterative slice + mul + add.
    //    For each expert j:
    //      prob_j  = router_probs[:, :, j:j+1]   (slice keeps dim for broadcast)
    //      w_j     = expert_output_j * prob_j
    //      accum  += w_j
    let mut accumulated: Option<String> = None;
    for (expert_idx, expert_out) in expert_outputs.iter().enumerate() {
        // Slice router prob for this expert: probs[:, :, expert_idx:expert_idx+1]
        let prob_slice = {
            let out_name = format!("l{layer_idx}_moe_prob_{expert_idx}");
            let op = Operation::new(
                "slice_by_index",
                format!("l{layer_idx}_moe_prob_slice_{expert_idx}_op"),
            )
            .with_input("x", Value::Reference(router_probs.clone()))
            .with_attr(
                "begin",
                Value::List(vec![
                    Value::Int(0),
                    Value::Int(0),
                    Value::Int(expert_idx as i64),
                ]),
            )
            .with_attr(
                "end",
                Value::List(vec![
                    Value::Int(-1),
                    Value::Int(-1),
                    Value::Int(expert_idx as i64 + 1),
                ]),
            )
            .with_output(&out_name);
            block.add_op(op);
            out_name
        };

        // Multiply expert output by its router weight
        let weighted = {
            let out_name = format!("l{layer_idx}_moe_weighted_{expert_idx}");
            let op = Operation::new("mul", format!("l{layer_idx}_moe_mul_{expert_idx}_op"))
                .with_input("x", Value::Reference(expert_out.clone()))
                .with_input("y", Value::Reference(prob_slice))
                .with_output(&out_name);
            block.add_op(op);
            out_name
        };

        accumulated = Some(match accumulated {
            None => weighted,
            Some(prev) => {
                let out_name = format!("l{layer_idx}_moe_accum_{expert_idx}");
                let op = Operation::new("add", format!("l{layer_idx}_moe_add_{expert_idx}_op"))
                    .with_input("x", Value::Reference(prev))
                    .with_input("y", Value::Reference(weighted))
                    .with_output(&out_name);
                block.add_op(op);
                out_name
            }
        });
    }

    let moe_out = accumulated.ok_or_else(|| {
        MilError::Validation(format!(
            "MoE block at layer {layer_idx}: num_experts must be > 0"
        ))
    })?;
    Ok(moe_out)
}

/// Emit a single expert FFN (gate + up + gelu + down).
fn emit_expert_ffn(
    block: &mut Block,
    provider: &dyn WeightProvider,
    layer_idx: usize,
    expert_idx: usize,
    expert_prefix: &str,
    input: &str,
    warnings: &mut Vec<String>,
) -> Result<String, MilError> {
    let ep = format!("l{layer_idx}_expert{expert_idx}");

    // Gate projection
    let gate = emit_linear(
        block,
        provider,
        &format!("{expert_prefix}.gate_proj"),
        input,
        &format!("{ep}_gate_proj"),
        warnings,
    )?;

    // Up projection
    let up = emit_linear(
        block,
        provider,
        &format!("{expert_prefix}.up_proj"),
        input,
        &format!("{ep}_up_proj"),
        warnings,
    )?;

    // GELU activation on gate
    let gate_act = {
        let out_name = format!("{ep}_gate_gelu");
        let op = Operation::new("gelu", format!("{ep}_gelu_op"))
            .with_input("x", Value::Reference(gate))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Element-wise multiply: gate_act * up
    let mlp_hidden = {
        let out_name = format!("{ep}_mlp_hidden");
        let op = Operation::new("mul", format!("{ep}_mlp_mul_op"))
            .with_input("x", Value::Reference(gate_act))
            .with_input("y", Value::Reference(up))
            .with_output(&out_name);
        block.add_op(op);
        out_name
    };

    // Down projection
    let down = emit_linear(
        block,
        provider,
        &format!("{expert_prefix}.down_proj"),
        &mlp_hidden,
        &format!("{ep}_down_proj"),
        warnings,
    )?;

    Ok(down)
}
