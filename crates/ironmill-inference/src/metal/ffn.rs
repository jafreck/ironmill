//! FFN and MoE block encoding.

use ironmill_metal_sys::ComputeEncoder;

use crate::engine::InferenceError;

use super::buffers::IntermediateBuffers;
use super::ops;
use super::projection::encode_projection;
use super::weights::{LayerWeights, WeightBuffer};

/// Encode the FFN block: gate + up projections, SiLU activation, and down projection.
pub(crate) fn encode_ffn_block(
    enc: &ComputeEncoder,
    pipelines: &super::ops::MetalPipelines,
    bufs: &IntermediateBuffers,
    lw: &LayerWeights,
    h: usize,
    inter: usize,
    token_count: usize,
    use_gelu: bool,
) -> Result<(), InferenceError> {
    // For decode (token_count == 1), try fused gate+up+activation when both are affine INT4.
    // This eliminates the separate activation dispatch and intermediate buffer writes.
    let used_fused_gate_up_act = if token_count == 1 {
        if let (WeightBuffer::AffineQuantized(aq_gate), WeightBuffer::AffineQuantized(aq_up)) =
            (&lw.gate_proj, &lw.up_proj)
        {
            if aq_gate.bit_width == 4 && aq_up.bit_width == 4 {
                ops::encode_fused_ffn_gate_up_act_int4(
                    enc,
                    pipelines
                        .affine
                        .fused_ffn_gate_up_act_int4
                        .get(aq_gate.group_size)
                        .expect("unsupported group_size for fused_ffn_gate_up_act_int4"),
                    &bufs.norm_out,
                    aq_gate,
                    aq_up,
                    &bufs.ffn_gate, // output: activation result
                    inter as u32,
                    h as u32,
                    use_gelu,
                );
                true
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };

    if !used_fused_gate_up_act {
        // Fallback: separate gate+up projections, then activation.
        let used_batched = if token_count == 1 {
            if let (WeightBuffer::AffineQuantized(aq_gate), WeightBuffer::AffineQuantized(aq_up)) =
                (&lw.gate_proj, &lw.up_proj)
            {
                if aq_gate.bit_width == 4 && aq_up.bit_width == 4 {
                    ops::encode_batched_affine_matvec_int4(
                        enc,
                        pipelines
                            .affine
                            .batched_matvec_int4
                            .get(aq_gate.group_size)
                            .expect("unsupported group_size for batched_matvec_int4"),
                        &bufs.norm_out,
                        aq_gate,
                        &bufs.ffn_gate,
                        aq_up,
                        &bufs.ffn_up,
                        inter as u32,
                        h as u32,
                    );
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if !used_batched {
            encode_projection(
                enc,
                &bufs.norm_out,
                &lw.gate_proj,
                &bufs.ffn_gate,
                pipelines,
                token_count,
                inter,
                h,
            )?;
            encode_projection(
                enc,
                &bufs.norm_out,
                &lw.up_proj,
                &bufs.ffn_up,
                pipelines,
                token_count,
                inter,
                h,
            )?;
        }

        enc.memory_barrier_with_resources(&[&bufs.ffn_gate, &bufs.ffn_up]);

        let act_pipeline = if use_gelu {
            &pipelines.activation.ffn_gelu_gate
        } else {
            &pipelines.activation.silu_gate
        };
        ops::encode_silu_gate(
            enc,
            act_pipeline,
            &bufs.ffn_gate,
            &bufs.ffn_up,
            &bufs.ffn_gate,
            (token_count * inter) as u32,
        );
    }

    enc.memory_barrier_with_resources(&[&bufs.ffn_gate]);

    // Down projection
    encode_projection(
        enc,
        &bufs.ffn_gate,
        &lw.down_proj,
        &bufs.ffn_down,
        pipelines,
        token_count,
        h,
        inter,
    )?;

    Ok(())
}

/// Encode MoE (Mixture of Experts) dispatch: router → expert FFNs → weighted combine.
///
/// Dense evaluation: all experts are run and top-k selection + weighted sum is
/// applied afterward. The MoE output is written to `bufs.moe_combined` and then
/// added to `bufs.ffn_down` (the dense MLP output).
pub(crate) fn encode_moe_block(
    enc: &ComputeEncoder,
    pipelines: &super::ops::MetalPipelines,
    bufs: &IntermediateBuffers,
    lw: &LayerWeights,
    h: usize,
    moe_inter: usize,
    token_count: usize,
    num_experts: usize,
    top_k: usize,
) -> Result<(), InferenceError> {
    let router_weight = lw
        .router_weight
        .as_ref()
        .ok_or_else(|| InferenceError::runtime("MoE layer missing router weight".to_string()))?;
    let router_logits = bufs
        .moe_router_logits
        .as_ref()
        .ok_or_else(|| InferenceError::runtime("MoE buffers not allocated".to_string()))?;
    let expert_gate_buf = bufs.moe_expert_gate.as_ref().unwrap();
    let expert_up_buf = bufs.moe_expert_up.as_ref().unwrap();
    let expert_outputs = bufs.moe_expert_outputs.as_ref().unwrap();
    let moe_combined = bufs.moe_combined.as_ref().unwrap();

    // 1. Router: linear(norm_out → router_logits) [hidden_size → num_experts]
    encode_projection(
        enc,
        &bufs.norm_out,
        router_weight,
        router_logits,
        pipelines,
        token_count,
        num_experts,
        h,
    )?;
    enc.memory_barrier_with_resources(&[router_logits]);

    // 2. Softmax over router logits [token_count, num_experts]
    ops::encode_moe_softmax(
        enc,
        &pipelines.moe.softmax,
        router_logits,
        num_experts as u32,
        token_count as u32,
    );
    enc.memory_barrier_with_resources(&[router_logits]);

    // 3. Dense eval: run all expert FFNs
    let expert_slice_size = token_count * h;
    for e in 0..num_experts {
        let e_gate = &lw.expert_gate_projs[e];
        let e_up = &lw.expert_up_projs[e];
        let e_down = &lw.expert_down_projs[e];

        // Gate projection: norm_out → expert_gate_buf [hidden → moe_inter]
        encode_projection(
            enc,
            &bufs.norm_out,
            e_gate,
            expert_gate_buf,
            pipelines,
            token_count,
            moe_inter,
            h,
        )?;

        // Up projection: norm_out → expert_up_buf [hidden → moe_inter]
        encode_projection(
            enc,
            &bufs.norm_out,
            e_up,
            expert_up_buf,
            pipelines,
            token_count,
            moe_inter,
            h,
        )?;
        enc.memory_barrier_with_resources(&[expert_gate_buf, expert_up_buf]);

        // GELU activation on gate (in-place)
        ops::encode_moe_gelu(
            enc,
            &pipelines.moe.gelu,
            expert_gate_buf,
            (token_count * moe_inter) as u32,
        );
        enc.memory_barrier_with_resources(&[expert_gate_buf]);

        // Element-wise multiply: gate *= up (in-place on gate)
        ops::encode_moe_mul(
            enc,
            &pipelines.moe.mul,
            expert_gate_buf,
            expert_up_buf,
            (token_count * moe_inter) as u32,
        );
        enc.memory_barrier_with_resources(&[expert_gate_buf]);

        // Down projection: expert_gate_buf → expert_outputs[e] [moe_inter → hidden]
        // We write to the slice expert_outputs[e * token_count * h ..]
        // Since encode_projection writes to the start of the output buffer,
        // we need to use moe_combined as a temp and then copy.
        encode_projection(
            enc,
            expert_gate_buf,
            e_down,
            moe_combined,
            pipelines,
            token_count,
            h,
            moe_inter,
        )?;
        enc.memory_barrier_with_resources(&[moe_combined]);

        // Copy moe_combined → expert_outputs at offset [e * token_count * h]
        enc.set_pipeline(&pipelines.elementwise.copy_buffer);
        enc.set_buffer(moe_combined, 0, 0);
        enc.set_buffer(expert_outputs, e * expert_slice_size * 2, 1);
        let copy_size = expert_slice_size as u32;
        enc.set_bytes(&copy_size.to_le_bytes(), 2);
        let tg_size = 256usize.min(expert_slice_size);
        let tg_count = expert_slice_size.div_ceil(tg_size);
        enc.dispatch_threadgroups((tg_count, 1, 1), (tg_size, 1, 1));
        enc.memory_barrier_with_resources(&[expert_outputs]);
    }

    // 4. Weighted combine: top-k selection + weighted sum → moe_combined
    ops::encode_moe_weighted_combine(
        enc,
        &pipelines.moe.weighted_combine,
        router_logits,
        expert_outputs,
        moe_combined,
        num_experts as u32,
        top_k as u32,
        h as u32,
        token_count as u32,
    );
    enc.memory_barrier_with_resources(&[moe_combined]);

    // 5. Add MoE output to dense MLP output: ffn_down += moe_combined
    ops::encode_residual_add(
        enc,
        &pipelines.elementwise.residual_add,
        &bufs.ffn_down,
        moe_combined,
        &bufs.ffn_down,
        (token_count * h) as u32,
    );
    enc.memory_barrier_with_resources(&[&bufs.ffn_down]);

    Ok(())
}
