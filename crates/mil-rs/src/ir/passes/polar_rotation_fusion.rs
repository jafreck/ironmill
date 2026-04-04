//! Inter-layer rotation fusion pass for PolarQuant.
//!
//! Runs after PolarQuant and optimizes the rotation/un-rotation overhead:
//!
//! - **Paired layers**: Consecutive linear ops connected through element-wise
//!   activations (relu, leaky_relu) — the rotation applied to one layer's
//!   output cancels with the next layer's input, so no extra matmul is needed.
//!
//! - **Unpaired layers**: Boundary layers (first/last, adjacent to non-fusible
//!   ops like softmax/layernorm) — emits an explicit inverse rotation matmul
//!   (`constexpr_lut_to_dense → mul(norms) → matmul(W_scaled, R_inv)`).

use half::f16;
use std::collections::{HashMap, HashSet};

use crate::error::Result;
use crate::ir::operation::Operation;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::ScalarType;
use crate::ir::types::{TensorData, Value};

use super::replace_reference;
use super::rotation::rotate_rows_hadamard;
use super::util::build_consumer_map;

/// Linear-family op types whose weights can be PolarQuant-rotated.
const LINEAR_FAMILY: &[&str] = &["matmul", "linear", "conv"];

/// Post-PolarQuant pass that fuses adjacent rotation matrices.
///
/// For paired layers, rotations cancel through activations — no extra
/// ops needed. For unpaired boundary layers, emits an explicit inverse
/// rotation matmul (one-time cost at model load via CoreML constant folding).
pub struct PolarRotationFusionPass {
    /// Activation ops that are safe to fuse through.
    /// Default: relu, leaky_relu (piecewise linear, best approximation).
    safe_activations: Vec<String>,
}

impl PolarRotationFusionPass {
    pub fn new() -> Self {
        Self {
            safe_activations: vec!["relu".to_string(), "leaky_relu".to_string()],
        }
    }
}

impl Default for PolarRotationFusionPass {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for PolarRotationFusionPass {
    fn name(&self) -> &str {
        "polar-rotation-fusion"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for function in program.functions.values_mut() {
            let block = &mut function.body;

            let consumer_map = build_consumer_map(block.operations.as_slice());

            let (polar_ops, linear_indices) =
                collect_polar_and_linear_ops(block.operations.as_slice());

            if polar_ops.is_empty() {
                continue;
            }

            let linear_to_polar =
                map_linear_to_polar(block.operations.as_slice(), &linear_indices, &polar_ops);

            let paired = find_paired_ops(
                block.operations.as_slice(),
                &consumer_map,
                &linear_indices,
                &linear_to_polar,
                &self.safe_activations,
            );

            // For paired polar ops, remove the polar_quant_seed attribute.
            for &pi in &paired {
                block.operations[pi].attributes.remove("polar_quant_seed");
            }

            let unpaired: Vec<usize> = polar_ops
                .iter()
                .filter(|pi| !paired.contains(pi))
                .copied()
                .collect();

            let mut insertions: Vec<(usize, Vec<Operation>)> = Vec::new();
            let mut replacements: Vec<(String, String)> = Vec::new();
            let mut r_inv_cache: HashMap<(usize, u64), String> = HashMap::new();

            for &pi in &unpaired {
                if let Some((insert_after, new_ops, source_ref, final_output)) =
                    build_unpaired_unrotation(block.operations.as_slice(), pi, &mut r_inv_cache)
                {
                    insertions.push((insert_after, new_ops));
                    replacements.push((source_ref, final_output));
                    block.operations[pi].attributes.remove("polar_quant_seed");
                }
            }

            apply_insertions_and_rewrites(block, insertions, replacements);
        }
        Ok(())
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Collect indices of polar LUT ops and linear-family ops from a block.
fn collect_polar_and_linear_ops(operations: &[Operation]) -> (Vec<usize>, Vec<usize>) {
    let polar_ops: Vec<usize> = operations
        .iter()
        .enumerate()
        .filter(|(_, op)| {
            op.op_type == "constexpr_lut_to_dense" && op.attributes.contains_key("polar_quant_seed")
        })
        .map(|(i, _)| i)
        .collect();

    let linear_indices: Vec<usize> = operations
        .iter()
        .enumerate()
        .filter(|(_, op)| LINEAR_FAMILY.contains(&op.op_type.as_str()))
        .map(|(i, _)| i)
        .collect();

    (polar_ops, linear_indices)
}

/// Map each linear op to the polar LUT that produces its weight input.
fn map_linear_to_polar(
    operations: &[Operation],
    linear_indices: &[usize],
    polar_ops: &[usize],
) -> HashMap<usize, usize> {
    let mut linear_to_polar: HashMap<usize, usize> = HashMap::new();
    for &li in linear_indices {
        let linear_op = &operations[li];
        for input_val in linear_op.inputs.values() {
            if let Value::Reference(ref_name) = input_val {
                if let Some(pi) = find_polar_source(operations, ref_name, polar_ops) {
                    linear_to_polar.insert(li, pi);
                    break;
                }
            }
        }
    }
    linear_to_polar
}

/// Find polar ops whose rotations cancel through safe activations (paired layers).
fn find_paired_ops(
    operations: &[Operation],
    consumer_map: &HashMap<String, Vec<(usize, String)>>,
    linear_indices: &[usize],
    linear_to_polar: &HashMap<usize, usize>,
    safe_activations: &[String],
) -> HashSet<usize> {
    let mut paired: HashSet<usize> = HashSet::new();

    let elementwise_ops: Vec<String> = {
        let mut ops = safe_activations.to_vec();
        ops.extend(["add", "mul", "sub"].iter().map(|s| s.to_string()));
        ops
    };

    for &l1_idx in linear_indices {
        if !linear_to_polar.contains_key(&l1_idx) {
            continue;
        }
        let l1_output = match operations[l1_idx].outputs.first() {
            Some(o) => o.clone(),
            None => continue,
        };

        const MAX_PAIRING_DEPTH: usize = 4;
        if let Some(l2_idx) = trace_to_linear(
            operations,
            consumer_map,
            &l1_output,
            &elementwise_ops,
            linear_indices,
            MAX_PAIRING_DEPTH,
        ) {
            if let Some(&p2) = linear_to_polar.get(&l2_idx) {
                let p1 = linear_to_polar[&l1_idx];
                paired.insert(p1);
                paired.insert(p2);
            }
        }
    }

    paired
}

/// Build inverse rotation matmul ops for a single unpaired polar weight.
///
/// Returns `(insert_after_idx, new_ops, source_ref, final_output)` or `None`
/// if the polar op lacks required attributes.
fn build_unpaired_unrotation(
    operations: &[Operation],
    pi: usize,
    r_inv_cache: &mut HashMap<(usize, u64), String>,
) -> Option<(usize, Vec<Operation>, String, String)> {
    let (seed, padded_cols, polar_output) = {
        let op = &operations[pi];
        let seed = match op.attributes.get("polar_quant_seed") {
            Some(Value::Int(s)) => *s as u64,
            _ => return None,
        };
        let padded_cols = get_padded_cols(op);
        let output = op.outputs.first().cloned().unwrap_or_default();
        (seed, padded_cols, output)
    };

    if padded_cols == 0 {
        return None;
    }

    let mul_output = format!("{polar_output}_polar_scaled");
    let mul_idx = operations.iter().position(|op| {
        op.op_type == "mul" && op.outputs.first().map(|s| s.as_str()) == Some(&mul_output)
    });
    let insert_after = mul_idx.unwrap_or(pi);

    let orig_cols = get_original_cols(&operations[pi]);
    let n_padded = padded_cols;
    let n = if orig_cols > 0 { orig_cols } else { n_padded };

    let lut_dtype = match operations[pi].attributes.get("lut") {
        Some(Value::Tensor { dtype, .. }) => *dtype,
        _ => ScalarType::Float32,
    };

    let cache_key = (n_padded, seed);
    let mut new_ops: Vec<Operation> = Vec::new();

    let r_inv_output = if let Some(existing) = r_inv_cache.get(&cache_key) {
        existing.clone()
    } else {
        let mut identity = vec![0.0f32; n_padded * n_padded];
        for i in 0..n_padded {
            identity[i * n_padded + i] = 1.0;
        }
        rotate_rows_hadamard(&mut identity, n_padded, n_padded, seed);

        let r_inv_data: Vec<u8> = match lut_dtype {
            ScalarType::Float16 => identity
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                .collect(),
            _ => identity.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };

        let r_inv_name = format!("{polar_output}_r_inv");
        let r_inv_const = Operation::new("const", format!("{polar_output}_r_inv_const"))
            .with_input(
                "val",
                Value::Tensor {
                    data: TensorData::inline(r_inv_data),
                    shape: vec![n_padded, n_padded],
                    dtype: lut_dtype,
                },
            )
            .with_output(&r_inv_name);
        new_ops.push(r_inv_const);
        r_inv_cache.insert(cache_key, r_inv_name.clone());
        r_inv_name
    };

    let source_ref = if mul_idx.is_some() {
        mul_output.clone()
    } else {
        polar_output.clone()
    };

    let final_output = if n < n_padded {
        let matmul_padded_output = format!("{polar_output}_unrotated_padded");
        let matmul_op = Operation::new("matmul", format!("{polar_output}_unrotate_matmul"))
            .with_input("x", Value::Reference(source_ref.clone()))
            .with_input("y", Value::Reference(r_inv_output))
            .with_attr("transpose_x", Value::Bool(false))
            .with_attr("transpose_y", Value::Bool(false))
            .with_output(&matmul_padded_output);
        new_ops.push(matmul_op);

        let slice_output = format!("{polar_output}_unrotated");
        let end_data: Vec<u8> = [-1i32, n as i32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let slice_op = Operation::new("slice_by_index", format!("{polar_output}_unrotate_slice"))
            .with_input("x", Value::Reference(matmul_padded_output))
            .with_attr(
                "end",
                Value::Tensor {
                    data: TensorData::inline(end_data),
                    shape: vec![2],
                    dtype: ScalarType::Int32,
                },
            )
            .with_attr(
                "end_mask",
                Value::Tensor {
                    data: TensorData::inline(vec![1u8, 0]),
                    shape: vec![2],
                    dtype: ScalarType::Bool,
                },
            )
            .with_output(&slice_output);
        new_ops.push(slice_op);
        slice_output
    } else {
        let matmul_output = format!("{polar_output}_unrotated");
        let matmul_op = Operation::new("matmul", format!("{polar_output}_unrotate_matmul"))
            .with_input("x", Value::Reference(source_ref.clone()))
            .with_input("y", Value::Reference(r_inv_output))
            .with_attr("transpose_x", Value::Bool(false))
            .with_attr("transpose_y", Value::Bool(false))
            .with_output(&matmul_output);
        new_ops.push(matmul_op);
        matmul_output
    };

    Some((insert_after, new_ops, source_ref, final_output))
}

/// Insert synthesized ops into the block and rewire downstream references.
fn apply_insertions_and_rewrites(
    block: &mut crate::ir::program::Block,
    insertions: Vec<(usize, Vec<Operation>)>,
    replacements: Vec<(String, String)>,
) {
    let mut offset = 0usize;
    for (idx, new_ops) in &insertions {
        let insert_at = idx + 1 + offset;
        for (j, op) in new_ops.iter().enumerate() {
            block.operations.insert(insert_at + j, op.clone());
        }
        offset += new_ops.len();
    }

    for (old_name, new_name) in &replacements {
        replace_reference(block, old_name, new_name);
        // Fix the matmul/slice op chain's own `x` input back to the original
        // source. replace_reference rewrote ALL references including the
        // matmul's own x input.
        for op in &mut block.operations {
            if op.outputs.iter().any(|o| o == new_name) || op.name.contains("unrotate_matmul") {
                if let Some(x_val) = op.inputs.get_mut("x") {
                    if matches!(x_val, Value::Reference(r) if r == new_name) {
                        *x_val = Value::Reference(old_name.clone());
                    }
                }
            }
        }
    }
}

/// Trace forward from `output_name` through a chain of element-wise ops to
/// find the next linear-family op that consumes the result. Returns the
/// linear op's index if found within `max_depth` hops.
fn trace_to_linear(
    ops: &[Operation],
    consumer_map: &HashMap<String, Vec<(usize, String)>>,
    output_name: &str,
    elementwise_ops: &[String],
    linear_indices: &[usize],
    max_depth: usize,
) -> Option<usize> {
    trace_to_linear_recursive(
        ops,
        consumer_map,
        output_name,
        elementwise_ops,
        linear_indices,
        0,
        max_depth,
    )
}

fn trace_to_linear_recursive(
    ops: &[Operation],
    consumer_map: &HashMap<String, Vec<(usize, String)>>,
    output_name: &str,
    elementwise_ops: &[String],
    linear_indices: &[usize],
    depth: usize,
    max_depth: usize,
) -> Option<usize> {
    if depth >= max_depth {
        return None;
    }

    let consumers = consumer_map.get(output_name)?;

    for &(consumer_idx, _) in consumers {
        let consumer_op = &ops[consumer_idx];

        // Found a linear op — return it.
        if linear_indices.contains(&consumer_idx) {
            return Some(consumer_idx);
        }

        // If element-wise, trace through it.
        if elementwise_ops.contains(&consumer_op.op_type) {
            let next_output = consumer_op.outputs.first()?;
            if let Some(found) = trace_to_linear_recursive(
                ops,
                consumer_map,
                next_output,
                elementwise_ops,
                linear_indices,
                depth + 1,
                max_depth,
            ) {
                return Some(found);
            }
        }
    }

    None
}

/// Trace a reference back to see if it originates from a polar op.
/// Handles the chain: constexpr_lut_to_dense(output) -> const(norms) -> mul(scaled)
/// Also traces recursively through `mul` ops (e.g. norms scaling) and
/// `const(norms)` nodes that appear in the pairing path.
fn find_polar_source(ops: &[Operation], ref_name: &str, polar_indices: &[usize]) -> Option<usize> {
    find_polar_source_recursive(ops, ref_name, polar_indices, 0)
}

/// Recursive helper with depth limit to prevent infinite loops.
fn find_polar_source_recursive(
    ops: &[Operation],
    ref_name: &str,
    polar_indices: &[usize],
    depth: usize,
) -> Option<usize> {
    const MAX_DEPTH: usize = 8;
    if depth > MAX_DEPTH {
        return None;
    }

    // Direct match: ref_name is a polar op's output.
    for &pi in polar_indices {
        if ops[pi].outputs.first().map(|s| s.as_str()) == Some(ref_name) {
            return Some(pi);
        }
    }

    // Find the op that produces ref_name.
    let producer = ops
        .iter()
        .find(|op| op.outputs.first().map(|s| s.as_str()) == Some(ref_name))?;

    // Trace through mul ops (norms scaling) and const ops (norms values).
    if producer.op_type == "mul" || producer.op_type == "const" {
        for val in producer.inputs.values() {
            if let Value::Reference(inner_ref) = val {
                if let Some(pi) =
                    find_polar_source_recursive(ops, inner_ref, polar_indices, depth + 1)
                {
                    return Some(pi);
                }
            }
        }
    }

    None
}

/// Extract the original (unpadded) last dimension from the shape attribute.
fn get_original_cols(op: &Operation) -> usize {
    match op.attributes.get("shape") {
        Some(Value::Tensor {
            data,
            shape,
            dtype: ScalarType::UInt32,
        }) => {
            if shape.is_empty() {
                return 0;
            }
            let n_dims = shape[0];
            if data.byte_len() < n_dims * 4 {
                return 0;
            }
            let bytes = data.as_bytes().expect("tensor not materialized");
            let last_offset = (n_dims - 1) * 4;
            u32::from_le_bytes([
                bytes[last_offset],
                bytes[last_offset + 1],
                bytes[last_offset + 2],
                bytes[last_offset + 3],
            ]) as usize
        }
        _ => 0,
    }
}

/// Extract padded_cols from a constexpr_lut_to_dense op's shape attribute.
/// The shape attribute stores the original tensor shape; padded_cols is the
/// last dimension rounded up to the next power of two.
fn get_padded_cols(op: &Operation) -> usize {
    let cols = match op.attributes.get("shape") {
        Some(Value::Tensor {
            data,
            shape,
            dtype: ScalarType::UInt32,
        }) => {
            if shape.is_empty() {
                return 0;
            }
            // Last dimension from the UInt32 tensor.
            let n_dims = shape[0];
            if data.byte_len() < n_dims * 4 {
                return 0;
            }
            let bytes = data.as_bytes().expect("tensor not materialized");
            let last_offset = (n_dims - 1) * 4;
            u32::from_le_bytes([
                bytes[last_offset],
                bytes[last_offset + 1],
                bytes[last_offset + 2],
                bytes[last_offset + 3],
            ]) as usize
        }
        _ => return 0,
    };
    cols.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::pass::Pass;
    use crate::ir::program::{Function, Program};
    use crate::ir::tensor::ScalarType;
    use crate::ir::types::TensorData;
    use crate::ir::types::Value;

    /// Helper: create a constexpr_lut_to_dense op with polar_quant_seed.
    fn polar_lut_op(name: &str, output: &str, seed: i64, cols: usize) -> Operation {
        let shape_data: Vec<u8> = [4u32, cols as u32]
            .iter()
            .flat_map(|&d| d.to_le_bytes())
            .collect();
        // Include a minimal FP16 LUT so the fusion pass infers the correct dtype.
        let lut_data: Vec<u8> = (0..16u16)
            .flat_map(|v| half::f16::from_f32(v as f32).to_le_bytes())
            .collect();
        Operation::new("constexpr_lut_to_dense", name)
            .with_output(output)
            .with_attr("polar_quant_seed", Value::Int(seed))
            .with_attr(
                "lut",
                Value::Tensor {
                    data: TensorData::inline(lut_data),
                    shape: vec![16],
                    dtype: ScalarType::Float16,
                },
            )
            .with_attr(
                "shape",
                Value::Tensor {
                    data: TensorData::inline(shape_data),
                    shape: vec![2],
                    dtype: ScalarType::UInt32,
                },
            )
    }

    /// Helper: create a linear-family op (matmul).
    fn matmul_op(name: &str, x_ref: &str, w_ref: &str, output: &str) -> Operation {
        Operation::new("matmul", name)
            .with_input("x", Value::Reference(x_ref.to_string()))
            .with_input("weight", Value::Reference(w_ref.to_string()))
            .with_output(output)
    }

    /// Helper: create an activation op.
    fn activation_op(op_type: &str, name: &str, x_ref: &str, output: &str) -> Operation {
        Operation::new(op_type, name)
            .with_input("x", Value::Reference(x_ref.to_string()))
            .with_output(output)
    }

    /// Two adjacent matmul ops whose weights have polar_quant_seed
    /// connected directly (L1 output feeds L2 input) are paired.
    #[test]
    fn fusion_pairs_consecutive_linears() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        // W1: polar weight -> matmul L1
        func.body.add_op(polar_lut_op("w1_lut", "w1_out", 42, 32));
        func.body
            .add_op(matmul_op("L1", "input", "w1_out", "l1_out"));

        // W2: polar weight -> matmul L2
        func.body.add_op(polar_lut_op("w2_lut", "w2_out", 42, 32));
        func.body
            .add_op(matmul_op("L2", "l1_out", "w2_out", "l2_out"));

        func.body.outputs.push("l2_out".into());
        program.add_function(func);

        PolarRotationFusionPass::new().run(&mut program).unwrap();

        // Both polar ops should have polar_quant_seed removed (paired).
        let ops = &program.functions["main"].body.operations;
        for op in ops {
            if op.op_type == "constexpr_lut_to_dense" {
                assert!(
                    !op.attributes.contains_key("polar_quant_seed"),
                    "paired polar op should have seed removed: {}",
                    op.name
                );
            }
        }

        // No R_inv const or unrotate matmul ops should have been inserted.
        assert!(
            !ops.iter().any(|op| op.name.contains("r_inv")),
            "paired ops should not get R_inv fallback"
        );
    }

    /// Linear → ReLU → linear chain is paired through the activation.
    #[test]
    fn fusion_pairs_through_relu() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        func.body.add_op(polar_lut_op("w1_lut", "w1_out", 42, 16));
        func.body
            .add_op(matmul_op("L1", "input", "w1_out", "l1_out"));
        func.body
            .add_op(activation_op("relu", "relu1", "l1_out", "relu_out"));

        func.body.add_op(polar_lut_op("w2_lut", "w2_out", 42, 16));
        func.body
            .add_op(matmul_op("L2", "relu_out", "w2_out", "l2_out"));

        func.body.outputs.push("l2_out".into());
        program.add_function(func);

        PolarRotationFusionPass::new().run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        for op in ops {
            if op.op_type == "constexpr_lut_to_dense" {
                assert!(
                    !op.attributes.contains_key("polar_quant_seed"),
                    "through-relu paired op should have seed removed: {}",
                    op.name
                );
            }
        }

        assert!(
            !ops.iter().any(|op| op.name.contains("r_inv")),
            "paired through relu should not get R_inv fallback"
        );
    }

    /// Softmax between linears breaks the fusion chain.
    #[test]
    fn fusion_breaks_at_softmax() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        func.body.add_op(polar_lut_op("w1_lut", "w1_out", 42, 8));
        func.body
            .add_op(matmul_op("L1", "input", "w1_out", "l1_out"));
        // Softmax is NOT a safe activation.
        func.body
            .add_op(activation_op("softmax", "sm1", "l1_out", "sm_out"));

        func.body.add_op(polar_lut_op("w2_lut", "w2_out", 42, 8));
        func.body
            .add_op(matmul_op("L2", "sm_out", "w2_out", "l2_out"));

        func.body.outputs.push("l2_out".into());
        program.add_function(func);

        PolarRotationFusionPass::new().run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;

        // Both should be unpaired, so they get fallback R_inv matmul ops.
        // With R_inv deduplication by (padded_cols, seed), both layers share
        // the same R_inv const since they have the same seed and dims.
        let r_inv_ops: Vec<_> = ops.iter().filter(|op| op.name.contains("r_inv")).collect();
        assert!(
            !r_inv_ops.is_empty(),
            "softmax break should produce R_inv fallback(s)"
        );

        // Should have unrotate matmuls for both unpaired layers.
        let unrotate_matmuls: Vec<_> = ops
            .iter()
            .filter(|op| op.name.contains("unrotate_matmul"))
            .collect();
        assert_eq!(
            unrotate_matmuls.len(),
            2,
            "both unpaired layers need unrotate matmuls, got {}",
            unrotate_matmuls.len()
        );

        // All polar_quant_seed attributes should be removed.
        for op in ops {
            assert!(
                !op.attributes.contains_key("polar_quant_seed"),
                "seed should be removed after pass: {}",
                op.name
            );
        }
    }

    /// An unpaired boundary layer gets const(R_inv) + matmul ops inserted.
    #[test]
    fn fusion_unpaired_gets_matmul_fallback() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        // Single polar weight with no partner.
        func.body.add_op(polar_lut_op("w1_lut", "w1_out", 42, 8));
        func.body
            .add_op(matmul_op("L1", "input", "w1_out", "l1_out"));
        func.body.outputs.push("l1_out".into());
        program.add_function(func);

        let ops_before = program.functions["main"].body.operations.len();

        PolarRotationFusionPass::new().run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;

        // Should have new ops: const(R_inv) + matmul(unrotate).
        assert!(
            ops.len() > ops_before,
            "unpaired op should produce extra ops: before={ops_before}, after={}",
            ops.len()
        );

        // Find the R_inv const.
        let r_inv = ops
            .iter()
            .find(|op| op.op_type == "const" && op.name.contains("r_inv"));
        assert!(r_inv.is_some(), "should have R_inv const op");

        let r_inv_op = r_inv.unwrap();
        // R_inv should be an [N, N] Float16 tensor where N = padded_cols.
        match r_inv_op.inputs.get("val") {
            Some(Value::Tensor {
                shape,
                dtype: ScalarType::Float16,
                ..
            }) => {
                assert_eq!(shape.len(), 2, "R_inv should be 2D");
                assert_eq!(shape[0], shape[1], "R_inv should be square");
                assert!(
                    shape[0].is_power_of_two(),
                    "R_inv dimension should be power of two"
                );
            }
            other => panic!("R_inv const should have Float16 tensor, got {other:?}"),
        }

        // Find the unrotate matmul.
        let unrotate = ops
            .iter()
            .find(|op| op.op_type == "matmul" && op.name.contains("unrotate"));
        assert!(unrotate.is_some(), "should have unrotate matmul op");

        // polar_quant_seed should be removed.
        let lut_op = ops.iter().find(|op| op.op_type == "constexpr_lut_to_dense");
        assert!(
            lut_op.is_some() && !lut_op.unwrap().attributes.contains_key("polar_quant_seed"),
            "seed should be removed after fallback"
        );
    }

    /// Pass is a no-op when no PolarQuant ops exist.
    #[test]
    fn fusion_handles_no_polar_ops() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        let plain_const = Operation::new("const", "c1")
            .with_input(
                "val",
                Value::Tensor {
                    data: TensorData::inline(vec![0u8; 16]),
                    shape: vec![4],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output("c1_out");
        func.body.add_op(plain_const);
        func.body
            .add_op(activation_op("relu", "r1", "c1_out", "r1_out"));
        func.body.outputs.push("r1_out".into());
        program.add_function(func);

        let ops_before = program.functions["main"].body.operations.len();

        PolarRotationFusionPass::new().run(&mut program).unwrap();

        let ops_after = program.functions["main"].body.operations.len();
        assert_eq!(
            ops_before, ops_after,
            "no-op pass should not change the graph"
        );
    }
}
