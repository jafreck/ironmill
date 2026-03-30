//! Model → ANE-sized sub-program splitter.
//!
//! A full transformer cannot be compiled as a single ANE program.
//! This module splits the model into per-layer sub-programs that
//! execute sequentially, all targeting ANE.

use std::collections::{BTreeMap, HashMap, HashSet};

use mil_rs::ir::{Function, Operation, Program, TensorType, Value};

use crate::{Result, TensorDescriptor};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The result of splitting a model into ANE-sized sub-programs.
#[derive(Debug)]
pub struct ModelSplit {
    pub programs: Vec<SubProgram>,
}

/// A single sub-program suitable for ANE compilation.
#[derive(Debug)]
pub struct SubProgram {
    /// Name (e.g., "embedding", "layer_0", "layer_1", "lm_head").
    pub name: String,
    /// The MIL IR for this sub-program (a complete, standalone Program).
    pub program: Program,
    /// Inputs consumed by this sub-program.
    pub inputs: Vec<TensorDescriptor>,
    /// Outputs produced by this sub-program.
    pub outputs: Vec<TensorDescriptor>,
}

/// Configuration for the model splitter.
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Maximum weight data size per sub-program (bytes).
    /// Sub-programs exceeding this are further chunked.
    pub max_weight_size: usize,
    /// When `true`, split each `layer_N` sub-program at the attention
    /// boundary into `layer_N_pre_attn` and `layer_N_post_attn`.
    ///
    /// Required for autoregressive inference where the attention + KV
    /// cache is managed externally (both FP16 baseline and TurboQuant).
    pub split_attention: bool,
    /// When `true` (and `split_attention` is also `true`), emit the
    /// attention cluster ops as `layer_N_fp16_attn` sub-programs instead
    /// of discarding them. Used for FP16 baseline inference.
    pub emit_attention: bool,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            max_weight_size: 128 * 1024 * 1024, // 128 MB — increased to avoid weight-limit
            // chunking that creates extra sub-programs exceeding the ANE budget
            split_attention: false,
            emit_attention: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Split a [`Program`] into sub-programs suitable for ANE execution.
///
/// The splitter identifies natural boundaries in the model:
/// - Embedding layer
/// - Transformer layers (each layer becomes a sub-program)
/// - Final layer norm + LM head
///
/// Each sub-program is a standalone [`Program`] that can be independently
/// compiled and executed.
pub fn split_for_ane(program: &Program, config: &SplitConfig) -> Result<ModelSplit> {
    let func = match program.main() {
        Some(f) => f,
        None => {
            // No main function → nothing to split.
            return Ok(ModelSplit {
                programs: Vec::new(),
            });
        }
    };

    if func.body.operations.is_empty() {
        return Ok(ModelSplit {
            programs: Vec::new(),
        });
    }

    // Build a type-map from the function's declared inputs so we can resolve
    // cross-boundary references.
    let mut type_map: HashMap<String, TensorType> = HashMap::new();
    for (name, ty) in &func.inputs {
        type_map.insert(name.clone(), ty.clone());
    }
    // Also record output types produced by every operation.
    for op in &func.body.operations {
        for (out_name, out_ty) in op.outputs.iter().zip(op.output_types.iter()) {
            if let Some(ty) = out_ty {
                type_map.insert(out_name.clone(), ty.clone());
            }
        }
    }

    // Classify operations into groups.
    let groups = classify_ops(&func.body.operations);

    // Build sub-programs from the classified groups.
    let mut sub_programs: Vec<SubProgram> = Vec::new();
    for (name, ops) in &groups {
        if config.split_attention && name.starts_with("layer_") {
            // Split this layer at the attention boundary.
            let (pre, attn, post) = split_at_attention_boundary(ops);
            let pre_name = format!("{name}_pre_attn");
            let post_name = format!("{name}_post_attn");
            if !pre.is_empty() {
                sub_programs.push(build_sub_program(&pre_name, &pre, &type_map, Some(ops)));
            }
            if config.emit_attention && !attn.is_empty() {
                let attn_name = format!("{name}_fp16_attn");
                sub_programs.push(build_sub_program(&attn_name, &attn, &type_map, Some(ops)));
            }
            if !post.is_empty() {
                sub_programs.push(build_sub_program(&post_name, &post, &type_map, Some(ops)));
            }
        } else {
            let sp = build_sub_program(name, ops, &type_map, None);
            sub_programs.push(sp);
        }
    }

    // Enforce weight-size limits: chunk oversized sub-programs.
    sub_programs = enforce_weight_limit(sub_programs, config);

    Ok(ModelSplit {
        programs: sub_programs,
    })
}

// ---------------------------------------------------------------------------
// Layer number extraction
// ---------------------------------------------------------------------------

/// Extract a layer number from an operation name.
///
/// Matches layer/block patterns anywhere in the name (not just at the start)
/// to handle both direct names (`layer_0_attn_q`) and ONNX-converted names
/// with prefixes (`_model_layers_0_attn_...`, `/model/layers.0/...`).
///
/// Recognized patterns: `layer_0_attn_q`, `layers.3.ffn.up`, `block_12_norm`,
/// `_model_layers_0_attn_q`, `layer0_w`, `layer.0.weight`.
fn extract_layer_number(op_name: &str) -> Option<usize> {
    let lower = op_name.to_ascii_lowercase();
    // Search for layer/block patterns anywhere in the name.
    for pattern in ["layers.", "layers_", "layer.", "layer_", "block.", "block_"] {
        if let Some(idx) = lower.find(pattern) {
            let rest = &lower[idx + pattern.len()..];
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !num_str.is_empty() {
                return num_str.parse().ok();
            }
        }
    }
    // Also match "layerN" without separator.
    if let Some(idx) = lower.find("layer") {
        let rest = &lower[idx + "layer".len()..];
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !num_str.is_empty() {
            return num_str.parse().ok();
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Operation classification
// ---------------------------------------------------------------------------

/// Classify operations into named groups.
///
/// Returns a list of `(group_name, ops)` in execution order.
fn classify_ops(operations: &[Operation]) -> Vec<(String, Vec<Operation>)> {
    // First, see if there is any layer-numbering pattern at all.
    let layer_numbers: Vec<Option<usize>> = operations
        .iter()
        .map(|op| extract_layer_number(&op.name))
        .collect();

    let has_layers = layer_numbers.iter().any(|n| n.is_some());

    if !has_layers {
        // Fallback: single group with all ops.
        return vec![("main".to_string(), operations.to_vec())];
    }

    // Gather the ordered set of unique layer numbers (BTreeMap keeps order).
    let unique_layers: BTreeMap<usize, ()> = layer_numbers
        .iter()
        .filter_map(|n| n.map(|v| (v, ())))
        .collect();

    let last_layer = *unique_layers.keys().last().unwrap();

    // Find the index of the last op that belongs to the last layer number.
    let mut groups: Vec<(String, Vec<Operation>)> = Vec::new();
    let mut pre_ops: Vec<Operation> = Vec::new();
    let mut layer_ops: BTreeMap<usize, Vec<Operation>> = BTreeMap::new();
    let mut post_ops: Vec<Operation> = Vec::new();

    let mut seen_last_layer = false;
    let mut seen_any_layer = false;

    for (op, layer_num) in operations.iter().zip(layer_numbers.iter()) {
        match layer_num {
            Some(n) => {
                seen_any_layer = true;
                if *n == last_layer {
                    seen_last_layer = true;
                }
                layer_ops.entry(*n).or_default().push(op.clone());
            }
            None => {
                if !seen_any_layer {
                    pre_ops.push(op.clone());
                } else if seen_last_layer {
                    post_ops.push(op.clone());
                } else {
                    // Non-numbered op between layers — attach to the
                    // preceding layer (or pre if before first layer).
                    let recent = layer_ops.keys().last().copied();
                    if let Some(k) = recent {
                        layer_ops.entry(k).or_default().push(op.clone());
                    } else {
                        pre_ops.push(op.clone());
                    }
                }
            }
        }
    }

    if !pre_ops.is_empty() {
        groups.push(("embedding".to_string(), pre_ops));
    }

    for (n, ops) in &layer_ops {
        groups.push((format!("layer_{n}"), ops.clone()));
    }

    if !post_ops.is_empty() {
        groups.push(("lm_head".to_string(), post_ops));
    }

    groups
}

// ---------------------------------------------------------------------------
// Op dependency graph
// ---------------------------------------------------------------------------

/// Directed dependency graph over a flat `&[Operation]` list.
///
/// Each op index maps to the set of op indices it feeds into (forward)
/// and the set of op indices it consumes from (backward). Edges are
/// derived from `Value::Reference` entries in each op's inputs.
struct OpGraph {
    /// op index → set of op indices that consume this op's outputs.
    forward: Vec<HashSet<usize>>,
    /// op index → set of op indices whose outputs this op consumes.
    backward: Vec<HashSet<usize>>,
}

impl OpGraph {
    /// Build the dependency graph from a flat, topologically-ordered op list.
    fn build(ops: &[Operation]) -> Self {
        let n = ops.len();
        let mut forward = vec![HashSet::new(); n];
        let mut backward = vec![HashSet::new(); n];

        // Map output name → producing op index.
        let mut output_to_idx: HashMap<&str, usize> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            for out in &op.outputs {
                output_to_idx.insert(out.as_str(), i);
            }
        }

        // Wire up edges from Value::Reference inputs.
        for (consumer_idx, op) in ops.iter().enumerate() {
            for val in op.inputs.values() {
                if let Value::Reference(ref_name) = val {
                    if let Some(&producer_idx) = output_to_idx.get(ref_name.as_str()) {
                        forward[producer_idx].insert(consumer_idx);
                        backward[consumer_idx].insert(producer_idx);
                    }
                }
            }
        }

        Self { forward, backward }
    }

    /// Collect all ops reachable by walking backward (transitively) from
    /// `start`, *excluding* `start` itself.
    fn walk_backward(&self, start: usize) -> HashSet<usize> {
        let mut visited = HashSet::new();
        let mut stack: Vec<usize> = self.backward[start].iter().copied().collect();
        while let Some(idx) = stack.pop() {
            if visited.insert(idx) {
                stack.extend(self.backward[idx].iter().copied());
            }
        }
        visited
    }

    /// Collect all ops reachable by walking forward (transitively) from
    /// `start`, *excluding* `start` itself.
    fn walk_forward(&self, start: usize) -> HashSet<usize> {
        let mut visited = HashSet::new();
        let mut stack: Vec<usize> = self.forward[start].iter().copied().collect();
        while let Some(idx) = stack.pop() {
            if visited.insert(idx) {
                stack.extend(self.forward[idx].iter().copied());
            }
        }
        visited
    }
}

// ---------------------------------------------------------------------------
// Structural anchor detection
// ---------------------------------------------------------------------------

/// Returns `true` if the op is a const (weight/bias/scalar literal).
fn is_const_op(op: &Operation) -> bool {
    op.op_type == "const" || op.op_type.starts_with("constexpr_")
}

/// Find indices of Q/K/V projection matmuls.
///
/// Q/K/V projections are matmul/linear ops that:
///   - have at least one const input (the weight), and
///   - share the same non-const activation input (the norm output).
///
/// The group of 2–3 matmuls sharing a non-const input and appearing
/// earliest in topological order are the projections.
fn find_projection_matmuls(ops: &[Operation], graph: &OpGraph) -> Vec<usize> {
    // Map output name → producing op index (for resolving references).
    let mut output_to_idx: HashMap<&str, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for out in &op.outputs {
            output_to_idx.insert(out.as_str(), i);
        }
    }

    // Find matmul/linear ops with at least one const input.
    let matmuls_with_const: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(i, op)| {
            (op.op_type == "matmul" || op.op_type == "linear")
                && graph.backward[*i]
                    .iter()
                    .any(|&pred| is_const_op(&ops[pred]))
        })
        .map(|(i, _)| i)
        .collect();

    // Group by their non-const input reference name. Q/K/V projections
    // share the same activation input (e.g., the norm output).
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for &idx in &matmuls_with_const {
        for val in ops[idx].inputs.values() {
            if let Value::Reference(ref_name) = val {
                if let Some(&pred_idx) = output_to_idx.get(ref_name.as_str()) {
                    if !is_const_op(&ops[pred_idx]) {
                        groups.entry(ref_name.clone()).or_default().push(idx);
                    }
                }
            }
        }
    }

    // Pick the earliest group with 2+ members (Q/K/V projections).
    let mut best: Option<Vec<usize>> = None;
    for (_, mut indices) in groups {
        if indices.len() >= 2 {
            indices.sort();
            if best.is_none() || indices[0] < best.as_ref().unwrap()[0] {
                best = Some(indices);
            }
        }
    }

    best.unwrap_or_default()
}

/// Find the O-projection matmul by walking forward from softmax.
///
/// The O-projection is the first matmul/linear with a const weight
/// reachable downstream from the softmax (via AV matmul → output
/// reshape/transpose → O-proj). It excludes the original Q/K/V
/// projection matmuls.
fn find_o_projection(
    ops: &[Operation],
    graph: &OpGraph,
    softmax_indices: &[usize],
    proj_matmuls: &[usize],
) -> Option<usize> {
    let proj_set: HashSet<usize> = proj_matmuls.iter().copied().collect();

    for &s in softmax_indices {
        let forward = graph.walk_forward(s);
        let mut candidates: Vec<usize> = forward
            .iter()
            .copied()
            .filter(|&idx| {
                !proj_set.contains(&idx)
                    && (ops[idx].op_type == "matmul" || ops[idx].op_type == "linear")
                    && graph.backward[idx]
                        .iter()
                        .any(|&pred| is_const_op(&ops[pred]))
            })
            .collect();
        candidates.sort();
        if let Some(&first) = candidates.first() {
            return Some(first);
        }
    }
    None
}

/// Find softmax ops (the attention core marker).
///
/// Matches standard `softmax` as well as variants produced by different
/// ONNX→MIL decomposition paths (`reduce_softmax`, `_softmax` suffixes,
/// and fused `scaled_dot_product_attention` / `sdpa` ops).
fn find_softmax_ops(ops: &[Operation]) -> Vec<usize> {
    ops.iter()
        .enumerate()
        .filter(|(_, op)| op.op_type == "softmax" || op.op_type.ends_with("_softmax"))
        .map(|(i, _)| i)
        .collect()
}

/// Find fused attention ops (GQA/MHA that were not decomposed into
/// individual matmul/softmax ops). These serve as both the attention
/// core marker AND the attention cluster.
fn find_fused_attention_ops(ops: &[Operation]) -> Vec<usize> {
    ops.iter()
        .enumerate()
        .filter(|(_, op)| {
            op.op_type == "scaled_dot_product_attention"
                || op.op_type == "sdpa"
                || op.op_type == "GroupQueryAttention"
                || op.op_type == "grouped_query_attention"
                || op.op_type == "MultiHeadAttention"
        })
        .map(|(i, _)| i)
        .collect()
}

// ---------------------------------------------------------------------------
// Attention-boundary splitting
// ---------------------------------------------------------------------------

/// Split a layer's ops into pre-attention and post-attention groups.
///
/// Uses **structural graph traversal** rather than name-based heuristics.
/// The split identifies the attention cluster by finding softmax ops and
/// Q/K/V projection matmuls, then classifies:
///
/// - **Pre-attention**: ops that feed *into* the Q/K/V projection inputs
///   (input norm, embedding lookups, weight consts for projections).
/// - **Attention cluster** (stripped): the projections themselves, RoPE,
///   per-head reshapes/norms, QK^T matmul, softmax, AV matmul, and the
///   output reshape/transpose.
/// - **Post-attention**: ops reachable forward from the O-projection
///   output (O projection, residual add, FFN, their weight consts).
///
/// Falls back to the legacy name-based heuristic if structural detection
/// fails.
fn split_at_attention_boundary(
    ops: &[Operation],
) -> (Vec<Operation>, Vec<Operation>, Vec<Operation>) {
    if std::env::var("IRONMILL_SPLIT_DEBUG").is_ok() {
        eprintln!("split_at_attention_boundary: {} ops", ops.len());
        for (i, op) in ops.iter().enumerate() {
            eprintln!("  [{i}] {} (type: {})", op.name, op.op_type);
        }
    }

    if let Some(result) = try_structural_split(ops) {
        return result;
    }

    // Fallback: legacy name-based heuristic.
    eprintln!(
        "warning: structural attention split failed, falling back to name heuristic ({} ops)",
        ops.len()
    );
    split_at_attention_boundary_by_name(ops)
}

/// Attempt the graph-based structural split. Returns `None` if the
/// structural anchors can't be identified (no softmax, no projections).
fn try_structural_split(
    ops: &[Operation],
) -> Option<(Vec<Operation>, Vec<Operation>, Vec<Operation>)> {
    let graph = OpGraph::build(ops);

    // 1. Find softmax ops — the attention core marker.
    let softmax_indices = find_softmax_ops(ops);

    // 2. If no softmax found, check for fused attention ops (GQA/MHA
    //    that weren't decomposed into individual ops).
    if softmax_indices.is_empty() {
        let fused_attn_indices = find_fused_attention_ops(ops);
        if !fused_attn_indices.is_empty() {
            return try_fused_attention_split(ops, &graph, &fused_attn_indices);
        }
        return None;
    }

    // 3. Find Q/K/V projection matmuls (matmul with const weight that
    //    share the same activation input). Expect 2–3.
    let proj_matmuls = find_projection_matmuls(ops, &graph);
    if proj_matmuls.len() < 2 {
        return None;
    }

    // 4. Pre-attn: the projections themselves + everything feeding into them.
    //    TurboQuant (and FP16 attention) need Q/K/V projection outputs
    //    from pre_attn, so the projections must be IN pre_attn, not stripped.
    let mut pre_attn_set: HashSet<usize> = HashSet::new();
    for &proj in &proj_matmuls {
        // Include the projection matmul itself.
        pre_attn_set.insert(proj);
        // Include all ops feeding into it (norm, weight consts, etc.).
        pre_attn_set.extend(graph.walk_backward(proj));
    }

    // 5. Find the O-projection: first matmul/linear with const weight
    //    reachable forward from softmax (excludes Q/K/V projections).
    let o_proj_idx = find_o_projection(ops, &graph, &softmax_indices, &proj_matmuls);

    // 6. Post-attn: O-projection + everything forward from it.
    let mut post_attn_set: HashSet<usize> = HashSet::new();
    if let Some(o_idx) = o_proj_idx {
        post_attn_set.insert(o_idx);
        post_attn_set.extend(graph.walk_forward(o_idx));
    }

    // Add const ops that exclusively feed post-attn ops.
    for (i, op) in ops.iter().enumerate() {
        if !is_const_op(op) || graph.forward[i].is_empty() {
            continue;
        }
        if graph.forward[i].iter().all(|c| post_attn_set.contains(c)) {
            post_attn_set.insert(i);
        }
    }

    // 7. Collect pre-attn, attention cluster, and post-attn ops in
    //    topological order.
    let mut pre_attn = Vec::new();
    let mut attn_cluster = Vec::new();
    let mut post_attn = Vec::new();

    for (i, op) in ops.iter().enumerate() {
        if pre_attn_set.contains(&i) {
            pre_attn.push(op.clone());
        } else if post_attn_set.contains(&i) {
            post_attn.push(op.clone());
        } else {
            attn_cluster.push(op.clone());
        }
    }

    // If both halves are empty, fall back.
    if pre_attn.is_empty() && post_attn.is_empty() {
        return None;
    }

    Some((pre_attn, attn_cluster, post_attn))
}

/// Structural split when the attention block is a single fused op
/// (e.g., `GroupQueryAttention` or `MultiHeadAttention`).
///
/// The fused op itself IS the attention cluster. Pre-attn = everything
/// that feeds into it (walk backward). Post-attn = everything it feeds
/// into (walk forward).
fn try_fused_attention_split(
    ops: &[Operation],
    graph: &OpGraph,
    fused_indices: &[usize],
) -> Option<(Vec<Operation>, Vec<Operation>, Vec<Operation>)> {
    // Use the first fused attention op as the split point.
    let fused_idx = fused_indices[0];

    // Walk backward from the fused op to find all ops that feed into it.
    let mut attn_cluster: HashSet<usize> = HashSet::new();
    attn_cluster.insert(fused_idx);
    // Include any other fused attention ops.
    for &idx in fused_indices {
        attn_cluster.insert(idx);
    }

    // Pre-attn: ops that are ancestors of the fused op's non-const inputs.
    let mut pre_attn_set: HashSet<usize> = HashSet::new();
    for &pred in &graph.backward[fused_idx] {
        if !is_const_op(&ops[pred]) {
            pre_attn_set.insert(pred);
            pre_attn_set.extend(graph.walk_backward(pred));
        }
    }

    // Also include const ops that exclusively feed pre-attn ops.
    for (i, op) in ops.iter().enumerate() {
        if !is_const_op(op) || graph.forward[i].is_empty() {
            continue;
        }
        if graph.forward[i].iter().all(|c| pre_attn_set.contains(c)) {
            pre_attn_set.insert(i);
        }
    }

    // Post-attn: everything forward from the fused op.
    let mut post_attn_set: HashSet<usize> = HashSet::new();
    for &succ in &graph.forward[fused_idx] {
        post_attn_set.insert(succ);
        post_attn_set.extend(graph.walk_forward(succ));
    }

    // Add const ops that exclusively feed post-attn ops.
    for (i, op) in ops.iter().enumerate() {
        if !is_const_op(op) || graph.forward[i].is_empty() {
            continue;
        }
        if !pre_attn_set.contains(&i)
            && !attn_cluster.contains(&i)
            && graph.forward[i].iter().all(|c| post_attn_set.contains(c))
        {
            post_attn_set.insert(i);
        }
    }

    // Collect in topological order.
    let mut pre_attn = Vec::new();
    let mut attn_ops = Vec::new();
    let mut post_attn = Vec::new();

    for (i, op) in ops.iter().enumerate() {
        if pre_attn_set.contains(&i) {
            pre_attn.push(op.clone());
        } else if post_attn_set.contains(&i) {
            post_attn.push(op.clone());
        } else {
            attn_ops.push(op.clone());
        }
    }

    if pre_attn.is_empty() && post_attn.is_empty() {
        return None;
    }

    Some((pre_attn, attn_ops, post_attn))
}

/// Legacy name-based attention split (fallback).
fn split_at_attention_boundary_by_name(
    ops: &[Operation],
) -> (Vec<Operation>, Vec<Operation>, Vec<Operation>) {
    let mut attn_start = None;
    let mut attn_end = None;

    for (i, op) in ops.iter().enumerate() {
        let name_lower = op.name.to_ascii_lowercase();
        let is_attn_op = name_lower.contains("rotaryembedding")
            || name_lower.contains("rotary_embedding")
            || name_lower.contains("_rotary_")
            || name_lower.contains("_rope_")
            || name_lower.contains("_cos_gather")
            || name_lower.contains("_sin_gather")
            || name_lower.contains("_cos_gathered")
            || name_lower.contains("_sin_gathered")
            || name_lower.contains("pos_ids_reformat")
            || name_lower.contains("position_ids_reformat")
            || name_lower.contains("_q_reshape")
            || name_lower.contains("_k_reshape")
            || name_lower.contains("_v_reshape")
            || name_lower.contains("_q_transpose")
            || name_lower.contains("_k_transpose")
            || name_lower.contains("_v_transpose")
            || name_lower.contains("_k_tile")
            || name_lower.contains("_v_tile")
            || name_lower.contains("_k_tiled")
            || name_lower.contains("_v_tiled")
            || name_lower.contains("_qk_matmul")
            || name_lower.contains("_qk_scale")
            || name_lower.contains("_attn_softmax")
            || name_lower.contains("_av_matmul")
            || name_lower.contains("_attn_out_reshape")
            || name_lower.contains("_attn_out_transpose");

        if is_attn_op {
            if attn_start.is_none() {
                attn_start = Some(i);
            }
            attn_end = Some(i);
        }
    }

    match (attn_start, attn_end) {
        (Some(start), Some(end)) => {
            let pre_attn = ops[..start].to_vec();
            let attn_cluster = ops[start..=end].to_vec();
            let post_attn = ops[end + 1..].to_vec();
            if pre_attn.is_empty() {
                // All ops before attention are attention ops, or there
                // are no ops at all — put everything in pre_attn.
                (ops.to_vec(), vec![], vec![])
            } else {
                (pre_attn, attn_cluster, post_attn)
            }
        }
        _ => (ops.to_vec(), vec![], vec![]),
    }
}

// ---------------------------------------------------------------------------
// Sub-program construction
// ---------------------------------------------------------------------------

/// Build a standalone [`SubProgram`] from a group of operations.
///
/// `all_ops` is the full set of operations in the parent function,
/// used to identify cross-boundary outputs (values produced by this
/// group that are consumed by ops outside this group).
fn build_sub_program(
    name: &str,
    ops: &[Operation],
    type_map: &HashMap<String, TensorType>,
    all_ops: Option<&[Operation]>,
) -> SubProgram {
    // Values produced within this group.
    let produced: HashSet<String> = ops
        .iter()
        .flat_map(|op| op.outputs.iter().cloned())
        .collect();

    // Values referenced by ops in this group.
    let referenced: HashSet<String> = ops
        .iter()
        .flat_map(|op| op.inputs.values())
        .filter_map(|v| match v {
            Value::Reference(r) => Some(r.clone()),
            _ => None,
        })
        .collect();

    // Inputs = referenced but not produced locally (cross-boundary).
    let mut input_pairs: Vec<(String, TensorType)> = referenced
        .difference(&produced)
        .filter_map(|r| type_map.get(r).map(|ty| (r.clone(), ty.clone())))
        .collect();
    // Deterministic ordering.
    input_pairs.sort_by(|a, b| a.0.cmp(&b.0));

    // Outputs: find values produced by this group that are consumed by
    // ops outside this group (cross-boundary outputs).
    let output_names: Vec<String> = if let Some(all) = all_ops {
        // Collect all references from ops NOT in this group.
        let external_refs: HashSet<String> = all
            .iter()
            .filter(|op| !produced.contains(op.outputs.first().unwrap_or(&String::new())))
            .flat_map(|op| op.inputs.values())
            .filter_map(|v| match v {
                Value::Reference(r) if produced.contains(r) => Some(r.clone()),
                _ => None,
            })
            .collect();
        let mut outs: Vec<String> = external_refs.into_iter().collect();
        outs.sort(); // deterministic
        if outs.is_empty() {
            // Terminal sub-program (e.g., lm_head) — use last op's outputs.
            ops.last().map(|op| op.outputs.clone()).unwrap_or_default()
        } else {
            outs
        }
    } else {
        // No context — fall back to last op's outputs.
        ops.last().map(|op| op.outputs.clone()).unwrap_or_default()
    };

    // Build Function.
    let mut func = Function::new("main");
    func.inputs = input_pairs.clone();
    func.body.operations = ops.to_vec();
    func.body.outputs = output_names.clone();

    // Build Program.
    let mut prog = Program::new("1.0");
    prog.add_function(func);

    // Build TensorDescriptors.
    let input_descriptors: Vec<TensorDescriptor> = input_pairs
        .iter()
        .map(|(n, ty)| tensor_descriptor_from(n, ty))
        .collect();

    let output_descriptors: Vec<TensorDescriptor> = output_names
        .iter()
        .filter_map(|n| type_map.get(n).map(|ty| tensor_descriptor_from(n, ty)))
        .collect();

    SubProgram {
        name: name.to_string(),
        program: prog,
        inputs: input_descriptors,
        outputs: output_descriptors,
    }
}

/// Convert a MIL [`TensorType`] to a 4-element shape + [`TensorDescriptor`].
fn tensor_descriptor_from(name: &str, ty: &TensorType) -> TensorDescriptor {
    let mut shape = [1usize; 4];
    for (i, dim) in ty.shape.iter().enumerate() {
        if i >= 4 {
            break;
        }
        shape[i] = dim.unwrap_or(1);
    }
    TensorDescriptor {
        name: name.to_string(),
        shape,
        dtype: ty.scalar_type,
    }
}

// ---------------------------------------------------------------------------
// Weight-size enforcement
// ---------------------------------------------------------------------------

/// Sum the weight (Tensor literal) data in a set of operations.
fn weight_data_size(ops: &[Operation]) -> usize {
    ops.iter()
        .flat_map(|op| op.inputs.values().chain(op.attributes.values()))
        .map(|v| match v {
            Value::Tensor { data, .. } => data.len(),
            _ => 0,
        })
        .sum()
}

/// If any sub-program exceeds the weight limit, chunk it further.
fn enforce_weight_limit(sub_programs: Vec<SubProgram>, config: &SplitConfig) -> Vec<SubProgram> {
    let mut result: Vec<SubProgram> = Vec::new();

    for sp in sub_programs {
        let ops = &sp.program.main().unwrap().body.operations;
        let total = weight_data_size(ops);

        if total <= config.max_weight_size || ops.len() <= 1 {
            result.push(sp);
            continue;
        }

        // Chunk ops so each chunk stays under the limit.
        let type_map = build_type_map_from_program(&sp.program);
        let mut chunk_idx = 0usize;
        let mut current_ops: Vec<Operation> = Vec::new();
        let mut current_size = 0usize;

        for op in ops {
            let op_size = weight_data_size(std::slice::from_ref(op));
            // Start a new chunk if adding this op would exceed the limit,
            // unless the current chunk is empty (single oversized op).
            if !current_ops.is_empty() && current_size + op_size > config.max_weight_size {
                let chunk_name = format!("{}_{}", sp.name, chunk_idx);
                result.push(build_sub_program(
                    &chunk_name,
                    &current_ops,
                    &type_map,
                    None,
                ));
                chunk_idx += 1;
                current_ops.clear();
                current_size = 0;
            }
            current_ops.push(op.clone());
            current_size += op_size;
        }

        if !current_ops.is_empty() {
            let chunk_name = if chunk_idx == 0 {
                sp.name.clone()
            } else {
                format!("{}_{}", sp.name, chunk_idx)
            };
            result.push(build_sub_program(
                &chunk_name,
                &current_ops,
                &type_map,
                None,
            ));
        }
    }

    result
}

/// Reconstruct a type-map from a [`Program`]'s main function.
fn build_type_map_from_program(program: &Program) -> HashMap<String, TensorType> {
    let mut map = HashMap::new();
    if let Some(func) = program.main() {
        for (name, ty) in &func.inputs {
            map.insert(name.clone(), ty.clone());
        }
        for op in &func.body.operations {
            for (out_name, out_ty) in op.outputs.iter().zip(op.output_types.iter()) {
                if let Some(ty) = out_ty {
                    map.insert(out_name.clone(), ty.clone());
                }
            }
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::{Operation, ScalarType, TensorType, Value};

    /// Helper: build a simple operation with a given name and op_type.
    fn make_op(name: &str, op_type: &str) -> Operation {
        Operation::new(op_type, name)
    }

    /// Helper: build an operation that references an input and produces an output.
    fn make_op_with_io(
        name: &str,
        op_type: &str,
        input_ref: &str,
        output_name: &str,
        output_type: Option<TensorType>,
    ) -> Operation {
        let mut op = Operation::new(op_type, name)
            .with_input("x", Value::Reference(input_ref.to_string()))
            .with_output(output_name);
        if let Some(ty) = output_type {
            op.output_types = vec![Some(ty)];
        }
        op
    }

    /// Helper: build a simple program from a list of ops, with given function inputs.
    fn make_program(ops: Vec<Operation>, func_inputs: Vec<(String, TensorType)>) -> Program {
        let mut func = Function::new("main");
        func.inputs = func_inputs;
        func.body.operations = ops.clone();
        if let Some(last) = ops.last() {
            func.body.outputs = last.outputs.clone();
        }
        let mut prog = Program::new("1.0");
        prog.add_function(func);
        prog
    }

    fn f32_type(shape: &[usize]) -> TensorType {
        TensorType::new(ScalarType::Float32, shape.to_vec())
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn split_empty_program() {
        let prog = Program::new("1.0");
        let split = split_for_ane(&prog, &SplitConfig::default()).unwrap();
        assert!(split.programs.is_empty());
    }

    #[test]
    fn split_empty_body() {
        let prog = make_program(vec![], vec![]);
        let split = split_for_ane(&prog, &SplitConfig::default()).unwrap();
        assert!(split.programs.is_empty());
    }

    #[test]
    fn split_single_layer() {
        // A model with only layer_0 ops → one sub-program.
        let ops = vec![
            make_op("layer_0_attn_q", "linear"),
            make_op("layer_0_attn_v", "linear"),
            make_op("layer_0_ffn_up", "linear"),
        ];
        let prog = make_program(ops, vec![]);
        let split = split_for_ane(&prog, &SplitConfig::default()).unwrap();

        assert_eq!(split.programs.len(), 1);
        assert_eq!(split.programs[0].name, "layer_0");
        assert_eq!(
            split.programs[0]
                .program
                .main()
                .unwrap()
                .body
                .operations
                .len(),
            3
        );
    }

    #[test]
    fn split_multi_layer() {
        // Embedding ops + two layers + post ops.
        let ops = vec![
            make_op("embed_tok", "gather"),
            make_op("layer_0_attn_q", "linear"),
            make_op("layer_0_ffn", "linear"),
            make_op("layer_1_attn_q", "linear"),
            make_op("layer_1_ffn", "linear"),
            make_op("final_norm", "layer_norm"),
            make_op("lm_head_proj", "linear"),
        ];
        let prog = make_program(ops, vec![]);
        let split = split_for_ane(&prog, &SplitConfig::default()).unwrap();

        let names: Vec<&str> = split.programs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, &["embedding", "layer_0", "layer_1", "lm_head"]);

        // layer_0 has 2 ops, layer_1 has 2 ops.
        assert_eq!(
            split.programs[1]
                .program
                .main()
                .unwrap()
                .body
                .operations
                .len(),
            2
        );
        assert_eq!(
            split.programs[2]
                .program
                .main()
                .unwrap()
                .body
                .operations
                .len(),
            2
        );
    }

    #[test]
    fn split_detects_layer_numbers() {
        assert_eq!(extract_layer_number("layer_0_attn_q"), Some(0));
        assert_eq!(extract_layer_number("layers.3.ffn.up"), Some(3));
        assert_eq!(extract_layer_number("block_12_norm"), Some(12));
        assert_eq!(extract_layer_number("embedding_lookup"), None);
        assert_eq!(extract_layer_number("lm_head"), None);
        // ONNX-converted names with prefixes
        assert_eq!(extract_layer_number("_model_layers_0_attn_q"), Some(0));
        assert_eq!(extract_layer_number("/model/layers.27/mlp/down"), Some(27));
    }

    #[test]
    fn split_cross_boundary_inputs() {
        // layer_0 produces "h0", layer_1 consumes "h0".
        let ty = f32_type(&[1, 128]);
        let ops = vec![
            make_op_with_io("layer_0_linear", "linear", "x_in", "h0", Some(ty.clone())),
            make_op_with_io("layer_1_linear", "linear", "h0", "h1", Some(ty.clone())),
        ];
        let func_inputs = vec![("x_in".to_string(), ty.clone())];
        let prog = make_program(ops, func_inputs);
        let split = split_for_ane(&prog, &SplitConfig::default()).unwrap();

        assert_eq!(split.programs.len(), 2);

        // layer_1's sub-program should list "h0" as an input.
        let layer1 = &split.programs[1];
        assert!(
            layer1.inputs.iter().any(|d| d.name == "h0"),
            "layer_1 should have cross-boundary input 'h0'"
        );
    }

    #[test]
    fn split_preserves_ops() {
        let ops = vec![
            make_op("embed", "gather"),
            make_op("layer_0_q", "linear"),
            make_op("layer_0_v", "linear"),
            make_op("layer_1_q", "linear"),
            make_op("layer_1_v", "linear"),
            make_op("norm", "layer_norm"),
        ];
        let total = ops.len();
        let prog = make_program(ops, vec![]);
        let split = split_for_ane(&prog, &SplitConfig::default()).unwrap();

        let split_total: usize = split
            .programs
            .iter()
            .map(|sp| sp.program.main().unwrap().body.operations.len())
            .sum();
        assert_eq!(split_total, total);
    }

    #[test]
    fn split_non_transformer() {
        // No layer numbering → fallback to a single "main" sub-program.
        let ops = vec![
            make_op("conv_a", "conv2d"),
            make_op("relu_a", "relu"),
            make_op("pool_a", "avg_pool"),
        ];
        let prog = make_program(ops.clone(), vec![]);
        let split = split_for_ane(&prog, &SplitConfig::default()).unwrap();

        assert_eq!(split.programs.len(), 1);
        assert_eq!(split.programs[0].name, "main");
        assert_eq!(
            split.programs[0]
                .program
                .main()
                .unwrap()
                .body
                .operations
                .len(),
            ops.len()
        );
    }

    #[test]
    fn split_weight_size_limit() {
        // Create ops with large tensor weights that exceed the limit.
        let big_tensor = Value::Tensor {
            data: vec![0u8; 40 * 1024 * 1024], // 40 MB each
            shape: vec![1024, 1024],
            dtype: ScalarType::Float32,
        };
        let mut op0 = make_op("layer_0_big", "linear");
        op0.inputs.insert("weight".to_string(), big_tensor.clone());
        let mut op1 = make_op("layer_0_big2", "linear");
        op1.inputs.insert("weight".to_string(), big_tensor);

        let prog = make_program(vec![op0, op1], vec![]);
        let config = SplitConfig {
            max_weight_size: 50 * 1024 * 1024, // 50 MB limit
            ..Default::default()
        };
        let split = split_for_ane(&prog, &config).unwrap();

        // The single layer_0 (80 MB total) should be chunked into 2.
        assert!(
            split.programs.len() >= 2,
            "expected chunking, got {} sub-programs",
            split.programs.len()
        );
    }

    #[test]
    fn split_attention_boundary() {
        // Simulate a layer with pre-attn ops, attention ops, and post-attn
        // ops. Uses the name-based heuristic fallback (no data-flow links).
        let ops = vec![
            make_op("embed_tok", "gather"),
            // Layer 0: pre-attn
            make_op("layer_0_norm", "layer_norm"),
            make_op("layer_0_q_linear", "linear"),
            make_op("layer_0_k_linear", "linear"),
            make_op("layer_0_v_linear", "linear"),
            // Layer 0: attention cluster
            make_op("layer_0_q_reshape_op", "reshape"),
            make_op("layer_0_q_transpose_op", "transpose"),
            make_op("layer_0_k_reshape_op", "reshape"),
            make_op("layer_0_k_transpose_op", "transpose"),
            make_op("layer_0_v_reshape_op", "reshape"),
            make_op("layer_0_v_transpose_op", "transpose"),
            make_op("layer_0_qk_matmul", "matmul"),
            make_op("layer_0_qk_scale", "mul"),
            make_op("layer_0_attn_softmax", "softmax"),
            make_op("layer_0_av_matmul", "matmul"),
            make_op("layer_0_attn_out_transpose", "transpose"),
            make_op("layer_0_attn_out_reshape", "reshape"),
            // Layer 0: post-attn
            make_op("layer_0_o_proj", "linear"),
            make_op("layer_0_ffn_up", "linear"),
            make_op("layer_0_ffn_down", "linear"),
            // Post
            make_op("final_norm", "layer_norm"),
            make_op("lm_head_proj", "linear"),
        ];
        let prog = make_program(ops, vec![]);
        let config = SplitConfig {
            split_attention: true,
            ..Default::default()
        };
        let split = split_for_ane(&prog, &config).unwrap();

        let names: Vec<&str> = split.programs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            &[
                "embedding",
                "layer_0_pre_attn",
                "layer_0_post_attn",
                "lm_head"
            ]
        );

        // pre_attn should have: norm, q_linear, k_linear, v_linear (4 ops)
        let pre_attn = &split.programs[1];
        assert_eq!(
            pre_attn.program.main().unwrap().body.operations.len(),
            4,
            "pre_attn should have 4 ops"
        );

        // post_attn should have: o_proj, ffn_up, ffn_down (3 ops)
        let post_attn = &split.programs[2];
        assert_eq!(
            post_attn.program.main().unwrap().body.operations.len(),
            3,
            "post_attn should have 3 ops"
        );
    }

    #[test]
    fn split_attention_disabled_keeps_whole_layers() {
        let ops = vec![
            make_op("layer_0_norm", "layer_norm"),
            make_op("layer_0_q_reshape_op", "reshape"),
            make_op("layer_0_attn_softmax", "softmax"),
            make_op("layer_0_ffn", "linear"),
        ];
        let prog = make_program(ops, vec![]);
        let config = SplitConfig::default(); // split_attention = false
        let split = split_for_ane(&prog, &config).unwrap();

        let names: Vec<&str> = split.programs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, &["layer_0"]);
    }

    #[test]
    fn split_emit_attention_produces_fp16_attn_sub() {
        // With emit_attention = true, the attention cluster ops should
        // appear as a layer_N_fp16_attn sub-program.
        let ops = vec![
            make_op("embed_tok", "gather"),
            // Layer 0: pre-attn
            make_op("layer_0_norm", "layer_norm"),
            make_op("layer_0_q_linear", "linear"),
            make_op("layer_0_k_linear", "linear"),
            make_op("layer_0_v_linear", "linear"),
            // Layer 0: attention cluster
            make_op("layer_0_q_reshape_op", "reshape"),
            make_op("layer_0_q_transpose_op", "transpose"),
            make_op("layer_0_k_reshape_op", "reshape"),
            make_op("layer_0_k_transpose_op", "transpose"),
            make_op("layer_0_v_reshape_op", "reshape"),
            make_op("layer_0_v_transpose_op", "transpose"),
            make_op("layer_0_qk_matmul", "matmul"),
            make_op("layer_0_qk_scale", "mul"),
            make_op("layer_0_attn_softmax", "softmax"),
            make_op("layer_0_av_matmul", "matmul"),
            make_op("layer_0_attn_out_transpose", "transpose"),
            make_op("layer_0_attn_out_reshape", "reshape"),
            // Layer 0: post-attn
            make_op("layer_0_o_proj", "linear"),
            make_op("layer_0_ffn_up", "linear"),
            make_op("layer_0_ffn_down", "linear"),
            // Post
            make_op("final_norm", "layer_norm"),
            make_op("lm_head_proj", "linear"),
        ];
        let prog = make_program(ops, vec![]);
        let config = SplitConfig {
            split_attention: true,
            emit_attention: true,
            ..Default::default()
        };
        let split = split_for_ane(&prog, &config).unwrap();

        let names: Vec<&str> = split.programs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            &[
                "embedding",
                "layer_0_pre_attn",
                "layer_0_fp16_attn",
                "layer_0_post_attn",
                "lm_head"
            ]
        );

        // fp16_attn should contain the attention cluster ops.
        let fp16_attn = &split.programs[2];
        let fp16_ops: Vec<&str> = fp16_attn
            .program
            .main()
            .unwrap()
            .body
            .operations
            .iter()
            .map(|op| op.name.as_str())
            .collect();
        assert!(
            fp16_ops.contains(&"layer_0_attn_softmax"),
            "fp16_attn should contain softmax, got: {fp16_ops:?}"
        );
        assert!(
            fp16_ops.contains(&"layer_0_qk_matmul"),
            "fp16_attn should contain QK matmul, got: {fp16_ops:?}"
        );
    }

    #[test]
    fn split_emit_attention_false_strips_cluster() {
        // With emit_attention = false (default), attention cluster is discarded.
        let ops = vec![
            make_op("embed_tok", "gather"),
            make_op("layer_0_norm", "layer_norm"),
            make_op("layer_0_q_reshape_op", "reshape"),
            make_op("layer_0_attn_softmax", "softmax"),
            make_op("layer_0_attn_out_reshape", "reshape"),
            make_op("layer_0_o_proj", "linear"),
            make_op("final_norm", "layer_norm"),
            make_op("lm_head_proj", "linear"),
        ];
        let prog = make_program(ops, vec![]);
        let config = SplitConfig {
            split_attention: true,
            emit_attention: false,
            ..Default::default()
        };
        let split = split_for_ane(&prog, &config).unwrap();

        let names: Vec<&str> = split.programs.iter().map(|s| s.name.as_str()).collect();
        // No fp16_attn sub-program should exist.
        assert!(
            !names.iter().any(|n| n.contains("fp16_attn")),
            "emit_attention=false should not produce fp16_attn sub-programs, got: {names:?}"
        );
    }

    // -------------------------------------------------------------------
    // Structural split helpers
    // -------------------------------------------------------------------

    /// Build a const (weight) op that produces a named output.
    fn make_const_op(name: &str, output_name: &str) -> Operation {
        Operation::new("const", name)
            .with_input(
                "val",
                Value::Tensor {
                    data: vec![0u8; 16],
                    shape: vec![4, 4],
                    dtype: ScalarType::Float16,
                },
            )
            .with_output(output_name)
    }

    /// Build a connected op with one `x` input reference and one output.
    fn make_connected_op(
        name: &str,
        op_type: &str,
        input_ref: &str,
        output_name: &str,
    ) -> Operation {
        Operation::new(op_type, name)
            .with_input("x", Value::Reference(input_ref.to_string()))
            .with_output(output_name)
    }

    /// Build a matmul op with two named inputs (x, y) and one output.
    fn make_matmul_op(name: &str, x_ref: &str, y_ref: &str, output_name: &str) -> Operation {
        Operation::new("matmul", name)
            .with_input("x", Value::Reference(x_ref.to_string()))
            .with_input("y", Value::Reference(y_ref.to_string()))
            .with_output(output_name)
    }

    /// Build a standard GQA layer op list with proper data-flow links.
    ///
    /// Graph:
    /// ```text
    /// [input] → norm → ─┬─ (+ q_weight const) → q_proj_matmul → q_reshape → q_transpose ──┐
    ///                    ├─ (+ k_weight const) → k_proj_matmul → k_reshape → k_transpose ──┤
    ///                    └─ (+ v_weight const) → v_proj_matmul → v_reshape → v_transpose ──┤
    ///                                                                                       │
    ///                    ┌───────────────── qk_matmul (Q × K^T) ◄──────────────────────────┘
    ///                    │
    ///                    └→ scale → softmax → av_matmul (attn × V) → out_reshape → out_transpose
    ///                                                                                       │
    ///                    (+ o_weight const) → o_proj_matmul ◄──────────────────────────────┘
    ///                    │
    ///                    └→ residual_add → (+ up_weight const) → ffn_up → ffn_act
    ///                       → (+ down_weight const) → ffn_down → ffn_residual
    /// ```
    fn make_gqa_layer_ops(prefix: &str) -> Vec<Operation> {
        let p = prefix;
        vec![
            // Input norm
            make_connected_op(
                &format!("{p}_norm"),
                "layer_norm",
                "hidden_in",
                &format!("{p}_norm_out"),
            ),
            // Q/K/V weight consts
            make_const_op(&format!("{p}_q_weight"), &format!("{p}_q_weight_out")),
            make_const_op(&format!("{p}_k_weight"), &format!("{p}_k_weight_out")),
            make_const_op(&format!("{p}_v_weight"), &format!("{p}_v_weight_out")),
            // Q/K/V projection matmuls
            make_matmul_op(
                &format!("{p}_q_proj"),
                &format!("{p}_norm_out"),
                &format!("{p}_q_weight_out"),
                &format!("{p}_q_proj_out"),
            ),
            make_matmul_op(
                &format!("{p}_k_proj"),
                &format!("{p}_norm_out"),
                &format!("{p}_k_weight_out"),
                &format!("{p}_k_proj_out"),
            ),
            make_matmul_op(
                &format!("{p}_v_proj"),
                &format!("{p}_norm_out"),
                &format!("{p}_v_weight_out"),
                &format!("{p}_v_proj_out"),
            ),
            // Attention reshapes/transposes
            make_connected_op(
                &format!("{p}_q_reshape"),
                "reshape",
                &format!("{p}_q_proj_out"),
                &format!("{p}_q_reshaped"),
            ),
            make_connected_op(
                &format!("{p}_q_transpose"),
                "transpose",
                &format!("{p}_q_reshaped"),
                &format!("{p}_q_transposed"),
            ),
            make_connected_op(
                &format!("{p}_k_reshape"),
                "reshape",
                &format!("{p}_k_proj_out"),
                &format!("{p}_k_reshaped"),
            ),
            make_connected_op(
                &format!("{p}_k_transpose"),
                "transpose",
                &format!("{p}_k_reshaped"),
                &format!("{p}_k_transposed"),
            ),
            make_connected_op(
                &format!("{p}_v_reshape"),
                "reshape",
                &format!("{p}_v_proj_out"),
                &format!("{p}_v_reshaped"),
            ),
            make_connected_op(
                &format!("{p}_v_transpose"),
                "transpose",
                &format!("{p}_v_reshaped"),
                &format!("{p}_v_transposed"),
            ),
            // QK^T matmul
            make_matmul_op(
                &format!("{p}_qk_matmul"),
                &format!("{p}_q_transposed"),
                &format!("{p}_k_transposed"),
                &format!("{p}_qk_scores"),
            ),
            // Scale
            make_connected_op(
                &format!("{p}_qk_scale"),
                "mul",
                &format!("{p}_qk_scores"),
                &format!("{p}_qk_scaled"),
            ),
            // Softmax
            make_connected_op(
                &format!("{p}_softmax"),
                "softmax",
                &format!("{p}_qk_scaled"),
                &format!("{p}_attn_weights"),
            ),
            // AV matmul
            make_matmul_op(
                &format!("{p}_av_matmul"),
                &format!("{p}_attn_weights"),
                &format!("{p}_v_transposed"),
                &format!("{p}_av_out"),
            ),
            // Output reshape/transpose
            make_connected_op(
                &format!("{p}_out_reshape"),
                "reshape",
                &format!("{p}_av_out"),
                &format!("{p}_out_reshaped"),
            ),
            make_connected_op(
                &format!("{p}_out_transpose"),
                "transpose",
                &format!("{p}_out_reshaped"),
                &format!("{p}_out_transposed"),
            ),
            // O projection
            make_const_op(&format!("{p}_o_weight"), &format!("{p}_o_weight_out")),
            make_matmul_op(
                &format!("{p}_o_proj"),
                &format!("{p}_out_transposed"),
                &format!("{p}_o_weight_out"),
                &format!("{p}_o_proj_out"),
            ),
            // Residual add
            make_matmul_op(
                &format!("{p}_residual_add"),
                &format!("{p}_o_proj_out"),
                "hidden_in",
                &format!("{p}_residual_out"),
            ),
            // FFN
            make_const_op(&format!("{p}_up_weight"), &format!("{p}_up_weight_out")),
            make_matmul_op(
                &format!("{p}_ffn_up"),
                &format!("{p}_residual_out"),
                &format!("{p}_up_weight_out"),
                &format!("{p}_ffn_up_out"),
            ),
            make_connected_op(
                &format!("{p}_ffn_act"),
                "relu",
                &format!("{p}_ffn_up_out"),
                &format!("{p}_ffn_act_out"),
            ),
            make_const_op(&format!("{p}_down_weight"), &format!("{p}_down_weight_out")),
            make_matmul_op(
                &format!("{p}_ffn_down"),
                &format!("{p}_ffn_act_out"),
                &format!("{p}_down_weight_out"),
                &format!("{p}_ffn_down_out"),
            ),
        ]
    }

    // -------------------------------------------------------------------
    // Structural split tests
    // -------------------------------------------------------------------

    #[test]
    fn structural_split_standard_gqa() {
        // Standard GQA layer with proper data-flow links.
        // The structural split should identify the attention cluster by
        // following data flow, not op names.
        let layer_ops = make_gqa_layer_ops("layer_0");
        let (pre, attn, post) = split_at_attention_boundary(&layer_ops);

        let pre_names: Vec<&str> = pre.iter().map(|op| op.name.as_str()).collect();
        let attn_names: Vec<&str> = attn.iter().map(|op| op.name.as_str()).collect();
        let post_names: Vec<&str> = post.iter().map(|op| op.name.as_str()).collect();

        // Pre-attn: norm + Q/K/V weight consts (norm feeds into projections).
        assert!(
            pre_names.contains(&"layer_0_norm"),
            "pre-attn should contain the norm op, got: {pre_names:?}"
        );

        // Post-attn: O projection, residual, FFN ops and their weight consts.
        assert!(
            post_names.contains(&"layer_0_o_proj"),
            "post-attn should contain O projection, got: {post_names:?}"
        );
        assert!(
            post_names.contains(&"layer_0_ffn_down"),
            "post-attn should contain FFN down, got: {post_names:?}"
        );

        // Attention cluster ops should be in the attn group, not pre or post.
        assert!(!attn.is_empty(), "attention cluster should not be empty");
        assert!(
            attn_names.contains(&"layer_0_softmax"),
            "attention cluster should contain softmax, got: {attn_names:?}"
        );
        let all_returned: HashSet<&str> =
            pre_names.iter().chain(post_names.iter()).copied().collect();
        assert!(
            !all_returned.contains("layer_0_softmax"),
            "softmax should be stripped (in attention cluster)"
        );
        assert!(
            !all_returned.contains("layer_0_qk_matmul"),
            "QK matmul should be stripped (in attention cluster)"
        );
        assert!(
            !all_returned.contains("layer_0_av_matmul"),
            "AV matmul should be stripped (in attention cluster)"
        );
        // Q/K/V projection matmuls should be in pre_attn (not stripped)
        // so TurboQuant receives correct Q/K/V inputs.
        assert!(
            pre_names.contains(&"layer_0_q_proj"),
            "Q projection matmul should be in pre_attn, got: {pre_names:?}"
        );
    }

    #[test]
    fn structural_split_qwen3_per_head_norms() {
        // Qwen3-like layer with per-head K/V norms between projection
        // and attention. The K-norm reshapes should be classified as
        // attention cluster ops, NOT pre-attn — this was the original bug.
        let p = "layer_1";
        let mut ops = vec![
            // Input norm
            make_connected_op(
                &format!("{p}_norm"),
                "layer_norm",
                "hidden_in",
                &format!("{p}_norm_out"),
            ),
            // Q/K/V weight consts
            make_const_op(&format!("{p}_q_weight"), &format!("{p}_q_weight_out")),
            make_const_op(&format!("{p}_k_weight"), &format!("{p}_k_weight_out")),
            make_const_op(&format!("{p}_v_weight"), &format!("{p}_v_weight_out")),
            // Q/K/V projection matmuls
            make_matmul_op(
                &format!("{p}_q_proj"),
                &format!("{p}_norm_out"),
                &format!("{p}_q_weight_out"),
                &format!("{p}_q_proj_out"),
            ),
            make_matmul_op(
                &format!("{p}_k_proj"),
                &format!("{p}_norm_out"),
                &format!("{p}_k_weight_out"),
                &format!("{p}_k_proj_out"),
            ),
            make_matmul_op(
                &format!("{p}_v_proj"),
                &format!("{p}_norm_out"),
                &format!("{p}_v_weight_out"),
                &format!("{p}_v_proj_out"),
            ),
            // --- Per-head K norm (the Qwen3 pattern that caused the bug) ---
            // K reshape for per-head norm (1024 → [batch, heads, head_dim])
            make_connected_op(
                &format!("{p}_attn_k_norm_Reshape_0"),
                "reshape",
                &format!("{p}_k_proj_out"),
                &format!("{p}_k_for_norm"),
            ),
            // K per-head norm (SimplifiedLayerNormalization)
            make_connected_op(
                &format!("{p}_attn_k_norm"),
                "layer_norm",
                &format!("{p}_k_for_norm"),
                &format!("{p}_k_normed"),
            ),
            // K reshape back after norm
            make_connected_op(
                &format!("{p}_attn_k_norm_Reshape_1"),
                "reshape",
                &format!("{p}_k_normed"),
                &format!("{p}_k_normed_reshaped"),
            ),
            // --- Per-head Q norm ---
            make_connected_op(
                &format!("{p}_attn_q_norm_Reshape_0"),
                "reshape",
                &format!("{p}_q_proj_out"),
                &format!("{p}_q_for_norm"),
            ),
            make_connected_op(
                &format!("{p}_attn_q_norm"),
                "layer_norm",
                &format!("{p}_q_for_norm"),
                &format!("{p}_q_normed"),
            ),
            make_connected_op(
                &format!("{p}_attn_q_norm_Reshape_1"),
                "reshape",
                &format!("{p}_q_normed"),
                &format!("{p}_q_normed_reshaped"),
            ),
            // V reshape (no norm for V in Qwen3)
            make_connected_op(
                &format!("{p}_v_reshape"),
                "reshape",
                &format!("{p}_v_proj_out"),
                &format!("{p}_v_reshaped"),
            ),
        ];
        // Standard attention core after the norms.
        ops.extend(vec![
            make_connected_op(
                &format!("{p}_q_transpose"),
                "transpose",
                &format!("{p}_q_normed_reshaped"),
                &format!("{p}_q_transposed"),
            ),
            make_connected_op(
                &format!("{p}_k_transpose"),
                "transpose",
                &format!("{p}_k_normed_reshaped"),
                &format!("{p}_k_transposed"),
            ),
            make_connected_op(
                &format!("{p}_v_transpose"),
                "transpose",
                &format!("{p}_v_reshaped"),
                &format!("{p}_v_transposed"),
            ),
            make_matmul_op(
                &format!("{p}_qk_matmul"),
                &format!("{p}_q_transposed"),
                &format!("{p}_k_transposed"),
                &format!("{p}_qk_scores"),
            ),
            make_connected_op(
                &format!("{p}_qk_scale"),
                "mul",
                &format!("{p}_qk_scores"),
                &format!("{p}_qk_scaled"),
            ),
            make_connected_op(
                &format!("{p}_softmax"),
                "softmax",
                &format!("{p}_qk_scaled"),
                &format!("{p}_attn_weights"),
            ),
            make_matmul_op(
                &format!("{p}_av_matmul"),
                &format!("{p}_attn_weights"),
                &format!("{p}_v_transposed"),
                &format!("{p}_av_out"),
            ),
            make_connected_op(
                &format!("{p}_out_reshape"),
                "reshape",
                &format!("{p}_av_out"),
                &format!("{p}_out_reshaped"),
            ),
            make_connected_op(
                &format!("{p}_out_transpose"),
                "transpose",
                &format!("{p}_out_reshaped"),
                &format!("{p}_out_transposed"),
            ),
            // O projection + post-attn
            make_const_op(&format!("{p}_o_weight"), &format!("{p}_o_weight_out")),
            make_matmul_op(
                &format!("{p}_o_proj"),
                &format!("{p}_out_transposed"),
                &format!("{p}_o_weight_out"),
                &format!("{p}_o_proj_out"),
            ),
            make_const_op(&format!("{p}_down_weight"), &format!("{p}_down_weight_out")),
            make_matmul_op(
                &format!("{p}_ffn_down"),
                &format!("{p}_o_proj_out"),
                &format!("{p}_down_weight_out"),
                &format!("{p}_ffn_down_out"),
            ),
        ]);

        let (pre, attn, post) = split_at_attention_boundary(&ops);

        let pre_names: Vec<&str> = pre.iter().map(|op| op.name.as_str()).collect();
        let attn_names: Vec<&str> = attn.iter().map(|op| op.name.as_str()).collect();
        let post_names: Vec<&str> = post.iter().map(|op| op.name.as_str()).collect();
        let all_returned: HashSet<&str> =
            pre_names.iter().chain(post_names.iter()).copied().collect();

        // The K-norm reshapes must NOT be in pre-attn (this was the original bug).
        assert!(
            !pre_names.contains(&"layer_1_attn_k_norm_Reshape_0"),
            "K-norm reshape should NOT be in pre-attn (was the original bug), pre: {pre_names:?}"
        );
        assert!(
            !pre_names.contains(&"layer_1_attn_k_norm_Reshape_1"),
            "K-norm reshape-back should NOT be in pre-attn, pre: {pre_names:?}"
        );

        // K-norm ops should be in the attention cluster.
        assert!(
            attn_names.contains(&"layer_1_attn_k_norm_Reshape_0"),
            "K-norm reshape should be in attention cluster, got: {attn_names:?}"
        );
        assert!(
            attn_names.contains(&"layer_1_attn_k_norm"),
            "K per-head norm should be in attention cluster, got: {attn_names:?}"
        );
        assert!(
            !all_returned.contains("layer_1_attn_k_norm_Reshape_0"),
            "K-norm reshape should not be in pre or post"
        );
        assert!(
            !all_returned.contains("layer_1_attn_k_norm"),
            "K per-head norm should not be in pre or post"
        );

        // Pre-attn should contain the input norm.
        assert!(
            pre_names.contains(&"layer_1_norm"),
            "pre-attn should contain input norm, got: {pre_names:?}"
        );

        // Post-attn should contain O projection and FFN.
        assert!(
            post_names.contains(&"layer_1_o_proj"),
            "post-attn should contain O projection, got: {post_names:?}"
        );
        assert!(
            post_names.contains(&"layer_1_ffn_down"),
            "post-attn should contain FFN down, got: {post_names:?}"
        );
    }

    #[test]
    fn structural_split_no_attention() {
        // Layer with no softmax → fallback puts everything in pre-attn.
        let ops = vec![
            make_connected_op("layer_0_norm", "layer_norm", "x", "norm_out"),
            make_const_op("layer_0_w", "w_out"),
            make_matmul_op("layer_0_linear", "norm_out", "w_out", "linear_out"),
        ];
        let (pre, attn, post) = split_at_attention_boundary(&ops);

        assert_eq!(pre.len(), 3, "all ops should be in pre-attn");
        assert!(attn.is_empty(), "attn cluster should be empty");
        assert!(post.is_empty(), "post-attn should be empty");
    }

    #[test]
    fn structural_split_layer_0_edge_case() {
        // Layer 0 edge case: the input norm might trace to a function
        // input rather than a previous op. The structural split should
        // still correctly identify pre-attn ops.
        let layer_ops = make_gqa_layer_ops("layer_0");
        let (pre, _attn, post) = split_at_attention_boundary(&layer_ops);

        // Pre-attn should not be empty (layer 0 previously had empty pre-attn).
        assert!(
            !pre.is_empty(),
            "layer 0 pre-attn should NOT be empty (was the bug)"
        );

        // Should at least contain the norm op.
        let pre_names: Vec<&str> = pre.iter().map(|op| op.name.as_str()).collect();
        assert!(
            pre_names.contains(&"layer_0_norm"),
            "layer 0 pre-attn should contain norm, got: {pre_names:?}"
        );

        // Post-attn should contain O proj and FFN.
        assert!(!post.is_empty(), "layer 0 post-attn should not be empty");
        let post_names: Vec<&str> = post.iter().map(|op| op.name.as_str()).collect();
        assert!(
            post_names.contains(&"layer_0_o_proj"),
            "layer 0 post-attn should contain O proj, got: {post_names:?}"
        );
    }

    #[test]
    fn opgraph_build_and_walk() {
        // Unit test for OpGraph itself.
        let ops = vec![
            make_const_op("w", "w_out"),
            make_connected_op("norm", "layer_norm", "x_in", "norm_out"),
            make_matmul_op("proj", "norm_out", "w_out", "proj_out"),
            make_connected_op("act", "relu", "proj_out", "act_out"),
        ];
        let graph = OpGraph::build(&ops);

        // w (0) → proj (2)
        assert!(graph.forward[0].contains(&2));
        // norm (1) → proj (2)
        assert!(graph.forward[1].contains(&2));
        // proj (2) → act (3)
        assert!(graph.forward[2].contains(&3));

        // Backward walk from act should reach w, norm, proj.
        let ancestors = graph.walk_backward(3);
        assert!(ancestors.contains(&0), "should reach const w");
        assert!(ancestors.contains(&1), "should reach norm");
        assert!(ancestors.contains(&2), "should reach proj");

        // Forward walk from norm should reach proj, act.
        let descendants = graph.walk_forward(1);
        assert!(descendants.contains(&2), "should reach proj");
        assert!(descendants.contains(&3), "should reach act");
    }

    // -------------------------------------------------------------------
    // Fused attention and softmax variant tests
    // -------------------------------------------------------------------

    #[test]
    fn structural_split_fused_gqa_op() {
        // A layer where attention is a single fused GroupQueryAttention op
        // (not decomposed into matmul/softmax). The structural split should
        // use the fused op as the attention cluster.
        let p = "layer_0";
        let ops = vec![
            // Input norm
            make_connected_op(
                &format!("{p}_norm"),
                "layer_norm",
                "hidden_in",
                &format!("{p}_norm_out"),
            ),
            // Q/K/V weight consts (fed directly to the fused op)
            make_const_op(&format!("{p}_q_weight"), &format!("{p}_q_weight_out")),
            make_const_op(&format!("{p}_k_weight"), &format!("{p}_k_weight_out")),
            make_const_op(&format!("{p}_v_weight"), &format!("{p}_v_weight_out")),
            // Fused GroupQueryAttention op
            Operation::new("GroupQueryAttention", &format!("{p}_gqa"))
                .with_input("x", Value::Reference(format!("{p}_norm_out")))
                .with_input("q_weight", Value::Reference(format!("{p}_q_weight_out")))
                .with_input("k_weight", Value::Reference(format!("{p}_k_weight_out")))
                .with_input("v_weight", Value::Reference(format!("{p}_v_weight_out")))
                .with_output(&format!("{p}_gqa_out")),
            // O projection
            make_const_op(&format!("{p}_o_weight"), &format!("{p}_o_weight_out")),
            make_matmul_op(
                &format!("{p}_o_proj"),
                &format!("{p}_gqa_out"),
                &format!("{p}_o_weight_out"),
                &format!("{p}_o_proj_out"),
            ),
            // FFN
            make_const_op(&format!("{p}_down_weight"), &format!("{p}_down_weight_out")),
            make_matmul_op(
                &format!("{p}_ffn_down"),
                &format!("{p}_o_proj_out"),
                &format!("{p}_down_weight_out"),
                &format!("{p}_ffn_down_out"),
            ),
        ];

        let (pre, attn, post) = split_at_attention_boundary(&ops);

        let pre_names: Vec<&str> = pre.iter().map(|op| op.name.as_str()).collect();
        let attn_names: Vec<&str> = attn.iter().map(|op| op.name.as_str()).collect();
        let post_names: Vec<&str> = post.iter().map(|op| op.name.as_str()).collect();
        let all_returned: HashSet<&str> =
            pre_names.iter().chain(post_names.iter()).copied().collect();

        // Pre-attn should contain the norm.
        assert!(
            pre_names.contains(&"layer_0_norm"),
            "pre-attn should contain norm, got: {pre_names:?}"
        );

        // The fused GQA op should be in the attention cluster.
        assert!(
            attn_names.contains(&"layer_0_gqa"),
            "fused GQA op should be in attention cluster, got: {attn_names:?}"
        );
        assert!(
            !all_returned.contains("layer_0_gqa"),
            "fused GQA op should not be in pre or post, but found in: {all_returned:?}"
        );

        // Post-attn should contain O projection and FFN.
        assert!(
            post_names.contains(&"layer_0_o_proj"),
            "post-attn should contain O projection, got: {post_names:?}"
        );
        assert!(
            post_names.contains(&"layer_0_ffn_down"),
            "post-attn should contain FFN down, got: {post_names:?}"
        );
    }

    #[test]
    fn structural_split_fused_gqa_snake_case() {
        // The GqaFusionPass emits "grouped_query_attention" (snake_case).
        // Verify the structural split recognises this variant.
        let p = "layer_0";
        let ops = vec![
            make_connected_op(
                &format!("{p}_norm"),
                "layer_norm",
                "hidden_in",
                &format!("{p}_norm_out"),
            ),
            make_const_op(&format!("{p}_q_weight"), &format!("{p}_q_weight_out")),
            make_const_op(&format!("{p}_k_weight"), &format!("{p}_k_weight_out")),
            make_const_op(&format!("{p}_v_weight"), &format!("{p}_v_weight_out")),
            Operation::new("grouped_query_attention", &format!("{p}_gqa"))
                .with_input("Q", Value::Reference(format!("{p}_norm_out")))
                .with_input("K", Value::Reference(format!("{p}_q_weight_out")))
                .with_input("V", Value::Reference(format!("{p}_k_weight_out")))
                .with_output(&format!("{p}_gqa_out")),
            make_const_op(&format!("{p}_o_weight"), &format!("{p}_o_weight_out")),
            make_matmul_op(
                &format!("{p}_o_proj"),
                &format!("{p}_gqa_out"),
                &format!("{p}_o_weight_out"),
                &format!("{p}_o_proj_out"),
            ),
            make_const_op(&format!("{p}_down_weight"), &format!("{p}_down_weight_out")),
            make_matmul_op(
                &format!("{p}_ffn_down"),
                &format!("{p}_o_proj_out"),
                &format!("{p}_down_weight_out"),
                &format!("{p}_ffn_down_out"),
            ),
        ];

        let (pre, attn, post) = split_at_attention_boundary(&ops);
        let attn_names: Vec<&str> = attn.iter().map(|op| op.name.as_str()).collect();
        let post_names: Vec<&str> = post.iter().map(|op| op.name.as_str()).collect();

        assert!(
            attn_names.contains(&"layer_0_gqa"),
            "snake_case GQA op should be in attention cluster, got: {attn_names:?}"
        );
        assert!(
            post_names.contains(&"layer_0_o_proj"),
            "post-attn should contain O projection, got: {post_names:?}"
        );
    }

    #[test]
    fn structural_split_sdpa_fused_op() {
        // A layer where attention is a single fused scaled_dot_product_attention
        // op. It should be routed through the fused attention path (not the
        // softmax path).
        let p = "layer_0";
        let ops = vec![
            // Input norm
            make_connected_op(
                &format!("{p}_norm"),
                "layer_norm",
                "hidden_in",
                &format!("{p}_norm_out"),
            ),
            // Q/K/V weight consts (fed directly to the fused op)
            make_const_op(&format!("{p}_q_weight"), &format!("{p}_q_weight_out")),
            make_const_op(&format!("{p}_k_weight"), &format!("{p}_k_weight_out")),
            make_const_op(&format!("{p}_v_weight"), &format!("{p}_v_weight_out")),
            // Fused scaled_dot_product_attention op
            Operation::new("scaled_dot_product_attention", &format!("{p}_sdpa"))
                .with_input("x", Value::Reference(format!("{p}_norm_out")))
                .with_input("q_weight", Value::Reference(format!("{p}_q_weight_out")))
                .with_input("k_weight", Value::Reference(format!("{p}_k_weight_out")))
                .with_input("v_weight", Value::Reference(format!("{p}_v_weight_out")))
                .with_output(&format!("{p}_sdpa_out")),
            // O projection
            make_const_op(&format!("{p}_o_weight"), &format!("{p}_o_weight_out")),
            make_matmul_op(
                &format!("{p}_o_proj"),
                &format!("{p}_sdpa_out"),
                &format!("{p}_o_weight_out"),
                &format!("{p}_o_proj_out"),
            ),
            // FFN
            make_const_op(&format!("{p}_down_weight"), &format!("{p}_down_weight_out")),
            make_matmul_op(
                &format!("{p}_ffn_down"),
                &format!("{p}_o_proj_out"),
                &format!("{p}_down_weight_out"),
                &format!("{p}_ffn_down_out"),
            ),
        ];

        let (pre, attn, post) = split_at_attention_boundary(&ops);

        let pre_names: Vec<&str> = pre.iter().map(|op| op.name.as_str()).collect();
        let attn_names: Vec<&str> = attn.iter().map(|op| op.name.as_str()).collect();
        let post_names: Vec<&str> = post.iter().map(|op| op.name.as_str()).collect();
        let all_returned: HashSet<&str> =
            pre_names.iter().chain(post_names.iter()).copied().collect();

        // Pre-attn should contain the norm.
        assert!(
            pre_names.contains(&"layer_0_norm"),
            "pre-attn should contain norm, got: {pre_names:?}"
        );

        // The fused SDPA op should be in the attention cluster.
        assert!(
            attn_names.contains(&"layer_0_sdpa"),
            "fused SDPA op should be in attention cluster, got: {attn_names:?}"
        );
        assert!(
            !all_returned.contains("layer_0_sdpa"),
            "fused SDPA op should not be in pre or post, but found in: {all_returned:?}"
        );

        // SDPA must NOT appear in find_softmax_ops.
        assert!(
            find_softmax_ops(&ops).is_empty(),
            "scaled_dot_product_attention should not be detected by find_softmax_ops"
        );

        // SDPA must appear in find_fused_attention_ops.
        assert!(
            !find_fused_attention_ops(&ops).is_empty(),
            "scaled_dot_product_attention should be detected by find_fused_attention_ops"
        );

        // Post-attn should contain O projection and FFN.
        assert!(
            post_names.contains(&"layer_0_o_proj"),
            "post-attn should contain O projection, got: {post_names:?}"
        );
        assert!(
            post_names.contains(&"layer_0_ffn_down"),
            "post-attn should contain FFN down, got: {post_names:?}"
        );
    }

    #[test]
    fn structural_split_softmax_variant_op_type() {
        // A layer where softmax has a variant op_type (e.g. `reduce_softmax`).
        let p = "layer_0";
        let mut ops = vec![
            // Input norm
            make_connected_op(
                &format!("{p}_norm"),
                "layer_norm",
                "hidden_in",
                &format!("{p}_norm_out"),
            ),
            // Q/K/V weight consts
            make_const_op(&format!("{p}_q_weight"), &format!("{p}_q_weight_out")),
            make_const_op(&format!("{p}_k_weight"), &format!("{p}_k_weight_out")),
            make_const_op(&format!("{p}_v_weight"), &format!("{p}_v_weight_out")),
            // Q/K/V projection matmuls
            make_matmul_op(
                &format!("{p}_q_proj"),
                &format!("{p}_norm_out"),
                &format!("{p}_q_weight_out"),
                &format!("{p}_q_proj_out"),
            ),
            make_matmul_op(
                &format!("{p}_k_proj"),
                &format!("{p}_norm_out"),
                &format!("{p}_k_weight_out"),
                &format!("{p}_k_proj_out"),
            ),
            make_matmul_op(
                &format!("{p}_v_proj"),
                &format!("{p}_norm_out"),
                &format!("{p}_v_weight_out"),
                &format!("{p}_v_proj_out"),
            ),
        ];
        // QK matmul
        ops.push(make_matmul_op(
            &format!("{p}_qk_matmul"),
            &format!("{p}_q_proj_out"),
            &format!("{p}_k_proj_out"),
            &format!("{p}_qk_scores"),
        ));
        // Use variant softmax op_type: "reduce_softmax"
        ops.push(make_connected_op(
            &format!("{p}_softmax"),
            "reduce_softmax",
            &format!("{p}_qk_scores"),
            &format!("{p}_attn_weights"),
        ));
        // AV matmul
        ops.push(make_matmul_op(
            &format!("{p}_av_matmul"),
            &format!("{p}_attn_weights"),
            &format!("{p}_v_proj_out"),
            &format!("{p}_av_out"),
        ));
        // Output reshape
        ops.push(make_connected_op(
            &format!("{p}_out_reshape"),
            "reshape",
            &format!("{p}_av_out"),
            &format!("{p}_out_reshaped"),
        ));
        // O projection
        ops.push(make_const_op(
            &format!("{p}_o_weight"),
            &format!("{p}_o_weight_out"),
        ));
        ops.push(make_matmul_op(
            &format!("{p}_o_proj"),
            &format!("{p}_out_reshaped"),
            &format!("{p}_o_weight_out"),
            &format!("{p}_o_proj_out"),
        ));
        // FFN
        ops.push(make_const_op(
            &format!("{p}_down_weight"),
            &format!("{p}_down_weight_out"),
        ));
        ops.push(make_matmul_op(
            &format!("{p}_ffn_down"),
            &format!("{p}_o_proj_out"),
            &format!("{p}_down_weight_out"),
            &format!("{p}_ffn_down_out"),
        ));

        let (pre, attn, post) = split_at_attention_boundary(&ops);

        let pre_names: Vec<&str> = pre.iter().map(|op| op.name.as_str()).collect();
        let attn_names: Vec<&str> = attn.iter().map(|op| op.name.as_str()).collect();
        let post_names: Vec<&str> = post.iter().map(|op| op.name.as_str()).collect();

        // Pre-attn should contain the norm.
        assert!(
            pre_names.contains(&"layer_0_norm"),
            "pre-attn should contain norm, got: {pre_names:?}"
        );

        // The reduce_softmax variant should be in the attention cluster.
        assert!(
            attn_names.contains(&"layer_0_softmax"),
            "softmax variant should be in attention cluster, got: {attn_names:?}"
        );

        // Post-attn should contain O projection and FFN.
        assert!(
            post_names.contains(&"layer_0_o_proj"),
            "post-attn should contain O projection, got: {post_names:?}"
        );
        assert!(
            post_names.contains(&"layer_0_ffn_down"),
            "post-attn should contain FFN down, got: {post_names:?}"
        );
    }
}
