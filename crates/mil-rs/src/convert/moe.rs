//! Mixture-of-Experts (MoE) architecture detection and model splitting.
//!
//! Detects MoE patterns in a MIL [`Program`] and splits the model into
//! separate programs for shared layers and individual experts. This enables
//! efficient deployment on Apple Neural Engine by keeping each expert within
//! ANE operation limits.
//!
//! # Detection Strategy
//!
//! MoE models are identified by two complementary strategies:
//!
//! 1. **Name-based**: Operations with names containing "expert" patterns
//!    (common in HuggingFace/Mixtral model exports).
//! 2. **Structure-based**: A linear→softmax router pattern followed by parallel
//!    linear/matmul paths that share a common input and don't depend on each other.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::MilError;
use crate::ir::{Block, Function, Operation, Program, ScalarType, TensorType, Value};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Detected MoE topology within a program.
#[derive(Debug, Clone)]
pub struct MoeTopology {
    /// Number of experts detected.
    pub expert_count: usize,
    /// Indices of operations forming the router/gating network.
    pub router_op_indices: Vec<usize>,
    /// Per-expert operation indices. `expert_op_indices[i]` contains the
    /// operation indices belonging to expert `i`.
    pub expert_op_indices: Vec<Vec<usize>>,
    /// Indices of shared operations (everything not router or expert).
    pub shared_op_indices: Vec<usize>,
    /// Output name of the router (feeds expert selection).
    pub router_output: String,
    /// Input tensor name consumed by each expert group.
    pub expert_input_names: Vec<String>,
    /// Output tensor name produced by each expert group.
    pub expert_output_names: Vec<String>,
}

/// Result of splitting a MoE program into shared + per-expert artifacts.
#[derive(Debug)]
pub struct MoeSplitResult {
    /// Program containing shared layers (embeddings, router, norms, LM head).
    pub shared: Program,
    /// Per-expert programs, one for each expert's FFN layers.
    pub experts: Vec<Program>,
    /// Topology manifest for runtime orchestration.
    pub manifest: MoeManifest,
}

/// JSON-serializable manifest describing MoE topology for runtime orchestration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeManifest {
    /// Number of experts.
    pub expert_count: usize,
    /// Router output tensor name.
    pub router_output: String,
    /// Per-expert I/O descriptors.
    pub experts: Vec<ExpertDescriptor>,
    /// Execution stages in order.
    pub stages: Vec<Stage>,
}

/// I/O descriptor for a single expert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertDescriptor {
    /// Expert index (0-based).
    pub index: usize,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
}

/// A named execution stage in the MoE pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    /// Stage name.
    pub name: String,
    /// Artifact to run (e.g., "shared", "expert-0").
    pub artifact: String,
    /// Input tensor names for this stage.
    pub inputs: Vec<String>,
    /// Output tensor names from this stage.
    pub outputs: Vec<String>,
}

/// Activation frequency profile for expert fusion.
///
/// Maps expert names or indices to their observed activation frequency or
/// raw activation count.  Deserialized from a JSON file produced by a
/// calibration run.
///
/// ```json
/// {
///   "expert_frequencies": {
///     "0": 0.45,
///     "1": 0.30,
///     "2": 0.15,
///     "3": 0.10
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertFrequencyProfile {
    /// Maps expert name/index (as string) to activation frequency (0.0–1.0)
    /// or raw activation count.  Keys are matched against expert indices
    /// from [`MoeTopology`].
    pub expert_frequencies: HashMap<String, f64>,
}

impl ExpertFrequencyProfile {
    /// Load a profile from a JSON file.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Return the top-K expert indices sorted by descending frequency.
    ///
    /// Expert keys are parsed as `usize`.  Keys that don't parse are
    /// silently ignored.
    ///
    /// # Errors
    ///
    /// Returns [`MilError::Validation`] if any frequency value is NaN or
    /// infinite.
    pub fn top_k(&self, k: usize) -> Result<Vec<usize>, MilError> {
        if self.expert_frequencies.values().any(|v| !v.is_finite()) {
            return Err(MilError::Validation(
                "expert frequency profile contains NaN or infinite values".into(),
            ));
        }

        let mut entries: Vec<(usize, f64)> = self
            .expert_frequencies
            .iter()
            .filter_map(|(key, &freq)| key.parse::<usize>().ok().map(|id| (id, freq)))
            .collect();
        entries.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(entries.into_iter().take(k).map(|(id, _)| id).collect())
    }

    /// Build a uniform profile where every expert has equal frequency.
    pub fn uniform(expert_count: usize) -> Self {
        let freq = if expert_count > 0 {
            1.0 / expert_count as f64
        } else {
            0.0
        };
        let expert_frequencies = (0..expert_count).map(|i| (i.to_string(), freq)).collect();
        ExpertFrequencyProfile { expert_frequencies }
    }
}

/// Result of fusing top-K experts into a single dense program.
#[derive(Debug)]
pub struct MoeFuseResult {
    /// The fused dense program — router and non-selected experts removed.
    pub program: Program,
    /// Indices of the experts that were kept (sorted by frequency, descending).
    pub kept_expert_indices: Vec<usize>,
    /// Indices of the experts that were discarded.
    pub discarded_expert_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct RouterInfo {
    op_indices: Vec<usize>,
    output_name: String,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Detect MoE architecture patterns in a MIL program.
///
/// Returns `Some(MoeTopology)` if a Mixture-of-Experts pattern is found,
/// `None` otherwise. Detection uses both name-based heuristics and
/// structural analysis of the operation graph.
pub fn detect_moe(program: &Program) -> Option<MoeTopology> {
    let func = program.main()?;
    let ops = &func.body.operations;
    if ops.len() < 4 {
        return None;
    }

    let output_map = build_output_map(ops);
    let consumer_map = build_consumer_map(ops);

    // Try name-based detection first (most reliable for known model formats)
    if let Some(topology) = detect_by_name(ops, &output_map, &consumer_map) {
        return Some(topology);
    }

    // Fall back to structural detection
    detect_by_structure(ops, &output_map, &consumer_map)
}

/// Split a MoE program into shared layers and per-expert programs.
///
/// The `topology` should come from [`detect_moe`]. Each expert becomes its
/// own [`Program`] with function inputs for values produced by shared layers,
/// and function outputs consumed downstream.
pub fn split_moe(program: &Program, topology: &MoeTopology) -> MoeSplitResult {
    let func = program.main().expect("program must have a main function");
    let ops = &func.body.operations;
    let version = &program.version;

    // Shared program: router + shared ops
    let mut shared_indices: Vec<usize> = topology.shared_op_indices.clone();
    shared_indices.extend(&topology.router_op_indices);
    shared_indices.sort();
    shared_indices.dedup();

    let shared = extract_subprogram(
        ops,
        &shared_indices,
        &func.inputs,
        &func.body.outputs,
        "main",
        version,
    );

    // Per-expert programs
    let experts: Vec<Program> = topology
        .expert_op_indices
        .iter()
        .enumerate()
        .map(|(i, indices)| {
            extract_subprogram(
                ops,
                indices,
                &func.inputs,
                &func.body.outputs,
                &format!("expert_{i}"),
                version,
            )
        })
        .collect();

    let manifest = build_manifest(topology, &shared, &experts);

    MoeSplitResult {
        shared,
        experts,
        manifest,
    }
}

/// Fuse the top-K most frequently activated experts into a single dense program.
///
/// Given a MoE [`Program`] and an [`ExpertFrequencyProfile`], this function:
///
/// 1. Detects MoE topology via [`detect_moe`].
/// 2. Selects the top-K experts by activation frequency.
/// 3. Builds a new dense program that keeps only shared ops plus the
///    selected expert ops, wired together with a weighted-average
///    combination layer (weights proportional to activation frequency).
/// 4. Removes router/gating ops since expert selection is now static.
///
/// Returns `None` if no MoE pattern is detected or if K is zero.
///
/// # Errors
///
/// Returns [`MilError::Validation`] if the frequency profile contains
/// non-finite values.
pub fn fuse_top_k_experts(
    program: &Program,
    k: usize,
    profile: &ExpertFrequencyProfile,
) -> Result<Option<MoeFuseResult>, MilError> {
    if k == 0 {
        return Ok(None);
    }

    let topology = match detect_moe(program) {
        Some(t) => t,
        None => return Ok(None),
    };
    let func = match program.main() {
        Some(f) => f,
        None => return Ok(None),
    };
    let ops = &func.body.operations;

    // Select top-K expert indices
    let mut selected = profile.top_k(k)?;
    // Clamp to available experts
    selected.retain(|&id| id < topology.expert_count);
    if selected.is_empty() {
        // Fallback: take the first K experts
        selected = (0..k.min(topology.expert_count)).collect();
    }

    let selected_set: HashSet<usize> = selected.iter().copied().collect();
    let discarded: Vec<usize> = (0..topology.expert_count)
        .filter(|i| !selected_set.contains(i))
        .collect();

    // Compute per-expert frequency weights for averaging, normalised to sum=1
    let raw_weights: Vec<f64> = selected
        .iter()
        .map(|&id| {
            profile
                .expert_frequencies
                .get(&id.to_string())
                .copied()
                .unwrap_or(1.0)
        })
        .collect();
    let weight_sum: f64 = raw_weights.iter().sum();
    let norm_weights: Vec<f64> = if weight_sum > 0.0 {
        raw_weights.iter().map(|w| w / weight_sum).collect()
    } else {
        vec![1.0 / selected.len() as f64; selected.len()]
    };

    // Collect operation indices to keep:
    //  - shared ops (excluding router ops)
    //  - selected expert ops
    let router_set: HashSet<usize> = topology.router_op_indices.iter().copied().collect();
    let mut keep_indices: Vec<usize> = topology
        .shared_op_indices
        .iter()
        .filter(|i| !router_set.contains(i))
        .copied()
        .collect();
    for &expert_id in &selected {
        keep_indices.extend(&topology.expert_op_indices[expert_id]);
    }
    keep_indices.sort();
    keep_indices.dedup();

    // Build a new block with the kept ops
    let mut block = Block::new();
    for &idx in &keep_indices {
        block.add_op(ops[idx].clone());
    }

    // Add combination ops: weighted average of selected expert outputs
    if selected.len() == 1 {
        // Single expert — its output is the fused output directly
        let expert_id = selected[0];
        let expert_output = topology.expert_output_names[expert_id].clone();
        block.outputs.push(expert_output);
    } else {
        // Multiple experts — weighted sum
        let mut prev_acc: Option<String> = None;
        for (i, &expert_id) in selected.iter().enumerate() {
            let expert_output = &topology.expert_output_names[expert_id];
            let weight = norm_weights[i];

            // Scale: mul(expert_output, weight)
            let scale_name = format!("fuse_scale_{expert_id}");
            let scale_out = format!("fuse_scaled_{expert_id}");
            let scale_op = Operation::new("const_mul", &scale_name)
                .with_input("x", Value::Reference(expert_output.clone()))
                .with_attr("scale", Value::Float(weight))
                .with_output(&scale_out);
            block.add_op(scale_op);

            if let Some(ref acc) = prev_acc {
                // Accumulate: add(prev, scaled)
                let add_name = format!("fuse_add_{expert_id}");
                let add_out = format!("fuse_acc_{expert_id}");
                let add_op = Operation::new("add", &add_name)
                    .with_input("x", Value::Reference(acc.clone()))
                    .with_input("y", Value::Reference(scale_out.clone()))
                    .with_output(&add_out);
                block.add_op(add_op);
                prev_acc = Some(add_out);
            } else {
                prev_acc = Some(scale_out);
            }
        }

        // Find any downstream ops that consumed expert outputs or the
        // combination output and see if they should be appended.  For now
        // we surface the fused accumulator as the block output.
        if let Some(acc) = prev_acc {
            block.outputs.push(acc);
        }
    }

    // Append any shared post-expert ops (e.g. lm_head) that consumed
    // the original combination output.  We detect these by looking for
    // shared ops that are NOT yet in keep_indices.
    let keep_set: HashSet<usize> = keep_indices.iter().copied().collect();
    let produced: HashSet<String> = block
        .operations
        .iter()
        .flat_map(|op| op.outputs.clone())
        .collect();

    // Find shared ops that depend on produced values but are not yet included
    let mut post_ops: Vec<usize> = Vec::new();
    for &idx in &topology.shared_op_indices {
        if keep_set.contains(&idx) || router_set.contains(&idx) {
            continue;
        }
        let refs = collect_references(&ops[idx]);
        if refs.iter().any(|r| produced.contains(r)) {
            post_ops.push(idx);
        }
    }

    if !post_ops.is_empty() {
        // We need to rewire: replace references to the old combination
        // output with our fused output.
        let fused_output = block.outputs.last().cloned().unwrap_or_default();

        // Collect the set of output names produced by ops already in the block.
        let all_kept_outputs: HashSet<String> = block
            .operations
            .iter()
            .flat_map(|op| op.outputs.clone())
            .collect();

        // Replace the block outputs with the final downstream ops' outputs
        block.outputs.clear();

        // Prepare rewired ops before adding them to the block
        let mut rewired_ops: Vec<Operation> = Vec::new();
        for &idx in &post_ops {
            let mut op = ops[idx].clone();
            let available: HashSet<&str> = all_kept_outputs.iter().map(|s| s.as_str()).collect();
            for value in op.inputs.values_mut() {
                rewire_value(value, &available, &fused_output);
            }
            rewired_ops.push(op);
        }
        for op in rewired_ops {
            block.add_op(op);
        }

        // Use original function outputs if they appear in the block
        let block_outputs: HashSet<String> = block
            .operations
            .iter()
            .flat_map(|op| op.outputs.clone())
            .collect();
        for out in &func.body.outputs {
            if block_outputs.contains(out) {
                block.outputs.push(out.clone());
            }
        }
        if block.outputs.is_empty() {
            // Fallback: last op's outputs
            if let Some(last) = block.operations.last() {
                block.outputs = last.outputs.clone();
            }
        }
    }

    // Build the fused function and program
    let mut fused_func = Function::new("main");
    fused_func.inputs = func.inputs.clone();
    fused_func.body = block;

    let mut fused_program = Program::new(&program.version);
    fused_program.add_function(fused_func);

    Ok(Some(MoeFuseResult {
        program: fused_program,
        kept_expert_indices: selected,
        discarded_expert_indices: discarded,
    }))
}

/// Rewire a [`Value`] reference: if it points to a name not in
/// `available`, replace it with `replacement`.
fn rewire_value(value: &mut Value, available: &HashSet<&str>, replacement: &str) {
    match value {
        Value::Reference(name) => {
            if !available.contains(name.as_str()) {
                *name = replacement.to_string();
            }
        }
        Value::List(items) => {
            for item in items {
                rewire_value(item, available, replacement);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Detection: name-based
// ---------------------------------------------------------------------------

/// Detect MoE by looking for "expert" / "gate" / "router" in operation names.
fn detect_by_name(
    ops: &[Operation],
    output_map: &HashMap<String, usize>,
    _consumer_map: &HashMap<String, Vec<usize>>,
) -> Option<MoeTopology> {
    let mut expert_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut router_indices = Vec::new();

    for (i, op) in ops.iter().enumerate() {
        let name_lower = op.name.to_lowercase();
        if let Some(expert_id) = extract_expert_id(&name_lower) {
            expert_groups.entry(expert_id).or_default().push(i);
        } else if name_lower.contains("gate") || name_lower.contains("router") {
            router_indices.push(i);
        }
    }

    if expert_groups.len() < 2 {
        return None;
    }

    // If no named router, try structural router detection
    if router_indices.is_empty() {
        if let Some(router) = find_router_pattern(ops, output_map) {
            router_indices = router.op_indices;
        }
    }

    // Sort expert groups by expert ID
    let mut sorted: Vec<(usize, Vec<usize>)> = expert_groups.into_iter().collect();
    sorted.sort_by_key(|(id, _)| *id);
    let expert_op_indices: Vec<Vec<usize>> = sorted.into_iter().map(|(_, v)| v).collect();
    let expert_count = expert_op_indices.len();

    // Everything else is shared
    let mut assigned: HashSet<usize> = HashSet::new();
    for group in &expert_op_indices {
        assigned.extend(group);
    }
    assigned.extend(&router_indices);
    let shared_op_indices: Vec<usize> = (0..ops.len()).filter(|i| !assigned.contains(i)).collect();

    let router_output = router_indices
        .last()
        .and_then(|&i| ops[i].outputs.first().cloned())
        .unwrap_or_default();

    let (expert_input_names, expert_output_names) = compute_expert_io(ops, &expert_op_indices);

    Some(MoeTopology {
        expert_count,
        router_op_indices: router_indices,
        expert_op_indices,
        shared_op_indices,
        router_output,
        expert_input_names,
        expert_output_names,
    })
}

// ---------------------------------------------------------------------------
// Detection: structure-based
// ---------------------------------------------------------------------------

/// Detect MoE by finding linear→softmax router + parallel linear/matmul paths.
fn detect_by_structure(
    ops: &[Operation],
    output_map: &HashMap<String, usize>,
    consumer_map: &HashMap<String, Vec<usize>>,
) -> Option<MoeTopology> {
    // A router is required to distinguish MoE from other parallel patterns
    let router = find_router_pattern(ops, output_map)?;
    let router_set: HashSet<usize> = router.op_indices.iter().copied().collect();

    // Collect linear/matmul ops not in the router
    let candidate_ops: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(i, op)| {
            (op.op_type == "linear" || op.op_type == "matmul") && !router_set.contains(i)
        })
        .map(|(i, _)| i)
        .collect();

    // Group by shared input reference
    let mut input_groups: HashMap<String, Vec<usize>> = HashMap::new();
    for &idx in &candidate_ops {
        for ref_name in collect_references(&ops[idx]) {
            input_groups.entry(ref_name).or_default().push(idx);
        }
    }

    // Find the largest parallel group (≥ 2 ops sharing an input, no mutual deps)
    let mut best_parallel: Option<Vec<usize>> = None;
    for group in input_groups.values() {
        if group.len() >= 2
            && are_parallel(group, ops, output_map)
            && best_parallel.as_ref().is_none_or(|b| group.len() > b.len())
        {
            best_parallel = Some(group.clone());
        }
    }

    let parallel_starts = best_parallel?;
    if parallel_starts.len() < 2 {
        return None;
    }

    // Trace forward from each parallel start to build expert subgraphs.
    // An op is included only if ALL its input references come from within the
    // traced set, preventing combination ops from leaking in.
    let expert_op_indices: Vec<Vec<usize>> = parallel_starts
        .iter()
        .map(|&start| trace_expert_subgraph(start, ops, consumer_map))
        .collect();

    let expert_count = expert_op_indices.len();

    let mut assigned: HashSet<usize> = router_set;
    for group in &expert_op_indices {
        assigned.extend(group);
    }
    let shared_op_indices: Vec<usize> = (0..ops.len()).filter(|i| !assigned.contains(i)).collect();

    let (expert_input_names, expert_output_names) = compute_expert_io(ops, &expert_op_indices);

    Some(MoeTopology {
        expert_count,
        router_op_indices: router.op_indices,
        expert_op_indices,
        shared_op_indices,
        router_output: router.output_name,
        expert_input_names,
        expert_output_names,
    })
}

// ---------------------------------------------------------------------------
// Graph analysis helpers
// ---------------------------------------------------------------------------

/// Map each output name to the index of the op that produces it.
fn build_output_map(ops: &[Operation]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for output in &op.outputs {
            map.insert(output.clone(), i);
        }
    }
    map
}

/// Map each output name to the indices of ops that consume it.
fn build_consumer_map(ops: &[Operation]) -> HashMap<String, Vec<usize>> {
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for ref_name in collect_references(op) {
            map.entry(ref_name).or_default().push(i);
        }
    }
    map
}

/// Collect all `Value::Reference` names from an operation's inputs.
fn collect_references(op: &Operation) -> Vec<String> {
    let mut refs = Vec::new();
    for value in op.inputs.values() {
        collect_refs_value(value, &mut refs);
    }
    refs
}

fn collect_refs_value(value: &Value, refs: &mut Vec<String>) {
    match value {
        Value::Reference(name) => refs.push(name.clone()),
        Value::List(items) => {
            for item in items {
                collect_refs_value(item, refs);
            }
        }
        _ => {}
    }
}

/// Extract expert ID from an operation name.
///
/// Matches patterns like `"expert_0"`, `"experts.1"`, `"expert-2"`, `"expert3"`.
fn extract_expert_id(name: &str) -> Option<usize> {
    let pos = name.find("expert")?;
    let after = &name[pos + 6..];

    // Skip optional 's' (for "experts")
    let after = after.strip_prefix('s').unwrap_or(after);

    // Skip separator (., _, -, or nothing)
    let after = after
        .strip_prefix('.')
        .or_else(|| after.strip_prefix('_'))
        .or_else(|| after.strip_prefix('-'))
        .unwrap_or(after);

    let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

/// Find a router pattern: linear/matmul → softmax (optionally → topk).
fn find_router_pattern(
    ops: &[Operation],
    output_map: &HashMap<String, usize>,
) -> Option<RouterInfo> {
    // Primary: look for softmax preceded by linear/matmul
    for (i, op) in ops.iter().enumerate() {
        if op.op_type != "softmax" {
            continue;
        }
        for ref_name in collect_references(op) {
            if let Some(&pred_idx) = output_map.get(&ref_name) {
                let pred = &ops[pred_idx];
                if pred.op_type == "linear" || pred.op_type == "matmul" {
                    let output_name = op.outputs.first()?.clone();
                    return Some(RouterInfo {
                        op_indices: vec![pred_idx, i],
                        output_name,
                    });
                }
            }
        }
    }

    // Secondary: topk preceded by softmax preceded by linear/matmul
    for (i, op) in ops.iter().enumerate() {
        if op.op_type != "topk" {
            continue;
        }
        for ref_name in collect_references(op) {
            if let Some(&softmax_idx) = output_map.get(&ref_name) {
                if ops[softmax_idx].op_type != "softmax" {
                    continue;
                }
                for sr in collect_references(&ops[softmax_idx]) {
                    if let Some(&gate_idx) = output_map.get(&sr) {
                        let gate = &ops[gate_idx];
                        if gate.op_type == "linear" || gate.op_type == "matmul" {
                            let output_name = op.outputs.first()?.clone();
                            return Some(RouterInfo {
                                op_indices: vec![gate_idx, softmax_idx, i],
                                output_name,
                            });
                        }
                    }
                }
            }
        }
    }

    None
}

/// Check if a group of ops are parallel (no mutual data dependencies).
fn are_parallel(group: &[usize], ops: &[Operation], output_map: &HashMap<String, usize>) -> bool {
    let group_set: HashSet<usize> = group.iter().copied().collect();
    for &idx in group {
        for ref_name in collect_references(&ops[idx]) {
            if let Some(&producer) = output_map.get(&ref_name) {
                if group_set.contains(&producer) {
                    return false;
                }
            }
        }
    }
    true
}

/// Trace forward from a start op to build an expert subgraph.
///
/// Includes an op only if ALL of its input references resolve to outputs
/// already within the traced set — this prevents combination ops (which also
/// consume from the router or other experts) from leaking in.
fn trace_expert_subgraph(
    start: usize,
    ops: &[Operation],
    consumer_map: &HashMap<String, Vec<usize>>,
) -> Vec<usize> {
    let mut result = vec![start];
    let mut result_outputs: HashSet<String> = ops[start].outputs.iter().cloned().collect();
    let mut visited: HashSet<usize> = HashSet::from([start]);
    let mut queue: Vec<usize> = Vec::new();

    for output in &ops[start].outputs {
        if let Some(consumers) = consumer_map.get(output) {
            queue.extend(consumers.iter().copied());
        }
    }

    while let Some(idx) = queue.pop() {
        if !visited.insert(idx) {
            continue;
        }

        let refs = collect_references(&ops[idx]);
        let all_from_trace = !refs.is_empty() && refs.iter().all(|r| result_outputs.contains(r));
        if !all_from_trace {
            continue;
        }

        result.push(idx);
        result_outputs.extend(ops[idx].outputs.iter().cloned());

        for output in &ops[idx].outputs {
            if let Some(consumers) = consumer_map.get(output) {
                for &c in consumers {
                    if !visited.contains(&c) {
                        queue.push(c);
                    }
                }
            }
        }
    }

    result.sort();
    result
}

/// Compute input and output tensor names for each expert group.
fn compute_expert_io(
    ops: &[Operation],
    expert_groups: &[Vec<usize>],
) -> (Vec<String>, Vec<String>) {
    let mut input_names = Vec::new();
    let mut output_names = Vec::new();

    for group in expert_groups {
        let group_outputs: HashSet<String> =
            group.iter().flat_map(|&i| ops[i].outputs.clone()).collect();

        // Inputs: references not produced within this group
        let mut inputs: Vec<String> = Vec::new();
        for &idx in group {
            for ref_name in collect_references(&ops[idx]) {
                if !group_outputs.contains(&ref_name) && !inputs.contains(&ref_name) {
                    inputs.push(ref_name);
                }
            }
        }
        input_names.push(inputs.first().cloned().unwrap_or_default());

        // Output: last op's first output
        let last_output = group
            .last()
            .and_then(|&i| ops[i].outputs.first().cloned())
            .unwrap_or_default();
        output_names.push(last_output);
    }

    (input_names, output_names)
}

// ---------------------------------------------------------------------------
// Splitting
// ---------------------------------------------------------------------------

/// Extract a subprogram containing only the ops at the given indices.
fn extract_subprogram(
    ops: &[Operation],
    indices: &[usize],
    original_inputs: &[(String, TensorType)],
    original_outputs: &[String],
    func_name: &str,
    version: &str,
) -> Program {
    // Outputs produced by ops in this subset
    let produced: HashSet<String> = indices
        .iter()
        .flat_map(|&i| ops[i].outputs.clone())
        .collect();

    // References needed by ops in this subset but not produced within it
    let mut needed: Vec<String> = Vec::new();
    for &idx in indices {
        for ref_name in collect_references(&ops[idx]) {
            if !produced.contains(&ref_name) && !needed.contains(&ref_name) {
                needed.push(ref_name);
            }
        }
    }

    // Function inputs: original inputs that are referenced + cross-partition refs
    let mut func_inputs: Vec<(String, TensorType)> = Vec::new();
    for ref_name in &needed {
        if let Some((_, ty)) = original_inputs.iter().find(|(n, _)| n == ref_name) {
            func_inputs.push((ref_name.clone(), ty.clone()));
        } else {
            // Cross-partition input — dynamic shape placeholder
            let placeholder = TensorType::with_dynamic_shape(ScalarType::Float32, vec![None]);
            func_inputs.push((ref_name.clone(), placeholder));
        }
    }

    // Block with selected ops in original order
    let mut block = Block::new();
    for &idx in indices {
        block.add_op(ops[idx].clone());
    }

    // Outputs: ops whose outputs are in the original function outputs
    let original_out_set: HashSet<&str> = original_outputs.iter().map(|s| s.as_str()).collect();
    for &idx in indices {
        for output in &ops[idx].outputs {
            if original_out_set.contains(output.as_str()) && !block.outputs.contains(output) {
                block.outputs.push(output.clone());
            }
        }
    }
    // Fallback: use last op's outputs
    if block.outputs.is_empty() {
        if let Some(&last_idx) = indices.last() {
            block.outputs = ops[last_idx].outputs.clone();
        }
    }

    let mut func = Function::new(func_name);
    func.inputs = func_inputs;
    func.body = block;

    let mut program = Program::new(version);
    program.add_function(func);
    program
}

/// Build the MoE manifest from topology and split programs.
///
/// The shared program is monolithic — it contains both the router and any
/// combination logic.  It runs once with the model inputs and produces router
/// weights as its outputs.  The runtime is responsible for dispatching expert
/// inputs based on those weights and combining expert outputs; there is no
/// separate "combination" artifact.
fn build_manifest(topology: &MoeTopology, shared: &Program, experts: &[Program]) -> MoeManifest {
    let shared_func = shared.main().expect("shared program has main");
    let shared_inputs: Vec<String> = shared_func.inputs.iter().map(|(n, _)| n.clone()).collect();
    let shared_outputs = shared_func.body.outputs.clone();

    let expert_descriptors: Vec<ExpertDescriptor> = experts
        .iter()
        .enumerate()
        .map(|(i, prog)| {
            let func = prog.main().expect("expert program has main");
            ExpertDescriptor {
                index: i,
                inputs: func.inputs.iter().map(|(n, _)| n.clone()).collect(),
                outputs: func.body.outputs.clone(),
            }
        })
        .collect();

    // Stage 1: shared program (router + combination logic) runs once with
    // model inputs and produces router weights as intermediate outputs.
    let mut stages = vec![Stage {
        name: "shared".into(),
        artifact: "shared".into(),
        inputs: shared_inputs,
        outputs: shared_outputs,
    }];

    // Stage 2..N: each expert runs with dispatched inputs.
    for (i, desc) in expert_descriptors.iter().enumerate() {
        stages.push(Stage {
            name: format!("expert-{i}"),
            artifact: format!("expert-{i}"),
            inputs: desc.inputs.clone(),
            outputs: desc.outputs.clone(),
        });
    }

    MoeManifest {
        expert_count: topology.expert_count,
        router_output: topology.router_output.clone(),
        experts: expert_descriptors,
        stages,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a mock MoE program with named expert ops.
    fn make_named_moe_program() -> Program {
        let mut program = Program::new("1.0.0");
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 512]);
        let mut func = Function::new("main").with_input("hidden", input_ty);

        // Shared: embedding
        func.body.add_op(
            Operation::new("linear", "embedding")
                .with_input("x", Value::Reference("hidden".into()))
                .with_output("embed_out"),
        );

        // Router
        func.body.add_op(
            Operation::new("linear", "router_gate")
                .with_input("x", Value::Reference("embed_out".into()))
                .with_output("gate_logits"),
        );
        func.body.add_op(
            Operation::new("softmax", "router_softmax")
                .with_input("x", Value::Reference("gate_logits".into()))
                .with_output("gate_weights"),
        );

        // Expert 0: linear → relu → linear
        func.body.add_op(
            Operation::new("linear", "expert_0_w1")
                .with_input("x", Value::Reference("embed_out".into()))
                .with_output("expert_0_h"),
        );
        func.body.add_op(
            Operation::new("relu", "expert_0_act")
                .with_input("x", Value::Reference("expert_0_h".into()))
                .with_output("expert_0_a"),
        );
        func.body.add_op(
            Operation::new("linear", "expert_0_w2")
                .with_input("x", Value::Reference("expert_0_a".into()))
                .with_output("expert_0_out"),
        );

        // Expert 1: linear → relu → linear
        func.body.add_op(
            Operation::new("linear", "expert_1_w1")
                .with_input("x", Value::Reference("embed_out".into()))
                .with_output("expert_1_h"),
        );
        func.body.add_op(
            Operation::new("relu", "expert_1_act")
                .with_input("x", Value::Reference("expert_1_h".into()))
                .with_output("expert_1_a"),
        );
        func.body.add_op(
            Operation::new("linear", "expert_1_w2")
                .with_input("x", Value::Reference("expert_1_a".into()))
                .with_output("expert_1_out"),
        );

        // Combination
        func.body.add_op(
            Operation::new("mul", "weight_0")
                .with_input("x", Value::Reference("expert_0_out".into()))
                .with_input("y", Value::Reference("gate_weights".into()))
                .with_output("weighted_0"),
        );
        func.body.add_op(
            Operation::new("mul", "weight_1")
                .with_input("x", Value::Reference("expert_1_out".into()))
                .with_input("y", Value::Reference("gate_weights".into()))
                .with_output("weighted_1"),
        );
        func.body.add_op(
            Operation::new("add", "combine")
                .with_input("x", Value::Reference("weighted_0".into()))
                .with_input("y", Value::Reference("weighted_1".into()))
                .with_output("moe_out"),
        );

        // Shared: output head
        func.body.add_op(
            Operation::new("linear", "lm_head")
                .with_input("x", Value::Reference("moe_out".into()))
                .with_output("logits"),
        );

        func.body.outputs.push("logits".into());
        program.add_function(func);
        program
    }

    /// Build a mock MoE program using only structural patterns (no expert names).
    fn make_structural_moe_program() -> Program {
        let mut program = Program::new("1.0.0");
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 512]);
        let mut func = Function::new("main").with_input("hidden", input_ty);

        // Shared: layer norm
        func.body.add_op(
            Operation::new("layer_norm", "norm_0")
                .with_input("x", Value::Reference("hidden".into()))
                .with_output("normed"),
        );

        // Router: linear → softmax
        func.body.add_op(
            Operation::new("linear", "gate_proj")
                .with_input("x", Value::Reference("normed".into()))
                .with_output("gate_logits"),
        );
        func.body.add_op(
            Operation::new("softmax", "gate_softmax")
                .with_input("x", Value::Reference("gate_logits".into()))
                .with_output("gate_probs"),
        );

        // Parallel path A: linear → relu → linear (same input "normed")
        func.body.add_op(
            Operation::new("linear", "ffn_a_up")
                .with_input("x", Value::Reference("normed".into()))
                .with_output("a_up"),
        );
        func.body.add_op(
            Operation::new("relu", "ffn_a_act")
                .with_input("x", Value::Reference("a_up".into()))
                .with_output("a_act"),
        );
        func.body.add_op(
            Operation::new("linear", "ffn_a_down")
                .with_input("x", Value::Reference("a_act".into()))
                .with_output("a_out"),
        );

        // Parallel path B: linear → relu → linear (same input "normed")
        func.body.add_op(
            Operation::new("linear", "ffn_b_up")
                .with_input("x", Value::Reference("normed".into()))
                .with_output("b_up"),
        );
        func.body.add_op(
            Operation::new("relu", "ffn_b_act")
                .with_input("x", Value::Reference("b_up".into()))
                .with_output("b_act"),
        );
        func.body.add_op(
            Operation::new("linear", "ffn_b_down")
                .with_input("x", Value::Reference("b_act".into()))
                .with_output("b_out"),
        );

        // Combination
        func.body.add_op(
            Operation::new("add", "merge")
                .with_input("x", Value::Reference("a_out".into()))
                .with_input("y", Value::Reference("b_out".into()))
                .with_output("merged"),
        );

        func.body.outputs.push("merged".into());
        program.add_function(func);
        program
    }

    // -- extract_expert_id ------------------------------------------------

    #[test]
    fn expert_id_underscore() {
        assert_eq!(extract_expert_id("expert_0_w1"), Some(0));
    }

    #[test]
    fn expert_id_dot() {
        assert_eq!(extract_expert_id("experts.1.fc"), Some(1));
    }

    #[test]
    fn expert_id_dash() {
        assert_eq!(extract_expert_id("block_expert-2"), Some(2));
    }

    #[test]
    fn expert_id_no_separator() {
        assert_eq!(extract_expert_id("expert3_linear"), Some(3));
    }

    #[test]
    fn expert_id_large() {
        assert_eq!(extract_expert_id("layer.experts.15.down"), Some(15));
    }

    #[test]
    fn expert_id_no_match() {
        assert_eq!(extract_expert_id("no_match_here"), None);
    }

    #[test]
    fn expert_id_bare_expert() {
        assert_eq!(extract_expert_id("expert"), None);
    }

    #[test]
    fn expert_id_expert_text_only() {
        assert_eq!(extract_expert_id("expert_gate"), None);
    }

    // -- detect_moe (name-based) ------------------------------------------

    #[test]
    fn detect_moe_by_name() {
        let program = make_named_moe_program();
        let topo = detect_moe(&program).expect("should detect MoE");

        assert_eq!(topo.expert_count, 2);
        assert_eq!(topo.expert_op_indices.len(), 2);
        // Each expert has 3 ops: w1, act, w2
        assert_eq!(topo.expert_op_indices[0].len(), 3);
        assert_eq!(topo.expert_op_indices[1].len(), 3);
        // Router indices are non-empty
        assert!(!topo.router_op_indices.is_empty());
        // Router output should be the softmax output
        assert_eq!(topo.router_output, "gate_weights");
    }

    #[test]
    fn detect_moe_shared_ops_exclude_experts_and_router() {
        let program = make_named_moe_program();
        let topo = detect_moe(&program).unwrap();

        let all_ops = program.main().unwrap().body.operations.len();
        let assigned = topo.shared_op_indices.len()
            + topo.router_op_indices.len()
            + topo
                .expert_op_indices
                .iter()
                .map(|g| g.len())
                .sum::<usize>();
        assert_eq!(assigned, all_ops);
    }

    // -- detect_moe (structure-based) -------------------------------------

    #[test]
    fn detect_moe_by_structure() {
        let program = make_structural_moe_program();
        let topo = detect_moe(&program).expect("should detect MoE structurally");

        assert_eq!(topo.expert_count, 2);
        assert_eq!(topo.expert_op_indices.len(), 2);
        // Each path: up_linear, activation, down_linear = 3 ops
        assert_eq!(topo.expert_op_indices[0].len(), 3);
        assert_eq!(topo.expert_op_indices[1].len(), 3);
    }

    // -- no MoE -----------------------------------------------------------

    #[test]
    fn no_moe_in_simple_model() {
        let mut program = Program::new("1.0.0");
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);
        let mut func = Function::new("main").with_input("image", input_ty);

        func.body.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("image".into()))
                .with_output("conv_out"),
        );
        func.body.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_output("relu_out"),
        );
        func.body.outputs.push("relu_out".into());
        program.add_function(func);

        assert!(detect_moe(&program).is_none());
    }

    #[test]
    fn no_moe_in_empty_program() {
        let program = Program::new("1.0.0");
        assert!(detect_moe(&program).is_none());
    }

    // -- split_moe --------------------------------------------------------

    #[test]
    fn split_produces_correct_program_count() {
        let program = make_named_moe_program();
        let topo = detect_moe(&program).unwrap();
        let result = split_moe(&program, &topo);

        assert_eq!(result.experts.len(), 2);
        assert!(result.shared.main().is_some());
    }

    #[test]
    fn split_expert_programs_have_correct_ops() {
        let program = make_named_moe_program();
        let topo = detect_moe(&program).unwrap();
        let result = split_moe(&program, &topo);

        for (i, expert) in result.experts.iter().enumerate() {
            let func = expert.main().expect("expert should have main");
            assert_eq!(
                func.body.operations.len(),
                3,
                "expert {i} should have 3 ops"
            );
            assert_eq!(func.body.operations[0].op_type, "linear");
            assert_eq!(func.body.operations[1].op_type, "relu");
            assert_eq!(func.body.operations[2].op_type, "linear");
        }
    }

    #[test]
    fn split_shared_program_excludes_experts() {
        let program = make_named_moe_program();
        let topo = detect_moe(&program).unwrap();
        let result = split_moe(&program, &topo);

        let shared_func = result.shared.main().unwrap();
        for op in &shared_func.body.operations {
            assert!(
                extract_expert_id(&op.name.to_lowercase()).is_none(),
                "shared program should not contain expert op: {}",
                op.name
            );
        }
    }

    #[test]
    fn split_expert_programs_have_inputs_for_cross_partition_refs() {
        let program = make_named_moe_program();
        let topo = detect_moe(&program).unwrap();
        let result = split_moe(&program, &topo);

        for expert in &result.experts {
            let func = expert.main().unwrap();
            // Expert needs embed_out as input (cross-partition)
            assert!(
                func.inputs.iter().any(|(n, _)| n == "embed_out"),
                "expert should have embed_out as input"
            );
        }
    }

    // -- manifest ---------------------------------------------------------

    #[test]
    fn manifest_has_correct_structure() {
        let program = make_named_moe_program();
        let topo = detect_moe(&program).unwrap();
        let result = split_moe(&program, &topo);

        assert_eq!(result.manifest.expert_count, 2);
        assert_eq!(result.manifest.router_output, "gate_weights");
        assert_eq!(result.manifest.experts.len(), 2);
        assert_eq!(result.manifest.experts[0].index, 0);
        assert_eq!(result.manifest.experts[1].index, 1);
    }

    #[test]
    fn manifest_stages_order() {
        let program = make_named_moe_program();
        let topo = detect_moe(&program).unwrap();
        let result = split_moe(&program, &topo);

        // shared, expert-0, expert-1 (no separate combination stage)
        assert_eq!(result.manifest.stages.len(), 3);
        assert_eq!(result.manifest.stages[0].name, "shared");
        assert_eq!(result.manifest.stages[0].artifact, "shared");
        assert!(!result.manifest.stages[0].outputs.is_empty());
        assert_eq!(result.manifest.stages[1].name, "expert-0");
        assert_eq!(result.manifest.stages[2].name, "expert-1");
    }

    #[test]
    fn manifest_serializes_to_json() {
        let program = make_named_moe_program();
        let topo = detect_moe(&program).unwrap();
        let result = split_moe(&program, &topo);

        let json = serde_json::to_string_pretty(&result.manifest).expect("should serialize");
        assert!(json.contains("\"expert_count\": 2"));
        assert!(json.contains("\"router_output\""));
        assert!(json.contains("\"shared\""));
    }

    // -- ExpertFrequencyProfile -------------------------------------------

    #[test]
    fn profile_from_json() {
        let json = r#"{"expert_frequencies": {"0": 0.45, "1": 0.30, "2": 0.15, "3": 0.10}}"#;
        let profile = ExpertFrequencyProfile::from_json(json).unwrap();
        assert_eq!(profile.expert_frequencies.len(), 4);
        assert!((profile.expert_frequencies["0"] - 0.45).abs() < 1e-9);
    }

    #[test]
    fn profile_top_k_ordering() {
        let json = r#"{"expert_frequencies": {"0": 0.10, "1": 0.45, "2": 0.30, "3": 0.15}}"#;
        let profile = ExpertFrequencyProfile::from_json(json).unwrap();
        let top2 = profile.top_k(2).unwrap();
        assert_eq!(top2, vec![1, 2]);
    }

    #[test]
    fn profile_top_k_clamps_to_available() {
        let json = r#"{"expert_frequencies": {"0": 0.5, "1": 0.5}}"#;
        let profile = ExpertFrequencyProfile::from_json(json).unwrap();
        let top5 = profile.top_k(5).unwrap();
        assert_eq!(top5.len(), 2);
    }

    #[test]
    fn profile_top_k_rejects_nan() {
        let mut profile = ExpertFrequencyProfile::uniform(2);
        profile.expert_frequencies.insert("0".into(), f64::NAN);
        assert!(profile.top_k(2).is_err());
    }

    #[test]
    fn profile_top_k_rejects_infinity() {
        let mut profile = ExpertFrequencyProfile::uniform(2);
        profile.expert_frequencies.insert("1".into(), f64::INFINITY);
        assert!(profile.top_k(2).is_err());
    }

    #[test]
    fn profile_uniform() {
        let profile = ExpertFrequencyProfile::uniform(4);
        assert_eq!(profile.expert_frequencies.len(), 4);
        for v in profile.expert_frequencies.values() {
            assert!((v - 0.25).abs() < 1e-9);
        }
    }

    // -- fuse_top_k_experts -----------------------------------------------

    /// Build a 4-expert MoE program for fusion tests.
    fn make_four_expert_program() -> Program {
        let mut program = Program::new("1.0.0");
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 512]);
        let mut func = Function::new("main").with_input("hidden", input_ty);

        // Shared: embedding
        func.body.add_op(
            Operation::new("linear", "embedding")
                .with_input("x", Value::Reference("hidden".into()))
                .with_output("embed_out"),
        );

        // Router
        func.body.add_op(
            Operation::new("linear", "router_gate")
                .with_input("x", Value::Reference("embed_out".into()))
                .with_output("gate_logits"),
        );
        func.body.add_op(
            Operation::new("softmax", "router_softmax")
                .with_input("x", Value::Reference("gate_logits".into()))
                .with_output("gate_weights"),
        );

        // 4 experts: each is linear → relu → linear
        for i in 0..4 {
            func.body.add_op(
                Operation::new("linear", &format!("expert_{i}_w1"))
                    .with_input("x", Value::Reference("embed_out".into()))
                    .with_output(&format!("expert_{i}_h")),
            );
            func.body.add_op(
                Operation::new("relu", &format!("expert_{i}_act"))
                    .with_input("x", Value::Reference(format!("expert_{i}_h")))
                    .with_output(&format!("expert_{i}_a")),
            );
            func.body.add_op(
                Operation::new("linear", &format!("expert_{i}_w2"))
                    .with_input("x", Value::Reference(format!("expert_{i}_a")))
                    .with_output(&format!("expert_{i}_out")),
            );
        }

        // Combination (simplified: just add all)
        func.body.add_op(
            Operation::new("add", "combine_01")
                .with_input("x", Value::Reference("expert_0_out".into()))
                .with_input("y", Value::Reference("expert_1_out".into()))
                .with_output("sum_01"),
        );
        func.body.add_op(
            Operation::new("add", "combine_23")
                .with_input("x", Value::Reference("expert_2_out".into()))
                .with_input("y", Value::Reference("expert_3_out".into()))
                .with_output("sum_23"),
        );
        func.body.add_op(
            Operation::new("add", "combine_all")
                .with_input("x", Value::Reference("sum_01".into()))
                .with_input("y", Value::Reference("sum_23".into()))
                .with_output("moe_out"),
        );

        // Output head
        func.body.add_op(
            Operation::new("linear", "lm_head")
                .with_input("x", Value::Reference("moe_out".into()))
                .with_output("logits"),
        );

        func.body.outputs.push("logits".into());
        program.add_function(func);
        program
    }

    #[test]
    fn fuse_top_2_from_4_experts() {
        let program = make_four_expert_program();
        let profile_json =
            r#"{"expert_frequencies": {"0": 0.10, "1": 0.45, "2": 0.30, "3": 0.15}}"#;
        let profile = ExpertFrequencyProfile::from_json(profile_json).unwrap();

        let result = fuse_top_k_experts(&program, 2, &profile)
            .expect("no error")
            .expect("should fuse");
        assert_eq!(result.kept_expert_indices, vec![1, 2]);
        assert_eq!(result.discarded_expert_indices, vec![0, 3]);

        // Fused program should have a main function
        let fused_func = result.program.main().expect("should have main");

        // Should NOT contain router ops
        for op in &fused_func.body.operations {
            assert_ne!(op.op_type, "softmax", "router softmax should be removed");
            let name_lower = op.name.to_lowercase();
            assert!(
                !name_lower.contains("router"),
                "router ops should be removed: {}",
                op.name
            );
        }

        // Should NOT contain discarded experts
        for op in &fused_func.body.operations {
            let name_lower = op.name.to_lowercase();
            assert!(
                !name_lower.contains("expert_0"),
                "expert 0 should be discarded: {}",
                op.name
            );
            assert!(
                !name_lower.contains("expert_3"),
                "expert 3 should be discarded: {}",
                op.name
            );
        }

        // Should contain kept expert ops
        let op_names: Vec<&str> = fused_func
            .body
            .operations
            .iter()
            .map(|op| op.name.as_str())
            .collect();
        assert!(op_names.contains(&"expert_1_w1"));
        assert!(op_names.contains(&"expert_2_w1"));

        // Should have fusion combination ops
        let has_fuse_ops = op_names.iter().any(|n| n.starts_with("fuse_"));
        assert!(has_fuse_ops, "should have fuse_ combination ops");
    }

    #[test]
    fn fuse_single_expert() {
        let program = make_named_moe_program();
        let profile_json = r#"{"expert_frequencies": {"0": 0.7, "1": 0.3}}"#;
        let profile = ExpertFrequencyProfile::from_json(profile_json).unwrap();

        let result = fuse_top_k_experts(&program, 1, &profile)
            .expect("no error")
            .expect("should fuse");
        assert_eq!(result.discarded_expert_indices, vec![1]);

        let fused_func = result.program.main().unwrap();
        // Single expert — no fuse_ combination ops needed
        let has_fuse_scale = fused_func
            .body
            .operations
            .iter()
            .any(|op| op.name.starts_with("fuse_scale"));
        assert!(!has_fuse_scale, "single expert should not need scaling ops");
    }

    #[test]
    fn fuse_returns_none_for_non_moe() {
        let mut program = Program::new("1.0.0");
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 10]);
        let mut func = Function::new("main").with_input("x", input_ty);
        func.body.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("x".into()))
                .with_output("out"),
        );
        func.body.outputs.push("out".into());
        program.add_function(func);

        let profile = ExpertFrequencyProfile::uniform(2);
        assert!(fuse_top_k_experts(&program, 1, &profile).unwrap().is_none());
    }

    #[test]
    fn fuse_returns_none_for_k_zero() {
        let program = make_named_moe_program();
        let profile = ExpertFrequencyProfile::uniform(2);
        assert!(fuse_top_k_experts(&program, 0, &profile).unwrap().is_none());
    }

    #[test]
    fn fuse_with_uniform_profile() {
        let program = make_named_moe_program();
        let profile = ExpertFrequencyProfile::uniform(2);

        let result = fuse_top_k_experts(&program, 2, &profile)
            .expect("no error")
            .expect("should fuse");
        // Both experts kept
        assert_eq!(result.kept_expert_indices.len(), 2);
        assert!(result.discarded_expert_indices.is_empty());
    }

    #[test]
    fn fuse_preserves_shared_ops() {
        let program = make_named_moe_program();
        let profile_json = r#"{"expert_frequencies": {"0": 0.6, "1": 0.4}}"#;
        let profile = ExpertFrequencyProfile::from_json(profile_json).unwrap();

        let result = fuse_top_k_experts(&program, 1, &profile)
            .expect("no error")
            .expect("should fuse");
        let fused_func = result.program.main().unwrap();

        // The shared embedding op should still be present
        let has_embedding = fused_func
            .body
            .operations
            .iter()
            .any(|op| op.name == "embedding");
        assert!(has_embedding, "shared embedding op should be preserved");
    }
}
