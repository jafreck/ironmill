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
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            max_weight_size: 64 * 1024 * 1024, // 64 MB
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
        let sp = build_sub_program(name, ops, &type_map);
        sub_programs.push(sp);
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

/// Extract the first numeric component from an operation name.
///
/// Matches patterns like `layer_0_attn_q`, `layers.3.ffn.up`, `block_12_norm`.
fn extract_layer_number(op_name: &str) -> Option<usize> {
    for part in op_name.split(['_', '.']) {
        if let Ok(n) = part.parse::<usize>() {
            return Some(n);
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

    let mut groups: Vec<(String, Vec<Operation>)> = Vec::new();
    let mut pre_ops: Vec<Operation> = Vec::new();
    let mut layer_ops: BTreeMap<usize, Vec<Operation>> = BTreeMap::new();
    let mut post_ops: Vec<Operation> = Vec::new();

    // Track whether we've seen the last layer yet, so trailing non-layer ops
    // go into "post".
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
                    // Find the most recent layer number.
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
// Sub-program construction
// ---------------------------------------------------------------------------

/// Build a standalone [`SubProgram`] from a group of operations.
fn build_sub_program(
    name: &str,
    ops: &[Operation],
    type_map: &HashMap<String, TensorType>,
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

    // Outputs = last operation's outputs (simplified heuristic).
    let output_names: Vec<String> = ops.last().map(|op| op.outputs.clone()).unwrap_or_default();

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
                result.push(build_sub_program(&chunk_name, &current_ops, &type_map));
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
            result.push(build_sub_program(&chunk_name, &current_ops, &type_map));
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
        };
        let split = split_for_ane(&prog, &config).unwrap();

        // The single layer_0 (80 MB total) should be chunked into 2.
        assert!(
            split.programs.len() >= 2,
            "expected chunking, got {} sub-programs",
            split.programs.len()
        );
    }
}
