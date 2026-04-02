//! Operator splitting pass for large models targeting the Apple Neural Engine.
//!
//! ANE has limited on-chip memory (~1 GB on iOS, ~2 GB on macOS). When a
//! single `matmul`, `linear`, or `conv` operation exceeds the memory budget
//! the pass decomposes it into smaller ANE-sized tiles and concatenates the
//! results.
//!
//! **MatMul / Linear splitting**
//!
//! A matmul `[M, N] × [N, K]` is tiled along the `K` dimension into `T`
//! tiles of `[M, N] × [N, K/T]`, each producing `[M, K/T]`. The tiles are
//! concatenated along the last axis to recover `[M, K]`.
//!
//! **Conv splitting**
//!
//! Convolutions are split along the output-channel dimension (`C_out`).
//! Each tile produces a slice of output feature maps and results are
//! concatenated along the channel axis.
//!
//! **Multi-head attention splitting**
//!
//! When the pass detects a multi-head attention pattern (matmul with an
//! `n_heads` attribute or a known head dimension), it splits across heads
//! so that each head fits within the ANE budget.

use std::collections::HashMap;

use mil_rs::error::{MilError, Result};
use mil_rs::ir::Operation;
use mil_rs::ir::Pass;
use mil_rs::ir::Value;
use mil_rs::ir::{Block, Program};
use mil_rs::ir::{ScalarType, TensorType};

use mil_rs::ir::passes::replace_reference;

/// Default ANE memory budget: 1 GB (iOS).
pub const DEFAULT_MEMORY_BUDGET: usize = 1_073_741_824;

/// Decompose large matmul/linear/conv operations into ANE-sized tiles.
pub struct OpSplittingPass {
    /// Maximum memory (in bytes) a single operation may consume before
    /// it is split into tiles.
    pub memory_budget_bytes: usize,
}

impl OpSplittingPass {
    /// Create a pass with the given per-op memory budget in bytes.
    pub fn new(memory_budget_bytes: usize) -> Self {
        Self {
            memory_budget_bytes,
        }
    }
}

impl Default for OpSplittingPass {
    fn default() -> Self {
        Self::new(DEFAULT_MEMORY_BUDGET)
    }
}

impl Pass for OpSplittingPass {
    fn name(&self) -> &str {
        "op-splitting"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        for func in program.functions.values_mut() {
            let type_map = build_type_map(&func.body);
            split_block(&mut func.body, self.memory_budget_bytes, &type_map)?;
        }
        Ok(())
    }
}

// ── Memory estimation ─────────────────────────────────────────────────

/// Bytes per element for a [`ScalarType`].
fn dtype_bytes(dtype: ScalarType) -> usize {
    match dtype {
        ScalarType::Float64 | ScalarType::Int64 | ScalarType::UInt64 => 8,
        ScalarType::Float32 | ScalarType::Int32 | ScalarType::UInt32 => 4,
        ScalarType::Float16 | ScalarType::Int16 | ScalarType::UInt16 => 2,
        ScalarType::Int8 | ScalarType::UInt8 | ScalarType::Bool => 1,
    }
}

/// Total elements for a fully-static shape.  Returns `None` when any
/// dimension is dynamic.
fn static_elements(shape: &[Option<usize>]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, d| d.map(|v| acc.saturating_mul(v)))
}

/// Estimate the memory footprint of a single operation (weights + activations).
fn estimate_op_memory(op: &Operation, type_map: &HashMap<String, TensorType>) -> usize {
    match op.op_type.as_str() {
        "matmul" | "linear" => estimate_matmul_memory(op, type_map),
        "conv" | "conv_transpose" => estimate_conv_memory(op, type_map),
        _ => 0,
    }
}

/// Estimate memory for matmul/linear: weight tensor + input activation + output activation.
fn estimate_matmul_memory(op: &Operation, type_map: &HashMap<String, TensorType>) -> usize {
    let mut total: usize = 0;

    // Weight tensor (may be an inline const or a reference to a const op).
    let weight_key = if op.op_type == "linear" {
        "weight"
    } else {
        "y"
    };
    total += tensor_input_bytes(op, weight_key, type_map);

    // Input activation.
    let input_key = "x";
    total += tensor_input_bytes(op, input_key, type_map);

    // Output activation — use output_types if available.
    for tt in op.output_types.iter().flatten() {
        if let Some(elems) = static_elements(&tt.shape) {
            total += elems.saturating_mul(dtype_bytes(tt.scalar_type));
        }
    }

    total
}

/// Estimate memory for conv: weight kernel + input feature map + output feature map.
fn estimate_conv_memory(op: &Operation, type_map: &HashMap<String, TensorType>) -> usize {
    let mut total: usize = 0;

    total += tensor_input_bytes(op, "weight", type_map);
    total += tensor_input_bytes(op, "x", type_map);

    for tt in op.output_types.iter().flatten() {
        if let Some(elems) = static_elements(&tt.shape) {
            total += elems.saturating_mul(dtype_bytes(tt.scalar_type));
        }
    }

    total
}

/// Resolve the byte size of a named input (inline `Tensor` or reference to a typed op output).
fn tensor_input_bytes(
    op: &Operation,
    input_name: &str,
    type_map: &HashMap<String, TensorType>,
) -> usize {
    match op.inputs.get(input_name) {
        Some(Value::Tensor { data, .. }) => data.len(),
        Some(Value::Reference(ref_name)) => {
            if let Some(tt) = type_map.get(ref_name) {
                static_elements(&tt.shape)
                    .map(|e| e.saturating_mul(dtype_bytes(tt.scalar_type)))
                    .unwrap_or(0)
            } else {
                0
            }
        }
        _ => 0,
    }
}

// ── Type map construction ─────────────────────────────────────────────

/// Build a map from output-value name → TensorType, using the
/// `output_types` annotations on each operation.
fn build_type_map(block: &Block) -> HashMap<String, TensorType> {
    let mut map = HashMap::new();
    for op in &block.operations {
        for (i, name) in op.outputs.iter().enumerate() {
            if let Some(Some(tt)) = op.output_types.get(i) {
                map.insert(name.clone(), tt.clone());
            }
        }
    }
    map
}

// ── Splitting logic ───────────────────────────────────────────────────

/// Process a block, splitting oversized ops in-place.
fn split_block(
    block: &mut Block,
    budget: usize,
    type_map: &HashMap<String, TensorType>,
) -> Result<()> {
    let mut idx = 0;
    while idx < block.operations.len() {
        let mem = estimate_op_memory(&block.operations[idx], type_map);
        if mem > budget {
            let op = block.operations[idx].clone();
            let replacements = match op.op_type.as_str() {
                "matmul" | "linear" => split_matmul_op(&op, budget, type_map)?,
                "conv" | "conv_transpose" => split_conv_op(&op, budget, type_map)?,
                _ => None,
            };

            if let Some(new_ops) = replacements {
                // The last new op is a concat whose output replaces the
                // original op's output.
                let original_output = op.outputs.first().cloned().unwrap_or_default();
                let concat_output = new_ops
                    .last()
                    .and_then(|o| o.outputs.first())
                    .cloned()
                    .unwrap_or_default();

                // Replace the single op with the tile ops + concat.
                block.operations.splice(idx..=idx, new_ops.iter().cloned());

                // Rewire downstream references.
                if original_output != concat_output {
                    replace_reference(block, &original_output, &concat_output);
                }

                // Skip past the newly inserted ops.
                idx += new_ops.len();
                continue;
            }
        }
        idx += 1;
    }
    Ok(())
}

/// Determine the number of tiles required so that each tile fits within
/// `budget` bytes.  `total_mem` is the full op footprint; we linearly
/// estimate that tiling by `T` divides the *weight* dimension
/// proportionally.
fn tiles_needed(total_mem: usize, budget: usize) -> usize {
    if budget == 0 || total_mem <= budget {
        return 1;
    }
    // Ceiling division.
    total_mem.div_ceil(budget)
}

// ── MatMul / Linear splitting ─────────────────────────────────────────

/// Split a matmul/linear into tiles along the output (K) dimension.
///
/// Original: `out = matmul(x, weight)` where weight is `[N, K]`.
/// Tiled:
///   tile_0 = matmul(x, weight_slice_0)   // weight_slice_0 is [N, K/T]
///   tile_1 = matmul(x, weight_slice_1)
///   ...
///   out = concat(tile_0, tile_1, ..., axis=-1)
fn split_matmul_op(
    op: &Operation,
    budget: usize,
    type_map: &HashMap<String, TensorType>,
) -> Result<Option<Vec<Operation>>> {
    let mem = estimate_matmul_memory(op, type_map);
    let mut n_tiles = tiles_needed(mem, budget);
    if n_tiles <= 1 {
        return Ok(None);
    }

    // Resolve the output dimension K from the original op's output type or
    // from the weight shape.
    let weight_key = if op.op_type == "linear" {
        "weight"
    } else {
        "y"
    };
    let k_dim = resolve_output_dim(op, weight_key, type_map);
    let k_dim = match k_dim {
        Some(k) => k,
        None => return Ok(None), // cannot split without known shapes
    };

    // Clamp tiles to the split dimension to avoid zero-sized slices.
    n_tiles = n_tiles.min(k_dim);

    // Determine dtype from the weight.
    let weight_dtype = resolve_input_dtype(op, weight_key, type_map).unwrap_or(ScalarType::Float32);

    let base_name = &op.name;
    let original_output = op
        .outputs
        .first()
        .ok_or_else(|| MilError::Validation(format!("op '{}' has no outputs", op.name)))?;

    let mut tile_ops = Vec::with_capacity(n_tiles + 1);
    let mut tile_outputs = Vec::with_capacity(n_tiles);

    for t in 0..n_tiles {
        let tile_k_start = t * k_dim / n_tiles;
        let tile_k_end = (t + 1) * k_dim / n_tiles;
        let tile_k = tile_k_end - tile_k_start;

        let tile_name = format!("{base_name}_tile{t}");
        let tile_out = format!("{original_output}_tile{t}");

        // Build the slice operation for the weight.
        let weight_slice_name = format!("{base_name}_wslice{t}");
        let weight_slice_out = format!("{base_name}_wslice{t}_out");

        let mut slice_op =
            Operation::new("slice_by_index", &weight_slice_name).with_output(&weight_slice_out);

        // Forward the weight input.
        if let Some(w) = op.inputs.get(weight_key) {
            slice_op.inputs.insert("x".to_string(), w.clone());
        }

        // begin / end for the last dimension.
        let weight_rank = resolve_input_rank(op, weight_key, type_map).unwrap_or(2);
        slice_op.attributes.insert(
            "begin".to_string(),
            Value::List(
                (0..weight_rank)
                    .map(|i| {
                        if i == weight_rank - 1 {
                            Value::Int(tile_k_start as i64)
                        } else {
                            Value::Int(0)
                        }
                    })
                    .collect(),
            ),
        );
        slice_op.attributes.insert(
            "end".to_string(),
            Value::List(
                (0..weight_rank)
                    .map(|i| {
                        if i == weight_rank - 1 {
                            Value::Int(tile_k_end as i64)
                        } else {
                            Value::Int(-1) // full extent
                        }
                    })
                    .collect(),
            ),
        );
        // Set output type for the sliced weight.
        if let Some(wt) = resolve_input_type(op, weight_key, type_map) {
            let mut sliced_shape = wt.shape.clone();
            if let Some(last) = sliced_shape.last_mut() {
                *last = Some(tile_k);
            }
            slice_op.output_types = vec![Some(TensorType::with_dynamic_shape(
                wt.scalar_type,
                sliced_shape,
            ))];
        }

        tile_ops.push(slice_op);

        // Build the tile matmul/linear.
        let mut tile_op = Operation::new(&op.op_type, &tile_name).with_output(&tile_out);

        // Copy all inputs, replacing the weight with the slice.
        for (k, v) in &op.inputs {
            if k == weight_key {
                tile_op
                    .inputs
                    .insert(k.clone(), Value::Reference(weight_slice_out.clone()));
            } else {
                tile_op.inputs.insert(k.clone(), v.clone());
            }
        }

        // Copy attributes (e.g. transpose flags).
        tile_op.attributes = op.attributes.clone();

        // Set output type for the tile.
        if let Some(Some(orig_tt)) = op.output_types.first() {
            let mut tile_shape = orig_tt.shape.clone();
            if let Some(last) = tile_shape.last_mut() {
                *last = Some(tile_k);
            }
            tile_op.output_types = vec![Some(TensorType::with_dynamic_shape(
                orig_tt.scalar_type,
                tile_shape,
            ))];
        } else {
            tile_op.output_types = vec![None];
        }

        // For linear ops, also copy bias-slice if a bias exists.
        if op.op_type == "linear" {
            if let Some(bias) = op.inputs.get("bias") {
                let bias_slice_name = format!("{base_name}_bslice{t}");
                let bias_slice_out = format!("{base_name}_bslice{t}_out");
                let mut bias_slice =
                    Operation::new("slice_by_index", &bias_slice_name).with_output(&bias_slice_out);
                bias_slice.inputs.insert("x".to_string(), bias.clone());
                bias_slice.attributes.insert(
                    "begin".to_string(),
                    Value::List(vec![Value::Int(tile_k_start as i64)]),
                );
                bias_slice.attributes.insert(
                    "end".to_string(),
                    Value::List(vec![Value::Int(tile_k_end as i64)]),
                );
                bias_slice.output_types = vec![Some(TensorType::new(weight_dtype, vec![tile_k]))];

                tile_op
                    .inputs
                    .insert("bias".to_string(), Value::Reference(bias_slice_out));
                tile_ops.push(bias_slice);
            }
        }

        tile_outputs.push(tile_out.clone());
        tile_ops.push(tile_op);
    }

    // Concat tile outputs.
    let concat_out = original_output.clone();
    let concat_name = format!("{base_name}_concat");
    let mut concat_op = Operation::new("concat", &concat_name).with_output(&concat_out);
    concat_op.inputs.insert(
        "values".to_string(),
        Value::List(
            tile_outputs
                .iter()
                .map(|n| Value::Reference(n.clone()))
                .collect(),
        ),
    );
    concat_op
        .attributes
        .insert("axis".to_string(), Value::Int(-1));

    // Preserve original output type on the concat.
    if let Some(oty) = op.output_types.first() {
        concat_op.output_types = vec![oty.clone()];
    } else {
        concat_op.output_types = vec![None];
    }

    tile_ops.push(concat_op);
    Ok(Some(tile_ops))
}

// ── Conv splitting ────────────────────────────────────────────────────

/// Split a conv/conv_transpose along the output-channel dimension.
fn split_conv_op(
    op: &Operation,
    budget: usize,
    type_map: &HashMap<String, TensorType>,
) -> Result<Option<Vec<Operation>>> {
    let mem = estimate_conv_memory(op, type_map);
    let mut n_tiles = tiles_needed(mem, budget);
    if n_tiles <= 1 {
        return Ok(None);
    }

    // Conv weight shape is typically [C_out, C_in, kH, kW].
    // We split along C_out (dim 0).
    let c_out = resolve_conv_output_channels(op, type_map);
    let c_out = match c_out {
        Some(c) => c,
        None => return Ok(None),
    };

    // Clamp tiles to the split dimension to avoid zero-sized slices.
    n_tiles = n_tiles.min(c_out);

    let weight_dtype = resolve_input_dtype(op, "weight", type_map).unwrap_or(ScalarType::Float32);

    let base_name = &op.name;
    let original_output = op
        .outputs
        .first()
        .ok_or_else(|| MilError::Validation(format!("op '{}' has no outputs", op.name)))?;

    let mut tile_ops = Vec::with_capacity(n_tiles + 1);
    let mut tile_outputs = Vec::with_capacity(n_tiles);

    for t in 0..n_tiles {
        let ch_start = t * c_out / n_tiles;
        let ch_end = (t + 1) * c_out / n_tiles;
        let tile_ch = ch_end - ch_start;

        // Slice weight along dim 0.
        let wslice_name = format!("{base_name}_wslice{t}");
        let wslice_out = format!("{base_name}_wslice{t}_out");
        let mut wslice = Operation::new("slice_by_index", &wslice_name).with_output(&wslice_out);
        if let Some(w) = op.inputs.get("weight") {
            wslice.inputs.insert("x".to_string(), w.clone());
        }

        let weight_rank = resolve_input_rank(op, "weight", type_map).unwrap_or(4);
        wslice.attributes.insert(
            "begin".to_string(),
            Value::List(
                (0..weight_rank)
                    .map(|i| {
                        if i == 0 {
                            Value::Int(ch_start as i64)
                        } else {
                            Value::Int(0)
                        }
                    })
                    .collect(),
            ),
        );
        wslice.attributes.insert(
            "end".to_string(),
            Value::List(
                (0..weight_rank)
                    .map(|i| {
                        if i == 0 {
                            Value::Int(ch_end as i64)
                        } else {
                            Value::Int(-1)
                        }
                    })
                    .collect(),
            ),
        );

        // Set sliced weight output type.
        if let Some(wt) = resolve_input_type(op, "weight", type_map) {
            let mut sliced = wt.shape.clone();
            if let Some(first) = sliced.first_mut() {
                *first = Some(tile_ch);
            }
            wslice.output_types =
                vec![Some(TensorType::with_dynamic_shape(wt.scalar_type, sliced))];
        }
        tile_ops.push(wslice);

        // Slice bias if present.
        if let Some(bias) = op.inputs.get("bias") {
            let bslice_name = format!("{base_name}_bslice{t}");
            let bslice_out = format!("{base_name}_bslice{t}_out");
            let mut bslice =
                Operation::new("slice_by_index", &bslice_name).with_output(&bslice_out);
            bslice.inputs.insert("x".to_string(), bias.clone());
            bslice.attributes.insert(
                "begin".to_string(),
                Value::List(vec![Value::Int(ch_start as i64)]),
            );
            bslice.attributes.insert(
                "end".to_string(),
                Value::List(vec![Value::Int(ch_end as i64)]),
            );
            bslice.output_types = vec![Some(TensorType::new(weight_dtype, vec![tile_ch]))];
            tile_ops.push(bslice);
        }

        // Tile conv.
        let tile_name = format!("{base_name}_tile{t}");
        let tile_out = format!("{original_output}_tile{t}");
        let mut tile_op = Operation::new(&op.op_type, &tile_name).with_output(&tile_out);

        for (k, v) in &op.inputs {
            if k == "weight" {
                tile_op
                    .inputs
                    .insert(k.clone(), Value::Reference(wslice_out.clone()));
            } else if k == "bias" {
                let bslice_out = format!("{base_name}_bslice{t}_out");
                tile_op
                    .inputs
                    .insert(k.clone(), Value::Reference(bslice_out));
            } else {
                tile_op.inputs.insert(k.clone(), v.clone());
            }
        }
        tile_op.attributes = op.attributes.clone();

        // Output type: same as original but with tile_ch channels.
        if let Some(Some(orig_tt)) = op.output_types.first() {
            let mut tile_shape = orig_tt.shape.clone();
            // Channel dim is typically dim 1 in [N, C, H, W].
            if tile_shape.len() >= 2 {
                tile_shape[1] = Some(tile_ch);
            }
            tile_op.output_types = vec![Some(TensorType::with_dynamic_shape(
                orig_tt.scalar_type,
                tile_shape,
            ))];
        } else {
            tile_op.output_types = vec![None];
        }

        tile_outputs.push(tile_out);
        tile_ops.push(tile_op);
    }

    // Concat along channel dim (axis=1 for NCHW).
    let concat_out = original_output.clone();
    let concat_name = format!("{base_name}_concat");
    let mut concat_op = Operation::new("concat", &concat_name).with_output(&concat_out);
    concat_op.inputs.insert(
        "values".to_string(),
        Value::List(
            tile_outputs
                .iter()
                .map(|n| Value::Reference(n.clone()))
                .collect(),
        ),
    );
    concat_op
        .attributes
        .insert("axis".to_string(), Value::Int(1));

    if let Some(oty) = op.output_types.first() {
        concat_op.output_types = vec![oty.clone()];
    } else {
        concat_op.output_types = vec![None];
    }

    tile_ops.push(concat_op);
    Ok(Some(tile_ops))
}

// ── Multi-head attention splitting ────────────────────────────────────

/// Detect multi-head attention patterns and split across heads.
///
/// This looks for matmul/linear ops that carry an `n_heads` attribute
/// (set by attention-fusion or the ONNX converter) and splits the
/// operation so each head is computed independently.
pub fn split_attention_heads(
    block: &mut Block,
    budget: usize,
    type_map: &HashMap<String, TensorType>,
) -> Result<()> {
    let mut idx = 0;
    while idx < block.operations.len() {
        let op = &block.operations[idx];

        let n_heads = match op.attributes.get("n_heads") {
            Some(Value::Int(n)) if *n > 1 => *n as usize,
            _ => {
                idx += 1;
                continue;
            }
        };

        let mem = estimate_op_memory(op, type_map);
        if mem <= budget {
            idx += 1;
            continue;
        }

        // Each head should ideally fit in budget.
        let per_head_mem = mem / n_heads;
        if per_head_mem > budget {
            // Heads are still too large — fall through to regular splitting.
            idx += 1;
            continue;
        }

        let op = block.operations[idx].clone();
        let original_output = match op.outputs.first() {
            Some(o) => o.clone(),
            None => {
                idx += 1;
                continue;
            }
        };

        let base_name = &op.name;
        let head_dim = resolve_head_dim(&op, n_heads, type_map);

        if head_dim.is_none() {
            idx += 1;
            continue;
        }
        let head_dim = head_dim.unwrap();
        if head_dim == 0 {
            idx += 1;
            continue;
        }

        let mut new_ops = Vec::new();
        let mut head_outputs = Vec::new();

        for h in 0..n_heads {
            let h_start = h * head_dim;
            let h_end = (h + 1) * head_dim;

            // Slice input along the head dimension (last axis).
            let input_key = "x";
            let in_slice_name = format!("{base_name}_head{h}_in");
            let in_slice_out = format!("{base_name}_head{h}_in_out");
            let mut in_slice =
                Operation::new("slice_by_index", &in_slice_name).with_output(&in_slice_out);
            if let Some(inp) = op.inputs.get(input_key) {
                in_slice.inputs.insert("x".to_string(), inp.clone());
            }
            let input_rank = resolve_input_rank(&op, input_key, type_map).unwrap_or(3);
            in_slice.attributes.insert(
                "begin".to_string(),
                Value::List(
                    (0..input_rank)
                        .map(|i| {
                            if i == input_rank - 1 {
                                Value::Int(h_start as i64)
                            } else {
                                Value::Int(0)
                            }
                        })
                        .collect(),
                ),
            );
            in_slice.attributes.insert(
                "end".to_string(),
                Value::List(
                    (0..input_rank)
                        .map(|i| {
                            if i == input_rank - 1 {
                                Value::Int(h_end as i64)
                            } else {
                                Value::Int(-1)
                            }
                        })
                        .collect(),
                ),
            );
            in_slice.output_types = vec![None];
            new_ops.push(in_slice);

            // Slice weight.
            let weight_key = if op.op_type == "linear" {
                "weight"
            } else {
                "y"
            };
            let w_slice_name = format!("{base_name}_head{h}_w");
            let w_slice_out = format!("{base_name}_head{h}_w_out");
            let mut w_slice =
                Operation::new("slice_by_index", &w_slice_name).with_output(&w_slice_out);
            if let Some(w) = op.inputs.get(weight_key) {
                w_slice.inputs.insert("x".to_string(), w.clone());
            }
            let weight_rank = resolve_input_rank(&op, weight_key, type_map).unwrap_or(2);
            // Slice both dims of weight for head: rows [h_start..h_end], cols [h_start..h_end].
            // For per-head weight: slice along dim 0 (output features).
            w_slice.attributes.insert(
                "begin".to_string(),
                Value::List(
                    (0..weight_rank)
                        .map(|i| {
                            if i == 0 {
                                Value::Int(h_start as i64)
                            } else {
                                Value::Int(0)
                            }
                        })
                        .collect(),
                ),
            );
            w_slice.attributes.insert(
                "end".to_string(),
                Value::List(
                    (0..weight_rank)
                        .map(|i| {
                            if i == 0 {
                                Value::Int(h_end as i64)
                            } else {
                                Value::Int(-1)
                            }
                        })
                        .collect(),
                ),
            );
            w_slice.output_types = vec![None];
            new_ops.push(w_slice);

            // Per-head matmul/linear.
            let head_name = format!("{base_name}_head{h}");
            let head_out = format!("{original_output}_head{h}");
            let mut head_op = Operation::new(&op.op_type, &head_name).with_output(&head_out);
            head_op
                .inputs
                .insert(input_key.to_string(), Value::Reference(in_slice_out));
            head_op
                .inputs
                .insert(weight_key.to_string(), Value::Reference(w_slice_out));
            // Copy other inputs/attrs but drop n_heads.
            for (k, v) in &op.attributes {
                if k != "n_heads" {
                    head_op.attributes.insert(k.clone(), v.clone());
                }
            }
            head_op.output_types = vec![None];

            head_outputs.push(head_out);
            new_ops.push(head_op);
        }

        // Concat heads.
        let concat_out = original_output.clone();
        let concat_name = format!("{base_name}_head_concat");
        let mut concat_op = Operation::new("concat", &concat_name).with_output(&concat_out);
        concat_op.inputs.insert(
            "values".to_string(),
            Value::List(
                head_outputs
                    .iter()
                    .map(|n| Value::Reference(n.clone()))
                    .collect(),
            ),
        );
        concat_op
            .attributes
            .insert("axis".to_string(), Value::Int(-1));
        if let Some(oty) = op.output_types.first() {
            concat_op.output_types = vec![oty.clone()];
        } else {
            concat_op.output_types = vec![None];
        }
        new_ops.push(concat_op);

        block.operations.splice(idx..=idx, new_ops.iter().cloned());
        idx += new_ops.len();
    }
    Ok(())
}

// ── Shape helpers ─────────────────────────────────────────────────────

/// Resolve the last-dimension size of a weight tensor (K in [N, K]).
fn resolve_output_dim(
    op: &Operation,
    weight_key: &str,
    type_map: &HashMap<String, TensorType>,
) -> Option<usize> {
    match op.inputs.get(weight_key) {
        Some(Value::Tensor { shape, .. }) => shape.last().copied(),
        Some(Value::Reference(name)) => type_map
            .get(name)
            .and_then(|tt| tt.shape.last().copied().flatten()),
        _ => None,
    }
}

/// Resolve the rank of a named input tensor.
fn resolve_input_rank(
    op: &Operation,
    key: &str,
    type_map: &HashMap<String, TensorType>,
) -> Option<usize> {
    match op.inputs.get(key) {
        Some(Value::Tensor { shape, .. }) => Some(shape.len()),
        Some(Value::Reference(name)) => type_map.get(name).map(|tt| tt.rank()),
        _ => None,
    }
}

/// Resolve the ScalarType of a named input.
fn resolve_input_dtype(
    op: &Operation,
    key: &str,
    type_map: &HashMap<String, TensorType>,
) -> Option<ScalarType> {
    match op.inputs.get(key) {
        Some(Value::Tensor { dtype, .. }) => Some(*dtype),
        Some(Value::Reference(name)) => type_map.get(name).map(|tt| tt.scalar_type),
        _ => None,
    }
}

/// Resolve the full TensorType of a named input.
fn resolve_input_type(
    op: &Operation,
    key: &str,
    type_map: &HashMap<String, TensorType>,
) -> Option<TensorType> {
    match op.inputs.get(key) {
        Some(Value::Tensor { shape, dtype, .. }) => Some(TensorType::new(*dtype, shape.clone())),
        Some(Value::Reference(name)) => type_map.get(name).cloned(),
        _ => None,
    }
}

/// Resolve conv output channels from weight shape (dim 0).
fn resolve_conv_output_channels(
    op: &Operation,
    type_map: &HashMap<String, TensorType>,
) -> Option<usize> {
    match op.inputs.get("weight") {
        Some(Value::Tensor { shape, .. }) => shape.first().copied(),
        Some(Value::Reference(name)) => type_map
            .get(name)
            .and_then(|tt| tt.shape.first().copied().flatten()),
        _ => None,
    }
}

/// Resolve per-head dimension from the output type and head count.
fn resolve_head_dim(
    op: &Operation,
    n_heads: usize,
    _type_map: &HashMap<String, TensorType>,
) -> Option<usize> {
    // Use the output type's last dimension.
    if let Some(Some(tt)) = op.output_types.first() {
        if let Some(Some(total)) = tt.shape.last() {
            if total % n_heads == 0 {
                return Some(total / n_heads);
            }
        }
    }
    None
}

// ── Human-readable size parsing ───────────────────────────────────────

/// Parse a human-readable memory size string (e.g. "1GB", "512MB", "2gb")
/// into bytes.
///
/// Supported suffixes: `B`, `KB`, `MB`, `GB`, `TB` (case-insensitive).
/// Plain integers without a suffix are treated as bytes.
pub fn parse_memory_size(s: &str) -> std::result::Result<usize, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty memory size string".into());
    }

    // Split into numeric prefix and alphabetic suffix.
    let mut split_pos = s.len();
    for (i, c) in s.char_indices().rev() {
        if c.is_ascii_digit() || c == '.' {
            split_pos = i + c.len_utf8();
            break;
        }
    }

    let num_str = &s[..split_pos];
    let suffix = s[split_pos..].trim().to_uppercase();

    let num: f64 = num_str
        .parse()
        .map_err(|_| format!("invalid number in memory size: '{num_str}'"))?;

    let multiplier: f64 = match suffix.as_str() {
        "" | "B" => 1.0,
        "KB" | "K" => 1024.0,
        "MB" | "M" => 1024.0 * 1024.0,
        "GB" | "G" => 1024.0 * 1024.0 * 1024.0,
        "TB" | "T" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => return Err(format!("unknown size suffix: '{other}'")),
    };

    let bytes = num * multiplier;
    if !bytes.is_finite() || bytes < 0.0 || bytes > usize::MAX as f64 {
        return Err(format!("memory size overflows: '{s}'"));
    }
    Ok(bytes as usize)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::Operation;
    use mil_rs::ir::Value;
    use mil_rs::ir::{Block, Function, Program};
    use mil_rs::ir::{ScalarType, TensorType};

    /// Helper: create a matmul op with an inline weight tensor.
    fn matmul_op(name: &str, n: usize, k: usize) -> Operation {
        let weight_data = vec![0u8; n * k * 4]; // f32
        Operation::new("matmul", name)
            .with_input("x", Value::Reference("input".into()))
            .with_input(
                "y",
                Value::Tensor {
                    data: weight_data,
                    shape: vec![n, k],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output(format!("{name}_out"))
    }

    /// Helper: create a linear op.
    fn linear_op(name: &str, n: usize, k: usize) -> Operation {
        let weight_data = vec![0u8; n * k * 4];
        Operation::new("linear", name)
            .with_input("x", Value::Reference("input".into()))
            .with_input(
                "weight",
                Value::Tensor {
                    data: weight_data,
                    shape: vec![k, n],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output(format!("{name}_out"))
    }

    /// Helper: create a conv op.
    fn conv_op(name: &str, c_out: usize, c_in: usize, kh: usize, kw: usize) -> Operation {
        let weight_data = vec![0u8; c_out * c_in * kh * kw * 4];
        Operation::new("conv", name)
            .with_input("x", Value::Reference("input".into()))
            .with_input(
                "weight",
                Value::Tensor {
                    data: weight_data,
                    shape: vec![c_out, c_in, kh, kw],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output(format!("{name}_out"))
    }

    fn program_with_ops(ops: Vec<Operation>) -> Program {
        let mut func = Function::new("main");
        let last_out = ops.last().and_then(|o| o.outputs.first()).cloned();
        for op in ops {
            func.body.add_op(op);
        }
        if let Some(out) = last_out {
            func.body.outputs.push(out);
        }
        let mut program = Program::new("1.0.0");
        program.add_function(func);
        program
    }

    #[test]
    fn pass_name() {
        let pass = OpSplittingPass::default();
        assert_eq!(pass.name(), "op-splitting");
    }

    #[test]
    fn default_budget_is_1gb() {
        let pass = OpSplittingPass::default();
        assert_eq!(pass.memory_budget_bytes, 1_073_741_824);
    }

    #[test]
    fn small_op_not_split() {
        let op = matmul_op("mm0", 64, 64);
        let mut program = program_with_ops(vec![op]);
        let pass = OpSplittingPass::default();
        pass.run(&mut program).unwrap();
        // Op fits in budget — should remain 1 op (not split).
        assert_eq!(program.functions["main"].body.operations.len(), 1);
    }

    #[test]
    fn large_matmul_is_split() {
        // Create a matmul whose weight alone exceeds a tiny budget.
        let op = matmul_op("mm0", 256, 512);
        // weight = 256*512*4 = 524288 bytes
        let mut program = program_with_ops(vec![op]);

        // Use a 300KB budget so the op must be split.
        let pass = OpSplittingPass::new(300_000);
        pass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        // Should have been split into tiles + slices + concat.
        assert!(ops.len() > 1, "expected splitting, got {} ops", ops.len());
        // Last op should be concat.
        assert_eq!(ops.last().unwrap().op_type, "concat");
        // Block output should still be mm0_out.
        assert_eq!(program.functions["main"].body.outputs[0], "mm0_out");
    }

    #[test]
    fn linear_is_split() {
        let op = linear_op("lin0", 256, 512);
        let mut program = program_with_ops(vec![op]);
        let pass = OpSplittingPass::new(300_000);
        pass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        assert!(ops.len() > 1);
        assert_eq!(ops.last().unwrap().op_type, "concat");
    }

    #[test]
    fn conv_is_split() {
        // weight = 256 * 128 * 3 * 3 * 4 = 1,179,648 bytes
        let op = conv_op("conv0", 256, 128, 3, 3);
        let mut program = program_with_ops(vec![op]);
        let pass = OpSplittingPass::new(500_000);
        pass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        assert!(ops.len() > 1);
        assert_eq!(ops.last().unwrap().op_type, "concat");
    }

    #[test]
    fn tiles_needed_basic() {
        assert_eq!(tiles_needed(100, 200), 1);
        assert_eq!(tiles_needed(200, 200), 1);
        assert_eq!(tiles_needed(201, 200), 2);
        assert_eq!(tiles_needed(400, 200), 2);
        assert_eq!(tiles_needed(401, 200), 3);
        assert_eq!(tiles_needed(100, 0), 1);
    }

    #[test]
    fn parse_memory_size_valid() {
        assert_eq!(parse_memory_size("1GB").unwrap(), 1_073_741_824);
        assert_eq!(parse_memory_size("2GB").unwrap(), 2_147_483_648);
        assert_eq!(parse_memory_size("512MB").unwrap(), 536_870_912);
        assert_eq!(parse_memory_size("1024KB").unwrap(), 1_048_576);
        assert_eq!(parse_memory_size("1024").unwrap(), 1024);
        assert_eq!(parse_memory_size("1gb").unwrap(), 1_073_741_824);
        assert_eq!(parse_memory_size("1G").unwrap(), 1_073_741_824);
    }

    #[test]
    fn parse_memory_size_invalid() {
        assert!(parse_memory_size("").is_err());
        assert!(parse_memory_size("abc").is_err());
        assert!(parse_memory_size("1XB").is_err());
    }

    #[test]
    fn output_preserved_after_split() {
        // Ensure downstream references are correctly rewired.
        let mm = matmul_op("mm0", 256, 512);
        let relu = Operation::new("relu", "relu0")
            .with_input("x", Value::Reference("mm0_out".into()))
            .with_output("relu0_out");

        let mut program = program_with_ops(vec![mm, relu]);
        program.functions.get_mut("main").unwrap().body.outputs = vec!["relu0_out".into()];

        let pass = OpSplittingPass::new(300_000);
        pass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        // Find the relu and check its input still references the concat output.
        let relu_op = ops.iter().find(|o| o.op_type == "relu").unwrap();
        match relu_op.inputs.get("x") {
            Some(Value::Reference(name)) => assert_eq!(name, "mm0_out"),
            other => panic!("expected Reference to mm0_out, got {other:?}"),
        }
    }

    #[test]
    fn dtype_bytes_values() {
        assert_eq!(dtype_bytes(ScalarType::Float32), 4);
        assert_eq!(dtype_bytes(ScalarType::Float16), 2);
        assert_eq!(dtype_bytes(ScalarType::Int8), 1);
        assert_eq!(dtype_bytes(ScalarType::Float64), 8);
    }

    #[test]
    fn estimate_matmul_memory_basic() {
        let op = matmul_op("mm", 128, 256);
        let type_map = HashMap::new();
        let mem = estimate_op_memory(&op, &type_map);
        // Weight: 128 * 256 * 4 = 131072 bytes (inline tensor).
        assert!(mem >= 131072);
    }

    #[test]
    fn multiple_ops_only_large_split() {
        let small = matmul_op("small", 8, 8);
        let large = matmul_op("large", 256, 512);
        let mut program = program_with_ops(vec![small, large]);

        let pass = OpSplittingPass::new(300_000);
        pass.run(&mut program).unwrap();

        let ops = &program.functions["main"].body.operations;
        // small should remain, large should be split.
        assert!(ops.iter().any(|o| o.name == "small"));
        assert!(ops.iter().any(|o| o.op_type == "concat"));
    }

    #[test]
    fn attention_head_split() {
        // Create a matmul with n_heads=4 and a known output type.
        let mut op = matmul_op("attn", 256, 256);
        op.attributes.insert("n_heads".to_string(), Value::Int(4));
        op.output_types = vec![Some(TensorType::new(ScalarType::Float32, vec![1, 256]))];

        let mut block = Block::new();
        block.add_op(op);
        block.outputs.push("attn_out".into());

        let type_map = build_type_map(&block);
        // Budget smaller than full op but larger than per-head.
        // Full weight = 256*256*4 = 262144. Per head ≈ 65536.
        split_attention_heads(&mut block, 100_000, &type_map).unwrap();

        // Should have been split into per-head slices + concat.
        assert!(block.operations.len() > 1);
        assert_eq!(block.operations.last().unwrap().op_type, "concat");
    }
}
