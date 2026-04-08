//! Helper functions for the decode compilation pipeline.
//!
//! These functions manipulate MIL IR programs to prepare them for ANE
//! compilation. Ported from `ironmill-inference`'s decode module so that
//! `compile_decode_bundle()` can run without a runtime dependency.

use std::collections::{BTreeMap, HashMap, HashSet};

use half::f16;
use mil_rs::ir::{Operation, Program, ScalarType, TensorType, Value};

use crate::ane::split::SubProgram;
use ironmill_core::ane::TensorDescriptor;

// ---------------------------------------------------------------------------
// Shared tensor conversion helper
// ---------------------------------------------------------------------------

/// Convert materialized tensor bytes to `Vec<f16>`.
///
/// Supports `Float32` → fp16 conversion and `Float16` passthrough.
/// Returns `None` for other scalar types.
fn bytes_to_f16(data: &[u8], dtype: ScalarType) -> Option<Vec<f16>> {
    match dtype {
        ScalarType::Float32 => Some(
            data.chunks_exact(4)
                .map(|b| f16::from_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                .collect(),
        ),
        ScalarType::Float16 => Some(
            data.chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect(),
        ),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// RoPE cache extraction
// ---------------------------------------------------------------------------

/// Extracted RoPE cos/sin cache data: `(cos_values, sin_values, values_per_position)`.
pub type RopeCacheData = (Vec<f16>, Vec<f16>, usize);

/// Extract RoPE cos/sin cache data from model const ops.
///
/// Walks the program looking for `gather` ops whose names contain "cos"
/// or "sin" (produced by RotaryEmbedding decomposition). Traces back to
/// the const table input and extracts it.
///
/// Returns `(cos_values, sin_values, values_per_position)` where each
/// value array is flat `[num_positions * values_per_position]` in fp16.
///
/// Skips tensors whose data is not materialized.
pub fn extract_rope_caches(program: &Program) -> Option<RopeCacheData> {
    let func = program.main()?;

    // Map output_name → (data, shape, dtype) for const ops.
    let mut const_map: HashMap<&str, (&[u8], &[usize], ScalarType)> = HashMap::new();
    for op in &func.body.operations {
        if op.op_type == "const" {
            let tensor = op.inputs.get("val").or_else(|| op.attributes.get("val"));
            if let Some(Value::Tensor { data, shape, dtype }) = tensor {
                if let Some(bytes) = data.as_bytes() {
                    for out in &op.outputs {
                        const_map.insert(out.as_str(), (bytes, shape.as_slice(), *dtype));
                    }
                }
            }
        }
    }

    let mut cos_result: Option<(Vec<f16>, usize, usize)> = None;
    let mut sin_result: Option<(Vec<f16>, usize, usize)> = None;

    for op in &func.body.operations {
        if op.op_type != "gather" {
            continue;
        }
        let name = op.name.to_ascii_lowercase();
        let is_cos = name.contains("cos");
        let is_sin = name.contains("sin");
        if !is_cos && !is_sin {
            continue;
        }

        // Trace the gather's "x" input back to a const table.
        if let Some(Value::Reference(cache_ref)) = op.inputs.get("x") {
            if let Some(&(data, shape, dtype)) = const_map.get(cache_ref.as_str()) {
                if shape.len() < 2 {
                    continue;
                }
                let f16_values = match bytes_to_f16(data, dtype) {
                    Some(v) => v,
                    None => continue,
                };
                let num_pos = shape[0];
                let dim = shape[1];

                if is_cos && cos_result.is_none() {
                    cos_result = Some((f16_values.clone(), num_pos, dim));
                }
                if is_sin && sin_result.is_none() {
                    sin_result = Some((f16_values, num_pos, dim));
                }
            }
        }
    }

    match (cos_result, sin_result) {
        (Some((cos, _num_pos, dim)), Some((sin, _, _))) => Some((cos, sin, dim)),
        _ => None,
    }
}

/// Precompute RoPE cos/sin cache tables from scratch.
///
/// Used as a fallback when the model's const tables can't be extracted.
pub fn precompute_rope_cache(head_dim: usize, max_pos: usize, theta: f32) -> RopeCacheData {
    let half_dim = head_dim / 2;
    let mut cos_cache = Vec::with_capacity(max_pos * half_dim);
    let mut sin_cache = Vec::with_capacity(max_pos * half_dim);

    for pos in 0..max_pos {
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            cos_cache.push(f16::from_f32(angle.cos()));
            sin_cache.push(f16::from_f32(angle.sin()));
        }
    }
    (cos_cache, sin_cache, half_dim)
}

// ---------------------------------------------------------------------------
// QK norm weight extraction
// ---------------------------------------------------------------------------

/// Extract per-layer QK normalization weights from the original program.
///
/// Qwen3 models apply per-head RMSNorm to Q and K after projection.
/// Returns `Some(Vec<(q_norm, k_norm)>)` indexed by layer, or `None`
/// if the model doesn't use QK normalization.
///
/// Skips tensors whose data is not materialized or has unsupported dtype.
pub fn extract_qk_norm_weights(program: &Program) -> Option<Vec<(Vec<f16>, Vec<f16>)>> {
    let func = program.main()?;

    let mut q_norms: BTreeMap<usize, Vec<f16>> = BTreeMap::new();
    let mut k_norms: BTreeMap<usize, Vec<f16>> = BTreeMap::new();

    for op in &func.body.operations {
        if op.op_type != "const" {
            continue;
        }
        let name = op.name.to_lowercase();
        let is_q_norm = name.contains("q_norm") && name.contains("weight");
        let is_k_norm = name.contains("k_norm") && name.contains("weight");
        if !is_q_norm && !is_k_norm {
            continue;
        }

        let layer_idx = match extract_layer_number_from_name(&name) {
            Some(l) => l,
            None => continue,
        };

        let tensor = op.inputs.get("val").or_else(|| op.attributes.get("val"));
        if let Some(Value::Tensor { data, shape, dtype }) = tensor {
            if shape.len() == 1 {
                let Some(data) = data.as_bytes() else {
                    continue;
                };
                let Some(values) = bytes_to_f16(data, *dtype) else {
                    continue;
                };
                if is_q_norm {
                    q_norms.insert(layer_idx, values);
                } else {
                    k_norms.insert(layer_idx, values);
                }
            }
        }
    }

    if q_norms.is_empty() {
        return None;
    }

    let num_layers = q_norms.keys().last().map(|&k| k + 1).unwrap_or(0);
    let mut result = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let q = q_norms.get(&i).cloned().unwrap_or_default();
        let k = k_norms.get(&i).cloned().unwrap_or_default();
        result.push((q, k));
    }
    Some(result)
}

/// Extract a layer number from an operation name.
pub fn extract_layer_number_from_name(name: &str) -> Option<usize> {
    for pattern in ["layers.", "layers_", "layer.", "layer_"] {
        if let Some(idx) = name.find(pattern) {
            let rest = &name[idx + pattern.len()..];
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !num_str.is_empty() {
                return num_str.parse().ok();
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Gather / split / concat / sub replacement
// ---------------------------------------------------------------------------

/// Replace `gather`, `split`, `concat`, and `sub` ops with function inputs.
///
/// These ops are unsupported on ANE at runtime or trigger compiler bugs.
/// Each is replaced with a function input that the CPU fills at decode time,
/// preserving all downstream references.
pub fn replace_gather_with_inputs(program: &mut Program) {
    let strip_ops = ["gather", "split", "concat", "sub"];

    for func in program.functions.values_mut() {
        let mut replaced_outputs: Vec<(String, TensorType)> = Vec::new();

        for op in &func.body.operations {
            if !strip_ops.contains(&op.op_type.as_str()) {
                continue;
            }
            for (idx, out_name) in op.outputs.iter().enumerate() {
                let out_type = op
                    .output_types
                    .get(idx)
                    .cloned()
                    .flatten()
                    .unwrap_or_else(|| {
                        TensorType::with_dynamic_shape(
                            ScalarType::Float16,
                            vec![Some(1), Some(1), Some(1), Some(1)],
                        )
                    });
                replaced_outputs.push((out_name.clone(), out_type));
            }
        }

        if replaced_outputs.is_empty() {
            continue;
        }

        // Add new function inputs for each replaced value.
        for (name, ty) in &replaced_outputs {
            let input_name = format!("cpu_{name}");
            func.inputs.push((input_name, ty.clone()));
        }

        // Rewrite references: replace output names with new input names.
        let rename_map: HashMap<String, String> = replaced_outputs
            .iter()
            .map(|(name, _)| (name.clone(), format!("cpu_{name}")))
            .collect();

        for op in &mut func.body.operations {
            for val in op.inputs.values_mut() {
                if let Value::Reference(r) = val {
                    if let Some(new_name) = rename_map.get(r.as_str()) {
                        *r = new_name.clone();
                    }
                }
            }
        }

        // Remove the stripped ops.
        func.body
            .operations
            .retain(|op| !strip_ops.contains(&op.op_type.as_str()));

        // Dead code elimination: remove const ops whose outputs are
        // no longer referenced by any remaining op or the function output.
        let mut referenced_names: HashSet<String> = HashSet::new();
        for op in &func.body.operations {
            for val in op.inputs.values() {
                if let Value::Reference(r) = val {
                    referenced_names.insert(r.clone());
                }
            }
            for val in op.attributes.values() {
                if let Value::Reference(r) = val {
                    referenced_names.insert(r.clone());
                }
            }
        }
        for out_name in &func.body.outputs {
            referenced_names.insert(out_name.clone());
        }
        // Remove unreferenced const ops (iterate until stable).
        loop {
            let before = func.body.operations.len();
            func.body.operations.retain(|op| {
                if op.op_type == "const" {
                    op.outputs.iter().any(|o| referenced_names.contains(o))
                } else {
                    true
                }
            });
            if func.body.operations.len() == before {
                break;
            }
            referenced_names.clear();
            for op in &func.body.operations {
                for val in op.inputs.values() {
                    if let Value::Reference(r) = val {
                        referenced_names.insert(r.clone());
                    }
                }
                for val in op.attributes.values() {
                    if let Value::Reference(r) = val {
                        referenced_names.insert(r.clone());
                    }
                }
            }
            for out_name in &func.body.outputs {
                referenced_names.insert(out_name.clone());
            }
        }

        // Remove unreferenced function inputs (dangling after op stripping).
        func.inputs
            .retain(|(name, _)| referenced_names.contains(name));
    }
}

// ---------------------------------------------------------------------------
// Prune unreferenced inputs
// ---------------------------------------------------------------------------

/// Remove function inputs that are not referenced by any op in the body.
pub fn prune_unreferenced_inputs(program: &mut Program) {
    for func in program.functions.values_mut() {
        let mut referenced: HashSet<String> = HashSet::new();
        for op in &func.body.operations {
            for val in op.inputs.values() {
                if let Value::Reference(r) = val {
                    referenced.insert(r.clone());
                }
            }
            for val in op.attributes.values() {
                if let Value::Reference(r) = val {
                    referenced.insert(r.clone());
                }
            }
        }
        for out_name in &func.body.outputs {
            referenced.insert(out_name.clone());
        }
        func.inputs.retain(|(name, _)| referenced.contains(name));
    }
}

// ---------------------------------------------------------------------------
// FFN residual injection
// ---------------------------------------------------------------------------

/// Inject second residual add into post_attn sub-programs.
///
/// ONNX SkipLayerNorm fuses the second residual (skip_add + FFN_output)
/// into the next layer's input norm. After splitting, post_attn outputs
/// just FFN_output. This injects `add(skip_add, FFN_output)` to restore it.
pub fn inject_ffn_residual(program: &mut Program) {
    let func = match program.functions.values_mut().next() {
        Some(f) => f,
        None => return,
    };

    // Find the skip_add variable: an `add` op whose output name contains "skip_add".
    let skip_add_name = func
        .body
        .operations
        .iter()
        .find(|op| {
            op.op_type == "add"
                && op
                    .outputs
                    .first()
                    .map(|n| n.contains("skip_add"))
                    .unwrap_or(false)
        })
        .and_then(|op| op.outputs.first().cloned());

    let skip_add_name = match skip_add_name {
        Some(n) => n,
        None => return,
    };

    // Check if the output already includes a residual add referencing skip_add.
    if let Some(out) = func.body.outputs.first() {
        let already_has_residual = func.body.operations.iter().any(|op| {
            op.outputs.iter().any(|o| o == out)
                && op.op_type == "add"
                && op
                    .inputs
                    .values()
                    .any(|v| matches!(v, Value::Reference(r) if r == &skip_add_name))
        });
        if already_has_residual {
            return;
        }
    }

    let current_output = match func.body.outputs.first().cloned() {
        Some(n) => n,
        None => return,
    };

    // Get output type from the producing op.
    let output_type = func
        .body
        .operations
        .iter()
        .find(|op| op.outputs.iter().any(|o| o == &current_output))
        .and_then(|op| {
            let idx = op.outputs.iter().position(|o| o == &current_output)?;
            op.output_types.get(idx)?.clone()
        });

    // Append: new_output = add(skip_add, down_proj)
    let new_output_name = format!("{current_output}_with_residual");
    let mut add_op = Operation::new("add", format!("{new_output_name}_op"))
        .with_input("x", Value::Reference(skip_add_name))
        .with_input("y", Value::Reference(current_output));
    add_op.outputs = vec![new_output_name.clone()];
    add_op.output_types = vec![output_type];

    func.body.operations.push(add_op);
    func.body.outputs = vec![new_output_name];
}

// ---------------------------------------------------------------------------
// Cache-write fusion
// ---------------------------------------------------------------------------

/// Configuration subset needed for cache-write fusion.
pub struct CacheWriteConfig {
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key-value attention heads (may differ in GQA).
    pub num_kv_heads: usize,
    /// Dimensionality of each attention head.
    pub head_dim: usize,
}

/// Inject TurboQuant cache-write ops into a pre_attn sub-program.
///
/// Reorders outputs to canonical `[Q, K, V, ...extras]` order.
/// Returns `Ok(true)` if injection was applied, `Ok(false)` if skipped.
pub fn inject_cache_write_ops(
    sub: &mut SubProgram,
    config: &CacheWriteConfig,
) -> anyhow::Result<bool> {
    let func = sub
        .program
        .functions
        .values_mut()
        .next()
        .ok_or_else(|| anyhow::anyhow!("pre_attn has no function"))?;

    let num_outputs = func.body.outputs.len();
    if num_outputs < 3 {
        return Ok(false);
    }

    let kv_ch = config.num_kv_heads * config.head_dim;
    let q_ch = config.num_heads * config.head_dim;

    // Identify Q, K_proj, V_proj outputs by name pattern.
    let mut q_idx = None;
    let mut k_idx = None;
    let mut v_idx = None;
    for (i, name) in func.body.outputs.iter().enumerate() {
        let lower = name.to_lowercase();
        if lower.contains("k_proj") {
            k_idx = Some(i);
        } else if lower.contains("v_proj") {
            v_idx = Some(i);
        } else if lower.contains("q_proj") {
            q_idx = Some(i);
        }
    }

    // Fallback: identify by channel count.
    if q_idx.is_none() || k_idx.is_none() || v_idx.is_none() {
        for (i, td) in sub.outputs.iter().enumerate() {
            if td.shape[1] == q_ch && q_idx.is_none() {
                q_idx = Some(i);
            } else if td.shape[1] == kv_ch {
                if k_idx.is_none() {
                    k_idx = Some(i);
                } else if v_idx.is_none() {
                    v_idx = Some(i);
                }
            }
        }
    }

    let (q_idx, k_idx, v_idx) = match (q_idx, k_idx, v_idx) {
        (Some(q), Some(k), Some(v)) => (q, k, v),
        _ => return Ok(false),
    };

    // Reorder function outputs to [Q, K, V, ...extras...].
    let q_name = func.body.outputs[q_idx].clone();
    let k_name = func.body.outputs[k_idx].clone();
    let v_name = func.body.outputs[v_idx].clone();
    let mut extras: Vec<String> = Vec::new();
    for (i, name) in func.body.outputs.iter().enumerate() {
        if i != q_idx && i != k_idx && i != v_idx {
            extras.push(name.clone());
        }
    }
    let mut new_outputs = vec![q_name, k_name, v_name];
    new_outputs.extend(extras);
    func.body.outputs = new_outputs;

    let q_td = sub.outputs[q_idx].clone();
    let k_td = sub.outputs[k_idx].clone();
    let v_td = sub.outputs[v_idx].clone();
    let mut extra_tds: Vec<TensorDescriptor> = Vec::new();
    for (i, td) in sub.outputs.iter().enumerate() {
        if i != q_idx && i != k_idx && i != v_idx {
            extra_tds.push(td.clone());
        }
    }
    let mut new_tds = vec![q_td, k_td, v_td];
    new_tds.extend(extra_tds);
    sub.outputs = new_tds;

    Ok(true)
}

// ---------------------------------------------------------------------------
// CPU weight extraction
// ---------------------------------------------------------------------------

/// CPU-side weight tensor extracted from a sub-program.
pub struct CpuWeight {
    /// Raw fp16 bytes, row-major.
    pub data: Vec<u8>,
    /// `[rows, cols]`.
    pub shape: [usize; 2],
}

/// Extract the largest weight tensor from a sub-program for CPU execution.
///
/// Walks the sub-program's ops looking for `const` ops with tensor values.
/// Returns the largest one (embedding table or lm_head weight), converted
/// to fp16 if needed. Skips tensors whose data is not materialized.
pub fn extract_cpu_weight(sub: &SubProgram, _label: &str) -> Option<CpuWeight> {
    let func = sub.program.main()?;
    let mut best: Option<(usize, Vec<u8>, [usize; 2], ScalarType)> = None;

    for op in &func.body.operations {
        if op.op_type == "const" {
            let tensor = op.inputs.get("val").or_else(|| op.attributes.get("val"));
            if let Some(Value::Tensor { data, shape, dtype }) = tensor {
                if shape.len() >= 2 && data.byte_len() > best.as_ref().map_or(0, |b| b.0) {
                    if let Some(bytes) = data.as_bytes() {
                        best = Some((
                            data.byte_len(),
                            bytes.to_vec(),
                            [shape[0], shape[1]],
                            *dtype,
                        ));
                    }
                }
            }
        }
    }

    best.and_then(|(_, data, shape, dtype)| {
        let fp16_data = if dtype == ScalarType::Float32 {
            bytes_to_f16(&data, dtype)?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        } else {
            data
        };
        Some(CpuWeight {
            data: fp16_data,
            shape,
        })
    })
}

/// Extract a 1D weight (e.g., RMSNorm gamma) from a sub-program.
/// Finds the smallest 1D const tensor whose name contains `hint`.
///
/// Skips tensors whose data is not materialized or has unsupported dtype.
pub fn extract_1d_weight(sub: &SubProgram, hint: &str) -> Option<Vec<f16>> {
    let func = sub.program.main()?;

    for op in &func.body.operations {
        if op.op_type != "const" {
            continue;
        }
        let name_matches = op.name.to_lowercase().contains(hint);
        if !name_matches {
            continue;
        }
        let tensor = op.inputs.get("val").or_else(|| op.attributes.get("val"));
        if let Some(Value::Tensor { data, shape, dtype }) = tensor {
            if shape.len() == 1 {
                let Some(data) = data.as_bytes() else {
                    continue;
                };
                let Some(values) = bytes_to_f16(data, *dtype) else {
                    continue;
                };
                return Some(values);
            }
        }
    }
    None
}

/// Convert Float32 const ops to Float16, materialize dynamic shapes,
/// and decompose unsupported ops for ANE compatibility.
///
/// Returns an error if any Float32 tensor data is not materialized.
pub fn convert_f32_consts_to_f16(program: &mut Program) -> crate::error::Result<()> {
    for func in program.functions.values_mut() {
        // First pass: decompose unsupported ops.
        let mut new_ops = Vec::with_capacity(func.body.operations.len());
        for op in &func.body.operations {
            if op.op_type == "reciprocal" {
                let x_input = op
                    .inputs
                    .get("x")
                    .cloned()
                    .unwrap_or(Value::Reference("unknown".into()));
                let mut div_op = Operation::new("real_div", &op.name)
                    .with_input("x", Value::Float(1.0))
                    .with_input("y", x_input);
                for out_name in &op.outputs {
                    div_op = div_op.with_output(out_name);
                }
                div_op.output_types = op.output_types.clone();
                new_ops.push(div_op);
            } else {
                new_ops.push(op.clone());
            }
        }
        func.body.operations = new_ops;

        // Second pass: convert dtypes and materialize shapes.
        for op in &mut func.body.operations {
            for val in op.inputs.values_mut().chain(op.attributes.values_mut()) {
                if let Value::Tensor {
                    data,
                    shape: _,
                    dtype,
                } = val
                {
                    if *dtype == ScalarType::Float32 {
                        let bytes = data.as_bytes().ok_or_else(|| {
                            crate::error::CompileError::WeightLoadError(
                                "tensor not materialized".into(),
                            )
                        })?;
                        let f32_values: Vec<f32> = bytes
                            .chunks_exact(4)
                            .map(|b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect();
                        let f16_bytes: Vec<u8> = f32_values
                            .iter()
                            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                            .collect();
                        *data = mil_rs::ir::TensorData::Inline(f16_bytes);
                        *dtype = ScalarType::Float16;
                    }
                }
            }
            for t in op.output_types.iter_mut().flatten() {
                if t.scalar_type == ScalarType::Float32 {
                    t.scalar_type = ScalarType::Float16;
                }
                for dim in &mut t.shape {
                    if dim.is_none() {
                        *dim = Some(1);
                    }
                }
            }
        }
        for (_, ty) in &mut func.inputs {
            if ty.scalar_type == ScalarType::Float32 {
                ty.scalar_type = ScalarType::Float16;
            }
            for dim in &mut ty.shape {
                if dim.is_none() {
                    *dim = Some(1);
                }
            }
        }
    }
    Ok(())
}

/// Pad the S dimension (dim 3) to at least `min_seq` across function
/// inputs, op output types, and reshape shape constants.
///
/// # Panics
///
/// Panics if any tensor data is `TensorData::External` (not materialized).
pub fn apply_min_seq_padding(program: &mut Program, min_seq: usize) {
    for func in program.functions.values_mut() {
        for (_, ty) in &mut func.inputs {
            if ty.shape.len() >= 4 {
                if let Some(s) = ty.shape[3] {
                    if s < min_seq {
                        ty.shape[3] = Some(min_seq);
                    }
                }
            }
        }
        for op in &mut func.body.operations {
            for t in op.output_types.iter_mut().flatten() {
                if t.shape.len() >= 4 {
                    if let Some(s) = t.shape[3] {
                        if s < min_seq {
                            t.shape[3] = Some(min_seq);
                        }
                    }
                }
            }
            if op.op_type == "reshape" {
                if let Some(Value::Tensor { shape, data, dtype }) = op.inputs.get_mut("shape") {
                    if shape.len() == 1 && *dtype == ScalarType::Int32 {
                        let ndims = shape[0];
                        if let Some(data) = data.as_bytes_mut() {
                            if ndims >= 4 && data.len() >= ndims * 4 {
                                let off = (ndims - 1) * 4;
                                let last = i32::from_le_bytes([
                                    data[off],
                                    data[off + 1],
                                    data[off + 2],
                                    data[off + 3],
                                ]);
                                if last > 0 && (last as usize) < min_seq {
                                    let b = (min_seq as i32).to_le_bytes();
                                    data[off] = b[0];
                                    data[off + 1] = b[1];
                                    data[off + 2] = b[2];
                                    data[off + 3] = b[3];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
