//! MIL IR construction for TurboQuant ANE sub-programs.
//!
//! Builds `mil_rs::ir::Program` objects for the cache-write (quantization)
//! and cache-read + attention (dequantization + SDPA) sub-programs that run
//! on the Apple Neural Engine. Programs are serialized to MIL text via
//! `ironmill_compile::ane::mil_text::program_to_mil_text`.

use half::f16;
use mil_rs::ir::passes::beta_quantizer::beta_optimal_levels;
use mil_rs::ir::passes::rotation::{rotate_rows_hadamard, unrotate_rows_hadamard};
use mil_rs::ir::{Block, Function, Operation, Program, ScalarType, TensorType, Value};

use super::model::TurboQuantConfig;

/// Minimum spatial dimension for ANE I/O tensors with high channel counts.
///
/// ANE rejects `[1, C, 1, S]` tensors when C > ~768 and S < 32.
/// Padding S to 32 ensures the shape is always accepted, regardless
/// of the channel count. Only column 0 carries real data; the other
/// 31 columns are zero-padded.
pub(crate) const MIN_IO_SEQ: usize = 32;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create an inline int32 tensor value for use as an op input (shape, axes, etc.).
fn int32_tensor(values: &[i32]) -> Value {
    Value::Tensor {
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        shape: vec![values.len()],
        dtype: ScalarType::Int32,
    }
}

/// Add a scalar const op to a block.
fn add_const_op(block: &mut Block, name: &str, val: Value) {
    let op = Operation::new("const", name)
        .with_input("val", val)
        .with_output(name);
    block.add_op(op);
}

/// Add an int32 tensor const op to a block.
fn add_const_tensor_op(block: &mut Block, name: &str, values: &[i32]) {
    let op = Operation::new("const", name)
        .with_input("val", int32_tensor(values))
        .with_output(name);
    block.add_op(op);
}

/// Generate the Hadamard rotation matrix as fp16 bytes.
///
/// Returns a `head_dim × head_dim` rotation matrix converted to fp16 LE bytes.
pub(crate) fn generate_rotation_weights(head_dim: usize, seed: u64) -> Vec<u8> {
    let mut identity = vec![0.0f32; head_dim * head_dim];
    for i in 0..head_dim {
        identity[i * head_dim + i] = 1.0;
    }
    rotate_rows_hadamard(&mut identity, head_dim, head_dim, seed);
    identity
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}

/// Generate the inverse (un-rotation) matrix as fp16 bytes.
///
/// Used by QJL residual sign computation (CPU-side) when QJL is enabled.
#[allow(dead_code)]
fn generate_unrotation_weights(head_dim: usize, seed: u64) -> Vec<u8> {
    let mut identity = vec![0.0f32; head_dim * head_dim];
    for i in 0..head_dim {
        identity[i * head_dim + i] = 1.0;
    }
    unrotate_rows_hadamard(&mut identity, head_dim, head_dim, seed);
    identity
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}

/// Compute the quantization scale for mapping Beta-optimal levels into INT8.
///
/// `inv_scale = 127.0 / max_level` where `max_level` is the largest
/// Beta-optimal reconstruction level for the given dimension and bit-width.
pub(crate) fn compute_inv_scale(head_dim: usize, n_bits: u8) -> f32 {
    let levels = beta_optimal_levels(head_dim, n_bits);
    let max_level = levels.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if max_level == 0.0 {
        1.0
    } else {
        127.0 / max_level
    }
}

/// Compute the dequantization scale (inverse of `inv_scale`).
pub fn compute_deq_scale(head_dim: usize, n_bits: u8) -> f32 {
    let inv = compute_inv_scale(head_dim, n_bits);
    if inv == 0.0 { 1.0 } else { 1.0 / inv }
}

// ---------------------------------------------------------------------------
// Cache-write sub-program
// ---------------------------------------------------------------------------

/// Build the MIL IR program for the cache-write sub-program.
///
/// Takes fp16 K/V projections and a rotation matrix, applies Hadamard
/// rotation + Beta-optimal quantization, outputs fp16 values clamped
/// to INT8 range (cast to INT8 → back to fp16 since ANE rejects INT8
/// function outputs).
///
/// # Inputs
///
/// - `a_input0`: K projection `[1, kv_ch, 1, MIN_IO_SEQ]` fp16
/// - `a_input1`: V projection `[1, kv_ch, 1, MIN_IO_SEQ]` fp16
/// - `a_input2`: Rotation matrix `[1, 1, head_dim, head_dim]` fp16
///
/// # Returns
///
/// `(program, weights)` where `weights` contains the rotation matrix
/// fp16 bytes to populate the input IOSurface tensor.
pub fn build_cache_write_program(config: &TurboQuantConfig) -> (Program, Vec<(String, Vec<u8>)>) {
    let ch = config.num_kv_heads * config.head_dim;
    let s = MIN_IO_SEQ;
    let inv_scale = compute_inv_scale(config.head_dim, config.n_bits);

    let mut weights: Vec<(String, Vec<u8>)> = Vec::new();

    // Rotation matrix passed as function input, not BLOBFILE.
    let rot_data = generate_rotation_weights(config.head_dim, config.rotation_seed);
    weights.push(("rotation_matrix".to_string(), rot_data));

    let in_ty = TensorType::new(ScalarType::Float16, vec![1, ch, 1, s]);
    let rot_ty = TensorType::new(
        ScalarType::Float16,
        vec![1, 1, config.head_dim, config.head_dim],
    );

    let mut func = Function::new("main")
        .with_input("k_proj", in_ty.clone())
        .with_input("v_proj", in_ty)
        .with_input("rot_mat", rot_ty);

    // Shared constants
    add_const_op(&mut func.body, "inv_scale", Value::Float(inv_scale as f64));
    add_const_op(&mut func.body, "zero_point", Value::Float(0.0));
    add_const_op(&mut func.body, "clip_lo", Value::Float(-128.0));
    add_const_op(&mut func.body, "clip_hi", Value::Float(127.0));
    add_const_op(&mut func.body, "bF", Value::Bool(false));

    // K pipeline
    build_quantize_chain(&mut func.body, "k", "k_proj", config, "k_out");
    // V pipeline
    build_quantize_chain(&mut func.body, "v", "v_proj", config, "v_out");

    func.body.outputs.push("k_out".into());
    func.body.outputs.push("v_out".into());

    let mut program = Program::new("1.0.0");
    program.add_function(func);

    (program, weights)
}

/// Build the quantization op chain for a single K or V tensor.
fn build_quantize_chain(
    block: &mut Block,
    prefix: &str,
    input_name: &str,
    config: &TurboQuantConfig,
    output_name: &str,
) {
    let ch = config.num_kv_heads * config.head_dim;
    let s = MIN_IO_SEQ;
    let reshape_4d = vec![1, config.num_kv_heads, config.head_dim, s];
    let flat_shape = vec![1, ch, 1, s];

    // Reshape to [1, num_kv_heads, head_dim, S] for per-head rotation
    let reshaped = format!("{prefix}_reshaped");
    let mut op = Operation::new("reshape", &reshaped)
        .with_input("x", Value::Reference(input_name.into()))
        .with_input(
            "shape",
            int32_tensor(&[
                1,
                config.num_kv_heads as i32,
                config.head_dim as i32,
                s as i32,
            ]),
        )
        .with_output(&reshaped);
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, reshape_4d.clone()));
    block.add_op(op);

    // matmul: rotation × reshaped (broadcast over heads)
    let rotated = format!("{prefix}_rotated");
    let mut op = Operation::new("matmul", &rotated)
        .with_input("x", Value::Reference("rot_mat".into()))
        .with_input("y", Value::Reference(reshaped))
        .with_input("transpose_x", Value::Reference("bF".into()))
        .with_input("transpose_y", Value::Reference("bF".into()))
        .with_output(&rotated);
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, reshape_4d));
    block.add_op(op);

    // Reshape back to [1, ch, 1, S]
    let flat = format!("{prefix}_flat");
    let mut op = Operation::new("reshape", &flat)
        .with_input("x", Value::Reference(rotated))
        .with_input("shape", int32_tensor(&[1, ch as i32, 1, s as i32]))
        .with_output(&flat);
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, flat_shape.clone()));
    block.add_op(op);

    // mul: scale to INT8 range
    let scaled = format!("{prefix}_scaled");
    let op = Operation::new("mul", &scaled)
        .with_input("x", Value::Reference(flat))
        .with_input("y", Value::Reference("inv_scale".into()))
        .with_output(&scaled);
    block.add_op(op);

    // add: apply zero point
    let shifted = format!("{prefix}_shifted");
    let op = Operation::new("add", &shifted)
        .with_input("x", Value::Reference(scaled))
        .with_input("y", Value::Reference("zero_point".into()))
        .with_output(&shifted);
    block.add_op(op);

    // round
    let rounded = format!("{prefix}_rounded");
    let op = Operation::new("round", &rounded)
        .with_input("x", Value::Reference(shifted))
        .with_output(&rounded);
    block.add_op(op);

    // clip to [-128, 127]
    let clamped = format!("{prefix}_clamped");
    let op = Operation::new("clip", &clamped)
        .with_input("x", Value::Reference(rounded))
        .with_input("alpha", Value::Reference("clip_lo".into()))
        .with_input("beta", Value::Reference("clip_hi".into()))
        .with_output(&clamped);
    block.add_op(op);

    // cast to int8 then back to fp16 for function output
    // (ANE rejects INT8 function outputs; the fp16 values are already
    // rounded/clamped to [-128, 127] so the cast is lossless)
    let int8_name = format!("{prefix}_int8");
    let mut op = Operation::new("cast", &int8_name)
        .with_input("x", Value::Reference(clamped))
        .with_input("dtype", Value::String("int8".into()))
        .with_output(&int8_name);
    op.output_types[0] = Some(TensorType::new(ScalarType::Int8, flat_shape.clone()));
    block.add_op(op);

    let mut op = Operation::new("cast", output_name)
        .with_input("x", Value::Reference(int8_name))
        .with_input("dtype", Value::String("fp16".into()))
        .with_output(output_name);
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, flat_shape));
    block.add_op(op);
}

// ---------------------------------------------------------------------------
// Attention sub-program
// ---------------------------------------------------------------------------

/// Configuration for the unified attention MIL emitter.
///
/// Controls whether dequantization and rotation stages are included
/// in the generated MIL program.
pub struct AttentionMilConfig {
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of KV attention heads (may differ for GQA).
    pub num_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Maximum sequence length (cache allocation size).
    pub max_seq_len: usize,
    /// Current sequence length to slice from the cache.
    pub seq_len: usize,
    /// When `Some(scale)`, emit `mul(deq_scale)` after slicing the cache.
    pub dequant_scale: Option<f32>,
    /// When `Some(seed)`, emit Hadamard rotation for Q and output un-rotation.
    /// Uses the Q-rotation approach: rotate Q by R (O(1) per token) instead
    /// of un-rotating the entire K/V cache by R⁻¹ (O(seq_len) per token).
    /// The function will take an additional 4th input (rotation matrix)
    /// and return the matrix as a weight blob.
    pub unrotation_seed: Option<u64>,
    /// When true, cache inputs are INT8 tensors and a `cast(int8→fp16)`
    /// is emitted before any dequantization. This halves KV cache memory.
    pub cache_int8: bool,
    /// When true, the attention program accepts an additional QJL correction
    /// input `[1, num_heads, 1, seq_len]` and adds it to the scaled QK scores
    /// before the causal mask and softmax. This corrects inner product bias
    /// from TurboQuant quantization (TurboQuant Stage 2).
    pub enable_qjl: bool,
}

/// Build the MIL IR program for an attention sub-program.
///
/// Produces a unified program that handles TurboQuant and plain FP16
/// attention based on the config:
///
/// - **TurboQuant (INT8 cache, Q-rotation)**: `cache_int8: true, dequant_scale: Some(..), unrotation_seed: Some(..)`
///   → 5 inputs: Q(fp16), K_cache(int8), V_cache(int8), rotation_matrix(fp16), mask(fp16)
///   → pipeline: slice → cast(int8→fp16) → mul(deq_scale) → rotate Q → attention → un-rotate output
///
/// - **TurboQuant + QJL**: as above, plus `enable_qjl: true`
///   → 6 inputs: Q, K_cache, V_cache, rotation_matrix, mask, qjl_correction(fp16)
///   → QJL correction is added to scaled QK scores before mask and softmax
///
/// - **FP16 baseline**: `cache_int8: false, dequant_scale: None, unrotation_seed: None`
///   → 4 inputs: Q(fp16), K_cache(fp16), V_cache(fp16), mask(fp16)
///   → pipeline: slice → attention
///
/// # Returns
///
/// `(program, weights)` where `weights` contains the rotation matrix
/// when rotation is enabled (empty otherwise).
pub fn build_attention_program(config: &AttentionMilConfig) -> (Program, Vec<(String, Vec<u8>)>) {
    let head_dim = config.head_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let seq_len = config.seq_len;
    let kv_ch = num_kv_heads * head_dim;
    let q_ch = num_heads * head_dim;
    let s = MIN_IO_SEQ;
    let gqa_groups = num_heads / num_kv_heads;
    let scale_factor = 1.0 / (head_dim as f32).sqrt();

    let cache_dtype = if config.cache_int8 {
        ScalarType::Int8
    } else {
        ScalarType::Float16
    };

    let q_shape = vec![1, q_ch, 1, s];
    let cache_shape = vec![1, kv_ch, 1, config.max_seq_len];
    let sliced_shape = vec![1, kv_ch, 1, seq_len];
    let kv_head_4d = vec![1, num_kv_heads, head_dim, seq_len];
    let attn_head_4d = vec![1, num_heads, head_dim, seq_len];
    let q_head_4d = vec![1, num_heads, head_dim, s];
    let qk_shape = vec![1, num_heads, s, seq_len];
    let mask_shape = vec![1, 1, 1, seq_len];
    let rot_shape = vec![1, 1, head_dim, head_dim];

    let mut weights: Vec<(String, Vec<u8>)> = Vec::new();

    // Build function inputs (order determines a_input{i} naming)
    let mut func = Function::new("main")
        .with_input("q", TensorType::new(ScalarType::Float16, q_shape.clone()))
        .with_input("k_cache", TensorType::new(cache_dtype, cache_shape.clone()))
        .with_input("v_cache", TensorType::new(cache_dtype, cache_shape));

    if let Some(seed) = config.unrotation_seed {
        let rot_data = generate_rotation_weights(head_dim, seed);
        weights.push(("rotation_matrix".to_string(), rot_data));
        func = func.with_input("rot_mat", TensorType::new(ScalarType::Float16, rot_shape));
    }

    func = func.with_input("mask", TensorType::new(ScalarType::Float16, mask_shape));

    // Optional QJL correction input
    let qjl_correction_shape = vec![1, num_heads, 1, seq_len];
    if config.enable_qjl {
        func = func.with_input(
            "qjl_correction",
            TensorType::new(ScalarType::Float16, qjl_correction_shape),
        );
    }

    // --- Constants ---
    if let Some(deq) = config.dequant_scale {
        add_const_op(&mut func.body, "deq_scale", Value::Float(deq as f64));
    }
    add_const_op(
        &mut func.body,
        "scale_factor",
        Value::Float(scale_factor as f64),
    );
    add_const_op(&mut func.body, "bF", Value::Bool(false));
    add_const_op(&mut func.body, "bT", Value::Bool(true));
    add_const_op(&mut func.body, "softmax_axis", Value::Int(-1));

    // slice_by_index constants
    add_const_tensor_op(&mut func.body, "slice_begin", &[0, 0, 0, 0]);
    add_const_tensor_op(
        &mut func.body,
        "slice_end_k",
        &[1, kv_ch as i32, 1, seq_len as i32],
    );

    // --- Slice K and V caches to [1, kv_ch, 1, seq_len] ---
    if config.cache_int8 {
        // INT8 cache: slice as int8, then cast to fp16
        let mut op = Operation::new("slice_by_index", "k_sliced_i8")
            .with_input("x", Value::Reference("k_cache".into()))
            .with_input("begin", Value::Reference("slice_begin".into()))
            .with_input("end", Value::Reference("slice_end_k".into()))
            .with_output("k_sliced_i8");
        op.output_types[0] = Some(TensorType::new(ScalarType::Int8, sliced_shape.clone()));
        func.body.add_op(op);

        let mut op = Operation::new("cast", "k_sliced")
            .with_input("x", Value::Reference("k_sliced_i8".into()))
            .with_input("dtype", Value::String("fp16".into()))
            .with_output("k_sliced");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, sliced_shape.clone()));
        func.body.add_op(op);

        let mut op = Operation::new("slice_by_index", "v_sliced_i8")
            .with_input("x", Value::Reference("v_cache".into()))
            .with_input("begin", Value::Reference("slice_begin".into()))
            .with_input("end", Value::Reference("slice_end_k".into()))
            .with_output("v_sliced_i8");
        op.output_types[0] = Some(TensorType::new(ScalarType::Int8, sliced_shape.clone()));
        func.body.add_op(op);

        let mut op = Operation::new("cast", "v_sliced")
            .with_input("x", Value::Reference("v_sliced_i8".into()))
            .with_input("dtype", Value::String("fp16".into()))
            .with_output("v_sliced");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, sliced_shape.clone()));
        func.body.add_op(op);
    } else {
        // FP16 cache: slice directly
        let mut op = Operation::new("slice_by_index", "k_sliced")
            .with_input("x", Value::Reference("k_cache".into()))
            .with_input("begin", Value::Reference("slice_begin".into()))
            .with_input("end", Value::Reference("slice_end_k".into()))
            .with_output("k_sliced");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, sliced_shape.clone()));
        func.body.add_op(op);

        let mut op = Operation::new("slice_by_index", "v_sliced")
            .with_input("x", Value::Reference("v_cache".into()))
            .with_input("begin", Value::Reference("slice_begin".into()))
            .with_input("end", Value::Reference("slice_end_k".into()))
            .with_output("v_sliced");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, sliced_shape.clone()));
        func.body.add_op(op);
    }

    // --- Optional: dequantization ---
    let (k_ready, v_ready) = if config.dequant_scale.is_some() {
        let mut op = Operation::new("mul", "k_dscaled")
            .with_input("x", Value::Reference("k_sliced".into()))
            .with_input("y", Value::Reference("deq_scale".into()))
            .with_output("k_dscaled");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, sliced_shape.clone()));
        func.body.add_op(op);

        let mut op = Operation::new("mul", "v_dscaled")
            .with_input("x", Value::Reference("v_sliced".into()))
            .with_input("y", Value::Reference("deq_scale".into()))
            .with_output("v_dscaled");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, sliced_shape));
        func.body.add_op(op);

        ("k_dscaled", "v_dscaled")
    } else {
        ("k_sliced", "v_sliced")
    };

    // --- Attention computation ---

    // Reshape Q: [1, q_ch, 1, S] → [1, num_heads, head_dim, S]
    let mut op = Operation::new("reshape", "q_reshaped")
        .with_input("x", Value::Reference("q".into()))
        .with_input(
            "shape",
            int32_tensor(&[1, num_heads as i32, head_dim as i32, s as i32]),
        )
        .with_output("q_reshaped");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, q_head_4d.clone()));
    func.body.add_op(op);

    // Optional Q-rotation (O(1) per token)
    let q_attn_name = if config.unrotation_seed.is_some() {
        let mut op = Operation::new("matmul", "q_rotated")
            .with_input("x", Value::Reference("rot_mat".into()))
            .with_input("y", Value::Reference("q_reshaped".into()))
            .with_input("transpose_x", Value::Reference("bF".into()))
            .with_input("transpose_y", Value::Reference("bF".into()))
            .with_output("q_rotated");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, q_head_4d));
        func.body.add_op(op);
        "q_rotated"
    } else {
        "q_reshaped"
    };

    // Reshape K: [1, kv_ch, 1, seq_len] → [1, num_kv_heads, head_dim, seq_len]
    let mut op = Operation::new("reshape", "k_heads")
        .with_input("x", Value::Reference(k_ready.into()))
        .with_input(
            "shape",
            int32_tensor(&[1, num_kv_heads as i32, head_dim as i32, seq_len as i32]),
        )
        .with_output("k_heads");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, kv_head_4d.clone()));
    func.body.add_op(op);

    // Reshape V
    let mut op = Operation::new("reshape", "v_heads")
        .with_input("x", Value::Reference(v_ready.into()))
        .with_input(
            "shape",
            int32_tensor(&[1, num_kv_heads as i32, head_dim as i32, seq_len as i32]),
        )
        .with_output("v_heads");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, kv_head_4d));
    func.body.add_op(op);

    // GQA head expansion: repeat_interleave each KV head for its Q heads.
    // tile() repeats the whole block ([kv0..kv7, kv0..kv7]), but GQA needs
    // each head repeated consecutively ([kv0,kv0, kv1,kv1, ..., kv7,kv7]).
    // Implement via reshape→tile→reshape:
    //   [1, nkv, hd, seq] → [1, nkv, 1, hd, seq] → tile [1,1,G,1,1]
    //   → [1, nkv, G, hd, seq] → reshape [1, nh, hd, seq]
    let (k_attn, v_attn) = if gqa_groups > 1 {
        let kv_unsqueezed = vec![1, num_kv_heads, 1, head_dim, seq_len];
        let kv_tiled_5d = vec![1, num_kv_heads, gqa_groups, head_dim, seq_len];

        // Reshape K to 5D
        add_const_tensor_op(
            &mut func.body,
            "k_unsqueeze_shape",
            &kv_unsqueezed.iter().map(|&x| x as i32).collect::<Vec<_>>(),
        );
        let mut op = Operation::new("reshape", "k_unsqueeze")
            .with_input("x", Value::Reference("k_heads".into()))
            .with_input("shape", Value::Reference("k_unsqueeze_shape".into()))
            .with_output("k_unsqueeze");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, kv_unsqueezed.clone()));
        func.body.add_op(op);

        // Tile along the new axis (axis 2)
        add_const_tensor_op(&mut func.body, "gqa_reps", &[1, 1, gqa_groups as i32, 1, 1]);
        let mut op = Operation::new("tile", "k_tiled")
            .with_input("x", Value::Reference("k_unsqueeze".into()))
            .with_input("reps", Value::Reference("gqa_reps".into()))
            .with_output("k_tiled");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, kv_tiled_5d.clone()));
        func.body.add_op(op);

        // Reshape back to 4D: [1, nh, hd, seq]
        add_const_tensor_op(
            &mut func.body,
            "attn_head_shape",
            &attn_head_4d.iter().map(|&x| x as i32).collect::<Vec<_>>(),
        );
        let mut op = Operation::new("reshape", "k_attn")
            .with_input("x", Value::Reference("k_tiled".into()))
            .with_input("shape", Value::Reference("attn_head_shape".into()))
            .with_output("k_attn");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, attn_head_4d.clone()));
        func.body.add_op(op);

        // Same for V
        let mut op = Operation::new("reshape", "v_unsqueeze")
            .with_input("x", Value::Reference("v_heads".into()))
            .with_input("shape", Value::Reference("k_unsqueeze_shape".into()))
            .with_output("v_unsqueeze");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, kv_unsqueezed));
        func.body.add_op(op);

        let mut op = Operation::new("tile", "v_tiled")
            .with_input("x", Value::Reference("v_unsqueeze".into()))
            .with_input("reps", Value::Reference("gqa_reps".into()))
            .with_output("v_tiled");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, kv_tiled_5d));
        func.body.add_op(op);

        let mut op = Operation::new("reshape", "v_attn")
            .with_input("x", Value::Reference("v_tiled".into()))
            .with_input("shape", Value::Reference("attn_head_shape".into()))
            .with_output("v_attn");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, attn_head_4d));
        func.body.add_op(op);

        ("k_attn", "v_attn")
    } else {
        ("k_heads", "v_heads")
    };

    // QK = matmul(Q^T, K) → [1, num_heads, S, seq_len]
    let mut op = Operation::new("matmul", "qk")
        .with_input("x", Value::Reference(q_attn_name.into()))
        .with_input("y", Value::Reference(k_attn.into()))
        .with_input("transpose_x", Value::Reference("bT".into()))
        .with_input("transpose_y", Value::Reference("bF".into()))
        .with_output("qk");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, qk_shape.clone()));
    func.body.add_op(op);

    // Scale QK
    let mut op = Operation::new("mul", "qk_scaled")
        .with_input("x", Value::Reference("qk".into()))
        .with_input("y", Value::Reference("scale_factor".into()))
        .with_output("qk_scaled");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, qk_shape.clone()));
    func.body.add_op(op);

    // Optional QJL bias correction (added before mask and softmax)
    let qk_pre_mask = if config.enable_qjl {
        let mut op = Operation::new("add", "qk_corrected")
            .with_input("x", Value::Reference("qk_scaled".into()))
            .with_input("y", Value::Reference("qjl_correction".into()))
            .with_output("qk_corrected");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, qk_shape.clone()));
        func.body.add_op(op);
        "qk_corrected"
    } else {
        "qk_scaled"
    };

    // Apply causal mask
    let mut op = Operation::new("add", "qk_masked")
        .with_input("x", Value::Reference(qk_pre_mask.into()))
        .with_input("y", Value::Reference("mask".into()))
        .with_output("qk_masked");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, qk_shape.clone()));
    func.body.add_op(op);

    // Softmax
    let mut op = Operation::new("softmax", "attn_weights")
        .with_input("x", Value::Reference("qk_masked".into()))
        .with_input("axis", Value::Reference("softmax_axis".into()))
        .with_output("attn_weights");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, qk_shape));
    func.body.add_op(op);

    // attn_out = V × attn_weights^T → [1, num_heads, head_dim, S]
    let attn_pre_shape = vec![1, num_heads, head_dim, s];
    let mut op = Operation::new("matmul", "attn_pre")
        .with_input("x", Value::Reference(v_attn.into()))
        .with_input("y", Value::Reference("attn_weights".into()))
        .with_input("transpose_x", Value::Reference("bF".into()))
        .with_input("transpose_y", Value::Reference("bT".into()))
        .with_output("attn_pre");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, attn_pre_shape.clone()));
    func.body.add_op(op);

    // Optional output un-rotation
    let final_attn = if config.unrotation_seed.is_some() {
        let mut op = Operation::new("matmul", "attn_unrot")
            .with_input("x", Value::Reference("rot_mat".into()))
            .with_input("y", Value::Reference("attn_pre".into()))
            .with_input("transpose_x", Value::Reference("bT".into()))
            .with_input("transpose_y", Value::Reference("bF".into()))
            .with_output("attn_unrot");
        op.output_types[0] = Some(TensorType::new(ScalarType::Float16, attn_pre_shape));
        func.body.add_op(op);
        "attn_unrot"
    } else {
        "attn_pre"
    };

    // Reshape to output: [1, q_ch, 1, S]
    let mut op = Operation::new("reshape", "attn_out")
        .with_input("x", Value::Reference(final_attn.into()))
        .with_input("shape", int32_tensor(&[1, q_ch as i32, 1, s as i32]))
        .with_output("attn_out");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, q_shape));
    func.body.add_op(op);

    func.body.outputs.push("attn_out".into());

    let mut program = Program::new("1.0.0");
    program.add_function(func);

    (program, weights)
}

/// Build the MIL IR program for FP16 attention (convenience wrapper).
///
/// Equivalent to calling `build_attention_program` with no dequantization or
/// unrotation. Operates on FP16 K/V cache tensors directly.
///
/// # Inputs
/// - `a_input0`: Q (rotated), `[1, q_ch, 1, MIN_IO_SEQ]` fp16
/// - `a_input1`: K cache, `[1, kv_ch, 1, max_seq_len]` fp16
/// - `a_input2`: V cache, `[1, kv_ch, 1, max_seq_len]` fp16
/// - `a_input3`: Mask, `[1, 1, 1, seq_len]` fp16
///
/// # Output
/// - `z_output0`: attention output, `[1, q_ch, 1, MIN_IO_SEQ]` fp16
pub fn build_fp16_attention_program(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
) -> Program {
    let config = AttentionMilConfig {
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len,
        seq_len,
        dequant_scale: None,
        unrotation_seed: None,
        cache_int8: false,
        enable_qjl: false,
    };
    let (program, _weights) = build_attention_program(&config);
    program
}

// ---------------------------------------------------------------------------
// QJL correction sub-program
// ---------------------------------------------------------------------------

/// Build the MIL IR program for the QJL 1-bit bias correction sub-program.
///
/// Takes raw Q tensor and pre-stored residual signs, computes
/// the JL-based correction to attention logits.
///
/// # Arguments
///
/// * `config` – TurboQuant configuration.
/// * `seq_len` – Current sequence length for the residual sign buffer.
///
/// # Returns
///
/// `(program, weights)` — no weight blobs needed (only scalar consts).
pub fn build_qjl_program(
    config: &TurboQuantConfig,
    seq_len: usize,
) -> (Program, Vec<(String, Vec<u8>)>) {
    let head_dim = config.head_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let q_ch = num_heads * head_dim;
    let kv_ch = num_kv_heads * head_dim;

    let q_shape = vec![1, q_ch, 1, MIN_IO_SEQ];
    let q_col0_shape = vec![1, q_ch, 1, 1];
    let q_head_shape = vec![1, num_heads, head_dim, 1];
    let residual_shape = vec![1, kv_ch, 1, seq_len];
    let residual_head_shape = vec![1, num_kv_heads, head_dim, seq_len];
    let correction_shape = vec![1, num_heads, 1, seq_len];

    let qjl_scale = 1.0 / (head_dim as f32).sqrt();

    let mut func = Function::new("main")
        .with_input("q", TensorType::new(ScalarType::Float16, q_shape))
        .with_input(
            "residual",
            TensorType::new(ScalarType::Float16, residual_shape),
        );

    // Constants
    add_const_op(&mut func.body, "zero", Value::Float(0.0));
    add_const_op(&mut func.body, "pos_one", Value::Float(1.0));
    add_const_op(&mut func.body, "neg_one", Value::Float(-1.0));
    add_const_op(&mut func.body, "qjl_scale", Value::Float(qjl_scale as f64));
    add_const_op(&mut func.body, "bF", Value::Bool(false));
    add_const_op(&mut func.body, "bT", Value::Bool(true));

    // Slice Q to column 0: [1, q_ch, 1, MIN_IO_SEQ] → [1, q_ch, 1, 1]
    add_const_tensor_op(&mut func.body, "q_slice_begin", &[0, 0, 0, 0]);
    add_const_tensor_op(&mut func.body, "q_slice_end", &[1, q_ch as i32, 1, 1]);
    let mut op = Operation::new("slice_by_index", "q_col0")
        .with_input("x", Value::Reference("q".into()))
        .with_input("begin", Value::Reference("q_slice_begin".into()))
        .with_input("end", Value::Reference("q_slice_end".into()))
        .with_output("q_col0");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, q_col0_shape));
    func.body.add_op(op);

    // Reshape Q: [1, q_ch, 1, 1] → [1, num_heads, head_dim, 1]
    let mut op = Operation::new("reshape", "q_reshaped")
        .with_input("x", Value::Reference("q_col0".into()))
        .with_input(
            "shape",
            int32_tensor(&[1, num_heads as i32, head_dim as i32, 1]),
        )
        .with_output("q_reshaped");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, q_head_shape.clone()));
    func.body.add_op(op);

    // greater(q_reshaped, zero) → q_pos (bool)
    let mut op = Operation::new("greater", "q_pos")
        .with_input("x", Value::Reference("q_reshaped".into()))
        .with_input("y", Value::Reference("zero".into()))
        .with_output("q_pos");
    op.output_types[0] = Some(TensorType::new(ScalarType::Bool, q_head_shape.clone()));
    func.body.add_op(op);

    // select(cond=q_pos, a=pos_one, b=neg_one) → q_sign (fp16)
    let mut op = Operation::new("select", "q_sign")
        .with_input("cond", Value::Reference("q_pos".into()))
        .with_input("a", Value::Reference("pos_one".into()))
        .with_input("b", Value::Reference("neg_one".into()))
        .with_output("q_sign");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, q_head_shape));
    func.body.add_op(op);

    // Reshape residual: [1, kv_ch, 1, seq_len] → [1, num_kv_heads, head_dim, seq_len]
    let mut op = Operation::new("reshape", "residual_reshaped")
        .with_input("x", Value::Reference("residual".into()))
        .with_input(
            "shape",
            int32_tensor(&[1, num_kv_heads as i32, head_dim as i32, seq_len as i32]),
        )
        .with_output("residual_reshaped");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, residual_head_shape));
    func.body.add_op(op);

    // matmul(q_sign^T, residual_reshaped) → correction
    let mut op = Operation::new("matmul", "correction")
        .with_input("x", Value::Reference("q_sign".into()))
        .with_input("y", Value::Reference("residual_reshaped".into()))
        .with_input("transpose_x", Value::Reference("bT".into()))
        .with_input("transpose_y", Value::Reference("bF".into()))
        .with_output("correction");
    op.output_types[0] = Some(TensorType::new(
        ScalarType::Float16,
        correction_shape.clone(),
    ));
    func.body.add_op(op);

    // mul(correction, qjl_scale) → output
    let mut op = Operation::new("mul", "correction_scaled")
        .with_input("x", Value::Reference("correction".into()))
        .with_input("y", Value::Reference("qjl_scale".into()))
        .with_output("correction_scaled");
    op.output_types[0] = Some(TensorType::new(ScalarType::Float16, correction_shape));
    func.body.add_op(op);

    func.body.outputs.push("correction_scaled".into());

    let mut program = Program::new("1.0.0");
    program.add_function(func);

    (program, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironmill_compile::ane::mil_text::{MilTextConfig, program_to_mil_text};

    /// Serialize a Program to MIL text for test assertions.
    fn to_mil(program: &Program) -> String {
        let config = MilTextConfig::default();
        let (text, _) = program_to_mil_text(program, &config).expect("MIL text emission failed");
        text
    }

    fn test_config() -> TurboQuantConfig {
        TurboQuantConfig {
            n_bits: 8,
            max_seq_len: 128,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 64,
            num_layers: 1,
            rotation_seed: 42,
            enable_qjl: false,
        }
    }

    fn test_tq_attn_config() -> AttentionMilConfig {
        let tq = test_config();
        let deq_scale = compute_deq_scale(tq.head_dim, tq.n_bits);
        AttentionMilConfig {
            num_heads: tq.num_heads,
            num_kv_heads: tq.num_kv_heads,
            head_dim: tq.head_dim,
            max_seq_len: tq.max_seq_len,
            seq_len: 32,
            dequant_scale: Some(deq_scale),
            unrotation_seed: Some(tq.rotation_seed),
            cache_int8: true,
            enable_qjl: false,
        }
    }

    #[test]
    fn cache_write_mil_is_valid_program() {
        let config = test_config();
        let (program, weights) = build_cache_write_program(&config);
        let mil = to_mil(&program);

        // Should start with program header
        assert!(mil.starts_with("program(1.3)"));
        // Should contain function declaration
        assert!(mil.contains("func main<ios18>"));
        // Should have three inputs (K, V, rotation matrix)
        assert!(mil.contains("a_input0"));
        assert!(mil.contains("a_input1"));
        assert!(mil.contains("a_input2"));
        // Should have two outputs
        assert!(mil.contains("z_output0"));
        assert!(mil.contains("z_output1"));
        // Should contain quantization ops
        assert!(mil.contains("matmul"));
        assert!(mil.contains("round"));
        assert!(mil.contains("clip"));
        assert!(mil.contains("cast"));
        assert!(mil.contains("int8"));
        // Should produce rotation matrix weight
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].0, "rotation_matrix");
        // Weight should be head_dim * head_dim * 2 bytes (fp16)
        assert_eq!(weights[0].1.len(), 64 * 64 * 2);
    }

    #[test]
    fn attention_mil_is_valid_program() {
        let config = test_tq_attn_config();
        let (program, weights) = build_attention_program(&config);
        let mil = to_mil(&program);

        assert!(mil.starts_with("program(1.3)"));
        assert!(mil.contains("func main<ios18>"));
        // Five inputs: Q, K_cache, V_cache, rotation_matrix, mask
        assert!(mil.contains("a_input0"));
        assert!(mil.contains("a_input1"));
        assert!(mil.contains("a_input2"));
        assert!(mil.contains("a_input3"));
        assert!(mil.contains("a_input4"));
        // One output
        assert!(mil.contains("z_output0"));
        // Cache inputs are INT8 with inline cast to fp16
        assert!(
            mil.contains("int8"),
            "TQ attention should use int8 cache inputs"
        );
        assert!(mil.contains("cast"), "TQ attention should cast int8→fp16");
        // Should contain dequant scale
        assert!(mil.contains("deq_scale"));
        // Should contain attention ops
        assert!(mil.contains("softmax"));
        assert!(mil.contains("matmul"));
        assert!(mil.contains("scale_factor"));
        // Should produce rotation matrix weight (for Q-rotation + output un-rotation)
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].0, "rotation_matrix");
    }

    #[test]
    fn rotation_weight_size_matches_head_dim() {
        let data = generate_rotation_weights(128, 42);
        // 128 * 128 elements * 2 bytes per fp16
        assert_eq!(data.len(), 128 * 128 * 2);
    }

    #[test]
    fn inv_scale_is_positive() {
        let scale = compute_inv_scale(64, 8);
        assert!(scale > 0.0, "inv_scale should be positive, got {scale}");
    }

    #[test]
    fn qjl_correction_mil_is_valid_program() {
        let config = test_config();
        let (program, weights) = build_qjl_program(&config, 32);
        let mil = to_mil(&program);

        // Should start with program header
        assert!(mil.starts_with("program(1.3)"));
        // Should contain function declaration
        assert!(mil.contains("func main<ios18>"));
        // Should have two inputs
        assert!(mil.contains("a_input0"));
        assert!(mil.contains("a_input1"));
        // Should have output
        assert!(mil.contains("z_output0"));
        // Should contain sign extraction ops
        assert!(mil.contains("greater"));
        assert!(mil.contains("select"));
        // Should contain correction ops
        assert!(mil.contains("matmul"));
        assert!(mil.contains("qjl_scale"));
        // No weight blobs needed
        assert!(weights.is_empty());
    }

    #[test]
    fn qjl_correction_mil_various_configs() {
        // Small config
        let small = TurboQuantConfig {
            n_bits: 4,
            max_seq_len: 64,
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 32,
            num_layers: 1,
            rotation_seed: 42,
            enable_qjl: true,
        };
        let (program, weights) = build_qjl_program(&small, 16);
        let mil = to_mil(&program);
        assert!(mil.starts_with("program(1.3)"));
        assert!(mil.contains("func main<ios18>"));
        assert!(weights.is_empty());

        // Large config
        let large = TurboQuantConfig {
            n_bits: 2,
            max_seq_len: 4096,
            num_heads: 64,
            num_kv_heads: 64,
            head_dim: 128,
            num_layers: 32,
            rotation_seed: 123,
            enable_qjl: true,
        };
        let (program, weights) = build_qjl_program(&large, 2048);
        let mil = to_mil(&program);
        assert!(mil.starts_with("program(1.3)"));
        assert!(mil.contains("func main<ios18>"));
        assert!(weights.is_empty());

        // Single head
        let single = TurboQuantConfig {
            n_bits: 8,
            max_seq_len: 32,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 64,
            num_layers: 1,
            rotation_seed: 42,
            enable_qjl: true,
        };
        let (program, weights) = build_qjl_program(&single, 1);
        let mil = to_mil(&program);
        assert!(mil.starts_with("program(1.3)"));
        assert!(mil.contains("func main<ios18>"));
        assert!(weights.is_empty());
    }

    #[test]
    fn attention_mil_gqa_uses_tile() {
        // GQA config: 32 query heads, 8 KV heads (4x expansion)
        let deq_scale = compute_deq_scale(64, 8);
        let config = AttentionMilConfig {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            max_seq_len: 128,
            seq_len: 32,
            dequant_scale: Some(deq_scale),
            unrotation_seed: Some(42),
            cache_int8: true,
            enable_qjl: false,
        };
        let (program, _weights) = build_attention_program(&config);
        let mil = to_mil(&program);

        // Should contain tile op for GQA expansion
        assert!(mil.contains("tile"), "GQA attention should use tile op");
        assert!(mil.contains("gqa_reps"), "GQA should have reps constant");
        // Reps should be [1, 4, 1, 1] for 32/8 = 4x expansion
        assert!(mil.contains("[1,4,1,1]"), "GQA reps should be [1,4,1,1]");
        // Should still have valid program structure
        assert!(mil.contains("softmax"));
        assert!(mil.contains("z_output0"));
    }

    #[test]
    fn attention_mil_mha_no_tile() {
        // MHA config: num_heads == num_kv_heads, no tile needed
        let config = test_tq_attn_config();
        let (program, _weights) = build_attention_program(&config);
        let mil = to_mil(&program);
        assert!(!mil.contains("tile"), "MHA should not use tile op");
        assert!(!mil.contains("gqa_reps"), "MHA should not have gqa_reps");
    }

    #[test]
    fn fp16_attention_mil_is_valid_program() {
        let program = build_fp16_attention_program(32, 32, 64, 128, 128);
        let mil = to_mil(&program);
        assert!(mil.starts_with("program(1.3)"));
        assert!(mil.contains("func main<ios18>"));
        // Four fp16 inputs: Q, K_cache, V_cache, mask
        assert!(mil.contains("a_input0"));
        assert!(mil.contains("a_input1"));
        assert!(mil.contains("a_input2"));
        assert!(mil.contains("a_input3"));
        // All inputs are fp16 (no int8)
        assert!(!mil.contains("int8"), "FP16 attention should not use int8");
        // No dequantization ops
        assert!(
            !mil.contains("deq_scale"),
            "FP16 attention should not dequantize"
        );
        // No unrotation (no 5th input)
        assert!(
            !mil.contains("a_input4"),
            "FP16 attention should have 4 inputs, not 5"
        );
        // Should contain attention ops
        assert!(mil.contains("slice_by_index"));
        assert!(mil.contains("softmax"));
        assert!(mil.contains("matmul"));
        assert!(mil.contains("scale_factor"));
        assert!(mil.contains("z_output0"));
    }

    #[test]
    fn fp16_attention_mil_gqa_uses_tile() {
        // GQA: 32 query heads, 8 KV heads
        let program = build_fp16_attention_program(32, 8, 64, 128, 128);
        let mil = to_mil(&program);
        assert!(
            mil.contains("tile"),
            "FP16 GQA attention should use tile op"
        );
        assert!(mil.contains("gqa_reps"));
        assert!(mil.contains("[1,4,1,1]"));
        assert!(mil.contains("softmax"));
    }

    #[test]
    fn fp16_attention_mil_mha_no_tile() {
        // MHA: num_heads == num_kv_heads
        let program = build_fp16_attention_program(32, 32, 64, 128, 128);
        let mil = to_mil(&program);
        assert!(!mil.contains("tile"), "FP16 MHA should not use tile op");
        assert!(!mil.contains("gqa_reps"));
    }
}
