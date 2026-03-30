//! MIL text generation for TurboQuant ANE sub-programs.
//!
//! Generates MIL text strings for the cache-write (quantization) and
//! cache-read + attention (dequantization + SDPA) sub-programs that run
//! on the Apple Neural Engine.

use half::f16;
use mil_rs::ir::passes::beta_quantizer::beta_optimal_levels;
use mil_rs::ir::passes::rotation::{rotate_rows_hadamard, unrotate_rows_hadamard};
use std::fmt::Write;

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

/// Standard MIL program wrapper used by the ANE compiler.
fn mil_program_wrapper(func_body: &str) -> String {
    // The buildInfo dict uses {{ }} as delimiters in MIL text.
    // In Rust format!, {{ → { and }} → }, so we need quadruple braces
    // for the outer dict delimiters.
    format!(
        "program(1.3)\n\
         [buildInfo = dict<string, string>(\
         {{{{\"coremlc-component-MIL\", \"3510.2.1\"}}, \
         {{\"coremlc-version\", \"3505.4.1\"}}, \
         {{\"coremltools-component-milinternal\", \"\"}}, \
         {{\"coremltools-version\", \"9.0\"}}}})]\n\
         {{\n\
         {func_body}\n\
         }}"
    )
}

/// Format a shape as MIL text, e.g. `[1,128,1,1]`.
fn fmt_shape(shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("[{}]", dims.join(","))
}

/// Generate the Hadamard rotation matrix as fp16 bytes.
///
/// Returns a `head_dim × head_dim` rotation matrix converted to fp16 LE bytes.
fn generate_rotation_weights(head_dim: usize, seed: u64) -> Vec<u8> {
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
fn compute_inv_scale(head_dim: usize, n_bits: u8) -> f32 {
    let levels = beta_optimal_levels(head_dim, n_bits);
    let max_level = levels.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if max_level == 0.0 {
        1.0
    } else {
        127.0 / max_level
    }
}

/// Compute the dequantization scale (inverse of `inv_scale`).
fn compute_deq_scale(head_dim: usize, n_bits: u8) -> f32 {
    let inv = compute_inv_scale(head_dim, n_bits);
    if inv == 0.0 { 1.0 } else { 1.0 / inv }
}

// ---------------------------------------------------------------------------
// Cache-write sub-program
// ---------------------------------------------------------------------------

/// Generate MIL text for the cache-write sub-program.
///
/// Takes fp16 K/V projections and a rotation matrix, applies Hadamard
/// rotation + Beta-optimal quantization, outputs fp16 values clamped
/// to INT8 range (cast to INT8 → back to fp16 since ANE rejects INT8
/// function outputs).
///
/// # Inputs
///
/// - `a_input0`: K projection `[1, kv_ch, 1, 1]` fp16
/// - `a_input1`: V projection `[1, kv_ch, 1, 1]` fp16
/// - `a_input2`: Rotation matrix `[1, 1, head_dim, head_dim]` fp16
///
/// # Returns
///
/// `(mil_text, weights)` where `weights` contains the rotation matrix
/// fp16 bytes to populate the input IOSurface tensor.
pub fn emit_cache_write_mil(config: &TurboQuantConfig) -> (String, Vec<(String, Vec<u8>)>) {
    let ch = config.num_kv_heads * config.head_dim;
    let in_shape = fmt_shape(&[1, ch, 1, MIN_IO_SEQ]);
    let rot_shape = fmt_shape(&[1, 1, config.head_dim, config.head_dim]);

    let inv_scale = compute_inv_scale(config.head_dim, config.n_bits);

    let mut body = String::new();
    let mut weights: Vec<(String, Vec<u8>)> = Vec::new();

    // Rotation matrix passed as function input (a_input2), not BLOBFILE.
    // BLOBFILE weight references fail with compile_mil_text — see
    // docs/development/ane-blobfile-investigation.md for details.
    let rot_data = generate_rotation_weights(config.head_dim, config.rotation_seed);
    weights.push(("rotation_matrix".to_string(), rot_data));

    // --- Build function body ---

    writeln!(
        body,
        "        fp16 inv_scale = const()[name=string(\"inv_scale\"), val=fp16({inv_scale})];",
    )
    .unwrap();

    writeln!(
        body,
        "        fp16 zero_point = const()[name=string(\"zero_point\"), val=fp16(0.0)];"
    )
    .unwrap();

    writeln!(
        body,
        "        fp16 clip_lo = const()[name=string(\"clip_lo\"), val=fp16(-128.0)];"
    )
    .unwrap();

    writeln!(
        body,
        "        fp16 clip_hi = const()[name=string(\"clip_hi\"), val=fp16(127.0)];"
    )
    .unwrap();

    writeln!(
        body,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    )
    .unwrap();

    // --- K pipeline (rotation_matrix = a_input2) ---
    emit_quantize_chain(&mut body, "k", "a_input0", &in_shape, &rot_shape, config);

    // --- V pipeline ---
    emit_quantize_chain(&mut body, "v", "a_input1", &in_shape, &rot_shape, config);

    // Assemble function — rotation matrix is a_input2
    let func = format!(
        "    func main<ios18>(tensor<fp16, {in_shape}> a_input0, \
         tensor<fp16, {in_shape}> a_input1, \
         tensor<fp16, {rot_shape}> a_input2) {{\n\
         {body}\
         \n    }} -> (z_output0, z_output1);"
    );

    let mil_text = mil_program_wrapper(&func);
    (mil_text, weights)
}

/// Emit the quantization op chain for a single K or V tensor.
fn emit_quantize_chain(
    body: &mut String,
    prefix: &str,
    input_name: &str,
    _in_shape: &str,
    rot_shape: &str,
    config: &TurboQuantConfig,
) {
    let ch = config.num_kv_heads * config.head_dim;
    let _ = rot_shape; // rotation applied via matmul broadcast

    let s = MIN_IO_SEQ;

    // Reshape to [1, num_kv_heads, head_dim, S] for per-head rotation
    let reshape_4d = fmt_shape(&[1, config.num_kv_heads, config.head_dim, s]);
    let reshaped = format!("{prefix}_reshaped");
    writeln!(
        body,
        "        tensor<fp16, {reshape_4d}> {reshaped} = reshape(\
         x={input_name}, shape=tensor<int32, [4]>([1,{},{},{s}]))\
         [name=string(\"{reshaped}\")];",
        config.num_kv_heads, config.head_dim
    )
    .unwrap();

    // matmul: [1, 1, head_dim, head_dim] × [1, num_kv_heads, head_dim, 1]
    // -> [1, num_kv_heads, head_dim, 1]  (rotation applied per head via broadcast)
    let rotated = format!("{prefix}_rotated");
    writeln!(
        body,
        "        tensor<fp16, {reshape_4d}> {rotated} = matmul(\
         x=a_input2, y={reshaped}, transpose_x=bF, transpose_y=bF)\
         [name=string(\"{rotated}\")];"
    )
    .unwrap();

    // Reshape back to [1, ch, 1, S]
    let flat_shape = fmt_shape(&[1, ch, 1, s]);
    let flat = format!("{prefix}_flat");
    writeln!(
        body,
        "        tensor<fp16, {flat_shape}> {flat} = reshape(\
         x={rotated}, shape=tensor<int32, [4]>([1,{ch},1,{s}]))\
         [name=string(\"{flat}\")];"
    )
    .unwrap();

    // mul: scale to INT8 range
    let scaled = format!("{prefix}_scaled");
    writeln!(
        body,
        "        tensor<fp16, {flat_shape}> {scaled} = mul(\
         x={flat}, y=inv_scale)\
         [name=string(\"{scaled}\")];"
    )
    .unwrap();

    // add: apply zero point
    let shifted = format!("{prefix}_shifted");
    writeln!(
        body,
        "        tensor<fp16, {flat_shape}> {shifted} = add(\
         x={scaled}, y=zero_point)\
         [name=string(\"{shifted}\")];"
    )
    .unwrap();

    // round
    let rounded = format!("{prefix}_rounded");
    writeln!(
        body,
        "        tensor<fp16, {flat_shape}> {rounded} = round(\
         x={shifted})\
         [name=string(\"{rounded}\")];"
    )
    .unwrap();

    // clip to [-128, 127]
    let clamped = format!("{prefix}_clamped");
    writeln!(
        body,
        "        tensor<fp16, {flat_shape}> {clamped} = clip(\
         x={rounded}, alpha=clip_lo, beta=clip_hi)\
         [name=string(\"{clamped}\")];"
    )
    .unwrap();

    // cast to int8 then back to fp16 for function output
    // (ANE rejects INT8 function outputs; the fp16 values are already
    // rounded/clamped to [-128, 127] so the cast is lossless)
    let int8_name = format!("{prefix}_int8");
    writeln!(
        body,
        "        tensor<int8, {flat_shape}> {int8_name} = cast(\
         x={clamped}, dtype=string(\"int8\"))\
         [name=string(\"{int8_name}\")];"
    )
    .unwrap();

    let output = if prefix == "k" {
        "z_output0"
    } else {
        "z_output1"
    };
    writeln!(
        body,
        "        tensor<fp16, {flat_shape}> {output} = cast(\
         x={int8_name}, dtype=string(\"fp16\"))\
         [name=string(\"{output}\")];"
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Attention sub-program
// ---------------------------------------------------------------------------

/// Generate MIL text for the cache-read + attention sub-program.
///
/// Reads INT8 cached K/V, dequantizes to fp16, un-rotates, and computes
/// scaled dot-product attention.
///
/// # Arguments
///
/// * `config` – TurboQuant configuration.
/// * `seq_len` – Current sequence length to slice from the cache.
///
/// # Returns
///
/// `(mil_text, weights)` where `weights` contains the inverse rotation matrix
/// and dequantization constants.
pub fn emit_attention_mil(
    config: &TurboQuantConfig,
    seq_len: usize,
) -> (String, Vec<(String, Vec<u8>)>) {
    let head_dim = config.head_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let kv_ch = num_kv_heads * head_dim;
    let q_ch = num_heads * head_dim;

    let q_shape = fmt_shape(&[1, q_ch, 1, MIN_IO_SEQ]);
    let cache_shape = fmt_shape(&[1, kv_ch, 1, config.max_seq_len]);
    let sliced_shape = fmt_shape(&[1, kv_ch, 1, seq_len]);
    let rot_shape = fmt_shape(&[1, 1, head_dim, head_dim]);

    let gqa_groups = num_heads / num_kv_heads; // >=1; validated by TurboQuantConfig
    let kv_head_4d = fmt_shape(&[1, num_kv_heads, head_dim, seq_len]);
    let attn_head_4d = fmt_shape(&[1, num_heads, head_dim, seq_len]);
    let q_head_4d = fmt_shape(&[1, num_heads, head_dim, MIN_IO_SEQ]);
    // QK shape: [1, num_heads, MIN_IO_SEQ, seq_len] after matmul with transpose
    let qk_shape = fmt_shape(&[1, num_heads, MIN_IO_SEQ, seq_len]);

    let deq_scale = compute_deq_scale(head_dim, config.n_bits);
    let scale_factor = 1.0 / (head_dim as f32).sqrt();

    let mut body = String::new();
    let mut weights: Vec<(String, Vec<u8>)> = Vec::new();

    // --- Un-rotation matrix delivered as function input (a_input3) ---
    // BLOBFILE weight references fail with compile_mil_text — see
    // docs/development/ane-blobfile-investigation.md
    let unrot_data = generate_unrotation_weights(head_dim, config.rotation_seed);
    weights.push(("unrotation_matrix".to_string(), unrot_data));

    // --- Constants ---
    writeln!(
        body,
        "        fp16 deq_scale = const()[name=string(\"deq_scale\"), val=fp16({deq_scale})];"
    )
    .unwrap();

    writeln!(
        body,
        "        fp16 deq_offset = const()[name=string(\"deq_offset\"), val=fp16(0.0)];"
    )
    .unwrap();

    writeln!(
        body,
        "        fp16 scale_factor = const()[name=string(\"scale_factor\"), val=fp16({scale_factor})];"
    )
    .unwrap();

    writeln!(
        body,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    )
    .unwrap();

    writeln!(
        body,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    )
    .unwrap();

    writeln!(
        body,
        "        int32 softmax_axis = const()[name=string(\"softmax_axis\"), val=int32(-1)];"
    )
    .unwrap();

    // slice_by_index constants
    writeln!(
        body,
        "        tensor<int32, [4]> slice_begin = const()[name=string(\"slice_begin\"), \
         val=tensor<int32, [4]>([0,0,0,0])];"
    )
    .unwrap();

    writeln!(
        body,
        "        tensor<int32, [4]> slice_end_k = const()[name=string(\"slice_end_k\"), \
         val=tensor<int32, [4]>([1,{kv_ch},1,{seq_len}])];"
    )
    .unwrap();

    // --- K dequant pipeline ---
    emit_dequantize_chain(
        &mut body,
        "k",
        "a_input1",
        &cache_shape,
        &sliced_shape,
        &rot_shape,
        config,
        seq_len,
    );

    // --- V dequant pipeline ---
    emit_dequantize_chain(
        &mut body,
        "v",
        "a_input2",
        &cache_shape,
        &sliced_shape,
        &rot_shape,
        config,
        seq_len,
    );

    // --- Attention computation ---

    // Reshape Q: [1, q_ch, 1, S] -> [1, num_heads, head_dim, S]
    let s = MIN_IO_SEQ;
    writeln!(
        body,
        "        tensor<fp16, {q_head_4d}> q_reshaped = reshape(\
         x=a_input0, shape=tensor<int32, [4]>([1,{num_heads},{head_dim},{s}]))\
         [name=string(\"q_reshaped\")];"
    )
    .unwrap();

    // Reshape K_unrotated: [1, kv_ch, 1, seq_len] -> [1, num_kv_heads, head_dim, seq_len]
    writeln!(
        body,
        "        tensor<fp16, {kv_head_4d}> k_heads = reshape(\
         x=k_unrotated, shape=tensor<int32, [4]>([1,{num_kv_heads},{head_dim},{seq_len}]))\
         [name=string(\"k_heads\")];"
    )
    .unwrap();

    // Reshape V_unrotated: [1, kv_ch, 1, seq_len] -> [1, num_kv_heads, head_dim, seq_len]
    writeln!(
        body,
        "        tensor<fp16, {kv_head_4d}> v_heads = reshape(\
         x=v_unrotated, shape=tensor<int32, [4]>([1,{num_kv_heads},{head_dim},{seq_len}]))\
         [name=string(\"v_heads\")];"
    )
    .unwrap();

    // GQA head expansion: tile KV heads to match query heads when num_heads > num_kv_heads
    let (k_attn_name, v_attn_name) = if gqa_groups > 1 {
        writeln!(
            body,
            "        tensor<int32, [4]> gqa_reps = const()[name=string(\"gqa_reps\"), \
             val=tensor<int32, [4]>([1,{gqa_groups},1,1])];"
        )
        .unwrap();

        writeln!(
            body,
            "        tensor<fp16, {attn_head_4d}> k_attn = tile(\
             x=k_heads, reps=gqa_reps)\
             [name=string(\"k_attn\")];"
        )
        .unwrap();

        writeln!(
            body,
            "        tensor<fp16, {attn_head_4d}> v_attn = tile(\
             x=v_heads, reps=gqa_reps)\
             [name=string(\"v_attn\")];"
        )
        .unwrap();

        ("k_attn", "v_attn")
    } else {
        // MHA: num_heads == num_kv_heads, no expansion needed
        ("k_heads", "v_heads")
    };

    // QK = matmul(Q^T, K) -> [1, num_heads, S, seq_len]
    // Q is [1, num_heads, head_dim, S], K is [1, num_heads, head_dim, seq_len]
    // transpose_x=true -> Q^T is [1, num_heads, S, head_dim]
    // Q^T x K -> [1, num_heads, S, seq_len]
    writeln!(
        body,
        "        tensor<fp16, {qk_shape}> qk = matmul(\
         x=q_reshaped, y={k_attn_name}, transpose_x=bT, transpose_y=bF)\
         [name=string(\"qk\")];"
    )
    .unwrap();

    // Scale QK
    writeln!(
        body,
        "        tensor<fp16, {qk_shape}> qk_scaled = mul(\
         x=qk, y=scale_factor)\
         [name=string(\"qk_scaled\")];"
    )
    .unwrap();

    // Softmax
    writeln!(
        body,
        "        tensor<fp16, {qk_shape}> attn_weights = softmax(\
         x=qk_scaled, axis=softmax_axis)\
         [name=string(\"attn_weights\")];"
    )
    .unwrap();

    // attn_out = attn_weights x V^T -> [1, num_heads, S, head_dim]
    // attn_weights is [1, num_heads, S, seq_len]
    // V is [1, num_heads, head_dim, seq_len]
    // V^T is [1, num_heads, seq_len, head_dim]
    // attn_weights x V^T = [1, num_heads, S, head_dim]
    let attn_pre_shape = fmt_shape(&[1, num_heads, s, head_dim]);
    writeln!(
        body,
        "        tensor<fp16, {attn_pre_shape}> attn_pre = matmul(\
         x=attn_weights, y={v_attn_name}, transpose_x=bF, transpose_y=bT)\
         [name=string(\"attn_pre\")];"
    )
    .unwrap();

    // Reshape to output: [1, q_ch, 1, S]
    writeln!(
        body,
        "        tensor<fp16, {q_shape}> z_output0 = reshape(\
         x=attn_pre, shape=tensor<int32, [4]>([1,{q_ch},1,{s}]))\
         [name=string(\"z_output0\")];"
    )
    .unwrap();

    // Assemble function — un-rotation matrix is a_input3
    let func = format!(
        "    func main<ios18>(tensor<fp16, {q_shape}> a_input0, \
         tensor<int8, {cache_shape}> a_input1, \
         tensor<int8, {cache_shape}> a_input2, \
         tensor<fp16, {rot_shape}> a_input3) {{\n\
         {body}\
         \n    }} -> (z_output0);"
    );

    let mil_text = mil_program_wrapper(&func);
    (mil_text, weights)
}

/// Generate MIL text for an FP16 attention sub-program.
///
/// Like TurboQuant's `emit_attention_mil` but operates on FP16 K/V
/// cache tensors directly — no dequantization or unrotation needed.
/// RoPE is applied on CPU before calling this program.
///
/// # Inputs
/// - `a_input0`: Q (rotated), `[1, q_ch, 1, MIN_IO_SEQ]` fp16
/// - `a_input1`: K cache, `[1, kv_ch, 1, max_seq_len]` fp16
/// - `a_input2`: V cache, `[1, kv_ch, 1, max_seq_len]` fp16
///
/// # Output
/// - `z_output0`: attention output, `[1, q_ch, 1, MIN_IO_SEQ]` fp16
pub fn emit_fp16_attention_mil(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
) -> String {
    let kv_ch = num_kv_heads * head_dim;
    let q_ch = num_heads * head_dim;

    let q_shape = fmt_shape(&[1, q_ch, 1, MIN_IO_SEQ]);
    let cache_shape = fmt_shape(&[1, kv_ch, 1, max_seq_len]);
    let sliced_shape = fmt_shape(&[1, kv_ch, 1, seq_len]);

    let gqa_groups = num_heads / num_kv_heads;
    let kv_head_4d = fmt_shape(&[1, num_kv_heads, head_dim, seq_len]);
    let attn_head_4d = fmt_shape(&[1, num_heads, head_dim, seq_len]);
    let q_head_4d = fmt_shape(&[1, num_heads, head_dim, MIN_IO_SEQ]);
    let qk_shape = fmt_shape(&[1, num_heads, MIN_IO_SEQ, seq_len]);

    let scale_factor = 1.0 / (head_dim as f32).sqrt();

    let mut body = String::new();

    // --- Constants ---
    writeln!(
        body,
        "        fp16 scale_factor = const()[name=string(\"scale_factor\"), val=fp16({scale_factor})];"
    )
    .unwrap();

    writeln!(
        body,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    )
    .unwrap();

    writeln!(
        body,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    )
    .unwrap();

    writeln!(
        body,
        "        int32 softmax_axis = const()[name=string(\"softmax_axis\"), val=int32(-1)];"
    )
    .unwrap();

    // slice_by_index constants
    writeln!(
        body,
        "        tensor<int32, [4]> slice_begin = const()[name=string(\"slice_begin\"), \
         val=tensor<int32, [4]>([0,0,0,0])];"
    )
    .unwrap();

    writeln!(
        body,
        "        tensor<int32, [4]> slice_end_k = const()[name=string(\"slice_end_k\"), \
         val=tensor<int32, [4]>([1,{kv_ch},1,{seq_len}])];"
    )
    .unwrap();

    // --- Slice K and V caches to [1, kv_ch, 1, seq_len] ---
    writeln!(
        body,
        "        tensor<fp16, {sliced_shape}> k_sliced = slice_by_index(\
         x=a_input1, begin=slice_begin, end=slice_end_k)\
         [name=string(\"k_sliced\")];"
    )
    .unwrap();

    writeln!(
        body,
        "        tensor<fp16, {sliced_shape}> v_sliced = slice_by_index(\
         x=a_input2, begin=slice_begin, end=slice_end_k)\
         [name=string(\"v_sliced\")];"
    )
    .unwrap();

    // --- Reshape Q: [1, q_ch, 1, S] -> [1, num_heads, head_dim, S] ---
    let s = MIN_IO_SEQ;
    writeln!(
        body,
        "        tensor<fp16, {q_head_4d}> q_reshaped = reshape(\
         x=a_input0, shape=tensor<int32, [4]>([1,{num_heads},{head_dim},{s}]))\
         [name=string(\"q_reshaped\")];"
    )
    .unwrap();

    // Reshape K: [1, kv_ch, 1, seq_len] -> [1, num_kv_heads, head_dim, seq_len]
    writeln!(
        body,
        "        tensor<fp16, {kv_head_4d}> k_heads = reshape(\
         x=k_sliced, shape=tensor<int32, [4]>([1,{num_kv_heads},{head_dim},{seq_len}]))\
         [name=string(\"k_heads\")];"
    )
    .unwrap();

    // Reshape V: [1, kv_ch, 1, seq_len] -> [1, num_kv_heads, head_dim, seq_len]
    writeln!(
        body,
        "        tensor<fp16, {kv_head_4d}> v_heads = reshape(\
         x=v_sliced, shape=tensor<int32, [4]>([1,{num_kv_heads},{head_dim},{seq_len}]))\
         [name=string(\"v_heads\")];"
    )
    .unwrap();

    // GQA head expansion: tile KV heads to match query heads
    let (k_attn_name, v_attn_name) = if gqa_groups > 1 {
        writeln!(
            body,
            "        tensor<int32, [4]> gqa_reps = const()[name=string(\"gqa_reps\"), \
             val=tensor<int32, [4]>([1,{gqa_groups},1,1])];"
        )
        .unwrap();

        writeln!(
            body,
            "        tensor<fp16, {attn_head_4d}> k_attn = tile(\
             x=k_heads, reps=gqa_reps)\
             [name=string(\"k_attn\")];"
        )
        .unwrap();

        writeln!(
            body,
            "        tensor<fp16, {attn_head_4d}> v_attn = tile(\
             x=v_heads, reps=gqa_reps)\
             [name=string(\"v_attn\")];"
        )
        .unwrap();

        ("k_attn", "v_attn")
    } else {
        ("k_heads", "v_heads")
    };

    // QK = matmul(Q^T, K) -> [1, num_heads, S, seq_len]
    writeln!(
        body,
        "        tensor<fp16, {qk_shape}> qk = matmul(\
         x=q_reshaped, y={k_attn_name}, transpose_x=bT, transpose_y=bF)\
         [name=string(\"qk\")];"
    )
    .unwrap();

    // Scale QK
    writeln!(
        body,
        "        tensor<fp16, {qk_shape}> qk_scaled = mul(\
         x=qk, y=scale_factor)\
         [name=string(\"qk_scaled\")];"
    )
    .unwrap();

    // Softmax
    writeln!(
        body,
        "        tensor<fp16, {qk_shape}> attn_weights = softmax(\
         x=qk_scaled, axis=softmax_axis)\
         [name=string(\"attn_weights\")];"
    )
    .unwrap();

    // attn_out = attn_weights x V^T -> [1, num_heads, S, head_dim]
    let attn_pre_shape = fmt_shape(&[1, num_heads, s, head_dim]);
    writeln!(
        body,
        "        tensor<fp16, {attn_pre_shape}> attn_pre = matmul(\
         x=attn_weights, y={v_attn_name}, transpose_x=bF, transpose_y=bT)\
         [name=string(\"attn_pre\")];"
    )
    .unwrap();

    // Reshape to output: [1, q_ch, 1, S]
    writeln!(
        body,
        "        tensor<fp16, {q_shape}> z_output0 = reshape(\
         x=attn_pre, shape=tensor<int32, [4]>([1,{q_ch},1,{s}]))\
         [name=string(\"z_output0\")];"
    )
    .unwrap();

    // Assemble function — 3 inputs: Q, K_cache, V_cache (all fp16)
    let func = format!(
        "    func main<ios18>(tensor<fp16, {q_shape}> a_input0, \
         tensor<fp16, {cache_shape}> a_input1, \
         tensor<fp16, {cache_shape}> a_input2) {{\n\
         {body}\
         \n    }} -> (z_output0);"
    );

    mil_program_wrapper(&func)
}

/// Emit the dequantization op chain for a single K or V cache tensor.
#[allow(clippy::too_many_arguments)]
fn emit_dequantize_chain(
    body: &mut String,
    prefix: &str,
    input_name: &str,
    cache_shape: &str,
    sliced_shape: &str,
    rot_shape: &str,
    config: &TurboQuantConfig,
    seq_len: usize,
) {
    let _ = cache_shape;
    let _ = rot_shape;
    let head_dim = config.head_dim;
    let num_kv_heads = config.num_kv_heads;
    let kv_ch = num_kv_heads * head_dim;

    // slice_by_index: extract [0..seq_len] from the cache
    let sliced = format!("{prefix}_sliced");
    writeln!(
        body,
        "        tensor<int8, {sliced_shape}> {sliced} = slice_by_index(\
         x={input_name}, begin=slice_begin, end=slice_end_k)\
         [name=string(\"{sliced}\")];"
    )
    .unwrap();

    // cast int8 -> fp16
    let fp16_name = format!("{prefix}_fp16");
    writeln!(
        body,
        "        tensor<fp16, {sliced_shape}> {fp16_name} = cast(\
         x={sliced}, dtype=string(\"fp16\"))\
         [name=string(\"{fp16_name}\")];"
    )
    .unwrap();

    // mul: apply dequantization scale
    let scaled = format!("{prefix}_dscaled");
    writeln!(
        body,
        "        tensor<fp16, {sliced_shape}> {scaled} = mul(\
         x={fp16_name}, y=deq_scale)\
         [name=string(\"{scaled}\")];"
    )
    .unwrap();

    // sub: remove offset (zero for symmetric quantization)
    let dequant = format!("{prefix}_dequant");
    writeln!(
        body,
        "        tensor<fp16, {sliced_shape}> {dequant} = sub(\
         x={scaled}, y=deq_offset)\
         [name=string(\"{dequant}\")];"
    )
    .unwrap();

    // Reshape for per-head un-rotation: [1, kv_ch, 1, seq_len] -> [1, num_kv_heads, head_dim, seq_len]
    let head_shape = fmt_shape(&[1, num_kv_heads, head_dim, seq_len]);
    let reshaped = format!("{prefix}_deq_reshaped");
    writeln!(
        body,
        "        tensor<fp16, {head_shape}> {reshaped} = reshape(\
         x={dequant}, shape=tensor<int32, [4]>([1,{num_kv_heads},{head_dim},{seq_len}]))\
         [name=string(\"{reshaped}\")];"
    )
    .unwrap();

    // matmul: un-rotate per head via broadcast
    // [1, 1, head_dim, head_dim] × [1, num_kv_heads, head_dim, seq_len]
    // -> [1, num_kv_heads, head_dim, seq_len]
    let unrotated_head = format!("{prefix}_unrot_head");
    writeln!(
        body,
        "        tensor<fp16, {head_shape}> {unrotated_head} = matmul(\
         x=a_input3, y={reshaped}, transpose_x=bF, transpose_y=bF)\
         [name=string(\"{unrotated_head}\")];"
    )
    .unwrap();

    // Reshape back to [1, kv_ch, 1, seq_len]
    let unrotated = format!("{prefix}_unrotated");
    writeln!(
        body,
        "        tensor<fp16, {sliced_shape}> {unrotated} = reshape(\
         x={unrotated_head}, shape=tensor<int32, [4]>([1,{kv_ch},1,{seq_len}]))\
         [name=string(\"{unrotated}\")];"
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// QJL correction sub-program
// ---------------------------------------------------------------------------

/// Generate MIL text for the QJL 1-bit bias correction sub-program.
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
/// `(mil_text, weights)` — no weight blobs needed (only scalar consts).
pub fn emit_qjl_correction_mil(
    config: &TurboQuantConfig,
    seq_len: usize,
) -> (String, Vec<(String, Vec<u8>)>) {
    let head_dim = config.head_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let q_ch = num_heads * head_dim;
    let kv_ch = num_kv_heads * head_dim;

    let q_shape = fmt_shape(&[1, q_ch, 1, 1]);
    let q_head_shape = fmt_shape(&[1, num_heads, head_dim, 1]);
    let residual_shape = fmt_shape(&[1, kv_ch, 1, seq_len]);
    let residual_head_shape = fmt_shape(&[1, num_kv_heads, head_dim, seq_len]);
    let correction_shape = fmt_shape(&[1, num_heads, 1, seq_len]);

    let qjl_scale = 1.0 / (head_dim as f32).sqrt();

    let mut body = String::new();
    let weights: Vec<(String, Vec<u8>)> = Vec::new();

    // --- Constants ---

    writeln!(
        body,
        "        fp16 zero = const()[name=string(\"zero\"), val=fp16(0)];"
    )
    .unwrap();

    writeln!(
        body,
        "        fp16 pos_one = const()[name=string(\"pos_one\"), val=fp16(1)];"
    )
    .unwrap();

    writeln!(
        body,
        "        fp16 neg_one = const()[name=string(\"neg_one\"), val=fp16(-1)];"
    )
    .unwrap();

    writeln!(
        body,
        "        fp16 qjl_scale = const()[name=string(\"qjl_scale\"), val=fp16({qjl_scale})];"
    )
    .unwrap();

    writeln!(
        body,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    )
    .unwrap();

    writeln!(
        body,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    )
    .unwrap();

    // --- Reshape Q: [1, q_ch, 1, 1] -> [1, num_heads, head_dim, 1] ---

    writeln!(
        body,
        "        tensor<fp16, {q_head_shape}> q_reshaped = reshape(\
         x=a_input0, shape=tensor<int32, [4]>([1,{num_heads},{head_dim},1]))\
         [name=string(\"q_reshaped\")];"
    )
    .unwrap();

    // --- Sign extraction from reshaped Q ---

    // greater(x=q_reshaped, y=zero) → q_pos (bool)
    writeln!(
        body,
        "        tensor<bool, {q_head_shape}> q_pos = greater(\
         x=q_reshaped, y=zero)\
         [name=string(\"q_pos\")];"
    )
    .unwrap();

    // select(cond=q_pos, a=pos_one, b=neg_one) → q_sign (fp16)
    writeln!(
        body,
        "        tensor<fp16, {q_head_shape}> q_sign = select(\
         cond=q_pos, a=pos_one, b=neg_one)\
         [name=string(\"q_sign\")];"
    )
    .unwrap();

    // --- Reshape residual: [1, kv_ch, 1, seq_len] -> [1, num_kv_heads, head_dim, seq_len] ---

    writeln!(
        body,
        "        tensor<fp16, {residual_head_shape}> residual_reshaped = reshape(\
         x=a_input1, shape=tensor<int32, [4]>([1,{num_kv_heads},{head_dim},{seq_len}]))\
         [name=string(\"residual_reshaped\")];"
    )
    .unwrap();

    // --- Correction computation ---

    // matmul(x=q_sign, y=residual_reshaped, transpose_x=true, transpose_y=false)
    // q_sign:            [1, num_heads, head_dim, 1]
    // transpose_x=true:  [1, num_heads, 1, head_dim]
    // residual_reshaped: [1, num_kv_heads, head_dim, seq_len]
    // result:            [1, num_heads, 1, seq_len]
    writeln!(
        body,
        "        tensor<fp16, {correction_shape}> correction = matmul(\
         x=q_sign, y=residual_reshaped, transpose_x=bT, transpose_y=bF)\
         [name=string(\"correction\")];"
    )
    .unwrap();

    // mul(x=correction, y=qjl_scale) → z_output0
    writeln!(
        body,
        "        tensor<fp16, {correction_shape}> z_output0 = mul(\
         x=correction, y=qjl_scale)\
         [name=string(\"z_output0\")];"
    )
    .unwrap();

    // Assemble function
    let func = format!(
        "    func main<ios18>(tensor<fp16, {q_shape}> a_input0, \
         tensor<fp16, {residual_shape}> a_input1) {{\n\
         {body}\
         \n    }} -> (z_output0);"
    );

    let mil_text = mil_program_wrapper(&func);
    (mil_text, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn cache_write_mil_is_valid_program() {
        let config = test_config();
        let (mil, weights) = emit_cache_write_mil(&config);

        // Should start with program header
        assert!(mil.starts_with("program(1.3)"));
        // Should contain function declaration
        assert!(mil.contains("func main<ios18>"));
        // Should have two inputs
        assert!(mil.contains("a_input0"));
        assert!(mil.contains("a_input1"));
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
        let config = test_config();
        let (mil, weights) = emit_attention_mil(&config, 32);

        assert!(mil.starts_with("program(1.3)"));
        assert!(mil.contains("func main<ios18>"));
        // Three inputs: Q, K_cache, V_cache
        assert!(mil.contains("a_input0"));
        assert!(mil.contains("a_input1"));
        assert!(mil.contains("a_input2"));
        // One output
        assert!(mil.contains("z_output0"));
        // Should contain dequant ops
        assert!(mil.contains("slice_by_index"));
        assert!(mil.contains("cast"));
        // Should contain attention ops
        assert!(mil.contains("softmax"));
        assert!(mil.contains("matmul"));
        assert!(mil.contains("scale_factor"));
        // Should produce un-rotation matrix weight
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].0, "unrotation_matrix");
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
        let (mil, weights) = emit_qjl_correction_mil(&config, 32);

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
        let (mil, weights) = emit_qjl_correction_mil(&small, 16);
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
        let (mil, weights) = emit_qjl_correction_mil(&large, 2048);
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
        let (mil, weights) = emit_qjl_correction_mil(&single, 1);
        assert!(mil.starts_with("program(1.3)"));
        assert!(mil.contains("func main<ios18>"));
        assert!(weights.is_empty());
    }

    #[test]
    fn attention_mil_gqa_uses_tile() {
        // GQA config: 32 query heads, 8 KV heads (4x expansion)
        let gqa_config = TurboQuantConfig {
            n_bits: 8,
            max_seq_len: 128,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            num_layers: 1,
            rotation_seed: 42,
            enable_qjl: false,
        };
        let (mil, _weights) = emit_attention_mil(&gqa_config, 32);

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
        let config = test_config();
        let (mil, _weights) = emit_attention_mil(&config, 32);
        assert!(!mil.contains("tile"), "MHA should not use tile op");
        assert!(!mil.contains("gqa_reps"), "MHA should not have gqa_reps");
    }

    #[test]
    fn fp16_attention_mil_is_valid_program() {
        let mil = emit_fp16_attention_mil(32, 32, 64, 128, 128);
        assert!(mil.starts_with("program(1.3)"));
        assert!(mil.contains("func main<ios18>"));
        // Three fp16 inputs: Q, K_cache, V_cache
        assert!(mil.contains("a_input0"));
        assert!(mil.contains("a_input1"));
        assert!(mil.contains("a_input2"));
        // All inputs are fp16 (no int8)
        assert!(!mil.contains("int8"), "FP16 attention should not use int8");
        // No dequantization ops
        assert!(
            !mil.contains("deq_scale"),
            "FP16 attention should not dequantize"
        );
        assert!(
            !mil.contains("deq_offset"),
            "FP16 attention should not dequantize"
        );
        // No unrotation
        assert!(
            !mil.contains("unrotated"),
            "FP16 attention should not unrotate"
        );
        assert!(
            !mil.contains("a_input3"),
            "FP16 attention should have 3 inputs, not 4"
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
        let mil = emit_fp16_attention_mil(32, 8, 64, 128, 128);
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
        let mil = emit_fp16_attention_mil(32, 32, 64, 128, 128);
        assert!(!mil.contains("tile"), "FP16 MHA should not use tile op");
        assert!(!mil.contains("gqa_reps"));
    }
}
