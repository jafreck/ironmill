//! ANE op name fuzzer — discover alternative names for unsupported ops and
//! find undocumented ops that the ANE compiler accepts.
//!
//! Run with: cargo run -p ironmill-ane --example ane_op_fuzz
//!
//! The ANE compiler's MIL parser is a black box. Ops that failed under one
//! name might succeed under an alias. This fuzzer tries hundreds of
//! plausible MIL op names against the compiler to discover what it accepts.

use mil_rs::ffi::ane::AneCompiler;
use std::collections::HashSet;

const BUILD_INFO: &str = r#"[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]"#;

fn mil_program(body: &str, inputs: &str, output: &str) -> String {
    format!(
        "program(1.3)\n{BUILD_INFO}\n{{\n    func main<ios18>({inputs}) {{\n{body}\n    }} -> ({output});\n}}"
    )
}

/// Try a unary fp16→fp16 op
fn try_unary(op: &str) -> bool {
    let mil = mil_program(
        &format!(
            "        tensor<fp16, [1,32,1,32]> z_output0 = {op}(x=a_input0)[name=string(\"z_output0\")];"
        ),
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    AneCompiler::compile_mil_text(&mil, &[]).is_ok()
}

/// Try a binary fp16→fp16 op
fn try_binary(op: &str) -> bool {
    let mil = mil_program(
        &format!(
            "        tensor<fp16, [1,32,1,32]> z_output0 = {op}(x=a_input0, y=a_input1)[name=string(\"z_output0\")];"
        ),
        "tensor<fp16, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1",
        "z_output0",
    );
    AneCompiler::compile_mil_text(&mil, &[]).is_ok()
}

/// Try a unary fp16→bool op
fn try_unary_to_bool(op: &str) -> bool {
    let mil = mil_program(
        &format!(
            "        tensor<bool, [1,32,1,32]> z_output0 = {op}(x=a_input0)[name=string(\"z_output0\")];"
        ),
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    AneCompiler::compile_mil_text(&mil, &[]).is_ok()
}

/// Try a binary fp16→bool op
fn try_binary_to_bool(op: &str) -> bool {
    let mil = mil_program(
        &format!(
            "        tensor<bool, [1,32,1,32]> z_output0 = {op}(x=a_input0, y=a_input1)[name=string(\"z_output0\")];"
        ),
        "tensor<fp16, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1",
        "z_output0",
    );
    AneCompiler::compile_mil_text(&mil, &[]).is_ok()
}

/// Try a reduction op with const axes
fn try_reduce(op: &str) -> bool {
    let mil = mil_program(
        &format!(
            "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n\
                     bool kd = const()[name=string(\"kd\"), val=bool(true)];\n\
                     tensor<fp16, [1,32,1,1]> z_output0 = {op}(x=a_input0, axes=rax, keep_dims=kd)[name=string(\"z_output0\")];"
        ),
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    AneCompiler::compile_mil_text(&mil, &[]).is_ok()
}

/// Try a binary bool→bool op
fn try_binary_bool(op: &str) -> bool {
    let mil = mil_program(
        &format!(
            "        tensor<bool, [1,32,1,32]> z_output0 = {op}(x=a_input0, y=a_input1)[name=string(\"z_output0\")];"
        ),
        "tensor<bool, [1,32,1,32]> a_input0, tensor<bool, [1,32,1,32]> a_input1",
        "z_output0",
    );
    AneCompiler::compile_mil_text(&mil, &[]).is_ok()
}

/// Try ternary op (like scatter: data, indices, updates)
fn try_scatter(op: &str) -> bool {
    let mil = mil_program(
        &format!(
            "        int32 ax = const()[name=string(\"ax\"), val=int32(3)];\n\
                     tensor<fp16, [1,32,1,32]> z_output0 = {op}(data=a_input0, indices=a_input1, updates=a_input2, axis=ax)[name=string(\"z_output0\")];"
        ),
        "tensor<fp16, [1,32,1,32]> a_input0, tensor<int32, [1,32,1,4]> a_input1, tensor<fp16, [1,32,1,4]> a_input2",
        "z_output0",
    );
    AneCompiler::compile_mil_text(&mil, &[]).is_ok()
}

/// Try op with "input" arg name instead of "x" (like quantize)
fn try_unary_input_arg(op: &str) -> bool {
    let mil = mil_program(
        &format!(
            "        tensor<fp16, [1,32,1,32]> z_output0 = {op}(input=a_input0)[name=string(\"z_output0\")];"
        ),
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    AneCompiler::compile_mil_text(&mil, &[]).is_ok()
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║             ANE Op Name Fuzzer                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Track already-known ops to highlight NEW discoveries
    let known_supported: HashSet<&str> = [
        "add",
        "sub",
        "mul",
        "real_div",
        "relu",
        "sigmoid",
        "tanh",
        "softmax",
        "sqrt",
        "square",
        "exp",
        "exp2",
        "erf",
        "ceil",
        "floor",
        "round",
        "atan",
        "pow",
        "clip",
        "abs",
        "sign",
        "maximum",
        "minimum",
        "floor_div",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "equal",
        "not_equal",
        "reduce_sum",
        "reduce_mean",
        "reduce_max",
        "reduce_min",
        "reduce_l2_norm",
        "matmul",
        "conv",
        "layer_norm",
        "reshape",
        "transpose",
        "concat",
        "slice_by_index",
        "slice_by_size",
        "split",
        "expand_dims",
        "squeeze",
        "tile",
        "select",
        "logical_not",
        "cast",
        "dequantize",
        "const",
        "scaled_dot_product_attention",
    ]
    .into_iter()
    .collect();

    let mut found: Vec<(String, &str)> = Vec::new(); // (op_name, template_type)

    // ═══ Part 1: Name variants for known-failing ops ═══
    println!("══ Part 1: Name variants for failing ops ══");
    println!();

    let neg_variants = ["neg", "negate", "negative", "minus", "unary_minus", "neg_"];
    let log_variants = [
        "log",
        "ln",
        "log_e",
        "natural_log",
        "log_natural",
        "logarithm",
        "log2",
        "log10",
        "log1p",
    ];
    let rsqrt_variants = [
        "rsqrt",
        "inverse_sqrt",
        "reciprocal_sqrt",
        "inv_sqrt",
        "isqrt",
        "reciprocal",
    ];
    let inverse_variants = ["inverse", "reciprocal", "inv", "recip", "rcp"];
    let mod_variants = ["mod", "fmod", "remainder", "rem", "modulo", "modulus"];
    let scatter_variants = [
        "scatter",
        "scatter_nd",
        "scatter_along_axis",
        "scatter_elements",
        "scatter_update",
        "tensor_scatter",
        "scatter_add",
        "index_put",
        "scatter_to",
        "place",
    ];
    let logical_variants = [
        "logical_and",
        "logic_and",
        "bitwise_and",
        "bool_and",
        "and",
        "logical_or",
        "logic_or",
        "bitwise_or",
        "bool_or",
        "or",
        "logical_xor",
        "logic_xor",
        "bitwise_xor",
        "bool_xor",
        "xor",
    ];
    let trig_variants = [
        "sin", "sine", "cos", "cosine", "tan", "tangent", "asin", "arcsin", "acos", "arccos",
        "atan", "arctan", "sinh", "arcsinh", "asinh", "cosh", "arccosh", "acosh", "atan2",
    ];
    let gather_variants = [
        "gather",
        "gather_nd",
        "gather_along_axis",
        "gather_elements",
        "index_select",
        "take",
        "take_along_axis",
        "embedding",
        "lookup",
        "index",
    ];
    let pool_variants = [
        "avg_pool",
        "average_pool",
        "average_pooling",
        "mean_pool",
        "global_avg_pool",
        "adaptive_avg_pool",
        "max_pool",
        "maximum_pool",
        "maximum_pooling",
        "global_max_pool",
        "adaptive_max_pool",
        "l2_pool",
    ];
    let norm_variants = [
        "batch_norm",
        "batchnorm",
        "batch_normalization",
        "instance_norm",
        "instance_normalization",
        "instancenorm",
        "group_norm",
        "groupnorm",
        "group_normalization",
        "rms_norm",
        "rmsnorm",
    ];
    let quantize_variants = [
        "quantize",
        "quant",
        "fake_quant",
        "quantize_linear",
        "quantize_per_tensor",
        "quantize_per_channel",
    ];
    let where_variants = ["where", "cond", "if_else", "conditional", "ternary"];

    struct FuzzGroup<'a> {
        label: &'a str,
        variants: &'a [&'a str],
        test_fn: fn(&str) -> bool,
    }

    let groups = vec![
        FuzzGroup {
            label: "neg alternatives",
            variants: &neg_variants,
            test_fn: try_unary,
        },
        FuzzGroup {
            label: "log alternatives",
            variants: &log_variants,
            test_fn: try_unary,
        },
        FuzzGroup {
            label: "rsqrt alternatives",
            variants: &rsqrt_variants,
            test_fn: try_unary,
        },
        FuzzGroup {
            label: "inverse alternatives",
            variants: &inverse_variants,
            test_fn: try_unary,
        },
        FuzzGroup {
            label: "mod alternatives",
            variants: &mod_variants,
            test_fn: try_binary,
        },
        FuzzGroup {
            label: "scatter alternatives",
            variants: &scatter_variants,
            test_fn: try_scatter,
        },
        FuzzGroup {
            label: "logical alternatives",
            variants: &logical_variants,
            test_fn: try_binary_bool,
        },
        FuzzGroup {
            label: "trig alternatives",
            variants: &trig_variants,
            test_fn: try_unary,
        },
        FuzzGroup {
            label: "gather alternatives",
            variants: &gather_variants,
            test_fn: try_unary,
        },
        FuzzGroup {
            label: "pool alternatives",
            variants: &pool_variants,
            test_fn: try_unary,
        },
        FuzzGroup {
            label: "norm alternatives",
            variants: &norm_variants,
            test_fn: try_unary,
        },
        FuzzGroup {
            label: "quantize alternatives",
            variants: &quantize_variants,
            test_fn: try_unary_input_arg,
        },
        FuzzGroup {
            label: "where/cond alternatives",
            variants: &where_variants,
            test_fn: try_unary,
        },
    ];

    for group in &groups {
        let hits: Vec<&&str> = group
            .variants
            .iter()
            .filter(|&&name| (group.test_fn)(name))
            .collect();
        if hits.is_empty() {
            println!("  {}: no hits", group.label);
        } else {
            for &&name in &hits {
                let is_new = !known_supported.contains(name);
                let marker = if is_new { "🆕" } else { "  " };
                println!("  {marker} ✅ {name}  [{}]", group.label);
                if is_new {
                    found.push((name.to_string(), group.label));
                }
            }
        }
    }

    // ═══ Part 2: Broad discovery sweep — common ML op names ═══
    println!();
    println!("══ Part 2: Broad discovery sweep ══");
    println!();

    // Comprehensive list of plausible MIL op names from CoreML, ONNX, TF, PyTorch
    let unary_candidates = [
        // Activations
        "gelu",
        "silu",
        "swish",
        "mish",
        "hardswish",
        "hardsigmoid",
        "hardtanh",
        "softsign",
        "softplus",
        "selu",
        "elu",
        "celu",
        "leaky_relu",
        "prelu",
        "thresholded_relu",
        "relu6",
        "gelu_tanh_approx",
        "gelu_approx",
        // Math
        "log_softmax",
        "cumsum",
        "cumprod",
        "expm1",
        "log2",
        "log10",
        "log1p",
        "square_root",
        "cube_root",
        "cbrt",
        "normalize",
        "l2_normalize",
        "l2_norm",
        "reciprocal",
        "recip",
        "rcp",
        "negate",
        "negative",
        "bitwise_not",
        "not",
        "identity",
        "copy",
        "clone",
        "noop",
        "no_op",
        "flatten",
        "contiguous",
        "to_dense",
        "one_hot",
        "argmax",
        "argmin",
        "sort",
        "argsort",
        "topk",
        "top_k",
        "unique",
        "nonzero",
        "non_zero",
        "isnan",
        "is_nan",
        "isinf",
        "is_inf",
        "isfinite",
        "is_finite",
        "fill",
        "zeros_like",
        "ones_like",
        "full_like",
        "cumulative_sum",
        "cumulative_prod",
        "reverse",
        "flip",
        "mirror",
        "range",
        "arange",
        "linspace",
        "size",
        "shape",
        "rank",
        "numel",
        // Quantization-related
        "fake_quantize",
        "quant",
        "dequant",
        "quantize_linear",
        "dequantize_linear",
    ];

    let binary_candidates = [
        // Math
        "atan2",
        "hypot",
        "ldexp",
        "copysign",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "left_shift",
        "right_shift",
        // Comparison
        "is_close",
        "allclose",
        // Element-wise
        "fma",
        "addmul",
        "addcmul",
        "lerp",
        "linear_interpolation",
        "cross",
        "dot",
        "inner",
        "outer",
        "where_nonzero",
    ];

    let reduce_candidates = [
        "reduce_l1_norm",
        "reduce_log_sum",
        "reduce_log_sum_exp",
        "reduce_sum_square",
        "reduce_any",
        "reduce_all",
        "reduce_argmax",
        "reduce_argmin",
        "reduce_variance",
        "reduce_std",
        "reduce_median",
        "reduce_mode",
        "reduce_count_nonzero",
        "sum",
        "mean",
        "max",
        "min",
        "prod",
        "any",
        "all",
        "argmax",
        "argmin",
        "logsumexp",
    ];

    println!("  Testing {} unary candidates...", unary_candidates.len());
    for &name in &unary_candidates {
        if try_unary(name) && !known_supported.contains(name) {
            println!("    🆕 ✅ {name}  [unary]");
            found.push((name.to_string(), "unary"));
        }
    }

    println!("  Testing {} binary candidates...", binary_candidates.len());
    for &name in &binary_candidates {
        if try_binary(name) && !known_supported.contains(name) {
            println!("    🆕 ✅ {name}  [binary]");
            found.push((name.to_string(), "binary"));
        }
    }

    println!(
        "  Testing {} reduction candidates...",
        reduce_candidates.len()
    );
    for &name in &reduce_candidates {
        if try_reduce(name) && !known_supported.contains(name) {
            println!("    🆕 ✅ {name}  [reduction]");
            found.push((name.to_string(), "reduction"));
        }
    }

    // ═══ Part 3: Prefix/suffix pattern sweep ═══
    println!();
    println!("══ Part 3: Prefix/suffix patterns ══");
    println!();

    let base_ops = [
        "neg", "log", "sin", "cos", "tan", "mod", "gather", "scatter", "pool", "norm", "quantize",
        "where",
    ];
    let prefixes = ["", "c_", "f_", "ml_", "ane_", "coreml_", "nn_", "torch_"];
    let suffixes = ["", "_v2", "_fast", "_approx", "_exact", "_ane"];

    let mut prefix_hits = 0;
    for &base in &base_ops {
        for &prefix in &prefixes {
            for &suffix in &suffixes {
                let name = format!("{prefix}{base}{suffix}");
                if name == base {
                    continue;
                } // skip already-tested base
                if try_unary(&name) && !known_supported.contains(name.as_str()) {
                    println!("    🆕 ✅ {name}  [prefix/suffix unary]");
                    found.push((name, "prefix/suffix"));
                    prefix_hits += 1;
                }
            }
        }
    }
    if prefix_hits == 0 {
        println!("  No hits from prefix/suffix patterns.");
    }

    // ═══ Part 4: CoreML constexpr_ family ═══
    println!();
    println!("══ Part 4: constexpr_ ops ══");
    println!();

    let constexpr_candidates = [
        "constexpr_affine_dequantize",
        "constexpr_lut_to_dense",
        "constexpr_sparse_to_dense",
        "constexpr_blockwise_shift_scale",
        "constexpr_cast",
        "constexpr_sparse_blockwise_shift_scale",
    ];

    for &name in &constexpr_candidates {
        // constexpr ops produce const outputs — test with simple tensor output
        let mil = mil_program(
            &format!(
                "        tensor<fp16, [1,32,1,32]> z_output0 = {name}(x=a_input0)[name=string(\"z_output0\")];"
            ),
            "tensor<fp16, [1,32,1,32]> a_input0",
            "z_output0",
        );
        if AneCompiler::compile_mil_text(&mil, &[]).is_ok() {
            println!("    🆕 ✅ {name}");
            found.push((name.to_string(), "constexpr"));
        }
    }
    println!("  (constexpr ops may need weight blobs to work — false negatives possible)");

    // ═══ Summary ═══
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  DISCOVERIES");
    println!("═══════════════════════════════════════════════════════════════════");
    if found.is_empty() {
        println!("  No new ops discovered beyond known set.");
    } else {
        println!("  Found {} new ops:", found.len());
        for (name, category) in &found {
            println!("    🆕 {name:<35} [{category}]");
        }
    }
    println!();
}
