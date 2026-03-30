//! Probe Apple's ANE compiler to discover which MIL ops it actually accepts.
//!
//! Run with: cargo run -p ironmill-ane --example ane_op_probe
//!
//! Uses the same MIL text format as maderix/ANE and ironmill's own emitter.
//! Axes and keep_dims for reductions are passed as named const references,
//! matching the syntax that Apple's compiler expects.

fn main() {
    probe::run();
}

mod probe {
    use ironmill_ane_sys::AneCompiler;

    const BUILD_INFO: &str = r#"[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]"#;

    fn mil_program(body: &str, inputs: &str, output: &str) -> String {
        format!(
            "program(1.3)\n{BUILD_INFO}\n{{\n    func main<ios18>({inputs}) {{\n{body}\n    }} -> ({output});\n}}"
        )
    }

    /// Binary elementwise op: two same-shape fp16 inputs → one fp16 output
    fn binary_fp16(op: &str) -> String {
        mil_program(
            &format!(
                "        tensor<fp16, [1,32,1,32]> z_output0 = {op}(x=a_input0, y=a_input1)[name=string(\"z_output0\")];"
            ),
            "tensor<fp16, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1",
            "z_output0",
        )
    }

    /// Unary elementwise op: one fp16 input → one fp16 output
    fn unary_fp16(op: &str) -> String {
        mil_program(
            &format!(
                "        tensor<fp16, [1,32,1,32]> z_output0 = {op}(x=a_input0)[name=string(\"z_output0\")];"
            ),
            "tensor<fp16, [1,32,1,32]> a_input0",
            "z_output0",
        )
    }

    /// Reduction along last axis (-1) using named const refs (maderix-style syntax).
    fn reduce_last_axis(op: &str) -> String {
        mil_program(
            &format!(
                "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n\
                         bool kd = const()[name=string(\"kd\"), val=bool(true)];\n\
                         tensor<fp16, [1,32,1,1]> z_output0 = {op}(x=a_input0, axes=rax, keep_dims=kd)[name=string(\"z_output0\")];"
            ),
            "tensor<fp16, [1,32,1,32]> a_input0",
            "z_output0",
        )
    }

    /// Reduction along channel axis (1) using named const refs.
    fn reduce_channel_axis(op: &str) -> String {
        mil_program(
            &format!(
                "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n\
                         bool kd = const()[name=string(\"kd\"), val=bool(true)];\n\
                         tensor<fp16, [1,1,1,32]> z_output0 = {op}(x=a_input0, axes=rax, keep_dims=kd)[name=string(\"z_output0\")];"
            ),
            "tensor<fp16, [1,32,1,32]> a_input0",
            "z_output0",
        )
    }

    /// Comparison op returning bool tensor
    fn compare_op(op: &str) -> String {
        mil_program(
            &format!(
                "        tensor<bool, [1,32,1,32]> z_output0 = {op}(x=a_input0, y=a_input1)[name=string(\"z_output0\")];"
            ),
            "tensor<fp16, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1",
            "z_output0",
        )
    }

    /// Unary op with scalar const second arg (e.g. pow(x, y=-0.5))
    fn unary_with_scalar(op: &str, param_name: &str, val: &str) -> String {
        mil_program(
            &format!(
                "        fp16 sc = const()[name=string(\"sc\"), val=fp16({val})];\n\
                         tensor<fp16, [1,32,1,32]> z_output0 = {op}(x=a_input0, {param_name}=sc)[name=string(\"z_output0\")];"
            ),
            "tensor<fp16, [1,32,1,32]> a_input0",
            "z_output0",
        )
    }

    struct OpProbe {
        name: &'static str,
        category: &'static str,
        mil_text: String,
    }

    pub fn run() {
        let probes = vec![
            // ═══ SANITY CHECKS (known allowlisted ops) ═══
            OpProbe {
                name: "add",
                category: "sanity",
                mil_text: binary_fp16("add"),
            },
            OpProbe {
                name: "mul",
                category: "sanity",
                mil_text: binary_fp16("mul"),
            },
            OpProbe {
                name: "sub",
                category: "sanity",
                mil_text: binary_fp16("sub"),
            },
            OpProbe {
                name: "real_div",
                category: "sanity",
                mil_text: binary_fp16("real_div"),
            },
            OpProbe {
                name: "relu",
                category: "sanity",
                mil_text: unary_fp16("relu"),
            },
            OpProbe {
                name: "sigmoid",
                category: "sanity",
                mil_text: unary_fp16("sigmoid"),
            },
            OpProbe {
                name: "tanh",
                category: "sanity",
                mil_text: unary_fp16("tanh"),
            },
            OpProbe {
                name: "sqrt",
                category: "sanity",
                mil_text: unary_fp16("sqrt"),
            },
            OpProbe {
                name: "clip",
                category: "sanity",
                mil_text: mil_program(
                    "        fp16 lo = const()[name=string(\"lo\"), val=fp16(0.0)];\n\
                         fp16 hi = const()[name=string(\"hi\"), val=fp16(1.0)];\n\
                         tensor<fp16, [1,32,1,32]> z_output0 = clip(x=a_input0, alpha=lo, beta=hi)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "pow",
                category: "sanity",
                mil_text: unary_with_scalar("pow", "y", "-0.5"),
            },
            OpProbe {
                name: "softmax",
                category: "sanity",
                mil_text: mil_program(
                    "        int32 ax = const()[name=string(\"ax\"), val=int32(1)];\n\
                         tensor<fp16, [1,32,1,32]> z_output0 = softmax(x=a_input0, axis=ax)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "gather",
                category: "sanity",
                mil_text: mil_program(
                    "        int32 ax = const()[name=string(\"ax\"), val=int32(3)];\n\
                         tensor<fp16, [1,32,1,8]> z_output0 = gather(x=a_input0, indices=a_input1, axis=ax)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0, tensor<int32, [8]> a_input1",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "reshape",
                category: "sanity",
                mil_text: mil_program(
                    "        tensor<int32, [3]> sh = const()[name=string(\"sh\"), val=tensor<int32, [3]>([1,32,32])];\n\
                         tensor<fp16, [1,32,32]> z_output0 = reshape(x=a_input0, shape=sh)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "transpose",
                category: "sanity",
                mil_text: mil_program(
                    "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n\
                         tensor<fp16, [1,32,32,1]> z_output0 = transpose(x=a_input0, perm=pm)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "matmul",
                category: "sanity",
                mil_text: mil_program(
                    "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n\
                         tensor<fp16, [1,32,32]> z_output0 = matmul(x=a_input0, y=a_input1, transpose_x=bF, transpose_y=bF)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,32]> a_input0, tensor<fp16, [1,32,32]> a_input1",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "slice_by_index",
                category: "sanity",
                mil_text: mil_program(
                    "        tensor<int32, [4]> sb = const()[name=string(\"sb\"), val=tensor<int32, [4]>([0,0,0,0])];\n\
                         tensor<int32, [4]> se = const()[name=string(\"se\"), val=tensor<int32, [4]>([1,32,1,16])];\n\
                         tensor<fp16, [1,32,1,16]> z_output0 = slice_by_index(x=a_input0, begin=sb, end=se)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "slice_by_size",
                category: "sanity",
                mil_text: mil_program(
                    "        tensor<int32, [4]> sb = const()[name=string(\"sb\"), val=tensor<int32, [4]>([0,0,0,0])];\n\
                         tensor<int32, [4]> ss = const()[name=string(\"ss\"), val=tensor<int32, [4]>([1,32,1,16])];\n\
                         tensor<fp16, [1,32,1,16]> z_output0 = slice_by_size(x=a_input0, begin=sb, size=ss)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "concat",
                category: "sanity",
                mil_text: mil_program(
                    "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n\
                         bool cid = const()[name=string(\"cid\"), val=bool(false)];\n\
                         tensor<fp16, [1,64,1,32]> z_output0 = concat(axis=cax, interleave=cid, values=(a_input0, a_input1))[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "cast_fp16_fp32",
                category: "sanity",
                mil_text: mil_program(
                    "        tensor<fp32, [1,32,1,32]> z_output0 = cast(x=a_input0, dtype=string(\"fp32\"))[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            // ═══ COMPARISON OPS (needed for QJL) ═══
            OpProbe {
                name: "greater",
                category: "comparison",
                mil_text: compare_op("greater"),
            },
            OpProbe {
                name: "greater_equal",
                category: "comparison",
                mil_text: compare_op("greater_equal"),
            },
            OpProbe {
                name: "less",
                category: "comparison",
                mil_text: compare_op("less"),
            },
            OpProbe {
                name: "less_equal",
                category: "comparison",
                mil_text: compare_op("less_equal"),
            },
            OpProbe {
                name: "equal",
                category: "comparison",
                mil_text: compare_op("equal"),
            },
            OpProbe {
                name: "not_equal",
                category: "comparison",
                mil_text: compare_op("not_equal"),
            },
            // ═══ UNARY MATH ═══
            OpProbe {
                name: "abs",
                category: "unary_math",
                mil_text: unary_fp16("abs"),
            },
            OpProbe {
                name: "sign",
                category: "unary_math",
                mil_text: unary_fp16("sign"),
            },
            OpProbe {
                name: "exp",
                category: "unary_math",
                mil_text: unary_fp16("exp"),
            },
            OpProbe {
                name: "exp2",
                category: "unary_math",
                mil_text: unary_fp16("exp2"),
            },
            OpProbe {
                name: "log",
                category: "unary_math",
                mil_text: unary_fp16("log"),
            },
            OpProbe {
                name: "rsqrt",
                category: "unary_math",
                mil_text: unary_fp16("rsqrt"),
            },
            OpProbe {
                name: "inverse",
                category: "unary_math",
                mil_text: unary_fp16("inverse"),
            },
            OpProbe {
                name: "ceil",
                category: "unary_math",
                mil_text: unary_fp16("ceil"),
            },
            OpProbe {
                name: "floor",
                category: "unary_math",
                mil_text: unary_fp16("floor"),
            },
            OpProbe {
                name: "round",
                category: "unary_math",
                mil_text: unary_fp16("round"),
            },
            OpProbe {
                name: "neg",
                category: "unary_math",
                mil_text: unary_fp16("neg"),
            },
            OpProbe {
                name: "square",
                category: "unary_math",
                mil_text: unary_fp16("square"),
            },
            OpProbe {
                name: "erf",
                category: "unary_math",
                mil_text: unary_fp16("erf"),
            },
            OpProbe {
                name: "sin",
                category: "unary_math",
                mil_text: unary_fp16("sin"),
            },
            OpProbe {
                name: "cos",
                category: "unary_math",
                mil_text: unary_fp16("cos"),
            },
            OpProbe {
                name: "tan",
                category: "unary_math",
                mil_text: unary_fp16("tan"),
            },
            OpProbe {
                name: "asin",
                category: "unary_math",
                mil_text: unary_fp16("asin"),
            },
            OpProbe {
                name: "acos",
                category: "unary_math",
                mil_text: unary_fp16("acos"),
            },
            OpProbe {
                name: "atan",
                category: "unary_math",
                mil_text: unary_fp16("atan"),
            },
            OpProbe {
                name: "sinh",
                category: "unary_math",
                mil_text: unary_fp16("sinh"),
            },
            OpProbe {
                name: "cosh",
                category: "unary_math",
                mil_text: unary_fp16("cosh"),
            },
            // ═══ BINARY MATH ═══
            OpProbe {
                name: "maximum",
                category: "binary_math",
                mil_text: binary_fp16("maximum"),
            },
            OpProbe {
                name: "minimum",
                category: "binary_math",
                mil_text: binary_fp16("minimum"),
            },
            OpProbe {
                name: "floor_div",
                category: "binary_math",
                mil_text: binary_fp16("floor_div"),
            },
            OpProbe {
                name: "mod",
                category: "binary_math",
                mil_text: binary_fp16("mod"),
            },
            // ═══ REDUCTIONS (last axis, with const refs — maderix-confirmed syntax) ═══
            OpProbe {
                name: "reduce_sum (axis -1)",
                category: "reduction",
                mil_text: reduce_last_axis("reduce_sum"),
            },
            OpProbe {
                name: "reduce_mean (axis -1)",
                category: "reduction",
                mil_text: reduce_last_axis("reduce_mean"),
            },
            OpProbe {
                name: "reduce_max (axis -1)",
                category: "reduction",
                mil_text: reduce_last_axis("reduce_max"),
            },
            OpProbe {
                name: "reduce_min (axis -1)",
                category: "reduction",
                mil_text: reduce_last_axis("reduce_min"),
            },
            OpProbe {
                name: "reduce_prod (axis -1)",
                category: "reduction",
                mil_text: reduce_last_axis("reduce_prod"),
            },
            OpProbe {
                name: "reduce_l2_norm (axis -1)",
                category: "reduction",
                mil_text: reduce_last_axis("reduce_l2_norm"),
            },
            OpProbe {
                name: "reduce_sum (axis 1)",
                category: "reduction",
                mil_text: reduce_channel_axis("reduce_sum"),
            },
            OpProbe {
                name: "reduce_mean (axis 1)",
                category: "reduction",
                mil_text: reduce_channel_axis("reduce_mean"),
            },
            OpProbe {
                name: "reduce_max (axis 1)",
                category: "reduction",
                mil_text: reduce_channel_axis("reduce_max"),
            },
            OpProbe {
                name: "reduce_min (axis 1)",
                category: "reduction",
                mil_text: reduce_channel_axis("reduce_min"),
            },
            // ═══ LOGICAL ═══
            OpProbe {
                name: "logical_not",
                category: "logical",
                mil_text: mil_program(
                    "        tensor<bool, [1,32,1,32]> z_output0 = logical_not(x=a_input0)[name=string(\"z_output0\")];",
                    "tensor<bool, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "logical_and",
                category: "logical",
                mil_text: mil_program(
                    "        tensor<bool, [1,32,1,32]> z_output0 = logical_and(x=a_input0, y=a_input1)[name=string(\"z_output0\")];",
                    "tensor<bool, [1,32,1,32]> a_input0, tensor<bool, [1,32,1,32]> a_input1",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "logical_or",
                category: "logical",
                mil_text: mil_program(
                    "        tensor<bool, [1,32,1,32]> z_output0 = logical_or(x=a_input0, y=a_input1)[name=string(\"z_output0\")];",
                    "tensor<bool, [1,32,1,32]> a_input0, tensor<bool, [1,32,1,32]> a_input1",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "logical_xor",
                category: "logical",
                mil_text: mil_program(
                    "        tensor<bool, [1,32,1,32]> z_output0 = logical_xor(x=a_input0, y=a_input1)[name=string(\"z_output0\")];",
                    "tensor<bool, [1,32,1,32]> a_input0, tensor<bool, [1,32,1,32]> a_input1",
                    "z_output0",
                ),
            },
            // ═══ SCATTER (cache mutation) ═══
            OpProbe {
                name: "scatter",
                category: "scatter",
                mil_text: mil_program(
                    "        int32 ax = const()[name=string(\"ax\"), val=int32(3)];\n\
                         tensor<fp16, [1,32,1,32]> z_output0 = scatter(data=a_input0, indices=a_input1, updates=a_input2, axis=ax)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0, tensor<int32, [1,32,1,4]> a_input1, tensor<fp16, [1,32,1,4]> a_input2",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "scatter_nd",
                category: "scatter",
                mil_text: mil_program(
                    "        tensor<fp16, [1,32,1,32]> z_output0 = scatter_nd(data=a_input0, indices=a_input1, updates=a_input2)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0, tensor<int32, [1,4,1,1]> a_input1, tensor<fp16, [1,4,1,32]> a_input2",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "scatter_along_axis",
                category: "scatter",
                mil_text: mil_program(
                    "        int32 ax = const()[name=string(\"ax\"), val=int32(3)];\n\
                         tensor<fp16, [1,32,1,32]> z_output0 = scatter_along_axis(data=a_input0, indices=a_input1, updates=a_input2, axis=ax)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0, tensor<int32, [1,32,1,4]> a_input1, tensor<fp16, [1,32,1,4]> a_input2",
                    "z_output0",
                ),
            },
            // ═══ CAST ═══
            OpProbe {
                name: "cast fp16→bool",
                category: "cast",
                mil_text: mil_program(
                    "        tensor<bool, [1,32,1,32]> z_output0 = cast(x=a_input0, dtype=string(\"bool\"))[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "cast bool→fp16",
                category: "cast",
                mil_text: mil_program(
                    "        tensor<fp16, [1,32,1,32]> z_output0 = cast(x=a_input0, dtype=string(\"fp16\"))[name=string(\"z_output0\")];",
                    "tensor<bool, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "cast fp16→int32",
                category: "cast",
                mil_text: mil_program(
                    "        tensor<int32, [1,32,1,32]> z_output0 = cast(x=a_input0, dtype=string(\"int32\"))[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "cast int32→fp16",
                category: "cast",
                mil_text: mil_program(
                    "        tensor<fp16, [1,32,1,32]> z_output0 = cast(x=a_input0, dtype=string(\"fp16\"))[name=string(\"z_output0\")];",
                    "tensor<int32, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "cast fp16→int8",
                category: "cast",
                mil_text: mil_program(
                    "        tensor<int8, [1,32,1,32]> z_output0 = cast(x=a_input0, dtype=string(\"int8\"))[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "cast fp32→fp16",
                category: "cast",
                mil_text: mil_program(
                    "        tensor<fp16, [1,32,1,32]> z_output0 = cast(x=a_input0, dtype=string(\"fp16\"))[name=string(\"z_output0\")];",
                    "tensor<fp32, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            // ═══ CONDITIONAL ═══
            OpProbe {
                name: "select",
                category: "conditional",
                mil_text: mil_program(
                    "        tensor<fp16, [1,32,1,32]> z_output0 = select(cond=a_input0, a=a_input1, b=a_input2)[name=string(\"z_output0\")];",
                    "tensor<bool, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1, tensor<fp16, [1,32,1,32]> a_input2",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "where",
                category: "conditional",
                mil_text: mil_program(
                    "        tensor<fp16, [1,32,1,32]> z_output0 = where(cond=a_input0, x=a_input1, y=a_input2)[name=string(\"z_output0\")];",
                    "tensor<bool, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1, tensor<fp16, [1,32,1,32]> a_input2",
                    "z_output0",
                ),
            },
            // ═══ QUANTIZATION (INT8 W8A8 from maderix) ═══
            OpProbe {
                name: "quantize",
                category: "quantize",
                mil_text: mil_program(
                    "        fp16 sc = const()[name=string(\"sc\"), val=fp16(0.1)];\n\
                         int8 zp = const()[name=string(\"zp\"), val=int8(0)];\n\
                         tensor<int8, [1,32,1,32]> z_output0 = quantize(input=a_input0, scale=sc, zero_point=zp)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "dequantize",
                category: "quantize",
                mil_text: mil_program(
                    "        fp16 sc = const()[name=string(\"sc\"), val=fp16(0.1)];\n\
                         int8 zp = const()[name=string(\"zp\"), val=int8(0)];\n\
                         tensor<fp16, [1,32,1,32]> z_output0 = dequantize(input=a_input0, scale=sc, zero_point=zp)[name=string(\"z_output0\")];",
                    "tensor<int8, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            // ═══ NORMALIZATION ═══
            OpProbe {
                name: "layer_norm",
                category: "norm",
                mil_text: mil_program(
                    "        tensor<int32, [1]> nax = const()[name=string(\"nax\"), val=tensor<int32, [1]>([3])];\n\
                         fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n\
                         tensor<fp16, [1,32,1,32]> z_output0 = layer_norm(x=a_input0, axes=nax, epsilon=eps)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "batch_norm",
                category: "norm",
                mil_text: mil_program(
                    "        tensor<fp16, [32]> mn = const()[name=string(\"mn\"), val=tensor<fp16, [32]>(BLOBFILE(path=string(\"@model_path/weights/mean.bin\"), offset=uint64(64)))];\n\
                         tensor<fp16, [32]> vr = const()[name=string(\"vr\"), val=tensor<fp16, [32]>(BLOBFILE(path=string(\"@model_path/weights/var.bin\"), offset=uint64(64)))];\n\
                         fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n\
                         tensor<fp16, [1,32,1,32]> z_output0 = batch_norm(x=a_input0, mean=mn, variance=vr, epsilon=eps)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            // ═══ POOLING ═══
            OpProbe {
                name: "avg_pool",
                category: "pool",
                mil_text: mil_program(
                    "        tensor<int32, [2]> ks = const()[name=string(\"ks\"), val=tensor<int32, [2]>([1,2])];\n\
                         tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,2])];\n\
                         tensor<fp16, [1,32,1,16]> z_output0 = avg_pool(x=a_input0, kernel_sizes=ks, strides=st)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "max_pool",
                category: "pool",
                mil_text: mil_program(
                    "        tensor<int32, [2]> ks = const()[name=string(\"ks\"), val=tensor<int32, [2]>([1,2])];\n\
                         tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,2])];\n\
                         tensor<fp16, [1,32,1,16]> z_output0 = max_pool(x=a_input0, kernel_sizes=ks, strides=st)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            // ═══ MISC ═══
            OpProbe {
                name: "expand_dims",
                category: "shape",
                mil_text: mil_program(
                    "        tensor<int32, [1]> axes = const()[name=string(\"axes\"), val=tensor<int32, [1]>([0])];\n\
                         tensor<fp16, [1,1,32,1,32]> z_output0 = expand_dims(x=a_input0, axes=axes)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "squeeze",
                category: "shape",
                mil_text: mil_program(
                    "        tensor<int32, [1]> axes = const()[name=string(\"axes\"), val=tensor<int32, [1]>([2])];\n\
                         tensor<fp16, [1,32,32]> z_output0 = squeeze(x=a_input0, axes=axes)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "split",
                category: "shape",
                mil_text: mil_program(
                    "        int32 ax = const()[name=string(\"ax\"), val=int32(1)];\n\
                         int32 ns = const()[name=string(\"ns\"), val=int32(2)];\n\
                         tensor<fp16, [1,16,1,32]> z_output0, tensor<fp16, [1,16,1,32]> z_discard = split(x=a_input0, num_splits=ns, axis=ax)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
            OpProbe {
                name: "tile",
                category: "shape",
                mil_text: mil_program(
                    "        tensor<int32, [4]> reps = const()[name=string(\"reps\"), val=tensor<int32, [4]>([1,1,1,2])];\n\
                         tensor<fp16, [1,32,1,64]> z_output0 = tile(x=a_input0, reps=reps)[name=string(\"z_output0\")];",
                    "tensor<fp16, [1,32,1,32]> a_input0",
                    "z_output0",
                ),
            },
        ];

        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║              ANE Compiler Op Support Probe v2                   ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Testing {} ops against Apple's private ANE compiler           ║",
            probes.len()
        );
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();

        let mut results: Vec<(&str, &str, bool, String)> = Vec::new();

        for probe in &probes {
            let result = AneCompiler::compile_mil_text(&probe.mil_text, &[]);
            let (supported, detail) = match result {
                Ok(_ptr) => (true, "compiled OK".to_string()),
                Err(e) => {
                    let msg = format!("{e}");
                    let short = if msg.len() > 60 {
                        format!("{}…", &msg[..60])
                    } else {
                        msg
                    };
                    (false, short)
                }
            };

            let icon = if supported { "✅" } else { "❌" };
            println!(
                "  {icon} {:<30} [{:<12}]  {detail}",
                probe.name, probe.category
            );
            results.push((probe.name, probe.category, supported, detail));
        }

        // Summary by category
        println!();
        println!("═══════════════════════════════════════════════════════════════════");
        println!("  SUMMARY BY CATEGORY");
        println!("═══════════════════════════════════════════════════════════════════");

        let categories: Vec<&str> = vec![
            "sanity",
            "comparison",
            "unary_math",
            "binary_math",
            "reduction",
            "logical",
            "scatter",
            "cast",
            "conditional",
            "quantize",
            "norm",
            "pool",
            "shape",
        ];

        for cat in &categories {
            let cat_results: Vec<_> = results.iter().filter(|r| r.1 == *cat).collect();
            if cat_results.is_empty() {
                continue;
            }
            let supported: Vec<_> = cat_results.iter().filter(|r| r.2).collect();
            let unsupported: Vec<_> = cat_results.iter().filter(|r| !r.2).collect();

            println!();
            println!("  {cat} ({}/{})", supported.len(), cat_results.len());
            if !supported.is_empty() {
                let names: Vec<_> = supported.iter().map(|r| r.0).collect();
                println!("    ✅ {}", names.join(", "));
            }
            if !unsupported.is_empty() {
                let names: Vec<_> = unsupported.iter().map(|r| r.0).collect();
                println!("    ❌ {}", names.join(", "));
            }
        }

        let total_supported = results.iter().filter(|r| r.2).count();
        let total = results.len();
        println!();
        println!("  Total: {total_supported}/{total} ops supported by ANE compiler");

        // TurboQuant feasibility
        println!();
        println!("═══════════════════════════════════════════════════════════════════");
        println!("  TURBOQUANT FEASIBILITY");
        println!("═══════════════════════════════════════════════════════════════════");

        let has = |name: &str| results.iter().any(|r| r.0 == name && r.2);
        let has_cat = |cat: &str| results.iter().any(|r| r.1 == cat && r.2);
        let check = |ok: bool| if ok { "✅" } else { "❌" };

        println!();
        println!(
            "  {} Comparison ops (QJL sign test)",
            check(has_cat("comparison"))
        );
        println!("  {} sign() direct", check(has("sign")));
        println!("  {} abs()", check(has("abs")));
        println!("  {} select (conditional)", check(has("select")));
        println!(
            "  {} Cast fp16↔bool",
            check(has("cast fp16→bool") && has("cast bool→fp16"))
        );
        println!(
            "  {} reduce_sum",
            check(has("reduce_sum (axis -1)") || has("reduce_sum (axis 1)"))
        );
        println!(
            "  {} reduce_mean",
            check(has("reduce_mean (axis -1)") || has("reduce_mean (axis 1)"))
        );
        println!("  {} exp (softmax component)", check(has("exp")));
        println!("  {} pow (normalization)", check(has("pow")));
        println!("  {} softmax", check(has("softmax")));
        println!("  {} matmul (Hadamard rotation)", check(has("matmul")));
        println!("  {} gather (LUT dequant)", check(has("gather")));
        println!("  {} scatter (cache mutation)", check(has_cat("scatter")));
        println!(
            "  {} quantize/dequantize (INT8)",
            check(has("quantize") || has("dequantize"))
        );
        println!();
    }
}
