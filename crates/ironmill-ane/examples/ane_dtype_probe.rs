//! Probe ANE compiler for sub-byte and integer data type support.
//!
//! Tests whether the ANE compiler accepts INT4, UINT4, INT8, UINT8,
//! and other non-fp16 tensor types in MIL programs.
//!
//! Run with: cargo run -p ironmill-ane --example ane_dtype_probe

use mil_rs::ffi::ane::AneCompiler;

const BUILD_INFO: &str = r#"[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]"#;

fn mil_program(body: &str, inputs: &str, output: &str) -> String {
    format!(
        "program(1.3)\n{BUILD_INFO}\n{{\n    func main<ios18>({inputs}) {{\n{body}\n    }} -> ({output});\n}}"
    )
}

fn try_compile(label: &str, mil_text: &str) -> bool {
    match AneCompiler::compile_mil_text(mil_text, &[]) {
        Ok(_) => {
            println!("  ✅ {label}");
            true
        }
        Err(e) => {
            let msg = format!("{e}");
            let short = if msg.len() > 70 {
                format!("{}…", &msg[..70])
            } else {
                msg
            };
            println!("  ❌ {label}: {short}");
            false
        }
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║             ANE Data Type Probe                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ═══ Part 1: What dtype strings does the MIL parser accept? ═══
    println!("══ Part 1: Input tensor dtypes ══");
    println!();

    let dtypes = [
        // Standard
        ("fp16", "fp16"),
        ("fp32", "fp32"),
        ("int8", "int8"),
        ("uint8", "uint8"),
        ("int16", "int16"),
        ("uint16", "uint16"),
        ("int32", "int32"),
        ("uint32", "uint32"),
        ("bool", "bool"),
        // Sub-byte / low-bit
        ("int4", "int4"),
        ("uint4", "uint4"),
        ("int2", "int2"),
        ("uint2", "uint2"),
        ("int1", "int1"),
        ("uint1", "uint1"),
        // Float variants
        ("bf16", "bf16"),
        ("fp8", "fp8"),
        ("fp8e4m3", "fp8e4m3"),
        ("fp8e5m2", "fp8e5m2"),
        ("fp64", "fp64"),
        ("float16", "float16"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("half", "half"),
        ("float", "float"),
        ("double", "double"),
    ];

    for (label, dtype) in &dtypes {
        let mil = mil_program(
            &format!(
                "        tensor<{dtype}, [1,32,1,32]> z_output0 = identity(x=a_input0)[name=string(\"z_output0\")];"
            ),
            &format!("tensor<{dtype}, [1,32,1,32]> a_input0"),
            "z_output0",
        );
        try_compile(&format!("input tensor<{label}>"), &mil);
    }

    // ═══ Part 2: Cast to/from integer types ═══
    println!();
    println!("══ Part 2: Cast operations with integer types ══");
    println!();

    let cast_pairs = [
        ("fp16", "int4"),
        ("fp16", "uint4"),
        ("fp16", "int8"),
        ("fp16", "uint8"),
        ("fp16", "int16"),
        ("fp16", "int32"),
        ("int8", "fp16"),
        ("int4", "fp16"),
        ("uint4", "fp16"),
        ("uint8", "fp16"),
        ("int16", "fp16"),
    ];

    for (from, to) in &cast_pairs {
        let mil = mil_program(
            &format!(
                "        tensor<{to}, [1,32,1,32]> z_output0 = cast(x=a_input0, dtype=string(\"{to}\"))[name=string(\"z_output0\")];"
            ),
            &format!("tensor<{from}, [1,32,1,32]> a_input0"),
            "z_output0",
        );
        try_compile(&format!("cast {from} → {to}"), &mil);
    }

    // ═══ Part 3: Arithmetic on integer types ═══
    println!();
    println!("══ Part 3: Arithmetic on integer types ══");
    println!();

    let int_types = ["int8", "uint8", "int4", "uint4", "int16", "int32"];
    let ops = ["add", "mul", "sub"];

    for dtype in &int_types {
        for op in &ops {
            let mil = mil_program(
                &format!(
                    "        tensor<{dtype}, [1,32,1,32]> z_output0 = {op}(x=a_input0, y=a_input1)[name=string(\"z_output0\")];"
                ),
                &format!(
                    "tensor<{dtype}, [1,32,1,32]> a_input0, tensor<{dtype}, [1,32,1,32]> a_input1"
                ),
                "z_output0",
            );
            try_compile(&format!("{op} on {dtype}"), &mil);
        }
    }

    // ═══ Part 4: Mixed-precision (fp16 input, int output and vice versa) ═══
    println!();
    println!("══ Part 4: Mixed-precision operations ══");
    println!();

    // quantize-like: fp16 → int via mul+round+clip+cast chain
    for out_dtype in ["int8", "int4", "uint4", "uint8"] {
        let mil = mil_program(
            &format!(
                "        fp16 sc = const()[name=string(\"sc\"), val=fp16(0.1)];\n\
                         tensor<fp16, [1,32,1,32]> scaled = mul(x=a_input0, y=sc)[name=string(\"scaled\")];\n\
                         tensor<fp16, [1,32,1,32]> rounded = round(x=scaled)[name=string(\"rounded\")];\n\
                         tensor<{out_dtype}, [1,32,1,32]> z_output0 = cast(x=rounded, dtype=string(\"{out_dtype}\"))[name=string(\"z_output0\")];"
            ),
            "tensor<fp16, [1,32,1,32]> a_input0",
            "z_output0",
        );
        try_compile(&format!("fp16 → mul → round → cast → {out_dtype}"), &mil);
    }

    // dequantize-like: int → fp16 via cast+mul+add chain
    for in_dtype in ["int8", "int4", "uint4", "uint8"] {
        let mil = mil_program(
            &format!(
                "        tensor<fp16, [1,32,1,32]> casted = cast(x=a_input0, dtype=string(\"fp16\"))[name=string(\"casted\")];\n\
                         fp16 sc = const()[name=string(\"sc\"), val=fp16(0.1)];\n\
                         tensor<fp16, [1,32,1,32]> z_output0 = mul(x=casted, y=sc)[name=string(\"z_output0\")];"
            ),
            &format!("tensor<{in_dtype}, [1,32,1,32]> a_input0"),
            "z_output0",
        );
        try_compile(
            &format!("{in_dtype} → cast fp16 → mul (dequant path)"),
            &mil,
        );
    }

    // ═══ Part 5: matmul with integer types ═══
    println!();
    println!("══ Part 5: Matmul with integer types ══");
    println!();

    for dtype in ["int8", "int4", "uint4", "uint8"] {
        // matmul producing fp16 output from int inputs
        let mil = mil_program(
            &format!(
                "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n\
                         tensor<fp16, [1,32,32]> z_output0 = matmul(x=a_input0, y=a_input1, transpose_x=bF, transpose_y=bF)[name=string(\"z_output0\")];"
            ),
            &format!("tensor<{dtype}, [1,32,32]> a_input0, tensor<{dtype}, [1,32,32]> a_input1"),
            "z_output0",
        );
        try_compile(&format!("matmul({dtype}, {dtype}) → fp16"), &mil);
    }

    // ═══ Part 6: constexpr_affine_dequantize with int4 ═══
    println!();
    println!("══ Part 6: Dequantize op with integer types ══");
    println!();

    for dtype in ["int8", "int4", "uint4", "uint8"] {
        let mil = mil_program(
            &format!(
                "        fp16 sc = const()[name=string(\"sc\"), val=fp16(0.1)];\n\
                         {dtype} zp = const()[name=string(\"zp\"), val={dtype}(0)];\n\
                         tensor<fp16, [1,32,1,32]> z_output0 = dequantize(input=a_input0, scale=sc, zero_point=zp)[name=string(\"z_output0\")];"
            ),
            &format!("tensor<{dtype}, [1,32,1,32]> a_input0"),
            "z_output0",
        );
        try_compile(&format!("dequantize({dtype} → fp16)"), &mil);
    }

    // ═══ Part 7: conv with int types (what maderix tested) ═══
    println!();
    println!("══ Part 7: Conv with integer weights (maderix-style) ══");
    println!();

    // maderix reported INT8 W8A8 via conv — test the pattern
    for w_dtype in ["int8", "int4", "uint4"] {
        let mil = mil_program(
            &format!(
                "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n\
                         int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n\
                         tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n\
                         string pt = const()[name=string(\"pt\"), val=string(\"custom\")];\n\
                         tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n\
                         tensor<{w_dtype}, [32,32,1,1]> w = const()[name=string(\"w\"), val=tensor<{w_dtype}, [32,32,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n\
                         tensor<fp16, [1,32,1,32]> z_output0 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=w,x=a_input0)[name=string(\"z_output0\")];"
            ),
            "tensor<fp16, [1,32,1,32]> a_input0",
            "z_output0",
        );
        // This needs a weight blob — try without one to see if parser accepts the types
        try_compile(&format!("conv(fp16 input, {w_dtype} weight)"), &mil);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Done. Review results above to determine INT4 feasibility.");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
}
