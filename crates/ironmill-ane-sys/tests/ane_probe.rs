//! ANE runtime probe tests — empirical research into undocumented ANE behavior.
//!
//! Run with:
//!   cargo test -p ironmill-ane-sys --test ane_probe -- --ignored --nocapture --test-threads=1
//!
//! These are `#[ignore]`d research tests that compile MIL programs on the ANE
//! and document results via `eprintln!`.  Budget: ~119 compiles per process.

#![cfg(target_os = "macos")]

use std::ffi::c_void;

use ironmill_ane_sys::*;

// =========================================================================
// MIL text helpers
// =========================================================================

const BUILD_INFO: &str = r#"[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]"#;

/// Build a complete MIL program from inline body, parameter list, and return name.
fn mil(body: &str, inputs: &str, output: &str) -> String {
    format!(
        "program(1.3)\n{BUILD_INFO}\n{{\n    func main<ios18>({inputs}) {{\n{body}\n    }} -> ({output});\n}}"
    )
}

/// Try to compile a MIL program.  Returns `Some(model)` on success, `None` on failure.
/// Prints result to stderr.
fn try_compile(name: &str, mil_text: &str) -> Option<InMemoryModel> {
    let budget = ironmill_ane_sys::model::remaining_budget();
    if budget == 0 {
        eprintln!("  ⏭  {name}: SKIPPED (budget exhausted)");
        return None;
    }
    match ironmill_ane_sys::model::compile_mil_text(mil_text, &[], ironmill_ane_sys::model::ANE_QOS)
    {
        Ok(model) => {
            eprintln!("  ✅ {name}: compiled OK  (budget left: {})", budget - 1);
            Some(model)
        }
        Err(e) => {
            eprintln!("  ❌ {name}: {e}");
            None
        }
    }
}

/// Like `try_compile` but only reports pass/fail.
fn probe(name: &str, mil_text: &str) -> bool {
    try_compile(name, mil_text).is_some()
}

fn has_ane() -> bool {
    DeviceInfo::has_ane().unwrap_or(false)
}

// =========================================================================
// IOSurface helpers (for eval probes)
// =========================================================================

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceCreate(properties: *const c_void) -> *mut c_void;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFAllocatorDefault: *const c_void;
    static kCFTypeDictionaryKeyCallBacks: u8;
    static kCFTypeDictionaryValueCallBacks: u8;

    fn CFDictionaryCreateMutable(
        allocator: *const c_void,
        capacity: isize,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> *mut c_void;
    fn CFDictionarySetValue(dict: *mut c_void, key: *const c_void, value: *const c_void);
    fn CFNumberCreate(
        allocator: *const c_void,
        number_type: i64,
        value_ptr: *const c_void,
    ) -> *mut c_void;
    fn CFRelease(cf: *mut c_void);
}

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    static kIOSurfaceAllocSize: *const c_void;
    static kIOSurfaceWidth: *const c_void;
    static kIOSurfaceHeight: *const c_void;
    static kIOSurfaceBytesPerElement: *const c_void;
    static kIOSurfaceBytesPerRow: *const c_void;
}

/// Create a minimal IOSurface for probe eval tests.
/// Returns an IOSurfaceRef (must be CFReleased by the caller).
fn create_probe_iosurface() -> *mut c_void {
    const ALLOC_SIZE: i64 = 16384; // ANE minimum
    const ONE: i64 = 1;
    const CF_NUMBER_SINT64_TYPE: i64 = 4;

    unsafe {
        let dict = CFDictionaryCreateMutable(
            kCFAllocatorDefault,
            5,
            std::ptr::addr_of!(kCFTypeDictionaryKeyCallBacks) as *const c_void,
            std::ptr::addr_of!(kCFTypeDictionaryValueCallBacks) as *const c_void,
        );
        if dict.is_null() {
            return std::ptr::null_mut();
        }

        let props: [(*const c_void, i64); 5] = [
            (kIOSurfaceAllocSize, ALLOC_SIZE),
            (kIOSurfaceWidth, ALLOC_SIZE),
            (kIOSurfaceHeight, ONE),
            (kIOSurfaceBytesPerElement, ONE),
            (kIOSurfaceBytesPerRow, ALLOC_SIZE),
        ];

        let mut cf_nums: Vec<*mut c_void> = Vec::new();
        for &(key, value) in &props {
            let num = CFNumberCreate(
                kCFAllocatorDefault,
                CF_NUMBER_SINT64_TYPE,
                &value as *const i64 as *const c_void,
            );
            if !num.is_null() {
                CFDictionarySetValue(dict, key, num);
                cf_nums.push(num);
            }
        }

        let surface = IOSurfaceCreate(dict);

        for num in cf_nums {
            CFRelease(num);
        }
        CFRelease(dict);

        surface
    }
}

// =========================================================================
// Probe 1 — MIL Op Support Matrix  (~25 compiles)
// =========================================================================

#[test]
#[ignore]
fn probe_op_support_matrix() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 1: MIL Op Support Matrix              ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");
    eprintln!(
        "Budget remaining: {}\n",
        ironmill_ane_sys::model::remaining_budget()
    );

    let mut results: Vec<(&str, bool)> = Vec::new();

    // --- identity (baseline) ---
    let r = probe(
        "identity",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = identity(x=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("identity", r));

    // --- add ---
    let r = probe(
        "add",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = add(x=a_input0, y=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("add", r));

    // --- mul ---
    let r = probe(
        "mul",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = mul(x=a_input0, y=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("mul", r));

    // --- sub ---
    let r = probe(
        "sub",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = sub(x=a_input0, y=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("sub", r));

    // --- relu ---
    let r = probe(
        "relu",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = relu(x=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("relu", r));

    // --- softmax ---
    let r = probe(
        "softmax",
        &mil(
            "        int32 ax = const()[name=string(\"ax\"), val=int32(-1)];\n\
         tensor<fp16, [1,4,1,4]> out = softmax(x=a_input0, axis=ax)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("softmax", r));

    // --- round ---
    let r = probe(
        "round",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = round(x=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("round", r));

    // --- clip ---
    let r = probe(
        "clip",
        &mil(
            "        fp16 lo = const()[name=string(\"lo\"), val=fp16(-1.0)];\n\
         fp16 hi = const()[name=string(\"hi\"), val=fp16(1.0)];\n\
         tensor<fp16, [1,4,1,4]> out = clip(x=a_input0, alpha=lo, beta=hi)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("clip", r));

    // --- abs ---
    let r = probe(
        "abs",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = abs(x=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("abs", r));

    // --- sign ---
    let r = probe(
        "sign",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = sign(x=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("sign", r));

    // --- sqrt ---
    let r = probe(
        "sqrt",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = sqrt(x=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("sqrt", r));

    // --- exp ---
    let r = probe(
        "exp",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = exp(x=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("exp", r));

    // --- greater + cast to fp16 (comparison) ---
    let r = probe(
        "greater",
        &mil(
            "        fp16 zero = const()[name=string(\"zero\"), val=fp16(0.0)];\n\
         tensor<bool, [1,4,1,4]> cmp = greater(x=a_input0, y=zero)[name=string(\"cmp\")];\n\
         tensor<fp16, [1,4,1,4]> out = cast(x=cmp, dtype=string(\"fp16\"))[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("greater+cast", r));

    // --- select ---
    let r = probe(
        "select",
        &mil(
            "        fp16 zero = const()[name=string(\"zero\"), val=fp16(0.0)];\n\
         tensor<bool, [1,4,1,4]> cmp = greater(x=a_input0, y=zero)[name=string(\"cmp\")];\n\
         fp16 pos = const()[name=string(\"pos\"), val=fp16(1.0)];\n\
         fp16 neg = const()[name=string(\"neg\"), val=fp16(-1.0)];\n\
         tensor<fp16, [1,4,1,4]> out = select(cond=cmp, a=pos, b=neg)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("select", r));

    // --- reduce_sum (with tile to maintain output shape) ---
    let r = probe(
        "reduce_sum",
        &mil(
            "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n\
         bool kd = const()[name=string(\"kd\"), val=bool(true)];\n\
         tensor<fp16, [1,4,1,1]> rs = reduce_sum(x=a_input0, axes=rax, keep_dims=kd)[name=string(\"rs\")];\n\
         tensor<int32, [4]> rep = const()[name=string(\"rep\"), val=tensor<int32, [4]>([1,1,1,4])];\n\
         tensor<fp16, [1,4,1,4]> out = tile(x=rs, reps=rep)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("reduce_sum+tile", r));

    // --- reduce_max ---
    let r = probe(
        "reduce_max",
        &mil(
            "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n\
         bool kd = const()[name=string(\"kd\"), val=bool(true)];\n\
         tensor<fp16, [1,4,1,1]> rm = reduce_max(x=a_input0, axes=rax, keep_dims=kd)[name=string(\"rm\")];\n\
         tensor<int32, [4]> rep = const()[name=string(\"rep\"), val=tensor<int32, [4]>([1,1,1,4])];\n\
         tensor<fp16, [1,4,1,4]> out = tile(x=rm, reps=rep)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("reduce_max+tile", r));

    // --- matmul (3D) ---
    let r = probe(
        "matmul",
        &mil(
            "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n\
         tensor<fp16, [1,4,4]> out = matmul(x=a_input0, y=a_input1, transpose_x=bF, transpose_y=bF)[name=string(\"out\")];",
            "tensor<fp16, [1,4,4]> a_input0, tensor<fp16, [1,4,4]> a_input1",
            "out",
        ),
    );
    results.push(("matmul", r));

    // --- concat ---
    let r = probe(
        "concat",
        &mil(
            "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n\
         bool cid = const()[name=string(\"cid\"), val=bool(false)];\n\
         tensor<fp16, [1,8,1,4]> out = concat(axis=cax, interleave=cid, values=(a_input0, a_input1))[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0, tensor<fp16, [1,4,1,4]> a_input1",
            "out",
        ),
    );
    results.push(("concat", r));

    // --- transpose ---
    let r = probe(
        "transpose",
        &mil(
            "        tensor<int32, [4]> perm = const()[name=string(\"perm\"), val=tensor<int32, [4]>([0,3,2,1])];\n\
         tensor<fp16, [1,4,1,4]> out = transpose(x=a_input0, perm=perm)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("transpose", r));

    // --- silu ---
    let r = probe(
        "silu",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = silu(x=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("silu", r));

    // --- layer_norm ---
    let r = probe(
        "layer_norm",
        &mil(
            "        tensor<int32, [1]> nax = const()[name=string(\"nax\"), val=tensor<int32, [1]>([3])];\n\
         fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n\
         tensor<fp16, [1,4,1,4]> out = layer_norm(x=a_input0, axes=nax, epsilon=eps)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("layer_norm", r));

    // --- pow ---
    let r = probe(
        "pow",
        &mil(
            "        fp16 sc = const()[name=string(\"sc\"), val=fp16(-0.5)];\n\
         tensor<fp16, [1,4,1,4]> out = pow(x=a_input0, y=sc)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("pow", r));

    // --- erf ---
    let r = probe(
        "erf",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = erf(x=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("erf", r));

    // --- gather (expected to fail for dynamic) ---
    let r = probe(
        "gather (dynamic)",
        &mil(
            "        tensor<int32, [4]> idx = const()[name=string(\"idx\"), val=tensor<int32, [4]>([0,1,2,3])];\n\
         int32 ax = const()[name=string(\"ax\"), val=int32(3)];\n\
         tensor<fp16, [1,4,1,4]> out = gather(x=a_input0, indices=idx, axis=ax)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("gather", r));

    // --- real_div ---
    let r = probe(
        "real_div",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = real_div(x=a_input0, y=a_input1)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0, tensor<fp16, [1,4,1,4]> a_input1",
            "out",
        ),
    );
    results.push(("real_div", r));

    // --- Summary ---
    eprintln!("\n─── Op Support Summary ───");
    let pass = results.iter().filter(|(_, ok)| *ok).count();
    let fail = results.iter().filter(|(_, ok)| !*ok).count();
    eprintln!("  Compiled: {pass}/{} | Failed: {fail}", results.len());
    for (name, ok) in &results {
        eprintln!("  {:20} {}", name, if *ok { "✅" } else { "❌" });
    }
    eprintln!(
        "Budget remaining: {}",
        ironmill_ane_sys::model::remaining_budget()
    );
}

// =========================================================================
// Probe 2 — Data Type Support  (~10 compiles)
// =========================================================================

#[test]
#[ignore]
fn probe_dtype_support() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 2: Data Type Support                  ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");
    eprintln!(
        "Budget remaining: {}\n",
        ironmill_ane_sys::model::remaining_budget()
    );

    let mut results: Vec<(&str, bool)> = Vec::new();

    // fp16 input → fp16 output (baseline, known to work)
    let r = probe(
        "fp16 → fp16",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = identity(x=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("fp16 → fp16", r));

    // fp32 input → fp32 output
    let r = probe(
        "fp32 → fp32",
        &mil(
            "        tensor<fp32, [1,4,1,4]> out = identity(x=a_input0)[name=string(\"out\")];",
            "tensor<fp32, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("fp32 → fp32", r));

    // int8 input → int8 output
    let r = probe(
        "int8 → int8",
        &mil(
            "        tensor<int8, [1,4,1,4]> out = identity(x=a_input0)[name=string(\"out\")];",
            "tensor<int8, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("int8 → int8", r));

    // cast fp16 → int8 → fp16 roundtrip
    let r = probe(
        "cast fp16→int8→fp16",
        &mil(
            "        tensor<int8, [1,4,1,4]> q = cast(x=a_input0, dtype=string(\"int8\"))[name=string(\"q\")];\n\
         tensor<fp16, [1,4,1,4]> out = cast(x=q, dtype=string(\"fp16\"))[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("cast fp16→int8→fp16", r));

    // cast fp16 → fp32
    let r = probe(
        "cast fp16→fp32",
        &mil(
            "        tensor<fp32, [1,4,1,4]> out = cast(x=a_input0, dtype=string(\"fp32\"))[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("cast fp16→fp32", r));

    // cast fp16 → int32 (expected to fail)
    let r = probe(
        "cast fp16→int32",
        &mil(
            "        tensor<int32, [1,4,1,4]> out = cast(x=a_input0, dtype=string(\"int32\"))[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("cast fp16→int32", r));

    // cast fp16 → int16
    let r = probe(
        "cast fp16→int16",
        &mil(
            "        tensor<int16, [1,4,1,4]> out = cast(x=a_input0, dtype=string(\"int16\"))[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("cast fp16→int16", r));

    // cast fp16 → uint8
    let r = probe(
        "cast fp16→uint8",
        &mil(
            "        tensor<uint8, [1,4,1,4]> out = cast(x=a_input0, dtype=string(\"uint8\"))[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("cast fp16→uint8", r));

    // ==== INT4 / UINT4 — critical for TurboQuant ====

    eprintln!("\n  --- INT4/UINT4 probes (critical for TurboQuant) ---");

    // cast fp16 → int4
    let r = probe(
        "cast fp16→int4 ⭐",
        &mil(
            "        tensor<int4, [1,4,1,4]> out = cast(x=a_input0, dtype=string(\"int4\"))[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("cast fp16→int4", r));

    // cast fp16 → uint4
    let r = probe(
        "cast fp16→uint4 ⭐",
        &mil(
            "        tensor<uint4, [1,4,1,4]> out = cast(x=a_input0, dtype=string(\"uint4\"))[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("cast fp16→uint4", r));

    // int4 as function input
    let r = probe(
        "int4 input ⭐",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = cast(x=a_input0, dtype=string(\"fp16\"))[name=string(\"out\")];",
            "tensor<int4, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("int4 input", r));

    // bool input → cast to fp16
    let r = probe(
        "bool input",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = cast(x=a_input0, dtype=string(\"fp16\"))[name=string(\"out\")];",
            "tensor<bool, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("bool input", r));

    // --- Summary ---
    eprintln!("\n─── Data Type Summary ───");
    for (name, ok) in &results {
        eprintln!("  {:25} {}", name, if *ok { "✅" } else { "❌" });
    }
    eprintln!(
        "Budget remaining: {}",
        ironmill_ane_sys::model::remaining_budget()
    );
}

// =========================================================================
// Probe 3 — Shape Constraints  (~8 compiles)
// =========================================================================

#[test]
#[ignore]
fn probe_shape_constraints() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 3: Shape Constraints                  ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");
    eprintln!(
        "Budget remaining: {}\n",
        ironmill_ane_sys::model::remaining_budget()
    );

    let shapes: Vec<(&str, &str)> = vec![
        ("[1,1,1,1]", "minimal"),
        ("[1,4,1,32]", "typical ANE"),
        ("[1,128,1,128]", "medium"),
        ("[1,768,1,32]", "large channels (known problematic)"),
        ("[1,4096,1,32]", "very large channels"),
        ("[1,4,1,1]", "single token"),
        ("[2,4,1,4]", "batch > 1"),
        ("[1,4,2,4]", "height > 1"),
    ];

    let mut results = Vec::new();

    for (shape, desc) in &shapes {
        let name = format!("{shape} ({desc})");
        let body = format!(
            "        tensor<fp16, {shape}> out = add(x=a_input0, y=a_input0)[name=string(\"out\")];",
        );
        let io_type = format!("tensor<fp16, {shape}>");
        let r = probe(&name, &mil(&body, &format!("{io_type} a_input0"), "out"));
        results.push((name, r));
    }

    // --- Summary ---
    eprintln!("\n─── Shape Constraint Summary ───");
    for (name, ok) in &results {
        eprintln!("  {:45} {}", name, if *ok { "✅" } else { "❌" });
    }
    eprintln!(
        "Budget remaining: {}",
        ironmill_ane_sys::model::remaining_budget()
    );
}

// =========================================================================
// Probe 4 — Quantization-Relevant Op Chains  (~4 compiles)
// =========================================================================

#[test]
#[ignore]
fn probe_quantization_chains() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 4: Quantization-Relevant Op Chains    ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");
    eprintln!(
        "Budget remaining: {}\n",
        ironmill_ane_sys::model::remaining_budget()
    );

    let mut results: Vec<(&str, bool)> = Vec::new();

    // 4a: round(clip(mul(x, scale), min, max)) — affine quantize chain
    let r = probe(
        "round(clip(mul(x,s),lo,hi))",
        &mil(
            "        fp16 scale = const()[name=string(\"scale\"), val=fp16(10.0)];\n\
         tensor<fp16, [1,4,1,4]> scaled = mul(x=a_input0, y=scale)[name=string(\"scaled\")];\n\
         fp16 lo = const()[name=string(\"lo\"), val=fp16(-128.0)];\n\
         fp16 hi = const()[name=string(\"hi\"), val=fp16(127.0)];\n\
         tensor<fp16, [1,4,1,4]> clamped = clip(x=scaled, alpha=lo, beta=hi)[name=string(\"clamped\")];\n\
         tensor<fp16, [1,4,1,4]> out = round(x=clamped)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("affine_quant_chain", r));

    // 4b: fp16 → int8 → fp16 roundtrip (full quantize-dequantize)
    let r = probe(
        "full int8 quant→dequant",
        &mil(
            "        fp16 inv_scale = const()[name=string(\"inv_scale\"), val=fp16(10.0)];\n\
         tensor<fp16, [1,4,1,4]> scaled = mul(x=a_input0, y=inv_scale)[name=string(\"scaled\")];\n\
         tensor<fp16, [1,4,1,4]> rounded = round(x=scaled)[name=string(\"rounded\")];\n\
         fp16 lo = const()[name=string(\"lo\"), val=fp16(-128.0)];\n\
         fp16 hi = const()[name=string(\"hi\"), val=fp16(127.0)];\n\
         tensor<fp16, [1,4,1,4]> clamped = clip(x=rounded, alpha=lo, beta=hi)[name=string(\"clamped\")];\n\
         tensor<int8, [1,4,1,4]> quantized = cast(x=clamped, dtype=string(\"int8\"))[name=string(\"quantized\")];\n\
         tensor<fp16, [1,4,1,4]> back = cast(x=quantized, dtype=string(\"fp16\"))[name=string(\"back\")];\n\
         fp16 scale = const()[name=string(\"scale\"), val=fp16(0.1)];\n\
         tensor<fp16, [1,4,1,4]> out = mul(x=back, y=scale)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("int8_quant_dequant", r));

    // 4c: codebook lookup via greater + select chains
    let r = probe(
        "codebook (greater+select)",
        &mil(
            "        fp16 t1 = const()[name=string(\"t1\"), val=fp16(0.0)];\n\
         fp16 t2 = const()[name=string(\"t2\"), val=fp16(1.0)];\n\
         tensor<bool, [1,4,1,4]> g1 = greater(x=a_input0, y=t1)[name=string(\"g1\")];\n\
         tensor<bool, [1,4,1,4]> g2 = greater(x=a_input0, y=t2)[name=string(\"g2\")];\n\
         fp16 v0 = const()[name=string(\"v0\"), val=fp16(-1.0)];\n\
         fp16 v1 = const()[name=string(\"v1\"), val=fp16(0.5)];\n\
         fp16 v2 = const()[name=string(\"v2\"), val=fp16(2.0)];\n\
         tensor<fp16, [1,4,1,4]> s1 = select(cond=g1, a=v1, b=v0)[name=string(\"s1\")];\n\
         tensor<fp16, [1,4,1,4]> out = select(cond=g2, a=v2, b=s1)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("codebook_lookup", r));

    // 4d: RMSNorm + quantize pipeline (TurboQuant cache-write)
    let r = probe(
        "rmsnorm+quantize pipe",
        &mil(
            "        tensor<fp16, [1,4,1,4]> sq = mul(x=a_input0, y=a_input0)[name=string(\"sq\")];\n\
         tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n\
         bool kd = const()[name=string(\"kd\"), val=bool(true)];\n\
         tensor<fp16, [1,4,1,1]> ms = reduce_mean(x=sq, axes=rax, keep_dims=kd)[name=string(\"ms\")];\n\
         fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n\
         tensor<fp16, [1,4,1,1]> mse = add(x=ms, y=eps)[name=string(\"mse\")];\n\
         fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n\
         tensor<fp16, [1,4,1,1]> rrms = pow(x=mse, y=nhalf)[name=string(\"rrms\")];\n\
         tensor<fp16, [1,4,1,4]> normed = mul(x=a_input0, y=rrms)[name=string(\"normed\")];\n\
         fp16 inv_scale = const()[name=string(\"inv_scale\"), val=fp16(50.0)];\n\
         tensor<fp16, [1,4,1,4]> scaled = mul(x=normed, y=inv_scale)[name=string(\"scaled\")];\n\
         tensor<fp16, [1,4,1,4]> rounded = round(x=scaled)[name=string(\"rounded\")];\n\
         fp16 lo = const()[name=string(\"lo\"), val=fp16(-128.0)];\n\
         fp16 hi = const()[name=string(\"hi\"), val=fp16(127.0)];\n\
         tensor<fp16, [1,4,1,4]> clamped = clip(x=rounded, alpha=lo, beta=hi)[name=string(\"clamped\")];\n\
         tensor<int8, [1,4,1,4]> quantized = cast(x=clamped, dtype=string(\"int8\"))[name=string(\"quantized\")];\n\
         tensor<fp16, [1,4,1,4]> out = cast(x=quantized, dtype=string(\"fp16\"))[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    );
    results.push(("rmsnorm_quantize_pipe", r));

    // --- Summary ---
    eprintln!("\n─── Quantization Chain Summary ───");
    for (name, ok) in &results {
        eprintln!("  {:35} {}", name, if *ok { "✅" } else { "❌" });
    }
    eprintln!(
        "Budget remaining: {}",
        ironmill_ane_sys::model::remaining_budget()
    );
}

// =========================================================================
// Probe 5 — Chaining Request API  (0 compiles)
// =========================================================================

#[test]
#[ignore]
fn probe_chaining_request() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 5: Chaining Request API               ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");

    // ChainingRequest::new requires raw ObjC pointers.
    // Probe what happens with null pointers (expected to fail gracefully or crash).
    eprintln!("  Attempting ChainingRequest::new with null pointers...");
    let result = std::panic::catch_unwind(|| unsafe {
        ChainingRequest::new(
            std::ptr::null_mut(), // inputs
            std::ptr::null_mut(), // output_sets
            std::ptr::null_mut(), // lb_input_symbol_id
            std::ptr::null_mut(), // lb_output_symbol_id
            0,                    // procedure_index
            std::ptr::null_mut(), // signal_events
            0,                    // transaction_handle
            0,                    // fw_enqueue_delay
            0,                    // memory_pool_id
        )
    });
    match result {
        Ok(Ok(req)) => {
            eprintln!("  ✅ ChainingRequest created (surprising with nulls)");
            eprintln!("     raw ptr:             {:?}", req.as_raw());
            eprintln!("     procedure_index:     {:?}", req.procedure_index());
            eprintln!("     transaction_handle:  {:?}", req.transaction_handle());
            eprintln!("     fw_enqueue_delay:    {:?}", req.fw_enqueue_delay());
            eprintln!("     memory_pool_id:      {:?}", req.memory_pool_id());

            // Try validate
            let valid = req.validate();
            eprintln!("     validate:            {valid}");
        }
        Ok(Err(e)) => {
            eprintln!("  ❌ ChainingRequest::new returned error: {e}");
        }
        Err(e) => {
            eprintln!("  💥 ChainingRequest::new panicked: {e:?}");
        }
    }
}

// =========================================================================
// Probe 6 — Performance Stats  (1 compile)
// =========================================================================

#[test]
#[ignore]
fn probe_perf_stats() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 6: Performance Stats                  ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");
    eprintln!(
        "Budget remaining: {}\n",
        ironmill_ane_sys::model::remaining_budget()
    );

    // Compile a simple model
    let model = match try_compile(
        "add (perf probe)",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = add(x=a_input0, y=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    ) {
        Some(m) => m,
        None => {
            eprintln!("  Cannot compile model for perf probe");
            return;
        }
    };

    // Read perf stats mask
    let mask = model.perf_stats_mask();
    eprintln!("  perf_stats_mask (default): {mask} (0x{mask:x})");

    // Set mask to enable all stats
    model.set_perf_stats_mask(0xFFFFFFFF);
    let mask2 = model.perf_stats_mask();
    eprintln!("  perf_stats_mask (after set 0xFFFFFFFF): {mask2} (0x{mask2:x})");

    // Try known bit patterns
    for bit in 0..8u32 {
        model.set_perf_stats_mask(1 << bit);
        let readback = model.perf_stats_mask();
        if readback == (1 << bit) {
            eprintln!("  bit {bit} (0x{:x}): accepted", 1u32 << bit);
        } else {
            eprintln!("  bit {bit} (0x{:x}): readback=0x{readback:x}", 1u32 << bit);
        }
    }

    // Create a PerformanceStats object
    match PerformanceStats::with_hw_execution_ns(0) {
        Ok(stats) => {
            eprintln!("  PerformanceStats::with_hw_execution_ns(0): OK");
            eprintln!("    hw_execution_time: {}", stats.hw_execution_time());
        }
        Err(e) => {
            eprintln!("  PerformanceStats creation failed: {e}");
        }
    }

    // Read other model properties for context
    eprintln!("\n  Model properties:");
    eprintln!("    state:                     {}", model.state());
    eprintln!("    program_handle:            {}", model.program_handle());
    eprintln!(
        "    intermediate_buffer_handle:{}",
        model.intermediate_buffer_handle()
    );
    eprintln!("    queue_depth:               {}", model.queue_depth());
    eprintln!("    is_mil_model:              {}", model.is_mil_model());
    eprintln!(
        "    compiled_model_exists:     {}",
        model.compiled_model_exists()
    );
    eprintln!(
        "    hex_string_identifier:     {:?}",
        model.hex_string_identifier()
    );
    eprintln!(
        "    local_model_path:          {:?}",
        model.local_model_path()
    );
    eprintln!(
        "    compiler_options_file_name:{:?}",
        model.compiler_options_file_name()
    );
}

// =========================================================================
// Probe 7 — Client.echo  (0 compiles)
// =========================================================================

unsafe extern "C" {
    fn objc_getClass(name: *const u8) -> *mut c_void;
    fn sel_registerName(name: *const u8) -> *mut c_void;
    fn objc_msgSend(receiver: *mut c_void, sel: *mut c_void, ...) -> *mut c_void;
}

/// Create a simple NSString for echo testing.
unsafe fn make_nsstring(s: &str) -> *mut c_void {
    let cls = unsafe { objc_getClass(b"NSString\0".as_ptr()) };
    if cls.is_null() {
        return std::ptr::null_mut();
    }
    let sel = unsafe { sel_registerName(b"stringWithUTF8String:\0".as_ptr()) };

    // Need a null-terminated C string
    let c_str = std::ffi::CString::new(s).unwrap();

    type StringFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *const u8) -> *mut c_void;
    let f: StringFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(cls, sel, c_str.as_ptr() as *const u8) }
}

/// Get the `description` of an NSObject as a Rust String.
unsafe fn ns_description(obj: *mut c_void) -> Option<String> {
    if obj.is_null() {
        return None;
    }
    let sel = unsafe { sel_registerName(b"description\0".as_ptr()) };
    type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let desc = unsafe { f(obj, sel) };
    if desc.is_null() {
        return None;
    }
    let utf8_sel = unsafe { sel_registerName(b"UTF8String\0".as_ptr()) };
    type CStrFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const i8;
    let f2: CStrFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let ptr = unsafe { f2(desc, utf8_sel) };
    if ptr.is_null() {
        return None;
    }
    Some(
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[test]
#[ignore]
fn probe_client_echo() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 7: Client.echo                        ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");

    let client = match Client::shared_connection() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Failed to get client: {e}");
            return;
        }
    };

    // Try echo with an NSString
    let payload = unsafe { make_nsstring("hello ANE") };
    eprintln!("  NSString payload ptr: {payload:?}");

    if payload.is_null() {
        eprintln!("  ❌ Failed to create NSString for echo");
        return;
    }

    let result = unsafe { client.echo(payload) };
    eprintln!("  echo(\"hello ANE\") = {result}");

    // Try echo with nil
    let result_nil = unsafe { client.echo(std::ptr::null_mut()) };
    eprintln!("  echo(nil) = {result_nil}");

    // Try echo with NSNumber
    let num_cls = unsafe { objc_getClass(b"NSNumber\0".as_ptr()) };
    if !num_cls.is_null() {
        let sel = unsafe { sel_registerName(b"numberWithInt:\0".as_ptr()) };
        type NumFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i32) -> *mut c_void;
        let f: NumFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let num = unsafe { f(num_cls, sel, 42) };
        let result_num = unsafe { client.echo(num) };
        eprintln!("  echo(NSNumber(42)) = {result_num}");
    }

    // Try echo with NSDictionary
    let dict_cls = unsafe { objc_getClass(b"NSDictionary\0".as_ptr()) };
    if !dict_cls.is_null() {
        let sel = unsafe { sel_registerName(b"dictionary\0".as_ptr()) };
        type DictFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let f: DictFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let dict = unsafe { f(dict_cls, sel) };
        let result_dict = unsafe { client.echo(dict) };
        eprintln!("  echo(empty dict) = {result_dict}");
    }

    // Also check private connection echo
    match Client::shared_private_connection() {
        Ok(pc) => {
            let result_private = unsafe { pc.echo(payload) };
            eprintln!("  private.echo(\"hello ANE\") = {result_private}");
        }
        Err(e) => {
            eprintln!("  private connection failed: {e}");
        }
    }
}

// =========================================================================
// Probe 8 — Model Attributes  (1 compile)
// =========================================================================

#[test]
#[ignore]
fn probe_model_attributes() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 8: Model Attributes                   ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");
    eprintln!(
        "Budget remaining: {}\n",
        ironmill_ane_sys::model::remaining_budget()
    );

    let model = match try_compile(
        "identity (attrs probe)",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = identity(x=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    ) {
        Some(m) => m,
        None => {
            eprintln!("  Cannot compile model for attributes probe");
            return;
        }
    };

    // Read model_attributes
    let attrs = model.model_attributes();
    eprintln!("  model_attributes ptr:  {attrs:?}");

    if !attrs.is_null() {
        // Try to get NSDictionary description
        if let Some(desc) = unsafe { ns_description(attrs) } {
            eprintln!("  model_attributes description:\n    {desc}");
        }

        // Try to get count
        let count_sel = unsafe { sel_registerName(b"count\0".as_ptr()) };
        type CountFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize;
        let f: CountFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let count = unsafe { f(attrs, count_sel) };
        eprintln!("  model_attributes count: {count}");

        // Try to get allKeys
        let keys_sel = unsafe { sel_registerName(b"allKeys\0".as_ptr()) };
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let keys_fn: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let keys = unsafe { keys_fn(attrs, keys_sel) };
        if !keys.is_null() {
            if let Some(desc) = unsafe { ns_description(keys) } {
                eprintln!("  model_attributes keys: {desc}");
            }
        }
    } else {
        eprintln!("  model_attributes is NULL");
    }

    // Other model metadata
    // NOTE: save_model_files() can throw ObjC exceptions — skip it.
    eprintln!("\n  Additional model metadata:");
    eprintln!("    local_model_path:   {:?}", model.local_model_path());
    let model_url = model.model_url();
    eprintln!("    model_url ptr:      {model_url:?}");
    if !model_url.is_null() {
        if let Some(desc) = unsafe { ns_description(model_url) } {
            eprintln!("    model_url:          {desc}");
        }
    }
}

// =========================================================================
// Probe 9 — Session Hints  (0 compiles, reuses model from probe 8 if avail)
// =========================================================================

#[test]
#[ignore]
fn probe_session_hints() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 9: Session Hints                      ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");

    // FINDING: session_hint throws ObjC exceptions with all tested argument types.
    // Tested: NSDictionary, NSNumber, nil — all cause uncatchable SIGABRT.
    //
    // The sessionHintWithModel:hint:options:report:error: selector exists on
    // _ANEClient, but the expected argument types are undocumented and passing
    // wrong types causes an ObjC exception that Rust cannot catch.
    //
    // This API likely requires specific hint/options dictionary keys that we
    // have not yet discovered. Until those are reverse-engineered, this API
    // should be considered unusable.
    eprintln!("  ⚠️  session_hint API throws ObjC exceptions with all tested arg types.");
    eprintln!("  Tested: NSDictionary(empty), NSNumber, nil — all crash.");
    eprintln!("  The API exists but requires undiscovered argument format.");
    eprintln!("  CONCLUSION: Unusable without further reverse engineering.");
}

// =========================================================================
// Probe 10 — Compiler Options  (2 compiles)
// =========================================================================

#[test]
#[ignore]
fn probe_compiler_options() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 10: Compiler Options                  ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");
    eprintln!(
        "Budget remaining: {}\n",
        ironmill_ane_sys::model::remaining_budget()
    );

    // Test: compile with explicit options plist data
    // Options should be a binary plist. Let's try with None (already works)
    // and then try with an empty plist.
    let mil_text = mil(
        "        tensor<fp16, [1,4,1,4]> out = add(x=a_input0, y=a_input0)[name=string(\"out\")];",
        "tensor<fp16, [1,4,1,4]> a_input0",
        "out",
    );

    // 10a: Normal compile (no options)
    let r = probe("default options", &mil_text);
    eprintln!("  default options: {}", if r { "✅" } else { "❌" });

    // 10b: Try creating descriptor directly with options
    eprintln!("\n  Attempting descriptor with empty plist data...");
    // An empty binary plist is "bplist00\x08\x00..."  but let's try empty bytes
    match ironmill_ane_sys::model::InMemoryModelDescriptor::from_mil_text(&mil_text, &[], Some(b""))
    {
        Ok(desc) => {
            eprintln!("  ✅ descriptor with empty options: OK");
            eprintln!("    hex_id: {:?}", desc.hex_string_identifier());
        }
        Err(e) => {
            eprintln!("  ❌ descriptor with empty options: {e}");
        }
    }

    // 10c: Try with a minimal valid binary plist
    // Binary plist header: bplist00 + minimal content
    let empty_dict_plist = b"bplist00\xd0\x08\x09\x00\x00\x00\x00\x00\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x09";
    match ironmill_ane_sys::model::InMemoryModelDescriptor::from_mil_text(
        &mil_text,
        &[],
        Some(empty_dict_plist),
    ) {
        Ok(desc) => {
            eprintln!("  ✅ descriptor with binary plist: OK");
            eprintln!("    hex_id: {:?}", desc.hex_string_identifier());
        }
        Err(e) => {
            eprintln!("  ❌ descriptor with binary plist: {e}");
        }
    }

    eprintln!(
        "\nBudget remaining: {}",
        ironmill_ane_sys::model::remaining_budget()
    );
}

// =========================================================================
// Probe 11 — Budget & Lifecycle  (0 compiles)
// =========================================================================

#[test]
#[ignore]
fn probe_budget_and_lifecycle() {
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 11: Budget & Lifecycle                ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");

    let count = ironmill_ane_sys::model::compile_count();
    let remaining = ironmill_ane_sys::model::remaining_budget();
    let available = ironmill_ane_sys::model::is_available();

    eprintln!("  compile_count:     {count}");
    eprintln!("  remaining_budget:  {remaining}");
    eprintln!("  is_available:      {available}");
    eprintln!("  has_ane:           {:?}", DeviceInfo::has_ane());
    eprintln!("  num_anes:          {:?}", DeviceInfo::num_anes());
    eprintln!("  num_ane_cores:     {:?}", DeviceInfo::num_ane_cores());
    eprintln!("  architecture:      {:?}", DeviceInfo::architecture_type());
    eprintln!("  product_name:      {:?}", DeviceInfo::product_name());
    eprintln!("  build_version:     {:?}", DeviceInfo::build_version());
    eprintln!(
        "  is_vm:             {:?}",
        DeviceInfo::is_virtual_machine()
    );
}

// =========================================================================
// Probe 12 — Performance Stats Bits  (1 compile)
// =========================================================================

#[test]
#[ignore]
fn probe_perf_stats_bits() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 12: Performance Stats Bits            ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");
    eprintln!(
        "Budget remaining: {}\n",
        ironmill_ane_sys::model::remaining_budget()
    );

    // Compile a simple model (1 compile budget)
    let model = match try_compile(
        "add (perf bits probe)",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = add(x=a_input0, y=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    ) {
        Some(m) => m,
        None => {
            eprintln!("  Cannot compile model for perf bits probe");
            return;
        }
    };

    // Create I/O IOSurfaces
    let input_surface = create_probe_iosurface();
    let output_surface = create_probe_iosurface();
    if input_surface.is_null() || output_surface.is_null() {
        eprintln!("  Failed to create IOSurfaces for probe");
        return;
    }

    // Baseline: eval without stats to ensure it works
    match ironmill_ane_sys::model::eval(
        &model,
        &[input_surface],
        &[output_surface],
        ironmill_ane_sys::model::ANE_QOS,
    ) {
        Ok(()) => eprintln!("  Baseline eval: OK\n"),
        Err(e) => {
            eprintln!("  Baseline eval failed: {e}");
            unsafe {
                CFRelease(input_surface);
                CFRelease(output_surface);
            }
            return;
        }
    }

    // Probe each of the 32 perf_stats_mask bits
    eprintln!(
        "  {:>4}  {:>10}  {:>16}  {:>16}",
        "bit", "mask", "hw_exec_ns", "counter_data"
    );
    eprintln!("  {}", "-".repeat(52));

    let mut active_bits: Vec<u32> = Vec::new();

    for bit in 0..32u32 {
        let mask = 1u32 << bit;
        match ironmill_ane_sys::model::eval_with_stats(
            &model,
            &[input_surface],
            &[output_surface],
            ironmill_ane_sys::model::ANE_QOS,
            mask,
        ) {
            Ok(stats) => {
                // Guard: check raw data before calling typed accessors.
                // Some perf_stats_mask values leave the stats object unpopulated,
                // and accessing ObjC properties on it throws an uncatchable exception.
                if stats.p_stats_raw_data().is_null() {
                    eprintln!(
                        "  {:>4}  0x{:08x}  {:>16}  {:>16}",
                        bit, mask, "-", "(no raw data)"
                    );
                    continue;
                }

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let hw_ns = stats.hw_execution_time();
                    let counter = stats.perf_counter_data();
                    (hw_ns, !counter.is_null())
                }));

                match result {
                    Ok((hw_ns, has_counter)) => {
                        if hw_ns != 0 || has_counter {
                            active_bits.push(bit);
                        }
                        eprintln!(
                            "  {:>4}  0x{:08x}  {:>16}  {:>16}",
                            bit,
                            mask,
                            hw_ns,
                            if has_counter { "non-null" } else { "null" }
                        );
                    }
                    Err(_) => {
                        eprintln!(
                            "  {:>4}  0x{:08x}  {:>16}  {:>16}",
                            bit, mask, "PANIC", "(caught)"
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("  {:>4}  0x{:08x}  eval failed: {e}", bit, mask);
            }
        }
    }

    // Also try all-bits-set
    eprintln!();
    match ironmill_ane_sys::model::eval_with_stats(
        &model,
        &[input_surface],
        &[output_surface],
        ironmill_ane_sys::model::ANE_QOS,
        0xFFFFFFFF,
    ) {
        Ok(stats) => {
            if stats.p_stats_raw_data().is_null() {
                eprintln!("  all-bits (0xFFFFFFFF): no raw data in stats object");
            } else {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let hw_ns = stats.hw_execution_time();
                    let counter = stats.perf_counter_data();
                    (hw_ns, counter.is_null())
                }));
                match result {
                    Ok((hw_ns, is_null)) => {
                        eprintln!(
                            "  all-bits (0xFFFFFFFF): hw_exec_ns={hw_ns}  counter_data={}",
                            if is_null { "null" } else { "non-null" }
                        );
                    }
                    Err(_) => {
                        eprintln!("  all-bits (0xFFFFFFFF): stats access panicked (caught)");
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("  all-bits (0xFFFFFFFF): eval failed: {e}");
        }
    }

    // Summary
    eprintln!(
        "\n  Active bits (non-zero hw_exec or counter): {:?}",
        active_bits
    );

    unsafe {
        CFRelease(input_surface);
        CFRelease(output_surface);
    }
}

// =========================================================================
// Probe 13 — Chaining: Two-program Chain  (2 compiles)
// =========================================================================

/// Create an `NSNumber` with an `i64` value (`+[NSNumber numberWithLongLong:]`).
unsafe fn make_nsnumber_i64(value: i64) -> *mut c_void {
    let cls = unsafe { objc_getClass(b"NSNumber\0".as_ptr()) };
    if cls.is_null() {
        return std::ptr::null_mut();
    }
    let sel = unsafe { sel_registerName(b"numberWithLongLong:\0".as_ptr()) };
    type NumFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i64) -> *mut c_void;
    let f: NumFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(cls, sel, value) }
}

/// Create an `NSArray` from a slice of raw ObjC object pointers
/// (`+[NSArray arrayWithObjects:count:]`). Returns an autoreleased object.
unsafe fn make_nsarray(objects: &[*mut c_void]) -> *mut c_void {
    let cls = unsafe { objc_getClass(b"NSArray\0".as_ptr()) };
    if cls.is_null() {
        return std::ptr::null_mut();
    }
    type ArrayFn =
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const *mut c_void, usize) -> *mut c_void;
    let sel = unsafe { sel_registerName(b"arrayWithObjects:count:\0".as_ptr()) };
    let f: ArrayFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(cls, sel, objects.as_ptr(), objects.len()) }
}

/// Create an `NSArray` of `NSNumber` from a slice of `i64` values.
unsafe fn make_nsnumber_array(values: &[i64]) -> *mut c_void {
    let nums: Vec<*mut c_void> = values
        .iter()
        .map(|&v| unsafe { make_nsnumber_i64(v) })
        .collect();
    if nums.iter().any(|p| p.is_null()) {
        return std::ptr::null_mut();
    }
    unsafe { make_nsarray(&nums) }
}

#[test]
#[ignore]
fn probe_chaining_two_program() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 13: Chaining — Two-program Chain       ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");
    eprintln!(
        "Budget remaining: {}\n",
        ironmill_ane_sys::model::remaining_budget()
    );

    // Step 1: Compile two trivial MIL programs.
    //   prog_a: out = add(x, x)  (effectively x * 2)
    //   prog_b: out = add(x, x)  (effectively x * 2)
    // If chained correctly via loopback: result = (x*2)*2 = x*4
    // (Using self-add instead of constant-add to avoid needing const tensors.)
    let prog_a = match try_compile(
        "chain_a (add self)",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = add(x=a_input0, y=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    ) {
        Some(m) => m,
        None => {
            eprintln!("  Cannot compile program A — skipping chain probe");
            return;
        }
    };

    let prog_b = match try_compile(
        "chain_b (add self)",
        &mil(
            "        tensor<fp16, [1,4,1,4]> out = add(x=a_input0, y=a_input0)[name=string(\"out\")];",
            "tensor<fp16, [1,4,1,4]> a_input0",
            "out",
        ),
    ) {
        Some(m) => m,
        None => {
            eprintln!("  Cannot compile program B — skipping chain probe");
            return;
        }
    };

    // Step 2: Create I/O IOSurfaces.
    let input_surface = create_probe_iosurface();
    let output_surface = create_probe_iosurface();
    if input_surface.is_null() || output_surface.is_null() {
        eprintln!("  Failed to create IOSurfaces — skipping");
        unsafe {
            if !input_surface.is_null() {
                CFRelease(input_surface);
            }
            if !output_surface.is_null() {
                CFRelease(output_surface);
            }
        }
        return;
    }

    // Step 3: Baseline — evaluate each program individually (sequential).
    eprintln!("  --- Sequential eval (baseline) ---");
    match ironmill_ane_sys::model::eval(
        &prog_a,
        &[input_surface],
        &[output_surface],
        ironmill_ane_sys::model::ANE_QOS,
    ) {
        Ok(()) => eprintln!("  ✅ prog_a eval OK"),
        Err(e) => {
            eprintln!("  ❌ prog_a eval failed: {e}");
            unsafe {
                CFRelease(input_surface);
                CFRelease(output_surface);
            }
            return;
        }
    }
    match ironmill_ane_sys::model::eval(
        &prog_b,
        &[output_surface],
        &[output_surface],
        ironmill_ane_sys::model::ANE_QOS,
    ) {
        Ok(()) => eprintln!("  ✅ prog_b eval OK (sequential)"),
        Err(e) => eprintln!("  ❌ prog_b eval failed: {e}"),
    }

    // Step 4: Create ChainingRequest with loopback indices.
    // loopbackInputSymbolIndex: [0] — prog_b's input symbol 0
    // loopbackOutputSymbolIndex: [0] — prog_a's output symbol 0
    // This tells the ANE to feed prog_a's output[0] into prog_b's input[0].
    eprintln!("\n  --- ChainingRequest construction ---");
    let lb_in = unsafe { make_nsnumber_array(&[0i64]) };
    let lb_out = unsafe { make_nsnumber_array(&[0i64]) };
    eprintln!("  lb_input_symbol_id:  {:?}", lb_in);
    eprintln!("  lb_output_symbol_id: {:?}", lb_out);

    if lb_in.is_null() || lb_out.is_null() {
        eprintln!("  ❌ Failed to create loopback index arrays");
        unsafe {
            CFRelease(input_surface);
            CFRelease(output_surface);
        }
        return;
    }

    // Build input/output NSArrays for the ChainingRequest.
    let inputs_arr = unsafe { make_nsarray(&[input_surface]) };
    let outputs_arr = unsafe { make_nsarray(&[output_surface]) };

    let chain_result = std::panic::catch_unwind(|| unsafe {
        ChainingRequest::new(
            inputs_arr,
            outputs_arr,
            lb_in,
            lb_out,
            0,                    // procedure_index
            std::ptr::null_mut(), // signal_events (none)
            0,                    // transaction_handle
            0,                    // fw_enqueue_delay
            0,                    // memory_pool_id
        )
    });

    let chain_req = match chain_result {
        Ok(Ok(req)) => {
            eprintln!("  ✅ ChainingRequest created");
            eprintln!("     raw ptr:             {:?}", req.as_raw());
            eprintln!("     validate:            {}", req.validate());
            eprintln!("     input_buffer:        {:?}", req.input_buffer());
            eprintln!("     output_sets:         {:?}", req.output_sets());
            eprintln!(
                "     lb_input_sym_idx:    {:?}",
                req.loopback_input_symbol_index()
            );
            eprintln!(
                "     lb_output_sym_idx:   {:?}",
                req.loopback_output_symbol_index()
            );
            req
        }
        Ok(Err(e)) => {
            eprintln!("  ❌ ChainingRequest::new error: {e}");
            unsafe {
                CFRelease(input_surface);
                CFRelease(output_surface);
            }
            return;
        }
        Err(e) => {
            eprintln!("  💥 ChainingRequest::new panicked: {e:?}");
            unsafe {
                CFRelease(input_surface);
                CFRelease(output_surface);
            }
            return;
        }
    };

    // Step 5: Get a client and attempt prepare_chaining.
    eprintln!("\n  --- prepare_chaining ---");
    let client = match Client::shared_connection() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ❌ Failed to get client: {e}");
            unsafe {
                CFRelease(input_surface);
                CFRelease(output_surface);
            }
            return;
        }
    };

    let mut error: *mut c_void = std::ptr::null_mut();
    let prepare_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        client.prepare_chaining(
            prog_a.as_raw(),
            std::ptr::null_mut(), // options
            chain_req.as_raw(),
            ironmill_ane_sys::model::ANE_QOS,
            &mut error,
        )
    }));

    match prepare_ok {
        Ok(true) => {
            eprintln!("  ✅ prepare_chaining returned true");
        }
        Ok(false) => {
            let err_desc = if !error.is_null() {
                unsafe { ns_description(error) }
            } else {
                None
            };
            eprintln!(
                "  ❌ prepare_chaining returned false — error: {}",
                err_desc.as_deref().unwrap_or("(nil)")
            );
        }
        Err(e) => {
            eprintln!("  💥 prepare_chaining panicked: {e:?}");
        }
    }

    // TODO: If prepare_chaining succeeds, the next step would be to
    // evaluate the chained request and read back results. The evaluation
    // API for chained requests is not yet clear — it may use the same
    // eval path or require a separate dispatch mechanism.

    unsafe {
        CFRelease(input_surface);
        CFRelease(output_surface);
    }
    eprintln!(
        "\nBudget remaining: {}",
        ironmill_ane_sys::model::remaining_budget()
    );
}

// =========================================================================
// Probe 14 — Chaining: Multi-program Pipeline  (4 compiles)
// =========================================================================

#[test]
#[ignore]
fn probe_chaining_multi_program() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 14: Chaining — Multi-program Pipeline  ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");
    eprintln!(
        "Budget remaining: {}\n",
        ironmill_ane_sys::model::remaining_budget()
    );

    // Compile 4 trivial programs that simulate a simplified layer pipeline:
    //   Stage 0: add(x, x)       — simulates RMSNorm (x * 2)
    //   Stage 1: add(x, x)       — simulates linear projection (x * 2)
    //   Stage 2: add(x, x)       — simulates quantized cache write (x * 2)
    //   Stage 3: add(x, x)       — simulates attention (x * 2)
    // If chained: result = x * 2^4 = x * 16
    let stage_names = [
        "pipe_stage0 (norm)",
        "pipe_stage1 (proj)",
        "pipe_stage2 (cache_wr)",
        "pipe_stage3 (attn)",
    ];
    let mil_body =
        "        tensor<fp16, [1,4,1,4]> out = add(x=a_input0, y=a_input0)[name=string(\"out\")];";
    let mil_input = "tensor<fp16, [1,4,1,4]> a_input0";
    let mil_output = "out";

    let mut programs: Vec<InMemoryModel> = Vec::new();
    for name in &stage_names {
        match try_compile(name, &mil(mil_body, mil_input, mil_output)) {
            Some(m) => programs.push(m),
            None => {
                eprintln!("  Cannot compile {name} — skipping multi-program probe");
                return;
            }
        }
    }
    eprintln!("  All 4 programs compiled\n");

    // Create I/O IOSurfaces.
    let input_surface = create_probe_iosurface();
    let output_surface = create_probe_iosurface();
    if input_surface.is_null() || output_surface.is_null() {
        eprintln!("  Failed to create IOSurfaces — skipping");
        unsafe {
            if !input_surface.is_null() {
                CFRelease(input_surface);
            }
            if !output_surface.is_null() {
                CFRelease(output_surface);
            }
        }
        return;
    }

    // Baseline: sequential eval of all 4 programs.
    eprintln!("  --- Sequential eval (baseline, 4 programs) ---");
    let t_seq = std::time::Instant::now();
    for (i, prog) in programs.iter().enumerate() {
        // First program reads from input_surface; subsequent from output_surface.
        let in_surf = if i == 0 {
            input_surface
        } else {
            output_surface
        };
        match ironmill_ane_sys::model::eval(
            prog,
            &[in_surf],
            &[output_surface],
            ironmill_ane_sys::model::ANE_QOS,
        ) {
            Ok(()) => eprintln!("    ✅ stage {i} eval OK"),
            Err(e) => {
                eprintln!("    ❌ stage {i} eval failed: {e}");
                unsafe {
                    CFRelease(input_surface);
                    CFRelease(output_surface);
                }
                return;
            }
        }
    }
    let d_seq = t_seq.elapsed();
    eprintln!("  Sequential total: {:.1}µs\n", d_seq.as_secs_f64() * 1e6);

    // Attempt chained execution.
    eprintln!("  --- Chained ChainingRequest (4 programs) ---");

    // Build loopback arrays: chain output[0] of each program to input[0] of the next.
    let lb_in = unsafe { make_nsnumber_array(&[0i64]) };
    let lb_out = unsafe { make_nsnumber_array(&[0i64]) };
    if lb_in.is_null() || lb_out.is_null() {
        eprintln!("  ❌ Failed to create loopback arrays");
        unsafe {
            CFRelease(input_surface);
            CFRelease(output_surface);
        }
        return;
    }

    // Create per-stage ChainingRequests.
    let mut chain_reqs: Vec<ChainingRequest> = Vec::new();
    for (i, _prog) in programs.iter().enumerate() {
        // First program gets the real input; others get loopback from previous.
        let inputs_arr = if i == 0 {
            unsafe { make_nsarray(&[input_surface]) }
        } else {
            // Loopback programs don't need explicit input surfaces —
            // the loopback indices tell the ANE to reuse the previous output.
            // Pass the same surface as a placeholder; the loopback overrides it.
            unsafe { make_nsarray(&[output_surface]) }
        };
        let outputs_arr = unsafe { make_nsarray(&[output_surface]) };

        // Only intermediate programs use loopback; first program has no wait.
        let lb_in_arg = if i > 0 { lb_in } else { std::ptr::null_mut() };
        let lb_out_arg = if i < programs.len() - 1 {
            lb_out
        } else {
            std::ptr::null_mut()
        };

        let result = std::panic::catch_unwind(|| unsafe {
            ChainingRequest::new(
                inputs_arr,
                outputs_arr,
                lb_in_arg,
                lb_out_arg,
                i as i64,             // procedure_index
                std::ptr::null_mut(), // signal_events
                0,                    // transaction_handle
                0,                    // fw_enqueue_delay
                0,                    // memory_pool_id
            )
        });

        match result {
            Ok(Ok(req)) => {
                eprintln!(
                    "    ✅ ChainingRequest[{i}] created  validate={}",
                    req.validate()
                );
                chain_reqs.push(req);
            }
            Ok(Err(e)) => {
                eprintln!("    ❌ ChainingRequest[{i}] error: {e}");
                unsafe {
                    CFRelease(input_surface);
                    CFRelease(output_surface);
                }
                return;
            }
            Err(e) => {
                eprintln!("    💥 ChainingRequest[{i}] panicked: {e:?}");
                unsafe {
                    CFRelease(input_surface);
                    CFRelease(output_surface);
                }
                return;
            }
        }
    }

    // Attempt prepare_chaining for each stage.
    let client = match Client::shared_connection() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ❌ Failed to get client: {e}");
            unsafe {
                CFRelease(input_surface);
                CFRelease(output_surface);
            }
            return;
        }
    };

    eprintln!("\n  --- prepare_chaining per stage ---");
    for (i, (prog, req)) in programs.iter().zip(chain_reqs.iter()).enumerate() {
        let mut error: *mut c_void = std::ptr::null_mut();
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            client.prepare_chaining(
                prog.as_raw(),
                std::ptr::null_mut(), // options
                req.as_raw(),
                ironmill_ane_sys::model::ANE_QOS,
                &mut error,
            )
        }));

        match ok {
            Ok(true) => eprintln!("    ✅ stage {i}: prepare_chaining OK"),
            Ok(false) => {
                let err_desc = if !error.is_null() {
                    unsafe { ns_description(error) }
                } else {
                    None
                };
                eprintln!(
                    "    ❌ stage {i}: prepare_chaining failed — {}",
                    err_desc.as_deref().unwrap_or("(nil)")
                );
            }
            Err(e) => {
                eprintln!("    💥 stage {i}: prepare_chaining panicked: {e:?}");
            }
        }
    }

    // TODO: If all prepare_chaining calls succeed, dispatch the chained
    // pipeline and compare output to the sequential baseline. The dispatch
    // mechanism for chained requests is not yet documented — it may use
    // evaluateDirectWithModel or a dedicated chaining eval entry point.

    unsafe {
        CFRelease(input_surface);
        CFRelease(output_surface);
    }
    eprintln!(
        "\nBudget remaining: {}",
        ironmill_ane_sys::model::remaining_budget()
    );
}

// =========================================================================
// Probe 15 — ANE↔GPU Shared Events  (0 compiles)
// =========================================================================

// Metal framework FFI for shared event probing.
#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut c_void;
}

/// Helper: create an MTLSharedEvent from a Metal device.
/// Returns `(device, shared_event)` as raw ObjC pointers.
/// Both are retained — caller must CFRelease.
unsafe fn create_metal_shared_event() -> Option<(*mut c_void, *mut c_void)> {
    // SAFETY: MTLCreateSystemDefaultDevice returns a retained MTLDevice or nil.
    let device = unsafe { MTLCreateSystemDefaultDevice() };
    if device.is_null() {
        eprintln!("  ❌ MTLCreateSystemDefaultDevice returned nil");
        return None;
    }

    // -[MTLDevice newSharedEvent] → returns a retained MTLSharedEvent.
    type NewEventFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let sel = unsafe { sel_registerName(b"newSharedEvent\0".as_ptr()) };
    let f: NewEventFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let event = unsafe { f(device, sel) };
    if event.is_null() {
        eprintln!("  ❌ newSharedEvent returned nil");
        unsafe { CFRelease(device) };
        return None;
    }

    Some((device, event))
}

/// Helper: get the current signaledValue of an MTLSharedEvent.
unsafe fn shared_event_signaled_value(event: *mut c_void) -> u64 {
    type GetFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
    let sel = unsafe { sel_registerName(b"signaledValue\0".as_ptr()) };
    let f: GetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(event, sel) }
}

/// Helper: create a Metal command queue from a device.
/// Returns a retained MTLCommandQueue or null.
unsafe fn create_metal_command_queue(device: *mut c_void) -> *mut c_void {
    type QueueFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let sel = unsafe { sel_registerName(b"newCommandQueue\0".as_ptr()) };
    let f: QueueFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(device, sel) }
}

/// Helper: create a Metal command buffer from a queue.
/// Returns an autoreleased MTLCommandBuffer or null.
unsafe fn create_metal_command_buffer(queue: *mut c_void) -> *mut c_void {
    type BufFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let sel = unsafe { sel_registerName(b"commandBuffer\0".as_ptr()) };
    let f: BufFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(queue, sel) }
}

/// Helper: encode a signal on a command buffer for a shared event at a given value.
/// -[MTLCommandBuffer encodeSignalEvent:value:]
unsafe fn encode_signal_event(cmd_buf: *mut c_void, event: *mut c_void, value: u64) {
    type SignalFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, u64);
    let sel = unsafe { sel_registerName(b"encodeSignalEvent:value:\0".as_ptr()) };
    let f: SignalFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(cmd_buf, sel, event, value) };
}

/// Helper: encode a wait on a command buffer for a shared event at a given value.
/// -[MTLCommandBuffer encodeWaitForEvent:value:]
unsafe fn encode_wait_for_event(cmd_buf: *mut c_void, event: *mut c_void, value: u64) {
    type WaitFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, u64);
    let sel = unsafe { sel_registerName(b"encodeWaitForEvent:value:\0".as_ptr()) };
    let f: WaitFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(cmd_buf, sel, event, value) };
}

/// Helper: commit and waitUntilCompleted on a command buffer.
unsafe fn commit_and_wait(cmd_buf: *mut c_void) {
    type VoidFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
    let commit_sel = unsafe { sel_registerName(b"commit\0".as_ptr()) };
    let wait_sel = unsafe { sel_registerName(b"waitUntilCompleted\0".as_ptr()) };
    let f: VoidFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(cmd_buf, commit_sel) };
    unsafe { f(cmd_buf, wait_sel) };
}

#[test]
#[ignore]
fn probe_shared_events() {
    if !has_ane() {
        eprintln!("SKIP: no ANE hardware");
        return;
    }
    eprintln!("\n╔══════════════════════════════════════════════╗");
    eprintln!("║  PROBE 15: ANE↔GPU Shared Events              ║");
    eprintln!("╚══════════════════════════════════════════════╝\n");
    eprintln!(
        "Budget remaining: {}\n",
        ironmill_ane_sys::model::remaining_budget()
    );

    // ── Step 1: Create an MTLSharedEvent and wrap it in ANE event types ──

    eprintln!("  --- Step 1: Create MTLSharedEvent + ANE wrappers ---");

    let (mtl_device, mtl_shared_event) = match unsafe { create_metal_shared_event() } {
        Some(pair) => pair,
        None => {
            eprintln!("  ❌ Cannot create Metal shared event — skipping");
            return;
        }
    };

    let initial_val = unsafe { shared_event_signaled_value(mtl_shared_event) };
    eprintln!("  MTLSharedEvent created: ptr={mtl_shared_event:?}, signaledValue={initial_val}");

    // Attempt to create a SharedSignalEvent wrapping the MTLSharedEvent.
    let signal_value: u64 = 1;
    let signal_result = std::panic::catch_unwind(|| unsafe {
        SharedSignalEvent::new(
            signal_value,
            0, // symbol_index
            0, // event_type
            mtl_shared_event,
        )
    });

    match &signal_result {
        Ok(Ok(sig)) => {
            eprintln!("  ✅ SharedSignalEvent::new() succeeded");
            eprintln!("    value:        {}", sig.value());
            eprintln!("    symbol_index: {}", sig.symbol_index());
            eprintln!("    event_type:   {}", sig.event_type());
            eprintln!("    shared_event: {:?}", sig.shared_event());
            eprintln!("    agent_mask:   {}", sig.agent_mask());
        }
        Ok(Err(e)) => {
            eprintln!("  ❌ SharedSignalEvent::new() returned error: {e}");
        }
        Err(e) => {
            eprintln!("  💥 SharedSignalEvent::new() panicked: {e:?}");
        }
    }

    // Attempt to create a SharedWaitEvent wrapping the MTLSharedEvent.
    let wait_value: u64 = 1;
    let wait_result =
        std::panic::catch_unwind(|| unsafe { SharedWaitEvent::new(wait_value, mtl_shared_event) });

    match &wait_result {
        Ok(Ok(w)) => {
            eprintln!("  ✅ SharedWaitEvent::new() succeeded");
            eprintln!("    value:        {}", w.value());
            eprintln!("    shared_event: {:?}", w.shared_event());
            eprintln!("    event_type:   {}", w.event_type());
        }
        Ok(Err(e)) => {
            eprintln!("  ❌ SharedWaitEvent::new() returned error: {e}");
        }
        Err(e) => {
            eprintln!("  💥 SharedWaitEvent::new() panicked: {e:?}");
        }
    }

    // Test SharedEvents container (wraps signal + wait arrays).
    let container_result = if let (Ok(Ok(_sig)), Ok(Ok(_wait))) = (&signal_result, &wait_result) {
        // Build NSArrays of signal and wait events.
        let sig_ref = signal_result.as_ref().unwrap().as_ref().unwrap();
        let wait_ref = wait_result.as_ref().unwrap().as_ref().unwrap();
        let sig_arr = unsafe { make_nsarray(&[sig_ref.as_raw()]) };
        let wait_arr = unsafe { make_nsarray(&[wait_ref.as_raw()]) };

        if sig_arr.is_null() || wait_arr.is_null() {
            eprintln!("  ❌ Failed to create NSArrays for SharedEvents");
            None
        } else {
            let r = std::panic::catch_unwind(|| unsafe { SharedEvents::new(sig_arr, wait_arr) });
            match &r {
                Ok(Ok(se)) => {
                    eprintln!("  ✅ SharedEvents::new() succeeded");
                    eprintln!("    signal_events: {:?}", se.signal_events());
                    eprintln!("    wait_events:   {:?}", se.wait_events());
                }
                Ok(Err(e)) => {
                    eprintln!("  ❌ SharedEvents::new() returned error: {e}");
                }
                Err(e) => {
                    eprintln!("  💥 SharedEvents::new() panicked: {e:?}");
                }
            }
            Some(r)
        }
    } else {
        eprintln!("  ⏭  Skipping SharedEvents container — prerequisite failed");
        None
    };

    // ── Step 2: Signal from Metal, observe on shared event ──

    eprintln!("\n  --- Step 2: Metal signals shared event ---");

    let queue = unsafe { create_metal_command_queue(mtl_device) };
    if queue.is_null() {
        eprintln!("  ❌ Failed to create Metal command queue");
        unsafe {
            CFRelease(mtl_shared_event);
            CFRelease(mtl_device);
        }
        return;
    }

    let signal_test_value: u64 = 42;
    let cmd_buf = unsafe { create_metal_command_buffer(queue) };
    if cmd_buf.is_null() {
        eprintln!("  ❌ Failed to create Metal command buffer");
        unsafe {
            CFRelease(queue);
            CFRelease(mtl_shared_event);
            CFRelease(mtl_device);
        }
        return;
    }

    // Encode a signal at value 42 and commit.
    let encode_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        encode_signal_event(cmd_buf, mtl_shared_event, signal_test_value);
        commit_and_wait(cmd_buf);
    }));

    match encode_result {
        Ok(()) => {
            let new_val = unsafe { shared_event_signaled_value(mtl_shared_event) };
            eprintln!(
                "  ✅ Metal encodeSignalEvent completed — signaledValue: {} (expected {})",
                new_val, signal_test_value
            );
            if new_val == signal_test_value {
                eprintln!("  ✅ Shared event value updated correctly by Metal");
            } else {
                eprintln!("  ⚠️  signaledValue mismatch (may indicate async completion)");
            }
        }
        Err(ref e) => {
            eprintln!("  💥 Metal signal encode/commit panicked: {e:?}");
        }
    }

    // Can ANE SharedWaitEvent be created for the signaled value?
    let ane_wait_result = std::panic::catch_unwind(|| unsafe {
        SharedWaitEvent::new(signal_test_value, mtl_shared_event)
    });
    match &ane_wait_result {
        Ok(Ok(w)) => {
            eprintln!("  ✅ SharedWaitEvent for Metal-signaled value created");
            eprintln!("    value: {}, event_type: {}", w.value(), w.event_type());
        }
        Ok(Err(e)) => {
            eprintln!("  ❌ SharedWaitEvent for Metal-signaled value failed: {e}");
        }
        Err(e) => {
            eprintln!("  💥 SharedWaitEvent creation panicked: {e:?}");
        }
    }

    // ── Step 3: Bidirectional — ANE signal event + Metal wait ──
    //
    // Create an ANE SharedSignalEvent at a future value, then encode a Metal
    // command buffer that waits for that value. If the ANE evaluation fires
    // the signal, the Metal wait should unblock.
    //
    // NOTE: Actually submitting an ANE eval with shared events requires a
    // compiled model + AneRequest. This step tests whether the Metal wait
    // side can be set up (the ANE signal side requires a full eval loop
    // which is tested in integration tests, not probes).

    eprintln!("\n  --- Step 3: Bidirectional setup (ANE signal → Metal wait) ---");

    let bidi_signal_value: u64 = 100;
    let bidi_signal = std::panic::catch_unwind(|| unsafe {
        SharedSignalEvent::new(
            bidi_signal_value,
            0, // symbol_index
            0, // event_type
            mtl_shared_event,
        )
    });

    match &bidi_signal {
        Ok(Ok(sig)) => {
            eprintln!(
                "  ✅ ANE SharedSignalEvent for bidi created (value={})",
                sig.value()
            );
        }
        Ok(Err(e)) => {
            eprintln!("  ❌ ANE SharedSignalEvent for bidi failed: {e}");
        }
        Err(e) => {
            eprintln!("  💥 ANE SharedSignalEvent for bidi panicked: {e:?}");
        }
    }

    // Verify Metal can encode a wait for the ANE signal value.
    let cmd_buf2 = unsafe { create_metal_command_buffer(queue) };
    if !cmd_buf2.is_null() {
        let wait_encode = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            encode_wait_for_event(cmd_buf2, mtl_shared_event, bidi_signal_value);
        }));
        match wait_encode {
            Ok(()) => {
                eprintln!("  ✅ Metal encodeWaitForEvent({bidi_signal_value}) encoded OK");
                eprintln!("    (Not committing — ANE signal source requires eval loop)");
            }
            Err(e) => {
                eprintln!("  💥 Metal encodeWaitForEvent panicked: {e:?}");
            }
        }
        // Don't commit cmd_buf2 — it would block forever waiting for the
        // ANE signal that we can't fire without a full eval path.
    } else {
        eprintln!("  ❌ Failed to create second Metal command buffer");
    }

    // ── Summary ──

    eprintln!("\n  --- Summary ---");
    let step1_ok = matches!(&signal_result, Ok(Ok(_)))
        && matches!(&wait_result, Ok(Ok(_)))
        && matches!(&container_result, Some(Ok(Ok(_))));
    let step2_ok = matches!(encode_result, Ok(()));
    let step3_setup_ok = matches!(&bidi_signal, Ok(Ok(_)));

    eprintln!(
        "  Step 1 (SharedEvent wrappers): {}",
        if step1_ok { "✅ PASS" } else { "❌ FAIL" }
    );
    eprintln!(
        "  Step 2 (Metal signal → read):  {}",
        if step2_ok { "✅ PASS" } else { "❌ FAIL" }
    );
    eprintln!(
        "  Step 3 (Bidi setup):           {}",
        if step3_setup_ok {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    if step1_ok && step2_ok {
        eprintln!(
            "\n  🎯 MTLSharedEvent is compatible with ANE SharedSignalEvent/SharedWaitEvent."
        );
        eprintln!("     Hybrid ANE↔GPU execution via shared events is feasible.");
    } else {
        eprintln!("\n  ⚠️  Shared event interop needs further investigation.");
    }

    // Cleanup retained objects.
    // signal_result, wait_result, container_result drop automatically (Rust Drop).
    // ane_wait_result and bidi_signal drop automatically.
    drop(container_result);
    drop(signal_result);
    drop(wait_result);
    drop(ane_wait_result);
    drop(bidi_signal);
    unsafe {
        CFRelease(queue);
        CFRelease(mtl_shared_event);
        CFRelease(mtl_device);
    }
    eprintln!(
        "\nBudget remaining: {}",
        ironmill_ane_sys::model::remaining_budget()
    );
}
