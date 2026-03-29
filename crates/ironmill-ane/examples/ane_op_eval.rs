//! ANE op eval verification — tests that ops produce correct numerical results.
//!
//! This goes beyond compile-testing: it writes known inputs, runs them through
//! the ANE, reads outputs, and compares against CPU reference values.
//!
//! Run with: cargo run -p ironmill-ane --example ane_op_eval
//!
//! Re-run after macOS updates to detect behavioural changes in the ANE compiler.

use half::f16;
use ironmill_ane::program::CompiledProgram;
use ironmill_ane::runtime::AneRuntime;
use ironmill_ane::tensor::AneTensor;
use mil_rs::ffi::ane::AneCompiler;
use mil_rs::ir::ScalarType;

// ── MIL text helpers ────────────────────────────────────────────────────

const BUILD_INFO: &str = r#"[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]"#;

fn mil_program(body: &str, inputs: &str, output: &str) -> String {
    format!(
        "program(1.3)\n{BUILD_INFO}\n{{\n    func main<ios18>({inputs}) {{\n{body}\n    }} -> ({output});\n}}"
    )
}

// ── Test infrastructure ─────────────────────────────────────────────────

/// Compile MIL text, load it, and return (runtime, loaded_program).
fn compile_and_load(mil_text: &str) -> Option<(AneRuntime, ironmill_ane::program::LoadedProgram)> {
    let ptr = match AneCompiler::compile_mil_text(mil_text, &[]) {
        Ok(ptr) => ptr,
        Err(e) => {
            eprintln!("    compile failed: {e}");
            return None;
        }
    };

    let runtime = match AneRuntime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("    runtime init failed: {e}");
            return None;
        }
    };

    let compiled = unsafe { CompiledProgram::from_raw(ptr) };
    let loaded = match runtime.load_program(&compiled) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("    load failed: {e}");
            return None;
        }
    };

    Some((runtime, loaded))
}

/// Run an ANE program with the given f16 input vectors.
/// `in_shapes` and `out_shapes` are (channels, seq_len) pairs.
fn run_ane(
    mil_text: &str,
    inputs_data: &[Vec<f16>],
    in_shapes: &[(usize, usize)],
    out_shapes: &[(usize, usize)],
) -> Option<Vec<Vec<f16>>> {
    let (runtime, loaded) = compile_and_load(mil_text)?;

    // Compute uniform alloc sizes
    let max_in_alloc = in_shapes
        .iter()
        .map(|(c, s)| (c * s * 2).max(49152))
        .max()
        .unwrap_or(49152);
    let max_out_alloc = out_shapes
        .iter()
        .map(|(c, s)| (c * s * 2).max(49152))
        .max()
        .unwrap_or(49152);

    let mut in_tensors: Vec<AneTensor> = in_shapes
        .iter()
        .map(|&(c, s)| {
            AneTensor::new_with_min_alloc(c, s, ScalarType::Float16, max_in_alloc).unwrap()
        })
        .collect();

    let mut out_tensors: Vec<AneTensor> = out_shapes
        .iter()
        .map(|&(c, s)| {
            AneTensor::new_with_min_alloc(c, s, ScalarType::Float16, max_out_alloc).unwrap()
        })
        .collect();

    // Write inputs
    for (tensor, data) in in_tensors.iter_mut().zip(inputs_data.iter()) {
        tensor.write_f16(data).unwrap();
    }

    let in_refs: Vec<&AneTensor> = in_tensors.iter().collect();
    let mut out_refs: Vec<&mut AneTensor> = out_tensors.iter_mut().collect();

    match runtime.eval(&loaded, &in_refs, &mut out_refs) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("    eval failed: {e}");
            return None;
        }
    }

    // Read outputs
    let results: Vec<Vec<f16>> = out_tensors.iter().map(|t| t.read_f16().unwrap()).collect();
    Some(results)
}

/// Compare f16 results against f32 reference values with tolerance.
fn check_results(name: &str, actual: &[f16], expected: &[f32], atol: f32) -> bool {
    if actual.len() != expected.len() {
        println!(
            "    ❌ {name}: length mismatch (got {}, expected {})",
            actual.len(),
            expected.len()
        );
        return false;
    }
    let mut max_err: f32 = 0.0;
    let mut fail_idx: Option<usize> = None;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_f32 = a.to_f32();
        let err = (a_f32 - e).abs();
        if err > max_err {
            max_err = err;
        }
        if err > atol && fail_idx.is_none() {
            fail_idx = Some(i);
        }
    }
    if let Some(i) = fail_idx {
        println!(
            "    ❌ {name}: FAIL at [{i}] got={:.4} expected={:.4} (max_err={:.6}, atol={atol})",
            actual[i].to_f32(),
            expected[i],
            max_err
        );
        false
    } else {
        println!("    ✅ {name}: PASS (max_err={max_err:.6}, atol={atol})");
        true
    }
}

/// Compare bool results (stored as f16: 0.0=false, 1.0=true).
#[allow(dead_code)]
fn check_bool_results(name: &str, actual: &[f16], expected: &[bool]) -> bool {
    if actual.len() != expected.len() {
        println!(
            "    ❌ {name}: length mismatch (got {}, expected {})",
            actual.len(),
            expected.len()
        );
        return false;
    }
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_bool = a.to_f32() != 0.0;
        if a_bool != e {
            println!("    ❌ {name}: FAIL at [{i}] got={a_bool} expected={e}");
            return false;
        }
    }
    println!("    ✅ {name}: PASS");
    true
}

// ── Helpers to build f16 input data ─────────────────────────────────────

fn f16v(vals: &[f32]) -> Vec<f16> {
    vals.iter().map(|&v| f16::from_f32(v)).collect()
}

/// Pad an f16 vector to fill a [1, C, 1, S] tensor.
#[allow(dead_code)]
fn f16_padded(vals: &[f32], channels: usize, seq_len: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; channels * seq_len];
    for (i, &v) in vals.iter().enumerate() {
        if i < out.len() {
            out[i] = f16::from_f32(v);
        }
    }
    out
}

// ── Test cases ──────────────────────────────────────────────────────────

const C: usize = 32;
const S: usize = 32;
const N: usize = C * S; // 1024 elements

fn test_add() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<fp16, [1,32,1,32]> z_output0 = add(x=a_input0, y=a_input1)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..N).map(|i| (N - i) as f32 * 0.1).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    match run_ane(&mil, &[f16v(&a), f16v(&b)], &[(C, S), (C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("add", &out[0], &expected, 0.05)),
        None => (false, false),
    }
}

fn test_sub() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<fp16, [1,32,1,32]> z_output0 = sub(x=a_input0, y=a_input1)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..N).map(|i| i as f32 * 0.3).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

    match run_ane(&mil, &[f16v(&a), f16v(&b)], &[(C, S), (C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("sub", &out[0], &expected, 0.15)),
        None => (false, false),
    }
}

fn test_mul() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<fp16, [1,32,1,32]> z_output0 = mul(x=a_input0, y=a_input1)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| (i as f32 * 0.01) - 5.0).collect();
    let b: Vec<f32> = (0..N).map(|i| (i as f32 * 0.02) - 10.0).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

    match run_ane(&mil, &[f16v(&a), f16v(&b)], &[(C, S), (C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("mul", &out[0], &expected, 0.5)),
        None => (false, false),
    }
}

fn test_relu() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<fp16, [1,32,1,32]> z_output0 = relu(x=a_input0)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| (i as f32) - (N as f32 / 2.0)).collect();
    let expected: Vec<f32> = a.iter().map(|&x| x.max(0.0)).collect();

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("relu", &out[0], &expected, 0.01)),
        None => (false, false),
    }
}

fn test_abs() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<fp16, [1,32,1,32]> z_output0 = abs(x=a_input0)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| (i as f32) - (N as f32 / 2.0)).collect();
    let expected: Vec<f32> = a.iter().map(|&x| x.abs()).collect();

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("abs", &out[0], &expected, 0.01)),
        None => (false, false),
    }
}

fn test_sign() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<fp16, [1,32,1,32]> z_output0 = sign(x=a_input0)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N)
        .map(|i| {
            if i < N / 3 {
                -(i as f32 + 1.0)
            } else if i < 2 * N / 3 {
                0.0
            } else {
                i as f32 + 1.0
            }
        })
        .collect();
    // sign(0) = 0 on ANE (mathematically correct), not 1.0 as Rust's signum returns
    let expected: Vec<f32> = a
        .iter()
        .map(|&x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect();

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("sign", &out[0], &expected, 0.01)),
        None => (false, false),
    }
}

fn test_sqrt() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<fp16, [1,32,1,32]> z_output0 = sqrt(x=a_input0)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let expected: Vec<f32> = a.iter().map(|&x| x.sqrt()).collect();

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("sqrt", &out[0], &expected, 0.05)),
        None => (false, false),
    }
}

fn test_exp() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<fp16, [1,32,1,32]> z_output0 = exp(x=a_input0)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    // Keep values small to stay in fp16 range
    let a: Vec<f32> = (0..N).map(|i| (i as f32 * 0.01) - 5.0).collect();
    let expected: Vec<f32> = a.iter().map(|&x| x.exp()).collect();

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("exp", &out[0], &expected, 0.5)),
        None => (false, false),
    }
}

fn test_greater() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<bool, [1,32,1,32]> cmp = greater(x=a_input0, y=a_input1)[name=string(\"cmp\")];\n\
                 tensor<fp16, [1,32,1,32]> z_output0 = cast(x=cmp, dtype=string(\"fp16\"))[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| (i % 7) as f32).collect();
    let b: Vec<f32> = (0..N).map(|_| 3.0f32).collect();
    let expected: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| if x > y { 1.0 } else { 0.0 })
        .collect();

    match run_ane(&mil, &[f16v(&a), f16v(&b)], &[(C, S), (C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("greater", &out[0], &expected, 0.01)),
        None => (false, false),
    }
}

fn test_select() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<bool, [1,32,1,32]> cmp = greater(x=a_input0, y=a_input1)[name=string(\"cmp\")];\n\
                 fp16 pos = const()[name=string(\"pos\"), val=fp16(1.0)];\n\
                 fp16 neg = const()[name=string(\"neg\"), val=fp16(-1.0)];\n\
                 tensor<fp16, [1,32,1,32]> z_output0 = select(cond=cmp, a=pos, b=neg)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| (i as f32) - (N as f32 / 2.0)).collect();
    let b: Vec<f32> = vec![0.0; N];
    let expected: Vec<f32> = a
        .iter()
        .map(|&x| if x > 0.0 { 1.0 } else { -1.0 })
        .collect();

    match run_ane(&mil, &[f16v(&a), f16v(&b)], &[(C, S), (C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("select", &out[0], &expected, 0.01)),
        None => (false, false),
    }
}

fn test_reduce_sum() -> (bool, bool) {
    // Reduce along spatial axis, then broadcast back to full size to avoid
    // small-output-tensor alignment issues with ANE IOSurface layout.
    let mil = mil_program(
        "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n\
                 bool kd = const()[name=string(\"kd\"), val=bool(true)];\n\
                 tensor<fp16, [1,32,1,1]> rs = reduce_sum(x=a_input0, axes=rax, keep_dims=kd)[name=string(\"rs\")];\n\
                 tensor<int32, [4]> rep = const()[name=string(\"rep\"), val=tensor<int32, [4]>([1,1,1,32])];\n\
                 tensor<fp16, [1,32,1,32]> z_output0 = tile(x=rs, reps=rep)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| ((i % S) as f32 + 1.0) * 0.1).collect();
    // Each channel c sums S values, then that sum is tiled across S positions.
    let mut expected = vec![0.0f32; N];
    for c in 0..C {
        let start = c * S;
        let channel_sum: f32 = a[start..start + S].iter().sum();
        for s in 0..S {
            expected[start + s] = channel_sum;
        }
    }

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("reduce_sum", &out[0], &expected, 1.0)),
        None => (false, false),
    }
}

fn test_reduce_mean() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n\
                 bool kd = const()[name=string(\"kd\"), val=bool(true)];\n\
                 tensor<fp16, [1,32,1,1]> rm = reduce_mean(x=a_input0, axes=rax, keep_dims=kd)[name=string(\"rm\")];\n\
                 tensor<int32, [4]> rep = const()[name=string(\"rep\"), val=tensor<int32, [4]>([1,1,1,32])];\n\
                 tensor<fp16, [1,32,1,32]> z_output0 = tile(x=rm, reps=rep)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| ((i % S) as f32 + 1.0) * 0.1).collect();
    let mut expected = vec![0.0f32; N];
    for c in 0..C {
        let start = c * S;
        let channel_mean: f32 = a[start..start + S].iter().sum::<f32>() / S as f32;
        for s in 0..S {
            expected[start + s] = channel_mean;
        }
    }

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("reduce_mean", &out[0], &expected, 0.05)),
        None => (false, false),
    }
}

fn test_reduce_max() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n\
                 bool kd = const()[name=string(\"kd\"), val=bool(true)];\n\
                 tensor<fp16, [1,32,1,1]> rm = reduce_max(x=a_input0, axes=rax, keep_dims=kd)[name=string(\"rm\")];\n\
                 tensor<int32, [4]> rep = const()[name=string(\"rep\"), val=tensor<int32, [4]>([1,1,1,32])];\n\
                 tensor<fp16, [1,32,1,32]> z_output0 = tile(x=rm, reps=rep)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| ((i % S) as f32 - 16.0) * 0.5).collect();
    let mut expected = vec![0.0f32; N];
    for c in 0..C {
        let start = c * S;
        let channel_max = a[start..start + S]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        for s in 0..S {
            expected[start + s] = channel_max;
        }
    }

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("reduce_max", &out[0], &expected, 0.01)),
        None => (false, false),
    }
}

fn test_pow() -> (bool, bool) {
    let mil = mil_program(
        "        fp16 sc = const()[name=string(\"sc\"), val=fp16(-0.5)];\n\
                 tensor<fp16, [1,32,1,32]> z_output0 = pow(x=a_input0, y=sc)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let expected: Vec<f32> = a.iter().map(|&x| x.powf(-0.5)).collect();

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (
            true,
            check_results("pow(-0.5) [rsqrt]", &out[0], &expected, 0.1),
        ),
        None => (false, false),
    }
}

fn test_softmax() -> (bool, bool) {
    let mil = mil_program(
        "        int32 ax = const()[name=string(\"ax\"), val=int32(-1)];\n\
                 tensor<fp16, [1,32,1,32]> z_output0 = softmax(x=a_input0, axis=ax)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| ((i % S) as f32 - 16.0) * 0.2).collect();
    // Compute per-channel softmax over S elements
    let mut expected = vec![0.0f32; N];
    for c in 0..C {
        let start = c * S;
        let max_val = a[start..start + S]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = a[start..start + S]
            .iter()
            .map(|&x| (x - max_val).exp())
            .collect::<Vec<_>>()
            .iter()
            .sum();
        for s in 0..S {
            expected[start + s] = (a[start + s] - max_val).exp() / exp_sum;
        }
    }

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("softmax", &out[0], &expected, 0.01)),
        None => (false, false),
    }
}

fn test_clip() -> (bool, bool) {
    let mil = mil_program(
        "        fp16 lo = const()[name=string(\"lo\"), val=fp16(-1.0)];\n\
                 fp16 hi = const()[name=string(\"hi\"), val=fp16(1.0)];\n\
                 tensor<fp16, [1,32,1,32]> z_output0 = clip(x=a_input0, alpha=lo, beta=hi)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N)
        .map(|i| (i as f32 - (N as f32 / 2.0)) * 0.01)
        .collect();
    let expected: Vec<f32> = a.iter().map(|&x| x.clamp(-1.0, 1.0)).collect();

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("clip", &out[0], &expected, 0.01)),
        None => (false, false),
    }
}

fn test_maximum() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<fp16, [1,32,1,32]> z_output0 = maximum(x=a_input0, y=a_input1)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0, tensor<fp16, [1,32,1,32]> a_input1",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| (i as f32 - 500.0) * 0.1).collect();
    let b: Vec<f32> = (0..N).map(|i| (700.0 - i as f32) * 0.1).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x.max(y)).collect();

    match run_ane(&mil, &[f16v(&a), f16v(&b)], &[(C, S), (C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("maximum", &out[0], &expected, 0.05)),
        None => (false, false),
    }
}

fn test_erf() -> (bool, bool) {
    let mil = mil_program(
        "        tensor<fp16, [1,32,1,32]> z_output0 = erf(x=a_input0)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N)
        .map(|i| (i as f32 - (N as f32 / 2.0)) * 0.005)
        .collect();
    // Use a polynomial approx for erf reference — or just compute via libm
    fn erf_ref(x: f32) -> f32 {
        // Abramowitz and Stegun approximation
        let sign = x.signum();
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        sign * (1.0 - poly * (-x * x).exp())
    }
    let expected: Vec<f32> = a.iter().map(|&x| erf_ref(x)).collect();

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (true, check_results("erf", &out[0], &expected, 0.02)),
        None => (false, false),
    }
}

// ── Composite TurboQuant patterns ───────────────────────────────────────

fn test_qjl_sign_extraction() -> (bool, bool) {
    // QJL sign extraction: sign(x) → {+1, -1}
    // Equivalent to: select(greater(x, 0), 1.0, -1.0)
    let mil = mil_program(
        "        fp16 zero = const()[name=string(\"zero\"), val=fp16(0.0)];\n\
                 tensor<bool, [1,32,1,32]> pos = greater(x=a_input0, y=zero)[name=string(\"pos\")];\n\
                 fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n\
                 fp16 neg = const()[name=string(\"neg\"), val=fp16(-1.0)];\n\
                 tensor<fp16, [1,32,1,32]> z_output0 = select(cond=pos, a=one, b=neg)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N)
        .map(|i| {
            let v = (i as f32 - (N as f32 / 2.0)) * 0.1;
            if v == 0.0 { 0.001 } else { v } // avoid exact zero
        })
        .collect();
    let expected: Vec<f32> = a
        .iter()
        .map(|&x| if x > 0.0 { 1.0 } else { -1.0 })
        .collect();

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (
            true,
            check_results("QJL sign extraction", &out[0], &expected, 0.01),
        ),
        None => (false, false),
    }
}

fn test_rmsnorm_pattern() -> (bool, bool) {
    // RMSNorm: x * rsqrt(mean(x^2) + eps)
    // This is the core PolarQuant normalization step.
    let mil = mil_program(
        "        tensor<fp16, [1,32,1,32]> sq = mul(x=a_input0, y=a_input0)[name=string(\"sq\")];\n\
                 tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n\
                 bool kd = const()[name=string(\"kd\"), val=bool(true)];\n\
                 tensor<fp16, [1,32,1,1]> ms = reduce_mean(x=sq, axes=rax, keep_dims=kd)[name=string(\"ms\")];\n\
                 fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n\
                 tensor<fp16, [1,32,1,1]> mse = add(x=ms, y=eps)[name=string(\"mse\")];\n\
                 fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n\
                 tensor<fp16, [1,32,1,1]> rrms = pow(x=mse, y=nhalf)[name=string(\"rrms\")];\n\
                 tensor<fp16, [1,32,1,32]> z_output0 = mul(x=a_input0, y=rrms)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| ((i % S) as f32 - 16.0) * 0.2).collect();
    // Compute RMSNorm reference per channel
    let mut expected = vec![0.0f32; N];
    for c in 0..C {
        let start = c * S;
        let mean_sq: f32 = a[start..start + S].iter().map(|&x| x * x).sum::<f32>() / S as f32;
        let rrms = 1.0 / (mean_sq + 1e-5f32).sqrt();
        for s in 0..S {
            expected[start + s] = a[start + s] * rrms;
        }
    }

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (
            true,
            check_results("RMSNorm (PolarQuant)", &out[0], &expected, 0.1),
        ),
        None => (false, false),
    }
}

fn test_affine_quantize_pattern() -> (bool, bool) {
    // Affine quantization: round(clip(x / scale + zero_point, 0, 255)) * scale - zero_point * scale
    // Simplified: round(x * inv_scale) then clamp, replicating the dequant path
    let mil = mil_program(
        "        fp16 scale = const()[name=string(\"scale\"), val=fp16(0.05)];\n\
                 tensor<fp16, [1,32,1,32]> scaled = mul(x=a_input0, y=scale)[name=string(\"scaled\")];\n\
                 tensor<fp16, [1,32,1,32]> rounded = round(x=scaled)[name=string(\"rounded\")];\n\
                 fp16 lo = const()[name=string(\"lo\"), val=fp16(-128.0)];\n\
                 fp16 hi = const()[name=string(\"hi\"), val=fp16(127.0)];\n\
                 tensor<fp16, [1,32,1,32]> z_output0 = clip(x=rounded, alpha=lo, beta=hi)[name=string(\"z_output0\")];",
        "tensor<fp16, [1,32,1,32]> a_input0",
        "z_output0",
    );
    let a: Vec<f32> = (0..N).map(|i| (i as f32 - 512.0) * 10.0).collect();
    let expected: Vec<f32> = a
        .iter()
        .map(|&x| (x * 0.05).round().clamp(-128.0, 127.0))
        .collect();

    match run_ane(&mil, &[f16v(&a)], &[(C, S)], &[(C, S)]) {
        Some(out) => (
            true,
            check_results("affine quantize", &out[0], &expected, 1.0),
        ),
        None => (false, false),
    }
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║             ANE Op Eval Verification                            ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Tests compile + eval + numerical correctness on real ANE       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let tests: Vec<(&str, fn() -> (bool, bool))> = vec![
        // Basic ops
        ("add", test_add),
        ("sub", test_sub),
        ("mul", test_mul),
        ("relu", test_relu),
        ("abs", test_abs),
        ("sign", test_sign),
        ("sqrt", test_sqrt),
        ("exp", test_exp),
        ("pow", test_pow),
        ("clip", test_clip),
        ("maximum", test_maximum),
        ("erf", test_erf),
        ("softmax", test_softmax),
        // Comparisons & conditional
        ("greater", test_greater),
        ("select", test_select),
        // Reductions
        ("reduce_sum", test_reduce_sum),
        ("reduce_mean", test_reduce_mean),
        ("reduce_max", test_reduce_max),
        // TurboQuant composite patterns
        ("QJL sign extract", test_qjl_sign_extraction),
        ("RMSNorm", test_rmsnorm_pattern),
        ("affine quantize", test_affine_quantize_pattern),
    ];

    let mut compile_pass = 0;
    let mut compile_fail = 0;
    let mut eval_pass = 0;
    let mut eval_fail = 0;

    for (name, test_fn) in &tests {
        println!("  Testing {name}...");
        let (compiled, correct) = test_fn();
        if compiled {
            compile_pass += 1;
            if correct {
                eval_pass += 1;
            } else {
                eval_fail += 1;
            }
        } else {
            compile_fail += 1;
        }
    }

    let total = compile_pass + compile_fail;
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Compiled: {compile_pass}/{total}");
    println!("  Eval correct: {eval_pass}/{compile_pass}");
    if compile_fail > 0 {
        println!("  Compile failures: {compile_fail}");
    }
    if eval_fail > 0 {
        println!("  Eval failures (wrong results): {eval_fail}");
    }
    println!();

    if eval_fail > 0 || compile_fail > 0 {
        std::process::exit(1);
    }
}
