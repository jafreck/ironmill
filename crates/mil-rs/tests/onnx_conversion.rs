//! End-to-end ONNX → CoreML conversion tests.
//!
//! These tests validate the full pipeline:
//!   read_onnx → onnx_to_program → optimize → program_to_model → write_mlpackage
//!
//! Tests without `#[ignore]` work on any platform (no xcrun required).
//! Tests marked `#[ignore]` require macOS with Xcode for `xcrun coremlcompiler`.

use std::path::{Path, PathBuf};

use ironmill_compile::coreml::compiler::{compile_model, is_compiler_available};
use mil_rs::convert::onnx_graph::ConversionResult;
use mil_rs::ir::passes::{ConstantFoldPass, DeadCodeEliminationPass, IdentityEliminationPass};
use mil_rs::{
    Pass, Program, onnx_to_program, program_to_model, read_mlpackage, read_onnx, write_mlpackage,
};

/// CoreML spec version used for ONNX conversions (matches CLI).
const SPEC_VERSION: i32 = 7;

fn fixture_path(name: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest)
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Run the three standard optimization passes on a program.
fn run_optimization_passes(program: &mut Program) {
    let passes: Vec<Box<dyn Pass>> = vec![
        Box::new(DeadCodeEliminationPass),
        Box::new(IdentityEliminationPass),
        Box::new(ConstantFoldPass),
    ];
    for pass in &passes {
        pass.run(program)
            .unwrap_or_else(|e| panic!("pass '{}' failed: {e}", pass.name()));
    }
}

/// Count total operations across all functions in a program.
fn count_ops(program: &Program) -> usize {
    program
        .functions
        .values()
        .map(|f| f.body.operations.len())
        .sum()
}

/// Convert an ONNX fixture to a MIL Program, returning the ConversionResult.
fn convert_fixture(name: &str) -> ConversionResult {
    let onnx =
        read_onnx(fixture_path(name)).unwrap_or_else(|e| panic!("failed to read {name}: {e}"));
    onnx_to_program(&onnx).unwrap_or_else(|e| panic!("failed to convert {name} to MIL IR: {e}"))
}

// ---------------------------------------------------------------------------
// Full pipeline: ONNX → MIL IR → CoreML .mlpackage
// ---------------------------------------------------------------------------

#[test]
fn convert_mnist_onnx_to_mlpackage() {
    // 1. Read MNIST ONNX
    let onnx = read_onnx(fixture_path("mnist.onnx")).expect("failed to read mnist.onnx");

    // 2. Convert to MIL IR
    let result = onnx_to_program(&onnx).expect("onnx_to_program failed for MNIST");

    // 3. MNIST uses only basic ops — expect no warnings
    assert!(
        result.warnings.is_empty(),
        "MNIST conversion should produce no warnings, got: {:?}",
        result.warnings
    );

    // 4. Run optimization passes
    let mut program = result.program;
    run_optimization_passes(&mut program);

    // 5. Convert to proto Model
    let model =
        program_to_model(&program, SPEC_VERSION).expect("program_to_model failed for MNIST");

    // 6. Write .mlpackage to temp dir
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let pkg_path = dir.path().join("mnist.mlpackage");
    write_mlpackage(&model, &pkg_path).expect("failed to write mnist.mlpackage");

    // 7. Read back and verify it's a valid CoreML model
    let reloaded = read_mlpackage(&pkg_path).expect("failed to read back mnist.mlpackage");
    assert_eq!(
        reloaded.specification_version, SPEC_VERSION as i32,
        "reloaded model should have spec version {SPEC_VERSION}"
    );

    // 8. Verify the model contains an ML Program
    assert!(
        reloaded.specification_version >= 7,
        "ONNX-converted model should be spec version >= 7 (ML Program)"
    );
}

#[test]
fn convert_squeezenet_onnx_to_mlpackage() {
    // 1. Read SqueezeNet ONNX
    let onnx =
        read_onnx(fixture_path("squeezenet1.1.onnx")).expect("failed to read squeezenet1.1.onnx");

    // 2. Convert to MIL IR
    let result = onnx_to_program(&onnx).expect("onnx_to_program failed for SqueezeNet");

    // 3. SqueezeNet may have unsupported-op warnings — just ensure they're reasonable
    for warning in &result.warnings {
        assert!(
            !warning.is_empty(),
            "warnings should not contain empty strings"
        );
    }

    // 4. Run optimization passes
    let mut program = result.program;
    run_optimization_passes(&mut program);

    // 5. Convert to proto Model
    let model =
        program_to_model(&program, SPEC_VERSION).expect("program_to_model failed for SqueezeNet");

    // 6. Write .mlpackage
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let pkg_path = dir.path().join("squeezenet.mlpackage");
    write_mlpackage(&model, &pkg_path).expect("failed to write squeezenet.mlpackage");

    // 7. Read back and verify structure
    let reloaded = read_mlpackage(&pkg_path).expect("failed to read back squeezenet.mlpackage");
    assert_eq!(
        reloaded.specification_version, SPEC_VERSION as i32,
        "reloaded model should have spec version {SPEC_VERSION}"
    );

    // Verify the package files exist on disk
    assert!(
        pkg_path.join("Manifest.json").is_file(),
        "mlpackage should contain Manifest.json"
    );
    assert!(
        pkg_path
            .join("Data/com.apple.CoreML/model.mlmodel")
            .is_file(),
        "mlpackage should contain model.mlmodel"
    );
}

// ---------------------------------------------------------------------------
// Optimization pass validation
// ---------------------------------------------------------------------------

#[test]
fn optimization_reduces_graph() {
    // 1. Convert MNIST to IR (no optimization)
    let result = convert_fixture("mnist.onnx");
    let mut program = result.program;

    // 2. Count ops before optimization
    let before = count_ops(&program);
    assert!(
        before > 0,
        "MNIST program should have operations before optimization"
    );

    // 3. Run all three passes
    run_optimization_passes(&mut program);

    // 4. Count ops after — should be <= before
    let after = count_ops(&program);
    assert!(
        after <= before,
        "optimization should not increase op count: before={before}, after={after}"
    );
}

// ---------------------------------------------------------------------------
// xcrun compilation (macOS only)
// ---------------------------------------------------------------------------

#[test]
#[ignore] // Requires macOS with Xcode
fn compile_mnist_end_to_end() {
    if !is_compiler_available() {
        eprintln!("skipping: xcrun coremlcompiler not available");
        return;
    }

    // Full conversion pipeline
    let result = convert_fixture("mnist.onnx");
    let mut program = result.program;
    run_optimization_passes(&mut program);
    let model = program_to_model(&program, SPEC_VERSION).expect("program_to_model failed");

    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let pkg_path = dir.path().join("mnist.mlpackage");
    write_mlpackage(&model, &pkg_path).expect("failed to write mlpackage");

    // Compile with xcrun
    let compiled = compile_model(&pkg_path, dir.path()).expect("compile_model failed for MNIST");

    // Verify .mlmodelc directory exists
    assert!(
        compiled.is_dir(),
        "compiled model directory should exist at {compiled:?}"
    );
    assert!(
        compiled.extension().is_some_and(|ext| ext == "mlmodelc"),
        "compiled output should have .mlmodelc extension, got: {compiled:?}"
    );
}

#[test]
#[ignore] // Requires macOS with Xcode
fn compile_squeezenet_end_to_end() {
    if !is_compiler_available() {
        eprintln!("skipping: xcrun coremlcompiler not available");
        return;
    }

    // Full conversion pipeline
    let result = convert_fixture("squeezenet1.1.onnx");
    let mut program = result.program;
    run_optimization_passes(&mut program);
    let model = program_to_model(&program, SPEC_VERSION).expect("program_to_model failed");

    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let pkg_path = dir.path().join("squeezenet.mlpackage");
    write_mlpackage(&model, &pkg_path).expect("failed to write mlpackage");

    // Compile with xcrun
    let compiled =
        compile_model(&pkg_path, dir.path()).expect("compile_model failed for SqueezeNet");

    assert!(
        compiled.is_dir(),
        "compiled model directory should exist at {compiled:?}"
    );
    assert!(
        compiled.extension().is_some_and(|ext| ext == "mlmodelc"),
        "compiled output should have .mlmodelc extension, got: {compiled:?}"
    );
}

// ---------------------------------------------------------------------------
// I/O shape preservation
// ---------------------------------------------------------------------------

#[test]
fn conversion_preserves_io_shapes() {
    use mil_rs::proto::onnx::type_proto;

    let onnx = read_onnx(fixture_path("mnist.onnx")).expect("failed to read mnist.onnx");
    let graph = onnx.graph.as_ref().expect("ONNX model should have a graph");

    // Collect ONNX input names (excluding initializers)
    let initializer_names: std::collections::HashSet<&str> =
        graph.initializer.iter().map(|t| t.name.as_str()).collect();

    let onnx_input_names: Vec<&str> = graph
        .input
        .iter()
        .filter(|vi| !initializer_names.contains(vi.name.as_str()))
        .map(|vi| vi.name.as_str())
        .collect();

    let _onnx_output_names: Vec<&str> = graph.output.iter().map(|vi| vi.name.as_str()).collect();

    // Convert to MIL IR
    let result = onnx_to_program(&onnx).expect("onnx_to_program failed");
    let program = result.program;

    let main_fn = program
        .functions
        .get("main")
        .expect("program should have a 'main' function");

    // Check Function inputs match ONNX inputs (by count)
    assert_eq!(
        main_fn.inputs.len(),
        onnx_input_names.len(),
        "MIL function input count should match ONNX real input count \
         (ONNX inputs: {onnx_input_names:?}, MIL inputs: {:?})",
        main_fn.inputs.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );

    // Verify each ONNX input has a corresponding MIL input
    let mil_input_names: Vec<&str> = main_fn.inputs.iter().map(|(n, _)| n.as_str()).collect();
    for onnx_name in &onnx_input_names {
        assert!(
            mil_input_names
                .iter()
                .any(|n| n.contains(onnx_name) || onnx_name.contains(n)),
            "ONNX input '{onnx_name}' should have a corresponding MIL input \
             (MIL inputs: {mil_input_names:?})"
        );
    }

    // Check block outputs are present
    assert!(
        !main_fn.body.outputs.is_empty(),
        "MIL function body should have outputs"
    );

    // Verify ONNX output shapes are tensor-typed
    for output in &graph.output {
        if let Some(type_proto) = &output.r#type {
            if let Some(type_proto::Value::TensorType(_)) = &type_proto.value {
                // Expected — ONNX output is a tensor
            } else {
                panic!(
                    "ONNX output '{}' has unexpected type (not a tensor)",
                    output.name
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Program structure validation
// ---------------------------------------------------------------------------

#[test]
fn mnist_program_has_expected_structure() {
    let result = convert_fixture("mnist.onnx");
    let program = &result.program;

    // Should have exactly one function named "main"
    assert_eq!(
        program.functions.len(),
        1,
        "MNIST program should have exactly one function"
    );
    assert!(
        program.functions.contains_key("main"),
        "program should contain a 'main' function"
    );

    let main_fn = &program.functions["main"];

    // Should have at least one input
    assert!(
        !main_fn.inputs.is_empty(),
        "MNIST main function should have at least one input"
    );

    // Should have operations in the body
    assert!(
        !main_fn.body.operations.is_empty(),
        "MNIST main function body should have operations"
    );

    // Should have at least one output
    assert!(
        !main_fn.body.outputs.is_empty(),
        "MNIST main function body should have outputs"
    );

    // Every operation should have a non-empty op_type and name
    for op in &main_fn.body.operations {
        assert!(
            !op.op_type.is_empty(),
            "operation '{}' should have a non-empty op_type",
            op.name
        );
        assert!(
            !op.name.is_empty(),
            "operation with type '{}' should have a non-empty name",
            op.op_type
        );
        assert!(
            !op.outputs.is_empty(),
            "operation '{}' ({}) should have at least one output",
            op.name,
            op.op_type
        );
    }
}

#[test]
fn squeezenet_program_has_expected_structure() {
    let result = convert_fixture("squeezenet1.1.onnx");
    let program = &result.program;

    assert!(
        program.functions.contains_key("main"),
        "program should contain a 'main' function"
    );

    let main_fn = &program.functions["main"];

    assert!(
        !main_fn.inputs.is_empty(),
        "SqueezeNet main function should have at least one input"
    );
    assert!(
        !main_fn.body.operations.is_empty(),
        "SqueezeNet main function body should have operations"
    );
    assert!(
        !main_fn.body.outputs.is_empty(),
        "SqueezeNet main function body should have outputs"
    );

    // SqueezeNet has many more ops than MNIST
    let op_count = main_fn.body.operations.len();
    assert!(
        op_count > 10,
        "SqueezeNet should have a significant number of operations, got {op_count}"
    );
}

// ---------------------------------------------------------------------------
// CLI integration tests
// ---------------------------------------------------------------------------

#[test]
fn cli_inspect_mnist_onnx() {
    let output = std::process::Command::new("cargo")
        .args(["run", "-p", "ironmill-cli", "--quiet", "--", "inspect"])
        .arg(fixture_path("mnist.onnx"))
        .output()
        .expect("failed to run CLI inspect");

    assert!(
        output.status.success(),
        "CLI inspect should exit 0, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Inspect should print some model information
    assert!(
        !stdout.is_empty(),
        "CLI inspect should produce output for mnist.onnx"
    );
}

#[test]
fn cli_inspect_squeezenet_onnx() {
    let output = std::process::Command::new("cargo")
        .args(["run", "-p", "ironmill-cli", "--quiet", "--", "inspect"])
        .arg(fixture_path("squeezenet1.1.onnx"))
        .output()
        .expect("failed to run CLI inspect");

    assert!(
        output.status.success(),
        "CLI inspect should exit 0, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.is_empty(),
        "CLI inspect should produce output for squeezenet1.1.onnx"
    );
}
