//! End-to-end inference tests for CoreML and ANE runtimes.
//!
//! Exercises the full pipeline: ONNX → MIL IR → optimize → compile → load → predict.

use std::path::PathBuf;

use ironmill_compile::coreml::compiler::compile_model;
#[allow(deprecated)]
use mil_rs::ir::passes::{
    ConstantFoldPass, DeadCodeEliminationPass, IdentityEliminationPass,
};
use mil_rs::ir::Pass;
use mil_rs::{onnx_to_program, program_to_model, read_onnx, write_mlpackage};

use ironmill_inference::coreml_runtime::{ComputeUnits, Model, build_dummy_input};

const SPEC_VERSION: i32 = 7;

fn fixture_path(name: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
        .join("../../tests/fixtures")
        .join(name)
}

fn compile_onnx_to_mlmodelc(onnx_name: &str) -> (tempfile::TempDir, PathBuf) {
    let onnx_path = fixture_path(onnx_name);
    assert!(onnx_path.exists(), "fixture {onnx_name} not found");

    let mut model_proto = read_onnx(&onnx_path).expect("read_onnx failed");
    let result = onnx_to_program(&mut model_proto).expect("onnx_to_program failed");
    let mut program = result.program;

    ConstantFoldPass.run(&mut program).unwrap();
    DeadCodeEliminationPass.run(&mut program).unwrap();
    IdentityEliminationPass.run(&mut program).unwrap();

    let model = program_to_model(&program, SPEC_VERSION).expect("program_to_model failed");

    let dir = tempfile::tempdir().expect("tempdir");
    let pkg_path = dir.path().join(format!(
        "{}.mlpackage",
        onnx_path.file_stem().unwrap().to_str().unwrap()
    ));
    write_mlpackage(&model, &pkg_path).expect("write_mlpackage failed");

    let compiled = compile_model(&pkg_path, dir.path()).expect("compile_model failed");
    assert!(compiled.is_dir(), "compiled .mlmodelc should exist");

    (dir, compiled)
}

#[test]
fn coreml_inference_mnist() {
    let (_dir, mlmodelc) = compile_onnx_to_mlmodelc("mnist.onnx");

    let model = Model::load(&mlmodelc, ComputeUnits::CpuOnly).expect("Model::load failed");
    let desc = model.input_description().expect("input_description failed");
    assert!(!desc.features.is_empty(), "model should have inputs");

    let input = build_dummy_input(&desc).expect("build_dummy_input failed");
    let output = model.predict(&input).expect("predict failed");

    // Verify we got an output feature provider back.
    let _ = output.as_feature_provider();
}

#[test]
fn coreml_inference_squeezenet() {
    let (_dir, mlmodelc) = compile_onnx_to_mlmodelc("squeezenet1.1.onnx");

    let model = Model::load(&mlmodelc, ComputeUnits::All).expect("Model::load failed");
    let desc = model.input_description().expect("input_description failed");
    assert!(!desc.features.is_empty(), "model should have inputs");

    let input = build_dummy_input(&desc).expect("build_dummy_input failed");
    let output = model.predict(&input).expect("predict failed");

    let _ = output.as_feature_provider();
}

#[test]
fn coreml_inference_on_neural_engine() {
    let (_dir, mlmodelc) = compile_onnx_to_mlmodelc("mnist.onnx");

    // Request Neural Engine — CoreML will fall back to CPU/GPU if ANE unavailable.
    let model =
        Model::load(&mlmodelc, ComputeUnits::CpuAndNeuralEngine).expect("Model::load failed");
    let desc = model.input_description().expect("input_description failed");
    let input = build_dummy_input(&desc).expect("build_dummy_input failed");
    let output = model
        .predict(&input)
        .expect("predict on ANE compute units failed");

    let _ = output.as_feature_provider();
}
