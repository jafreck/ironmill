//! Integration tests for mil-rs with real model files.

use std::path::{Path, PathBuf};

use mil_rs::ir::{Block, Function, Pass, PassPipeline};
use mil_rs::reader::print_model_summary;
use mil_rs::{
    Operation, Program, ScalarType, TensorType, Value, model_to_program, program_to_model,
    read_mlmodel, read_mlpackage, write_mlmodel, write_mlpackage,
};

fn fixture_path(name: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest)
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn read_and_inspect_mobilenet() {
    let model =
        read_mlmodel(fixture_path("MobileNet.mlmodel")).expect("failed to read MobileNet.mlmodel");

    assert!(model.specification_version > 0);
    let desc = model
        .description
        .as_ref()
        .expect("model should have a description");
    assert!(!desc.input.is_empty(), "model should have inputs");
    assert!(!desc.output.is_empty(), "model should have outputs");

    // Smoke-test: print_model_summary should not panic.
    print_model_summary(&model);
}

#[test]
fn round_trip_mobilenet_mlmodel() {
    let original =
        read_mlmodel(fixture_path("MobileNet.mlmodel")).expect("failed to read MobileNet.mlmodel");

    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let out_path = dir.path().join("roundtrip.mlmodel");

    write_mlmodel(&original, &out_path).expect("failed to write mlmodel");

    let reloaded = read_mlmodel(&out_path).expect("failed to read back written mlmodel");

    assert_eq!(
        original.specification_version,
        reloaded.specification_version
    );

    let orig_desc = original.description.as_ref().unwrap();
    let new_desc = reloaded.description.as_ref().unwrap();
    assert_eq!(orig_desc.input.len(), new_desc.input.len());
    assert_eq!(orig_desc.output.len(), new_desc.output.len());

    // Byte-level equality: protobuf encoding is deterministic.
    use prost::Message;
    assert_eq!(original.encode_to_vec(), reloaded.encode_to_vec());
}

#[test]
fn round_trip_mobilenet_mlpackage() {
    let original =
        read_mlmodel(fixture_path("MobileNet.mlmodel")).expect("failed to read MobileNet.mlmodel");

    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let pkg_path = dir.path().join("roundtrip.mlpackage");

    write_mlpackage(&original, &pkg_path).expect("failed to write mlpackage");

    assert!(pkg_path.join("Manifest.json").is_file());
    assert!(
        pkg_path
            .join("Data/com.apple.CoreML/model.mlmodel")
            .is_file()
    );

    let reloaded = read_mlpackage(&pkg_path).expect("failed to read back written mlpackage");

    assert_eq!(
        original.specification_version,
        reloaded.specification_version
    );

    let orig_desc = original.description.as_ref().unwrap();
    let new_desc = reloaded.description.as_ref().unwrap();
    assert_eq!(orig_desc.input.len(), new_desc.input.len());
    assert_eq!(orig_desc.output.len(), new_desc.output.len());

    use prost::Message;
    assert_eq!(original.encode_to_vec(), reloaded.encode_to_vec());
}

#[test]
fn read_simple_mlpackage_manifest() {
    // The simple.mlpackage fixture has a valid Manifest.json but no actual
    // protobuf model file, so read_mlpackage should return an error.
    let pkg = fixture_path("simple.mlpackage");
    let result = read_mlpackage(&pkg);
    let err = result.expect_err("expected error for missing model spec");
    assert!(
        matches!(err, mil_rs::MilError::InvalidPackage(_)),
        "expected InvalidPackage, got: {err}"
    );
    assert!(
        err.to_string().contains("model spec not found"),
        "error should mention missing model spec: {err}"
    );
}

#[test]
fn mobilenet_is_not_ml_program() {
    // MobileNet is a NeuralNetwork model, not an ML Program.
    // model_to_program should return an UnsupportedOp error.
    let model =
        read_mlmodel(fixture_path("MobileNet.mlmodel")).expect("failed to read MobileNet.mlmodel");

    let result = model_to_program(&model);
    let err = result.expect_err("expected error for non-ML-Program model");
    assert!(
        matches!(err, mil_rs::MilError::UnsupportedOp(_)),
        "expected UnsupportedOp, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Transpose serialization tests
// ---------------------------------------------------------------------------

/// Build a program with explicit transpose ops and verify they serialize
/// correctly to protobuf with proper structure and concrete dimensions.
#[test]
fn transpose_ops_serialize_correctly() {
    let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);

    // Create a program with explicit transpose ops (NCHW→NHWC and back).
    let perm_nhwc: Vec<u8> = [0i32, 2, 3, 1]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let perm_nchw: Vec<u8> = [0i32, 3, 1, 2]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let t1 = Operation::new("transpose", "t_nhwc")
        .with_input("x", Value::Reference("input".into()))
        .with_input(
            "perm",
            Value::Tensor {
                data: perm_nhwc,
                shape: vec![4],
                dtype: ScalarType::Int32,
            },
        )
        .with_output("nhwc_out");

    let t2 = Operation::new("transpose", "t_nchw")
        .with_input("x", Value::Reference("nhwc_out".into()))
        .with_input(
            "perm",
            Value::Tensor {
                data: perm_nchw,
                shape: vec![4],
                dtype: ScalarType::Int32,
            },
        )
        .with_output("nchw_out");

    let mut block = Block::new();
    block.add_op(t1);
    block.add_op(t2);
    block.outputs.push("nchw_out".into());

    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);
    program.functions.get_mut("main").unwrap().body = block;

    // Run the pipeline (layout pass will cancel the inverse pair, but
    // TypeRepropagationPass ensures any surviving transposes have types).
    let pipeline = PassPipeline::new();
    pipeline.run(&mut program).unwrap();

    // The inverse pair should be cancelled — verify no transposes remain.
    let ops = &program.functions["main"].body.operations;
    let transpose_count = ops.iter().filter(|op| op.op_type == "transpose").count();
    assert_eq!(
        transpose_count, 0,
        "inverse transpose pair should be cancelled"
    );

    // Serialize — must not panic.
    let model = program_to_model(&program, 8).expect("program_to_model should succeed");

    use prost::Message;
    let bytes = model.encode_to_vec();
    let decoded = mil_rs::proto::specification::Model::decode(bytes.as_slice())
        .expect("protobuf should decode");
    assert_eq!(decoded.specification_version, 8);
}

/// Verify that the layout pass + serialization produces a valid mlpackage.
#[test]
fn layout_pass_model_writes_valid_mlpackage() {
    let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);

    let conv = Operation::new("conv", "conv_0")
        .with_input("x", Value::Reference("input".into()))
        .with_output("conv_out");
    let relu = Operation::new("relu", "relu_0")
        .with_input("x", Value::Reference("conv_out".into()))
        .with_output("relu_out");

    let mut block = Block::new();
    block.add_op(conv);
    block.add_op(relu);
    block.outputs.push("relu_out".into());

    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);
    program.functions.get_mut("main").unwrap().body = block;

    // Run the full default pipeline (which now includes layout optimization).
    let pipeline = PassPipeline::new();
    pipeline.run(&mut program).expect("pipeline should succeed");

    let model = program_to_model(&program, 8).expect("program_to_model should succeed");

    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let pkg_path = dir.path().join("layout_test.mlpackage");

    write_mlpackage(&model, &pkg_path).expect("write_mlpackage should succeed");

    // Read it back and verify structure.
    let reloaded = read_mlpackage(&pkg_path).expect("should read back the written mlpackage");
    assert_eq!(reloaded.specification_version, 8);

    // Verify the model has valid structure (description, inputs, outputs).
    let desc = reloaded
        .description
        .as_ref()
        .expect("model should have description");
    assert!(!desc.input.is_empty(), "model should have inputs");
    assert!(!desc.output.is_empty(), "model should have outputs");
}
