//! Integration tests for mil-rs with real model files.

use std::path::{Path, PathBuf};

use mil_rs::ir::{Block, Function, LayoutOptimizationPass, Pass, PassPipeline};
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

/// Build a program with a single conv op so the layout pass inserts transposes,
/// serialize to protobuf, and verify the transpose ops have correct structure.
#[test]
fn transpose_ops_serialize_correctly() {
    let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);

    let conv = Operation::new("conv", "conv_0")
        .with_input("x", Value::Reference("input".into()))
        .with_output("conv_out");

    let mut block = Block::new();
    block.add_op(conv);
    block.outputs.push("conv_out".into());

    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);
    program.functions.get_mut("main").unwrap().body = block;

    // Run layout pass to insert transposes.
    LayoutOptimizationPass.run(&mut program).unwrap();

    let ops = &program.functions["main"].body.operations;
    let transpose_count = ops.iter().filter(|op| op.op_type == "transpose").count();
    assert!(
        transpose_count >= 2,
        "layout pass should insert at least 2 transposes, got {transpose_count}"
    );

    // Serialize to protobuf model — this must not panic.
    let model = program_to_model(&program, 8).expect("program_to_model should succeed");

    // Verify the model round-trips through protobuf encoding.
    use prost::Message;
    let bytes = model.encode_to_vec();
    let decoded = mil_rs::proto::specification::Model::decode(bytes.as_slice())
        .expect("protobuf should decode");
    assert_eq!(decoded.specification_version, 8);

    // Verify the serialized program has transpose ops with correct structure.
    let proto_program = match &decoded.r#type {
        Some(mil_rs::proto::specification::model::Type::MlProgram(p)) => p,
        _ => panic!("expected MlProgram model type"),
    };

    let proto_func = proto_program
        .functions
        .values()
        .next()
        .expect("should have a function");
    let proto_block = proto_func
        .block_specializations
        .values()
        .next()
        .expect("should have a block");

    // Find transpose operations in the serialized proto.
    let proto_transposes: Vec<_> = proto_block
        .operations
        .iter()
        .filter(|op| op.r#type == "transpose")
        .collect();
    assert!(
        proto_transposes.len() >= 2,
        "serialized model should have at least 2 transpose ops"
    );

    for proto_op in &proto_transposes {
        // Each transpose must have "x" and "perm" inputs.
        assert!(
            proto_op.inputs.contains_key("x"),
            "transpose op should have 'x' input"
        );
        assert!(
            proto_op.inputs.contains_key("perm"),
            "transpose op should have 'perm' input"
        );

        // Each transpose must have exactly one output with a type.
        assert_eq!(proto_op.outputs.len(), 1, "transpose should have 1 output");
        let out = &proto_op.outputs[0];
        assert!(
            out.r#type.is_some(),
            "transpose output '{}' should have a type",
            out.name
        );

        // The output type should be a tensor with rank 4 (same as input).
        let vt = out.r#type.as_ref().unwrap();
        if let Some(mil_rs::proto::mil_spec::value_type::Type::TensorType(tt)) = &vt.r#type {
            assert_eq!(tt.rank, 4, "transpose output should have rank 4");
            // Dimensions should be marked unknown (since transpose permutes them).
            for dim in &tt.dimensions {
                assert!(
                    matches!(
                        dim.dimension,
                        Some(mil_rs::proto::mil_spec::dimension::Dimension::Unknown(_))
                    ),
                    "transpose output dimensions should be unknown"
                );
            }
        } else {
            panic!("transpose output type should be TensorType");
        }
    }
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
