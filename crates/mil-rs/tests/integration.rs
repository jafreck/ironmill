//! Integration tests for mil-rs with real model files.

use std::path::{Path, PathBuf};

use mil_rs::reader::print_model_summary;
use mil_rs::{model_to_program, read_mlmodel, read_mlpackage, write_mlmodel, write_mlpackage};

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
