//! End-to-end integration tests for the GPU compilation pipeline.
//!
//! Exercises the full path: SafeTensors → template emission (External refs)
//! → PassPipeline (with spill-after-quantize) → MilWeightProvider extraction
//! → GPU bundle write.

mod common;

use std::sync::Arc;

use common::*;
use ironmill_compile::gpu::GpuCompileBuilder;
use ironmill_compile::gpu::bundle::write_gpu_bundle;
use mil_rs::ir::{Pass, PassPipeline};
use ironmill_compile::templates::weights_to_program;
use ironmill_compile::weights::safetensors::SafeTensorsProvider;
use ironmill_compile::weights::{MilWeightProvider, WeightProvider};
use mil_rs::ir::passes::Fp16QuantizePass;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_model_dir(dir: &std::path::Path) {
    write_safetensors_model_dir(dir, &config_json("llama"), &build_llama_base_tensors());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full pipeline with FP16 quantization — no weight compression, just dtype cast.
#[test]
fn gpu_compile_fp16_e2e() {
    let model_dir = tempfile::tempdir().unwrap();
    write_model_dir(model_dir.path());

    let pipeline = PassPipeline::new().with_fp16().expect("fp16 pipeline");

    let provider = GpuCompileBuilder::new(model_dir.path())
        .with_pass_pipeline(pipeline)
        .build()
        .expect("GPU compile should succeed");

    let names = provider.tensor_names();
    assert!(!names.is_empty(), "provider should have tensors");
    assert!(provider.has_tensor("model.embed_tokens.weight"));

    let bundle_dir = tempfile::tempdir().unwrap();
    write_gpu_bundle(&provider, bundle_dir.path()).expect("bundle write should succeed");

    // Verify manifest
    let manifest_path = bundle_dir.path().join("manifest.json");
    assert!(manifest_path.exists(), "manifest.json should exist");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap())
            .expect("manifest should be valid JSON");
    assert!(
        manifest["tensors"].is_object(),
        "manifest should have tensors"
    );

    // Verify weights directory
    let weights_dir = bundle_dir.path().join("weights");
    assert!(weights_dir.exists(), "weights/ directory should exist");
    let weight_files: Vec<_> = std::fs::read_dir(&weights_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert!(
        !weight_files.is_empty(),
        "weights/ should contain tensor files"
    );
}

/// Full pipeline with INT4 affine quantization (group_size = HIDDEN).
#[test]
fn gpu_compile_int4_e2e() {
    let model_dir = tempfile::tempdir().unwrap();
    write_model_dir(model_dir.path());

    let pipeline = PassPipeline::new()
        .with_int4(HIDDEN)
        .expect("int4 pipeline");

    let provider = GpuCompileBuilder::new(model_dir.path())
        .with_pass_pipeline(pipeline)
        .build()
        .expect("INT4 GPU compile should succeed");

    let names = provider.tensor_names();
    assert!(!names.is_empty());

    let bundle_dir = tempfile::tempdir().unwrap();
    write_gpu_bundle(&provider, bundle_dir.path()).expect("INT4 bundle write should succeed");

    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(bundle_dir.path().join("manifest.json")).unwrap(),
    )
    .expect("manifest should be valid JSON");

    let tensors = manifest["tensors"]
        .as_object()
        .expect("tensors should be object");

    // At least some tensors should have been affine-quantized.
    let has_quantized = tensors
        .values()
        .any(|t| t["format"].as_str() == Some("affine_dequantize"));
    assert!(
        has_quantized,
        "INT4 bundle should contain affine_dequantize tensors"
    );
}

/// Empty pipeline — unquantized passthrough with raw FP16 weights.
#[test]
fn gpu_compile_unquantized_e2e() {
    let model_dir = tempfile::tempdir().unwrap();
    write_model_dir(model_dir.path());

    let provider = GpuCompileBuilder::new(model_dir.path())
        .build()
        .expect("unquantized GPU compile should succeed");

    let names = provider.tensor_names();
    assert!(!names.is_empty());

    let bundle_dir = tempfile::tempdir().unwrap();
    write_gpu_bundle(&provider, bundle_dir.path())
        .expect("unquantized bundle write should succeed");

    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(bundle_dir.path().join("manifest.json")).unwrap(),
    )
    .unwrap();

    let tensors = manifest["tensors"].as_object().unwrap();
    for (name, info) in tensors {
        assert_eq!(
            info["format"].as_str(),
            Some("dense"),
            "tensor {name} should be dense in unquantized bundle"
        );
    }
}

/// Verify the spill-after-quantize path (GpuCompileBuilder) matches the
/// eager-materialization path for FP16 — data must not be corrupted.
#[test]
fn gpu_compile_spill_produces_correct_output() {
    let model_dir = tempfile::tempdir().unwrap();
    write_model_dir(model_dir.path());

    // Path 1: Through GpuCompileBuilder (spill-after-quantize active)
    let pipeline = PassPipeline::new().with_fp16().unwrap();
    let provider_spill = GpuCompileBuilder::new(model_dir.path())
        .with_pass_pipeline(pipeline)
        .build()
        .expect("spill path should succeed");

    // Path 2: Manual eager materialization — no spilling
    let provider_arc: Arc<dyn WeightProvider + Send + Sync> = Arc::from(Box::new(
        SafeTensorsProvider::load(model_dir.path()).unwrap(),
    )
        as Box<dyn WeightProvider + Send + Sync>);
    let config = provider_arc.config().clone();
    let result = weights_to_program(provider_arc.as_ref()).unwrap();
    let mut program = result.program;
    program.set_weight_provider(provider_arc.clone());
    program.materialize_all().unwrap();
    Fp16QuantizePass.run(&mut program).unwrap();
    let provider_eager = MilWeightProvider::new(&mut program, config).unwrap();

    // Compare tensor name sets
    let mut names_spill: Vec<String> = provider_spill
        .tensor_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut names_eager: Vec<String> = provider_eager
        .tensor_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    names_spill.sort();
    names_eager.sort();
    assert_eq!(names_spill, names_eager, "tensor name lists should match");

    // Compare tensor data for each weight
    for name in &names_spill {
        let t_spill = provider_spill.tensor(name).unwrap();
        let t_eager = provider_eager.tensor(name).unwrap();
        assert_eq!(t_spill.dtype, t_eager.dtype, "dtype mismatch for {name}");
        assert_eq!(t_spill.shape, t_eager.shape, "shape mismatch for {name}");
        assert_eq!(
            t_spill.data.as_ref(),
            t_eager.data.as_ref(),
            "data mismatch for tensor {name}"
        );
    }
}

/// Verify that the spill-after-quantize mechanism actually creates External
/// refs mid-pipeline and that the round-trip preserves data integrity.
#[test]
fn spill_creates_external_refs_and_round_trips() {
    use mil_rs::TensorData;
    use mil_rs::Value;

    let model_dir = tempfile::tempdir().unwrap();
    write_model_dir(model_dir.path());

    let provider_arc: Arc<dyn WeightProvider + Send + Sync> = Arc::from(Box::new(
        SafeTensorsProvider::load(model_dir.path()).unwrap(),
    )
        as Box<dyn WeightProvider + Send + Sync>);
    let config = provider_arc.config().clone();
    let result = weights_to_program(provider_arc.as_ref()).unwrap();
    let mut program = result.program;
    program.set_weight_provider(provider_arc.clone());

    // Before spilling: large tensors should be External (from template emission),
    // small tensors should be Inline.
    let has_external_before = program.main().unwrap().body.operations.iter().any(|op| {
        op.attributes.values().chain(op.inputs.values()).any(|v| {
            matches!(
                v,
                Value::Tensor {
                    data: TensorData::External { .. },
                    ..
                }
            )
        })
    });
    assert!(
        has_external_before,
        "template emission should produce External refs for large tensors"
    );

    // Materialize all, then run FP16 pass to produce new inline data.
    program.materialize_all().unwrap();
    Fp16QuantizePass.run(&mut program).unwrap();

    // After FP16 pass, quantized tensors are inline. Spill them.
    program.spill_inline_tensors(4096).unwrap();

    // After spilling: large tensors should be External again (spilled to disk).
    let has_external_after_spill = program.main().unwrap().body.operations.iter().any(|op| {
        op.attributes.values().chain(op.inputs.values()).any(|v| {
            matches!(
                v,
                Value::Tensor {
                    data: TensorData::External { .. },
                    ..
                }
            )
        })
    });
    assert!(
        has_external_after_spill,
        "spill_inline_tensors should produce External refs for large tensors"
    );

    // Materialize again and verify data is readable.
    program.materialize_all().unwrap();
    let all_inline = program.main().unwrap().body.operations.iter().all(|op| {
        op.attributes
            .values()
            .chain(op.inputs.values())
            .all(|v| match v {
                Value::Tensor { data, .. } => data.is_inline(),
                _ => true,
            })
    });
    assert!(
        all_inline,
        "all tensors should be inline after materialize_all"
    );

    // Final extraction should succeed and produce valid tensors.
    let provider = MilWeightProvider::new(&mut program, config).unwrap();
    assert!(!provider.tensor_names().is_empty());
}
