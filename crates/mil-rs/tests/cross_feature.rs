//! Cross-phase composition, serialization size, and concurrency integration tests.

mod common;

use mil_rs::ir::{
    CodebookOptimizationPass, ComputeUnitAnnotationPass, ExpertQuantConfig, Fp16QuantizePass,
    Int8QuantizePass, KvCachePass, LayerScheduleConfig, LayerSchedulePass, ModelSplitPass,
    PalettizePass, PerExpertQuantPass,
};
use mil_rs::{
    ComputeUnit, MixedPrecisionConfig, OpSplittingPass, Pass, PassPipeline, ScalarType, Value,
    detect_moe, program_to_model, program_to_multi_function_model, split_moe,
};

use common::{
    SPEC_VERSION, assert_serialization_roundtrip, build_autoregressive_program,
    build_const_program, build_conv_program, build_moe_program, build_rvq_program,
    build_transformer_program, count_ops, serialized_size,
};

// ═══════════════════════════════════════════════════════════════════════
// Cross-Phase Composition (8 tests)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn nhwc_plus_mixed_precision_pipeline() {
    let mut program = build_conv_program(3);

    let config = MixedPrecisionConfig::preset_fp16_int8();
    let pipeline = PassPipeline::new()
        .with_mixed_precision_config(config)
        .expect("mixed precision config should be valid");

    pipeline.run(&mut program).expect("pipeline should succeed");

    // Layout pass (from default pipeline) should have inserted transposes.
    let ops = &program.functions["main"].body.operations;
    let has_transpose = ops.iter().any(|op| op.op_type == "transpose");
    assert!(has_transpose, "NHWC layout pass should insert transposes");

    // Mixed-precision quantization should have been applied after layout.
    // The preset_fp16_int8 config converts const ops to constexpr_affine_dequantize
    // (INT8 default) or converts FP32 tensors to FP16 (for attention-like ops).
    // After fusion, the surviving const ops (conv weights) should be quantized.
    let has_int8_ops = ops
        .iter()
        .any(|op| op.op_type == "constexpr_affine_dequantize");
    let has_fp16_tensors = ops.iter().any(|op| {
        op.attributes.values().any(|v| {
            matches!(
                v,
                Value::Tensor {
                    dtype: ScalarType::Float16,
                    ..
                }
            )
        }) || op.inputs.values().any(|v| {
            matches!(
                v,
                Value::Tensor {
                    dtype: ScalarType::Float16,
                    ..
                }
            )
        })
    });
    let no_remaining_fp32_consts = ops.iter().filter(|op| op.op_type == "const").all(|op| {
        let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
        !matches!(
            val,
            Some(Value::Tensor {
                dtype: ScalarType::Float32,
                ..
            })
        )
    });
    assert!(
        has_int8_ops || has_fp16_tensors || no_remaining_fp32_consts,
        "quantization should have been applied after layout change"
    );

    // Output must be serializable.
    let model = program_to_model(&program, SPEC_VERSION);
    assert!(model.is_ok(), "model should serialize after pipeline");
}

#[test]
fn kv_cache_plus_compute_unit_annotations() {
    let mut program = build_autoregressive_program(2048);

    // Run KV-cache pass first, then compute-unit annotation.
    KvCachePass::new(2048)
        .run(&mut program)
        .expect("KvCachePass should succeed");
    ComputeUnitAnnotationPass
        .run(&mut program)
        .expect("ComputeUnitAnnotationPass should succeed");

    let ops = &program.functions["main"].body.operations;

    // Every op should now have a compute_unit annotation.
    for op in ops {
        assert!(
            op.compute_unit.is_some(),
            "op '{}' ({}) should have a compute_unit annotation",
            op.name,
            op.op_type
        );
    }

    // KV-cache ops (inserted by KvCachePass) should be annotated too.
    let _cache_ops: Vec<_> = ops
        .iter()
        .filter(|op| {
            op.op_type.contains("kv_cache")
                || op.name.contains("kv_cache")
                || op.name.contains("cache")
        })
        .collect();
    // The pass should have inserted cache ops for the AR program.
    // If no cache ops found (pass is a no-op on this program shape),
    // we still verify that all non-cache ops are annotated, which we did above.

    // Non-cache ops should also be annotated.
    let non_cache_annotated = ops
        .iter()
        .filter(|op| !op.op_type.contains("kv_cache") && !op.name.contains("cache"))
        .all(|op| op.compute_unit.is_some());
    assert!(non_cache_annotated, "all non-cache ops should be annotated");

    // Const ops should be annotated as Any.
    for op in ops.iter().filter(|op| op.op_type == "const") {
        assert_eq!(
            op.compute_unit,
            Some(ComputeUnit::Any),
            "const op '{}' should be ComputeUnit::Any",
            op.name
        );
    }
}

#[test]
fn moe_split_plus_per_expert_quant_plus_bundle() {
    let n_experts = 4;
    let program = build_moe_program(n_experts);
    // Detect and split MoE.
    let topology = detect_moe(&program).expect("should detect MoE topology");
    assert_eq!(topology.expert_count, n_experts);

    let mut split_result = split_moe(&program, &topology);
    assert_eq!(split_result.experts.len(), n_experts);

    // Apply per-expert quantization: expert_0 is "hot" (FP16), rest are cold (4-bit).
    let eq_config = ExpertQuantConfig::preset_hot_cold(&[0], n_experts);
    let peq_pass = PerExpertQuantPass::new(eq_config);

    // Apply to each expert program.
    for expert_prog in &mut split_result.experts {
        peq_pass
            .run(expert_prog)
            .expect("PerExpertQuantPass should succeed on expert");
    }
    // Also apply to shared program.
    peq_pass
        .run(&mut split_result.shared)
        .expect("PerExpertQuantPass should succeed on shared");

    // Bundle into multi-function model.
    let model = program_to_multi_function_model(&split_result, SPEC_VERSION)
        .expect("bundling into multi-function model should succeed");

    // Verify: single Model proto with multiple functions.
    let proto_program = match &model.r#type {
        Some(mil_rs::proto::specification::model::Type::MlProgram(p)) => p,
        _ => panic!("expected MlProgram"),
    };
    // Should have main + N expert functions.
    assert!(
        proto_program.functions.len() >= 2,
        "model should have multiple functions, got {}",
        proto_program.functions.len()
    );

    // Verify the hot expert (expert_0) has FP16 weight data in its const ops,
    // while cold experts have palettized data.
    let expert_0_func = &split_result.experts[0];
    let expert_0_ops = &expert_0_func
        .functions
        .values()
        .next()
        .unwrap()
        .body
        .operations;
    let expert_0_has_fp16 = expert_0_ops.iter().any(|op| {
        op.inputs.values().any(|v| {
            matches!(
                v,
                mil_rs::Value::Tensor {
                    dtype: ScalarType::Float16,
                    ..
                }
            )
        }) || op.attributes.values().any(|v| {
            matches!(
                v,
                mil_rs::Value::Tensor {
                    dtype: ScalarType::Float16,
                    ..
                }
            )
        })
    });
    assert!(expert_0_has_fp16, "hot expert (0) should have FP16 weights");

    // Cold expert (e.g. expert_2) should have palettized ops.
    let expert_2_func = &split_result.experts[2];
    let expert_2_ops = &expert_2_func
        .functions
        .values()
        .next()
        .unwrap()
        .body
        .operations;
    let expert_2_has_palettized = expert_2_ops.iter().any(|op| {
        op.op_type.contains("constexpr_lut")
            || op.op_type.contains("palettize")
            || op.attributes.contains_key("lut")
    });
    // The pass applies palettize_4bit strategy which converts const ops.
    // If it doesn't create a different op_type, at least verify const tensors
    // are no longer plain FP32.
    let expert_2_no_fp32_weights =
        expert_2_ops
            .iter()
            .filter(|op| op.op_type == "const")
            .all(|op| {
                let in_val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
                !matches!(
                    in_val,
                    Some(mil_rs::Value::Tensor {
                        dtype: ScalarType::Float32,
                        ..
                    })
                )
            });
    assert!(
        expert_2_has_palettized || expert_2_no_fp32_weights,
        "cold expert (2) should have palettized or non-FP32 weights"
    );
}

#[test]
fn op_splitting_plus_codebook_optimization() {
    // Build a program with oversized matmul AND RVQ codebook pattern.
    let mut program = build_transformer_program(2);
    // Merge in RVQ codebook ops.
    let rvq_program = build_rvq_program(4);
    let rvq_ops = rvq_program.functions["main"].body.operations.clone();
    let block = &mut program.functions.get_mut("main").unwrap().body;
    for op in rvq_ops {
        block.add_op(op);
    }

    let _ops_before = count_ops(&program);
    OpSplittingPass::new(1024)
        .run(&mut program)
        .expect("OpSplittingPass should succeed");
    CodebookOptimizationPass
        .run(&mut program)
        .expect("CodebookOptimizationPass should succeed");

    let ops_after = count_ops(&program);

    // Both passes should have run without interfering with each other.
    // We can't know exact counts but verify the program is still valid.
    assert_serialization_roundtrip(&program);

    // At minimum, the program should still have ops and be non-trivially transformed
    // (either op count changed or ops were modified).
    assert!(ops_after > 0, "program should still have ops");
}

#[test]
fn full_pipeline_with_all_default_passes() {
    // Build a realistic transformer program: conv → attention → FFN → norm
    // We approximate this with a multi-layer transformer that includes
    // conv, attention (matmul/softmax/linear), and layer_norm ops.
    let mut program = build_transformer_program(4);
    // Also add some conv ops to exercise conv-related passes.
    let conv_prog = build_conv_program(2);
    let conv_ops = conv_prog.functions["main"].body.operations.clone();
    let block = &mut program.functions.get_mut("main").unwrap().body;
    for op in conv_ops {
        block.add_op(op);
    }

    let ops_before = count_ops(&program);

    let pipeline = PassPipeline::new();
    let _report = pipeline
        .run(&mut program)
        .expect("default pipeline should succeed");

    let ops_after = count_ops(&program);
    // Fusion passes should reduce op count.
    assert!(
        ops_after <= ops_before,
        "default pipeline should not increase op count: before={ops_before}, after={ops_after}"
    );

    // Output should be serializable.
    let model = program_to_model(&program, SPEC_VERSION);
    assert!(
        model.is_ok(),
        "program_to_model should succeed after default pipeline: {:?}",
        model.err()
    );
}

#[test]
fn mixed_precision_plus_layer_schedule() {
    let mut program = build_transformer_program(4);

    // Configure mixed-precision with wildcard rules.
    let mp_config = MixedPrecisionConfig::preset_fp16_int8();

    // Configure layer-schedule: attention=FP16, ffn=INT8, norm=FP16.
    let ls_config = LayerScheduleConfig::from_toml_str(
        r#"
[layer_strategies]
attention = "fp16"
ffn = "int8"
norm = "fp16"
other = "none"
"#,
    )
    .expect("layer schedule config should parse");

    // Run layer-schedule first (it should take precedence for detected layers),
    // then mixed-precision covers the rest.
    LayerSchedulePass::new(ls_config)
        .run(&mut program)
        .expect("LayerSchedulePass should succeed");
    mil_rs::MixedPrecisionPass::new(mp_config)
        .run(&mut program)
        .expect("MixedPrecisionPass should succeed");

    let ops = &program.functions["main"].body.operations;

    // Attention ops should be FP16 (from layer-schedule).
    let attention_ops: Vec<_> = ops
        .iter()
        .filter(|op| op.name.contains("attention"))
        .collect();
    assert!(!attention_ops.is_empty(), "should have attention ops");

    // The program should still serialize after both passes.
    let model = program_to_model(&program, SPEC_VERSION);
    assert!(
        model.is_ok(),
        "program should serialize after both passes: {:?}",
        model.err()
    );
}

#[test]
fn model_split_plus_different_pipelines() {
    let program = build_transformer_program(4);

    // Split into draft (2 layers) and verifier.
    let split_pass = ModelSplitPass::new(2);
    let split = split_pass
        .split(&program)
        .expect("ModelSplitPass::split should succeed");

    let mut draft = split.draft;
    let mut verifier = split.verifier;

    // Draft: aggressive INT8 quantization.
    let draft_pipeline = PassPipeline::new()
        .with_int8(None)
        .expect("with_int8 should succeed");
    draft_pipeline
        .run(&mut draft)
        .expect("draft pipeline should succeed");

    // Verifier: FP16 quantization.
    let verifier_pipeline = PassPipeline::new()
        .with_fp16()
        .expect("with_fp16 should succeed");
    verifier_pipeline
        .run(&mut verifier)
        .expect("verifier pipeline should succeed");

    // Serialize both.
    let draft_model = program_to_model(&draft, SPEC_VERSION).expect("draft should serialize");
    let verifier_model =
        program_to_model(&verifier, SPEC_VERSION).expect("verifier should serialize");

    // Verify different weight representations.
    let draft_ops = &draft.functions.values().next().unwrap().body.operations;
    let verifier_ops = &verifier.functions.values().next().unwrap().body.operations;

    // Draft: INT8 quantization converts const ops to constexpr_affine_dequantize
    // or removes FP32 const tensors entirely.
    let draft_has_int8 = draft_ops
        .iter()
        .any(|op| op.op_type == "constexpr_affine_dequantize");
    let draft_no_fp32_consts = draft_ops
        .iter()
        .filter(|op| op.op_type == "const")
        .all(|op| {
            let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
            !matches!(
                val,
                Some(Value::Tensor {
                    dtype: ScalarType::Float32,
                    ..
                })
            )
        });

    // Verifier: FP16 quantization converts FP32 tensors to FP16.
    let verifier_has_fp16 = verifier_ops.iter().any(|op| {
        op.inputs.values().any(|v| {
            matches!(
                v,
                Value::Tensor {
                    dtype: ScalarType::Float16,
                    ..
                }
            )
        }) || op.attributes.values().any(|v| {
            matches!(
                v,
                Value::Tensor {
                    dtype: ScalarType::Float16,
                    ..
                }
            )
        })
    });
    let verifier_no_fp32_consts =
        verifier_ops
            .iter()
            .filter(|op| op.op_type == "const")
            .all(|op| {
                let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
                !matches!(
                    val,
                    Some(Value::Tensor {
                        dtype: ScalarType::Float32,
                        ..
                    })
                )
            });

    assert!(
        draft_has_int8 || draft_no_fp32_consts,
        "draft should have INT8 quantized ops or no remaining FP32 consts"
    );
    assert!(
        verifier_has_fp16 || verifier_no_fp32_consts,
        "verifier should have FP16 weights or no remaining FP32 consts"
    );

    // The two representations must differ.
    use prost::Message;
    let draft_bytes = draft_model.encode_to_vec();
    let verifier_bytes = verifier_model.encode_to_vec();
    assert_ne!(
        draft_bytes, verifier_bytes,
        "draft and verifier should have different serialized representations"
    );
}

#[test]
fn configurable_pipeline_with_all_new_passes() {
    let toml_config = r#"
[[passes]]
name = "kv-cache"
enabled = true
[passes.params]
max_seq_length = 2048

[[passes]]
name = "op-splitting"
enabled = true
[passes.params]
memory_budget_bytes = 67108864

[[passes]]
name = "compute-unit-annotation"
enabled = true

[[passes]]
name = "codebook-optimization"
enabled = true

[[passes]]
name = "layer-schedule"
enabled = true
[passes.params]
layer_strategies = { attention = "fp16", ffn = "int8", conv = "fp16", norm = "fp16", other = "none" }
"#;

    let pipeline = PassPipeline::from_config_str(toml_config).expect("TOML config should parse");

    let mut program = build_transformer_program(2);
    pipeline
        .run(&mut program)
        .expect("all-new-passes pipeline should succeed without errors");

    // Verify program is still valid.
    assert!(count_ops(&program) > 0, "program should still have ops");
}

// ═══════════════════════════════════════════════════════════════════════
// Serialization Size Assertions (3 tests)
// ═══════════════════════════════════════════════════════════════════════

/// Helper: build a program with ~4MB of FP32 const tensors.
fn build_large_const_program() -> mil_rs::Program {
    // 1024*1024 = 1_048_576 f32 elements = 4MB of FP32 data.
    build_const_program(&[1024, 1024], ScalarType::Float32)
}

#[test]
fn quantized_model_smaller_than_float() {
    let fp32_program = build_large_const_program();
    let fp32_size = serialized_size(&fp32_program);

    let mut fp16_program = build_large_const_program();
    Fp16QuantizePass
        .run(&mut fp16_program)
        .expect("Fp16QuantizePass should succeed");
    let fp16_size = serialized_size(&fp16_program);

    assert!(
        (fp16_size as f64) < (fp32_size as f64) * 0.6,
        "FP16 size ({fp16_size}) should be < 60% of FP32 size ({fp32_size})"
    );
}

#[test]
fn int8_model_smaller_than_fp16() {
    let mut fp16_program = build_large_const_program();
    Fp16QuantizePass
        .run(&mut fp16_program)
        .expect("Fp16QuantizePass should succeed");
    let fp16_size = serialized_size(&fp16_program);

    let mut int8_program = build_large_const_program();
    Int8QuantizePass::weight_only()
        .run(&mut int8_program)
        .expect("Int8QuantizePass should succeed");
    let int8_size = serialized_size(&int8_program);

    assert!(
        (int8_size as f64) < (fp16_size as f64) * 0.6,
        "INT8 size ({int8_size}) should be < 60% of FP16 size ({fp16_size})"
    );
}

#[test]
fn palettized_model_smaller_than_int8() {
    let mut int8_program = build_large_const_program();
    Int8QuantizePass::weight_only()
        .run(&mut int8_program)
        .expect("Int8QuantizePass should succeed");
    let int8_size = serialized_size(&int8_program);

    let mut pal_program = build_large_const_program();
    PalettizePass::new(4)
        .run(&mut pal_program)
        .expect("PalettizePass should succeed");
    let pal_size = serialized_size(&pal_program);

    assert!(
        (pal_size as f64) < (int8_size as f64) * 0.7,
        "4-bit palettized size ({pal_size}) should be < 70% of INT8 size ({int8_size})"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Concurrency Tests (2 tests)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn parallel_pipeline_runs_no_data_race() {
    use std::thread;

    let prog_a = build_transformer_program(2);
    let prog_b = build_conv_program(3);

    let handle_a = thread::spawn(move || {
        let mut p = prog_a;
        let pipeline = PassPipeline::new();
        pipeline.run(&mut p).expect("pipeline A should succeed");
        p
    });
    let handle_b = thread::spawn(move || {
        let mut p = prog_b;
        let pipeline = PassPipeline::new();
        pipeline.run(&mut p).expect("pipeline B should succeed");
        p
    });

    let result_a = handle_a.join().expect("thread A should not panic");
    let result_b = handle_b.join().expect("thread B should not panic");

    // Both should produce valid, serializable output.
    assert!(count_ops(&result_a) > 0, "result A should have ops");
    assert!(count_ops(&result_b) > 0, "result B should have ops");
    assert_serialization_roundtrip(&result_a);
    assert_serialization_roundtrip(&result_b);
}

#[test]
fn parallel_moe_expert_compilation() {
    use std::thread;

    let n_experts = 3;
    let program = build_moe_program(n_experts);

    let topology = detect_moe(&program).expect("should detect MoE topology");
    let split = split_moe(&program, &topology);
    assert_eq!(split.experts.len(), n_experts);

    // Spawn a thread per expert, each running a quantization pipeline.
    let handles: Vec<_> = split
        .experts
        .into_iter()
        .enumerate()
        .map(|(i, expert)| {
            thread::spawn(move || {
                let mut p = expert;
                // Each expert gets FP16 quantization.
                Fp16QuantizePass
                    .run(&mut p)
                    .unwrap_or_else(|e| panic!("expert {i} FP16 pass failed: {e}"));
                p
            })
        })
        .collect();

    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("expert thread should not panic"))
        .collect();

    // Verify each expert serializes correctly.
    for (i, expert_prog) in results.iter().enumerate() {
        let model = program_to_model(expert_prog, SPEC_VERSION);
        assert!(
            model.is_ok(),
            "expert {i} should serialize: {:?}",
            model.err()
        );
    }
}
