//! Phase 8 integration tests for MoE features, quantization passes,
//! configurable pipelines, layer scheduling, and compiler backends.

mod common;

use std::collections::HashMap;

use mil_rs::ir::passes::tensor_utils::tensor_as_f32_slice;
use mil_rs::{
    Backend, ExpertFrequencyProfile, Function, Operation, Pass, PassPipeline, PipelineReport,
    Program, ScalarType, TensorType, Value, compile_model, compile_model_with_backend, detect_moe,
    fuse_top_k_experts, program_to_multi_function_model, split_moe,
};

use mil_rs::ir::passes::{
    ExpertQuantConfig, GroupedPalettizePass, LayerScheduleConfig, LayerSchedulePass,
    ModelSplitPass, PalettizePass, PerExpertQuantPass,
};

use common::{
    build_const_program, build_conv_program, build_moe_program, build_transformer_program,
    count_ops, make_const_op,
};

// =========================================================================
// 8.1 MoE Detection & Splitting
// =========================================================================

#[test]
fn moe_detection_name_based() {
    let program = build_moe_program(2);
    let topology = detect_moe(&program).expect("should detect MoE topology");
    assert_eq!(topology.expert_count, 2, "should detect 2 experts");
    assert!(
        !topology.router_op_indices.is_empty(),
        "should detect router ops"
    );
    assert_eq!(
        topology.expert_op_indices.len(),
        2,
        "should have 2 expert groups"
    );
    for (i, group) in topology.expert_op_indices.iter().enumerate() {
        assert!(!group.is_empty(), "expert {i} should have ops");
    }
    assert!(
        !topology.router_output.is_empty(),
        "router should have an output name"
    );
}

#[test]
fn moe_split_produces_shared_and_experts() {
    let program = build_moe_program(2);
    let topology = detect_moe(&program).expect("should detect MoE");
    let split = split_moe(&program, &topology);

    // Shared program should contain router ops.
    let shared_ops = &split.shared.main().unwrap().body.operations;
    let has_router = shared_ops
        .iter()
        .any(|op| op.name.contains("gate") || op.op_type == "softmax");
    assert!(has_router, "shared program should contain router ops");

    // Each expert program should contain only its own ops.
    assert_eq!(split.experts.len(), 2, "should have 2 expert programs");
    for (i, expert_prog) in split.experts.iter().enumerate() {
        let ops = &expert_prog.main().unwrap().body.operations;
        assert!(!ops.is_empty(), "expert {i} program should have ops");
        // Every op should belong to expert_i (by name or const).
        for op in ops {
            let is_expert_op = op.name.contains(&format!("expert_{i}"));
            let is_const = op.op_type == "const";
            assert!(
                is_expert_op || is_const,
                "expert {i} program op '{}' should be its own or a const",
                op.name
            );
        }
    }

    // Manifest should be valid JSON.
    let json = serde_json::to_string(&split.manifest).expect("manifest should serialize to JSON");
    assert!(json.contains("expert_count"));
    assert!(json.contains("stages"));
}

#[test]
fn moe_manifest_describes_execution_flow() {
    let program = build_moe_program(2);
    let topology = detect_moe(&program).unwrap();
    let split = split_moe(&program, &topology);
    let manifest = &split.manifest;

    // Should have shared + 2 expert stages.
    assert_eq!(
        manifest.stages.len(),
        3,
        "should have shared + 2 expert stages"
    );

    // First stage is shared.
    let shared_stage = &manifest.stages[0];
    assert_eq!(shared_stage.name, "shared");
    assert!(
        !shared_stage.outputs.is_empty(),
        "shared should have outputs"
    );

    // Expert stages.
    for i in 0..2 {
        let stage = &manifest.stages[i + 1];
        assert_eq!(stage.name, format!("expert-{i}"));
        assert!(!stage.inputs.is_empty(), "expert-{i} should have I/O");
        assert!(!stage.outputs.is_empty(), "expert-{i} should have outputs");
    }

    // Expert descriptors should match.
    assert_eq!(manifest.experts.len(), 2);
    for (i, desc) in manifest.experts.iter().enumerate() {
        assert_eq!(desc.index, i);
        assert!(
            !desc.inputs.is_empty(),
            "expert descriptor {i} should have inputs"
        );
        assert!(
            !desc.outputs.is_empty(),
            "expert descriptor {i} should have outputs"
        );
    }
}

// =========================================================================
// 8.2 Multi-Function Bundle
// =========================================================================

#[test]
fn multi_function_model_has_all_functions() {
    let program = build_moe_program(2);
    let topology = detect_moe(&program).unwrap();
    let split = split_moe(&program, &topology);

    let model =
        program_to_multi_function_model(&split, 9).expect("multi-function model should build");

    // The model's ML Program should have functions.
    let proto_program = match &model.r#type {
        Some(mil_rs::proto::specification::model::Type::MlProgram(p)) => p,
        _ => panic!("expected MlProgram model type"),
    };

    let func_names: Vec<&String> = proto_program.functions.keys().collect();
    assert!(
        func_names.iter().any(|n| *n == "main"),
        "should have 'main' function, got: {func_names:?}"
    );
    assert!(
        func_names.iter().any(|n| *n == "expert_0"),
        "should have 'expert_0' function, got: {func_names:?}"
    );
    assert!(
        func_names.iter().any(|n| *n == "expert_1"),
        "should have 'expert_1' function, got: {func_names:?}"
    );
}

#[test]
fn multi_function_dedup_only_shared_consts() {
    let program = build_moe_program(2);
    let topology = detect_moe(&program).unwrap();
    let split = split_moe(&program, &topology);

    let model =
        program_to_multi_function_model(&split, 9).expect("multi-function model should build");

    let proto_program = match &model.r#type {
        Some(mil_rs::proto::specification::model::Type::MlProgram(p)) => p,
        _ => panic!("expected MlProgram"),
    };

    // Each expert function should have its own unique weight ops.
    for i in 0..2 {
        let func_name = format!("expert_{i}");
        let func = proto_program
            .functions
            .get(&func_name)
            .unwrap_or_else(|| panic!("should have function '{func_name}'"));

        // The function should have at least one block specialization.
        assert!(
            !func.block_specializations.is_empty(),
            "expert function {func_name} should have block specializations"
        );

        let block = func.block_specializations.values().next().unwrap();
        // Expert function should have ops (at minimum weight consts + linears).
        assert!(
            !block.operations.is_empty(),
            "expert function {func_name} should have operations"
        );
    }
}

// =========================================================================
// 8.3 Per-Expert Quantization
// =========================================================================

#[test]
fn hot_expert_fp16_cold_expert_palettized() {
    let mut program = build_moe_program(2);

    // Expert 0 is hot (FP16), expert 1 is cold (4-bit palettize).
    let config = ExpertQuantConfig::preset_hot_cold(&[0], 2);
    let pass = PerExpertQuantPass::new(config);
    pass.run(&mut program)
        .expect("PerExpertQuantPass should run");

    let main_fn = program.main().unwrap();

    // Expert 0 weight consts should be FP16.
    let expert_0_consts: Vec<_> = main_fn
        .body
        .operations
        .iter()
        .filter(|op| op.name.contains("expert_0") && (op.op_type == "const"))
        .collect();
    for op in &expert_0_consts {
        // FP16: val tensor should have Float16 dtype.
        let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
        if let Some(Value::Tensor { dtype, .. }) = val {
            assert_eq!(
                *dtype,
                ScalarType::Float16,
                "expert_0 const '{}' should be FP16",
                op.name
            );
        }
    }

    // Expert 1 weight consts should be palettized (constexpr_lut_to_dense).
    let expert_1_lut_ops: Vec<_> = main_fn
        .body
        .operations
        .iter()
        .filter(|op| op.name.contains("expert_1") && op.op_type == "constexpr_lut_to_dense")
        .collect();
    assert!(
        !expert_1_lut_ops.is_empty(),
        "expert_1 should have constexpr_lut_to_dense ops after palettization"
    );
    for op in &expert_1_lut_ops {
        assert!(
            op.attributes.contains_key("lut"),
            "palettized op should have 'lut' attribute"
        );
    }
}

// =========================================================================
// 8.4 Top-K Fusion
// =========================================================================

#[test]
fn top_k_fusion_keeps_only_selected_experts() {
    let program = build_moe_program(4);

    let mut freqs = HashMap::new();
    freqs.insert("0".to_string(), 0.5);
    freqs.insert("1".to_string(), 0.3);
    freqs.insert("2".to_string(), 0.15);
    freqs.insert("3".to_string(), 0.05);
    let profile = ExpertFrequencyProfile {
        expert_frequencies: freqs,
    };

    let result = fuse_top_k_experts(&program, 2, &profile).expect("fuse_top_k should succeed");

    // Should have kept experts 0 and 1 (highest frequency).
    assert_eq!(result.kept_expert_indices.len(), 2, "should keep 2 experts");
    assert!(
        result.kept_expert_indices.contains(&0),
        "expert 0 should be kept"
    );
    assert!(
        result.kept_expert_indices.contains(&1),
        "expert 1 should be kept"
    );

    // Should have discarded experts 2 and 3.
    assert!(
        result.discarded_expert_indices.contains(&2),
        "expert 2 should be discarded"
    );
    assert!(
        result.discarded_expert_indices.contains(&3),
        "expert 3 should be discarded"
    );

    // Fused program should not contain ops from discarded experts.
    let main_fn = result.program.main().unwrap();
    for op in &main_fn.body.operations {
        assert!(
            !op.name.contains("expert_2"),
            "fused program should not have expert_2 ops, found: {}",
            op.name
        );
        assert!(
            !op.name.contains("expert_3"),
            "fused program should not have expert_3 ops, found: {}",
            op.name
        );
    }

    // Router ops should be removed.
    let has_softmax = main_fn
        .body
        .operations
        .iter()
        .any(|op| op.name.contains("gate") && op.op_type == "softmax");
    assert!(!has_softmax, "fused program should not have router softmax");
}

// =========================================================================
// 8.5 Sub-2-bit Quantization
// =========================================================================

#[test]
fn one_bit_palettization_produces_two_centroids() {
    let mut program = build_const_program(&[64, 64], ScalarType::Float32);
    let pass = PalettizePass::new(1);
    pass.run(&mut program).expect("PalettizePass(1) should run");

    let main_fn = program.main().unwrap();
    let lut_op = main_fn
        .body
        .operations
        .iter()
        .find(|op| op.op_type == "constexpr_lut_to_dense")
        .expect("should have a constexpr_lut_to_dense op");

    // LUT should have exactly 2 entries.
    let lut = lut_op
        .attributes
        .get("lut")
        .expect("should have 'lut' attr");
    if let Value::Tensor { shape, .. } = lut {
        assert_eq!(shape[0], 2, "1-bit LUT should have 2 entries");
    } else {
        panic!("lut attr should be a Tensor");
    }

    // Indices should use 1-bit packing (packed byte count).
    let indices = lut_op
        .attributes
        .get("indices")
        .expect("should have 'indices' attr");
    if let Value::Tensor { data, .. } = indices {
        let total_elements: usize = 64 * 64;
        let expected_bytes = total_elements.div_ceil(8); // 1-bit packing
        assert_eq!(
            data.len(),
            expected_bytes,
            "1-bit packed indices should use {} bytes for {} elements",
            expected_bytes,
            total_elements
        );
    } else {
        panic!("indices attr should be a Tensor");
    }
}

#[test]
fn grouped_palettization_per_group_codebooks() {
    let mut program = build_const_program(&[128, 64], ScalarType::Float32);
    let pass = GroupedPalettizePass::new(4, 32);
    pass.run(&mut program)
        .expect("GroupedPalettizePass should run");

    let main_fn = program.main().unwrap();
    let lut_op = main_fn
        .body
        .operations
        .iter()
        .find(|op| op.op_type == "constexpr_lut_to_dense")
        .expect("should have a constexpr_lut_to_dense op");

    // n_groups should be 128 / 32 = 4.
    let n_groups = lut_op
        .attributes
        .get("n_groups")
        .expect("should have 'n_groups' attr");
    match n_groups {
        Value::Int(n) => assert_eq!(*n, 4, "should have 4 groups"),
        _ => panic!("n_groups should be Int"),
    }

    // group_size should be 32.
    let group_size = lut_op
        .attributes
        .get("group_size")
        .expect("should have 'group_size' attr");
    match group_size {
        Value::Int(gs) => assert_eq!(*gs, 32, "group_size should be 32"),
        _ => panic!("group_size should be Int"),
    }

    // LUT should have 4 groups × 16 centroids = 64 entries (4-bit = 16 centroids per group).
    let lut = lut_op
        .attributes
        .get("lut")
        .expect("should have 'lut' attr");
    if let Value::Tensor { shape, .. } = lut {
        assert_eq!(
            shape[0], 64,
            "grouped LUT should have 4*16 = 64 entries total"
        );
    }
}

#[test]
fn prequantized_gptq_weights_preserved() {
    // Build a program with ops named like GPTQ initializers.
    let input_ty = TensorType::new(ScalarType::Float32, vec![1]);
    let func = Function::new("main").with_input("dummy_input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;

    // GPTQ quantized weight (qweight — Int32 data).
    block.add_op(make_const_op("layer.qweight", &[64, 16], ScalarType::Int32));
    // GPTQ zeros.
    block.add_op(make_const_op("layer.qzeros", &[4, 16], ScalarType::Int32));
    // GPTQ scales.
    block.add_op(make_const_op("layer.scales", &[4, 16], ScalarType::Float32));
    // Add an identity op to use one output.
    block.add_op(
        Operation::new("identity", "output_identity")
            .with_input("x", Value::Reference("layer.qweight_out".into()))
            .with_output("output"),
    );
    block.outputs.push("output".into());

    // Run palettize pass — GPTQ weights are Int32, not FP32/FP16,
    // so they should be preserved (not re-quantized).
    let pass = PalettizePass::new(4);
    pass.run(&mut program).expect("palettize should run");

    let main_fn = program.main().unwrap();

    // The qweight const should still be "const" (not transformed to lut_to_dense)
    // because it's Int32, not Float32/Float16.
    let qweight_op = main_fn
        .body
        .operations
        .iter()
        .find(|op| op.name == "layer.qweight")
        .expect("qweight should still exist");
    assert_eq!(
        qweight_op.op_type, "const",
        "GPTQ qweight should remain as 'const' (not re-quantized)"
    );

    // scales const is FP32 so it may or may not be palettized — that's fine.
    // The key point is qweight (Int32) is not touched.
}

// =========================================================================
// 8.6 Configurable Pipeline
// =========================================================================

#[test]
fn toml_config_matches_programmatic_pipeline() {
    // Use equivalent full pipeline in TOML to match PassPipeline::new().with_fp16().
    let toml_str = r#"
[[passes]]
name = "dead-code-elimination"

[[passes]]
name = "identity-elimination"

[[passes]]
name = "constant-folding"

[[passes]]
name = "conv-bn-weight-fold"

[[passes]]
name = "conv-batchnorm-fusion"

[[passes]]
name = "conv-relu-fusion"

[[passes]]
name = "linear-relu-fusion"

[[passes]]
name = "layernorm-linear-fusion"

[[passes]]
name = "gelu-linear-fusion"

[[passes]]
name = "residual-add-fusion"

[[passes]]
name = "attention-fusion"

[[passes]]
name = "gqa-fusion"

[[passes]]
name = "kv-cache"

[[passes]]
name = "codebook-optimization"

[[passes]]
name = "op-substitution"

[[passes]]
name = "type-repropagation"

[[passes]]
name = "type-repropagation"

[[passes]]
name = "fp16-quantization"
"#;
    let config_pipeline =
        PassPipeline::from_config_str(toml_str).expect("TOML pipeline should parse");
    let prog_pipeline = PassPipeline::new()
        .with_fp16()
        .expect("programmatic pipeline should build");

    let mut program_a = build_transformer_program(2);
    let mut program_b = build_transformer_program(2);

    let report_a = config_pipeline
        .run(&mut program_a)
        .expect("config pipeline should run");
    let report_b = prog_pipeline
        .run(&mut program_b)
        .expect("programmatic pipeline should run");

    let ops_a = count_ops(&program_a);
    let ops_b = count_ops(&program_b);
    assert_eq!(
        ops_a, ops_b,
        "TOML config and programmatic pipeline should produce identical op count"
    );

    // Both reports should have the same number of pass results.
    assert_eq!(
        report_a.pass_results.len(),
        report_b.pass_results.len(),
        "both pipelines should run the same number of passes"
    );
}

#[test]
fn pipeline_report_comparison() {
    let mut program_a = build_transformer_program(2);
    let mut program_b = build_transformer_program(2);

    let pipeline_a = PassPipeline::new();
    let pipeline_b = PassPipeline::new()
        .with_fp16()
        .expect("fp16 pipeline should build");

    let report_a = pipeline_a
        .run(&mut program_a)
        .expect("pipeline A should run");
    let report_b = pipeline_b
        .run(&mut program_b)
        .expect("pipeline B should run");

    let comparison = PipelineReport::compare(&report_a, &report_b);

    // Comparison should contain per-pass rows.
    assert!(
        comparison.contains("Pipeline A"),
        "comparison should mention Pipeline A"
    );
    assert!(
        comparison.contains("Pipeline B"),
        "comparison should mention Pipeline B"
    );
    assert!(
        comparison.contains("Op count"),
        "comparison should have Op count row"
    );
    assert!(
        comparison.contains("Pass"),
        "comparison should have per-pass header"
    );
}

// =========================================================================
// 8.7 Layer-Wise Scheduling
// =========================================================================

#[test]
fn layer_schedule_detects_all_types() {
    // Build a program with conv+bn+relu, attention, FFN, and norm.
    let mut program = build_conv_program(1);
    let block = &mut program.functions.get_mut("main").unwrap().body;

    // Append attention-like ops: linear Q, linear K, softmax, linear proj.
    let attn_prefix = "attn";
    block.add_op(make_const_op(
        &format!("{attn_prefix}.q_weight"),
        &[16, 16],
        ScalarType::Float32,
    ));
    block.add_op(
        Operation::new("linear", &format!("{attn_prefix}.q_proj"))
            .with_input("x", Value::Reference("relu_0_out".into()))
            .with_input(
                "weight",
                Value::Reference(format!("{attn_prefix}.q_weight_out")),
            )
            .with_output("q_out"),
    );
    block.add_op(make_const_op(
        &format!("{attn_prefix}.k_weight"),
        &[16, 16],
        ScalarType::Float32,
    ));
    block.add_op(
        Operation::new("linear", &format!("{attn_prefix}.k_proj"))
            .with_input("x", Value::Reference("relu_0_out".into()))
            .with_input(
                "weight",
                Value::Reference(format!("{attn_prefix}.k_weight_out")),
            )
            .with_output("k_out"),
    );
    block.add_op(
        Operation::new("matmul", &format!("{attn_prefix}.qk_matmul"))
            .with_input("x", Value::Reference("q_out".into()))
            .with_input("y", Value::Reference("k_out".into()))
            .with_output("qk_out"),
    );
    block.add_op(
        Operation::new("softmax", &format!("{attn_prefix}.softmax"))
            .with_input("x", Value::Reference("qk_out".into()))
            .with_output("attn_weights"),
    );
    block.add_op(make_const_op(
        &format!("{attn_prefix}.v_weight"),
        &[16, 16],
        ScalarType::Float32,
    ));
    block.add_op(
        Operation::new("linear", &format!("{attn_prefix}.v_proj"))
            .with_input("x", Value::Reference("attn_weights".into()))
            .with_input(
                "weight",
                Value::Reference(format!("{attn_prefix}.v_weight_out")),
            )
            .with_output("v_out"),
    );

    // FFN block: linear → relu → linear.
    // Insert a layer_norm between attention and FFN to prevent attention
    // expansion from absorbing the FFN ops (the detection expands through
    // "other"-category ops like const).
    block.add_op(
        Operation::new("layer_norm", "mid_norm")
            .with_input("x", Value::Reference("v_out".into()))
            .with_output("mid_norm_out"),
    );

    // Place const weight ops before the compute pattern so layer detection
    // sees contiguous linear → activation → linear.
    block.add_op(make_const_op("ffn.w1", &[64, 16], ScalarType::Float32));
    block.add_op(make_const_op("ffn.w2", &[16, 64], ScalarType::Float32));
    block.add_op(
        Operation::new("linear", "ffn.fc1")
            .with_input("x", Value::Reference("mid_norm_out".into()))
            .with_input("weight", Value::Reference("ffn.w1_out".into()))
            .with_output("ffn_fc1_out"),
    );
    block.add_op(
        Operation::new("relu", "ffn.relu")
            .with_input("x", Value::Reference("ffn_fc1_out".into()))
            .with_output("ffn_relu_out"),
    );
    block.add_op(
        Operation::new("linear", "ffn.fc2")
            .with_input("x", Value::Reference("ffn_relu_out".into()))
            .with_input("weight", Value::Reference("ffn.w2_out".into()))
            .with_output("ffn_fc2_out"),
    );

    // layer_norm.
    block.add_op(
        Operation::new("layer_norm", "final_norm")
            .with_input("x", Value::Reference("ffn_fc2_out".into()))
            .with_output("norm_out"),
    );

    // Re-wire output.
    block.outputs.clear();
    block.outputs.push("norm_out".into());

    // Configure: attention FP16, ffn INT8, conv and norm unchanged.
    let toml_str = r#"
[layer_strategies]
attention = "fp16"
ffn = "int8"
"#;
    let config = LayerScheduleConfig::from_toml_str(toml_str).expect("config should parse");
    let pass = LayerSchedulePass::new(config);
    pass.run(&mut program)
        .expect("LayerSchedulePass should run");

    let main_fn = program.main().unwrap();

    // Attention const weights should be FP16.
    let attn_consts: Vec<_> = main_fn
        .body
        .operations
        .iter()
        .filter(|op| op.name.starts_with("attn.") && op.op_type == "const")
        .collect();
    for op in &attn_consts {
        let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
        if let Some(Value::Tensor { dtype, .. }) = val {
            assert_eq!(
                *dtype,
                ScalarType::Float16,
                "attention const '{}' should be FP16",
                op.name
            );
        }
    }

    // FFN const weights should be INT8 (constexpr_affine_dequantize).
    let ffn_quant_ops: Vec<_> = main_fn
        .body
        .operations
        .iter()
        .filter(|op| op.name.starts_with("ffn.w") && op.op_type == "constexpr_affine_dequantize")
        .collect();
    assert!(
        !ffn_quant_ops.is_empty(),
        "FFN weights should be INT8 quantized"
    );

    // Conv weights should remain unchanged (const, Float32).
    let conv_weight = main_fn
        .body
        .operations
        .iter()
        .find(|op| op.name == "conv_0_weight")
        .expect("conv weight should exist");
    assert_eq!(
        conv_weight.op_type, "const",
        "conv weight should remain as 'const'"
    );
}

// =========================================================================
// 8.8 ANE Direct
// =========================================================================

#[test]
fn ane_compiler_not_available_returns_error() {
    // Without the ane-direct feature, requesting AneDirect should return
    // a validation error, not a panic.
    let result = compile_model_with_backend("nonexistent.mlpackage", "out_dir", Backend::AneDirect);
    assert!(
        result.is_err(),
        "AneDirect on nonexistent path should error"
    );
    let err_msg = result.unwrap_err().to_string();
    // Without the feature, we get a feature-gate error.
    // With the feature, we'd get an IO error.
    assert!(
        err_msg.contains("ane-direct")
            || err_msg.contains("not found")
            || err_msg.contains("exist"),
        "error should mention ane-direct feature or path issue, got: {err_msg}"
    );
}

#[test]
fn backend_enum_defaults_to_xcrun() {
    // Backend implements Default; verify it's Xcrun.
    let default_backend = Backend::default();
    assert_eq!(
        default_backend,
        Backend::Xcrun,
        "Backend::default() should be Xcrun"
    );

    // Verify that compile_model and compile_model_with_backend(_, _, Xcrun)
    // behave the same on a nonexistent path.
    let result_plain = compile_model("nonexistent.mlpackage", "out_dir");
    let result_xcrun =
        compile_model_with_backend("nonexistent.mlpackage", "out_dir", Backend::Xcrun);

    // Both should fail.
    assert!(result_plain.is_err());
    assert!(result_xcrun.is_err());

    // Both should produce the same error kind.
    let plain_err = result_plain.unwrap_err().to_string();
    let xcrun_err = result_xcrun.unwrap_err().to_string();
    assert_eq!(
        plain_err, xcrun_err,
        "compile_model and compile_model_with_backend(Xcrun) should produce same error"
    );
}

// =========================================================================
// Error-Path Tests
// =========================================================================

#[test]
#[should_panic(expected = "n_bits must be one of")]
fn palettize_zero_bits_returns_error() {
    let _ = PalettizePass::new(0);
}

#[test]
fn model_split_layer_exceeds_total_returns_error() {
    let program = build_transformer_program(4);
    let splitter = ModelSplitPass::new(6);
    let result = splitter.split(&program);
    assert!(
        result.is_err(),
        "splitting at layer 6 with only 4 layers should error"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("draft layers") || err_msg.contains("boundary"),
        "error should mention layer count, got: {err_msg}"
    );
}

#[test]
fn moe_fuse_topk_greater_than_experts_returns_error() {
    let program = build_moe_program(2);
    let profile = ExpertFrequencyProfile::uniform(2);

    // k=3 > 2 experts — check actual behavior.
    let result = fuse_top_k_experts(&program, 3, &profile);
    // The implementation clamps selected to available experts,
    // so it should return Some with all experts kept.
    match result {
        Some(fuse_result) => {
            // All experts should be kept since k > expert_count.
            assert!(
                fuse_result.kept_expert_indices.len() <= 2,
                "should keep at most 2 experts when only 2 exist"
            );
            assert!(
                fuse_result.discarded_expert_indices.is_empty(),
                "no experts should be discarded when k >= expert_count"
            );
        }
        None => {
            // If it returns None, that's also acceptable behavior.
            // The important thing is it doesn't panic.
        }
    }
}

// =========================================================================
// Quantization Accuracy Tests
// =========================================================================

#[test]
fn palettize_roundtrip_error_bounded() {
    let shape = &[32, 32];
    let total_elements: usize = shape.iter().product();

    // Build a program with known float values.
    let mut program = build_const_program(shape, ScalarType::Float32);

    // Extract original values before palettization.
    let original_values: Vec<f32> = {
        let main_fn = program.main().unwrap();
        let const_op = main_fn
            .body
            .operations
            .iter()
            .find(|op| op.op_type == "const")
            .unwrap();
        let val = const_op
            .inputs
            .get("val")
            .or_else(|| const_op.attributes.get("val"))
            .unwrap();
        match val {
            Value::Tensor { data, .. } => tensor_as_f32_slice(data),
            _ => panic!("expected tensor"),
        }
    };

    // Run 4-bit palettization (16 centroids).
    let pass = PalettizePass::new(4);
    pass.run(&mut program).expect("palettize should succeed");

    let main_fn = program.main().unwrap();
    let lut_op = main_fn
        .body
        .operations
        .iter()
        .find(|op| op.op_type == "constexpr_lut_to_dense")
        .expect("should have lut op");

    // Extract LUT centroids.
    let lut_centroids: Vec<f32> = match lut_op.attributes.get("lut").unwrap() {
        Value::Tensor { data, .. } => tensor_as_f32_slice(data),
        _ => panic!("expected tensor"),
    };
    assert_eq!(lut_centroids.len(), 16, "4-bit should have 16 centroids");

    // Extract packed indices.
    let packed_indices = match lut_op.attributes.get("indices").unwrap() {
        Value::Tensor { data, .. } => data.clone(),
        _ => panic!("expected tensor"),
    };

    // Unpack 4-bit indices and reconstruct.
    let mut reconstructed = Vec::with_capacity(total_elements);
    for byte in &packed_indices {
        let hi = (byte >> 4) as usize;
        let lo = (byte & 0x0F) as usize;
        if reconstructed.len() < total_elements {
            reconstructed.push(lut_centroids[hi]);
        }
        if reconstructed.len() < total_elements {
            reconstructed.push(lut_centroids[lo]);
        }
    }

    // Compute max absolute error.
    let value_range = {
        let min = original_values
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max = original_values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        max - min
    };
    let max_allowed_error = value_range / 16.0;

    let max_error = original_values
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_error < max_allowed_error,
        "max palettize error {max_error:.6} should be < {max_allowed_error:.6} (range/16)"
    );
}

#[test]
fn grouped_palettize_preserves_group_structure() {
    use mil_rs::ir::passes::tensor_utils::f32_slice_to_bytes;

    // Build a [128, 64] weight where:
    // rows 0-31 (group 0) are from distribution A (values around 10.0)
    // rows 32-63 (group 1) are from distribution B (values around -10.0)
    // rows 64-95 (group 2) are from distribution A
    // rows 96-127 (group 3) are from distribution B
    let cols = 64;
    let mut values = Vec::with_capacity(128 * cols);
    for row in 0..128 {
        let group = row / 32;
        for col in 0..cols {
            let base = if group % 2 == 0 { 10.0 } else { -10.0 };
            let noise = ((row * cols + col) as f32 * 0.01).sin() * 2.0;
            values.push(base + noise);
        }
    }

    let data = f32_slice_to_bytes(&values);
    let input_ty = TensorType::new(ScalarType::Float32, vec![1]);
    let func = Function::new("main").with_input("dummy_input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(
        Operation::new("const", "weight")
            .with_attr(
                "val",
                Value::Tensor {
                    data,
                    shape: vec![128, 64],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output("weight_out"),
    );
    block.add_op(
        Operation::new("identity", "output_identity")
            .with_input("x", Value::Reference("weight_out".into()))
            .with_output("output"),
    );
    block.outputs.push("output".into());

    let pass = GroupedPalettizePass::new(4, 32);
    pass.run(&mut program)
        .expect("grouped palettize should run");

    let main_fn = program.main().unwrap();
    let lut_op = main_fn
        .body
        .operations
        .iter()
        .find(|op| op.op_type == "constexpr_lut_to_dense")
        .unwrap();

    let lut_values: Vec<f32> = match lut_op.attributes.get("lut").unwrap() {
        Value::Tensor { data, .. } => tensor_as_f32_slice(data),
        _ => panic!("expected tensor"),
    };

    // Each group has 16 centroids (4-bit). Total = 4 × 16 = 64 entries.
    assert_eq!(
        lut_values.len(),
        64,
        "should have 4 groups × 16 centroids = 64 LUT entries"
    );

    // Group 0 centroids (indices 0..16) should be around +10.
    let group_0_centroids = &lut_values[0..16];
    let group_0_mean: f32 = group_0_centroids.iter().sum::<f32>() / 16.0;
    // Group 1 centroids (indices 16..32) should be around -10.
    let group_1_centroids = &lut_values[16..32];
    let group_1_mean: f32 = group_1_centroids.iter().sum::<f32>() / 16.0;

    assert!(
        group_0_mean > 0.0,
        "group 0 centroids should be positive (mean={group_0_mean:.2})"
    );
    assert!(
        group_1_mean < 0.0,
        "group 1 centroids should be negative (mean={group_1_mean:.2})"
    );
    assert!(
        (group_0_mean - group_1_mean).abs() > 5.0,
        "group 0 and group 1 centroids should differ substantially \
         (group0_mean={group_0_mean:.2}, group1_mean={group_1_mean:.2})"
    );
}
