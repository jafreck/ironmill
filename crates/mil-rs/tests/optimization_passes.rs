//! Phase 5 integration tests for optimization passes.
//!
//! Tests cover NHWC layout, ANE validation, mixed-precision, fusion patterns,
//! compute-unit annotations, error paths, and quantization numerical accuracy.

mod common;

use mil_rs::ir::passes::tensor_utils::{f32_slice_to_bytes, tensor_as_f32_slice};
use mil_rs::ir::passes::{
    ComputeUnitAnnotationPass, Fp16QuantizePass, Granularity, Int8QuantizePass,
    LayoutOptimizationPass, MixedPrecisionConfig, MixedPrecisionPass, OpSplittingPass,
    PolarQuantPass,
};
use mil_rs::{
    Block, ComputeUnit, Function, Operation, Pass, PassPipeline, Program, ScalarType, TensorType,
    Value, model_to_program, program_to_model, validate_ane_compatibility,
    validation_report_to_json,
};

// ═══════════════════════════════════════════════════════════════════════
// 5.1: NHWC Layout
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn pipeline_on_conv_model_produces_valid_proto() {
    let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);
    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    let mut prev = "input".to_string();
    for i in 0..3 {
        let out = format!("conv_{i}_out");
        block.add_op(
            Operation::new("conv", &format!("conv_{i}"))
                .with_input("x", Value::Reference(prev))
                .with_output(&out),
        );
        prev = out;
    }
    block.outputs.push(prev);

    let pipeline = PassPipeline::new();
    pipeline.run(&mut program).expect("pipeline should succeed");

    let model = program_to_model(&program, 8).expect("program_to_model should succeed");

    // Decode the proto and verify the model is valid.
    use prost::Message;
    let bytes = model.encode_to_vec();
    let decoded =
        mil_rs::proto::specification::Model::decode(bytes.as_slice()).expect("should decode");

    let proto_program = match &decoded.r#type {
        Some(mil_rs::proto::specification::model::Type::MlProgram(p)) => p,
        _ => panic!("expected MlProgram"),
    };
    let proto_func = proto_program.functions.values().next().expect("function");
    let proto_block = proto_func
        .block_specializations
        .values()
        .next()
        .expect("block");

    // Conv ops should be preserved.
    let conv_ops: Vec<_> = proto_block
        .operations
        .iter()
        .filter(|op| op.r#type == "conv")
        .collect();
    assert_eq!(conv_ops.len(), 3, "should have 3 conv ops");
}

#[test]
fn nhwc_layout_idempotent() {
    let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 64, 64]);
    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(
        Operation::new("conv", "conv_0")
            .with_input("x", Value::Reference("input".into()))
            .with_output("conv_out"),
    );
    block.outputs.push("conv_out".into());

    // First run — no transpose pairs to cancel, so the pass is a no-op.
    LayoutOptimizationPass.run(&mut program).unwrap();
    let ops_first: Vec<String> = program.functions["main"]
        .body
        .operations
        .iter()
        .map(|op| format!("{}:{}", op.op_type, op.name))
        .collect();
    assert_eq!(
        ops_first,
        vec!["conv:conv_0"],
        "pass should be no-op on a conv without transpose pairs"
    );

    // Second run — still a no-op, idempotent.
    LayoutOptimizationPass.run(&mut program).unwrap();
    let ops_second: Vec<String> = program.functions["main"]
        .body
        .operations
        .iter()
        .map(|op| format!("{}:{}", op.op_type, op.name))
        .collect();
    assert_eq!(ops_first, ops_second, "layout pass should be idempotent");

    // Program remains serializable after two runs.
    program_to_model(&program, 8).expect("should serialize after second layout run");
}

// ═══════════════════════════════════════════════════════════════════════
// 5.2: ANE Validation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn validate_conv_exceeding_ane_limits() {
    let mut program = Program::new("1");
    let mut func = Function::new("main");
    let input_type = TensorType::new(ScalarType::Float16, vec![1, 32, 64, 64]);
    func.inputs.push(("input".to_string(), input_type));

    // Weight [64, 32, 32, 32] — kernel 32×32 exceeds ANE limit of 16.
    let weight_type = TensorType::new(ScalarType::Float16, vec![64, 32, 32, 32]);
    let mut weight_const = Operation::new("const", "weight")
        .with_input(
            "val",
            Value::Tensor {
                data: vec![0u8; 64 * 32 * 32 * 32 * 2],
                shape: vec![64, 32, 32, 32],
                dtype: ScalarType::Float16,
            },
        )
        .with_output("weight_out");
    weight_const.output_types = vec![Some(weight_type)];

    let conv = Operation::new("conv", "large_conv")
        .with_input("x", Value::Reference("input".into()))
        .with_input("weight", Value::Reference("weight_out".into()))
        .with_output("conv_out");

    func.body.add_op(weight_const);
    func.body.add_op(conv);
    func.body.outputs.push("conv_out".into());
    program.add_function(func);

    let report = validate_ane_compatibility(&program);

    // The conv op should be in fallback_ops and NOT ane_eligible.
    let conv_report = report
        .fallback_ops
        .iter()
        .find(|r| r.name == "large_conv")
        .expect("large_conv should be in fallback_ops");
    assert!(
        !conv_report.ane_eligible,
        "large_conv should NOT be ane_eligible"
    );
    assert!(
        conv_report.reason.as_ref().unwrap().contains("kernel"),
        "reason should mention kernel: {:?}",
        conv_report.reason
    );

    // GPU fallback: the ane_compute_pct should be less than 100% because the
    // conv is not eligible (const ops are excluded from the denominator).
    assert!(
        report.ane_compute_pct < 100.0,
        "ane_compute_pct should be < 100% due to conv fallback, got {}",
        report.ane_compute_pct
    );
}

#[test]
fn validate_json_report_structure() {
    let mut program = Program::new("1");
    let mut func = Function::new("main");
    let input_type = TensorType::new(ScalarType::Float16, vec![1, 32, 64, 64]);
    func.inputs.push(("input".to_string(), input_type));

    // ANE-eligible: relu
    func.body.add_op(
        Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("input".into()))
            .with_output("relu_out"),
    );

    // NOT ANE-eligible: a custom op
    func.body.add_op(
        Operation::new("custom_unsupported_op", "custom_0")
            .with_input("x", Value::Reference("relu_out".into()))
            .with_output("custom_out"),
    );

    func.body.outputs.push("custom_out".into());
    program.add_function(func);

    let report = validate_ane_compatibility(&program);
    let json_str = validation_report_to_json(&report);
    let json: serde_json::Value = serde_json::from_str(&json_str).expect("should parse JSON");

    // Per-op ane_eligible field
    let compat = json["ane_compatible"]
        .as_array()
        .expect("ane_compatible array");
    let fallback = json["fallback_ops"].as_array().expect("fallback_ops array");
    for op_report in compat.iter().chain(fallback.iter()) {
        assert!(
            op_report.get("ane_eligible").is_some(),
            "every op report should have ane_eligible field"
        );
        assert!(
            op_report.get("performance_annotations").is_some(),
            "every op report should have performance_annotations"
        );
    }

    // total_estimated_flops and ane_compute_pct
    let total_flops = json["total_estimated_flops"]
        .as_u64()
        .expect("total_estimated_flops should be a number");
    assert!(total_flops >= 0, "total_estimated_flops should be >= 0");

    let ane_pct = json["ane_compute_pct"]
        .as_f64()
        .expect("ane_compute_pct should be a number");
    assert!(
        (0.0..=100.0).contains(&ane_pct),
        "ane_compute_pct should be 0–100, got {ane_pct}"
    );
}

#[test]
fn validate_flops_proportional_to_shape() {
    fn make_conv_program(h: usize, w: usize) -> Program {
        let mut program = Program::new("1");
        let mut func = Function::new("main");
        let input_type = TensorType::new(ScalarType::Float16, vec![1, 3, h, w]);
        func.inputs.push(("input".to_string(), input_type));

        // Weight [16, 3, 3, 3]
        let weight_type = TensorType::new(ScalarType::Float16, vec![16, 3, 3, 3]);
        let mut weight_const = Operation::new("const", "weight")
            .with_input(
                "val",
                Value::Tensor {
                    data: vec![0u8; 16 * 3 * 3 * 3 * 2],
                    shape: vec![16, 3, 3, 3],
                    dtype: ScalarType::Float16,
                },
            )
            .with_output("weight_out");
        weight_const.output_types = vec![Some(weight_type)];

        let conv = Operation::new("conv", "conv_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference("weight_out".into()))
            .with_output("conv_out");

        func.body.add_op(weight_const);
        func.body.add_op(conv);
        func.body.outputs.push("conv_out".into());
        program.add_function(func);
        program
    }

    let small = make_conv_program(64, 64);
    let large = make_conv_program(128, 128);

    let report_small = validate_ane_compatibility(&small);
    let report_large = validate_ane_compatibility(&large);

    assert!(
        report_small.total_estimated_flops > 0,
        "small program should have non-zero flops"
    );
    assert!(
        report_large.total_estimated_flops > 0,
        "large program should have non-zero flops"
    );

    // 128×128 is 4× the spatial area of 64×64 → ~4× FLOPs.
    let ratio =
        report_large.total_estimated_flops as f64 / report_small.total_estimated_flops as f64;
    assert!(
        (2.0..=6.0).contains(&ratio),
        "flops ratio should be approximately 4×, got {ratio:.2}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 5.3: Mixed-Precision
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn mixed_precision_pipeline_integration() {
    let input_ty = TensorType::new(ScalarType::Float32, vec![1, 32, 64]);
    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;

    // 2 attention-named consts (should become FP16)
    block.add_op(common::make_const_op(
        "attention.q_weight",
        &[64, 64],
        ScalarType::Float32,
    ));
    block.add_op(common::make_const_op(
        "attention.k_weight",
        &[64, 64],
        ScalarType::Float32,
    ));

    // 2 FFN-named consts (should become INT8 / constexpr_affine_dequantize)
    block.add_op(common::make_const_op(
        "ffn.fc1_weight",
        &[64, 64],
        ScalarType::Float32,
    ));
    block.add_op(common::make_const_op(
        "ffn.fc2_weight",
        &[64, 64],
        ScalarType::Float32,
    ));

    // Wire all consts through linears so they're not dead code.
    block.add_op(
        Operation::new("linear", "attention.q_proj")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference("attention.q_weight_out".into()))
            .with_output("q_out"),
    );
    block.add_op(
        Operation::new("linear", "attention.k_proj")
            .with_input("x", Value::Reference("q_out".into()))
            .with_input("weight", Value::Reference("attention.k_weight_out".into()))
            .with_output("k_out"),
    );
    block.add_op(
        Operation::new("linear", "ffn.fc1")
            .with_input("x", Value::Reference("k_out".into()))
            .with_input("weight", Value::Reference("ffn.fc1_weight_out".into()))
            .with_output("fc1_out"),
    );
    block.add_op(
        Operation::new("linear", "ffn.fc2")
            .with_input("x", Value::Reference("fc1_out".into()))
            .with_input("weight", Value::Reference("ffn.fc2_weight_out".into()))
            .with_output("final_out"),
    );
    block.outputs.push("final_out".into());

    // Run ONLY the mixed-precision pass (not the full pipeline with DCE/fusion).
    let pass = MixedPrecisionPass::preset_fp16_int8();
    pass.run(&mut program)
        .expect("mixed precision should succeed");

    let ops = &program.functions["main"].body.operations;

    // Attention consts should still be "const" with FP16 dtype.
    for name in &["attention.q_weight", "attention.k_weight"] {
        let op = ops.iter().find(|o| o.name == *name).unwrap_or_else(|| {
            panic!("should find op '{name}'");
        });
        assert_eq!(op.op_type, "const", "{name} should still be const");
        let val = op
            .inputs
            .get("val")
            .or_else(|| op.attributes.get("val"))
            .expect("should have val");
        match val {
            Value::Tensor { dtype, .. } => {
                assert_eq!(*dtype, ScalarType::Float16, "{name} should be FP16");
            }
            _ => panic!("{name} val should be a Tensor"),
        }
    }

    // FFN consts should become constexpr_affine_dequantize.
    for name in &["ffn.fc1_weight", "ffn.fc2_weight"] {
        let op = ops.iter().find(|o| o.name == *name).unwrap_or_else(|| {
            panic!("should find op '{name}'");
        });
        assert_eq!(
            op.op_type, "constexpr_affine_dequantize",
            "{name} should be INT8 quantized"
        );
        assert!(
            op.attributes.contains_key("quantized_data"),
            "{name} should have quantized_data"
        );
        assert!(
            op.attributes.contains_key("scale"),
            "{name} should have scale"
        );
    }
}

#[test]
fn mixed_precision_config_from_toml_file() {
    let toml_content = r#"
[rules]
"*attention*" = "fp16"
"*ffn*" = "int8"
default = "none"
"#;

    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("mixed_precision.toml");
    std::fs::write(&config_path, toml_content).expect("write toml");

    let config = MixedPrecisionConfig::from_toml_file(&config_path).expect("load config");

    // Build a program and apply the pass directly (no DCE pipeline).
    let input_ty = TensorType::new(ScalarType::Float32, vec![1, 32, 64]);
    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(common::make_const_op(
        "attention.weight",
        &[64, 64],
        ScalarType::Float32,
    ));
    block.add_op(common::make_const_op(
        "ffn.weight",
        &[64, 64],
        ScalarType::Float32,
    ));
    // Wire consts through linears so they're used.
    block.add_op(
        Operation::new("linear", "attention.proj")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference("attention.weight_out".into()))
            .with_output("attn_out"),
    );
    block.add_op(
        Operation::new("linear", "ffn.proj")
            .with_input("x", Value::Reference("attn_out".into()))
            .with_input("weight", Value::Reference("ffn.weight_out".into()))
            .with_output("final_out"),
    );
    block.outputs.push("final_out".into());

    let pass = MixedPrecisionPass::new(config);
    pass.run(&mut program).expect("mixed precision pass");

    let ops = &program.functions["main"].body.operations;

    let attn_op = ops
        .iter()
        .find(|o| o.name == "attention.weight")
        .expect("attention.weight");
    assert_eq!(
        attn_op.op_type, "const",
        "attention weight should be const (FP16)"
    );
    let val = attn_op
        .inputs
        .get("val")
        .or_else(|| attn_op.attributes.get("val"))
        .expect("val");
    match val {
        Value::Tensor { dtype, .. } => assert_eq!(*dtype, ScalarType::Float16),
        _ => panic!("expected Tensor"),
    }

    let ffn_op = ops
        .iter()
        .find(|o| o.name == "ffn.weight")
        .expect("ffn.weight");
    assert_eq!(ffn_op.op_type, "constexpr_affine_dequantize");
}

#[test]
fn mixed_precision_with_pipeline_method() {
    let toml_content = r#"
[rules]
"*attention*" = "fp16"
"*ffn*" = "int8"
default = "fp16"
"#;

    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = dir.path().join("mp_pipeline.toml");
    std::fs::write(&config_path, toml_content).expect("write toml");

    let input_ty = TensorType::new(ScalarType::Float32, vec![1, 32, 64]);
    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(common::make_const_op(
        "attention.w",
        &[64, 64],
        ScalarType::Float32,
    ));
    block.add_op(common::make_const_op(
        "ffn.w",
        &[64, 64],
        ScalarType::Float32,
    ));
    block.add_op(
        Operation::new("linear", "attention.proj")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference("attention.w_out".into()))
            .with_output("attn_out"),
    );
    block.add_op(
        Operation::new("linear", "ffn.proj")
            .with_input("x", Value::Reference("attn_out".into()))
            .with_input("weight", Value::Reference("ffn.w_out".into()))
            .with_output("final_out"),
    );
    block.outputs.push("final_out".into());

    // Use PassPipeline::with_mixed_precision (file-based pipeline method).
    let pipeline = PassPipeline::new()
        .with_mixed_precision(&config_path)
        .expect("with_mixed_precision");
    pipeline.run(&mut program).expect("pipeline");

    let ops = &program.functions["main"].body.operations;

    // Attention const should be FP16.
    let attn_op = ops.iter().find(|o| o.name == "attention.w");
    assert!(attn_op.is_some(), "attention const should still exist");
    if let Some(op) = attn_op {
        assert_eq!(op.op_type, "const");
        let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
        if let Some(Value::Tensor { dtype, .. }) = val {
            assert_eq!(*dtype, ScalarType::Float16, "attention should be FP16");
        }
    }

    // FFN const should be INT8 quantized.
    let ffn_op = ops.iter().find(|o| o.name == "ffn.w");
    assert!(ffn_op.is_some(), "ffn const should still exist");
    if let Some(op) = ffn_op {
        assert_eq!(op.op_type, "constexpr_affine_dequantize");
    }
}

#[test]
fn per_channel_int8_produces_correct_shapes() {
    let mut program = Program::new("1");
    let input_ty = TensorType::new(ScalarType::Float32, vec![1]);
    let func = Function::new("main").with_input("dummy", input_ty);
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(common::make_const_op(
        "weight",
        &[64, 128],
        ScalarType::Float32,
    ));
    block.add_op(
        Operation::new("identity", "out_id")
            .with_input("x", Value::Reference("weight_out".into()))
            .with_output("final_out"),
    );
    block.outputs.push("final_out".into());

    Int8QuantizePass::new(None, Granularity::PerChannel)
        .run(&mut program)
        .unwrap();

    let op = program.functions["main"]
        .body
        .operations
        .iter()
        .find(|o| o.name == "weight")
        .expect("weight op");
    assert_eq!(op.op_type, "constexpr_affine_dequantize");

    // Scale should be a tensor with shape [64] (per output channel).
    match op.attributes.get("scale").expect("scale attr") {
        Value::Tensor { shape, .. } => {
            assert_eq!(shape, &[64], "scale shape should be [64] for per-channel");
        }
        _ => panic!("scale should be a Tensor for per-channel quantization"),
    }

    // Zero point should also be [64].
    match op.attributes.get("zero_point").expect("zero_point attr") {
        Value::Tensor { shape, .. } => {
            assert_eq!(shape, &[64], "zero_point shape should be [64]");
        }
        _ => panic!("zero_point should be a Tensor for per-channel"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5.4: Fusion Patterns
// ═══════════════════════════════════════════════════════════════════════

/// Build a program with a realistic transformer block for fusion testing.
///
/// Structure:
///   FFN path:   layer_norm → linear (fuse LN+Linear)
///               → gelu → linear (fuse GELU+Linear)
///               → add(result, input) (residual add fusion)
///   Attention:  expand(K) → transpose → matmul(Q, K^T) → real_div
///               → softmax → matmul(attn, V) (GQA fusion)
///   → output
fn build_fusible_transformer_block() -> (Program, usize) {
    let input_ty = TensorType::new(ScalarType::Float32, vec![1, 32, 64]);
    let func = Function::new("main").with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;

    // --- FFN path (LayerNorm+Linear, GELU+Linear, residual add) ---
    block.add_op(
        Operation::new("layer_norm", "ffn.layer_norm")
            .with_input("x", Value::Reference("input".into()))
            .with_output("ln_out"),
    );
    block.add_op(common::make_const_op(
        "ffn.w1",
        &[64, 64],
        ScalarType::Float32,
    ));
    block.add_op(
        Operation::new("linear", "ffn.linear1")
            .with_input("x", Value::Reference("ln_out".into()))
            .with_input("weight", Value::Reference("ffn.w1_out".into()))
            .with_output("ffn1_out"),
    );
    block.add_op(
        Operation::new("gelu", "ffn.gelu")
            .with_input("x", Value::Reference("ffn1_out".into()))
            .with_output("gelu_out"),
    );
    block.add_op(common::make_const_op(
        "ffn.w2",
        &[64, 64],
        ScalarType::Float32,
    ));
    block.add_op(
        Operation::new("linear", "ffn.linear2")
            .with_input("x", Value::Reference("gelu_out".into()))
            .with_input("weight", Value::Reference("ffn.w2_out".into()))
            .with_output("ffn2_out"),
    );
    block.add_op(
        Operation::new("add", "ffn.residual_add")
            .with_input("x", Value::Reference("ffn2_out".into()))
            .with_input("y", Value::Reference("input".into()))
            .with_output("residual_out"),
    );

    // --- GQA Attention path ---
    // Q projection (using residual_out as Q)
    // K, V base values
    block.add_op(common::make_const_op(
        "k_base",
        &[1, 8, 32, 8],
        ScalarType::Float32,
    ));
    block.add_op(common::make_const_op(
        "v_base",
        &[1, 8, 32, 8],
        ScalarType::Float32,
    ));

    // K broadcast (expand) — signals GQA
    block.add_op(
        Operation::new("expand", "k_expand")
            .with_input("x", Value::Reference("k_base_out".into()))
            .with_output("k_expanded"),
    );
    // Transpose K
    block.add_op(
        Operation::new("transpose", "k_transpose")
            .with_input("x", Value::Reference("k_expanded".into()))
            .with_output("k_t"),
    );
    // Scores matmul: Q × K^T
    block.add_op(
        Operation::new("matmul", "attn.scores_matmul")
            .with_input("x", Value::Reference("residual_out".into()))
            .with_input("y", Value::Reference("k_t".into()))
            .with_output("scores"),
    );
    // Scale
    block.add_op(
        Operation::new("real_div", "attn.scale")
            .with_input("x", Value::Reference("scores".into()))
            .with_input("y", Value::Float(8.0))
            .with_output("scaled_scores"),
    );
    // Softmax
    block.add_op(
        Operation::new("softmax", "attn.softmax")
            .with_input("x", Value::Reference("scaled_scores".into()))
            .with_attr("axis", Value::Int(-1))
            .with_output("attn_weights"),
    );
    // V broadcast (expand)
    block.add_op(
        Operation::new("expand", "v_expand")
            .with_input("x", Value::Reference("v_base_out".into()))
            .with_output("v_expanded"),
    );
    // Output matmul: attn × V
    block.add_op(
        Operation::new("matmul", "attn.output_matmul")
            .with_input("x", Value::Reference("attn_weights".into()))
            .with_input("y", Value::Reference("v_expanded".into()))
            .with_output("attn_out"),
    );

    block.outputs.push("attn_out".into());

    let initial_count = common::count_ops(&program);
    (program, initial_count)
}

#[test]
fn full_transformer_block_fusion() {
    let (mut program, initial_count) = build_fusible_transformer_block();

    // Run fusion passes individually to test fusion behavior precisely
    // (the full pipeline includes layout/substitution passes that add ops).
    use mil_rs::ir::passes::{
        GeluLinearFusionPass, GqaFusionPass, LayerNormLinearFusionPass, ResidualAddFusionPass,
    };

    // GQA first (before attention fusion would consume the pattern).
    GqaFusionPass.run(&mut program).unwrap();
    LayerNormLinearFusionPass.run(&mut program).unwrap();
    GeluLinearFusionPass.run(&mut program).unwrap();
    ResidualAddFusionPass.run(&mut program).unwrap();

    let ops = &program.functions["main"].body.operations;

    // LayerNorm+Linear fusion: layer_norm should have has_fused_linear.
    let ln_op = ops.iter().find(|o| o.op_type == "layer_norm");
    if let Some(ln) = ln_op {
        assert_eq!(
            ln.attributes.get("has_fused_linear"),
            Some(&Value::Bool(true)),
            "layer_norm should have fused linear"
        );
    }

    // GELU+Linear fusion: gelu should have has_fused_linear.
    let gelu_op = ops.iter().find(|o| o.op_type == "gelu");
    if let Some(g) = gelu_op {
        assert_eq!(
            g.attributes.get("has_fused_linear"),
            Some(&Value::Bool(true)),
            "gelu should have fused linear"
        );
    }

    // Residual add: the add should be tagged with is_residual attribute.
    let residual_ops: Vec<_> = ops
        .iter()
        .filter(|o| {
            o.op_type == "add" && o.attributes.get("is_residual") == Some(&Value::Bool(true))
        })
        .collect();
    assert!(
        !residual_ops.is_empty(),
        "should have at least one add op tagged with is_residual"
    );

    // GQA fusion: should produce a grouped_query_attention op.
    let gqa_ops: Vec<_> = ops
        .iter()
        .filter(|o| o.op_type == "grouped_query_attention")
        .collect();
    assert!(
        !gqa_ops.is_empty(),
        "should have at least one grouped_query_attention op"
    );

    // Op count should decrease after fusions.
    let final_count = common::count_ops(&program);
    assert!(
        final_count < initial_count,
        "fusion should reduce op count: initial={initial_count}, final={final_count}"
    );
}

#[test]
fn fusion_order_independence() {
    // Build a program with independent fusion targets: conv+bn pair and gelu+linear pair.
    fn build_independent_fusion_program() -> Program {
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 8, 8]);
        let func = Function::new("main").with_input("input", input_ty);
        let mut program = Program::new("1");
        program.add_function(func);

        let block = &mut program.functions.get_mut("main").unwrap().body;

        // Conv + BatchNorm pair (independent)
        block.add_op(common::make_const_op(
            "conv_w",
            &[16, 3, 3, 3],
            ScalarType::Float32,
        ));
        block.add_op(
            Operation::new("conv", "conv_0")
                .with_input("x", Value::Reference("input".into()))
                .with_input("weight", Value::Reference("conv_w_out".into()))
                .with_output("conv_out"),
        );
        block.add_op(common::make_const_op("bn_mean", &[16], ScalarType::Float32));
        block.add_op(common::make_const_op("bn_var", &[16], ScalarType::Float32));
        block.add_op(common::make_const_op(
            "bn_gamma",
            &[16],
            ScalarType::Float32,
        ));
        block.add_op(common::make_const_op("bn_beta", &[16], ScalarType::Float32));
        block.add_op(
            Operation::new("batch_norm", "bn_0")
                .with_input("x", Value::Reference("conv_out".into()))
                .with_input("mean", Value::Reference("bn_mean_out".into()))
                .with_input("variance", Value::Reference("bn_var_out".into()))
                .with_input("gamma", Value::Reference("bn_gamma_out".into()))
                .with_input("beta", Value::Reference("bn_beta_out".into()))
                .with_output("bn_out"),
        );

        // GELU + Linear pair (independent — uses a separate input path)
        block.add_op(
            Operation::new("gelu", "gelu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("gelu_out"),
        );
        block.add_op(common::make_const_op(
            "lin_w",
            &[16, 3],
            ScalarType::Float32,
        ));
        block.add_op(
            Operation::new("linear", "lin_0")
                .with_input("x", Value::Reference("gelu_out".into()))
                .with_input("weight", Value::Reference("lin_w_out".into()))
                .with_output("lin_out"),
        );

        // Combine both outputs
        block.add_op(
            Operation::new("add", "combine")
                .with_input("x", Value::Reference("bn_out".into()))
                .with_input("y", Value::Reference("lin_out".into()))
                .with_output("final_out"),
        );
        block.outputs.push("final_out".into());
        program
    }

    // Order 1: ConvBN first, then GELU+Linear
    let mut prog1 = build_independent_fusion_program();
    use mil_rs::ir::passes::{ConvBatchNormFusionPass, GeluLinearFusionPass};
    ConvBatchNormFusionPass.run(&mut prog1).unwrap();
    GeluLinearFusionPass.run(&mut prog1).unwrap();

    // Order 2: GELU+Linear first, then ConvBN
    let mut prog2 = build_independent_fusion_program();
    GeluLinearFusionPass.run(&mut prog2).unwrap();
    ConvBatchNormFusionPass.run(&mut prog2).unwrap();

    let count1 = common::count_ops(&prog1);
    let count2 = common::count_ops(&prog2);
    assert_eq!(
        count1, count2,
        "fusion order should not affect final op count: order1={count1}, order2={count2}"
    );

    let types1: Vec<String> = prog1.functions["main"]
        .body
        .operations
        .iter()
        .map(|o| o.op_type.clone())
        .collect();
    let types2: Vec<String> = prog2.functions["main"]
        .body
        .operations
        .iter()
        .map(|o| o.op_type.clone())
        .collect();
    // Sort to compare regardless of insertion order.
    let mut sorted1 = types1.clone();
    sorted1.sort();
    let mut sorted2 = types2.clone();
    sorted2.sort();
    assert_eq!(
        sorted1, sorted2,
        "fusion order should produce same op types"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 5.5: Compute Unit Annotations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn compute_unit_annotation_roundtrip() {
    let mut program = Program::new("1");
    let mut func = Function::new("main");
    let input_type = TensorType::new(ScalarType::Float16, vec![1, 32, 64, 64]);
    func.inputs.push(("input".to_string(), input_type));

    // Relu (ANE-eligible)
    let relu = Operation::new("relu", "relu_0")
        .with_input("x", Value::Reference("input".into()))
        .with_output("relu_out");

    // Custom/unsupported op (should get CPU)
    let custom = Operation::new("custom_unknown_op", "custom_0")
        .with_input("x", Value::Reference("relu_out".into()))
        .with_output("custom_out");

    func.body.add_op(relu);
    func.body.add_op(custom);
    func.body.outputs.push("custom_out".into());
    program.add_function(func);

    // Run compute unit annotation.
    ComputeUnitAnnotationPass.run(&mut program).unwrap();

    // Verify annotations.
    let ops = &program.functions["main"].body.operations;
    let relu_cu = ops
        .iter()
        .find(|o| o.op_type == "relu")
        .unwrap()
        .compute_unit;
    let custom_cu = ops
        .iter()
        .find(|o| o.op_type == "custom_unknown_op")
        .unwrap()
        .compute_unit;
    assert_eq!(relu_cu, Some(ComputeUnit::Ane));
    assert_eq!(custom_cu, Some(ComputeUnit::Cpu));

    // Serialize to proto and verify compute_unit attributes are present.
    let model = program_to_model(&program, 8).expect("program_to_model");

    use prost::Message;
    let bytes = model.encode_to_vec();
    let decoded =
        mil_rs::proto::specification::Model::decode(bytes.as_slice()).expect("should decode");

    let proto_program = match &decoded.r#type {
        Some(mil_rs::proto::specification::model::Type::MlProgram(p)) => p,
        _ => panic!("expected MlProgram"),
    };
    let proto_func = proto_program.functions.values().next().expect("function");
    let proto_block = proto_func
        .block_specializations
        .values()
        .next()
        .expect("block");

    // Find relu in proto — should have compute_unit attribute set to "ane".
    let proto_relu = proto_block
        .operations
        .iter()
        .find(|op| op.r#type == "relu")
        .expect("relu in proto");
    assert!(
        proto_relu.attributes.contains_key("compute_unit"),
        "relu should have compute_unit attribute in proto"
    );

    // Find custom op — should have compute_unit attribute set to "cpu".
    let proto_custom = proto_block
        .operations
        .iter()
        .find(|op| op.r#type == "custom_unknown_op")
        .expect("custom op in proto");
    assert!(
        proto_custom.attributes.contains_key("compute_unit"),
        "custom op should have compute_unit attribute in proto"
    );

    // Verify roundtrip preserves the attribute (even if not in the dedicated field).
    let roundtripped = model_to_program(&model).expect("model_to_program");
    let rt_ops = &roundtripped.functions["main"].body.operations;

    let rt_relu = rt_ops
        .iter()
        .find(|o| o.op_type == "relu")
        .expect("relu after roundtrip");
    assert!(
        rt_relu.attributes.contains_key("compute_unit")
            || rt_relu.compute_unit == Some(ComputeUnit::Ane),
        "relu compute_unit should survive roundtrip (as attribute or field)"
    );

    let rt_custom = rt_ops
        .iter()
        .find(|o| o.op_type == "custom_unknown_op")
        .expect("custom op after roundtrip");
    assert!(
        rt_custom.attributes.contains_key("compute_unit")
            || rt_custom.compute_unit == Some(ComputeUnit::Cpu),
        "custom op compute_unit should survive roundtrip (as attribute or field)"
    );
}

#[test]
fn annotations_match_validation() {
    let mut program = Program::new("1");
    let mut func = Function::new("main");
    let input_type = TensorType::new(ScalarType::Float16, vec![1, 32, 64, 64]);
    func.inputs.push(("input".to_string(), input_type));

    // ANE-eligible ops.
    func.body.add_op(
        Operation::new("relu", "relu_0")
            .with_input("x", Value::Reference("input".into()))
            .with_output("relu_out"),
    );

    // Non-ANE op.
    func.body.add_op(
        Operation::new("custom_op", "custom_0")
            .with_input("x", Value::Reference("relu_out".into()))
            .with_output("custom_out"),
    );

    func.body.outputs.push("custom_out".into());
    program.add_function(func);

    // Run validation.
    let report = validate_ane_compatibility(&program);

    // Run compute unit annotation on a clone.
    let mut annotated = program.clone();
    ComputeUnitAnnotationPass.run(&mut annotated).unwrap();

    // For every non-const op, check consistency.
    for op in &annotated.functions["main"].body.operations {
        if op.op_type == "const" {
            continue;
        }

        let validation_eligible = report
            .ane_compatible
            .iter()
            .chain(report.fallback_ops.iter())
            .find(|r| r.name == op.name)
            .map(|r| r.ane_eligible)
            .unwrap_or(false);

        let annotation_ane = op.compute_unit == Some(ComputeUnit::Ane);

        assert_eq!(
            annotation_ane, validation_eligible,
            "op '{}' ({}) annotation={:?} but validation ane_eligible={}",
            op.name, op.op_type, op.compute_unit, validation_eligible
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 7: Error-Path Tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn mixed_precision_invalid_toml_returns_error() {
    // Malformed: rules should be a table of key-value pairs, not this.
    let bad_toml = r#"
[rules]
this is not valid toml at all !!!
"#;

    let result = MixedPrecisionConfig::from_toml_str(bad_toml);
    assert!(
        result.is_err(),
        "malformed TOML should return Err, got: {:?}",
        result
    );
}

#[test]
fn op_splitting_budget_smaller_than_one_tile() {
    let mut program = Program::new("1");
    let input_ty = TensorType::new(ScalarType::Float32, vec![64, 64]);
    let mut func = Function::new("main").with_input("x", input_ty.clone());
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;

    // matmul [64, 64] × [64, 64]
    block.add_op(common::make_const_op(
        "weight",
        &[64, 64],
        ScalarType::Float32,
    ));
    block.add_op(
        Operation::new("matmul", "matmul_0")
            .with_input("x", Value::Reference("x".into()))
            .with_input("y", Value::Reference("weight_out".into()))
            .with_output("mm_out"),
    );
    block.outputs.push("mm_out".into());

    // Budget of 1 byte — impossibly small.
    let result = OpSplittingPass::new(1).run(&mut program);
    assert!(
        result.is_ok(),
        "op splitting with tiny budget should not panic: {:?}",
        result
    );

    // Program should still be valid (has outputs, has ops).
    assert!(
        !program.functions["main"].body.operations.is_empty(),
        "program should still have operations"
    );
    assert!(
        !program.functions["main"].body.outputs.is_empty(),
        "program should still have outputs"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 8: Quantization Numerical Accuracy
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn fp16_roundtrip_error_bounded() {
    // Known FP32 values in range -10.0 to 10.0.
    let n = 256;
    let original_values: Vec<f32> = (0..n)
        .map(|i| (i as f32 / n as f32) * 20.0 - 10.0)
        .collect();

    let mut program = common::build_const_program(&[n], ScalarType::Float32);

    // Replace the const data with our known values.
    let block = &mut program.functions.get_mut("main").unwrap().body;
    for op in &mut block.operations {
        if op.op_type == "const" && op.name == "weight" {
            if let Some(val) = op.attributes.get_mut("val") {
                if let Value::Tensor { data, .. } = val {
                    *data = f32_slice_to_bytes(&original_values);
                }
            }
        }
    }

    // Run FP16 quantization.
    Fp16QuantizePass.run(&mut program).unwrap();

    // Extract FP16 data and convert back to FP32.
    let op = program.functions["main"]
        .body
        .operations
        .iter()
        .find(|o| o.name == "weight")
        .expect("weight op");
    let val = op
        .inputs
        .get("val")
        .or_else(|| op.attributes.get("val"))
        .expect("val");
    match val {
        Value::Tensor { data, dtype, .. } => {
            assert_eq!(*dtype, ScalarType::Float16);
            let roundtripped: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            assert_eq!(roundtripped.len(), original_values.len());

            let max_error = original_values
                .iter()
                .zip(roundtripped.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);

            assert!(
                max_error < 1e-2,
                "FP16 roundtrip max error should be < 1e-2 for range [-10, 10], got {max_error}"
            );
        }
        _ => panic!("expected Tensor"),
    }
}

#[test]
fn int8_roundtrip_error_bounded() {
    let n = 256;
    let original_values: Vec<f32> = (0..n)
        .map(|i| (i as f32 / n as f32) * 6.0 - 3.0) // range [-3, 3]
        .collect();

    let mut program = common::build_const_program(&[n], ScalarType::Float32);
    let block = &mut program.functions.get_mut("main").unwrap().body;
    for op in &mut block.operations {
        if op.op_type == "const" && op.name == "weight" {
            if let Some(val) = op.attributes.get_mut("val") {
                if let Value::Tensor { data, .. } = val {
                    *data = f32_slice_to_bytes(&original_values);
                }
            }
        }
    }

    Int8QuantizePass::weight_only().run(&mut program).unwrap();

    let op = program.functions["main"]
        .body
        .operations
        .iter()
        .find(|o| o.name == "weight")
        .expect("weight op");
    assert_eq!(op.op_type, "constexpr_affine_dequantize");

    // Extract scale and zero_point.
    let scale = match op.attributes.get("scale").expect("scale") {
        Value::Float(f) => *f as f32,
        _ => panic!("scale should be Float for per-tensor"),
    };
    let zero_point = match op.attributes.get("zero_point").expect("zero_point") {
        Value::Float(f) => *f as f32,
        _ => panic!("zero_point should be Float for per-tensor"),
    };

    // Extract quantized data.
    let quantized = match op.attributes.get("quantized_data").expect("quantized_data") {
        Value::Tensor { data, .. } => data,
        _ => panic!("quantized_data should be Tensor"),
    };

    // Dequantize: x_approx = (q - zero_point) * scale
    let dequantized: Vec<f32> = quantized
        .iter()
        .map(|&q| (q as f32 - zero_point) * scale)
        .collect();
    assert_eq!(dequantized.len(), original_values.len());

    let max_abs = original_values
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    let max_error = original_values
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    let bound = max_abs / 127.0 + 0.01;
    assert!(
        max_error < bound,
        "INT8 max error should be < {bound}, got {max_error}"
    );
}

#[test]
fn int8_quantize_all_zeros_tensor() {
    let n = 64 * 64;
    let zeros: Vec<f32> = vec![0.0; n];

    let mut program = Program::new("1");
    let input_ty = TensorType::new(ScalarType::Float32, vec![1]);
    let func = Function::new("main").with_input("dummy", input_ty);
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(common::make_const_op_with_values(
        "zeros_weight",
        &[64, 64],
        &zeros,
    ));
    block.add_op(
        Operation::new("identity", "out_id")
            .with_input("x", Value::Reference("zeros_weight_out".into()))
            .with_output("final_out"),
    );
    block.outputs.push("final_out".into());

    Int8QuantizePass::weight_only().run(&mut program).unwrap();

    let op = program.functions["main"]
        .body
        .operations
        .iter()
        .find(|o| o.name == "zeros_weight")
        .expect("zeros_weight");
    assert_eq!(op.op_type, "constexpr_affine_dequantize");

    // Scale should be near-zero or exactly 1.0 (degenerate case).
    let scale = match op.attributes.get("scale").expect("scale") {
        Value::Float(f) => *f as f32,
        _ => panic!("expected Float scale"),
    };
    // For all-zeros, min == max == 0.0, so scale is the degenerate-case value.
    // The degenerate case in the code sets scale = 1.0.
    // Check: either scale is very small or it's the degenerate 1.0.
    assert!(
        scale.abs() <= 1.0 + f32::EPSILON,
        "scale for all-zeros should be <= 1.0, got {scale}"
    );

    // Dequantized output should be all zeros (or very close).
    let zero_point = match op.attributes.get("zero_point").expect("zp") {
        Value::Float(f) => *f as f32,
        _ => panic!("expected Float zp"),
    };
    let quantized = match op.attributes.get("quantized_data").expect("qd") {
        Value::Tensor { data, .. } => data,
        _ => panic!("expected Tensor qd"),
    };
    let dequantized: Vec<f32> = quantized
        .iter()
        .map(|&q| (q as f32 - zero_point) * scale)
        .collect();

    let max_error = dequantized.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_error < 1e-5,
        "dequantized all-zeros should be near zero, max_error={max_error}"
    );
}

#[test]
fn int8_quantize_uniform_value_tensor() {
    let n = 64 * 64;
    let uniform_val = 3.14_f32;
    let values: Vec<f32> = vec![uniform_val; n];

    let mut program = Program::new("1");
    let input_ty = TensorType::new(ScalarType::Float32, vec![1]);
    let func = Function::new("main").with_input("dummy", input_ty);
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(common::make_const_op_with_values(
        "uniform_weight",
        &[64, 64],
        &values,
    ));
    block.add_op(
        Operation::new("identity", "out_id")
            .with_input("x", Value::Reference("uniform_weight_out".into()))
            .with_output("final_out"),
    );
    block.outputs.push("final_out".into());

    Int8QuantizePass::weight_only().run(&mut program).unwrap();

    let op = program.functions["main"]
        .body
        .operations
        .iter()
        .find(|o| o.name == "uniform_weight")
        .expect("uniform_weight");
    assert_eq!(op.op_type, "constexpr_affine_dequantize");

    let scale = match op.attributes.get("scale").expect("scale") {
        Value::Float(f) => *f as f32,
        _ => panic!("expected Float scale"),
    };
    let zero_point = match op.attributes.get("zero_point").expect("zp") {
        Value::Float(f) => *f as f32,
        _ => panic!("expected Float zp"),
    };
    let quantized = match op.attributes.get("quantized_data").expect("qd") {
        Value::Tensor { data, .. } => data,
        _ => panic!("expected Tensor qd"),
    };

    let dequantized: Vec<f32> = quantized
        .iter()
        .map(|&q| (q as f32 - zero_point) * scale)
        .collect();

    // All dequantized values should be close to 3.14.
    // The degenerate case (all same values) uses scale=1.0, so the max rounding
    // error can be up to 0.5.
    for (i, &v) in dequantized.iter().enumerate() {
        assert!(
            (v - uniform_val).abs() < 0.5,
            "dequantized[{i}] = {v}, expected ≈ {uniform_val}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PolarQuant serialization round-trip
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn polar_quant_serialization_round_trip() {
    // Build a program with a const op large enough for PolarQuant (>= 1024 elements).
    let rows = 8;
    let cols = 128;
    let num_elements = rows * cols; // 1024
    let tensor_data: Vec<f32> = (0..num_elements).map(|i| (i as f32 * 0.01).sin()).collect();
    let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let input_ty = TensorType::new(ScalarType::Float32, vec![1, cols]);
    let mut program = Program::new("1");
    let func = Function::new("main").with_input("input", input_ty);
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(
        Operation::new("const", "weight_0")
            .with_input(
                "val",
                Value::Tensor {
                    data: tensor_bytes,
                    shape: vec![rows, cols],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output("weight_0_out"),
    );
    block.add_op(
        Operation::new("linear", "linear_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input("weight", Value::Reference("weight_0_out".into()))
            .with_output("linear_out"),
    );
    block.outputs.push("linear_out".into());

    // Run PolarQuant pass.
    let pass = PolarQuantPass::new(4);
    pass.run(&mut program).unwrap();

    // Verify the ops were transformed correctly before serialization.
    let ops = &program.functions["main"].body.operations;
    assert_eq!(ops[0].op_type, "constexpr_lut_to_dense");
    assert_eq!(ops[1].op_type, "const"); // norms
    assert_eq!(ops[2].op_type, "mul");
    // linear should reference the mul output now
    assert_eq!(ops[3].op_type, "linear");

    // Serialize to protobuf.
    let model = program_to_model(&program, 7).expect("serialization should succeed");

    // Deserialize back.
    let deserialized = model_to_program(&model).expect("deserialization should succeed");

    // Verify the deserialized program has the same op structure.
    let rt_ops = &deserialized.functions["main"].body.operations;
    assert_eq!(
        rt_ops.len(),
        ops.len(),
        "round-trip should preserve op count"
    );
    assert_eq!(rt_ops[0].op_type, "constexpr_lut_to_dense");
    assert_eq!(rt_ops[1].op_type, "const");
    assert_eq!(rt_ops[2].op_type, "mul");
    assert_eq!(rt_ops[3].op_type, "linear");

    // Verify the mul op still references the correct inputs.
    match rt_ops[2].inputs.get("x") {
        Some(Value::Reference(name)) => assert_eq!(name, "weight_0_out"),
        other => panic!("expected mul x to reference weight_0_out, got {other:?}"),
    }
    match rt_ops[2].inputs.get("y") {
        Some(Value::Reference(name)) => assert_eq!(name, "weight_0_out_polar_norms"),
        other => panic!("expected mul y to reference norms, got {other:?}"),
    }

    // Verify LUT attributes survived serialization.
    assert!(
        rt_ops[0].attributes.contains_key("lut"),
        "constexpr_lut_to_dense should have lut attribute after round-trip"
    );
    assert!(
        rt_ops[0].attributes.contains_key("indices"),
        "constexpr_lut_to_dense should have indices attribute after round-trip"
    );
    assert!(
        rt_ops[0].attributes.contains_key("shape"),
        "constexpr_lut_to_dense should have shape attribute after round-trip"
    );
}

#[test]
fn polar_quant_lut_shape_preserved_in_proto() {
    let rows = 16;
    let cols = 64;
    let num_elements = rows * cols; // 1024
    let tensor_data: Vec<f32> = (0..num_elements).map(|i| (i as f32 * 0.05).cos()).collect();
    let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let input_ty = TensorType::new(ScalarType::Float32, vec![1, cols]);
    let mut program = Program::new("1");
    let func = Function::new("main").with_input("input", input_ty);
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(
        Operation::new("const", "weight_0")
            .with_input(
                "val",
                Value::Tensor {
                    data: tensor_bytes,
                    shape: vec![rows, cols],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output("weight_0_out"),
    );
    block.outputs.push("weight_0_out".into());

    let n_bits: u8 = 4;
    let pass = PolarQuantPass::new(n_bits);
    pass.run(&mut program).unwrap();

    // Check LUT and indices shapes before serialization.
    let ops = &program.functions["main"].body.operations;
    let lut_shape_before = match ops[0].attributes.get("lut") {
        Some(Value::Tensor { shape, .. }) => shape.clone(),
        other => panic!("expected LUT tensor, got {other:?}"),
    };
    let indices_shape_before = match ops[0].attributes.get("indices") {
        Some(Value::Tensor { shape, .. }) => shape.clone(),
        other => panic!("expected indices tensor, got {other:?}"),
    };
    let orig_shape_before = match ops[0].attributes.get("shape") {
        Some(Value::Tensor { shape, .. }) => shape.clone(),
        other => panic!("expected shape tensor, got {other:?}"),
    };

    // LUT should have 2^n_bits = 16 entries.
    assert_eq!(lut_shape_before, vec![1 << n_bits]);

    // Original shape tensor records the rank.
    assert_eq!(orig_shape_before, vec![2]); // rank-2: [rows, cols]

    // Serialize and deserialize.
    let model = program_to_model(&program, 7).expect("serialization should succeed");
    let deserialized = model_to_program(&model).expect("deserialization should succeed");

    let rt_ops = &deserialized.functions["main"].body.operations;
    assert_eq!(rt_ops[0].op_type, "constexpr_lut_to_dense");

    // Verify LUT is preserved (may come back as Tensor or List depending on dtype).
    let lut_len_after = match rt_ops[0].attributes.get("lut") {
        Some(Value::Tensor { shape, .. }) => shape[0],
        Some(Value::List(items)) => items.len(),
        other => panic!("expected LUT tensor or list after round-trip, got {other:?}"),
    };
    assert_eq!(
        lut_shape_before[0], lut_len_after,
        "LUT size should be preserved through serialization"
    );

    // Verify indices shape is preserved.
    let indices_shape_after = match rt_ops[0].attributes.get("indices") {
        Some(Value::Tensor { shape, .. }) => shape.clone(),
        other => panic!("expected indices tensor after round-trip, got {other:?}"),
    };
    assert_eq!(
        indices_shape_before, indices_shape_after,
        "indices shape should be preserved through serialization"
    );

    // Verify original shape tensor is preserved.
    let orig_shape_after = match rt_ops[0].attributes.get("shape") {
        Some(Value::Tensor { shape, .. }) => shape.clone(),
        other => panic!("expected shape tensor after round-trip, got {other:?}"),
    };
    assert_eq!(
        orig_shape_before, orig_shape_after,
        "original shape tensor should be preserved through serialization"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// PolarQuant unit + quality tests
// ═══════════════════════════════════════════════════════════════════════

/// Helper: build a program with a single const op of the given shape (FP32)
/// and a matmul consumer.
fn make_program_with_const(shape: Vec<usize>) -> Program {
    let numel: usize = shape.iter().product();
    let tensor_data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.001).collect();
    let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let input_ty = TensorType::new(ScalarType::Float32, vec![1, shape[shape.len() - 1]]);
    let mut func = Function::new("main");
    func = func.with_input("input", input_ty);

    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(
        Operation::new("const", "weight_0")
            .with_input(
                "val",
                Value::Tensor {
                    data: tensor_bytes,
                    shape,
                    dtype: ScalarType::Float32,
                },
            )
            .with_output("weight_0_out"),
    );
    block.add_op(
        Operation::new("matmul", "matmul_0")
            .with_input("x", Value::Reference("input".into()))
            .with_input("y", Value::Reference("weight_0_out".into()))
            .with_output("output"),
    );
    block.outputs.push("output".into());

    program
}

#[test]
fn polar_quant_converts_const_to_lut() {
    let mut program = make_program_with_const(vec![8, 128]); // 1024 elements
    let pass = PolarQuantPass::new(4);
    pass.run(&mut program).unwrap();

    let ops = &program.functions["main"].body.operations;
    assert_eq!(
        ops[0].op_type, "constexpr_lut_to_dense",
        "const op should be converted to constexpr_lut_to_dense"
    );
    assert!(ops[0].attributes.contains_key("lut"));
    assert!(ops[0].attributes.contains_key("indices"));
    assert!(ops[0].attributes.contains_key("shape"));
}

#[test]
fn polar_quant_preserves_output_shape() {
    let original_shape = vec![16, 64]; // 1024 elements
    let mut program = make_program_with_const(original_shape.clone());
    let pass = PolarQuantPass::new(4);
    pass.run(&mut program).unwrap();

    let ops = &program.functions["main"].body.operations;
    let shape_attr = match ops[0].attributes.get("shape") {
        Some(Value::Tensor { data, shape, dtype }) => {
            assert_eq!(*dtype, ScalarType::UInt32);
            // shape tensor records rank as its own shape
            assert_eq!(*shape, vec![original_shape.len()]);
            // decode the stored dimensions
            let dims: Vec<u32> = data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            dims.into_iter().map(|d| d as usize).collect::<Vec<_>>()
        }
        other => panic!("expected shape tensor, got {other:?}"),
    };
    assert_eq!(
        shape_attr, original_shape,
        "shape attribute should match original weight shape"
    );
}

#[test]
fn polar_quant_skips_small_tensors() {
    // 512 elements < 1024 min_elements threshold
    let mut program = make_program_with_const(vec![8, 64]);
    let pass = PolarQuantPass::new(4);
    pass.run(&mut program).unwrap();

    let ops = &program.functions["main"].body.operations;
    assert_eq!(
        ops[0].op_type, "const",
        "small tensors should not be quantized"
    );
}

#[test]
fn polar_quant_skips_non_float32() {
    // Build a program with an Int32 const tensor (>= 1024 elements).
    let shape = vec![8, 128];
    let numel: usize = shape.iter().product();
    let tensor_bytes: Vec<u8> = (0..numel as i32).flat_map(|v| v.to_le_bytes()).collect();

    let input_ty = TensorType::new(ScalarType::Float32, vec![1, 128]);
    let mut func = Function::new("main");
    func = func.with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(
        Operation::new("const", "weight_0")
            .with_input(
                "val",
                Value::Tensor {
                    data: tensor_bytes,
                    shape,
                    dtype: ScalarType::Int32,
                },
            )
            .with_output("weight_0_out"),
    );
    block.outputs.push("weight_0_out".into());

    let pass = PolarQuantPass::new(4);
    pass.run(&mut program).unwrap();

    let ops = &program.functions["main"].body.operations;
    assert_eq!(
        ops[0].op_type, "const",
        "non-FP32 tensors should not be quantized"
    );
}

#[test]
fn polar_quant_deterministic_with_seed() {
    let run_with_seed = |seed: u64| -> Vec<u8> {
        let mut program = make_program_with_const(vec![8, 128]);
        let pass = PolarQuantPass {
            n_bits: 4,
            seed,
            min_elements: 1024,
        };
        pass.run(&mut program).unwrap();
        let ops = &program.functions["main"].body.operations;
        match ops[0].attributes.get("indices") {
            Some(Value::Tensor { data, .. }) => data.clone(),
            other => panic!("expected indices tensor, got {other:?}"),
        }
    };

    let indices_a = run_with_seed(42);
    let indices_b = run_with_seed(42);
    assert_eq!(
        indices_a, indices_b,
        "same seed should produce identical indices"
    );
}

#[test]
fn polar_quant_different_seeds_differ() {
    let run_with_seed = |seed: u64| -> Vec<u8> {
        // Use structured data so rotation produces meaningfully different results.
        let rows = 8;
        let cols = 128;
        let numel = rows * cols;
        let tensor_data: Vec<f32> = (0..numel)
            .map(|i| {
                let r = (i / cols) as f32;
                let c = (i % cols) as f32;
                (r * 0.7 + c * 0.03).sin() * (r * 0.1).cos()
            })
            .collect();
        let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let input_ty = TensorType::new(ScalarType::Float32, vec![1, cols]);
        let mut func = Function::new("main");
        func = func.with_input("input", input_ty);
        let mut program = Program::new("1");
        program.add_function(func);
        let block = &mut program.functions.get_mut("main").unwrap().body;
        block.add_op(
            Operation::new("const", "weight_0")
                .with_input(
                    "val",
                    Value::Tensor {
                        data: tensor_bytes,
                        shape: vec![rows, cols],
                        dtype: ScalarType::Float32,
                    },
                )
                .with_output("weight_0_out"),
        );
        block.outputs.push("weight_0_out".into());

        let pass = PolarQuantPass {
            n_bits: 4,
            seed,
            min_elements: 1024,
        };
        pass.run(&mut program).unwrap();
        let ops = &program.functions["main"].body.operations;
        match ops[0].attributes.get("indices") {
            Some(Value::Tensor { data, .. }) => data.clone(),
            other => panic!("expected indices tensor, got {other:?}"),
        }
    };

    let indices_a = run_with_seed(1);
    let indices_b = run_with_seed(9999);
    assert_ne!(
        indices_a, indices_b,
        "different seeds should produce different quantized indices"
    );
}

#[test]
fn polar_quant_handles_non_power_of_two() {
    // Inner dim 100 is not a power of two; should still work via padding.
    let mut program = make_program_with_const(vec![12, 100]); // 1200 elements
    let pass = PolarQuantPass::new(4);
    pass.run(&mut program).unwrap();

    let ops = &program.functions["main"].body.operations;
    assert_eq!(
        ops[0].op_type, "constexpr_lut_to_dense",
        "non-power-of-two inner dim should be quantized via padding"
    );
    // Verify the shape attribute still records the original shape.
    match ops[0].attributes.get("shape") {
        Some(Value::Tensor { data, .. }) => {
            let dims: Vec<u32> = data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(dims, vec![12, 100]);
        }
        other => panic!("expected shape tensor, got {other:?}"),
    }
}

#[test]
fn polar_quant_handles_rank1() {
    let mut program = make_program_with_const(vec![1024]);
    let pass = PolarQuantPass::new(4);
    pass.run(&mut program).unwrap();

    let ops = &program.functions["main"].body.operations;
    assert_eq!(
        ops[0].op_type, "constexpr_lut_to_dense",
        "rank-1 tensor should be quantized"
    );
    // Shape attribute should record [1024].
    match ops[0].attributes.get("shape") {
        Some(Value::Tensor { data, .. }) => {
            let dims: Vec<u32> = data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(dims, vec![1024]);
        }
        other => panic!("expected shape tensor, got {other:?}"),
    }
}

#[test]
fn polar_quant_rejects_invalid_bits() {
    let result = PassPipeline::new().with_polar_quant(5);
    assert!(result.is_err(), "n_bits=5 should be rejected");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("n_bits must be 2 or 4"),
        "error should mention valid bit values, got: {err_msg}"
    );

    let result = PassPipeline::new().with_polar_quant(1);
    assert!(result.is_err(), "n_bits=1 should be rejected");

    let result = PassPipeline::new().with_polar_quant(3);
    assert!(
        result.is_err(),
        "n_bits=3 should be rejected (unsupported CoreML LUT size)"
    );

    // Valid values should succeed.
    for bits in [2, 4] {
        assert!(
            PassPipeline::new().with_polar_quant(bits).is_ok(),
            "n_bits={bits} should be accepted"
        );
    }
}

#[test]
fn polar_4bit_round_trip_quality() {
    // Create deterministic FP32 weights using sin/cos patterns.
    let rows = 8;
    let cols = 128;
    let numel = rows * cols; // 1024
    let original_data: Vec<f32> = (0..numel)
        .map(|i| {
            let x = i as f32 * 0.01;
            x.sin() * 0.5 + x.cos() * 0.3
        })
        .collect();
    let tensor_bytes: Vec<u8> = original_data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let input_ty = TensorType::new(ScalarType::Float32, vec![1, cols]);
    let mut func = Function::new("main");
    func = func.with_input("input", input_ty);
    let mut program = Program::new("1");
    program.add_function(func);

    let block = &mut program.functions.get_mut("main").unwrap().body;
    block.add_op(
        Operation::new("const", "weight_0")
            .with_input(
                "val",
                Value::Tensor {
                    data: tensor_bytes,
                    shape: vec![rows, cols],
                    dtype: ScalarType::Float32,
                },
            )
            .with_output("weight_0_out"),
    );
    block.outputs.push("weight_0_out".into());

    let pass = PolarQuantPass::new(4);
    pass.run(&mut program).unwrap();

    let ops = &program.functions["main"].body.operations;
    assert_eq!(ops[0].op_type, "constexpr_lut_to_dense");

    // Extract the LUT and decode to f32.
    let lut_f32: Vec<f32> = match ops[0].attributes.get("lut") {
        Some(Value::Tensor {
            data,
            dtype: ScalarType::Float16,
            ..
        }) => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Some(Value::Tensor {
            data,
            dtype: ScalarType::Float32,
            ..
        }) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("expected lut tensor, got {other:?}"),
    };

    // Extract the packed indices.
    let (indices_data, _indices_shape) = match ops[0].attributes.get("indices") {
        Some(Value::Tensor { data, shape, .. }) => (data.clone(), shape.clone()),
        other => panic!("expected indices tensor, got {other:?}"),
    };

    // Unpack 4-bit indices (MSB-first packing).
    let n_bits: usize = 4;
    let mut unpacked_indices = Vec::with_capacity(numel);
    for i in 0..numel {
        let bit_offset = i * n_bits;
        let byte_pos = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;
        let mask = (1u16 << n_bits) - 1;

        let mut word = (indices_data[byte_pos] as u16) << 8;
        if byte_pos + 1 < indices_data.len() {
            word |= indices_data[byte_pos + 1] as u16;
        }
        let shift = 16 - n_bits - bit_in_byte;
        let idx = ((word >> shift) & mask) as usize;
        unpacked_indices.push(idx);
    }

    // Extract norms from the norms const op.
    let norms_f32: Vec<f32> = match ops[1]
        .inputs
        .get("val")
        .or_else(|| ops[1].attributes.get("val"))
    {
        Some(Value::Tensor {
            data,
            dtype: ScalarType::Float16,
            ..
        }) => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Some(Value::Tensor {
            data,
            dtype: ScalarType::Float32,
            ..
        }) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("expected norms tensor, got {other:?}"),
    };

    // Reconstruct the quantized weights: lut[index] * norm.
    // The norms have shape [rows, 1], one per row.
    let mut reconstructed = Vec::with_capacity(numel);
    for r in 0..rows {
        let norm = norms_f32[r];
        for c in 0..cols {
            let idx = unpacked_indices[r * cols + c];
            reconstructed.push(lut_f32[idx] * norm);
        }
    }

    // Compute MSE between original and reconstructed.
    let mse: f64 = original_data
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| {
            let diff = *o as f64 - *r as f64;
            diff * diff
        })
        .sum::<f64>()
        / numel as f64;

    assert!(
        mse < 0.5,
        "4-bit PolarQuant MSE should be < 0.5, got {mse:.6}"
    );
}

// ─── Architecture-Specific PolarQuant Tests ────────────────────────────
// Require transformer model fixtures. Skipped if fixtures not present.

#[test]
fn polar_quant_on_whisper_tiny_encoder() {
    let path = common::fixture_path("whisper-tiny-encoder.onnx");
    if !path.exists() {
        eprintln!("SKIP: whisper-tiny-encoder.onnx not found (run download-fixtures.sh)");
        return;
    }

    let onnx = mil_rs::read_onnx(&path).unwrap();
    let cr = mil_rs::onnx_to_program(&onnx).unwrap();
    let mut program = cr.program;

    let pass = PolarQuantPass::new(4);
    pass.run(&mut program).unwrap();

    let lut_count = program
        .functions
        .values()
        .flat_map(|f| &f.body.operations)
        .filter(|op| op.op_type == "constexpr_lut_to_dense")
        .count();

    eprintln!("whisper-tiny-encoder: {lut_count} tensors quantized by PolarQuant");
    assert!(
        lut_count > 0,
        "PolarQuant should quantize at least one tensor in whisper-tiny encoder"
    );

    let model = program_to_model(&program, 7).unwrap();
    let _rt = model_to_program(&model).unwrap();
}

#[test]
fn polar_quant_on_distilbert() {
    let path = common::fixture_path("distilbert.onnx");
    if !path.exists() {
        eprintln!("SKIP: distilbert.onnx not found (run download-fixtures.sh)");
        return;
    }

    let onnx = mil_rs::read_onnx(&path).unwrap();
    let cr = mil_rs::onnx_to_program(&onnx).unwrap();
    let mut program = cr.program;

    let pass = PolarQuantPass::new(4);
    pass.run(&mut program).unwrap();

    let lut_count = program
        .functions
        .values()
        .flat_map(|f| &f.body.operations)
        .filter(|op| op.op_type == "constexpr_lut_to_dense")
        .count();

    eprintln!("distilbert: {lut_count} tensors quantized by PolarQuant");
    assert!(
        lut_count > 0,
        "PolarQuant should quantize at least one tensor in DistilBERT"
    );

    let model = program_to_model(&program, 7).unwrap();
    let _rt = model_to_program(&model).unwrap();
}

#[test]
fn polar_quant_on_vit_base() {
    let path = common::fixture_path("vit-base.onnx");
    if !path.exists() {
        eprintln!("SKIP: vit-base.onnx not found (run download-fixtures.sh)");
        return;
    }

    let onnx = mil_rs::read_onnx(&path).unwrap();
    let cr = mil_rs::onnx_to_program(&onnx).unwrap();
    let mut program = cr.program;

    let pass = PolarQuantPass::new(4);
    pass.run(&mut program).unwrap();

    let lut_count = program
        .functions
        .values()
        .flat_map(|f| &f.body.operations)
        .filter(|op| op.op_type == "constexpr_lut_to_dense")
        .count();

    eprintln!("vit-base: {lut_count} tensors quantized by PolarQuant");
    assert!(
        lut_count > 0,
        "PolarQuant should quantize at least one tensor in ViT-base"
    );

    let model = program_to_model(&program, 7).unwrap();
    let _rt = model_to_program(&model).unwrap();
}
