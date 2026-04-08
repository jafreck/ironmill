//! Smoke tests for the test-model fixture.
//!
//! Validates that [`common::test_model::create_test_program`] produces a
//! well-formed MIL program with the expected structure, and that
//! [`common::test_model::create_test_calibration_data`] returns valid tokens.

mod common;

use common::test_model;
use mil_rs::ir::{ScalarType, Value};

// ── Program structure ───────────────────────────────────────────────────

#[test]
fn test_program_has_main_function() {
    let program = test_model::create_test_program();
    let main = program.main().expect("program should have a main function");
    assert_eq!(main.name, "main");
}

#[test]
fn test_program_has_input_ids_input() {
    let program = test_model::create_test_program();
    let main = program.main().unwrap();
    assert_eq!(main.inputs.len(), 1);
    assert_eq!(main.inputs[0].0, "input_ids");
    assert_eq!(main.inputs[0].1.scalar_type, ScalarType::Int32);
}

#[test]
fn test_program_has_block_output() {
    let program = test_model::create_test_program();
    let main = program.main().unwrap();
    assert_eq!(
        main.body.outputs.len(),
        1,
        "block should have exactly one output"
    );
    assert!(
        main.body.outputs[0].contains("lm_head"),
        "output should be the LM head projection"
    );
}

// ── Const ops ───────────────────────────────────────────────────────────

#[test]
fn test_program_has_const_ops() {
    let program = test_model::create_test_program();
    let ops = &program.main().unwrap().body.operations;
    let const_ops: Vec<_> = ops.iter().filter(|op| op.op_type == "const").collect();

    // Embedding + 2 layers * (attn_norm + q + k + v + o + ffn_norm + gate + up + down)
    // + final_norm + lm_head = 1 + 2*9 + 1 + 1 = 21
    assert_eq!(
        const_ops.len(),
        21,
        "expected 21 const ops (embed + 2*9 layer weights + final_norm + lm_head)"
    );
}

#[test]
fn test_const_ops_carry_fp32_tensor_data() {
    let program = test_model::create_test_program();
    let ops = &program.main().unwrap().body.operations;

    for op in ops.iter().filter(|op| op.op_type == "const") {
        let val = op
            .inputs
            .get("val")
            .unwrap_or_else(|| panic!("const op '{}' missing 'val' input", op.name));
        match val {
            Value::Tensor { dtype, data, shape } => {
                assert_eq!(
                    *dtype,
                    ScalarType::Float32,
                    "const op '{}' should be FP32",
                    op.name
                );
                let expected_bytes = shape.iter().product::<usize>() * 4;
                assert_eq!(
                    data.byte_len(),
                    expected_bytes,
                    "const op '{}' data length mismatch: shape {:?}",
                    op.name,
                    shape
                );
            }
            other => panic!(
                "const op '{}' val should be Tensor, got {:?}",
                op.name, other
            ),
        }
    }
}

#[test]
fn test_const_ops_have_outputs() {
    let program = test_model::create_test_program();
    let ops = &program.main().unwrap().body.operations;

    for op in ops.iter().filter(|op| op.op_type == "const") {
        assert_eq!(
            op.outputs.len(),
            1,
            "const op '{}' should have exactly one output",
            op.name
        );
    }
}

// ── Linear ops ──────────────────────────────────────────────────────────

#[test]
fn test_program_has_linear_ops() {
    let program = test_model::create_test_program();
    let ops = &program.main().unwrap().body.operations;
    let linear_ops: Vec<_> = ops.iter().filter(|op| op.op_type == "linear").collect();

    // 2 layers * (q + k + v + o + gate + up + down) + lm_head = 2*7 + 1 = 15
    assert_eq!(
        linear_ops.len(),
        15,
        "expected 15 linear ops (2*7 per-layer + lm_head)"
    );
}

#[test]
fn test_linear_ops_reference_const_weights() {
    let program = test_model::create_test_program();
    let ops = &program.main().unwrap().body.operations;

    let const_outputs: std::collections::HashSet<String> = ops
        .iter()
        .filter(|op| op.op_type == "const")
        .flat_map(|op| op.outputs.iter().cloned())
        .collect();

    for op in ops.iter().filter(|op| op.op_type == "linear") {
        let weight_ref = match op.inputs.get("weight") {
            Some(Value::Reference(name)) => name,
            other => panic!(
                "linear op '{}' weight should be a Reference, got {:?}",
                op.name, other
            ),
        };
        assert!(
            const_outputs.contains(weight_ref),
            "linear op '{}' references weight '{}' which is not a const output",
            op.name,
            weight_ref
        );
    }
}

// ── Weight shapes ───────────────────────────────────────────────────────

#[test]
fn test_weight_shapes_are_group_quantization_compatible() {
    let program = test_model::create_test_program();
    let ops = &program.main().unwrap().body.operations;
    let group_size = 32_usize;

    let mut two_d_weight_count = 0;
    for op in ops.iter().filter(|op| op.op_type == "const") {
        if let Some(Value::Tensor { shape, .. }) = op.inputs.get("val") {
            if shape.len() == 2 {
                two_d_weight_count += 1;
                // Both dimensions should be divisible by group_size
                // (or at least one dimension, depending on axis).
                let divisible = shape.iter().any(|&d| d % group_size == 0);
                assert!(
                    divisible,
                    "2D weight '{}' shape {:?} has no dimension divisible by group_size={}",
                    op.name, shape, group_size
                );
            }
        }
    }

    // At least 4 two-dimensional weight matrices (spec requirement)
    assert!(
        two_d_weight_count >= 4,
        "expected at least 4 two-dimensional weight matrices, got {}",
        two_d_weight_count
    );
}

// ── Determinism ─────────────────────────────────────────────────────────

#[test]
fn test_program_is_deterministic() {
    let p1 = test_model::create_test_program();
    let p2 = test_model::create_test_program();

    let ops1 = &p1.main().unwrap().body.operations;
    let ops2 = &p2.main().unwrap().body.operations;

    assert_eq!(ops1.len(), ops2.len(), "op count should be identical");

    for (a, b) in ops1.iter().zip(ops2.iter()) {
        assert_eq!(a.op_type, b.op_type);
        assert_eq!(a.name, b.name);
        assert_eq!(a.outputs, b.outputs);
        // Compare tensor data byte-for-byte
        if let (Some(Value::Tensor { data: d1, .. }), Some(Value::Tensor { data: d2, .. })) =
            (a.inputs.get("val"), b.inputs.get("val"))
        {
            assert_eq!(
                d1, d2,
                "tensor data for '{}' should be deterministic",
                a.name
            );
        }
    }
}

// ── Other op types ──────────────────────────────────────────────────────

#[test]
fn test_program_has_expected_op_types() {
    let program = test_model::create_test_program();
    let ops = &program.main().unwrap().body.operations;

    let mut op_type_counts = std::collections::HashMap::new();
    for op in ops {
        *op_type_counts.entry(op.op_type.as_str()).or_insert(0_usize) += 1;
    }

    assert_eq!(op_type_counts.get("const"), Some(&21));
    assert_eq!(op_type_counts.get("linear"), Some(&15));
    assert_eq!(op_type_counts.get("gather"), Some(&1));
    assert_eq!(op_type_counts.get("rms_norm"), Some(&5)); // 2*(attn+ffn) + final
    assert_eq!(op_type_counts.get("matmul"), Some(&4)); // 2 layers * 2
    assert_eq!(op_type_counts.get("silu"), Some(&2));
    assert_eq!(op_type_counts.get("mul"), Some(&2));
    assert_eq!(op_type_counts.get("add"), Some(&4)); // 2 layers * 2 residuals
}

// ── Calibration data ────────────────────────────────────────────────────

#[test]
fn test_calibration_data_shape() {
    let data = test_model::create_test_calibration_data(8, 64, 42);
    assert_eq!(data.len(), 8);
    for seq in &data {
        assert_eq!(seq.len(), 64);
    }
}

#[test]
fn test_calibration_data_in_vocab_range() {
    let vocab = test_model::vocab_size() as u32;
    let data = test_model::create_test_calibration_data(16, 128, 99);
    for seq in &data {
        for &token in seq {
            assert!(token < vocab, "token {} >= vocab_size {}", token, vocab);
        }
    }
}

#[test]
fn test_calibration_data_is_deterministic() {
    let d1 = test_model::create_test_calibration_data(4, 32, 123);
    let d2 = test_model::create_test_calibration_data(4, 32, 123);
    assert_eq!(
        d1, d2,
        "calibration data should be deterministic for same seed"
    );
}

#[test]
fn test_calibration_data_different_seeds_differ() {
    let d1 = test_model::create_test_calibration_data(4, 32, 1);
    let d2 = test_model::create_test_calibration_data(4, 32, 2);
    assert_ne!(d1, d2, "different seeds should produce different data");
}
