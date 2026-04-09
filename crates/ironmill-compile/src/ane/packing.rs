//! Spatial I/O packing for ANE sub-programs.
//!
//! Packs multiple function inputs into a single IOSurface tensor along
//! dim 3 (spatial), and unpacks them with `slice_by_size` at the start of
//! the MIL. This eliminates multi-input ANE requests (robustness) and
//! reduces the overhead of S=32 padding per tensor.
//!
//! Outputs are **not** packed — ANE does not support `concat`.

use std::collections::HashMap;

use mil_rs::ir::{Operation, ScalarType, TensorType, Value};

use crate::ane::split::SubProgram;

/// Recursively rewrite `Value::Reference` names in a value tree.
#[allow(dead_code)]
fn rewrite_refs(val: &mut Value, rename_map: &HashMap<String, String>) {
    match val {
        Value::Reference(r) => {
            if let Some(new_name) = rename_map.get(r.as_str()) {
                *r = new_name.clone();
            }
        }
        Value::List(items) => {
            for item in items {
                rewrite_refs(item, rename_map);
            }
        }
        _ => {}
    }
}
use ironmill_core::ane::packing::InputPacking;

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// Pack multiple function inputs into a single spatially-concatenated tensor.
///
/// Returns `Some(InputPacking)` if packing was applied, `None` if the
/// sub-program was left unchanged (single input, mixed channel counts, etc.).
///
/// The transform modifies the sub-program's MIL IR in place:
/// 1. Creates a single packed input `a_packed_input: [1, C, 1, total_S]`
/// 2. Inserts `slice_by_size` ops to extract each original input
/// 3. Replaces references in the function body
/// 4. Updates `SubProgram.inputs` to reflect the single packed input
#[allow(dead_code)]
pub fn pack_inputs(sub: &mut SubProgram) -> Option<InputPacking> {
    // Skip sub-programs with 0 or 1 input — nothing to pack.
    if sub.inputs.len() <= 1 {
        return None;
    }

    let func = sub.program.main()?;

    // All inputs must have the same channel count (dim 1) to pack along spatial.
    let channel_counts: Vec<usize> = sub.inputs.iter().map(|td| td.shape[1]).collect();
    let max_c = *channel_counts.iter().max()?;
    if !channel_counts.iter().all(|&c| c == max_c) {
        return None;
    }

    // All inputs must have the same dtype.
    let dtype = sub.inputs[0].dtype;
    if !sub.inputs.iter().all(|td| td.dtype == dtype) {
        return None;
    }

    // Compute spatial offsets and total size.
    let spatial_sizes: Vec<usize> = sub.inputs.iter().map(|td| td.shape[3]).collect();
    let mut offsets = Vec::with_capacity(sub.inputs.len());
    let mut offset = 0usize;
    for &s in &spatial_sizes {
        offsets.push(offset);
        offset += s;
    }
    let total_s = offset;

    // Build the packed input name and type.
    let packed_name = "a_packed_input".to_string();
    let packed_type = TensorType::new(dtype, vec![1, max_c, 1, total_s]);

    // Build slice_by_size ops to extract each original input.
    let original_names: Vec<String> = func.inputs.iter().map(|(n, _)| n.clone()).collect();
    let mut new_ops: Vec<Operation> = Vec::with_capacity(sub.inputs.len() * 3);

    for (i, orig_name) in original_names.iter().enumerate() {
        let s_i = spatial_sizes[i];
        let off_i = offsets[i];

        // Const for begin: [0, 0, 0, offset_i]
        let begin_name = format!("_pack_begin_{i}");
        let off_i32 = i32::try_from(off_i)
            .expect("spatial offset exceeds i32::MAX — model too large for ANE packing");
        let begin_data: Vec<u8> = [0i32, 0, 0, off_i32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let begin_op = Operation::new("const", &begin_name)
            .with_input(
                "val",
                Value::Tensor {
                    data: mil_rs::ir::TensorData::Inline(begin_data),
                    shape: vec![4],
                    dtype: ScalarType::Int32,
                },
            )
            .with_output(&begin_name);

        // Const for size: [1, C_i, 1, S_i]
        let size_name = format!("_pack_size_{i}");
        let c_i = channel_counts[i];
        let c_i32 = i32::try_from(c_i)
            .expect("channel count exceeds i32::MAX — model too large for ANE packing");
        let s_i32 = i32::try_from(s_i)
            .expect("spatial size exceeds i32::MAX — model too large for ANE packing");
        let size_data: Vec<u8> = [1i32, c_i32, 1, s_i32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let size_op = Operation::new("const", &size_name)
            .with_input(
                "val",
                Value::Tensor {
                    data: mil_rs::ir::TensorData::Inline(size_data),
                    shape: vec![4],
                    dtype: ScalarType::Int32,
                },
            )
            .with_output(&size_name);

        // slice_by_size op
        let slice_out = format!("{orig_name}_unpacked");
        let mut slice_op = Operation::new("slice_by_size", format!("_pack_slice_{i}"))
            .with_input("x", Value::Reference(packed_name.clone()))
            .with_input("begin", Value::Reference(begin_name.clone()))
            .with_input("size", Value::Reference(size_name.clone()))
            .with_output(&slice_out);

        // Set output type to the original input's shape.
        let orig_type = TensorType::new(dtype, vec![1, c_i, 1, s_i]);
        slice_op.output_types = vec![Some(orig_type)];

        new_ops.push(begin_op);
        new_ops.push(size_op);
        new_ops.push(slice_op);
    }

    // Now mutably access the function to apply changes.
    // Program::main_mut() doesn't exist, so access via functions map.
    let func_mut = sub.program.functions.values_mut().next()?;

    // Replace references in the existing body ops: original input names → unpacked names.
    let rename_map: HashMap<String, String> = original_names
        .iter()
        .map(|n| (n.clone(), format!("{n}_unpacked")))
        .collect();

    for op in &mut func_mut.body.operations {
        for val in op.inputs.values_mut() {
            rewrite_refs(val, &rename_map);
        }
    }

    // Also rename references in block outputs (return values).
    for out in &mut func_mut.body.outputs {
        if let Some(new_name) = rename_map.get(out.as_str()) {
            *out = new_name.clone();
        }
    }

    // Prepend slice ops before the original body.
    let mut combined = new_ops;
    combined.append(&mut func_mut.body.operations);
    func_mut.body.operations = combined;

    // Replace function inputs with the single packed input.
    func_mut.inputs = vec![(packed_name.clone(), packed_type)];

    // Update sub-program input descriptors.
    sub.inputs = vec![ironmill_core::ane::TensorDescriptor {
        name: packed_name,
        shape: [1, max_c, 1, total_s],
        dtype,
    }];

    Some(InputPacking {
        offsets,
        sizes: spatial_sizes,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::{Function, Operation, Program, ScalarType, TensorType, Value};

    /// Build a minimal sub-program with the given named inputs and a simple
    /// pass-through body (add the inputs together).
    fn make_sub_program(
        name: &str,
        input_specs: &[(&str, usize, usize)], // (name, channels, seq_len)
    ) -> SubProgram {
        let dtype = ScalarType::Float16;
        let mut func = Function::new("main");

        for &(iname, c, s) in input_specs {
            func.inputs
                .push((iname.to_string(), TensorType::new(dtype, vec![1, c, 1, s])));
        }

        // Body: if 2 inputs, add them; if 1, relu; if >2, chain adds.
        let mut ops = Vec::new();
        if input_specs.len() == 1 {
            let mut op = Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference(input_specs[0].0.to_string()))
                .with_output("out_0");
            op.output_types = vec![Some(TensorType::new(
                dtype,
                vec![1, input_specs[0].1, 1, input_specs[0].2],
            ))];
            ops.push(op);
        } else {
            // Add first two
            let out_c = input_specs[0].1;
            let out_s = input_specs[0].2;
            let mut prev_out = "add_0".to_string();
            let mut add_op = Operation::new("add", "add_0")
                .with_input("x", Value::Reference(input_specs[0].0.to_string()))
                .with_input("y", Value::Reference(input_specs[1].0.to_string()))
                .with_output("add_0");
            add_op.output_types = vec![Some(TensorType::new(dtype, vec![1, out_c, 1, out_s]))];
            ops.push(add_op);

            for (i, spec) in input_specs.iter().enumerate().skip(2) {
                let out_name = format!("add_{i}");
                let mut op = Operation::new("add", &out_name)
                    .with_input("x", Value::Reference(prev_out))
                    .with_input("y", Value::Reference(spec.0.to_string()))
                    .with_output(&out_name);
                op.output_types = vec![Some(TensorType::new(dtype, vec![1, out_c, 1, out_s]))];
                ops.push(op);
                prev_out = out_name;
            }
        }

        let final_output = ops.last().unwrap().outputs[0].clone();
        func.body.operations = ops;
        func.body.outputs = vec![final_output.clone()];

        let mut prog = Program::new("1.0");
        prog.add_function(func);

        let inputs = input_specs
            .iter()
            .map(|&(n, c, s)| ironmill_core::ane::TensorDescriptor {
                name: n.to_string(),
                shape: [1, c, 1, s],
                dtype,
            })
            .collect();

        let out_c = input_specs[0].1;
        let out_s = input_specs[0].2;
        let outputs = vec![ironmill_core::ane::TensorDescriptor {
            name: final_output,
            shape: [1, out_c, 1, out_s],
            dtype,
        }];

        SubProgram {
            name: name.to_string(),
            program: prog,
            inputs,
            outputs,
        }
    }

    #[test]
    fn skip_single_input() {
        let mut sub = make_sub_program("test", &[("a_input0", 768, 32)]);
        let result = pack_inputs(&mut sub);
        assert!(result.is_none(), "single input should not be packed");
        assert_eq!(sub.inputs.len(), 1);
    }

    #[test]
    fn skip_mixed_channels() {
        let mut sub = make_sub_program("test", &[("a_input0", 768, 32), ("a_input1", 256, 32)]);
        let result = pack_inputs(&mut sub);
        assert!(
            result.is_none(),
            "mixed channel counts should not be packed"
        );
        assert_eq!(sub.inputs.len(), 2);
    }

    #[test]
    fn pack_two_same_channel_inputs() {
        let mut sub = make_sub_program("test", &[("a_input0", 768, 32), ("a_input1", 768, 32)]);
        let result = pack_inputs(&mut sub);

        let packing = result.expect("should pack 2 same-channel inputs");
        assert_eq!(packing.offsets, vec![0, 32]);
        assert_eq!(packing.sizes, vec![32, 32]);

        // Sub-program should now have a single input.
        assert_eq!(sub.inputs.len(), 1);
        assert_eq!(sub.inputs[0].name, "a_packed_input");
        assert_eq!(sub.inputs[0].shape, [1, 768, 1, 64]);

        // Function should have a single input.
        let func = sub.program.main().unwrap();
        assert_eq!(func.inputs.len(), 1);
        assert_eq!(func.inputs[0].0, "a_packed_input");

        // Body should start with const+const+slice, const+const+slice, then add.
        // 6 new ops (2 inputs × 3 ops each) + 1 original add = 7
        assert_eq!(func.body.operations.len(), 7);

        // First 6 ops are packing ops.
        assert_eq!(func.body.operations[0].op_type, "const");
        assert_eq!(func.body.operations[1].op_type, "const");
        assert_eq!(func.body.operations[2].op_type, "slice_by_size");
        assert_eq!(func.body.operations[3].op_type, "const");
        assert_eq!(func.body.operations[4].op_type, "const");
        assert_eq!(func.body.operations[5].op_type, "slice_by_size");

        // The add op's inputs should reference the unpacked names.
        let add_op = &func.body.operations[6];
        assert_eq!(add_op.op_type, "add");
        assert_eq!(
            add_op.inputs.get("x"),
            Some(&Value::Reference("a_input0_unpacked".to_string()))
        );
        assert_eq!(
            add_op.inputs.get("y"),
            Some(&Value::Reference("a_input1_unpacked".to_string()))
        );
    }

    #[test]
    fn pack_three_inputs() {
        let mut sub = make_sub_program(
            "test",
            &[
                ("a_input0", 128, 32),
                ("a_input1", 128, 32),
                ("a_input2", 128, 32),
            ],
        );
        let result = pack_inputs(&mut sub);

        let packing = result.expect("should pack 3 same-channel inputs");
        assert_eq!(packing.offsets, vec![0, 32, 64]);
        assert_eq!(packing.sizes, vec![32, 32, 32]);
        assert_eq!(sub.inputs.len(), 1);
        assert_eq!(sub.inputs[0].shape, [1, 128, 1, 96]);
    }

    #[test]
    fn pack_asymmetric_spatial() {
        let mut sub = make_sub_program("test", &[("a_input0", 512, 32), ("a_input1", 512, 64)]);
        let result = pack_inputs(&mut sub);

        let packing = result.expect("should pack inputs with different S");
        assert_eq!(packing.offsets, vec![0, 32]);
        assert_eq!(packing.sizes, vec![32, 64]);
        assert_eq!(sub.inputs[0].shape, [1, 512, 1, 96]);
    }

    #[test]
    fn packed_slice_output_types_are_correct() {
        let mut sub = make_sub_program("test", &[("a_input0", 768, 32), ("a_input1", 768, 32)]);
        pack_inputs(&mut sub);

        let func = sub.program.main().unwrap();
        // slice ops are at indices 2 and 5
        let slice_0 = &func.body.operations[2];
        assert_eq!(slice_0.op_type, "slice_by_size");
        assert_eq!(
            slice_0.output_types[0],
            Some(TensorType::new(ScalarType::Float16, vec![1, 768, 1, 32]))
        );

        let slice_1 = &func.body.operations[5];
        assert_eq!(slice_1.op_type, "slice_by_size");
        assert_eq!(
            slice_1.output_types[0],
            Some(TensorType::new(ScalarType::Float16, vec![1, 768, 1, 32]))
        );
    }

    #[test]
    fn outputs_are_not_modified() {
        let mut sub = make_sub_program("test", &[("a_input0", 768, 32), ("a_input1", 768, 32)]);
        let orig_outputs = sub.outputs.clone();
        pack_inputs(&mut sub);
        assert_eq!(sub.outputs, orig_outputs);
    }

    #[test]
    fn skip_empty_inputs() {
        let mut sub = SubProgram {
            name: "empty".to_string(),
            program: {
                let mut p = Program::new("1.0");
                p.add_function(Function::new("main"));
                p
            },
            inputs: vec![],
            outputs: vec![],
        };
        let result = pack_inputs(&mut sub);
        assert!(result.is_none());
    }
}
