//! Deviation-Aware Correction (DAC) pass.
//!
//! After weight quantization, the activation distribution shifts. This pass
//! adjusts LayerNorm weight (gamma) and bias (beta) parameters to compensate:
//!   gamma_new = gamma * (sigma_fp16 / sigma_quant)
//!   beta_new  = beta + gamma * (mu_fp16 - mu_quant)
//!
//! This requires two sets of activation statistics: one from the FP16 model
//! and one from the quantized model, both collected on the same calibration data.

use std::collections::HashMap;

use super::super::tensor_utils::{f32_slice_to_bytes, tensor_as_f32_slice};
use crate::error::Result;
use crate::ir::pass::Pass;
use crate::ir::program::Program;
use crate::ir::tensor::ScalarType;
use crate::ir::types::{TensorData, Value};

/// Per-channel activation statistics for a single layer.
pub struct ChannelStats {
    /// Per-channel mean.
    pub mean: Vec<f32>,
    /// Per-channel variance.
    pub variance: Vec<f32>,
}

/// Norm op types eligible for DAC correction.
const CORRECTABLE_NORM_OPS: &[&str] = &["layer_norm", "rms_norm"];

/// Deviation-Aware Correction (DAC) pass.
///
/// Walks `layer_norm` and `rms_norm` ops, and adjusts their gamma (and beta
/// for `layer_norm`) to compensate for activation distribution shifts caused
/// by weight quantization.
pub struct DacPass {
    /// Per-layer activation statistics from FP16 model run.
    /// Key: layer name matching the norm op's `name` field.
    pub fp16_stats: HashMap<String, ChannelStats>,
    /// Per-layer activation statistics from quantized model run.
    pub quant_stats: HashMap<String, ChannelStats>,
}

impl Pass for DacPass {
    fn name(&self) -> &str {
        "dac"
    }

    fn run(&self, program: &mut Program) -> Result<()> {
        let provider = program.weight_provider.clone();
        let resolve = super::super::util::make_resolver(&provider);

        for function in program.functions.values_mut() {
            for op in &mut function.body.operations {
                if !CORRECTABLE_NORM_OPS.contains(&op.op_type.as_str()) {
                    continue;
                }

                // Materialize External tensors in inputs before reading params.
                for val in op.inputs.values_mut() {
                    if let Value::Tensor { data, .. } = val {
                        data.materialize_with(|key| resolve(key))?;
                    }
                }

                let fp16 = match self.fp16_stats.get(&op.name) {
                    Some(s) => s,
                    None => continue,
                };
                let quant = match self.quant_stats.get(&op.name) {
                    Some(s) => s,
                    None => continue,
                };

                let num_channels = fp16.mean.len();

                // Validate dimensions match.
                if fp16.variance.len() != num_channels
                    || quant.mean.len() != num_channels
                    || quant.variance.len() != num_channels
                {
                    continue;
                }

                // Compute per-channel correction factors.
                let ratio: Vec<f32> = (0..num_channels)
                    .map(|c| {
                        // Use a relative floor to prevent extreme amplification
                        // when quantization collapses variance to near-zero.
                        let min_var = 1e-6_f32.max(fp16.variance[c] * 0.01);
                        let denom = quant.variance[c].max(min_var);
                        (fp16.variance[c] / denom).sqrt()
                    })
                    .collect();

                let shift: Vec<f32> = (0..num_channels)
                    .map(|c| fp16.mean[c] - quant.mean[c])
                    .collect();

                // --- Adjust gamma ---
                adjust_param(&mut op.inputs, "gamma", num_channels, |gamma_floats| {
                    // Also capture old gamma for beta adjustment below.
                    let old_gamma = gamma_floats.to_vec();
                    for c in 0..gamma_floats.len().min(num_channels) {
                        gamma_floats[c] *= ratio[c];
                    }
                    old_gamma
                });

                // --- Adjust beta (layer_norm only; rms_norm has no beta) ---
                if op.op_type == "layer_norm" {
                    // We need old gamma to compute the beta shift. Re-read it
                    // from the (now-updated) gamma, dividing out the ratio to
                    // recover old values.
                    let old_gamma: Vec<f32> = read_param(&op.inputs, "gamma")
                        .map(|g| {
                            g.iter()
                                .enumerate()
                                .map(|(c, &v)| if c < num_channels { v / ratio[c] } else { v })
                                .collect()
                        })
                        .unwrap_or_else(|| vec![1.0; num_channels]);

                    adjust_param(&mut op.inputs, "beta", num_channels, |beta_floats| {
                        for c in 0..beta_floats.len().min(num_channels) {
                            beta_floats[c] += old_gamma[c] * shift[c];
                        }
                        Vec::new() // unused return
                    });
                }
            }
        }
        Ok(())
    }
}

/// Read a parameter tensor from the op's inputs map, returning its f32 values.
fn read_param(inputs: &HashMap<String, Value>, key: &str) -> Option<Vec<f32>> {
    match inputs.get(key) {
        Some(Value::Tensor {
            data,
            dtype: ScalarType::Float32,
            ..
        }) => Some(tensor_as_f32_slice(
            data.as_bytes().expect("tensor not materialized"),
        )),
        _ => None,
    }
}

/// Modify a parameter tensor stored inline in the op's inputs map.
///
/// Calls `f` with the mutable f32 slice; `f` may return auxiliary data (used
/// by the caller to thread old-gamma into the beta adjustment).
fn adjust_param<F>(
    inputs: &mut HashMap<String, Value>,
    key: &str,
    _num_channels: usize,
    f: F,
) -> Option<Vec<f32>>
where
    F: FnOnce(&mut Vec<f32>) -> Vec<f32>,
{
    if let Some(Value::Tensor { data, shape, dtype }) = inputs.get(key).cloned() {
        if dtype == ScalarType::Float32 {
            let mut floats = tensor_as_f32_slice(data.as_bytes().expect("tensor not materialized"));
            let result = f(&mut floats);
            inputs.insert(
                key.to_string(),
                Value::Tensor {
                    data: TensorData::Inline(f32_slice_to_bytes(&floats)),
                    shape,
                    dtype,
                },
            );
            return Some(result);
        }
    }
    None
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operation::Operation;
    use crate::ir::program::Function;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn make_tensor(values: &[f32]) -> Value {
        Value::Tensor {
            data: TensorData::Inline(f32_bytes(values)),
            shape: vec![values.len()],
            dtype: ScalarType::Float32,
        }
    }

    fn stats(mean: Vec<f32>, variance: Vec<f32>) -> ChannelStats {
        ChannelStats { mean, variance }
    }

    /// Build a single-function program with one norm op that has inline
    /// gamma (and optionally beta) tensors.
    fn program_with_norm(
        op_type: &str,
        norm_name: &str,
        gamma: &[f32],
        beta: Option<&[f32]>,
    ) -> Program {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");

        let mut norm_op = Operation::new(op_type, norm_name)
            .with_input("x", Value::Reference("input".into()))
            .with_input("gamma", make_tensor(gamma))
            .with_output("norm_out");

        if let Some(b) = beta {
            norm_op = norm_op.with_input("beta", make_tensor(b));
        }

        func.body.add_op(norm_op);
        func.body.outputs.push("norm_out".into());
        program.add_function(func);
        program
    }

    // -----------------------------------------------------------------------
    // DAC adjusts gamma and beta correctly for known statistics
    // -----------------------------------------------------------------------

    #[test]
    fn dac_adjusts_gamma_and_beta_for_layer_norm() {
        let gamma = [2.0_f32, 4.0, 1.0];
        let beta = [0.5_f32, -1.0, 0.0];

        let fp16_mean = [1.0_f32, 2.0, 3.0];
        let fp16_var = [4.0_f32, 9.0, 16.0];
        let quant_mean = [1.5_f32, 1.0, 4.0];
        let quant_var = [1.0_f32, 4.0, 4.0];

        let mut fp16_stats = HashMap::new();
        fp16_stats.insert(
            "ln_0".to_string(),
            stats(fp16_mean.to_vec(), fp16_var.to_vec()),
        );
        let mut quant_stats = HashMap::new();
        quant_stats.insert(
            "ln_0".to_string(),
            stats(quant_mean.to_vec(), quant_var.to_vec()),
        );

        let pass = DacPass {
            fp16_stats,
            quant_stats,
        };

        let mut program = program_with_norm("layer_norm", "ln_0", &gamma, Some(&beta));
        pass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];

        // Expected ratio[c] = sqrt(fp16_var[c] / quant_var[c])
        // ratio = [sqrt(4/1), sqrt(9/4), sqrt(16/4)] = [2.0, 1.5, 2.0]
        let expected_ratio = [2.0_f32, 1.5, 2.0];

        // Expected shift[c] = fp16_mean[c] - quant_mean[c]
        // shift = [1.0 - 1.5, 2.0 - 1.0, 3.0 - 4.0] = [-0.5, 1.0, -1.0]
        let expected_shift = [-0.5_f32, 1.0, -1.0];

        // gamma_new[c] = gamma[c] * ratio[c]
        let expected_gamma: Vec<f32> = gamma
            .iter()
            .zip(expected_ratio.iter())
            .map(|(g, r)| g * r)
            .collect();
        // = [4.0, 6.0, 2.0]

        // beta_new[c] = beta[c] + gamma_old[c] * shift[c]
        let expected_beta: Vec<f32> = beta
            .iter()
            .zip(gamma.iter().zip(expected_shift.iter()))
            .map(|(b, (g, s))| b + g * s)
            .collect();
        // = [0.5 + 2.0*(-0.5), -1.0 + 4.0*1.0, 0.0 + 1.0*(-1.0)]
        // = [-0.5, 3.0, -1.0]

        let actual_gamma = match op.inputs.get("gamma") {
            Some(Value::Tensor { data, .. }) => {
                tensor_as_f32_slice(data.as_bytes().expect("tensor not materialized"))
            }
            _ => panic!("gamma should be a tensor"),
        };
        let actual_beta = match op.inputs.get("beta") {
            Some(Value::Tensor { data, .. }) => {
                tensor_as_f32_slice(data.as_bytes().expect("tensor not materialized"))
            }
            _ => panic!("beta should be a tensor"),
        };

        for (i, (a, e)) in actual_gamma.iter().zip(expected_gamma.iter()).enumerate() {
            assert!((a - e).abs() < 1e-6, "gamma[{i}]: got {a}, expected {e}");
        }
        for (i, (a, e)) in actual_beta.iter().zip(expected_beta.iter()).enumerate() {
            assert!((a - e).abs() < 1e-6, "beta[{i}]: got {a}, expected {e}");
        }
    }

    // -----------------------------------------------------------------------
    // RMSNorm only adjusts gamma (no beta)
    // -----------------------------------------------------------------------

    #[test]
    fn dac_adjusts_only_gamma_for_rms_norm() {
        let gamma = [3.0_f32, 1.0];
        let fp16_var = [9.0_f32, 16.0];
        let quant_var = [1.0_f32, 4.0];

        let mut fp16_stats = HashMap::new();
        fp16_stats.insert(
            "rmsn_0".to_string(),
            stats(vec![0.0, 0.0], fp16_var.to_vec()),
        );
        let mut quant_stats = HashMap::new();
        quant_stats.insert(
            "rmsn_0".to_string(),
            stats(vec![0.0, 0.0], quant_var.to_vec()),
        );

        let pass = DacPass {
            fp16_stats,
            quant_stats,
        };

        let mut program = program_with_norm("rms_norm", "rmsn_0", &gamma, None);
        pass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];

        // ratio = [sqrt(9/1), sqrt(16/4)] = [3.0, 2.0]
        // gamma_new = [3.0*3.0, 1.0*2.0] = [9.0, 2.0]
        let actual_gamma = match op.inputs.get("gamma") {
            Some(Value::Tensor { data, .. }) => {
                tensor_as_f32_slice(data.as_bytes().expect("tensor not materialized"))
            }
            _ => panic!("gamma should be a tensor"),
        };
        assert!((actual_gamma[0] - 9.0).abs() < 1e-6);
        assert!((actual_gamma[1] - 2.0).abs() < 1e-6);

        // rms_norm should not have a beta input.
        assert!(
            op.inputs.get("beta").is_none(),
            "rms_norm should not have beta"
        );
    }

    // -----------------------------------------------------------------------
    // Ops without matching stats are skipped
    // -----------------------------------------------------------------------

    #[test]
    fn dac_skips_ops_without_matching_stats() {
        let gamma = [2.0_f32, 3.0];
        let beta = [1.0_f32, -1.0];

        // Empty stats — no keys match.
        let pass = DacPass {
            fp16_stats: HashMap::new(),
            quant_stats: HashMap::new(),
        };

        let mut program = program_with_norm("layer_norm", "ln_0", &gamma, Some(&beta));
        pass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];

        let actual_gamma = match op.inputs.get("gamma") {
            Some(Value::Tensor { data, .. }) => {
                tensor_as_f32_slice(data.as_bytes().expect("tensor not materialized"))
            }
            _ => panic!("gamma should be a tensor"),
        };
        let actual_beta = match op.inputs.get("beta") {
            Some(Value::Tensor { data, .. }) => {
                tensor_as_f32_slice(data.as_bytes().expect("tensor not materialized"))
            }
            _ => panic!("beta should be a tensor"),
        };

        assert_eq!(actual_gamma, gamma.to_vec(), "gamma should be unchanged");
        assert_eq!(actual_beta, beta.to_vec(), "beta should be unchanged");
    }

    // -----------------------------------------------------------------------
    // Stats present only in fp16 (missing quant) → skip
    // -----------------------------------------------------------------------

    #[test]
    fn dac_skips_when_quant_stats_missing() {
        let gamma = [1.0_f32, 1.0];

        let mut fp16_stats = HashMap::new();
        fp16_stats.insert("ln_0".to_string(), stats(vec![0.0, 0.0], vec![1.0, 1.0]));

        let pass = DacPass {
            fp16_stats,
            quant_stats: HashMap::new(),
        };

        let mut program = program_with_norm("layer_norm", "ln_0", &gamma, Some(&[0.0, 0.0]));
        pass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        let actual_gamma = match op.inputs.get("gamma") {
            Some(Value::Tensor { data, .. }) => {
                tensor_as_f32_slice(data.as_bytes().expect("tensor not materialized"))
            }
            _ => panic!("gamma should be a tensor"),
        };
        assert_eq!(actual_gamma, gamma.to_vec(), "gamma should be unchanged");
    }

    // -----------------------------------------------------------------------
    // Non-norm ops are not touched
    // -----------------------------------------------------------------------

    #[test]
    fn dac_ignores_non_norm_ops() {
        let mut program = Program::new("1.0.0");
        let mut func = Function::new("main");
        func.body.add_op(
            Operation::new("relu", "relu_0")
                .with_input("x", Value::Reference("input".into()))
                .with_output("relu_out"),
        );
        func.body.outputs.push("relu_out".into());
        program.add_function(func);

        let mut fp16_stats = HashMap::new();
        fp16_stats.insert("relu_0".to_string(), stats(vec![0.0], vec![1.0]));
        let mut quant_stats = HashMap::new();
        quant_stats.insert("relu_0".to_string(), stats(vec![0.0], vec![1.0]));

        let pass = DacPass {
            fp16_stats,
            quant_stats,
        };

        pass.run(&mut program).unwrap();

        let op = &program.functions["main"].body.operations[0];
        assert_eq!(op.op_type, "relu");
    }
}
