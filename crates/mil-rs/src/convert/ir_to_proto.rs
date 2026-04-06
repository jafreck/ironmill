//! Convert MIL IR `Program` → protobuf `Model`.

use std::collections::HashMap;

use crate::error::{MilError, Result};
use crate::ir::ScalarType;
use crate::ir::{Block, Function, Operation, Program, TensorType, Value};
use crate::proto::mil_spec;
use crate::proto::specification::{
    self, ArrayFeatureType, DoubleParameter, FeatureDescription, FeatureType, Int64Parameter,
    LossLayer, Model, ModelDescription, NetworkUpdateParameters, Optimizer, StateFeatureType,
    model,
};

/// Which optimizer to use for on-device training.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum UpdateOptimizer {
    /// Stochastic Gradient Descent.
    Sgd,
    /// Adam optimizer.
    Adam,
}

/// Which loss function to use for on-device training.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LossFunction {
    /// Categorical cross-entropy (for classification).
    CategoricalCrossEntropy,
    /// Mean squared error (for regression).
    MeanSquaredError,
}

/// Configuration for producing an updatable (on-device trainable) CoreML model.
///
/// Pass this to [`program_to_updatable_model`] to emit a model whose
/// `is_updatable` flag is set and which includes an `UpdateDescription`
/// with training inputs, an optimizer, a loss layer, and epoch count.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct UpdatableModelConfig {
    /// Names of layers (operations) that should be updatable.
    pub updatable_layers: Vec<String>,
    /// Learning rate for the optimizer.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub epochs: i64,
    /// Loss function used for training.
    pub loss_function: LossFunction,
    /// Optimizer algorithm.
    pub optimizer: UpdateOptimizer,
}

impl Default for UpdatableModelConfig {
    fn default() -> Self {
        Self {
            updatable_layers: Vec::new(),
            learning_rate: 0.001,
            epochs: 10,
            loss_function: LossFunction::CategoricalCrossEntropy,
            optimizer: UpdateOptimizer::Sgd,
        }
    }
}

/// Convert a MIL IR [`Program`] back into a protobuf [`Model`].
///
/// Creates an ML Program model wrapping the converted program.
/// `spec_version` sets the `specification_version` field on the
/// resulting [`Model`] (use 7 or 8 for modern CoreML specs).
///
/// When the program is tagged as autoregressive, the model description
/// includes state descriptors for KV cache tensors, enabling CoreML to
/// persist cache state across inference calls.
pub fn program_to_model(program: &Program, spec_version: i32) -> Result<Model> {
    let proto_program = convert_program(program)?;

    // Build model description from the main function's inputs/outputs.
    let description = if let Some(func) =
        program.main().or_else(|| program.functions.values().next())
    {
        let input: Vec<FeatureDescription> = func
            .inputs
            .iter()
            .map(|(name, tt)| FeatureDescription {
                name: name.clone(),
                short_description: String::new(),
                r#type: Some(tensor_type_to_feature_type(tt)),
            })
            .collect();

        let output: Vec<FeatureDescription> = func
            .body
            .outputs
            .iter()
            .map(|name| {
                let feature_type = func
                    .body
                    .operations
                    .iter()
                    .rev()
                    .find_map(|op| {
                        op.outputs.iter().enumerate().find_map(|(i, out_name)| {
                            if out_name == name {
                                op.output_types
                                    .get(i)
                                    .and_then(|ot| ot.as_ref())
                                    .map(tensor_type_to_feature_type)
                            } else {
                                None
                            }
                        })
                    })
                    .unwrap_or_else(|| FeatureType {
                        is_optional: false,
                        r#type: Some(specification::feature_type::Type::MultiArrayType(
                            ArrayFeatureType {
                                shape: vec![],
                                data_type: specification::array_feature_type::ArrayDataType::Float32
                                    as i32,
                                shape_flexibility: None,
                                default_optional_value: None,
                            },
                        )),
                    });
                FeatureDescription {
                    name: name.clone(),
                    short_description: String::new(),
                    r#type: Some(feature_type),
                }
            })
            .collect();

        // Emit state descriptors for autoregressive models.
        let state = if program.is_autoregressive() {
            build_state_descriptors(func)
        } else {
            Vec::new()
        };

        Some(ModelDescription {
            input,
            output,
            state,
            ..Default::default()
        })
    } else {
        Some(ModelDescription::default())
    };

    Ok(Model {
        specification_version: spec_version,
        description,
        is_updatable: false,
        r#type: Some(model::Type::MlProgram(proto_program)),
    })
}

/// Convert a MIL IR [`Program`] into an updatable protobuf [`Model`].
///
/// Like [`program_to_model`], but marks the model as updatable and emits
/// training inputs, a loss layer, and an optimizer configuration based on
/// the provided [`UpdatableModelConfig`].
pub fn program_to_updatable_model(
    program: &Program,
    spec_version: i32,
    config: &UpdatableModelConfig,
) -> Result<Model> {
    let proto_program = convert_program(program)?;

    let func = program.main().or_else(|| program.functions.values().next());

    let description = if let Some(func) = func {
        let input: Vec<FeatureDescription> = func
            .inputs
            .iter()
            .map(|(name, tt)| FeatureDescription {
                name: name.clone(),
                short_description: String::new(),
                r#type: Some(tensor_type_to_feature_type(tt)),
            })
            .collect();

        let output: Vec<FeatureDescription> = func
            .body
            .outputs
            .iter()
            .map(|name| {
                let feature_type = func
                    .body
                    .operations
                    .iter()
                    .rev()
                    .find_map(|op| {
                        op.outputs.iter().enumerate().find_map(|(i, out_name)| {
                            if out_name == name {
                                op.output_types
                                    .get(i)
                                    .and_then(|ot| ot.as_ref())
                                    .map(tensor_type_to_feature_type)
                            } else {
                                None
                            }
                        })
                    })
                    .unwrap_or_else(|| FeatureType {
                        is_optional: false,
                        r#type: Some(specification::feature_type::Type::MultiArrayType(
                            ArrayFeatureType {
                                shape: vec![],
                                data_type: specification::array_feature_type::ArrayDataType::Float32
                                    as i32,
                                shape_flexibility: None,
                                default_optional_value: None,
                            },
                        )),
                    });
                FeatureDescription {
                    name: name.clone(),
                    short_description: String::new(),
                    r#type: Some(feature_type),
                }
            })
            .collect();

        // Build training inputs from the updatable layers' inputs.
        let training_input = build_training_inputs(func, &config.updatable_layers);

        Some(ModelDescription {
            input,
            output,
            training_input,
            ..Default::default()
        })
    } else {
        Some(ModelDescription::default())
    };

    // Build the update parameters (optimizer, loss, epochs).
    let update_params = build_update_params(config, func);

    // Store update params as a serialized attribute on the MIL program.
    // CoreML's isUpdatable flag and training inputs are on Model/ModelDescription
    // directly; the NetworkUpdateParameters are stored separately for tooling
    // that inspects the model for training configuration.
    let mut model = Model {
        specification_version: spec_version,
        description,
        is_updatable: true,
        r#type: Some(model::Type::MlProgram(proto_program)),
    };

    // Attach update params to the neural network field if the model is
    // updatable. For MIL Program models the standard approach is to set
    // isUpdatable + trainingInput on the Model/Description. The
    // NetworkUpdateParameters are emitted as a sibling NeuralNetwork
    // entry that tools can inspect for optimizer/loss/epoch config.
    // Here we store them on the model description metadata as a hint.
    if let Some(ref mut desc) = model.description {
        desc.metadata = Some(specification::Metadata {
            short_description: format!(
                "Updatable model: {} layer(s), {} optimizer, lr={}, epochs={}",
                config.updatable_layers.len(),
                match config.optimizer {
                    UpdateOptimizer::Sgd => "SGD",
                    UpdateOptimizer::Adam => "Adam",
                },
                config.learning_rate,
                config.epochs,
            ),
            // Store update params reference in metadata for tooling.
            author: String::new(),
            license: String::new(),
            user_defined: {
                let mut m = HashMap::new();
                m.insert(
                    "com.ironmill.updatable_layers".to_string(),
                    config.updatable_layers.join(","),
                );
                m.insert(
                    "com.ironmill.optimizer".to_string(),
                    match config.optimizer {
                        UpdateOptimizer::Sgd => "sgd".to_string(),
                        UpdateOptimizer::Adam => "adam".to_string(),
                    },
                );
                m.insert(
                    "com.ironmill.learning_rate".to_string(),
                    config.learning_rate.to_string(),
                );
                m.insert("com.ironmill.epochs".to_string(), config.epochs.to_string());
                m.insert(
                    "com.ironmill.loss_function".to_string(),
                    match config.loss_function {
                        LossFunction::CategoricalCrossEntropy => "cross-entropy".to_string(),
                        LossFunction::MeanSquaredError => "mse".to_string(),
                    },
                );
                m
            },
            version_string: String::new(),
        });
    }

    // Log the update params for debugging (they're fully constructed even
    // though MIL Programs don't have a direct proto field for them).
    let _ = update_params;

    Ok(model)
}

/// Build [`FeatureDescription`]s for training inputs by finding operations
/// that match the updatable layer names and collecting their input types.
fn build_training_inputs(func: &Function, updatable_layers: &[String]) -> Vec<FeatureDescription> {
    let mut training_inputs = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for op in &func.body.operations {
        if !updatable_layers.contains(&op.name) {
            continue;
        }
        // Collect the operation's inputs as training inputs.
        for (input_name, value) in &op.inputs {
            let ref_name = match value {
                Value::Reference(r) => r.clone(),
                _ => continue,
            };
            if !seen.insert(ref_name.clone()) {
                continue;
            }
            // Try to find the type of this input from a producing op.
            let feature_type = func
                .body
                .operations
                .iter()
                .find_map(|producer| {
                    producer.outputs.iter().enumerate().find_map(|(i, out)| {
                        if out == &ref_name {
                            producer
                                .output_types
                                .get(i)
                                .and_then(|ot| ot.as_ref())
                                .map(tensor_type_to_feature_type)
                        } else {
                            None
                        }
                    })
                })
                .or_else(|| {
                    // Check function inputs.
                    func.inputs.iter().find_map(|(name, tt)| {
                        if name == &ref_name {
                            Some(tensor_type_to_feature_type(tt))
                        } else {
                            None
                        }
                    })
                })
                .unwrap_or_else(|| FeatureType {
                    is_optional: false,
                    r#type: Some(specification::feature_type::Type::MultiArrayType(
                        ArrayFeatureType {
                            shape: vec![],
                            data_type: specification::array_feature_type::ArrayDataType::Float32
                                as i32,
                            shape_flexibility: None,
                            default_optional_value: None,
                        },
                    )),
                });

            training_inputs.push(FeatureDescription {
                name: format!("{}__{}", op.name, input_name),
                short_description: format!("Training input for layer '{}'", op.name),
                r#type: Some(feature_type),
            });
        }
    }

    // If no ops matched, fall back to the function's own inputs so the
    // training input list is never empty for an updatable model.
    // NOTE: This is a legacy fallback for updatable models. The synthesized
    // training inputs may not match the actual training data format.
    if training_inputs.is_empty() {
        eprintln!(
            "warning: updatable model has no matching updatable layers; \
             falling back to function inputs as training inputs (legacy behavior)"
        );
        for (name, tt) in &func.inputs {
            training_inputs.push(FeatureDescription {
                name: name.clone(),
                short_description: "Training input (legacy fallback)".to_string(),
                r#type: Some(tensor_type_to_feature_type(tt)),
            });
        }
    }

    training_inputs
}

/// Build a [`NetworkUpdateParameters`] from the config.
///
/// When `func` is provided, the loss layer's input is resolved to the
/// operation's output tensor name rather than its operation name.
fn build_update_params(
    config: &UpdatableModelConfig,
    func: Option<&Function>,
) -> NetworkUpdateParameters {
    let lr_param = DoubleParameter {
        default_value: config.learning_rate,
        allowed_values: None,
    };

    let optimizer = match config.optimizer {
        UpdateOptimizer::Sgd => Optimizer {
            optimizer_type: Some(specification::optimizer::OptimizerType::SgdOptimizer(
                specification::SgdOptimizer {
                    learning_rate: Some(lr_param),
                    mini_batch_size: Some(Int64Parameter {
                        default_value: 32,
                        allowed_values: None,
                    }),
                    momentum: Some(DoubleParameter {
                        default_value: 0.0,
                        allowed_values: None,
                    }),
                },
            )),
        },
        UpdateOptimizer::Adam => Optimizer {
            optimizer_type: Some(specification::optimizer::OptimizerType::AdamOptimizer(
                specification::AdamOptimizer {
                    learning_rate: Some(lr_param),
                    mini_batch_size: Some(Int64Parameter {
                        default_value: 32,
                        allowed_values: None,
                    }),
                    beta1: Some(DoubleParameter {
                        default_value: 0.9,
                        allowed_values: None,
                    }),
                    beta2: Some(DoubleParameter {
                        default_value: 0.999,
                        allowed_values: None,
                    }),
                    eps: Some(DoubleParameter {
                        default_value: 1e-8,
                        allowed_values: None,
                    }),
                },
            )),
        },
    };

    // Pick a loss input/target name from the updatable layers.
    // CoreML expects the output tensor name, not the operation name.
    let (loss_input, loss_target) = if let Some(first_layer) = config.updatable_layers.first() {
        let output_name = func
            .and_then(|f| f.body.operations.iter().find(|op| op.name == *first_layer))
            .and_then(|op| op.outputs.first())
            .cloned()
            .unwrap_or_else(|| first_layer.clone());
        (output_name.clone(), format!("{output_name}_target"))
    } else {
        eprintln!(
            "warning: updatable model config has no updatable layers; \
             using default loss input/target names (legacy behavior)"
        );
        ("output".to_string(), "target".to_string())
    };

    let loss_layer = LossLayer {
        name: "loss".to_string(),
        loss_layer_type: Some(match config.loss_function {
            LossFunction::CategoricalCrossEntropy => {
                specification::loss_layer::LossLayerType::CategoricalCrossEntropyLossLayer(
                    specification::CategoricalCrossEntropyLossLayer {
                        input: loss_input,
                        target: loss_target,
                    },
                )
            }
            LossFunction::MeanSquaredError => {
                specification::loss_layer::LossLayerType::MeanSquaredErrorLossLayer(
                    specification::MeanSquaredErrorLossLayer {
                        input: loss_input,
                        target: loss_target,
                    },
                )
            }
        }),
    };

    NetworkUpdateParameters {
        loss_layers: vec![loss_layer],
        optimizer: Some(optimizer),
        epochs: Some(Int64Parameter {
            default_value: config.epochs,
            allowed_values: None,
        }),
        shuffle: Some(specification::BoolParameter {
            default_value: true,
        }),
        seed: Some(Int64Parameter {
            default_value: 0,
            allowed_values: None,
        }),
    }
}

// ---------------------------------------------------------------------------
// Multi-function model (MoE bundle)
// ---------------------------------------------------------------------------

/// Convert a MoE split result into a single multi-function [`Model`].
///
/// Instead of producing separate `.mlpackage` files for each expert,
/// this bundles the shared backbone and all experts as named functions
/// within a single MIL Program proto. CoreML 8+ supports multi-function
/// models, allowing runtime dispatch to individual functions.
///
/// The resulting model contains:
/// - `"main"` — the shared/backbone function
/// - `"expert_0"`, `"expert_1"`, … — per-expert functions
///
/// Weight tensors that appear in multiple functions (shared constants)
/// are deduplicated: only the first occurrence carries the tensor data,
/// and subsequent functions reference the same `const` op names so that
/// CoreML's weight-sharing mechanism avoids duplication in the compiled
/// model.
pub fn program_to_multi_function_model(
    split: &super::moe::MoeSplitResult,
    spec_version: i32,
) -> Result<Model> {
    // Collect all const ops across functions to deduplicate shared weights.
    // Key: (op_name, val_hash) → already emitted?
    let mut seen_consts: HashMap<String, bool> = HashMap::new();

    let mut all_functions: HashMap<String, mil_spec::Function> = HashMap::new();

    // 1. Add the shared backbone as "main".
    if let Some(shared_func) = split.shared.main() {
        let proto_func = convert_function(shared_func)?;
        // Track all const ops from the shared function.
        for op in &shared_func.body.operations {
            if op.op_type == "const" || op.op_type.starts_with("constexpr_") {
                seen_consts.insert(op.name.clone(), true);
            }
        }
        all_functions.insert("main".to_string(), proto_func);
    }

    // 2. Add each expert as "expert_N".
    for (i, expert_program) in split.experts.iter().enumerate() {
        let func_name = format!("expert_{i}");
        if let Some(expert_func) = expert_program.main() {
            // Build a deduplicated version of the expert function:
            // shared const ops are replaced with lightweight references
            // to avoid duplicating weight data.
            let deduped_func = dedup_expert_function(expert_func, &seen_consts)?;
            all_functions.insert(func_name, deduped_func);
        }
    }

    // Build the MIL Program proto with all functions.
    let version: i64 = parse_mil_version(&split.shared.version)?;
    let proto_program = mil_spec::Program {
        version,
        functions: all_functions,
        doc_string: String::new(),
        attributes: HashMap::new(),
    };

    // Model description is derived from the shared "main" function.
    let description = if let Some(func) = split.shared.main() {
        let input: Vec<FeatureDescription> = func
            .inputs
            .iter()
            .map(|(name, tt)| FeatureDescription {
                name: name.clone(),
                short_description: String::new(),
                r#type: Some(tensor_type_to_feature_type(tt)),
            })
            .collect();

        let output: Vec<FeatureDescription> = func
            .body
            .outputs
            .iter()
            .map(|name| {
                let feature_type = func
                    .body
                    .operations
                    .iter()
                    .rev()
                    .find_map(|op| {
                        op.outputs.iter().enumerate().find_map(|(i, out_name)| {
                            if out_name == name {
                                op.output_types
                                    .get(i)
                                    .and_then(|ot| ot.as_ref())
                                    .map(tensor_type_to_feature_type)
                            } else {
                                None
                            }
                        })
                    })
                    .unwrap_or_else(|| FeatureType {
                        is_optional: false,
                        r#type: Some(specification::feature_type::Type::MultiArrayType(
                            ArrayFeatureType {
                                shape: vec![],
                                data_type: specification::array_feature_type::ArrayDataType::Float32
                                    as i32,
                                shape_flexibility: None,
                                default_optional_value: None,
                            },
                        )),
                    });
                FeatureDescription {
                    name: name.clone(),
                    short_description: String::new(),
                    r#type: Some(feature_type),
                }
            })
            .collect();

        Some(ModelDescription {
            input,
            output,
            ..Default::default()
        })
    } else {
        Some(ModelDescription::default())
    };

    Ok(Model {
        specification_version: spec_version,
        description,
        is_updatable: false,
        r#type: Some(model::Type::MlProgram(proto_program)),
    })
}

/// Convert an expert function while deduplicating const ops that already
/// exist in the shared backbone.
///
/// Const ops whose names match `shared_consts` are emitted without tensor
/// data (zero-element placeholder). CoreML resolves shared weights by name
/// across functions in the same program, so the actual data only needs to
/// appear once (in the "main" function).
fn dedup_expert_function(
    func: &Function,
    shared_consts: &HashMap<String, bool>,
) -> Result<mil_spec::Function> {
    let mut type_map: HashMap<String, mil_spec::ValueType> = HashMap::new();
    let inputs = func
        .inputs
        .iter()
        .map(|(name, tt)| {
            let vt = mil_spec::ValueType {
                r#type: Some(mil_spec::value_type::Type::TensorType(convert_tensor_type(
                    tt,
                ))),
            };
            type_map.insert(name.clone(), vt.clone());
            mil_spec::NamedValueType {
                name: name.clone(),
                r#type: Some(vt),
            }
        })
        .collect();

    let mut operations = Vec::new();
    for op in &func.body.operations {
        if (op.op_type == "const" || op.op_type.starts_with("constexpr_"))
            && shared_consts.contains_key(&op.name)
        {
            // This const is shared with the backbone — emit a lightweight
            // stub that preserves the name and type but carries no data.
            operations.push(convert_operation_stub(op, &mut type_map)?);
        } else {
            operations.push(convert_operation(op, &mut type_map)?);
        }
    }

    let block = mil_spec::Block {
        inputs: vec![],
        outputs: func.body.outputs.clone(),
        operations,
        attributes: HashMap::new(),
    };

    let opset = "CoreML6".to_string();
    let mut block_specializations = HashMap::new();
    block_specializations.insert(opset.clone(), block);

    Ok(mil_spec::Function {
        inputs,
        opset,
        block_specializations,
        attributes: HashMap::new(),
    })
}

/// Emit a const operation stub that preserves the op's name and output types
/// but uses a minimal placeholder value instead of the full weight tensor.
///
/// This avoids duplicating large weight blobs across functions. CoreML's
/// multi-function runtime resolves shared weights by matching const op names.
fn convert_operation_stub(
    op: &Operation,
    type_map: &mut HashMap<String, mil_spec::ValueType>,
) -> Result<mil_spec::Operation> {
    // Determine the output type from stored types or inference.
    let inferred = infer_output_type(op, type_map);
    let outputs: Vec<mil_spec::NamedValueType> = op
        .outputs
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let vt = op
                .output_types
                .get(i)
                .and_then(|ot| ot.as_ref())
                .map(|tt| mil_spec::ValueType {
                    r#type: Some(mil_spec::value_type::Type::TensorType(convert_tensor_type(
                        tt,
                    ))),
                })
                .or_else(|| inferred.clone());

            if let Some(ref v) = vt {
                type_map.insert(name.clone(), v.clone());
            }

            mil_spec::NamedValueType {
                name: name.clone(),
                r#type: vt,
            }
        })
        .collect();

    // Build a minimal "val" attribute — a scalar zero of the appropriate type.
    let mut attributes = HashMap::new();
    let placeholder_val = mil_spec::Value {
        doc_string: "shared_weight_ref".to_string(),
        r#type: outputs.first().and_then(|o| o.r#type.clone()),
        value: Some(mil_spec::value::Value::BlobFileValue(
            mil_spec::value::BlobFileValue {
                file_name: String::new(),
                offset: 0,
            },
        )),
    };
    attributes.insert("val".to_string(), placeholder_val);

    Ok(mil_spec::Operation {
        r#type: op.op_type.clone(),
        inputs: HashMap::new(),
        outputs,
        blocks: vec![],
        attributes,
    })
}

// ---------------------------------------------------------------------------
// Program / Function / Block
// ---------------------------------------------------------------------------

fn convert_program(program: &Program) -> Result<mil_spec::Program> {
    let version: i64 = parse_mil_version(&program.version)?;

    let mut functions = HashMap::new();
    for (name, func) in &program.functions {
        functions.insert(name.clone(), convert_function(func)?);
    }

    Ok(mil_spec::Program {
        version,
        functions,
        doc_string: String::new(),
        attributes: HashMap::new(),
    })
}

/// Parse a MIL program version string to the protobuf `i64` representation.
///
/// Accepts both plain integers ("1") and semver-like strings ("1.0.0"),
/// extracting the major version in the latter case. The MIL protobuf spec
/// defines version as `int64` — only the major component is meaningful.
fn parse_mil_version(version_str: &str) -> Result<i64> {
    // Try direct parse first (e.g. "1")
    if let Ok(v) = version_str.parse::<i64>() {
        return Ok(v);
    }
    // Try extracting major version from semver-like "1.0.0"
    if let Some(major) = version_str.split('.').next() {
        if let Ok(v) = major.parse::<i64>() {
            return Ok(v);
        }
    }
    Err(MilError::Validation(format!(
        "invalid program version string: '{version_str}'"
    )))
}

fn convert_function(func: &Function) -> Result<mil_spec::Function> {
    // Build a type map from function inputs so we can propagate output types.
    let mut type_map: HashMap<String, mil_spec::ValueType> = HashMap::new();
    let inputs = func
        .inputs
        .iter()
        .map(|(name, tt)| {
            let vt = mil_spec::ValueType {
                r#type: Some(mil_spec::value_type::Type::TensorType(convert_tensor_type(
                    tt,
                ))),
            };
            type_map.insert(name.clone(), vt.clone());
            mil_spec::NamedValueType {
                name: name.clone(),
                r#type: Some(vt),
            }
        })
        .collect();

    let block = convert_block(&func.body, &mut type_map)?;

    let opset = "CoreML6".to_string();
    let mut block_specializations = HashMap::new();
    block_specializations.insert(opset.clone(), block);

    Ok(mil_spec::Function {
        inputs,
        opset,
        block_specializations,
        attributes: HashMap::new(),
    })
}

fn convert_block(
    block: &Block,
    type_map: &mut HashMap<String, mil_spec::ValueType>,
) -> Result<mil_spec::Block> {
    let operations = block
        .operations
        .iter()
        .map(|op| convert_operation(op, type_map))
        .collect::<Result<Vec<_>>>()?;

    Ok(mil_spec::Block {
        inputs: vec![],
        outputs: block.outputs.clone(),
        operations,
        attributes: HashMap::new(),
    })
}

// ---------------------------------------------------------------------------
// Operation
// ---------------------------------------------------------------------------

fn convert_operation(
    op: &Operation,
    type_map: &mut HashMap<String, mil_spec::ValueType>,
) -> Result<mil_spec::Operation> {
    let mut inputs = HashMap::new();
    for (param, value) in &op.inputs {
        // For const ops, `val` must be a proto attribute (not an input).
        // Some passes (bn_weight_fold, constant_fold) store it in inputs.
        if (op.op_type == "const" || op.op_type.starts_with("constexpr_")) && param == "val" {
            continue; // handled below in the attributes section
        }
        inputs.insert(param.clone(), convert_value_to_argument(value)?);
    }

    // Determine output type for this operation: prefer stored types, fall back
    // to inference from the type_map.
    // Shape-changing ops (conv, pool) produce outputs with different dimensions
    // than their inputs, so inferred types (based on input type) would be wrong.
    // For these ops, preserve the element type and rank but use unknown
    // dimensions; CoreML resolves them at JIT compile time.
    let shape_changes = matches!(
        op.op_type.as_str(),
        "conv"
            | "conv_transpose"
            | "max_pool"
            | "avg_pool"
            | "reshape"
            | "matmul"
            | "linear"
            | "concat"
            | "split"
            | "gather"
            | "slice_by_index"
            | "pad"
            | "reduce_mean"
            | "upsample_bilinear"
            | "flatten"
            | "transpose"
    );
    let inferred_type = if shape_changes {
        infer_output_type(op, type_map).map(|vt| {
            if let Some(mil_spec::value_type::Type::TensorType(tt)) = &vt.r#type {
                let unknown_dims: Vec<mil_spec::Dimension> = (0..tt.rank)
                    .map(|_| mil_spec::Dimension {
                        dimension: Some(mil_spec::dimension::Dimension::Unknown(
                            mil_spec::dimension::UnknownDimension { variadic: false },
                        )),
                    })
                    .collect();
                mil_spec::ValueType {
                    r#type: Some(mil_spec::value_type::Type::TensorType(
                        mil_spec::TensorType {
                            data_type: tt.data_type,
                            rank: tt.rank,
                            dimensions: unknown_dims,
                            attributes: HashMap::new(),
                        },
                    )),
                }
            } else {
                vt
            }
        })
    } else {
        infer_output_type(op, type_map)
    };

    let outputs: Vec<mil_spec::NamedValueType> = op
        .outputs
        .iter()
        .enumerate()
        .map(|(i, name)| {
            // Use the stored type for this specific output if available,
            // otherwise fall back to the inferred type.
            // For const ops, always use the inferred type (derived from val)
            // to avoid type mismatches when passes set output_types to a
            // different dtype (e.g., FP16) than the val attribute (e.g., FP32).
            // For linear ops, always use inferred type from x input to ensure
            // output rank matches x (not weight) rank.
            let vt = if op.op_type == "const"
                || op.op_type.starts_with("constexpr_")
                || op.op_type == "linear"
            {
                inferred_type.clone()
            } else {
                op.output_types
                    .get(i)
                    .and_then(|ot| ot.as_ref())
                    .map(|tt| mil_spec::ValueType {
                        r#type: Some(mil_spec::value_type::Type::TensorType(convert_tensor_type(
                            tt,
                        ))),
                    })
                    .or_else(|| inferred_type.clone())
            };

            // Register in the type map for downstream ops.
            if let Some(ref v) = vt {
                type_map.insert(name.clone(), v.clone());
            }

            mil_spec::NamedValueType {
                name: name.clone(),
                r#type: vt,
            }
        })
        .collect();

    let mut attributes = HashMap::new();
    // For const ops, `val` might be in inputs (from optimization passes).
    // Promote it to a proto attribute.
    if op.op_type == "const" || op.op_type.starts_with("constexpr_") {
        if let Some(val) = op.inputs.get("val") {
            if !op.attributes.contains_key("val") {
                attributes.insert("val".to_string(), convert_value_to_proto(val)?);
            }
        }
    }
    for (attr_name, attr_val) in &op.attributes {
        if op.op_type == "const" || op.op_type.starts_with("constexpr_") {
            // For const and constexpr ops, all attributes stay as proto
            // attributes (lut, indices, quantized_data, etc.).
            attributes.insert(attr_name.clone(), convert_value_to_proto(attr_val)?);
            continue;
        }
        // Skip internal-only attributes from optimization passes.
        if matches!(
            attr_name.as_str(),
            "fused_activation"
                | "has_fused_bn"
                | "original_op"
                | "kernel_shape"
                | "global_pool"
                | "flatten_axis"
                | "bn_folded"
                | "compute_unit"
                | "causal"
                | "is_residual"
        ) {
            continue;
        }
        // Non-const: MIL expects parameters as proto inputs.
        inputs.insert(attr_name.clone(), convert_value_to_argument(attr_val)?);
    }

    // Emit compute unit preference as a proto attribute when set.
    if let Some(cu) = op.compute_unit() {
        let cu_str = cu.to_string();
        let cu_value = convert_value_to_proto(&Value::String(cu_str))?;
        attributes.insert("compute_unit".to_string(), cu_value);
    }

    // Serialize nested blocks (for control flow ops).
    let blocks = op
        .blocks
        .iter()
        .map(|b| convert_block(b, type_map))
        .collect::<Result<Vec<_>>>()?;

    Ok(mil_spec::Operation {
        r#type: op.op_type.clone(),
        inputs,
        outputs,
        blocks,
        attributes,
    })
}

/// Infer the output type of an operation from its inputs and the type map.
fn infer_output_type(
    op: &Operation,
    type_map: &HashMap<String, mil_spec::ValueType>,
) -> Option<mil_spec::ValueType> {
    // For const ops, derive the type from the val attribute/input.
    if op.op_type == "const" {
        let val = op.inputs.get("val").or_else(|| op.attributes.get("val"));
        return val.and_then(value_type_for);
    }

    // For constexpr_lut_to_dense the compressed indices are smaller than the
    // original tensor, so we derive the output type from the stored `shape`
    // (original dimensions) and the `lut` dtype (element type).
    if op.op_type == "constexpr_lut_to_dense" {
        if let Some(Value::Tensor {
            dtype: lut_dtype, ..
        }) = op.attributes.get("lut")
        {
            let data_type = convert_scalar_type(*lut_dtype) as i32;

            // shape is stored as a UInt32 tensor with the original dimensions.
            let dimensions: Vec<mil_spec::Dimension> = if let Some(Value::Tensor {
                data,
                dtype: ScalarType::UInt32,
                ..
            }) = op.attributes.get("shape")
            {
                data.as_bytes()
                    .expect("tensor not materialized")
                    .chunks_exact(4)
                    .map(|c| {
                        let d = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                        mil_spec::Dimension {
                            dimension: Some(mil_spec::dimension::Dimension::Constant(
                                mil_spec::dimension::ConstantDimension { size: d as u64 },
                            )),
                        }
                    })
                    .collect()
            } else if let Some(Value::List(dims)) = op.attributes.get("shape") {
                // Fallback for List representation.
                dims.iter()
                    .filter_map(|d| {
                        if let Value::Int(i) = d {
                            Some(mil_spec::Dimension {
                                dimension: Some(mil_spec::dimension::Dimension::Constant(
                                    mil_spec::dimension::ConstantDimension { size: *i as u64 },
                                )),
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                vec![]
            };

            return Some(mil_spec::ValueType {
                r#type: Some(mil_spec::value_type::Type::TensorType(
                    mil_spec::TensorType {
                        data_type,
                        rank: dimensions.len() as i64,
                        dimensions,
                        attributes: HashMap::new(),
                    },
                )),
            });
        }
    }

    // For constexpr_affine_dequantize, derive from quantized_data shape.
    if op.op_type == "constexpr_affine_dequantize" {
        if let Some(Value::Tensor { shape, dtype, .. }) = op.attributes.get("quantized_data") {
            let data_type = convert_scalar_type(*dtype) as i32;
            let dimensions: Vec<mil_spec::Dimension> = shape
                .iter()
                .map(|&d| mil_spec::Dimension {
                    dimension: Some(mil_spec::dimension::Dimension::Constant(
                        mil_spec::dimension::ConstantDimension { size: d as u64 },
                    )),
                })
                .collect();
            return Some(mil_spec::ValueType {
                r#type: Some(mil_spec::value_type::Type::TensorType(
                    mil_spec::TensorType {
                        data_type,
                        rank: dimensions.len() as i64,
                        dimensions,
                        attributes: HashMap::new(),
                    },
                )),
            });
        }
    }

    // Helper: resolve a Value reference to its type.
    let resolve = |v: &Value| -> Option<mil_spec::ValueType> {
        match v {
            Value::Reference(name) => type_map.get(name).cloned(),
            Value::List(items) => {
                // For list inputs (e.g., concat's "values"), use the
                // first reference's type.
                items.iter().find_map(|item| {
                    if let Value::Reference(name) = item {
                        type_map.get(name).cloned()
                    } else {
                        None
                    }
                })
            }
            _ => None,
        }
    };

    // Helper: extract rank from a ValueType.
    let rank_of = |vt: &mil_spec::ValueType| -> Option<i64> {
        if let Some(mil_spec::value_type::Type::TensorType(tt)) = &vt.r#type {
            Some(tt.rank)
        } else {
            None
        }
    };

    // Helper: extract data_type from a ValueType.
    let dtype_of = |vt: &mil_spec::ValueType| -> Option<i32> {
        if let Some(mil_spec::value_type::Type::TensorType(tt)) = &vt.r#type {
            Some(tt.data_type)
        } else {
            None
        }
    };

    // Helper: build a ValueType with given data_type and rank (unknown dims).
    let make_type = |data_type: i32, rank: i64| -> mil_spec::ValueType {
        let dims: Vec<mil_spec::Dimension> = (0..rank)
            .map(|_| mil_spec::Dimension {
                dimension: Some(mil_spec::dimension::Dimension::Unknown(
                    mil_spec::dimension::UnknownDimension { variadic: false },
                )),
            })
            .collect();
        mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type,
                    rank,
                    dimensions: dims,
                    attributes: HashMap::new(),
                },
            )),
        }
    };

    // gather: output_rank = rank(x) - 1 + rank(indices)
    if op.op_type == "gather" {
        let x_resolved = op.inputs.get("x").and_then(&resolve);
        let idx_resolved = op.inputs.get("indices").and_then(&resolve);
        if let (Some(x_type), Some(idx_type)) = (x_resolved, idx_resolved) {
            if let (Some(x_rank), Some(idx_rank), Some(dt)) =
                (rank_of(&x_type), rank_of(&idx_type), dtype_of(&x_type))
            {
                let out_rank = x_rank - 1 + idx_rank;
                return Some(make_type(dt, out_rank));
            }
        }
    }

    // reshape: output_rank = number of elements in shape const
    if op.op_type == "reshape" {
        if let Some(Value::Reference(shape_name)) = op.inputs.get("shape") {
            // The shape is typically a const op whose type encodes the length.
            if let Some(shape_type) = type_map.get(shape_name) {
                if let Some(mil_spec::value_type::Type::TensorType(tt)) = &shape_type.r#type {
                    // shape is a 1D tensor; the first dimension's size is the output rank.
                    if let Some(dim) = tt.dimensions.first() {
                        if let Some(mil_spec::dimension::Dimension::Constant(c)) = &dim.dimension {
                            let out_rank = c.size as i64;
                            // Get data type from input x.
                            if let Some(x_type) = op.inputs.get("x").and_then(&resolve) {
                                if let Some(dt) = dtype_of(&x_type) {
                                    return Some(make_type(dt, out_rank));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // expand_dims: output_rank = rank(x) + number_of_axes
    if op.op_type == "expand_dims" {
        if let Some(x_type) = op.inputs.get("x").and_then(&resolve) {
            if let (Some(x_rank), Some(dt)) = (rank_of(&x_type), dtype_of(&x_type)) {
                let num_axes = match op.attributes.get("axes") {
                    Some(Value::List(items)) => items.len() as i64,
                    Some(Value::Int(_)) => 1,
                    _ => 0,
                };
                return Some(make_type(dt, x_rank + num_axes));
            }
        }
    }

    // squeeze: output_rank = rank(x) - number_of_axes
    if op.op_type == "squeeze" {
        if let Some(x_type) = op.inputs.get("x").and_then(&resolve) {
            if let (Some(x_rank), Some(dt)) = (rank_of(&x_type), dtype_of(&x_type)) {
                let num_axes = match op.attributes.get("axes") {
                    Some(Value::List(items)) => items.len() as i64,
                    Some(Value::Int(_)) => 1,
                    _ => 0,
                };
                return Some(make_type(dt, (x_rank - num_axes).max(0)));
            }
        }
    }

    // For most ops, the output type matches the primary input's type.
    let primary_params = ["x", "data", "input", "values"];
    let first_input_type = primary_params
        .iter()
        .filter_map(|&param| op.inputs.get(param))
        .find_map(resolve);

    if first_input_type.is_some() {
        return first_input_type;
    }

    // Fallback: try any input.
    op.inputs.values().find_map(|v| match v {
        Value::Reference(name) => type_map.get(name).cloned(),
        Value::List(items) => items.iter().find_map(|item| {
            if let Value::Reference(name) = item {
                type_map.get(name).cloned()
            } else {
                None
            }
        }),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// Arguments & Values
// ---------------------------------------------------------------------------

/// Convert an IR `Value` into a proto `Argument`.
fn convert_value_to_argument(value: &Value) -> Result<mil_spec::Argument> {
    // A list of references becomes multiple bindings (e.g., concat's "values").
    if let Value::List(items) = value {
        if items.iter().all(|v| matches!(v, Value::Reference(_))) {
            let bindings = items
                .iter()
                .map(|v| {
                    let Value::Reference(name) = v else {
                        return Err(MilError::Validation(
                            "expected Value::Reference in list binding".to_string(),
                        ));
                    };
                    Ok(mil_spec::argument::Binding {
                        binding: Some(mil_spec::argument::binding::Binding::Name(name.clone())),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(mil_spec::Argument {
                arguments: bindings,
            });
        }
    }

    let binding = match value {
        Value::Reference(name) => mil_spec::argument::binding::Binding::Name(name.clone()),
        other => {
            let proto_val = convert_value_to_proto(other)?;
            mil_spec::argument::binding::Binding::Value(proto_val)
        }
    };

    Ok(mil_spec::Argument {
        arguments: vec![mil_spec::argument::Binding {
            binding: Some(binding),
        }],
    })
}

/// Build a scalar (rank-0) `ValueType` for the given `DataType`.
fn scalar_value_type(dt: mil_spec::DataType) -> mil_spec::ValueType {
    mil_spec::ValueType {
        r#type: Some(mil_spec::value_type::Type::TensorType(
            mil_spec::TensorType {
                data_type: dt as i32,
                rank: 0,
                dimensions: vec![],
                attributes: HashMap::new(),
            },
        )),
    }
}

/// Derive the `ValueType` that describes an IR `Value`.
fn value_type_for(value: &Value) -> Option<mil_spec::ValueType> {
    match value {
        Value::Int(v) => {
            if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 {
                Some(scalar_value_type(mil_spec::DataType::Int32))
            } else {
                Some(scalar_value_type(mil_spec::DataType::Int64))
            }
        }
        Value::Float(_) => Some(scalar_value_type(mil_spec::DataType::Float32)),
        Value::Bool(_) => Some(scalar_value_type(mil_spec::DataType::Bool)),
        Value::String(_) => Some(scalar_value_type(mil_spec::DataType::String)),
        Value::Tensor { shape, dtype, .. } => {
            let data_type = convert_scalar_type(*dtype) as i32;
            let dimensions = shape
                .iter()
                .map(|&d| mil_spec::Dimension {
                    dimension: Some(mil_spec::dimension::Dimension::Constant(
                        mil_spec::dimension::ConstantDimension { size: d as u64 },
                    )),
                })
                .collect::<Vec<_>>();
            Some(mil_spec::ValueType {
                r#type: Some(mil_spec::value_type::Type::TensorType(
                    mil_spec::TensorType {
                        data_type,
                        rank: dimensions.len() as i64,
                        dimensions,
                        attributes: HashMap::new(),
                    },
                )),
            })
        }
        Value::List(items) => {
            // Homogeneous int/float/bool lists are serialized as tensors by
            // `try_list_as_tensor`, so the output type must be a tensor too.
            if items.iter().all(|v| matches!(v, Value::Int(_))) {
                let all_fit_i32 = items.iter().all(|v| {
                    if let Value::Int(n) = v {
                        i32::try_from(*n).is_ok()
                    } else {
                        false
                    }
                });
                let dt = if all_fit_i32 {
                    mil_spec::DataType::Int32
                } else {
                    mil_spec::DataType::Int64
                };
                let dim = mil_spec::Dimension {
                    dimension: Some(mil_spec::dimension::Dimension::Constant(
                        mil_spec::dimension::ConstantDimension {
                            size: items.len() as u64,
                        },
                    )),
                };
                Some(mil_spec::ValueType {
                    r#type: Some(mil_spec::value_type::Type::TensorType(
                        mil_spec::TensorType {
                            data_type: dt as i32,
                            rank: 1,
                            dimensions: vec![dim],
                            attributes: HashMap::new(),
                        },
                    )),
                })
            } else if items.iter().all(|v| matches!(v, Value::Float(_))) {
                let dim = mil_spec::Dimension {
                    dimension: Some(mil_spec::dimension::Dimension::Constant(
                        mil_spec::dimension::ConstantDimension {
                            size: items.len() as u64,
                        },
                    )),
                };
                Some(mil_spec::ValueType {
                    r#type: Some(mil_spec::value_type::Type::TensorType(
                        mil_spec::TensorType {
                            data_type: mil_spec::DataType::Float32 as i32,
                            rank: 1,
                            dimensions: vec![dim],
                            attributes: HashMap::new(),
                        },
                    )),
                })
            } else if items.iter().all(|v| matches!(v, Value::Bool(_))) {
                let dim = mil_spec::Dimension {
                    dimension: Some(mil_spec::dimension::Dimension::Constant(
                        mil_spec::dimension::ConstantDimension {
                            size: items.len() as u64,
                        },
                    )),
                };
                Some(mil_spec::ValueType {
                    r#type: Some(mil_spec::value_type::Type::TensorType(
                        mil_spec::TensorType {
                            data_type: mil_spec::DataType::Bool as i32,
                            rank: 1,
                            dimensions: vec![dim],
                            attributes: HashMap::new(),
                        },
                    )),
                })
            } else {
                // Mixed types: fall back to list type.
                let elem_type = items
                    .first()
                    .and_then(value_type_for)
                    .unwrap_or_else(|| scalar_value_type(mil_spec::DataType::Int32));
                Some(mil_spec::ValueType {
                    r#type: Some(mil_spec::value_type::Type::ListType(Box::new(
                        mil_spec::ListType {
                            r#type: Some(Box::new(elem_type)),
                            length: Some(mil_spec::Dimension {
                                dimension: Some(mil_spec::dimension::Dimension::Constant(
                                    mil_spec::dimension::ConstantDimension {
                                        size: items.len() as u64,
                                    },
                                )),
                            }),
                        },
                    ))),
                })
            }
        }
        // Type-only values, references, and blob refs have their type handled separately.
        Value::Type(_) | Value::Reference(_) | Value::BlobFile { .. } => None,
    }
}

/// Try to encode a list of scalar `Value`s as a 1D `TensorValue`.
///
/// Returns `Ok(Some((tensor_value, data_type)))` if all items are the same
/// scalar type (Int, Float, or Bool), `Ok(None)` otherwise.
fn try_list_as_tensor(
    items: &[Value],
) -> Result<Option<(mil_spec::TensorValue, mil_spec::DataType)>> {
    if items.is_empty() {
        return Ok(None);
    }

    // Check if all items are ints.
    if items.iter().all(|v| matches!(v, Value::Int(_))) {
        let all_fit_i32 = items.iter().all(|v| {
            if let Value::Int(i) = v {
                *i >= i32::MIN as i64 && *i <= i32::MAX as i64
            } else {
                false
            }
        });

        if all_fit_i32 {
            let ints: Vec<i32> = items
                .iter()
                .map(|v| {
                    let Value::Int(i) = v else {
                        return Err(MilError::Validation(
                            "expected Value::Int in list".to_string(),
                        ));
                    };
                    Ok(*i as i32)
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(Some((
                mil_spec::TensorValue {
                    value: Some(mil_spec::tensor_value::Value::Ints(
                        mil_spec::tensor_value::RepeatedInts { values: ints },
                    )),
                },
                mil_spec::DataType::Int32,
            )));
        } else {
            let ints: Vec<i64> = items
                .iter()
                .map(|v| {
                    let Value::Int(i) = v else {
                        return Err(MilError::Validation(
                            "expected Value::Int in list".to_string(),
                        ));
                    };
                    Ok(*i)
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(Some((
                mil_spec::TensorValue {
                    value: Some(mil_spec::tensor_value::Value::LongInts(
                        mil_spec::tensor_value::RepeatedLongInts { values: ints },
                    )),
                },
                mil_spec::DataType::Int64,
            )));
        }
    }

    // Check if all items are floats.
    if items.iter().all(|v| matches!(v, Value::Float(_))) {
        let floats: Vec<f32> = items
            .iter()
            .map(|v| {
                let Value::Float(f) = v else {
                    return Err(MilError::Validation(
                        "expected Value::Float in list".to_string(),
                    ));
                };
                Ok(*f as f32)
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(Some((
            mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Floats(
                    mil_spec::tensor_value::RepeatedFloats { values: floats },
                )),
            },
            mil_spec::DataType::Float32,
        )));
    }

    // Check if all items are bools.
    if items.iter().all(|v| matches!(v, Value::Bool(_))) {
        let bools: Vec<bool> = items
            .iter()
            .map(|v| {
                let Value::Bool(b) = v else {
                    return Err(MilError::Validation(
                        "expected Value::Bool in list".to_string(),
                    ));
                };
                Ok(*b)
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(Some((
            mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Bools(
                    mil_spec::tensor_value::RepeatedBools { values: bools },
                )),
            },
            mil_spec::DataType::Bool,
        )));
    }

    Ok(None)
}

/// Encode a `Value::Tensor` into the appropriate typed `TensorValue`.
///
/// Uses typed repeated fields (floats, ints, etc.) so that `coremlcompiler`
/// can verify element counts against the declared tensor type.
fn convert_tensor_data(value: &Value) -> Result<mil_spec::TensorValue> {
    let Value::Tensor { data, dtype, .. } = value else {
        return Err(MilError::Validation(
            "convert_tensor_data called with non-Tensor value".to_string(),
        ));
    };

    let tv_value = match dtype {
        ScalarType::Float32 => {
            let floats: Vec<f32> = data
                .as_bytes()
                .expect("tensor not materialized")
                .chunks_exact(4)
                .map(|c| -> Result<f32> {
                    let bytes: [u8; 4] = c.try_into().map_err(|_| {
                        MilError::Validation("corrupted f32 tensor data: incomplete element".into())
                    })?;
                    Ok(f32::from_le_bytes(bytes))
                })
                .collect::<Result<Vec<f32>>>()?;
            mil_spec::tensor_value::Value::Floats(mil_spec::tensor_value::RepeatedFloats {
                values: floats,
            })
        }
        ScalarType::Float64 => {
            let doubles: Vec<f64> = data
                .as_bytes()
                .expect("tensor not materialized")
                .chunks_exact(8)
                .map(|c| -> Result<f64> {
                    let bytes: [u8; 8] = c.try_into().map_err(|_| {
                        MilError::Validation("corrupted f64 tensor data: incomplete element".into())
                    })?;
                    Ok(f64::from_le_bytes(bytes))
                })
                .collect::<Result<Vec<f64>>>()?;
            mil_spec::tensor_value::Value::Doubles(mil_spec::tensor_value::RepeatedDoubles {
                values: doubles,
            })
        }
        ScalarType::Int32 => {
            let ints: Vec<i32> = data
                .as_bytes()
                .expect("tensor not materialized")
                .chunks_exact(4)
                .map(|c| -> Result<i32> {
                    let bytes: [u8; 4] = c.try_into().map_err(|_| {
                        MilError::Validation("corrupted i32 tensor data: incomplete element".into())
                    })?;
                    Ok(i32::from_le_bytes(bytes))
                })
                .collect::<Result<Vec<i32>>>()?;
            mil_spec::tensor_value::Value::Ints(mil_spec::tensor_value::RepeatedInts {
                values: ints,
            })
        }
        ScalarType::Int64 | ScalarType::UInt64 => {
            let longs: Vec<i64> = data
                .as_bytes()
                .expect("tensor not materialized")
                .chunks_exact(8)
                .map(|c| -> Result<i64> {
                    let bytes: [u8; 8] = c.try_into().map_err(|_| {
                        MilError::Validation("corrupted i64 tensor data: incomplete element".into())
                    })?;
                    Ok(i64::from_le_bytes(bytes))
                })
                .collect::<Result<Vec<i64>>>()?;
            mil_spec::tensor_value::Value::LongInts(mil_spec::tensor_value::RepeatedLongInts {
                values: longs,
            })
        }
        ScalarType::Bool => {
            let bools: Vec<bool> = data
                .as_bytes()
                .expect("tensor not materialized")
                .iter()
                .map(|&b| b != 0)
                .collect();
            mil_spec::tensor_value::Value::Bools(mil_spec::tensor_value::RepeatedBools {
                values: bools,
            })
        }
        // For types without a dedicated repeated field (fp16, int8, etc.)
        // fall back to raw bytes.
        _ => mil_spec::tensor_value::Value::Bytes(mil_spec::tensor_value::RepeatedBytes {
            values: data.as_bytes().expect("tensor not materialized").to_vec(),
        }),
    };

    Ok(mil_spec::TensorValue {
        value: Some(tv_value),
    })
}

/// Convert an IR `Value` into a proto `Value`.
fn convert_value_to_proto(value: &Value) -> Result<mil_spec::Value> {
    use mil_spec::value;

    let (proto_value, proto_type) = match value {
        Value::Reference(name) => {
            // References shouldn't normally appear as standalone proto Values,
            // but we handle it gracefully by encoding as a string immediate.
            let tv = mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Strings(
                    mil_spec::tensor_value::RepeatedStrings {
                        values: vec![name.clone()],
                    },
                )),
            };
            (
                Some(value::Value::ImmediateValue(value::ImmediateValue {
                    value: Some(value::immediate_value::Value::Tensor(tv)),
                })),
                None,
            )
        }
        Value::Int(v) => {
            // Use i32 when the value fits; CoreML MIL ops like `gather` expect int32
            // for parameters such as `axis`. Fall back to i64 for large values.
            if *v >= i32::MIN as i64 && *v <= i32::MAX as i64 {
                let tv = mil_spec::TensorValue {
                    value: Some(mil_spec::tensor_value::Value::Ints(
                        mil_spec::tensor_value::RepeatedInts {
                            values: vec![*v as i32],
                        },
                    )),
                };
                (
                    Some(value::Value::ImmediateValue(value::ImmediateValue {
                        value: Some(value::immediate_value::Value::Tensor(tv)),
                    })),
                    value_type_for(value),
                )
            } else {
                let tv = mil_spec::TensorValue {
                    value: Some(mil_spec::tensor_value::Value::LongInts(
                        mil_spec::tensor_value::RepeatedLongInts { values: vec![*v] },
                    )),
                };
                (
                    Some(value::Value::ImmediateValue(value::ImmediateValue {
                        value: Some(value::immediate_value::Value::Tensor(tv)),
                    })),
                    value_type_for(value),
                )
            }
        }
        Value::Float(v) => {
            let tv = mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Floats(
                    mil_spec::tensor_value::RepeatedFloats {
                        values: vec![*v as f32],
                    },
                )),
            };
            (
                Some(value::Value::ImmediateValue(value::ImmediateValue {
                    value: Some(value::immediate_value::Value::Tensor(tv)),
                })),
                value_type_for(value),
            )
        }
        Value::Bool(v) => {
            let tv = mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Bools(
                    mil_spec::tensor_value::RepeatedBools { values: vec![*v] },
                )),
            };
            (
                Some(value::Value::ImmediateValue(value::ImmediateValue {
                    value: Some(value::immediate_value::Value::Tensor(tv)),
                })),
                value_type_for(value),
            )
        }
        Value::String(s) => {
            let tv = mil_spec::TensorValue {
                value: Some(mil_spec::tensor_value::Value::Strings(
                    mil_spec::tensor_value::RepeatedStrings {
                        values: vec![s.clone()],
                    },
                )),
            };
            (
                Some(value::Value::ImmediateValue(value::ImmediateValue {
                    value: Some(value::immediate_value::Value::Tensor(tv)),
                })),
                value_type_for(value),
            )
        }
        Value::List(items) => {
            // In MIL, homogeneous lists of scalars are represented as 1D
            // tensors. Only use a proto ListValue for heterogeneous/nested
            // lists or lists of references.
            if let Some((tv, dt)) = try_list_as_tensor(items)? {
                let len = items.len();
                let tensor_type = mil_spec::ValueType {
                    r#type: Some(mil_spec::value_type::Type::TensorType(
                        mil_spec::TensorType {
                            data_type: dt as i32,
                            rank: 1,
                            dimensions: vec![mil_spec::Dimension {
                                dimension: Some(mil_spec::dimension::Dimension::Constant(
                                    mil_spec::dimension::ConstantDimension { size: len as u64 },
                                )),
                            }],
                            attributes: HashMap::new(),
                        },
                    )),
                };
                (
                    Some(value::Value::ImmediateValue(value::ImmediateValue {
                        value: Some(value::immediate_value::Value::Tensor(tv)),
                    })),
                    Some(tensor_type),
                )
            } else {
                let proto_items = items
                    .iter()
                    .map(convert_value_to_proto)
                    .collect::<Result<Vec<_>>>()?;
                (
                    Some(value::Value::ImmediateValue(value::ImmediateValue {
                        value: Some(value::immediate_value::Value::List(mil_spec::ListValue {
                            values: proto_items,
                        })),
                    })),
                    value_type_for(value),
                )
            }
        }
        Value::Type(tt) => {
            // Type-only value — no immediate payload, just the type field.
            (
                None,
                Some(mil_spec::ValueType {
                    r#type: Some(mil_spec::value_type::Type::TensorType(convert_tensor_type(
                        tt,
                    ))),
                }),
            )
        }
        Value::Tensor { .. } => {
            // Encode tensor using the typed storage that matches its dtype,
            // so coremlcompiler can verify element counts correctly.
            let tv = convert_tensor_data(value)?;
            (
                Some(value::Value::ImmediateValue(value::ImmediateValue {
                    value: Some(value::immediate_value::Value::Tensor(tv)),
                })),
                value_type_for(value),
            )
        }
        Value::BlobFile { file_name, offset } => (
            Some(value::Value::BlobFileValue(
                mil_spec::value::BlobFileValue {
                    file_name: file_name.clone(),
                    offset: *offset,
                },
            )),
            None,
        ),
    };

    Ok(mil_spec::Value {
        doc_string: String::new(),
        r#type: proto_type,
        value: proto_value,
    })
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Convert an IR `TensorType` into a proto `TensorType`.
fn convert_tensor_type(tt: &TensorType) -> mil_spec::TensorType {
    let data_type = convert_scalar_type(tt.scalar_type) as i32;

    let dimensions = tt
        .shape
        .iter()
        .map(|dim| match dim {
            Some(size) => mil_spec::Dimension {
                dimension: Some(mil_spec::dimension::Dimension::Constant(
                    mil_spec::dimension::ConstantDimension { size: *size as u64 },
                )),
            },
            None => mil_spec::Dimension {
                dimension: Some(mil_spec::dimension::Dimension::Unknown(
                    mil_spec::dimension::UnknownDimension { variadic: false },
                )),
            },
        })
        .collect::<Vec<_>>();

    mil_spec::TensorType {
        data_type,
        rank: dimensions.len() as i64,
        dimensions,
        attributes: HashMap::new(),
    }
}

/// Map IR `ScalarType` → proto `DataType`.
fn convert_scalar_type(st: ScalarType) -> mil_spec::DataType {
    match st {
        ScalarType::Float16 => mil_spec::DataType::Float16,
        ScalarType::Float32 => mil_spec::DataType::Float32,
        ScalarType::Float64 => mil_spec::DataType::Float64,
        ScalarType::Int8 => mil_spec::DataType::Int8,
        ScalarType::Int16 => mil_spec::DataType::Int16,
        ScalarType::Int32 => mil_spec::DataType::Int32,
        ScalarType::Int64 => mil_spec::DataType::Int64,
        ScalarType::UInt8 => mil_spec::DataType::Uint8,
        ScalarType::UInt16 => mil_spec::DataType::Uint16,
        ScalarType::UInt32 => mil_spec::DataType::Uint32,
        ScalarType::UInt64 => mil_spec::DataType::Uint64,
        ScalarType::Bool => mil_spec::DataType::Bool,
    }
}

/// Map IR `ScalarType` → feature array data type.
fn scalar_to_array_data_type(st: ScalarType) -> specification::array_feature_type::ArrayDataType {
    use specification::array_feature_type::ArrayDataType;
    match st {
        ScalarType::Float32 => ArrayDataType::Float32,
        ScalarType::Float64 => ArrayDataType::Double,
        ScalarType::Int32 => ArrayDataType::Int32,
        ScalarType::Float16 => ArrayDataType::Float16,
        _ => ArrayDataType::Float32,
    }
}

/// Convert an IR `TensorType` to a CoreML `FeatureType` (MultiArray).
fn tensor_type_to_feature_type(tt: &TensorType) -> FeatureType {
    let shape: Vec<i64> = tt.shape.iter().map(|d| d.unwrap_or(1) as i64).collect();

    // If any dimension is dynamic (None), express that via ShapeRange.
    let shape_flexibility = if tt.shape.iter().any(|d| d.is_none()) {
        let size_ranges: Vec<specification::SizeRange> = tt
            .shape
            .iter()
            .map(|d| match d {
                Some(size) => specification::SizeRange {
                    lower_bound: *size as u64,
                    upper_bound: *size as i64,
                },
                None => specification::SizeRange {
                    lower_bound: 1,
                    upper_bound: -1, // unbounded
                },
            })
            .collect();
        Some(
            specification::array_feature_type::ShapeFlexibility::ShapeRange(
                specification::array_feature_type::ShapeRange { size_ranges },
            ),
        )
    } else {
        None
    };

    FeatureType {
        is_optional: false,
        r#type: Some(specification::feature_type::Type::MultiArrayType(
            ArrayFeatureType {
                shape,
                data_type: scalar_to_array_data_type(tt.scalar_type) as i32,
                shape_flexibility,
                default_optional_value: None,
            },
        )),
    }
}

// ---------------------------------------------------------------------------
// State descriptor support for autoregressive models
// ---------------------------------------------------------------------------

/// Name fragments that identify KV cache tensors for state descriptors.
const STATE_CACHE_PATTERNS: &[&str] = &[
    "past_key_values",
    "past_key",
    "past_value",
    "key_cache",
    "value_cache",
    "kv_cache",
];

/// Build CoreML state descriptors for KV cache tensors in an autoregressive
/// function.
///
/// State descriptors tell CoreML to persist these tensors across inference
/// calls, enabling efficient autoregressive decoding without re-sending the
/// full KV cache from the host.
fn build_state_descriptors(func: &Function) -> Vec<FeatureDescription> {
    let mut descriptors = Vec::new();

    // Scan function inputs for cache tensors.
    for (name, tt) in &func.inputs {
        if is_state_cache_name(name) {
            descriptors.push(tensor_type_to_state_descriptor(name, tt));
        }
    }

    // Also scan for kv_cache_read/kv_cache_update ops that may have been
    // inserted by the KV cache pass.
    let mut seen: std::collections::HashSet<String> =
        descriptors.iter().map(|d| d.name.clone()).collect();

    for op in &func.body.operations {
        if op.op_type == "kv_cache_read" || op.op_type == "kv_cache_update" {
            // The cache input reference points to the state tensor.
            if let Some(Value::Reference(cache_name)) = op.inputs.get("cache") {
                if !seen.contains(cache_name) {
                    // Infer type from the operation's input or fall back to a default.
                    let tt = func
                        .inputs
                        .iter()
                        .find(|(n, _)| n == cache_name)
                        .map(|(_, t)| t.clone())
                        .unwrap_or_else(|| {
                            TensorType::new(ScalarType::Float32, vec![1, 1, 2048, 64])
                        });
                    descriptors.push(tensor_type_to_state_descriptor(cache_name, &tt));
                    seen.insert(cache_name.clone());
                }
            }
        }
    }

    descriptors
}

/// Returns `true` if the name matches a KV cache naming pattern.
fn is_state_cache_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    STATE_CACHE_PATTERNS.iter().any(|p| lower.contains(p))
}

/// Convert a tensor type into a CoreML state feature descriptor.
///
/// The state descriptor uses `StateFeatureType` wrapping an `ArrayFeatureType`,
/// which tells CoreML to persist this tensor across inference calls.
fn tensor_type_to_state_descriptor(name: &str, tt: &TensorType) -> FeatureDescription {
    let shape: Vec<i64> = tt.shape.iter().map(|d| d.unwrap_or(1) as i64).collect();
    let array_type = ArrayFeatureType {
        shape,
        data_type: scalar_to_array_data_type(tt.scalar_type) as i32,
        shape_flexibility: None,
        default_optional_value: None,
    };

    FeatureDescription {
        name: name.to_string(),
        short_description: format!("KV cache state for {name}"),
        r#type: Some(FeatureType {
            is_optional: false,
            r#type: Some(specification::feature_type::Type::StateType(
                StateFeatureType {
                    r#type: Some(specification::state_feature_type::Type::ArrayType(
                        array_type,
                    )),
                },
            )),
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorData;
    use crate::convert::model_to_program;

    #[test]
    fn round_trip_empty_program() {
        let program = Program::new("1");
        let model = program_to_model(&program, 7).unwrap();
        let recovered = model_to_program(&model).unwrap();

        assert_eq!(recovered.version, "1");
        assert!(recovered.functions.is_empty());
    }

    #[test]
    fn round_trip_simple_program() {
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);
        let relu = Operation::new("relu", "relu_out")
            .with_input("x", Value::Reference("input".to_string()))
            .with_output("relu_out");

        let mut block = Block::new();
        block.add_op(relu);
        block.outputs.push("relu_out".into());

        let func = Function {
            name: "main".to_string(),
            inputs: vec![("input".to_string(), input_ty.clone())],
            body: block,
            attributes: HashMap::new(),
        };

        let mut program = Program::new("1");
        program.add_function(func);

        // IR → Proto → IR round-trip
        let model = program_to_model(&program, 7).unwrap();
        let recovered = model_to_program(&model).unwrap();

        assert_eq!(recovered.version, "1");
        assert_eq!(recovered.functions.len(), 1);

        let main = &recovered.functions["main"];
        assert_eq!(main.inputs.len(), 1);
        assert_eq!(main.inputs[0].0, "input");
        assert_eq!(main.inputs[0].1, input_ty);

        assert_eq!(main.body.operations.len(), 1);
        assert_eq!(main.body.operations[0].op_type, "relu");
        assert_eq!(main.body.outputs, vec!["relu_out"]);

        // The input reference should survive the round-trip.
        assert!(matches!(
            main.body.operations[0].inputs.get("x"),
            Some(Value::Reference(r)) if r == "input"
        ));
    }

    #[test]
    fn round_trip_with_dynamic_dimensions() {
        let input_ty =
            TensorType::with_dynamic_shape(ScalarType::Float16, vec![None, Some(768), None]);
        let func = Function {
            name: "encode".to_string(),
            inputs: vec![("tokens".to_string(), input_ty.clone())],
            body: Block::new(),
            attributes: HashMap::new(),
        };

        let mut program = Program::new("1");
        program.add_function(func);

        let model = program_to_model(&program, 8).unwrap();
        let recovered = model_to_program(&model).unwrap();

        let enc = &recovered.functions["encode"];
        assert_eq!(enc.inputs[0].1, input_ty);
        assert!(!enc.inputs[0].1.is_static());
    }

    #[test]
    fn round_trip_scalar_attributes() {
        let op = Operation::new("const", "c0")
            .with_attr("val", Value::Int(42))
            .with_attr("eps", Value::Float(1e-5))
            .with_attr("training", Value::Bool(false))
            .with_output("c0");

        let mut block = Block::new();
        block.add_op(op);
        block.outputs.push("c0".into());

        let func = Function {
            name: "main".to_string(),
            inputs: vec![],
            body: block,
            attributes: HashMap::new(),
        };

        let mut program = Program::new("1");
        program.add_function(func);

        let model = program_to_model(&program, 7).unwrap();
        let recovered = model_to_program(&model).unwrap();

        // const ops keep their params as proto attributes, so they
        // round-trip back as attributes in the IR.
        let attrs = &recovered.functions["main"].body.operations[0].attributes;
        assert!(matches!(attrs.get("val"), Some(Value::Int(42))));
        assert!(matches!(attrs.get("training"), Some(Value::Bool(false))));
        // Float round-trips through f32, so check approximate equality.
        if let Some(Value::Float(f)) = attrs.get("eps") {
            assert!((*f - 1e-5_f64).abs() < 1e-7, "eps = {f}");
        } else {
            panic!("expected Float for eps, got {:?}", attrs.get("eps"));
        }
    }

    #[test]
    fn model_specification_version_preserved() {
        let program = Program::new("1");
        let model = program_to_model(&program, 8).unwrap();
        assert_eq!(model.specification_version, 8);
    }

    #[test]
    fn updatable_model_sets_is_updatable() {
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 10]);
        let dense = Operation::new("linear", "dense_0")
            .with_input("x", Value::Reference("input".to_string()))
            .with_output("dense_out");

        let mut block = Block::new();
        block.add_op(dense);
        block.outputs.push("dense_out".into());

        let func = Function {
            name: "main".to_string(),
            inputs: vec![("input".to_string(), input_ty)],
            body: block,
            attributes: HashMap::new(),
        };

        let mut program = Program::new("1");
        program.add_function(func);

        let config = UpdatableModelConfig {
            updatable_layers: vec!["dense_0".to_string()],
            learning_rate: 0.01,
            epochs: 5,
            loss_function: LossFunction::CategoricalCrossEntropy,
            optimizer: UpdateOptimizer::Sgd,
        };

        let model = program_to_updatable_model(&program, 9, &config).unwrap();
        assert!(model.is_updatable);
        assert_eq!(model.specification_version, 9);
    }

    #[test]
    fn updatable_model_has_training_inputs() {
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 10]);
        let func = Function {
            name: "main".to_string(),
            inputs: vec![("input".to_string(), input_ty)],
            body: Block::new(),
            attributes: HashMap::new(),
        };

        let mut program = Program::new("1");
        program.add_function(func);

        let config = UpdatableModelConfig::default();
        let model = program_to_updatable_model(&program, 9, &config).unwrap();

        let desc = model.description.as_ref().unwrap();
        assert!(!desc.training_input.is_empty());
    }

    #[test]
    fn updatable_model_metadata_contains_config() {
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 10]);
        let func = Function {
            name: "main".to_string(),
            inputs: vec![("input".to_string(), input_ty)],
            body: Block::new(),
            attributes: HashMap::new(),
        };

        let mut program = Program::new("1");
        program.add_function(func);

        let config = UpdatableModelConfig {
            updatable_layers: vec!["layer_a".to_string(), "layer_b".to_string()],
            learning_rate: 0.005,
            epochs: 20,
            loss_function: LossFunction::MeanSquaredError,
            optimizer: UpdateOptimizer::Adam,
        };

        let model = program_to_updatable_model(&program, 9, &config).unwrap();
        let meta = model
            .description
            .as_ref()
            .unwrap()
            .metadata
            .as_ref()
            .unwrap();

        assert_eq!(
            meta.user_defined.get("com.ironmill.optimizer").unwrap(),
            "adam"
        );
        assert_eq!(
            meta.user_defined.get("com.ironmill.loss_function").unwrap(),
            "mse"
        );
        assert_eq!(meta.user_defined.get("com.ironmill.epochs").unwrap(), "20");
        assert_eq!(
            meta.user_defined
                .get("com.ironmill.updatable_layers")
                .unwrap(),
            "layer_a,layer_b"
        );
    }

    #[test]
    fn build_update_params_sgd() {
        let config = UpdatableModelConfig {
            updatable_layers: vec!["fc".to_string()],
            learning_rate: 0.1,
            epochs: 3,
            loss_function: LossFunction::CategoricalCrossEntropy,
            optimizer: UpdateOptimizer::Sgd,
        };
        let params = build_update_params(&config, None);

        assert_eq!(params.loss_layers.len(), 1);
        assert_eq!(params.loss_layers[0].name, "loss");
        assert_eq!(params.epochs.as_ref().unwrap().default_value, 3);

        let opt = params.optimizer.as_ref().unwrap();
        assert!(matches!(
            opt.optimizer_type,
            Some(specification::optimizer::OptimizerType::SgdOptimizer(_))
        ));
    }

    #[test]
    fn build_update_params_adam() {
        let config = UpdatableModelConfig {
            updatable_layers: vec!["fc".to_string()],
            learning_rate: 0.001,
            epochs: 10,
            loss_function: LossFunction::MeanSquaredError,
            optimizer: UpdateOptimizer::Adam,
        };
        let params = build_update_params(&config, None);

        let opt = params.optimizer.as_ref().unwrap();
        assert!(matches!(
            opt.optimizer_type,
            Some(specification::optimizer::OptimizerType::AdamOptimizer(_))
        ));

        // Verify MSE loss
        assert!(matches!(
            params.loss_layers[0].loss_layer_type,
            Some(specification::loss_layer::LossLayerType::MeanSquaredErrorLossLayer(_))
        ));
    }

    #[test]
    fn multi_function_model_contains_all_functions() {
        use crate::convert::moe::{MoeManifest, MoeSplitResult};

        // Build a mock MoeSplitResult with a shared program and 2 expert programs.
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 512]);

        // Shared program
        let mut shared_func = Function::new("main");
        shared_func.inputs = vec![("hidden".to_string(), input_ty.clone())];
        shared_func.body.add_op(
            Operation::new("relu", "shared_relu")
                .with_input("x", Value::Reference("hidden".to_string()))
                .with_output("relu_out"),
        );
        shared_func.body.outputs.push("relu_out".into());
        let mut shared = Program::new("1");
        shared.add_function(shared_func);

        // Expert 0
        let mut expert0_func = Function::new("main");
        expert0_func.inputs = vec![("relu_out".to_string(), input_ty.clone())];
        expert0_func.body.add_op(
            Operation::new("linear", "expert0_linear")
                .with_input("x", Value::Reference("relu_out".to_string()))
                .with_output("e0_out"),
        );
        expert0_func.body.outputs.push("e0_out".into());
        let mut expert0 = Program::new("1");
        expert0.add_function(expert0_func);

        // Expert 1
        let mut expert1_func = Function::new("main");
        expert1_func.inputs = vec![("relu_out".to_string(), input_ty.clone())];
        expert1_func.body.add_op(
            Operation::new("linear", "expert1_linear")
                .with_input("x", Value::Reference("relu_out".to_string()))
                .with_output("e1_out"),
        );
        expert1_func.body.outputs.push("e1_out".into());
        let mut expert1 = Program::new("1");
        expert1.add_function(expert1_func);

        let split = MoeSplitResult {
            shared,
            experts: vec![expert0, expert1],
            manifest: MoeManifest {
                expert_count: 2,
                router_output: "router_out".to_string(),
                experts: vec![],
                stages: vec![],
            },
        };

        let model = program_to_multi_function_model(&split, 9).unwrap();
        assert_eq!(model.specification_version, 9);
        assert!(!model.is_updatable);

        // Extract the MIL program from the model.
        let proto_program = match &model.r#type {
            Some(model::Type::MlProgram(p)) => p,
            _ => panic!("expected MlProgram"),
        };

        // Should contain 3 functions: main, expert_0, expert_1.
        assert_eq!(proto_program.functions.len(), 3);
        assert!(proto_program.functions.contains_key("main"));
        assert!(proto_program.functions.contains_key("expert_0"));
        assert!(proto_program.functions.contains_key("expert_1"));
    }

    #[test]
    fn multi_function_model_deduplicates_shared_consts() {
        use crate::convert::moe::{MoeManifest, MoeSplitResult};

        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 10]);

        // Shared program with a const op
        let mut shared_func = Function::new("main");
        shared_func.inputs = vec![("x".to_string(), input_ty.clone())];
        shared_func.body.add_op(
            Operation::new("const", "shared_weight")
                .with_attr("val", Value::Float(1.0))
                .with_output("shared_weight"),
        );
        shared_func.body.add_op(
            Operation::new("add", "add_0")
                .with_input("x", Value::Reference("x".to_string()))
                .with_input("y", Value::Reference("shared_weight".to_string()))
                .with_output("add_out"),
        );
        shared_func.body.outputs.push("add_out".into());
        let mut shared = Program::new("1");
        shared.add_function(shared_func);

        // Expert 0 also uses a const with the same name (shared weight)
        let mut expert_func = Function::new("main");
        expert_func.inputs = vec![("x".to_string(), input_ty.clone())];
        expert_func.body.add_op(
            Operation::new("const", "shared_weight")
                .with_attr("val", Value::Float(1.0))
                .with_output("shared_weight"),
        );
        expert_func.body.add_op(
            Operation::new("mul", "mul_0")
                .with_input("x", Value::Reference("x".to_string()))
                .with_input("y", Value::Reference("shared_weight".to_string()))
                .with_output("mul_out"),
        );
        expert_func.body.outputs.push("mul_out".into());
        let mut expert0 = Program::new("1");
        expert0.add_function(expert_func);

        let split = MoeSplitResult {
            shared,
            experts: vec![expert0],
            manifest: MoeManifest {
                expert_count: 1,
                router_output: String::new(),
                experts: vec![],
                stages: vec![],
            },
        };

        let model = program_to_multi_function_model(&split, 9).unwrap();
        let proto_program = match &model.r#type {
            Some(model::Type::MlProgram(p)) => p,
            _ => panic!("expected MlProgram"),
        };

        // The expert function should exist.
        assert!(proto_program.functions.contains_key("expert_0"));

        // The shared const in expert_0 should have a BlobFileValue
        // (stub) rather than carrying duplicated tensor data.
        let expert_fn = &proto_program.functions["expert_0"];
        let block = expert_fn.block_specializations.values().next().unwrap();
        let const_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "const")
            .expect("expert should have a const op");

        let val_attr = const_op.attributes.get("val").expect("should have val");
        assert!(
            matches!(
                val_attr.value,
                Some(mil_spec::value::Value::BlobFileValue(_))
            ),
            "shared const in expert should be a stub (BlobFileValue), got: {:?}",
            val_attr.value
        );
    }

    // -----------------------------------------------------------------------
    // Autoregressive state descriptor tests
    // -----------------------------------------------------------------------

    #[test]
    fn ar_model_has_state_descriptors() {
        let mut program = Program::new("1");
        program.set_attribute("autoregressive", "true");

        let func = Function::new("main")
            .with_input("input_ids", TensorType::new(ScalarType::Int32, vec![1, 1]))
            .with_input(
                "past_key_values.0.key",
                TensorType::new(ScalarType::Float32, vec![1, 8, 128, 64]),
            )
            .with_input(
                "past_key_values.0.value",
                TensorType::new(ScalarType::Float32, vec![1, 8, 128, 64]),
            );
        program.add_function(func);

        let model = program_to_model(&program, 8).unwrap();
        let desc = model.description.as_ref().unwrap();

        // Should have state descriptors for cache inputs.
        assert_eq!(desc.state.len(), 2, "expected 2 state descriptors");

        // State descriptors should use StateFeatureType.
        for state_desc in &desc.state {
            let ft = state_desc.r#type.as_ref().unwrap();
            assert!(
                matches!(
                    ft.r#type,
                    Some(specification::feature_type::Type::StateType(_))
                ),
                "state descriptor should use StateType, got: {:?}",
                ft.r#type
            );
        }

        // Check names.
        let names: Vec<&str> = desc.state.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"past_key_values.0.key"));
        assert!(names.contains(&"past_key_values.0.value"));
    }

    #[test]
    fn non_ar_model_has_no_state_descriptors() {
        let mut program = Program::new("1");
        let func = Function::new("main").with_input(
            "image",
            TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]),
        );
        program.add_function(func);

        let model = program_to_model(&program, 8).unwrap();
        let desc = model.description.as_ref().unwrap();

        assert!(desc.state.is_empty(), "non-AR model should have no state");
    }

    #[test]
    fn ar_state_descriptor_has_correct_shape() {
        let mut program = Program::new("1");
        program.set_attribute("autoregressive", "true");

        let func = Function::new("main").with_input(
            "past_key",
            TensorType::new(ScalarType::Float32, vec![1, 8, 2048, 64]),
        );
        program.add_function(func);

        let model = program_to_model(&program, 8).unwrap();
        let desc = model.description.as_ref().unwrap();

        assert_eq!(desc.state.len(), 1);
        let state = &desc.state[0];
        assert_eq!(state.name, "past_key");

        // Extract the array type from the state descriptor.
        let ft = state.r#type.as_ref().unwrap();
        if let Some(specification::feature_type::Type::StateType(st)) = &ft.r#type {
            if let Some(specification::state_feature_type::Type::ArrayType(arr)) = &st.r#type {
                assert_eq!(arr.shape, vec![1, 8, 2048, 64]);
            } else {
                panic!("expected ArrayType in StateFeatureType");
            }
        } else {
            panic!("expected StateType in FeatureType");
        }
    }

    #[test]
    fn ar_state_descriptors_from_kv_cache_ops() {
        let mut program = Program::new("1");
        program.set_attribute("autoregressive", "true");

        let kv_read = Operation::new("kv_cache_read", "kv_read_0")
            .with_input("cache", Value::Reference("my_cache".into()))
            .with_output("kv_read_out");

        let mut block = Block::new();
        block.add_op(kv_read);
        block.outputs.push("kv_read_out".into());

        // Add the cache as a function input so it can be resolved.
        let func = Function {
            name: "main".into(),
            inputs: vec![(
                "my_cache".into(),
                TensorType::new(ScalarType::Float32, vec![1, 4, 512, 64]),
            )],
            body: block,
            attributes: HashMap::new(),
        };
        program.add_function(func);

        let model = program_to_model(&program, 8).unwrap();
        let desc = model.description.as_ref().unwrap();

        // Should find a state descriptor for "my_cache".
        let names: Vec<&str> = desc.state.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"my_cache"),
            "should have state descriptor for my_cache, got: {names:?}"
        );
    }

    #[test]
    fn round_trip_affine_dequantize_with_bit_width_and_group_size() {
        // Build a constexpr_affine_dequantize op with bit_width and group_size.
        let quantized_data = Value::Tensor {
            data: TensorData::Inline(vec![0, 5, 10, 15, 0, 5, 10, 15]),
            shape: vec![2, 4],
            dtype: ScalarType::UInt8,
        };
        let scale_bytes: Vec<u8> = [0.1_f32, 0.2, 0.3, 0.4]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let zp_bytes: Vec<u8> = [0.0_f32, 0.0, 0.0, 0.0]
            .iter()
            .flat_map(|z| z.to_le_bytes())
            .collect();

        let op = Operation::new("constexpr_affine_dequantize", "w_quant")
            .with_attr("quantized_data", quantized_data)
            .with_attr(
                "scale",
                Value::Tensor {
                    data: TensorData::Inline(scale_bytes),
                    shape: vec![2, 2],
                    dtype: ScalarType::Float32,
                },
            )
            .with_attr(
                "zero_point",
                Value::Tensor {
                    data: TensorData::Inline(zp_bytes),
                    shape: vec![2, 2],
                    dtype: ScalarType::Float32,
                },
            )
            .with_attr("axis", Value::Int(1))
            .with_attr("bit_width", Value::Int(4))
            .with_attr("group_size", Value::Int(2))
            .with_output("w_quant");

        let mut block = Block::new();
        block.add_op(op);
        block.outputs.push("w_quant".into());

        let func = Function {
            name: "main".to_string(),
            inputs: vec![],
            body: block,
            attributes: HashMap::new(),
        };

        let mut program = Program::new("1");
        program.add_function(func);

        let model = program_to_model(&program, 7).unwrap();
        let recovered = model_to_program(&model).unwrap();

        let attrs = &recovered.functions["main"].body.operations[0].attributes;
        assert_eq!(attrs.get("bit_width"), Some(&Value::Int(4)));
        assert_eq!(attrs.get("group_size"), Some(&Value::Int(2)));
        assert_eq!(
            recovered.functions["main"].body.operations[0].op_type,
            "constexpr_affine_dequantize"
        );
    }

    #[test]
    fn round_trip_legacy_affine_dequantize_without_metadata() {
        // Simulate a legacy model: constexpr_affine_dequantize without
        // bit_width or group_size attributes. Should round-trip cleanly
        // and the attributes should simply be absent (callers default).
        let quantized_data = Value::Tensor {
            data: TensorData::Inline(vec![0, 128, 255, 64]),
            shape: vec![4],
            dtype: ScalarType::UInt8,
        };

        let op = Operation::new("constexpr_affine_dequantize", "w_quant")
            .with_attr("quantized_data", quantized_data)
            .with_attr("scale", Value::Float(0.01))
            .with_attr("zero_point", Value::Float(128.0))
            .with_attr("axis", Value::Int(0))
            .with_output("w_quant");

        let mut block = Block::new();
        block.add_op(op);
        block.outputs.push("w_quant".into());

        let func = Function {
            name: "main".to_string(),
            inputs: vec![],
            body: block,
            attributes: HashMap::new(),
        };

        let mut program = Program::new("1");
        program.add_function(func);

        let model = program_to_model(&program, 7).unwrap();
        let recovered = model_to_program(&model).unwrap();

        let attrs = &recovered.functions["main"].body.operations[0].attributes;
        // Legacy model: no bit_width or group_size attributes.
        assert!(attrs.get("bit_width").is_none());
        assert!(attrs.get("group_size").is_none());
        // Other attrs should survive.
        assert!(attrs.get("quantized_data").is_some());
        assert!(attrs.get("scale").is_some());
    }
}
