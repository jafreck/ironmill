use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use super::tensor::TensorType;
use super::types::Value;

/// Preferred compute unit for running an operation on Apple hardware.
///
/// CoreML supports per-operation compute unit preferences, allowing the
/// runtime to schedule work on the most appropriate hardware. These
/// annotations are informational hints — the runtime may override them
/// when hardware is unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ComputeUnit {
    /// Apple Neural Engine — best for supported dense ops with aligned shapes.
    Ane,
    /// GPU — fallback for ops the ANE can't handle efficiently.
    Gpu,
    /// CPU — general fallback.
    Cpu,
    /// Let the runtime decide (no preference).
    Any,
}

impl fmt::Display for ComputeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeUnit::Ane => write!(f, "ane"),
            ComputeUnit::Gpu => write!(f, "gpu"),
            ComputeUnit::Cpu => write!(f, "cpu"),
            ComputeUnit::Any => write!(f, "any"),
        }
    }
}

impl FromStr for ComputeUnit {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ane" => Ok(ComputeUnit::Ane),
            "gpu" => Ok(ComputeUnit::Gpu),
            "cpu" => Ok(ComputeUnit::Cpu),
            "any" => Ok(ComputeUnit::Any),
            other => Err(format!("unknown compute unit: {other}")),
        }
    }
}

/// A single operation in the MIL graph.
///
/// Operations correspond to MIL ops (e.g., `conv`, `matmul`, `relu`, `softmax`)
/// and map to both ONNX source ops and CoreML target ops.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Operation {
    /// The MIL operation type (e.g., "conv", "matmul", "relu").
    pub op_type: String,

    /// Unique name for this operation within the graph.
    pub name: String,

    /// Named inputs — keys are parameter names, values reference other ops' outputs.
    pub inputs: HashMap<String, Value>,

    /// Named outputs produced by this operation.
    pub outputs: Vec<String>,

    /// Type information for each output (parallel to `outputs`).
    /// Entries are `None` when the type is unknown and must be inferred.
    pub output_types: Vec<Option<TensorType>>,

    /// Operation-specific attributes (e.g., kernel size, stride, padding).
    pub attributes: HashMap<String, Value>,

    /// Compute unit preference for this operation.
    pub compute_unit: Option<ComputeUnit>,
}

impl Operation {
    /// Create a new operation with the given type and name.
    pub fn new(op_type: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            op_type: op_type.into(),
            name: name.into(),
            inputs: HashMap::new(),
            outputs: Vec::new(),
            output_types: Vec::new(),
            attributes: HashMap::new(),
            compute_unit: None,
        }
    }

    /// Add a named input to this operation.
    pub fn with_input(mut self, name: impl Into<String>, value: Value) -> Self {
        self.inputs.insert(name.into(), value);
        self
    }

    /// Add a named output to this operation.
    pub fn with_output(mut self, name: impl Into<String>) -> Self {
        self.outputs.push(name.into());
        self.output_types.push(None);
        self
    }

    /// Add an attribute to this operation.
    pub fn with_attr(mut self, name: impl Into<String>, value: Value) -> Self {
        self.attributes.insert(name.into(), value);
        self
    }

    /// Get the compute unit preference for this operation, if set.
    pub fn compute_unit(&self) -> Option<ComputeUnit> {
        self.compute_unit
    }

    /// Set the compute unit preference for this operation.
    pub fn set_compute_unit(&mut self, cu: ComputeUnit) {
        self.compute_unit = Some(cu);
    }
}
