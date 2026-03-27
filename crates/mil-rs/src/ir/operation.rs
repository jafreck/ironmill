use std::collections::HashMap;

use super::types::Value;

/// A single operation in the MIL graph.
///
/// Operations correspond to MIL ops (e.g., `conv`, `matmul`, `relu`, `softmax`)
/// and map to both ONNX source ops and CoreML target ops.
#[derive(Debug, Clone)]
pub struct Operation {
    /// The MIL operation type (e.g., "conv", "matmul", "relu").
    pub op_type: String,

    /// Unique name for this operation within the graph.
    pub name: String,

    /// Named inputs — keys are parameter names, values reference other ops' outputs.
    pub inputs: HashMap<String, Value>,

    /// Named outputs produced by this operation.
    pub outputs: Vec<String>,

    /// Operation-specific attributes (e.g., kernel size, stride, padding).
    pub attributes: HashMap<String, Value>,
}

impl Operation {
    /// Create a new operation with the given type and name.
    pub fn new(op_type: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            op_type: op_type.into(),
            name: name.into(),
            inputs: HashMap::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
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
        self
    }

    /// Add an attribute to this operation.
    pub fn with_attr(mut self, name: impl Into<String>, value: Value) -> Self {
        self.attributes.insert(name.into(), value);
        self
    }
}
