use std::collections::{HashMap, HashSet};

use crate::error::{MilError, Result};

use super::Operation;
use super::types::Value;

/// A MIL computation graph.
///
/// Contains a set of named operations connected by value references.
/// This is the central data structure that model converters build
/// and optimization passes transform.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Unique name for this graph/function.
    pub name: String,

    /// Ordered list of operations in topological order.
    pub operations: Vec<Operation>,

    /// Input names and their types.
    pub inputs: Vec<String>,

    /// Output names.
    pub outputs: Vec<String>,

    /// Named attributes / metadata.
    pub attributes: HashMap<String, String>,
}

impl Graph {
    /// Create a new empty graph with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            operations: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Add an operation to the graph.
    pub fn add_op(&mut self, op: Operation) {
        self.operations.push(op);
    }

    /// Validate the graph for structural correctness.
    ///
    /// Checks for:
    /// - All value references resolve to a defined output
    /// - No duplicate operation names
    /// - Outputs reference existing values
    pub fn validate(&self) -> Result<()> {
        // Collect all defined values: graph inputs + operation outputs.
        let mut defined: HashSet<&str> = HashSet::new();
        for name in &self.inputs {
            defined.insert(name.as_str());
        }

        // Check for duplicate operation names.
        let mut op_names: HashSet<&str> = HashSet::new();
        for op in &self.operations {
            if !op.name.is_empty() && !op_names.insert(op.name.as_str()) {
                return Err(MilError::Validation(format!(
                    "duplicate operation name: '{}'",
                    op.name
                )));
            }
            for out in &op.outputs {
                defined.insert(out.as_str());
            }
        }

        // Check that all value references in operation inputs resolve.
        for op in &self.operations {
            for (param, value) in &op.inputs {
                Self::check_refs(value, &defined, &op.name, param)?;
            }
        }

        // Check that graph outputs reference existing values.
        for out in &self.outputs {
            if !defined.contains(out.as_str()) {
                return Err(MilError::Validation(format!(
                    "graph output '{}' references undefined value",
                    out
                )));
            }
        }

        Ok(())
    }

    /// Recursively check that all `Value::Reference`s within a value resolve.
    fn check_refs(
        value: &Value,
        defined: &HashSet<&str>,
        op_name: &str,
        param: &str,
    ) -> Result<()> {
        match value {
            Value::Reference(name) => {
                if !defined.contains(name.as_str()) {
                    return Err(MilError::UndefinedValue(format!(
                        "operation '{}' input '{}' references undefined value '{}'",
                        op_name, param, name
                    )));
                }
            }
            Value::List(items) => {
                for item in items {
                    Self::check_refs(item, defined, op_name, param)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}
