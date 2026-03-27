use std::collections::HashMap;

use crate::error::Result;

use super::Operation;

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
        // TODO: implement full validation
        Ok(())
    }
}
