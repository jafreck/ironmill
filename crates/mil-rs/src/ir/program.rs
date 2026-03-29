use std::collections::HashMap;

use indexmap::IndexMap;

use super::operation::Operation;
use super::tensor::TensorType;

/// A complete MIL program — the top-level container for a CoreML ML Program model.
///
/// A program contains one or more named functions. The "main" function is
/// the entry point for inference.
#[derive(Debug, Clone)]
pub struct Program {
    /// Program version (e.g., "1.0.0").
    pub version: String,
    /// Named functions in this program. Key is the function name.
    pub functions: IndexMap<String, Function>,
    /// Program-level attributes (e.g., `autoregressive`, `max_seq_length`).
    pub attributes: HashMap<String, String>,
}

impl Program {
    /// Create a new program with the given version and no functions.
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            functions: IndexMap::new(),
            attributes: HashMap::new(),
        }
    }

    /// Add a function to the program.
    pub fn add_function(&mut self, function: Function) {
        self.functions.insert(function.name.clone(), function);
    }

    /// Return the "main" function, falling back to the first function if
    /// no function is explicitly named "main".
    pub fn main(&self) -> Option<&Function> {
        self.functions
            .get("main")
            .or_else(|| self.functions.values().next())
    }

    /// Set a program-level attribute.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Check whether a program-level attribute equals a given value.
    pub fn has_attribute(&self, key: &str, value: &str) -> bool {
        self.attributes.get(key).is_some_and(|v| v == value)
    }

    /// Returns `true` if this program has been tagged as autoregressive.
    pub fn is_autoregressive(&self) -> bool {
        self.has_attribute("autoregressive", "true")
    }

    /// Estimate the total floating-point operations (FLOPs) for a forward pass.
    ///
    /// Walks all functions and sums per-op FLOPs using shape-aware formulas
    /// for conv, matmul, linear, attention, and other compute ops.
    pub fn total_flops(&self) -> u64 {
        crate::analysis::flops::estimate_program_flops(self)
    }
}

/// A function in the MIL program — a graph with a typed signature.
///
/// Functions have named, typed inputs and a body ([`Block`]) that
/// contains the computation graph.
#[derive(Debug, Clone)]
pub struct Function {
    /// Function name.
    pub name: String,
    /// Ordered named inputs with their types.
    pub inputs: Vec<(String, TensorType)>,
    /// The function body.
    pub body: Block,
}

impl Function {
    /// Create a new function with an empty body.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inputs: Vec::new(),
            body: Block::new(),
        }
    }

    /// Add a typed input parameter.
    pub fn with_input(mut self, name: impl Into<String>, ty: TensorType) -> Self {
        self.inputs.push((name.into(), ty));
        self
    }
}

/// A block within a function — a scope that contains operations.
///
/// Blocks can be nested for control flow (e.g., `cond`, `while_loop`),
/// though most models only use a single top-level block.
#[derive(Debug, Clone)]
pub struct Block {
    /// Operations in topological order within this block.
    pub operations: Vec<Operation>,
    /// Output value names from this block.
    pub outputs: Vec<String>,
}

impl Block {
    /// Create a new empty block.
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add an operation to the block.
    pub fn add_op(&mut self, op: Operation) {
        self.operations.push(op);
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::tensor::ScalarType;

    #[test]
    fn create_program_with_function() {
        let input_ty = TensorType::new(ScalarType::Float32, vec![1, 3, 224, 224]);
        let func = Function::new("main").with_input("image", input_ty.clone());

        let mut program = Program::new("1.0.0");
        program.add_function(func);

        assert_eq!(program.version, "1.0.0");
        assert_eq!(program.functions.len(), 1);
        let main = &program.functions["main"];
        assert_eq!(main.name, "main");
        assert_eq!(main.inputs.len(), 1);
        assert_eq!(main.inputs[0].0, "image");
        assert_eq!(main.inputs[0].1, input_ty);
    }

    #[test]
    fn main_returns_main_function() {
        let mut program = Program::new("1.0.0");
        program.add_function(Function::new("preprocess"));
        program.add_function(Function::new("main"));

        let main = program.main().expect("should find main");
        assert_eq!(main.name, "main");
    }

    #[test]
    fn main_falls_back_to_first_function() {
        let mut program = Program::new("1.0.0");
        program.add_function(Function::new("only_func"));

        let main = program.main().expect("should return first function");
        assert_eq!(main.name, "only_func");
    }

    #[test]
    fn main_returns_none_for_empty_program() {
        let program = Program::new("1.0.0");
        assert!(program.main().is_none());
    }

    #[test]
    fn add_operations_to_block() {
        let mut block = Block::new();
        let relu = Operation::new("relu", "relu_0").with_output("relu_out");
        let softmax = Operation::new("softmax", "softmax_0").with_output("softmax_out");

        block.add_op(relu);
        block.add_op(softmax);
        block.outputs.push("softmax_out".into());

        assert_eq!(block.operations.len(), 2);
        assert_eq!(block.operations[0].name, "relu_0");
        assert_eq!(block.operations[1].name, "softmax_0");
        assert_eq!(block.outputs, vec!["softmax_out"]);
    }

    #[test]
    fn program_attributes() {
        let mut program = Program::new("1.0.0");
        assert!(!program.is_autoregressive());
        assert!(!program.has_attribute("autoregressive", "true"));

        program.set_attribute("autoregressive", "true");
        assert!(program.is_autoregressive());
        assert!(program.has_attribute("autoregressive", "true"));
        assert!(!program.has_attribute("autoregressive", "false"));
    }

    #[test]
    fn program_attributes_default_empty() {
        let program = Program::new("1.0.0");
        assert!(program.attributes.is_empty());
    }
}
