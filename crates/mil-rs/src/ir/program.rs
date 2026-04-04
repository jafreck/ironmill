use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;

use super::operation::Operation;
use super::tensor::TensorType;
use super::types::Value;
use crate::error::MilError;
use crate::weights::WeightProvider;

use std::fmt;

/// A complete MIL program — the top-level container for a CoreML ML Program model.
///
/// A program contains one or more named functions. The "main" function is
/// the entry point for inference.
#[derive(Clone)]
pub struct Program {
    /// Program version (e.g., "1.0.0").
    pub version: String,
    /// Named functions in this program. Key is the function name.
    pub functions: IndexMap<String, Function>,
    /// Program-level attributes (e.g., `autoregressive`, `max_seq_length`).
    pub attributes: HashMap<String, String>,

    /// Optional weight provider for resolving `TensorData::External` references.
    pub(crate) weight_provider: Option<Arc<dyn WeightProvider + Send + Sync>>,
}

impl fmt::Debug for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Program")
            .field("version", &self.version)
            .field("functions", &self.functions)
            .field("attributes", &self.attributes)
            .field(
                "weight_provider",
                &self.weight_provider.as_ref().map(|_| ".."),
            )
            .finish()
    }
}

impl Program {
    /// Create a new program with the given version and no functions.
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            functions: IndexMap::new(),
            attributes: HashMap::new(),
            weight_provider: None,
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

    /// Return a mutable reference to the "main" function.
    pub fn main_mut(&mut self) -> Option<&mut Function> {
        if self.functions.contains_key("main") {
            self.functions.get_mut("main")
        } else {
            self.functions.values_mut().next()
        }
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

    /// Attach a weight provider for lazy tensor resolution.
    pub fn set_weight_provider(&mut self, provider: Arc<dyn WeightProvider + Send + Sync>) {
        self.weight_provider = Some(provider);
    }

    /// Resolve an external tensor key to its byte data.
    pub fn resolve_tensor(&self, key: &str) -> Result<Vec<u8>, MilError> {
        let provider = self.weight_provider.as_ref().ok_or_else(|| {
            MilError::Validation(format!(
                "no weight provider attached; cannot resolve external tensor '{key}'"
            ))
        })?;
        let tensor = provider.tensor(key)?;
        Ok(tensor.data.into_owned())
    }

    /// Returns `true` if a weight provider is attached.
    pub fn has_weight_provider(&self) -> bool {
        self.weight_provider.is_some()
    }

    /// Clone the weight provider `Arc`, if one is attached.
    pub fn weight_provider(&self) -> Option<Arc<dyn WeightProvider + Send + Sync>> {
        self.weight_provider.clone()
    }

    /// Remove and return the weight provider, leaving `None` in its place.
    pub fn take_weight_provider(&mut self) -> Option<Arc<dyn WeightProvider + Send + Sync>> {
        self.weight_provider.take()
    }

    /// Materialize all `TensorData::External` references in the program.
    pub fn materialize_all(&mut self) -> Result<(), MilError> {
        let provider = self
            .weight_provider
            .clone()
            .ok_or_else(|| MilError::Validation("no weight provider for materialization".into()))?;

        for function in self.functions.values_mut() {
            Self::materialize_block(&mut function.body, &*provider)?;
        }
        Ok(())
    }

    fn materialize_block(block: &mut Block, provider: &dyn WeightProvider) -> Result<(), MilError> {
        for op in &mut block.operations {
            for val in op.inputs.values_mut().chain(op.attributes.values_mut()) {
                if let Value::Tensor { data, .. } = val {
                    data.materialize_with(|key| {
                        let tensor = provider.tensor(key)?;
                        Ok(tensor.data.into_owned())
                    })?;
                }
            }
        }
        Ok(())
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
    use crate::ir::types::TensorData;
    use crate::weights::{Architecture, ModelConfig, WeightProvider, WeightTensor};
    use std::borrow::Cow;

    /// Minimal mock weight provider for testing `materialize_all`.
    struct StubProvider {
        config: ModelConfig,
    }

    impl StubProvider {
        fn new() -> Self {
            Self {
                config: ModelConfig {
                    architecture: Architecture::Llama,
                    hidden_size: 64,
                    intermediate_size: 128,
                    num_hidden_layers: 1,
                    num_attention_heads: 4,
                    num_key_value_heads: 4,
                    head_dim: 16,
                    vocab_size: 32,
                    max_position_embeddings: 128,
                    rms_norm_eps: 1e-5,
                    rope_theta: 10000.0,
                    tie_word_embeddings: false,
                    extra: Default::default(),
                },
            }
        }
    }

    impl WeightProvider for StubProvider {
        fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError> {
            // Return deterministic data based on the key
            match name {
                "weight_a" => Ok(WeightTensor {
                    data: Cow::Owned(vec![10, 20, 30, 40]),
                    shape: vec![2, 2],
                    dtype: ScalarType::UInt8,
                    quant_info: crate::weights::QuantizationInfo::None,
                }),
                "weight_b" => Ok(WeightTensor {
                    data: Cow::Owned(vec![50, 60]),
                    shape: vec![2],
                    dtype: ScalarType::UInt8,
                    quant_info: crate::weights::QuantizationInfo::None,
                }),
                _ => Err(MilError::Validation(format!("unknown tensor: {name}"))),
            }
        }

        fn tensor_names(&self) -> Vec<&str> {
            vec!["weight_a", "weight_b"]
        }

        fn config(&self) -> &ModelConfig {
            &self.config
        }
    }

    #[test]
    fn test_materialize_all() {
        let mut program = Program::new("1.0");

        // Build a function with two ops containing External tensors
        let mut func = Function::new("main");
        let op_a = Operation::new("const", "const_a")
            .with_output("a_out")
            .with_input(
                "value",
                Value::Tensor {
                    data: TensorData::external("weight_a".to_string(), 4),
                    shape: vec![2, 2],
                    dtype: ScalarType::UInt8,
                },
            );
        let op_b = Operation::new("const", "const_b")
            .with_output("b_out")
            .with_input(
                "value",
                Value::Tensor {
                    data: TensorData::external("weight_b".to_string(), 2),
                    shape: vec![2],
                    dtype: ScalarType::UInt8,
                },
            );
        func.body.add_op(op_a);
        func.body.add_op(op_b);
        func.body.outputs = vec!["a_out".into(), "b_out".into()];
        program.add_function(func);

        // Attach stub provider and materialize
        program.set_weight_provider(Arc::new(StubProvider::new()));
        program
            .materialize_all()
            .expect("materialize should succeed");

        // Verify all tensors are now Inline with correct data
        let main = program.main().unwrap();
        for op in &main.body.operations {
            if let Some(Value::Tensor { data, .. }) = op.inputs.get("value") {
                assert!(data.is_inline(), "tensor in {} should be inline", op.name);
            }
        }
        let op_a = &main.body.operations[0];
        let op_b = &main.body.operations[1];
        if let Some(Value::Tensor { data, .. }) = op_a.inputs.get("value") {
            assert_eq!(data.as_bytes(), Some(&[10, 20, 30, 40][..]));
        }
        if let Some(Value::Tensor { data, .. }) = op_b.inputs.get("value") {
            assert_eq!(data.as_bytes(), Some(&[50, 60][..]));
        }
    }

    #[test]
    fn test_materialize_all_no_provider_errors() {
        let mut program = Program::new("1.0");
        let mut func = Function::new("main");
        let op = Operation::new("const", "c0").with_input(
            "value",
            Value::Tensor {
                data: TensorData::external("k".to_string(), 1),
                shape: vec![1],
                dtype: ScalarType::UInt8,
            },
        );
        func.body.add_op(op);
        program.add_function(func);

        let result = program.materialize_all();
        assert!(result.is_err(), "should fail without a weight provider");
    }

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

    /// Build a program with External tensor refs, run Fp16QuantizePass, and
    /// verify the output matches an identical program built with Inline tensors.
    #[test]
    fn external_tensor_quantize_matches_inline() {
        use crate::ir::pass::Pass;
        use crate::ir::passes::fp16_quantize::Fp16QuantizePass;

        // 4 floats = 16 bytes of FP32 data
        let fp32_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // --- Eager program (inline data) ---
        let mut eager = Program::new("1.0");
        let mut func_eager = Function::new("main");
        let op = Operation::new("const", "w").with_output("w_out").with_attr(
            "val",
            Value::Tensor {
                data: TensorData::inline(fp32_data.clone()),
                shape: vec![2, 2],
                dtype: ScalarType::Float32,
            },
        );
        func_eager.body.add_op(op);
        func_eager.body.outputs = vec!["w_out".into()];
        eager.add_function(func_eager);

        // --- Lazy program (external ref) ---
        let mut lazy = Program::new("1.0");
        let mut func_lazy = Function::new("main");
        let op = Operation::new("const", "w").with_output("w_out").with_attr(
            "val",
            Value::Tensor {
                data: TensorData::external("weight_fp32".to_string(), fp32_data.len()),
                shape: vec![2, 2],
                dtype: ScalarType::Float32,
            },
        );
        func_lazy.body.add_op(op);
        func_lazy.body.outputs = vec!["w_out".into()];
        lazy.add_function(func_lazy);

        // Provider that serves the FP32 tensor
        struct Fp32Provider {
            data: Vec<u8>,
            config: ModelConfig,
        }
        impl WeightProvider for Fp32Provider {
            fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, MilError> {
                if name == "weight_fp32" {
                    Ok(WeightTensor {
                        data: Cow::Borrowed(&self.data),
                        shape: vec![2, 2],
                        dtype: ScalarType::Float32,
                        quant_info: crate::weights::QuantizationInfo::None,
                    })
                } else {
                    Err(MilError::Validation(format!("unknown: {name}")))
                }
            }
            fn tensor_names(&self) -> Vec<&str> {
                vec!["weight_fp32"]
            }
            fn config(&self) -> &ModelConfig {
                &self.config
            }
        }

        lazy.set_weight_provider(Arc::new(Fp32Provider {
            data: fp32_data,
            config: ModelConfig {
                architecture: Architecture::Llama,
                hidden_size: 64,
                intermediate_size: 128,
                num_hidden_layers: 1,
                num_attention_heads: 4,
                num_key_value_heads: 4,
                head_dim: 16,
                vocab_size: 32,
                max_position_embeddings: 128,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                tie_word_embeddings: false,
                extra: Default::default(),
            },
        }));

        // Run FP16 quantization on both
        Fp16QuantizePass.run(&mut eager).expect("eager pass failed");
        Fp16QuantizePass.run(&mut lazy).expect("lazy pass failed");

        // Extract results and compare
        let eager_op = &eager.main().unwrap().body.operations[0];
        let lazy_op = &lazy.main().unwrap().body.operations[0];

        let eager_val = eager_op.attributes.get("val").expect("eager val");
        let lazy_val = lazy_op.attributes.get("val").expect("lazy val");

        if let (
            Value::Tensor {
                data: eager_data,
                shape: eager_shape,
                dtype: eager_dtype,
            },
            Value::Tensor {
                data: lazy_data,
                shape: lazy_shape,
                dtype: lazy_dtype,
            },
        ) = (eager_val, lazy_val)
        {
            assert_eq!(eager_dtype, lazy_dtype, "dtype mismatch");
            assert_eq!(
                *eager_dtype,
                ScalarType::Float16,
                "should be FP16 after pass"
            );
            assert_eq!(eager_shape, lazy_shape, "shape mismatch");
            assert!(lazy_data.is_inline(), "lazy tensor should be materialized");
            assert_eq!(
                eager_data.as_bytes(),
                lazy_data.as_bytes(),
                "quantized bytes differ between eager and lazy paths"
            );
        } else {
            panic!("expected Tensor values");
        }
    }
}
