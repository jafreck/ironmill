use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use indexmap::IndexMap;
use tempfile::TempDir;

use super::operation::Operation;
use super::tensor::TensorType;
use super::types::{TensorData, Value};
use crate::error::MilError;
use crate::weights::WeightProvider;

use std::fmt;

/// A complete MIL program — the top-level container for a CoreML ML Program model.
///
/// A program contains one or more named functions. The "main" function is
/// the entry point for inference.
#[non_exhaustive]
pub struct Program {
    /// Program version (e.g., "1.0.0").
    pub version: String,
    /// Named functions in this program. Key is the function name.
    pub functions: IndexMap<String, Function>,
    /// Program-level attributes (e.g., `autoregressive`, `max_seq_length`).
    pub attributes: HashMap<String, String>,

    /// Optional weight provider for resolving `TensorData::External` references.
    pub(crate) weight_provider: Option<Arc<dyn WeightProvider + Send + Sync>>,

    /// Temp directory for spilled tensor data (auto-cleaned on drop).
    /// Not cloned — each Program gets its own spill state.
    pub(crate) spill_dir: Option<TempDir>,
    /// Maps spill keys to file paths for resolution.
    pub(crate) spill_index: HashMap<String, PathBuf>,
}

impl Clone for Program {
    fn clone(&self) -> Self {
        Self {
            version: self.version.clone(),
            functions: self.functions.clone(),
            attributes: self.attributes.clone(),
            weight_provider: self.weight_provider.clone(),
            spill_dir: None,
            spill_index: HashMap::new(),
        }
    }
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
            .field("spill_dir", &self.spill_dir.is_some())
            .field("spill_index_len", &self.spill_index.len())
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
            spill_dir: None,
            spill_index: HashMap::new(),
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
        // Check spill directory first (quantized data written between passes)
        if let Some(path) = self.spill_index.get(key) {
            return std::fs::read(path).map_err(MilError::Io);
        }
        // Fall back to weight provider (original weight data)
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
    ///
    /// Resolves from the spill index first (for data written between passes),
    /// then falls back to the weight provider (for original weight data).
    /// Returns `Ok(())` immediately if no External tensors exist.
    pub fn materialize_all(&mut self) -> Result<(), MilError> {
        let provider = self.weight_provider.clone();
        let spill_index = self.spill_index.clone();

        // Check if there are any External tensors to materialize.
        let has_external = self.functions.values().any(|f| {
            f.body.operations.iter().any(|op| {
                op.inputs
                    .values()
                    .chain(op.attributes.values())
                    .any(|v| matches!(v, Value::Tensor { data, .. } if data.is_external()))
            })
        });
        if !has_external {
            return Ok(());
        }

        // Need at least a spill index or provider to resolve External refs.
        if provider.is_none() && spill_index.is_empty() {
            return Err(MilError::Validation(
                "no weight provider or spill index for materialization".into(),
            ));
        }

        for function in self.functions.values_mut() {
            Self::materialize_block(&mut function.body, provider.as_deref(), &spill_index)?;
        }
        Ok(())
    }

    fn materialize_block(
        block: &mut Block,
        provider: Option<&(dyn WeightProvider + Send + Sync)>,
        spill_index: &HashMap<String, PathBuf>,
    ) -> Result<(), MilError> {
        for op in &mut block.operations {
            for val in op.inputs.values_mut().chain(op.attributes.values_mut()) {
                if let Value::Tensor { data, .. } = val {
                    data.materialize_with(|key| {
                        // Check spill first
                        if let Some(path) = spill_index.get(key) {
                            return std::fs::read(path).map_err(MilError::Io);
                        }
                        let p = provider.ok_or_else(|| {
                            MilError::Validation(format!(
                                "no weight provider attached; cannot resolve tensor '{key}'"
                            ))
                        })?;
                        let tensor = p.tensor(key)?;
                        Ok(tensor.data.into_owned())
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Spill large inline tensors to a temp directory, replacing them with
    /// `TensorData::External` references. Called between passes to bound
    /// peak memory to one unquantized tensor at a time.
    ///
    /// Tensors smaller than `min_bytes` are kept inline (scalars, norms, biases).
    pub fn spill_inline_tensors(&mut self, min_bytes: usize) -> Result<(), MilError> {
        // Lazily create the spill directory
        if self.spill_dir.is_none() {
            self.spill_dir = Some(tempfile::tempdir().map_err(|e| {
                MilError::Validation(format!("failed to create spill directory: {e}"))
            })?);
        }
        let dir = self.spill_dir.as_ref().unwrap().path().to_owned();
        let mut counter = self.spill_index.len();

        for function in self.functions.values_mut() {
            Self::spill_block(
                &mut function.body,
                &dir,
                min_bytes,
                &mut counter,
                &mut self.spill_index,
            )?;
        }
        Ok(())
    }

    fn spill_block(
        block: &mut Block,
        dir: &std::path::Path,
        min_bytes: usize,
        counter: &mut usize,
        index: &mut HashMap<String, PathBuf>,
    ) -> Result<(), MilError> {
        for op in &mut block.operations {
            for val in op.inputs.values_mut().chain(op.attributes.values_mut()) {
                if let Value::Tensor { data, .. } = val {
                    if data.is_inline() && data.byte_len() >= min_bytes {
                        let key = format!("_spill_{counter}");
                        *counter += 1;
                        let path = dir.join(&key);
                        let bytes = data.as_bytes().unwrap();
                        std::fs::write(&path, bytes).map_err(MilError::Io)?;
                        let byte_len = bytes.len();
                        *data = TensorData::external(key.clone(), byte_len);
                        index.insert(key, path);
                    }
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
#[non_exhaustive]
pub struct Function {
    /// Function name.
    pub name: String,
    /// Ordered named inputs with their types.
    pub inputs: Vec<(String, TensorType)>,
    /// The function body.
    pub body: Block,
    /// Function-level attributes/metadata from the proto.
    pub attributes: HashMap<String, String>,
}

impl Function {
    /// Create a new function with an empty body.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inputs: Vec::new(),
            body: Block::new(),
            attributes: HashMap::new(),
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
#[non_exhaustive]
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

    /// Create a block with the given operations and outputs.
    pub fn with_operations(operations: Vec<Operation>, outputs: Vec<String>) -> Self {
        Self {
            operations,
            outputs,
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
                    query_pre_attn_scalar: 0,
                    hidden_act: "silu".to_string(),
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
                query_pre_attn_scalar: 0,
                hidden_act: "silu".to_string(),
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

    #[test]
    fn test_spill_and_rematerialize() {
        // Build a program with a large inline tensor
        let mut program = Program::new("1.0");
        let mut func = Function::new("main");
        let big_data = vec![42u8; 8192]; // Above 4096 threshold
        let op = Operation::new("const", "big_const")
            .with_output("big_out")
            .with_attr(
                "val",
                Value::Tensor {
                    data: TensorData::inline(big_data.clone()),
                    shape: vec![8192],
                    dtype: ScalarType::UInt8,
                },
            );
        func.body.add_op(op);
        func.body.outputs = vec!["big_out".into()];
        program.add_function(func);

        // Spill with threshold = 4096
        program
            .spill_inline_tensors(4096)
            .expect("spill should succeed");

        // Verify the tensor is now External
        let main = program.main().unwrap();
        let op = &main.body.operations[0];
        if let Some(Value::Tensor { data, .. }) = op.attributes.get("val") {
            assert!(data.is_external(), "should be spilled to disk");
            assert_eq!(data.byte_len(), 8192);
        } else {
            panic!("expected tensor attribute");
        }

        // Attach a dummy provider and materialize
        program.set_weight_provider(Arc::new(StubProvider::new()));
        program
            .materialize_all()
            .expect("materialize should succeed");

        // Verify data is back
        let main = program.main().unwrap();
        let op = &main.body.operations[0];
        if let Some(Value::Tensor { data, .. }) = op.attributes.get("val") {
            assert!(data.is_inline(), "should be materialized");
            assert_eq!(data.as_bytes().unwrap(), &big_data[..]);
        } else {
            panic!("expected tensor attribute after materialize");
        }
    }

    #[test]
    fn test_spill_skips_small_tensors() {
        let mut program = Program::new("1.0");
        let mut func = Function::new("main");
        let small_data = vec![1u8; 100]; // Below 4096 threshold
        let op = Operation::new("const", "small_const")
            .with_output("small_out")
            .with_attr(
                "val",
                Value::Tensor {
                    data: TensorData::inline(small_data.clone()),
                    shape: vec![100],
                    dtype: ScalarType::UInt8,
                },
            );
        func.body.add_op(op);
        func.body.outputs = vec!["small_out".into()];
        program.add_function(func);

        program
            .spill_inline_tensors(4096)
            .expect("spill should succeed");

        // Small tensor should remain inline
        let main = program.main().unwrap();
        let op = &main.body.operations[0];
        if let Some(Value::Tensor { data, .. }) = op.attributes.get("val") {
            assert!(data.is_inline(), "small tensor should stay inline");
            assert_eq!(data.as_bytes().unwrap(), &small_data[..]);
        } else {
            panic!("expected tensor attribute");
        }
    }
}
