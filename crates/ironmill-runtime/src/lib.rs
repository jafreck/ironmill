#![forbid(unsafe_code)]

//! Common runtime traits for ironmill backends.
//!
//! Provides a unified interface for compiling and running inference across
//! different backends (CoreML via xcrun, direct ANE, etc.). Consumers write
//! backend-agnostic code against [`RuntimeBackend`] and [`RuntimeModel`],
//! then select a concrete implementation at runtime.
//!
//! # Example
//!
//! ```ignore
//! use ironmill_runtime::{RuntimeBackend, RuntimeModel, RuntimeTensor};
//!
//! fn benchmark(backends: &[Box<dyn RuntimeBackend>], program: &Program) {
//!     for backend in backends {
//!         let model = backend.compile(program)?;
//!         let inputs = build_dummy_inputs(&model.input_description());
//!         let outputs = model.predict(&inputs)?;
//!     }
//! }
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

/// Element data type for runtime tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    Float32,
    Float16,
    Int32,
    Float64,
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float32 => write!(f, "float32"),
            Self::Float16 => write!(f, "float16"),
            Self::Int32 => write!(f, "int32"),
            Self::Float64 => write!(f, "float64"),
        }
    }
}

impl ElementType {
    /// Number of bytes per element.
    pub fn byte_size(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
            Self::Int32 => 4,
            Self::Float64 => 8,
        }
    }
}

/// A named tensor for model I/O.
///
/// This is a simple owned buffer — backends convert to/from their native
/// tensor representations (e.g. `MLMultiArray`, IOSurface-backed buffers).
#[derive(Debug, Clone)]
pub struct RuntimeTensor {
    /// Feature name (matches model input/output descriptions).
    pub name: String,
    /// Raw bytes in native endian order.
    pub data: Vec<u8>,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: ElementType,
}

impl RuntimeTensor {
    /// Create a new tensor with the given name, shape, and dtype, filled with zeros.
    pub fn zeros(name: impl Into<String>, shape: Vec<usize>, dtype: ElementType) -> Self {
        let numel: usize = shape.iter().product();
        let data = vec![0u8; numel * dtype.byte_size()];
        Self {
            name: name.into(),
            data,
            shape,
            dtype,
        }
    }

    /// Number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

// ---------------------------------------------------------------------------
// Input description
// ---------------------------------------------------------------------------

/// Description of a single model input feature.
#[derive(Debug, Clone)]
pub struct InputFeatureDesc {
    /// Feature name.
    pub name: String,
    /// Expected shape (0 = dynamic dimension).
    pub shape: Vec<usize>,
    /// Expected element type.
    pub dtype: ElementType,
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// A compiled model that can run inference.
///
/// Implementations hold whatever native state is needed (loaded CoreML model,
/// allocated ANE buffers, etc.) and convert between [`RuntimeTensor`] and
/// their internal representations.
pub trait RuntimeModel {
    /// Describe the model's expected inputs.
    fn input_description(&self) -> Vec<InputFeatureDesc>;

    /// Run inference on the given inputs and return the outputs.
    fn predict(&self, inputs: &[RuntimeTensor]) -> anyhow::Result<Vec<RuntimeTensor>>;
}

/// A backend that can compile an IR program into a runnable model.
///
/// Each backend (CoreML, ANE direct, etc.) implements this trait. The CLI
/// and benchmark harness select a backend at runtime and use it through
/// this uniform interface.
pub trait RuntimeBackend: Send + Sync {
    /// Human-readable name for this backend (e.g. "coreml", "ane-direct").
    fn name(&self) -> &str;

    /// Compile a model from the given path and return a loaded [`RuntimeModel`].
    ///
    /// The `model_path` may be a `.mlpackage`, `.mlmodelc`, or other format
    /// depending on the backend.
    fn load(&self, model_path: &std::path::Path) -> anyhow::Result<Box<dyn RuntimeModel>>;
}

/// Build dummy input tensors from input descriptions (for benchmarking).
pub fn build_dummy_inputs(desc: &[InputFeatureDesc]) -> Vec<RuntimeTensor> {
    desc.iter()
        .map(|d| {
            let shape: Vec<usize> = d
                .shape
                .iter()
                .map(|&s| if s == 0 { 1 } else { s })
                .collect();
            RuntimeTensor::zeros(&d.name, shape, d.dtype)
        })
        .collect()
}
