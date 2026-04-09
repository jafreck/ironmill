//! Core types for ironmill inference — tensors, element types, feature descriptions.
//!
//! These types provide the runtime-agnostic data model shared by both ANE
//! and CoreML backends. Originally defined in `ironmill-runtime`, they now
//! live here as the canonical location.

use std::fmt;

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

/// Element data type for runtime tensors.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    /// 32-bit IEEE 754 floating point.
    Float32,
    /// 16-bit IEEE 754 floating point.
    Float16,
    /// 16-bit brain floating point.
    BFloat16,
    /// 32-bit signed integer.
    Int32,
    /// 8-bit signed integer.
    Int8,
    /// 64-bit IEEE 754 floating point.
    Float64,
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float32 => write!(f, "float32"),
            Self::Float16 => write!(f, "float16"),
            Self::BFloat16 => write!(f, "bfloat16"),
            Self::Int32 => write!(f, "int32"),
            Self::Int8 => write!(f, "int8"),
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
            Self::BFloat16 => 2,
            Self::Int32 => 4,
            Self::Int8 => 1,
            Self::Float64 => 8,
        }
    }
}

/// A named tensor for model I/O.
///
/// This is a simple owned buffer — backends convert to/from their native
/// tensor representations (e.g. `MLMultiArray`, IOSurface-backed buffers).
#[non_exhaustive]
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
#[non_exhaustive]
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
    fn predict(
        &self,
        inputs: &[RuntimeTensor],
    ) -> Result<Vec<RuntimeTensor>, crate::engine::InferenceError>;
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
    fn load(
        &self,
        model_path: &std::path::Path,
    ) -> Result<Box<dyn RuntimeModel>, crate::engine::InferenceError>;
}

// ---------------------------------------------------------------------------
// Logits
// ---------------------------------------------------------------------------

/// Raw (unnormalized) logit scores from model inference.
#[derive(Debug, Clone)]
pub struct Logits(Vec<f32>);

impl Logits {
    /// Create a new `Logits` from a raw vector of scores.
    pub fn new(data: Vec<f32>) -> Self {
        Self(data)
    }
    /// Consume the newtype and return the inner vector.
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }
    /// Number of logit scores.
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Returns `true` if there are no logit scores.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::ops::Deref for Logits {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.0
    }
}

impl std::ops::DerefMut for Logits {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.0
    }
}
