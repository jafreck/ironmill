//! JIT compilation types and tensor transform pipeline (§11).
//!
//! Defines the [`TensorTransform`] trait and [`TransformPipeline`] for
//! applying quantization / casting transforms to weight tensors during
//! model loading. Actual GPU kernel compilation is future work.

use mil_rs::ir::ScalarType;
use mil_rs::weights::{ModelConfig, QuantizationInfo, WeightTensor};

/// A transform applied directly to weight tensors during loading.
pub trait TensorTransform: Send + Sync {
    /// Human-readable name for diagnostics and logging.
    fn name(&self) -> &str;

    /// Apply the transform to `tensor`.
    ///
    /// Returns `Ok(Some(..))` with the transformed result, `Ok(None)` to
    /// skip (leave unchanged), or `Err` on failure.
    fn transform(
        &self,
        name: &str,
        tensor: WeightTensor<'_>,
        config: &ModelConfig,
    ) -> Result<Option<TransformedTensor>, TransformError>;
}

/// Result of a tensor transform.
#[non_exhaustive]
pub struct TransformedTensor {
    /// Raw bytes of the transformed tensor data.
    pub data: Vec<u8>,
    /// Dimensions of the tensor.
    pub shape: Vec<usize>,
    /// Element data type after transformation.
    pub dtype: ScalarType,
    /// Quantization parameters applied during the transform.
    pub quant_info: QuantizationInfo,
}

/// Error during tensor transformation.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    /// A transform step failed.
    #[error("transform failed: {0}")]
    Failed(String),
    /// The tensor type or layout is not supported by this transform.
    #[error("unsupported tensor: {0}")]
    Unsupported(String),
    /// An I/O error occurred during transformation.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Pipeline of tensor transforms, applied during weight loading.
pub struct TransformPipeline {
    transforms: Vec<Box<dyn TensorTransform>>,
}

impl TransformPipeline {
    /// Create a new empty transform pipeline.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Append a transform step to the pipeline.
    pub fn with_transform(mut self, transform: Box<dyn TensorTransform>) -> Self {
        self.transforms.push(transform);
        self
    }

    /// Add an int4 affine quantization transform (not yet implemented).
    #[allow(unused)]
    pub fn with_int4(self, _group_size: usize) -> Result<Self, TransformError> {
        // TODO: Implement Int4AffineTransform — requires int4 packing layout
        // and affine dequantization parameters (scale, zero_point per group).
        Err(TransformError::Unsupported(
            "Int4AffineTransform not yet implemented".into(),
        ))
    }

    /// Add a float16 cast transform (not yet implemented).
    #[allow(unused)]
    pub fn with_fp16(self) -> Result<Self, TransformError> {
        // TODO: Implement Fp16CastTransform — cast each weight tensor
        // element from its source dtype (f32, bf16, etc.) to fp16.
        Err(TransformError::Unsupported(
            "Fp16CastTransform not yet implemented".into(),
        ))
    }

    /// Add a polar quantization transform (not yet implemented).
    #[allow(unused)]
    pub fn with_polar_quant(self, _bits: u8) -> Result<Self, TransformError> {
        // TODO: Implement PolarQuantTransform — requires polar coordinate
        // encoding of weight matrices and appropriate dequant kernel support.
        Err(TransformError::Unsupported(
            "PolarQuantTransform not yet implemented".into(),
        ))
    }

    /// Return a slice of the registered transforms.
    pub fn transforms(&self) -> &[Box<dyn TensorTransform>] {
        &self.transforms
    }
}

impl Default for TransformPipeline {
    fn default() -> Self {
        Self::new()
    }
}
