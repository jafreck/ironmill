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
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: ScalarType,
    pub quant_info: QuantizationInfo,
}

/// Error during tensor transformation.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    #[error("transform failed: {0}")]
    Failed(String),
    #[error("unsupported tensor: {0}")]
    Unsupported(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Pipeline of tensor transforms, applied during weight loading.
pub struct TransformPipeline {
    transforms: Vec<Box<dyn TensorTransform>>,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    pub fn with_transform(mut self, transform: Box<dyn TensorTransform>) -> Self {
        self.transforms.push(transform);
        self
    }

    pub fn with_int4(self, _group_size: usize) -> Self {
        // TODO: Add Int4AffineTransform
        self
    }

    pub fn with_fp16(self) -> Self {
        // TODO: Add Fp16CastTransform
        self
    }

    pub fn with_polar_quant(self, _bits: u8) -> Self {
        // TODO: Add PolarQuantTransform
        self
    }

    pub fn transforms(&self) -> &[Box<dyn TensorTransform>] {
        &self.transforms
    }
}

impl Default for TransformPipeline {
    fn default() -> Self {
        Self::new()
    }
}
