use mil_rs::ir::ScalarType;

pub mod bundle;
pub mod mil_text;
pub mod packing;

/// Describes an ANE tensor's name, shape, and element type.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDescriptor {
    /// Name identifying this tensor within the model.
    pub name: String,
    /// Four-dimensional shape of the tensor (N, C, H, W).
    pub shape: [usize; 4],
    /// Element data type of the tensor.
    pub dtype: ScalarType,
}
