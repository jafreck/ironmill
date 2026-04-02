use mil_rs::ir::ScalarType;

pub mod bundle;
pub mod mil_text;
pub mod packing;

/// Describes an ANE tensor's name, shape, and element type.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: [usize; 4],
    pub dtype: ScalarType,
}
