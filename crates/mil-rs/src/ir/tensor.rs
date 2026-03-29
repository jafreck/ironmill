/// Tensor data types supported by MIL / CoreML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool,
}

impl ScalarType {
    /// Returns the number of bytes required to store a single element of this type.
    pub fn byte_size(&self) -> usize {
        match self {
            ScalarType::Bool | ScalarType::Int8 | ScalarType::UInt8 => 1,
            ScalarType::Float16 | ScalarType::Int16 | ScalarType::UInt16 => 2,
            ScalarType::Float32 | ScalarType::Int32 | ScalarType::UInt32 => 4,
            ScalarType::Float64 | ScalarType::Int64 | ScalarType::UInt64 => 8,
        }
    }
}

/// A tensor type descriptor — shape + element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    /// Element data type.
    pub scalar_type: ScalarType,

    /// Shape dimensions. `None` entries represent dynamic dimensions.
    pub shape: Vec<Option<usize>>,
}

impl TensorType {
    /// Create a tensor type with a fully static shape.
    pub fn new(scalar_type: ScalarType, shape: Vec<usize>) -> Self {
        Self {
            scalar_type,
            shape: shape.into_iter().map(Some).collect(),
        }
    }

    /// Create a tensor type with potentially dynamic dimensions.
    pub fn with_dynamic_shape(scalar_type: ScalarType, shape: Vec<Option<usize>>) -> Self {
        Self { scalar_type, shape }
    }

    /// Returns `true` if all dimensions are statically known.
    pub fn is_static(&self) -> bool {
        self.shape.iter().all(|d| d.is_some())
    }

    /// Returns the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}
