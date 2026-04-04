use super::tensor::{ScalarType, TensorType};
use crate::error::MilError;

/// Storage backing for tensor data in the IR.
///
/// Small tensors (scalars, norms, RoPE tables) remain inline. Large weight
/// tensors can be backed by an external provider, deferring byte-level
/// access until a pass or writer actually needs the data.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    /// Tensor bytes are stored inline (owned).
    Inline(Vec<u8>),

    /// Tensor data lives in an external source. The `provider_key` is the
    /// canonical weight name used to retrieve it from a WeightProvider.
    External {
        /// Canonical weight name.
        provider_key: String,
        /// Byte length of the tensor data.
        byte_len: usize,
    },
}

impl TensorData {
    /// Create inline tensor data from owned bytes.
    pub fn inline(data: Vec<u8>) -> Self {
        TensorData::Inline(data)
    }

    /// Create an external tensor reference.
    pub fn external(provider_key: String, byte_len: usize) -> Self {
        TensorData::External {
            provider_key,
            byte_len,
        }
    }

    /// Returns the byte length of the tensor data without materializing.
    pub fn byte_len(&self) -> usize {
        match self {
            TensorData::Inline(data) => data.len(),
            TensorData::External { byte_len, .. } => *byte_len,
        }
    }

    /// Returns a reference to the inline bytes, or `None` if external.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            TensorData::Inline(data) => Some(data),
            TensorData::External { .. } => None,
        }
    }

    /// Returns a mutable reference to the inline bytes, or `None` if external.
    pub fn as_bytes_mut(&mut self) -> Option<&mut Vec<u8>> {
        match self {
            TensorData::Inline(data) => Some(data),
            TensorData::External { .. } => None,
        }
    }

    /// Returns `true` if the data is stored inline.
    pub fn is_inline(&self) -> bool {
        matches!(self, TensorData::Inline(_))
    }

    /// Returns `true` if the data is externally backed.
    pub fn is_external(&self) -> bool {
        matches!(self, TensorData::External { .. })
    }

    /// Consume this `TensorData` and return the inline bytes.
    ///
    /// # Panics
    /// Panics if the data is `External`. Callers must resolve first.
    pub fn into_bytes(self) -> Vec<u8> {
        match self {
            TensorData::Inline(data) => data,
            TensorData::External { provider_key, .. } => {
                panic!(
                    "cannot into_bytes() on External tensor '{provider_key}'; \
                     call resolve_with() or materialize_with() first"
                )
            }
        }
    }

    /// Consume this `TensorData` and return the inline bytes, resolving
    /// external data via the provided closure if necessary.
    pub fn resolve_with<F>(self, loader: F) -> Result<Vec<u8>, MilError>
    where
        F: FnOnce(&str) -> Result<Vec<u8>, MilError>,
    {
        match self {
            TensorData::Inline(data) => Ok(data),
            TensorData::External { provider_key, .. } => loader(&provider_key),
        }
    }

    /// Resolve external data in-place using the provided closure.
    /// No-op if already inline.
    pub fn materialize_with<F>(&mut self, loader: F) -> Result<(), MilError>
    where
        F: FnOnce(&str) -> Result<Vec<u8>, MilError>,
    {
        if let TensorData::External {
            ref provider_key, ..
        } = *self
        {
            let key = provider_key.clone();
            let data = loader(&key)?;
            *self = TensorData::Inline(data);
        }
        Ok(())
    }
}

impl From<Vec<u8>> for TensorData {
    fn from(data: Vec<u8>) -> Self {
        TensorData::Inline(data)
    }
}

/// A value in the MIL IR — can be a reference to another op's output,
/// a literal constant, or a tensor type descriptor.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Reference to another operation's output by name.
    Reference(String),

    /// An integer constant.
    Int(i64),

    /// A floating-point constant.
    Float(f64),

    /// A boolean constant.
    Bool(bool),

    /// A string constant.
    String(String),

    /// A list of values (e.g., for shapes, axes, padding).
    List(Vec<Value>),

    /// A tensor type descriptor (used in type annotations).
    Type(TensorType),

    /// Raw tensor data (for weights/constants from ONNX initializers).
    Tensor {
        /// Tensor data — inline or externally backed.
        data: TensorData,
        /// Dimensions of the tensor.
        shape: Vec<usize>,
        /// Element data type.
        dtype: ScalarType,
    },
}

impl Value {
    /// Convenience constructor for inline tensor values.
    pub fn tensor(data: Vec<u8>, shape: Vec<usize>, dtype: ScalarType) -> Self {
        Value::Tensor {
            data: TensorData::Inline(data),
            shape,
            dtype,
        }
    }
}
