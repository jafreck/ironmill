use super::tensor::TensorType;

/// A value in the MIL IR — can be a reference to another op's output,
/// a literal constant, or a tensor type descriptor.
#[derive(Debug, Clone)]
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
}
