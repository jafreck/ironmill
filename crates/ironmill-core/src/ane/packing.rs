//! ANE input packing types.

/// Describes how multiple logical inputs were packed into a single tensor.
#[derive(Debug, Clone)]
pub struct InputPacking {
    /// Spatial offset for each original input within the packed tensor.
    pub offsets: Vec<usize>,
    /// Spatial size (S dimension) of each original input.
    pub sizes: Vec<usize>,
}
