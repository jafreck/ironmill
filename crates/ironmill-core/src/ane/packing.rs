//! ANE input packing types.

use super::bundle::BundleInputPacking;

/// Describes how multiple logical inputs were packed into a single tensor.
#[derive(Debug, Clone)]
pub struct InputPacking {
    /// Spatial offset for each original input within the packed tensor.
    pub offsets: Vec<usize>,
    /// Spatial size (S dimension) of each original input.
    pub sizes: Vec<usize>,
}

impl From<BundleInputPacking> for InputPacking {
    fn from(m: BundleInputPacking) -> Self {
        Self {
            offsets: m.offsets,
            sizes: m.sizes,
        }
    }
}
