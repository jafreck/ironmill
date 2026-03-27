//! Writers for various CoreML model formats.

pub mod mlmodel;
pub mod mlpackage;

pub use mlmodel::write_mlmodel;
pub use mlpackage::write_mlpackage;
