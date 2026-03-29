//! File-level reader for GGUF model files.

use std::path::Path;

use crate::MilError;
use crate::convert::weights::gguf::GgufProvider;

/// Read a GGUF model file. For split-shard models, pass any shard path.
pub fn read_gguf(gguf_path: &Path) -> Result<GgufProvider, MilError> {
    GgufProvider::load(gguf_path)
}
