//! File-level reader for SafeTensors model directories.

use std::path::Path;

use crate::MilError;
use crate::convert::weights::safetensors::SafeTensorsProvider;

/// Read a SafeTensors model directory. The directory should contain
/// `config.json` and one or more `.safetensors` files.
pub fn read_safetensors(model_dir: &Path) -> Result<SafeTensorsProvider, MilError> {
    SafeTensorsProvider::load(model_dir)
}
