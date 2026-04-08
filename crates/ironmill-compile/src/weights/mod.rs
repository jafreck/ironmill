//! Weight provider abstraction layer.
//!
//! Core types ([`Architecture`], [`ModelConfig`], [`WeightTensor`],
//! [`WeightProvider`]) live in `mil_rs::weights` and are re-exported here.
//! Format-specific providers ([`GgufProvider`], [`SafeTensorsProvider`])
//! are defined in this crate.

use memmap2::Mmap;

pub use mil_rs::weights::{
    Architecture, ModelConfig, QuantizationInfo, WeightProvider, WeightTensor,
};

pub mod calibration;
pub mod gguf;
pub mod mil_provider;
pub mod quantized;
pub mod safetensors;

pub use gguf::GgufProvider;
pub use mil_provider::MilWeightProvider;
pub use quantized::QuantizedWeightProvider;
pub use safetensors::SafeTensorsProvider;

/// Memory-maps a file for read-only access.
///
/// # Safety
///
/// This function assumes the file will not be modified or truncated while the
/// mapping is alive. This is acceptable here because model weight
/// files are read-only assets that are never modified during loading.
#[allow(unsafe_code)]
pub(crate) fn mmap_read_only(file: &std::fs::File) -> std::io::Result<Mmap> {
    // SAFETY: We use mmap for zero-copy access to large model files.  The
    // safety invariant is that the underlying file must not be modified or
    // truncated while the mapping exists.  Model weight files are read-only
    // assets that are not mutated during loading, so this invariant holds.
    unsafe { Mmap::map(file) }
}
