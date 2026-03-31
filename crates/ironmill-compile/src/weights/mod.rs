//! Weight provider abstraction layer.
//!
//! Core types ([`Architecture`], [`ModelConfig`], [`WeightTensor`],
//! [`WeightProvider`]) live in `mil_rs::weights` and are re-exported here.
//! Format-specific providers ([`GgufProvider`], [`SafeTensorsProvider`])
//! are defined in this crate.

pub use mil_rs::weights::{
    Architecture, ModelConfig, QuantizationInfo, WeightProvider, WeightTensor,
};

pub mod gguf;
pub mod safetensors;

pub use gguf::GgufProvider;
pub use safetensors::SafeTensorsProvider;
