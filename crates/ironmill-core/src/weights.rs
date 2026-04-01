//! Weight provider types, re-exported from mil-rs.
//!
//! These types are defined in `mil_rs::weights` and re-exported here so that
//! consumers can import shared types from `ironmill_core` without depending
//! on `mil-rs` directly.

pub use mil_rs::weights::{
    Architecture, ModelConfig, QuantizationInfo, WeightProvider, WeightTensor,
};
