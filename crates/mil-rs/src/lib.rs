//! # mil-rs
//!
//! Rust implementation of Apple's Model Intermediate Language (MIL) IR and CoreML protobuf format.
//!
//! This crate provides the foundational data structures for working with CoreML models
//! in Rust — without any Python dependency.
//!
//! ## Modules
//!
//! - [`ir`] — The MIL intermediate representation (graph, operations, types, tensors)
//! - [`error`] — Error types for the crate

pub mod error;
pub mod ir;

/// Re-export key types at crate root for convenience.
pub use error::MilError;
pub use ir::{Graph, Operation, TensorType, Value};
