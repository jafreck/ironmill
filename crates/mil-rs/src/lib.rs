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
//! - [`proto`] — Auto-generated protobuf types for the CoreML specification
//! - [`error`] — Error types for the crate

pub mod error;
pub mod ir;
pub mod proto;
pub mod reader;

/// Re-export key types at crate root for convenience.
pub use error::MilError;
pub use ir::{Block, Function, Graph, Operation, Pass, Program, TensorType, Value};
#[cfg(not(doctest))]
pub use proto::specification::Model;
pub use reader::read_mlmodel;
