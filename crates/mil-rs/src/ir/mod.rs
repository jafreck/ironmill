//! The MIL Intermediate Representation.
//!
//! This module defines the core graph-based IR that sits between model input formats
//! (ONNX, SafeTensors, GGUF) and CoreML's protobuf output format.
//!
//! The design mirrors Apple's Model Intermediate Language (MIL) as documented in
//! `coremltools`, but implemented as idiomatic Rust with strong typing.

mod graph;
mod operation;
mod tensor;
mod types;

pub use graph::Graph;
pub use operation::Operation;
pub use tensor::TensorType;
pub use types::Value;
