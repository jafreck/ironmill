//! Bidirectional conversion between protobuf types and the Rust-native MIL IR.
//!
//! - [`model_to_program`] converts a protobuf [`Model`](crate::proto::specification::Model) into
//!   a MIL IR [`Program`](crate::ir::Program).
//! - [`program_to_model`] converts a MIL IR [`Program`](crate::ir::Program) back into a protobuf
//!   [`Model`](crate::proto::specification::Model).
//! - [`onnx_to_mil`] converts individual ONNX [`NodeProto`](crate::proto::onnx::NodeProto)
//!   operations into MIL IR [`Operation`](crate::ir::Operation)s.
//! - [`onnx_graph`] converts an entire ONNX [`ModelProto`](crate::proto::onnx::ModelProto)
//!   into a MIL IR [`Program`](crate::ir::Program).

pub mod ir_to_proto;
pub mod onnx_graph;
pub mod onnx_to_mil;
pub mod proto_to_ir;

pub use ir_to_proto::program_to_model;
pub use onnx_graph::onnx_to_program;
pub use onnx_to_mil::convert_node;
pub use proto_to_ir::model_to_program;
