//! Bidirectional conversion between protobuf types and the Rust-native MIL IR.
//!
//! - [`model_to_program`] converts a protobuf [`Model`](crate::proto::specification::Model) into
//!   a MIL IR [`Program`](crate::ir::Program).
//! - [`program_to_model`] converts a MIL IR [`Program`](crate::ir::Program) back into a protobuf
//!   [`Model`](crate::proto::specification::Model).

pub mod ir_to_proto;
pub mod proto_to_ir;

pub use ir_to_proto::program_to_model;
pub use proto_to_ir::model_to_program;
