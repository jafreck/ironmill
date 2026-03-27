//! # mil-rs
//!
//! Read, write, and manipulate Apple CoreML models in Rust — no Python required.
//!
//! `mil-rs` provides a strongly-typed MIL (Model Intermediate Language)
//! intermediate representation alongside protobuf serialization for
//! Apple's CoreML on-disk formats (`.mlmodel` and `.mlpackage`).
//!
//! ## Quick start
//!
//! ```no_run
//! use mil_rs::{read_mlmodel, model_to_program, program_to_model, write_mlpackage};
//!
//! // Read a .mlmodel and convert to the typed IR
//! let model = read_mlmodel("input.mlmodel").unwrap();
//! let program = model_to_program(&model).unwrap();
//!
//! // Inspect the program
//! for (name, _func) in &program.functions {
//!     println!("function: {name}");
//! }
//!
//! // Convert back and write as .mlpackage
//! let out = program_to_model(&program, model.specification_version as i32).unwrap();
//! write_mlpackage(&out, "output.mlpackage").unwrap();
//! ```
//!
//! ## Modules
//!
//! | Module      | Description |
//! |-------------|-------------|
//! | [`ir`]      | MIL intermediate representation — [`Program`], [`Function`], [`Block`], [`Operation`], [`Graph`], tensor types |
//! | [`proto`]   | Auto-generated protobuf types from Apple's CoreML `.proto` specification |
//! | [`reader`]  | Read `.mlmodel` and `.mlpackage` files into a protobuf [`Model`] |
//! | [`writer`]  | Write a protobuf [`Model`] to `.mlmodel` or `.mlpackage` |
//! | [`convert`] | Bidirectional conversion between protobuf [`Model`] and IR [`Program`] |
//! | [`error`]   | [`MilError`] enum and [`Result`](error::Result) type alias |
//!
//! ## Supported model types
//!
//! The proto ↔ IR conversion supports **ML Program** models (CoreML spec v7+).
//! Legacy `NeuralNetwork` models can be read and written at the protobuf level
//! but cannot be converted to the MIL IR.

pub mod convert;
pub mod error;
pub mod ir;
pub mod proto;
pub mod reader;
pub mod writer;

// Re-export key types at crate root for convenience.

/// Convert a protobuf [`Model`] into a MIL IR [`Program`].
pub use convert::model_to_program;
/// Convert a MIL IR [`Program`] back into a protobuf [`Model`].
pub use convert::program_to_model;

/// Error type for all operations in this crate.
pub use error::MilError;

pub use ir::{Block, Function, Graph, Operation, Pass, Program, ScalarType, TensorType, Value};

#[cfg(not(doctest))]
pub use proto::specification::Model;

/// Read a `.mlmodel` file into a protobuf [`Model`].
pub use reader::read_mlmodel;
/// Read a `.mlpackage` directory into a protobuf [`Model`].
pub use reader::read_mlpackage;
/// Read an ONNX `.onnx` file into an ONNX [`proto::onnx::ModelProto`].
pub use reader::read_onnx;

/// Write a protobuf [`Model`] to a `.mlmodel` file.
pub use writer::write_mlmodel;
/// Write a protobuf [`Model`] to a `.mlpackage` directory.
pub use writer::write_mlpackage;
