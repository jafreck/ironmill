//! Safe FFI bindings for Apple Metal and MPS frameworks.
//!
//! `ironmill-metal-sys` provides safe Rust wrappers around Apple's Metal and
//! Metal Performance Shaders (MPS) Objective-C APIs.  All unsafe Metal
//! interaction is contained within this crate.
//!
//! # macOS Only
//!
//! This crate only compiles on macOS — a `compile_error!` is emitted on
//! other platforms.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_os = "macos"))]
compile_error!("ironmill-metal-sys only supports macOS");

pub mod buffer;
pub mod command;
pub mod device;
pub mod error;
pub mod mps;
pub mod pipeline;
pub mod shader;

pub(crate) mod objc;

pub use buffer::{MetalBuffer, StorageMode};
pub use command::{CommandBuffer, CommandBufferStatus, CommandQueue, ComputeEncoder};
pub use device::{GpuFamily, MetalDevice};
pub use error::MetalSysError;
pub use mps::{MpsMatrix, MpsMatrixMultiply, MpsMatrixMultiplyConfig};
pub use pipeline::ComputePipeline;
pub use shader::{FunctionConstantValues, ShaderFunction, ShaderLibrary};
