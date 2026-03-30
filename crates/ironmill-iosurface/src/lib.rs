//! Safe IOSurface tensor management for Apple Neural Engine I/O.
//!
//! This crate provides a safe Rust API over macOS IOSurface memory for ANE
//! tensor allocation, data transfer, and lifecycle management with correct
//! padding and alignment per ANE constraints.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_os = "macos"))]
compile_error!("ironmill-iosurface requires macOS (IOSurface.framework)");

mod surface;
mod tensor;

pub use surface::IOSurfaceError;
pub use tensor::{AneTensor, uniform_alloc_size};

/// Result type alias for IOSurface operations.
pub type Result<T> = std::result::Result<T, IOSurfaceError>;
