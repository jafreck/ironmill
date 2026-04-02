//! Safe Rust bindings to mlx-c (the C API for Apple's MLX framework).
//!
//! `ironmill-mlx-sys` provides safe wrappers around the mlx-c FFI.  All
//! unsafe MLX interaction is contained within this crate.
//!
//! # macOS Only
//!
//! This crate only compiles on macOS — a `compile_error!` is emitted on
//! other platforms.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_os = "macos"))]
compile_error!("ironmill-mlx-sys only supports macOS");

/// N-dimensional array type backed by MLX.
pub mod array;
/// Device selection (CPU / GPU).
pub mod device;
/// Error types for MLX operations.
pub mod error;
/// Fused operations (RMS norm, RoPE, scaled dot-product attention).
pub mod fast_ops;
/// Custom Metal kernel dispatch.
pub mod metal_kernel;
/// Element-wise and tensor operations.
pub mod ops;
/// Stream (execution queue) management.
pub mod stream;

pub use array::MlxArray;
pub use device::MlxDevice;
pub use error::MlxSysError;
pub use fast_ops::{rms_norm, rope, scaled_dot_product_attention};
pub use metal_kernel::{MetalKernelParams, metal_kernel};
pub use ops::{
    add, broadcast_to, concat, expand_dims, matmul, multiply, reshape, silu, slice, transpose,
    transpose_axes,
};
pub use stream::MlxStream;

// ---------------------------------------------------------------------------
// MlxDtype — maps to mlx_dtype_* constants
// ---------------------------------------------------------------------------

/// Element data types for MLX arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MlxDtype {
    /// Boolean.
    Bool = 0,
    /// Unsigned 8-bit integer.
    Uint8 = 1,
    /// Unsigned 16-bit integer.
    Uint16 = 2,
    /// Unsigned 32-bit integer.
    Uint32 = 3,
    /// Unsigned 64-bit integer.
    Uint64 = 4,
    /// Signed 8-bit integer.
    Int8 = 5,
    /// Signed 16-bit integer.
    Int16 = 6,
    /// Signed 32-bit integer.
    Int32 = 7,
    /// Signed 64-bit integer.
    Int64 = 8,
    /// IEEE 754 half precision (16-bit float).
    Float16 = 9,
    /// IEEE 754 single precision (32-bit float).
    Float32 = 10,
    /// IEEE 754 double precision (64-bit float).
    Float64 = 11,
    /// Brain floating point (16-bit).
    Bfloat16 = 12,
    /// Complex 64-bit (two 32-bit floats).
    Complex64 = 13,
}

impl MlxDtype {
    /// Returns the size of a single element of this dtype in bytes.
    pub fn size(self) -> usize {
        match self {
            MlxDtype::Bool | MlxDtype::Uint8 | MlxDtype::Int8 => 1,
            MlxDtype::Uint16 | MlxDtype::Int16 | MlxDtype::Float16 | MlxDtype::Bfloat16 => 2,
            MlxDtype::Uint32 | MlxDtype::Int32 | MlxDtype::Float32 => 4,
            MlxDtype::Uint64 | MlxDtype::Int64 | MlxDtype::Float64 | MlxDtype::Complex64 => 8,
        }
    }

    /// Returns a human-readable name for this dtype.
    pub fn name(self) -> &'static str {
        match self {
            MlxDtype::Bool => "bool",
            MlxDtype::Uint8 => "uint8",
            MlxDtype::Uint16 => "uint16",
            MlxDtype::Uint32 => "uint32",
            MlxDtype::Uint64 => "uint64",
            MlxDtype::Int8 => "int8",
            MlxDtype::Int16 => "int16",
            MlxDtype::Int32 => "int32",
            MlxDtype::Int64 => "int64",
            MlxDtype::Float16 => "float16",
            MlxDtype::Float32 => "float32",
            MlxDtype::Float64 => "float64",
            MlxDtype::Bfloat16 => "bfloat16",
            MlxDtype::Complex64 => "complex64",
        }
    }

    /// Attempt to convert from a raw u32 value.
    pub fn from_raw(value: u32) -> Option<Self> {
        match value {
            0 => Some(MlxDtype::Bool),
            1 => Some(MlxDtype::Uint8),
            2 => Some(MlxDtype::Uint16),
            3 => Some(MlxDtype::Uint32),
            4 => Some(MlxDtype::Uint64),
            5 => Some(MlxDtype::Int8),
            6 => Some(MlxDtype::Int16),
            7 => Some(MlxDtype::Int32),
            8 => Some(MlxDtype::Int64),
            9 => Some(MlxDtype::Float16),
            10 => Some(MlxDtype::Float32),
            11 => Some(MlxDtype::Float64),
            12 => Some(MlxDtype::Bfloat16),
            13 => Some(MlxDtype::Complex64),
            _ => None,
        }
    }
}

impl std::fmt::Display for MlxDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// FFI module — raw bindings to mlx-c
// ---------------------------------------------------------------------------

/// Raw FFI bindings to mlx-c.
///
/// When mlx-c is available (no `mlx_stub` cfg), real bindings are used via
/// bindgen-generated code.  Otherwise minimal stub types are declared so the
/// safe wrappers compile without the native library.
#[cfg(not(mlx_stub))]
pub(crate) mod ffi {
    #![allow(
        non_upper_case_globals,
        non_camel_case_types,
        non_snake_case,
        dead_code
    )]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[cfg(mlx_stub)]
pub(crate) mod ffi {
    #![allow(
        non_upper_case_globals,
        non_camel_case_types,
        non_snake_case,
        dead_code
    )]
    use std::os::raw::{c_uint, c_void};

    pub type mlx_dtype = c_uint;
    pub type mlx_device_type = c_uint;

    pub const mlx_device_type__MLX_CPU: mlx_device_type = 0;
    pub const mlx_device_type__MLX_GPU: mlx_device_type = 1;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct mlx_array_ {
        pub ctx: *mut c_void,
    }
    pub type mlx_array = mlx_array_;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct mlx_device_ {
        pub ctx: *mut c_void,
    }
    pub type mlx_device = mlx_device_;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct mlx_stream_ {
        pub ctx: *mut c_void,
    }
    pub type mlx_stream = mlx_stream_;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct mlx_vector_array_ {
        pub ctx: *mut c_void,
    }
    pub type mlx_vector_array = mlx_vector_array_;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct mlx_vector_string_ {
        pub ctx: *mut c_void,
    }
    pub type mlx_vector_string = mlx_vector_string_;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct mlx_optional_float_ {
        pub value: f32,
        pub has_value: bool,
    }
    pub type mlx_optional_float = mlx_optional_float_;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct mlx_fast_metal_kernel_config_ {
        pub ctx: *mut c_void,
    }
    pub type mlx_fast_metal_kernel_config = mlx_fast_metal_kernel_config_;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct mlx_fast_metal_kernel_ {
        pub ctx: *mut c_void,
    }
    pub type mlx_fast_metal_kernel = mlx_fast_metal_kernel_;
}
