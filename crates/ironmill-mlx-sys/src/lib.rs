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

pub mod array;
pub mod device;
pub mod error;
pub mod fast_ops;
pub mod metal_kernel;
pub mod ops;
pub mod stream;

pub use array::MlxArray;
pub use device::MlxDevice;
pub use error::MlxSysError;
pub use fast_ops::{rms_norm, rope, scaled_dot_product_attention};
pub use metal_kernel::metal_kernel;
pub use ops::{add, expand_dims, matmul, multiply, reshape, silu, slice, transpose};
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
    /// Brain floating point (16-bit).
    Bfloat16 = 11,
    /// Complex 64-bit (two 32-bit floats).
    Complex64 = 12,
}

impl MlxDtype {
    /// Returns the size of a single element of this dtype in bytes.
    pub fn size(self) -> usize {
        match self {
            MlxDtype::Bool | MlxDtype::Uint8 | MlxDtype::Int8 => 1,
            MlxDtype::Uint16 | MlxDtype::Int16 | MlxDtype::Float16 | MlxDtype::Bfloat16 => 2,
            MlxDtype::Uint32 | MlxDtype::Int32 | MlxDtype::Float32 => 4,
            MlxDtype::Uint64 | MlxDtype::Int64 | MlxDtype::Complex64 => 8,
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
            11 => Some(MlxDtype::Bfloat16),
            12 => Some(MlxDtype::Complex64),
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
/// When mlx-c is available (no `mlx_stub` cfg), real bindings are used.
/// Otherwise opaque pointer types and stub externs are declared so the crate
/// compiles without the native library.
pub(crate) mod ffi {
    #![allow(non_camel_case_types)]

    use std::ffi::c_void;

    // Opaque handle types — always available regardless of stub mode.
    pub type mlx_array = *mut c_void;
    pub type mlx_stream = *mut c_void;
    pub type mlx_device = *mut c_void;
    #[allow(dead_code)]
    pub type mlx_string = *mut c_void;
    pub type mlx_vector_array = *mut c_void;

    // When real bindings are generated, include them.
    #[cfg(not(mlx_stub))]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    // When in stub mode, declare the extern signatures so the safe wrappers
    // compile.  The linker is never invoked in stub mode because every call
    // site returns an error before reaching FFI.
    #[cfg(mlx_stub)]
    #[allow(dead_code)]
    unsafe extern "C" {
        // Lifecycle
        pub fn mlx_retain(obj: *mut c_void);
        pub fn mlx_free(obj: *mut c_void);

        // Array construction
        pub fn mlx_array_from_data(
            data: *const c_void,
            shape: *const i32,
            ndim: i32,
            dtype: u32,
        ) -> mlx_array;
        pub fn mlx_array_from_float(val: f32) -> mlx_array;

        // Array queries
        pub fn mlx_array_dtype(arr: mlx_array) -> u32;
        pub fn mlx_array_ndim(arr: mlx_array) -> i32;
        pub fn mlx_array_size(arr: mlx_array) -> i32;
        pub fn mlx_array_itemsize(arr: mlx_array) -> usize;
        pub fn mlx_array_shape(arr: mlx_array) -> *const i32;
        pub fn mlx_array_data_ptr(arr: mlx_array) -> *const c_void;

        // Device
        pub fn mlx_default_device() -> mlx_device;
        pub fn mlx_device_new(device_type: i32) -> mlx_device;

        // Stream
        pub fn mlx_stream_new(device: mlx_device) -> mlx_stream;
        pub fn mlx_default_gpu_stream() -> mlx_stream;

        // Eval
        pub fn mlx_eval(arr: mlx_array);
        pub fn mlx_async_eval(arr: mlx_array);

        // Ops
        pub fn mlx_matmul(a: mlx_array, b: mlx_array, stream: mlx_stream) -> mlx_array;
        pub fn mlx_add(a: mlx_array, b: mlx_array, stream: mlx_stream) -> mlx_array;
        pub fn mlx_multiply(a: mlx_array, b: mlx_array, stream: mlx_stream) -> mlx_array;
        pub fn mlx_reshape(
            a: mlx_array,
            shape: *const i32,
            ndim: i32,
            stream: mlx_stream,
        ) -> mlx_array;
        pub fn mlx_transpose_all(a: mlx_array, stream: mlx_stream) -> mlx_array;
        pub fn mlx_sigmoid(a: mlx_array, stream: mlx_stream) -> mlx_array;
        pub fn mlx_slice(
            a: mlx_array,
            start: *const i32,
            stop: *const i32,
            strides: *const i32,
            ndim: i32,
            stream: mlx_stream,
        ) -> mlx_array;
        pub fn mlx_expand_dims(
            a: mlx_array,
            axes: *const i32,
            naxes: i32,
            stream: mlx_stream,
        ) -> mlx_array;

        // Fast ops
        pub fn mlx_fast_rms_norm(
            x: mlx_array,
            weight: mlx_array,
            eps: f32,
            stream: mlx_stream,
        ) -> mlx_array;
        pub fn mlx_fast_rope(
            x: mlx_array,
            dims: i32,
            traditional: bool,
            base: f32,
            scale: f32,
            offset: i32,
            stream: mlx_stream,
        ) -> mlx_array;
        pub fn mlx_fast_scaled_dot_product_attention(
            q: mlx_array,
            k: mlx_array,
            v: mlx_array,
            scale: f32,
            mask: mlx_array,
            stream: mlx_stream,
        ) -> mlx_array;

        // Metal kernel
        pub fn mlx_fast_metal_kernel(
            name: *const std::ffi::c_char,
            inputs: mlx_vector_array,
            outputs: mlx_vector_array,
            source: *const std::ffi::c_char,
            grid: *const usize,
            threadgroup: *const usize,
            output_shapes: *const *const i32,
            output_shape_ndims: *const i32,
            output_dtypes: *const u32,
            num_outputs: i32,
            stream: mlx_stream,
        ) -> mlx_vector_array;

        // Vector array helpers
        pub fn mlx_vector_array_new() -> mlx_vector_array;
        pub fn mlx_vector_array_add(vec: mlx_vector_array, arr: mlx_array);
        pub fn mlx_vector_array_get(vec: mlx_vector_array, index: i32) -> mlx_array;
        pub fn mlx_vector_array_size(vec: mlx_vector_array) -> i32;
    }
}
