//! Safe wrappers for core MLX operations.

use crate::array::MlxArray;
use crate::error::MlxSysError;
use crate::stream::MlxStream;

#[cfg(not(mlx_stub))]
use crate::ffi;

// ---------------------------------------------------------------------------
// Helper — check a returned array pointer
// ---------------------------------------------------------------------------

#[cfg(not(mlx_stub))]
fn check_array(raw: *mut std::ffi::c_void, op: &str) -> Result<MlxArray, MlxSysError> {
    if raw.is_null() {
        return Err(MlxSysError::MlxC(format!("{op} returned null")));
    }
    Ok(unsafe { MlxArray::from_raw(raw) })
}

#[cfg(mlx_stub)]
fn stub_err<T>(_op: &str) -> Result<T, MlxSysError> {
    Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// Matrix multiplication: `a @ b`.
pub fn matmul(a: &MlxArray, b: &MlxArray, stream: &MlxStream) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (a, b, stream);
        stub_err("matmul")
    }

    #[cfg(not(mlx_stub))]
    {
        let raw = unsafe { ffi::mlx_matmul(a.as_raw_ptr(), b.as_raw_ptr(), stream.as_raw_ptr()) };
        check_array(raw, "matmul")
    }
}

/// Element-wise addition: `a + b`.
pub fn add(a: &MlxArray, b: &MlxArray, stream: &MlxStream) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (a, b, stream);
        stub_err("add")
    }

    #[cfg(not(mlx_stub))]
    {
        let raw = unsafe { ffi::mlx_add(a.as_raw_ptr(), b.as_raw_ptr(), stream.as_raw_ptr()) };
        check_array(raw, "add")
    }
}

/// Element-wise multiplication: `a * b`.
pub fn multiply(a: &MlxArray, b: &MlxArray, stream: &MlxStream) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (a, b, stream);
        stub_err("multiply")
    }

    #[cfg(not(mlx_stub))]
    {
        let raw = unsafe { ffi::mlx_multiply(a.as_raw_ptr(), b.as_raw_ptr(), stream.as_raw_ptr()) };
        check_array(raw, "multiply")
    }
}

/// Reshape an array to `new_shape`.
pub fn reshape(
    a: &MlxArray,
    new_shape: &[i32],
    stream: &MlxStream,
) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (a, new_shape, stream);
        stub_err("reshape")
    }

    #[cfg(not(mlx_stub))]
    {
        let raw = unsafe {
            ffi::mlx_reshape(
                a.as_raw_ptr(),
                new_shape.as_ptr(),
                new_shape.len() as i32,
                stream.as_raw_ptr(),
            )
        };
        check_array(raw, "reshape")
    }
}

/// Transpose all axes of an array (reverse axis order).
pub fn transpose(a: &MlxArray, stream: &MlxStream) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (a, stream);
        stub_err("transpose")
    }

    #[cfg(not(mlx_stub))]
    {
        let raw = unsafe { ffi::mlx_transpose_all(a.as_raw_ptr(), stream.as_raw_ptr()) };
        check_array(raw, "transpose")
    }
}

/// Transpose an array with a specific axis permutation (lazy).
pub fn transpose_axes(
    a: &MlxArray,
    axes: &[i32],
    stream: &MlxStream,
) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (a, axes, stream);
        stub_err("transpose_axes")
    }

    #[cfg(not(mlx_stub))]
    {
        let raw = unsafe {
            ffi::mlx_transpose(
                a.as_raw_ptr(),
                axes.as_ptr(),
                axes.len() as i32,
                stream.as_raw_ptr(),
            )
        };
        check_array(raw, "transpose_axes")
    }
}

/// SiLU (Sigmoid Linear Unit) activation: `x * sigmoid(x)`.
///
/// Implemented as `x * sigmoid(x)` using the available `mlx_sigmoid`.
pub fn silu(a: &MlxArray, stream: &MlxStream) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (a, stream);
        stub_err("silu")
    }

    #[cfg(not(mlx_stub))]
    {
        // silu(x) = x * sigmoid(x)
        let sig = unsafe { ffi::mlx_sigmoid(a.as_raw_ptr(), stream.as_raw_ptr()) };
        if sig.is_null() {
            return Err(MlxSysError::MlxC("sigmoid returned null".into()));
        }
        let raw = unsafe { ffi::mlx_multiply(a.as_raw_ptr(), sig, stream.as_raw_ptr()) };
        // Free the intermediate sigmoid result.
        unsafe { ffi::mlx_free(sig) };
        check_array(raw, "silu")
    }
}

/// Slice an array along each axis with `start`, `stop`, `strides`.
pub fn slice(
    a: &MlxArray,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    stream: &MlxStream,
) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (a, start, stop, strides, stream);
        stub_err("slice")
    }

    #[cfg(not(mlx_stub))]
    {
        let ndim = start.len() as i32;
        let raw = unsafe {
            ffi::mlx_slice(
                a.as_raw_ptr(),
                start.as_ptr(),
                stop.as_ptr(),
                strides.as_ptr(),
                ndim,
                stream.as_raw_ptr(),
            )
        };
        check_array(raw, "slice")
    }
}

/// Insert singleton dimensions at the specified axes.
pub fn expand_dims(
    a: &MlxArray,
    axes: &[i32],
    stream: &MlxStream,
) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (a, axes, stream);
        stub_err("expand_dims")
    }

    #[cfg(not(mlx_stub))]
    {
        let raw = unsafe {
            ffi::mlx_expand_dims(
                a.as_raw_ptr(),
                axes.as_ptr(),
                axes.len() as i32,
                stream.as_raw_ptr(),
            )
        };
        check_array(raw, "expand_dims")
    }
}

/// Concatenate arrays along an axis (lazy).
pub fn concat(
    arrays: &[&MlxArray],
    axis: i32,
    stream: &MlxStream,
) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (arrays, axis, stream);
        stub_err("concat")
    }

    #[cfg(not(mlx_stub))]
    {
        let vec = unsafe { ffi::mlx_vector_array_new() };
        if vec.is_null() {
            return Err(MlxSysError::MlxC("failed to create vector_array".into()));
        }
        for arr in arrays {
            unsafe { ffi::mlx_vector_array_add(vec, arr.as_raw_ptr()) };
        }
        let raw = unsafe { ffi::mlx_concatenate(vec, axis, stream.as_raw_ptr()) };
        unsafe { ffi::mlx_free(vec) };
        check_array(raw, "concat")
    }
}

/// Broadcast an array to a target shape (lazy).
pub fn broadcast_to(
    a: &MlxArray,
    shape: &[i32],
    stream: &MlxStream,
) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (a, shape, stream);
        stub_err("broadcast_to")
    }

    #[cfg(not(mlx_stub))]
    {
        let raw = unsafe {
            ffi::mlx_broadcast_to(
                a.as_raw_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
                stream.as_raw_ptr(),
            )
        };
        check_array(raw, "broadcast_to")
    }
}
