//! Safe wrappers for core MLX operations.

use crate::array::MlxArray;
use crate::error::MlxSysError;
use crate::stream::MlxStream;

#[cfg(not(mlx_stub))]
use crate::ffi;

// ---------------------------------------------------------------------------
// Helper — run an op that writes into an output mlx_array
// ---------------------------------------------------------------------------

#[cfg(not(mlx_stub))]
fn check_ret(ret: i32, op: &str) -> Result<(), MlxSysError> {
    if ret != 0 {
        return Err(MlxSysError::MlxC(format!("{op} failed (error code {ret})")));
    }
    Ok(())
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
        // SAFETY: mlx_array_new allocates a fresh empty handle (no invalid
        // pointer access). The subsequent FFI calls receive valid mlx_array
        // handles (a.raw, b.raw) owned by their MlxArray wrappers, and
        // stream.raw is a valid mlx_stream. On success, `result` is a valid
        // owned handle transferred to MlxArray::from_raw.
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe { ffi::mlx_matmul(&mut result, a.raw, b.raw, stream.raw) };
        check_ret(ret, "matmul")?;
        Ok(unsafe { MlxArray::from_raw(result) })
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
        // SAFETY: All raw handles (a.raw, b.raw, stream.raw) are valid,
        // owned by their safe wrappers. mlx_array_new returns a valid empty
        // handle. On success, result ownership transfers to MlxArray.
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe { ffi::mlx_add(&mut result, a.raw, b.raw, stream.raw) };
        check_ret(ret, "add")?;
        Ok(unsafe { MlxArray::from_raw(result) })
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
        // SAFETY: All raw handles (a.raw, b.raw, stream.raw) are valid,
        // owned by their safe wrappers. Result ownership transfers on success.
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe { ffi::mlx_multiply(&mut result, a.raw, b.raw, stream.raw) };
        check_ret(ret, "multiply")?;
        Ok(unsafe { MlxArray::from_raw(result) })
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
        // SAFETY: a.raw and stream.raw are valid handles. new_shape.as_ptr()
        // and new_shape.len() describe a valid i32 slice. Result ownership
        // transfers on success.
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe {
            ffi::mlx_reshape(
                &mut result,
                a.raw,
                new_shape.as_ptr(),
                new_shape.len(),
                stream.raw,
            )
        };
        check_ret(ret, "reshape")?;
        Ok(unsafe { MlxArray::from_raw(result) })
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
        // SAFETY: a.raw and stream.raw are valid handles owned by their
        // safe wrappers. Result ownership transfers on success.
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe { ffi::mlx_transpose(&mut result, a.raw, stream.raw) };
        check_ret(ret, "transpose")?;
        Ok(unsafe { MlxArray::from_raw(result) })
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
        // SAFETY: a.raw and stream.raw are valid handles. axes.as_ptr() and
        // axes.len() describe a valid i32 slice. Result ownership transfers
        // on success.
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe {
            ffi::mlx_transpose_axes(&mut result, a.raw, axes.as_ptr(), axes.len(), stream.raw)
        };
        check_ret(ret, "transpose_axes")?;
        Ok(unsafe { MlxArray::from_raw(result) })
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
        // SAFETY: a.raw and stream.raw are valid handles. Intermediate `sig`
        // handle is freed on error or after use. Result ownership transfers
        // on success via MlxArray::from_raw.
        let mut sig = unsafe { ffi::mlx_array_new() };
        let ret = unsafe { ffi::mlx_sigmoid(&mut sig, a.raw, stream.raw) };
        if ret != 0 {
            // SAFETY: sig is a valid handle from mlx_array_new.
            unsafe { ffi::mlx_array_free(sig) };
            return Err(MlxSysError::MlxC("sigmoid failed".into()));
        }
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe { ffi::mlx_multiply(&mut result, a.raw, sig, stream.raw) };
        // SAFETY: sig is a valid handle; freed after use regardless of success.
        unsafe { ffi::mlx_array_free(sig) };
        check_ret(ret, "silu")?;
        Ok(unsafe { MlxArray::from_raw(result) })
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
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe {
            ffi::mlx_slice(
                &mut result,
                a.raw,
                start.as_ptr(),
                start.len(),
                stop.as_ptr(),
                stop.len(),
                strides.as_ptr(),
                strides.len(),
                stream.raw,
            )
        };
        check_ret(ret, "slice")?;
        Ok(unsafe { MlxArray::from_raw(result) })
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
        // SAFETY: a.raw and stream.raw are valid handles. axes.as_ptr() and
        // axes.len() describe a valid i32 slice. Result ownership transfers
        // on success.
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe {
            ffi::mlx_expand_dims_axes(&mut result, a.raw, axes.as_ptr(), axes.len(), stream.raw)
        };
        check_ret(ret, "expand_dims")?;
        Ok(unsafe { MlxArray::from_raw(result) })
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
        // SAFETY: mlx_vector_array_new returns a valid vector handle (null-
        // checked below). Each arr.raw is a valid array handle. The vector
        // is freed after the concatenation call. Result ownership transfers
        // on success.
        let vec = unsafe { ffi::mlx_vector_array_new() };
        if vec.ctx.is_null() {
            return Err(MlxSysError::MlxC(
                "mlx_vector_array_new returned null".into(),
            ));
        }
        for arr in arrays {
            // SAFETY: vec is valid (null-checked above), arr.raw is a valid handle.
            unsafe { ffi::mlx_vector_array_append_value(vec, arr.raw) };
        }
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe { ffi::mlx_concatenate_axis(&mut result, vec, axis, stream.raw) };
        // SAFETY: vec is a valid vector handle; freed after use.
        unsafe { ffi::mlx_vector_array_free(vec) };
        check_ret(ret, "concat")?;
        Ok(unsafe { MlxArray::from_raw(result) })
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
        // SAFETY: a.raw and stream.raw are valid handles. shape.as_ptr() and
        // shape.len() describe a valid i32 slice. Result ownership transfers
        // on success.
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe {
            ffi::mlx_broadcast_to(&mut result, a.raw, shape.as_ptr(), shape.len(), stream.raw)
        };
        check_ret(ret, "broadcast_to")?;
        Ok(unsafe { MlxArray::from_raw(result) })
    }
}
