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
        let mut sig = unsafe { ffi::mlx_array_new() };
        let ret = unsafe { ffi::mlx_sigmoid(&mut sig, a.raw, stream.raw) };
        if ret != 0 {
            unsafe { ffi::mlx_array_free(sig) };
            return Err(MlxSysError::MlxC("sigmoid failed".into()));
        }
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe { ffi::mlx_multiply(&mut result, a.raw, sig, stream.raw) };
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
        let vec = unsafe { ffi::mlx_vector_array_new() };
        if vec.ctx.is_null() {
            return Err(MlxSysError::MlxC(
                "mlx_vector_array_new returned null".into(),
            ));
        }
        for arr in arrays {
            unsafe { ffi::mlx_vector_array_append_value(vec, arr.raw) };
        }
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe { ffi::mlx_concatenate_axis(&mut result, vec, axis, stream.raw) };
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
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe {
            ffi::mlx_broadcast_to(&mut result, a.raw, shape.as_ptr(), shape.len(), stream.raw)
        };
        check_ret(ret, "broadcast_to")?;
        Ok(unsafe { MlxArray::from_raw(result) })
    }
}
