//! Safe wrappers for `mlx_fast_*` optimized operations.

use crate::array::MlxArray;
use crate::error::MlxSysError;
use crate::stream::MlxStream;

#[cfg(not(mlx_stub))]
use crate::ffi;
#[cfg(not(mlx_stub))]
use std::ffi::c_void;

// ---------------------------------------------------------------------------
// Fast operations
// ---------------------------------------------------------------------------

/// Fused RMS normalization.
pub fn rms_norm(
    x: &MlxArray,
    weight: &MlxArray,
    eps: f32,
    stream: &MlxStream,
) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (x, weight, eps, stream);
        Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
    }

    #[cfg(not(mlx_stub))]
    {
        let raw = unsafe {
            ffi::mlx_fast_rms_norm(
                x.as_raw_ptr(),
                weight.as_raw_ptr(),
                eps,
                stream.as_raw_ptr(),
            )
        };
        if raw.is_null() {
            return Err(MlxSysError::MlxC("rms_norm returned null".into()));
        }
        Ok(unsafe { MlxArray::from_raw(raw) })
    }
}

/// Rotary positional embedding (RoPE).
pub fn rope(
    x: &MlxArray,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: i32,
    stream: &MlxStream,
) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (x, dims, traditional, base, scale, offset, stream);
        Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
    }

    #[cfg(not(mlx_stub))]
    {
        let raw = unsafe {
            ffi::mlx_fast_rope(
                x.as_raw_ptr(),
                dims,
                traditional,
                base,
                scale,
                offset,
                stream.as_raw_ptr(),
            )
        };
        if raw.is_null() {
            return Err(MlxSysError::MlxC("rope returned null".into()));
        }
        Ok(unsafe { MlxArray::from_raw(raw) })
    }
}

/// Fused scaled dot-product attention.
pub fn scaled_dot_product_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: Option<&MlxArray>,
    stream: &MlxStream,
) -> Result<MlxArray, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (q, k, v, scale, mask, stream);
        Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
    }

    #[cfg(not(mlx_stub))]
    {
        let mask_ptr: *mut c_void = match mask {
            Some(m) => m.as_raw_ptr(),
            None => std::ptr::null_mut(),
        };
        let raw = unsafe {
            ffi::mlx_fast_scaled_dot_product_attention(
                q.as_raw_ptr(),
                k.as_raw_ptr(),
                v.as_raw_ptr(),
                scale,
                mask_ptr,
                stream.as_raw_ptr(),
            )
        };
        if raw.is_null() {
            return Err(MlxSysError::MlxC(
                "scaled_dot_product_attention returned null".into(),
            ));
        }
        Ok(unsafe { MlxArray::from_raw(raw) })
    }
}
