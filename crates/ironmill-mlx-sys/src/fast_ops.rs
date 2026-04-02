//! Safe wrappers for `mlx_fast_*` optimized operations.

use crate::array::MlxArray;
use crate::error::MlxSysError;
use crate::stream::MlxStream;

#[cfg(not(mlx_stub))]
use crate::ffi;

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
        // SAFETY: x.raw and weight.raw are valid mlx_array handles, stream.raw
        // is a valid mlx_stream. mlx_array_new returns a valid empty handle.
        // On error, result is freed. On success, ownership transfers to MlxArray.
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret =
            unsafe { ffi::mlx_fast_rms_norm(&mut result, x.raw, weight.raw, eps, stream.raw) };
        if ret != 0 {
            // SAFETY: result is a valid handle from mlx_array_new.
            unsafe { ffi::mlx_array_free(result) };
            return Err(MlxSysError::MlxC("rms_norm failed".into()));
        }
        Ok(unsafe { MlxArray::from_raw(result) })
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
        let opt_base = ffi::mlx_optional_float {
            value: base,
            has_value: true,
        };
        // No custom frequencies — pass an empty/null array.
        let null_freqs = ffi::mlx_array {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: x.raw and stream.raw are valid handles. null_freqs is a
        // sentinel null-context array signaling "no custom frequencies" to
        // the C API. On error, result is freed.
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe {
            ffi::mlx_fast_rope(
                &mut result,
                x.raw,
                dims,
                traditional,
                opt_base,
                scale,
                offset,
                null_freqs,
                stream.raw,
            )
        };
        if ret != 0 {
            // SAFETY: result is a valid handle from mlx_array_new.
            unsafe { ffi::mlx_array_free(result) };
            return Err(MlxSysError::MlxC("rope failed".into()));
        }
        Ok(unsafe { MlxArray::from_raw(result) })
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
        let null_arr = ffi::mlx_array {
            ctx: std::ptr::null_mut(),
        };

        // Determine mask mode and mask array.
        let (mask_mode_cstr, mask_arr) = match mask {
            Some(m) => {
                // "array" tells mlx-c to use the provided mask array.
                (c"array", m.raw)
            }
            None => (c"", null_arr),
        };

        // SAFETY: q/k/v .raw are valid mlx_array handles. mask_mode_cstr is a
        // valid C string. mask_arr is either a valid handle or a null-context
        // sentinel. null_arr is a sentinel for unused sinks. stream.raw is valid.
        // On error, result is freed.
        let mut result = unsafe { ffi::mlx_array_new() };
        let ret = unsafe {
            ffi::mlx_fast_scaled_dot_product_attention(
                &mut result,
                q.raw,
                k.raw,
                v.raw,
                scale,
                mask_mode_cstr.as_ptr(),
                mask_arr,
                null_arr, // sinks
                stream.raw,
            )
        };
        if ret != 0 {
            // SAFETY: result is a valid handle from mlx_array_new.
            unsafe { ffi::mlx_array_free(result) };
            return Err(MlxSysError::MlxC(
                "scaled_dot_product_attention failed".into(),
            ));
        }
        Ok(unsafe { MlxArray::from_raw(result) })
    }
}
