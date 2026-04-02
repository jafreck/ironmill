//! Safe wrapper for MLX streams and evaluation.

use crate::device::MlxDevice;
use crate::error::MlxSysError;
use crate::ffi;

// ---------------------------------------------------------------------------
// MlxStream
// ---------------------------------------------------------------------------

/// Safe wrapper around an mlx-c stream handle.
///
/// Streams in MLX represent an ordered sequence of operations on a device.
#[cfg_attr(mlx_stub, allow(dead_code))]
pub struct MlxStream {
    pub(crate) raw: ffi::mlx_stream,
}

// SAFETY: MLX streams are thread-safe reference-counted handles.
unsafe impl Send for MlxStream {}
unsafe impl Sync for MlxStream {}

impl MlxStream {
    /// Create a new stream on the given device.
    pub fn new(device: &MlxDevice) -> Result<Self, MlxSysError> {
        #[cfg(mlx_stub)]
        {
            let _ = device;
            Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
        }

        #[cfg(not(mlx_stub))]
        {
            // SAFETY: device.raw is a valid mlx_device handle. The returned
            // stream handle is null-checked.
            let raw = unsafe { ffi::mlx_stream_new_device(device.raw) };
            if raw.ctx.is_null() {
                return Err(MlxSysError::MlxC("failed to create stream".into()));
            }
            Ok(Self { raw })
        }
    }

    /// Returns the default GPU stream.
    pub fn default_gpu() -> Result<Self, MlxSysError> {
        #[cfg(mlx_stub)]
        {
            Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
        }

        #[cfg(not(mlx_stub))]
        {
            // SAFETY: mlx_default_gpu_stream_new is a C function that returns
            // the default GPU stream handle. The returned handle is null-checked.
            let raw = unsafe { ffi::mlx_default_gpu_stream_new() };
            if raw.ctx.is_null() {
                return Err(MlxSysError::MlxC("failed to get default GPU stream".into()));
            }
            Ok(Self { raw })
        }
    }
}

impl Drop for MlxStream {
    fn drop(&mut self) {
        #[cfg(not(mlx_stub))]
        if !self.raw.ctx.is_null() {
            // SAFETY: self.raw is a valid mlx_stream handle; null-checked above.
            unsafe { ffi::mlx_stream_free(self.raw) };
            self.raw.ctx = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Synchronously evaluate the given arrays.
///
/// This forces computation of all lazy operations needed to produce the
/// values of the provided arrays.
pub fn eval(outputs: &[&crate::array::MlxArray]) -> Result<(), MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = outputs;
        Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
    }

    #[cfg(not(mlx_stub))]
    {
        for arr in outputs {
            // SAFETY: arr.raw is a valid mlx_array handle owned by MlxArray.
            let ret = unsafe { ffi::mlx_array_eval(arr.raw) };
            if ret != 0 {
                return Err(MlxSysError::MlxC("mlx_array_eval failed".into()));
            }
        }
        Ok(())
    }
}

/// Asynchronously evaluate the given arrays.
///
/// Schedules computation without blocking.
pub fn async_eval(outputs: &[&crate::array::MlxArray]) -> Result<(), MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = outputs;
        Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
    }

    #[cfg(not(mlx_stub))]
    {
        // SAFETY: mlx_vector_array_new returns a valid vector handle (null-
        // checked below). Each arr.raw is a valid array handle. The vector
        // is freed after the async eval call.
        let vec = unsafe { ffi::mlx_vector_array_new() };
        if vec.ctx.is_null() {
            return Err(MlxSysError::MlxC(
                "mlx_vector_array_new returned null".into(),
            ));
        }
        for arr in outputs {
            // SAFETY: vec is valid (null-checked above), arr.raw is a valid handle.
            unsafe { ffi::mlx_vector_array_append_value(vec, arr.raw) };
        }
        // SAFETY: vec is a valid vector handle. mlx_async_eval schedules
        // evaluation without blocking.
        let ret = unsafe { ffi::mlx_async_eval(vec) };
        // SAFETY: vec is a valid vector handle; freed after use.
        unsafe { ffi::mlx_vector_array_free(vec) };
        if ret != 0 {
            return Err(MlxSysError::MlxC("mlx_async_eval failed".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Metal memory management
// ---------------------------------------------------------------------------

/// Clear MLX's internal buffer cache.
///
/// This releases pooled Metal buffers back to the system. Useful after
/// `reset()` or after processing long sequences to prevent memory
/// fragmentation. Safe to call at any time — MLX will re-allocate
/// buffers as needed.
pub fn metal_clear_cache() -> Result<(), MlxSysError> {
    #[cfg(mlx_stub)]
    {
        Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
    }

    #[cfg(not(mlx_stub))]
    {
        // SAFETY: mlx_clear_cache is a C function that releases pooled Metal
        // buffers. Safe to call at any time per mlx-c documentation.
        let ret = unsafe { ffi::mlx_clear_cache() };
        if ret != 0 {
            return Err(MlxSysError::MlxC("mlx_clear_cache failed".into()));
        }
        Ok(())
    }
}
