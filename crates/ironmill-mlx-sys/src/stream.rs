//! Safe wrapper for MLX streams and evaluation.

use std::ffi::c_void;

use crate::device::MlxDevice;
use crate::error::MlxSysError;
use crate::ffi;

// ---------------------------------------------------------------------------
// MlxStream
// ---------------------------------------------------------------------------

/// Safe wrapper around an mlx-c stream handle.
///
/// Streams in MLX represent an ordered sequence of operations on a device.
pub struct MlxStream {
    raw: *mut c_void,
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
            let raw = unsafe { ffi::mlx_stream_new(device.as_raw_ptr()) };
            if raw.is_null() {
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
            let raw = unsafe { ffi::mlx_default_gpu_stream() };
            if raw.is_null() {
                return Err(MlxSysError::MlxC("failed to get default GPU stream".into()));
            }
            Ok(Self { raw })
        }
    }

    /// Returns the raw stream pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for MlxStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::mlx_free(self.raw) };
            self.raw = std::ptr::null_mut();
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
            unsafe { ffi::mlx_eval(arr.as_raw_ptr()) };
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
        for arr in outputs {
            unsafe { ffi::mlx_async_eval(arr.as_raw_ptr()) };
        }
        Ok(())
    }
}
