//! Safe wrapper for MLX devices.

use std::ffi::c_void;

use crate::error::MlxSysError;
use crate::ffi;

// ---------------------------------------------------------------------------
// Device type constants
// ---------------------------------------------------------------------------

#[cfg(not(mlx_stub))]
const MLX_DEVICE_CPU: i32 = 0;
#[cfg(not(mlx_stub))]
const MLX_DEVICE_GPU: i32 = 1;

// ---------------------------------------------------------------------------
// MlxDevice
// ---------------------------------------------------------------------------

/// Safe wrapper around an mlx-c device handle.
pub struct MlxDevice {
    raw: *mut c_void,
}

// SAFETY: MLX devices are thread-safe reference-counted handles.
unsafe impl Send for MlxDevice {}
unsafe impl Sync for MlxDevice {}

impl MlxDevice {
    /// Returns a handle to the default GPU device.
    pub fn default_gpu() -> Result<Self, MlxSysError> {
        #[cfg(mlx_stub)]
        {
            Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
        }

        #[cfg(not(mlx_stub))]
        {
            let raw = unsafe { ffi::mlx_device_new(MLX_DEVICE_GPU) };
            if raw.is_null() {
                return Err(MlxSysError::MlxC("failed to create GPU device".into()));
            }
            Ok(Self { raw })
        }
    }

    /// Returns a handle to the default CPU device.
    pub fn default_cpu() -> Result<Self, MlxSysError> {
        #[cfg(mlx_stub)]
        {
            Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
        }

        #[cfg(not(mlx_stub))]
        {
            let raw = unsafe { ffi::mlx_device_new(MLX_DEVICE_CPU) };
            if raw.is_null() {
                return Err(MlxSysError::MlxC("failed to create CPU device".into()));
            }
            Ok(Self { raw })
        }
    }

    /// Returns the raw device pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for MlxDevice {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::mlx_free(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
