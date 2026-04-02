//! Safe wrapper for MLX devices.

use crate::error::MlxSysError;
use crate::ffi;

// ---------------------------------------------------------------------------
// MlxDevice
// ---------------------------------------------------------------------------

/// Safe wrapper around an mlx-c device handle.
#[cfg_attr(mlx_stub, allow(dead_code))]
pub struct MlxDevice {
    pub(crate) raw: ffi::mlx_device,
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
            let raw = unsafe { ffi::mlx_device_new_type(ffi::mlx_device_type__MLX_GPU, 0) };
            if raw.ctx.is_null() {
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
            let raw = unsafe { ffi::mlx_device_new_type(ffi::mlx_device_type__MLX_CPU, 0) };
            if raw.ctx.is_null() {
                return Err(MlxSysError::MlxC("failed to create CPU device".into()));
            }
            Ok(Self { raw })
        }
    }
}

impl Drop for MlxDevice {
    fn drop(&mut self) {
        #[cfg(not(mlx_stub))]
        if !self.raw.ctx.is_null() {
            unsafe { ffi::mlx_device_free(self.raw) };
            self.raw.ctx = std::ptr::null_mut();
        }
    }
}
