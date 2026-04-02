//! Safe wrapper for MLX arrays.

use std::ffi::c_void;

use crate::MlxDtype;
use crate::error::MlxSysError;
use crate::ffi;
use crate::stream::MlxStream;

// ---------------------------------------------------------------------------
// MlxArray
// ---------------------------------------------------------------------------

/// Safe wrapper around an mlx-c `mlx_array` handle.
///
/// Arrays in MLX are lazily evaluated — data is not computed until
/// [`crate::stream::eval`] or [`crate::stream::async_eval`] is called.
pub struct MlxArray {
    raw: *mut c_void,
}

// SAFETY: MLX arrays are reference-counted and thread-safe.
unsafe impl Send for MlxArray {}
unsafe impl Sync for MlxArray {}

impl MlxArray {
    /// Create an `MlxArray` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid, retained `mlx_array` handle.
    #[cfg_attr(mlx_stub, allow(dead_code))]
    pub(crate) unsafe fn from_raw(raw: *mut c_void) -> Self {
        Self { raw }
    }

    /// Create an array by copying data from a byte slice.
    pub fn from_data_copy(
        data: &[u8],
        shape: &[usize],
        dtype: MlxDtype,
        _stream: &MlxStream,
    ) -> Result<Self, MlxSysError> {
        #[cfg(mlx_stub)]
        {
            let _ = (data, shape, dtype, _stream);
            Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
        }

        #[cfg(not(mlx_stub))]
        {
            let shape_i32: Vec<i32> = shape
                .iter()
                .map(|&s| i32::try_from(s).unwrap_or(i32::MAX))
                .collect();
            let raw = unsafe {
                ffi::mlx_array_from_data(
                    data.as_ptr() as *const c_void,
                    shape_i32.as_ptr(),
                    shape_i32.len() as i32,
                    dtype as u32,
                )
            };
            if raw.is_null() {
                return Err(MlxSysError::MlxC("failed to create array from data".into()));
            }
            Ok(Self { raw })
        }
    }

    /// Create a scalar array from a single `f32` value.
    pub fn from_scalar(
        val: f32,
        _dtype: MlxDtype,
        _stream: &MlxStream,
    ) -> Result<Self, MlxSysError> {
        #[cfg(mlx_stub)]
        {
            let _ = (val, _dtype, _stream);
            Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
        }

        #[cfg(not(mlx_stub))]
        {
            let raw = unsafe { ffi::mlx_array_from_float(val) };
            if raw.is_null() {
                return Err(MlxSysError::MlxC("failed to create scalar array".into()));
            }
            Ok(Self { raw })
        }
    }

    /// Returns the shape of this array.
    pub fn shape(&self) -> Vec<usize> {
        #[cfg(mlx_stub)]
        {
            Vec::new()
        }

        #[cfg(not(mlx_stub))]
        {
            let ndim = unsafe { ffi::mlx_array_ndim(self.raw) } as usize;
            let shape_ptr = unsafe { ffi::mlx_array_shape(self.raw) };
            if shape_ptr.is_null() || ndim == 0 {
                return Vec::new();
            }
            let shape_slice = unsafe { std::slice::from_raw_parts(shape_ptr, ndim) };
            shape_slice.iter().map(|&d| d as usize).collect()
        }
    }

    /// Returns the dtype of this array.
    pub fn dtype(&self) -> MlxDtype {
        #[cfg(mlx_stub)]
        {
            MlxDtype::Float32
        }

        #[cfg(not(mlx_stub))]
        {
            let raw_dtype = unsafe { ffi::mlx_array_dtype(self.raw) };
            MlxDtype::from_raw(raw_dtype).unwrap_or(MlxDtype::Float32)
        }
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        #[cfg(mlx_stub)]
        {
            0
        }

        #[cfg(not(mlx_stub))]
        {
            unsafe { ffi::mlx_array_ndim(self.raw) as usize }
        }
    }

    /// Returns the total number of elements.
    pub fn size(&self) -> usize {
        #[cfg(mlx_stub)]
        {
            0
        }

        #[cfg(not(mlx_stub))]
        {
            unsafe { ffi::mlx_array_size(self.raw) as usize }
        }
    }

    /// Returns the size in bytes of a single element.
    pub fn item_size(&self) -> usize {
        #[cfg(mlx_stub)]
        {
            0
        }

        #[cfg(not(mlx_stub))]
        {
            unsafe { ffi::mlx_array_itemsize(self.raw) }
        }
    }

    /// Returns a contiguous slice view of the evaluated array data.
    ///
    /// The array must have been evaluated (via [`crate::stream::eval`])
    /// before calling this method.
    ///
    /// # Safety
    ///
    /// The caller must ensure `T` matches the array's dtype and that the
    /// array has been evaluated.
    pub fn as_contiguous_slice<T>(&self) -> Result<&[T], MlxSysError> {
        #[cfg(mlx_stub)]
        {
            Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
        }

        #[cfg(not(mlx_stub))]
        {
            let ptr = unsafe { ffi::mlx_array_data_ptr(self.raw) };
            if ptr.is_null() {
                return Err(MlxSysError::MlxC(
                    "array data pointer is null (not yet evaluated?)".into(),
                ));
            }
            let len = self.size();
            // SAFETY: Caller guarantees T matches dtype and array is evaluated.
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const T, len) };
            Ok(slice)
        }
    }

    /// Returns the raw `mlx_array` pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Clone for MlxArray {
    fn clone(&self) -> Self {
        if !self.raw.is_null() {
            unsafe { ffi::mlx_retain(self.raw) };
        }
        Self { raw: self.raw }
    }
}

impl Drop for MlxArray {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::mlx_free(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
