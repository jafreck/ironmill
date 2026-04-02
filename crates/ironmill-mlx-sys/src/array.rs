//! Safe wrapper for MLX arrays.

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
    pub(crate) raw: ffi::mlx_array,
}

// SAFETY: MLX arrays are reference-counted and thread-safe.
unsafe impl Send for MlxArray {}
unsafe impl Sync for MlxArray {}

impl MlxArray {
    /// Create an `MlxArray` from an `ffi::mlx_array` value.
    ///
    /// # Safety
    ///
    /// The value must be a valid, owned `mlx_array` handle (the caller
    /// transfers ownership).
    #[cfg_attr(mlx_stub, allow(dead_code))]
    pub(crate) unsafe fn from_raw(raw: ffi::mlx_array) -> Self {
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
                .map(|&s| {
                    i32::try_from(s).map_err(|_| {
                        MlxSysError::Build(format!("dimension {s} exceeds i32::MAX ({})", i32::MAX))
                    })
                })
                .collect::<Result<Vec<i32>, MlxSysError>>()?;
            let raw = unsafe {
                ffi::mlx_array_new_data(
                    data.as_ptr() as *const std::ffi::c_void,
                    shape_i32.as_ptr(),
                    shape_i32.len() as i32,
                    dtype as u32,
                )
            };
            if raw.ctx.is_null() {
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
            // SAFETY: mlx_array_new_float is a simple C constructor that
            // creates a scalar array. The returned handle is null-checked.
            let raw = unsafe { ffi::mlx_array_new_float(val) };
            if raw.ctx.is_null() {
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
            // SAFETY: self.raw is a valid mlx_array handle (owned by this
            // MlxArray). mlx_array_ndim/shape return values that remain valid
            // for the lifetime of the handle. The slice is constructed from
            // the shape pointer with ndim elements.
            let ndim = unsafe { ffi::mlx_array_ndim(self.raw) };
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
            // SAFETY: self.raw is a valid mlx_array handle.
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
            // SAFETY: self.raw is a valid mlx_array handle.
            unsafe { ffi::mlx_array_ndim(self.raw) }
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
            // SAFETY: self.raw is a valid mlx_array handle.
            unsafe { ffi::mlx_array_size(self.raw) }
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
            // SAFETY: self.raw is a valid mlx_array handle.
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
    /// The caller must ensure that `T` matches the array's [`MlxDtype`] (e.g.
    /// `f32` for [`MlxDtype::Float32`]) and that the array has been evaluated.
    pub unsafe fn as_contiguous_slice<T>(&self) -> Result<&[T], MlxSysError> {
        #[cfg(mlx_stub)]
        {
            Err(MlxSysError::MlxC("mlx-c not available (stub mode)".into()))
        }

        #[cfg(not(mlx_stub))]
        {
            // Use mlx_array_data_uint8 as a generic byte pointer; caller
            // guarantees T matches the real dtype.
            // SAFETY: self.raw is a valid, evaluated mlx_array handle. The
            // returned pointer is valid for the array's data lifetime.
            let ptr = unsafe { ffi::mlx_array_data_uint8(self.raw) };
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

    /// Returns the raw `ffi::mlx_array` handle (Copy).
    pub fn as_mlx_array(&self) -> ffi::mlx_array {
        self.raw
    }
}

impl Clone for MlxArray {
    fn clone(&self) -> Self {
        #[cfg(not(mlx_stub))]
        {
            // SAFETY: mlx_array_new returns a valid empty handle.
            // mlx_array_set copies the reference count (not data), so both
            // new_arr and self.raw remain valid.
            let mut new_arr = unsafe { ffi::mlx_array_new() };
            unsafe { ffi::mlx_array_set(&mut new_arr, self.raw) };
            Self { raw: new_arr }
        }

        #[cfg(mlx_stub)]
        {
            Self { raw: self.raw }
        }
    }
}

impl Drop for MlxArray {
    fn drop(&mut self) {
        #[cfg(not(mlx_stub))]
        if !self.raw.ctx.is_null() {
            // SAFETY: self.raw is a valid mlx_array handle; null-checked above.
            unsafe { ffi::mlx_array_free(self.raw) };
            self.raw.ctx = std::ptr::null_mut();
        }
    }
}
