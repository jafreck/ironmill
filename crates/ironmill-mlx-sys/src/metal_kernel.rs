//! Safe wrapper for `mlx_fast_metal_kernel()`.

use crate::MlxDtype;
use crate::array::MlxArray;
use crate::error::MlxSysError;
use crate::stream::MlxStream;

#[cfg(not(mlx_stub))]
use crate::ffi;
#[cfg(not(mlx_stub))]
use std::ffi::{CString, c_void};

/// Launch a custom Metal kernel via mlx-c.
///
/// This wraps `mlx_fast_metal_kernel` and returns the output arrays.
#[allow(clippy::too_many_arguments)]
pub fn metal_kernel(
    name: &str,
    inputs: &[&MlxArray],
    outputs: &[&MlxArray],
    source: &str,
    grid: [usize; 3],
    threadgroup: [usize; 3],
    output_shapes: &[&[usize]],
    output_dtypes: &[MlxDtype],
    stream: &MlxStream,
) -> Result<Vec<MlxArray>, MlxSysError> {
    #[cfg(mlx_stub)]
    {
        let _ = (
            name,
            inputs,
            outputs,
            source,
            grid,
            threadgroup,
            output_shapes,
            output_dtypes,
            stream,
        );
        Err(MlxSysError::KernelCompile(
            "mlx-c not available (stub mode)".into(),
        ))
    }

    #[cfg(not(mlx_stub))]
    {
        let c_name = CString::new(name).map_err(|e| MlxSysError::KernelCompile(e.to_string()))?;
        let c_source =
            CString::new(source).map_err(|e| MlxSysError::KernelCompile(e.to_string()))?;

        // Build input vector array.
        let input_vec = unsafe { ffi::mlx_vector_array_new() };
        for arr in inputs {
            unsafe { ffi::mlx_vector_array_add(input_vec, arr.as_raw_ptr()) };
        }

        // Build output vector array.
        let output_vec = unsafe { ffi::mlx_vector_array_new() };
        for arr in outputs {
            unsafe { ffi::mlx_vector_array_add(output_vec, arr.as_raw_ptr()) };
        }

        // Convert output shapes to C-compatible format.
        let shapes_i32: Vec<Vec<i32>> = output_shapes
            .iter()
            .map(|s| s.iter().map(|&d| d as i32).collect())
            .collect();
        let shape_ptrs: Vec<*const i32> = shapes_i32.iter().map(|s| s.as_ptr()).collect();
        let shape_ndims: Vec<i32> = shapes_i32.iter().map(|s| s.len() as i32).collect();

        // Convert dtypes.
        let dtypes_u32: Vec<u32> = output_dtypes.iter().map(|&d| d as u32).collect();

        let result = unsafe {
            ffi::mlx_fast_metal_kernel(
                c_name.as_ptr(),
                input_vec,
                output_vec,
                c_source.as_ptr(),
                grid.as_ptr(),
                threadgroup.as_ptr(),
                shape_ptrs.as_ptr(),
                shape_ndims.as_ptr(),
                dtypes_u32.as_ptr(),
                output_dtypes.len() as i32,
                stream.as_raw_ptr(),
            )
        };

        // Free the input/output vector handles.
        unsafe {
            ffi::mlx_free(input_vec as *mut c_void);
            ffi::mlx_free(output_vec as *mut c_void);
        }

        if result.is_null() {
            return Err(MlxSysError::KernelCompile(
                "mlx_fast_metal_kernel returned null".into(),
            ));
        }

        // Extract results from the returned vector array.
        let n = unsafe { ffi::mlx_vector_array_size(result) };
        let mut out = Vec::with_capacity(n as usize);
        for i in 0..n {
            let arr = unsafe { ffi::mlx_vector_array_get(result, i) };
            if arr.is_null() {
                // Free the result vector before returning error.
                unsafe { ffi::mlx_free(result as *mut c_void) };
                return Err(MlxSysError::KernelCompile(format!(
                    "null array at index {i} in kernel output"
                )));
            }
            // Retain since we're taking ownership.
            unsafe { ffi::mlx_retain(arr) };
            out.push(unsafe { MlxArray::from_raw(arr) });
        }

        // Free the result vector.
        unsafe { ffi::mlx_free(result as *mut c_void) };

        Ok(out)
    }
}
