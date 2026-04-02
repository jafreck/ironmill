//! Safe wrapper for `mlx_fast_metal_kernel`.

use crate::MlxDtype;
use crate::array::MlxArray;
use crate::error::MlxSysError;
use crate::stream::MlxStream;

#[cfg(not(mlx_stub))]
use crate::ffi;
#[cfg(not(mlx_stub))]
use std::ffi::CString;

/// Launch a custom Metal kernel via mlx-c.
///
/// This wraps the `mlx_fast_metal_kernel` + config API and returns the output
/// arrays.
#[allow(clippy::too_many_arguments)]
pub fn metal_kernel(
    name: &str,
    inputs: &[&MlxArray],
    _outputs: &[&MlxArray],
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
            _outputs,
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

        // Build input / output name vectors. Use generic names since the
        // original API didn't require explicit names.
        let input_names: Vec<CString> = (0..inputs.len())
            .map(|i| CString::new(format!("input{i}")).unwrap())
            .collect();
        let output_names: Vec<CString> = (0..output_shapes.len())
            .map(|i| CString::new(format!("output{i}")).unwrap())
            .collect();

        let input_name_ptrs: Vec<*const std::ffi::c_char> =
            input_names.iter().map(|s| s.as_ptr()).collect();
        let output_name_ptrs: Vec<*const std::ffi::c_char> =
            output_names.iter().map(|s| s.as_ptr()).collect();

        let in_names_vec = unsafe {
            ffi::mlx_vector_string_new_data(
                input_name_ptrs.as_ptr() as *mut *const std::ffi::c_char,
                input_name_ptrs.len(),
            )
        };
        let out_names_vec = unsafe {
            ffi::mlx_vector_string_new_data(
                output_name_ptrs.as_ptr() as *mut *const std::ffi::c_char,
                output_name_ptrs.len(),
            )
        };

        // Create the metal kernel object.
        let empty_header = c"";
        let kernel = unsafe {
            ffi::mlx_fast_metal_kernel_new(
                c_name.as_ptr(),
                in_names_vec,
                out_names_vec,
                c_source.as_ptr(),
                empty_header.as_ptr(),
                false, // ensure_row_contiguous
                false, // atomic_outputs
            )
        };

        unsafe {
            ffi::mlx_vector_string_free(in_names_vec);
            ffi::mlx_vector_string_free(out_names_vec);
        }

        // Build the config.
        let config = unsafe { ffi::mlx_fast_metal_kernel_config_new() };
        unsafe {
            ffi::mlx_fast_metal_kernel_config_set_grid(
                config,
                grid[0] as i32,
                grid[1] as i32,
                grid[2] as i32,
            );
            ffi::mlx_fast_metal_kernel_config_set_thread_group(
                config,
                threadgroup[0] as i32,
                threadgroup[1] as i32,
                threadgroup[2] as i32,
            );
        }

        // Add output specifications.
        for (shape, &dtype) in output_shapes.iter().zip(output_dtypes.iter()) {
            let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
            unsafe {
                ffi::mlx_fast_metal_kernel_config_add_output_arg(
                    config,
                    shape_i32.as_ptr(),
                    shape_i32.len(),
                    dtype as u32,
                );
            }
        }

        // Build input vector array.
        let input_vec = unsafe { ffi::mlx_vector_array_new() };
        for arr in inputs {
            unsafe { ffi::mlx_vector_array_append_value(input_vec, arr.raw) };
        }

        // Apply the kernel.
        let mut output_vec = unsafe { ffi::mlx_vector_array_new() };
        let ret = unsafe {
            ffi::mlx_fast_metal_kernel_apply(&mut output_vec, kernel, input_vec, config, stream.raw)
        };

        // Cleanup kernel resources.
        unsafe {
            ffi::mlx_fast_metal_kernel_free(kernel);
            ffi::mlx_fast_metal_kernel_config_free(config);
            ffi::mlx_vector_array_free(input_vec);
        }

        if ret != 0 {
            unsafe { ffi::mlx_vector_array_free(output_vec) };
            return Err(MlxSysError::KernelCompile(
                "mlx_fast_metal_kernel_apply failed".into(),
            ));
        }

        // Extract results from the returned vector array.
        let n = unsafe { ffi::mlx_vector_array_size(output_vec) };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut arr = unsafe { ffi::mlx_array_new() };
            let ret = unsafe { ffi::mlx_vector_array_get(&mut arr, output_vec, i) };
            if ret != 0 || arr.ctx.is_null() {
                unsafe {
                    ffi::mlx_array_free(arr);
                    ffi::mlx_vector_array_free(output_vec);
                }
                return Err(MlxSysError::KernelCompile(format!(
                    "null array at index {i} in kernel output"
                )));
            }
            out.push(unsafe { MlxArray::from_raw(arr) });
        }

        unsafe { ffi::mlx_vector_array_free(output_vec) };

        Ok(out)
    }
}
