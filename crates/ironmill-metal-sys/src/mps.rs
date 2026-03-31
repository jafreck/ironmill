//! Safe wrappers for Metal Performance Shaders (MPS) matrix operations.

use std::ffi::c_void;

use crate::buffer::MetalBuffer;
use crate::command::CommandBuffer;
use crate::device::MetalDevice;
use crate::error::MetalSysError;
use crate::objc::{self, sel};

// ---------------------------------------------------------------------------
// Framework link
// ---------------------------------------------------------------------------

#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {}

// ---------------------------------------------------------------------------
// MpsMatrixMultiply
// ---------------------------------------------------------------------------

/// Safe wrapper around `MPSMatrixMultiplication`.
pub struct MpsMatrixMultiply {
    raw: *mut c_void,
}

// SAFETY: MPS objects are thread-safe once fully initialized.
unsafe impl Send for MpsMatrixMultiply {}
unsafe impl Sync for MpsMatrixMultiply {}

impl MpsMatrixMultiply {
    /// Create a new MPS matrix multiplication kernel.
    ///
    /// # Arguments
    /// * `device` — The Metal device to create the kernel on.
    /// * `transpose_left` — Whether to transpose the left matrix.
    /// * `transpose_right` — Whether to transpose the right matrix.
    /// * `result_rows` — Number of rows in the result matrix.
    /// * `result_columns` — Number of columns in the result matrix.
    /// * `interior_columns` — The shared dimension (columns of left / rows of right).
    /// * `alpha` — Scalar multiplier for the product.
    /// * `beta` — Scalar multiplier for the existing result (for accumulate).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &MetalDevice,
        transpose_left: bool,
        transpose_right: bool,
        result_rows: usize,
        result_columns: usize,
        interior_columns: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<Self, MetalSysError> {
        let cls = unsafe { objc::objc_getClass(sel!("MPSMatrixMultiplication")) };
        if cls.is_null() {
            return Err(MetalSysError::Mps(
                "MPSMatrixMultiplication class not found".into(),
            ));
        }

        // +alloc
        type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let alloc_sel = unsafe { objc::sel_registerName(sel!("alloc")) };
        let alloc_fn: AllocFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let obj = unsafe { alloc_fn(cls, alloc_sel) };
        if obj.is_null() {
            return Err(MetalSysError::Mps(
                "MPSMatrixMultiplication alloc failed".into(),
            ));
        }

        // -initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:
        type InitFn = unsafe extern "C" fn(
            *mut c_void, // self
            *mut c_void, // _cmd
            *mut c_void, // device
            bool,        // transposeLeft
            bool,        // transposeRight
            usize,       // resultRows
            usize,       // resultColumns
            usize,       // interiorColumns
            f64,         // alpha
            f64,         // beta
        ) -> *mut c_void;
        let init_sel = unsafe {
            objc::sel_registerName(sel!(
                "initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:"
            ))
        };
        let init_fn: InitFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw = unsafe {
            init_fn(
                obj,
                init_sel,
                device.as_raw_ptr(),
                transpose_left,
                transpose_right,
                result_rows,
                result_columns,
                interior_columns,
                alpha,
                beta,
            )
        };
        if raw.is_null() {
            return Err(MetalSysError::Mps(
                "MPSMatrixMultiplication init failed".into(),
            ));
        }
        Ok(Self { raw })
    }

    /// Encode the matrix multiplication into a command buffer.
    ///
    /// This does **not** commit the command buffer — the caller must call
    /// `command_buffer.commit()` after encoding.
    pub fn encode(
        &self,
        command_buffer: &CommandBuffer,
        left: &MpsMatrix,
        right: &MpsMatrix,
        result: &MpsMatrix,
    ) {
        // -encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:
        type EncodeFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        );
        let sel = unsafe {
            objc::sel_registerName(sel!(
                "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"
            ))
        };
        let f: EncodeFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe {
            f(
                self.raw,
                sel,
                command_buffer.as_raw_ptr(),
                left.as_raw_ptr(),
                right.as_raw_ptr(),
                result.as_raw_ptr(),
            )
        };
    }
}

impl Drop for MpsMatrixMultiply {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// MpsMatrix
// ---------------------------------------------------------------------------

/// Safe wrapper around `MPSMatrix`.
pub struct MpsMatrix {
    raw: *mut c_void,
}

// SAFETY: MPSMatrix is thread-safe once created.
unsafe impl Send for MpsMatrix {}
unsafe impl Sync for MpsMatrix {}

impl MpsMatrix {
    /// Create an `MPSMatrix` descriptor and then an `MPSMatrix` from a Metal buffer.
    ///
    /// The buffer is expected to contain FP16 data with the given layout.
    ///
    /// # Arguments
    /// * `buffer` — The Metal buffer containing the matrix data.
    /// * `rows` — Number of rows.
    /// * `columns` — Number of columns.
    /// * `row_bytes` — Stride in bytes between rows.
    pub fn from_buffer(
        buffer: &MetalBuffer,
        rows: usize,
        columns: usize,
        row_bytes: usize,
    ) -> Result<Self, MetalSysError> {
        // First create an MPSMatrixDescriptor.
        // +[MPSMatrixDescriptor matrixDescriptorWithRows:columns:rowBytes:dataType:]
        // MPSDataTypeFloat16 = 0x10000000 | 16 = 268435472
        let data_type_f16: usize = 0x10000000 | 16;

        let desc_cls = unsafe { objc::objc_getClass(sel!("MPSMatrixDescriptor")) };
        if desc_cls.is_null() {
            return Err(MetalSysError::Mps(
                "MPSMatrixDescriptor class not found".into(),
            ));
        }

        type DescFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            usize,
            usize,
            usize,
            usize,
        ) -> *mut c_void;
        let desc_sel = unsafe {
            objc::sel_registerName(sel!("matrixDescriptorWithRows:columns:rowBytes:dataType:"))
        };
        let desc_fn: DescFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let descriptor =
            unsafe { desc_fn(desc_cls, desc_sel, rows, columns, row_bytes, data_type_f16) };
        if descriptor.is_null() {
            return Err(MetalSysError::Mps(
                "failed to create MPSMatrixDescriptor".into(),
            ));
        }

        // Now create the MPSMatrix with the buffer and descriptor.
        // [[MPSMatrix alloc] initWithBuffer:descriptor:]
        let matrix_cls = unsafe { objc::objc_getClass(sel!("MPSMatrix")) };
        if matrix_cls.is_null() {
            return Err(MetalSysError::Mps("MPSMatrix class not found".into()));
        }

        type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let alloc_sel = unsafe { objc::sel_registerName(sel!("alloc")) };
        let alloc_fn: AllocFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let obj = unsafe { alloc_fn(matrix_cls, alloc_sel) };
        if obj.is_null() {
            return Err(MetalSysError::Mps("MPSMatrix alloc failed".into()));
        }

        type InitFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let init_sel = unsafe { objc::sel_registerName(sel!("initWithBuffer:descriptor:")) };
        let init_fn: InitFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let raw = unsafe { init_fn(obj, init_sel, buffer.as_raw_ptr(), descriptor) };
        if raw.is_null() {
            return Err(MetalSysError::Mps("MPSMatrix init failed".into()));
        }

        Ok(Self { raw })
    }

    /// Returns the raw `MPSMatrix` pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for MpsMatrix {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
