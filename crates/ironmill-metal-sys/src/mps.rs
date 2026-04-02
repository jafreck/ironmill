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

// SAFETY: Declares the MetalPerformanceShaders framework link for MPS class access.
#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {}

/// MPS data-type encoding for 16-bit floating point (`MPSDataTypeFloat16`).
///
/// The MPS type system encodes the floating-point flag in bit 28
/// (`0x10000000`) and the bit-width in the lower bits, giving
/// `0x10000000 | 16 = 268435472`.
const MPS_DATA_TYPE_FLOAT16: usize = 0x10000000 | 16;

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

/// Configuration for creating an [`MpsMatrixMultiply`] kernel.
pub struct MpsMatrixMultiplyConfig {
    /// Whether to transpose the left matrix.
    pub transpose_left: bool,
    /// Whether to transpose the right matrix.
    pub transpose_right: bool,
    /// Number of rows in the result matrix.
    pub result_rows: usize,
    /// Number of columns in the result matrix.
    pub result_columns: usize,
    /// The shared dimension (columns of left / rows of right).
    pub interior_columns: usize,
    /// Scalar multiplier for the product.
    pub alpha: f64,
    /// Scalar multiplier for the existing result (for accumulate).
    pub beta: f64,
}

impl MpsMatrixMultiply {
    /// Create a new MPS matrix multiplication kernel.
    pub fn new(
        device: &MetalDevice,
        config: &MpsMatrixMultiplyConfig,
    ) -> Result<Self, MetalSysError> {
        // SAFETY: objc_getClass with a valid null-terminated class name.
        // The MPS framework is linked above, so the class is available.
        let cls = unsafe { objc::objc_getClass(sel!("MPSMatrixMultiplication")) };
        if cls.is_null() {
            return Err(MetalSysError::Mps(
                "MPSMatrixMultiplication class not found".into(),
            ));
        }

        // +alloc
        // SAFETY: `cls` is a valid ObjC class pointer (null-checked above).
        // "alloc" returns a +1 retained uninitialized instance.
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
        // SAFETY: `obj` is a valid allocated MPSMatrixMultiplication (null-checked
        // above). `device.as_raw_ptr()` is a valid id<MTLDevice>. The init
        // method consumes the alloc'd object and returns a retained initialized
        // instance or nil.
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
                config.transpose_left,
                config.transpose_right,
                config.result_rows,
                config.result_columns,
                config.interior_columns,
                config.alpha,
                config.beta,
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
        // SAFETY: `self.raw` is a valid retained MPSMatrixMultiplication. All
        // argument pointers are valid retained ObjC objects obtained from their
        // respective safe wrappers.
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
            // SAFETY: self.raw is a retained ObjC object; null-checked above.
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
        let data_type_f16: usize = MPS_DATA_TYPE_FLOAT16;

        // SAFETY: objc_getClass with a valid null-terminated class name.
        let desc_cls = unsafe { objc::objc_getClass(sel!("MPSMatrixDescriptor")) };
        if desc_cls.is_null() {
            return Err(MetalSysError::Mps(
                "MPSMatrixDescriptor class not found".into(),
            ));
        }

        // SAFETY: `desc_cls` is a valid class (null-checked above).
        // "matrixDescriptorWithRows:columns:rowBytes:dataType:" is a class
        // method that returns an autoreleased descriptor.
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
        // SAFETY: objc_getClass with a valid null-terminated class name.
        let matrix_cls = unsafe { objc::objc_getClass(sel!("MPSMatrix")) };
        if matrix_cls.is_null() {
            return Err(MetalSysError::Mps("MPSMatrix class not found".into()));
        }

        // SAFETY: `matrix_cls` is a valid class (null-checked above).
        // "alloc" returns a +1 retained uninitialized instance.
        type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let alloc_sel = unsafe { objc::sel_registerName(sel!("alloc")) };
        let alloc_fn: AllocFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        let obj = unsafe { alloc_fn(matrix_cls, alloc_sel) };
        if obj.is_null() {
            return Err(MetalSysError::Mps("MPSMatrix alloc failed".into()));
        }

        // SAFETY: `obj` is a valid allocated MPSMatrix (null-checked above).
        // `buffer.as_raw_ptr()` is a valid id<MTLBuffer> and `descriptor` is
        // a valid MPSMatrixDescriptor. "initWithBuffer:descriptor:" returns a
        // retained initialized instance or nil.
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
            // SAFETY: self.raw is a retained ObjC object; null-checked above.
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
