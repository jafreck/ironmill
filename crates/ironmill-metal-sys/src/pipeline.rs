//! Safe wrapper for `MTLComputePipelineState`.

use std::ffi::c_void;

use crate::objc::{self, sel};

// ---------------------------------------------------------------------------
// ComputePipeline
// ---------------------------------------------------------------------------

/// Safe wrapper around an `MTLComputePipelineState`.
pub struct ComputePipeline {
    raw: *mut c_void,
}

// SAFETY: MTLComputePipelineState is thread-safe and immutable once created.
unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    /// Create from a raw retained `id<MTLComputePipelineState>`.
    pub(crate) fn from_raw(raw: *mut c_void) -> Self {
        Self { raw }
    }

    /// Returns the maximum number of threads per threadgroup for this pipeline.
    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        type LenFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize;
        let sel = unsafe { objc::sel_registerName(sel!("maxTotalThreadsPerThreadgroup")) };
        let f: LenFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Returns the thread execution width (SIMD width) for this pipeline.
    pub fn thread_execution_width(&self) -> usize {
        type LenFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize;
        let sel = unsafe { objc::sel_registerName(sel!("threadExecutionWidth")) };
        let f: LenFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Returns the raw `id<MTLComputePipelineState>` pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: CFRelease on a retained ObjC object.
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
