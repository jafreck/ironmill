//! Safe wrapper for `MTLBuffer`.

use std::ffi::c_void;

use crate::error::MetalSysError;
use crate::objc::{self, sel};

// ---------------------------------------------------------------------------
// StorageMode
// ---------------------------------------------------------------------------

/// Metal buffer storage mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageMode {
    /// CPU and GPU share the same memory (unified memory). Accessible from CPU.
    Shared,
    /// GPU-only memory. Not accessible from CPU.
    Private,
    /// Separate CPU and GPU copies, manually synchronized.
    Managed,
}

impl StorageMode {
    /// Returns the `MTLResourceOptions` value for this storage mode.
    pub(crate) fn resource_options(self) -> usize {
        match self {
            // MTLResourceStorageModeShared = 0 << 4
            StorageMode::Shared => 0,
            // MTLResourceStorageModeManaged = 1 << 4
            StorageMode::Managed => 1 << 4,
            // MTLResourceStorageModePrivate = 2 << 4
            StorageMode::Private => 2 << 4,
        }
    }
}

// ---------------------------------------------------------------------------
// MetalBuffer
// ---------------------------------------------------------------------------

/// Safe wrapper around an `MTLBuffer` (id<MTLBuffer>).
pub struct MetalBuffer {
    raw: *mut c_void,
    mode: StorageMode,
}

// SAFETY: Metal buffers can be moved between threads.
// Not Sync: write_bytes/read_bytes mutate raw memory through &self.
unsafe impl Send for MetalBuffer {}

impl MetalBuffer {
    /// Create a `MetalBuffer` from a raw pointer and known storage mode.
    ///
    /// The pointer must be a valid, retained `id<MTLBuffer>`.
    pub(crate) fn from_raw(raw: *mut c_void, mode: StorageMode) -> Self {
        Self { raw, mode }
    }

    /// Returns the length of the buffer in bytes.
    pub fn length(&self) -> usize {
        // SAFETY: `self.raw` is a valid, retained id<MTLBuffer>.
        // "length" returns the buffer's allocation size in bytes.
        type LenFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize;
        let sel = unsafe { objc::sel_registerName(sel!("length")) };
        let f: LenFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Returns a pointer to the buffer's CPU-accessible contents.
    ///
    /// Only valid for `Shared` and `Managed` storage modes. Returns a null
    /// pointer for `Private` buffers.
    pub fn contents(&self) -> *mut c_void {
        // SAFETY: `self.raw` is a valid, retained id<MTLBuffer>. "contents"
        // returns a CPU-accessible pointer (or null for Private buffers).
        type ContentsFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { objc::sel_registerName(sel!("contents")) };
        let f: ContentsFn = unsafe { std::mem::transmute(objc::objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    /// Returns the raw `id<MTLBuffer>` pointer.
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw
    }

    /// Returns the storage mode of this buffer.
    pub fn storage_mode(&self) -> StorageMode {
        self.mode
    }

    /// Write bytes into the buffer at the given byte offset.
    ///
    /// Only valid for `Shared` mode buffers.
    pub fn write_bytes(&self, data: &[u8], offset: usize) -> Result<(), MetalSysError> {
        if self.mode == StorageMode::Private {
            return Err(MetalSysError::InvalidArgument(
                "cannot write_bytes to a Private buffer".into(),
            ));
        }
        if self.mode == StorageMode::Managed {
            return Err(MetalSysError::InvalidArgument(
                "cannot write_bytes to a Managed buffer".into(),
            ));
        }
        let len = self.length();
        if offset.checked_add(data.len()).is_none_or(|end| end > len) {
            return Err(MetalSysError::InvalidArgument(format!(
                "write_bytes: offset({offset}) + data.len({}) > buffer length({len})",
                data.len()
            )));
        }
        let ptr = self.contents();
        if ptr.is_null() {
            return Err(MetalSysError::BufferAllocation(
                "buffer contents pointer is null".into(),
            ));
        }
        // SAFETY: We verified bounds and that the pointer is non-null.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), (ptr as *mut u8).add(offset), data.len());
        }
        Ok(())
    }

    /// Read bytes from the buffer at the given byte offset.
    ///
    /// Only valid for `Shared` mode buffers.
    pub fn read_bytes(&self, dst: &mut [u8], offset: usize) -> Result<(), MetalSysError> {
        if self.mode == StorageMode::Private {
            return Err(MetalSysError::InvalidArgument(
                "cannot read_bytes from a Private buffer".into(),
            ));
        }
        let len = self.length();
        if offset.checked_add(dst.len()).is_none_or(|end| end > len) {
            return Err(MetalSysError::InvalidArgument(format!(
                "read_bytes: offset({offset}) + dst.len({}) > buffer length({len})",
                dst.len()
            )));
        }
        let ptr = self.contents();
        if ptr.is_null() {
            return Err(MetalSysError::BufferAllocation(
                "buffer contents pointer is null".into(),
            ));
        }
        // SAFETY: We verified bounds and that the pointer is non-null.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (ptr as *const u8).add(offset),
                dst.as_mut_ptr(),
                dst.len(),
            );
        }
        Ok(())
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: CFRelease on a retained ObjC object.
            unsafe { objc::CFRelease(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}
