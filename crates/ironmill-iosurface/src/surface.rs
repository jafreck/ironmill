//! IOSurface FFI bindings and safe creation/locking helpers.

use std::ffi::c_void;

// ── FFI bindings (macOS) ─────────────────────────────────────────

#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub(crate) mod ffi {
    use std::ffi::c_void;

    #[link(name = "IOSurface", kind = "framework")]
    unsafe extern "C" {
        pub fn IOSurfaceCreate(properties: *const c_void) -> *mut c_void;
        pub fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
        pub fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
        pub fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
        pub fn IOSurfaceGetAllocSize(surface: *mut c_void) -> usize;
    }

    #[link(name = "IOSurface", kind = "framework")]
    unsafe extern "C" {
        pub static kIOSurfaceAllocSize: *const c_void;
        pub static kIOSurfaceWidth: *const c_void;
        pub static kIOSurfaceHeight: *const c_void;
        pub static kIOSurfaceBytesPerElement: *const c_void;
        pub static kIOSurfaceBytesPerRow: *const c_void;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub fn CFRelease(cf: *mut c_void);
        pub fn CFDictionaryCreateMutable(
            allocator: *const c_void,
            capacity: isize,
            key_callbacks: *const c_void,
            value_callbacks: *const c_void,
        ) -> *mut c_void;
        pub fn CFDictionarySetValue(dict: *mut c_void, key: *const c_void, value: *const c_void);
        pub fn CFNumberCreate(
            allocator: *const c_void,
            number_type: i64,
            value_ptr: *const c_void,
        ) -> *mut c_void;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub static kCFAllocatorDefault: *const c_void;
        // Opaque structs — we only need their addresses.
        pub static kCFTypeDictionaryKeyCallBacks: u8;
        pub static kCFTypeDictionaryValueCallBacks: u8;
    }

    /// `kCFNumberSInt64Type` = 4.
    pub const CF_NUMBER_SINT64_TYPE: i64 = 4;

    /// `kIOSurfaceLockReadOnly` = 1.
    pub const IOSURFACE_LOCK_READ_ONLY: u32 = 1;
}

// ── Error type ───────────────────────────────────────────────────

/// Errors from IOSurface operations.
#[derive(Debug, thiserror::Error)]
pub enum IOSurfaceError {
    /// IOSurface allocation failed.
    #[error("IOSurface allocation failed: {0}")]
    AllocFailed(String),

    /// IOSurface lock/unlock or base-address retrieval failed.
    #[error("IOSurface lock failed: {0}")]
    LockFailed(String),

    /// Data copy to/from an IOSurface failed.
    #[error("IOSurface copy failed: {0}")]
    CopyFailed(String),
}

// ── Constants ────────────────────────────────────────────────────

/// Minimum IOSurface allocation size required by the ANE hardware.
///
/// The ANE rejects IOSurface tensors below this size with status 0x1d.
/// We use 16 KB as a safe lower bound.
pub(crate) const ANE_MIN_SURFACE_BYTES: usize = 16384; // 16 KB

// ── IOSurface creation ───────────────────────────────────────────

#[cfg(target_os = "macos")]
pub(crate) fn create_iosurface(alloc_size: usize) -> crate::Result<*mut c_void> {
    // SAFETY: All FFI calls use CoreFoundation/IOSurface APIs correctly:
    // - CFDictionaryCreateMutable is given valid callback struct addresses
    // - CFNumberCreate values are valid i64 pointers
    // - CFDictionarySetValue uses keys/values from the same CF allocator
    // - IOSurfaceCreate takes a valid CFDictionary
    // - All CF objects are released after use
    // - Return value is checked for null before use
    unsafe {
        let dict = ffi::CFDictionaryCreateMutable(
            ffi::kCFAllocatorDefault,
            5,
            std::ptr::addr_of!(ffi::kCFTypeDictionaryKeyCallBacks) as *const c_void,
            std::ptr::addr_of!(ffi::kCFTypeDictionaryValueCallBacks) as *const c_void,
        );
        if dict.is_null() {
            return Err(IOSurfaceError::AllocFailed(
                "CFDictionaryCreateMutable returned null".into(),
            ));
        }

        let alloc_i64 = alloc_size as i64;
        let one_i64: i64 = 1;
        let props: [(*const c_void, i64); 5] = [
            (ffi::kIOSurfaceAllocSize, alloc_i64),
            (ffi::kIOSurfaceWidth, alloc_i64),
            (ffi::kIOSurfaceHeight, one_i64),
            (ffi::kIOSurfaceBytesPerElement, one_i64),
            (ffi::kIOSurfaceBytesPerRow, alloc_i64),
        ];

        let mut cf_numbers = Vec::with_capacity(5);
        for &(key, value) in &props {
            let num = ffi::CFNumberCreate(
                ffi::kCFAllocatorDefault,
                ffi::CF_NUMBER_SINT64_TYPE,
                &value as *const i64 as *const c_void,
            );
            ffi::CFDictionarySetValue(dict, key, num);
            cf_numbers.push(num);
        }

        let surface = ffi::IOSurfaceCreate(dict);

        for num in cf_numbers {
            ffi::CFRelease(num);
        }
        ffi::CFRelease(dict);

        if surface.is_null() {
            return Err(IOSurfaceError::AllocFailed(
                "IOSurfaceCreate returned null".into(),
            ));
        }

        Ok(surface)
    }
}
