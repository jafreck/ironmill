//! Objective-C runtime and CoreFoundation FFI helpers for Metal bindings.
//!
//! This module provides the low-level ObjC interop primitives shared across
//! all Metal wrapper modules (device, buffer, command, pipeline, shader, mps).

use std::ffi::c_void;

use crate::error::MetalSysError;

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    pub(crate) fn objc_getClass(name: *const u8) -> *mut c_void;
    pub(crate) fn sel_registerName(name: *const u8) -> *mut c_void;
    pub(crate) fn objc_msgSend(receiver: *mut c_void, sel: *mut c_void) -> *mut c_void;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub(crate) fn CFRelease(cf: *mut c_void);
}

// ---------------------------------------------------------------------------
// sel! macro — null-terminated C strings from Rust literals
// ---------------------------------------------------------------------------

/// Create a null-terminated byte pointer suitable for Objective-C runtime calls.
macro_rules! sel {
    ($s:expr) => {
        concat!($s, "\0").as_ptr()
    };
}
pub(crate) use sel;

// ---------------------------------------------------------------------------
// NSString
// ---------------------------------------------------------------------------

/// Create an `NSString` from a Rust string slice via
/// `[[NSString alloc] initWithUTF8String:]`.
/// Returns a retained object — caller must `CFRelease` when done.
pub(crate) fn create_nsstring(s: &str) -> Result<*mut c_void, MetalSysError> {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let nsstring_cls = unsafe { objc_getClass(sel!("NSString")) };
    if nsstring_cls.is_null() {
        return Err(MetalSysError::FrameworkNotFound);
    }

    let mut buf = Vec::with_capacity(s.len() + 1);
    buf.extend_from_slice(s.as_bytes());
    buf.push(0);

    type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let alloc_sel = unsafe { sel_registerName(sel!("alloc")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let alloc_fn: AllocFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: sending +alloc to a valid class pointer.
    let raw = unsafe { alloc_fn(nsstring_cls, alloc_sel) };
    if raw.is_null() {
        return Err(MetalSysError::InvalidArgument(
            "NSString alloc failed".into(),
        ));
    }

    type InitFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *const u8) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let init_sel = unsafe { sel_registerName(sel!("initWithUTF8String:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let init_fn: InitFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: raw is a valid allocated NSString; buf is null-terminated.
    let obj = unsafe { init_fn(raw, init_sel, buf.as_ptr()) };
    if obj.is_null() {
        return Err(MetalSysError::InvalidArgument(
            "failed to create NSString".into(),
        ));
    }
    Ok(obj)
}

// ---------------------------------------------------------------------------
// NSString extraction
// ---------------------------------------------------------------------------

/// Extract a Rust `String` from an `NSString` via `-[NSString UTF8String]`.
pub(crate) fn nsstring_to_string(nsstring: *mut c_void) -> String {
    if nsstring.is_null() {
        return String::new();
    }
    type Utf8Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const u8;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let utf8_sel = unsafe { sel_registerName(sel!("UTF8String")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let utf8_fn: Utf8Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: nsstring is a valid NSString.
    let cstr = unsafe { utf8_fn(nsstring, utf8_sel) };
    if cstr.is_null() {
        return String::new();
    }
    // SAFETY: cstr is a valid null-terminated UTF-8 string from NSString.
    unsafe { std::ffi::CStr::from_ptr(cstr as *const i8) }
        .to_str()
        .unwrap_or("")
        .to_string()
}

// ---------------------------------------------------------------------------
// NSError extraction
// ---------------------------------------------------------------------------

/// Extract the `localizedDescription` string from an `NSError`.
pub(crate) fn extract_nserror_description(error: *mut c_void) -> String {
    type DescFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("localizedDescription")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: DescFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: error is a valid NSError object.
    let desc = unsafe { f(error, sel) };
    if desc.is_null() {
        return "unknown error (nil description)".into();
    }
    nsstring_to_string(desc)
}

// ---------------------------------------------------------------------------
// ObjC retain helper
// ---------------------------------------------------------------------------

/// Send `retain` to an ObjC object, incrementing its reference count.
pub(crate) fn objc_retain(obj: *mut c_void) {
    type RetainFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let retain_sel = unsafe { sel_registerName(sel!("retain")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let retain_fn: RetainFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: obj is a valid ObjC object.
    unsafe { retain_fn(obj, retain_sel) };
}
