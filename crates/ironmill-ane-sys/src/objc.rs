//! Consolidated Objective-C runtime and CoreFoundation FFI helpers.
//!
//! Both the ANE compiler and runtime need the same set of low-level
//! ObjC interop primitives.  This module provides a single copy of
//! those declarations and helper functions.

use std::ffi::c_void;

use crate::error::AneSysError;

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    pub(crate) fn objc_getClass(name: *const u8) -> *mut c_void;
    pub(crate) fn sel_registerName(name: *const u8) -> *mut c_void;
    pub(crate) fn objc_msgSend(receiver: *mut c_void, sel: *mut c_void) -> *mut c_void;
}

#[link(name = "dl")]
unsafe extern "C" {
    pub(crate) fn dlopen(path: *const u8, mode: i32) -> *mut c_void;
}

pub(crate) const RTLD_NOW: i32 = 0x2;

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
// ObjC introspection
// ---------------------------------------------------------------------------

/// Check if an ObjC object responds to a given selector.
pub(crate) fn responds_to_selector(obj: *mut c_void, sel: *mut c_void) -> bool {
    type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
    // SAFETY: obj and sel are valid ObjC pointers; respondsToSelector: is
    // defined on NSObject and always safe to call.
    let rts_sel = unsafe { sel_registerName(sel!("respondsToSelector:")) };
    let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(obj, rts_sel, sel) != 0 }
}

// ---------------------------------------------------------------------------
// NSData
// ---------------------------------------------------------------------------

/// Create an `NSData` from a byte slice via `[[NSData alloc] initWithBytes:length:]`.
/// Returns a retained object — caller must `CFRelease` when done.
pub(crate) fn create_nsdata(bytes: &[u8]) -> Result<*mut c_void, AneSysError> {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let nsdata_cls = unsafe { objc_getClass(sel!("NSData")) };
    if nsdata_cls.is_null() {
        return Err(AneSysError::FrameworkNotFound(
            "NSData class not found".into(),
        ));
    }

    type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let alloc_sel = unsafe { sel_registerName(sel!("alloc")) };
    // SAFETY: transmute objc_msgSend to the correct signature for +[NSData alloc].
    let alloc_fn: AllocFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: sending +alloc to a valid class pointer.
    let raw = unsafe { alloc_fn(nsdata_cls, alloc_sel) };
    if raw.is_null() {
        return Err(AneSysError::CompilationFailed("NSData alloc failed".into()));
    }

    type InitFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *const u8, usize) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let init_sel = unsafe { sel_registerName(sel!("initWithBytes:length:")) };
    // SAFETY: transmute objc_msgSend to the correct signature for -initWithBytes:length:.
    let init_fn: InitFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: raw is a valid allocated NSData; bytes pointer and length are valid.
    let obj = unsafe { init_fn(raw, init_sel, bytes.as_ptr(), bytes.len()) };
    if obj.is_null() {
        return Err(AneSysError::CompilationFailed(
            "failed to create NSData".into(),
        ));
    }
    Ok(obj)
}

// ---------------------------------------------------------------------------
// NSString
// ---------------------------------------------------------------------------

/// Create an `NSString` from a Rust string slice via
/// `[[NSString alloc] initWithUTF8String:]`.
/// Returns a retained object — caller must `CFRelease` when done.
pub(crate) fn create_nsstring(s: &str) -> Result<*mut c_void, AneSysError> {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let nsstring_cls = unsafe { objc_getClass(sel!("NSString")) };
    if nsstring_cls.is_null() {
        return Err(AneSysError::FrameworkNotFound(
            "NSString class not found".into(),
        ));
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
        return Err(AneSysError::CompilationFailed(
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
        return Err(AneSysError::CompilationFailed(
            "failed to create NSString".into(),
        ));
    }
    Ok(obj)
}

// ---------------------------------------------------------------------------
// NSNumber
// ---------------------------------------------------------------------------

/// Create an `NSNumber` from an `i64` via `[[NSNumber alloc] initWithLongLong:]`.
/// Returns a retained object — caller must `CFRelease` when done.
pub(crate) fn create_nsnumber(value: i64) -> Result<*mut c_void, AneSysError> {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSNumber")) };
    if cls.is_null() {
        return Err(AneSysError::FrameworkNotFound(
            "NSNumber class not found".into(),
        ));
    }

    type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let alloc_sel = unsafe { sel_registerName(sel!("alloc")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let alloc_fn: AllocFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: sending +alloc to a valid class pointer.
    let raw = unsafe { alloc_fn(cls, alloc_sel) };
    if raw.is_null() {
        return Err(AneSysError::CompilationFailed(
            "NSNumber alloc failed".into(),
        ));
    }

    type InitFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i64) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let init_sel = unsafe { sel_registerName(sel!("initWithLongLong:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let init_fn: InitFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: raw is a valid allocated NSNumber; value is a plain i64.
    let obj = unsafe { init_fn(raw, init_sel, value) };
    if obj.is_null() {
        return Err(AneSysError::CompilationFailed(
            "failed to create NSNumber".into(),
        ));
    }
    Ok(obj)
}

/// Create an `NSNumber` from an `i64` via `+[NSNumber numberWithLongLong:]`.
/// Returns an autoreleased object.
pub(crate) fn ns_number_autoreleased(value: i64) -> *mut c_void {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSNumber")) };
    type NumFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i64) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("numberWithLongLong:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: NumFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: cls is a valid class pointer; value is a plain i64.
    unsafe { f(cls, sel, value) }
}

// ---------------------------------------------------------------------------
// NSDictionary / NSMutableDictionary
// ---------------------------------------------------------------------------

/// Create an empty `NSDictionary`.
pub(crate) fn ns_empty_dict() -> Result<*mut c_void, AneSysError> {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSDictionary")) };
    if cls.is_null() {
        return Err(AneSysError::FrameworkNotFound(
            "NSDictionary class not found".into(),
        ));
    }

    type NewFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let new_sel = unsafe { sel_registerName(sel!("new")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: NewFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: sending +new to a valid class pointer.
    let obj = unsafe { f(cls, new_sel) };
    if obj.is_null() {
        return Err(AneSysError::CompilationFailed(
            "failed to create NSDictionary".into(),
        ));
    }
    Ok(obj)
}

/// Create an empty `NSDictionary` (autoreleased, infallible).
///
/// Used by the runtime where Foundation classes are already resolved.
pub(crate) fn ns_empty_dict_unchecked() -> *mut c_void {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSDictionary")) };
    type NewFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let new_sel = unsafe { sel_registerName(sel!("new")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: NewFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: sending +new to a valid class pointer.
    unsafe { f(cls, new_sel) }
}

/// Create an empty `NSMutableDictionary`.
pub(crate) fn ns_mutable_dict() -> Result<*mut c_void, AneSysError> {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSMutableDictionary")) };
    if cls.is_null() {
        return Err(AneSysError::FrameworkNotFound(
            "NSMutableDictionary class not found".into(),
        ));
    }

    type NewFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let new_sel = unsafe { sel_registerName(sel!("new")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: NewFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: sending +new to a valid class pointer.
    let obj = unsafe { f(cls, new_sel) };
    if obj.is_null() {
        return Err(AneSysError::CompilationFailed(
            "failed to create NSMutableDictionary".into(),
        ));
    }
    Ok(obj)
}

/// Set a key-value pair on an `NSMutableDictionary`.
pub(crate) fn ns_dict_set(dict: *mut c_void, key: *mut c_void, value: *mut c_void) {
    type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void);
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("setObject:forKey:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: dict is a valid NSMutableDictionary; key/value are valid ObjC objects.
    unsafe { f(dict, sel, value, key) };
}

// ---------------------------------------------------------------------------
// NSMutableArray
// ---------------------------------------------------------------------------

/// Create an empty `NSMutableArray`.
pub(crate) fn ns_mutable_array() -> *mut c_void {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSMutableArray")) };
    type NewFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let new_sel = unsafe { sel_registerName(sel!("new")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: NewFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: sending +new to a valid class pointer.
    unsafe { f(cls, new_sel) }
}

/// Append an object to an `NSMutableArray`.
pub(crate) fn ns_array_add(array: *mut c_void, object: *mut c_void) {
    type AddFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("addObject:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: AddFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: array is a valid NSMutableArray; object is a valid ObjC object.
    unsafe { f(array, sel, object) };
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

    type Utf8Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const u8;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let utf8_sel = unsafe { sel_registerName(sel!("UTF8String")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let utf8_fn: Utf8Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: desc is a valid NSString returned from localizedDescription.
    let cstr = unsafe { utf8_fn(desc, utf8_sel) };
    if cstr.is_null() {
        return "unknown error (nil UTF8String)".into();
    }

    // SAFETY: cstr is a valid null-terminated UTF-8 string from NSString.
    unsafe { std::ffi::CStr::from_ptr(cstr as *const i8) }
        .to_str()
        .unwrap_or("unknown error (invalid UTF-8)")
        .to_string()
}

// ---------------------------------------------------------------------------
// ObjC retain
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

// ---------------------------------------------------------------------------
// NSURL
// ---------------------------------------------------------------------------

/// Create an `NSURL` from a filesystem path via `[NSURL fileURLWithPath:]`.
#[allow(dead_code)]
pub(crate) fn create_nsurl_from_path(path: &std::path::Path) -> Result<*mut c_void, AneSysError> {
    let path_str = path.to_str().ok_or_else(|| {
        AneSysError::InvalidInput(format!("path contains invalid UTF-8: {}", path.display()))
    })?;

    let nsstring = create_nsstring(path_str)?;

    // SAFETY: objc_getClass with a valid null-terminated class name.
    let nsurl_cls = unsafe { objc_getClass(sel!("NSURL")) };
    if nsurl_cls.is_null() {
        return Err(AneSysError::FrameworkNotFound(
            "NSURL class not found".into(),
        ));
    }

    type FileUrlWithPathFn =
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;

    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("fileURLWithPath:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let send: FileUrlWithPathFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: nsurl_cls is a valid class; nsstring is a valid NSString.
    let url = unsafe { send(nsurl_cls, sel, nsstring) };
    if url.is_null() {
        return Err(AneSysError::CompilationFailed(
            "failed to create NSURL from path".into(),
        ));
    }
    Ok(url)
}
