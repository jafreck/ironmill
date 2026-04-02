//! Consolidated Objective-C runtime and CoreFoundation FFI helpers.
//!
//! Both the ANE compiler and runtime need the same set of low-level
//! ObjC interop primitives.  This module provides a single copy of
//! those declarations and helper functions.

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::error::AneSysError;

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    pub(crate) fn objc_getClass(name: *const u8) -> *mut c_void;
    pub(crate) fn sel_registerName(name: *const u8) -> *mut c_void;
    pub(crate) fn objc_msgSend(receiver: *mut c_void, sel: *mut c_void) -> *mut c_void;
    pub(crate) fn objc_autoreleasePoolPush() -> *mut c_void;
    pub(crate) fn objc_autoreleasePoolPop(pool: *mut c_void);
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

// Mach kernel APIs for process information.
unsafe extern "C" {
    pub(crate) fn mach_task_self() -> u32;
    pub(crate) fn task_info(
        target_task: u32,
        flavor: u32,
        task_info_out: *mut u8,
        task_info_count: *mut u32,
    ) -> i32;
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
// Framework loading
// ---------------------------------------------------------------------------

/// Cached framework handle stored as `usize` because `*mut c_void` is not
/// `Sync`.  The `OnceLock` ensures the dlopen happens exactly once.
static FRAMEWORK: OnceLock<usize> = OnceLock::new();

/// ANE framework path used by dlopen.
const ANE_FRAMEWORK_PATH: &str =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine\0";

/// Lazy framework handle — load the ANE framework once, cache in static.
/// Replaces the inline dlopen calls scattered in compiler.rs and runtime.rs.
pub(crate) fn ane_framework() -> Result<*mut c_void, AneSysError> {
    let &handle = FRAMEWORK.get_or_init(|| {
        // SAFETY: dlopen with a valid null-terminated path.
        let h = unsafe { dlopen(ANE_FRAMEWORK_PATH.as_ptr(), RTLD_NOW) };
        h as usize
    });
    let ptr = handle as *mut c_void;
    if ptr.is_null() {
        return Err(AneSysError::FrameworkNotFound(
            "failed to dlopen AppleNeuralEngine.framework".into(),
        ));
    }
    Ok(ptr)
}

/// Get ObjC class by name from the loaded framework, returning error if not found.
pub(crate) fn get_class(name: &str) -> Result<*mut c_void, AneSysError> {
    // Ensure framework is loaded first.
    ane_framework()?;

    let mut buf = Vec::with_capacity(name.len() + 1);
    buf.extend_from_slice(name.as_bytes());
    buf.push(0);

    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(buf.as_ptr()) };
    if cls.is_null() {
        return Err(AneSysError::ClassNotFound(name.into()));
    }
    Ok(cls)
}

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
pub(crate) fn ns_number_autoreleased(value: i64) -> Result<*mut c_void, AneSysError> {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSNumber")) };
    if cls.is_null() {
        return Err(AneSysError::FrameworkNotFound(
            "NSNumber class not found".into(),
        ));
    }
    type NumFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i64) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("numberWithLongLong:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: NumFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: cls is a valid class pointer; value is a plain i64.
    let obj = unsafe { f(cls, sel, value) };
    if obj.is_null() {
        return Err(AneSysError::CompilationFailed(
            "failed to create NSNumber (autoreleased)".into(),
        ));
    }
    Ok(obj)
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
pub(crate) fn ns_empty_dict_unchecked() -> Result<*mut c_void, AneSysError> {
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
            "failed to create NSDictionary (unchecked)".into(),
        ));
    }
    Ok(obj)
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
pub(crate) fn ns_mutable_array() -> Result<*mut c_void, AneSysError> {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSMutableArray")) };
    if cls.is_null() {
        return Err(AneSysError::FrameworkNotFound(
            "NSMutableArray class not found".into(),
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
            "failed to create NSMutableArray".into(),
        ));
    }
    Ok(obj)
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
        // SAFETY: CFRelease on retained NSString.
        unsafe { CFRelease(nsstring) };
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

    // Release the NSString now that NSURL has consumed it.
    // SAFETY: CFRelease on retained NSString.
    unsafe { CFRelease(nsstring) };

    if url.is_null() {
        return Err(AneSysError::CompilationFailed(
            "failed to create NSURL from path".into(),
        ));
    }
    Ok(url)
}

// ---------------------------------------------------------------------------
// NSArray (from slice)
// ---------------------------------------------------------------------------

/// Create an `NSArray` from a slice of object pointers via
/// `+[NSArray arrayWithObjects:count:]`.
/// Returns an autoreleased object.
///
/// # Safety
///
/// Every pointer in `objects` must be a valid, non-null ObjC object pointer.
#[allow(dead_code)]
pub(crate) unsafe fn create_nsarray(objects: &[*mut c_void]) -> *mut c_void {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSArray")) };
    if cls.is_null() {
        return std::ptr::null_mut();
    }

    type ArrayFn =
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const *mut c_void, usize) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("arrayWithObjects:count:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: ArrayFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: cls is a valid class; objects pointer and count are valid.
    unsafe { f(cls, sel, objects.as_ptr(), objects.len()) }
}

// ---------------------------------------------------------------------------
// NSNumber — typed constructors
// ---------------------------------------------------------------------------

/// Create an `NSNumber` from a `u32` via `+[NSNumber numberWithUnsignedInt:]`.
/// Returns an autoreleased object.
///
/// # Safety
///
/// Must be called from a thread with an active autorelease pool (or caller
/// must arrange to retain/release).
#[allow(dead_code)]
pub(crate) unsafe fn create_nsnumber_u32(val: u32) -> *mut c_void {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSNumber")) };
    if cls.is_null() {
        return std::ptr::null_mut();
    }

    type NumFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u32) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("numberWithUnsignedInt:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: NumFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: cls is a valid class; val is a plain u32.
    unsafe { f(cls, sel, val) }
}

/// Create an `NSNumber` from a `u64` via `+[NSNumber numberWithUnsignedLongLong:]`.
/// Returns an autoreleased object.
///
/// # Safety
///
/// Must be called from a thread with an active autorelease pool (or caller
/// must arrange to retain/release).
#[allow(dead_code)]
pub(crate) unsafe fn create_nsnumber_u64(val: u64) -> *mut c_void {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSNumber")) };
    if cls.is_null() {
        return std::ptr::null_mut();
    }

    type NumFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("numberWithUnsignedLongLong:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: NumFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: cls is a valid class; val is a plain u64.
    unsafe { f(cls, sel, val) }
}

/// Create an `NSNumber` from an `i64` via `+[NSNumber numberWithLongLong:]`.
/// Returns an autoreleased object.
///
/// # Safety
///
/// Must be called from a thread with an active autorelease pool (or caller
/// must arrange to retain/release).
#[allow(dead_code)]
pub(crate) unsafe fn create_nsnumber_i64(val: i64) -> *mut c_void {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSNumber")) };
    if cls.is_null() {
        return std::ptr::null_mut();
    }

    type NumFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i64) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("numberWithLongLong:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: NumFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: cls is a valid class; val is a plain i64.
    unsafe { f(cls, sel, val) }
}

/// Create an `NSNumber` from a `bool` via `+[NSNumber numberWithBool:]`.
/// Returns an autoreleased object.
///
/// # Safety
///
/// Must be called from a thread with an active autorelease pool (or caller
/// must arrange to retain/release).
#[allow(dead_code)]
pub(crate) unsafe fn create_nsnumber_bool(val: bool) -> *mut c_void {
    // SAFETY: objc_getClass with a valid null-terminated class name.
    let cls = unsafe { objc_getClass(sel!("NSNumber")) };
    if cls.is_null() {
        return std::ptr::null_mut();
    }

    type NumFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i8) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("numberWithBool:")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: NumFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: cls is a valid class; val is converted to ObjC BOOL (i8).
    unsafe { f(cls, sel, val as i8) }
}

// ---------------------------------------------------------------------------
// NSString — read back
// ---------------------------------------------------------------------------

/// Read an NSString object pointer into a Rust String.
/// Returns `None` if the pointer is null or the string is not valid UTF-8.
///
/// # Safety
///
/// `nsstring` must be a valid `NSString` pointer or null.
#[allow(dead_code)]
pub(crate) unsafe fn nsstring_to_string(nsstring: *mut c_void) -> Option<String> {
    if nsstring.is_null() {
        return None;
    }

    type Utf8Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const u8;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let utf8_sel = unsafe { sel_registerName(sel!("UTF8String")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let utf8_fn: Utf8Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: nsstring is a valid NSString pointer.
    let cstr = unsafe { utf8_fn(nsstring, utf8_sel) };
    if cstr.is_null() {
        return None;
    }

    // SAFETY: cstr is a valid null-terminated UTF-8 string from NSString.
    unsafe { std::ffi::CStr::from_ptr(cstr as *const i8) }
        .to_str()
        .ok()
        .map(|s| s.to_string())
}

// ---------------------------------------------------------------------------
// NSNumber — read back
// ---------------------------------------------------------------------------

/// Read a `u64` from an `NSNumber` via `-[NSNumber unsignedLongLongValue]`.
///
/// # Safety
///
/// `nsnumber` must be a valid, non-null `NSNumber` pointer.
#[allow(dead_code)]
pub(crate) unsafe fn nsnumber_to_u64(nsnumber: *mut c_void) -> u64 {
    type ValFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("unsignedLongLongValue")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: ValFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: nsnumber is a valid NSNumber pointer.
    unsafe { f(nsnumber, sel) }
}

/// Read a `u32` from an `NSNumber` via `-[NSNumber unsignedIntValue]`.
///
/// # Safety
///
/// `nsnumber` must be a valid, non-null `NSNumber` pointer.
#[allow(dead_code)]
pub(crate) unsafe fn nsnumber_to_u32(nsnumber: *mut c_void) -> u32 {
    type ValFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("unsignedIntValue")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: ValFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: nsnumber is a valid NSNumber pointer.
    unsafe { f(nsnumber, sel) }
}

/// Read an `i64` from an `NSNumber` via `-[NSNumber longLongValue]`.
///
/// # Safety
///
/// `nsnumber` must be a valid, non-null `NSNumber` pointer.
#[allow(dead_code)]
pub(crate) unsafe fn nsnumber_to_i64(nsnumber: *mut c_void) -> i64 {
    type ValFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i64;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("longLongValue")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: ValFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: nsnumber is a valid NSNumber pointer.
    unsafe { f(nsnumber, sel) }
}

/// Read a `bool` from an `NSNumber` via `-[NSNumber boolValue]`.
///
/// # Safety
///
/// `nsnumber` must be a valid, non-null `NSNumber` pointer.
#[allow(dead_code)]
pub(crate) unsafe fn nsnumber_to_bool(nsnumber: *mut c_void) -> bool {
    type ValFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let sel = unsafe { sel_registerName(sel!("boolValue")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let f: ValFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: nsnumber is a valid NSNumber pointer.
    unsafe { f(nsnumber, sel) != 0 }
}
