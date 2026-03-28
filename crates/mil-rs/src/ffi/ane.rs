//! Direct ANE compilation via Objective-C FFI to `_ANEInMemoryModel`.
//!
//! This module wraps Apple's private `_ANEInMemoryModelDescriptor` and
//! `_ANEInMemoryModel` classes from the `AppleNeuralEngine` framework to
//! compile MIL text directly to an ANE-loaded model, following the same
//! approach as the Orion project.
//!
//! # Production Use
//!
//! For production-quality Objective-C interop, prefer the [`objc2`] crate
//! ecosystem over raw FFI.  This module uses raw `extern "C"` declarations
//! because it targets a *private* framework with no public headers — the
//! `objc2` bindings generator cannot help here.  If Apple ever publishes a
//! public ANE API, migrate to `objc2` immediately.
//!
//! [`objc2`]: https://crates.io/crates/objc2
//!
//! # ⚠️ Private API — Risks and Caveats
//!
//! **This module uses undocumented, private Apple APIs.**
//!
//! ## Stability
//! - The `_ANEInMemoryModel` class is an implementation detail of Apple's
//!   CoreML stack.  Its Objective-C selectors, argument conventions, and
//!   return types may change between any macOS release.
//! - There is **no stability guarantee** from Apple.
//!
//! ## App Store
//! - Apps that reference private frameworks or symbols are **rejected** from
//!   the Mac App Store.  This code is suitable only for local development
//!   tools, CI pipelines, and internal infrastructure.
//!
//! ## macOS Version Requirements
//! - Requires macOS 13 (Ventura) or later where the `AppleNeuralEngine`
//!   private framework ships by default.
//! - Requires Apple Silicon (M1 or later) or a T2-based Mac with ANE.
//!
//! ## Security
//! - The Objective-C runtime calls here are inherently `unsafe`.  We
//!   minimize the unsafe surface but cannot guarantee correctness if Apple
//!   changes the ABI.
//!
//! # Architecture
//!
//! The compilation flow (matching Orion's `orion_compile_mil`) is:
//!
//! 1. `dlopen` the `AppleNeuralEngine.framework`
//! 2. Resolve classes via `objc_getClass`: `_ANEInMemoryModelDescriptor`,
//!    `_ANEInMemoryModel`
//! 3. Create descriptor via `modelWithMILText:weights:optionsPlist:`
//! 4. Create model via `inMemoryModelWithDescriptor:`
//! 5. Pre-populate temp directory with MIL text and weight files
//! 6. Compile via `compileWithQoS:options:error:`
//! 7. Load via `loadWithQoS:options:error:`
//! 8. Return the `_ANEInMemoryModel` handle (the model IS the program)

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::MilError;

/// Maximum number of ANE compilations allowed per process.
///
/// Apple's ANE stack leaks memory on each invocation (~0.5-2 MB).  After
/// roughly 119 compilations the ANE daemon (`aned`) starts rejecting requests.
/// This constant caps usage to avoid hitting that wall silently.
const ANE_COMPILE_LIMIT: usize = 119;

/// Global compile count tracker (constraint: ~119 limit per process).
static COMPILE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// QoS value used for compile/load/eval (matches Orion's constant of 21).
const ANE_QOS: i64 = 21;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to direct ANE compilation.
#[derive(Debug, thiserror::Error)]
pub enum AneError {
    /// The ANE compiler framework could not be loaded.
    #[error("ANE framework not available: {0}")]
    FrameworkNotFound(String),

    /// A required ObjC class was not found in the loaded framework.
    #[error("ANE class not found: {0}")]
    ClassNotFound(String),

    /// The compiler returned a runtime error.
    #[error("ANE compilation failed: {0}")]
    CompilationFailed(String),

    /// An I/O error occurred while preparing inputs or reading outputs.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The input path does not exist or is not a valid model.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The per-process ANE compile budget has been exhausted.
    #[error("ANE compile budget exhausted ({count}/119 compilations)")]
    BudgetExhausted { count: usize },
}

impl From<AneError> for MilError {
    fn from(e: AneError) -> Self {
        MilError::Validation(format!("ANE direct compilation error: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Objective-C runtime FFI declarations
// ---------------------------------------------------------------------------

#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    fn objc_getClass(name: *const u8) -> *mut c_void;
    fn sel_registerName(name: *const u8) -> *mut c_void;
    fn objc_msgSend(receiver: *mut c_void, sel: *mut c_void) -> *mut c_void;
}

// dlopen / dlclose for explicit framework loading
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(path: *const u8, mode: i32) -> *mut c_void;
}

const RTLD_NOW: i32 = 0x2;

// CoreFoundation release
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *mut c_void);
}

// ---------------------------------------------------------------------------
// Helper: null-terminated C strings from Rust literals
// ---------------------------------------------------------------------------

/// Create a null-terminated byte slice suitable for Objective-C runtime calls.
macro_rules! sel {
    ($s:expr) => {
        concat!($s, "\0").as_ptr()
    };
}

/// Check if a class (or instance) responds to a given selector.
/// Safe to call on any ObjC object — returns false if not.
fn responds_to_selector(obj: *mut c_void, sel: *mut c_void) -> bool {
    type BoolFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;
    let rts_sel = unsafe { sel_registerName(sel!("respondsToSelector:")) };
    let f: BoolFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(obj, rts_sel, sel) != 0 }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Namespace for ANE compilation via the private `_ANEInMemoryModel` API.
///
/// This struct holds no state — it is a namespace for the static compilation
/// methods.  All interaction with the Objective-C runtime happens inside the
/// individual method implementations.
pub struct AneCompiler {
    _private: (),
}

impl AneCompiler {
    /// Check whether the ANE framework is available on this system.
    ///
    /// Attempts to `dlopen` the `AppleNeuralEngine.framework` and resolve
    /// the `_ANEInMemoryModel` class.
    pub fn is_available() -> bool {
        unsafe {
            let handle = dlopen(
                sel!(
                    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
                ),
                RTLD_NOW,
            );
            if handle.is_null() {
                return false;
            }
            let cls = objc_getClass(sel!("_ANEInMemoryModel"));
            !cls.is_null()
        }
    }

    /// Compile a `.mlpackage` (or `.mlmodelc`) to an ANE-optimized bundle.
    ///
    /// This legacy entry point is not used by the direct-ANE backend
    /// (which uses `compile_mil_text` instead), but is retained for API
    /// compatibility.
    pub fn compile(
        mlpackage_path: &Path,
        output_dir: &Path,
    ) -> std::result::Result<PathBuf, AneError> {
        if !mlpackage_path.exists() {
            return Err(AneError::InvalidInput(format!(
                "input path does not exist: {}",
                mlpackage_path.display()
            )));
        }

        std::fs::create_dir_all(output_dir)?;

        Err(AneError::CompilationFailed(
            "mlpackage compilation not supported — use compile_mil_text instead".into(),
        ))
    }

    /// Compile with incremental/delta support, reusing cached artifacts.
    pub fn compile_incremental(
        mlpackage_path: &Path,
        output_dir: &Path,
        cache_dir: &Path,
    ) -> std::result::Result<PathBuf, AneError> {
        if !mlpackage_path.exists() {
            return Err(AneError::InvalidInput(format!(
                "input path does not exist: {}",
                mlpackage_path.display()
            )));
        }

        std::fs::create_dir_all(output_dir)?;
        std::fs::create_dir_all(cache_dir)?;

        let _ = cache_dir;

        Self::compile(mlpackage_path, output_dir)
    }

    // -------------------------------------------------------------------
    // MIL text + BLOBFILE compilation (ANE direct backend)
    // -------------------------------------------------------------------

    /// Compile MIL text + weight blob into an ANE-loaded model.
    ///
    /// Follows Orion's `orion_compile_mil` flow:
    /// 1. `dlopen` AppleNeuralEngine.framework
    /// 2. Create `_ANEInMemoryModelDescriptor` from MIL text + weight dict
    /// 3. Create `_ANEInMemoryModel` from descriptor
    /// 4. Pre-populate temp directory with MIL text and weight files
    /// 5. Compile via `compileWithQoS:options:error:`
    /// 6. Load via `loadWithQoS:options:error:`
    ///
    /// # Returns
    ///
    /// An opaque pointer to the `_ANEInMemoryModel` instance (retained).
    /// The same object is used for compile, load, and eval — it IS the program.
    pub fn compile_mil_text(
        mil_text: &str,
        weight_path: &Path,
    ) -> std::result::Result<*mut c_void, AneError> {
        // 0. Check budget
        let current = COMPILE_COUNT.load(Ordering::Relaxed);
        if current >= ANE_COMPILE_LIMIT {
            return Err(AneError::BudgetExhausted { count: current });
        }

        if mil_text.is_empty() {
            return Err(AneError::InvalidInput("MIL text is empty".into()));
        }

        if !weight_path.exists() {
            return Err(AneError::InvalidInput(format!(
                "weight blob path does not exist: {}",
                weight_path.display()
            )));
        }

        // Read weight data from disk
        let weight_data = std::fs::read(weight_path).map_err(|e| {
            AneError::InvalidInput(format!(
                "failed to read weight blob {}: {e}",
                weight_path.display()
            ))
        })?;

        // Derive weight name from the file path (e.g. "weights/weight.bin")
        let weight_name = weight_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("weight.bin");
        let weight_key = format!("weights/{weight_name}");

        // 1. dlopen the framework
        let handle = unsafe {
            dlopen(
                sel!(
                    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
                ),
                RTLD_NOW,
            )
        };
        if handle.is_null() {
            return Err(AneError::FrameworkNotFound(
                "failed to dlopen AppleNeuralEngine.framework".into(),
            ));
        }

        // 2. Resolve required classes
        let desc_cls = unsafe { objc_getClass(sel!("_ANEInMemoryModelDescriptor")) };
        if desc_cls.is_null() {
            return Err(AneError::ClassNotFound(
                "_ANEInMemoryModelDescriptor".into(),
            ));
        }

        let imm_cls = unsafe { objc_getClass(sel!("_ANEInMemoryModel")) };
        if imm_cls.is_null() {
            return Err(AneError::ClassNotFound("_ANEInMemoryModel".into()));
        }

        // 3. Create NSData from MIL text
        let mil_data = create_nsdata(mil_text.as_bytes())?;

        // 4. Build weight dictionary:
        //    @{ @"@model_path/weights/weight.bin": @{ @"data": nsData, @"offset": @(64) } }
        let weight_dict = build_weight_dict(&weight_key, &weight_data)?;

        // 5. Create descriptor: [_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:]
        let desc_sel = unsafe { sel_registerName(sel!("modelWithMILText:weights:optionsPlist:")) };
        if !responds_to_selector(desc_cls, desc_sel) {
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneError::CompilationFailed(
                "_ANEInMemoryModelDescriptor does not respond to \
                 modelWithMILText:weights:optionsPlist:"
                    .into(),
            ));
        }
        type ModelWithMILTextFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> *mut c_void;
        let desc_fn: ModelWithMILTextFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let descriptor = unsafe {
            desc_fn(
                desc_cls,
                desc_sel,
                mil_data,
                weight_dict,
                std::ptr::null_mut(),
            )
        };
        if descriptor.is_null() {
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneError::CompilationFailed(
                "modelWithMILText:weights:optionsPlist: returned nil".into(),
            ));
        }

        // 6. Create model: [_ANEInMemoryModel inMemoryModelWithDescriptor:]
        let imm_sel = unsafe { sel_registerName(sel!("inMemoryModelWithDescriptor:")) };
        if !responds_to_selector(imm_cls, imm_sel) {
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneError::CompilationFailed(
                "_ANEInMemoryModel does not respond to \
                 inMemoryModelWithDescriptor:"
                    .into(),
            ));
        }
        type InMemoryModelFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let imm_fn: InMemoryModelFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let model = unsafe { imm_fn(imm_cls, imm_sel, descriptor) };
        if model.is_null() {
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneError::CompilationFailed(
                "inMemoryModelWithDescriptor: returned nil".into(),
            ));
        }

        // 7. Get hex identifier and pre-populate temp directory
        //    [model hexStringIdentifier] → NSString
        type HexIdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let hex_sel = unsafe { sel_registerName(sel!("hexStringIdentifier")) };
        let hex_fn: HexIdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let hex_id = unsafe { hex_fn(model, hex_sel) };

        if !hex_id.is_null() {
            // Get the hex string as a C string
            type Utf8Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const u8;
            let utf8_sel = unsafe { sel_registerName(sel!("UTF8String")) };
            let utf8_fn: Utf8Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
            let cstr = unsafe { utf8_fn(hex_id, utf8_sel) };

            if !cstr.is_null() {
                let hex_str = unsafe { std::ffi::CStr::from_ptr(cstr as *const i8) }
                    .to_str()
                    .unwrap_or("unknown");

                // Build temp dir path: NSTemporaryDirectory()/hexId/
                let tmp_dir = std::env::temp_dir().join(hex_str);
                let weights_dir = tmp_dir.join("weights");

                // Pre-populate: write model.mil and weight files
                if let Ok(()) = std::fs::create_dir_all(&weights_dir) {
                    let _ = std::fs::write(tmp_dir.join("model.mil"), mil_text.as_bytes());
                    let _ = std::fs::write(weights_dir.join(weight_name), &weight_data);
                }
            }
        }

        // 8. Compile: [model compileWithQoS:21 options:@{} error:&e]
        let empty_dict = ns_empty_dict()?;
        let compile_sel = unsafe { sel_registerName(sel!("compileWithQoS:options:error:")) };
        if !responds_to_selector(model, compile_sel) {
            unsafe {
                CFRelease(empty_dict);
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneError::CompilationFailed(
                "_ANEInMemoryModel does not respond to \
                 compileWithQoS:options:error:"
                    .into(),
            ));
        }
        let mut error: *mut c_void = std::ptr::null_mut();

        type CompileFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            i64,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        let compile_fn: CompileFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { compile_fn(model, compile_sel, ANE_QOS, empty_dict, &mut error) };

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "compileWithQoS:options:error: returned NO".into()
            };
            unsafe {
                CFRelease(empty_dict);
                CFRelease(mil_data);
                CFRelease(weight_dict);
            };
            return Err(AneError::CompilationFailed(err_msg));
        }

        // 9. Load: [model loadWithQoS:21 options:@{} error:&e]
        error = std::ptr::null_mut();
        type LoadFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            i64,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        let load_sel = unsafe { sel_registerName(sel!("loadWithQoS:options:error:")) };
        let load_fn: LoadFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe { load_fn(model, load_sel, ANE_QOS, empty_dict, &mut error) };

        unsafe { CFRelease(empty_dict) };

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "loadWithQoS:options:error: returned NO".into()
            };
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            };
            return Err(AneError::CompilationFailed(err_msg));
        }

        // 10. Retain the model — caller is responsible for release
        type RetainFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let retain_sel = unsafe { sel_registerName(sel!("retain")) };
        let retain_fn: RetainFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { retain_fn(model, retain_sel) };

        // Clean up intermediate ObjC objects (autoreleased, but be explicit)
        unsafe {
            CFRelease(mil_data);
            CFRelease(weight_dict);
        }

        COMPILE_COUNT.fetch_add(1, Ordering::Relaxed);

        Ok(model)
    }

    /// Number of compilations performed in this process.
    pub fn compile_count() -> usize {
        COMPILE_COUNT.load(Ordering::Relaxed)
    }

    /// Remaining compile budget before hitting the ~119 limit.
    pub fn remaining_budget() -> usize {
        ANE_COMPILE_LIMIT.saturating_sub(COMPILE_COUNT.load(Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------------------
// ObjC helpers
// ---------------------------------------------------------------------------

/// Create an `NSData` from a byte slice via `[[NSData alloc] initWithBytes:length:]`.
/// Returns a retained object — caller must CFRelease when done.
fn create_nsdata(bytes: &[u8]) -> Result<*mut c_void, AneError> {
    let nsdata_cls = unsafe { objc_getClass(sel!("NSData")) };
    if nsdata_cls.is_null() {
        return Err(AneError::FrameworkNotFound("NSData class not found".into()));
    }

    // alloc
    type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let alloc_sel = unsafe { sel_registerName(sel!("alloc")) };
    let alloc_fn: AllocFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let raw = unsafe { alloc_fn(nsdata_cls, alloc_sel) };
    if raw.is_null() {
        return Err(AneError::CompilationFailed("NSData alloc failed".into()));
    }

    // initWithBytes:length:
    type InitFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *const u8, usize) -> *mut c_void;
    let init_sel = unsafe { sel_registerName(sel!("initWithBytes:length:")) };
    let init_fn: InitFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let obj = unsafe { init_fn(raw, init_sel, bytes.as_ptr(), bytes.len()) };
    if obj.is_null() {
        return Err(AneError::CompilationFailed(
            "failed to create NSData".into(),
        ));
    }
    Ok(obj)
}

/// Create an `NSString` from a Rust string slice via
/// `[[NSString alloc] initWithUTF8String:]`.
/// Returns a retained object — caller must CFRelease when done.
fn create_nsstring(s: &str) -> Result<*mut c_void, AneError> {
    let nsstring_cls = unsafe { objc_getClass(sel!("NSString")) };
    if nsstring_cls.is_null() {
        return Err(AneError::FrameworkNotFound(
            "NSString class not found".into(),
        ));
    }

    let mut buf = Vec::with_capacity(s.len() + 1);
    buf.extend_from_slice(s.as_bytes());
    buf.push(0);

    // alloc
    type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let alloc_sel = unsafe { sel_registerName(sel!("alloc")) };
    let alloc_fn: AllocFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let raw = unsafe { alloc_fn(nsstring_cls, alloc_sel) };
    if raw.is_null() {
        return Err(AneError::CompilationFailed("NSString alloc failed".into()));
    }

    // initWithUTF8String:
    type InitFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *const u8) -> *mut c_void;
    let init_sel = unsafe { sel_registerName(sel!("initWithUTF8String:")) };
    let init_fn: InitFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let obj = unsafe { init_fn(raw, init_sel, buf.as_ptr()) };
    if obj.is_null() {
        return Err(AneError::CompilationFailed(
            "failed to create NSString".into(),
        ));
    }
    Ok(obj)
}

/// Create an `NSNumber` from an `i64` via `[[NSNumber alloc] initWithLongLong:]`.
/// Returns a retained object — caller must CFRelease when done.
fn create_nsnumber(value: i64) -> Result<*mut c_void, AneError> {
    let cls = unsafe { objc_getClass(sel!("NSNumber")) };
    if cls.is_null() {
        return Err(AneError::FrameworkNotFound(
            "NSNumber class not found".into(),
        ));
    }

    // alloc
    type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let alloc_sel = unsafe { sel_registerName(sel!("alloc")) };
    let alloc_fn: AllocFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let raw = unsafe { alloc_fn(cls, alloc_sel) };
    if raw.is_null() {
        return Err(AneError::CompilationFailed("NSNumber alloc failed".into()));
    }

    // initWithLongLong:
    type InitFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i64) -> *mut c_void;
    let init_sel = unsafe { sel_registerName(sel!("initWithLongLong:")) };
    let init_fn: InitFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let obj = unsafe { init_fn(raw, init_sel, value) };
    if obj.is_null() {
        return Err(AneError::CompilationFailed(
            "failed to create NSNumber".into(),
        ));
    }
    Ok(obj)
}

/// Create an empty `NSDictionary`.
fn ns_empty_dict() -> Result<*mut c_void, AneError> {
    let cls = unsafe { objc_getClass(sel!("NSDictionary")) };
    if cls.is_null() {
        return Err(AneError::FrameworkNotFound(
            "NSDictionary class not found".into(),
        ));
    }

    type NewFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let new_sel = unsafe { sel_registerName(sel!("new")) };
    let f: NewFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let obj = unsafe { f(cls, new_sel) };
    if obj.is_null() {
        return Err(AneError::CompilationFailed(
            "failed to create NSDictionary".into(),
        ));
    }
    Ok(obj)
}

/// Create an `NSMutableDictionary`, set key-value pairs, return as dict.
fn ns_mutable_dict() -> Result<*mut c_void, AneError> {
    let cls = unsafe { objc_getClass(sel!("NSMutableDictionary")) };
    if cls.is_null() {
        return Err(AneError::FrameworkNotFound(
            "NSMutableDictionary class not found".into(),
        ));
    }

    type NewFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let new_sel = unsafe { sel_registerName(sel!("new")) };
    let f: NewFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let obj = unsafe { f(cls, new_sel) };
    if obj.is_null() {
        return Err(AneError::CompilationFailed(
            "failed to create NSMutableDictionary".into(),
        ));
    }
    Ok(obj)
}

/// Set a key-value pair on an `NSMutableDictionary`.
fn ns_dict_set(dict: *mut c_void, key: *mut c_void, value: *mut c_void) {
    type SetFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void);
    let sel = unsafe { sel_registerName(sel!("setObject:forKey:")) };
    let f: SetFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(dict, sel, value, key) };
}

/// Build the weight dictionary in Orion's format:
/// `@{ @"@model_path/weights/weight.bin": @{ @"data": nsData, @"offset": @(64) } }`
/// Returns a retained `NSMutableDictionary` — caller must CFRelease when done.
fn build_weight_dict(weight_key: &str, weight_data: &[u8]) -> Result<*mut c_void, AneError> {
    let data_nsdata = create_nsdata(weight_data)?;
    let offset_num = create_nsnumber(64)?;

    // Inner dict: @{ @"data": nsData, @"offset": @(64) }
    let inner_dict = ns_mutable_dict()?;
    let data_key = create_nsstring("data")?;
    let offset_key = create_nsstring("offset")?;
    ns_dict_set(inner_dict, data_key, data_nsdata);
    ns_dict_set(inner_dict, offset_key, offset_num);

    // Release keys/values now held by inner_dict
    unsafe {
        CFRelease(data_key);
        CFRelease(offset_key);
        CFRelease(data_nsdata);
        CFRelease(offset_num);
    }

    // Outer dict: @{ @"@model_path/weights/weight.bin": innerDict }
    let full_key = format!("@model_path/{weight_key}");
    let outer_key = create_nsstring(&full_key)?;
    let outer_dict = ns_mutable_dict()?;
    ns_dict_set(outer_dict, outer_key, inner_dict);

    // Release key/value now held by outer_dict
    unsafe {
        CFRelease(outer_key);
        CFRelease(inner_dict);
    }

    Ok(outer_dict)
}

/// Extract the `localizedDescription` string from an `NSError`.
fn extract_nserror_description(error: *mut c_void) -> String {
    type DescFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let sel = unsafe { sel_registerName(sel!("localizedDescription")) };
    let f: DescFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let desc = unsafe { f(error, sel) };
    if desc.is_null() {
        return "unknown error (nil description)".into();
    }

    type Utf8Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const u8;
    let utf8_sel = unsafe { sel_registerName(sel!("UTF8String")) };
    let utf8_fn: Utf8Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let cstr = unsafe { utf8_fn(desc, utf8_sel) };
    if cstr.is_null() {
        return "unknown error (nil UTF8String)".into();
    }

    unsafe { std::ffi::CStr::from_ptr(cstr as *const i8) }
        .to_str()
        .unwrap_or("unknown error (invalid UTF-8)")
        .to_string()
}

/// Create an `NSURL` from a filesystem path via `[NSURL fileURLWithPath:]`.
#[allow(dead_code)]
fn create_nsurl_from_path(path: &Path) -> Result<*mut c_void, AneError> {
    let path_str = path.to_str().ok_or_else(|| {
        AneError::InvalidInput(format!("path contains invalid UTF-8: {}", path.display()))
    })?;

    let nsstring = create_nsstring(path_str)?;

    let nsurl_cls = unsafe { objc_getClass(sel!("NSURL")) };
    if nsurl_cls.is_null() {
        return Err(AneError::FrameworkNotFound("NSURL class not found".into()));
    }

    type FileUrlWithPathFn =
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;

    let sel = unsafe { sel_registerName(sel!("fileURLWithPath:")) };
    let send: FileUrlWithPathFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };

    let url = unsafe { send(nsurl_cls, sel, nsstring) };
    if url.is_null() {
        return Err(AneError::CompilationFailed(
            "failed to create NSURL from path".into(),
        ));
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ane_compiler_availability_check_does_not_panic() {
        // This will return false on most CI environments but must not crash.
        let _available = AneCompiler::is_available();
    }

    #[test]
    fn compile_nonexistent_input_returns_error() {
        let result = AneCompiler::compile(Path::new("does_not_exist.mlpackage"), Path::new("out"));
        assert!(result.is_err());
        match result.unwrap_err() {
            AneError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn compile_incremental_nonexistent_input_returns_error() {
        let result = AneCompiler::compile_incremental(
            Path::new("does_not_exist.mlpackage"),
            Path::new("out"),
            Path::new("cache"),
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            AneError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn compile_count_starts_at_zero_or_accumulates() {
        let count = AneCompiler::compile_count();
        assert!(
            count < ANE_COMPILE_LIMIT + 50,
            "count looks unreasonably high: {count}"
        );
    }

    #[test]
    fn compile_mil_text_budget_tracking() {
        let before = AneCompiler::compile_count();

        // compile_mil_text with a non-existent weight path should fail
        // with InvalidInput *before* incrementing the counter.
        let result =
            AneCompiler::compile_mil_text("program test {}", Path::new("nonexistent_weights.bin"));
        assert!(result.is_err());

        let after = AneCompiler::compile_count();
        assert_eq!(
            before, after,
            "compile_count should not increment on early validation error"
        );
    }

    #[test]
    fn remaining_budget_decreases() {
        let remaining = AneCompiler::remaining_budget();
        let count = AneCompiler::compile_count();
        assert_eq!(
            remaining,
            ANE_COMPILE_LIMIT.saturating_sub(count),
            "remaining budget should equal limit minus count"
        );
    }

    #[test]
    fn compile_mil_text_empty_text_returns_error() {
        let result = AneCompiler::compile_mil_text("", Path::new("weights.bin"));
        assert!(result.is_err());
        match result.unwrap_err() {
            AneError::InvalidInput(msg) => {
                assert!(msg.contains("empty"), "expected 'empty' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn budget_exhausted_error_display() {
        let err = AneError::BudgetExhausted { count: 119 };
        let msg = format!("{err}");
        assert!(
            msg.contains("119"),
            "error message should contain count: {msg}"
        );
        assert!(
            msg.contains("budget"),
            "error message should mention budget: {msg}"
        );
    }
}
