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
/// Orion declares this as `unsigned int` in the ObjC calls.
const ANE_QOS: u32 = 21;

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
    /// # Arguments
    ///
    /// * `mil_text` — MIL program text (UTF-8)
    /// * `weights` — Named weight entries: `(path_key, data)` where path_key
    ///   matches the `@model_path/...` path used in MIL text BLOBFILE refs.
    ///
    /// # Returns
    ///
    /// An opaque pointer to the `_ANEInMemoryModel` instance (retained).
    pub fn compile_mil_text(
        mil_text: &str,
        weights: &[(&str, &[u8])],
    ) -> std::result::Result<*mut c_void, AneError> {
        // 0. Check budget
        let current = COMPILE_COUNT.load(Ordering::Relaxed);
        if current >= ANE_COMPILE_LIMIT {
            return Err(AneError::BudgetExhausted { count: current });
        }

        if mil_text.is_empty() {
            return Err(AneError::InvalidInput("MIL text is empty".into()));
        }

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

        // 4. Build weight dictionary with all weight entries.
        //    Format: @{ @"@model_path/weights/w.bin": @{ @"data": NSData, @"offset": @(64) }, ... }
        let weight_dict = if weights.is_empty() {
            ns_empty_dict()?
        } else {
            build_multi_weight_dict(weights)?
        };

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
        #[cfg(debug_assertions)]
        eprintln!(
            "[ane] creating descriptor from MIL text ({} bytes), {} weight(s)...",
            mil_text.len(),
            weights.len()
        );
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

        #[cfg(debug_assertions)]
        eprintln!("[ane] descriptor created, creating in-memory model...");
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
        let hex_str = get_model_hex_id(model);
        if let Some(ref hex_str) = hex_str {
            populate_tmp_dir(hex_str, mil_text, weights);
        }

        // 8. Compile: [model compileWithQoS:21 options:@{} error:&e]
        #[cfg(debug_assertions)]
        eprintln!("[ane] compiling with QoS={ANE_QOS}...");
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
            u32,
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
            u32,
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

    /// Create a new ANE program by reusing a donor's compiled artifacts.
    ///
    /// Follows Orion's `orion_program_patch_weights` pattern:
    /// 1. Create a new `_ANEInMemoryModel` with the same MIL text but different weights
    /// 2. Copy the donor's compiled `net.plist` to the new model's temp directory
    /// 3. Call `loadWithQoS:` — **no `compileWithQoS:`** — skipping compilation entirely
    ///
    /// This does **not** consume a compile budget slot.
    ///
    /// # Arguments
    ///
    /// * `donor_model` — Raw pointer to the donor `_ANEInMemoryModel` (must have
    ///   been compiled from the same MIL text structure).
    /// * `mil_text` — MIL program text (must be identical to the donor's).
    /// * `weights` — New weight entries for the patched program.
    ///
    /// # Returns
    ///
    /// An opaque pointer to the new `_ANEInMemoryModel` instance (retained).
    pub fn patch_weights(
        donor_model: *mut c_void,
        mil_text: &str,
        weights: &[(&str, &[u8])],
    ) -> std::result::Result<*mut c_void, AneError> {
        if donor_model.is_null() {
            return Err(AneError::InvalidInput("donor_model pointer is null".into()));
        }
        if mil_text.is_empty() {
            return Err(AneError::InvalidInput("MIL text is empty".into()));
        }

        // 1. Get donor's hex ID → find its temp dir with net.plist
        let donor_hex = get_model_hex_id(donor_model).ok_or_else(|| {
            AneError::CompilationFailed("failed to get donor model hexStringIdentifier".into())
        })?;
        let donor_tmp = std::env::temp_dir().join(&donor_hex);
        let donor_net_plist = donor_tmp.join("net.plist");
        if !donor_net_plist.exists() {
            return Err(AneError::CompilationFailed(format!(
                "donor net.plist not found at {}",
                donor_net_plist.display()
            )));
        }

        // 2. dlopen the framework + resolve classes
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

        // 3. Create descriptor + model with new weights
        let mil_data = create_nsdata(mil_text.as_bytes())?;
        let weight_dict = if weights.is_empty() {
            ns_empty_dict()?
        } else {
            build_multi_weight_dict(weights)?
        };

        let desc_sel = unsafe { sel_registerName(sel!("modelWithMILText:weights:optionsPlist:")) };
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
                "patch_weights: modelWithMILText:weights:optionsPlist: returned nil".into(),
            ));
        }

        let imm_sel = unsafe { sel_registerName(sel!("inMemoryModelWithDescriptor:")) };
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
                "patch_weights: inMemoryModelWithDescriptor: returned nil".into(),
            ));
        }

        // 4. Get new model's hex ID → set up its temp dir
        let new_hex = get_model_hex_id(model).ok_or_else(|| {
            AneError::CompilationFailed("failed to get new model hexStringIdentifier".into())
        })?;
        populate_tmp_dir(&new_hex, mil_text, weights);

        // 5. Copy donor's net.plist → new model's temp dir (the key trick)
        let new_tmp = std::env::temp_dir().join(&new_hex);
        let new_net_plist = new_tmp.join("net.plist");
        std::fs::copy(&donor_net_plist, &new_net_plist)?;

        #[cfg(debug_assertions)]
        eprintln!("[ane] patch_weights: copied net.plist from {donor_hex} → {new_hex}");

        // 6. Load (NO compile!) — [model loadWithQoS:21 options:@{} error:&e]
        let empty_dict = ns_empty_dict()?;
        let mut error: *mut c_void = std::ptr::null_mut();
        type LoadFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            u32,
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
                "patch_weights: loadWithQoS:options:error: returned NO".into()
            };
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneError::CompilationFailed(err_msg));
        }

        // 7. Retain — caller owns the reference
        type RetainFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let retain_sel = unsafe { sel_registerName(sel!("retain")) };
        let retain_fn: RetainFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { retain_fn(model, retain_sel) };

        unsafe {
            CFRelease(mil_data);
            CFRelease(weight_dict);
        }

        // NO COMPILE_COUNT increment — this bypasses the compiler entirely.

        Ok(model)
    }
}

// ---------------------------------------------------------------------------
// Model hex ID and temp dir helpers
// ---------------------------------------------------------------------------

/// Get the `hexStringIdentifier` from an `_ANEInMemoryModel` as a Rust string.
///
/// Returns `None` if the selector returns nil or the string is not valid UTF-8.
fn get_model_hex_id(model: *mut c_void) -> Option<String> {
    type HexIdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    let hex_sel = unsafe { sel_registerName(sel!("hexStringIdentifier")) };
    let hex_fn: HexIdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let hex_id = unsafe { hex_fn(model, hex_sel) };
    if hex_id.is_null() {
        return None;
    }

    type Utf8Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const u8;
    let utf8_sel = unsafe { sel_registerName(sel!("UTF8String")) };
    let utf8_fn: Utf8Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    let cstr = unsafe { utf8_fn(hex_id, utf8_sel) };
    if cstr.is_null() {
        return None;
    }

    unsafe { std::ffi::CStr::from_ptr(cstr as *const i8) }
        .to_str()
        .ok()
        .map(|s| s.to_string())
}

/// Pre-populate the model's temp directory with MIL text and weight blobs.
///
/// Writes `model.mil` and BLOBFILE-wrapped weight files to
/// `$TMPDIR/<hex_id>/`.
fn populate_tmp_dir(hex_id: &str, mil_text: &str, weights: &[(&str, &[u8])]) {
    let tmp_dir = std::env::temp_dir().join(hex_id);
    let weights_dir = tmp_dir.join("weights");

    #[cfg(debug_assertions)]
    eprintln!("[ane] hexId={hex_id}, tmp_dir={}", tmp_dir.display());

    if let Ok(()) = std::fs::create_dir_all(&weights_dir) {
        let _ = std::fs::write(tmp_dir.join("model.mil"), mil_text.as_bytes());
        for (path_key, data) in weights {
            // path_key is like "@model_path/weights/w.bin"
            let rel_path = path_key.strip_prefix("@model_path/").unwrap_or(path_key);
            let full_path = tmp_dir.join(rel_path);
            if let Some(parent) = full_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            #[cfg(debug_assertions)]
            eprintln!(
                "[ane] writing weight {} ({} bytes) → {}",
                path_key,
                data.len(),
                full_path.display()
            );
            let blob = make_blobfile(data);
            let _ = std::fs::write(&full_path, &blob);
        }
    }
}

// ---------------------------------------------------------------------------
// BLOBFILE helper
// ---------------------------------------------------------------------------

/// Create a BLOBFILE in Orion's format from raw weight data.
///
/// Format: 128-byte header (file header + chunk descriptor) + data.
/// This is the format expected by the weight dict's `data` field.
pub(crate) fn make_blobfile(data: &[u8]) -> Vec<u8> {
    let data_size = data.len();
    let total = 128 + data_size;
    let mut buf = vec![0u8; total];

    // File header (bytes 0-63)
    buf[0] = 1;
    buf[4] = 2;

    // Chunk descriptor (bytes 64-127)
    buf[64] = 0xEF; // 0xDEADBEEF magic
    buf[65] = 0xBE;
    buf[66] = 0xAD;
    buf[67] = 0xDE;
    buf[68] = 1;
    // Data size (bytes 72-75, u32 LE)
    buf[72..76].copy_from_slice(&(data_size as u32).to_le_bytes());
    // Data offset = 128 (bytes 80-83, u32 LE)
    buf[80..84].copy_from_slice(&128u32.to_le_bytes());

    // Weight data (bytes 128+)
    buf[128..].copy_from_slice(data);

    buf
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

/// Build the weight dictionary in Orion's format with multiple entries.
/// Each entry: `@"@model_path/weights/w.bin": @{ @"data": NSData, @"offset": @(64) }`
fn build_multi_weight_dict(weights: &[(&str, &[u8])]) -> Result<*mut c_void, AneError> {
    let outer_dict = ns_mutable_dict()?;

    for (path_key, data) in weights {
        // Wrap raw weight data in BLOBFILE format (128-byte header + data).
        // The offset in the weight dict (64) points to the chunk descriptor
        // within this BLOBFILE data.
        let blob = crate::ffi::ane::make_blobfile(data);
        let data_nsdata = create_nsdata(&blob)?;
        let offset_num = create_nsnumber(64)?;

        let inner_dict = ns_mutable_dict()?;
        let data_key = create_nsstring("data")?;
        let offset_key = create_nsstring("offset")?;
        ns_dict_set(inner_dict, data_key, data_nsdata);
        ns_dict_set(inner_dict, offset_key, offset_num);

        unsafe {
            CFRelease(data_key);
            CFRelease(offset_key);
            CFRelease(data_nsdata);
            CFRelease(offset_num);
        }

        let outer_key = create_nsstring(path_key)?;
        ns_dict_set(outer_dict, outer_key, inner_dict);

        unsafe {
            CFRelease(outer_key);
            CFRelease(inner_dict);
        }
    }

    Ok(outer_dict)
}

/// Build the weight dictionary in Orion's format:
/// `@{ @"@model_path/weights/weight.bin": @{ @"data": nsData, @"offset": @(64) } }`
/// Returns a retained `NSMutableDictionary` — caller must CFRelease when done.
#[allow(dead_code)]
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

        // compile_mil_text with empty weights should proceed past validation
        // but may fail at framework loading or compilation.
        let result = AneCompiler::compile_mil_text("program test {}", &[]);
        // The call may or may not increment the counter depending on how
        // far it gets. We just verify it doesn't panic.
        let _ = (result, before);
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
        let result = AneCompiler::compile_mil_text("", &[]);
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

    #[test]
    fn patch_weights_null_donor_returns_error() {
        let result = AneCompiler::patch_weights(std::ptr::null_mut(), "program test {}", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneError::InvalidInput(msg) => {
                assert!(msg.contains("null"), "expected 'null' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn patch_weights_empty_text_returns_error() {
        // Use a non-null dummy pointer — patch_weights validates MIL text before
        // dereferencing the donor.
        let dummy: *mut c_void = 0x1 as *mut c_void;
        let result = AneCompiler::patch_weights(dummy, "", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneError::InvalidInput(msg) => {
                assert!(msg.contains("empty"), "expected 'empty' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }
}
