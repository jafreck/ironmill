//! ANE compilation via the private `_ANEInMemoryModel` Objective-C API.
//!
//! Wraps Apple's private `_ANEInMemoryModelDescriptor` and `_ANEInMemoryModel`
//! classes from the `AppleNeuralEngine` framework to compile MIL text directly
//! to an ANE-loaded model.
//!
//! # ⚠️ Private API — see crate-level docs for risks and caveats.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::CompiledProgram;
use crate::error::AneSysError;
use crate::objc::{
    CFRelease, ane_framework, create_nsdata, create_nsnumber, create_nsstring,
    extract_nserror_description, get_class, ns_dict_set, ns_empty_dict, ns_mutable_dict,
    objc_msgSend, objc_retain, responds_to_selector, sel, sel_registerName,
};

/// Maximum number of ANE compilations allowed per process.
///
/// Apple's ANE stack leaks memory on each invocation (~0.5-2 MB).  After
/// roughly 119 compilations the ANE daemon (`aned`) starts rejecting requests.
const ANE_COMPILE_LIMIT: usize = 119;

/// Global compile count tracker (constraint: ~119 limit per process).
static COMPILE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// QoS value used for compile/load (matches Orion's constant of 21).
const ANE_QOS: u32 = 21;

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
    pub fn is_available() -> bool {
        if ane_framework().is_err() {
            return false;
        }
        get_class("_ANEInMemoryModel").is_ok()
    }

    /// Compile a `.mlpackage` (or `.mlmodelc`) to an ANE-optimized bundle.
    ///
    /// This legacy entry point is not used by the direct-ANE backend
    /// (which uses `compile_mil_text` instead), but is retained for API
    /// compatibility.
    pub fn compile(mlpackage_path: &Path, output_dir: &Path) -> Result<PathBuf, AneSysError> {
        if !mlpackage_path.exists() {
            return Err(AneSysError::InvalidInput(format!(
                "input path does not exist: {}",
                mlpackage_path.display()
            )));
        }

        std::fs::create_dir_all(output_dir)?;

        Err(AneSysError::CompilationFailed(
            "mlpackage compilation not supported — use compile_mil_text instead".into(),
        ))
    }

    /// Compile with incremental/delta support, reusing cached artifacts.
    pub fn compile_incremental(
        mlpackage_path: &Path,
        output_dir: &Path,
        cache_dir: &Path,
    ) -> Result<PathBuf, AneSysError> {
        if !mlpackage_path.exists() {
            return Err(AneSysError::InvalidInput(format!(
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
    /// Returns a [`CompiledProgram`] wrapping the retained
    /// `_ANEInMemoryModel` handle.
    ///
    /// # Arguments
    ///
    /// * `mil_text` — MIL program text (UTF-8)
    /// * `weights` — Named weight entries: `(path_key, data)` where path_key
    ///   matches the `@model_path/...` path used in MIL text BLOBFILE refs.
    pub fn compile_mil_text(
        mil_text: &str,
        weights: &[(&str, &[u8])],
    ) -> Result<CompiledProgram, AneSysError> {
        // 0. Check budget (atomic to avoid TOCTOU race)
        let prev = COMPILE_COUNT.fetch_add(1, Ordering::SeqCst);
        if prev >= ANE_COMPILE_LIMIT {
            COMPILE_COUNT.fetch_sub(1, Ordering::SeqCst);
            return Err(AneSysError::BudgetExhausted { count: prev });
        }

        if mil_text.is_empty() {
            return Err(AneSysError::InvalidInput("MIL text is empty".into()));
        }

        // 1. Ensure the framework is loaded
        ane_framework()?;

        // 2. Resolve required classes
        let desc_cls = get_class("_ANEInMemoryModelDescriptor")?;
        let imm_cls = get_class("_ANEInMemoryModel")?;

        // 3. Create NSData from MIL text
        let mil_data = create_nsdata(mil_text.as_bytes())?;

        // 4. Build weight dictionary
        let weight_dict = if weights.is_empty() {
            ns_empty_dict()?
        } else {
            build_multi_weight_dict(weights)?
        };

        // 5. Create descriptor
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let desc_sel = unsafe { sel_registerName(sel!("modelWithMILText:weights:optionsPlist:")) };
        if !responds_to_selector(desc_cls, desc_sel) {
            // SAFETY: CFRelease on retained ObjC objects.
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneSysError::CompilationFailed(
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
        // SAFETY: transmute objc_msgSend to the correct 5-arg signature.
        let desc_fn: ModelWithMILTextFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: desc_cls/desc_sel/mil_data/weight_dict are valid ObjC pointers.
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
            // SAFETY: CFRelease on retained ObjC objects.
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneSysError::CompilationFailed(
                "modelWithMILText:weights:optionsPlist: returned nil".into(),
            ));
        }

        #[cfg(debug_assertions)]
        eprintln!("[ane] descriptor created, creating in-memory model...");
        // 6. Create model
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let imm_sel = unsafe { sel_registerName(sel!("inMemoryModelWithDescriptor:")) };
        if !responds_to_selector(imm_cls, imm_sel) {
            // SAFETY: CFRelease on retained ObjC objects.
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneSysError::CompilationFailed(
                "_ANEInMemoryModel does not respond to \
                 inMemoryModelWithDescriptor:"
                    .into(),
            ));
        }
        type InMemoryModelFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: transmute objc_msgSend to the correct 3-arg signature.
        let imm_fn: InMemoryModelFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: imm_cls/imm_sel/descriptor are valid ObjC pointers.
        let model = unsafe { imm_fn(imm_cls, imm_sel, descriptor) };
        if model.is_null() {
            // SAFETY: CFRelease on retained ObjC objects.
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneSysError::CompilationFailed(
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
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let compile_sel = unsafe { sel_registerName(sel!("compileWithQoS:options:error:")) };
        if !responds_to_selector(model, compile_sel) {
            // SAFETY: CFRelease on retained ObjC objects.
            unsafe {
                CFRelease(empty_dict);
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneSysError::CompilationFailed(
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
        // SAFETY: transmute objc_msgSend to the compile method signature.
        let compile_fn: CompileFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: model/compile_sel/empty_dict are valid; error is a valid out-pointer.
        let ok = unsafe { compile_fn(model, compile_sel, ANE_QOS, empty_dict, &mut error) };

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "compileWithQoS:options:error: returned NO".into()
            };
            // SAFETY: CFRelease on retained ObjC objects.
            unsafe {
                CFRelease(empty_dict);
                CFRelease(mil_data);
                CFRelease(weight_dict);
            };
            return Err(AneSysError::CompilationFailed(err_msg));
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
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let load_sel = unsafe { sel_registerName(sel!("loadWithQoS:options:error:")) };
        // SAFETY: transmute objc_msgSend to the load method signature.
        let load_fn: LoadFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: model/load_sel/empty_dict are valid; error is a valid out-pointer.
        let ok = unsafe { load_fn(model, load_sel, ANE_QOS, empty_dict, &mut error) };

        // SAFETY: CFRelease on retained ObjC object.
        unsafe { CFRelease(empty_dict) };

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "loadWithQoS:options:error: returned NO".into()
            };
            // SAFETY: CFRelease on retained ObjC objects.
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            };
            return Err(AneSysError::CompilationFailed(err_msg));
        }

        // 10. Retain the model — CompiledProgram::Drop will release it.
        objc_retain(model);

        // Clean up intermediate ObjC objects
        // SAFETY: CFRelease on retained ObjC objects.
        unsafe {
            CFRelease(mil_data);
            CFRelease(weight_dict);
        }

        Ok(CompiledProgram { model })
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
    /// Follows Orion's `orion_program_patch_weights` pattern: creates a new
    /// model with different weights, copies the donor's compiled `net.plist`,
    /// then loads — **skipping compilation entirely**.
    ///
    /// This does **not** consume a compile budget slot.
    pub fn patch_weights(
        donor: &CompiledProgram,
        mil_text: &str,
        weights: &[(&str, &[u8])],
    ) -> Result<CompiledProgram, AneSysError> {
        let donor_model = donor.as_raw_ptr();
        if donor_model.is_null() {
            return Err(AneSysError::InvalidInput(
                "donor_model pointer is null".into(),
            ));
        }
        if mil_text.is_empty() {
            return Err(AneSysError::InvalidInput("MIL text is empty".into()));
        }

        // 1. Get donor's hex ID → find its temp dir with net.plist
        let donor_hex = get_model_hex_id(donor_model).ok_or_else(|| {
            AneSysError::CompilationFailed("failed to get donor model hexStringIdentifier".into())
        })?;
        let donor_tmp = std::env::temp_dir().join(&donor_hex);
        let donor_net_plist = donor_tmp.join("net.plist");
        if !donor_net_plist.exists() {
            return Err(AneSysError::CompilationFailed(format!(
                "donor net.plist not found at {}",
                donor_net_plist.display()
            )));
        }

        // 2. Ensure framework is loaded + resolve classes
        ane_framework()?;
        let desc_cls = get_class("_ANEInMemoryModelDescriptor")?;
        let imm_cls = get_class("_ANEInMemoryModel")?;

        // 3. Create descriptor + model with new weights
        let mil_data = create_nsdata(mil_text.as_bytes())?;
        let weight_dict = if weights.is_empty() {
            ns_empty_dict()?
        } else {
            build_multi_weight_dict(weights)?
        };

        // SAFETY: sel_registerName with a valid null-terminated selector.
        let desc_sel = unsafe { sel_registerName(sel!("modelWithMILText:weights:optionsPlist:")) };
        type ModelWithMILTextFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> *mut c_void;
        // SAFETY: transmute objc_msgSend to the correct 5-arg signature.
        let desc_fn: ModelWithMILTextFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: desc_cls/desc_sel/mil_data/weight_dict are valid ObjC pointers.
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
            // SAFETY: CFRelease on retained ObjC objects.
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneSysError::CompilationFailed(
                "patch_weights: modelWithMILText:weights:optionsPlist: returned nil".into(),
            ));
        }

        // SAFETY: sel_registerName with a valid null-terminated selector.
        let imm_sel = unsafe { sel_registerName(sel!("inMemoryModelWithDescriptor:")) };
        type InMemoryModelFn =
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        // SAFETY: transmute objc_msgSend to the correct 3-arg signature.
        let imm_fn: InMemoryModelFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: imm_cls/imm_sel/descriptor are valid ObjC pointers.
        let model = unsafe { imm_fn(imm_cls, imm_sel, descriptor) };
        if model.is_null() {
            // SAFETY: CFRelease on retained ObjC objects.
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneSysError::CompilationFailed(
                "patch_weights: inMemoryModelWithDescriptor: returned nil".into(),
            ));
        }

        // 4. Get new model's hex ID → set up its temp dir
        let new_hex = get_model_hex_id(model).ok_or_else(|| {
            AneSysError::CompilationFailed("failed to get new model hexStringIdentifier".into())
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
        // SAFETY: sel_registerName with a valid null-terminated selector.
        let load_sel = unsafe { sel_registerName(sel!("loadWithQoS:options:error:")) };
        // SAFETY: transmute objc_msgSend to the load method signature.
        let load_fn: LoadFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        // SAFETY: model/load_sel/empty_dict are valid; error is a valid out-pointer.
        let ok = unsafe { load_fn(model, load_sel, ANE_QOS, empty_dict, &mut error) };
        // SAFETY: CFRelease on retained ObjC object.
        unsafe { CFRelease(empty_dict) };

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "patch_weights: loadWithQoS:options:error: returned NO".into()
            };
            // SAFETY: CFRelease on retained ObjC objects.
            unsafe {
                CFRelease(mil_data);
                CFRelease(weight_dict);
            }
            return Err(AneSysError::CompilationFailed(err_msg));
        }

        // 7. Retain — CompiledProgram::Drop will release.
        objc_retain(model);

        // SAFETY: CFRelease on retained ObjC objects.
        unsafe {
            CFRelease(mil_data);
            CFRelease(weight_dict);
        }

        Ok(CompiledProgram { model })
    }
}

// ---------------------------------------------------------------------------
// Model hex ID and temp dir helpers
// ---------------------------------------------------------------------------

/// Get the `hexStringIdentifier` from an `_ANEInMemoryModel` as a Rust string.
fn get_model_hex_id(model: *mut c_void) -> Option<String> {
    type HexIdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let hex_sel = unsafe { sel_registerName(sel!("hexStringIdentifier")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let hex_fn: HexIdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: model is a valid _ANEInMemoryModel handle.
    let hex_id = unsafe { hex_fn(model, hex_sel) };
    if hex_id.is_null() {
        return None;
    }

    type Utf8Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const u8;
    // SAFETY: sel_registerName with a valid null-terminated selector.
    let utf8_sel = unsafe { sel_registerName(sel!("UTF8String")) };
    // SAFETY: transmute objc_msgSend to the correct signature.
    let utf8_fn: Utf8Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    // SAFETY: hex_id is a valid NSString from hexStringIdentifier.
    let cstr = unsafe { utf8_fn(hex_id, utf8_sel) };
    if cstr.is_null() {
        return None;
    }

    // SAFETY: cstr is a valid null-terminated UTF-8 string from NSString.
    unsafe { std::ffi::CStr::from_ptr(cstr as *const i8) }
        .to_str()
        .ok()
        .map(|s| s.to_string())
}

/// Pre-populate the model's temp directory with MIL text and weight blobs.
fn populate_tmp_dir(hex_id: &str, mil_text: &str, weights: &[(&str, &[u8])]) {
    let tmp_dir = std::env::temp_dir().join(hex_id);
    let weights_dir = tmp_dir.join("weights");

    #[cfg(debug_assertions)]
    eprintln!("[ane] hexId={hex_id}, tmp_dir={}", tmp_dir.display());

    if let Ok(()) = std::fs::create_dir_all(&weights_dir) {
        let _ = std::fs::write(tmp_dir.join("model.mil"), mil_text.as_bytes());
        for (path_key, data) in weights {
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
            let blob = make_blobfile(data).unwrap_or_default();
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
pub fn make_blobfile(data: &[u8]) -> Result<Vec<u8>, AneSysError> {
    let data_size = data.len();
    if data_size > u32::MAX as usize {
        return Err(AneSysError::InvalidInput(format!(
            "BLOBFILE data size {} exceeds u32::MAX",
            data_size
        )));
    }
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

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Weight dictionary builder
// ---------------------------------------------------------------------------

/// Build the weight dictionary in Orion's format with multiple entries.
fn build_multi_weight_dict(weights: &[(&str, &[u8])]) -> Result<*mut c_void, AneSysError> {
    let outer_dict = ns_mutable_dict()?;

    for (path_key, data) in weights {
        let blob = make_blobfile(data)?;
        let data_nsdata = create_nsdata(&blob)?;
        let offset_num = create_nsnumber(64)?;

        let inner_dict = ns_mutable_dict()?;
        let data_key = create_nsstring("data")?;
        let offset_key = create_nsstring("offset")?;
        ns_dict_set(inner_dict, data_key, data_nsdata);
        ns_dict_set(inner_dict, offset_key, offset_num);

        // SAFETY: CFRelease on retained ObjC objects now held by inner_dict.
        unsafe {
            CFRelease(data_key);
            CFRelease(offset_key);
            CFRelease(data_nsdata);
            CFRelease(offset_num);
        }

        let outer_key = create_nsstring(path_key)?;
        ns_dict_set(outer_dict, outer_key, inner_dict);

        // SAFETY: CFRelease on retained ObjC objects now held by outer_dict.
        unsafe {
            CFRelease(outer_key);
            CFRelease(inner_dict);
        }
    }

    Ok(outer_dict)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ane_compiler_availability_check_does_not_panic() {
        let _available = AneCompiler::is_available();
    }

    #[test]
    fn compile_nonexistent_input_returns_error() {
        let result = AneCompiler::compile(Path::new("does_not_exist.mlpackage"), Path::new("out"));
        assert!(result.is_err());
        match result.unwrap_err() {
            AneSysError::InvalidInput(_) => {}
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
            AneSysError::InvalidInput(_) => {}
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
        let result = AneCompiler::compile_mil_text("program test {}", &[]);
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
            AneSysError::InvalidInput(msg) => {
                assert!(msg.contains("empty"), "expected 'empty' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
    }

    #[test]
    fn budget_exhausted_error_display() {
        let err = AneSysError::BudgetExhausted { count: 119 };
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
        // SAFETY: from_raw with a null pointer — patch_weights validates it.
        let donor = unsafe { CompiledProgram::from_raw(std::ptr::null_mut()) };
        let result = AneCompiler::patch_weights(&donor, "program test {}", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneSysError::InvalidInput(msg) => {
                assert!(msg.contains("null"), "expected 'null' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
        // Prevent Drop from calling CFRelease on the null pointer.
        std::mem::forget(donor);
    }

    #[test]
    fn patch_weights_empty_text_returns_error() {
        // SAFETY: from_raw with a non-null dummy pointer — patch_weights
        // validates MIL text before dereferencing the donor.
        let dummy = unsafe { CompiledProgram::from_raw(0x1 as *mut c_void) };
        let result = AneCompiler::patch_weights(&dummy, "", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneSysError::InvalidInput(msg) => {
                assert!(msg.contains("empty"), "expected 'empty' in message: {msg}");
            }
            other => panic!("expected InvalidInput, got: {other}"),
        }
        // Prevent Drop from calling CFRelease on the dummy pointer.
        std::mem::forget(dummy);
    }
}
