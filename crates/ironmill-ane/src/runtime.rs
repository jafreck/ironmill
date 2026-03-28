//! ANE runtime — `_ANEInMemoryModel` lifecycle and program execution.
//!
//! Wraps the private ANE classes for loading compiled programs and
//! executing inference on the Apple Neural Engine, following Orion's
//! actual API (`_ANEInMemoryModel`, `_ANERequest`, `_ANEIOSurfaceObject`).
//!
//! `AneRuntime` is `Send` but **not** `Sync` — the underlying ANE objects
//! are assumed thread-unsafe. Use one runtime per thread or wrap in a `Mutex`.

use std::ffi::c_void;

use crate::program::{CompiledProgram, LoadedProgram};
use crate::tensor::AneTensor;
use crate::{AneError, Result};

// ── Objective-C FFI (macOS only) ─────────────────────────────────

#[cfg(target_os = "macos")]
#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    fn objc_getClass(name: *const u8) -> *mut c_void;
    fn sel_registerName(name: *const u8) -> *mut c_void;
    fn objc_msgSend(receiver: *mut c_void, sel: *mut c_void) -> *mut c_void;
}

#[cfg(target_os = "macos")]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(path: *const u8, mode: i32) -> *mut c_void;
}

#[cfg(target_os = "macos")]
const RTLD_NOW: i32 = 0x2;

/// QoS value used for compile/load/eval (matches Orion's constant of 21).
#[cfg(target_os = "macos")]
const ANE_QOS: i64 = 21;

/// Create a null-terminated byte pointer from a string literal.
macro_rules! sel {
    ($s:expr) => {
        concat!($s, "\0").as_ptr()
    };
}

// ── CoreFoundation / ObjC helpers (macOS only) ───────────────────

#[cfg(target_os = "macos")]
mod cf {
    use std::ffi::c_void;

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub fn CFRelease(cf: *mut c_void);
    }

    #[link(name = "objc", kind = "dylib")]
    unsafe extern "C" {
        fn objc_getClass(name: *const u8) -> *mut c_void;
        fn sel_registerName(name: *const u8) -> *mut c_void;
        fn objc_msgSend(receiver: *mut c_void, sel: *mut c_void) -> *mut c_void;
    }

    /// Create an empty `NSMutableArray`.
    pub unsafe fn ns_mutable_array() -> *mut c_void {
        unsafe {
            let cls = objc_getClass(sel!("NSMutableArray"));
            type NewFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
            let new_sel = sel_registerName(sel!("new"));
            let f: NewFn = std::mem::transmute(objc_msgSend as *const ());
            f(cls, new_sel)
        }
    }

    /// Append an object to an `NSMutableArray`.
    pub unsafe fn ns_array_add(array: *mut c_void, object: *mut c_void) {
        unsafe {
            type AddFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
            let sel = sel_registerName(sel!("addObject:"));
            let f: AddFn = std::mem::transmute(objc_msgSend as *const ());
            f(array, sel, object);
        }
    }

    /// Create an empty `NSDictionary`.
    pub unsafe fn ns_empty_dict() -> *mut c_void {
        unsafe {
            let cls = objc_getClass(sel!("NSDictionary"));
            type NewFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
            let new_sel = sel_registerName(sel!("new"));
            let f: NewFn = std::mem::transmute(objc_msgSend as *const ());
            f(cls, new_sel)
        }
    }

    /// Create an `NSNumber` from an integer.
    pub unsafe fn ns_number(value: i64) -> *mut c_void {
        unsafe {
            let cls = objc_getClass(sel!("NSNumber"));
            type NumFn = unsafe extern "C" fn(*mut c_void, *mut c_void, i64) -> *mut c_void;
            let sel = sel_registerName(sel!("numberWithLongLong:"));
            let f: NumFn = std::mem::transmute(objc_msgSend as *const ());
            f(cls, sel, value)
        }
    }
}

// ── AneRuntime ───────────────────────────────────────────────────

/// ANE runtime wrapping the private ANE framework classes.
///
/// Resolves `_ANEIOSurfaceObject` and `_ANERequest` classes on init.
/// The `_ANEInMemoryModel` (compiled + loaded) is the program handle.
pub struct AneRuntime {
    /// `_ANEIOSurfaceObject` class pointer.
    aio_cls: *mut c_void,
    /// `_ANERequest` class pointer.
    req_cls: *mut c_void,
}

// SAFETY: The class pointers are only accessed through &self methods.
// Not Sync because the underlying ANE framework is not thread-safe.
unsafe impl Send for AneRuntime {}

// ── Validation helpers (platform-independent) ────────────────────

/// Validate that all tensors in a slice have the same allocation size.
fn validate_uniform_alloc(tensors: &[&AneTensor], role: &str) -> Result<()> {
    if tensors.len() > 1 {
        let size = tensors[0].alloc_size();
        if tensors.iter().any(|t| t.alloc_size() != size) {
            return Err(AneError::EvalFailed {
                status: 0,
                context: format!("all {role} tensors must have the same allocation size"),
            });
        }
    }
    Ok(())
}

/// Validate that the tensor slices are non-empty.
fn validate_nonempty(inputs: &[&AneTensor], outputs: &[&mut AneTensor]) -> Result<()> {
    if inputs.is_empty() {
        return Err(AneError::EvalFailed {
            status: 0,
            context: "at least one input tensor is required".into(),
        });
    }
    if outputs.is_empty() {
        return Err(AneError::EvalFailed {
            status: 0,
            context: "at least one output tensor is required".into(),
        });
    }
    Ok(())
}

// ── macOS implementation ─────────────────────────────────────────

#[cfg(target_os = "macos")]
impl AneRuntime {
    /// Initialize the ANE runtime.
    ///
    /// `dlopen`s the AppleNeuralEngine framework and resolves the
    /// `_ANEIOSurfaceObject` and `_ANERequest` classes.
    pub fn new() -> Result<Self> {
        // dlopen the framework
        let handle = unsafe {
            dlopen(
                sel!(
                    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
                ),
                RTLD_NOW,
            )
        };
        if handle.is_null() {
            return Err(AneError::Other(anyhow::anyhow!(
                "failed to dlopen AppleNeuralEngine.framework"
            )));
        }

        let aio_cls = unsafe { objc_getClass(sel!("_ANEIOSurfaceObject")) };
        if aio_cls.is_null() {
            return Err(AneError::Other(anyhow::anyhow!(
                "_ANEIOSurfaceObject class not found"
            )));
        }

        let req_cls = unsafe { objc_getClass(sel!("_ANERequest")) };
        if req_cls.is_null() {
            return Err(AneError::Other(anyhow::anyhow!(
                "_ANERequest class not found"
            )));
        }

        Ok(Self { aio_cls, req_cls })
    }

    /// Check if the ANE runtime is available on this system.
    pub fn is_available() -> bool {
        let handle = unsafe {
            dlopen(
                sel!(
                    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
                ),
                RTLD_NOW,
            )
        };
        if handle.is_null() {
            return false;
        }
        let cls = unsafe { objc_getClass(sel!("_ANEInMemoryModel")) };
        !cls.is_null()
    }

    /// Load a compiled program for execution.
    ///
    /// In the Orion-based API, the model is already compiled + loaded after
    /// `compile_mil_text`. This transfers the model handle from
    /// `CompiledProgram` to `LoadedProgram`.
    pub fn load_program(&self, program: &CompiledProgram) -> Result<LoadedProgram> {
        if program.model.is_null() {
            return Err(AneError::EvalFailed {
                status: 0,
                context: "CompiledProgram has null model handle".into(),
            });
        }

        // Retain the model for the LoadedProgram handle
        type RetainFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let retain_sel = unsafe { sel_registerName(sel!("retain")) };
        let retain_fn: RetainFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { retain_fn(program.model, retain_sel) };

        Ok(LoadedProgram {
            model: program.model,
        })
    }

    /// Execute a loaded program with input/output tensors.
    ///
    /// Follows Orion's eval flow:
    /// 1. Wrap IOSurfaces in `_ANEIOSurfaceObject`
    /// 2. Build index arrays
    /// 3. Build `_ANERequest`
    /// 4. Call `evaluateWithQoS:options:request:error:` on the model
    pub fn eval(
        &self,
        program: &LoadedProgram,
        inputs: &[&AneTensor],
        outputs: &mut [&mut AneTensor],
    ) -> Result<()> {
        validate_nonempty(inputs, outputs)?;
        validate_uniform_alloc(inputs, "input")?;

        let output_refs: Vec<&AneTensor> = outputs.iter().map(|t| &**t).collect();
        validate_uniform_alloc(&output_refs, "output")?;

        // Wrap inputs in _ANEIOSurfaceObject and build index arrays
        let in_arr = unsafe { cf::ns_mutable_array() };
        let in_idx = unsafe { cf::ns_mutable_array() };
        for (i, tensor) in inputs.iter().enumerate() {
            let wrapped = self.wrap_iosurface(tensor.as_ptr())?;
            unsafe {
                cf::ns_array_add(in_arr, wrapped);
                cf::ns_array_add(in_idx, cf::ns_number(i as i64));
            }
        }

        // Wrap outputs
        let out_arr = unsafe { cf::ns_mutable_array() };
        let out_idx = unsafe { cf::ns_mutable_array() };
        for (i, tensor) in outputs.iter().enumerate() {
            let wrapped = self.wrap_iosurface(tensor.as_ptr())?;
            unsafe {
                cf::ns_array_add(out_arr, wrapped);
                cf::ns_array_add(out_idx, cf::ns_number(i as i64));
            }
        }

        // Build _ANERequest
        let zero = unsafe { cf::ns_number(0) };
        type RequestFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> *mut c_void;
        let req_sel = unsafe {
            sel_registerName(sel!(
                "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"
            ))
        };
        let req_fn: RequestFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let request = unsafe {
            req_fn(
                self.req_cls,
                req_sel,
                in_arr,
                in_idx,
                out_arr,
                out_idx,
                std::ptr::null_mut(), // weightsBuffer: nil
                std::ptr::null_mut(), // perfStats: nil
                zero,                 // procedureIndex: @0
            )
        };

        if request.is_null() {
            unsafe {
                cf::CFRelease(in_arr);
                cf::CFRelease(in_idx);
                cf::CFRelease(out_arr);
                cf::CFRelease(out_idx);
            }
            return Err(AneError::EvalFailed {
                status: 0,
                context: "failed to create _ANERequest".into(),
            });
        }

        // Evaluate: [model evaluateWithQoS:21 options:@{} request:req error:&e]
        let empty_dict = unsafe { cf::ns_empty_dict() };
        let mut error: *mut c_void = std::ptr::null_mut();

        type EvalFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            i64,
            *mut c_void,
            *mut c_void,
            *mut *mut c_void,
        ) -> i8;
        let eval_sel = unsafe { sel_registerName(sel!("evaluateWithQoS:options:request:error:")) };
        let eval_fn: EvalFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ok = unsafe {
            eval_fn(
                program.model,
                eval_sel,
                ANE_QOS,
                empty_dict,
                request,
                &mut error,
            )
        };

        // Release temporary ObjC collections
        unsafe {
            cf::CFRelease(in_arr);
            cf::CFRelease(in_idx);
            cf::CFRelease(out_arr);
            cf::CFRelease(out_idx);
            cf::CFRelease(empty_dict);
        }

        if ok == 0 {
            let err_msg = if !error.is_null() {
                extract_nserror_description(error)
            } else {
                "evaluateWithQoS:options:request:error: returned NO".into()
            };
            return Err(AneError::EvalFailed {
                status: 1,
                context: err_msg,
            });
        }

        Ok(())
    }

    /// Unload a program, freeing ANE resources.
    ///
    /// Calls `[model unloadWithQoS:21 error:&e]` then releases the model.
    pub fn unload_program(&self, program: LoadedProgram) {
        if !program.model.is_null() {
            // [model unloadWithQoS:21 error:&e]
            let mut error: *mut c_void = std::ptr::null_mut();
            type UnloadFn =
                unsafe extern "C" fn(*mut c_void, *mut c_void, i64, *mut *mut c_void) -> i8;
            let sel = unsafe { sel_registerName(sel!("unloadWithQoS:error:")) };
            let unload_fn: UnloadFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
            unsafe { unload_fn(program.model, sel, ANE_QOS, &mut error) };

            // Release the model
            unsafe { cf::CFRelease(program.model) };
        }
    }

    /// Wrap an IOSurface pointer in `_ANEIOSurfaceObject`.
    fn wrap_iosurface(&self, surface: *mut c_void) -> Result<*mut c_void> {
        type WrapFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("objectWithIOSurface:")) };
        let f: WrapFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let wrapped = unsafe { f(self.aio_cls, sel, surface) };
        if wrapped.is_null() {
            return Err(AneError::EvalFailed {
                status: 0,
                context: "objectWithIOSurface: returned nil".into(),
            });
        }
        Ok(wrapped)
    }
}

/// Extract the `localizedDescription` string from an `NSError`.
#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
impl Drop for AneRuntime {
    fn drop(&mut self) {
        // Class pointers are not owned — no cleanup needed.
    }
}

// ── Non-macOS fallback ───────────────────────────────────────────

#[cfg(not(target_os = "macos"))]
impl AneRuntime {
    /// ANE is not available on non-macOS platforms.
    pub fn new() -> Result<Self> {
        Err(AneError::Other(anyhow::anyhow!(
            "ANE runtime requires macOS"
        )))
    }

    /// Always returns `false` on non-macOS.
    pub fn is_available() -> bool {
        false
    }

    /// Not available on non-macOS.
    pub fn load_program(&self, _program: &CompiledProgram) -> Result<LoadedProgram> {
        Err(AneError::Other(anyhow::anyhow!(
            "ANE runtime requires macOS"
        )))
    }

    /// Not available on non-macOS.
    pub fn eval(
        &self,
        _program: &LoadedProgram,
        _inputs: &[&AneTensor],
        _outputs: &mut [&mut AneTensor],
    ) -> Result<()> {
        Err(AneError::Other(anyhow::anyhow!(
            "ANE runtime requires macOS"
        )))
    }

    /// Not available on non-macOS.
    pub fn unload_program(&self, _program: LoadedProgram) {}
}

#[cfg(not(target_os = "macos"))]
impl Drop for AneRuntime {
    fn drop(&mut self) {}
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mil_rs::ir::ScalarType;

    #[test]
    fn runtime_is_available() {
        // Should not panic regardless of platform.
        let _ = AneRuntime::is_available();
    }

    #[test]
    fn runtime_new_on_non_ane() {
        // On CI / non-ANE systems, new() should return an error rather than
        // panicking. On real ANE hardware it may succeed — both are acceptable.
        let result = AneRuntime::new();
        match result {
            Ok(_) => {} // Running on a system with ANE — that's fine.
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("ANE")
                        || msg.contains("dlopen")
                        || msg.contains("class not found")
                        || msg.contains("ANE runtime requires macOS"),
                    "unexpected error: {msg}"
                );
            }
        }
    }

    #[test]
    fn eval_validates_uniform_input_size() {
        let t1 = AneTensor::new_with_min_alloc(4, 8, ScalarType::Float16, 49152).unwrap();
        let t2 = AneTensor::new_with_min_alloc(4, 8, ScalarType::Float16, 65536).unwrap();
        let refs: Vec<&AneTensor> = vec![&t1, &t2];

        let err = validate_uniform_alloc(&refs, "input").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("input tensors must have the same allocation size"),
            "{msg}"
        );
    }

    #[test]
    fn eval_validates_uniform_output_size() {
        let t1 = AneTensor::new_with_min_alloc(4, 8, ScalarType::Float16, 49152).unwrap();
        let t2 = AneTensor::new_with_min_alloc(4, 8, ScalarType::Float16, 65536).unwrap();
        let refs: Vec<&AneTensor> = vec![&t1, &t2];

        let err = validate_uniform_alloc(&refs, "output").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("output tensors must have the same allocation size"),
            "{msg}"
        );
    }

    #[test]
    fn eval_validates_nonempty_inputs() {
        let mut t_out = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        let inputs: &[&AneTensor] = &[];
        let outputs: &mut [&mut AneTensor] = &mut [&mut t_out];

        let err = validate_nonempty(inputs, outputs).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("at least one input"), "{msg}");
    }

    #[test]
    fn eval_validates_nonempty_outputs() {
        let t_in = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        let inputs: &[&AneTensor] = &[&t_in];
        let outputs: &mut [&mut AneTensor] = &mut [];

        let err = validate_nonempty(inputs, outputs).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("at least one output"), "{msg}");
    }

    #[test]
    fn uniform_alloc_passes_with_same_size() {
        let t1 = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        let t2 = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        let refs: Vec<&AneTensor> = vec![&t1, &t2];

        validate_uniform_alloc(&refs, "input").unwrap();
    }

    #[test]
    fn uniform_alloc_passes_with_single() {
        let t1 = AneTensor::new(4, 8, ScalarType::Float16).unwrap();
        let refs: Vec<&AneTensor> = vec![&t1];

        validate_uniform_alloc(&refs, "input").unwrap();
    }
}
