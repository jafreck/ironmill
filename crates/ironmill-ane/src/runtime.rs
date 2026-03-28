//! ANE runtime — `_ANEClient` lifecycle and program execution.
//!
//! Wraps the private `_ANEClient` class for loading compiled programs
//! and executing inference on the Apple Neural Engine.
//!
//! `AneRuntime` is `Send` but **not** `Sync` — the underlying `_ANEClient`
//! is assumed thread-unsafe. Use one runtime per thread or wrap in a `Mutex`.

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
}

// ── AneRuntime ───────────────────────────────────────────────────

/// ANE runtime wrapping `_ANEClient`.
///
/// `AneRuntime` is `Send` but NOT `Sync` — the underlying `_ANEClient`
/// is assumed thread-unsafe. Use one runtime per thread or wrap in `Mutex`.
pub struct AneRuntime {
    client: *mut c_void,
}

// SAFETY: The client pointer is only accessed through &self / &mut self
// methods, so sending between threads is safe. Not Sync because the
// underlying _ANEClient is not thread-safe.
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
    /// Initialize the ANE client.
    ///
    /// Creates a new `_ANEClient` instance via ObjC FFI.
    pub fn new() -> Result<Self> {
        let cls = unsafe { objc_getClass(sel!("_ANEClient")) };
        if cls.is_null() {
            return Err(AneError::Other(anyhow::anyhow!(
                "_ANEClient class not found"
            )));
        }

        // [[_ANEClient alloc] init]
        type AllocFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let alloc_sel = unsafe { sel_registerName(sel!("alloc")) };
        let init_sel = unsafe { sel_registerName(sel!("init")) };
        let alloc: AllocFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };

        let instance = unsafe { alloc(cls, alloc_sel) };
        if instance.is_null() {
            return Err(AneError::Other(anyhow::anyhow!("_ANEClient alloc failed")));
        }
        let instance = unsafe { alloc(instance, init_sel) };
        if instance.is_null() {
            return Err(AneError::Other(anyhow::anyhow!("_ANEClient init failed")));
        }

        Ok(Self { client: instance })
    }

    /// Check if the ANE runtime is available on this system.
    pub fn is_available() -> bool {
        let cls = unsafe { objc_getClass(sel!("_ANEClient")) };
        !cls.is_null()
    }

    /// Load a compiled program for execution.
    pub fn load_program(&self, program: &CompiledProgram) -> Result<LoadedProgram> {
        // [client loadProgram:program]
        type LoadFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("loadProgram:")) };
        let load: LoadFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };

        let loaded = unsafe { load(self.client, sel, program.inner) };
        if loaded.is_null() {
            return Err(AneError::EvalFailed {
                status: 0,
                context: "loadProgram: returned null".into(),
            });
        }

        Ok(LoadedProgram { inner: loaded })
    }

    /// Execute a loaded program with input/output tensors.
    ///
    /// Inputs and outputs must be sorted alphabetically by their MIL
    /// variable names (constraints #3, #13).
    /// All inputs must have uniform allocation size (constraint #12).
    /// All outputs must have uniform allocation size (constraint #2).
    ///
    /// The weight dict is always empty `@{}` since weights are baked at
    /// compile time (constraint #11).
    pub fn eval(
        &self,
        program: &LoadedProgram,
        inputs: &[&AneTensor],
        outputs: &mut [&mut AneTensor],
    ) -> Result<()> {
        // Validate constraints.
        validate_nonempty(inputs, outputs)?;
        validate_uniform_alloc(inputs, "input")?;

        // Collect output refs for uniform alloc validation.
        let output_refs: Vec<&AneTensor> = outputs.iter().map(|t| &**t).collect();
        validate_uniform_alloc(&output_refs, "output")?;

        // Build NSMutableArray of IOSurface pointers for inputs.
        let input_array = unsafe { cf::ns_mutable_array() };
        for tensor in inputs {
            unsafe { cf::ns_array_add(input_array, tensor.as_ptr()) };
        }

        // Build NSMutableArray of IOSurface pointers for outputs.
        let output_array = unsafe { cf::ns_mutable_array() };
        for tensor in outputs.iter() {
            unsafe { cf::ns_array_add(output_array, tensor.as_ptr()) };
        }

        // Empty weight dictionary (constraint #11: weights baked at compile time).
        let weight_dict = unsafe { cf::ns_empty_dict() };

        // ANEProgramProcessRequestDirect(program, inputs, outputs, weights)
        type EvalFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> i64;
        let sel = unsafe {
            sel_registerName(sel!(
                "ANEProgramProcessRequestDirect:inputs:outputs:weights:"
            ))
        };
        let eval: EvalFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };

        let status = unsafe {
            eval(
                self.client,
                sel,
                program.inner,
                input_array,
                output_array,
                weight_dict,
            )
        };

        // Release the temporary ObjC collections.
        unsafe {
            cf::CFRelease(input_array);
            cf::CFRelease(output_array);
            cf::CFRelease(weight_dict);
        }

        if status != 0 {
            return Err(AneError::EvalFailed {
                status: status as u32,
                context: "ANEProgramProcessRequestDirect failed".into(),
            });
        }

        Ok(())
    }

    /// Unload a program, freeing ANE resources.
    pub fn unload_program(&self, program: LoadedProgram) {
        if !program.inner.is_null() {
            // [client unloadProgram:program]
            type UnloadFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
            let sel = unsafe { sel_registerName(sel!("unloadProgram:")) };
            let unload: UnloadFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
            unsafe { unload(self.client, sel, program.inner) };
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for AneRuntime {
    fn drop(&mut self) {
        if !self.client.is_null() {
            // [client release]
            unsafe {
                type ReleaseFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
                let release_sel = sel_registerName(sel!("release"));
                let release: ReleaseFn = std::mem::transmute(objc_msgSend as *const ());
                release(self.client, release_sel);
            }
        }
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
                    msg.contains("_ANEClient") || msg.contains("ANE runtime requires macOS"),
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
