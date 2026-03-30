//! ANE runtime — `_ANEInMemoryModel` lifecycle and program execution.
//!
//! Wraps [`ironmill_ane_sys::AneRuntime`] with tensor-level validation
//! and conversion from [`AneTensor`] to raw IOSurface pointers.
//!
//! `AneRuntime` is `Send` but **not** `Sync` — the underlying ANE objects
//! are assumed thread-unsafe. Use one runtime per thread or wrap in a `Mutex`.

use std::ffi::c_void;

use crate::program::{CompiledProgram, LoadedProgram};
use crate::tensor::AneTensor;
use crate::{AneError, Result};

// ── AneRuntime ───────────────────────────────────────────────────

/// ANE runtime wrapping the private ANE framework classes.
///
/// Delegates to [`ironmill_ane_sys::AneRuntime`] for the low-level
/// Objective-C interop, adding tensor validation and IOSurface extraction.
pub struct AneRuntime {
    inner: ironmill_ane_sys::AneRuntime,
}

// SAFETY: The inner runtime is Send; we add no shared mutable state.
#[allow(unsafe_code)]
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
    pub fn new() -> Result<Self> {
        let inner = ironmill_ane_sys::AneRuntime::new()
            .map_err(|e| AneError::Other(anyhow::anyhow!("{e}")))?;
        Ok(Self { inner })
    }

    /// Check if the ANE runtime is available on this system.
    pub fn is_available() -> bool {
        ironmill_ane_sys::AneRuntime::is_available()
    }

    /// Load a compiled program for execution.
    pub fn load_program(&self, program: &CompiledProgram) -> Result<LoadedProgram> {
        self.inner
            .load_program(program)
            .map_err(|e| AneError::EvalFailed {
                status: 0,
                context: format!("{e}"),
            })
    }

    /// Execute a loaded program with input/output tensors.
    pub fn eval(
        &self,
        program: &LoadedProgram,
        inputs: &[&AneTensor],
        outputs: &mut [&mut AneTensor],
    ) -> Result<()> {
        self.eval_raw(program.as_raw_ptr(), inputs, outputs)
    }

    /// Core eval implementation that takes a raw model pointer.
    ///
    /// Both `eval` (for `LoadedProgram`) and `eval_compiled` (for
    /// `CompiledProgram`) delegate here.
    #[allow(unsafe_code)]
    pub(crate) fn eval_raw(
        &self,
        model: *mut c_void,
        inputs: &[&AneTensor],
        outputs: &mut [&mut AneTensor],
    ) -> Result<()> {
        validate_nonempty(inputs, outputs)?;
        validate_uniform_alloc(inputs, "input")?;

        let output_refs: Vec<&AneTensor> = outputs.iter().map(|t| &**t).collect();
        validate_uniform_alloc(&output_refs, "output")?;

        let input_surfaces: Vec<*mut c_void> = inputs.iter().map(|t| t.as_ptr()).collect();
        let output_surfaces: Vec<*mut c_void> = outputs.iter().map(|t| t.as_ptr()).collect();

        // SAFETY: model is a valid ANE model pointer, surfaces come from AneTensor.
        unsafe {
            self.inner
                .eval_raw(model, &input_surfaces, &output_surfaces)
        }
        .map_err(|e| AneError::EvalFailed {
            status: 1,
            context: format!("{e}"),
        })
    }

    /// Load a previously compiled (and unloaded) program into the ANE.
    pub fn load_compiled(&self, program: &CompiledProgram) -> Result<()> {
        self.inner
            .load_compiled(program)
            .map_err(|e| AneError::Other(anyhow::anyhow!("loadWithQoS failed: {e}")))
    }

    /// Unload a compiled program from the ANE, freeing the execution slot.
    pub fn unload_compiled(&self, program: &CompiledProgram) {
        self.inner.unload_compiled(program);
    }

    /// Load, evaluate, and unload a compiled program in one call.
    pub fn eval_compiled(
        &self,
        program: &CompiledProgram,
        inputs: &[&AneTensor],
        outputs: &mut [&mut AneTensor],
    ) -> Result<()> {
        self.load_compiled(program)?;
        let result = self.eval_raw(program.as_raw_ptr(), inputs, outputs);
        self.unload_compiled(program);
        result
    }

    /// Unload a program, freeing ANE resources.
    pub fn unload_program(&self, program: LoadedProgram) {
        self.inner.unload_program(program);
    }
}

#[cfg(target_os = "macos")]
impl Drop for AneRuntime {
    fn drop(&mut self) {
        // Inner runtime handles cleanup.
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

    /// Not available on non-macOS.
    pub fn load_compiled(&self, _program: &CompiledProgram) -> Result<()> {
        Err(AneError::Other(anyhow::anyhow!(
            "ANE runtime requires macOS"
        )))
    }

    /// Not available on non-macOS.
    pub fn unload_compiled(&self, _program: &CompiledProgram) {}

    /// Not available on non-macOS.
    pub fn eval_compiled(
        &self,
        _program: &CompiledProgram,
        _inputs: &[&AneTensor],
        _outputs: &mut [&mut AneTensor],
    ) -> Result<()> {
        Err(AneError::Other(anyhow::anyhow!(
            "ANE runtime requires macOS"
        )))
    }
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
