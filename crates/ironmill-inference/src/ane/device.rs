//! `AneDevice` trait and `HardwareAneDevice` implementation.
//!
//! Defines the hardware dispatch contract for the Apple Neural Engine and
//! provides `HardwareAneDevice` which wraps [`ironmill_ane_sys::model`]
//! behind the trait. Consumers can swap in test doubles without touching
//! real hardware.

use std::sync::Mutex;

use crate::ane::{AneError, Result};
use ironmill_iosurface::AneTensor;

// ── AneDevice trait ──────────────────────────────────────────────

/// Hardware dispatch contract for the Apple Neural Engine.
///
/// Abstracts compile → eval lifecycle so that tests and benchmarks can
/// substitute a mock or recording device.
pub trait AneDevice: Send + Sync {
    /// An opaque compiled program handle. Drop releases hardware resources.
    type Program: Send;

    /// Compile MIL text (with optional BLOBFILE weights) into a program.
    fn compile(&self, mil_text: &str, weights: &[(&str, &[u8])], qos: u32)
    -> Result<Self::Program>;

    /// Compile using a donor program's net.plist (weight patching).
    /// Avoids consuming a compile budget slot.
    fn compile_patched(
        &self,
        donor: &Self::Program,
        mil_text: &str,
        weights: &[(&str, &[u8])],
        qos: u32,
    ) -> Result<Self::Program>;

    /// Evaluate a compiled program with pre-allocated I/O tensors.
    fn eval(
        &self,
        program: &Self::Program,
        inputs: &[&AneTensor],
        outputs: &mut [&mut AneTensor],
        qos: u32,
    ) -> Result<()>;

    /// Number of programs compiled so far.
    fn compile_count(&self) -> usize;

    /// Remaining compile budget before ANE refuses.
    fn remaining_budget(&self) -> usize;
}

// ── HardwareProgram ──────────────────────────────────────────────

/// A program compiled and loaded on real ANE hardware.
///
/// Wraps an [`ironmill_ane_sys::InMemoryModel`] which owns the retained
/// `_ANEInMemoryModel` handle.
pub struct HardwareProgram {
    pub(crate) model: ironmill_ane_sys::InMemoryModel,
}

// ── HardwareAneDevice ────────────────────────────────────────────

/// Real ANE device backed by [`ironmill_ane_sys::model`].
///
/// Eval calls are serialised through a `Mutex` because the underlying
/// Objective-C classes are not thread-safe.
pub struct HardwareAneDevice {
    /// Serialises ANE eval calls.
    eval_lock: Mutex<()>,
}

/// Convert an [`ironmill_ane_sys::AneSysError`] into an [`AneError`].
fn map_sys_err(e: ironmill_ane_sys::AneSysError) -> AneError {
    use ironmill_ane_sys::AneSysError;
    match e {
        AneSysError::CompilationFailed(msg) => AneError::CompileFailed {
            status: 0,
            context: msg,
        },
        AneSysError::EvalFailed { status, context } => AneError::EvalFailed { status, context },
        AneSysError::BudgetExhausted { count } => AneError::BudgetExhausted {
            used: count,
            limit: 119,
        },
        other => AneError::Other(anyhow::anyhow!("{other}")),
    }
}

// ── macOS implementation ─────────────────────────────────────────

#[cfg(target_os = "macos")]
impl HardwareAneDevice {
    /// Create a new `HardwareAneDevice`.
    ///
    /// Checks that the ANE framework is available; returns an error if not.
    pub fn new() -> Result<Self> {
        if !ironmill_ane_sys::model::is_available() {
            return Err(AneError::Other(anyhow::anyhow!(
                "ANE is not available on this system"
            )));
        }
        Ok(Self {
            eval_lock: Mutex::new(()),
        })
    }
}

#[cfg(target_os = "macos")]
impl AneDevice for HardwareAneDevice {
    type Program = HardwareProgram;

    fn compile(
        &self,
        mil_text: &str,
        weights: &[(&str, &[u8])],
        qos: u32,
    ) -> Result<HardwareProgram> {
        let model = ironmill_ane_sys::model::compile_mil_text(mil_text, weights, qos)
            .map_err(map_sys_err)?;
        Ok(HardwareProgram { model })
    }

    fn compile_patched(
        &self,
        donor: &HardwareProgram,
        mil_text: &str,
        weights: &[(&str, &[u8])],
        qos: u32,
    ) -> Result<HardwareProgram> {
        let model = ironmill_ane_sys::model::patch_weights(&donor.model, mil_text, weights, qos)
            .map_err(map_sys_err)?;
        Ok(HardwareProgram { model })
    }

    fn eval(
        &self,
        program: &HardwareProgram,
        inputs: &[&AneTensor],
        outputs: &mut [&mut AneTensor],
        qos: u32,
    ) -> Result<()> {
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

        let input_ptrs: Vec<*mut std::ffi::c_void> = inputs.iter().map(|t| t.as_ptr()).collect();
        let output_ptrs: Vec<*mut std::ffi::c_void> = outputs.iter().map(|t| t.as_ptr()).collect();

        let _guard = self.eval_lock.lock().expect("ANE eval lock poisoned");
        ironmill_ane_sys::model::eval(&program.model, &input_ptrs, &output_ptrs, qos)
            .map_err(map_sys_err)
    }

    fn compile_count(&self) -> usize {
        ironmill_ane_sys::model::compile_count()
    }

    fn remaining_budget(&self) -> usize {
        ironmill_ane_sys::model::remaining_budget()
    }
}

// ── Non-macOS fallback ───────────────────────────────────────────

#[cfg(not(target_os = "macos"))]
impl HardwareAneDevice {
    /// ANE is not available on non-macOS platforms.
    pub fn new() -> Result<Self> {
        Err(AneError::Other(anyhow::anyhow!(
            "ANE runtime requires macOS"
        )))
    }
}

#[cfg(not(target_os = "macos"))]
impl AneDevice for HardwareAneDevice {
    type Program = HardwareProgram;

    fn compile(
        &self,
        _mil_text: &str,
        _weights: &[(&str, &[u8])],
        _qos: u32,
    ) -> Result<HardwareProgram> {
        Err(AneError::Other(anyhow::anyhow!(
            "ANE runtime requires macOS"
        )))
    }

    fn compile_patched(
        &self,
        _donor: &HardwareProgram,
        _mil_text: &str,
        _weights: &[(&str, &[u8])],
        _qos: u32,
    ) -> Result<HardwareProgram> {
        Err(AneError::Other(anyhow::anyhow!(
            "ANE runtime requires macOS"
        )))
    }

    fn eval(
        &self,
        _program: &HardwareProgram,
        _inputs: &[&AneTensor],
        _outputs: &mut [&mut AneTensor],
        _qos: u32,
    ) -> Result<()> {
        Err(AneError::Other(anyhow::anyhow!(
            "ANE runtime requires macOS"
        )))
    }

    fn compile_count(&self) -> usize {
        0
    }

    fn remaining_budget(&self) -> usize {
        0
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_device_new() {
        // On CI / non-ANE systems this returns an error; on real hardware
        // it may succeed. Both outcomes are acceptable.
        let result = HardwareAneDevice::new();
        match result {
            Ok(_) => {}
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("ANE")
                        || msg.contains("dlopen")
                        || msg.contains("class not found")
                        || msg.contains("requires macOS"),
                    "unexpected error: {msg}"
                );
            }
        }
    }

    #[test]
    fn trait_is_object_safe() {
        // Verifies the trait can be used as `dyn AneDevice<Program = HardwareProgram>`.
        fn _assert_object_safe(_d: &dyn AneDevice<Program = HardwareProgram>) {}
    }
}
