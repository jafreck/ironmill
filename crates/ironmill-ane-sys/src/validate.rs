//! Wrappers for ANE validation C functions.
//!
//! The ANE framework exports plain C functions (not ObjC methods) for
//! validating network descriptions. We look them up via `dlsym` on the
//! framework handle returned by [`crate::objc::ane_framework`].

use std::ffi::c_void;

use crate::error::AneSysError;

// ---------------------------------------------------------------------------
// dlsym — resolve symbols from a dlopen handle
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
}

/// Helper: look up a C symbol from the ANE framework.
unsafe fn lookup_symbol(name: &[u8]) -> Result<*mut c_void, AneSysError> {
    debug_assert!(
        name.last() == Some(&0),
        "symbol name must be null-terminated"
    );
    let handle = crate::objc::ane_framework()?;
    // SAFETY: `handle` is a valid dlopen handle; `name` is null-terminated.
    let sym = unsafe { dlsym(handle, name.as_ptr()) };
    if sym.is_null() {
        // Strip the trailing NUL for the error message.
        let pretty = std::str::from_utf8(&name[..name.len() - 1]).unwrap_or("<invalid>");
        return Err(AneSysError::FrameworkNotFound(format!(
            "{pretty} not found in ANE framework"
        )));
    }
    Ok(sym)
}

// ---------------------------------------------------------------------------
// ANEGetValidateNetworkSupportedVersion
// ---------------------------------------------------------------------------

/// Query the ANE framework for the supported validation network version.
///
/// Calls `ANEGetValidateNetworkSupportedVersion()` which returns a `u64`
/// version identifier.
pub fn get_validate_network_supported_version() -> Result<u64, AneSysError> {
    unsafe {
        let sym = lookup_symbol(b"ANEGetValidateNetworkSupportedVersion\0")?;
        let f: unsafe extern "C" fn() -> u64 = std::mem::transmute(sym);
        Ok(f())
    }
}

// ---------------------------------------------------------------------------
// ANEValidateMILNetworkOnHost / ANEValidateMLIRNetworkOnHost
//
// The exact signatures are undocumented. We expose raw function-pointer
// accessors so callers can transmute to the correct type once the ABI is
// reverse-engineered further.
// ---------------------------------------------------------------------------

/// Opaque function pointer to an ANE validation entry point.
///
/// The caller is responsible for transmuting this to the correct signature
/// and calling it safely.
pub type RawValidateFn = *mut c_void;

/// Obtain a raw pointer to `ANEValidateMILNetworkOnHost`.
///
/// # Safety
///
/// The returned pointer must be transmuted to the correct function signature
/// before being called. The signature is currently undocumented.
pub fn validate_mil_network_on_host_ptr() -> Result<RawValidateFn, AneSysError> {
    unsafe { lookup_symbol(b"ANEValidateMILNetworkOnHost\0") }
}

/// Obtain a raw pointer to `ANEValidateMLIRNetworkOnHost`.
///
/// # Safety
///
/// The returned pointer must be transmuted to the correct function signature
/// before being called. The signature is currently undocumented.
pub fn validate_mlir_network_on_host_ptr() -> Result<RawValidateFn, AneSysError> {
    unsafe { lookup_symbol(b"ANEValidateMLIRNetworkOnHost\0") }
}
