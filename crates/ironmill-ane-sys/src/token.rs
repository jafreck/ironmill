//! Wrapper for `_ANEModelToken`.
//!
//! A model token identifies a model + process combination for the ANE daemon.
//! Tokens can be created from code-signing identities or audit tokens.

use std::ffi::c_void;

use crate::error::AneSysError;
use crate::objc::{
    CFRelease, create_nsstring, get_class, nsstring_to_string, objc_msgSend, objc_retain,
    safe_release, sel, sel_registerName,
};

// ---------------------------------------------------------------------------
// ModelToken — wraps _ANEModelToken
// ---------------------------------------------------------------------------

/// Safe wrapper around `_ANEModelToken`.
///
/// Owns a retained ObjC object; released on drop.
pub struct ModelToken {
    raw: *mut c_void,
}

// SAFETY: The raw handle is only accessed through &self methods.
unsafe impl Send for ModelToken {}

impl std::fmt::Debug for ModelToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelToken")
            .field("raw", &self.raw)
            .finish()
    }
}

impl ModelToken {
    /// Raw ObjC object pointer.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// Create a token from code-signing and team identities.
    ///
    /// Wraps `+[_ANEModelToken tokenWithCsIdentity:teamIdentity:modelIdentifier:processIdentifier:]`.
    pub fn with_cs_identity(
        cs_identity: &str,
        team_identity: &str,
        model_identifier: &str,
        process_identifier: i32,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEModelToken")?;

        let ns_cs = create_nsstring(cs_identity)?;
        let ns_team = create_nsstring(team_identity)?;
        let ns_model = create_nsstring(model_identifier)?;

        type FactoryFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            i32,
        ) -> *mut c_void;
        let sel = unsafe {
            sel_registerName(sel!(
                "tokenWithCsIdentity:teamIdentity:modelIdentifier:processIdentifier:"
            ))
        };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, ns_cs, ns_team, ns_model, process_identifier) };

        // Release the NSStrings now that the factory has consumed them.
        unsafe {
            CFRelease(ns_model);
            CFRelease(ns_team);
            CFRelease(ns_cs);
        }

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "tokenWithCsIdentity:teamIdentity:modelIdentifier:processIdentifier: returned nil".into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// Create a token from a Mach audit token.
    ///
    /// Wraps `+[_ANEModelToken tokenWithAuditToken:modelIdentifier:processIdentifier:]`.
    ///
    /// The audit token is a 32-byte (8 × u32) structure passed as raw bytes.
    pub fn with_audit_token(
        audit_token: &[u32; 8],
        model_identifier: &str,
        process_identifier: i32,
    ) -> Result<Self, AneSysError> {
        let cls = get_class("_ANEModelToken")?;
        let ns_model = create_nsstring(model_identifier)?;

        // The ObjC method expects the audit token as a by-value struct `{?=[8I]}`
        // (32 bytes on the stack). We pass a pointer to the array and let the
        // transmuted function signature handle the ABI.
        type FactoryFn = unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *const u32,
            *mut c_void,
            i32,
        ) -> *mut c_void;
        let sel = unsafe {
            sel_registerName(sel!(
                "tokenWithAuditToken:modelIdentifier:processIdentifier:"
            ))
        };
        let f: FactoryFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let obj = unsafe { f(cls, sel, audit_token.as_ptr(), ns_model, process_identifier) };

        unsafe { CFRelease(ns_model) };

        if obj.is_null() {
            return Err(AneSysError::NullPointer {
                context: "tokenWithAuditToken:modelIdentifier:processIdentifier: returned nil"
                    .into(),
            });
        }
        objc_retain(obj);
        Ok(Self { raw: obj })
    }

    /// The model identifier string.
    pub fn model_identifier(&self) -> Option<String> {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("modelIdentifier")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(self.raw, sel) };
        unsafe { nsstring_to_string(ns) }
    }

    /// The code-signing identity string.
    pub fn cs_identity(&self) -> Option<String> {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("csIdentity")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(self.raw, sel) };
        unsafe { nsstring_to_string(ns) }
    }

    /// The team identity string.
    pub fn team_identity(&self) -> Option<String> {
        type IdFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("teamIdentity")) };
        let f: IdFn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(self.raw, sel) };
        unsafe { nsstring_to_string(ns) }
    }

    /// The process identifier.
    pub fn process_identifier(&self) -> i32 {
        type I32Fn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
        let sel = unsafe { sel_registerName(sel!("processIdentifier")) };
        let f: I32Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(self.raw, sel) }
    }

    // -----------------------------------------------------------------------
    // Class methods that take audit tokens
    // -----------------------------------------------------------------------

    /// Look up the code-signing ID for an audit token and process identifier.
    ///
    /// Wraps `+[_ANEModelToken codeSigningIDFor:processIdentifier:]`.
    pub fn code_signing_id_for(
        audit_token: &[u32; 8],
        process_identifier: i32,
    ) -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEModelToken")?;
        type Fn = unsafe extern "C" fn(*mut c_void, *mut c_void, *const u32, i32) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("codeSigningIDFor:processIdentifier:")) };
        let f: Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel, audit_token.as_ptr(), process_identifier) };
        Ok(unsafe { nsstring_to_string(ns) })
    }

    /// Look up the team ID for an audit token and process identifier.
    ///
    /// Wraps `+[_ANEModelToken teamIDFor:processIdentifier:]`.
    pub fn team_id_for(
        audit_token: &[u32; 8],
        process_identifier: i32,
    ) -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEModelToken")?;
        type Fn = unsafe extern "C" fn(*mut c_void, *mut c_void, *const u32, i32) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("teamIDFor:processIdentifier:")) };
        let f: Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel, audit_token.as_ptr(), process_identifier) };
        Ok(unsafe { nsstring_to_string(ns) })
    }

    /// Look up the process name for an audit token and process identifier.
    ///
    /// Wraps `+[_ANEModelToken processNameFor:identifier:]`.
    pub fn process_name_for(
        audit_token: &[u32; 8],
        identifier: i32,
    ) -> Result<Option<String>, AneSysError> {
        let cls = get_class("_ANEModelToken")?;
        type Fn = unsafe extern "C" fn(*mut c_void, *mut c_void, *const u32, i32) -> *mut c_void;
        let sel = unsafe { sel_registerName(sel!("processNameFor:identifier:")) };
        let f: Fn = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        let ns = unsafe { f(cls, sel, audit_token.as_ptr(), identifier) };
        Ok(unsafe { nsstring_to_string(ns) })
    }
}

impl Drop for ModelToken {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            safe_release(self.raw);
            self.raw = std::ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_cs_identity_smoke() {
        // May fail if framework not loaded; should not panic.
        let _ = ModelToken::with_cs_identity("com.example.test", "TEAM123", "model1", 1);
    }

    #[test]
    fn class_method_smoke() {
        let token = [0u32; 8];
        let _ = ModelToken::code_signing_id_for(&token, 1);
        let _ = ModelToken::team_id_for(&token, 1);
        let _ = ModelToken::process_name_for(&token, 1);
    }
}
