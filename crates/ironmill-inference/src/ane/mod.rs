//! ANE (Apple Neural Engine) direct inference backend.
//!
//! Provides the `AneInference` autoregressive decode loop and the safe
//! runtime wrapper around `ironmill_ane_sys::AneRuntime`.

pub mod decode;
#[allow(unsafe_code)]
pub mod runtime;
pub mod turboquant;

pub use decode::AneInference;
pub use runtime::AneRuntime;

// Re-export error/result from the parent module for local convenience.
pub(crate) use super::AneError;
pub(crate) type Result<T> = std::result::Result<T, AneError>;
