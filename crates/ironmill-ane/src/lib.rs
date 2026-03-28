//! ANE (Apple Neural Engine) direct runtime backend for ironmill.
//!
//! This crate provides a Rust-native interface to Apple's private ANE APIs
//! (`_ANEClient`, `_ANECompiler`) for compiling and executing models directly
//! on the Neural Engine, bypassing CoreML's `MLModel` path.
//!
//! # ⚠️ Private API Warning
//!
//! This crate uses **undocumented Apple private APIs** that may change between
//! macOS releases. It is feature-gated behind `ane-direct` and should not be
//! used in Mac App Store submissions.
//!
//! # Architecture
//!
//! The crate mirrors `ironmill-coreml` but targets the ANE directly:
//!
//! | Module      | Purpose                                    |
//! |-------------|--------------------------------------------|
//! | `blobfile`  | BLOBFILE weight format writer               |
//! | `tensor`    | IOSurface-backed tensor I/O                 |
//! | `runtime`   | `_ANEClient` lifecycle and program execution|
//! | `cache`     | Compiled program cache with disk persistence|
//! | `split`     | Model → ANE-sized sub-program splitter      |
//! | `program`   | Compiled program handle types               |

#[cfg(not(target_os = "macos"))]
compile_error!("ironmill-ane only supports macOS");

pub mod blobfile;
pub mod cache;
pub mod program;
pub mod runtime;
pub mod split;
pub mod tensor;

use std::path::PathBuf;

use mil_rs::ir::ScalarType;

// ── Error type ────────────────────────────────────────────────────

/// Errors from the ANE runtime backend.
#[derive(Debug, thiserror::Error)]
pub enum AneError {
    /// ANE compilation failed with a status code.
    #[error("ANE compilation failed (status {status:#x}): {context}")]
    CompileFailed { status: u32, context: String },

    /// ANE eval failed with a status code.
    #[error("ANE eval failed (status {status:#x}): {context}")]
    EvalFailed { status: u32, context: String },

    /// IOSurface creation or I/O failed.
    #[error("IOSurface error: {0}")]
    SurfaceError(String),

    /// The compile budget (~119 per process) has been exhausted.
    #[error("ANE compile budget exhausted ({used}/{limit} compilations used)")]
    BudgetExhausted { used: usize, limit: usize },

    /// A generic error from an underlying operation.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Result type alias for ANE operations.
pub type Result<T> = std::result::Result<T, AneError>;

// ── Configuration ─────────────────────────────────────────────────

/// Configuration for the ANE runtime backend.
#[derive(Debug, Clone)]
pub struct AneConfig {
    /// Maximum number of compiled programs to cache in memory.
    /// Must stay under the ~119 per-process compile limit.
    pub max_programs: usize,
    /// Directory for persisting compiled programs to disk.
    /// Bypasses the per-process compile limit on subsequent runs.
    pub cache_dir: Option<PathBuf>,
    /// Enable INT4 data type support (experimental).
    pub enable_int4: bool,
}

impl Default for AneConfig {
    fn default() -> Self {
        Self {
            max_programs: 100,
            cache_dir: None,
            enable_int4: false,
        }
    }
}

// ── Tensor descriptor ─────────────────────────────────────────────

/// Describes a tensor's shape and data type for I/O specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    /// Variable name in the MIL program.
    pub name: String,
    /// Shape in ANE layout: `[1, C, 1, S]`.
    pub shape: [usize; 4],
    /// Element data type.
    pub dtype: ScalarType,
}
