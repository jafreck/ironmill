//! Low-level safe FFI bindings for Apple Neural Engine private APIs.
//!
//! `ironmill-ane-sys` consolidates the Objective-C runtime FFI code that was
//! previously duplicated across `mil-rs` (compiler) and `ironmill-ane`
//! (runtime) into a single crate with a safe public API over unsafe
//! internals.
//!
//! # ⚠️ Private API Warning
//!
//! This crate uses **undocumented Apple private APIs** that may change between
//! macOS releases.  It should not be used in Mac App Store submissions.
//!
//! # macOS Only
//!
//! This crate only compiles on macOS — a `compile_error!` is emitted on
//! other platforms.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_os = "macos"))]
compile_error!("ironmill-ane-sys only supports macOS");

pub mod buffers_ready;
pub mod client;
pub mod device;
pub mod error;
pub mod events;
pub mod iosurface;
pub mod mapper;
pub mod model;
pub(crate) mod objc;
pub mod perf;
pub mod process;
pub mod program;
pub mod request;
pub mod token;
pub mod util;
pub mod validate;
pub mod virtual_client;
pub mod weight;

pub use buffers_ready::{InputBuffersReady, OutputSetEnqueue};
pub use client::{Client, DaemonConnection};
pub use device::{DeviceController, DeviceInfo};
pub use error::AneSysError;
pub use events::{SharedEvents, SharedSignalEvent, SharedWaitEvent};
pub use iosurface::{AneBuffer, AneIOSurfaceObject, IOSurfaceOutputSets};
pub use mapper::ProgramIOSurfacesMapper;
pub use model::{InMemoryModel, InMemoryModelDescriptor};
pub use perf::PerformanceStats;
// NOTE: PerformanceStatsIOSurface and QoSMapper are pub(crate) — currently
// unused outside tests. Access via `ironmill_ane_sys::perf::` if needed internally.
pub use process::current_rss;
pub use program::ProgramForEvaluation;
pub use request::{AneRequest, ChainingRequest};
pub use token::ModelToken;
// NOTE: AneCloneHelper, AneErrors, and AneLog are pub(crate) — currently
// unused outside tests. Access via `ironmill_ane_sys::util::` if needed internally.
pub use validate::{
    RawValidateFn, get_validate_network_supported_version, validate_mil_network_on_host_ptr,
    validate_mlir_network_on_host_ptr,
};
pub use virtual_client::{
    BuildVersionInfo, DeviceExtendedInfo, DeviceInfoInner, VirtANEModel, VirtualClient,
};
pub use weight::{ModelInstanceParameters, ProcedureData, Weight};
