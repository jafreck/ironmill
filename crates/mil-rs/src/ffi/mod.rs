//! FFI bindings for direct Apple Neural Engine compilation.
//!
//! This module provides Rust wrappers around Apple's private
//! `_ANEInMemoryModelDescriptor` and `_ANEInMemoryModel` Objective-C classes,
//! enabling direct ANE compilation and execution following the same approach
//! as the Orion project.
//!
//! # ⚠️ Private API Warning
//!
//! The APIs wrapped here are **private Apple frameworks** and are:
//! - **Not documented** or supported by Apple
//! - **Subject to change or removal** without notice across macOS versions
//! - **Not permitted** in Mac App Store submissions
//!
//! Use this module only for development, benchmarking, or internal tooling.

pub mod ane;
