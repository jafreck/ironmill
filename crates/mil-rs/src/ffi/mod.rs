//! FFI bindings for direct Apple Neural Engine compilation.
//!
//! This module provides Rust wrappers around Apple's private `_ANECompiler`
//! Objective-C class, enabling direct ANE compilation without shelling out to
//! `xcrun coremlcompiler`.
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
