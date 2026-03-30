//! Analysis passes for MIL IR programs.
//!
//! Read-only analysis that computes metrics from a [`Program`] without
//! modifying it.

pub mod arch;
pub mod flops;
