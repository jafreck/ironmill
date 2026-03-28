//! Model → ANE-sized sub-program splitter.
//!
//! A full transformer cannot be compiled as a single ANE program.
//! This module splits the model into per-layer sub-programs that
//! execute sequentially, all targeting ANE.

// TODO: Implement split_for_ane — see spec Task 7
