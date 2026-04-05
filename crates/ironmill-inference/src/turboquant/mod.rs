//! Shared TurboQuant logic used by the Metal backend.
//!
//! Contains the backend-independent math and data generation:
//! - Lloyd-Max codebook computation
//! - Rotation sign and QJL matrix generation
//! - Outlier channel detection from weight norms

pub mod cache_layout;
pub mod codebook;
pub mod outlier;
pub mod rotation;
