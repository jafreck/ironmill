//! Calibration data collection for quantisation algorithms (AWQ, GPTQ).
//!
//! During a calibration forward pass the inference engine calls
//! [`ActivationHook::on_linear_input`] for every linear projection. Two
//! concrete stores are provided:
//!
//! * [`AwqActivationStore`] — lightweight, O(n_features) per projection.
//!   Tracks per-channel mean and max absolute magnitudes.
//! * [`GptqActivationStore`] — heavy, O(n_features²) per projection.
//!   Accumulates the Hessian X^T X for second-order weight updates.

mod awq_store;
pub mod dataset;
mod gptq_store;
mod hook;
mod runner;

// Public API
pub use dataset::CalibrationDataset;
pub use hook::ActivationHook;
pub use runner::{CalibratingEngine, CalibrationRunner, HessianHook, QuipHessianAccumulator};

// Internal only — re-export for test use in metal backend
#[cfg(test)]
pub(crate) use awq_store::AwqActivationStore;
