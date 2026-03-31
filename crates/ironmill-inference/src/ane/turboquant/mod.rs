//! TurboQuant INT8 KV cache compression for ANE inference.

pub mod mil_emitter;
pub mod model;

pub use mil_emitter::{AttentionMilConfig, compute_deq_scale};
pub use model::{KvCacheManager, TurboQuantConfig, TurboQuantModel};
