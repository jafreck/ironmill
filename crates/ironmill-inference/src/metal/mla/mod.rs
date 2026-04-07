//! Multi-Head Latent Attention (MLA) support for DeepSeek-V2/V3 models.
//!
//! MLA compresses the KV cache into a low-dimensional latent space,
//! dramatically reducing memory usage. Weight absorption at model load
//! time fuses the up-projection matrices (W_uk, W_uv) into the Q and O
//! projection weights, eliminating runtime decompression:
//!
//!   W_q_absorbed = W_uk^T · W_q_nope   (per head)
//!   W_o_absorbed = W_o · W_uv          (per head)
//!
//! After absorption, standard attention kernels operate directly on the
//! compressed latent vectors. The only special handling is the RoPE key
//! portion, which is stored separately from the latent cache.

pub mod absorption;
pub mod cache;

pub(crate) use absorption::absorb_mla_weights;
pub use absorption::absorb_weights;
pub use cache::MlaKvCache;
pub use mil_rs::weights::MlaConfig;
