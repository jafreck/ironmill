//! Benchmark suite implementations.
//!
//! Each sub-module provides a [`BenchmarkSuite`](crate::suite::BenchmarkSuite)
//! implementation for a specific benchmark type.

#[cfg(feature = "metal")]
pub mod context_decode;
pub mod coreml;
#[cfg(feature = "metal")]
pub mod decode;
#[cfg(feature = "metal")]
pub mod perplexity_metal;
#[cfg(feature = "metal")]
pub mod prefill;
pub mod quality;

use crate::suite::SuiteRegistry;

/// Register all built-in benchmark suites.
///
/// To add a new benchmark, create a module in `suites/` implementing
/// `BenchmarkSuite`, then add it here.
pub fn register_suites(registry: &mut SuiteRegistry) {
    registry.register(Box::new(coreml::CoremlSuite));

    #[cfg(feature = "metal")]
    {
        registry.register(Box::new(decode::MetalDecodeSuite));
        registry.register(Box::new(prefill::MetalPrefillSuite));
        registry.register(Box::new(context_decode::MetalContextDecodeSuite));
        registry.register(Box::new(perplexity_metal::MetalPerplexitySuite));
    }

    registry.register(Box::new(quality::QualitySuite));
}
