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

#[cfg(test)]
#[cfg(all(feature = "ane", target_os = "macos"))]
mod tests {
    use crate::ane::turboquant::mil_emitter::build_cache_write_program;
    use crate::ane::turboquant::TurboQuantConfig;
    use crate::ane::HardwareAneDevice;
    use crate::turboquant::codebook::lloyd_max_gaussian;
    use ironmill_core::ane::mil_text::{MilTextConfig, program_to_mil_text};

    #[test]
    fn test_generated_cache_write_mil() {
        let config = TurboQuantConfig::new(8, 128, 32, 32, 64, 1).unwrap();
        let (levels, boundaries) = lloyd_max_gaussian(config.head_dim, config.n_bits);
        let (program, weights) = build_cache_write_program(&config, &boundaries, &levels);
        let mil_config = MilTextConfig::default();
        let (mil_text, _) = program_to_mil_text(&program, &mil_config).unwrap();
        let _weight_refs: Vec<(&str, &[u8])> = weights
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();

        let device = match HardwareAneDevice::new() {
            Ok(d) => d,
            Err(e) => {
                println!("    ANE not available: {e}");
                return;
            }
        };
        match device.compile(&mil_text, &[], 33) {
            Ok(_) => {
                println!("    ✅ generated cache-write MIL: compiles on ANE");
            }
            Err(e) => {
                panic!("generated cache-write MIL: compile failed: {e}");
            }
        }
    }
}
