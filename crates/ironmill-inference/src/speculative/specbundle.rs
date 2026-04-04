//! SpecBundle checkpoint loader for EAGLE-3 draft heads.
//!
//! A SpecBundle is a directory containing serialized draft-head weights
//! and a JSON manifest. The expected layout is:
//!
//! ```text
//! specbundle/
//! ├── manifest.json    — { "hidden_dim": N, "vocab_size": V, "num_layers": L }
//! └── weights.bin      — concatenated f16 layer weights
//! ```

use std::path::Path;

use half::f16;
use serde::Deserialize;

use crate::engine::InferenceError;
use crate::speculative::draft_head::DraftHead;

/// JSON manifest describing the draft head's dimensions.
#[derive(Debug, Deserialize)]
struct SpecManifest {
    hidden_dim: usize,
    vocab_size: usize,
    num_layers: usize,
}

/// Load a [`DraftHead`] from a SpecBundle directory.
///
/// Reads `manifest.json` for dimensions and `weights.bin` for the raw f16
/// weight data. Returns an error if the bundle is missing, malformed, or
/// the weight file size doesn't match the manifest.
pub fn load_spec_bundle(path: &Path) -> Result<DraftHead, InferenceError> {
    let manifest_path = path.join("manifest.json");
    let weights_path = path.join("weights.bin");

    let manifest_data = std::fs::read_to_string(&manifest_path).map_err(|e| {
        InferenceError::Runtime(format!(
            "failed to read spec bundle manifest at {}: {e}",
            manifest_path.display()
        ))
    })?;

    let manifest: SpecManifest = serde_json::from_str(&manifest_data)
        .map_err(|e| InferenceError::Runtime(format!("invalid spec bundle manifest: {e}")))?;

    let weight_bytes = std::fs::read(&weights_path).map_err(|e| {
        InferenceError::Runtime(format!(
            "failed to read spec bundle weights at {}: {e}",
            weights_path.display()
        ))
    })?;

    let elements_per_layer = manifest.hidden_dim * manifest.vocab_size;
    let expected_bytes = manifest.num_layers * elements_per_layer * 2; // f16 = 2 bytes
    if weight_bytes.len() != expected_bytes {
        return Err(InferenceError::Runtime(format!(
            "spec bundle weight size mismatch: expected {expected_bytes} bytes, got {}",
            weight_bytes.len()
        )));
    }

    // Reinterpret bytes as f16 values.
    let all_f16: Vec<f16> = weight_bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    let layer_weights: Vec<Vec<f16>> = all_f16
        .chunks_exact(elements_per_layer)
        .map(|chunk| chunk.to_vec())
        .collect();

    Ok(DraftHead::new(
        layer_weights,
        manifest.hidden_dim,
        manifest.vocab_size,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_bundle(dir: &Path, hidden_dim: usize, vocab_size: usize, num_layers: usize) {
        std::fs::create_dir_all(dir).unwrap();

        let manifest = format!(
            r#"{{"hidden_dim": {hidden_dim}, "vocab_size": {vocab_size}, "num_layers": {num_layers}}}"#
        );
        std::fs::write(dir.join("manifest.json"), manifest).unwrap();

        let elements = num_layers * hidden_dim * vocab_size;
        let weights: Vec<u8> = (0..elements)
            .flat_map(|i| {
                let val = f16::from_f32((i % 5) as f32 * 0.1);
                val.to_le_bytes()
            })
            .collect();
        std::fs::write(dir.join("weights.bin"), weights).unwrap();
    }

    #[test]
    fn speculative_specbundle_load_roundtrip() {
        let dir = std::env::current_dir()
            .unwrap()
            .join("_test_specbundle_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        write_test_bundle(&dir, 4, 8, 2);

        let head = load_spec_bundle(&dir).unwrap();
        assert_eq!(head.hidden_dim, 4);
        assert_eq!(head.vocab_size, 8);
        assert_eq!(head.layer_weights.len(), 2);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn speculative_specbundle_missing_manifest() {
        let dir = std::env::current_dir()
            .unwrap()
            .join("_test_specbundle_missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // No manifest.json
        std::fs::write(dir.join("weights.bin"), b"dummy").unwrap();

        let err = load_spec_bundle(&dir).unwrap_err();
        assert!(err.to_string().contains("manifest"), "error: {err}");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn speculative_specbundle_weight_size_mismatch() {
        let dir = std::env::current_dir()
            .unwrap()
            .join("_test_specbundle_mismatch");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let manifest = r#"{"hidden_dim": 4, "vocab_size": 8, "num_layers": 1}"#;
        std::fs::write(dir.join("manifest.json"), manifest).unwrap();
        // Write too few bytes (need 4*8*2 = 64 bytes, give 10)
        std::fs::write(dir.join("weights.bin"), vec![0u8; 10]).unwrap();

        let err = load_spec_bundle(&dir).unwrap_err();
        assert!(err.to_string().contains("size mismatch"), "error: {err}");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
