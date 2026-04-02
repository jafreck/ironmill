//! Writer for `.mlpackage` directories (CoreML's modern package format).
//!
//! Creates the standard directory structure:
//!
//! ```text
//! output.mlpackage/
//! ├── Manifest.json
//! └── Data/
//!     └── com.apple.CoreML/
//!         └── model.mlmodel
//! ```

use std::collections::HashMap;
use std::path::Path;

use prost::Message;

use crate::error::{MilError, Result};
use crate::proto::specification::Model;
use crate::reader::mlpackage::{ItemInfo, Manifest};

const MODEL_PATH: &str = "com.apple.CoreML/model.mlmodel";

/// Write a CoreML [`Model`] to a `.mlpackage` directory.
///
/// Creates the package directory, a `Manifest.json`, and writes the
/// protobuf-encoded model to `Data/com.apple.CoreML/model.mlmodel`.
///
/// # Errors
///
/// Returns [`MilError::Io`] if directories or
/// files cannot be created or written.
///
/// # Examples
///
/// ```no_run
/// use mil_rs::{read_mlmodel, write_mlpackage};
///
/// let model = read_mlmodel("input.mlmodel").unwrap();
/// write_mlpackage(&model, "output.mlpackage").unwrap();
/// ```
pub fn write_mlpackage(model: &Model, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();

    let data_dir = path.join("Data").join("com.apple.CoreML");
    std::fs::create_dir_all(&data_dir)?;

    // Write manifest
    let manifest = build_manifest();
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| MilError::InvalidPackage(format!("failed to serialize Manifest.json: {e}")))?;
    std::fs::write(path.join("Manifest.json"), json)?;

    // Write model protobuf
    let bytes = model.encode_to_vec();
    std::fs::write(data_dir.join("model.mlmodel"), bytes)?;

    Ok(())
}

/// Generate a random UUID v4 string (uppercase, no external dependency).
fn random_uuid() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id() as u64;
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);

    // Combine time, PID, and counter to avoid collisions.
    let seed = (nanos as u64) ^ (pid << 32) ^ count.wrapping_mul(0x9E3779B97F4A7C15);
    // Simple xorshift-based PRNG seeded from the combined value.
    let mut state = seed | 1;
    let mut bytes = [0u8; 16];
    for chunk in bytes.chunks_exact_mut(8) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        chunk.copy_from_slice(&state.to_le_bytes());
    }
    // Set version (4) and variant (RFC 4122).
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    format!(
        "{:02X}{:02X}{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    )
}

/// Build a default `Manifest` for a single-model package.
fn build_manifest() -> Manifest {
    let model_id = random_uuid();
    let mut entries = HashMap::new();
    entries.insert(
        model_id.clone(),
        ItemInfo {
            path: MODEL_PATH.to_string(),
            author: Some("com.apple.CoreML".to_string()),
            description: Some("CoreML Model Specification".to_string()),
            name: Some("model.mlmodel".to_string()),
        },
    );

    Manifest {
        file_format_version: "1.0.0".to_string(),
        item_info_entries: entries,
        root_model_identifier: model_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::{read_mlmodel, read_mlpackage};

    fn fixture_path(name: &str) -> std::path::PathBuf {
        let manifest = env!("CARGO_MANIFEST_DIR");
        Path::new(manifest)
            .join("..")
            .join("..")
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn round_trip_mlpackage() {
        let original = read_mlmodel(fixture_path("MobileNet.mlmodel"))
            .expect("failed to read MobileNet.mlmodel");

        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let pkg_path = dir.path().join("output.mlpackage");

        write_mlpackage(&original, &pkg_path).expect("failed to write mlpackage");

        // Verify directory structure
        assert!(pkg_path.join("Manifest.json").is_file());
        assert!(
            pkg_path
                .join("Data/com.apple.CoreML/model.mlmodel")
                .is_file()
        );

        let reloaded = read_mlpackage(&pkg_path).expect("failed to read back written mlpackage");

        assert_eq!(
            original.specification_version,
            reloaded.specification_version
        );

        let orig_desc = original.description.as_ref().unwrap();
        let new_desc = reloaded.description.as_ref().unwrap();
        assert_eq!(orig_desc.input.len(), new_desc.input.len());
        assert_eq!(orig_desc.output.len(), new_desc.output.len());

        assert_eq!(
            std::mem::discriminant(&original.r#type),
            std::mem::discriminant(&reloaded.r#type),
        );

        assert_eq!(original.encode_to_vec(), reloaded.encode_to_vec());
    }

    #[test]
    fn manifest_json_is_valid() {
        let original = read_mlmodel(fixture_path("MobileNet.mlmodel")).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let pkg_path = dir.path().join("check_manifest.mlpackage");
        write_mlpackage(&original, &pkg_path).unwrap();

        let contents = std::fs::read_to_string(pkg_path.join("Manifest.json")).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_str(&contents).expect("Manifest.json should be valid JSON");

        assert_eq!(manifest["fileFormatVersion"], "1.0.0");

        let root_id = manifest["rootModelIdentifier"].as_str().unwrap();
        let entry = &manifest["itemInfoEntries"][root_id];
        assert!(entry.is_object(), "root model entry should exist");
        assert_eq!(entry["path"], "com.apple.CoreML/model.mlmodel");
        assert_eq!(entry["name"], "model.mlmodel");
        assert_eq!(entry["author"], "com.apple.CoreML");
    }

    #[test]
    fn error_on_nonexistent_parent_dir() {
        let model = read_mlmodel(fixture_path("MobileNet.mlmodel")).unwrap();
        let bad_path = Path::new("/nonexistent_dir_abc123/output.mlpackage");
        let result = write_mlpackage(&model, bad_path);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), MilError::Io(_)),
            "expected Io error for missing parent directory"
        );
    }
}
