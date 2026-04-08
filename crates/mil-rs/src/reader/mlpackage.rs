//! Reader for `.mlpackage` directories (CoreML's modern package format).
//!
//! An `.mlpackage` is a directory containing a `Manifest.json` that
//! describes the model items and their paths, plus a `Data/` subdirectory
//! holding the model spec (protobuf) and weight files.

use std::collections::HashMap;
use std::path::Path;

use prost::Message;
use serde::{Deserialize, Serialize};

use crate::error::{MilError, Result};
use crate::proto::specification::Model;

/// Parsed representation of an `.mlpackage` `Manifest.json`.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Manifest {
    pub(crate) file_format_version: String,
    pub(crate) item_info_entries: HashMap<String, ItemInfo>,
    pub(crate) root_model_identifier: String,
}

/// A single item entry inside a `Manifest.json`.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ItemInfo {
    pub(crate) path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) author: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
}

/// Read a `.mlpackage` directory and extract the CoreML model.
///
/// An `.mlpackage` is a directory containing a `Manifest.json` that
/// describes the model items and their paths. The model spec protobuf
/// is located inside the `Data/` subdirectory at the path indicated
/// by the manifest's root model identifier.
///
/// # Errors
///
/// Returns [`MilError::InvalidPackage`] if:
/// - `path` is not a directory
/// - `Manifest.json` is missing or malformed
/// - The root model identifier is not found in the manifest
/// - The model spec file is missing from the package
///
/// Returns [`MilError::Protobuf`] if the model spec is not valid protobuf.
pub fn read_mlpackage(path: impl AsRef<Path>) -> Result<Model> {
    let path = path.as_ref();

    if !path.is_dir() {
        return Err(MilError::InvalidPackage(format!(
            "not a directory: {}",
            path.display()
        )));
    }

    let manifest = read_manifest(path)?;

    let item = manifest
        .item_info_entries
        .get(&manifest.root_model_identifier)
        .ok_or_else(|| {
            MilError::InvalidPackage(format!(
                "root model identifier '{}' not found in manifest entries",
                manifest.root_model_identifier
            ))
        })?;

    let model_path = path.join("Data").join(&item.path);

    if !model_path.is_file() {
        return Err(MilError::InvalidPackage(format!(
            "model spec not found at: {}",
            model_path.display()
        )));
    }

    let bytes = std::fs::read(&model_path)?;
    let model = Model::decode(&bytes[..])?;
    Ok(model)
}

/// Read and parse the `Manifest.json` from an `.mlpackage` directory.
fn read_manifest(package_dir: &Path) -> Result<Manifest> {
    let manifest_path = package_dir.join("Manifest.json");

    if !manifest_path.is_file() {
        return Err(MilError::InvalidPackage(format!(
            "Manifest.json not found in: {}",
            package_dir.display()
        )));
    }

    let contents = std::fs::read_to_string(&manifest_path)?;
    let manifest: Manifest = serde_json::from_str(&contents)
        .map_err(|e| MilError::InvalidPackage(format!("failed to parse Manifest.json: {e}")))?;

    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn parse_manifest_json() {
        let pkg = fixture_path("simple.mlpackage");
        let manifest = read_manifest(&pkg).expect("failed to parse Manifest.json");

        assert_eq!(manifest.file_format_version, "2.0.0");
        assert_eq!(
            manifest.root_model_identifier,
            "com.apple.CoreML/model.mlmodel"
        );
        assert!(
            manifest
                .item_info_entries
                .contains_key("com.apple.CoreML/model.mlmodel")
        );

        let item = &manifest.item_info_entries["com.apple.CoreML/model.mlmodel"];
        assert_eq!(item.path, "com.apple.CoreML/model.mlmodel");
        assert_eq!(item.author.as_deref(), Some("ironmill test fixture"));
        assert_eq!(item.description.as_deref(), Some("Minimal test model"));
    }

    #[test]
    fn error_on_non_directory() {
        let result = read_mlpackage("Cargo.toml");
        let err = result.expect_err("expected error for non-directory path");
        assert!(
            matches!(err, MilError::InvalidPackage(_)),
            "expected InvalidPackage, got: {err}"
        );
        assert!(
            err.to_string().contains("not a directory"),
            "error message should mention 'not a directory': {err}"
        );
    }

    #[test]
    fn error_on_missing_manifest() {
        // Use an arbitrary directory that exists but has no Manifest.json.
        let dir = env!("CARGO_MANIFEST_DIR");
        let result = read_mlpackage(dir);
        let err = result.expect_err("expected error for missing Manifest.json");
        assert!(
            matches!(err, MilError::InvalidPackage(_)),
            "expected InvalidPackage, got: {err}"
        );
        assert!(
            err.to_string().contains("Manifest.json not found"),
            "error message should mention missing Manifest.json: {err}"
        );
    }

    #[test]
    fn error_on_missing_model_spec() {
        // The simple.mlpackage fixture has a Manifest.json but no actual
        // protobuf model file, so read_mlpackage should return an error.
        let pkg = fixture_path("simple.mlpackage");
        let result = read_mlpackage(&pkg);
        let err = result.expect_err("expected error for missing model spec");
        assert!(
            matches!(err, MilError::InvalidPackage(_)),
            "expected InvalidPackage, got: {err}"
        );
        assert!(
            err.to_string().contains("model spec not found"),
            "error message should mention missing model spec: {err}"
        );
    }
}
