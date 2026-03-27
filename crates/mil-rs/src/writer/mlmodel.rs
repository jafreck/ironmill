//! Writer for `.mlmodel` files (single protobuf binary).

use std::path::Path;

use prost::Message;

use crate::error::Result;
use crate::proto::specification::Model;

/// Write a CoreML [`Model`] to a `.mlmodel` file (single protobuf binary).
///
/// # Errors
///
/// Returns [`MilError::Io`](crate::error::MilError::Io) if the file cannot be written.
///
/// # Examples
///
/// ```no_run
/// use mil_rs::{read_mlmodel, write_mlmodel};
///
/// let model = read_mlmodel("input.mlmodel").unwrap();
/// write_mlmodel(&model, "output.mlmodel").unwrap();
/// ```
pub fn write_mlmodel(model: &Model, path: impl AsRef<Path>) -> Result<()> {
    let bytes = model.encode_to_vec();
    std::fs::write(path.as_ref(), bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::read_mlmodel;

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
    fn round_trip_mlmodel() {
        let original = read_mlmodel(fixture_path("MobileNet.mlmodel"))
            .expect("failed to read MobileNet.mlmodel");

        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let out_path = dir.path().join("roundtrip.mlmodel");

        write_mlmodel(&original, &out_path).expect("failed to write mlmodel");

        let reloaded = read_mlmodel(&out_path).expect("failed to read back written mlmodel");

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

        // Byte-level equality: encoding is deterministic for the same message.
        assert_eq!(original.encode_to_vec(), reloaded.encode_to_vec());
    }

    #[test]
    fn error_on_nonexistent_parent_dir() {
        let model = read_mlmodel(fixture_path("MobileNet.mlmodel")).unwrap();
        let bad_path = Path::new("/nonexistent_dir_abc123/model.mlmodel");
        let result = write_mlmodel(&model, bad_path);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), crate::error::MilError::Io(_)),
            "expected Io error for missing parent directory"
        );
    }
}
