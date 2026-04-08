//! Reader for ONNX `.onnx` model files (protobuf binary).

use std::path::{Path, PathBuf};

use memmap2::Mmap;
use prost::Message;

use crate::error::Result;
use crate::proto::onnx::ModelProto;

/// Read an ONNX `.onnx` file and deserialize it into a [`ModelProto`].
///
/// The file is memory-mapped so that prost decodes directly from mapped
/// pages, avoiding a full-file heap allocation. The mmap is dropped after
/// decoding, leaving only the prost-owned structures in memory.
///
/// # Errors
///
/// Returns [`MilError::Io`] if the file cannot be read, or
/// [`MilError::Protobuf`] if the bytes are not valid protobuf.
///
/// # Examples
///
/// ```no_run
/// use mil_rs::reader::onnx::read_onnx;
///
/// let model = read_onnx("model.onnx").unwrap();
/// println!("IR version: {}", model.ir_version);
/// ```
pub fn read_onnx(path: impl AsRef<Path>) -> Result<ModelProto> {
    let file = std::fs::File::open(path.as_ref())?;
    let mmap = mmap_read_only(&file)?;
    let model = ModelProto::decode(&mmap[..])?;
    Ok(model)
}

/// Read an ONNX `.onnx` file, returning the model and its parent directory.
///
/// The directory is needed to resolve external data files referenced by
/// `TensorProto.data_location == EXTERNAL`.
pub fn read_onnx_with_dir(path: impl AsRef<Path>) -> Result<(ModelProto, PathBuf)> {
    let path = path.as_ref();
    let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let file = std::fs::File::open(path)?;
    let mmap = mmap_read_only(&file)?;
    let model = ModelProto::decode(&mmap[..])?;
    Ok((model, dir))
}

/// Memory-maps a file for read-only access.
///
/// # Safety
///
/// This function assumes the file will not be modified or truncated while the
/// mapping is alive. This is acceptable here because model files are
/// read-only assets that are never modified during loading.
#[allow(unsafe_code)]
fn mmap_read_only(file: &std::fs::File) -> std::io::Result<Mmap> {
    // SAFETY: We use mmap for zero-copy access to large model files.  The
    // safety invariant is that the underlying file must not be modified or
    // truncated while the mapping exists.  Model weight files are read-only
    // assets that are not mutated during loading, so this invariant holds.
    unsafe { Mmap::map(file) }
}

/// Print a human-readable summary of an ONNX model to stdout.
///
/// Displays IR version, opset imports, producer info, graph inputs/outputs,
/// node count, and initializer count.
pub fn print_onnx_summary(model: &ModelProto) {
    println!("ONNX Model Summary");
    println!("===================");
    println!("IR version: {}", model.ir_version);

    if !model.producer_name.is_empty() {
        println!("Producer: {}", model.producer_name);
    }
    if !model.producer_version.is_empty() {
        println!("Producer version: {}", model.producer_version);
    }
    if !model.domain.is_empty() {
        println!("Domain: {}", model.domain);
    }
    if model.model_version != 0 {
        println!("Model version: {}", model.model_version);
    }
    if !model.doc_string.is_empty() {
        println!("Doc: {}", model.doc_string);
    }

    if !model.opset_import.is_empty() {
        println!("\nOpset imports:");
        for opset in &model.opset_import {
            let domain = if opset.domain.is_empty() {
                "ai.onnx"
            } else {
                &opset.domain
            };
            println!("  - {} v{}", domain, opset.version);
        }
    }

    if let Some(graph) = &model.graph {
        println!("\nGraph: {}", graph.name);

        if !graph.input.is_empty() {
            println!("\nInputs ({}):", graph.input.len());
            for input in &graph.input {
                println!("  - {}", value_info_summary(input));
            }
        }

        if !graph.output.is_empty() {
            println!("\nOutputs ({}):", graph.output.len());
            for output in &graph.output {
                println!("  - {}", value_info_summary(output));
            }
        }

        println!("\nNodes: {}", graph.node.len());
        println!("Initializers: {}", graph.initializer.len());
    }
}

fn value_info_summary(vi: &crate::proto::onnx::ValueInfoProto) -> String {
    if vi.doc_string.is_empty() {
        vi.name.clone()
    } else {
        format!("{} — {}", vi.name, vi.doc_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MilError;

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
    #[ignore] // Requires test fixture files
    fn read_mnist_onnx() {
        let path = fixture_path("mnist.onnx");
        let model = read_onnx(&path).expect("failed to read mnist.onnx");

        assert!(model.ir_version > 0, "IR version should be set");

        let graph = model.graph.as_ref().expect("model should have a graph");
        assert!(!graph.input.is_empty(), "graph should have inputs");
        assert!(!graph.output.is_empty(), "graph should have outputs");
        assert!(!graph.node.is_empty(), "graph should have nodes");
    }

    #[test]
    #[ignore] // Requires test fixture files
    fn read_squeezenet_onnx() {
        let path = fixture_path("squeezenet1.1.onnx");
        let model = read_onnx(&path).expect("failed to read squeezenet1.1.onnx");

        assert!(model.ir_version > 0);
        assert!(
            !model.opset_import.is_empty(),
            "model should have opset imports"
        );

        let graph = model.graph.as_ref().expect("model should have a graph");
        assert!(!graph.node.is_empty(), "graph should have nodes");
        assert!(
            !graph.initializer.is_empty(),
            "squeezenet should have initializers (weights)"
        );
    }

    #[test]
    #[ignore] // Requires test fixture files
    fn print_onnx_summary_does_not_panic() {
        let path = fixture_path("mnist.onnx");
        let model = read_onnx(&path).unwrap();
        print_onnx_summary(&model);
    }

    #[test]
    fn error_on_missing_file() {
        let result = read_onnx("nonexistent.onnx");
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), MilError::Io(_)),
            "expected Io error"
        );
    }

    #[test]
    fn error_on_invalid_protobuf() {
        let dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(dir).join("Cargo.toml");
        let result = read_onnx(&path);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), MilError::ProtobufDecode(_)),
            "expected ProtobufDecode error"
        );
    }
}
