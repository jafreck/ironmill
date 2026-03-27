use std::path::PathBuf;

fn main() {
    let proto_dir = PathBuf::from("proto");

    let mut coreml_protos = Vec::new();
    let mut onnx_protos = Vec::new();

    for entry in std::fs::read_dir(&proto_dir).expect("failed to read proto/ directory") {
        let path = entry.expect("failed to read entry").path();
        match path.extension().and_then(|e| e.to_str()) {
            Some("proto") => coreml_protos.push(path),
            Some("proto3") => onnx_protos.push(path),
            _ => {}
        }
    }

    assert!(
        !coreml_protos.is_empty(),
        "no .proto files found in {proto_dir:?}"
    );

    // Compile CoreML protos.
    prost_build::Config::new()
        .disable_comments(["."])
        .compile_protos(&coreml_protos, &[&proto_dir])
        .expect("failed to compile CoreML protobuf files");

    // Compile ONNX protos (separate pass to avoid package conflicts).
    if !onnx_protos.is_empty() {
        prost_build::Config::new()
            .disable_comments(["."])
            .compile_protos(&onnx_protos, &[&proto_dir])
            .expect("failed to compile ONNX protobuf files");
    }
}
