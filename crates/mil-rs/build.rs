use std::path::PathBuf;

fn main() {
    let proto_dir = PathBuf::from("proto");

    let proto_files: Vec<PathBuf> = std::fs::read_dir(&proto_dir)
        .expect("failed to read proto/ directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().is_some_and(|ext| ext == "proto") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    assert!(
        !proto_files.is_empty(),
        "no .proto files found in {proto_dir:?}"
    );

    prost_build::Config::new()
        .disable_comments(["."])
        .compile_protos(&proto_files, &[&proto_dir])
        .expect("failed to compile protobuf files");
}
