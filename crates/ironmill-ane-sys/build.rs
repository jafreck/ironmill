fn main() {
    // Compile the Objective-C helper that wraps CFRelease in @try/@catch.
    // This prevents ObjC exceptions during Drop from aborting the process
    // with "Rust cannot catch foreign exceptions".
    cc::Build::new()
        .file("objc/safe_release.m")
        .compile("ane_objc_helpers");

    println!("cargo:rerun-if-changed=objc/safe_release.m");
}
