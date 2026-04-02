use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_DIR");

    // Only attempt real bindings on macOS.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        println!("cargo:rustc-cfg=mlx_stub");
        println!("cargo:warning=ironmill-mlx-sys: non-macOS target, using stub bindings");
        return;
    }

    // Check for a pre-built mlx-c installation via MLX_DIR.
    if let Ok(mlx_dir) = std::env::var("MLX_DIR") {
        let mlx_path = PathBuf::from(&mlx_dir);
        let include_dir = mlx_path.join("include");
        let lib_dir = mlx_path.join("lib");

        if include_dir.exists() && lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-lib=mlxc");
            println!("cargo:rustc-link-lib=c++");

            generate_bindings(&include_dir);
            return;
        }

        println!(
            "cargo:warning=MLX_DIR={} does not contain include/ and lib/, falling back to stub",
            mlx_dir
        );
    }

    // Try to find mlx-c source to build via CMake.
    let vendored_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by cargo"),
    )
    .join("vendor")
    .join("mlx-c");

    if vendored_dir.join("CMakeLists.txt").exists() {
        let dst = cmake::Config::new(&vendored_dir)
            .define("MLX_BUILD_TESTS", "OFF")
            .define("MLX_BUILD_EXAMPLES", "OFF")
            .define("BUILD_SHARED_LIBS", "OFF")
            .build();

        let lib_dir = dst.join("lib");
        let include_dir = dst.join("include");

        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=mlxc");
        println!("cargo:rustc-link-lib=c++");

        generate_bindings(&include_dir);
        return;
    }

    // No mlx-c available — emit stub flag.
    println!("cargo:rustc-cfg=mlx_stub");
    println!(
        "cargo:warning=ironmill-mlx-sys: mlx-c not found, using stub bindings. Set MLX_DIR or vendor mlx-c source."
    );
}

fn generate_bindings(include_dir: &std::path::Path) {
    let header = include_dir.join("mlx").join("c").join("mlx.h");
    if !header.exists() {
        println!(
            "cargo:warning=mlx.h not found at {}, using stub bindings",
            header.display()
        );
        println!("cargo:rustc-cfg=mlx_stub");
        return;
    }

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("failed to generate mlx-c bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR must be set by cargo"))
        .join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("failed to write mlx-c bindings");
}
