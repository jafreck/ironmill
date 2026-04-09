use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::tempdir;

/// Resolve a fixture file relative to the workspace root's `tests/fixtures/`.
fn fixture_path(name: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest)
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Build a `Command` that runs `cargo run -p ironmill-cli --quiet --`.
fn ironmill_cmd() -> Command {
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "-p", "ironmill-cli", "--quiet", "--"]);
    cmd
}

/// Skip a test if the fixture file is not present.
/// Returns the fixture path if it exists.
fn require_fixture(name: &str) -> PathBuf {
    let path = fixture_path(name);
    if !path.exists() {
        eprintln!("SKIPPING: fixture not found: {}", path.display());
        eprintln!("Run ./scripts/download-fixtures.sh to download test fixtures");
        // Return the path anyway — the test will check and return early
    }
    path
}

/// Recursively compute the total size (in bytes) of a directory.
fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if path.is_dir() {
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let ft = entry.file_type().unwrap();
            if ft.is_file() {
                total += entry.metadata().unwrap().len();
            } else if ft.is_dir() {
                total += dir_size(&entry.path());
            }
        }
    }
    total
}

// ─── Compile Subcommand ────────────────────────────────────────────────

#[test]
fn cli_compile_default() {
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile");

    assert!(
        result.status.success(),
        "compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists(), "output .mlpackage must exist");
    assert!(dir_size(&output) > 0, "output directory must be non-empty");
}

#[test]
fn cli_compile_with_fp16() {
    // Baseline: compile without quantization.
    let tmp_base = tempdir().unwrap();
    let base_output = tmp_base.path().join("mnist.mlpackage");

    let base = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["-o", base_output.to_str().unwrap()])
        .output()
        .expect("baseline compile failed");
    assert!(
        base.status.success(),
        "baseline: {}",
        String::from_utf8_lossy(&base.stderr)
    );

    // FP16 compile.
    let tmp_fp16 = tempdir().unwrap();
    let fp16_output = tmp_fp16.path().join("mnist.mlpackage");

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--quantize", "fp16"])
        .args(["-o", fp16_output.to_str().unwrap()])
        .output()
        .expect("fp16 compile failed");

    assert!(
        result.status.success(),
        "fp16 compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(fp16_output.exists(), "fp16 output .mlpackage must exist");

    let base_size = dir_size(&base_output);
    let fp16_size = dir_size(&fp16_output);
    assert!(
        fp16_size <= base_size,
        "fp16 package ({fp16_size} bytes) should be <= baseline ({base_size} bytes)"
    );
}

#[test]
fn cli_compile_with_int8() {
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--quantize", "int8"])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile --quantize int8");

    assert!(
        result.status.success(),
        "int8 compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists(), "int8 output .mlpackage must exist");
}

#[test]
fn cli_compile_with_mixed_precision_preset() {
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--quantize", "mixed-fp16-int8"])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile --quantize mixed-fp16-int8");

    assert!(
        result.status.success(),
        "mixed-fp16-int8 compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        output.exists(),
        "mixed-precision output .mlpackage must exist"
    );
}

#[test]
fn cli_compile_with_quantize_config() {
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let config_path = tmp.path().join("quant.toml");
    std::fs::write(
        &config_path,
        r#"[rules]
"*" = "fp16"
default = "fp16"
"#,
    )
    .unwrap();

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--quantize-config", config_path.to_str().unwrap()])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile --quantize-config");

    assert!(
        result.status.success(),
        "quantize-config compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        output.exists(),
        "quantize-config output .mlpackage must exist"
    );
}

#[test]
fn cli_compile_no_fusion() {
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .arg("--no-fusion")
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile --no-fusion");

    assert!(
        result.status.success(),
        "no-fusion compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists(), "no-fusion output .mlpackage must exist");
}

#[test]
fn cli_compile_with_input_shape() {
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--input-shape", "input:1,1,28,28"])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile --input-shape");

    assert!(
        result.status.success(),
        "input-shape compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists(), "input-shape output .mlpackage must exist");
}

#[test]
fn cli_compile_annotate_compute_units() {
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .arg("--annotate-compute-units")
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile --annotate-compute-units");

    assert!(
        result.status.success(),
        "annotate-compute-units compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        output.exists(),
        "annotate-compute-units output .mlpackage must exist"
    );
}

#[test]
fn cli_compile_ane_memory_budget() {
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--ane-memory-budget", "512MB"])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile --ane-memory-budget");

    assert!(
        result.status.success(),
        "ane-memory-budget compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        output.exists(),
        "ane-memory-budget output .mlpackage must exist"
    );
}

#[test]
fn cli_compile_split_draft_layers() {
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--split-draft-layers", "2"])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile --split-draft-layers");

    // MNIST has no transformer layers, so split-draft-layers fails gracefully
    // with an informative error rather than panicking.
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        !result.status.success(),
        "split-draft-layers on a non-transformer model should fail"
    );
    assert!(
        stderr.contains("split") || stderr.contains("layer") || stderr.contains("draft"),
        "error should mention split/layer/draft context, got:\n{stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "split-draft-layers should not panic"
    );
}

#[test]
fn cli_compile_pipeline_config() {
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let config_path = tmp.path().join("pipeline.toml");
    std::fs::write(
        &config_path,
        r#"[[passes]]
name = "dead-code-elimination"
enabled = true

[[passes]]
name = "fp16-quantization"
enabled = true
"#,
    )
    .unwrap();

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--pipeline-config", config_path.to_str().unwrap()])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile --pipeline-config");

    assert!(
        result.status.success(),
        "pipeline-config compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        output.exists(),
        "pipeline-config output .mlpackage must exist"
    );
}

// ─── Validate Subcommand ───────────────────────────────────────────────

#[test]
fn cli_validate_text_format() {
    let result = ironmill_cmd()
        .args(["validate"])
        .arg(fixture_path("mnist.onnx"))
        .output()
        .expect("failed to run ironmill validate");

    assert!(
        result.status.success(),
        "validate failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("ANE"),
        "validate text output should mention ANE, got:\n{stdout}"
    );
}

#[test]
fn cli_validate_json_format() {
    let result = ironmill_cmd()
        .args(["validate"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--format", "json"])
        .output()
        .expect("failed to run ironmill validate --format json");

    assert!(
        result.status.success(),
        "validate json failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let stdout = String::from_utf8_lossy(&result.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("validate --format json should produce valid JSON");
    assert!(
        parsed.get("ane_compute_pct").is_some(),
        "JSON output must contain 'ane_compute_pct' field, got:\n{stdout}"
    );
}

// ─── Other Subcommands ─────────────────────────────────────────────────

#[test]
fn cli_inspect() {
    let result = ironmill_cmd()
        .args(["inspect"])
        .arg(fixture_path("mnist.onnx"))
        .output()
        .expect("failed to run ironmill inspect");

    assert!(
        result.status.success(),
        "inspect failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(!stdout.is_empty(), "inspect should produce output");
    // ONNX inspect prints op count and tensor names.
    assert!(
        stdout.contains("op") || stdout.contains("Op"),
        "inspect should mention op count, got:\n{stdout}"
    );
    assert!(
        stdout.contains("Input")
            || stdout.contains("input")
            || stdout.contains("Output")
            || stdout.contains("output"),
        "inspect should mention input/output tensor names, got:\n{stdout}"
    );
}

#[test]
fn cli_compile_pipeline_subcommand() {
    let tmp = tempdir().unwrap();
    let output_dir = tmp.path().join("pipeline-out");
    let fixture = fixture_path("mnist.onnx");

    let manifest_path = tmp.path().join("manifest.toml");
    std::fs::write(
        &manifest_path,
        format!(
            r#"[pipeline]
name = "test-pipeline"

[[stages]]
name = "stage1"
onnx = "{onnx}"
quantize = "none"
depends_on = []
"#,
            onnx = fixture.display()
        ),
    )
    .unwrap();

    let result = ironmill_cmd()
        .args(["compile-pipeline"])
        .arg(manifest_path.to_str().unwrap())
        .args(["-o", output_dir.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile-pipeline");

    assert!(
        result.status.success(),
        "compile-pipeline failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output_dir.exists(), "output directory must exist");

    let stage_pkg = output_dir.join("stage1.mlpackage");
    assert!(
        stage_pkg.exists(),
        "stage1.mlpackage must exist at {}",
        stage_pkg.display()
    );
    assert!(
        output_dir.join("pipeline.json").exists(),
        "pipeline.json must exist in output directory"
    );
}

#[test]
fn cli_pipeline_report() {
    let tmp = tempdir().unwrap();

    let config_a_path = tmp.path().join("config_a.toml");
    std::fs::write(
        &config_a_path,
        r#"[[passes]]
name = "dead-code-elimination"
enabled = true
"#,
    )
    .unwrap();

    let config_b_path = tmp.path().join("config_b.toml");
    std::fs::write(
        &config_b_path,
        r#"[[passes]]
name = "dead-code-elimination"
enabled = true

[[passes]]
name = "fp16-quantization"
enabled = true
"#,
    )
    .unwrap();

    let result = ironmill_cmd()
        .args(["pipeline-report"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--config-a", config_a_path.to_str().unwrap()])
        .args(["--config-b", config_b_path.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill pipeline-report");

    assert!(
        result.status.success(),
        "pipeline-report failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        !stdout.is_empty(),
        "pipeline-report should produce comparison output"
    );
    // The report compares two pipeline runs; expect some comparison content.
    assert!(
        stdout.contains("Op") || stdout.contains("op") || stdout.contains("Pipeline"),
        "pipeline-report should contain comparison metrics, got:\n{stdout}"
    );
}

// ─── Error Cases ───────────────────────────────────────────────────────

#[test]
fn cli_compile_missing_input() {
    let result = ironmill_cmd()
        .args(["compile", "nonexistent.onnx"])
        .output()
        .expect("failed to run ironmill compile");

    assert!(
        !result.status.success(),
        "compile with missing input should fail"
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        !stderr.is_empty(),
        "stderr should contain an error message for missing input"
    );
}

#[test]
fn cli_compile_conflicting_quantize_flags() {
    // --quantize is a single String option; passing two --quantize flags
    // means clap takes the last value. This should not panic.
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--quantize", "fp16", "--quantize", "int8"])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile with conflicting quantize flags");

    // Should either succeed (last value wins) or fail gracefully — not panic.
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        !stderr.contains("panicked"),
        "conflicting quantize flags should not panic, got:\n{stderr}"
    );
}

#[test]
fn cli_compile_invalid_epochs() {
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("mnist.mlpackage");

    let result = ironmill_cmd()
        .args(["compile"])
        .arg(fixture_path("mnist.onnx"))
        .args(["--updatable-layers", "x", "--epochs=-5"])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile with invalid epochs");

    assert!(
        !result.status.success(),
        "compile with --epochs -5 should fail"
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("positive")
            || stderr.contains("Positive")
            || stderr.contains("must be")
            || stderr.contains("epoch"),
        "stderr should mention validation error for epochs, got:\n{stderr}"
    );
}

// ─── Architecture-Specific Model Tests ─────────────────────────────────
// These tests require additional fixtures from ./scripts/download-fixtures.sh
// They are skipped gracefully if fixtures are not present.

#[test]
fn cli_compile_whisper_tiny_encoder() {
    let fixture = require_fixture("whisper-tiny-encoder.onnx");
    if !fixture.exists() {
        return;
    }
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("whisper-enc.mlpackage");
    let result = ironmill_cmd()
        .args(["compile"])
        .arg(&fixture)
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile");
    assert!(
        result.status.success(),
        "whisper-tiny-encoder compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists());
}

#[test]
fn cli_compile_whisper_tiny_encoder_polar_quant() {
    let fixture = require_fixture("whisper-tiny-encoder.onnx");
    if !fixture.exists() {
        return;
    }
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("whisper-enc-polar.mlpackage");
    let result = ironmill_cmd()
        .args(["compile"])
        .arg(&fixture)
        .args(["--polar-quantize", "4"])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile");
    assert!(
        result.status.success(),
        "whisper-tiny-encoder polar-quant compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists());
    assert!(
        dir_size(&output) > 0,
        "polar-quantized output directory must be non-empty"
    );
}

#[test]
fn cli_compile_whisper_tiny_decoder() {
    let fixture = require_fixture("whisper-tiny-decoder.onnx");
    if !fixture.exists() {
        return;
    }
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("whisper-dec.mlpackage");
    let result = ironmill_cmd()
        .args(["compile"])
        .arg(&fixture)
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile");
    assert!(
        result.status.success(),
        "whisper-tiny-decoder compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists());
}

#[test]
fn cli_compile_whisper_tiny_decoder_polar_quant() {
    let fixture = require_fixture("whisper-tiny-decoder.onnx");
    if !fixture.exists() {
        return;
    }
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("whisper-dec-polar.mlpackage");
    let result = ironmill_cmd()
        .args(["compile"])
        .arg(&fixture)
        .args(["--polar-quantize", "4"])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile");
    assert!(
        result.status.success(),
        "whisper-tiny-decoder polar-quant compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists());
    assert!(
        dir_size(&output) > 0,
        "polar-quantized output directory must be non-empty"
    );
}

#[test]
fn cli_compile_distilbert() {
    let fixture = require_fixture("distilbert.onnx");
    if !fixture.exists() {
        return;
    }
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("distilbert.mlpackage");
    let result = ironmill_cmd()
        .args(["compile"])
        .arg(&fixture)
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile");
    assert!(
        result.status.success(),
        "distilbert compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists());
}

#[test]
fn cli_compile_distilbert_polar_quant() {
    let fixture = require_fixture("distilbert.onnx");
    if !fixture.exists() {
        return;
    }
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("distilbert-polar.mlpackage");
    let result = ironmill_cmd()
        .args(["compile"])
        .arg(&fixture)
        .args(["--polar-quantize", "4"])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile");
    assert!(
        result.status.success(),
        "distilbert polar-quant compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists());
    assert!(
        dir_size(&output) > 0,
        "polar-quantized output directory must be non-empty"
    );
}

#[test]
fn cli_compile_vit_base() {
    let fixture = require_fixture("vit-base.onnx");
    if !fixture.exists() {
        return;
    }
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("vit-base.mlpackage");
    let result = ironmill_cmd()
        .args(["compile"])
        .arg(&fixture)
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile");
    assert!(
        result.status.success(),
        "vit-base compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists());
}

#[test]
fn cli_compile_vit_base_polar_quant() {
    let fixture = require_fixture("vit-base.onnx");
    if !fixture.exists() {
        return;
    }
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("vit-base-polar.mlpackage");
    let result = ironmill_cmd()
        .args(["compile"])
        .arg(&fixture)
        .args(["--polar-quantize", "4"])
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile");
    assert!(
        result.status.success(),
        "vit-base polar-quant compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists());
    assert!(
        dir_size(&output) > 0,
        "polar-quantized output directory must be non-empty"
    );
}

#[test]
fn cli_compile_qwen3_0_6b() {
    let fixture = fixture_path("qwen3-0.6b.onnx");
    if !fixture.exists() {
        eprintln!("SKIP: qwen3-0.6b.onnx not found (run download-fixtures.sh)");
        return;
    }
    let tmp = tempdir().unwrap();
    let output = tmp.path().join("qwen3.mlpackage");
    let result = ironmill_cmd()
        .args(["compile"])
        .arg(&fixture)
        .args(["-o", output.to_str().unwrap()])
        .output()
        .expect("failed to run ironmill compile");
    assert!(
        result.status.success(),
        "qwen3-0.6b compile failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.exists());
}
