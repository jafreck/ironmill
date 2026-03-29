//! Regression tracking via named baselines.
//!
//! Save benchmark results to `~/.ironmill/baselines/<name>.json` and compare
//! subsequent runs against them to detect performance regressions.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Threshold classification for a performance change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionStatus {
    /// Less than 15% slower — within noise.
    Ok,
    /// 15–25% slower — possible regression.
    Warn,
    /// More than 25% slower — likely regression.
    Fail,
    /// Any amount faster.
    Improved,
}

/// A single comparison between current and baseline results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionEntry {
    pub model: String,
    pub optimization: String,
    pub backend: String,
    pub metric: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub delta_pct: f64,
    pub status: RegressionStatus,
}

/// Full regression comparison report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    pub baseline_name: String,
    pub entries: Vec<RegressionEntry>,
}

/// Saved baseline data — a map of (model/opt/backend) -> metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineData {
    pub name: String,
    pub timestamp: String,
    pub entries: Vec<BaselineEntry>,
}

/// A single entry in a saved baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineEntry {
    pub model: String,
    pub optimization: String,
    pub backend: String,
    pub median_ms: f64,
    pub mean_ms: f64,
    pub p95_ms: f64,
    pub inferences_per_sec: f64,
    pub tflops: Option<f64>,
}

/// Get the baselines directory (~/.ironmill/baselines/).
fn baselines_dir() -> Result<PathBuf> {
    let home = dirs_path()?;
    let dir = home.join(".ironmill").join("baselines");
    Ok(dir)
}

/// Simple home directory detection.
fn dirs_path() -> Result<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .context("could not determine home directory")
}

/// Save results as a named baseline.
pub fn save_baseline(name: &str, entries: &[BaselineEntry]) -> Result<PathBuf> {
    let dir = baselines_dir()?;
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create baselines dir: {}", dir.display()))?;

    let data = BaselineData {
        name: name.to_string(),
        timestamp: chrono_timestamp(),
        entries: entries.to_vec(),
    };

    let path = dir.join(format!("{name}.json"));
    let json = serde_json::to_string_pretty(&data)?;
    fs::write(&path, json)
        .with_context(|| format!("failed to write baseline: {}", path.display()))?;

    Ok(path)
}

/// Load a named baseline.
pub fn load_baseline(name: &str) -> Result<BaselineData> {
    let path = baselines_dir()?.join(format!("{name}.json"));
    let content = fs::read_to_string(&path)
        .with_context(|| format!("baseline not found: {}", path.display()))?;
    let data: BaselineData = serde_json::from_str(&content)?;
    Ok(data)
}

/// Load the most recently modified baseline.
#[allow(dead_code)]
pub fn load_latest_baseline() -> Result<Option<BaselineData>> {
    let dir = baselines_dir()?;
    if !dir.exists() {
        return Ok(None);
    }

    let mut latest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            if let Ok(meta) = entry.metadata() {
                if let Ok(modified) = meta.modified() {
                    if latest.as_ref().is_none_or(|(t, _)| modified > *t) {
                        latest = Some((modified, path));
                    }
                }
            }
        }
    }

    match latest {
        Some((_, path)) => {
            let content = fs::read_to_string(&path)?;
            let data: BaselineData = serde_json::from_str(&content)?;
            Ok(Some(data))
        }
        None => Ok(None),
    }
}

/// List all saved baselines.
#[allow(dead_code)]
pub fn list_baselines() -> Result<Vec<String>> {
    let dir = baselines_dir()?;
    if !dir.exists() {
        return Ok(vec![]);
    }

    let mut names = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                names.push(stem.to_string());
            }
        }
    }
    names.sort();
    Ok(names)
}

/// Compare current results against a baseline.
pub fn compare_against_baseline(
    baseline: &BaselineData,
    current: &[BaselineEntry],
) -> RegressionReport {
    let mut entries = Vec::new();

    // Index baseline by (model, opt, backend)
    let baseline_map: HashMap<(&str, &str, &str), &BaselineEntry> = baseline
        .entries
        .iter()
        .map(|e| {
            (
                (
                    e.model.as_str(),
                    e.optimization.as_str(),
                    e.backend.as_str(),
                ),
                e,
            )
        })
        .collect();

    for cur in current {
        let key = (
            cur.model.as_str(),
            cur.optimization.as_str(),
            cur.backend.as_str(),
        );

        if let Some(base) = baseline_map.get(&key) {
            // Compare median latency
            let delta_pct = if base.median_ms > 0.0 {
                ((cur.median_ms - base.median_ms) / base.median_ms) * 100.0
            } else {
                0.0
            };

            entries.push(RegressionEntry {
                model: cur.model.clone(),
                optimization: cur.optimization.clone(),
                backend: cur.backend.clone(),
                metric: "median_ms".to_string(),
                baseline_value: base.median_ms,
                current_value: cur.median_ms,
                delta_pct,
                status: classify_regression(delta_pct),
            });

            // Compare inferences/sec (inverted — higher is better)
            let inf_delta = if base.inferences_per_sec > 0.0 {
                ((base.inferences_per_sec - cur.inferences_per_sec) / base.inferences_per_sec)
                    * 100.0
            } else {
                0.0
            };

            entries.push(RegressionEntry {
                model: cur.model.clone(),
                optimization: cur.optimization.clone(),
                backend: cur.backend.clone(),
                metric: "inf/sec".to_string(),
                baseline_value: base.inferences_per_sec,
                current_value: cur.inferences_per_sec,
                delta_pct: -inf_delta, // negative means slower
                status: classify_regression(inf_delta),
            });
        }
    }

    RegressionReport {
        baseline_name: baseline.name.clone(),
        entries,
    }
}

/// Classify a performance change percentage into a regression status.
fn classify_regression(delta_pct: f64) -> RegressionStatus {
    if delta_pct < 0.0 {
        RegressionStatus::Improved
    } else if delta_pct < 15.0 {
        RegressionStatus::Ok
    } else if delta_pct < 25.0 {
        RegressionStatus::Warn
    } else {
        RegressionStatus::Fail
    }
}

/// Format a regression report as a human-readable table.
pub fn format_regression_report(report: &RegressionReport) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    writeln!(
        out,
        "Regression check vs baseline {:?}",
        report.baseline_name
    )
    .unwrap();
    writeln!(out, "{}", "─".repeat(80)).unwrap();
    writeln!(
        out,
        "{:<24} {:<8} {:<12} {:>8} {:>8} {:>8} {:>8}",
        "Model + Opt", "Backend", "Metric", "Base", "Now", "Δ", "Status"
    )
    .unwrap();
    writeln!(
        out,
        "{:<24} {:<8} {:<12} {:>8} {:>8} {:>8} {:>8}",
        "───────────────────────",
        "───────",
        "───────────",
        "───────",
        "───────",
        "───────",
        "───────"
    )
    .unwrap();

    for entry in &report.entries {
        let label = format!("{} {}", entry.model, entry.optimization);
        let status_str = match entry.status {
            RegressionStatus::Ok => "✓ OK",
            RegressionStatus::Warn => "⚠ WARN",
            RegressionStatus::Fail => "✗ FAIL",
            RegressionStatus::Improved => "↑ IMPROVED",
        };

        writeln!(
            out,
            "{:<24} {:<8} {:<12} {:>8.2} {:>8.2} {:>+7.1}% {:>8}",
            label,
            entry.backend,
            entry.metric,
            entry.baseline_value,
            entry.current_value,
            entry.delta_pct,
            status_str
        )
        .unwrap();
    }

    out
}

/// Generate an ISO 8601-ish timestamp without external crate dependencies.
fn chrono_timestamp() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    // Simple unix timestamp as fallback (no chrono dependency)
    format!("{}", dur.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_regression() {
        assert_eq!(classify_regression(-5.0), RegressionStatus::Improved);
        assert_eq!(classify_regression(0.0), RegressionStatus::Ok);
        assert_eq!(classify_regression(10.0), RegressionStatus::Ok);
        assert_eq!(classify_regression(14.9), RegressionStatus::Ok);
        assert_eq!(classify_regression(15.0), RegressionStatus::Warn);
        assert_eq!(classify_regression(20.0), RegressionStatus::Warn);
        assert_eq!(classify_regression(24.9), RegressionStatus::Warn);
        assert_eq!(classify_regression(25.0), RegressionStatus::Fail);
        assert_eq!(classify_regression(50.0), RegressionStatus::Fail);
    }

    #[test]
    fn test_compare_against_baseline() {
        let baseline = BaselineData {
            name: "v0.1.0".to_string(),
            timestamp: "12345".to_string(),
            entries: vec![BaselineEntry {
                model: "MobileNetV2".to_string(),
                optimization: "fp16".to_string(),
                backend: "ane".to_string(),
                median_ms: 0.50,
                mean_ms: 0.52,
                p95_ms: 0.61,
                inferences_per_sec: 2000.0,
                tflops: Some(0.62),
            }],
        };

        let current = vec![BaselineEntry {
            model: "MobileNetV2".to_string(),
            optimization: "fp16".to_string(),
            backend: "ane".to_string(),
            median_ms: 0.48,
            mean_ms: 0.50,
            p95_ms: 0.58,
            inferences_per_sec: 2083.0,
            tflops: Some(0.65),
        }];

        let report = compare_against_baseline(&baseline, &current);
        assert_eq!(report.baseline_name, "v0.1.0");
        assert_eq!(report.entries.len(), 2); // median_ms + inf/sec

        // Median improved (lower is better)
        let median_entry = &report.entries[0];
        assert_eq!(median_entry.metric, "median_ms");
        assert!(median_entry.delta_pct < 0.0); // improved
        assert_eq!(median_entry.status, RegressionStatus::Improved);

        // Inf/sec improved (higher is better)
        let inf_entry = &report.entries[1];
        assert_eq!(inf_entry.metric, "inf/sec");
        assert_eq!(inf_entry.status, RegressionStatus::Improved);
    }

    #[test]
    fn test_compare_regression_detected() {
        let baseline = BaselineData {
            name: "v0.1.0".to_string(),
            timestamp: "12345".to_string(),
            entries: vec![BaselineEntry {
                model: "SqueezeNet".to_string(),
                optimization: "fp16".to_string(),
                backend: "ane".to_string(),
                median_ms: 0.30,
                mean_ms: 0.32,
                p95_ms: 0.38,
                inferences_per_sec: 3333.0,
                tflops: None,
            }],
        };

        let current = vec![BaselineEntry {
            model: "SqueezeNet".to_string(),
            optimization: "fp16".to_string(),
            backend: "ane".to_string(),
            median_ms: 0.38,
            mean_ms: 0.40,
            p95_ms: 0.45,
            inferences_per_sec: 2631.0,
            tflops: None,
        }];

        let report = compare_against_baseline(&baseline, &current);
        let median_entry = &report.entries[0];
        assert!(median_entry.delta_pct > 25.0); // ~26.7%
        assert_eq!(median_entry.status, RegressionStatus::Fail);
    }

    #[test]
    fn test_format_regression_report() {
        let report = RegressionReport {
            baseline_name: "v0.1.0".to_string(),
            entries: vec![
                RegressionEntry {
                    model: "MobileNetV2".to_string(),
                    optimization: "fp16".to_string(),
                    backend: "ane".to_string(),
                    metric: "median_ms".to_string(),
                    baseline_value: 0.50,
                    current_value: 0.48,
                    delta_pct: -4.0,
                    status: RegressionStatus::Improved,
                },
                RegressionEntry {
                    model: "SqueezeNet".to_string(),
                    optimization: "fp16".to_string(),
                    backend: "ane".to_string(),
                    metric: "median_ms".to_string(),
                    baseline_value: 0.30,
                    current_value: 0.38,
                    delta_pct: 26.7,
                    status: RegressionStatus::Fail,
                },
            ],
        };

        let output = format_regression_report(&report);
        assert!(output.contains("v0.1.0"));
        assert!(output.contains("MobileNetV2"));
        assert!(output.contains("↑ IMPROVED"));
        assert!(output.contains("✗ FAIL"));
    }

    #[test]
    fn test_save_load_baseline_roundtrip() {
        // Use a temp directory to avoid polluting ~/.ironmill
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test-baseline.json");

        let data = BaselineData {
            name: "test".to_string(),
            timestamp: "12345".to_string(),
            entries: vec![BaselineEntry {
                model: "TestModel".to_string(),
                optimization: "fp16".to_string(),
                backend: "ane".to_string(),
                median_ms: 1.0,
                mean_ms: 1.1,
                p95_ms: 1.5,
                inferences_per_sec: 1000.0,
                tflops: Some(0.5),
            }],
        };

        let json = serde_json::to_string_pretty(&data).unwrap();
        fs::write(&path, &json).unwrap();

        let loaded: BaselineData =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(loaded.name, "test");
        assert_eq!(loaded.entries.len(), 1);
        assert!((loaded.entries[0].median_ms - 1.0).abs() < 0.01);
    }
}
