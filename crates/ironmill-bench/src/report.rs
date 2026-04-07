use std::fmt::Write;

use serde::Serialize;

use crate::config::Settings;
use crate::hardware::HardwareInfo;
use crate::inference::{MemoryMetrics, UtilizationMetrics};
use crate::power::EnergyMetrics;
use crate::stats::{AggregatedResult, SignificanceResult};

/// Output format selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    Table,
    Json,
    Csv,
    Markdown,
}

/// A row in the report, representing one (model, optimization, backend) combination.
#[derive(Debug, Clone, Serialize)]
pub struct ReportRow {
    pub model: String,
    pub optimization: String,
    pub backend: String,
    /// KV cache quantization mode (TurboQuant).
    pub kv_quant: String,
    pub result: AggregatedResult,
    pub significance: Option<SignificanceResult>,
    /// Energy efficiency metrics (when --power is used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy: Option<EnergyMetrics>,
    /// ANE utilization breakdown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utilization: Option<UtilizationSummary>,
    /// Memory footprint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemorySummary>,
    /// Model load time in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_time_ms: Option<f64>,
}

/// Summary of utilization metrics for serialization.
#[derive(Debug, Clone, Serialize)]
pub struct UtilizationSummary {
    pub predict_pct: f64,
    pub dispatch_overhead_ms: f64,
}

/// Summary of memory metrics for serialization.
#[derive(Debug, Clone, Serialize)]
pub struct MemorySummary {
    pub rss_after_load_mb: f64,
    pub peak_rss_mb: f64,
    pub rss_growth_mb: f64,
    pub model_file_size_mb: f64,
    pub efficiency_ratio: f64,
}

impl UtilizationSummary {
    /// Build a summary from raw utilization metrics.
    pub fn from_metrics(m: &UtilizationMetrics) -> Self {
        Self {
            predict_pct: m.utilization_pct(),
            dispatch_overhead_ms: m.dispatch_overhead_ms(),
        }
    }
}

impl MemorySummary {
    /// Build a summary from raw memory metrics.
    pub fn from_metrics(m: &MemoryMetrics) -> Self {
        Self {
            rss_after_load_mb: m.rss_after_load as f64 / (1024.0 * 1024.0),
            peak_rss_mb: m.peak_rss as f64 / (1024.0 * 1024.0),
            rss_growth_mb: m.rss_growth_mb(),
            model_file_size_mb: m.model_file_size as f64 / (1024.0 * 1024.0),
            efficiency_ratio: m.efficiency_ratio(),
        }
    }
}

/// Full benchmark report.
#[derive(Debug, Clone, Serialize)]
pub struct BenchReport {
    pub rows: Vec<ReportRow>,
    pub settings: Settings,
}

/// Structured JSON output with versioned schema.
#[derive(Debug, Clone, Serialize)]
pub struct StructuredReport {
    pub version: &'static str,
    pub timestamp: String,
    pub hardware: HardwareInfo,
    pub ironmill_version: String,
    pub results: Vec<StructuredResult>,
}

/// A single result in the structured JSON output.
#[derive(Debug, Clone, Serialize)]
pub struct StructuredResult {
    pub model: String,
    pub optimization: String,
    pub backend: String,
    /// KV cache quantization mode (TurboQuant).
    pub kv_quant: String,
    pub iterations: usize,
    pub warmup: usize,
    pub runs: usize,
    pub latency: LatencyStats,
    pub throughput: ThroughputStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy: Option<EnergyMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utilization: Option<UtilizationSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemorySummary>,
}

/// Latency statistics for a benchmark result.
#[derive(Debug, Clone, Serialize)]
pub struct LatencyStats {
    pub mean_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub stddev_ms: f64,
    pub cv: f64,
}

/// Throughput statistics for a benchmark result.
#[derive(Debug, Clone, Serialize)]
pub struct ThroughputStats {
    pub inferences_per_sec: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tflops: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tok_per_sec: Option<f64>,
}

/// Build a structured JSON report from a BenchReport.
pub fn build_structured_report(report: &BenchReport) -> StructuredReport {
    let hardware = HardwareInfo::detect();
    let timestamp = {
        use std::time::SystemTime;
        let dur = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        format!("{}", dur.as_secs())
    };

    let results = report
        .rows
        .iter()
        .map(|row| {
            let r = &row.result.pooled;
            StructuredResult {
                model: row.model.clone(),
                optimization: row.optimization.clone(),
                backend: row.backend.clone(),
                kv_quant: row.kv_quant.clone(),
                iterations: report.settings.iterations,
                warmup: report.settings.warmup,
                runs: report.settings.runs,
                latency: LatencyStats {
                    mean_ms: r.mean,
                    median_ms: r.median,
                    p95_ms: r.p95,
                    p99_ms: r.p99,
                    stddev_ms: r.stddev,
                    cv: r.cv,
                },
                throughput: ThroughputStats {
                    inferences_per_sec: r.inferences_per_sec,
                    tflops: r.tflops,
                    tokens_per_sec: r.tokens_per_sec,
                    ttft_ms: r.ttft_ms,
                    decode_tok_per_sec: r.decode_tok_per_sec,
                },
                energy: row.energy.clone(),
                utilization: row.utilization.clone(),
                memory: row.memory.clone(),
            }
        })
        .collect();

    StructuredReport {
        version: "1",
        timestamp,
        hardware,
        ironmill_version: env!("CARGO_PKG_VERSION").to_string(),
        results,
    }
}

/// Format the report in the specified output format.
pub fn format_report(report: &BenchReport, format: OutputFormat) -> String {
    match format {
        OutputFormat::Table => format_table(report),
        OutputFormat::Json => format_structured_json(report),
        OutputFormat::Csv => format_csv(report),
        OutputFormat::Markdown => format_markdown(report),
    }
}

fn significance_stars(sig: &Option<SignificanceResult>) -> &'static str {
    match sig {
        Some(s) if s.significant && s.p_value < 0.001 => " ***",
        Some(s) if s.significant && s.p_value < 0.01 => " **",
        Some(s) if s.significant && s.p_value < 0.05 => " *",
        _ => "",
    }
}

/// Format as a human-readable table (default output).
///
/// When multiple backends are present, displays a cross-tabulated grid with
/// backends as columns, a speedup column (vs CPU FP32 baseline), and a
/// significance indicator.
fn format_table(report: &BenchReport) -> String {
    let backends: Vec<String> = {
        let mut b: Vec<String> = report
            .rows
            .iter()
            .map(|r| r.backend.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        // Ensure a consistent order: cpu, gpu, ane, all, then anything else.
        let order = ["cpu", "gpu", "ane", "all"];
        b.sort_by_key(|name| order.iter().position(|&o| o == name).unwrap_or(order.len()));
        b
    };

    let multi_backend = backends.len() > 1 && !backends.contains(&"all".to_string());

    if multi_backend {
        format_cross_table(report, &backends)
    } else {
        format_flat_table(report)
    }
}

/// Cross-tabulated table: optimizations × backends with ANE speedup.
fn format_cross_table(report: &BenchReport, backends: &[String]) -> String {
    use std::collections::HashMap;

    let mut out = String::new();

    // Group rows by model.
    let mut models: Vec<String> = Vec::new();
    for row in &report.rows {
        if !models.contains(&row.model) {
            models.push(row.model.clone());
        }
    }

    // Index rows by (model, optimization, backend).
    let mut index: HashMap<(&str, &str, &str), &ReportRow> = HashMap::new();
    for row in &report.rows {
        index.insert((&row.model, &row.optimization, &row.backend), row);
    }

    let has_ane = backends.contains(&"ane".to_string());

    for model in &models {
        if !out.is_empty() {
            writeln!(out).unwrap();
        }

        writeln!(
            out,
            "{} — {} runs × {} iterations (median latency, inf/sec)",
            model, report.settings.runs, report.settings.iterations
        )
        .unwrap();

        // Header.
        let sep = "─".repeat(60 + backends.len() * 10);
        writeln!(out, "{sep}").unwrap();
        write!(out, "{:<18}", "Optimization").unwrap();
        for b in backends {
            write!(out, "  {:>11}", b.to_uppercase()).unwrap();
        }
        if has_ane {
            write!(out, "  {:>12}", "ANE speedup").unwrap();
        }
        writeln!(out).unwrap();

        write!(out, "{:<18}", "─────────────────").unwrap();
        for _ in backends {
            write!(out, "  {:>11}", "───────────").unwrap();
        }
        if has_ane {
            write!(out, "  {:>12}", "────────────").unwrap();
        }
        writeln!(out).unwrap();

        // Collect unique optimizations in order.
        let mut opts: Vec<String> = Vec::new();
        for row in &report.rows {
            if row.model == *model && !opts.contains(&row.optimization) {
                opts.push(row.optimization.clone());
            }
        }

        // Find the CPU baseline median for speedup calculation.
        let cpu_baseline_median = opts
            .first()
            .and_then(|first_opt| index.get(&(model.as_str(), first_opt.as_str(), "cpu")))
            .map(|r| r.result.pooled.median);

        for opt in &opts {
            write!(out, "{:<18}", opt).unwrap();
            for b in backends {
                if let Some(row) = index.get(&(model.as_str(), opt.as_str(), b.as_str())) {
                    let stars = significance_stars(&row.significance);
                    write!(out, "  {:>5.1}ms{:<3}", row.result.pooled.median, stars).unwrap();
                } else {
                    write!(out, "  {:>11}", "—").unwrap();
                }
            }
            if has_ane {
                if let (Some(cpu_base), Some(ane_row)) = (
                    cpu_baseline_median,
                    index.get(&(model.as_str(), opt.as_str(), "ane")),
                ) {
                    let ane_med = ane_row.result.pooled.median;
                    if ane_med > 0.0 {
                        write!(out, "  {:>10.1}×", cpu_base / ane_med).unwrap();
                    } else {
                        write!(out, "  {:>12}", "—").unwrap();
                    }
                } else {
                    write!(out, "  {:>12}", "—").unwrap();
                }
            }
            writeln!(out).unwrap();
        }
    }

    out
}

/// Flat table (single backend or "all").
fn format_flat_table(report: &BenchReport) -> String {
    let mut out = String::new();

    let mut current_model: Option<&str> = None;

    for row in &report.rows {
        if current_model != Some(&row.model) {
            if current_model.is_some() {
                writeln!(out).unwrap();
            }
            current_model = Some(&row.model);

            writeln!(
                out,
                "{} — {} runs × {} iterations",
                row.model, report.settings.runs, report.settings.iterations
            )
            .unwrap();
            writeln!(
                out,
                "──────────────────────────────────────────────────────"
            )
            .unwrap();
            writeln!(
                out,
                "{:<18} {:>10}  {:>7}  {:>7}  {:>7}  {:>8}  {:>8}",
                "Configuration", "mean±sd", "median", "p95", "p99", "inf/sec", "TFLOPS"
            )
            .unwrap();
            writeln!(
                out,
                "{:<18} {:>10}  {:>7}  {:>7}  {:>7}  {:>8}  {:>8}",
                "─────────────────",
                "─────────",
                "───────",
                "───────",
                "───────",
                "────────",
                "────────"
            )
            .unwrap();
        }

        let r = &row.result.pooled;
        let stars = significance_stars(&row.significance);
        let label = if row.backend == "all" || row.backend.is_empty() {
            row.optimization.clone()
        } else {
            format!("{}/{}", row.optimization, row.backend)
        };
        let label = if row.kv_quant != "none" && !row.kv_quant.is_empty() {
            format!("{} [{}]", label, row.kv_quant)
        } else {
            label
        };

        let tflops_str = r.tflops.map_or("—".to_string(), |t| format!("{:.2}", t));

        writeln!(
            out,
            "{:<18} {:>5.1}±{:.1}ms  {:>5.1}ms  {:>5.1}ms  {:>5.1}ms  {:>8.0}  {:>8}{}",
            label,
            r.mean,
            r.stddev,
            r.median,
            r.p95,
            r.p99,
            r.inferences_per_sec,
            tflops_str,
            stars
        )
        .unwrap();
    }

    out
}

/// Format as structured JSON with versioned schema and hardware info.
fn format_structured_json(report: &BenchReport) -> String {
    let structured = build_structured_report(report);
    serde_json::to_string_pretty(&structured).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
}

/// Format as CSV.
fn format_csv(report: &BenchReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "model,optimization,backend,kv_quant,mean,stddev,median,p95,p99,min,max,cv,inf_per_sec,tflops,p_value"
    )
    .unwrap();

    for row in &report.rows {
        let r = &row.result.pooled;
        let p_value = row
            .significance
            .as_ref()
            .map(|s| format!("{:.6}", s.p_value))
            .unwrap_or_default();
        let tflops = r.tflops.map(|t| format!("{:.4}", t)).unwrap_or_default();

        writeln!(
            out,
            "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.2},{},{}",
            row.model,
            row.optimization,
            row.backend,
            row.kv_quant,
            r.mean,
            r.stddev,
            r.median,
            r.p95,
            r.p99,
            r.min,
            r.max,
            r.cv,
            r.inferences_per_sec,
            tflops,
            p_value
        )
        .unwrap();
    }

    out
}

/// Format as Markdown.
fn format_markdown(report: &BenchReport) -> String {
    let mut out = String::new();

    let mut current_model: Option<&str> = None;

    for row in &report.rows {
        if current_model != Some(&row.model) {
            if current_model.is_some() {
                writeln!(out).unwrap();
            }
            current_model = Some(&row.model);

            writeln!(
                out,
                "### {} — {} runs × {} iterations",
                row.model, report.settings.runs, report.settings.iterations
            )
            .unwrap();
            writeln!(out).unwrap();
            writeln!(
                out,
                "| Configuration | mean±sd | median | p95 | p99 | sig |"
            )
            .unwrap();
            writeln!(out, "|---|---|---|---|---|---|").unwrap();
        }

        let r = &row.result.pooled;
        let stars = significance_stars(&row.significance).trim();
        let label = if row.backend == "all" || row.backend.is_empty() {
            row.optimization.clone()
        } else {
            format!("{}/{}", row.optimization, row.backend)
        };
        let label = if row.kv_quant != "none" && !row.kv_quant.is_empty() {
            format!("{} [{}]", label, row.kv_quant)
        } else {
            label
        };

        writeln!(
            out,
            "| {} | {:.1}±{:.1}ms | {:.1}ms | {:.1}ms | {:.1}ms | {} |",
            label, r.mean, r.stddev, r.median, r.p95, r.p99, stars
        )
        .unwrap();
    }

    out
}

/// Entry for the GPU quantized inference comparison table.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GpuComparisonEntry {
    pub config: String,
    pub perplexity: Option<f64>,
    pub tok_per_sec: f64,
    pub ms_per_tok: f64,
    pub gpu_mb: f64,
}

/// Format a GPU quantized inference comparison table.
///
/// The first entry is used as the baseline for ΔPPL and ΔMem columns.
#[allow(dead_code)]
pub fn format_gpu_comparison_table(entries: &[GpuComparisonEntry]) -> String {
    let mut out = String::new();

    writeln!(
        out,
        "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Config", "PPL", "tok/s", "ms/tok", "GPU MB", "ΔPPL", "ΔMem"
    )
    .unwrap();
    writeln!(out, "{}", "─".repeat(72)).unwrap();

    let baseline_ppl = entries.first().and_then(|e| e.perplexity);
    let baseline_mem = entries.first().map(|e| e.gpu_mb);

    for (i, entry) in entries.iter().enumerate() {
        let ppl_str = entry
            .perplexity
            .map_or("—".to_string(), |p| format!("{:.1}", p));

        let delta_ppl = if i == 0 {
            "—".to_string()
        } else {
            match (entry.perplexity, baseline_ppl) {
                (Some(p), Some(bp)) if bp > 0.0 => {
                    format!("{:+.1}%", (p - bp) / bp * 100.0)
                }
                _ => "—".to_string(),
            }
        };

        let delta_mem = if i == 0 {
            "—".to_string()
        } else {
            match baseline_mem {
                Some(bm) if bm > 0.0 => {
                    format!("{:+.0}%", (entry.gpu_mb - bm) / bm * 100.0)
                }
                _ => "—".to_string(),
            }
        };

        writeln!(
            out,
            "{:<20} {:>8} {:>8.1} {:>8.1} {:>8.0} {:>8} {:>8}",
            entry.config,
            ppl_str,
            entry.tok_per_sec,
            entry.ms_per_tok,
            entry.gpu_mb,
            delta_ppl,
            delta_mem
        )
        .unwrap();
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::{aggregate_runs, compute_stats};

    fn sample_report() -> BenchReport {
        let lat1: Vec<f64> = (0..100).map(|i| 5.0 + (i as f64) * 0.01).collect();
        let lat2: Vec<f64> = (0..100).map(|i| 3.0 + (i as f64) * 0.01).collect();

        let run1 = compute_stats("baseline", &lat1);
        let run2 = compute_stats("fp16", &lat2);

        let agg1 = aggregate_runs("baseline", &[run1]);
        let agg2 = aggregate_runs("fp16", &[run2]);

        BenchReport {
            rows: vec![
                ReportRow {
                    model: "MobileNetV2".to_string(),
                    optimization: "baseline".to_string(),
                    backend: "all".to_string(),
                    kv_quant: "none".to_string(),
                    result: agg1,
                    significance: None,
                    energy: None,
                    utilization: None,
                    memory: None,
                    load_time_ms: None,
                },
                ReportRow {
                    model: "MobileNetV2".to_string(),
                    optimization: "fp16".to_string(),
                    backend: "all".to_string(),
                    kv_quant: "none".to_string(),
                    result: agg2,
                    significance: Some(SignificanceResult {
                        significant: true,
                        p_value: 0.0001,
                        effect_size: 2.5,
                        ci_lower: 1.5,
                        ci_upper: 2.5,
                        method: "welch_t_test".to_string(),
                    }),
                    energy: None,
                    utilization: None,
                    memory: None,
                    load_time_ms: None,
                },
            ],
            settings: Settings {
                iterations: 1,
                warmup: 0,
                runs: 1,
                backends: vec!["all".to_string()],
            },
        }
    }

    #[test]
    fn test_format_table() {
        let report = sample_report();
        let table = format_report(&report, OutputFormat::Table);
        assert!(table.contains("MobileNetV2"));
        assert!(table.contains("baseline"));
        assert!(table.contains("fp16"));
        assert!(table.contains("***"));
        assert!(table.contains("Configuration"));
    }

    #[test]
    fn test_format_json() {
        let report = sample_report();
        let json = format_report(&report, OutputFormat::Json);
        assert!(json.starts_with('{'));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["version"], "1");
        assert!(parsed["hardware"].is_object());
        assert!(parsed["results"].is_array());
        assert_eq!(parsed["results"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_format_csv() {
        let report = sample_report();
        let csv = format_report(&report, OutputFormat::Csv);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "model,optimization,backend,kv_quant,mean,stddev,median,p95,p99,min,max,cv,inf_per_sec,tflops,p_value"
        );
        assert_eq!(lines.len(), 3); // header + 2 data rows
        assert!(lines[1].starts_with("MobileNetV2,baseline,"));
        assert!(lines[2].starts_with("MobileNetV2,fp16,"));
    }

    #[test]
    fn test_format_markdown() {
        let report = sample_report();
        let md = format_report(&report, OutputFormat::Markdown);
        assert!(md.contains("### MobileNetV2"));
        assert!(md.contains("| Configuration |"));
        assert!(md.contains("|---|"));
        assert!(md.contains("***"));
    }

    #[test]
    fn test_significance_stars_levels() {
        assert_eq!(significance_stars(&None), "");
        assert_eq!(
            significance_stars(&Some(SignificanceResult {
                significant: true,
                p_value: 0.0001,
                effect_size: 0.0,
                ci_lower: 0.0,
                ci_upper: 0.0,
                method: String::new(),
            })),
            " ***"
        );
        assert_eq!(
            significance_stars(&Some(SignificanceResult {
                significant: true,
                p_value: 0.005,
                effect_size: 0.0,
                ci_lower: 0.0,
                ci_upper: 0.0,
                method: String::new(),
            })),
            " **"
        );
        assert_eq!(
            significance_stars(&Some(SignificanceResult {
                significant: true,
                p_value: 0.03,
                effect_size: 0.0,
                ci_lower: 0.0,
                ci_upper: 0.0,
                method: String::new(),
            })),
            " *"
        );
        assert_eq!(
            significance_stars(&Some(SignificanceResult {
                significant: false,
                p_value: 0.5,
                effect_size: 0.0,
                ci_lower: 0.0,
                ci_upper: 0.0,
                method: String::new(),
            })),
            ""
        );
    }
}
