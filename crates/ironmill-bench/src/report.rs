use std::fmt::Write;

use serde::Serialize;

use crate::config::Settings;
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
    pub result: AggregatedResult,
    pub significance: Option<SignificanceResult>,
}

/// Full benchmark report.
#[derive(Debug, Clone, Serialize)]
pub struct BenchReport {
    pub rows: Vec<ReportRow>,
    pub settings: Settings,
}

/// Format the report in the specified output format.
pub fn format_report(report: &BenchReport, format: OutputFormat) -> String {
    match format {
        OutputFormat::Table => format_table(report),
        OutputFormat::Json => format_json(report),
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
fn format_table(report: &BenchReport) -> String {
    let mut out = String::new();

    // Group rows by model
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
                "{:<18} {:>10}  {:>7}  {:>7}  {:>7}",
                "Configuration", "mean±sd", "median", "p95", "p99"
            )
            .unwrap();
            writeln!(
                out,
                "{:<18} {:>10}  {:>7}  {:>7}  {:>7}",
                "─────────────────", "─────────", "───────", "───────", "───────"
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

        writeln!(
            out,
            "{:<18} {:>5.1}±{:.1}ms  {:>5.1}ms  {:>5.1}ms  {:>5.1}ms{}",
            label, r.mean, r.stddev, r.median, r.p95, r.p99, stars
        )
        .unwrap();
    }

    out
}

/// Format as JSON.
fn format_json(report: &BenchReport) -> String {
    serde_json::to_string_pretty(report).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
}

/// Format as CSV.
fn format_csv(report: &BenchReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "model,optimization,backend,mean,stddev,median,p95,p99,min,max,cv,p_value"
    )
    .unwrap();

    for row in &report.rows {
        let r = &row.result.pooled;
        let p_value = row
            .significance
            .as_ref()
            .map(|s| format!("{:.6}", s.p_value))
            .unwrap_or_default();

        writeln!(
            out,
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{}",
            row.model,
            row.optimization,
            row.backend,
            r.mean,
            r.stddev,
            r.median,
            r.p95,
            r.p99,
            r.min,
            r.max,
            r.cv,
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

        writeln!(
            out,
            "| {} | {:.1}±{:.1}ms | {:.1}ms | {:.1}ms | {:.1}ms | {} |",
            label, r.mean, r.stddev, r.median, r.p95, r.p99, stars
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
                    result: agg1,
                    significance: None,
                },
                ReportRow {
                    model: "MobileNetV2".to_string(),
                    optimization: "fp16".to_string(),
                    backend: "all".to_string(),
                    result: agg2,
                    significance: Some(SignificanceResult {
                        significant: true,
                        p_value: 0.0001,
                        effect_size: 2.5,
                        ci_lower: 1.5,
                        ci_upper: 2.5,
                        method: "welch_t_test".to_string(),
                    }),
                },
            ],
            settings: Settings {
                iterations: 200,
                warmup: 20,
                runs: 3,
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
        assert!(parsed["rows"].is_array());
        assert_eq!(parsed["rows"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_format_csv() {
        let report = sample_report();
        let csv = format_report(&report, OutputFormat::Csv);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "model,optimization,backend,mean,stddev,median,p95,p99,min,max,cv,p_value"
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
