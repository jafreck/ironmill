//! Energy and power measurement via macOS `powermetrics`.
//!
//! Samples CPU, GPU, and ANE power draw during benchmark runs to compute
//! per-watt efficiency metrics. Requires `sudo` access for `powermetrics`;
//! gracefully degrades when unavailable.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Power samples collected during a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerMetrics {
    /// Average CPU package power in watts.
    pub cpu_watts: f64,
    /// Average GPU power in watts.
    pub gpu_watts: f64,
    /// Average ANE power in watts (0.0 if unavailable).
    pub ane_watts: f64,
    /// Combined package power in watts.
    pub package_watts: f64,
    /// Number of samples collected.
    pub sample_count: usize,
}

/// Derived energy efficiency metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyMetrics {
    /// Inferences per watt (inferences/sec ÷ package_watts).
    pub inferences_per_watt: f64,
    /// Tokens per watt (tok/s ÷ package_watts), if applicable.
    pub tokens_per_watt: Option<f64>,
    /// TFLOPS per watt.
    pub tflops_per_watt: Option<f64>,
    /// Joules per inference (package_watts × median_sec).
    pub joules_per_inference: f64,
    /// Power draw during measurement.
    pub power: PowerMetrics,
    /// Idle power sampled before benchmark (for delta calculation).
    pub idle_power: Option<PowerMetrics>,
    /// Delta watts (loaded - idle).
    pub delta_watts: Option<f64>,
}

/// A handle to a background powermetrics sampling process.
pub struct PowerSampler {
    child: Option<Child>,
    receiver: Option<mpsc::Receiver<PowerSample>>,
    handle: Option<thread::JoinHandle<()>>,
}

#[derive(Debug, Clone)]
struct PowerSample {
    cpu_watts: f64,
    gpu_watts: f64,
    ane_watts: f64,
    package_watts: f64,
}

impl PowerSampler {
    /// Start sampling power metrics in the background.
    ///
    /// Returns `None` if `powermetrics` is not available or requires sudo.
    pub fn start(interval_ms: u64, max_samples: usize) -> Option<Self> {
        // Try to start powermetrics
        let mut child = Command::new("sudo")
            .args([
                "-n", // non-interactive (fail if password needed)
                "powermetrics",
                "--samplers",
                "cpu_power,gpu_power,ane_power",
                "-i",
                &interval_ms.to_string(),
                "-n",
                &max_samples.to_string(),
                "--format",
                "plist",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;

        let stdout = child.stdout.take()?;
        let (tx, rx) = mpsc::channel();

        let handle = thread::spawn(move || {
            // Read plist output and parse power values
            // powermetrics with --format plist outputs a series of plist dicts
            let reader = BufReader::new(stdout);
            let mut current_plist = String::new();
            let mut in_plist = false;

            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => break,
                };

                if line.contains("<?xml") || line.contains("<plist") {
                    in_plist = true;
                    current_plist.clear();
                }

                if in_plist {
                    current_plist.push_str(&line);
                    current_plist.push('\n');
                }

                if line.contains("</plist>") {
                    in_plist = false;
                    if let Some(sample) = parse_power_plist(&current_plist) {
                        if tx.send(sample).is_err() {
                            break;
                        }
                    }
                    current_plist.clear();
                }
            }
        });

        Some(PowerSampler {
            child: Some(child),
            receiver: Some(rx),
            handle: Some(handle),
        })
    }

    /// Collect all samples and compute average power metrics.
    pub fn finish(mut self) -> Option<PowerMetrics> {
        // Kill the child process
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        // Wait for the reader thread
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }

        // Collect all samples
        let rx = self.receiver.take()?;
        let samples: Vec<PowerSample> = rx.try_iter().collect();

        if samples.is_empty() {
            return None;
        }

        let n = samples.len() as f64;
        Some(PowerMetrics {
            cpu_watts: samples.iter().map(|s| s.cpu_watts).sum::<f64>() / n,
            gpu_watts: samples.iter().map(|s| s.gpu_watts).sum::<f64>() / n,
            ane_watts: samples.iter().map(|s| s.ane_watts).sum::<f64>() / n,
            package_watts: samples.iter().map(|s| s.package_watts).sum::<f64>() / n,
            sample_count: samples.len(),
        })
    }
}

impl Drop for PowerSampler {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Sample idle power for a given duration.
pub fn sample_idle_power(duration: Duration) -> Option<PowerMetrics> {
    let samples = (duration.as_millis() / 100).max(1) as usize;
    let sampler = PowerSampler::start(100, samples)?;
    thread::sleep(duration);
    sampler.finish()
}

/// Check whether power sampling is available (powermetrics + sudo).
pub fn is_power_available() -> bool {
    Command::new("sudo")
        .args([
            "-n",
            "powermetrics",
            "--samplers",
            "cpu_power",
            "-i",
            "100",
            "-n",
            "1",
            "--format",
            "plist",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Compute energy efficiency metrics from power measurements and throughput.
pub fn compute_energy_metrics(
    power: PowerMetrics,
    idle_power: Option<PowerMetrics>,
    inferences_per_sec: f64,
    median_sec: f64,
    tflops: Option<f64>,
    tokens_per_sec: Option<f64>,
) -> EnergyMetrics {
    let package_watts = power.package_watts.max(0.001); // avoid div by zero

    let delta_watts = idle_power
        .as_ref()
        .map(|idle| power.package_watts - idle.package_watts);

    EnergyMetrics {
        inferences_per_watt: inferences_per_sec / package_watts,
        tokens_per_watt: tokens_per_sec.map(|t| t / package_watts),
        tflops_per_watt: tflops.map(|t| t / package_watts),
        joules_per_inference: package_watts * median_sec,
        power,
        idle_power,
        delta_watts,
    }
}

/// Parse power values from a powermetrics plist output.
fn parse_power_plist(plist: &str) -> Option<PowerSample> {
    // powermetrics plist format has keys like:
    //   <key>CPU Power</key> or <key>Package Power</key> or <key>combined_power (mW)</key>
    // The exact keys vary by macOS version. We look for common patterns.

    let mut cpu_watts = 0.0;
    let mut gpu_watts = 0.0;
    let mut ane_watts = 0.0;
    let mut package_watts = 0.0;
    let mut found_any = false;

    let lines: Vec<&str> = plist.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let line = line.trim();
        if !line.starts_with("<key>") {
            continue;
        }

        let key = line
            .strip_prefix("<key>")
            .and_then(|s| s.strip_suffix("</key>"))
            .unwrap_or("");

        let next_value = lines.get(i + 1).and_then(|next| {
            let next = next.trim();
            // Parse <real>...</real> or <integer>...</integer>
            next.strip_prefix("<real>")
                .and_then(|s| s.strip_suffix("</real>"))
                .and_then(|s| s.parse::<f64>().ok())
                .or_else(|| {
                    next.strip_prefix("<integer>")
                        .and_then(|s| s.strip_suffix("</integer>"))
                        .and_then(|s| s.parse::<f64>().ok())
                })
        });

        if let Some(val) = next_value {
            let key_lower = key.to_lowercase();
            // Convert milliwatts to watts if needed
            let (watts, _is_mw) = if key_lower.contains("(mw)") {
                (val / 1000.0, true)
            } else {
                (val, false)
            };

            if key_lower.contains("package power") || key_lower.contains("combined_power") {
                package_watts = watts;
                found_any = true;
            } else if key_lower.contains("cpu power")
                || (key_lower.contains("cpu") && key_lower.contains("power"))
            {
                cpu_watts = watts;
                found_any = true;
            } else if key_lower.contains("gpu power")
                || (key_lower.contains("gpu") && key_lower.contains("power"))
            {
                gpu_watts = watts;
                found_any = true;
            } else if key_lower.contains("ane power") || key_lower.contains("neural") {
                ane_watts = watts;
                found_any = true;
            }
        }
    }

    if !found_any {
        return None;
    }

    // If no explicit package power, sum components
    if package_watts == 0.0 {
        package_watts = cpu_watts + gpu_watts + ane_watts;
    }

    Some(PowerSample {
        cpu_watts,
        gpu_watts,
        ane_watts,
        package_watts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_power_plist_basic() {
        let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN">
<plist version="1.0">
<dict>
    <key>CPU Power</key>
    <real>5.2</real>
    <key>GPU Power</key>
    <real>3.1</real>
    <key>ANE Power</key>
    <real>2.0</real>
    <key>Package Power</key>
    <real>12.5</real>
</dict>
</plist>"#;

        let sample = parse_power_plist(plist).unwrap();
        assert!((sample.cpu_watts - 5.2).abs() < 0.01);
        assert!((sample.gpu_watts - 3.1).abs() < 0.01);
        assert!((sample.ane_watts - 2.0).abs() < 0.01);
        assert!((sample.package_watts - 12.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_power_plist_milliwatts() {
        let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>combined_power (mW)</key>
    <integer>8500</integer>
    <key>cpu_power (mW)</key>
    <integer>5000</integer>
    <key>gpu_power (mW)</key>
    <integer>2500</integer>
</dict>
</plist>"#;

        let sample = parse_power_plist(plist).unwrap();
        assert!((sample.cpu_watts - 5.0).abs() < 0.01);
        assert!((sample.gpu_watts - 2.5).abs() < 0.01);
        assert!((sample.package_watts - 8.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_power_plist_empty() {
        let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>unrelated</key>
    <string>hello</string>
</dict>
</plist>"#;
        assert!(parse_power_plist(plist).is_none());
    }

    #[test]
    fn test_compute_energy_metrics() {
        let power = PowerMetrics {
            cpu_watts: 5.0,
            gpu_watts: 3.0,
            ane_watts: 2.0,
            package_watts: 10.0,
            sample_count: 20,
        };
        let idle = PowerMetrics {
            cpu_watts: 1.0,
            gpu_watts: 0.5,
            ane_watts: 0.0,
            package_watts: 2.0,
            sample_count: 20,
        };
        let metrics = compute_energy_metrics(
            power,
            Some(idle),
            1000.0,    // 1000 inf/sec
            0.001,     // 1ms median
            Some(0.5), // 0.5 TFLOPS
            None,
        );
        assert!((metrics.inferences_per_watt - 100.0).abs() < 0.01);
        assert!((metrics.joules_per_inference - 0.01).abs() < 0.001);
        assert!((metrics.tflops_per_watt.unwrap() - 0.05).abs() < 0.001);
        assert!((metrics.delta_watts.unwrap() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_energy_metrics_no_idle() {
        let power = PowerMetrics {
            cpu_watts: 5.0,
            gpu_watts: 0.0,
            ane_watts: 0.0,
            package_watts: 5.0,
            sample_count: 10,
        };
        let metrics = compute_energy_metrics(power, None, 500.0, 0.002, None, Some(100.0));
        assert!((metrics.inferences_per_watt - 100.0).abs() < 0.01);
        assert!((metrics.tokens_per_watt.unwrap() - 20.0).abs() < 0.01);
        assert!(metrics.tflops_per_watt.is_none());
        assert!(metrics.delta_watts.is_none());
    }

    #[test]
    fn test_power_sampler_unavailable() {
        // On most CI/test environments, sudo powermetrics won't work
        // Just verify it returns None gracefully
        // (This test may succeed locally with sudo configured)
        let result = PowerSampler::start(100, 1);
        // Either starts successfully or returns None - both are valid
        if let Some(sampler) = result {
            let _ = sampler.finish();
        }
    }

    #[test]
    fn test_parse_power_package_fallback() {
        // When no package power, should sum components
        let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>CPU Power</key>
    <real>4.0</real>
    <key>GPU Power</key>
    <real>3.0</real>
    <key>ANE Power</key>
    <real>1.0</real>
</dict>
</plist>"#;
        let sample = parse_power_plist(plist).unwrap();
        assert!((sample.package_watts - 8.0).abs() < 0.01);
    }
}
