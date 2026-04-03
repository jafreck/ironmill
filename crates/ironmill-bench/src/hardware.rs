//! Hardware detection for Apple Silicon systems.
//!
//! Reads chip model, core counts, RAM, and macOS version using system commands
//! and `sysctl` queries.

use std::process::Command;

use serde::{Deserialize, Serialize};

/// Hardware information for the current system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// Chip model (e.g., "Apple M4 Max").
    pub chip: String,
    /// CPU performance + efficiency core count.
    pub cores_cpu: u32,
    /// GPU core count.
    pub cores_gpu: u32,
    /// Neural Engine core count.
    pub ne_cores: u32,
    /// Total RAM in GB.
    pub ram_gb: u64,
    /// macOS version string.
    pub macos: String,
}

impl HardwareInfo {
    /// Detect hardware information for the current system.
    pub fn detect() -> Self {
        let chip = detect_chip_model();
        let ne_cores = ne_cores_from_chip(&chip);
        Self {
            chip,
            cores_cpu: detect_cpu_cores(),
            cores_gpu: detect_gpu_cores(),
            ne_cores,
            ram_gb: detect_ram_gb(),
            macos: detect_macos_version(),
        }
    }
}

/// Read chip model from system_profiler or sysctl.
fn detect_chip_model() -> String {
    // Try sysctl machdep.cpu.brand_string first
    if let Some(brand) = sysctl_string("machdep.cpu.brand_string") {
        return brand;
    }

    // Fallback: system_profiler
    if let Ok(output) = Command::new("system_profiler")
        .args(["SPHardwareDataType", "-detailLevel", "mini"])
        .output()
    {
        let text = String::from_utf8_lossy(&output.stdout);
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("Chip:") || trimmed.starts_with("Processor Name:") {
                if let Some(val) = trimmed.split(':').nth(1) {
                    return val.trim().to_string();
                }
            }
        }
    }

    // Fallback: hw.model
    sysctl_string("hw.model").unwrap_or_else(|| "Unknown".to_string())
}

/// Detect total CPU core count.
fn detect_cpu_cores() -> u32 {
    sysctl_u64("hw.ncpu").unwrap_or(0) as u32
}

/// Detect GPU core count (Apple Silicon specific).
fn detect_gpu_cores() -> u32 {
    // Try system_profiler for GPU core count
    if let Ok(output) = Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-detailLevel", "mini"])
        .output()
    {
        let text = String::from_utf8_lossy(&output.stdout);
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.contains("Total Number of Cores:") || trimmed.contains("Cores:") {
                if let Some(num) = extract_number(trimmed) {
                    return num as u32;
                }
            }
        }
    }
    0
}

/// Estimate Neural Engine core count from chip model string.
fn ne_cores_from_chip(chip: &str) -> u32 {
    // ANE core count isn't directly exposed via sysctl.
    // Use chip model to estimate: M1/M2/M3/M4 = 16 cores, A-series varies.
    let lower = chip.to_lowercase();
    if lower.contains("m1")
        || lower.contains("m2")
        || lower.contains("m3")
        || lower.contains("m4")
        || lower.contains("m5")
    {
        16
    } else {
        0
    }
}

/// Detect total RAM in GB.
fn detect_ram_gb() -> u64 {
    sysctl_u64("hw.memsize")
        .map(|bytes| bytes / (1024 * 1024 * 1024))
        .unwrap_or(0)
}

/// Detect macOS version.
fn detect_macos_version() -> String {
    if let Ok(output) = Command::new("sw_vers").arg("-productVersion").output() {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !version.is_empty() {
            return version;
        }
    }
    "Unknown".to_string()
}

/// Read a string value from sysctl.
fn sysctl_string(name: &str) -> Option<String> {
    let output = Command::new("sysctl").arg("-n").arg(name).output().ok()?;
    if output.status.success() {
        let val = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if val.is_empty() { None } else { Some(val) }
    } else {
        None
    }
}

/// Read a numeric value from sysctl.
fn sysctl_u64(name: &str) -> Option<u64> {
    sysctl_string(name)?.parse().ok()
}

/// Extract the first number from a string.
fn extract_number(s: &str) -> Option<u64> {
    let num_str: String = s
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num_str.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_hardware() {
        let info = HardwareInfo::detect();
        // On macOS, we should get some values
        assert!(!info.chip.is_empty());
        assert!(info.cores_cpu > 0);
        assert!(info.ram_gb > 0);
        assert!(!info.macos.is_empty());
    }

    #[test]
    fn test_hardware_serialization() {
        let info = HardwareInfo {
            chip: "Apple M4 Max".to_string(),
            cores_cpu: 16,
            cores_gpu: 40,
            ne_cores: 16,
            ram_gb: 64,
            macos: "15.4".to_string(),
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: HardwareInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.chip, "Apple M4 Max");
        assert_eq!(parsed.cores_gpu, 40);
        assert_eq!(parsed.ram_gb, 64);
    }

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("Total Cores: 40"), Some(40));
        assert_eq!(extract_number("16 cores"), Some(16));
        assert_eq!(extract_number("no numbers here"), None);
        assert_eq!(extract_number(""), None);
    }
}
