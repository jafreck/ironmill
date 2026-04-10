//! Feature-gated Metal profiling infrastructure.
//!
//! Provides [`ProfiledEncoder`] which wraps a command queue and tracks
//! per-category and optionally per-dispatch GPU timing. Compiled only
//! when the `profile-metal` feature is enabled — zero cost in production.
//!
//! # Usage
//!
//! ```ignore
//! let mut profiler = ProfiledEncoder::new(&queue, "embed")?;
//!
//! // Encode operations using profiler.encoder()
//! encode_embedding(profiler.encoder(), ...);
//!
//! // Switch category (commits current cmd_buf, measures GPU time)
//! profiler.switch_category("proj")?;
//! encode_projection(profiler.encoder(), ...);
//!
//! // Finish and collect timings
//! let report = profiler.finish()?;
//! report.print_table();
//! ```

use ironmill_metal_sys::{CommandBuffer, CommandQueue, ComputeEncoder};

use crate::engine::InferenceError;

/// Profiling granularity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilingGranularity {
    /// Measure GPU time per operation category (proj, attn, ffn, norm, embed, lm_head).
    /// Low overhead — one command buffer switch per category boundary.
    Category,
    /// Measure GPU time per individual dispatch call.
    /// High overhead (~50ms total) but gives per-kernel timing.
    Dispatch,
}

/// A single timing record from the profiler.
#[derive(Debug, Clone)]
pub struct TimingRecord {
    /// Category name (e.g., "proj", "attn", "ffn").
    pub category: &'static str,
    /// Optional dispatch label (e.g., "q_proj", "gate+up_fused") for dispatch-level granularity.
    pub label: Option<String>,
    /// GPU time in milliseconds.
    pub gpu_ms: f64,
    /// Optional byte count for bandwidth calculation.
    pub bytes: Option<usize>,
}

/// Profiling report containing all timing records and metadata.
#[derive(Debug)]
pub struct ProfilingReport {
    /// Individual timing records.
    pub records: Vec<TimingRecord>,
    /// Wall clock time in milliseconds.
    pub wall_ms: f64,
}

impl ProfilingReport {
    /// Aggregate timings by category.
    pub fn by_category(&self) -> Vec<(&'static str, f64)> {
        let mut cats: Vec<(&'static str, f64)> = Vec::new();
        for rec in &self.records {
            if let Some(entry) = cats.iter_mut().find(|(c, _)| *c == rec.category) {
                entry.1 += rec.gpu_ms;
            } else {
                cats.push((rec.category, rec.gpu_ms));
            }
        }
        cats
    }

    /// Total GPU time in milliseconds.
    pub fn total_gpu_ms(&self) -> f64 {
        self.records.iter().map(|r| r.gpu_ms).sum()
    }

    /// Print human-readable profiling table to stderr.
    pub fn print_table(&self) {
        let cats = self.by_category();
        let total_gpu = self.total_gpu_ms();
        let overhead = self.wall_ms - total_gpu;

        eprintln!("[profile-metal] decode breakdown:");
        // Compute per-layer-type totals for summary.
        let mut proj_gdn_ms = 0.0_f64;
        let mut attn_gdn_ms = 0.0_f64;
        let mut proj_std_ms = 0.0_f64;
        let mut attn_std_ms = 0.0_f64;
        let mut gdn_count = 0_usize;
        let mut std_count = 0_usize;
        for (cat, ms) in &cats {
            match *cat {
                "proj_gdn" => proj_gdn_ms += ms,
                "attn_gdn" => {
                    attn_gdn_ms += ms;
                    gdn_count = self
                        .records
                        .iter()
                        .filter(|r| r.category == "attn_gdn")
                        .count();
                }
                "proj_std" => proj_std_ms += ms,
                "attn_std" => {
                    attn_std_ms += ms;
                    std_count = self
                        .records
                        .iter()
                        .filter(|r| r.category == "attn_std")
                        .count();
                }
                _ => {}
            }
        }
        // Display categories — merge GDN/STD into combined totals and show per-layer-type detail.
        for (cat, ms) in &cats {
            let label = match *cat {
                "proj_gdn" | "proj_std" => continue, // shown in summary below
                "attn_gdn" | "attn_std" => continue, // shown in summary below
                "proj" => "Projections (Q/K/V/O):",
                "ffn" => "FFN (gate+up+act+down):",
                "attn" => "Attention:",
                "norm" => "Norms + residual:",
                "embed" => "Embedding:",
                "lm_head" => "LM head:",
                _ => cat,
            };
            eprintln!(
                "  {:<32} {:>6.2}ms  ({:>4.1}%)",
                label,
                ms,
                ms / self.wall_ms * 100.0
            );
        }
        // Print GDN/Standard split if present.
        let has_split = proj_gdn_ms > 0.0 || proj_std_ms > 0.0;
        if has_split {
            let total_proj = proj_gdn_ms + proj_std_ms;
            let total_attn = attn_gdn_ms + attn_std_ms;
            eprintln!(
                "  {:<32} {:>6.2}ms  ({:>4.1}%)",
                "Projections (total):",
                total_proj,
                total_proj / self.wall_ms * 100.0
            );
            if gdn_count > 0 {
                eprintln!(
                    "    {:<30} {:>6.2}ms  ({} GDN layers, {:.2}ms/layer)",
                    "GDN projections:",
                    proj_gdn_ms,
                    gdn_count,
                    proj_gdn_ms / gdn_count as f64
                );
            }
            if std_count > 0 {
                let std_proj_count = self
                    .records
                    .iter()
                    .filter(|r| r.category == "proj_std")
                    .count();
                eprintln!(
                    "    {:<30} {:>6.2}ms  ({} Std layers, {:.2}ms/layer)",
                    "Std projections (QKV+O):",
                    proj_std_ms,
                    std_count,
                    proj_std_ms / (std_count as f64)
                );
                // Standard layers have 2 proj records each (QKV + O-proj).
                // Show per-phase split if we have enough records.
                if std_proj_count == std_count * 2 {
                    let std_proj_recs: Vec<f64> = self
                        .records
                        .iter()
                        .filter(|r| r.category == "proj_std")
                        .map(|r| r.gpu_ms)
                        .collect();
                    let qkv_total: f64 = std_proj_recs.iter().step_by(2).sum();
                    let oproj_total: f64 = std_proj_recs.iter().skip(1).step_by(2).sum();
                    eprintln!(
                        "      QKV: {:.2}ms ({:.2}ms/layer)  O-proj: {:.2}ms ({:.2}ms/layer)",
                        qkv_total,
                        qkv_total / std_count as f64,
                        oproj_total,
                        oproj_total / std_count as f64,
                    );
                }
            }
            eprintln!(
                "  {:<32} {:>6.2}ms  ({:>4.1}%)",
                "Attention (total):",
                total_attn,
                total_attn / self.wall_ms * 100.0
            );
            if gdn_count > 0 {
                eprintln!(
                    "    {:<30} {:>6.2}ms  ({} GDN layers, {:.2}ms/layer)",
                    "GDN recurrent+O+res+norm:",
                    attn_gdn_ms,
                    gdn_count,
                    attn_gdn_ms / gdn_count as f64
                );
            }
            if std_count > 0 {
                eprintln!(
                    "    {:<30} {:>6.2}ms  ({} Std layers, {:.2}ms/layer)",
                    "Std attention:",
                    attn_std_ms,
                    std_count,
                    attn_std_ms / std_count as f64
                );
            }
        }
        eprintln!("  ─────────────────────────────────");
        eprintln!("  Total GPU:              {:>6.2}ms", total_gpu);
        eprintln!("  Wall clock:             {:>6.2}ms", self.wall_ms);
        eprintln!(
            "  Overhead (idle/CPU):    {:>6.2}ms  ({:>4.1}%)",
            overhead,
            overhead / self.wall_ms * 100.0
        );

        // Per-category bandwidth if any records have byte counts
        let has_bw = self.records.iter().any(|r| r.bytes.is_some());
        if has_bw {
            eprintln!();
            eprintln!("  Per-category bandwidth:");
            for (cat, ms) in &cats {
                let cat_bytes: usize = self
                    .records
                    .iter()
                    .filter(|r| r.category == *cat && r.bytes.is_some())
                    .map(|r| r.bytes.unwrap())
                    .sum();
                if cat_bytes > 0 {
                    let bw_gbs = cat_bytes as f64 / (*ms / 1000.0) / 1e9;
                    eprintln!(
                        "    {cat:<8} {:.1} MB → {:>6.1} GB/s",
                        cat_bytes as f64 / 1e6,
                        bw_gbs
                    );
                }
            }
        }
    }

    /// Print per-dispatch detail table to stderr.
    pub fn print_dispatch_detail(&self) {
        if self.records.iter().all(|r| r.label.is_none()) {
            return;
        }
        eprintln!();
        eprintln!("[profile-metal] per-dispatch detail:");
        eprintln!(
            "  {:>4} {:>8} {:>28} {:>10} {:>10}",
            "#", "cat", "dispatch", "GPU (ms)", "BW (GB/s)"
        );
        for (i, rec) in self.records.iter().enumerate() {
            let label = rec.label.as_deref().unwrap_or("-");
            let bw = if let Some(bytes) = rec.bytes {
                let gbs = bytes as f64 / (rec.gpu_ms / 1000.0) / 1e9;
                format!("{gbs:.1}")
            } else {
                "-".to_string()
            };
            eprintln!(
                "  {:>4} {:>8} {:>28} {:>10.3} {:>10}",
                i, rec.category, label, rec.gpu_ms, bw
            );
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self, model: &str, hardware: &str) -> serde_json::Value {
        let cats = self.by_category();
        let mut categories = serde_json::Map::new();
        for (cat, ms) in &cats {
            let cat_bytes: usize = self
                .records
                .iter()
                .filter(|r| r.category == *cat && r.bytes.is_some())
                .map(|r| r.bytes.unwrap())
                .sum();
            let mut entry = serde_json::Map::new();
            entry.insert(
                "gpu_ms".into(),
                serde_json::json!((*ms * 100.0).round() / 100.0),
            );
            if cat_bytes > 0 {
                entry.insert(
                    "weight_mb".into(),
                    serde_json::json!((cat_bytes as f64 / 1e6 * 10.0).round() / 10.0),
                );
                let bw_gbs = cat_bytes as f64 / (*ms / 1000.0) / 1e9;
                entry.insert(
                    "bw_gbs".into(),
                    serde_json::json!((bw_gbs * 10.0).round() / 10.0),
                );
            }
            categories.insert(cat.to_string(), serde_json::Value::Object(entry));
        }

        let dispatches: Vec<serde_json::Value> = self
            .records
            .iter()
            .map(|r| {
                let mut d = serde_json::Map::new();
                d.insert("category".into(), serde_json::json!(r.category));
                if let Some(ref label) = r.label {
                    d.insert("label".into(), serde_json::json!(label));
                }
                d.insert(
                    "gpu_ms".into(),
                    serde_json::json!((r.gpu_ms * 1000.0).round() / 1000.0),
                );
                if let Some(bytes) = r.bytes {
                    d.insert("bytes".into(), serde_json::json!(bytes));
                }
                serde_json::Value::Object(d)
            })
            .collect();

        serde_json::json!({
            "model": model,
            "hardware": hardware,
            "total_gpu_ms": (self.total_gpu_ms() * 100.0).round() / 100.0,
            "wall_ms": (self.wall_ms * 100.0).round() / 100.0,
            "categories": serde_json::Value::Object(categories),
            "dispatches": dispatches,
        })
    }
}

/// Wraps a Metal command queue to collect per-category GPU timing.
///
/// At each category boundary, the current command buffer is committed and
/// waited on to measure its GPU execution time. A new command buffer and
/// encoder are created for the next category.
///
/// When used with [`ProfilingGranularity::Dispatch`], timing is collected
/// per individual dispatch call rather than per category.
pub struct ProfiledEncoder<'a> {
    queue: &'a CommandQueue,
    cmd_buf: CommandBuffer,
    enc: ComputeEncoder,
    current_category: &'static str,
    current_label: Option<String>,
    current_bytes: Option<usize>,
    records: Vec<TimingRecord>,
    wall_start: std::time::Instant,
    granularity: ProfilingGranularity,
    /// When true, P1 fusion is disabled for accurate per-category timing.
    pub disable_p1_fusion: bool,
}

impl<'a> ProfiledEncoder<'a> {
    /// Create a new profiled encoder starting in the given category.
    pub fn new(
        queue: &'a CommandQueue,
        initial_category: &'static str,
    ) -> Result<Self, InferenceError> {
        let cmd_buf = queue
            .command_buffer()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        let enc = cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        Ok(Self {
            queue,
            cmd_buf,
            enc,
            current_category: initial_category,
            current_label: None,
            current_bytes: None,
            records: Vec::new(),
            wall_start: std::time::Instant::now(),
            granularity: ProfilingGranularity::Category,
            disable_p1_fusion: true,
        })
    }

    /// Set the profiling granularity.
    pub fn set_granularity(&mut self, granularity: ProfilingGranularity) {
        self.granularity = granularity;
    }

    /// Get the current compute encoder for encoding dispatches.
    pub fn encoder(&self) -> &ComputeEncoder {
        &self.enc
    }

    /// Switch to a new operation category.
    ///
    /// Commits the current command buffer, records its GPU time, and
    /// creates a fresh command buffer + encoder for the new category.
    pub fn switch_category(&mut self, new_category: &'static str) -> Result<(), InferenceError> {
        self.commit_current()?;
        self.current_category = new_category;
        self.current_label = None;
        self.current_bytes = None;
        self.new_cmd_buf()?;
        Ok(())
    }

    /// Record a dispatch label and byte count for per-dispatch tracking.
    ///
    /// In [`ProfilingGranularity::Dispatch`] mode, this commits the current
    /// command buffer after every labeled dispatch to get per-kernel timing.
    ///
    /// In [`ProfilingGranularity::Category`] mode, the label and bytes are
    /// accumulated into the current category's timing record.
    pub fn record_dispatch(
        &mut self,
        label: &str,
        bytes: Option<usize>,
    ) -> Result<(), InferenceError> {
        if self.granularity == ProfilingGranularity::Dispatch {
            // Per-dispatch: commit after every dispatch
            self.current_label = Some(label.to_string());
            self.current_bytes = bytes;
            self.commit_current()?;
            self.current_label = None;
            self.current_bytes = None;
            self.new_cmd_buf()?;
        } else {
            // Category mode: just accumulate byte count
            if let Some(b) = bytes {
                self.current_bytes = Some(self.current_bytes.unwrap_or(0) + b);
            }
        }
        Ok(())
    }

    /// Finish profiling: commit the last command buffer and return the report.
    pub fn finish(mut self) -> Result<ProfilingReport, InferenceError> {
        self.commit_current()?;
        let wall_ms = self.wall_start.elapsed().as_secs_f64() * 1000.0;
        Ok(ProfilingReport {
            records: self.records,
            wall_ms,
        })
    }

    // Internal: commit current command buffer and record timing
    fn commit_current(&mut self) -> Result<(), InferenceError> {
        self.enc.end_encoding();
        self.cmd_buf.commit();
        self.cmd_buf.wait_until_completed();
        let gpu_ms = (self.cmd_buf.gpu_end_time() - self.cmd_buf.gpu_start_time()) * 1000.0;
        self.records.push(TimingRecord {
            category: self.current_category,
            label: self.current_label.take(),
            gpu_ms,
            bytes: self.current_bytes.take(),
        });
        Ok(())
    }

    // Internal: create a new command buffer + encoder
    fn new_cmd_buf(&mut self) -> Result<(), InferenceError> {
        self.cmd_buf = self
            .queue
            .command_buffer()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        self.enc = self
            .cmd_buf
            .compute_encoder()
            .map_err(|e| InferenceError::runtime(e.to_string()))?;
        Ok(())
    }
}
