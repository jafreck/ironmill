//! Compile-time dispatch graph for the Metal decode pipeline.
//!
//! The decode pipeline structure is completely static for a given model
//! architecture — every decode step executes the same sequence of operations
//! with the same buffer dependencies. This module pre-computes that sequence
//! at model load time, enabling:
//!
//! - **Barrier optimization**: compute the minimum set of memory barriers
//!   from actual read/write dependencies, eliminating conservative barriers.
//! - **Branch elimination**: resolve all architecture-specific branches
//!   (GDN vs Standard, output gates, V-norm, etc.) once at compile time.
//! - **Lifetime analysis**: identify buffer aliasing opportunities from
//!   non-overlapping lifetimes.

use mil_rs::weights::ModelConfig;

use super::plan::{AttentionKind, LayerPlan};

// ── Buffer slots ───────────────────────────────────────────────

/// Identifies an intermediate buffer in the decode pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Buf {
    HiddenState,
    NormOut,
    Residual,
    QProj,
    KProj,
    VProj,
    QGate,
    AttnOut,
    FfnGate,
    FfnUp,
    FfnDown,
}

impl Buf {
    pub(crate) const COUNT: usize = 11;

    fn index(self) -> usize {
        self as usize
    }
}

// ── Graph nodes ────────────────────────────────────────────────

/// A node in the decode graph: one logical operation with its buffer deps.
#[derive(Clone, Debug)]
struct GraphNode {
    #[allow(dead_code)]
    label: &'static str,
    reads: Vec<Buf>,
    writes: Vec<Buf>,
}

// ── Compiled barrier point ─────────────────────────────────────

/// A memory barrier to insert between graph nodes.
#[derive(Clone, Debug)]
pub(crate) struct BarrierPoint {
    /// Insert this barrier before executing the node at `before_node`.
    #[allow(dead_code)]
    pub(crate) before_node: usize,
    /// Buffer slots that need a barrier.
    pub(crate) slots: Vec<Buf>,
}

// ── Decode graph ───────────────────────────────────────────────

/// Pre-compiled decode graph for a specific model architecture.
///
/// Built once at model load time. Used to generate optimized barrier
/// schedules and analyze buffer lifetimes.
pub(crate) struct DecodeGraph {
    nodes: Vec<GraphNode>,
    /// Minimum barrier schedule computed from read/write dependencies.
    pub(crate) barriers: Vec<BarrierPoint>,
    /// Number of barriers in the current (conservative) pipeline.
    pub(crate) conservative_barrier_count: usize,
}

impl DecodeGraph {
    /// Build the decode graph from layer plans and model config.
    ///
    /// This captures the exact operation sequence for token_count=1 decode,
    /// including all architecture-specific decisions pre-resolved from the
    /// layer plans.
    pub(crate) fn build(layer_plans: &[LayerPlan], mc: &ModelConfig) -> Self {
        let mut nodes = Vec::new();
        let _nh = mc.num_attention_heads;

        // Embedding + norm
        nodes.push(GraphNode {
            label: "embed_norm",
            reads: vec![],
            writes: vec![Buf::HiddenState, Buf::NormOut],
        });

        for plan in layer_plans.iter() {
            let _layer_hd = plan.head_dim as usize;
            let _layer_nkv = plan.num_kv_heads as usize;

            match &plan.attention {
                AttentionKind::Gdn => {
                    // GDN projections read norm_out, write to GDN-internal buffers.
                    // From the pipeline's perspective, norm_out is consumed.
                    nodes.push(GraphNode {
                        label: "gdn_proj",
                        reads: vec![Buf::NormOut],
                        writes: vec![],
                    });
                    // GDN core: recurrent + O-proj + residual + norm.
                    // Reads hidden_state (for residual add).
                    // Writes norm_out, residual, hidden_state.
                    nodes.push(GraphNode {
                        label: "gdn_core",
                        reads: vec![Buf::HiddenState],
                        writes: vec![Buf::NormOut, Buf::Residual, Buf::HiddenState],
                    });
                }
                AttentionKind::Standard {
                    has_output_gate,
                    has_v_norm,
                } => {
                    // Q/K/V projections (and optional gate).
                    let mut proj_writes = vec![Buf::QProj, Buf::KProj, Buf::VProj];
                    if *has_output_gate {
                        proj_writes.push(Buf::QGate);
                    }
                    nodes.push(GraphNode {
                        label: "std_qkv_proj",
                        reads: vec![Buf::NormOut],
                        writes: proj_writes,
                    });

                    // QK norm + RoPE: in-place on q_proj, k_proj.
                    let mut rope_reads = vec![Buf::QProj, Buf::KProj];
                    let mut rope_writes = vec![Buf::QProj, Buf::KProj];
                    // V-norm writes v_proj in-place (Gemma 4).
                    if *has_v_norm {
                        rope_reads.push(Buf::VProj);
                        rope_writes.push(Buf::VProj);
                    }
                    nodes.push(GraphNode {
                        label: "qk_norm_rope",
                        reads: rope_reads,
                        writes: rope_writes,
                    });

                    // KV cache + attention.
                    nodes.push(GraphNode {
                        label: "kv_attn",
                        reads: vec![Buf::QProj, Buf::KProj, Buf::VProj],
                        writes: vec![Buf::AttnOut],
                    });

                    // Sigmoid gate (optional).
                    if *has_output_gate {
                        nodes.push(GraphNode {
                            label: "sigmoid_gate",
                            reads: vec![Buf::AttnOut, Buf::QGate],
                            writes: vec![Buf::AttnOut],
                        });
                    }

                    // O projection.
                    nodes.push(GraphNode {
                        label: "o_proj",
                        reads: vec![Buf::AttnOut],
                        writes: vec![Buf::FfnDown],
                    });

                    // Residual + post-attention norm.
                    nodes.push(GraphNode {
                        label: "post_attn_res_norm",
                        reads: vec![Buf::HiddenState, Buf::FfnDown],
                        writes: vec![Buf::NormOut, Buf::Residual],
                    });
                }
            }

            // FFN block (shared by both layer types).
            // Fused gate+up+act path: reads norm_out, writes ffn_gate.
            nodes.push(GraphNode {
                label: "ffn_gate_up_act",
                reads: vec![Buf::NormOut],
                writes: vec![Buf::FfnGate],
            });
            // Down projection.
            nodes.push(GraphNode {
                label: "ffn_down_proj",
                reads: vec![Buf::FfnGate],
                writes: vec![Buf::FfnDown],
            });

            // End-of-layer residual + norm (prepares for next layer).
            nodes.push(GraphNode {
                label: "eol_res_norm",
                reads: vec![Buf::Residual, Buf::FfnDown],
                writes: vec![Buf::HiddenState, Buf::NormOut],
            });
        }

        // Final norm.
        nodes.push(GraphNode {
            label: "final_norm",
            reads: vec![Buf::HiddenState],
            writes: vec![Buf::NormOut],
        });

        // LM head projection.
        nodes.push(GraphNode {
            label: "lm_head",
            reads: vec![Buf::NormOut],
            writes: vec![], // logits buffer not tracked
        });

        // Compute barrier counts.
        let conservative = count_conservative_barriers(layer_plans);
        let barriers = compute_minimum_barriers(&nodes);

        DecodeGraph {
            nodes,
            barriers,
            conservative_barrier_count: conservative,
        }
    }

    /// Number of barriers in the optimized schedule.
    pub(crate) fn optimized_barrier_count(&self) -> usize {
        self.barriers.len()
    }

    /// Print analysis comparing conservative vs optimized barriers.
    pub(crate) fn print_analysis(&self) {
        let opt = self.optimized_barrier_count();
        let cons = self.conservative_barrier_count;
        let saved = cons.saturating_sub(opt);
        let total_nodes = self.nodes.len();
        let total_barrier_slots: usize = self.barriers.iter().map(|b| b.slots.len()).sum();

        eprintln!("[decode-graph] Compiled decode graph:");
        eprintln!("  Operations:           {total_nodes}");
        eprintln!("  Conservative barriers: {cons}");
        eprintln!(
            "  Optimized barriers:    {opt} ({saved} eliminated, {:.1}% reduction)",
            if cons > 0 {
                saved as f64 / cons as f64 * 100.0
            } else {
                0.0
            }
        );
        eprintln!("  Total barrier slots:   {total_barrier_slots}");

        // Lifetime analysis.
        let lifetimes = compute_lifetimes(&self.nodes);
        let aliases = find_aliases(&lifetimes);
        if !aliases.is_empty() {
            eprintln!("  Buffer aliases:");
            for (a, b) in &aliases {
                eprintln!("    {:?} ↔ {:?} (non-overlapping lifetimes)", a, b);
            }
        }
    }
}

// ── Barrier computation ────────────────────────────────────────

/// Compute the minimum barrier set from read/write dependencies.
///
/// For each node that reads a buffer, check if the most recent write to
/// that buffer has been covered by a barrier. If not, include it in a
/// barrier before this node.
fn compute_minimum_barriers(nodes: &[GraphNode]) -> Vec<BarrierPoint> {
    // Track the last write step for each buffer slot.
    let mut last_write: [Option<usize>; Buf::COUNT] = [None; Buf::COUNT];
    // Track the last barrier step for each buffer slot.
    let mut last_barrier: [Option<usize>; Buf::COUNT] = [None; Buf::COUNT];
    let mut barriers = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        // Check which reads need a barrier.
        let mut needed: Vec<Buf> = Vec::new();
        for &buf in &node.reads {
            if let Some(write_step) = last_write[buf.index()] {
                let is_covered = last_barrier[buf.index()].is_some_and(|b| b > write_step);
                if !is_covered {
                    needed.push(buf);
                }
            }
        }

        if !needed.is_empty() {
            // Deduplicate.
            needed.sort_by_key(|b| b.index());
            needed.dedup();
            for &buf in &needed {
                last_barrier[buf.index()] = Some(i);
            }
            barriers.push(BarrierPoint {
                before_node: i,
                slots: needed,
            });
        }

        // Record writes.
        for &buf in &node.writes {
            last_write[buf.index()] = Some(i);
        }
    }

    barriers
}

/// Count barriers in the current conservative pipeline.
///
/// This counts the explicit `memory_barrier_with_resources` calls in
/// pipeline.rs + internal barriers in ffn.rs/gdn.rs/attention.rs.
fn count_conservative_barriers(layer_plans: &[LayerPlan]) -> usize {
    let mut count = 0;

    // Embedding path: 1 barrier (hidden_state + norm_out).
    count += 1;

    for plan in layer_plans {
        match &plan.attention {
            AttentionKind::Gdn => {
                // GDN internal barriers (projections: ~3, core: ~2) + pipeline barrier.
                count += 6;
            }
            AttentionKind::Standard {
                has_output_gate, ..
            } => {
                // QKV proj barrier, QK norm+rope barrier, attn barrier,
                // O-proj barrier, residual+norm barrier.
                count += 5;
                if *has_output_gate {
                    count += 1; // sigmoid gate barrier
                }
            }
        }

        // FFN: internal barrier (gate+up), activation barrier, + pipeline ffn_down barrier.
        // With fused gate+up+act: 1 internal barrier + 1 pipeline barrier.
        count += 2;

        // End-of-layer: 1 barrier (hidden_state + norm_out).
        count += 1;
    }

    // Final norm barrier + LM head (no barrier after).
    count += 1;

    count
}

// ── Lifetime analysis ──────────────────────────────────────────

/// Buffer lifetime: [first_write, last_read] node indices.
#[derive(Clone, Debug)]
struct Lifetime {
    buf: Buf,
    first_write: usize,
    last_read: usize,
}

/// Compute per-buffer lifetimes from the graph.
fn compute_lifetimes(nodes: &[GraphNode]) -> Vec<Lifetime> {
    let mut first_write: [Option<usize>; Buf::COUNT] = [None; Buf::COUNT];
    let mut last_read: [Option<usize>; Buf::COUNT] = [None; Buf::COUNT];

    for (i, node) in nodes.iter().enumerate() {
        for &buf in &node.writes {
            if first_write[buf.index()].is_none() {
                first_write[buf.index()] = Some(i);
            }
        }
        for &buf in &node.reads {
            last_read[buf.index()] = Some(i);
        }
    }

    let mut lifetimes = Vec::new();
    for idx in 0..Buf::COUNT {
        if let (Some(fw), Some(lr)) = (first_write[idx], last_read[idx]) {
            let buf = match idx {
                0 => Buf::HiddenState,
                1 => Buf::NormOut,
                2 => Buf::Residual,
                3 => Buf::QProj,
                4 => Buf::KProj,
                5 => Buf::VProj,
                6 => Buf::QGate,
                7 => Buf::AttnOut,
                8 => Buf::FfnGate,
                9 => Buf::FfnUp,
                10 => Buf::FfnDown,
                _ => unreachable!(),
            };
            lifetimes.push(Lifetime {
                buf,
                first_write: fw,
                last_read: lr,
            });
        }
    }

    lifetimes
}

/// Find buffer pairs with non-overlapping lifetimes that could alias.
fn find_aliases(lifetimes: &[Lifetime]) -> Vec<(Buf, Buf)> {
    let mut aliases = Vec::new();
    for i in 0..lifetimes.len() {
        for j in (i + 1)..lifetimes.len() {
            let a = &lifetimes[i];
            let b = &lifetimes[j];
            // Non-overlapping if one's last_read < other's first_write.
            if a.last_read < b.first_write || b.last_read < a.first_write {
                aliases.push((a.buf, b.buf));
            }
        }
    }
    aliases
}

// ── Barrier tracker (runtime) ──────────────────────────────────

/// Runtime barrier tracker that computes minimum barriers on-the-fly.
///
/// Used during pipeline encoding instead of manual barrier placement.
/// Tracks which buffers have been written and only inserts barriers when
/// a subsequent read requires visibility of those writes.
pub(crate) struct BarrierTracker {
    /// For each buffer slot, whether it has been written since the last
    /// barrier for that slot.
    dirty: [bool; Buf::COUNT],
}

impl BarrierTracker {
    pub(crate) fn new() -> Self {
        Self {
            dirty: [false; Buf::COUNT],
        }
    }

    /// Mark that a buffer slot was written by the current dispatch.
    #[allow(dead_code)]
    pub(crate) fn mark_write(&mut self, slot: Buf) {
        self.dirty[slot.index()] = true;
    }

    /// Mark multiple buffer slots as written.
    pub(crate) fn mark_writes(&mut self, slots: &[Buf]) {
        for &s in slots {
            self.dirty[s.index()] = true;
        }
    }

    /// Insert a memory barrier for any dirty buffers in `reads`, then
    /// clear their dirty flags. Returns the number of slots barriered.
    ///
    /// This is the core optimization: only barrier buffers that were
    /// actually written since their last barrier, eliminating redundant
    /// barriers for buffers that haven't changed.
    pub(crate) fn barrier_for_reads(
        &mut self,
        enc: &ironmill_metal_sys::ComputeEncoder,
        reads: &[Buf],
        bufs: &super::buffers::IntermediateBuffers,
    ) -> usize {
        let mut metal_bufs: Vec<&ironmill_metal_sys::MetalBuffer> = Vec::new();
        for &slot in reads {
            if self.dirty[slot.index()] {
                if let Some(buf) = resolve_buf(slot, bufs) {
                    metal_bufs.push(buf);
                }
                self.dirty[slot.index()] = false;
            }
        }
        let count = metal_bufs.len();
        if !metal_bufs.is_empty() {
            enc.memory_barrier_with_resources(&metal_bufs);
        }
        count
    }

    /// Total number of dirty slots (for diagnostics).
    #[allow(dead_code)]
    pub(crate) fn dirty_count(&self) -> usize {
        self.dirty.iter().filter(|&&d| d).count()
    }
}

/// Map a buffer slot to the actual MetalBuffer reference.
pub(crate) fn resolve_buf(
    slot: Buf,
    bufs: &super::buffers::IntermediateBuffers,
) -> Option<&ironmill_metal_sys::MetalBuffer> {
    match slot {
        Buf::HiddenState => Some(&bufs.hidden_state),
        Buf::NormOut => Some(&bufs.norm_out),
        Buf::Residual => Some(&bufs.residual),
        Buf::QProj => Some(&bufs.q_proj),
        Buf::KProj => Some(&bufs.k_proj),
        Buf::VProj => Some(&bufs.v_proj),
        Buf::QGate => bufs.q_gate.as_ref(),
        Buf::AttnOut => Some(&bufs.attn_out),
        Buf::FfnGate => Some(&bufs.ffn_gate),
        Buf::FfnUp => Some(&bufs.ffn_up),
        Buf::FfnDown => Some(&bufs.ffn_down),
    }
}
