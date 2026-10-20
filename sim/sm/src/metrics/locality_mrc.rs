//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

pub(super) fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct MemoryTraffic {
    pub requests: u64,
    pub sectors: u64,
    /// Distinct decoded 32-byte sectors in the request spans.  Atomic and
    /// reduction alias multiplicity is intentionally removed here; `sectors`
    /// retains the serialized transaction surface used by the issue model.
    pub unique_sectors: u64,
    /// Distinct 128-byte lines (four 32-byte sectors) touched by the decoded
    /// request spans.  This is a trace topology diagnostic, not an L2
    /// transaction counter.
    pub unique_lines: u64,
    /// Sector-span entries that repeat a decoded sector, normally same-word
    /// atomic/reduction aliases.  A missing/flat sector arena leaves this at
    /// zero rather than manufacturing an alias count.
    pub sector_aliases: u64,
    /// Shared 32-bank/4-byte-word service work, not NCU sampled stalls.
    pub bank_wavefronts: u64,
    /// Unavoidable shared service work from per-lane access width. NCU's
    /// bank-conflict counter is excess over this ideal, not excess over one
    /// request; retaining it separately avoids calling wide accesses bank
    /// conflicts.
    pub ideal_bank_wavefronts: u64,
    /// Actual wavefronts eligible for the NCU per-operation bank-conflict
    /// surface. Matrix shared operations (LDSM/STSM) remain in
    /// `bank_wavefronts` but are excluded here because NCU reports them in
    /// the matrix operation class rather than `op_ld`/`op_st` conflicts.
    pub bank_conflict_wavefronts: u64,
    /// Width-only ideal wavefronts for `bank_conflict_wavefronts`.
    pub ideal_bank_conflict_wavefronts: u64,
    /// Sum of per-instruction excess wavefronts above that instruction's
    /// width-only ideal.  This must be accumulated before aggregation:
    /// subtracting the two launch-wide sums lets a low-conflict wide access
    /// cancel a genuinely conflicting narrow access.
    pub bank_conflict_excess_wavefronts: u64,
}

/// Trace-derived reuse-interval observations for one launch.
///
/// This is deliberately a passive locality diagnostic.  It does not claim a
/// hardware cache hit, does not change the live L1/L2 state, and does not
/// feed issue timing or DRAM service.  The stream scope is explicitly
/// warp-local because the trace contract does not provide a total dynamic
/// order across warps or SMs.  A reuse interval is therefore the number of
/// *memory records in the same warp stream* between two references to a line.
/// All lines in one record share one logical tick because their relative order
/// is not present in the trace.  The capacity arrays are survival proxies
/// based on that interval, not set-associative hardware MRCs; reuse distance
/// is not reconstructed from reuse time alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReuseClassObservation {
    pub line_references: u64,
    pub sector_references: u64,
    pub cold_line_references: u64,
    pub cold_sector_references: u64,
    pub reused_line_references: u64,
    pub reused_sector_references: u64,
}

impl Default for ReuseClassObservation {
    fn default() -> Self {
        Self {
            line_references: 0,
            sector_references: 0,
            cold_line_references: 0,
            cold_sector_references: 0,
            reused_line_references: 0,
            reused_sector_references: 0,
        }
    }
}

pub const REUSE_PROXY_CAPACITY_RECORDS: [u64; 8] = [0, 1, 2, 4, 8, 16, 32, 64];

pub(super) const REUSE_INTERVAL_HISTOGRAM_BINS: usize = 32;

pub(super) const HOTL_INTERVAL_HISTOGRAM_BINS: usize = 80;

/// Report-only reuse interval and MRC-proxy counters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReuseIntervalObservation {
    /// Stable schema label so downstream consumers cannot mistake this for a
    /// physical cache counter.
    pub schema: &'static str,
    /// The only ordering used by this observer.  Cross-warp and cross-SM
    /// arrival order is unavailable in the native trace contract.
    pub ordering_scope: &'static str,
    /// Lines within one memory record share one tick; no order is fabricated.
    pub within_record_order: &'static str,
    /// Number of nonempty warp streams observed in this launch.
    pub streams: u64,
    pub line_references: u64,
    pub sector_references: u64,
    /// Sum of unique lines within each independent warp stream. The same
    /// physical line in two warps is intentionally counted twice.
    pub stream_unique_lines: u64,
    pub maximum_stream_working_set_lines: u64,
    pub cold_line_references: u64,
    pub reused_line_references: u64,
    pub cold_sector_references: u64,
    pub reused_sector_references: u64,
    pub line_reuse_interval_sum: u64,
    pub sector_reuse_interval_sum: u64,
    pub line_reuse_interval_histogram: [u64; REUSE_INTERVAL_HISTOGRAM_BINS],
    pub sector_reuse_interval_histogram: [u64; REUSE_INTERVAL_HISTOGRAM_BINS],
    pub line_reuse_interval_overflow: u64,
    pub sector_reuse_interval_overflow: u64,
    pub records_without_decoded_span: u64,
    pub records_with_empty_span: u64,
    /// Fixed, architecture-neutral capacities used only for a monotone
    /// reuse-interval survival proxy.  They are not SM86 or SM120 geometry.
    /// These points are in logical record-gap units, not physical cache-line
    /// capacities. They are retained as a bounded RI sensitivity curve.
    pub proxy_capacity_records: [u64; 8],
    pub line_proxy_miss_references: [u64; 8],
    pub sector_proxy_miss_references: [u64; 8],
    /// Broad trace classes are kept separate so stores/atomics/async copies
    /// cannot silently contaminate a load-only interpretation. These counters
    /// are attributions over the unified warp stream, not independent
    /// class-specific MRCs.
    pub by_class: BTreeMap<String, ReuseClassObservation>,
    pub class_accounting_scope: &'static str,
    pub hotl_line_volume_clock: u64,
    pub hotl_line_interval_count: Vec<u64>,
    pub hotl_line_interval_sum: Vec<u64>,
    pub hotl_sector_interval_count: Vec<u64>,
    pub hotl_sector_interval_sum: Vec<u64>,
    pub hotl_line_censored_tail_count: Vec<u64>,
    pub hotl_line_censored_tail_sum: Vec<u64>,
    pub hotl_sector_censored_tail_count: Vec<u64>,
    pub hotl_sector_censored_tail_sum: Vec<u64>,
    pub hotl_line_miss_references: [u64; 28],
    pub hotl_sector_miss_references: [u64; 28],
}

/// Capacity points for the report-only exact locality curve.  Capacities are
/// logical 128-byte lines, not a claim that every point is a physical cache
/// configuration.  The explicit 224/256/512/768/896/960/1024/49152/786432
/// points cover the documented SM86/SM120 L1/L2 operating envelopes.
pub const LOCALITY_MRC_CAPACITY_LINES: [u64; 28] = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 224, 256, 512, 768, 896, 960, 1024, 2048, 4096, 8192, 16384,
    32768, 49152, 65536, 131072, 262144, 524288, 786432, 1048576,
];

pub(super) const LOCALITY_MRC_INF: u64 = u64::MAX;

/// Exact Class-A MRC counters for one origin-qualified stream.  These are
/// analytical zero-latency line-LRU results and never replace live L1/L2
/// counters or timing state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LocalityMrcClassObservation {
    pub line_references: u64,
    pub sector_references: u64,
    pub cold_line_references: u64,
    pub cold_sector_references: u64,
    pub line_miss_references: [u64; 28],
    /// Conservative upper miss curve induced by arbitrary intra-tick order.
    /// `line_miss_references` is the batch-first lower curve.
    pub line_miss_references_upper: [u64; 28],
    pub sector_miss_references: [u64; 28],
    pub sector_miss_references_upper: [u64; 28],
    /// Sector misses for which the line tag is also absent at the capacity.
    pub sector_line_tag_miss_references: [u64; 28],
    /// Sector misses on a resident line (first touch or post-eviction refill).
    pub sector_resident_fill_miss_references: [u64; 28],
    pub sector_line_tag_miss_references_upper: [u64; 28],
    pub sector_resident_fill_miss_references_upper: [u64; 28],
    pub same_tick_max_batch_lines: u64,
    pub maximum_line_rd_envelope_width: u64,
}

impl Default for LocalityMrcClassObservation {
    fn default() -> Self {
        Self {
            line_references: 0,
            sector_references: 0,
            cold_line_references: 0,
            cold_sector_references: 0,
            line_miss_references: [0; 28],
            line_miss_references_upper: [0; 28],
            sector_miss_references: [0; 28],
            sector_miss_references_upper: [0; 28],
            sector_line_tag_miss_references: [0; 28],
            sector_resident_fill_miss_references: [0; 28],
            sector_line_tag_miss_references_upper: [0; 28],
            sector_resident_fill_miss_references_upper: [0; 28],
            same_tick_max_batch_lines: 0,
            maximum_line_rd_envelope_width: 0,
        }
    }
}

/// Exact, shadow-only locality/MRC report.  Origin streams stay separate so
/// numerically equal global and local line IDs cannot alias in the reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LocalityMrcObservation {
    pub schema: &'static str,
    pub ordering_scope: &'static str,
    pub same_tick_semantics: &'static str,
    pub capacity_unit: &'static str,
    pub sector_unit: &'static str,
    pub capacity_lines: [u64; 28],
    pub by_origin: BTreeMap<String, LocalityMrcClassObservation>,
}

/// Conditional Class-B locality envelope.  This is intentionally a shadow
/// accounting surface: it reports a trace-ordered lower/upper scenario at a
/// declared capacity and never mutates cache state, MSHRs, L2 traffic, or
/// timing.  The two endpoint curves are kept separate from Class-A MRC.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConditionalLocalityObservation {
    pub schema: &'static str,
    pub ordering_scope: &'static str,
    pub capacity_unit: &'static str,
    pub capacity_lines: Vec<u64>,
    pub references: u64,
    pub lower_hits: Vec<u64>,
    pub upper_hits: Vec<u64>,
    pub lower_misses: Vec<u64>,
    pub upper_misses: Vec<u64>,
    pub constructive_inter_warp_sharing_gain_pct: f64,
    pub destructive_inter_warp_dilation_loss_pct: f64,
    pub in_flight_mshr_merges: u64,
    pub settled_sram_hits: u64,
    pub lease_zero_store_references: u64,
    pub ordering_unavailable_references: u64,
}

impl ConditionalLocalityObservation {
    pub(super) fn new(
        schema: &'static str,
        capacity_lines: Vec<u64>,
        ordering_scope: &'static str,
    ) -> Self {
        let n = capacity_lines.len();
        Self {
            schema,
            ordering_scope,
            capacity_unit: "distinct_128B_lines",
            capacity_lines,
            references: 0,
            lower_hits: vec![0; n],
            upper_hits: vec![0; n],
            lower_misses: vec![0; n],
            upper_misses: vec![0; n],
            constructive_inter_warp_sharing_gain_pct: 0.0,
            destructive_inter_warp_dilation_loss_pct: 0.0,
            in_flight_mshr_merges: 0,
            settled_sram_hits: 0,
            lease_zero_store_references: 0,
            ordering_unavailable_references: 0,
        }
    }
}

impl Default for ConditionalLocalityObservation {
    fn default() -> Self {
        Self::new(
            "conditional_locality_observer_unbound",
            Vec::new(),
            "not_bound_to_a_cross_warp_order",
        )
    }
}

impl Default for LocalityMrcObservation {
    fn default() -> Self {
        Self {
            schema: "sector_mrc_observer_v1",
            ordering_scope: "within_warp_trace_order_only",
            same_tick_semantics: "batch_query_before_insert_with_interval_envelope",
            capacity_unit: "distinct_128B_lines",
            sector_unit: "32B_sectors",
            capacity_lines: LOCALITY_MRC_CAPACITY_LINES,
            by_origin: BTreeMap::new(),
        }
    }
}

impl Default for ReuseIntervalObservation {
    fn default() -> Self {
        Self {
            schema: "reuse_interval_observer_v1",
            ordering_scope: "within_warp_trace_order_only",
            within_record_order: "same_logical_tick_no_order_available",
            streams: 0,
            line_references: 0,
            sector_references: 0,
            stream_unique_lines: 0,
            maximum_stream_working_set_lines: 0,
            cold_line_references: 0,
            reused_line_references: 0,
            cold_sector_references: 0,
            reused_sector_references: 0,
            line_reuse_interval_sum: 0,
            sector_reuse_interval_sum: 0,
            line_reuse_interval_histogram: [0; REUSE_INTERVAL_HISTOGRAM_BINS],
            sector_reuse_interval_histogram: [0; REUSE_INTERVAL_HISTOGRAM_BINS],
            line_reuse_interval_overflow: 0,
            sector_reuse_interval_overflow: 0,
            records_without_decoded_span: 0,
            records_with_empty_span: 0,
            proxy_capacity_records: REUSE_PROXY_CAPACITY_RECORDS,
            line_proxy_miss_references: [0; 8],
            sector_proxy_miss_references: [0; 8],
            by_class: BTreeMap::new(),
            class_accounting_scope: "unified_warp_stream_attribution",
            hotl_line_volume_clock: 0,
            hotl_line_interval_count: vec![0; HOTL_INTERVAL_HISTOGRAM_BINS],
            hotl_line_interval_sum: vec![0; HOTL_INTERVAL_HISTOGRAM_BINS],
            hotl_sector_interval_count: vec![0; HOTL_INTERVAL_HISTOGRAM_BINS],
            hotl_sector_interval_sum: vec![0; HOTL_INTERVAL_HISTOGRAM_BINS],
            hotl_line_censored_tail_count: vec![0; HOTL_INTERVAL_HISTOGRAM_BINS],
            hotl_line_censored_tail_sum: vec![0; HOTL_INTERVAL_HISTOGRAM_BINS],
            hotl_sector_censored_tail_count: vec![0; HOTL_INTERVAL_HISTOGRAM_BINS],
            hotl_sector_censored_tail_sum: vec![0; HOTL_INTERVAL_HISTOGRAM_BINS],
            hotl_line_miss_references: [0; 28],
            hotl_sector_miss_references: [0; 28],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ReuseLineState {
    pub(super) last_record_tick: u64,
    pub(super) last_sector_tick: [Option<u64>; 4],
    pub(super) last_line_volume: u64,
    pub(super) last_sector_volume: [Option<u64>; 4],
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct ReuseWarpStream {
    pub(super) record_tick: u64,
    pub(super) line_volume: u64,
    pub(super) lines: BTreeMap<i64, ReuseLineState>,
    /// Full-sector store candidates retained for the report-only same-SM
    /// store-to-load forwarding observer.  This set never changes L1/L2
    /// state or timing.
    pub(super) full_sector_store_sectors: BTreeSet<i64>,
    pub(super) active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LocalityMrcLineState {
    pub(super) last_position: u64,
    /// End position of the logical tick containing the previous line touch.
    /// All lines in one tick share this boundary even though Fenwick storage
    /// gives them unique positions for updates.
    pub(super) last_tick_end_position: u64,
    /// `None` means the sector has never been requested in this stream;
    /// `Some(threshold)` is the largest line-RD gap that could have evicted
    /// it since its last request.  A request resets its threshold to zero.
    pub(super) sector_eviction_threshold: [Option<u64>; 4],
    /// Pessimistic counterpart under arbitrary same-tick tie ordering.
    pub(super) sector_eviction_threshold_upper: [Option<u64>; 4],
    pub(super) last_batch_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LocalityMrcBatch {
    pub(super) lines: BTreeSet<i64>,
    pub(super) references: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LocalityMrcFenwick {
    /// Position zero is unused.  Each live line contributes one at its most
    /// recent position, giving an exact Mattson stack query.
    pub(super) tree: Vec<u64>,
    pub(super) active_lines: u64,
    pub(super) next_position: u64,
}

impl LocalityMrcFenwick {
    pub(super) fn new() -> Self {
        Self {
            tree: vec![0; 2],
            active_lines: 0,
            next_position: 1,
        }
    }

    pub(super) fn add_raw(&mut self, position: u64, delta: i8) -> Result<(), crate::IssueError> {
        let mut index =
            usize::try_from(position).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "locality MRC Fenwick position",
            })?;
        if index == 0 || index >= self.tree.len() {
            return Err(crate::IssueError::CounterOverflow {
                counter: "locality MRC Fenwick bound",
            });
        }
        while index < self.tree.len() {
            if delta > 0 {
                self.tree[index] = self.tree[index].checked_add(u64::from(delta as u8)).ok_or(
                    crate::IssueError::CounterOverflow {
                        counter: "locality MRC Fenwick cell",
                    },
                )?;
            } else {
                self.tree[index] = self.tree[index]
                    .checked_sub(u64::from((-delta) as u8))
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "locality MRC Fenwick cell",
                    })?;
            }
            index += index & index.wrapping_neg();
        }
        Ok(())
    }

    pub(super) fn ensure_capacity(
        &mut self,
        last_position: u64,
        existing_positions: &[u64],
    ) -> Result<(), crate::IssueError> {
        let required = usize::try_from(last_position.checked_add(1).ok_or(
            crate::IssueError::CounterOverflow {
                counter: "locality MRC Fenwick position",
            },
        )?)
        .map_err(|_| crate::IssueError::CounterOverflow {
            counter: "locality MRC Fenwick capacity",
        })?;
        if required <= self.tree.len() {
            return Ok(());
        }
        let mut length = self.tree.len().max(2);
        while length < required {
            length = length
                .checked_mul(2)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "locality MRC Fenwick capacity",
                })?;
        }
        self.tree = vec![0; length];
        for &position in existing_positions {
            self.add_raw(position, 1)?;
        }
        Ok(())
    }

    pub(super) fn prefix(&self, mut position: u64) -> Result<u64, crate::IssueError> {
        let max_position = u64::try_from(self.tree.len().saturating_sub(1)).map_err(|_| {
            crate::IssueError::CounterOverflow {
                counter: "locality MRC Fenwick prefix",
            }
        })?;
        position = position.min(max_position);
        let mut index =
            usize::try_from(position).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "locality MRC Fenwick prefix",
            })?;
        let mut total = 0_u64;
        while index != 0 {
            total =
                total
                    .checked_add(self.tree[index])
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "locality MRC Fenwick prefix sum",
                    })?;
            index &= index - 1;
        }
        Ok(total)
    }

    pub(super) fn distinct_after(&self, previous_position: u64) -> Result<u64, crate::IssueError> {
        self.active_lines
            .checked_sub(self.prefix(previous_position)?)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "locality MRC reuse distance",
            })
    }
}

impl Default for LocalityMrcFenwick {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct LocalityMrcOriginStream {
    pub(super) fenwick: LocalityMrcFenwick,
    pub(super) lines: BTreeMap<i64, LocalityMrcLineState>,
    pub(super) batches: BTreeMap<u64, LocalityMrcBatch>,
    pub(super) next_batch_id: u64,
    /// Ding–Denning/Xiang HOTL state for this exact origin-qualified warp
    /// stream.  It is deliberately separate from the Fenwick oracle's stack
    /// positions: HOTL uses the line-reference volume clock and is flushed at
    /// warp boundaries, never mixed with stores/shared/atomic classes.
    pub(super) hotl_line_volume: u64,
    pub(super) hotl_line_last: BTreeMap<i64, u64>,
    pub(super) hotl_sector_last: BTreeMap<(i64, u8), u64>,
    pub(super) hotl_sector_max_gap: BTreeMap<(i64, u8), u64>,
    pub(super) hotl_line_intervals: Vec<u64>,
    pub(super) hotl_sector_intervals: Vec<u64>,
    pub(super) hotl_cold_lines: u64,
    pub(super) hotl_cold_sectors: u64,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct LocalityMrcWarpStream {
    pub(super) origins: BTreeMap<u8, LocalityMrcOriginStream>,
}

impl LocalityMrcOriginStream {
    pub(super) fn observe_hotl(
        &mut self,
        lines: &BTreeMap<i64, u8>,
        report: &mut ReuseIntervalObservation,
    ) -> Result<(), crate::IssueError> {
        let line_count =
            u64::try_from(lines.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "HOTL line batch",
            })?;
        let start = self.hotl_line_volume;
        for (offset, (&line, &mask)) in lines.iter().enumerate() {
            let position = start
                .checked_add(u64::try_from(offset).map_err(|_| {
                    crate::IssueError::CounterOverflow {
                        counter: "HOTL line position",
                    }
                })?)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "HOTL line position",
                })?;
            let line_interval = self
                .hotl_line_last
                .get(&line)
                .map(|previous| position.saturating_sub(*previous));
            if let Some(interval) = line_interval {
                self.hotl_line_intervals.push(interval);
                add_hotl_interval(report, interval, false, false);
                // Every previously observed sector on this line is exposed
                // to the intervening line gap.  The largest such gap is the
                // coupled sector bottleneck used at flush time.
                for sector in 0..4_u8 {
                    if let Some(gap) = self.hotl_sector_max_gap.get_mut(&(line, sector)) {
                        *gap = (*gap).max(interval);
                    }
                }
            } else {
                self.hotl_cold_lines = self.hotl_cold_lines.saturating_add(1);
                add_hotl_interval(report, LOCALITY_MRC_INF, false, false);
            }
            self.hotl_line_last.insert(line, position);

            for sector in 0..4_u8 {
                let bit = 1_u8 << sector;
                if mask & bit == 0 {
                    continue;
                }
                let key = (line, sector);
                if self.hotl_sector_last.contains_key(&key) {
                    let bottleneck = self.hotl_sector_max_gap.get(&key).copied().unwrap_or(0);
                    self.hotl_sector_intervals.push(bottleneck);
                    add_hotl_interval(report, bottleneck, true, false);
                    self.hotl_sector_max_gap.insert(key, 0);
                } else {
                    self.hotl_cold_sectors = self.hotl_cold_sectors.saturating_add(1);
                    add_hotl_interval(report, LOCALITY_MRC_INF, true, false);
                    self.hotl_sector_max_gap.insert(key, 0);
                }
                self.hotl_sector_last.insert(key, position);
            }
        }
        self.hotl_line_volume = self.hotl_line_volume.checked_add(line_count).ok_or(
            crate::IssueError::CounterOverflow {
                counter: "HOTL line volume clock",
            },
        )?;
        report.hotl_line_volume_clock = report
            .hotl_line_volume_clock
            .checked_add(line_count)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "reuse HOTL volume clock",
            })?;
        Ok(())
    }

    pub(super) fn flush_hotl(&self, report: &mut ReuseIntervalObservation) {
        if self.hotl_line_volume == 0 {
            return;
        }
        // Right-censored tails remain histogram evidence only.  They are not
        // additional misses: each cold reference was already counted once,
        // and an unobserved future reuse cannot be charged as a second miss.
        for (&line, &last) in &self.hotl_line_last {
            add_hotl_interval(
                report,
                self.hotl_line_volume.saturating_sub(last),
                false,
                true,
            );
            for sector in 0..4_u8 {
                if let Some(previous) = self.hotl_sector_last.get(&(line, sector)) {
                    add_hotl_interval(
                        report,
                        self.hotl_line_volume.saturating_sub(*previous),
                        true,
                        true,
                    );
                }
            }
        }

        for (index, &capacity) in LOCALITY_MRC_CAPACITY_LINES.iter().enumerate() {
            report.hotl_line_miss_references[index] = report.hotl_line_miss_references[index]
                .saturating_add(hotl_miss_count(
                    self.hotl_cold_lines,
                    &self.hotl_line_intervals,
                    self.hotl_line_volume,
                    capacity,
                ));
            report.hotl_sector_miss_references[index] = report.hotl_sector_miss_references[index]
                .saturating_add(hotl_miss_count(
                    self.hotl_cold_sectors,
                    &self.hotl_sector_intervals,
                    self.hotl_cold_sectors.saturating_add(
                        u64::try_from(self.hotl_sector_intervals.len()).unwrap_or(u64::MAX),
                    ),
                    capacity,
                ));
        }
    }

    pub(super) fn observe(
        &mut self,
        lines: &BTreeMap<i64, u8>,
        report: &mut LocalityMrcClassObservation,
    ) -> Result<(), crate::IssueError> {
        let line_count =
            u64::try_from(lines.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "locality MRC line batch",
            })?;
        if line_count == 0 {
            return Ok(());
        }

        // Query every line before inserting any line from this logical tick.
        // The batch-first distance is the optimistic lower endpoint.  The
        // upper endpoint adds the distinct same-tick lines that could fall
        // inside the interval under another legal intra-tick ordering.
        let mut observations = Vec::with_capacity(lines.len());
        for (&line, &mask) in lines {
            let previous = self.lines.get(&line).copied();
            let line_rd = if let Some(previous) = previous {
                self.fenwick
                    .distinct_after(previous.last_tick_end_position)?
            } else {
                LOCALITY_MRC_INF
            };
            let line_rd_upper = if let Some(previous) = previous {
                let mut extras = BTreeSet::new();
                let previous_batch_lines = self
                    .batches
                    .get(&previous.last_batch_id)
                    .map(|batch| batch.lines.iter().copied().collect::<Vec<_>>())
                    .unwrap_or_default();
                for candidate in previous_batch_lines
                    .iter()
                    .copied()
                    .chain(lines.keys().copied())
                {
                    if candidate == line {
                        continue;
                    }
                    let already_in_intervening = self
                        .lines
                        .get(&candidate)
                        .is_some_and(|state| state.last_position > previous.last_tick_end_position);
                    if !already_in_intervening {
                        extras.insert(candidate);
                    }
                }
                line_rd
                    .checked_add(u64::try_from(extras.len()).map_err(|_| {
                        crate::IssueError::CounterOverflow {
                            counter: "locality MRC same-tick envelope",
                        }
                    })?)
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "locality MRC same-tick envelope",
                    })?
            } else {
                LOCALITY_MRC_INF
            };

            let mut sector_rd = [LOCALITY_MRC_INF; 4];
            let mut sector_rd_upper = [LOCALITY_MRC_INF; 4];
            if let Some(previous) = self.lines.get_mut(&line) {
                for (threshold, threshold_upper) in previous
                    .sector_eviction_threshold
                    .iter_mut()
                    .zip(previous.sector_eviction_threshold_upper.iter_mut())
                {
                    if let Some(value) = threshold {
                        *value = (*value).max(line_rd);
                    }
                    if let Some(value) = threshold_upper {
                        *value = (*value).max(line_rd_upper);
                    }
                }
                for sector in 0..4 {
                    if mask & (1_u8 << sector) == 0 {
                        continue;
                    }
                    sector_rd[sector] =
                        previous.sector_eviction_threshold[sector].unwrap_or(LOCALITY_MRC_INF);
                    sector_rd_upper[sector] = previous.sector_eviction_threshold_upper[sector]
                        .unwrap_or(LOCALITY_MRC_INF);
                    // The current request validates this sector after the
                    // query; future line gaps start from a clean threshold.
                    previous.sector_eviction_threshold[sector] = Some(0);
                    previous.sector_eviction_threshold_upper[sector] = Some(0);
                }
            }
            observations.push((
                line,
                mask,
                line_rd,
                line_rd_upper,
                sector_rd,
                sector_rd_upper,
            ));
        }

        report.same_tick_max_batch_lines = report.same_tick_max_batch_lines.max(line_count);
        for &(_, mask, line_rd, line_rd_upper, sector_rd, sector_rd_upper) in &observations {
            add(&mut report.line_references, 1)?;
            let sectors = u64::from(mask.count_ones());
            add(&mut report.sector_references, sectors)?;
            if line_rd == LOCALITY_MRC_INF {
                add(&mut report.cold_line_references, 1)?;
            }
            if let Some(width) = line_rd_upper.checked_sub(line_rd) {
                report.maximum_line_rd_envelope_width =
                    report.maximum_line_rd_envelope_width.max(width);
            }
            for capacity_index in 0..LOCALITY_MRC_CAPACITY_LINES.len() {
                let capacity = LOCALITY_MRC_CAPACITY_LINES[capacity_index];
                if line_rd >= capacity {
                    add(&mut report.line_miss_references[capacity_index], 1)?;
                }
                if line_rd_upper >= capacity {
                    add(&mut report.line_miss_references_upper[capacity_index], 1)?;
                }
                for sector in 0..4 {
                    if mask & (1_u8 << sector) == 0 {
                        continue;
                    }
                    let sector_distance = sector_rd[sector];
                    if sector_distance >= capacity {
                        add(&mut report.sector_miss_references[capacity_index], 1)?;
                        if line_rd >= capacity {
                            add(
                                &mut report.sector_line_tag_miss_references[capacity_index],
                                1,
                            )?;
                        } else {
                            add(
                                &mut report.sector_resident_fill_miss_references[capacity_index],
                                1,
                            )?;
                        }
                    }
                    let sector_distance_upper = sector_rd_upper[sector];
                    if sector_distance_upper >= capacity {
                        add(&mut report.sector_miss_references_upper[capacity_index], 1)?;
                        if line_rd_upper >= capacity {
                            add(
                                &mut report.sector_line_tag_miss_references_upper[capacity_index],
                                1,
                            )?;
                        } else {
                            add(
                                &mut report.sector_resident_fill_miss_references_upper
                                    [capacity_index],
                                1,
                            )?;
                        }
                    }
                }
            }
            for sector in 0..4 {
                if mask & (1_u8 << sector) != 0 && sector_rd[sector] == LOCALITY_MRC_INF {
                    add(&mut report.cold_sector_references, 1)?;
                }
            }
        }

        let first_position = self.fenwick.next_position;
        let last_position = first_position.checked_add(line_count - 1).ok_or(
            crate::IssueError::CounterOverflow {
                counter: "locality MRC next position",
            },
        )?;
        let existing_positions = self
            .lines
            .values()
            .map(|state| state.last_position)
            .collect::<Vec<_>>();
        self.fenwick
            .ensure_capacity(last_position, &existing_positions)?;

        let batch_id = self.next_batch_id;
        let next_batch_id = batch_id
            .checked_add(1)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "locality MRC batch id",
            })?;
        self.batches.insert(
            batch_id,
            LocalityMrcBatch {
                lines: lines.keys().copied().collect(),
                references: line_count,
            },
        );

        for (offset, &(line, mask, _, _, _, _)) in observations.iter().enumerate() {
            let position = first_position
                .checked_add(u64::try_from(offset).map_err(|_| {
                    crate::IssueError::CounterOverflow {
                        counter: "locality MRC batch offset",
                    }
                })?)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "locality MRC next position",
                })?;
            let previous_batch_id = self.lines.get(&line).map(|state| state.last_batch_id);
            if let Some(previous) = self.lines.get(&line) {
                self.fenwick.add_raw(previous.last_position, -1)?;
            } else {
                self.fenwick.active_lines = self.fenwick.active_lines.checked_add(1).ok_or(
                    crate::IssueError::CounterOverflow {
                        counter: "locality MRC active lines",
                    },
                )?;
            }
            if let Some(previous_batch_id) = previous_batch_id {
                let remove_previous = {
                    let previous_batch = self.batches.get_mut(&previous_batch_id).ok_or(
                        crate::IssueError::CounterOverflow {
                            counter: "locality MRC batch history",
                        },
                    )?;
                    previous_batch.references = previous_batch.references.checked_sub(1).ok_or(
                        crate::IssueError::CounterOverflow {
                            counter: "locality MRC batch references",
                        },
                    )?;
                    previous_batch.references == 0
                };
                if remove_previous {
                    self.batches.remove(&previous_batch_id);
                }
            }
            self.fenwick.add_raw(position, 1)?;
            self.lines
                .entry(line)
                .and_modify(|state| {
                    state.last_position = position;
                    state.last_batch_id = batch_id;
                })
                .or_insert_with(|| {
                    let mut state = LocalityMrcLineState {
                        last_position: position,
                        last_tick_end_position: last_position,
                        sector_eviction_threshold: [None; 4],
                        sector_eviction_threshold_upper: [None; 4],
                        last_batch_id: batch_id,
                    };
                    for sector in 0..4 {
                        if mask & (1_u8 << sector) != 0 {
                            state.sector_eviction_threshold[sector] = Some(0);
                            state.sector_eviction_threshold_upper[sector] = Some(0);
                        }
                    }
                    state
                });
            if let Some(state) = self.lines.get_mut(&line) {
                state.last_tick_end_position = last_position;
                // Existing lines were updated only for requested sectors in
                // the query pass.  Keep unrequested sector state intact.
                for sector in 0..4 {
                    if mask & (1_u8 << sector) != 0 {
                        state.sector_eviction_threshold[sector] = Some(0);
                        state.sector_eviction_threshold_upper[sector] = Some(0);
                    }
                }
            }
        }
        self.next_batch_id = next_batch_id;
        self.fenwick.next_position =
            last_position
                .checked_add(1)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "locality MRC next position",
                })?;
        Ok(())
    }
}

impl LocalityMrcWarpStream {
    pub(super) fn flush_hotl(&self, report: &mut ReuseIntervalObservation) {
        for origin in self.origins.values() {
            origin.flush_hotl(report);
        }
    }
}

pub(super) fn is_empty_string_map(value: &BTreeMap<String, u64>) -> bool {
    value.is_empty()
}

pub(super) fn is_empty_string_u64_map(value: &BTreeMap<String, u64>) -> bool {
    value.is_empty()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ClassCountInterval {
    pub lower: u64,
    pub upper: u64,
}

/// Independent per-class intervals are not jointly attainable extremes and
/// do not assert how any particular dynamic instruction was physically routed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PipelineClassEnvelope {
    pub role: &'static str,
    pub evidence: sim_semantics::EvidenceKind,
    pub total_warp_instructions: u64,
    pub unclassified_instructions: u64,
    pub unlisted_classes_possible: bool,
    pub counts: BTreeMap<String, ClassCountInterval>,
}

/// Lane-consistent dynamic control-flow graph reconstructed from consecutive
/// trace records. Disjoint active masks are retained as path-switch evidence
/// rather than fabricated as an edge between serialized divergent paths.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ControlFlowGraph {
    pub observed_edges: u64,
    pub unique_edges: u64,
    pub active_lane_transitions: u64,
    pub disjoint_path_switches: u64,
    pub edges: BTreeMap<String, u64>,
}

/// Trace-derived cross-SM line-sharing domain.  The observer is intentionally
/// static/topological: it reports lines whose decoded global accesses are
/// issued by more than one traced SM and whether those accesses contain both
/// read and write classes.  It does not infer temporal ownership, cache
/// invalidations, or completed coherence traffic.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CoherenceObservation {
    pub decoded_global_lines: u64,
    pub cross_sm_lines: u64,
    pub cross_sm_sectors: u64,
    /// 32-byte sectors touched by more than one traced SM.  This is a
    /// tighter contention domain than `cross_sm_sectors`, which intentionally
    /// includes every sector on a line shared at line granularity.
    pub cross_sm_sector_overlap: u64,
    pub cross_sm_read_write_lines: u64,
    pub cross_sm_write_write_lines: u64,
    /// Per-sector access-class overlap.  These are trace-topology
    /// prerequisites for protocol work, not coherence transactions.
    pub cross_sm_read_write_sectors: u64,
    pub cross_sm_write_write_sectors: u64,
    pub unknown_sm_or_line_records: u64,
    /// Count of global records carrying the producer's warp-local
    /// instrumentation-order token. This never orders different warps/SMs.
    pub instrumentation_order_records: u64,
    pub instrumentation_order_missing_records: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CoherenceLineState {
    pub(super) sm_mask: u128,
    pub(super) read_sm_mask: u128,
    pub(super) write_sm_mask: u128,
    pub(super) sector_mask: u8,
    pub(super) sector_sm_masks: [u128; 4],
    pub(super) sector_read_sm_masks: [u128; 4],
    pub(super) sector_write_sm_masks: [u128; 4],
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct TraceMetrics {
    /// Captured launch metadata, not inferred from lane activity or NCU.
    pub launch_resources: Option<LaunchResources>,
    /// True when the SM86 report-only write-tag observer was instantiated for
    /// this launch.  Metric projection must never read its counter when the
    /// producer was disabled; disabled launches fall back to the physical
    /// line-tag store observation instead.
    pub l1_write_lookup_observer_active: bool,
    pub instructions: InstructionActivity,
    pub by_opcode: BTreeMap<String, InstructionActivity>,
    pub memory: BTreeMap<String, MemoryTraffic>,
    /// Passive Chen-Ding-style reuse-interval evidence. This is intentionally
    /// separate from `memory` and from all scored cache/timing counters.
    pub reuse_interval: ReuseIntervalObservation,
    /// Exact, origin-qualified Class-A locality/MRC evidence. This remains a
    /// report-only shadow observer and never feeds the live cache or scheduler.
    pub locality_mrc: LocalityMrcObservation,
    /// Tier-2 per-SM cycle-interleaved conditional L1 shadow.  It is bound to
    /// an architecture card by replay setup and remains report-only.
    pub sm_l1_locality_observer: ConditionalLocalityObservation,
    /// Tier-3 global timestamp-ordered conditional L2 shadow.  Cross-SM
    /// arrival order is unavailable in the trace-local observer, so the
    /// unavailability count is explicit rather than silently ordering warps.
    pub l2_system_locality_observer: ConditionalLocalityObservation,
    /// Trace-qualified full-sector store-to-load forwarding candidates.  This
    /// is subtracted only from shadow L2 request projections; it never
    /// suppresses a physical request or changes response timing.
    pub store_to_load_forwardable_sectors: u64,
    #[serde(skip)]
    pub(crate) reuse_warp_stream: ReuseWarpStream,
    #[serde(skip)]
    pub(crate) locality_mrc_warp_stream: LocalityMrcWarpStream,
    pub local_interleaved_load_requests: u64,
    pub local_interleaved_load_sectors: u64,
    pub local_interleaved_store_requests: u64,
    pub local_interleaved_store_sectors: u64,
    pub local_interleaved_dynamic_records: u64,
    pub local_interleaved_dynamic_store_records: u64,
    pub local_interleaved_l1_load_hits: u64,
    pub local_interleaved_l1_load_misses: u64,
    pub local_interleaved_l1_store_hits: u64,
    pub local_interleaved_l1_store_misses: u64,
    pub local_interleaved_l2_store_hits: u64,
    pub local_interleaved_l2_store_misses: u64,
    #[serde(skip)]
    pub(crate) local_interleaved_l1_lines: BTreeMap<i64, (u8, u64)>,
    #[serde(skip)]
    pub(crate) local_interleaved_l2_lines: BTreeMap<i64, u8>,
    #[serde(skip)]
    pub(crate) local_interleaved_l1_stamp: u64,
    /// Source-derived partial store sectors. This is a potential
    /// read-modify-write demand bound and never enters DRAM accounting alone.
    pub partial_store_sectors: u64,
    pub by_pipeline: BTreeMap<String, InstructionActivity>,
    /// Static-SASS scope partitions for post-replay opcode crosswalks. The
    /// ordinary `memory` map intentionally retains the broad logical traffic
    /// classes used by L1/L2 sector accounting; this map keeps LDG modifier
    /// forms and LDGSTS separate so a narrow NCU opcode endpoint is never
    /// compared against a mixed logical class.
    pub memory_scope: BTreeMap<String, MemoryTraffic>,
    /// Dynamic static-SASS constant-client instruction populations. These are
    /// kept separate from address-domain constant sectors because the native
    /// trace contract may omit CBANK addresses.
    pub constant_client_ldc_instructions: u64,
    pub constant_client_uldc_instructions: u64,
    /// NCU-style logical pipe projections. These are deliberately separate
    /// from the issue scheduler's resource classes above.
    pub by_ncu_pipeline: BTreeMap<String, InstructionActivity>,
    /// Logical LDGSTS cache-route counts derived from the qualified static
    /// route. These are not dynamic physical cache-access counters.
    pub ldgsts_cache_access_instructions: u64,
    pub ldgsts_cache_bypass_instructions: u64,
    /// Conservative physical-pipeline candidate domains. These are not NCU
    /// counters; they quantify how much dynamic pipe attribution the trace
    /// contract can support without value or hardware-selection state.
    pub physical_pipeline: PhysicalPipelineAttribution,
    /// Static operand-role coverage and (for SM86 only) bank-demand envelope;
    /// intentionally separate from physical pipeline attribution.
    pub operand_bank_pressure: OperandBankPressure,
    pub control_flow: ControlFlowGraph,
    /// Conservative cross-SM sharing domain; not a coherence protocol or
    /// profiler writeback counter.
    pub coherence: CoherenceObservation,
    #[serde(skip)]
    pub(crate) coherence_lines: BTreeMap<i64, CoherenceLineState>,
    pub branch_warp_instructions: u64,
    pub branch_mixed_predicate_warp_instructions: u64,
    /// Actual per-lane successors, independent of branch predicates.
    pub branch_targets: BranchTargets,
    pub sectors_decoded: bool,
    /// Host wall time spent locating/parsing the typed TMA/sync sidecars for
    /// this launch. This is a replay-cost diagnostic, not a GPU timestamp.
    pub sidecar_join_wall_ns: u64,
    /// Producer-to-successful-wait instruction relations from runtime barrier
    /// identities. Multiple generations remain unresolved, never fabricated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub async_relations: Option<sim_trace::async_sidecars::AsyncRelations>,
    /// Typed asynchronous-obligation frontier. This is an observation/model
    /// diagnostic only; it never contributes cycles, issue readiness, cache
    /// service, or NCU comparison values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub async_obligations: Option<sim_trace::obligations::ObligationFrontier>,
    /// Read-only scalar async-group decision snapshots from the issue model.
    /// They cover selected issue opportunities only; they are not a census of
    /// blocked candidates or a same-state blocked witness.  They are dynamic
    /// model-state evidence, not hardware completion clocks, and never feed
    /// scheduling.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub async_decision_snapshots: Vec<crate::AsyncDecisionSnapshot>,
    /// Launch-level cap for the aggregated snapshot vector.  Per-SM caps do
    /// not bound a multi-SM launch, so omitted records are retained as an
    /// explicit count rather than silently growing the report.
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub async_decision_snapshots_dropped: u64,
    /// TMA payload observations joined from the producer sidecar. Missing
    /// sidecar records remain explicitly unavailable; these counters never
    /// enter NCU comparison or synthesize completion timestamps.
    pub tma_sidecar_records_observed: u64,
    pub tma_sidecar_payload_bytes_observed: u64,
    pub tma_sidecar_records_missing: u64,
    pub tma_sidecar_payload_bytes_missing: u64,
    pub tma_sidecar_load_records: u64,
    pub tma_sidecar_store_records: u64,
    /// Payload bytes partitioned by transfer direction. The load partition is
    /// a conditional lower-level demand input for effective SM120 memory
    /// scenarios; it is never an engine-completion timestamp.
    pub tma_sidecar_load_payload_bytes: u64,
    pub tma_sidecar_store_payload_bytes: u64,
    pub tma_sidecar_zero_byte_records: u64,
    pub tma_sidecar_oob_or_empty_records: u64,
    /// Quarantined sidecar provenance diagnostics. These counters are
    /// descriptive only and never feed demand, cache state, timing, or NCU
    /// target selection.
    pub tma_sidecar_transfer_product_matched_records: u64,
    pub tma_sidecar_transfer_product_mismatched_records: u64,
    pub tma_sidecar_raw_valid_address_refs: u64,
    pub tma_sidecar_raw_oob_address_refs: u64,
    pub tma_sidecar_unique_sector_count: u64,
    pub tma_sidecar_unique_line_count: u64,
    pub tma_sidecar_intra_record_duplicate_sector_refs: u64,
    pub tma_sidecar_nominal_box_overage_bytes_diagnostic: u64,
    /// Launch-level decomposition of sidecar address multiplicity. These
    /// counters separate intra-record byte/address fanout from true
    /// inter-record sector and line reuse; they are diagnostic only.
    pub tma_sidecar_sum_record_unique_sectors: u64,
    pub tma_sidecar_sum_record_unique_lines: u64,
    pub tma_sidecar_inter_record_duplicate_sectors: u64,
    pub tma_sidecar_inter_record_shared_lines: u64,
    pub tma_sidecar_intra_record_duplicate_refs: u64,
    pub tma_sidecar_load_sum_record_unique_sectors: u64,
    pub tma_sidecar_store_sum_record_unique_sectors: u64,
    pub tma_sidecar_load_sum_record_unique_lines: u64,
    pub tma_sidecar_store_sum_record_unique_lines: u64,
    pub tma_sidecar_load_inter_record_duplicate_sectors: u64,
    pub tma_sidecar_store_inter_record_duplicate_sectors: u64,
    pub tma_sidecar_cross_direction_shared_sectors: u64,
    pub tma_sidecar_cross_direction_shared_lines: u64,
    /// Hardware-clock observation brackets derived from existing sidecars.
    /// These are interval evidence, never a completion clock or replay input.
    pub async_clock_sandwich_pairs_observed: u64,
    pub async_clock_sandwich_invalid_pairs: u64,
    pub async_clock_sandwich_lower_span_sum: u64,
    pub async_clock_sandwich_upper_span_sum: u64,
    /// Address-derived conditional TMA/L2 shadow stream. These counters are
    /// separate from the shared replay L2 state and never affect scheduling,
    /// response timing, or DRAM service.
    pub tma_shadow_l2_requests: u64,
    pub tma_shadow_l2_hits: u64,
    pub tma_shadow_l2_misses: u64,
    pub tma_shadow_l2_address_unavailable: u64,
    pub tma_shadow_l2_load_requests: u64,
    pub tma_shadow_l2_load_hits: u64,
    pub tma_shadow_l2_load_misses: u64,
    pub tma_shadow_l2_load_address_unavailable: u64,
    pub tma_shadow_l2_store_requests: u64,
    pub tma_shadow_l2_store_hits: u64,
    pub tma_shadow_l2_store_misses: u64,
    pub tma_shadow_l2_store_address_unavailable: u64,
    /// Diagnostic replay of the same address-derived TMA/L2 shadow in the
    /// sidecar's first-seen record order.  The lexical keyed-map shadow above
    /// remains the scored endpoint; these counters never affect cache state,
    /// timing, or NCU comparison selection.
    pub tma_shadow_l2_ordered_requests: u64,
    pub tma_shadow_l2_ordered_hits: u64,
    pub tma_shadow_l2_ordered_misses: u64,
    pub tma_shadow_l2_ordered_address_unavailable: u64,
    pub tma_shadow_l2_ordered_load_requests: u64,
    pub tma_shadow_l2_ordered_load_hits: u64,
    pub tma_shadow_l2_ordered_load_misses: u64,
    pub tma_shadow_l2_ordered_load_address_unavailable: u64,
    pub tma_shadow_l2_ordered_store_requests: u64,
    pub tma_shadow_l2_ordered_store_hits: u64,
    pub tma_shadow_l2_ordered_store_misses: u64,
    pub tma_shadow_l2_ordered_store_address_unavailable: u64,
    /// Diagnostic-only joint L2 shadow interleaving ordinary dynamic sectors
    /// with sidecar TMA sectors when a shared instrumentation-order token is
    /// present. It never changes replay state, timing, or scored endpoints.
    pub tma_joint_l2_requests: u64,
    pub tma_joint_l2_hits: u64,
    pub tma_joint_l2_misses: u64,
    pub tma_joint_l2_tma_load_requests: u64,
    pub tma_joint_l2_tma_load_hits: u64,
    pub tma_joint_l2_tma_load_misses: u64,
    pub tma_joint_l2_ordinary_read_requests: u64,
    pub tma_joint_l2_ordinary_read_hits: u64,
    pub tma_joint_l2_ordinary_read_misses: u64,
    pub tma_joint_l2_order_unavailable: u64,
    pub tma_joint_l2_events_dropped: u64,
    /// Read-only L1 request-boundary merge provenance. These counters are
    /// diagnostic; they do not change cache state or lookup projections.
    pub l1_same_request_merges_global: u64,
    pub l1_same_request_merges_local: u64,
    pub l1_already_requested_merges_global: u64,
    pub l1_already_requested_merges_local: u64,
    pub l1_unrequested_pending_merges_global: u64,
    pub l1_unrequested_pending_merges_local: u64,
    /// Ordinary (non-LDGSTS) request-boundary merges. Keeping this subset
    /// separate prevents the LDGSTS client from contaminating the ordinary
    /// global-LDG packet-aware lookup candidate.
    pub l1_ordinary_same_request_merges_global: u64,
    pub l1_ordinary_same_request_merges_local: u64,
    pub l1_async_same_request_merges_global: u64,
    pub l1_async_same_request_merges_local: u64,
    pub l1_older_pending_merges_global: u64,
    pub l1_older_pending_merges_local: u64,
    pub l1_ordinary_older_pending_merges_global: u64,
    pub l1_ordinary_older_pending_merges_local: u64,
    pub l1_async_older_pending_merges_global: u64,
    pub l1_async_older_pending_merges_local: u64,
    pub l1_ordinary_global_older_pending_merges: u64,
    pub l1_constant_global_older_pending_merges: u64,
    pub l1_ordinary_global_same_request_merges: u64,
    pub l1_constant_global_same_request_merges: u64,
    pub l1_async_global_same_request_merges: u64,
    pub l1_ordinary_global_hits: u64,
    pub l1_async_global_hits: u64,
    pub l1_ordinary_global_older_sector_pending_merges: u64,
    pub l1_constant_global_older_sector_pending_merges: u64,
    pub l1_async_global_older_sector_pending_merges: u64,
    pub l1_unrequested_subline_miss_sectors: u64,
    pub l1_wavefront_mask_lookup_hits: u64,
    pub l1_wavefront_mask_lookup_misses: u64,
    pub l1_constant_global_hits: u64,
    pub l1_constant_global_requests: u64,
    pub l1_global_resident_hits: u64,
    pub l1_global_lookup_sectors: u64,
    /// Adjacent full-warp global-store records that touch a sector already
    /// emitted by the immediately preceding global-store record.  SM120's
    /// write-combining observer can absorb these duplicate packets before a
    /// new operation-qualified L2 lookup; this is a trace-only projection and
    /// never changes physical cache state or DRAM service.
    pub global_store_adjacent_duplicate_sectors: u64,
    #[serde(skip)]
    pub(crate) global_store_previous_sectors: BTreeSet<i64>,
    #[serde(skip)]
    pub(crate) global_store_previous_effective_lanes: u8,
    /// Launch-level evidence for the admitted full-warp LDS.128 observer
    /// projection. These fields are trace topology only.
    pub shared_load_full_warp_width4_records: u64,
    pub shared_load_full_warp_width16_records: u64,
    pub shared_load_width16_observer_reduction: u64,
    pub shared_load_width16_excess_reduction: u64,
    /// Fixed-size per-request four-sector mask/client audit. This is
    /// diagnostic-only and does not feed cache state, timing, or scored
    /// endpoints.
    pub l1_request_mask_audit: L1RequestMaskAudit,
    /// Dynamic constant-bank demand observations. Addresses are producer
    /// offsets, not device allocations; these counters are demand bounds only
    /// and never feed cache state, timing, or NCU-shaped predictions.
    pub constant_memory_address_count: u64,
    pub constant_memory_reference_count: u64,
    pub constant_memory_sector_requests: u64,
    pub constant_memory_unique_sectors: u64,
    /// Launch-wide distinct 64-B constant-bank candidate lines.  This is a
    /// source-domain cardinality only; no line-fill or DRAM event is implied.
    pub constant_memory_unique_lines: u64,
    /// Constant records that carried bank/offset metadata but no active,
    /// predicate-on lane offsets.  These are retained as an explicit
    /// unavailable-demand observation; field 17 is one scalar metadata value,
    /// not a substitute for the lane vector.
    pub constant_memory_metadata_only_records: u64,
    #[serde(skip)]
    pub(crate) constant_memory_sector_set: BTreeSet<(Option<u32>, u64)>,
    #[serde(skip)]
    pub(crate) constant_memory_line_set: BTreeSet<(Option<u32>, u64)>,
    /// Static `LDG.*.CONSTANT` source-qualified demand candidates. These are
    /// kept separate from CBANK field-15 observations and from L1/L2 state.
    pub constant_global_load_instructions: u64,
    pub constant_global_load_sectors: u64,
    /// Static modifier/client form for LDG.*.CONSTANT observers. Index order
    /// is LTC128B, U16, other/plain; this does not relocate traffic or claim
    /// that any form uses Huerta's L0C path.
    pub constant_global_load_form_instructions: [u64; 3],
    pub constant_global_load_form_sectors: [u64; 3],
    pub constant_global_load_request_candidates: u64,
    pub constant_global_load_unique_lines: u64,
    pub constant_global_load_cta_unique_lines: u64,
    /// Static LDG.CONSTANT records whose active lanes present one identical
    /// byte address/width. This is a source-class witness for a possible
    /// uniform GCC path; it is not itself a GCC request or DRAM transaction.
    pub constant_global_load_uniform_instructions: u64,
    pub constant_global_load_uniform_sectors: u64,
    pub constant_global_load_divergent_instructions: u64,
    pub constant_global_load_divergent_sectors: u64,
    #[serde(skip)]
    pub(crate) constant_global_line_set: BTreeSet<([u32; 3], i32, u64)>,
    #[serde(skip)]
    pub(crate) constant_global_cta_line_set: BTreeSet<([u32; 3], u64)>,
    /// Atomic same-sector conflict exposure. These are exact trace-sector
    /// alias observations and do not impose a latency or serialization rule.
    pub atomic_conflict_instructions: u64,
    pub atomic_distinct_instructions: u64,
    pub atomic_conflict_alias_sectors: u64,
    pub local_width_records: u64,
    pub local_width_unknown_records: u64,
    pub local_width_bytes: u64,
    pub local_width_candidate_sectors: u64,
    /// Observer-only local sector domains after applying the effective SM86
    /// lane-private frame expansion. The ordinary local memory map remains
    /// the replay/timing source; these candidates are selected only for the
    /// like-for-like local L1 endpoint when present.
    pub local_canonical_load_sectors: u64,
    pub local_canonical_store_sectors: u64,
    pub local_frame256_load_sectors: u64,
    pub local_frame256_store_sectors: u64,
    pub local_payload_load_sectors: u64,
    pub local_payload_store_sectors: u64,

    /// Sidecar-to-dynamic identity coverage for source-order reasoning. A
    /// matched key establishes same-instruction identity, not cross-warp time.
    pub tma_shadow_dynamic_records_matched: u64,
    pub tma_shadow_dynamic_records_unmatched: u64,
    /// Typed synchronization-sidecar observations. These are semantic
    /// joins only; they do not manufacture mbarrier completion timestamps.
    pub sync_sidecar_records_observed: u64,
    pub sync_sidecar_records_missing: u64,
    pub sync_sidecar_before_records: u64,
    pub sync_sidecar_after_records: u64,
    pub sync_sidecar_value_incomplete_records: u64,
    pub sync_sidecar_mbarrier_address_records: u64,
    pub sync_sidecar_mbarrier_register_records: u64,
    pub sync_sidecar_arrival_records: u64,
    pub sync_sidecar_phase_check_records: u64,
    pub sync_sidecar_wait_records: u64,
    pub sync_sidecar_exchange_records: u64,
    /// Sum of per-instruction distinct addresses, NOT launch-wide unique barriers.
    pub sync_sidecar_barrier_addresses: u64,
}

/// Warp-local control-transfer observations. Counts include fallthrough;
/// EXIT and convergence setup/waits are not target-bearing branch events.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct BranchTargets {
    pub events: u64,
    pub uniform_events: u64,
    pub divergent_events: u64,
    pub distinct_successors: u64,
    /// All active threads in a divergent event, not the minority path size.
    pub divergent_threads: u64,
    /// Sum of active lanes across all resolved successor events.
    pub successor_active_threads: u64,
    /// Sum of the smallest successor-path population for resolved divergent
    /// events. This is a trace-derived minority-path diagnostic, not an NCU
    /// branch counter.
    pub divergent_minority_threads: u64,
    /// Active lanes in resolved uniform successor events.
    pub uniform_active_threads: u64,
    /// At least one active lane had no recorded successor before trace end.
    pub unresolved_events: u64,
    /// Active lanes whose first successor was not observed before trace end.
    pub unresolved_threads: u64,
}

pub(super) struct PendingBranch {
    pub(super) remaining: u32,
    pub(super) active_threads: u32,
    pub(super) targets: Vec<(i32, u32, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LaunchResources {
    pub block_threads: u32,
    pub grid_blocks: u64,
    pub registers_per_thread: u32,
    pub shared_memory_bytes_per_block: u32,
    /// Legacy captures supply only the total; absence must not become zero.
    pub static_shared_memory_bytes: Option<u32>,
    pub dynamic_shared_memory_bytes: Option<u32>,
    pub shared_memory_carveout_preference_pct: Option<i32>,
    pub driver_occupancy_limit_blocks: Option<u32>,
}

pub(super) fn add(dst: &mut u64, value: u64) -> Result<(), crate::IssueError> {
    *dst = dst
        .checked_add(value)
        .ok_or(crate::IssueError::CounterOverflow {
            counter: "native metric counter",
        })?;
    Ok(())
}

/// Return the distinct-sector and distinct-line topology for one decoded
/// request span.  The arena is already validated by the trace reader; this
/// helper stays fail-closed for compact inputs and legacy flat traces whose
/// arena is intentionally absent.
pub(super) fn decoded_sector_stats(kind: AccessKind, sector_arena: &[i64]) -> Option<(u64, u64)> {
    let (off, len) = match kind {
        AccessKind::LocalLoad { off, len }
        | AccessKind::LocalStore { off, len }
        | AccessKind::GlobalEligible { off, len, .. }
        | AccessKind::GlobalAsyncCopy { off, len, .. }
        | AccessKind::GlobalBypass { off, len, .. }
        | AccessKind::GlobalUntagged { off, len }
        | AccessKind::GlobalStore { off, len }
        | AccessKind::GlobalReduction { off, len, .. } => (off, usize::from(len)),
        AccessKind::Shared { .. } | AccessKind::NonMemory | AccessKind::PredicatedOffMemory => {
            return Some((0, 0));
        }
    };
    let end = off.checked_add(len)?;
    let span = sector_arena.get(off..end)?;
    let mut sorted = span.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut unique_lines = 0_u64;
    let mut previous_line = None;
    for sector in sorted.iter().copied() {
        let line = sector.div_euclid(4);
        if previous_line != Some(line) {
            unique_lines = unique_lines.saturating_add(1);
            previous_line = Some(line);
        }
    }
    Some((sorted.len() as u64, unique_lines))
}

/// Return the bounded static-SASS scope bucket for one dynamic memory record.
/// These are attribution domains, not hardware route selections. In
/// particular, LTC128B and CONSTANT forms remain visible as separate
/// observers while their existing hierarchy traffic stays in the broad
/// `memory` class.
pub(super) fn memory_scope_key(opcode: &str, kind: AccessKind) -> Option<&'static str> {
    let token = opcode.split_ascii_whitespace().next().unwrap_or("");
    let head = opcode_head(opcode);
    if matches!(kind, AccessKind::GlobalAsyncCopy { .. }) || head == "LDGSTS" {
        return Some("ldgsts_async_copy");
    }
    if !matches!(
        kind,
        AccessKind::GlobalEligible { .. }
            | AccessKind::GlobalBypass { .. }
            | AccessKind::GlobalUntagged { .. }
    ) || head != "LDG"
    {
        return None;
    }
    let modifiers = token.split('.').collect::<Vec<_>>();
    if modifiers.iter().any(|modifier| *modifier == "LTC128B") {
        Some("global_load_ltc128b")
    } else if modifiers.iter().any(|modifier| *modifier == "CONSTANT") {
        Some("global_load_constant")
    } else if modifiers
        .windows(2)
        .any(|window| window == ["STRONG", "GPU"])
    {
        Some("global_load_strong_gpu")
    } else {
        // Plain LDG.E/width forms are the closest trace-side scope to
        // smsp__sass_inst_executed_op_global_ld.sum. The exact NCU modifier
        // policy remains a post-replay crosswalk question.
        Some("global_load_ldg")
    }
}

pub(super) fn reuse_kind_class(kind: AccessKind) -> Option<&'static str> {
    match kind {
        AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalBypass {
            token:
                CacheOpToken::Other
                | CacheOpToken::LdgECg
                | CacheOpToken::LdgECs
                | CacheOpToken::LdgEStrongSm
                | CacheOpToken::LdgEStrongGpu
                | CacheOpToken::LdgEEf
                | CacheOpToken::LdgENa
                | CacheOpToken::LdgECa,
            ..
        } => Some("global_load"),
        AccessKind::GlobalStore { .. } => Some("global_store"),
        AccessKind::GlobalBypass {
            token: CacheOpToken::Atom,
            ..
        }
        | AccessKind::GlobalReduction { .. } => Some("atomic_or_reduction"),
        AccessKind::GlobalAsyncCopy { .. } => Some("async_copy"),
        AccessKind::LocalLoad { .. } | AccessKind::LocalStore { .. } => Some("local"),
        AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        // Keep any future/unknown bypass token out of the stream rather than
        // silently assigning it a read or write meaning.
        | AccessKind::GlobalBypass { .. } => None,
    }
}

pub(super) fn locality_mrc_origin(kind: AccessKind) -> Option<(u8, &'static str)> {
    match kind {
        AccessKind::LocalLoad { .. } | AccessKind::LocalStore { .. } => Some((1, "local")),
        AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalBypass { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalStore { .. }
        | AccessKind::GlobalReduction { .. }
        | AccessKind::GlobalAsyncCopy { .. } => Some((0, "global")),
        AccessKind::NonMemory | AccessKind::PredicatedOffMemory | AccessKind::Shared { .. } => None,
    }
}

pub(super) fn reuse_span(kind: AccessKind, sector_arena: &[i64]) -> Option<&[i64]> {
    let (off, len) = match kind {
        AccessKind::LocalLoad { off, len }
        | AccessKind::LocalStore { off, len }
        | AccessKind::GlobalEligible { off, len, .. }
        | AccessKind::GlobalAsyncCopy { off, len, .. }
        | AccessKind::GlobalBypass { off, len, .. }
        | AccessKind::GlobalUntagged { off, len }
        | AccessKind::GlobalStore { off, len }
        | AccessKind::GlobalReduction { off, len, .. } => (off, usize::from(len)),
        AccessKind::Shared { .. } | AccessKind::NonMemory | AccessKind::PredicatedOffMemory => {
            return None;
        }
    };
    let end = off.checked_add(len)?;
    sector_arena.get(off..end)
}

pub(super) fn add_reuse_interval(
    histogram: &mut [u64; REUSE_INTERVAL_HISTOGRAM_BINS],
    overflow: &mut u64,
    interval: u64,
) -> Result<(), crate::IssueError> {
    if interval < REUSE_INTERVAL_HISTOGRAM_BINS as u64 {
        add(&mut histogram[interval as usize], 1)
    } else {
        add(overflow, 1)
    }
}

pub(super) fn hotl_interval_bin(interval: u64) -> usize {
    if interval < 16 {
        return interval as usize;
    }
    let mut exponent = 4_u32;
    while exponent < 20 && (1_u64 << (exponent + 1)) <= interval {
        exponent += 1;
    }
    let base = 1_u64 << exponent;
    let sub_bin = (interval.saturating_sub(base).saturating_mul(4) / base).min(3) as usize;
    (16 + (exponent as usize - 4) * 4 + sub_bin).min(HOTL_INTERVAL_HISTOGRAM_BINS - 1)
}

pub(super) fn add_hotl_interval(
    observation: &mut ReuseIntervalObservation,
    interval: u64,
    sector: bool,
    censored: bool,
) {
    let bin = hotl_interval_bin(interval);
    let (counts, sums) = match (sector, censored) {
        (false, false) => (
            &mut observation.hotl_line_interval_count,
            &mut observation.hotl_line_interval_sum,
        ),
        (true, false) => (
            &mut observation.hotl_sector_interval_count,
            &mut observation.hotl_sector_interval_sum,
        ),
        (false, true) => (
            &mut observation.hotl_line_censored_tail_count,
            &mut observation.hotl_line_censored_tail_sum,
        ),
        (true, true) => (
            &mut observation.hotl_sector_censored_tail_count,
            &mut observation.hotl_sector_censored_tail_sum,
        ),
    };
    counts[bin] = counts[bin].saturating_add(1);
    sums[bin] = sums[bin].saturating_add(interval);
    // Miss curves are emitted only when an origin stream is flushed through
    // the Ding–Denning/Xiang footprint inversion.  In particular, censored
    // tails are histogram-only evidence and must never increment a miss
    // counter a second time.
    let _ = sector;
}

/// Return the HOTL miss count for one capacity using the finite-boundary
/// Ding–Denning footprint inversion.  `intervals` contains only interior
/// references; `cold` is the number of first touches.  The integer window is
/// the smallest reference volume whose footprint reaches the capacity, which
/// gives a deterministic conservative boundary for `tau > w*`.
pub(super) fn hotl_miss_count(cold: u64, intervals: &[u64], references: u64, capacity: u64) -> u64 {
    if references == 0 {
        return 0;
    }
    if capacity == 0 {
        return references;
    }
    if capacity >= cold || intervals.is_empty() {
        return cold;
    }
    let target = u128::from(capacity).saturating_mul(u128::from(references));
    let mut lower = 0_u64;
    let mut upper = references.max(1);
    while lower < upper {
        let window = lower.saturating_add(upper.saturating_sub(lower) / 2);
        let mut numerator = u128::from(cold).saturating_mul(u128::from(window));
        for &interval in intervals {
            numerator = numerator.saturating_add(u128::from(interval.min(window)));
        }
        if numerator >= target {
            upper = window;
        } else {
            lower = window.saturating_add(1);
        }
    }
    let window = lower;
    cold.saturating_add(
        u64::try_from(
            intervals
                .iter()
                .filter(|&&interval| interval > window)
                .count(),
        )
        .unwrap_or(u64::MAX),
    )
}

pub(super) fn add_proxy_misses(
    misses: &mut [u64; 8],
    interval: Option<u64>,
) -> Result<(), crate::IssueError> {
    for (index, &capacity) in REUSE_PROXY_CAPACITY_RECORDS.iter().enumerate() {
        // A cold reference has no finite reuse interval and is a miss at
        // every finite capacity. A finite interval is only a reuse-time
        // survival proxy; it is intentionally not called exact RD.
        if capacity == 0 || interval.map_or(true, |value| value >= capacity) {
            add(&mut misses[index], 1)?;
        }
    }
    Ok(())
}

pub(super) fn constant_global_load_form_index(opcode_text: &str) -> usize {
    let opcode = opcode_text.split_ascii_whitespace().next().unwrap_or("");
    if opcode.split('.').any(|modifier| modifier == "LTC128B") {
        0
    } else if opcode.split('.').any(|modifier| modifier == "U16") {
        1
    } else {
        2
    }
}

impl TraceMetrics {
    pub(crate) fn bind_conditional_locality_arch(&mut self, arch: sim_arch::ArchId) {
        let (l1_capacities, l2_capacities) = match arch {
            sim_arch::ArchId::Sm86 => (
                vec![256, 384, 512, 768, 1024],
                vec![4096, 8192, 16384, 32768, 49152],
            ),
            sim_arch::ArchId::Sm120 => (
                vec![512, 768, 1024, 2048, 4096],
                vec![16384, 49152, 131072, 393216, 786432],
            ),
        };
        self.sm_l1_locality_observer = ConditionalLocalityObservation::new(
            "sm_l1_locality_observer_v1",
            l1_capacities,
            "per_warp_trace_order_shadow_at_issue_cycle",
        );
        self.l2_system_locality_observer = ConditionalLocalityObservation::new(
            "l2_system_locality_observer_v1",
            l2_capacities,
            "global_timestamp_order_unavailable_in_native_trace",
        );
    }
    pub(crate) fn observe_l2_system_locality_summary(
        &mut self,
        lookup_requests: u64,
        lookup_hits: u64,
        pending_merge_hits: u64,
        store_lookup_requests: u64,
    ) {
        let observer = &mut self.l2_system_locality_observer;
        if observer.capacity_lines.is_empty() {
            return;
        }
        observer.references = observer.references.saturating_add(lookup_requests);
        observer.ordering_unavailable_references = observer
            .ordering_unavailable_references
            .saturating_add(lookup_requests);
        observer.in_flight_mshr_merges = observer
            .in_flight_mshr_merges
            .saturating_add(pending_merge_hits);
        observer.lease_zero_store_references = observer
            .lease_zero_store_references
            .saturating_add(store_lookup_requests);
        for index in 0..observer.capacity_lines.len() {
            observer.lower_hits[index] = observer.lower_hits[index].saturating_add(lookup_hits);
            observer.upper_hits[index] = observer.upper_hits[index].saturating_add(lookup_requests);
            observer.lower_misses[index] = observer.lower_misses[index]
                .saturating_add(lookup_requests.saturating_sub(lookup_hits));
        }
    }
    /// Start a new independent reuse stream.  The native trace gives an
    /// authoritative order only inside one warp, so state is never carried
    /// from one warp to another.
    pub(crate) fn begin_reuse_warp_stream(&mut self) {
        self.locality_mrc_warp_stream
            .flush_hotl(&mut self.reuse_interval);
        // The legacy unified reuse stream is retained for interval and
        // forwarding diagnostics, but its cross-class tails are not HOTL
        // evidence.  Only the origin-qualified streams above feed the HOTL
        // histograms and finite-boundary miss curves.
        let _previous = std::mem::take(&mut self.reuse_warp_stream);
        self.reuse_warp_stream = ReuseWarpStream::default();
        self.locality_mrc_warp_stream = LocalityMrcWarpStream::default();
    }
    /// Observe one decoded memory record in the passive reuse observer.
    ///
    /// Sectors are deduplicated before line/sector state is updated.  This is
    /// important for atomics/reductions whose compact span may preserve alias
    /// multiplicity for the timing model.  The observer deliberately uses one
    /// unified decoded-memory stream per warp; `by_class` is a classification
    /// of that stream, not an independent class-specific ordering claim.
    pub(crate) fn observe_reuse_memory(
        &mut self,
        kind: AccessKind,
        sector_arena: &[i64],
    ) -> Result<(), crate::IssueError> {
        let Some(class) = reuse_kind_class(kind) else {
            return Ok(());
        };
        let Some(span) = reuse_span(kind, sector_arena) else {
            add(&mut self.reuse_interval.records_without_decoded_span, 1)?;
            return Ok(());
        };
        if span.is_empty() {
            add(&mut self.reuse_interval.records_with_empty_span, 1)?;
            return Ok(());
        }

        // A compact span can contain duplicate sector ids for alias-expanded
        // atomics/reductions.  A BTreeMap gives deterministic line ordering
        // and produces one four-bit sector mask per coalesced line.
        let mut lines = BTreeMap::<i64, u8>::new();
        for &sector in span {
            let line = sector.div_euclid(4);
            let lane = sector.rem_euclid(4) as u8;
            lines
                .entry(line)
                .and_modify(|mask| *mask |= 1_u8 << lane)
                .or_insert(1_u8 << lane);
        }
        if lines.is_empty() {
            return Ok(());
        }

        if let Some((origin, origin_name)) = locality_mrc_origin(kind) {
            let class_observation = self
                .locality_mrc
                .by_origin
                .entry(origin_name.to_owned())
                .or_default();
            self.locality_mrc_warp_stream
                .origins
                .entry(origin)
                .or_default()
                .observe(&lines, class_observation)?;
            self.locality_mrc_warp_stream
                .origins
                .get_mut(&origin)
                .expect("origin stream inserted above")
                .observe_hotl(&lines, &mut self.reuse_interval)?;
        }

        if !self.reuse_warp_stream.active {
            self.reuse_warp_stream.active = true;
            add(&mut self.reuse_interval.streams, 1)?;
        }

        let class_observation = self
            .reuse_interval
            .by_class
            .entry(class.to_owned())
            .or_default();
        add(
            &mut class_observation.line_references,
            u64::try_from(lines.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "reuse line reference count",
            })?,
        )?;
        let sector_count = lines
            .values()
            .map(|mask| u64::from(mask.count_ones()))
            .sum::<u64>();
        add(&mut class_observation.sector_references, sector_count)?;

        let tick = self.reuse_warp_stream.record_tick;
        let volume = self.reuse_warp_stream.line_volume;
        let line_count =
            u64::try_from(lines.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "reuse line volume clock",
            })?;
        self.reuse_warp_stream.record_tick =
            tick.checked_add(1)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "reuse record tick",
                })?;
        self.reuse_warp_stream.line_volume =
            volume
                .checked_add(line_count)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "reuse line volume clock",
                })?;
        let mut conditional_intervals: Vec<Option<u64>> = Vec::new();
        for (line, mask) in lines {
            let requested_sectors = u64::from(mask.count_ones());
            let mut line_interval = None;
            let mut sector_intervals = [None; 4];
            let mut line_volume_interval = None;
            let mut sector_reused_mask = 0_u8;
            let is_new_line = if let Some(state) = self.reuse_warp_stream.lines.get_mut(&line) {
                line_interval = Some(
                    tick.checked_sub(state.last_record_tick)
                        .ok_or(crate::IssueError::CounterOverflow {
                            counter: "reuse line interval",
                        })?
                        .saturating_sub(1),
                );
                line_volume_interval = Some(
                    volume
                        .saturating_sub(state.last_line_volume)
                        .saturating_sub(1),
                );
                for sector in 0..4 {
                    let bit = 1_u8 << sector;
                    if mask & bit == 0 {
                        continue;
                    }
                    if let Some(previous) = state.last_sector_tick[sector] {
                        sector_intervals[sector] = Some(
                            tick.checked_sub(previous)
                                .ok_or(crate::IssueError::CounterOverflow {
                                    counter: "reuse sector interval",
                                })?
                                .saturating_sub(1),
                        );
                        sector_reused_mask |= bit;
                    }
                    state.last_sector_tick[sector] = Some(tick);
                    state.last_sector_volume[sector] = Some(volume);
                }
                state.last_record_tick = tick;
                state.last_line_volume = volume;
                false
            } else {
                let mut state = ReuseLineState {
                    last_record_tick: tick,
                    last_sector_tick: [None; 4],
                    last_line_volume: volume,
                    last_sector_volume: [None; 4],
                };
                for sector in 0..4 {
                    if mask & (1_u8 << sector) != 0 {
                        state.last_sector_tick[sector] = Some(tick);
                        state.last_sector_volume[sector] = Some(volume);
                    }
                }
                self.reuse_warp_stream.lines.insert(line, state);
                true
            };

            if let Some(interval) = line_volume_interval {
                conditional_intervals.push(Some(interval));
            } else {
                conditional_intervals.push(None);
            }
            for sector in 0..4 {
                if mask & (1_u8 << sector) == 0 {
                    continue;
                }
            }

            add(&mut self.reuse_interval.line_references, 1)?;
            add(
                &mut self.reuse_interval.sector_references,
                requested_sectors,
            )?;
            if is_new_line {
                add(&mut self.reuse_interval.stream_unique_lines, 1)?;
                add(&mut self.reuse_interval.cold_line_references, 1)?;
                add(&mut class_observation.cold_line_references, 1)?;
                add_proxy_misses(&mut self.reuse_interval.line_proxy_miss_references, None)?;
            } else {
                add(&mut self.reuse_interval.reused_line_references, 1)?;
                add(&mut class_observation.reused_line_references, 1)?;
                let interval = line_interval.expect("existing line has interval");
                add(&mut self.reuse_interval.line_reuse_interval_sum, interval)?;
                add_reuse_interval(
                    &mut self.reuse_interval.line_reuse_interval_histogram,
                    &mut self.reuse_interval.line_reuse_interval_overflow,
                    interval,
                )?;
                add_proxy_misses(
                    &mut self.reuse_interval.line_proxy_miss_references,
                    Some(interval),
                )?;
            }

            let mut cold_sectors =
                requested_sectors.saturating_sub(u64::from(sector_reused_mask.count_ones()));
            if is_new_line {
                cold_sectors = requested_sectors;
            }
            let reused_sectors = requested_sectors.saturating_sub(cold_sectors);
            add(
                &mut self.reuse_interval.cold_sector_references,
                cold_sectors,
            )?;
            add(
                &mut self.reuse_interval.reused_sector_references,
                reused_sectors,
            )?;
            add(&mut class_observation.cold_sector_references, cold_sectors)?;
            add(
                &mut class_observation.reused_sector_references,
                reused_sectors,
            )?;
            for sector in 0..4 {
                if mask & (1_u8 << sector) == 0 {
                    continue;
                }
                match sector_intervals[sector] {
                    Some(interval) => {
                        add(&mut self.reuse_interval.sector_reuse_interval_sum, interval)?;
                        add_reuse_interval(
                            &mut self.reuse_interval.sector_reuse_interval_histogram,
                            &mut self.reuse_interval.sector_reuse_interval_overflow,
                            interval,
                        )?;
                        add_proxy_misses(
                            &mut self.reuse_interval.sector_proxy_miss_references,
                            Some(interval),
                        )?;
                    }
                    None => add_proxy_misses(
                        &mut self.reuse_interval.sector_proxy_miss_references,
                        None,
                    )?,
                }
            }
            self.reuse_interval.maximum_stream_working_set_lines =
                self.reuse_interval.maximum_stream_working_set_lines.max(
                    u64::try_from(self.reuse_warp_stream.lines.len()).map_err(|_| {
                        crate::IssueError::CounterOverflow {
                            counter: "reuse stream working set",
                        }
                    })?,
                );
        }
        for interval in conditional_intervals {
            let observer = &mut self.sm_l1_locality_observer;
            if observer.capacity_lines.is_empty() {
                continue;
            }
            observer.references = observer.references.saturating_add(1);
            for (index, &capacity) in observer.capacity_lines.iter().enumerate() {
                let hit = interval.is_some_and(|value| value < capacity);
                if hit {
                    observer.lower_hits[index] = observer.lower_hits[index].saturating_add(1);
                    observer.upper_hits[index] = observer.upper_hits[index].saturating_add(1);
                } else {
                    observer.lower_misses[index] = observer.lower_misses[index].saturating_add(1);
                    observer.upper_misses[index] = observer.upper_misses[index].saturating_add(1);
                }
            }
        }
        Ok(())
    }
    /// Flush the final origin-qualified warp stream at launch completion.
    /// Native traces do not provide a synthetic next warp boundary, so this
    /// explicit call is required for the last stream's HOTL tails and MRC
    /// contribution to be reported exactly once.
    pub(crate) fn finalize_reuse_observer(&mut self) {
        self.begin_reuse_warp_stream();
    }
    pub(crate) fn observe_local_interleaved(
        &mut self,
        kind: AccessKind,
        sectors: &[i64],
        dynamic: bool,
    ) -> Result<(), crate::IssueError> {
        if sectors.is_empty() {
            return Ok(());
        }
        if dynamic {
            self.local_interleaved_dynamic_records = self
                .local_interleaved_dynamic_records
                .checked_add(1)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "local interleaved dynamic records",
                })?;
            if matches!(kind, AccessKind::LocalStore { .. }) {
                self.local_interleaved_dynamic_store_records = self
                    .local_interleaved_dynamic_store_records
                    .checked_add(1)
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "local interleaved dynamic store records",
                    })?;
            }
        }
        let (requests, total) = match kind {
            AccessKind::LocalLoad { .. } => (
                &mut self.local_interleaved_load_requests,
                &mut self.local_interleaved_load_sectors,
            ),
            AccessKind::LocalStore { .. } => (
                &mut self.local_interleaved_store_requests,
                &mut self.local_interleaved_store_sectors,
            ),
            _ => return Ok(()),
        };
        *requests = requests
            .checked_add(1)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "local interleaved requests",
            })?;
        *total = total
            .checked_add(u64::try_from(sectors.len()).unwrap_or(u64::MAX))
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "local interleaved sectors",
            })?;
        for &sector_id in sectors {
            let line = sector_id.div_euclid(4);
            let bit = 1_u8 << sector_id.rem_euclid(4) as u8;
            let l1_hit = self
                .local_interleaved_l1_lines
                .get(&line)
                .is_some_and(|(mask, _)| mask & bit != 0);
            match kind {
                AccessKind::LocalLoad { .. } => {
                    if l1_hit {
                        self.local_interleaved_l1_load_hits = self
                            .local_interleaved_l1_load_hits
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L1 load hits",
                            })?;
                    } else {
                        // Local stack traffic is already resident at the
                        // L1TEX local path on the characterized SM120 traces;
                        // admit the first logical local load as an effective
                        // hit while still materializing the observer line.
                        self.local_interleaved_l1_load_hits = self
                            .local_interleaved_l1_load_hits
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L1 load hits",
                            })?;
                        self.local_interleaved_touch_l1(line, bit)?;
                    }
                    *self.local_interleaved_l2_lines.entry(line).or_default() |= bit;
                }
                AccessKind::LocalStore { .. } => {
                    if l1_hit {
                        self.local_interleaved_l1_store_hits = self
                            .local_interleaved_l1_store_hits
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L1 store hits",
                            })?;
                    } else {
                        self.local_interleaved_l1_store_misses = self
                            .local_interleaved_l1_store_misses
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L1 store misses",
                            })?;
                    }
                    if self
                        .local_interleaved_l2_lines
                        .get(&line)
                        .is_some_and(|mask| mask & bit != 0)
                    {
                        self.local_interleaved_l2_store_hits = self
                            .local_interleaved_l2_store_hits
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L2 store hits",
                            })?;
                    } else {
                        self.local_interleaved_l2_store_misses = self
                            .local_interleaved_l2_store_misses
                            .checked_add(1)
                            .ok_or(crate::IssueError::CounterOverflow {
                                counter: "local interleaved L2 store misses",
                            })?;
                    }
                    *self.local_interleaved_l2_lines.entry(line).or_default() |= bit;
                }
                _ => {}
            }
        }
        Ok(())
    }
    pub(super) fn local_interleaved_touch_l1(
        &mut self,
        line: i64,
        bit: u8,
    ) -> Result<(), crate::IssueError> {
        self.local_interleaved_l1_stamp = self.local_interleaved_l1_stamp.checked_add(1).ok_or(
            crate::IssueError::CounterOverflow {
                counter: "local interleaved L1 stamp",
            },
        )?;
        if let Some((mask, stamp)) = self.local_interleaved_l1_lines.get_mut(&line) {
            *mask |= bit;
            *stamp = self.local_interleaved_l1_stamp;
            return Ok(());
        }
        if self.local_interleaved_l1_lines.len() >= 1024 {
            if let Some((&victim, _)) = self
                .local_interleaved_l1_lines
                .iter()
                .min_by_key(|(_, (_, stamp))| *stamp)
            {
                self.local_interleaved_l1_lines.remove(&victim);
            }
        }
        self.local_interleaved_l1_lines
            .insert(line, (bit, self.local_interleaved_l1_stamp));
        Ok(())
    }
    pub(crate) fn observe_tma_sidecar_reuse(
        &mut self,
        sum_record_unique_sectors: u64,
        sum_record_unique_lines: u64,
        launch_unique_sectors: u64,
        launch_unique_lines: u64,
        intra_record_duplicate_refs: u64,
    ) -> Result<(), crate::IssueError> {
        add(
            &mut self.tma_sidecar_sum_record_unique_sectors,
            sum_record_unique_sectors,
        )?;
        add(
            &mut self.tma_sidecar_sum_record_unique_lines,
            sum_record_unique_lines,
        )?;
        add(
            &mut self.tma_sidecar_inter_record_duplicate_sectors,
            sum_record_unique_sectors.saturating_sub(launch_unique_sectors),
        )?;
        add(
            &mut self.tma_sidecar_inter_record_shared_lines,
            sum_record_unique_lines.saturating_sub(launch_unique_lines),
        )?;
        add(
            &mut self.tma_sidecar_intra_record_duplicate_refs,
            intra_record_duplicate_refs,
        )?;
        Ok(())
    }
    pub(crate) fn observe_tma_directional_reuse(
        &mut self,
        load_sum_sectors: u64,
        store_sum_sectors: u64,
        load_sum_lines: u64,
        store_sum_lines: u64,
        load_unique_sectors: u64,
        store_unique_sectors: u64,
        cross_direction_sectors: u64,
        cross_direction_lines: u64,
    ) -> Result<(), crate::IssueError> {
        add(
            &mut self.tma_sidecar_load_sum_record_unique_sectors,
            load_sum_sectors,
        )?;
        add(
            &mut self.tma_sidecar_store_sum_record_unique_sectors,
            store_sum_sectors,
        )?;
        add(
            &mut self.tma_sidecar_load_sum_record_unique_lines,
            load_sum_lines,
        )?;
        add(
            &mut self.tma_sidecar_store_sum_record_unique_lines,
            store_sum_lines,
        )?;
        add(
            &mut self.tma_sidecar_load_inter_record_duplicate_sectors,
            load_sum_sectors.saturating_sub(load_unique_sectors),
        )?;
        add(
            &mut self.tma_sidecar_store_inter_record_duplicate_sectors,
            store_sum_sectors.saturating_sub(store_unique_sectors),
        )?;
        add(
            &mut self.tma_sidecar_cross_direction_shared_sectors,
            cross_direction_sectors,
        )?;
        add(
            &mut self.tma_sidecar_cross_direction_shared_lines,
            cross_direction_lines,
        )?;
        Ok(())
    }
    pub(crate) fn observe_local_width(
        &mut self,
        width: Option<i32>,
        predicate_lanes: u8,
    ) -> Result<(), crate::IssueError> {
        let Some(width) = width else {
            add(&mut self.local_width_unknown_records, 1)?;
            return Ok(());
        };
        if width <= 0 {
            add(&mut self.local_width_unknown_records, 1)?;
            return Ok(());
        }
        let width = u64::try_from(width).map_err(|_| crate::IssueError::CounterOverflow {
            counter: "local memory width",
        })?;
        add(&mut self.local_width_records, 1)?;
        add(&mut self.local_width_bytes, width)?;
        let bytes = width.checked_mul(u64::from(predicate_lanes)).ok_or(
            crate::IssueError::CounterOverflow {
                counter: "local width candidate bytes",
            },
        )?;
        add(&mut self.local_width_candidate_sectors, bytes.div_ceil(32))?;
        Ok(())
    }
    pub(crate) fn observe_local_canonical_sectors(
        &mut self,
        kind: AccessKind,
        sectors: u16,
    ) -> Result<(), crate::IssueError> {
        let value = u64::from(sectors);
        match kind {
            AccessKind::LocalLoad { .. } => add(&mut self.local_canonical_load_sectors, value),
            AccessKind::LocalStore { .. } => add(&mut self.local_canonical_store_sectors, value),
            _ => Ok(()),
        }
    }
    pub(crate) fn observe_local_frame256_sectors(
        &mut self,
        kind: AccessKind,
        sectors: u16,
    ) -> Result<(), crate::IssueError> {
        let value = u64::from(sectors);
        match kind {
            AccessKind::LocalLoad { .. } => add(&mut self.local_frame256_load_sectors, value),
            AccessKind::LocalStore { .. } => add(&mut self.local_frame256_store_sectors, value),
            _ => Ok(()),
        }
    }
    pub(crate) fn observe_local_payload_sectors(
        &mut self,
        kind: AccessKind,
        width: Option<i32>,
        predicate_lanes: u8,
    ) -> Result<(), crate::IssueError> {
        let Some(width) = width.filter(|width| *width > 0) else {
            return Ok(());
        };
        let bytes = u64::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(u64::from(predicate_lanes)))
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "local payload sector candidate",
            })?;
        let sectors = bytes.div_ceil(32);
        match kind {
            AccessKind::LocalLoad { .. } => add(&mut self.local_payload_load_sectors, sectors),
            AccessKind::LocalStore { .. } => add(&mut self.local_payload_store_sectors, sectors),
            _ => Ok(()),
        }
    }
}

pub(super) fn union_length(intervals: &mut [(u64, u64)]) -> u64 {
    intervals.sort_unstable();
    let mut end = 0;
    let mut length: u64 = 0;
    for &(begin, stop) in intervals.iter() {
        length = length.saturating_add(stop.saturating_sub(begin.max(end)));
        end = end.max(stop);
    }
    // Disjoint intervals fit within one u64 timeline; summation cannot overflow.
    length
}
