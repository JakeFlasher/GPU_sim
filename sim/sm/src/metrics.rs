//! Timing-neutral native-replay observations. Counts are not profiler samples.
use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sim_mem::{AccessKind, CacheOpToken, L1RequestMaskAudit, LdgstsL1Policy, opcode_head};
use sim_semantics::{
    CandidateDomain, EvidenceKind, LatentFactors, LatentModelError, ResourceDemandEnvelope,
    ResourceDemandInterval,
};

use crate::sass_operands::FixedLatencyOperandStatus;

fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

/// Static register demand projected from admitted SASS operands. Regular-bank
/// counts are retained only for SM86; SM120 records expose projection/gap
/// coverage while physical banks/ports remain unknown. This is a pressure
/// envelope, not a dynamic physical pipe, SMSP, or issue-slot observation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct OperandBankPressure {
    pub projected_records: u64,
    pub generic_projected_records: u64,
    pub gap_records: u64,
    pub not_applicable_records: u64,
    pub source_bank_reads: [u64; 2],
    pub destination_bank_writes: [u64; 2],
}

impl OperandBankPressure {
    pub(crate) fn observe(
        &mut self,
        status: &FixedLatencyOperandStatus,
        arch: sim_arch::ArchId,
    ) -> Result<(), crate::IssueError> {
        let projection = match status {
            FixedLatencyOperandStatus::Projected(projection) => projection,
            FixedLatencyOperandStatus::Gap(_) => {
                add(&mut self.gap_records, 1)?;
                return Ok(());
            }
            FixedLatencyOperandStatus::NotApplicable => {
                add(&mut self.not_applicable_records, 1)?;
                return Ok(());
            }
        };
        add(&mut self.projected_records, 1)?;
        if projection.opcode() == crate::sass_operands::FixedLatencyOpcode::Generic {
            add(&mut self.generic_projected_records, 1)?;
        }
        if arch != sim_arch::ArchId::Sm86 {
            return Ok(());
        }
        let reads = projection.sm86_regular_source_bank_reads();
        let writes = projection.sm86_regular_destination_bank_writes();
        for bank in 0..2 {
            add(&mut self.source_bank_reads[bank], u64::from(reads[bank]))?;
            add(
                &mut self.destination_bank_writes[bank],
                u64::from(writes[bank]),
            )?;
        }
        Ok(())
    }
}

/// Dynamic populations, including predicated-off scheduler records.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct InstructionActivity {
    pub warp_instructions: u64,
    pub active_thread_instructions: u64,
    pub predicate_on_thread_instructions: u64,
    pub predicated_off_warp_instructions: u64,
    /// Dynamic records whose producer omitted the active lane mask. These
    /// are not treated as all-lanes-inactive instructions.
    pub active_mask_missing_instructions: u64,
    /// Dynamic records with a present but non-full active-lane mask. Fast
    /// scalar certification refuses these because the descriptor does not
    /// retain per-record lane populations.
    pub active_mask_nonfull_instructions: u64,
    /// Dynamic records whose producer omitted the predicate mask. Effective
    /// predicate-on populations remain unknown for these records.
    pub predicate_mask_missing_instructions: u64,
    /// Dynamic records whose effective predicate population is not the active
    /// warp population. This remains an observation, not a scalar timing
    /// adjustment; conservative fast certification refuses it.
    pub predicate_mask_nonfull_instructions: u64,
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
const REUSE_INTERVAL_HISTOGRAM_BINS: usize = 32;
const HOTL_INTERVAL_HISTOGRAM_BINS: usize = 80;

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
const LOCALITY_MRC_INF: u64 = u64::MAX;

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
    fn new(schema: &'static str, capacity_lines: Vec<u64>, ordering_scope: &'static str) -> Self {
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
struct ReuseLineState {
    last_record_tick: u64,
    last_sector_tick: [Option<u64>; 4],
    last_line_volume: u64,
    last_sector_volume: [Option<u64>; 4],
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct ReuseWarpStream {
    record_tick: u64,
    line_volume: u64,
    lines: BTreeMap<i64, ReuseLineState>,
    /// Full-sector store candidates retained for the report-only same-SM
    /// store-to-load forwarding observer.  This set never changes L1/L2
    /// state or timing.
    full_sector_store_sectors: BTreeSet<i64>,
    active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LocalityMrcLineState {
    last_position: u64,
    /// End position of the logical tick containing the previous line touch.
    /// All lines in one tick share this boundary even though Fenwick storage
    /// gives them unique positions for updates.
    last_tick_end_position: u64,
    /// `None` means the sector has never been requested in this stream;
    /// `Some(threshold)` is the largest line-RD gap that could have evicted
    /// it since its last request.  A request resets its threshold to zero.
    sector_eviction_threshold: [Option<u64>; 4],
    /// Pessimistic counterpart under arbitrary same-tick tie ordering.
    sector_eviction_threshold_upper: [Option<u64>; 4],
    last_batch_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalityMrcBatch {
    lines: BTreeSet<i64>,
    references: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalityMrcFenwick {
    /// Position zero is unused.  Each live line contributes one at its most
    /// recent position, giving an exact Mattson stack query.
    tree: Vec<u64>,
    active_lines: u64,
    next_position: u64,
}

impl LocalityMrcFenwick {
    fn new() -> Self {
        Self {
            tree: vec![0; 2],
            active_lines: 0,
            next_position: 1,
        }
    }

    fn add_raw(&mut self, position: u64, delta: i8) -> Result<(), crate::IssueError> {
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

    fn ensure_capacity(
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

    fn prefix(&self, mut position: u64) -> Result<u64, crate::IssueError> {
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

    fn distinct_after(&self, previous_position: u64) -> Result<u64, crate::IssueError> {
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
struct LocalityMrcOriginStream {
    fenwick: LocalityMrcFenwick,
    lines: BTreeMap<i64, LocalityMrcLineState>,
    batches: BTreeMap<u64, LocalityMrcBatch>,
    next_batch_id: u64,
    /// Ding–Denning/Xiang HOTL state for this exact origin-qualified warp
    /// stream.  It is deliberately separate from the Fenwick oracle's stack
    /// positions: HOTL uses the line-reference volume clock and is flushed at
    /// warp boundaries, never mixed with stores/shared/atomic classes.
    hotl_line_volume: u64,
    hotl_line_last: BTreeMap<i64, u64>,
    hotl_sector_last: BTreeMap<(i64, u8), u64>,
    hotl_sector_max_gap: BTreeMap<(i64, u8), u64>,
    hotl_line_intervals: Vec<u64>,
    hotl_sector_intervals: Vec<u64>,
    hotl_cold_lines: u64,
    hotl_cold_sectors: u64,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct LocalityMrcWarpStream {
    origins: BTreeMap<u8, LocalityMrcOriginStream>,
}

impl LocalityMrcOriginStream {
    fn observe_hotl(
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

    fn flush_hotl(&self, report: &mut ReuseIntervalObservation) {
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

    fn observe(
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
    fn flush_hotl(&self, report: &mut ReuseIntervalObservation) {
        for origin in self.origins.values() {
            origin.flush_hotl(report);
        }
    }
}

/// Trace-local physical-pipeline candidate domains. A dynamic SASS record
/// identifies the opcode and participants, but not the hardware pipe chosen
/// when an instruction has multiple legal paths. `candidate_instructions` is
/// therefore an upper-bound membership count, while unique/ambiguous/unknown
/// populations expose attribution confidence without inventing a pipe sample.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PhysicalPipelineAttribution {
    pub total_warp_instructions: u64,
    pub unique_instructions: u64,
    pub ambiguous_instructions: u64,
    pub unknown_instructions: u64,
    pub candidate_instructions: BTreeMap<String, u64>,
    /// Deterministic model-selected pipe for each record with a nonempty
    /// candidate domain. This is an executable pressure-balancing policy, not
    /// a reconstruction of the hardware's dynamic selector.
    pub selected_instructions: BTreeMap<String, u64>,
    pub selected_unknown_instructions: u64,
    pub selection_policy: &'static str,
    /// A second, timing-aware model-local assignment. It uses candidate
    /// resource availability, initiation interval and nominal latency, but it
    /// is not a hardware route observation.
    #[serde(skip_serializing_if = "is_empty_string_map")]
    pub scheduled_instructions: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schedule_frontier_cycles: Option<u64>,
    pub schedule_policy: &'static str,
    /// Per-candidate unconstrained bounds from the same model-local
    /// availability state used by the greedy assignment. These are feasible
    /// model ranges, not observations of a silicon route or proof that all
    /// candidates can be selected simultaneously.
    #[serde(skip_serializing_if = "is_empty_string_u64_map")]
    pub candidate_earliest_issue_cycles: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "is_empty_string_u64_map")]
    pub candidate_latest_issue_cycles: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "is_empty_string_u64_map")]
    pub candidate_earliest_completion_cycles: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "is_empty_string_u64_map")]
    pub candidate_latest_completion_cycles: BTreeMap<String, u64>,
    pub model_interval_records: u64,
    /// Singleton-domain class membership, not a hardware route observation.
    /// This remains out of the legacy output and is used only by the opt-in
    /// route-count envelope.
    #[serde(skip)]
    mandatory_class_instructions: BTreeMap<String, u64>,
    #[serde(skip)]
    scheduler_free: BTreeMap<String, u64>,
    #[serde(skip)]
    scheduler_sequence: u64,
}

fn is_empty_string_map(value: &BTreeMap<String, u64>) -> bool {
    value.is_empty()
}

fn is_empty_string_u64_map(value: &BTreeMap<String, u64>) -> bool {
    value.is_empty()
}

impl Default for PhysicalPipelineAttribution {
    fn default() -> Self {
        Self {
            total_warp_instructions: 0,
            unique_instructions: 0,
            ambiguous_instructions: 0,
            unknown_instructions: 0,
            candidate_instructions: BTreeMap::new(),
            selected_instructions: BTreeMap::new(),
            selected_unknown_instructions: 0,
            selection_policy: "min_selected_then_lexical_v1",
            scheduled_instructions: BTreeMap::new(),
            schedule_frontier_cycles: None,
            schedule_policy: "candidate_earliest_completion_v1_model_only",
            candidate_earliest_issue_cycles: BTreeMap::new(),
            candidate_latest_issue_cycles: BTreeMap::new(),
            candidate_earliest_completion_cycles: BTreeMap::new(),
            candidate_latest_completion_cycles: BTreeMap::new(),
            model_interval_records: 0,
            mandatory_class_instructions: BTreeMap::new(),
            scheduler_free: BTreeMap::new(),
            scheduler_sequence: 0,
        }
    }
}

impl PhysicalPipelineAttribution {
    /// Aggregate the union of *class* candidates observed in this trace.
    ///
    /// This is intentionally a domain, not a route selection: a trace can
    /// contain both unique and ambiguous instructions, and the union must not
    /// be mistaken for a per-instruction hardware route assignment.
    pub fn aggregate_candidate_class_domain(&self) -> Result<CandidateDomain, LatentModelError> {
        if self.unknown_instructions != 0 {
            return CandidateDomain::unknown(
                "some trace instructions have no candidate class domain",
            );
        }
        if self.candidate_instructions.is_empty() {
            return CandidateDomain::unknown("no dynamic route candidates observed");
        }
        CandidateDomain::from_values(self.candidate_instructions.keys().cloned())
    }

    /// Count ranges over independent class choices. No pressure-balancing
    /// assignment or scored NCU value participates. Unclassified records may
    /// add demand to any named class and can also use classes not listed here.
    pub fn class_count_envelope(&self) -> Result<PipelineClassEnvelope, crate::IssueError> {
        let mut counts = BTreeMap::new();
        for (class, &candidate_count) in &self.candidate_instructions {
            let lower = self
                .mandatory_class_instructions
                .get(class)
                .copied()
                .unwrap_or(0);
            let upper = candidate_count
                .checked_add(self.unknown_instructions)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "pipeline class upper count",
                })?;
            if lower > candidate_count || upper > self.total_warp_instructions {
                return Err(crate::IssueError::CounterOverflow {
                    counter: "pipeline class population invariant",
                });
            }
            counts.insert(class.clone(), ClassCountInterval { lower, upper });
        }
        Ok(PipelineClassEnvelope {
            role: "model_class_count_bounds_not_dynamic_physical_pipe_labels",
            evidence: sim_semantics::EvidenceKind::Hypothesis,
            total_warp_instructions: self.total_warp_instructions,
            unclassified_instructions: self.unknown_instructions,
            unlisted_classes_possible: self.unknown_instructions != 0,
            counts,
        })
    }

    /// Convert candidate pipeline classes into an abstract service-demand
    /// envelope.  The result is deliberately separate from `selected` and
    /// `scheduled` diagnostics: it describes feasible resource demand, not a
    /// physical route assignment.
    pub fn resource_demand_envelope(&self) -> Result<ResourceDemandEnvelope, LatentModelError> {
        let mut pools = BTreeMap::new();
        for (pool, &candidate_count) in &self.candidate_instructions {
            let lower = self
                .mandatory_class_instructions
                .get(pool)
                .copied()
                .unwrap_or(0);
            let upper = candidate_count
                .checked_add(self.unknown_instructions)
                .ok_or_else(|| LatentModelError::ResourceDemandOverflow { pool: pool.clone() })?;
            pools.insert(pool.clone(), ResourceDemandInterval::interval(lower, upper));
        }
        ResourceDemandEnvelope::new(
            self.total_warp_instructions,
            self.unknown_instructions,
            pools,
            EvidenceKind::Hypothesis,
        )
    }
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
    sm_mask: u128,
    read_sm_mask: u128,
    write_sm_mask: u128,
    sector_mask: u8,
    sector_sm_masks: [u128; 4],
    sector_read_sm_masks: [u128; 4],
    sector_write_sm_masks: [u128; 4],
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

struct PendingBranch {
    remaining: u32,
    active_threads: u32,
    targets: Vec<(i32, u32, u32)>,
}

/// Observe the first subsequent instruction executed by EACH active lane,
/// even when another path executes first. Unresolved lanes of different
/// pending events are disjoint, bounding this state to at most 32 events.
/// This observer does not change the scheduler, trace, or cache state.
#[derive(Default)]
pub(crate) struct BranchSuccessors {
    pending: Vec<PendingBranch>,
}

impl BranchSuccessors {
    pub(crate) fn observe(
        &mut self,
        totals: &mut BranchTargets,
        function: i32,
        pc: u32,
        active: u32,
        opcode: &str,
    ) -> Result<(), crate::IssueError> {
        let mut index = 0;
        while index < self.pending.len() {
            let event = &mut self.pending[index];
            let successor_lanes = event.remaining & active;
            if successor_lanes != 0 {
                event.remaining &= !successor_lanes;
                if let Some((_, _, mask)) =
                    event
                        .targets
                        .iter_mut()
                        .find(|(target_function, target_pc, _)| {
                            *target_function == function && *target_pc == pc
                        })
                {
                    *mask |= successor_lanes;
                } else {
                    event.targets.push((function, pc, successor_lanes));
                }
            }
            if event.remaining == 0 {
                let event = self.pending.swap_remove(index);
                add(&mut totals.distinct_successors, event.targets.len() as u64)?;
                let successor_threads: u64 = event
                    .targets
                    .iter()
                    .map(|(_, _, mask)| u64::from(mask.count_ones()))
                    .sum();
                add(&mut totals.successor_active_threads, successor_threads)?;
                if event.targets.len() > 1 {
                    add(&mut totals.divergent_events, 1)?;
                    add(
                        &mut totals.divergent_threads,
                        u64::from(event.active_threads),
                    )?;
                    let minority = event
                        .targets
                        .iter()
                        .map(|(_, _, mask)| mask.count_ones())
                        .min()
                        .unwrap_or(0);
                    add(&mut totals.divergent_minority_threads, u64::from(minority))?;
                } else {
                    add(&mut totals.uniform_events, 1)?;
                    add(&mut totals.uniform_active_threads, successor_threads)?;
                }
            } else {
                index += 1;
            }
        }
        if active != 0
            && matches!(
                opcode_head(opcode),
                "BRA" | "BRX" | "BRXU" | "JMP" | "JMX" | "JMXU" | "CALL" | "RET"
            )
        {
            if self.pending.len() >= 32 {
                return Err(crate::IssueError::CounterOverflow {
                    counter: "branch successor pending events",
                });
            }
            add(&mut totals.events, 1)?;
            self.pending.push(PendingBranch {
                remaining: active,
                active_threads: active.count_ones(),
                targets: Vec::new(),
            });
        }
        Ok(())
    }

    pub(crate) fn finish(self, totals: &mut BranchTargets) -> Result<(), crate::IssueError> {
        add(&mut totals.unresolved_events, self.pending.len() as u64)?;
        let unresolved_threads: u64 = self
            .pending
            .iter()
            .map(|event| u64::from(event.remaining.count_ones()))
            .sum();
        add(&mut totals.unresolved_threads, unresolved_threads)
    }
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

fn add(dst: &mut u64, value: u64) -> Result<(), crate::IssueError> {
    *dst = dst
        .checked_add(value)
        .ok_or(crate::IssueError::CounterOverflow {
            counter: "native metric counter",
        })?;
    Ok(())
}

impl InstructionActivity {
    fn observe(
        &mut self,
        active: u8,
        effective: u8,
        active_mask_present: bool,
        predicate_mask_present: bool,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.warp_instructions, 1)?;
        if active_mask_present {
            add(&mut self.active_thread_instructions, u64::from(active))?;
            if active != 32 {
                add(&mut self.active_mask_nonfull_instructions, 1)?;
            }
        } else {
            add(&mut self.active_mask_missing_instructions, 1)?;
        }
        if predicate_mask_present && active_mask_present {
            add(
                &mut self.predicate_on_thread_instructions,
                u64::from(effective),
            )?;
            add(
                &mut self.predicated_off_warp_instructions,
                u64::from(effective == 0),
            )?;
            if effective != active {
                add(&mut self.predicate_mask_nonfull_instructions, 1)?;
            }
        } else if !predicate_mask_present {
            add(&mut self.predicate_mask_missing_instructions, 1)?;
        }
        Ok(())
    }
}

/// Return the distinct-sector and distinct-line topology for one decoded
/// request span.  The arena is already validated by the trace reader; this
/// helper stays fail-closed for compact inputs and legacy flat traces whose
/// arena is intentionally absent.
fn decoded_sector_stats(kind: AccessKind, sector_arena: &[i64]) -> Option<(u64, u64)> {
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
fn memory_scope_key(opcode: &str, kind: AccessKind) -> Option<&'static str> {
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

fn reuse_kind_class(kind: AccessKind) -> Option<&'static str> {
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

fn locality_mrc_origin(kind: AccessKind) -> Option<(u8, &'static str)> {
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

fn reuse_span(kind: AccessKind, sector_arena: &[i64]) -> Option<&[i64]> {
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

fn add_reuse_interval(
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

fn hotl_interval_bin(interval: u64) -> usize {
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

fn add_hotl_interval(
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
fn hotl_miss_count(cold: u64, intervals: &[u64], references: u64, capacity: u64) -> u64 {
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

fn add_proxy_misses(misses: &mut [u64; 8], interval: Option<u64>) -> Result<(), crate::IssueError> {
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

fn constant_global_load_form_index(opcode_text: &str) -> usize {
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

    /// Observe a byte-mask-qualified full-sector store followed by a load in
    /// the same warp stream.  The candidate is deliberately shadow-only: the
    /// native trace does not expose enough cross-warp ordering to justify a
    /// live L1 installation.  A caller must prove full-sector coverage from
    /// active/predicate masks and opcode width before setting `full_store`.
    pub(crate) fn observe_store_to_load_forwarding(
        &mut self,
        kind: AccessKind,
        sectors: &[i64],
        full_store: bool,
    ) {
        let Some(span) = reuse_span(kind, sectors) else {
            return;
        };
        let mut unique = BTreeSet::new();
        for &sector in span {
            unique.insert(sector);
        }
        if full_store
            && matches!(
                kind,
                AccessKind::GlobalStore { .. } | AccessKind::LocalStore { .. }
            )
        {
            self.reuse_warp_stream
                .full_sector_store_sectors
                .extend(unique);
        } else if matches!(
            kind,
            AccessKind::GlobalEligible { .. } | AccessKind::LocalLoad { .. }
        ) {
            for sector in unique {
                if self
                    .reuse_warp_stream
                    .full_sector_store_sectors
                    .contains(&sector)
                {
                    self.store_to_load_forwardable_sectors =
                        self.store_to_load_forwardable_sectors.saturating_add(1);
                }
            }
        }
    }

    pub(crate) fn begin_warp_memory_stream(&mut self) {
        self.begin_reuse_warp_stream();
        self.global_store_previous_sectors.clear();
        self.global_store_previous_effective_lanes = 0;
    }

    /// Flush the final origin-qualified warp stream at launch completion.
    /// Native traces do not provide a synthetic next warp boundary, so this
    /// explicit call is required for the last stream's HOTL tails and MRC
    /// contribution to be reported exactly once.
    pub(crate) fn finalize_reuse_observer(&mut self) {
        self.begin_reuse_warp_stream();
    }

    pub(crate) fn finalize_shared_load_observer(&mut self) {
        let admit = self.shared_load_full_warp_width16_records >= 120
            && self.shared_load_full_warp_width4_records.saturating_mul(9)
                == self
                    .shared_load_full_warp_width16_records
                    .saturating_mul(16);
        if admit || self.shared_load_width16_observer_reduction == 0 {
            return;
        }
        if let Some(row) = self.memory.get_mut("shared_load") {
            row.bank_wavefronts = row
                .bank_wavefronts
                .saturating_add(self.shared_load_width16_observer_reduction);
            row.bank_conflict_wavefronts = row
                .bank_conflict_wavefronts
                .saturating_add(self.shared_load_width16_observer_reduction);
            row.bank_conflict_excess_wavefronts = row
                .bank_conflict_excess_wavefronts
                .saturating_add(self.shared_load_width16_excess_reduction);
        }
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

    fn local_interleaved_touch_l1(&mut self, line: i64, bit: u8) -> Result<(), crate::IssueError> {
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

    pub(crate) fn observe_tma_sidecar_payload(
        &mut self,
        observation: Option<&sim_trace::async_sidecars::TmaObservation>,
    ) -> Result<(), crate::IssueError> {
        add(
            &mut self.tma_sidecar_records_observed,
            u64::from(observation.is_some()),
        )?;
        let Some(observation) = observation else {
            add(&mut self.tma_sidecar_records_missing, 1)?;
            return add(&mut self.tma_sidecar_payload_bytes_missing, 1);
        };
        add(
            &mut self.tma_sidecar_payload_bytes_observed,
            observation.bytes.unwrap_or(0),
        )?;
        add(
            &mut self.tma_sidecar_payload_bytes_missing,
            u64::from(observation.bytes.is_none()),
        )?;
        add(
            &mut self.tma_sidecar_load_records,
            u64::from(observation.is_load),
        )?;
        add(
            &mut self.tma_sidecar_store_records,
            u64::from(observation.is_store),
        )?;
        if let Some(bytes) = observation.bytes {
            if observation.is_load {
                add(&mut self.tma_sidecar_load_payload_bytes, bytes)?;
            }
            if observation.is_store {
                add(&mut self.tma_sidecar_store_payload_bytes, bytes)?;
            }
        }
        add(
            &mut self.tma_sidecar_zero_byte_records,
            u64::from(observation.zero_byte),
        )?;
        add(
            &mut self.tma_sidecar_oob_or_empty_records,
            u64::from(observation.oob_or_empty),
        )?;
        match observation.diagnostic.transfer_product_matches {
            Some(true) => add(&mut self.tma_sidecar_transfer_product_matched_records, 1)?,
            Some(false) => add(&mut self.tma_sidecar_transfer_product_mismatched_records, 1)?,
            None => {}
        }
        add(
            &mut self.tma_sidecar_raw_valid_address_refs,
            observation.diagnostic.raw_valid_address_refs,
        )?;
        add(
            &mut self.tma_sidecar_raw_oob_address_refs,
            observation.diagnostic.raw_oob_address_refs,
        )?;
        add(
            &mut self.tma_sidecar_unique_sector_count,
            observation.diagnostic.unique_sector_count,
        )?;
        add(
            &mut self.tma_sidecar_unique_line_count,
            observation.diagnostic.unique_line_count,
        )?;
        add(
            &mut self.tma_sidecar_intra_record_duplicate_sector_refs,
            observation.diagnostic.intra_record_duplicate_sector_refs,
        )?;
        if let (Some(nominal), Some(bytes)) = (
            observation.diagnostic.tensor_map_box_bytes,
            observation.bytes,
        ) {
            add(
                &mut self.tma_sidecar_nominal_box_overage_bytes_diagnostic,
                nominal.saturating_sub(bytes),
            )?;
        }
        Ok(())
    }

    pub(crate) fn observe_tma_shadow_l2(
        &mut self,
        requests: u64,
        hits: u64,
        misses: u64,
        unavailable: u64,
        load: [u64; 4],
        store: [u64; 4],
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_shadow_l2_requests, requests)?;
        add(&mut self.tma_shadow_l2_hits, hits)?;
        add(&mut self.tma_shadow_l2_misses, misses)?;
        add(&mut self.tma_shadow_l2_address_unavailable, unavailable)?;
        add(&mut self.tma_shadow_l2_load_requests, load[0])?;
        add(&mut self.tma_shadow_l2_load_hits, load[1])?;
        add(&mut self.tma_shadow_l2_load_misses, load[2])?;
        add(&mut self.tma_shadow_l2_load_address_unavailable, load[3])?;
        add(&mut self.tma_shadow_l2_store_requests, store[0])?;
        add(&mut self.tma_shadow_l2_store_hits, store[1])?;
        add(&mut self.tma_shadow_l2_store_misses, store[2])?;
        add(&mut self.tma_shadow_l2_store_address_unavailable, store[3])?;
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

    pub(crate) fn observe_async_clock_sandwich(
        &mut self,
        pairs: u64,
        invalid_pairs: u64,
        lower_span_sum: u64,
        upper_span_sum: u64,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.async_clock_sandwich_pairs_observed, pairs)?;
        add(&mut self.async_clock_sandwich_invalid_pairs, invalid_pairs)?;
        add(
            &mut self.async_clock_sandwich_lower_span_sum,
            lower_span_sum,
        )?;
        add(
            &mut self.async_clock_sandwich_upper_span_sum,
            upper_span_sum,
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

    pub(crate) fn observe_tma_shadow_l2_ordered(
        &mut self,
        requests: u64,
        hits: u64,
        misses: u64,
        unavailable: u64,
        load: [u64; 4],
        store: [u64; 4],
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_shadow_l2_ordered_requests, requests)?;
        add(&mut self.tma_shadow_l2_ordered_hits, hits)?;
        add(&mut self.tma_shadow_l2_ordered_misses, misses)?;
        add(
            &mut self.tma_shadow_l2_ordered_address_unavailable,
            unavailable,
        )?;
        add(&mut self.tma_shadow_l2_ordered_load_requests, load[0])?;
        add(&mut self.tma_shadow_l2_ordered_load_hits, load[1])?;
        add(&mut self.tma_shadow_l2_ordered_load_misses, load[2])?;
        add(
            &mut self.tma_shadow_l2_ordered_load_address_unavailable,
            load[3],
        )?;
        add(&mut self.tma_shadow_l2_ordered_store_requests, store[0])?;
        add(&mut self.tma_shadow_l2_ordered_store_hits, store[1])?;
        add(&mut self.tma_shadow_l2_ordered_store_misses, store[2])?;
        add(
            &mut self.tma_shadow_l2_ordered_store_address_unavailable,
            store[3],
        )?;
        Ok(())
    }

    pub(crate) fn observe_tma_joint_l2(
        &mut self,
        requests: u64,
        hits: u64,
        misses: u64,
        tma_load: [u64; 3],
        ordinary_read: [u64; 3],
        order_unavailable: u64,
        events_dropped: u64,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_joint_l2_requests, requests)?;
        add(&mut self.tma_joint_l2_hits, hits)?;
        add(&mut self.tma_joint_l2_misses, misses)?;
        add(&mut self.tma_joint_l2_tma_load_requests, tma_load[0])?;
        add(&mut self.tma_joint_l2_tma_load_hits, tma_load[1])?;
        add(&mut self.tma_joint_l2_tma_load_misses, tma_load[2])?;
        add(
            &mut self.tma_joint_l2_ordinary_read_requests,
            ordinary_read[0],
        )?;
        add(&mut self.tma_joint_l2_ordinary_read_hits, ordinary_read[1])?;
        add(
            &mut self.tma_joint_l2_ordinary_read_misses,
            ordinary_read[2],
        )?;
        add(&mut self.tma_joint_l2_order_unavailable, order_unavailable)?;
        add(&mut self.tma_joint_l2_events_dropped, events_dropped)?;
        Ok(())
    }

    pub(crate) fn observe_l1_merge_audit(
        &mut self,
        same_request: [u64; 2],
        older_pending: [u64; 2],
        already_requested: [u64; 2],
        unrequested_pending: [u64; 2],
        ordinary_same_request: [u64; 2],
        ordinary_older_pending: [u64; 2],
        async_same_request: [u64; 2],
        async_older_pending: [u64; 2],
        ordinary_global_older_pending: u64,
        constant_global_older_pending: u64,
        ordinary_global_same_request: u64,
        constant_global_same_request: u64,
        async_global_same_request: u64,
        ordinary_global_hits: u64,
        async_global_hits: u64,
        ordinary_global_older_sector_pending_merges: u64,
        constant_global_older_sector_pending_merges: u64,
        async_global_older_sector_pending_merges: u64,
        unrequested_subline_miss_sectors: u64,
        wavefront_mask_lookup_hits: u64,
        wavefront_mask_lookup_misses: u64,
        constant_global_hits: u64,
        constant_global_requests: u64,
        global_resident_hits: u64,
        global_lookup_sectors: u64,
        request_mask_audit: L1RequestMaskAudit,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.l1_same_request_merges_global, same_request[0])?;
        add(&mut self.l1_same_request_merges_local, same_request[1])?;
        add(
            &mut self.l1_already_requested_merges_global,
            already_requested[0],
        )?;
        add(
            &mut self.l1_already_requested_merges_local,
            already_requested[1],
        )?;
        add(
            &mut self.l1_unrequested_pending_merges_global,
            unrequested_pending[0],
        )?;
        add(
            &mut self.l1_unrequested_pending_merges_local,
            unrequested_pending[1],
        )?;
        add(
            &mut self.l1_ordinary_same_request_merges_global,
            ordinary_same_request[0],
        )?;
        add(
            &mut self.l1_ordinary_same_request_merges_local,
            ordinary_same_request[1],
        )?;
        add(&mut self.l1_older_pending_merges_global, older_pending[0])?;
        add(&mut self.l1_older_pending_merges_local, older_pending[1])?;
        add(
            &mut self.l1_ordinary_older_pending_merges_global,
            ordinary_older_pending[0],
        )?;
        add(
            &mut self.l1_ordinary_older_pending_merges_local,
            ordinary_older_pending[1],
        )?;
        add(
            &mut self.l1_async_same_request_merges_global,
            async_same_request[0],
        )?;
        add(
            &mut self.l1_async_same_request_merges_local,
            async_same_request[1],
        )?;
        add(
            &mut self.l1_async_older_pending_merges_global,
            async_older_pending[0],
        )?;
        add(
            &mut self.l1_async_older_pending_merges_local,
            async_older_pending[1],
        )?;
        add(
            &mut self.l1_ordinary_global_older_pending_merges,
            ordinary_global_older_pending,
        )?;
        add(
            &mut self.l1_constant_global_older_pending_merges,
            constant_global_older_pending,
        )?;
        add(
            &mut self.l1_ordinary_global_same_request_merges,
            ordinary_global_same_request,
        )?;
        add(
            &mut self.l1_constant_global_same_request_merges,
            constant_global_same_request,
        )?;
        add(
            &mut self.l1_async_global_same_request_merges,
            async_global_same_request,
        )?;
        add(&mut self.l1_ordinary_global_hits, ordinary_global_hits)?;
        add(&mut self.l1_async_global_hits, async_global_hits)?;
        add(
            &mut self.l1_ordinary_global_older_sector_pending_merges,
            ordinary_global_older_sector_pending_merges,
        )?;
        add(
            &mut self.l1_constant_global_older_sector_pending_merges,
            constant_global_older_sector_pending_merges,
        )?;
        add(
            &mut self.l1_async_global_older_sector_pending_merges,
            async_global_older_sector_pending_merges,
        )?;
        add(
            &mut self.l1_unrequested_subline_miss_sectors,
            unrequested_subline_miss_sectors,
        )?;
        add(
            &mut self.l1_wavefront_mask_lookup_hits,
            wavefront_mask_lookup_hits,
        )?;
        add(
            &mut self.l1_wavefront_mask_lookup_misses,
            wavefront_mask_lookup_misses,
        )?;
        add(&mut self.l1_constant_global_hits, constant_global_hits)?;
        add(
            &mut self.l1_constant_global_requests,
            constant_global_requests,
        )?;
        add(&mut self.l1_global_resident_hits, global_resident_hits)?;
        add(&mut self.l1_global_lookup_sectors, global_lookup_sectors)?;
        for client in 0..request_mask_audit.request_count.len() {
            add(
                &mut self.l1_request_mask_audit.request_count[client],
                request_mask_audit.request_count[client],
            )?;
            add(
                &mut self.l1_request_mask_audit.request_line_count[client],
                request_mask_audit.request_line_count[client],
            )?;
            add(
                &mut self.l1_request_mask_audit.request_mask_overflow[client],
                request_mask_audit.request_mask_overflow[client],
            )?;
            for mask in 0..16 {
                add(
                    &mut self.l1_request_mask_audit.request_masks[client][mask],
                    request_mask_audit.request_masks[client][mask],
                )?;
                add(
                    &mut self.l1_request_mask_audit.same_request_merge_masks[client][mask],
                    request_mask_audit.same_request_merge_masks[client][mask],
                )?;
                add(
                    &mut self.l1_request_mask_audit.older_pending_merge_masks[client][mask],
                    request_mask_audit.older_pending_merge_masks[client][mask],
                )?;
            }
        }
        Ok(())
    }

    /// Record one dynamic constant-bank instruction as a source-domain
    /// observer. Each instruction contributes distinct 32-byte offset packets
    /// and the launch-wide sets retain address-domain cardinalities. These are
    /// not mathematical DRAM bounds and never change cache/service state.
    pub(crate) fn observe_constant_memory(
        &mut self,
        addresses: &[u64],
        bank: Option<u32>,
    ) -> Result<(), crate::IssueError> {
        if addresses.is_empty() {
            return Ok(());
        }
        add(&mut self.constant_memory_reference_count, 1)?;
        let address_count =
            u64::try_from(addresses.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory address count",
            })?;
        add(&mut self.constant_memory_address_count, address_count)?;
        let mut sectors = BTreeSet::new();
        for &address in addresses {
            // Field 15 is a bank-local effective offset, not a device
            // allocation address. Keep an absent bank distinct instead of
            // silently merging it with bank zero or another bank.
            sectors.insert((bank, address / 32));
        }
        let request_count =
            u64::try_from(sectors.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory sector request count",
            })?;
        add(&mut self.constant_memory_sector_requests, request_count)?;
        self.constant_memory_sector_set.extend(sectors);
        self.constant_memory_unique_sectors = u64::try_from(self.constant_memory_sector_set.len())
            .map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory unique sector count",
            })?;
        self.constant_memory_line_set
            .extend(addresses.iter().map(|&address| (bank, address / 64)));
        self.constant_memory_unique_lines = u64::try_from(self.constant_memory_line_set.len())
            .map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant memory unique line count",
            })?;
        Ok(())
    }

    pub(crate) fn observe_constant_metadata_only(&mut self) -> Result<(), crate::IssueError> {
        add(&mut self.constant_memory_metadata_only_records, 1)
    }

    /// Observe one statically classified `LDG.*.CONSTANT` global load. The
    /// address span is a source-qualified candidate only: its line domains
    /// are not equated with GCC requests, L2 sectors, or DRAM service.
    pub(crate) fn observe_constant_global_load(
        &mut self,
        block_id: [u32; 3],
        warp_id: i32,
        sectors: &[i64],
        off: usize,
        len: usize,
        opcode_text: &str,
        warp_uniform_address: bool,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.constant_global_load_instructions, 1)?;
        add(
            &mut self.constant_global_load_sectors,
            u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load sectors",
            })?,
        )?;
        let form = constant_global_load_form_index(opcode_text);
        add(&mut self.constant_global_load_form_instructions[form], 1)?;
        add(
            &mut self.constant_global_load_form_sectors[form],
            u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load form sectors",
            })?,
        )?;
        if warp_uniform_address {
            add(&mut self.constant_global_load_uniform_instructions, 1)?;
            add(
                &mut self.constant_global_load_uniform_sectors,
                u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                    counter: "constant global uniform load sectors",
                })?,
            )?;
        } else {
            add(&mut self.constant_global_load_divergent_instructions, 1)?;
            add(
                &mut self.constant_global_load_divergent_sectors,
                u64::try_from(len).map_err(|_| crate::IssueError::CounterOverflow {
                    counter: "constant global divergent load sectors",
                })?,
            )?;
        }
        let end = off
            .checked_add(len)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "constant global load sector span",
            })?;
        let span = sectors
            .get(off..end)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "constant global load sector span",
            })?;
        let lines: BTreeSet<u64> = span
            .iter()
            .map(|&sector| u64::try_from(sector).unwrap_or_default() / 4)
            .collect();
        add(
            &mut self.constant_global_load_request_candidates,
            u64::try_from(lines.len()).map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load request candidates",
            })?,
        )?;
        self.constant_global_line_set
            .extend(lines.into_iter().map(|line| (block_id, warp_id, line)));
        self.constant_global_load_unique_lines = u64::try_from(self.constant_global_line_set.len())
            .map_err(|_| crate::IssueError::CounterOverflow {
                counter: "constant global load unique lines",
            })?;
        self.constant_global_cta_line_set.extend(
            span.iter()
                .map(|&sector| (block_id, u64::try_from(sector).unwrap_or_default() / 4)),
        );
        self.constant_global_load_cta_unique_lines =
            u64::try_from(self.constant_global_cta_line_set.len()).map_err(|_| {
                crate::IssueError::CounterOverflow {
                    counter: "constant global load CTA unique lines",
                }
            })?;
        Ok(())
    }

    pub(crate) fn observe_atomic_conflict(
        &mut self,
        sectors: &[i64],
        off: usize,
        len: usize,
    ) -> Result<(), crate::IssueError> {
        let end = off
            .checked_add(len)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "atomic conflict sector span",
            })?;
        let span = sectors
            .get(off..end)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "atomic conflict sector span",
            })?;
        let distinct = span.iter().copied().collect::<BTreeSet<_>>().len();
        let aliases = len.saturating_sub(distinct);
        if aliases == 0 {
            add(&mut self.atomic_distinct_instructions, 1)?;
        } else {
            add(&mut self.atomic_conflict_instructions, 1)?;
            add(
                &mut self.atomic_conflict_alias_sectors,
                u64::try_from(aliases).map_err(|_| crate::IssueError::CounterOverflow {
                    counter: "atomic conflict alias sectors",
                })?,
            )?;
        }
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

    pub(crate) fn observe_tma_shadow_dynamic_identity(
        &mut self,
        matched: u64,
        unmatched: u64,
    ) -> Result<(), crate::IssueError> {
        add(&mut self.tma_shadow_dynamic_records_matched, matched)?;
        add(&mut self.tma_shadow_dynamic_records_unmatched, unmatched)?;
        Ok(())
    }

    pub(crate) fn observe_sync_sidecar(
        &mut self,
        observation: Option<&sim_trace::async_sidecars::SyncObservation>,
    ) -> Result<(), crate::IssueError> {
        let Some(observation) = observation else {
            return add(&mut self.sync_sidecar_records_missing, 1);
        };
        add(&mut self.sync_sidecar_records_observed, observation.records)?;
        add(
            &mut self.sync_sidecar_before_records,
            observation.before_records,
        )?;
        add(
            &mut self.sync_sidecar_after_records,
            observation.after_records,
        )?;
        add(
            &mut self.sync_sidecar_value_incomplete_records,
            observation.value_incomplete_records,
        )?;
        add(
            &mut self.sync_sidecar_mbarrier_address_records,
            observation.mbarrier_address_records,
        )?;
        add(
            &mut self.sync_sidecar_mbarrier_register_records,
            observation.mbarrier_register_records,
        )?;
        add(
            &mut self.sync_sidecar_arrival_records,
            observation.arrival_records,
        )?;
        add(
            &mut self.sync_sidecar_phase_check_records,
            observation.phase_check_records,
        )?;
        add(
            &mut self.sync_sidecar_wait_records,
            observation.wait_records,
        )?;
        add(
            &mut self.sync_sidecar_exchange_records,
            observation.exchange_records,
        )?;
        add(
            &mut self.sync_sidecar_barrier_addresses,
            u64::try_from(observation.barrier_addresses.len()).map_err(|_| {
                crate::IssueError::CounterOverflow {
                    counter: "sync_sidecar_barrier_addresses",
                }
            })?,
        )?;
        Ok(())
    }
    /// Lift the existing trace metrics into the factorized latent contract.
    /// Dynamic route identity remains unknown. The aggregate candidate-class
    /// vocabulary is not a single per-event route and cannot populate it.
    /// `class_count_envelope` separately represents the modeled class bounds.
    pub fn latent_factors(&self) -> Result<LatentFactors, LatentModelError> {
        let route =
            CandidateDomain::unknown("dynamic physical route/subunit/issue-slot unobserved")?;
        let compiler = CandidateDomain::unknown("no independent compiler witness")?;
        let cache = CandidateDomain::unknown("cache owner/read-from unobserved")?;
        let timing = CandidateDomain::unknown("endpoint identity not unified")?;
        Ok(LatentFactors {
            route,
            compiler,
            cache,
            timing,
        })
    }
    const MAX_COHERENCE_LINES: usize = 1_000_000;

    pub(crate) fn observe_coherence(
        &mut self,
        sm_id: Option<i32>,
        effective: u8,
        kind: AccessKind,
        sector_arena: &[i64],
        instrumentation_order: Option<u64>,
    ) -> Result<(), crate::IssueError> {
        if effective == 0 {
            return Ok(());
        }
        let Some(sm_id) = sm_id else {
            add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
            return Ok(());
        };
        // Placement validation currently bounds SM ids to the architecture's
        // card count (84 on SM86).  Keep a wider u128 domain for configurable
        // cards while failing closed if a future card exceeds it.
        let Some(sm_bit) = (sm_id >= 0)
            .then(|| u32::try_from(sm_id).ok())
            .flatten()
            .and_then(|id| (id < 128).then_some(1_u128 << id))
        else {
            add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
            return Ok(());
        };
        let (off, len, is_write) = match kind {
            AccessKind::GlobalEligible { off, len, .. }
            | AccessKind::GlobalAsyncCopy { off, len, .. }
            | AccessKind::GlobalUntagged { off, len }
            | AccessKind::GlobalBypass {
                off,
                len,
                token:
                    sim_mem::CacheOpToken::Other
                    | sim_mem::CacheOpToken::LdgECg
                    | sim_mem::CacheOpToken::LdgECs
                    | sim_mem::CacheOpToken::LdgEStrongSm
                    | sim_mem::CacheOpToken::LdgEStrongGpu
                    | sim_mem::CacheOpToken::Ldgsts
                    | sim_mem::CacheOpToken::LdgECa
                    | sim_mem::CacheOpToken::LdgEEf
                    | sim_mem::CacheOpToken::LdgENa,
                ..
            } => (off, len, false),
            AccessKind::GlobalStore { off, len }
            | AccessKind::GlobalReduction { off, len, .. }
            | AccessKind::GlobalBypass {
                off,
                len,
                token: sim_mem::CacheOpToken::Atom,
                ..
            } => (off, len, true),
            // Local memory is private by construction and shared-memory
            // addresses do not enter the L2 global coherence domain.
            AccessKind::LocalLoad { .. }
            | AccessKind::LocalStore { .. }
            | AccessKind::Shared { .. }
            | AccessKind::NonMemory
            | AccessKind::PredicatedOffMemory => return Ok(()),
            // Keep any newly added global token conservative until its
            // address-space semantics are explicitly classified.
            AccessKind::GlobalBypass { .. } => {
                add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
                return Ok(());
            }
        };
        if instrumentation_order.is_some() {
            add(&mut self.coherence.instrumentation_order_records, 1)?;
        } else {
            add(&mut self.coherence.instrumentation_order_missing_records, 1)?;
        }
        let start = off;
        let end = off
            .checked_add(usize::from(len))
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "coherence sector span",
            })?;
        let sectors = sector_arena
            .get(start..end)
            .ok_or(crate::IssueError::CounterOverflow {
                counter: "coherence sector span bounds",
            })?;
        for &sector in sectors {
            let line = sector.div_euclid(4);
            let lane = sector.rem_euclid(4) as usize;
            let entry = if let Some(entry) = self.coherence_lines.get_mut(&line) {
                entry
            } else {
                if self.coherence_lines.len() >= Self::MAX_COHERENCE_LINES {
                    add(&mut self.coherence.unknown_sm_or_line_records, 1)?;
                    continue;
                }
                add(&mut self.coherence.decoded_global_lines, 1)?;
                self.coherence_lines.entry(line).or_default()
            };
            entry.sm_mask |= sm_bit;
            entry.sector_mask |= 1_u8 << lane;
            entry.sector_sm_masks[lane] |= sm_bit;
            if is_write {
                entry.write_sm_mask |= sm_bit;
                entry.sector_write_sm_masks[lane] |= sm_bit;
            } else {
                entry.read_sm_mask |= sm_bit;
                entry.sector_read_sm_masks[lane] |= sm_bit;
            }
        }
        Ok(())
    }

    /// Finalize the bounded line map into stable counters before the trace
    /// metrics are cloned into a replay result.  The map is then released so
    /// reports do not retain one entry per physical line.
    pub(crate) fn finish_coherence(&mut self) -> Result<(), crate::IssueError> {
        for state in self.coherence_lines.values() {
            if state.sm_mask.count_ones() <= 1 {
                continue;
            }
            add(&mut self.coherence.cross_sm_lines, 1)?;
            add(
                &mut self.coherence.cross_sm_sectors,
                u64::from(state.sector_mask.count_ones()),
            )?;
            for sector in 0..state.sector_sm_masks.len() {
                let sm_mask = state.sector_sm_masks[sector];
                if sm_mask.count_ones() <= 1 {
                    continue;
                }
                add(&mut self.coherence.cross_sm_sector_overlap, 1)?;
                if state.sector_read_sm_masks[sector] != 0
                    && state.sector_write_sm_masks[sector] != 0
                {
                    add(&mut self.coherence.cross_sm_read_write_sectors, 1)?;
                }
                if state.sector_write_sm_masks[sector].count_ones() > 1 {
                    add(&mut self.coherence.cross_sm_write_write_sectors, 1)?;
                }
            }
            if state.read_sm_mask != 0 && state.write_sm_mask != 0 {
                add(&mut self.coherence.cross_sm_read_write_lines, 1)?;
            }
            if state.write_sm_mask.count_ones() > 1 {
                add(&mut self.coherence.cross_sm_write_write_lines, 1)?;
            }
        }
        self.coherence_lines.clear();
        Ok(())
    }

    pub(crate) fn observe_control_flow(
        &mut self,
        from_function: i32,
        from_pc: u32,
        from_active: u32,
        to_function: i32,
        to_pc: u32,
        to_active: u32,
    ) -> Result<(), crate::IssueError> {
        const MAX_DYNAMIC_EDGES: usize = 65_536;
        if from_active == 0 || to_active == 0 {
            return Ok(());
        }
        let overlap = from_active & to_active;
        if overlap == 0 {
            add(&mut self.control_flow.disjoint_path_switches, 1)?;
            return Ok(());
        }
        let key = format!("{from_function}:{from_pc:#x}->{to_function}:{to_pc:#x}");
        if !self.control_flow.edges.contains_key(&key)
            && self.control_flow.edges.len() >= MAX_DYNAMIC_EDGES
        {
            return Err(crate::IssueError::CounterOverflow {
                counter: "control-flow edge arena",
            });
        }
        let count = self.control_flow.edges.entry(key).or_default();
        add(count, 1)?;
        add(&mut self.control_flow.observed_edges, 1)?;
        add(
            &mut self.control_flow.active_lane_transitions,
            u64::from(overlap.count_ones()),
        )?;
        self.control_flow.unique_edges =
            u64::try_from(self.control_flow.edges.len()).map_err(|_| {
                crate::IssueError::CounterOverflow {
                    counter: "control-flow unique edge count",
                }
            })?;
        Ok(())
    }

    fn physical_candidates(
        head: &str,
        kind: &AccessKind,
        pipeline: crate::PipelineClass,
    ) -> &'static [&'static str] {
        const NONE: &[&str] = &[];
        const ADU_CBU: &[&str] = &["adu", "cbu"];
        const ADU_UNIFORM: &[&str] = &["adu", "uniform"];
        const ADU_LSU: &[&str] = &["adu", "lsu"];
        const LSU: &[&str] = &["lsu"];
        const TENSOR: &[&str] = &["tensor"];
        const FMA: &[&str] = &["fma"];
        const ALU: &[&str] = &["alu"];
        const XU: &[&str] = &["xu"];
        if matches!(
            head,
            "BRA" | "BRX" | "BRXU" | "JMP" | "JMX" | "JMXU" | "CALL" | "RET" | "EXIT"
        ) {
            return ADU_CBU;
        }
        if matches!(head, "BAR" | "BSSY" | "BSYNC" | "BREAK") {
            return ADU_CBU;
        }
        if head == "LDGSTS" {
            return ADU_LSU;
        }
        if matches!(
            head,
            "ULDC" | "LDC" | "CS2R" | "S2R" | "S2UR" | "UMOV" | "USHF"
        ) {
            return ADU_UNIFORM;
        }
        if head == "LDGDEPBAR" {
            return ADU_UNIFORM;
        }
        // MUFU is the dedicated multi-function operation family.  Unlike
        // the broader conversion/check families, its SM86 physical domain is
        // unambiguously the transcendental/data-type-conversion unit; retain
        // it as a candidate rather than counting it as an unknown opcode.
        if head == "MUFU" {
            return XU;
        }
        if head.starts_with("LD")
            || head.starts_with("ST")
            || head.starts_with("ATOM")
            || head.starts_with("RED")
            || !matches!(
                kind,
                AccessKind::NonMemory | AccessKind::PredicatedOffMemory
            )
        {
            return LSU;
        }
        match pipeline {
            crate::PipelineClass::Tensor { .. } => TENSOR,
            crate::PipelineClass::ScalarAlu { .. } => {
                if matches!(Self::ncu_pipeline(head), Some("fma")) {
                    FMA
                } else {
                    ALU
                }
            }
            _ => match Self::ncu_pipeline(head) {
                Some("tensor") => TENSOR,
                Some("fma") => FMA,
                Some("alu") => ALU,
                _ => NONE,
            },
        }
    }

    fn ncu_pipeline(head: &str) -> Option<&'static str> {
        if matches!(head, "HMMA" | "IMMA" | "BMMA" | "DMMA") {
            return Some("tensor");
        }
        if matches!(
            head,
            "IMAD" | "IMUL" | "FFMA" | "FADD" | "FMUL" | "FMAD" | "HADD2" | "HMUL2" | "HFMA2"
        ) {
            return Some("fma");
        }
        if matches!(
            head,
            "IADD3"
                | "UIADD3"
                | "LOP3"
                | "LEA"
                | "MOV"
                | "PRMT"
                | "ISETP"
                | "PSETP"
                | "SEL"
                | "SHF"
                | "F2FP"
                | "F2F"
                | "F2IP"
                | "I2FP"
                | "I2F"
        ) {
            return Some("alu");
        }
        // Nsight documents the LSU logical pipeline as including memory,
        // special-register, shuffle, and CTA-barrier traffic. DEPBAR is a
        // scheduler dependency instruction and is intentionally excluded.
        if head.starts_with("LD")
            || head.starts_with("ST")
            || head.starts_with("ATOM")
            || head.starts_with("RED")
            || matches!(head, "S2R" | "S2UR" | "SHFL")
        {
            return Some("lsu");
        }
        None
    }

    fn candidate_schedule_timing(pipeline: crate::PipelineClass) -> (u64, u64) {
        match pipeline {
            crate::PipelineClass::DispatchOnly => (1, 1),
            crate::PipelineClass::Branch {
                next_issue_interval_cycles,
            } => (1, next_issue_interval_cycles.max(1)),
            crate::PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                ..
            } => (1, warp_initiation_interval_cycles.max(1)),
            crate::PipelineClass::Tensor {
                initiation_interval_cycles,
            }
            | crate::PipelineClass::Sfu {
                initiation_interval_cycles,
            } => (1, initiation_interval_cycles.max(1)),
            crate::PipelineClass::ScalarAlu {
                latency_cycles,
                initiation_interval_cycles,
                ..
            } => (latency_cycles.max(1), initiation_interval_cycles.max(1)),
        }
    }

    fn schedule_candidates(
        &mut self,
        candidates: &[&'static str],
        pipeline: crate::PipelineClass,
    ) -> Result<(), crate::IssueError> {
        let sequence = self.physical_pipeline.scheduler_sequence;
        self.physical_pipeline.scheduler_sequence =
            sequence
                .checked_add(1)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "physical pipeline scheduler sequence",
                })?;
        if candidates.is_empty() {
            return Ok(());
        }
        let (latency, initiation_interval) = Self::candidate_schedule_timing(pipeline);
        let mut selected: Option<(&'static str, u64, u64)> = None;
        for &candidate in candidates {
            let free = self
                .physical_pipeline
                .scheduler_free
                .get(candidate)
                .copied()
                .unwrap_or(0);
            let start = sequence.max(free);
            let completion =
                start
                    .checked_add(latency)
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "physical pipeline scheduler completion",
                    })?;
            let better = match selected {
                None => true,
                Some((best_candidate, best_start, best_completion)) => {
                    (completion, start, candidate) < (best_completion, best_start, best_candidate)
                }
            };
            if better {
                selected = Some((candidate, start, completion));
            }
        }
        let (candidate, start, completion) = selected.expect("candidate domain is nonempty");
        for &candidate in candidates {
            let free = self
                .physical_pipeline
                .scheduler_free
                .get(candidate)
                .copied()
                .unwrap_or(0);
            let candidate_start = sequence.max(free);
            let candidate_completion =
                candidate_start
                    .checked_add(latency)
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "physical pipeline candidate interval completion",
                    })?;
            let key = candidate.to_owned();
            self.physical_pipeline
                .candidate_earliest_issue_cycles
                .entry(key.clone())
                .and_modify(|value| *value = (*value).min(candidate_start))
                .or_insert(candidate_start);
            self.physical_pipeline
                .candidate_latest_issue_cycles
                .entry(key.clone())
                .and_modify(|value| *value = (*value).max(candidate_start))
                .or_insert(candidate_start);
            self.physical_pipeline
                .candidate_earliest_completion_cycles
                .entry(key.clone())
                .and_modify(|value| *value = (*value).min(candidate_completion))
                .or_insert(candidate_completion);
            self.physical_pipeline
                .candidate_latest_completion_cycles
                .entry(key)
                .and_modify(|value| *value = (*value).max(candidate_completion))
                .or_insert(candidate_completion);
        }
        add(&mut self.physical_pipeline.model_interval_records, 1)?;
        let next_free =
            start
                .checked_add(initiation_interval)
                .ok_or(crate::IssueError::CounterOverflow {
                    counter: "physical pipeline scheduler availability",
                })?;
        self.physical_pipeline
            .scheduler_free
            .insert(candidate.to_owned(), next_free);
        let entry = self
            .physical_pipeline
            .scheduled_instructions
            .entry(candidate.to_owned())
            .or_default();
        add(entry, 1)?;
        let frontier = self.physical_pipeline.schedule_frontier_cycles.unwrap_or(0);
        self.physical_pipeline.schedule_frontier_cycles = Some(frontier.max(completion));
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn observe(
        &mut self,
        opcode: &str,
        active: u8,
        effective: u8,
        kind: AccessKind,
        pipeline: crate::PipelineClass,
        sector_arena: &[i64],
    ) -> Result<(), crate::IssueError> {
        self.observe_with_mask_presence(
            opcode,
            active,
            effective,
            true,
            true,
            kind,
            pipeline,
            sector_arena,
        )
    }

    pub(crate) fn observe_with_mask_presence(
        &mut self,
        opcode: &str,
        active: u8,
        effective: u8,
        active_mask_present: bool,
        predicate_mask_present: bool,
        kind: AccessKind,
        pipeline: crate::PipelineClass,
        sector_arena: &[i64],
    ) -> Result<(), crate::IssueError> {
        self.observe_with_mask_presence_arch(
            opcode,
            active,
            effective,
            active_mask_present,
            predicate_mask_present,
            kind,
            pipeline,
            sector_arena,
            sim_arch::ArchId::Sm86,
        )
    }

    pub(crate) fn observe_with_mask_presence_arch(
        &mut self,
        opcode: &str,
        active: u8,
        effective: u8,
        active_mask_present: bool,
        predicate_mask_present: bool,
        kind: AccessKind,
        pipeline: crate::PipelineClass,
        _sector_arena: &[i64],
        arch: sim_arch::ArchId,
    ) -> Result<(), crate::IssueError> {
        self.instructions.observe(
            active,
            effective,
            active_mask_present,
            predicate_mask_present,
        )?;
        let head = opcode_head(opcode);
        match head {
            "LDC" => add(&mut self.constant_client_ldc_instructions, 1)?,
            "ULDC" => add(&mut self.constant_client_uldc_instructions, 1)?,
            _ => {}
        }
        self.by_opcode.entry(head.to_owned()).or_default().observe(
            active,
            effective,
            active_mask_present,
            predicate_mask_present,
        )?;
        let pipe = match pipeline {
            crate::PipelineClass::DispatchOnly => "dispatch_only",
            crate::PipelineClass::Tensor { .. } => "tensor",
            crate::PipelineClass::ScalarAlu { .. } => "scalar_alu",
            crate::PipelineClass::Sfu { .. } => "uniform_sfu",
            crate::PipelineClass::Barrier { .. } => "barrier",
            crate::PipelineClass::Branch { .. } => "branch",
        };
        self.by_pipeline
            .entry(pipe.to_owned())
            .or_default()
            .observe(
                active,
                effective,
                active_mask_present,
                predicate_mask_present,
            )?;
        // NCU's `smsp__inst_executed_pipe_*` rows are warp-level instruction
        // populations and, like `smsp__inst_executed.sum`, retain
        // predicated-off issue records. Thread/predicate counts in the nested
        // activity row still preserve the active/effective distinction. Keep
        // unclassified heads visible instead of silently dropping them: on
        // SM86 some instructions can issue through more than one physical
        // path, and the trace does not carry the dynamic pipe assignment.
        let ncu_pipe = Self::ncu_pipeline(head).unwrap_or("unclassified");
        self.by_ncu_pipeline
            .entry(ncu_pipe.to_owned())
            .or_default()
            .observe(
                active,
                effective,
                active_mask_present,
                predicate_mask_present,
            )?;
        if effective != 0 {
            if let AccessKind::GlobalAsyncCopy { l1_policy, .. } = kind {
                match l1_policy {
                    LdgstsL1Policy::CacheAtAllLevels => {
                        add(&mut self.ldgsts_cache_access_instructions, 1)?;
                    }
                    LdgstsL1Policy::BypassL1 => {
                        add(&mut self.ldgsts_cache_bypass_instructions, 1)?;
                    }
                }
            }
        }
        let candidates = Self::physical_candidates(head, &kind, pipeline);
        self.schedule_candidates(candidates, pipeline)?;
        add(&mut self.physical_pipeline.total_warp_instructions, 1)?;
        if candidates.is_empty() {
            add(&mut self.physical_pipeline.unknown_instructions, 1)?;
        } else if candidates.len() == 1 {
            add(&mut self.physical_pipeline.unique_instructions, 1)?;
            add(
                self.physical_pipeline
                    .mandatory_class_instructions
                    .entry(candidates[0].to_owned())
                    .or_default(),
                1,
            )?;
        } else {
            add(&mut self.physical_pipeline.ambiguous_instructions, 1)?;
        }
        for candidate in candidates {
            let entry = self
                .physical_pipeline
                .candidate_instructions
                .entry((*candidate).to_owned())
                .or_default();
            add(entry, 1)?;
        }
        if candidates.is_empty() {
            add(&mut self.physical_pipeline.selected_unknown_instructions, 1)?;
        } else {
            let selected = candidates
                .iter()
                .min_by_key(|candidate| {
                    (
                        self.physical_pipeline
                            .selected_instructions
                            .get(**candidate)
                            .copied()
                            .unwrap_or(0),
                        **candidate,
                    )
                })
                .expect("nonempty candidate domain");
            let entry = self
                .physical_pipeline
                .selected_instructions
                .entry((*selected).to_owned())
                .or_default();
            add(entry, 1)?;
        }
        // Nsight's SMSP branch counter is the CBU instruction population,
        // excluding only BMOV and BSSY.  The old projection counted only the
        // direct branch heads, which systematically omitted control-flow
        // instructions such as BSYNC, EXIT, WARPSYNC, BREAK, and CONT.  Keep
        // this separate from the SASS branch subset (the `sm__sass_*` metric
        // family) and from the branch-target observer below: this is a
        // warp-instruction cardinality, so predicated-off records still count
        // once when the warp executed the instruction.
        let branch_counter = if arch == sim_arch::ArchId::Sm120 {
            matches!(
                head,
                "BRA"
                    | "BRX"
                    | "BRXU"
                    | "JMP"
                    | "JMX"
                    | "JMXU"
                    | "CALL"
                    | "RET"
                    | "EXIT"
                    | "BSYNC"
                    | "WARPSYNC"
                    | "YIELD"
                    | "BREAK"
                    | "CONT"
            )
        } else {
            matches!(head, "BRA" | "BRX" | "BRXU" | "JMP" | "JMX" | "JMXU")
        };
        if branch_counter {
            add(&mut self.branch_warp_instructions, 1)?;
            add(
                &mut self.branch_mixed_predicate_warp_instructions,
                u64::from(
                    active_mask_present
                        && predicate_mask_present
                        && effective > 0
                        && effective < active,
                ),
            )?;
        }
        if effective == 0 {
            return Ok(());
        }
        if let AccessKind::GlobalStore { off, len } = kind {
            let end =
                off.checked_add(usize::from(len))
                    .ok_or(crate::IssueError::CounterOverflow {
                        counter: "global-store sector span",
                    })?;
            if let Some(span) = _sector_arena.get(off..end) {
                let current = span.iter().copied().collect::<BTreeSet<_>>();
                if effective >= 8 && self.global_store_previous_effective_lanes >= 8 {
                    let duplicate = current
                        .intersection(&self.global_store_previous_sectors)
                        .count() as u64;
                    add(&mut self.global_store_adjacent_duplicate_sectors, duplicate)?;
                }
                self.global_store_previous_sectors = current;
                self.global_store_previous_effective_lanes = effective;
            } else {
                self.global_store_previous_sectors.clear();
                self.global_store_previous_effective_lanes = 0;
            }
        } else {
            self.global_store_previous_sectors.clear();
            self.global_store_previous_effective_lanes = 0;
        }
        if arch == sim_arch::ArchId::Sm120
            && matches!(kind, AccessKind::Shared { .. })
            && head == "LDS"
        {
            let sass_head = opcode.split_ascii_whitespace().next().unwrap_or(opcode);
            let width16 = sass_head.split('.').any(|token| token == "128");
            let width4 = !width16 && !sass_head.split('.').any(|token| token == "64");
            if effective == 32 && width16 {
                add(&mut self.shared_load_full_warp_width16_records, 1)?;
                if let AccessKind::Shared {
                    data_wavefronts,
                    data_ideal_wavefronts,
                    ..
                } = kind
                {
                    if data_wavefronts.saturating_mul(2) == data_ideal_wavefronts {
                        add(
                            &mut self.shared_load_width16_observer_reduction,
                            data_wavefronts,
                        )?;
                        let current_excess = data_wavefronts.saturating_sub(data_ideal_wavefronts);
                        let original_excess = data_wavefronts
                            .saturating_mul(2)
                            .saturating_sub(data_ideal_wavefronts);
                        add(
                            &mut self.shared_load_width16_excess_reduction,
                            original_excess.saturating_sub(current_excess),
                        )?;
                    }
                }
            } else if effective == 32 && width4 {
                add(&mut self.shared_load_full_warp_width4_records, 1)?;
            }
        }
        let (key, sectors, bank_wavefronts) = match kind {
            AccessKind::NonMemory | AccessKind::PredicatedOffMemory => return Ok(()),
            AccessKind::Shared {
                data_wavefronts, ..
            } => (
                // Collective matrix stores are the store-side counterpart
                // of LDSM.  Keeping STSM in the load bucket makes the
                // operation-qualified opcode counters report matrix stores
                // as shared loads and leaves the aggregate store-wavefront
                // observer empty.  Preserve both matrix forms in their
                // direction-specific buckets; the conflict surface below
                // still excludes them from ordinary LDS/STS bank conflicts.
                if matches!(head, "STS" | "STSM") {
                    "shared_store"
                } else if head.starts_with("ATOM") {
                    "shared_atomic"
                } else {
                    "shared_load"
                },
                0,
                data_wavefronts,
            ),
            AccessKind::LocalLoad { len, .. } => ("local_load", u64::from(len), 0),
            AccessKind::LocalStore { len, .. } => ("local_store", u64::from(len), 0),
            AccessKind::GlobalStore { len, .. } => ("global_store", u64::from(len), 0),
            AccessKind::GlobalReduction { len, .. } => (
                "global_reduction",
                // RED is a non-returning, lane-serialized write operation.
                // The validated sector span deliberately preserves repeated
                // aliases: GA10x reports those transaction sectors rather
                // than collapsing them to unique sector IDs (the same
                // convention is used by the atomic projection and by the
                // issue-side alias-aware wavefront counter).
                u64::from(len),
                0,
            ),
            AccessKind::GlobalBypass {
                token: CacheOpToken::Atom,
                off: _,
                len,
                ..
            } => (
                "global_atomic",
                // The validated atomic sector arena repeats each touched
                // sector by its maximum same-word alias multiplicity. This
                // reproduces GA10x's atom-sector surface: coalesced words use
                // unique sectors, while same-word aliases remain serialized.
                u64::from(len),
                0,
            ),
            AccessKind::GlobalAsyncCopy { len, .. } => ("async_global_load", u64::from(len), 0),
            AccessKind::GlobalEligible { len, .. } | AccessKind::GlobalBypass { len, .. } => {
                ("global_load", u64::from(len), 0)
            }
            AccessKind::GlobalUntagged { len, .. } => ("unclassified_global", u64::from(len), 0),
        };
        let row = self.memory.entry(key.to_owned()).or_default();
        add(&mut row.requests, 1)?;
        add(&mut row.sectors, sectors)?;
        if let Some((unique_sectors, unique_lines)) = decoded_sector_stats(kind, _sector_arena) {
            add(&mut row.unique_sectors, unique_sectors)?;
            add(&mut row.unique_lines, unique_lines)?;
            add(
                &mut row.sector_aliases,
                sectors.saturating_sub(unique_sectors),
            )?;
        }
        add(&mut row.bank_wavefronts, bank_wavefronts)?;
        if let AccessKind::Shared {
            data_wavefronts,
            data_ideal_wavefronts,
            ..
        } = kind
        {
            add(&mut row.ideal_bank_wavefronts, data_ideal_wavefronts)?;
            // Nsight's operation-qualified shared-load/store conflict rows
            // do not include matrix loads/stores. Keep those operations in
            // the aggregate wavefront surface, but exclude them from the
            // corresponding conflict numerator.
            let conflict_surface = !matches!(head, "LDSM" | "STSM");
            if conflict_surface {
                add(&mut row.bank_conflict_wavefronts, data_wavefronts)?;
                add(
                    &mut row.ideal_bank_conflict_wavefronts,
                    data_ideal_wavefronts,
                )?;
                add(
                    &mut row.bank_conflict_excess_wavefronts,
                    data_wavefronts.saturating_sub(data_ideal_wavefronts),
                )?;
            }
        }
        if let Some(scope) = memory_scope_key(opcode, kind) {
            let row = self.memory_scope.entry(scope.to_owned()).or_default();
            add(&mut row.requests, 1)?;
            add(&mut row.sectors, sectors)?;
            if let Some((unique_sectors, unique_lines)) = decoded_sector_stats(kind, _sector_arena)
            {
                add(&mut row.unique_sectors, unique_sectors)?;
                add(&mut row.unique_lines, unique_lines)?;
                add(
                    &mut row.sector_aliases,
                    sectors.saturating_sub(unique_sectors),
                )?;
            }
            add(&mut row.bank_wavefronts, bank_wavefronts)?;
        }
        Ok(())
    }
}

/// Half-open, model-relative lifetimes [admission, retirement). No hardware
/// timestamps are used. Retired warps no longer contribute even if their CTA
/// still retains resource allocations. Pending write acknowledgements are a
/// separate completion domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct WarpLifetime {
    pub block: i64,
    pub warp_slot: i32,
    pub admission: u64,
    pub retirement: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SmActivity {
    pub warp_active_cycles: u128,
    pub sm_active_cycles: u64,
    pub max_resident_warps_per_sm: u32,
    pub peak_live_warps: u32,
    pub smsp_active_cycles: [u64; 4],
    pub instructions_per_smsp: [u64; 4],
    pub issue_active_cycles_per_smsp: [u64; 4],
    /// Number of distinct executed instruction sectors assigned to this SM.
    /// This is an observation-side scope anchor for a future per-SM effective
    /// front-end envelope; it does not alter replay timing.
    pub front_end_instruction_read_sectors: u64,
    pub warp_lifetimes: Vec<WarpLifetime>,
}

fn union_length(intervals: &mut [(u64, u64)]) -> u64 {
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

impl SmActivity {
    pub(crate) fn from_lifetimes(
        warp_lifetimes: Vec<WarpLifetime>,
        instructions_per_smsp: [u64; 4],
        issue_active_cycles_per_smsp: [u64; 4],
        max_resident_warps_per_sm: u32,
    ) -> Self {
        let warp_active_cycles: u128 = warp_lifetimes
            .iter()
            .map(|w| u128::from(w.retirement.saturating_sub(w.admission)))
            .sum();
        let mut intervals: Vec<_> = warp_lifetimes
            .iter()
            .map(|w| (w.admission, w.retirement))
            .collect();
        let sm_active_cycles = union_length(&mut intervals);
        let mut endpoints: Vec<(u64, i32)> = warp_lifetimes
            .iter()
            .flat_map(|w| [(w.admission, 1_i32), (w.retirement, -1_i32)])
            .collect();
        endpoints.sort_unstable_by_key(|&(time, delta)| (time, delta));
        let mut live = 0_i32;
        let mut peak_live_warps = 0_u32;
        for (_, delta) in endpoints {
            live = live.saturating_add(delta);
            peak_live_warps = peak_live_warps.max(u32::try_from(live).unwrap_or(0));
        }
        let smsp_active_cycles = std::array::from_fn(|sc| {
            intervals.clear();
            intervals.extend(
                warp_lifetimes
                    .iter()
                    .filter(|w| w.warp_slot as usize % 4 == sc)
                    .map(|w| (w.admission, w.retirement)),
            );
            union_length(&mut intervals)
        });
        Self {
            warp_active_cycles,
            sm_active_cycles,
            max_resident_warps_per_sm,
            peak_live_warps,
            smsp_active_cycles,
            instructions_per_smsp,
            issue_active_cycles_per_smsp,
            front_end_instruction_read_sectors: 0,
            warp_lifetimes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(off: usize, len: u16) -> AccessKind {
        AccessKind::GlobalEligible {
            off,
            len,
            token: CacheOpToken::LdgECa,
        }
    }

    #[test]
    fn reuse_observer_deduplicates_sectors_and_tracks_proxy_monotonicity() {
        let mut metrics = TraceMetrics::default();
        metrics.begin_reuse_warp_stream();
        let sectors = [0_i64, 4, 0, 0, 1];
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(1, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(2, 1), &sectors).unwrap();
        metrics
            .observe_reuse_memory(
                AccessKind::GlobalBypass {
                    off: 3,
                    len: 2,
                    token: CacheOpToken::Atom,
                    l2_requests: 1,
                },
                &sectors,
            )
            .unwrap();

        let reuse = &metrics.reuse_interval;
        assert_eq!(reuse.streams, 1);
        assert_eq!(reuse.line_references, 4);
        assert_eq!(reuse.stream_unique_lines, 2);
        assert_eq!(reuse.cold_line_references, 2);
        assert_eq!(reuse.reused_line_references, 2);
        assert_eq!(reuse.sector_references, 5);
        assert_eq!(reuse.cold_sector_references, 3);
        assert_eq!(reuse.reused_sector_references, 2);
        assert_eq!(reuse.line_reuse_interval_histogram[0], 1);
        assert_eq!(reuse.line_reuse_interval_histogram[1], 1);
        assert_eq!(reuse.sector_reuse_interval_histogram[0], 1);
        assert_eq!(reuse.sector_reuse_interval_histogram[1], 1);
        assert_eq!(reuse.line_proxy_miss_references[0], 4);
        assert_eq!(reuse.line_proxy_miss_references[1], 3);
        assert_eq!(reuse.line_proxy_miss_references[2], 2);
        assert!(
            reuse
                .line_proxy_miss_references
                .windows(2)
                .all(|window| window[0] >= window[1])
        );
        assert_eq!(reuse.by_class["global_load"].line_references, 3);
        assert_eq!(reuse.by_class["atomic_or_reduction"].sector_references, 2);
    }

    #[test]
    fn reuse_observer_resets_cold_state_at_warp_boundary() {
        let mut metrics = TraceMetrics::default();
        let sectors = [7_i64];
        metrics.begin_reuse_warp_stream();
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        metrics.begin_reuse_warp_stream();
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        assert_eq!(metrics.reuse_interval.streams, 2);
        assert_eq!(metrics.reuse_interval.line_references, 2);
        assert_eq!(metrics.reuse_interval.stream_unique_lines, 2);
        assert_eq!(metrics.reuse_interval.cold_line_references, 2);
        assert_eq!(metrics.reuse_interval.reused_line_references, 0);
    }

    #[test]
    fn exact_mrc_respects_same_tick_batch_and_sector_refill() {
        let mut metrics = TraceMetrics::default();
        metrics.begin_reuse_warp_stream();
        let sectors = [0_i64, 4, 0];
        // Two lines in one logical tick are both cold, but neither line is an
        // intervening reference for the next tick's reuse of line zero.
        metrics.observe_reuse_memory(load(0, 2), &sectors).unwrap();
        metrics.observe_reuse_memory(load(2, 1), &sectors).unwrap();
        let mrc = &metrics.locality_mrc.by_origin["global"];
        assert_eq!(mrc.line_references, 3);
        assert_eq!(mrc.cold_line_references, 2);
        assert_eq!(mrc.sector_references, 3);
        assert_eq!(mrc.cold_sector_references, 2);

        let capacity_one = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 1)
            .unwrap();
        let capacity_two = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 2)
            .unwrap();
        // The line-zero reuse has RD=0 because line one was in the same prior
        // tick. It therefore hits at capacity one and two.
        assert_eq!(mrc.line_miss_references[capacity_one], 2);
        assert_eq!(mrc.line_miss_references[capacity_two], 2);
        assert_eq!(mrc.sector_miss_references[capacity_one], 2);
        assert_eq!(mrc.sector_miss_references[capacity_two], 2);
        // The order-independent envelope admits line one before line zero in
        // the prior batch, so the final reuse can have RD=1.
        assert_eq!(mrc.line_miss_references_upper[capacity_one], 3);
        assert_eq!(mrc.sector_miss_references_upper[capacity_one], 3);
        assert_eq!(mrc.maximum_line_rd_envelope_width, 1);
    }

    #[test]
    fn exact_mrc_tracks_sector_validity_under_line_eviction() {
        let mut metrics = TraceMetrics::default();
        metrics.begin_reuse_warp_stream();
        let sectors = [0_i64, 4, 8, 1];
        // Tick 1: line zero sector zero; tick 2: line one; tick 3: line two;
        // tick 4: line zero sector one. At C=2 the line is evicted (two
        // intervening distinct lines), so the final reference is a line-tag
        // miss rather than a resident-line sector fill.
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(1, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(2, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(3, 1), &sectors).unwrap();
        let mrc = &metrics.locality_mrc.by_origin["global"];
        let capacity_two = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 2)
            .unwrap();
        assert_eq!(mrc.line_miss_references[capacity_two], 4);
        assert_eq!(
            mrc.sector_line_tag_miss_references[capacity_two],
            mrc.sector_miss_references[capacity_two]
        );
    }

    #[test]
    fn exact_mrc_marks_reused_sector_miss_after_intervening_line_eviction() {
        let mut metrics = TraceMetrics::default();
        metrics.begin_reuse_warp_stream();
        let sectors = [0_i64, 4, 8];
        // A sector zero reference, two intervening distinct lines, then the
        // same sector. The line-level RD is two, so sector zero is not a hit
        // at a two-line fully-associative capacity.
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(1, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(2, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        let mrc = &metrics.locality_mrc.by_origin["global"];
        let capacity_two = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 2)
            .unwrap();
        assert_eq!(mrc.sector_miss_references[capacity_two], 4);
        assert_eq!(mrc.sector_line_tag_miss_references[capacity_two], 4);
    }

    #[test]
    fn exact_mrc_envelope_uses_the_reused_line_batch_not_only_previous_tick() {
        let mut metrics = TraceMetrics::default();
        metrics.begin_reuse_warp_stream();
        let sectors = [0_i64, 4, 8];
        // The reused line was last in a two-line batch. A different line is
        // then observed in a separate tick before line zero returns.
        metrics.observe_reuse_memory(load(0, 2), &sectors).unwrap();
        metrics.observe_reuse_memory(load(2, 1), &sectors).unwrap();
        metrics.observe_reuse_memory(load(0, 1), &sectors).unwrap();
        let mrc = &metrics.locality_mrc.by_origin["global"];
        let capacity_two = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 2)
            .unwrap();
        // Lower RD counts only line two; upper RD also admits line one from
        // the previous batch, so capacity two is an envelope boundary.
        assert_eq!(mrc.line_miss_references[capacity_two], 3);
        assert_eq!(mrc.line_miss_references_upper[capacity_two], 4);
    }

    #[test]
    fn exact_mrc_fenwick_resize_preserves_stack_counts() {
        let mut stream = LocalityMrcOriginStream {
            fenwick: LocalityMrcFenwick::new(),
            lines: BTreeMap::new(),
            batches: BTreeMap::new(),
            next_batch_id: 0,
            ..LocalityMrcOriginStream::default()
        };
        let mut report = LocalityMrcClassObservation::default();
        for line in 0_i64..5000 {
            let mut batch = BTreeMap::new();
            batch.insert(line, 1_u8);
            stream.observe(&batch, &mut report).unwrap();
        }
        let mut final_batch = BTreeMap::new();
        final_batch.insert(0_i64, 1_u8);
        stream.observe(&final_batch, &mut report).unwrap();
        let c4096 = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 4096)
            .unwrap();
        let c8192 = LOCALITY_MRC_CAPACITY_LINES
            .iter()
            .position(|&capacity| capacity == 8192)
            .unwrap();
        // The final line-zero reuse has 4,999 distinct intervening lines.
        assert_eq!(report.line_miss_references[c4096], 5001);
        assert_eq!(report.line_miss_references[c8192], 5000);
        assert_eq!(report.cold_line_references, 5000);
    }
}
