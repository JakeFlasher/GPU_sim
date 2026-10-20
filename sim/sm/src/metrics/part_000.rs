// Lexical implementation section 1/3; included by the parent facade.
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

