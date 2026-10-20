//! Shared-L2 replay session state; drives event timing and cache interactions.

use super::*;

/// One address-derived event for a diagnostic joint L2 shadow. `order` is
/// the producer's instrumentation token, never a cycle or engine timestamp.
#[derive(Clone, Copy)]
pub(crate) struct JointL2ShadowEvent {
    pub(crate) order: u64,
    pub(crate) sequence: u64,
    pub(crate) sector_id: i64,
    pub(crate) class: L2OpClass,
    pub(crate) tma_load: bool,
    pub(crate) ordinary_read: bool,
}

pub(crate) fn observe_joint_l2_shadow(
    mut events: Vec<JointL2ShadowEvent>,
    config: L2Config,
    order_unavailable: u64,
    events_dropped: u64,
) -> Result<JointL2ShadowStats, ReplayError> {
    events.sort_unstable_by_key(|event| (event.order, event.sequence));
    let mut shadow = L2Model::new(config)?;
    let mut stats = JointL2ShadowStats {
        order_unavailable,
        events_dropped,
        ..JointL2ShadowStats::default()
    };
    for event in events {
        let line_id = event.sector_id.div_euclid(4);
        let sector = u8::try_from(event.sector_id.rem_euclid(4)).map_err(|_| {
            ReplayError::System(SystemError::L2(sim_mem::L2AccessError::SectorOutOfRange {
                sector: u8::MAX,
            }))
        })?;
        let outcome = shadow
            .access(line_id, sector, event.class)
            .map_err(|error| ReplayError::System(SystemError::L2(error)))?;
        stats.requests = stats
            .requests
            .checked_add(1)
            .ok_or(IssueError::CounterOverflow {
                counter: "joint TMA/L2 shadow requests",
            })?;
        let hit = u64::from(outcome.counted_hit);
        let miss = u64::from(!outcome.counted_hit);
        stats.hits = stats
            .hits
            .checked_add(hit)
            .ok_or(IssueError::CounterOverflow {
                counter: "joint TMA/L2 shadow hits",
            })?;
        stats.misses = stats
            .misses
            .checked_add(miss)
            .ok_or(IssueError::CounterOverflow {
                counter: "joint TMA/L2 shadow misses",
            })?;
        if event.tma_load {
            stats.tma_load[0] = stats.tma_load[0].saturating_add(1);
            stats.tma_load[1] = stats.tma_load[1].saturating_add(hit);
            stats.tma_load[2] = stats.tma_load[2].saturating_add(miss);
        }
        if event.ordinary_read {
            stats.ordinary_read[0] = stats.ordinary_read[0].saturating_add(1);
            stats.ordinary_read[1] = stats.ordinary_read[1].saturating_add(hit);
            stats.ordinary_read[2] = stats.ordinary_read[2].saturating_add(miss);
        }
    }
    Ok(stats)
}

#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error(transparent)]
    Trace(#[from] TraceError),
    #[error(transparent)]
    AsyncSidecar(#[from] sim_trace::async_sidecars::AsyncSidecarError),
    #[error(transparent)]
    Issue(#[from] IssueError),
    #[error(transparent)]
    System(#[from] SystemError),
    #[error("L2 config invalid: {0}")]
    L2Config(#[from] sim_mem::L2ConfigError),
    #[error("L2 construction failed: {0}")]
    L2Build(#[from] sim_mem::L2BuildError),
    #[error("L2 snapshot failed: {0}")]
    L2Snapshot(#[from] sim_mem::L2SnapshotError),
    #[error("L2 counter operation failed: {0}")]
    L2Counters(#[from] sim_mem::L2CounterError),
    #[error("retained L2 configuration mismatch: expected {expected:?}, got {got:?}")]
    RetainedL2ConfigurationMismatch { expected: L2Config, got: L2Config },
    #[error(
        "retained L2 launch boundary is not fully drained: {pending_sectors} sector(s) remain pending"
    )]
    RetainedL2NotDrained { pending_sectors: usize },
    #[error(
        "memory-service candidates require L1SharedL2 replay so their issue-time topology is evaluated on the full hierarchy"
    )]
    MemoryServiceCandidatesRequireSharedL2,
    #[error(
        "memory-service candidates require full launch coverage; sampled replay discards per-SM timing diagnostics"
    )]
    MemoryServiceCandidatesRequireFullCoverage,
    #[error(
        "memory-admission candidates require L1SharedL2 replay so exact L2 outcomes are available"
    )]
    MemoryAdmissionCandidatesRequireSharedL2,
    #[error(
        "memory-admission candidates require full launch coverage; sampled replay discards per-SM timing diagnostics"
    )]
    MemoryAdmissionCandidatesRequireFullCoverage,
    #[error("L1 payload admission events require shared-L2 replay")]
    PayloadAdmissionRequiresSharedL2,
    #[error("L1 payload admission event {index} has no dynamic trace identity")]
    PayloadAdmissionIdentityMissing { index: usize },
    #[error("L1 payload admission event {index} does not match a retained dynamic trace identity")]
    PayloadAdmissionIdentityNotObserved { index: usize },
    #[error(
        "L1 payload admission event {index} targets SM {event_sm}, but the retained dynamic instruction is on SM {trace_sm:?}"
    )]
    PayloadAdmissionSmMismatch {
        index: usize,
        event_sm: i32,
        trace_sm: Option<i32>,
    },
    #[error(
        "L1 payload admission event {index} binds to a non-store or unclassified GLOBAL access ({access_kind:?})"
    )]
    PayloadAdmissionNotStore {
        index: usize,
        access_kind: Option<pb::MemoryOpAccessKind>,
    },
    #[error("L1 payload admission event {index} has no retained physical warp-slot placement")]
    PayloadAdmissionWriterPlacementMissing { index: usize },
    #[error(
        "opt-in coherence service requires a complete decoded topology and lane masks; full_coverage={full_coverage}, decoded={decoded}, unknown_records={unknown_records}, active_masks_missing={active_masks_missing}, predicate_masks_missing={predicate_masks_missing}"
    )]
    CoherenceTopologyIncomplete {
        full_coverage: bool,
        decoded: bool,
        unknown_records: u64,
        active_masks_missing: u64,
        predicate_masks_missing: u64,
    },
    #[error("opt-in dirty writeback service requires complete unsampled launch coverage")]
    WritebackRequiresFullCoverage,
    #[error("core-only diagnostic requires complete unsampled trace coverage")]
    FastFrontierRequiresFullCoverage,
    #[error(
        "core-only diagnostic requires a single launch with no concurrent stream context; launches={launches}"
    )]
    FastFrontierLaunchScopeUnknown { launches: usize },
    #[error(
        "core-only diagnostic requires active and predicate-mask presence on every dynamic record; active_missing={active_missing}, predicate_missing={predicate_missing}"
    )]
    FastFrontierMasksIncomplete {
        active_missing: u64,
        predicate_missing: u64,
    },
    #[error("SM {sm_id} L1 counter aggregation failed: {source}")]
    L1Aggregation {
        sm_id: i32,
        #[source]
        source: L1CounterError,
    },
    #[error("SM {sm_id} L1 counter presence mismatch: expected={expected}, found={found}")]
    L1PresenceMismatch {
        sm_id: i32,
        expected: bool,
        found: bool,
    },
    #[error("replay architecture mismatch for {component}: expected {expected}, got {got}")]
    ArchitectureMismatch {
        component: &'static str,
        expected: ArchId,
        got: ArchId,
    },
    #[error(
        "trace binary_version {binary_version} is outside the unsigned architecture-version domain"
    )]
    TraceArchitectureVersionOutOfRange { binary_version: i32 },
    #[error(
        "trace binary_version {binary_version} is not an admitted deployment architecture: {source}"
    )]
    TraceArchitectureUnknown {
        binary_version: u32,
        #[source]
        source: sim_arch::ArchError,
    },
    #[error("architecture-card placement data is unavailable: {source}")]
    ArchitectureCardPlacement {
        #[source]
        source: CalibError,
    },
    #[error("architecture {arch} shared-memory capacity is unavailable from its card: {source}")]
    ArchitectureCardSharedMemoryCapacity {
        arch: ArchId,
        #[source]
        source: CalibError,
    },
    #[error("architecture {arch} card has invalid per-SM shared-memory capacity {bytes} bytes")]
    SharedMemoryCapacityInvalid { arch: ArchId, bytes: u64 },
    #[error(
        "kernel {kernel_id} launch shared-memory total {bytes} is outside the unsigned byte domain"
    )]
    SharedMemoryDemandInvalid { kernel_id: i32, bytes: i32 },
    #[error(
        "kernel {kernel_id} requires {bytes_per_cta} shared-memory bytes per CTA, exceeding architecture {arch} per-SM capacity {max_bytes_per_sm}"
    )]
    CtaSharedMemoryCapacityExceeded {
        kernel_id: i32,
        arch: ArchId,
        bytes_per_cta: u64,
        max_bytes_per_sm: u64,
    },
    #[error(
        "kernel {kernel_id} requires {bytes_per_cta} shared-memory bytes per CTA, exceeding architecture {arch} per-block opt-in limit {max_bytes_per_block}"
    )]
    CtaSharedMemoryBlockLimitExceeded {
        kernel_id: i32,
        arch: ArchId,
        bytes_per_cta: u64,
        max_bytes_per_block: u64,
    },
    #[error(
        "kernel {kernel_id} block has {threads_per_cta} threads, exceeding architecture {arch} per-block thread limit {max_threads_per_block}"
    )]
    CtaThreadCapacityExceeded {
        kernel_id: i32,
        arch: ArchId,
        threads_per_cta: u64,
        max_threads_per_block: u32,
    },
    #[error(
        "{path}: traced sm_id {sm_id} is outside architecture {arch} SKU {sku:?} identifier range 0..{sm_id_upper_bound}"
    )]
    SmIdOutOfRange {
        path: PathBuf,
        sm_id: i32,
        arch: ArchId,
        sku: Sku,
        sm_id_upper_bound: u32,
    },
    #[cfg(feature = "campaign")]
    #[error(
        "trace_sampling per_sm_slot_map block {launch_index} declares sm_id {sm_id} outside architecture {arch} SKU {sku:?} identifier range 0..{sm_id_upper_bound}"
    )]
    SamplingSmIdOutOfRange {
        launch_index: u64,
        sm_id: u64,
        arch: ArchId,
        sku: Sku,
        sm_id_upper_bound: u32,
    },
    #[cfg(feature = "campaign")]
    #[error(
        "trace_sampling per_sm_slot_map has no placement for observed launch block {launch_index}"
    )]
    SamplingPlacementMissing { launch_index: i64 },
    #[cfg(feature = "campaign")]
    #[error(
        "launch block {launch_index} traced sm_id {traced_sm_id}, but trace_sampling per_sm_slot_map declares {declared_sm_id}"
    )]
    SamplingPlacementMismatch {
        launch_index: i64,
        traced_sm_id: i32,
        declared_sm_id: u64,
    },

    #[error(
        "memory class {class:?} has no implemented replay path for architecture {arch}; the \
         simulator refuses to route it through generic global timing"
    )]
    UnsupportedMemoryPath {
        arch: ArchId,
        class: pb::MemoryOpClass,
    },
    #[error(
        "TMA opcode {opcode:?} at pc {pc:#x} on architecture {arch} has an unsupported \
         direction; only the source-global/destination-shared load form is modeled"
    )]
    UnsupportedTmaDirection {
        arch: ArchId,
        pc: u32,
        opcode: String,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: {class:?} operand routing is invalid: {source}"
    )]
    MemoryRoute {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        class: pb::MemoryOpClass,
        #[source]
        source: MemoryRouteError,
    },
    #[error(
        "global memory record at pc {pc:#x} on architecture {arch} has unknown static opcode head {opcode:?}"
    )]
    UnknownGlobalOpcode {
        arch: ArchId,
        pc: u32,
        opcode: String,
    },
    #[error(
        "local memory record at pc {pc:#x} on architecture {arch} has static opcode head {opcode:?}; expected an admitted LDL or STL role"
    )]
    UnknownLocalOpcode {
        arch: ArchId,
        pc: u32,
        opcode: String,
    },
    #[error("no issue schedule for function_unique_id {function_unique_id}")]
    ScheduleMissing { function_unique_id: i32 },
    #[error(
        "admitted launch ({device_id}, {stream_id}, {kernel_id}) is absent from its retained trace snapshot"
    )]
    #[cfg(feature = "campaign")]
    AdmittedLaunchMissing {
        device_id: i64,
        stream_id: i64,
        kernel_id: i32,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix} carries function_unique_id {found}, launch \
         kernel is {expected} — multi-function replay is not supported in v0"
    )]
    MultiFunctionUnsupported {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        expected: i32,
        found: i32,
    },
    #[error("{path}: no instruction record carries sm_id — v0 replay requires traced placement")]
    SmIdMissing { path: PathBuf },
    #[error(
        "{path}: sm_id changes within the block ({first} then {second}) — blocks never migrate"
    )]
    SmIdInconsistent {
        path: PathBuf,
        first: i32,
        second: i32,
    },
    #[error("{path}: warp {warp_id} warpid_sm changes within the warp ({first} then {second})")]
    WarpSlotInconsistent {
        path: PathBuf,
        warp_id: i32,
        first: i32,
        second: i32,
    },
    #[error(
        "{path}: warp {warp_id} has no warpid_sm on any record — v0 replay requires the traced \
         physical warp slot (sub-core assignment), like sm_id"
    )]
    WarpSlotMissing { path: PathBuf, warp_id: i32 },
    #[error("kernel launch block {block:?} overflows the u64 thread-count domain")]
    BlockSizeOverflow { block: (u32, u32, u32) },
    #[error(
        "{path}: threadblock warp roster has {actual} entries, expected {expected} from blockDim"
    )]
    WarpRosterCardinalityMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error(
        "{path}: threadblock warp id {warp_id} is outside the expected block-local range 0..{expected_warps}"
    )]
    WarpIdOutOfRange {
        path: PathBuf,
        warp_id: i32,
        expected_warps: u64,
    },
    #[error("{path}: threadblock warp map key {map_key} disagrees with embedded warp id {warp_id}")]
    WarpIdentityMismatch {
        path: PathBuf,
        map_key: i32,
        warp_id: i32,
    },
    #[error("{path}: warp {warp_id} record {record_ix} has no warpid_tb on the wire")]
    InstructionWarpIdMissing {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix} carries warpid_tb {record_warp_id}, which disagrees with its enclosing warp"
    )]
    InstructionWarpIdMismatch {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        record_warp_id: i32,
    },
    #[error("architecture {arch} has no admitted physical-warp-slot capacity: {source}")]
    WarpCapacityUnavailable {
        arch: ArchId,
        #[source]
        source: sim_arch::CapabilityError,
    },
    #[error(
        "{path}: warp {warp_id} physical slot {warp_slot} is outside architecture {arch} range 0..{max_resident_warps}"
    )]
    WarpSlotOutOfRange {
        path: PathBuf,
        warp_id: i32,
        warp_slot: i32,
        arch: ArchId,
        max_resident_warps: u32,
    },
    #[error("{path}: physical warp slot {warp_slot} is assigned to multiple warps in one CTA")]
    DuplicateWarpSlot { path: PathBuf, warp_slot: i32 },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: {sectors} unique sectors in one record \
         overflows the u16 arena index"
    )]
    SectorCountOverflow {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        sectors: usize,
    },
    #[error("{path}: warp {warp_id} record {record_ix}: sector expansion failed: {source}")]
    SectorExpansion {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        #[source]
        source: SectorExpansionError,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: effective payload-byte projection failed: {source}"
    )]
    EffectiveDataBytes {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        #[source]
        source: EffectiveDataBytesError,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: async-copy source/payload mismatch: source_active={source_active}, sectors={sectors}, payload_bytes={payload_bytes}"
    )]
    AsyncCopyPayloadMismatch {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        source_active: bool,
        sectors: usize,
        payload_bytes: u64,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: async-copy cohort projection {field} mismatch: expected {expected}, got {got}"
    )]
    AsyncCopyCohortProjectionMismatch {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        field: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: async-copy cohort replay arena cannot reserve {additional} additional {buffer} entries"
    )]
    AsyncCopyCohortCapacityUnavailable {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        buffer: &'static str,
        additional: usize,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: async-copy completion work {completion_work} exceeds compact replay-token maximum {max_completion_work}"
    )]
    AsyncCopyCompletionWorkOutOfRange {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        completion_work: u64,
        max_completion_work: u32,
    },
    #[error("{path}: warp {warp_id} record {record_ix}: shared-bank projection failed: {source}")]
    SharedBankProjection {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        #[source]
        source: SharedBankProjectionError,
    },
    #[error(
        "{path}: warp {warp_id} record {record_ix}: record count overflows u64 at {accumulated}"
    )]
    RecordCountOverflow {
        path: PathBuf,
        warp_id: i32,
        record_ix: usize,
        accumulated: u64,
    },
    #[error("collection '{collection}' length {len} does not fit u64")]
    CollectionLengthOutOfRange {
        collection: &'static str,
        len: usize,
    },
    #[error("SM {sm_id} instruction aggregation {accumulated} + {addend} overflows u64")]
    InstructionCountOverflow {
        sm_id: i32,
        accumulated: u64,
        addend: u64,
    },
    #[error("duplicate block launch index {index} across threadblock files")]
    DuplicateLaunchIndex { index: i64 },
    #[error(
        "instrumentation_order token {order} is duplicated within one launch (SM {sm_id}, block {launch_index}, warp slot {warp_slot}, instruction {instruction})"
    )]
    InstrumentationOrderDuplicate {
        order: u64,
        sm_id: i32,
        launch_index: i64,
        warp_slot: i32,
        instruction: usize,
    },
    #[error("instrumentation_order is not dense and 1-based: expected token {expected}, got {got}")]
    InstrumentationOrderNotDense { expected: u64, got: u64 },
    #[error(
        "selected asynchronous-copy record lacks instrumentation_order (SM {sm_id}, block {launch_index}, warp slot {warp_slot}, instruction {instruction}, pc {pc:#x})"
    )]
    InstrumentationOrderMissing {
        sm_id: i32,
        launch_index: i64,
        warp_slot: i32,
        instruction: usize,
        pc: u32,
    },
    #[error(
        "non-selected record carries instrumentation_order={order} (SM {sm_id}, block {launch_index}, warp slot {warp_slot}, instruction {instruction}, pc {pc:#x})"
    )]
    InstrumentationOrderUnexpected {
        order: u64,
        sm_id: i32,
        launch_index: i64,
        warp_slot: i32,
        instruction: usize,
        pc: u32,
    },
    #[error(
        "shared-system unmodeled-tail aggregate mismatch: system={system:?}, per_sm={per_sm:?}"
    )]
    SystemTailMismatch {
        system: TailFlags,
        per_sm: TailFlags,
    },
    #[error(
        "issued {issued} != records {records} — replay invariant broken (every traced record \
         must issue exactly once)"
    )]
    CountMismatch { issued: u64, records: u64 },
    #[error("kernel launch grid {grid:?} overflows the u64 block-count domain")]
    GridSizeOverflow { grid: (u32, u32, u32) },
    #[error("incomplete trace: {seen}/{grid_total} launch blocks present")]
    IncompleteTrace { seen: u64, grid_total: u64 },
    #[error("core-only diagnostic analysis failed: {detail}")]
    FastFrontier { detail: String },
    #[cfg(feature = "campaign")]
    #[error(
        "trace_sampling sidecar original_grid {sidecar:?} != launch gridDim {launch:?} \
         (a subset must NEVER rewrite gridDim)"
    )]
    GridDimRewritten {
        launch: (u32, u32, u32),
        sidecar: (u32, u32, u32),
    },
    #[cfg(feature = "campaign")]
    #[error("records_seen {records_seen} != trace_sampling sidecar records.captured {captured}")]
    RecordsCapturedMismatch { records_seen: u64, captured: u64 },
    #[cfg(feature = "campaign")]
    #[error(
        "trace_sampling sidecar selected block membership does not match replayed threadblocks \
         ({seen} seen, {declared} declared)"
    )]
    SelectedBlockMembershipMismatch { seen: u64, declared: u64 },
    #[cfg(feature = "campaign")]
    #[error("trace_sampling sidecar kernel {sidecar:?} != launch kernel {launch:?}")]
    SamplingKernelMismatch { launch: String, sidecar: String },
    #[cfg(feature = "campaign")]
    #[error("kernel launch block_dim is absent or degenerate")]
    BlockDimMissing,
    #[error("kernel launch cluster_dim is partially populated or has a negative axis: {cluster:?}")]
    ClusterDimInvalid { cluster: (i32, i32, i32) },
    #[error(
        "kernel launch cluster geometry {cluster:?} is not modeled for architecture {arch}; only (1, 1, 1) is admissible"
    )]
    ClusterGeometryUnsupported {
        arch: ArchId,
        cluster: (u32, u32, u32),
    },
    #[cfg(feature = "campaign")]
    #[error("trace_sampling sidecar original_block {sidecar:?} != launch blockDim {launch:?}")]
    BlockDimRewritten {
        launch: (u32, u32, u32),
        sidecar: (u32, u32, u32),
    },
}

/// Extract distinct 32-byte constant-bank sectors from one static SASS
/// instruction. Constant references are retained as source metadata only;
/// malformed/non-numeric operands are ignored rather than guessed.
pub(crate) fn static_constant_sectors(opcode_text: &str) -> Box<[u64]> {
    pub(crate) fn literal(bytes: &[u8]) -> Option<u64> {
        let text = std::str::from_utf8(bytes).ok()?.trim();
        if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
            u64::from_str_radix(hex, 16).ok()
        } else {
            text.parse().ok()
        }
    }

    let bytes = opcode_text.as_bytes();
    let mut sectors = BTreeSet::new();
    let mut cursor = 0usize;
    while cursor + 3 < bytes.len() {
        let is_constant = bytes[cursor] == b'c'
            && (bytes[cursor + 1] == b'['
                || (bytes[cursor + 1] == b'x'
                    && cursor + 2 < bytes.len()
                    && bytes[cursor + 2] == b'['));
        if !is_constant {
            cursor += 1;
            continue;
        }
        let first_open = if bytes[cursor + 1] == b'[' {
            cursor + 1
        } else {
            cursor + 2
        };
        let Some(first_close) = bytes[first_open + 1..]
            .iter()
            .position(|&value| value == b']')
            .map(|offset| first_open + 1 + offset)
        else {
            cursor += 1;
            continue;
        };
        let second_open = first_close + 1;
        if second_open >= bytes.len() || bytes[second_open] != b'[' {
            cursor = first_close + 1;
            continue;
        }
        let Some(second_close) = bytes[second_open + 1..]
            .iter()
            .position(|&value| value == b']')
            .map(|offset| second_open + 1 + offset)
        else {
            cursor = second_open + 1;
            continue;
        };
        if let (Some(bank), Some(offset)) = (
            literal(&bytes[first_open + 1..first_close]),
            literal(&bytes[second_open + 1..second_close]),
        ) {
            if let Some(address) = bank
                .checked_mul(65_536)
                .and_then(|base| base.checked_add(offset))
            {
                sectors.insert(address / 32);
            }
        }
        cursor = second_close + 1;
    }
    sectors.into_iter().collect()
}

pub(crate) fn require_system_tail_consistency(
    system: TailFlags,
    per_sm: TailFlags,
) -> Result<(), ReplayError> {
    if system != per_sm {
        return Err(ReplayError::SystemTailMismatch { system, per_sm });
    }
    Ok(())
}

pub(crate) fn classify_kernel_result(
    coverage: ExtractedCoverage,
    configuration: ReplayConfiguration,
    replay: FoldedKernelReplay,
) -> KernelReplayOutcome {
    match coverage {
        ExtractedCoverage::Full(coverage) => KernelReplayOutcome::Full(FullKernelReplayResult {
            coverage,
            configuration,
            replay,
        }),
        #[cfg(feature = "campaign")]
        ExtractedCoverage::Sampled(coverage) => {
            let FoldedKernelReplay {
                arch,
                instructions_issued,
                records_seen,
                per_sm: _,
                diagnostic_last_issue_cycles: _,
                last_processed_system_event_time: _,
                terminal_modeled_event_time: _,
                sm_active_terminal_event_time: _,
                l1,
                l2,
                l2_by_source,
                dram,
                front_end,
                trace_metrics: _,
                unmodeled_tails,
            } = replay;
            KernelReplayOutcome::SampledObserved(SampledKernelReplayResult {
                coverage,
                configuration,
                counters: SampledObservedCounters {
                    arch,
                    instructions_issued,
                    records_seen,
                    l1,
                    l2,
                    l2_by_source,
                    dram,
                    front_end,
                    unmodeled_tails,
                },
            })
        }
    }
}

pub(crate) fn lookup_schedule<'a>(
    schedules: &'a BTreeMap<i32, FunctionSchedule>,
    loc: &KernelLoc<'_>,
) -> Result<&'a FunctionSchedule, ReplayError> {
    let expected_fuid = loc.kernel().function_unique_id;
    schedules
        .get(&expected_fuid)
        .ok_or(ReplayError::ScheduleMissing {
            function_unique_id: expected_fuid,
        })
}

/// Memory-hierarchy mode for one kernel replay — the ONE typed selector
/// every harness goes through (a pair of optional configs would let a
/// plumbing slip degrade a cell to the wrong mode silently).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayMemMode {
    /// No cache model (flat global latency).
    Flat,
    /// Per-SM L1s with constant fill latency below them (the pre-L2
    /// lockstep/ablation surface).
    L1Const(L1Config),
    /// Per-SM L1s over ONE shared L2 (the full hierarchy): L1 misses
    /// priced L2-hit vs DRAM by the shared state through the event
    /// system.
    L1SharedL2 {
        l1: L1Config,
        l2: L2Config,
        dram: DramBw,
        coherence: Option<CoherenceConfig>,
    },
}

/// Ordered same-device/same-stream replay state. Only the shared L2 model
/// survives a successful launch boundary. Per-SM L1, MSHR, scheduler,
/// dependence, async-copy, pipeline, CTA, event-queue, and DRAM-server state is
/// rebuilt by the ordinary one-launch system path on every call.
pub struct StreamL2ReplaySession {
    pub(crate) trace_root: PathBuf,
    pub(crate) trace_lineage: TraceLineage,
    pub(crate) configuration: ReplayConfiguration,
    pub(crate) placement: PlacementConfig,
    pub(crate) device_id: i64,
    pub(crate) stream_id: i64,
    pub(crate) expected_launches: Vec<OrderedSessionLaunch>,
    pub(crate) completed_launches: usize,
    pub(crate) l2: L2Model,
    pub(crate) cumulative_reported_l2: L2Counters,
}

pub(crate) fn require_replay_arch(
    expected: ArchId,
    got: ArchId,
    component: &'static str,
) -> Result<(), ReplayError> {
    if got != expected {
        return Err(ReplayError::ArchitectureMismatch {
            component,
            expected,
            got,
        });
    }
    Ok(())
}

pub(crate) fn require_trace_architecture(
    td: &TraceDir,
    expected: ArchId,
) -> Result<(), ReplayError> {
    let binary_version = td.trace().binary_version;
    let numeric = u32::try_from(binary_version)
        .map_err(|_| ReplayError::TraceArchitectureVersionOutOfRange { binary_version })?;
    let traced =
        ArchId::from_numeric(numeric).map_err(|source| ReplayError::TraceArchitectureUnknown {
            binary_version: numeric,
            source,
        })?;
    require_replay_arch(expected, traced, "trace binary_version")
}

pub(crate) fn validate_replay_mode_architecture(
    arch: ArchId,
    mode: ReplayMemMode,
) -> Result<(), ReplayError> {
    match mode {
        ReplayMemMode::Flat => {}
        ReplayMemMode::L1Const(l1) => {
            require_replay_arch(arch, l1.arch, "L1 configuration")?;
        }
        ReplayMemMode::L1SharedL2 {
            l1,
            l2,
            dram,
            coherence: _,
        } => {
            require_replay_arch(arch, l1.arch, "L1 configuration")?;
            require_replay_arch(arch, l2.arch, "L2 configuration")?;
            require_replay_arch(arch, dram.arch(), "DRAM configuration")?;
        }
    }
    Ok(())
}

pub(crate) fn async_copy_destination_fallback_sms(
    payload_contract: MemoryPayloadContract,
    per_sm_blocks: &BTreeMap<i32, Vec<BlockReplay>>,
) -> BTreeSet<i32> {
    per_sm_blocks
        .iter()
        .filter_map(|(&sm, blocks)| {
            blocks
                .iter()
                .flat_map(|block| &block.warps)
                .flat_map(|warp| &warp.insts)
                .any(|inst| {
                    matches!(inst.kind, AccessKind::GlobalAsyncCopy { .. })
                        && (payload_contract
                            != MemoryPayloadContract::AddressCarryingLdgstsDestinationV3
                            || matches!(
                                inst.kind,
                                AccessKind::GlobalAsyncCopy {
                                    l1_policy: LdgstsL1Policy::BypassL1,
                                    ..
                                }
                            ))
                })
                .then_some(sm)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn replay_kernel_with_schedule_and_l2_and_candidates_with_payload_events(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    retained_l2: Option<&mut L2Model>,
    memory_service_candidates: &[MemoryServiceCandidate],
    memory_admission_candidates: &[MemoryAdmissionCandidate],
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
    payload_events: &[L1PayloadAdmissionEvent],
) -> Result<KernelReplayOutcome, ReplayError> {
    if (!memory_service_candidates.is_empty()
        || !memory_admission_candidates.is_empty()
        || memory_admission_feedback.is_some())
        && schedule.arch != ArchId::Sm86
    {
        return Err(ReplayError::ArchitectureMismatch {
            component: "SM86 memory candidate lane",
            expected: ArchId::Sm86,
            got: schedule.arch,
        });
    }
    if !payload_events.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::PayloadAdmissionRequiresSharedL2);
    }
    if !memory_service_candidates.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::MemoryServiceCandidatesRequireSharedL2);
    }
    if !memory_admission_candidates.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. })
    {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireSharedL2);
    }
    if memory_admission_feedback.is_some() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireSharedL2);
    }
    validate_memory_service_candidates(memory_service_candidates).map_err(IssueError::from)?;
    validate_memory_admission_candidates(memory_admission_candidates).map_err(IssueError::from)?;
    if let Some(candidate) = memory_admission_feedback {
        candidate.validate().map_err(IssueError::from)?;
    }
    let configuration =
        validate_replay_architecture(schedule, params, mode, memory_admission_feedback)?;
    let placement = PlacementConfig::from_arch_card(arch_card, schedule.arch)?;
    require_trace_architecture(td, schedule.arch)?;
    // Ordinary launches can use the hierarchy classifier in flat mode.
    // Mixed async/TMA launches use the alpha compact classifier to preserve
    // opaque sidecar payloads without borrowing the SM86 completion protocol;
    // that classifier also retains all observed global/local sector spans.
    let alpha_flat_sector_decode = matches!(mode, ReplayMemMode::Flat)
        && params.resource_source() == IssueResourceSource::Sm120Alpha
        && !schedule.by_pc.values().any(|slot| {
            slot.ldgsts_l1_policy().is_some()
                || matches!(
                    slot.op_class(),
                    OpClass::AsyncCopyCommit | OpClass::AsyncCopyWait
                )
                || matches!(
                    opcode_head(slot.opcode_text()),
                    "UTMALDG" | "UTMASTG" | "UBLKCP"
                )
        });
    let decode_sectors = !matches!(mode, ReplayMemMode::Flat) || alpha_flat_sector_decode;
    let ExtractedKernel {
        per_sm_blocks,
        per_sm_instruction_read_sectors,
        records_seen,
        coverage,
        shared_memory_residency,
        front_end,
        mut trace_metrics,
    } = extract_per_sm_blocks(
        td,
        loc,
        schedule,
        placement,
        decode_sectors,
        match mode {
            ReplayMemMode::L1SharedL2 { l2, .. } => Some(l2),
            _ => None,
        },
    )?;
    // Keep the producer/consumer contract explicit for the report-only SM86
    // write-tag observer. The emitter must not consume `write_lookup_hits`
    // when the issue model did not instantiate the observer for this launch.
    trace_metrics.l1_write_lookup_observer_active =
        stateful_write_lookup_observer_enabled(schedule);
    let instrumentation_order = observe_async_instrumentation_order(&per_sm_blocks, schedule)?;
    #[cfg(feature = "campaign")]
    if matches!(
        mode,
        ReplayMemMode::L1SharedL2 {
            dram: DramBw::CardedWriteback { .. },
            ..
        }
    ) && matches!(&coverage, ExtractedCoverage::Sampled(_))
    {
        return Err(ReplayError::WritebackRequiresFullCoverage);
    }
    if let ReplayMemMode::L1SharedL2 {
        coherence: Some(_), ..
    } = mode
    {
        let full_coverage = matches!(&coverage, ExtractedCoverage::Full(_));
        let active_masks_missing = trace_metrics.instructions.active_mask_missing_instructions;
        let predicate_masks_missing = trace_metrics
            .instructions
            .predicate_mask_missing_instructions;
        if !full_coverage
            || !trace_metrics.sectors_decoded
            || trace_metrics.coherence.unknown_sm_or_line_records != 0
            || active_masks_missing != 0
            || predicate_masks_missing != 0
        {
            return Err(ReplayError::CoherenceTopologyIncomplete {
                full_coverage,
                decoded: trace_metrics.sectors_decoded,
                unknown_records: trace_metrics.coherence.unknown_sm_or_line_records,
                active_masks_missing,
                predicate_masks_missing,
            });
        }
    }
    #[cfg(feature = "campaign")]
    if !memory_service_candidates.is_empty() && matches!(&coverage, ExtractedCoverage::Sampled(_)) {
        return Err(ReplayError::MemoryServiceCandidatesRequireFullCoverage);
    }
    #[cfg(feature = "campaign")]
    if !memory_admission_candidates.is_empty() && matches!(&coverage, ExtractedCoverage::Sampled(_))
    {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireFullCoverage);
    }
    #[cfg(feature = "campaign")]
    if memory_admission_feedback.is_some() && matches!(&coverage, ExtractedCoverage::Sampled(_)) {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireFullCoverage);
    }
    let async_copy_destination_fallback_sms =
        async_copy_destination_fallback_sms(td.memory_payload_contract(), &per_sm_blocks);
    let mut replay = match mode {
        ReplayMemMode::Flat | ReplayMemMode::L1Const(_) => {
            let l1_config = match mode {
                ReplayMemMode::L1Const(cfg) => Some(cfg),
                _ => None,
            };
            let mut per_sm = BTreeMap::new();
            for (sm, blocks) in &per_sm_blocks {
                let mut result = simulate_sm_with_residency(
                    blocks,
                    schedule,
                    params,
                    shared_memory_residency,
                    l1_config,
                )?;
                result.activity.front_end_instruction_read_sectors =
                    per_sm_instruction_read_sectors
                        .get(sm)
                        .copied()
                        .unwrap_or(0);
                if async_copy_destination_fallback_sms.contains(sm) {
                    result.unmodeled_tails.ldgsts_dst_shared_timing = true;
                }
                if let Some(observation) = instrumentation_order.get(sm) {
                    result.readiness.async_instrumentation_order = Some(Box::new(*observation));
                }
                per_sm.insert(*sm, result);
            }
            for result in per_sm.values() {
                append_bounded_async_decision_snapshots(
                    &mut trace_metrics,
                    &result.readiness.async_decision_snapshots,
                    result.readiness.async_decision_snapshots_dropped,
                )?;
            }
            fold_kernel_result(
                schedule.arch,
                per_sm,
                records_seen,
                l1_config.is_some(),
                None,
                None,
            )?
        }
        ReplayMemMode::L1SharedL2 {
            l1,
            l2,
            dram,
            coherence,
        } => {
            let mut fresh_l2;
            let l2_model = match retained_l2 {
                Some(model) => {
                    if *model.config() != l2 {
                        return Err(ReplayError::RetainedL2ConfigurationMismatch {
                            expected: l2,
                            got: *model.config(),
                        });
                    }
                    model
                }
                None => {
                    fresh_l2 = L2Model::new(l2)?;
                    &mut fresh_l2
                }
            };
            let pending_sectors = l2_model.pending_sector_count();
            if pending_sectors != 0 {
                return Err(ReplayError::RetainedL2NotDrained { pending_sectors });
            }
            let counters_before = *l2_model.counters()?;
            let mut sys = if payload_events.is_empty() {
                simulate_kernel_shared_with_residency_and_candidates_with_coherence(
                    &per_sm_blocks,
                    schedule,
                    params,
                    shared_memory_residency,
                    l1,
                    l2_model,
                    dram,
                    memory_service_candidates,
                    memory_admission_candidates,
                    memory_admission_feedback,
                    coherence,
                )?
            } else {
                simulate_kernel_shared_with_residency_and_candidates_with_coherence_and_payload_events(
                    &per_sm_blocks,
                    schedule,
                    params,
                    shared_memory_residency,
                    l1,
                    l2_model,
                    dram,
                    memory_service_candidates,
                    memory_admission_candidates,
                    memory_admission_feedback,
                    coherence,
                    payload_events,
                )?
            };
            let pending_sectors = l2_model.pending_sector_count();
            if pending_sectors != 0 {
                return Err(ReplayError::RetainedL2NotDrained { pending_sectors });
            }
            let launch_l2 = sys.l2.checked_delta_since(&counters_before)?;
            trace_metrics.observe_l2_system_locality_summary(
                launch_l2.lookup_requests,
                launch_l2.lookup_hits,
                sys.l2_by_source.read_pending_merge_hits,
                sys.l2_by_source.store_lookup_requests,
            );
            for sm in &async_copy_destination_fallback_sms {
                if let Some(result) = sys.per_sm.get_mut(sm) {
                    result.unmodeled_tails.ldgsts_dst_shared_timing = true;
                }
            }
            for (sm, sectors) in &per_sm_instruction_read_sectors {
                if let Some(result) = sys.per_sm.get_mut(sm) {
                    result.activity.front_end_instruction_read_sectors = *sectors;
                }
            }
            for (sm, observation) in &instrumentation_order {
                let result = sys.per_sm.get_mut(sm).unwrap_or_else(|| {
                    unreachable!("order observation SM escaped the simulated SM roster")
                });
                result.readiness.async_instrumentation_order = Some(Box::new(*observation));
            }
            if !async_copy_destination_fallback_sms.is_empty() {
                sys.unmodeled_tails.ldgsts_dst_shared_timing = true;
            }
            for result in sys.per_sm.values() {
                append_bounded_async_decision_snapshots(
                    &mut trace_metrics,
                    &result.readiness.async_decision_snapshots,
                    result.readiness.async_decision_snapshots_dropped,
                )?;
            }
            require_replay_arch(schedule.arch, sys.arch, "system result")?;
            let system_unmodeled_tails = sys.unmodeled_tails;
            let replay = fold_kernel_result_with_sm_active_time(
                schedule.arch,
                sys.per_sm,
                records_seen,
                true,
                sys.terminal_event_time,
                sys.sm_active_terminal_event_time,
                Some((launch_l2, sys.l2_by_source, sys.dram)),
            )?;
            require_system_tail_consistency(system_unmodeled_tails, replay.unmodeled_tails)?;
            replay
        }
    };
    for result in replay.per_sm.values() {
        if let Some(audit) = result.l1_merge_audit {
            let (global_hits, global_lookups) = result
                .l1
                .map(|counters| {
                    let origin = counters.origin(sim_mem::MemoryOrigin::Global);
                    (origin.hits, origin.lookup_sectors())
                })
                .unwrap_or((0, 0));
            trace_metrics.observe_l1_merge_audit(
                audit.same_request_merges,
                audit.older_pending_merges,
                audit.already_requested_merges,
                audit.unrequested_pending_merges,
                audit.ordinary_same_request_merges,
                audit.ordinary_older_pending_merges,
                audit.async_same_request_merges,
                audit.async_older_pending_merges,
                audit.ordinary_global_older_pending_merges,
                audit.constant_global_older_pending_merges,
                audit.ordinary_global_same_request_merges,
                audit.constant_global_same_request_merges,
                audit.async_global_same_request_merges,
                audit.ordinary_global_hits,
                audit.async_global_hits,
                audit.ordinary_global_older_sector_pending_merges,
                audit.constant_global_older_sector_pending_merges,
                audit.async_global_older_sector_pending_merges,
                audit.unrequested_subline_miss_sectors,
                audit.wavefront_mask_lookup_hits,
                audit.wavefront_mask_lookup_misses,
                audit.constant_global_hits,
                audit.constant_global_requests,
                global_hits,
                global_lookups,
                audit.request_masks,
            )?;
        }
    }
    trace_metrics.finalize_reuse_observer();
    trace_metrics.finalize_shared_load_observer();
    replay.front_end = front_end;
    replay.trace_metrics = trace_metrics;
    // A drained hypothetical service queue is not a completion relation for
    // native stores/atomics. Keep the system's unknown tails: one completion
    // timestamp or the absence of traced cross-SM sharing cannot prove that
    // all write/visibility obligations were represented.
    Ok(classify_kernel_result(coverage, configuration, replay))
}

pub(crate) fn validate_replay_access_kind(
    meta: MemoryAccessMeta,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<(), ReplayError> {
    let Some(got) = meta.access_kind else {
        return Ok(()); // additive legacy field: absence remains admissible
    };
    let (valid, expected) = match meta.op_class {
        pb::MemoryOpClass::Global | pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => (
            matches!(
                got,
                pb::MemoryOpAccessKind::Load | pb::MemoryOpAccessKind::Store
            ),
            "LOAD or STORE",
        ),
        pb::MemoryOpClass::Atomic => (got == pb::MemoryOpAccessKind::Atomic, "ATOMIC"),
        pb::MemoryOpClass::AsyncCopy => (got == pb::MemoryOpAccessKind::AsyncCopy, "ASYNC_COPY"),
        // These routes fail as unsupported before replay. Do not invent a
        // direction contract for a class with no modeled path.
        pb::MemoryOpClass::Unspecified
        | pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => return Ok(()),
    };
    if valid {
        return Ok(());
    }
    Err(ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class: meta.op_class,
        source: MemoryRouteError::AccessKindInvalid { got, expected },
    })
}

pub(crate) fn classify_flat_global_sectors(
    role: MemRole,
    warp: &ValidatedWarp,
    memory: &ValidatedMemoryInstruction,
    sectors: &mut Vec<i64>,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let sector_scratch = match warp.sector_plan(memory) {
        ValidatedSectorPlan::Sectors(sectors) => sectors,
        ValidatedSectorPlan::Error { error, .. } => {
            return Err(deferred_sector_error(error, path, warp_id, record_ix));
        }
    };
    append_staged_global_sectors(role, sectors, sector_scratch, path, warp_id, record_ix)
}

pub(crate) fn compact_memory_route_error(error: &ValidatedMemoryRouteError) -> MemoryRouteError {
    match *error {
        ValidatedMemoryRouteError::GlobalOperandCount { got } => {
            MemoryRouteError::GlobalOperandCount { got }
        }
        ValidatedMemoryRouteError::GlobalSpaceInvalid { operand, space } => {
            MemoryRouteError::GlobalSpaceInvalid { operand, space }
        }
        ValidatedMemoryRouteError::SharedLocalOperandCount { got } => {
            MemoryRouteError::SharedLocalOperandCount { got }
        }
        ValidatedMemoryRouteError::SharedLocalSpaceInvalid { operand, space } => {
            MemoryRouteError::SharedLocalSpaceInvalid { operand, space }
        }
        ValidatedMemoryRouteError::AtomicOperandCount { got } => {
            MemoryRouteError::AtomicOperandCount { got }
        }
        ValidatedMemoryRouteError::AtomicSpaceMissing { operand } => {
            MemoryRouteError::AtomicSpaceMissing { operand }
        }
        ValidatedMemoryRouteError::AtomicSpaceUnspecified { operand } => {
            MemoryRouteError::AtomicSpaceUnspecified { operand }
        }
        ValidatedMemoryRouteError::AtomicSpacesConflict => MemoryRouteError::AtomicSpacesConflict,
        ValidatedMemoryRouteError::AsyncCopySourceCount { got } => {
            MemoryRouteError::AsyncCopySourceCount { got }
        }
        ValidatedMemoryRouteError::AsyncCopySourceSpaceMissing => {
            MemoryRouteError::AsyncCopySourceSpaceMissing
        }
        ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid { space } => {
            MemoryRouteError::AsyncCopySourceSpaceInvalid { space }
        }
        ValidatedMemoryRouteError::AsyncCopyOperandCount { got, expected } => {
            MemoryRouteError::AsyncCopyOperandCount { got, expected }
        }
        ValidatedMemoryRouteError::AsyncCopyDestinationSpaceMissing => {
            MemoryRouteError::AsyncCopyDestinationSpaceMissing
        }
        ValidatedMemoryRouteError::AsyncCopyDestinationSpaceInvalid { space } => {
            MemoryRouteError::AsyncCopyDestinationSpaceInvalid { space }
        }
    }
}

pub(crate) fn deferred_sector_error(
    error: &ValidatedSectorError,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> ReplayError {
    if let ValidatedSectorError::SectorCountOverflow { sectors } = *error {
        return ReplayError::SectorCountOverflow {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            sectors,
        };
    }
    let source = match *error {
        ValidatedSectorError::DataWidthInvalid { data_width_bytes } => {
            SectorExpansionError::DataWidthInvalid { data_width_bytes }
        }
        ValidatedSectorError::AddressOverflow {
            lane,
            base,
            data_width_bytes,
        } => SectorExpansionError::AddressOverflow {
            lane,
            base,
            data_width_bytes,
        },
        ValidatedSectorError::UniqueSectorCountOutOfRange { sectors } => {
            SectorExpansionError::UniqueSectorCountOutOfRange { sectors }
        }
        ValidatedSectorError::CardinalityMismatch { expected, produced } => {
            SectorExpansionError::CardinalityMismatch { expected, produced }
        }
        ValidatedSectorError::CapacityUnavailable { buffer, additional } => {
            SectorExpansionError::CapacityUnavailable { buffer, additional }
        }
        ValidatedSectorError::SectorCountOverflow { .. } => unreachable!("handled above"),
    };
    ReplayError::SectorExpansion {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        source,
    }
}

pub(crate) fn checked_staged_global_kind(
    role: MemRole,
    start: usize,
    count: usize,
    l2_requests: u16,
) -> Option<AccessKind> {
    let len = u16::try_from(count).ok()?;
    start.checked_add(count)?;
    Some(match role {
        MemRole::EligibleLoad(token) => AccessKind::GlobalEligible {
            off: start,
            len,
            token,
        },
        MemRole::BypassLoad(token) => AccessKind::GlobalBypass {
            off: start,
            len,
            token,
            l2_requests: if token == CacheOpToken::Atom {
                l2_requests
            } else {
                0
            },
        },
        MemRole::Store => AccessKind::GlobalStore { off: start, len },
        MemRole::Reduction => AccessKind::GlobalReduction {
            off: start,
            len,
            l2_requests,
        },
        MemRole::None => AccessKind::GlobalUntagged { off: start, len },
    })
}

pub(crate) fn project_async_copy_completion_work(
    arch: ArchId,
    l1_policy: LdgstsL1Policy,
    payload_bytes: u64,
    destination_topology: Option<LdgstsDestinationTopology>,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<u32, ReplayError> {
    // The flat SM120 alpha path returns before this helper and charges an
    // explicit global-demand approximation.  Any caller that reaches the
    // compact async completion machinery must therefore be an architecture
    // with an admitted LDGSTS protocol; never map SM120 to SM86 here.
    if arch == ArchId::Sm120 {
        return Err(ReplayError::UnsupportedMemoryPath {
            arch,
            class: pb::MemoryOpClass::AsyncCopy,
        });
    }
    let timing_arch = arch;
    let throughput = timing_arch
        .ldgsts_completion_throughput()
        .require("ldgsts_completion_throughput")
        .map_err(IssueError::from)?;
    let cache_all_rate = throughput.cache_all_payload_bytes_per_cycle;
    let bypass_rate = throughput.bypass_payload_bytes_per_cycle;
    let units_per_cycle = cache_all_rate.max(bypass_rate);
    if cache_all_rate == 0
        || bypass_rate == 0
        || units_per_cycle % cache_all_rate != 0
        || units_per_cycle % bypass_rate != 0
    {
        return Err(IssueError::InvalidLdgstsCompletionThroughput {
            arch,
            cache_all_bytes_per_cycle: cache_all_rate,
            bypass_bytes_per_cycle: bypass_rate,
        }
        .into());
    }
    let route_rate = match l1_policy {
        LdgstsL1Policy::CacheAtAllLevels => cache_all_rate,
        LdgstsL1Policy::BypassL1 => bypass_rate,
    };
    let units_per_byte = units_per_cycle / route_rate;
    let payload_work = payload_bytes.checked_mul(units_per_byte).ok_or(
        IssueError::TimeMultiplicationOverflow {
            operation: "LDGSTS payload-fallback completion work",
            lhs: payload_bytes,
            rhs: units_per_byte,
        },
    )?;
    if l1_policy == LdgstsL1Policy::BypassL1
        && let Some(topology) = destination_topology
    {
        let payload_work_compact = u32::try_from(payload_work).map_err(|_| {
            ReplayError::AsyncCopyCompletionWorkOutOfRange {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                completion_work: payload_work,
                max_completion_work: ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX,
            }
        })?;
        // A topology-only copy has executing destination lanes but no source
        // cohort. The compact observation token uses one reserved cohort slot
        // to keep the token non-inactive; the detailed sidecar remains empty.
        let source_cohorts = topology.source_cohorts.max(1);
        return encode_async_copy_payload_and_topology(
            payload_work_compact,
            source_cohorts,
            topology.destination_line_fanout_total,
            topology.cache_all_bank_wavefronts_max,
            topology.bypass_wavefronts_max,
        )
        .ok_or(ReplayError::AsyncCopyCompletionWorkOutOfRange {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            completion_work: payload_work
                .max(u64::from(topology.destination_line_fanout_total))
                .max(u64::from(topology.cache_all_bank_wavefronts_max))
                .max(u64::from(topology.bypass_wavefronts_max)),
            max_completion_work: ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX,
        });
    }
    let exact_destination = if l1_policy == LdgstsL1Policy::CacheAtAllLevels {
        match destination_topology {
            Some(topology) => {
                let wavefronts = topology.cache_all_bank_wavefronts_total;
                let timing = timing_arch
                    .ldgsts_destination_wavefront_timing()
                    .require("ldgsts_destination_wavefront_timing")
                    .map_err(IssueError::from)?;
                match timing.projection {
                    LdgstsDestinationWavefrontProjection::SourceCohorts128BytesBanks32Words4Bytes => {}
                }
                if timing.cycles_per_wavefront == 0 {
                    return Err(IssueError::InvalidLdgstsDestinationWavefrontTiming {
                        arch,
                        cycles_per_wavefront: timing.cycles_per_wavefront,
                    }
                    .into());
                }
                let destination_cycles = timing
                    .cycles_per_wavefront
                    .checked_mul(u64::from(wavefronts))
                    .ok_or(IssueError::TimeMultiplicationOverflow {
                        operation: "LDGSTS destination-wavefront completion cycles",
                        lhs: timing.cycles_per_wavefront,
                        rhs: u64::from(wavefronts),
                    })?;
                let destination_work = destination_cycles.checked_mul(units_per_cycle).ok_or(
                    IssueError::TimeMultiplicationOverflow {
                        operation: "LDGSTS destination-wavefront completion work",
                        lhs: destination_cycles,
                        rhs: units_per_cycle,
                    },
                )?;
                Some((wavefronts, destination_work))
            }
            None => None,
        }
    } else {
        // Focused bypass probes show a different source/destination coupling.
        // The compact token above retains its trace-derived topology for cold
        // diagnostics, while production timing continues to use payload work
        // until a bypass-specific per-copy mechanism is characterized.
        None
    };
    if payload_work != 0
        && let Some((_wavefronts, destination_work)) = exact_destination
        && destination_work >= payload_work
    {
        let compact = u32::try_from(destination_work).map_err(|_| {
            ReplayError::AsyncCopyCompletionWorkOutOfRange {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                completion_work: destination_work,
                max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
            }
        })?;
        return encode_async_copy_destination_work(compact).ok_or(
            ReplayError::AsyncCopyCompletionWorkOutOfRange {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                completion_work: destination_work,
                max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
            },
        );
    }
    let completion_work = exact_destination.map_or(payload_work, |(_, destination_work)| {
        payload_work.max(destination_work)
    });
    let compact = u32::try_from(completion_work).map_err(|_| {
        ReplayError::AsyncCopyCompletionWorkOutOfRange {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            completion_work,
            max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
        }
    })?;
    encode_async_copy_payload_work(compact).ok_or(ReplayError::AsyncCopyCompletionWorkOutOfRange {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        completion_work,
        max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
    })
}
