//! Replay request records, memory replies, and validation contracts; no cache mutation.

use super::*;

pub(crate) const MAX_ASYNC_DECISION_SNAPSHOTS: usize = 1_024;

/// One dynamic instruction, compacted for replay.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReplayInst {
    /// Function-relative pc — the join key into [`FunctionSchedule`].
    pub(crate) pc: u32,
    pub(crate) kind: AccessKind,
    /// Effective executing lanes for the dynamic instruction. Native traces
    /// carry both masks; legacy captures remain all-lane.
    pub(crate) lane_mask: u32,
    /// Active lanes also evaluate the guard when its outcome is false.
    pub(crate) active_lane_mask: u32,
}

/// Exact trace-derived topology of one flexible 128-byte LDGSTS source cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AsyncCopyCohortDescriptor {
    pub(crate) payload_bytes: u32,
    pub(crate) destination_lines: u32,
    pub(crate) bank_wavefronts: u32,
    pub(crate) bypass_wavefronts: u32,
}

/// Sparse arena span for one dynamic LDGSTS record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AsyncCopyCohortSpan {
    pub(crate) instruction: usize,
    pub(crate) sector_off: usize,
    pub(crate) sector_len: u16,
    pub(crate) cohort_off: usize,
    pub(crate) cohort_len: u8,
    pub(crate) membership_off: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AsyncCopyCohortView<'a> {
    pub(crate) sector_off: usize,
    pub(crate) sector_len: u16,
    pub(crate) cohorts: &'a [AsyncCopyCohortDescriptor],
    pub(crate) sector_membership_masks: &'a [u32],
}

/// Memory-model mode for one SM replay.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SmMemMode {
    /// No cache model: flat global latency.
    Flat,
    /// Per-SM L1 with constant (L2-hit) fill latency below it.
    L1Const(L1Config),
    /// Per-SM L1 over the SHARED L2: misses become requests/responses.
    L1Shared(L1Config),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AsyncDecisionSnapshot {
    pub function_unique_id: i32,
    pub pc: u32,
    pub warp_slot: i32,
    pub instruction: u64,
    pub protocol: &'static str,
    pub committed_groups_before: u64,
    pub required_prefix_len: u64,
    pub wait_threshold: u32,
    pub readiness_cycle: u64,
    pub decision_cycle: u64,
    pub issue_cycle: u64,
    /// This diagnostic is emitted only for a warp selected by the current
    /// arbitration pass.  It is not a census of blocked candidates and does
    /// not prove that the async prefix was the cause preventing another warp
    /// from issuing.
    pub observation_scope: &'static str,
    /// Explicit semantic disposition prevents consumers from treating the
    /// selected-only snapshot as the stronger same-state blocked witness that
    /// would require instrumentation before arbitration.
    pub semantics: &'static str,
    pub membership: &'static str,
    pub limiting_cause: Option<&'static str>,
}

/// Result of replaying one SM's blocks to completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmSimResult {
    /// Exact: one per traced record fed in.
    pub instructions_issued: u64,
    /// Unadjusted modeled SM clock frontier (time of last issue + 1). An
    /// explicit observation projection may add an architecture-scoped launch
    /// boundary; that boundary is not hidden in this causal timestamp. The
    /// response tail of final loads is not counted here.
    pub cycles: u64,
    /// Latest completion of an active SM-local pipeline after scheduler issue.
    /// This remains separate from `cycles`, the issue frontier.
    pub modeled_local_completion_cycles: Option<u64>,
    /// This SM's L1 counters when an L1 model was attached.
    pub l1: Option<L1Counters>,
    /// Read-only request-boundary provenance for L1 merge classification.
    pub l1_merge_audit: Option<L1MergeAudit>,
    /// Explicit capability gaps for write-class tails present in this replay.
    pub unmodeled_tails: TailFlags,
    /// Simulator-internal causal readiness diagnostics. These are not NCU
    /// sample counts and their overlapping exposure fields are not elapsed
    /// kernel cycles.
    pub readiness: ReadinessDiagnostics,
    /// Model-relative active-warp integral and per-SMSP issue observations.
    pub activity: SmActivity,
}

pub(crate) const SM120_FLAT_SERVICE_SECTOR_THRESHOLD: u64 = 512;

pub(crate) const SM120_FLAT_GLOBAL_32B_SERVICE_CYCLES_PER_SECTOR: u64 = 14;

pub(crate) const SM120_FLAT_GLOBAL_64B_SERVICE_CYCLES_PER_SECTOR: u64 = 10;

pub(crate) const SM120_FLAT_GLOBAL_128B_SERVICE_CYCLES_PER_SECTOR: u64 = 8;

pub(crate) const SM120_FLAT_SHARED_SERVICE_CYCLES_PER_WAVEFRONT: u64 = 7;

pub(crate) const SM120_FLAT_MULTI_WARP_GLOBAL_SERVICE_CYCLES_PER_SECTOR: u64 = 2;

pub(crate) const SM120_FLAT_MULTI_WARP_DENSE_EXTRA_CYCLES: u64 = 7;

pub(crate) const SM120_FLAT_MULTI_WARP_MEDIUM_EXTRA_CYCLES: u64 = 40;

pub(crate) const SM120_FLAT_SHARED_MEDIUM_EXTRA_CYCLES_PER_WAVEFRONT: u64 = 42;

pub(crate) const SM120_FLAT_ASYNC_GROUP_SERVICE_CYCLES: u64 = 1_250;

pub(crate) const SM120_FLAT_STORE_RETIRE_CYCLES_PER_SECTOR: u64 = 19;

pub(crate) const SM120_FLAT_REDUCTION_RETIRE_CYCLES_PER_SECTOR: u64 = 17;

pub(crate) fn sm120_flat_global_service_rate(opcode_text: &str) -> u64 {
    let opcode = opcode_text
        .split_ascii_whitespace()
        .next()
        .unwrap_or(opcode_text);
    if opcode.contains(".128") {
        SM120_FLAT_GLOBAL_128B_SERVICE_CYCLES_PER_SECTOR
    } else if opcode.contains(".64") {
        SM120_FLAT_GLOBAL_64B_SERVICE_CYCLES_PER_SECTOR
    } else {
        SM120_FLAT_GLOBAL_32B_SERVICE_CYCLES_PER_SECTOR
    }
}

pub(crate) fn sm120_flat_service_work(kind: AccessKind, opcode_text: &str) -> Option<(bool, u64)> {
    match kind {
        AccessKind::GlobalEligible { len, .. }
        | AccessKind::GlobalBypass { len, .. }
        | AccessKind::GlobalUntagged { len, .. }
        | AccessKind::GlobalStore { len, .. }
        | AccessKind::GlobalReduction { len, .. }
            if len != 0 =>
        {
            Some((
                true,
                u64::from(len).saturating_mul(sm120_flat_global_service_rate(opcode_text)),
            ))
        }
        AccessKind::Shared { wavefronts, .. } if wavefronts != 0 => Some((
            false,
            wavefronts.saturating_mul(SM120_FLAT_SHARED_SERVICE_CYCLES_PER_WAVEFRONT),
        )),
        _ => None,
    }
}

/// Group witness for one sector request belonging to an effective scalar
/// STORE/RED warp instruction. The system consumes these witnesses only to
/// reconstruct exact per-instruction L2 hit/miss work for opt-in admission
/// candidates; they do not affect hierarchy identity or ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmissionRequestWitness(pub(crate) NonZeroU128);

impl AdmissionRequestWitness {
    pub(crate) const VALID: u128 = 1 << 127;
}

impl AdmissionRequestWitness {
    pub(crate) const U16_MASK: u128 = u16::MAX as u128;
}

impl AdmissionRequestWitness {
    pub(crate) const U32_MASK: u128 = u32::MAX as u128;
}

impl AdmissionRequestWitness {
    pub(crate) const fn new(
        warp: u32,
        instruction: u32,
        subcore: u8,
        request_index: u16,
        request_count: u16,
    ) -> Option<Self> {
        if subcore >= SUB_CORES as u8 || request_count == 0 || request_index >= request_count {
            return None;
        }
        let packed = Self::VALID
            | warp as u128
            | ((instruction as u128) << 32)
            | ((subcore as u128) << 64)
            | ((request_index as u128) << 66)
            | ((request_count as u128) << 82);
        match NonZeroU128::new(packed) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

impl AdmissionRequestWitness {
    pub(crate) const fn warp(self) -> u32 {
        (self.0.get() & Self::U32_MASK) as u32
    }
}

impl AdmissionRequestWitness {
    pub(crate) const fn instruction(self) -> u32 {
        ((self.0.get() >> 32) & Self::U32_MASK) as u32
    }
}

impl AdmissionRequestWitness {
    pub(crate) const fn subcore(self) -> u8 {
        ((self.0.get() >> 64) & 0b11) as u8
    }
}

impl AdmissionRequestWitness {
    pub(crate) const fn request_index(self) -> u16 {
        ((self.0.get() >> 66) & Self::U16_MASK) as u16
    }
}

impl AdmissionRequestWitness {
    pub(crate) const fn request_count(self) -> u16 {
        ((self.0.get() >> 82) & Self::U16_MASK) as u16
    }
}

/// Exact event payload for one opt-in modeled STORE/RED completion. The
/// admission witness names the instruction group; the shared-system target
/// names the owning SM, so identical warp/instruction indices on different SMs
/// cannot exchange acknowledgements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryWriteAck {
    pub(crate) witness: AdmissionRequestWitness,
    pub(crate) operation: MemoryOperation,
    pub(crate) issue_at: u64,
    pub(crate) accepted_at: u64,
    pub(crate) service_done_at: u64,
    pub(crate) fence_visible_at: u64,
}

/// Static client-scope tag for the passive L2 scope observer. It never
/// changes the physical hierarchy route, cache state, DRAM, or timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum L2ClientScope {
    OrdinaryLdg,
    ConstantLdg,
    Ltc128b,
    AsyncLdgstsCache,
    AsyncLdgstsBypass,
    Local,
    Store,
    Atomic,
    Reduction,
    Unknown,
}

impl L2ClientScope {
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::OrdinaryLdg => 0,
            Self::ConstantLdg => 1,
            Self::Ltc128b => 2,
            Self::AsyncLdgstsCache => 3,
            Self::AsyncLdgstsBypass => 4,
            Self::Local => 5,
            Self::Store => 6,
            Self::Atomic => 7,
            Self::Reduction => 8,
            Self::Unknown => 9,
        }
    }
}

/// A memory request the SM emits in L1Shared mode. The system layer
/// schedules the L2 access at `issue_time` (the engine's global
/// `(time, seq)` order supplies the cross-SM interleaving) and routes the
/// response per `reply`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemRequest {
    pub issue_time: u64,
    /// Physical 32 B hierarchy sector id (`byte_address >> 5`).
    pub sector_id: i64,
    /// Address-space provenance of the anchor sector (`sector_id`). For folded
    /// line fills, [`Self::local_sector_mask`] carries exact per-sector
    /// provenance while this field anchors response-delay attribution. Neither
    /// field participates in L1/L2 tags, fill keys, response identity, or event
    /// ordering.
    pub origin: MemoryOrigin,
    /// Sectors of `sector_id`'s 128 B line carried by THIS request
    /// (bit = sector index in line; always contains `sector_id`'s own
    /// bit). More than one bit only on L1 LINE-fill requests whose
    /// emission batch merged additional sectors onto the fill — NCU
    /// counts each 32 B sector at the LTS, so a multi-sector line fill
    /// must reach the L2 as that many sector accesses, not one request
    /// (§6.4 slice 2; witness accounting lives in the progress log,
    /// never in sim sources — Rule 1). The L2 handler walks the mask
    /// in ascending sector order and prices the reply on the WORST
    /// sector outcome (the fill completes when its slowest sector
    /// arrives).
    pub sector_mask: u8,
    /// Subset of `sector_mask` attributed to Local address-space traffic.
    /// Global provenance is the complement within `sector_mask`. Keeping the
    /// split on the folded physical request preserves one worst-of-mask reply
    /// while allowing L2 hits and DRAM fetches to be accounted per origin.
    pub local_sector_mask: u8,
    pub source: L2Source,
    /// Semantic memory operation. The shared system derives the L2 hit rule
    /// and DRAM-fetch requirement from this value and rejects incoherent
    /// `(source, operation, reply)` combinations before hierarchy mutation.
    pub operation: MemoryOperation,
    /// Optional operation-class observer tag. This never changes the
    /// physical hierarchy operation, response timing, or DRAM fetch policy;
    /// it only lets static generic-store forms use the operation-qualified
    /// lookup counters without changing their established read route.
    pub(crate) observer_operation: Option<MemoryOperation>,
    pub(crate) client_scope: L2ClientScope,
    /// Static SASS marker for a regular-register constant load.  Constant
    /// loads still use the normal global allocation/cache state, but their
    /// response may cross a distinct SM-side return path on SM86.
    pub(crate) constant_load: bool,
    /// Static pressure-class marker. The broad constant marker is retained
    /// for provenance; only this narrower marker is allowed to price the
    /// measured multi-site return-service delay. Response timing is attached
    /// to each `FillWaiter`, so this request-level copy never prices a folded
    /// fill.
    pub(crate) constant_deep: bool,
    pub reply: ReplyTo,
    pub(crate) admission_witness: Option<AdmissionRequestWitness>,
}

impl MemRequest {
    /// Operation used only by operation-qualified observer counters.  The
    /// physical hierarchy continues to use `operation`; generic `ST.E` keeps
    /// its established untagged physical route while the observer sees it as
    /// a store.
    pub fn effective_observer_operation(&self) -> MemoryOperation {
        self.observer_operation.unwrap_or(self.operation)
    }
}

/// Process-local identity of one live shared-L2 actor. It is deliberately
/// opaque and never enters simulator outputs; its only role is to prevent a
/// response capability emitted by one actor or replay generation from being
/// accepted by another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ResponseOwner(pub(crate) u64);

pub(crate) static NEXT_RESPONSE_OWNER: AtomicU64 = AtomicU64::new(1);

pub(crate) fn allocate_response_owner_from(next: &AtomicU64) -> Result<ResponseOwner, IssueError> {
    next.fetch_update(
        Ordering::Relaxed,
        Ordering::Relaxed,
        |current| match current {
            0 => None,
            u64::MAX => Some(0),
            _ => current.checked_add(1),
        },
    )
    .map(ResponseOwner)
    .map_err(|_| IssueError::ResponseOwnerExhausted)
}

pub(crate) fn allocate_response_owner() -> Result<ResponseOwner, IssueError> {
    allocate_response_owner_from(&NEXT_RESPONSE_OWNER)
}

/// Stable identity of one response-bearing request. Its fields are private:
/// callers may route a token emitted in [`MemRequest::reply`], but cannot mint
/// an identity for a request the SM never emitted. The owner component is
/// process-unique for the lifetime of the process and rejects cross-core and
/// stale-replay token swaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResponseId {
    pub(crate) owner: ResponseOwner,
    pub(crate) warp: usize,
    pub(crate) instruction: usize,
    pub(crate) sector_index: usize,
}

impl ResponseId {
    pub(crate) const fn new(
        owner: ResponseOwner,
        warp: usize,
        instruction: usize,
        sector_index: usize,
    ) -> Self {
        Self {
            owner,
            warp,
            instruction,
            sector_index,
        }
    }
}

impl ResponseId {
    pub(crate) const fn boundary_identity(self) -> (u64, u64, u64) {
        (
            self.warp as u64,
            self.instruction as u64,
            self.sector_index as u64,
        )
    }
}

/// What the response to one modeled memory request must do at arrival.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReplyTo {
    /// Complete this L1 fill (installing the line/sector) and resolve its
    /// registered ordinary-barrier and/or async-copy-group targets.
    L1Fill {
        fill: FillKey,
        response_id: ResponseId,
    },
    /// Clear one pending unit on (warp, barrier) directly — bypass and
    /// untagged loads that never touch the modeled L1.
    Warp {
        warp: usize,
        barrier: DepBarrier,
        response_id: ResponseId,
    },
    /// Complete one global-source sector in a warp's non-bulk async-copy
    /// group. The group may still be open or may have been committed while the
    /// request was in flight.
    AsyncCopy {
        warp: usize,
        group: AsyncCopyGroupId,
        /// An unusual static form may also expose the same completion through an
        /// ordinary dependence barrier; both counters clear atomically.
        barrier: Option<DepBarrier>,
        response_id: ResponseId,
    },
    /// No SM dependency consumes returned data. Stores and reductions are
    /// intrinsically non-returning; an ATOM may also use this when its returned
    /// value has no dependence barrier, while retaining `Atomic` request
    /// semantics and its modeled RMW fetch.
    None,
}

impl ReplyTo {
    pub(crate) const fn response_id(self) -> Option<ResponseId> {
        match self {
            Self::L1Fill { response_id, .. }
            | Self::Warp { response_id, .. }
            | Self::AsyncCopy { response_id, .. } => Some(response_id),
            Self::None => None,
        }
    }
}

impl MemRequest {
    pub(crate) const fn local_mask(origin: MemoryOrigin, sector_bit: u8) -> u8 {
        match origin {
            MemoryOrigin::Global => 0,
            MemoryOrigin::Local => sector_bit,
        }
    }
}

impl MemRequest {
    pub(crate) fn fold_sector_origin(
        &mut self,
        origin: MemoryOrigin,
        sector_bit: u8,
        constant_load: bool,
        constant_deep: bool,
    ) {
        self.sector_mask |= sector_bit;
        if origin == MemoryOrigin::Local {
            self.local_sector_mask |= sector_bit;
        }
        self.constant_load |= constant_load;
        self.constant_deep |= constant_deep;
    }
}

impl MemRequest {
    /// An L1 fill request (line alloc or resident-sector fetch): Read
    /// class by construction — nothing that fills the modeled L1 is a
    /// write (stores hit-test without allocating, atomics bypass). The
    /// typed constructors keep the (source, operation, reply) coupling in
    /// one place so an invalid combination is unwritable at emission
    /// sites (gate finding 2026-07-07).
    pub(crate) fn l1_fill(
        issue_time: u64,
        sector_id: i64,
        origin: MemoryOrigin,
        fill: FillKey,
        response_id: ResponseId,
        constant_load: bool,
        constant_deep: bool,
        client_scope: L2ClientScope,
    ) -> Self {
        let sector_bit = 1 << (sector_id & 3) as u8;
        Self {
            issue_time,
            sector_id,
            origin,
            sector_mask: sector_bit,
            local_sector_mask: Self::local_mask(origin, sector_bit),
            source: L2Source::L1Fill,
            operation: MemoryOperation::Read,
            observer_operation: None,
            client_scope,
            constant_load,
            constant_deep,
            reply: ReplyTo::L1Fill { fill, response_id },
            admission_witness: None,
        }
    }
}

impl MemRequest {
    /// A follow-up sector fetch for a merge onto an older in-flight
    /// fill. It carries its own L1 sector-fill response token: sectored
    /// MSHRs complete per returned 32-byte child, so the late sector cannot
    /// clear its waiter when the parent's line child returns.
    pub(crate) fn l1_followup_fetch(
        issue_time: u64,
        sector_id: i64,
        origin: MemoryOrigin,
        fill: FillKey,
        response_id: ResponseId,
        constant_load: bool,
        constant_deep: bool,
        client_scope: L2ClientScope,
    ) -> Self {
        let sector_bit = 1 << (sector_id & 3) as u8;
        Self {
            issue_time,
            sector_id,
            origin,
            sector_mask: sector_bit,
            local_sector_mask: Self::local_mask(origin, sector_bit),
            source: L2Source::L1Fill,
            operation: MemoryOperation::Read,
            observer_operation: None,
            client_scope,
            constant_load,
            constant_deep,
            reply: ReplyTo::L1Fill { fill, response_id },
            admission_witness: None,
        }
    }
}

impl MemRequest {
    /// A plain hierarchy store sector (STG/STL): write-class tag-match rule
    /// at the L2, intrinsically non-returning.
    pub(crate) fn store(issue_time: u64, sector_id: i64, origin: MemoryOrigin) -> Self {
        Self::write(issue_time, sector_id, origin, MemoryOperation::Store)
    }
}

impl MemRequest {
    /// A non-returning hierarchy reduction sector (RED/REDG). Its write data
    /// validates the sector immediately; the event model separately charges
    /// a missing-sector RMW read when the L2 line is cold. The remaining
    /// reduction/coherence semantics stay exposed in [`TailFlags`].
    pub(crate) fn reduction(issue_time: u64, sector_id: i64) -> Self {
        Self::write(
            issue_time,
            sector_id,
            MemoryOrigin::Global,
            MemoryOperation::Reduction,
        )
    }
}

impl MemRequest {
    pub(crate) fn write(
        issue_time: u64,
        sector_id: i64,
        origin: MemoryOrigin,
        operation: MemoryOperation,
    ) -> Self {
        let sector_bit = 1 << (sector_id & 3) as u8;
        Self {
            issue_time,
            sector_id,
            origin,
            sector_mask: sector_bit,
            local_sector_mask: Self::local_mask(origin, sector_bit),
            source: L2Source::Store,
            operation,
            observer_operation: None,
            client_scope: match (origin, operation) {
                (MemoryOrigin::Local, MemoryOperation::Store) => L2ClientScope::Local,
                (MemoryOrigin::Global, MemoryOperation::Store) => L2ClientScope::Store,
                (_, MemoryOperation::Atomic) => L2ClientScope::Atomic,
                (_, MemoryOperation::Reduction) => L2ClientScope::Reduction,
                _ => L2ClientScope::Unknown,
            },
            constant_load: false,
            constant_deep: false,
            reply: ReplyTo::None,
            admission_witness: None,
        }
    }
}

impl MemRequest {
    pub(crate) fn has_valid_contract(self) -> bool {
        let base = matches!(
            (self.source, self.operation, self.reply),
            (
                L2Source::L1Fill,
                MemoryOperation::Read,
                ReplyTo::L1Fill { .. } | ReplyTo::None
            ) | (
                L2Source::Bypass,
                MemoryOperation::Read,
                ReplyTo::Warp { .. } | ReplyTo::AsyncCopy { .. } | ReplyTo::None,
            ) | (
                L2Source::Untagged,
                MemoryOperation::Read,
                ReplyTo::Warp { .. } | ReplyTo::None
            ) | (
                L2Source::Bypass,
                MemoryOperation::Atomic,
                ReplyTo::Warp { .. } | ReplyTo::None
            ) | (
                L2Source::Store,
                MemoryOperation::Store | MemoryOperation::Reduction,
                ReplyTo::None,
            )
        );
        let admission = match self.admission_witness {
            None => true,
            Some(witness) => {
                self.source == L2Source::Store
                    && self.origin == MemoryOrigin::Global
                    && matches!(
                        self.operation,
                        MemoryOperation::Store | MemoryOperation::Reduction
                    )
                    && witness.request_count() != 0
                    && witness.request_index() < witness.request_count()
                    && usize::from(witness.subcore()) < SUB_CORES
            }
        };
        base && admission
    }
}

/// Identify regular-register constant-cache-qualified loads from static SASS.
/// They continue through the normal global/L1/L2 allocation path; the marker
/// only selects the SM86 constant-return service when the shared system prices
/// the response.  Keeping this classification textual avoids treating dynamic
/// trace metadata or measured NCU values as simulator inputs.
pub(crate) fn is_constant_load_opcode(opcode_text: &str) -> bool {
    let opcode = opcode_text.split_ascii_whitespace().next().unwrap_or("");
    opcode.split('.').any(|modifier| modifier == "CONSTANT")
        && matches!(opcode_head(opcode_text), "LDG" | "LD" | "LDC" | "LDL")
}

/// Count distinct static constant-load request streams. The released SM86
/// memory model has a five-entry local memory dispatch window (four queue
/// entries plus its dispatch latch); a stream count beyond that window is the
/// conservative point at which the L0C/GCC transport can remain occupied.
/// This is derived solely from SASS shape, never from a workload id or an NCU
/// target.
pub(crate) fn constant_stream_count(schedule: &FunctionSchedule) -> usize {
    schedule
        .by_pc
        .values()
        .filter(|slot| is_constant_load_opcode(slot.opcode_text()))
        .count()
}

/// Select the observer-only write-tag window for SM86 global stores.  The
/// 256-set/2-tag-slot window is an architecture-level report-only capability;
/// it must be instantiated for every cacheable global-store stream so the
/// consumer cannot silently read a disabled counter on ordinary STG launches.
/// Bypass/untagged forms never reach this `GlobalStore` path and therefore do
/// not allocate observer tags.
pub(crate) fn stateful_write_lookup_observer_enabled(schedule: &FunctionSchedule) -> bool {
    if !matches!(
        schedule.arch.l1_write_lookup_window().status(),
        sim_arch::CapabilityStatus::Available
    ) {
        return false;
    }
    let mut has_global_store = false;
    for slot in schedule.by_pc.values() {
        let head = opcode_head(slot.opcode_text());
        has_global_store |= matches!(head, "STG" | "ST" | "STL");
    }
    has_global_store
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MemoryPipelineIssue {
    pub(crate) operand_ready_at: u64,
    pub(crate) shared_accept_at: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryIssuePipelinePlan {
    pub(crate) next: MemoryIssuePipelineState,
    pub(crate) issues: [Option<MemoryPipelineIssue>; SUB_CORES],
}

#[derive(Debug, Clone)]
pub(crate) struct ScalarExecutionPipelinePlan {
    pub(crate) next: ScalarExecutionPipelineState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResultWritebackKind {
    FixedHadd2 { bank: u8 },
    L1TexStrongSmU32 { bank: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PendingVariableWriteback {
    pub(crate) subcore: u8,
    pub(crate) bank: u8,
    pub(crate) writeback_at: u64,
    pub(crate) visible_at: u64,
    pub(crate) warp: usize,
    pub(crate) barrier: DepBarrier,
}

#[derive(Debug)]
pub(crate) struct ResultWritebackState {
    pub(crate) timing: FixedL1TexWritebackPriority,
    pub(crate) pending: Vec<PendingVariableWriteback>,
}

pub(crate) fn scalar_regular_destination_bank(opcode_text: &str) -> Option<u8> {
    let (_, operands) = opcode_text.trim().split_once(char::is_whitespace)?;
    let destination = operands.split(',').next()?.trim();
    let index = destination.strip_prefix('R')?.parse::<u16>().ok()?;
    Some((index & 1) as u8)
}

pub(crate) fn is_clocklo_issue(slot: &IssueSlot) -> bool {
    slot.is_clocklo_read()
}

pub(crate) fn result_writeback_kind(slot: &IssueSlot, arch: ArchId) -> Option<ResultWritebackKind> {
    if arch != ArchId::Sm86 {
        return None;
    }
    if let FixedLatencyOperandStatus::Projected(projection) = slot.fixed_latency_operands()
        && projection.opcode() == FixedLatencyOpcode::Hadd2
        && let [destination] = projection.destinations()
        && let Some(bank) = destination.register().sm86_regular_bank()
    {
        return Some(ResultWritebackKind::FixedHadd2 { bank });
    }
    if slot.mem_role() == MemRole::EligibleLoad(CacheOpToken::LdgEStrongSm)
        && slot.opcode_text().split_whitespace().next() == Some("LDG.E.STRONG.SM")
    {
        return scalar_regular_destination_bank(slot.opcode_text())
            .map(|bank| ResultWritebackKind::L1TexStrongSmU32 { bank });
    }
    None
}

pub(crate) fn checked_u64_counter_add(
    counter: &'static str,
    current: u64,
    add: u64,
) -> Result<u64, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

pub(crate) fn checked_u32_counter_add(
    counter: &'static str,
    current: u32,
    add: u32,
) -> Result<u32, IssueError> {
    current
        .checked_add(add)
        .ok_or(IssueError::CounterOverflow { counter })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BypassAsyncCopyInput<'a> {
    pub(crate) instruction: usize,
    pub(crate) sector_off: usize,
    pub(crate) sector_len: u16,
    pub(crate) source_cohorts: u64,
    pub(crate) destination_lines: u64,
    pub(crate) max_bank_wavefronts: u64,
    pub(crate) max_bypass_wavefronts: u64,
    pub(crate) cohorts: Option<&'a [AsyncCopyCohortDescriptor]>,
    pub(crate) sector_membership_masks: Option<&'a [u32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BypassAsyncCopyCohortObservation {
    pub(crate) pending_sources: u32,
    pub(crate) first_source_response: Option<u64>,
    pub(crate) last_source_response: Option<u64>,
    pub(crate) payload_work: u64,
    pub(crate) destination_lines: u64,
    pub(crate) bank_wavefronts: u64,
    pub(crate) bypass_wavefronts: u64,
    pub(crate) topology_penalty: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BypassPipelineCopyCandidate {
    pub(crate) instruction: usize,
    pub(crate) payload_completion: u64,
    pub(crate) topology_penalty: u64,
}

pub(crate) const BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES: u64 = 4;

pub(crate) const BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES: u64 = 2;

pub(crate) const BYPASS_PIPELINE_CANDIDATE_Q: [u64; 5] = [1, 2, 4, 8, 16];

/// Independent RTX 3090 Ti clock64 dependent-MUFU probe endpoint. This is an
/// effective source-qualified latency for same-register SFU chains, not a
/// universal SFU latency or a claim about hidden pipeline stages.
pub(crate) const SM86_DEPENDENT_SFU_LATENCY_CYCLES: u64 = 30;

pub(crate) fn bypass_topology_candidate_penalty(
    bypass_wavefronts: u64,
    bank_wavefronts: u64,
) -> Result<u64, IssueError> {
    if bypass_wavefronts == 0 || bank_wavefronts == 0 {
        return Ok(0);
    }
    let wavefront_penalty = bypass_wavefronts
        .saturating_sub(1)
        .checked_mul(BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES)
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "bypass topology candidate wavefront penalty",
            lhs: bypass_wavefronts.saturating_sub(1),
            rhs: BYPASS_TOPOLOGY_CANDIDATE_WAVEFRONT_CYCLES,
        })?;
    let bank_penalty = bank_wavefronts
        .saturating_sub(1)
        .checked_mul(BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES)
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "bypass topology candidate bank penalty",
            lhs: bank_wavefronts.saturating_sub(1),
            rhs: BYPASS_TOPOLOGY_CANDIDATE_BANK_CYCLES,
        })?;
    checked_cycle_add(
        "bypass topology candidate local penalty",
        wavefront_penalty,
        bank_penalty,
    )
}

pub(crate) fn bypass_pipeline_candidate_completions(
    copies: &mut [BypassPipelineCopyCandidate],
) -> Result<[u64; BYPASS_PIPELINE_CANDIDATE_Q.len()], IssueError> {
    copies.sort_by_key(|copy| (copy.payload_completion, copy.instruction));
    let mut completions = [0_u64; BYPASS_PIPELINE_CANDIDATE_Q.len()];
    for (completion, q) in completions.iter_mut().zip(BYPASS_PIPELINE_CANDIDATE_Q) {
        *completion = bypass_pipeline_candidate_completion(copies, q)?;
    }
    Ok(completions)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RegisterReadyEntry {
    pub(crate) register: SassRegister,
    pub(crate) lane_mask: u32,
    pub(crate) ready_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Sm86RfcKey {
    pub(crate) block: usize,
    pub(crate) warp_slot: i32,
    pub(crate) bank: u8,
    pub(crate) register: u16,
    pub(crate) operand: u8,
}

pub(crate) fn sm86_register_read_plan(
    status: &FixedLatencyOperandStatus,
    block: usize,
    warp_slot: i32,
    now: u64,
    port_ready: [u64; 2],
    rfc: &VecDeque<Sm86RfcKey>,
) -> Result<Sm86RegisterReadPlan, IssueError> {
    let projection = match status {
        FixedLatencyOperandStatus::Projected(projection) => projection,
        FixedLatencyOperandStatus::NotApplicable => {
            return Ok(Sm86RegisterReadPlan::default());
        }
        FixedLatencyOperandStatus::Gap(_) => {
            return Ok(Sm86RegisterReadPlan {
                unsupported_shape: true,
                ..Sm86RegisterReadPlan::default()
            });
        }
    };
    let mut plan = Sm86RegisterReadPlan {
        modeled: true,
        next_port_ready: port_ready,
        ..Sm86RegisterReadPlan::default()
    };
    for (operand_ix, source) in projection.sources().iter().enumerate() {
        let SassSourceOperand::Register(source) = source else {
            continue;
        };
        let register = source.register();
        if register.file() != SassRegisterFile::Regular || register.index().is_none() {
            continue;
        }
        // The first arm deliberately refuses vector-register RFC/port
        // attribution rather than inventing a multi-cell slot mapping.
        if register.span_registers() != 1 {
            plan.unsupported_shape = true;
            continue;
        }
        let bank = register
            .sm86_regular_bank()
            .ok_or(IssueError::CounterOverflow {
                counter: "SM86 register-read bank",
            })?;
        let key = Sm86RfcKey {
            block,
            warp_slot,
            bank,
            register: register.index().ok_or(IssueError::CounterOverflow {
                counter: "SM86 register-read index",
            })?,
            operand: u8::try_from(operand_ix).map_err(|_| IssueError::CounterOverflow {
                counter: "SM86 register-read operand position",
            })?,
        };
        if source.reused() {
            if rfc.contains(&key) {
                plan.rfc_hits =
                    plan.rfc_hits
                        .checked_add(1)
                        .ok_or(IssueError::CounterOverflow {
                            counter: "SM86 register-file reuse hits",
                        })?;
                plan.rfc_consumes.push(key);
                continue;
            }
            plan.rfc_misses =
                plan.rfc_misses
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "SM86 register-file reuse misses",
                    })?;
            // A miss still reads the regular register and seeds the compiler
            // managed entry for the next same-key use.
            plan.rfc_inserts.push(key);
        }
        let bank_index = usize::from(bank);
        plan.reads[bank_index] =
            plan.reads[bank_index]
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "SM86 register-file source reads",
                })?;
    }
    if plan.reads.iter().any(|&reads| reads > 3) {
        plan.unsupported_shape = true;
        return Ok(plan);
    }
    let first_read = now.checked_add(1).ok_or(IssueError::TimeOverflow {
        operation: "SM86 register-read port window",
        base: now,
        delta: 1,
    })?;
    for bank in 0..2 {
        if plan.reads[bank] == 0 {
            continue;
        }
        let start = first_read.max(plan.next_port_ready[bank]);
        plan.port_conflict_cycles = plan.port_conflict_cycles.max(start - first_read);
        plan.next_port_ready[bank] =
            start
                .checked_add(plan.reads[bank])
                .ok_or(IssueError::TimeOverflow {
                    operation: "SM86 register-read port reservation",
                    base: start,
                    delta: plan.reads[bank],
                })?;
    }
    Ok(plan)
}

impl WarpState<'_> {
    pub(crate) fn can_fetch_instruction(&self, capacity: usize) -> bool {
        !self.done
            && self.front_end_fetch_cursor < self.insts.len()
            && self.instruction_buffer.len() < capacity
    }
}

impl WarpState<'_> {
    pub(crate) fn instruction_front_end_ready_at(&self) -> Option<u64> {
        let buffered = self.instruction_buffer.front()?;
        (buffered.instruction == self.next).then_some(buffered.ready_at)
    }
}

impl WarpState<'_> {
    pub(crate) fn preflight_buffered_issue(&self, warp: usize) -> Result<(), IssueError> {
        let Some(buffered) = self.instruction_buffer.front() else {
            return Err(IssueError::InstructionBufferMissing {
                warp,
                instruction: self.next,
            });
        };
        if buffered.instruction != self.next {
            return Err(IssueError::InstructionBufferOrder {
                warp,
                expected: self.next,
                got: buffered.instruction,
            });
        }
        Ok(())
    }
}

impl WarpState<'_> {
    pub(crate) fn consume_buffered_issue(&mut self, warp: usize) -> Result<(), IssueError> {
        self.preflight_buffered_issue(warp)?;
        self.instruction_buffer.pop_front();
        Ok(())
    }
}

impl WarpState<'_> {
    pub(crate) fn next_issue_retires(&self) -> Result<bool, IssueError> {
        let next = checked_usize_counter_add("warp instruction cursor", self.next, 1)?;
        Ok(next == self.insts.len())
    }
}

impl WarpState<'_> {
    pub(crate) fn preflight_pending_add(
        &self,
        warp: usize,
        barrier: Option<DepBarrier>,
        add: u32,
    ) -> Result<(), IssueError> {
        let Some(barrier) = barrier else {
            return Ok(());
        };
        let current = self.barrier_pending[barrier.index()];
        current
            .checked_add(add)
            .ok_or(IssueError::PendingBarrierOverflow {
                warp,
                barrier: barrier.wire(),
                current,
                add,
            })?;
        Ok(())
    }
}

impl WarpState<'_> {
    pub(crate) fn preflight_pending_remove(
        &self,
        warp: usize,
        barrier: DepBarrier,
        remove: u32,
    ) -> Result<(), IssueError> {
        let current = self.barrier_pending[barrier.index()];
        current
            .checked_sub(remove)
            .ok_or(IssueError::PendingBarrierUnderflow {
                warp,
                barrier: barrier.wire(),
                current,
                remove,
            })?;
        Ok(())
    }
}

impl WarpState<'_> {
    /// Producer bookkeeping: pending (unknown write-back) units on a
    /// dependence barrier; `None` carries no dependency.
    pub(crate) fn add_pending(
        &mut self,
        warp: usize,
        barrier: Option<DepBarrier>,
        add: u32,
    ) -> Result<(), IssueError> {
        self.preflight_pending_add(warp, barrier, add)?;
        if let Some(barrier) = barrier {
            let pending = &mut self.barrier_pending[barrier.index()];
            let current = *pending;
            *pending = current
                .checked_add(add)
                .ok_or(IssueError::PendingBarrierOverflow {
                    warp,
                    barrier: barrier.wire(),
                    current,
                    add,
                })?;
        }
        Ok(())
    }
}

impl WarpState<'_> {
    /// Pending units resolved at `now`: decrement AND max-accumulate the
    /// clear stamp (hardware counter semantics — the single copy of the
    /// pairing every response path must preserve).
    pub(crate) fn clear_pending(
        &mut self,
        warp: usize,
        barrier: DepBarrier,
        remove: u32,
        now: u64,
    ) -> Result<(), IssueError> {
        self.preflight_pending_remove(warp, barrier, remove)?;
        let b = barrier.index();
        let current = self.barrier_pending[b];
        self.barrier_pending[b] =
            current
                .checked_sub(remove)
                .ok_or(IssueError::PendingBarrierUnderflow {
                    warp,
                    barrier: barrier.wire(),
                    current,
                    remove,
                })?;
        if now >= self.barrier_clear[b] {
            self.barrier_clear[b] = now;
            self.barrier_origin[b] = DependencyOrigin::L1Tex;
        }
        Ok(())
    }
}

impl WarpState<'_> {
    /// Release pending units WITHOUT a stamp (a retry entry's own units;
    /// its per-sector waiters carry the stamps).
    pub(crate) fn release_pending(
        &mut self,
        warp: usize,
        barrier: Option<DepBarrier>,
        remove: u32,
    ) -> Result<(), IssueError> {
        let Some(barrier) = barrier else {
            return Ok(());
        };
        self.preflight_pending_remove(warp, barrier, remove)?;
        let pending = &mut self.barrier_pending[barrier.index()];
        let current = *pending;
        *pending = current
            .checked_sub(remove)
            .ok_or(IssueError::PendingBarrierUnderflow {
                warp,
                barrier: barrier.wire(),
                current,
                remove,
            })?;
        Ok(())
    }
}

impl WarpState<'_> {
    /// Max-accumulate a KNOWN clear stamp on a dependence barrier (counter
    /// semantics: never overwrite); `None` carries no dependency.
    pub(crate) fn stamp_clear(
        &mut self,
        barrier: Option<DepBarrier>,
        at: u64,
        origin: DependencyOrigin,
    ) {
        if let Some(barrier) = barrier {
            let index = barrier.index();
            if at >= self.barrier_clear[index] {
                self.barrier_clear[index] = at;
                self.barrier_origin[index] = origin;
            }
        }
    }
}

impl WarpState<'_> {
    /// Earliest time the control wait mask and any explicit ordinary `DEPBAR`
    /// operand clear; `None` while a required barrier has responses outstanding
    /// (unknown clear time). SB0 asynchronous-copy waits are handled separately.
    pub(crate) fn wait_clear_time(&self) -> Option<u64> {
        let Some(slot) = self.next_slot else {
            return Some(0);
        };
        let mut t = 0;
        for b in 0..DEP_BARRIERS {
            if slot.control.wait_mask.contains_index(b) {
                if self.barrier_pending[b] > 0 {
                    return None;
                }
                t = t.max(self.barrier_clear[b]);
            }
        }
        for b in 0..DEP_BARRIERS {
            if !self.ordinary_dependency_wait_mask.contains_index(b)
                || slot.control.wait_mask.contains_index(b)
            {
                continue;
            }
            if self.barrier_pending[b] > 0 {
                return None;
            }
            t = t.max(self.barrier_clear[b]);
        }
        Some(t)
    }
}

impl WarpState<'_> {
    /// Earliest time at which the current instruction is ready when the
    /// immediate-constant (L0C/L1C) response frontier is deliberately
    /// ignored.  Huerta's SM86 scheduler treats that path specially: a
    /// greedy warp may remain the candidate for a bounded miss-switch
    /// window while all ordinary control/dependency frontiers are ready.
    pub(crate) fn ready_at_without_immediate_constant(&self) -> Option<u64> {
        if self.done || self.at_block_barrier {
            return None;
        }
        let dependency_ready = self.wait_clear_time()?;
        Some(
            self.eligible
                .max(dependency_ready)
                .max(self.cta_barrier_ready)
                .max(self.barrier_pipeline_ready)
                .max(self.branch_pipeline_ready),
        )
    }
}

impl WarpState<'_> {
    pub(crate) fn ready_at(&self) -> Option<u64> {
        self.ready_at_without_immediate_constant()
            .map(|ready| ready.max(self.imc_wait_until))
    }
}

pub(crate) fn async_copy_completion_work(kind: AccessKind) -> Option<u64> {
    match kind {
        AccessKind::GlobalAsyncCopy {
            completion_work, ..
        } => match decode_async_copy_completion(completion_work) {
            AsyncCopyCompletion::Inactive => None,
            AsyncCopyCompletion::PayloadWork(work)
            | AsyncCopyCompletion::ExactDestinationWork(work) => Some(u64::from(work)),
            AsyncCopyCompletion::PayloadAndObservedTopology { payload_work, .. } => {
                Some(u64::from(payload_work))
            }
        },
        AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalBypass { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalStore { .. }
        | AccessKind::GlobalReduction { .. } => None,
    }
}

pub(crate) fn async_copy_destination_wavefronts(
    kind: AccessKind,
    params: &IssueModelParams,
) -> Result<Option<u64>, IssueError> {
    let AccessKind::GlobalAsyncCopy {
        completion_work,
        l1_policy: LdgstsL1Policy::CacheAtAllLevels,
        ..
    } = kind
    else {
        return Ok(None);
    };
    let AsyncCopyCompletion::ExactDestinationWork(work) =
        decode_async_copy_completion(completion_work)
    else {
        return Ok(None);
    };
    let work_per_wavefront = params
        .ldgsts_cache_all_destination_wavefront_cycles
        .checked_mul(params.ldgsts_delivery_units_per_cycle())
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "LDGSTS destination work per wavefront",
            lhs: params.ldgsts_cache_all_destination_wavefront_cycles,
            rhs: params.ldgsts_delivery_units_per_cycle(),
        })?;
    let work = u64::from(work);
    if work_per_wavefront == 0 || work % work_per_wavefront != 0 {
        return Err(IssueError::InvalidAsyncCopyDestinationWork {
            completion_work: work,
            work_per_wavefront,
        });
    }
    Ok(Some(work / work_per_wavefront))
}

/// `LDGDEPBAR` encodes SB0 in the ordinary write-barrier field, but that bit is
/// the async-group protocol itself rather than a generic 25-cycle producer.
pub(crate) fn ordinary_write_barrier(slot: &IssueSlot) -> Option<DepBarrier> {
    if slot.op_class == OpClass::AsyncCopyCommit {
        None
    } else {
        slot.control.write_barrier
    }
}

/// Data-ready time for one issued record at time `t` in the Flat/L1Const
/// modes (fill times known at issue), driving the L1 state/counters when
/// attached. Producer visibility is computed when either an ordinary write
/// barrier or an async-copy group consumes it; cache accesses still run without
/// either observer because they mutate modeled state.
pub(crate) fn data_ready_time(
    kind: AccessKind,
    sectors: &[i64],
    t: u64,
    params: &IssueModelParams,
    l1: Option<&mut L1Cache>,
    write_barrier: Option<DepBarrier>,
    observe_async_copy: bool,
) -> Result<Option<u64>, IssueError> {
    let needs_ready = write_barrier.is_some() || observe_async_copy;
    match kind {
        AccessKind::PredicatedOffMemory => Ok(None),
        AccessKind::NonMemory => {
            if needs_ready {
                params.non_memory_ready(t).map(Some)
            } else {
                Ok(None)
            }
        }
        AccessKind::Shared { wavefronts, .. } => {
            if needs_ready {
                params.shared_ready(t, wavefronts).map(Some)
            } else {
                Ok(None)
            }
        }
        AccessKind::GlobalEligible { off, len, .. } | AccessKind::LocalLoad { off, len } => {
            let origin = match kind {
                AccessKind::GlobalEligible { .. } => MemoryOrigin::Global,
                AccessKind::LocalLoad { .. } => MemoryOrigin::Local,
                _ => unreachable!("match arm admits only hierarchy loads"),
            };
            match l1 {
                Some(l1) => {
                    let request_sectors = &sectors[off..off + usize::from(len)];
                    l1.begin_lookup_batch_for_client_with_sectors(
                        L1LookupClient::Unknown,
                        request_sectors,
                    );
                    let mut ready = needs_ready
                        .then(|| {
                            checked_cycle_add("L1 hit-pipeline floor", t, params.l1_hit_latency)
                        })
                        .transpose()?;
                    for &sector in &sectors[off..off + usize::from(len)] {
                        if let Some(current) = &mut ready {
                            let (_, sector_ready) = l1.access_eligible(origin, sector, t)?;
                            *current = (*current).max(sector_ready);
                        } else {
                            l1.access_eligible_unobserved(origin, sector, t)?;
                        }
                    }
                    Ok(ready)
                }
                None => {
                    if needs_ready {
                        checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                            .map(Some)
                    } else {
                        Ok(None)
                    }
                }
            }
        }
        AccessKind::GlobalBypass { len, token, .. } => match l1 {
            Some(l1) => {
                l1.count_bypass(
                    MemoryOrigin::Global,
                    u64::from(len),
                    token == CacheOpToken::Atom,
                )?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            None => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
        AccessKind::GlobalAsyncCopy {
            off,
            len,
            l1_policy,
            ..
        } => match (l1, l1_policy) {
            (Some(l1), LdgstsL1Policy::CacheAtAllLevels) => {
                let request_sectors = &sectors[off..off + usize::from(len)];
                l1.begin_async_copy_lookup_batch_with_sectors(request_sectors);
                let mut ready = needs_ready
                    .then(|| {
                        checked_cycle_add("LDGSTS L1 hit-pipeline floor", t, params.l1_hit_latency)
                    })
                    .transpose()?;
                for &sector in &sectors[off..off + usize::from(len)] {
                    if let Some(current) = &mut ready {
                        let (_, sector_ready) = l1.access_eligible_with_mshr_floor(
                            MemoryOrigin::Global,
                            sector,
                            t,
                            params.ldgsts_l1_mshr_floor,
                        )?;
                        *current = (*current).max(sector_ready);
                    } else {
                        l1.access_eligible_unobserved_with_mshr_floor(
                            MemoryOrigin::Global,
                            sector,
                            t,
                            params.ldgsts_l1_mshr_floor,
                        )?;
                    }
                }
                Ok(ready)
            }
            (Some(l1), LdgstsL1Policy::BypassL1) => {
                l1.count_bypass(MemoryOrigin::Global, u64::from(len), false)?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            (None, _) => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
        AccessKind::GlobalUntagged { len, .. } => match l1 {
            Some(l1) => {
                l1.count_untagged(MemoryOrigin::Global, u64::from(len))?;
                if needs_ready {
                    params.store_visibility_ready(t).map(Some)
                } else {
                    Ok(None)
                }
            }
            None => {
                if needs_ready {
                    checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                        .map(Some)
                } else {
                    Ok(None)
                }
            }
        },
        AccessKind::GlobalStore { off, len }
        | AccessKind::GlobalReduction { off, len, .. }
        | AccessKind::LocalStore { off, len } => {
            let origin = match kind {
                AccessKind::GlobalStore { .. } | AccessKind::GlobalReduction { .. } => {
                    MemoryOrigin::Global
                }
                AccessKind::LocalStore { .. } => MemoryOrigin::Local,
                _ => unreachable!("match arm admits only hierarchy writes"),
            };
            let reduction = matches!(kind, AccessKind::GlobalReduction { .. });
            match l1 {
                Some(l1) => {
                    if let Some(hits) = projected_l1_write_lookup_hits(kind, sectors) {
                        l1.count_write_lookup_hits(origin, hits)?;
                    }
                    for &sector in &sectors[off..off + usize::from(len)] {
                        if reduction {
                            l1.probe_reduction(origin, sector, t)?;
                        } else {
                            l1.probe_store(origin, sector, t)?;
                        }
                    }
                    if needs_ready {
                        params.store_visibility_ready(t).map(Some)
                    } else {
                        Ok(None)
                    }
                }
                None => {
                    if needs_ready {
                        checked_cycle_add("flat global visibility", t, params.global_flat_latency)
                            .map(Some)
                    } else {
                        Ok(None)
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ImmediateConstantPending {
    pub(crate) line: u64,
    pub(crate) ready_at: u64,
}

#[derive(Debug)]
pub(crate) struct ImmediateConstantPlan {
    pub(crate) picks: [Option<usize>; SUB_CORES],
    pub(crate) cache: Option<Box<[ImmediateConstantCacheState; SUB_CORES]>>,
    pub(crate) waits: [Option<(usize, u64)>; SUB_CORES],
    /// Additional non-greedy candidates that were probed and parked while
    /// scanning past an IMC miss in the same scheduler opportunity.
    pub(crate) extra_waits: Vec<(usize, u64)>,
    pub(crate) miss_observed: bool,
}

/// Resumable single-SM replay actor. See the module docs for the mode
/// semantics and the run-ahead (horizon) contract.
#[derive(Debug)]
pub(crate) struct SharedWriteLookupRuntime {
    pub(crate) observer: Option<Rc<RefCell<L1WriteLookupObserver>>>,
    /// Per-SMSP source-hit return-port frontiers for cacheable LDGSTS.
    pub(crate) async_l1_hit_service_ready: [u64; SUB_CORES],
}

#[cfg(feature = "reference")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReferenceScalarMacroExecution {
    pub(crate) warp_slot: i32,
    pub(crate) first_issue: u64,
    pub(crate) last_issue: u64,
    pub(crate) end_cycle: u64,
    pub(crate) instruction_count: u64,
}

pub(crate) struct SmCore<'a> {
    pub(crate) schedule: &'a FunctionSchedule,
    /// Static schedule values in PC order. Contiguous binary search avoids a
    /// pointer-chasing B-tree lookup for every dynamic issue while preserving
    /// the schedule map as the authoritative join/diagnostic surface.
    pub(crate) schedule_slots: Vec<(u64, &'a IssueSlot)>,
    /// Static PC-indexed memory timing table. It is sorted once at admission so
    /// the hot issue loop uses a branch-predictable binary search instead of a
    /// randomized hash lookup for every resident warp.
    pub(crate) memory_pipeline_schedule: Box<[(u64, MemoryPipelineSlot)]>,
    /// Static PCs that enter the measured multi-site constant operand path.
    /// Sorted static PC set; issue-cycle membership is binary-searched to
    /// avoid randomized hashing on every resident-warp readiness check.
    pub(crate) constant_path_pcs: Box<[u64]>,
    /// Pre-parsed immediate-constant line references keyed by static PC.
    /// Sorted static PC -> immediate-constant line table.
    pub(crate) immediate_constant_schedule: Box<[(u32, Box<[u64]>)]>,
    /// Private IMC/L0C state for each SMSP; absent for compact replay inputs.
    pub(crate) immediate_constant_cache: Option<Box<[ImmediateConstantCacheState; SUB_CORES]>>,
    /// Remaining Huerta SM86 L0C miss-switch opportunities for each SMSP.
    /// A successful issue reloads this bounded countdown; while the held
    /// (greedy) warp is otherwise ready but its immediate-constant operand is
    /// not, the scheduler keeps it selected until the countdown expires.
    /// This is intentionally separate from `imc_wait_until`, which is the
    /// response frontier for the specific line request.
    pub(crate) immediate_constant_switch_remaining: [u64; SUB_CORES],
    pub(crate) params: IssueModelParams,
    pub(crate) shared_memory_residency: SharedMemoryResidency,
    pub(crate) physical_warp_slots: u32,
    pub(crate) instruction_front_end: Option<InstructionFrontEndState>,
    pub(crate) memory_issue_pipeline: Option<MemoryIssuePipelineState>,
    pub(crate) mem: MemModel,
    pub(crate) blocks: &'a [BlockReplay],
    pub(crate) block_live_warps: Vec<u32>,
    pub(crate) instruction_limit: u64,
    pub(crate) warps: Vec<WarpState<'a>>,
    /// Cold protocol storage parallel to `warps`; entries allocate only after
    /// an effective asynchronous-copy source is issued.
    pub(crate) async_copies: Vec<AsyncCopySlot>,
    /// Construction-time specialization flag. Kernels without async-copy
    /// sources or controls bypass the protocol path once per scheduler batch,
    /// not once per issued instruction or resident-warp scan.
    pub(crate) has_async_copy_protocol: bool,
    pub(crate) block_states: Vec<BlockState>,
    pub(crate) subcore_warps: [Vec<usize>; SUB_CORES],
    pub(crate) hold: [Option<usize>; SUB_CORES],
    /// Fetch/decode continuation is distinct from issue hold. The SM86
    /// control bit Yield drops issue hold for the next cycle, but Huerta's
    /// fetch/decode model continues the warp issued in the previous cycle
    /// while its instruction buffer has capacity.
    pub(crate) fetch_last_issued: [Option<usize>; SUB_CORES],
    /// Architecture-local CTA-barrier issue resource present when the joined
    /// function contains a characterized barrier instruction.
    pub(crate) barrier_issue: Option<CtaBarrierIssue>,
    /// Earliest cycle at which the one SM-shared barrier server can accept
    /// another warp arrival.
    pub(crate) barrier_pipe_ready: u64,
    /// Earliest issue time of each SMSP's local Tensor Core. This timeline is
    /// kernel-local but survives CTA retirement/admission waves on the SM.
    pub(crate) tensor_pipe_ready: [u64; SUB_CORES],
    /// Independent per-SMSP uniform/SFU initiation timeline.
    pub(crate) sfu_pipe_ready: [u64; SUB_CORES],
    /// Finite scalar INT/FP execution windows, one per SMSP.
    pub(crate) scalar_execution_pipeline: Option<Box<ScalarExecutionPipelineState>>,
    /// Full-mode SM86 effective register-read envelope.  Ports are per SMSP
    /// and bank; the reuse set is compiler-keyed and intentionally not a
    /// physical register-file claim.
    pub(crate) sm86_rf_enabled: bool,
    pub(crate) sm86_rf_port_ready: [[u64; 2]; SUB_CORES],
    pub(crate) sm86_rfc: [VecDeque<Sm86RfcKey>; SUB_CORES],
    /// Cold shared observer/async-return state is boxed to keep the actor's
    /// compact hot structure within the established 4 KiB bound.
    pub(crate) shared_write_lookup_runtime: Box<SharedWriteLookupRuntime>,
    /// Earliest free cycle of the cache-all LDGSTS destination service shared
    /// by every resident warp on this SM.
    pub(crate) ldgsts_destination_pipe_ready: u64,
    /// Pending barrier-observed L1-hit result writebacks. Allocated only when
    /// one joined function contains both the admitted HADD2 producer and the
    /// admitted sm_86 STRONG.SM scalar load form.
    pub(crate) result_writebacks: Option<ResultWritebackState>,
    pub(crate) unmodeled_tails: TailFlags,
    /// Box the cold diagnostic aggregate so adding a new observation surface
    /// cannot inflate the scheduler's hot actor beyond its compact-size
    /// invariant.
    pub(crate) readiness: Box<ReadinessDiagnostics>,
    pub(crate) memory_service_candidates: MemoryServiceCandidateState,
    pub(crate) emit_admission_witnesses: bool,
    pub(crate) memory_admission_feedback: Option<Box<MemoryAdmissionFeedbackState>>,
    /// Effective picks from the previous transaction in the current logical
    /// cycle. Distinguishes dispatch progress from a parked IMC miss.
    pub(crate) last_issue_picks: [Option<usize>; SUB_CORES],
    pub(crate) instructions_per_smsp: [u64; SUB_CORES],
    pub(crate) issue_active_cycles_per_smsp: [u64; SUB_CORES],
    pub(crate) last_observed_issue: [Option<u64>; SUB_CORES],
    pub(crate) warp_lifetimes: Vec<WarpLifetime>,
    pub(crate) issued: u64,
    pub(crate) t: u64,
    /// Latest modeled SM-local completion frontier that can occur after the
    /// last issue (for example, a terminal CTA-barrier release).
    pub(crate) local_completion_time: Option<u64>,
    /// P15 logical route-service availability.  Full replay only; fast
    /// diagnostics never construct or consume this state.
    pub(crate) sm120_flat_service_enabled: bool,
    pub(crate) sm120_flat_global_service_ready: u64,
    pub(crate) sm120_flat_shared_service_ready: u64,
    pub(crate) sm120_flat_multiwarp_service_enabled: bool,
    pub(crate) sm120_flat_multiwarp_global_service_ready: u64,
    pub(crate) sm120_flat_multiwarp_dense: bool,
    pub(crate) sm120_flat_multiwarp_medium: bool,
    pub(crate) sm120_flat_shared_medium: bool,
    pub(crate) sm120_flat_async_dense: bool,
    // Logical recent-source window, not a physical cache geometry.
    pub(crate) sm120_async_sources: VecDeque<(i64, u64)>,
    pub(crate) sm120_flat_async_service_enabled: bool,
    pub(crate) sm120_flat_async_service_ready: u64,
    pub(crate) sm120_flat_async_depth_service_per_group: u64,
    pub(crate) sm120_flat_async_depth_service_ready: u64,
    pub(crate) sm120_flat_store_retire_enabled: bool,
    pub(crate) sm120_flat_store_retire_ready: u64,
    pub(crate) arrival_stamp: u64,
    pub(crate) resident_warps: u32,
    pub(crate) resident_shared_memory_bytes: u64,
    pub(crate) resident_warp_slots: Box<HashSet<i32>>,
    pub(crate) unfinished_warps: usize,
    pub(crate) next_block: usize,
    /// Test/reference-only switch. The production default retains next-event
    /// advancement; the reference lane walks locally idle cycles one at a
    /// time so differential runs can separate clock jumps from state changes.
    pub(crate) dense_reference: bool,
    #[cfg(feature = "reference")]
    pub(crate) advance_diagnostics: AdvanceDiagnostics,
    #[cfg(feature = "reference")]
    pub(crate) reference_issue_effects: Vec<(u64, u64, u32)>,
}

#[derive(Debug)]
pub(crate) enum SharedAccessError {
    Stalled,
    Issue(IssueError),
}

/// One eligible sector access in L1Shared mode at time `now`, emitting a
/// request on a fresh miss. Returns `Ok(known_ready)` for hits,
/// `Ok(None)` when the data waits on a response, and
/// `Err(SharedAccessError::Stalled)` on an MSHR stall (caller parks the
/// remainder). Accounting failures remain typed.
///
/// A sector merging onto a line fill emitted during the CURRENT
/// emission batch (`st.line_folds`, keyed by line and issue timestamp)
/// folds its bit into that request's `sector_mask`, so one L2
/// transaction carries same-cycle sectors and worst-of-mask pricing
/// covers every warp/instruction/retry-entry at that timestamp. A merge onto
/// an OLDER in-flight fill emits a follow-up sector fetch with an independent
/// `FillKey::Sector` response; the L2 must see every fetched sector exactly
/// once per fill and the waiter clears only at that sector's ready event.
pub(crate) fn access_shared(
    st: &mut SharedL2State,
    warps: &mut [WarpState<'_>],
    wix: usize,
    target: Option<FillTarget>,
    mshr_floor: Option<NonZeroU32>,
    origin: MemoryOrigin,
    constant_load: bool,
    constant_deep: bool,
    client_scope: L2ClientScope,
    sector_id: i64,
    response_id: ResponseId,
    now: u64,
    out: &mut Vec<MemRequest>,
) -> Result<Option<u64>, SharedAccessError> {
    let sbit = 1u8 << (sector_id & 3) as u8;
    match access_shared_l1(
        &mut st.l1,
        origin,
        sector_id,
        now,
        target.is_some(),
        mshr_floor,
    )
    .map_err(IssueError::from)?
    {
        SharedL1Access::Hit { ready } => {
            if origin == MemoryOrigin::Global {
                record_global_l1_observation(st, sector_id, false, true, true, None, now)?;
            }
            Ok(ready)
        }
        SharedL1Access::MissRequest { fill } => {
            if origin == MemoryOrigin::Global {
                // The directory must not publish ownership before the L1 fill
                // response is visible. A later fill-visible event is required
                // to make this sector resident in the visibility protocol.
                record_global_l1_observation(
                    st,
                    sector_id,
                    false,
                    false,
                    false,
                    Some(fill_generation(fill)),
                    now,
                )?;
            }
            let request = MemRequest::l1_fill(
                now,
                sector_id,
                origin,
                fill,
                response_id,
                constant_load,
                constant_deep,
                client_scope,
            );
            register_response(st, request.reply)?;
            st.response_masks.insert(response_id, sbit);
            out.push(request);
            if let FillKey::Line { line, .. } = fill {
                st.fill_masks.insert(line, sbit);
                st.line_folds.entry((line, now)).or_insert(out.len() - 1);
            }
            add_fill_waiter(st, warps, fill, wix, target, constant_deep)?;
            Ok(None)
        }
        SharedL1Access::MergedPending { fill } => {
            if origin == MemoryOrigin::Global {
                record_global_l1_observation(
                    st,
                    sector_id,
                    false,
                    false,
                    false,
                    Some(fill_generation(fill)),
                    now,
                )?;
            }
            if let FillKey::Line { line, .. } = fill {
                // Pairing invariant (proven at the 2026-07-07 gate): a
                // line-pending merge always has a live fill_masks entry
                // (created at MissRequest, removed at the response that
                // also completes the fill). Fail LOUD if a future
                // change breaks it — a fabricated empty mask would
                // silently re-fetch already-sent sectors.
                let already_sent = st.fill_masks.get(&line).ok_or(SharedAccessError::Issue(
                    IssueError::FillMaskMissing { line },
                ))? & sbit
                    != 0;
                if !already_sent {
                    if let Some(&ix) = st.line_folds.get(&(line, now)) {
                        *st.fill_masks
                            .get_mut(&line)
                            .expect("fill mask checked above") |= sbit;
                        out[ix].fold_sector_origin(origin, sbit, constant_load, constant_deep);
                        if let Some(response_id) = out[ix].reply.response_id() {
                            *st.response_masks.entry(response_id).or_insert(0) |= sbit;
                        }
                    } else {
                        let sector_fill = st
                            .l1
                            .reserve_pending_line_sector(fill, (sector_id & 3) as u8)
                            .map_err(IssueError::from)?;
                        *st.fill_masks
                            .get_mut(&line)
                            .expect("fill mask checked above") |= sbit;
                        let request = MemRequest::l1_followup_fetch(
                            now,
                            sector_id,
                            origin,
                            sector_fill,
                            response_id,
                            constant_load,
                            constant_deep,
                            client_scope,
                        );
                        register_response(st, request.reply)?;
                        st.response_masks.insert(response_id, sbit);
                        out.push(request);
                        add_fill_waiter(st, warps, sector_fill, wix, target, constant_deep)?;
                        return Ok(None);
                    }
                }
                // Re-touch of an already-sent sector: MSHR merge, no
                // new fetch (matches NCU's xbar sector accounting).
            }
            // Sector-fill merges need nothing: that sector's fetch was
            // sent with its own request.
            add_fill_waiter(st, warps, fill, wix, target, constant_deep)?;
            Ok(None)
        }
        SharedL1Access::Stalled => Err(SharedAccessError::Stalled),
    }
}

/// Replay one SM's blocks to completion under the v0 issue model
/// (synchronous Flat/L1Const modes; the shared-L2 mode runs through
/// [`SmCore`] + the event system). `l1_config`, when supplied, attaches
/// a FRESH per-SM L1 (cold cache per kernel, NCU `--cache-control all`
/// parity).
pub(crate) fn simulate_sm_with_residency(
    blocks: &[BlockReplay],
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    shared_memory_residency: SharedMemoryResidency,
    l1_config: Option<L1Config>,
) -> Result<SmSimResult, IssueError> {
    let mode = match l1_config {
        Some(cfg) => SmMemMode::L1Const(cfg),
        None => SmMemMode::Flat,
    };
    let mut core =
        SmCore::new_with_residency(blocks, schedule, params, shared_memory_residency, mode)?;
    let mut out = Vec::new();
    match core.advance(ReplayHorizon::Unbounded, &mut out)? {
        Advance::Done => {}
        Advance::Idle { .. } => unreachable!("synchronous modes emit no requests"),
    }
    debug_assert!(out.is_empty(), "synchronous modes emit no requests");
    Ok(core.result())
}
