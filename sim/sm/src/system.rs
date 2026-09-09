//! Kernel-level event system: the per-SM issue actors ([`SmCore`]) over
//! the SHARED L2 model, coordinated by the deterministic engine queue
//! (design §6.1 + §6.3 — the request/response inversion).
//!
//! Event shape (tagged enum, no virtual dispatch):
//! - `SmWake` — resume an SM actor; it runs issue cycles strictly before
//!   the next queued event (plus its own emitted-request response bound,
//!   enforced inside [`SmCore::advance`]).
//! - `L2Access` — a request carrying 1..=4 sectors of one 128 B line
//!   arrives at the shared L2. The engine's `(time, seq)` order IS the
//!   cross-SM interleaving the timeless Phase-1 replay approximated
//!   with trace order. Every requested pending sector remains an L2 lookup
//!   miss but, under the architecture-gated sm_86 policy, joins the matching
//!   allocation-generation/sector's original lower-level fetch. A response
//!   cannot precede its L2-hit lower bound, any newly fetched sector's DRAM
//!   latency/bandwidth completion, or any joined sector's registered original
//!   ready time. The carded aggregate DRAM byte server ([`DramBw`]) can only
//!   push newly issued fetches later than the fixed latency arm.
//! - `L2Fill` — installs a reply-less or mixed-request read fetch at its own
//!   data-ready time, independently of a later joined-sector response.
//! - `AtomicCompletion` — retains the intrinsic return-completion time of an
//!   ATOM whose value has no dependence-barrier observer.
//! - `MemoryWriteAck` — for an explicitly timing-active STORE/RED feedback
//!   candidate, drains the candidate's already-derived fence-visible frontier
//!   and only then publishes it to the issuing warp's `MEMBAR.SC.GPU` state.
//! - `MemResponse` — first completes any fill carried by the same request,
//!   then completes the originating SM's L1 fill (or clears a direct
//!   bypass/untagged barrier unit) at exactly the data-ready time and resumes
//!   that SM. Stale L2 installation never suppresses the SM response.
//!
//! Store and reduction mutation-only requests take no data response. Ordinary
//! replay changes L2 state in arrival order and nothing waits on them; the
//! opt-in feedback completion lane adds the typed acknowledgement above.
//!
//! Determinism: SM actors are created in ascending traced `sm_id` order,
//! all scheduling flows through the `(time, seq)` queue, and L2 lookup and
//! fill mutation occur only in their typed event handlers — two runs over the
//! same inputs produce bit-identical schedules and counters.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::num::NonZeroU64;
use std::rc::Rc;

/// Per-SM lower bounds on responses to requests whose `L2Access` has
/// been SCHEDULED but not yet PROCESSED (so the true response time is
/// not in `response_due` yet). `SmCore::advance`'s internal cap covers
/// requests emitted during ONE call; this map covers the window BETWEEN
/// calls — without it, any event that resumes the SM inside that window
/// (a stale wake, an unrelated response) hands it a horizon that
/// ignores the in-flight requests, and the SM runs issue cycles past
/// its own future responses (stale L1 state, late barrier clears; found
/// by the gate's event-vs-constant differential, seed 133).
/// A `BTreeMap<bound, count>` rather than a heap because entries leave
/// in L2Access-processing order, which is not bound order.
struct UnpricedBounds(Vec<BTreeMap<u64, u32>>);

#[derive(Debug, Clone, Copy)]
enum UnpricedRemoval {
    Decrement { sm: usize, bound: u64, next: u32 },
    Remove { sm: usize, bound: u64 },
}

impl UnpricedRemoval {
    fn commit(self, bounds: &mut UnpricedBounds) {
        match self {
            Self::Decrement { sm, bound, next } => {
                bounds.0[sm].insert(bound, next);
            }
            Self::Remove { sm, bound } => {
                bounds.0[sm].remove(&bound);
            }
        }
    }
}

impl UnpricedBounds {
    fn insert(&mut self, sm: usize, bound: u64) -> Result<(), SystemError> {
        let next = self.0[sm]
            .get(&bound)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(SystemError::CounterOverflow {
                counter: "unpriced response-bound multiplicity",
            })?;
        self.0[sm].insert(bound, next);
        Ok(())
    }

    fn prepare_remove(&self, sm: usize, bound: u64) -> Result<UnpricedRemoval, SystemError> {
        match self.0[sm].get(&bound).copied() {
            Some(n) if n > 1 => Ok(UnpricedRemoval::Decrement {
                sm,
                bound,
                next: n - 1,
            }),
            Some(_) => Ok(UnpricedRemoval::Remove { sm, bound }),
            None => Err(SystemError::UnpricedBoundMissing { sm, bound }),
        }
    }

    fn min(&self, sm: usize) -> Option<u64> {
        self.0[sm].first_key_value().map(|(&t, _)| t)
    }
}

use sim_arch::{ArchId, L2PendingReadSemantics};
use sim_engine::{EngineError, EventQueue, TargetId};
use sim_mem::{
    L1Config, L2Counters, L2DirtyResidency, L2EventMaskPreview, L2Eviction, L2FillToken, L2Model,
    L2PendingSectorKey, L2Source, MemoryOrigin,
};

/// Per-actor pending due-times with the one choreography the driver
/// repeats for responses and wakes: `push` at scheduling, `pop_expect`
/// at the matching event (heap order must equal event order), `next`
/// for horizons/covered checks.
struct DueTimes(Vec<BinaryHeap<Reverse<u64>>>);

impl DueTimes {
    fn new(n: usize) -> Self {
        Self((0..n).map(|_| BinaryHeap::new()).collect())
    }

    fn push(&mut self, ix: usize, t: u64) {
        self.0[ix].push(Reverse(t));
    }

    fn pop_expect(
        &mut self,
        component: &'static str,
        ix: usize,
        t: u64,
    ) -> Result<(), SystemError> {
        let got = self.next(ix);
        if got != Some(t) {
            return Err(SystemError::DueTimeMismatch {
                component,
                sm: ix,
                expected: t,
                got,
            });
        }
        self.0[ix].pop();
        Ok(())
    }

    fn next(&self, ix: usize) -> Option<u64> {
        self.0[ix].peek().map(|&Reverse(t)| t)
    }

    fn is_empty(&self) -> bool {
        self.0.iter().all(BinaryHeap::is_empty)
    }
}

/// One address-space origin's per-[`L2Source`] observer lanes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct L2OriginSourceStats {
    pub sectors: [u64; 4],
    pub hits: [u64; 4],
}

/// Per-[`L2Source`] sector/hit split — the driver-side EMISSION observer
/// (the cache model itself no longer counts provenance; gate-3 altitude
/// note c). Indexed by discriminant order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2SourceSplit {
    arch: Option<ArchId>,
    pub sectors: [u64; 4],
    pub hits: [u64; 4],
    origin_sectors: [[u64; 4]; 2],
    origin_hits: [[u64; 4]; 2],
}

impl Default for L2SourceSplit {
    fn default() -> Self {
        Self {
            arch: None,
            sectors: [0; 4],
            hits: [0; 4],
            origin_sectors: [[0; 4]; 2],
            origin_hits: [[0; 4]; 2],
        }
    }
}

const L2_SOURCES: [L2Source; 4] = [
    L2Source::L1Fill,
    L2Source::Bypass,
    L2Source::Untagged,
    L2Source::Store,
];

impl L2SourceSplit {
    pub fn for_arch(arch: ArchId) -> Self {
        Self {
            arch: Some(arch),
            ..Self::default()
        }
    }

    pub fn arch(&self) -> Option<ArchId> {
        self.arch
    }

    pub fn source(&self, s: L2Source) -> (u64, u64) {
        (self.sectors[s as usize], self.hits[s as usize])
    }

    pub fn source_for_origin(&self, origin: MemoryOrigin, source: L2Source) -> (u64, u64) {
        let origin_ix = origin.index();
        let source_ix = source as usize;
        (
            self.origin_sectors[origin_ix][source_ix],
            self.origin_hits[origin_ix][source_ix],
        )
    }

    pub fn origin(&self, origin: MemoryOrigin) -> L2OriginSourceStats {
        L2OriginSourceStats {
            sectors: self.origin_sectors[origin.index()],
            hits: self.origin_hits[origin.index()],
        }
    }

    /// Build a coherent synthetic observer from explicit origin lanes.
    pub fn from_origins(
        arch: ArchId,
        global: L2OriginSourceStats,
        local: L2OriginSourceStats,
    ) -> Result<Self, SystemCounterError> {
        let mut split = Self::for_arch(arch);
        split.origin_sectors[MemoryOrigin::Global.index()] = global.sectors;
        split.origin_hits[MemoryOrigin::Global.index()] = global.hits;
        split.origin_sectors[MemoryOrigin::Local.index()] = local.sectors;
        split.origin_hits[MemoryOrigin::Local.index()] = local.hits;
        for source in L2_SOURCES {
            let source_ix = source as usize;
            split.sectors[source_ix] = global.sectors[source_ix]
                .checked_add(local.sectors[source_ix])
                .ok_or(SystemCounterError::L2OriginPartitionOverflow {
                    counter: "sectors",
                    l2_source: source,
                    global: global.sectors[source_ix],
                    local: local.sectors[source_ix],
                })?;
            split.hits[source_ix] = global.hits[source_ix]
                .checked_add(local.hits[source_ix])
                .ok_or(SystemCounterError::L2OriginPartitionOverflow {
                    counter: "hits",
                    l2_source: source,
                    global: global.hits[source_ix],
                    local: local.hits[source_ix],
                })?;
        }
        split.validate_origin_partition()?;
        Ok(split)
    }

    /// Explicitly attribute an already-built unified synthetic observer to one
    /// origin. Modeled requests always mutate an explicit lane directly.
    pub fn with_unified_origin(mut self, origin: MemoryOrigin) -> Self {
        self.origin_sectors = [[0; 4]; 2];
        self.origin_hits = [[0; 4]; 2];
        self.origin_sectors[origin.index()] = self.sectors;
        self.origin_hits[origin.index()] = self.hits;
        self
    }

    pub fn validate_origin_partition(&self) -> Result<(), SystemCounterError> {
        for source in L2_SOURCES {
            let source_ix = source as usize;
            let global_sectors = self.origin_sectors[MemoryOrigin::Global.index()][source_ix];
            let local_sectors = self.origin_sectors[MemoryOrigin::Local.index()][source_ix];
            let global_hits = self.origin_hits[MemoryOrigin::Global.index()][source_ix];
            let local_hits = self.origin_hits[MemoryOrigin::Local.index()][source_ix];
            if global_hits > global_sectors {
                return Err(SystemCounterError::L2OriginHitExceedsSectors {
                    origin: MemoryOrigin::Global,
                    l2_source: source,
                    hits: global_hits,
                    sectors: global_sectors,
                });
            }
            if local_hits > local_sectors {
                return Err(SystemCounterError::L2OriginHitExceedsSectors {
                    origin: MemoryOrigin::Local,
                    l2_source: source,
                    hits: local_hits,
                    sectors: local_sectors,
                });
            }
            let sectors = global_sectors.checked_add(local_sectors).ok_or(
                SystemCounterError::L2OriginPartitionOverflow {
                    counter: "sectors",
                    l2_source: source,
                    global: global_sectors,
                    local: local_sectors,
                },
            )?;
            if self.sectors[source_ix] != sectors {
                return Err(SystemCounterError::L2OriginPartitionMismatch {
                    counter: "sectors",
                    l2_source: source,
                    unified: self.sectors[source_ix],
                    global: global_sectors,
                    local: local_sectors,
                });
            }

            let hits = global_hits.checked_add(local_hits).ok_or(
                SystemCounterError::L2OriginPartitionOverflow {
                    counter: "hits",
                    l2_source: source,
                    global: global_hits,
                    local: local_hits,
                },
            )?;
            if self.hits[source_ix] != hits {
                return Err(SystemCounterError::L2OriginPartitionMismatch {
                    counter: "hits",
                    l2_source: source,
                    unified: self.hits[source_ix],
                    global: global_hits,
                    local: local_hits,
                });
            }
            if self.hits[source_ix] > self.sectors[source_ix] {
                return Err(SystemCounterError::L2SourceHitExceedsSectors {
                    l2_source: source,
                    hits: self.hits[source_ix],
                    sectors: self.sectors[source_ix],
                });
            }
        }
        Ok(())
    }

    pub fn add_checked(&mut self, other: &L2SourceSplit) -> Result<(), SystemCounterError> {
        let expected = self
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L2 source split",
            })?;
        let got = other
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L2 source split",
            })?;
        if expected != got {
            return Err(sim_arch::ArchitectureProvenanceError::Mismatch {
                component: "L2 source split",
                expected,
                got,
            }
            .into());
        }
        let mut sectors = [0; 4];
        let mut hits = [0; 4];
        for i in 0..4 {
            sectors[i] = self.sectors[i].checked_add(other.sectors[i]).ok_or(
                SystemCounterError::Overflow {
                    counter: "L2 source sectors",
                },
            )?;
            hits[i] =
                self.hits[i]
                    .checked_add(other.hits[i])
                    .ok_or(SystemCounterError::Overflow {
                        counter: "L2 source hits",
                    })?;
        }
        let mut origin_sectors = [[0; 4]; 2];
        let mut origin_hits = [[0; 4]; 2];
        for origin in MemoryOrigin::ALL {
            let origin_ix = origin.index();
            for source_ix in 0..4 {
                origin_sectors[origin_ix][source_ix] = self.origin_sectors[origin_ix][source_ix]
                    .checked_add(other.origin_sectors[origin_ix][source_ix])
                    .ok_or(SystemCounterError::Overflow {
                        counter: "L2 origin/source sectors",
                    })?;
                origin_hits[origin_ix][source_ix] = self.origin_hits[origin_ix][source_ix]
                    .checked_add(other.origin_hits[origin_ix][source_ix])
                    .ok_or(SystemCounterError::Overflow {
                        counter: "L2 origin/source hits",
                    })?;
            }
        }
        let next = Self {
            arch: self.arch,
            sectors,
            hits,
            origin_sectors,
            origin_hits,
        };
        self.validate_origin_partition()?;
        other.validate_origin_partition()?;
        next.validate_origin_partition()?;
        *self = next;
        Ok(())
    }
}

use crate::issue::{
    AccessKind, AdmissionRequestWitness, Advance, BlockReplay, IssueError, IssueModelParams,
    MemRequest, MemoryAdmissionCandidate, MemoryAdmissionCandidateState, MemoryOperation,
    MemoryServiceCandidate, MemoryWriteAck, ReplayHorizon, ReplyTo, SharedMemoryResidency, SmCore,
    SmMemMode, SmSimResult, TailFlags, checked_cycle_add, stateful_write_lookup_observer_enabled,
    validate_memory_admission_candidates, validate_memory_service_candidates,
};
use crate::program::FunctionSchedule;
use sim_mem::L1WriteLookupObserver;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SystemCounterError {
    #[error(transparent)]
    Architecture(#[from] sim_arch::ArchitectureProvenanceError),
    #[error("{counter} counter overflows u64")]
    Overflow { counter: &'static str },
    #[error(
        "L2 {l2_source:?} origin partition for {counter} overflows u64: Global={global}, Local={local}"
    )]
    L2OriginPartitionOverflow {
        counter: &'static str,
        l2_source: L2Source,
        global: u64,
        local: u64,
    },
    #[error(
        "L2 {l2_source:?} origin partition mismatch for {counter}: unified={unified}, Global={global}, Local={local}"
    )]
    L2OriginPartitionMismatch {
        counter: &'static str,
        l2_source: L2Source,
        unified: u64,
        global: u64,
        local: u64,
    },
    #[error(
        "L2 {l2_source:?} hit count exceeds unified sector count: hits={hits}, sectors={sectors}"
    )]
    L2SourceHitExceedsSectors {
        l2_source: L2Source,
        hits: u64,
        sectors: u64,
    },
    #[error(
        "L2 {origin:?}/{l2_source:?} hit count exceeds origin sector count: hits={hits}, sectors={sectors}"
    )]
    L2OriginHitExceedsSectors {
        origin: MemoryOrigin,
        l2_source: L2Source,
        hits: u64,
        sectors: u64,
    },
    #[error("DRAM origin partition for {counter} overflows u64: Global={global}, Local={local}")]
    DramOriginPartitionOverflow {
        counter: &'static str,
        global: u64,
        local: u64,
    },
    #[error(
        "DRAM origin partition mismatch for {counter}: unified={unified}, Global={global}, Local={local}"
    )]
    DramOriginPartitionMismatch {
        counter: &'static str,
        unified: u64,
        global: u64,
        local: u64,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum SystemError {
    #[error(transparent)]
    Counters(#[from] SystemCounterError),
    #[error(transparent)]
    Issue(#[from] IssueError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    L2PendingReadCapability(#[from] sim_arch::CapabilityError),
    #[error(transparent)]
    L2Snapshot(#[from] sim_mem::L2SnapshotError),
    #[error(transparent)]
    L2(#[from] sim_mem::L2AccessError),
    #[error(
        "event queue drained with sm {sm_id} unfinished ({outstanding} response(s) recorded \
         outstanding) — request/response wiring invariant broken"
    )]
    Deadlock { sm_id: i32, outstanding: usize },
    #[error(
        "event-loop livelock at t={time}: {idle_wakes} consecutive SM wakes issued nothing, \
         emitted nothing and left the clock frozen — horizon/wake wiring invariant broken"
    )]
    Livelock { time: u64, idle_wakes: u64 },
    #[error("system architecture mismatch for {component}: expected {expected}, got {got}")]
    ArchitectureMismatch {
        component: &'static str,
        expected: ArchId,
        got: ArchId,
    },

    #[error(
        "latency ladder inverted: the DRAM arm ({dram}) prices below the L2-hit arm ({l2_hit}) \
         — the run-ahead causality contract takes the L2-hit arm as the response lower bound"
    )]
    LatencyLadderInverted { l2_hit: u64, dram: u64 },
    #[error(
        "DRAM bandwidth config invalid (bytes_per_sec={bytes_per_sec}, sm_clock_hz={sm_clock_hz}) \
         — zero rates must fail closed, never be clamped into an absurd or inert server"
    )]
    DramBandwidthInvalid {
        bytes_per_sec: u64,
        sm_clock_hz: u64,
    },
    #[error("DRAM fixed-point {operation} overflows u128")]
    DramFixedPointOverflow { operation: &'static str },
    #[error(
        "DRAM completion {completion_fp}/{bytes_per_sec} exceeds the u64 cycle domain after ceiling division"
    )]
    DramCycleOverflow {
        completion_fp: u128,
        bytes_per_sec: u128,
    },
    #[error("SM actor count {count} exceeds the u32 target-id domain")]
    ActorCountOutOfRange { count: usize },
    #[error("event target {target} is outside the {actor_count}-actor system")]
    EventTargetOutOfRange { target: u32, actor_count: usize },
    #[error(
        "{component} due-time mismatch for SM index {sm}: expected event t={expected}, queued head was {got:?}"
    )]
    DueTimeMismatch {
        component: &'static str,
        sm: usize,
        expected: u64,
        got: Option<u64>,
    },
    #[error(
        "L2 request timestamp {issue_time} does not match its event timestamp {event_time} for SM index {sm}"
    )]
    RequestTimeMismatch {
        sm: usize,
        issue_time: u64,
        event_time: u64,
    },
    #[error("L2 request for sector {sector_id} has invalid sector mask {sector_mask:#06b}")]
    SectorMaskInvalid { sector_id: i64, sector_mask: u8 },
    #[error(
        "L2 request for {origin:?}-origin anchor sector {sector_id} has incoherent Local mask {local_sector_mask:#06b} within request mask {sector_mask:#06b}"
    )]
    OriginMaskInvalid {
        sector_id: i64,
        origin: MemoryOrigin,
        sector_mask: u8,
        local_sector_mask: u8,
    },
    #[error(
        "L2 request for sector {sector_id} has incoherent source/operation/reply contract: source={request_source:?}, operation={operation:?}, reply={reply:?}"
    )]
    MemoryRequestContractInvalid {
        sector_id: i64,
        request_source: L2Source,
        operation: MemoryOperation,
        reply: ReplyTo,
    },
    #[error(
        "memory-admission observation on SM {sm_id} interleaved group warp={warp} instruction={instruction} before the prior group completed"
    )]
    AdmissionObservationInterleaved {
        sm_id: i32,
        warp: u32,
        instruction: u32,
    },
    #[error(
        "memory-admission observation on SM {sm_id} expected request index {expected}, got {got} for warp={warp} instruction={instruction}"
    )]
    AdmissionObservationSequence {
        sm_id: i32,
        warp: u32,
        instruction: u32,
        expected: u16,
        got: u16,
    },
    #[error(
        "memory-admission observation on SM {sm_id} changed group identity or operation at warp={warp} instruction={instruction}"
    )]
    AdmissionObservationIdentity {
        sm_id: i32,
        warp: u32,
        instruction: u32,
    },
    #[error("memory-admission L2 observation on SM {sm_id} has {hits} hits for {sectors} sectors")]
    AdmissionObservationHits { sm_id: i32, sectors: u64, hits: u64 },
    #[error("event queue drained with an incomplete memory-admission group on SM {sm_id}")]
    AdmissionObservationIncomplete { sm_id: i32 },
    #[error(
        "event queue drained with {pending} modeled write acknowledgement(s) pending on SM {sm_id}"
    )]
    MemoryWriteAckPending { sm_id: i32, pending: usize },
    #[error("event queue drained with modeled write-ack due times still registered")]
    MemoryWriteAckDueNotDrained,
    #[error(
        "L2 fill schedule mismatch for sector {sector_id}: token={has_token}, ready={has_ready}"
    )]
    FillScheduleMismatch {
        sector_id: i64,
        has_token: bool,
        has_ready: bool,
    },
    #[error("unpriced response bound {bound} for SM index {sm} is absent")]
    UnpricedBoundMissing { sm: usize, bound: u64 },
    #[error("response ready time {ready} precedes lower bound {bound} for SM index {sm}")]
    ResponseBeforeBound { sm: usize, ready: u64, bound: u64 },
    #[error(
        "response ready time {response_ready} precedes L2 fill ready time {fill_ready} for sector {sector_id}"
    )]
    ResponseBeforeFill {
        sector_id: i64,
        response_ready: u64,
        fill_ready: u64,
    },
    #[error("pending L2 sector {key:?} has no registered lower-level ready time")]
    PendingL2ReadyMissing { key: L2PendingSectorKey },
    #[error("pending L2 sector {key:?} has no ready-time entry to remove")]
    PendingL2ReadyRemovalMissing { key: L2PendingSectorKey },
    #[error(
        "pending L2 sector {key:?} already has lower-level ready time {ready} during registration"
    )]
    PendingL2ReadyAlreadyRegistered { key: L2PendingSectorKey, ready: u64 },
    #[error(
        "pending L2 sector {key:?} completed at t={got}, but its registered ready time is t={expected}"
    )]
    PendingL2ReadyTimeMismatch {
        key: L2PendingSectorKey,
        expected: u64,
        got: u64,
    },
    #[error("event queue drained with {entries} pending L2 sector ready-time entries")]
    PendingL2ReadyNotDrained { entries: usize },
    #[error("{counter} counter overflows u64")]
    CounterOverflow { counter: &'static str },
}

fn require_arch(expected: ArchId, got: ArchId, component: &'static str) -> Result<(), SystemError> {
    if got != expected {
        return Err(SystemError::ArchitectureMismatch {
            component,
            expected,
            got,
        });
    }
    Ok(())
}

fn require_l2_pending_read_semantics(arch: ArchId) -> Result<(), SystemError> {
    match arch
        .l2_pending_read_semantics()
        .require("l2_pending_read_semantics")?
    {
        L2PendingReadSemantics::AllocationSectorMergeV1 => Ok(()),
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct TestRequestMaskOverride {
    origin: MemoryOrigin,
    sector_mask: u8,
    local_sector_mask: u8,
}

#[cfg(test)]
thread_local! {
    static TEST_L2_COUNTER_ARCH_OVERRIDE: std::cell::Cell<Option<ArchId>> =
        const { std::cell::Cell::new(None) };
    static TEST_L2_ACCESS_ERROR_AFTER: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    static TEST_REQUEST_MASK_OVERRIDE: std::cell::Cell<Option<TestRequestMaskOverride>> =
        const { std::cell::Cell::new(None) };
    static TEST_REQUEST_OPERATION_OVERRIDE: std::cell::Cell<Option<MemoryOperation>> =
        const { std::cell::Cell::new(None) };
}

fn checked_l2_counters<'a>(
    expected: ArchId,
    l2: &'a L2Model,
) -> Result<&'a L2Counters, SystemError> {
    let counters = l2.counters().map_err(sim_mem::L2AccessError::Counter)?;
    #[cfg(test)]
    let got = TEST_L2_COUNTER_ARCH_OVERRIDE
        .with(|override_arch| override_arch.get())
        .or_else(|| counters.arch());
    #[cfg(not(test))]
    let got = counters.arch();
    let got = got.ok_or_else(|| {
        sim_mem::L2AccessError::Counter(sim_mem::L2CounterError::Architecture(
            sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L2 counters",
            },
        ))
    })?;
    if got != expected {
        return Err(
            sim_mem::L2AccessError::Counter(sim_mem::L2CounterError::Architecture(
                sim_arch::ArchitectureProvenanceError::Mismatch {
                    component: "L2 counters",
                    expected,
                    got,
                },
            ))
            .into(),
        );
    }
    Ok(counters)
}

#[cfg(test)]
fn maybe_override_request_masks(req: &mut MemRequest) {
    TEST_REQUEST_MASK_OVERRIDE.with(|value| {
        if let Some(override_value) = value.replace(None) {
            req.origin = override_value.origin;
            req.sector_mask = override_value.sector_mask;
            req.local_sector_mask = override_value.local_sector_mask;
        }
    });
    TEST_REQUEST_OPERATION_OVERRIDE.with(|value| {
        if let Some(operation) = value.replace(None) {
            req.operation = operation;
        }
    });
}

#[cfg(test)]
fn maybe_inject_l2_access_error() -> Result<(), SystemError> {
    TEST_L2_ACCESS_ERROR_AFTER.with(|remaining| match remaining.get() {
        Some(0) => {
            remaining.set(None);
            Err(
                sim_mem::L2AccessError::Counter(sim_mem::L2CounterError::Overflow {
                    field: sim_mem::L2CounterField::Hits,
                })
                .into(),
            )
        }
        Some(count) => {
            remaining.set(Some(count - 1));
            Ok(())
        }
        None => Ok(()),
    })
}

/// Backstop against event-loop LIVELOCK — a regression class this driver
/// actually shipped once (same-time SmWake ping-pong, before the horizon
/// contract was narrowed to an SM's own responses). The signature is
/// unmistakable: SmWake events that issue nothing and emit nothing while
/// the clock stays frozen, forever. In any healthy run a parked SM
/// yields such an idle wake at most once per REAL event at that
/// timestamp (a response or L2 access resets the streak, and a
/// future-time wake advances the clock), so consecutive idle wakes are
/// bounded by the SM count; crossing `2 × cores + 8` proves a broken
/// invariant. Fail closed in milliseconds — plain `cargo test` has no
/// timeout, and the pre-fix binaries spun at full CPU for hours.
struct SpinGuard {
    limit: u64,
    last_time: u64,
    idle_wakes: u64,
}

impl SpinGuard {
    fn new(cores: usize) -> Result<Self, SystemError> {
        let cores =
            u64::try_from(cores).map_err(|_| SystemError::ActorCountOutOfRange { count: cores })?;
        let limit = cores.checked_mul(2).and_then(|n| n.checked_add(8)).ok_or(
            SystemError::CounterOverflow {
                counter: "spin-guard limit",
            },
        )?;
        Ok(Self {
            limit,
            last_time: 0,
            idle_wakes: 0,
        })
    }

    fn on_progress_event(&mut self, time: u64) {
        self.last_time = time;
        self.idle_wakes = 0;
    }

    /// Record one processed event. `idle_wake` marks an `SmWake` whose
    /// actor neither issued an instruction nor emitted a request; every
    /// other event kind (and any clock advance) resets the streak.
    fn on_event(&mut self, time: u64, idle_wake: bool) -> Result<(), SystemError> {
        if time != self.last_time {
            self.last_time = time;
            self.idle_wakes = 0;
        }
        if !idle_wake {
            self.idle_wakes = 0;
            return Ok(());
        }
        self.idle_wakes = self
            .idle_wakes
            .checked_add(1)
            .ok_or(SystemError::CounterOverflow {
                counter: "consecutive idle wakes",
            })?;
        if self.idle_wakes > self.limit {
            return Err(SystemError::Livelock {
                time,
                idle_wakes: self.idle_wakes,
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
enum Ev {
    SmWake,
    L2Access(MemRequest),
    L2AccessBatch(Box<[MemRequest]>),
    L2Fill(L2FillToken),
    AtomicCompletion,
    MemoryWriteAck(MemoryWriteAck),
    MemResponse {
        reply: ReplyTo,
        fill: Option<L2FillToken>,
    },
}

enum L2AccessEventRequests {
    Single([MemRequest; 1]),
    Batch(Box<[MemRequest]>),
}

impl AsRef<[MemRequest]> for L2AccessEventRequests {
    fn as_ref(&self) -> &[MemRequest] {
        match self {
            Self::Single(request) => request,
            Self::Batch(requests) => requests,
        }
    }
}

#[cfg(feature = "performance-counters")]
#[derive(Debug, Default)]
struct EventPerformanceCounters {
    queue_events_processed: u64,
    processed: u64,
    sm_wake: u64,
    l2_access: u64,
    l2_access_batches: u64,
    l2_access_batched_requests: u64,
    l2_fill: u64,
    atomic_completion: u64,
    memory_write_ack: u64,
    mem_response: u64,
    same_cycle_after_first: u64,
    queue_same_cycle_after_first: u64,
    maximum_queue_depth: u64,
    previous_time: Option<u64>,
}

#[cfg(feature = "performance-counters")]
impl EventPerformanceCounters {
    fn observe(
        &mut self,
        event: &Ev,
        time: u64,
        queue_depth_before_pop: usize,
        logical_events: usize,
    ) -> Result<(), SystemError> {
        self.queue_events_processed =
            self.queue_events_processed
                .checked_add(1)
                .ok_or(SystemError::CounterOverflow {
                    counter: "performance queue events processed",
                })?;
        let logical_events =
            u64::try_from(logical_events).map_err(|_| SystemError::CounterOverflow {
                counter: "performance logical event multiplicity",
            })?;
        self.processed =
            self.processed
                .checked_add(logical_events)
                .ok_or(SystemError::CounterOverflow {
                    counter: "performance events processed",
                })?;
        match event {
            Ev::L2AccessBatch(_) => {
                self.l2_access_batches =
                    self.l2_access_batches
                        .checked_add(1)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "performance L2 access batches",
                        })?;
                self.l2_access_batched_requests = self
                    .l2_access_batched_requests
                    .checked_add(logical_events)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "performance batched L2 access requests",
                    })?;
                self.l2_access = self.l2_access.checked_add(logical_events).ok_or(
                    SystemError::CounterOverflow {
                        counter: "performance L2 access requests",
                    },
                )?;
            }
            Ev::L2Access(_) => {
                debug_assert_eq!(logical_events, 1);
                self.l2_access =
                    self.l2_access
                        .checked_add(1)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "performance L2 access requests",
                        })?;
            }
            Ev::SmWake
            | Ev::L2Fill(_)
            | Ev::AtomicCompletion
            | Ev::MemoryWriteAck(_)
            | Ev::MemResponse { .. } => {
                debug_assert_eq!(logical_events, 1);
                let kind = match event {
                    Ev::SmWake => &mut self.sm_wake,
                    Ev::L2Fill(_) => &mut self.l2_fill,
                    Ev::AtomicCompletion => &mut self.atomic_completion,
                    Ev::MemoryWriteAck(_) => &mut self.memory_write_ack,
                    Ev::MemResponse { .. } => &mut self.mem_response,
                    Ev::L2Access(_) | Ev::L2AccessBatch(_) => unreachable!(),
                };
                *kind = kind.checked_add(1).ok_or(SystemError::CounterOverflow {
                    counter: "performance event-kind count",
                })?;
            }
        }
        if self.previous_time == Some(time) {
            self.queue_same_cycle_after_first = self
                .queue_same_cycle_after_first
                .checked_add(1)
                .ok_or(SystemError::CounterOverflow {
                    counter: "performance same-cycle queue events",
                })?;
            self.same_cycle_after_first = self
                .same_cycle_after_first
                .checked_add(logical_events)
                .ok_or(SystemError::CounterOverflow {
                counter: "performance same-cycle events",
            })?;
        } else {
            self.same_cycle_after_first = self
                .same_cycle_after_first
                .checked_add(logical_events.saturating_sub(1))
                .ok_or(SystemError::CounterOverflow {
                    counter: "performance same-cycle events",
                })?;
        }
        self.previous_time = Some(time);
        self.maximum_queue_depth =
            self.maximum_queue_depth
                .max(u64::try_from(queue_depth_before_pop).map_err(|_| {
                    SystemError::CounterOverflow {
                        counter: "performance event-queue depth",
                    }
                })?);
        Ok(())
    }
}

#[derive(Default)]
struct PendingL2Ready(HashMap<L2PendingSectorKey, u64>);

impl PendingL2Ready {
    fn merged_ready(&self, preview: L2EventMaskPreview) -> Result<Option<u64>, SystemError> {
        let mut latest = None;
        for sector in 0..4 {
            let Some(key) = preview.pending_sector_key(sector) else {
                continue;
            };
            let ready = self
                .0
                .get(&key)
                .copied()
                .ok_or(SystemError::PendingL2ReadyMissing { key })?;
            latest = Some(latest.map_or(ready, |current: u64| current.max(ready)));
        }
        Ok(latest)
    }

    fn preflight_register(&self, preview: L2EventMaskPreview) -> Result<(), SystemError> {
        for sector in 0..4 {
            let Some(key) = preview.fill_sector_key(sector) else {
                continue;
            };
            if let Some(&ready) = self.0.get(&key) {
                return Err(SystemError::PendingL2ReadyAlreadyRegistered { key, ready });
            }
        }
        Ok(())
    }

    fn register(&mut self, fill: L2FillToken, ready: u64) -> Result<(), SystemError> {
        for sector in 0..4 {
            let Some(key) = fill.pending_sector_key(sector) else {
                continue;
            };
            if let Some(&existing_ready) = self.0.get(&key) {
                return Err(SystemError::PendingL2ReadyAlreadyRegistered {
                    key,
                    ready: existing_ready,
                });
            }
        }
        for sector in 0..4 {
            if let Some(key) = fill.pending_sector_key(sector) {
                self.0.insert(key, ready);
            }
        }
        Ok(())
    }

    fn preflight_remove(&self, fill: L2FillToken, now: u64) -> Result<(), SystemError> {
        for sector in 0..4 {
            let Some(key) = fill.pending_sector_key(sector) else {
                continue;
            };
            let ready = self
                .0
                .get(&key)
                .copied()
                .ok_or(SystemError::PendingL2ReadyRemovalMissing { key })?;
            if ready != now {
                return Err(SystemError::PendingL2ReadyTimeMismatch {
                    key,
                    expected: ready,
                    got: now,
                });
            }
        }
        Ok(())
    }

    fn commit_remove(&mut self, fill: L2FillToken) {
        for sector in 0..4 {
            if let Some(key) = fill.pending_sector_key(sector) {
                let removed = self.0.remove(&key);
                debug_assert!(removed.is_some(), "pending L2 removal was preflighted");
            }
        }
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

struct L2ResponseCommit {
    ready: u64,
    origin_ix: usize,
    next_delayed: u64,
    next_origin_delayed: u64,
    unpriced_removal: UnpricedRemoval,
}

#[derive(Debug, Clone)]
struct PendingAdmissionInstruction {
    witness: AdmissionRequestWitness,
    operation: MemoryOperation,
    issue_time: u64,
    next_request_index: u16,
    hit_sectors: u64,
    miss_sectors: u64,
    regions: Vec<i64>,
}

#[derive(Debug, Clone, Copy)]
struct CompletedAdmissionInstruction {
    witness: AdmissionRequestWitness,
    operation: MemoryOperation,
    issue_time: u64,
    hit_sectors: u64,
    miss_sectors: u64,
    l2_requests: u64,
    unique_regions: u64,
}

#[derive(Default)]
struct AdmissionInstructionObserver {
    pending: Option<PendingAdmissionInstruction>,
}

impl AdmissionInstructionObserver {
    #[allow(clippy::too_many_arguments)]
    fn observe(
        &mut self,
        sm_id: i32,
        witness: AdmissionRequestWitness,
        operation: MemoryOperation,
        issue_time: u64,
        sectors: u64,
        hits: u64,
        sector_id: i64,
    ) -> Result<Option<CompletedAdmissionInstruction>, SystemError> {
        if hits > sectors {
            return Err(SystemError::AdmissionObservationHits {
                sm_id,
                sectors,
                hits,
            });
        }
        if witness.request_index() == 0 {
            if self.pending.is_some() {
                return Err(SystemError::AdmissionObservationInterleaved {
                    sm_id,
                    warp: witness.warp(),
                    instruction: witness.instruction(),
                });
            }
            self.pending = Some(PendingAdmissionInstruction {
                witness,
                operation,
                issue_time,
                next_request_index: 0,
                hit_sectors: 0,
                miss_sectors: 0,
                regions: Vec::new(),
            });
        }
        let pending = self
            .pending
            .as_mut()
            .ok_or(SystemError::AdmissionObservationSequence {
                sm_id,
                warp: witness.warp(),
                instruction: witness.instruction(),
                expected: 0,
                got: witness.request_index(),
            })?;
        if pending.witness.warp() != witness.warp()
            || pending.witness.instruction() != witness.instruction()
            || pending.witness.subcore() != witness.subcore()
            || pending.witness.request_count() != witness.request_count()
            || pending.operation != operation
            || pending.issue_time != issue_time
        {
            return Err(SystemError::AdmissionObservationIdentity {
                sm_id,
                warp: witness.warp(),
                instruction: witness.instruction(),
            });
        }
        if pending.next_request_index != witness.request_index() {
            return Err(SystemError::AdmissionObservationSequence {
                sm_id,
                warp: witness.warp(),
                instruction: witness.instruction(),
                expected: pending.next_request_index,
                got: witness.request_index(),
            });
        }
        pending.hit_sectors =
            pending
                .hit_sectors
                .checked_add(hits)
                .ok_or(SystemError::CounterOverflow {
                    counter: "memory-admission observed hit sectors",
                })?;
        pending.miss_sectors = pending.miss_sectors.checked_add(sectors - hits).ok_or(
            SystemError::CounterOverflow {
                counter: "memory-admission observed miss sectors",
            },
        )?;
        let region = sector_id.div_euclid(8);
        if !pending.regions.contains(&region) {
            pending.regions.push(region);
        }
        pending.next_request_index =
            pending
                .next_request_index
                .checked_add(1)
                .ok_or(SystemError::CounterOverflow {
                    counter: "memory-admission request index",
                })?;
        if pending.next_request_index == witness.request_count() {
            let completed = self
                .pending
                .take()
                .expect("completed admission observation owns pending state");
            return Ok(Some(CompletedAdmissionInstruction {
                witness: completed.witness,
                operation: completed.operation,
                issue_time: completed.issue_time,
                hit_sectors: completed.hit_sectors,
                miss_sectors: completed.miss_sectors,
                l2_requests: u64::from(completed.next_request_index),
                unique_regions: u64::try_from(completed.regions.len()).map_err(|_| {
                    SystemError::CounterOverflow {
                        counter: "memory-admission unique completion regions",
                    }
                })?,
            }));
        }
        Ok(None)
    }
}

struct L2SystemCommit {
    source_ix: usize,
    source_sectors: u64,
    source_hits: u64,
    origin_source_sectors: [u64; 2],
    origin_source_hits: [u64; 2],
    read_sectors: u64,
    origin_read_sectors: [u64; 2],
    topology_stats: DramStats,
    dram_free_fp: Option<u128>,
    fill_ready: Option<u64>,
    atomic_completion_ready: Option<u64>,
    response: Option<L2ResponseCommit>,
    observed_sectors: u64,
    observed_hits: u64,
    /// Observer-only read/atomic contribution. Store/reduction packetization
    /// is reconstructed from the issuing instruction topology after the event
    /// stream drains, so it never changes physical L2 state or response time.
    lookup_observer_requests: u64,
    lookup_observer_hits: u64,
}

/// Aggregate DRAM read-bandwidth model (the §6.3 "DRAM
/// bandwidth-latency" increment). The carded covariate is the vendor
/// peak at the LOCKED operating point (`ArchCard.dram_bandwidth` —
/// bytes/second over the locked SM clock, ~747 B/SM-cycle on the
/// 3090 Ti); the model is one G/D/1 byte server over the shared hierarchy
/// read-miss sector stream: each fetching sector demands 32 B, and a
/// reply-carrying request or intrinsically returning ATOM cannot complete
/// before BOTH its latency arm and the server's completion of ITS OWN request's
/// bytes. Bytes of reply-less read follow-up fetches (late merges) are served
/// but price no response — the merged waiter still clears on the original
/// response, so under congestion a follow-up's bytes may complete
/// after its waiter cleared (the documented merge-timing
/// approximation; the cost surfaces on later requests). Uncongested
/// streams see pure latency (the server only ever ADDS delay).
///
/// v0 scope: DRAM READS only — write data validates Store/RED sectors without
/// a fetch, while ATOM's returning RMW read fetches missing sectors. Dirty
/// writeback, RED's RMW read, and coherence remain typed unmodeled tails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DramBw {
    /// Latency-only (pre-increment behavior): the ablation mode and the
    /// event-vs-const equivalence-differential mode — the constant path
    /// has no bandwidth concept, so equivalence is proven here.
    Unmodeled { arch: ArchId },
    /// Carded aggregate service rate at the locked operating point.
    Carded {
        arch: ArchId,
        bytes_per_sec: u64,
        sm_clock_hz: u64,
    },
}

impl DramBw {
    pub const fn arch(self) -> ArchId {
        match self {
            Self::Unmodeled { arch } | Self::Carded { arch, .. } => arch,
        }
    }
}

/// The G/D/1 byte server. Fixed-point time in units of
/// `cycle × bytes_per_sec` (u128): serving `b` bytes takes
/// `b × sm_clock_hz` fp-units (= b × Hz / Bps cycles), so no division
/// happens until the final ceil back to cycles — bit-exact and
/// deterministic in queue order.
struct DramServer {
    free_fp: u128,
    bytes_per_sec: u128,
    sm_clock_hz: u128,
}

impl DramServer {
    /// The caller (`simulate_kernel_shared`) rejects zero rates with a
    /// typed error before construction — no clamping here (a clamp
    /// would turn a bad card into a silently absurd 1 B/s server and
    /// arm the fixed-point u64 ceiling on large traces; gate finding
    /// 2026-07-07).
    fn new(bw: DramBw) -> Option<Self> {
        match bw {
            DramBw::Unmodeled { .. } => None,
            DramBw::Carded {
                bytes_per_sec,
                sm_clock_hz,
                ..
            } => Some(Self {
                free_fp: 0,
                bytes_per_sec: u128::from(bytes_per_sec),
                sm_clock_hz: u128::from(sm_clock_hz),
            }),
        }
    }

    /// Compute a service completion without mutating the server. The
    /// caller uses this to preflight the request's exact outcome-dependent
    /// fetch before any L2 or accounting state commits.
    fn preview(&self, arrival: u64, bytes: u64) -> Result<(u128, u64), SystemError> {
        let arrival_fp = u128::from(arrival).checked_mul(self.bytes_per_sec).ok_or(
            SystemError::DramFixedPointOverflow {
                operation: "arrival scaling",
            },
        )?;
        let service_fp = u128::from(bytes).checked_mul(self.sm_clock_hz).ok_or(
            SystemError::DramFixedPointOverflow {
                operation: "service scaling",
            },
        )?;
        let start = arrival_fp.max(self.free_fp);
        let done = start
            .checked_add(service_fp)
            .ok_or(SystemError::DramFixedPointOverflow {
                operation: "service completion",
            })?;
        let ready_fp = done.div_ceil(self.bytes_per_sec);
        let ready = u64::try_from(ready_fp).map_err(|_| SystemError::DramCycleOverflow {
            completion_fp: done,
            bytes_per_sec: self.bytes_per_sec,
        })?;
        Ok((done, ready))
    }
}

/// One address-space origin's DRAM observer lane.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DramOriginStats {
    pub read_sectors: u64,
    pub bw_delayed_responses: u64,
}

/// Driver-side DRAM traffic and timing-neutral lower-level topology observer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DramStats {
    /// Architecture provenance bound by the shared-L2 system. Unbound
    /// synthetic stats are rejected before aggregation or emission.
    arch: Option<ArchId>,
    /// 32 B DRAM read fetches DEMANDED by the run (L2-missing sectors
    /// of read operations plus missing sectors of returning ATOM RMW reads.
    /// Counted in BOTH modes: under `Unmodeled` it audits the ablation's demand
    /// with no server attached. Plain stores and non-returning RED never enter
    /// this modeled read-fetch count.
    pub read_sectors: u64,
    /// Responses whose ready time the bandwidth server pushed PAST the
    /// latency arm (0 in uncongested runs and in `Unmodeled` mode).
    /// Attribution caveat: reply-less follow-up fetches add bytes but
    /// price nothing, so under congestion their service cost lands on
    /// whichever reply-carrying request queues next — this counter
    /// localizes congestion in TIME, not to the causing warp.
    pub bw_delayed_responses: u64,
    /// L2 replacement topology. Dirty entries are candidate writeback work,
    /// not scheduled or completed DRAM transactions.
    pub l2_evicted_lines: u64,
    pub l2_evicted_valid_sectors: u64,
    pub l2_evicted_pending_lines: u64,
    pub l2_evicted_pending_sectors: u64,
    pub l2_dirty_eviction_lines: u64,
    pub l2_dirty_evicted_sectors: u64,
    /// Maximum terminal dirty residency across the launches aggregated into
    /// this observer. A gauge is never summed as traffic.
    pub maximum_terminal_l2_dirty_lines: u64,
    pub maximum_terminal_l2_dirty_sectors: u64,
    by_origin: [DramOriginStats; 2],
}

impl Default for DramStats {
    fn default() -> Self {
        Self {
            arch: None,
            read_sectors: 0,
            bw_delayed_responses: 0,
            l2_evicted_lines: 0,
            l2_evicted_valid_sectors: 0,
            l2_evicted_pending_lines: 0,
            l2_evicted_pending_sectors: 0,
            l2_dirty_eviction_lines: 0,
            l2_dirty_evicted_sectors: 0,
            maximum_terminal_l2_dirty_lines: 0,
            maximum_terminal_l2_dirty_sectors: 0,
            by_origin: [DramOriginStats::default(); 2],
        }
    }
}

impl DramStats {
    pub fn for_arch(arch: ArchId) -> Self {
        Self {
            arch: Some(arch),
            ..Self::default()
        }
    }

    pub fn arch(&self) -> Option<ArchId> {
        self.arch
    }

    pub fn origin(&self, origin: MemoryOrigin) -> DramOriginStats {
        self.by_origin[origin.index()]
    }

    fn with_l2_eviction(mut self, eviction: L2Eviction) -> Result<Self, SystemError> {
        macro_rules! add {
            ($field:ident, $value:expr, $name:literal) => {
                self.$field = self
                    .$field
                    .checked_add($value)
                    .ok_or(SystemError::CounterOverflow { counter: $name })?;
            };
        }
        add!(l2_evicted_lines, 1, "L2 evicted lines");
        add!(
            l2_evicted_valid_sectors,
            u64::from(eviction.valid_mask.count_ones()),
            "L2 evicted valid sectors"
        );
        if eviction.pending_mask != 0 {
            add!(l2_evicted_pending_lines, 1, "L2 evicted pending lines");
            add!(
                l2_evicted_pending_sectors,
                u64::from(eviction.pending_mask.count_ones()),
                "L2 evicted pending sectors"
            );
        }
        if eviction.dirty_mask != 0 {
            add!(l2_dirty_eviction_lines, 1, "L2 dirty eviction lines");
            add!(
                l2_dirty_evicted_sectors,
                u64::from(eviction.dirty_mask.count_ones()),
                "L2 dirty evicted sectors"
            );
        }
        Ok(self)
    }

    fn with_terminal_l2_dirty_residency(
        mut self,
        residency: L2DirtyResidency,
    ) -> Result<Self, SystemError> {
        let lines = u64::try_from(residency.lines).map_err(|_| SystemError::CounterOverflow {
            counter: "terminal L2 dirty resident lines",
        })?;
        let sectors =
            u64::try_from(residency.sectors).map_err(|_| SystemError::CounterOverflow {
                counter: "terminal L2 dirty resident sectors",
            })?;
        self.maximum_terminal_l2_dirty_lines = self.maximum_terminal_l2_dirty_lines.max(lines);
        self.maximum_terminal_l2_dirty_sectors =
            self.maximum_terminal_l2_dirty_sectors.max(sectors);
        Ok(self)
    }

    /// Build a coherent synthetic observer from explicit origin lanes.
    pub fn from_origins(
        arch: ArchId,
        global: DramOriginStats,
        local: DramOriginStats,
    ) -> Result<Self, SystemCounterError> {
        let read_sectors = global.read_sectors.checked_add(local.read_sectors).ok_or(
            SystemCounterError::DramOriginPartitionOverflow {
                counter: "read sectors",
                global: global.read_sectors,
                local: local.read_sectors,
            },
        )?;
        let bw_delayed_responses = global
            .bw_delayed_responses
            .checked_add(local.bw_delayed_responses)
            .ok_or(SystemCounterError::DramOriginPartitionOverflow {
                counter: "bandwidth-delayed responses",
                global: global.bw_delayed_responses,
                local: local.bw_delayed_responses,
            })?;
        Ok(Self {
            arch: Some(arch),
            read_sectors,
            bw_delayed_responses,
            l2_evicted_lines: 0,
            l2_evicted_valid_sectors: 0,
            l2_evicted_pending_lines: 0,
            l2_evicted_pending_sectors: 0,
            l2_dirty_eviction_lines: 0,
            l2_dirty_evicted_sectors: 0,
            maximum_terminal_l2_dirty_lines: 0,
            maximum_terminal_l2_dirty_sectors: 0,
            by_origin: [global, local],
        })
    }

    /// Explicitly attribute an already-built unified synthetic observer to one
    /// origin. Modeled requests always mutate an explicit lane directly.
    pub fn with_unified_origin(mut self, origin: MemoryOrigin) -> Self {
        let unified = DramOriginStats {
            read_sectors: self.read_sectors,
            bw_delayed_responses: self.bw_delayed_responses,
        };
        self.by_origin = [DramOriginStats::default(); 2];
        self.by_origin[origin.index()] = unified;
        self
    }

    pub fn validate_origin_partition(&self) -> Result<(), SystemCounterError> {
        let global = self.origin(MemoryOrigin::Global);
        let local = self.origin(MemoryOrigin::Local);
        let read_sectors = global.read_sectors.checked_add(local.read_sectors).ok_or(
            SystemCounterError::DramOriginPartitionOverflow {
                counter: "read sectors",
                global: global.read_sectors,
                local: local.read_sectors,
            },
        )?;
        if self.read_sectors != read_sectors {
            return Err(SystemCounterError::DramOriginPartitionMismatch {
                counter: "read sectors",
                unified: self.read_sectors,
                global: global.read_sectors,
                local: local.read_sectors,
            });
        }
        let bw_delayed_responses = global
            .bw_delayed_responses
            .checked_add(local.bw_delayed_responses)
            .ok_or(SystemCounterError::DramOriginPartitionOverflow {
                counter: "bandwidth-delayed responses",
                global: global.bw_delayed_responses,
                local: local.bw_delayed_responses,
            })?;
        if self.bw_delayed_responses != bw_delayed_responses {
            return Err(SystemCounterError::DramOriginPartitionMismatch {
                counter: "bandwidth-delayed responses",
                unified: self.bw_delayed_responses,
                global: global.bw_delayed_responses,
                local: local.bw_delayed_responses,
            });
        }
        Ok(())
    }

    pub fn add_checked(&mut self, other: &DramStats) -> Result<(), SystemCounterError> {
        let expected = self
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "DRAM stats",
            })?;
        let got = other
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "DRAM stats",
            })?;
        if expected != got {
            return Err(sim_arch::ArchitectureProvenanceError::Mismatch {
                component: "DRAM stats",
                expected,
                got,
            }
            .into());
        }
        let read_sectors = self.read_sectors.checked_add(other.read_sectors).ok_or(
            SystemCounterError::Overflow {
                counter: "DRAM read sectors",
            },
        )?;
        let bw_delayed_responses = self
            .bw_delayed_responses
            .checked_add(other.bw_delayed_responses)
            .ok_or(SystemCounterError::Overflow {
                counter: "DRAM bandwidth-delayed responses",
            })?;
        macro_rules! add_topology {
            ($field:ident, $name:literal) => {
                self.$field
                    .checked_add(other.$field)
                    .ok_or(SystemCounterError::Overflow { counter: $name })?
            };
        }
        let l2_evicted_lines = add_topology!(l2_evicted_lines, "L2 evicted lines");
        let l2_evicted_valid_sectors =
            add_topology!(l2_evicted_valid_sectors, "L2 evicted valid sectors");
        let l2_evicted_pending_lines =
            add_topology!(l2_evicted_pending_lines, "L2 evicted pending lines");
        let l2_evicted_pending_sectors =
            add_topology!(l2_evicted_pending_sectors, "L2 evicted pending sectors");
        let l2_dirty_eviction_lines =
            add_topology!(l2_dirty_eviction_lines, "L2 dirty eviction lines");
        let l2_dirty_evicted_sectors =
            add_topology!(l2_dirty_evicted_sectors, "L2 dirty evicted sectors");
        let mut by_origin = self.by_origin;
        for origin in MemoryOrigin::ALL {
            let origin_ix = origin.index();
            by_origin[origin_ix].read_sectors = self.by_origin[origin_ix]
                .read_sectors
                .checked_add(other.by_origin[origin_ix].read_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM origin read sectors",
                })?;
            by_origin[origin_ix].bw_delayed_responses = self.by_origin[origin_ix]
                .bw_delayed_responses
                .checked_add(other.by_origin[origin_ix].bw_delayed_responses)
                .ok_or(SystemCounterError::Overflow {
                    counter: "DRAM origin bandwidth-delayed responses",
                })?;
        }
        let next = Self {
            arch: self.arch,
            read_sectors,
            bw_delayed_responses,
            l2_evicted_lines,
            l2_evicted_valid_sectors,
            l2_evicted_pending_lines,
            l2_evicted_pending_sectors,
            l2_dirty_eviction_lines,
            l2_dirty_evicted_sectors,
            maximum_terminal_l2_dirty_lines: self
                .maximum_terminal_l2_dirty_lines
                .max(other.maximum_terminal_l2_dirty_lines),
            maximum_terminal_l2_dirty_sectors: self
                .maximum_terminal_l2_dirty_sectors
                .max(other.maximum_terminal_l2_dirty_sectors),
            by_origin,
        };
        self.validate_origin_partition()?;
        other.validate_origin_partition()?;
        next.validate_origin_partition()?;
        *self = next;
        Ok(())
    }
}

/// Internal result of one kernel launch through the shared-L2 system. Public
/// consumers receive only the configuration-bound replay result.
#[derive(Debug, Clone)]
pub(crate) struct SystemResult {
    /// Deployment architecture validated across schedule and all hierarchy configs.
    pub(crate) arch: ArchId,
    /// Per traced SM id.
    pub(crate) per_sm: BTreeMap<i32, SmSimResult>,
    /// Shared-L2 counters for the whole launch.
    pub(crate) l2: L2Counters,
    /// Per-source L2 sector/hit split (driver-side emission observer).
    pub(crate) l2_by_source: L2SourceSplit,
    /// DRAM read-traffic observer (server engagement diagnostics).
    pub(crate) dram: DramStats,
    /// Union of per-SM unmodeled write-class capability tails.
    pub(crate) unmodeled_tails: TailFlags,
    /// Raw timestamp of the last successfully processed modeled event.
    /// `None` means the queue was empty; `Some(0)` is a real event at time zero.
    /// This is a diagnostic timestamp, not an elapsed-cycle prediction.
    pub(crate) terminal_event_time: Option<u64>,
    /// Last successfully processed modeled event that is not an opt-in
    /// MemoryWriteAck.  This preserves an SM-active terminal lower bound when
    /// a feedback lane drains a service acknowledgement after the final SM
    /// issue; the full event frontier above remains available separately.
    pub(crate) sm_active_terminal_event_time: Option<u64>,
}

/// Run every SM's blocks to completion over one shared L2. `l2` is
/// caller-constructed (and caller-drained afterwards) so the geometry
/// resolution stays at the composition root; it must be FRESH per kernel
/// (cold cache, NCU `--cache-control all` parity — same contract as the
/// per-SM L1s). The caller's model is unchanged on every typed error; a
/// successful run commits the fully drained staged L2 as its final operation.
/// `dram` selects the read-bandwidth model ([`DramBw`]):
/// `Unmodeled` is the latency-only ablation / equivalence-differential
/// mode; `Carded` rates are validated non-zero here (typed
/// [`SystemError::DramBandwidthInvalid`], never a clamp).
#[cfg(test)]
pub(crate) fn simulate_kernel_shared_with_residency(
    per_sm_blocks: &BTreeMap<i32, Vec<BlockReplay>>,
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    shared_memory_residency: SharedMemoryResidency,
    l1_config: L1Config,
    l2: &mut L2Model,
    dram: DramBw,
) -> Result<SystemResult, SystemError> {
    simulate_kernel_shared_with_residency_and_candidates(
        per_sm_blocks,
        schedule,
        params,
        shared_memory_residency,
        l1_config,
        l2,
        dram,
        &[],
        &[],
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_kernel_shared_with_residency_and_candidates(
    per_sm_blocks: &BTreeMap<i32, Vec<BlockReplay>>,
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    shared_memory_residency: SharedMemoryResidency,
    l1_config: L1Config,
    l2: &mut L2Model,
    dram: DramBw,
    memory_service_candidates: &[MemoryServiceCandidate],
    memory_admission_candidates: &[MemoryAdmissionCandidate],
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<SystemResult, SystemError> {
    let arch = schedule.arch;
    require_arch(arch, params.arch, "issue parameters")?;
    require_arch(arch, l1_config.arch, "L1 configuration")?;
    require_arch(arch, l2.config().arch, "L2 configuration")?;
    checked_l2_counters(arch, l2)?;
    require_arch(arch, dram.arch(), "DRAM configuration")?;
    require_l2_pending_read_semantics(arch)?;
    params.validate_l1_timing(l1_config)?;
    // Fail closed on an inverted ladder: every horizon/run-ahead bound
    // below assumes no response can price under the L2-hit arm.
    if params.global_flat_latency < params.l1_miss_latency {
        return Err(SystemError::LatencyLadderInverted {
            l2_hit: params.l1_miss_latency,
            dram: params.global_flat_latency,
        });
    }
    // Fail closed on a zero bandwidth rate: a clamped server would be
    // absurd (1 B/s) or silently inert (0 Hz), and a 1 B/s server arms
    // the fixed-point u64 ceiling on large traces.
    if let DramBw::Carded {
        bytes_per_sec,
        sm_clock_hz,
        ..
    } = dram
        && (bytes_per_sec == 0 || sm_clock_hz == 0)
    {
        return Err(SystemError::DramBandwidthInvalid {
            bytes_per_sec,
            sm_clock_hz,
        });
    }
    if u32::try_from(per_sm_blocks.len()).is_err() {
        return Err(SystemError::ActorCountOutOfRange {
            count: per_sm_blocks.len(),
        });
    }
    validate_memory_service_candidates(memory_service_candidates).map_err(IssueError::from)?;
    validate_memory_admission_candidates(memory_admission_candidates).map_err(IssueError::from)?;
    if let Some(candidate) = memory_admission_feedback {
        candidate.validate_feedback(params.read_barrier_latency)?;
    }
    let shared_write_lookup_observer = stateful_write_lookup_observer_enabled(schedule)
        .then(|| L1WriteLookupObserver::for_arch(arch))
        .transpose()
        .map_err(IssueError::from)?
        .map(|observer| Rc::new(RefCell::new(observer)));
    let sm_ids: Vec<i32> = per_sm_blocks.keys().copied().collect();
    let mut cores: Vec<SmCore<'_>> = per_sm_blocks
        .values()
        .map(|blocks| {
            let mut core = SmCore::new_with_residency_and_observer(
                blocks,
                schedule,
                params,
                shared_memory_residency,
                SmMemMode::L1Shared(l1_config),
                shared_write_lookup_observer.clone(),
            )?;
            core.install_memory_service_candidates(memory_service_candidates)?;
            if !memory_admission_candidates.is_empty() {
                core.enable_memory_admission_observations();
            }
            if let Some(candidate) = memory_admission_feedback {
                core.install_memory_admission_feedback(candidate)?;
            }
            Ok::<_, IssueError>(core)
        })
        .collect::<Result<_, _>>()?;
    let observe_admission = !memory_admission_candidates.is_empty();
    let feedback_admission = memory_admission_feedback.is_some();
    let observe_admission_groups = observe_admission || feedback_admission;
    let mut admission_states = observe_admission
        .then(|| {
            (0..cores.len())
                .map(|_| MemoryAdmissionCandidateState::checked(memory_admission_candidates))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(IssueError::from)?;
    let mut admission_observers = observe_admission_groups.then(|| {
        (0..cores.len())
            .map(|_| AdmissionInstructionObserver::default())
            .collect::<Vec<_>>()
    });

    let mut queue: EventQueue<Ev> = EventQueue::new();
    #[cfg(feature = "performance-counters")]
    let batch_replyless_writes = memory_admission_feedback.is_none()
        && std::env::var_os("SIM_RUN_DISABLE_L2_ACCESS_BATCHING").is_none();
    #[cfg(not(feature = "performance-counters"))]
    let batch_replyless_writes = memory_admission_feedback.is_none();
    let mut scratch: Vec<MemRequest> = Vec::new();
    // Scheduled-but-unapplied response times PER SM: the only external
    // events that can affect an SM's own state, hence its advance
    // horizon. Other SMs' events are deliberately NOT barriers — cross-SM
    // effects flow only through the L2 access stream, which the queue
    // already orders by (time, seq). (Treating every queued event as a
    // barrier would also livelock: two SMs parked at the same timestamp
    // would bounce off each other's wake events forever.)
    let mut response_due = DueTimes::new(cores.len());
    // Opt-in STORE/RED completions are self-affecting external events: a
    // matching MEMBAR remains unpriced until the event is processed, and the
    // actor must not run past the earliest pending acknowledgement.
    let mut write_ack_due = DueTimes::new(cores.len());
    // Pending wake times PER SM, deduplicating at the source: a wake is
    // scheduled only when no pending wake at an earlier-or-equal time
    // covers it (an earlier wake re-parks and reschedules if still
    // needed). Without this, every response that wakes an SM which then
    // re-parks at the same future horizon piles up another stale wake —
    // hundreds popping back-to-back at one timestamp on real cells —
    // wasted heap traffic AND an unbounded idle-wake streak the livelock
    // guard could not be told apart from a real spin.
    let mut wake_due = DueTimes::new(cores.len());
    let mut unpriced = UnpricedBounds((0..cores.len()).map(|_| BTreeMap::new()).collect());
    let mut pending_l2_ready = PendingL2Ready::default();
    let mut l2_by_source = L2SourceSplit::for_arch(arch);
    let mut dram_server = DramServer::new(dram);
    let mut dram_stats = DramStats::for_arch(arch);
    for ix in 0..cores.len() {
        let target = u32::try_from(ix)
            .map_err(|_| SystemError::ActorCountOutOfRange { count: cores.len() })?;
        queue.schedule_at(0, TargetId(target), Ev::SmWake)?;
        wake_due.push(ix, 0);
    }

    let mut spin = SpinGuard::new(cores.len())?;
    let mut terminal_event_time = None;
    let mut sm_active_terminal_event_time = None;
    #[cfg(feature = "performance-counters")]
    let mut performance = std::env::var_os("SIM_RUN_PERFORMANCE_COUNTERS")
        .is_some()
        .then(EventPerformanceCounters::default);
    // The caller owns `l2` across this boundary. Complete all replay admission
    // and event-system setup first so their established typed errors keep
    // precedence over snapshot allocation failure. Then replay the entire
    // kernel on a private snapshot and publish it only after event draining,
    // deadlock checks, and result construction all succeed.
    let initial_lookup_observer = *checked_l2_counters(arch, l2)?;
    let mut lookup_observer_hits = 0_u64;
    let mut lookup_observer_requests = 0_u64;
    let mut staged_l2 = l2.try_snapshot()?;
    while let Some(ev) = queue.pop()? {
        let is_write_ack = matches!(&ev.kind, Ev::MemoryWriteAck(_));
        #[cfg(feature = "performance-counters")]
        if let Some(performance) = performance.as_mut() {
            let queue_depth_before_pop =
                queue
                    .len()
                    .checked_add(1)
                    .ok_or(SystemError::CounterOverflow {
                        counter: "performance event-queue depth",
                    })?;
            let logical_events = match &ev.kind {
                Ev::L2AccessBatch(requests) => requests.len(),
                _ => 1,
            };
            performance.observe(&ev.kind, ev.time, queue_depth_before_pop, logical_events)?;
        }
        let sm_ix =
            usize::try_from(ev.target.0).map_err(|_| SystemError::EventTargetOutOfRange {
                target: ev.target.0,
                actor_count: cores.len(),
            })?;
        if sm_ix >= cores.len() {
            return Err(SystemError::EventTargetOutOfRange {
                target: ev.target.0,
                actor_count: cores.len(),
            });
        }
        match ev.kind {
            Ev::SmWake => {
                wake_due.pop_expect("SM wake", sm_ix, ev.time)?;
                let progress = run_sm(
                    sm_ix,
                    &mut cores,
                    &mut queue,
                    &mut response_due,
                    &mut write_ack_due,
                    &mut wake_due,
                    &mut unpriced,
                    batch_replyless_writes,
                    &mut scratch,
                    params.l1_miss_latency,
                    memory_admission_feedback,
                )?;
                spin.on_event(ev.time, !progress)?;
            }
            kind @ (Ev::L2Access(_) | Ev::L2AccessBatch(_)) => {
                let requests = match kind {
                    Ev::L2Access(request) => L2AccessEventRequests::Single([request]),
                    Ev::L2AccessBatch(requests) => L2AccessEventRequests::Batch(requests),
                    _ => unreachable!(),
                };
                let mut process_request = |req: MemRequest| -> Result<(), SystemError> {
                    if req.issue_time != ev.time {
                        return Err(SystemError::RequestTimeMismatch {
                            sm: sm_ix,
                            issue_time: req.issue_time,
                            event_time: ev.time,
                        });
                    }
                    #[cfg(test)]
                    let req = {
                        let mut req = req;
                        maybe_override_request_masks(&mut req);
                        req
                    };
                    let own_sector_bit = 1u8 << (req.sector_id & 3) as u8;
                    if req.sector_mask == 0
                        || req.sector_mask & !0b1111 != 0
                        || req.sector_mask & own_sector_bit == 0
                    {
                        return Err(SystemError::SectorMaskInvalid {
                            sector_id: req.sector_id,
                            sector_mask: req.sector_mask,
                        });
                    }
                    let anchor_is_local = req.local_sector_mask & own_sector_bit != 0;
                    if req.local_sector_mask & !req.sector_mask != 0
                        || anchor_is_local != (req.origin == MemoryOrigin::Local)
                    {
                        return Err(SystemError::OriginMaskInvalid {
                            sector_id: req.sector_id,
                            origin: req.origin,
                            sector_mask: req.sector_mask,
                            local_sector_mask: req.local_sector_mask,
                        });
                    }
                    if !req.has_valid_contract() {
                        return Err(SystemError::MemoryRequestContractInvalid {
                            sector_id: req.sector_id,
                            request_source: req.source,
                            operation: req.operation,
                            reply: req.reply,
                        });
                    }
                    let feedback_unpriced_removal =
                        feedback_prices_request(memory_admission_feedback, req)
                            .then(|| unpriced.prepare_remove(sm_ix, req.issue_time))
                            .transpose()?;
                    #[cfg(test)]
                    maybe_inject_l2_access_error()?;

                    // Prepare the exact ascending mask walk against a private copy
                    // of its set. The callback completes every fallible timing and
                    // counter calculation before the L2 plan commits; only then can
                    // the API issue a fill token. Event insertion follows
                    // immediately. If insertion fails, the kernel-private L2
                    // snapshot is discarded and the caller-owned model is unchanged.
                    let line = req.sector_id >> 2;
                    let (l2_access, commit) = staged_l2.access_mask_event_coalesced_transactional(
                        line,
                        req.sector_mask,
                        req.operation.l2_class(),
                        |preview| -> Result<L2SystemCommit, SystemError> {
                            let lookup = preview.lookup;
                            let merged_ready = pending_l2_ready.merged_ready(preview)?;
                            pending_l2_ready.preflight_register(preview)?;
                            let requested_sectors = u64::from(lookup.sectors);
                            let counted_hits = u64::from(lookup.counted_hits);
                            let (lookup_observer_requests, lookup_observer_hits) = match req
                                .operation
                            {
                                MemoryOperation::Read => (
                                    requested_sectors,
                                    counted_hits
                                        .checked_add(u64::from(
                                            preview.pending_merge_mask().count_ones(),
                                        ))
                                        .ok_or(SystemError::CounterOverflow {
                                            counter: "L2 read lookup observer hits",
                                        })?,
                                ),
                                // ATOM is a returning write-class request.
                                // The L2 tag-stage observer counts every
                                // atomic sector as a lookup hit, while
                                // `counted_hits` continues to describe
                                // physical data presence for response
                                // pricing.
                                MemoryOperation::Atomic => (requested_sectors, requested_sectors),
                                MemoryOperation::Store | MemoryOperation::Reduction => (0, 0),
                            };
                            let source_ix = req.source as usize;
                            let origin_masks = [
                                req.sector_mask & !req.local_sector_mask,
                                req.local_sector_mask,
                            ];
                            let origin_requested =
                                origin_masks.map(|mask| u64::from(mask.count_ones()));
                            let origin_hits = origin_masks.map(|mask| {
                                u64::from((mask & lookup.counted_hit_mask).count_ones())
                            });
                            let next_source_sectors = l2_by_source.sectors[source_ix]
                                .checked_add(requested_sectors)
                                .ok_or(SystemCounterError::Overflow {
                                    counter: "L2 source sectors",
                                })?;
                            let next_source_hits = l2_by_source.hits[source_ix]
                                .checked_add(counted_hits)
                                .ok_or(SystemCounterError::Overflow {
                                    counter: "L2 source hits",
                                })?;
                            let mut next_origin_source_sectors = [0u64; 2];
                            let mut next_origin_source_hits = [0u64; 2];
                            for origin in MemoryOrigin::ALL {
                                let origin_ix = origin.index();
                                next_origin_source_sectors[origin_ix] = l2_by_source.origin_sectors
                                    [origin_ix][source_ix]
                                    .checked_add(origin_requested[origin_ix])
                                    .ok_or(SystemCounterError::Overflow {
                                        counter: "L2 origin/source sectors",
                                    })?;
                                next_origin_source_hits[origin_ix] = l2_by_source.origin_hits
                                    [origin_ix][source_ix]
                                    .checked_add(origin_hits[origin_ix])
                                    .ok_or(SystemCounterError::Overflow {
                                        counter: "L2 origin/source hits",
                                    })?;
                            }

                            let fetch_mask = match req.operation {
                                MemoryOperation::Read => preview.fill_mask(),
                                operation if operation.fetches_missing_data() => {
                                    req.sector_mask & !lookup.present_mask
                                }
                                _ => 0,
                            };
                            let fetch_sectors = u64::from(fetch_mask.count_ones());
                            let origin_fetches = origin_masks
                                .map(|mask| u64::from((mask & fetch_mask).count_ones()));
                            let next_read_sectors = dram_stats
                                .read_sectors
                                .checked_add(fetch_sectors)
                                .ok_or(SystemError::CounterOverflow {
                                    counter: "DRAM read sectors",
                                })?;
                            let mut next_origin_read_sectors = [0u64; 2];
                            for origin in MemoryOrigin::ALL {
                                let origin_ix = origin.index();
                                next_origin_read_sectors[origin_ix] = dram_stats.by_origin
                                    [origin_ix]
                                    .read_sectors
                                    .checked_add(origin_fetches[origin_ix])
                                    .ok_or(SystemError::CounterOverflow {
                                        counter: "DRAM origin read sectors",
                                    })?;
                            }
                            let topology_stats = match preview.eviction() {
                                Some(eviction) => dram_stats.with_l2_eviction(eviction)?,
                                None => dram_stats,
                            };
                            let service_preview = if fetch_sectors > 0 {
                                let bytes = fetch_sectors.checked_mul(32).ok_or(
                                    SystemError::CounterOverflow {
                                        counter: "DRAM request bytes",
                                    },
                                )?;
                                match &dram_server {
                                    Some(server) => Some(server.preview(req.issue_time, bytes)?),
                                    None => None,
                                }
                            } else {
                                None
                            };
                            let bw_ready = service_preview.map_or(0, |(_, ready)| ready);

                            let fetch_latency_ready = if fetch_sectors > 0 {
                                let operation = if req.reply != ReplyTo::None {
                                    "DRAM response visibility"
                                } else if req.operation == MemoryOperation::Atomic {
                                    "unobserved ATOM DRAM completion"
                                } else {
                                    "DRAM fill visibility"
                                };
                                Some(checked_cycle_add(
                                    operation,
                                    req.issue_time,
                                    params.global_flat_latency,
                                )?)
                            } else {
                                None
                            };
                            let fill_ready = if preview.fill_mask() != 0 {
                                Some(
                                    fetch_latency_ready
                                        .ok_or(SystemError::FillScheduleMismatch {
                                            sector_id: req.sector_id,
                                            has_token: true,
                                            has_ready: false,
                                        })?
                                        .max(bw_ready),
                                )
                            } else {
                                None
                            };

                            let response = if req.reply != ReplyTo::None {
                                let response_origin_ix = req.origin.index();
                                let bound = checked_cycle_add(
                                    "unpriced L2-hit response bound",
                                    req.issue_time,
                                    params.l1_miss_latency,
                                )?;
                                let unpriced_removal = unpriced.prepare_remove(sm_ix, bound)?;
                                let latency_ready = bound
                                    .max(fetch_latency_ready.unwrap_or(0))
                                    .max(merged_ready.unwrap_or(0));
                                let service_ready = latency_ready.max(bw_ready);
                                // A folded physical fill can serve ordinary
                                // and deep constant waiters together. Their
                                // return-service delta is therefore stamped
                                // per fill waiter by the SM response path;
                                // this event stays at the shared hierarchy
                                // ready time and does not over-delay ordinary
                                // consumers or alter DRAM pricing.
                                if service_ready < bound {
                                    return Err(SystemError::ResponseBeforeBound {
                                        sm: sm_ix,
                                        ready: service_ready,
                                        bound,
                                    });
                                }
                                let delayed = u64::from(service_ready > latency_ready);
                                let next_delayed = dram_stats
                                    .bw_delayed_responses
                                    .checked_add(delayed)
                                    .ok_or(SystemError::CounterOverflow {
                                        counter: "DRAM bandwidth-delayed responses",
                                    })?;
                                let next_origin_delayed = dram_stats.by_origin[response_origin_ix]
                                    .bw_delayed_responses
                                    .checked_add(delayed)
                                    .ok_or(SystemError::CounterOverflow {
                                        counter: "DRAM origin bandwidth-delayed responses",
                                    })?;
                                Some(L2ResponseCommit {
                                    ready: service_ready,
                                    origin_ix: response_origin_ix,
                                    next_delayed,
                                    next_origin_delayed,
                                    unpriced_removal,
                                })
                            } else {
                                None
                            };

                            let atomic_completion_ready =
                                if req.operation == MemoryOperation::Atomic && response.is_none() {
                                    let latency_ready = match fetch_latency_ready {
                                        Some(ready) => ready,
                                        None => checked_cycle_add(
                                            "unobserved ATOM L2-hit completion",
                                            req.issue_time,
                                            params.l1_miss_latency,
                                        )?,
                                    };
                                    Some(latency_ready.max(bw_ready))
                                } else {
                                    None
                                };

                            Ok(L2SystemCommit {
                                source_ix,
                                source_sectors: next_source_sectors,
                                source_hits: next_source_hits,
                                origin_source_sectors: next_origin_source_sectors,
                                origin_source_hits: next_origin_source_hits,
                                read_sectors: next_read_sectors,
                                origin_read_sectors: next_origin_read_sectors,
                                topology_stats,
                                dram_free_fp: service_preview.map(|(done, _)| done),
                                fill_ready,
                                atomic_completion_ready,
                                response,
                                observed_sectors: requested_sectors,
                                observed_hits: counted_hits,
                                lookup_observer_requests,
                                lookup_observer_hits,
                            })
                        },
                    )?;

                    lookup_observer_requests = lookup_observer_requests
                        .checked_add(commit.lookup_observer_requests)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 lookup observer requests",
                        })?;
                    lookup_observer_hits = lookup_observer_hits
                        .checked_add(commit.lookup_observer_hits)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 lookup observer hits",
                        })?;

                    if observe_admission_groups && let Some(witness) = req.admission_witness {
                        if let Some(completed) = admission_observers
                            .as_mut()
                            .expect("enabled admission observer storage")[sm_ix]
                            .observe(
                                sm_ids[sm_ix],
                                witness,
                                req.operation,
                                req.issue_time,
                                commit.observed_sectors,
                                commit.observed_hits,
                                req.sector_id,
                            )?
                        {
                            if observe_admission {
                                admission_states
                                    .as_mut()
                                    .expect("enabled admission candidate storage")[sm_ix]
                                    .observe(
                                        usize::from(completed.witness.subcore()),
                                        usize::try_from(completed.witness.warp()).map_err(
                                            |_| SystemError::CounterOverflow {
                                                counter: "memory-admission witness warp",
                                            },
                                        )?,
                                        completed.operation,
                                        completed.hit_sectors,
                                        completed.miss_sectors,
                                        completed.issue_time,
                                    )?;
                            }
                            if feedback_admission {
                                if let Some(ack) = cores[sm_ix].complete_memory_admission_feedback(
                                    completed.witness,
                                    completed.operation,
                                    completed.hit_sectors,
                                    completed.miss_sectors,
                                    completed.l2_requests,
                                    completed.unique_regions,
                                    completed.issue_time,
                                )? {
                                    queue.schedule_at(
                                        ack.fence_visible_at,
                                        ev.target,
                                        Ev::MemoryWriteAck(ack),
                                    )?;
                                    write_ack_due.push(sm_ix, ack.fence_visible_at);
                                }
                            }
                        }
                    }

                    let has_token = l2_access.fill.is_some();
                    let has_ready = commit.fill_ready.is_some();
                    if has_token != has_ready {
                        return Err(SystemError::FillScheduleMismatch {
                            sector_id: req.sector_id,
                            has_token,
                            has_ready,
                        });
                    }
                    if let (Some(fill), Some(ready)) = (l2_access.fill, commit.fill_ready) {
                        pending_l2_ready.register(fill, ready)?;
                    }
                    // A response that becomes ready with its own new fill carries the
                    // token so installation precedes delivery in one event. A mixed
                    // new/pending read can instead finish its new fetch first and wait
                    // longer for an older pending sector; expose that fill at its true
                    // ready time, then deliver a fill-less response.
                    if let Some(response) = commit.response.as_ref() {
                        let response_fill = match (l2_access.fill, commit.fill_ready) {
                            (Some(fill), Some(fill_ready)) if fill_ready == response.ready => {
                                Some(fill)
                            }
                            (Some(fill), Some(fill_ready)) if fill_ready < response.ready => {
                                queue.schedule_at(fill_ready, ev.target, Ev::L2Fill(fill))?;
                                None
                            }
                            (Some(_), Some(fill_ready)) => {
                                return Err(SystemError::ResponseBeforeFill {
                                    sector_id: req.sector_id,
                                    response_ready: response.ready,
                                    fill_ready,
                                });
                            }
                            (None, None) => None,
                            _ => {
                                return Err(SystemError::FillScheduleMismatch {
                                    sector_id: req.sector_id,
                                    has_token,
                                    has_ready,
                                });
                            }
                        };
                        queue.schedule_at(
                            response.ready,
                            ev.target,
                            Ev::MemResponse {
                                reply: req.reply,
                                fill: response_fill,
                            },
                        )?;
                    } else if let (Some(fill), Some(ready)) = (l2_access.fill, commit.fill_ready) {
                        queue.schedule_at(ready, ev.target, Ev::L2Fill(fill))?;
                    } else if let Some(ready) = commit.atomic_completion_ready {
                        queue.schedule_at(ready, ev.target, Ev::AtomicCompletion)?;
                    }

                    l2_by_source.sectors[commit.source_ix] = commit.source_sectors;
                    l2_by_source.hits[commit.source_ix] = commit.source_hits;
                    for origin in MemoryOrigin::ALL {
                        let origin_ix = origin.index();
                        l2_by_source.origin_sectors[origin_ix][commit.source_ix] =
                            commit.origin_source_sectors[origin_ix];
                        l2_by_source.origin_hits[origin_ix][commit.source_ix] =
                            commit.origin_source_hits[origin_ix];
                    }
                    dram_stats.read_sectors = commit.read_sectors;
                    for origin in MemoryOrigin::ALL {
                        let origin_ix = origin.index();
                        dram_stats.by_origin[origin_ix].read_sectors =
                            commit.origin_read_sectors[origin_ix];
                    }
                    dram_stats.l2_evicted_lines = commit.topology_stats.l2_evicted_lines;
                    dram_stats.l2_evicted_valid_sectors =
                        commit.topology_stats.l2_evicted_valid_sectors;
                    dram_stats.l2_evicted_pending_lines =
                        commit.topology_stats.l2_evicted_pending_lines;
                    dram_stats.l2_evicted_pending_sectors =
                        commit.topology_stats.l2_evicted_pending_sectors;
                    dram_stats.l2_dirty_eviction_lines =
                        commit.topology_stats.l2_dirty_eviction_lines;
                    dram_stats.l2_dirty_evicted_sectors =
                        commit.topology_stats.l2_dirty_evicted_sectors;
                    if let Some(done) = commit.dram_free_fp
                        && let Some(server) = &mut dram_server
                    {
                        server.free_fp = done;
                    }
                    if let Some(response) = commit.response {
                        dram_stats.bw_delayed_responses = response.next_delayed;
                        dram_stats.by_origin[response.origin_ix].bw_delayed_responses =
                            response.next_origin_delayed;
                        response_due.push(sm_ix, response.ready);
                        response.unpriced_removal.commit(&mut unpriced);
                    }
                    if let Some(removal) = feedback_unpriced_removal {
                        removal.commit(&mut unpriced);
                    }
                    spin.on_progress_event(ev.time);
                    Ok(())
                };
                for request in requests.as_ref().iter().copied() {
                    process_request(request)?;
                }
            }
            Ev::L2Fill(fill) => {
                pending_l2_ready.preflight_remove(fill, ev.time)?;
                let _completion = staged_l2.complete_fill(fill)?;
                pending_l2_ready.commit_remove(fill);
                spin.on_progress_event(ev.time);
            }
            Ev::AtomicCompletion => {
                spin.on_progress_event(ev.time);
            }
            Ev::MemoryWriteAck(ack) => {
                write_ack_due.pop_expect("modeled write acknowledgement", sm_ix, ev.time)?;
                cores[sm_ix].complete_memory_write_ack(ack, ev.time)?;
                run_sm(
                    sm_ix,
                    &mut cores,
                    &mut queue,
                    &mut response_due,
                    &mut write_ack_due,
                    &mut wake_due,
                    &mut unpriced,
                    batch_replyless_writes,
                    &mut scratch,
                    params.l1_miss_latency,
                    memory_admission_feedback,
                )?;
                spin.on_event(ev.time, false)?;
            }
            Ev::MemResponse { reply, fill } => {
                if let Some(fill) = fill {
                    pending_l2_ready.preflight_remove(fill, ev.time)?;
                }
                cores[sm_ix].preflight_response(reply, ev.time)?;
                if let Some(fill) = fill {
                    // `Stale` is an expected no-op after eviction; delivery of
                    // data to the requester remains valid and must continue.
                    let _completion = staged_l2.complete_fill(fill)?;
                    pending_l2_ready.commit_remove(fill);
                }
                response_due.pop_expect("memory response", sm_ix, ev.time)?;
                scratch.clear();
                cores[sm_ix].on_response(reply, ev.time, &mut scratch)?;
                // Retries the completion unblocked become L2 accesses at
                // the response time.
                schedule_emitted_requests(
                    sm_ix,
                    ev.target,
                    &mut queue,
                    &mut unpriced,
                    &mut scratch,
                    params.l1_miss_latency,
                    batch_replyless_writes,
                    memory_admission_feedback,
                )?;
                run_sm(
                    sm_ix,
                    &mut cores,
                    &mut queue,
                    &mut response_due,
                    &mut write_ack_due,
                    &mut wake_due,
                    &mut unpriced,
                    batch_replyless_writes,
                    &mut scratch,
                    params.l1_miss_latency,
                    memory_admission_feedback,
                )?;
                spin.on_event(ev.time, false)?;
            }
        }
        terminal_event_time = Some(ev.time);
        if !is_write_ack {
            sm_active_terminal_event_time = Some(ev.time);
        }
    }

    if pending_l2_ready.len() != 0 {
        return Err(SystemError::PendingL2ReadyNotDrained {
            entries: pending_l2_ready.len(),
        });
    }
    if !write_ack_due.is_empty() {
        return Err(SystemError::MemoryWriteAckDueNotDrained);
    }
    if observe_admission_groups {
        for (ix, observer) in admission_observers
            .as_ref()
            .expect("enabled admission observer storage")
            .iter()
            .enumerate()
        {
            if observer.pending.is_some() {
                return Err(SystemError::AdmissionObservationIncomplete { sm_id: sm_ids[ix] });
            }
        }
    }

    // The L2 state machine above deliberately retains every physical sector
    // mutation.  NCU's L1/TEX-origin write observer is a separate packetized
    // stream: a regular store wavefront contributes its sectors-per-wavefront
    // work package, while an explicit `.128` store preserves the full sector
    // span.  Reconstruct this from the admitted trace/schedule only after the
    // event stream drains; no target counter or timing value enters the rule.
    let write_lookup_sectors = l2_write_observer_sectors(per_sm_blocks, schedule)?;
    let lookup_hits = initial_lookup_observer
        .lookup_hits
        .checked_add(lookup_observer_hits)
        .and_then(|value| value.checked_add(write_lookup_sectors))
        .ok_or(SystemError::CounterOverflow {
            counter: "L2 lookup observer hits",
        })?;
    let lookup_requests = initial_lookup_observer
        .lookup_requests
        .checked_add(lookup_observer_requests)
        .and_then(|value| value.checked_add(write_lookup_sectors))
        .ok_or(SystemError::CounterOverflow {
            counter: "L2 lookup observer requests",
        })?;
    staged_l2.replace_lookup_observer(lookup_hits, lookup_requests)?;

    let mut per_sm = BTreeMap::new();
    let mut unmodeled_tails = TailFlags::default();
    for (ix, core) in cores.iter().enumerate() {
        let pending_write_acks = core.pending_memory_write_acks();
        if pending_write_acks != 0 {
            return Err(SystemError::MemoryWriteAckPending {
                sm_id: sm_ids[ix],
                pending: pending_write_acks,
            });
        }
        if !core.is_done() {
            return Err(SystemError::Deadlock {
                sm_id: sm_ids[ix],
                outstanding: core.outstanding_responses(),
            });
        }
        let mut sm_result = core.result();
        if observe_admission {
            sm_result.readiness.memory_admission_candidates = admission_states
                .as_ref()
                .expect("enabled admission candidate storage")[ix]
                .diagnostics();
        }
        unmodeled_tails = unmodeled_tails.union(sm_result.unmodeled_tails);
        per_sm.insert(sm_ids[ix], sm_result);
    }
    let l2_counters = *checked_l2_counters(arch, &staged_l2)?;
    #[cfg(feature = "performance-counters")]
    if let Some(performance) = performance {
        eprintln!(
            "SIM_EVENT_PERF events_processed={} queue_events_processed={} sm_wake={} \
             l2_access={} l2_access_batches={} l2_access_batched_requests={} l2_fill={} \
             atomic_completion={} memory_write_ack={} mem_response={} same_cycle_after_first={} \
             queue_same_cycle_after_first={} maximum_queue_depth={}",
            performance.processed,
            performance.queue_events_processed,
            performance.sm_wake,
            performance.l2_access,
            performance.l2_access_batches,
            performance.l2_access_batched_requests,
            performance.l2_fill,
            performance.atomic_completion,
            performance.memory_write_ack,
            performance.mem_response,
            performance.same_cycle_after_first,
            performance.queue_same_cycle_after_first,
            performance.maximum_queue_depth,
        );
    }
    dram_stats = dram_stats.with_terminal_l2_dirty_residency(staged_l2.dirty_residency())?;
    let result = SystemResult {
        arch,
        per_sm,
        l2: l2_counters,
        l2_by_source,
        dram: dram_stats,
        unmodeled_tails,
        terminal_event_time,
        sm_active_terminal_event_time,
    };
    *l2 = staged_l2;
    Ok(result)
}

#[cfg(test)]
fn simulate_kernel_shared(
    per_sm_blocks: &BTreeMap<i32, Vec<BlockReplay>>,
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    l1_config: L1Config,
    l2: &mut L2Model,
    dram: DramBw,
) -> Result<SystemResult, SystemError> {
    simulate_kernel_shared_with_residency(
        per_sm_blocks,
        schedule,
        params,
        SharedMemoryResidency::unconstrained_fixture(),
        l1_config,
        l2,
        dram,
    )
}

fn replay_horizon(response: Option<u64>, unpriced: Option<u64>) -> ReplayHorizon {
    match (response, unpriced) {
        (Some(left), Some(right)) => ReplayHorizon::Bounded(left.min(right)),
        (Some(bound), None) | (None, Some(bound)) => ReplayHorizon::Bounded(bound),
        (None, None) => ReplayHorizon::Unbounded,
    }
}

fn batchable_replyless_write(request: MemRequest) -> bool {
    request.reply == ReplyTo::None
        && matches!(
            request.operation,
            MemoryOperation::Store | MemoryOperation::Reduction
        )
}

/// Reconstruct the SM86 L2 write-observer sector work from the admitted
/// instruction stream.  Nsight Compute defines an L2 request as carrying up
/// to four sectors; the simulator's physical event stream intentionally keeps
/// each sector mutation so cache state and response pricing remain exact.  For
/// the observer, ordinary store instructions are packetized into one
/// sectors-per-wavefront work package (`ceil(sectors / ceil(sectors / 4))`).
/// Explicit 128-bit stores retain their complete sector span because their
/// SASS encoding already expresses the full-width packet.  This is a
/// topology rule, not a workload-specific fit, and it is applied equally to
/// every admitted SM86 launch (including retained stream-L2 launches).
fn l2_write_observer_sectors(
    per_sm_blocks: &BTreeMap<i32, Vec<BlockReplay>>,
    schedule: &FunctionSchedule,
) -> Result<u64, SystemError> {
    let mut total = 0_u64;
    for blocks in per_sm_blocks.values() {
        for block in blocks {
            for warp in &block.warps {
                for inst in &warp.insts {
                    let (off, len) = match inst.kind {
                        AccessKind::GlobalStore { off, len }
                        | AccessKind::GlobalReduction { off, len, .. }
                        | AccessKind::LocalStore { off, len } => (off, len),
                        _ => continue,
                    };
                    if len == 0 {
                        continue;
                    }
                    let end =
                        off.checked_add(usize::from(len))
                            .ok_or(SystemError::CounterOverflow {
                                counter: "L2 write observer sector span",
                            })?;
                    if end > warp.sectors.len() {
                        return Err(SystemError::CounterOverflow {
                            counter: "L2 write observer sector span",
                        });
                    }
                    let sectors = u64::from(len);
                    let slot = schedule.by_pc.get(&u64::from(inst.pc)).ok_or(
                        SystemError::CounterOverflow {
                            counter: "L2 write observer schedule join",
                        },
                    )?;
                    let opcode = slot
                        .opcode_text()
                        .split_ascii_whitespace()
                        .next()
                        .unwrap_or("");
                    let observed = if opcode.split('.').any(|modifier| modifier == "128") {
                        sectors
                    } else {
                        let wavefronts = sectors.div_ceil(4);
                        sectors.div_ceil(wavefronts)
                    };
                    total = total
                        .checked_add(observed)
                        .ok_or(SystemError::CounterOverflow {
                            counter: "L2 write observer sectors",
                        })?;
                }
            }
        }
    }
    Ok(total)
}

fn feedback_prices_request(
    candidate: Option<MemoryAdmissionCandidate>,
    request: MemRequest,
) -> bool {
    request.admission_witness.is_some()
        && candidate.is_some_and(|candidate| candidate.matches_operation(request.operation))
}

#[allow(clippy::too_many_arguments)]
fn schedule_emitted_requests(
    sm: usize,
    target: TargetId,
    queue: &mut EventQueue<Ev>,
    unpriced: &mut UnpricedBounds,
    scratch: &mut Vec<MemRequest>,
    l2_hit_latency: u64,
    batch_replyless_writes: bool,
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<(), SystemError> {
    let mut index = 0;
    while index < scratch.len() {
        let request = scratch[index];
        if batch_replyless_writes && batchable_replyless_write(request) {
            let mut end = index + 1;
            while end < scratch.len()
                && batchable_replyless_write(scratch[end])
                && scratch[end].issue_time == request.issue_time
            {
                end += 1;
            }
            if end - index > 1 {
                let requests = end - index;
                let sequence_span = NonZeroU64::new(u64::try_from(requests).map_err(|_| {
                    SystemError::CounterOverflow {
                        counter: "L2 access batch sequence span",
                    }
                })?)
                .expect("a batched run contains at least two requests");
                queue.schedule_at_spanning(
                    request.issue_time,
                    target,
                    Ev::L2AccessBatch(scratch[index..end].to_vec().into_boxed_slice()),
                    sequence_span,
                )?;
                index = end;
                continue;
            }
        }
        if request.reply != ReplyTo::None {
            let bound = checked_cycle_add(
                "unpriced L2-hit response bound",
                request.issue_time,
                l2_hit_latency,
            )?;
            unpriced.insert(sm, bound)?;
        }
        // A timing-active write candidate cannot be priced until this future
        // L2 access is processed.  The memory issue pipeline may emit the
        // request several cycles ahead of the actor clock, so retain that
        // access timestamp as a self-affecting bound just like an unpriced
        // read response.  Otherwise an unrelated wake can let the SM run past
        // the event that releases its feedback queue or schedules its ack.
        if feedback_prices_request(memory_admission_feedback, request) {
            unpriced.insert(sm, request.issue_time)?;
        }
        queue.schedule_at(request.issue_time, target, Ev::L2Access(request))?;
        index += 1;
    }
    scratch.clear();
    Ok(())
}

/// Resume one SM actor: advance to its earliest self-affecting external
/// event — the earlier of its next scheduled response and the lower
/// bound of any request still awaiting L2 pricing — schedule its
/// emitted requests, and re-arm its wake timer if it has one. Returns
/// whether the actor made progress (issued or emitted anything) — the
/// livelock guard's signal.
#[allow(clippy::too_many_arguments)] // driver-internal plumbing, one call shape
fn run_sm(
    ix: usize,
    cores: &mut [SmCore<'_>],
    queue: &mut EventQueue<Ev>,
    response_due: &mut DueTimes,
    write_ack_due: &mut DueTimes,
    wake_due: &mut DueTimes,
    unpriced: &mut UnpricedBounds,
    batch_replyless_writes: bool,
    scratch: &mut Vec<MemRequest>,
    l2_hit_latency: u64,
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<bool, SystemError> {
    scratch.clear();
    let write_or_unpriced = match (write_ack_due.next(ix), unpriced.min(ix)) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(bound), None) | (None, Some(bound)) => Some(bound),
        (None, None) => None,
    };
    let horizon = replay_horizon(response_due.next(ix), write_or_unpriced);
    let issued_before = cores[ix].instructions_issued();
    let adv = cores[ix].advance(horizon, scratch)?;
    let progress = cores[ix].instructions_issued() > issued_before || !scratch.is_empty();
    let target =
        u32::try_from(ix).map_err(|_| SystemError::ActorCountOutOfRange { count: cores.len() })?;
    schedule_emitted_requests(
        ix,
        TargetId(target),
        queue,
        unpriced,
        scratch,
        l2_hit_latency,
        batch_replyless_writes,
        memory_admission_feedback,
    )?;
    if let Advance::Idle {
        wake_at: Some(wake),
    } = adv
    {
        let covered = wake_due.next(ix).is_some_and(|t| t <= wake);
        if !covered {
            queue.schedule_at(wake, TargetId(target), Ev::SmWake)?;
            wake_due.push(ix, wake);
        }
    }
    Ok(progress)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::BTreeMap;
    use std::num::NonZeroU32;

    use sim_arch::LdgstsCompletionThroughput;
    use sim_mem::{
        CacheOpToken, L2Config, L2CounterError, L2FillCompletion, L2OpClass, L2Replacement,
        LdgstsL1Policy, SliceHash,
    };

    use crate::issue::{MemoryServiceCandidateOperation, MemoryServiceCandidateScope};

    use super::*;

    #[test]
    fn admission_observer_reconstructs_one_instruction_from_ordered_l2_requests() {
        let candidate = MemoryAdmissionCandidate {
            candidate_id: 1,
            scope: MemoryServiceCandidateScope::Sm,
            operation: MemoryServiceCandidateOperation::Store,
            queue_capacity_instructions: 4,
            admission_latency_cycles: 8,
            l2_hit_sector_units: 1,
            l2_hit_sector_cycles: 1,
            l2_miss_sector_units: 1,
            l2_miss_sector_cycles: 10,
            l2_miss_service_width_sectors: 0,
            membar_waits_for_service_completion: false,
            write_completion_latency_cycles: 0,
            write_completion_request_cycles: 0,
            write_completion_region_cycles: 0,
            write_completion_parallel_lanes: 0,
        };
        let mut state = MemoryAdmissionCandidateState::checked(&[candidate]).unwrap();
        let mut observer = AdmissionInstructionObserver::default();
        let base = AdmissionRequestWitness::new(2, 7, 1, 0, 2).unwrap();
        assert!(
            observer
                .observe(3, base, MemoryOperation::Store, 11, 1, 0, 16)
                .unwrap()
                .is_none()
        );
        assert!(observer.pending.is_some());
        let completed = observer
            .observe(
                3,
                AdmissionRequestWitness::new(2, 7, 1, 1, 2).unwrap(),
                MemoryOperation::Store,
                11,
                1,
                1,
                17,
            )
            .unwrap()
            .unwrap();
        assert_eq!(completed.l2_requests, 2);
        assert_eq!(completed.unique_regions, 1);
        state
            .observe(
                usize::from(completed.witness.subcore()),
                usize::try_from(completed.witness.warp()).unwrap(),
                completed.operation,
                completed.hit_sectors,
                completed.miss_sectors,
                completed.issue_time,
            )
            .unwrap();
        assert!(observer.pending.is_none());
        let result = state.diagnostics().entries()[0];
        assert_eq!(result.instructions_observed, 1);
        assert_eq!(result.l2_hit_sectors_observed, 1);
        assert_eq!(result.l2_miss_sectors_observed, 1);
        assert_eq!(result.service_busy_cycles, 11);
        assert_eq!(result.terminal_service_completion_time, 22);
    }

    #[test]
    fn feedback_admission_parks_warp_until_queue_service_start() {
        let sched = schedule(vec![
            (0, slot(cb(1), "STG.E")),
            (16, slot(cb(1), "STG.E")),
            (32, slot(cb(1), "STG.E")),
            (48, slot(cb(1), "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalStore { off: 1, len: 1 },
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::GlobalStore { off: 2, len: 1 },
                    },
                    ReplayInst {
                        pc: 48,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors: vec![0, 4, 8],
            }],
        }];
        let per_sm = BTreeMap::from([(0, blocks)]);
        let candidate = MemoryAdmissionCandidate {
            candidate_id: 1,
            scope: MemoryServiceCandidateScope::Sm,
            operation: MemoryServiceCandidateOperation::Store,
            queue_capacity_instructions: 1,
            admission_latency_cycles: 8,
            l2_hit_sector_units: 1,
            l2_hit_sector_cycles: 1,
            l2_miss_sector_units: 1,
            l2_miss_sector_cycles: 10,
            l2_miss_service_width_sectors: 0,
            membar_waits_for_service_completion: false,
            write_completion_latency_cycles: 0,
            write_completion_request_cycles: 0,
            write_completion_region_cycles: 0,
            write_completion_parallel_lanes: 0,
        };
        let mut l2 = l2_model();
        let result = simulate_kernel_shared_with_residency_and_candidates(
            &per_sm,
            &sched,
            &params(),
            SharedMemoryResidency::unconstrained_fixture(),
            l1_config(),
            &mut l2,
            DramBw::Unmodeled { arch: ArchId::Sm86 },
            &[],
            &[],
            Some(candidate),
        )
        .unwrap();
        let sm = &result.per_sm[&0];
        assert_eq!(sm.cycles, 12);
        assert_eq!(sm.readiness.schema_version(), 10);
        assert!(sm.readiness.memory_write_ack.is_none());
        assert_eq!(sm.readiness.limiting_cycles.memory_admission_queue, 9);
        let feedback = sm
            .readiness
            .memory_admission_feedback
            .as_deref()
            .expect("feedback candidate diagnostics are present");
        assert_eq!(feedback.instructions_admitted, 3);
        assert_eq!(feedback.l2_miss_sectors_observed, 3);
        assert_eq!(feedback.terminal_service_completion_time, 30);
    }

    #[test]
    fn feedback_completion_parks_gpu_membar_on_issuing_warp_frontier() {
        let sched = schedule(vec![
            (0, slot(cb(1), "CS2R R8, SR_CLOCKLO")),
            (16, slot(cb(1), "STG.E")),
            (32, slot(cb(1), "CS2R R9, SR_CLOCKLO")),
            (48, slot(cb(1), "MEMBAR.SC.GPU")),
            (64, slot(cb(1), "CS2R R10, SR_CLOCKLO")),
            (80, slot(cb(1), "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::NonMemory,
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::NonMemory,
                    },
                    ReplayInst {
                        pc: 48,
                        kind: AccessKind::NonMemory,
                    },
                    ReplayInst {
                        pc: 64,
                        kind: AccessKind::NonMemory,
                    },
                    ReplayInst {
                        pc: 80,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors: vec![0],
            }],
        }];
        let per_sm = BTreeMap::from([(0, blocks)]);
        let candidate = MemoryAdmissionCandidate {
            candidate_id: 2,
            scope: MemoryServiceCandidateScope::Sm,
            operation: MemoryServiceCandidateOperation::Store,
            queue_capacity_instructions: 1,
            admission_latency_cycles: 8,
            l2_hit_sector_units: 1,
            l2_hit_sector_cycles: 1,
            l2_miss_sector_units: 1,
            l2_miss_sector_cycles: 10,
            l2_miss_service_width_sectors: 0,
            membar_waits_for_service_completion: true,
            write_completion_latency_cycles: 0,
            write_completion_request_cycles: 0,
            write_completion_region_cycles: 0,
            write_completion_parallel_lanes: 0,
        };
        let mut l2 = l2_model();
        let result = simulate_kernel_shared_with_residency_and_candidates(
            &per_sm,
            &sched,
            &params(),
            SharedMemoryResidency::unconstrained_fixture(),
            l1_config(),
            &mut l2,
            DramBw::Unmodeled { arch: ArchId::Sm86 },
            &[],
            &[],
            Some(candidate),
        )
        .unwrap();
        let readiness = &result.per_sm[&0].readiness;
        assert_eq!(readiness.schema_version(), 22);
        let write_ack = readiness
            .memory_write_ack
            .as_deref()
            .expect("completion feedback emits a drained write acknowledgement");
        assert_eq!(write_ack.acks_scheduled, 1);
        assert_eq!(write_ack.acks_processed, 1);
        assert_eq!(write_ack.latest_issue_time, 1);
        assert_eq!(write_ack.latest_acceptance_time, 1);
        assert_eq!(write_ack.latest_service_completion_time, 11);
        assert_eq!(write_ack.latest_fence_visibility_time, 11);
        assert_eq!(write_ack.terminal_drain_time, 11);
        assert_eq!(result.terminal_event_time, Some(11));
        assert_eq!(readiness.limiting_cycles.memory_write_completion, 9);
        assert_eq!(readiness.clock_issue.warps_with_three_reads, 1);
        assert_eq!(
            readiness.clock_issue.maximum_first_to_second_issue_cycles,
            2
        );
        assert_eq!(
            readiness.clock_issue.maximum_second_to_third_issue_cycles,
            10
        );

        let mut latency_candidate = candidate;
        latency_candidate.write_completion_latency_cycles = 3;
        let mut latency_l2 = l2_model();
        let latency_result = simulate_kernel_shared_with_residency_and_candidates(
            &per_sm,
            &sched,
            &params(),
            SharedMemoryResidency::unconstrained_fixture(),
            l1_config(),
            &mut latency_l2,
            DramBw::Unmodeled { arch: ArchId::Sm86 },
            &[],
            &[],
            Some(latency_candidate),
        )
        .unwrap();
        let latency_readiness = &latency_result.per_sm[&0].readiness;
        assert_eq!(latency_readiness.schema_version(), 22);
        let latency_ack = latency_readiness
            .memory_write_ack
            .as_deref()
            .expect("visibility latency emits a drained write acknowledgement");
        assert_eq!(latency_ack.latest_service_completion_time, 11);
        assert_eq!(latency_ack.latest_fence_visibility_time, 14);
        assert_eq!(latency_ack.terminal_drain_time, 14);
        assert_eq!(latency_result.terminal_event_time, Some(14));
        assert_eq!(
            latency_readiness.limiting_cycles.memory_write_completion,
            12
        );
        assert_eq!(
            latency_readiness
                .clock_issue
                .maximum_second_to_third_issue_cycles,
            13
        );
    }

    #[test]
    fn feedback_write_ack_survives_warp_exit_and_groups_multi_sector_store() {
        let sched = schedule(vec![(0, slot(cb(1), "STG.E")), (16, slot(cb(1), "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 2 },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors: vec![0, 4],
            }],
        }];
        let candidate = MemoryAdmissionCandidate {
            candidate_id: 3,
            scope: MemoryServiceCandidateScope::Sm,
            operation: MemoryServiceCandidateOperation::Store,
            queue_capacity_instructions: 1,
            admission_latency_cycles: 8,
            l2_hit_sector_units: 1,
            l2_hit_sector_cycles: 1,
            l2_miss_sector_units: 1,
            l2_miss_sector_cycles: 10,
            l2_miss_service_width_sectors: 0,
            membar_waits_for_service_completion: true,
            write_completion_latency_cycles: 0,
            write_completion_request_cycles: 0,
            write_completion_region_cycles: 0,
            write_completion_parallel_lanes: 0,
        };
        let per_sm = BTreeMap::from([(0, blocks)]);
        let mut l2 = l2_model();
        let result = simulate_kernel_shared_with_residency_and_candidates(
            &per_sm,
            &sched,
            &params(),
            SharedMemoryResidency::unconstrained_fixture(),
            l1_config(),
            &mut l2,
            DramBw::Unmodeled { arch: ArchId::Sm86 },
            &[],
            &[],
            Some(candidate),
        )
        .unwrap();

        let sm = &result.per_sm[&0];
        assert_eq!(sm.cycles, 2, "the issuing warp exits before its ack");
        assert_eq!(result.terminal_event_time, Some(20));
        let ack = sm
            .readiness
            .memory_write_ack
            .as_deref()
            .expect("timing-active feedback emits a write ack");
        assert_eq!(ack.acks_scheduled, 1);
        assert_eq!(ack.acks_processed, 1);
        assert_eq!(ack.maximum_pending_acks, 1);
        assert_eq!(ack.latest_issue_time, 0);
        assert_eq!(ack.latest_acceptance_time, 0);
        assert_eq!(ack.latest_service_completion_time, 20);
        assert_eq!(ack.latest_fence_visibility_time, 20);
        assert_eq!(ack.terminal_drain_time, 20);
        assert_eq!(result.l2_by_source.source(L2Source::Store), (2, 0));
        assert!(result.unmodeled_tails.dirty_writeback);
        assert!(result.unmodeled_tails.coherence);

        // The architecture memory pipeline timestamps the L2 mutation in the
        // future relative to scheduler issue.  Feedback pricing must keep that
        // timestamp in the actor horizon; running past it used to fail with
        // `HorizonBeforeClock` on real campaign traces.
        let mut pipeline_params = params();
        pipeline_params.memory_issue_pipeline = Some(
            ArchId::Sm86
                .memory_issue_pipeline()
                .require("test memory issue pipeline")
                .unwrap(),
        );
        let mut pipeline_l2 = l2_model();
        let pipeline_result = simulate_kernel_shared_with_residency_and_candidates(
            &per_sm,
            &sched,
            &pipeline_params,
            SharedMemoryResidency::unconstrained_fixture(),
            l1_config(),
            &mut pipeline_l2,
            DramBw::Unmodeled { arch: ArchId::Sm86 },
            &[],
            &[],
            Some(candidate),
        )
        .unwrap();
        let pipeline_readiness = &pipeline_result.per_sm[&0].readiness;
        let pipeline = pipeline_readiness
            .memory_issue_pipeline
            .as_deref()
            .expect("store uses the architecture memory pipeline");
        let pipeline_ack = pipeline_readiness
            .memory_write_ack
            .as_deref()
            .expect("future L2 pricing still drains the write ack");
        assert_eq!(pipeline.terminal_shared_accept_cycle, 14);
        assert_eq!(pipeline_ack.latest_issue_time, 14);
        assert_eq!(pipeline_ack.latest_service_completion_time, 34);
        assert_eq!(pipeline_ack.terminal_drain_time, 34);
        assert_eq!(pipeline_result.terminal_event_time, Some(34));
    }

    #[test]
    fn feedback_observer_prices_contiguous_same_cycle_smsp_groups() {
        let sched = schedule(vec![(0, slot(cb(1), "STG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: (0..4)
                .map(|warp_slot| WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 2 },
                    }],
                    sectors: vec![i64::from(warp_slot) * 8, i64::from(warp_slot) * 8 + 4],
                })
                .collect(),
        }];
        let per_sm = BTreeMap::from([(0, blocks)]);
        let candidate = MemoryAdmissionCandidate {
            candidate_id: 2,
            scope: MemoryServiceCandidateScope::Smsp,
            operation: MemoryServiceCandidateOperation::Store,
            queue_capacity_instructions: 1,
            admission_latency_cycles: 8,
            l2_hit_sector_units: 1,
            l2_hit_sector_cycles: 1,
            l2_miss_sector_units: 1,
            l2_miss_sector_cycles: 10,
            l2_miss_service_width_sectors: 0,
            membar_waits_for_service_completion: false,
            write_completion_latency_cycles: 0,
            write_completion_request_cycles: 0,
            write_completion_region_cycles: 0,
            write_completion_parallel_lanes: 0,
        };
        let mut l2 = l2_model();
        let result = simulate_kernel_shared_with_residency_and_candidates(
            &per_sm,
            &sched,
            &params(),
            SharedMemoryResidency::unconstrained_fixture(),
            l1_config(),
            &mut l2,
            DramBw::Unmodeled { arch: ArchId::Sm86 },
            &[],
            &[],
            Some(candidate),
        )
        .unwrap();
        let sm = &result.per_sm[&0];
        let diagnostics = sm
            .readiness
            .memory_admission_feedback
            .as_deref()
            .expect("feedback candidate diagnostics are present");
        assert_eq!(sm.instructions_issued, 4);
        assert_eq!(sm.cycles, 1);
        assert_eq!(diagnostics.instructions_admitted, 4);
        assert_eq!(diagnostics.l2_hit_sectors_observed, 0);
        assert_eq!(diagnostics.l2_miss_sectors_observed, 8);
        assert_eq!(diagnostics.service_busy_cycles, 80);
        assert_eq!(diagnostics.maximum_waiting_instructions, 1);
        assert_eq!(diagnostics.terminal_service_completion_time, 80);
    }
    use crate::control_bits::{ControlBits, DepBarrier, WaitMask};
    use crate::issue::{AccessKind, IssueResourceSource, ReplayInst, WarpReplay};
    use crate::program::{
        IssueSlot, OpClass, PipelineClass, is_clocklo_opcode_text, is_gpu_membar_opcode_text,
    };

    fn cb(stall: u8) -> ControlBits {
        ControlBits::new(stall, false, 7, 7, 0, 0).expect("valid test controls")
    }

    fn barrier(value: u8) -> DepBarrier {
        DepBarrier::new(value).expect("valid test barrier")
    }

    fn wait_mask(bits: u8) -> WaitMask {
        WaitMask::new(bits).expect("valid test wait mask")
    }

    fn replyless_request(
        issue_time: u64,
        sector_id: i64,
        operation: MemoryOperation,
    ) -> MemRequest {
        MemRequest {
            issue_time,
            sector_id,
            origin: MemoryOrigin::Global,
            sector_mask: 1 << (sector_id & 3),
            local_sector_mask: 0,
            source: L2Source::Store,
            operation,
            constant_load: false,
            constant_deep: false,
            reply: ReplyTo::None,
            admission_witness: None,
        }
    }

    #[test]
    fn emitted_request_batching_is_narrow_and_preserves_cross_sm_fifo() {
        let mut queue = EventQueue::new();
        let mut unpriced = UnpricedBounds(vec![BTreeMap::new(), BTreeMap::new()]);
        let mut sm0 = vec![
            replyless_request(5, 4, MemoryOperation::Store),
            replyless_request(5, 5, MemoryOperation::Store),
        ];
        let mut sm1 = vec![
            replyless_request(5, 8, MemoryOperation::Reduction),
            replyless_request(5, 9, MemoryOperation::Reduction),
            replyless_request(6, 12, MemoryOperation::Store),
            replyless_request(6, 13, MemoryOperation::Atomic),
        ];
        schedule_emitted_requests(
            0,
            TargetId(0),
            &mut queue,
            &mut unpriced,
            &mut sm0,
            34,
            true,
            None,
        )
        .unwrap();
        schedule_emitted_requests(
            1,
            TargetId(1),
            &mut queue,
            &mut unpriced,
            &mut sm1,
            34,
            true,
            None,
        )
        .unwrap();
        assert!(sm0.is_empty() && sm1.is_empty());
        assert_eq!(queue.len(), 4);

        let first = queue.pop().unwrap().unwrap();
        assert_eq!((first.time, first.seq, first.target), (5, 0, TargetId(0)));
        assert!(matches!(
            first.kind,
            Ev::L2AccessBatch(requests) if requests.len() == 2
        ));
        let second = queue.pop().unwrap().unwrap();
        assert_eq!(
            (second.time, second.seq, second.target),
            (5, 2, TargetId(1))
        );
        assert!(matches!(
            second.kind,
            Ev::L2AccessBatch(requests) if requests.len() == 2
        ));
        let third = queue.pop().unwrap().unwrap();
        assert!(matches!(
            third.kind,
            Ev::L2Access(request)
                if request.operation == MemoryOperation::Store && request.issue_time == 6
        ));
        let fourth = queue.pop().unwrap().unwrap();
        assert!(matches!(
            fourth.kind,
            Ev::L2Access(request)
                if request.operation == MemoryOperation::Atomic && request.issue_time == 6
        ));

        let mut legacy_queue = EventQueue::new();
        let mut legacy_unpriced = UnpricedBounds(vec![BTreeMap::new()]);
        let mut legacy = vec![
            replyless_request(5, 4, MemoryOperation::Store),
            replyless_request(5, 5, MemoryOperation::Store),
        ];
        schedule_emitted_requests(
            0,
            TargetId(0),
            &mut legacy_queue,
            &mut legacy_unpriced,
            &mut legacy,
            34,
            false,
            None,
        )
        .unwrap();
        assert_eq!(legacy_queue.len(), 2);
        assert!(matches!(
            legacy_queue.pop().unwrap().unwrap().kind,
            Ev::L2Access(_)
        ));
        assert!(matches!(
            legacy_queue.pop().unwrap().unwrap().kind,
            Ev::L2Access(_)
        ));
    }

    #[cfg(feature = "performance-counters")]
    #[test]
    fn performance_counters_separate_logical_and_queue_events() {
        let mut counters = EventPerformanceCounters::default();
        let requests = (0..4)
            .map(|sector| replyless_request(7, sector, MemoryOperation::Store))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        counters
            .observe(&Ev::L2AccessBatch(requests), 7, 3, 4)
            .unwrap();
        counters.observe(&Ev::SmWake, 7, 2, 1).unwrap();
        assert_eq!(counters.processed, 5);
        assert_eq!(counters.queue_events_processed, 2);
        assert_eq!(counters.l2_access, 4);
        assert_eq!(counters.l2_access_batches, 1);
        assert_eq!(counters.l2_access_batched_requests, 4);
        assert_eq!(counters.same_cycle_after_first, 4);
        assert_eq!(counters.queue_same_cycle_after_first, 1);
        assert_eq!(counters.maximum_queue_depth, 3);
    }

    fn slot(control: ControlBits, text: &str) -> IssueSlot {
        IssueSlot {
            control,
            op_class: OpClass::Other,
            async_copy_wait_max_pending_groups: 0,
            pipeline_class: PipelineClass::DispatchOnly,
            mem_role: if text.trim_start().starts_with("LDGSTS") {
                sim_mem::MemRole::BypassLoad(CacheOpToken::Ldgsts)
            } else {
                sim_mem::MemRole::None
            },
            ldgsts_l1_policy: text.trim_start().starts_with("LDGSTS").then_some(
                if text.split('.').any(|modifier| modifier == "BYPASS") {
                    LdgstsL1Policy::BypassL1
                } else {
                    LdgstsL1Policy::CacheAtAllLevels
                },
            ),
            fixed_latency_operands: crate::sass_operands::project_fixed_latency_operands(
                text,
                control,
                ArchId::Sm86,
            ),
            clocklo_read: is_clocklo_opcode_text(text),
            gpu_membar: is_gpu_membar_opcode_text(text),
            opcode_text: text.to_owned(),
        }
    }

    fn classified_slot(control: ControlBits, op_class: OpClass, text: &str) -> IssueSlot {
        IssueSlot {
            op_class,
            ..slot(control, text)
        }
    }

    fn schedule(slots: Vec<(u64, IssueSlot)>) -> FunctionSchedule {
        FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k___0".to_owned(),
            function_name: "k".to_owned(),
            by_pc: slots.into_iter().collect::<BTreeMap<_, _>>(),
        }
    }

    fn params() -> IssueModelParams {
        IssueModelParams {
            arch: ArchId::Sm86,
            resource_source: IssueResourceSource::SyntheticTestFixture,
            instruction_front_end: None,
            memory_issue_pipeline: None,
            scalar_execution_pipeline: None,
            ldgsts_l1_hit_return_service: None,
            constant_memory_return_service: None,
            immediate_constant_cache: None,
            l1_hit_latency: 34,
            l1_miss_latency: 244,
            global_flat_latency: 423,
            shared_write_barrier_latency: 23,
            shared_additional_wavefront_latency: 2,
            non_memory_write_barrier_latency: 25,
            read_barrier_latency: 8,
            ldgsts_l1_mshr_floor: NonZeroU32::new(1).unwrap(),
            ldgsts_completion_throughput: LdgstsCompletionThroughput {
                cache_all_payload_bytes_per_cycle: 64,
                bypass_payload_bytes_per_cycle: 32,
            },
            ldgsts_cache_all_destination_wavefront_cycles: 2,
            ldgsts_global_to_shared_transport_latency: 0,
            ldgsts_cache_all_destination_service_wavefronts: 2,
            ldgsts_cache_all_destination_service_cycles: 3,
            max_resident_warps_per_sm: 48,
            max_resident_blocks_per_sm: u32::MAX,
        }
    }

    fn l1_config() -> L1Config {
        L1Config {
            arch: ArchId::Sm86,
            line_capacity: 768,
            mshr_count: Some(32),
            hit_latency: 34,
            miss_latency: 244,
        }
    }

    fn l2_model() -> L2Model {
        L2Model::new(L2Config {
            arch: ArchId::Sm86,
            capacity_bytes: 1 << 20,
            line_bytes: 128,
            sector_bytes: 32,
            num_slices: 2,
            associativity: 16,
            replacement: L2Replacement::Lru,
            slice_hash: SliceHash::LineMod,
        })
        .unwrap()
    }

    fn single_way_l2_model() -> L2Model {
        L2Model::new(L2Config {
            arch: ArchId::Sm86,
            capacity_bytes: 128,
            line_bytes: 128,
            sector_bytes: 32,
            num_slices: 1,
            associativity: 1,
            replacement: L2Replacement::Lru,
            slice_hash: SliceHash::LineMod,
        })
        .unwrap()
    }

    struct L2TestFaultGuard;

    impl Drop for L2TestFaultGuard {
        fn drop(&mut self) {
            TEST_L2_COUNTER_ARCH_OVERRIDE.with(|value| value.set(None));
            TEST_L2_ACCESS_ERROR_AFTER.with(|value| value.set(None));
            TEST_REQUEST_MASK_OVERRIDE.with(|value| value.set(None));
            TEST_REQUEST_OPERATION_OVERRIDE.with(|value| value.set(None));
        }
    }

    fn override_l2_counter_architecture(arch: ArchId) -> L2TestFaultGuard {
        let previous = TEST_L2_COUNTER_ARCH_OVERRIDE.with(|value| value.replace(Some(arch)));
        assert!(previous.is_none(), "nested L2 counter override");
        L2TestFaultGuard
    }

    fn inject_l2_error_after(successful_accesses: usize) -> L2TestFaultGuard {
        let previous =
            TEST_L2_ACCESS_ERROR_AFTER.with(|value| value.replace(Some(successful_accesses)));
        assert!(previous.is_none(), "nested L2 access-error injection");
        L2TestFaultGuard
    }

    fn override_next_request_masks(override_value: TestRequestMaskOverride) -> L2TestFaultGuard {
        let previous = TEST_REQUEST_MASK_OVERRIDE.with(|value| value.replace(Some(override_value)));
        assert!(previous.is_none(), "nested request-mask override");
        L2TestFaultGuard
    }

    fn override_next_request_operation(operation: MemoryOperation) -> L2TestFaultGuard {
        let previous = TEST_REQUEST_OPERATION_OVERRIDE.with(|value| value.replace(Some(operation)));
        assert!(previous.is_none(), "nested request-operation override");
        L2TestFaultGuard
    }

    /// The sm_86 carded rate (~747 B/SM-cycle): far above any of these
    /// tests' byte demand, so the server is ACTIVE but engages nothing —
    /// every pre-server cycle pin holds with the realistic covariate.
    fn carded_bw() -> DramBw {
        DramBw::Carded {
            arch: ArchId::Sm86,
            bytes_per_sec: 1_008_000_000_000,
            sm_clock_hz: 1_350_000_000,
        }
    }

    fn assert_all_current_traffic_is_global(result: &SystemResult) {
        for source in [
            L2Source::L1Fill,
            L2Source::Bypass,
            L2Source::Untagged,
            L2Source::Store,
        ] {
            let total = result.l2_by_source.source(source);
            let global = result
                .l2_by_source
                .source_for_origin(MemoryOrigin::Global, source);
            let local = result
                .l2_by_source
                .source_for_origin(MemoryOrigin::Local, source);
            assert_eq!(total.0, global.0 + local.0);
            assert_eq!(total.1, global.1 + local.1);
            assert_eq!(local, (0, 0));
        }
        let global = result.dram.origin(MemoryOrigin::Global);
        let local = result.dram.origin(MemoryOrigin::Local);
        assert_eq!(
            result.dram.read_sectors,
            global.read_sectors + local.read_sectors
        );
        assert_eq!(
            result.dram.bw_delayed_responses,
            global.bw_delayed_responses + local.bw_delayed_responses
        );
        assert_eq!(local, DramOriginStats::default());
    }

    fn dependent_load_blocks(sector: i64) -> Vec<BlockReplay> {
        dependent_load_span_blocks(vec![sector])
    }

    fn dependent_load_span_blocks(sectors: Vec<i64>) -> Vec<BlockReplay> {
        let len = u16::try_from(sectors.len()).unwrap();
        vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors,
            }],
        }]
    }

    fn load_consumer_schedule() -> FunctionSchedule {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        schedule(vec![(0, slot(load, "LDG.E")), (16, slot(consumer, "EXIT"))])
    }

    fn async_copy_blocks(with_wait: bool) -> Vec<BlockReplay> {
        let mut insts = vec![ReplayInst {
            pc: 0,
            kind: AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 1,
                completion_work: 1024,
                l1_policy: LdgstsL1Policy::BypassL1,
            },
        }];
        if with_wait {
            insts.extend([
                ReplayInst {
                    pc: 16,
                    kind: AccessKind::NonMemory,
                },
                ReplayInst {
                    pc: 32,
                    kind: AccessKind::NonMemory,
                },
                ReplayInst {
                    pc: 48,
                    kind: AccessKind::NonMemory,
                },
            ]);
        }
        vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts,
                sectors: vec![40],
            }],
        }]
    }

    fn async_copy_schedule(with_wait: bool) -> FunctionSchedule {
        let mut slots = vec![(0, slot(cb(1), "LDGSTS.E.BYPASS"))];
        if with_wait {
            slots.extend([
                (
                    16,
                    classified_slot(
                        ControlBits::new(1, true, 0, 7, 0, 0).unwrap(),
                        OpClass::AsyncCopyCommit,
                        "LDGDEPBAR",
                    ),
                ),
                (
                    32,
                    classified_slot(cb(1), OpClass::AsyncCopyWait, "DEPBAR.LE SB0, 0x0"),
                ),
                (48, slot(cb(1), "EXIT")),
            ]);
        }
        schedule(slots)
    }

    fn cacheable_async_copy_blocks(with_wait: bool) -> Vec<BlockReplay> {
        let mut blocks = async_copy_blocks(with_wait);
        let AccessKind::GlobalAsyncCopy {
            completion_work,
            l1_policy,
            ..
        } = &mut blocks[0].warps[0].insts[0].kind
        else {
            unreachable!("async-copy fixture owns a global async source")
        };
        *completion_work = 512;
        *l1_policy = LdgstsL1Policy::CacheAtAllLevels;
        blocks
    }

    fn cacheable_async_copy_schedule(with_wait: bool) -> FunctionSchedule {
        let mut slots = vec![(0, slot(cb(1), "LDGSTS.E"))];
        if with_wait {
            slots.extend([
                (
                    16,
                    classified_slot(
                        ControlBits::new(1, true, 0, 7, 0, 0).unwrap(),
                        OpClass::AsyncCopyCommit,
                        "LDGDEPBAR",
                    ),
                ),
                (
                    32,
                    classified_slot(cb(1), OpClass::AsyncCopyWait, "DEPBAR.LE SB0, 0x0"),
                ),
                (48, slot(cb(1), "EXIT")),
            ]);
        }
        schedule(slots)
    }

    #[test]
    fn cacheable_async_copy_uses_l1_fill_response_path() {
        let sched = cacheable_async_copy_schedule(true);
        let per_sm = BTreeMap::from([(0, cacheable_async_copy_blocks(true))]);
        let mut l2 = l2_model();

        let result = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!(result.per_sm[&0].instructions_issued, 4);
        assert_eq!(result.per_sm[&0].cycles, 433);
        let l1 = result.per_sm[&0].l1.unwrap();
        assert_eq!(l1.misses, 1);
        assert_eq!(l1.bypass_sectors, 0);
        assert_eq!(result.l2_by_source.source(L2Source::L1Fill), (1, 0));
        assert_eq!(result.l2_by_source.source(L2Source::Bypass), (0, 0));
        assert_eq!(result.dram.read_sectors, 1);
        assert_eq!(result.terminal_event_time, Some(423));
    }

    #[test]
    fn async_copy_wait_uses_dram_miss_response_time() {
        let sched = async_copy_schedule(true);
        let per_sm = BTreeMap::from([(0, async_copy_blocks(true))]);
        let mut l2 = l2_model();

        let result = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!(result.per_sm[&0].instructions_issued, 4);
        assert_eq!(result.per_sm[&0].cycles, 441);
        assert_eq!(result.l2_by_source.source(L2Source::Bypass), (1, 0));
        assert_eq!(result.dram.read_sectors, 1);
        assert_eq!(result.terminal_event_time, Some(423));
    }

    #[test]
    fn async_copy_wait_uses_l2_hit_response_time() {
        let sched = async_copy_schedule(true);
        let per_sm = BTreeMap::from([(0, async_copy_blocks(true))]);
        let mut l2 = l2_model();
        let warm = l2
            .access_mask_event_transactional(10, 0b0001, L2OpClass::Read, |_| {
                Ok::<(), sim_mem::L2AccessError>(())
            })
            .unwrap()
            .0;
        l2.complete_fill(warm.fill.expect("cold warmup mints a fill"))
            .unwrap();

        let result = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!(result.per_sm[&0].cycles, 262);
        assert_eq!(result.l2_by_source.source(L2Source::Bypass), (1, 1));
        assert_eq!(result.dram.read_sectors, 0);
        assert_eq!(result.terminal_event_time, Some(244));
    }

    #[test]
    fn equal_time_async_copy_responses_are_deterministic() {
        let sched = async_copy_schedule(true);
        let run = || {
            let mut blocks = async_copy_blocks(true);
            blocks[0].warps[0].insts[0].kind = AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 2,
                completion_work: 1024,
                l1_policy: LdgstsL1Policy::BypassL1,
            };
            blocks[0].warps[0].sectors = vec![40, 41];
            let per_sm = BTreeMap::from([(0, blocks)]);
            let mut l2 = l2_model();
            simulate_kernel_shared(
                &per_sm,
                &sched,
                &params(),
                l1_config(),
                &mut l2,
                carded_bw(),
            )
            .unwrap()
        };

        // The carded server can service both 32 B reads inside one cycle, so
        // the fixed 423-cycle DRAM arm gives both response events the same time.
        // Their queue sequence and final group completion must be reproducible.
        let first = run();
        let second = run();
        for result in [&first, &second] {
            assert_eq!(result.per_sm[&0].instructions_issued, 4);
            assert_eq!(result.per_sm[&0].cycles, 441);
            assert_eq!(result.l2_by_source.source(L2Source::Bypass), (2, 0));
            assert_eq!(result.dram.read_sectors, 2);
            assert_eq!(result.dram.bw_delayed_responses, 0);
            assert_eq!(result.terminal_event_time, Some(423));
        }
        assert_eq!(first.l2, second.l2);
        assert_eq!(first.l2_by_source, second.l2_by_source);
        assert_eq!(first.dram, second.dram);
        assert_eq!(first.per_sm[&0].cycles, second.per_sm[&0].cycles);
        assert_eq!(first.terminal_event_time, second.terminal_event_time);
    }

    #[test]
    fn final_async_copy_response_survives_past_issue_frontier() {
        let sched = async_copy_schedule(false);
        let per_sm = BTreeMap::from([(0, async_copy_blocks(false))]);
        let mut l2 = l2_model();

        let result = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!(result.per_sm[&0].cycles, 1);
        assert_eq!(result.per_sm[&0].instructions_issued, 1);
        assert_eq!(result.terminal_event_time, Some(423));
        assert_eq!(result.l2_by_source.source(L2Source::Bypass), (1, 0));
        assert_eq!(result.dram.read_sectors, 1);
    }

    #[test]
    fn empty_system_still_validates_l1_timing_lockstep() {
        let sched = schedule(Vec::new());
        let per_sm = BTreeMap::new();
        let mut cfg = l1_config();
        cfg.miss_latency += 1;
        let mut l2 = l2_model();
        assert!(matches!(
            simulate_kernel_shared(&per_sm, &sched, &params(), cfg, &mut l2, carded_bw(),),
            Err(SystemError::Issue(IssueError::L1TimingMismatch {
                field: "miss",
                expected: 244,
                got: 245,
            }))
        ));
    }

    #[test]
    fn empty_system_still_rejects_feedback_operand_latency_drift() {
        let sched = schedule(Vec::new());
        let mut l2 = l2_model();
        let before = *l2.counters().unwrap();
        let candidate = MemoryAdmissionCandidate {
            candidate_id: 3,
            scope: MemoryServiceCandidateScope::Sm,
            operation: MemoryServiceCandidateOperation::Store,
            queue_capacity_instructions: 1,
            admission_latency_cycles: 7,
            l2_hit_sector_units: 1,
            l2_hit_sector_cycles: 1,
            l2_miss_sector_units: 1,
            l2_miss_sector_cycles: 10,
            l2_miss_service_width_sectors: 0,
            membar_waits_for_service_completion: false,
            write_completion_latency_cycles: 0,
            write_completion_request_cycles: 0,
            write_completion_region_cycles: 0,
            write_completion_parallel_lanes: 0,
        };
        assert!(matches!(
            simulate_kernel_shared_with_residency_and_candidates(
                &BTreeMap::new(),
                &sched,
                &params(),
                SharedMemoryResidency::unconstrained_fixture(),
                l1_config(),
                &mut l2,
                carded_bw(),
                &[],
                &[],
                Some(candidate),
            ),
            Err(SystemError::Issue(
                IssueError::MemoryAdmissionFeedbackLatencyMismatch {
                    candidate_id: 3,
                    configured_cycles: 7,
                    issue_model_cycles: 8,
                }
            ))
        ));
        assert_eq!(*l2.counters().unwrap(), before);
    }

    #[test]
    fn shared_system_does_not_inherit_sm86_pending_reads_on_sm120() {
        assert!(matches!(
            require_l2_pending_read_semantics(ArchId::Sm120),
            Err(SystemError::L2PendingReadCapability(
                sim_arch::CapabilityError::ResearchGap {
                    arch: ArchId::Sm120,
                    capability: "l2_pending_read_semantics",
                    ..
                }
            ))
        ));
    }

    #[test]
    fn empty_and_time_zero_event_boundaries_remain_distinct() {
        let sched = schedule(Vec::new());

        let mut empty_l2 = l2_model();
        let empty = simulate_kernel_shared(
            &BTreeMap::new(),
            &sched,
            &params(),
            l1_config(),
            &mut empty_l2,
            carded_bw(),
        )
        .unwrap();
        assert_eq!(empty.terminal_event_time, None);

        let mut actor_l2 = l2_model();
        let actor = simulate_kernel_shared(
            &BTreeMap::from([(0, Vec::new())]),
            &sched,
            &params(),
            l1_config(),
            &mut actor_l2,
            carded_bw(),
        )
        .unwrap();
        assert_eq!(actor.terminal_event_time, Some(0));
    }

    #[test]
    fn dram_server_accepts_exact_maximum_and_rejects_true_overflow_transactionally() {
        let exact = DramServer::new(DramBw::Carded {
            arch: ArchId::Sm86,
            bytes_per_sec: 32,
            sm_clock_hz: u64::MAX,
        })
        .unwrap();
        assert_eq!(
            exact.preview(0, 32).unwrap(),
            (u128::from(32_u8) * u128::from(u64::MAX), u64::MAX)
        );
        assert_eq!(exact.free_fp, 0);

        let cycle_overflow = DramServer::new(DramBw::Carded {
            arch: ArchId::Sm86,
            bytes_per_sec: 16,
            sm_clock_hz: u64::MAX,
        })
        .unwrap();
        assert!(matches!(
            cycle_overflow.preview(0, 32),
            Err(SystemError::DramCycleOverflow { .. })
        ));
        assert_eq!(cycle_overflow.free_fp, 0);

        let fixed_point_overflow = DramServer {
            free_fp: u128::MAX,
            bytes_per_sec: 1,
            sm_clock_hz: 1,
        };
        assert!(matches!(
            fixed_point_overflow.preview(0, 1),
            Err(SystemError::DramFixedPointOverflow {
                operation: "service completion",
            })
        ));
        assert_eq!(fixed_point_overflow.free_fp, u128::MAX);
    }

    #[test]
    fn dram_cycle_overflow_precedes_l2_mutation() {
        let sched = load_consumer_schedule();
        let per_sm = BTreeMap::from([(0, dependent_load_blocks(4))]);
        let mut l2 = l2_model();
        let err = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            DramBw::Carded {
                arch: ArchId::Sm86,
                bytes_per_sec: 16,
                sm_clock_hz: u64::MAX,
            },
        )
        .unwrap_err();
        assert!(matches!(err, SystemError::DramCycleOverflow { .. }));
        assert_eq!(l2.counters().unwrap().total().unwrap(), 0);
    }

    #[test]
    fn empty_system_rejects_mismatched_l2_counter_architecture() {
        let sched = schedule(Vec::new());
        let _fault = override_l2_counter_architecture(ArchId::Sm120);
        let mut l2 = l2_model();

        let err = simulate_kernel_shared(
            &BTreeMap::new(),
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SystemError::L2(sim_mem::L2AccessError::Counter(
                L2CounterError::Architecture(sim_arch::ArchitectureProvenanceError::Mismatch {
                    component: "L2 counters",
                    expected: ArchId::Sm86,
                    got: ArchId::Sm120,
                })
            ))
        ));
        assert_eq!(l2.config().arch, ArchId::Sm86);
        assert_eq!(l2.counters().unwrap().arch(), Some(ArchId::Sm86));
        assert_eq!(l2.counters().unwrap().total().unwrap(), 0);
    }

    #[test]
    fn later_l2_error_rolls_back_earlier_kernel_accesses() {
        let mut load_control = cb(1);
        load_control.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load_control, "LDG.E"))]);
        let load_blocks = || {
            vec![BlockReplay {
                launch_index: 0,
                warps: vec![WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot: 0,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    }],
                    sectors: vec![4],
                }],
            }]
        };
        // Both loads issue at t=0. The lower-SM request is processed first and
        // mutates the staged cache. Inject the same typed L2 counter error the
        // former public counter seam produced on the second request.
        let per_sm = BTreeMap::from([(0, load_blocks()), (1, load_blocks())]);
        let _fault = inject_l2_error_after(1);
        let mut l2 = l2_model();
        let before = *l2.counters().unwrap();
        assert_eq!(
            l2.preview_mask(1, 0b0001, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            0
        );

        let err = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SystemError::L2(sim_mem::L2AccessError::Counter(L2CounterError::Overflow {
                field: sim_mem::L2CounterField::Hits,
            }))
        ));
        assert_eq!(
            *l2.counters().unwrap(),
            before,
            "a failed kernel published L2 counters"
        );
        assert_eq!(
            l2.preview_mask(1, 0b0001, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            0,
            "the first reply-bearing load escaped the failed kernel transaction"
        );
    }

    #[test]
    fn later_error_in_replyless_batch_rolls_back_earlier_request() {
        let sched = schedule(vec![(0, slot(cb(1), "STG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot: 0,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    }],
                    sectors: vec![4],
                },
                WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot: 1,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    }],
                    sectors: vec![8],
                },
            ],
        }];
        let _fault = inject_l2_error_after(1);
        let mut l2 = l2_model();
        let before = *l2.counters().unwrap();

        let error = simulate_kernel_shared(
            &BTreeMap::from([(0, blocks)]),
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            SystemError::L2(sim_mem::L2AccessError::Counter(L2CounterError::Overflow {
                field: sim_mem::L2CounterField::Hits,
            }))
        ));
        assert_eq!(*l2.counters().unwrap(), before);
        for line in [1, 2] {
            assert_eq!(
                l2.preview_mask(line, 0b0001, L2OpClass::WriteAtomic)
                    .unwrap()
                    .present_sectors,
                0
            );
        }
    }

    #[test]
    fn l2_hit_does_not_preflight_hypothetical_dram_service() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(cb(1), "STG.E")),
            (16, slot(load, "LDG.E")),
            (32, slot(consumer, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalEligible {
                            off: 1,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors: vec![4, 4],
            }],
        }];
        let per_sm = BTreeMap::from([(0, blocks)]);
        let mut l2 = l2_model();
        let result = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            DramBw::Carded {
                arch: ArchId::Sm86,
                bytes_per_sec: 16,
                sm_clock_hz: u64::MAX,
            },
        )
        .unwrap();
        assert_eq!(result.dram.read_sectors, 0);
        assert_eq!(result.l2.hits, 1);
        assert_eq!(result.l2.misses, 1);
    }

    #[test]
    fn typed_horizon_preserves_exact_maximum_bounds() {
        assert_eq!(replay_horizon(None, None), ReplayHorizon::Unbounded);
        assert_eq!(
            replay_horizon(Some(u64::MAX), None),
            ReplayHorizon::Bounded(u64::MAX)
        );
        assert_eq!(
            replay_horizon(None, Some(u64::MAX)),
            ReplayHorizon::Bounded(u64::MAX)
        );
        assert_eq!(
            replay_horizon(Some(u64::MAX), Some(7)),
            ReplayHorizon::Bounded(7)
        );
    }

    #[test]
    fn due_time_mismatch_preserves_the_queued_head() {
        let mut due = DueTimes::new(1);
        due.push(0, 7);
        assert!(matches!(
            due.pop_expect("test", 0, 8),
            Err(SystemError::DueTimeMismatch {
                component: "test",
                sm: 0,
                expected: 8,
                got: Some(7),
            })
        ));
        assert_eq!(due.next(0), Some(7));
    }

    /// A reply-producing load can be the final instruction. Its issue
    /// frontier is t=1, but the modeled response event still completes at
    /// t=423 and must remain visible as a separate diagnostic timestamp.
    #[test]
    fn final_load_response_survives_past_the_issue_frontier() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalEligible {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::LdgECa,
                    },
                }],
                sectors: vec![4],
            }],
        }];
        let mut l2 = l2_model();
        let result = simulate_kernel_shared(
            &BTreeMap::from([(0, blocks)]),
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();
        assert_eq!(result.per_sm[&0].instructions_issued, 1);
        assert_eq!(result.per_sm[&0].cycles, 1);
        assert_eq!(result.terminal_event_time, Some(423));
        assert_all_current_traffic_is_global(&result);
    }

    #[test]
    fn exact_maximum_response_time_is_a_real_terminal_event() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalEligible {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::LdgECa,
                    },
                }],
                sectors: vec![4],
            }],
        }];
        let mut p = params();
        p.global_flat_latency = u64::MAX;
        let mut l2 = l2_model();

        let result = simulate_kernel_shared(
            &BTreeMap::from([(0, blocks)]),
            &sched,
            &p,
            l1_config(),
            &mut l2,
            DramBw::Unmodeled { arch: ArchId::Sm86 },
        )
        .unwrap();

        assert_eq!(result.per_sm[&0].instructions_issued, 1);
        assert_eq!(result.per_sm[&0].cycles, 1);
        assert_eq!(result.terminal_event_time, Some(u64::MAX));
        assert_eq!(result.l2.misses, 1);
        assert_eq!(*l2.counters().unwrap(), result.l2);
    }

    #[test]
    fn response_time_overflow_rolls_back_prior_l2_mutation() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(cb(1), "STG.E")), (16, slot(load, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalEligible {
                            off: 1,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                ],
                // The store mutates line 2 successfully at t=0. The cold load
                // then overflows its response time at t=1 on a different line.
                sectors: vec![8, 4],
            }],
        }];
        let mut p = params();
        p.global_flat_latency = u64::MAX;
        let mut l2 = l2_model();
        let before = *l2.counters().unwrap();

        let error = simulate_kernel_shared(
            &BTreeMap::from([(0, blocks)]),
            &sched,
            &p,
            l1_config(),
            &mut l2,
            DramBw::Unmodeled { arch: ArchId::Sm86 },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            SystemError::Issue(IssueError::TimeOverflow {
                operation: "DRAM response visibility",
                base: 1,
                delta: u64::MAX,
            })
        ));
        assert_eq!(
            *l2.counters().unwrap(),
            before,
            "a failed kernel published staged L2 counters"
        );
        assert_eq!(
            l2.preview_mask(2, 0b0001, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            0,
            "the earlier successful store escaped the failed kernel transaction"
        );
        assert_eq!(
            l2.preview_mask(1, 0b0001, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            0,
            "the overflowing load mutated the caller L2"
        );
    }

    #[test]
    fn local_and_global_share_physical_hierarchy_with_per_sector_attribution() {
        let sched = schedule(vec![
            (0, slot(cb(1), "STL")),
            (16, slot(cb(1), "IADD3")),
            (32, slot(cb(1), "LDG.E")),
            (48, slot(cb(1), "LDL")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![
                WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot: 0,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: AccessKind::LocalStore { off: 0, len: 1 },
                    }],
                    sectors: vec![40],
                },
                WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot: 1,
                    insts: vec![
                        ReplayInst {
                            pc: 16,
                            kind: AccessKind::NonMemory,
                        },
                        ReplayInst {
                            pc: 32,
                            kind: AccessKind::GlobalEligible {
                                off: 0,
                                len: 1,
                                token: CacheOpToken::LdgECa,
                            },
                        },
                    ],
                    sectors: vec![40],
                },
                WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot: 2,
                    insts: vec![
                        ReplayInst {
                            pc: 16,
                            kind: AccessKind::NonMemory,
                        },
                        ReplayInst {
                            pc: 48,
                            kind: AccessKind::LocalLoad { off: 0, len: 1 },
                        },
                    ],
                    sectors: vec![41],
                },
            ],
        }];
        let mut l2 = l2_model();
        let result = simulate_kernel_shared(
            &BTreeMap::from([(0, blocks)]),
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!(result.l2_by_source.source(L2Source::L1Fill), (2, 1));
        assert_eq!(
            result
                .l2_by_source
                .source_for_origin(MemoryOrigin::Global, L2Source::L1Fill),
            (1, 1)
        );
        assert_eq!(
            result
                .l2_by_source
                .source_for_origin(MemoryOrigin::Local, L2Source::L1Fill),
            (1, 0)
        );
        assert_eq!(result.l2_by_source.source(L2Source::Store), (1, 0));
        assert_eq!(
            result
                .l2_by_source
                .source_for_origin(MemoryOrigin::Local, L2Source::Store),
            (1, 0)
        );
        assert_eq!(result.dram.origin(MemoryOrigin::Global).read_sectors, 0);
        assert_eq!(result.dram.origin(MemoryOrigin::Local).read_sectors, 1);

        let l1 = result.per_sm[&0]
            .l1
            .expect("shared replay carries L1 counters");
        assert_eq!(l1.origin(MemoryOrigin::Global).misses, 1);
        let local = l1.origin(MemoryOrigin::Local);
        assert_eq!(local.misses, 1);
        assert_eq!(local.merges, 1);
        assert_eq!(local.store_sectors, 1);
        assert_eq!(local.store_hits, 0);
        l1.validate_origin_partition().unwrap();
        result.l2_by_source.validate_origin_partition().unwrap();
        result.dram.validate_origin_partition().unwrap();
    }

    #[test]
    fn invalid_origin_masks_precede_l2_failures_and_leave_caller_state() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let sched = schedule(vec![(0, slot(load, "LDL"))]);
        let cases = [
            (
                AccessKind::LocalLoad { off: 0, len: 1 },
                TestRequestMaskOverride {
                    origin: MemoryOrigin::Local,
                    sector_mask: 0b0001,
                    local_sector_mask: 0,
                },
            ),
            (
                AccessKind::GlobalEligible {
                    off: 0,
                    len: 1,
                    token: CacheOpToken::LdgECa,
                },
                TestRequestMaskOverride {
                    origin: MemoryOrigin::Global,
                    sector_mask: 0b0001,
                    local_sector_mask: 0b0010,
                },
            ),
        ];

        for (kind, mask_override) in cases {
            let blocks = vec![BlockReplay {
                launch_index: 0,
                warps: vec![WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot: 0,
                    insts: vec![ReplayInst { pc: 0, kind }],
                    sectors: vec![4],
                }],
            }];
            let mut l2 = l2_model();
            let before = *l2.counters().unwrap();
            let _mask_fault = override_next_request_masks(mask_override);
            let _later_l2_fault = inject_l2_error_after(0);

            let error = simulate_kernel_shared(
                &BTreeMap::from([(0, blocks)]),
                &sched,
                &params(),
                l1_config(),
                &mut l2,
                carded_bw(),
            )
            .unwrap_err();

            assert!(matches!(
                error,
                SystemError::OriginMaskInvalid {
                    sector_id: 4,
                    origin,
                    sector_mask,
                    local_sector_mask,
                } if origin == mask_override.origin
                    && sector_mask == mask_override.sector_mask
                    && local_sector_mask == mask_override.local_sector_mask
            ));
            assert_eq!(
                TEST_L2_ACCESS_ERROR_AFTER.with(std::cell::Cell::get),
                Some(0),
                "origin validation reached the later L2/counter stage"
            );
            assert_eq!(*l2.counters().unwrap(), before);
            assert_eq!(
                l2.preview_mask(1, 0b0001, L2OpClass::Read)
                    .unwrap()
                    .present_sectors,
                0,
                "invalid origin metadata published staged L2 state"
            );
        }
    }

    #[test]
    fn invalid_operation_reply_contract_precedes_l2_mutation() {
        let sched = load_consumer_schedule();
        let per_sm = BTreeMap::from([(0, dependent_load_blocks(4))]);
        let mut l2 = l2_model();
        let before = *l2.counters().unwrap();
        let _contract_fault = override_next_request_operation(MemoryOperation::Store);
        let _later_l2_fault = inject_l2_error_after(0);

        let error = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            SystemError::MemoryRequestContractInvalid {
                sector_id: 4,
                request_source: L2Source::L1Fill,
                operation: MemoryOperation::Store,
                reply: ReplyTo::L1Fill { .. },
            }
        ));
        assert_eq!(
            TEST_L2_ACCESS_ERROR_AFTER.with(std::cell::Cell::get),
            Some(0),
            "request contract validation reached the later L2 stage"
        );
        assert_eq!(*l2.counters().unwrap(), before);
        assert_eq!(
            l2.preview_mask(1, 0b0001, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            0
        );
    }

    /// One SM, one cold load: L2 misses → the consumer waits the DRAM
    /// arm (423), not the L2-hit arm (244).
    #[test]
    fn cold_load_prices_the_dram_arm() {
        let sched = load_consumer_schedule();
        let per_sm = BTreeMap::from([(0, dependent_load_blocks(4))]);
        let mut l2 = l2_model();
        let r = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();
        assert_eq!(r.per_sm[&0].instructions_issued, 2);
        assert_eq!(r.per_sm[&0].cycles, 424);
        assert_eq!(r.l2.misses, 1);
        assert_eq!(r.l2.hits, 0);
    }

    /// Two same-cycle reads of one absent sector both observe it pending rather
    /// than valid. They remain distinct L2 lookup misses, but the second read
    /// joins the allocation/sector's original lower-level fetch and ready time.
    #[test]
    fn same_cycle_pending_reads_coalesce_lower_fetch() {
        let sched = load_consumer_schedule();
        let per_sm = BTreeMap::from([(0, dependent_load_blocks(4)), (7, dependent_load_blocks(4))]);
        let mut l2 = l2_model();
        let r = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();
        // Both lookups are ordered before the original data-ready completion at
        // t=423. The matching pending sector shares that one fetch; each SM still
        // receives its own response at the deterministic original completion.
        assert_eq!(r.l2.misses, 2);
        assert_eq!(r.l2.hits, 0);
        assert_eq!(r.dram.read_sectors, 1);
        assert_eq!(r.per_sm[&0].cycles, 424);
        assert_eq!(r.per_sm[&7].cycles, 424);
    }

    #[test]
    fn pending_l2_ready_state_is_generation_sector_typed_and_transactional() {
        let mut l2 = l2_model();
        let (first, first_preview) = l2
            .access_mask_event_coalesced_transactional(1, 0b0001, L2OpClass::Read, |preview| {
                Ok::<_, sim_mem::L2AccessError>(preview)
            })
            .unwrap();
        let fill = first.fill.expect("cold read owns one lower-level fetch");
        let key = first_preview
            .fill_sector_key(0)
            .expect("fill preview exposes its opaque sector identity");
        let mut pending = PendingL2Ready::default();
        pending.preflight_register(first_preview).unwrap();
        pending.register(fill, 423).unwrap();
        assert_eq!(pending.len(), 1);
        assert!(matches!(
            pending.preflight_register(first_preview),
            Err(SystemError::PendingL2ReadyAlreadyRegistered {
                key: duplicate,
                ready: 423,
            }) if duplicate == key
        ));
        assert_eq!(pending.len(), 1, "collision preflight mutated the map");

        let (_, merge_preview) = l2
            .access_mask_event_coalesced_transactional(1, 0b0001, L2OpClass::Read, |preview| {
                Ok::<_, sim_mem::L2AccessError>(preview)
            })
            .unwrap();
        assert_eq!(pending.merged_ready(merge_preview).unwrap(), Some(423));
        assert!(matches!(
            pending.preflight_remove(fill, 422),
            Err(SystemError::PendingL2ReadyTimeMismatch {
                key: mismatched,
                expected: 423,
                got: 422,
            }) if mismatched == key
        ));
        assert_eq!(pending.len(), 1, "time mismatch mutated the map");

        pending.preflight_remove(fill, 423).unwrap();
        pending.commit_remove(fill);
        assert_eq!(pending.len(), 0);
        assert!(matches!(
            pending.merged_ready(merge_preview),
            Err(SystemError::PendingL2ReadyMissing { key: missing }) if missing == key
        ));
        assert!(matches!(
            pending.preflight_remove(fill, 423),
            Err(SystemError::PendingL2ReadyRemovalMissing { key: missing }) if missing == key
        ));
    }

    #[test]
    fn missing_pending_ready_rolls_back_the_kernel_snapshot() {
        let sched = load_consumer_schedule();
        let per_sm = BTreeMap::from([(0, dependent_load_blocks(4))]);
        let mut l2 = l2_model();
        let pending = l2
            .access_mask_event_coalesced_transactional(1, 0b0001, L2OpClass::Read, |_| {
                Ok::<(), sim_mem::L2AccessError>(())
            })
            .unwrap()
            .0
            .fill
            .expect("fixture leaves one sector pending");
        let before = *l2.counters().unwrap();

        let error = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap_err();

        assert!(matches!(error, SystemError::PendingL2ReadyMissing { .. }));
        assert_eq!(*l2.counters().unwrap(), before);
        assert_eq!(
            l2.complete_fill(pending).unwrap(),
            L2FillCompletion::Completed {
                newly_valid_mask: 0b0001,
                duplicate_mask: 0,
            },
            "failed replay mutated the caller's pending allocation"
        );
    }

    #[test]
    fn mixed_new_and_pending_mask_fetches_only_new_sectors() {
        let sched = load_consumer_schedule();
        let per_sm = BTreeMap::from([
            (0, dependent_load_span_blocks(vec![4, 5])),
            (7, dependent_load_span_blocks(vec![4, 5, 6])),
        ]);
        let mut l2 = l2_model();

        let result = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!(result.l2.misses, 5, "pending sectors remain lookup misses");
        assert_eq!(result.l2.hits, 0);
        assert_eq!(result.l2.line_misses, 1);
        assert_eq!(result.l2.sector_fill_misses, 4);
        assert_eq!(result.l2_by_source.source(L2Source::L1Fill), (5, 0));
        assert_eq!(
            result.dram.read_sectors, 3,
            "the second mask fetches only its one new sector"
        );
        assert_eq!(result.per_sm[&0].cycles, 424);
        assert_eq!(result.per_sm[&7].cycles, 424);
        assert_eq!(
            l2.preview_mask(1, 0b0111, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            3
        );
    }

    #[test]
    fn later_pending_read_reuses_the_original_ready_time() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(load, "LDG.E")),
            (16, slot(consumer, "EXIT")),
            (64, slot(cb(7), "IADD3")),
            (80, slot(load, "LDG.E")),
            (96, slot(consumer, "EXIT")),
        ]);
        let delayed = vec![BlockReplay {
            launch_index: 1,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 64,
                        kind: AccessKind::NonMemory,
                    },
                    ReplayInst {
                        pc: 80,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 96,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors: vec![4],
            }],
        }];
        let per_sm = BTreeMap::from([(0, dependent_load_blocks(4)), (7, delayed)]);
        let mut l2 = l2_model();

        let result = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!(result.l2.misses, 2);
        assert_eq!(result.dram.read_sectors, 1);
        assert_eq!(result.per_sm[&0].cycles, 424);
        assert_eq!(
            result.per_sm[&7].cycles, 424,
            "the delayed lookup joins the t=423 original completion"
        );
        assert_eq!(result.terminal_event_time, Some(423));
    }

    #[test]
    fn write_supersession_keeps_the_original_read_response_and_removal() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(load, "LDG.E")),
            (16, slot(consumer, "EXIT")),
            (32, slot(cb(1), "STG.E")),
            (48, slot(cb(1), "EXIT")),
        ]);
        let storer = vec![BlockReplay {
            launch_index: 1,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    },
                    ReplayInst {
                        pc: 48,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors: vec![4],
            }],
        }];
        let per_sm = BTreeMap::from([(0, dependent_load_blocks(4)), (7, storer)]);
        let mut l2 = l2_model();

        let result = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!((result.l2.misses, result.l2.hits), (1, 1));
        assert_eq!(result.l2.write_tag_hits, 1);
        assert_eq!(result.l2_by_source.source(L2Source::L1Fill), (1, 0));
        assert_eq!(result.l2_by_source.source(L2Source::Store), (1, 1));
        assert_eq!(result.dram.read_sectors, 1);
        assert_eq!(result.per_sm[&0].cycles, 424);
        assert_eq!(
            l2.preview_mask(1, 0b0001, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            1,
            "the store-valid sector survives the original read completion"
        );
    }

    #[test]
    fn equal_time_fill_and_read_follow_event_sequence_order() {
        #[derive(Clone, Copy)]
        enum TieEvent {
            Fill(L2FillToken),
            Read,
        }

        for fill_first in [true, false] {
            let mut l2 = single_way_l2_model();
            let initial = l2
                .access_mask_event_transactional(0, 0b0001, L2OpClass::Read, |_| {
                    Ok::<(), sim_mem::L2AccessError>(())
                })
                .unwrap()
                .0;
            let fill = initial.fill.expect("cold event read mints a fill token");
            let ordered = if fill_first {
                [TieEvent::Fill(fill), TieEvent::Read]
            } else {
                [TieEvent::Read, TieEvent::Fill(fill)]
            };
            let mut queue = EventQueue::new();
            for event in ordered {
                queue.schedule_at(17, TargetId(0), event).unwrap();
            }

            let mut read_present = None;
            while let Some(event) = queue.pop().unwrap() {
                match event.kind {
                    TieEvent::Fill(token) => {
                        assert!(matches!(
                            l2.complete_fill(token).unwrap(),
                            L2FillCompletion::Completed { .. }
                        ));
                    }
                    TieEvent::Read => {
                        let access = l2
                            .access_mask_event_transactional(0, 0b0001, L2OpClass::Read, |_| {
                                Ok::<(), sim_mem::L2AccessError>(())
                            })
                            .unwrap()
                            .0;
                        read_present = Some(access.lookup.present_sectors);
                    }
                }
            }
            assert_eq!(read_present, Some(u32::from(fill_first)));
        }
    }

    #[test]
    fn equal_time_eviction_and_old_fill_follow_event_sequence_order() {
        #[derive(Clone, Copy)]
        enum TieEvent {
            Fill(L2FillToken),
            Evict,
        }

        for fill_first in [true, false] {
            let mut l2 = single_way_l2_model();
            let initial = l2
                .access_mask_event_transactional(0, 0b0001, L2OpClass::Read, |_| {
                    Ok::<(), sim_mem::L2AccessError>(())
                })
                .unwrap()
                .0;
            let old_fill = initial.fill.expect("cold event read mints a fill token");
            let ordered = if fill_first {
                [TieEvent::Fill(old_fill), TieEvent::Evict]
            } else {
                [TieEvent::Evict, TieEvent::Fill(old_fill)]
            };
            let mut queue = EventQueue::new();
            for event in ordered {
                queue.schedule_at(23, TargetId(0), event).unwrap();
            }

            let mut completion = None;
            while let Some(event) = queue.pop().unwrap() {
                match event.kind {
                    TieEvent::Fill(token) => {
                        completion = Some(l2.complete_fill(token).unwrap());
                    }
                    TieEvent::Evict => {
                        let access = l2
                            .access_mask_event_transactional(1, 0b0001, L2OpClass::Read, |_| {
                                Ok::<(), sim_mem::L2AccessError>(())
                            })
                            .unwrap()
                            .0;
                        assert!(access.fill.is_some());
                    }
                }
            }

            if fill_first {
                assert!(matches!(
                    completion,
                    Some(L2FillCompletion::Completed {
                        newly_valid_mask: 0b0001,
                        duplicate_mask: 0,
                    })
                ));
            } else {
                assert_eq!(completion, Some(L2FillCompletion::Stale));
            }
            assert_eq!(
                l2.preview_mask(0, 0b0001, L2OpClass::Read)
                    .unwrap()
                    .present_sectors,
                0,
                "same-cycle ordering must never resurrect the evicted line"
            );
        }
    }

    /// A fill may lose its allocation before data returns. Installation is then
    /// stale, but the data-ready event must still clear the originating SM's
    /// barrier; stale cache state is not permission to drop a response.
    #[test]
    fn stale_fill_does_not_suppress_its_sm_response() {
        let sched = load_consumer_schedule();
        let per_sm = BTreeMap::from([(0, dependent_load_blocks(4)), (7, dependent_load_blocks(8))]);
        let mut l2 = single_way_l2_model();
        let r = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!(r.l2.misses, 2);
        assert_eq!(r.l2.hits, 0);
        assert_eq!(r.per_sm[&0].instructions_issued, 2);
        assert_eq!(r.per_sm[&7].instructions_issued, 2);
        assert_eq!(r.per_sm[&0].cycles, 424);
        assert_eq!(r.per_sm[&7].cycles, 424);
        assert_eq!(
            l2.preview_mask(1, 1, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            0,
            "the evicted fill was not resurrected"
        );
        assert_eq!(
            l2.preview_mask(2, 1, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            1,
            "the surviving allocation did not complete"
        );
    }

    /// A cross-batch L1 merge emits a reply-less follow-up read. Its L2 token
    /// must survive independently of the original line-fill response and install
    /// at its own data-ready event, even after the SM has finished.
    #[test]
    fn replyless_followup_fetch_receives_a_fill_only_completion() {
        let mut load_m = cb(1);
        load_m.write_barrier = Some(barrier(2));
        let mut load_l = cb(1);
        load_l.write_barrier = Some(barrier(3));
        let mut load_merge = cb(1);
        load_merge.write_barrier = Some(barrier(2));
        load_merge.wait_mask = wait_mask(0b000100);
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b001100);
        let sched = schedule(vec![
            (0, slot(load_m, "LDG.E")),
            (16, slot(load_l, "LDG.E")),
            (32, slot(load_merge, "LDG.E")),
            (48, slot(consumer, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalEligible {
                            off: 1,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::GlobalEligible {
                            off: 2,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 48,
                        kind: AccessKind::NonMemory,
                    },
                ],
                // M: sector 80 (line 20). L: sector 40 (line 10).
                // The later merge adds sector 41 to line 10.
                sectors: vec![80, 40, 41],
            }],
        }];
        let mut l2 = l2_model();
        let r = simulate_kernel_shared(
            &BTreeMap::from([(0, blocks)]),
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!(r.per_sm[&0].instructions_issued, 4);
        assert_eq!(r.per_sm[&0].cycles, 458);
        assert_eq!(r.l2.misses, 3);
        assert_eq!(r.l2.line_misses, 2);
        assert_eq!(r.l2.sector_fill_misses, 1);
        assert_eq!(r.dram.read_sectors, 3);
        assert_eq!(r.terminal_event_time, Some(846));
        assert_eq!(
            l2.preview_mask(10, 0b0011, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            2,
            "reply-less sector did not install after the original response"
        );
    }

    /// Same-sector reuse WITHIN one SM is an L1 hit and never reaches
    /// the L2; a different-line load from another SM does.
    #[test]
    fn l1_filters_the_l2_stream() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(load, "LDG.E")),
            (16, slot(consumer, "FFMA")),
            (32, slot(load, "LDG.E")),
            (48, slot(consumer, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::NonMemory,
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::GlobalEligible {
                            off: 1,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 48,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors: vec![4, 4],
            }],
        }];
        let per_sm = BTreeMap::from([(3, blocks)]);
        let mut l2 = l2_model();
        let r = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();
        let l1 = r.per_sm[&3].l1.unwrap();
        assert_eq!((l1.hits, l1.misses), (1, 1), "second load is an L1 hit");
        assert_eq!(r.l2.total().unwrap(), 1, "only the L1 miss reached the L2");
    }

    /// Store sectors mutate the shared L2 in arrival order without a
    /// response: a later load from ANOTHER SM hits the stored line
    /// (write-allocate), and the queue drains cleanly.
    #[test]
    fn store_allocates_for_a_later_load_across_sms() {
        let mut store_cb = cb(1);
        store_cb.read_barrier = Some(barrier(1));
        let store_sched_pcs = vec![(0, slot(store_cb, "STG.E")), (16, slot(cb(1), "EXIT"))];
        let sched = schedule(
            store_sched_pcs
                .into_iter()
                .chain([
                    (32, {
                        let mut load = cb(1);
                        load.write_barrier = Some(barrier(0));
                        slot(load, "LDG.E")
                    }),
                    (48, {
                        let mut consumer = cb(1);
                        consumer.wait_mask = wait_mask(0b000001);
                        slot(consumer, "EXIT")
                    }),
                ])
                .collect(),
        );
        let storer = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors: vec![4],
            }],
        }];
        let loader = vec![BlockReplay {
            launch_index: 1,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 48,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors: vec![4],
            }],
        }];
        let per_sm = BTreeMap::from([(0, storer), (1, loader)]);
        let mut l2 = l2_model();
        let r = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();
        // SM 0's store (t=0, lower seq) allocates; SM 1's load fill hits.
        assert_eq!(r.l2_by_source.source(sim_mem::L2Source::Store), (1, 0));
        assert_eq!(r.l2_by_source.source(sim_mem::L2Source::L1Fill), (1, 1));
        assert_eq!(r.per_sm[&1].cycles, 245, "L2-hit arm for the loader");
        assert_eq!(
            *l2.counters().unwrap(),
            r.l2,
            "successful replay did not commit staged L2"
        );
        assert_eq!(
            l2.preview_mask(1, 0b0001, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            1,
            "committed caller L2 does not contain the replayed store"
        );
    }

    #[test]
    fn dirty_store_eviction_topology_is_timing_neutral_and_keeps_tails() {
        let sched = schedule(vec![
            (0, slot(cb(1), "STG.E")),
            (16, slot(cb(1), "STG.E")),
            (32, slot(cb(1), "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalStore { off: 1, len: 1 },
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors: vec![0, 4],
            }],
        }];
        let mut l2 = single_way_l2_model();
        let result = simulate_kernel_shared(
            &BTreeMap::from([(0, blocks)]),
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            DramBw::Unmodeled { arch: ArchId::Sm86 },
        )
        .unwrap();

        assert_eq!(result.per_sm[&0].cycles, 3);
        assert_eq!(result.l2_by_source.source(L2Source::Store), (2, 0));
        assert_eq!(result.dram.read_sectors, 0);
        assert_eq!(result.dram.l2_evicted_lines, 1);
        assert_eq!(result.dram.l2_evicted_valid_sectors, 1);
        assert_eq!(result.dram.l2_evicted_pending_lines, 0);
        assert_eq!(result.dram.l2_evicted_pending_sectors, 0);
        assert_eq!(result.dram.l2_dirty_eviction_lines, 1);
        assert_eq!(result.dram.l2_dirty_evicted_sectors, 1);
        assert_eq!(result.dram.maximum_terminal_l2_dirty_lines, 1);
        assert_eq!(result.dram.maximum_terminal_l2_dirty_sectors, 1);
        assert!(result.unmodeled_tails.dirty_writeback);
        assert!(result.unmodeled_tails.coherence);
    }

    #[test]
    fn packetized_multi_sector_write_observer_replaces_physical_lookup_count() {
        // The physical event model records one L2 lookup per sector, while
        // the observer rewrites ordinary (non-.128) stores into the carded
        // sectors-per-wavefront packet count. A legitimate packetized count
        // is therefore lower than the physical lookup counter and must still
        // complete without a cumulative-observer regression error.
        let sched = schedule(vec![(0, slot(cb(1), "STG.E")), (16, slot(cb(1), "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 5 },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::NonMemory,
                    },
                ],
                sectors: vec![0, 1, 2, 3, 4],
            }],
        }];
        let mut l2 = l2_model();
        let result = simulate_kernel_shared(
            &BTreeMap::from([(0, blocks)]),
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();
        assert_eq!(result.l2.lookup_requests, 3);
        assert_eq!(result.l2.lookup_hits, 3);
        assert_eq!(result.l2.total().unwrap(), 5);
    }

    /// §6.4 split between counting and pricing: an ATOM to a fresh
    /// sector of a RESIDENT line COUNTS as a hit (write-class tag-match
    /// rule) but its reply is PRICED on the DRAM arm — the atomic's
    /// return value needs the sector data the L2 does not hold yet.
    #[test]
    fn atomic_tag_hit_counts_hit_but_prices_dram() {
        let mut atom_cb = cb(1);
        atom_cb.write_barrier = Some(barrier(2));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000100);
        let sched = schedule(vec![
            (0, slot(cb(1), "STG.E")),
            (16, slot(atom_cb, "ATOMG.E.ADD.STRONG.GPU")),
            (32, slot(consumer, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalBypass {
                            off: 1,
                            len: 1,
                            token: CacheOpToken::Atom,
                            l2_requests: 1,
                        },
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::NonMemory,
                    },
                ],
                // Sectors 4 and 5 share line 1: the store allocates the
                // line (validating sector 0), the atomic touches the
                // still-invalid sector 1.
                sectors: vec![4, 5],
            }],
        }];
        let per_sm = BTreeMap::from([(0, blocks)]);
        let mut l2 = l2_model();
        let r = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();
        assert_eq!(
            r.l2_by_source.source(sim_mem::L2Source::Bypass),
            (1, 1),
            "the atomic's sector is COUNTED a hit (tag match)"
        );
        assert_eq!(r.l2.write_tag_hits, 1);
        assert_eq!(r.dram.read_sectors, 1, "ATOM RMW fetches the absent sector");
        assert_eq!(
            r.unmodeled_tails,
            TailFlags {
                red_rmw: false,
                dirty_writeback: true,
                coherence: true,
                ldgsts_dst_shared_timing: false,
            }
        );
        assert_eq!(
            r.per_sm[&0].cycles, 425,
            "…but the reply is PRICED on the DRAM arm (issue t=1 + 423 + 1)"
        );
    }

    #[test]
    fn atomic_fetch_is_intrinsic_when_no_barrier_observes_the_return() {
        let sched = schedule(vec![(0, slot(cb(1), "ATOMG.E.ADD.STRONG.GPU"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalBypass {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::Atom,
                        l2_requests: 1,
                    },
                }],
                sectors: vec![4],
            }],
        }];
        let mut l2 = l2_model();

        let result = simulate_kernel_shared(
            &BTreeMap::from([(0, blocks)]),
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!(result.l2_by_source.source(L2Source::Bypass), (1, 0));
        assert_eq!(result.dram.read_sectors, 1);
        assert_eq!(result.per_sm[&0].instructions_issued, 1);
        assert_eq!(result.per_sm[&0].cycles, 1);
        assert_eq!(
            result.terminal_event_time,
            Some(423),
            "the intrinsic ATOM return remains a modeled terminal event"
        );
        assert_eq!(
            result.unmodeled_tails,
            TailFlags {
                red_rmw: false,
                dirty_writeback: true,
                coherence: true,
                ldgsts_dst_shared_timing: false,
            }
        );
    }

    #[test]
    fn reduction_is_write_class_without_read_fetch_or_store_attribution() {
        let sched = schedule(vec![(0, slot(cb(1), "RED.E.ADD.STRONG.GPU"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalReduction {
                        off: 0,
                        len: 1,
                        l2_requests: 1,
                    },
                }],
                sectors: vec![4],
            }],
        }];
        let mut l2 = l2_model();

        let result = simulate_kernel_shared(
            &BTreeMap::from([(0, blocks)]),
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();

        assert_eq!(result.l2_by_source.source(L2Source::Store), (1, 0));
        assert_eq!(result.dram.read_sectors, 0);
        let l1 = result.per_sm[&0].l1.unwrap();
        assert_eq!(l1.store_sectors, 0);
        assert_eq!(l1.reduction_sectors, 1);
        assert_eq!(l1.reduction_hits, 0);
        assert_eq!(
            result.unmodeled_tails,
            TailFlags {
                red_rmw: true,
                dirty_writeback: true,
                coherence: true,
                ldgsts_dst_shared_timing: false,
            }
        );
        assert_eq!(
            l2.preview_mask(1, 0b0001, L2OpClass::Read)
                .unwrap()
                .present_sectors,
            1,
            "RED write data validates the sector without modeling its RMW read"
        );
    }

    /// §6.4 slice 2 — worst-of-mask pricing: a 2-sector line fill whose
    /// first sector is L2-resident (a prior store validated it) but
    /// whose merged sector is absent COUNTS one hit + one miss and
    /// PRICES the response on the DRAM arm (the fill completes when
    /// its slowest sector arrives).
    #[test]
    fn multi_sector_fill_prices_on_worst_sector() {
        let mut load = cb(1);
        load.write_barrier = Some(barrier(0));
        let mut consumer = cb(1);
        consumer.wait_mask = wait_mask(0b000001);
        let sched = schedule(vec![
            (0, slot(cb(1), "STG.E")),
            (16, slot(load, "LDG.E")),
            (32, slot(consumer, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    },
                    ReplayInst {
                        pc: 16,
                        kind: AccessKind::GlobalEligible {
                            off: 1,
                            len: 2,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        pc: 32,
                        kind: AccessKind::NonMemory,
                    },
                ],
                // Store validates line 1 sector 0; the load's fill
                // carries sectors 0 (present) + 1 (absent).
                sectors: vec![4, 4, 5],
            }],
        }];
        let per_sm = BTreeMap::from([(0, blocks)]);
        let mut l2 = l2_model();
        let r = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();
        assert_eq!(
            r.l2_by_source.source(sim_mem::L2Source::L1Fill),
            (2, 1),
            "the mask walk counts BOTH sectors (one hits the stored data)"
        );
        assert_eq!(
            r.per_sm[&0].cycles, 425,
            "priced on the ABSENT sector's DRAM arm (issue t=1 + 423 + 1), not the resident one"
        );
    }

    /// The DRAM byte server engages only past the carded rate: with a
    /// TINY service rate (1 byte per 10 cycles), two cold single-sector
    /// fills from two SMs serialize on the 32 B fetches — the first
    /// response lands past its own service time and the second waits
    /// for the first's bytes; with the CARDED rate the same run prices
    /// pure latency and bw_delayed_responses is zero.
    #[test]
    fn dram_server_delays_past_the_carded_rate() {
        let build = || {
            let mut load = cb(1);
            load.write_barrier = Some(barrier(0));
            let mut consumer = cb(1);
            consumer.wait_mask = wait_mask(0b000001);
            let sched = schedule(vec![(0, slot(load, "LDG.E")), (16, slot(consumer, "EXIT"))]);
            // Distinct lines per SM: two independent cold 32 B fetches.
            let mk = |sector: i64| dependent_load_blocks(sector);
            (sched, BTreeMap::from([(0, mk(4)), (1, mk(64))]))
        };

        // Carded rate: server active, engages nothing.
        let (sched, per_sm) = build();
        let mut l2 = l2_model();
        let r = simulate_kernel_shared(
            &per_sm,
            &sched,
            &params(),
            l1_config(),
            &mut l2,
            carded_bw(),
        )
        .unwrap();
        assert_eq!(r.dram.read_sectors, 2);
        assert_eq!(r.dram.bw_delayed_responses, 0);
        let base0 = r.per_sm[&0].cycles;
        let base1 = r.per_sm[&1].cycles;

        // 1 byte per 10 cycles: 32 B take 320 cycles, so the FIRST
        // fetch (completes at 320) stays under the 423 latency arm
        // while the SECOND, queued behind it, completes at 640 and
        // presses past — exactly one bandwidth-delayed response.
        let (sched, per_sm) = build();
        let mut l2 = l2_model();
        let slow = DramBw::Carded {
            arch: ArchId::Sm86,
            bytes_per_sec: 1,
            sm_clock_hz: 10,
        };
        let r2 =
            simulate_kernel_shared(&per_sm, &sched, &params(), l1_config(), &mut l2, slow).unwrap();
        assert_eq!(r2.dram.read_sectors, 2);
        assert_eq!(r2.dram.bw_delayed_responses, 1, "second fetch queues");
        assert_eq!(r2.terminal_event_time, Some(640));
        assert_eq!(r2.per_sm[&0].cycles, base0, "first fetch: 320 < 423 arm");
        assert!(
            r2.per_sm[&1].cycles > base1,
            "second fetch waits for the first's bytes (640 > 423)"
        );
        assert_eq!(
            r2.per_sm[&1].cycles - base1,
            640 - 423,
            "delayed exactly to the server completion"
        );
    }

    /// The livelock backstop: unbounded idle wakes at a frozen clock are
    /// a typed error, never an infinite spin. (All the OTHER tests in
    /// this module are the false-positive check: real multi-SM runs with
    /// wake bounces never trip the guard.)
    #[test]
    fn spin_guard_trips_on_frozen_clock_idle_wakes() {
        let mut g = SpinGuard::new(2).unwrap(); // limit = 12
        for i in 0..12 {
            assert!(g.on_event(42, true).is_ok(), "wake {i} within bound");
        }
        match g.on_event(42, true) {
            Err(SystemError::Livelock { time, idle_wakes }) => {
                assert_eq!(time, 42);
                assert_eq!(idle_wakes, 13);
            }
            other => panic!("expected Livelock, got {other:?}"),
        }
    }

    #[test]
    fn spin_guard_resets_on_clock_advance_and_real_events() {
        let mut g = SpinGuard::new(0).unwrap(); // limit = 8: tightest bound
        // A clock advance resets the streak...
        for t in 0..100u64 {
            assert!(g.on_event(t, true).is_ok());
        }
        // ...and so does any non-idle event at the same timestamp.
        for _ in 0..100 {
            for _ in 0..8 {
                assert!(g.on_event(7, true).is_ok());
            }
            assert!(g.on_event(7, false).is_ok());
        }
    }

    /// Determinism: two identical runs produce identical counters,
    /// cycle maps, DRAM stats, and source splits — under BOTH the
    /// carded rate (server active, idle) and a tiny rate that ENGAGES
    /// the server (gate finding: the engaged-server path had no
    /// determinism coverage).
    #[test]
    fn identical_runs_are_bit_identical() {
        let sched = load_consumer_schedule();
        for bw in [
            carded_bw(),
            DramBw::Carded {
                arch: ArchId::Sm86,
                bytes_per_sec: 1,
                sm_clock_hz: 10,
            },
        ] {
            let run = || {
                let per_sm = BTreeMap::from([
                    (0, dependent_load_blocks(4)),
                    (5, dependent_load_blocks(8)),
                    (9, dependent_load_blocks(4)),
                ]);
                let mut l2 = l2_model();
                simulate_kernel_shared(&per_sm, &sched, &params(), l1_config(), &mut l2, bw)
                    .unwrap()
            };
            let a = run();
            let b = run();
            assert_eq!(a.l2, b.l2);
            assert_eq!(a.dram, b.dram);
            assert_eq!(a.l2_by_source, b.l2_by_source);
            assert_eq!(a.terminal_event_time, b.terminal_event_time);
            assert_eq!(
                a.per_sm.values().map(|r| r.cycles).collect::<Vec<_>>(),
                b.per_sm.values().map(|r| r.cycles).collect::<Vec<_>>()
            );
        }
    }

    /// Zero rates are a typed configuration error, never a clamp.
    #[test]
    fn source_counter_overflow_is_transactional() {
        let mut dst = L2SourceSplit::for_arch(ArchId::Sm86);
        dst.sectors[0] = u64::MAX;
        dst.hits[1] = 9;
        let mut src = L2SourceSplit::for_arch(ArchId::Sm86);
        src.sectors[0] = 1;
        src.hits[1] = 4;
        let before = dst;
        assert!(matches!(
            dst.add_checked(&src),
            Err(SystemCounterError::Overflow {
                counter: "L2 source sectors",
            })
        ));
        assert_eq!(dst, before);
    }

    #[test]
    fn source_and_dram_aggregation_reject_incoherent_partitions_transactionally() {
        let mut split = L2SourceSplit::for_arch(ArchId::Sm86);
        split.sectors[L2Source::L1Fill as usize] = 1;
        let split_addend = L2SourceSplit::for_arch(ArchId::Sm86);
        let split_before = split;
        assert_eq!(
            split.add_checked(&split_addend),
            Err(SystemCounterError::L2OriginPartitionMismatch {
                counter: "sectors",
                l2_source: L2Source::L1Fill,
                unified: 1,
                global: 0,
                local: 0,
            })
        );
        assert_eq!(split, split_before);

        let mut dram = DramStats::for_arch(ArchId::Sm86);
        dram.read_sectors = 1;
        let dram_addend = DramStats::for_arch(ArchId::Sm86);
        let dram_before = dram;
        assert_eq!(
            dram.add_checked(&dram_addend),
            Err(SystemCounterError::DramOriginPartitionMismatch {
                counter: "read sectors",
                unified: 1,
                global: 0,
                local: 0,
            })
        );
        assert_eq!(dram, dram_before);
    }

    #[test]
    fn source_aggregation_rejects_hits_exceeding_source_sectors() {
        let result = L2SourceSplit::from_origins(
            ArchId::Sm86,
            L2OriginSourceStats {
                sectors: [1, 0, 0, 0],
                hits: [2, 0, 0, 0],
            },
            L2OriginSourceStats::default(),
        );
        assert_eq!(
            result,
            Err(SystemCounterError::L2OriginHitExceedsSectors {
                origin: MemoryOrigin::Global,
                l2_source: L2Source::L1Fill,
                hits: 2,
                sectors: 1,
            })
        );

        let mut forged = L2SourceSplit::for_arch(ArchId::Sm86);
        forged.sectors[L2Source::Store as usize] = 1;
        forged.hits[L2Source::Store as usize] = 2;
        forged.origin_sectors[MemoryOrigin::Global.index()][L2Source::Store as usize] = 1;
        forged.origin_hits[MemoryOrigin::Global.index()][L2Source::Store as usize] = 2;
        assert_eq!(
            forged.validate_origin_partition(),
            Err(SystemCounterError::L2OriginHitExceedsSectors {
                origin: MemoryOrigin::Global,
                l2_source: L2Source::Store,
                hits: 2,
                sectors: 1,
            })
        );
    }

    #[test]
    fn source_and_dram_aggregation_preserve_origin_partitions() {
        let mut source_total = L2SourceSplit::from_origins(
            ArchId::Sm86,
            L2OriginSourceStats {
                sectors: [2, 0, 0, 0],
                hits: [1, 0, 0, 0],
            },
            L2OriginSourceStats {
                sectors: [1, 0, 0, 0],
                hits: [1, 0, 0, 0],
            },
        )
        .unwrap();
        let source_addend = L2SourceSplit::from_origins(
            ArchId::Sm86,
            L2OriginSourceStats {
                sectors: [4, 0, 0, 0],
                hits: [2, 0, 0, 0],
            },
            L2OriginSourceStats {
                sectors: [1, 0, 0, 0],
                hits: [1, 0, 0, 0],
            },
        )
        .unwrap();

        source_total.add_checked(&source_addend).unwrap();
        assert_eq!(source_total.source(L2Source::L1Fill), (8, 5));
        assert_eq!(
            source_total.source_for_origin(MemoryOrigin::Global, L2Source::L1Fill),
            (6, 3)
        );
        assert_eq!(
            source_total.source_for_origin(MemoryOrigin::Local, L2Source::L1Fill),
            (2, 2)
        );

        let mut dram_total = DramStats::from_origins(
            ArchId::Sm86,
            DramOriginStats {
                read_sectors: 2,
                bw_delayed_responses: 1,
            },
            DramOriginStats {
                read_sectors: 1,
                bw_delayed_responses: 0,
            },
        )
        .unwrap();
        let dram_addend = DramStats::from_origins(
            ArchId::Sm86,
            DramOriginStats {
                read_sectors: 3,
                bw_delayed_responses: 1,
            },
            DramOriginStats {
                read_sectors: 2,
                bw_delayed_responses: 1,
            },
        )
        .unwrap();

        dram_total.add_checked(&dram_addend).unwrap();
        assert_eq!(dram_total.read_sectors, 8);
        assert_eq!(dram_total.bw_delayed_responses, 3);
        assert_eq!(
            dram_total.origin(MemoryOrigin::Global),
            DramOriginStats {
                read_sectors: 5,
                bw_delayed_responses: 2,
            }
        );
        assert_eq!(
            dram_total.origin(MemoryOrigin::Local),
            DramOriginStats {
                read_sectors: 3,
                bw_delayed_responses: 1,
            }
        );
    }

    #[test]
    fn dram_counter_aggregation_rejects_mixed_architecture() {
        let mut sm86 = DramStats::for_arch(ArchId::Sm86);
        let sm120 = DramStats::for_arch(ArchId::Sm120);
        assert!(matches!(
            sm86.add_checked(&sm120),
            Err(SystemCounterError::Architecture(
                sim_arch::ArchitectureProvenanceError::Mismatch {
                    component: "DRAM stats",
                    expected: ArchId::Sm86,
                    got: ArchId::Sm120,
                }
            ))
        ));
    }
}
