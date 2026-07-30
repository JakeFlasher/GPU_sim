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
//!   with trace order. The lookup outcome prices the response on the
//!   measured latency ladder — all sectors present →
//!   `issue_time + l1_miss_latency` (the card's L1-miss/L2-hit point),
//!   any sector absent → `issue_time + global_flat_latency` (the
//!   card's DRAM point) — then the carded aggregate DRAM byte server
//!   ([`DramBw`]) may push the ready time PAST the latency arm when
//!   the hierarchy read-miss stream exceeds the locked-operating-point
//!   rate (it can only ever ADD delay).
//! - `L2Fill` — installs a reply-less read fetch at its data-ready time.
//! - `AtomicCompletion` — retains the intrinsic return-completion time of an
//!   ATOM whose value has no dependence-barrier observer.
//! - `MemResponse` — first completes any fill carried by the same request,
//!   then completes the originating SM's L1 fill (or clears a direct
//!   bypass/untagged barrier unit) at exactly the data-ready time and resumes
//!   that SM. Stale L2 installation never suppresses the SM response.
//!
//! Store and reduction mutation-only requests take no response event: they
//! change L2 state in arrival order and nothing waits on them (v0).
//!
//! Determinism: SM actors are created in ascending traced `sm_id` order,
//! all scheduling flows through the `(time, seq)` queue, and L2 lookup and
//! fill mutation occur only in their typed event handlers — two runs over the
//! same inputs produce bit-identical schedules and counters.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

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

use sim_arch::ArchId;
use sim_engine::{EngineError, EventQueue, TargetId};
use sim_mem::{L1Config, L2Counters, L2FillToken, L2Model, L2Source, MemoryOrigin};

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

            let global_hits = self.origin_hits[MemoryOrigin::Global.index()][source_ix];
            let local_hits = self.origin_hits[MemoryOrigin::Local.index()][source_ix];
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
    Advance, BlockReplay, IssueError, IssueModelParams, MemRequest, MemoryOperation, ReplayHorizon,
    ReplyTo, SmCore, SmMemMode, SmSimResult, TailFlags, checked_cycle_add,
};
use crate::program::FunctionSchedule;

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

#[derive(Debug, Clone, Copy)]
enum Ev {
    SmWake,
    L2Access(MemRequest),
    L2Fill(L2FillToken),
    AtomicCompletion,
    MemResponse {
        reply: ReplyTo,
        fill: Option<L2FillToken>,
    },
}

struct L2ResponseCommit {
    ready: u64,
    origin_ix: usize,
    next_delayed: u64,
    next_origin_delayed: u64,
    unpriced_removal: UnpricedRemoval,
}

struct L2SystemCommit {
    source_ix: usize,
    source_sectors: u64,
    source_hits: u64,
    origin_source_sectors: [u64; 2],
    origin_source_hits: [u64; 2],
    read_sectors: u64,
    origin_read_sectors: [u64; 2],
    dram_free_fp: Option<u128>,
    fill_ready: Option<u64>,
    atomic_completion_ready: Option<u64>,
    response: Option<L2ResponseCommit>,
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

/// Driver-side DRAM traffic observer.
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
    by_origin: [DramOriginStats; 2],
}

impl Default for DramStats {
    fn default() -> Self {
        Self {
            arch: None,
            read_sectors: 0,
            bw_delayed_responses: 0,
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
pub(crate) fn simulate_kernel_shared(
    per_sm_blocks: &BTreeMap<i32, Vec<BlockReplay>>,
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    l1_config: L1Config,
    l2: &mut L2Model,
    dram: DramBw,
) -> Result<SystemResult, SystemError> {
    let arch = schedule.arch;
    require_arch(arch, params.arch, "issue parameters")?;
    require_arch(arch, l1_config.arch, "L1 configuration")?;
    require_arch(arch, l2.config().arch, "L2 configuration")?;
    checked_l2_counters(arch, l2)?;
    require_arch(arch, dram.arch(), "DRAM configuration")?;
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
    let sm_ids: Vec<i32> = per_sm_blocks.keys().copied().collect();
    let mut cores: Vec<SmCore<'_>> = per_sm_blocks
        .values()
        .map(|blocks| SmCore::new(blocks, schedule, params, SmMemMode::L1Shared(l1_config)))
        .collect::<Result<_, _>>()?;

    let mut queue: EventQueue<Ev> = EventQueue::new();
    let mut scratch: Vec<MemRequest> = Vec::new();
    // Scheduled-but-unapplied response times PER SM: the only external
    // events that can affect an SM's own state, hence its advance
    // horizon. Other SMs' events are deliberately NOT barriers — cross-SM
    // effects flow only through the L2 access stream, which the queue
    // already orders by (time, seq). (Treating every queued event as a
    // barrier would also livelock: two SMs parked at the same timestamp
    // would bounce off each other's wake events forever.)
    let mut response_due = DueTimes::new(cores.len());
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
    // The caller owns `l2` across this boundary. Complete all replay admission
    // and event-system setup first so their established typed errors keep
    // precedence over snapshot allocation failure. Then replay the entire
    // kernel on a private snapshot and publish it only after event draining,
    // deadlock checks, and result construction all succeed.
    let mut staged_l2 = l2.try_snapshot()?;
    while let Some(ev) = queue.pop()? {
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
                    &mut wake_due,
                    &mut unpriced,
                    &mut scratch,
                    params.l1_miss_latency,
                )?;
                spin.on_event(ev.time, !progress)?;
            }
            Ev::L2Access(req) => {
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
                #[cfg(test)]
                maybe_inject_l2_access_error()?;

                // Prepare the exact ascending mask walk against a private copy
                // of its set. The callback completes every fallible timing and
                // counter calculation before the L2 plan commits; only then can
                // the API issue a fill token. Event insertion follows
                // immediately. If insertion fails, the kernel-private L2
                // snapshot is discarded and the caller-owned model is unchanged.
                let line = req.sector_id >> 2;
                let (l2_access, commit) = staged_l2.access_mask_event_transactional(
                    line,
                    req.sector_mask,
                    req.operation.l2_class(),
                    |preview| -> Result<L2SystemCommit, SystemError> {
                        let lookup = preview.lookup;
                        let requested_sectors = u64::from(lookup.sectors);
                        let counted_hits = u64::from(lookup.counted_hits);
                        let source_ix = req.source as usize;
                        let origin_masks = [
                            req.sector_mask & !req.local_sector_mask,
                            req.local_sector_mask,
                        ];
                        let origin_requested =
                            origin_masks.map(|mask| u64::from(mask.count_ones()));
                        let origin_hits = origin_masks
                            .map(|mask| u64::from((mask & lookup.counted_hit_mask).count_ones()));
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

                        let fetch_mask = if req.operation.fetches_missing_data() {
                            req.sector_mask & !lookup.present_mask
                        } else {
                            0
                        };
                        let fetch_sectors = u64::from(fetch_mask.count_ones());
                        let origin_fetches =
                            origin_masks.map(|mask| u64::from((mask & fetch_mask).count_ones()));
                        let next_read_sectors = dram_stats
                            .read_sectors
                            .checked_add(fetch_sectors)
                            .ok_or(SystemError::CounterOverflow {
                                counter: "DRAM read sectors",
                            })?;
                        let mut next_origin_read_sectors = [0u64; 2];
                        for origin in MemoryOrigin::ALL {
                            let origin_ix = origin.index();
                            next_origin_read_sectors[origin_ix] = dram_stats.by_origin[origin_ix]
                                .read_sectors
                                .checked_add(origin_fetches[origin_ix])
                                .ok_or(SystemError::CounterOverflow {
                                    counter: "DRAM origin read sectors",
                                })?;
                        }
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

                        let response = if req.reply != ReplyTo::None {
                            let response_origin_ix = req.origin.index();
                            let bound = checked_cycle_add(
                                "unpriced L2-hit response bound",
                                req.issue_time,
                                params.l1_miss_latency,
                            )?;
                            let unpriced_removal = unpriced.prepare_remove(sm_ix, bound)?;
                            let latency_ready = if lookup.all_present() {
                                bound
                            } else {
                                checked_cycle_add(
                                    "DRAM response visibility",
                                    req.issue_time,
                                    params.global_flat_latency,
                                )?
                            };
                            let ready = latency_ready.max(bw_ready);
                            if ready < bound {
                                return Err(SystemError::ResponseBeforeBound {
                                    sm: sm_ix,
                                    ready,
                                    bound,
                                });
                            }
                            let delayed = u64::from(ready > latency_ready);
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
                                ready,
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
                                let latency_ready = if lookup.all_present() {
                                    checked_cycle_add(
                                        "unobserved ATOM L2-hit completion",
                                        req.issue_time,
                                        params.l1_miss_latency,
                                    )?
                                } else {
                                    checked_cycle_add(
                                        "unobserved ATOM DRAM completion",
                                        req.issue_time,
                                        params.global_flat_latency,
                                    )?
                                };
                                Some(latency_ready.max(bw_ready))
                            } else {
                                None
                            };

                        let fill_ready = match (preview.fill_mask(), response.as_ref()) {
                            (0, _) => None,
                            (_, Some(response)) => Some(response.ready),
                            (_, None) => Some(
                                checked_cycle_add(
                                    "DRAM fill visibility",
                                    req.issue_time,
                                    params.global_flat_latency,
                                )?
                                .max(bw_ready),
                            ),
                        };

                        Ok(L2SystemCommit {
                            source_ix,
                            source_sectors: next_source_sectors,
                            source_hits: next_source_hits,
                            origin_source_sectors: next_origin_source_sectors,
                            origin_source_hits: next_origin_source_hits,
                            read_sectors: next_read_sectors,
                            origin_read_sectors: next_origin_read_sectors,
                            dram_free_fp: service_preview.map(|(done, _)| done),
                            fill_ready,
                            atomic_completion_ready,
                            response,
                        })
                    },
                )?;

                let has_token = l2_access.fill.is_some();
                let has_ready = commit.fill_ready.is_some();
                if has_token != has_ready {
                    return Err(SystemError::FillScheduleMismatch {
                        sector_id: req.sector_id,
                        has_token,
                        has_ready,
                    });
                }
                // Reply-bearing fills travel in one event so completion is
                // ordered before delivery without relying on two equal-time
                // sequence numbers. Reply-less read fetches still receive a
                // fill-only event. An ATOM whose return is unobserved retains a
                // timing-only completion event; Store/RED need no completion.
                if let Some(response) = commit.response.as_ref() {
                    queue.schedule_at(
                        response.ready,
                        ev.target,
                        Ev::MemResponse {
                            reply: req.reply,
                            fill: l2_access.fill,
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
                spin.on_progress_event(ev.time);
            }
            Ev::L2Fill(fill) => {
                let _completion = staged_l2.complete_fill(fill)?;
                spin.on_progress_event(ev.time);
            }
            Ev::AtomicCompletion => {
                spin.on_progress_event(ev.time);
            }
            Ev::MemResponse { reply, fill } => {
                if let Some(fill) = fill {
                    // `Stale` is an expected no-op after eviction; delivery of
                    // data to the requester remains valid and must continue.
                    let _completion = staged_l2.complete_fill(fill)?;
                }
                cores[sm_ix].preflight_response(reply, ev.time)?;
                response_due.pop_expect("memory response", sm_ix, ev.time)?;
                scratch.clear();
                cores[sm_ix].on_response(reply, ev.time, &mut scratch)?;
                // Retries the completion unblocked become L2 accesses at
                // the response time.
                for req in scratch.drain(..) {
                    if req.reply != ReplyTo::None {
                        let bound = checked_cycle_add(
                            "retry unpriced L2-hit response bound",
                            req.issue_time,
                            params.l1_miss_latency,
                        )?;
                        unpriced.insert(sm_ix, bound)?;
                    }
                    queue.schedule_at(req.issue_time, ev.target, Ev::L2Access(req))?;
                }
                run_sm(
                    sm_ix,
                    &mut cores,
                    &mut queue,
                    &mut response_due,
                    &mut wake_due,
                    &mut unpriced,
                    &mut scratch,
                    params.l1_miss_latency,
                )?;
                spin.on_event(ev.time, false)?;
            }
        }
        terminal_event_time = Some(ev.time);
    }

    let mut per_sm = BTreeMap::new();
    let mut unmodeled_tails = TailFlags::default();
    for (ix, core) in cores.iter().enumerate() {
        if !core.is_done() {
            return Err(SystemError::Deadlock {
                sm_id: sm_ids[ix],
                outstanding: core.outstanding_responses(),
            });
        }
        let sm_result = core.result();
        unmodeled_tails = unmodeled_tails.union(sm_result.unmodeled_tails);
        per_sm.insert(sm_ids[ix], sm_result);
    }
    let l2_counters = *checked_l2_counters(arch, &staged_l2)?;
    let result = SystemResult {
        arch,
        per_sm,
        l2: l2_counters,
        l2_by_source,
        dram: dram_stats,
        unmodeled_tails,
        terminal_event_time,
    };
    *l2 = staged_l2;
    Ok(result)
}

fn replay_horizon(response: Option<u64>, unpriced: Option<u64>) -> ReplayHorizon {
    match (response, unpriced) {
        (Some(left), Some(right)) => ReplayHorizon::Bounded(left.min(right)),
        (Some(bound), None) | (None, Some(bound)) => ReplayHorizon::Bounded(bound),
        (None, None) => ReplayHorizon::Unbounded,
    }
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
    wake_due: &mut DueTimes,
    unpriced: &mut UnpricedBounds,
    scratch: &mut Vec<MemRequest>,
    l2_hit_latency: u64,
) -> Result<bool, SystemError> {
    scratch.clear();
    let horizon = replay_horizon(response_due.next(ix), unpriced.min(ix));
    let issued_before = cores[ix].instructions_issued();
    let adv = cores[ix].advance(horizon, scratch)?;
    let progress = cores[ix].instructions_issued() > issued_before || !scratch.is_empty();
    let target =
        u32::try_from(ix).map_err(|_| SystemError::ActorCountOutOfRange { count: cores.len() })?;
    for req in scratch.drain(..) {
        if req.reply != ReplyTo::None {
            let bound = checked_cycle_add(
                "unpriced L2-hit response bound",
                req.issue_time,
                l2_hit_latency,
            )?;
            unpriced.insert(ix, bound)?;
        }
        queue.schedule_at(req.issue_time, TargetId(target), Ev::L2Access(req))?;
    }
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

    use sim_mem::{CacheOpToken, L2Config, L2CounterError, L2OpClass, L2Replacement, SliceHash};

    use super::*;
    use crate::control_bits::{ControlBits, DepBarrier, WaitMask};
    use crate::issue::{AccessKind, ReplayInst, WarpReplay};
    use crate::program::{IssueSlot, OpClass};

    fn cb(stall: u8) -> ControlBits {
        ControlBits::new(stall, false, 7, 7, 0, 0).expect("valid test controls")
    }

    fn barrier(value: u8) -> DepBarrier {
        DepBarrier::new(value).expect("valid test barrier")
    }

    fn wait_mask(bits: u8) -> WaitMask {
        WaitMask::new(bits).expect("valid test wait mask")
    }

    fn slot(control: ControlBits, text: &str) -> IssueSlot {
        IssueSlot {
            control,
            op_class: OpClass::Other,
            mem_role: sim_mem::MemRole::None,
            opcode_text: text.to_owned(),
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
            l1_hit_latency: 34,
            l1_miss_latency: 244,
            global_flat_latency: 423,
            shared_write_barrier_latency: 23,
            non_memory_write_barrier_latency: 25,
            read_barrier_latency: 8,
            max_resident_warps_per_sm: 48,
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
        vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
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
                ],
                sectors: vec![sector],
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
                    warp_slot: 0,
                    insts: vec![ReplayInst {
                        pc: 0,
                        kind: AccessKind::LocalStore { off: 0, len: 1 },
                    }],
                    sectors: vec![40],
                },
                WarpReplay {
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
    /// than valid. FIFO order remains deterministic, but no implicit L2 MSHR or
    /// request coalescing is invented: each read counts a miss, fetches, and
    /// carries its own generation-bound completion.
    #[test]
    fn same_cycle_pending_reads_remain_distinct_misses() {
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
        // Both L2 accesses are ordered before either data-ready completion at
        // t=423. The first completion validates the sector and the second is an
        // explicit duplicate completion; both SM responses still arrive.
        assert_eq!(r.l2.misses, 2);
        assert_eq!(r.l2.hits, 0);
        assert_eq!(r.dram.read_sectors, 2);
        assert_eq!(r.per_sm[&0].cycles, 424);
        assert_eq!(r.per_sm[&7].cycles, 424);
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
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalBypass {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::Atom,
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
            }
        );
    }

    #[test]
    fn reduction_is_write_class_without_read_fetch_or_store_attribution() {
        let sched = schedule(vec![(0, slot(cb(1), "RED.E.ADD.STRONG.GPU"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                warp_slot: 0,
                insts: vec![ReplayInst {
                    pc: 0,
                    kind: AccessKind::GlobalReduction { off: 0, len: 1 },
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
