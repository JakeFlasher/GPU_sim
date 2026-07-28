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
//!   the global read-miss stream exceeds the locked-operating-point
//!   rate (it can only ever ADD delay).
//! - `MemResponse` — completes the originating SM's L1 fill (or clears a
//!   direct bypass/untagged barrier unit) at exactly the data-ready
//!   time, then resumes that SM.
//!
//! Store and other mutation-only requests take no response event: they
//! change L2 state in arrival order and nothing waits on them (v0).
//!
//! Determinism: SM actors are created in ascending traced `sm_id` order,
//! all scheduling flows through the `(time, seq)` queue, and the L2 is
//! touched only from `L2Access` handlers — two runs over the same inputs
//! produce bit-identical schedules and counters.

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

    fn remove(&mut self, sm: usize, bound: u64) -> Result<(), SystemError> {
        match self.0[sm].get(&bound).copied() {
            Some(n) if n > 1 => {
                self.0[sm].insert(bound, n - 1);
            }
            Some(_) => {
                self.0[sm].remove(&bound);
            }
            None => return Err(SystemError::UnpricedBoundMissing { sm, bound }),
        }
        Ok(())
    }

    fn require(&self, sm: usize, bound: u64) -> Result<(), SystemError> {
        if self.0[sm].contains_key(&bound) {
            Ok(())
        } else {
            Err(SystemError::UnpricedBoundMissing { sm, bound })
        }
    }

    fn min(&self, sm: usize) -> Option<u64> {
        self.0[sm].first_key_value().map(|(&t, _)| t)
    }
}

use sim_arch::ArchId;
use sim_engine::{EngineError, EventQueue, TargetId};
use sim_mem::{L1Config, L2Counters, L2Model, L2Source};

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

/// Per-[`L2Source`] sector/hit split — the driver-side EMISSION observer
/// (the cache model itself no longer counts provenance; gate-3 altitude
/// note c). Indexed by discriminant order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2SourceSplit {
    arch: Option<ArchId>,
    pub sectors: [u64; 4],
    pub hits: [u64; 4],
}

impl Default for L2SourceSplit {
    fn default() -> Self {
        Self {
            arch: None,
            sectors: [0; 4],
            hits: [0; 4],
        }
    }
}

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
        self.sectors = sectors;
        self.hits = hits;
        Ok(())
    }
}

use crate::issue::{
    Advance, BlockReplay, IssueError, IssueModelParams, MemRequest, ReplyTo, SmCore, SmMemMode,
    SmSimResult, checked_cycle_add,
};
use crate::program::FunctionSchedule;

#[derive(Debug, thiserror::Error)]
pub enum SystemCounterError {
    #[error(transparent)]
    Architecture(#[from] sim_arch::ArchitectureProvenanceError),
    #[error("{counter} counter overflows u64")]
    Overflow { counter: &'static str },
}

#[derive(Debug, thiserror::Error)]
pub enum SystemError {
    #[error(transparent)]
    Counters(#[from] SystemCounterError),
    #[error(transparent)]
    Issue(#[from] IssueError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(
        "event queue drained with sm {sm_id} unfinished ({outstanding} response(s) recorded \
         outstanding) — request/response wiring invariant broken"
    )]
    Deadlock { sm_id: i32, outstanding: u64 },
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
        "DRAM completion {completion_fp}/{bytes_per_sec} reaches or exceeds u64::MAX, reserved as the unbounded horizon"
    )]
    DramCycleDomainExhausted {
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
    MemResponse(ReplyTo),
}

/// Aggregate DRAM read-bandwidth model (the §6.3 "DRAM
/// bandwidth-latency" increment). The carded covariate is the vendor
/// peak at the LOCKED operating point (`ArchCard.dram_bandwidth` —
/// bytes/second over the locked SM clock, ~747 B/SM-cycle on the
/// 3090 Ti); the model is one G/D/1 byte server over the GLOBAL
/// read-miss sector stream: each fetching sector demands 32 B, and a
/// reply-carrying request cannot complete before BOTH its latency arm
/// and the server's completion of ITS OWN request's bytes. Bytes of
/// reply-less follow-up fetches (late merges) are served but price no
/// response — the merged waiter still clears on the original fill's
/// response, so under congestion a follow-up's bytes may complete
/// after its waiter cleared (the documented merge-timing
/// approximation; the cost surfaces on later requests). Uncongested
/// streams see pure latency (the server only ever ADDS delay).
///
/// v0 scope: DRAM READS only — a write-allocate store validates its
/// sector from the write data (no fetch), and write-back / RED-ATOM
/// read-modify-write DRAM traffic is unmodeled (the atomic stream is
/// absent from M1 traces; see the §6.4 capture-gap note).
#[derive(Debug, Clone, Copy)]
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
        let ready = done.div_ceil(self.bytes_per_sec);
        if ready >= u128::from(u64::MAX) {
            return Err(SystemError::DramCycleDomainExhausted {
                completion_fp: done,
                bytes_per_sec: self.bytes_per_sec,
            });
        }
        Ok((done, ready as u64))
    }
}

/// Driver-side DRAM traffic observer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DramStats {
    /// Architecture provenance bound by the shared-L2 system. Unbound
    /// synthetic stats are rejected before aggregation or emission.
    arch: Option<ArchId>,
    /// 32 B DRAM read fetches DEMANDED by the run (L2-missing sectors
    /// of read class, plus reply-carrying write-class misses — the
    /// ATOM RMW arm). Counted in BOTH modes: under `Unmodeled` it
    /// audits the ablation's demand with no server attached.
    pub read_sectors: u64,
    /// Responses whose ready time the bandwidth server pushed PAST the
    /// latency arm (0 in uncongested runs and in `Unmodeled` mode).
    /// Attribution caveat: reply-less follow-up fetches add bytes but
    /// price nothing, so under congestion their service cost lands on
    /// whichever reply-carrying request queues next — this counter
    /// localizes congestion in TIME, not to the causing warp.
    pub bw_delayed_responses: u64,
}

impl Default for DramStats {
    fn default() -> Self {
        Self {
            arch: None,
            read_sectors: 0,
            bw_delayed_responses: 0,
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
        self.read_sectors = read_sectors;
        self.bw_delayed_responses = bw_delayed_responses;
        Ok(())
    }
}

/// Result of one kernel launch through the shared-L2 system.
#[derive(Debug, Clone)]
pub struct SystemResult {
    /// Deployment architecture validated across schedule and all hierarchy configs.
    pub arch: ArchId,
    /// Per traced SM id.
    pub per_sm: BTreeMap<i32, SmSimResult>,
    /// Shared-L2 counters for the whole launch.
    pub l2: L2Counters,
    /// Per-source L2 sector/hit split (driver-side emission observer).
    pub l2_by_source: L2SourceSplit,
    /// DRAM read-traffic observer (server engagement diagnostics).
    pub dram: DramStats,
}

/// Run every SM's blocks to completion over one shared L2. `l2` is
/// caller-constructed (and caller-drained afterwards) so the geometry
/// resolution stays at the composition root; it must be FRESH per kernel
/// (cold cache, NCU `--cache-control all` parity — same contract as the
/// per-SM L1s). `dram` selects the read-bandwidth model ([`DramBw`]):
/// `Unmodeled` is the latency-only ablation / equivalence-differential
/// mode; `Carded` rates are validated non-zero here (typed
/// [`SystemError::DramBandwidthInvalid`], never a clamp).
pub fn simulate_kernel_shared(
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

                // Predict this exact ascending mask walk without mutating
                // the caller-owned L2. Outcome-dependent timing, counters,
                // and DRAM service can then fail before the first access.
                let line = req.sector_id >> 2;
                let preview = l2.preview_mask(line, req.sector_mask, req.class);
                let requested_sectors = u64::from(preview.sectors);
                let counted_hits = u64::from(preview.counted_hits);
                let source_ix = req.source as usize;
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

                let can_fetch = req.class == sim_mem::L2OpClass::Read || req.reply != ReplyTo::None;
                let fetch_sectors = if can_fetch {
                    requested_sectors
                        .checked_sub(u64::from(preview.present_sectors))
                        .ok_or(SystemError::CounterOverflow {
                            counter: "DRAM fetch sectors per request",
                        })?
                } else {
                    0
                };
                let next_read_sectors = dram_stats.read_sectors.checked_add(fetch_sectors).ok_or(
                    SystemError::CounterOverflow {
                        counter: "DRAM read sectors",
                    },
                )?;
                let service_preview = if fetch_sectors > 0 {
                    let bytes =
                        fetch_sectors
                            .checked_mul(32)
                            .ok_or(SystemError::CounterOverflow {
                                counter: "DRAM request bytes",
                            })?;
                    match &dram_server {
                        Some(server) => Some(server.preview(req.issue_time, bytes)?),
                        None => None,
                    }
                } else {
                    None
                };
                let bw_ready = service_preview.map_or(0, |(_, ready)| ready);

                let response_plan = if req.reply != ReplyTo::None {
                    let bound = checked_cycle_add(
                        "unpriced L2-hit response bound",
                        req.issue_time,
                        params.l1_miss_latency,
                    )?;
                    unpriced.require(sm_ix, bound)?;
                    let latency_ready = if preview.all_present() {
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
                    let next_delayed = dram_stats
                        .bw_delayed_responses
                        .checked_add(u64::from(ready > latency_ready))
                        .ok_or(SystemError::CounterOverflow {
                            counter: "DRAM bandwidth-delayed responses",
                        })?;
                    Some((bound, ready, next_delayed))
                } else {
                    None
                };

                // Commit the already-preflighted mask walk in the same
                // ascending order used by preview_mask.
                for sector in 0..4u8 {
                    if req.sector_mask & (1 << sector) != 0 {
                        l2.access(line, sector, req.class);
                    }
                }
                l2_by_source.sectors[source_ix] = next_source_sectors;
                l2_by_source.hits[source_ix] = next_source_hits;
                dram_stats.read_sectors = next_read_sectors;
                if let Some((done, _)) = service_preview
                    && let Some(server) = &mut dram_server
                {
                    server.free_fp = done;
                }
                if let Some((bound, ready, next_delayed)) = response_plan {
                    dram_stats.bw_delayed_responses = next_delayed;
                    queue.schedule_at(ready, ev.target, Ev::MemResponse(req.reply))?;
                    response_due.push(sm_ix, ready);
                    unpriced.remove(sm_ix, bound)?;
                }
                spin.on_event(ev.time, false)?;
            }
            Ev::MemResponse(reply) => {
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
    }

    let mut per_sm = BTreeMap::new();
    for (ix, core) in cores.iter().enumerate() {
        if !core.is_done() {
            return Err(SystemError::Deadlock {
                sm_id: sm_ids[ix],
                outstanding: core.outstanding_responses(),
            });
        }
        per_sm.insert(sm_ids[ix], core.result());
    }
    Ok(SystemResult {
        arch,
        per_sm,
        l2: l2.counters,
        l2_by_source,
        dram: dram_stats,
    })
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
    let responses = response_due.next(ix).unwrap_or(u64::MAX);
    let horizon = responses.min(unpriced.min(ix).unwrap_or(u64::MAX));
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

    use sim_mem::{CacheOpToken, L2Config, L2Replacement, SliceHash};

    use super::*;
    use crate::control_bits::{BARRIER_NONE, ControlBits};
    use crate::issue::{AccessKind, ReplayInst, WarpReplay};
    use crate::program::{IssueSlot, OpClass};

    fn cb(stall: u8) -> ControlBits {
        ControlBits {
            stall,
            yield_flag: false,
            write_barrier: BARRIER_NONE,
            read_barrier: BARRIER_NONE,
            wait_mask: 0,
            reuse: 0,
        }
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
        load.write_barrier = 0;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000001;
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
    fn dram_server_rejects_reserved_and_overflowed_completions_transactionally() {
        let reserved = DramServer::new(DramBw::Carded {
            arch: ArchId::Sm86,
            bytes_per_sec: 32,
            sm_clock_hz: u64::MAX,
        })
        .unwrap();
        assert!(matches!(
            reserved.preview(0, 32),
            Err(SystemError::DramCycleDomainExhausted { .. })
        ));
        assert_eq!(reserved.free_fp, 0);

        let overflow = DramServer {
            free_fp: u128::MAX,
            bytes_per_sec: 1,
            sm_clock_hz: 1,
        };
        assert!(matches!(
            overflow.preview(0, 1),
            Err(SystemError::DramFixedPointOverflow {
                operation: "service completion",
            })
        ));
        assert_eq!(overflow.free_fp, u128::MAX);
    }

    #[test]
    fn dram_domain_error_precedes_l2_mutation() {
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
                bytes_per_sec: 32,
                sm_clock_hz: u64::MAX,
            },
        )
        .unwrap_err();
        assert!(matches!(err, SystemError::DramCycleDomainExhausted { .. }));
        assert_eq!(l2.counters.total().unwrap(), 0);
    }

    #[test]
    fn l2_hit_does_not_preflight_hypothetical_dram_service() {
        let mut load = cb(1);
        load.write_barrier = 0;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000001;
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
                bytes_per_sec: 32,
                sm_clock_hz: u64::MAX,
            },
        )
        .unwrap();
        assert_eq!(result.dram.read_sectors, 0);
        assert_eq!(result.l2.hits, 1);
        assert_eq!(result.l2.misses, 1);
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

    /// TWO SMs load the same sector: the later request (per the queue's
    /// deterministic order) hits the line SM 0's miss installed, so its
    /// consumer runs at the L2-HIT arm — cross-SM sharing that per-SM
    /// constant latency cannot express.
    #[test]
    fn second_sm_hits_the_line_the_first_sm_filled() {
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
        // Both SMs issue their load at t=0; SM 0's access is processed
        // first (lower seq), misses and installs; SM 7's SAME-cycle
        // access then hits the (occupancy-model) resident sector.
        assert_eq!(r.l2.misses, 1);
        assert_eq!(r.l2.hits, 1);
        assert_eq!(r.per_sm[&0].cycles, 424, "miss chain: DRAM arm");
        assert_eq!(r.per_sm[&7].cycles, 245, "hit chain: L2-hit arm");
    }

    /// Same-sector reuse WITHIN one SM is an L1 hit and never reaches
    /// the L2; a different-line load from another SM does.
    #[test]
    fn l1_filters_the_l2_stream() {
        let mut load = cb(1);
        load.write_barrier = 0;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000001;
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
        store_cb.read_barrier = 1;
        let store_sched_pcs = vec![(0, slot(store_cb, "STG.E")), (16, slot(cb(1), "EXIT"))];
        let sched = schedule(
            store_sched_pcs
                .into_iter()
                .chain([
                    (32, {
                        let mut load = cb(1);
                        load.write_barrier = 0;
                        slot(load, "LDG.E")
                    }),
                    (48, {
                        let mut consumer = cb(1);
                        consumer.wait_mask = 0b000001;
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
    }

    /// §6.4 split between counting and pricing: an ATOM to a fresh
    /// sector of a RESIDENT line COUNTS as a hit (write-class tag-match
    /// rule) but its reply is PRICED on the DRAM arm — the atomic's
    /// return value needs the sector data the L2 does not hold yet.
    #[test]
    fn atomic_tag_hit_counts_hit_but_prices_dram() {
        let mut atom_cb = cb(1);
        atom_cb.write_barrier = 2;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000100;
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
        assert_eq!(
            r.per_sm[&0].cycles, 425,
            "…but the reply is PRICED on the DRAM arm (issue t=1 + 423 + 1)"
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
        load.write_barrier = 0;
        let mut consumer = cb(1);
        consumer.wait_mask = 0b000001;
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
            load.write_barrier = 0;
            let mut consumer = cb(1);
            consumer.wait_mask = 0b000001;
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
