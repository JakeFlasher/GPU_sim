// Lexical implementation section 1/4; included by the parent facade.
use std::cell::RefCell;

use std::cmp::Reverse;

#[cfg(feature = "reference")]
use std::collections::BTreeSet;

use std::collections::{BTreeMap, BinaryHeap, HashMap};

#[cfg(feature = "reference")]
use std::fs::File;

#[cfg(feature = "reference")]
use std::io::{self, BufWriter, Write};

use std::num::NonZeroU64;

#[cfg(feature = "reference")]
use std::path::PathBuf;

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
struct PendingBounds(Vec<BTreeMap<u64, u32>>);

#[derive(Debug, Clone, Copy)]
enum PendingRemoval {
    Decrement { sm: usize, bound: u64, next: u32 },
    Remove { sm: usize, bound: u64 },
}

impl PendingRemoval {
    fn commit(self, bounds: &mut PendingBounds) {
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

impl PendingBounds {
    fn insert(&mut self, sm: usize, bound: u64) -> Result<(), SystemError> {
        let next = self.0[sm]
            .get(&bound)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(SystemError::CounterOverflow {
                counter: "pending response-bound multiplicity",
            })?;
        self.0[sm].insert(bound, next);
        Ok(())
    }

    fn prepare_remove(&self, sm: usize, bound: u64) -> Result<PendingRemoval, SystemError> {
        match self.0[sm].get(&bound).copied() {
            Some(n) if n > 1 => Ok(PendingRemoval::Decrement {
                sm,
                bound,
                next: n - 1,
            }),
            Some(_) => Ok(PendingRemoval::Remove { sm, bound }),
            None => Err(SystemError::PendingBoundMissing { sm, bound }),
        }
    }

    fn min(&self, sm: usize) -> Option<u64> {
        self.0[sm].first_key_value().map(|(&t, _)| t)
    }

    fn is_empty(&self) -> bool {
        self.0.iter().all(BTreeMap::is_empty)
    }

    fn pending_entries(&self) -> usize {
        self.0.iter().map(BTreeMap::len).sum()
    }
}

use sim_arch::{ArchId, L2PendingReadSemantics};

use sim_engine::{EngineError, EventQueue, TargetId};

use sim_mem::{
    CoherenceConfig, CoherencePlan, CoherenceQueue, CoherenceRequest, L1Config,
    L1PayloadResidencyWitness, L1VisibilityConfig, L1VisibilityPlan, L1VisibilityQueue, L2Counters,
    L2DirtyResidency, L2EventMaskPreview, L2Eviction, L2FillToken, L2Model, L2PendingSectorKey,
    L2Source, MemoryOrigin, NativePayloadJoinKey, PayloadForwardingLedgerV1, SectorPayloadV1,
    ServicePlan, ServiceRequest, WritebackQueue,
};

#[cfg(feature = "reference")]
use sim_semantics::{
    BoundaryStateError, DramBoundaryState, DramStateError, L2RequestKey,
    NativeBoundaryExecutionResult, PowerBoundaryState, PowerStateError, QueueBoundaryState,
    QueueStateError, TYPED_STATE_SCHEMA_VERSION, TypedAdmissionWitness, TypedCoreBoundaryState,
    TypedEventBoundaryRecord, TypedEventIdentity, TypedEventPayload, TypedEventTransition,
    TypedFillIdentity, TypedL2Source, TypedMemoryOperation, TypedMemoryOrigin, TypedMemoryRequest,
    TypedObservationContract, TypedPendingEvent, TypedResponseIdentity, TypedResponseKind,
    TypedScalarMacroSpec, compare_native_boundary_projection, compare_native_region_contract,
    execute_typed_scalar_macro, native_boundary_execution_path,
};

/// Stable response identity used only at the typed boundary. The process-local
/// response-owner token remains private; these three fields are the identity
/// already exposed by the semantic event payload.
type ResponseBoundaryId = (u64, u64, u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DueEntry {
    time: u64,
    insertion: u64,
    response: Option<ResponseBoundaryId>,
    l2_allocation_generation: Option<u64>,
}

impl Ord for DueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.time, self.insertion).cmp(&(other.time, other.insertion))
    }
}

impl PartialOrd for DueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Per-actor pending due-times with the one choreography the driver repeats
/// for responses and wakes. Response entries retain their semantic identity,
/// so a typed snapshot can normalize an exact due cycle and delivery checks
/// cannot accidentally pair two same-cycle responses by timestamp alone.
struct DueTimes {
    entries: Vec<BinaryHeap<Reverse<DueEntry>>>,
    next_insertion: u64,
}

impl DueTimes {
    fn new(n: usize) -> Self {
        Self {
            entries: (0..n).map(|_| BinaryHeap::new()).collect(),
            next_insertion: 0,
        }
    }

    fn push(&mut self, ix: usize, t: u64) -> Result<(), SystemError> {
        self.push_entry(ix, t, None, None)
    }

    fn push_response(
        &mut self,
        ix: usize,
        t: u64,
        response: Option<ResponseBoundaryId>,
        l2_allocation_generation: Option<u64>,
    ) -> Result<(), SystemError> {
        self.push_entry(ix, t, response, l2_allocation_generation)
    }

    fn push_entry(
        &mut self,
        ix: usize,
        t: u64,
        response: Option<ResponseBoundaryId>,
        l2_allocation_generation: Option<u64>,
    ) -> Result<(), SystemError> {
        let insertion = self
            .next_insertion
            .checked_add(1)
            .ok_or(SystemError::CounterOverflow {
                counter: "due-time insertion sequence",
            })?;
        self.entries[ix].push(Reverse(DueEntry {
            time: t,
            insertion: self.next_insertion,
            response,
            l2_allocation_generation,
        }));
        self.next_insertion = insertion;
        Ok(())
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
        self.entries[ix].pop();
        Ok(())
    }

    fn pop_response_expect(
        &mut self,
        component: &'static str,
        ix: usize,
        t: u64,
        response: Option<ResponseBoundaryId>,
    ) -> Result<(), SystemError> {
        let Some(Reverse(entry)) = self.entries[ix].peek().copied() else {
            return Err(SystemError::DueTimeMismatch {
                component,
                sm: ix,
                expected: t,
                got: None,
            });
        };
        if entry.time != t {
            return Err(SystemError::DueTimeMismatch {
                component,
                sm: ix,
                expected: t,
                got: Some(entry.time),
            });
        }
        if entry.response != response {
            return Err(SystemError::ResponseIdentityMismatch {
                component,
                sm: ix,
                expected: response,
                got: entry.response,
            });
        }
        self.entries[ix].pop();
        Ok(())
    }

    fn next(&self, ix: usize) -> Option<u64> {
        self.entries[ix].peek().map(|&Reverse(entry)| entry.time)
    }

    #[cfg(feature = "reference")]
    fn response_entries(&self, ix: usize) -> Vec<(ResponseBoundaryId, u64, Option<u64>)> {
        let mut entries = self.entries[ix]
            .iter()
            .filter_map(|Reverse(entry)| {
                entry.response.map(|response| {
                    (
                        response,
                        entry.time,
                        entry.l2_allocation_generation,
                        entry.insertion,
                    )
                })
            })
            .collect::<Vec<_>>();
        entries.sort_unstable_by_key(|(_, _, _, insertion)| *insertion);
        entries
            .into_iter()
            .map(|(response, time, generation, _)| (response, time, generation))
            .collect()
    }

    fn is_empty(&self) -> bool {
        self.entries.iter().all(BinaryHeap::is_empty)
    }

    fn pending_entries(&self) -> usize {
        self.entries.iter().map(BinaryHeap::len).sum()
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
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct L2SourceSplit {
    arch: Option<ArchId>,
    pub sectors: [u64; 4],
    pub hits: [u64; 4],
    /// Operation-class lookup observer, separate from the mixed source lanes.
    /// These fields are populated only when the request operation is known;
    /// they do not alter L2 state or timing.
    pub read_lookup_requests: u64,
    pub read_lookup_hits: u64,
    /// Subset of read lookup hits caused only by coalescing with an already
    /// pending fill.  This is retained as a diagnostic so source-qualified
    /// read observers can be compared with and without pending-merge hits;
    /// it does not alter cache state or timing.
    pub read_pending_merge_hits: u64,
    pub store_lookup_requests: u64,
    pub store_lookup_hits: u64,
    /// Observer-only resident hits among the launch-local coalesced store
    /// packet candidate. This is distinct from the physical per-record store
    /// hit lane: repeated writes to one sector may hit the timed L2 model but
    /// are one source-TEX packet at the external client boundary.
    pub store_packet_lookup_hits: u64,
    /// Resident-sector hits among packet candidates, before write-tag
    /// coalescing. This is a report-only lower-bound observer.
    pub store_packet_resident_lookup_hits: u64,
    /// Observer-only per-SM line/sector packet candidate. A second global
    /// store to the same sector within the bounded admission window is
    /// retained as one candidate packet; this never changes the L2 state,
    /// source-sector baseline, DRAM demand, or timing.
    pub store_packet_candidate_sectors: u64,
    /// Passive client-scope shadow counters. They never enter physical
    /// L2/DRAM/timing state or scored hit/miss totals.
    pub client_scope_requests: [u64; 10],
    pub client_scope_resident_hits: [u64; 10],
    pub client_scope_pending_merges: [u64; 10],
    pub client_scope_fill_misses: [u64; 10],
    pub ltc128b_shadow_promoted_sectors: u64,
    pub atomic_lookup_requests: u64,
    pub atomic_lookup_hits: u64,
    pub reduction_lookup_requests: u64,
    pub reduction_lookup_hits: u64,
    /// Source-qualified samples of the number of pending L2 sectors observed
    /// immediately before each event request. This is a PRT/fill-pressure
    /// observer, not a GA102 capacity claim or a timing input.
    pub pending_pressure_samples: [u64; 4],
    /// Sum of pending-sector counts for the corresponding source samples.
    /// Consumers can form a bounded mean pressure without treating a gauge as
    /// additive traffic or selecting a hidden hash.
    pub pending_pressure_sum: [u64; 4],
    origin_sectors: [[u64; 4]; 2],
    origin_hits: [[u64; 4]; 2],
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
        let mut pending_pressure_samples = [0; 4];
        let mut pending_pressure_sum = [0; 4];
        let mut client_scope_requests = [0; 10];
        let mut client_scope_resident_hits = [0; 10];
        let mut client_scope_pending_merges = [0; 10];
        let mut client_scope_fill_misses = [0; 10];
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
            pending_pressure_samples[i] = self.pending_pressure_samples[i]
                .checked_add(other.pending_pressure_samples[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 source pending-pressure samples",
                })?;
            pending_pressure_sum[i] = self.pending_pressure_sum[i]
                .checked_add(other.pending_pressure_sum[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 source pending-pressure sum",
                })?;
        }
        for i in 0..10 {
            client_scope_requests[i] = self.client_scope_requests[i]
                .checked_add(other.client_scope_requests[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 client-scope requests",
                })?;
            client_scope_resident_hits[i] = self.client_scope_resident_hits[i]
                .checked_add(other.client_scope_resident_hits[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 client-scope resident hits",
                })?;
            client_scope_pending_merges[i] = self.client_scope_pending_merges[i]
                .checked_add(other.client_scope_pending_merges[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 client-scope pending merges",
                })?;
            client_scope_fill_misses[i] = self.client_scope_fill_misses[i]
                .checked_add(other.client_scope_fill_misses[i])
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 client-scope fill misses",
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
            read_lookup_requests: self
                .read_lookup_requests
                .checked_add(other.read_lookup_requests)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 read lookup requests",
                })?,
            read_lookup_hits: self
                .read_lookup_hits
                .checked_add(other.read_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 read lookup hits",
                })?,
            read_pending_merge_hits: self
                .read_pending_merge_hits
                .checked_add(other.read_pending_merge_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 read pending-merge lookup hits",
                })?,
            store_lookup_requests: self
                .store_lookup_requests
                .checked_add(other.store_lookup_requests)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 store lookup requests",
                })?,
            store_lookup_hits: self
                .store_lookup_hits
                .checked_add(other.store_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 store lookup hits",
                })?,
            store_packet_lookup_hits: self
                .store_packet_lookup_hits
                .checked_add(other.store_packet_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 store packet lookup hits",
                })?,
            store_packet_resident_lookup_hits: self
                .store_packet_resident_lookup_hits
                .checked_add(other.store_packet_resident_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 resident store packet lookup hits",
                })?,
            store_packet_candidate_sectors: self
                .store_packet_candidate_sectors
                .checked_add(other.store_packet_candidate_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 store packet candidate sectors",
                })?,
            atomic_lookup_requests: self
                .atomic_lookup_requests
                .checked_add(other.atomic_lookup_requests)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 atomic lookup requests",
                })?,
            atomic_lookup_hits: self
                .atomic_lookup_hits
                .checked_add(other.atomic_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 atomic lookup hits",
                })?,
            reduction_lookup_requests: self
                .reduction_lookup_requests
                .checked_add(other.reduction_lookup_requests)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 reduction lookup requests",
                })?,
            reduction_lookup_hits: self
                .reduction_lookup_hits
                .checked_add(other.reduction_lookup_hits)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 reduction lookup hits",
                })?,
            pending_pressure_samples,
            pending_pressure_sum,
            client_scope_requests,
            client_scope_resident_hits,
            client_scope_pending_merges,
            client_scope_fill_misses,
            ltc128b_shadow_promoted_sectors: self
                .ltc128b_shadow_promoted_sectors
                .checked_add(other.ltc128b_shadow_promoted_sectors)
                .ok_or(SystemCounterError::Overflow {
                    counter: "L2 LTC128B shadow promoted sectors",
                })?,
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

#[cfg(feature = "reference")]
use crate::issue::AdvanceDiagnostics;

use crate::issue::{
    AdmissionRequestWitness, Advance, BlockReplay, IssueError, IssueModelParams, L2ClientScope,
    MemRequest, MemoryAdmissionCandidate, MemoryAdmissionCandidateState, MemoryOperation,
    MemoryServiceCandidate, MemoryServiceCandidateOperation, MemoryWriteAck, PayloadStoreSite,
    ReplayHorizon, ReplyTo, SharedMemoryResidency, SmCore, SmMemMode, SmSimResult, TailFlags,
    checked_cycle_add, stateful_write_lookup_observer_enabled,
    validate_memory_admission_candidates, validate_memory_service_candidates,
};

use crate::program::FunctionSchedule;

#[cfg(feature = "reference")]
use sim_mem::FillKey;

use sim_mem::{L1WriteLookupObserver, MemRole};

/// Gate the opt-in write-admission/completion arm on a source-qualified
/// workload shape. A candidate may be present in the replay configuration for
/// every launch, but it is timing-active only when the static stream contains
/// global STORE/RED work and either a device-scope fence/ERRBAR/CCTL marker or
/// the candidate's explicit terminal-EXIT drain arm. The latter is a bounded
/// no-fence path: it reuses the typed acknowledgement stream rather than
/// claiming a generic store queue or a physical CTA-retirement mechanism.
fn qualifies_store_fence_feedback(
    schedule: &FunctionSchedule,
    candidate: MemoryAdmissionCandidate,
) -> bool {
    let has_global_write = schedule.by_pc.values().any(|slot| {
        matches!(
            (candidate.operation, slot.mem_role()),
            (MemoryServiceCandidateOperation::Store, MemRole::Store)
                | (
                    MemoryServiceCandidateOperation::Reduction,
                    MemRole::Reduction
                )
        )
    });
    let has_device_fence = schedule.by_pc.values().any(|slot| {
        slot.visibility_fence_semantics()
            .is_some_and(|(scope, _)| scope == sim_mem::VisibilityScope::Device)
    });
    let has_errbar_or_cctl = schedule.by_pc.values().any(|slot| {
        let opcode = slot.opcode_text().split_ascii_whitespace().next();
        matches!(opcode, Some("ERRBAR") | Some("CCTL.IVALL"))
    });
    has_global_write
        && (has_device_fence
            || has_errbar_or_cctl
            || ((candidate.exit_waits_for_service_completion
                || candidate.exit_waits_for_admission_drain)
                && !has_device_fence
                && !has_errbar_or_cctl))
}

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
    #[cfg(feature = "reference")]
    #[error("reference trace: {0}")]
    ReferenceTrace(#[from] ReferenceTraceError),
    #[cfg(feature = "reference")]
    #[error("typed power state: {0}")]
    Power(#[from] PowerStateError),
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
    #[error(transparent)]
    Writeback(#[from] sim_mem::WritebackError),
    #[error(transparent)]
    Coherence(#[from] sim_mem::CoherenceError),
    #[error(transparent)]
    L1Visibility(#[from] sim_mem::L1VisibilityError),
    #[error("logical L1 payload forwarding failed: {0}")]
    PayloadForwarding(String),
    #[error("L1 visibility protocol requires an opt-in coherence configuration")]
    L1VisibilityProtocolRequiresCoherence,
    #[error("resident L1 dirty state requires the carded writeback queue")]
    L1DirtyWritebackRequiresQueue,
    #[error("resident L1 dirty state requires the typed L1 visibility protocol")]
    L1DirtyWritebackRequiresVisibilityProtocol,
    #[error(
        "resident L1 writeback is incompatible with timing candidate/feedback lanes until hidden L2 write admissions are modeled"
    )]
    L1DirtyWritebackCandidateConflict,
    #[error(
        "resident L1/L2 dirty masks overlap on line {line_id}: L1={l1_mask:#06b}, L2={l2_mask:#06b}; payload merge is required"
    )]
    L1L2DirtyOverlap {
        line_id: i64,
        l1_mask: u8,
        l2_mask: u8,
    },
    #[error("L1 payload admission event targets unknown traced SM {sm_id}")]
    L1PayloadAdmissionSmUnknown { sm_id: i32 },
    #[error("L1 payload admission event at t={time} carries no payload records")]
    L1PayloadAdmissionEmpty { time: u64 },
    #[error("L1 payload admission event with dynamic identity has no retained store site")]
    L1PayloadAdmissionWriterMissing,
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
        "{component} response identity mismatch for SM index {sm}: expected {expected:?}, queued head was {got:?}"
    )]
    ResponseIdentityMismatch {
        component: &'static str,
        sm: usize,
        expected: Option<(u64, u64, u64)>,
        got: Option<(u64, u64, u64)>,
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
    #[error("event queue drained with {entries} pending response due-time entries")]
    ResponseDueNotDrained { entries: usize },
    #[error("event queue drained with {entries} pending SM wake due-time entries")]
    WakeDueNotDrained { entries: usize },
    #[error("event queue drained with {entries} pending L1 payload-admission due-time entries")]
    PayloadAdmissionDueNotDrained { entries: usize },
    #[error("event queue drained with {entries} pending response-bound entries")]
    PendingBoundsNotDrained { entries: usize },
    #[error("event queue drained with sm {sm_id} still owning {outstanding} response(s)")]
    ResponseNotDrained { sm_id: i32, outstanding: usize },
    #[error(
        "L2 fill schedule mismatch for sector {sector_id}: token={has_token}, ready={has_ready}"
    )]
    FillScheduleMismatch {
        sector_id: i64,
        has_token: bool,
        has_ready: bool,
    },
    #[error("pending response bound {bound} for SM index {sm} is absent")]
    PendingBoundMissing { sm: usize, bound: u64 },
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
    #[error(
        "event queue drained with {sectors} dirty writeback sectors and {entries} pending completions"
    )]
    WritebackNotDrained { sectors: u32, entries: usize },
    #[error(
        "event queue drained with {sectors} coherence service sectors and {entries} pending completions"
    )]
    CoherenceNotDrained { sectors: u32, entries: usize },
    #[error("L1 visibility completion at {time} conflicts with an existing SM wake for actor {sm}")]
    L1VisibilityWakeConflict { sm: usize, time: u64 },
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

fn require_l2_pending_read_semantics(arch: ArchId, effective_l2: bool) -> Result<(), SystemError> {
    if effective_l2 {
        if arch != ArchId::Sm120 {
            return Err(SystemError::ArchitectureMismatch {
                component: "effective L2 scenario",
                expected: ArchId::Sm120,
                got: arch,
            });
        }
        // The effective path explicitly selects the same allocation/sector
        // merge rule used by the implemented L2 engine. This is a scenario
        // assumption, not an SM120 architectural claim.
        return Ok(());
    }
    match arch
        .l2_pending_read_semantics()
        .require("l2_pending_read_semantics")?
    {
        L2PendingReadSemantics::AllocationSectorMergeV1 => Ok(()),
    }
}

fn checked_l2_counters(expected: ArchId, l2: &L2Model) -> Result<&L2Counters, SystemError> {
    let counters = l2.counters().map_err(sim_mem::L2AccessError::Counter)?;
    let got = counters.arch();
    let got = got.ok_or(sim_mem::L2AccessError::Counter(
        sim_mem::L2CounterError::Architecture(sim_arch::ArchitectureProvenanceError::Unbound {
            component: "L2 counters",
        }),
    ))?;
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
    L1PayloadAdmission {
        #[allow(dead_code)] // consumed by the optional typed reference projection
        identity_present: bool,
        writer: Option<PayloadStoreSite>,
        payloads: Box<[SectorPayloadV1]>,
        witnesses: Box<[L1PayloadResidencyWitness]>,
    },
    L2Access(MemRequest),
    L2AccessBatch(Box<[MemRequest]>),
    L2Fill(L2FillToken),
    /// Historical event name for an intrinsic no-response write-class RMW
    /// completion. It is used by both returning ATOM and non-returning RED;
    /// the public traffic counters keep those operations separate.
    AtomicCompletion,
    L2WritebackComplete {
        plan: ServicePlan,
    },
    L2CoherenceComplete {
        plan: CoherencePlan,
    },
    L1VisibilityComplete {
        plan: L1VisibilityPlan,
    },
    L1VisibilityAccess(sim_mem::L1AccessObservation),
    L1VisibilityFence(sim_mem::L1FenceObservation),
    L1FillVisible(sim_mem::L1FillVisibleObservation),
    L1InvalidateAll(sim_mem::L1InvalidateAllObservation),
    MemoryWriteAck(MemoryWriteAck),
    MemResponse {
        reply: ReplyTo,
        fill: Option<L2FillToken>,
    },
}

#[cfg(feature = "reference")]
fn event_name(event: &Ev) -> &'static str {
    match event {
        Ev::SmWake => "sm_wake",
        Ev::L1PayloadAdmission { .. } => "l1_payload_admission",
        Ev::L2Access(_) => "l2_access",
        Ev::L2AccessBatch(_) => "l2_access_batch",
        Ev::L2Fill(_) => "l2_fill",
        Ev::AtomicCompletion => "atomic_completion",
        Ev::L2WritebackComplete { .. } => "l2_writeback_complete",
        Ev::L2CoherenceComplete { .. } => "l2_coherence_complete",
        Ev::L1VisibilityComplete { .. } => "l1_visibility_complete",
        Ev::L1VisibilityAccess(_) => "l1_visibility_access",
        Ev::L1VisibilityFence(_) => "l1_visibility_fence",
        Ev::L1FillVisible(_) => "l1_fill_visible",
        Ev::L1InvalidateAll(_) => "l1_invalidate_all",
        Ev::MemoryWriteAck(_) => "memory_write_ack",
        Ev::MemResponse { .. } => "memory_response",
    }
}

#[cfg(feature = "reference")]
fn typed_event_payload(event: &Ev) -> TypedEventPayload {
    let typed_response_kind = |reply: ReplyTo| match reply {
        ReplyTo::L1Fill {
            fill: FillKey::Line { .. },
            ..
        } => Some(TypedResponseKind::L1Fill),
        ReplyTo::L1Fill {
            fill: FillKey::Sector { .. },
            ..
        } => Some(TypedResponseKind::L1SectorFill),
        ReplyTo::Warp { .. } => Some(TypedResponseKind::Warp),
        ReplyTo::AsyncCopy { .. } => Some(TypedResponseKind::AsyncCopy),
        ReplyTo::None => None,
    };
    let typed_origin = |origin: MemoryOrigin| match origin {
        MemoryOrigin::Global => TypedMemoryOrigin::Global,
        MemoryOrigin::Local => TypedMemoryOrigin::Local,
    };
    let typed_source = |source: L2Source| match source {
        L2Source::L1Fill => TypedL2Source::L1Fill,
        L2Source::Bypass => TypedL2Source::Bypass,
        L2Source::Untagged => TypedL2Source::Untagged,
        L2Source::Store => TypedL2Source::Store,
    };
    let typed_operation = |operation: MemoryOperation| match operation {
        MemoryOperation::Read => TypedMemoryOperation::Read,
        MemoryOperation::Store => TypedMemoryOperation::Store,
        MemoryOperation::Atomic => TypedMemoryOperation::Atomic,
        MemoryOperation::Reduction => TypedMemoryOperation::Reduction,
    };
    let response = |reply: ReplyTo| {
        reply.response_id().map(|id| {
            let (warp, instruction, sector_index) = id.boundary_identity();
            TypedResponseIdentity {
                warp,
                instruction,
                sector_index,
                response_kind: typed_response_kind(reply)
                    .expect("response id always has a response kind"),
            }
        })
    };
    let request_payload = |request: &MemRequest| TypedMemoryRequest {
        issue_time: request.issue_time,
        sector_id: request.sector_id,
        origin: typed_origin(request.origin),
        sector_mask: request.sector_mask,
        local_sector_mask: request.local_sector_mask,
        source: typed_source(request.source),
        operation: typed_operation(request.operation),
        constant_load: request.constant_load,
        constant_deep: request.constant_deep,
        reply: response(request.reply),
        admission_witness: request
            .admission_witness
            .map(|witness| TypedAdmissionWitness {
                warp: witness.warp(),
                instruction: witness.instruction(),
                subcore: witness.subcore(),
                request_index: witness.request_index(),
                request_count: witness.request_count(),
            }),
    };
    match event {
        Ev::SmWake => TypedEventPayload::SmWake,
        Ev::L1PayloadAdmission {
            payloads,
            witnesses,
            identity_present,
            ..
        } => TypedEventPayload::L1PayloadAdmission {
            payload_records: u64::try_from(payloads.len()).unwrap_or(u64::MAX),
            witness_lines: u64::try_from(witnesses.len()).unwrap_or(u64::MAX),
            identity_present: *identity_present,
        },
        Ev::L2Access(request) => TypedEventPayload::L2Access {
            request: request_payload(request),
        },
        Ev::L2AccessBatch(requests) => TypedEventPayload::L2AccessBatch {
            requests: requests.iter().map(request_payload).collect(),
        },
        Ev::L2Fill(fill) => {
            let (line_id, generation, sector_mask) = fill.boundary_identity();
            TypedEventPayload::L2Fill {
                fill: TypedFillIdentity {
                    line_id,
                    generation,
                    sector_mask,
                },
            }
        }
        Ev::AtomicCompletion => TypedEventPayload::AtomicCompletion,
        Ev::L2WritebackComplete { plan } => TypedEventPayload::WritebackCompletion {
            token: plan.token(),
            line_id: plan.line_id().unwrap_or_default(),
            dirty_mask: plan.dirty_mask(),
            service_done_at: plan.service_done_cycle(),
        },
        Ev::L2CoherenceComplete { plan } => TypedEventPayload::CoherenceCompletion {
            token: plan.token(),
            line_id: plan.request_line_id(),
            sector_mask: plan.request_sector_mask(),
            service_sectors: plan.service_sectors(),
            service_done_at: plan.service_done_cycle(),
        },
        Ev::L1VisibilityComplete { plan } => TypedEventPayload::L1VisibilityCompletion {
            token: plan.token(),
            line_id: plan.line_id(),
            sector_mask: plan.sector_mask(),
            service_sectors: plan.service_sectors(),
            service_done_at: plan.service_done_cycle(),
        },
        Ev::L1VisibilityAccess(observation) => TypedEventPayload::L1VisibilityAccess {
            sequence: observation.sequence,
            requester: observation.requester,
            line_id: observation.line_id,
            sector_mask: observation.sector_mask,
            is_write: observation.is_write,
            l1_resident_after: observation.l1_resident_after,
            fill_will_reside: observation.fill_will_reside,
            fill_generation: observation.fill_generation,
            hit: observation.hit,
            issue_cycle: observation.issue_cycle,
        },
        Ev::L1VisibilityFence(observation) => TypedEventPayload::L1VisibilityFence {
            sequence: observation.sequence,
            requester: observation.requester,
            pc: observation.pc,
            issue_cycle: observation.issue_cycle,
            scope: format!("{:?}", observation.scope),
            order: format!("{:?}", observation.order),
        },
        Ev::L1FillVisible(observation) => TypedEventPayload::L1FillVisible {
            sequence: observation.sequence,
            requester: observation.requester,
            line_id: observation.line_id,
            sector_mask: observation.sector_mask,
            fill_generation: observation.fill_generation,
            visible_cycle: observation.visible_cycle,
        },
        Ev::L1InvalidateAll(observation) => TypedEventPayload::L1InvalidateAll {
            sequence: observation.sequence,
            requester: observation.requester,
            issue_cycle: observation.issue_cycle,
        },
        Ev::MemoryWriteAck(ack) => TypedEventPayload::MemoryWriteAck {
            warp: ack.witness.warp(),
            instruction: ack.witness.instruction(),
            subcore: ack.witness.subcore(),
            request_index: ack.witness.request_index(),
            request_count: ack.witness.request_count(),
            operation: typed_operation(ack.operation),
            issue_at: ack.issue_at,
            accepted_at: ack.accepted_at,
            service_done_at: ack.service_done_at,
            fence_visible_at: ack.fence_visible_at,
        },
        Ev::MemResponse { reply, fill } => TypedEventPayload::MemoryResponse {
            reply: response(*reply),
            fill: fill.map(|fill| {
                let (line_id, generation, sector_mask) = fill.boundary_identity();
                TypedFillIdentity {
                    line_id,
                    generation,
                    sector_mask,
                }
            }),
        },
    }
}

#[cfg(feature = "reference")]
#[derive(Debug, thiserror::Error)]
pub enum ReferenceTraceError {
    #[error("I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("JSON encoding failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("record limit {limit} reached")]
    RecordLimit { limit: u64 },
    #[error("SIM_SM_REFERENCE_MAX_RECORDS must be an integer greater than 1, found {value:?}")]
    InvalidLimit { value: String },
    #[error("typed L2 projection failed: {0}")]
    L2Projection(#[from] sim_mem::L2ProjectionError),
    #[error("typed DRAM projection failed: {0}")]
    Dram(#[from] DramStateError),
    #[error("typed power projection failed: {0}")]
    Power(#[from] PowerStateError),
    #[error("typed queue projection failed: {0}")]
    Queue(#[from] QueueStateError),
    #[error("typed boundary structure validation failed: {0}")]
    Boundary(#[from] BoundaryStateError),
    #[error("reference macro witness is invalid: {0}")]
    Witness(String),
}

#[cfg(feature = "reference")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ReferenceMacroWitness {
    schema: String,
    target: u32,
    start_sequence: u64,
    spec: TypedScalarMacroSpec,
    detailed_internal_effects: Vec<TypedEventTransition>,
    detailed_labels: Vec<String>,
    detailed_exit: TypedEventBoundaryRecord,
}

#[cfg(feature = "reference")]
struct ReferenceTrace {
    writer: BufWriter<File>,
    records: u64,
    max_records: u64,
    previous_event: Option<serde_json::Value>,
    issue_effect_cursors: Vec<usize>,
    pending_typed_entry: Option<TypedEventBoundaryRecord>,
    witness: Option<ReferenceMacroWitness>,
    pending_native_result: Option<NativeBoundaryExecutionResult>,
    active_native_result: Option<NativeBoundaryExecutionResult>,
    pending_region_end: Option<TypedEventBoundaryRecord>,
    native_macro_dispatches: u64,
    native_macro_fallbacks: u64,
}

#[cfg(feature = "reference")]
const DEFAULT_REFERENCE_TRACE_MAX_RECORDS: u64 = 8_192;

#[cfg(feature = "reference")]
fn typed_scalar_macro_spec_from_env() -> Option<TypedScalarMacroSpec> {
    let instruction_count = std::env::var("SIM_SM_REFERENCE_TYPED_SCALAR_COUNT")
        .ok()?
        .parse::<u64>()
        .ok()?;
    let latency_cycles = std::env::var("SIM_SM_REFERENCE_TYPED_SCALAR_LATENCY")
        .ok()?
        .parse::<u64>()
        .ok()?;
    let initiation_interval_cycles = std::env::var("SIM_SM_REFERENCE_TYPED_SCALAR_INTERVAL")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1);
    let label = std::env::var("SIM_SM_REFERENCE_TYPED_SCALAR_LABEL")
        .unwrap_or_else(|_| "sm_wake".to_owned());
    Some(TypedScalarMacroSpec {
        instruction_count,
        latency_cycles,
        initiation_interval_cycles,
        label,
        issue_times: None,
        terminal_front_end_exit: None,
    })
}

/// Lossless replacements relative to the preceding event record. Equal-length
/// arrays keep their positions; a shape change replaces the entire container.
/// Every state is still projected after every event, independently of which
/// component the event is expected to change.
#[cfg(feature = "reference")]
#[derive(Clone, serde::Serialize)]
#[serde(untagged)]
enum ReferencePathSegment {
    Field(String),
    Index(usize),
}

#[cfg(feature = "reference")]
#[derive(serde::Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum ReferenceOperation<'a> {
    Replace {
        path: Vec<ReferencePathSegment>,
        value: &'a serde_json::Value,
    },
    Splice {
        path: Vec<ReferencePathSegment>,
        start: usize,
        delete_count: usize,
        values: Vec<&'a serde_json::Value>,
    },
}

#[cfg(feature = "reference")]
fn reference_operations<'a>(
    previous: &serde_json::Value,
    current: &'a serde_json::Value,
    path: &mut Vec<ReferencePathSegment>,
    changes: &mut Vec<ReferenceOperation<'a>>,
) {
    use serde_json::Value;
    if previous == current {
        return;
    }
    match (previous, current) {
        (Value::Object(old), Value::Object(new)) if old.keys().eq(new.keys()) => {
            for (key, value) in new {
                path.push(ReferencePathSegment::Field(key.clone()));
                reference_operations(&old[key], value, path, changes);
                path.pop();
            }
        }
        (Value::Array(old), Value::Array(new)) if old.len() == new.len() => {
            for (index, (old, new)) in old.iter().zip(new).enumerate() {
                path.push(ReferencePathSegment::Index(index));
                reference_operations(old, new, path, changes);
                path.pop();
            }
        }
        (Value::Array(old), Value::Array(new)) => {
            let prefix = old
                .iter()
                .zip(new)
                .take_while(|(left, right)| *left == *right)
                .count();
            let suffix_limit = (old.len() - prefix).min(new.len() - prefix);
            let mut suffix = 0;
            while suffix < suffix_limit
                && old[old.len() - 1 - suffix] == new[new.len() - 1 - suffix]
            {
                suffix += 1;
            }
            let new_end = new.len() - suffix;
            changes.push(ReferenceOperation::Splice {
                path: path.clone(),
                start: prefix,
                delete_count: old.len() - prefix - suffix,
                values: new[prefix..new_end].iter().collect(),
            });
        }
        _ => changes.push(ReferenceOperation::Replace {
            path: path.clone(),
            value: current,
        }),
    }
}

