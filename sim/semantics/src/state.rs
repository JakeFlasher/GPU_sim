//! Typed boundary state shared by the detailed and accelerated executors.
//!
//! These records are deliberately model-relative.  They describe the state
//! that the simulator can transform and compare; they do not claim that a
//! virtual trace address is a physical cache address or that activity counts
//! are measured watts.

use serde::{Deserialize, Serialize};

use crate::boundary::{BlockBoundaryState, WarpBoundaryState};

pub const TYPED_STATE_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AddressDomain {
    VirtualAbsolute,
    AllocationRelative,
    Physical,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CanonicalLineKey {
    VirtualAbsolute {
        line_id: i64,
    },
    AllocationRelative {
        allocation_id: u32,
        line_offset: u64,
    },
    Physical {
        line_id: u64,
    },
    Unknown {
        line_id: Option<i64>,
    },
}

impl CanonicalLineKey {
    pub fn domain(&self) -> AddressDomain {
        match self {
            Self::VirtualAbsolute { .. } => AddressDomain::VirtualAbsolute,
            Self::AllocationRelative { .. } => AddressDomain::AllocationRelative,
            Self::Physical { .. } => AddressDomain::Physical,
            Self::Unknown { .. } => AddressDomain::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L2AccessDisciplineState {
    Immediate,
    Event,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L2SectorStatus {
    Valid,
    Pending,
    Dirty,
    Stale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L2RequestState {
    Outstanding,
    Completed,
    Stale,
    NonReturning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2GeometryState {
    pub architecture: String,
    pub capacity_bytes: u64,
    pub line_bytes: u32,
    pub sector_bytes: u32,
    pub num_slices: u32,
    pub sets_per_slice: u64,
    pub associativity: u32,
    pub replacement: String,
    pub slice_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2CounterState {
    pub hits: u64,
    pub misses: u64,
    pub lookup_hits: u64,
    pub lookup_requests: u64,
    pub line_misses: u64,
    pub sector_fill_misses: u64,
    pub write_tag_hits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2WayState {
    pub line: CanonicalLineKey,
    pub valid_mask: u8,
    pub dirty_mask: u8,
    pub pending_mask: u8,
    pub recency: i64,
    pub generation: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2SetState {
    pub slice: u32,
    pub set: u64,
    pub ways: Vec<L2WayState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2RequestKey {
    pub sm_id: u32,
    pub warp_slot: i32,
    pub instruction: u64,
    pub sector_index: u32,
    pub response_kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2SectorOwnership {
    pub sector: u8,
    pub status: L2SectorStatus,
    pub producer: Option<L2RequestKey>,
    pub consumers: Vec<L2RequestKey>,
    pub ready_at: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2AllocationState {
    pub line: CanonicalLineKey,
    pub generation: u64,
    pub sectors: Vec<L2SectorOwnership>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2RequestOwner {
    pub request: L2RequestKey,
    pub line: CanonicalLineKey,
    pub sector_mask: u8,
    pub state: L2RequestState,
    pub due: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2BoundaryState {
    pub schema_version: u32,
    pub architecture: String,
    pub geometry: L2GeometryState,
    pub address_domain: AddressDomain,
    pub clock: i64,
    pub bip_counter: u64,
    pub access_discipline: Option<L2AccessDisciplineState>,
    pub counters: L2CounterState,
    pub observer_lookup_hits: u64,
    pub observer_lookup_requests: u64,
    pub next_generation: Option<u64>,
    pub sets: Vec<L2SetState>,
    pub allocations: Vec<L2AllocationState>,
    pub request_owners: Vec<L2RequestOwner>,
    pub ownership_complete: bool,
    pub unknown_state: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L1FillDueState {
    Timed { cycle: u64 },
    Untimed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L1LineFillState {
    pub line: i64,
    pub due: L1FillDueState,
    pub sequence: u64,
    pub generation: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L1SectorFillState {
    pub line: i64,
    pub sector: u8,
    pub generation: u32,
    pub due: L1FillDueState,
    pub sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L1CounterState {
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    pub pending_lookup_hits: u64,
    pub stall_events: u64,
    pub line_fills: u64,
    pub line_alloc_misses: u64,
    pub sector_resident_misses: u64,
    pub bypass_sectors: u64,
    pub bypass_atom_sectors: u64,
    pub untagged_sectors: u64,
    pub store_sectors: u64,
    pub store_hits: u64,
    pub store_tag_hits: u64,
    pub write_lookup_hits: u64,
    pub reduction_sectors: u64,
    pub reduction_hits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L1BoundaryState {
    pub schema_version: u32,
    pub architecture: String,
    pub line_capacity: u64,
    pub mshr_count: Option<u32>,
    pub hit_latency: u64,
    pub miss_latency: u64,
    pub line_order: Vec<(u64, i64)>,
    pub stamp_of: Vec<(i64, u64)>,
    pub sectors: Vec<(i64, u8)>,
    pub line_fills: Vec<L1LineFillState>,
    pub sector_fills: Vec<L1SectorFillState>,
    pub line_generations: Vec<(i64, u32)>,
    pub next_stamp: Option<u64>,
    pub fill_sequence: Option<u64>,
    pub active_lookup_batch: Option<(u64, bool)>,
    pub min_fill_due: Option<u64>,
    pub fill_timing: Option<String>,
    pub counters: L1CounterState,
    pub write_observer_present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsyncGroupBoundaryState {
    pub sequence: u64,
    pub committed_at: u64,
    pub pending_sources: u32,
    pub source_completion_time: u64,
    pub delivery_work: u64,
    pub destination_wavefronts: u64,
    pub destination_service_queued: bool,
    pub destination_completion_time: u64,
    pub completion_time: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsyncCopyBoundaryState {
    pub open: Option<AsyncGroupBoundaryState>,
    pub committed: Vec<AsyncGroupBoundaryState>,
    pub bypass_observation_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryPipelineBoundaryState {
    pub queue_entries_with_dispatch: u32,
    pub local_initiation_interval_cycles: u64,
    pub shared_accept_interval_cycles: u64,
    pub queue_release_times: Vec<Vec<u64>>,
    pub local_completion_ready: Vec<u64>,
    pub shared_accept_ready: u64,
    pub instructions_admitted: u64,
    pub operand_pipeline_delay_cycles: u64,
    pub shared_queue_delay_cycles: u64,
    pub maximum_operand_pipeline_delay_cycles: u64,
    pub maximum_shared_queue_delay_cycles: u64,
    pub maximum_queue_occupancy: u64,
    pub terminal_shared_accept_cycle: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarPipelineBoundaryState {
    pub latency_cycles: u64,
    pub initiation_interval_cycles: u64,
    pub in_flight_capacity: u32,
    pub release_times: Vec<Vec<u64>>,
    pub next_service: Vec<u64>,
    pub instructions_admitted: u64,
    pub service_busy_cycles: u64,
    pub queue_delay_cycles: u64,
    pub maximum_queue_occupancy: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionCacheEntryState {
    pub function_unique_id: i32,
    pub line: u64,
    pub ready_at: u64,
    pub last_access: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionCacheState {
    pub ways: u64,
    pub access_sequence: u64,
    pub sets: Vec<Vec<InstructionCacheEntryState>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionStreamLineState {
    pub function_unique_id: i32,
    pub line: u64,
    pub ready_at: u64,
    pub demanded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionStreamState {
    pub lines: Vec<InstructionStreamLineState>,
    pub next_line: Option<(i32, u64)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionFrontEndBoundaryState {
    pub line_bytes: u32,
    pub l0_to_l1_cycles: u64,
    pub l1_to_l0_cycles: u64,
    pub l1_request_ports: u32,
    pub l1_reply_ports: u32,
    pub stream_buffer_lines: u32,
    pub stream_buffers_per_subcore: u32,
    pub prefetches_per_cycle: u32,
    pub instruction_buffer_entries: u32,
    pub fetch_decode_width: u32,
    pub fetch_to_issue_stages: u64,
    pub l0: Vec<InstructionCacheState>,
    pub l1: InstructionCacheState,
    pub streams: Vec<InstructionStreamState>,
    pub l1_request_ready: u64,
    pub l1_reply_ready: u64,
    pub l0_response_ready: Vec<u64>,
    pub prefetch_priority: u64,
    pub cold_fill_latency: u64,
    pub instructions_fetched: u64,
    pub demand_l0_hits: u64,
    pub demand_stream_buffer_hits: u64,
    pub demand_l0_misses: u64,
    pub demand_l1_hits: u64,
    pub demand_l1_misses: u64,
    pub prefetch_requests: u64,
    pub prefetch_l1_hits: u64,
    pub prefetch_l1_misses: u64,
    pub stream_restarts: u64,
    pub maximum_fetch_latency_cycles: u64,
}

impl L2BoundaryState {
    pub fn validate(&self) -> Result<(), L2StateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(L2StateError::SchemaVersion {
                found: self.schema_version,
                expected: TYPED_STATE_SCHEMA_VERSION,
            });
        }
        if self.geometry.line_bytes == 0 || self.geometry.sector_bytes == 0 {
            return Err(L2StateError::InvalidGeometry);
        }
        if self.geometry.num_slices == 0 || self.geometry.associativity == 0 {
            return Err(L2StateError::InvalidGeometry);
        }
        if self.counters.lookup_hits > self.counters.lookup_requests {
            return Err(L2StateError::LookupHitsExceedRequests);
        }
        let mut previous = None;
        for set in &self.sets {
            if set.slice >= self.geometry.num_slices || set.set >= self.geometry.sets_per_slice {
                return Err(L2StateError::SetOutOfRange {
                    slice: set.slice,
                    set: set.set,
                });
            }
            if set.ways.len() > self.geometry.associativity as usize {
                return Err(L2StateError::SetOverfull { set: set.set });
            }
            if let Some(previous) = previous
                && (set.slice, set.set) <= previous
            {
                return Err(L2StateError::UnsortedSets);
            }
            previous = Some((set.slice, set.set));
            for way in &set.ways {
                if way.pending_mask & way.valid_mask != 0 {
                    return Err(L2StateError::PendingAndValidOverlap);
                }
                if way.dirty_mask & !way.valid_mask != 0 {
                    return Err(L2StateError::DirtyWithoutValid);
                }
            }
        }
        if self.ownership_complete && !self.unknown_state.is_empty() {
            return Err(L2StateError::CompleteWithUnknownState);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum L2StateError {
    #[error("typed L2 schema {found} is unsupported; expected {expected}")]
    SchemaVersion { found: u32, expected: u32 },
    #[error("typed L2 geometry contains a zero dimension")]
    InvalidGeometry,
    #[error("typed L2 lookup hits exceed lookup requests")]
    LookupHitsExceedRequests,
    #[error("typed L2 set ({slice},{set}) is outside declared geometry")]
    SetOutOfRange { slice: u32, set: u64 },
    #[error("typed L2 set {set} contains too many ways")]
    SetOverfull { set: u64 },
    #[error("typed L2 sets are not in canonical order")]
    UnsortedSets,
    #[error("typed L2 way has pending and valid sectors in the same bit")]
    PendingAndValidOverlap,
    #[error("typed L2 way has dirty sectors that are not valid")]
    DirtyWithoutValid,
    #[error("typed L2 state is marked complete while unknown fields remain")]
    CompleteWithUnknownState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerModelStatus {
    Omitted,
    ActivityOnly,
    AnalyticCoefficients,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerDomainStatus {
    Clocked,
    Idle,
    Gated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerClockState {
    pub clock_domain: String,
    pub fixed_frequency_hz: Option<u64>,
    pub dvfs_modeled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerDomainState {
    pub domain: String,
    pub status: PowerDomainStatus,
    pub last_cycle: u64,
    pub clocked_cycles: u64,
    pub gated_cycles: u64,
    pub sm_wake_events: u64,
    pub completion_events: u64,
    pub memory_events: u64,
    pub queue_events: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerSampleState {
    pub cycle: u64,
    pub activity_total: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PowerBoundaryState {
    pub schema_version: u32,
    pub status: PowerModelStatus,
    pub clock: PowerClockState,
    pub sample_period: Option<u64>,
    pub next_sample: Option<u64>,
    pub domains: Vec<PowerDomainState>,
    pub samples: Vec<PowerSampleState>,
    pub omitted_observations: Vec<String>,
    pub unknown_state: Vec<String>,
}

impl PowerBoundaryState {
    pub fn activity_only(frequency_hz: Option<u64>, sample_period: Option<u64>) -> Self {
        let next_sample = sample_period.filter(|period| *period != 0);
        Self {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            status: PowerModelStatus::ActivityOnly,
            clock: PowerClockState {
                clock_domain: "sm_clock".to_owned(),
                fixed_frequency_hz: frequency_hz,
                dvfs_modeled: false,
            },
            sample_period: next_sample,
            next_sample,
            domains: vec![PowerDomainState {
                domain: "modeled_event_system".to_owned(),
                status: PowerDomainStatus::Idle,
                last_cycle: 0,
                clocked_cycles: 0,
                gated_cycles: 0,
                sm_wake_events: 0,
                completion_events: 0,
                memory_events: 0,
                queue_events: 0,
            }],
            samples: Vec::new(),
            omitted_observations: vec![
                "watts".to_owned(),
                "per_instruction_energy".to_owned(),
                "lane_utilization".to_owned(),
                "thermal_and_dvfs_feedback".to_owned(),
            ],
            unknown_state: Vec::new(),
        }
    }

    pub fn advance_to(&mut self, cycle: u64) -> Result<(), PowerStateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(PowerStateError::SchemaVersion);
        }
        // Sampling is part of the state transition. Stage the complete update
        // so an overflow or bounded-output refusal never leaves half an
        // integration interval committed.
        let mut next = self.clone();
        next.emit_samples_through(cycle)?;
        next.integrate_to(cycle)?;
        *self = next;
        Ok(())
    }

    fn integrate_to(&mut self, cycle: u64) -> Result<(), PowerStateError> {
        for domain in &mut self.domains {
            if cycle < domain.last_cycle {
                return Err(PowerStateError::TimeTravel);
            }
            let delta = cycle - domain.last_cycle;
            match domain.status {
                PowerDomainStatus::Clocked => {
                    domain.clocked_cycles = domain
                        .clocked_cycles
                        .checked_add(delta)
                        .ok_or(PowerStateError::CounterOverflow)?;
                }
                PowerDomainStatus::Gated => {
                    domain.gated_cycles = domain
                        .gated_cycles
                        .checked_add(delta)
                        .ok_or(PowerStateError::CounterOverflow)?;
                }
                PowerDomainStatus::Idle => {}
            }
            domain.last_cycle = cycle;
        }
        Ok(())
    }

    pub fn record_event(&mut self, event: &str) -> Result<(), PowerStateError> {
        let mut next = self.clone();
        next.record_event_inner(event)?;
        *self = next;
        Ok(())
    }

    fn record_event_inner(&mut self, event: &str) -> Result<(), PowerStateError> {
        let domain = self.domains.first_mut().ok_or(PowerStateError::NoDomains)?;
        domain.status = PowerDomainStatus::Clocked;
        domain.queue_events = domain
            .queue_events
            .checked_add(1)
            .ok_or(PowerStateError::CounterOverflow)?;
        if event.contains("l2") || event.contains("memory") || event == "mem_response" {
            domain.memory_events = domain
                .memory_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
        }
        if event == "sm_wake" {
            domain.sm_wake_events = domain
                .sm_wake_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
        }
        if event == "memory_response" || event == "l2_fill" || event == "atomic_completion" {
            domain.completion_events = domain
                .completion_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
        }
        Ok(())
    }

    fn emit_samples_through(&mut self, cycle: u64) -> Result<(), PowerStateError> {
        let Some(period) = self.sample_period else {
            return Ok(());
        };
        if period == 0 {
            return Err(PowerStateError::InvalidSamplePeriod);
        }
        let Some(mut next) = self.next_sample else {
            return Ok(());
        };
        while next <= cycle {
            if self.samples.len() >= 100_000 {
                return Err(PowerStateError::SampleLimit);
            }
            self.integrate_to(next)?;
            let activity_total = self.domains.iter().try_fold(0_u64, |sum, domain| {
                sum.checked_add(domain.clocked_cycles)
                    .ok_or(PowerStateError::CounterOverflow)
            })?;
            self.samples.push(PowerSampleState {
                cycle: next,
                activity_total,
            });
            next = next
                .checked_add(period)
                .ok_or(PowerStateError::TimeOverflow)?;
        }
        self.next_sample = Some(next);
        Ok(())
    }

    pub fn validate(&self) -> Result<(), PowerStateError> {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(PowerStateError::SchemaVersion);
        }
        if self.clock.dvfs_modeled && self.status != PowerModelStatus::AnalyticCoefficients {
            return Err(PowerStateError::UnknownDvfs);
        }
        if self.status != PowerModelStatus::Unknown && !self.unknown_state.is_empty() {
            return Err(PowerStateError::UnknownState);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PowerStateError {
    #[error("typed power schema is unsupported")]
    SchemaVersion,
    #[error("power state moved backwards in time")]
    TimeTravel,
    #[error("power activity counter overflowed")]
    CounterOverflow,
    #[error("power sample period must be non-zero")]
    InvalidSamplePeriod,
    #[error("power sample time overflowed")]
    TimeOverflow,
    #[error("power sample output exceeds the bounded 100000-sample limit")]
    SampleLimit,
    #[error("power state has no domains")]
    NoDomains,
    #[error("DVFS state is not represented by the selected power model")]
    UnknownDvfs,
    #[error("power state contains unknown fields")]
    UnknownState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Completeness {
    Complete,
    Incomplete { unknown: Vec<String> },
    Refused { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedContractKind {
    BoundaryDiagnosticV2,
    CycleExactMemoryV2,
    CycleExactPowerV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedObservationContract {
    pub schema_version: u32,
    pub kind: TypedContractKind,
    pub address_domain: AddressDomain,
    pub power: PowerModelStatus,
}

impl TypedObservationContract {
    pub fn diagnostic() -> Self {
        Self {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            kind: TypedContractKind::BoundaryDiagnosticV2,
            address_domain: AddressDomain::Unknown,
            power: PowerModelStatus::Omitted,
        }
    }

    pub fn cycle_exact_memory() -> Self {
        Self {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            kind: TypedContractKind::CycleExactMemoryV2,
            address_domain: AddressDomain::VirtualAbsolute,
            power: PowerModelStatus::Omitted,
        }
    }

    pub fn cycle_exact_power() -> Self {
        Self {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            kind: TypedContractKind::CycleExactPowerV2,
            address_domain: AddressDomain::VirtualAbsolute,
            power: PowerModelStatus::ActivityOnly,
        }
    }

    pub fn validate(&self, l2: &L2BoundaryState, power: &PowerBoundaryState) -> Completeness {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Completeness::Refused {
                reason: "typed_contract_schema_version".to_owned(),
            };
        }
        if let Err(error) = l2.validate() {
            return Completeness::Refused {
                reason: error.to_string(),
            };
        }
        if !l2.ownership_complete {
            return Completeness::Incomplete {
                unknown: l2.unknown_state.clone(),
            };
        }
        if l2
            .request_owners
            .iter()
            .any(|owner| matches!(owner.state, L2RequestState::Outstanding) && owner.due.is_none())
        {
            return Completeness::Incomplete {
                unknown: vec!["request_due_time_unknown".to_owned()],
            };
        }
        if self.kind != TypedContractKind::BoundaryDiagnosticV2 {
            if l2.address_domain != self.address_domain {
                return Completeness::Refused {
                    reason: "address_domain_mismatch".to_owned(),
                };
            }
        }
        if self.power != PowerModelStatus::Omitted {
            if power.status != self.power {
                return Completeness::Incomplete {
                    unknown: vec!["power_model_status".to_owned()],
                };
            }
            if let Err(error) = power.validate() {
                return Completeness::Refused {
                    reason: error.to_string(),
                };
            }
        }
        Completeness::Complete
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TypedEventPayload {
    SmWake,
    L2Access {
        sector_id: i64,
        sector_mask: u8,
        source: String,
        operation: String,
        reply_kind: String,
    },
    L2AccessBatch {
        request_count: u32,
    },
    L2Fill {
        line_id: i64,
        generation: u64,
        sector_mask: u8,
    },
    AtomicCompletion,
    MemoryWriteAck,
    MemoryResponse {
        reply_kind: String,
        has_fill: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedPendingEvent {
    pub time: u64,
    pub sequence: u64,
    pub target: u32,
    pub payload: TypedEventPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueBoundaryState {
    pub depth: u64,
    pub next_event_time: Option<u64>,
    pub pending: Vec<TypedPendingEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedEventIdentity {
    pub time: u64,
    pub sequence: u64,
    pub target: u32,
    pub payload: TypedEventPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedCoreBoundaryState {
    pub sm_id: u32,
    pub cycle: u64,
    pub issued: u64,
    pub resident_warps: u32,
    pub resident_shared_memory_bytes: u64,
    pub unfinished_warps: u64,
    pub next_block: u64,
    pub hold: [Option<u64>; 4],
    pub subcore_warps: Vec<Vec<i32>>,
    pub outstanding_responses: u64,
    pub barrier_pipe_ready: u64,
    pub tensor_pipe_ready: [u64; 4],
    pub ldgsts_destination_pipe_ready: u64,
    pub warp_states: Vec<WarpBoundaryState>,
    pub block_states: Vec<BlockBoundaryState>,
    pub l1: Option<L1BoundaryState>,
    pub front_end: Option<InstructionFrontEndBoundaryState>,
    pub memory_pipeline: Option<MemoryPipelineBoundaryState>,
    pub scalar_pipeline: Option<ScalarPipelineBoundaryState>,
    pub async_copy: Vec<Option<AsyncCopyBoundaryState>>,
    pub response_owners: Vec<L2RequestOwner>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedEventBoundaryRecord {
    pub schema_version: u32,
    pub contract: TypedObservationContract,
    pub event: TypedEventIdentity,
    pub queue: QueueBoundaryState,
    pub cores: Vec<TypedCoreBoundaryState>,
    pub shared_l2: L2BoundaryState,
    pub power: PowerBoundaryState,
    pub completeness: Completeness,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_power_integrates_idle_and_emits_exact_samples() {
        let mut power = PowerBoundaryState::activity_only(Some(1_350_000_000), Some(4));
        power.advance_to(2).unwrap();
        power.record_event("sm_wake").unwrap();
        power.advance_to(9).unwrap();
        assert_eq!(power.domains[0].clocked_cycles, 7);
        assert_eq!(
            power
                .samples
                .iter()
                .map(|sample| sample.cycle)
                .collect::<Vec<_>>(),
            vec![4, 8]
        );
        assert_eq!(
            power
                .samples
                .iter()
                .map(|sample| sample.activity_total)
                .collect::<Vec<_>>(),
            vec![2, 6]
        );
        power.validate().unwrap();
    }

    #[test]
    fn power_rejects_unmodeled_dvfs_in_activity_mode() {
        let mut power = PowerBoundaryState::activity_only(None, None);
        power.clock.dvfs_modeled = true;
        assert_eq!(power.validate(), Err(PowerStateError::UnknownDvfs));
    }

    #[test]
    fn diagnostic_contract_accepts_typed_but_incomplete_l2() {
        let mut power = PowerBoundaryState::activity_only(None, None);
        let mut l2 = minimal_l2_state();
        l2.ownership_complete = false;
        l2.unknown_state = vec!["requester_ownership_join_pending".to_owned()];
        assert!(matches!(
            TypedObservationContract::diagnostic().validate(&l2, &power),
            Completeness::Incomplete { .. }
        ));
        l2.ownership_complete = true;
        l2.unknown_state.clear();
        l2.request_owners.push(L2RequestOwner {
            request: L2RequestKey {
                sm_id: 0,
                warp_slot: 0,
                instruction: 0,
                sector_index: 0,
                response_kind: "l1_fill".to_owned(),
            },
            line: CanonicalLineKey::VirtualAbsolute { line_id: 0 },
            sector_mask: 1,
            state: L2RequestState::Outstanding,
            due: None,
        });
        assert!(matches!(
            TypedObservationContract::cycle_exact_power().validate(&l2, &power),
            Completeness::Incomplete { .. }
        ));
        l2.request_owners[0].due = Some(4);
        assert!(matches!(
            TypedObservationContract::cycle_exact_power().validate(&l2, &power),
            Completeness::Complete
        ));
        power.clock.dvfs_modeled = true;
        assert!(matches!(
            TypedObservationContract::cycle_exact_power().validate(&l2, &power),
            Completeness::Refused { .. }
        ));
    }

    fn minimal_l2_state() -> L2BoundaryState {
        L2BoundaryState {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            architecture: "sm_86".to_owned(),
            geometry: L2GeometryState {
                architecture: "sm_86".to_owned(),
                capacity_bytes: 128,
                line_bytes: 128,
                sector_bytes: 32,
                num_slices: 1,
                sets_per_slice: 1,
                associativity: 1,
                replacement: "lru".to_owned(),
                slice_hash: "line_mod".to_owned(),
            },
            address_domain: AddressDomain::VirtualAbsolute,
            clock: 0,
            bip_counter: 0,
            access_discipline: None,
            counters: L2CounterState {
                hits: 0,
                misses: 0,
                lookup_hits: 0,
                lookup_requests: 0,
                line_misses: 0,
                sector_fill_misses: 0,
                write_tag_hits: 0,
            },
            observer_lookup_hits: 0,
            observer_lookup_requests: 0,
            next_generation: None,
            sets: Vec::new(),
            allocations: Vec::new(),
            request_owners: Vec::new(),
            ownership_complete: true,
            unknown_state: Vec::new(),
        }
    }

    #[test]
    fn typed_l2_state_rejects_pending_and_valid_overlap() {
        let mut state = L2BoundaryState {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            architecture: "sm_86".to_owned(),
            geometry: L2GeometryState {
                architecture: "sm_86".to_owned(),
                capacity_bytes: 128,
                line_bytes: 128,
                sector_bytes: 32,
                num_slices: 1,
                sets_per_slice: 1,
                associativity: 1,
                replacement: "lru".to_owned(),
                slice_hash: "line_mod".to_owned(),
            },
            address_domain: AddressDomain::VirtualAbsolute,
            clock: 0,
            bip_counter: 0,
            access_discipline: Some(L2AccessDisciplineState::Event),
            counters: L2CounterState {
                hits: 0,
                misses: 0,
                lookup_hits: 0,
                lookup_requests: 0,
                line_misses: 0,
                sector_fill_misses: 0,
                write_tag_hits: 0,
            },
            observer_lookup_hits: 0,
            observer_lookup_requests: 0,
            next_generation: Some(1),
            sets: vec![L2SetState {
                slice: 0,
                set: 0,
                ways: vec![L2WayState {
                    line: CanonicalLineKey::VirtualAbsolute { line_id: 0 },
                    valid_mask: 1,
                    dirty_mask: 0,
                    pending_mask: 1,
                    recency: 0,
                    generation: Some(0),
                }],
            }],
            allocations: Vec::new(),
            request_owners: Vec::new(),
            ownership_complete: false,
            unknown_state: vec!["requester_ownership_join_pending".to_owned()],
        };
        assert_eq!(state.validate(), Err(L2StateError::PendingAndValidOverlap));
        state.sets[0].ways[0].pending_mask = 0;
        assert!(state.validate().is_ok());
    }
}
