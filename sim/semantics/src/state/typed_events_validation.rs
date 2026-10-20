//! Cohesive child mechanism extracted from the parent module.

use super::*;

/// Version 15 adds a required shared DRAM service state, preserving its exact
/// fractional-cycle frontier and distinguishing carded bandwidth from the
/// latency-only ablation. Version 14 added each core's cumulative local
/// completion frontier (explicit null means no released CTA barrier).
/// It keeps the lossless payloads from
/// version 3, widens dynamic requester sector indices to u64, and records
/// hidden logical sequence spans and the explicit serialized phase used by
/// batched queue events. It also preserves the shared-L2 allocation
/// generation associated with each outstanding response, so a delayed
/// response from an evicted same-line allocation cannot be joined to a newer
/// allocation. Older version-2 payloads are not structurally
/// compatible because they carried only a batch count and response kind. It
/// also carries the identity and cohort response mapping for each live bypass
/// asynchronous-copy observation; a count-only projection cannot resume an
/// `AsyncCopy` response without guessing its producer.
pub const TYPED_STATE_SCHEMA_VERSION: u32 = 15;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedMemoryOperation {
    Read,
    Store,
    Atomic,
    Reduction,
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
pub struct L2SetState {
    pub slice: u32,
    pub set: u64,
    pub ways: Vec<L2WayState>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2RequestKey {
    pub sm_id: u32,
    /// Dynamic response warp identity. This remains distinct from
    /// `warp_slot`, which can be reused after a warp retires while its
    /// response is still outstanding.
    pub response_warp: u64,
    pub warp_slot: i32,
    pub instruction: u64,
    /// Dynamic response indices originate in the host `usize` domain. Keep
    /// the typed boundary lossless instead of narrowing a large trace to u32.
    pub sector_index: u64,
    pub response_kind: TypedResponseKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct L2RequestOwner {
    pub request: L2RequestKey,
    pub line: CanonicalLineKey,
    pub sector_mask: u8,
    /// Shared-L2 allocation generation that priced this response. This is a
    /// different identity domain from the L1 fill generation carried by the
    /// response kind. `None` means the response did not traverse a typed L2
    /// allocation and cannot be a pending-sector consumer.
    pub allocation_generation: Option<u64>,
    pub state: L2RequestState,
    pub due: Option<u64>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum L1StateError {
    #[error("typed L1 schema is unsupported")]
    SchemaVersion,
    #[error("typed L1 architecture identity must be non-empty")]
    ArchitectureIdentity,
    #[error("typed L1 geometry has a zero capacity or MSHR count")]
    InvalidGeometry,
    #[error("typed L1 hit latency must be lower than miss latency")]
    InvalidLatencyOrder,
    #[error("typed L1 resident lines exceed the declared capacity")]
    LineCapacityExceeded,
    #[error("typed L1 line {line} is both resident and in-flight")]
    ResidentLineFillConflict { line: i64 },
    #[error("typed L1 in-flight fills exceed the declared MSHR count")]
    MshrCountExceeded,
    #[error("typed L1 line order is not canonical")]
    UnsortedLineOrder,
    #[error("typed L1 contains duplicate stamps")]
    DuplicateStamp,
    #[error("typed L1 stamp map is not an inverse of line order")]
    StampIdentityMismatch,
    #[error("typed L1 stamp map is not canonical")]
    UnsortedStampOf,
    #[error("typed L1 stamp maps have different cardinalities")]
    StampCardinalityMismatch,
    #[error("typed L1 sectors are not canonical")]
    UnsortedSectors,
    #[error("typed L1 sector mask is invalid")]
    InvalidSectorMask,
    #[error("typed L1 sector state has no resident line or line allocation for line {line}")]
    SectorLineWithoutAllocation { line: i64 },
    #[error(
        "typed L1 generation {generation} has no resident line or line allocation for line {line}"
    )]
    GenerationWithoutAllocation { line: i64, generation: u32 },
    #[error("typed L1 contains duplicate line fills")]
    DuplicateLineFill,
    #[error("typed L1 line fills are not canonical")]
    UnsortedLineFills,
    #[error("typed L1 fill sector is outside the four-sector line")]
    InvalidFillSector,
    #[error("typed L1 contains duplicate sector fills")]
    DuplicateSectorFill,
    #[error("typed L1 sector fills are not canonical")]
    UnsortedSectorFills,
    #[error("typed L1 line generations are not canonical")]
    UnsortedGenerations,
    #[error("typed L1 contains duplicate line generations")]
    DuplicateGeneration,
    #[error("typed L1 reuses one live allocation generation across lines")]
    DuplicateGenerationValue,
    #[error("typed L1 line fill generation disagrees with its live line generation")]
    LineFillGenerationMismatch,
    #[error("typed L1 line fill sequence disagrees with its allocation generation")]
    LineFillSequenceMismatch,
    #[error("typed L1 next stamp is below a live stamp")]
    NextStampMismatch,
    #[error("typed L1 fill sequence is below a live fill")]
    FillSequenceMismatch,
    #[error("typed L1 reuses one fill sequence across live fills")]
    DuplicateFillSequence,
    #[error("typed L1 fill timing discipline disagrees with live fill due states")]
    FillTimingMismatch,
    #[error("typed L1 live read allocation has no fill timing discipline")]
    FillTimingMissing,
    #[error("typed L1 minimum fill due time disagrees with live fills")]
    MinimumFillDueMismatch,
    #[error("typed L1 miss counter {misses} disagrees with components {components}")]
    MissCompositionMismatch { misses: u64, components: u64 },
    #[error("typed L1 pending lookup hits exceed merges")]
    PendingLookupHitsExceedMerges,
    #[error("typed L1 atomic bypass sectors exceed bypass sectors")]
    BypassAtomsExceedBypass,
    #[error("typed L1 store hits exceed store sectors")]
    StoreHitsExceedStores,
    #[error("typed L1 reduction hits exceed reduction sectors")]
    ReductionHitsExceedReductions,
    #[error("typed L1 counter arithmetic overflowed")]
    CounterOverflow,
    #[error("typed L1 lookup batch starts after the fill sequence frontier")]
    LookupBatchAheadOfFillSequence,
    #[error("typed L1 timed fill due at {due} is before core cycle {cycle}")]
    FillDueInPast { due: u64, cycle: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AsyncCopyStateError {
    #[error("typed async-copy committed groups are not strictly ordered")]
    UnsortedGroups,
    #[error("typed async-copy committed groups are not ordered by commit time")]
    CommittedTimeOrder,
    #[error("typed async-copy committed time {committed_at} is after core cycle {cycle}")]
    CommittedTimeAfterCoreCycle { committed_at: u64, cycle: u64 },
    #[error("typed async-copy open group is older than a committed group")]
    OpenGroupOrder,
    #[error("typed async-copy group timestamps are not ordered")]
    TimestampOrder,
    #[error("typed async-copy destination service state is inconsistent")]
    DestinationService,
    #[error("typed async-copy destination service was queued before all sources completed")]
    DestinationPendingSources,
    #[error("typed async-copy next sequence does not follow its live groups")]
    NextSequenceMismatch,
    #[error(
        "typed async-copy bypass observation count {declared} disagrees with captured {captured}"
    )]
    BypassObservationCountMismatch { declared: u64, captured: u64 },
    #[error("typed async-copy bypass observation belongs to missing group {group}")]
    BypassObservationGroupMissing { group: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} is marked complete without cohort detail"
    )]
    BypassObservationDetailMissing { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} exceeds its group {field} bound: group={group_value}, observation={observation_value}"
    )]
    BypassObservationGroupBound {
        group: u64,
        instruction: u64,
        field: &'static str,
        group_value: u64,
        observation_value: u64,
    },
    #[error(
        "typed async-copy bypass observations are not in canonical order at group {group}, instruction {instruction}"
    )]
    BypassObservationOrder { group: u64, instruction: u64 },
    #[error("typed async-copy bypass observation {group}/{instruction} has zero delivery work")]
    BypassObservationZeroWork { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has inconsistent cohort count"
    )]
    BypassObservationCohortCount { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has missing sector membership"
    )]
    BypassObservationMembershipMissing { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has invalid sector membership"
    )]
    BypassObservationMembershipInvalid { group: u64, instruction: u64 },
    #[error("typed async-copy bypass observation {group}/{instruction} membership overflows")]
    BypassObservationMembershipOverflow { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has inconsistent pending sources"
    )]
    BypassObservationPendingSources { group: u64, instruction: u64 },
    #[error("typed async-copy bypass observation {group}/{instruction} sector span overflows")]
    BypassObservationSectorOverflow { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has inconsistent cohort pending state"
    )]
    BypassObservationCohortPending { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} has inconsistent response timestamps"
    )]
    BypassObservationResponseTimes { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} aggregate arithmetic overflows"
    )]
    BypassObservationAggregateOverflow { group: u64, instruction: u64 },
    #[error(
        "typed async-copy bypass observation {group}/{instruction} aggregate {field} is inconsistent"
    )]
    BypassObservationAggregateMismatch {
        group: u64,
        instruction: u64,
        field: &'static str,
        expected: u64,
        got: u64,
    },
}

pub(crate) fn expected_l2_placement(
    line_id: i64,
    slice_hash: &str,
    num_slices: u32,
    sets_per_slice: u64,
) -> Result<(u32, u64), L2StateError> {
    if num_slices == 0 || sets_per_slice == 0 {
        return Err(L2StateError::InvalidGeometry);
    }
    let slices = i128::from(num_slices);
    let sets = i128::from(sets_per_slice);
    let slice = match slice_hash {
        "line_mod" => i128::from(line_id).rem_euclid(slices),
        "sector_mod" => (i128::from(line_id).rem_euclid(slices) * 4).rem_euclid(slices),
        "xor_folded" => i128::from(line_id ^ (line_id >> 5) ^ (line_id >> 11)).rem_euclid(slices),
        "knuth_mult" => {
            i128::from(line_id.wrapping_mul(2_654_435_761) & 0xffff_ffff).rem_euclid(slices)
        }
        "shift_mod" => i128::from(line_id >> 1).rem_euclid(slices),
        "bit_reverse" => i128::from((line_id as u16).reverse_bits()).rem_euclid(slices),
        _ => return Err(L2StateError::UnknownSliceHash),
    };
    let set = i128::from(line_id).div_euclid(slices).rem_euclid(sets);
    Ok((
        u32::try_from(slice).map_err(|_| L2StateError::InvalidGeometry)?,
        u64::try_from(set).map_err(|_| L2StateError::InvalidGeometry)?,
    ))
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
    pub domain_clocked_cycles: Vec<u64>,
    pub domain_gated_cycles: Vec<u64>,
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
    #[error("power clock domain name must be non-empty")]
    InvalidClockDomain,
    #[error("power domain names must be non-empty and unique")]
    DomainIdentity,
    #[error("fixed power frequency must be non-zero when provided")]
    InvalidFrequency,
    #[error("power sample schedule is inconsistent with its period")]
    SampleSchedule,
    #[error("power samples are not strictly ordered or activity decreased")]
    UnsortedSamples,
    #[error("power sample domain vectors do not match the domain count")]
    SampleDomainCount,
    #[error("power sample activity total disagrees with its domain integrals")]
    SampleActivityMismatch,
    #[error("power sample is ahead of the corresponding domain integral")]
    SampleAheadOfDomain,
    #[error("power domain integrals are ahead of their elapsed cycle")]
    IntegralAheadOfClock,
    #[error("power domains do not share one boundary timestamp")]
    DomainTimeMismatch,
    #[error("power sample integrals are ahead of their sample cycle")]
    SampleIntegralAheadOfCycle,
    #[error("power sample time is ahead of the corresponding domain clock")]
    SampleAheadOfDomainTime,
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
}

impl TypedObservationContract {
    pub fn cycle_exact_memory() -> Self {
        Self {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            kind: TypedContractKind::CycleExactMemoryV2,
            address_domain: AddressDomain::VirtualAbsolute,
            power: PowerModelStatus::Omitted,
        }
    }
}

impl TypedObservationContract {
    pub fn cycle_exact_power() -> Self {
        Self {
            schema_version: TYPED_STATE_SCHEMA_VERSION,
            kind: TypedContractKind::CycleExactPowerV2,
            address_domain: AddressDomain::VirtualAbsolute,
            power: PowerModelStatus::ActivityOnly,
        }
    }
}

impl TypedObservationContract {
    pub fn validate(
        &self,
        l2: &L2BoundaryState,
        power: &PowerBoundaryState,
        cores: &[TypedCoreBoundaryState],
    ) -> Completeness {
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Completeness::Refused {
                reason: "typed_contract_schema_version".to_owned(),
            };
        }
        match self.kind {
            TypedContractKind::CycleExactMemoryV2 if self.power != PowerModelStatus::Omitted => {
                return Completeness::Refused {
                    reason: "memory_contract_power_mismatch".to_owned(),
                };
            }
            TypedContractKind::CycleExactPowerV2
                if self.power != PowerModelStatus::ActivityOnly =>
            {
                return Completeness::Refused {
                    reason: "power_contract_status_mismatch".to_owned(),
                };
            }
            _ => {}
        }
        if let Err(error) = l2.validate() {
            return Completeness::Refused {
                reason: error.to_string(),
            };
        }
        if let Err(error) = power.validate() {
            return Completeness::Refused {
                reason: error.to_string(),
            };
        }
        let mut core_ids = BTreeSet::new();
        let mut previous_core_id = None;
        for core in cores {
            if let Err(error) = core.validate() {
                return Completeness::Refused {
                    reason: error.to_string(),
                };
            }
            if !core_ids.insert(core.sm_id) {
                return Completeness::Refused {
                    reason: "duplicate_typed_core_sm_identity".to_owned(),
                };
            }
            if previous_core_id.is_some_and(|previous| core.sm_id <= previous) {
                return Completeness::Refused {
                    reason: "typed_core_identity_order".to_owned(),
                };
            }
            previous_core_id = Some(core.sm_id);
        }
        // The current typed core, queue, and causal-frontier contracts model
        // the four-SMSP SM86 topology. Keep diagnostic projections
        // architecture-neutral, but never let an SM120 or unknown L2 card
        // identity borrow this exactness claim until its own typed schema and
        // transition rules exist.
        if !matches!(self.kind, TypedContractKind::BoundaryDiagnosticV2)
            && l2.architecture != "sm_86"
        {
            return Completeness::Refused {
                reason: "sm86_exact_contract_requires_sm_86".to_owned(),
            };
        }
        if !l2.ownership_complete {
            return Completeness::Incomplete {
                unknown: l2.unknown_state.clone(),
            };
        }
        if !matches!(self.kind, TypedContractKind::BoundaryDiagnosticV2) {
            let mut unknown = cores
                .iter()
                .flat_map(|core| core.unprojected_timing_state.iter().cloned())
                .collect::<BTreeSet<_>>();
            if cores.iter().any(|core| {
                core.warp_states
                    .iter()
                    .any(|warp| warp.barrier_pending.iter().any(|&pending| pending != 0))
            }) {
                // The response-owner projection has no dependency-barrier
                // destination. Do not trust a producer-supplied marker list
                // to hide this live transition state.
                unknown.insert("dependency_barrier_ownership".to_owned());
            }
            if !unknown.is_empty() {
                return Completeness::Incomplete {
                    unknown: unknown.into_iter().collect(),
                };
            }
        }
        // The shared projection and each SM projection duplicate live
        // response ownership at different scopes. A complete boundary must
        // carry the same canonical owner records in both places; otherwise a
        // response can be present in L2 while absent from the core (or the
        // reverse), and the next transition cannot be reconstructed exactly.
        let mut shared_owners = l2.request_owners.clone();
        let mut core_owners = cores
            .iter()
            .flat_map(|core| core.response_owners.iter().cloned())
            .collect::<Vec<_>>();
        let owner_order = |left: &L2RequestOwner, right: &L2RequestOwner| {
            (
                &left.request,
                &left.line,
                left.sector_mask,
                left.allocation_generation,
                &left.state,
                left.due,
            )
                .cmp(&(
                    &right.request,
                    &right.line,
                    right.sector_mask,
                    right.allocation_generation,
                    &right.state,
                    right.due,
                ))
        };
        shared_owners.sort_by(owner_order);
        core_owners.sort_by(owner_order);
        if shared_owners != core_owners {
            return Completeness::Incomplete {
                unknown: vec!["l2_core_response_owner_join".to_owned()],
            };
        }
        // The diagnostic contract may intentionally carry only a shared-L2
        // projection, but cycle-exact contracts promise per-SM issue,
        // frontier, and resource state. An empty roster therefore cannot be
        // certified as an exact boundary even when the shared cache and
        // response-owner join are otherwise complete.
        if !matches!(self.kind, TypedContractKind::BoundaryDiagnosticV2) && cores.is_empty() {
            return Completeness::Incomplete {
                unknown: vec!["core_state_missing".to_owned()],
            };
        }
        for state in cores
            .iter()
            .flat_map(|core| core.async_copy.iter().flatten())
        {
            if let Err(error) = state.validate() {
                return Completeness::Refused {
                    reason: error.to_string(),
                };
            }
        }
        if cores.iter().any(|core| {
            core.async_copy.iter().flatten().any(|state| {
                !state.bypass_observations_complete
                    || state.bypass_observation_count
                        != u64::try_from(state.bypass_observations.len()).unwrap_or(u64::MAX)
            })
        }) {
            return Completeness::Incomplete {
                unknown: vec!["async_bypass_observation_state".to_owned()],
            };
        }
        if cores
            .iter()
            .any(|core| core.l1.as_ref().is_some_and(|l1| l1.write_observer_present))
        {
            return Completeness::Incomplete {
                unknown: vec!["l1_write_lookup_observer_state".to_owned()],
            };
        }
        if l2.request_owners.iter().any(|owner| {
            matches!(
                owner.request.response_kind,
                TypedResponseKind::Warp | TypedResponseKind::AsyncCopy
            ) && matches!(owner.line, CanonicalLineKey::Unknown { .. })
        }) {
            return Completeness::Incomplete {
                unknown: vec!["opaque_response_allocation_identity".to_owned()],
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
        if self.kind != TypedContractKind::BoundaryDiagnosticV2
            && l2.address_domain != self.address_domain
        {
            return Completeness::Refused {
                reason: "address_domain_mismatch".to_owned(),
            };
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

pub(crate) fn validate_fill_identity(fill: &TypedFillIdentity) -> Result<(), BoundaryStateError> {
    if fill.sector_mask == 0 || fill.sector_mask & !0x0f != 0 {
        return Err(BoundaryStateError::FillIdentityMask {
            sector_mask: fill.sector_mask,
        });
    }
    Ok(())
}

pub(crate) fn validate_response_identity(
    response: &TypedResponseIdentity,
) -> Result<(), BoundaryStateError> {
    if matches!(response.response_kind, TypedResponseKind::None) {
        return Err(BoundaryStateError::ResponseIdentityKindNone);
    }
    Ok(())
}

impl TypedMemoryRequest {
    pub(crate) fn validate_contract(&self) -> Result<(), BoundaryStateError> {
        let own_sector_bit = 1_u8 << (self.sector_id & 3) as u8;
        if self.sector_mask == 0
            || self.sector_mask & !0x0f != 0
            || self.sector_mask & own_sector_bit == 0
            || self.local_sector_mask & !self.sector_mask != 0
            || (self.local_sector_mask & own_sector_bit != 0)
                != matches!(self.origin, TypedMemoryOrigin::Local)
        {
            return Err(BoundaryStateError::MemoryRequestMask {
                sector_id: self.sector_id,
                sector_mask: self.sector_mask,
                local_sector_mask: self.local_sector_mask,
            });
        }
        if (self.constant_load || self.constant_deep)
            && (!self.constant_load
                || !matches!(self.source, TypedL2Source::L1Fill)
                || !matches!(self.operation, TypedMemoryOperation::Read))
        {
            return Err(BoundaryStateError::MemoryRequestConstantContract {
                request_source: self.source,
                operation: self.operation,
                constant_load: self.constant_load,
                constant_deep: self.constant_deep,
            });
        }
        let reply = self.reply.as_ref().map(|reply| reply.response_kind);
        if matches!(reply, Some(TypedResponseKind::L1SectorFill))
            && self.sector_mask.count_ones() != 1
        {
            return Err(BoundaryStateError::MemoryRequestSectorReplyMismatch {
                response_kind: TypedResponseKind::L1SectorFill,
                sector_mask: self.sector_mask,
            });
        }
        let valid_contract = matches!(
            (self.source, self.operation, reply),
            (TypedL2Source::L1Fill, TypedMemoryOperation::Read, None)
                | (
                    TypedL2Source::L1Fill,
                    TypedMemoryOperation::Read,
                    Some(TypedResponseKind::L1Fill | TypedResponseKind::L1SectorFill),
                )
                | (
                    TypedL2Source::Bypass,
                    TypedMemoryOperation::Read,
                    None | Some(TypedResponseKind::Warp | TypedResponseKind::AsyncCopy),
                )
                | (
                    TypedL2Source::Untagged,
                    TypedMemoryOperation::Read,
                    None | Some(TypedResponseKind::Warp),
                )
                | (
                    TypedL2Source::Bypass,
                    TypedMemoryOperation::Atomic,
                    None | Some(TypedResponseKind::Warp),
                )
                | (
                    TypedL2Source::Store,
                    TypedMemoryOperation::Store | TypedMemoryOperation::Reduction,
                    None,
                )
        );
        if !valid_contract {
            return Err(BoundaryStateError::MemoryRequestContract {
                request_source: self.source,
                operation: self.operation,
                reply,
            });
        }
        if let Some(witness) = &self.admission_witness
            && (witness.subcore >= 4
                || witness.request_count == 0
                || witness.request_index >= witness.request_count
                || !matches!(
                    (self.source, self.origin, self.operation),
                    (
                        TypedL2Source::Store,
                        TypedMemoryOrigin::Global,
                        TypedMemoryOperation::Store | TypedMemoryOperation::Reduction
                    )
                ))
        {
            return Err(BoundaryStateError::MemoryRequestWitness {
                subcore: witness.subcore,
                request_index: witness.request_index,
                request_count: witness.request_count,
            });
        }
        Ok(())
    }
}

pub(crate) fn validate_batch_witnesses(
    requests: &[TypedMemoryRequest],
) -> Result<(), BoundaryStateError> {
    // The native observer carries one pending instruction across events, but
    // closes it as soon as request_count is reached.  The scheduler can then
    // place the next instruction's index-zero request in the same timestamped
    // batch.  Keep the active group's expected index so we accept those legal
    // group transitions while still rejecting interleaving and skips.
    let mut current: Option<(TypedAdmissionWitness, TypedMemoryOperation, u64, u16)> = None;
    let mut interrupted_incomplete_group = false;
    for request in requests {
        let Some(witness) = request.admission_witness else {
            // A witness sequence describes every request emitted by one
            // instruction. If it interrupts a group whose advertised count
            // has not been observed, retain that fact: otherwise a forged
            // batch could hide a gap between indices and restart at an
            // arbitrary witness index.
            if let Some((previous_witness, _, _, expected)) = current
                && expected != previous_witness.request_count
            {
                interrupted_incomplete_group = true;
            }
            continue;
        };
        if interrupted_incomplete_group {
            return Err(BoundaryStateError::MemoryRequestWitnessGroupMismatch);
        }
        let Some((previous_witness, previous_operation, previous_issue_time, expected)) = current
        else {
            let next = witness.request_index.checked_add(1).ok_or(
                BoundaryStateError::MemoryRequestWitnessSequence {
                    expected: u16::MAX,
                    found: witness.request_index,
                },
            )?;
            current = Some((witness, request.operation, request.issue_time, next));
            continue;
        };

        let same_group = previous_witness.warp == witness.warp
            && previous_witness.instruction == witness.instruction
            && previous_witness.subcore == witness.subcore
            && previous_witness.request_count == witness.request_count
            && previous_operation == request.operation
            && previous_issue_time == request.issue_time;
        if same_group {
            if witness.request_index != expected {
                return Err(BoundaryStateError::MemoryRequestWitnessSequence {
                    expected,
                    found: witness.request_index,
                });
            }
            let next = expected.checked_add(1).ok_or(
                BoundaryStateError::MemoryRequestWitnessSequence {
                    expected: u16::MAX,
                    found: witness.request_index,
                },
            )?;
            current = Some((witness, request.operation, request.issue_time, next));
            continue;
        }

        if expected != previous_witness.request_count {
            return Err(BoundaryStateError::MemoryRequestWitnessGroupMismatch);
        }
        if witness.request_index != 0 {
            return Err(BoundaryStateError::MemoryRequestWitnessSequence {
                expected: 0,
                found: witness.request_index,
            });
        }
        current = Some((witness, request.operation, request.issue_time, 1));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum QueueStateError {
    #[error(transparent)]
    Sequence(#[from] EventSequenceError),
    #[error("typed queue depth {depth} differs from pending event count {pending}")]
    DepthMismatch { depth: u64, pending: usize },
    #[error("typed queue contains duplicate sequence {sequence}")]
    DuplicateSequence { sequence: u64 },
    #[error("typed queue events are not in (time,sequence) order")]
    Unsorted,
    #[error("typed queue next-event time disagrees with its pending head")]
    NextTimeMismatch,
    #[error("event sequence {sequence} overlaps a reservation ending at {previous_end}")]
    OverlappingSpans { sequence: u64, previous_end: u64 },
    #[error("reservation ending at {end} reaches the next unused sequence {next_sequence}")]
    UnreservedSequence { end: u64, next_sequence: u64 },
}

impl TypedEventIdentity {
    pub fn sequence_end(&self) -> Result<u64, EventSequenceError> {
        self.payload.sequence_end(self.sequence, self.sequence_span)
    }
}

pub(crate) fn validate_memory_pipeline(
    state: &MemoryPipelineBoundaryState,
    cycle: u64,
) -> Result<(), CoreStateError> {
    if state.queue_entries_with_dispatch == 0
        || state.local_initiation_interval_cycles == 0
        || state.shared_accept_interval_cycles == 0
        || state.queue_release_times.len() != 4
        || state.local_completion_ready.len() != 4
    {
        return Err(CoreStateError::MemoryPipelineShape);
    }
    let mut live_entries = 0_u64;
    let mut maximum_live_entries = 0_u64;
    let mut maximum_release = None;
    for (subcore, queue) in state.queue_release_times.iter().enumerate() {
        if queue.iter().any(|release| *release < cycle) {
            return Err(CoreStateError::MemoryPipelineReleaseInPast);
        }
        if !queue.is_empty() && state.local_completion_ready[subcore] == 0 {
            return Err(CoreStateError::MemoryPipelineLocalFrontierMissing { subcore });
        }
        if queue.windows(2).any(|window| window[1] <= window[0]) {
            return Err(CoreStateError::MemoryPipelineOrder);
        }
        if queue.len() > state.queue_entries_with_dispatch as usize {
            return Err(CoreStateError::MemoryPipelineCapacity);
        }
        live_entries = live_entries
            .checked_add(u64::try_from(queue.len()).unwrap_or(u64::MAX))
            .ok_or(CoreStateError::CounterOverflow)?;
        maximum_live_entries =
            maximum_live_entries.max(u64::try_from(queue.len()).unwrap_or(u64::MAX));
        maximum_release = queue.iter().copied().max().max(maximum_release);
    }
    if state.instructions_admitted < live_entries {
        return Err(CoreStateError::MemoryPipelineAdmissionCount);
    }
    // Every native memory admission advances the selected SMSP's local
    // completion frontier. Once all queue entries drain, that historical
    // frontier remains nonzero; aggregate admissions with no live entries and
    // four zero local frontiers cannot be produced by the executable plan.
    if state.instructions_admitted != 0
        && live_entries == 0
        && state.local_completion_ready.iter().all(|&ready| ready == 0)
    {
        return Err(CoreStateError::MemoryPipelineHistory);
    }
    if state.maximum_queue_occupancy < maximum_live_entries
        || state.maximum_queue_occupancy > u64::from(state.queue_entries_with_dispatch)
        || (state.instructions_admitted != 0 && state.maximum_queue_occupancy == 0)
        || state.maximum_queue_occupancy > state.instructions_admitted
    {
        return Err(CoreStateError::MemoryPipelineOccupancy);
    }
    if state.maximum_operand_pipeline_delay_cycles > state.operand_pipeline_delay_cycles
        || state.maximum_shared_queue_delay_cycles > state.shared_queue_delay_cycles
    {
        return Err(CoreStateError::MemoryPipelineDelayMaximum);
    }
    if state.instructions_admitted == 0
        && (state.local_completion_ready.iter().any(|&ready| ready != 0)
            || state.terminal_shared_accept_cycle != 0
            || state.operand_pipeline_delay_cycles != 0
            || state.shared_queue_delay_cycles != 0
            || state.maximum_operand_pipeline_delay_cycles != 0
            || state.maximum_shared_queue_delay_cycles != 0)
    {
        return Err(CoreStateError::MemoryPipelineDiagnostics);
    }
    // This is an inequality bound, not a scheduled timestamp. Shared-port
    // contention can move the terminal acceptance past every local operand
    // completion, so this sum may exceed u64 even when all native frontiers
    // fit. Keep the comparison exact without rejecting that valid state.
    let local_frontier_limit = u128::from(state.terminal_shared_accept_cycle)
        + u128::from(state.local_initiation_interval_cycles);
    if state
        .local_completion_ready
        .iter()
        .any(|&ready| u128::from(ready) > local_frontier_limit)
    {
        return Err(CoreStateError::MemoryPipelineLocalFrontier);
    }
    if maximum_release.is_some_and(|release| state.terminal_shared_accept_cycle < release) {
        return Err(CoreStateError::MemoryPipelineTerminalCycle);
    }
    let expected_ready = if state.instructions_admitted == 0 {
        0
    } else {
        state
            .terminal_shared_accept_cycle
            .checked_add(state.shared_accept_interval_cycles)
            .ok_or(CoreStateError::CounterOverflow)?
    };
    if state.shared_accept_ready != expected_ready {
        return Err(CoreStateError::MemoryPipelineSharedFrontier);
    }
    if let Some(maximum_release) = maximum_release {
        let live_expected = maximum_release
            .checked_add(state.shared_accept_interval_cycles)
            .ok_or(CoreStateError::CounterOverflow)?;
        if state.shared_accept_ready < live_expected {
            return Err(CoreStateError::MemoryPipelineSharedFrontier);
        }
    }
    Ok(())
}

pub(crate) fn validate_block_barrier_state(
    block_index: usize,
    block: &BlockBoundaryState,
    warps: &[WarpBoundaryState],
    cycle: u64,
) -> Result<(), CoreStateError> {
    if block.barrier_generation == 0 && block.barrier_rearm_ready != 0 {
        return Err(CoreStateError::BlockBarrierRearmWithoutGeneration {
            block: block_index,
            rearm_ready: block.barrier_rearm_ready,
        });
    }
    let mut observed_waiting = 0_u64;
    let mut first_dispatch: Option<u64> = None;
    let mut first_arrival: Option<u64> = None;
    for warp in warps {
        if !warp.at_block_barrier {
            if warp.block_barrier_dispatch.is_some() || warp.block_barrier_arrival.is_some() {
                return Err(CoreStateError::BlockBarrierTimestampWithoutFlag {
                    block: block_index,
                    warp_slot: warp.warp_slot,
                });
            }
            continue;
        }
        if warp.done {
            return Err(CoreStateError::BlockBarrierDoneWarp {
                block: block_index,
                warp_slot: warp.warp_slot,
            });
        }
        let dispatch =
            warp.block_barrier_dispatch
                .ok_or(CoreStateError::BlockBarrierTimestampMissing {
                    block: block_index,
                    warp_slot: warp.warp_slot,
                })?;
        let arrival =
            warp.block_barrier_arrival
                .ok_or(CoreStateError::BlockBarrierTimestampMissing {
                    block: block_index,
                    warp_slot: warp.warp_slot,
                })?;
        if dispatch > cycle {
            return Err(CoreStateError::BlockBarrierDispatchAfterCycle {
                block: block_index,
                warp_slot: warp.warp_slot,
                dispatch,
                cycle,
            });
        }
        if arrival < dispatch {
            return Err(CoreStateError::BlockBarrierTimestampOrder {
                block: block_index,
                warp_slot: warp.warp_slot,
                dispatch,
                arrival,
            });
        }
        observed_waiting = observed_waiting
            .checked_add(1)
            .ok_or(CoreStateError::CounterOverflow)?;
        first_dispatch = Some(first_dispatch.map_or(dispatch, |current| current.min(dispatch)));
        first_arrival = Some(first_arrival.map_or(arrival, |current| current.min(arrival)));
    }
    if observed_waiting != block.waiting {
        return Err(CoreStateError::BlockBarrierWaitingCount {
            block: block_index,
            declared: block.waiting,
            observed: observed_waiting,
        });
    }
    if first_dispatch != block.barrier_first_dispatch {
        return Err(CoreStateError::BlockBarrierFirstDispatch {
            block: block_index,
            expected: first_dispatch,
            found: block.barrier_first_dispatch,
        });
    }
    if first_arrival != block.barrier_first_arrival {
        return Err(CoreStateError::BlockBarrierFirstArrival {
            block: block_index,
            expected: first_arrival,
            found: block.barrier_first_arrival,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BoundaryStateError {
    #[error("typed event boundary schema version is unsupported")]
    SchemaVersion,
    #[error("typed event phase {phase} is unsupported by the single-phase engine")]
    UnsupportedPhase { phase: u8 },
    #[error(
        "typed {payload} payload field {field} has timestamp {found}, expected event time {expected}"
    )]
    PayloadTimeMismatch {
        payload: &'static str,
        field: &'static str,
        expected: u64,
        found: u64,
    },
    #[error(
        "typed {payload} payload timestamp {earlier_field}={earlier} is after {later_field}={later}"
    )]
    PayloadTimestampOrder {
        payload: &'static str,
        earlier_field: &'static str,
        earlier: u64,
        later_field: &'static str,
        later: u64,
    },
    #[error(
        "typed memory request for sector {sector_id} has invalid masks sector={sector_mask:#06b}, local={local_sector_mask:#06b}"
    )]
    MemoryRequestMask {
        sector_id: i64,
        sector_mask: u8,
        local_sector_mask: u8,
    },
    #[error(
        "typed memory request has incoherent source/operation/reply contract: source={request_source:?}, operation={operation:?}, reply={reply:?}"
    )]
    MemoryRequestContract {
        request_source: TypedL2Source,
        operation: TypedMemoryOperation,
        reply: Option<TypedResponseKind>,
    },
    #[error(
        "typed memory request admission witness has subcore {subcore}, index {request_index}, count {request_count}"
    )]
    MemoryRequestWitness {
        subcore: u8,
        request_index: u16,
        request_count: u16,
    },
    #[error("typed batched memory requests change admission witness group identity")]
    MemoryRequestWitnessGroupMismatch,
    #[error("typed batched memory request witness index expected {expected}, found {found}")]
    MemoryRequestWitnessSequence { expected: u16, found: u16 },
    #[error(
        "typed memory request constant markers are incoherent: source={request_source:?}, operation={operation:?}, constant_load={constant_load}, constant_deep={constant_deep}"
    )]
    MemoryRequestConstantContract {
        request_source: TypedL2Source,
        operation: TypedMemoryOperation,
        constant_load: bool,
        constant_deep: bool,
    },
    #[error(
        "typed memory request sector mask {sector_mask:#06b} is incompatible with {response_kind:?}"
    )]
    MemoryRequestSectorReplyMismatch {
        response_kind: TypedResponseKind,
        sector_mask: u8,
    },
    #[error("typed L2 fill identity has invalid sector mask {sector_mask:#06b}")]
    FillIdentityMask { sector_mask: u8 },
    #[error("typed response identity uses response kind none")]
    ResponseIdentityKindNone,
    #[error("typed memory response event is missing its returning reply identity")]
    MemoryResponseReplyMissing,
    #[error(
        "typed memory response fill is paired with incompatible response kind {response_kind:?}"
    )]
    MemoryResponseFillReplyMismatch { response_kind: TypedResponseKind },
    #[error("typed memory write acknowledgement has unsupported operation {operation:?}")]
    MemoryWriteAckOperation { operation: TypedMemoryOperation },
    #[error(
        "typed memory write acknowledgement witness has subcore {subcore}, index {request_index}, count {request_count}"
    )]
    MemoryWriteAckWitness {
        subcore: u8,
        request_index: u16,
        request_count: u16,
    },
    #[error(transparent)]
    Sequence(#[from] EventSequenceError),
    #[error("pending sequence {sequence} overlaps the current event's reservation")]
    CurrentEventOverlap { sequence: u64 },
    #[error("pending event ({time},{sequence}) is at or before the current event")]
    PendingEventOrder { time: u64, sequence: u64 },
    #[error("typed event target {target} is outside {cores} core snapshots")]
    EventTargetOutOfRange { target: u32, cores: usize },
    #[error("typed core SM identity {found} is not after previous identity {previous}")]
    CoreIdentityOrder { previous: u32, found: u32 },
    #[error("typed power domain time does not match event time {event_time}")]
    PowerTimeMismatch { event_time: u64 },
    #[error(
        "typed response owner for SM {sm_id} is due at {due}, before boundary event time {event_time}"
    )]
    ResponseDueBeforeBoundary {
        sm_id: u32,
        due: u64,
        event_time: u64,
    },
    #[error(
        "typed response owner/queue join is inconsistent for SM {sm_id} warp {response_warp} instruction {instruction} sector {sector_index} kind {response_kind:?} due {due}"
    )]
    ResponseOwnerQueueJoin {
        sm_id: u32,
        response_warp: u64,
        instruction: u64,
        sector_index: u64,
        response_kind: TypedResponseKind,
        due: u64,
    },
    #[error(
        "typed L2 pending-fill/queue join is inconsistent for line {line_id} generation {generation} ready {ready}"
    )]
    L2FillQueueJoin {
        line_id: i64,
        generation: u64,
        ready: u64,
    },
    #[error(
        "typed pending L2 sector {line:?}/{generation}/{sector} is ready at {ready_at}, before boundary event time {event_time}"
    )]
    L2PendingReadyBeforeBoundary {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
        ready_at: u64,
        event_time: u64,
    },
    #[error(transparent)]
    Queue(#[from] QueueStateError),
    #[error(transparent)]
    Core(#[from] CoreStateError),
    #[error(transparent)]
    L2(#[from] L2StateError),
    #[error(transparent)]
    Dram(#[from] DramStateError),
    #[error(transparent)]
    Power(#[from] PowerStateError),
    #[error("typed boundary completeness declaration disagrees with its contract and state")]
    CompletenessMismatch {
        declared: Completeness,
        derived: Completeness,
    },
}
