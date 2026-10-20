// Lexical implementation section 2/3; included by the parent facade.
fn expected_l2_placement(
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

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum L2StateError {
    #[error("typed L2 requester uses the invalid u32::MAX SM identity sentinel")]
    InvalidSmIdentity,
    #[error("typed L2 schema {found} is unsupported; expected {expected}")]
    SchemaVersion { found: u32, expected: u32 },
    #[error("typed L2 geometry contains a zero dimension")]
    InvalidGeometry,
    #[error("typed L2 access discipline disagrees with resident allocation generations")]
    DisciplineGenerationMismatch,
    #[error("typed L2 access discipline is missing for an active complete state")]
    DisciplineMissing,
    #[error("typed L2 clock is negative")]
    NegativeClock,
    #[error("typed L2 line identity disagrees with the declared address domain")]
    AddressDomainMismatch,
    #[error("typed L2 root and geometry architecture identities disagree")]
    ArchitectureMismatch,
    #[error("typed L2 geometry uses an unknown replacement policy")]
    UnknownReplacement,
    #[error("typed L2 capacity {found} disagrees with geometry product {expected}")]
    GeometryMismatch { expected: u128, found: u64 },
    #[error("typed L2 lookup hits exceed lookup requests")]
    LookupHitsExceedRequests,
    #[error("typed L2 observer lookup hits exceed observer lookup requests")]
    ObserverLookupHitsExceedRequests,
    #[error("typed L2 miss counter {misses} disagrees with components {components}")]
    MissCompositionMismatch { misses: u64, components: u64 },
    #[error("typed L2 write-tag hits exceed hits")]
    WriteTagHitsExceedHits,
    #[error("typed L2 counter arithmetic overflowed")]
    CounterOverflow,
    #[error("typed L2 set ({slice},{set}) is outside declared geometry")]
    SetOutOfRange { slice: u32, set: u64 },
    #[error("typed L2 set {set} contains too many ways")]
    SetOverfull { set: u64 },
    #[error("typed L2 sets are not in canonical order")]
    UnsortedSets,
    #[error("typed L2 contains duplicate set ({slice},{set})")]
    DuplicateSet { slice: u32, set: u64 },
    #[error("typed L2 set {set} contains duplicate way lines")]
    DuplicateWayLine { set: u64 },
    #[error("typed L2 way {line:?} has neither valid nor pending sectors")]
    EmptyWay { line: CanonicalLineKey },
    #[error("typed L2 {replacement} way {line:?} has invalid recency {recency} at clock {clock}")]
    InvalidRecency {
        line: CanonicalLineKey,
        recency: i64,
        clock: i64,
        replacement: &'static str,
    },
    #[error("typed L2 set ({slice},{set}) contains duplicate recency {recency}")]
    DuplicateRecency { slice: u32, set: u64, recency: i64 },
    #[error("typed L2 contains duplicate resident line/generation identities")]
    DuplicateWayIdentity,
    #[error("typed L2 way contains a sector mask outside the four-sector line")]
    InvalidSectorMask,
    #[error("complete typed L2 way {line:?} has no allocation generation")]
    WayGenerationMissing { line: CanonicalLineKey },
    #[error("typed L2 geometry uses an unknown slice hash")]
    UnknownSliceHash,
    #[error(
        "typed L2 line {line:?} is stored in slice/set ({found_slice},{found_set}), expected ({expected_slice},{expected_set})"
    )]
    LinePlacementMismatch {
        line: CanonicalLineKey,
        found_slice: u32,
        found_set: u64,
        expected_slice: u32,
        expected_set: u64,
    },
    #[error("typed L2 contains duplicate allocation generations")]
    DuplicateAllocation,
    #[error("typed L2 reuses allocation generation {generation} across live lines")]
    DuplicateGeneration { generation: u64 },
    #[error("typed L2 next-generation frontier is not above live generations")]
    GenerationFrontierMismatch,
    #[error("typed L2 allocations are not in canonical line/generation order")]
    UnsortedAllocations,
    #[error("typed L2 allocation contains duplicate or out-of-range sectors")]
    DuplicateAllocationSector,
    #[error("typed L2 allocation sectors are not in ascending order")]
    UnsortedAllocationSectors,
    #[error("typed L2 contains duplicate requester ownership")]
    DuplicateRequestOwner,
    #[error("typed L2 response identity is duplicated")]
    DuplicateResponseIdentity,
    #[error("typed L2 requester ownership is not in canonical order")]
    UnsortedRequestOwners,
    #[error("typed L2 request owner has a negative warp slot {warp_slot}")]
    NegativeRequestWarpSlot { warp_slot: i32 },
    #[error("typed L2 L1 response owner has an invalid sector mask")]
    InvalidRequestOwnerSectorMask,
    #[error("typed L2 L1 response owner uses a different address domain")]
    RequestOwnerAddressDomainMismatch,
    #[error("typed L2 L1 response owner is missing its allocation generation")]
    RequestOwnerGenerationMissing,
    #[error("typed L2 L1 response owner has no matching allocation")]
    RequestOwnerAllocationMissing,
    #[error("typed L2 L1 response owner references sectors outside its allocation")]
    RequestOwnerSectorMismatch,
    #[error("typed L2 opaque response owner carries cache allocation identity")]
    OpaqueRequestOwnerIdentity,
    #[error("typed L2 response owner has no supported response kind")]
    InvalidRequestOwnerKind,
    #[error("typed L2 allocation and resident-way identities disagree")]
    AllocationWayMismatch,
    #[error("typed L2 allocation has no matching resident way")]
    AllocationWithoutWay,
    #[error("typed L2 allocation sector statuses disagree with resident masks")]
    AllocationMaskMismatch {
        line: CanonicalLineKey,
        generation: u64,
    },
    #[error("typed L2 non-pending sector carries requester ownership")]
    NonPendingOwnership {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 non-pending sector carries a ready time without an outstanding fill owner")]
    ReadyTimeOnNonPending {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
        ready_at: u64,
    },
    #[error("typed L2 sector consumer is duplicated")]
    DuplicateSectorConsumer,
    #[error("complete typed L2 pending sector references a non-outstanding request owner")]
    PendingConsumerNotOutstanding {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 sector consumers are not in canonical request order")]
    UnsortedSectorConsumers,
    #[error("typed L2 sector ownership references an unknown request")]
    OwnershipReferenceMissing,
    #[error(
        "typed L2 sector ownership references a different line, sector, or allocation generation"
    )]
    OwnershipReferenceMismatch {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 producer is not one of the sector consumers")]
    ProducerNotConsumer,
    #[error("complete typed L2 ownership has a pending sector without a producer")]
    CompletePendingWithoutProducer {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("complete typed L2 ownership has a pending sector without a ready time")]
    CompletePendingWithoutReadyTime {
        line: CanonicalLineKey,
        generation: u64,
        sector: u8,
    },
    #[error("typed L2 way has pending and valid sectors in the same bit")]
    PendingAndValidOverlap,
    #[error("typed L2 way has dirty sectors that are not valid")]
    DirtyWithoutValid,
    #[error("typed L2 state is marked complete while unknown fields remain")]
    CompleteWithUnknownState,
    #[error("typed L2 state is incomplete without an unknown-state reason")]
    IncompleteWithoutUnknown,
    #[error("typed L2 unknown-state reasons are not in canonical order")]
    UnknownStateOrder,
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
            sample_period,
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
        if self.schema_version != TYPED_STATE_SCHEMA_VERSION {
            return Err(PowerStateError::SchemaVersion);
        }
        let mut next = self.clone();
        next.record_event_inner(event)?;
        *self = next;
        Ok(())
    }

    /// Apply the narrow scalar-macro power projection used by the reference
    /// event loop.  The macro is accepted only for a closed issue-only region;
    /// this method records its declared clock/event contribution before the
    /// continuation event is observed.  It is not a hardware energy model and
    /// does not manufacture completion tokens.
    pub fn record_scalar_macro_completion(
        &mut self,
        start_cycle: u64,
        end_cycle: u64,
    ) -> Result<(), PowerStateError> {
        if end_cycle < start_cycle {
            return Err(PowerStateError::TimeTravel);
        }
        let mut next = self.clone();
        next.advance_to(start_cycle)?;
        for domain in &mut next.domains {
            if end_cycle < domain.last_cycle {
                return Err(PowerStateError::TimeTravel);
            }
            domain.status = PowerDomainStatus::Clocked;
            domain.queue_events = domain
                .queue_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
            domain.sm_wake_events = domain
                .sm_wake_events
                .checked_add(1)
                .ok_or(PowerStateError::CounterOverflow)?;
            domain.last_cycle = end_cycle;
        }
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
        if event.contains("l2")
            || event.contains("memory")
            || event == "mem_response"
            || event == "atomic_completion"
        {
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
        if event == "memory_response"
            || event == "l2_fill"
            || event == "atomic_completion"
            || event == "memory_write_ack"
        {
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
                domain_clocked_cycles: self
                    .domains
                    .iter()
                    .map(|domain| domain.clocked_cycles)
                    .collect(),
                domain_gated_cycles: self
                    .domains
                    .iter()
                    .map(|domain| domain.gated_cycles)
                    .collect(),
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
        if self.domains.is_empty() {
            return Err(PowerStateError::NoDomains);
        }
        if self.clock.clock_domain.is_empty() {
            return Err(PowerStateError::InvalidClockDomain);
        }
        if self.clock.fixed_frequency_hz == Some(0) {
            return Err(PowerStateError::InvalidFrequency);
        }
        if self.clock.dvfs_modeled && self.status != PowerModelStatus::AnalyticCoefficients {
            return Err(PowerStateError::UnknownDvfs);
        }
        if self.status != PowerModelStatus::Unknown && !self.unknown_state.is_empty() {
            return Err(PowerStateError::UnknownState);
        }
        let mut domain_names = BTreeSet::new();
        let mut domain_time = None;
        for domain in &self.domains {
            if domain.domain.is_empty() || !domain_names.insert(&domain.domain) {
                return Err(PowerStateError::DomainIdentity);
            }
            let integrated = domain
                .clocked_cycles
                .checked_add(domain.gated_cycles)
                .ok_or(PowerStateError::CounterOverflow)?;
            if integrated > domain.last_cycle {
                return Err(PowerStateError::IntegralAheadOfClock);
            }
            if let Some(previous) = domain_time
                && previous != domain.last_cycle
            {
                return Err(PowerStateError::DomainTimeMismatch);
            }
            domain_time = Some(domain.last_cycle);
        }
        match (self.sample_period, self.next_sample) {
            (Some(0), _) => return Err(PowerStateError::InvalidSamplePeriod),
            (None, Some(_)) => return Err(PowerStateError::SampleSchedule),
            (Some(_), None) => return Err(PowerStateError::SampleSchedule),
            _ => {}
        }
        if self.sample_period.is_none() && !self.samples.is_empty() {
            return Err(PowerStateError::SampleSchedule);
        }
        let mut previous_sample: Option<&PowerSampleState> = None;
        for sample in &self.samples {
            if sample.domain_clocked_cycles.len() != self.domains.len()
                || sample.domain_gated_cycles.len() != self.domains.len()
            {
                return Err(PowerStateError::SampleDomainCount);
            }
            let calculated_activity =
                sample
                    .domain_clocked_cycles
                    .iter()
                    .try_fold(0_u64, |sum, value| {
                        sum.checked_add(*value)
                            .ok_or(PowerStateError::CounterOverflow)
                    })?;
            if calculated_activity != sample.activity_total {
                return Err(PowerStateError::SampleActivityMismatch);
            }
            if sample
                .domain_clocked_cycles
                .iter()
                .zip(&self.domains)
                .any(|(sampled, domain)| *sampled > domain.clocked_cycles)
                || sample
                    .domain_gated_cycles
                    .iter()
                    .zip(&self.domains)
                    .any(|(sampled, domain)| *sampled > domain.gated_cycles)
            {
                return Err(PowerStateError::SampleAheadOfDomain);
            }
            for (&clocked, &gated) in sample
                .domain_clocked_cycles
                .iter()
                .zip(&sample.domain_gated_cycles)
            {
                if clocked
                    .checked_add(gated)
                    .ok_or(PowerStateError::CounterOverflow)?
                    > sample.cycle
                {
                    return Err(PowerStateError::SampleIntegralAheadOfCycle);
                }
            }
            if let Some(previous) = previous_sample
                && (sample.cycle <= previous.cycle
                    || sample.activity_total < previous.activity_total
                    || sample
                        .domain_clocked_cycles
                        .iter()
                        .zip(&previous.domain_clocked_cycles)
                        .any(|(current, previous)| current < previous)
                    || sample
                        .domain_gated_cycles
                        .iter()
                        .zip(&previous.domain_gated_cycles)
                        .any(|(current, previous)| current < previous))
            {
                return Err(PowerStateError::UnsortedSamples);
            }
            previous_sample = Some(sample);
        }
        if self.samples.iter().any(|sample| {
            self.domains
                .iter()
                .any(|domain| sample.cycle > domain.last_cycle)
        }) {
            return Err(PowerStateError::SampleAheadOfDomainTime);
        }
        if let Some(period) = self.sample_period {
            // `activity_only` starts the schedule at cycle `period` and
            // advances it by exactly one period for every emitted sample.
            // Keeping the complete cadence is part of the resumable state:
            // accepting a later `next_sample` would silently skip an
            // integration interval, while an off-phase sample would shift
            // every future boundary. Run this after per-sample integrity
            // checks so malformed sample payloads retain their precise error.
            let mut expected = period;
            for sample in &self.samples {
                if sample.cycle != expected {
                    return Err(PowerStateError::SampleSchedule);
                }
                expected = expected
                    .checked_add(period)
                    .ok_or(PowerStateError::TimeOverflow)?;
            }
            if self.next_sample != Some(expected) {
                return Err(PowerStateError::SampleSchedule);
            }
        }
        if let (Some(last_sample), Some(next_sample)) = (previous_sample, self.next_sample)
            && next_sample <= last_sample.cycle
        {
            return Err(PowerStateError::SampleSchedule);
        }
        if let (Some(next_sample), Some(last_cycle)) = (self.next_sample, domain_time)
            && next_sample <= last_cycle
        {
            return Err(PowerStateError::SampleSchedule);
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TypedEventPayload {
    SmWake,
    /// An explicit value-bearing L1 admission supplied by an external event
    /// producer. The typed boundary retains only bounded cardinalities; the
    /// payload bytes and allocation identities remain in the native event
    /// input and are never treated as hardware completion evidence.
    L1PayloadAdmission {
        payload_records: u64,
        witness_lines: u64,
        identity_present: bool,
    },
    L2Access {
        request: TypedMemoryRequest,
    },
    L2AccessBatch {
        requests: Vec<TypedMemoryRequest>,
    },
    L2Fill {
        fill: TypedFillIdentity,
    },
    AtomicCompletion,
    WritebackCompletion {
        token: u64,
        line_id: i64,
        dirty_mask: u8,
        service_done_at: u64,
    },
    CoherenceCompletion {
        token: u64,
        line_id: i64,
        sector_mask: u8,
        service_sectors: u32,
        service_done_at: u64,
    },
    L1VisibilityCompletion {
        token: u64,
        line_id: i64,
        sector_mask: u8,
        service_sectors: u32,
        service_done_at: u64,
    },
    L1VisibilityAccess {
        sequence: u64,
        requester: u8,
        line_id: i64,
        sector_mask: u8,
        is_write: bool,
        l1_resident_after: bool,
        fill_will_reside: bool,
        fill_generation: Option<u32>,
        hit: bool,
        issue_cycle: u64,
    },
    L1VisibilityFence {
        sequence: u64,
        requester: u8,
        pc: u32,
        issue_cycle: u64,
        scope: String,
        order: String,
    },
    L1FillVisible {
        sequence: u64,
        requester: u8,
        line_id: i64,
        sector_mask: u8,
        fill_generation: u32,
        visible_cycle: u64,
    },
    L1InvalidateAll {
        sequence: u64,
        requester: u8,
        issue_cycle: u64,
    },
    MemoryWriteAck {
        warp: u32,
        instruction: u32,
        subcore: u8,
        request_index: u16,
        request_count: u16,
        operation: TypedMemoryOperation,
        issue_at: u64,
        accepted_at: u64,
        service_done_at: u64,
        fence_visible_at: u64,
    },
    MemoryResponse {
        reply: Option<TypedResponseIdentity>,
        fill: Option<TypedFillIdentity>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedResponseIdentity {
    pub warp: u64,
    pub instruction: u64,
    pub sector_index: u64,
    pub response_kind: TypedResponseKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedFillIdentity {
    pub line_id: i64,
    pub generation: u64,
    pub sector_mask: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedAdmissionWitness {
    pub warp: u32,
    pub instruction: u32,
    pub subcore: u8,
    pub request_index: u16,
    pub request_count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedMemoryRequest {
    pub issue_time: u64,
    pub sector_id: i64,
    pub origin: TypedMemoryOrigin,
    pub sector_mask: u8,
    pub local_sector_mask: u8,
    pub source: TypedL2Source,
    pub operation: TypedMemoryOperation,
    pub constant_load: bool,
    pub constant_deep: bool,
    pub reply: Option<TypedResponseIdentity>,
    pub admission_witness: Option<TypedAdmissionWitness>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedPendingEvent {
    pub time: u64,
    pub sequence: u64,
    /// Current engine phase. Phase zero is the engine's single serialized
    /// class; future multi-phase engines must expose their ordering here.
    pub phase: u8,
    /// Actual engine reservation, including the anchor sequence.
    pub sequence_span: u64,
    pub target: u32,
    pub payload: TypedEventPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueBoundaryState {
    pub depth: u64,
    pub next_event_time: Option<u64>,
    /// Next unused insertion position; null explicitly denotes exhaustion.
    /// Absence is a malformed snapshot, not an exhausted queue.
    #[serde(deserialize_with = "Option::<u64>::deserialize")]
    pub next_sequence: Option<u64>,
    pub pending: Vec<TypedPendingEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EventSequenceError {
    #[error("event at sequence {sequence} reserves zero positions")]
    EmptySpan { sequence: u64 },
    #[error("event at sequence {sequence} overflows its {span}-position reservation")]
    SpanOverflow { sequence: u64, span: u64 },
    #[error(
        "event at sequence {sequence} reserves {span} positions for {logical_events} logical events"
    )]
    PayloadSpanMismatch {
        sequence: u64,
        span: u64,
        logical_events: usize,
    },
}

fn sequence_end(sequence: u64, span: u64) -> Result<u64, EventSequenceError> {
    let offset = span
        .checked_sub(1)
        .ok_or(EventSequenceError::EmptySpan { sequence })?;
    sequence
        .checked_add(offset)
        .ok_or(EventSequenceError::SpanOverflow { sequence, span })
}

impl TypedEventPayload {
    fn validate_event_time(&self, event_time: u64) -> Result<(), BoundaryStateError> {
        let check = |payload: &'static str, field: &'static str, found: u64| {
            if found != event_time {
                return Err(BoundaryStateError::PayloadTimeMismatch {
                    payload,
                    field,
                    expected: event_time,
                    found,
                });
            }
            Ok(())
        };
        match self {
            Self::L2Access { request } => {
                request.validate_contract()?;
                check("l2_access", "issue_time", request.issue_time)
            }
            Self::L2AccessBatch { requests } => {
                validate_batch_witnesses(requests)?;
                requests.iter().try_for_each(|request| {
                    request.validate_contract()?;
                    check("l2_access_batch", "issue_time", request.issue_time)
                })
            }
            Self::MemoryWriteAck {
                operation,
                subcore,
                request_index,
                request_count,
                issue_at,
                accepted_at,
                service_done_at,
                fence_visible_at,
                ..
            } => {
                if !matches!(
                    operation,
                    TypedMemoryOperation::Store | TypedMemoryOperation::Reduction
                ) {
                    return Err(BoundaryStateError::MemoryWriteAckOperation {
                        operation: *operation,
                    });
                }
                if *subcore >= 4 || *request_count == 0 || *request_index >= *request_count {
                    return Err(BoundaryStateError::MemoryWriteAckWitness {
                        subcore: *subcore,
                        request_index: *request_index,
                        request_count: *request_count,
                    });
                }
                if accepted_at < issue_at {
                    return Err(BoundaryStateError::PayloadTimestampOrder {
                        payload: "memory_write_ack",
                        earlier_field: "issue_at",
                        earlier: *issue_at,
                        later_field: "accepted_at",
                        later: *accepted_at,
                    });
                }
                if service_done_at < accepted_at {
                    return Err(BoundaryStateError::PayloadTimestampOrder {
                        payload: "memory_write_ack",
                        earlier_field: "accepted_at",
                        earlier: *accepted_at,
                        later_field: "service_done_at",
                        later: *service_done_at,
                    });
                }
                if fence_visible_at < service_done_at {
                    return Err(BoundaryStateError::PayloadTimestampOrder {
                        payload: "memory_write_ack",
                        earlier_field: "service_done_at",
                        earlier: *service_done_at,
                        later_field: "fence_visible_at",
                        later: *fence_visible_at,
                    });
                }
                check("memory_write_ack", "fence_visible_at", *fence_visible_at)
            }
            Self::L2Fill { fill } => validate_fill_identity(fill),
            Self::MemoryResponse { reply, fill } => {
                let Some(reply) = reply else {
                    return Err(BoundaryStateError::MemoryResponseReplyMissing);
                };
                validate_response_identity(reply)?;
                if let Some(fill) = fill {
                    validate_fill_identity(fill)?;
                    if !matches!(
                        reply.response_kind,
                        TypedResponseKind::L1Fill
                            | TypedResponseKind::L1SectorFill
                            | TypedResponseKind::Warp
                            | TypedResponseKind::AsyncCopy
                    ) {
                        return Err(BoundaryStateError::MemoryResponseFillReplyMismatch {
                            response_kind: reply.response_kind,
                        });
                    }
                }
                Ok(())
            }
            Self::SmWake | Self::AtomicCompletion => Ok(()),
            Self::L1PayloadAdmission { .. } => Ok(()),
            Self::WritebackCompletion {
                service_done_at, ..
            } => check("writeback_completion", "service_done_at", *service_done_at),
            Self::CoherenceCompletion {
                service_done_at, ..
            } => check("coherence_completion", "service_done_at", *service_done_at),
            Self::L1VisibilityCompletion {
                service_done_at, ..
            } => check(
                "l1_visibility_completion",
                "service_done_at",
                *service_done_at,
            ),
            Self::L1VisibilityAccess { issue_cycle, .. } => {
                check("l1_visibility_access", "issue_cycle", *issue_cycle)
            }
            Self::L1VisibilityFence { issue_cycle, .. } => {
                check("l1_visibility_fence", "issue_cycle", *issue_cycle)
            }
            Self::L1FillVisible { visible_cycle, .. } => {
                check("l1_fill_visible", "visible_cycle", *visible_cycle)
            }
            Self::L1InvalidateAll { issue_cycle, .. } => {
                check("l1_invalidate_all", "issue_cycle", *issue_cycle)
            }
        }
    }

    fn sequence_end(&self, sequence: u64, span: u64) -> Result<u64, EventSequenceError> {
        let end = sequence_end(sequence, span)?;
        let logical_events = match self {
            Self::L2AccessBatch { requests } => requests.len(),
            _ => 1,
        };
        if u64::try_from(logical_events) != Ok(span) {
            return Err(EventSequenceError::PayloadSpanMismatch {
                sequence,
                span,
                logical_events,
            });
        }
        Ok(end)
    }
}

fn validate_fill_identity(fill: &TypedFillIdentity) -> Result<(), BoundaryStateError> {
    if fill.sector_mask == 0 || fill.sector_mask & !0x0f != 0 {
        return Err(BoundaryStateError::FillIdentityMask {
            sector_mask: fill.sector_mask,
        });
    }
    Ok(())
}

fn validate_response_identity(response: &TypedResponseIdentity) -> Result<(), BoundaryStateError> {
    if matches!(response.response_kind, TypedResponseKind::None) {
        return Err(BoundaryStateError::ResponseIdentityKindNone);
    }
    Ok(())
}

impl TypedMemoryRequest {
    fn validate_contract(&self) -> Result<(), BoundaryStateError> {
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

fn validate_batch_witnesses(requests: &[TypedMemoryRequest]) -> Result<(), BoundaryStateError> {
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

impl QueueBoundaryState {
    pub fn validate(&self) -> Result<(), QueueStateError> {
        if self.depth != u64::try_from(self.pending.len()).unwrap_or(u64::MAX) {
            return Err(QueueStateError::DepthMismatch {
                depth: self.depth,
                pending: self.pending.len(),
            });
        }
        let mut previous = None;
        let mut reservations = BTreeMap::new();
        for event in &self.pending {
            let end = event
                .payload
                .sequence_end(event.sequence, event.sequence_span)?;
            if reservations.insert(event.sequence, end).is_some() {
                return Err(QueueStateError::DuplicateSequence {
                    sequence: event.sequence,
                });
            }
            self.validate_reservation_end(end)?;
            if let Some(previous) = previous
                && (event.time, event.phase, event.sequence) <= previous
            {
                return Err(QueueStateError::Unsorted);
            }
            previous = Some((event.time, event.phase, event.sequence));
        }
        // Timestamp order need not be insertion order. Reservations occupy
        // one shared sequence space across all timestamps; never expand a
        // span into individual positions to check for overlap.
        let mut previous_end = None;
        for (sequence, end) in reservations {
            if let Some(previous_end) = previous_end
                && sequence <= previous_end
            {
                return Err(QueueStateError::OverlappingSpans {
                    sequence,
                    previous_end,
                });
            }
            previous_end = Some(end);
        }
        if self.next_event_time != self.pending.first().map(|event| event.time) {
            return Err(QueueStateError::NextTimeMismatch);
        }
        Ok(())
    }

    fn validate_reservation_end(&self, end: u64) -> Result<(), QueueStateError> {
        if let Some(next_sequence) = self.next_sequence
            && end >= next_sequence
        {
            return Err(QueueStateError::UnreservedSequence { end, next_sequence });
        }
        Ok(())
    }
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedEventIdentity {
    pub time: u64,
    pub sequence: u64,
    pub sequence_span: u64,
    pub phase: u8,
    pub target: u32,
    pub payload: TypedEventPayload,
}

impl TypedEventIdentity {
    pub fn sequence_end(&self) -> Result<u64, EventSequenceError> {
        self.payload.sequence_end(self.sequence, self.sequence_span)
    }
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
    /// Native maximum CTA-barrier release time across all admitted blocks,
    /// including retired ones. Independent of the issue clock and the memory
    /// pipeline's terminal acceptance: release can lie ahead of `cycle`.
    /// Explicit null denotes no released generation; the field is required.
    #[serde(deserialize_with = "Option::<u64>::deserialize")]
    pub local_completion_time: Option<u64>,
    /// Timing state that affects future issue but is not yet serialized
    /// field-by-field. Exact contracts reify whenever this list is nonempty.
    pub unprojected_timing_state: Vec<String>,
    pub async_copy: Vec<Option<AsyncCopyBoundaryState>>,
    pub response_owners: Vec<L2RequestOwner>,
}

impl TypedCoreBoundaryState {
    pub fn validate(&self) -> Result<(), CoreStateError> {
        if self.sm_id == u32::MAX {
            return Err(CoreStateError::InvalidSmIdentity);
        }
        if self.subcore_warps.len() != 4 {
            return Err(CoreStateError::SubcoreCount {
                found: self.subcore_warps.len(),
            });
        }
        let mut previous_unprojected = None;
        for reason in &self.unprojected_timing_state {
            if reason.is_empty() || previous_unprojected.is_some_and(|previous| reason <= previous)
            {
                return Err(CoreStateError::UnprojectedTimingState);
            }
            previous_unprojected = Some(reason);
        }
        if self.next_block != u64::try_from(self.block_states.len()).unwrap_or(u64::MAX) {
            return Err(CoreStateError::NextBlockMismatch {
                next_block: self.next_block,
                block_states: self.block_states.len(),
            });
        }
        let mut roster = BTreeSet::new();
        for warps in &self.subcore_warps {
            for &warp_slot in warps {
                if warp_slot < 0 {
                    return Err(CoreStateError::NegativeWarpSlot { warp_slot });
                }
                if !roster.insert(warp_slot) {
                    return Err(CoreStateError::DuplicateResidentWarp { warp_slot });
                }
            }
        }
        for (subcore, warp_slot) in self
            .hold
            .iter()
            .enumerate()
            .filter_map(|(subcore, slot)| slot.as_ref().map(|&warp_slot| (subcore, warp_slot)))
        {
            let warp_slot = i32::try_from(warp_slot)
                .map_err(|_| CoreStateError::HoldWarpMissing { warp_slot })?;
            if !roster.contains(&warp_slot) {
                return Err(CoreStateError::HoldWarpMissing {
                    warp_slot: warp_slot as u64,
                });
            }
            if !self.subcore_warps[subcore].contains(&warp_slot) {
                return Err(CoreStateError::HoldWarpWrongSubcore { subcore, warp_slot });
            }
        }
        let mut known_warps = BTreeSet::new();
        for warp in &self.warp_states {
            if warp.warp_slot < 0 {
                return Err(CoreStateError::NegativeWarpSlot {
                    warp_slot: warp.warp_slot,
                });
            }
            validate_warp_instruction_buffer(warp, self.front_end.as_ref())?;
            if usize::try_from(warp.block)
                .ok()
                .is_none_or(|block| block >= self.block_states.len())
            {
                return Err(CoreStateError::UnknownWarpBlock { block: warp.block });
            }
            if !known_warps.insert((warp.block, warp.warp_slot)) {
                return Err(CoreStateError::DuplicateWarpState {
                    block: warp.block,
                    warp_slot: warp.warp_slot,
                });
            }
        }
        let issued_from_warps = self.warp_states.iter().try_fold(0_u64, |sum, warp| {
            sum.checked_add(warp.next_instruction)
                .ok_or(CoreStateError::CounterOverflow)
        })?;
        if issued_from_warps != self.issued {
            return Err(CoreStateError::IssuedCount {
                declared: self.issued,
                warps: issued_from_warps,
            });
        }
        let mut block_warps = BTreeSet::new();
        let mut active_block_slots = BTreeMap::new();
        let mut unfinished_by_block = BTreeMap::new();
        let mut unfinished_state_count = 0_u64;
        let mut resident_warp_reservation = 0_u64;
        let mut expected_first_warp = 0_u64;
        for warp in &self.warp_states {
            if !warp.done {
                unfinished_state_count = unfinished_state_count
                    .checked_add(1)
                    .ok_or(CoreStateError::CounterOverflow)?;
                let count = unfinished_by_block.entry(warp.block).or_insert(0_u64);
                *count = count
                    .checked_add(1)
                    .ok_or(CoreStateError::CounterOverflow)?;
                if !roster.contains(&warp.warp_slot) {
                    return Err(CoreStateError::UnfinishedWarpNotResident {
                        warp_slot: warp.warp_slot,
                    });
                }
            }
        }
        for (block_index, block) in self.block_states.iter().enumerate() {
            if block.unfinished != 0 {
                resident_warp_reservation = resident_warp_reservation
                    .checked_add(u64::from(block.warp_count))
                    .ok_or(CoreStateError::CounterOverflow)?;
            }
            if block.warp_count > u32::try_from(block.admitted_warps).unwrap_or(u32::MAX) {
                return Err(CoreStateError::BlockWarpCount {
                    block: block_index,
                    declared: block.warp_count,
                    admitted: block.admitted_warps,
                });
            }
            if block.unfinished > block.admitted_warps {
                return Err(CoreStateError::BlockUnfinishedCount { block: block_index });
            }
            if block.unfinished > u64::from(block.warp_count) {
                return Err(CoreStateError::BlockUnfinishedExceedsWarpCount {
                    block: block_index,
                    unfinished: block.unfinished,
                    warp_count: block.warp_count,
                });
            }
            if block.waiting > block.unfinished {
                return Err(CoreStateError::BlockWaitingCount { block: block_index });
            }
            if block.warp_slots.len() != usize::try_from(block.admitted_warps).unwrap_or(usize::MAX)
            {
                return Err(CoreStateError::BlockWarpRoster {
                    block: block_index,
                    declared: block.admitted_warps,
                    roster: block.warp_slots.len(),
                });
            }
            for &warp_slot in &block.warp_slots {
                if warp_slot < 0
                    || !known_warps
                        .contains(&(u64::try_from(block_index).unwrap_or(u64::MAX), warp_slot))
                {
                    return Err(CoreStateError::BlockWarpMissing { warp_slot });
                }
                if !block_warps.insert((block_index, warp_slot)) {
                    return Err(CoreStateError::DuplicateBlockWarp { warp_slot });
                }
            }
            if block.first_warp != expected_first_warp {
                return Err(CoreStateError::BlockFirstWarpOrder {
                    block: block_index,
                    expected: expected_first_warp,
                    found: block.first_warp,
                });
            }
            let range_end = block.first_warp.checked_add(block.admitted_warps).ok_or(
                CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                },
            )?;
            let warp_state_count = u64::try_from(self.warp_states.len()).unwrap_or(u64::MAX);
            if range_end > warp_state_count {
                return Err(CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                });
            }
            let range_start =
                usize::try_from(block.first_warp).map_err(|_| CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                })?;
            let range_end =
                usize::try_from(range_end).map_err(|_| CoreStateError::BlockWarpRange {
                    block: block_index,
                    first_warp: block.first_warp,
                    admitted_warps: block.admitted_warps,
                    warp_states: self.warp_states.len(),
                })?;
            let declared_slots = block.warp_slots.iter().copied().collect::<BTreeSet<_>>();
            let range_slots = self.warp_states[range_start..range_end]
                .iter()
                .map(|warp| warp.warp_slot)
                .collect::<BTreeSet<_>>();
            if declared_slots != range_slots {
                return Err(CoreStateError::BlockWarpRangeRoster { block: block_index });
            }
            for warp in &self.warp_states[range_start..range_end] {
                if warp.block != u64::try_from(block_index).unwrap_or(u64::MAX) {
                    return Err(CoreStateError::BlockWarpRangeIdentity {
                        block: block_index,
                        found: warp.block,
                    });
                }
            }
            validate_block_barrier_state(
                block_index,
                block,
                &self.warp_states[range_start..range_end],
                self.cycle,
            )?;
            expected_first_warp = range_end as u64;
            if block.unfinished != 0 {
                for &warp_slot in &block.warp_slots {
                    if let Some(&previous_block) = active_block_slots.get(&warp_slot) {
                        return Err(CoreStateError::ActiveBlockWarpSlotCollision {
                            warp_slot,
                            previous_block,
                            block: block_index,
                        });
                    }
                    active_block_slots.insert(warp_slot, block_index);
                }
            }
            let unfinished_states = unfinished_by_block
                .get(&u64::try_from(block_index).unwrap_or(u64::MAX))
                .copied()
                .unwrap_or(0);
            if unfinished_states != block.unfinished {
                return Err(CoreStateError::BlockWarpStateCount {
                    block: block_index,
                    declared: block.unfinished,
                    states: unfinished_states,
                });
            }
        }
        for (subcore, warps) in self.subcore_warps.iter().enumerate() {
            for &warp_slot in warps {
                let expected_subcore = usize::try_from(warp_slot.rem_euclid(4))
                    .expect("SM86 subcore modulo is within usize");
                if expected_subcore != subcore {
                    return Err(CoreStateError::ResidentWarpWrongSubcore {
                        warp_slot,
                        subcore,
                        expected_subcore,
                    });
                }
            }
        }
        let mut live_arrivals = BTreeSet::new();
        for warp in self.warp_states.iter().filter(|warp| !warp.done) {
            if !live_arrivals.insert(warp.arrival) {
                return Err(CoreStateError::DuplicateResidentWarpArrival {
                    arrival: warp.arrival,
                });
            }
        }
        let warp_state_count = u64::try_from(self.warp_states.len()).unwrap_or(u64::MAX);
        if expected_first_warp != warp_state_count {
            return Err(CoreStateError::WarpArenaTail {
                expected: expected_first_warp,
                found: warp_state_count,
            });
        }
        if u64::from(self.resident_warps) != resident_warp_reservation {
            return Err(CoreStateError::ResidentWarpReservationCount {
                declared: self.resident_warps,
                reservation: resident_warp_reservation,
            });
        }
        let unfinished_sum = self.block_states.iter().try_fold(0_u64, |sum, block| {
            sum.checked_add(block.unfinished)
                .ok_or(CoreStateError::CounterOverflow)
        })?;
        if unfinished_sum != self.unfinished_warps {
            return Err(CoreStateError::UnfinishedWarpCount {
                declared: self.unfinished_warps,
                blocks: unfinished_sum,
            });
        }
        if unfinished_state_count != self.unfinished_warps {
            return Err(CoreStateError::WarpStateCount {
                declared: self.unfinished_warps,
                states: unfinished_state_count,
            });
        }
        let shared_memory_sum = self
            .block_states
            .iter()
            .filter(|block| block.unfinished != 0)
            .try_fold(0_u64, |sum, block| {
                sum.checked_add(block.shared_memory_bytes)
                    .ok_or(CoreStateError::CounterOverflow)
            })?;
        if shared_memory_sum != self.resident_shared_memory_bytes {
            return Err(CoreStateError::SharedMemoryCount {
                declared: self.resident_shared_memory_bytes,
                blocks: shared_memory_sum,
            });
        }
        for &warp_slot in &roster {
            if !known_warps.iter().any(|(_, slot)| *slot == warp_slot) {
                return Err(CoreStateError::ResidentWarpMissing { warp_slot });
            }
        }
        if u64::try_from(roster.len()).unwrap_or(u64::MAX) != unfinished_state_count {
            return Err(CoreStateError::CurrentRosterCount {
                roster: roster.len(),
                unfinished_states: unfinished_state_count,
            });
        }
        if self.outstanding_responses
            != u64::try_from(self.response_owners.len()).unwrap_or(u64::MAX)
        {
            return Err(CoreStateError::ResponseCount {
                declared: self.outstanding_responses,
                owners: self.response_owners.len(),
            });
        }
        let mut previous = None;
        let mut seen = BTreeSet::new();
        for owner in &self.response_owners {
            if owner.request.sm_id != self.sm_id {
                return Err(CoreStateError::ResponseOwnerSm {
                    owner: owner.request.sm_id,
                });
            }
            let response_warp = usize::try_from(owner.request.response_warp).map_err(|_| {
                CoreStateError::ResponseOwnerWarpOutOfRange {
                    response_warp: owner.request.response_warp,
                    warp_states: self.warp_states.len(),
                }
            })?;
            let referenced_warp = self.warp_states.get(response_warp).ok_or(
                CoreStateError::ResponseOwnerWarpOutOfRange {
                    response_warp: owner.request.response_warp,
                    warp_states: self.warp_states.len(),
                },
            )?;
            if referenced_warp.warp_slot != owner.request.warp_slot {
                return Err(CoreStateError::ResponseOwnerWarpSlotMismatch {
                    response_warp: owner.request.response_warp,
                    owner_warp_slot: owner.request.warp_slot,
                    state_warp_slot: referenced_warp.warp_slot,
                });
            }
            if owner.request.instruction >= referenced_warp.next_instruction {
                return Err(CoreStateError::ResponseOwnerInstructionAfterCursor {
                    response_warp: owner.request.response_warp,
                    instruction: owner.request.instruction,
                    next_instruction: referenced_warp.next_instruction,
                });
            }
            if !matches!(owner.state, L2RequestState::Outstanding) {
                return Err(CoreStateError::ResponseOwnerState);
            }
            if owner.request.warp_slot < 0 {
                return Err(CoreStateError::NegativeResponseOwnerWarpSlot {
                    warp_slot: owner.request.warp_slot,
                });
            }
            let key = (
                owner.request.sm_id,
                owner.request.response_warp,
                owner.request.warp_slot,
                owner.request.instruction,
                owner.request.sector_index,
                owner.request.response_kind,
            );
            if !seen.insert(key) {
                return Err(CoreStateError::DuplicateResponseOwner);
            }
            if let Some(previous) = previous
                && key <= previous
            {
                return Err(CoreStateError::UnsortedResponseOwners);
            }
            previous = Some(key);
        }
        if let Some(l1) = &self.l1 {
            l1.validate_at_cycle(self.cycle)?
        }
        if let Some(memory) = &self.memory_pipeline {
            validate_memory_pipeline(memory, self.cycle)?;
        }
        if let Some(scalar) = &self.scalar_pipeline {
            validate_scalar_pipeline(scalar, self.cycle)?;
        }
        if let Some(front_end) = &self.front_end {
            front_end.validate()?
        }
        if !self.async_copy.is_empty() && self.async_copy.len() != self.warp_states.len() {
            return Err(CoreStateError::AsyncCopySlots);
        }
        for state in self.async_copy.iter().flatten() {
            state.validate_at_cycle(self.cycle)?
        }
        let has_released_barrier = self
            .block_states
            .iter()
            .any(|block| block.barrier_generation != 0);
        if has_released_barrier != self.local_completion_time.is_some()
            || self.local_completion_time == Some(0)
        {
            return Err(CoreStateError::LocalCompletionHistory);
        }
        Ok(())
    }
}

fn validate_warp_instruction_buffer(
    warp: &WarpBoundaryState,
    front_end: Option<&InstructionFrontEndBoundaryState>,
) -> Result<(), CoreStateError> {
    if warp.done && !warp.instruction_buffer.is_empty() {
        return Err(CoreStateError::DoneWarpInstructionBuffer {
            warp_slot: warp.warp_slot,
        });
    }
    let Some(front_end) = front_end else {
        if !warp.instruction_buffer.is_empty() {
            return Err(CoreStateError::InstructionBufferWithoutFrontEnd {
                warp_slot: warp.warp_slot,
            });
        }
        return Ok(());
    };
    let capacity = usize::try_from(front_end.instruction_buffer_entries).unwrap_or(usize::MAX);
    if warp.instruction_buffer.len() > capacity {
        return Err(CoreStateError::InstructionBufferCapacity {
            warp_slot: warp.warp_slot,
            found: warp.instruction_buffer.len(),
            capacity,
        });
    }
    if warp.front_end_fetch_cursor < warp.next_instruction {
        return Err(CoreStateError::InstructionCursorBeforeNext {
            warp_slot: warp.warp_slot,
            cursor: warp.front_end_fetch_cursor,
            next: warp.next_instruction,
        });
    }
    let mut previous: Option<u64> = None;
    for &(instruction, _) in &warp.instruction_buffer {
        if instruction < warp.next_instruction {
            return Err(CoreStateError::InstructionBufferBeforeNext {
                warp_slot: warp.warp_slot,
                next: warp.next_instruction,
                found: instruction,
            });
        }
        if instruction >= warp.front_end_fetch_cursor {
            return Err(CoreStateError::InstructionBufferBeyondCursor {
                warp_slot: warp.warp_slot,
                cursor: warp.front_end_fetch_cursor,
                found: instruction,
            });
        }
        if let Some(previous) = previous
            && instruction != previous.saturating_add(1)
        {
            return Err(CoreStateError::InstructionBufferOrder {
                warp_slot: warp.warp_slot,
                previous,
                found: instruction,
            });
        }
        previous = Some(instruction);
    }
    Ok(())
}

fn validate_memory_pipeline(
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

