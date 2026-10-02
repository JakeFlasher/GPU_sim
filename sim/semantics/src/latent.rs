//! Unknown-preserving mechanism envelopes for the SM86 replay boundary.
//!
//! The native trace and public profiler surfaces constrain several latent
//! mechanisms without identifying them.  This module gives those constraints
//! one small, reusable representation:
//!
//! * candidate domains retain ambiguity instead of selecting a route/owner;
//! * finite candidate worlds expose `must`/`may` summaries mechanically;
//! * frontier bounds remain separate and checked for ordering;
//! * endpoint-contract differences stay visible; matching text alone is not
//!   proof of measurement identity or a cross-domain timing conversion.
//!
//! These types are deliberately observational.  They do not add a hidden
//! hardware oracle, consume NCU labels, or turn a model-local choice into a
//! silicon fact.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const LATENT_SCHEMA_VERSION: u32 = 1;
pub const RESOURCE_DEMAND_SCHEMA_VERSION: u32 = 1;
const MAX_CANDIDATE_VALUES: usize = 4096;
const MAX_WORLDS: usize = 4096;
const MAX_RESOURCE_POOLS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LatentModelError {
    #[error("candidate domain has no values")]
    EmptyCandidateDomain,
    #[error("candidate value must be nonempty")]
    EmptyCandidateValue,
    #[error("finite candidates cannot carry unknown evidence provenance")]
    UnknownCandidateEvidence,
    #[error("candidate domain exceeds {MAX_CANDIDATE_VALUES} values")]
    CandidateDomainTooLarge,
    #[error("unknown-domain reason must be nonempty")]
    EmptyUnknownReason,
    #[error("candidate-domain restriction removed every feasible value")]
    CandidateIntersectionEmpty,
    #[error("candidate world set must be nonempty")]
    EmptyWorldSet,
    #[error("candidate world set exceeds {MAX_WORLDS} worlds")]
    WorldSetTooLarge,
    #[error("candidate world field {factor} must be nonempty")]
    EmptyWorldField { factor: &'static str },
    #[error("event producer must be nonempty")]
    EmptyEventProducer,
    #[error("event function_unique_id must be positive, got {0}")]
    InvalidFunctionId(i32),
    #[error("event SM id must be nonnegative, got {0}")]
    InvalidSmId(i32),
    #[error("frontier bound is inverted: lower={lower:?}, upper={upper:?}")]
    InvertedFrontier {
        lower: Option<u64>,
        upper: Option<u64>,
    },
    #[error("frontier ordering is impossible: {earlier} must not follow {later}")]
    FrontierOrder {
        earlier: &'static str,
        later: &'static str,
    },
    #[error("endpoint field {field} must be nonempty")]
    EmptyEndpointField { field: &'static str },
    #[error("endpoint field {field} contains an empty item")]
    EmptyEndpointItem { field: &'static str },
    #[error("unsupported latent schema version {found}; expected {expected}")]
    UnsupportedSchema { found: u32, expected: u32 },
    #[error("resource-demand schema version {found} is unsupported; expected {expected}")]
    UnsupportedResourceDemandSchema { found: u32, expected: u32 },
    #[error("resource pool name must be nonempty")]
    EmptyResourcePool,
    #[error("resource-demand pool count exceeds {MAX_RESOURCE_POOLS}")]
    ResourcePoolCountTooLarge,
    #[error("resource-demand interval for pool {pool} is inverted: lower={lower}, upper={upper}")]
    InvertedResourceDemand {
        pool: String,
        lower: u64,
        upper: u64,
    },
    #[error("resource-demand interval for pool {pool} exceeds total events {total}: upper={upper}")]
    ResourceDemandExceedsTotal {
        pool: String,
        upper: u64,
        total: u64,
    },
    #[error("resource-demand interval for pool {pool} overflows while adding unknown events")]
    ResourceDemandOverflow { pool: String },
    #[error("service-rate profile source must be nonempty")]
    EmptyServiceRateSource,
    #[error("service-rate profile must contain at least one pool")]
    EmptyServiceRateProfile,
    #[error("service-rate profile pool {pool} has non-positive capacity or timing")]
    InvalidServiceRate { pool: String },
    #[error("service-rate profile has no entry for resource pool {pool}")]
    MissingServiceRate { pool: String },
    #[error("NCU observer padding interval is inverted")]
    InvalidObserverPadding,
}

/// Evidence provenance is intentionally weaker than “hardware truth”.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Observed,
    Derived,
    Hypothesis,
    Unknown,
}

impl EvidenceKind {
    pub const fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Hypothesis, _) | (_, Self::Hypothesis) => Self::Hypothesis,
            (Self::Derived, _) | (_, Self::Derived) => Self::Derived,
            (Self::Observed, Self::Observed) => Self::Observed,
        }
    }
}

/// A finite candidate domain or an explicitly unresolved field.
///
/// `Unknown` is not represented as an empty set: an empty set means a
/// contradiction and is rejected by constructors and restriction operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum CandidateDomain {
    Unknown {
        reason: String,
    },
    Candidates {
        values: BTreeSet<String>,
        evidence: EvidenceKind,
    },
}

impl Default for CandidateDomain {
    fn default() -> Self {
        Self::Unknown {
            reason: "unobserved".to_owned(),
        }
    }
}

impl CandidateDomain {
    pub fn unknown(reason: impl Into<String>) -> Result<Self, LatentModelError> {
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(LatentModelError::EmptyUnknownReason);
        }
        Ok(Self::Unknown { reason })
    }

    pub fn candidate(value: impl Into<String>) -> Result<Self, LatentModelError> {
        Self::from_values([value])
    }

    pub fn from_values<I, S>(values: I) -> Result<Self, LatentModelError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::with_evidence(values, EvidenceKind::Hypothesis)
    }

    pub fn with_evidence<I, S>(values: I, evidence: EvidenceKind) -> Result<Self, LatentModelError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        if evidence == EvidenceKind::Unknown {
            return Err(LatentModelError::UnknownCandidateEvidence);
        }
        let mut set = BTreeSet::new();
        for value in values {
            let value = value.into();
            if value.trim().is_empty() {
                return Err(LatentModelError::EmptyCandidateValue);
            }
            set.insert(value);
            if set.len() > MAX_CANDIDATE_VALUES {
                return Err(LatentModelError::CandidateDomainTooLarge);
            }
        }
        if set.is_empty() {
            return Err(LatentModelError::EmptyCandidateDomain);
        }
        Ok(Self::Candidates {
            values: set,
            evidence,
        })
    }

    pub fn validate(&self) -> Result<(), LatentModelError> {
        match self {
            Self::Unknown { reason } => {
                if reason.trim().is_empty() {
                    Err(LatentModelError::EmptyUnknownReason)
                } else {
                    Ok(())
                }
            }
            Self::Candidates { values, evidence } => {
                if *evidence == EvidenceKind::Unknown {
                    return Err(LatentModelError::UnknownCandidateEvidence);
                }
                if values.is_empty() {
                    return Err(LatentModelError::EmptyCandidateDomain);
                }
                if values.len() > MAX_CANDIDATE_VALUES {
                    return Err(LatentModelError::CandidateDomainTooLarge);
                }
                if values.iter().any(|value| value.trim().is_empty()) {
                    return Err(LatentModelError::EmptyCandidateValue);
                }
                Ok(())
            }
        }
    }

    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown { .. })
    }

    /// A singleton is necessary in these candidate worlds, not necessarily
    /// on silicon. Its evidence basis must be retained by the consumer.
    pub fn must_value(&self) -> Option<&str> {
        if self.validate().is_err() {
            return None;
        }
        match self {
            Self::Candidates { values, .. } if values.len() == 1 => {
                values.iter().next().map(String::as_str)
            }
            _ => None,
        }
    }

    pub fn may_values(&self) -> Option<&BTreeSet<String>> {
        match self {
            Self::Candidates { values, .. } => Some(values),
            Self::Unknown { .. } => None,
        }
    }

    pub const fn evidence(&self) -> EvidenceKind {
        match self {
            Self::Candidates { evidence, .. } => *evidence,
            Self::Unknown { .. } => EvidenceKind::Unknown,
        }
    }

    /// Apply a new evidence constraint.  Unknown fields become the supplied
    /// finite domain; known fields are intersected and cannot be repaired to
    /// an empty set.
    pub fn restrict<I, S>(&self, allowed: I) -> Result<Self, LatentModelError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.validate()?;
        let allowed = match Self::from_values(allowed)? {
            Self::Candidates { values, .. } => values,
            Self::Unknown { .. } => unreachable!("from_values never returns Unknown"),
        };
        match self {
            Self::Unknown { .. } => Self::from_values(allowed),
            Self::Candidates { values, .. } => {
                let intersection: BTreeSet<String> =
                    values.intersection(&allowed).cloned().collect();
                if intersection.is_empty() {
                    return Err(LatentModelError::CandidateIntersectionEmpty);
                }
                Self::from_values(intersection)
            }
        }
    }

    /// A union is used when two independent explanations are retained.  If
    /// either explanation is unknown, the union remains unknown.
    pub fn union(&self, other: &Self) -> Result<Self, LatentModelError> {
        self.validate()?;
        other.validate()?;
        match (self, other) {
            (Self::Unknown { reason }, _) => Self::unknown(format!("unknown_union:{reason}")),
            (_, Self::Unknown { reason }) => Self::unknown(format!("unknown_union:{reason}")),
            (
                Self::Candidates {
                    values: left,
                    evidence: left_evidence,
                },
                Self::Candidates {
                    values: right,
                    evidence: right_evidence,
                },
            ) => Self::with_evidence(
                left.union(right).cloned(),
                left_evidence.combine(*right_evidence),
            ),
        }
    }
}

/// Stable identity for one dynamic event.  It is independent of vector
/// position and therefore safe to use when candidate evidence is reassembled.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventKey {
    pub launch_uid: u64,
    pub function_uid: i32,
    pub cta_uid: u64,
    pub warp_uid: u32,
    pub instruction_ordinal: u64,
    pub pc: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lane_or_sector: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sm_id: Option<i32>,
    pub producer: String,
}

impl EventKey {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        launch_uid: u64,
        function_uid: i32,
        cta_uid: u64,
        warp_uid: u32,
        instruction_ordinal: u64,
        pc: u64,
        lane_or_sector: Option<u32>,
        sm_id: Option<i32>,
        producer: impl Into<String>,
    ) -> Result<Self, LatentModelError> {
        if function_uid <= 0 {
            return Err(LatentModelError::InvalidFunctionId(function_uid));
        }
        if let Some(sm_id) = sm_id
            && sm_id < 0
        {
            return Err(LatentModelError::InvalidSmId(sm_id));
        }
        let producer = producer.into();
        if producer.trim().is_empty() {
            return Err(LatentModelError::EmptyEventProducer);
        }
        Ok(Self {
            launch_uid,
            function_uid,
            cta_uid,
            warp_uid,
            instruction_ordinal,
            pc,
            lane_or_sector,
            sm_id,
            producer,
        })
    }

    pub fn validate(&self) -> Result<(), LatentModelError> {
        if self.function_uid <= 0 {
            return Err(LatentModelError::InvalidFunctionId(self.function_uid));
        }
        if let Some(sm_id) = self.sm_id
            && sm_id < 0
        {
            return Err(LatentModelError::InvalidSmId(sm_id));
        }
        if self.producer.trim().is_empty() {
            return Err(LatentModelError::EmptyEventProducer);
        }
        Ok(())
    }
}

/// One bounded interval in the simulator's own cycle domain. `None` denotes
/// an unknown side, not zero and not infinity represented by a sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierBound {
    pub lower: Option<u64>,
    pub upper: Option<u64>,
}

impl FrontierBound {
    pub const fn unknown() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }

    pub const fn exact(value: u64) -> Self {
        Self {
            lower: Some(value),
            upper: Some(value),
        }
    }

    pub const fn lower_bound(value: u64) -> Self {
        Self {
            lower: Some(value),
            upper: None,
        }
    }

    pub const fn upper_bound(value: u64) -> Self {
        Self {
            lower: None,
            upper: Some(value),
        }
    }

    pub const fn interval(lower: u64, upper: u64) -> Result<Self, LatentModelError> {
        if lower > upper {
            return Err(LatentModelError::InvertedFrontier {
                lower: Some(lower),
                upper: Some(upper),
            });
        }
        Ok(Self {
            lower: Some(lower),
            upper: Some(upper),
        })
    }

    pub const fn validate(self) -> Result<(), LatentModelError> {
        if let (Some(lower), Some(upper)) = (self.lower, self.upper)
            && lower > upper
        {
            return Err(LatentModelError::InvertedFrontier {
                lower: Some(lower),
                upper: Some(upper),
            });
        }
        Ok(())
    }

    /// Intersect independent bounds. This is the operation for adding
    /// evidence; an impossible intersection is a typed contradiction.
    pub fn intersect(self, other: Self) -> Result<Self, LatentModelError> {
        self.validate()?;
        other.validate()?;
        let lower = match (self.lower, other.lower) {
            (Some(left), Some(right)) => Some(if left > right { left } else { right }),
            (Some(value), None) | (None, Some(value)) => Some(value),
            (None, None) => None,
        };
        let upper = match (self.upper, other.upper) {
            (Some(left), Some(right)) => Some(if left < right { left } else { right }),
            (Some(value), None) | (None, Some(value)) => Some(value),
            (None, None) => None,
        };
        let result = Self { lower, upper };
        result.validate()?;
        Ok(result)
    }
}

impl Default for FrontierBound {
    fn default() -> Self {
        Self::unknown()
    }
}

/// A frontier explicitly required by this simulator completion contract even
/// when neither of its bounds is known. Not listing an entirely unobserved
/// field means it is outside this projection, not known to have zero cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(usize)]
pub enum CompletionRequirement {
    Issue,
    ModeledEvent,
    ResourceService,
    TypedTail,
    CacheHistory,
    Visibility,
    WritebackService,
}

/// Copy-friendly field-activation witness for completion projection. A bound
/// with no values is normally inactive; callers that know an unobserved tail
/// is still required set the corresponding bit here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequirements {
    bits: [bool; 7],
}

impl CompletionRequirements {
    pub const fn none() -> Self {
        Self { bits: [false; 7] }
    }

    pub const fn contains(self, requirement: CompletionRequirement) -> bool {
        self.bits[requirement as usize]
    }

    pub fn insert(&mut self, requirement: CompletionRequirement) {
        self.bits[requirement as usize] = true;
    }
}

impl Default for CompletionRequirements {
    fn default() -> Self {
        Self::none()
    }
}

impl FromIterator<CompletionRequirement> for CompletionRequirements {
    fn from_iter<T: IntoIterator<Item = CompletionRequirement>>(iter: T) -> Self {
        let mut requirements = Self::none();
        for requirement in iter {
            requirements.insert(requirement);
        }
        requirements
    }
}

/// Separate issue, completion, visibility, endpoint, and typed-tail domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierVector {
    pub issue: FrontierBound,
    pub modeled_event: FrontierBound,
    /// Completion of an abstract service schedule composed after the
    /// simulator's modeled-event frontier. This is model-local state, not a
    /// physical-pipe or hardware-completion observation.
    pub resource_service: FrontierBound,
    pub launch_envelope: FrontierBound,
    pub typed_tail: FrontierBound,
    /// Unresolved cache-history service/visibility tail. This is an
    /// uncertainty frontier, never a claim that topology overlap is a
    /// coherence bottleneck.
    pub cache_history: FrontierBound,
    pub visibility: FrontierBound,
    pub writeback_service: FrontierBound,
    pub endpoint_start: FrontierBound,
    pub endpoint_end: FrontierBound,
    /// Present bounds are active requirements too. This set additionally
    /// distinguishes active-but-wholly-unknown state from fields not projected.
    #[serde(default)]
    pub required_completion: CompletionRequirements,
}

impl FrontierVector {
    pub fn validate(&self) -> Result<(), LatentModelError> {
        for bound in [
            self.issue,
            self.modeled_event,
            self.resource_service,
            self.launch_envelope,
            self.typed_tail,
            self.cache_history,
            self.visibility,
            self.writeback_service,
            self.endpoint_start,
            self.endpoint_end,
        ] {
            bound.validate()?;
        }
        // Only edges that are mechanically in the same ordered domain are
        // checked.  Launch envelopes, visibility, and writeback are kept
        // separate because their relative order is intervention-dependent.
        for (earlier_name, earlier, later_name, later) in [
            ("issue", self.issue, "modeled_event", self.modeled_event),
            (
                "modeled_event",
                self.modeled_event,
                "typed_tail",
                self.typed_tail,
            ),
            (
                "endpoint_start",
                self.endpoint_start,
                "endpoint_end",
                self.endpoint_end,
            ),
        ] {
            if let (Some(lower), Some(upper)) = (earlier.lower, later.upper)
                && lower > upper
            {
                return Err(LatentModelError::FrontierOrder {
                    earlier: earlier_name,
                    later: later_name,
                });
            }
        }
        Ok(())
    }

    /// Compose a model-local service interval after a known frontier.
    /// Unknown sides remain unknown; no sentinel represents an unbounded
    /// upper side.
    pub fn compose_after(
        base: FrontierBound,
        service: ServiceCycleInterval,
    ) -> Result<FrontierBound, LatentModelError> {
        base.validate()?;
        if service.lower_cycles > service.upper_cycles {
            return Err(LatentModelError::InvertedFrontier {
                lower: Some(service.lower_cycles),
                upper: Some(service.upper_cycles),
            });
        }
        let lower = match base.lower {
            Some(value) => Some(value.checked_add(service.lower_cycles).ok_or_else(|| {
                LatentModelError::ResourceDemandOverflow {
                    pool: "frontier_service_lower".to_owned(),
                }
            })?),
            None => None,
        };
        let upper = match base.upper {
            Some(value) => Some(value.checked_add(service.upper_cycles).ok_or_else(|| {
                LatentModelError::ResourceDemandOverflow {
                    pool: "frontier_service_upper".to_owned(),
                }
            })?),
            None => None,
        };
        let result = FrontierBound { lower, upper };
        result.validate()?;
        Ok(result)
    }

    /// Compose two frontiers whose overlap policy is explicit. A scalar
    /// `max(...)` is only valid for the parallel case; serial admission/service
    /// edges use checked interval addition.
    pub fn compose_with_overlap(
        admission: FrontierBound,
        service: FrontierBound,
        mode: ServiceOverlapMode,
    ) -> Result<FrontierBound, LatentModelError> {
        admission.validate()?;
        service.validate()?;
        match mode {
            ServiceOverlapMode::SerialAfterAdmission => {
                let lower = match (admission.lower, service.lower) {
                    (Some(left), Some(right)) => {
                        Some(left.checked_add(right).ok_or_else(|| {
                            LatentModelError::ResourceDemandOverflow {
                                pool: "service_serial_lower".to_owned(),
                            }
                        })?)
                    }
                    _ => None,
                };
                let upper = match (admission.upper, service.upper) {
                    (Some(left), Some(right)) => {
                        Some(left.checked_add(right).ok_or_else(|| {
                            LatentModelError::ResourceDemandOverflow {
                                pool: "service_serial_upper".to_owned(),
                            }
                        })?)
                    }
                    _ => None,
                };
                let result = FrontierBound { lower, upper };
                result.validate()?;
                Ok(result)
            }
            ServiceOverlapMode::ParallelPoolMax => {
                let lower = match (admission.lower, service.lower) {
                    (Some(left), Some(right)) => Some(left.max(right)),
                    _ => None,
                };
                let upper = match (admission.upper, service.upper) {
                    (Some(left), Some(right)) => Some(left.max(right)),
                    _ => None,
                };
                let result = FrontierBound { lower, upper };
                result.validate()?;
                Ok(result)
            }
        }
    }

    /// Compute elapsed duration from an endpoint start/end interval. The
    /// caller must establish that both bounds use one clock/origin contract.
    pub fn elapsed(
        start: FrontierBound,
        end: FrontierBound,
    ) -> Result<FrontierBound, LatentModelError> {
        start.validate()?;
        end.validate()?;
        let lower = match (end.lower, start.upper) {
            (Some(end), Some(start)) => Some(end.saturating_sub(start)),
            _ => None,
        };
        let upper = match (end.upper, start.lower) {
            (Some(end), Some(start)) if end >= start => Some(end - start),
            (Some(_), Some(_)) => {
                return Err(LatentModelError::FrontierOrder {
                    earlier: "endpoint_start",
                    later: "endpoint_end",
                });
            }
            _ => None,
        };
        let result = FrontierBound { lower, upper };
        result.validate()?;
        Ok(result)
    }

    fn completion_bounds(&self) -> impl Iterator<Item = FrontierBound> + '_ {
        [
            (CompletionRequirement::Issue, self.issue),
            (CompletionRequirement::ModeledEvent, self.modeled_event),
            (
                CompletionRequirement::ResourceService,
                self.resource_service,
            ),
            (CompletionRequirement::TypedTail, self.typed_tail),
            (CompletionRequirement::CacheHistory, self.cache_history),
            (
                CompletionRequirement::WritebackService,
                self.writeback_service,
            ),
            (CompletionRequirement::Visibility, self.visibility),
        ]
        .into_iter()
        .filter(|(field, bound)| {
            self.required_completion.contains(*field)
                || bound.lower.is_some()
                || bound.upper.is_some()
        })
        .map(|(_, bound)| bound)
        .chain(
            (self.endpoint_end.lower.is_some() || self.endpoint_end.upper.is_some())
                .then_some(self.endpoint_end),
        )
    }

    /// Completion of all declared required frontiers in the same cycle domain.
    /// The launch envelope is not part of this contract: it is a separate
    /// observation model, not an additional completion event.
    pub fn simulator_completion(&self) -> FrontierBound {
        let bounds: Vec<_> = self.completion_bounds().collect();
        let lower = bounds.iter().filter_map(|bound| bound.lower).max();
        let active_unbounded = bounds.iter().any(|bound| bound.upper.is_none());
        let upper = if active_unbounded {
            None
        } else {
            bounds.iter().filter_map(|bound| bound.upper).max()
        };
        FrontierBound { lower, upper }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockDomain {
    SimulatorCycles,
    DeviceTimer,
    NsightCompute,
    Cupti,
    CudaEvent,
    HostMonotonic,
    ReplayWall,
    Unknown,
}

/// Declared start/end semantics. A structural match is only a necessary
/// compatibility check: identity, actual clock regime, intervention, and
/// synchronized endpoints still need independent evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointDescriptor {
    pub clock_domain: ClockDomain,
    pub start_trigger: String,
    pub end_trigger: String,
    pub included_work: Vec<String>,
    pub excluded_work: Vec<String>,
    pub serialization: String,
    pub cache_state: String,
    pub clock_control: String,
}

impl EndpointDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clock_domain: ClockDomain,
        start_trigger: impl Into<String>,
        end_trigger: impl Into<String>,
        included_work: Vec<String>,
        excluded_work: Vec<String>,
        serialization: impl Into<String>,
        cache_state: impl Into<String>,
        clock_control: impl Into<String>,
    ) -> Result<Self, LatentModelError> {
        let descriptor = Self {
            clock_domain,
            start_trigger: start_trigger.into(),
            end_trigger: end_trigger.into(),
            included_work,
            excluded_work,
            serialization: serialization.into(),
            cache_state: cache_state.into(),
            clock_control: clock_control.into(),
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn validate(&self) -> Result<(), LatentModelError> {
        for (field, value) in [
            ("start_trigger", self.start_trigger.as_str()),
            ("end_trigger", self.end_trigger.as_str()),
            ("serialization", self.serialization.as_str()),
            ("cache_state", self.cache_state.as_str()),
            ("clock_control", self.clock_control.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(LatentModelError::EmptyEndpointField { field });
            }
        }
        for (field, values) in [
            ("included_work", &self.included_work),
            ("excluded_work", &self.excluded_work),
        ] {
            if values.iter().any(|value| value.trim().is_empty()) {
                return Err(LatentModelError::EmptyEndpointItem { field });
            }
        }
        Ok(())
    }

    pub fn same_contract_as(&self, other: &Self) -> bool {
        self.clock_domain != ClockDomain::Unknown
            && self.validate().is_ok()
            && other.validate().is_ok()
            && self == other
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentFactors {
    pub route: CandidateDomain,
    pub compiler: CandidateDomain,
    pub cache: CandidateDomain,
    pub timing: CandidateDomain,
}

impl Default for LatentFactors {
    fn default() -> Self {
        Self {
            route: CandidateDomain::default(),
            compiler: CandidateDomain::default(),
            cache: CandidateDomain::default(),
            timing: CandidateDomain::default(),
        }
    }
}

impl LatentFactors {
    pub fn validate(&self) -> Result<(), LatentModelError> {
        self.route.validate()?;
        self.compiler.validate()?;
        self.cache.validate()?;
        self.timing.validate()
    }
}

/// A model-local service-demand interval for one abstract resource pool.
///
/// `lower` is demand that every surviving candidate assignment must pay;
/// `upper` is demand that a feasible assignment may pay.  Neither bound is a
/// dynamic physical-pipe count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceDemandInterval {
    pub lower: u64,
    pub upper: u64,
}

impl ResourceDemandInterval {
    pub const fn exact(value: u64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    pub const fn interval(lower: u64, upper: u64) -> Self {
        Self { lower, upper }
    }
}

/// Aggregate demand over abstract resource pools.  This is the first
/// predictive-closure seam: it turns candidate route classes into a typed
/// service envelope without selecting a silicon route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceDemandEnvelope {
    pub schema_version: u32,
    pub total_events: u64,
    pub unclassified_events: u64,
    pub pools: BTreeMap<String, ResourceDemandInterval>,
    pub evidence: EvidenceKind,
}

impl ResourceDemandEnvelope {
    pub fn new(
        total_events: u64,
        unclassified_events: u64,
        pools: BTreeMap<String, ResourceDemandInterval>,
        evidence: EvidenceKind,
    ) -> Result<Self, LatentModelError> {
        let envelope = Self {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            total_events,
            unclassified_events,
            pools,
            evidence,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<(), LatentModelError> {
        if self.schema_version != RESOURCE_DEMAND_SCHEMA_VERSION {
            return Err(LatentModelError::UnsupportedResourceDemandSchema {
                found: self.schema_version,
                expected: RESOURCE_DEMAND_SCHEMA_VERSION,
            });
        }
        if self.pools.len() > MAX_RESOURCE_POOLS {
            return Err(LatentModelError::ResourcePoolCountTooLarge);
        }
        if self.unclassified_events > self.total_events {
            return Err(LatentModelError::ResourceDemandExceedsTotal {
                pool: "unclassified".to_owned(),
                upper: self.unclassified_events,
                total: self.total_events,
            });
        }
        for (pool, interval) in &self.pools {
            if pool.trim().is_empty() {
                return Err(LatentModelError::EmptyResourcePool);
            }
            if interval.lower > interval.upper {
                return Err(LatentModelError::InvertedResourceDemand {
                    pool: pool.clone(),
                    lower: interval.lower,
                    upper: interval.upper,
                });
            }
            if interval.upper > self.total_events {
                return Err(LatentModelError::ResourceDemandExceedsTotal {
                    pool: pool.clone(),
                    upper: interval.upper,
                    total: self.total_events,
                });
            }
        }
        Ok(())
    }
}

/// Positive abstract service parameters used by the route quotient.  These
/// are calibration inputs, not claims about a proprietary physical unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceRate {
    pub capacity: u32,
    pub initiation_interval_cycles: u64,
    /// True when the fitted initiation interval already includes the measured
    /// effect of the probe's service capacity. This prevents projecting an
    /// aggregate throughput slope and dividing by capacity a second time.
    #[serde(default)]
    pub capacity_already_applied: bool,
    pub latency_lower_cycles: u64,
    pub latency_upper_cycles: u64,
}

/// Versioned calibration profile for abstract resource pools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceRateProfile {
    pub schema_version: u32,
    pub source: String,
    pub pools: BTreeMap<String, ServiceRate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ncu_start_padding_cycles: Option<ServiceCycleInterval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ncu_end_padding_cycles: Option<ServiceCycleInterval>,
    /// Probe dimensions intentionally not represented by the current rate
    /// equation. Their presence must not be mistaken for fitted state.
    #[serde(default)]
    pub unmodeled_dimensions: Vec<String>,
}

impl ServiceRateProfile {
    pub fn validate(&self) -> Result<(), LatentModelError> {
        if self.schema_version != RESOURCE_DEMAND_SCHEMA_VERSION {
            return Err(LatentModelError::UnsupportedResourceDemandSchema {
                found: self.schema_version,
                expected: RESOURCE_DEMAND_SCHEMA_VERSION,
            });
        }
        if self.source.trim().is_empty() {
            return Err(LatentModelError::EmptyServiceRateSource);
        }
        if self.pools.is_empty() {
            return Err(LatentModelError::EmptyServiceRateProfile);
        }
        if self.pools.len() > MAX_RESOURCE_POOLS {
            return Err(LatentModelError::ResourcePoolCountTooLarge);
        }
        for (pool, rate) in &self.pools {
            if pool.trim().is_empty()
                || rate.capacity == 0
                || rate.initiation_interval_cycles == 0
                || rate.latency_lower_cycles == 0
                || rate.latency_lower_cycles > rate.latency_upper_cycles
            {
                return Err(LatentModelError::InvalidServiceRate { pool: pool.clone() });
            }
        }
        for padding in [self.ncu_start_padding_cycles, self.ncu_end_padding_cycles]
            .into_iter()
            .flatten()
        {
            if padding.lower_cycles > padding.upper_cycles {
                return Err(LatentModelError::InvalidObserverPadding);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceCycleInterval {
    pub lower_cycles: u64,
    pub upper_cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceOverlapMode {
    SerialAfterAdmission,
    ParallelPoolMax,
}

/// How an abstract service stage relates to the detailed modeled-event
/// frontier. This is a modeling contract, not a claim about physical GPU
/// overlap. Full-trace demand is not silently treated as residual work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceStageRole {
    ResidualClosure,
    ReplacementTiming,
    IndependentPostReplay,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceComposition {
    pub admission: FrontierBound,
    pub service: FrontierBound,
    pub completion: FrontierBound,
    pub overlap: ServiceOverlapMode,
    pub role: ServiceStageRole,
}

impl ServiceComposition {
    pub fn new(
        admission: FrontierBound,
        service: FrontierBound,
        overlap: ServiceOverlapMode,
    ) -> Result<Self, LatentModelError> {
        Self::new_with_role(
            admission,
            service,
            overlap,
            ServiceStageRole::ResidualClosure,
        )
    }

    pub fn new_with_role(
        admission: FrontierBound,
        service: FrontierBound,
        overlap: ServiceOverlapMode,
        role: ServiceStageRole,
    ) -> Result<Self, LatentModelError> {
        let completion = match role {
            ServiceStageRole::ReplacementTiming => {
                service.validate()?;
                service
            }
            ServiceStageRole::ResidualClosure | ServiceStageRole::IndependentPostReplay => {
                FrontierVector::compose_with_overlap(admission, service, overlap)?
            }
        };
        Ok(Self {
            admission,
            service,
            completion,
            overlap,
            role,
        })
    }
}

/// Projection of abstract demand through a positive service-rate profile.
/// This is a model interval, not a silicon timing observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceServiceProjection {
    pub schema_version: u32,
    pub role: &'static str,
    pub profile_source: String,
    pub lower_cycles: u64,
    /// `None` means the demand includes an unresolved class/tail. A finite
    /// interval from the named pools must not manufacture an upper bound for
    /// that unresolved work.
    pub upper_cycles: Option<u64>,
    pub unclassified_events: u64,
    pub pools: BTreeMap<String, ServiceCycleInterval>,
    pub pool_capacities: BTreeMap<String, u32>,
    pub composition: ServiceOverlapMode,
}

/// Minimal explicit queue/frontier model for service-stage reporting. It
/// records a serial admission convention and keeps route assignment
/// correlation unresolved; it is not a physical scheduler simulation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceQueueProjection {
    pub schema_version: u32,
    pub role: &'static str,
    pub admission: FrontierBound,
    pub issue_frontier: FrontierBound,
    pub completion_frontier: FrontierBound,
    pub queue_depth: FrontierBound,
    pub demand_events: u64,
    pub unclassified_events: u64,
    pub assignment_semantics: &'static str,
}

/// Aggregate cache/coherence candidates that are justified by an address
/// trace.  These are semantic possibilities, not physical owner or
/// read-from receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheCandidateSummary {
    pub schema_version: u32,
    pub role: &'static str,
    pub decoded_global_lines: u64,
    pub cross_sm_lines: u64,
    pub cross_sm_read_write_lines: u64,
    pub cross_sm_write_write_lines: u64,
    pub read_from: CandidateDomain,
    pub owner: CandidateDomain,
    pub hardware_completion: CandidateDomain,
    /// Temporal order is separate from topology overlap. It remains unknown
    /// unless an independent per-access order witness is supplied.
    pub temporal_order: CandidateDomain,
    /// Fence scope/order is not inferred from a line-sharing count.
    pub fence_scope: CandidateDomain,
    /// Allocation generation/read-from identity is explicit when available;
    /// line ids alone do not establish it.
    pub line_generation: CandidateDomain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheWitnessDomains {
    pub temporal_order: CandidateDomain,
    pub fence_scope: CandidateDomain,
    pub line_generation: CandidateDomain,
}

impl CacheWitnessDomains {
    pub fn unobserved() -> Result<Self, LatentModelError> {
        Ok(Self {
            temporal_order: CandidateDomain::unknown("cross-SM access order unobserved")?,
            fence_scope: CandidateDomain::unknown("fence scope/order unobserved")?,
            line_generation: CandidateDomain::unknown(
                "line generation/read-from identity unobserved",
            )?,
        })
    }

    fn validate(&self) -> Result<(), LatentModelError> {
        self.temporal_order.validate()?;
        self.fence_scope.validate()?;
        self.line_generation.validate()
    }

    /// Derive only the order witness the compact trace actually carries:
    /// per-SM record order when all decoded records have bounded identity.
    /// Cross-SM arbitration, fences, and allocation generations remain
    /// unknown even in this case.
    pub fn from_trace_observation(
        sectors_decoded: bool,
        unknown_sm_or_line_records: u64,
        instrumentation_order_records: u64,
        instrumentation_order_missing_records: u64,
    ) -> Result<Self, LatentModelError> {
        let temporal_order = if sectors_decoded
            && unknown_sm_or_line_records == 0
            && instrumentation_order_records != 0
            && instrumentation_order_missing_records == 0
        {
            CandidateDomain::with_evidence(
                ["warp_local_instrumentation_order"],
                EvidenceKind::Observed,
            )?
        } else {
            CandidateDomain::unknown("within-SM trace order unavailable")?
        };
        Ok(Self {
            temporal_order,
            fence_scope: CandidateDomain::unknown("fence scope/order unobserved")?,
            line_generation: CandidateDomain::unknown(
                "line generation/read-from identity unobserved",
            )?,
        })
    }
}

/// Endpoint-specific NCU projection over simulator frontiers.  Missing
/// padding deliberately leaves predicted bounds unknown rather than silently
/// equating simulator completion with NCU completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcuObserverProjection {
    pub schema_version: u32,
    pub role: &'static str,
    pub target_metric: &'static str,
    pub simulator_lower_cycles: Option<u64>,
    pub simulator_upper_cycles: Option<u64>,
    pub predicted_lower_cycles: Option<u64>,
    pub predicted_upper_cycles: Option<u64>,
    pub start_padding_cycles: Option<ServiceCycleInterval>,
    pub end_padding_cycles: Option<ServiceCycleInterval>,
    pub conversion: &'static str,
}

impl NcuObserverProjection {
    pub fn from_frontiers(
        frontiers: FrontierVector,
        profile: Option<&ServiceRateProfile>,
    ) -> Result<Self, LatentModelError> {
        frontiers.validate()?;
        if let Some(profile) = profile {
            profile.validate()?;
        }
        let endpoint_fields_present = frontiers.endpoint_start.lower.is_some()
            || frontiers.endpoint_start.upper.is_some()
            || frontiers.endpoint_end.lower.is_some()
            || frontiers.endpoint_end.upper.is_some();
        let simulator_completion = if endpoint_fields_present {
            // An explicit endpoint pair denotes elapsed duration. A known
            // endpoint must not hide an active unknown completion tail.
            let mut completion_end = frontiers.endpoint_end;
            let required = frontiers.simulator_completion();
            if let (Some(lower), Some(upper)) = (required.lower, completion_end.upper)
                && lower > upper
            {
                return Err(LatentModelError::FrontierOrder {
                    earlier: "required_completion",
                    later: "endpoint_end",
                });
            }
            completion_end.lower = required.lower;
            let active_unbounded = frontiers
                .completion_bounds()
                .any(|bound| bound.upper.is_none());
            if active_unbounded {
                completion_end.upper = None;
            }
            FrontierVector::elapsed(frontiers.endpoint_start, completion_end)?
        } else {
            frontiers.simulator_completion()
        };
        let simulator_lower_cycles = simulator_completion.lower;
        let simulator_upper_cycles = simulator_completion.upper;
        let start_padding_cycles = profile.and_then(|p| p.ncu_start_padding_cycles);
        let end_padding_cycles = profile.and_then(|p| p.ncu_end_padding_cycles);
        let predicted_lower_cycles = match (
            simulator_lower_cycles,
            start_padding_cycles,
            end_padding_cycles,
        ) {
            (Some(base), Some(start), Some(end)) => Some(
                base.checked_add(start.lower_cycles)
                    .and_then(|value| value.checked_add(end.lower_cycles))
                    .ok_or_else(|| LatentModelError::ResourceDemandOverflow {
                        pool: "ncu_observer_lower".to_owned(),
                    })?,
            ),
            _ => None,
        };
        let predicted_upper_cycles = match (
            simulator_upper_cycles,
            start_padding_cycles,
            end_padding_cycles,
        ) {
            (Some(base), Some(start), Some(end)) => Some(
                base.checked_add(start.upper_cycles)
                    .and_then(|value| value.checked_add(end.upper_cycles))
                    .ok_or_else(|| LatentModelError::ResourceDemandOverflow {
                        pool: "ncu_observer_upper".to_owned(),
                    })?,
            ),
            // Any active unresolved completion frontier has no finite upper
            // endpoint.  Explicit endpoint pairs do not override work that
            // is declared to occur before that endpoint.
            _ => None,
        };
        Ok(Self {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            role: "ncu_observer_elapsed_interval_not_hardware_endpoint_identity",
            target_metric: "gpc__cycles_elapsed.max",
            simulator_lower_cycles,
            simulator_upper_cycles,
            predicted_lower_cycles,
            predicted_upper_cycles,
            start_padding_cycles,
            end_padding_cycles,
            conversion: "declared_elapsed_cycle_domain_only_no_cuda_event_or_host_conversion",
        })
    }
}

impl CacheCandidateSummary {
    pub fn from_topology(
        decoded_global_lines: u64,
        cross_sm_lines: u64,
        cross_sm_read_write_lines: u64,
        cross_sm_write_write_lines: u64,
    ) -> Result<Self, LatentModelError> {
        Self::from_topology_with_witnesses(
            decoded_global_lines,
            cross_sm_lines,
            cross_sm_read_write_lines,
            cross_sm_write_write_lines,
            CacheWitnessDomains::unobserved()?,
        )
    }

    pub fn from_topology_with_witnesses(
        decoded_global_lines: u64,
        cross_sm_lines: u64,
        cross_sm_read_write_lines: u64,
        cross_sm_write_write_lines: u64,
        witnesses: CacheWitnessDomains,
    ) -> Result<Self, LatentModelError> {
        witnesses.validate()?;
        let read_from = if decoded_global_lines == 0 {
            CandidateDomain::unknown("no decoded global lines")?
        } else if cross_sm_read_write_lines != 0 || cross_sm_write_write_lines != 0 {
            CandidateDomain::from_values([
                "initial_or_prior_write",
                "same_sm_or_local_forwarding",
                "cross_sm_write_or_forwarding",
            ])?
        } else {
            CandidateDomain::from_values(["initial_or_prior_write", "same_sm_or_local_forwarding"])?
        };
        Ok(Self {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            role: "trace_topology_candidate_cache_state_not_hardware_receipt",
            decoded_global_lines,
            cross_sm_lines,
            cross_sm_read_write_lines,
            cross_sm_write_write_lines,
            read_from,
            owner: CandidateDomain::unknown("physical cache owner unobserved")?,
            hardware_completion: CandidateDomain::unknown("hardware completion unobserved")?,
            temporal_order: witnesses.temporal_order,
            fence_scope: witnesses.fence_scope,
            line_generation: witnesses.line_generation,
        })
    }

    /// Convert unresolved cache history into a separate conservative frontier.
    /// A shared line does not add a positive delay claim; it only removes a
    /// finite upper endpoint when owner/read-from/completion state is absent.
    pub fn history_frontier(&self) -> FrontierBound {
        if self.cross_sm_lines == 0 {
            FrontierBound::unknown()
        } else if self.owner.is_unknown()
            || self.read_from.is_unknown()
            || self.hardware_completion.is_unknown()
        {
            FrontierBound::lower_bound(0)
        } else {
            FrontierBound::exact(0)
        }
    }
}

impl ResourceDemandEnvelope {
    pub fn project_service_cycles(
        &self,
        profile: &ServiceRateProfile,
    ) -> Result<ResourceServiceProjection, LatentModelError> {
        self.validate()?;
        profile.validate()?;
        let mut pools = BTreeMap::new();
        let mut pool_capacities = BTreeMap::new();
        let mut lower_cycles = 0_u64;
        let mut upper_cycles = Some(0_u64);
        for (pool, demand) in &self.pools {
            let rate = profile
                .pools
                .get(pool)
                .ok_or_else(|| LatentModelError::MissingServiceRate { pool: pool.clone() })?;
            let lower_work = demand
                .lower
                .checked_mul(rate.initiation_interval_cycles)
                .ok_or_else(|| LatentModelError::ResourceDemandOverflow { pool: pool.clone() })?;
            let upper_work = demand
                .upper
                .checked_mul(rate.initiation_interval_cycles)
                .ok_or_else(|| LatentModelError::ResourceDemandOverflow { pool: pool.clone() })?;
            let lower_service = if rate.capacity_already_applied {
                lower_work
            } else {
                ceil_div(lower_work, u64::from(rate.capacity))?
            };
            let upper_service = if rate.capacity_already_applied {
                upper_work
            } else {
                ceil_div(upper_work, u64::from(rate.capacity))?
            };
            // A pool with no mandatory work contributes no mandatory startup
            // latency.  This keeps optional/zero-demand pools from inflating
            // the lower bound merely because the profile contains them.
            let lower_latency = (lower_service != 0)
                .then_some(rate.latency_lower_cycles)
                .unwrap_or(0);
            let upper_latency = (upper_service != 0)
                .then_some(rate.latency_upper_cycles)
                .unwrap_or(0);
            let lower = lower_service
                .checked_add(lower_latency)
                .ok_or_else(|| LatentModelError::ResourceDemandOverflow { pool: pool.clone() })?;
            let upper = upper_service
                .checked_add(upper_latency)
                .ok_or_else(|| LatentModelError::ResourceDemandOverflow { pool: pool.clone() })?;
            lower_cycles = lower_cycles.max(lower);
            if let Some(current) = upper_cycles {
                upper_cycles = Some(current.max(upper));
            }
            pools.insert(
                pool.clone(),
                ServiceCycleInterval {
                    lower_cycles: lower,
                    upper_cycles: upper,
                },
            );
            pool_capacities.insert(pool.clone(), rate.capacity);
        }
        if self.unclassified_events != 0 {
            // Unclassified records may use a missing/slow resource class. A
            // named-pool maximum is not a safe substitute for that unknown
            // route, so retain an open upper instead of returning false
            // precision (including the all-unclassified/empty-pool case).
            upper_cycles = None;
        }
        Ok(ResourceServiceProjection {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            role: "model_service_interval_not_silicon_timing",
            profile_source: profile.source.clone(),
            lower_cycles,
            upper_cycles,
            unclassified_events: self.unclassified_events,
            pools,
            pool_capacities,
            // Resource pools are alternative/parallel service frontiers in
            // this projection; admission-to-observer composition is selected
            // separately by `ServiceComposition`.
            composition: ServiceOverlapMode::ParallelPoolMax,
        })
    }

    /// Build a declared serial admission/service queue interval. The queue
    /// uses total dynamic-record count as its depth, while per-pool route
    /// assignment remains a marginal envelope rather than a jointly feasible
    /// schedule. This is the explicit model convention used by R4 probes.
    pub fn project_service_queue(
        &self,
        profile: &ServiceRateProfile,
        admission: FrontierBound,
    ) -> Result<ServiceQueueProjection, LatentModelError> {
        let service = self.project_service_cycles(profile)?;
        admission.validate()?;
        let completion_lower = match admission.lower {
            Some(base) => Some(base.checked_add(service.lower_cycles).ok_or_else(|| {
                LatentModelError::ResourceDemandOverflow {
                    pool: "queue_completion_lower".to_owned(),
                }
            })?),
            None => None,
        };
        let completion_upper = match (admission.upper, service.upper_cycles) {
            (Some(base), Some(service)) => Some(base.checked_add(service).ok_or_else(|| {
                LatentModelError::ResourceDemandOverflow {
                    pool: "queue_completion_upper".to_owned(),
                }
            })?),
            _ => None,
        };
        let completion_frontier = FrontierBound {
            lower: completion_lower,
            upper: completion_upper,
        };
        completion_frontier.validate()?;
        Ok(ServiceQueueProjection {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            role: "serial_admission_service_queue_model_not_hardware_schedule",
            admission,
            issue_frontier: admission,
            completion_frontier,
            queue_depth: FrontierBound::exact(self.total_events),
            demand_events: self.total_events,
            unclassified_events: self.unclassified_events,
            assignment_semantics: "marginal_pool_intervals_not_joint_route_assignment",
        })
    }
}

fn ceil_div(numerator: u64, denominator: u64) -> Result<u64, LatentModelError> {
    if denominator == 0 {
        return Err(LatentModelError::InvalidServiceRate {
            pool: "division".to_owned(),
        });
    }
    numerator
        .checked_add(denominator - 1)
        .map(|value| value / denominator)
        .ok_or_else(|| LatentModelError::ResourceDemandOverflow {
            pool: "division".to_owned(),
        })
}

/// One concrete explanation retained in a finite world set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentWorld {
    pub route: String,
    pub compiler: String,
    pub cache: String,
    pub timing: String,
}

impl LatentWorld {
    pub fn new(
        route: impl Into<String>,
        compiler: impl Into<String>,
        cache: impl Into<String>,
        timing: impl Into<String>,
    ) -> Result<Self, LatentModelError> {
        let world = Self {
            route: route.into(),
            compiler: compiler.into(),
            cache: cache.into(),
            timing: timing.into(),
        };
        for (factor, value) in [
            ("route", world.route.as_str()),
            ("compiler", world.compiler.as_str()),
            ("cache", world.cache.as_str()),
            ("timing", world.timing.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(LatentModelError::EmptyWorldField { factor });
            }
        }
        Ok(world)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentWorldSet {
    pub worlds: Vec<LatentWorld>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MustMay {
    /// Agreement inside the supplied hypothetical world set is not an
    /// observed silicon singleton.
    pub must: Option<String>,
    pub may: BTreeSet<String>,
    pub evidence: EvidenceKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentSummary {
    pub route: MustMay,
    pub compiler: MustMay,
    pub cache: MustMay,
    pub timing: MustMay,
}

impl LatentWorldSet {
    pub fn new(worlds: Vec<LatentWorld>) -> Result<Self, LatentModelError> {
        let set = Self { worlds };
        set.validate()?;
        Ok(set)
    }

    pub fn validate(&self) -> Result<(), LatentModelError> {
        if self.worlds.is_empty() {
            return Err(LatentModelError::EmptyWorldSet);
        }
        if self.worlds.len() > MAX_WORLDS {
            return Err(LatentModelError::WorldSetTooLarge);
        }
        for world in &self.worlds {
            for (factor, value) in [
                ("route", world.route.as_str()),
                ("compiler", world.compiler.as_str()),
                ("cache", world.cache.as_str()),
                ("timing", world.timing.as_str()),
            ] {
                if value.trim().is_empty() {
                    return Err(LatentModelError::EmptyWorldField { factor });
                }
            }
        }
        Ok(())
    }

    pub fn summarize(&self) -> Result<LatentSummary, LatentModelError> {
        self.validate()?;
        let summarize = |values: BTreeSet<String>| {
            let must = (values.len() == 1)
                .then(|| values.iter().next().cloned())
                .flatten();
            MustMay {
                must,
                may: values,
                evidence: EvidenceKind::Hypothesis,
            }
        };
        let mut route = BTreeSet::new();
        let mut compiler = BTreeSet::new();
        let mut cache = BTreeSet::new();
        let mut timing = BTreeSet::new();
        for world in &self.worlds {
            route.insert(world.route.clone());
            compiler.insert(world.compiler.clone());
            cache.insert(world.cache.clone());
            timing.insert(world.timing.clone());
        }
        Ok(LatentSummary {
            route: summarize(route),
            compiler: summarize(compiler),
            cache: summarize(cache),
            timing: summarize(timing),
        })
    }
}

/// A complete mechanism envelope for one event or aggregate observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanismEnvelope {
    pub schema_version: u32,
    pub event: EventKey,
    pub factors: LatentFactors,
    pub frontiers: FrontierVector,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<EndpointDescriptor>,
}

impl MechanismEnvelope {
    pub fn new(
        event: EventKey,
        factors: LatentFactors,
        frontiers: FrontierVector,
        endpoint: Option<EndpointDescriptor>,
    ) -> Result<Self, LatentModelError> {
        let envelope = Self {
            schema_version: LATENT_SCHEMA_VERSION,
            event,
            factors,
            frontiers,
            endpoint,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<(), LatentModelError> {
        if self.schema_version != LATENT_SCHEMA_VERSION {
            return Err(LatentModelError::UnsupportedSchema {
                found: self.schema_version,
                expected: LATENT_SCHEMA_VERSION,
            });
        }
        self.event.validate()?;
        self.factors.validate()?;
        self.frontiers.validate()?;
        if let Some(endpoint) = &self.endpoint {
            endpoint.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_domains_preserve_unknown_and_reject_empty_intersections() {
        let unknown = CandidateDomain::unknown("no route witness").unwrap();
        let constrained = unknown.restrict(["adu", "cbu"]).unwrap();
        assert_eq!(constrained.must_value(), None);
        assert_eq!(constrained.may_values().unwrap().len(), 2);
        assert!(matches!(
            constrained.restrict(["lsu"]),
            Err(LatentModelError::CandidateIntersectionEmpty)
        ));
        assert!(matches!(
            CandidateDomain::from_values(std::iter::empty::<String>()),
            Err(LatentModelError::EmptyCandidateDomain)
        ));
    }

    #[test]
    fn finite_worlds_compute_must_and_may_without_balancing_guesses() {
        let worlds = LatentWorldSet::new(vec![
            LatentWorld::new("adu", "abi_v8", "l1_hit", "device_timer").unwrap(),
            LatentWorld::new("cbu", "abi_v8", "l1_hit", "device_timer").unwrap(),
        ])
        .unwrap();
        let summary = worlds.summarize().unwrap();
        assert_eq!(summary.route.must, None);
        assert_eq!(summary.route.may.len(), 2);
        assert_eq!(summary.compiler.must.as_deref(), Some("abi_v8"));
        assert_eq!(summary.cache.must.as_deref(), Some("l1_hit"));
    }

    #[test]
    fn frontier_intersection_and_order_are_checked() {
        let issue = FrontierBound::exact(4);
        let issue = issue.intersect(FrontierBound::lower_bound(2)).unwrap();
        assert_eq!(issue, FrontierBound::exact(4));
        let mut vector = FrontierVector {
            issue,
            modeled_event: FrontierBound::upper_bound(3),
            ..FrontierVector::default()
        };
        assert!(matches!(
            vector.validate(),
            Err(LatentModelError::FrontierOrder { .. })
        ));
        vector.modeled_event = FrontierBound::lower_bound(4);
        assert!(vector.validate().is_ok());
    }

    #[test]
    fn unknown_typed_tail_prevents_finite_completion_upper_bound() {
        let vector = FrontierVector {
            issue: FrontierBound::exact(10),
            modeled_event: FrontierBound::exact(20),
            typed_tail: FrontierBound::lower_bound(20),
            ..FrontierVector::default()
        };
        let completion = vector.simulator_completion();
        assert_eq!(completion.lower, Some(20));
        assert_eq!(completion.upper, None);
        let observer = NcuObserverProjection::from_frontiers(vector, None).unwrap();
        assert_eq!(observer.simulator_lower_cycles, Some(20));
        assert_eq!(observer.simulator_upper_cycles, None);
        assert_eq!(observer.predicted_upper_cycles, None);
    }

    #[test]
    fn elapsed_duration_is_invariant_to_absolute_origin_shift() {
        let first = FrontierVector::elapsed(
            FrontierBound::interval(0, 10).unwrap(),
            FrontierBound::interval(100, 120).unwrap(),
        )
        .unwrap();
        let shifted = FrontierVector::elapsed(
            FrontierBound::interval(900, 910).unwrap(),
            FrontierBound::interval(1000, 1020).unwrap(),
        )
        .unwrap();
        assert_eq!(first, shifted);
        assert_eq!(first, FrontierBound::interval(90, 120).unwrap());
    }

    #[test]
    fn observer_projection_uses_elapsed_endpoint_pair_not_absolute_end() {
        let profile = ServiceRateProfile {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            source: "probe".to_owned(),
            pools: BTreeMap::from([(
                "alu".to_owned(),
                ServiceRate {
                    capacity: 1,
                    initiation_interval_cycles: 1,
                    capacity_already_applied: false,
                    latency_lower_cycles: 1,
                    latency_upper_cycles: 1,
                },
            )]),
            ncu_start_padding_cycles: Some(ServiceCycleInterval {
                lower_cycles: 0,
                upper_cycles: 0,
            }),
            ncu_end_padding_cycles: Some(ServiceCycleInterval {
                lower_cycles: 0,
                upper_cycles: 0,
            }),
            unmodeled_dimensions: Vec::new(),
        };
        let mut first = FrontierVector {
            endpoint_start: FrontierBound::exact(0),
            endpoint_end: FrontierBound::exact(100),
            ..FrontierVector::default()
        };
        let first_projection =
            NcuObserverProjection::from_frontiers(first, Some(&profile)).unwrap();
        first.endpoint_start = FrontierBound::exact(900);
        first.endpoint_end = FrontierBound::exact(1000);
        let shifted_projection =
            NcuObserverProjection::from_frontiers(first, Some(&profile)).unwrap();
        assert_eq!(first_projection.simulator_lower_cycles, Some(100));
        assert_eq!(first_projection.simulator_upper_cycles, Some(100));
        assert_eq!(first_projection, shifted_projection);
    }

    #[test]
    fn endpoint_identity_blocks_cross_domain_comparisons() {
        let raw = EndpointDescriptor::new(
            ClockDomain::DeviceTimer,
            "report_before",
            "report_after",
            vec!["qmd_body".to_owned()],
            vec!["host_submission".to_owned()],
            "wfi_serialized",
            "application_state",
            "external_lock_1350_10501",
        )
        .unwrap();
        let mut ncu = raw.clone();
        ncu.clock_domain = ClockDomain::NsightCompute;
        assert!(!raw.same_contract_as(&ncu));
        assert!(raw.same_contract_as(&raw));
    }

    #[test]
    fn event_and_envelope_validation_is_fail_closed() {
        let key = EventKey::new(1, 7, 0, 2, 4, 0x20, Some(1), Some(0), "native_trace").unwrap();
        let envelope = MechanismEnvelope::new(
            key,
            LatentFactors::default(),
            FrontierVector::default(),
            None,
        )
        .unwrap();
        let encoded = serde_json::to_vec(&envelope).unwrap();
        let decoded: MechanismEnvelope = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, envelope);
        assert!(matches!(
            EventKey::new(1, 0, 0, 0, 0, 0, None, None, "trace"),
            Err(LatentModelError::InvalidFunctionId(0))
        ));
    }

    #[test]
    fn resource_demand_rejects_inverted_and_overlarge_intervals() {
        let mut pools = BTreeMap::new();
        pools.insert("alu".to_owned(), ResourceDemandInterval::interval(2, 1));
        assert!(matches!(
            ResourceDemandEnvelope::new(2, 0, pools, EvidenceKind::Hypothesis),
            Err(LatentModelError::InvertedResourceDemand { .. })
        ));

        let mut pools = BTreeMap::new();
        pools.insert("alu".to_owned(), ResourceDemandInterval::interval(0, 3));
        assert!(matches!(
            ResourceDemandEnvelope::new(2, 0, pools, EvidenceKind::Hypothesis),
            Err(LatentModelError::ResourceDemandExceedsTotal { .. })
        ));
    }

    #[test]
    fn service_rate_profile_requires_positive_calibrated_parameters() {
        let mut pools = BTreeMap::new();
        pools.insert(
            "alu".to_owned(),
            ServiceRate {
                capacity: 1,
                initiation_interval_cycles: 1,
                capacity_already_applied: false,
                latency_lower_cycles: 1,
                latency_upper_cycles: 2,
            },
        );
        let profile = ServiceRateProfile {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            source: "independent_probe_v1".to_owned(),
            pools,
            ncu_start_padding_cycles: None,
            ncu_end_padding_cycles: None,
            unmodeled_dimensions: Vec::new(),
        };
        assert!(profile.validate().is_ok());
        let encoded = serde_json::to_vec(&profile).unwrap();
        let decoded: ServiceRateProfile = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, profile);

        let mut invalid = profile.clone();
        invalid.pools.get_mut("alu").unwrap().capacity = 0;
        assert!(matches!(
            invalid.validate(),
            Err(LatentModelError::InvalidServiceRate { .. })
        ));
    }

    #[test]
    fn resource_demand_projects_to_model_service_interval_without_route_selection() {
        let mut pools = BTreeMap::new();
        pools.insert("alu".to_owned(), ResourceDemandInterval::interval(2, 4));
        let demand = ResourceDemandEnvelope::new(4, 0, pools, EvidenceKind::Hypothesis).unwrap();
        let mut rates = BTreeMap::new();
        rates.insert(
            "alu".to_owned(),
            ServiceRate {
                capacity: 2,
                initiation_interval_cycles: 2,
                capacity_already_applied: false,
                latency_lower_cycles: 1,
                latency_upper_cycles: 3,
            },
        );
        let profile = ServiceRateProfile {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            source: "independent_probe_v1".to_owned(),
            pools: rates,
            ncu_start_padding_cycles: Some(ServiceCycleInterval {
                lower_cycles: 0,
                upper_cycles: 2,
            }),
            ncu_end_padding_cycles: Some(ServiceCycleInterval {
                lower_cycles: 1,
                upper_cycles: 3,
            }),
            unmodeled_dimensions: Vec::new(),
        };
        let projection = demand.project_service_cycles(&profile).unwrap();
        assert_eq!(projection.lower_cycles, 3);
        assert_eq!(projection.upper_cycles, Some(7));
        assert_eq!(projection.role, "model_service_interval_not_silicon_timing");
        let frontiers = FrontierVector {
            modeled_event: FrontierBound::interval(10, 12).unwrap(),
            ..FrontierVector::default()
        };
        let observer = NcuObserverProjection::from_frontiers(frontiers, Some(&profile)).unwrap();
        assert_eq!(observer.predicted_lower_cycles, Some(11));
        assert_eq!(observer.predicted_upper_cycles, Some(17));
    }

    #[test]
    fn capacity_adjusted_probe_slope_is_not_divided_by_capacity_again() {
        let demand = ResourceDemandEnvelope::new(
            4,
            0,
            BTreeMap::from([("alu".to_owned(), ResourceDemandInterval::interval(4, 4))]),
            EvidenceKind::Hypothesis,
        )
        .unwrap();
        let profile = ServiceRateProfile {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            source: "independent_throughput_probe_v2".to_owned(),
            pools: BTreeMap::from([(
                "alu".to_owned(),
                ServiceRate {
                    capacity: 2,
                    initiation_interval_cycles: 3,
                    capacity_already_applied: true,
                    latency_lower_cycles: 2,
                    latency_upper_cycles: 2,
                },
            )]),
            ncu_start_padding_cycles: None,
            ncu_end_padding_cycles: None,
            unmodeled_dimensions: Vec::new(),
        };
        let projection = demand.project_service_cycles(&profile).unwrap();
        assert_eq!(projection.lower_cycles, 14);
        assert_eq!(
            projection.pools["alu"],
            ServiceCycleInterval {
                lower_cycles: 14,
                upper_cycles: 14,
            }
        );
    }

    #[test]
    fn unresolved_unclassified_demand_never_collapses_to_zero_or_finite_upper() {
        let demand =
            ResourceDemandEnvelope::new(1, 1, BTreeMap::new(), EvidenceKind::Hypothesis).unwrap();
        let profile = ServiceRateProfile {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            source: "independent_probe_v1".to_owned(),
            pools: BTreeMap::from([(
                "alu".to_owned(),
                ServiceRate {
                    capacity: 1,
                    initiation_interval_cycles: 1,
                    capacity_already_applied: false,
                    latency_lower_cycles: 2,
                    latency_upper_cycles: 2,
                },
            )]),
            ncu_start_padding_cycles: None,
            ncu_end_padding_cycles: None,
            unmodeled_dimensions: Vec::new(),
        };
        let projection = demand.project_service_cycles(&profile).unwrap();
        assert_eq!(projection.lower_cycles, 0);
        assert_eq!(projection.upper_cycles, None);
    }

    #[test]
    fn zero_demand_does_not_pay_optional_pool_startup_latency() {
        let demand = ResourceDemandEnvelope::new(
            0,
            0,
            BTreeMap::from([(String::from("alu"), ResourceDemandInterval::exact(0))]),
            EvidenceKind::Hypothesis,
        )
        .unwrap();
        let profile = ServiceRateProfile {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            source: "independent_probe_v1".to_owned(),
            pools: BTreeMap::from([(
                "alu".to_owned(),
                ServiceRate {
                    capacity: 1,
                    initiation_interval_cycles: 4,
                    capacity_already_applied: false,
                    latency_lower_cycles: 7,
                    latency_upper_cycles: 9,
                },
            )]),
            ncu_start_padding_cycles: None,
            ncu_end_padding_cycles: None,
            unmodeled_dimensions: Vec::new(),
        };
        let projection = demand.project_service_cycles(&profile).unwrap();
        assert_eq!(projection.lower_cycles, 0);
        assert_eq!(projection.upper_cycles, Some(0));
        assert_eq!(
            projection.pools["alu"],
            ServiceCycleInterval {
                lower_cycles: 0,
                upper_cycles: 0,
            }
        );
    }

    #[test]
    fn explicit_endpoint_does_not_hide_active_unknown_service_visibility_or_writeback() {
        let profile = ServiceRateProfile {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            source: "independent_probe_v1".to_owned(),
            pools: BTreeMap::from([(
                "alu".to_owned(),
                ServiceRate {
                    capacity: 1,
                    initiation_interval_cycles: 1,
                    capacity_already_applied: false,
                    latency_lower_cycles: 1,
                    latency_upper_cycles: 1,
                },
            )]),
            ncu_start_padding_cycles: Some(ServiceCycleInterval {
                lower_cycles: 0,
                upper_cycles: 0,
            }),
            ncu_end_padding_cycles: Some(ServiceCycleInterval {
                lower_cycles: 0,
                upper_cycles: 0,
            }),
            unmodeled_dimensions: Vec::new(),
        };
        for field in ["resource", "visibility", "writeback"] {
            let mut vector = FrontierVector {
                endpoint_start: FrontierBound::exact(0),
                endpoint_end: FrontierBound::exact(20),
                ..FrontierVector::default()
            };
            match field {
                "resource" => vector.resource_service = FrontierBound::lower_bound(3),
                "visibility" => vector.visibility = FrontierBound::lower_bound(3),
                "writeback" => vector.writeback_service = FrontierBound::lower_bound(3),
                _ => unreachable!(),
            }
            let projection = NcuObserverProjection::from_frontiers(vector, Some(&profile)).unwrap();
            assert_eq!(projection.simulator_upper_cycles, None, "{field}");
            assert_eq!(projection.predicted_upper_cycles, None, "{field}");
        }
    }

    #[test]
    fn wholly_unknown_completion_field_is_open_only_when_declared_required() {
        let profile = ServiceRateProfile {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            source: "independent_probe_v1".to_owned(),
            pools: BTreeMap::from([(
                "alu".to_owned(),
                ServiceRate {
                    capacity: 1,
                    initiation_interval_cycles: 1,
                    capacity_already_applied: false,
                    latency_lower_cycles: 1,
                    latency_upper_cycles: 1,
                },
            )]),
            ncu_start_padding_cycles: Some(ServiceCycleInterval {
                lower_cycles: 0,
                upper_cycles: 0,
            }),
            ncu_end_padding_cycles: Some(ServiceCycleInterval {
                lower_cycles: 0,
                upper_cycles: 0,
            }),
            unmodeled_dimensions: Vec::new(),
        };
        let base = FrontierVector {
            endpoint_start: FrontierBound::exact(0),
            endpoint_end: FrontierBound::exact(20),
            ..FrontierVector::default()
        };
        let inactive = NcuObserverProjection::from_frontiers(base, Some(&profile)).unwrap();
        assert_eq!(inactive.simulator_upper_cycles, Some(20));
        let mut active = base;
        active.required_completion = [CompletionRequirement::Visibility].into_iter().collect();
        let active = NcuObserverProjection::from_frontiers(active, Some(&profile)).unwrap();
        assert_eq!(active.simulator_upper_cycles, None);
    }

    #[test]
    fn service_composition_makes_overlap_policy_explicit() {
        let admission = FrontierBound::exact(100);
        // Serial mode receives a duration; parallel mode receives an
        // already-originated absolute service frontier. The explicit mode
        // prevents silently adding and maxing values from different domains.
        let service = FrontierBound::interval(20, 30).unwrap();
        let serial =
            ServiceComposition::new(admission, service, ServiceOverlapMode::SerialAfterAdmission)
                .unwrap();
        let parallel = ServiceComposition::new(
            admission,
            FrontierBound::interval(110, 130).unwrap(),
            ServiceOverlapMode::ParallelPoolMax,
        )
        .unwrap();
        assert_eq!(
            serial.completion,
            FrontierBound::interval(120, 130).unwrap()
        );
        assert_eq!(
            parallel.completion,
            FrontierBound::interval(110, 130).unwrap()
        );
        assert_ne!(serial.completion, parallel.completion);
    }

    #[test]
    fn service_stage_role_prevents_full_trace_demand_from_being_silently_residual() {
        let admission = FrontierBound::exact(100);
        let service = FrontierBound::interval(20, 30).unwrap();
        let replacement = ServiceComposition::new_with_role(
            admission,
            service,
            ServiceOverlapMode::SerialAfterAdmission,
            ServiceStageRole::ReplacementTiming,
        )
        .unwrap();
        assert_eq!(replacement.role, ServiceStageRole::ReplacementTiming);
        assert_eq!(replacement.completion, service);
        let independent = ServiceComposition::new_with_role(
            admission,
            service,
            ServiceOverlapMode::SerialAfterAdmission,
            ServiceStageRole::IndependentPostReplay,
        )
        .unwrap();
        assert_eq!(independent.role, ServiceStageRole::IndependentPostReplay);
        assert_eq!(
            independent.completion,
            FrontierBound::interval(120, 130).unwrap()
        );
    }

    #[test]
    fn service_queue_projection_declares_depth_and_open_completion() {
        let demand = ResourceDemandEnvelope::new(
            3,
            1,
            BTreeMap::from([("alu".to_owned(), ResourceDemandInterval::interval(1, 3))]),
            EvidenceKind::Hypothesis,
        )
        .unwrap();
        let profile = ServiceRateProfile {
            schema_version: RESOURCE_DEMAND_SCHEMA_VERSION,
            source: "independent_probe_v1".to_owned(),
            pools: BTreeMap::from([(
                "alu".to_owned(),
                ServiceRate {
                    capacity: 1,
                    initiation_interval_cycles: 2,
                    capacity_already_applied: false,
                    latency_lower_cycles: 1,
                    latency_upper_cycles: 2,
                },
            )]),
            ncu_start_padding_cycles: None,
            ncu_end_padding_cycles: None,
            unmodeled_dimensions: Vec::new(),
        };
        let queue = demand
            .project_service_queue(&profile, FrontierBound::exact(10))
            .unwrap();
        assert_eq!(queue.issue_frontier, FrontierBound::exact(10));
        assert_eq!(queue.queue_depth, FrontierBound::exact(3));
        assert_eq!(queue.completion_frontier.upper, None);
        assert_eq!(
            queue.assignment_semantics,
            "marginal_pool_intervals_not_joint_route_assignment"
        );
    }

    #[test]
    fn cache_candidate_summary_keeps_owner_and_completion_unknown() {
        let summary = CacheCandidateSummary::from_topology(4, 2, 1, 0).unwrap();
        assert!(summary.owner.is_unknown());
        assert!(summary.hardware_completion.is_unknown());
        assert!(summary.temporal_order.is_unknown());
        assert!(summary.fence_scope.is_unknown());
        assert!(summary.line_generation.is_unknown());
        assert_eq!(summary.read_from.may_values().unwrap().len(), 3);
    }

    #[test]
    fn reordered_topology_stays_unknown_without_temporal_witness() {
        let witnesses = CacheWitnessDomains::unobserved().unwrap();
        let forward =
            CacheCandidateSummary::from_topology_with_witnesses(4, 2, 1, 0, witnesses.clone())
                .unwrap();
        let reordered =
            CacheCandidateSummary::from_topology_with_witnesses(4, 2, 1, 0, witnesses).unwrap();
        assert_eq!(forward.temporal_order, reordered.temporal_order);
        assert!(forward.temporal_order.is_unknown());
        assert!(forward.line_generation.is_unknown());
    }

    #[test]
    fn supplied_generation_and_fence_witnesses_do_not_promote_owner_or_completion() {
        let witnesses = CacheWitnessDomains {
            temporal_order: CandidateDomain::candidate("trace_order_v1").unwrap(),
            fence_scope: CandidateDomain::candidate("device_scope_fence").unwrap(),
            line_generation: CandidateDomain::candidate("generation_7").unwrap(),
        };
        let summary =
            CacheCandidateSummary::from_topology_with_witnesses(4, 2, 1, 0, witnesses).unwrap();
        assert_eq!(summary.temporal_order.must_value(), Some("trace_order_v1"));
        assert_eq!(summary.fence_scope.must_value(), Some("device_scope_fence"));
        assert_eq!(summary.line_generation.must_value(), Some("generation_7"));
        assert!(summary.owner.is_unknown());
        assert!(summary.hardware_completion.is_unknown());
    }

    #[test]
    fn trace_order_witness_is_only_warp_local() {
        let witnessed = CacheWitnessDomains::from_trace_observation(true, 0, 4, 0).unwrap();
        assert_eq!(
            witnessed.temporal_order.must_value(),
            Some("warp_local_instrumentation_order")
        );
        let incomplete = CacheWitnessDomains::from_trace_observation(true, 0, 4, 1).unwrap();
        assert!(incomplete.temporal_order.is_unknown());
    }

    #[test]
    fn generation_change_is_retained_as_a_witness_not_a_new_owner_claim() {
        let make = |generation: &str| CacheWitnessDomains {
            temporal_order: CandidateDomain::candidate("warp_local_order").unwrap(),
            fence_scope: CandidateDomain::candidate("none_observed").unwrap(),
            line_generation: CandidateDomain::candidate(generation).unwrap(),
        };
        let first =
            CacheCandidateSummary::from_topology_with_witnesses(2, 1, 1, 0, make("generation_1"))
                .unwrap();
        let second =
            CacheCandidateSummary::from_topology_with_witnesses(2, 1, 1, 0, make("generation_2"))
                .unwrap();
        assert_ne!(first.line_generation, second.line_generation);
        assert!(first.owner.is_unknown());
        assert!(second.hardware_completion.is_unknown());
    }

    #[test]
    fn topology_overlap_removes_only_the_cache_history_upper_bound() {
        let summary = CacheCandidateSummary::from_topology(4, 2, 1, 0).unwrap();
        assert_eq!(summary.history_frontier(), FrontierBound::lower_bound(0));
        let no_overlap = CacheCandidateSummary::from_topology(4, 0, 0, 0).unwrap();
        assert!(no_overlap.history_frontier().lower.is_none());
        assert!(no_overlap.history_frontier().upper.is_none());
    }
}
