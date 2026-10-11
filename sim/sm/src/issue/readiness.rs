use std::collections::{HashMap, HashSet};

use serde::Serialize;
use sim_mem::{AccessKind, CacheOpToken};

use super::{
    DependencyOrigin, IssueError, ReadinessCause, ReadinessDiagnostics, checked_u64_counter_add,
};

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessCounterError {
    #[error(
        "readiness diagnostic counter {field} overflowed while adding {addend} to {accumulated}"
    )]
    Overflow {
        field: &'static str,
        accumulated: u64,
        addend: u64,
    },
    #[error("memory-service candidate configuration differs at index {index}")]
    MemoryServiceCandidateMismatch { index: usize },
    #[error("memory-admission candidate configuration differs at index {index}")]
    MemoryAdmissionCandidateMismatch { index: usize },
}

pub(super) fn checked_readiness_add(
    field: &'static str,
    accumulated: u64,
    addend: u64,
) -> Result<u64, ReadinessCounterError> {
    accumulated
        .checked_add(addend)
        .ok_or(ReadinessCounterError::Overflow {
            field,
            accumulated,
            addend,
        })
}

pub(super) const fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

pub(super) const fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

pub(super) const fn is_zero_u16(value: &u16) -> bool {
    *value == 0
}

pub(super) const fn is_false(value: &bool) -> bool {
    !*value
}

/// Exclusive and overlapping readiness delay by causal simulator relation.
///
/// `ReadinessDiagnostics::limiting_cycles` is exclusive: one deterministic
/// cause owns each issued instruction's pre-ready delay. `constraint_exposure`
/// retains every active relation and may overlap, so its fields must not be
/// summed as elapsed kernel time. `eligible_not_selected` is the delay after a
/// warp's complete issue frontier was ready but before the scheduler selected
/// it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ReadinessCauseCycles {
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub instruction_front_end: u64,
    pub control_stall: u64,
    pub dependence_counter_read: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub dependence_counter_global_load_operand_read: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub dependence_counter_global_store_operand_read: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub dependence_counter_global_atomic_operand_read: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub dependence_counter_global_reduction_operand_read: u64,
    pub dependence_counter_non_memory_write: u64,
    pub dependence_counter_shared_write: u64,
    pub dependence_counter_l1tex: u64,
    pub dependence_counter_unknown_origin: u64,
    pub async_source: u64,
    pub async_destination_local: u64,
    pub async_destination_sm_service: u64,
    pub cta_barrier_rendezvous: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub pipeline_barrier: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub pipeline_branch: u64,
    pub pipeline_tensor: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub pipeline_sfu: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub pipeline_alu: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub memory_admission_queue: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub memory_issue_queue: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub memory_write_completion: u64,
    pub eligible_not_selected: u64,
}

impl ReadinessCauseCycles {
    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(instruction_front_end);
        add!(control_stall);
        add!(dependence_counter_read);
        add!(dependence_counter_global_load_operand_read);
        add!(dependence_counter_global_store_operand_read);
        add!(dependence_counter_global_atomic_operand_read);
        add!(dependence_counter_global_reduction_operand_read);
        add!(dependence_counter_non_memory_write);
        add!(dependence_counter_shared_write);
        add!(dependence_counter_l1tex);
        add!(dependence_counter_unknown_origin);
        add!(async_source);
        add!(async_destination_local);
        add!(async_destination_sm_service);
        add!(cta_barrier_rendezvous);
        add!(pipeline_barrier);
        add!(pipeline_branch);
        add!(pipeline_tensor);
        add!(pipeline_sfu);
        add!(pipeline_alu);
        add!(memory_admission_queue);
        add!(memory_issue_queue);
        add!(memory_write_completion);
        add!(eligible_not_selected);
        *self = next;
        Ok(())
    }
}

pub(super) fn observe_readiness_constraint(
    projected: &mut ReadinessDiagnostics,
    base: u64,
    until: u64,
    cause: ReadinessCause,
    ready_at: &mut u64,
    limiting: &mut Option<ReadinessCause>,
    tied: &mut u64,
) -> Result<(), IssueError> {
    if until > base {
        projected
            .constraint_exposure_cycles
            .add_cause(cause, until - base)?;
    }
    if until > *ready_at {
        *ready_at = until;
        *limiting = (until > base).then_some(cause);
        *tied = 0;
    } else if until == *ready_at && until > base {
        match *limiting {
            Some(existing) if existing != cause => {
                *tied = checked_u64_counter_add("tied readiness cause", *tied, 1)?;
                *limiting = Some(existing.min(cause));
            }
            None => *limiting = Some(cause),
            Some(_) => {}
        }
    }
    Ok(())
}

pub(super) fn dependency_origin(kind: AccessKind) -> DependencyOrigin {
    match kind {
        AccessKind::NonMemory | AccessKind::PredicatedOffMemory => DependencyOrigin::NonMemoryWrite,
        AccessKind::Shared { .. } => DependencyOrigin::SharedWrite,
        AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalBypass { .. }
        | AccessKind::GlobalAsyncCopy { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalStore { .. }
        | AccessKind::GlobalReduction { .. } => DependencyOrigin::L1Tex,
    }
}

pub(super) fn operand_read_dependency_origin(kind: AccessKind) -> DependencyOrigin {
    match kind {
        AccessKind::GlobalBypass {
            token: CacheOpToken::Atom,
            ..
        } => DependencyOrigin::GlobalAtomicOperandRead,
        AccessKind::GlobalBypass {
            token: CacheOpToken::Ldgsts,
            ..
        } => DependencyOrigin::Read,
        AccessKind::GlobalEligible { .. }
        | AccessKind::GlobalUntagged { .. }
        | AccessKind::GlobalBypass { .. } => DependencyOrigin::GlobalLoadOperandRead,
        AccessKind::GlobalStore { .. } => DependencyOrigin::GlobalStoreOperandRead,
        AccessKind::GlobalReduction { .. } => DependencyOrigin::GlobalReductionOperandRead,
        AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. }
        | AccessKind::GlobalAsyncCopy { .. } => DependencyOrigin::Read,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct AsyncReadinessDiagnostics {
    pub groups_committed: u64,
    pub groups_waited: u64,
    pub source_limited_groups: u64,
    pub destination_local_limited_groups: u64,
    pub destination_sm_service_limited_groups: u64,
    pub source_cycles: u64,
    pub destination_local_cycles: u64,
    pub destination_sm_service_cycles: u64,
    pub completion_hidden_before_wait_cycles: u64,
    /// Trace-derived cache-all LDGSTS destination work. This is an observer
    /// count, not an NCU store-wavefront claim and not a timing input.
    pub destination_groups_observed: u64,
    pub destination_wavefronts_observed: u64,
}

impl AsyncReadinessDiagnostics {
    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.async_groups.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(groups_committed);
        add!(groups_waited);
        add!(source_limited_groups);
        add!(destination_local_limited_groups);
        add!(destination_sm_service_limited_groups);
        add!(source_cycles);
        add!(destination_local_cycles);
        add!(destination_sm_service_cycles);
        add!(completion_hidden_before_wait_cycles);
        add!(destination_groups_observed);
        add!(destination_wavefronts_observed);
        *self = next;
        Ok(())
    }
}

/// Timing-neutral cross-warp asynchronous-copy issue observation.
///
/// Coissue fields describe effective `LDGSTS` records already selected by the
/// four SMSP schedulers in one transaction. Committed-group population fields
/// describe sibling-warp state before any selected instruction mutates its
/// owning group. None of these counters participates in issue selection,
/// readiness, hierarchy requests, destination service, or completion time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct AsyncIssueOverlapDiagnostics {
    pub coissue_cycles: u64,
    pub coissued_copies: u64,
    pub cache_all_coissued_copies: u64,
    pub bypass_coissued_copies: u64,
    pub mixed_policy_coissue_cycles: u64,
    /// Structural excess demand for a hypothetical one-acceptance-per-cycle
    /// interface. This is a copy count, not an elapsed-cycle prediction.
    pub single_acceptor_excess_copies: u64,
    pub maximum_coissue_width: u64,
    pub copies_with_other_warp_committed_groups: u64,
    pub other_warp_committed_groups_sum: u64,
    pub maximum_other_warp_committed_groups: u64,
}

impl AsyncIssueOverlapDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.coissue_cycles == 0 && self.copies_with_other_warp_committed_groups == 0
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.async_issue_overlap.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(coissue_cycles);
        add!(coissued_copies);
        add!(cache_all_coissued_copies);
        add!(bypass_coissued_copies);
        add!(mixed_policy_coissue_cycles);
        add!(single_acceptor_excess_copies);
        next.maximum_coissue_width = next.maximum_coissue_width.max(addend.maximum_coissue_width);
        add!(copies_with_other_warp_committed_groups);
        add!(other_warp_committed_groups_sum);
        next.maximum_other_warp_committed_groups = next
            .maximum_other_warp_committed_groups
            .max(addend.maximum_other_warp_committed_groups);
        *self = next;
        Ok(())
    }
}

/// Timing-free reconstruction of selected asynchronous-copy protocol events
/// in the producer's callback/channel order. This diagnostic never feeds the
/// issue scheduler, memory hierarchy, completion frontiers, or typed tails.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct AsyncInstrumentationOrderDiagnostics {
    pub selected_events: u64,
    pub issue_events: u64,
    pub commit_events: u64,
    pub wait_events: u64,
    pub issue_events_with_other_warp_committed_groups: u64,
    pub other_warp_committed_groups_sum: u64,
    pub maximum_other_warp_committed_groups: u64,
    pub protocol_contradictions: u64,
}

impl AsyncInstrumentationOrderDiagnostics {
    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.async_instrumentation_order.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(selected_events);
        add!(issue_events);
        add!(commit_events);
        add!(wait_events);
        add!(issue_events_with_other_warp_committed_groups);
        add!(other_warp_committed_groups_sum);
        next.maximum_other_warp_committed_groups = next
            .maximum_other_warp_committed_groups
            .max(addend.maximum_other_warp_committed_groups);
        add!(protocol_contradictions);
        *self = next;
        Ok(())
    }
}

/// Timing-neutral summary for one class of asynchronous-copy waits.
///
/// Every field is an additive observation over issued waits. `exposure_cycles`
/// is the required committed-prefix completion frontier relative to the time
/// the wait became the warp's current instruction; it never participates in
/// readiness or event timing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct AsyncWaitClassDiagnostics {
    pub wait_count: u64,
    pub immediate_sum: u64,
    pub committed_groups_before: u64,
    pub consumed_prefix_groups: u64,
    pub groups_remaining: u64,
    pub exposure_cycles: u64,
}

impl AsyncWaitClassDiagnostics {
    pub(super) fn add_checked(
        &mut self,
        addend: &Self,
        fields: [&'static str; 6],
    ) -> Result<(), ReadinessCounterError> {
        let next = Self {
            wait_count: checked_readiness_add(fields[0], self.wait_count, addend.wait_count)?,
            immediate_sum: checked_readiness_add(
                fields[1],
                self.immediate_sum,
                addend.immediate_sum,
            )?,
            committed_groups_before: checked_readiness_add(
                fields[2],
                self.committed_groups_before,
                addend.committed_groups_before,
            )?,
            consumed_prefix_groups: checked_readiness_add(
                fields[3],
                self.consumed_prefix_groups,
                addend.consumed_prefix_groups,
            )?,
            groups_remaining: checked_readiness_add(
                fields[4],
                self.groups_remaining,
                addend.groups_remaining,
            )?,
            exposure_cycles: checked_readiness_add(
                fields[5],
                self.exposure_cycles,
                addend.exposure_cycles,
            )?,
        };
        *self = next;
        Ok(())
    }
}

const PARTIAL_ASYNC_WAIT_FIELDS: [&str; 6] = [
    "readiness.async_wait_prefix.partial.wait_count",
    "readiness.async_wait_prefix.partial.immediate_sum",
    "readiness.async_wait_prefix.partial.committed_groups_before",
    "readiness.async_wait_prefix.partial.consumed_prefix_groups",
    "readiness.async_wait_prefix.partial.groups_remaining",
    "readiness.async_wait_prefix.partial.exposure_cycles",
];
const DRAINING_ASYNC_WAIT_FIELDS: [&str; 6] = [
    "readiness.async_wait_prefix.draining.wait_count",
    "readiness.async_wait_prefix.draining.immediate_sum",
    "readiness.async_wait_prefix.draining.committed_groups_before",
    "readiness.async_wait_prefix.draining.consumed_prefix_groups",
    "readiness.async_wait_prefix.draining.groups_remaining",
    "readiness.async_wait_prefix.draining.exposure_cycles",
];

/// Partial (`wait_group N`, `N > 0`) and draining (`wait_group 0`) wait
/// observations. The enclosing readiness field is allocated only when the
/// static program contains a nonzero wait immediate, preserving predecessor
/// wait-zero JSON byte-for-byte.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct AsyncWaitPrefixDiagnostics {
    pub partial: AsyncWaitClassDiagnostics,
    pub draining: AsyncWaitClassDiagnostics,
}

impl AsyncWaitPrefixDiagnostics {
    pub(super) fn observe(
        &mut self,
        immediate: u32,
        committed_groups_before: u64,
        consumed_prefix_groups: u64,
        exposure_cycles: u64,
    ) -> Result<(), ReadinessCounterError> {
        let observation = AsyncWaitClassDiagnostics {
            wait_count: 1,
            immediate_sum: u64::from(immediate),
            committed_groups_before,
            consumed_prefix_groups,
            groups_remaining: committed_groups_before - consumed_prefix_groups,
            exposure_cycles,
        };
        if immediate == 0 {
            self.draining
                .add_checked(&observation, DRAINING_ASYNC_WAIT_FIELDS)
        } else {
            self.partial
                .add_checked(&observation, PARTIAL_ASYNC_WAIT_FIELDS)
        }
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        next.partial
            .add_checked(&addend.partial, PARTIAL_ASYNC_WAIT_FIELDS)?;
        next.draining
            .add_checked(&addend.draining, DRAINING_ASYNC_WAIT_FIELDS)?;
        *self = next;
        Ok(())
    }
}

/// Cold, bypass-only per-copy observation. It is boxed out of the scheduler's
/// inline readiness object and never participates in issue or wait timing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct BypassAsyncCopyDiagnostics {
    pub copies_observed: u64,
    /// Compatibility projection: source cohorts multiplied by each copy's
    /// maximum bank multiplicity. Exact for the retained uniform cohorts.
    pub destination_wavefronts_observed: u64,
    /// Source cohorts multiplied by each copy's maximum bypass projection.
    /// Exact for the retained uniform cohorts.
    pub bypass_wavefronts_observed: u64,
    pub source_cohorts_observed: u64,
    pub destination_lines_observed: u64,
    pub max_bank_wavefronts_sum: u64,
    pub max_bypass_wavefronts_sum: u64,
    pub copy_source_spread_cycles: u64,
    pub per_copy_completion_cycles: u64,
    pub group_aggregate_excess_cycles: u64,
    pub topology_candidate_penalty_cycles: u64,
    pub topology_candidate_completion_cycles: u64,
    pub topology_candidate_later_cycles: u64,
    pub topology_candidate_earlier_cycles: u64,
    /// Timing-neutral destination-pipeline groups projected at a wait. The
    /// candidate fields below are sums across groups, never readiness inputs.
    pub pipeline_groups_observed: u64,
    /// Candidate completion measured from group commit for acceptance width 1.
    pub pipeline_q1_completion_cycles: u64,
    /// Candidate completion still exposed when the wait became current.
    pub pipeline_q1_exposure_cycles: u64,
    /// Candidate completion measured from group commit for acceptance width 2.
    pub pipeline_q2_completion_cycles: u64,
    /// Candidate completion still exposed when the wait became current.
    pub pipeline_q2_exposure_cycles: u64,
    /// Candidate completion measured from group commit for acceptance width 4.
    pub pipeline_q4_completion_cycles: u64,
    /// Candidate completion still exposed when the wait became current.
    pub pipeline_q4_exposure_cycles: u64,
    /// Candidate completion measured from group commit for acceptance width 8.
    pub pipeline_q8_completion_cycles: u64,
    /// Candidate completion still exposed when the wait became current.
    pub pipeline_q8_exposure_cycles: u64,
    /// Candidate completion measured from group commit for acceptance width 16.
    pub pipeline_q16_completion_cycles: u64,
    /// Candidate completion still exposed when the wait became current.
    pub pipeline_q16_exposure_cycles: u64,
    /// Groups with an exact trace-derived source-sector/cohort join.
    pub cohort_groups_observed: u64,
    pub cohort_copies_observed: u64,
    /// Source-sector-to-cohort memberships (a sector may count more than once).
    pub cohort_response_memberships_observed: u64,
    /// Sum of each cohort's first-to-last source-response span.
    pub cohort_response_span_cycles: u64,
    /// Sum of the frozen topology penalties across observed cohorts.
    pub cohort_topology_penalty_cycles: u64,
    /// Candidate completion measured from group commit.
    pub cohort_candidate_completion_cycles: u64,
    /// Candidate completion still exposed when the wait became current.
    pub cohort_candidate_exposure_cycles: u64,
    pub cohort_candidate_later_than_per_copy_cycles: u64,
    pub cohort_candidate_earlier_than_per_copy_cycles: u64,
    pub cohort_candidate_later_than_group_aggregate_cycles: u64,
    pub cohort_candidate_earlier_than_group_aggregate_cycles: u64,
    pub cohort_candidate_later_than_production_cycles: u64,
    pub cohort_candidate_earlier_than_production_cycles: u64,
    /// Groups with an exact cohort response envelope projected by the
    /// timing-neutral reservation-overlap observation.
    pub cohort_overlap_groups_observed: u64,
    /// Sum across copies of each physical response envelope,
    /// `max(last_response) - min(first_response)`.
    pub cohort_response_envelope_cycles: u64,
    /// Sum across cohorts of frozen topology work hidden before that cohort's
    /// first response, measured from the earliest response in its copy.
    pub cohort_topology_prearrival_hidden_cycles: u64,
    /// Sum across cohorts of the remaining frozen topology work hidden by that
    /// cohort's first-to-last response stream.
    pub cohort_topology_stream_hidden_cycles: u64,
    /// Sum across groups of the maximum residual topology exposure across
    /// cohorts and copies. This observation never participates in readiness.
    pub cohort_topology_residual_exposure_cycles: u64,
}

impl BypassAsyncCopyDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.copies_observed == 0
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let next = Self {
            copies_observed: checked_readiness_add(
                "readiness.bypass_async_copies.copies_observed",
                self.copies_observed,
                addend.copies_observed,
            )?,
            destination_wavefronts_observed: checked_readiness_add(
                "readiness.bypass_async_copies.destination_wavefronts_observed",
                self.destination_wavefronts_observed,
                addend.destination_wavefronts_observed,
            )?,
            bypass_wavefronts_observed: checked_readiness_add(
                "readiness.bypass_async_copies.bypass_wavefronts_observed",
                self.bypass_wavefronts_observed,
                addend.bypass_wavefronts_observed,
            )?,
            source_cohorts_observed: checked_readiness_add(
                "readiness.bypass_async_copies.source_cohorts_observed",
                self.source_cohorts_observed,
                addend.source_cohorts_observed,
            )?,
            destination_lines_observed: checked_readiness_add(
                "readiness.bypass_async_copies.destination_lines_observed",
                self.destination_lines_observed,
                addend.destination_lines_observed,
            )?,
            max_bank_wavefronts_sum: checked_readiness_add(
                "readiness.bypass_async_copies.max_bank_wavefronts_sum",
                self.max_bank_wavefronts_sum,
                addend.max_bank_wavefronts_sum,
            )?,
            max_bypass_wavefronts_sum: checked_readiness_add(
                "readiness.bypass_async_copies.max_bypass_wavefronts_sum",
                self.max_bypass_wavefronts_sum,
                addend.max_bypass_wavefronts_sum,
            )?,
            copy_source_spread_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.copy_source_spread_cycles",
                self.copy_source_spread_cycles,
                addend.copy_source_spread_cycles,
            )?,
            per_copy_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.per_copy_completion_cycles",
                self.per_copy_completion_cycles,
                addend.per_copy_completion_cycles,
            )?,
            group_aggregate_excess_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.group_aggregate_excess_cycles",
                self.group_aggregate_excess_cycles,
                addend.group_aggregate_excess_cycles,
            )?,
            topology_candidate_penalty_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.topology_candidate_penalty_cycles",
                self.topology_candidate_penalty_cycles,
                addend.topology_candidate_penalty_cycles,
            )?,
            topology_candidate_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.topology_candidate_completion_cycles",
                self.topology_candidate_completion_cycles,
                addend.topology_candidate_completion_cycles,
            )?,
            topology_candidate_later_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.topology_candidate_later_cycles",
                self.topology_candidate_later_cycles,
                addend.topology_candidate_later_cycles,
            )?,
            topology_candidate_earlier_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.topology_candidate_earlier_cycles",
                self.topology_candidate_earlier_cycles,
                addend.topology_candidate_earlier_cycles,
            )?,
            pipeline_groups_observed: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_groups_observed",
                self.pipeline_groups_observed,
                addend.pipeline_groups_observed,
            )?,
            pipeline_q1_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q1_completion_cycles",
                self.pipeline_q1_completion_cycles,
                addend.pipeline_q1_completion_cycles,
            )?,
            pipeline_q1_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q1_exposure_cycles",
                self.pipeline_q1_exposure_cycles,
                addend.pipeline_q1_exposure_cycles,
            )?,
            pipeline_q2_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q2_completion_cycles",
                self.pipeline_q2_completion_cycles,
                addend.pipeline_q2_completion_cycles,
            )?,
            pipeline_q2_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q2_exposure_cycles",
                self.pipeline_q2_exposure_cycles,
                addend.pipeline_q2_exposure_cycles,
            )?,
            pipeline_q4_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q4_completion_cycles",
                self.pipeline_q4_completion_cycles,
                addend.pipeline_q4_completion_cycles,
            )?,
            pipeline_q4_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q4_exposure_cycles",
                self.pipeline_q4_exposure_cycles,
                addend.pipeline_q4_exposure_cycles,
            )?,
            pipeline_q8_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q8_completion_cycles",
                self.pipeline_q8_completion_cycles,
                addend.pipeline_q8_completion_cycles,
            )?,
            pipeline_q8_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q8_exposure_cycles",
                self.pipeline_q8_exposure_cycles,
                addend.pipeline_q8_exposure_cycles,
            )?,
            pipeline_q16_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q16_completion_cycles",
                self.pipeline_q16_completion_cycles,
                addend.pipeline_q16_completion_cycles,
            )?,
            pipeline_q16_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.pipeline_q16_exposure_cycles",
                self.pipeline_q16_exposure_cycles,
                addend.pipeline_q16_exposure_cycles,
            )?,
            cohort_groups_observed: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_groups_observed",
                self.cohort_groups_observed,
                addend.cohort_groups_observed,
            )?,
            cohort_copies_observed: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_copies_observed",
                self.cohort_copies_observed,
                addend.cohort_copies_observed,
            )?,
            cohort_response_memberships_observed: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_response_memberships_observed",
                self.cohort_response_memberships_observed,
                addend.cohort_response_memberships_observed,
            )?,
            cohort_response_span_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_response_span_cycles",
                self.cohort_response_span_cycles,
                addend.cohort_response_span_cycles,
            )?,
            cohort_topology_penalty_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_topology_penalty_cycles",
                self.cohort_topology_penalty_cycles,
                addend.cohort_topology_penalty_cycles,
            )?,
            cohort_candidate_completion_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_candidate_completion_cycles",
                self.cohort_candidate_completion_cycles,
                addend.cohort_candidate_completion_cycles,
            )?,
            cohort_candidate_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_candidate_exposure_cycles",
                self.cohort_candidate_exposure_cycles,
                addend.cohort_candidate_exposure_cycles,
            )?,
            cohort_candidate_later_than_per_copy_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_candidate_later_than_per_copy_cycles",
                self.cohort_candidate_later_than_per_copy_cycles,
                addend.cohort_candidate_later_than_per_copy_cycles,
            )?,
            cohort_candidate_earlier_than_per_copy_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_candidate_earlier_than_per_copy_cycles",
                self.cohort_candidate_earlier_than_per_copy_cycles,
                addend.cohort_candidate_earlier_than_per_copy_cycles,
            )?,
            cohort_candidate_later_than_group_aggregate_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_candidate_later_than_group_aggregate_cycles",
                self.cohort_candidate_later_than_group_aggregate_cycles,
                addend.cohort_candidate_later_than_group_aggregate_cycles,
            )?,
            cohort_candidate_earlier_than_group_aggregate_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_candidate_earlier_than_group_aggregate_cycles",
                self.cohort_candidate_earlier_than_group_aggregate_cycles,
                addend.cohort_candidate_earlier_than_group_aggregate_cycles,
            )?,
            cohort_candidate_later_than_production_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_candidate_later_than_production_cycles",
                self.cohort_candidate_later_than_production_cycles,
                addend.cohort_candidate_later_than_production_cycles,
            )?,
            cohort_candidate_earlier_than_production_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_candidate_earlier_than_production_cycles",
                self.cohort_candidate_earlier_than_production_cycles,
                addend.cohort_candidate_earlier_than_production_cycles,
            )?,
            cohort_overlap_groups_observed: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_overlap_groups_observed",
                self.cohort_overlap_groups_observed,
                addend.cohort_overlap_groups_observed,
            )?,
            cohort_response_envelope_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_response_envelope_cycles",
                self.cohort_response_envelope_cycles,
                addend.cohort_response_envelope_cycles,
            )?,
            cohort_topology_prearrival_hidden_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_topology_prearrival_hidden_cycles",
                self.cohort_topology_prearrival_hidden_cycles,
                addend.cohort_topology_prearrival_hidden_cycles,
            )?,
            cohort_topology_stream_hidden_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_topology_stream_hidden_cycles",
                self.cohort_topology_stream_hidden_cycles,
                addend.cohort_topology_stream_hidden_cycles,
            )?,
            cohort_topology_residual_exposure_cycles: checked_readiness_add(
                "readiness.bypass_async_copies.cohort_topology_residual_exposure_cycles",
                self.cohort_topology_residual_exposure_cycles,
                addend.cohort_topology_residual_exposure_cycles,
            )?,
        };
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CtaBarrierDiagnostics {
    pub generations: u64,
    pub total_arrival_spread_cycles: u64,
    pub max_arrival_spread_cycles: u64,
    pub total_rendezvous_warp_cycles: u64,
    pub releases_after_participant_retirement: u64,
}

impl CtaBarrierDiagnostics {
    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.cta_barriers.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(generations);
        add!(total_arrival_spread_cycles);
        next.max_arrival_spread_cycles = next
            .max_arrival_spread_cycles
            .max(addend.max_arrival_spread_cycles);
        add!(total_rendezvous_warp_cycles);
        add!(releases_after_participant_retirement);
        *self = next;
        Ok(())
    }
}

/// Bounded schema-2 diagnostics for the SM-shared CTA-barrier issue server.
/// Timestamp sums and issue spans preserve first/last-generation information
/// without retaining an instruction- or cycle-sized event log.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct BarrierPipelineDiagnostics {
    pub instructions_issued: u64,
    pub issue_server_busy_cycles: u64,
    pub generations_completed: u64,
    pub generation_rearm_events: u64,
    pub first_issue_timestamp_sum: u64,
    pub last_issue_timestamp_sum: u64,
    pub total_generation_issue_span_cycles: u64,
    pub max_generation_issue_span_cycles: u64,
    pub first_arrival_timestamp_sum: u64,
    pub last_arrival_timestamp_sum: u64,
    pub total_generation_arrival_span_cycles: u64,
    pub max_generation_arrival_span_cycles: u64,
}

impl BarrierPipelineDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.instructions_issued == 0
            && self.issue_server_busy_cycles == 0
            && self.generations_completed == 0
            && self.generation_rearm_events == 0
            && self.first_issue_timestamp_sum == 0
            && self.last_issue_timestamp_sum == 0
            && self.total_generation_issue_span_cycles == 0
            && self.max_generation_issue_span_cycles == 0
            && self.first_arrival_timestamp_sum == 0
            && self.last_arrival_timestamp_sum == 0
            && self.total_generation_arrival_span_cycles == 0
            && self.max_generation_arrival_span_cycles == 0
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.barrier_pipeline.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(instructions_issued);
        add!(issue_server_busy_cycles);
        add!(generations_completed);
        add!(generation_rearm_events);
        add!(first_issue_timestamp_sum);
        add!(last_issue_timestamp_sum);
        add!(total_generation_issue_span_cycles);
        next.max_generation_issue_span_cycles = next
            .max_generation_issue_span_cycles
            .max(addend.max_generation_issue_span_cycles);
        add!(first_arrival_timestamp_sum);
        add!(last_arrival_timestamp_sum);
        add!(total_generation_arrival_span_cycles);
        next.max_generation_arrival_span_cycles = next
            .max_generation_arrival_span_cycles
            .max(addend.max_generation_arrival_span_cycles);
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ControlPipelineDiagnostics {
    pub branch_instructions_issued: u64,
    pub branch_resolution_cycles: u64,
}

impl ControlPipelineDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.branch_instructions_issued == 0 && self.branch_resolution_cycles == 0
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let next = Self {
            branch_instructions_issued: checked_readiness_add(
                "readiness.control_pipeline.branch_instructions_issued",
                self.branch_instructions_issued,
                addend.branch_instructions_issued,
            )?,
            branch_resolution_cycles: checked_readiness_add(
                "readiness.control_pipeline.branch_resolution_cycles",
                self.branch_resolution_cycles,
                addend.branch_resolution_cycles,
            )?,
        };
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ResultWritebackDiagnostics {
    pub fixed_over_l1tex_collisions: u64,
    pub l1tex_deferral_cycles: u64,
}

/// Issue-time interval carried by kernels that bracket a measured region with
/// `CS2R ..., SR_CLOCKLO`. This is a simulator diagnostic, not a hardware clock
/// emulation: it reports scheduler issue timestamps in the same first/last
/// boundary shape used by in-kernel CLOCK microbenchmarks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ClockIssueDiagnostics {
    pub reads_issued: u64,
    pub warps_with_interval: u64,
    pub maximum_first_to_last_issue_cycles: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub warps_with_three_reads: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub maximum_first_to_second_issue_cycles: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub maximum_second_to_third_issue_cycles: u64,
}

/// Trace-derived scalar-global memory work presented by issued warp
/// instructions. Ideal wavefronts use the exact sm_86 shape observed by the
/// memory-retirement campaign: ordinary sectors pack four per wavefront, while
/// ATOM/RED additionally preserve same-sector alias multiplicity. Unique 128 B
/// lines and exact atomic L2 requests remain separate work dimensions. Hardware may
/// emit extra output wavefronts on internal replay/conflict; those are not
/// inferable from lane addresses. This diagnostic activates no timing service.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct GlobalMemoryIssueDiagnostics {
    pub load_instructions: u64,
    pub store_instructions: u64,
    pub atomic_instructions: u64,
    pub reduction_instructions: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub load_operand_read_barrier_instructions: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub store_operand_read_barrier_instructions: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub atomic_operand_read_barrier_instructions: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub reduction_operand_read_barrier_instructions: u64,
    pub load_sectors: u64,
    pub store_sectors: u64,
    pub atomic_sectors: u64,
    pub reduction_sectors: u64,
    pub load_unique_lines: u64,
    pub store_unique_lines: u64,
    pub atomic_unique_lines: u64,
    pub reduction_unique_lines: u64,
    pub atomic_l2_requests: u64,
    pub reduction_l2_requests: u64,
    pub load_ideal_wavefronts: u64,
    pub store_ideal_wavefronts: u64,
    pub atomic_ideal_wavefronts: u64,
    pub reduction_ideal_wavefronts: u64,
    pub first_issue_timestamp_sum: u64,
    pub last_issue_timestamp_sum: u64,
    pub total_issue_span_cycles: u64,
    pub max_issue_span_cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GlobalMemoryIssueClass {
    Load,
    Store,
    Atomic,
    Reduction,
}

fn maximum_sector_multiplicity(sectors: &[i64]) -> u64 {
    // This runs for every global issue candidate. Count aliases once instead
    // of rescanning the suffix for every sector; wide vector accesses stay
    // linear in their validated sector span.
    let mut counts = HashMap::new();
    let mut maximum = 0_u64;
    for &sector in sectors {
        let count = counts.entry(sector).or_insert(0_u64);
        *count += 1;
        maximum = maximum.max(*count);
    }
    maximum
}

fn unique_128b_lines(sectors: &[i64]) -> u64 {
    // Keep line uniqueness linear as well; the previous prefix scan became
    // quadratic on large coalesced spans.
    let mut lines = HashSet::new();
    for &sector in sectors {
        lines.insert(sector.div_euclid(4));
    }
    u64::try_from(lines.len()).unwrap_or(u64::MAX)
}

pub(super) fn projected_global_memory_issue(
    kind: AccessKind,
    sector_arena: &[i64],
) -> Option<(GlobalMemoryIssueClass, u64, u64, u64, u64)> {
    let atomic_l2_requests = kind.atomic_l2_requests().map(u64::from).unwrap_or(0);
    let (class, off, len, alias_aware) = match kind {
        AccessKind::GlobalEligible { off, len, .. } | AccessKind::GlobalUntagged { off, len } => {
            (GlobalMemoryIssueClass::Load, off, len, false)
        }
        AccessKind::GlobalBypass {
            off,
            len,
            token: CacheOpToken::Atom,
            ..
        } => (GlobalMemoryIssueClass::Atomic, off, len, true),
        AccessKind::GlobalBypass {
            token: CacheOpToken::Ldgsts,
            ..
        }
        | AccessKind::GlobalAsyncCopy { .. } => {
            // LDGSTS uses its own exact source/destination work domains.
            return None;
        }
        AccessKind::GlobalBypass { off, len, .. } => {
            (GlobalMemoryIssueClass::Load, off, len, false)
        }
        AccessKind::GlobalStore { off, len } => (GlobalMemoryIssueClass::Store, off, len, false),
        AccessKind::GlobalReduction { off, len, .. } => {
            (GlobalMemoryIssueClass::Reduction, off, len, true)
        }
        AccessKind::NonMemory
        | AccessKind::PredicatedOffMemory
        | AccessKind::Shared { .. }
        | AccessKind::LocalLoad { .. }
        | AccessKind::LocalStore { .. } => return None,
    };
    if len == 0 {
        return None;
    }
    let sectors = &sector_arena[off..off + usize::from(len)];
    let sector_count = u64::from(len);
    let packed = sector_count.div_ceil(4);
    let wavefronts = if alias_aware {
        packed.max(maximum_sector_multiplicity(sectors))
    } else {
        packed
    };
    Some((
        class,
        sector_count,
        wavefronts,
        unique_128b_lines(sectors),
        atomic_l2_requests,
    ))
}

/// L1TEX write lookup observer calibrated on an independent STORE cohort.
/// Each ideal wavefront allocates 64 B (two 32 B sectors) at lookup time; one
/// miss is charged per occupied 64 B lookup unit and the remaining sectors are
/// lookup hits. This is accounting only and never changes cache state/timing.
pub(super) fn projected_l1_write_lookup_hits(
    kind: AccessKind,
    sector_arena: &[i64],
) -> Option<u64> {
    let (class, sectors, wavefronts, _, _) = projected_global_memory_issue(kind, sector_arena)?;
    if !matches!(
        class,
        GlobalMemoryIssueClass::Store | GlobalMemoryIssueClass::Reduction
    ) {
        return None;
    }
    let sectors_per_wavefront = sectors.div_ceil(wavefronts);
    let misses = wavefronts
        .checked_mul(sectors_per_wavefront.div_ceil(2))?
        .min(sectors);
    Some(sectors - misses)
}

impl GlobalMemoryIssueDiagnostics {
    pub(super) const fn has_memory_work(&self) -> bool {
        self.load_instructions != 0
            || self.store_instructions != 0
            || self.atomic_instructions != 0
            || self.reduction_instructions != 0
    }

    pub const fn is_empty(&self) -> bool {
        !self.has_memory_work() && !self.has_operand_read_barriers()
    }

    pub const fn has_operand_read_barriers(&self) -> bool {
        self.load_operand_read_barrier_instructions != 0
            || self.store_operand_read_barrier_instructions != 0
            || self.atomic_operand_read_barrier_instructions != 0
            || self.reduction_operand_read_barrier_instructions != 0
    }

    pub(super) fn observe(
        &mut self,
        class: GlobalMemoryIssueClass,
        sectors: u64,
        wavefronts: u64,
        unique_lines: u64,
        atomic_l2_requests: u64,
        operand_read_barrier: bool,
        now: u64,
    ) -> Result<(), ReadinessCounterError> {
        let was_empty = !self.has_memory_work();
        let mut next = *self;
        macro_rules! add_class {
            ($prefix:literal, $instructions:ident, $sector_field:ident, $wavefront_field:ident) => {{
                next.$instructions = checked_readiness_add(
                    concat!("readiness.global_memory_issue.", $prefix, "_instructions"),
                    next.$instructions,
                    1,
                )?;
                next.$sector_field = checked_readiness_add(
                    concat!("readiness.global_memory_issue.", $prefix, "_sectors"),
                    next.$sector_field,
                    sectors,
                )?;
                next.$wavefront_field = checked_readiness_add(
                    concat!(
                        "readiness.global_memory_issue.",
                        $prefix,
                        "_ideal_wavefronts"
                    ),
                    next.$wavefront_field,
                    wavefronts,
                )?;
            }};
        }
        match class {
            GlobalMemoryIssueClass::Load => {
                add_class!(
                    "load",
                    load_instructions,
                    load_sectors,
                    load_ideal_wavefronts
                )
            }
            GlobalMemoryIssueClass::Store => add_class!(
                "store",
                store_instructions,
                store_sectors,
                store_ideal_wavefronts
            ),
            GlobalMemoryIssueClass::Atomic => add_class!(
                "atomic",
                atomic_instructions,
                atomic_sectors,
                atomic_ideal_wavefronts
            ),
            GlobalMemoryIssueClass::Reduction => add_class!(
                "reduction",
                reduction_instructions,
                reduction_sectors,
                reduction_ideal_wavefronts
            ),
        }
        if operand_read_barrier {
            let (field, counter) = match class {
                GlobalMemoryIssueClass::Load => (
                    "readiness.global_memory_issue.load_operand_read_barrier_instructions",
                    &mut next.load_operand_read_barrier_instructions,
                ),
                GlobalMemoryIssueClass::Store => (
                    "readiness.global_memory_issue.store_operand_read_barrier_instructions",
                    &mut next.store_operand_read_barrier_instructions,
                ),
                GlobalMemoryIssueClass::Atomic => (
                    "readiness.global_memory_issue.atomic_operand_read_barrier_instructions",
                    &mut next.atomic_operand_read_barrier_instructions,
                ),
                GlobalMemoryIssueClass::Reduction => (
                    "readiness.global_memory_issue.reduction_operand_read_barrier_instructions",
                    &mut next.reduction_operand_read_barrier_instructions,
                ),
            };
            *counter = checked_readiness_add(field, *counter, 1)?;
        }
        match class {
            GlobalMemoryIssueClass::Load => {
                next.load_unique_lines = checked_readiness_add(
                    "readiness.global_memory_issue.load_unique_lines",
                    next.load_unique_lines,
                    unique_lines,
                )?;
            }
            GlobalMemoryIssueClass::Store => {
                next.store_unique_lines = checked_readiness_add(
                    "readiness.global_memory_issue.store_unique_lines",
                    next.store_unique_lines,
                    unique_lines,
                )?;
            }
            GlobalMemoryIssueClass::Atomic => {
                next.atomic_unique_lines = checked_readiness_add(
                    "readiness.global_memory_issue.atomic_unique_lines",
                    next.atomic_unique_lines,
                    unique_lines,
                )?;
                next.atomic_l2_requests = checked_readiness_add(
                    "readiness.global_memory_issue.atomic_l2_requests",
                    next.atomic_l2_requests,
                    atomic_l2_requests,
                )?;
            }
            GlobalMemoryIssueClass::Reduction => {
                next.reduction_unique_lines = checked_readiness_add(
                    "readiness.global_memory_issue.reduction_unique_lines",
                    next.reduction_unique_lines,
                    unique_lines,
                )?;
                next.reduction_l2_requests = checked_readiness_add(
                    "readiness.global_memory_issue.reduction_l2_requests",
                    next.reduction_l2_requests,
                    atomic_l2_requests,
                )?;
            }
        }
        if was_empty {
            next.first_issue_timestamp_sum = now;
        }
        next.last_issue_timestamp_sum = now;
        let span = now.saturating_sub(next.first_issue_timestamp_sum);
        next.total_issue_span_cycles = span;
        next.max_issue_span_cycles = span;
        *self = next;
        Ok(())
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let mut next = *self;
        macro_rules! add {
            ($field:ident) => {
                next.$field = checked_readiness_add(
                    concat!("readiness.global_memory_issue.", stringify!($field)),
                    next.$field,
                    addend.$field,
                )?;
            };
        }
        add!(load_instructions);
        add!(store_instructions);
        add!(atomic_instructions);
        add!(reduction_instructions);
        add!(load_operand_read_barrier_instructions);
        add!(store_operand_read_barrier_instructions);
        add!(atomic_operand_read_barrier_instructions);
        add!(reduction_operand_read_barrier_instructions);
        add!(load_sectors);
        add!(store_sectors);
        add!(atomic_sectors);
        add!(reduction_sectors);
        add!(load_unique_lines);
        add!(store_unique_lines);
        add!(atomic_unique_lines);
        add!(reduction_unique_lines);
        add!(atomic_l2_requests);
        add!(reduction_l2_requests);
        add!(load_ideal_wavefronts);
        add!(store_ideal_wavefronts);
        add!(atomic_ideal_wavefronts);
        add!(reduction_ideal_wavefronts);
        add!(first_issue_timestamp_sum);
        add!(last_issue_timestamp_sum);
        add!(total_issue_span_cycles);
        next.max_issue_span_cycles = next.max_issue_span_cycles.max(addend.max_issue_span_cycles);
        *self = next;
        Ok(())
    }
}
