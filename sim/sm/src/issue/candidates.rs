//! Memory-service and admission candidate types; report-only scheduling inputs.
use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use super::is_gpu_membar_issue;
use super::readiness::{GlobalMemoryIssueClass, is_false, is_zero_u16, is_zero_u32, is_zero_u64};
use super::{
    AccessKind, AdmissionRequestWitness, IssueError, IssueSlot, MemoryOperation, MemoryWriteAck,
    ReadinessCounterError, SUB_CORES, checked_cycle_add, checked_readiness_add,
    checked_u64_counter_add, checked_usize_counter_add,
};

pub const MAX_MEMORY_SERVICE_CANDIDATES: usize = 16;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryServiceCandidateScope {
    #[default]
    Sm,
    Smsp,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryServiceCandidateOperation {
    #[default]
    Load,
    Store,
    Atomic,
    Reduction,
}

/// One opt-in streaming service hypothesis. A `(units, cycles)` pair of
/// `(0, 0)` disables that work dimension; otherwise both terms must be
/// positive. Service work is additive and rounded up per issued warp
/// instruction. `completion_latency_cycles` extends only the observed terminal
/// timestamp, not the server's next-accept horizon.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryServiceCandidate {
    pub candidate_id: u8,
    pub scope: MemoryServiceCandidateScope,
    pub operation: MemoryServiceCandidateOperation,
    pub ideal_wavefront_units: u32,
    pub ideal_wavefront_cycles: u32,
    pub unique_line_units: u32,
    pub unique_line_cycles: u32,
    pub completion_latency_cycles: u32,
}

impl MemoryServiceCandidate {
    pub fn validate(self) -> Result<(), MemoryServiceCandidateError> {
        let ideal_disabled = self.ideal_wavefront_units == 0 && self.ideal_wavefront_cycles == 0;
        let line_disabled = self.unique_line_units == 0 && self.unique_line_cycles == 0;
        if (self.ideal_wavefront_units == 0) != (self.ideal_wavefront_cycles == 0) {
            return Err(MemoryServiceCandidateError::InvalidRate {
                candidate_id: self.candidate_id,
                dimension: "ideal_wavefront",
                units: self.ideal_wavefront_units,
                cycles: self.ideal_wavefront_cycles,
            });
        }
        if (self.unique_line_units == 0) != (self.unique_line_cycles == 0) {
            return Err(MemoryServiceCandidateError::InvalidRate {
                candidate_id: self.candidate_id,
                dimension: "unique_line",
                units: self.unique_line_units,
                cycles: self.unique_line_cycles,
            });
        }
        if ideal_disabled && line_disabled {
            return Err(MemoryServiceCandidateError::NoServiceWork {
                candidate_id: self.candidate_id,
            });
        }
        Ok(())
    }

    pub(super) const fn matches(self, class: GlobalMemoryIssueClass) -> bool {
        matches!(
            (self.operation, class),
            (
                MemoryServiceCandidateOperation::Load,
                GlobalMemoryIssueClass::Load
            ) | (
                MemoryServiceCandidateOperation::Store,
                GlobalMemoryIssueClass::Store
            ) | (
                MemoryServiceCandidateOperation::Atomic,
                GlobalMemoryIssueClass::Atomic
            ) | (
                MemoryServiceCandidateOperation::Reduction,
                GlobalMemoryIssueClass::Reduction
            )
        )
    }
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum MemoryServiceCandidateError {
    #[error("candidate set has {count} entries; maximum is {maximum}")]
    TooMany { count: usize, maximum: usize },
    #[error("candidate id {candidate_id} is duplicated")]
    DuplicateId { candidate_id: u8 },
    #[error(
        "candidate {candidate_id} {dimension} rate must have both terms zero or both positive (units={units}, cycles={cycles})"
    )]
    InvalidRate {
        candidate_id: u8,
        dimension: &'static str,
        units: u32,
        cycles: u32,
    },
    #[error("candidate {candidate_id} enables no service-work dimension")]
    NoServiceWork { candidate_id: u8 },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct MemoryServiceCandidateResult {
    pub candidate: MemoryServiceCandidate,
    pub instructions_observed: u64,
    pub ideal_wavefronts_observed: u64,
    pub unique_lines_observed: u64,
    pub busy_cycles: u64,
    pub queue_delay_cycles: u64,
    pub terminal_completion_time: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryServiceCandidateDiagnostics {
    count: u8,
    entries: Option<Box<[MemoryServiceCandidateResult; MAX_MEMORY_SERVICE_CANDIDATES]>>,
}

impl Serialize for MemoryServiceCandidateDiagnostics {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.entries()[..].serialize(serializer)
    }
}

impl MemoryServiceCandidateDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn entries(&self) -> &[MemoryServiceCandidateResult] {
        self.entries
            .as_deref()
            .map_or(&[], |entries| &entries[..usize::from(self.count)])
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        if self.is_empty() {
            *self = addend.clone();
            return Ok(());
        }
        if addend.is_empty() {
            return Ok(());
        }
        if self.count != addend.count {
            return Err(ReadinessCounterError::MemoryServiceCandidateMismatch { index: 0 });
        }
        let mut next = self.clone();
        for index in 0..usize::from(self.count) {
            let current = self.entries()[index];
            let other = addend.entries()[index];
            if current.candidate != other.candidate {
                return Err(ReadinessCounterError::MemoryServiceCandidateMismatch { index });
            }
            let result = &mut next
                .entries
                .as_deref_mut()
                .expect("nonempty candidate diagnostics own entries")[index];
            macro_rules! add {
                ($field:ident) => {
                    result.$field = checked_readiness_add(
                        concat!("readiness.memory_service_candidates.", stringify!($field)),
                        result.$field,
                        other.$field,
                    )?;
                };
            }
            add!(instructions_observed);
            add!(ideal_wavefronts_observed);
            add!(unique_lines_observed);
            add!(busy_cycles);
            add!(queue_delay_cycles);
            result.terminal_completion_time = result
                .terminal_completion_time
                .max(other.terminal_completion_time);
        }
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct MemoryServiceCandidateRuntime {
    result: MemoryServiceCandidateResult,
    sm_ready: u64,
    smsp_ready: [u64; SUB_CORES],
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct MemoryServiceCandidateState {
    count: u8,
    entries: Option<Box<[MemoryServiceCandidateRuntime; MAX_MEMORY_SERVICE_CANDIDATES]>>,
}

fn candidate_rate_cycles(work: u64, units: u32, cycles: u32) -> Result<u64, IssueError> {
    if units == 0 {
        return Ok(0);
    }
    work.checked_mul(u64::from(cycles))
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "memory-service candidate work",
            lhs: work,
            rhs: u64::from(cycles),
        })
        .map(|scaled| scaled.div_ceil(u64::from(units)))
}

/// Price a fluid rational rate, optionally in fixed-width service quanta.
/// Width zero preserves the schema-7/9/10 legacy formula. A positive width
/// charges a complete service quantum for a partially occupied final batch,
/// making service width independently observable from asymptotic rate.
fn candidate_rate_cycles_with_width(
    work: u64,
    units: u32,
    cycles: u32,
    width: u32,
) -> Result<u64, IssueError> {
    if width == 0 {
        return candidate_rate_cycles(work, units, cycles);
    }
    let width = u64::from(width);
    let rounds = work.div_ceil(width);
    let round_cycles = candidate_rate_cycles(width, units, cycles)?;
    rounds
        .checked_mul(round_cycles)
        .ok_or(IssueError::TimeMultiplicationOverflow {
            operation: "memory-service candidate fixed-width rounds",
            lhs: rounds,
            rhs: round_cycles,
        })
}

impl MemoryServiceCandidateState {
    pub(super) const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub(super) fn checked(
        candidates: &[MemoryServiceCandidate],
    ) -> Result<Self, MemoryServiceCandidateError> {
        if candidates.len() > MAX_MEMORY_SERVICE_CANDIDATES {
            return Err(MemoryServiceCandidateError::TooMany {
                count: candidates.len(),
                maximum: MAX_MEMORY_SERVICE_CANDIDATES,
            });
        }
        let mut state = Self::default();
        if !candidates.is_empty() {
            state.entries = Some(Box::new(
                [MemoryServiceCandidateRuntime::default(); MAX_MEMORY_SERVICE_CANDIDATES],
            ));
        }
        for (index, &candidate) in candidates.iter().enumerate() {
            candidate.validate()?;
            if candidates[..index]
                .iter()
                .any(|prior| prior.candidate_id == candidate.candidate_id)
            {
                return Err(MemoryServiceCandidateError::DuplicateId {
                    candidate_id: candidate.candidate_id,
                });
            }
            state
                .entries
                .as_deref_mut()
                .expect("nonempty candidate set owns runtime entries")[index]
                .result
                .candidate = candidate;
        }
        state.count = candidates.len() as u8;
        Ok(state)
    }

    pub(super) fn diagnostics(&self) -> MemoryServiceCandidateDiagnostics {
        if self.is_empty() {
            return MemoryServiceCandidateDiagnostics::default();
        }
        let mut entries =
            Box::new([MemoryServiceCandidateResult::default(); MAX_MEMORY_SERVICE_CANDIDATES]);
        let runtime_entries = self
            .entries
            .as_deref()
            .expect("nonempty candidate state owns runtime entries");
        for (entry, runtime) in entries
            .iter_mut()
            .zip(runtime_entries.iter())
            .take(usize::from(self.count))
        {
            *entry = runtime.result;
        }
        MemoryServiceCandidateDiagnostics {
            count: self.count,
            entries: Some(entries),
        }
    }

    pub(super) fn observe(
        &mut self,
        subcore: usize,
        class: GlobalMemoryIssueClass,
        ideal_wavefronts: u64,
        unique_lines: u64,
        now: u64,
    ) -> Result<(), IssueError> {
        let entries = self
            .entries
            .as_deref_mut()
            .expect("observe is called only for a nonempty candidate state");
        for runtime in entries.iter_mut().take(usize::from(self.count)) {
            let candidate = runtime.result.candidate;
            if !candidate.matches(class) {
                continue;
            }
            let ideal_cycles = candidate_rate_cycles(
                ideal_wavefronts,
                candidate.ideal_wavefront_units,
                candidate.ideal_wavefront_cycles,
            )?;
            let line_cycles = candidate_rate_cycles(
                unique_lines,
                candidate.unique_line_units,
                candidate.unique_line_cycles,
            )?;
            let busy = checked_cycle_add(
                "memory-service candidate additive work",
                ideal_cycles,
                line_cycles,
            )?;
            let ready = match candidate.scope {
                MemoryServiceCandidateScope::Sm => runtime.sm_ready,
                MemoryServiceCandidateScope::Smsp => runtime.smsp_ready[subcore],
            };
            let start = now.max(ready);
            let done = checked_cycle_add("memory-service candidate completion", start, busy)?;
            match candidate.scope {
                MemoryServiceCandidateScope::Sm => runtime.sm_ready = done,
                MemoryServiceCandidateScope::Smsp => runtime.smsp_ready[subcore] = done,
            }
            let completion = checked_cycle_add(
                "memory-service candidate latency",
                done,
                u64::from(candidate.completion_latency_cycles),
            )?;
            let result = &mut runtime.result;
            result.instructions_observed = checked_u64_counter_add(
                "memory-service candidate instructions",
                result.instructions_observed,
                1,
            )?;
            result.ideal_wavefronts_observed = checked_u64_counter_add(
                "memory-service candidate ideal wavefronts",
                result.ideal_wavefronts_observed,
                ideal_wavefronts,
            )?;
            result.unique_lines_observed = checked_u64_counter_add(
                "memory-service candidate unique lines",
                result.unique_lines_observed,
                unique_lines,
            )?;
            result.busy_cycles = checked_u64_counter_add(
                "memory-service candidate busy cycles",
                result.busy_cycles,
                busy,
            )?;
            result.queue_delay_cycles = checked_u64_counter_add(
                "memory-service candidate queue delay",
                result.queue_delay_cycles,
                start - now,
            )?;
            result.terminal_completion_time = result.terminal_completion_time.max(completion);
        }
        Ok(())
    }
}

pub(crate) fn validate_memory_service_candidates(
    candidates: &[MemoryServiceCandidate],
) -> Result<(), MemoryServiceCandidateError> {
    MemoryServiceCandidateState::checked(candidates).map(|_| ())
}

pub const MAX_MEMORY_ADMISSION_CANDIDATES: usize = 16;
pub const MAX_MEMORY_ADMISSION_QUEUE_CAPACITY: u16 = 256;
pub const MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS: u32 = 256;
pub const MAX_WRITE_COMPLETION_PARALLEL_LANES: u16 = 32;

/// One observational finite input-queue hypothesis. Capacity counts waiting
/// warp instructions; the active instruction resides in the separate shared
/// downstream server. A queue slot is released at service start, and a same-
/// cycle dequeue may admit a new instruction. Service follows observed per-SM
/// issue order. Miss work may optionally use fixed-width service quanta; zero
/// width retains the legacy fluid rational rate. The observational evaluator
/// never changes scheduler or cache ordering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryAdmissionCandidate {
    pub candidate_id: u8,
    pub scope: MemoryServiceCandidateScope,
    pub operation: MemoryServiceCandidateOperation,
    pub queue_capacity_instructions: u16,
    pub admission_latency_cycles: u32,
    pub l2_hit_sector_units: u32,
    pub l2_hit_sector_cycles: u32,
    pub l2_miss_sector_units: u32,
    pub l2_miss_sector_cycles: u32,
    /// Zero retains fluid rational service. A positive value batches L2-miss
    /// sectors into fixed-width quanta and charges a full final quantum.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub l2_miss_service_width_sectors: u32,
    /// Opt-in completion lane: the issuing warp's `MEMBAR.SC.GPU` waits for
    /// its own latest matching write-service acknowledgement event. Queue
    /// admission still releases at service start.
    #[serde(default, skip_serializing_if = "is_false")]
    pub membar_waits_for_service_completion: bool,
    /// Opt-in terminal drain lane: a terminal `EXIT` waits for the issuing
    /// warp's modeled STORE/RED acknowledgement events even without an
    /// explicit MEMBAR. This reuses the typed acknowledgement stream and is
    /// deliberately separate from the MEMBAR stall domain.
    #[serde(default, skip_serializing_if = "is_false")]
    pub exit_waits_for_service_completion: bool,
    /// Opt-in SMSP/L1TEX admission-drain lane. A terminal EXIT waits only for
    /// the local store-admission/XBAR drain horizon, not L2 visibility.
    #[serde(default, skip_serializing_if = "is_false")]
    pub exit_waits_for_admission_drain: bool,
    /// Non-occupying visibility/RMW tail after service done. It changes only
    /// the per-warp MEMBAR completion frontier, never admission service.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub write_completion_latency_cycles: u32,
    /// Independent post-service RED/STORE completion work. Both dimensions are
    /// additive; zero disables a dimension.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub write_completion_request_cycles: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub write_completion_region_cycles: u32,
    #[serde(default, skip_serializing_if = "is_zero_u16")]
    pub write_completion_parallel_lanes: u16,
}

impl MemoryAdmissionCandidate {
    pub fn validate(self) -> Result<(), MemoryAdmissionCandidateError> {
        if !matches!(
            self.operation,
            MemoryServiceCandidateOperation::Store | MemoryServiceCandidateOperation::Reduction
        ) {
            return Err(MemoryAdmissionCandidateError::UnsupportedOperation {
                candidate_id: self.candidate_id,
                operation: self.operation,
            });
        }
        if self.queue_capacity_instructions == 0
            || self.queue_capacity_instructions > MAX_MEMORY_ADMISSION_QUEUE_CAPACITY
        {
            return Err(MemoryAdmissionCandidateError::CapacityOutOfRange {
                candidate_id: self.candidate_id,
                capacity: self.queue_capacity_instructions,
                maximum: MAX_MEMORY_ADMISSION_QUEUE_CAPACITY,
            });
        }
        if self.l2_miss_service_width_sectors > MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS {
            return Err(MemoryAdmissionCandidateError::ServiceWidthOutOfRange {
                candidate_id: self.candidate_id,
                width: self.l2_miss_service_width_sectors,
                maximum: MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS,
            });
        }
        if self.write_completion_latency_cycles != 0 && !self.membar_waits_for_service_completion {
            return Err(
                MemoryAdmissionCandidateError::CompletionLatencyWithoutMembar {
                    candidate_id: self.candidate_id,
                    latency: self.write_completion_latency_cycles,
                },
            );
        }
        if self.exit_waits_for_service_completion && !self.membar_waits_for_service_completion {
            return Err(MemoryAdmissionCandidateError::ExitCompletionWithoutAck {
                candidate_id: self.candidate_id,
            });
        }
        for (dimension, units, cycles) in [
            (
                "l2_hit_sector",
                self.l2_hit_sector_units,
                self.l2_hit_sector_cycles,
            ),
            (
                "l2_miss_sector",
                self.l2_miss_sector_units,
                self.l2_miss_sector_cycles,
            ),
        ] {
            if units == 0 || cycles == 0 {
                return Err(MemoryAdmissionCandidateError::InvalidRate {
                    candidate_id: self.candidate_id,
                    dimension,
                    units,
                    cycles,
                });
            }
        }
        let miss_scaled = u64::from(self.l2_miss_sector_cycles)
            .checked_mul(u64::from(self.l2_hit_sector_units))
            .expect("u32 cross product fits u64");
        let hit_scaled = u64::from(self.l2_hit_sector_cycles)
            .checked_mul(u64::from(self.l2_miss_sector_units))
            .expect("u32 cross product fits u64");
        if miss_scaled < hit_scaled {
            return Err(MemoryAdmissionCandidateError::MissRateFasterThanHit {
                candidate_id: self.candidate_id,
            });
        }
        let completion_work =
            self.write_completion_request_cycles != 0 || self.write_completion_region_cycles != 0;
        if completion_work && !self.membar_waits_for_service_completion {
            return Err(MemoryAdmissionCandidateError::CompletionWorkWithoutMembar {
                candidate_id: self.candidate_id,
            });
        }
        if completion_work
            && (self.write_completion_parallel_lanes == 0
                || self.write_completion_parallel_lanes > MAX_WRITE_COMPLETION_PARALLEL_LANES)
        {
            return Err(MemoryAdmissionCandidateError::CompletionLanesOutOfRange {
                candidate_id: self.candidate_id,
                lanes: self.write_completion_parallel_lanes,
                maximum: MAX_WRITE_COMPLETION_PARALLEL_LANES,
            });
        }
        if !completion_work && self.write_completion_parallel_lanes != 0 {
            return Err(MemoryAdmissionCandidateError::CompletionLanesWithoutWork {
                candidate_id: self.candidate_id,
                lanes: self.write_completion_parallel_lanes,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_feedback(self, issue_model_cycles: u64) -> Result<(), IssueError> {
        self.validate()?;
        if u64::from(self.admission_latency_cycles) != issue_model_cycles {
            return Err(IssueError::MemoryAdmissionFeedbackLatencyMismatch {
                candidate_id: self.candidate_id,
                configured_cycles: self.admission_latency_cycles,
                issue_model_cycles,
            });
        }
        Ok(())
    }

    pub(crate) const fn matches_operation(self, operation: MemoryOperation) -> bool {
        matches!(
            (self.operation, operation),
            (
                MemoryServiceCandidateOperation::Store,
                MemoryOperation::Store
            ) | (
                MemoryServiceCandidateOperation::Reduction,
                MemoryOperation::Reduction
            )
        )
    }
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAdmissionCandidateError {
    #[error("memory-admission candidate set has {count} entries; maximum is {maximum}")]
    TooMany { count: usize, maximum: usize },
    #[error("memory-admission candidate id {candidate_id} is duplicated")]
    DuplicateId { candidate_id: u8 },
    #[error(
        "memory-admission candidate {candidate_id} supports only store or reduction, got {operation:?}"
    )]
    UnsupportedOperation {
        candidate_id: u8,
        operation: MemoryServiceCandidateOperation,
    },
    #[error(
        "memory-admission candidate {candidate_id} queue capacity {capacity} is outside 1..={maximum}"
    )]
    CapacityOutOfRange {
        candidate_id: u8,
        capacity: u16,
        maximum: u16,
    },
    #[error(
        "memory-admission candidate {candidate_id} {dimension} rate must have positive terms (units={units}, cycles={cycles})"
    )]
    InvalidRate {
        candidate_id: u8,
        dimension: &'static str,
        units: u32,
        cycles: u32,
    },
    #[error("memory-admission candidate {candidate_id} prices an L2 miss faster than a hit")]
    MissRateFasterThanHit { candidate_id: u8 },
    #[error(
        "memory-admission candidate {candidate_id} L2-miss service width {width} exceeds maximum {maximum} sectors"
    )]
    ServiceWidthOutOfRange {
        candidate_id: u8,
        width: u32,
        maximum: u32,
    },
    #[error(
        "memory-admission candidate {candidate_id} declares write-completion latency {latency} without enabling MEMBAR completion feedback"
    )]
    CompletionLatencyWithoutMembar { candidate_id: u8, latency: u32 },
    #[error(
        "memory-admission candidate {candidate_id} declares write-completion work without enabling MEMBAR completion feedback"
    )]
    CompletionWorkWithoutMembar { candidate_id: u8 },
    #[error(
        "memory-admission candidate {candidate_id} write-completion lanes {lanes} are outside 1..={maximum}"
    )]
    CompletionLanesOutOfRange {
        candidate_id: u8,
        lanes: u16,
        maximum: u16,
    },
    #[error(
        "memory-admission candidate {candidate_id} declares {lanes} write-completion lanes without completion work"
    )]
    CompletionLanesWithoutWork { candidate_id: u8, lanes: u16 },
    #[error(
        "memory-admission candidate {candidate_id} enables EXIT completion without the typed acknowledgement lane"
    )]
    ExitCompletionWithoutAck { candidate_id: u8 },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct MemoryAdmissionCandidateResult {
    pub candidate: MemoryAdmissionCandidate,
    pub instructions_observed: u64,
    pub l2_hit_sectors_observed: u64,
    pub l2_miss_sectors_observed: u64,
    pub service_busy_cycles: u64,
    pub downstream_queue_delay_cycles: u64,
    pub queue_full_instructions: u64,
    pub total_admission_delay_cycles: u64,
    pub maximum_admission_delay_cycles: u64,
    pub maximum_warp_admission_delay_cycles: u64,
    pub terminal_operand_read_clear_time: u64,
    pub terminal_service_completion_time: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryAdmissionCandidateDiagnostics {
    count: u8,
    entries: Option<Box<[MemoryAdmissionCandidateResult; MAX_MEMORY_ADMISSION_CANDIDATES]>>,
}

impl Serialize for MemoryAdmissionCandidateDiagnostics {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.entries()[..].serialize(serializer)
    }
}

impl MemoryAdmissionCandidateDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn entries(&self) -> &[MemoryAdmissionCandidateResult] {
        self.entries
            .as_deref()
            .map_or(&[], |entries| &entries[..usize::from(self.count)])
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        if self.is_empty() {
            *self = addend.clone();
            return Ok(());
        }
        if addend.is_empty() {
            return Ok(());
        }
        if self.count != addend.count {
            return Err(ReadinessCounterError::MemoryAdmissionCandidateMismatch { index: 0 });
        }
        let mut next = self.clone();
        for index in 0..usize::from(self.count) {
            let current = self.entries()[index];
            let other = addend.entries()[index];
            if current.candidate != other.candidate {
                return Err(ReadinessCounterError::MemoryAdmissionCandidateMismatch { index });
            }
            let result = &mut next
                .entries
                .as_deref_mut()
                .expect("nonempty admission diagnostics own entries")[index];
            macro_rules! add {
                ($field:ident) => {
                    result.$field = checked_readiness_add(
                        concat!("readiness.memory_admission_candidates.", stringify!($field)),
                        result.$field,
                        other.$field,
                    )?;
                };
            }
            add!(instructions_observed);
            add!(l2_hit_sectors_observed);
            add!(l2_miss_sectors_observed);
            add!(service_busy_cycles);
            add!(downstream_queue_delay_cycles);
            add!(queue_full_instructions);
            add!(total_admission_delay_cycles);
            result.maximum_admission_delay_cycles = result
                .maximum_admission_delay_cycles
                .max(other.maximum_admission_delay_cycles);
            result.maximum_warp_admission_delay_cycles = result
                .maximum_warp_admission_delay_cycles
                .max(other.maximum_warp_admission_delay_cycles);
            result.terminal_operand_read_clear_time = result
                .terminal_operand_read_clear_time
                .max(other.terminal_operand_read_clear_time);
            result.terminal_service_completion_time = result
                .terminal_service_completion_time
                .max(other.terminal_service_completion_time);
        }
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct MemoryAdmissionCandidateRuntime {
    result: MemoryAdmissionCandidateResult,
    server_ready: u64,
    queue_service_starts: [VecDeque<u64>; SUB_CORES],
    warp_admission_delay: Vec<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MemoryAdmissionCandidateState {
    count: u8,
    entries: Option<Box<[MemoryAdmissionCandidateRuntime; MAX_MEMORY_ADMISSION_CANDIDATES]>>,
}

impl MemoryAdmissionCandidateState {
    pub(crate) fn checked(
        candidates: &[MemoryAdmissionCandidate],
    ) -> Result<Self, MemoryAdmissionCandidateError> {
        if candidates.len() > MAX_MEMORY_ADMISSION_CANDIDATES {
            return Err(MemoryAdmissionCandidateError::TooMany {
                count: candidates.len(),
                maximum: MAX_MEMORY_ADMISSION_CANDIDATES,
            });
        }
        let mut state = Self::default();
        if !candidates.is_empty() {
            state.entries = Some(Box::new(std::array::from_fn(|_| {
                MemoryAdmissionCandidateRuntime::default()
            })));
        }
        for (index, &candidate) in candidates.iter().enumerate() {
            candidate.validate()?;
            if candidates[..index]
                .iter()
                .any(|prior| prior.candidate_id == candidate.candidate_id)
            {
                return Err(MemoryAdmissionCandidateError::DuplicateId {
                    candidate_id: candidate.candidate_id,
                });
            }
            state
                .entries
                .as_deref_mut()
                .expect("nonempty admission candidate set owns runtime entries")[index]
                .result
                .candidate = candidate;
        }
        state.count = candidates.len() as u8;
        Ok(state)
    }

    pub(crate) fn diagnostics(&self) -> MemoryAdmissionCandidateDiagnostics {
        if self.count == 0 {
            return MemoryAdmissionCandidateDiagnostics::default();
        }
        let mut entries =
            Box::new([MemoryAdmissionCandidateResult::default(); MAX_MEMORY_ADMISSION_CANDIDATES]);
        let runtime = self
            .entries
            .as_deref()
            .expect("nonempty admission state owns runtime entries");
        for index in 0..usize::from(self.count) {
            entries[index] = runtime[index].result;
        }
        MemoryAdmissionCandidateDiagnostics {
            count: self.count,
            entries: Some(entries),
        }
    }

    pub(crate) fn observe(
        &mut self,
        subcore: usize,
        warp: usize,
        operation: MemoryOperation,
        l2_hit_sectors: u64,
        l2_miss_sectors: u64,
        now: u64,
    ) -> Result<(), IssueError> {
        let entries = self
            .entries
            .as_deref_mut()
            .expect("observe is called only for nonempty admission state");
        for runtime in &mut entries[..usize::from(self.count)] {
            let candidate = runtime.result.candidate;
            if !candidate.matches_operation(operation) {
                continue;
            }
            let hit_cycles = candidate_rate_cycles(
                l2_hit_sectors,
                candidate.l2_hit_sector_units,
                candidate.l2_hit_sector_cycles,
            )?;
            let miss_cycles = candidate_rate_cycles_with_width(
                l2_miss_sectors,
                candidate.l2_miss_sector_units,
                candidate.l2_miss_sector_cycles,
                candidate.l2_miss_service_width_sectors,
            )?;
            let busy = checked_cycle_add(
                "memory-admission candidate service work",
                hit_cycles,
                miss_cycles,
            )?;
            let queue_index = match candidate.scope {
                MemoryServiceCandidateScope::Sm => 0,
                MemoryServiceCandidateScope::Smsp => subcore,
            };
            let queue = &mut runtime.queue_service_starts[queue_index];
            let capacity = usize::from(candidate.queue_capacity_instructions);
            let released_at = if queue.len() == capacity {
                queue
                    .pop_front()
                    .expect("capacity-sized admission queue has an oldest service start")
            } else {
                0
            };
            let admitted_at = now.max(released_at);
            let service_start = admitted_at.max(runtime.server_ready);
            let service_done = checked_cycle_add(
                "memory-admission candidate service completion",
                service_start,
                busy,
            )?;
            let operand_clear = checked_cycle_add(
                "memory-admission candidate operand-read visibility",
                admitted_at,
                u64::from(candidate.admission_latency_cycles),
            )?;
            queue.push_back(service_start);
            debug_assert!(queue.len() <= capacity);
            runtime.server_ready = service_done;

            if runtime.warp_admission_delay.len() <= warp {
                runtime.warp_admission_delay.resize(warp + 1, 0);
            }
            let admission_delay = admitted_at - now;
            let warp_delay = checked_u64_counter_add(
                "memory-admission candidate per-warp delay",
                runtime.warp_admission_delay[warp],
                admission_delay,
            )?;
            runtime.warp_admission_delay[warp] = warp_delay;

            let result = &mut runtime.result;
            result.instructions_observed = checked_u64_counter_add(
                "memory-admission candidate instructions",
                result.instructions_observed,
                1,
            )?;
            result.l2_hit_sectors_observed = checked_u64_counter_add(
                "memory-admission candidate L2 hit sectors",
                result.l2_hit_sectors_observed,
                l2_hit_sectors,
            )?;
            result.l2_miss_sectors_observed = checked_u64_counter_add(
                "memory-admission candidate L2 miss sectors",
                result.l2_miss_sectors_observed,
                l2_miss_sectors,
            )?;
            result.service_busy_cycles = checked_u64_counter_add(
                "memory-admission candidate service busy cycles",
                result.service_busy_cycles,
                busy,
            )?;
            result.downstream_queue_delay_cycles = checked_u64_counter_add(
                "memory-admission candidate downstream queue delay",
                result.downstream_queue_delay_cycles,
                service_start - admitted_at,
            )?;
            result.queue_full_instructions = checked_u64_counter_add(
                "memory-admission candidate queue-full instructions",
                result.queue_full_instructions,
                u64::from(admission_delay != 0),
            )?;
            result.total_admission_delay_cycles = checked_u64_counter_add(
                "memory-admission candidate total admission delay",
                result.total_admission_delay_cycles,
                admission_delay,
            )?;
            result.maximum_admission_delay_cycles =
                result.maximum_admission_delay_cycles.max(admission_delay);
            result.maximum_warp_admission_delay_cycles =
                result.maximum_warp_admission_delay_cycles.max(warp_delay);
            result.terminal_operand_read_clear_time =
                result.terminal_operand_read_clear_time.max(operand_clear);
            result.terminal_service_completion_time =
                result.terminal_service_completion_time.max(service_done);
        }
        Ok(())
    }
}

pub(crate) fn validate_memory_admission_candidates(
    candidates: &[MemoryAdmissionCandidate],
) -> Result<(), MemoryAdmissionCandidateError> {
    MemoryAdmissionCandidateState::checked(candidates).map(|_| ())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct MemoryAdmissionFeedbackDiagnostics {
    pub candidate: MemoryAdmissionCandidate,
    pub instructions_admitted: u64,
    pub l2_hit_sectors_observed: u64,
    pub l2_miss_sectors_observed: u64,
    pub service_busy_cycles: u64,
    pub maximum_waiting_instructions: u64,
    pub terminal_service_completion_time: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub terminal_write_completion_time: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub write_completion_requests_observed: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub write_completion_regions_observed: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub write_completion_busy_cycles: u64,
}

/// Lifecycle of the opt-in modeled STORE/RED completion frontier. These are
/// simulator event timestamps, not silicon acknowledgements: the event makes
/// the already configured schema-11/12/13 completion relation participate in
/// shared-system ordering and terminal drain without claiming dirty writeback
/// or coherence completion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct MemoryWriteAckDiagnostics {
    pub acks_scheduled: u64,
    pub acks_processed: u64,
    pub maximum_pending_acks: u64,
    pub latest_issue_time: u64,
    pub latest_acceptance_time: u64,
    pub latest_service_completion_time: u64,
    pub latest_fence_visibility_time: u64,
    pub terminal_drain_time: u64,
}

impl MemoryWriteAckDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.acks_scheduled == 0 && self.acks_processed == 0
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        let next = Self {
            acks_scheduled: checked_readiness_add(
                "readiness.memory_write_ack.acks_scheduled",
                self.acks_scheduled,
                addend.acks_scheduled,
            )?,
            acks_processed: checked_readiness_add(
                "readiness.memory_write_ack.acks_processed",
                self.acks_processed,
                addend.acks_processed,
            )?,
            maximum_pending_acks: self.maximum_pending_acks.max(addend.maximum_pending_acks),
            latest_issue_time: self.latest_issue_time.max(addend.latest_issue_time),
            latest_acceptance_time: self
                .latest_acceptance_time
                .max(addend.latest_acceptance_time),
            latest_service_completion_time: self
                .latest_service_completion_time
                .max(addend.latest_service_completion_time),
            latest_fence_visibility_time: self
                .latest_fence_visibility_time
                .max(addend.latest_fence_visibility_time),
            terminal_drain_time: self.terminal_drain_time.max(addend.terminal_drain_time),
        };
        *self = next;
        Ok(())
    }
}

impl MemoryAdmissionFeedbackDiagnostics {
    pub const fn is_empty(&self) -> bool {
        self.candidate.queue_capacity_instructions == 0
    }

    pub(super) fn add_checked(&mut self, addend: &Self) -> Result<(), ReadinessCounterError> {
        if self.is_empty() {
            *self = *addend;
            return Ok(());
        }
        if addend.is_empty() {
            return Ok(());
        }
        if self.candidate != addend.candidate {
            return Err(ReadinessCounterError::MemoryAdmissionCandidateMismatch { index: 0 });
        }
        let next = Self {
            candidate: self.candidate,
            instructions_admitted: checked_readiness_add(
                "readiness.memory_admission_feedback.instructions_admitted",
                self.instructions_admitted,
                addend.instructions_admitted,
            )?,
            l2_hit_sectors_observed: checked_readiness_add(
                "readiness.memory_admission_feedback.l2_hit_sectors_observed",
                self.l2_hit_sectors_observed,
                addend.l2_hit_sectors_observed,
            )?,
            l2_miss_sectors_observed: checked_readiness_add(
                "readiness.memory_admission_feedback.l2_miss_sectors_observed",
                self.l2_miss_sectors_observed,
                addend.l2_miss_sectors_observed,
            )?,
            service_busy_cycles: checked_readiness_add(
                "readiness.memory_admission_feedback.service_busy_cycles",
                self.service_busy_cycles,
                addend.service_busy_cycles,
            )?,
            maximum_waiting_instructions: self
                .maximum_waiting_instructions
                .max(addend.maximum_waiting_instructions),
            terminal_service_completion_time: self
                .terminal_service_completion_time
                .max(addend.terminal_service_completion_time),
            terminal_write_completion_time: self
                .terminal_write_completion_time
                .max(addend.terminal_write_completion_time),
            write_completion_requests_observed: checked_readiness_add(
                "readiness.memory_admission_feedback.write_completion_requests_observed",
                self.write_completion_requests_observed,
                addend.write_completion_requests_observed,
            )?,
            write_completion_regions_observed: checked_readiness_add(
                "readiness.memory_admission_feedback.write_completion_regions_observed",
                self.write_completion_regions_observed,
                addend.write_completion_regions_observed,
            )?,
            write_completion_busy_cycles: checked_readiness_add(
                "readiness.memory_admission_feedback.write_completion_busy_cycles",
                self.write_completion_busy_cycles,
                addend.write_completion_busy_cycles,
            )?,
        };
        *self = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MemoryAdmissionFeedbackEntry {
    warp: u32,
    instruction: u32,
    release_at: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MemoryAdmissionFeedbackIssue {
    pub(super) subcore: usize,
    pub(super) warp: usize,
    pub(super) instruction: usize,
    pub(super) kind: AccessKind,
    pub(super) prior_clock_reads: u64,
    pub(super) clock_read: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MemoryAdmissionFeedbackIssuePlan {
    reservations: [Option<(usize, MemoryAdmissionFeedbackEntry)>; SUB_CORES],
    second_clock_warps: [Option<usize>; SUB_CORES],
    instructions_admitted: u64,
    maximum_waiting_instructions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemoryAdmissionFeedbackState {
    pub(super) candidate: MemoryAdmissionCandidate,
    queues: [VecDeque<MemoryAdmissionFeedbackEntry>; SUB_CORES],
    server_ready: u64,
    warp_completion_ready: Vec<u64>,
    warp_admission_drain_ready: Vec<u64>,
    warp_pending_write_acks: Vec<u32>,
    pending_write_acks: HashMap<(u32, u32), MemoryWriteAck>,
    completion_lane_ready: Vec<u64>,
    clock_second_issue: Vec<Option<u64>>,
    diagnostics: MemoryAdmissionFeedbackDiagnostics,
    write_ack_diagnostics: MemoryWriteAckDiagnostics,
    yield_requested: bool,
}

impl MemoryAdmissionFeedbackState {
    pub(crate) fn checked(
        candidate: MemoryAdmissionCandidate,
    ) -> Result<Self, MemoryAdmissionCandidateError> {
        candidate.validate()?;
        Ok(Self {
            candidate,
            queues: std::array::from_fn(|_| VecDeque::new()),
            server_ready: 0,
            warp_completion_ready: Vec::new(),
            warp_admission_drain_ready: Vec::new(),
            warp_pending_write_acks: Vec::new(),
            pending_write_acks: HashMap::new(),
            completion_lane_ready: vec![0; usize::from(candidate.write_completion_parallel_lanes)],
            clock_second_issue: Vec::new(),
            diagnostics: MemoryAdmissionFeedbackDiagnostics {
                candidate,
                ..MemoryAdmissionFeedbackDiagnostics::default()
            },
            write_ack_diagnostics: MemoryWriteAckDiagnostics::default(),
            yield_requested: false,
        })
    }

    pub(super) const fn queue_index(&self, subcore: usize) -> usize {
        match self.candidate.scope {
            MemoryServiceCandidateScope::Sm => 0,
            MemoryServiceCandidateScope::Smsp => subcore,
        }
    }

    pub(super) fn matches_kind(&self, kind: AccessKind) -> bool {
        matches!(
            (self.candidate.operation, kind),
            (
                MemoryServiceCandidateOperation::Store,
                AccessKind::GlobalStore { len: 1.., .. }
            ) | (
                MemoryServiceCandidateOperation::Reduction,
                AccessKind::GlobalReduction { len: 1.., .. }
            )
        )
    }

    pub(super) fn reap(&mut self, now: u64) {
        for queue in &mut self.queues {
            while queue
                .front()
                .and_then(|entry| entry.release_at)
                .is_some_and(|release| release <= now)
            {
                queue.pop_front();
            }
        }
    }

    pub(super) fn admission_ready_at(&self, subcore: usize, kind: AccessKind) -> Option<u64> {
        if !self.matches_kind(kind) {
            return Some(0);
        }
        let queue = &self.queues[self.queue_index(subcore)];
        if queue.len() < usize::from(self.candidate.queue_capacity_instructions) {
            return Some(0);
        }
        queue.front().and_then(|entry| entry.release_at)
    }

    pub(super) fn completion_ready_at(&self, warp: usize, slot: &IssueSlot) -> Option<u64> {
        if !self.candidate.membar_waits_for_service_completion || !is_gpu_membar_issue(slot) {
            return Some(0);
        }
        if self.has_pending_bounds_for_warp(warp)
            || self.warp_pending_write_acks.get(warp).copied().unwrap_or(0) != 0
        {
            return None;
        }
        Some(self.warp_completion_ready.get(warp).copied().unwrap_or(0))
    }

    /// Earliest timestamp at which a terminal EXIT may issue. `None` means
    /// that this warp still owns a pending store admission or a pending
    /// acknowledgement; the shared event loop will wake the SM when the
    /// typed acknowledgement is processed. No DRAM, dirty-writeback, or
    /// coherence timestamp is invented by this arm.
    pub(super) fn exit_ready_at(
        &self,
        warp: usize,
        slot: &IssueSlot,
        terminal: bool,
    ) -> Option<u64> {
        if !self.candidate.exit_waits_for_service_completion
            && !self.candidate.exit_waits_for_admission_drain
            || !terminal
            || !slot.opcode_text().trim_start().starts_with("EXIT")
        {
            return Some(0);
        }
        if self.has_pending_bounds_for_warp(warp) {
            return None;
        }
        if self.candidate.exit_waits_for_service_completion
            && self.warp_pending_write_acks.get(warp).copied().unwrap_or(0) != 0
        {
            return None;
        }
        let completion = self
            .candidate
            .exit_waits_for_service_completion
            .then(|| self.warp_completion_ready.get(warp).copied().unwrap_or(0))
            .unwrap_or(0);
        let drain = self
            .candidate
            .exit_waits_for_admission_drain
            .then(|| {
                self.warp_admission_drain_ready
                    .get(warp)
                    .copied()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        Some(completion.max(drain))
    }

    pub(super) fn second_clock_issue(&self, warp: usize) -> Option<u64> {
        self.clock_second_issue.get(warp).copied().flatten()
    }

    pub(super) fn plan_issue(
        &self,
        issues: &[Option<MemoryAdmissionFeedbackIssue>; SUB_CORES],
        now: u64,
    ) -> Result<MemoryAdmissionFeedbackIssuePlan, IssueError> {
        let active_lengths: [usize; SUB_CORES] = std::array::from_fn(|queue_index| {
            self.queues[queue_index]
                .iter()
                .filter(|entry| entry.release_at.is_none_or(|release| release > now))
                .count()
        });
        let mut additions = [0usize; SUB_CORES];
        let mut reservations = [None; SUB_CORES];
        let mut second_clock_warps = [None; SUB_CORES];
        let mut instructions_admitted = self.diagnostics.instructions_admitted;
        let mut maximum_waiting_instructions = self.diagnostics.maximum_waiting_instructions;
        for (position, issue) in issues.iter().copied().enumerate() {
            let Some(issue) = issue else { continue };
            if self.candidate.membar_waits_for_service_completion
                && issue.prior_clock_reads == 1
                && issue.clock_read
            {
                second_clock_warps[position] = Some(issue.warp);
            }
            if !self.matches_kind(issue.kind) {
                continue;
            }
            let queue_index = self.queue_index(issue.subcore);
            let next_length = active_lengths[queue_index]
                .checked_add(additions[queue_index])
                .and_then(|length| length.checked_add(1))
                .ok_or(IssueError::CounterOverflow {
                    counter: "feedback admission queue length",
                })?;
            if next_length > usize::from(self.candidate.queue_capacity_instructions) {
                return Err(IssueError::MemoryAdmissionFeedbackQueueFull {
                    subcore: issue.subcore,
                    time: now,
                });
            }
            let entry = MemoryAdmissionFeedbackEntry {
                warp: u32::try_from(issue.warp).map_err(|_| IssueError::CounterOverflow {
                    counter: "feedback admission warp",
                })?,
                instruction: u32::try_from(issue.instruction).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "feedback admission instruction",
                    }
                })?,
                release_at: None,
            };
            instructions_admitted = checked_u64_counter_add(
                "feedback admission instructions",
                instructions_admitted,
                1,
            )?;
            additions[queue_index] += 1;
            maximum_waiting_instructions = maximum_waiting_instructions.max(next_length as u64);
            reservations[position] = Some((queue_index, entry));
        }
        Ok(MemoryAdmissionFeedbackIssuePlan {
            reservations,
            second_clock_warps,
            instructions_admitted,
            maximum_waiting_instructions,
        })
    }

    pub(super) fn apply_issue(&mut self, plan: MemoryAdmissionFeedbackIssuePlan, now: u64) {
        self.reap(now);
        for warp in plan.second_clock_warps.into_iter().flatten() {
            if self.clock_second_issue.len() <= warp {
                self.clock_second_issue.resize(warp + 1, None);
            }
            self.clock_second_issue[warp] = Some(now);
        }
        let mut reserved = false;
        for (queue_index, entry) in plan.reservations.into_iter().flatten() {
            debug_assert!(
                self.queues[queue_index].len()
                    < usize::from(self.candidate.queue_capacity_instructions)
            );
            self.queues[queue_index].push_back(entry);
            reserved = true;
        }
        self.diagnostics.instructions_admitted = plan.instructions_admitted;
        self.diagnostics.maximum_waiting_instructions = plan.maximum_waiting_instructions;
        self.yield_requested |= reserved;
    }

    pub(super) fn permits_transaction_addition(
        &self,
        subcore: usize,
        kind: AccessKind,
        additions: &[usize; SUB_CORES],
        now: u64,
    ) -> bool {
        if !self.matches_kind(kind) {
            return true;
        }
        let capacity = usize::from(self.candidate.queue_capacity_instructions);
        let queue_index = self.queue_index(subcore);
        let occupied = self.queues[queue_index]
            .iter()
            .filter(|entry| entry.release_at.is_none_or(|release| release > now))
            .count();
        occupied + additions[queue_index] < capacity
    }

    pub(super) fn record_transaction_addition(
        &self,
        subcore: usize,
        kind: AccessKind,
        additions: &mut [usize; SUB_CORES],
    ) {
        if self.matches_kind(kind) {
            additions[self.queue_index(subcore)] += 1;
        }
    }

    pub(crate) fn complete(
        &mut self,
        witness: AdmissionRequestWitness,
        operation: MemoryOperation,
        hit_sectors: u64,
        miss_sectors: u64,
        l2_requests: u64,
        unique_regions: u64,
        issue_time: u64,
        now: u64,
    ) -> Result<Option<MemoryWriteAck>, IssueError> {
        if !self.candidate.matches_operation(operation) {
            return Ok(None);
        }
        let queue_index = self.queue_index(usize::from(witness.subcore()));
        let entry_index = self.queues[queue_index]
            .iter()
            .position(|entry| {
                entry.warp == witness.warp() && entry.instruction == witness.instruction()
            })
            .ok_or(IssueError::MemoryAdmissionFeedbackReservationMissing {
                warp: witness.warp(),
                instruction: witness.instruction(),
            })?;
        if self.queues[queue_index][entry_index].release_at.is_some() {
            return Err(IssueError::MemoryAdmissionFeedbackAlreadyPriced {
                warp: witness.warp(),
                instruction: witness.instruction(),
            });
        }
        let hit_cycles = candidate_rate_cycles(
            hit_sectors,
            self.candidate.l2_hit_sector_units,
            self.candidate.l2_hit_sector_cycles,
        )?;
        let miss_cycles = candidate_rate_cycles_with_width(
            miss_sectors,
            self.candidate.l2_miss_sector_units,
            self.candidate.l2_miss_sector_cycles,
            self.candidate.l2_miss_service_width_sectors,
        )?;
        let busy = checked_cycle_add("feedback admission service work", hit_cycles, miss_cycles)?;
        // The L2 request is the first externally observable completion point
        // for this feedback arm. A queued acknowledgement must never be
        // scheduled before the event that priced its request; doing so would
        // insert time travel into the shared event queue when delayed L1/L2
        // requests are processed after the candidate's local service clock.
        let service_start = issue_time.max(self.server_ready).max(now);
        let service_done =
            checked_cycle_add("feedback admission service completion", service_start, busy)?;
        let l2_hit_sectors_observed = checked_u64_counter_add(
            "feedback admission L2 hit sectors",
            self.diagnostics.l2_hit_sectors_observed,
            hit_sectors,
        )?;
        let l2_miss_sectors_observed = checked_u64_counter_add(
            "feedback admission L2 miss sectors",
            self.diagnostics.l2_miss_sectors_observed,
            miss_sectors,
        )?;
        let service_busy_cycles = checked_u64_counter_add(
            "feedback admission service busy cycles",
            self.diagnostics.service_busy_cycles,
            busy,
        )?;
        let terminal_service_completion_time = self
            .diagnostics
            .terminal_service_completion_time
            .max(service_done);
        let request_completion_cycles = l2_requests
            .checked_mul(u64::from(self.candidate.write_completion_request_cycles))
            .ok_or(IssueError::TimeMultiplicationOverflow {
                operation: "write-completion request work",
                lhs: l2_requests,
                rhs: u64::from(self.candidate.write_completion_request_cycles),
            })?;
        let region_completion_cycles = unique_regions
            .checked_mul(u64::from(self.candidate.write_completion_region_cycles))
            .ok_or(IssueError::TimeMultiplicationOverflow {
                operation: "write-completion region work",
                lhs: unique_regions,
                rhs: u64::from(self.candidate.write_completion_region_cycles),
            })?;
        let write_completion_busy = checked_cycle_add(
            "write-completion additive work",
            request_completion_cycles,
            region_completion_cycles,
        )?;
        let completion_lane = if write_completion_busy == 0 {
            None
        } else {
            self.completion_lane_ready
                .iter()
                .copied()
                .enumerate()
                .min_by_key(|&(index, ready)| (ready, index))
        };
        let write_service_done = if let Some((_, lane_ready)) = completion_lane {
            checked_cycle_add(
                "write-completion lane service",
                service_done.max(lane_ready),
                write_completion_busy,
            )?
        } else {
            service_done
        };
        let write_completion_done = if self.candidate.membar_waits_for_service_completion {
            checked_cycle_add(
                "feedback write completion visibility",
                write_service_done,
                u64::from(self.candidate.write_completion_latency_cycles),
            )?
        } else {
            service_done
        };
        let terminal_write_completion_time = if self.candidate.membar_waits_for_service_completion {
            self.diagnostics
                .terminal_write_completion_time
                .max(write_completion_done)
        } else {
            self.diagnostics.terminal_write_completion_time
        };
        let drain_sectors =
            hit_sectors
                .checked_add(miss_sectors)
                .ok_or(IssueError::CounterOverflow {
                    counter: "store admission drain sectors",
                })?;
        let queue_capacity = usize::from(self.candidate.queue_capacity_instructions);
        let queue_was_full = self.queues[queue_index].len() >= queue_capacity;
        let packet_cycles = drain_sectors.div_ceil(2);
        let drain_busy = if queue_was_full {
            u64::from(self.candidate.admission_latency_cycles)
                .checked_add(packet_cycles)
                .ok_or(IssueError::TimeMultiplicationOverflow {
                    operation: "store admission drain cycles",
                    lhs: u64::from(self.candidate.admission_latency_cycles),
                    rhs: packet_cycles,
                })?
        } else {
            0
        };
        let admission_drain_done =
            checked_cycle_add("store admission drain horizon", service_start, drain_busy)?;
        let completion_requests_observed = checked_u64_counter_add(
            "feedback write-completion requests",
            self.diagnostics.write_completion_requests_observed,
            if write_completion_busy == 0 {
                0
            } else {
                l2_requests
            },
        )?;
        let completion_regions_observed = checked_u64_counter_add(
            "feedback write-completion regions",
            self.diagnostics.write_completion_regions_observed,
            if write_completion_busy == 0 {
                0
            } else {
                unique_regions
            },
        )?;
        let completion_busy_cycles = checked_u64_counter_add(
            "feedback write-completion busy cycles",
            self.diagnostics.write_completion_busy_cycles,
            write_completion_busy,
        )?;
        let ack = self
            .candidate
            .membar_waits_for_service_completion
            .then_some(MemoryWriteAck {
                witness,
                operation,
                issue_at: issue_time,
                accepted_at: service_start,
                service_done_at: service_done,
                fence_visible_at: write_completion_done,
            });
        let pending_plan = if let Some(ack) = ack {
            let key = (witness.warp(), witness.instruction());
            if self.pending_write_acks.contains_key(&key) {
                return Err(IssueError::MemoryWriteAckAlreadyPending {
                    warp: witness.warp(),
                    instruction: witness.instruction(),
                });
            }
            let warp =
                usize::try_from(witness.warp()).map_err(|_| IssueError::CounterOverflow {
                    counter: "feedback completion warp index",
                })?;
            let required_warp_slots =
                checked_usize_counter_add("feedback completion warp slots", warp, 1)?;
            let warp_pending = self
                .warp_pending_write_acks
                .get(warp)
                .copied()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "pending modeled write acknowledgements per warp",
                })?;
            let acks_scheduled = checked_u64_counter_add(
                "modeled write acknowledgements scheduled",
                self.write_ack_diagnostics.acks_scheduled,
                1,
            )?;
            let pending_total = self.pending_write_acks.len().checked_add(1).ok_or(
                IssueError::CounterOverflow {
                    counter: "pending modeled write acknowledgements",
                },
            )?;
            let pending_total =
                u64::try_from(pending_total).map_err(|_| IssueError::CounterOverflow {
                    counter: "pending modeled write acknowledgements",
                })?;
            Some((
                key,
                ack,
                warp,
                required_warp_slots,
                warp_pending,
                acks_scheduled,
                pending_total,
            ))
        } else {
            None
        };

        self.queues[queue_index][entry_index].release_at = Some(service_start);
        self.server_ready = service_done;
        let drain_warp =
            usize::try_from(witness.warp()).map_err(|_| IssueError::CounterOverflow {
                counter: "admission drain warp index",
            })?;
        if self.warp_admission_drain_ready.len() <= drain_warp {
            self.warp_admission_drain_ready.resize(drain_warp + 1, 0);
        }
        self.warp_admission_drain_ready[drain_warp] =
            self.warp_admission_drain_ready[drain_warp].max(admission_drain_done);
        if let Some((lane, _)) = completion_lane {
            self.completion_lane_ready[lane] = write_service_done;
        }
        if let Some((
            key,
            ack,
            warp,
            required_warp_slots,
            warp_pending,
            acks_scheduled,
            pending_total,
        )) = pending_plan
        {
            if self.warp_pending_write_acks.len() <= warp {
                self.warp_pending_write_acks.resize(required_warp_slots, 0);
            }
            self.warp_pending_write_acks[warp] = warp_pending;
            let replaced = self.pending_write_acks.insert(key, ack);
            debug_assert!(replaced.is_none(), "write-ack insertion was preflighted");
            self.write_ack_diagnostics.acks_scheduled = acks_scheduled;
            self.write_ack_diagnostics.maximum_pending_acks = self
                .write_ack_diagnostics
                .maximum_pending_acks
                .max(pending_total);
            self.write_ack_diagnostics.latest_issue_time = self
                .write_ack_diagnostics
                .latest_issue_time
                .max(ack.issue_at);
            self.write_ack_diagnostics.latest_acceptance_time = self
                .write_ack_diagnostics
                .latest_acceptance_time
                .max(ack.accepted_at);
            self.write_ack_diagnostics.latest_service_completion_time = self
                .write_ack_diagnostics
                .latest_service_completion_time
                .max(ack.service_done_at);
            self.write_ack_diagnostics.latest_fence_visibility_time = self
                .write_ack_diagnostics
                .latest_fence_visibility_time
                .max(ack.fence_visible_at);
        }
        self.diagnostics.l2_hit_sectors_observed = l2_hit_sectors_observed;
        self.diagnostics.l2_miss_sectors_observed = l2_miss_sectors_observed;
        self.diagnostics.service_busy_cycles = service_busy_cycles;
        self.diagnostics.terminal_service_completion_time = terminal_service_completion_time;
        self.diagnostics.terminal_write_completion_time = terminal_write_completion_time;
        self.diagnostics.write_completion_requests_observed = completion_requests_observed;
        self.diagnostics.write_completion_regions_observed = completion_regions_observed;
        self.diagnostics.write_completion_busy_cycles = completion_busy_cycles;
        Ok(ack)
    }

    pub(crate) fn acknowledge_write(
        &mut self,
        ack: MemoryWriteAck,
        now: u64,
    ) -> Result<(), IssueError> {
        let key = (ack.witness.warp(), ack.witness.instruction());
        let expected = self.pending_write_acks.get(&key).copied().ok_or(
            IssueError::MemoryWriteAckNotPending {
                warp: ack.witness.warp(),
                instruction: ack.witness.instruction(),
            },
        )?;
        if expected != ack {
            return Err(IssueError::MemoryWriteAckMismatch {
                warp: ack.witness.warp(),
                instruction: ack.witness.instruction(),
            });
        }
        if now != ack.fence_visible_at {
            return Err(IssueError::MemoryWriteAckTimeMismatch {
                warp: ack.witness.warp(),
                instruction: ack.witness.instruction(),
                expected: ack.fence_visible_at,
                got: now,
            });
        }
        let warp =
            usize::try_from(ack.witness.warp()).map_err(|_| IssueError::CounterOverflow {
                counter: "write-ack warp index",
            })?;
        let pending = self.warp_pending_write_acks.get(warp).copied().unwrap_or(0);
        let next_pending = pending.checked_sub(1).ok_or(IssueError::CounterUnderflow {
            counter: "pending modeled write acknowledgements per warp",
        })?;
        let acks_processed = checked_u64_counter_add(
            "modeled write acknowledgements processed",
            self.write_ack_diagnostics.acks_processed,
            1,
        )?;
        let required_warp_slots =
            checked_usize_counter_add("write-ack completion warp slots", warp, 1)?;
        let completion_ready = self
            .warp_completion_ready
            .get(warp)
            .copied()
            .unwrap_or(0)
            .max(ack.fence_visible_at);

        let removed = self.pending_write_acks.remove(&key);
        debug_assert_eq!(removed, Some(ack), "write-ack removal was preflighted");
        self.warp_pending_write_acks[warp] = next_pending;
        if self.warp_completion_ready.len() <= warp {
            self.warp_completion_ready.resize(required_warp_slots, 0);
        }
        self.warp_completion_ready[warp] = completion_ready;
        self.write_ack_diagnostics.acks_processed = acks_processed;
        self.write_ack_diagnostics.terminal_drain_time =
            self.write_ack_diagnostics.terminal_drain_time.max(now);
        Ok(())
    }

    pub(super) fn has_pending_bounds(&self) -> bool {
        self.queues
            .iter()
            .flatten()
            .any(|entry| entry.release_at.is_none())
    }

    pub(super) fn has_pending_bounds_for_warp(&self, warp: usize) -> bool {
        let warp = u32::try_from(warp).ok();
        self.queues
            .iter()
            .flatten()
            .any(|entry| entry.release_at.is_none() && Some(entry.warp) == warp)
    }

    pub(super) fn has_pending_write_acks(&self) -> bool {
        !self.pending_write_acks.is_empty()
    }

    pub(super) fn pending_write_ack_count(&self) -> usize {
        self.pending_write_acks.len()
    }

    pub(super) fn take_yield_requested(&mut self) -> bool {
        std::mem::take(&mut self.yield_requested)
    }

    pub(crate) const fn diagnostics(&self) -> MemoryAdmissionFeedbackDiagnostics {
        self.diagnostics
    }

    pub(crate) const fn write_ack_diagnostics(&self) -> MemoryWriteAckDiagnostics {
        self.write_ack_diagnostics
    }
}
