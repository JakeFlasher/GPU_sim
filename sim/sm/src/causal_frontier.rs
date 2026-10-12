//! Resource-aware model-local frontier analysis for explicit replay
//! descriptors.
//!
//! This is an explicit descriptor-model executor, not a native SM executor or
//! causal proof. A producer supplies dependency and completion facts for a
//! descriptor model; the model executor may install a typed macro transition,
//! emits its effects, and resumes detailed descriptors. It refuses unknown
//! effects and does not claim equivalence to `SmCore` or silicon. Native state
//! refinement remains a later, separate contract.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use serde::Serialize;
use sim_arch::ArchId;
use sim_mem::MemRole;
use sim_semantics::{
    ExecutionEvidence, FidelityDecision, IntervalCertificate, ReifyReason, certify_interval,
};

use crate::program::{FunctionSchedule, OpClass, PipelineClass};

const SMSP_COUNT: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalFrontierInstruction {
    pub sequence: u64,
    pub physical_warp: i32,
    pub pipeline: PipelineClass,
    pub latency_cycles: u64,
    pub initiation_interval_cycles: u64,
    pub in_flight_capacity: u32,
    /// Event-cardinality metadata only. The recurrence owns one completion
    /// token; values other than one are never summarized and are reported as
    /// explicit exact-path reifications.
    pub completion_units: u32,
    pub control_clean: bool,
    pub dependency_free: bool,
    pub external_side_effect: bool,
}

impl CausalFrontierInstruction {
    /// Join one dynamic record to the simulator's public static schedule. The
    /// producer must supply completion and dependency facts that are absent from
    /// the static sidecar; no latency is inferred from an NCU aggregate.
    pub fn from_schedule(
        schedule: &FunctionSchedule,
        pc: u64,
        sequence: u64,
        physical_warp: i32,
        latency_cycles: u64,
        completion_units: u32,
        dependency_free: bool,
        external_side_effect: bool,
    ) -> Result<Self, CausalFrontierError> {
        // The recurrence below models the admitted SM86 four-SMSP topology.
        // Refuse other architecture schedules instead of silently applying
        // Ampere subcore placement and claiming cycle-equivalent progress.
        if !matches!(schedule.arch(), ArchId::Sm86 | ArchId::Sm120) {
            return Err(CausalFrontierError::UnsupportedArchitecture {
                arch: schedule.arch(),
            });
        }
        let slot = schedule
            .issue_slot(pc)
            .ok_or(CausalFrontierError::PcNotInSchedule { pc })?;
        // Static semantic classes are stronger than caller-provided flags.
        // CTA barriers and async-copy controls carry cross-warp state, and
        // memory roles mutate queues/caches that this recurrence does not
        // project. Dispatch-only slots are intentionally uncharacterized.
        if slot.op_class() != OpClass::Other {
            return Err(CausalFrontierError::UnsupportedOperationClass {
                pc,
                class: slot.op_class(),
            });
        }
        if !matches!(slot.mem_role(), MemRole::None) {
            return Err(CausalFrontierError::UnsupportedMemoryRole { pc });
        }
        if matches!(slot.pipeline_class(), PipelineClass::DispatchOnly) {
            return Err(CausalFrontierError::UnsupportedPipeline { pc });
        }
        if let PipelineClass::ScalarAlu {
            latency_cycles: expected,
            ..
        } = slot.pipeline_class()
            && latency_cycles != expected
        {
            return Err(CausalFrontierError::TimingMismatch {
                pc,
                expected,
                got: latency_cycles,
            });
        }
        let (initiation_interval_cycles, in_flight_capacity) = match slot.pipeline_class() {
            PipelineClass::DispatchOnly => (1, u32::MAX),
            PipelineClass::Branch {
                next_issue_interval_cycles,
            } => (next_issue_interval_cycles, u32::MAX),
            PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                ..
            } => (warp_initiation_interval_cycles, u32::MAX),
            PipelineClass::Tensor {
                initiation_interval_cycles,
            } => (initiation_interval_cycles, u32::MAX),
            PipelineClass::ScalarAlu {
                initiation_interval_cycles,
                in_flight_capacity,
                ..
            } => (initiation_interval_cycles, in_flight_capacity),
            PipelineClass::Sfu {
                initiation_interval_cycles,
            } => (initiation_interval_cycles, u32::MAX),
        };
        Ok(Self {
            sequence,
            physical_warp,
            pipeline: slot.pipeline_class(),
            latency_cycles,
            initiation_interval_cycles,
            in_flight_capacity,
            completion_units,
            control_clean: (schedule.arch() == ArchId::Sm120
                || slot.control().stall() == 0)
                && (schedule.arch() == ArchId::Sm120
                    || !slot.control().yield_flag())
                && slot.control().write_barrier().is_none()
                && slot.control().read_barrier().is_none()
                && slot.control().wait_mask().bits() == 0
                && slot.control().reuse() == 0,
            dependency_free,
            external_side_effect,
        })
    }

    fn summarizable(self) -> bool {
        self.execution_evidence().locally_summarizable()
    }

    fn execution_evidence(self) -> ExecutionEvidence {
        ExecutionEvidence {
            sequence: self.sequence,
            pipeline_key: pipeline_index(self.pipeline) as u8,
            latency_cycles: self.latency_cycles,
            initiation_interval_cycles: self.initiation_interval_cycles,
            in_flight_capacity: self.in_flight_capacity,
            completion_units: self.completion_units,
            control_clean: self.control_clean,
            dependency_free: self.dependency_free,
            external_side_effect: self.external_side_effect,
        }
    }
}

/// Resident population of one SM for the entire descriptor interval.
///
/// A complete roster is a producer-supplied premise, independent of the
/// candidate instruction list. It must include stalled/non-candidate warps.
/// Native producers must split at population changes or use `Unknown`; a
/// point-in-time snapshot alone does not exclude later admissions. This is
/// occupancy evidence for the descriptor model, not a native timing proof.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CausalFrontierResidency {
    #[default]
    Unknown,
    Complete {
        physical_warps: Vec<i32>,
    },
    /// Complete residency plus an independently derived witness for every
    /// interval eligible for the descriptor summary. The legacy `Complete`
    /// variant deliberately carries no dependency authority.
    CompleteWithEvidence {
        physical_warps: Vec<i32>,
        evidence: Vec<CausalFrontierEvidence>,
    },
}

/// Origin of an eligibility witness. `ProducerPremise` is retained for
/// diagnostics but is never sufficient for certification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalFrontierWitnessOrigin {
    StaticAnalysis,
    IndependentTraceAnalysis,
    ProducerPremise,
}

/// One fact derived outside the descriptor eligibility loop. The analyzer
/// checks these facts against the candidate instruction, so toggling a
/// descriptor flag without regenerating the witness forces exact fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CausalFrontierWitness {
    pub physical_warp: i32,
    pub sequence: u64,
    pub control_clean: bool,
    pub dependency_free: bool,
    pub external_side_effect: bool,
    pub completion_units: u32,
    pub origin: CausalFrontierWitnessOrigin,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CausalFrontierEvidence {
    pub physical_warp: i32,
    pub first_sequence: u64,
    pub last_sequence: u64,
    pub witnesses: Vec<CausalFrontierWitness>,
}

impl CausalFrontierEvidence {
    pub fn from_independent_facts(
        physical_warp: i32,
        first_sequence: u64,
        last_sequence: u64,
        witnesses: Vec<CausalFrontierWitness>,
    ) -> Result<Self, CausalFrontierError> {
        if first_sequence > last_sequence || witnesses.is_empty() {
            return Err(CausalFrontierError::InvalidEvidenceRange {
                physical_warp,
                first_sequence,
                last_sequence,
            });
        }
        let expected = last_sequence
            .checked_sub(first_sequence)
            .and_then(|width| width.checked_add(1))
            .ok_or(CausalFrontierError::InvalidEvidenceRange {
                physical_warp,
                first_sequence,
                last_sequence,
            })?;
        if u64::try_from(witnesses.len()).ok() != Some(expected)
            || witnesses.iter().enumerate().any(|(index, witness)| {
                witness.physical_warp != physical_warp
                    || witness.sequence != first_sequence.saturating_add(index as u64)
                    || witness.origin == CausalFrontierWitnessOrigin::ProducerPremise
            })
        {
            return Err(CausalFrontierError::IncompleteIndependentEvidence {
                physical_warp,
                first_sequence,
                last_sequence,
            });
        }
        Ok(Self {
            physical_warp,
            first_sequence,
            last_sequence,
            witnesses,
        })
    }

    fn covers(&self, instruction: CausalFrontierInstruction) -> bool {
        self.physical_warp == instruction.physical_warp
            && self.first_sequence <= instruction.sequence
            && instruction.sequence <= self.last_sequence
            && self.witnesses.iter().any(|witness| {
                witness.physical_warp == instruction.physical_warp
                    && witness.sequence == instruction.sequence
                    && witness.control_clean == instruction.control_clean
                    && witness.dependency_free == instruction.dependency_free
                    && witness.external_side_effect == instruction.external_side_effect
                    && witness.completion_units == instruction.completion_units
                    && witness.origin != CausalFrontierWitnessOrigin::ProducerPremise
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CausalFrontierReport {
    /// Canonical declared residency used for summary eligibility.
    pub residency: CausalFrontierResidency,
    pub exact_event_count: u64,
    pub causal_event_count: u64,
    pub exact_final_cycle: u64,
    pub causal_final_cycle: u64,
    pub critical_path_delta: i128,
    pub candidate_regions: u64,
    pub collapsed_regions: u64,
    pub reified_regions: u64,
    pub candidate_instructions: u64,
    pub collapsed_instructions: u64,
    pub exact_last_issue_cycle: Option<u64>,
    pub causal_last_issue_cycle: Option<u64>,
    pub exact_execution_steps: u64,
    pub causal_execution_steps: u64,
    /// Number of executable macro transitions installed by the selective
    /// model executor. This is a dispatch count, not a native-silicon claim.
    pub macro_dispatches: u64,
    pub terminal_state_matches: bool,
    /// Mechanism-local envelopes remain separate; summing them would not
    /// produce elapsed cycles when SMSPs overlap.
    pub certified_regions: Vec<CertifiedRegion>,
    pub reified_region_details: Vec<ReifiedRegion>,
    /// Runtime-only planner telemetry. It is deliberately excluded from the
    /// serialized report so repeated analyses remain byte-deterministic.
    #[serde(skip_serializing)]
    pub planner_time_ns: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CertifiedRegion {
    pub physical_warp: i32,
    pub first_sequence: u64,
    pub last_sequence: u64,
    pub first_issue_cycle: u64,
    pub last_issue_cycle: u64,
    pub completion_cycle: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ReifiedRegion {
    pub physical_warp: i32,
    pub first_sequence: u64,
    pub last_sequence: u64,
    pub reason: ReifyReason,
}

/// One observable effect emitted by the narrow scalar macro transition.
/// Effects are retained so a continuation can compare intermediate state, not
/// only the terminal completion cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MacroEffect {
    pub sequence: u64,
    pub physical_warp: i32,
    pub issue: u64,
    pub completion: u64,
}

/// Executable model transition for one homogeneous scalar region. The
/// transition is derived from an `IntervalCertificate`'s closed form, then
/// installed into `ModelExecutor`; detailed execution resumes at the next
/// descriptor after the transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MacroTransition {
    pub physical_warp: i32,
    pub first_sequence: u64,
    pub last_sequence: u64,
    pub instruction_count: u64,
    pub first_issue: u64,
    pub last_issue: u64,
    pub next_issue: u64,
    pub dispatch_ready: u64,
    pub last_completion: u64,
    pub pending_completions: Vec<u64>,
    pub effects: Vec<MacroEffect>,
}

impl MacroTransition {
    fn derive(
        first: CausalFrontierInstruction,
        certificate: IntervalCertificate,
        start: u64,
    ) -> Result<Self, ReifyReason> {
        if certificate.first_sequence() != first.sequence {
            return Err(ReifyReason::NonContiguousSequence);
        }
        let projection = certificate.project(start)?;
        let observations = certificate.issue_completion_times(start)?;
        let effects = observations
            .into_iter()
            .enumerate()
            .map(|(index, (issue, completion))| {
                let sequence = first
                    .sequence
                    .checked_add(u64::try_from(index).map_err(|_| ReifyReason::ArithmeticOverflow)?)
                    .ok_or(ReifyReason::ArithmeticOverflow)?;
                Ok(MacroEffect {
                    sequence,
                    physical_warp: first.physical_warp,
                    issue,
                    completion,
                })
            })
            .collect::<Result<Vec<_>, ReifyReason>>()?;
        Ok(Self {
            physical_warp: first.physical_warp,
            first_sequence: certificate.first_sequence(),
            last_sequence: certificate.last_sequence(),
            instruction_count: certificate.instruction_count(),
            first_issue: projection.first_issue,
            last_issue: projection.last_issue,
            next_issue: projection.next_issue,
            dispatch_ready: projection.dispatch_ready,
            last_completion: projection.last_completion,
            pending_completions: projection.pending_completions,
            effects,
        })
    }
}

/// Public execution snapshot used to compare a macro continuation with the
/// detailed model path. It intentionally exposes only model state, never a
/// native `SmCore` or hardware state claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelExecutionSnapshot {
    pub warp_ready: Vec<(i32, u64)>,
    pub dispatch_ready: [u64; SMSP_COUNT as usize],
    pub pipeline_ready: [[u64; 3]; SMSP_COUNT as usize],
    pub pending_completions: Vec<(u8, u8, Vec<u64>)>,
    pub completion: u64,
    pub last_issue: Option<u64>,
    pub effects: Vec<MacroEffect>,
}

/// Dispatch actually taken by `ModelExecutor::execute_decision`. A `Detailed`
/// result means the candidate was refused and one exact instruction was
/// executed; the caller continues from the next descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "dispatch")]
pub enum ModelDispatch {
    Macro { transition: MacroTransition },
    Detailed { reason: ReifyReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalFrontierError {
    EmptyProgram,
    InvalidPhysicalWarp {
        physical_warp: i32,
    },
    DuplicatePhysicalWarp {
        physical_warp: i32,
    },
    DuplicateResidentWarp {
        physical_warp: i32,
    },
    ProgramWarpNotResident {
        physical_warp: i32,
    },
    DuplicateSequence {
        physical_warp: i32,
        sequence: u64,
    },
    InvalidTiming {
        sequence: u64,
    },
    CompletionOverflow {
        sequence: u64,
    },
    CounterOverflow {
        counter: &'static str,
    },
    StateMismatch {
        exact_completion: u64,
        causal_completion: u64,
        exact_last_issue: Option<u64>,
        causal_last_issue: Option<u64>,
    },
    ObservationMismatch {
        sequence: u64,
        physical_warp: i32,
        exact_issue: u64,
        causal_issue: u64,
        exact_completion: u64,
        causal_completion: u64,
    },
    ObservationCountMismatch {
        exact: usize,
        causal: usize,
    },
    PcNotInSchedule {
        pc: u64,
    },
    UnsupportedOperationClass {
        pc: u64,
        class: OpClass,
    },
    UnsupportedMemoryRole {
        pc: u64,
    },
    UnsupportedPipeline {
        pc: u64,
    },
    UnsupportedArchitecture {
        arch: ArchId,
    },
    TimingMismatch {
        pc: u64,
        expected: u64,
        got: u64,
    },
    InvalidEvidenceRange {
        physical_warp: i32,
        first_sequence: u64,
        last_sequence: u64,
    },
    IncompleteIndependentEvidence {
        physical_warp: i32,
        first_sequence: u64,
        last_sequence: u64,
    },
    MacroTransitionRejected {
        reason: ReifyReason,
    },
}

impl std::fmt::Display for CausalFrontierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyProgram => write!(f, "causal program is empty"),
            Self::InvalidPhysicalWarp { physical_warp } => {
                write!(f, "physical warp {physical_warp} is outside the SM range")
            }
            Self::DuplicatePhysicalWarp { physical_warp } => {
                write!(
                    f,
                    "physical warp {physical_warp} appears in multiple programs"
                )
            }
            Self::DuplicateResidentWarp { physical_warp } => {
                write!(
                    f,
                    "physical warp {physical_warp} is duplicated in the resident roster"
                )
            }
            Self::ProgramWarpNotResident { physical_warp } => {
                write!(
                    f,
                    "program warp {physical_warp} is absent from the complete resident roster"
                )
            }
            Self::DuplicateSequence {
                physical_warp,
                sequence,
            } => write!(
                f,
                "physical warp {physical_warp} repeats instruction sequence {sequence}"
            ),
            Self::InvalidTiming { sequence } => {
                write!(
                    f,
                    "instruction {sequence} has zero timing or completion units"
                )
            }
            Self::CompletionOverflow { sequence } => {
                write!(f, "instruction {sequence} completion time overflows")
            }
            Self::CounterOverflow { counter } => {
                write!(f, "causal-frontier counter {counter} overflows")
            }
            Self::StateMismatch {
                exact_completion,
                causal_completion,
                exact_last_issue,
                causal_last_issue,
            } => write!(
                f,
                "cycle-accuracy invariant failed: exact completion {exact_completion}, causal completion {causal_completion}, exact last issue {exact_last_issue:?}, causal last issue {causal_last_issue:?}"
            ),
            Self::ObservationMismatch {
                sequence,
                physical_warp,
                exact_issue,
                causal_issue,
                exact_completion,
                causal_completion,
            } => write!(
                f,
                "cycle-accuracy observation mismatch at sequence {sequence}, warp {physical_warp}: exact issue/completion {exact_issue}/{exact_completion}, causal {causal_issue}/{causal_completion}"
            ),
            Self::ObservationCountMismatch { exact, causal } => write!(
                f,
                "cycle-accuracy observation count mismatch: exact {exact}, causal {causal}"
            ),
            Self::PcNotInSchedule { pc } => write!(f, "pc {pc} is absent from the issue schedule"),
            Self::UnsupportedOperationClass { pc, class } => {
                write!(
                    f,
                    "pc {pc} has operation class {class:?}, which cannot be summarized"
                )
            }
            Self::UnsupportedMemoryRole { pc } => {
                write!(f, "pc {pc} has a memory role and cannot be summarized")
            }
            Self::UnsupportedPipeline { pc } => {
                write!(f, "pc {pc} has an uncharacterized dispatch-only pipeline")
            }
            Self::UnsupportedArchitecture { arch } => write!(
                f,
                "causal-frontier topology is only admitted for sm_86; got {arch}"
            ),
            Self::TimingMismatch { pc, expected, got } => {
                write!(
                    f,
                    "pc {pc} supplies latency {got}, but the card requires {expected}"
                )
            }
            Self::MacroTransitionRejected { reason } => {
                write!(
                    f,
                    "macro transition rejected; exact continuation required: {reason:?}"
                )
            }
            Self::InvalidEvidenceRange {
                physical_warp,
                first_sequence,
                last_sequence,
            } => write!(
                f,
                "invalid witness range for warp {physical_warp}: {first_sequence}..={last_sequence}"
            ),
            Self::IncompleteIndependentEvidence {
                physical_warp,
                first_sequence,
                last_sequence,
            } => write!(
                f,
                "independent witness does not cover warp {physical_warp} range {first_sequence}..={last_sequence}"
            ),
        }
    }
}

impl std::error::Error for CausalFrontierError {}

fn pipeline_index(pipeline: PipelineClass) -> usize {
    match pipeline {
        PipelineClass::DispatchOnly
        | PipelineClass::Branch { .. }
        | PipelineClass::Barrier { .. } => 0,
        PipelineClass::Tensor { .. } | PipelineClass::Sfu { .. } => 1,
        PipelineClass::ScalarAlu { .. } => 2,
    }
}

fn subcore(physical_warp: i32) -> Result<usize, CausalFrontierError> {
    if !(0..48).contains(&physical_warp) {
        return Err(CausalFrontierError::InvalidPhysicalWarp { physical_warp });
    }
    Ok((physical_warp as usize) % usize::from(SMSP_COUNT))
}

fn append_instruction(
    instruction: CausalFrontierInstruction,
    warp_ready: &mut HashMap<i32, u64>,
    dispatch_ready: &mut [u64; SMSP_COUNT as usize],
    pipeline_ready: &mut [[u64; 3]; SMSP_COUNT as usize],
    in_flight_done: &mut [[Vec<u64>; 3]; SMSP_COUNT as usize],
    completion: &mut u64,
) -> Result<(u64, u64), CausalFrontierError> {
    if instruction.initiation_interval_cycles == 0
        || instruction.in_flight_capacity == 0
        || instruction.completion_units == 0
    {
        return Err(CausalFrontierError::InvalidTiming {
            sequence: instruction.sequence,
        });
    }
    let smsp = subcore(instruction.physical_warp)?;
    let pipe = pipeline_index(instruction.pipeline);
    let mut issue = warp_ready
        .get(&instruction.physical_warp)
        .copied()
        .unwrap_or(0)
        .max(dispatch_ready[smsp])
        .max(pipeline_ready[smsp][pipe]);
    let releases = &mut in_flight_done[smsp][pipe];
    releases.retain(|done| *done > issue);
    while releases.len() >= instruction.in_flight_capacity as usize {
        let Some(earliest) = releases.iter().copied().min() else {
            break;
        };
        issue = issue.max(earliest);
        releases.retain(|done| *done > issue);
    }
    let done = issue.checked_add(instruction.latency_cycles).ok_or(
        CausalFrontierError::CompletionOverflow {
            sequence: instruction.sequence,
        },
    )?;
    let next_issue = issue
        .checked_add(instruction.initiation_interval_cycles)
        .ok_or(CausalFrontierError::CompletionOverflow {
            sequence: instruction.sequence,
        })?;
    *warp_ready.entry(instruction.physical_warp).or_default() = next_issue;
    dispatch_ready[smsp] = issue
        .checked_add(1)
        .ok_or(CausalFrontierError::CompletionOverflow {
            sequence: instruction.sequence,
        })?;
    pipeline_ready[smsp][pipe] = next_issue;
    releases.push(done);
    *completion = (*completion).max(done);
    Ok((issue, done))
}

/// Resource state for the descriptor model. Detailed and summarized runs own
/// independent instances; no timing value is copied between their results.
/// Matching this state is model-local self-equivalence, not equivalence to
/// `SmCore` or to silicon.
#[derive(Debug)]
pub struct ModelExecutor {
    warp_ready: HashMap<i32, u64>,
    dispatch_ready: [u64; SMSP_COUNT as usize],
    pipeline_ready: [[u64; 3]; SMSP_COUNT as usize],
    in_flight_done: [[Vec<u64>; 3]; SMSP_COUNT as usize],
    completion: u64,
    last_issue: Option<u64>,
    observations: Vec<MacroEffect>,
}

impl PartialEq for ModelExecutor {
    fn eq(&self, other: &Self) -> bool {
        self.warp_ready == other.warp_ready
            && self.dispatch_ready == other.dispatch_ready
            && self.pipeline_ready == other.pipeline_ready
            && self.in_flight_done == other.in_flight_done
            && self.completion == other.completion
            && self.last_issue == other.last_issue
    }
}

impl Eq for ModelExecutor {}

fn ensure_terminal_state_match(
    exact: &ModelExecutor,
    causal: &ModelExecutor,
) -> Result<(), CausalFrontierError> {
    if exact == causal {
        return Ok(());
    }
    Err(CausalFrontierError::StateMismatch {
        exact_completion: exact.completion,
        causal_completion: causal.completion,
        exact_last_issue: exact.last_issue,
        causal_last_issue: causal.last_issue,
    })
}

fn ensure_observation_match(
    exact: &ModelExecutor,
    causal: &ModelExecutor,
) -> Result<(), CausalFrontierError> {
    let mut exact_observations = exact.observations.clone();
    let mut causal_observations = causal.observations.clone();
    let key = |observation: &MacroEffect| (observation.sequence, observation.physical_warp);
    exact_observations.sort_unstable_by_key(key);
    causal_observations.sort_unstable_by_key(key);
    if exact_observations.len() != causal_observations.len() {
        return Err(CausalFrontierError::ObservationCountMismatch {
            exact: exact_observations.len(),
            causal: causal_observations.len(),
        });
    }
    if let Some((exact, causal)) = exact_observations
        .iter()
        .zip(causal_observations.iter())
        .find(|(exact, causal)| exact != causal)
    {
        return Err(CausalFrontierError::ObservationMismatch {
            sequence: exact.sequence,
            physical_warp: exact.physical_warp,
            exact_issue: exact.issue,
            causal_issue: causal.issue,
            exact_completion: exact.completion,
            causal_completion: causal.completion,
        });
    }
    Ok(())
}

impl Default for ModelExecutor {
    fn default() -> Self {
        Self {
            warp_ready: HashMap::new(),
            dispatch_ready: [0; SMSP_COUNT as usize],
            pipeline_ready: [[0; 3]; SMSP_COUNT as usize],
            in_flight_done: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
            completion: 0,
            last_issue: None,
            observations: Vec::new(),
        }
    }
}
impl ModelExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> ModelExecutionSnapshot {
        let mut warp_ready = self
            .warp_ready
            .iter()
            .map(|(&warp, &ready)| (warp, ready))
            .collect::<Vec<_>>();
        warp_ready.sort_unstable();
        let pending_completions = self
            .in_flight_done
            .iter()
            .enumerate()
            .flat_map(|(smsp, pipelines)| {
                pipelines
                    .iter()
                    .enumerate()
                    .filter_map(move |(pipe, pending)| {
                        (!pending.is_empty()).then(|| (smsp as u8, pipe as u8, pending.clone()))
                    })
            })
            .collect();
        ModelExecutionSnapshot {
            warp_ready,
            dispatch_ready: self.dispatch_ready,
            pipeline_ready: self.pipeline_ready,
            pending_completions,
            completion: self.completion,
            last_issue: self.last_issue,
            effects: self.observations.clone(),
        }
    }

    pub fn execute_detailed(
        &mut self,
        instruction: CausalFrontierInstruction,
    ) -> Result<(), CausalFrontierError> {
        let (issue, completion) = append_instruction(
            instruction,
            &mut self.warp_ready,
            &mut self.dispatch_ready,
            &mut self.pipeline_ready,
            &mut self.in_flight_done,
            &mut self.completion,
        )?;
        self.last_issue = Some(
            self.last_issue
                .map_or(issue, |previous| previous.max(issue)),
        );
        self.observations.push(MacroEffect {
            sequence: instruction.sequence,
            physical_warp: instruction.physical_warp,
            issue,
            completion,
        });
        Ok(())
    }

    pub fn execute_macro(
        &mut self,
        first: CausalFrontierInstruction,
        certificate: IntervalCertificate,
    ) -> Result<MacroTransition, CausalFrontierError> {
        self.execute_macro_region(std::slice::from_ref(&first), certificate)
    }

    /// Consume a certificate only when it is bound to the complete descriptor
    /// region that it summarizes.  The old first-descriptor-only API remains
    /// available for single-instruction descriptors, but a multi-instruction
    /// certificate must use this method so a changed descriptor cannot inherit
    /// another region's timing/completion state.
    pub fn execute_macro_region(
        &mut self,
        region: &[CausalFrontierInstruction],
        certificate: IntervalCertificate,
    ) -> Result<MacroTransition, CausalFrontierError> {
        let Some(&first) = region.first() else {
            return Err(CausalFrontierError::MacroTransitionRejected {
                reason: ReifyReason::CertificateBinding,
            });
        };
        let expected_count = u64::try_from(region.len()).map_err(|_| {
            CausalFrontierError::MacroTransitionRejected {
                reason: ReifyReason::CertificateBinding,
            }
        })?;
        let bound = certificate.instruction_count() == expected_count
            && certificate.first_sequence() == first.sequence
            && certificate.last_sequence()
                == region.last().map_or(first.sequence, |item| item.sequence)
            && region.iter().enumerate().all(|(index, instruction)| {
                let sequence = first.sequence.checked_add(index as u64);
                let evidence = instruction.execution_evidence();
                sequence == Some(instruction.sequence)
                    && instruction.physical_warp == first.physical_warp
                    && evidence.pipeline_key == certificate.pipeline_key()
                    && evidence.latency_cycles == certificate.latency_cycles()
                    && evidence.initiation_interval_cycles
                        == certificate.initiation_interval_cycles()
                    && evidence.in_flight_capacity == certificate.in_flight_capacity()
                    && evidence.completion_units == 1
                    && evidence.control_clean
                    && evidence.dependency_free
                    && !evidence.external_side_effect
            });
        if !bound {
            return Err(CausalFrontierError::MacroTransitionRejected {
                reason: ReifyReason::CertificateBinding,
            });
        }
        // Descriptor validation already bounded physical_warp before either run.
        let smsp = first.physical_warp as usize % usize::from(SMSP_COUNT);
        let pipe = pipeline_index(first.pipeline);
        let start = self
            .warp_ready
            .get(&first.physical_warp)
            .copied()
            .unwrap_or(0)
            .max(self.dispatch_ready[smsp])
            .max(self.pipeline_ready[smsp][pipe]);
        if self.in_flight_done[smsp][pipe]
            .iter()
            .any(|done| *done > start)
        {
            return Err(CausalFrontierError::MacroTransitionRejected {
                reason: ReifyReason::IncomingPipelineWork,
            });
        }
        let transition = MacroTransition::derive(first, certificate, start)
            .map_err(|reason| CausalFrontierError::MacroTransitionRejected { reason })?;
        self.observations.extend(transition.effects.iter().copied());
        self.warp_ready
            .insert(first.physical_warp, transition.next_issue);
        self.dispatch_ready[smsp] = transition.dispatch_ready;
        self.pipeline_ready[smsp][pipe] = transition.next_issue;
        self.in_flight_done[smsp][pipe] = transition.pending_completions.clone();
        self.completion = self.completion.max(transition.last_completion);
        self.last_issue = Some(self.last_issue.map_or(transition.last_issue, |previous| {
            previous.max(transition.last_issue)
        }));
        Ok(transition)
    }

    pub fn execute_decision(
        &mut self,
        first: CausalFrontierInstruction,
        decision: FidelityDecision,
    ) -> Result<ModelDispatch, CausalFrontierError> {
        self.execute_decision_region(std::slice::from_ref(&first), decision)
    }

    pub fn execute_decision_region(
        &mut self,
        region: &[CausalFrontierInstruction],
        decision: FidelityDecision,
    ) -> Result<ModelDispatch, CausalFrontierError> {
        let Some(&first) = region.first() else {
            return Err(CausalFrontierError::MacroTransitionRejected {
                reason: ReifyReason::CertificateBinding,
            });
        };
        match decision {
            FidelityDecision::Summary { certificate } => {
                match self.execute_macro_region(region, certificate) {
                    Ok(transition) => Ok(ModelDispatch::Macro { transition }),
                    Err(CausalFrontierError::MacroTransitionRejected { reason }) => {
                        self.execute_detailed(first)?;
                        Ok(ModelDispatch::Detailed { reason })
                    }
                    Err(error) => Err(error),
                }
            }
            FidelityDecision::Reify { reason } => {
                self.execute_detailed(first)?;
                Ok(ModelDispatch::Detailed { reason })
            }
        }
    }
}

/// Analyze descriptors with unknown residency. Candidates remain detailed
/// because their instruction list cannot establish exclusive ownership.
pub fn summarize(
    programs: &[Vec<CausalFrontierInstruction>],
) -> Result<CausalFrontierReport, CausalFrontierError> {
    summarize_with_residency(programs, CausalFrontierResidency::Unknown)
}

/// Execute detailed and summarized recurrences with an explicit resident
/// population. A single program can use the descriptor recurrence; multiple
/// warp programs remain detailed unless a future API supplies a globally
/// comparable issue/arbitration witness. This does not replace native
/// SM/memory execution or establish its timing.
pub fn summarize_with_residency(
    programs: &[Vec<CausalFrontierInstruction>],
    mut residency: CausalFrontierResidency,
) -> Result<CausalFrontierReport, CausalFrontierError> {
    let planner_start = Instant::now();
    if programs.is_empty() {
        return Err(CausalFrontierError::EmptyProgram);
    }
    let mut exact_events = 0u64;
    let mut physical = HashSet::new();
    let mut residents_per_smsp = [0u32; SMSP_COUNT as usize];
    let (complete_residents, independent_evidence) = match &mut residency {
        CausalFrontierResidency::Unknown => (None, None),
        CausalFrontierResidency::Complete { physical_warps } => {
            physical_warps.sort_unstable();
            let mut residents = HashSet::new();
            for &physical_warp in physical_warps.iter() {
                let smsp = subcore(physical_warp)?;
                if !residents.insert(physical_warp) {
                    return Err(CausalFrontierError::DuplicateResidentWarp { physical_warp });
                }
                // Each physical warp is unique and in the fixed 48-slot SM.
                residents_per_smsp[smsp] = residents_per_smsp[smsp].checked_add(1).ok_or(
                    CausalFrontierError::CounterOverflow {
                        counter: "resident warps per SM partition",
                    },
                )?;
            }
            (Some(residents), None)
        }
        CausalFrontierResidency::CompleteWithEvidence {
            physical_warps,
            evidence,
        } => {
            physical_warps.sort_unstable();
            let mut residents = HashSet::new();
            for &physical_warp in physical_warps.iter() {
                let smsp = subcore(physical_warp)?;
                if !residents.insert(physical_warp) {
                    return Err(CausalFrontierError::DuplicateResidentWarp { physical_warp });
                }
                residents_per_smsp[smsp] = residents_per_smsp[smsp].checked_add(1).ok_or(
                    CausalFrontierError::CounterOverflow {
                        counter: "resident warps per SM partition",
                    },
                )?;
            }
            (Some(residents), Some(evidence.clone()))
        }
    };
    let mut ordered = Vec::new();
    let mut canonical_programs = Vec::with_capacity(programs.len());
    for program in programs {
        let mut canonical = program.clone();
        canonical.sort_by_key(|instruction| instruction.sequence);
        canonical_programs.push(canonical);
    }
    for (program_index, program) in canonical_programs.iter().enumerate() {
        let Some(first) = program.first() else {
            return Err(CausalFrontierError::EmptyProgram);
        };
        if !physical.insert(first.physical_warp) {
            return Err(CausalFrontierError::DuplicatePhysicalWarp {
                physical_warp: first.physical_warp,
            });
        }
        subcore(first.physical_warp)?;
        if complete_residents
            .as_ref()
            .is_some_and(|residents| !residents.contains(&first.physical_warp))
        {
            return Err(CausalFrontierError::ProgramWarpNotResident {
                physical_warp: first.physical_warp,
            });
        }
        let mut previous = None;
        for (index, instruction) in program.iter().enumerate() {
            if instruction.physical_warp != first.physical_warp {
                return Err(CausalFrontierError::DuplicatePhysicalWarp {
                    physical_warp: instruction.physical_warp,
                });
            }
            if previous == Some(instruction.sequence) {
                return Err(CausalFrontierError::DuplicateSequence {
                    physical_warp: first.physical_warp,
                    sequence: instruction.sequence,
                });
            }
            previous = Some(instruction.sequence);
            // `completion_units` contributes to observed event cardinality,
            // while the timing recurrence retains one completion token. The
            // summary path rejects every non-unit value above, so this
            // accounting cannot create a false compressed transition.
            exact_events = exact_events
                .checked_add(1 + u64::from(instruction.completion_units))
                .ok_or(CausalFrontierError::CompletionOverflow {
                    sequence: instruction.sequence,
                })?;
            ordered.push((
                instruction.sequence,
                instruction.physical_warp,
                program_index,
                index,
            ));
        }
    }
    ordered.sort_unstable();
    let mut exact = ModelExecutor::default();
    for (_, _, program, instruction) in &ordered {
        exact.execute_detailed(canonical_programs[*program][*instruction])?;
    }

    let mut causal = ModelExecutor::default();
    let mut causal_events = exact_events;
    let mut causal_execution_steps = 0_u64;
    let mut macro_dispatches = 0_u64;
    let mut candidate_regions = 0_u64;
    let mut candidate_instructions = 0_u64;
    let mut collapsed_instructions = 0_u64;
    let mut certified_regions = Vec::new();
    let mut reified_region_details = Vec::new();
    let mut consumed = vec![0; canonical_programs.len()];
    let mut scanned_until = vec![0; canonical_programs.len()];
    for (_, _, program_index, index) in &ordered {
        let (program_index, index) = (*program_index, *index);
        if index < consumed[program_index] {
            continue;
        }
        let program = &canonical_programs[program_index];
        let first = program[index];
        if first.completion_units != 1 {
            // Completion multiplicity is not a report-only counter. The
            // compact interval certificate materializes one completion token
            // per instruction, so any descriptor with additional effects is
            // explicitly reified on the exact path.
            reified_region_details.push(ReifiedRegion {
                physical_warp: first.physical_warp,
                first_sequence: first.sequence,
                last_sequence: first.sequence,
                reason: ReifyReason::CompletionMultiplicity,
            });
        }
        if index >= scanned_until[program_index] && first.summarizable() {
            let mut end = index + 1;
            while end < program.len()
                && program[end].summarizable()
                && program[end].pipeline == first.pipeline
            {
                end += 1;
            }
            scanned_until[program_index] = end;
            if end - index >= 2 {
                candidate_regions = candidate_regions.checked_add(1).ok_or(
                    CausalFrontierError::CounterOverflow {
                        counter: "candidate regions",
                    },
                )?;
                candidate_instructions = candidate_instructions
                    .checked_add(u64::try_from(end - index).map_err(|_| {
                        CausalFrontierError::CounterOverflow {
                            counter: "candidate instructions",
                        }
                    })?)
                    .ok_or(CausalFrontierError::CounterOverflow {
                        counter: "candidate instructions",
                    })?;
                let interval_instructions = program[index..end].to_vec();
                let evidence = interval_instructions
                    .iter()
                    .copied()
                    .map(CausalFrontierInstruction::execution_evidence);
                let evidence_covers_interval = independent_evidence.as_ref().is_some_and(|items| {
                    interval_instructions
                        .iter()
                        .all(|instruction| items.iter().any(|item| item.covers(*instruction)))
                });
                let decision = if canonical_programs.len() > 1 {
                    // `sequence` is local to each warp. Sorting local ordinals
                    // is a deterministic implementation detail, not a native
                    // scheduler witness; never let it certify cross-warp
                    // contention or same-cycle arbitration.
                    FidelityDecision::Reify {
                        reason: ReifyReason::CrossWarpOrderUnknown,
                    }
                } else if complete_residents.is_none() {
                    FidelityDecision::Reify {
                        reason: ReifyReason::UnknownResidency,
                    }
                } else if !evidence_covers_interval {
                    FidelityDecision::Reify {
                        reason: ReifyReason::MissingIndependentEvidence,
                    }
                } else {
                    certify_interval(evidence, residents_per_smsp[subcore(first.physical_warp)?])
                };
                let dispatch = causal.execute_decision_region(&interval_instructions, decision)?;
                match dispatch {
                    ModelDispatch::Macro { transition } => {
                        let region = CertifiedRegion {
                            physical_warp: transition.physical_warp,
                            first_sequence: transition.first_sequence,
                            last_sequence: transition.last_sequence,
                            first_issue_cycle: transition.first_issue,
                            last_issue_cycle: transition.last_issue,
                            completion_cycle: transition.last_completion,
                        };
                        let length = (end - index) as u64;
                        let reduction = length
                            .checked_mul(2)
                            .and_then(|value| value.checked_sub(1))
                            .ok_or(CausalFrontierError::CompletionOverflow {
                                sequence: first.sequence,
                            })?;
                        causal_events = causal_events.checked_sub(reduction).ok_or(
                            CausalFrontierError::CompletionOverflow {
                                sequence: first.sequence,
                            },
                        )?;
                        collapsed_instructions = collapsed_instructions.checked_add(length).ok_or(
                            CausalFrontierError::CounterOverflow {
                                counter: "collapsed instructions",
                            },
                        )?;
                        certified_regions.push(region);
                        macro_dispatches = macro_dispatches.checked_add(1).ok_or(
                            CausalFrontierError::CounterOverflow {
                                counter: "macro dispatches",
                            },
                        )?;
                        consumed[program_index] = end;
                        causal_execution_steps = causal_execution_steps.checked_add(1).ok_or(
                            CausalFrontierError::CounterOverflow {
                                counter: "causal execution steps",
                            },
                        )?;
                        continue;
                    }
                    ModelDispatch::Detailed { reason } => {
                        reified_region_details.push(ReifiedRegion {
                            physical_warp: first.physical_warp,
                            first_sequence: first.sequence,
                            last_sequence: program[end - 1].sequence,
                            reason,
                        });
                        consumed[program_index] = index + 1;
                        causal_execution_steps = causal_execution_steps.checked_add(1).ok_or(
                            CausalFrontierError::CounterOverflow {
                                counter: "causal execution steps",
                            },
                        )?;
                        continue;
                    }
                }
            }
        }
        causal.execute_detailed(first)?;
        consumed[program_index] = index + 1;
        causal_execution_steps =
            causal_execution_steps
                .checked_add(1)
                .ok_or(CausalFrontierError::CounterOverflow {
                    counter: "causal execution steps",
                })?;
    }
    let exact_execution_steps =
        u64::try_from(ordered.len()).map_err(|_| CausalFrontierError::CounterOverflow {
            counter: "exact execution steps",
        })?;
    let collapsed_regions = u64::try_from(certified_regions.len()).map_err(|_| {
        CausalFrontierError::CounterOverflow {
            counter: "collapsed regions",
        }
    })?;
    let reified_regions = u64::try_from(reified_region_details.len()).map_err(|_| {
        CausalFrontierError::CounterOverflow {
            counter: "reified regions",
        }
    })?;
    ensure_terminal_state_match(&exact, &causal)?;
    ensure_observation_match(&exact, &causal)?;
    Ok(CausalFrontierReport {
        residency,
        exact_event_count: exact_events,
        causal_event_count: causal_events,
        exact_final_cycle: exact.completion,
        causal_final_cycle: causal.completion,
        critical_path_delta: i128::from(causal.completion) - i128::from(exact.completion),
        exact_last_issue_cycle: exact.last_issue,
        causal_last_issue_cycle: causal.last_issue,
        exact_execution_steps,
        causal_execution_steps,
        macro_dispatches,
        terminal_state_matches: exact == causal,
        candidate_regions,
        collapsed_regions,
        reified_regions,
        candidate_instructions,
        collapsed_instructions,
        certified_regions,
        reified_region_details,
        planner_time_ns: planner_start.elapsed().as_nanos(),
    })
}

