//! Resource-aware causal-frontier analysis for explicit replay descriptors.
//!
//! This is an analysis seam, not a second SM executor. A producer supplies the
//! dependency and completion facts that the native replay already knows; the
//! analyzer refuses to summarize unknown effects and preserves the same local
//! issue/completion envelope when it does summarize an interval.

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
        if schedule.arch() != ArchId::Sm86 {
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
            control_clean: slot.control().stall() == 0
                && !slot.control().yield_flag()
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
#[derive(Debug)]
struct ModelState {
    warp_ready: HashMap<i32, u64>,
    dispatch_ready: [u64; SMSP_COUNT as usize],
    pipeline_ready: [[u64; 3]; SMSP_COUNT as usize],
    in_flight_done: [[Vec<u64>; 3]; SMSP_COUNT as usize],
    completion: u64,
    last_issue: Option<u64>,
    observations: Vec<InstructionObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InstructionObservation {
    sequence: u64,
    physical_warp: i32,
    issue: u64,
    completion: u64,
}

impl PartialEq for ModelState {
    fn eq(&self, other: &Self) -> bool {
        self.warp_ready == other.warp_ready
            && self.dispatch_ready == other.dispatch_ready
            && self.pipeline_ready == other.pipeline_ready
            && self.in_flight_done == other.in_flight_done
            && self.completion == other.completion
            && self.last_issue == other.last_issue
    }
}

impl Eq for ModelState {}

fn ensure_terminal_state_match(
    exact: &ModelState,
    causal: &ModelState,
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
    exact: &ModelState,
    causal: &ModelState,
) -> Result<(), CausalFrontierError> {
    let mut exact_observations = exact.observations.clone();
    let mut causal_observations = causal.observations.clone();
    let key =
        |observation: &InstructionObservation| (observation.sequence, observation.physical_warp);
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

impl Default for ModelState {
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
impl ModelState {
    fn append(
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
        self.observations.push(InstructionObservation {
            sequence: instruction.sequence,
            physical_warp: instruction.physical_warp,
            issue,
            completion,
        });
        Ok(())
    }

    fn apply_summary(
        &mut self,
        first: CausalFrontierInstruction,
        certificate: IntervalCertificate,
    ) -> Result<CertifiedRegion, ReifyReason> {
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
            return Err(ReifyReason::IncomingPipelineWork);
        }
        let projected = certificate.project(start)?;
        let observations = certificate.issue_completion_times(start)?;
        for (index, (issue, completion)) in observations.into_iter().enumerate() {
            let sequence = first
                .sequence
                .checked_add(u64::try_from(index).map_err(|_| ReifyReason::ArithmeticOverflow)?)
                .ok_or(ReifyReason::ArithmeticOverflow)?;
            self.observations.push(InstructionObservation {
                sequence,
                physical_warp: first.physical_warp,
                issue,
                completion,
            });
        }
        self.warp_ready
            .insert(first.physical_warp, projected.next_issue);
        self.dispatch_ready[smsp] = projected.dispatch_ready;
        self.pipeline_ready[smsp][pipe] = projected.next_issue;
        self.in_flight_done[smsp][pipe] = projected.pending_completions;
        self.completion = self.completion.max(projected.last_completion);
        self.last_issue = Some(self.last_issue.map_or(projected.last_issue, |previous| {
            previous.max(projected.last_issue)
        }));
        Ok(CertifiedRegion {
            physical_warp: first.physical_warp,
            first_sequence: certificate.first_sequence(),
            last_sequence: certificate.last_sequence(),
            first_issue_cycle: start,
            last_issue_cycle: projected.last_issue,
            completion_cycle: projected.last_completion,
        })
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
/// population. Exclusive SMSPs are independent in this descriptor model.
/// This does not replace native SM/memory execution or establish its timing.
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
    let complete_residents = match &mut residency {
        CausalFrontierResidency::Unknown => None,
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
            Some(residents)
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
    let mut exact = ModelState::default();
    for (_, _, program, instruction) in &ordered {
        exact.append(canonical_programs[*program][*instruction])?;
    }

    let mut causal = ModelState::default();
    let mut causal_events = exact_events;
    let mut causal_execution_steps = 0_u64;
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
                let evidence = program[index..end]
                    .iter()
                    .copied()
                    .map(CausalFrontierInstruction::execution_evidence);
                let decision = if complete_residents.is_some() {
                    certify_interval(evidence, residents_per_smsp[subcore(first.physical_warp)?])
                } else {
                    FidelityDecision::Reify {
                        reason: ReifyReason::UnknownResidency,
                    }
                };
                let attempt = match decision {
                    FidelityDecision::Summary { certificate } => {
                        causal.apply_summary(first, certificate)
                    }
                    FidelityDecision::Reify { reason } => Err(reason),
                };
                match attempt {
                    Ok(region) => {
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
                        consumed[program_index] = end;
                        causal_execution_steps = causal_execution_steps.checked_add(1).ok_or(
                            CausalFrontierError::CounterOverflow {
                                counter: "causal execution steps",
                            },
                        )?;
                        continue;
                    }
                    Err(reason) => reified_region_details.push(ReifiedRegion {
                        physical_warp: first.physical_warp,
                        first_sequence: first.sequence,
                        last_sequence: program[end - 1].sequence,
                        reason,
                    }),
                }
            }
        }
        causal.append(first)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    const SCALAR: PipelineClass = PipelineClass::ScalarAlu {
        latency_cycles: 1,
        initiation_interval_cycles: 1,
        in_flight_capacity: 8,
    };

    fn instruction(sequence: u64, external_side_effect: bool) -> CausalFrontierInstruction {
        CausalFrontierInstruction {
            sequence,
            physical_warp: 0,
            pipeline: SCALAR,
            latency_cycles: 1,
            initiation_interval_cycles: 1,
            in_flight_capacity: 8,
            completion_units: 1,
            control_clean: true,
            dependency_free: true,
            external_side_effect,
        }
    }

    #[test]
    fn canonicalizes_unsorted_program_before_region_scan() {
        let mut unsorted = vec![
            instruction(2, false),
            instruction(0, false),
            instruction(1, true),
        ];
        unsorted[0].latency_cycles = 10;
        let report = summarize(std::slice::from_ref(&unsorted)).unwrap();
        let sorted = vec![unsorted[1], unsorted[2], unsorted[0]];
        let sorted_report = summarize(&[sorted]).unwrap();
        assert_eq!(report.exact_final_cycle, sorted_report.exact_final_cycle);
        assert_eq!(report.candidate_regions, 0);
        assert_eq!(report.collapsed_regions, 0);
        assert_eq!(report.reified_regions, 0);
    }

    #[test]
    fn rejects_reused_physical_warp_programs() {
        let error =
            summarize(&[vec![instruction(0, false)], vec![instruction(1, false)]]).unwrap_err();
        assert_eq!(
            error,
            CausalFrontierError::DuplicatePhysicalWarp { physical_warp: 0 }
        );
    }

    #[test]
    fn rejects_duplicate_sequences_within_one_program() {
        let error = summarize(&[vec![instruction(4, false), instruction(4, false)]]).unwrap_err();
        assert_eq!(
            error,
            CausalFrontierError::DuplicateSequence {
                physical_warp: 0,
                sequence: 4,
            }
        );
    }

    #[test]
    fn independently_executed_summary_preserves_capacity_and_suffix_state() {
        let mut program: Vec<_> = (0..8)
            .map(|sequence| CausalFrontierInstruction {
                latency_cycles: 10,
                initiation_interval_cycles: 2,
                in_flight_capacity: 2,
                ..instruction(sequence, false)
            })
            .collect();
        program.push(CausalFrontierInstruction {
            latency_cycles: 1,
            initiation_interval_cycles: 2,
            in_flight_capacity: 2,
            ..instruction(8, true)
        });
        let report = summarize_with_residency(
            &[program],
            CausalFrontierResidency::Complete {
                physical_warps: vec![0],
            },
        )
        .unwrap();
        assert_eq!(report.exact_final_cycle, 42);
        assert_eq!(report.causal_final_cycle, 42);
        assert_eq!(report.exact_last_issue_cycle, Some(40));
        assert_eq!(report.causal_last_issue_cycle, Some(40));
        assert_eq!(report.exact_execution_steps, 9);
        assert_eq!(report.causal_execution_steps, 2);
        assert_eq!(report.certified_regions[0].last_issue_cycle, 32);
        assert!(report.terminal_state_matches);
    }

    #[test]
    fn incoming_pipeline_completions_force_detailed_execution() {
        let mut program = vec![CausalFrontierInstruction {
            latency_cycles: 20,
            in_flight_capacity: 1,
            ..instruction(0, true)
        }];
        program.extend((1..5).map(|sequence| CausalFrontierInstruction {
            latency_cycles: 4,
            in_flight_capacity: 1,
            ..instruction(sequence, false)
        }));
        let report = summarize_with_residency(
            &[program],
            CausalFrontierResidency::Complete {
                physical_warps: vec![0],
            },
        )
        .unwrap();
        assert_eq!(report.collapsed_regions, 0);
        assert_eq!(
            report.reified_region_details[0].reason,
            ReifyReason::IncomingPipelineWork
        );
        assert_eq!(report.causal_final_cycle, 36);
        assert!(report.terminal_state_matches);
    }

    #[test]
    fn heterogeneous_latency_cannot_claim_the_first_latency_for_every_instruction() {
        let mut first = instruction(0, false);
        first.latency_cycles = 100;
        let report = summarize_with_residency(
            &[vec![first, instruction(1, false), instruction(2, false)]],
            CausalFrontierResidency::Complete {
                physical_warps: vec![0],
            },
        )
        .unwrap();
        assert_eq!(report.collapsed_regions, 0);
        assert_eq!(
            report.reified_region_details[0].reason,
            ReifyReason::MixedTiming
        );
        assert_eq!(report.causal_final_cycle, 100);
        assert!(report.terminal_state_matches);
    }

    #[test]
    fn equal_sequence_cross_warp_order_is_input_order_independent() {
        let mut second = instruction(0, false);
        second.physical_warp = 1;
        let first = instruction(0, false);
        let a = summarize(&[vec![first], vec![second]]).unwrap();
        let b = summarize(&[vec![second], vec![first]]).unwrap();
        assert_eq!(
            (a.exact_event_count, a.exact_final_cycle),
            (b.exact_event_count, b.exact_final_cycle)
        );
    }

    #[test]
    fn serialized_report_excludes_runtime_planner_telemetry() {
        let report = summarize(&[vec![instruction(0, false)]]).unwrap();
        let value = serde_json::to_value(report).unwrap();
        assert!(value.get("planner_time_ns").is_none());
    }

    #[test]
    fn state_mismatch_fails_closed_before_a_report_can_certify_a_skip() {
        let exact = ModelState::default();
        let mut causal = ModelState::default();
        causal.completion = 1;
        assert!(matches!(
            ensure_terminal_state_match(&exact, &causal),
            Err(CausalFrontierError::StateMismatch {
                exact_completion: 0,
                causal_completion: 1,
                ..
            })
        ));
    }

    #[test]
    fn observation_count_mismatch_is_not_misreported_as_overflow() {
        let exact = ModelState::default();
        let mut causal = ModelState::default();
        causal.observations.push(InstructionObservation {
            sequence: 0,
            physical_warp: 0,
            issue: 0,
            completion: 1,
        });
        assert_eq!(
            ensure_observation_match(&exact, &causal),
            Err(CausalFrontierError::ObservationCountMismatch {
                exact: 0,
                causal: 1
            })
        );
    }

    #[test]
    fn deterministic_adversarial_sweep_preserves_all_issue_and_completion_cycles() {
        // A tiny local generator keeps this property test reproducible without
        // adding a dependency or hiding the workload shape in a fixture.
        let mut seed = 0x9e37_79b9_u64;
        let mut next = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // The low bits of this LCG have short periods; using them for
            // topology and pipeline choices couples otherwise independent axes.
            seed >> 32
        };
        let mut collapsed_regions = 0;
        let mut reified_regions = 0;
        for case in 0..256 {
            let program_count = 1 + (next() % 4) as usize;
            let mut physical_warps = Vec::with_capacity(program_count);
            while physical_warps.len() < program_count {
                let candidate = (next() % 48) as i32;
                if !physical_warps.contains(&candidate) {
                    physical_warps.push(candidate);
                }
            }
            let mut programs = Vec::with_capacity(program_count);
            for &physical_warp in &physical_warps {
                let length = 1 + (next() % 12) as usize;
                let latency = 1 + next() % 20;
                let initiation = 1 + next() % 8;
                let capacity = 1 + (next() % 4) as u32;
                let pipeline = if next() & 1 == 0 {
                    PipelineClass::ScalarAlu {
                        latency_cycles: latency,
                        initiation_interval_cycles: initiation,
                        in_flight_capacity: capacity,
                    }
                } else {
                    PipelineClass::Tensor {
                        initiation_interval_cycles: initiation,
                    }
                };
                let program = (0..length)
                    .map(|sequence| CausalFrontierInstruction {
                        sequence: sequence as u64,
                        physical_warp,
                        pipeline,
                        latency_cycles: latency,
                        initiation_interval_cycles: initiation,
                        in_flight_capacity: capacity,
                        completion_units: if next() % 5 == 0 { 2 } else { 1 },
                        control_clean: next() % 7 != 0,
                        dependency_free: next() % 11 != 0,
                        external_side_effect: next() % 13 == 0,
                    })
                    .collect();
                programs.push(program);
            }
            let report = summarize_with_residency(
                &programs,
                CausalFrontierResidency::Complete { physical_warps },
            )
            .unwrap_or_else(|error| panic!("case {case}: {error}"));
            collapsed_regions += report.collapsed_regions;
            reified_regions += report.reified_regions;
            assert!(report.terminal_state_matches);
            assert_eq!(
                report.exact_last_issue_cycle,
                report.causal_last_issue_cycle
            );

            // The same descriptors must remain detailed when residency is
            // unknown, or when stalled non-candidate warps share every SMSP.
            for residency in [
                CausalFrontierResidency::Unknown,
                CausalFrontierResidency::Complete {
                    physical_warps: (0..48).collect(),
                },
            ] {
                let detailed = summarize_with_residency(&programs, residency)
                    .unwrap_or_else(|error| panic!("case {case}: {error}"));
                assert_eq!(detailed.collapsed_regions, 0, "case {case}");
                assert_eq!(detailed.causal_event_count, detailed.exact_event_count);
                assert_eq!(detailed.causal_final_cycle, report.causal_final_cycle);
                assert_eq!(
                    detailed.causal_last_issue_cycle,
                    report.causal_last_issue_cycle
                );
            }
        }
        assert!(
            collapsed_regions > 0,
            "adversarial sweep never executed a summary"
        );
        assert!(
            reified_regions > 0,
            "adversarial sweep never refused a candidate"
        );
        eprintln!(
            "256 cases: {collapsed_regions} summarized regions, {reified_regions} reified regions with complete residency"
        );
    }
}
