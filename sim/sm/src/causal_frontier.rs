//! Resource-aware causal-frontier analysis for explicit replay descriptors.
//!
//! This is an analysis seam, not a second SM executor. A producer supplies the
//! dependency and completion facts that the native replay already knows; the
//! analyzer refuses to summarize unknown effects and preserves the same local
//! issue/completion envelope when it does summarize an interval.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use serde::Serialize;

use crate::program::{FunctionSchedule, PipelineClass};

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
        let slot = schedule
            .issue_slot(pc)
            .ok_or(CausalFrontierError::PcNotInSchedule { pc })?;
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
        self.control_clean
            && self.dependency_free
            && !self.external_side_effect
            && self.completion_units == 1
            && self.initiation_interval_cycles != 0
            && self.in_flight_capacity != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CausalFrontierReport {
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
    /// Runtime-only planner telemetry. It is deliberately excluded from the
    /// serialized report so repeated analyses remain byte-deterministic.
    #[serde(skip_serializing)]
    pub planner_time_ns: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalFrontierError {
    EmptyProgram,
    InvalidPhysicalWarp { physical_warp: i32 },
    DuplicatePhysicalWarp { physical_warp: i32 },
    DuplicateSequence { physical_warp: i32, sequence: u64 },
    InvalidTiming { sequence: u64 },
    CompletionOverflow { sequence: u64 },
    PcNotInSchedule { pc: u64 },
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
            Self::PcNotInSchedule { pc } => write!(f, "pc {pc} is absent from the issue schedule"),
        }
    }
}

impl std::error::Error for CausalFrontierError {}

fn pipeline_index(pipeline: PipelineClass) -> usize {
    match pipeline {
        PipelineClass::DispatchOnly
        | PipelineClass::Branch { .. }
        | PipelineClass::Barrier { .. } => 0,
        PipelineClass::Tensor { .. } => 1,
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
) -> Result<(), CausalFrontierError> {
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
    Ok(())
}

/// Analyze explicit programs with the same issue/completion recurrence in both
/// modes. Causal collapse is admitted only for a single unfinished physical
/// warp on an SMSP; every refused candidate remains an exact instruction stream.
pub fn summarize(
    programs: &[Vec<CausalFrontierInstruction>],
) -> Result<CausalFrontierReport, CausalFrontierError> {
    let planner_start = Instant::now();
    let mut exact_events = 0u64;
    let mut exact_final = 0u64;
    let mut candidates = 0u64;
    let mut candidate_instructions = 0u64;
    let mut collapsed = 0u64;
    let mut collapsed_instructions = 0u64;
    let mut reified = 0u64;
    let mut physical = HashSet::new();
    let mut active_per_smsp = [0usize; SMSP_COUNT as usize];
    let mut ordered = Vec::new();
    // Producers may stream records in arrival order. Canonicalize each
    // physical-warp program before both the exact recurrence and the causal
    // region scan; otherwise the two passes can disagree about which adjacent
    // instructions form a collapse candidate.
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
        active_per_smsp[subcore(first.physical_warp)?] += 1;
        let mut sequences = HashSet::with_capacity(program.len());
        for (index, instruction) in program.iter().enumerate() {
            if instruction.physical_warp != first.physical_warp {
                return Err(CausalFrontierError::DuplicatePhysicalWarp {
                    physical_warp: instruction.physical_warp,
                });
            }
            if !sequences.insert(instruction.sequence) {
                return Err(CausalFrontierError::DuplicateSequence {
                    physical_warp: first.physical_warp,
                    sequence: instruction.sequence,
                });
            }
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
    let mut warp_ready = HashMap::new();
    let mut dispatch_ready = [0u64; SMSP_COUNT as usize];
    let mut pipeline_ready = [[0u64; 3]; SMSP_COUNT as usize];
    let mut in_flight_done: [[Vec<u64>; 3]; SMSP_COUNT as usize] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for (_, _, program_index, instruction_index) in &ordered {
        append_instruction(
            canonical_programs[*program_index][*instruction_index],
            &mut warp_ready,
            &mut dispatch_ready,
            &mut pipeline_ready,
            &mut in_flight_done,
            &mut exact_final,
        )?;
    }
    let mut causal_events = exact_events;
    let causal_final = exact_final;
    for program in &canonical_programs {
        let mut index = 0;
        while index < program.len() {
            let first = program[index];
            if !first.summarizable() {
                index += 1;
                continue;
            }
            let mut end = index + 1;
            while end < program.len()
                && program[end].summarizable()
                && program[end].pipeline == first.pipeline
            {
                end += 1;
            }
            let length = end - index;
            if length < 2 {
                index = end;
                continue;
            }
            candidates += 1;
            candidate_instructions += length as u64;
            let smsp = subcore(first.physical_warp)?;
            if active_per_smsp[smsp] != 1 {
                reified += 1;
                index = end;
                continue;
            }
            collapsed += 1;
            collapsed_instructions += length as u64;
            let reduction = (length as u64)
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
            // The exact recurrence above is retained as the causal frontier;
            // summarization changes event representation, not completion time.
            index = end;
        }
    }
    Ok(CausalFrontierReport {
        exact_event_count: exact_events,
        causal_event_count: causal_events,
        exact_final_cycle: exact_final,
        causal_final_cycle: causal_final,
        critical_path_delta: i128::from(causal_final) - i128::from(exact_final),
        candidate_regions: candidates,
        collapsed_regions: collapsed,
        reified_regions: reified,
        candidate_instructions,
        collapsed_instructions,
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
}
