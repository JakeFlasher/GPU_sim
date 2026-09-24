//! Production custom-program frontend: declarative resources -> shared event
//! engine, without pretending a new GPU is an SM86 ArchId. Native SASS remains
//! a separate frontend. Functional semantics are the documented u32 SIMT ISA.

use serde::Serialize;
use sim_arch::{
    simt::{
        FunctionalResult, SimtError, SimtExecutor, SimtGroupState, SimtProgram, SimtTimingExecutor,
        TimingResult,
    },
    simt_pipeline::{self, MemoryDomain},
    spec::ArchitectureSpec,
};
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct CustomGroupResult {
    pub group_id: usize,
    pub admission_cycle: u64,
    pub retirement_cycle: u64,
    pub functional: FunctionalResult,
    pub timing: TimingResult,
}

#[derive(Debug, Serialize)]
pub struct CustomResult {
    pub schema_version: u32,
    pub backend: &'static str,
    pub semantics: &'static str,
    pub architecture_spec: ArchitectureSpec,
    pub final_cycle: u64,
    pub completion_events: u64,
    pub groups: Vec<CustomGroupResult>,
}

pub fn run(spec: &ArchitectureSpec, programs: &[SimtProgram]) -> Result<CustomResult, SimtError> {
    if programs.is_empty() || programs.len() > 64 {
        return Err(SimtError::InvalidProgram(
            "custom run requires 1..=64 programs",
        ));
    }
    // Shared reads are legal. Conflicting inter-group accesses require a
    // memory-consistency/synchronization frontend, not host iteration order.
    // Retain every phase: overwriting an address's last phase can hide an
    // earlier same-phase race. Global conflicts remain forbidden across all
    // phases; only shared memory has a barrier ordering contract here.
    let mut accesses = BTreeMap::<(MemoryDomain, u32, usize), (usize, bool)>::new();
    let mut functional = Vec::new();
    let mut tasks = Vec::new();
    let mut barrier_positions = BTreeMap::<u8, Vec<(usize, usize)>>::new();
    for (g, program) in programs.iter().enumerate() {
        let mut seen = BTreeMap::new();
        for (index, instruction) in program.instructions().iter().enumerate() {
            if let sim_arch::simt::SimtInstruction::GroupBarrier { id } = instruction {
                if seen.insert(*id, index).is_some() {
                    return Err(SimtError::InvalidProgram(
                        "barrier id is reused in one group",
                    ));
                }
                barrier_positions.entry(*id).or_default().push((g, index));
            }
        }
    }
    for (id, positions) in &barrier_positions {
        if positions.len() != programs.len() {
            return Err(SimtError::InvalidProgram(
                "barrier participation is incomplete",
            ));
        }
        let first_index = positions[0].1;
        if positions.iter().any(|(_, index)| *index != first_index) {
            return Err(SimtError::InvalidProgram(
                "barrier generation must use the same program position in every group",
            ));
        }
        let _ = id;
    }
    if !barrier_positions.is_empty()
        && programs.len() > usize::try_from(spec.execution().max_resident_groups()).unwrap_or(0)
    {
        return Err(SimtError::InvalidProgram(
            "barrier CTA requires all participating groups resident",
        ));
    }
    if !barrier_positions.is_empty()
        && programs
            .windows(2)
            .any(|window| window[0].instructions().len() != window[1].instructions().len())
    {
        return Err(SimtError::InvalidProgram(
            "custom groups must use equal instruction counts",
        ));
    }
    let mut group_states = programs
        .iter()
        .map(|program| {
            SimtExecutor::new(spec, program)?;
            SimtGroupState::new(spec.execution().simt_width())
        })
        .collect::<Result<Vec<_>, SimtError>>()?;
    let mut global_initial = BTreeMap::new();
    let mut shared_memory = BTreeMap::new();
    for program in programs {
        for (&address, &value) in program.initial_memory() {
            if global_initial
                .insert(address, value)
                .is_some_and(|prior| prior != value)
            {
                return Err(SimtError::InvalidProgram(
                    "conflicting initial global memory",
                ));
            }
        }
        for (&address, &value) in program.initial_shared_memory() {
            if shared_memory
                .insert(address, value)
                .is_some_and(|prior| prior != value)
            {
                return Err(SimtError::InvalidProgram(
                    "conflicting initial shared memory",
                ));
            }
        }
    }
    // Both address spaces are CTA-visible initial state. Per-group maps are
    // retained only so the conservative global-conflict contract can report
    // each group's final state independently.
    let mut global_memories = vec![global_initial; programs.len()];
    for (g, program) in programs.iter().enumerate() {
        let mut phase = 0;
        for inst in program.instructions() {
            if let Some((domain, base, stride, write)) = simt_pipeline::memory_access(inst) {
                for lane in 0..spec.execution().simt_width() {
                    let addr = base
                        .checked_add(
                            stride
                                .checked_mul(u32::from(lane))
                                .ok_or(SimtError::AddressOverflow)?,
                        )
                        .ok_or(SimtError::AddressOverflow)?;
                    let key = (
                        domain,
                        addr,
                        if domain == MemoryDomain::Shared {
                            phase
                        } else {
                            0
                        },
                    );
                    if let Some((owner, wrote)) = accesses.get(&key)
                        && *owner != g
                        && (*wrote || write)
                    {
                        return Err(SimtError::InvalidProgram(
                            "inter-group memory conflict requires synchronization/execution support",
                        ));
                    }
                    accesses
                        .entry(key)
                        .and_modify(|(_, wrote)| {
                            *wrote |= write;
                        })
                        .or_insert((g, write));
                }
            }
            if let sim_arch::simt::SimtInstruction::GroupBarrier { .. } = inst {
                phase += 1;
            }
        }
        tasks.push(simt_pipeline::tasks(spec, program)?);
    }
    if barrier_positions.is_empty() {
        // Shared memory is CTA-wide even when this bounded run has no
        // rendezvous. Execute each independent group against the common
        // initial shared map while retaining a private global map. Static
        // conflict checks above reject unsynchronized cross-group writes.
        for (group, program) in programs.iter().enumerate() {
            for instruction in program.instructions() {
                group_states[group].execute(
                    instruction,
                    &mut global_memories[group],
                    &mut shared_memory,
                )?;
            }
        }
    } else {
        for index in 0..programs[0].instructions().len() {
            for (group, program) in programs.iter().enumerate() {
                group_states[group].execute(
                    &program.instructions()[index],
                    &mut global_memories[group],
                    &mut shared_memory,
                )?;
            }
        }
    }
    for (group, state) in group_states.into_iter().enumerate() {
        functional.push(state.finish(spec, &global_memories[group], &shared_memory)?);
    }
    // A barrier generation synchronizes every group's prior task and releases
    // every group's first post-barrier task only after all arrivals complete.
    for positions in barrier_positions.values() {
        for (g, index) in positions {
            for (other_g, other_index) in positions {
                if g != other_g && *index > 0 {
                    tasks[*g][*index]
                        .cross_dependencies
                        .push((*other_g, *other_index - 1));
                }
                if g != other_g && *index + 1 < tasks[*g].len() {
                    tasks[*g][*index + 1]
                        .cross_dependencies
                        .push((*other_g, *other_index));
                }
            }
        }
    }
    let schedule = sim_engine::pipeline::schedule(
        &simt_pipeline::resources(spec)?,
        &tasks,
        spec.execution().max_resident_groups(),
    )?;
    let mut groups = Vec::new();
    for (g, (program, state)) in programs.iter().zip(functional).enumerate() {
        let mut timing =
            SimtTimingExecutor::new(spec, program)?.project_schedule(&schedule.groups[g])?;
        for event in &mut timing.semantic_events {
            event.native.warp_id = g as i32;
        }
        groups.push(CustomGroupResult {
            group_id: g,
            admission_cycle: schedule.admissions[g],
            retirement_cycle: schedule.retirements[g],
            functional: state,
            timing,
        });
    }
    Ok(CustomResult {
        schema_version: 1,
        backend: "sim_sm_custom_event_pipeline",
        semantics: "model_relative_u32_simt_shared_barrier_ordered",
        architecture_spec: spec.clone(),
        final_cycle: schedule.final_cycle,
        completion_events: schedule.completion_events,
        groups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn spec() -> ArchitectureSpec {
        ArchitectureSpec::parse(
            Path::new("pipeline2.json"),
            include_bytes!("../../../arch_profiles/specs/simt_u32_pipeline2.json"),
        )
        .expect("pipeline2 spec should parse")
    }

    fn program(path: &'static str, bytes: &'static [u8]) -> SimtProgram {
        SimtProgram::parse(Path::new(path), bytes).expect("custom program should parse")
    }

    #[test]
    fn admits_two_disjoint_groups_through_shared_pipeline() {
        let result = run(
            &spec(),
            &[
                program(
                    "overlap.json",
                    include_bytes!(
                        "../../../arch_profiles/programs/simt_u32_pipeline_overlap.json"
                    ),
                ),
                program(
                    "lane_store.json",
                    include_bytes!("../../../arch_profiles/programs/simt_u32_lane_store_300.json"),
                ),
            ],
        )
        .expect("disjoint groups should execute");
        assert_eq!(result.groups.len(), 2);
        assert_eq!(result.final_cycle, 45);
        assert_eq!(result.completion_events, 10);
    }

    #[test]
    fn rejects_cross_group_memory_conflict_without_ordering() {
        let mut conflicting = serde_json::from_slice::<serde_json::Value>(include_bytes!(
            "../../../arch_profiles/programs/simt_u32_lane_store_300.json"
        ))
        .expect("program JSON should parse");
        conflicting["instructions"][2]["base"] = serde_json::json!(100);
        let conflicting = SimtProgram::parse(
            Path::new("conflicting.json"),
            &serde_json::to_vec(&conflicting).expect("program JSON should serialize"),
        )
        .expect("conflicting program should be structurally valid");
        let error = run(
            &spec(),
            &[
                program(
                    "overlap.json",
                    include_bytes!(
                        "../../../arch_profiles/programs/simt_u32_pipeline_overlap.json"
                    ),
                ),
                conflicting,
            ],
        )
        .expect_err("unordered cross-group read/write must fail closed");
        assert!(error.to_string().contains("inter-group memory conflict"));
    }

    #[test]
    fn group_barrier_releases_post_barrier_work_after_all_arrivals() {
        let result = run(
            &spec(),
            &[
                program(
                    "barrier_a.json",
                    include_bytes!("../../../arch_profiles/programs/simt_u32_barrier_a.json"),
                ),
                program(
                    "barrier_b.json",
                    include_bytes!("../../../arch_profiles/programs/simt_u32_barrier_b.json"),
                ),
            ],
        )
        .expect("matching CTA barriers should execute");
        let first = &result.groups[0].timing.events;
        let second = &result.groups[1].timing.events;
        let first_barrier = first
            .iter()
            .find(|event| event.operation == "group_barrier")
            .unwrap();
        let second_barrier = second
            .iter()
            .find(|event| event.operation == "group_barrier")
            .unwrap();
        let first_store = first
            .iter()
            .find(|event| event.operation == "store_shared")
            .unwrap();
        let second_store = second
            .iter()
            .find(|event| event.operation == "store_shared")
            .unwrap();
        assert!(
            first_store.start_cycle
                >= first_barrier
                    .complete_cycle
                    .max(second_barrier.complete_cycle)
        );
        assert!(
            second_store.start_cycle
                >= first_barrier
                    .complete_cycle
                    .max(second_barrier.complete_cycle)
        );
    }

    #[test]
    fn rejects_mismatched_barrier_participation() {
        let mut raw = serde_json::from_slice::<serde_json::Value>(include_bytes!(
            "../../../arch_profiles/programs/simt_u32_barrier_b.json"
        ))
        .expect("barrier program JSON should parse");
        raw["instructions"].as_array_mut().unwrap().remove(2);
        let missing = SimtProgram::parse(
            Path::new("missing_barrier.json"),
            &serde_json::to_vec(&raw).expect("barrier program JSON should serialize"),
        )
        .expect("program without barrier remains structurally valid");
        let error = run(
            &spec(),
            &[
                program(
                    "barrier_a.json",
                    include_bytes!("../../../arch_profiles/programs/simt_u32_barrier_a.json"),
                ),
                missing,
            ],
        )
        .expect_err("barrier participation mismatch must fail closed");
        assert!(error.to_string().contains("barrier participation"));
    }

    #[test]
    fn rejects_same_phase_shared_writes_after_barrier() {
        let mut raw = serde_json::from_slice::<serde_json::Value>(include_bytes!(
            "../../../arch_profiles/programs/simt_u32_barrier_b.json"
        ))
        .expect("barrier program JSON should parse");
        raw["instructions"][3]["base"] = serde_json::json!(200);
        let conflicting = SimtProgram::parse(
            Path::new("same_phase_shared_conflict.json"),
            &serde_json::to_vec(&raw).expect("barrier program JSON should serialize"),
        )
        .expect("conflicting program should be structurally valid");
        let error = run(
            &spec(),
            &[
                program(
                    "barrier_a.json",
                    include_bytes!("../../../arch_profiles/programs/simt_u32_barrier_a.json"),
                ),
                conflicting,
            ],
        )
        .expect_err("same-phase shared writes must fail closed");
        assert!(error.to_string().contains("inter-group memory conflict"));
    }

    #[test]
    fn global_conflicts_remain_rejected_across_barrier_phases() {
        let producer = program(
            "global_producer.json",
            br#"{
              "schema_version": 1,
              "architecture": "example_simt8",
              "isa": "simt_u32_v1",
              "instructions": [
                {"op":"set_lane_id", "dst":0},
                {"op":"add_imm", "dst":1, "src":0, "imm":1},
                {"op":"store", "base":700, "stride":4, "src":0},
                {"op":"group_barrier", "id":1},
                {"op":"add_imm", "dst":2, "src":1, "imm":1},
                {"op":"halt"}
              ]
            }"#,
        );
        let consumer = program(
            "global_consumer.json",
            br#"{
              "schema_version": 1,
              "architecture": "example_simt8",
              "isa": "simt_u32_v1",
              "instructions": [
                {"op":"set_lane_id", "dst":0},
                {"op":"add_imm", "dst":1, "src":0, "imm":1},
                {"op":"add_imm", "dst":2, "src":1, "imm":1},
                {"op":"group_barrier", "id":1},
                {"op":"load", "dst":1, "base":700, "stride":4},
                {"op":"halt"}
              ]
            }"#,
        );
        let error = run(&spec(), &[producer, consumer])
            .expect_err("global memory ordering is outside the barrier contract");
        assert!(error.to_string().contains("inter-group memory conflict"));
    }

    #[test]
    fn barrier_propagates_shared_value_to_consumer_group() {
        let result = run(
            &spec(),
            &[
                program(
                    "producer.json",
                    include_bytes!(
                        "../../../arch_profiles/programs/simt_u32_barrier_producer.json"
                    ),
                ),
                program(
                    "consumer.json",
                    include_bytes!(
                        "../../../arch_profiles/programs/simt_u32_barrier_consumer.json"
                    ),
                ),
            ],
        )
        .expect("barrier producer/consumer should execute");
        assert_eq!(result.groups[1].functional.registers[0][2], 100);
        assert_eq!(
            result.groups[0].functional.shared_memory.get(&500),
            Some(&100)
        );
    }

    #[test]
    fn initial_address_space_state_is_common_without_barrier() {
        let producer = program(
            "initial_shared_owner.json",
            br#"{
              "schema_version": 1,
              "architecture": "example_simt8",
              "isa": "simt_u32_v1",
              "initial_shared_memory": {"500": 77},
              "initial_memory": {"600": 88},
              "instructions": [{"op":"halt"}]
            }"#,
        );
        let consumer = program(
            "initial_shared_reader.json",
            br#"{
              "schema_version": 1,
              "architecture": "example_simt8",
              "isa": "simt_u32_v1",
              "instructions": [
                {"op":"load_shared", "dst":0, "base":500, "stride":0},
                {"op":"load", "dst":1, "base":600, "stride":0},
                {"op":"halt"}
              ]
            }"#,
        );
        let result = run(&spec(), &[producer, consumer])
            .expect("CTA initial shared state should be visible to every group");
        assert_eq!(result.groups[1].functional.registers[0][0], 77);
        assert_eq!(result.groups[1].functional.registers[0][1], 88);
        assert_eq!(
            result.groups[1].functional.shared_memory.get(&500),
            Some(&77)
        );
        assert_eq!(result.groups[1].functional.memory.get(&600), Some(&88));
    }
}
