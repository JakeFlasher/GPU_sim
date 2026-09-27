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
    spec::{ArchitectureSpec, ResourceKind},
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

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
    /// Causal/resource bounds for design-space studies. These values are
    /// derived only from the declarative schedule; they are not silicon or
    /// profiler counters.
    pub schedule_diagnostics: CustomScheduleDiagnostics,
    pub resources: Vec<CustomResourceReport>,
    pub groups: Vec<CustomGroupResult>,
}

#[derive(Debug, Serialize)]
pub struct CustomScheduleDiagnostics {
    /// Identifies the interpretation of the reported lower-bound fields. The
    /// scheduler first greedily chooses one concrete route for each task, so
    /// these are bounds for that selected route assignment, not a universal
    /// optimum over all legal alternative-resource assignments.
    pub lower_bound_scope: &'static str,
    /// Latest task issue timestamp in the shared event schedule.
    pub last_issue_cycle: u64,
    /// Latest task completion timestamp (equal to the schedule frontier).
    pub completion_frontier_cycle: u64,
    /// Longest explicit dependency path, using each task's declared service
    /// duration and excluding unrelated resource contention.
    pub dependency_critical_path_cycles: u64,
    /// Largest serialized issue-beat span on one resource, ending after the
    /// last beat rather than after its unused initiation interval. This is a
    /// lower bound excluding startup, latency, admission and arbitration.
    pub resource_serial_work_bound_cycles: u64,
    /// Maximum of the two conservative model-relative lower bounds above.
    /// The bound is conditional on the scheduler's selected resource route;
    /// it is not an optimization lower bound across alternative routes.
    pub model_lower_bound_cycles: u64,
    /// Completion frontier minus the lower bound (checked, never saturated).
    /// This residual is not attributed to a particular hardware cause.
    pub lower_bound_gap_cycles: u64,
    /// Resource attaining the serialized-work bound, if any resource task
    /// was present. The first declared resource wins a tie.
    pub bottleneck_resource: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CustomResourceReport {
    pub name: String,
    pub kind: ResourceKind,
    pub issue_width: u32,
    pub initiation_interval: u64,
    pub latency_cycles: Option<u64>,
    pub max_in_flight: Option<u32>,
    pub task_count: u64,
    pub total_work: u128,
    pub service_cycles: u128,
    pub peak_in_flight: u32,
    /// Number of work items the resource could initiate over the observed
    /// schedule horizon at its declared width and initiation interval.  This
    /// is a model-relative capacity, not a hardware counter.
    pub ideal_work_capacity: u128,
    /// Work divided by the resource's ideal issue capacity over the complete
    /// model-relative schedule. This is not a hardware utilization counter.
    pub utilization_pct: Option<f64>,
}

fn resource_reports(
    spec: &ArchitectureSpec,
    tasks: &[Vec<sim_engine::pipeline::Task>],
    schedule: &sim_engine::pipeline::Schedule,
) -> Result<Vec<CustomResourceReport>, SimtError> {
    let mut reports = spec
        .resources()
        .iter()
        .map(|resource| CustomResourceReport {
            name: resource.name().to_owned(),
            kind: resource.kind(),
            issue_width: resource.issue_width(),
            initiation_interval: resource.initiation_interval(),
            latency_cycles: resource.latency_cycles(),
            max_in_flight: resource.max_in_flight(),
            task_count: 0,
            total_work: 0,
            service_cycles: 0,
            peak_in_flight: 0,
            ideal_work_capacity: 0,
            utilization_pct: None,
        })
        .collect::<Vec<_>>();
    let mut occupancy_events = vec![Vec::<(u64, i32)>::new(); reports.len()];
    for (group, group_tasks) in tasks.iter().enumerate() {
        for (index, (task, interval)) in group_tasks.iter().zip(&schedule.groups[group]).enumerate()
        {
            let Some(resource) = schedule.selected_resources[group][index] else {
                continue;
            };
            let config = &reports[resource];
            let span = task.work.div_ceil(u64::from(config.issue_width));
            let service = span
                .checked_mul(config.initiation_interval)
                .ok_or(SimtError::CycleOverflow)?;
            let report = &mut reports[resource];
            report.task_count = report
                .task_count
                .checked_add(1)
                .ok_or(SimtError::CycleOverflow)?;
            report.total_work = report
                .total_work
                .checked_add(u128::from(task.work))
                .ok_or(SimtError::CycleOverflow)?;
            report.service_cycles = report
                .service_cycles
                .checked_add(u128::from(service))
                .ok_or(SimtError::CycleOverflow)?;
            // `max_in_flight` is enforced by the scheduler until completion,
            // not merely until the final initiation beat.  Use the scheduled
            // completion frontier here so the reported outstanding population
            // describes the same capacity domain as the engine.
            occupancy_events[resource].push((interval.start, 1));
            occupancy_events[resource].push((interval.complete, -1));
        }
    }
    for (index, events) in occupancy_events.iter_mut().enumerate() {
        events.sort_unstable_by_key(|&(time, delta)| (time, delta));
        let mut in_flight = 0_i32;
        for &(_, delta) in events.iter() {
            in_flight = in_flight.saturating_add(delta);
            reports[index].peak_in_flight = reports[index]
                .peak_in_flight
                .max(u32::try_from(in_flight).unwrap_or(0));
        }
        if schedule.final_cycle != 0 {
            // A resource with an initiation interval greater than one cannot
            // issue `width` work every cycle.  The old denominator used
            // final_cycle * width, overstating capacity (and understating
            // utilization) by exactly the initiation interval.  Count the
            // initiation slots in the inclusive [0, final_cycle) horizon;
            // using ceil keeps a one-beat operation that completes at the
            // horizon from producing a zero-capacity report.
            let interval = u128::from(reports[index].initiation_interval);
            let slots = u128::from(schedule.final_cycle)
                .checked_add(interval - 1)
                .and_then(|cycles| cycles.checked_div(interval))
                .ok_or(SimtError::CycleOverflow)?;
            let capacity = slots
                .checked_mul(u128::from(reports[index].issue_width))
                .ok_or(SimtError::CycleOverflow)?;
            reports[index].ideal_work_capacity = capacity;
            reports[index].utilization_pct =
                (capacity != 0).then(|| reports[index].total_work as f64 * 100.0 / capacity as f64);
        }
    }
    Ok(reports)
}

fn task_service_duration(
    resources: &[sim_engine::pipeline::Resource],
    task: &sim_engine::pipeline::Task,
    selected: Option<usize>,
) -> Result<u64, SimtError> {
    let Some(resource) = selected else {
        return Ok(0);
    };
    let config = resources
        .get(resource)
        .ok_or(SimtError::InvalidProgram("task resource is out of range"))?;
    let span = task.work.div_ceil(u64::from(config.width));
    let service = span
        .checked_mul(config.initiation_interval)
        .ok_or(SimtError::CycleOverflow)?;
    let last_beat_end = service
        .checked_sub(config.initiation_interval)
        .and_then(|value| value.checked_add(1))
        .ok_or(SimtError::CycleOverflow)?;
    last_beat_end
        .checked_add(config.latency)
        .ok_or(SimtError::CycleOverflow)
}

fn schedule_diagnostics(
    spec: &ArchitectureSpec,
    tasks: &[Vec<sim_engine::pipeline::Task>],
    schedule: &sim_engine::pipeline::Schedule,
) -> Result<CustomScheduleDiagnostics, SimtError> {
    let resources = simt_pipeline::resources(spec)?;
    let mut resource_work = vec![0_u64; resources.len()];
    let mut last_issue_cycle = 0_u64;
    let mut saw_task = false;
    for (group, group_tasks) in tasks.iter().enumerate() {
        for (index, task) in group_tasks.iter().enumerate() {
            let interval = schedule
                .groups
                .get(group)
                .and_then(|intervals| intervals.get(index))
                .ok_or(SimtError::InvalidProgram("schedule/task length mismatch"))?;
            saw_task = true;
            last_issue_cycle = last_issue_cycle.max(interval.start);
            if let Some(resource) = schedule.selected_resources[group][index] {
                let config = resources
                    .get(resource)
                    .ok_or(SimtError::InvalidProgram("task resource is out of range"))?;
                let span = task.work.div_ceil(u64::from(config.width));
                let service = span
                    .checked_mul(config.initiation_interval)
                    .ok_or(SimtError::CycleOverflow)?;
                resource_work[resource] = resource_work[resource]
                    .checked_add(service)
                    .ok_or(SimtError::CycleOverflow)?;
            }
        }
    }

    // Build the explicit dependency graph and evaluate its longest nominal
    // service path. Local dependencies are ordered by construction, while
    // barrier cross-dependencies are validated here instead of relying on an
    // incidental group iteration order.
    let mut offsets = Vec::with_capacity(tasks.len());
    let mut node_count = 0usize;
    for group in tasks {
        offsets.push(node_count);
        node_count = node_count
            .checked_add(group.len())
            .ok_or(SimtError::CycleOverflow)?;
    }
    let mut predecessors = vec![Vec::<usize>::new(); node_count];
    let mut durations = vec![0_u64; node_count];
    for (group, group_tasks) in tasks.iter().enumerate() {
        for (index, task) in group_tasks.iter().enumerate() {
            let node = offsets[group] + index;
            durations[node] =
                task_service_duration(&resources, task, schedule.selected_resources[group][index])?;
            let mut deps = BTreeSet::new();
            for dependency in &task.dependencies {
                deps.insert(offsets[group] + *dependency);
            }
            for (dependency_group, dependency_task) in &task.cross_dependencies {
                let Some(group_tasks) = tasks.get(*dependency_group) else {
                    return Err(SimtError::InvalidProgram(
                        "cross dependency group is out of range",
                    ));
                };
                if *dependency_task >= group_tasks.len() {
                    return Err(SimtError::InvalidProgram(
                        "cross dependency task is out of range",
                    ));
                }
                deps.insert(offsets[*dependency_group] + *dependency_task);
            }
            predecessors[node].extend(deps);
        }
    }
    let mut successors = vec![Vec::<usize>::new(); node_count];
    let mut indegree = vec![0_u32; node_count];
    for (node, deps) in predecessors.iter().enumerate() {
        indegree[node] = u32::try_from(deps.len()).map_err(|_| SimtError::CycleOverflow)?;
        for dependency in deps {
            successors[*dependency].push(node);
        }
    }
    let mut path_end = vec![0_u64; node_count];
    let mut ready = VecDeque::new();
    for (node, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            ready.push_back(node);
        }
    }
    let mut visited = 0usize;
    let mut dependency_critical_path_cycles = 0_u64;
    while let Some(node) = ready.pop_front() {
        visited += 1;
        path_end[node] = path_end[node]
            .checked_add(durations[node])
            .ok_or(SimtError::CycleOverflow)?;
        dependency_critical_path_cycles = dependency_critical_path_cycles.max(path_end[node]);
        for successor in &successors[node] {
            path_end[*successor] = path_end[*successor].max(path_end[node]);
            indegree[*successor] -= 1;
            if indegree[*successor] == 0 {
                ready.push_back(*successor);
            }
        }
    }
    if visited != node_count {
        return Err(SimtError::InvalidProgram(
            "custom dependency graph contains a cycle",
        ));
    }

    // The engine reserves span * II before another task may initiate, but
    // completion need not wait for the final unused II. For example one beat
    // with II=3 and latency=1 completes at cycle 2, not cycle 3. Sum the
    // reservations, remove the last interval, and include the last one-cycle
    // beat. Resource latency remains in the dependency-path bound above.
    for (work, resource) in resource_work.iter_mut().zip(&resources) {
        if *work != 0 {
            *work = work
                .checked_sub(resource.initiation_interval)
                .and_then(|value| value.checked_add(1))
                .ok_or(SimtError::CycleOverflow)?;
        }
    }
    let (bottleneck_index, resource_serial_work_bound_cycles) = resource_work
        .iter()
        .enumerate()
        .fold((None, 0), |best, (index, &work)| {
            if work > best.1 {
                (Some(index), work)
            } else {
                best
            }
        });
    let model_lower_bound_cycles =
        dependency_critical_path_cycles.max(resource_serial_work_bound_cycles);
    let completion_frontier_cycle = schedule.final_cycle;
    let lower_bound_gap_cycles = completion_frontier_cycle
        .checked_sub(model_lower_bound_cycles)
        .ok_or(SimtError::InvalidProgram(
            "schedule completion precedes its dependency/resource lower bound",
        ))?;
    let bottleneck_resource =
        bottleneck_index.map(|index| spec.resources()[index].name().to_owned());
    Ok(CustomScheduleDiagnostics {
        lower_bound_scope: "selected_route_assignment_v1",
        last_issue_cycle: if saw_task { last_issue_cycle } else { 0 },
        completion_frontier_cycle,
        dependency_critical_path_cycles,
        resource_serial_work_bound_cycles,
        model_lower_bound_cycles,
        lower_bound_gap_cycles,
        bottleneck_resource,
    })
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
        let mut timing = SimtTimingExecutor::new(spec, program)?.project_schedule_with_selection(
            &schedule.groups[g],
            &schedule.selected_resources[g],
        )?;
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
    let resources = resource_reports(spec, &tasks, &schedule)?;
    let schedule_diagnostics = schedule_diagnostics(spec, &tasks, &schedule)?;
    Ok(CustomResult {
        schema_version: 1,
        backend: "sim_sm_custom_event_pipeline",
        semantics: "model_relative_u32_simt_shared_barrier_ordered",
        architecture_spec: spec.clone(),
        final_cycle: schedule.final_cycle,
        completion_events: schedule.completion_events,
        schedule_diagnostics,
        resources,
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

    fn initiation_interval_spec() -> ArchitectureSpec {
        ArchitectureSpec::parse(
            Path::new("initiation_interval.json"),
            br#"{
              "schema_version": 1,
              "architecture": "example_simt8",
              "isa": "simt_u32_v1",
              "instruction_set": ["set_lane_id", "halt"],
              "execution": {"simt_width": 8, "max_resident_groups": 1},
              "resources": [{
                "name": "alu",
                "kind": "compute",
                "issue_width": 1,
                "latency_cycles": 1,
                "initiation_interval": 3
              }],
              "evidence": "unvalidated"
            }"#,
        )
        .expect("initiation interval spec should parse")
    }

    fn long_latency_spec() -> ArchitectureSpec {
        ArchitectureSpec::parse(
            Path::new("long_latency.json"),
            br#"{
              "schema_version": 1,
              "architecture": "example_simt8",
              "isa": "simt_u32_v1",
              "instruction_set": ["set_lane_id", "halt"],
              "execution": {"simt_width": 8, "max_resident_groups": 1},
              "resources": [{
                "name": "alu",
                "kind": "compute",
                "issue_width": 1,
                "latency_cycles": 20
              }],
              "evidence": "unvalidated"
            }"#,
        )
        .expect("long latency spec should parse")
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
        assert_eq!(result.resources.len(), 2);
        assert_eq!(result.resources[0].task_count, 5);
        assert_eq!(result.resources[1].task_count, 3);
        assert_eq!(result.resources[1].peak_in_flight, 3);
        let diagnostics = &result.schedule_diagnostics;
        assert_eq!(
            diagnostics.lower_bound_scope,
            "selected_route_assignment_v1"
        );
        assert_eq!(diagnostics.last_issue_cycle, 45);
        assert_eq!(diagnostics.completion_frontier_cycle, 45);
        assert_eq!(diagnostics.dependency_critical_path_cycles, 32);
        assert_eq!(diagnostics.resource_serial_work_bound_cycles, 24);
        assert_eq!(diagnostics.model_lower_bound_cycles, 32);
        assert_eq!(diagnostics.lower_bound_gap_cycles, 13);
        assert_eq!(diagnostics.bottleneck_resource.as_deref(), Some("memory"));
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

    #[test]
    fn utilization_accounts_for_initiation_interval_capacity() {
        let result = run(
            &initiation_interval_spec(),
            &[program(
                "single_set_lane.json",
                br#"{
                  "schema_version": 1,
                  "architecture": "example_simt8",
                  "isa": "simt_u32_v1",
                  "instructions": [
                    {"op":"set_lane_id", "dst":0},
                    {"op":"halt"}
                  ]
                }"#,
            )],
        )
        .expect("single task should execute");
        let alu = &result.resources[0];
        assert_eq!(result.final_cycle, 2);
        assert_eq!(alu.total_work, 1);
        assert_eq!(alu.ideal_work_capacity, 1);
        assert!((alu.utilization_pct.expect("capacity is nonzero") - 100.0).abs() < 1e-9);
        let diagnostics = &result.schedule_diagnostics;
        assert_eq!(diagnostics.resource_serial_work_bound_cycles, 1);
        assert_eq!(diagnostics.model_lower_bound_cycles, 2);
        assert_eq!(diagnostics.lower_bound_gap_cycles, 0);
    }

    #[test]
    fn schedule_bottleneck_ties_choose_first_declared_resource() {
        let spec = ArchitectureSpec::parse(
            Path::new("tied_resources.json"),
            br#"{
              "schema_version": 1,
              "architecture": "example_simt8",
              "isa": "simt_u32_v1",
              "instruction_set": ["set_lane_id", "add_imm", "halt"],
              "execution": {"simt_width": 8, "max_resident_groups": 1},
              "resources": [
                {"name":"first", "kind":"compute", "issue_width":1, "latency_cycles":1},
                {"name":"second", "kind":"compute", "issue_width":1, "latency_cycles":1}
              ],
              "operation_resources": {"set_lane_id":"first", "add_imm":"second"},
              "evidence": "unvalidated"
            }"#,
        )
        .expect("tie spec should parse");
        let result = run(
            &spec,
            &[program(
                "tie.json",
                br#"{
                  "schema_version": 1,
                  "architecture": "example_simt8",
                  "isa": "simt_u32_v1",
                  "instructions": [
                    {"op":"set_lane_id", "dst":0},
                    {"op":"add_imm", "dst":1, "src":0, "imm":1},
                    {"op":"halt"}
                  ]
                }"#,
            )],
        )
        .expect("tied resources should execute");
        assert_eq!(result.final_cycle, 4);
        assert_eq!(
            result
                .schedule_diagnostics
                .resource_serial_work_bound_cycles,
            1
        );
        assert_eq!(
            result.schedule_diagnostics.bottleneck_resource.as_deref(),
            Some("first")
        );
    }

    #[test]
    fn peak_in_flight_tracks_completion_not_issue_service() {
        let result = run(
            &long_latency_spec(),
            &[program(
                "independent_set_lanes.json",
                br#"{
                  "schema_version": 1,
                  "architecture": "example_simt8",
                  "isa": "simt_u32_v1",
                  "instructions": [
                    {"op":"set_lane_id", "dst":0},
                    {"op":"set_lane_id", "dst":1},
                    {"op":"halt"}
                  ]
                }"#,
            )],
        )
        .expect("independent resource operations should execute");
        assert_eq!(result.resources[0].peak_in_flight, 2);
    }

    #[test]
    fn schedule_diagnostics_do_not_invent_a_bottleneck_without_resource_work() {
        let result = run(
            &spec(),
            &[program(
                "halt_only.json",
                br#"{
                  "schema_version": 1,
                  "architecture": "example_simt8",
                  "isa": "simt_u32_v1",
                  "instructions": [{"op":"halt"}]
                }"#,
            )],
        )
        .expect("halt-only custom group should execute");
        let diagnostics = &result.schedule_diagnostics;
        assert_eq!(diagnostics.completion_frontier_cycle, 0);
        assert_eq!(diagnostics.dependency_critical_path_cycles, 0);
        assert_eq!(diagnostics.resource_serial_work_bound_cycles, 0);
        assert_eq!(diagnostics.bottleneck_resource, None);
    }
}
