//! Lower the bounded functional ISA into the shared engine's resource DAG.
//! Values are read at issue; RAW/WAW wait for completion. Memory reads are
//! complete at their response, so same-word RAW/WAR/WAW are explicitly ordered.

use crate::simt::{
    SimtError, SimtExecutor, SimtInstruction, SimtProgram, lane_address, operation_name,
};
use crate::spec::{ArchitectureSpec, ResourceKind};
use sim_engine::pipeline::{Resource, Task};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryDomain {
    Global,
    Shared,
}

pub fn resources(spec: &ArchitectureSpec) -> Result<Vec<Resource>, SimtError> {
    spec.resources()
        .iter()
        .map(|r| {
            Ok(Resource {
                width: r.issue_width(),
                latency: r.latency_cycles().ok_or(SimtError::UnknownLatency)?,
                initiation_interval: r.initiation_interval(),
                max_in_flight: r.max_in_flight(),
            })
        })
        .collect()
}

pub fn tasks(spec: &ArchitectureSpec, program: &SimtProgram) -> Result<Vec<Task>, SimtError> {
    SimtExecutor::new(spec, program)?;
    for name in spec.instruction_set() {
        if ![
            "set_lane_id",
            "load",
            "load_shared",
            "add_imm",
            "add",
            "broadcast_lane0",
            "reduce_add",
            "store",
            "store_shared",
            "atomic_add_shared",
            "async_store",
            "wait_group",
            "group_barrier",
            "halt",
        ]
        .contains(&name.as_str())
        {
            return Err(SimtError::UnsupportedInstruction(name.clone()));
        }
    }
    if spec
        .operation_resources()
        .keys()
        .any(|op| matches!(op.as_str(), "halt" | "wait_group" | "group_barrier"))
    {
        return Err(SimtError::InvalidProgram(
            "halt/wait/barrier are dependency drains, not resource operations",
        ));
    }
    let mut regs = [None; 32];
    let mut memory = BTreeMap::<(MemoryDomain, u32), (Option<usize>, Vec<usize>)>::new();
    let mut async_groups = BTreeMap::new();
    let mut tasks = Vec::new();
    let mut last_barrier = None;
    for (i, instruction) in program.instructions().iter().enumerate() {
        let mut deps = BTreeSet::new();
        deps.extend(last_barrier);
        let (sources, dst): (Vec<u8>, Option<u8>) = match instruction {
            SimtInstruction::SetLaneId { dst }
            | SimtInstruction::Load { dst, .. }
            | SimtInstruction::LoadShared { dst, .. } => (vec![], Some(*dst)),
            SimtInstruction::AddImm { src, dst, .. } => (vec![*src], Some(*dst)),
            SimtInstruction::BroadcastLane0 { src, dst } => (vec![*src], Some(*dst)),
            SimtInstruction::ReduceAdd { src, dst } => (vec![*src], Some(*dst)),
            SimtInstruction::Add { lhs, rhs, dst } => (vec![*lhs, *rhs], Some(*dst)),
            SimtInstruction::Store { src, .. }
            | SimtInstruction::StoreShared { src, .. }
            | SimtInstruction::AsyncStore { src, .. } => (vec![*src], None),
            SimtInstruction::AtomicAddShared { src, dst, .. } => (vec![*src], Some(*dst)),
            _ => (vec![], None),
        };
        for r in sources.into_iter().chain(dst) {
            deps.extend(regs[usize::from(r)]);
        }
        if let Some(dst) = dst {
            regs[usize::from(dst)] = Some(i);
        }
        let mut kind = ResourceKind::Compute;
        let mut work = 1;
        if let Some((domain, base, stride, write)) = memory_access(instruction) {
            kind = ResourceKind::Memory;
            work = u64::from(spec.execution().simt_width());
            // Deduplicate broadcast reads before mutating the per-word DAG.
            let addresses = (0..spec.execution().simt_width())
                .map(|lane| lane_address(base, stride, usize::from(lane)))
                .collect::<Result<BTreeSet<_>, _>>()?;
            for addr in addresses {
                let state = memory.entry((domain, addr)).or_default();
                deps.extend(state.0);
                if write {
                    deps.extend(state.1.iter().copied());
                    state.0 = Some(i);
                    state.1.clear();
                } else {
                    state.1.push(i);
                }
            }
        }
        let resource = match instruction {
            SimtInstruction::WaitGroup { group } => {
                deps.insert(
                    async_groups
                        .remove(group)
                        .ok_or(SimtError::MissingAsyncGroup(*group))?,
                );
                None
            }
            SimtInstruction::Halt | SimtInstruction::GroupBarrier { .. } => {
                if let Some(group) = async_groups.keys().next() {
                    return Err(SimtError::UnfinishedAsyncGroup(*group));
                }
                deps.extend(0..i);
                if matches!(instruction, SimtInstruction::GroupBarrier { .. }) {
                    last_barrier = Some(i);
                }
                None
            }
            _ => {
                if let SimtInstruction::AsyncStore { group, .. } = instruction
                    && async_groups.insert(*group, i).is_some()
                {
                    return Err(SimtError::AsyncGroupReuse(*group));
                }
                let default = if kind == ResourceKind::Memory {
                    "memory"
                } else {
                    "alu"
                };
                let name = spec
                    .operation_resources()
                    .get(operation_name(instruction))
                    .map_or(default, String::as_str);
                let index = spec
                    .resources()
                    .iter()
                    .position(|r| r.name() == name)
                    .ok_or_else(|| SimtError::MissingResource(name.to_owned()))?;
                if ![kind, ResourceKind::Custom].contains(&spec.resources()[index].kind()) {
                    return Err(SimtError::InvalidProgram(
                        "operation resource kind mismatch",
                    ));
                }
                Some(index)
            }
        };
        tasks.push(Task {
            resource,
            work,
            dependencies: deps.into_iter().collect(),
            cross_dependencies: Vec::new(),
        });
    }
    Ok(tasks)
}

pub fn memory_access(inst: &SimtInstruction) -> Option<(MemoryDomain, u32, u32, bool)> {
    match inst {
        SimtInstruction::Load { base, stride, .. } => {
            Some((MemoryDomain::Global, *base, *stride, false))
        }
        SimtInstruction::LoadShared { base, stride, .. } => {
            Some((MemoryDomain::Shared, *base, *stride, false))
        }
        SimtInstruction::Store { base, stride, .. }
        | SimtInstruction::AsyncStore { base, stride, .. } => {
            Some((MemoryDomain::Global, *base, *stride, true))
        }
        SimtInstruction::StoreShared { base, stride, .. } => {
            Some((MemoryDomain::Shared, *base, *stride, true))
        }
        SimtInstruction::AtomicAddShared { base, stride, .. } => {
            Some((MemoryDomain::Shared, *base, *stride, true))
        }
        _ => None,
    }
}
