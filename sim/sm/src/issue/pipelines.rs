//! Semantic submodule; report-only boundary state and serde contracts.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct MemoryIssuePipelineState {
    pub(super) config: MemoryIssuePipeline,
    pub(super) queue_release_times: [VecDeque<u64>; SUB_CORES],
    pub(super) local_completion_ready: [u64; SUB_CORES],
    pub(super) shared_accept_ready: u64,
    pub(super) diagnostics: MemoryIssuePipelineDiagnostics,
}

/// Scheduler-visible occupancy for the architecture's scalar INT/FP
/// execution stream.  Dispatch remains one instruction per SMSP cycle; an
/// entry reserves a future execution slot and leaves the window when that
/// slot starts.  This captures finite pipelined service without turning the
/// model into a register scoreboard or a per-cycle global scan.
/// Research-only scalar resource candidate. It is intentionally not wired
/// into production IssueModelParams; the bounded issue-port probe found that a
/// single generic queue does not transfer across independent and mixed
/// workloads. Keeping the implementation here makes future probes
/// reproducible without silently changing the deployed lane.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(super) struct ScalarExecutionPipelineState {
    pub(super) config: ScalarExecutionPipeline,
    pub(super) release_times: [VecDeque<u64>; SUB_CORES],
    pub(super) next_service: [u64; SUB_CORES],
    pub(super) diagnostics: ScalarExecutionPipelineDiagnostics,
}

#[allow(dead_code)]
impl ScalarExecutionPipelineState {
    pub(super) fn new(config: ScalarExecutionPipeline) -> Result<Self, IssueError> {
        if config.latency_cycles == 0
            || config.initiation_interval_cycles == 0
            || config.in_flight_capacity == 0
        {
            return Err(IssueError::ScalarExecutionPipelineConfigurationInvalid {
                latency: config.latency_cycles,
                initiation: config.initiation_interval_cycles,
                capacity: config.in_flight_capacity,
            });
        }
        Ok(Self {
            config,
            release_times: std::array::from_fn(|_| VecDeque::new()),
            next_service: [0; SUB_CORES],
            diagnostics: ScalarExecutionPipelineDiagnostics::default(),
        })
    }

    pub(super) fn prune(&mut self, now: u64) {
        for queue in &mut self.release_times {
            while queue.front().is_some_and(|&release| release <= now) {
                queue.pop_front();
            }
        }
    }

    pub(super) fn admission_ready_at(&self, subcore: usize) -> u64 {
        let queue = &self.release_times[subcore];
        if queue.len() < self.config.in_flight_capacity as usize {
            0
        } else {
            queue.front().copied().unwrap_or(0)
        }
    }

    pub(super) fn plan(
        &self,
        selected: [Option<PipelineClass>; SUB_CORES],
        now: u64,
    ) -> Result<ScalarExecutionPipelinePlan, IssueError> {
        let mut next = self.clone();
        for (subcore, slot) in selected.into_iter().enumerate() {
            let Some(PipelineClass::ScalarAlu {
                latency_cycles,
                initiation_interval_cycles,
                ..
            }) = slot
            else {
                continue;
            };
            if next.release_times[subcore].len() >= self.config.in_flight_capacity as usize {
                return Err(IssueError::ScalarExecutionPipelineQueueFull { subcore, time: now });
            }
            let ready_for_service =
                checked_cycle_add("scalar execution pipeline latency", now, latency_cycles)?;
            let service_start = ready_for_service.max(next.next_service[subcore]);
            next.next_service[subcore] = checked_cycle_add(
                "scalar execution pipeline initiation",
                service_start,
                initiation_interval_cycles,
            )?;
            next.release_times[subcore].push_back(service_start);
            let diagnostics = &mut next.diagnostics;
            diagnostics.instructions_admitted = checked_u64_counter_add(
                "scalar execution pipeline admitted instruction",
                diagnostics.instructions_admitted,
                1,
            )?;
            diagnostics.service_busy_cycles = checked_u64_counter_add(
                "scalar execution pipeline service work",
                diagnostics.service_busy_cycles,
                initiation_interval_cycles,
            )?;
            diagnostics.queue_delay_cycles = checked_u64_counter_add(
                "scalar execution pipeline queue delay",
                diagnostics.queue_delay_cycles,
                service_start.saturating_sub(ready_for_service),
            )?;
            diagnostics.maximum_queue_occupancy = diagnostics.maximum_queue_occupancy.max(
                u64::try_from(next.release_times[subcore].len()).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "scalar execution pipeline queue occupancy",
                    }
                })?,
            );
        }
        Ok(ScalarExecutionPipelinePlan { next })
    }
}

impl MemoryIssuePipelineState {
    pub(super) fn new(config: MemoryIssuePipeline) -> Result<Self, IssueError> {
        if config.queue_entries_with_dispatch == 0 {
            return Err(IssueError::MemoryIssuePipelineConfigurationInvalid {
                field: "queue_entries_with_dispatch",
                value: 0,
            });
        }
        for (field, value) in [
            (
                "local_initiation_interval_cycles",
                config.local_initiation_interval_cycles,
            ),
            (
                "shared_accept_interval_cycles",
                config.shared_accept_interval_cycles,
            ),
        ] {
            if value == 0 {
                return Err(IssueError::MemoryIssuePipelineConfigurationInvalid { field, value });
            }
        }
        Ok(Self {
            config,
            queue_release_times: std::array::from_fn(|_| VecDeque::new()),
            local_completion_ready: [0; SUB_CORES],
            shared_accept_ready: 0,
            diagnostics: MemoryIssuePipelineDiagnostics::default(),
        })
    }

    pub(super) fn prune(&mut self, now: u64) {
        for queue in &mut self.queue_release_times {
            while queue.front().is_some_and(|&release| release <= now) {
                queue.pop_front();
            }
        }
    }

    pub(super) fn admission_ready_at(&self, subcore: usize) -> u64 {
        let queue = &self.queue_release_times[subcore];
        if queue.len() < self.config.queue_entries_with_dispatch as usize {
            0
        } else {
            queue.front().copied().unwrap_or(0)
        }
    }

    pub(super) fn plan(
        &self,
        selected: [Option<MemoryPipelineSlot>; SUB_CORES],
        now: u64,
    ) -> Result<MemoryIssuePipelinePlan, IssueError> {
        let mut next = self.clone();
        let mut issues = [None; SUB_CORES];
        for (subcore, slot) in selected.into_iter().enumerate() {
            let Some(slot) = slot else { continue };
            if next.queue_release_times[subcore].len()
                >= next.config.queue_entries_with_dispatch as usize
            {
                return Err(IssueError::MemoryIssuePipelineQueueFull { subcore, time: now });
            }
            let nominal_operand_ready = checked_cycle_add(
                "memory operand/address readiness",
                now,
                slot.operand_ready_latency,
            )?;
            let operand_ready_at = nominal_operand_ready.max(next.local_completion_ready[subcore]);
            next.local_completion_ready[subcore] = checked_cycle_add(
                "memory local initiation interval",
                operand_ready_at,
                next.config.local_initiation_interval_cycles,
            )?;
            let shared_accept_at = operand_ready_at.max(next.shared_accept_ready);
            next.shared_accept_ready = checked_cycle_add(
                "memory SM-shared acceptance interval",
                shared_accept_at,
                next.config.shared_accept_interval_cycles,
            )?;
            next.queue_release_times[subcore].push_back(shared_accept_at);
            let diagnostics = &mut next.diagnostics;
            diagnostics.instructions_admitted = checked_u64_counter_add(
                "memory-pipeline admitted instruction",
                diagnostics.instructions_admitted,
                1,
            )?;
            if slot.constant_pressure {
                diagnostics.constant_pressure_instructions = checked_u64_counter_add(
                    "memory-pipeline constant-pressure instruction",
                    diagnostics.constant_pressure_instructions,
                    1,
                )?;
            }
            diagnostics.operand_pipeline_delay_cycles = checked_u64_counter_add(
                "memory-pipeline operand delay",
                diagnostics.operand_pipeline_delay_cycles,
                operand_ready_at - nominal_operand_ready,
            )?;
            diagnostics.shared_queue_delay_cycles = checked_u64_counter_add(
                "memory-pipeline shared queue delay",
                diagnostics.shared_queue_delay_cycles,
                shared_accept_at - operand_ready_at,
            )?;
            diagnostics.maximum_operand_pipeline_delay_cycles = diagnostics
                .maximum_operand_pipeline_delay_cycles
                .max(operand_ready_at - nominal_operand_ready);
            diagnostics.maximum_shared_queue_delay_cycles = diagnostics
                .maximum_shared_queue_delay_cycles
                .max(shared_accept_at - operand_ready_at);
            diagnostics.maximum_queue_occupancy = diagnostics.maximum_queue_occupancy.max(
                u64::try_from(next.queue_release_times[subcore].len()).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "memory-pipeline queue occupancy",
                    }
                })?,
            );
            diagnostics.terminal_shared_accept_cycle = diagnostics
                .terminal_shared_accept_cycle
                .max(shared_accept_at);
            issues[subcore] = Some(MemoryPipelineIssue {
                operand_ready_at,
                shared_accept_at,
            });
        }
        Ok(MemoryIssuePipelinePlan { next, issues })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ReadinessCause {
    InstructionFrontEnd,
    DependenceCounterRead,
    DependenceCounterGlobalLoadOperandRead,
    DependenceCounterGlobalStoreOperandRead,
    DependenceCounterGlobalAtomicOperandRead,
    DependenceCounterGlobalReductionOperandRead,
    DependenceCounterNonMemoryWrite,
    DependenceCounterSharedWrite,
    DependenceCounterL1Tex,
    DependenceCounterUnknownOrigin,
    AsyncSource,
    AsyncDestinationLocal,
    AsyncDestinationSmService,
    CtaBarrierRendezvous,
    PipelineBarrier,
    PipelineBranch,
    PipelineTensor,
    PipelineSfu,
    PipelineAlu,
    MemoryAdmissionQueue,
    MemoryIssueQueue,
    MemoryWriteCompletion,
    RegisterRaw,
    RegisterPredicate,
    RegisterWaw,
    ControlStall,
}

impl From<DependencyOrigin> for ReadinessCause {
    fn from(origin: DependencyOrigin) -> Self {
        match origin {
            DependencyOrigin::Read => Self::DependenceCounterRead,
            DependencyOrigin::GlobalLoadOperandRead => Self::DependenceCounterGlobalLoadOperandRead,
            DependencyOrigin::GlobalStoreOperandRead => {
                Self::DependenceCounterGlobalStoreOperandRead
            }
            DependencyOrigin::GlobalAtomicOperandRead => {
                Self::DependenceCounterGlobalAtomicOperandRead
            }
            DependencyOrigin::GlobalReductionOperandRead => {
                Self::DependenceCounterGlobalReductionOperandRead
            }
            DependencyOrigin::NonMemoryWrite => Self::DependenceCounterNonMemoryWrite,
            DependencyOrigin::SharedWrite => Self::DependenceCounterSharedWrite,
            DependencyOrigin::L1Tex => Self::DependenceCounterL1Tex,
            DependencyOrigin::Unknown => Self::DependenceCounterUnknownOrigin,
        }
    }
}

impl ReadinessCauseCycles {
    pub(super) fn add_cause(
        &mut self,
        cause: ReadinessCause,
        delta: u64,
    ) -> Result<(), IssueError> {
        let (field, counter): (&'static str, &mut u64) = match cause {
            ReadinessCause::InstructionFrontEnd => (
                "instruction front-end readiness",
                &mut self.instruction_front_end,
            ),
            ReadinessCause::RegisterRaw => ("register RAW", &mut self.register_raw),
            ReadinessCause::RegisterPredicate => {
                ("register predicate", &mut self.register_predicate)
            }
            ReadinessCause::RegisterWaw => ("register WAW", &mut self.register_waw),
            ReadinessCause::ControlStall => ("control stall readiness", &mut self.control_stall),
            ReadinessCause::DependenceCounterRead => (
                "read-dependence counter readiness",
                &mut self.dependence_counter_read,
            ),
            ReadinessCause::DependenceCounterGlobalLoadOperandRead => (
                "global-load operand-read counter readiness",
                &mut self.dependence_counter_global_load_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalStoreOperandRead => (
                "global-store operand-read counter readiness",
                &mut self.dependence_counter_global_store_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalAtomicOperandRead => (
                "global-atomic operand-read counter readiness",
                &mut self.dependence_counter_global_atomic_operand_read,
            ),
            ReadinessCause::DependenceCounterGlobalReductionOperandRead => (
                "global-reduction operand-read counter readiness",
                &mut self.dependence_counter_global_reduction_operand_read,
            ),
            ReadinessCause::DependenceCounterNonMemoryWrite => (
                "non-memory write-dependence counter readiness",
                &mut self.dependence_counter_non_memory_write,
            ),
            ReadinessCause::DependenceCounterSharedWrite => (
                "shared write-dependence counter readiness",
                &mut self.dependence_counter_shared_write,
            ),
            ReadinessCause::DependenceCounterL1Tex => (
                "L1TEX-dependence counter readiness",
                &mut self.dependence_counter_l1tex,
            ),
            ReadinessCause::DependenceCounterUnknownOrigin => (
                "unknown-origin dependence counter readiness",
                &mut self.dependence_counter_unknown_origin,
            ),
            ReadinessCause::AsyncSource => {
                ("asynchronous-copy source readiness", &mut self.async_source)
            }
            ReadinessCause::AsyncDestinationLocal => (
                "asynchronous-copy local destination readiness",
                &mut self.async_destination_local,
            ),
            ReadinessCause::AsyncDestinationSmService => (
                "asynchronous-copy SM destination-service readiness",
                &mut self.async_destination_sm_service,
            ),
            ReadinessCause::CtaBarrierRendezvous => (
                "CTA barrier rendezvous readiness",
                &mut self.cta_barrier_rendezvous,
            ),
            ReadinessCause::PipelineBarrier => {
                ("CTA-barrier pipeline readiness", &mut self.pipeline_barrier)
            }
            ReadinessCause::PipelineBranch => {
                ("branch-resolution readiness", &mut self.pipeline_branch)
            }
            ReadinessCause::PipelineTensor => {
                ("tensor-pipeline readiness", &mut self.pipeline_tensor)
            }
            ReadinessCause::PipelineSfu => {
                ("uniform/SFU-pipeline readiness", &mut self.pipeline_sfu)
            }
            ReadinessCause::PipelineAlu => {
                ("scalar-ALU pipeline readiness", &mut self.pipeline_alu)
            }
            ReadinessCause::MemoryAdmissionQueue => (
                "memory-admission queue readiness",
                &mut self.memory_admission_queue,
            ),
            ReadinessCause::MemoryIssueQueue => (
                "memory issue-pipeline queue readiness",
                &mut self.memory_issue_queue,
            ),
            ReadinessCause::MemoryWriteCompletion => (
                "memory write-completion readiness",
                &mut self.memory_write_completion,
            ),
        };
        *counter = checked_u64_counter_add(field, *counter, delta)?;
        Ok(())
    }

    pub(super) fn add_eligible_not_selected(&mut self, delta: u64) -> Result<(), IssueError> {
        self.eligible_not_selected = checked_u64_counter_add(
            "eligible-not-selected readiness",
            self.eligible_not_selected,
            delta,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct FixedRegisterReady {
    /// Data-register RAW readiness.
    pub(super) raw: u64,
    /// Predicate-register readiness is tracked independently of data RAW.
    pub(super) predicate: u64,
    /// Destination overlap is a WAW/partial-lane overwrite dependency.
    pub(super) waw: u64,
}

impl FixedRegisterReady {
    pub(super) fn latest(self) -> u64 {
        self.raw.max(self.predicate).max(self.waw)
    }
}
