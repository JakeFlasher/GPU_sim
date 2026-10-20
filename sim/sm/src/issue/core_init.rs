//! SMSP construction, block admission, and residency initialization; mutates issue state.

use super::*;

impl<'a> SmCore<'a> {
    /// Build the actor and admit the initial block wave. `blocks` must be
    /// this SM's blocks in launch order; every instruction's pc must
    /// resolve in `schedule` (single-function launches — the replay
    /// driver enforces that before calling). Cache modes attach a FRESH
    /// per-SM L1 (cold cache per kernel, NCU `--cache-control all`
    /// parity).
    pub(crate) fn new_with_residency(
        blocks: &'a [BlockReplay],
        schedule: &'a FunctionSchedule,
        params: &IssueModelParams,
        shared_memory_residency: SharedMemoryResidency,
        mode: SmMemMode,
    ) -> Result<Self, IssueError> {
        Self::new_with_residency_and_observer(
            blocks,
            schedule,
            params,
            shared_memory_residency,
            mode,
            None,
        )
    }
    pub(crate) fn new_with_residency_and_observer(
        blocks: &'a [BlockReplay],
        schedule: &'a FunctionSchedule,
        params: &IssueModelParams,
        shared_memory_residency: SharedMemoryResidency,
        mode: SmMemMode,
        write_lookup_observer: Option<Rc<RefCell<L1WriteLookupObserver>>>,
    ) -> Result<Self, IssueError> {
        params.validate_resource_source()?;
        if shared_memory_residency.max_bytes_per_sm() == 0 {
            return Err(IssueError::SharedMemoryCapacityInvalid {
                arch: params.arch,
                max_bytes_per_sm: shared_memory_residency.max_bytes_per_sm(),
            });
        }
        if params.arch != ArchId::Sm86
            && !matches!(params.resource_source, IssueResourceSource::Sm120Alpha)
        {
            return Err(IssueError::ArchitectureUnsupported { arch: params.arch });
        }
        if params.arch != schedule.arch {
            return Err(IssueError::ArchitectureMismatch {
                component: "issue parameters",
                expected: schedule.arch,
                got: params.arch,
            });
        }
        let physical_warp_slots =
            if matches!(params.resource_source, IssueResourceSource::Sm120Alpha) {
                params.max_resident_warps_per_sm
            } else {
                params
                    .arch
                    .issue_hypotheses()
                    .require("physical_warp_slots")?
                    .max_resident_warps_per_sm
            };
        let schedule_slots = schedule
            .by_pc
            .iter()
            .map(|(&pc, slot)| (pc, slot))
            .collect::<Vec<_>>();
        let constant_streams = constant_stream_count(schedule);
        let constant_pressure_threshold = params
            .memory_issue_pipeline
            .map_or(u32::MAX, |config| config.queue_entries_with_dispatch);
        let mut constant_path_pcs = schedule
            .by_pc
            .iter()
            .filter_map(|(&pc, slot)| {
                deep_constant_load(slot, constant_streams, constant_pressure_threshold)
                    .then_some(pc)
            })
            .collect::<Vec<_>>();
        constant_path_pcs.sort_unstable();
        let immediate_constant_config = matches!(mode, SmMemMode::L1Shared(_))
            .then_some(params.immediate_constant_cache)
            .flatten();
        let immediate_constant_schedule =
            immediate_constant_config.map_or_else(Vec::new, |config| {
                schedule
                    .by_pc
                    .iter()
                    .filter_map(|(&pc, slot)| {
                        let lines = immediate_constant_lines(slot.opcode_text(), config.line_bytes);
                        (!lines.is_empty()).then_some((pc as u32, lines))
                    })
                    .collect::<Vec<_>>()
            });
        let immediate_constant_cache: Option<Box<[ImmediateConstantCacheState; SUB_CORES]>> =
            immediate_constant_config
                .map(|config| {
                    Ok::<Box<[ImmediateConstantCacheState; SUB_CORES]>, IssueError>(Box::new([
                        ImmediateConstantCacheState::new(config)?,
                        ImmediateConstantCacheState::new(config)?,
                        ImmediateConstantCacheState::new(config)?,
                        ImmediateConstantCacheState::new(config)?,
                    ]))
                })
                .transpose()?;
        let mut memory_pipeline_schedule = Vec::new();
        if let Some(config) = params.memory_issue_pipeline {
            for (&pc, slot) in &schedule.by_pc {
                if let Some(memory_slot) =
                    memory_pipeline_slot(slot, config, constant_path_pcs.binary_search(&pc).is_ok())
                {
                    memory_pipeline_schedule.push((pc, memory_slot));
                }
            }
        }
        // Validate the complete dynamic/static join and every actor-cardinality
        // conversion before constructing mutable replay state. A later block
        // admission must not discover an unknown PC or overflow after earlier
        // blocks have already committed instructions.
        let mut block_live_warps = Vec::with_capacity(blocks.len());
        let mut instruction_limit = 0u64;
        let mut arrival_limit = 0u64;
        let mut sm120_flat_global_sectors = 0u64;
        let mut sm120_flat_shared_wavefronts = 0u64;
        let mut sm120_flat_live_warps = 0u32;
        let mut sm120_flat_async_group_count = 0u64;
        let mut sm120_flat_atomic_present = false;
        let has_async_copy_protocol = schedule.by_pc.values().any(|slot| {
            matches!(
                slot.op_class,
                OpClass::AsyncCopyCommit | OpClass::AsyncCopyWait
            ) || slot.mem_role == sim_mem::MemRole::BypassLoad(CacheOpToken::Ldgsts)
        });
        let has_nonzero_async_wait = schedule.by_pc.values().any(|slot| {
            slot.op_class == OpClass::AsyncCopyWait && slot.async_copy_wait_max_pending_groups != 0
        });
        for block in blocks {
            let live_warp_count = block
                .warps
                .iter()
                .filter(|warp| !warp.insts.is_empty())
                .count();
            let live_warps = u32::try_from(live_warp_count).map_err(|_| {
                IssueError::LiveWarpCountOutOfRange {
                    launch_index: block.launch_index,
                    warps: live_warp_count,
                }
            })?;
            sm120_flat_live_warps = sm120_flat_live_warps.saturating_add(live_warps);
            for warp in &block.warps {
                for inst in &warp.insts {
                    if schedule
                        .issue_slot(u64::from(inst.pc))
                        .is_some_and(|slot| opcode_head(slot.opcode_text()) == "LDGDEPBAR")
                    {
                        sm120_flat_async_group_count =
                            sm120_flat_async_group_count.saturating_add(1);
                    }
                    match inst.kind {
                        AccessKind::GlobalBypass {
                            token: CacheOpToken::Atom,
                            len,
                            ..
                        } => {
                            sm120_flat_atomic_present = true;
                            sm120_flat_global_sectors =
                                sm120_flat_global_sectors.saturating_add(u64::from(len));
                        }
                        AccessKind::GlobalReduction { len, .. } => {
                            sm120_flat_global_sectors =
                                sm120_flat_global_sectors.saturating_add(u64::from(len));
                        }
                        AccessKind::GlobalEligible { len, .. }
                        | AccessKind::GlobalBypass { len, .. }
                        | AccessKind::GlobalUntagged { len, .. }
                        | AccessKind::GlobalStore { len, .. } => {
                            sm120_flat_global_sectors =
                                sm120_flat_global_sectors.saturating_add(u64::from(len));
                        }
                        AccessKind::Shared { wavefronts, .. } => {
                            sm120_flat_shared_wavefronts =
                                sm120_flat_shared_wavefronts.saturating_add(wavefronts);
                        }
                        AccessKind::NonMemory
                        | AccessKind::PredicatedOffMemory
                        | AccessKind::LocalLoad { .. }
                        | AccessKind::LocalStore { .. }
                        | AccessKind::GlobalAsyncCopy { .. } => {}
                    }
                }
            }
            if live_warps > params.max_resident_warps_per_sm {
                return Err(IssueError::BlockResidencyLimitExceeded {
                    launch_index: block.launch_index,
                    live_warps,
                    arch: params.arch,
                    max_resident_warps: params.max_resident_warps_per_sm,
                });
            }
            if live_warps != 0 && params.max_resident_blocks_per_sm == 0 {
                return Err(IssueError::BlockCountResidencyLimitExceeded {
                    launch_index: block.launch_index,
                    arch: params.arch,
                    max_resident_blocks: params.max_resident_blocks_per_sm,
                });
            }
            if live_warps != 0
                && shared_memory_residency.bytes_per_cta()
                    > shared_memory_residency.max_bytes_per_sm()
            {
                return Err(IssueError::BlockSharedMemoryLimitExceeded {
                    launch_index: block.launch_index,
                    bytes_per_cta: shared_memory_residency.bytes_per_cta(),
                    arch: params.arch,
                    max_bytes_per_sm: shared_memory_residency.max_bytes_per_sm(),
                });
            }
            let mut block_slots = HashSet::with_capacity(block.warps.len());
            for warp in &block.warps {
                let slot =
                    u32::try_from(warp.warp_slot).map_err(|_| IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: params.arch,
                        physical_warp_slots,
                    })?;
                if slot >= physical_warp_slots {
                    return Err(IssueError::WarpSlotOutOfRange {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                        arch: params.arch,
                        physical_warp_slots,
                    });
                }
                if !block_slots.insert(warp.warp_slot) {
                    return Err(IssueError::DuplicateWarpSlot {
                        launch_index: block.launch_index,
                        warp_slot: warp.warp_slot,
                    });
                }
            }
            block_live_warps.push(live_warps);
            arrival_limit = checked_u64_counter_add(
                "warp arrival stamp",
                arrival_limit,
                u64::from(live_warps),
            )?;
            for warp in &block.warps {
                let warp_instructions =
                    u64::try_from(warp.insts.len()).map_err(|_| IssueError::CounterOverflow {
                        counter: "replay instruction cardinality",
                    })?;
                instruction_limit = checked_u64_counter_add(
                    "replay instruction cardinality",
                    instruction_limit,
                    warp_instructions,
                )?;
                for inst in &warp.insts {
                    let slot = lookup_dense(
                        &schedule_slots,
                        schedule,
                        inst.pc,
                        block.launch_index,
                        warp.warp_slot,
                    )?;
                    slot.control
                        .validate()
                        .map_err(|source| IssueError::ControlDomain {
                            pc: inst.pc,
                            launch_index: block.launch_index,
                            warp_slot: warp.warp_slot,
                            source,
                        })?;
                    validate_sector_span(
                        inst.kind,
                        warp.sectors.len(),
                        inst.pc,
                        block.launch_index,
                        warp.warp_slot,
                    )?;
                }
            }
        }
        let unmodeled_tails = blocks
            .iter()
            .flat_map(|block| &block.warps)
            .flat_map(|warp| &warp.insts)
            .fold(TailFlags::default(), |flags, inst| {
                flags.union(TailFlags::for_access(inst.kind))
            });
        let l1_arch = match mode {
            SmMemMode::Flat => None,
            SmMemMode::L1Const(cfg) | SmMemMode::L1Shared(cfg) => Some(cfg.arch),
        };
        if let Some(got) = l1_arch
            && got != schedule.arch
        {
            return Err(IssueError::ArchitectureMismatch {
                component: "L1 configuration",
                expected: schedule.arch,
                got,
            });
        }
        if let SmMemMode::L1Const(cfg) | SmMemMode::L1Shared(cfg) = mode {
            params.validate_l1_timing(cfg)?;
        }
        let mem = match mode {
            SmMemMode::Flat => MemModel::Flat,
            SmMemMode::L1Const(cfg) => MemModel::L1Const(L1Cache::new(cfg)?),
            SmMemMode::L1Shared(cfg) => MemModel::L1Shared(SharedL2State {
                l1: L1Cache::new(cfg)?,
                l1_visibility_sequence: 0,
                l1_visibility_observations: Vec::new(),
                l1_fence_observations: Vec::new(),
                l1_fill_visibility_observations: Vec::new(),
                l1_invalidate_all_observations: Vec::new(),
                l1_visibility_requester: None,
                payload_store_writers: None,
                response_owner: allocate_response_owner()?,
                outstanding_replies: HashSet::new(),
                outstanding_response_ids: HashMap::new(),
                response_masks: HashMap::new(),
                fill_waiters: HashMap::new(),
                fill_masks: HashMap::new(),
                line_folds: HashMap::new(),
                retry: VecDeque::new(),
            }),
        };
        let disable_reference_front_end = {
            #[cfg(feature = "reference")]
            {
                std::env::var_os("SIM_SM_REFERENCE_DISABLE_FRONTEND").is_some()
            }
            #[cfg(not(feature = "reference"))]
            {
                false
            }
        };
        let instruction_front_end = if disable_reference_front_end {
            None
        } else {
            params
                .instruction_front_end
                .map(|config| InstructionFrontEndState::new(config, params.l1_miss_latency))
                .transpose()?
        };
        let memory_issue_pipeline = params
            .memory_issue_pipeline
            .map(MemoryIssuePipelineState::new)
            .transpose()?;
        let scalar_execution_pipeline = params
            .scalar_execution_pipeline
            .map(ScalarExecutionPipelineState::new)
            .transpose()?
            .map(Box::new);
        // The observer is consumed only by shared-L2 cache accounting. In
        // particular, a flat replay must not depend on cache geometry.
        let stateful_write_lookup_observer = matches!(mode, SmMemMode::L1Shared { .. })
            && stateful_write_lookup_observer_enabled(schedule);
        let write_lookup_observer = if stateful_write_lookup_observer {
            match write_lookup_observer {
                Some(observer) => Some(observer),
                None => Some(Rc::new(RefCell::new(
                    L1WriteLookupObserver::for_arch(params.arch).map_err(IssueError::from)?,
                ))),
            }
        } else {
            None
        };
        let mut barrier_issue = None;
        for slot in schedule.by_pc.values() {
            let PipelineClass::Barrier {
                warp_initiation_interval_cycles,
                generation_rearm_cycles,
            } = slot.pipeline_class
            else {
                continue;
            };
            let declared = CtaBarrierIssue {
                warp_initiation_interval_cycles,
                generation_rearm_cycles,
            };
            if let Some(expected) = barrier_issue
                && expected != declared
            {
                return Err(IssueError::BarrierPipelineConfigurationMismatch {
                    expected,
                    got: declared,
                });
            }
            barrier_issue = Some(declared);
        }
        let has_fixed_writeback = schedule.by_pc.values().any(|slot| {
            matches!(
                result_writeback_kind(slot, params.arch),
                Some(ResultWritebackKind::FixedHadd2 { .. })
            )
        });
        let has_variable_writeback = schedule.by_pc.values().any(|slot| {
            matches!(
                result_writeback_kind(slot, params.arch),
                Some(ResultWritebackKind::L1TexStrongSmU32 { .. })
            )
        });
        let result_writebacks = if has_fixed_writeback && has_variable_writeback {
            Some(ResultWritebackState {
                timing: params
                    .arch
                    .fixed_l1tex_writeback_priority()
                    .require("fixed_l1tex_writeback_priority")?,
                pending: Vec::new(),
            })
        } else {
            None
        };
        let readiness = ReadinessDiagnostics {
            async_wait_prefix: has_nonzero_async_wait
                .then(|| Box::new(AsyncWaitPrefixDiagnostics::default())),
            sm86_register_read: (params.arch == ArchId::Sm86
                && matches!(mode, SmMemMode::L1Shared(_)))
            .then(|| Box::new(Sm86RegisterReadDiagnostics::default())),
            ..ReadinessDiagnostics::default()
        };
        #[cfg(feature = "reference")]
        let dense_reference = reference_mode_from_env()?;
        #[cfg(not(feature = "reference"))]
        let dense_reference = false;
        let mut core = Self {
            schedule,
            schedule_slots,
            memory_pipeline_schedule: memory_pipeline_schedule.into_boxed_slice(),
            constant_path_pcs: constant_path_pcs.into_boxed_slice(),
            immediate_constant_schedule: immediate_constant_schedule.into_boxed_slice(),
            immediate_constant_cache,
            immediate_constant_switch_remaining: [0; SUB_CORES],
            params: *params,
            shared_memory_residency,
            physical_warp_slots,
            instruction_front_end,
            memory_issue_pipeline,
            mem,
            blocks,
            block_live_warps,
            instruction_limit,
            warps: Vec::new(),
            async_copies: Vec::new(),
            has_async_copy_protocol,
            block_states: Vec::new(),
            subcore_warps: Default::default(),
            hold: [None; SUB_CORES],
            fetch_last_issued: [None; SUB_CORES],
            barrier_issue,
            barrier_pipe_ready: 0,
            tensor_pipe_ready: [0; SUB_CORES],
            sfu_pipe_ready: [0; SUB_CORES],
            scalar_execution_pipeline,
            sm86_rf_enabled: params.arch == ArchId::Sm86 && matches!(mode, SmMemMode::L1Shared(_)),
            sm86_rf_port_ready: [[0; 2]; SUB_CORES],
            sm86_rfc: std::array::from_fn(|_| VecDeque::new()),
            shared_write_lookup_runtime: Box::new(SharedWriteLookupRuntime {
                observer: write_lookup_observer,
                async_l1_hit_service_ready: [0; SUB_CORES],
            }),
            ldgsts_destination_pipe_ready: 0,
            result_writebacks,
            unmodeled_tails,
            readiness: Box::new(readiness),
            memory_service_candidates: MemoryServiceCandidateState::default(),
            emit_admission_witnesses: false,
            memory_admission_feedback: None,
            last_issue_picks: [None; SUB_CORES],
            instructions_per_smsp: [0; SUB_CORES],
            issue_active_cycles_per_smsp: [0; SUB_CORES],
            last_observed_issue: [None; SUB_CORES],
            warp_lifetimes: Vec::new(),
            issued: 0,
            // Keep the scheduler/event clock anchored at zero.  The
            // architecture-scoped SM-active observation boundary is an
            // explicit output projection (see `sm_active_observation_frontier`)
            // rather than hidden in every readiness timestamp and cumulative
            // diagnostic.  This preserves the causal lower-bound semantics of
            // the issue model and prevents a fixed observation interval from
            // being charged once per resident SM or per event.
            t: 0,
            local_completion_time: None,
            sm120_flat_service_enabled: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_live_warps == 1
                && (sm120_flat_global_sectors >= SM120_FLAT_SERVICE_SECTOR_THRESHOLD
                    || sm120_flat_shared_wavefronts >= SM120_FLAT_SERVICE_SECTOR_THRESHOLD),
            sm120_flat_global_service_ready: 0,
            sm120_flat_shared_service_ready: 0,
            sm120_flat_multiwarp_service_enabled: params.resource_source()
                == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_live_warps > 1
                && sm120_flat_global_sectors >= SM120_FLAT_SERVICE_SECTOR_THRESHOLD,
            sm120_flat_multiwarp_global_service_ready: 0,
            sm120_flat_multiwarp_dense: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_live_warps > 1
                && sm120_flat_global_sectors >= 8 * SM120_FLAT_SERVICE_SECTOR_THRESHOLD,
            sm120_flat_multiwarp_medium: params.resource_source()
                == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_live_warps > 1
                && sm120_flat_global_sectors >= SM120_FLAT_SERVICE_SECTOR_THRESHOLD,
            sm120_flat_shared_medium: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_live_warps == 1
                && sm120_flat_shared_wavefronts >= 128,
            sm120_async_sources: VecDeque::new(),
            sm120_flat_async_dense: params.resource_source() == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_async_group_count > 8,
            sm120_flat_async_service_enabled: params.resource_source()
                == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && sm120_flat_async_group_count <= 8
                && schedule
                    .by_pc
                    .values()
                    .any(|slot| opcode_head(slot.opcode_text()) == "LDGDEPBAR"),
            sm120_flat_async_service_ready: 0,
            // Fresh q7 group-depth probes are a logical overlap surface:
            // depth 1 needs no extra arm, depth 2 needs ~44 cycles/group,
            // and depth 4 needs ~13 cycles/group. This is explicitly shape
            // conditioned and never presented as a native queue width.
            sm120_flat_async_depth_service_per_group: if sm120_flat_async_group_count <= 64 {
                0
            } else if sm120_flat_async_group_count <= 128 {
                44
            } else if sm120_flat_async_group_count > 128 {
                13
            } else {
                0
            },
            sm120_flat_async_depth_service_ready: 0,
            sm120_flat_store_retire_enabled: params.resource_source()
                == IssueResourceSource::Sm120Alpha
                && matches!(mode, SmMemMode::Flat)
                && !sm120_flat_atomic_present
                && sm120_flat_live_warps == 1
                && (128..SM120_FLAT_SERVICE_SECTOR_THRESHOLD).contains(&sm120_flat_global_sectors),
            sm120_flat_store_retire_ready: 0,
            arrival_stamp: 0,
            resident_warps: 0,
            resident_shared_memory_bytes: 0,
            resident_warp_slots: Box::new(HashSet::new()),
            unfinished_warps: 0,
            next_block: 0,
            dense_reference,
            #[cfg(feature = "reference")]
            advance_diagnostics: AdvanceDiagnostics::default(),
            #[cfg(feature = "reference")]
            reference_issue_effects: Vec::new(),
        };
        core.admit_blocks(core.t)?;
        Ok(core)
    }
    pub(crate) fn payload_observation_enabled(&self) -> bool {
        matches!(&self.mem, MemModel::L1Shared(state) if state.payload_store_writers.is_some())
    }
    /// Admit blocks in launch order while all modeled constraints allow: the
    /// resident-block and resident-warp ceilings, the architecture-card
    /// shared-memory capacity, and disjoint physical warp slots across
    /// concurrent CTAs. A colliding later CTA
    /// waits for the earlier CTA to retire; reuse after retirement is valid. A
    /// block whose warps are ALL empty occupies no residency (it can never retire
    /// through the issue path, so counting it would leak resources and block later
    /// admissions forever).
    pub(super) fn admit_blocks(&mut self, t: u64) -> Result<(), IssueError> {
        while self.next_block < self.blocks.len() {
            let blk = &self.blocks[self.next_block];
            let live_warps = self.block_live_warps[self.next_block];
            let block_shared_memory_bytes = if live_warps == 0 {
                0
            } else {
                self.shared_memory_residency.bytes_per_cta()
            };
            let resident_blocks = self
                .block_states
                .iter()
                .filter(|state| state.unfinished != 0)
                .count() as u32;
            let next_resident_blocks = resident_blocks
                .checked_add(u32::from(live_warps != 0))
                .ok_or(IssueError::CounterOverflow {
                    counter: "resident blocks",
                })?;
            let Some(next_resident_warps) = self.resident_warps.checked_add(live_warps) else {
                // Construction proved this CTA fits alone, so an overflowing sum
                // can only mean it must wait for the current wave to retire.
                if self.resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident warp",
                });
            };
            let Some(next_resident_shared_memory_bytes) = self
                .resident_shared_memory_bytes
                .checked_add(block_shared_memory_bytes)
            else {
                if self.resident_warps > 0 {
                    break;
                }
                return Err(IssueError::CounterOverflow {
                    counter: "resident shared-memory byte",
                });
            };
            let pending_slots = blk
                .warps
                .iter()
                .filter(|warp| !warp.insts.is_empty())
                .map(|warp| warp.warp_slot)
                .collect::<Vec<_>>();
            let slots_collide = pending_slots
                .iter()
                .any(|slot| self.resident_warp_slots.contains(slot));
            if self.resident_warps > 0
                && (next_resident_blocks > self.params.max_resident_blocks_per_sm
                    || next_resident_warps > self.params.max_resident_warps_per_sm
                    || next_resident_shared_memory_bytes
                        > self.shared_memory_residency.max_bytes_per_sm()
                    || slots_collide)
            {
                break;
            }

            let admitted_warps =
                usize::try_from(live_warps).map_err(|_| IssueError::CounterOverflow {
                    counter: "admitted warp cardinality",
                })?;
            let next_unfinished_warps = checked_usize_counter_add(
                "unfinished warp",
                self.unfinished_warps,
                admitted_warps,
            )?;
            let next_arrival_stamp = checked_u64_counter_add(
                "warp arrival stamp",
                self.arrival_stamp,
                u64::from(live_warps),
            )?;
            let next_block = checked_usize_counter_add("next block", self.next_block, 1)?;

            // Build the complete block state off to the side. A schedule error
            // cannot leave a prefix of the block admitted.
            let block_ix = self.block_states.len();
            let first_warp = self.warps.len();
            let mut subcore_additions = [0usize; SUB_CORES];
            let mut pending_warps = Vec::with_capacity(admitted_warps);
            for w in &blk.warps {
                if w.insts.is_empty() {
                    continue;
                }
                let subcore =
                    usize::try_from(w.warp_slot).map_err(|_| IssueError::WarpSlotOutOfRange {
                        launch_index: blk.launch_index,
                        warp_slot: w.warp_slot,
                        arch: self.params.arch,
                        physical_warp_slots: self.physical_warp_slots,
                    })? % SUB_CORES;
                let pos_in_subcore = checked_usize_counter_add(
                    "sub-core resident warp",
                    self.subcore_warps[subcore].len(),
                    subcore_additions[subcore],
                )?;
                let arrival_offset = u64::try_from(pending_warps.len()).map_err(|_| {
                    IssueError::CounterOverflow {
                        counter: "warp arrival stamp",
                    }
                })?;
                let arrival = checked_u64_counter_add(
                    "warp arrival stamp",
                    self.arrival_stamp,
                    arrival_offset,
                )?;
                let next_slot = lookup_dense(
                    &self.schedule_slots,
                    self.schedule,
                    w.insts[0].pc,
                    blk.launch_index,
                    w.warp_slot,
                )?;
                pending_warps.push((
                    subcore,
                    WarpState {
                        insts: &w.insts,
                        sectors: &w.sectors,
                        async_copy_cohorts: w.async_copy_cohorts.as_deref(),
                        next: 0,
                        front_end_fetch_cursor: 0,
                        instruction_buffer: VecDeque::new(),
                        next_slot: Some(next_slot),
                        current_since: t,
                        admission_time: t,
                        eligible: t,
                        imc_wait_until: 0,
                        ordinary_dependency_wait_mask: next_slot.ordinary_dependency_wait_mask(),
                        barrier_clear: [0; DEP_BARRIERS],
                        barrier_origin: [DependencyOrigin::Unknown; DEP_BARRIERS],
                        barrier_pending: [0; DEP_BARRIERS],
                        register_ready: Vec::new(),
                        block: block_ix,
                        launch_index: blk.launch_index,
                        warp_id: w.warp_id,
                        warp_slot: w.warp_slot,
                        pos_in_subcore,
                        arrival,
                        at_block_barrier: false,
                        cta_barrier_ready: 0,
                        barrier_pipeline_ready: 0,
                        sfu_dependency_ready: 0,
                        branch_pipeline_ready: 0,
                        block_barrier_dispatch: None,
                        block_barrier_arrival: None,
                        clock_first_issue: None,
                        clock_reads_issued: 0,
                        done: false,
                    },
                ));
                subcore_additions[subcore] = checked_usize_counter_add(
                    "sub-core resident warp",
                    subcore_additions[subcore],
                    1,
                )?;
            }

            for (subcore, warp) in pending_warps {
                let ix = self.warps.len();
                self.subcore_warps[subcore].push(ix);
                self.warps.push(warp);
                if self.has_async_copy_protocol {
                    self.async_copies.push(AsyncCopySlot::default());
                }
            }
            debug_assert!(
                if self.has_async_copy_protocol {
                    self.warps.len() == self.async_copies.len()
                } else {
                    self.async_copies.is_empty()
                },
                "async-copy sidecar shape must match its construction-time protocol mode"
            );
            for &slot in &pending_slots {
                let inserted = self.resident_warp_slots.insert(slot);
                debug_assert!(
                    inserted,
                    "admission preflight rejected resident slot collisions"
                );
            }
            self.block_states.push(BlockState {
                unfinished: admitted_warps,
                waiting: 0,
                warp_count: live_warps,
                shared_memory_bytes: block_shared_memory_bytes,
                warp_slots: pending_slots.into_boxed_slice(),
                first_warp,
                admitted_warps,
                barrier_generation: 0,
                barrier_first_dispatch: None,
                barrier_first_arrival: None,
                barrier_rearm_ready: 0,
            });
            self.resident_warps = next_resident_warps;
            self.resident_shared_memory_bytes = next_resident_shared_memory_bytes;
            self.unfinished_warps = next_unfinished_warps;
            self.arrival_stamp = next_arrival_stamp;
            self.next_block = next_block;
        }
        Ok(())
    }
}
