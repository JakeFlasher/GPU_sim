// Lexical implementation section 4/8; included by the parent facade.
impl<'a> SmCore<'a> {#[inline]
    fn memory_pipeline_slot(&self, pc: u64) -> Option<MemoryPipelineSlot> {
        self.memory_pipeline_schedule
            .binary_search_by_key(&pc, |(key, _)| *key)
            .ok()
            .map(|index| self.memory_pipeline_schedule[index].1)
    }

    #[inline]
    fn is_constant_path_pc(&self, pc: u64) -> bool {
        self.constant_path_pcs.binary_search(&pc).is_ok()
    }

    #[inline]
    fn immediate_constant_lines(&self, pc: u32) -> Option<&[u64]> {
        self.immediate_constant_schedule
            .binary_search_by_key(&pc, |(key, _)| *key)
            .ok()
            .map(|index| self.immediate_constant_schedule[index].1.as_ref())
    }

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

    pub(crate) fn install_memory_service_candidates(
        &mut self,
        candidates: &[MemoryServiceCandidate],
    ) -> Result<(), IssueError> {
        self.memory_service_candidates = MemoryServiceCandidateState::checked(candidates)?;
        Ok(())
    }

    #[cfg(feature = "reference")]
    pub(crate) fn advance_diagnostics(&self) -> AdvanceDiagnostics {
        self.advance_diagnostics
    }

    /// Install one closed scalar macro into the native model actor. This
    /// reference-only seam is intentionally narrow: no front-end, memory,
    /// pipeline, barrier, or completion state may cross the boundary. The
    /// typed boundary executor independently validates the corresponding
    /// contract; this method only installs the already-derived state delta and
    /// then leaves `advance` to execute the continuation.
    #[cfg(feature = "reference")]
    pub(crate) fn execute_reference_scalar_macro(
        &mut self,
        warp_slot: i32,
        instruction_count: u64,
        initiation_interval_cycles: u64,
        terminal_block_barrier: bool,
        detailed_effects: &[TypedEventTransition],
        terminal_front_end_exit: Option<&InstructionFrontEndBoundaryState>,
    ) -> Result<ReferenceScalarMacroExecution, IssueError> {
        if instruction_count == 0 || initiation_interval_cycles == 0 {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "nonzero_count_and_initiation_interval_required",
            });
        }
        let memory_pipeline_open = self.memory_issue_pipeline.as_ref().is_some_and(|pipeline| {
            pipeline
                .queue_release_times
                .iter()
                .any(|queue| !queue.is_empty())
                || pipeline
                    .local_completion_ready
                    .iter()
                    .any(|&ready| ready > self.t)
                || pipeline.shared_accept_ready > self.t
        });
        let scalar_pipeline_open =
            self.scalar_execution_pipeline
                .as_ref()
                .is_some_and(|pipeline| {
                    pipeline.release_times.iter().any(|queue| !queue.is_empty())
                        || pipeline.next_service.iter().any(|&ready| ready > self.t)
                });
        let constant_cache_open = self
            .immediate_constant_cache
            .as_ref()
            .is_some_and(|caches| {
                caches.iter().any(|cache| {
                    !cache.pending.is_empty()
                        || cache.next_service > self.t
                        || cache.sets.iter().any(|set| !set.is_empty())
                })
            });
        if memory_pipeline_open
            || scalar_pipeline_open
            || constant_cache_open
            || !self.unmodeled_tails.is_empty()
            || self.outstanding_responses() != 0
            || self.block_states.iter().any(|block| block.waiting != 0)
            || self.warps.iter().any(|warp| {
                warp.at_block_barrier
                    || warp.barrier_pending.iter().any(|&pending| pending != 0)
                    || warp.next_slot.is_none() && !warp.done
            })
        {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "future_relevant_native_state_is_not_closed",
            });
        }
        let warp_index = self
            .warps
            .iter()
            .position(|warp| warp.warp_slot == warp_slot)
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "warp_slot_not_resident",
            })?;
        let (next, inst_len, launch_index) = {
            let warp = &self.warps[warp_index];
            (warp.next, warp.insts.len(), warp.launch_index)
        };
        let end_index = next
            .checked_add(usize::try_from(instruction_count).map_err(|_| {
                IssueError::ReferenceMacroUnsupported {
                    reason: "instruction_count_out_of_range",
                }
            })?)
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "instruction_count_overflow",
            })?;
        if end_index > inst_len {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "macro_exceeds_warp_stream",
            });
        }
        let instruction_count_usize = end_index - next;
        // A comparator-selected production witness carries one detailed issue
        // effect per collapsed instruction.  The model-local helper tests
        // intentionally omit effects and use the scalar recurrence, so keep
        // that compatibility path; whenever effects are supplied, however,
        // reject a truncated or non-monotone span before any actor mutation.
        if terminal_block_barrier && detailed_effects.len() != instruction_count_usize
            || (!detailed_effects.is_empty()
                && (detailed_effects.len() != instruction_count_usize
                    || detailed_effects
                        .windows(2)
                        .any(|window| window[1].event.time < window[0].event.time)))
        {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "detailed_effect_span_not_closed",
            });
        }
        if self.instruction_front_end.is_some() && !terminal_block_barrier {
            return Err(IssueError::ReferenceMacroUnsupported {
                reason: "front_end_transition_not_implemented",
            });
        }
        let front_end_transition = if terminal_block_barrier && self.instruction_front_end.is_some()
        {
            let boundary =
                terminal_front_end_exit.ok_or(IssueError::ReferenceMacroUnsupported {
                    reason: "terminal_front_end_exit_missing",
                })?;
            Some(InstructionFrontEndState::from_typed_boundary(boundary)?)
        } else {
            None
        };
        for (offset, instruction) in (next..end_index).enumerate() {
            let slot = lookup_dense(
                &self.schedule_slots,
                self.schedule,
                self.warps[warp_index].insts[instruction].pc,
                launch_index,
                warp_slot,
            )?;
            let terminal_barrier_slot =
                terminal_block_barrier && offset + 2 == instruction_count_usize;
            let terminal_exit_slot =
                terminal_block_barrier && offset + 1 == instruction_count_usize;
            let supported_issue_slot = terminal_barrier_slot
                || terminal_exit_slot
                || matches!(slot.pipeline_class, PipelineClass::ScalarAlu { .. });
            let terminal_sequence_slot = terminal_block_barrier
                && matches!(slot.pipeline_class, PipelineClass::DispatchOnly);
            if !supported_issue_slot && !terminal_sequence_slot
                || !matches!(slot.mem_role, MemRole::None)
                || (!terminal_barrier_slot
                    && !terminal_exit_slot
                    && slot.op_class != OpClass::Other)
                || (terminal_barrier_slot && slot.op_class != OpClass::BlockBarrier)
                || (terminal_exit_slot && !slot.opcode_text().starts_with("EXIT"))
            {
                return Err(IssueError::ReferenceMacroUnsupported {
                    reason: "scalar_macro_descriptor_not_closed",
                });
            }
        }
        // Resolve the detailed continuation slot before mutating the actor.
        // A malformed schedule must take exact fallback without leaving a
        // partially advanced warp behind.
        let next_slot = if end_index < inst_len {
            Some(lookup_dense(
                &self.schedule_slots,
                self.schedule,
                self.warps[warp_index].insts[end_index].pc,
                launch_index,
                warp_slot,
            )?)
        } else {
            None
        };
        let subcore = self.warps[warp_index]
            .warp_slot
            .rem_euclid(SUB_CORES as i32) as usize;
        let block = self.warps[warp_index].block;
        let first_issue = detailed_effects
            .first()
            .map(|effect| effect.event.time)
            .unwrap_or_else(|| self.t.max(self.warps[warp_index].eligible));
        let last_issue = detailed_effects
            .last()
            .map(|effect| effect.event.time)
            .or_else(|| {
                instruction_count
                    .checked_sub(1)
                    .and_then(|count| count.checked_mul(initiation_interval_cycles))
                    .and_then(|offset| first_issue.checked_add(offset))
            })
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "macro_timing_overflow",
            })?;
        let end_cycle = last_issue
            .checked_add(1)
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "macro_end_overflow",
            })?;
        self.warps[warp_index].next = end_index;
        self.warps[warp_index].eligible = if terminal_block_barrier {
            last_issue
        } else {
            end_cycle
        };
        self.warps[warp_index].current_since = end_cycle;
        self.warps[warp_index].next_slot = next_slot;
        if let Some(boundary) = terminal_front_end_exit {
            let _ = boundary;
            self.warps[warp_index].front_end_fetch_cursor = end_index;
            self.warps[warp_index].instruction_buffer.clear();
        }
        self.issued =
            self.issued
                .checked_add(instruction_count)
                .ok_or(IssueError::CounterOverflow {
                    counter: "reference macro issued",
                })?;
        self.instructions_per_smsp[subcore] = self.instructions_per_smsp[subcore]
            .checked_add(instruction_count)
            .ok_or(IssueError::CounterOverflow {
                counter: "reference macro SMSP instructions",
            })?;
        self.issue_active_cycles_per_smsp[subcore] = self.issue_active_cycles_per_smsp[subcore]
            .checked_add(instruction_count)
            .ok_or(IssueError::CounterOverflow {
                counter: "reference macro SMSP active cycles",
            })?;
        self.hold[subcore] = Some(warp_index);
        if let Some(front_end) = front_end_transition {
            self.instruction_front_end = Some(front_end);
        }
        if terminal_block_barrier {
            let barrier_issue = detailed_effects
                .get(detailed_effects.len().saturating_sub(2))
                .map(|effect| effect.event.time)
                .ok_or(IssueError::ReferenceMacroUnsupported {
                    reason: "terminal_barrier_effect_missing",
                })?;
            let barrier_arrival =
                barrier_issue
                    .checked_add(2)
                    .ok_or(IssueError::ReferenceMacroUnsupported {
                        reason: "terminal_barrier_arrival_overflow",
                    })?;
            let barrier_release =
                barrier_arrival
                    .checked_add(1)
                    .ok_or(IssueError::ReferenceMacroUnsupported {
                        reason: "terminal_barrier_release_overflow",
                    })?;
            self.block_states[block].waiting = 0;
            self.block_states[block].barrier_generation = self.block_states[block]
                .barrier_generation
                .checked_add(1)
                .ok_or(IssueError::CounterOverflow {
                    counter: "reference macro barrier generation",
                })?;
            self.block_states[block].barrier_rearm_ready = barrier_arrival;
            self.block_states[block].barrier_first_dispatch = None;
            self.block_states[block].barrier_first_arrival = None;
            self.barrier_pipe_ready = barrier_arrival;
            self.local_completion_time = Some(barrier_release);
            self.warps[warp_index].barrier_clear[0] = detailed_effects
                .get(detailed_effects.len().saturating_sub(4))
                .map(|effect| effect.event.time)
                .ok_or(IssueError::ReferenceMacroUnsupported {
                    reason: "terminal_barrier_clear_effect_missing",
                })?;
            self.warps[warp_index].barrier_pipeline_ready = barrier_release;
            self.warps[warp_index].at_block_barrier = true;
            self.warps[warp_index].next_slot = None;
            // Retire after the terminal EXIT issue.  `retire_warp` records
            // activity through `t + 1`; using the barrier release here would
            // shorten the warp lifetime relative to detailed replay even
            // though the barrier's local completion frontier is earlier.
            self.retire_warp(subcore, warp_index, last_issue)?;
            self.t = end_cycle;
        }
        self.t = end_cycle;
        Ok(ReferenceScalarMacroExecution {
            warp_slot,
            first_issue,
            last_issue,
            end_cycle,
            instruction_count,
        })
    }

    /// Consume the native comparator's selected execution path. Exact replay
    /// returns no macro transition; the typed-macro branch performs the native
    /// state mutation and returns its continuation frontier.
    #[cfg(feature = "reference")]
    pub(crate) fn execute_reference_macro_path(
        &mut self,
        path: &NativeBoundaryExecutionPath,
        warp_slot: i32,
        instruction_count: u64,
        initiation_interval_cycles: u64,
        terminal_block_barrier: bool,
        detailed_effects: &[TypedEventTransition],
        terminal_front_end_exit: Option<&InstructionFrontEndBoundaryState>,
    ) -> Result<Option<ReferenceScalarMacroExecution>, IssueError> {
        match path {
            NativeBoundaryExecutionPath::TypedMacro => self
                .execute_reference_scalar_macro(
                    warp_slot,
                    instruction_count,
                    initiation_interval_cycles,
                    terminal_block_barrier,
                    detailed_effects,
                    terminal_front_end_exit,
                )
                .map(Some),
            NativeBoundaryExecutionPath::ExactReplay { .. } => Ok(None),
        }
    }

    /// Consume a comparator-selected typed contract. This remains an opt-in
    /// reference seam; callers must validate the contract and continuation
    /// before invoking it, and exact replay is represented by `Ok(None)`.
    #[cfg(feature = "reference")]
    pub(crate) fn execute_reference_boundary_result(
        &mut self,
        result: &NativeBoundaryExecutionResult,
    ) -> Result<Option<ReferenceScalarMacroExecution>, IssueError> {
        let NativeBoundaryExecutionResult::TypedMacro { contract } = result else {
            return Ok(None);
        };
        let core = contract
            .entry
            .cores
            .first()
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "typed_macro_core_missing",
            })?;
        let warp = core
            .warp_states
            .first()
            .ok_or(IssueError::ReferenceMacroUnsupported {
                reason: "typed_macro_warp_missing",
            })?;
        let count = u64::try_from(contract.internal_effects.len()).map_err(|_| {
            IssueError::ReferenceMacroUnsupported {
                reason: "typed_macro_effect_count_out_of_range",
            }
        })?;
        let interval = contract
            .internal_effects
            .windows(2)
            .next()
            .and_then(|effects| effects[1].event.time.checked_sub(effects[0].event.time))
            .unwrap_or(1)
            .max(1);
        let terminal_block_barrier = contract
            .observable_labels
            .iter()
            .any(|label| label == "terminal_scalar_barrier_v1");
        let terminal_front_end_exit = contract
            .exit
            .cores
            .first()
            .and_then(|core| core.front_end.as_ref());
        self.execute_reference_macro_path(
            &NativeBoundaryExecutionPath::TypedMacro,
            warp.warp_slot,
            count,
            interval,
            terminal_block_barrier,
            &contract.internal_effects,
            terminal_front_end_exit,
        )
    }

    #[cfg(feature = "reference")]
    pub(crate) fn reference_issue_effects_since(&self, start: usize) -> &[(u64, u64, u32)] {
        self.reference_issue_effects.get(start..).unwrap_or(&[])
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_l2_request_key(&self, sm_id: u32, reply: ReplyTo) -> Option<L2RequestKey> {
        let response_id = reply.response_id()?;
        let response_kind = match reply {
            ReplyTo::L1Fill {
                fill: FillKey::Line { .. },
                ..
            } => TypedResponseKind::L1Fill,
            ReplyTo::L1Fill {
                fill: FillKey::Sector { .. },
                ..
            } => TypedResponseKind::L1SectorFill,
            ReplyTo::Warp { .. } => TypedResponseKind::Warp,
            ReplyTo::AsyncCopy { .. } => TypedResponseKind::AsyncCopy,
            ReplyTo::None => return None,
        };
        let warp_slot = self.warps.get(response_id.warp)?.warp_slot;
        let (response_warp, instruction, sector_index) = response_id.boundary_identity();
        Some(L2RequestKey {
            sm_id,
            response_warp,
            warp_slot,
            instruction,
            sector_index,
            response_kind,
        })
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_l2_request_owners(
        &self,
        sm_id: u32,
        response_due: &[(u64, u64, u64, u64, Option<u64>)],
    ) -> Vec<L2RequestOwner> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Vec::new();
        };
        let mut owners = state
            .outstanding_response_ids
            .iter()
            .map(|(response_id, reply)| {
                let direct_identity = self
                    .warps
                    .get(response_id.warp)
                    .and_then(|warp| warp.sectors.get(response_id.sector_index))
                    .map(|sector_id| {
                        (
                            CanonicalLineKey::VirtualAbsolute {
                                line_id: sector_id.div_euclid(4),
                            },
                            1_u8 << sector_id.rem_euclid(4) as u8,
                        )
                    });
                let (line, sector_mask, response_kind) = match reply {
                    ReplyTo::L1Fill { fill, .. } => match fill {
                        FillKey::Line { line, .. } => (
                            CanonicalLineKey::VirtualAbsolute { line_id: *line },
                            state
                                .response_masks
                                .get(response_id)
                                .copied()
                                .or_else(|| state.fill_masks.get(line).copied())
                                .unwrap_or(0),
                            TypedResponseKind::L1Fill,
                        ),
                        FillKey::Sector { line, sector, .. } => (
                            CanonicalLineKey::VirtualAbsolute { line_id: *line },
                            state
                                .response_masks
                                .get(response_id)
                                .copied()
                                .unwrap_or(1_u8 << *sector),
                            TypedResponseKind::L1SectorFill,
                        ),
                    },
                    ReplyTo::Warp { .. } => {
                        let (line, sector_mask) = direct_identity
                            .unwrap_or((CanonicalLineKey::Unknown { line_id: None }, 0));
                        (line, sector_mask, TypedResponseKind::Warp)
                    }
                    ReplyTo::AsyncCopy { .. } => {
                        let (line, sector_mask) = direct_identity
                            .unwrap_or((CanonicalLineKey::Unknown { line_id: None }, 0));
                        (line, sector_mask, TypedResponseKind::AsyncCopy)
                    }
                    ReplyTo::None => (
                        CanonicalLineKey::Unknown { line_id: None },
                        0,
                        TypedResponseKind::None,
                    ),
                };
                let warp_slot = self
                    .warps
                    .get(response_id.warp)
                    .map(|warp| warp.warp_slot)
                    .unwrap_or(-1);
                let (warp, instruction, sector_index) = response_id.boundary_identity();
                let due = response_due.iter().find(
                    |(candidate_warp, candidate_instruction, candidate_sector, _, _)| {
                        (*candidate_warp, *candidate_instruction, *candidate_sector)
                            == (warp, instruction, sector_index)
                    },
                );
                L2RequestOwner {
                    request: L2RequestKey {
                        sm_id,
                        response_warp: warp,
                        warp_slot,
                        instruction,
                        sector_index,
                        response_kind,
                    },
                    line,
                    sector_mask,
                    allocation_generation: due.and_then(|(_, _, _, _, generation)| *generation),
                    state: L2RequestState::Outstanding,
                    due: due.map(|(_, _, _, due, _)| *due),
                }
            })
            .collect::<Vec<_>>();
        owners.sort_by(|left, right| {
            (
                left.request.sm_id,
                left.request.response_warp,
                left.request.warp_slot,
                left.request.instruction,
                left.request.sector_index,
                &left.request.response_kind,
            )
                .cmp(&(
                    right.request.sm_id,
                    right.request.response_warp,
                    right.request.warp_slot,
                    right.request.instruction,
                    right.request.sector_index,
                    &right.request.response_kind,
                ))
        });
        owners
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_l1_state(&self) -> Option<L1BoundaryState> {
        self.mem.l1().map(L1Cache::typed_boundary_state)
    }

    /// Enable complete eligible-global L1 access observations for an explicit
    /// native visibility experiment. The requester is the independently
    /// retained physical `sm_id`, not an actor-vector index.
    pub(crate) fn enable_l1_visibility_observations(
        &mut self,
        requester: u8,
    ) -> Result<(), IssueError> {
        if requester >= 128 {
            return Err(IssueError::CounterOverflow {
                counter: "L1 visibility requester domain",
            });
        }
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 visibility requires shared-L2",
            });
        };
        st.l1_visibility_requester = Some(requester);
        Ok(())
    }

    pub(crate) fn enable_l1_resident_writeback(&mut self) -> Result<(), IssueError> {
        let Some(state) = self.mem.shared_l1_mut() else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 resident writeback requires shared-L2",
            });
        };
        state
            .l1
            .set_store_policy(L1StorePolicy::ResidentHitWriteBack);
        Ok(())
    }

    pub(crate) fn l1_dirty_residency(&self) -> sim_mem::L1DirtyResidency {
        self.mem
            .l1()
            .map_or_else(sim_mem::L1DirtyResidency::default, L1Cache::dirty_residency)
    }

    pub(crate) fn preview_l1_dirty_flush(&self) -> Result<sim_mem::L1DirtyFlushPlan, IssueError> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 dirty flush requires shared-L2",
            });
        };
        Ok(state.l1.preview_dirty_flush())
    }

    pub(crate) fn payload_observation_enabled(&self) -> bool {
        matches!(&self.mem, MemModel::L1Shared(state) if state.payload_store_writers.is_some())
    }

    pub(crate) fn preview_l1_dirty_payloads(
        &self,
    ) -> Result<Vec<sim_mem::SectorPayloadV1>, IssueError> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 payload preview requires shared-L2",
            });
        };
        state.l1.preview_dirty_payloads().map_err(IssueError::from)
    }

    pub(crate) fn commit_l1_dirty_flush(
        &mut self,
        plan: sim_mem::L1DirtyFlushPlan,
    ) -> Result<(), IssueError> {
        let Some(state) = self.mem.shared_l1_mut() else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 dirty flush requires shared-L2",
            });
        };
        state.l1.commit_dirty_flush(plan).map_err(IssueError::from)
    }

    pub(crate) fn preview_l1_dirty_payload_admission(
        &self,
        payloads: &[sim_mem::SectorPayloadV1],
        witnesses: &[sim_mem::L1PayloadResidencyWitness],
    ) -> Result<sim_mem::L1DirtyPayloadAdmissionPlan, IssueError> {
        let MemModel::L1Shared(state) = &self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 payload admission requires shared-L2",
            });
        };
        state
            .l1
            .preview_dirty_payload_admission_with_witness(payloads, witnesses)
            .map_err(IssueError::from)
    }

    pub(crate) fn commit_l1_dirty_payload_admission(
        &mut self,
        plan: sim_mem::L1DirtyPayloadAdmissionPlan,
    ) -> Result<(), IssueError> {
        let MemModel::L1Shared(state) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 payload admission requires shared-L2",
            });
        };
        state
            .l1
            .commit_dirty_payload_admission(plan)
            .map_err(IssueError::from)
    }

    /// Apply one event-time, witness-aware dirty-payload admission as a
    /// single native-facing operation. The underlying preview/commit pair is
    /// still exposed internally so a future event producer can retain the
    /// plan across an explicit scheduling boundary; this convenience path is
    /// atomic with respect to the current SM actor.
    #[allow(dead_code)] // no producer currently supplies this event to replay
    pub(crate) fn admit_l1_dirty_payloads_with_witness(
        &mut self,
        payloads: &[sim_mem::SectorPayloadV1],
        witnesses: &[sim_mem::L1PayloadResidencyWitness],
    ) -> Result<(), IssueError> {
        let plan = self.preview_l1_dirty_payload_admission(payloads, witnesses)?;
        self.commit_l1_dirty_payload_admission(plan)
    }

    pub(crate) fn watch_l1_payload_sectors(&mut self, payloads: &[sim_mem::SectorPayloadV1]) {
        if let MemModel::L1Shared(st) = &mut self.mem {
            let writers = st
                .payload_store_writers
                .get_or_insert_with(Default::default);
            for payload in payloads {
                writers
                    .entry((payload.line_id(), payload.sector()))
                    .or_insert(None);
            }
        }
    }

    /// The public replay path binds the site from the retained trace, never
    /// from a producer's guessed issue cycle or reused physical warp slot.
    pub(crate) fn admit_issued_l1_store_payloads(
        &mut self,
        site: PayloadStoreSite,
        at: u64,
        payloads: &[sim_mem::SectorPayloadV1],
        witnesses: &[sim_mem::L1PayloadResidencyWitness],
    ) -> Result<(), IssueError> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "store payload admission requires shared-L2",
            });
        };
        for payload in payloads {
            let line = payload.line_id();
            let sector = payload.sector();
            let refuse = |reason| IssueError::PayloadStoreBinding {
                line,
                sector,
                reason,
            };
            let writer = st
                .payload_store_writers
                .as_ref()
                .and_then(|writers| writers.get(&(line, sector)))
                .and_then(Option::as_ref)
                .ok_or_else(|| refuse("referenced store has not issued"))?;
            if writer.site != site {
                return Err(refuse("current writer is a different dynamic store"));
            }
            if writer.accepted_at > at {
                return Err(refuse("payload event precedes modeled store acceptance"));
            }
            if !writer.retained_dirty {
                return Err(refuse("store did not retain this sector dirty in L1"));
            }
            if writer.attached {
                return Err(refuse(
                    "payload already attached to this dynamic store sector",
                ));
            }
            if st
                .l1
                .resident_payload_state(line)
                .map(|(generation, _)| generation)
                != writer.resident_generation
            {
                return Err(refuse("resident allocation changed after the store"));
            }
        }
        st.l1.attach_observed_store_payloads(payloads, witnesses)?;
        for payload in payloads {
            if let Some(Some(writer)) = st
                .payload_store_writers
                .as_mut()
                .and_then(|writers| writers.get_mut(&(payload.line_id(), payload.sector())))
            {
                writer.attached = true;
            }
        }
        Ok(())
    }

    pub(crate) fn take_l1_visibility_events(&mut self) -> Vec<sim_mem::L1VisibilityEvent> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Vec::new();
        };
        let mut events = st
            .l1_visibility_observations
            .drain(..)
            .map(sim_mem::L1VisibilityEvent::Access)
            .chain(
                st.l1_fence_observations
                    .drain(..)
                    .map(sim_mem::L1VisibilityEvent::Fence),
            )
            .chain(
                st.l1_fill_visibility_observations
                    .drain(..)
                    .map(sim_mem::L1VisibilityEvent::FillVisible),
            )
            .chain(
                st.l1_invalidate_all_observations
                    .drain(..)
                    .map(sim_mem::L1VisibilityEvent::InvalidateAll),
            )
            .collect::<Vec<_>>();
        events.sort_unstable_by_key(|event| match event {
            sim_mem::L1VisibilityEvent::Access(observation) => observation.sequence,
            sim_mem::L1VisibilityEvent::Fence(observation) => observation.sequence,
            sim_mem::L1VisibilityEvent::FillVisible(observation) => observation.sequence,
            sim_mem::L1VisibilityEvent::InvalidateAll(observation) => observation.sequence,
        });
        events
    }

    /// Count visibility observations that have been produced by the SM but
    /// not yet transferred to the shared event queue.  The system layer drains
    /// this vector at every actor boundary; `advance` uses the count to stop
    /// immediately after an issue cycle that created a visibility event,
    /// preventing strict L1 replay from batching later issue cycles ahead of
    /// its own same-cycle coherence observation.
    pub(crate) fn pending_l1_visibility_event_count(&self) -> usize {
        let MemModel::L1Shared(st) = &self.mem else {
            return 0;
        };
        st.l1_visibility_observations
            .len()
            .saturating_add(st.l1_fence_observations.len())
            .saturating_add(st.l1_fill_visibility_observations.len())
            .saturating_add(st.l1_invalidate_all_observations.len())
    }

    pub(crate) fn invalidate_l1(
        &mut self,
        line_id: i64,
        sector_mask: u8,
    ) -> Result<(), IssueError> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 invalidation requires shared-L2",
            });
        };
        st.l1.invalidate(line_id, sector_mask)?;
        Ok(())
    }

    pub(crate) fn invalidate_l1_all(&mut self) -> Result<u64, IssueError> {
        let MemModel::L1Shared(st) = &mut self.mem else {
            return Err(IssueError::ResponseModeMismatch {
                mode: "L1 invalidation requires shared-L2",
            });
        };
        Ok(st.l1.invalidate_all()?)
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_local_completion_time(&self) -> Option<u64> {
        self.local_completion_time
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_unprojected_timing_state(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if self.sfu_pipe_ready.iter().any(|&ready| ready != 0) {
            // This resource survives CTA retirement/admission, but schema 15
            // does not serialize its availability. Never certify an exact
            // continuation using only the independent tensor frontier.
            reasons.push("sfu_pipeline_state".to_owned());
        }
        if !self.unmodeled_tails.is_empty() {
            // TailFlags are capability boundaries for completion effects that
            // are deliberately outside the modeled event stream. Preserve
            // them as a contract-level refusal instead of allowing a typed
            // boundary to imply complete writeback/coherence/RMW behavior.
            reasons.push("unmodeled_completion_tails".to_owned());
        }
        if let MemModel::L1Shared(state) = &self.mem
            && (!state.fill_waiters.is_empty()
                || !state.fill_masks.is_empty()
                || !state.line_folds.is_empty()
                || !state.retry.is_empty())
        {
            // These shared-L2 side maps select which parked sector range,
            // waiter, and per-fill mask transition runs after a response.
            // L1 tags and response owners alone cannot reconstruct that
            // transaction boundary.
            reasons.push("shared_l2_transient_state".to_owned());
        }
        if self
            .typed_l1_state()
            .is_some_and(|l1| !l1.line_fills.is_empty() || !l1.sector_fills.is_empty())
        {
            // The typed response owner carries L2 allocation identity, while
            // the L1 response handler consumes a private FillKey generation.
            // Without the owner-to-L1-fill edge, a live response cannot be
            // resumed exactly even though the L1 fill timing itself is typed.
            reasons.push("l1_fill_response_identity".to_owned());
        }
        if self
            .warps
            .iter()
            .any(|warp| warp.barrier_pending.iter().any(|&pending| pending != 0))
        {
            // Response owners retain requester/fill identity, but the typed
            // owner schema does not yet carry the dependency-barrier index
            // that each response clears. A live counter is therefore not
            // sufficient to resume the next issue transition exactly.
            reasons.push("dependency_barrier_ownership".to_owned());
        }
        let future_constant_path = self.warps.iter().any(|warp| {
            warp.insts.get(warp.next..).is_some_and(|remaining| {
                remaining
                    .iter()
                    .any(|instruction| self.is_constant_path_pc(u64::from(instruction.pc)))
            })
        });
        if future_constant_path
            && self
                .immediate_constant_cache
                .as_ref()
                .is_some_and(|caches| {
                    caches.iter().any(|cache| {
                        cache.sets.iter().any(|set| !set.is_empty())
                            || !cache.pending.is_empty()
                            || cache.next_service != 0
                            || !cache.diagnostics.is_empty()
                    })
                })
        {
            reasons.push("immediate_constant_cache_state".to_owned());
        }
        if self
            .shared_write_lookup_runtime
            .async_l1_hit_service_ready
            .iter()
            .any(|&ready| ready != 0)
        {
            reasons.push("async_l1_hit_service_state".to_owned());
        }
        if self.result_writebacks.is_some() {
            reasons.push("result_writeback_state".to_owned());
        }
        if !self.memory_service_candidates.is_empty() {
            reasons.push("memory_service_candidates".to_owned());
        }
        if self.memory_admission_feedback.is_some() {
            reasons.push("memory_admission_feedback".to_owned());
        }
        reasons.sort_unstable();
        reasons
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_async_copy_states(&self) -> Vec<Option<AsyncCopyBoundaryState>> {
        self.async_copies
            .iter()
            .map(|slot| {
                slot.state.as_ref().map(|state| {
                    let project = |group: AsyncCopyGroup| AsyncGroupBoundaryState {
                        sequence: group.id.0,
                        committed_at: group.committed_at,
                        pending_sources: group.pending_sources,
                        source_completion_time: group.source_completion_time,
                        delivery_work: group.delivery_work,
                        destination_wavefronts: group.destination_wavefronts,
                        destination_service_queued: group.destination_service_queued,
                        destination_completion_time: group.destination_completion_time,
                        completion_time: group.completion_time,
                    };
                    let mut bypass_observations = state
                        .bypass_copies
                        .iter()
                        .map(|copy| AsyncCopyBypassObservationState {
                            group: copy.group.0,
                            instruction: u64::try_from(copy.instruction).unwrap_or(u64::MAX),
                            pending_sources: copy.pending_sources,
                            source_completion_time: copy.source_completion_time,
                            delivery_work: copy.delivery_work,
                            source_cohorts: copy.source_cohorts,
                            destination_lines: copy.destination_lines,
                            max_bank_wavefronts: copy.max_bank_wavefronts,
                            max_bypass_wavefronts: copy.max_bypass_wavefronts,
                            sector_off: u64::try_from(copy.sector_off).unwrap_or(u64::MAX),
                            sector_membership_masks: copy.sector_membership_masks.to_vec(),
                            cohorts: copy
                                .cohorts
                                .iter()
                                .map(|cohort| AsyncCopyCohortBoundaryState {
                                    pending_sources: cohort.pending_sources,
                                    first_source_response: cohort.first_source_response,
                                    last_source_response: cohort.last_source_response,
                                    payload_work: cohort.payload_work,
                                    destination_lines: cohort.destination_lines,
                                    bank_wavefronts: cohort.bank_wavefronts,
                                    bypass_wavefronts: cohort.bypass_wavefronts,
                                    topology_penalty: cohort.topology_penalty,
                                })
                                .collect(),
                        })
                        .collect::<Vec<_>>();
                    bypass_observations
                        .sort_by_key(|observation| (observation.group, observation.instruction));
                    // Aggregate-only fallback captures preserve useful
                    // diagnostics but do not retain the cohort/membership
                    // topology required by an exact transition contract.
                    let bypass_observations_complete =
                        bypass_observations.iter().all(|observation| {
                            observation.source_cohorts != 0
                                && !observation.sector_membership_masks.is_empty()
                                && !observation.cohorts.is_empty()
                        });
                    AsyncCopyBoundaryState {
                        open: state.open_group.map(project),
                        committed: state
                            .committed_groups
                            .iter()
                            .copied()
                            .map(project)
                            .collect(),
                        bypass_observation_count: u64::try_from(state.bypass_copies.len())
                            .unwrap_or(u64::MAX),
                        next_sequence: state.next_sequence,
                        bypass_observations_complete,
                        bypass_observations,
                    }
                })
            })
            .collect()
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_memory_pipeline_state(&self) -> Option<MemoryPipelineBoundaryState> {
        self.memory_issue_pipeline.as_ref().map(|state| {
            let diagnostics = state.diagnostics;
            MemoryPipelineBoundaryState {
                queue_entries_with_dispatch: state.config.queue_entries_with_dispatch,
                local_initiation_interval_cycles: state.config.local_initiation_interval_cycles,
                shared_accept_interval_cycles: state.config.shared_accept_interval_cycles,
                queue_release_times: state
                    .queue_release_times
                    .iter()
                    .map(|queue| queue.iter().copied().collect())
                    .collect(),
                local_completion_ready: state.local_completion_ready.to_vec(),
                shared_accept_ready: state.shared_accept_ready,
                instructions_admitted: diagnostics.instructions_admitted,
                operand_pipeline_delay_cycles: diagnostics.operand_pipeline_delay_cycles,
                shared_queue_delay_cycles: diagnostics.shared_queue_delay_cycles,
                maximum_operand_pipeline_delay_cycles: diagnostics
                    .maximum_operand_pipeline_delay_cycles,
                maximum_shared_queue_delay_cycles: diagnostics.maximum_shared_queue_delay_cycles,
                maximum_queue_occupancy: diagnostics.maximum_queue_occupancy,
                terminal_shared_accept_cycle: diagnostics.terminal_shared_accept_cycle,
            }
        })
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_scalar_pipeline_state(&self) -> Option<ScalarPipelineBoundaryState> {
        self.scalar_execution_pipeline.as_ref().map(|state| {
            let diagnostics = state.diagnostics;
            ScalarPipelineBoundaryState {
                latency_cycles: state.config.latency_cycles,
                initiation_interval_cycles: state.config.initiation_interval_cycles,
                in_flight_capacity: state.config.in_flight_capacity,
                release_times: state
                    .release_times
                    .iter()
                    .map(|queue| queue.iter().copied().collect())
                    .collect(),
                next_service: state.next_service.to_vec(),
                instructions_admitted: diagnostics.instructions_admitted,
                service_busy_cycles: diagnostics.service_busy_cycles,
                queue_delay_cycles: diagnostics.queue_delay_cycles,
                maximum_queue_occupancy: diagnostics.maximum_queue_occupancy,
            }
        })
    }

    #[cfg(feature = "reference")]
    pub(crate) fn typed_front_end_state(&self) -> Option<InstructionFrontEndBoundaryState> {
        let state = self.instruction_front_end.as_ref()?;
        let cache = |cache: &SetAssociativeInstructionCache| InstructionCacheState {
            ways: u64::try_from(cache.ways).unwrap_or(u64::MAX),
            access_sequence: cache.access_sequence,
            sets: cache
                .sets
                .iter()
                .map(|set| {
                    set.iter()
                        .map(|entry| InstructionCacheEntryState {
                            function_unique_id: entry.key.function_unique_id,
                            line: entry.key.line,
                            ready_at: entry.ready_at,
                            last_access: entry.last_access,
                        })
                        .collect()
                })
                .collect(),
        };
        let streams = state
            .streams
            .iter()
            .map(|stream| InstructionStreamState {
                lines: stream
                    .lines
                    .iter()
                    .map(|line| InstructionStreamLineState {
                        function_unique_id: line.key.function_unique_id,
                        line: line.key.line,
                        ready_at: line.ready_at,
                        demanded: line.demanded,
                    })
                    .collect(),
                next_line: stream
                    .next_line
                    .map(|key| (key.function_unique_id, key.line)),
            })
            .collect();
        let diagnostics = state.diagnostics;
        Some(InstructionFrontEndBoundaryState {
            line_bytes: state.config.line_bytes,
            l0_to_l1_cycles: state.config.l0_to_l1_cycles,
            l1_to_l0_cycles: state.config.l1_to_l0_cycles,
            l1_request_ports: state.config.l1_request_ports,
            l1_reply_ports: state.config.l1_reply_ports,
            stream_buffer_lines: state.config.stream_buffer_lines,
            stream_buffers_per_subcore: state.config.stream_buffers_per_subcore,
            prefetches_per_cycle: state.config.prefetches_per_cycle,
            instruction_buffer_entries: state.config.instruction_buffer_entries,
            fetch_decode_width: state.config.fetch_decode_width,
            fetch_to_issue_stages: state.config.fetch_to_issue_stages,
            l0: state.l0.iter().map(cache).collect(),
            l1: cache(&state.l1),
            streams,
            l1_request_ready: state.l1_request_ready,
            l1_reply_ready: state.l1_reply_ready,
            l0_response_ready: state.l0_response_ready.to_vec(),
            prefetch_priority: u64::try_from(state.prefetch_priority).unwrap_or(u64::MAX),
            cold_fill_latency: state.cold_fill_latency,
            instructions_fetched: diagnostics.instructions_fetched,
            demand_l0_hits: diagnostics.demand_l0_hits,
            demand_stream_buffer_hits: diagnostics.demand_stream_buffer_hits,
            demand_l0_misses: diagnostics.demand_l0_misses,
            demand_l1_hits: diagnostics.demand_l1_hits,
            demand_l1_misses: diagnostics.demand_l1_misses,
            prefetch_requests: diagnostics.prefetch_requests,
            prefetch_l1_hits: diagnostics.prefetch_l1_hits,
            prefetch_l1_misses: diagnostics.prefetch_l1_misses,
            stream_restarts: diagnostics.stream_restarts,
            maximum_fetch_latency_cycles: diagnostics.maximum_fetch_latency_cycles,
        })
    }

    #[cfg(feature = "reference")]
    pub(crate) fn boundary_snapshot(&self) -> BoundarySnapshot {
        let warp_states = self
            .warps
            .iter()
            .map(|warp| WarpBoundaryState {
                warp_slot: warp.warp_slot,
                next_instruction: u64::try_from(warp.next).unwrap_or(u64::MAX),
                front_end_fetch_cursor: u64::try_from(warp.front_end_fetch_cursor)
                    .unwrap_or(u64::MAX),
                instruction_buffer: warp
                    .instruction_buffer
                    .iter()
                    .map(|buffered| {
                        (
                            u64::try_from(buffered.instruction).unwrap_or(u64::MAX),
                            buffered.ready_at,
                        )
                    })
                    .collect(),
                eligible: warp.eligible,
                imc_wait_until: warp.imc_wait_until,
                barrier_clear: warp.barrier_clear,
                barrier_pending: warp.barrier_pending,
                block: u64::try_from(warp.block).unwrap_or(u64::MAX),
                arrival: warp.arrival,
                at_block_barrier: warp.at_block_barrier,
                cta_barrier_ready: warp.cta_barrier_ready,
                barrier_pipeline_ready: warp.barrier_pipeline_ready,
                branch_pipeline_ready: warp.branch_pipeline_ready,
                block_barrier_dispatch: warp.block_barrier_dispatch,
                block_barrier_arrival: warp.block_barrier_arrival,
                done: warp.done,
            })
            .collect();
        let block_states = self
            .block_states
            .iter()
            .map(|block| BlockBoundaryState {
                unfinished: u64::try_from(block.unfinished).unwrap_or(u64::MAX),
                waiting: u64::try_from(block.waiting).unwrap_or(u64::MAX),
                warp_count: block.warp_count,
                shared_memory_bytes: block.shared_memory_bytes,
                warp_slots: block.warp_slots.to_vec(),
                first_warp: u64::try_from(block.first_warp).unwrap_or(u64::MAX),
                admitted_warps: u64::try_from(block.admitted_warps).unwrap_or(u64::MAX),
                barrier_generation: block.barrier_generation,
                barrier_first_dispatch: block.barrier_first_dispatch,
                barrier_first_arrival: block.barrier_first_arrival,
                barrier_rearm_ready: block.barrier_rearm_ready,
            })
            .collect();
        let hold = self.hold.map(|entry| {
            entry.and_then(|warp| {
                self.warps
                    .get(warp)
                    .map(|state| u64::try_from(state.warp_slot).unwrap_or(0))
            })
        });
        let subcore_warps = self
            .subcore_warps
            .iter()
            .map(|warps| {
                warps
                    .iter()
                    .filter_map(|&warp| self.warps.get(warp).map(|state| state.warp_slot))
                    .collect()
            })
            .collect();
        BoundarySnapshot {
            schema_version: sim_semantics::BOUNDARY_SCHEMA_VERSION,
            cycle: self.t,
            issued: self.issued,
            resident_warps: self.resident_warps,
            resident_shared_memory_bytes: self.resident_shared_memory_bytes,
            unfinished_warps: u64::try_from(self.unfinished_warps).unwrap_or(u64::MAX),
            next_block: u64::try_from(self.next_block).unwrap_or(u64::MAX),
            hold,
            subcore_warps,
            outstanding_responses: u64::try_from(self.outstanding_responses()).unwrap_or(u64::MAX),
            l1_state: match &self.mem {
                MemModel::Flat => None,
                MemModel::L1Const(cache) => Some(cache.boundary_state()),
                MemModel::L1Shared(state) => Some(state.l1.boundary_state()),
            },
            instruction_front_end_state: self
                .instruction_front_end
                .as_ref()
                .map(|state| format!("{state:?}")),
            memory_issue_pipeline_state: self
                .memory_issue_pipeline
                .as_ref()
                .map(|state| format!("{state:?}")),
            scalar_execution_pipeline_state: self
                .scalar_execution_pipeline
                .as_ref()
                .map(|state| format!("{state:?}")),
            async_copy_state: self
                .async_copies
                .iter()
                .map(|slot| slot.state.as_deref().map(|state| format!("{state:?}")))
                .collect(),
            barrier_pipe_ready: self.barrier_pipe_ready,
            tensor_pipe_ready: self.tensor_pipe_ready,
            ldgsts_destination_pipe_ready: self.ldgsts_destination_pipe_ready,
            warp_states,
            block_states,
            omitted_state: vec!["legacy_core_debug_fields_retained".to_owned()],
        }
    }

    pub(crate) fn enable_memory_admission_observations(&mut self) {
        self.emit_admission_witnesses = true;
    }

    pub(crate) fn install_memory_admission_feedback(
        &mut self,
        candidate: MemoryAdmissionCandidate,
    ) -> Result<(), IssueError> {
        candidate.validate_feedback(self.params.read_barrier_latency)?;
        self.memory_admission_feedback =
            Some(Box::new(MemoryAdmissionFeedbackState::checked(candidate)?));
        self.emit_admission_witnesses = true;
        Ok(())
    }

    pub(crate) fn complete_memory_admission_feedback(
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
        self.memory_admission_feedback
            .as_mut()
            .ok_or(IssueError::MemoryAdmissionFeedbackReservationMissing {
                warp: witness.warp(),
                instruction: witness.instruction(),
            })?
            .complete(
                witness,
                operation,
                hit_sectors,
                miss_sectors,
                l2_requests,
                unique_regions,
                issue_time,
                now,
            )
    }

    pub(crate) fn complete_memory_write_ack(
        &mut self,
        ack: MemoryWriteAck,
        now: u64,
    ) -> Result<(), IssueError> {
        self.memory_admission_feedback
            .as_mut()
            .ok_or(IssueError::MemoryWriteAckNotPending {
                warp: ack.witness.warp(),
                instruction: ack.witness.instruction(),
            })?
            .acknowledge_write(ack, now)
    }

    pub(crate) fn pending_memory_write_acks(&self) -> usize {
        self.memory_admission_feedback
            .as_deref()
            .map_or(0, MemoryAdmissionFeedbackState::pending_write_ack_count)
    }

    pub(crate) fn is_done(&self) -> bool {
        self.unfinished_warps == 0
    }

    /// Responses the system still owes this SM (diagnostics).
    pub(crate) fn outstanding_responses(&self) -> usize {
        match &self.mem {
            MemModel::L1Shared(st) => st.outstanding_replies.len(),
            _ => 0,
        }
    }

    /// Instructions issued so far (monotonic; the system driver's
    /// progress signal).
    pub(crate) fn instructions_issued(&self) -> u64 {
        self.issued
    }

    /// Result snapshot; meaningful once [`Self::advance`] returned
    /// [`Advance::Done`].
    pub(crate) fn result(&self) -> SmSimResult {
        let mut readiness = (*self.readiness).clone();
        // The released Huerta scheduler orders a greedy warp followed by
        // descending dynamic warp IDs.  The replay admission path currently
        // stamps warps in the validated block-roster order; retain an explicit
        // audit of any within-CTA inversion instead of silently assuming the
        // two orders are equivalent. This diagnostic never selects timing.
        let mut priority_mismatches = 0_u64;
        for left in 0..self.warps.len() {
            for right in (left + 1)..self.warps.len() {
                let a = &self.warps[left];
                let b = &self.warps[right];
                if a.block != b.block
                    || a.warp_slot.rem_euclid(SUB_CORES as i32)
                        != b.warp_slot.rem_euclid(SUB_CORES as i32)
                {
                    continue;
                }
                let arrival_order = a.arrival.cmp(&b.arrival);
                let dynamic_order = a.warp_id.cmp(&b.warp_id);
                if arrival_order != dynamic_order {
                    priority_mismatches = priority_mismatches.saturating_add(1);
                }
            }
        }
        readiness.youngest_priority_order_mismatches = priority_mismatches;
        if let Some(front_end) = self.instruction_front_end.as_ref() {
            readiness.instruction_front_end = Some(Box::new(front_end.diagnostics));
        }
        if let Some(memory_pipeline) = self.memory_issue_pipeline.as_ref() {
            readiness.memory_issue_pipeline = Some(Box::new(memory_pipeline.diagnostics));
        }
        if let Some(cache) = self.immediate_constant_cache.as_ref() {
            let mut diagnostics = ImmediateConstantCacheDiagnostics::default();
            for state in cache.iter() {
                let addend = state.diagnostics();
                diagnostics.references = diagnostics.references.saturating_add(addend.references);
                diagnostics.lookup_hits =
                    diagnostics.lookup_hits.saturating_add(addend.lookup_hits);
                diagnostics.lookup_misses = diagnostics
                    .lookup_misses
                    .saturating_add(addend.lookup_misses);
                diagnostics.pending_tag_hits = diagnostics
                    .pending_tag_hits
                    .saturating_add(addend.pending_tag_hits);
                diagnostics.tag_misses = diagnostics.tag_misses.saturating_add(addend.tag_misses);
                diagnostics.pending_merges = diagnostics
                    .pending_merges
                    .saturating_add(addend.pending_merges);
                diagnostics.lines_filled =
                    diagnostics.lines_filled.saturating_add(addend.lines_filled);
                diagnostics.miss_wait_events = diagnostics
                    .miss_wait_events
                    .saturating_add(addend.miss_wait_events);
                diagnostics.miss_wait_cycles = diagnostics
                    .miss_wait_cycles
                    .saturating_add(addend.miss_wait_cycles);
                diagnostics.greedy_switch_block_cycles = diagnostics
                    .greedy_switch_block_cycles
                    .saturating_add(addend.greedy_switch_block_cycles);
                diagnostics.non_greedy_scan_attempts = diagnostics
                    .non_greedy_scan_attempts
                    .saturating_add(addend.non_greedy_scan_attempts);
                diagnostics.non_greedy_scan_misses = diagnostics
                    .non_greedy_scan_misses
                    .saturating_add(addend.non_greedy_scan_misses);
                diagnostics.non_greedy_scan_successes = diagnostics
                    .non_greedy_scan_successes
                    .saturating_add(addend.non_greedy_scan_successes);
                diagnostics.maximum_pending_lines = diagnostics
                    .maximum_pending_lines
                    .max(addend.maximum_pending_lines);
            }
            if !diagnostics.is_empty() {
                readiness.immediate_constant_cache = Some(Box::new(diagnostics));
            }
        }
        readiness.memory_service_candidates = self.memory_service_candidates.diagnostics();
        if let Some(feedback) = self.memory_admission_feedback.as_ref() {
            readiness.memory_admission_feedback = Some(Box::new(feedback.diagnostics()));
            let write_ack = feedback.write_ack_diagnostics();
            if !write_ack.is_empty() {
                readiness.memory_write_ack = Some(Box::new(write_ack));
            }
        }
        let pipeline_completion = self
            .memory_issue_pipeline
            .as_ref()
            .filter(|pipeline| !pipeline.diagnostics.is_empty())
            .map(|pipeline| pipeline.diagnostics.terminal_shared_accept_cycle);
        let modeled_local_completion_cycles =
            match (pipeline_completion, self.local_completion_time) {
                (Some(pipeline), Some(local)) => Some(pipeline.max(local)),
                (Some(value), None) | (None, Some(value)) => Some(value),
                (None, None) => None,
            };
        let activity = SmActivity::from_lifetimes(
            self.warp_lifetimes.clone(),
            self.instructions_per_smsp,
            self.issue_active_cycles_per_smsp,
            self.params.max_resident_warps_per_sm,
        );
        SmSimResult {
            instructions_issued: self.issued,
            cycles: self.t,
            modeled_local_completion_cycles,
            l1: self.mem.l1().map(|c| c.counters),
            l1_merge_audit: self.mem.l1().map(L1Cache::merge_audit),
            unmodeled_tails: self.unmodeled_tails,
            readiness,
            activity,
        }
    }

    /// Retire one warp after its final instruction has become semantically
    /// complete. Ordinary instructions call this immediately; a final
    /// BAR.SYNC calls it from the CTA-release path so the barrier remains a
    /// real synchronization point.
    fn retire_warp(&mut self, subcore: usize, warp_index: usize, t: u64) -> Result<(), IssueError> {
        if self.warps[warp_index].done {
            return Ok(());
        }
        let block = self.warps[warp_index].block;
        let admission_time = self.warps[warp_index].admission_time;
        let warp_slot = self.warps[warp_index].warp_slot;
        let pos = self.warps[warp_index].pos_in_subcore;
        if self.subcore_warps[subcore].get(pos).copied() != Some(warp_index) {
            return Err(IssueError::CounterUnderflow {
                counter: "sub-core resident warp",
            });
        }
        self.warp_lifetimes.push(WarpLifetime {
            block: self.warps[warp_index].launch_index,
            warp_slot,
            admission: admission_time,
            retirement: checked_cycle_add("warp activity end", t, 1)?,
        });
        // A final BAR.SYNC is parked with itself in the CGGTY hold slot while
        // it waits for the CTA rendezvous.  Clear that slot before removing
        // the warp: swap_remove can otherwise leave a stale index that now
        // names a different resident warp (or a done warp), changing the
        // next pick/fetch decision after release.  Ordinary retirement also
        // benefits from the same invariant.
        if self.hold[subcore] == Some(warp_index) {
            self.hold[subcore] = None;
        }
        if self.fetch_last_issued[subcore] == Some(warp_index) {
            self.fetch_last_issued[subcore] = None;
        }
        self.warps[warp_index].done = true;
        self.warps[warp_index].next_slot = None;
        self.warps[warp_index].ordinary_dependency_wait_mask = WaitMask::NONE;
        self.warps[warp_index].at_block_barrier = false;
        self.warps[warp_index].block_barrier_dispatch = None;
        self.warps[warp_index].block_barrier_arrival = None;
        self.unfinished_warps =
            self.unfinished_warps
                .checked_sub(1)
                .ok_or(IssueError::CounterUnderflow {
                    counter: "unfinished warp",
                })?;
        self.block_states[block].unfinished = self.block_states[block]
            .unfinished
            .checked_sub(1)
            .ok_or(IssueError::CounterUnderflow {
                counter: "block unfinished warp",
            })?;

        // Retire from the sub-core pick list (swap-remove + position fix-up)
        // so scans remain bounded by resident warps.
        self.subcore_warps[subcore].swap_remove(pos);
        if let Some(&moved) = self.subcore_warps[subcore].get(pos) {
            self.warps[moved].pos_in_subcore = pos;
        }

        if self.block_states[block].unfinished == 0 {
            let state = &self.block_states[block];
            self.resident_warps = self.resident_warps.checked_sub(state.warp_count).ok_or(
                IssueError::CounterUnderflow {
                    counter: "resident warp",
                },
            )?;
            self.resident_shared_memory_bytes = self
                .resident_shared_memory_bytes
                .checked_sub(state.shared_memory_bytes)
                .ok_or(IssueError::CounterUnderflow {
                    counter: "resident shared-memory byte",
                })?;
            for &warp_slot in &state.warp_slots {
                let removed = self.resident_warp_slots.remove(&warp_slot);
                if !removed {
                    return Err(IssueError::ResidentWarpSlotMissing {
                        launch_index: self.warps[state.first_warp].launch_index,
                        warp_slot,
                    });
                }
            }
            let admit_at = checked_cycle_add("block-wave admission", t, 1)?;
            self.admit_blocks(admit_at)?;
        }
        Ok(())
    }
}


