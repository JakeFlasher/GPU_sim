// Lexical implementation section 3/4; included by the parent facade.
/// Decode one launch into per-SM block replays on the traced placement.
/// `decode_sectors` selects the L1-mode classification (lane decode +
/// unique-sector expansion) over the flat-mode class check.
fn extract_per_sm_blocks(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    placement: PlacementConfig,
    decode_sectors: bool,
    tma_shadow_l2: Option<L2Config>,
) -> Result<ExtractedKernel, ReplayError> {
    let expected_fuid = schedule.function_unique_id;
    let shared_memory_residency = placement.shared_memory_residency(loc.kernel())?;
    let block = loc
        .kernel()
        .block_dim
        .as_ref()
        .filter(|dim| dim.x > 0 && dim.y > 0 && dim.z > 0)
        .map(|dim| (dim.x as u32, dim.y as u32, dim.z as u32));
    let expected_block_warps = block.map(checked_block_warps).transpose()?;
    #[cfg(feature = "campaign")]
    let validate_warp_roster = expected_block_warps.is_some()
        && td
            .sampling_sidecar()
            .is_none_or(|sidecar| Some(sidecar.original_block()) == block);
    #[cfg(not(feature = "campaign"))]
    let validate_warp_roster = expected_block_warps.is_some();
    // Physical slot validation is part of the traced launch envelope.  Do
    // not obtain it from `issue_hypotheses()`: SM120 intentionally has no
    // admitted scheduler hypothesis, while the 5090 runtime card does carry
    // a measured max-resident-warp limit.
    let max_resident_warps = placement.max_resident_warps_per_sm;
    let cluster = normalized_cluster_dim(loc.kernel(), schedule.arch)?;
    // sm_id → blocks (launch index kept for in-SM ordering).
    let mut per_sm_blocks: BTreeMap<i32, Vec<BlockReplay>> = BTreeMap::new();
    let mut seen_indices: BTreeSet<i64> = BTreeSet::new();
    let mut tma_dynamic_keys_seen: BTreeSet<TmaObservationKey> = BTreeSet::new();
    let collect_joint_l2 = tma_shadow_l2.is_some();
    let mut joint_l2_events = Vec::new();
    let mut joint_l2_order_unavailable = 0_u64;
    let mut joint_l2_events_dropped = 0_u64;
    let mut joint_l2_sequence = 0_u64;
    let mut traced_sm_by_launch_index: BTreeMap<i64, i32> = BTreeMap::new();
    let mut records_seen: u64 = 0;
    let mut instruction_sectors = BTreeSet::new();
    let mut constant_sectors = BTreeSet::new();
    let mut instruction_lines = BTreeSet::new();
    let static_instruction_lines = schedule
        .by_pc
        .keys()
        .map(|pc| pc / 128)
        .collect::<BTreeSet<_>>();

    let mut instruction_sectors_by_sm: BTreeMap<i32, BTreeSet<(i32, u32)>> = BTreeMap::new();
    let launch_resources = (loc.kernel().number_of_registers > 0
        || loc.kernel().size_shared_memory > 0
        || loc.kernel().shared_memory_static_bytes.is_some()
        || loc.kernel().shared_memory_dynamic_bytes.is_some()
        || loc.kernel().shared_memory_carveout_preference_pct.is_some()
        || loc.kernel().driver_occupancy_limit_blocks.is_some())
    .then(|| crate::metrics::LaunchResources {
        block_threads: loc
            .kernel()
            .block_dim
            .as_ref()
            .map(|dim| {
                dim.x
                    .max(0)
                    .saturating_mul(dim.y.max(0))
                    .saturating_mul(dim.z.max(0)) as u32
            })
            .unwrap_or(0),
        grid_blocks: loc
            .kernel()
            .grid_dim
            .as_ref()
            .map(|dim| {
                u64::from(dim.x.max(0) as u32)
                    .saturating_mul(u64::from(dim.y.max(0) as u32))
                    .saturating_mul(u64::from(dim.z.max(0) as u32))
            })
            .unwrap_or(0),
        // TraceDir validated nonnegative counts and the split/total sum.
        registers_per_thread: loc.kernel().number_of_registers as u32,
        shared_memory_bytes_per_block: loc.kernel().size_shared_memory as u32,
        static_shared_memory_bytes: loc.kernel().shared_memory_static_bytes,
        dynamic_shared_memory_bytes: loc.kernel().shared_memory_dynamic_bytes,
        shared_memory_carveout_preference_pct: loc.kernel().shared_memory_carveout_preference_pct,
        driver_occupancy_limit_blocks: loc.kernel().driver_occupancy_limit_blocks,
    });
    let mut trace_metrics = TraceMetrics {
        launch_resources,
        // SM120 flat mixed-async launches use the compact classifier but also
        // retain every observed global/local source span. Opaque TMA payload
        // bytes are reported separately and do not manufacture sector ids.
        sectors_decoded: decode_sectors || schedule.arch == ArchId::Sm120,
        ..TraceMetrics::default()
    };
    trace_metrics.bind_conditional_locality_arch(schedule.arch);
    let sidecar_join_started = Instant::now();
    let observations = if schedule.arch == ArchId::Sm120 {
        td.async_sidecars()?
            .launch(loc.device_id(), loc.stream_id(), loc.kernel().id)
    } else {
        None
    };
    trace_metrics.async_relations = observations
        .filter(|o| !o.tma.is_empty() || !o.sync.is_empty())
        .map(|o| o.relations(cluster.into()));
    trace_metrics.async_obligations = observations
        .filter(|o| !o.tma.is_empty() || !o.sync.is_empty())
        .map(|o| sim_trace::obligations::ObligationFrontier::from_launch(o, cluster.into()).0);
    if let Some(observations) = observations {
        let (
            sum_record_unique_sectors,
            sum_record_unique_lines,
            launch_unique_sectors,
            launch_unique_lines,
            intra_record_duplicate_refs,
            load_sum_sectors,
            store_sum_sectors,
            load_sum_lines,
            store_sum_lines,
            load_unique_sectors,
            store_unique_sectors,
            cross_direction_sectors,
            cross_direction_lines,
        ) = tma_sidecar_reuse_summary(observations);
        trace_metrics.observe_tma_sidecar_reuse(
            sum_record_unique_sectors,
            sum_record_unique_lines,
            launch_unique_sectors,
            launch_unique_lines,
            intra_record_duplicate_refs,
        )?;
        trace_metrics.observe_tma_directional_reuse(
            load_sum_sectors,
            store_sum_sectors,
            load_sum_lines,
            store_sum_lines,
            load_unique_sectors,
            store_unique_sectors,
            cross_direction_sectors,
            cross_direction_lines,
        )?;
        if let Some(relations) = trace_metrics.async_relations.as_ref() {
            let (pairs, invalid, lower_sum, upper_sum) =
                async_clock_sandwich_summary(observations, relations);
            trace_metrics.observe_async_clock_sandwich(pairs, invalid, lower_sum, upper_sum)?;
        }
    }
    if let (Some(observations), Some(l2)) = (observations, tma_shadow_l2) {
        let stats = observe_tma_l2_shadow(&observations.tma, l2)?;
        trace_metrics.observe_tma_shadow_l2(
            stats.requests,
            stats.hits,
            stats.misses,
            stats.unavailable,
            stats.load,
            stats.store,
        )?;
        let ordered = observe_tma_l2_shadow_ordered(observations, l2)?;
        trace_metrics.observe_tma_shadow_l2_ordered(
            ordered.requests,
            ordered.hits,
            ordered.misses,
            ordered.unavailable,
            ordered.load,
            ordered.store,
        )?;
    }
    let tma_observations = observations.map(|o| &o.tma);
    let sync_observations = observations.map(|o| &o.sync);
    trace_metrics.sidecar_join_wall_ns =
        u64::try_from(sidecar_join_started.elapsed().as_nanos()).unwrap_or(u64::MAX);

    for path in td.threadblock_paths(loc)? {
        let tb = td.validated_threadblock(&path)?;
        let launch_index = tb.launch_index();
        if !seen_indices.insert(launch_index) {
            return Err(ReplayError::DuplicateLaunchIndex {
                index: launch_index,
            });
        }

        let admitted_block_warps = expected_block_warps.filter(|_| validate_warp_roster);
        if let Some(expected_block_warps) = admitted_block_warps {
            let actual_block_warps =
                checked_collection_len("threadblock warp roster", tb.warps().len())?;
            if actual_block_warps != expected_block_warps {
                return Err(ReplayError::WarpRosterCardinalityMismatch {
                    path,
                    expected: expected_block_warps,
                    actual: actual_block_warps,
                });
            }
        }
        let mut block_sm: Option<i32> = None;
        let block_id = tb.block_id();
        let mut seen_warp_slots = BTreeSet::new();
        let mut warps = Vec::with_capacity(tb.warps().len());
        for warp in tb.warps() {
            let warp_id = warp.map_key();
            if admitted_block_warps.is_some_and(|expected_block_warps| {
                warp_id < 0 || u64::try_from(warp_id).unwrap_or(u64::MAX) >= expected_block_warps
            }) {
                return Err(ReplayError::WarpIdOutOfRange {
                    path,
                    warp_id,
                    expected_warps: admitted_block_warps.expect("checked as present"),
                });
            }
            if warp.warp_id() != warp_id {
                return Err(ReplayError::WarpIdentityMismatch {
                    path,
                    map_key: warp_id,
                    warp_id: warp.warp_id(),
                });
            }
            let mut warp_slot: Option<i32> = None;
            let mut insts = Vec::with_capacity(warp.instructions().len());
            let mut sectors: Vec<i64> = Vec::new();
            let mut async_copy_cohorts = AsyncCopyCohortReplay::default();
            let mut branch_successors = crate::metrics::BranchSuccessors::default();
            let mut previous_control_flow: Option<(i32, u32, u32)> = None;
            trace_metrics.begin_warp_memory_stream();
            for (record_ix, inst) in warp.instructions().enumerate() {
                let record_warp_id = inst.warpid_tb();
                if record_warp_id != warp_id {
                    return Err(ReplayError::InstructionWarpIdMismatch {
                        path,
                        warp_id,
                        record_ix,
                        record_warp_id,
                    });
                }
                records_seen = checked_record_count(records_seen, &path, warp_id, record_ix)?;
                let constant_addresses = inst.constant_memory_addresses();
                // Only active-lane address vectors contribute to the demand
                // interval. Bank/offset metadata without addresses can be a
                // predicated-off or incomplete record and is retained as
                // typed metadata, never fabricated into traffic.
                let constant_bank = inst.constant_memory_bank_offset().map(|(bank, _)| bank);
                trace_metrics.observe_constant_memory(constant_addresses, constant_bank)?;
                if constant_addresses.is_empty() && inst.constant_memory_metadata_present() {
                    trace_metrics.observe_constant_metadata_only()?;
                }
                let preflight = preflight_compact_memory(
                    inst.memory(),
                    warp,
                    schedule,
                    inst.pc(),
                    &path,
                    warp_id,
                    record_ix,
                )?;
                if inst.function_unique_id() != expected_fuid {
                    return Err(ReplayError::MultiFunctionUnsupported {
                        path,
                        warp_id,
                        record_ix,
                        expected: expected_fuid,
                        found: inst.function_unique_id(),
                    });
                }
                instruction_sectors.insert((inst.function_unique_id(), inst.pc() / 32));
                if let Some(slot) = schedule.issue_slot(u64::from(inst.pc())) {
                    constant_sectors.extend(static_constant_sectors(slot.opcode_text()));
                }
                instruction_lines.insert((inst.function_unique_id(), inst.pc() / 128));

                let is_tma_opcode = schedule.arch == ArchId::Sm120
                    && schedule
                        .issue_slot(u64::from(inst.pc()))
                        .is_some_and(|slot| {
                            matches!(
                                opcode_head(slot.opcode_text()),
                                "UTMALDG" | "UTMASTG" | "UBLKCP"
                            )
                        });
                if is_tma_opcode {
                    tma_dynamic_keys_seen.insert(TmaObservationKey {
                        cta: block_id,
                        warp: warp_id,
                        instruction: record_ix,
                        pc: inst.pc(),
                        function: inst.function_unique_id(),
                    });
                }
                if is_tma_opcode && (decode_sectors || preflight.is_none()) {
                    let key = TmaObservationKey {
                        cta: block_id,
                        warp: warp_id,
                        instruction: record_ix,
                        pc: inst.pc(),
                        function: inst.function_unique_id(),
                    };
                    trace_metrics
                        .observe_tma_sidecar_payload(tma_observations.and_then(|o| o.get(&key)))?;
                }
                if schedule.arch == ArchId::Sm120
                    && schedule
                        .issue_slot(u64::from(inst.pc()))
                        .is_some_and(|slot| is_sync_sidecar_opcode(slot.opcode_text()))
                {
                    let key = TmaObservationKey {
                        cta: block_id,
                        warp: warp_id,
                        instruction: record_ix,
                        pc: inst.pc(),
                        function: inst.function_unique_id(),
                    };
                    trace_metrics
                        .observe_sync_sidecar(sync_observations.and_then(|o| o.get(&key)))?;
                }
                if let Some(sm) = inst.sm_id() {
                    match block_sm {
                        None => block_sm = Some(sm),
                        Some(first) if first != sm => {
                            return Err(ReplayError::SmIdInconsistent {
                                path,
                                first,
                                second: sm,
                            });
                        }
                        Some(_) => {}
                    }
                }
                if let Some(slot) = inst.warpid_sm() {
                    match warp_slot {
                        None => warp_slot = Some(slot),
                        Some(first) if first != slot => {
                            return Err(ReplayError::WarpSlotInconsistent {
                                path,
                                warp_id,
                                first,
                                second: slot,
                            });
                        }
                        Some(_) => {}
                    }
                }
                let partial_store_sectors = preflight
                    .as_ref()
                    .map_or(0, |memory| warp.partial_store_sectors(memory.memory));
                let local_interleaved_sectors = preflight.as_ref().map_or(&[][..], |memory| {
                    warp.local_interleaved_sector_plan(memory.memory)
                });
                let local_interleaved_dynamic = preflight
                    .as_ref()
                    .is_some_and(|memory| memory.memory.local_interleaved_is_dynamic());
                let kind = if decode_sectors {
                    classify_compact_with_l1(
                        preflight,
                        warp,
                        schedule,
                        schedule.arch,
                        &mut sectors,
                        &mut async_copy_cohorts,
                        block_id,
                        tma_observations,
                        &path,
                        warp_id,
                        record_ix,
                    )?
                } else {
                    classify_compact_flat(
                        preflight,
                        warp,
                        schedule,
                        schedule.arch,
                        inst.pc(),
                        &path,
                        warp_id,
                        record_ix,
                        &mut sectors,
                        block_id,
                        tma_observations,
                    )?
                };
                if matches!(
                    kind,
                    AccessKind::GlobalStore { .. } | AccessKind::LocalStore { .. }
                ) {
                    trace_metrics.partial_store_sectors = trace_metrics
                        .partial_store_sectors
                        .checked_add(partial_store_sectors)
                        .ok_or(IssueError::CounterOverflow {
                            counter: "partial store sectors",
                        })?;
                }
                if collect_joint_l2 {
                    let order = inst.instrumentation_order();
                    if let Some((off, len, class, ordinary_read)) = joint_l2_ordinary_span(kind) {
                        if len != 0 {
                            let end = off.saturating_add(len);
                            if let Some(span) = sectors.get(off..end) {
                                if let Some(order) = order {
                                    for &sector_id in span {
                                        if joint_l2_events.len() >= MAX_JOINT_L2_SHADOW_EVENTS {
                                            joint_l2_events_dropped =
                                                joint_l2_events_dropped.saturating_add(1);
                                            break;
                                        }
                                        joint_l2_events.push(JointL2ShadowEvent {
                                            order,
                                            sequence: joint_l2_sequence,
                                            sector_id,
                                            class,
                                            tma_load: false,
                                            ordinary_read,
                                        });
                                        joint_l2_sequence = joint_l2_sequence.saturating_add(1);
                                    }
                                } else {
                                    joint_l2_order_unavailable =
                                        joint_l2_order_unavailable.saturating_add(1);
                                }
                            }
                        }
                    }
                    if is_tma_opcode {
                        let key = TmaObservationKey {
                            cta: block_id,
                            warp: warp_id,
                            instruction: record_ix,
                            pc: inst.pc(),
                            function: inst.function_unique_id(),
                        };
                        if let Some(observation) =
                            tma_observations.and_then(|records| records.get(&key))
                        {
                            let ids = if observation.is_load {
                                &observation.source_sector_ids[..]
                            } else if observation.is_store {
                                &observation.destination_sector_ids[..]
                            } else {
                                &[][..]
                            };
                            if ids.is_empty() {
                                joint_l2_order_unavailable =
                                    joint_l2_order_unavailable.saturating_add(1);
                            } else if let Some(order) = order {
                                let class = if observation.is_load {
                                    L2OpClass::Read
                                } else {
                                    L2OpClass::WriteAtomic
                                };
                                for &sector_id in ids.iter() {
                                    if joint_l2_events.len() >= MAX_JOINT_L2_SHADOW_EVENTS {
                                        joint_l2_events_dropped =
                                            joint_l2_events_dropped.saturating_add(1);
                                        break;
                                    }
                                    joint_l2_events.push(JointL2ShadowEvent {
                                        order,
                                        sequence: joint_l2_sequence,
                                        sector_id,
                                        class,
                                        tma_load: observation.is_load,
                                        ordinary_read: false,
                                    });
                                    joint_l2_sequence = joint_l2_sequence.saturating_add(1);
                                }
                            } else {
                                joint_l2_order_unavailable =
                                    joint_l2_order_unavailable.saturating_add(1);
                            }
                        }
                    }
                }
                if schedule.arch == ArchId::Sm86
                    && schedule
                        .issue_slot(u64::from(inst.pc()))
                        .is_some_and(|slot| is_static_constant_global_load(slot.opcode_text()))
                {
                    let span = match kind {
                        AccessKind::GlobalEligible { off, len, .. }
                        | AccessKind::GlobalBypass { off, len, .. }
                        | AccessKind::GlobalUntagged { off, len, .. } => Some((off, len)),
                        _ => None,
                    };
                    if let Some((off, len)) = span {
                        trace_metrics.observe_constant_global_load(
                            block_id,
                            warp_id,
                            &sectors,
                            off,
                            usize::from(len),
                            schedule
                                .issue_slot(u64::from(inst.pc()))
                                .map(|slot| slot.opcode_text())
                                .unwrap_or(""),
                            preflight
                                .as_ref()
                                .is_some_and(|memory| memory.memory.warp_uniform_address()),
                        )?;
                    }
                }
                if let Some((off, len)) = match kind {
                    AccessKind::GlobalBypass {
                        token: CacheOpToken::Atom,
                        off,
                        len,
                        ..
                    }
                    | AccessKind::GlobalEligible {
                        token: CacheOpToken::Atom,
                        off,
                        len,
                        ..
                    } => Some((off, len)),
                    _ => None,
                } {
                    trace_metrics.observe_atomic_conflict(&sectors, off, usize::from(len))?;
                }
                if matches!(
                    kind,
                    AccessKind::LocalLoad { .. } | AccessKind::LocalStore { .. }
                ) {
                    trace_metrics
                        .observe_local_width(inst.data_width(), inst.predicate_on_lanes())?;
                    trace_metrics.observe_local_payload_sectors(
                        kind,
                        inst.data_width(),
                        inst.predicate_on_lanes(),
                    )?;
                    if schedule.arch == ArchId::Sm86 {
                        if let Some(memory) = inst.memory() {
                            trace_metrics.observe_local_canonical_sectors(
                                kind,
                                memory.local_canonical_sectors(),
                            )?;
                            trace_metrics.observe_local_frame256_sectors(
                                kind,
                                memory.local_frame256_sectors(),
                            )?;
                        }
                    }
                }

                let active_mask_present = inst.active_mask_present();
                let predicate_mask_present = inst.predicate_mask_present();
                let (opcode, pipeline) = schedule
                    .issue_slot(u64::from(inst.pc()))
                    .map_or(("UNKNOWN", PipelineClass::DispatchOnly), |slot| {
                        (slot.opcode_text(), slot.pipeline_class())
                    });
                trace_metrics.observe_with_mask_presence_arch(
                    opcode,
                    inst.active_lanes(),
                    inst.predicate_on_lanes(),
                    active_mask_present,
                    predicate_mask_present,
                    kind,
                    pipeline,
                    &sectors,
                    schedule.arch,
                )?;
                // Passive reuse-interval evidence is collected from the same
                // decoded, coalesced span but never feeds cache state, DRAM,
                // or issue timing. The observer resets at each warp because
                // cross-warp arrival order is not present in the trace.
                trace_metrics.observe_reuse_memory(kind, &sectors)?;
                let opcode_head = opcode.split('.').next().unwrap_or(opcode);
                let narrow_store = opcode
                    .split('.')
                    .any(|modifier| matches!(modifier, "U8" | "S8" | "U16" | "S16"));
                let full_sector_store = matches!(opcode_head, "STG" | "STL" | "ST")
                    && !narrow_store
                    && active_mask_present
                    && predicate_mask_present
                    && inst.active_lanes() == 32
                    && inst.predicate_on_lanes() == 32
                    && inst.data_width().is_some_and(|width| width >= 4);
                trace_metrics.observe_store_to_load_forwarding(kind, &sectors, full_sector_store);
                trace_metrics.observe_local_interleaved(
                    kind,
                    local_interleaved_sectors,
                    local_interleaved_dynamic,
                )?;
                if let Some(slot) = schedule.issue_slot(u64::from(inst.pc())) {
                    trace_metrics
                        .operand_bank_pressure
                        .observe(slot.fixed_latency_operands(), schedule.arch)?;
                }
                if active_mask_present && predicate_mask_present {
                    trace_metrics.observe_coherence(
                        inst.sm_id(),
                        inst.predicate_on_lanes(),
                        kind,
                        &sectors,
                        inst.instrumentation_order(),
                    )?;
                }
                if active_mask_present {
                    if let Some((from_function, from_pc, from_active)) = previous_control_flow {
                        trace_metrics.observe_control_flow(
                            from_function,
                            from_pc,
                            from_active,
                            inst.function_unique_id(),
                            inst.pc(),
                            inst.active_mask(),
                        )?;
                    }
                    previous_control_flow =
                        Some((inst.function_unique_id(), inst.pc(), inst.active_mask()));
                    branch_successors.observe(
                        &mut trace_metrics.branch_targets,
                        inst.function_unique_id(),
                        inst.pc(),
                        inst.active_mask(),
                        opcode,
                    )?;
                } else {
                    // Do not connect CFG edges or resolve branch successors
                    // through a record whose lane population is absent.
                    previous_control_flow = None;
                }
                if let Some(order) = inst.instrumentation_order() {
                    async_copy_cohorts
                        .instrumentation_order
                        .push(InstrumentationOrderAnchor {
                            instruction: record_ix,
                            order,
                        });
                }
                insts.push(ReplayInst {
                    pc: inst.pc(),
                    kind,
                    active_lane_mask: if inst.active_mask_present() {
                        inst.active_mask()
                    } else {
                        u32::MAX
                    },
                    lane_mask: if inst.predicate_mask_present() {
                        inst.predicate_on_mask()
                    } else if inst.active_mask_present() {
                        inst.active_mask()
                    } else {
                        u32::MAX
                    },
                });
            }
            branch_successors.finish(&mut trace_metrics.branch_targets)?;
            // Fail-closed traced placement: every roster warp must expose its
            // physical slot. An empty warp has no placement witness and cannot be
            // used to prove either the architecture slot bound or modeled CTA
            // co-residency.
            let Some(warp_slot) = warp_slot else {
                return Err(ReplayError::WarpSlotMissing { path, warp_id });
            };
            if warp_slot < 0 || u32::try_from(warp_slot).unwrap_or(u32::MAX) >= max_resident_warps {
                return Err(ReplayError::WarpSlotOutOfRange {
                    path,
                    warp_id,
                    warp_slot,
                    arch: schedule.arch,
                    max_resident_warps,
                });
            }
            if !seen_warp_slots.insert(warp_slot) {
                return Err(ReplayError::DuplicateWarpSlot { path, warp_slot });
            }
            warps.push(WarpReplay {
                warp_id,
                warp_slot,
                insts,
                sectors,
                async_copy_cohorts: (!async_copy_cohorts.spans.is_empty()
                    || !async_copy_cohorts.instrumentation_order.is_empty())
                .then(|| Box::new(async_copy_cohorts)),
            });
        }
        let Some(sm) = block_sm else {
            return Err(ReplayError::SmIdMissing { path });
        };
        placement.require_traced_sm(&path, sm)?;
        traced_sm_by_launch_index.insert(launch_index, sm);
        let block_replay = BlockReplay {
            launch_index,
            warps,
        };
        instruction_sectors_by_sm
            .entry(sm)
            .or_default()
            .extend(per_sm_instruction_sector_set(
                std::slice::from_ref(&block_replay),
                expected_fuid,
            ));
        per_sm_blocks.entry(sm).or_default().push(block_replay);
    }
    // Completeness gate (fail-closed) on the path replay ACTUALLY consumes: a
    // subset trace (coverage < grid) is admissible ONLY with a valid
    // trace_sampling.json sidecar whose original_grid matches the launch record
    // (gridDim is never rewritten) and whose captured record count matches
    // records_seen. Absent/non-sampling sidecars never waive full coverage.
    let (gx, gy, gz) = kernel_grid(loc.kernel())?;
    let grid = (gx, gy, gz);
    let grid_total = checked_grid_total(grid)?;
    let seen = checked_collection_len("replayed block membership", seen_indices.len())?;
    let launch = ReplayLaunchCoverageBinding {
        device_id: loc.device_id(),
        stream_id: loc.stream_id(),
        kernel_id: loc.kernel().id,
        function_unique_id: loc.kernel().function_unique_id,
        kernel_mangled_name: loc.kernel().name.clone(),
        grid,
        block,
        cluster,
    };
    #[cfg(not(feature = "campaign"))]
    let coverage = {
        if seen != grid_total {
            return Err(ReplayError::IncompleteTrace { seen, grid_total });
        }
        ExtractedCoverage::Full(FullLaunchCoverage {
            launch,
            blocks_observed: seen,
            source: FullCoverageSource::UnsampledComplete,
        })
    };
    #[cfg(feature = "campaign")]
    let coverage = match td.sampling_sidecar() {
        None => {
            if seen != grid_total {
                return Err(ReplayError::IncompleteTrace { seen, grid_total });
            }
            ExtractedCoverage::Full(FullLaunchCoverage {
                launch,
                blocks_observed: seen,
                source: FullCoverageSource::UnsampledComplete,
            })
        }
        Some(sc) => {
            let launch_block = launch.block.ok_or(ReplayError::BlockDimMissing)?;
            if sc.kernel_mangled_name() != loc.kernel().name.as_str() {
                return Err(ReplayError::SamplingKernelMismatch {
                    launch: loc.kernel().name.clone(),
                    sidecar: sc.kernel_mangled_name().to_owned(),
                });
            }
            if sc.original_grid() != grid {
                return Err(ReplayError::GridDimRewritten {
                    launch: grid,
                    sidecar: sc.original_grid(),
                });
            }
            if sc.original_block() != launch_block {
                return Err(ReplayError::BlockDimRewritten {
                    launch: launch_block,
                    sidecar: sc.original_block(),
                });
            }
            let coverage = match sc.method() {
                TraceSamplingMethod::WaveSegmentStratifiedSubset(provenance) => {
                    if provenance.selected_block_ids() != &seen_indices {
                        let declared = checked_collection_len(
                            "trace_sampling selected block membership",
                            provenance.selected_block_ids().len(),
                        )?;
                        return Err(ReplayError::SelectedBlockMembershipMismatch {
                            seen,
                            declared,
                        });
                    }
                    ExtractedCoverage::Sampled(SampledLaunchCoverage {
                        launch,
                        population_blocks: grid_total,
                        blocks_observed: seen,
                        provenance: provenance.clone(),
                    })
                }
                TraceSamplingMethod::LosslessCompressedFull(provenance) => {
                    if seen != grid_total {
                        return Err(ReplayError::IncompleteTrace { seen, grid_total });
                    }
                    ExtractedCoverage::Full(FullLaunchCoverage {
                        launch,
                        blocks_observed: seen,
                        source: FullCoverageSource::LosslessCompressedFull(provenance.clone()),
                    })
                }
            };
            validate_sampling_placement(sc, &traced_sm_by_launch_index, placement)?;
            if records_seen != sc.captured_records() {
                return Err(ReplayError::RecordsCapturedMismatch {
                    records_seen,
                    captured: sc.captured_records(),
                });
            }
            coverage
        }
    };
    for blocks in per_sm_blocks.values_mut() {
        blocks.sort_by_key(|b| b.launch_index);
    }
    trace_metrics.finish_coherence()?;
    if let Some(l2) = tma_shadow_l2 {
        let joint = observe_joint_l2_shadow(
            joint_l2_events,
            l2,
            joint_l2_order_unavailable,
            joint_l2_events_dropped,
        )?;
        trace_metrics.observe_tma_joint_l2(
            joint.requests,
            joint.hits,
            joint.misses,
            joint.tma_load,
            joint.ordinary_read,
            joint.order_unavailable,
            joint.events_dropped,
        )?;
    }
    if let Some(observations) = observations {
        let mut matched = 0_u64;
        let mut unmatched = 0_u64;
        for (key, observation) in &observations.tma {
            let has_address = (observation.is_load && !observation.source_sector_ids.is_empty())
                || (observation.is_store && !observation.destination_sector_ids.is_empty());
            if !has_address {
                continue;
            }
            if tma_dynamic_keys_seen.contains(key) {
                matched = matched.checked_add(1).ok_or(IssueError::CounterOverflow {
                    counter: "TMA shadow dynamic identity matches",
                })?;
            } else {
                unmatched = unmatched
                    .checked_add(1)
                    .ok_or(IssueError::CounterOverflow {
                        counter: "TMA shadow dynamic identity gaps",
                    })?;
            }
        }
        trace_metrics.observe_tma_shadow_dynamic_identity(matched, unmatched)?;
    }
    Ok(ExtractedKernel {
        per_sm_blocks,
        per_sm_instruction_read_sectors: instruction_sectors_by_sm
            .into_iter()
            .map(|(sm, sectors)| {
                u64::try_from(sectors.len())
                    .map(|count| (sm, count))
                    .map_err(|_| IssueError::CounterOverflow {
                        counter: "per-SM compulsory instruction-read sector cardinality",
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?,
        records_seen,
        coverage,
        shared_memory_residency,
        front_end: FrontEndCounters {
            compulsory_instruction_read_sectors: u64::try_from(instruction_sectors.len()).map_err(
                |_| IssueError::CounterOverflow {
                    counter: "compulsory instruction-read sector cardinality",
                },
            )?,
            compulsory_constant_read_sectors: u64::try_from(constant_sectors.len()).map_err(
                |_| IssueError::CounterOverflow {
                    counter: "compulsory constant-read sector cardinality",
                },
            )?,
            compulsory_instruction_line_read_sectors: u64::try_from(instruction_lines.len())
                .map_err(|_| IssueError::CounterOverflow {
                    counter: "compulsory instruction-read line cardinality",
                })?
                .checked_mul(4)
                .ok_or(IssueError::CounterOverflow {
                    counter: "compulsory instruction-read line sector cardinality",
                })?,
            static_instruction_line_read_sectors: u64::try_from(static_instruction_lines.len())
                .map_err(|_| IssueError::CounterOverflow {
                    counter: "static instruction-read line cardinality",
                })?
                .checked_mul(4)
                .ok_or(IssueError::CounterOverflow {
                    counter: "static instruction-read line sector cardinality",
                })?,
        },
        trace_metrics,
    })
}

/// Validate every consumed threadblock and classify launch coverage without
/// running the timing model. The campaign feature uses this for its manifest
/// compatibility path.
#[cfg(feature = "campaign")]
pub(crate) fn admit_launch_coverage(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
) -> Result<ExtractedCoverage, ReplayError> {
    let placement = PlacementConfig::from_arch_card(arch_card, schedule.arch)?;
    require_trace_architecture(td, schedule.arch)?;
    Ok(extract_per_sm_blocks(td, loc, schedule, placement, false, None)?.coverage)
}

/// Fold per-SM results into the kernel-level view, enforcing the
/// one-issue-per-record invariant.
fn fold_kernel_result(
    arch: ArchId,
    per_sm: BTreeMap<i32, SmSimResult>,
    records_seen: u64,
    expect_l1: bool,
    last_processed_system_event_time: Option<u64>,
    l2: Option<(L2Counters, L2SourceSplit, DramStats)>,
) -> Result<FoldedKernelReplay, ReplayError> {
    fold_kernel_result_with_sm_active_time(
        arch,
        per_sm,
        records_seen,
        expect_l1,
        last_processed_system_event_time,
        last_processed_system_event_time,
        l2,
    )
}

fn fold_kernel_result_with_sm_active_time(
    arch: ArchId,
    per_sm: BTreeMap<i32, SmSimResult>,
    records_seen: u64,
    expect_l1: bool,
    last_processed_system_event_time: Option<u64>,
    sm_active_terminal_event_time: Option<u64>,
    l2: Option<(L2Counters, L2SourceSplit, DramStats)>,
) -> Result<FoldedKernelReplay, ReplayError> {
    let mut instructions_issued: u64 = 0;
    let mut diagnostic_last_issue_cycles: u64 = 0;
    let mut latest_local_completion_cycles: Option<u64> = None;
    let mut unmodeled_tails = TailFlags::default();
    let mut l1_totals = expect_l1.then(|| L1Counters::for_arch(arch));
    for (&sm_id, result) in &per_sm {
        instructions_issued = instructions_issued
            .checked_add(result.instructions_issued)
            .ok_or(ReplayError::InstructionCountOverflow {
                sm_id,
                accumulated: instructions_issued,
                addend: result.instructions_issued,
            })?;
        diagnostic_last_issue_cycles = diagnostic_last_issue_cycles.max(result.cycles);
        if let Some(completion) = result.modeled_local_completion_cycles {
            latest_local_completion_cycles = Some(
                latest_local_completion_cycles
                    .map_or(completion, |current| current.max(completion)),
            );
        }
        unmodeled_tails = unmodeled_tails.union(result.unmodeled_tails);
        let found_l1 = result.l1.is_some();
        if found_l1 != expect_l1 {
            return Err(ReplayError::L1PresenceMismatch {
                sm_id,
                expected: expect_l1,
                found: found_l1,
            });
        }
        if let (Some(c), Some(total)) = (&result.l1, &mut l1_totals) {
            total
                .add_checked(c)
                .map_err(|source| ReplayError::L1Aggregation { sm_id, source })?;
        }
    }
    if instructions_issued != records_seen {
        return Err(ReplayError::CountMismatch {
            issued: instructions_issued,
            records: records_seen,
        });
    }
    let issue_frontier = (instructions_issued != 0).then_some(diagnostic_last_issue_cycles);
    let terminal_modeled_event_time = [
        issue_frontier,
        last_processed_system_event_time,
        latest_local_completion_cycles,
    ]
    .into_iter()
    .flatten()
    .max();
    let (l2, l2_by_source, dram) = match l2 {
        Some((counters, split, dram)) => (Some(counters), Some(split), Some(dram)),
        None => (None, None, None),
    };
    Ok(FoldedKernelReplay {
        arch,
        instructions_issued,
        records_seen,
        per_sm,
        diagnostic_last_issue_cycles,
        last_processed_system_event_time,
        terminal_modeled_event_time,
        sm_active_terminal_event_time: [
            issue_frontier,
            sm_active_terminal_event_time,
            latest_local_completion_cycles,
        ]
        .into_iter()
        .flatten()
        .max(),
        l1: l1_totals,
        l2,
        l2_by_source,
        dram,
        front_end: FrontEndCounters::default(),
        trace_metrics: TraceMetrics::default(),
        unmodeled_tails,
    })
}

fn require_system_tail_consistency(
    system: TailFlags,
    per_sm: TailFlags,
) -> Result<(), ReplayError> {
    if system != per_sm {
        return Err(ReplayError::SystemTailMismatch { system, per_sm });
    }
    Ok(())
}

fn classify_kernel_result(
    coverage: ExtractedCoverage,
    configuration: ReplayConfiguration,
    replay: FoldedKernelReplay,
) -> KernelReplayOutcome {
    match coverage {
        ExtractedCoverage::Full(coverage) => KernelReplayOutcome::Full(FullKernelReplayResult {
            coverage,
            configuration,
            replay,
        }),
        #[cfg(feature = "campaign")]
        ExtractedCoverage::Sampled(coverage) => {
            let FoldedKernelReplay {
                arch,
                instructions_issued,
                records_seen,
                per_sm: _,
                diagnostic_last_issue_cycles: _,
                last_processed_system_event_time: _,
                terminal_modeled_event_time: _,
                sm_active_terminal_event_time: _,
                l1,
                l2,
                l2_by_source,
                dram,
                front_end,
                trace_metrics: _,
                unmodeled_tails,
            } = replay;
            KernelReplayOutcome::SampledObserved(SampledKernelReplayResult {
                coverage,
                configuration,
                counters: SampledObservedCounters {
                    arch,
                    instructions_issued,
                    records_seen,
                    l1,
                    l2,
                    l2_by_source,
                    dram,
                    front_end,
                    unmodeled_tails,
                },
            })
        }
    }
}

fn lookup_schedule<'a>(
    schedules: &'a BTreeMap<i32, FunctionSchedule>,
    loc: &KernelLoc<'_>,
) -> Result<&'a FunctionSchedule, ReplayError> {
    let expected_fuid = loc.kernel().function_unique_id;
    schedules
        .get(&expected_fuid)
        .ok_or(ReplayError::ScheduleMissing {
            function_unique_id: expected_fuid,
        })
}

/// Memory-hierarchy mode for one kernel replay — the ONE typed selector
/// every harness goes through (a pair of optional configs would let a
/// plumbing slip degrade a cell to the wrong mode silently).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayMemMode {
    /// No cache model (flat global latency).
    Flat,
    /// Per-SM L1s with constant fill latency below them (the pre-L2
    /// lockstep/ablation surface).
    L1Const(L1Config),
    /// Per-SM L1s over ONE shared L2 (the full hierarchy): L1 misses
    /// priced L2-hit vs DRAM by the shared state through the event
    /// system.
    L1SharedL2 {
        l1: L1Config,
        l2: L2Config,
        dram: DramBw,
        coherence: Option<CoherenceConfig>,
    },
}

/// Temporal CTA co-residency provenance. The current trace wire carries spatial
/// `sm_id`/`warpid_sm` placement but no block start/end timestamps, so exact
/// hardware co-residency cannot be claimed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoResidency {
    /// Blocks are admitted in launch-index order under the modeled resident-warp
    /// ceiling and architecture-card shared-memory capacity; concurrent CTAs must
    /// also use disjoint traced physical warp slots. This is never a claim that the
    /// hardware used the same overlap.
    Modeled,
}

/// Exact structural identity of every issue and memory-hierarchy parameter used
/// for one replay. No digest or caller label substitutes for these values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayConfiguration {
    issue_params: IssueModelParams,
    memory_mode: ReplayMemMode,
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
    co_residency: CoResidency,
}

impl ReplayConfiguration {
    /// Build a declaration-side identity while enforcing architecture lockstep
    /// across all nested memory configurations.
    pub fn checked(
        issue_params: IssueModelParams,
        memory_mode: ReplayMemMode,
    ) -> Result<Self, ReplayError> {
        validate_replay_mode_architecture(issue_params.arch, memory_mode)?;
        issue_params.validate_resource_source()?;
        Ok(Self {
            issue_params,
            memory_mode,
            memory_admission_feedback: None,
            co_residency: CoResidency::Modeled,
        })
    }

    /// Build the structural identity for the timing-active schema-10 lane.
    /// The candidate's operand-visibility floor must equal the issue model's
    /// existing read-barrier latency: feedback replaces queue readiness only
    /// and must not silently introduce (or ignore) a second latency source.
    pub fn checked_with_memory_admission_feedback(
        issue_params: IssueModelParams,
        memory_mode: ReplayMemMode,
        candidate: MemoryAdmissionCandidate,
    ) -> Result<Self, ReplayError> {
        if !matches!(memory_mode, ReplayMemMode::L1SharedL2 { .. }) {
            return Err(ReplayError::MemoryAdmissionCandidatesRequireSharedL2);
        }
        candidate.validate_feedback(issue_params.read_barrier_latency)?;
        let mut configuration = Self::checked(issue_params, memory_mode)?;
        configuration.memory_admission_feedback = Some(candidate);
        Ok(configuration)
    }

    pub fn arch(&self) -> ArchId {
        self.issue_params.arch
    }

    pub fn issue_params(&self) -> IssueModelParams {
        self.issue_params
    }

    pub fn memory_mode(&self) -> ReplayMemMode {
        self.memory_mode
    }

    /// Timing-active finite admission candidate, when schema 10 is enabled.
    pub fn memory_admission_feedback(&self) -> Option<MemoryAdmissionCandidate> {
        self.memory_admission_feedback
    }

    pub fn co_residency(&self) -> CoResidency {
        self.co_residency
    }

    /// Return the residual architecture-scoped SM-active observation boundary
    /// used by the explicit frontier projection. Once the causal IMC path is
    /// active, the independent 964-cycle legacy residual is not additive: it
    /// represented the missing constant-bank work and would double-charge the
    /// same silicon interval. The current SM86 causal lane therefore exposes a
    /// zero residual while retaining the historical capability for old data.
    pub fn sm_active_activation_cycles(&self) -> u64 {
        if !matches!(self.memory_mode, ReplayMemMode::L1SharedL2 { .. })
            || !matches!(
                self.issue_params.resource_source(),
                IssueResourceSource::ArchitectureDefaults
            )
        {
            return 0;
        }
        if self.issue_params.immediate_constant_cache().is_some() {
            return 0;
        }
        self.issue_params
            .arch
            .sm_active_observation_envelope()
            .require("sm_active_observation_envelope")
            .map_or(0, |envelope| envelope.cycles())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrderedSessionLaunch {
    kernel_id: i32,
    function_unique_id: i32,
}

/// Typed failures for an ordered same-stream replay session. Session errors
/// are separate from one-launch replay failures so callers can distinguish a
/// malformed launch from a launch-chain ownership/order mismatch.
#[derive(Debug, thiserror::Error)]
pub enum ReplaySessionError {
    #[error(transparent)]
    Replay(#[from] ReplayError),
    #[error("stream-L2 cache scope requires L1SharedL2 replay mode, got {mode:?}")]
    SharedL2ModeRequired { mode: ReplayMemMode },
    #[error("stream-L2 replay does not persist the opt-in coherence directory; use launch scope")]
    CoherenceRequiresLaunchScope,
    #[error("trace stream ({device_id}, {stream_id}) contains no kernel launches")]
    EmptyStream { device_id: i64, stream_id: i64 },
    #[error("replay session is bound to trace root {expected}, got {got}")]
    TraceRootMismatch { expected: PathBuf, got: PathBuf },
    #[error("replay session is bound to a different loaded trace lineage")]
    TraceLineageMismatch,
    #[error("replay session is bound to device {expected}, got {got}")]
    DeviceMismatch { expected: i64, got: i64 },
    #[error("replay session is bound to stream {expected}, got {got}")]
    StreamMismatch { expected: i64, got: i64 },
    #[error("replay session configuration mismatch: expected {expected:?}, got {got:?}")]
    ConfigurationMismatch {
        expected: ReplayConfiguration,
        got: ReplayConfiguration,
    },
    #[error(
        "replay session architecture-card placement mismatch: expected sku={expected_sku:?}, sm_count={expected_sm_count}, max_shared_bytes_per_sm={expected_max_shared_bytes_per_sm}; got sku={got_sku:?}, sm_count={got_sm_count}, max_shared_bytes_per_sm={got_max_shared_bytes_per_sm}"
    )]
    ArchitectureCardMismatch {
        expected_sku: Sku,
        expected_sm_count: u32,
        expected_max_shared_bytes_per_sm: u64,
        got_sku: Sku,
        got_sm_count: u32,
        got_max_shared_bytes_per_sm: u64,
    },
    #[error(
        "replay session launch position {position} expects kernel {expected_kernel_id}/function {expected_function_unique_id}, got kernel {got_kernel_id}/function {got_function_unique_id}"
    )]
    LaunchOrderMismatch {
        position: usize,
        expected_kernel_id: i32,
        expected_function_unique_id: i32,
        got_kernel_id: i32,
        got_function_unique_id: i32,
    },
    #[error("replay session already consumed all {completed_launches} launches")]
    LaunchSequenceComplete { completed_launches: usize },
    #[error("stream-L2 replay outcome omitted its per-launch L2 counter delta")]
    LaunchL2CountersMissing,
    #[error(
        "stream-L2 cumulative counters do not reconcile: reported={reported:?}, model={model:?}"
    )]
    CumulativeL2CounterMismatch {
        reported: L2Counters,
        model: L2Counters,
    },
}

/// Ordered same-device/same-stream replay state. Only the shared L2 model
/// survives a successful launch boundary. Per-SM L1, MSHR, scheduler,
/// dependence, async-copy, pipeline, CTA, event-queue, and DRAM-server state is
/// rebuilt by the ordinary one-launch system path on every call.
pub struct StreamL2ReplaySession {
    trace_root: PathBuf,
    trace_lineage: TraceLineage,
    configuration: ReplayConfiguration,
    placement: PlacementConfig,
    device_id: i64,
    stream_id: i64,
    expected_launches: Vec<OrderedSessionLaunch>,
    completed_launches: usize,
    l2: L2Model,
    cumulative_reported_l2: L2Counters,
}

impl StreamL2ReplaySession {
    /// Bind a session to one trace root and one exact device/stream launch
    /// sequence. Construction is accepted only for the shared-L2 replay mode.
    pub fn new(
        td: &TraceDir,
        arch_card: &ArchCard,
        params: &IssueModelParams,
        mode: ReplayMemMode,
        device_id: i64,
        stream_id: i64,
    ) -> Result<Self, ReplaySessionError> {
        let configuration = ReplayConfiguration::checked(*params, mode)?;
        let l2_config = match mode {
            ReplayMemMode::L1SharedL2 { l2, coherence, .. } => {
                if coherence.is_some() {
                    return Err(ReplaySessionError::CoherenceRequiresLaunchScope);
                }
                l2
            }
            _ => return Err(ReplaySessionError::SharedL2ModeRequired { mode }),
        };
        require_trace_architecture(td, configuration.arch())?;
        let placement = PlacementConfig::from_arch_card(arch_card, configuration.arch())?;
        let expected_launches: Vec<_> = td
            .kernels()
            .filter(|loc| loc.device_id() == device_id && loc.stream_id() == stream_id)
            .map(|loc| OrderedSessionLaunch {
                kernel_id: loc.kernel().id,
                function_unique_id: loc.kernel().function_unique_id,
            })
            .collect();
        if expected_launches.is_empty() {
            return Err(ReplaySessionError::EmptyStream {
                device_id,
                stream_id,
            });
        }
        let l2 = L2Model::new(l2_config).map_err(ReplayError::from)?;
        Ok(Self {
            trace_root: td.root().to_path_buf(),
            trace_lineage: td.lineage(),
            configuration,
            placement,
            device_id,
            stream_id,
            expected_launches,
            completed_launches: 0,
            l2,
            cumulative_reported_l2: L2Counters::for_arch(configuration.arch()),
        })
    }

    pub fn configuration(&self) -> ReplayConfiguration {
        self.configuration
    }

    pub fn device_id(&self) -> i64 {
        self.device_id
    }

    pub fn stream_id(&self) -> i64 {
        self.stream_id
    }

    pub fn completed_launches(&self) -> usize {
        self.completed_launches
    }

    pub fn remaining_launches(&self) -> usize {
        self.expected_launches.len() - self.completed_launches
    }

    pub fn is_complete(&self) -> bool {
        self.completed_launches == self.expected_launches.len()
    }

    pub fn cumulative_l2_counters(&self) -> L2Counters {
        self.cumulative_reported_l2
    }

    pub fn pending_l2_sectors(&self) -> usize {
        self.l2.pending_sector_count()
    }

    /// Replay the next launch transactionally. The session cache and cursor are
    /// committed together only after the launch result, drained-state check,
    /// per-launch delta, and cumulative-counter reconciliation all succeed.
    pub fn replay_next(
        &mut self,
        td: &TraceDir,
        loc: &KernelLoc<'_>,
        schedules: &BTreeMap<i32, FunctionSchedule>,
        arch_card: &ArchCard,
        params: &IssueModelParams,
        mode: ReplayMemMode,
    ) -> Result<KernelReplayOutcome, ReplaySessionError> {
        if td.root() != self.trace_root {
            return Err(ReplaySessionError::TraceRootMismatch {
                expected: self.trace_root.clone(),
                got: td.root().to_path_buf(),
            });
        }
        if td.lineage() != self.trace_lineage {
            return Err(ReplaySessionError::TraceLineageMismatch);
        }
        if loc.device_id() != self.device_id {
            return Err(ReplaySessionError::DeviceMismatch {
                expected: self.device_id,
                got: loc.device_id(),
            });
        }
        if loc.stream_id() != self.stream_id {
            return Err(ReplaySessionError::StreamMismatch {
                expected: self.stream_id,
                got: loc.stream_id(),
            });
        }
        let Some(expected_launch) = self.expected_launches.get(self.completed_launches) else {
            return Err(ReplaySessionError::LaunchSequenceComplete {
                completed_launches: self.completed_launches,
            });
        };
        if loc.kernel().id != expected_launch.kernel_id
            || loc.kernel().function_unique_id != expected_launch.function_unique_id
        {
            return Err(ReplaySessionError::LaunchOrderMismatch {
                position: self.completed_launches,
                expected_kernel_id: expected_launch.kernel_id,
                expected_function_unique_id: expected_launch.function_unique_id,
                got_kernel_id: loc.kernel().id,
                got_function_unique_id: loc.kernel().function_unique_id,
            });
        }

        let got_configuration = ReplayConfiguration::checked(*params, mode)?;
        if got_configuration != self.configuration {
            return Err(ReplaySessionError::ConfigurationMismatch {
                expected: self.configuration,
                got: got_configuration,
            });
        }
        let got_placement = PlacementConfig::from_arch_card(arch_card, self.configuration.arch())?;
        if got_placement != self.placement {
            return Err(ReplaySessionError::ArchitectureCardMismatch {
                expected_sku: self.placement.sku,
                expected_sm_count: self.placement.sm_count,
                expected_max_shared_bytes_per_sm: self.placement.max_shared_bytes_per_sm,
                got_sku: got_placement.sku,
                got_sm_count: got_placement.sm_count,
                got_max_shared_bytes_per_sm: got_placement.max_shared_bytes_per_sm,
            });
        }
        if self.l2.pending_sector_count() != 0 {
            return Err(ReplayError::RetainedL2NotDrained {
                pending_sectors: self.l2.pending_sector_count(),
            }
            .into());
        }

        let mut staged_l2 = self.l2.try_snapshot().map_err(ReplayError::from)?;
        let schedule = lookup_schedule(schedules, loc)?;
        let outcome = replay_kernel_with_schedule_and_l2(
            td,
            loc,
            schedule,
            arch_card,
            params,
            mode,
            Some(&mut staged_l2),
        )?;
        if staged_l2.pending_sector_count() != 0 {
            return Err(ReplayError::RetainedL2NotDrained {
                pending_sectors: staged_l2.pending_sector_count(),
            }
            .into());
        }
        let launch_delta = outcome
            .l2_counters()
            .copied()
            .ok_or(ReplaySessionError::LaunchL2CountersMissing)?;
        let mut cumulative_reported_l2 = self.cumulative_reported_l2;
        cumulative_reported_l2
            .add_checked(&launch_delta)
            .map_err(ReplayError::from)?;
        let model_counters = *staged_l2.counters().map_err(ReplayError::from)?;
        if cumulative_reported_l2 != model_counters {
            return Err(ReplaySessionError::CumulativeL2CounterMismatch {
                reported: cumulative_reported_l2,
                model: model_counters,
            });
        }

        self.l2 = staged_l2;
        self.cumulative_reported_l2 = cumulative_reported_l2;
        self.completed_launches += 1;
        Ok(outcome)
    }
}

fn require_replay_arch(
    expected: ArchId,
    got: ArchId,
    component: &'static str,
) -> Result<(), ReplayError> {
    if got != expected {
        return Err(ReplayError::ArchitectureMismatch {
            component,
            expected,
            got,
        });
    }
    Ok(())
}

fn require_trace_architecture(td: &TraceDir, expected: ArchId) -> Result<(), ReplayError> {
    let binary_version = td.trace().binary_version;
    let numeric = u32::try_from(binary_version)
        .map_err(|_| ReplayError::TraceArchitectureVersionOutOfRange { binary_version })?;
    let traced =
        ArchId::from_numeric(numeric).map_err(|source| ReplayError::TraceArchitectureUnknown {
            binary_version: numeric,
            source,
        })?;
    require_replay_arch(expected, traced, "trace binary_version")
}

fn validate_replay_mode_architecture(arch: ArchId, mode: ReplayMemMode) -> Result<(), ReplayError> {
    match mode {
        ReplayMemMode::Flat => {}
        ReplayMemMode::L1Const(l1) => {
            require_replay_arch(arch, l1.arch, "L1 configuration")?;
        }
        ReplayMemMode::L1SharedL2 {
            l1,
            l2,
            dram,
            coherence: _,
        } => {
            require_replay_arch(arch, l1.arch, "L1 configuration")?;
            require_replay_arch(arch, l2.arch, "L2 configuration")?;
            require_replay_arch(arch, dram.arch(), "DRAM configuration")?;
        }
    }
    Ok(())
}

fn validate_replay_architecture(
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<ReplayConfiguration, ReplayError> {
    require_replay_arch(schedule.arch, params.arch, "issue parameters")?;
    match memory_admission_feedback {
        Some(candidate) => {
            ReplayConfiguration::checked_with_memory_admission_feedback(*params, mode, candidate)
        }
        None => ReplayConfiguration::checked(*params, mode),
    }
}

fn async_copy_destination_fallback_sms(
    payload_contract: MemoryPayloadContract,
    per_sm_blocks: &BTreeMap<i32, Vec<BlockReplay>>,
) -> BTreeSet<i32> {
    per_sm_blocks
        .iter()
        .filter_map(|(&sm, blocks)| {
            blocks
                .iter()
                .flat_map(|block| &block.warps)
                .flat_map(|warp| &warp.insts)
                .any(|inst| {
                    matches!(inst.kind, AccessKind::GlobalAsyncCopy { .. })
                        && (payload_contract
                            != MemoryPayloadContract::AddressCarryingLdgstsDestinationV3
                            || matches!(
                                inst.kind,
                                AccessKind::GlobalAsyncCopy {
                                    l1_policy: LdgstsL1Policy::BypassL1,
                                    ..
                                }
                            ))
                })
                .then_some(sm)
        })
        .collect()
}

fn replay_kernel_with_schedule_and_l2(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    retained_l2: Option<&mut L2Model>,
) -> Result<KernelReplayOutcome, ReplayError> {
    replay_kernel_with_schedule_and_l2_and_candidates(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        retained_l2,
        &[],
        &[],
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn replay_kernel_with_schedule_and_l2_and_candidates(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    retained_l2: Option<&mut L2Model>,
    memory_service_candidates: &[MemoryServiceCandidate],
    memory_admission_candidates: &[MemoryAdmissionCandidate],
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
) -> Result<KernelReplayOutcome, ReplayError> {
    replay_kernel_with_schedule_and_l2_and_candidates_with_payload_events(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        retained_l2,
        memory_service_candidates,
        memory_admission_candidates,
        memory_admission_feedback,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn replay_kernel_with_schedule_and_l2_and_candidates_with_payload_events(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    retained_l2: Option<&mut L2Model>,
    memory_service_candidates: &[MemoryServiceCandidate],
    memory_admission_candidates: &[MemoryAdmissionCandidate],
    memory_admission_feedback: Option<MemoryAdmissionCandidate>,
    payload_events: &[L1PayloadAdmissionEvent],
) -> Result<KernelReplayOutcome, ReplayError> {
    if (!memory_service_candidates.is_empty()
        || !memory_admission_candidates.is_empty()
        || memory_admission_feedback.is_some())
        && schedule.arch != ArchId::Sm86
    {
        return Err(ReplayError::ArchitectureMismatch {
            component: "SM86 memory candidate lane",
            expected: ArchId::Sm86,
            got: schedule.arch,
        });
    }
    if !payload_events.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::PayloadAdmissionRequiresSharedL2);
    }
    if !memory_service_candidates.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::MemoryServiceCandidatesRequireSharedL2);
    }
    if !memory_admission_candidates.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. })
    {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireSharedL2);
    }
    if memory_admission_feedback.is_some() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireSharedL2);
    }
    validate_memory_service_candidates(memory_service_candidates).map_err(IssueError::from)?;
    validate_memory_admission_candidates(memory_admission_candidates).map_err(IssueError::from)?;
    if let Some(candidate) = memory_admission_feedback {
        candidate.validate().map_err(IssueError::from)?;
    }
    let configuration =
        validate_replay_architecture(schedule, params, mode, memory_admission_feedback)?;
    let placement = PlacementConfig::from_arch_card(arch_card, schedule.arch)?;
    require_trace_architecture(td, schedule.arch)?;
    // Ordinary launches can use the hierarchy classifier in flat mode.
    // Mixed async/TMA launches use the alpha compact classifier to preserve
    // opaque sidecar payloads without borrowing the SM86 completion protocol;
    // that classifier also retains all observed global/local sector spans.
    let alpha_flat_sector_decode = matches!(mode, ReplayMemMode::Flat)
        && params.resource_source() == IssueResourceSource::Sm120Alpha
        && !schedule.by_pc.values().any(|slot| {
            slot.ldgsts_l1_policy().is_some()
                || matches!(
                    slot.op_class(),
                    OpClass::AsyncCopyCommit | OpClass::AsyncCopyWait
                )
                || matches!(
                    opcode_head(slot.opcode_text()),
                    "UTMALDG" | "UTMASTG" | "UBLKCP"
                )
        });
    let decode_sectors = !matches!(mode, ReplayMemMode::Flat) || alpha_flat_sector_decode;
    let ExtractedKernel {
        per_sm_blocks,
        per_sm_instruction_read_sectors,
        records_seen,
        coverage,
        shared_memory_residency,
        front_end,
        mut trace_metrics,
    } = extract_per_sm_blocks(
        td,
        loc,
        schedule,
        placement,
        decode_sectors,
        match mode {
            ReplayMemMode::L1SharedL2 { l2, .. } => Some(l2),
            _ => None,
        },
    )?;
    // Keep the producer/consumer contract explicit for the report-only SM86
    // write-tag observer. The emitter must not consume `write_lookup_hits`
    // when the issue model did not instantiate the observer for this launch.
    trace_metrics.l1_write_lookup_observer_active =
        stateful_write_lookup_observer_enabled(schedule);
    let instrumentation_order = observe_async_instrumentation_order(&per_sm_blocks, schedule)?;
    #[cfg(feature = "campaign")]
    if matches!(
        mode,
        ReplayMemMode::L1SharedL2 {
            dram: DramBw::CardedWriteback { .. },
            ..
        }
    ) && matches!(&coverage, ExtractedCoverage::Sampled(_))
    {
        return Err(ReplayError::WritebackRequiresFullCoverage);
    }
    if let ReplayMemMode::L1SharedL2 {
        coherence: Some(_), ..
    } = mode
    {
        let full_coverage = matches!(&coverage, ExtractedCoverage::Full(_));
        let active_masks_missing = trace_metrics.instructions.active_mask_missing_instructions;
        let predicate_masks_missing = trace_metrics
            .instructions
            .predicate_mask_missing_instructions;
        if !full_coverage
            || !trace_metrics.sectors_decoded
            || trace_metrics.coherence.unknown_sm_or_line_records != 0
            || active_masks_missing != 0
            || predicate_masks_missing != 0
        {
            return Err(ReplayError::CoherenceTopologyIncomplete {
                full_coverage,
                decoded: trace_metrics.sectors_decoded,
                unknown_records: trace_metrics.coherence.unknown_sm_or_line_records,
                active_masks_missing,
                predicate_masks_missing,
            });
        }
    }
    #[cfg(feature = "campaign")]
    if !memory_service_candidates.is_empty() && matches!(&coverage, ExtractedCoverage::Sampled(_)) {
        return Err(ReplayError::MemoryServiceCandidatesRequireFullCoverage);
    }
    #[cfg(feature = "campaign")]
    if !memory_admission_candidates.is_empty() && matches!(&coverage, ExtractedCoverage::Sampled(_))
    {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireFullCoverage);
    }
    #[cfg(feature = "campaign")]
    if memory_admission_feedback.is_some() && matches!(&coverage, ExtractedCoverage::Sampled(_)) {
        return Err(ReplayError::MemoryAdmissionCandidatesRequireFullCoverage);
    }
    let async_copy_destination_fallback_sms =
        async_copy_destination_fallback_sms(td.memory_payload_contract(), &per_sm_blocks);
    let mut replay = match mode {
        ReplayMemMode::Flat | ReplayMemMode::L1Const(_) => {
            let l1_config = match mode {
                ReplayMemMode::L1Const(cfg) => Some(cfg),
                _ => None,
            };
            let mut per_sm = BTreeMap::new();
            for (sm, blocks) in &per_sm_blocks {
                let mut result = simulate_sm_with_residency(
                    blocks,
                    schedule,
                    params,
                    shared_memory_residency,
                    l1_config,
                )?;
                result.activity.front_end_instruction_read_sectors =
                    per_sm_instruction_read_sectors
                        .get(sm)
                        .copied()
                        .unwrap_or(0);
                if async_copy_destination_fallback_sms.contains(sm) {
                    result.unmodeled_tails.ldgsts_dst_shared_timing = true;
                }
                if let Some(observation) = instrumentation_order.get(sm) {
                    result.readiness.async_instrumentation_order = Some(Box::new(*observation));
                }
                per_sm.insert(*sm, result);
            }
            for result in per_sm.values() {
                append_bounded_async_decision_snapshots(
                    &mut trace_metrics,
                    &result.readiness.async_decision_snapshots,
                    result.readiness.async_decision_snapshots_dropped,
                )?;
            }
            fold_kernel_result(
                schedule.arch,
                per_sm,
                records_seen,
                l1_config.is_some(),
                None,
                None,
            )?
        }
        ReplayMemMode::L1SharedL2 {
            l1,
            l2,
            dram,
            coherence,
        } => {
            let mut fresh_l2;
            let l2_model = match retained_l2 {
                Some(model) => {
                    if *model.config() != l2 {
                        return Err(ReplayError::RetainedL2ConfigurationMismatch {
                            expected: l2,
                            got: *model.config(),
                        });
                    }
                    model
                }
                None => {
                    fresh_l2 = L2Model::new(l2)?;
                    &mut fresh_l2
                }
            };
            let pending_sectors = l2_model.pending_sector_count();
            if pending_sectors != 0 {
                return Err(ReplayError::RetainedL2NotDrained { pending_sectors });
            }
            let counters_before = *l2_model.counters()?;
            let mut sys = if payload_events.is_empty() {
                simulate_kernel_shared_with_residency_and_candidates_with_coherence(
                    &per_sm_blocks,
                    schedule,
                    params,
                    shared_memory_residency,
                    l1,
                    l2_model,
                    dram,
                    memory_service_candidates,
                    memory_admission_candidates,
                    memory_admission_feedback,
                    coherence,
                )?
            } else {
                simulate_kernel_shared_with_residency_and_candidates_with_coherence_and_payload_events(
                    &per_sm_blocks,
                    schedule,
                    params,
                    shared_memory_residency,
                    l1,
                    l2_model,
                    dram,
                    memory_service_candidates,
                    memory_admission_candidates,
                    memory_admission_feedback,
                    coherence,
                    payload_events,
                )?
            };
            let pending_sectors = l2_model.pending_sector_count();
            if pending_sectors != 0 {
                return Err(ReplayError::RetainedL2NotDrained { pending_sectors });
            }
            let launch_l2 = sys.l2.checked_delta_since(&counters_before)?;
            trace_metrics.observe_l2_system_locality_summary(
                launch_l2.lookup_requests,
                launch_l2.lookup_hits,
                sys.l2_by_source.read_pending_merge_hits,
                sys.l2_by_source.store_lookup_requests,
            );
            for sm in &async_copy_destination_fallback_sms {
                if let Some(result) = sys.per_sm.get_mut(sm) {
                    result.unmodeled_tails.ldgsts_dst_shared_timing = true;
                }
            }
            for (sm, sectors) in &per_sm_instruction_read_sectors {
                if let Some(result) = sys.per_sm.get_mut(sm) {
                    result.activity.front_end_instruction_read_sectors = *sectors;
                }
            }
            for (sm, observation) in &instrumentation_order {
                let result = sys.per_sm.get_mut(sm).unwrap_or_else(|| {
                    unreachable!("order observation SM escaped the simulated SM roster")
                });
                result.readiness.async_instrumentation_order = Some(Box::new(*observation));
            }
            if !async_copy_destination_fallback_sms.is_empty() {
                sys.unmodeled_tails.ldgsts_dst_shared_timing = true;
            }
            for result in sys.per_sm.values() {
                append_bounded_async_decision_snapshots(
                    &mut trace_metrics,
                    &result.readiness.async_decision_snapshots,
                    result.readiness.async_decision_snapshots_dropped,
                )?;
            }
            require_replay_arch(schedule.arch, sys.arch, "system result")?;
            let system_unmodeled_tails = sys.unmodeled_tails;
            let replay = fold_kernel_result_with_sm_active_time(
                schedule.arch,
                sys.per_sm,
                records_seen,
                true,
                sys.terminal_event_time,
                sys.sm_active_terminal_event_time,
                Some((launch_l2, sys.l2_by_source, sys.dram)),
            )?;
            require_system_tail_consistency(system_unmodeled_tails, replay.unmodeled_tails)?;
            replay
        }
    };
    for result in replay.per_sm.values() {
        if let Some(audit) = result.l1_merge_audit {
            let (global_hits, global_lookups) = result
                .l1
                .map(|counters| {
                    let origin = counters.origin(sim_mem::MemoryOrigin::Global);
                    (origin.hits, origin.lookup_sectors())
                })
                .unwrap_or((0, 0));
            trace_metrics.observe_l1_merge_audit(
                audit.same_request_merges,
                audit.older_pending_merges,
                audit.already_requested_merges,
                audit.unrequested_pending_merges,
                audit.ordinary_same_request_merges,
                audit.ordinary_older_pending_merges,
                audit.async_same_request_merges,
                audit.async_older_pending_merges,
                audit.ordinary_global_older_pending_merges,
                audit.constant_global_older_pending_merges,
                audit.ordinary_global_same_request_merges,
                audit.constant_global_same_request_merges,
                audit.async_global_same_request_merges,
                audit.ordinary_global_hits,
                audit.async_global_hits,
                audit.ordinary_global_older_sector_pending_merges,
                audit.constant_global_older_sector_pending_merges,
                audit.async_global_older_sector_pending_merges,
                audit.unrequested_subline_miss_sectors,
                audit.wavefront_mask_lookup_hits,
                audit.wavefront_mask_lookup_misses,
                audit.constant_global_hits,
                audit.constant_global_requests,
                global_hits,
                global_lookups,
                audit.request_masks,
            )?;
        }
    }
    trace_metrics.finalize_reuse_observer();
    trace_metrics.finalize_shared_load_observer();
    replay.front_end = front_end;
    replay.trace_metrics = trace_metrics;
    // A drained hypothetical service queue is not a completion relation for
    // native stores/atomics. Keep the system's unknown tails: one completion
    // timestamp or the absence of traced cross-SM sharing cannot prove that
    // all write/visibility obligations were represented.
    Ok(classify_kernel_result(coverage, configuration, replay))
}

fn replay_kernel_with_schedule(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedule: &FunctionSchedule,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<KernelReplayOutcome, ReplayError> {
    replay_kernel_with_schedule_and_l2(td, loc, schedule, arch_card, params, mode, None)
}

/// Replay the one launch represented by [`AdmittedLaunchEvidence`]. This is the
/// compatibility entry point used by the optional historical campaign runner.
///
/// Caches are fresh per kernel (cold, NCU `--cache-control all` parity).
#[cfg(feature = "campaign")]
pub fn replay_kernel(
    evidence: &AdmittedLaunchEvidence,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<KernelReplayOutcome, ReplayError> {
    let identity = evidence.identity();
    let td = evidence.trace_dir();
    let loc = td
        .kernels()
        .find(|loc| {
            loc.device_id() == identity.device_id()
                && loc.stream_id() == identity.stream_id()
                && loc.kernel().id == identity.kernel_id()
                && loc.kernel().function_unique_id == identity.function_unique_id()
        })
        .ok_or(ReplayError::AdmittedLaunchMissing {
            device_id: identity.device_id(),
            stream_id: identity.stream_id(),
            kernel_id: identity.kernel_id(),
        })?;
    replay_kernel_with_schedule(
        td,
        &loc,
        evidence.schedule(),
        evidence.arch_card(),
        params,
        mode,
    )
}

/// Replay one launch directly from loaded trace, schedule, architecture-card,
/// and model inputs. This is the ordinary research entry point; it does not
/// require an evidence manifest or an admission step.
pub fn replay_trace_kernel(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<KernelReplayOutcome, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    replay_kernel_with_schedule(td, loc, schedule, arch_card, params, mode)
}

/// Replay one launch with explicit producer-supplied value-bearing L1 events.
/// Each event carries its simulator-cycle timestamp and a residency witness;
/// this API never converts NCU/CUPTI/device time or infers a cycle from payload
/// sequence. Ordinary replay remains unchanged when this entry point is not
/// selected.
pub fn replay_trace_kernel_with_l1_payload_events(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    payload_events: &[L1PayloadAdmissionEvent],
) -> Result<KernelReplayOutcome, ReplayError> {
    if !payload_events.is_empty() && !matches!(mode, ReplayMemMode::L1SharedL2 { .. }) {
        return Err(ReplayError::PayloadAdmissionRequiresSharedL2);
    }
    let schedule = lookup_schedule(schedules, loc)?;
    let locations = td.dynamic_memory_join_locations(loc)?;
    let mut bound_events = payload_events.to_vec();
    for (index, event) in bound_events.iter_mut().enumerate() {
        let Some(identity) = event.identity else {
            return Err(ReplayError::PayloadAdmissionIdentityMissing { index });
        };
        let Some(location) = locations.get(&identity) else {
            return Err(ReplayError::PayloadAdmissionIdentityNotObserved { index });
        };
        if location.sm_id != Some(event.sm_id) {
            return Err(ReplayError::PayloadAdmissionSmMismatch {
                index,
                event_sm: event.sm_id,
                trace_sm: location.sm_id,
            });
        }
        if location.access_kind != Some(pb::MemoryOpAccessKind::Store) {
            return Err(ReplayError::PayloadAdmissionNotStore {
                index,
                access_kind: location.access_kind,
            });
        }
        let Some(warp_slot) = location.warp_slot else {
            return Err(ReplayError::PayloadAdmissionWriterPlacementMissing { index });
        };
        event.writer = Some(PayloadStoreSite {
            launch_index: location.launch_index,
            warp_slot,
            instruction_index: usize::try_from(location.instruction_index)
                .map_err(|_| ReplayError::PayloadAdmissionWriterPlacementMissing { index })?,
        });
    }
    replay_kernel_with_schedule_and_l2_and_candidates_with_payload_events(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        None,
        &[],
        &[],
        None,
        &bound_events,
    )
}

/// Replay one launch while evaluating a bounded set of observational streaming
/// memory-service hypotheses at the exact issue timestamps. Candidates neither
/// delay issue nor add events; their per-SM diagnostics expose counterfactual
/// service busy time, queueing, and terminal completion. This research surface
/// is intentionally restricted to the full shared-L2 hierarchy mode.
pub fn replay_trace_kernel_with_memory_service_candidates(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_service_candidates: &[MemoryServiceCandidate],
) -> Result<KernelReplayOutcome, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    replay_kernel_with_schedule_and_l2_and_candidates(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        None,
        memory_service_candidates,
        &[],
        None,
    )
}

/// Replay one launch while evaluating bounded finite operand-admission queues
/// against exact per-instruction L2 hit/miss outcomes. The evaluator preserves
/// the observed issue and cache order and emits schema-9 diagnostics only; it
/// does not alter scheduler timing or hierarchy state.
pub fn replay_trace_kernel_with_memory_admission_candidates(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_admission_candidates: &[MemoryAdmissionCandidate],
) -> Result<KernelReplayOutcome, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    replay_kernel_with_schedule_and_l2_and_candidates(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        None,
        &[],
        memory_admission_candidates,
        None,
    )
}

/// Replay one launch with one feedback-coupled finite admission candidate.
/// Unlike schema-9 observation, this lane participates in warp readiness and
/// therefore may change issue and cache order. It remains explicit opt-in.
pub fn replay_trace_kernel_with_memory_admission_feedback(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
    memory_admission_feedback: MemoryAdmissionCandidate,
) -> Result<KernelReplayOutcome, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    replay_kernel_with_schedule_and_l2_and_candidates(
        td,
        loc,
        schedule,
        arch_card,
        params,
        mode,
        None,
        &[],
        &[],
        Some(memory_admission_feedback),
    )
}

/// Replay caller-supplied trace/schedule inputs for debugging only. The returned
/// diagnostics retain counters and directional timestamps but erase the full
/// replay type, so they cannot enter the legacy launch-set aggregation API.
#[cfg(feature = "campaign")]
pub fn replay_unadmitted_diagnostics(
    td: &TraceDir,
    loc: &KernelLoc<'_>,
    schedules: &BTreeMap<i32, FunctionSchedule>,
    arch_card: &ArchCard,
    params: &IssueModelParams,
    mode: ReplayMemMode,
) -> Result<UnadmittedReplayDiagnostics, ReplayError> {
    let schedule = lookup_schedule(schedules, loc)?;
    let outcome = replay_kernel_with_schedule(td, loc, schedule, arch_card, params, mode)?;
    Ok(UnadmittedReplayDiagnostics::from_outcome(outcome))
}

