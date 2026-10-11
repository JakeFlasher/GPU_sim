use super::*;

#[test]
fn instruction_stream_prefetches_the_next_line_and_merges_same_line_fetches() {
    let config = ArchId::Sm86
        .instruction_front_end()
        .require("instruction_front_end")
        .unwrap();
    let mut front_end = InstructionFrontEndState::new(config, 4).unwrap();

    // Demand line 0: 39 cycles to L1 + 4-cycle cold fill + 39 cycles
    // back to L0 + two fetch/decode stages.
    assert_eq!(front_end.fetch_instruction(0, 1, 0, 0).unwrap(), 84);
    front_end.prefetch_cycle(0).unwrap();

    // The next line is already in the stream buffer. A second instruction
    // in that line then merges through the pending L0 tag.
    assert_eq!(front_end.fetch_instruction(0, 1, 128, 1).unwrap(), 85);
    assert_eq!(front_end.fetch_instruction(0, 1, 144, 2).unwrap(), 86);
    assert_eq!(front_end.diagnostics.demand_l0_misses, 1);
    assert_eq!(front_end.diagnostics.demand_stream_buffer_hits, 1);
    assert_eq!(front_end.diagnostics.demand_l0_hits, 1);
    assert_eq!(front_end.diagnostics.demand_l1_misses, 1);
    assert_eq!(front_end.diagnostics.prefetch_l1_misses, 1);
}

#[test]
fn instruction_cache_miss_reservation_consumes_one_access_sequence() {
    let mut cache = SetAssociativeInstructionCache::new("L0I", 1, 1).unwrap();
    let key = InstructionLineKey {
        function_unique_id: 1,
        line: 0,
    };
    assert_eq!(cache.probe(key).unwrap(), None);
    cache.reserve(key, 4).unwrap();
    assert_eq!(cache.access_sequence, 1);
    assert_eq!(cache.probe(key).unwrap(), Some(4));
    assert_eq!(cache.access_sequence, 2);

    // A miss immediately before the final representable sequence must
    // still reserve successfully; probing an absent tag has no LRU touch.
    let mut boundary = SetAssociativeInstructionCache::new("L0I", 1, 1).unwrap();
    boundary.access_sequence = u64::MAX - 1;
    assert_eq!(boundary.probe(key).unwrap(), None);
    boundary.reserve(key, 8).unwrap();
    assert_eq!(boundary.access_sequence, u64::MAX);
}

#[test]
fn demanded_stream_line_wakes_prefetch_before_stalled_warp() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0); 3])],
    }];
    let config = ArchId::Sm86
        .instruction_front_end()
        .require("instruction_front_end")
        .unwrap();
    let mut p = params();
    p.instruction_front_end = Some(config);
    let mut snapshots = Vec::new();
    #[cfg(feature = "reference")]
    let mut advance_diagnostics = Vec::new();
    for dense in [false, true] {
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        core.dense_reference = dense;
        core.warps[0].eligible = 100;
        core.warps[0].front_end_fetch_cursor = 3;
        core.warps[0].instruction_buffer = (0..3)
            .map(|instruction| BufferedInstruction {
                instruction,
                ready_at: 0,
            })
            .collect();
        let front_end = core.instruction_front_end.as_mut().unwrap();
        let capacity = u64::from(config.stream_buffer_lines);
        front_end.streams[0].lines = (0..capacity)
            .map(|line| StreamBufferLine {
                key: InstructionLineKey {
                    function_unique_id: 1,
                    line,
                },
                ready_at: 10,
                demanded: line == 0,
            })
            .collect();
        front_end.streams[0].next_line = Some(InstructionLineKey {
            function_unique_id: 1,
            line: capacity,
        });
        assert!(!core.has_instruction_front_end_work().unwrap());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Bounded(50), &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: Some(50) }
        );
        assert!(requests.is_empty());
        assert_eq!(core.issued, 0);
        assert_eq!(core.t, 50, "a bounded idle return must park the SM clock");
        let front_end = core.instruction_front_end.as_ref().unwrap();
        assert_eq!(front_end.diagnostics.prefetch_requests, 1);
        assert_eq!(front_end.l1_request_ready, 11);
        let snapshot = format!("{front_end:?}");
        assert_eq!(
            core.advance(ReplayHorizon::Bounded(50), &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: Some(50) }
        );
        assert_eq!(core.t, 50);
        assert_eq!(core.issued, 0);
        assert_eq!(
            format!("{:?}", core.instruction_front_end.as_ref().unwrap()),
            snapshot,
            "revisiting a parked horizon must not replay front-end work"
        );
        snapshots.push(snapshot);
        #[cfg(feature = "reference")]
        advance_diagnostics.push(core.advance_diagnostics);
    }
    assert_eq!(snapshots[0], snapshots[1]);
    #[cfg(feature = "reference")]
    {
        assert!(advance_diagnostics[0].skipped_cycles > 0);
        assert_eq!(advance_diagnostics[1].skipped_cycles, 0);
        assert!(
            advance_diagnostics[1].local_cycle_steps > advance_diagnostics[0].local_cycle_steps,
            "dense mode must walk the idle cycles below the external horizon"
        );
    }
}

#[cfg(feature = "reference")]
#[test]
fn bounded_horizon_prunes_expired_memory_pipeline_releases() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut p = params();
    p.memory_issue_pipeline = Some(
        ArchId::Sm86
            .memory_issue_pipeline()
            .require("memory_issue_pipeline")
            .unwrap(),
    );
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    let pipeline = core
        .memory_issue_pipeline
        .as_mut()
        .expect("the fixture enables the memory issue pipeline");
    pipeline.queue_release_times[0].push_back(5);
    pipeline.diagnostics.instructions_admitted = 1;
    core.warps[0].eligible = 100;

    assert_eq!(
        core.advance(ReplayHorizon::Bounded(10), &mut Vec::new())
            .unwrap(),
        Advance::Idle { wake_at: Some(10) }
    );
    assert_eq!(core.t, 10);
    let typed = core
        .typed_memory_pipeline_state()
        .expect("typed memory pipeline remains installed");
    assert!(
        typed
            .queue_release_times
            .iter()
            .flatten()
            .all(|&release| release >= core.t)
    );
}

#[cfg(feature = "reference")]
#[test]
fn already_at_bounded_horizon_prunes_releases_at_that_cycle() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut p = params();
    p.memory_issue_pipeline = Some(
        ArchId::Sm86
            .memory_issue_pipeline()
            .require("memory_issue_pipeline")
            .unwrap(),
    );
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    let pipeline = core
        .memory_issue_pipeline
        .as_mut()
        .expect("the fixture enables the memory issue pipeline");
    pipeline.queue_release_times[0].push_back(10);
    pipeline.diagnostics.instructions_admitted = 1;
    core.t = 10;
    core.warps[0].eligible = 100;

    assert_eq!(
        core.advance(ReplayHorizon::Bounded(10), &mut Vec::new())
            .unwrap(),
        Advance::Idle { wake_at: Some(10) }
    );
    let typed = core
        .typed_memory_pipeline_state()
        .expect("typed memory pipeline remains installed");
    assert!(typed.queue_release_times.iter().all(Vec::is_empty));
}

#[test]
fn instruction_front_end_fetch_overflow_is_transactional() {
    let config = ArchId::Sm86
        .instruction_front_end()
        .require("instruction_front_end")
        .unwrap();
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "EXIT")),
        (128, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(128)])],
    }];
    let mut p = params();
    p.instruction_front_end = Some(config);
    p.l1_miss_latency = 4;
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    let sequence_before = {
        let front_end = core
            .instruction_front_end
            .as_mut()
            .expect("fixture enables the instruction front end");
        front_end.l1.access_sequence = u64::MAX - 1;
        front_end.l1.access_sequence
    };
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::CounterOverflow {
            counter: "instruction-cache access sequence"
        })
    ));
    assert!(out.is_empty());
    assert_eq!(core.warps[0].front_end_fetch_cursor, 0);
    assert_eq!(core.warps[1].front_end_fetch_cursor, 0);
    assert!(core.warps[0].instruction_buffer.is_empty());
    assert!(core.warps[1].instruction_buffer.is_empty());
    let front_end = core
        .instruction_front_end
        .as_ref()
        .expect("front end remains installed after failed cycle");
    assert_eq!(front_end.l1.access_sequence, sequence_before);
    assert_eq!(front_end.diagnostics.instructions_fetched, 0);
}

#[test]
fn active_instruction_front_end_delays_first_issue_without_changing_count() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut front_end_params = params();
    front_end_params.instruction_front_end = Some(
        ArchId::Sm86
            .instruction_front_end()
            .require("instruction_front_end")
            .unwrap(),
    );
    front_end_params.l1_miss_latency = 4;

    let result = simulate_sm(&blocks, &sched, &front_end_params, None).unwrap();
    assert_eq!(result.instructions_issued, 1);
    assert_eq!(result.cycles, 85);
    assert_eq!(result.readiness.limiting_cycles.instruction_front_end, 84);
    let diagnostics = result
        .readiness
        .instruction_front_end
        .as_deref()
        .expect("active model emits front-end diagnostics");
    assert_eq!(diagnostics.instructions_fetched, 1);
    assert_eq!(diagnostics.demand_l0_misses, 1);
    assert_eq!(diagnostics.stream_restarts, 1);
}

#[cfg(feature = "reference")]
#[test]
fn native_memory_pipeline_projection_accepts_maximum_shared_frontier() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDS R0, [R2]"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: (0..2)
            .map(|warp_slot| {
                warp(
                    warp_slot,
                    vec![ReplayInst {
                        lane_mask: u32::MAX,
                        pc: 0,
                        kind: AccessKind::Shared { wavefronts: 1 },
                    }],
                )
            })
            .collect(),
    }];
    let mut p = params();
    p.memory_issue_pipeline = Some(
        ArchId::Sm86
            .memory_issue_pipeline()
            .require("memory_issue_pipeline")
            .unwrap(),
    );
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    core.t = u64::MAX - 13;
    for state in &mut core.warps {
        state.current_since = core.t;
        state.eligible = core.t;
    }
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    assert!(requests.is_empty());
    let pipeline = core.typed_memory_pipeline_state().unwrap();
    // Both operands finish at MAX-4. The shared port accepts the two
    // SMSPs at MAX-4 and MAX-2, leaving a representable MAX frontier.
    assert_eq!(pipeline.instructions_admitted, 2);
    assert_eq!(
        pipeline.local_completion_ready,
        vec![u64::MAX, u64::MAX, 0, 0]
    );
    assert_eq!(pipeline.terminal_shared_accept_cycle, u64::MAX - 2);
    assert_eq!(pipeline.shared_accept_ready, u64::MAX);
    let snapshot = core.boundary_snapshot();
    let mut typed = sim_semantics::TypedCoreBoundaryState {
        sm_id: 0,
        cycle: snapshot.cycle,
        issued: snapshot.issued,
        resident_warps: snapshot.resident_warps,
        resident_shared_memory_bytes: snapshot.resident_shared_memory_bytes,
        unfinished_warps: snapshot.unfinished_warps,
        next_block: snapshot.next_block,
        hold: snapshot.hold,
        subcore_warps: snapshot.subcore_warps,
        outstanding_responses: snapshot.outstanding_responses,
        barrier_pipe_ready: snapshot.barrier_pipe_ready,
        tensor_pipe_ready: snapshot.tensor_pipe_ready,
        ldgsts_destination_pipe_ready: snapshot.ldgsts_destination_pipe_ready,
        warp_states: snapshot.warp_states,
        block_states: snapshot.block_states,
        l1: core.typed_l1_state(),
        front_end: core.typed_front_end_state(),
        memory_pipeline: Some(pipeline),
        scalar_pipeline: core.typed_scalar_pipeline_state(),
        local_completion_time: core.typed_local_completion_time(),
        unprojected_timing_state: Vec::new(),
        async_copy: core.typed_async_copy_states(),
        response_owners: core.typed_l2_request_owners(0, &[]),
    };
    assert_eq!(typed.validate(), Ok(()));
    // Unlike the validation-only bound, the shared port's next actual
    // timestamp must still fit in the native cycle domain.
    typed
        .memory_pipeline
        .as_mut()
        .unwrap()
        .terminal_shared_accept_cycle = u64::MAX;
    assert_eq!(
        typed.validate(),
        Err(sim_semantics::CoreStateError::CounterOverflow)
    );
}

#[test]
fn memory_issue_pipeline_reproduces_the_published_ampere_burst_table() {
    let config = ArchId::Sm86
        .memory_issue_pipeline()
        .require("memory_issue_pipeline")
        .unwrap();
    let expected = [
        vec![vec![2, 3, 4, 5, 6, 13, 17, 21]],
        vec![
            vec![2, 3, 4, 5, 6, 13, 17, 21],
            vec![2, 3, 4, 5, 6, 15, 19, 23],
        ],
        vec![
            vec![2, 3, 4, 5, 6, 13, 19, 25],
            vec![2, 3, 4, 5, 6, 15, 21, 27],
            vec![2, 3, 4, 5, 6, 17, 23, 29],
        ],
        vec![
            vec![2, 3, 4, 5, 6, 13, 21, 29],
            vec![2, 3, 4, 5, 6, 15, 23, 31],
            vec![2, 3, 4, 5, 6, 17, 25, 33],
            vec![2, 3, 4, 5, 6, 19, 27, 35],
        ],
    ];
    for active_subcores in 1..=SUB_CORES {
        let mut state = MemoryIssuePipelineState::new(config).unwrap();
        let mut observed = vec![Vec::new(); active_subcores];
        for now in 2..=64 {
            state.prune(now);
            let mut selected = [None; SUB_CORES];
            for subcore in 0..active_subcores {
                if observed[subcore].len() < 8 && state.admission_ready_at(subcore) <= now {
                    observed[subcore].push(now);
                    selected[subcore] = Some(MemoryPipelineSlot {
                        operand_ready_latency: 11,
                    });
                }
            }
            state = state.plan(selected, now).unwrap().next;
            if observed.iter().all(|issues| issues.len() == 8) {
                break;
            }
        }
        assert_eq!(observed, expected[active_subcores - 1]);
    }
}

#[test]
fn memory_pipeline_static_war_latencies_follow_the_published_table() {
    let config = ArchId::Sm86
        .memory_issue_pipeline()
        .require("memory_issue_pipeline")
        .unwrap();
    let latency = |text: &str| {
        memory_pipeline_slot(&slot(cb(1), OpClass::Other, text), config, false)
            .expect("published memory family")
            .operand_ready_latency
    };
    assert_eq!(latency("LDG.E R0, [R2.64]"), 11);
    assert_eq!(latency("LDG.E R0, [UR2]"), 9);
    assert_eq!(latency("STG.E.128 [R2.64], R4"), 20);
    assert_eq!(latency("STG.E.64 [UR2], R4"), 12);
    assert_eq!(latency("LDS R0, [R2]"), 9);
    assert_eq!(latency("STS.64 [R2], R4"), 14);
    assert_eq!(latency("LDGSTS.E.128 [R2], [R4.64]"), 13);
    assert!(
        memory_pipeline_slot(
            &slot(cb(1), OpClass::Other, "ATOMG.E.ADD [R2], R4"),
            config,
            false
        )
        .is_none()
    );
}

#[test]
fn constant_pressure_boundary_comes_from_memory_window_and_u16_barrier() {
    let six = schedule(
        (0..6)
            .map(|pc| {
                (
                    pc * 16,
                    slot(cb(1), OpClass::Other, "LDG.E.CONSTANT R0, [R2.64]"),
                )
            })
            .collect(),
    );
    assert_eq!(constant_stream_count(&six), 6);
    let six_slot = &six.by_pc[&0];
    assert!(!deep_constant_load(six_slot, 6, 5));

    let seven = schedule(
        (0..7)
            .map(|pc| {
                (
                    pc * 16,
                    slot(cb(1), OpClass::Other, "LDG.E.CONSTANT R0, [R2.64]"),
                )
            })
            .collect(),
    );
    assert_eq!(constant_stream_count(&seven), 7);
    assert!(deep_constant_load(&seven.by_pc[&0], 7, 5));

    let u16_short = slot(
        ControlBits::new(1, true, 2, 7, 0, 0).unwrap(),
        OpClass::Other,
        "LDG.E.U16.CONSTANT R0, [R2.64]",
    );
    let u16_long = slot(
        ControlBits::new(1, true, 3, 7, 0, 0).unwrap(),
        OpClass::Other,
        "LDG.E.U16.CONSTANT R0, [R2.64]",
    );
    assert!(!deep_constant_load(&u16_short, 6, 5));
    assert!(deep_constant_load(&u16_long, 6, 5));
}

#[test]
fn single_warp_stall_timing() {
    // Three instructions, stalls 1,1,1: issues at t=0,1,2 → cycles 3.
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "IADD3")),
        (16, slot(cb(1), OpClass::Other, "IADD3")),
        (32, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16), inst(32)])],
    }];
    let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(r.instructions_issued, 3);
    assert_eq!(r.cycles, 3);
    assert!(r.l1.is_none());
    assert_eq!(r.readiness.limiting_cycles.control_stall, 2);
    assert_eq!(r.readiness.constraint_exposure_cycles.control_stall, 2);
    assert_eq!(r.readiness.delayed_ready_issues, 0);
}

#[test]
fn phase_opcode_recognition_uses_the_static_sass_ascii_grammar() {
    let ascii_clock = slot(cb(1), OpClass::Other, "\t CS2R R4, SR_CLOCKLO \r\n");
    let wrong_source = slot(cb(1), OpClass::Other, "CS2R R4, SR_CLOCKLO64");
    assert!(is_clocklo_issue(&ascii_clock));
    assert!(!is_clocklo_issue(&wrong_source));

    let ascii_membar = slot(cb(1), OpClass::Other, "\tMEMBAR.SC.GPU ;");
    assert!(is_gpu_membar_issue(&ascii_membar));
    assert_eq!(
        ascii_membar.visibility_fence_semantics(),
        Some((
            sim_mem::VisibilityScope::Device,
            sim_mem::VisibilityOrder::SequentiallyConsistent,
        ))
    );
    assert_eq!(
        slot(cb(1), OpClass::Other, "MEMBAR.SC.SYS").visibility_fence_semantics(),
        Some((
            sim_mem::VisibilityScope::System,
            sim_mem::VisibilityOrder::SequentiallyConsistent,
        ))
    );
    assert_eq!(
        slot(cb(1), OpClass::Other, "MEMBAR.CTA").visibility_fence_semantics(),
        Some((
            sim_mem::VisibilityScope::Block,
            sim_mem::VisibilityOrder::SequentiallyConsistent,
        ))
    );
}

#[test]
fn clocklo_issue_interval_matches_the_bracketed_scheduler_region() {
    let sched = schedule(vec![
        (0, slot(cb(2), OpClass::Other, "CS2R R4, SR_CLOCKLO")),
        (16, slot(cb(3), OpClass::Other, "CS2R R8, SRZ")),
        (32, slot(cb(1), OpClass::Other, "CS2R R6, SR_CLOCKLO")),
        (48, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16), inst(32), inst(48)])],
    }];
    let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(result.readiness.clock_issue.reads_issued, 2);
    assert_eq!(result.readiness.clock_issue.warps_with_interval, 1);
    assert_eq!(
        result
            .readiness
            .clock_issue
            .maximum_first_to_last_issue_cycles,
        5
    );
    assert_eq!(result.readiness.schema_version(), 5);
}

#[test]
fn same_bank_hadd2_defers_only_the_coincident_l1tex_writeback() {
    let noncollision = result_priority_case(11, 22);
    let collision = result_priority_case(11, 23);
    let opposite = result_priority_case(14, 23);

    assert_eq!(noncollision.cycles, 279);
    assert_eq!(collision.cycles, 280);
    assert_eq!(opposite.cycles, 279);
    assert!(noncollision.readiness.result_writeback.is_empty());
    assert!(opposite.readiness.result_writeback.is_empty());
    assert_eq!(
        collision.readiness.result_writeback,
        ResultWritebackDiagnostics {
            fixed_over_l1tex_collisions: 1,
            l1tex_deferral_cycles: 1,
        }
    );
    assert_eq!(collision.readiness.global_memory_issue.load_instructions, 1);
    assert_eq!(collision.readiness.schema_version(), 6);
}
