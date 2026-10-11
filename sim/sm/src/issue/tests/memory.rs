use super::*;

#[test]
fn predicated_off_memory_creates_no_dependency_producer() {
    let mut producer = cb(1);
    producer.read_barrier = Some(barrier(0));
    producer.write_barrier = Some(barrier(1));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000011);
    let sched = schedule(vec![
        (0, slot(producer, OpClass::Other, "STG.E.STRONG.SM")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::PredicatedOffMemory,
                },
                inst(16),
            ],
            sectors: Vec::new(),
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    assert!(requests.is_empty());
    let result = core.result();
    assert_eq!(result.cycles, 2);
    assert!(result.unmodeled_tails.is_empty());
    assert!(result.readiness.schema_version() < 8);
    assert_eq!(result.readiness.global_memory_issue.store_instructions, 0);
    assert_eq!(
        result
            .readiness
            .global_memory_issue
            .store_operand_read_barrier_instructions,
        0
    );
    assert_eq!(result.readiness.limiting_cycles.dependence_counter_read, 0);
    assert_eq!(
        result
            .readiness
            .limiting_cycles
            .dependence_counter_global_store_operand_read,
        0
    );
}

#[test]
fn pending_l1tex_writeback_keeps_origin_when_read_barrier_reuses_counter() {
    let mut producer = cb(1);
    producer.write_barrier = Some(barrier(0));
    producer.read_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000001);
    let sched = schedule(vec![
        (0, slot(producer, OpClass::Other, "LDG.E.CA")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                },
                inst(16),
            ],
            sectors: vec![0],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(requests.len(), 1);
    let mut sink = Vec::new();
    core.on_response(requests[0].reply, 300, &mut sink).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
        Advance::Done
    );
    let result = core.result();
    assert_eq!(result.cycles, 301);
    assert_eq!(result.readiness.schema_version(), 8);
    assert_eq!(
        result
            .readiness
            .global_memory_issue
            .load_operand_read_barrier_instructions,
        1
    );
    assert_eq!(
        result.readiness.limiting_cycles.dependence_counter_l1tex,
        300
    );
    assert_eq!(
        result
            .readiness
            .limiting_cycles
            .dependence_counter_global_load_operand_read,
        0
    );

    // Synthetic latency inversion: if the response wins before the fixed
    // operand-read clear, it must not steal the later read attribution.
    let mut early_params = params();
    early_params.l1_hit_latency = 1;
    early_params.l1_miss_latency = 2;
    let early_l1 = L1Config {
        hit_latency: 1,
        miss_latency: 2,
        ..l1_config()
    };
    let mut early = SmCore::new(
        &blocks,
        &sched,
        &early_params,
        SmMemMode::L1Shared(early_l1),
    )
    .unwrap();
    let mut early_requests = Vec::new();
    assert_eq!(
        early
            .advance(ReplayHorizon::Unbounded, &mut early_requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    let mut early_sink = Vec::new();
    early
        .on_response(early_requests[0].reply, 2, &mut early_sink)
        .unwrap();
    assert_eq!(
        early
            .advance(ReplayHorizon::Unbounded, &mut early_sink)
            .unwrap(),
        Advance::Done
    );
    let early_result = early.result();
    assert_eq!(early_result.cycles, 9);
    assert_eq!(
        early_result
            .readiness
            .limiting_cycles
            .dependence_counter_global_load_operand_read,
        8
    );
    assert_eq!(
        early_result
            .readiness
            .limiting_cycles
            .dependence_counter_l1tex,
        0
    );
}

#[test]
fn pending_l1tex_writeback_beats_later_fast_producer_origin() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let mut fixed = cb(1);
    fixed.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000001);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E.CA")),
        (16, slot(fixed, OpClass::Other, "IADD3")),
        (32, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                },
                inst(16),
                inst(32),
            ],
            sectors: vec![0],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(requests.len(), 1);
    let mut sink = Vec::new();
    core.on_response(requests[0].reply, 300, &mut sink).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
        Advance::Done
    );
    let result = core.result();
    assert_eq!(result.cycles, 301);
    assert_eq!(
        result.readiness.limiting_cycles.dependence_counter_l1tex,
        299
    );
    assert_eq!(
        result
            .readiness
            .limiting_cycles
            .dependence_counter_non_memory_write,
        0
    );
}

#[test]
fn shared_cacheable_async_copy_completes_all_same_line_fill_waiters() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (16, async_commit_slot()),
        (32, async_wait_slot(0)),
        (48, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![async_copy_inst(0, 0, 2), inst(16), inst(32), inst(48)],
            sectors: vec![40, 41],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(requests.len(), 1, "same-line sectors fold into one fill");
    assert_eq!(requests[0].source, L2Source::L1Fill);
    assert_eq!(requests[0].sector_mask, 0b0011);
    let ReplyTo::L1Fill { fill, .. } = requests[0].reply else {
        panic!("cacheable LDGSTS source must ride an L1 fill")
    };
    let group = core.async_copies[0]
        .state
        .as_deref()
        .unwrap()
        .committed_groups[0];
    assert_eq!(group.pending_sources, 2);
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode")
    };
    assert!(st.fill_waiters.contains_key(&fill));

    let mut emitted = Vec::new();
    core.on_response(requests[0].reply, 300, &mut emitted)
        .unwrap();
    assert!(emitted.is_empty());
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0]
            .pending_sources,
        0
    );
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0]
            .completion_time,
        301
    );
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut emitted)
            .unwrap(),
        Advance::Done
    );
    assert_eq!(core.result().cycles, 303);
    let counters = core.l1_counters().unwrap();
    assert_eq!(counters.misses, 2);
    assert_eq!(counters.merges, 1);
    assert_eq!(counters.bypass_sectors, 0);
}

#[test]
fn mixed_constant_and_ordinary_fill_waiters_keep_return_latency_local() {
    let mut ordinary = cb(1);
    ordinary.write_barrier = Some(barrier(0));
    let mut constant = cb(1);
    constant.write_barrier = Some(barrier(0));
    // Seven static constant streams cross the SM86 pressure threshold,
    // making the constant load at pc=16 consume the deep return path.
    let sched = schedule(vec![
        (0, slot(ordinary, OpClass::Other, "LDG.E.CA")),
        (16, slot(constant, OpClass::Other, "LDG.E.CONSTANT")),
        (32, slot(cb(1), OpClass::Other, "EXIT")),
        (48, slot(cb(1), OpClass::Other, "LDG.E.CONSTANT")),
        (64, slot(cb(1), OpClass::Other, "LDG.E.CONSTANT")),
        (80, slot(cb(1), OpClass::Other, "LDG.E.CONSTANT")),
        (96, slot(cb(1), OpClass::Other, "LDG.E.CONSTANT")),
        (112, slot(cb(1), OpClass::Other, "LDG.E.CONSTANT")),
        (128, slot(cb(1), OpClass::Other, "LDG.E.CONSTANT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        lane_mask: u32::MAX,
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(32),
                ],
                sectors: vec![0],
            },
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 1,
                insts: vec![
                    ReplayInst {
                        lane_mask: u32::MAX,
                        pc: 16,
                        kind: eligible(0, 1),
                    },
                    inst(32),
                ],
                sectors: vec![0],
            },
        ],
    }];
    let mut p = params();
    p.memory_issue_pipeline = Some(
        ArchId::Sm86
            .memory_issue_pipeline()
            .require("memory_issue_pipeline")
            .unwrap(),
    );
    p.constant_memory_return_service = Some(ConstantMemoryReturnService {
        additional_cycles: 8,
    });
    let mut core = shared_core(&blocks, &sched, &p);
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    assert_eq!(requests.len(), 1, "same-line loads share one physical fill");
    let reply = requests[0].reply;
    core.on_response(reply, 244, &mut Vec::new()).unwrap();
    assert_eq!(core.warps[0].barrier_clear[0], 244);
    assert_eq!(core.warps[1].barrier_clear[0], 252);
}

#[test]
fn shared_cacheable_async_copy_l1_hit_completes_without_a_request() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (16, async_commit_slot()),
        (32, async_wait_slot(0)),
        (48, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![async_copy_inst(0, 0, 1), inst(16), inst(32), inst(48)],
            sectors: vec![40],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let MemModel::L1Shared(st) = &mut core.mem else {
        panic!("expected shared mode")
    };
    let EvAccess::MissRequest { fill } = st
        .l1
        .access_eligible_ev(MemoryOrigin::Global, 40, 0)
        .unwrap()
    else {
        panic!("expected cold prefill miss")
    };
    st.l1.complete_fill(fill).unwrap();
    let hits_before = st.l1.counters.hits;
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    assert!(requests.is_empty());
    assert_eq!(
        core.result().cycles,
        37,
        "wait must issue at hit-ready t=34 plus one payload cycle"
    );
    assert_eq!(core.l1_counters().unwrap().hits, hits_before + 1);
    assert_eq!(core.outstanding_responses(), 0);
    assert!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups
            .is_empty()
    );
}

#[test]
fn shared_cacheable_async_copy_retry_retains_group_identity() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (16, async_commit_slot()),
        (32, async_wait_slot(0)),
        (48, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![async_copy_inst(0, 0, 2), inst(16), inst(32), inst(48)],
            sectors: vec![0, 4],
        }],
    }];
    let mut cfg = l1_config();
    cfg.mshr_count = Some(1);
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::L1Shared(cfg)).unwrap();
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(requests.len(), 1);
    let group = core.async_copies[0]
        .state
        .as_deref()
        .unwrap()
        .committed_groups[0]
        .id;
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0]
            .pending_sources,
        2
    );
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode")
    };
    assert_eq!(st.retry.len(), 1);
    assert_eq!(st.retry[0].mshr_floor, NonZeroU32::new(1));
    assert_eq!(
        st.retry[0].target.and_then(FillTarget::async_group),
        Some(group)
    );

    let mut retried = Vec::new();
    core.on_response(requests[0].reply, 244, &mut retried)
        .unwrap();
    assert_eq!(retried.len(), 1, "freed MSHR must issue the parked line");
    assert!(matches!(retried[0].reply, ReplyTo::L1Fill { .. }));
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0]
            .pending_sources,
        1
    );
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode")
    };
    assert!(st.retry.is_empty());

    let mut emitted = Vec::new();
    core.on_response(retried[0].reply, 500, &mut emitted)
        .unwrap();
    assert!(emitted.is_empty());
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut emitted)
            .unwrap(),
        Advance::Done
    );
    assert_eq!(core.result().cycles, 503);
    assert_eq!(core.outstanding_responses(), 0);
}

#[test]
fn shared_cacheable_floor_survives_retry_and_stays_access_local() {
    let mut p = params();
    let floor = NonZeroU32::new(2).unwrap();
    p.ldgsts_l1_mshr_floor = floor;
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (16, async_commit_slot()),
        (32, async_wait_slot(0)),
        (48, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![async_copy_inst(0, 0, 3), inst(16), inst(32), inst(48)],
            sectors: vec![0, 4, 8],
        }],
    }];
    let mut cfg = l1_config();
    cfg.mshr_count = Some(1);
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::L1Shared(cfg)).unwrap();
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(
        requests.len(),
        2,
        "the floor admits two distinct line fills"
    );
    #[cfg(feature = "reference")]
    assert_eq!(core.typed_l1_state().unwrap().validate(), Ok(()));
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode")
    };
    assert_eq!(st.retry.len(), 1);
    assert_eq!(st.retry[0].mshr_floor, Some(floor));

    let mut retried = Vec::new();
    core.on_response(requests[0].reply, 244, &mut retried)
        .unwrap();
    assert_eq!(
        retried.len(),
        1,
        "the retained floor must admit the retry while one fill remains"
    );
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode")
    };
    assert!(st.retry.is_empty());

    let ordinary_sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
    let ordinary_blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: eligible(0, 2),
            }],
            sectors: vec![0, 4],
        }],
    }];
    let mut ordinary_cfg = l1_config();
    ordinary_cfg.mshr_count = Some(1);
    let mut ordinary = SmCore::new(
        &ordinary_blocks,
        &ordinary_sched,
        &p,
        SmMemMode::L1Shared(ordinary_cfg),
    )
    .unwrap();
    let mut ordinary_requests = Vec::new();
    assert_eq!(
        ordinary
            .advance(ReplayHorizon::Unbounded, &mut ordinary_requests)
            .unwrap(),
        Advance::Done
    );
    assert_eq!(ordinary_requests.len(), 1);
    let MemModel::L1Shared(st) = &ordinary.mem else {
        panic!("expected shared mode")
    };
    assert_eq!(st.retry.len(), 1);
    assert_eq!(st.retry[0].mshr_floor, None);
}

#[test]
fn cacheable_async_fill_underflow_preserves_the_response_transaction() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![async_copy_inst(0, 0, 1)],
            sectors: vec![40],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    let reply = requests[0].reply;
    let ReplyTo::L1Fill { fill, .. } = reply else {
        panic!("expected L1 fill response")
    };
    let group = core.async_copies[0]
        .state
        .as_deref_mut()
        .unwrap()
        .open_group
        .as_mut()
        .unwrap();
    let group_id = group.id;
    group.pending_sources = 0;
    let before_l1 = *core.l1_counters().unwrap();
    let mut emitted = Vec::new();

    assert!(matches!(
        core.on_response(reply, 244, &mut emitted),
        Err(IssueError::AsyncCopyPendingUnderflow {
            warp: 0,
            group,
            current: 0,
            remove: 1,
        }) if group == group_id
    ));
    assert!(emitted.is_empty());
    assert_eq!(core.outstanding_responses(), 1);
    assert_eq!(*core.l1_counters().unwrap(), before_l1);
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode")
    };
    assert!(st.l1.is_fill_in_flight(fill));
    assert!(st.fill_waiters.contains_key(&fill));
}

#[test]
fn cacheable_async_payload_floor_overflow_preserves_response_state() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![async_copy_inst(0, 0, 1)],
            sectors: vec![40],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    let reply = requests[0].reply;
    let ReplyTo::L1Fill { fill, .. } = reply else {
        panic!("expected L1 fill response")
    };
    let before_group = core.async_copies[0].state.as_deref().unwrap().open_group;
    let before_l1 = *core.l1_counters().unwrap();
    let mut emitted = Vec::new();

    assert!(matches!(
        core.on_response(reply, u64::MAX, &mut emitted),
        Err(IssueError::TimeOverflow {
            operation: "LDGSTS payload completion floor",
            base: u64::MAX,
            delta: 1,
        })
    ));
    assert!(emitted.is_empty());
    assert_eq!(core.outstanding_responses(), 1);
    assert_eq!(
        core.async_copies[0].state.as_deref().unwrap().open_group,
        before_group
    );
    assert_eq!(*core.l1_counters().unwrap(), before_l1);
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode")
    };
    assert!(st.l1.is_fill_in_flight(fill));
    assert!(st.fill_waiters.contains_key(&fill));
}

#[test]
fn shared_async_copy_waits_for_all_sector_responses_out_of_order() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDGSTS.E.BYPASS")),
        (16, async_commit_slot()),
        (32, async_wait_slot(0)),
        (48, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                bypass_async_copy_inst(0, 0, 2),
                inst(16),
                inst(32),
                inst(48),
            ],
            sectors: vec![40, 41],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|request| request.has_valid_contract()));
    let group = match requests[0].reply {
        ReplyTo::AsyncCopy {
            warp: 0,
            group,
            barrier: None,
            ..
        } => group,
        reply => panic!("unexpected first async-copy reply {reply:?}"),
    };
    assert!(matches!(
        requests[1].reply,
        ReplyTo::AsyncCopy {
            warp: 0,
            group: other,
            barrier: None,
            ..
        } if other == group
    ));
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups
            .len(),
        1
    );
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0]
            .pending_sources,
        2
    );

    let mut emitted = Vec::new();
    core.on_response(requests[1].reply, 300, &mut emitted)
        .unwrap();
    assert!(emitted.is_empty());
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0]
            .pending_sources,
        1
    );
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0]
            .completion_time,
        302
    );
    assert!(matches!(
        core.preflight_response(requests[1].reply, 300),
        Err(IssueError::ResponseTokenNotOutstanding { .. })
    ));
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut emitted)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );

    core.on_response(requests[0].reply, 400, &mut emitted)
        .unwrap();
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0]
            .pending_sources,
        0
    );
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0]
            .completion_time,
        402
    );
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut emitted)
            .unwrap(),
        Advance::Done
    );
    assert_eq!(core.result().cycles, 404);
    assert!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups
            .is_empty()
    );
    assert_eq!(core.outstanding_responses(), 0);
    assert!(matches!(
        core.on_response(requests[0].reply, 404, &mut emitted),
        Err(IssueError::ResponseTokenNotOutstanding { reply })
            if reply == requests[0].reply
    ));
}

#[test]
fn async_copy_response_can_complete_an_open_group_before_commit() {
    let sched = schedule(vec![(0, async_commit_slot())]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut core = shared_core(&blocks, &sched, &params());

    let p = params();
    let group = core.async_copies[0]
        .add_pending_async_copy_sources(0, 2, 0, 0, None, &p)
        .unwrap();
    core.async_copies[0]
        .complete_async_copy_sources(0, group, None, 2, 91, &p)
        .unwrap();
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .open_group
            .unwrap()
            .completion_time,
        91
    );
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .open_group
            .unwrap()
            .pending_sources,
        0
    );

    assert!(core.async_copies[0].commit_async_copy_group(100));
    assert!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .open_group
            .is_none()
    );
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups
            .len(),
        1
    );
    assert!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups[0]
            .is_complete_at(91)
    );
}

#[test]
fn async_copy_response_underflow_is_transactional() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E.BYPASS"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![bypass_async_copy_inst(0, 0, 1)],
            sectors: vec![40],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    let reply = requests[0].reply;
    let ReplyTo::AsyncCopy { group, .. } = reply else {
        panic!("expected async-copy response")
    };
    core.async_copies[0]
        .state
        .as_deref_mut()
        .unwrap()
        .open_group
        .as_mut()
        .expect("issued copy owns an open group")
        .pending_sources = 0;
    let before_group = core.async_copies[0].state.as_deref().unwrap().open_group;
    let mut emitted = Vec::new();

    assert!(matches!(
        core.on_response(reply, 244, &mut emitted),
        Err(IssueError::AsyncCopyPendingUnderflow {
            warp: 0,
            group: got_group,
            current: 0,
            remove: 1,
        }) if got_group == group
    ));
    assert_eq!(
        core.async_copies[0].state.as_deref().unwrap().open_group,
        before_group
    );
    assert_eq!(core.outstanding_responses(), 1);
    assert!(emitted.is_empty());
}

#[test]
fn async_copy_response_missing_group_is_transactional() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E.BYPASS"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![bypass_async_copy_inst(0, 0, 1)],
            sectors: vec![40],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    let reply = requests[0].reply;
    let ReplyTo::AsyncCopy { group, .. } = reply else {
        panic!("expected async-copy response")
    };
    core.async_copies[0]
        .state
        .as_deref_mut()
        .unwrap()
        .open_group = None;
    let mut emitted = Vec::new();

    assert!(matches!(
        core.on_response(reply, 244, &mut emitted),
        Err(IssueError::AsyncCopyGroupNotFound {
            warp: 0,
            group: got_group,
        }) if got_group == group
    ));
    assert_eq!(core.outstanding_responses(), 1);
    assert!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .open_group
            .is_none()
    );
    assert!(emitted.is_empty());
}

#[test]
fn issue_count_overflow_rejects_the_whole_cycle_before_mutation() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    core.issued = u64::MAX;
    let before_unfinished = core.unfinished_warps;
    let before_resident = core.resident_warps;
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::CounterOverflow {
            counter: "issued instruction"
        })
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), u64::MAX);
    assert_eq!(core.warps[0].next, 0);
    assert!(!core.warps[0].done);
    assert_eq!(core.unfinished_warps, before_unfinished);
    assert_eq!(core.resident_warps, before_resident);
}

#[test]
fn missing_resident_slot_rejects_retirement_before_mutation() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 11,
        warps: vec![warp(2, vec![inst(0)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    assert!(core.resident_warp_slots.remove(&2));
    let before_unfinished = core.unfinished_warps;
    let before_resident = core.resident_warps;
    let before_subcore = core.subcore_warps[2].clone();
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::ResidentWarpSlotMissing {
            launch_index: 11,
            warp_slot: 2,
        })
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert!(!core.warps[0].done);
    assert_eq!(core.unfinished_warps, before_unfinished);
    assert_eq!(core.resident_warps, before_resident);
    assert_eq!(core.subcore_warps[2], before_subcore);
}

#[test]
fn pending_overflow_rejects_issue_before_cache_or_request_mutation() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: eligible(0, 1),
            }],
            sectors: vec![0],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    core.warps[0].barrier_pending[0] = u32::MAX;
    let before_l1 = *core.l1_counters().unwrap();
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::PendingBarrierOverflow {
            warp: 0,
            barrier: 0,
            current: u32::MAX,
            add: 1,
        })
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert_eq!(core.warps[0].barrier_pending[0], u32::MAX);
    assert_eq!(core.outstanding_responses(), 0);
    assert_eq!(*core.l1_counters().unwrap(), before_l1);
}

#[test]
fn warm_hit_accepts_maximum_pending_count_without_a_new_unit() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: eligible(0, 1),
            }],
            sectors: vec![0],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let MemModel::L1Shared(st) = &mut core.mem else {
        panic!("expected shared mode");
    };
    let EvAccess::MissRequest { fill } = st
        .l1
        .access_eligible_ev(MemoryOrigin::Global, 0, 0)
        .unwrap()
    else {
        panic!("expected cold prefill miss");
    };
    st.l1.complete_fill(fill).unwrap();
    core.warps[0].barrier_pending[0] = u32::MAX;
    let before = *core.l1_counters().unwrap();
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 1);
    assert_eq!(core.warps[0].next, 1);
    assert_eq!(core.warps[0].barrier_pending[0], u32::MAX);
    let after = *core.l1_counters().unwrap();
    assert_eq!(after.hits, before.hits + 1);
    assert_eq!(after.misses, before.misses);
    assert_eq!(core.outstanding_responses(), 0);
}

#[test]
fn multi_subcore_l1_overflow_rejects_the_whole_issue_cycle() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "STG.E"))]);
    let store = || ReplayInst {
        lane_mask: u32::MAX,
        pc: 0,
        kind: AccessKind::GlobalStore { off: 0, len: 1 },
    };
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![store()],
                sectors: vec![0],
            },
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 1,
                insts: vec![store()],
                sectors: vec![4],
            },
        ],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let MemModel::L1Shared(st) = &mut core.mem else {
        panic!("expected shared mode");
    };
    st.l1.counters.store_sectors = u64::MAX - 1;
    let before = *core.l1_counters().unwrap();
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::L1Access(L1AccessError::Counter(
            L1CounterError::Overflow {
                field: L1CounterField::StoreSectors,
            }
        )))
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert_eq!(core.warps[1].next, 0);
    assert_eq!(*core.l1_counters().unwrap(), before);
}

#[test]
fn constant_l1_retry_time_error_rejects_all_subcores_before_mutation() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            },
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 1,
                insts: vec![ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![8],
            },
        ],
    }];
    let mut p = params();
    p.l1_hit_latency = 2;
    p.l1_miss_latency = 10;
    let cfg = p.l1_config(8, 1);
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::L1Const(cfg)).unwrap();
    let MemModel::L1Const(l1) = &mut core.mem else {
        panic!("expected constant L1 mode");
    };
    assert!(matches!(
        l1.access_eligible(MemoryOrigin::Global, 0, u64::MAX - 15).unwrap(),
        (AccessOutcome::MissLineAlloc, ready) if ready == u64::MAX - 5
    ));
    core.t = u64::MAX - 14;
    let before_l1 = *core.l1_counters().unwrap();
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::L1Access(L1AccessError::TimeOverflow {
            operation: "miss visibility",
            base,
            delta: 10,
        })) if base == u64::MAX - 5
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert_eq!(core.warps[1].next, 0);
    assert_eq!(*core.l1_counters().unwrap(), before_l1);
    let MemModel::L1Const(l1) = &core.mem else {
        panic!("expected constant L1 mode");
    };
    assert!(l1.is_line_fill_in_flight(0));
}

#[test]
fn constant_l1_retry_payload_overflow_rejects_issue_before_cache_mutation() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: AccessKind::GlobalAsyncCopy {
                    off: 0,
                    len: 1,
                    completion_work: 128,
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                },
            }],
            sectors: vec![8],
        }],
    }];
    let mut p = params();
    p.l1_hit_latency = 2;
    p.l1_miss_latency = 10;
    let cfg = p.l1_config(8, 1);
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::L1Const(cfg)).unwrap();
    let MemModel::L1Const(l1) = &mut core.mem else {
        panic!("expected constant L1 mode");
    };
    assert!(matches!(
        l1.access_eligible(MemoryOrigin::Global, 0, u64::MAX - 21)
            .unwrap(),
        (AccessOutcome::MissLineAlloc, ready) if ready == u64::MAX - 11
    ));
    core.t = u64::MAX - 20;
    let before_l1 = *core.l1_counters().unwrap();
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::TimeOverflow {
            operation: "LDGSTS payload completion floor",
            base,
            delta: 2,
        }) if base == u64::MAX - 1
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert!(core.async_copies[0].state.is_none());
    assert_eq!(*core.l1_counters().unwrap(), before_l1);
    let MemModel::L1Const(l1) = &core.mem else {
        panic!("expected constant L1 mode");
    };
    assert!(l1.is_line_fill_in_flight(0));
}

#[test]
fn same_cycle_fill_chain_overflow_rejects_all_subcores_before_mutation() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![0],
            },
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 1,
                insts: vec![ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                }],
                sectors: vec![8],
            },
        ],
    }];
    let mut p = params();
    p.l1_hit_latency = 2;
    p.l1_miss_latency = 10;
    let cfg = p.l1_config(8, 1);
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::L1Const(cfg)).unwrap();
    core.t = u64::MAX - 15;
    let before_l1 = *core.l1_counters().unwrap();
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::L1Access(L1AccessError::TimeOverflow {
            operation: "miss visibility",
            base,
            delta: 10,
        })) if base == u64::MAX - 5
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert_eq!(core.warps[1].next, 0);
    assert_eq!(*core.l1_counters().unwrap(), before_l1);
    let MemModel::L1Const(l1) = &core.mem else {
        panic!("expected constant L1 mode");
    };
    assert!(!l1.is_line_fill_in_flight(0));
    assert!(!l1.is_line_fill_in_flight(2));
}

#[test]
fn response_tokens_reject_cross_core_and_stale_replay_swaps() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: eligible(0, 1),
            }],
            sectors: vec![0],
        }],
    }];
    let mut first = shared_core(&blocks, &sched, &params());
    let mut second = shared_core(&blocks, &sched, &params());
    let mut first_requests = Vec::new();
    let mut second_requests = Vec::new();
    first
        .advance(ReplayHorizon::Unbounded, &mut first_requests)
        .unwrap();
    second
        .advance(ReplayHorizon::Unbounded, &mut second_requests)
        .unwrap();
    let first_reply = first_requests[0].reply;
    let second_reply = second_requests[0].reply;
    assert_ne!(first_reply, second_reply);
    let second_before = *second.l1_counters().unwrap();
    let mut emitted = Vec::new();
    assert!(matches!(
        second.on_response(first_reply, 244, &mut emitted),
        Err(IssueError::ResponseTokenNotOutstanding { reply }) if reply == first_reply
    ));
    assert!(emitted.is_empty());
    assert_eq!(second.outstanding_responses(), 1);
    assert_eq!(*second.l1_counters().unwrap(), second_before);

    drop(first);
    let mut next_generation = shared_core(&blocks, &sched, &params());
    let mut next_requests = Vec::new();
    next_generation
        .advance(ReplayHorizon::Unbounded, &mut next_requests)
        .unwrap();
    assert_ne!(first_reply, next_requests[0].reply);
    assert!(matches!(
        next_generation.on_response(first_reply, 244, &mut emitted),
        Err(IssueError::ResponseTokenNotOutstanding { reply }) if reply == first_reply
    ));
    assert_eq!(next_generation.outstanding_responses(), 1);
}

#[test]
fn response_to_non_shared_mode_is_typed_and_nonmutating() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert!(matches!(
        core.on_response(ReplyTo::None, 0, &mut out),
        Err(IssueError::ResponseModeMismatch { mode: "flat" })
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
}

#[test]
fn fill_waiter_underflow_rejects_response_before_any_mutation() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(1);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                },
                inst(16),
            ],
            sectors: vec![0],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    let reply = requests[0].reply;
    let ReplyTo::L1Fill { fill, .. } = reply else {
        panic!("expected fill response");
    };
    core.warps[0].barrier_pending[0] = 0;
    let before_l1 = *core.l1_counters().unwrap();
    let mut emitted = Vec::new();

    assert!(matches!(
        core.on_response(reply, 244, &mut emitted),
        Err(IssueError::PendingBarrierUnderflow {
            warp: 0,
            barrier: 0,
            current: 0,
            remove: 1,
        })
    ));
    assert!(emitted.is_empty());
    assert_eq!(core.outstanding_responses(), 1);
    assert_eq!(*core.l1_counters().unwrap(), before_l1);
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode");
    };
    assert!(st.l1.is_fill_in_flight(fill));
    assert!(st.fill_waiters.contains_key(&fill));
    if let FillKey::Line { line, .. } = fill {
        assert!(st.fill_masks.contains_key(&line));
    }
}

#[test]
fn retry_l1_overflow_preserves_the_entire_response_transaction() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(1);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                },
                inst(16),
            ],
            sectors: vec![0, 8],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    let reply = requests[0].reply;
    let ReplyTo::L1Fill { fill, .. } = reply else {
        panic!("expected fill response");
    };
    core.warps[0].barrier_pending[0] = 2;
    let MemModel::L1Shared(st) = &mut core.mem else {
        panic!("expected shared mode");
    };
    let lookup_batch = st.l1.begin_lookup_batch();
    st.retry.push_back(Retry {
        warp: 0,
        instruction: 0,
        origin: MemoryOrigin::Global,
        constant_load: false,
        constant_deep: false,
        target: Some(FillTarget::Barrier(barrier(0))),
        mshr_floor: None,
        lookup_batch,
        lo: 1,
        hi: 2,
    });
    st.l1.counters.misses = u64::MAX;
    let before_l1 = *core.l1_counters().unwrap();
    let mut emitted = Vec::new();

    assert!(matches!(
        core.on_response(reply, 244, &mut emitted),
        Err(IssueError::L1Access(L1AccessError::Counter(
            L1CounterError::Overflow {
                field: L1CounterField::Misses,
            }
        )))
    ));
    assert!(emitted.is_empty());
    assert_eq!(core.outstanding_responses(), 1);
    assert_eq!(core.warps[0].barrier_pending[0], 2);
    assert_eq!(*core.l1_counters().unwrap(), before_l1);
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode");
    };
    assert!(st.l1.is_fill_in_flight(fill));
    assert!(st.fill_waiters.contains_key(&fill));
    assert_eq!(st.retry.len(), 1);
    let retry = st.retry.front().unwrap();
    assert_eq!(retry.warp, 0);
    assert_eq!(retry.lo, 1);
    assert_eq!(retry.hi, 2);
    if let FillKey::Line { line, .. } = fill {
        assert!(st.fill_masks.contains_key(&line));
    }
}

#[test]
fn deep_constant_retry_overflow_is_preflighted_before_response_mutation() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(1);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E.CONSTANT")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                },
                inst(16),
            ],
            sectors: vec![0, 0],
        }],
    }];
    let mut p = params();
    p.l1_hit_latency = 1;
    p.l1_miss_latency = 2;
    p.constant_memory_return_service = Some(ConstantMemoryReturnService {
        additional_cycles: 8,
    });
    let l1 = L1Config {
        hit_latency: 1,
        miss_latency: 2,
        ..l1_config()
    };
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::L1Shared(l1)).unwrap();
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    let reply = requests[0].reply;
    core.warps[0].barrier_pending[0] = 2;
    let MemModel::L1Shared(st) = &mut core.mem else {
        panic!("expected shared mode");
    };
    let lookup_batch = st.l1.begin_lookup_batch();
    st.retry.push_back(Retry {
        warp: 0,
        instruction: 0,
        origin: MemoryOrigin::Global,
        constant_load: true,
        constant_deep: true,
        target: Some(FillTarget::Barrier(barrier(0))),
        mshr_floor: None,
        lookup_batch,
        lo: 1,
        hi: 2,
    });
    let before_l1 = *core.l1_counters().unwrap();
    let before_pending = core.warps[0].barrier_pending[0];
    let mut emitted = Vec::new();
    let now = u64::MAX - 2;
    let error = core.on_response(reply, now, &mut emitted).unwrap_err();
    assert!(matches!(
        error,
        IssueError::TimeOverflow {
            operation: "constant retry L1-hit return latency",
            ..
        }
    ));
    assert!(emitted.is_empty());
    assert_eq!(core.outstanding_responses(), 1);
    assert_eq!(core.warps[0].barrier_pending[0], before_pending);
    assert_eq!(*core.l1_counters().unwrap(), before_l1);
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode");
    };
    assert!(st.l1.is_fill_in_flight(match reply {
        ReplyTo::L1Fill { fill, .. } => fill,
        _ => panic!("expected L1 fill reply"),
    }));
    assert_eq!(st.retry.len(), 1);
}

#[test]
fn retry_replacement_accepts_exact_maximum_pending_count() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(1);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                },
                inst(16),
            ],
            sectors: vec![0, 8, 12],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    let reply = requests[0].reply;
    core.warps[0].barrier_pending[0] = u32::MAX;
    let MemModel::L1Shared(st) = &mut core.mem else {
        panic!("expected shared mode");
    };
    let lookup_batch = st.l1.begin_lookup_batch();
    st.retry.push_back(Retry {
        warp: 0,
        instruction: 0,
        origin: MemoryOrigin::Global,
        constant_load: false,
        constant_deep: false,
        target: Some(FillTarget::Barrier(barrier(0))),
        mshr_floor: None,
        lookup_batch,
        lo: 1,
        hi: 3,
    });
    let mut emitted = Vec::new();

    core.on_response(reply, 244, &mut emitted).unwrap();
    assert_eq!(emitted.len(), 2);
    assert_eq!(core.outstanding_responses(), 2);
    assert_eq!(core.warps[0].barrier_pending[0], u32::MAX);
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode");
    };
    assert!(st.retry.is_empty());
}

#[test]
fn admission_counter_failure_does_not_partially_admit_a_block() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![
        BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        },
        BlockReplay {
            launch_index: 1,
            warps: vec![warp(1, vec![inst(0)])],
        },
    ];
    let mut p = params();
    p.max_resident_warps_per_sm = 1;
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    assert_eq!(core.next_block, 1);
    core.resident_warps = 0;
    core.arrival_stamp = u64::MAX;
    let before_warps = core.warps.len();
    let before_blocks = core.block_states.len();
    let before_subcores = core.subcore_warps.each_ref().map(Vec::len);

    assert!(matches!(
        core.admit_blocks(1),
        Err(IssueError::CounterOverflow {
            counter: "warp arrival stamp"
        })
    ));
    assert_eq!(core.warps.len(), before_warps);
    assert_eq!(core.block_states.len(), before_blocks);
    assert_eq!(core.subcore_warps.each_ref().map(Vec::len), before_subcores);
    assert_eq!(core.next_block, 1);
    assert_eq!(core.arrival_stamp, u64::MAX);
}

#[test]
fn exact_max_accounting_total_is_valid_when_nothing_is_added() {
    assert_eq!(
        checked_u64_counter_add("test total", u64::MAX, 0).unwrap(),
        u64::MAX
    );
}

#[test]
fn shared_mode_miss_blocks_consumer_until_response() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000001);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                },
                inst(16),
            ],
            sectors: vec![0],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();
    // The load misses → request emitted; the consumer's wait has a
    // PENDING barrier → no computable timer → blocked on response.
    let a = core.advance(ReplayHorizon::Unbounded, &mut out).unwrap();
    assert_eq!(a, Advance::Idle { wake_at: None });
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].issue_time, 0);
    assert_eq!(out[0].source, L2Source::L1Fill);
    let reply = out[0].reply;
    let ReplyTo::L1Fill { response_id, .. } = reply else {
        panic!("expected an L1 fill response token");
    };
    let mut out2 = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Bounded(300), &mut out2)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(
        core.t, 300,
        "a bounded response wait must advance the local clock to its horizon"
    );
    assert!(matches!(
        core.on_response(
            ReplyTo::Warp {
                warp: 0,
                barrier: barrier(0),
                response_id,
            },
            423,
            &mut out2,
        ),
        Err(IssueError::ResponseTokenNotOutstanding { .. })
    ));
    assert!(matches!(
        core.on_response(
            ReplyTo::L1Fill {
                fill: FillKey::Line {
                    line: 999,
                    generation: 0,
                },
                response_id,
            },
            423,
            &mut out2,
        ),
        Err(IssueError::ResponseTokenNotOutstanding { .. })
    ));
    assert!(out2.is_empty());
    assert_eq!(core.outstanding_responses(), 1);
    assert_eq!(core.warps[0].barrier_pending[0], 1);
    // Response at the DRAM arm (423): consumer issues at 423.
    core.on_response(reply, 423, &mut out2).unwrap();
    assert!(out2.is_empty());
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out2).unwrap(),
        Advance::Done
    );
    let r = core.result();
    assert_eq!(r.instructions_issued, 2);
    assert_eq!(r.cycles, 424);
}

#[test]
fn shared_mode_horizon_caps_run_ahead_after_a_request() {
    // Independent instructions after a miss: the SM may run ahead,
    // but never past issue_time + l1_miss_latency (the earliest
    // possible response).
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E")),
        (16, slot(cb(15), OpClass::Other, "IADD3")),
        (32, slot(cb(15), OpClass::Other, "IADD3")),
        (48, slot(cb(15), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                },
                inst(16),
                inst(32),
                inst(48),
            ],
            sectors: vec![0],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();
    // 4 independent issues would finish at t=46 — well inside the
    // 244-cycle response bound, so the run completes...
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(out.len(), 1);
    // ...but a tight EXTERNAL horizon yields first.
    let blocks2 = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: eligible(0, 1),
                },
                inst(16),
                inst(32),
                inst(48),
            ],
            sectors: vec![0],
        }],
    }];
    let mut core2 = shared_core(&blocks2, &sched, &params());
    let mut out2 = Vec::new();
    let a = core2
        .advance(ReplayHorizon::Bounded(10), &mut out2)
        .unwrap();
    assert_eq!(
        a,
        Advance::Idle { wake_at: Some(10) },
        "the next known timer (16) lies past the horizon: park AT the \
             horizon, where an event may ready an earlier warp"
    );
    assert_eq!(
        core2.advance(ReplayHorizon::Unbounded, &mut out2).unwrap(),
        Advance::Done
    );
}

#[test]
fn shared_mode_merged_waiters_all_clear_on_one_response() {
    // Two warps load the SAME sector; warp 1 merges into warp 0's
    // fill. Both consumers wait; ONE response clears both.
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000001);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let mk_warp = |slot_ix: i32| WarpReplay {
        async_copy_cohorts: None,
        warp_slot: slot_ix,
        insts: vec![
            ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: eligible(0, 1),
            },
            inst(16),
        ],
        sectors: vec![9],
    };
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![mk_warp(0), mk_warp(1)],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(out.len(), 1, "second warp merged, no second request");
    let c = core.l1_counters().unwrap();
    assert_eq!((c.misses, c.merges), (2, 1));
    core.on_response(out[0].reply, 300, &mut out).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    // Both consumers issued at 300 (distinct sub-cores) → cycles 301.
    assert_eq!(core.result().cycles, 301);
}

#[test]
fn unbarriered_retry_hit_skips_unused_timing_arms() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: eligible(0, 1),
            }],
            sectors: vec![0, 8],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let MemModel::L1Shared(st) = &mut core.mem else {
        panic!("expected shared mode");
    };
    let EvAccess::MissRequest { fill } = st
        .l1
        .access_eligible_ev(MemoryOrigin::Global, 8, 0)
        .unwrap()
    else {
        panic!("expected cold prefill miss");
    };
    st.l1.complete_fill(fill).unwrap();

    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(out.len(), 1, "cold issued load still owns an L1 fill");
    let reply = out.pop().unwrap().reply;
    let MemModel::L1Shared(st) = &mut core.mem else {
        panic!("expected shared mode");
    };
    let lookup_batch = st.l1.begin_lookup_batch();
    st.retry.push_back(Retry {
        warp: 0,
        instruction: 0,
        origin: MemoryOrigin::Global,
        constant_load: false,
        constant_deep: false,
        target: None,
        mshr_floor: None,
        lookup_batch,
        lo: 1,
        hi: 2,
    });
    let hits_before = st.l1.counters.hits;

    core.on_response(reply, u64::MAX - 1, &mut out).unwrap();
    assert!(out.is_empty());
    assert_eq!(core.outstanding_responses(), 0);
    assert_eq!(core.l1_counters().unwrap().hits, hits_before + 1);
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode");
    };
    assert!(st.retry.is_empty());
    assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
}

#[test]
fn shared_mode_mshr_stall_parks_and_retries_on_completion() {
    // MSHR capacity 1: the second warp's DIFFERENT-line load stalls,
    // parks, and retries when the first fill completes.
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000001);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        lane_mask: u32::MAX,
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                ],
                sectors: vec![0],
            },
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 1,
                insts: vec![
                    ReplayInst {
                        lane_mask: u32::MAX,
                        pc: 0,
                        kind: eligible(0, 1),
                    },
                    inst(16),
                ],
                sectors: vec![8],
            },
        ],
    }];
    let mut cfg = l1_config();
    cfg.mshr_count = Some(1);
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::L1Shared(cfg)).unwrap();
    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(out.len(), 1, "second load stalled, not requested");
    assert_eq!(core.l1_counters().unwrap().stall_events, 1);
    // First fill completes → retry emits the parked request at 244.
    let first = out[0].reply;
    out.clear();
    assert!(matches!(
        core.on_response(first, u64::MAX - 1, &mut out),
        Err(IssueError::TimeOverflow {
            operation: "retry self-response horizon",
            base,
            delta: 244,
        }) if base == u64::MAX - 1
    ));
    assert!(out.is_empty());
    assert_eq!(core.outstanding_responses(), 1);
    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode");
    };
    assert_eq!(st.retry.len(), 1);
    assert_eq!(core.warps[1].barrier_pending[0], 1);

    core.on_response(first, 244, &mut out).unwrap();
    assert_eq!(out.len(), 1, "retry sent the parked sector");
    assert_eq!(out[0].issue_time, 244);
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Idle { wake_at: None }
    );
    let second = out[0].reply;
    core.on_response(second, 488, &mut out).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    let r = core.result();
    assert_eq!(r.instructions_issued, 4);
    // Warp 1's consumer issues at 488 → cycles 489.
    assert_eq!(r.cycles, 489);
}

#[test]
fn shared_mode_bypass_load_waits_on_direct_reply() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(2));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000100);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E.CG")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalBypass {
                        off: 0,
                        len: 2,
                        token: CacheOpToken::LdgECg,
                        l2_requests: 0,
                    },
                },
                inst(16),
            ],
            sectors: vec![40, 41],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(out.len(), 2, "one request per bypass sector");
    assert!(out.iter().all(|r| r.source == L2Source::Bypass));
    assert!(
        out.iter().all(|r| r.operation == MemoryOperation::Read),
        ".cg bypass loads remain read operations"
    );
    assert!(matches!(
        out[0].reply,
        ReplyTo::Warp {
            warp: 0,
            barrier: got,
            ..
        } if got == barrier(2)
    ));
    // Both responses must land before the wait clears.
    let mut sink = Vec::new();
    core.on_response(out[0].reply, 244, &mut sink).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
        Advance::Idle { wake_at: None },
        "one of two pending units still outstanding"
    );
    core.on_response(out[1].reply, 423, &mut sink).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
        Advance::Done
    );
    assert_eq!(core.result().cycles, 424);
}

#[test]
fn invalid_response_times_are_rejected_before_mutation() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(2));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000100);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E.CG")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalBypass {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::LdgECg,
                        l2_requests: 0,
                    },
                },
                inst(16),
            ],
            sectors: vec![40],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(core.outstanding_responses(), 1);
    assert_eq!(core.warps[0].barrier_pending[2], 1);
    let ReplyTo::Warp { response_id, .. } = requests[0].reply else {
        panic!("expected a direct warp response token");
    };

    let mut retries = Vec::new();
    assert_eq!(core.now(), 244);
    assert!(matches!(
        core.on_response(requests[0].reply, 0, &mut retries),
        Err(IssueError::ResponseBeforeClock { now: 0, clock: 244 })
    ));
    assert!(matches!(
        core.on_response(
            ReplyTo::Warp {
                warp: 1,
                barrier: barrier(2),
                response_id,
            },
            244,
            &mut retries,
        ),
        Err(IssueError::ResponseWarpOutOfRange { warp: 1, warps: 1 })
    ));
    assert!(matches!(
        core.on_response(
            ReplyTo::Warp {
                warp: 0,
                barrier: barrier(1),
                response_id,
            },
            244,
            &mut retries,
        ),
        Err(IssueError::ResponseTokenNotOutstanding { .. })
    ));
    assert!(matches!(
        core.on_response(ReplyTo::None, 244, &mut retries),
        Err(IssueError::UnexpectedMutationOnlyResponse)
    ));
    assert!(retries.is_empty());
    assert_eq!(core.now(), 244);
    assert_eq!(core.outstanding_responses(), 1);
    assert_eq!(core.warps[0].barrier_pending[2], 1);
    assert_eq!(core.warps[0].barrier_clear[2], 0);

    core.on_response(requests[0].reply, 244, &mut retries)
        .unwrap();
    assert!(matches!(
        core.on_response(requests[0].reply, 244, &mut retries),
        Err(IssueError::ResponseTokenNotOutstanding { .. })
    ));
    assert!(retries.is_empty());
    assert_eq!(core.now(), 244);
    assert_eq!(core.outstanding_responses(), 0);
    assert_eq!(core.warps[0].barrier_pending[2], 0);
    assert_eq!(core.warps[0].barrier_clear[2], 244);
}

#[test]
fn valid_response_at_exact_maximum_clears_its_barrier() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(2));
    let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E.CG"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: AccessKind::GlobalBypass {
                    off: 0,
                    len: 1,
                    token: CacheOpToken::LdgECg,
                    l2_requests: 0,
                },
            }],
            sectors: vec![40],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    assert_eq!(requests.len(), 1);
    assert_eq!(core.warps[0].barrier_pending[2], 1);
    let mut retries = Vec::new();

    core.on_response(requests[0].reply, u64::MAX, &mut retries)
        .unwrap();

    assert!(retries.is_empty());
    assert_eq!(core.outstanding_responses(), 0);
    assert_eq!(core.warps[0].barrier_pending[2], 0);
    assert_eq!(core.warps[0].barrier_clear[2], u64::MAX);
}

#[test]
fn modeled_async_copy_source_does_not_claim_a_visibility_tail() {
    assert_eq!(
        TailFlags::for_access(AccessKind::GlobalAsyncCopy {
            off: 0,
            len: 1,
            completion_work: 4,
            l1_policy: LdgstsL1Policy::CacheAtAllLevels,
        }),
        TailFlags::default()
    );
}

#[test]
fn shared_mode_store_emits_mutation_only_requests() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "STG.E")),
        (16, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalStore { off: 0, len: 2 },
                },
                inst(16),
            ],
            sectors: vec![100, 101],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|r| r.source == L2Source::Store
        && r.origin == MemoryOrigin::Global
        && r.operation == MemoryOperation::Store
        && r.reply == ReplyTo::None));
    let result = core.result();
    assert_eq!(result.l1.unwrap().store_sectors, 2);
    assert_eq!(
        result.unmodeled_tails,
        TailFlags {
            red_rmw: false,
            dirty_writeback: true,
            coherence: true,
            ldgsts_dst_shared_timing: false,
        }
    );
}

#[test]
fn shared_mode_reduction_is_nonreturning_without_store_attribution() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "RED.E.ADD"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: AccessKind::GlobalReduction {
                    off: 0,
                    len: 2,
                    l2_requests: 1,
                },
            }],
            sectors: vec![100, 101],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|request| request.source == L2Source::Store
        && request.operation == MemoryOperation::Reduction
        && request.reply == ReplyTo::None
        && request.has_valid_contract()));
    let result = core.result();
    let l1 = result.l1.unwrap();
    assert_eq!(l1.store_sectors, 0);
    assert_eq!(l1.reduction_sectors, 2);
    assert_eq!(l1.reduction_hits, 0);
    assert_eq!(
        result.unmodeled_tails,
        TailFlags {
            red_rmw: true,
            dirty_writeback: true,
            coherence: true,
            ldgsts_dst_shared_timing: false,
        }
    );
}

/// §6.4 slice 2 — one instruction's cold 4-sector line miss reaches
/// the L2 as ONE request CARRYING ALL FOUR SECTORS (the merged
/// sectors fold into the fill's mask; NCU counts 4 LTS sectors),
/// with a single response completing every waiter.
#[test]
fn shared_mode_line_fill_folds_batch_sectors_into_one_request() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(2));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000100);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalEligible {
                        off: 0,
                        len: 4,
                        token: CacheOpToken::LdgECa,
                    },
                },
                inst(16),
            ],
            sectors: vec![40, 41, 42, 43],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(out.len(), 1, "one L2 transaction for the whole line");
    assert_eq!(out[0].origin, MemoryOrigin::Global);
    assert_eq!(out[0].sector_mask, 0b1111);
    assert!(matches!(
        out[0].reply,
        ReplyTo::L1Fill {
            fill: FillKey::Line { line: 10, .. },
            ..
        }
    ));
    let mut sink = Vec::new();
    core.on_response(out[0].reply, 244, &mut sink).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
        Advance::Done
    );
    assert_eq!(core.result().cycles, 245);
}

#[test]
fn constant_load_marker_reaches_shared_fill_request() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let sched = schedule(vec![(
        0,
        slot(load, OpClass::Other, "LDG.E.U16.CONSTANT R0, [R2.64]"),
    )]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: AccessKind::GlobalEligible {
                    off: 0,
                    len: 2,
                    token: CacheOpToken::LdgECa,
                },
            }],
            sectors: vec![40, 41],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();
    let _advance = core.advance(ReplayHorizon::Unbounded, &mut out).unwrap();
    assert_eq!(out.len(), 1);
    assert!(out[0].constant_load);
    // The broad constant marker must not imply the pressure-class return
    // service for a low-stream U16 load.
    assert!(!out[0].constant_deep);
}

#[test]
fn mixed_origin_same_batch_fold_preserves_one_physical_request() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(2));
    let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: AccessKind::GlobalEligible {
                    off: 0,
                    len: 1,
                    token: CacheOpToken::LdgECa,
                },
            }],
            sectors: vec![40, 41],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].origin, MemoryOrigin::Global);
    assert_eq!(out[0].sector_mask, 0b0001);
    assert_eq!(out[0].local_sector_mask, 0);

    let (mem, warps) = (&mut core.mem, &mut core.warps);
    let MemModel::L1Shared(st) = mem else {
        panic!("expected shared mode");
    };
    let response_id = ResponseId::new(st.response_owner, 0, 1, 1);
    assert!(
        access_shared(
            st,
            warps,
            0,
            None,
            None,
            MemoryOrigin::Local,
            false,
            false,
            41,
            response_id,
            0,
            &mut out,
        )
        .unwrap()
        .is_none()
    );

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].origin, MemoryOrigin::Global);
    assert_eq!(out[0].sector_mask, 0b0011);
    assert_eq!(out[0].local_sector_mask, 0b0010);
    assert_eq!(st.fill_masks.get(&10), Some(&0b0011));
    assert_eq!(st.l1.counters.origin(MemoryOrigin::Global).misses, 1);
    let local = st.l1.counters.origin(MemoryOrigin::Local);
    assert_eq!(local.misses, 1);
    assert_eq!(local.merges, 1);
}

#[test]
fn mixed_origin_retry_fold_is_transactional_and_preserves_one_request() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(2));
    let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDG.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: AccessKind::GlobalEligible {
                    off: 0,
                    len: 1,
                    token: CacheOpToken::LdgECa,
                },
            }],
            // Sector 40 owns the response. The two injected retry entries
            // target distinct sectors of line 11.
            sectors: vec![40, 44, 45],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    let reply = requests[0].reply;
    let fill = match reply {
        ReplyTo::L1Fill { fill, .. } => fill,
        _ => panic!("expected L1 fill response"),
    };

    let (before_counters, before_masks, before_pending, before_retries);
    {
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("expected shared mode");
        };
        let first_lookup_batch = st.l1.begin_lookup_batch();
        st.retry.push_back(Retry {
            warp: 0,
            instruction: 1,
            origin: MemoryOrigin::Global,
            constant_load: false,
            constant_deep: false,
            target: None,
            mshr_floor: None,
            lookup_batch: first_lookup_batch,
            lo: 1,
            hi: 2,
        });
        let second_lookup_batch = st.l1.begin_lookup_batch();
        st.retry.push_back(Retry {
            warp: 0,
            instruction: 2,
            origin: MemoryOrigin::Local,
            constant_load: false,
            constant_deep: false,
            target: None,
            mshr_floor: None,
            lookup_batch: second_lookup_batch,
            lo: 2,
            hi: 3,
        });
        before_counters = st.l1.counters;
        before_masks = st.fill_masks.clone();
        before_pending = core.warps[0].barrier_pending;
        before_retries = st.retry.len();
    }
    let mut emitted = Vec::new();

    assert!(matches!(
        core.on_response(reply, u64::MAX, &mut emitted),
        Err(IssueError::TimeOverflow {
            operation: "retry self-response horizon",
            base: u64::MAX,
            ..
        })
    ));
    assert!(emitted.is_empty());
    {
        let MemModel::L1Shared(st) = &core.mem else {
            panic!("expected shared mode");
        };
        assert!(st.outstanding_replies.contains(&reply));
        assert!(st.l1.is_fill_in_flight(fill));
        assert_eq!(st.l1.counters, before_counters);
        assert_eq!(st.fill_masks, before_masks);
        assert_eq!(st.retry.len(), before_retries);
        assert_eq!(core.warps[0].barrier_pending, before_pending);
    }

    core.on_response(reply, 244, &mut emitted).unwrap();
    assert_eq!(emitted.len(), 1);
    assert_eq!(emitted[0].origin, MemoryOrigin::Global);
    assert_eq!(emitted[0].sector_mask, 0b0011);
    assert_eq!(emitted[0].local_sector_mask, 0b0010);

    let MemModel::L1Shared(st) = &core.mem else {
        panic!("expected shared mode");
    };
    assert!(!st.outstanding_replies.contains(&reply));
    assert!(!st.l1.is_fill_in_flight(fill));
    assert!(st.retry.is_empty());
    assert_eq!(core.warps[0].barrier_pending, [0; DEP_BARRIERS]);
    assert_eq!(st.l1.counters.origin(MemoryOrigin::Global).misses, 2);
    assert_eq!(st.l1.counters.origin(MemoryOrigin::Local).misses, 1);
}

/// §6.4 slice 2 — a LATER instruction merging onto an in-flight
/// line fill: a NEW sector gets an independent sector response
/// (the L2 must see it once), a re-touched sector sends nothing
/// (MSHR merge), and each waiter clears on its own returned child.
#[test]
fn shared_mode_same_batch_cross_instruction_merge_folds() {
    // Folding is scoped to one issue timestamp.  This fixture advances
    // the same actor through t=0 and t=15, so the later sector must not
    // inherit the first request's response event.
    let mut load_a = cb(1);
    load_a.write_barrier = Some(barrier(2));
    let mut load_b = cb(1);
    load_b.write_barrier = Some(barrier(2));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000100);
    let sched = schedule(vec![
        (0, slot(load_a, OpClass::Other, "LDG.E")),
        (16, slot(load_b, OpClass::Other, "LDG.E")),
        (32, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalEligible {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::LdgECa,
                    },
                },
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 16,
                    kind: AccessKind::GlobalEligible {
                        off: 1,
                        len: 2,
                        token: CacheOpToken::LdgECa,
                    },
                },
                inst(32),
            ],
            // A: sector 40. B: sector 41 (folds) + 40 (re-touch).
            sectors: vec![40, 41, 40],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(out.len(), 2, "the late sector owns a separate child");
    assert_eq!(out[0].sector_mask, 0b01);
    assert_eq!(out[1].sector_mask, 0b10);
    assert!(matches!(
        out[0].reply,
        ReplyTo::L1Fill {
            fill: FillKey::Line { line: 10, .. },
            ..
        }
    ));
    assert!(matches!(
        out[1].reply,
        ReplyTo::L1Fill {
            fill: FillKey::Sector {
                line: 10,
                sector: 1,
                ..
            },
            ..
        }
    ));
    let mut sink = Vec::new();
    core.on_response(out[0].reply, 244, &mut sink).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
        Advance::Idle { wake_at: None },
        "the late sector waiter remains pending"
    );
    core.on_response(out[1].reply, 260, &mut sink).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
        Advance::Done,
        "the child response clears the late-sector waiter"
    );
}

/// A merge onto a fill emitted in an EARLIER batch (the request
/// already left with the SM's previous yield) cannot fold — it
/// sends exactly one sector follow-up response, and the late
/// waiter clears only when that child returns.
#[test]
fn shared_mode_cross_batch_merge_sends_sector_followup_once() {
    let mut load_m = cb(1);
    load_m.write_barrier = Some(barrier(2));
    let mut load_l = cb(1);
    load_l.write_barrier = Some(barrier(3));
    // Merging load: gated on barrier 2 (fill M's return) so it
    // issues in a LATER advance call while fill L is still in
    // flight; its own waiter registers on barrier 2.
    let mut load_merge = cb(1);
    load_merge.write_barrier = Some(barrier(2));
    load_merge.wait_mask = wait_mask(0b000100);
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b001100);
    let sched = schedule(vec![
        (0, slot(load_m, OpClass::Other, "LDG.E")),
        (16, slot(load_l, OpClass::Other, "LDG.E")),
        (32, slot(load_merge, OpClass::Other, "LDG.E")),
        (48, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalEligible {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::LdgECa,
                    },
                },
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 16,
                    kind: AccessKind::GlobalEligible {
                        off: 1,
                        len: 1,
                        token: CacheOpToken::LdgECa,
                    },
                },
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 32,
                    kind: AccessKind::GlobalEligible {
                        off: 2,
                        len: 1,
                        token: CacheOpToken::LdgECa,
                    },
                },
                inst(48),
            ],
            // M: sector 80 (line 20). L: sector 40 (line 10).
            // Merge: sector 41 (line 10, new sector).
            sectors: vec![80, 40, 41],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(out.len(), 2, "two fills; the merging load is gated");
    assert!(matches!(
        out[0].reply,
        ReplyTo::L1Fill {
            fill: FillKey::Line { line: 20, .. },
            ..
        }
    ));
    assert!(matches!(
        out[1].reply,
        ReplyTo::L1Fill {
            fill: FillKey::Line { line: 10, .. },
            ..
        }
    ));
    assert_eq!(out[1].sector_mask, 0b01);
    // Fill M returns; the merging load issues in a NEW batch while
    // fill L is still in flight.
    let mut out2 = Vec::new();
    core.on_response(out[0].reply, 423, &mut out2).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out2).unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(out2.len(), 1, "exactly one follow-up fetch");
    assert_eq!(out2[0].sector_mask, 0b10, "only the new sector");
    assert!(matches!(
        out2[0].reply,
        ReplyTo::L1Fill {
            fill: FillKey::Sector {
                line: 10,
                sector: 1,
                ..
            },
            ..
        }
    ));
    assert_eq!(out2[0].source, L2Source::L1Fill);
    // Fill L's response clears its own waiter (barrier 3), but the
    // late-sector waiter (barrier 2) remains until the child response.
    let mut sink = Vec::new();
    core.on_response(out[1].reply, 846, &mut sink).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
        Advance::Idle { wake_at: None }
    );
    core.on_response(out2[0].reply, 847, &mut sink).unwrap();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut sink).unwrap(),
        Advance::Done
    );
}

/// ATOM records ride BYPASS provenance (they never stage into the
/// modeled L1) but retain returning RMW operation semantics. The system
/// derives the write-class L2 rule and missing-sector fetch from that type.
#[test]
fn shared_mode_atom_is_bypass_provenance_write_class() {
    let mut atom = cb(1);
    atom.write_barrier = Some(barrier(2));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000100);
    let sched = schedule(vec![
        (0, slot(atom, OpClass::Other, "ATOMG.E.ADD.STRONG.GPU")),
        (16, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalBypass {
                        off: 0,
                        len: 1,
                        token: CacheOpToken::Atom,
                        l2_requests: 1,
                    },
                },
                inst(16),
            ],
            sectors: vec![40],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Idle { wake_at: None }
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].source, L2Source::Bypass);
    assert_eq!(out[0].operation, MemoryOperation::Atomic);
    assert_eq!(out[0].operation.l2_class(), L2OpClass::WriteAtomic);
    assert!(
        matches!(
            out[0].reply,
            ReplyTo::Warp {
                warp: 0,
                barrier: got,
                ..
            } if got == barrier(2)
        ),
        "the atomic's return value still waits on the reply"
    );
}
