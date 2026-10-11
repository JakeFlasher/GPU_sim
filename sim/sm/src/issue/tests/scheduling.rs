use super::*;

#[test]
fn stall_counts_delay_reissue() {
    // stalls 4,4: issues at t=0,4 → cycles 5.
    let sched = schedule(vec![
        (0, slot(cb(4), OpClass::Other, "FFMA")),
        (16, slot(cb(4), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16)])],
    }];
    let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(r.instructions_issued, 2);
    assert_eq!(r.cycles, 5);
}

#[test]
fn same_smsp_tensor_issues_observe_the_initiation_interval() {
    let sched = schedule(vec![
        (0, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
        (16, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 2);
    assert_eq!(
        core.result().cycles,
        17,
        "tensor issues must occur at t=0 and t=16"
    );
    assert_eq!(core.tensor_pipe_ready, [32, 0, 0, 0]);
}

#[test]
fn separate_smsps_have_independent_tensor_timelines() {
    let sched = schedule(vec![(0, tensor_slot(cb(1), "HMMA.16816.F32.BF16", 16))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(0)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(core.instructions_issued(), 2);
    assert_eq!(core.result().cycles, 1, "both SMSPs should issue at t=0");
    assert_eq!(core.tensor_pipe_ready, [16, 16, 0, 0]);
}

#[test]
fn dispatch_only_work_fills_a_tensor_busy_gap() {
    let sched = schedule(vec![
        (0, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
        (16, slot(cb(1), OpClass::Other, "IADD3")),
        (32, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16), inst(32)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Bounded(2), &mut out).unwrap(),
        Advance::Idle { wake_at: Some(2) }
    );
    assert_eq!(core.instructions_issued(), 2);
    assert_eq!(core.warps[0].next, 2);
    assert_eq!(core.tensor_pipe_ready[0], 16);

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(core.result().cycles, 17);
}

#[test]
fn blocked_held_tensor_warp_yields_dispatch_to_another_warp() {
    let sched = schedule(vec![
        (0, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
        (16, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
        (32, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        // Warp 4 is younger, so CGGTY picks and holds it at t=0.
        warps: vec![warp(0, vec![inst(32)]), warp(4, vec![inst(0), inst(16)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Bounded(1), &mut out).unwrap(),
        Advance::Idle { wake_at: Some(1) }
    );
    assert_eq!(core.hold[0], Some(1));
    assert_eq!(core.warps[1].next, 1);

    assert_eq!(
        core.advance(ReplayHorizon::Bounded(2), &mut out).unwrap(),
        Advance::Idle { wake_at: Some(2) }
    );
    assert!(core.warps[0].done, "ready dispatch warp was not selected");
    assert_eq!(core.warps[1].next, 1, "blocked held tensor warp issued");
    assert_eq!(core.instructions_issued(), 2);
}

#[test]
fn tensor_blocked_idle_reports_the_exact_resource_wake_time() {
    let sched = schedule(vec![
        (0, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
        (16, tensor_slot(cb(1), "HMMA.16816.F32", 16)),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Bounded(1), &mut out).unwrap(),
        Advance::Idle { wake_at: Some(1) }
    );
    assert_eq!(
        core.advance(ReplayHorizon::Bounded(16), &mut out).unwrap(),
        Advance::Idle { wake_at: Some(16) }
    );
    assert_eq!(core.instructions_issued(), 1);
    assert_eq!(core.warps[0].next, 1);
}

#[test]
fn tensor_timeline_survives_cta_admission_waves() {
    let sched = schedule(vec![(0, tensor_slot(cb(1), "HMMA.16816.F32", 16))]);
    let blocks = vec![
        BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        },
        BlockReplay {
            launch_index: 1,
            warps: vec![warp(0, vec![inst(0)])],
        },
    ];
    let mut p = params();
    p.max_resident_warps_per_sm = 1;
    let r = simulate_sm(&blocks, &sched, &p, None).unwrap();

    assert_eq!(r.instructions_issued, 2);
    assert_eq!(r.cycles, 17, "second CTA must wait for the t=16 timeline");
}

#[test]
fn tensor_timestamp_overflow_rejects_the_cycle_before_mutation() {
    let sched = schedule(vec![(0, tensor_slot(cb(1), "HMMA.16816.F32", 16))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    core.t = u64::MAX - 15;
    let before_unfinished = core.unfinished_warps;
    let before_resident = core.resident_warps;
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::TimeOverflow {
            operation: "tensor-pipeline availability",
            base,
            delta: 16,
        }) if base == u64::MAX - 15
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert!(!core.warps[0].done);
    assert_eq!(core.tensor_pipe_ready, [0; SUB_CORES]);
    assert_eq!(core.unfinished_warps, before_unfinished);
    assert_eq!(core.resident_warps, before_resident);
}

#[test]
fn flat_write_barrier_wait_blocks_consumer_without_l1() {
    // inst0 (global load, no L1 attached) sets barrier 0; inst1 waits.
    // Barrier clears at the flat latency 400 → inst1 at t=400.
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000001);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E")),
        (16, slot(consumer, OpClass::Other, "FFMA")),
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
    let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(r.instructions_issued, 2);
    assert_eq!(r.cycles, 401);
    assert_eq!(r.readiness.limiting_cycles.dependence_counter_l1tex, 400);
    assert_eq!(
        r.readiness
            .constraint_exposure_cycles
            .dependence_counter_l1tex,
        400
    );
}

#[test]
fn ordinary_depbar_operand_waits_for_non_memory_producer_without_async_mutation() {
    // The spacing probe's retained clock reads compile as an S2UR producer
    // on SB1..SB5 followed by `DEPBAR.LE SBn, 0x0`. The DEPBAR control word
    // has no wait mask, so the operand itself must consume the ordinary
    // non-memory producer latency (25 cycles in this fixture).
    let mut producer = cb(1);
    producer.write_barrier = Some(barrier(2));
    let sched = schedule(vec![
        (0, slot(producer, OpClass::Other, "S2UR UR4, SR_CLOCK")),
        (16, slot(cb(7), OpClass::Other, "DEPBAR.LE SB2, 0x0")),
        // Retained only to enable the async protocol sidecar in this
        // fixture, making absence of accidental group mutation observable.
        (32, async_wait_slot(0)),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    assert!(requests.is_empty());
    assert!(core.async_copies[0].state.is_none());
    let result = core.result();
    assert_eq!(result.instructions_issued, 2);
    assert_eq!(result.cycles, 26);
    assert_eq!(
        result
            .readiness
            .limiting_cycles
            .dependence_counter_non_memory_write,
        25
    );
    assert_eq!(result.readiness.async_groups.groups_committed, 0);
    assert_eq!(result.readiness.async_groups.groups_waited, 0);
}

#[test]
fn ordinary_depbar_brace_set_waits_for_all_named_scoreboards_without_async_mutation() {
    // Exact form emitted by the cohort probe: an LDGSTS operand-read
    // producer on SB1 plus R2UR write producers on SB2/SB3. The encoded
    // control wait mask is empty; the explicit operands must wait for the
    // maximum ordinary clear stamp (SB3 at t=27).
    let mut ldgsts = cb(1);
    ldgsts.read_barrier = Some(barrier(1));
    let mut r2ur_sb2 = cb(1);
    r2ur_sb2.write_barrier = Some(barrier(2));
    let mut r2ur_sb3 = cb(1);
    r2ur_sb3.write_barrier = Some(barrier(3));
    let sched = schedule(vec![
        (0, slot(ldgsts, OpClass::Other, "LDGSTS.E.BYPASS.128")),
        (16, slot(r2ur_sb2, OpClass::Other, "R2UR UR4, R2")),
        (32, slot(r2ur_sb3, OpClass::Other, "R2UR UR5, R3")),
        (48, slot(cb(7), OpClass::Other, "DEPBAR.LE SB1, 0x0, {3,2}")),
        // Enables the async protocol sidecar without executing an async
        // instruction, making accidental SB0 group mutation observable.
        (64, async_wait_slot(0)),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16), inst(32), inst(48)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    assert!(requests.is_empty());
    assert!(core.async_copies[0].state.is_none());
    let result = core.result();
    assert_eq!(result.instructions_issued, 4);
    assert_eq!(result.cycles, 28);
    assert_eq!(result.readiness.async_groups.groups_committed, 0);
    assert_eq!(result.readiness.async_groups.groups_waited, 0);
}

#[test]
fn async_sb0_wait_composes_with_its_encoded_ordinary_wait_mask() {
    let run = |async_source_ready: u64| {
        let mut producer = cb(1);
        producer.write_barrier = Some(barrier(1));
        let mut composed_wait = async_wait_slot(0);
        composed_wait.control.wait_mask = wait_mask(0b000010);
        let sched = schedule(vec![
            (0, slot(producer, OpClass::Other, "R2UR UR4, R2")),
            (16, composed_wait),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16)])],
        }];
        let p = params();
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        core.async_copies[0]
            .add_known_async_copy(0, async_source_ready, 0, 0, &p)
            .unwrap();
        assert!(core.async_copies[0].commit_async_copy_group(0));
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        assert!(requests.is_empty());
        core.result()
    };

    // The same instruction waits for whichever independently modeled
    // frontier is later: SB1's ordinary producer at t=25, or SB0's group.
    let ordinary_limited = run(10);
    assert_eq!(ordinary_limited.cycles, 26);
    assert_eq!(ordinary_limited.readiness.async_groups.groups_waited, 1);
    let async_limited = run(40);
    assert_eq!(async_limited.cycles, 41);
    assert_eq!(async_limited.readiness.async_groups.groups_waited, 1);
}

#[test]
fn shared_bank_wavefronts_extend_dependent_completion_in_both_memory_modes() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000001);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDS")),
        (16, slot(consumer, OpClass::Other, "IADD3")),
    ]);
    let blocks = |wavefronts| {
        vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(
                0,
                vec![
                    ReplayInst {
                        lane_mask: u32::MAX,
                        pc: 0,
                        kind: AccessKind::Shared { wavefronts },
                    },
                    inst(16),
                ],
            )],
        }]
    };

    for l1 in [None, Some(l1_config())] {
        let one = simulate_sm(&blocks(1), &sched, &params(), l1).unwrap();
        let four = simulate_sm(&blocks(4), &sched, &params(), l1).unwrap();
        assert_eq!(one.cycles, 24);
        assert_eq!(four.cycles, 30);
        assert_eq!(four.cycles - one.cycles, 3 * 2);
    }
}

#[test]
fn shared_wavefront_latency_multiplication_overflow_is_nonmutating() {
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let sched = schedule(vec![(0, slot(load, OpClass::Other, "LDS"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(
            0,
            vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: AccessKind::Shared { wavefronts: 3 },
            }],
        )],
    }];
    let mut p = params();
    p.shared_additional_wavefront_latency = u64::MAX;
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::TimeMultiplicationOverflow {
            operation: "shared-memory additional-wavefront latency",
            lhs: u64::MAX,
            rhs: 2,
        })
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
}

#[test]
fn l1_hit_shortens_the_dependent_load_chain() {
    // Two dependent loads of the SAME sector + consumers. With the L1
    // attached, the first load misses (ready at 244) and the second
    // HITS (ready at +34); without it both cost the flat 400.
    let mut load = cb(1);
    load.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000001);
    let sched = schedule(vec![
        (0, slot(load, OpClass::Other, "LDG.E")),
        (16, slot(consumer, OpClass::Other, "FFMA")),
        (32, slot(load, OpClass::Other, "LDG.E")),
        (48, slot(consumer, OpClass::Other, "EXIT")),
    ]);
    let mk = || {
        vec![BlockReplay {
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
                    ReplayInst {
                        lane_mask: u32::MAX,
                        pc: 32,
                        kind: eligible(1, 1),
                    },
                    inst(48),
                ],
                sectors: vec![7, 7],
            }],
        }]
    };
    let with_l1 = simulate_sm(&mk(), &sched, &params(), Some(l1_config())).unwrap();
    let without = simulate_sm(&mk(), &sched, &params(), None).unwrap();
    let c = with_l1.l1.unwrap();
    assert_eq!(c.hits, 1);
    assert_eq!(c.misses, 1);
    assert_eq!(c.line_alloc_misses, 1);
    // miss chain: barrier clears 244, consumer at 244; hit chain: the
    // second load issues at 245, hit ready 245+34; EXIT at 279.
    assert!(with_l1.cycles < without.cycles);
    assert_eq!(without.cycles, 802);
}

#[test]
fn bypass_untagged_and_store_sectors_are_counted() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDG.E.CG")),
        (16, slot(cb(1), OpClass::Other, "LDG.E")),
        (32, slot(cb(1), OpClass::Other, "STG.E")),
        (48, slot(cb(1), OpClass::Other, "EXIT")),
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
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 16,
                    kind: AccessKind::GlobalUntagged { off: 2, len: 1 },
                },
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 32,
                    kind: AccessKind::GlobalStore { off: 3, len: 2 },
                },
                inst(48),
            ],
            sectors: vec![0, 1, 2, 3, 4],
        }],
    }];
    let r = simulate_sm(&blocks, &sched, &params(), Some(l1_config())).unwrap();
    let c = r.l1.unwrap();
    assert_eq!(c.bypass_sectors, 2);
    assert_eq!(c.untagged_sectors, 1);
    assert_eq!(c.store_sectors, 2);
    assert_eq!(c.store_hits, 0, "stores never allocate");
    assert_eq!(c.hits + c.misses, 0, "no modeled accesses in this stream");
}

#[test]
fn yield_bit_excludes_warp_next_cycle() {
    // CGGTY paper: yield = "do not issue this warp next cycle"; with
    // no other warp ready that is a one-cycle bubble. stall 1 +
    // yield → second instruction issues at t=2, cycles=3.
    let mut y = cb(1);
    y.yield_flag = true;
    let sched = schedule(vec![
        (0, slot(y, OpClass::Other, "IADD3")),
        (16, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16)])],
    }];
    let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(r.cycles, 3);
}

#[test]
fn shared_barrier_index_accumulates_like_a_counter() {
    // Two producers on barrier 0: a slow global (flat 400) then a
    // fast non-memory (25). The consumer waits on bit 0 and must
    // wait for the SLOW producer — hardware barriers are counters,
    // a later fast producer must not shorten the pending wait.
    let mut slow = cb(1);
    slow.write_barrier = Some(barrier(0));
    let mut fast = cb(1);
    fast.write_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000001);
    let sched = schedule(vec![
        (0, slot(slow, OpClass::Other, "LDG.E")),
        (16, slot(fast, OpClass::Other, "S2R")),
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
    let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    // Consumer waits until t=400 (slow), NOT t=1+25 (fast overwrite).
    assert_eq!(r.cycles, 401);
}
