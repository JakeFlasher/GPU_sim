use super::*;

#[test]
fn all_empty_block_does_not_leak_residency() {
    // Block 0's warps are all traced-empty; block 1 must still be
    // admitted and replayed even under a tight residency cap.
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![
        BlockReplay {
            launch_index: 0,
            warps: (0..4).map(|s| warp(s, vec![])).collect(),
        },
        BlockReplay {
            launch_index: 1,
            warps: vec![warp(0, vec![inst(0)])],
        },
    ];
    let mut p = params();
    p.max_resident_warps_per_sm = 4;
    let r = simulate_sm(&blocks, &sched, &p, None).unwrap();
    assert_eq!(r.instructions_issued, 1);
}

#[test]
fn low_level_replay_rejects_invalid_physical_warp_slots() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);

    let negative = vec![BlockReplay {
        launch_index: 7,
        warps: vec![warp(-1, vec![inst(0)])],
    }];
    assert!(matches!(
        SmCore::new(&negative, &sched, &params(), SmMemMode::Flat),
        Err(IssueError::WarpSlotOutOfRange {
            launch_index: 7,
            warp_slot: -1,
            arch: ArchId::Sm86,
            physical_warp_slots: 48,
        })
    ));

    let out_of_range = vec![BlockReplay {
        launch_index: 8,
        warps: vec![warp(48, vec![inst(0)])],
    }];
    assert!(matches!(
        SmCore::new(&out_of_range, &sched, &params(), SmMemMode::Flat),
        Err(IssueError::WarpSlotOutOfRange {
            launch_index: 8,
            warp_slot: 48,
            arch: ArchId::Sm86,
            physical_warp_slots: 48,
        })
    ));

    let duplicate = vec![BlockReplay {
        launch_index: 9,
        warps: vec![warp(3, vec![inst(0)]), warp(3, vec![inst(0)])],
    }];
    assert!(matches!(
        SmCore::new(&duplicate, &sched, &params(), SmMemMode::Flat),
        Err(IssueError::DuplicateWarpSlot {
            launch_index: 9,
            warp_slot: 3,
        })
    ));
}

#[test]
fn first_cta_must_fit_the_resident_warp_ceiling() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 4,
        warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(0)])],
    }];
    let mut p = params();
    p.max_resident_warps_per_sm = 1;
    assert!(matches!(
        SmCore::new(&blocks, &sched, &p, SmMemMode::Flat),
        Err(IssueError::BlockResidencyLimitExceeded {
            launch_index: 4,
            live_warps: 2,
            arch: ArchId::Sm86,
            max_resident_warps: 1,
        })
    ));

    let blocks = vec![BlockReplay {
        launch_index: 5,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    p.max_resident_warps_per_sm = 0;
    assert!(matches!(
        SmCore::new(&blocks, &sched, &p, SmMemMode::Flat),
        Err(IssueError::BlockResidencyLimitExceeded {
            launch_index: 5,
            live_warps: 1,
            arch: ArchId::Sm86,
            max_resident_warps: 0,
        })
    ));
}

#[test]
fn first_cta_must_fit_the_architecture_shared_memory_capacity() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 6,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    assert!(matches!(
        SmCore::new_with_residency(
            &blocks,
            &sched,
            &params(),
            SharedMemoryResidency::new(64, 65),
            SmMemMode::Flat,
        ),
        Err(IssueError::BlockSharedMemoryLimitExceeded {
            launch_index: 6,
            bytes_per_cta: 65,
            arch: ArchId::Sm86,
            max_bytes_per_sm: 64,
        })
    ));
}

#[test]
fn colliding_ctas_wait_then_reuse_the_physical_slot() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
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
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    assert_eq!(
        core.next_block, 1,
        "colliding CTA was admitted concurrently"
    );
    assert_eq!(core.block_states.len(), 1);
    assert_eq!(&*core.resident_warp_slots, &HashSet::from([0]));

    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Bounded(1), &mut out).unwrap(),
        Advance::Idle { wake_at: Some(1) }
    );
    assert_eq!(
        core.next_block, 2,
        "retirement did not admit the waiting CTA"
    );
    assert_eq!(core.block_states.len(), 2);
    assert_eq!(&*core.resident_warp_slots, &HashSet::from([0]));

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert!(core.resident_warp_slots.is_empty());
    assert_eq!(core.instructions_issued(), 2);
}

#[test]
fn greedy_hold_then_youngest_switch() {
    // Two warps on the same sub-core (slots 0 and 4); everyone ready:
    // one issue per cycle on the shared sub-core → 4 cycles.
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "IADD3")),
        (16, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![
            warp(0, vec![inst(0), inst(16)]),
            warp(4, vec![inst(0), inst(16)]),
        ],
    }];
    let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(r.instructions_issued, 4);
    assert_eq!(r.cycles, 4);
}

#[test]
fn warps_on_distinct_subcores_issue_in_parallel() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "IADD3")),
        (16, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: (0..4).map(|s| warp(s, vec![inst(0), inst(16)])).collect(),
    }];
    let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(r.instructions_issued, 8);
    assert_eq!(r.cycles, 2);
}

#[test]
fn block_barrier_synchronizes_block() {
    // Warp 0 reaches BAR early and must wait for warp 1, which has an
    // extra pre-barrier instruction with a 12-cycle stall.
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "IADD3")),
        (8, slot(cb(12), OpClass::Other, "FFMA")),
        (
            16,
            slot(cb(1), OpClass::BlockBarrier, "BAR.SYNC.DEFER_BLOCKING"),
        ),
        (32, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![
            warp(0, vec![inst(0), inst(16), inst(32)]),
            warp(1, vec![inst(0), inst(8), inst(16), inst(32)]),
        ],
    }];
    let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(r.instructions_issued, 7);
    assert!(r.cycles >= 15, "cycles={}", r.cycles);
    assert_eq!(r.readiness.cta_barriers.generations, 1);
    assert_eq!(r.readiness.cta_barriers.total_arrival_spread_cycles, 12);
    assert_eq!(r.readiness.cta_barriers.max_arrival_spread_cycles, 12);
    assert_eq!(r.readiness.cta_barriers.total_rendezvous_warp_cycles, 14);
    assert_eq!(r.readiness.limiting_cycles.cta_barrier_rendezvous, 14);
}

#[test]
fn terminal_block_barrier_waits_for_all_participants() {
    // A barrier can legally be the final instruction in a kernel.  It
    // still has CTA rendezvous semantics: the first warp must not retire
    // the block before the other participant arrives.
    let sched = schedule(vec![(0, barrier_pipeline_slot(cb(1)))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(0)])],
    }];
    let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(result.instructions_issued, 2);
    assert_eq!(result.readiness.cta_barriers.generations, 1);
    assert_eq!(result.modeled_local_completion_cycles, Some(5));
}

#[test]
fn terminal_barrier_release_clears_the_held_warp_slot() {
    // A final BAR.SYNC is held while waiting for its CTA peers.  Once the
    // generation releases, retirement must clear the CGGTY hold pointer;
    // otherwise the index can alias a later swap-removed/admitted warp.
    let sched = schedule(vec![(0, barrier_pipeline_slot(cb(1)))]);
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
    assert_eq!(core.hold, [None; SUB_CORES]);
    assert!(core.warps.iter().all(|warp| warp.done));
}

#[test]
fn terminal_barrier_missing_resident_slot_is_transactional() {
    // The deferred final-barrier retirement path must run the same
    // residency preflight as ordinary retirement.  Corrupting the slot
    // witness should fail before dispatch, not after one participant has
    // been retired.
    let sched = schedule(vec![(0, barrier_pipeline_slot(cb(1)))]);
    let blocks = vec![BlockReplay {
        launch_index: 12,
        warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(0)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    assert!(core.resident_warp_slots.remove(&1));
    let before = (
        core.instructions_issued(),
        core.unfinished_warps,
        core.resident_warps,
        core.subcore_warps.each_ref().map(Vec::clone),
        core.hold,
    );
    let mut out = Vec::new();
    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::ResidentWarpSlotMissing {
            launch_index: 12,
            warp_slot: 1,
        })
    ));
    assert!(out.is_empty());
    assert_eq!(
        (
            core.instructions_issued(),
            core.unfinished_warps,
            core.resident_warps,
            core.subcore_warps.each_ref().map(Vec::clone),
            core.hold,
        ),
        before
    );
    assert!(core.warps.iter().all(|warp| !warp.done && warp.next == 0));
}

#[test]
fn one_warp_terminal_block_barrier_still_forms_a_generation() {
    let sched = schedule(vec![(0, barrier_pipeline_slot(cb(1)))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(result.instructions_issued, 1);
    assert_eq!(result.readiness.cta_barriers.generations, 1);
    assert_eq!(result.readiness.cta_barriers.total_arrival_spread_cycles, 0);
    assert_eq!(result.modeled_local_completion_cycles, Some(3));
}

#[test]
fn terminal_barrier_participant_retires_after_mixed_generation_release() {
    let sched = schedule(vec![
        (0, barrier_pipeline_slot(cb(1))),
        (16, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(0), inst(16)])],
    }];
    let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(result.instructions_issued, 3);
    assert_eq!(result.readiness.cta_barriers.generations, 1);
    assert_eq!(result.readiness.barrier_pipeline.generations_completed, 1);
    assert!(
        result
            .modeled_local_completion_cycles
            .is_some_and(|cycles| cycles >= 3)
    );
}

#[test]
fn block_after_terminal_barrier_is_admitted_at_release_time() {
    let sched = schedule(vec![
        (0, barrier_pipeline_slot(cb(1))),
        (16, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![
        BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(0)])],
        },
        BlockReplay {
            launch_index: 1,
            warps: vec![warp(0, vec![inst(16)]), warp(1, vec![inst(16)])],
        },
    ];
    let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(result.instructions_issued, 4);
    assert!(result.cycles >= 6, "cycles={}", result.cycles);
}

#[cfg(feature = "reference")]
#[test]
fn typed_local_completion_preserves_terminal_and_retired_cta_frontiers() {
    let sched = schedule(vec![
        (0, barrier_pipeline_slot(cb(1))),
        (16, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    // A terminal barrier wave followed by plain work in reused slots.
    // The issue clock first trails the known release; the retained
    // frontier survives the later retirement and remains distinct from
    // the issue clock.
    let blocks = vec![
        BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(0)])],
        },
        BlockReplay {
            launch_index: 1,
            warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(0)])],
        },
        BlockReplay {
            launch_index: 2,
            warps: vec![warp(0, vec![inst(16)]), warp(1, vec![inst(16)])],
        },
    ];
    for dense in [false, true] {
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        core.dense_reference = dense;
        assert_eq!(core.typed_local_completion_time(), None);
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Bounded(1), &mut requests)
                .unwrap(),
            Advance::Idle { wake_at: Some(1) }
        );
        assert_eq!(core.typed_local_completion_time(), Some(5));
        assert_eq!(core.result().modeled_local_completion_cycles, Some(5));
        assert!(core.t < 5);
        assert!(
            !core
                .typed_unprojected_timing_state()
                .iter()
                .any(|reason| { reason == "local_completion_frontier" })
        );
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        assert_eq!(core.typed_local_completion_time(), Some(11));
        assert_eq!(core.result().modeled_local_completion_cycles, Some(11));
        assert!(requests.is_empty());
    }
}

#[test]
fn barrier_release_after_early_waiter_is_not_time_travel() {
    // Warp 0 parks at a final barrier first.  Warp 1 exits much later;
    // that retirement is the event that makes the barrier releasable, so
    // release/admission cannot use the historical arrival+1 timestamp.
    let sched = schedule(vec![
        (0, barrier_pipeline_slot(cb(1))),
        (16, slot(cb(12), OpClass::Other, "FFMA")),
        (32, slot(cb(1), OpClass::Other, "EXIT")),
        (48, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![
        BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)]), warp(1, vec![inst(16), inst(32)])],
        },
        BlockReplay {
            launch_index: 1,
            warps: vec![warp(0, vec![inst(48)])],
        },
    ];
    let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(result.instructions_issued, 4);
    // The successor CTA must wait until the delayed participant's
    // retirement (t=12) plus the release/admission edge, rather than being
    // admitted at the stale first-arrival timestamp (t=3/4).
    assert!(result.cycles >= 14, "cycles={}", result.cycles);
}

#[test]
fn sm_shared_barrier_service_queues_parallel_smsp_dispatches() {
    let sched = schedule(vec![
        (0, barrier_pipeline_slot(cb(1))),
        (16, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: (0..4)
            .map(|warp_slot| warp(warp_slot, vec![inst(0), inst(16)]))
            .collect(),
    }];
    let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();

    assert_eq!(result.instructions_issued, 8);
    assert_eq!(result.cycles, 10);
    assert_eq!(result.readiness.schema_version(), 2);
    assert_eq!(result.readiness.barrier_pipeline.instructions_issued, 4);
    assert_eq!(
        result.readiness.barrier_pipeline.issue_server_busy_cycles,
        8
    );
    assert_eq!(result.readiness.barrier_pipeline.generations_completed, 1);
    assert_eq!(
        result
            .readiness
            .barrier_pipeline
            .total_generation_issue_span_cycles,
        0
    );
    assert_eq!(
        result
            .readiness
            .barrier_pipeline
            .max_generation_issue_span_cycles,
        0
    );
    assert_eq!(
        result
            .readiness
            .barrier_pipeline
            .total_generation_arrival_span_cycles,
        6
    );
    assert_eq!(
        result
            .readiness
            .barrier_pipeline
            .max_generation_arrival_span_cycles,
        6
    );
    assert_eq!(result.readiness.cta_barriers.total_arrival_spread_cycles, 6);
    assert!(result.readiness.constraint_exposure_cycles.pipeline_barrier > 0);
    assert!(
        result
            .readiness
            .constraint_exposure_cycles
            .cta_barrier_rendezvous
            > 0
    );
}

#[test]
fn consecutive_generations_reuse_the_same_shared_barrier_service() {
    let sched = schedule(vec![
        (0, barrier_pipeline_slot(cb(1))),
        (16, barrier_pipeline_slot(cb(1))),
        (32, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: (0..4)
            .map(|warp_slot| warp(warp_slot, vec![inst(0), inst(16), inst(32)]))
            .collect(),
    }];
    let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();

    assert_eq!(result.instructions_issued, 12);
    assert_eq!(result.cycles, 19);
    assert_eq!(result.readiness.barrier_pipeline.instructions_issued, 8);
    assert_eq!(
        result.readiness.barrier_pipeline.issue_server_busy_cycles,
        16
    );
    assert_eq!(result.readiness.barrier_pipeline.generations_completed, 2);
    assert_eq!(result.readiness.barrier_pipeline.generation_rearm_events, 0);
    assert_eq!(
        result
            .readiness
            .barrier_pipeline
            .total_generation_issue_span_cycles,
        0
    );
    assert_eq!(
        result
            .readiness
            .barrier_pipeline
            .total_generation_arrival_span_cycles,
        12
    );
}

#[test]
fn forward_branch_resolution_extends_only_the_issuing_warps_next_frontier() {
    let branch_control = ControlBits::new(5, true, 7, 7, 0, 0).unwrap();
    let sched = schedule(vec![
        (0, branch_pipeline_slot(branch_control)),
        (16, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16)])],
    }];
    let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();

    assert_eq!(result.instructions_issued, 2);
    assert_eq!(result.cycles, 11);
    assert_eq!(result.readiness.schema_version(), 3);
    assert_eq!(
        result.readiness.control_pipeline.branch_instructions_issued,
        1
    );
    assert_eq!(
        result.readiness.control_pipeline.branch_resolution_cycles,
        10
    );
    assert_eq!(result.readiness.limiting_cycles.pipeline_branch, 10);
}
