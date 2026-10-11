use super::*;

#[test]
fn flat_async_copy_waits_for_source_completion_without_generic_sb0_latency() {
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
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert!(out.is_empty());
    assert_eq!(
        core.result().cycles,
        403,
        "wait must issue at source-ready t=400 plus one payload cycle"
    );
    assert_eq!(core.warps[0].barrier_clear[0], 0);
    let async_copy = core.async_copies[0]
        .state
        .as_deref()
        .expect("effective copy allocates protocol state");
    assert!(async_copy.open_group.is_none());
    assert!(async_copy.committed_groups.is_empty());
    assert_eq!(core.result().readiness.async_groups.groups_committed, 1);
    assert_eq!(core.result().readiness.async_groups.groups_waited, 1);
    assert_eq!(
        core.result()
            .readiness
            .async_groups
            .destination_local_limited_groups,
        1
    );
    assert_eq!(core.result().readiness.async_groups.source_cycles, 399);
    assert_eq!(
        core.result()
            .readiness
            .async_groups
            .destination_local_cycles,
        1
    );
    assert_eq!(
        core.result()
            .readiness
            .limiting_cycles
            .async_destination_local,
        400
    );
}

#[test]
fn async_wait_prefix_observation_separates_partial_and_draining_waits() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (16, async_commit_slot()),
        (32, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (48, async_commit_slot()),
        (64, async_wait_slot(1)),
        (80, async_wait_slot(0)),
        (96, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                async_copy_inst(0, 0, 1),
                inst(16),
                async_copy_inst(32, 1, 1),
                inst(48),
                inst(64),
                inst(80),
                inst(96),
            ],
            sectors: vec![40, 80],
        }],
    }];
    let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    let mut unobserved_core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    unobserved_core.readiness.async_wait_prefix = None;
    let mut out = Vec::new();
    assert_eq!(
        unobserved_core
            .advance(ReplayHorizon::Unbounded, &mut out)
            .unwrap(),
        Advance::Done
    );
    assert!(out.is_empty());
    let unobserved = unobserved_core.result();

    assert_eq!(result.cycles, 406);
    assert_eq!(result.cycles, unobserved.cycles);
    assert_eq!(result.instructions_issued, unobserved.instructions_issued);
    assert_eq!(
        result.readiness.async_groups,
        unobserved.readiness.async_groups
    );
    assert_eq!(
        result.readiness.limiting_cycles,
        unobserved.readiness.limiting_cycles
    );
    assert_eq!(
        result.readiness.constraint_exposure_cycles,
        unobserved.readiness.constraint_exposure_cycles
    );
    assert_eq!(result.readiness.schema_version(), 19);
    assert_eq!(
        result.readiness.async_wait_prefix.as_deref(),
        Some(&AsyncWaitPrefixDiagnostics {
            partial: AsyncWaitClassDiagnostics {
                wait_count: 1,
                immediate_sum: 1,
                committed_groups_before: 2,
                consumed_prefix_groups: 1,
                groups_remaining: 1,
                exposure_cycles: 397,
            },
            draining: AsyncWaitClassDiagnostics {
                wait_count: 1,
                immediate_sum: 0,
                committed_groups_before: 1,
                consumed_prefix_groups: 1,
                groups_remaining: 0,
                exposure_cycles: 3,
            },
        })
    );
}

#[test]
fn wait_zero_only_program_preserves_predecessor_readiness_json() {
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
    let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    let readiness_json = serde_json::to_value(&result.readiness).unwrap();

    assert_eq!(result.readiness.schema_version(), 1);
    assert!(result.readiness.async_wait_prefix.is_none());
    assert!(readiness_json.get("async_wait_prefix").is_none());
}

#[test]
fn async_issue_overlap_distinguishes_same_smsps_from_cross_smsp_mixed_routes() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
    let make_blocks = |warp_slots: [i32; 4]| {
        let policies = [
            LdgstsL1Policy::CacheAtAllLevels,
            LdgstsL1Policy::BypassL1,
            LdgstsL1Policy::CacheAtAllLevels,
            LdgstsL1Policy::BypassL1,
        ];
        vec![BlockReplay {
            launch_index: 0,
            warps: warp_slots
                .into_iter()
                .zip(policies)
                .enumerate()
                .map(|(index, (warp_slot, policy))| WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot,
                    insts: vec![async_copy_inst_with_policy(0, 0, 1, policy)],
                    sectors: vec![40 + index as i64],
                })
                .collect(),
        }]
    };

    let same_smsp = simulate_sm(&make_blocks([0, 4, 8, 12]), &sched, &params(), None).unwrap();
    let cross_smsp = simulate_sm(&make_blocks([0, 1, 2, 3]), &sched, &params(), None).unwrap();

    assert_eq!(same_smsp.instructions_issued, 4);
    assert_eq!(same_smsp.cycles, 4);
    assert!(same_smsp.readiness.async_issue_overlap.is_none());
    assert_eq!(same_smsp.readiness.schema_version(), 1);
    assert!(
        serde_json::to_value(&same_smsp.readiness)
            .unwrap()
            .get("async_issue_overlap")
            .is_none(),
        "serialized predecessor shape must omit an empty observation"
    );

    assert_eq!(cross_smsp.instructions_issued, 4);
    assert_eq!(cross_smsp.cycles, 1);
    assert_eq!(cross_smsp.readiness.schema_version(), 20);
    assert_eq!(
        cross_smsp.readiness.async_issue_overlap.as_deref(),
        Some(&AsyncIssueOverlapDiagnostics {
            coissue_cycles: 1,
            coissued_copies: 4,
            cache_all_coissued_copies: 2,
            bypass_coissued_copies: 2,
            mixed_policy_coissue_cycles: 1,
            single_acceptor_excess_copies: 3,
            maximum_coissue_width: 4,
            copies_with_other_warp_committed_groups: 0,
            other_warp_committed_groups_sum: 0,
            maximum_other_warp_committed_groups: 0,
        })
    );
}

#[test]
fn dual_issue_does_not_erase_same_warp_control_stalls() {
    let sched = schedule(vec![
        (0, slot(cb(7), OpClass::Other, "NOP")),
        (16, slot(cb(1), OpClass::Other, "NOP")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16)])],
    }];
    let mut p = params();
    p.dual_issue_supported = true;
    let result = simulate_sm(&blocks, &sched, &p, None).unwrap();
    assert_eq!(result.instructions_issued, 2);
    assert_eq!(result.cycles, 8);
}

#[test]
fn extra_issue_slot_serves_independently_ready_warps() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "NOP"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)]), warp(4, vec![inst(0)])],
    }];
    for dual in [false, true] {
        let mut p = params();
        p.dual_issue_supported = dual;
        let result = simulate_sm(&blocks, &sched, &p, None).unwrap();
        assert_eq!(result.instructions_issued, 2);
        assert_eq!(result.cycles, if dual { 1 } else { 2 });
    }
}

#[test]
fn sfu_readiness_does_not_increment_tensor_and_aggregates() {
    let mut local = ReadinessCauseCycles::default();
    local.add_cause(ReadinessCause::PipelineSfu, 3).unwrap();
    assert_eq!(local.pipeline_tensor, 0);
    let mut total = ReadinessCauseCycles::default();
    total.add_checked(&local).unwrap();
    assert_eq!(total.pipeline_sfu, 3);
    assert_eq!(serde_json::to_value(total).unwrap()["pipeline_sfu"], 3);
    let mut readiness = ReadinessDiagnostics::default();
    readiness.limiting_cycles = total;
    assert_eq!(readiness.schema_version(), 26);
}

#[cfg(feature = "reference")]
#[test]
fn unprojected_sfu_availability_prevents_exact_continuation() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "NOP"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let p = params();
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    assert!(
        !core
            .typed_unprojected_timing_state()
            .contains(&"sfu_pipeline_state".to_owned())
    );
    core.sfu_pipe_ready[0] = 4;
    assert!(
        core.typed_unprojected_timing_state()
            .contains(&"sfu_pipeline_state".to_owned())
    );
}

#[test]
fn async_issue_overlap_counts_only_other_live_warps_committed_groups() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
    let policies = [
        LdgstsL1Policy::CacheAtAllLevels,
        LdgstsL1Policy::CacheAtAllLevels,
        LdgstsL1Policy::CacheAtAllLevels,
        LdgstsL1Policy::BypassL1,
    ];
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: policies
            .into_iter()
            .enumerate()
            .map(|(warp_slot, policy)| WarpReplay {
                async_copy_cohorts: None,
                warp_slot: warp_slot as i32,
                insts: vec![async_copy_inst_with_policy(0, 0, 1, policy)],
                sectors: vec![40 + warp_slot as i64],
            })
            .collect(),
    }];
    let p = params();
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    for (warp, groups) in [(0, 2), (1, 1), (2, 2), (3, 2)] {
        for _ in 0..groups {
            core.async_copies[warp]
                .add_known_async_copy(warp, 0, 64, 0, &p)
                .unwrap();
            assert!(core.async_copies[warp].commit_async_copy_group(0));
        }
    }
    core.warps[0].eligible = u64::MAX;
    core.warps[1].eligible = u64::MAX;
    core.warps[2].eligible = u64::MAX;

    let mut requests = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Bounded(1), &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: Some(1) }
    );
    assert!(requests.is_empty());
    assert_eq!(core.instructions_issued(), 1);
    assert_eq!(core.now(), 1);
    assert_eq!(core.result().readiness.schema_version(), 20);
    assert_eq!(
        core.result().readiness.async_issue_overlap.as_deref(),
        Some(&AsyncIssueOverlapDiagnostics {
            coissue_cycles: 0,
            coissued_copies: 0,
            cache_all_coissued_copies: 0,
            bypass_coissued_copies: 0,
            mixed_policy_coissue_cycles: 0,
            single_acceptor_excess_copies: 0,
            maximum_coissue_width: 0,
            copies_with_other_warp_committed_groups: 1,
            // Warp 3's own two retained groups are excluded; its three
            // siblings own five committed groups.
            other_warp_committed_groups_sum: 5,
            maximum_other_warp_committed_groups: 5,
        })
    );
}

#[test]
fn async_copy_payload_rate_is_route_local_and_sector_count_independent() {
    for (policy, opcode, expected_cycles) in [
        (LdgstsL1Policy::CacheAtAllLevels, "LDGSTS.E", 406),
        (LdgstsL1Policy::BypassL1, "LDGSTS.E.BYPASS", 410),
    ] {
        for len in [1_u16, 16] {
            let sched = schedule(vec![
                (0, slot(cb(1), OpClass::Other, opcode)),
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
                        ReplayInst {
                            lane_mask: u32::MAX,
                            pc: 0,
                            kind: AccessKind::GlobalAsyncCopy {
                                off: 0,
                                len,
                                completion_work: if policy == LdgstsL1Policy::CacheAtAllLevels {
                                    256
                                } else {
                                    512
                                },
                                l1_policy: policy,
                            },
                        },
                        inst(16),
                        inst(32),
                        inst(48),
                    ],
                    sectors: (0..len).map(i64::from).collect(),
                }],
            }];

            let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
            assert_eq!(
                result.cycles, expected_cycles,
                "policy={policy:?}, sectors={len}"
            );
        }
    }
}

#[test]
fn exact_destination_wavefronts_serialize_across_sm_subcores() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (16, async_commit_slot()),
    ]);
    let completion_work = sim_mem::encode_async_copy_destination_work(512).unwrap();
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: (0..4)
            .map(|warp_slot| WarpReplay {
                async_copy_cohorts: None,
                warp_slot,
                insts: vec![
                    ReplayInst {
                        lane_mask: u32::MAX,
                        pc: 0,
                        kind: AccessKind::GlobalAsyncCopy {
                            off: 0,
                            len: 1,
                            completion_work,
                            l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                        },
                    },
                    inst(16),
                ],
                sectors: vec![40],
            })
            .collect(),
    }];
    let mut p = params();
    p.global_flat_latency = 0;
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    let mut requests = Vec::new();

    assert!(core.issue_cycle(&mut requests).unwrap());
    assert!(requests.is_empty());
    assert_eq!(core.ldgsts_destination_pipe_ready, 0);
    core.t = 1;
    assert!(core.issue_cycle(&mut requests).unwrap());
    core.enqueue_ready_async_copy_destinations().unwrap();
    assert_eq!(core.ldgsts_destination_pipe_ready, 24);
    let destination_ready = core
        .async_copies
        .iter()
        .map(|copy| {
            copy.state
                .as_deref()
                .unwrap()
                .committed_groups
                .front()
                .copied()
                .unwrap()
                .destination_completion_time
        })
        .collect::<Vec<_>>();
    assert_eq!(destination_ready, vec![6, 12, 18, 24]);
    let completion_ready = core
        .async_copies
        .iter()
        .map(|copy| {
            copy.state
                .as_deref()
                .unwrap()
                .committed_groups
                .front()
                .copied()
                .unwrap()
                .completion_time
        })
        .collect::<Vec<_>>();
    assert_eq!(completion_ready, vec![8, 12, 18, 24]);
}

#[test]
fn shared_mode_attaches_exact_destination_work_once() {
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
                    completion_work: sim_mem::encode_async_copy_destination_work(128).unwrap(),
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                },
            }],
            sectors: vec![40],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();

    assert!(core.issue_cycle(&mut requests).unwrap());
    let group = core.async_copies[0]
        .state
        .as_deref()
        .unwrap()
        .open_group
        .unwrap();
    assert_eq!(group.destination_wavefronts, 1);
    assert_eq!(group.delivery_work, 128);
    assert_eq!(group.pending_sources, 1);
    assert_eq!(requests.len(), 1);
}

#[test]
fn exact_destination_service_waits_for_actual_source_time() {
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
            insts: vec![
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalAsyncCopy {
                        off: 0,
                        len: 1,
                        completion_work: sim_mem::encode_async_copy_destination_work(128).unwrap(),
                        l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    },
                },
                inst(16),
                inst(32),
                inst(48),
            ],
            sectors: vec![40],
        }],
    }];
    let mut p = params();
    p.global_flat_latency = 100;
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Bounded(101), &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: Some(101) }
    );
    assert_eq!(core.ldgsts_destination_pipe_ready, 102);
    let group = core.async_copies[0]
        .state
        .as_deref()
        .unwrap()
        .committed_groups
        .front()
        .unwrap();
    assert!(group.destination_service_queued);
    assert_eq!(group.destination_completion_time, 102);
    assert_eq!(
        core.warps[0].next, 2,
        "wait issued before service completed"
    );

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    assert_eq!(core.result().cycles, 104);
}

#[test]
fn exact_destination_pipe_overflow_is_transactional() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (16, async_commit_slot()),
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
                    kind: AccessKind::GlobalAsyncCopy {
                        off: 0,
                        len: 1,
                        completion_work: sim_mem::encode_async_copy_destination_work(128).unwrap(),
                        l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                    },
                },
                inst(16),
            ],
            sectors: vec![40],
        }],
    }];
    let mut p = params();
    p.global_flat_latency = 0;
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    let mut requests = Vec::new();

    assert!(core.issue_cycle(&mut requests).unwrap());
    core.t = 1;
    assert!(core.issue_cycle(&mut requests).unwrap());
    core.ldgsts_destination_pipe_ready = u64::MAX;

    assert!(matches!(
        core.enqueue_ready_async_copy_destinations(),
        Err(IssueError::TimeOverflow {
            operation: "LDGSTS aggregate destination-wavefront service",
            base: u64::MAX,
            delta: 2,
        })
    ));
    assert_eq!(core.ldgsts_destination_pipe_ready, u64::MAX);
    assert_eq!(core.instructions_issued(), 2);
    assert_eq!(core.warps[0].next, 2);
    let group = core.async_copies[0]
        .state
        .as_deref()
        .unwrap()
        .committed_groups
        .front()
        .unwrap();
    assert!(!group.destination_service_queued);
    assert_eq!(group.destination_completion_time, 0);
    assert!(requests.is_empty());
}

#[test]
fn malformed_exact_destination_work_rejects_issue_before_mutation() {
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
                    completion_work: sim_mem::encode_async_copy_destination_work(130).unwrap(),
                    l1_policy: LdgstsL1Policy::CacheAtAllLevels,
                },
            }],
            sectors: vec![40],
        }],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut requests = Vec::new();

    assert!(matches!(
        core.issue_cycle(&mut requests),
        Err(IssueError::InvalidAsyncCopyDestinationWork {
            completion_work: 130,
            work_per_wavefront: 128,
        })
    ));
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert!(core.async_copies[0].state.is_none());
    assert!(requests.is_empty());
}

#[test]
fn async_copy_group_accumulates_each_copy_payload_once() {
    for (policy, opcode, expected_cycles) in [
        (LdgstsL1Policy::CacheAtAllLevels, "LDGSTS.E", 411),
        (LdgstsL1Policy::BypassL1, "LDGSTS.E.BYPASS", 419),
    ] {
        let sched = schedule(vec![
            (0, slot(cb(1), OpClass::Other, opcode)),
            (16, slot(cb(1), OpClass::Other, opcode)),
            (32, async_commit_slot()),
            (48, async_wait_slot(0)),
            (64, slot(cb(1), OpClass::Other, "EXIT")),
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
                        kind: AccessKind::GlobalAsyncCopy {
                            off: 0,
                            len: 1,
                            completion_work: if policy == LdgstsL1Policy::CacheAtAllLevels {
                                256
                            } else {
                                512
                            },
                            l1_policy: policy,
                        },
                    },
                    ReplayInst {
                        lane_mask: u32::MAX,
                        pc: 16,
                        kind: AccessKind::GlobalAsyncCopy {
                            off: 1,
                            len: 16,
                            completion_work: if policy == LdgstsL1Policy::CacheAtAllLevels {
                                256
                            } else {
                                512
                            },
                            l1_policy: policy,
                        },
                    },
                    inst(32),
                    inst(48),
                    inst(64),
                ],
                sectors: (0_i64..17).collect(),
            }],
        }];

        let result = simulate_sm(&blocks, &sched, &params(), None).unwrap();
        assert_eq!(result.cycles, expected_cycles, "policy={policy:?}");
    }
}

#[test]
fn async_copy_completion_work_mismatch_rejects_issue_before_mutation() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E.BYPASS"))]);
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
                    completion_work: 0,
                    l1_policy: LdgstsL1Policy::BypassL1,
                },
            }],
            sectors: vec![40],
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let before = *core.l1_counters().unwrap();
    let mut requests = Vec::new();

    let error = core
        .advance(ReplayHorizon::Unbounded, &mut requests)
        .unwrap_err();
    assert!(matches!(
        error,
        IssueError::AsyncCopyCompletionWorkMismatch {
            warp: 0,
            sectors: 1,
            completion_work: 0,
        }
    ));
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert!(core.async_copies[0].state.is_none());
    assert_eq!(*core.l1_counters().unwrap(), before);
    assert!(requests.is_empty());
}

#[test]
fn bypass_topology_only_zero_fill_keeps_async_group() {
    let completion_work = sim_mem::encode_async_copy_payload_and_topology(0, 1, 1, 1, 1)
        .expect("topology-only bypass token");
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
                ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalAsyncCopy {
                        off: 0,
                        len: 0,
                        completion_work,
                        l1_policy: LdgstsL1Policy::BypassL1,
                    },
                },
                inst(16),
                inst(32),
                inst(48),
            ],
            sectors: Vec::new(),
        }],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let mut requests = Vec::new();
    assert!(core.issue_cycle(&mut requests).unwrap());
    let group = core.async_copies[0]
        .state
        .as_deref()
        .and_then(|state| state.open_group)
        .expect("zero-fill topology must open an async group");
    assert_eq!(group.pending_sources, 0);
    assert_eq!(group.delivery_work, 0);
    assert!(requests.is_empty());
    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
}

#[test]
fn flat_async_copy_idle_skips_directly_to_source_completion() {
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
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Bounded(402), &mut out).unwrap(),
        Advance::Idle { wake_at: Some(402) }
    );
    assert_eq!(core.now(), 402);
    assert_eq!(core.instructions_issued(), 3);
    assert_eq!(core.warps[0].next, 3, "wait must issue at t=401");

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(core.result().cycles, 403);
}

#[test]
fn another_warp_fills_an_async_copy_wait_gap() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (16, async_commit_slot()),
        (32, async_wait_slot(0)),
        (48, slot(cb(1), OpClass::Other, "EXIT")),
        (64, slot(cb(1), OpClass::Other, "IADD3")),
        (80, slot(cb(1), OpClass::Other, "IADD3")),
        (96, slot(cb(1), OpClass::Other, "IADD3")),
        (112, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![
            warp(0, vec![inst(64), inst(80), inst(96), inst(112)]),
            WarpReplay {
                async_copy_cohorts: None,
                // Same SMSP and younger, so this warp is held after t=0.
                warp_slot: 4,
                insts: vec![async_copy_inst(0, 0, 1), inst(16), inst(32), inst(48)],
                sectors: vec![40],
            },
        ],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Bounded(6), &mut out).unwrap(),
        Advance::Idle { wake_at: Some(6) }
    );
    assert!(core.warps[0].done, "ready warp did not fill the wait gap");
    assert_eq!(core.warps[1].next, 2, "async wait issued before t=400");
    assert_eq!(core.instructions_issued(), 6);

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(core.instructions_issued(), 8);
    assert_eq!(core.result().cycles, 403);
}

#[test]
fn constant_l1_async_copy_uses_eligible_source_route() {
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

    let result = simulate_sm(&blocks, &sched, &params(), Some(l1_config())).unwrap();
    assert_eq!(
        result.cycles, 247,
        "wait must issue at source-ready t=244 plus one payload cycle"
    );
    let counters = result.l1.unwrap();
    assert_eq!(counters.bypass_sectors, 0);
    assert_eq!(counters.misses, 1);
}

#[test]
fn wait_one_preserves_newest_group_for_a_later_wait_zero() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (16, async_commit_slot()),
        (32, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (48, async_commit_slot()),
        (64, async_wait_slot(1)),
        (80, async_wait_slot(0)),
        (96, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![
                async_copy_inst(0, 0, 1),
                inst(16),
                async_copy_inst(32, 1, 1),
                inst(48),
                inst(64),
                inst(80),
                inst(96),
            ],
            sectors: vec![40, 44],
        }],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Bounded(402), &mut out).unwrap(),
        Advance::Idle { wake_at: Some(402) }
    );
    assert_eq!(core.warps[0].next, 5, "wait-1 must issue at t=401");
    let async_copy = core.async_copies[0].state.as_deref().unwrap();
    assert_eq!(async_copy.committed_groups.len(), 1);
    assert_eq!(async_copy.committed_groups[0].completion_time, 404);

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(core.result().cycles, 406);
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
fn empty_async_copy_commit_is_trivially_complete_and_consumes_no_sequence() {
    let sched = schedule(vec![
        (0, async_commit_slot()),
        (16, async_wait_slot(0)),
        (32, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16), inst(32)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(
        core.result().cycles,
        4,
        "the commit yield bit delays wait by one cycle, not by 25"
    );
    assert!(
        core.async_copies[0].state.is_none(),
        "an empty commit must not allocate or consume protocol state"
    );
    assert_eq!(core.warps[0].barrier_clear[0], 0);
}

#[test]
fn async_copy_group_sequence_exhaustion_is_nonmutating() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "LDGSTS.E")),
        (16, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![async_copy_inst(0, 0, 1), inst(16)],
            sectors: vec![40],
        }],
    }];
    let mut core =
        SmCore::new(&blocks, &sched, &params(), SmMemMode::L1Const(l1_config())).unwrap();
    core.async_copies[0].state = Some(Box::new(AsyncCopyState {
        next_sequence: None,
        ..AsyncCopyState::new()
    }));
    let before = *core.l1_counters().unwrap();
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::AsyncCopyGroupSequenceExhausted { warp: 0 })
    ));
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.now(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert_eq!(*core.l1_counters().unwrap(), before);
    assert!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .open_group
            .is_none()
    );
    assert!(out.is_empty());
}

#[test]
fn committed_async_groups_preserve_fifo_wait_visibility() {
    let sched = schedule(vec![(0, async_wait_slot(1))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();

    let p = params();
    let older = core.async_copies[0]
        .add_pending_async_copy_sources(0, 1, 0, 0, None, &p)
        .unwrap();
    assert!(core.async_copies[0].commit_async_copy_group(1));
    let newer = core.async_copies[0]
        .add_pending_async_copy_sources(0, 1, 0, 0, None, &p)
        .unwrap();
    assert!(core.async_copies[0].commit_async_copy_group(2));

    core.async_copies[0]
        .complete_async_copy_sources(0, newer, None, 1, 50, &p)
        .unwrap();
    assert_eq!(
        core.async_copies[0].async_wait_clear_time(1),
        None,
        "a newer completion cannot make the required older prefix visible"
    );

    core.async_copies[0]
        .complete_async_copy_sources(0, older, None, 1, 100, &p)
        .unwrap();
    assert_eq!(core.async_copies[0].async_wait_clear_time(1), Some(100));
    core.async_copies[0].consume_async_wait(1, 100);
    assert_eq!(
        core.async_copies[0]
            .state
            .as_deref()
            .unwrap()
            .committed_groups
            .iter()
            .map(|group| group.id)
            .collect::<Vec<_>>(),
        vec![newer]
    );
}

#[test]
fn bypass_copy_observation_preserves_individual_source_frontiers() {
    let p = params();
    let mut slot = AsyncCopySlot::default();
    let group = slot
        .add_pending_async_copy_sources(
            0,
            2,
            64,
            0,
            Some(BypassAsyncCopyInput {
                instruction: 10,
                sector_off: 0,
                sector_len: 2,
                source_cohorts: 8,
                destination_lines: 8,
                max_bank_wavefronts: 1,
                max_bypass_wavefronts: 1,
                cohorts: None,
                sector_membership_masks: None,
            }),
            &p,
        )
        .unwrap();
    assert_eq!(
        slot.add_pending_async_copy_sources(
            0,
            2,
            64,
            0,
            Some(BypassAsyncCopyInput {
                instruction: 20,
                sector_off: 2,
                sector_len: 2,
                source_cohorts: 8,
                destination_lines: 16,
                max_bank_wavefronts: 2,
                max_bypass_wavefronts: 4,
                cohorts: None,
                sector_membership_masks: None,
            }),
            &p,
        )
        .unwrap(),
        group
    );
    assert!(slot.commit_async_copy_group(10));

    slot.complete_async_copy_sources(
        0,
        group,
        Some(ResponseId::new(ResponseOwner(1), 0, 10, 0)),
        1,
        30,
        &p,
    )
    .unwrap();
    slot.complete_async_copy_sources(
        0,
        group,
        Some(ResponseId::new(ResponseOwner(1), 0, 10, 1)),
        1,
        40,
        &p,
    )
    .unwrap();
    slot.complete_async_copy_sources(
        0,
        group,
        Some(ResponseId::new(ResponseOwner(1), 0, 20, 2)),
        1,
        50,
        &p,
    )
    .unwrap();
    slot.complete_async_copy_sources(
        0,
        group,
        Some(ResponseId::new(ResponseOwner(1), 0, 20, 3)),
        1,
        70,
        &p,
    )
    .unwrap();

    let committed = slot.state.as_deref().unwrap().committed_groups[0];
    let projected = slot
        .bypass_copy_projection(0, &committed, &p, 80)
        .unwrap()
        .unwrap();
    let per_copy = p
        .ldgsts_completion_ready(40, 64)
        .unwrap()
        .max(p.ldgsts_completion_ready(70, 64).unwrap());
    let aggregate = committed.local_completion_time(&p).unwrap();
    assert_eq!(projected.copies, 2);
    assert_eq!(projected.source_cohorts, 16);
    assert_eq!(projected.destination_lines, 24);
    assert_eq!(projected.max_bank_wavefronts_sum, 3);
    assert_eq!(projected.max_bypass_wavefronts_sum, 5);
    assert_eq!(projected.bank_wavefront_upper_bound, 24);
    assert_eq!(projected.bypass_wavefront_upper_bound, 40);
    assert_eq!(projected.source_spread_cycles, 30);
    assert_eq!(projected.per_copy_completion_cycles, per_copy - 10);
    assert_eq!(
        projected.group_aggregate_excess_cycles,
        aggregate - per_copy
    );
    assert_eq!(projected.topology_candidate_penalty_cycles, 14);
    assert_eq!(projected.topology_candidate_completion_cycles, 75);
    assert_eq!(projected.topology_candidate_later_cycles, 13);
    assert_eq!(projected.topology_candidate_earlier_cycles, 0);
    assert_eq!(projected.pipeline_q1_completion_cycles, 75);
    assert_eq!(projected.pipeline_q1_exposure_cycles, 5);
    assert_eq!(projected.pipeline_q2_completion_cycles, 75);
    assert_eq!(projected.pipeline_q2_exposure_cycles, 5);
    assert_eq!(projected.pipeline_q4_completion_cycles, 75);
    assert_eq!(projected.pipeline_q4_exposure_cycles, 5);
    assert_eq!(projected.pipeline_q8_completion_cycles, 75);
    assert_eq!(projected.pipeline_q8_exposure_cycles, 5);
    assert_eq!(projected.pipeline_q16_completion_cycles, 75);
    assert_eq!(projected.pipeline_q16_exposure_cycles, 5);
    assert_eq!(slot.async_wait_clear_time(0), Some(aggregate));

    slot.consume_async_wait(0, aggregate);
    let state = slot.state.as_deref().unwrap();
    assert!(state.committed_groups.is_empty());
    assert!(state.bypass_copies.is_empty());
}

#[cfg(feature = "reference")]
#[test]
fn typed_bypass_copy_preserves_native_response_state_with_optional_cohorts() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E.BYPASS"))]);
    for exact_cohorts in [false, true] {
        let sidecar = exact_cohorts.then(|| {
            Box::new(AsyncCopyCohortReplay {
                spans: vec![AsyncCopyCohortSpan {
                    instruction: 0,
                    sector_off: 0,
                    sector_len: 3,
                    cohort_off: 0,
                    cohort_len: 2,
                    membership_off: 0,
                }],
                cohorts: vec![
                    AsyncCopyCohortDescriptor {
                        payload_bytes: 32,
                        destination_lines: 1,
                        bank_wavefronts: 1,
                        bypass_wavefronts: 1,
                    },
                    AsyncCopyCohortDescriptor {
                        payload_bytes: 32,
                        destination_lines: 2,
                        bank_wavefronts: 2,
                        bypass_wavefronts: 3,
                    },
                ],
                sector_membership_masks: vec![0b01, 0b11, 0b10],
                instrumentation_order: Vec::new(),
            })
        });
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: sidecar,
                warp_slot: 0,
                insts: vec![ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalAsyncCopy {
                        off: 0,
                        len: 3,
                        completion_work: sim_mem::encode_async_copy_payload_and_topology(
                            128, 2, 3, 2, 3,
                        )
                        .unwrap(),
                        l1_policy: LdgstsL1Policy::BypassL1,
                    },
                }],
                sectors: vec![40, 41, 42],
            }],
        }];
        let mut core = shared_core(&blocks, &sched, &params());
        let mut requests = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut requests)
                .unwrap(),
            Advance::Done
        );
        assert_eq!(requests.len(), 3);
        // The warp has retired, but responses and their source sectors
        // remain addressable by dynamic warp/instruction/sector identity.
        let owners = core.typed_l2_request_owners(0, &[]);
        for (sector, owner) in owners.iter().enumerate() {
            assert_eq!(
                owner.line,
                CanonicalLineKey::VirtualAbsolute { line_id: 10 }
            );
            assert_eq!(owner.sector_mask, 1 << sector);
        }
        for completed in 0..=3 {
            let projected = core.typed_async_copy_states().remove(0).unwrap();
            assert_eq!(
                projected.validate(),
                Ok(()),
                "exact_cohorts={exact_cohorts}, completed={completed}"
            );
            assert_eq!(projected.bypass_observations_complete, exact_cohorts);
            let copy = &projected.bypass_observations[0];
            assert_eq!(copy.pending_sources, 3 - completed as u32);
            assert_eq!(copy.source_cohorts, 2);
            assert_eq!(copy.cohorts.len(), if exact_cohorts { 2 } else { 0 });
            if completed < 3 {
                let index = [2, 0, 1][completed];
                core.on_response(
                    requests[index].reply,
                    300 + completed as u64,
                    &mut Vec::new(),
                )
                .unwrap();
            }
        }
        assert_eq!(core.outstanding_responses(), 0);
    }
}

#[test]
fn bypass_cohort_reservation_score_reproduces_negative_swapped_pairing() {
    let (slow_expensive, expensive_wait) = paired_reservation_projection(true);
    let (slow_cheap, cheap_wait) = paired_reservation_projection(false);

    assert_eq!(
        expensive_wait, cheap_wait,
        "observation must not alter timing"
    );
    assert_eq!(slow_expensive.cohort_response_envelope_cycles, 20);
    assert_eq!(slow_cheap.cohort_response_envelope_cycles, 20);
    assert_eq!(slow_expensive.cohort_topology_prearrival_hidden_cycles, 18);
    assert_eq!(slow_cheap.cohort_topology_prearrival_hidden_cycles, 0);
    assert_eq!(slow_expensive.cohort_topology_stream_hidden_cycles, 0);
    assert_eq!(slow_cheap.cohort_topology_stream_hidden_cycles, 0);
    assert_eq!(slow_expensive.cohort_topology_residual_exposure_cycles, 0);
    assert_eq!(slow_cheap.cohort_topology_residual_exposure_cycles, 18);
    let paired_score = i64::try_from(slow_expensive.cohort_topology_residual_exposure_cycles)
        .unwrap()
        - i64::try_from(slow_cheap.cohort_topology_residual_exposure_cycles).unwrap();
    assert_eq!(paired_score, -18);
}

#[test]
fn bypass_cohort_observation_handles_out_of_order_and_multi_cohort_responses() {
    let p = params();
    let descriptors = [
        AsyncCopyCohortDescriptor {
            payload_bytes: 32,
            destination_lines: 1,
            bank_wavefronts: 1,
            bypass_wavefronts: 1,
        },
        AsyncCopyCohortDescriptor {
            payload_bytes: 32,
            destination_lines: 2,
            bank_wavefronts: 2,
            bypass_wavefronts: 3,
        },
    ];
    let masks = [0b01, 0b11, 0b10];
    let mut slot = AsyncCopySlot::default();
    let group = slot
        .add_pending_async_copy_sources(
            0,
            3,
            128,
            0,
            Some(exact_two_cohort_input(&descriptors, &masks)),
            &p,
        )
        .unwrap();
    assert!(slot.commit_async_copy_group(10));

    // Sector identities complete 2, 0, 1. The final sector belongs to
    // both cohorts but decrements the production group/copy only once.
    for (sector_index, now) in [(22, 30), (20, 40), (21, 50)] {
        slot.complete_async_copy_sources(
            0,
            group,
            Some(ResponseId::new(ResponseOwner(1), 0, 10, sector_index)),
            1,
            now,
            &p,
        )
        .unwrap();
    }
    let state = slot.state.as_deref().unwrap();
    let committed = state.committed_groups[0];
    let copy = &state.bypass_copies[0];
    assert_eq!(committed.pending_sources, 0);
    assert_eq!(copy.pending_sources, 0);
    assert_eq!(copy.cohorts[0].pending_sources, 0);
    assert_eq!(copy.cohorts[0].first_source_response, Some(40));
    assert_eq!(copy.cohorts[0].last_source_response, Some(50));
    assert_eq!(copy.cohorts[1].pending_sources, 0);
    assert_eq!(copy.cohorts[1].first_source_response, Some(30));
    assert_eq!(copy.cohorts[1].last_source_response, Some(50));

    let projected = slot
        .bypass_copy_projection(0, &committed, &p, 45)
        .unwrap()
        .unwrap();
    assert_eq!(projected.cohort_groups_observed, 1);
    assert_eq!(projected.cohort_copies_observed, 1);
    assert_eq!(projected.cohort_response_memberships_observed, 4);
    assert_eq!(projected.cohort_response_span_cycles, 30);
    assert_eq!(projected.cohort_topology_penalty_cycles, 10);
    assert_eq!(projected.cohort_candidate_completion_cycles, 41);
    assert_eq!(projected.cohort_candidate_exposure_cycles, 6);
    assert_eq!(projected.cohort_candidate_later_than_per_copy_cycles, 0);
    assert_eq!(projected.cohort_candidate_earlier_than_per_copy_cycles, 1);
    assert_eq!(
        projected.cohort_candidate_later_than_group_aggregate_cycles,
        0
    );
    assert_eq!(
        projected.cohort_candidate_earlier_than_group_aggregate_cycles,
        1
    );
    assert_eq!(projected.cohort_candidate_later_than_production_cycles, 0);
    assert_eq!(projected.cohort_candidate_earlier_than_production_cycles, 1);
    assert_eq!(projected.cohort_overlap_groups_observed, 1);
    assert_eq!(projected.cohort_response_envelope_cycles, 20);
    assert_eq!(projected.cohort_topology_prearrival_hidden_cycles, 0);
    assert_eq!(projected.cohort_topology_stream_hidden_cycles, 10);
    assert_eq!(projected.cohort_topology_residual_exposure_cycles, 0);
    assert_eq!(slot.async_wait_clear_time(0), Some(52));
}

#[test]
fn bypass_cohort_projection_rejects_reversed_response_times_in_release() {
    let p = params();
    let descriptors = [AsyncCopyCohortDescriptor {
        payload_bytes: 32,
        destination_lines: 1,
        bank_wavefronts: 1,
        bypass_wavefronts: 1,
    }];
    let masks = [0b1];
    let mut slot = AsyncCopySlot::default();
    let group = slot
        .add_pending_async_copy_sources(
            0,
            1,
            64,
            0,
            Some(BypassAsyncCopyInput {
                instruction: 10,
                sector_off: 20,
                sector_len: 1,
                source_cohorts: 1,
                destination_lines: 1,
                max_bank_wavefronts: 1,
                max_bypass_wavefronts: 1,
                cohorts: Some(&descriptors),
                sector_membership_masks: Some(&masks),
            }),
            &p,
        )
        .unwrap();
    assert!(slot.commit_async_copy_group(10));
    slot.complete_async_copy_sources(
        0,
        group,
        Some(ResponseId::new(ResponseOwner(1), 0, 10, 20)),
        1,
        40,
        &p,
    )
    .unwrap();
    let committed = {
        let state = slot.state.as_deref_mut().unwrap();
        state.bypass_copies[0].cohorts[0].first_source_response = Some(60);
        state.bypass_copies[0].cohorts[0].last_source_response = Some(50);
        state.committed_groups[0]
    };

    assert!(matches!(
        slot.bypass_copy_projection(0, &committed, &p, 60),
        Err(IssueError::AsyncCopyCohortProjectionInvalid {
            field: "cohort response order",
            expected: 60,
            got: 50,
            ..
        })
    ));
}

#[test]
fn bypass_cohort_validation_is_transactional_at_issue_and_response() {
    let p = params();
    let descriptors = [
        AsyncCopyCohortDescriptor {
            payload_bytes: 32,
            destination_lines: 1,
            bank_wavefronts: 1,
            bypass_wavefronts: 1,
        },
        AsyncCopyCohortDescriptor {
            payload_bytes: 32,
            destination_lines: 2,
            bank_wavefronts: 2,
            bypass_wavefronts: 3,
        },
    ];
    let mut slot = AsyncCopySlot::default();
    let invalid_masks = [0b01, 0b100, 0b10];
    assert!(matches!(
        slot.add_pending_async_copy_sources(
            0,
            3,
            128,
            0,
            Some(exact_two_cohort_input(&descriptors, &invalid_masks)),
            &p,
        ),
        Err(IssueError::AsyncCopyCohortProjectionInvalid {
            field: "source-sector cohort mask",
            ..
        })
    ));
    assert!(slot.state.is_none());

    let masks = [0b01, 0b11, 0b10];
    let group = slot
        .add_pending_async_copy_sources(
            0,
            3,
            128,
            0,
            Some(exact_two_cohort_input(&descriptors, &masks)),
            &p,
        )
        .unwrap();
    let before_group = slot.state.as_deref().unwrap().open_group.unwrap();
    let before_copy = slot.state.as_deref().unwrap().bypass_copies[0].clone();
    assert!(matches!(
        slot.complete_async_copy_sources(
            0,
            group,
            Some(ResponseId::new(ResponseOwner(1), 0, 10, 99)),
            1,
            30,
            &p,
        ),
        Err(IssueError::AsyncCopyCohortSectorOutOfRange { .. })
    ));
    assert_eq!(
        slot.state.as_deref().unwrap().open_group,
        Some(before_group)
    );
    assert_eq!(slot.state.as_deref().unwrap().bypass_copies[0], before_copy);
}

#[test]
fn bypass_pipeline_candidate_tracks_queue_and_idle_arrivals() {
    let mut copies = vec![
        BypassPipelineCopyCandidate {
            instruction: 20,
            payload_completion: 20,
            topology_penalty: 4,
        },
        BypassPipelineCopyCandidate {
            instruction: 30,
            payload_completion: 22,
            topology_penalty: 6,
        },
        BypassPipelineCopyCandidate {
            instruction: 10,
            payload_completion: 20,
            topology_penalty: 8,
        },
    ];

    assert_eq!(
        bypass_pipeline_candidate_completions(&mut copies).unwrap(),
        [38, 32, 29, 28, 28]
    );
    assert_eq!(
        copies
            .iter()
            .map(|copy| (copy.payload_completion, copy.instruction))
            .collect::<Vec<_>>(),
        [(20, 10), (20, 20), (22, 30)]
    );

    copies.push(BypassPipelineCopyCandidate {
        instruction: 40,
        payload_completion: 100,
        topology_penalty: 2,
    });
    assert_eq!(
        bypass_pipeline_candidate_completions(&mut copies).unwrap(),
        [102; BYPASS_PIPELINE_CANDIDATE_Q.len()]
    );

    let observation = ReadinessDiagnostics {
        bypass_async_copies: Some(Box::new(BypassAsyncCopyDiagnostics {
            copies_observed: 3,
            pipeline_groups_observed: 1,
            pipeline_q1_completion_cycles: 38,
            pipeline_q1_exposure_cycles: 18,
            cohort_groups_observed: 1,
            cohort_copies_observed: 3,
            cohort_response_memberships_observed: 12,
            cohort_candidate_completion_cycles: 30,
            cohort_candidate_exposure_cycles: 10,
            cohort_overlap_groups_observed: 1,
            cohort_response_envelope_cycles: 24,
            cohort_topology_prearrival_hidden_cycles: 8,
            cohort_topology_stream_hidden_cycles: 4,
            // Schema 18 is selected by observed groups, not a nonzero
            // residual score.
            cohort_topology_residual_exposure_cycles: 0,
            ..BypassAsyncCopyDiagnostics::default()
        })),
        ..ReadinessDiagnostics::default()
    };
    assert_eq!(observation.schema_version(), 18);
    let mut aggregate = ReadinessDiagnostics::default();
    aggregate.add_checked(&observation).unwrap();
    aggregate.add_checked(&observation).unwrap();
    let aggregate_pipeline = aggregate.bypass_async_copies.as_deref().unwrap();
    assert_eq!(aggregate_pipeline.pipeline_groups_observed, 2);
    assert_eq!(aggregate_pipeline.pipeline_q1_completion_cycles, 76);
    assert_eq!(aggregate_pipeline.pipeline_q1_exposure_cycles, 36);
    assert_eq!(aggregate_pipeline.cohort_groups_observed, 2);
    assert_eq!(aggregate_pipeline.cohort_copies_observed, 6);
    assert_eq!(aggregate_pipeline.cohort_response_memberships_observed, 24);
    assert_eq!(aggregate_pipeline.cohort_candidate_completion_cycles, 60);
    assert_eq!(aggregate_pipeline.cohort_candidate_exposure_cycles, 20);
    assert_eq!(aggregate_pipeline.cohort_overlap_groups_observed, 2);
    assert_eq!(aggregate_pipeline.cohort_response_envelope_cycles, 48);
    assert_eq!(
        aggregate_pipeline.cohort_topology_prearrival_hidden_cycles,
        16
    );
    assert_eq!(aggregate_pipeline.cohort_topology_stream_hidden_cycles, 8);
    assert_eq!(
        aggregate_pipeline.cohort_topology_residual_exposure_cycles,
        0
    );

    aggregate
        .bypass_async_copies
        .as_deref_mut()
        .unwrap()
        .cohort_topology_residual_exposure_cycles = u64::MAX;
    let before = aggregate.clone();
    let overflow = ReadinessDiagnostics {
        bypass_async_copies: Some(Box::new(BypassAsyncCopyDiagnostics {
            cohort_topology_residual_exposure_cycles: 1,
            ..BypassAsyncCopyDiagnostics::default()
        })),
        ..ReadinessDiagnostics::default()
    };
    assert!(matches!(
        aggregate.add_checked(&overflow),
        Err(ReadinessCounterError::Overflow {
            field: "readiness.bypass_async_copies.cohort_topology_residual_exposure_cycles",
            accumulated: u64::MAX,
            addend: 1,
        })
    ));
    assert_eq!(aggregate, before);
}

#[test]
fn async_copy_pending_overflow_rejects_issue_without_mutation() {
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
    let group = core.async_copies[0]
        .add_pending_async_copy_sources(0, u32::MAX, 0, 0, None, &params())
        .unwrap();
    let before_group = core.async_copies[0].state.as_deref().unwrap().open_group;
    let before_l1 = *core.l1_counters().unwrap();
    let mut requests = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut requests),
        Err(IssueError::AsyncCopyPendingOverflow {
            warp: 0,
            group: got_group,
            current: u32::MAX,
            add: 1,
        }) if got_group == group
    ));
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert_eq!(
        core.async_copies[0].state.as_deref().unwrap().open_group,
        before_group
    );
    assert_eq!(*core.l1_counters().unwrap(), before_l1);
    assert!(requests.is_empty());
}
