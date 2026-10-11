use super::*;

#[test]
fn replay_instruction_stays_compact() {
    assert!(std::mem::size_of::<ReplayInst>() <= 24);
}

#[test]
fn immediate_constant_parser_excludes_indexed_constant_loads() {
    // True LDC uses the indexed constant-memory path even when a width
    // modifier is present; its bracketed operand must not create an IMC
    // miss in the scheduler.  Uniform ULDC is part of the IMC path.
    assert_eq!(
        &*immediate_constant_lines("ULDC.64 UR8, c[0x0][0x118]", 64),
        &[4]
    );
    assert!(immediate_constant_lines("LDC.64 R2, c[0x0][0x118]", 64).is_empty());
}

#[test]
fn immediate_constant_parser_keeps_cbank_operands_and_deduplicates_lines() {
    let lines = immediate_constant_lines("IADD3 R7, R7, c[0x0][0x0], c[0x0][0x20]", 64);
    assert_eq!(&*lines, &[0]);
    assert_eq!(
        &*immediate_constant_lines("ISETP.LT.AND P0, PT, RZ, c[0x0][0x180], PT", 64),
        &[6]
    );
}

#[test]
fn immediate_constant_cache_pipeline_overlaps_line_fills() {
    let config = ArchId::Sm86
        .immediate_constant_cache()
        .require("immediate_constant_cache")
        .unwrap();
    let mut cache = ImmediateConstantCacheState::new(config).unwrap();
    let ready = cache.reserve(&[0, 5], 0, 244).unwrap();
    assert_eq!(ready, Some(245));
    assert_eq!(cache.pending.len(), 2);
    assert_eq!(cache.pending[0].ready_at, 244);
    assert_eq!(cache.pending[1].ready_at, 245);
    assert_eq!(cache.next_service, 2);
    assert_eq!(cache.diagnostics.tag_misses, 2);

    // A second warp observes the first line as an outstanding reservation,
    // not as another miss request; completion makes it resident.
    let merged = cache.reserve(&[0], 1, 244).unwrap();
    assert_eq!(merged, Some(244));
    assert_eq!(cache.diagnostics.pending_merges, 1);
    cache.complete_ready(244);
    assert!(cache.contains(0));
    assert!(!cache.contains(5));
}

#[test]
fn immediate_constant_cache_fills_ready_lines_in_issue_order() {
    let config = ImmediateConstantCache {
        sets: 1,
        line_bytes: 64,
        ways: 1,
        mshr_entries: 2,
        access_queue_entries: 1,
        miss_initiation_interval_cycles: 1,
        sm_structure_latency_cycles: 1,
        l0_to_l1_cycles: 1,
        switch_after_miss_cycles: 1,
    };
    let mut cache = ImmediateConstantCacheState::new(config).unwrap();
    // Line 2 is issued first and is ready first; line 1 is issued second.
    // Both are completed together after a scheduler jump. The latest fill
    // (line 1) must be the resident one in this one-way set.
    assert_eq!(cache.reserve(&[2], 0, 4).unwrap(), Some(4));
    assert_eq!(cache.reserve(&[1], 1, 4).unwrap(), Some(5));
    cache.complete_ready(10);
    assert_eq!(cache.sets[0].as_slices().0, &[1]);
}

#[test]
fn disabled_candidate_storage_stays_off_the_hot_inline_path() {
    assert!(std::mem::size_of::<MemoryServiceCandidateDiagnostics>() <= 24);
    assert!(std::mem::size_of::<MemoryServiceCandidateState>() <= 24);
    assert!(std::mem::size_of::<MemoryAdmissionCandidateDiagnostics>() <= 24);
    assert!(std::mem::size_of::<MemoryAdmissionCandidateState>() <= 24);
    assert!(std::mem::size_of::<ReadinessDiagnostics>() <= 1024);
    assert!(
        std::mem::size_of::<SmCore<'static>>() <= 4096,
        "SmCore grew to {} bytes",
        std::mem::size_of::<SmCore<'static>>()
    );
    assert!(
        std::mem::size_of::<MemRequest>() <= 96,
        "MemRequest grew to {} bytes",
        std::mem::size_of::<MemRequest>()
    );
}

#[test]
fn legacy_issue_lane_is_explicit_and_sm86_only() {
    let legacy = IssueModelParams::from_legacy_card_latencies_for_arch(
        ArchId::Sm86,
        34.0,
        244.0,
        423.0,
        24.0,
    )
    .unwrap();
    assert_eq!(
        legacy.resource_source(),
        IssueResourceSource::ArchitectureDefaults
    );
    assert_eq!(legacy.non_memory_write_barrier_latency, 25);
    assert_eq!(legacy.read_barrier_latency, 8);
    assert_eq!(legacy.ldgsts_l1_mshr_floor.get(), 248);
    assert_eq!(legacy.max_resident_warps_per_sm, 48);
    legacy.validate_resource_source().unwrap();

    assert!(matches!(
        IssueModelParams::from_legacy_card_latencies_for_arch(
            ArchId::Sm120,
            34.0,
            244.0,
            423.0,
            24.0,
        ),
        Err(IssueError::Capability(
            sim_arch::CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                ..
            }
        ))
    ));
}

#[test]
fn mutated_legacy_hypotheses_cannot_retain_provenance_or_execute() {
    let baseline = IssueModelParams::from_legacy_card_latencies_for_arch(
        ArchId::Sm86,
        34.0,
        244.0,
        423.0,
        24.0,
    )
    .unwrap();
    let mut non_memory = baseline;
    non_memory.non_memory_write_barrier_latency += 1;
    let mut read = baseline;
    read.read_barrier_latency += 1;
    let mut ldgsts_floor = baseline;
    ldgsts_floor.ldgsts_l1_mshr_floor = NonZeroU32::new(249).unwrap();
    let mut occupancy = baseline;
    occupancy.max_resident_warps_per_sm += 1;
    let mut front_end = baseline;
    front_end
        .instruction_front_end
        .as_mut()
        .expect("production constructor binds the front end")
        .stream_buffer_lines += 1;

    for (field, params, expected, got) in [
        (
            "non_memory_write_barrier_latency",
            non_memory,
            25_u64,
            26_u64,
        ),
        ("read_barrier_latency", read, 8_u64, 9_u64),
        ("ldgsts_l1_mshr_floor", ldgsts_floor, 248_u64, 249_u64),
        ("max_resident_warps_per_sm", occupancy, 48_u64, 49_u64),
    ] {
        assert!(matches!(
            params.validate_resource_source(),
            Err(IssueError::ArchitectureDefaultMismatch {
                arch: ArchId::Sm86,
                field: got_field,
                expected: got_expected,
                got: got_value,
            }) if got_field == field && got_expected == expected && got_value == got
        ));
    }

    assert!(matches!(
        front_end.validate_resource_source(),
        Err(IssueError::ArchitectureDefaultMismatch {
            arch: ArchId::Sm86,
            field: "instruction_front_end.stream_buffer_lines",
            expected: 8,
            got: 9,
        })
    ));

    assert!(matches!(
        crate::replay::ReplayConfiguration::checked(non_memory, crate::replay::ReplayMemMode::Flat,),
        Err(crate::replay::ReplayError::Issue(
            IssueError::ArchitectureDefaultMismatch {
                arch: ArchId::Sm86,
                field: "non_memory_write_barrier_latency",
                expected: 25,
                got: 26,
            }
        ))
    ));
}

#[test]
fn strict_card_constructor_requires_same_architecture_and_all_resources() {
    let production_sm86 = ArchCard::load(&card_path("sm_86.json")).unwrap();
    let carded = IssueModelParams::from_card(ArchId::Sm86, &production_sm86)
        .expect("SM86 card now carries the consumed issue resources");
    assert!(matches!(
        carded.resource_source(),
        IssueResourceSource::Carded(_)
    ));

    let production_sm120 = ArchCard::load(&card_path("sm_120-laptop.json")).unwrap();
    assert!(matches!(
        IssueModelParams::from_card(ArchId::Sm86, &production_sm120),
        Err(IssueError::ArchitectureMismatch {
            component: "strict issue architecture card",
            expected: ArchId::Sm86,
            got: ArchId::Sm120,
        })
    ));
}

#[test]
fn complete_strict_card_lane_executes_only_with_consumed_sm86_topology() {
    let strict =
        IssueModelParams::from_card(ArchId::Sm86, &synthetic_carded_issue_resources()).unwrap();
    let IssueResourceSource::Carded(resources) = strict.resource_source() else {
        panic!("strict constructor must preserve its carded lane")
    };
    assert_eq!(resources.smsp.count, 4);
    assert_eq!(resources.smsp.issue_ports, 1);
    assert_eq!(resources.occupancy.max_resident_warps_per_sm, 48);
    assert_eq!(resources.occupancy.max_blocks, 16);
    assert_eq!(resources.dep_barriers.count, 6);
    assert_eq!(resources.barrier_latencies.read, 8);
    assert_eq!(resources.barrier_latencies.non_memory_write, 25);
    assert!(!resources.dual_issue_supported);
    assert_eq!(strict.non_memory_write_barrier_latency, 25);
    assert_eq!(strict.read_barrier_latency, 8);
    assert_eq!(strict.ldgsts_l1_mshr_floor.get(), 248);
    assert_eq!(strict.max_resident_warps_per_sm, 48);

    strict.validate_resource_source().unwrap();
    crate::replay::ReplayConfiguration::checked(strict, crate::replay::ReplayMemMode::Flat)
        .unwrap();

    let sched = schedule(Vec::new());
    SmCore::new(&[], &sched, &strict, SmMemMode::Flat).unwrap();

    let mut mismatched = strict;
    if let IssueResourceSource::Carded(resources) = mismatched.resource_source {
        mismatched.resource_source = IssueResourceSource::Carded(RequiredIssueResources {
            smsp: resources.smsp,
            occupancy: resources.occupancy,
            dep_barriers: resources.dep_barriers,
            barrier_latencies: resources.barrier_latencies,
            dual_issue_supported: true,
        });
    }
    assert!(matches!(
        mismatched.validate_resource_source(),
        Err(IssueError::CardedResourceMismatch {
            field: "dual_issue.supported",
            expected: 1,
            got: 0,
            ..
        })
    ));
}

#[test]
fn carded_issue_width_and_dual_issue_are_consumed() {
    let path = card_path("sm_86.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["smsp"]["issue_ports"] = serde_json::json!(3);
    value["dual_issue"]["supported"] = serde_json::json!(true);
    let card = ArchCard::parse(&path, &serde_json::to_vec(&value).unwrap()).unwrap();
    let params = IssueModelParams::from_card(ArchId::Sm86, &card).unwrap();
    assert_eq!(params.issue_ports, 3);
    assert!(params.dual_issue_supported);
    params.validate_resource_source().unwrap();
}

#[test]
fn hypothetical_issue_width_is_not_widened_by_card_dual_issue() {
    let path = card_path("sm_86.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["dual_issue"]["supported"] = serde_json::json!(true);
    let card = ArchCard::parse(&path, &serde_json::to_vec(&value).unwrap()).unwrap();
    let params = IssueModelParams::from_card(ArchId::Sm86, &card)
        .unwrap()
        .with_hypothetical_issue_ports(1)
        .unwrap();
    assert_eq!(params.issue_ports(), 1);
    assert!(!params.dual_issue_supported());
    params.validate_resource_source().unwrap();
}

#[test]
fn latency_conversion_rejects_u64_boundary_before_cast() {
    assert!(checked_latency("boundary", u64::MAX as f64).is_err());
    assert!(checked_latency("infinite", f64::INFINITY).is_err());
    assert!(checked_latency("negative", -1.0).is_err());
    assert_eq!(checked_latency("fractional", 12.01).unwrap(), 13);
}

#[test]
fn maximum_response_owner_is_allocated_before_exhaustion() {
    let next = AtomicU64::new(u64::MAX);
    assert_eq!(
        allocate_response_owner_from(&next).unwrap(),
        ResponseOwner(u64::MAX)
    );
    assert_eq!(next.load(Ordering::Relaxed), 0);
    assert!(matches!(
        allocate_response_owner_from(&next),
        Err(IssueError::ResponseOwnerExhausted)
    ));
    assert_eq!(next.load(Ordering::Relaxed), 0);
}

#[test]
fn past_horizon_is_rejected_before_mutation() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    core.t = 1;
    let mut out = Vec::new();
    assert!(matches!(
        core.advance(ReplayHorizon::Bounded(0), &mut out),
        Err(IssueError::HorizonBeforeClock {
            horizon: 0,
            clock: 1,
        })
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
}

#[test]
fn every_memory_sector_span_is_validated_before_core_construction() {
    let kinds = [
        AccessKind::GlobalEligible {
            off: 1,
            len: 1,
            token: CacheOpToken::LdgECa,
        },
        AccessKind::GlobalBypass {
            off: 1,
            len: 1,
            token: CacheOpToken::LdgECg,
            l2_requests: 0,
        },
        AccessKind::GlobalUntagged { off: 1, len: 1 },
        AccessKind::GlobalStore { off: 1, len: 1 },
    ];
    for kind in kinds {
        assert!(matches!(
            construction_error(kind, vec![40]),
            IssueError::SectorSpanOutOfBounds {
                pc: 0,
                launch_index: 7,
                warp_slot: 3,
                off: 1,
                len: 1,
                end: 2,
                sectors: 1,
            }
        ));
    }
}

#[test]
fn atomic_l2_request_topology_is_validated_before_core_construction() {
    for kind in [
        AccessKind::GlobalBypass {
            off: 0,
            len: 1,
            token: CacheOpToken::Atom,
            l2_requests: 0,
        },
        AccessKind::GlobalBypass {
            off: 0,
            len: 1,
            token: CacheOpToken::LdgECg,
            l2_requests: 1,
        },
        AccessKind::GlobalReduction {
            off: 0,
            len: 1,
            l2_requests: 2,
        },
    ] {
        assert!(matches!(
            construction_error(kind, vec![40]),
            IssueError::AtomicRequestTopologyInvalid {
                pc: 0,
                launch_index: 7,
                warp_slot: 3,
                ..
            }
        ));
    }
}

#[test]
fn sector_span_overflow_and_empty_out_of_bounds_are_typed() {
    assert!(matches!(
        construction_error(
            AccessKind::GlobalStore {
                off: usize::MAX,
                len: 1,
            },
            Vec::new(),
        ),
        IssueError::SectorSpanOverflow {
            pc: 0,
            launch_index: 7,
            warp_slot: 3,
            off: usize::MAX,
            len: 1,
        }
    ));
    assert!(matches!(
        construction_error(AccessKind::GlobalUntagged { off: 1, len: 0 }, Vec::new(),),
        IssueError::SectorSpanOutOfBounds {
            off: 1,
            len: 0,
            end: 1,
            sectors: 0,
            ..
        }
    ));
}

#[test]
fn empty_sector_span_at_arena_end_is_valid() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LD.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: AccessKind::GlobalUntagged { off: 0, len: 0 },
            }],
            sectors: Vec::new(),
        }],
    }];
    SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat)
        .expect("empty span at the arena end is valid");
}

#[test]
fn invalid_public_control_fields_fail_before_core_construction() {
    let blocks = vec![BlockReplay {
        launch_index: 7,
        warps: vec![warp(3, vec![inst(0)])],
    }];

    let mut invalid_stall = cb(1);
    invalid_stall.stall = 16;
    let stall_schedule = schedule(vec![(0, slot(invalid_stall, OpClass::Other, "IADD3"))]);
    let Err(stall_error) = SmCore::new(&blocks, &stall_schedule, &params(), SmMemMode::Flat) else {
        panic!("invalid stall constructed replay state");
    };
    assert!(matches!(
        stall_error,
        IssueError::ControlDomain {
            pc: 0,
            launch_index: 7,
            warp_slot: 3,
            source: ControlDomainError::StallOutOfRange { stall: 16 },
        }
    ));

    let mut invalid_reuse = cb(1);
    invalid_reuse.reuse = 16;
    let reuse_schedule = schedule(vec![(0, slot(invalid_reuse, OpClass::Other, "IADD3"))]);
    let Err(reuse_error) = SmCore::new(&blocks, &reuse_schedule, &params(), SmMemMode::Flat) else {
        panic!("invalid reuse mask constructed replay state");
    };
    assert!(matches!(
        reuse_error,
        IssueError::ControlDomain {
            pc: 0,
            launch_index: 7,
            warp_slot: 3,
            source: ControlDomainError::ReuseOutOfRange { reuse: 16 },
        }
    ));
}

#[test]
fn producer_latency_overflow_is_typed() {
    let mut control = cb(1);
    control.write_barrier = Some(barrier(0));
    let sched = schedule(vec![(0, slot(control, OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut p = params();
    p.non_memory_write_barrier_latency = u64::MAX;
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    core.t = 1;
    let mut out = Vec::new();
    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::TimeOverflow {
            operation: "non-memory write-barrier visibility",
            base: 1,
            delta: u64::MAX,
        })
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
}

#[test]
fn absent_dependency_barriers_skip_unused_visibility_timing() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDS"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(
            0,
            vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: AccessKind::Shared { wavefronts: 1 },
            }],
        )],
    }];
    let mut p = params();
    p.shared_write_barrier_latency = u64::MAX;
    p.read_barrier_latency = u64::MAX;
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    core.t = u64::MAX - 1;
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert!(out.is_empty());
    assert_eq!(core.result().cycles, u64::MAX);
    assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
}

#[test]
fn present_read_barrier_overflow_remains_typed_and_nonmutating() {
    let mut control = cb(1);
    control.read_barrier = Some(barrier(0));
    let sched = schedule(vec![(0, slot(control, OpClass::Other, "LDS"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(
            0,
            vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: AccessKind::Shared { wavefronts: 1 },
            }],
        )],
    }];
    let mut p = params();
    p.read_barrier_latency = 2;
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    core.t = u64::MAX - 1;
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::TimeOverflow {
            operation: "read-dependence barrier visibility",
            base,
            delta: 2,
        }) if base == u64::MAX - 1
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
}

#[test]
fn warm_l1_hit_ignores_an_overflowing_unused_miss_arm() {
    let mut control = cb(1);
    control.write_barrier = Some(barrier(0));
    let sched = schedule(vec![(0, slot(control, OpClass::Other, "LDG.E"))]);
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
    let mut p = params();
    p.l1_hit_latency = 1;
    p.l1_miss_latency = 2;

    for shared in [false, true] {
        let mode = if shared {
            SmMemMode::L1Shared(p.l1_config(8, 1))
        } else {
            SmMemMode::L1Const(p.l1_config(8, 1))
        };
        let mut core = SmCore::new(&blocks, &sched, &p, mode).unwrap();
        match &mut core.mem {
            MemModel::L1Const(l1) => {
                assert!(matches!(
                    l1.access_eligible(MemoryOrigin::Global, 0, 0).unwrap(),
                    (AccessOutcome::MissLineAlloc, 2)
                ));
                assert!(matches!(
                    l1.access_eligible(MemoryOrigin::Global, 0, 2).unwrap(),
                    (AccessOutcome::Hit, 3)
                ));
            }
            MemModel::L1Shared(st) => {
                let EvAccess::MissRequest { fill } = st
                    .l1
                    .access_eligible_ev(MemoryOrigin::Global, 0, 0)
                    .unwrap()
                else {
                    panic!("expected cold prefill miss");
                };
                st.l1.complete_fill(fill).unwrap();
            }
            MemModel::Flat => panic!("expected an L1 mode"),
        }
        core.t = u64::MAX - 1;
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        assert_eq!(core.result().cycles, u64::MAX);
        assert_eq!(core.warps[0].barrier_clear[0], u64::MAX);
    }
}

#[test]
fn unbarriered_warm_l1_hit_skips_unused_hit_visibility() {
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
            sectors: vec![0],
        }],
    }];
    let mut p = params();
    p.l1_hit_latency = 2;
    p.l1_miss_latency = 3;

    for shared in [false, true] {
        let mode = if shared {
            SmMemMode::L1Shared(p.l1_config(8, 1))
        } else {
            SmMemMode::L1Const(p.l1_config(8, 1))
        };
        let mut core = SmCore::new(&blocks, &sched, &p, mode).unwrap();
        match &mut core.mem {
            MemModel::L1Const(l1) => {
                assert!(matches!(
                    l1.access_eligible(MemoryOrigin::Global, 0, 0).unwrap(),
                    (AccessOutcome::MissLineAlloc, 3)
                ));
                assert!(matches!(
                    l1.access_eligible(MemoryOrigin::Global, 0, 3).unwrap(),
                    (AccessOutcome::Hit, 5)
                ));
            }
            MemModel::L1Shared(st) => {
                let EvAccess::MissRequest { fill } = st
                    .l1
                    .access_eligible_ev(MemoryOrigin::Global, 0, 0)
                    .unwrap()
                else {
                    panic!("expected cold prefill miss");
                };
                st.l1.complete_fill(fill).unwrap();
            }
            MemModel::Flat => panic!("expected an L1 mode"),
        }
        core.t = u64::MAX - 1;
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        assert_eq!(core.result().cycles, u64::MAX);
        assert_eq!(core.warps[0].barrier_clear, [0; DEP_BARRIERS]);
    }
}

#[test]
fn cold_shared_l1_miss_overflow_remains_typed_and_nonmutating() {
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
            sectors: vec![0],
        }],
    }];
    let mut p = params();
    p.l1_hit_latency = 1;
    p.l1_miss_latency = 2;
    let mut core =
        SmCore::new(&blocks, &sched, &p, SmMemMode::L1Shared(p.l1_config(8, 1))).unwrap();
    core.t = u64::MAX - 1;
    let before = *core.l1_counters().unwrap();
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::TimeOverflow {
            operation: "self-response horizon",
            base,
            delta: 2,
        }) if base == u64::MAX - 1
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert_eq!(core.outstanding_responses(), 0);
    assert_eq!(*core.l1_counters().unwrap(), before);
}

#[test]
fn exact_maximum_barrier_time_is_valid() {
    let mut control = cb(1);
    control.write_barrier = Some(barrier(0));
    let sched = schedule(vec![(0, slot(control, OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut p = params();
    p.non_memory_write_barrier_latency = u64::MAX - 1;
    let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
    core.t = 1;
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 1);
    assert_eq!(core.warps[0].barrier_clear[0], u64::MAX);
    assert_eq!(core.result().cycles, 2);
}

#[test]
fn final_issue_can_advance_the_clock_to_exact_maximum() {
    for (stall, yield_flag) in [(15, false), (1, true)] {
        let mut control = cb(stall);
        control.yield_flag = yield_flag;
        let sched = schedule(vec![(0, slot(control, OpClass::Other, "EXIT"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0)])],
        }];
        let mut p = params();
        p.non_memory_write_barrier_latency = 1;
        p.read_barrier_latency = 1;
        let mut core = SmCore::new(&blocks, &sched, &p, SmMemMode::Flat).unwrap();
        core.t = u64::MAX - 1;
        let mut out = Vec::new();

        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        assert_eq!(core.result().instructions_issued, 1);
        assert_eq!(core.result().cycles, u64::MAX);
    }
}

#[test]
fn nonterminal_issue_still_requires_post_issue_eligibility() {
    for (stall, yield_flag, expected_operation, expected_delta) in [
        (15, false, "warp stall eligibility", 15),
        (1, true, "warp yield eligibility", 2),
    ] {
        let mut control = cb(stall);
        control.yield_flag = yield_flag;
        let sched = schedule(vec![
            (0, slot(control, OpClass::Other, "IADD3")),
            (16, slot(cb(1), OpClass::Other, "EXIT")),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![warp(0, vec![inst(0), inst(16)])],
        }];
        let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
        core.t = u64::MAX - 1;
        let mut out = Vec::new();

        assert!(matches!(
            core.advance(ReplayHorizon::Unbounded, &mut out),
            Err(IssueError::TimeOverflow {
                operation,
                base,
                delta,
            }) if operation == expected_operation
                && base == u64::MAX - 1
                && delta == expected_delta
        ));
        assert!(out.is_empty());
        assert_eq!(core.instructions_issued(), 0);
        assert_eq!(core.warps[0].next, 0);
    }
}

#[test]
fn maximum_bounded_horizon_is_distinct_from_unbounded_replay() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut core = SmCore::new(&blocks, &sched, &params(), SmMemMode::Flat).unwrap();
    core.t = u64::MAX;
    let mut out = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Bounded(u64::MAX), &mut out)
            .unwrap(),
        Advance::Idle {
            wake_at: Some(u64::MAX),
        }
    );
    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::TimeOverflow {
            operation: "issue-clock advance",
            base: u64::MAX,
            delta: 1,
        })
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
}

#[test]
fn self_response_horizon_overflow_is_typed() {
    let mut control = cb(1);
    control.write_barrier = Some(barrier(0));
    let sched = schedule(vec![(0, slot(control, OpClass::Other, "LDG.E.CG"))]);
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
            sectors: vec![0],
        }],
    }];
    let mut p = params();
    p.l1_miss_latency = u64::MAX;
    let mut core = SmCore::new(
        &blocks,
        &sched,
        &p,
        SmMemMode::L1Shared(p.l1_config(768, 32)),
    )
    .unwrap();
    core.t = 1;
    let mut out = Vec::new();
    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::TimeOverflow {
            operation: "self-response horizon",
            base: 1,
            delta: u64::MAX,
        })
    ));
    assert!(out.is_empty());
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert_eq!(core.outstanding_responses(), 0);
    assert_eq!(core.l1_counters().unwrap().bypass_sectors, 0);
}

#[test]
fn l1_timing_must_match_issue_parameters() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let p = params();
    let mut cfg = p.l1_config(768, 32);
    cfg.miss_latency += 1;
    assert!(matches!(
        SmCore::new(&blocks, &sched, &p, SmMemMode::L1Shared(cfg)),
        Err(IssueError::L1TimingMismatch {
            field: "miss",
            expected: 244,
            got: 245,
        })
    ));
}

#[test]
fn constant_cacheable_async_copy_uses_access_local_mshr_floor() {
    let mut p = params();
    p.ldgsts_l1_mshr_floor = NonZeroU32::new(2).unwrap();
    let sectors = [0, 4];

    let mut cacheable = L1Cache::new(p.l1_config(8, 1)).unwrap();
    assert_eq!(
        data_ready_time(
            async_copy_inst(0, 0, 2).kind,
            &sectors,
            0,
            &p,
            Some(&mut cacheable),
            None,
            true,
        )
        .unwrap(),
        Some(244)
    );
    assert_eq!(cacheable.counters.stall_events, 0);

    let mut ordinary = L1Cache::new(p.l1_config(8, 1)).unwrap();
    assert_eq!(
        data_ready_time(
            eligible(0, 2),
            &sectors,
            0,
            &p,
            Some(&mut ordinary),
            Some(barrier(0)),
            false,
        )
        .unwrap(),
        Some(488)
    );
    assert_eq!(ordinary.counters.stall_events, 1);
}
