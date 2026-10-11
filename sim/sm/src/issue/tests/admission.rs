use super::*;

#[test]
fn memory_admission_candidate_validation_is_bounded_and_hit_miss_ordered() {
    let mut candidates = [admission_candidate(0, MemoryServiceCandidateScope::Sm);
        MAX_MEMORY_ADMISSION_CANDIDATES + 1];
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.candidate_id = index as u8;
    }
    assert_eq!(
        MemoryAdmissionCandidateState::checked(&candidates),
        Err(MemoryAdmissionCandidateError::TooMany {
            count: MAX_MEMORY_ADMISSION_CANDIDATES + 1,
            maximum: MAX_MEMORY_ADMISSION_CANDIDATES,
        })
    );
    let mut invalid = admission_candidate(3, MemoryServiceCandidateScope::Sm);
    invalid.queue_capacity_instructions = 0;
    assert_eq!(
        invalid.validate(),
        Err(MemoryAdmissionCandidateError::CapacityOutOfRange {
            candidate_id: 3,
            capacity: 0,
            maximum: MAX_MEMORY_ADMISSION_QUEUE_CAPACITY,
        })
    );

    invalid = admission_candidate(6, MemoryServiceCandidateScope::Sm);
    invalid.write_completion_latency_cycles = 7;
    assert_eq!(
        invalid.validate(),
        Err(
            MemoryAdmissionCandidateError::CompletionLatencyWithoutMembar {
                candidate_id: 6,
                latency: 7,
            }
        )
    );
    invalid = admission_candidate(7, MemoryServiceCandidateScope::Sm);
    invalid.write_completion_request_cycles = 1;
    invalid.write_completion_parallel_lanes = 1;
    assert_eq!(
        invalid.validate(),
        Err(MemoryAdmissionCandidateError::CompletionWorkWithoutMembar { candidate_id: 7 })
    );
    invalid.membar_waits_for_service_completion = true;
    invalid.write_completion_parallel_lanes = 0;
    assert_eq!(
        invalid.validate(),
        Err(MemoryAdmissionCandidateError::CompletionLanesOutOfRange {
            candidate_id: 7,
            lanes: 0,
            maximum: MAX_WRITE_COMPLETION_PARALLEL_LANES,
        })
    );
    invalid.write_completion_request_cycles = 0;
    invalid.write_completion_parallel_lanes = 1;
    assert_eq!(
        invalid.validate(),
        Err(MemoryAdmissionCandidateError::CompletionLanesWithoutWork {
            candidate_id: 7,
            lanes: 1,
        })
    );
    invalid = admission_candidate(4, MemoryServiceCandidateScope::Sm);
    invalid.l2_miss_sector_cycles = 1;
    invalid.l2_hit_sector_cycles = 2;
    assert_eq!(
        invalid.validate(),
        Err(MemoryAdmissionCandidateError::MissRateFasterThanHit { candidate_id: 4 })
    );

    invalid = admission_candidate(5, MemoryServiceCandidateScope::Sm);
    invalid.l2_miss_service_width_sectors = MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS + 1;
    assert_eq!(
        invalid.validate(),
        Err(MemoryAdmissionCandidateError::ServiceWidthOutOfRange {
            candidate_id: 5,
            width: MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS + 1,
            maximum: MAX_MEMORY_ADMISSION_SERVICE_WIDTH_SECTORS,
        })
    );
}

#[test]
fn feedback_fence_waits_only_for_issuing_warps_unpriced_writes_and_acks() {
    for scope in [
        MemoryServiceCandidateScope::Sm,
        MemoryServiceCandidateScope::Smsp,
    ] {
        let mut candidate = admission_candidate(9, scope);
        candidate.queue_capacity_instructions = 4;
        candidate.membar_waits_for_service_completion = true;
        let mut feedback = MemoryAdmissionFeedbackState::checked(candidate).unwrap();
        let store = AccessKind::GlobalStore { off: 0, len: 1 };
        let membar = slot(cb(1), OpClass::Other, "MEMBAR.SC.GPU");
        let ordinary = slot(cb(1), OpClass::Other, "IADD3");

        feedback.reserve(1, 1, 0, store, 0).unwrap();
        assert_eq!(feedback.completion_ready_at(1, &membar), None);
        assert_eq!(
            feedback.completion_ready_at(0, &membar),
            Some(0),
            "an unrelated warp's unpriced write is not this fence's obligation"
        );
        feedback.reserve(0, 0, 0, store, 0).unwrap();
        assert_eq!(feedback.completion_ready_at(0, &membar), None);
        assert_eq!(feedback.completion_ready_at(0, &ordinary), Some(0));
        let ack = feedback
            .complete(
                AdmissionRequestWitness::new(0, 0, 0, 0, 1).unwrap(),
                MemoryOperation::Store,
                0,
                1,
                1,
                1,
                0,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            feedback.completion_ready_at(0, &membar),
            None,
            "pricing a write does not deliver its acknowledgement"
        );
        feedback
            .acknowledge_write(ack, ack.fence_visible_at)
            .unwrap();
        assert_eq!(
            feedback.completion_ready_at(0, &membar),
            Some(ack.fence_visible_at)
        );
        assert_eq!(feedback.completion_ready_at(1, &membar), None);
        assert!(
            feedback.has_unpriced(),
            "system drain still sees the other warp"
        );

        let other_ack = feedback
            .complete(
                AdmissionRequestWitness::new(1, 0, 1, 0, 1).unwrap(),
                MemoryOperation::Store,
                0,
                1,
                1,
                1,
                ack.fence_visible_at,
            )
            .unwrap()
            .unwrap();
        assert_eq!(feedback.completion_ready_at(1, &membar), None);
        feedback
            .acknowledge_write(other_ack, other_ack.fence_visible_at)
            .unwrap();
        assert_eq!(
            feedback.completion_ready_at(1, &membar),
            Some(other_ack.fence_visible_at)
        );
        assert_eq!(feedback.pending_write_ack_count(), 0);
        assert!(!feedback.has_unpriced());
        assert_eq!(feedback.write_ack_diagnostics().acks_scheduled, 2);
        assert_eq!(feedback.write_ack_diagnostics().acks_processed, 2);
    }
}

#[test]
fn memory_admission_miss_width_is_independent_from_asymptotic_rate() {
    assert_eq!(candidate_rate_cycles_with_width(5, 1, 10, 0).unwrap(), 50);
    assert_eq!(candidate_rate_cycles_with_width(5, 1, 10, 1).unwrap(), 50);
    assert_eq!(candidate_rate_cycles_with_width(5, 1, 10, 4).unwrap(), 80);
    assert_eq!(candidate_rate_cycles_with_width(8, 1, 10, 4).unwrap(), 80);

    let mut candidate = admission_candidate(6, MemoryServiceCandidateScope::Sm);
    candidate.l2_miss_service_width_sectors = 4;
    let mut state = MemoryAdmissionCandidateState::checked(&[candidate]).unwrap();
    state
        .observe(0, 0, MemoryOperation::Store, 0, 5, 0)
        .unwrap();
    let result = state.diagnostics().entries()[0];
    assert_eq!(result.service_busy_cycles, 32);
    assert_eq!(result.terminal_service_completion_time, 32);

    let mut completion_candidate = candidate;
    completion_candidate.membar_waits_for_service_completion = true;
    completion_candidate.write_completion_latency_cycles = 5;
    let mut feedback = MemoryAdmissionFeedbackState::checked(completion_candidate).unwrap();
    feedback
        .reserve(0, 3, 4, AccessKind::GlobalStore { off: 0, len: 1 }, 0)
        .unwrap();
    let first_ack = feedback
        .complete(
            AdmissionRequestWitness::new(3, 4, 0, 0, 1).unwrap(),
            MemoryOperation::Store,
            0,
            5,
            1,
            1,
            0,
        )
        .unwrap()
        .unwrap();
    assert_eq!(feedback.diagnostics().service_busy_cycles, 32);
    assert_eq!(feedback.diagnostics().terminal_service_completion_time, 32);

    feedback
        .reserve(1, 4, 5, AccessKind::GlobalStore { off: 0, len: 1 }, 0)
        .unwrap();
    let second_ack = feedback
        .complete(
            AdmissionRequestWitness::new(4, 5, 1, 0, 1).unwrap(),
            MemoryOperation::Store,
            0,
            5,
            1,
            1,
            0,
        )
        .unwrap()
        .unwrap();
    let membar = slot(cb(1), OpClass::Other, "MEMBAR.SC.GPU");
    let ordinary = slot(cb(1), OpClass::Other, "IADD3");
    assert_eq!(feedback.completion_ready_at(3, &membar), None);
    assert_eq!(feedback.completion_ready_at(4, &membar), None);
    feedback
        .acknowledge_write(first_ack, first_ack.fence_visible_at)
        .unwrap();
    assert_eq!(feedback.completion_ready_at(3, &membar), Some(37));
    assert_eq!(feedback.completion_ready_at(4, &membar), None);
    feedback
        .acknowledge_write(second_ack, second_ack.fence_visible_at)
        .unwrap();
    assert_eq!(feedback.completion_ready_at(4, &membar), Some(69));
    assert_eq!(feedback.completion_ready_at(3, &ordinary), Some(0));
    assert_eq!(feedback.diagnostics().terminal_write_completion_time, 69);

    let mut parallel_candidate = admission_candidate(8, MemoryServiceCandidateScope::Sm);
    parallel_candidate.l2_miss_sector_cycles = 1;
    parallel_candidate.membar_waits_for_service_completion = true;
    parallel_candidate.write_completion_request_cycles = 100;
    parallel_candidate.write_completion_parallel_lanes = 2;
    let mut parallel = MemoryAdmissionFeedbackState::checked(parallel_candidate).unwrap();
    let mut parallel_acks = Vec::new();
    for (warp, subcore) in [(0, 0), (1, 1)] {
        parallel
            .reserve(
                subcore,
                warp,
                0,
                AccessKind::GlobalStore { off: 0, len: 1 },
                0,
            )
            .unwrap();
        let ack = parallel
            .complete(
                AdmissionRequestWitness::new(warp as u32, 0, subcore as u8, 0, 1).unwrap(),
                MemoryOperation::Store,
                0,
                1,
                1,
                1,
                0,
            )
            .unwrap()
            .unwrap();
        parallel_acks.push(ack);
    }
    assert_eq!(parallel.completion_ready_at(0, &membar), None);
    assert_eq!(parallel.completion_ready_at(1, &membar), None);
    for ack in parallel_acks {
        parallel
            .acknowledge_write(ack, ack.fence_visible_at)
            .unwrap();
    }
    assert_eq!(parallel.completion_ready_at(0, &membar), Some(101));
    assert_eq!(parallel.completion_ready_at(1, &membar), Some(102));
    assert_eq!(parallel.diagnostics().write_completion_busy_cycles, 200);
    assert_eq!(parallel.diagnostics().write_completion_requests_observed, 2);
    assert_eq!(parallel.diagnostics().write_completion_regions_observed, 2);
    let readiness = ReadinessDiagnostics {
        memory_admission_feedback: Some(Box::new(parallel.diagnostics())),
        memory_write_ack: Some(Box::new(parallel.write_ack_diagnostics())),
        ..ReadinessDiagnostics::default()
    };
    assert_eq!(readiness.schema_version(), 22);
}

#[test]
fn legacy_admission_candidate_json_omits_zero_service_width() {
    let candidate = admission_candidate(7, MemoryServiceCandidateScope::Smsp);
    let encoded = serde_json::to_value(candidate).unwrap();
    assert!(encoded.get("l2_miss_service_width_sectors").is_none());
    assert!(encoded.get("membar_waits_for_service_completion").is_none());
    assert!(encoded.get("write_completion_latency_cycles").is_none());
    assert!(encoded.get("write_completion_request_cycles").is_none());
    assert!(encoded.get("write_completion_region_cycles").is_none());
    assert!(encoded.get("write_completion_parallel_lanes").is_none());
    let decoded: MemoryAdmissionCandidate = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, candidate);

    let mut widened = candidate;
    widened.l2_miss_service_width_sectors = 4;
    let encoded = serde_json::to_value(widened).unwrap();
    assert_eq!(encoded["l2_miss_service_width_sectors"], 4);
    assert_eq!(
        serde_json::from_value::<MemoryAdmissionCandidate>(encoded).unwrap(),
        widened
    );

    widened.membar_waits_for_service_completion = true;
    widened.write_completion_latency_cycles = 17;
    widened.write_completion_request_cycles = 4;
    widened.write_completion_parallel_lanes = 2;
    let encoded = serde_json::to_value(widened).unwrap();
    assert_eq!(encoded["membar_waits_for_service_completion"], true);
    assert_eq!(encoded["write_completion_latency_cycles"], 17);
    assert_eq!(encoded["write_completion_request_cycles"], 4);
    assert_eq!(encoded["write_completion_parallel_lanes"], 2);
    assert_eq!(
        serde_json::from_value::<MemoryAdmissionCandidate>(encoded).unwrap(),
        widened
    );
}

#[test]
fn memory_admission_candidates_share_service_but_scope_queue_capacity() {
    let candidates = [
        admission_candidate(1, MemoryServiceCandidateScope::Sm),
        admission_candidate(2, MemoryServiceCandidateScope::Smsp),
    ];
    let mut state = MemoryAdmissionCandidateState::checked(&candidates).unwrap();
    state
        .observe(0, 0, MemoryOperation::Store, 0, 1, 0)
        .unwrap();
    state
        .observe(1, 1, MemoryOperation::Store, 0, 1, 0)
        .unwrap();
    state
        .observe(0, 0, MemoryOperation::Store, 0, 1, 0)
        .unwrap();
    let diagnostics = state.diagnostics();
    let sm = diagnostics.entries()[0];
    let smsp = diagnostics.entries()[1];

    assert_eq!(sm.instructions_observed, 3);
    assert_eq!(sm.l2_miss_sectors_observed, 3);
    assert_eq!(sm.service_busy_cycles, 12);
    assert_eq!(sm.downstream_queue_delay_cycles, 8);
    assert_eq!(sm.queue_full_instructions, 1);
    assert_eq!(sm.total_admission_delay_cycles, 4);
    assert_eq!(sm.maximum_admission_delay_cycles, 4);
    assert_eq!(sm.maximum_warp_admission_delay_cycles, 4);
    assert_eq!(sm.terminal_operand_read_clear_time, 12);
    assert_eq!(sm.terminal_service_completion_time, 12);

    assert_eq!(smsp.instructions_observed, 3);
    assert_eq!(smsp.service_busy_cycles, 12);
    assert_eq!(smsp.downstream_queue_delay_cycles, 12);
    assert_eq!(smsp.queue_full_instructions, 0);
    assert_eq!(smsp.total_admission_delay_cycles, 0);
    assert_eq!(smsp.maximum_warp_admission_delay_cycles, 0);
    assert_eq!(diagnostics.entries().len(), 2);
}

#[test]
fn memory_admission_candidate_aggregation_is_checked_and_schema9() {
    let candidate = admission_candidate(6, MemoryServiceCandidateScope::Sm);
    let mut left = MemoryAdmissionCandidateState::checked(&[candidate]).unwrap();
    left.observe(0, 0, MemoryOperation::Store, 1, 0, 10)
        .unwrap();
    let mut right = MemoryAdmissionCandidateState::checked(&[candidate]).unwrap();
    right
        .observe(1, 0, MemoryOperation::Store, 0, 1, 20)
        .unwrap();
    let mut readiness = ReadinessDiagnostics::default();
    readiness.memory_admission_candidates = left.diagnostics();
    let mut addend = ReadinessDiagnostics::default();
    addend.memory_admission_candidates = right.diagnostics();
    readiness.add_checked(&addend).unwrap();
    let result = readiness.memory_admission_candidates.entries()[0];
    assert_eq!(result.instructions_observed, 2);
    assert_eq!(result.l2_hit_sectors_observed, 1);
    assert_eq!(result.l2_miss_sectors_observed, 1);
    assert_eq!(result.terminal_service_completion_time, 24);
    assert_eq!(readiness.schema_version(), 9);

    let mismatch = MemoryAdmissionCandidateState::checked(&[admission_candidate(
        7,
        MemoryServiceCandidateScope::Sm,
    )])
    .unwrap()
    .diagnostics();
    let mut bad = ReadinessDiagnostics::default();
    bad.memory_admission_candidates = mismatch;
    assert!(matches!(
        readiness.add_checked(&bad),
        Err(ReadinessCounterError::MemoryAdmissionCandidateMismatch { index: 0 })
    ));
}

#[test]
fn feedback_admission_reservation_and_pricing_overflows_are_transactional() {
    let candidate = admission_candidate(8, MemoryServiceCandidateScope::Sm);

    let mut reserve_overflow = MemoryAdmissionFeedbackState::checked(candidate).unwrap();
    reserve_overflow.diagnostics.instructions_admitted = u64::MAX;
    let before = reserve_overflow.clone();
    assert!(matches!(
        reserve_overflow.reserve(0, 0, 0, AccessKind::GlobalStore { off: 0, len: 1 }, 0,),
        Err(IssueError::CounterOverflow {
            counter: "feedback admission instructions"
        })
    ));
    assert_eq!(reserve_overflow, before);

    let mut pricing_overflow = MemoryAdmissionFeedbackState::checked(candidate).unwrap();
    pricing_overflow
        .reserve(0, 0, 0, AccessKind::GlobalStore { off: 0, len: 1 }, 0)
        .unwrap();
    pricing_overflow.diagnostics.l2_hit_sectors_observed = u64::MAX;
    let before = pricing_overflow.clone();
    let witness = AdmissionRequestWitness::new(0, 0, 0, 0, 1).unwrap();
    assert!(matches!(
        pricing_overflow.complete(witness, MemoryOperation::Store, 1, 0, 1, 1, 0),
        Err(IssueError::CounterOverflow {
            counter: "feedback admission L2 hit sectors"
        })
    ));
    assert_eq!(pricing_overflow, before);
}

#[test]
fn feedback_issue_plan_is_transactional_and_accounts_same_cycle_reservations() {
    let mut candidate = admission_candidate(9, MemoryServiceCandidateScope::Sm);
    candidate.queue_capacity_instructions = 2;
    candidate.membar_waits_for_service_completion = true;
    let mut feedback = MemoryAdmissionFeedbackState::checked(candidate).unwrap();
    let issues = [
        Some(MemoryAdmissionFeedbackIssue {
            subcore: 0,
            warp: 3,
            instruction: 7,
            kind: AccessKind::GlobalStore { off: 0, len: 1 },
            prior_clock_reads: 1,
            clock_read: true,
        }),
        Some(MemoryAdmissionFeedbackIssue {
            subcore: 1,
            warp: 4,
            instruction: 8,
            kind: AccessKind::GlobalStore { off: 0, len: 1 },
            prior_clock_reads: 0,
            clock_read: false,
        }),
        None,
        None,
    ];
    let before = feedback.clone();
    let plan = feedback.plan_issue(&issues, 11).unwrap();
    assert_eq!(feedback, before, "preflight must not mutate feedback state");
    feedback.apply_issue(plan, 11);
    assert_eq!(feedback.queues[0].len(), 2);
    assert_eq!(feedback.diagnostics.instructions_admitted, 2);
    assert_eq!(feedback.diagnostics.maximum_waiting_instructions, 2);
    assert_eq!(feedback.second_clock_issue(3), Some(11));
    assert!(feedback.take_yield_requested());

    let mut full = MemoryAdmissionFeedbackState::checked(candidate).unwrap();
    full.apply_issue(full.plan_issue(&issues, 11).unwrap(), 11);
    let overflow = [
        Some(MemoryAdmissionFeedbackIssue {
            subcore: 2,
            warp: 5,
            instruction: 9,
            kind: AccessKind::GlobalStore { off: 0, len: 1 },
            prior_clock_reads: 0,
            clock_read: false,
        }),
        None,
        None,
        None,
    ];
    let before = full.clone();
    assert!(matches!(
        full.plan_issue(&overflow, 11),
        Err(IssueError::MemoryAdmissionFeedbackQueueFull {
            subcore: 2,
            time: 11,
        })
    ));
    assert_eq!(full, before);
}

#[test]
fn memory_service_candidate_validation_is_bounded_and_unambiguous() {
    let mut candidates =
        [service_candidate(0, MemoryServiceCandidateScope::Sm); MAX_MEMORY_SERVICE_CANDIDATES + 1];
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.candidate_id = index as u8;
    }
    assert_eq!(
        MemoryServiceCandidateState::checked(&candidates),
        Err(MemoryServiceCandidateError::TooMany {
            count: MAX_MEMORY_SERVICE_CANDIDATES + 1,
            maximum: MAX_MEMORY_SERVICE_CANDIDATES,
        })
    );
    assert_eq!(
        MemoryServiceCandidateState::checked(&[
            service_candidate(7, MemoryServiceCandidateScope::Sm),
            service_candidate(7, MemoryServiceCandidateScope::Smsp),
        ]),
        Err(MemoryServiceCandidateError::DuplicateId { candidate_id: 7 })
    );
    let mut invalid = service_candidate(8, MemoryServiceCandidateScope::Sm);
    invalid.unique_line_units = 0;
    assert_eq!(
        invalid.validate(),
        Err(MemoryServiceCandidateError::InvalidRate {
            candidate_id: 8,
            dimension: "unique_line",
            units: 0,
            cycles: 1,
        })
    );
    invalid.unique_line_cycles = 0;
    invalid.ideal_wavefront_units = 0;
    invalid.ideal_wavefront_cycles = 0;
    assert_eq!(
        invalid.validate(),
        Err(MemoryServiceCandidateError::NoServiceWork { candidate_id: 8 })
    );
}

#[test]
fn memory_service_candidates_stream_at_issue_time_by_selected_scope() {
    let candidates = [
        service_candidate(1, MemoryServiceCandidateScope::Sm),
        service_candidate(2, MemoryServiceCandidateScope::Smsp),
    ];
    let mut state = MemoryServiceCandidateState::checked(&candidates).unwrap();
    state
        .observe(0, GlobalMemoryIssueClass::Store, 4, 2, 10)
        .unwrap();
    state
        .observe(1, GlobalMemoryIssueClass::Store, 4, 2, 12)
        .unwrap();
    let diagnostics = state.diagnostics();
    let sm = diagnostics.entries()[0];
    let smsp = diagnostics.entries()[1];

    assert_eq!(sm.instructions_observed, 2);
    assert_eq!(sm.ideal_wavefronts_observed, 8);
    assert_eq!(sm.unique_lines_observed, 4);
    assert_eq!(sm.busy_cycles, 16);
    assert_eq!(sm.queue_delay_cycles, 6);
    assert_eq!(sm.terminal_completion_time, 31);
    assert_eq!(smsp.instructions_observed, 2);
    assert_eq!(smsp.busy_cycles, 16);
    assert_eq!(smsp.queue_delay_cycles, 0);
    assert_eq!(smsp.terminal_completion_time, 25);
}

#[test]
fn memory_service_candidate_aggregation_is_transactional_and_uses_max_terminal() {
    let candidate = service_candidate(3, MemoryServiceCandidateScope::Sm);
    let mut left = MemoryServiceCandidateState::checked(&[candidate]).unwrap();
    left.observe(0, GlobalMemoryIssueClass::Store, 4, 2, 10)
        .unwrap();
    let mut right = MemoryServiceCandidateState::checked(&[candidate]).unwrap();
    right
        .observe(0, GlobalMemoryIssueClass::Store, 4, 2, 40)
        .unwrap();
    let mut aggregate = left.diagnostics();
    aggregate.add_checked(&right.diagnostics()).unwrap();
    assert_eq!(aggregate.entries()[0].instructions_observed, 2);
    assert_eq!(aggregate.entries()[0].busy_cycles, 16);
    assert_eq!(aggregate.entries()[0].terminal_completion_time, 53);

    let before = aggregate.clone();
    let mismatch = MemoryServiceCandidateState::checked(&[service_candidate(
        4,
        MemoryServiceCandidateScope::Sm,
    )])
    .unwrap()
    .diagnostics();
    assert_eq!(
        aggregate.add_checked(&mismatch),
        Err(ReadinessCounterError::MemoryServiceCandidateMismatch { index: 0 })
    );
    assert_eq!(aggregate, before);

    let mut readiness = ReadinessDiagnostics::default();
    readiness.memory_service_candidates = aggregate;
    assert_eq!(readiness.schema_version(), 7);
}

#[test]
fn result_writeback_diagnostics_aggregate_and_overflow_transactionally() {
    let mut aggregate = ReadinessDiagnostics::default();
    aggregate.result_writeback.fixed_over_l1tex_collisions = 2;
    aggregate.result_writeback.l1tex_deferral_cycles = 2;
    let mut addend = ReadinessDiagnostics::default();
    addend.result_writeback.fixed_over_l1tex_collisions = 3;
    addend.result_writeback.l1tex_deferral_cycles = 3;
    aggregate.add_checked(&addend).unwrap();
    assert_eq!(aggregate.result_writeback.fixed_over_l1tex_collisions, 5);
    assert_eq!(aggregate.result_writeback.l1tex_deferral_cycles, 5);
    assert_eq!(aggregate.schema_version(), 4);

    aggregate.result_writeback.l1tex_deferral_cycles = u64::MAX;
    let before = aggregate.clone();
    assert!(matches!(
        aggregate.add_checked(&addend),
        Err(ReadinessCounterError::Overflow {
            field: "readiness.result_writeback.l1tex_deferral_cycles",
            accumulated: u64::MAX,
            addend: 3,
        })
    ));
    assert_eq!(aggregate, before);
}

#[test]
fn finished_warp_does_not_block_barrier() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "IADD3")),
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
            warp(1, vec![inst(0)]),
        ],
    }];
    let r = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(r.instructions_issued, 4);
}

#[test]
fn residency_cap_forces_waves() {
    // Block 0 on sub-cores 0/1, block 1 on sub-cores 2/3 so the
    // uncapped run can use all four sub-cores at once.
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "IADD3")),
        (16, slot(cb(1), OpClass::Other, "EXIT")),
    ]);
    let mk_block = |ix: i64, slot0: i32| BlockReplay {
        launch_index: ix,
        warps: vec![
            warp(slot0, vec![inst(0), inst(16)]),
            warp(slot0 + 1, vec![inst(0), inst(16)]),
        ],
    };
    let blocks = vec![mk_block(0, 0), mk_block(1, 2)];
    let mut p = params();
    p.max_resident_warps_per_sm = 2;
    let r = simulate_sm(&blocks, &sched, &p, None).unwrap();
    assert_eq!(r.instructions_issued, 8);
    assert!(r.cycles >= 4, "cycles={}", r.cycles);
    let r2 = simulate_sm(&blocks, &sched, &params(), None).unwrap();
    assert_eq!(r2.cycles, 2);
}

#[test]
fn shared_memory_capacity_forces_waves_and_releases_on_retirement() {
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
    let residency = SharedMemoryResidency::new(64, 64);
    let mut core =
        SmCore::new_with_residency(&blocks, &sched, &params(), residency, SmMemMode::Flat).unwrap();
    assert_eq!(core.next_block, 1);
    assert_eq!(core.resident_shared_memory_bytes, 64);
    assert_eq!(&*core.resident_warp_slots, &HashSet::from([0]));

    let mut out = Vec::new();
    assert_eq!(
        core.advance(ReplayHorizon::Bounded(1), &mut out).unwrap(),
        Advance::Idle { wake_at: Some(1) }
    );
    assert_eq!(core.next_block, 2);
    assert_eq!(core.resident_shared_memory_bytes, 64);
    assert_eq!(&*core.resident_warp_slots, &HashSet::from([1]));

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
        Advance::Done
    );
    assert_eq!(core.resident_shared_memory_bytes, 0);
    assert!(core.resident_warp_slots.is_empty());
    assert_eq!(core.instructions_issued(), 2);
}

#[test]
fn unknown_pc_is_typed_error() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "IADD3"))]);
    let blocks = vec![BlockReplay {
        launch_index: 7,
        warps: vec![warp(3, vec![inst(0), inst(999)])],
    }];
    let err = simulate_sm(&blocks, &sched, &params(), None).unwrap_err();
    assert!(matches!(err, IssueError::PcNotInSchedule { pc: 999, .. }));
}

#[test]
fn admission_overflow_during_retirement_rejects_entire_issue_cycle_before_mutation() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![
        BlockReplay {
            launch_index: 8,
            warps: vec![warp(0, vec![inst(0)])],
        },
        BlockReplay {
            launch_index: 9,
            warps: vec![warp(1, vec![inst(0)])],
        },
    ];
    let mut parameters = params();
    parameters.max_resident_warps_per_sm = 1;
    let mut core = SmCore::new(&blocks, &sched, &parameters, SmMemMode::Flat).unwrap();
    // The second block is waiting for the first to retire.  Corrupt only
    // the admission stamp: retirement will otherwise mutate the first
    // warp before the deferred successor admission reports overflow.
    core.arrival_stamp = u64::MAX;
    let before = (
        core.instructions_issued(),
        core.unfinished_warps,
        core.resident_warps,
        core.subcore_warps.each_ref().map(Vec::clone),
        core.hold,
        core.warps
            .iter()
            .map(|warp| (warp.next, warp.eligible, warp.done))
            .collect::<Vec<_>>(),
    );
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::CounterOverflow {
            counter: "warp arrival stamp"
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
            core.warps
                .iter()
                .map(|warp| (warp.next, warp.eligible, warp.done))
                .collect::<Vec<_>>(),
        ),
        before
    );
}

#[test]
fn unfinished_warp_overflow_during_retirement_is_transactional() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![
        BlockReplay {
            launch_index: 10,
            warps: vec![warp(0, vec![inst(0)])],
        },
        BlockReplay {
            launch_index: 11,
            warps: vec![warp(1, vec![inst(0)]), warp(2, vec![inst(0)])],
        },
    ];
    let mut parameters = params();
    parameters.max_resident_warps_per_sm = 2;
    let mut core = SmCore::new(&blocks, &sched, &parameters, SmMemMode::Flat).unwrap();
    core.unfinished_warps = usize::MAX;
    let before = (
        core.instructions_issued(),
        core.unfinished_warps,
        core.resident_warps,
        core.subcore_warps.each_ref().map(Vec::clone),
        core.hold,
        core.warps
            .iter()
            .map(|warp| (warp.next, warp.eligible, warp.done))
            .collect::<Vec<_>>(),
    );
    let mut out = Vec::new();

    assert!(matches!(
        core.advance(ReplayHorizon::Unbounded, &mut out),
        Err(IssueError::CounterOverflow {
            counter: "unfinished warp"
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
            core.warps
                .iter()
                .map(|warp| (warp.next, warp.eligible, warp.done))
                .collect::<Vec<_>>(),
        ),
        before
    );
}

#[test]
fn async_l1_hit_service_overflow_is_transactional() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![async_copy_inst(0, 0, 1)],
            sectors: vec![0],
        }],
    }];
    let mut p = params();
    p.ldgsts_l1_hit_return_service = Some(LdgstsL1HitReturnService { interval_cycles: 1 });
    let mut core = shared_core(&blocks, &sched, &p);

    // Warm the event-mode L1 so the cacheable source enters the hit-return
    // service timeline after the async group has been allocated.
    let fill = {
        let MemModel::L1Shared(st) = &mut core.mem else {
            panic!("fixture must use shared L1");
        };
        let EvAccess::MissRequest { fill } = st
            .l1
            .access_eligible_ev(MemoryOrigin::Global, 0, 0)
            .unwrap()
        else {
            panic!("cold source must allocate an L1 fill");
        };
        st.l1.complete_fill(fill).unwrap();
        fill
    };
    let _ = fill;
    core.shared_write_lookup_runtime.async_l1_hit_service_ready[0] = u64::MAX;
    let before_l1 = *core.l1_counters().unwrap();
    let before_group = core.async_copies[0].state.as_deref().map(|state| {
        (
            state.open_group.map(|group| {
                (
                    group.id,
                    group.pending_sources,
                    group.source_completion_time,
                    group.delivery_work,
                    group.destination_wavefronts,
                    group.completion_time,
                )
            }),
            state.committed_groups.len(),
            state.bypass_copies.len(),
        )
    });
    let mut requests = Vec::new();

    assert!(matches!(
        core.issue_cycle(&mut requests),
        Err(IssueError::CounterOverflow {
            counter: "async L1 hit service timeline"
        })
    ));
    assert_eq!(core.instructions_issued(), 0);
    assert_eq!(core.warps[0].next, 0);
    assert_eq!(*core.l1_counters().unwrap(), before_l1);
    let after_group = core.async_copies[0].state.as_deref().map(|state| {
        (
            state.open_group.map(|group| {
                (
                    group.id,
                    group.pending_sources,
                    group.source_completion_time,
                    group.delivery_work,
                    group.destination_wavefronts,
                    group.completion_time,
                )
            }),
            state.committed_groups.len(),
            state.bypass_copies.len(),
        )
    });
    assert_eq!(after_group, before_group);
    assert!(requests.is_empty());
}

#[test]
fn async_l1_hit_service_charges_one_token_per_coalesced_instruction() {
    let run = |len: u16| {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDGSTS.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![async_copy_inst(0, 0, len)],
                sectors: (0..usize::from(len)).map(|value| value as i64).collect(),
            }],
        }];
        let mut p = params();
        p.ldgsts_l1_hit_return_service = Some(LdgstsL1HitReturnService { interval_cycles: 1 });
        let mut core = shared_core(&blocks, &sched, &p);
        {
            let MemModel::L1Shared(st) = &mut core.mem else {
                panic!("fixture must use shared L1");
            };
            let EvAccess::MissRequest { fill } = st
                .l1
                .access_eligible_ev(MemoryOrigin::Global, 0, 0)
                .unwrap()
            else {
                panic!("cold source must allocate an L1 fill");
            };
            st.l1.complete_fill(fill).unwrap();
            for sector in 1..usize::from(len) {
                assert!(matches!(
                    st.l1
                        .access_eligible_ev(MemoryOrigin::Global, 0, sector as u64)
                        .unwrap(),
                    EvAccess::Hit { .. }
                ));
            }
        }
        let mut requests = Vec::new();
        assert!(core.issue_cycle(&mut requests).is_ok());
        core.shared_write_lookup_runtime.async_l1_hit_service_ready[0]
    };

    // A coalesced two-sector instruction consumes the same single source
    // return token as a one-sector instruction; charging per sector would
    // make these frontiers differ.
    assert_eq!(run(2), run(1));
}

#[test]
fn witness_aware_payload_admission_uses_event_mode_residency_without_inference() {
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
    let mut core = shared_core(&blocks, &sched, &params());

    // Establish a model-local resident line through the same event-mode
    // fill path used by replay. The fill generation and sector mask are
    // admissible witness fields; the producer allocation identity below
    // remains independent and is never inferred from the line tag.
    let (resident_generation, resident_sector_mask) = {
        let MemModel::L1Shared(state) = &mut core.mem else {
            panic!("fixture must use shared L1");
        };
        let EvAccess::MissRequest { fill } = state
            .l1
            .access_eligible_ev(MemoryOrigin::Global, 0, 0)
            .unwrap()
        else {
            panic!("cold sector must allocate an event-mode fill");
        };
        state.l1.complete_fill(fill).unwrap();
        state
            .l1
            .resident_payload_state(0)
            .expect("completed line must expose model residency")
    };
    assert_eq!(resident_sector_mask & 1, 1);

    let payload = sim_mem::SectorPayloadV1::new_with_generation(
        11,
        3,
        0,
        0,
        1,
        [0x5a; sim_mem::PAYLOAD_SECTOR_BYTES],
        u32::MAX,
    )
    .unwrap();
    let witness = sim_mem::L1PayloadResidencyWitness {
        line_id: 0,
        allocation_id: payload.allocation_id(),
        allocation_generation: payload.allocation_generation(),
        resident_generation,
        resident_sector_mask,
    };
    assert_eq!(core.l1_dirty_residency().sectors, 0);
    core.admit_l1_dirty_payloads_with_witness(&[payload], &[witness])
        .unwrap();
    assert_eq!(core.l1_dirty_residency().sectors, 1);

    // A wrong producer allocation must fail closed before any native
    // mutation; a line tag alone is never sufficient evidence.
    let wrong = sim_mem::SectorPayloadV1::new_with_generation(
        12,
        3,
        0,
        0,
        2,
        [0x6b; sim_mem::PAYLOAD_SECTOR_BYTES],
        u32::MAX,
    )
    .unwrap();
    let error = core
        .preview_l1_dirty_payload_admission(
            &[wrong],
            &[sim_mem::L1PayloadResidencyWitness {
                line_id: 0,
                allocation_id: payload.allocation_id(),
                allocation_generation: payload.allocation_generation(),
                resident_generation,
                resident_sector_mask,
            }],
        )
        .expect_err("allocation identity must be independently witnessed");
    assert!(matches!(
        error,
        IssueError::L1Access(
            sim_mem::L1AccessError::DirtyPayloadResidencyWitnessMismatch {
                field: "payload allocation identity",
                ..
            }
        )
    ));
    assert_eq!(core.l1_dirty_residency().sectors, 1);
}

#[test]
fn feedback_admission_requires_the_issue_models_operand_visibility_latency() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "EXIT"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0)])],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    let matching = admission_candidate(9, MemoryServiceCandidateScope::Sm);
    core.install_memory_admission_feedback(matching).unwrap();
    assert_eq!(
        core.memory_admission_feedback
            .as_ref()
            .unwrap()
            .diagnostics()
            .candidate,
        matching
    );

    let mut mismatched = matching;
    mismatched.admission_latency_cycles = 7;
    assert!(matches!(
        core.install_memory_admission_feedback(mismatched),
        Err(IssueError::MemoryAdmissionFeedbackLatencyMismatch {
            candidate_id: 9,
            configured_cycles: 7,
            issue_model_cycles: 8,
        })
    ));
    assert_eq!(
        core.memory_admission_feedback
            .as_ref()
            .unwrap()
            .diagnostics()
            .candidate,
        matching,
        "a rejected replacement must preserve the installed candidate"
    );
}

#[test]
fn sm_scoped_same_cycle_capacity_denial_repicks_non_candidate_work() {
    let sched = schedule(vec![
        (0, slot(cb(1), OpClass::Other, "STG.E")),
        (16, slot(cb(1), OpClass::Other, "IADD3")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![
            // Older ready work on SMSP 1.
            warp(1, vec![inst(16)]),
            // Younger STORE on the same SMSP wins the initial CGGTY pick,
            // but loses the SM-wide one-entry queue to SMSP 0.
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 5,
                insts: vec![ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalStore { off: 0, len: 1 },
                }],
                sectors: vec![4],
            },
            WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![ReplayInst {
                    lane_mask: u32::MAX,
                    pc: 0,
                    kind: AccessKind::GlobalStore { off: 0, len: 1 },
                }],
                sectors: vec![0],
            },
        ],
    }];
    let mut core = shared_core(&blocks, &sched, &params());
    core.install_memory_admission_feedback(admission_candidate(
        10,
        MemoryServiceCandidateScope::Sm,
    ))
    .unwrap();
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Idle { wake_at: Some(1) }
    );
    assert_eq!(core.instructions_issued(), 2);
    assert_eq!(requests.len(), 1);
    assert!(
        core.warps[0].done,
        "the non-candidate fallback must dispatch"
    );
    assert!(
        !core.warps[1].done,
        "the capacity-denied younger STORE must remain resident"
    );
    assert!(core.warps[2].done, "SMSP 0 owns the queue reservation");
}

#[test]
fn installed_memory_service_candidate_observes_real_issue_without_delaying_core() {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "STG.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 0,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind: AccessKind::GlobalStore { off: 0, len: 4 },
            }],
            sectors: vec![0, 1, 2, 3],
        }],
    }];
    let candidate = service_candidate(9, MemoryServiceCandidateScope::Sm);
    let mut core = shared_core(&blocks, &sched, &params());
    core.install_memory_service_candidates(&[candidate])
        .unwrap();
    let mut requests = Vec::new();

    assert_eq!(
        core.advance(ReplayHorizon::Unbounded, &mut requests)
            .unwrap(),
        Advance::Done
    );
    let result = core.result();
    assert_eq!(result.cycles, 1, "candidate must not delay the issue clock");
    assert_eq!(result.readiness.schema_version(), 7);
    let observed = result.readiness.memory_service_candidates.entries()[0];
    assert_eq!(observed.instructions_observed, 1);
    assert_eq!(observed.ideal_wavefronts_observed, 1);
    assert_eq!(observed.unique_lines_observed, 1);
    assert_eq!(observed.busy_cycles, 3);
    assert_eq!(observed.queue_delay_cycles, 0);
    assert_eq!(observed.terminal_completion_time, 8);
}

#[test]
fn global_store_operand_read_barrier_has_distinct_schema8_attribution() {
    let mut producer = cb(1);
    producer.read_barrier = Some(barrier(0));
    let mut consumer = cb(1);
    consumer.wait_mask = wait_mask(0b000001);
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
                    kind: AccessKind::GlobalStore { off: 0, len: 1 },
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
        Advance::Done
    );
    let result = core.result();
    assert_eq!(result.cycles, 9);
    assert_eq!(result.readiness.schema_version(), 8);
    assert_eq!(
        result
            .readiness
            .global_memory_issue
            .store_operand_read_barrier_instructions,
        1
    );
    assert_eq!(
        result
            .readiness
            .limiting_cycles
            .dependence_counter_global_store_operand_read,
        8
    );
    assert_eq!(result.readiness.limiting_cycles.dependence_counter_read, 0);
}
