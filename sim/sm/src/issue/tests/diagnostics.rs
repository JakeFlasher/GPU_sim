use super::*;

#[test]
fn readiness_aggregation_overflow_is_transactional() {
    assert_eq!(ReadinessDiagnostics::default().schema_version(), 1);
    let mut aggregate = ReadinessDiagnostics::default();
    aggregate.limiting_cycles.control_stall = u64::MAX;
    aggregate.cta_barriers.max_arrival_spread_cycles = 19;
    let before = aggregate.clone();

    let mut addend = ReadinessDiagnostics::default();
    addend.limiting_cycles.control_stall = 1;
    addend.cta_barriers.max_arrival_spread_cycles = 23;

    assert!(matches!(
        aggregate.add_checked(&addend),
        Err(ReadinessCounterError::Overflow {
            field: "readiness.control_stall",
            accumulated: u64::MAX,
            addend: 1,
        })
    ));
    assert_eq!(aggregate, before);
}

#[test]
fn async_wait_prefix_aggregation_is_checked_and_transactional() {
    let observation = ReadinessDiagnostics {
        async_wait_prefix: Some(Box::new(AsyncWaitPrefixDiagnostics {
            partial: AsyncWaitClassDiagnostics {
                wait_count: 1,
                immediate_sum: 2,
                committed_groups_before: 3,
                consumed_prefix_groups: 1,
                groups_remaining: 2,
                exposure_cycles: 17,
            },
            draining: AsyncWaitClassDiagnostics {
                wait_count: 1,
                immediate_sum: 0,
                committed_groups_before: 2,
                consumed_prefix_groups: 2,
                groups_remaining: 0,
                exposure_cycles: 11,
            },
        })),
        ..ReadinessDiagnostics::default()
    };
    let mut aggregate = ReadinessDiagnostics::default();
    aggregate.add_checked(&observation).unwrap();
    aggregate.add_checked(&observation).unwrap();

    let waits = aggregate.async_wait_prefix.as_deref().unwrap();
    assert_eq!(aggregate.schema_version(), 19);
    assert_eq!(waits.partial.wait_count, 2);
    assert_eq!(waits.partial.immediate_sum, 4);
    assert_eq!(waits.partial.committed_groups_before, 6);
    assert_eq!(waits.partial.consumed_prefix_groups, 2);
    assert_eq!(waits.partial.groups_remaining, 4);
    assert_eq!(waits.partial.exposure_cycles, 34);
    assert_eq!(waits.draining.wait_count, 2);
    assert_eq!(waits.draining.committed_groups_before, 4);
    assert_eq!(waits.draining.consumed_prefix_groups, 4);
    assert_eq!(waits.draining.exposure_cycles, 22);

    aggregate
        .async_wait_prefix
        .as_deref_mut()
        .unwrap()
        .partial
        .exposure_cycles = u64::MAX;
    let before = aggregate.clone();
    assert!(matches!(
        aggregate.add_checked(&observation),
        Err(ReadinessCounterError::Overflow {
            field: "readiness.async_wait_prefix.partial.exposure_cycles",
            accumulated: u64::MAX,
            addend: 17,
        })
    ));
    assert_eq!(aggregate, before);
}

#[test]
fn async_issue_overlap_aggregation_is_checked_and_transactional() {
    let observation = ReadinessDiagnostics {
        async_issue_overlap: Some(Box::new(AsyncIssueOverlapDiagnostics {
            coissue_cycles: 1,
            coissued_copies: 4,
            cache_all_coissued_copies: 2,
            bypass_coissued_copies: 2,
            mixed_policy_coissue_cycles: 1,
            single_acceptor_excess_copies: 3,
            maximum_coissue_width: 4,
            copies_with_other_warp_committed_groups: 2,
            other_warp_committed_groups_sum: 7,
            maximum_other_warp_committed_groups: 5,
        })),
        ..ReadinessDiagnostics::default()
    };
    let mut aggregate = ReadinessDiagnostics::default();
    aggregate.add_checked(&observation).unwrap();
    aggregate.add_checked(&observation).unwrap();

    assert_eq!(aggregate.schema_version(), 20);
    assert_eq!(
        aggregate.async_issue_overlap.as_deref(),
        Some(&AsyncIssueOverlapDiagnostics {
            coissue_cycles: 2,
            coissued_copies: 8,
            cache_all_coissued_copies: 4,
            bypass_coissued_copies: 4,
            mixed_policy_coissue_cycles: 2,
            single_acceptor_excess_copies: 6,
            maximum_coissue_width: 4,
            copies_with_other_warp_committed_groups: 4,
            other_warp_committed_groups_sum: 14,
            maximum_other_warp_committed_groups: 5,
        })
    );

    aggregate
        .async_issue_overlap
        .as_deref_mut()
        .unwrap()
        .other_warp_committed_groups_sum = u64::MAX;
    let before = aggregate.clone();
    assert!(matches!(
        aggregate.add_checked(&observation),
        Err(ReadinessCounterError::Overflow {
            field: "readiness.async_issue_overlap.other_warp_committed_groups_sum",
            accumulated: u64::MAX,
            addend: 7,
        })
    ));
    assert_eq!(aggregate, before);
}

#[test]
fn scalar_global_output_wavefront_projection_preserves_atomic_aliases() {
    let unique_line_sectors: Vec<i64> = (0..32).map(|value| value * 4).collect();
    assert_eq!(
        projected_global_memory_issue(
            AccessKind::GlobalStore { off: 0, len: 32 },
            &unique_line_sectors,
        ),
        Some((GlobalMemoryIssueClass::Store, 32, 8, 32, 0))
    );
    assert_eq!(
        projected_global_memory_issue(
            AccessKind::GlobalReduction {
                off: 0,
                len: 4,
                l2_requests: 1,
            },
            &[0, 1, 2, 3],
        ),
        Some((GlobalMemoryIssueClass::Reduction, 4, 1, 1, 1))
    );
    assert_eq!(
        projected_global_memory_issue(
            AccessKind::GlobalReduction {
                off: 0,
                len: 32,
                l2_requests: 32,
            },
            &[7; 32],
        ),
        Some((GlobalMemoryIssueClass::Reduction, 32, 32, 1, 32))
    );
    assert_eq!(
        projected_global_memory_issue(
            AccessKind::GlobalBypass {
                off: 0,
                len: 1,
                token: CacheOpToken::Ldgsts,
                l2_requests: 0,
            },
            &[0],
        ),
        None
    );
}

#[test]
fn large_sector_projection_preserves_alias_and_line_counts() {
    let sectors: Vec<i64> = (0..4096).map(|index| i64::from(index % 8)).collect();
    assert_eq!(maximum_sector_multiplicity(&sectors), 512);
    assert_eq!(unique_128b_lines(&sectors), 2);
}

#[test]
fn scalar_global_operand_read_origins_are_operation_specific() {
    assert_eq!(
        operand_read_dependency_origin(eligible(0, 1)),
        DependencyOrigin::GlobalLoadOperandRead
    );
    assert_eq!(
        operand_read_dependency_origin(AccessKind::GlobalStore { off: 0, len: 1 }),
        DependencyOrigin::GlobalStoreOperandRead
    );
    assert_eq!(
        operand_read_dependency_origin(AccessKind::GlobalBypass {
            off: 0,
            len: 1,
            token: CacheOpToken::Atom,
            l2_requests: 1,
        }),
        DependencyOrigin::GlobalAtomicOperandRead
    );
    assert_eq!(
        operand_read_dependency_origin(AccessKind::GlobalReduction {
            off: 0,
            len: 1,
            l2_requests: 1,
        }),
        DependencyOrigin::GlobalReductionOperandRead
    );
    assert_eq!(
        operand_read_dependency_origin(async_copy_inst(0, 0, 1).kind),
        DependencyOrigin::Read
    );
}

#[test]
fn write_lookup_observer_uses_two_sectors_per_wavefront_allocation() {
    let sectors = (0_i64..8).collect::<Vec<_>>();
    assert_eq!(
        projected_l1_write_lookup_hits(AccessKind::GlobalStore { off: 0, len: 8 }, &sectors,),
        Some(4)
    );
    assert_eq!(
        projected_l1_write_lookup_hits(AccessKind::GlobalStore { off: 0, len: 2 }, &sectors,),
        Some(1)
    );
    assert_eq!(
        projected_l1_write_lookup_hits(AccessKind::GlobalStore { off: 0, len: 1 }, &sectors,),
        Some(0)
    );
    assert_eq!(
        projected_l1_write_lookup_hits(eligible(0, 4), &sectors),
        None
    );
}

#[test]
fn global_memory_issue_diagnostics_aggregate_and_overflow_transactionally() {
    let mut aggregate = ReadinessDiagnostics::default();
    aggregate
        .global_memory_issue
        .observe(GlobalMemoryIssueClass::Load, 4, 1, 1, 0, false, 10)
        .unwrap();
    aggregate
        .global_memory_issue
        .observe(GlobalMemoryIssueClass::Store, 32, 8, 32, 0, true, 30)
        .unwrap();
    let mut addend = ReadinessDiagnostics::default();
    addend
        .global_memory_issue
        .observe(GlobalMemoryIssueClass::Reduction, 32, 32, 1, 32, true, 7)
        .unwrap();
    aggregate.add_checked(&addend).unwrap();
    assert_eq!(aggregate.global_memory_issue.load_instructions, 1);
    assert_eq!(aggregate.global_memory_issue.store_ideal_wavefronts, 8);
    assert_eq!(aggregate.global_memory_issue.reduction_ideal_wavefronts, 32);
    assert_eq!(aggregate.global_memory_issue.store_unique_lines, 32);
    assert_eq!(aggregate.global_memory_issue.reduction_l2_requests, 32);
    assert_eq!(
        aggregate
            .global_memory_issue
            .store_operand_read_barrier_instructions,
        1
    );
    assert_eq!(
        aggregate
            .global_memory_issue
            .reduction_operand_read_barrier_instructions,
        1
    );
    assert_eq!(aggregate.global_memory_issue.total_issue_span_cycles, 20);
    assert_eq!(aggregate.global_memory_issue.max_issue_span_cycles, 20);
    assert_eq!(aggregate.schema_version(), 8);

    aggregate.global_memory_issue.store_sectors = u64::MAX;
    addend.global_memory_issue.store_sectors = 1;
    let before = aggregate.clone();
    assert!(matches!(
        aggregate.add_checked(&addend),
        Err(ReadinessCounterError::Overflow {
            field: "readiness.global_memory_issue.store_sectors",
            accumulated: u64::MAX,
            addend: 1,
        })
    ));
    assert_eq!(aggregate, before);
}
