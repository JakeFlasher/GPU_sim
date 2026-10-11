#[cfg(feature = "reference")]
use super::*;

#[cfg(feature = "reference")]
#[test]
fn reference_scalar_macro_installs_native_boundary_and_continues_detailed_issue() {
    let mut scalar = slot(cb(1), OpClass::Other, "IADD3 R1, R1, 1");
    scalar.pipeline_class = PipelineClass::ScalarAlu {
        latency_cycles: 1,
        initiation_interval_cycles: 1,
        in_flight_capacity: 8,
    };
    let schedule = schedule(vec![(0, scalar)]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(0), inst(0)])],
    }];
    let params = params();
    let mut detailed = SmCore::new_with_residency(
        &blocks,
        &schedule,
        &params,
        SharedMemoryResidency::unconstrained_fixture(),
        SmMemMode::Flat,
    )
    .unwrap();
    let mut detailed_requests = Vec::new();
    assert!(matches!(
        detailed.advance(ReplayHorizon::Bounded(2), &mut detailed_requests),
        Ok(Advance::Idle { .. })
    ));

    let mut macro_core = SmCore::new_with_residency(
        &blocks,
        &schedule,
        &params,
        SharedMemoryResidency::unconstrained_fixture(),
        SmMemMode::Flat,
    )
    .unwrap();
    let preserve = sim_semantics::native_boundary_execution_path(
        &sim_semantics::NativeBoundaryRefinementDecision::Preserve,
    );
    let transition = macro_core
        .execute_reference_macro_path(&preserve, 0, 2, 1, false, &[], None)
        .unwrap()
        .expect("preserve path must execute a macro");
    assert_eq!(transition.first_issue, 0);
    assert_eq!(transition.last_issue, 1);
    assert_eq!(transition.end_cycle, 2);
    assert_eq!(detailed.boundary_snapshot(), macro_core.boundary_snapshot());

    let mut detailed_requests = Vec::new();
    let mut macro_requests = Vec::new();
    assert!(matches!(
        detailed.advance(ReplayHorizon::Unbounded, &mut detailed_requests),
        Ok(Advance::Done)
    ));
    assert!(matches!(
        macro_core.advance(ReplayHorizon::Unbounded, &mut macro_requests),
        Ok(Advance::Done)
    ));
    assert_eq!(detailed.result().instructions_issued, 3);
    assert_eq!(macro_core.result().instructions_issued, 3);
    assert_eq!(detailed.result().cycles, macro_core.result().cycles);

    let mut fallback_core = SmCore::new_with_residency(
        &blocks,
        &schedule,
        &params,
        SharedMemoryResidency::unconstrained_fixture(),
        SmMemMode::Flat,
    )
    .unwrap();
    let fallback = sim_semantics::native_boundary_execution_path(
        &sim_semantics::NativeBoundaryRefinementDecision::Reify {
            reason: sim_semantics::NativeBoundaryReifyReason::ProjectionMismatch,
        },
    );
    assert!(
        fallback_core
            .execute_reference_macro_path(&fallback, 0, 2, 1, false, &[], None)
            .unwrap()
            .is_none()
    );
}

#[cfg(feature = "reference")]
#[test]
fn reference_scalar_macro_rejects_descriptor_mismatch_without_mutation() {
    let mut scalar = slot(cb(1), OpClass::Other, "IADD3 R1, R1, 1");
    scalar.pipeline_class = PipelineClass::ScalarAlu {
        latency_cycles: 1,
        initiation_interval_cycles: 1,
        in_flight_capacity: 8,
    };
    let schedule = schedule(vec![
        (0, scalar),
        (16, slot(cb(1), OpClass::Other, "BRA 0x20")),
    ]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16)])],
    }];
    let mut core = SmCore::new_with_residency(
        &blocks,
        &schedule,
        &params(),
        SharedMemoryResidency::unconstrained_fixture(),
        SmMemMode::Flat,
    )
    .unwrap();
    let before = core.boundary_snapshot();
    let typed_macro = sim_semantics::native_boundary_execution_path(
        &sim_semantics::NativeBoundaryRefinementDecision::Preserve,
    );
    assert!(matches!(
        core.execute_reference_macro_path(&typed_macro, 0, 2, 1, false, &[], None),
        Err(IssueError::ReferenceMacroUnsupported {
            reason: "scalar_macro_descriptor_not_closed"
        })
    ));
    assert_eq!(core.boundary_snapshot(), before);
}

#[cfg(feature = "reference")]
#[test]
fn reference_scalar_macro_rejects_truncated_detailed_effects_without_mutation() {
    let mut first = slot(cb(1), OpClass::Other, "IADD3 R1, R1, 1");
    first.pipeline_class = PipelineClass::ScalarAlu {
        latency_cycles: 1,
        initiation_interval_cycles: 1,
        in_flight_capacity: 8,
    };
    let mut second = slot(cb(1), OpClass::Other, "IADD3 R2, R2, 1");
    second.pipeline_class = PipelineClass::ScalarAlu {
        latency_cycles: 1,
        initiation_interval_cycles: 1,
        in_flight_capacity: 8,
    };
    let schedule = schedule(vec![(0, first), (16, second)]);
    let blocks = vec![BlockReplay {
        launch_index: 0,
        warps: vec![warp(0, vec![inst(0), inst(16)])],
    }];
    let mut core = SmCore::new_with_residency(
        &blocks,
        &schedule,
        &params(),
        SharedMemoryResidency::unconstrained_fixture(),
        SmMemMode::Flat,
    )
    .unwrap();
    let before = core.boundary_snapshot();
    let typed_macro = sim_semantics::native_boundary_execution_path(
        &sim_semantics::NativeBoundaryRefinementDecision::Preserve,
    );
    let truncated = vec![TypedEventTransition {
        phase: 0,
        event: sim_semantics::TypedEventIdentity {
            time: 0,
            sequence: 1,
            sequence_span: 1,
            phase: 0,
            target: 0,
            payload: sim_semantics::TypedEventPayload::SmWake,
        },
    }];
    assert!(matches!(
        core.execute_reference_macro_path(&typed_macro, 0, 2, 1, false, &truncated, None),
        Err(IssueError::ReferenceMacroUnsupported {
            reason: "detailed_effect_span_not_closed"
        })
    ));
    assert_eq!(core.boundary_snapshot(), before);
}
