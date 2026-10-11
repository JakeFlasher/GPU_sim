
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use sim_mem::{AccessOutcome, CacheOpToken, L1CounterError, L1CounterField};

    use super::*;
    use crate::control_bits::{ControlBits, DepBarrier, WaitMask};
    use crate::program::{
        FunctionSchedule, IssueSlot, OpClass, PipelineClass, is_clocklo_opcode_text,
        is_gpu_membar_opcode_text,
    };

    fn cb(stall: u8) -> ControlBits {
        ControlBits::new(stall, false, 7, 7, 0, 0).expect("valid test controls")
    }

    fn barrier(value: u8) -> DepBarrier {
        DepBarrier::new(value).expect("valid test barrier")
    }

    fn wait_mask(bits: u8) -> WaitMask {
        WaitMask::new(bits).expect("valid test wait mask")
    }

    fn slot(control: ControlBits, op_class: OpClass, text: &str) -> IssueSlot {
        IssueSlot {
            control,
            op_class,
            async_copy_wait_max_pending_groups: 0,
            pipeline_class: PipelineClass::DispatchOnly,
            mem_role: if text.trim_start().starts_with("LDGSTS") {
                sim_mem::MemRole::BypassLoad(CacheOpToken::Ldgsts)
            } else {
                sim_mem::MemRole::None
            },
            ldgsts_l1_policy: text.trim_start().starts_with("LDGSTS").then_some(
                if text.split('.').any(|modifier| modifier == "BYPASS") {
                    LdgstsL1Policy::BypassL1
                } else {
                    LdgstsL1Policy::CacheAtAllLevels
                },
            ),
            fixed_latency_operands: crate::sass_operands::project_fixed_latency_operands(
                text,
                control,
                ArchId::Sm86,
            ),
            clocklo_read: is_clocklo_opcode_text(text),
            gpu_membar: is_gpu_membar_opcode_text(text),
            opcode_text: text.to_owned(),
        }
    }

    fn barrier_pipeline_slot(control: ControlBits) -> IssueSlot {
        let mut result = slot(control, OpClass::BlockBarrier, "BAR.SYNC.DEFER_BLOCKING");
        result.pipeline_class = PipelineClass::Barrier {
            warp_initiation_interval_cycles: 2,
            generation_rearm_cycles: 0,
        };
        result
    }

    fn branch_pipeline_slot(control: ControlBits) -> IssueSlot {
        let mut result = slot(control, OpClass::Other, "BRA 0x20");
        result.pipeline_class = PipelineClass::Branch {
            next_issue_interval_cycles: 10,
        };
        result
    }

    fn tensor_slot(control: ControlBits, text: &str, interval: u64) -> IssueSlot {
        IssueSlot {
            pipeline_class: PipelineClass::Tensor {
                initiation_interval_cycles: interval,
            },
            ..slot(control, OpClass::Other, text)
        }
    }

    fn schedule(slots: Vec<(u64, IssueSlot)>) -> FunctionSchedule {
        FunctionSchedule {
            arch: ArchId::Sm86,
            function_unique_id: 1,
            kernel_name: "k___0".to_owned(),
            function_name: "k".to_owned(),
            by_pc: slots.into_iter().collect::<BTreeMap<_, _>>(),
        }
    }

    #[test]
    fn replay_instruction_stays_compact() {
        assert!(std::mem::size_of::<ReplayInst>() <= 24);
    }

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

    fn params() -> IssueModelParams {
        IssueModelParams {
            arch: ArchId::Sm86,
            resource_source: IssueResourceSource::SyntheticTestFixture,
            instruction_front_end: None,
            memory_issue_pipeline: None,
            scalar_execution_pipeline: None,
            issue_ports: 1,
            dual_issue_supported: false,
            ldgsts_l1_hit_return_service: None,
            constant_memory_return_service: None,
            immediate_constant_cache: None,
            l1_hit_latency: 34,
            l1_miss_latency: 244,
            global_flat_latency: 400,
            shared_write_barrier_latency: 23,
            shared_additional_wavefront_latency: 2,
            non_memory_write_barrier_latency: 25,
            read_barrier_latency: 8,
            ldgsts_l1_mshr_floor: NonZeroU32::new(1).unwrap(),
            ldgsts_completion_throughput: LdgstsCompletionThroughput {
                cache_all_payload_bytes_per_cycle: 64,
                bypass_payload_bytes_per_cycle: 32,
            },
            ldgsts_cache_all_destination_wavefront_cycles: 2,
            ldgsts_global_to_shared_transport_latency: 0,
            ldgsts_cache_all_destination_service_wavefronts: 2,
            ldgsts_cache_all_destination_service_cycles: 3,
            max_resident_warps_per_sm: 48,
            max_resident_blocks_per_sm: u32::MAX,
        }
    }

    fn card_path(name: &str) -> PathBuf {
        PathBuf::from("../../arch_profiles/cards")
            .join(name)
            .canonicalize()
            .expect("card exists in-repo")
    }

    fn synthetic_carded_issue_resources() -> ArchCard {
        let path = card_path("sm_86.json");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        // Fixture-only sentinels. Matching legacy values exercises lane
        // separation; it does not turn them into architecture-card values.
        value["smsp"] = serde_json::json!({"count": 4, "issue_ports": 1});
        value["occupancy"] = serde_json::json!({
            "max_resident_warps_per_sm": 48,
            "max_blocks": 16,
        });
        value["dep_barriers"] = serde_json::json!({"count": 6});
        value["barrier_latencies"] = serde_json::json!({
            "read": 8,
            "non_memory_write": 25,
        });
        value["dual_issue"] = serde_json::json!({"supported": false});
        ArchCard::parse(&path, &serde_json::to_vec(&value).unwrap()).unwrap()
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
            crate::replay::ReplayConfiguration::checked(
                non_memory,
                crate::replay::ReplayMemMode::Flat,
            ),
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

    fn construction_error(kind: AccessKind, sectors: Vec<i64>) -> IssueError {
        let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
        let blocks = vec![BlockReplay {
            launch_index: 7,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 3,
                insts: vec![ReplayInst { lane_mask: u32::MAX, pc: 0, kind }],
                sectors,
            }],
        }];
        let parameters = params();
        match SmCore::new(&blocks, &sched, &parameters, SmMemMode::Flat) {
            Err(error) => error,
            Ok(_) => panic!("invalid sector span constructed replay state"),
        }
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
                    lane_mask: u32::MAX, pc: 0,
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
        let Err(stall_error) = SmCore::new(&blocks, &stall_schedule, &params(), SmMemMode::Flat)
        else {
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
        let Err(reuse_error) = SmCore::new(&blocks, &reuse_schedule, &params(), SmMemMode::Flat)
        else {
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
                    lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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

    fn l1_config() -> L1Config {
        L1Config {
            arch: ArchId::Sm86,
            line_capacity: 768,
            mshr_count: Some(32),
            hit_latency: 34,
            miss_latency: 244,
        }
    }

    fn eligible(off: usize, len: u16) -> AccessKind {
        AccessKind::GlobalEligible {
            off,
            len,
            token: CacheOpToken::LdgECa,
        }
    }

    fn inst(pc: u32) -> ReplayInst {
        ReplayInst {
            lane_mask: u32::MAX, pc,
            kind: AccessKind::NonMemory,
        }
    }

    fn async_copy_inst_with_policy(
        pc: u32,
        off: usize,
        len: u16,
        l1_policy: LdgstsL1Policy,
    ) -> ReplayInst {
        let payload_bytes = u32::from(len) * 32;
        let completion_work = match l1_policy {
            LdgstsL1Policy::CacheAtAllLevels => payload_bytes,
            LdgstsL1Policy::BypassL1 => payload_bytes * 2,
        };
        ReplayInst {
            lane_mask: u32::MAX, pc,
            kind: AccessKind::GlobalAsyncCopy {
                off,
                len,
                completion_work,
                l1_policy,
            },
        }
    }

    fn async_copy_inst(pc: u32, off: usize, len: u16) -> ReplayInst {
        async_copy_inst_with_policy(pc, off, len, LdgstsL1Policy::CacheAtAllLevels)
    }

    fn bypass_async_copy_inst(pc: u32, off: usize, len: u16) -> ReplayInst {
        async_copy_inst_with_policy(pc, off, len, LdgstsL1Policy::BypassL1)
    }

    fn async_commit_slot() -> IssueSlot {
        slot(
            ControlBits::new(1, true, 0, 7, 0, 0).unwrap(),
            OpClass::AsyncCopyCommit,
            "LDGDEPBAR",
        )
    }

    fn async_wait_slot(max_pending_groups: u32) -> IssueSlot {
        IssueSlot {
            async_copy_wait_max_pending_groups: max_pending_groups,
            ..slot(
                cb(1),
                OpClass::AsyncCopyWait,
                &format!("DEPBAR.LE SB0, {max_pending_groups}"),
            )
        }
    }

    fn warp(slot_ix: i32, insts: Vec<ReplayInst>) -> WarpReplay {
        WarpReplay {
            async_copy_cohorts: None,
            warp_slot: slot_ix,
            insts,
            sectors: Vec::new(),
        }
    }

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
                            lane_mask: u32::MAX, pc: 0,
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

    fn result_priority_case(load_destination: u16, producer_cycle: u8) -> SmSimResult {
        let mut load_control = ControlBits::new(15, false, 0, 7, 0, 0).unwrap();
        load_control.write_barrier = Some(barrier(0));
        let mut consumer_control = cb(1);
        consumer_control.wait_mask = wait_mask(0b000001);
        let mut load_slot = slot(
            load_control,
            OpClass::Other,
            &format!("LDG.E.STRONG.SM R{load_destination}, [R2.64]"),
        );
        load_slot.mem_role = MemRole::EligibleLoad(CacheOpToken::LdgEStrongSm);
        let sched = schedule(vec![
            (0, load_slot),
            (
                16,
                slot(
                    cb(producer_cycle - 15),
                    OpClass::Other,
                    "IADD3 R10, R10, 0x1, RZ",
                ),
            ),
            (32, slot(cb(1), OpClass::Other, "HADD2 R9, R9, R6")),
            (
                48,
                slot(cb(1), OpClass::Other, "ISETP.NE.AND P0, PT, R10, 0x40, PT"),
            ),
            (64, slot(cb(29 - producer_cycle - 2), OpClass::Other, "NOP")),
            (
                80,
                slot(
                    consumer_control,
                    OpClass::Other,
                    "LOP3.LUT R12, R12, R11, R7, 0x96, !PT",
                ),
            ),
        ]);
        let blocks = vec![BlockReplay {
            launch_index: 0,
            warps: vec![WarpReplay {
                async_copy_cohorts: None,
                warp_slot: 0,
                insts: vec![
                    ReplayInst {
                        lane_mask: u32::MAX, pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgEStrongSm,
                        },
                    },
                    inst(16),
                    inst(32),
                    inst(48),
                    inst(64),
                    inst(80),
                ],
                sectors: vec![0],
            }],
        }];
        let mut core =
            SmCore::new(&blocks, &sched, &params(), SmMemMode::L1Const(l1_config())).unwrap();
        let l1 = core.mem.const_l1_mut().unwrap();
        l1.access_eligible_unobserved(MemoryOrigin::Global, 0, 0)
            .unwrap();
        l1.access_eligible_unobserved(MemoryOrigin::Global, 0, 244)
            .unwrap();
        core.t = 244;
        let mut out = Vec::new();
        assert_eq!(
            core.advance(ReplayHorizon::Unbounded, &mut out).unwrap(),
            Advance::Done
        );
        assert!(out.is_empty());
        core.result()
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
                                lane_mask: u32::MAX, pc: 0,
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
                            lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
                        kind: AccessKind::GlobalAsyncCopy {
                            off: 0,
                            len: 1,
                            completion_work: sim_mem::encode_async_copy_destination_work(128)
                                .unwrap(),
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
                        lane_mask: u32::MAX, pc: 0,
                        kind: AccessKind::GlobalAsyncCopy {
                            off: 0,
                            len: 1,
                            completion_work: sim_mem::encode_async_copy_destination_work(128)
                                .unwrap(),
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
                    lane_mask: u32::MAX, pc: 0,
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
                            lane_mask: u32::MAX, pc: 0,
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
                            lane_mask: u32::MAX, pc: 16,
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
                    lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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

    fn exact_two_cohort_input<'a>(
        descriptors: &'a [AsyncCopyCohortDescriptor],
        masks: &'a [u32],
    ) -> BypassAsyncCopyInput<'a> {
        BypassAsyncCopyInput {
            instruction: 10,
            sector_off: 20,
            sector_len: 3,
            source_cohorts: 2,
            destination_lines: 3,
            max_bank_wavefronts: 2,
            max_bypass_wavefronts: 3,
            cohorts: Some(descriptors),
            sector_membership_masks: Some(masks),
        }
    }

    fn paired_reservation_projection(
        slow_expensive: bool,
    ) -> (BypassAsyncCopyProjection, Option<u64>) {
        let compact = AsyncCopyCohortDescriptor {
            payload_bytes: 32,
            destination_lines: 1,
            bank_wavefronts: 1,
            bypass_wavefronts: 1,
        };
        let severe = AsyncCopyCohortDescriptor {
            payload_bytes: 32,
            destination_lines: 4,
            bank_wavefronts: 4,
            bypass_wavefronts: 4,
        };
        let descriptors = if slow_expensive {
            [compact, severe]
        } else {
            [severe, compact]
        };
        let masks = [0b01, 0b10];
        let p = params();
        let mut slot = AsyncCopySlot::default();
        let group = slot
            .add_pending_async_copy_sources(
                0,
                2,
                128,
                0,
                Some(BypassAsyncCopyInput {
                    instruction: 10,
                    sector_off: 20,
                    sector_len: 2,
                    source_cohorts: 2,
                    destination_lines: 5,
                    max_bank_wavefronts: 4,
                    max_bypass_wavefronts: 4,
                    cohorts: Some(&descriptors),
                    sector_membership_masks: Some(&masks),
                }),
                &p,
            )
            .unwrap();
        assert!(slot.commit_async_copy_group(10));
        for (sector_index, now) in [(20, 20), (21, 40)] {
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
        let committed = slot.state.as_deref().unwrap().committed_groups[0];
        let projection = slot
            .bypass_copy_projection(0, &committed, &p, 45)
            .unwrap()
            .unwrap();
        (projection, slot.async_wait_clear_time(0))
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
                        lane_mask: u32::MAX, pc: 0,
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
                            lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                            lane_mask: u32::MAX, pc: 0,
                            kind: eligible(0, 1),
                        },
                        inst(16),
                        ReplayInst {
                            lane_mask: u32::MAX, pc: 32,
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
                        lane_mask: u32::MAX, pc: 0,
                        kind: AccessKind::GlobalBypass {
                            off: 0,
                            len: 2,
                            token: CacheOpToken::LdgECg,
                            l2_requests: 0,
                        },
                    },
                    ReplayInst {
                        lane_mask: u32::MAX, pc: 16,
                        kind: AccessKind::GlobalUntagged { off: 2, len: 1 },
                    },
                    ReplayInst {
                        lane_mask: u32::MAX, pc: 32,
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
                        lane_mask: u32::MAX, pc: 0,
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

    fn service_candidate(
        candidate_id: u8,
        scope: MemoryServiceCandidateScope,
    ) -> MemoryServiceCandidate {
        MemoryServiceCandidate {
            candidate_id,
            scope,
            operation: MemoryServiceCandidateOperation::Store,
            ideal_wavefront_units: 2,
            ideal_wavefront_cycles: 3,
            unique_line_units: 1,
            unique_line_cycles: 1,
            completion_latency_cycles: 5,
        }
    }

    fn admission_candidate(
        candidate_id: u8,
        scope: MemoryServiceCandidateScope,
    ) -> MemoryAdmissionCandidate {
        MemoryAdmissionCandidate {
            candidate_id,
            scope,
            operation: MemoryServiceCandidateOperation::Store,
            queue_capacity_instructions: 1,
            admission_latency_cycles: 8,
            l2_hit_sector_units: 1,
            l2_hit_sector_cycles: 1,
            l2_miss_sector_units: 1,
            l2_miss_sector_cycles: 4,
            l2_miss_service_width_sectors: 0,
            membar_waits_for_service_completion: false,
            write_completion_latency_cycles: 0,
            write_completion_request_cycles: 0,
            write_completion_region_cycles: 0,
            write_completion_parallel_lanes: 0,
        }
    }

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
        let mut candidates = [service_candidate(0, MemoryServiceCandidateScope::Sm);
            MAX_MEMORY_SERVICE_CANDIDATES + 1];
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
            SmCore::new_with_residency(&blocks, &sched, &params(), residency, SmMemMode::Flat)
                .unwrap();
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

    // ------------------------------------------------------------------
    // SmCore event-mode (L1Shared) tests: a manual driver stands in for
    // the system layer.
    // ------------------------------------------------------------------

    fn shared_core<'a>(
        blocks: &'a [BlockReplay],
        sched: &'a FunctionSchedule,
        p: &IssueModelParams,
    ) -> SmCore<'a> {
        SmCore::new(blocks, sched, p, SmMemMode::L1Shared(l1_config())).unwrap()
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
                    lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
                        kind: AccessKind::GlobalStore { off: 0, len: 1 },
                    }],
                    sectors: vec![4],
                },
                WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot: 0,
                    insts: vec![ReplayInst {
                        lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                            lane_mask: u32::MAX, pc: 0,
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
                            lane_mask: u32::MAX, pc: 16,
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
                    lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
            lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
                        kind: eligible(0, 1),
                    }],
                    sectors: vec![0],
                },
                WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot: 1,
                    insts: vec![ReplayInst {
                        lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
                        kind: eligible(0, 1),
                    }],
                    sectors: vec![0],
                },
                WarpReplay {
                    async_copy_cohorts: None,
                    warp_slot: 1,
                    insts: vec![ReplayInst {
                        lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                            lane_mask: u32::MAX, pc: 0,
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
                            lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                    lane_mask: u32::MAX, pc: 0,
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
                        lane_mask: u32::MAX, pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        lane_mask: u32::MAX, pc: 16,
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
                        lane_mask: u32::MAX, pc: 0,
                        kind: AccessKind::GlobalEligible {
                            off: 0,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        lane_mask: u32::MAX, pc: 16,
                        kind: AccessKind::GlobalEligible {
                            off: 1,
                            len: 1,
                            token: CacheOpToken::LdgECa,
                        },
                    },
                    ReplayInst {
                        lane_mask: u32::MAX, pc: 32,
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
                        lane_mask: u32::MAX, pc: 0,
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