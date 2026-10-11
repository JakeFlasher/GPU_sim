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

fn construction_error(kind: AccessKind, sectors: Vec<i64>) -> IssueError {
    let sched = schedule(vec![(0, slot(cb(1), OpClass::Other, "LDG.E"))]);
    let blocks = vec![BlockReplay {
        launch_index: 7,
        warps: vec![WarpReplay {
            async_copy_cohorts: None,
            warp_slot: 3,
            insts: vec![ReplayInst {
                lane_mask: u32::MAX,
                pc: 0,
                kind,
            }],
            sectors,
        }],
    }];
    let parameters = params();
    match SmCore::new(&blocks, &sched, &parameters, SmMemMode::Flat) {
        Err(error) => error,
        Ok(_) => panic!("invalid sector span constructed replay state"),
    }
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
        lane_mask: u32::MAX,
        pc,
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
        lane_mask: u32::MAX,
        pc,
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
                    lane_mask: u32::MAX,
                    pc: 0,
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

fn paired_reservation_projection(slow_expensive: bool) -> (BypassAsyncCopyProjection, Option<u64>) {
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

mod admission;
mod async_copy;
mod configuration;
mod diagnostics;
mod front_end;
mod memory;
mod reference;
mod residency;
mod scheduling;
