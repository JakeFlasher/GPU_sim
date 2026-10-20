//! Interleaved response and service processing; mutates active event state.

use super::*;

pub(crate) const L2_SOURCES: [L2Source; 4] = [
    L2Source::L1Fill,
    L2Source::Bypass,
    L2Source::Untagged,
    L2Source::Store,
];

pub(crate) fn require_arch(
    expected: ArchId,
    got: ArchId,
    component: &'static str,
) -> Result<(), SystemError> {
    if got != expected {
        return Err(SystemError::ArchitectureMismatch {
            component,
            expected,
            got,
        });
    }
    Ok(())
}

pub(crate) fn require_l2_pending_read_semantics(
    arch: ArchId,
    effective_l2: bool,
) -> Result<(), SystemError> {
    if effective_l2 {
        if arch != ArchId::Sm120 {
            return Err(SystemError::ArchitectureMismatch {
                component: "effective L2 scenario",
                expected: ArchId::Sm120,
                got: arch,
            });
        }
        // The effective path explicitly selects the same allocation/sector
        // merge rule used by the implemented L2 engine. This is a scenario
        // assumption, not an SM120 architectural claim.
        return Ok(());
    }
    match arch
        .l2_pending_read_semantics()
        .require("l2_pending_read_semantics")?
    {
        L2PendingReadSemantics::AllocationSectorMergeV1 => Ok(()),
    }
}

pub(crate) fn checked_l2_counters(
    expected: ArchId,
    l2: &L2Model,
) -> Result<&L2Counters, SystemError> {
    let counters = l2.counters().map_err(sim_mem::L2AccessError::Counter)?;
    let got = counters.arch();
    let got = got.ok_or(sim_mem::L2AccessError::Counter(
        sim_mem::L2CounterError::Architecture(sim_arch::ArchitectureProvenanceError::Unbound {
            component: "L2 counters",
        }),
    ))?;
    if got != expected {
        return Err(
            sim_mem::L2AccessError::Counter(sim_mem::L2CounterError::Architecture(
                sim_arch::ArchitectureProvenanceError::Mismatch {
                    component: "L2 counters",
                    expected,
                    got,
                },
            ))
            .into(),
        );
    }
    Ok(counters)
}

#[cfg(feature = "reference")]
pub(crate) fn event_name(event: &Ev) -> &'static str {
    match event {
        Ev::SmWake => "sm_wake",
        Ev::L1PayloadAdmission { .. } => "l1_payload_admission",
        Ev::L2Access(_) => "l2_access",
        Ev::L2AccessBatch(_) => "l2_access_batch",
        Ev::L2Fill(_) => "l2_fill",
        Ev::AtomicCompletion => "atomic_completion",
        Ev::L2WritebackComplete { .. } => "l2_writeback_complete",
        Ev::L2CoherenceComplete { .. } => "l2_coherence_complete",
        Ev::L1VisibilityComplete { .. } => "l1_visibility_complete",
        Ev::L1VisibilityAccess(_) => "l1_visibility_access",
        Ev::L1VisibilityFence(_) => "l1_visibility_fence",
        Ev::L1FillVisible(_) => "l1_fill_visible",
        Ev::L1InvalidateAll(_) => "l1_invalidate_all",
        Ev::MemoryWriteAck(_) => "memory_write_ack",
        Ev::MemResponse { .. } => "memory_response",
    }
}

#[cfg(feature = "reference")]
pub(crate) fn typed_event_payload(event: &Ev) -> TypedEventPayload {
    let typed_response_kind = |reply: ReplyTo| match reply {
        ReplyTo::L1Fill {
            fill: FillKey::Line { .. },
            ..
        } => Some(TypedResponseKind::L1Fill),
        ReplyTo::L1Fill {
            fill: FillKey::Sector { .. },
            ..
        } => Some(TypedResponseKind::L1SectorFill),
        ReplyTo::Warp { .. } => Some(TypedResponseKind::Warp),
        ReplyTo::AsyncCopy { .. } => Some(TypedResponseKind::AsyncCopy),
        ReplyTo::None => None,
    };
    let typed_origin = |origin: MemoryOrigin| match origin {
        MemoryOrigin::Global => TypedMemoryOrigin::Global,
        MemoryOrigin::Local => TypedMemoryOrigin::Local,
    };
    let typed_source = |source: L2Source| match source {
        L2Source::L1Fill => TypedL2Source::L1Fill,
        L2Source::Bypass => TypedL2Source::Bypass,
        L2Source::Untagged => TypedL2Source::Untagged,
        L2Source::Store => TypedL2Source::Store,
    };
    let typed_operation = |operation: MemoryOperation| match operation {
        MemoryOperation::Read => TypedMemoryOperation::Read,
        MemoryOperation::Store => TypedMemoryOperation::Store,
        MemoryOperation::Atomic => TypedMemoryOperation::Atomic,
        MemoryOperation::Reduction => TypedMemoryOperation::Reduction,
    };
    let response = |reply: ReplyTo| {
        reply.response_id().map(|id| {
            let (warp, instruction, sector_index) = id.boundary_identity();
            TypedResponseIdentity {
                warp,
                instruction,
                sector_index,
                response_kind: typed_response_kind(reply)
                    .expect("response id always has a response kind"),
            }
        })
    };
    let request_payload = |request: &MemRequest| TypedMemoryRequest {
        issue_time: request.issue_time,
        sector_id: request.sector_id,
        origin: typed_origin(request.origin),
        sector_mask: request.sector_mask,
        local_sector_mask: request.local_sector_mask,
        source: typed_source(request.source),
        operation: typed_operation(request.operation),
        constant_load: request.constant_load,
        constant_deep: request.constant_deep,
        reply: response(request.reply),
        admission_witness: request
            .admission_witness
            .map(|witness| TypedAdmissionWitness {
                warp: witness.warp(),
                instruction: witness.instruction(),
                subcore: witness.subcore(),
                request_index: witness.request_index(),
                request_count: witness.request_count(),
            }),
    };
    match event {
        Ev::SmWake => TypedEventPayload::SmWake,
        Ev::L1PayloadAdmission {
            payloads,
            witnesses,
            identity_present,
            ..
        } => TypedEventPayload::L1PayloadAdmission {
            payload_records: u64::try_from(payloads.len()).unwrap_or(u64::MAX),
            witness_lines: u64::try_from(witnesses.len()).unwrap_or(u64::MAX),
            identity_present: *identity_present,
        },
        Ev::L2Access(request) => TypedEventPayload::L2Access {
            request: request_payload(request),
        },
        Ev::L2AccessBatch(requests) => TypedEventPayload::L2AccessBatch {
            requests: requests.iter().map(request_payload).collect(),
        },
        Ev::L2Fill(fill) => {
            let (line_id, generation, sector_mask) = fill.boundary_identity();
            TypedEventPayload::L2Fill {
                fill: TypedFillIdentity {
                    line_id,
                    generation,
                    sector_mask,
                },
            }
        }
        Ev::AtomicCompletion => TypedEventPayload::AtomicCompletion,
        Ev::L2WritebackComplete { plan } => TypedEventPayload::WritebackCompletion {
            token: plan.token(),
            line_id: plan.line_id().unwrap_or_default(),
            dirty_mask: plan.dirty_mask(),
            service_done_at: plan.service_done_cycle(),
        },
        Ev::L2CoherenceComplete { plan } => TypedEventPayload::CoherenceCompletion {
            token: plan.token(),
            line_id: plan.request_line_id(),
            sector_mask: plan.request_sector_mask(),
            service_sectors: plan.service_sectors(),
            service_done_at: plan.service_done_cycle(),
        },
        Ev::L1VisibilityComplete { plan } => TypedEventPayload::L1VisibilityCompletion {
            token: plan.token(),
            line_id: plan.line_id(),
            sector_mask: plan.sector_mask(),
            service_sectors: plan.service_sectors(),
            service_done_at: plan.service_done_cycle(),
        },
        Ev::L1VisibilityAccess(observation) => TypedEventPayload::L1VisibilityAccess {
            sequence: observation.sequence,
            requester: observation.requester,
            line_id: observation.line_id,
            sector_mask: observation.sector_mask,
            is_write: observation.is_write,
            l1_resident_after: observation.l1_resident_after,
            fill_will_reside: observation.fill_will_reside,
            fill_generation: observation.fill_generation,
            hit: observation.hit,
            issue_cycle: observation.issue_cycle,
        },
        Ev::L1VisibilityFence(observation) => TypedEventPayload::L1VisibilityFence {
            sequence: observation.sequence,
            requester: observation.requester,
            pc: observation.pc,
            issue_cycle: observation.issue_cycle,
            scope: format!("{:?}", observation.scope),
            order: format!("{:?}", observation.order),
        },
        Ev::L1FillVisible(observation) => TypedEventPayload::L1FillVisible {
            sequence: observation.sequence,
            requester: observation.requester,
            line_id: observation.line_id,
            sector_mask: observation.sector_mask,
            fill_generation: observation.fill_generation,
            visible_cycle: observation.visible_cycle,
        },
        Ev::L1InvalidateAll(observation) => TypedEventPayload::L1InvalidateAll {
            sequence: observation.sequence,
            requester: observation.requester,
            issue_cycle: observation.issue_cycle,
        },
        Ev::MemoryWriteAck(ack) => TypedEventPayload::MemoryWriteAck {
            warp: ack.witness.warp(),
            instruction: ack.witness.instruction(),
            subcore: ack.witness.subcore(),
            request_index: ack.witness.request_index(),
            request_count: ack.witness.request_count(),
            operation: typed_operation(ack.operation),
            issue_at: ack.issue_at,
            accepted_at: ack.accepted_at,
            service_done_at: ack.service_done_at,
            fence_visible_at: ack.fence_visible_at,
        },
        Ev::MemResponse { reply, fill } => TypedEventPayload::MemoryResponse {
            reply: response(*reply),
            fill: fill.map(|fill| {
                let (line_id, generation, sector_mask) = fill.boundary_identity();
                TypedFillIdentity {
                    line_id,
                    generation,
                    sector_mask,
                }
            }),
        },
    }
}
