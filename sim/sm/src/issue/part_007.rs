// Lexical implementation section 8/8; included by the parent facade.
fn validate_sector_span(
    kind: AccessKind,
    sectors: usize,
    pc: u32,
    launch_index: i64,
    warp_slot: i32,
) -> Result<(), IssueError> {
    let atomic_topology = match kind {
        AccessKind::GlobalBypass {
            len,
            token: CacheOpToken::Atom,
            l2_requests,
            ..
        }
        | AccessKind::GlobalReduction {
            len, l2_requests, ..
        } => Some((len, l2_requests, true)),
        AccessKind::GlobalBypass {
            len, l2_requests, ..
        } if l2_requests != 0 => Some((len, l2_requests, false)),
        _ => None,
    };
    if let Some((sector_transactions, l2_requests, is_atomic)) = atomic_topology
        && (!is_atomic
            || (sector_transactions == 0) != (l2_requests == 0)
            || l2_requests > sector_transactions)
    {
        return Err(IssueError::AtomicRequestTopologyInvalid {
            pc,
            launch_index,
            warp_slot,
            sectors: sector_transactions,
            l2_requests,
        });
    }
    let (off, len) = match kind {
        AccessKind::GlobalEligible { off, len, .. }
        | AccessKind::GlobalBypass { off, len, .. }
        | AccessKind::GlobalAsyncCopy { off, len, .. }
        | AccessKind::GlobalUntagged { off, len }
        | AccessKind::GlobalStore { off, len }
        | AccessKind::GlobalReduction { off, len, .. }
        | AccessKind::LocalLoad { off, len }
        | AccessKind::LocalStore { off, len } => (off, len),
        AccessKind::NonMemory | AccessKind::PredicatedOffMemory | AccessKind::Shared { .. } => {
            return Ok(());
        }
    };
    let end = off
        .checked_add(usize::from(len))
        .ok_or(IssueError::SectorSpanOverflow {
            pc,
            launch_index,
            warp_slot,
            off,
            len,
        })?;
    if end > sectors {
        return Err(IssueError::SectorSpanOutOfBounds {
            pc,
            launch_index,
            warp_slot,
            off,
            len,
            end,
            sectors,
        });
    }
    Ok(())
}

fn lookup_dense<'a>(
    schedule_slots: &[(u64, &'a IssueSlot)],
    schedule: &FunctionSchedule,
    pc: u32,
    launch_index: i64,
    warp_slot: i32,
) -> Result<&'a IssueSlot, IssueError> {
    schedule_slots
        .binary_search_by_key(&u64::from(pc), |(slot_pc, _)| *slot_pc)
        .map(|index| schedule_slots[index].1)
        .map_err(|_| IssueError::PcNotInSchedule {
            pc,
            launch_index,
            warp_slot,
            function_name: schedule.function_name.clone(),
        })
}

/// Replay one SM's blocks to completion under the v0 issue model
/// (synchronous Flat/L1Const modes; the shared-L2 mode runs through
/// [`SmCore`] + the event system). `l1_config`, when supplied, attaches
/// a FRESH per-SM L1 (cold cache per kernel, NCU `--cache-control all`
/// parity).
pub(crate) fn simulate_sm_with_residency(
    blocks: &[BlockReplay],
    schedule: &FunctionSchedule,
    params: &IssueModelParams,
    shared_memory_residency: SharedMemoryResidency,
    l1_config: Option<L1Config>,
) -> Result<SmSimResult, IssueError> {
    let mode = match l1_config {
        Some(cfg) => SmMemMode::L1Const(cfg),
        None => SmMemMode::Flat,
    };
    let mut core =
        SmCore::new_with_residency(blocks, schedule, params, shared_memory_residency, mode)?;
    let mut out = Vec::new();
    match core.advance(ReplayHorizon::Unbounded, &mut out)? {
        Advance::Done => {}
        Advance::Idle { .. } => unreachable!("synchronous modes emit no requests"),
    }
    debug_assert!(out.is_empty(), "synchronous modes emit no requests");
    Ok(core.result())
}

