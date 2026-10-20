// Lexical implementation section 4/4; included by the parent facade.
/// Resolve the two special Global routes implemented on the active sm_86 path.
/// The dynamic enum alone is not enough: the static sidecar must prove a
/// Global ATOM/RED family or LDGSTS role before the record enters replay.
fn require_sm86_special_role(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
    route: ReplayMemoryRoute,
    class: pb::MemoryOpClass,
) -> Result<(), ReplayError> {
    let alpha_sm120 = arch == ArchId::Sm120;
    if arch != ArchId::Sm86 && !alpha_sm120 {
        return Err(ReplayError::UnsupportedMemoryPath { arch, class });
    }
    let slot = schedule
        .by_pc
        .get(&u64::from(pc))
        .ok_or(ReplayError::UnsupportedMemoryPath { arch, class })?;
    let head = opcode_head(&slot.opcode_text);
    if alpha_sm120 && class == pb::MemoryOpClass::AsyncCopy && head == "LDGSTS" {
        // Alpha treats an observed async/TMA source as a flat global demand;
        // no SM86 LDGSTS completion or cache route is attached.
        return Ok(());
    }
    let valid = match (route, class) {
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Atomic) => match slot.mem_role {
            MemRole::BypassLoad(CacheOpToken::Atom) => {
                matches!(head, "ATOM" | "ATOMG") || (alpha_sm120 && head == "REDG")
            }
            MemRole::Reduction => matches!(head, "RED" | "REDG"),
            _ => false,
        },
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::AsyncCopy) => {
            matches!(slot.mem_role, MemRole::BypassLoad(CacheOpToken::Ldgsts)) && head == "LDGSTS"
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(ReplayError::UnsupportedMemoryPath { arch, class })
    }
}

fn require_ldgsts_l1_policy(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<LdgstsL1Policy, ReplayError> {
    schedule
        .by_pc
        .get(&u64::from(pc))
        .and_then(|slot| slot.ldgsts_l1_policy)
        .ok_or(ReplayError::UnsupportedMemoryPath {
            arch,
            class: pb::MemoryOpClass::AsyncCopy,
        })
}

/// Return the schedule's memory role while rejecting a dynamic global record
/// whose static sidecar opcode is neither classified nor one of the deliberate
/// untagged load-family heads. A pc absent from the schedule stays neutral here
/// so the issue loop reports its more specific `PcNotInSchedule` error.
fn require_known_global_role(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<MemRole, ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(MemRole::None);
    };
    if slot.mem_role == MemRole::None && !is_known_untagged_opcode(&slot.opcode_text) {
        // The SM120 alpha path admits newly observed Blackwell storage-family
        // spellings as untagged global traffic.  They retain their dynamic
        // address/sector stream and flat latency, but do not borrow a guessed
        // cache operator or writeback class.  Ordinary SM120 replay remains
        // fail-closed because this branch is keyed by the alpha-tagged
        // deployment schedule.
        if arch != ArchId::Sm86 {
            return Ok(MemRole::None);
        }
        return Err(ReplayError::UnknownGlobalOpcode {
            arch,
            pc,
            opcode: opcode_head(&slot.opcode_text).to_owned(),
        });
    }
    Ok(slot.mem_role)
}

fn is_static_constant_global_load(opcode_text: &str) -> bool {
    let opcode = opcode_text.split_ascii_whitespace().next().unwrap_or("");
    opcode.split('.').any(|modifier| modifier == "CONSTANT")
        && matches!(opcode_head(opcode_text), "LDG" | "LD" | "LDC" | "LDL")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalDirection {
    Load,
    Store,
}

/// Resolve Local traffic only from the joined LDL/STL sidecar role. A missing pc
/// remains neutral so the issue loop retains its dedicated `PcNotInSchedule`
/// diagnostic; every present non-LDL/STL role fails closed.
fn require_known_local_direction(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<Option<LocalDirection>, ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(None);
    };
    match (opcode_head(&slot.opcode_text), slot.mem_role) {
        ("LDL", MemRole::EligibleLoad(CacheOpToken::Ldl)) => Ok(Some(LocalDirection::Load)),
        ("STL", MemRole::EligibleLoad(CacheOpToken::Stl)) => Ok(Some(LocalDirection::Store)),
        (opcode, _) => Err(ReplayError::UnknownLocalOpcode {
            arch,
            pc,
            opcode: opcode.to_owned(),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplayMemoryRoute {
    Global,
    Local,
    Shared,
    Unsupported,
}

impl ReplayMemoryRoute {
    const fn canonical_space(self) -> pb::MemorySpace {
        match self {
            Self::Global => pb::MemorySpace::Global,
            Self::Local => pb::MemorySpace::Local,
            Self::Shared => pb::MemorySpace::Shared,
            Self::Unsupported => pb::MemorySpace::Unspecified,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StaticMemoryRoute {
    Modeled {
        route: ReplayMemoryRoute,
        class: pb::MemoryOpClass,
    },
    /// Unqualified ATOM/RED use PTX generic addressing. Their dynamic address
    /// operand can resolve to either global or shared memory, so the static
    /// mnemonic supplies the operation class but not one fixed route.
    GenericAtomic {
        class: pb::MemoryOpClass,
    },
    UnsupportedClass(pb::MemoryOpClass),
}

fn opcode_has_qualifier(opcode_text: &str, qualifier: &str) -> bool {
    opcode_text
        .split_whitespace()
        .next()
        .unwrap_or("")
        .split('.')
        .skip(1)
        .any(|part| part == qualifier)
}

/// Return whether a Blackwell bulk-copy spelling is the modeled
/// global-to-shared producer form.  The qualifier order is part of the
/// instruction contract: `UBLKCP.S.G` names a shared destination and global
/// source.  The legacy `UTMASTG` store form remains a supported effective
/// store-side trace path; it is not confused with the load-side direction
/// check below.
fn is_modeled_tma_load_opcode(opcode_text: &str) -> bool {
    let token = opcode_text.split_whitespace().next().unwrap_or("");
    let mut fields = token.split('.');
    match fields.next().unwrap_or("") {
        "UTMALDG" | "UTMASTG" => true,
        "UBLKCP" => matches!(fields.next(), Some("S")) && matches!(fields.next(), Some("G")),
        _ => false,
    }
}

fn is_tma_submission_opcode(opcode_text: &str) -> bool {
    matches!(opcode_head(opcode_text), "UTMALDG" | "UTMASTG" | "UBLKCP")
}

fn validate_tma_load_direction(
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
) -> Result<(), ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(());
    };
    if is_tma_submission_opcode(&slot.opcode_text) && !is_modeled_tma_load_opcode(&slot.opcode_text)
    {
        return Err(ReplayError::UnsupportedTmaDirection {
            arch,
            pc,
            opcode: slot.opcode_text.clone(),
        });
    }
    Ok(())
}

/// Space-coded SASS heads whose storage class is part of the mnemonic rather
/// than inferred from the dynamic record. ATOM and RED are generic, while
/// ATOMG/ATOMS constrain the route; LDGSTS uses its traced Global source. REDG,
/// LDGMC, DSMEM, and shared-matrix families are architecture-gated before any
/// record can borrow a modeled route.
fn static_memory_route(
    opcode_text: &str,
    arch: ArchId,
) -> Result<Option<StaticMemoryRoute>, &'static str> {
    let head = opcode_head(opcode_text);
    if matches!(head, "LDS" | "STS" | "LDSM" | "STSM") && opcode_has_qualifier(opcode_text, "MMIO")
    {
        let family = match head {
            "LDS" => "LDS.MMIO",
            "STS" => "STS.MMIO",
            "LDSM" => "LDSM.MMIO",
            "STSM" => "STSM.MMIO",
            _ => unreachable!("match guard admits only shared-memory heads"),
        };
        return match arch {
            ArchId::Sm86 => Err(family),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::UnsupportedClass(
                pb::MemoryOpClass::ClusterDsmem,
            ))),
        };
    }
    match head {
        "LDG" | "STG" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Global,
            class: pb::MemoryOpClass::Global,
        })),
        "LDGSTS" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Global,
            class: pb::MemoryOpClass::AsyncCopy,
        })),
        "ATOM" | "RED" => Ok(Some(StaticMemoryRoute::GenericAtomic {
            class: pb::MemoryOpClass::Atomic,
        })),
        "ATOMG" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Global,
            class: pb::MemoryOpClass::Atomic,
        })),
        "REDG" => match arch {
            ArchId::Sm86 => Err("REDG"),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::Modeled {
                route: ReplayMemoryRoute::Global,
                class: pb::MemoryOpClass::Atomic,
            })),
        },
        // Hopper's reducing-load family has no implemented replay semantics on
        // either target. In particular it must never borrow Shared timing when
        // contaminated dynamic metadata claims a Shared route.
        "LDGMC" => Err("LDGMC"),
        "LDS" | "STS" | "LDSM" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Shared,
            class: pb::MemoryOpClass::Shared,
        })),
        "ATOMS" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Shared,
            class: pb::MemoryOpClass::Atomic,
        })),
        "STSM" => match arch {
            ArchId::Sm86 => Err("STSM"),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::Modeled {
                route: ReplayMemoryRoute::Shared,
                class: pb::MemoryOpClass::Shared,
            })),
        },
        "STAS" | "REDAS" => match arch {
            ArchId::Sm86 => Err(if head == "STAS" { "STAS" } else { "REDAS" }),
            ArchId::Sm120 => Ok(Some(StaticMemoryRoute::UnsupportedClass(
                pb::MemoryOpClass::ClusterDsmem,
            ))),
        },
        "LDL" | "STL" => Ok(Some(StaticMemoryRoute::Modeled {
            route: ReplayMemoryRoute::Local,
            class: pb::MemoryOpClass::Local,
        })),
        "REDS" => Err("REDS"),
        _ if head.starts_with("ATOM") => Err("ATOM*"),
        _ if head.starts_with("RED") => Err("RED*"),
        _ => Ok(None),
    }
}

/// Reject disagreement between a known space-coded static SASS family and the
/// class/route resolved from the dynamic contract. This runs before allocation
/// identity and classification, so a contradictory record cannot borrow another
/// operation or storage class's timing and traffic semantics.
fn validate_static_memory_route(
    meta: MemoryAccessMeta,
    route: ReplayMemoryRoute,
    schedule: &FunctionSchedule,
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<(), ReplayError> {
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(());
    };
    let contextualize = |source| ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class: meta.op_class,
        source,
    };
    validate_tma_load_direction(schedule, schedule.arch, pc)?;
    let local_role = matches!(
        slot.mem_role,
        MemRole::EligibleLoad(CacheOpToken::Ldl | CacheOpToken::Stl)
    );
    if local_role && route != ReplayMemoryRoute::Local {
        return Err(contextualize(MemoryRouteError::StaticMemoryRouteMismatch {
            expected: pb::MemorySpace::Local,
            got: route.canonical_space(),
        }));
    }
    let expected = static_memory_route(&slot.opcode_text, schedule.arch).map_err(|family| {
        contextualize(MemoryRouteError::StaticMemoryFamilyUnsupported {
            family,
            arch: schedule.arch,
        })
    })?;
    match expected {
        Some(StaticMemoryRoute::Modeled {
            route: expected, ..
        }) if expected != route => {
            Err(contextualize(MemoryRouteError::StaticMemoryRouteMismatch {
                expected: expected.canonical_space(),
                got: route.canonical_space(),
            }))
        }
        Some(StaticMemoryRoute::Modeled {
            class: expected, ..
        }) if expected != meta.op_class => {
            Err(contextualize(MemoryRouteError::StaticMemoryClassMismatch {
                expected,
                got: meta.op_class,
            }))
        }
        Some(StaticMemoryRoute::GenericAtomic { class: expected }) => {
            if expected != meta.op_class {
                return Err(contextualize(MemoryRouteError::StaticMemoryClassMismatch {
                    expected,
                    got: meta.op_class,
                }));
            }
            if matches!(route, ReplayMemoryRoute::Global | ReplayMemoryRoute::Shared) {
                Ok(())
            } else {
                Err(contextualize(
                    MemoryRouteError::StaticGenericAtomicRouteInvalid {
                        got: route.canonical_space(),
                    },
                ))
            }
        }
        Some(StaticMemoryRoute::UnsupportedClass(expected)) if expected != meta.op_class => {
            Err(contextualize(MemoryRouteError::StaticMemoryClassMismatch {
                expected,
                got: meta.op_class,
            }))
        }
        _ => Ok(()),
    }
}

fn validate_replay_access_kind(
    meta: MemoryAccessMeta,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<(), ReplayError> {
    let Some(got) = meta.access_kind else {
        return Ok(()); // additive legacy field: absence remains admissible
    };
    let (valid, expected) = match meta.op_class {
        pb::MemoryOpClass::Global | pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => (
            matches!(
                got,
                pb::MemoryOpAccessKind::Load | pb::MemoryOpAccessKind::Store
            ),
            "LOAD or STORE",
        ),
        pb::MemoryOpClass::Atomic => (got == pb::MemoryOpAccessKind::Atomic, "ATOMIC"),
        pb::MemoryOpClass::AsyncCopy => (got == pb::MemoryOpAccessKind::AsyncCopy, "ASYNC_COPY"),
        // These routes fail as unsupported before replay. Do not invent a
        // direction contract for a class with no modeled path.
        pb::MemoryOpClass::Unspecified
        | pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => return Ok(()),
    };
    if valid {
        return Ok(());
    }
    Err(ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class: meta.op_class,
        source: MemoryRouteError::AccessKindInvalid { got, expected },
    })
}

fn validate_static_access_kind(
    meta: MemoryAccessMeta,
    schedule: &FunctionSchedule,
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<(), ReplayError> {
    let Some(got) = meta.access_kind else {
        return Ok(()); // additive legacy field: no dynamic direction to compare
    };
    // Keep a missing pc neutral so the issue loop preserves its specific
    // PcNotInSchedule diagnostic instead of manufacturing a direction claim.
    let Some(slot) = schedule.by_pc.get(&u64::from(pc)) else {
        return Ok(());
    };
    let head = opcode_head(&slot.opcode_text);
    let expected = if head.starts_with("ATOM") || head.starts_with("RED") {
        Some(pb::MemoryOpAccessKind::Atomic)
    } else {
        match head {
            "LDGSTS" => Some(pb::MemoryOpAccessKind::AsyncCopy),
            "STG" | "STL" | "STS" | "STSM" | "STAS" => Some(pb::MemoryOpAccessKind::Store),
            _ => match slot.mem_role {
                MemRole::EligibleLoad(CacheOpToken::Stl) => Some(pb::MemoryOpAccessKind::Store),
                MemRole::EligibleLoad(_) => Some(pb::MemoryOpAccessKind::Load),
                MemRole::BypassLoad(CacheOpToken::Atom) => Some(pb::MemoryOpAccessKind::Atomic),
                MemRole::BypassLoad(CacheOpToken::Ldgsts) => {
                    Some(pb::MemoryOpAccessKind::AsyncCopy)
                }
                MemRole::BypassLoad(_) => Some(pb::MemoryOpAccessKind::Load),
                MemRole::Store => Some(pb::MemoryOpAccessKind::Store),
                MemRole::Reduction => Some(pb::MemoryOpAccessKind::Atomic),
                MemRole::None if is_known_untagged_opcode(&slot.opcode_text) => {
                    Some(pb::MemoryOpAccessKind::Load)
                }
                MemRole::None => None,
            },
        }
    };
    let Some(expected) = expected else {
        return Ok(()); // the existing unknown-opcode gate owns this diagnostic
    };
    if got == expected {
        return Ok(());
    }
    Err(ReplayError::MemoryRoute {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        class: meta.op_class,
        source: MemoryRouteError::StaticAccessKindMismatch { got, expected },
    })
}

/// Flat mode validates the same static operation role as hierarchy replay.  The
/// legacy path deliberately carries an empty sector span, but the SM120 alpha
/// path may retain the already validated NVBit span so its logical global
/// service tail sees the captured request width even when a launch also
/// contains an opaque async/TMA record.
fn flat_global_kind(role: MemRole) -> AccessKind {
    match role {
        MemRole::EligibleLoad(token) => AccessKind::GlobalEligible {
            off: 0,
            len: 0,
            token,
        },
        MemRole::BypassLoad(token) => AccessKind::GlobalBypass {
            off: 0,
            len: 0,
            token,
            l2_requests: 0,
        },
        MemRole::Store => AccessKind::GlobalStore { off: 0, len: 0 },
        MemRole::Reduction => AccessKind::GlobalReduction {
            off: 0,
            len: 0,
            l2_requests: 0,
        },
        MemRole::None => AccessKind::GlobalUntagged { off: 0, len: 0 },
    }
}

fn classify_flat_global_sectors(
    role: MemRole,
    warp: &ValidatedWarp,
    memory: &ValidatedMemoryInstruction,
    sectors: &mut Vec<i64>,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let sector_scratch = match warp.sector_plan(memory) {
        ValidatedSectorPlan::Sectors(sectors) => sectors,
        ValidatedSectorPlan::Error { error, .. } => {
            return Err(deferred_sector_error(error, path, warp_id, record_ix));
        }
    };
    append_staged_global_sectors(role, sectors, sector_scratch, path, warp_id, record_ix)
}

#[derive(Clone, Copy)]
struct CompactReplayMemory<'a> {
    memory: &'a ValidatedMemoryInstruction,
    route: ReplayMemoryRoute,
    pc: u32,
}

fn compact_shared_kind(memory: &ValidatedMemoryInstruction, arch: ArchId) -> AccessKind {
    let data_wavefronts = if arch == ArchId::Sm86 {
        memory.shared_sm86_wavefronts_32x4()
    } else {
        memory.shared_bank_wavefronts_32x4()
    };
    let data_ideal_wavefronts = memory.shared_bank_ideal_wavefronts_32x4();
    AccessKind::Shared {
        // Keep scheduler timing on the established whole-warp projection.
        wavefronts: u64::from(memory.shared_bank_legacy_wavefronts_32x4()),
        ideal_wavefronts: u64::from(memory.shared_bank_legacy_ideal_wavefronts_32x4()),
        // SM86 retains the validated legacy observer; SM120 uses the newer
        // phase-local effective observer. Neither changes issue timing.
        data_wavefronts: u64::from(data_wavefronts),
        data_ideal_wavefronts: u64::from(data_ideal_wavefronts),
    }
}

fn compact_memory_route_error(error: &ValidatedMemoryRouteError) -> MemoryRouteError {
    match *error {
        ValidatedMemoryRouteError::GlobalOperandCount { got } => {
            MemoryRouteError::GlobalOperandCount { got }
        }
        ValidatedMemoryRouteError::GlobalSpaceInvalid { operand, space } => {
            MemoryRouteError::GlobalSpaceInvalid { operand, space }
        }
        ValidatedMemoryRouteError::SharedLocalOperandCount { got } => {
            MemoryRouteError::SharedLocalOperandCount { got }
        }
        ValidatedMemoryRouteError::SharedLocalSpaceInvalid { operand, space } => {
            MemoryRouteError::SharedLocalSpaceInvalid { operand, space }
        }
        ValidatedMemoryRouteError::AtomicOperandCount { got } => {
            MemoryRouteError::AtomicOperandCount { got }
        }
        ValidatedMemoryRouteError::AtomicSpaceMissing { operand } => {
            MemoryRouteError::AtomicSpaceMissing { operand }
        }
        ValidatedMemoryRouteError::AtomicSpaceUnspecified { operand } => {
            MemoryRouteError::AtomicSpaceUnspecified { operand }
        }
        ValidatedMemoryRouteError::AtomicSpacesConflict => MemoryRouteError::AtomicSpacesConflict,
        ValidatedMemoryRouteError::AsyncCopySourceCount { got } => {
            MemoryRouteError::AsyncCopySourceCount { got }
        }
        ValidatedMemoryRouteError::AsyncCopySourceSpaceMissing => {
            MemoryRouteError::AsyncCopySourceSpaceMissing
        }
        ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid { space } => {
            MemoryRouteError::AsyncCopySourceSpaceInvalid { space }
        }
        ValidatedMemoryRouteError::AsyncCopyOperandCount { got, expected } => {
            MemoryRouteError::AsyncCopyOperandCount { got, expected }
        }
        ValidatedMemoryRouteError::AsyncCopyDestinationSpaceMissing => {
            MemoryRouteError::AsyncCopyDestinationSpaceMissing
        }
        ValidatedMemoryRouteError::AsyncCopyDestinationSpaceInvalid { space } => {
            MemoryRouteError::AsyncCopyDestinationSpaceInvalid { space }
        }
    }
}

fn preflight_compact_memory<'a>(
    memory: Option<&'a ValidatedMemoryInstruction>,
    warp: &'a ValidatedWarp,
    schedule: &FunctionSchedule,
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<Option<CompactReplayMemory<'a>>, ReplayError> {
    let Some(memory) = memory else {
        return Ok(None);
    };
    let meta = memory.replay_meta();
    let route = match warp.memory_route(memory) {
        Ok(ValidatedMemoryRoute::Global) => ReplayMemoryRoute::Global,
        Ok(ValidatedMemoryRoute::Local) => ReplayMemoryRoute::Local,
        Ok(ValidatedMemoryRoute::Shared) => ReplayMemoryRoute::Shared,
        Ok(ValidatedMemoryRoute::Unsupported) => ReplayMemoryRoute::Unsupported,
        Err(error) => {
            return Err(ReplayError::MemoryRoute {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                class: meta.op_class,
                source: compact_memory_route_error(error),
            });
        }
    };
    validate_replay_access_kind(meta, path, warp_id, record_ix)?;
    validate_static_memory_route(meta, route, schedule, pc, path, warp_id, record_ix)?;
    validate_static_access_kind(meta, schedule, pc, path, warp_id, record_ix)?;
    Ok(Some(CompactReplayMemory { memory, route, pc }))
}

fn classify_compact_flat(
    preflight: Option<CompactReplayMemory<'_>>,
    warp: &ValidatedWarp,
    schedule: &FunctionSchedule,
    arch: ArchId,
    pc: u32,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
    sector_arena: &mut Vec<i64>,
    block_id: [u32; 3],
    tma_observations: Option<&BTreeMap<TmaObservationKey, TmaObservation>>,
) -> Result<AccessKind, ReplayError> {
    let Some(preflight) = preflight else {
        if arch == ArchId::Sm120
            && schedule.issue_slot(u64::from(pc)).is_some_and(|slot| {
                let head = opcode_head(&slot.opcode_text);
                matches!(head, "UTMALDG" | "UTMASTG" | "UBLKCP")
            })
        {
            validate_tma_load_direction(schedule, arch, pc)?;
            // TMA operands are carried by tensor-map/mbarrier sidecars rather
            // than ordinary dynamic address operands in the compact trace.
            // Consume the matching per-instruction payload observation when
            // present. A missing sidecar is explicitly unavailable and carries
            // zero logical completion work; it no longer fabricates 8 KiB.
            let key = TmaObservationKey {
                cta: block_id,
                warp: warp_id,
                instruction: record_ix,
                pc,
                function: warp.function_unique_id(),
            };
            let payload_bytes = tma_observations
                .and_then(|observations| observations.get(&key))
                .and_then(|observation| observation.bytes)
                .unwrap_or(0);
            let payload_work = u32::try_from(payload_bytes).map_err(|_| {
                ReplayError::AsyncCopyCompletionWorkOutOfRange {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    completion_work: payload_bytes,
                    max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                }
            })?;
            let completion_work = encode_async_copy_payload_work(payload_work).ok_or(
                ReplayError::AsyncCopyCompletionWorkOutOfRange {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    completion_work: payload_bytes,
                    max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                },
            )?;
            return Ok(AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 0,
                completion_work,
                // TMA tensor-map transfers use the distinct logical bypass
                // service arm in full replay. This is a calibration-domain
                // route, not a physical cache policy claim.
                l1_policy: LdgstsL1Policy::BypassL1,
            });
        }
        return Ok(AccessKind::NonMemory);
    };
    if !preflight.memory.has_effective_lanes() {
        return Ok(AccessKind::PredicatedOffMemory);
    }
    let meta = preflight.memory.replay_meta();
    match (preflight.route, meta.op_class) {
        (ReplayMemoryRoute::Shared, _) => Ok(compact_shared_kind(preflight.memory, arch)),
        (ReplayMemoryRoute::Local, pb::MemoryOpClass::Local) => {
            let Some(direction) = require_known_local_direction(schedule, arch, preflight.pc)?
            else {
                return Ok(AccessKind::NonMemory);
            };
            if arch == ArchId::Sm120 {
                let source_sectors = match warp.sector_plan(preflight.memory) {
                    ValidatedSectorPlan::Sectors(sectors) => sectors,
                    ValidatedSectorPlan::Error { error, .. } => {
                        return Err(deferred_sector_error(error, path, warp_id, record_ix));
                    }
                };
                append_staged_local_sectors(
                    direction,
                    sector_arena,
                    source_sectors,
                    path,
                    warp_id,
                    record_ix,
                )
            } else {
                Ok(match direction {
                    LocalDirection::Load => AccessKind::LocalLoad { off: 0, len: 0 },
                    LocalDirection::Store => AccessKind::LocalStore { off: 0, len: 0 },
                })
            }
        }
        (ReplayMemoryRoute::Local, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Unsupported, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Global) => {
            let role = require_known_global_role(schedule, arch, preflight.pc)?;
            if arch == ArchId::Sm120 {
                classify_flat_global_sectors(
                    role,
                    warp,
                    preflight.memory,
                    sector_arena,
                    path,
                    warp_id,
                    record_ix,
                )
            } else {
                Ok(flat_global_kind(role))
            }
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Atomic) => {
            require_sm86_special_role(
                schedule,
                arch,
                preflight.pc,
                preflight.route,
                pb::MemoryOpClass::Atomic,
            )?;
            let role = require_known_global_role(schedule, arch, preflight.pc)?;
            if arch == ArchId::Sm120 {
                classify_flat_global_sectors(
                    role,
                    warp,
                    preflight.memory,
                    sector_arena,
                    path,
                    warp_id,
                    record_ix,
                )
            } else {
                Ok(flat_global_kind(role))
            }
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::AsyncCopy) => {
            if arch == ArchId::Sm120 {
                // Full-replay-only alpha route: preserve an observed async
                // source as a typed completion token instead of collapsing it
                // into an ordinary global instruction.  A compact trace may
                // lack payload/sector evidence; that is zero logical work with
                // an inactive token, not an invented transaction.  This is not
                // a native TMA engine timestamp or queue model.
                let source_sectors = match warp.sector_plan(preflight.memory) {
                    ValidatedSectorPlan::Sectors(sectors) => sectors,
                    ValidatedSectorPlan::Error { error, .. } => {
                        return Err(deferred_sector_error(error, path, warp_id, record_ix));
                    }
                };
                let payload_bytes = warp.effective_data_bytes(preflight.memory);
                let sector_bytes = u64::try_from(source_sectors.len())
                    .ok()
                    .and_then(|count| count.checked_mul(32))
                    .unwrap_or(0);
                let logical_bytes = payload_bytes.max(sector_bytes);
                let completion_work = u32::try_from(logical_bytes)
                    .ok()
                    .and_then(encode_async_copy_payload_work)
                    .ok_or(ReplayError::AsyncCopyCompletionWorkOutOfRange {
                        path: path.to_path_buf(),
                        warp_id,
                        record_ix,
                        completion_work: logical_bytes,
                        max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                    })?;
                let kind = checked_staged_async_copy_kind(
                    LdgstsL1Policy::CacheAtAllLevels,
                    sector_arena.len(),
                    source_sectors.len(),
                    completion_work,
                )
                .ok_or(ReplayError::SectorCountOverflow {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    sectors: source_sectors.len(),
                })?;
                return append_staged_sectors(
                    Some(kind),
                    sector_arena,
                    source_sectors,
                    path,
                    warp_id,
                    record_ix,
                );
            }
            require_sm86_special_role(
                schedule,
                arch,
                preflight.pc,
                preflight.route,
                pb::MemoryOpClass::AsyncCopy,
            )?;
            require_known_global_role(schedule, arch, preflight.pc)?;
            let l1_policy = require_ldgsts_l1_policy(schedule, arch, preflight.pc)?;
            let source_active = match warp.sector_plan(preflight.memory) {
                ValidatedSectorPlan::Sectors(sectors) => !sectors.is_empty(),
                ValidatedSectorPlan::Error { error, .. } => {
                    return Err(deferred_sector_error(error, path, warp_id, record_ix));
                }
            };
            let payload_bytes = warp.effective_data_bytes(preflight.memory);
            if source_active != (payload_bytes != 0) {
                return Err(ReplayError::AsyncCopyPayloadMismatch {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    source_active,
                    sectors: usize::from(source_active),
                    payload_bytes,
                });
            }
            let completion_work = project_async_copy_completion_work(
                arch,
                l1_policy,
                payload_bytes,
                warp.ldgsts_destination_topology_128b_32x4(preflight.memory),
                path,
                warp_id,
                record_ix,
            )?;
            Ok(AccessKind::GlobalAsyncCopy {
                off: 0,
                len: 0,
                completion_work,
                l1_policy,
            })
        }
        (ReplayMemoryRoute::Global, class) => {
            Err(ReplayError::UnsupportedMemoryPath { arch, class })
        }
    }
}

fn deferred_sector_error(
    error: &ValidatedSectorError,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> ReplayError {
    if let ValidatedSectorError::SectorCountOverflow { sectors } = *error {
        return ReplayError::SectorCountOverflow {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            sectors,
        };
    }
    let source = match *error {
        ValidatedSectorError::DataWidthInvalid { data_width_bytes } => {
            SectorExpansionError::DataWidthInvalid { data_width_bytes }
        }
        ValidatedSectorError::AddressOverflow {
            lane,
            base,
            data_width_bytes,
        } => SectorExpansionError::AddressOverflow {
            lane,
            base,
            data_width_bytes,
        },
        ValidatedSectorError::UniqueSectorCountOutOfRange { sectors } => {
            SectorExpansionError::UniqueSectorCountOutOfRange { sectors }
        }
        ValidatedSectorError::CardinalityMismatch { expected, produced } => {
            SectorExpansionError::CardinalityMismatch { expected, produced }
        }
        ValidatedSectorError::CapacityUnavailable { buffer, additional } => {
            SectorExpansionError::CapacityUnavailable { buffer, additional }
        }
        ValidatedSectorError::SectorCountOverflow { .. } => unreachable!("handled above"),
    };
    ReplayError::SectorExpansion {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        source,
    }
}

fn classify_compact_with_l1(
    preflight: Option<CompactReplayMemory<'_>>,
    warp: &ValidatedWarp,
    schedule: &FunctionSchedule,
    arch: ArchId,
    sectors: &mut Vec<i64>,
    async_copy_cohorts: &mut AsyncCopyCohortReplay,
    block_id: [u32; 3],
    tma_observations: Option<&BTreeMap<TmaObservationKey, TmaObservation>>,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let Some(preflight) = preflight else {
        return Ok(AccessKind::NonMemory);
    };
    if !preflight.memory.has_effective_lanes() {
        return Ok(AccessKind::PredicatedOffMemory);
    }
    let meta = preflight.memory.replay_meta();
    let sector_plan = warp.sector_plan(preflight.memory);
    if let ValidatedSectorPlan::Error {
        phase: ValidatedSectorErrorPhase::BeforeStaticRole,
        error,
    } = sector_plan
    {
        return Err(deferred_sector_error(error, path, warp_id, record_ix));
    }

    enum StagedHierarchyRole {
        Global(MemRole),
        Local(LocalDirection),
    }
    let staged_role = match (preflight.route, meta.op_class) {
        (ReplayMemoryRoute::Shared, _) => {
            return Ok(compact_shared_kind(preflight.memory, arch));
        }
        (ReplayMemoryRoute::Unsupported, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
        (ReplayMemoryRoute::Local, pb::MemoryOpClass::Local) => {
            let Some(direction) = require_known_local_direction(schedule, arch, preflight.pc)?
            else {
                return Ok(AccessKind::NonMemory);
            };
            StagedHierarchyRole::Local(direction)
        }
        (ReplayMemoryRoute::Local, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::Global) => {
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, preflight.pc)?)
        }
        (ReplayMemoryRoute::Global, class @ pb::MemoryOpClass::Atomic) => {
            require_sm86_special_role(schedule, arch, preflight.pc, preflight.route, class)?;
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, preflight.pc)?)
        }
        (ReplayMemoryRoute::Global, pb::MemoryOpClass::AsyncCopy) if arch == ArchId::Sm120 => {
            // SM120 async source/destination semantics are admitted only into
            // the explicit alpha cache scenario.  The route remains typed as
            // an async token; it is never downgraded to ordinary global timing.
            StagedHierarchyRole::Global(MemRole::BypassLoad(CacheOpToken::Ldgsts))
        }
        (ReplayMemoryRoute::Global, class @ pb::MemoryOpClass::AsyncCopy) => {
            require_sm86_special_role(schedule, arch, preflight.pc, preflight.route, class)?;
            StagedHierarchyRole::Global(require_known_global_role(schedule, arch, preflight.pc)?)
        }
        (ReplayMemoryRoute::Global, class) => {
            return Err(ReplayError::UnsupportedMemoryPath { arch, class });
        }
    };
    let staged = match sector_plan {
        ValidatedSectorPlan::Sectors(staged) => staged,
        ValidatedSectorPlan::Error {
            phase: ValidatedSectorErrorPhase::AfterStaticRole,
            error,
        } => return Err(deferred_sector_error(error, path, warp_id, record_ix)),
        ValidatedSectorPlan::Error {
            phase: ValidatedSectorErrorPhase::BeforeStaticRole,
            ..
        } => unreachable!("handled above"),
    };
    match staged_role {
        StagedHierarchyRole::Global(_role) if meta.op_class == pb::MemoryOpClass::AsyncCopy => {
            let tma = arch == ArchId::Sm120
                && schedule
                    .issue_slot(u64::from(preflight.pc))
                    .is_some_and(|slot| {
                        matches!(
                            opcode_head(slot.opcode_text()),
                            "UTMALDG" | "UTMASTG" | "UBLKCP"
                        )
                    });
            let l1_policy = if tma {
                LdgstsL1Policy::BypassL1
            } else if arch == ArchId::Sm120 {
                let opcode = schedule
                    .issue_slot(u64::from(preflight.pc))
                    .map(|slot| slot.opcode_text())
                    .unwrap_or("");
                if opcode_has_qualifier(opcode, "BYPASS") {
                    LdgstsL1Policy::BypassL1
                } else {
                    LdgstsL1Policy::CacheAtAllLevels
                }
            } else {
                require_ldgsts_l1_policy(schedule, arch, preflight.pc)?
            };
            let completion_work_override = if arch == ArchId::Sm120 {
                let key = TmaObservationKey {
                    cta: block_id,
                    warp: warp_id,
                    instruction: record_ix,
                    pc: preflight.pc,
                    function: warp.function_unique_id(),
                };
                let payload_bytes = if tma {
                    tma_observations
                        .and_then(|observations| observations.get(&key))
                        .and_then(|observation| observation.bytes)
                        .ok_or(ReplayError::AsyncCopyPayloadMismatch {
                            path: path.to_path_buf(),
                            warp_id,
                            record_ix,
                            source_active: true,
                            sectors: staged.len(),
                            payload_bytes: 0,
                        })?
                } else {
                    warp.effective_data_bytes(preflight.memory)
                        .max(u64::try_from(staged.len()).unwrap_or(0).saturating_mul(32))
                };
                Some(
                    encode_async_copy_payload_work(u32::try_from(payload_bytes).map_err(|_| {
                        ReplayError::AsyncCopyCompletionWorkOutOfRange {
                            path: path.to_path_buf(),
                            warp_id,
                            record_ix,
                            completion_work: payload_bytes,
                            max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                        }
                    })?)
                    .ok_or(
                        ReplayError::AsyncCopyCompletionWorkOutOfRange {
                            path: path.to_path_buf(),
                            warp_id,
                            record_ix,
                            completion_work: payload_bytes,
                            max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
                        },
                    )?,
                )
            } else {
                None
            };
            append_staged_async_copy_sectors(
                arch,
                l1_policy,
                if tma {
                    0
                } else {
                    warp.effective_data_bytes(preflight.memory)
                        .max(u64::try_from(staged.len()).unwrap_or(0).saturating_mul(32))
                },
                warp.ldgsts_destination_topology_128b_32x4(preflight.memory),
                warp.ldgsts_source_cohorts_128b_32x4(preflight.memory),
                warp.ldgsts_source_sector_cohort_masks_128b_32x4(preflight.memory),
                sectors,
                async_copy_cohorts,
                staged,
                path,
                warp_id,
                record_ix,
                completion_work_override,
            )
        }
        StagedHierarchyRole::Global(role) => {
            append_staged_global_sectors(role, sectors, staged, path, warp_id, record_ix)
        }
        StagedHierarchyRole::Local(direction) => {
            append_staged_local_sectors(direction, sectors, staged, path, warp_id, record_ix)
        }
    }
}

fn checked_staged_global_kind(
    role: MemRole,
    start: usize,
    count: usize,
    l2_requests: u16,
) -> Option<AccessKind> {
    let len = u16::try_from(count).ok()?;
    start.checked_add(count)?;
    Some(match role {
        MemRole::EligibleLoad(token) => AccessKind::GlobalEligible {
            off: start,
            len,
            token,
        },
        MemRole::BypassLoad(token) => AccessKind::GlobalBypass {
            off: start,
            len,
            token,
            l2_requests: if token == CacheOpToken::Atom {
                l2_requests
            } else {
                0
            },
        },
        MemRole::Store => AccessKind::GlobalStore { off: start, len },
        MemRole::Reduction => AccessKind::GlobalReduction {
            off: start,
            len,
            l2_requests,
        },
        MemRole::None => AccessKind::GlobalUntagged { off: start, len },
    })
}

fn project_async_copy_completion_work(
    arch: ArchId,
    l1_policy: LdgstsL1Policy,
    payload_bytes: u64,
    destination_topology: Option<LdgstsDestinationTopology>,
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<u32, ReplayError> {
    // The flat SM120 alpha path returns before this helper and charges an
    // explicit global-demand approximation.  Any caller that reaches the
    // compact async completion machinery must therefore be an architecture
    // with an admitted LDGSTS protocol; never map SM120 to SM86 here.
    if arch == ArchId::Sm120 {
        return Err(ReplayError::UnsupportedMemoryPath {
            arch,
            class: pb::MemoryOpClass::AsyncCopy,
        });
    }
    let timing_arch = arch;
    let throughput = timing_arch
        .ldgsts_completion_throughput()
        .require("ldgsts_completion_throughput")
        .map_err(IssueError::from)?;
    let cache_all_rate = throughput.cache_all_payload_bytes_per_cycle;
    let bypass_rate = throughput.bypass_payload_bytes_per_cycle;
    let units_per_cycle = cache_all_rate.max(bypass_rate);
    if cache_all_rate == 0
        || bypass_rate == 0
        || units_per_cycle % cache_all_rate != 0
        || units_per_cycle % bypass_rate != 0
    {
        return Err(IssueError::InvalidLdgstsCompletionThroughput {
            arch,
            cache_all_bytes_per_cycle: cache_all_rate,
            bypass_bytes_per_cycle: bypass_rate,
        }
        .into());
    }
    let route_rate = match l1_policy {
        LdgstsL1Policy::CacheAtAllLevels => cache_all_rate,
        LdgstsL1Policy::BypassL1 => bypass_rate,
    };
    let units_per_byte = units_per_cycle / route_rate;
    let payload_work = payload_bytes.checked_mul(units_per_byte).ok_or(
        IssueError::TimeMultiplicationOverflow {
            operation: "LDGSTS payload-fallback completion work",
            lhs: payload_bytes,
            rhs: units_per_byte,
        },
    )?;
    if l1_policy == LdgstsL1Policy::BypassL1
        && let Some(topology) = destination_topology
    {
        let payload_work_compact = u32::try_from(payload_work).map_err(|_| {
            ReplayError::AsyncCopyCompletionWorkOutOfRange {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                completion_work: payload_work,
                max_completion_work: ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX,
            }
        })?;
        // A topology-only copy has executing destination lanes but no source
        // cohort. The compact observation token uses one reserved cohort slot
        // to keep the token non-inactive; the detailed sidecar remains empty.
        let source_cohorts = topology.source_cohorts.max(1);
        return encode_async_copy_payload_and_topology(
            payload_work_compact,
            source_cohorts,
            topology.destination_line_fanout_total,
            topology.cache_all_bank_wavefronts_max,
            topology.bypass_wavefronts_max,
        )
        .ok_or(ReplayError::AsyncCopyCompletionWorkOutOfRange {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            completion_work: payload_work
                .max(u64::from(topology.destination_line_fanout_total))
                .max(u64::from(topology.cache_all_bank_wavefronts_max))
                .max(u64::from(topology.bypass_wavefronts_max)),
            max_completion_work: ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX,
        });
    }
    let exact_destination = if l1_policy == LdgstsL1Policy::CacheAtAllLevels {
        match destination_topology {
            Some(topology) => {
                let wavefronts = topology.cache_all_bank_wavefronts_total;
                let timing = timing_arch
                    .ldgsts_destination_wavefront_timing()
                    .require("ldgsts_destination_wavefront_timing")
                    .map_err(IssueError::from)?;
                match timing.projection {
                    LdgstsDestinationWavefrontProjection::SourceCohorts128BytesBanks32Words4Bytes => {}
                }
                if timing.cycles_per_wavefront == 0 {
                    return Err(IssueError::InvalidLdgstsDestinationWavefrontTiming {
                        arch,
                        cycles_per_wavefront: timing.cycles_per_wavefront,
                    }
                    .into());
                }
                let destination_cycles = timing
                    .cycles_per_wavefront
                    .checked_mul(u64::from(wavefronts))
                    .ok_or(IssueError::TimeMultiplicationOverflow {
                        operation: "LDGSTS destination-wavefront completion cycles",
                        lhs: timing.cycles_per_wavefront,
                        rhs: u64::from(wavefronts),
                    })?;
                let destination_work = destination_cycles.checked_mul(units_per_cycle).ok_or(
                    IssueError::TimeMultiplicationOverflow {
                        operation: "LDGSTS destination-wavefront completion work",
                        lhs: destination_cycles,
                        rhs: units_per_cycle,
                    },
                )?;
                Some((wavefronts, destination_work))
            }
            None => None,
        }
    } else {
        // Focused bypass probes show a different source/destination coupling.
        // The compact token above retains its trace-derived topology for cold
        // diagnostics, while production timing continues to use payload work
        // until a bypass-specific per-copy mechanism is characterized.
        None
    };
    if payload_work != 0
        && let Some((_wavefronts, destination_work)) = exact_destination
        && destination_work >= payload_work
    {
        let compact = u32::try_from(destination_work).map_err(|_| {
            ReplayError::AsyncCopyCompletionWorkOutOfRange {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                completion_work: destination_work,
                max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
            }
        })?;
        return encode_async_copy_destination_work(compact).ok_or(
            ReplayError::AsyncCopyCompletionWorkOutOfRange {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                completion_work: destination_work,
                max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
            },
        );
    }
    let completion_work = exact_destination.map_or(payload_work, |(_, destination_work)| {
        payload_work.max(destination_work)
    });
    let compact = u32::try_from(completion_work).map_err(|_| {
        ReplayError::AsyncCopyCompletionWorkOutOfRange {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            completion_work,
            max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
        }
    })?;
    encode_async_copy_payload_work(compact).ok_or(ReplayError::AsyncCopyCompletionWorkOutOfRange {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        completion_work,
        max_completion_work: ASYNC_COPY_COMPLETION_VALUE_MASK,
    })
}

fn checked_staged_async_copy_kind(
    l1_policy: LdgstsL1Policy,
    start: usize,
    count: usize,
    completion_work: u32,
) -> Option<AccessKind> {
    let len = u16::try_from(count).ok()?;
    start.checked_add(count)?;
    Some(AccessKind::GlobalAsyncCopy {
        off: start,
        len,
        completion_work,
        l1_policy,
    })
}

fn checked_staged_local_kind(
    direction: LocalDirection,
    start: usize,
    count: usize,
) -> Option<AccessKind> {
    let len = u16::try_from(count).ok()?;
    start.checked_add(count)?;
    Some(match direction {
        LocalDirection::Load => AccessKind::LocalLoad { off: start, len },
        LocalDirection::Store => AccessKind::LocalStore { off: start, len },
    })
}

fn append_staged_sectors(
    kind: Option<AccessKind>,
    sectors: &mut Vec<i64>,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let count = sector_scratch.len();
    let overflow = |sectors: usize| ReplayError::SectorCountOverflow {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        sectors,
    };
    // `AccessKind::off` spans the native `usize` arena domain. Only one
    // record's compact length narrows to `u16`.
    let kind = kind.ok_or_else(|| overflow(count))?;
    sectors
        .try_reserve(count)
        .map_err(|_| ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source: SectorExpansionError::CapacityUnavailable {
                buffer: "warp sector arena",
                additional: count,
            },
        })?;
    sectors.extend_from_slice(sector_scratch);
    Ok(kind)
}

fn append_staged_global_sectors(
    role: MemRole,
    sectors: &mut Vec<i64>,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    let l2_requests = if matches!(
        role,
        MemRole::BypassLoad(CacheOpToken::Atom) | MemRole::Reduction
    ) {
        atomic_l2_request_count_128b(sector_scratch)
    } else {
        0
    };
    append_staged_sectors(
        checked_staged_global_kind(role, start, sector_scratch.len(), l2_requests),
        sectors,
        sector_scratch,
        path,
        warp_id,
        record_ix,
    )
}

/// Reconstruct exact L2 atomic request topology from the alias-expanded 32-byte
/// sector stream. Each request carries sectors from one 128-byte line; repeated
/// same-word lanes appear as repeated sector ids, so the line requires the
/// maximum sector multiplicity and distinct lines add.
fn atomic_l2_request_count_128b(sectors: &[i64]) -> u16 {
    let mut by_line: BTreeMap<i64, BTreeMap<i64, u16>> = BTreeMap::new();
    for &sector in sectors {
        let count = by_line
            .entry(sector.div_euclid(4))
            .or_default()
            .entry(sector)
            .or_default();
        *count = count
            .checked_add(1)
            .expect("one warp's bounded sector span cannot overflow u16");
    }
    let requests = by_line.values().fold(0_u32, |total, line| {
        total + u32::from(line.values().copied().max().unwrap_or(0))
    });
    u16::try_from(requests).expect("atomic L2 requests cannot exceed the bounded sector span")
}

fn append_staged_async_copy_sectors(
    arch: ArchId,
    l1_policy: LdgstsL1Policy,
    payload_bytes: u64,
    destination_topology: Option<LdgstsDestinationTopology>,
    source_cohorts: Option<&[LdgstsSourceCohort]>,
    sector_cohort_masks: Option<&[u32]>,
    sectors: &mut Vec<i64>,
    cohort_replay: &mut AsyncCopyCohortReplay,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
    completion_work_override: Option<u32>,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    let count = sector_scratch.len();
    let source_active = count != 0;
    if completion_work_override.is_none() && source_active != (payload_bytes != 0) {
        return Err(ReplayError::AsyncCopyPayloadMismatch {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source_active,
            sectors: count,
            payload_bytes,
        });
    }
    let completion_work = completion_work_override.map_or_else(
        || {
            project_async_copy_completion_work(
                arch,
                l1_policy,
                payload_bytes,
                destination_topology,
                path,
                warp_id,
                record_ix,
            )
        },
        Ok,
    )?;
    let overflow = |sectors: usize| ReplayError::SectorCountOverflow {
        path: path.to_path_buf(),
        warp_id,
        record_ix,
        sectors,
    };
    let kind = checked_staged_async_copy_kind(l1_policy, start, count, completion_work)
        .ok_or_else(|| overflow(count))?;

    let cohort_projection = match (source_cohorts, sector_cohort_masks) {
        (None, None) => None,
        (Some(cohorts), Some(masks))
            if !source_active && cohorts.is_empty() && masks.is_empty() =>
        {
            None
        }
        (Some(cohorts), Some(masks)) => {
            if masks.len() != count {
                return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    field: "source-sector membership count",
                    expected: count,
                    got: masks.len(),
                });
            }
            if cohorts.is_empty() || cohorts.len() > u32::BITS as usize {
                return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                    path: path.to_path_buf(),
                    warp_id,
                    record_ix,
                    field: "cohort count in u32 membership domain",
                    expected: usize::from(source_active),
                    got: cohorts.len(),
                });
            }
            Some((cohorts, masks))
        }
        (Some(cohorts), None) => {
            return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                field: "paired source-sector cohort masks",
                expected: count,
                got: cohorts.len(),
            });
        }
        (None, Some(masks)) => {
            return Err(ReplayError::AsyncCopyCohortProjectionMismatch {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                field: "paired source cohort descriptors",
                expected: 1,
                got: masks.len(),
            });
        }
    };

    sectors
        .try_reserve(count)
        .map_err(|_| ReplayError::SectorExpansion {
            path: path.to_path_buf(),
            warp_id,
            record_ix,
            source: SectorExpansionError::CapacityUnavailable {
                buffer: "warp sector arena",
                additional: count,
            },
        })?;
    if let Some((cohorts, masks)) = cohort_projection {
        for (buffer, additional, result) in [
            (
                "span",
                1,
                cohort_replay.spans.try_reserve(1).map_err(|_| ()),
            ),
            (
                "descriptor",
                cohorts.len(),
                cohort_replay
                    .cohorts
                    .try_reserve(cohorts.len())
                    .map_err(|_| ()),
            ),
            (
                "membership-mask",
                masks.len(),
                cohort_replay
                    .sector_membership_masks
                    .try_reserve(masks.len())
                    .map_err(|_| ()),
            ),
        ] {
            result.map_err(|_| ReplayError::AsyncCopyCohortCapacityUnavailable {
                path: path.to_path_buf(),
                warp_id,
                record_ix,
                buffer,
                additional,
            })?;
        }
    }

    sectors.extend_from_slice(sector_scratch);
    if let Some((cohorts, masks)) = cohort_projection {
        let cohort_off = cohort_replay.cohorts.len();
        let membership_off = cohort_replay.sector_membership_masks.len();
        cohort_replay
            .cohorts
            .extend(cohorts.iter().map(|cohort| AsyncCopyCohortDescriptor {
                payload_bytes: cohort.payload_bytes,
                destination_lines: cohort.destination_lines,
                bank_wavefronts: cohort.bank_wavefronts,
                bypass_wavefronts: cohort.bypass_wavefronts,
            }));
        cohort_replay
            .sector_membership_masks
            .extend_from_slice(masks);
        cohort_replay.spans.push(AsyncCopyCohortSpan {
            instruction: record_ix,
            sector_off: start,
            sector_len: u16::try_from(count).expect("checked async-copy replay kind length"),
            cohort_off,
            cohort_len: u8::try_from(cohorts.len())
                .expect("u32 membership domain fits the u8 cohort span"),
            membership_off,
        });
    }
    Ok(kind)
}

fn append_staged_local_sectors(
    direction: LocalDirection,
    sectors: &mut Vec<i64>,
    sector_scratch: &[i64],
    path: &Path,
    warp_id: i32,
    record_ix: usize,
) -> Result<AccessKind, ReplayError> {
    let start = sectors.len();
    append_staged_sectors(
        checked_staged_local_kind(direction, start, sector_scratch.len()),
        sectors,
        sector_scratch,
        path,
        warp_id,
        record_ix,
    )
}

