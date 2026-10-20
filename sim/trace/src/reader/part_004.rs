// Lexical implementation section 5/5; included by the parent facade.
fn project_ldgsts_source_cohorts_128b_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    source_sectors: &[i64],
    word_scratch: &mut Vec<i64>,
) -> Result<ProjectedLdgstsSourceCohorts, LdgstsDestinationProjectionError> {
    let mut source_index = None;
    let mut source_count = 0_usize;
    let mut destination_index = None;
    let mut destination_count = 0_usize;
    for (operand, address) in preflight.instruction().addresses.iter().enumerate() {
        let space = address
            .address_space
            .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
        match space {
            Some(pb::MemorySpace::Global | pb::MemorySpace::Generic) => {
                source_count += 1;
                source_index.get_or_insert(operand);
            }
            Some(pb::MemorySpace::Shared) => {
                destination_count += 1;
                destination_index.get_or_insert(operand);
            }
            Some(pb::MemorySpace::Unspecified | pb::MemorySpace::Local) | None => {}
        }
    }
    if source_count != 1 {
        return Err(LdgstsDestinationProjectionError::SourceOperandCount { got: source_count });
    }
    if destination_count != 1 {
        return Err(LdgstsDestinationProjectionError::DestinationOperandCount {
            got: destination_count,
        });
    }
    let source = preflight
        .decoded_address(source_index.expect("count proved source index"))
        .expect("source index came from instruction addresses");
    let destination = preflight
        .decoded_address(destination_index.expect("count proved destination index"))
        .expect("destination index came from instruction addresses");
    if source.data_width_bytes != destination.data_width_bytes {
        return Err(LdgstsDestinationProjectionError::OperandWidthMismatch {
            source_bytes: source.data_width_bytes,
            destination_bytes: destination.data_width_bytes,
        });
    }
    let width = u64::try_from(source.data_width_bytes)
        .unwrap_or_else(|_| unreachable!("preflight admitted a non-positive operand width"));
    let width_u32 = u32::try_from(width)
        .unwrap_or_else(|_| unreachable!("one source transfer width escaped u32"));
    let instruction_mask = preflight.meta().and_mask;

    const EMPTY_TRANSFER: LdgstsLaneTransfer = LdgstsLaneTransfer {
        source_start: 0,
        source_end: 0,
        source_first_sector: 0,
        source_last_sector: 0,
        destination_first_word: 0,
        destination_last_word: 0,
        lane: 0,
        cohort: 0,
    };
    let mut transfers = [EMPTY_TRANSFER; lanes::LANES_PER_WARP];
    let mut transfer_count = 0_usize;
    let mut zero_fill_words = Vec::new();
    let mut zero_fill_transfers = 0_u32;
    let mut projected_words = 0_u64;
    for lane in 0..lanes::LANES_PER_WARP {
        let source_address = source.lane_vector[lane];
        let destination_address = destination.lane_vector[lane];
        if width > 128 {
            return Err(
                LdgstsDestinationProjectionError::SourceTransferExceedsCohort {
                    lane,
                    width_bytes: width,
                },
            );
        }
        if destination_address == lanes::LANE_INACTIVE_SENTINEL
            && (instruction_mask >> lane) & 1 == 1
        {
            return Err(LdgstsDestinationProjectionError::DestinationLaneMissing { lane });
        }
        if destination_address == lanes::LANE_INACTIVE_SENTINEL {
            continue;
        }
        if source_address == lanes::LANE_INACTIVE_SENTINEL {
            // cp.async ignore-src lanes have no global source cohort, but an
            // executing lane still performs destination zero-fill work. Keep
            // that work in a separate zero-payload cohort so source sectors
            // remain empty while async completion stays nonzero.
            let destination_width_offset = i64::try_from(width - 1)
                .unwrap_or_else(|_| unreachable!("128-byte transfer width escaped i64"));
            let destination_last_byte = destination_address
                .checked_add(destination_width_offset)
                .ok_or(
                LdgstsDestinationProjectionError::DestinationIntervalOverflow {
                    lane,
                    base: destination_address,
                    width_bytes: width,
                },
            )?;
            let first_word = destination_address.div_euclid(4);
            let last_word = destination_last_byte.div_euclid(4);
            let words = u64::try_from(last_word - first_word + 1).unwrap_or_else(|_| {
                unreachable!("checked destination interval produced negative words")
            });
            projected_words = projected_words.checked_add(words).ok_or(
                LdgstsDestinationProjectionError::InstructionWordLimit {
                    words: u64::MAX,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                },
            )?;
            if projected_words > MAX_SHARED_WORDS_PER_INSTRUCTION {
                return Err(LdgstsDestinationProjectionError::InstructionWordLimit {
                    words: projected_words,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                });
            }
            let additional = usize::try_from(words)
                .unwrap_or_else(|_| unreachable!("bounded destination words escaped usize"));
            zero_fill_words.try_reserve(additional).map_err(|_| {
                LdgstsDestinationProjectionError::CapacityUnavailable { additional }
            })?;
            zero_fill_words.extend(first_word..=last_word);
            zero_fill_transfers = zero_fill_transfers.checked_add(1).ok_or(
                LdgstsDestinationProjectionError::InstructionWordLimit {
                    words: u64::MAX,
                    limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
                },
            )?;
            continue;
        }
        let source_start = source_address as u64;
        let source_end = source_start.checked_add(width).ok_or(
            LdgstsDestinationProjectionError::SourceIntervalOverflow {
                lane,
                base: source_start,
                width_bytes: width,
            },
        )?;
        let destination_width_offset = i64::try_from(width - 1)
            .unwrap_or_else(|_| unreachable!("128-byte transfer width escaped i64"));
        let destination_last_byte = destination_address
            .checked_add(destination_width_offset)
            .ok_or(
                LdgstsDestinationProjectionError::DestinationIntervalOverflow {
                    lane,
                    base: destination_address,
                    width_bytes: width,
                },
            )?;
        transfers[transfer_count] = LdgstsLaneTransfer {
            source_start,
            source_end,
            source_first_sector: source_address >> 5,
            source_last_sector: ((source_end - 1) as i64) >> 5,
            destination_first_word: destination_address.div_euclid(4),
            destination_last_word: destination_last_byte.div_euclid(4),
            lane: lane as u8,
            cohort: 0,
        };
        transfer_count += 1;
    }
    transfers[..transfer_count].sort_unstable_by_key(|transfer| {
        (transfer.source_start, transfer.source_end, transfer.lane)
    });

    word_scratch.clear();
    let mut cohort_start = None;
    let mut cohort_active_transfers = 0_u32;
    let mut cohort_payload_bytes = 0_u32;
    let mut cohort_ordinal = 0_u8;
    let mut cohorts = Vec::new();
    for transfer in &mut transfers[..transfer_count] {
        if cohort_start.is_some_and(|start| transfer.source_end - start > 128) {
            finish_ldgsts_destination_cohort(
                word_scratch,
                cohort_active_transfers,
                cohort_payload_bytes,
                &mut cohorts,
            )?;
            cohort_start = None;
            cohort_active_transfers = 0;
            cohort_payload_bytes = 0;
            cohort_ordinal = cohort_ordinal
                .checked_add(1)
                .ok_or(LdgstsDestinationProjectionError::SourceCohortCountOverflow)?;
        }
        cohort_start.get_or_insert(transfer.source_start);
        transfer.cohort = cohort_ordinal;
        cohort_active_transfers = cohort_active_transfers
            .checked_add(1)
            .expect("one source cohort cannot exceed the 32-lane warp");
        cohort_payload_bytes = cohort_payload_bytes
            .checked_add(width_u32)
            .expect("one source cohort payload cannot exceed 4096 bytes");
        let words =
            u64::try_from(transfer.destination_last_word - transfer.destination_first_word + 1)
                .unwrap_or_else(|_| {
                    unreachable!("checked destination interval produced negative words")
                });
        projected_words = projected_words.checked_add(words).ok_or(
            LdgstsDestinationProjectionError::InstructionWordLimit {
                words: u64::MAX,
                limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
            },
        )?;
        if projected_words > MAX_SHARED_WORDS_PER_INSTRUCTION {
            return Err(LdgstsDestinationProjectionError::InstructionWordLimit {
                words: projected_words,
                limit: MAX_SHARED_WORDS_PER_INSTRUCTION,
            });
        }
        let additional = usize::try_from(words)
            .unwrap_or_else(|_| unreachable!("bounded destination words escaped usize"));
        word_scratch
            .try_reserve(additional)
            .map_err(|_| LdgstsDestinationProjectionError::CapacityUnavailable { additional })?;
        word_scratch.extend(transfer.destination_first_word..=transfer.destination_last_word);
    }
    finish_ldgsts_destination_cohort(
        word_scratch,
        cohort_active_transfers,
        cohort_payload_bytes,
        &mut cohorts,
    )?;

    let mut topology = recompose_ldgsts_destination_topology(&cohorts)?;
    if zero_fill_transfers != 0 {
        let mut zero_cohorts = Vec::new();
        finish_ldgsts_destination_cohort(
            &mut zero_fill_words,
            zero_fill_transfers,
            0,
            &mut zero_cohorts,
        )?;
        let zero = recompose_ldgsts_destination_topology(&zero_cohorts)?;
        topology.destination_line_fanout_total = topology
            .destination_line_fanout_total
            .checked_add(zero.destination_line_fanout_total)
            .ok_or(LdgstsDestinationProjectionError::DestinationLineFanoutOverflow)?;
        topology.destination_line_fanout_max = topology
            .destination_line_fanout_max
            .max(zero.destination_line_fanout_max);
        topology.cache_all_bank_wavefronts_total = topology
            .cache_all_bank_wavefronts_total
            .checked_add(zero.cache_all_bank_wavefronts_total)
            .ok_or(LdgstsDestinationProjectionError::WavefrontCountOverflow)?;
        topology.cache_all_bank_wavefronts_max = topology
            .cache_all_bank_wavefronts_max
            .max(zero.cache_all_bank_wavefronts_max);
        topology.bypass_wavefronts_total = topology
            .bypass_wavefronts_total
            .checked_add(zero.bypass_wavefronts_total)
            .ok_or(LdgstsDestinationProjectionError::BypassWavefrontCountOverflow)?;
        topology.bypass_wavefronts_max = topology
            .bypass_wavefronts_max
            .max(zero.bypass_wavefronts_max);
    }
    debug_assert_eq!(
        cohorts
            .iter()
            .map(|cohort| u64::from(cohort.payload_bytes))
            .sum::<u64>(),
        u64::from(width_u32) * transfer_count as u64
    );

    let mut source_sector_cohort_masks = Vec::new();
    source_sector_cohort_masks
        .try_reserve(source_sectors.len())
        .map_err(|_| LdgstsDestinationProjectionError::CapacityUnavailable {
            additional: source_sectors.len(),
        })?;
    for &sector in source_sectors {
        let mut mask = 0_u32;
        for transfer in &transfers[..transfer_count] {
            if sector >= transfer.source_first_sector && sector <= transfer.source_last_sector {
                mask |= 1_u32
                    .checked_shl(u32::from(transfer.cohort))
                    .unwrap_or_else(|| unreachable!("32-lane cohort ordinal escaped u32 mask"));
            }
        }
        debug_assert_ne!(mask, 0, "compact source sector must belong to a cohort");
        source_sector_cohort_masks.push(mask);
    }
    Ok(ProjectedLdgstsSourceCohorts {
        topology,
        cohorts,
        source_sector_cohort_masks,
    })
}

/// Pair each active lane's LDGSTS source and destination transfer, form
/// flexible source cohorts whose complete byte span is at most 128 bytes, and
/// retain destination-line fanout separately from 32-bank/4-byte-word
/// multiplicity. The bypass projection is derived inside each cohort before
/// aggregation; it is an observational topology coordinate, not a timing rule.
/// Cohorts are anchored at the smallest uncovered source address rather than at
/// absolute 128-byte cache-line boundaries.
pub fn project_ldgsts_source_cohort_topology_128b_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<LdgstsDestinationTopology, LdgstsDestinationProjectionError> {
    Ok(project_ldgsts_source_cohorts_128b_32x4(preflight, &[], word_scratch)?.topology)
}

/// Compatibility wrapper for the established cache-all bank-wavefront scalar.
pub fn project_ldgsts_source_cohort_wavefronts_128b_32x4(
    preflight: PreflightedMemoryAccess<'_>,
    word_scratch: &mut Vec<i64>,
) -> Result<u32, LdgstsDestinationProjectionError> {
    Ok(
        project_ldgsts_source_cohort_topology_128b_32x4(preflight, word_scratch)?
            .cache_all_bank_wavefronts_total,
    )
}

/// A memory instruction decoded to what the memory hierarchy consumes —
/// the ALLOCATING convenience form (one heap `Vec` per memory record).
/// Hot walk paths (millions of records per kernel) use
/// [`visit_memory_access`] instead, which streams [`AddressAccess`] values
/// through a sink with zero per-record heap allocation.
#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub op_class: pb::MemoryOpClass,
    /// See [`MemoryAccessMeta::access_kind`].
    pub access_kind: Option<pb::MemoryOpAccessKind>,
    /// See [`MemoryAccessMeta::cache_op`].
    pub cache_op: Option<pb::InstructionCacheOp>,
    /// See [`MemoryAccessMeta::and_mask`].
    pub and_mask: u32,
    /// One decoded record per wire address record (empty when and_mask==0
    /// or the producer emitted no address payload).
    pub accesses: Vec<AddressAccess>,
}

fn known_enum<T: TryFrom<i32>>(
    raw: i32,
    field: &'static str,
    record: &impl Fn() -> String,
) -> Result<T, TraceError> {
    T::try_from(raw).map_err(|_| TraceError::UnknownEnumValue {
        record: record(),
        field,
        raw,
    })
}

fn memory_space_is_compatible(class: pb::MemoryOpClass, space: pb::MemorySpace) -> bool {
    match class {
        pb::MemoryOpClass::Global => {
            matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic)
        }
        pb::MemoryOpClass::Shared => space == pb::MemorySpace::Shared,
        pb::MemoryOpClass::Local => space == pb::MemorySpace::Local,
        pb::MemoryOpClass::Atomic => matches!(
            space,
            pb::MemorySpace::Global
                | pb::MemorySpace::Generic
                | pb::MemorySpace::Shared
                | pb::MemorySpace::Local
        ),
        // The generic structural pass admits the v3 destination space; the
        // enclosing payload-contract shape gate subsequently requires the exact
        // ordered (GLOBAL|GENERIC source, SHARED destination) pair. Historical
        // source-only contracts retain their stricter one-operand shape there.
        pb::MemoryOpClass::AsyncCopy => matches!(
            space,
            pb::MemorySpace::Global | pb::MemorySpace::Generic | pb::MemorySpace::Shared
        ),
        pb::MemoryOpClass::Unspecified
        | pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => true,
    }
}

fn decode_address_access(
    addr: &pb::Address,
    instruction_mask: u32,
    inst_width: i32,
    record: &impl Fn() -> String,
) -> Result<AddressAccess, TraceError> {
    let lane_mask = addr.lane_mask.unwrap_or(instruction_mask);
    if lane_mask & !instruction_mask != 0 {
        return Err(TraceError::AddressLaneMaskOutsideInstructionMask {
            record: record(),
            lane_mask,
            instruction_mask,
        });
    }
    // Match the Python gate's aggregate mask diagnosis before positional
    // reconstruction. A malformed list_all length has higher precedence there;
    // every other per-slot disagreement remains lane_position_mismatch.
    if addr.compression_format != lanes::FORMAT_LIST_ALL
        || addr.addrs.len() == lanes::LANES_PER_WARP
    {
        let real_count = if addr.compression_format == lanes::FORMAT_LIST_ALL {
            addr.addrs
                .iter()
                .filter(|&&slot| slot != lanes::LANE_INACTIVE_SENTINEL)
                .count()
        } else {
            addr.addrs.len()
        };
        if (lane_mask == 0 && real_count > 0)
            || (lane_mask != 0
                && addr.compression_format == lanes::FORMAT_LIST_ALL
                && real_count == 0)
        {
            return Err(TraceError::LaneMaskInconsistent { record: record() });
        }
    }

    let lane_vector =
        lanes::reconstruct_lane_vector(addr, lane_mask).map_err(|source| TraceError::Lane {
            record: record(),
            source,
        })?;
    let data_width_bytes = if addr.data_width > 0 {
        addr.data_width
    } else {
        inst_width
    };
    let address_space = match addr.address_space {
        None => None,
        Some(raw) => Some(known_enum::<pb::MemorySpace>(raw, "address_space", record)?),
    };
    Ok(AddressAccess {
        lane_vector,
        lane_mask,
        data_width_bytes,
        address_space,
        allocation_id: addr.allocation_id,
    })
}

/// Validate one traced instruction's complete memory wire contract without
/// invoking a consumer callback. A successful token borrows the immutable
/// instruction and may later stream its address records without another full
/// preflight.
///
/// "Memory instruction" uses the canonical Python predicate: address records OR
/// a present `memory_op_class`. Data-width alone does not reclassify a non-memory
/// record; access-kind or cache-op metadata on such a record is rejected as a
/// producer contradiction. Every instruction must carry `warpid_tb`, a
/// positive function id, both masks, and a low-32-consistent `vpc`. Memory records
/// additionally require a specified class and positive instruction width; unknown
/// enum values are typed schema skew, explicit memory-space UNSPECIFIED is rejected,
/// and every address record must reconstruct positionally. Non-memory instructions
/// return `Ok(None)` after their universal fields have been validated.
pub fn preflight_memory_access<'a>(
    inst: &'a pb::Instruction,
    record: impl Fn() -> String,
) -> Result<Option<PreflightedMemoryAccess<'a>>, TraceError> {
    let n_addresses = inst.addresses.len();
    // Canonical Python predicate: data_width and additive provenance fields do
    // not by themselves reclassify an otherwise non-memory instruction. The
    // additive fields are nevertheless contradictory on that record and are
    // rejected after their enum values receive complete wire validation.
    let looks_like_memory = n_addresses > 0 || inst.memory_op_class.is_some();

    // Keep the Python validator's fail-fast precedence: memory width first,
    // followed by the fields required on every instruction.
    let inst_width = if looks_like_memory {
        let Some(inst_width) = inst.data_width else {
            return Err(TraceError::DataWidthMissing { record: record() });
        };
        if inst_width <= 0 {
            return Err(TraceError::DataWidthInvalid {
                record: record(),
                got: inst_width,
            });
        }
        inst_width
    } else {
        0
    };

    if inst.warpid_tb.is_none() {
        return Err(TraceError::WarpidTbMissing { record: record() });
    }
    if inst.function_unique_id <= 0 {
        return Err(TraceError::FunctionUniqueIdInvalid {
            record: record(),
            got: inst.function_unique_id,
        });
    }
    let Some(active_mask) = inst.active_mask else {
        return Err(TraceError::ActiveMaskMissing { record: record() });
    };
    let Some(predicate_mask) = inst.predicate_mask else {
        return Err(TraceError::PredicateMaskMissing { record: record() });
    };
    let Some(vpc) = inst.vpc else {
        return Err(TraceError::VpcMissing { record: record() });
    };
    let vpc_low_32 = vpc as u32;
    if vpc_low_32 != inst.pc {
        return Err(TraceError::VpcInconsistentWithPc {
            record: record(),
            vpc_low_32,
            pc: inst.pc,
        });
    }

    if !looks_like_memory {
        let access_kind = match inst.memory_op_access_kind {
            None => None,
            Some(raw) => {
                let kind: pb::MemoryOpAccessKind =
                    known_enum(raw, "memory_op_access_kind", &record)?;
                if kind == pb::MemoryOpAccessKind::Unspecified {
                    return Err(TraceError::MemoryOpAccessKindUnspecified { record: record() });
                }
                Some(kind)
            }
        };
        let cache_op = match inst.instruction_cache_op {
            None => None,
            Some(raw) => {
                let op: pb::InstructionCacheOp = known_enum(raw, "instruction_cache_op", &record)?;
                if op == pb::InstructionCacheOp::Unspecified {
                    return Err(TraceError::InstructionCacheOpUnspecified { record: record() });
                }
                Some(op)
            }
        };
        if access_kind.is_some() {
            return Err(TraceError::MemoryMetadataOnNonMemoryInstruction {
                record: record(),
                field: "memory_op_access_kind",
            });
        }
        if cache_op.is_some() {
            return Err(TraceError::MemoryMetadataOnNonMemoryInstruction {
                record: record(),
                field: "instruction_cache_op",
            });
        }
        return Ok(None);
    }

    let Some(op_class_raw) = inst.memory_op_class else {
        return Err(TraceError::MemoryOpClassMissing {
            record: record(),
            n_addresses,
        });
    };
    let op_class: pb::MemoryOpClass = known_enum(op_class_raw, "memory_op_class", &record)?;
    if op_class == pb::MemoryOpClass::Unspecified {
        return Err(TraceError::MemoryOpClassUnspecified { record: record() });
    }

    let and_mask = active_mask & predicate_mask;
    // Validate every address, including zero-mask records. Those stream no
    // callbacks, but malformed lane payloads and unknown enums remain failures.
    for addr in &inst.addresses {
        decode_address_access(addr, and_mask, inst_width, &record)?;
    }
    // Structural preflight above intentionally completes before this semantic
    // compatibility gate. Explicit UNSPECIFIED is never a legacy fallback;
    // absence alone retains the class-derived behavior.
    for (operand, addr) in inst.addresses.iter().enumerate() {
        let Some(raw) = addr.address_space else {
            continue;
        };
        let space: pb::MemorySpace = known_enum(raw, "address_space", &record)?;
        if space == pb::MemorySpace::Unspecified {
            return Err(TraceError::MemorySpaceUnspecified {
                record: record(),
                operand,
            });
        }
        if !memory_space_is_compatible(op_class, space) {
            return Err(TraceError::MemoryOpClassAddressSpaceConflict {
                record: record(),
                operand,
                class: op_class,
                space,
            });
        }
    }

    let access_kind = match inst.memory_op_access_kind {
        None => None,
        Some(raw) => {
            let kind: pb::MemoryOpAccessKind = known_enum(raw, "memory_op_access_kind", &record)?;
            if kind == pb::MemoryOpAccessKind::Unspecified {
                return Err(TraceError::MemoryOpAccessKindUnspecified { record: record() });
            }
            Some(kind)
        }
    };
    let cache_op = match inst.instruction_cache_op {
        None => None,
        Some(raw) => {
            let op: pb::InstructionCacheOp = known_enum(raw, "instruction_cache_op", &record)?;
            if op == pb::InstructionCacheOp::Unspecified {
                return Err(TraceError::InstructionCacheOpUnspecified { record: record() });
            }
            Some(op)
        }
    };

    Ok(Some(PreflightedMemoryAccess {
        inst,
        meta: MemoryAccessMeta {
            op_class,
            access_kind,
            cache_op,
            and_mask,
        },
        inst_width,
    }))
}

/// Interpret one traced instruction per the wire contract, streaming each
/// decoded address record into `sink` — the zero-per-record-allocation form for
/// hot walk paths. Full preflight completes before the first callback, so an
/// error never exposes a partial callback prefix. The sink borrows a
/// stack-resident [`AddressAccess`]; copy it out (`Copy`) only if it must outlive
/// the call.
pub fn visit_memory_access<F>(
    inst: &pb::Instruction,
    record: impl Fn() -> String,
    sink: F,
) -> Result<Option<MemoryAccessMeta>, TraceError>
where
    F: FnMut(&AddressAccess),
{
    let Some(preflight) = preflight_memory_access(inst, record)? else {
        return Ok(None);
    };
    let meta = preflight.meta();
    preflight.visit_addresses(sink);
    Ok(Some(meta))
}

/// One backing allocation's device VA range.
#[derive(Debug, Clone, Copy)]
struct AllocRange {
    base: u64,
    size: u64,
}

/// The `<trace_dir>/extra_info/allocations.json` registry — the same sidecar the
/// Python `trace_contract` validator reads (an array of
/// `{allocation_id, base_address (hex string), size_bytes, ...}`). Used to
/// enforce the address-carrying contract's allocation identity in the Rust
/// reader, at parity with the Python gate.
#[derive(Debug, Clone, Default)]
pub struct AllocationRegistry {
    by_id: BTreeMap<u32, AllocRange>,
}

/// The EXACT six-field allocation-sidecar entry, at parity with the Python
/// validator's shape gate (`the trace contract`: all six fields required,
/// no extras). `deny_unknown_fields` + all-required non-Option fields make serde
/// reject a missing field, an unexpected key, a negative/boolean `allocation_id`
/// or `size_bytes`, and a non-bool `alive`. The explicit VALUE checks (positive
/// id/size, 0x-prefixed base, duplicate id) run in `from_str` after.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAllocation {
    allocation_id: u32,
    // context_id / device_id / alive are producer metadata not consumed by the
    // range check (Python does not gate on `alive` — a freed-at-trace-end
    // allocation was valid during the trace), but each MUST be present and
    // correctly typed for exact-shape parity. `device_id` may be 0 or negative.
    #[allow(dead_code)]
    context_id: i64,
    #[allow(dead_code)]
    device_id: i64,
    base_address: String,
    size_bytes: u64,
    #[allow(dead_code)]
    alive: bool,
}

fn parse_hex_u64(s: &str) -> Option<u64> {
    // Canonical cross-runtime grammar: an exact 0x/0X prefix followed by one or
    // more ASCII hex digits. In particular, Python-style digit separators are
    // not wire syntax.
    let digits = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(digits, 16).ok()
}

impl AllocationRegistry {
    /// Return an address's byte offset within a validated allocation.
    ///
    /// This exposes allocation-relative identity without exposing or inventing
    /// a physical address. `None` means the id is unknown, the subtraction
    /// underflows, or the address is at/after the allocation end.
    pub fn relative_byte_offset(&self, allocation_id: u32, address: u64) -> Option<u64> {
        let range = self.by_id.get(&allocation_id)?;
        let offset = address.checked_sub(range.base)?;
        (offset < range.size).then_some(offset)
    }

    /// Return a compact 32-byte sector's offset within a sector-aligned
    /// allocation.
    ///
    /// Global CUDA allocations are sector aligned in admitted traces. A
    /// malformed/non-aligned synthetic registry fails closed rather than
    /// silently rounding its base.
    pub fn relative_sector_offset(&self, allocation_id: u32, absolute_sector: i64) -> Option<u64> {
        let range = self.by_id.get(&allocation_id)?;
        if range.base & 31 != 0 {
            return None;
        }
        let sector = u64::try_from(absolute_sector).ok()?;
        let base_sector = range.base >> 5;
        let offset = sector.checked_sub(base_sector)?;
        let byte_offset = offset.checked_mul(32)?;
        (byte_offset < range.size).then_some(offset)
    }

    /// Load from `<root>/extra_info/allocations.json`. `Ok(None)` if the file is
    /// absent (legacy traces, or traces with no GLOBAL/GENERIC active-lane record
    /// requiring identity). `Err(AllocationsSidecarMalformed)` on bad JSON, a
    /// non-hex `base_address`, or a duplicate `allocation_id`.
    pub fn load(root: &Path) -> Result<Option<Self>, TraceError> {
        let path = root.join("extra_info").join("allocations.json");
        let reader = AnchoredDir::open(root)?;
        let raw = match reader
            .read_optional_bounded_regular(
                Path::new("extra_info/allocations.json"),
                ALLOCATIONS_JSON_MAX_BYTES,
            )
            .map_err(|source| map_snapshot_read_error("allocations_json", source))?
        {
            Some(file) => file.into_raw(),
            None => return Ok(None),
        };
        Self::parse(&path, &raw).map(Some)
    }

    fn parse(path: &Path, raw: &[u8]) -> Result<Self, TraceError> {
        Self::parse_with_limits(
            path,
            raw,
            ALLOCATIONS_JSON_MAX_BYTES,
            ALLOCATIONS_JSON_MAX_NODES,
            ALLOCATION_ENTRY_MAX_COUNT,
        )
    }

    fn parse_with_limits(
        path: &Path,
        raw: &[u8],
        maximum_bytes: u64,
        maximum_nodes: u64,
        maximum_entries: u64,
    ) -> Result<Self, TraceError> {
        sidecar_byte_limit(path, "allocations_json", raw, maximum_bytes)?;
        match preflight_json_shape(raw, maximum_nodes, Some(maximum_entries)) {
            Ok(_) => {}
            Err(JsonPreflightError::Malformed(message)) => {
                return Err(TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message,
                });
            }
            Err(JsonPreflightError::ResourceLimit(error)) => {
                return Err(map_json_preflight_limit(path, "allocations_json", error));
            }
        }
        let entries: Vec<RawAllocation> =
            serde_json::from_slice(raw).map_err(|e| TraceError::AllocationsSidecarMalformed {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;
        let mut by_id = BTreeMap::new();
        for e in entries {
            // Python parity: allocation ids start at 1 (0 is the unresolved
            // sentinel) and every allocation has a positive size.
            if e.allocation_id == 0 {
                return Err(TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message: "allocation_id must be positive (0 is the unresolved sentinel)"
                        .to_string(),
                });
            }
            if e.size_bytes == 0 {
                return Err(TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message: format!("allocation_id {} has size_bytes == 0", e.allocation_id),
                });
            }
            let base = parse_hex_u64(&e.base_address).ok_or_else(|| {
                TraceError::AllocationsSidecarMalformed {
                    path: path.to_path_buf(),
                    message: format!(
                        "allocation_id {} has a non-0x-hex base_address {:?}",
                        e.allocation_id, e.base_address
                    ),
                }
            })?;
            if by_id
                .insert(
                    e.allocation_id,
                    AllocRange {
                        base,
                        size: e.size_bytes,
                    },
                )
                .is_some()
            {
                return Err(TraceError::AllocationIdDuplicate {
                    path: path.to_path_buf(),
                    allocation_id: e.allocation_id,
                });
            }
        }
        Ok(Self { by_id })
    }
}

/// Whether an effective per-address space requires global allocation identity.
/// A record's own `address_space` is authoritative (GLOBAL/GENERIC require it,
/// SHARED/LOCAL are exempt); absent falls back to the instruction-level GLOBAL
/// class — exactly the Python gate's rule.
fn record_requires_identity(
    space: Option<pb::MemorySpace>,
    inst_class: Option<pb::MemoryOpClass>,
) -> bool {
    match space {
        Some(pb::MemorySpace::Global) | Some(pb::MemorySpace::Generic) => true,
        Some(_) => false,
        None => inst_class == Some(pb::MemoryOpClass::Global),
    }
}

fn validate_address_carrying_shape(
    preflight: PreflightedMemoryAccess<'_>,
    contract: MemoryPayloadContract,
    record: &impl Fn() -> String,
) -> Result<(), TraceError> {
    let meta = preflight.meta();
    let instruction = preflight.instruction();
    if !ADDRESS_CARRYING_PAYLOAD_CLASSES.contains(&meta.op_class) {
        return Err(TraceError::MemoryPayloadClassUndeclared {
            record: record(),
            class: meta.op_class,
        });
    }

    let operand_count = instruction.addresses.len();
    let invalid = |detail| {
        Err(TraceError::MemoryPayloadShapeInvalid {
            record: record(),
            class: meta.op_class,
            detail,
            got: operand_count,
        })
    };
    match meta.op_class {
        pb::MemoryOpClass::Global | pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => {
            if meta.and_mask != 0 && operand_count == 0 {
                return invalid("an active record requires at least one operand");
            }
        }
        pb::MemoryOpClass::Atomic => {
            if meta.and_mask == 0 && operand_count == 0 {
                return Ok(());
            }
            if operand_count != 1 {
                return invalid("expected exactly one operand, except zero-active may carry none");
            }
            if instruction.addresses[0].address_space.is_none() {
                return invalid("the sole operand requires an explicit concrete address_space");
            }
        }
        pb::MemoryOpClass::AsyncCopy => {
            if contract.carries_ldgsts_destination() {
                if operand_count != 2 {
                    return invalid(
                        "expected exactly an ordered GLOBAL/GENERIC source and SHARED destination",
                    );
                }
                let source_space = instruction.addresses[0]
                    .address_space
                    .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
                if !matches!(
                    source_space,
                    Some(pb::MemorySpace::Global | pb::MemorySpace::Generic)
                ) {
                    return invalid("operand 0 requires explicit GLOBAL or GENERIC source space");
                }
                let destination_space = instruction.addresses[1]
                    .address_space
                    .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
                if destination_space != Some(pb::MemorySpace::Shared) {
                    return invalid("operand 1 requires explicit SHARED destination space");
                }
            } else {
                if operand_count != 1 {
                    return invalid("expected exactly one source operand");
                }
                let source_space = instruction.addresses[0]
                    .address_space
                    .and_then(|raw| pb::MemorySpace::try_from(raw).ok());
                if !matches!(
                    source_space,
                    Some(pb::MemorySpace::Global | pb::MemorySpace::Generic)
                ) {
                    return invalid("the source operand requires explicit GLOBAL or GENERIC space");
                }
            }
        }
        pb::MemoryOpClass::Unspecified
        | pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => {
            unreachable!("undeclared classes return before shape validation")
        }
    }
    Ok(())
}

/// Enforce the Python-equivalent allocation identity on one instruction against
/// the loaded registry (`None` = no sidecar). The complete memory wire record is
/// preflighted before the first identity diagnosis, so an unknown enum or malformed
/// later lane vector cannot be masked by an earlier missing allocation id. A
/// GLOBAL/GENERIC-space record with active lanes must carry a non-zero
/// `allocation_id` in the registry containing every active lane address, or
/// the producer's explicit `unresolved_module_global` disposition. The latter
/// retains captured virtual addresses without claiming allocator bounds or
/// lifetime identity. SHARED/LOCAL records (e.g. ATOMS) are exempt.
pub fn check_allocation_identity(
    inst: &pb::Instruction,
    registry: Option<&AllocationRegistry>,
    record: impl Fn() -> String,
) -> Result<(), TraceError> {
    let Some(preflight) = preflight_memory_access(inst, &record)? else {
        return Ok(());
    };
    check_preflighted_allocation_identity(preflight, registry, record)
}

/// Apply the complete per-instruction contract and enclosing warp-identity
/// contract to every record in one decoded threadblock. Iteration follows sorted
/// protobuf map keys and then instruction order, matching the full-snapshot
/// admission walk.
pub fn validate_threadblock_contract(
    threadblock: &pb::Threadblock,
    label: impl AsRef<str>,
    registry: Option<&AllocationRegistry>,
    payload_contract: MemoryPayloadContract,
    expected_function_unique_id: i32,
) -> Result<(), TraceError> {
    let label = label.as_ref();
    for (&warp_id, warp) in &threadblock.warps {
        if warp.id != warp_id {
            return Err(TraceError::WarpMapKeyMismatch {
                threadblock: label.to_owned(),
                map_key: warp_id,
                warp_id: warp.id,
            });
        }
        let mut previous_instrumentation_order = None;
        for (instruction_index, instruction) in warp.instructions.iter().enumerate() {
            let record = || format!("{label} warp={warp_id} inst={instruction_index}");
            let preflight = preflight_memory_access(instruction, record)?;
            let Some(instruction_warp_id) = instruction.warpid_tb else {
                return Err(TraceError::WarpidTbMissing { record: record() });
            };
            if instruction_warp_id != warp_id {
                return Err(TraceError::WarpidTbMismatch {
                    record: record(),
                    expected: warp_id,
                    actual: instruction_warp_id,
                });
            }
            if instruction.function_unique_id != expected_function_unique_id {
                return Err(TraceError::FunctionUniqueIdMismatch {
                    record: record(),
                    expected: expected_function_unique_id,
                    actual: instruction.function_unique_id,
                });
            }
            if let Some(current) = instruction.instrumentation_order {
                if current == 0 {
                    return Err(TraceError::InstrumentationOrderZero { record: record() });
                }
                for (present, identity) in [
                    (instruction.sm_id.is_some(), "sm_id"),
                    (instruction.warpid_sm.is_some(), "warpid_sm"),
                ] {
                    if !present {
                        return Err(TraceError::InstrumentationOrderIdentityMissing {
                            record: record(),
                            identity,
                        });
                    }
                }
                if let Some(previous) = previous_instrumentation_order
                    && current <= previous
                {
                    return Err(TraceError::InstrumentationOrderNotIncreasing {
                        record: record(),
                        previous,
                        current,
                    });
                }
                previous_instrumentation_order = Some(current);
            }
            let Some(preflight) = preflight else {
                continue;
            };
            if matches!(
                payload_contract,
                MemoryPayloadContract::AddressCarryingV1
                    | MemoryPayloadContract::AddressCarryingSharedMemorySplitV2
                    | MemoryPayloadContract::AddressCarryingLdgstsDestinationV3
            ) {
                validate_address_carrying_shape(preflight, payload_contract, &record)?;
            }
            check_preflighted_allocation_identity(preflight, registry, record)?;
        }
    }
    Ok(())
}

/// Apply allocation identity to an instruction whose complete wire structure was
/// already validated. This split lets replay enforce its producer-shape route
/// before identity without rerunning fallible structural preflight or allocating an
/// intermediate decoded-payload vector; immutable address records are revisited by
/// the allocation-free visitor.
pub fn check_preflighted_allocation_identity(
    preflight: PreflightedMemoryAccess<'_>,
    registry: Option<&AllocationRegistry>,
    record: impl Fn() -> String,
) -> Result<(), TraceError> {
    let meta = preflight.meta();
    let inst_class = Some(meta.op_class);
    for addr in &preflight.instruction().addresses {
        let address_space = match addr.address_space {
            None => None,
            Some(raw) => Some(known_enum::<pb::MemorySpace>(
                raw,
                "address_space",
                &record,
            )?),
        };
        if !record_requires_identity(address_space, inst_class) {
            continue;
        }

        let access = decode_address_access(addr, meta.and_mask, preflight.inst_width, &record)?;
        let allocation_id = addr.allocation_id.unwrap_or(0);
        // Module/library-owned globals need not appear in cuMemAlloc hooks.
        // Honor the producer's explicit disposition and keep the recorded VA;
        // it is not proof of symbol bounds or lifetime, and must never become
        // a fabricated allocator id. Unannotated or unresolved_global records
        // still fail closed. No scored NCU information enters this decision.
        let module_global = match addr.allocation_identity_status.as_deref() {
            None | Some("resolved") => false,
            Some("unresolved_module_global") => true,
            Some(status) => {
                return Err(TraceError::AllocationIdentityStatusInvalid {
                    record: record(),
                    status: status.to_owned(),
                });
            }
        };
        if module_global {
            if allocation_id != 0 {
                return Err(TraceError::AllocationIdentityStatusConflict {
                    record: record(),
                    allocation_id,
                });
            }
            if access.lane_mask != 0 {
                continue;
            }
        }
        if allocation_id == 0 {
            if access.lane_mask != 0 {
                return Err(TraceError::AllocationIdMissing { record: record() });
            }
            continue;
        }
        let Some(registry) = registry else {
            return Err(TraceError::AllocationsSidecarMissing { record: record() });
        };
        let Some(range) = registry.by_id.get(&allocation_id) else {
            return Err(TraceError::AllocationIdUnknown {
                record: record(),
                allocation_id,
            });
        };
        if access.lane_mask == 0 {
            continue;
        }

        let base = u128::from(range.base);
        let end = base + u128::from(range.size);
        for (lane, &slot) in access.lane_vector.iter().enumerate() {
            if slot == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let lane_addr = slot as u64;
            let lane_addr_wide = u128::from(lane_addr);
            if !(base <= lane_addr_wide && lane_addr_wide < end) {
                return Err(TraceError::LaneOutsideAllocationRange {
                    record: record(),
                    lane,
                    lane_addr,
                    allocation_id,
                    base: range.base,
                    end,
                });
            }
        }
    }
    Ok(())
}

/// Allocating convenience wrapper over [`visit_memory_access`]: collects the
/// streamed records into a `Vec`. Identical contract and typed errors —
/// implemented on top of the streaming core so the two forms cannot drift.
pub fn memory_access(
    inst: &pb::Instruction,
    record: impl Fn() -> String,
) -> Result<Option<MemoryAccess>, TraceError> {
    let mut accesses = Vec::new();
    let meta = visit_memory_access(inst, record, |a| accesses.push(*a))?;
    Ok(meta.map(|m| MemoryAccess {
        op_class: m.op_class,
        access_kind: m.access_kind,
        cache_op: m.cache_op,
        and_mask: m.and_mask,
        accesses,
    }))
}

