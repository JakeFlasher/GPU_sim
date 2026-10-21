//! Compact validated trace records and memory routes; report-only immutable input.

use super::*;

/// Dynamic memory route proven from the validated instruction class and operand
/// spaces. Static SASS routing is still checked by `sim-sm` during replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatedMemoryRoute {
    Global,
    Local,
    Shared,
    Unsupported,
}

/// Compact, context-free form of dynamic operand-route failures. Replay adds
/// path/record context and preserves its established typed diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatedMemoryRouteError {
    GlobalOperandCount {
        got: usize,
    },
    GlobalSpaceInvalid {
        operand: usize,
        space: pb::MemorySpace,
    },
    SharedLocalOperandCount {
        got: usize,
    },
    SharedLocalSpaceInvalid {
        operand: usize,
        space: pb::MemorySpace,
    },
    AtomicOperandCount {
        got: usize,
    },
    AtomicSpaceMissing {
        operand: usize,
    },
    AtomicSpaceUnspecified {
        operand: usize,
    },
    AtomicSpacesConflict,
    AsyncCopySourceCount {
        got: usize,
    },
    AsyncCopySourceSpaceMissing,
    AsyncCopySourceSpaceInvalid {
        space: pb::MemorySpace,
    },
    AsyncCopyOperandCount {
        got: usize,
        expected: usize,
    },
    AsyncCopyDestinationSpaceMissing,
    AsyncCopyDestinationSpaceInvalid {
        space: pb::MemorySpace,
    },
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CompactMemoryRoute(pub(super) u32);

impl CompactMemoryRoute {
    pub(super) const GLOBAL: u32 = 0;
    pub(super) const ERROR_BASE: u32 = 4;

    pub(super) const fn ready(route: ValidatedMemoryRoute) -> Self {
        Self(match route {
            ValidatedMemoryRoute::Global => Self::GLOBAL,
            ValidatedMemoryRoute::Local => 1,
            ValidatedMemoryRoute::Shared => 2,
            ValidatedMemoryRoute::Unsupported => 3,
        })
    }

    pub(super) fn error(index: usize) -> Self {
        let index = u32::try_from(index)
            .unwrap_or_else(|_| unreachable!("validated route error arena exceeded u32"));
        Self(
            index
                .checked_add(Self::ERROR_BASE)
                .unwrap_or_else(|| unreachable!("validated route error index overflowed")),
        )
    }
}

/// Packed memory-route and protobuf-enum metadata. The wire preflight limits one
/// threadblock to 3,000,000 instructions, so a route-error arena index plus the
/// four ready-route discriminants fits in 22 bits. Packing the two small closed
/// enums into the remaining bits keeps each compact memory record within 24
/// bytes; exact u64 async payloads live in the sparse owning-warp arena.
#[derive(Debug, Clone, Copy)]
pub(super) struct CompactMemoryMeta(pub(super) u32);

impl CompactMemoryMeta {
    pub(super) const ROUTE_BITS: u32 = 22;
    pub(super) const ROUTE_MASK: u32 = (1 << Self::ROUTE_BITS) - 1;
    pub(super) const CLASS_SHIFT: u32 = Self::ROUTE_BITS;
    pub(super) const CLASS_MASK: u32 = 0xf;
    pub(super) const ACCESS_SHIFT: u32 = Self::CLASS_SHIFT + 4;
    pub(super) const ACCESS_MASK: u32 = 0x7;
    pub(super) const EFFECTIVE_SHIFT: u32 = Self::ACCESS_SHIFT + 3;

    pub(super) fn new(
        op_class: pb::MemoryOpClass,
        access_kind: Option<pb::MemoryOpAccessKind>,
        route: CompactMemoryRoute,
        has_effective_lanes: bool,
    ) -> Self {
        if route.0 > Self::ROUTE_MASK {
            unreachable!("bounded route-error arena exceeded compact encoding");
        }
        let access = access_kind.map_or(0, |kind| kind as u32 + 1);
        debug_assert!((op_class as u32) <= Self::CLASS_MASK);
        debug_assert!(access <= Self::ACCESS_MASK);
        Self(
            route.0
                | ((op_class as u32) << Self::CLASS_SHIFT)
                | (access << Self::ACCESS_SHIFT)
                | (u32::from(has_effective_lanes) << Self::EFFECTIVE_SHIFT),
        )
    }

    pub(super) const fn route(self) -> CompactMemoryRoute {
        CompactMemoryRoute(self.0 & Self::ROUTE_MASK)
    }

    pub(super) fn op_class(self) -> pb::MemoryOpClass {
        let encoded = ((self.0 >> Self::CLASS_SHIFT) & Self::CLASS_MASK) as i32;
        pb::MemoryOpClass::try_from(encoded)
            .unwrap_or_else(|_| unreachable!("validated memory class escaped compact encoding"))
    }

    pub(super) fn access_kind(self) -> Option<pb::MemoryOpAccessKind> {
        let encoded = (self.0 >> Self::ACCESS_SHIFT) & Self::ACCESS_MASK;
        if encoded == 0 {
            return None;
        }
        pb::MemoryOpAccessKind::try_from((encoded - 1) as i32)
            .map(Some)
            .unwrap_or_else(|_| unreachable!("validated access kind escaped compact encoding"))
    }

    pub(super) const fn has_effective_lanes(self) -> bool {
        self.0 & (1 << Self::EFFECTIVE_SHIFT) != 0
    }
}

/// Stage at which deferred sector expansion would have failed in the replay
/// classifier. Retaining the stage preserves its diagnostic precedence relative
/// to static SASS role checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatedSectorErrorPhase {
    BeforeStaticRole,
    AfterStaticRole,
}

/// Compact, context-free form of a sector-expansion failure. Replay restores the
/// path/warp/instruction context when this deferred error is observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidatedSectorError {
    DataWidthInvalid {
        data_width_bytes: i32,
    },
    AddressOverflow {
        lane: usize,
        base: i64,
        data_width_bytes: i32,
    },
    SectorCountOverflow {
        sectors: usize,
    },
    UniqueSectorCountOutOfRange {
        sectors: u128,
    },
    CardinalityMismatch {
        expected: usize,
        produced: usize,
    },
    CapacityUnavailable {
        buffer: &'static str,
        additional: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CompactSectorPlan {
    pub(super) offset_or_error: u32,
    pub(super) len: u16,
    pub(super) error_phase: u8,
}

impl CompactSectorPlan {
    pub(super) const READY: u8 = 0;
    pub(super) const BEFORE_STATIC_ROLE: u8 = 1;
    pub(super) const AFTER_STATIC_ROLE: u8 = 2;

    pub(super) fn ready(start: usize, len: u16) -> Self {
        Self {
            offset_or_error: u32::try_from(start)
                .unwrap_or_else(|_| unreachable!("validated sector arena exceeded u32")),
            len,
            error_phase: Self::READY,
        }
    }

    pub(super) fn error(phase: ValidatedSectorErrorPhase, index: usize) -> Self {
        Self {
            offset_or_error: u32::try_from(index)
                .unwrap_or_else(|_| unreachable!("validated sector error arena exceeded u32")),
            len: 0,
            error_phase: match phase {
                ValidatedSectorErrorPhase::BeforeStaticRole => Self::BEFORE_STATIC_ROLE,
                ValidatedSectorErrorPhase::AfterStaticRole => Self::AFTER_STATIC_ROLE,
            },
        }
    }
}

/// Borrowed sector payload or deferred expansion failure for one validated
/// memory instruction.
#[derive(Debug, Clone, Copy)]
pub enum ValidatedSectorPlan<'a> {
    Sectors(&'a [i64]),
    Error {
        phase: ValidatedSectorErrorPhase,
        error: &'a ValidatedSectorError,
    },
}

/// Compact memory metadata retained after the full Prost tree has been
/// validated and discarded. `projection` is route-disjoint: it stores a shared
/// bank-wavefront count for Shared records, an index into the owning warp's
/// sparse async-copy projection arena for Global async-copy records, or the
/// already-validated allocation id for address-carrying Global atomics.
pub struct ValidatedMemoryInstruction {
    pub(super) sectors: CompactSectorPlan,
    /// Diagnostic-only local sector span after reconstructing the physical
    /// warp-lane interleave of local-memory offsets. The replay sector plan
    /// intentionally remains the legacy logical-offset projection until the
    /// observer is independently admitted.
    pub(super) local_interleaved: CompactSectorPlan,
    pub(super) local_interleaved_dynamic: bool,
    pub(super) projection: u32,
    pub(super) partial_store_sectors: u64,
    /// Ideal 32-bank/4-byte-word shared service work for this instruction;
    /// zero for non-shared routes.  This is retained separately from the
    /// route-disjoint `projection` so the effective bank-conflict report can
    /// subtract unavoidable wide-access wavefronts rather than one request.
    pub(super) shared_ideal_wavefronts: u32,
    /// Whole-warp projection retained for SM scheduler timing.
    pub(super) shared_legacy_wavefronts: u32,
    /// Whole-warp width-only ideal retained for scheduler accounting.
    pub(super) shared_legacy_ideal_wavefronts: u32,
    /// Exact whole-warp A-branch observer retained for SM86 compatibility.
    pub(super) shared_sm86_wavefronts: u32,
    /// Observer-only local-memory sector cardinality after applying the
    /// effective SM86 lane-private frame stride. The regular `sectors` plan is
    /// retained unchanged for replay state/timing; this field exists solely to
    /// expose a source-derived local endpoint when compressed local offsets
    /// would otherwise collapse lanes onto one stack frame.
    pub(super) local_canonical_sectors: u16,
    /// Observer-only local endpoint using the 256-byte SM86 stack-frame
    /// allocation visible in the captured SASS prologue.  It is kept beside
    /// the transfer-width candidate so the emitter can form a bounded
    /// coalescing correction for dynamic-index local accesses.
    pub(super) local_frame256_sectors: u16,
    /// Observer-only witness that all active lanes use one byte address and
    /// width for this memory instruction.
    pub(super) warp_uniform_address: bool,

    pub(super) meta: CompactMemoryMeta,
    /// Zero means absent; otherwise this is wire enum value + 1. Keeping the
    /// tiny closed enum in one byte preserves the compact metadata bound.
    pub(super) cache_op: u8,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AsyncCopyProjection {
    pub(super) effective_data_bytes: u64,
    pub(super) destination_topology: CompactLdgstsDestinationTopology,
    pub(super) cohort_span: CompactArenaSpan,
    pub(super) source_sector_mask_span: CompactArenaSpan,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct CompactArenaSpan {
    pub(super) start: u32,
    pub(super) len: u16,
}

impl CompactArenaSpan {
    pub(super) fn new(start: usize, len: usize) -> Self {
        Self {
            start: u32::try_from(start)
                .unwrap_or_else(|_| unreachable!("validated sparse arena exceeded u32")),
            len: u16::try_from(len)
                .unwrap_or_else(|_| unreachable!("validated sparse span exceeded u16")),
        }
    }

    pub(super) fn slice<T>(self, arena: &[T]) -> &[T] {
        let start = usize::try_from(self.start)
            .unwrap_or_else(|_| unreachable!("validated sparse span escaped usize"));
        let end = start
            .checked_add(usize::from(self.len))
            .unwrap_or_else(|| unreachable!("validated sparse span overflowed"));
        arena
            .get(start..end)
            .unwrap_or_else(|| unreachable!("validated sparse span escaped its arena"))
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CompactLdgstsDestinationTopology {
    pub(super) source_cohorts: u8,
    pub(super) destination_line_fanout_total: u8,
    pub(super) destination_line_fanout_max: u8,
    pub(super) cache_all_bank_wavefronts_total: u8,
    pub(super) cache_all_bank_wavefronts_max: u8,
    pub(super) bypass_wavefronts_total: u8,
    pub(super) bypass_wavefronts_max: u8,
}

impl CompactLdgstsDestinationTopology {
    pub(super) const UNAVAILABLE: u8 = u8::MAX;

    pub(super) fn new(
        topology: Option<LdgstsDestinationTopology>,
    ) -> Result<Self, LdgstsDestinationProjectionError> {
        let Some(topology) = topology else {
            return Ok(Self {
                source_cohorts: Self::UNAVAILABLE,
                destination_line_fanout_total: 0,
                destination_line_fanout_max: 0,
                cache_all_bank_wavefronts_total: 0,
                cache_all_bank_wavefronts_max: 0,
                bypass_wavefronts_total: 0,
                bypass_wavefronts_max: 0,
            });
        };
        let compact = |field: &'static str, value: u32| {
            if value >= u32::from(Self::UNAVAILABLE) {
                return Err(
                    LdgstsDestinationProjectionError::CompactTopologyFieldOutOfRange {
                        field,
                        value,
                        limit: Self::UNAVAILABLE - 1,
                    },
                );
            }
            Ok(value as u8)
        };
        Ok(Self {
            source_cohorts: compact("source_cohorts", topology.source_cohorts)?,
            destination_line_fanout_total: compact(
                "destination_line_fanout_total",
                topology.destination_line_fanout_total,
            )?,
            destination_line_fanout_max: compact(
                "destination_line_fanout_max",
                topology.destination_line_fanout_max,
            )?,
            cache_all_bank_wavefronts_total: compact(
                "cache_all_bank_wavefronts_total",
                topology.cache_all_bank_wavefronts_total,
            )?,
            cache_all_bank_wavefronts_max: compact(
                "cache_all_bank_wavefronts_max",
                topology.cache_all_bank_wavefronts_max,
            )?,
            bypass_wavefronts_total: compact(
                "bypass_wavefronts_total",
                topology.bypass_wavefronts_total,
            )?,
            bypass_wavefronts_max: compact(
                "bypass_wavefronts_max",
                topology.bypass_wavefronts_max,
            )?,
        })
    }

    pub(super) const fn expand(self) -> Option<LdgstsDestinationTopology> {
        if self.source_cohorts == Self::UNAVAILABLE {
            return None;
        }
        Some(LdgstsDestinationTopology {
            source_cohorts: self.source_cohorts as u32,
            destination_line_fanout_total: self.destination_line_fanout_total as u32,
            destination_line_fanout_max: self.destination_line_fanout_max as u32,
            cache_all_bank_wavefronts_total: self.cache_all_bank_wavefronts_total as u32,
            cache_all_bank_wavefronts_max: self.cache_all_bank_wavefronts_max as u32,
            bypass_wavefronts_total: self.bypass_wavefronts_total as u32,
            bypass_wavefronts_max: self.bypass_wavefronts_max as u32,
        })
    }
}

impl AsyncCopyProjection {
    pub(super) fn new(
        effective_data_bytes: u64,
        destination_topology: Option<LdgstsDestinationTopology>,
        cohort_span: CompactArenaSpan,
        source_sector_mask_span: CompactArenaSpan,
    ) -> Result<Self, LdgstsDestinationProjectionError> {
        Ok(Self {
            effective_data_bytes,
            destination_topology: CompactLdgstsDestinationTopology::new(destination_topology)?,
            cohort_span,
            source_sector_mask_span,
        })
    }

    pub(super) const fn destination_topology(self) -> Option<LdgstsDestinationTopology> {
        self.destination_topology.expand()
    }
}

impl ValidatedMemoryInstruction {
    /// Reconstruct the metadata replay consumes. Cache operators are retained
    /// as wire observations; they are not interpreted as a timing policy here.
    pub fn replay_meta(&self) -> MemoryAccessMeta {
        MemoryAccessMeta {
            op_class: self.meta.op_class(),
            access_kind: self.meta.access_kind(),
            cache_op: if self.cache_op == 0 {
                None
            } else {
                Some(
                    pb::InstructionCacheOp::try_from(i32::from(self.cache_op - 1)).unwrap_or_else(
                        |_| unreachable!("validated cache operator escaped wire enum"),
                    ),
                )
            },
            and_mask: u32::from(self.meta.has_effective_lanes()),
        }
    }

    /// Whether at least one lane executes this dynamic memory instruction.
    pub const fn has_effective_lanes(&self) -> bool {
        self.meta.has_effective_lanes()
    }

    pub const fn local_interleaved_is_dynamic(&self) -> bool {
        self.local_interleaved_dynamic
    }

    /// Maximum distinct 4-byte words targeting one of 32 banks after collapsing
    /// broadcasts. Zero means the record had no effective shared address.
    /// Architecture timing must explicitly admit this named projection before
    /// interpreting it as hardware wavefronts. Callers resolve the record's
    /// Shared route before reading this route-disjoint projection.
    pub const fn shared_bank_wavefronts_32x4(&self) -> u32 {
        self.projection
    }

    /// Unavoidable wavefronts from per-lane access width, excluding bank
    /// conflicts.  A 4-byte access requires one wavefront; an 8-byte access
    /// requires two, even when every lane maps to a distinct bank.
    pub const fn shared_bank_ideal_wavefronts_32x4(&self) -> u32 {
        self.shared_ideal_wavefronts
    }

    /// Whole-warp shared service projection used by the SM scheduler. The
    /// phase-local observer projection is intentionally separate.
    pub const fn shared_bank_legacy_wavefronts_32x4(&self) -> u32 {
        self.shared_legacy_wavefronts
    }

    /// Whole-warp width-only ideal used by the scheduler-side projection.
    pub const fn shared_bank_legacy_ideal_wavefronts_32x4(&self) -> u32 {
        self.shared_legacy_ideal_wavefronts
    }

    pub const fn shared_sm86_wavefronts_32x4(&self) -> u32 {
        self.shared_sm86_wavefronts
    }

    /// Backing allocation identity for an address-carrying global atomic.
    ///
    /// The full reader has already required this id to resolve to the
    /// allocation registry and contain every effective lane address. Zero is
    /// retained only for predicated-off or legacy/non-atomic records and is
    /// exposed as `None`. This is timing-neutral identity: it does not assign a
    /// physical LTS class.
    pub fn global_atomic_allocation_id(&self) -> Option<u32> {
        (self.meta.op_class() == pb::MemoryOpClass::Atomic
            && self.meta.route().0 == CompactMemoryRoute::GLOBAL
            && self.projection != 0)
            .then_some(self.projection)
    }

    /// Trace-derived local endpoint candidate. It never feeds L1/L2 state or
    /// timing and is zero for non-local/unsupported records.
    pub const fn local_canonical_sectors(&self) -> u16 {
        self.local_canonical_sectors
    }

    /// Trace-derived local endpoint candidate using the captured 256-byte
    /// stack-frame stride. It never feeds cache state or timing.
    pub const fn local_frame256_sectors(&self) -> u16 {
        self.local_frame256_sectors
    }

    pub const fn warp_uniform_address(&self) -> bool {
        self.warp_uniform_address
    }
}

/// Compact immutable threadblock produced by the full payload validation walk.
pub struct ValidatedThreadblock {
    pub(super) launch_index: i64,
    pub(super) block_id: [u32; 3],
    pub(super) warps: Vec<ValidatedWarp>,
}

impl ValidatedThreadblock {
    pub(super) fn from_message(
        launch_index: i64,
        threadblock: pb::Threadblock,
        payload_contract: MemoryPayloadContract,
        label: &str,
    ) -> Result<Self, TraceError> {
        let warps = threadblock
            .warps
            .into_iter()
            .map(|(map_key, warp)| {
                ValidatedWarp::from_message(map_key, warp, payload_contract, label)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let block_id = threadblock
            .block_id
            .as_ref()
            .map(|id| [id.x as u32, id.y as u32, id.z as u32])
            .unwrap_or([0, 0, 0]);
        Ok(Self {
            launch_index,
            block_id,
            warps,
        })
    }

    pub const fn launch_index(&self) -> i64 {
        self.launch_index
    }

    pub const fn block_id(&self) -> [u32; 3] {
        self.block_id
    }

    pub fn warps(&self) -> &[ValidatedWarp] {
        &self.warps
    }
}

/// One validated warp in protobuf map-key order.
pub struct ValidatedWarp {
    pub(super) map_key: i32,
    pub(super) warp_id: i32,
    pub(super) function_unique_id: i32,
    pub(super) instructions: Vec<ValidatedInstruction>,
    pub(super) predicate_on_masks: Vec<u32>,
    pub(super) instrumentation_order_anchors: Vec<CompactInstrumentationOrderAnchor>,
    pub(super) memories: Vec<ValidatedMemoryInstruction>,
    /// Dynamic constant-bank offsets retained as a bounded, launch-local
    /// demand observer. These are not global allocation addresses and never
    /// participate in cache state or timing.
    pub(super) constant_memory_addresses: Vec<u64>,
    pub(super) async_copy_projections: Vec<AsyncCopyProjection>,
    pub(super) ldgsts_source_cohorts: Vec<LdgstsSourceCohort>,
    pub(super) ldgsts_source_sector_cohort_masks: Vec<u32>,
    pub(super) route_errors: Vec<ValidatedMemoryRouteError>,
    pub(super) sectors: Vec<i64>,
    pub(super) local_interleaved_sectors: Vec<i64>,
    pub(super) sector_errors: Vec<ValidatedSectorError>,
}

impl ValidatedWarp {
    pub const fn function_unique_id(&self) -> i32 {
        self.function_unique_id
    }

    pub(super) fn from_message(
        map_key: i32,
        warp: pb::Warp,
        payload_contract: MemoryPayloadContract,
        label: &str,
    ) -> Result<Self, TraceError> {
        let function_unique_id = warp
            .instructions
            .first()
            .map_or(0, |instruction| instruction.function_unique_id);
        let mut instructions = Vec::with_capacity(warp.instructions.len());
        let mut predicate_on_masks = Vec::with_capacity(warp.instructions.len());
        let mut instrumentation_order_anchors = Vec::new();
        let mut memories = Vec::new();
        let mut constant_memory_addresses = Vec::new();
        let mut async_copy_projections = Vec::new();
        let mut ldgsts_source_cohorts = Vec::new();
        let mut ldgsts_source_sector_cohort_masks = Vec::new();
        let mut route_errors = Vec::new();
        let mut sectors = Vec::new();
        let mut local_interleaved_sectors = Vec::new();
        let mut sector_errors = Vec::new();
        let mut shared_word_scratch = Vec::new();
        for (record_ix, instruction) in warp.instructions.into_iter().enumerate() {
            let constant_memory_span = CompactArenaSpan::new(
                constant_memory_addresses.len(),
                instruction.constant_memory_addresses.len(),
            );
            constant_memory_addresses.extend_from_slice(&instruction.constant_memory_addresses);
            if let Some(order) = instruction.instrumentation_order {
                instrumentation_order_anchors.push(CompactInstrumentationOrderAnchor {
                    order,
                    instruction_index: u32::try_from(record_ix).unwrap_or_else(|_| {
                        unreachable!("validated instruction index exceeded u32")
                    }),
                });
            }
            let memory_index = if instruction.memory_op_class.is_some() {
                let no_error_context = || String::new();
                let preflight = match preflight_memory_access(&instruction, no_error_context) {
                    Ok(Some(preflight)) => preflight,
                    Ok(None) | Err(_) => {
                        unreachable!("validated memory instruction failed compact preflight")
                    }
                };
                let meta = preflight.meta();
                let resolved_route = compact_memory_route(
                    &instruction,
                    meta.op_class,
                    payload_contract,
                    meta.and_mask,
                );
                let shared_bank_wavefronts_32x4 =
                    if matches!(resolved_route, Ok(ValidatedMemoryRoute::Shared)) {
                        project_shared_bank_wavefronts_32x4(preflight, &mut shared_word_scratch)
                            .map_err(|source| TraceError::SharedBankProjection {
                                record: format!("{label} warp={map_key} inst={record_ix}"),
                                source,
                            })?
                    } else {
                        0
                    };
                let shared_sm86_wavefronts_32x4 =
                    if matches!(resolved_route, Ok(ValidatedMemoryRoute::Shared)) {
                        project_shared_bank_wavefronts_32x4_sm86(
                            preflight,
                            &mut shared_word_scratch,
                        )
                        .map_err(|source| {
                            TraceError::SharedBankProjection {
                                record: format!("{label} warp={map_key} inst={record_ix}"),
                                source,
                            }
                        })?
                    } else {
                        0
                    };
                let shared_legacy_bank_wavefronts_32x4 =
                    if matches!(resolved_route, Ok(ValidatedMemoryRoute::Shared)) {
                        project_shared_bank_wavefronts_32x4_legacy(
                            preflight,
                            &mut shared_word_scratch,
                        )
                        .map_err(|source| {
                            TraceError::SharedBankProjection {
                                record: format!("{label} warp={map_key} inst={record_ix}"),
                                source,
                            }
                        })?
                    } else {
                        0
                    };
                let shared_bank_ideal_wavefronts_32x4 =
                    if matches!(resolved_route, Ok(ValidatedMemoryRoute::Shared)) {
                        project_shared_bank_ideal_wavefronts_32x4(preflight).map_err(|source| {
                            TraceError::SharedBankProjection {
                                record: format!("{label} warp={map_key} inst={record_ix}"),
                                source,
                            }
                        })?
                    } else {
                        0
                    };
                let shared_legacy_bank_ideal_wavefronts_32x4 =
                    if matches!(resolved_route, Ok(ValidatedMemoryRoute::Shared)) {
                        project_shared_bank_ideal_wavefronts_32x4_legacy(preflight).map_err(
                            |source| TraceError::SharedBankProjection {
                                record: format!("{label} warp={map_key} inst={record_ix}"),
                                source,
                            },
                        )?
                    } else {
                        0
                    };
                let is_async_copy_source = meta.op_class == pb::MemoryOpClass::AsyncCopy
                    && matches!(resolved_route, Ok(ValidatedMemoryRoute::Global));
                let effective_data_bytes = if is_async_copy_source {
                    if payload_contract.carries_ldgsts_destination() {
                        project_operand_effective_data_bytes(preflight, 0)
                    } else {
                        project_effective_data_bytes(preflight)
                    }
                    .map_err(|source| TraceError::EffectiveDataBytes {
                        record: format!("{label} warp={map_key} inst={record_ix}"),
                        source,
                    })?
                } else {
                    0
                };
                let global_atomic_transactions = meta.op_class == pb::MemoryOpClass::Atomic
                    && matches!(resolved_route, Ok(ValidatedMemoryRoute::Global))
                    && payload_contract.carries_atomic_addresses();
                let local_interleaved = if matches!(resolved_route, Ok(ValidatedMemoryRoute::Local))
                {
                    match compact_sector_vector_for_operand(preflight, None, true) {
                        Ok(staged) => {
                            let start = local_interleaved_sectors.len();
                            let len = u16::try_from(staged.len()).unwrap_or(0);
                            if usize::from(len) == staged.len() {
                                local_interleaved_sectors.extend_from_slice(&staged);
                                CompactSectorPlan::ready(start, len)
                            } else {
                                CompactSectorPlan::ready(local_interleaved_sectors.len(), 0)
                            }
                        }
                        Err(_) => CompactSectorPlan::ready(local_interleaved_sectors.len(), 0),
                    }
                } else {
                    CompactSectorPlan::ready(local_interleaved_sectors.len(), 0)
                };
                let local_interleaved_dynamic =
                    matches!(resolved_route, Ok(ValidatedMemoryRoute::Local))
                        && preflight
                            .instruction()
                            .addresses
                            .iter()
                            .any(|address| address.compression_format == lanes::FORMAT_BASE_DELTA);
                let sector_projection = if matches!(
                    resolved_route,
                    Ok(ValidatedMemoryRoute::Global | ValidatedMemoryRoute::Local)
                ) {
                    Some(if global_atomic_transactions {
                        compact_atomic_sector_transaction_vector_32b(preflight)
                    } else {
                        compact_sector_vector_for_operand(
                            preflight,
                            (is_async_copy_source && payload_contract.carries_ldgsts_destination())
                                .then_some(0),
                            false,
                        )
                    })
                } else {
                    None
                };
                let destination_projection =
                    if is_async_copy_source && payload_contract.carries_ldgsts_destination() {
                        let source_sectors = sector_projection
                            .as_ref()
                            .and_then(|projection| projection.as_ref().ok())
                            .map_or(&[][..], Vec::as_slice);
                        Some(
                            project_ldgsts_source_cohorts_128b_32x4(
                                preflight,
                                source_sectors,
                                &mut shared_word_scratch,
                            )
                            .map_err(|source| {
                                TraceError::LdgstsDestinationProjection {
                                    record: format!("{label} warp={map_key} inst={record_ix}"),
                                    source,
                                }
                            })?,
                        )
                    } else {
                        None
                    };
                let projection = if global_atomic_transactions {
                    preflight
                        .instruction()
                        .addresses
                        .first()
                        .and_then(|address| address.allocation_id)
                        .filter(|allocation_id| *allocation_id != 0)
                        .unwrap_or(0)
                } else if is_async_copy_source {
                    let index = match u32::try_from(async_copy_projections.len()) {
                        Ok(index) if index != u32::MAX => index,
                        Ok(_) | Err(_) => {
                            unreachable!("validated async-copy projection arena exceeded u32")
                        }
                    };
                    let cohort_span = CompactArenaSpan::new(
                        ldgsts_source_cohorts.len(),
                        destination_projection
                            .as_ref()
                            .map_or(0, |projection| projection.cohorts.len()),
                    );
                    let source_sector_mask_span = CompactArenaSpan::new(
                        ldgsts_source_sector_cohort_masks.len(),
                        destination_projection
                            .as_ref()
                            .map_or(0, |projection| projection.source_sector_cohort_masks.len()),
                    );
                    let destination_topology = destination_projection
                        .as_ref()
                        .map(|projection| projection.topology);
                    if let Some(destination_projection) = destination_projection {
                        ldgsts_source_cohorts.extend(destination_projection.cohorts);
                        ldgsts_source_sector_cohort_masks
                            .extend(destination_projection.source_sector_cohort_masks);
                    }
                    async_copy_projections.push(
                        AsyncCopyProjection::new(
                            effective_data_bytes,
                            destination_topology,
                            cohort_span,
                            source_sector_mask_span,
                        )
                        .map_err(|source| {
                            TraceError::LdgstsDestinationProjection {
                                record: format!("{label} warp={map_key} inst={record_ix}"),
                                source,
                            }
                        })?,
                    );
                    index
                } else {
                    shared_bank_wavefronts_32x4
                };
                let sector_plan = if let Some(sector_projection) = sector_projection {
                    match sector_projection {
                        Ok(staged) => {
                            let start = sectors.len();
                            let len = match u16::try_from(staged.len()) {
                                Ok(len) => len,
                                Err(_) => unreachable!("bounded sector vector exceeded u16"),
                            };
                            match sectors.try_reserve(staged.len()) {
                                Ok(()) => {
                                    sectors.extend_from_slice(&staged);
                                    CompactSectorPlan::ready(start, len)
                                }
                                Err(_) => {
                                    let index = sector_errors.len();
                                    sector_errors.push(ValidatedSectorError::CapacityUnavailable {
                                        buffer: "warp sector arena",
                                        additional: staged.len(),
                                    });
                                    CompactSectorPlan::error(
                                        ValidatedSectorErrorPhase::AfterStaticRole,
                                        index,
                                    )
                                }
                            }
                        }
                        Err((phase, error)) => {
                            let index = sector_errors.len();
                            sector_errors.push(error);
                            CompactSectorPlan::error(phase, index)
                        }
                    }
                } else {
                    CompactSectorPlan::ready(sectors.len(), 0)
                };
                let partial_store_sectors = if meta.access_kind
                    == Some(pb::MemoryOpAccessKind::Store)
                    && matches!(
                        resolved_route,
                        Ok(ValidatedMemoryRoute::Global | ValidatedMemoryRoute::Local)
                    ) {
                    project_partial_store_sectors(preflight).map_err(|source| {
                        TraceError::EffectiveDataBytes {
                            record: format!("{label} warp={map_key} inst={record_ix}"),
                            source,
                        }
                    })?
                } else {
                    0
                };
                let (local_canonical_sectors, local_frame256_sectors) =
                    if matches!(resolved_route, Ok(ValidatedMemoryRoute::Local)) {
                        (
                            local_canonical_sector_count(preflight, None),
                            local_canonical_sector_count(preflight, Some(256)),
                        )
                    } else {
                        (0, 0)
                    };
                let warp_uniform_address = warp_uniform_address(preflight);

                let route = match resolved_route {
                    Ok(route) => CompactMemoryRoute::ready(route),
                    Err(error) => {
                        let route = CompactMemoryRoute::error(route_errors.len());
                        route_errors.push(error);
                        route
                    }
                };
                let memory_index = match u32::try_from(memories.len()) {
                    Ok(index) if index != u32::MAX => index,
                    Ok(_) | Err(_) => unreachable!("validated memory arena exceeded u32"),
                };
                memories.push(ValidatedMemoryInstruction {
                    sectors: sector_plan,
                    local_interleaved,
                    local_interleaved_dynamic,
                    projection,
                    partial_store_sectors,
                    shared_ideal_wavefronts: shared_bank_ideal_wavefronts_32x4,
                    shared_legacy_wavefronts: shared_legacy_bank_wavefronts_32x4,
                    shared_legacy_ideal_wavefronts: shared_legacy_bank_ideal_wavefronts_32x4,
                    shared_sm86_wavefronts: shared_sm86_wavefronts_32x4,
                    local_canonical_sectors,
                    local_frame256_sectors,
                    warp_uniform_address,

                    meta: CompactMemoryMeta::new(
                        meta.op_class,
                        meta.access_kind,
                        route,
                        meta.and_mask != 0,
                    ),
                    cache_op: meta.cache_op.map_or(0, |cache_op| cache_op as u8 + 1),
                });
                memory_index
            } else {
                u32::MAX
            };
            let mut identity_presence = 0_u8;
            if instruction.sm_id.is_some() {
                identity_presence |= ValidatedInstruction::SM_ID_PRESENT;
            }
            if instruction.warpid_sm.is_some() {
                identity_presence |= ValidatedInstruction::WARP_SLOT_PRESENT;
            }
            let predicate_on_mask = instruction.active_mask.unwrap_or_default()
                & instruction.predicate_mask.unwrap_or_default();
            predicate_on_masks.push(predicate_on_mask);
            instructions.push(ValidatedInstruction {
                pc: instruction.pc,
                data_width: instruction.data_width,
                memory_index,
                constant_memory_span,
                constant_memory_bank: instruction.constant_memory_bank,
                constant_memory_offset: instruction.constant_memory_offset,
                sm_id: instruction.sm_id.unwrap_or_default(),
                warpid_sm: instruction.warpid_sm.unwrap_or_default(),
                identity_presence,
                active_mask_present: instruction.active_mask.is_some(),
                predicate_mask_present: instruction.predicate_mask.is_some(),
                active_mask: instruction.active_mask.unwrap_or_default(),
                predicate_on_lanes: predicate_on_mask.count_ones() as u8,
            });
        }
        Ok(Self {
            map_key,
            warp_id: warp.id,
            function_unique_id,
            instructions,
            predicate_on_masks,
            instrumentation_order_anchors,
            memories,
            constant_memory_addresses,
            async_copy_projections,
            ldgsts_source_cohorts,
            ldgsts_source_sector_cohort_masks,
            route_errors,
            sectors,
            local_interleaved_sectors,
            sector_errors,
        })
    }

    pub const fn map_key(&self) -> i32 {
        self.map_key
    }

    pub const fn warp_id(&self) -> i32 {
        self.warp_id
    }

    pub fn instructions(
        &self,
    ) -> impl ExactSizeIterator<Item = ValidatedInstructionRef<'_>> + DoubleEndedIterator + '_ {
        self.instructions
            .iter()
            .enumerate()
            .map(|(instruction_index, instruction)| {
                let memory = if instruction.memory_index == u32::MAX {
                    None
                } else {
                    let index = usize::try_from(instruction.memory_index)
                        .unwrap_or_else(|_| unreachable!("validated memory index escaped usize"));
                    Some(self.memories.get(index).unwrap_or_else(|| {
                        unreachable!("validated memory index escaped its arena")
                    }))
                };
                let instrumentation_order =
                    u32::try_from(instruction_index)
                        .ok()
                        .and_then(|instruction_index| {
                            self.instrumentation_order_anchors
                                .binary_search_by_key(&instruction_index, |anchor| {
                                    anchor.instruction_index
                                })
                                .ok()
                                .map(|index| self.instrumentation_order_anchors[index].order)
                        });
                let constant_memory_addresses = instruction
                    .constant_memory_span
                    .slice(&self.constant_memory_addresses);
                ValidatedInstructionRef {
                    instruction,
                    memory,
                    constant_memory_addresses,
                    function_unique_id: self.function_unique_id,
                    warpid_tb: self.map_key,
                    instruction_index: u32::try_from(instruction_index).unwrap_or_else(|_| {
                        unreachable!("validated instruction index exceeded u32")
                    }),
                    predicate_on_mask: self.predicate_on_masks[instruction_index],
                    instrumentation_order,
                }
            })
    }

    pub fn memory_route<'a>(
        &'a self,
        memory: &ValidatedMemoryInstruction,
    ) -> Result<ValidatedMemoryRoute, &'a ValidatedMemoryRouteError> {
        match memory.meta.route().0 {
            0 => Ok(ValidatedMemoryRoute::Global),
            1 => Ok(ValidatedMemoryRoute::Local),
            2 => Ok(ValidatedMemoryRoute::Shared),
            3 => Ok(ValidatedMemoryRoute::Unsupported),
            encoded => {
                let index = encoded
                    .checked_sub(CompactMemoryRoute::ERROR_BASE)
                    .unwrap_or_else(|| unreachable!("validated route encoding underflowed"));
                let index = usize::try_from(index)
                    .unwrap_or_else(|_| unreachable!("validated route error escaped usize"));
                Err(self
                    .route_errors
                    .get(index)
                    .unwrap_or_else(|| unreachable!("validated route error escaped its arena")))
            }
        }
    }

    pub(super) fn async_copy_projection(
        &self,
        memory: &ValidatedMemoryInstruction,
    ) -> Option<AsyncCopyProjection> {
        if memory.meta.op_class() != pb::MemoryOpClass::AsyncCopy
            || memory.meta.route().0 != CompactMemoryRoute::GLOBAL
        {
            return None;
        }
        let index = usize::try_from(memory.projection)
            .unwrap_or_else(|_| unreachable!("validated async-copy index escaped usize"));
        Some(*self.async_copy_projections.get(index).unwrap_or_else(|| {
            unreachable!("validated async-copy index escaped its projection arena")
        }))
    }

    /// Sum of active-lane source widths for one Global async-copy record. Exact
    /// u64 values live in a sparse warp-local arena so ordinary memory metadata
    /// remains compact. Other routes and operation classes return zero.
    pub fn effective_data_bytes(&self, memory: &ValidatedMemoryInstruction) -> u64 {
        self.async_copy_projection(memory)
            .map_or(0, |projection| projection.effective_data_bytes)
    }

    /// Potential partial-sector global-store RMW demand from decoded byte
    /// coverage. The hierarchy layer must intersect this with modeled cold
    /// store sectors before charging a lower-level read.
    pub fn partial_store_sectors(&self, memory: &ValidatedMemoryInstruction) -> u64 {
        memory.partial_store_sectors
    }

    /// Exact LDGSTS source/destination topology when the trace contract carries
    /// the lane-paired SHARED operand. `None` preserves the explicit source-only
    /// fallback boundary; `Some(default())` is an exact zero-active result.
    pub fn ldgsts_destination_topology_128b_32x4(
        &self,
        memory: &ValidatedMemoryInstruction,
    ) -> Option<LdgstsDestinationTopology> {
        self.async_copy_projection(memory)
            .and_then(AsyncCopyProjection::destination_topology)
    }

    /// Exact flexible 128-byte source-cohort descriptors for a destination-
    /// bearing LDGSTS record. `None` is the legacy source-only fallback;
    /// predicated-off destination-bearing records return `Some(&[])`.
    pub fn ldgsts_source_cohorts_128b_32x4(
        &self,
        memory: &ValidatedMemoryInstruction,
    ) -> Option<&[LdgstsSourceCohort]> {
        let projection = self.async_copy_projection(memory)?;
        projection.destination_topology()?;
        Some(projection.cohort_span.slice(&self.ldgsts_source_cohorts))
    }

    /// One cohort-membership bitmask per compact source sector, in the exact
    /// order returned by [`Self::sector_plan`]. A bit is set when that sector
    /// intersects source bytes assigned to the corresponding cohort; boundary
    /// sectors may therefore carry multiple bits. `None` preserves the legacy
    /// source-only fallback.
    pub fn ldgsts_source_sector_cohort_masks_128b_32x4(
        &self,
        memory: &ValidatedMemoryInstruction,
    ) -> Option<&[u32]> {
        let projection = self.async_copy_projection(memory)?;
        projection.destination_topology()?;
        Some(
            projection
                .source_sector_mask_span
                .slice(&self.ldgsts_source_sector_cohort_masks),
        )
    }

    /// Compatibility view of the established cache-all bank-wavefront total.
    /// New bypass work must consume the multidimensional topology above rather
    /// than treating this scalar as a transferable completion coordinate.
    pub fn ldgsts_destination_wavefronts_128b_32x4(
        &self,
        memory: &ValidatedMemoryInstruction,
    ) -> Option<u32> {
        self.ldgsts_destination_topology_128b_32x4(memory)
            .map(|topology| topology.cache_all_bank_wavefronts_total)
    }

    pub fn sector_plan<'a>(
        &'a self,
        memory: &ValidatedMemoryInstruction,
    ) -> ValidatedSectorPlan<'a> {
        match memory.sectors.error_phase {
            CompactSectorPlan::READY => {
                let start = usize::try_from(memory.sectors.offset_or_error).unwrap_or_else(|_| {
                    unreachable!("validated sector arena offset escaped usize")
                });
                let end = match start.checked_add(usize::from(memory.sectors.len)) {
                    Some(end) => end,
                    None => unreachable!("validated sector span overflowed"),
                };
                let Some(sectors) = self.sectors.get(start..end) else {
                    unreachable!("validated sector span escaped its arena")
                };
                ValidatedSectorPlan::Sectors(sectors)
            }
            code @ (CompactSectorPlan::BEFORE_STATIC_ROLE
            | CompactSectorPlan::AFTER_STATIC_ROLE) => {
                let index = usize::try_from(memory.sectors.offset_or_error)
                    .unwrap_or_else(|_| unreachable!("validated sector error index escaped usize"));
                let Some(error) = self.sector_errors.get(index) else {
                    unreachable!("validated sector error index escaped its arena")
                };
                let phase = if code == CompactSectorPlan::BEFORE_STATIC_ROLE {
                    ValidatedSectorErrorPhase::BeforeStaticRole
                } else {
                    ValidatedSectorErrorPhase::AfterStaticRole
                };
                ValidatedSectorPlan::Error { phase, error }
            }
            _ => unreachable!("validated sector plan carried an unknown phase"),
        }
    }

    pub fn local_interleaved_sector_plan<'a>(
        &'a self,
        memory: &ValidatedMemoryInstruction,
    ) -> &'a [i64] {
        if memory.local_interleaved.error_phase != CompactSectorPlan::READY {
            return &[];
        }
        let start = usize::try_from(memory.local_interleaved.offset_or_error)
            .unwrap_or_else(|_| unreachable!("local interleaved arena offset escaped usize"));
        let end = start
            .checked_add(usize::from(memory.local_interleaved.len))
            .unwrap_or_else(|| unreachable!("local interleaved span overflowed"));
        self.local_interleaved_sectors
            .get(start..end)
            .unwrap_or_else(|| unreachable!("local interleaved span escaped its arena"))
    }
}

/// Packed scheduler fields retained from one validated dynamic instruction.
pub(super) struct ValidatedInstruction {
    pub(super) pc: u32,
    pub(super) data_width: Option<i32>,
    pub(super) memory_index: u32,
    pub(super) constant_memory_span: CompactArenaSpan,
    pub(super) constant_memory_bank: Option<u32>,
    pub(super) constant_memory_offset: Option<u64>,
    pub(super) sm_id: i32,
    pub(super) warpid_sm: i32,
    pub(super) identity_presence: u8,
    pub(super) active_mask_present: bool,
    pub(super) predicate_mask_present: bool,
    // Preserve lane identity as well as population: per-lane successor PCs
    // cannot be reconstructed from popcounts when a warp splits/reconverges.
    pub(super) active_mask: u32,
    pub(super) predicate_on_lanes: u8,
}

/// Sparse order witness for one selected dynamic instruction. Keeping this in
/// a warp-local arena avoids widening every hot compact instruction when the
/// opt-in producer field is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CompactInstrumentationOrderAnchor {
    pub(super) order: u64,
    pub(super) instruction_index: u32,
}

impl ValidatedInstruction {
    pub(super) const SM_ID_PRESENT: u8 = 1 << 0;
    pub(super) const WARP_SLOT_PRESENT: u8 = 1 << 1;
}

/// One instruction borrowed together with the compact memory metadata from its
/// owning warp. The pairing prevents memory-arena indices from crossing warps.
#[derive(Clone, Copy)]
pub struct ValidatedInstructionRef<'a> {
    pub(super) instruction: &'a ValidatedInstruction,
    pub(super) memory: Option<&'a ValidatedMemoryInstruction>,
    pub(super) constant_memory_addresses: &'a [u64],
    pub(super) function_unique_id: i32,
    pub(super) warpid_tb: i32,
    pub(super) instruction_index: u32,
    pub(super) predicate_on_mask: u32,
    pub(super) instrumentation_order: Option<u64>,
}

impl<'a> ValidatedInstructionRef<'a> {
    /// Lane populations validated before compaction; no profiler values.
    pub const fn active_lanes(&self) -> u8 {
        self.instruction.active_mask.count_ones() as u8
    }
    pub const fn active_mask_present(&self) -> bool {
        self.instruction.active_mask_present
    }
    pub const fn predicate_mask_present(&self) -> bool {
        self.instruction.predicate_mask_present
    }
    pub const fn active_mask(&self) -> u32 {
        self.instruction.active_mask
    }
    pub const fn predicate_on_lanes(&self) -> u8 {
        self.instruction.predicate_on_lanes
    }

    pub const fn predicate_on_mask(&self) -> u32 {
        self.predicate_on_mask
    }

    pub const fn instruction_index(&self) -> u32 {
        self.instruction_index
    }

    pub const fn pc(&self) -> u32 {
        self.instruction.pc
    }

    pub const fn function_unique_id(&self) -> i32 {
        self.function_unique_id
    }

    pub const fn warpid_tb(&self) -> i32 {
        self.warpid_tb
    }

    pub const fn warpid_sm(&self) -> Option<i32> {
        if self.instruction.identity_presence & ValidatedInstruction::WARP_SLOT_PRESENT != 0 {
            Some(self.instruction.warpid_sm)
        } else {
            None
        }
    }

    pub const fn sm_id(&self) -> Option<i32> {
        if self.instruction.identity_presence & ValidatedInstruction::SM_ID_PRESENT != 0 {
            Some(self.instruction.sm_id)
        } else {
            None
        }
    }

    /// Existing callback/channel order attached by the opt-in NVBit producer.
    /// This is an observation token, never a hardware cycle or simulator time.
    pub const fn instrumentation_order(&self) -> Option<u64> {
        self.instrumentation_order
    }

    pub const fn memory(self) -> Option<&'a ValidatedMemoryInstruction> {
        self.memory
    }

    pub const fn data_width(self) -> Option<i32> {
        self.instruction.data_width
    }

    /// Dynamic constant-bank offsets observed by the producer. These are a
    /// demand-only trace projection; they are not device allocation addresses
    /// and must not be sent through the global cache/L2 path.
    pub const fn constant_memory_addresses(self) -> &'a [u64] {
        self.constant_memory_addresses
    }

    pub const fn constant_memory_bank_offset(self) -> Option<(u32, u64)> {
        match (
            self.instruction.constant_memory_bank,
            self.instruction.constant_memory_offset,
        ) {
            (Some(bank), Some(offset)) => Some((bank, offset)),
            _ => None,
        }
    }

    /// Whether either constant metadata field was present on the wire. A
    /// partial legacy record is retained as typed metadata and never treated
    /// as an active lane-demand observation.
    pub const fn constant_memory_metadata_present(self) -> bool {
        self.instruction.constant_memory_bank.is_some()
            || self.instruction.constant_memory_offset.is_some()
    }
}

pub(crate) struct ValidatedPayload {
    pub(super) block_index: i64,
    pub(super) records: u64,
    pub(super) block: ValidatedThreadblock,
}

/// Full-launch block-to-SM relation retained only after the sidecar parser proves
/// that every launch index occurs exactly once. This is spatial placement; it is
/// not evidence of temporal block co-residency.
#[cfg(feature = "campaign")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ValidatedPerSmSlotMap {
    pub(super) sm_by_launch_index: Box<[u64]>,
}

#[cfg(feature = "campaign")]
impl ValidatedPerSmSlotMap {
    pub(super) fn retain(
        roster_by_sm: &BTreeMap<u64, Vec<u64>>,
        expected_blocks: u64,
    ) -> Result<Self, String> {
        let mut placement_by_launch_index = BTreeMap::new();
        for (&sm_id, blocks) in roster_by_sm {
            for &block_id in blocks {
                if block_id >= expected_blocks {
                    return Err(format!(
                        "block {block_id} is outside retained per_sm_slot_map cardinality {expected_blocks}"
                    ));
                }
                if placement_by_launch_index.insert(block_id, sm_id).is_some() {
                    return Err(format!(
                        "block {block_id} occurs more than once in retained per_sm_slot_map"
                    ));
                }
            }
        }
        let actual_blocks = u64::try_from(placement_by_launch_index.len())
            .map_err(|_| "per_sm_slot_map cardinality exceeds u64".to_string())?;
        if actual_blocks != expected_blocks {
            let first_missing = placement_by_launch_index
                .keys()
                .copied()
                .zip(0_u64..)
                .find_map(|(actual, expected)| (actual != expected).then_some(expected))
                .unwrap_or(actual_blocks);
            return Err(format!(
                "launch index {first_missing} is absent from retained per_sm_slot_map"
            ));
        }
        Ok(Self {
            sm_by_launch_index: placement_by_launch_index
                .into_values()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }
}
