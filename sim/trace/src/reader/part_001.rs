// Lexical implementation section 2/5; included by the parent facade.
/// Failure while pairing one LDGSTS global-source transfer with its
/// lane-corresponding shared destination and reducing flexible 128-byte source
/// cohorts to destination-line and shared-bank topology.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum LdgstsDestinationProjectionError {
    #[error("expected exactly one GLOBAL/GENERIC source operand, got {got}")]
    SourceOperandCount { got: usize },
    #[error("expected exactly one SHARED destination operand, got {got}")]
    DestinationOperandCount { got: usize },
    #[error("lane {lane} has an active source but no corresponding shared destination")]
    DestinationLaneMissing { lane: usize },
    #[error(
        "source and destination operand widths differ: source={source_bytes} bytes, destination={destination_bytes} bytes"
    )]
    OperandWidthMismatch {
        source_bytes: i32,
        destination_bytes: i32,
    },
    #[error("lane {lane} source transfer width {width_bytes} exceeds one 128-byte cohort")]
    SourceTransferExceedsCohort { lane: usize, width_bytes: u64 },
    #[error(
        "lane {lane} source interval overflows the u64 address domain: base=0x{base:x}, width={width_bytes}"
    )]
    SourceIntervalOverflow {
        lane: usize,
        base: u64,
        width_bytes: u64,
    },
    #[error(
        "lane {lane} destination interval overflows the i64 address domain: base={base}, width={width_bytes}"
    )]
    DestinationIntervalOverflow {
        lane: usize,
        base: i64,
        width_bytes: u64,
    },
    #[error(
        "instruction projects {words} destination words, exceeding the per-instruction limit {limit}"
    )]
    InstructionWordLimit { words: u64, limit: u64 },
    #[error("destination projection scratch could not reserve {additional} additional words")]
    CapacityUnavailable { additional: usize },
    #[error("destination bank {bank} wavefront count overflowed u32")]
    BankWavefrontCountOverflow { bank: u8 },
    #[error("instruction destination wavefront sum overflowed u32")]
    WavefrontCountOverflow,
    #[error("instruction source-cohort count overflowed u32")]
    SourceCohortCountOverflow,
    #[error("instruction destination-line fanout sum overflowed u32")]
    DestinationLineFanoutOverflow,
    #[error(
        "cohort bypass projection overflowed u32: destination-line fanout {destination_lines} * bank wavefronts {bank_wavefronts}"
    )]
    BypassWavefrontProductOverflow {
        destination_lines: u32,
        bank_wavefronts: u32,
    },
    #[error("instruction bypass destination-wavefront sum overflowed u32")]
    BypassWavefrontCountOverflow,
    #[error("LDGSTS topology field {field}={value} exceeds compact limit {limit}")]
    CompactTopologyFieldOutOfRange {
        field: &'static str,
        value: u32,
        limit: u8,
    },
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
struct CompactSectorPlan {
    offset_or_error: u32,
    len: u16,
    error_phase: u8,
}

impl CompactSectorPlan {
    const READY: u8 = 0;
    const BEFORE_STATIC_ROLE: u8 = 1;
    const AFTER_STATIC_ROLE: u8 = 2;

    fn ready(start: usize, len: u16) -> Self {
        Self {
            offset_or_error: u32::try_from(start)
                .unwrap_or_else(|_| unreachable!("validated sector arena exceeded u32")),
            len,
            error_phase: Self::READY,
        }
    }

    fn error(phase: ValidatedSectorErrorPhase, index: usize) -> Self {
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
    sectors: CompactSectorPlan,
    /// Diagnostic-only local sector span after reconstructing the physical
    /// warp-lane interleave of local-memory offsets. The replay sector plan
    /// intentionally remains the legacy logical-offset projection until the
    /// observer is independently admitted.
    local_interleaved: CompactSectorPlan,
    local_interleaved_dynamic: bool,
    projection: u32,
    partial_store_sectors: u64,
    /// Ideal 32-bank/4-byte-word shared service work for this instruction;
    /// zero for non-shared routes.  This is retained separately from the
    /// route-disjoint `projection` so the effective bank-conflict report can
    /// subtract unavoidable wide-access wavefronts rather than one request.
    shared_ideal_wavefronts: u32,
    /// Whole-warp projection retained for SM scheduler timing.
    shared_legacy_wavefronts: u32,
    /// Whole-warp width-only ideal retained for scheduler accounting.
    shared_legacy_ideal_wavefronts: u32,
    /// Exact whole-warp A-branch observer retained for SM86 compatibility.
    shared_sm86_wavefronts: u32,
    /// Observer-only local-memory sector cardinality after applying the
    /// effective SM86 lane-private frame stride. The regular `sectors` plan is
    /// retained unchanged for replay state/timing; this field exists solely to
    /// expose a source-derived local endpoint when compressed local offsets
    /// would otherwise collapse lanes onto one stack frame.
    local_canonical_sectors: u16,
    /// Observer-only local endpoint using the 256-byte SM86 stack-frame
    /// allocation visible in the captured SASS prologue.  It is kept beside
    /// the transfer-width candidate so the emitter can form a bounded
    /// coalescing correction for dynamic-index local accesses.
    local_frame256_sectors: u16,
    /// Observer-only witness that all active lanes use one byte address and
    /// width for this memory instruction.
    warp_uniform_address: bool,

    meta: CompactMemoryMeta,
    /// Zero means absent; otherwise this is wire enum value + 1. Keeping the
    /// tiny closed enum in one byte preserves the compact metadata bound.
    cache_op: u8,
}

fn local_canonical_sector_count(
    preflight: PreflightedMemoryAccess<'_>,
    fixed_lane_stride: Option<i64>,
) -> u16 {
    let mut sectors = std::collections::BTreeSet::new();
    preflight.visit_addresses(|access| {
        let width = i64::from(access.data_width_bytes);
        // Characterized SM86 local records advance adjacent lane-private
        // slots by the transfer width itself (4B scalar, 16B vector, ...).
        // This endpoint candidate remains outside replay state/timing.
        let lane_stride = fixed_lane_stride.unwrap_or_else(|| width.max(1));
        for (lane, &base) in access.lane_vector.iter().enumerate() {
            if base == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let Some(lane_offset) = i64::try_from(lane)
                .ok()
                .and_then(|lane| lane.checked_mul(lane_stride))
            else {
                return;
            };
            let Some(first_byte) = base.checked_add(lane_offset) else {
                return;
            };
            let Some(last_byte) = first_byte.checked_add(width.saturating_sub(1)) else {
                return;
            };
            let first_sector = first_byte >> 5;
            let last_sector = last_byte >> 5;
            for sector in first_sector..=last_sector {
                sectors.insert(sector);
            }
        }
    });
    u16::try_from(sectors.len()).unwrap_or(u16::MAX)
}

fn warp_uniform_address(preflight: PreflightedMemoryAccess<'_>) -> bool {
    let mut first: Option<(i64, i32)> = None;
    let mut active = 0_u32;
    let mut uniform = true;
    preflight.visit_addresses(|access| {
        for lane in 0..lanes::LANES_PER_WARP {
            if (access.lane_mask >> lane) & 1 == 0 {
                continue;
            }
            let address = access.lane_vector[lane];
            if address == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            active = active.saturating_add(1);
            if let Some(expected) = first {
                if expected != (address, access.data_width_bytes) {
                    uniform = false;
                }
            } else {
                first = Some((address, access.data_width_bytes));
            }
        }
    });
    active != 0 && uniform
}

#[derive(Debug, Clone, Copy)]
struct AsyncCopyProjection {
    effective_data_bytes: u64,
    destination_topology: CompactLdgstsDestinationTopology,
    cohort_span: CompactArenaSpan,
    source_sector_mask_span: CompactArenaSpan,
}

#[derive(Debug, Clone, Copy, Default)]
struct CompactArenaSpan {
    start: u32,
    len: u16,
}

impl CompactArenaSpan {
    fn new(start: usize, len: usize) -> Self {
        Self {
            start: u32::try_from(start)
                .unwrap_or_else(|_| unreachable!("validated sparse arena exceeded u32")),
            len: u16::try_from(len)
                .unwrap_or_else(|_| unreachable!("validated sparse span exceeded u16")),
        }
    }

    fn slice<T>(self, arena: &[T]) -> &[T] {
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
struct CompactLdgstsDestinationTopology {
    source_cohorts: u8,
    destination_line_fanout_total: u8,
    destination_line_fanout_max: u8,
    cache_all_bank_wavefronts_total: u8,
    cache_all_bank_wavefronts_max: u8,
    bypass_wavefronts_total: u8,
    bypass_wavefronts_max: u8,
}

impl CompactLdgstsDestinationTopology {
    const UNAVAILABLE: u8 = u8::MAX;

    fn new(
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

    const fn expand(self) -> Option<LdgstsDestinationTopology> {
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
    fn new(
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

    const fn destination_topology(self) -> Option<LdgstsDestinationTopology> {
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
    launch_index: i64,
    block_id: [u32; 3],
    warps: Vec<ValidatedWarp>,
}

impl ValidatedThreadblock {
    fn from_message(
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
    map_key: i32,
    warp_id: i32,
    function_unique_id: i32,
    instructions: Vec<ValidatedInstruction>,
    predicate_on_masks: Vec<u32>,
    instrumentation_order_anchors: Vec<CompactInstrumentationOrderAnchor>,
    memories: Vec<ValidatedMemoryInstruction>,
    /// Dynamic constant-bank offsets retained as a bounded, launch-local
    /// demand observer. These are not global allocation addresses and never
    /// participate in cache state or timing.
    constant_memory_addresses: Vec<u64>,
    async_copy_projections: Vec<AsyncCopyProjection>,
    ldgsts_source_cohorts: Vec<LdgstsSourceCohort>,
    ldgsts_source_sector_cohort_masks: Vec<u32>,
    route_errors: Vec<ValidatedMemoryRouteError>,
    sectors: Vec<i64>,
    local_interleaved_sectors: Vec<i64>,
    sector_errors: Vec<ValidatedSectorError>,
}

impl ValidatedWarp {
    pub const fn function_unique_id(&self) -> i32 {
        self.function_unique_id
    }

    fn from_message(
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

    fn async_copy_projection(
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
struct ValidatedInstruction {
    pc: u32,
    data_width: Option<i32>,
    memory_index: u32,
    constant_memory_span: CompactArenaSpan,
    constant_memory_bank: Option<u32>,
    constant_memory_offset: Option<u64>,
    sm_id: i32,
    warpid_sm: i32,
    identity_presence: u8,
    active_mask_present: bool,
    predicate_mask_present: bool,
    // Preserve lane identity as well as population: per-lane successor PCs
    // cannot be reconstructed from popcounts when a warp splits/reconverges.
    active_mask: u32,
    predicate_on_lanes: u8,
}

/// Sparse order witness for one selected dynamic instruction. Keeping this in
/// a warp-local arena avoids widening every hot compact instruction when the
/// opt-in producer field is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompactInstrumentationOrderAnchor {
    order: u64,
    instruction_index: u32,
}

impl ValidatedInstruction {
    const SM_ID_PRESENT: u8 = 1 << 0;
    const WARP_SLOT_PRESENT: u8 = 1 << 1;
}

/// One instruction borrowed together with the compact memory metadata from its
/// owning warp. The pairing prevents memory-arena indices from crossing warps.
#[derive(Clone, Copy)]
pub struct ValidatedInstructionRef<'a> {
    instruction: &'a ValidatedInstruction,
    memory: Option<&'a ValidatedMemoryInstruction>,
    constant_memory_addresses: &'a [u64],
    function_unique_id: i32,
    warpid_tb: i32,
    instruction_index: u32,
    predicate_on_mask: u32,
    instrumentation_order: Option<u64>,
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

fn compact_memory_route(
    instruction: &pb::Instruction,
    class: pb::MemoryOpClass,
    payload_contract: MemoryPayloadContract,
    and_mask: u32,
) -> Result<ValidatedMemoryRoute, ValidatedMemoryRouteError> {
    let space = |address: &pb::Address| {
        address
            .address_space
            .map(|raw| match pb::MemorySpace::try_from(raw) {
                Ok(space) => space,
                Err(_) => unreachable!("validated address carried an unknown space"),
            })
    };
    let zero_effective_mask = and_mask == 0;
    match class {
        pb::MemoryOpClass::Unspecified => Ok(ValidatedMemoryRoute::Unsupported),
        pb::MemoryOpClass::Shared | pb::MemoryOpClass::Local => {
            if instruction.addresses.is_empty() && !zero_effective_mask {
                return Err(ValidatedMemoryRouteError::SharedLocalOperandCount { got: 0 });
            }
            let expected = if class == pb::MemoryOpClass::Shared {
                pb::MemorySpace::Shared
            } else {
                pb::MemorySpace::Local
            };
            for (operand, address) in instruction.addresses.iter().enumerate() {
                let Some(space) = space(address) else {
                    continue;
                };
                if space != expected {
                    return Err(ValidatedMemoryRouteError::SharedLocalSpaceInvalid {
                        operand,
                        space,
                    });
                }
            }
            Ok(if class == pb::MemoryOpClass::Shared {
                ValidatedMemoryRoute::Shared
            } else {
                ValidatedMemoryRoute::Local
            })
        }
        pb::MemoryOpClass::Global => {
            if instruction.addresses.is_empty() && !zero_effective_mask {
                return Err(ValidatedMemoryRouteError::GlobalOperandCount { got: 0 });
            }
            for (operand, address) in instruction.addresses.iter().enumerate() {
                let Some(space) = space(address) else {
                    continue;
                };
                if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                    return Err(ValidatedMemoryRouteError::GlobalSpaceInvalid { operand, space });
                }
            }
            Ok(ValidatedMemoryRoute::Global)
        }
        pb::MemoryOpClass::Atomic => {
            if payload_contract == MemoryPayloadContract::LegacyExclusion {
                let mut resolved = None;
                for (operand, address) in instruction.addresses.iter().enumerate() {
                    let Some(space) = space(address) else {
                        continue;
                    };
                    let candidate = match space {
                        pb::MemorySpace::Global | pb::MemorySpace::Generic => {
                            ValidatedMemoryRoute::Global
                        }
                        pb::MemorySpace::Shared => ValidatedMemoryRoute::Shared,
                        pb::MemorySpace::Local => ValidatedMemoryRoute::Local,
                        pb::MemorySpace::Unspecified => {
                            return Err(ValidatedMemoryRouteError::AtomicSpaceUnspecified {
                                operand,
                            });
                        }
                    };
                    match resolved {
                        Some(previous) if previous != candidate => {
                            return Err(ValidatedMemoryRouteError::AtomicSpacesConflict);
                        }
                        None => resolved = Some(candidate),
                        Some(_) => {}
                    }
                }
                return Ok(resolved.unwrap_or(ValidatedMemoryRoute::Global));
            }
            if instruction.addresses.is_empty() && zero_effective_mask {
                return Ok(ValidatedMemoryRoute::Global);
            }
            if instruction.addresses.len() != 1 {
                return Err(ValidatedMemoryRouteError::AtomicOperandCount {
                    got: instruction.addresses.len(),
                });
            }
            let Some(space) = space(&instruction.addresses[0]) else {
                return Err(ValidatedMemoryRouteError::AtomicSpaceMissing { operand: 0 });
            };
            match space {
                pb::MemorySpace::Global | pb::MemorySpace::Generic => {
                    Ok(ValidatedMemoryRoute::Global)
                }
                pb::MemorySpace::Shared => Ok(ValidatedMemoryRoute::Shared),
                pb::MemorySpace::Local => Ok(ValidatedMemoryRoute::Local),
                pb::MemorySpace::Unspecified => {
                    Err(ValidatedMemoryRouteError::AtomicSpaceUnspecified { operand: 0 })
                }
            }
        }
        pb::MemoryOpClass::AsyncCopy => {
            if payload_contract == MemoryPayloadContract::LegacyExclusion {
                for address in &instruction.addresses {
                    let Some(space) = space(address) else {
                        continue;
                    };
                    if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                        return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid {
                            space,
                        });
                    }
                }
                return Ok(ValidatedMemoryRoute::Global);
            }
            if payload_contract.carries_ldgsts_destination() {
                if instruction.addresses.len() != 2 {
                    return Err(ValidatedMemoryRouteError::AsyncCopyOperandCount {
                        got: instruction.addresses.len(),
                        expected: 2,
                    });
                }
                let Some(source_space) = space(&instruction.addresses[0]) else {
                    return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceMissing);
                };
                if !matches!(
                    source_space,
                    pb::MemorySpace::Global | pb::MemorySpace::Generic
                ) {
                    return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid {
                        space: source_space,
                    });
                }
                let Some(destination_space) = space(&instruction.addresses[1]) else {
                    return Err(ValidatedMemoryRouteError::AsyncCopyDestinationSpaceMissing);
                };
                if destination_space != pb::MemorySpace::Shared {
                    return Err(
                        ValidatedMemoryRouteError::AsyncCopyDestinationSpaceInvalid {
                            space: destination_space,
                        },
                    );
                }
                return Ok(ValidatedMemoryRoute::Global);
            }
            if instruction.addresses.len() != 1 {
                return Err(ValidatedMemoryRouteError::AsyncCopySourceCount {
                    got: instruction.addresses.len(),
                });
            }
            let Some(space) = space(&instruction.addresses[0]) else {
                return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceMissing);
            };
            if !matches!(space, pb::MemorySpace::Global | pb::MemorySpace::Generic) {
                return Err(ValidatedMemoryRouteError::AsyncCopySourceSpaceInvalid { space });
            }
            Ok(ValidatedMemoryRoute::Global)
        }
        pb::MemoryOpClass::Texture
        | pb::MemoryOpClass::Surface
        | pb::MemoryOpClass::Tma
        | pb::MemoryOpClass::ClusterDsmem
        | pb::MemoryOpClass::Tmem => Ok(ValidatedMemoryRoute::Unsupported),
    }
}

fn compact_sector_vector_for_operand(
    preflight: PreflightedMemoryAccess<'_>,
    selected_operand: Option<usize>,
    local_interleave: bool,
) -> Result<Vec<i64>, (ValidatedSectorErrorPhase, ValidatedSectorError)> {
    let mut intervals = Vec::new();
    let mut early_error = None;
    preflight.visit_addresses_enumerated(|operand, access| {
        if early_error.is_some() || selected_operand.is_some_and(|selected| selected != operand) {
            return;
        }
        if access.data_width_bytes <= 0 {
            early_error = Some(ValidatedSectorError::DataWidthInvalid {
                data_width_bytes: access.data_width_bytes,
            });
            return;
        }
        let width_offset = i64::from(access.data_width_bytes) - 1;
        for (lane, &base) in access.lane_vector.iter().enumerate() {
            if base == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let base = if local_interleave {
                let word = base.div_euclid(4);
                let byte = base.rem_euclid(4);
                let words_per_lane = (access.data_width_bytes.max(4) + 3) / 4;
                let Some(lane_word) = word
                    .checked_mul(i64::try_from(lanes::LANES_PER_WARP).unwrap_or(32))
                    .and_then(|value| {
                        value.checked_add(
                            i64::try_from(lane)
                                .unwrap_or(0)
                                .checked_mul(i64::from(words_per_lane))?,
                        )
                    })
                else {
                    early_error = Some(ValidatedSectorError::AddressOverflow {
                        lane,
                        base,
                        data_width_bytes: access.data_width_bytes,
                    });
                    return;
                };
                let Some(expanded) = lane_word
                    .checked_mul(4)
                    .and_then(|value| value.checked_add(byte))
                else {
                    early_error = Some(ValidatedSectorError::AddressOverflow {
                        lane,
                        base,
                        data_width_bytes: access.data_width_bytes,
                    });
                    return;
                };
                expanded
            } else {
                base
            };
            let Some(last_address) = base.checked_add(width_offset) else {
                early_error = Some(ValidatedSectorError::AddressOverflow {
                    lane,
                    base,
                    data_width_bytes: access.data_width_bytes,
                });
                return;
            };
            if intervals.try_reserve(1).is_err() {
                early_error = Some(ValidatedSectorError::CapacityUnavailable {
                    buffer: "sector interval plan",
                    additional: 1,
                });
                return;
            }
            intervals.push((base >> 5, last_address >> 5));
        }
    });
    if let Some(error) = early_error {
        return Err((ValidatedSectorErrorPhase::BeforeStaticRole, error));
    }

    let mut sorted = Vec::new();
    sorted.try_reserve(intervals.len()).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "sorted sector intervals",
                additional: intervals.len(),
            },
        )
    })?;
    sorted.extend_from_slice(&intervals);
    sorted.sort_unstable();
    let mut union_count = 0_u128;
    if let Some(&(initial_first, initial_last)) = sorted.first() {
        let mut first = initial_first;
        let mut last = initial_last;
        for &(next_first, next_last) in &sorted[1..] {
            if next_first <= last.saturating_add(1) {
                last = last.max(next_last);
            } else {
                union_count += (i128::from(last) - i128::from(first) + 1).unsigned_abs();
                first = next_first;
                last = next_last;
            }
        }
        union_count += (i128::from(last) - i128::from(first) + 1).unsigned_abs();
    }
    let unique_count = usize::try_from(union_count).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::UniqueSectorCountOutOfRange {
                sectors: union_count,
            },
        )
    })?;
    if unique_count > usize::from(u16::MAX) {
        return Err((
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::SectorCountOverflow {
                sectors: unique_count,
            },
        ));
    }

    // The union count above proves this expansion is bounded by u16::MAX.
    // Preserve first-seen interval/sector order with a set-backed walk instead
    // of repeatedly searching and draining the growing covered-interval list.
    let mut staged = Vec::new();
    staged.try_reserve(unique_count).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "sector staging",
                additional: unique_count,
            },
        )
    })?;
    let mut seen = HashSet::new();
    seen.try_reserve(unique_count).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "sector membership set",
                additional: unique_count,
            },
        )
    })?;
    for (first, last) in intervals {
        let mut sector = first;
        loop {
            if seen.insert(sector) {
                staged.push(sector);
            }
            if sector == last {
                break;
            }
            sector = sector
                .checked_add(1)
                .unwrap_or_else(|| unreachable!("validated interval end escaped i64"));
        }
    }
    if staged.len() != unique_count {
        return Err((
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CardinalityMismatch {
                expected: unique_count,
                produced: staged.len(),
            },
        ));
    }
    Ok(staged)
}

/// Project one address-carrying global atomic warp instruction onto the
/// 32-byte sector transactions observed at L1TEX/LTS on sm_86.
///
/// Distinct atomic words in one sector share a transaction, while lanes that
/// alias the same atomic word serialize. Consequently each touched sector is
/// repeated by the largest exact-address multiplicity among lanes touching
/// that sector. This retains ordinary coalescing (`lane_word`: four sectors),
/// distinct-line traffic (`lane_line`: 32 sectors), and same-word
/// serialization (`warp_word`: one sector repeated 32 times) without retaining
/// the full lane vector after trace validation.
fn compact_atomic_sector_transaction_vector_32b(
    preflight: PreflightedMemoryAccess<'_>,
) -> Result<Vec<i64>, (ValidatedSectorErrorPhase, ValidatedSectorError)> {
    let unique = compact_sector_vector_for_operand(preflight, None, false)?;
    if unique.is_empty() {
        return Ok(unique);
    }

    let mut intervals = Vec::new();
    let mut early_error = None;
    preflight.visit_addresses(|access| {
        if early_error.is_some() {
            return;
        }
        if access.data_width_bytes <= 0 {
            early_error = Some(ValidatedSectorError::DataWidthInvalid {
                data_width_bytes: access.data_width_bytes,
            });
            return;
        }
        let width_offset = i64::from(access.data_width_bytes) - 1;
        for (lane, &base) in access.lane_vector.iter().enumerate() {
            if base == lanes::LANE_INACTIVE_SENTINEL {
                continue;
            }
            let Some(last_address) = base.checked_add(width_offset) else {
                early_error = Some(ValidatedSectorError::AddressOverflow {
                    lane,
                    base,
                    data_width_bytes: access.data_width_bytes,
                });
                return;
            };
            if intervals.try_reserve(1).is_err() {
                early_error = Some(ValidatedSectorError::CapacityUnavailable {
                    buffer: "atomic address intervals",
                    additional: 1,
                });
                return;
            }
            intervals.push((base, base >> 5, last_address >> 5));
        }
    });
    if let Some(error) = early_error {
        return Err((ValidatedSectorErrorPhase::BeforeStaticRole, error));
    }
    intervals.sort_unstable();

    let mut multiplicities = Vec::new();
    multiplicities.try_reserve(unique.len()).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "atomic sector multiplicities",
                additional: unique.len(),
            },
        )
    })?;
    let mut transaction_count = 0_usize;
    for &sector in &unique {
        let mut previous_base = None;
        let mut aliases = 0_usize;
        let mut maximum_aliases = 0_usize;
        for &(base, first_sector, last_sector) in &intervals {
            if sector < first_sector || sector > last_sector {
                continue;
            }
            if previous_base == Some(base) {
                aliases += 1;
            } else {
                previous_base = Some(base);
                aliases = 1;
            }
            maximum_aliases = maximum_aliases.max(aliases);
        }
        debug_assert!(
            maximum_aliases > 0,
            "unique atomic sector has no lane interval"
        );
        transaction_count = transaction_count.checked_add(maximum_aliases).ok_or((
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::SectorCountOverflow {
                sectors: usize::MAX,
            },
        ))?;
        if transaction_count > usize::from(u16::MAX) {
            return Err((
                ValidatedSectorErrorPhase::AfterStaticRole,
                ValidatedSectorError::SectorCountOverflow {
                    sectors: transaction_count,
                },
            ));
        }
        multiplicities.push(maximum_aliases);
    }

    let mut staged = Vec::new();
    staged.try_reserve(transaction_count).map_err(|_| {
        (
            ValidatedSectorErrorPhase::AfterStaticRole,
            ValidatedSectorError::CapacityUnavailable {
                buffer: "atomic sector transaction staging",
                additional: transaction_count,
            },
        )
    })?;
    for (&sector, &multiplicity) in unique.iter().zip(&multiplicities) {
        staged.extend(std::iter::repeat_n(sector, multiplicity));
    }
    Ok(staged)
}

struct ValidatedPayload {
    block_index: i64,
    records: u64,
    block: ValidatedThreadblock,
}

fn validate_threadblock_payload(
    raw: &[u8],
    path: &Path,
    kernel: &pb::Kernel,
    allocations: Option<&AllocationRegistry>,
    memory_payload_contract: MemoryPayloadContract,
) -> Result<ValidatedPayload, TraceError> {
    if raw.is_empty() {
        return Err(TraceError::ThreadblockRecordsEmpty {
            path: path.to_path_buf(),
        });
    }
    let threadblock = decode_threadblock(raw, path)?;
    let block_index = block_launch_index(kernel, &threadblock, path)?;
    let records = threadblock.warps.values().try_fold(0_u64, |total, warp| {
        let count =
            u64::try_from(warp.instructions.len()).map_err(|_| TraceError::TraceCountOverflow {
                counter: "threadblock instruction count",
            })?;
        total
            .checked_add(count)
            .ok_or(TraceError::TraceCountOverflow {
                counter: "threadblock instruction count",
            })
    })?;
    if records == 0 {
        return Err(TraceError::ThreadblockRecordsEmpty {
            path: path.to_path_buf(),
        });
    }
    let label = path.to_string_lossy();
    validate_threadblock_contract(
        &threadblock,
        &label,
        allocations,
        memory_payload_contract,
        kernel.function_unique_id,
    )?;
    let block = ValidatedThreadblock::from_message(
        block_index,
        threadblock,
        memory_payload_contract,
        &label,
    )?;
    Ok(ValidatedPayload {
        block_index,
        records,
        block,
    })
}

fn validate_trace_semantics(
    root: &Path,
    trace: &pb::Trace,
    payload_paths: impl IntoIterator<Item = PathBuf>,
    memory_payload_contract: MemoryPayloadContract,
    mut validate_payload: impl FnMut(&Path, &pb::Kernel) -> Result<ValidatedPayload, TraceError>,
) -> Result<
    (
        TraceValidationSummary,
        BTreeMap<PathBuf, ValidatedThreadblock>,
    ),
    TraceError,
> {
    let payload_paths = payload_paths.into_iter().collect::<BTreeSet<_>>();
    let mut launch_ids = BTreeSet::new();
    let mut consumed_payloads = BTreeSet::new();
    let mut validated_payload = BTreeMap::new();
    let mut payloads_by_parent: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
    for path in &payload_paths {
        let parent =
            path.parent()
                .ok_or_else(|| TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: root.to_path_buf(),
                    message: format!("payload path has no launch parent: {}", path.display()),
                })?;
        payloads_by_parent
            .entry(parent.to_path_buf())
            .or_default()
            .push(path.clone());
    }
    let mut launch_count = 0_u64;
    let mut threadblock_count = 0_u64;
    let mut instruction_count = 0_u64;

    for (&device_key, device) in &trace.gpu_device {
        if u32::try_from(device_key).ok() != Some(device.id) {
            return Err(TraceError::DeviceMapKeyMismatch {
                map_key: device_key,
                embedded_id: device.id,
            });
        }
        for (&stream_key, stream) in &device.streams {
            if stream_key != stream.id {
                return Err(TraceError::StreamMapKeyMismatch {
                    map_key: stream_key,
                    embedded_id: stream.id,
                });
            }
            for kernel in &stream.kernels {
                if kernel.id <= 0 {
                    return Err(TraceError::LaunchIdentityInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                    });
                }
                if !launch_ids.insert((device_key, stream_key, kernel.id)) {
                    return Err(TraceError::LaunchIdentityDuplicate {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                    });
                }
                launch_count =
                    launch_count
                        .checked_add(1)
                        .ok_or(TraceError::TraceCountOverflow {
                            counter: "launch count",
                        })?;
                if kernel.function_unique_id <= 0 {
                    return Err(TraceError::FunctionUniqueIdInvalid {
                        record: format!(
                            "root launch device={device_key} stream={stream_key} kernel={}",
                            kernel.id
                        ),
                        got: kernel.function_unique_id,
                    });
                }
                if kernel.number_of_registers < 0 {
                    return Err(TraceError::TraceLaunchRegistersInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        registers: kernel.number_of_registers,
                    });
                }
                if kernel
                    .shared_memory_carveout_preference_pct
                    .is_some_and(|preference| !(-1..=100).contains(&preference))
                {
                    return Err(TraceError::TraceLaunchCarveoutInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        preference: kernel
                            .shared_memory_carveout_preference_pct
                            .expect("checked as present"),
                    });
                }
                if kernel
                    .driver_occupancy_limit_blocks
                    .is_some_and(|blocks| blocks == 0)
                {
                    return Err(TraceError::TraceLaunchOccupancyInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        blocks: kernel
                            .driver_occupancy_limit_blocks
                            .expect("checked as present"),
                    });
                }
                if !positive_geometry(kernel.grid_dim.as_ref()) {
                    return Err(TraceError::LaunchGeometryInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        field: "grid_dim",
                    });
                }
                if !positive_geometry(kernel.block_dim.as_ref()) {
                    return Err(TraceError::LaunchGeometryInvalid {
                        device_id: device_key,
                        stream_id: stream_key,
                        kernel_id: kernel.id,
                        field: "block_dim",
                    });
                }
                validate_shared_memory_split(
                    device_key,
                    stream_key,
                    kernel,
                    memory_payload_contract,
                )?;

                let launch_dir = root
                    .join("threadblocks")
                    .join(format!("device_{device_key}"))
                    .join(format!("stream_{stream_key}"))
                    .join(format!("kernel_{}", kernel.id));
                let Some(paths) = payloads_by_parent.get(launch_dir.as_path()) else {
                    return Err(TraceError::ThreadblocksMissing { dir: launch_dir });
                };
                let mut block_ids = BTreeSet::new();
                for path in paths {
                    let validated = validate_payload(path, kernel)?;
                    if !block_ids.insert(validated.block_index) {
                        return Err(TraceError::BlockIdDuplicate {
                            path: path.to_path_buf(),
                            index: validated.block_index,
                        });
                    }
                    validated_payload.insert(path.clone(), validated.block);
                    consumed_payloads.insert(path.clone());
                    threadblock_count =
                        threadblock_count
                            .checked_add(1)
                            .ok_or(TraceError::TraceCountOverflow {
                                counter: "threadblock count",
                            })?;
                    instruction_count = instruction_count.checked_add(validated.records).ok_or(
                        TraceError::TraceCountOverflow {
                            counter: "instruction count",
                        },
                    )?;
                }
            }
        }
    }
    if launch_count == 0 {
        return Err(TraceError::TraceLaunchesEmpty);
    }
    if let Some(path) = payload_paths
        .iter()
        .find(|path| !consumed_payloads.contains(path.as_path()))
    {
        return Err(TraceError::ThreadblockPayloadOrphan { path: path.clone() });
    }
    Ok((
        TraceValidationSummary {
            launch_count,
            threadblock_count,
            instruction_count,
        },
        validated_payload,
    ))
}

fn validate_trace_snapshot_semantics(
    root: &Path,
    trace: &pb::Trace,
    payload: &BTreeMap<PathBuf, Vec<u8>>,
    allocations: Option<&AllocationRegistry>,
    memory_payload_contract: MemoryPayloadContract,
) -> Result<
    (
        TraceValidationSummary,
        BTreeMap<PathBuf, ValidatedThreadblock>,
    ),
    TraceError,
> {
    validate_trace_semantics(
        root,
        trace,
        payload.keys().cloned(),
        memory_payload_contract,
        |path, kernel| {
            let raw =
                payload
                    .get(path)
                    .ok_or_else(|| TraceError::SamplingPayloadSnapshotMissing {
                        path: path.to_path_buf(),
                    })?;
            validate_threadblock_payload(raw, path, kernel, allocations, memory_payload_contract)
        },
    )
}

pub struct TraceDir {
    async_sidecars: OnceLock<
        Result<crate::async_sidecars::AsyncSidecars, crate::async_sidecars::AsyncSidecarError>,
    >,
    lineage: TraceLineage,
    root: PathBuf,
    trace: pb::Trace,
    /// Direct loaders retain the parsed execution-info join alongside the
    /// dynamic snapshot so replay cannot reopen a mutable sidecar by path.
    /// Campaign evidence carries this join separately in its manifest.
    exec_info: Option<ExecInfo>,
    memory_payload_contract: MemoryPayloadContract,
    allocations: Option<AllocationRegistry>,
    #[cfg(feature = "campaign")]
    sampling: Option<TraceSamplingSidecar>,
    stats: Option<TraceStats>,
    validation: TraceValidationSummary,
    payload_by_path: BTreeMap<PathBuf, Vec<u8>>,
    raw_payload_retained: bool,
    threadblock_by_path: BTreeMap<PathBuf, ValidatedThreadblock>,
}

/// In-process identity of one fully loaded trace snapshot. A `TraceDir` owns
/// all decoded replay inputs, so retaining this token prevents a stream-L2
/// session from combining its retained cache state with a separately opened
/// snapshot that happens to use the same filesystem path and launch IDs.
#[derive(Debug, Clone)]
pub struct TraceLineage(Arc<()>);

impl TraceLineage {
    fn new() -> Self {
        Self(Arc::new(()))
    }
}

impl PartialEq for TraceLineage {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for TraceLineage {}

impl TraceDir {
    /// Return the immutable in-process lineage token for this loaded snapshot.
    pub fn lineage(&self) -> TraceLineage {
        self.lineage.clone()
    }

    /// Parsed static execution metadata captured with a direct trace load.
    /// Campaign snapshots return `None` because their evidence admission owns
    /// the execution-info artifact and generation binding.
    pub fn exec_info(&self) -> Option<&ExecInfo> {
        self.exec_info.as_ref()
    }

    fn verify_direct_generations(
        &self,
        reader: &AnchoredDir,
        expected: &[(PathBuf, FileGenerationIdentity)],
    ) -> Result<(), TraceError> {
        for (relative, expected_generation) in expected {
            let actual = reader.inspect_regular_generation(relative)?;
            if actual != *expected_generation {
                return Err(TraceError::ThreadblockPayloadSnapshotInvalid {
                    root: self.root.clone(),
                    message: format!(
                        "direct trace artifact changed while loading: {}",
                        relative.display()
                    ),
                });
            }
        }
        Ok(())
    }
}

