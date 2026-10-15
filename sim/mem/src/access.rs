//! Memory-record classification shared by the SM issue model and the
//! memory hierarchy. Lives in sim-mem (not sim-sm) because the token is
//! MEMORY semantics: which level a record stages into, whether it
//! allocates, and (next increments) per-token L2 behavior — the
//! `CacheOpToken → MemRole → AccessKind` chain keeps the token instead of
//! flattening it away, so bypass/untagged/store sectors can take
//! token-specific hierarchy paths without re-joining the sidecar.

use sim_arch::ArchId;

use crate::cache_op::{
    CacheOpError, CacheOpToken, LdgstsL1Policy, classify_opcode_for_arch, derive_l1_eligible,
    opcode_head,
};

/// Dynamic address-space provenance for hierarchy accounting. This is
/// deliberately orthogonal to cache source, operation class, and physical
/// cache identity: Global and Local traffic share one cache state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryOrigin {
    Global,
    Local,
}

impl MemoryOrigin {
    pub const ALL: [Self; 2] = [Self::Global, Self::Local];

    pub const fn index(self) -> usize {
        match self {
            Self::Global => 0,
            Self::Local => 1,
        }
    }
}

/// How the memory hierarchy treats a global-class record at a static pc,
/// classified from the sidecar's SASS text (see `cache_op` for why the
/// Python classifier's own manifests are not consumed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemRole {
    /// Load-class opcode staging into the modeled L1.
    EligibleLoad(CacheOpToken),
    /// Load-class opcode bypassing the modeled L1 (`.cg`/`.cs`/strong
    /// scopes/LDGSTS/ATOM/TEX/OTHER).
    BypassLoad(CacheOpToken),
    /// Plain global-store head (STG): sectors hit-tested against the L1
    /// without allocating; write-allocate at the L2.
    Store,
    /// Non-returning global reduction. RED is recognized on both target
    /// architectures; Hopper+ REDG is recognized only where the static ISA family
    /// exists. Replay support remains a separate architecture-local gate. It
    /// follows the physical write-class hierarchy path, but remains distinct from
    /// both plain-store attribution and returning ATOM semantics.
    Reduction,
    /// Not load-class and not a store head (LD.E, LDC, LDSM, non-memory
    /// opcodes): counted in the unified sector denominator only when the
    /// dynamic record is global-class.
    None,
}

/// Is this opcode a plain global-store head? STS/STL are not global stores
/// (shared / local — STL is L1-eligible via its token), and RED/REDG have a
/// distinct non-returning reduction role.
pub fn is_global_store_opcode(opcode_text: &str) -> bool {
    opcode_head(opcode_text) == "STG"
}

/// Is this opcode a non-returning global reduction head?
pub fn is_global_reduction_opcode(opcode_text: &str) -> bool {
    matches!(opcode_head(opcode_text), "RED" | "REDG")
}

#[derive(Debug, thiserror::Error)]
pub enum MemRoleError {
    #[error("cache-op classification failed: {0}")]
    CacheOp(#[from] CacheOpError),
    #[error("static SASS family {family} is unavailable on architecture {arch}")]
    StaticMemoryFamilyUnsupported { family: &'static str, arch: ArchId },
}

/// Checked architecture-aware classifier used by simulator composition.
pub fn mem_role_of_checked(opcode_text: &str, arch_token: &str) -> Result<MemRole, MemRoleError> {
    let arch = ArchId::parse_deploy(arch_token).map_err(CacheOpError::from)?;
    Ok(match classify_opcode_for_arch(opcode_text, arch)? {
        // CUDA 13.2 lowers an sm_86 `ld.global.ca` to
        // `LDG.E.STRONG.SM`.  The fixed/L1TEX priority probe measures
        // that exact form as an L1 tag hit on GA102.  Keep the literal token
        // (the same spelling has different Blackwell cache semantics), but
        // route it through the Ampere L1 hierarchy here.
        Some(CacheOpToken::LdgEStrongSm) if matches!(arch, ArchId::Sm86 | ArchId::Sm120) => {
            // RTX 5090 `.ca` traces lower to STRONG.SM and independent NCU
            // rows retain an L1-eligible route; keep STRONG.GPU separate.
            MemRole::EligibleLoad(CacheOpToken::LdgEStrongSm)
        }
        Some(token) if derive_l1_eligible(token) => MemRole::EligibleLoad(token),
        Some(token) => MemRole::BypassLoad(token),
        None if is_global_store_opcode(opcode_text) => MemRole::Store,
        None if opcode_head(opcode_text) == "REDG" => match arch {
            ArchId::Sm86 => {
                return Err(MemRoleError::StaticMemoryFamilyUnsupported {
                    family: "REDG",
                    arch,
                });
            }
            ArchId::Sm120 => MemRole::Reduction,
        },
        None if is_global_reduction_opcode(opcode_text) => MemRole::Reduction,
        None => MemRole::None,
    })
}

/// Static heads that are intentionally left untagged by the closed cache-op
/// vocabulary but are known load-family instructions. A dynamic global record
/// with any other untagged head is a schema/sidecar mismatch and must fail
/// closed in replay.
pub fn is_known_untagged_opcode(opcode_text: &str) -> bool {
    matches!(opcode_head(opcode_text), "LD" | "LDC" | "LDS" | "LDSM")
}

/// static [`MemRole`] of its pc joined with the record's decoded unique
/// sectors. Sector payloads (`off`/`len`) index the owning warp's sector
/// arena.
pub const ASYNC_COPY_EXACT_DESTINATION_FLAG: u32 = 1 << 31;
pub const ASYNC_COPY_COMPLETION_VALUE_MASK: u32 = ASYNC_COPY_EXACT_DESTINATION_FLAG - 1;
const ASYNC_COPY_COMPOSITE_MARKER: u32 = 1;
const ASYNC_COPY_COMPOSITE_PAYLOAD_SHIFT: u32 = 1;
const ASYNC_COPY_COMPOSITE_PAYLOAD_MASK: u32 = 0xff;
const ASYNC_COPY_COMPOSITE_PAYLOAD_GRANULARITY: u32 = 8;
const ASYNC_COPY_COMPOSITE_COHORT_SHIFT: u32 = 9;
const ASYNC_COPY_COMPOSITE_COHORT_MASK: u32 = 0x1f;
const ASYNC_COPY_COMPOSITE_LINES_SHIFT: u32 = 14;
const ASYNC_COPY_COMPOSITE_LINES_MASK: u32 = 0x3f;
const ASYNC_COPY_COMPOSITE_BANK_SHIFT: u32 = 20;
const ASYNC_COPY_COMPOSITE_BANK_MASK: u32 = 0x1f;
const ASYNC_COPY_COMPOSITE_BYPASS_SHIFT: u32 = 25;
const ASYNC_COPY_COMPOSITE_BYPASS_MASK: u32 = 0x1f;
pub const ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX: u32 =
    ASYNC_COPY_COMPOSITE_PAYLOAD_MASK * ASYNC_COPY_COMPOSITE_PAYLOAD_GRANULARITY;

/// Meaning of the compact `GlobalAsyncCopy::completion_work` token. Cache-all
/// records may carry exact timing work. Destination-bearing bypass records
/// carry payload timing work plus an observational wavefront count; payload
/// work may be zero for a topology-only `ignore-src` copy, and that count
/// cannot affect timing until a bypass-specific mechanism is characterized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncCopyCompletion {
    Inactive,
    PayloadWork(u32),
    ExactDestinationWork(u32),
    PayloadAndObservedTopology {
        payload_work: u32,
        source_cohorts: u32,
        destination_lines: u32,
        max_bank_wavefronts: u32,
        max_bypass_wavefronts: u32,
    },
}

pub const fn decode_async_copy_completion(token: u32) -> AsyncCopyCompletion {
    if token == 0 {
        AsyncCopyCompletion::Inactive
    } else if token & ASYNC_COPY_EXACT_DESTINATION_FLAG != 0
        && token & ASYNC_COPY_COMPOSITE_MARKER != 0
    {
        AsyncCopyCompletion::PayloadAndObservedTopology {
            payload_work: ((token >> ASYNC_COPY_COMPOSITE_PAYLOAD_SHIFT)
                & ASYNC_COPY_COMPOSITE_PAYLOAD_MASK)
                * ASYNC_COPY_COMPOSITE_PAYLOAD_GRANULARITY,
            source_cohorts: ((token >> ASYNC_COPY_COMPOSITE_COHORT_SHIFT)
                & ASYNC_COPY_COMPOSITE_COHORT_MASK)
                + 1,
            destination_lines: ((token >> ASYNC_COPY_COMPOSITE_LINES_SHIFT)
                & ASYNC_COPY_COMPOSITE_LINES_MASK)
                + 1,
            max_bank_wavefronts: ((token >> ASYNC_COPY_COMPOSITE_BANK_SHIFT)
                & ASYNC_COPY_COMPOSITE_BANK_MASK)
                + 1,
            max_bypass_wavefronts: ((token >> ASYNC_COPY_COMPOSITE_BYPASS_SHIFT)
                & ASYNC_COPY_COMPOSITE_BYPASS_MASK)
                + 1,
        }
    } else if token & ASYNC_COPY_EXACT_DESTINATION_FLAG != 0 {
        AsyncCopyCompletion::ExactDestinationWork(token & ASYNC_COPY_COMPLETION_VALUE_MASK)
    } else {
        AsyncCopyCompletion::PayloadWork(token)
    }
}

pub const fn encode_async_copy_payload_work(work: u32) -> Option<u32> {
    if work <= ASYNC_COPY_COMPLETION_VALUE_MASK {
        Some(work)
    } else {
        None
    }
}

pub const fn encode_async_copy_destination_work(work: u32) -> Option<u32> {
    if work == 0 {
        Some(0)
    } else if work <= ASYNC_COPY_COMPLETION_VALUE_MASK && work & ASYNC_COPY_COMPOSITE_MARKER == 0 {
        Some(ASYNC_COPY_EXACT_DESTINATION_FLAG | work)
    } else {
        None
    }
}

pub const fn encode_async_copy_payload_and_topology(
    payload_work: u32,
    source_cohorts: u32,
    destination_lines: u32,
    max_bank_wavefronts: u32,
    max_bypass_wavefronts: u32,
) -> Option<u32> {
    if !payload_work.is_multiple_of(ASYNC_COPY_COMPOSITE_PAYLOAD_GRANULARITY)
        || source_cohorts == 0
        || destination_lines == 0
        || max_bank_wavefronts == 0
        || max_bypass_wavefronts == 0
    {
        None
    } else if payload_work <= ASYNC_COPY_COMPOSITE_PAYLOAD_WORK_MAX
        && source_cohorts <= ASYNC_COPY_COMPOSITE_COHORT_MASK + 1
        && destination_lines <= ASYNC_COPY_COMPOSITE_LINES_MASK + 1
        && max_bank_wavefronts <= ASYNC_COPY_COMPOSITE_BANK_MASK + 1
        && max_bypass_wavefronts <= ASYNC_COPY_COMPOSITE_BYPASS_MASK + 1
    {
        Some(
            ASYNC_COPY_EXACT_DESTINATION_FLAG
                | ASYNC_COPY_COMPOSITE_MARKER
                | ((payload_work / ASYNC_COPY_COMPOSITE_PAYLOAD_GRANULARITY)
                    << ASYNC_COPY_COMPOSITE_PAYLOAD_SHIFT)
                | ((source_cohorts - 1) << ASYNC_COPY_COMPOSITE_COHORT_SHIFT)
                | ((destination_lines - 1) << ASYNC_COPY_COMPOSITE_LINES_SHIFT)
                | ((max_bank_wavefronts - 1) << ASYNC_COPY_COMPOSITE_BANK_SHIFT)
                | ((max_bypass_wavefronts - 1) << ASYNC_COPY_COMPOSITE_BYPASS_SHIFT),
        )
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    NonMemory,
    /// A dynamically issued memory instruction whose effective
    /// `active_mask & predicate_mask` is zero. The scheduler still consumes
    /// the instruction slot and control-code stall, but the instruction
    /// creates no memory traffic, result writeback, or dependence-barrier
    /// producer.
    PredicatedOffMemory,
    /// Shared-memory class: no L1TEX sector traffic. `wavefronts` is the
    /// compact 32-bank/4-byte-word projection of the effective lane addresses;
    /// zero preserves fixed-latency behavior for uncaptured legacy payloads;
    /// dynamically zero-active records use [`Self::PredicatedOffMemory`].
    /// Architecture-local issue timing decides whether this projection
    /// is executable.
    Shared {
        wavefronts: u64,
        /// Unavoidable wavefronts from per-lane access width. This is kept
        /// separate from `wavefronts` so reports can subtract ideal wide-load
        /// service before calling the remainder a bank conflict.
        ideal_wavefronts: u64,
    },
    /// Local-memory load (LDL): modeled sectors in the same physical L1/L2
    /// hierarchy as Global traffic, with Local accounting provenance.
    LocalLoad {
        off: usize,
        len: u16,
    },
    /// Local-memory store (STL): write-through/no-allocate L1 probe and
    /// write-allocate L2 access in the shared physical hierarchy, with Local
    /// accounting provenance.
    LocalStore {
        off: usize,
        len: u16,
    },
    /// Manifest-eligible global-class record: modeled L1 sectors.
    GlobalEligible {
        off: usize,
        len: u16,
        token: CacheOpToken,
    },
    /// Sidecar-tagged INELIGIBLE record (`.cg`/`.cs`/LDGSTS/ATOM/...):
    /// bypasses the modeled L1 straight to the L2.
    GlobalBypass {
        off: usize,
        len: u16,
        token: CacheOpToken,
        /// Exact L2 request topology for global ATOMG. Zero for every other
        /// bypass token. Derived before compact replay discards lane addresses.
        l2_requests: u16,
    },
    /// Non-bulk global-to-shared async-copy source. Static classification still
    /// identifies `LDGSTS` as `MemRole::BypassLoad(Ldgsts)`; this dynamic form
    /// additionally preserves whether the current trace record has async
    /// activity. Activity is encoded by a non-inactive completion token,
    /// including topology-only bypass records whose payload work is zero and
    /// flat mode where the sector arena is intentionally omitted.
    GlobalAsyncCopy {
        off: usize,
        len: u16,
        /// Tagged completion token. Untagged values are work in the
        /// architecture's fixed 64-unit/cycle domain; tagged forms store either
        /// exact cache-all destination work or bypass payload work paired with
        /// an observational destination-wavefront count. Replay performs every
        /// checked projection before packing so this field stays compact.
        completion_work: u32,
        /// Static SASS source-cache route. This remains explicit on a
        /// predicated-off dynamic record even though its payload is zero.
        l1_policy: LdgstsL1Policy,
    },
    /// Global-class record whose pc has no recognized hierarchy role: counted
    /// in the unified denominator only.
    GlobalUntagged {
        off: usize,
        len: u16,
    },
    /// Plain global STORE (STG): sectors hit-tested against the L1 without
    /// allocating (v0 write-through no-allocate); allocate at the L2 (card
    /// policy `write_allocate_non_inclusive`).
    GlobalStore {
        off: usize,
        len: u16,
    },
    /// Non-returning global reduction (RED/REDG). It takes the same physical
    /// write-class L1/L2 path as a store but is excluded from ordinary
    /// global-store attribution; its unmodeled RMW tail is reported separately.
    GlobalReduction {
        off: usize,
        len: u16,
        /// Exact L2 requests represented by the alias-expanded sector span.
        l2_requests: u16,
    },
}

impl AccessKind {
    /// Exact L2 request count for a global returning atomic or reduction.
    /// `None` distinguishes non-atomic paths from a fully predicated-off atomic
    /// whose exact count is `Some(0)`.
    pub const fn atomic_l2_requests(self) -> Option<u16> {
        match self {
            Self::GlobalBypass {
                token: CacheOpToken::Atom,
                l2_requests,
                ..
            }
            | Self::GlobalReduction { l2_requests, .. } => Some(l2_requests),
            _ => None,
        }
    }
}
