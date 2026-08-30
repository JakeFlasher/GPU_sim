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
        // `LDG.E.STRONG.SM`.  The fixed/L1TEX priority campaign measures
        // that exact form as an L1 tag hit on GA102.  Keep the literal token
        // (the same spelling has different Blackwell cache semantics), but
        // route it through the Ampere L1 hierarchy here.
        Some(CacheOpToken::LdgEStrongSm) if arch == ArchId::Sm86 => {
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
const ASYNC_COPY_COMPOSITE_MARKER_MASK: u32 = 0x3f;
const ASYNC_COPY_COMPOSITE_MARKER: u32 = ASYNC_COPY_COMPOSITE_MARKER_MASK;
const ASYNC_COPY_COMPOSITE_PAYLOAD_SHIFT: u32 = 6;
const ASYNC_COPY_COMPOSITE_DESTINATION_SHIFT: u32 = 18;
pub const ASYNC_COPY_COMPOSITE_FIELD_MASK: u32 = 0x0fff;

/// Meaning of the compact `GlobalAsyncCopy::completion_work` token. Cache-all
/// records may carry exact timing work. Destination-bearing bypass records
/// carry payload timing work plus an observational wavefront count; that count
/// cannot affect timing until a bypass-specific mechanism is characterized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncCopyCompletion {
    Inactive,
    PayloadWork(u32),
    ExactDestinationWork(u32),
    PayloadAndObservedDestination {
        payload_work: u32,
        destination_wavefronts: u32,
    },
}

pub const fn decode_async_copy_completion(token: u32) -> AsyncCopyCompletion {
    if token == 0 {
        AsyncCopyCompletion::Inactive
    } else if token & ASYNC_COPY_EXACT_DESTINATION_FLAG != 0
        && token & ASYNC_COPY_COMPOSITE_MARKER_MASK == ASYNC_COPY_COMPOSITE_MARKER
    {
        AsyncCopyCompletion::PayloadAndObservedDestination {
            payload_work: (token >> ASYNC_COPY_COMPOSITE_PAYLOAD_SHIFT)
                & ASYNC_COPY_COMPOSITE_FIELD_MASK,
            destination_wavefronts: (token >> ASYNC_COPY_COMPOSITE_DESTINATION_SHIFT)
                & ASYNC_COPY_COMPOSITE_FIELD_MASK,
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
    } else if work <= ASYNC_COPY_COMPLETION_VALUE_MASK
        && work & ASYNC_COPY_COMPOSITE_MARKER_MASK != ASYNC_COPY_COMPOSITE_MARKER
    {
        Some(ASYNC_COPY_EXACT_DESTINATION_FLAG | work)
    } else {
        None
    }
}

pub const fn encode_async_copy_payload_and_destination(
    payload_work: u32,
    destination_wavefronts: u32,
) -> Option<u32> {
    if payload_work == 0 || destination_wavefronts == 0 {
        None
    } else if payload_work <= ASYNC_COPY_COMPOSITE_FIELD_MASK
        && destination_wavefronts <= ASYNC_COPY_COMPOSITE_FIELD_MASK
    {
        Some(
            ASYNC_COPY_EXACT_DESTINATION_FLAG
                | ASYNC_COPY_COMPOSITE_MARKER
                | (payload_work << ASYNC_COPY_COMPOSITE_PAYLOAD_SHIFT)
                | (destination_wavefronts << ASYNC_COPY_COMPOSITE_DESTINATION_SHIFT),
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
    /// additionally preserves whether the current trace record has an effective
    /// source. Effective source presence is encoded by `completion_work != 0`,
    /// including in flat mode where the sector arena is intentionally omitted.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_role_of(opcode_text: &str, arch_token: &str) -> MemRole {
        mem_role_of_checked(opcode_text, arch_token).unwrap()
    }

    #[test]
    fn dynamic_access_kind_stays_compact() {
        assert!(std::mem::size_of::<AccessKind>() <= 16);
    }

    #[test]
    fn atomic_request_topology_is_typed_and_zero_is_not_non_atomic() {
        assert_eq!(
            AccessKind::GlobalBypass {
                off: 0,
                len: 4,
                token: CacheOpToken::Atom,
                l2_requests: 1,
            }
            .atomic_l2_requests(),
            Some(1)
        );
        assert_eq!(
            AccessKind::GlobalReduction {
                off: 0,
                len: 0,
                l2_requests: 0,
            }
            .atomic_l2_requests(),
            Some(0)
        );
        assert_eq!(
            AccessKind::GlobalBypass {
                off: 0,
                len: 4,
                token: CacheOpToken::LdgECg,
                l2_requests: 0,
            }
            .atomic_l2_requests(),
            None
        );
    }

    #[test]
    fn async_copy_completion_token_round_trips_both_domains() {
        let work = encode_async_copy_payload_work(1024).unwrap();
        assert_eq!(
            decode_async_copy_completion(work),
            AsyncCopyCompletion::PayloadWork(1024)
        );
        let destination = encode_async_copy_destination_work(2048).unwrap();
        assert_eq!(
            decode_async_copy_completion(destination),
            AsyncCopyCompletion::ExactDestinationWork(2048)
        );
        let composite = encode_async_copy_payload_and_destination(1024, 32).unwrap();
        assert_eq!(
            decode_async_copy_completion(composite),
            AsyncCopyCompletion::PayloadAndObservedDestination {
                payload_work: 1024,
                destination_wavefronts: 32,
            }
        );
        assert_eq!(
            decode_async_copy_completion(encode_async_copy_destination_work(0).unwrap()),
            AsyncCopyCompletion::Inactive
        );
        assert!(encode_async_copy_payload_work(u32::MAX).is_none());
        assert!(encode_async_copy_destination_work(u32::MAX).is_none());
        assert!(
            encode_async_copy_payload_and_destination(ASYNC_COPY_COMPOSITE_FIELD_MASK + 1, 1,)
                .is_none()
        );
    }

    #[test]
    fn mem_roles_classified_from_sidecar_text() {
        assert_eq!(
            mem_role_of("LDG.E.U16.CONSTANT R0, [R2.64]", "sm_86"),
            MemRole::EligibleLoad(CacheOpToken::LdgECa)
        );
        assert_eq!(
            mem_role_of("LDGSTS.E.128 [R1], [R2.64]", "sm_86"),
            MemRole::BypassLoad(CacheOpToken::Ldgsts)
        );
        assert_eq!(
            mem_role_of("LDG.E.STRONG.SM R11, [R2.64]", "sm_86"),
            MemRole::EligibleLoad(CacheOpToken::LdgEStrongSm)
        );
        assert_eq!(
            mem_role_of("LDG.E.STRONG.SM R11, [R2.64]", "sm_120"),
            MemRole::BypassLoad(CacheOpToken::LdgEStrongSm)
        );
        assert_eq!(
            mem_role_of("ATOMG.E.CAS.STRONG.GPU PT, R4, [R2.64], R6, R7", "sm_86"),
            MemRole::BypassLoad(CacheOpToken::Atom)
        );
        assert_eq!(
            mem_role_of("STG.E.U16 [R2.64], R0", "sm_86"),
            MemRole::Store
        );
        assert_eq!(
            mem_role_of("RED.E.ADD.STRONG.GPU [R2.64], R0", "sm_86"),
            MemRole::Reduction
        );
        assert!(matches!(
            mem_role_of_checked("REDG.E.ADD.STRONG.GPU [R2.64], R0", "sm_86"),
            Err(MemRoleError::StaticMemoryFamilyUnsupported {
                family: "REDG",
                arch: ArchId::Sm86,
            })
        ));
        assert_eq!(
            mem_role_of("REDG.E.ADD.STRONG.GPU [R2.64], R0", "sm_120"),
            MemRole::Reduction
        );
        assert_eq!(
            mem_role_of("STL [R1], R0", "sm_86"),
            MemRole::EligibleLoad(CacheOpToken::Stl)
        );
        assert_eq!(mem_role_of("IADD3 R0, R1, R2, RZ", "sm_86"), MemRole::None);
        assert_eq!(mem_role_of("LD.E R0, [R2.64]", "sm_86"), MemRole::None);
        // Bare LDG.E off sm_86 is an architecture capability gap, never
        // conservatively reclassified into a generic bypass.
        assert!(mem_role_of_checked("LDG.E R0, [R2.64]", "sm_120").is_err());
        // STS is a shared store, never a global store head.
        assert_eq!(mem_role_of("STS [R1], R0", "sm_86"), MemRole::None);
    }
}
