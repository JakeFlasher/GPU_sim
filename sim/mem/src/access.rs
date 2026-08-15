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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    NonMemory,
    /// Shared-memory class: no L1TEX sector traffic. `wavefronts` is the
    /// compact 32-bank/4-byte-word projection of the effective lane addresses;
    /// zero preserves fixed-latency behavior for zero-active or uncaptured
    /// payloads. Architecture-local issue timing decides whether this projection
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
    },
    /// Non-bulk global-to-shared async-copy source. Static classification still
    /// identifies `LDGSTS` as `MemRole::BypassLoad(Ldgsts)`; this dynamic form
    /// additionally preserves whether the current trace record has an effective
    /// source. Effective source presence is encoded by `payload_bytes != 0`,
    /// including in flat mode where the sector arena is intentionally omitted.
    GlobalAsyncCopy {
        off: usize,
        len: u16,
        /// Exact active-lane source payload (`active lanes × per-lane width`) in
        /// bytes. This is independent of the deduplicated source-sector span and
        /// is zero exactly when the source is ineffective. Replay narrows the
        /// validated u64 projection at a typed boundary so this field does not
        /// widen every compact dynamic instruction.
        payload_bytes: u32,
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
    },
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
