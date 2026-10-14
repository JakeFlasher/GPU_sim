//! Cache-operator classification from SASS opcode text — a lockstep port
//! of `the trace producer cache-op decoder`'s token rules
//! (closed 12-token vocabulary, priority-ordered matching, and the sm_86
//! bare-`LDG.E` → `.CA` default), applied to the control-bits sidecar's
//! per-pc `opcode_text` instead of a separate cuobjdump pass.
//!
//! WHY the sidecar text and not the classifier's own manifests: the
//! Python classifier's kernel-section extraction concatenates SASS
//! sections across EVERY fatbin arch slice (measured 2026-07-06 on the
//! M1 layernorm binary: 12 tagged "LDG.E" pcs of which only 4 are LDG
//! instructions in the traced sm_86 section — one is the STG at 0x9a0,
//! and pcs appear twice because the sm_80 slice's layout bleeds in). The
//! Phase-1E control-bits extractor handles arch sections correctly
//! (`skipped_arch_sections`), is transcription-guarded, and its
//! `opcode_text` is the same cuobjdump SASS text the classifier parses —
//! so the token rules are ported and applied per-pc to the sidecar.
//!
//! Deliberate hardening divergence: the Python rules regex-SEARCH the
//! whole line (opcode + operands); here only the leading opcode token's
//! dotted head is matched. On real SASS the two agree (mnemonics never
//! appear inside operand syntax); first-token matching cannot
//! false-match a hypothetical operand.

use sim_arch::{ArchError, ArchId, BareLdgEDefault, CapabilityError, LdgstsL1CacheSemantics};

/// The closed 12-token vocabulary (schema lockstep with
/// `cache_op_map.schema.json` and the Python tuple).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOpToken {
    LdgECa,
    LdgECg,
    LdgECs,
    LdgEEf,
    LdgEStrongSm,
    LdgEStrongGpu,
    Ldl,
    Stl,
    Ldgsts,
    Atom,
    Tex,
    Other,
}

/// Common architecture-neutral
/// `derive_l1_eligible` partition. Simulator routing may refine a literal token
/// by architecture: [`crate::access::mem_role_of_checked`] admits the measured
/// SM86 and RTX 5090 `LDG.E.STRONG.SM` lowering into L1 while retaining the
/// distinct STRONG.GPU bypass route.
pub fn derive_l1_eligible(token: CacheOpToken) -> bool {
    matches!(
        token,
        CacheOpToken::LdgECa | CacheOpToken::LdgEEf | CacheOpToken::Ldl | CacheOpToken::Stl
    )
}

/// Arch tokens for which a bare `LDG.E` (no explicit cache-modifier
/// suffix) resolves to `LDG_E_CA`: nvcc's default `-dlcm` is `ca` on
/// Ampere and cuobjdump prints the default modifier bare. Mirrors the
/// Python `_BARE_LDG_E_DEFAULTS_TO_CA` frozenset; other arches stay
/// conservative (bare `LDG.E` → OTHER) until documented.
///
/// This exported slice remains a producer/parity vocabulary surface. Runtime
/// decisions go through [`ArchId::bare_ldg_e_default`] so a research gap is
/// visible to checked consumers rather than becoming a cross-arch default.
pub const BARE_LDG_E_DEFAULTS_TO_CA: &[&str] = &["sm_86"];

/// Source-cache route encoded by an architecture-supported `LDGSTS` SASS form.
/// This is separate from [`CacheOpToken`] so the stable 12-token
/// vocabulary remains stable while async-copy completion retains its own type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdgstsL1Policy {
    CacheAtAllLevels,
    BypassL1,
}

#[derive(Debug, thiserror::Error)]
pub enum CacheOpError {
    #[error("{0}")]
    Architecture(#[from] ArchError),
    #[error("{0}")]
    Capability(#[from] CapabilityError),
    #[error("LDGSTS opcode {opcode:?} has no source-cache interpretation on {arch}")]
    LdgstsOpcodeUnsupported { arch: ArchId, opcode: String },
}

/// Leading opcode token's head (before the first `.`): the shared
/// extraction every opcode classifier in the workspace uses. Sidecar
/// `opcode_text` never carries a predicate prefix (the Phase-1E
/// extractor stores predicates in their own field).
pub fn opcode_head(opcode_text: &str) -> &str {
    let opcode = opcode_text.split_whitespace().next().unwrap_or("");
    opcode.split('.').next().unwrap_or("")
}

/// Classify the global-source L1 route of one `LDGSTS` instruction. Non-LDGSTS
/// opcodes return `None` without consulting an architecture capability.
///
/// On the characterized Ampere form, the first modifier must be `.E` and a
/// dotted `.BYPASS` component selects the L2-only route. Width and locality
/// hints are orthogonal. This matches the direct `cp.async.ca`/`cp.async.cg`
/// lowering and cache-traffic measurement rather than inferring policy from a
/// workload label.
pub fn classify_ldgsts_l1_policy(
    opcode_text: &str,
    arch: ArchId,
) -> Result<Option<LdgstsL1Policy>, CacheOpError> {
    let opcode = opcode_text.split_whitespace().next().unwrap_or("");
    let mut parts = opcode.split('.');
    if parts.next() != Some("LDGSTS") {
        return Ok(None);
    }
    let semantics = arch
        .ldgsts_l1_cache_semantics()
        .require("ldgsts_l1_cache_semantics")?;
    if parts.next() != Some("E") {
        return Err(CacheOpError::LdgstsOpcodeUnsupported {
            arch,
            opcode: opcode.to_owned(),
        });
    }
    match semantics {
        LdgstsL1CacheSemantics::AmpereOptionalBypassModifierV1 => {
            let bypass = parts.any(|modifier| modifier == "BYPASS");
            Ok(Some(if bypass {
                LdgstsL1Policy::BypassL1
            } else {
                LdgstsL1Policy::CacheAtAllLevels
            }))
        }
    }
}

/// Classify one sidecar `opcode_text` line with a typed deployment
/// architecture.  Checked consumers receive a research-gap error when a
/// bare `LDG.E` requires an architecture-specific default that is unavailable.
pub fn classify_opcode_for_arch(
    opcode_text: &str,
    arch: ArchId,
) -> Result<Option<CacheOpToken>, CacheOpError> {
    let opcode = opcode_text.split_whitespace().next().unwrap_or("");
    let mut parts = opcode.split('.');
    let head = parts.next().unwrap_or("");
    let token = match head {
        "LDGSTS" => CacheOpToken::Ldgsts,
        "LDL" => CacheOpToken::Ldl,
        "STL" => CacheOpToken::Stl,
        "ATOM" | "ATOMS" | "ATOMG" => CacheOpToken::Atom,
        "TEX" | "TLD" => CacheOpToken::Tex,
        "LDG" => {
            let first = parts.next();
            let second = parts.next();
            let third = parts.next();
            if first != Some("E") {
                // Bare `LDG` (no `.E`) is not eligible for the CA upgrade.
                CacheOpToken::Other
            } else {
                match (second, third) {
                    (Some("STRONG"), Some("SM")) => CacheOpToken::LdgEStrongSm,
                    (Some("STRONG"), Some("GPU")) => CacheOpToken::LdgEStrongGpu,
                    (Some("CA"), _) => CacheOpToken::LdgECa,
                    (Some("CG"), _) => CacheOpToken::LdgECg,
                    (Some("CS"), _) => CacheOpToken::LdgECs,
                    (Some("EF"), _) => CacheOpToken::LdgEEf,
                    // No explicit cache modifier: require the carded
                    // architecture default instead of borrowing sm_86.
                    _ => match arch.bare_ldg_e_default().require("bare_ldg_e_default")? {
                        BareLdgEDefault::CacheAtAllLevels => CacheOpToken::LdgECa,
                    },
                }
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(token))
}

/// Checked string-token entry point used at simulator composition boundaries.
pub fn classify_opcode_checked(
    opcode_text: &str,
    arch_token: &str,
) -> Result<Option<CacheOpToken>, CacheOpError> {
    let arch = ArchId::parse_deploy(arch_token)?;
    classify_opcode_for_arch(opcode_text, arch)
}
