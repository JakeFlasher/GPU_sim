//! Cache-operator classification from SASS opcode text — a lockstep port
//! of `tools/cache_predictor/l1_cache_op_classifier.py`'s token rules
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

/// Port of the predictor manifest's architecture-neutral
/// `derive_l1_eligible` partition. Simulator routing may refine a literal token
/// by architecture: [`crate::access::mem_role_of_checked`] admits the measured
/// sm_86 `LDG.E.STRONG.SM` lowering into L1 while Blackwell retains the
/// manifest's strong-scope bypass behavior.
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
/// This is separate from [`CacheOpToken`] so the historical 12-token parity
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn classify_opcode(opcode_text: &str, arch_token: &str) -> Option<CacheOpToken> {
        classify_opcode_checked(opcode_text, arch_token).unwrap()
    }

    #[test]
    fn eligibility_partition_matches_python() {
        for (token, eligible) in [
            (CacheOpToken::LdgECa, true),
            (CacheOpToken::LdgEEf, true),
            (CacheOpToken::Ldl, true),
            (CacheOpToken::Stl, true),
            (CacheOpToken::LdgECg, false),
            (CacheOpToken::LdgECs, false),
            (CacheOpToken::LdgEStrongSm, false),
            (CacheOpToken::LdgEStrongGpu, false),
            (CacheOpToken::Ldgsts, false),
            (CacheOpToken::Atom, false),
            (CacheOpToken::Tex, false),
            (CacheOpToken::Other, false),
        ] {
            assert_eq!(derive_l1_eligible(token), eligible, "{token:?}");
        }
    }

    #[test]
    fn m1_roster_mnemonics_classify_like_the_python_rules() {
        // Every load-class mnemonic observed across the 17 M1 static
        // tables, with the token the Python rules produce on sm_86.
        let cases = [
            ("LDG.E R0, [R2.64]", Some(CacheOpToken::LdgECa)), // bare → CA
            ("LDG.E.64 R4, [R2.64]", Some(CacheOpToken::LdgECa)),
            ("LDG.E.U16 R0, [R2.64]", Some(CacheOpToken::LdgECa)),
            ("LDG.E.U16.CONSTANT R0, [R2.64]", Some(CacheOpToken::LdgECa)),
            ("LDG.E.CONSTANT R0, [R2.64]", Some(CacheOpToken::LdgECa)),
            ("LDG.E.LTC128B.128 R4, [R2.64]", Some(CacheOpToken::LdgECa)),
            (
                "LDG.E.STRONG.GPU R0, [R2.64]",
                Some(CacheOpToken::LdgEStrongGpu),
            ),
            ("LDG.E.CA R0, [R2.64]", Some(CacheOpToken::LdgECa)),
            ("LDG.E.CG R0, [R2.64]", Some(CacheOpToken::LdgECg)),
            (
                "LDGSTS.E.BYPASS.LTC128B.128 [R1], [R2.64]",
                Some(CacheOpToken::Ldgsts),
            ),
            ("LDGSTS.E.128 [R1], [R2.64]", Some(CacheOpToken::Ldgsts)),
            (
                "ATOMG.E.CAS.STRONG.GPU PT, R4, [R2.64], R6, R7",
                Some(CacheOpToken::Atom),
            ),
            // Non-load-class heads: never in the manifest.
            ("LD.E R0, [R2.64]", None),
            ("LDC.64 R4, c[0x0][0x28]", None),
            ("LDS.128 R4, [R1]", None),
            ("STS.128 [R1], R4", None),
            ("STG.E.128 [R2.64], R4", None),
            ("RED.E.ADD.STRONG.GPU [R2.64], R0", None),
            ("LDSM.16.M88.4 R4, [R1]", None),
            ("IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]", None),
            ("EXIT", None),
        ];
        for (text, expected) in cases {
            assert_eq!(classify_opcode(text, "sm_86"), expected, "{text}");
        }
    }

    #[test]
    fn adjacency_contract_matches_python_regex_literals() {
        // A cache-op modifier counts only IMMEDIATELY after `LDG.E` —
        // the Python `\bLDG\.E\.CG\b`-style literals are contiguous.
        // Non-adjacent modifiers fall to the bare-`LDG.E` rule (CA on
        // sm_86), exactly like the Python fall-through.
        assert_eq!(
            classify_opcode("LDG.E.128.CG R0, [R2.64]", "sm_86"),
            Some(CacheOpToken::LdgECa)
        );
        // Adjacent EF wins over a later STRONG scope (Python: the
        // contiguous STRONG literal is absent, `LDG.E.EF` matches).
        assert_eq!(
            classify_opcode("LDG.E.EF.STRONG.GPU R0, [R2.64]", "sm_86"),
            Some(CacheOpToken::LdgEEf)
        );
        // Word-boundary equivalence: `CAX` is not `CA`.
        assert_eq!(
            classify_opcode("LDG.E.CAX R0, [R2.64]", "sm_86"),
            Some(CacheOpToken::LdgECa)
        );
        // TLD4/TLDS heads are NOT load-like (the Python gate `\bTLD\b`
        // never matches them): silently-skipped set.
        assert_eq!(classify_opcode("TLD4.NODEP.2D R4, [R8]", "sm_86"), None);
        assert_eq!(classify_opcode("TLDS.LZ R4, [R8]", "sm_86"), None);
        assert_eq!(
            classify_opcode("TLD.LZ R4, [R8]", "sm_86"),
            Some(CacheOpToken::Tex)
        );
    }

    #[test]
    fn sm86_ldgsts_cache_modifier_selects_the_source_l1_route() {
        for (opcode, expected) in [
            (
                "LDGSTS.E.128 [R4], [R32.64]",
                LdgstsL1Policy::CacheAtAllLevels,
            ),
            (
                "LDGSTS.E.64 [R17], [R2.64]",
                LdgstsL1Policy::CacheAtAllLevels,
            ),
            (
                "LDGSTS.E.BYPASS.LTC128B.128 [R1], [R2.64]",
                LdgstsL1Policy::BypassL1,
            ),
        ] {
            assert_eq!(
                classify_ldgsts_l1_policy(opcode, ArchId::Sm86).unwrap(),
                Some(expected),
                "opcode {opcode}"
            );
        }
        assert_eq!(
            classify_ldgsts_l1_policy("LDG.E R0, [R2.64]", ArchId::Sm120).unwrap(),
            None,
            "non-LDGSTS opcodes do not borrow the capability"
        );
    }

    #[test]
    fn ldgsts_cache_policy_fails_typed_on_unknown_forms_and_sm120() {
        assert!(matches!(
            classify_ldgsts_l1_policy("LDGSTS.128 [R1], [R2.64]", ArchId::Sm86),
            Err(CacheOpError::LdgstsOpcodeUnsupported {
                arch: ArchId::Sm86,
                opcode,
            }) if opcode == "LDGSTS.128"
        ));
        assert!(matches!(
            classify_ldgsts_l1_policy("LDGSTS.E.128 [R1], [R2.64]", ArchId::Sm120),
            Err(CacheOpError::Capability(CapabilityError::ResearchGap {
                arch: ArchId::Sm120,
                capability: "ldgsts_l1_cache_semantics",
                reason,
            })) if !reason.is_empty()
        ));
    }

    #[test]
    fn opcode_head_extraction() {
        assert_eq!(opcode_head("BAR.SYNC.DEFER_BLOCKING"), "BAR");
        assert_eq!(opcode_head("STG.E.128 [R2.64], R4"), "STG");
        assert_eq!(opcode_head(""), "");
    }

    #[test]
    fn bare_ldg_e_fails_closed_off_sm86() {
        assert!(matches!(
            classify_opcode_checked("LDG.E R0, [R2.64]", "sm_120"),
            Err(CacheOpError::Capability(
                CapabilityError::ResearchGap { .. }
            ))
        ));
        assert_eq!(
            classify_opcode("LDG.E.CA R0, [R2.64]", "sm_120"),
            Some(CacheOpToken::LdgECa),
            "explicit .CA is arch-independent"
        );
        // Bare LDG without .E stays OTHER even on sm_86.
        assert_eq!(
            classify_opcode("LDG R0, [R2]", "sm_86"),
            Some(CacheOpToken::Other)
        );
    }
}
