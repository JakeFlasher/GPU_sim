"""Structural cross-surface parity: the cache-op classification contract
now has a Rust consumer surface (`sim/mem/src/cache_op.rs`, applied to
the control-bits sidecar's opcode text) in addition to the Python tuple
(`l1_cache_op_classifier.INSTRUCTION_CACHE_OP_TOKENS` + eligibility
partition) and the JSON schema enum (`cache_op_map.schema.json`).

The Rust port cannot fail-close at runtime against Python drift (it
classifies raw opcode text independently), so THIS test makes drift fail
in every `pytest` run — update BOTH surfaces in the same commit (pattern:
tools/sass_control/test_control_layout_rust_parity.py, per
BL-20260507-multi-surface-lockstep-changes).

Three pins:
  1. the Rust enum's 12 variants ↔ the Python 12-token tuple;
  2. the Rust eligibility partition ↔ ``L1_ELIGIBLE_TOKENS``;
  3. the Rust bare-``LDG.E``→``.CA`` arch set ↔
     ``_BARE_LDG_E_DEFAULTS_TO_CA``;
plus a behavioral corpus (the same vectors the Rust unit tests pin)
through the CANONICAL Python ``_match_opcode_token``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

from l1_cache_op_classifier import (  # noqa: E402
    INSTRUCTION_CACHE_OP_TOKENS,
    L1_ELIGIBLE_TOKENS,
    _BARE_LDG_E_DEFAULTS_TO_CA,
    _match_opcode_token,
    derive_l1_eligible,
)

_RUST_SRC = _REPO_ROOT / "sim/mem/src/cache_op.rs"

# Rust variant name ↔ wire token. This mapping is itself part of the
# pinned contract: a token added on either side without extending it
# fails the count/partition assertions below.
VARIANT_TO_TOKEN = {
    "LdgECa": "LDG_E_CA",
    "LdgECg": "LDG_E_CG",
    "LdgECs": "LDG_E_CS",
    "LdgEEf": "LDG_E_EF",
    "LdgEStrongSm": "LDG_E_STRONG_SM",
    "LdgEStrongGpu": "LDG_E_STRONG_GPU",
    "Ldl": "LDL",
    "Stl": "STL",
    "Ldgsts": "LDGSTS",
    "Atom": "ATOM",
    "Tex": "TEX",
    "Other": "OTHER",
}


def _rust_source() -> str:
    return _RUST_SRC.read_text(encoding="utf-8")


def _rust_enum_variants(src: str) -> list[str]:
    m = re.search(r"pub enum CacheOpToken \{(.*?)\n\}", src, re.DOTALL)
    assert m, "CacheOpToken enum not found in cache_op.rs"
    return re.findall(r"^\s{4}([A-Z][A-Za-z0-9]+),", m.group(1), re.MULTILINE)


def test_vocabulary_pinned_both_sides() -> None:
    variants = _rust_enum_variants(_rust_source())
    assert len(variants) == len(INSTRUCTION_CACHE_OP_TOKENS) == 12, (
        f"vocabulary width drifted: rust={len(variants)} "
        f"python={len(INSTRUCTION_CACHE_OP_TOKENS)} — update BOTH surfaces "
        f"(and the schema enum) in the same commit"
    )
    mapped = {VARIANT_TO_TOKEN[v] for v in variants}
    assert mapped == set(INSTRUCTION_CACHE_OP_TOKENS), (
        f"token sets disagree: only-rust={mapped - set(INSTRUCTION_CACHE_OP_TOKENS)} "
        f"only-python={set(INSTRUCTION_CACHE_OP_TOKENS) - mapped}"
    )


def test_eligibility_partition_pinned() -> None:
    src = _rust_source()
    m = re.search(
        r"pub fn derive_l1_eligible.*?matches!\(\s*token,\s*(.*?)\)\s*\n\}",
        src,
        re.DOTALL,
    )
    assert m, "derive_l1_eligible matches! arm not found in cache_op.rs"
    rust_eligible = {
        VARIANT_TO_TOKEN[v] for v in re.findall(r"CacheOpToken::([A-Za-z0-9]+)", m.group(1))
    }
    assert rust_eligible == set(L1_ELIGIBLE_TOKENS), (
        f"eligibility partitions disagree: rust={sorted(rust_eligible)} "
        f"python={sorted(L1_ELIGIBLE_TOKENS)}"
    )


def test_bare_ldg_e_arch_set_pinned() -> None:
    src = _rust_source()
    m = re.search(
        r"pub const BARE_LDG_E_DEFAULTS_TO_CA: &\[&str\] = &\[(.*?)\];", src, re.DOTALL
    )
    assert m, "BARE_LDG_E_DEFAULTS_TO_CA not found in cache_op.rs"
    rust_set = set(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
    assert rust_set == set(_BARE_LDG_E_DEFAULTS_TO_CA), (
        f"bare-LDG.E arch sets disagree: rust={sorted(rust_set)} "
        f"python={sorted(_BARE_LDG_E_DEFAULTS_TO_CA)} — the queued sm_120 "
        f"default must land on BOTH surfaces in the same commit"
    )


# The behavioral corpus the Rust unit tests pin
# (cache_op.rs::m1_roster_mnemonics_classify_like_the_python_rules and
# ::adjacency_contract_matches_python_regex_literals): the CANONICAL
# Python classifier must produce the same tokens.
_CORPUS_SM86 = [
    ("LDG.E R0, [R2.64]", "LDG_E_CA"),
    ("LDG.E.64 R4, [R2.64]", "LDG_E_CA"),
    ("LDG.E.U16 R0, [R2.64]", "LDG_E_CA"),
    ("LDG.E.U16.CONSTANT R0, [R2.64]", "LDG_E_CA"),
    ("LDG.E.CONSTANT R0, [R2.64]", "LDG_E_CA"),
    ("LDG.E.LTC128B.128 R4, [R2.64]", "LDG_E_CA"),
    ("LDG.E.STRONG.GPU R0, [R2.64]", "LDG_E_STRONG_GPU"),
    ("LDG.E.CA R0, [R2.64]", "LDG_E_CA"),
    ("LDG.E.CG R0, [R2.64]", "LDG_E_CG"),
    ("LDGSTS.E.BYPASS.LTC128B.128 [R1], [R2.64]", "LDGSTS"),
    ("LDGSTS.E.128 [R1], [R2.64]", "LDGSTS"),
    ("ATOMG.E.CAS.STRONG.GPU PT, R4, [R2.64], R6, R7", "ATOM"),
    # Adjacency contract: modifier counts only immediately after LDG.E.
    ("LDG.E.128.CG R0, [R2.64]", "LDG_E_CA"),
    ("LDG.E.EF.STRONG.GPU R0, [R2.64]", "LDG_E_EF"),
    ("LDG.E.CAX R0, [R2.64]", "LDG_E_CA"),
    ("TLD.LZ R4, [R8]", "TEX"),
    # Not load-class (silently-skipped set).
    ("LD.E R0, [R2.64]", None),
    ("LDC.64 R4, c[0x0][0x28]", None),
    ("LDS.128 R4, [R1]", None),
    ("STS.128 [R1], R4", None),
    ("STG.E.128 [R2.64], R4", None),
    ("RED.E.ADD.STRONG.GPU [R2.64], R0", None),
    ("LDSM.16.M88.4 R4, [R1]", None),
    ("IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]", None),
    ("EXIT", None),
    ("TLD4.NODEP.2D R4, [R8]", None),
    ("TLDS.LZ R4, [R8]", None),
]


def test_behavioral_corpus_matches_canonical_python() -> None:
    for text, expected in _CORPUS_SM86:
        m = _match_opcode_token(text, arch_token="sm_86")
        got = m[0] if m else None
        assert got == expected, (
            f"canonical Python classifier disagrees with the pinned corpus on "
            f"{text!r}: got {got}, corpus (== Rust unit tests) says {expected}"
        )
    # Off-sm_86 conservatism (the Rust bare_ldg_e_is_conservative_off_sm86
    # vectors).
    m = _match_opcode_token("LDG.E R0, [R2.64]", arch_token="sm_120")
    assert m and m[0] == "OTHER"
    m = _match_opcode_token("LDG.E.CA R0, [R2.64]", arch_token="sm_120")
    assert m and m[0] == "LDG_E_CA"


def test_eligibility_derivation_consistent() -> None:
    for token in INSTRUCTION_CACHE_OP_TOKENS:
        assert derive_l1_eligible(token) == (token in L1_ELIGIBLE_TOKENS)
