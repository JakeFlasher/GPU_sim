"""Structural cross-surface parity: the control-bits layout contract now has
THREE surfaces — the Python producer (extractor.PARSER_CONTRACT), the sidecar
JSON it emits, and the Rust consumer (sim/sm/src/control_bits.rs
EXPECTED_LAYOUT). The Rust loader fail-closes at runtime on drift, but only
on a machine holding sidecar data; THIS test makes the drift fail in every
`pytest` run (pattern: test_proto_generated_binding_parity.py, per
BL-20260507-multi-surface-lockstep-changes).

Run:  .venv/bin/python -m pytest tools/sass_control/test_control_layout_rust_parity.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_REPO_ROOT = _HERE.parents[1]
_RUST_CONSUMER = _REPO_ROOT / "sim" / "sm" / "src" / "control_bits.rs"

from extractor import PARSER_CONTRACT, decode_control  # noqa: E402

_TUPLE_RE = re.compile(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)\s*,')


def _rust_expected_layout() -> dict[str, str]:
    text = _RUST_CONSUMER.read_text(encoding="utf-8")
    match = re.search(
        r"const EXPECTED_LAYOUT: \[\(&str, &str\); (\d+)\] = \[(.*?)\];",
        text,
        re.DOTALL,
    )
    assert match, "EXPECTED_LAYOUT not found in control_bits.rs — renamed without updating parity test?"
    declared_len = int(match.group(1))
    entries = dict(_TUPLE_RE.findall(match.group(2)))
    assert len(entries) == declared_len, "EXPECTED_LAYOUT parse drift (regex missed entries)"
    return entries


def test_rust_layout_matches_producer_contract() -> None:
    assert _rust_expected_layout() == PARSER_CONTRACT["control_layout"], (
        "sim/sm EXPECTED_LAYOUT != extractor.PARSER_CONTRACT['control_layout'] — "
        "update BOTH surfaces in the same commit"
    )


def test_rust_decode_bit_table_matches_producer() -> None:
    """Beyond the prose: verify the Rust decode() bit slicing agrees with the
    producer's decode_control on words exercising every field, by asserting
    the Rust source encodes the identical shifts/masks."""
    text = _RUST_CONSUMER.read_text(encoding="utf-8")
    for fragment in (
        "(high_word >> 41) & 0x7f_ffff",
        "(control >> 21) & 0b11",
        "control & 0xf",
        "(control >> 4) & 1",
        "(control >> 5) & 0x7",
        "(control >> 8) & 0x7",
        "(control >> 11) & 0x3f",
        "(control >> 17) & 0xf",
    ):
        assert fragment in text, f"Rust decode() lost bit-table fragment {fragment!r}"
    # And the producer side still decodes the reference word the Rust unit
    # test pins (gemm_hmma pc=0): stall=2, yield=1, wb=rb=7, wait=0, reuse=0.
    ref = decode_control(0x000FE400078E00FF)
    assert ref == {
        "stall": 2,
        "yield_flag": 1,
        "write_barrier": 7,
        "read_barrier": 7,
        "wait_mask": 0,
        "reuse": 0,
    }
