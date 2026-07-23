"""Golden-fixture lockstep test: the Python side of the lane-reconstruction
LOCKSTEP PAIR (see lane_reconstruction_golden.json header).

The Rust twin is ``sim/trace/tests/lane_golden.rs``. Both consume the same
fixture; a semantic change to either decoder must extend the fixture in the
same commit so the other side fails until it is synced.

Run:  .venv/bin/python -m pytest tools/nvbit_tracer/test_lane_reconstruction_golden.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "workloads" / "simbench_bw" / "_pb_generated"))

import import_trace_contract as itc  # noqa: E402

GOLDEN = _HERE / "lane_reconstruction_golden.json"
LANES = 32
SENTINEL = itc.LANE_INACTIVE_SENTINEL

# Shared outcome tokens -> substrings of the Python decoder's reason strings.
_DECODER_ERROR_SUBSTRINGS = {
    "list_all_length": "list_all record has",
    "stride_addrs_nonempty": "non-empty addrs",
    "stride_single_lane": "popcount(and_mask)=1",
    "stride_non_contiguous": "not contiguous",
    "delta_count": "delta entr",
    "unknown_format": "unknown compression_format",
}


class _AddrStub:
    """Duck-typed stand-in for the address proto (the decoder only touches
    these four attributes)."""

    def __init__(self, case: dict):
        self.compression_format = case["compression_format"]
        self.base_address = case["base_address"]
        self.stride = case["stride"]
        if "addrs_raw_len" in case:
            self.addrs = [SENTINEL] * case["addrs_raw_len"]
        elif "addrs_list_all" in case:
            lanes = [SENTINEL] * LANES
            for lane, value in case["addrs_list_all"].items():
                lanes[int(lane)] = value
            self.addrs = lanes
        else:
            self.addrs = list(case["addrs"])


def _expected_vector(case: dict) -> list[int]:
    lanes = [SENTINEL] * LANES
    spec = case["expect_lanes"]
    if "__stride_from_base" in spec:
        base, stride = spec["__stride_from_base"]
        for k in range(LANES):
            if (case["and_mask"] >> k) & 1:
                lanes[k] = base + k * stride
        return lanes
    for lane, value in spec.items():
        lanes[int(lane)] = value
    return lanes


def _classify(case: dict) -> tuple[str | None, list[int] | None]:
    """Run the canonical decoder + positional check; return
    (outcome_token_or_None, vector_or_None)."""
    addr = _AddrStub(case)
    vector, reason = itc._reconstruct_lane_vector(addr, case["and_mask"])
    if vector is None:
        for token, substring in _DECODER_ERROR_SUBSTRINGS.items():
            if substring in reason:
                return token, None
        pytest.fail(f"{case['name']}: unmapped decoder reason {reason!r}")
    for lane in range(LANES):
        bit_set = bool((case["and_mask"] >> lane) & 1)
        if bit_set == (vector[lane] == SENTINEL):
            return "position_mismatch", None
    return None, list(vector)


def _cases() -> list[dict]:
    return json.loads(GOLDEN.read_text())["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_golden_case(case: dict) -> None:
    token, vector = _classify(case)
    if "error" in case:
        assert token == case["error"], (
            f"{case['name']}: expected error token {case['error']!r}, got "
            f"{token!r} (vector={vector})"
        )
    else:
        assert token is None, f"{case['name']}: unexpected error token {token!r}"
        assert vector == _expected_vector(case)


def test_fixture_covers_all_formats_and_errors() -> None:
    cases = _cases()
    formats = {c["compression_format"] for c in cases}
    assert {0, 1, 2}.issubset(formats), "golden must exercise every wire format"
    errors = {c["error"] for c in cases if "error" in c}
    assert errors == set(_DECODER_ERROR_SUBSTRINGS) | {"position_mismatch"}, (
        "golden must exercise every shared outcome token"
    )
