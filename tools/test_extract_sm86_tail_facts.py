from __future__ import annotations

import json
from pathlib import Path

from extract_sm86_tail_facts import extract_tail_facts


ROOT = Path(__file__).resolve().parents[1]


def test_live_message_trace_reports_fence_and_unknown_domains():
    report = extract_tail_facts(
        ROOT / "results/alpha/c1r_message_device_20261003/trace",
        ROOT / "results/alpha/c1r_message_device_20261003/control.json",
    )
    assert report["facts"]["stores"] > 0
    assert report["facts"]["atomics"] > 0
    assert any("MEMBAR" in value for value in report["facts"]["fence_opcodes_observed"])
    assert report["facts"]["owner_read_from_observed"] is False
    assert report["facts"]["hardware_completion_token_observed"] is False


def test_live_async_trace_reports_copy_protocol_and_unknown_completion():
    report = extract_tail_facts(
        ROOT / "results/alpha/c1r_async_matched_20261003/trace",
        ROOT / "results/alpha/c1r_async_matched_20261003/control.json",
    )
    assert report["facts"]["async_copies"] > 0
    assert report["facts"]["fence_scope_semantics_closed"] is False
    assert report["facts"]["hardware_completion_token_observed"] is False
