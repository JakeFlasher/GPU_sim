from __future__ import annotations

from report_sm86_tail_witness import report_tail_witness


def _document() -> dict:
    return {
        "launches": [{
            "kernel_name": "fixture",
            "coverage": {"kind": "full"},
            "counters": {},
            "trace_metrics": {"instructions": {"active_mask_missing_instructions": 0, "predicate_mask_missing_instructions": 0}, "coherence": {"unknown_sm_or_line_records": 0, "cross_sm_lines": 1, "instrumentation_order_missing_records": 0}},
            "timing": {"completion_frontier": {"status": "complete", "cycles": 10, "unmodeled_tails": {"reduction_read_modify_write": False, "dirty_writeback": False, "coherence": False, "async_copy_shared_visibility": False}}},
        }]
    }


def test_hardware_tail_domains_remain_open_without_independent_facts():
    report = report_tail_witness(_document())
    row = report["rows"][0]
    assert row["facts"]["launch_coverage"] is True
    assert row["independent_tail_witness_complete"] is False
    assert "line_generation" in row["open_domains"]
    assert "owner_read_from" in row["open_domains"]
    assert report["native_dispatch_eligible_launches"] == 0


def test_incomplete_simulator_tail_is_not_a_witness():
    document = _document()
    document["launches"][0]["timing"]["completion_frontier"]["status"] = "incomplete_unmodeled_retirement_tails"
    document["launches"][0]["timing"]["completion_frontier"]["unmodeled_tails"]["dirty_writeback"] = True
    report = report_tail_witness(document)
    assert report["rows"][0]["facts"]["simulator_tail_flags_empty"] is False
