from __future__ import annotations

import json
from pathlib import Path

from report_sm86_completion_frontier import report_frontiers


ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return json.loads((ROOT / "docs/sm86_completion_frontier_contract_v1.json").read_text())


def _document(status: str, tails: dict[str, bool] | None = None) -> dict:
    timing = {
        "completion_frontier": {
            "status": status,
            ("cycles" if status == "complete" else "lower_bound_cycles"): 100,
        },
        "gpc_observation_frontier": {
            "lower_bound_cycles": 100,
            "cycles": 120,
        },
    }
    if status != "complete":
        timing["completion_frontier"]["unmodeled_tails"] = tails or {
            "reduction_read_modify_write": False,
            "dirty_writeback": True,
            "coherence": False,
            "async_copy_shared_visibility": False,
        }
    return {"launches": [{"kernel_name": "fixture", "timing": timing}]}


def test_open_tail_never_becomes_finite_upper():
    report = report_frontiers(_document("incomplete_unmodeled_retirement_tails"), _contract())
    row = report["rows"][0]
    assert report["finite_upper_eligible_launches"] == 0
    assert row["finite_upper_eligible"] is False
    assert row["open_dimensions"] == ["dirty_writeback"]
    assert row["ncu_target_is_not_used"] is True


def test_complete_frontier_can_be_finite_without_observations():
    report = report_frontiers(_document("complete"), _contract())
    assert report["finite_upper_eligible_launches"] == 1
    assert report["rows"][0]["open_dimensions"] == []


def test_missing_tail_fields_are_rejected():
    document = _document("incomplete_unmodeled_retirement_tails")
    del document["launches"][0]["timing"]["completion_frontier"]["unmodeled_tails"]["coherence"]
    try:
        report_frontiers(document, _contract())
    except ValueError as error:
        assert "coherence" in str(error)
    else:
        raise AssertionError("incomplete tail map must be rejected")
