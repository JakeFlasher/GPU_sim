from __future__ import annotations

from copy import deepcopy

import pytest

from workloads.research_probes.analyze_sm86_live_closure import _derive_decision


def _rows() -> list[dict[str, object]]:
    return [
        {
            "cell": f"cell-{index}",
            "comparable_to_simulator_timing": True,
            "live": {"stable_within_five_percent": True},
            "simulator": {
                "readiness_schema": 6,
                "top_readiness_causes": [{"cause": "control_stall", "limiting_cycles": 1}],
                "tails": {"dirty_writeback": True, "coherence": True},
            },
            "residual": {"baseline_cycles": index + 1},
        }
        for index in range(2)
    ]


def test_decision_is_derived_from_validated_rows() -> None:
    ranking, validation, decision = _derive_decision(_rows())
    assert decision["decision"] == "no-go"
    assert decision["selected_diagnostic"] == "dirty_writeback_coherence_retirement"
    assert validation["decision_eligible"] is True
    assert ranking[0]["selection_score"] == 3


def test_unstable_repeat_is_not_silently_converted_to_no_go() -> None:
    rows = _rows()
    rows[0]["live"]["stable_within_five_percent"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="insufficient_evidence: live repeat is unstable"):
        _derive_decision(rows)


def test_missing_retirement_tails_is_not_silently_converted_to_no_go() -> None:
    rows = deepcopy(_rows())
    rows[1]["simulator"]["tails"]["dirty_writeback"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="insufficient_evidence: dirty_writeback"):
        _derive_decision(rows)
