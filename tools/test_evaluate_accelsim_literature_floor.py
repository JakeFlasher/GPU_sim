from __future__ import annotations

import json
from pathlib import Path

from tools.evaluate_accelsim_literature_floor import evaluate


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/accelsim_literature_lower_bound_manifest_20260928.json"
HOLDOUT = ROOT / "docs/sm86_holdout_replay_20260926.csv"


def test_current_development_holdout_is_not_scored_against_literature_floor() -> None:
    result = evaluate(MANIFEST, HOLDOUT)
    assert result["schema"] == "accelsim_literature_floor_evaluation_v1"
    assert result["scoring"] is False
    assert result["ground_truth_lane"] == "local_ncu"
    assert result["records"]
    assert all(record["ground_truth"] is False for record in result["records"])
    assert all(
        record["attainment_status"] in {
            "not_evaluable_unmatched",
            "not_evaluable_unadmitted_local_protocol",
        }
        for record in result["records"]
    )


def test_evaluation_is_json_serializable() -> None:
    json.dumps(evaluate(MANIFEST, None))
