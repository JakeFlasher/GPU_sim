from __future__ import annotations

import json
from pathlib import Path

import pytest
jsonschema = pytest.importorskip("jsonschema")

from prediction_contract import validate_prediction_document


ROOT = Path(__file__).resolve().parents[1]


def test_intervention_plan_and_prediction_artifacts_are_pre_observation():
    plan = json.loads((ROOT / "docs/sm86_ncu_intervention_plan_v1.json").read_text())
    schema = json.loads((ROOT / "docs/sm86_ncu_intervention_plan_v1.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(plan)
    assert {contrast["variable"] for contrast in plan["contrasts"]} == {"cache_policy", "clock_control"}
    for contrast in plan["contrasts"]:
        prediction = json.loads((ROOT / contrast["prediction"]).read_text())
        validate_prediction_document(prediction)
        assert prediction["prediction_stage"] == "pre_observation"
        assert all("observed" not in row and "ncu" not in row for row in prediction["rows"])


def test_intervention_comparison_is_observational_only_and_has_all_cache_baseline():
    comparison = json.loads((ROOT / "docs/sm86_accuracy_intervention_comparison_v1.json").read_text())
    schema = json.loads((ROOT / "docs/sm86_accuracy_intervention_comparison_v1.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(comparison)
    assert comparison["causal_claim"] is False
    assert comparison["prediction_before_observation"] is True
    assert all(len(row["baseline_all_cache"]["samples"]["gpc__cycles_elapsed.max"]) == 5 for row in comparison["rows"])
    assert any("cache_none" in row["contrasts"] for row in comparison["rows"])
    assert any("clock_base" in row["contrasts"] for row in comparison["rows"])
