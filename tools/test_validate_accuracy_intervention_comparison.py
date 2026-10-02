from __future__ import annotations

import json
from pathlib import Path

import pytest
jsonschema = pytest.importorskip("jsonschema")

from validate_accuracy_intervention_comparison import validate_comparison


ROOT = Path(__file__).resolve().parents[1]


def test_intervention_comparison_validates_all_declared_contrasts():
    plan = json.loads((ROOT / "docs/sm86_ncu_intervention_plan_v1.json").read_text())
    comparison = json.loads((ROOT / "docs/sm86_accuracy_intervention_comparison_v1.json").read_text())
    result = validate_comparison(plan, comparison, ROOT)
    assert len(result["contrast_summaries"]) == 2
    assert all(summary["causal_claim"] is False for summary in result["contrast_summaries"])
    schema = json.loads((ROOT / "docs/sm86_accuracy_intervention_comparison_v1.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(result)


def test_validator_rejects_missing_repetition():
    plan = json.loads((ROOT / "docs/sm86_ncu_intervention_plan_v1.json").read_text())
    comparison = json.loads((ROOT / "docs/sm86_accuracy_intervention_comparison_v1.json").read_text())
    comparison["rows"][0]["contrasts"]["cache_none"]["samples"]["gpc__cycles_elapsed.max"] = [1]
    with pytest.raises(ValueError, match="repetition count mismatch"):
        validate_comparison(plan, comparison, ROOT)
