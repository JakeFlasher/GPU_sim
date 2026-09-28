"""Structural checks for the published Accel-Sim aggregate target manifest."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/accelsim_literature_lower_bound_manifest_20260928.json"
SCHEMA = ROOT / "docs/accelsim_literature_lower_bound_manifest.schema.json"


def test_manifest_is_schema_valid_and_not_ground_truth() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    try:
        from jsonschema import Draft202012Validator
    except ImportError:  # pragma: no cover - the repo requirements provide it
        return
    errors = sorted(Draft202012Validator(schema).iter_errors(manifest), key=lambda error: list(error.path))
    assert not errors, "\n".join(f"{list(error.path)}: {error.message}" for error in errors)
    assert manifest["comparison_roles"]["ground_truth"] is False
    assert manifest["records"]
    for record in manifest["records"]:
        assert record["comparison_status"] == "lower_bound_reference_unmatched"
        assert record["target_status"] == "registered"
        assert record["threshold_type"] in {"ceiling", "floor"}
        assert record["target_direction"] in {"lower_is_better", "higher_is_better"}
        assert "gpu_tot_sim_cycle" not in record["metric"]
