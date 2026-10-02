from __future__ import annotations

import json
from pathlib import Path

import pytest

from prediction_contract import validate_prediction_document
from validate_sm86_ncu_accuracy_protocol import validate_manifest_sources, validate_manifest_splits, validate_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_declared_campaign_prediction_is_pre_observation_and_source_bound():
    protocol = validate_protocol(json.loads((ROOT / "docs/sm86_ncu_accuracy_protocol_v1.json").read_text()))
    manifest = json.loads((ROOT / "docs/sm86_accuracy_campaign_v1_manifest.json").read_text())
    validate_manifest_splits(protocol, manifest)
    validate_manifest_sources(manifest, require_paths=True)
    prediction = json.loads((ROOT / "docs/sm86_accuracy_campaign_v1_predictions_pre_ncu.json").read_text())
    validate_prediction_document(prediction)
    assert prediction["prediction_stage"] == "pre_observation"
    assert prediction["target_contract"]["repetitions"] == 5
    assert len(prediction["source_bindings"]) == len(prediction["rows"]) == 15
    assert all(
        not any(field in row for field in ("ncu", "observed", "target", "target_value"))
        for row in prediction["rows"]
    )


def test_accuracy_prediction_uses_explicitly_unvalidated_service_sidecar():
    profile = json.loads((ROOT / "docs/sm86_service_rate_profile_20261001.json").read_text())
    assert "unvalidated" in profile["source"]
    assert {"queue_depth", "occupancy", "contention"}.issubset(profile["unmodeled_dimensions"])
