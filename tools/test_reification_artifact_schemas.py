from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")


ROOT = Path(__file__).resolve().parents[1]


def _validate(schema_name: str, document_path: str) -> None:
    schema = json.loads((ROOT / "docs" / schema_name).read_text())
    document = json.loads((ROOT / document_path).read_text())
    jsonschema.Draft202012Validator(schema).validate(document)


def test_committed_r6_artifacts_are_schema_valid():
    _validate(
        "reification_prediction_manifest_v1.schema.json",
        "docs/sm86_reification_corpus_v1_20261002.json",
    )
    _validate(
        "reification_prediction_v2.schema.json",
        "docs/review_round3_20261002/predictions_pre_ncu.json",
    )
    _validate(
        "ncu_observation_v2.schema.json",
        "docs/review_round3_20261002/observations_post_ncu.json",
    )
    _validate(
        "reification_prediction_evaluation_v2.schema.json",
        "docs/review_round3_20261002/evaluation_join.json",
    )
