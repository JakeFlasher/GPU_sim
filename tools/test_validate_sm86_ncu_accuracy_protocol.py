from __future__ import annotations

import json
from pathlib import Path

import pytest

from validate_sm86_ncu_accuracy_protocol import validate_manifest_sources, validate_manifest_splits, validate_protocol


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads((ROOT / "docs/sm86_ncu_accuracy_protocol_v1.json").read_text())


def test_declared_accuracy_protocol_is_valid_and_covers_current_roster():
    protocol = validate_protocol(_protocol())
    manifest = json.loads((ROOT / "docs/sm86_reification_corpus_v1_20261002.json").read_text())
    validate_manifest_splits(protocol, manifest)


def test_protocol_rejects_inadequate_repetition_count():
    protocol = _protocol()
    protocol["measurement_contract"]["repetitions"] = 2
    with pytest.raises(ValueError, match="at least five"):
        validate_protocol(protocol)


def test_protocol_rejects_family_crossing():
    protocol = validate_protocol(_protocol())
    manifest = {
        "rows": [
            {"workload": "a", "family": "normalization", "split": "calibration"},
            {"workload": "b", "family": "normalization", "split": "held_out"},
        ]
    }
    with pytest.raises(ValueError, match="family .* violates protocol split"):
        validate_manifest_splits(protocol, manifest)


def test_protocol_declares_expansion_slots_and_source_fields_close():
    protocol = validate_protocol(_protocol())
    assert set(protocol["split_contract"]["planned_expansion_families"]) == {
        "scalar", "tensor", "memory", "synchronization", "mixed"
    }
    manifest = json.loads((ROOT / "docs/sm86_reification_corpus_v1_20261002.json").read_text())
    validate_manifest_sources(manifest, require_paths=True)


def test_source_closure_rejects_missing_input_path(tmp_path: Path):
    manifest = {"rows": [{"workload": "w", "trace": str(tmp_path / "missing"), "control": "c", "arch_card": "a", "pchase_card": "p"}]}
    with pytest.raises(ValueError, match="does not exist"):
        validate_manifest_sources(manifest, require_paths=True)
