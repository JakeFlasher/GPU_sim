"""Tests for the read-only frozen M1-bundle regression anchor validator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import validate_m1_anchor as vma  # noqa: E402

_BUNDLE = _REPO / "eval" / "frozen_labels" / "sm86_m1_bundle_v21.json"


def _evidence_present() -> bool:
    bundle = json.loads(_BUNDLE.read_text())
    return (_REPO / bundle["labels_root"]).is_dir() and \
           (_REPO / bundle["trace_manifest"]).is_file()


def test_bundle_is_tracked_and_wellformed():
    bundle = json.loads(_BUNDLE.read_text())
    assert bundle["bundle_id"] == "sm86_m1_bundle_v21"
    assert bundle["n_cells"] == vma.EXPECTED_ADMITTED
    assert len(bundle["cells"]) == vma.EXPECTED_ADMITTED
    assert len(bundle["excluded_cells"]) == vma.EXPECTED_EXCLUDED


def test_anchor_reproduces_when_evidence_present():
    if not _evidence_present():
        pytest.skip("machine-local M1 evidence annex not present on this host")
    findings = vma.validate_anchor(_BUNDLE)
    assert findings == [], findings


def test_evidence_absent_raises(tmp_path):
    """A bundle naming a nonexistent evidence root must fail-closed (not pass)."""
    bundle = json.loads(_BUNDLE.read_text())
    bundle["labels_root"] = "data/annex/does_not_exist_ever"
    tampered = tmp_path / "bundle.json"
    tampered.write_text(json.dumps(bundle))
    with pytest.raises(vma.EvidenceAbsent):
        vma.validate_anchor(tampered, repo_root=_REPO)


def test_tampered_count_is_detected_as_drift(tmp_path):
    if not _evidence_present():
        pytest.skip("machine-local M1 evidence annex not present on this host")
    bundle = json.loads(_BUNDLE.read_text())
    # Drop a cell: n_cells now disagrees with len(cells) -> a drift finding.
    bundle["cells"] = bundle["cells"][:-1]
    tampered = tmp_path / "bundle.json"
    tampered.write_text(json.dumps(bundle))
    findings = vma.validate_anchor(tampered, repo_root=_REPO)
    assert findings, "expected drift finding for a tampered admitted count"
