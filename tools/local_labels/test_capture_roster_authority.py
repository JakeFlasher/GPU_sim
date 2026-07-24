"""Tests for the durable-v2 sm_86 capture-roster projection."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import jsonschema
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import build_capture_roster_authority as bra  # noqa: E402

_SCHEMA = json.loads((_REPO / bra.SCHEMA_REL).read_text())
_COMMITTED = _REPO / bra.CANONICAL_OUT_REL


def _authority() -> dict:
    return json.loads(_COMMITTED.read_bytes())


def test_schema_valid():
    authority = _authority()
    jsonschema.validate(authority, _SCHEMA)
    assert set(authority["source_pins"]) == {
        "portable_lcd_roster_sha256",
        "simbench_build_witness_sha256",
        "simbench_roster_sha256",
        "sol_port_scope_sha256",
        "sol_runnability_sha256",
        "witness_compat_registry_sha256",
    }
    assert all(len(value) == 64 for value in authority["source_pins"].values())


def test_accounting_identity_and_totals():
    authority = _authority()
    counts = authority["counts"]
    assert counts == bra.EXPECTED_COUNTS
    assert len(authority["cells"]) == 344
    assert counts["runnable"] == 122
    assert counts["excluded"] == 222
    assert counts["pending_build"] == 0
    assert counts["pending_reconstitution"] == 0
    assert counts["pending_shrink"] == 0
    assert authority["accounting_identity_holds"] is True


def test_integrity_cross_checks_are_active_and_pass():
    authority = _authority()
    identities = [
        (cell["roster_source"], cell["slug"]) for cell in authority["cells"]
    ]
    assert len(identities) == len(set(identities)) == 344
    assert identities == sorted(identities)
    observed = Counter(cell["resolution_status"] for cell in authority["cells"])
    assert observed == {"runnable": 122, "excluded": 222}
    assert sum(authority["counts"][key] for key in (
        "runnable", "pending_build", "pending_reconstitution",
        "pending_shrink", "excluded")) == authority["counts"]["universe"]


def test_source_partition_sums():
    authority = _authority()
    source = Counter(cell["roster_source"] for cell in authority["cells"])
    runnable = Counter(
        cell["roster_source"] for cell in authority["cells"]
        if cell["resolution_status"] == "runnable")
    assert source == {"portable_lcd": 17, "simbench_bw": 92, "sol_execbench": 235}
    assert runnable == {"portable_lcd": 17, "simbench_bw": 31, "sol_execbench": 74}


def test_every_runnable_cell_resolves_a_binary():
    for cell in _authority()["cells"]:
        if cell["resolution_status"] != "runnable":
            continue
        assert isinstance(cell["binary"], str) and cell["binary"]
        assert isinstance(cell["app_args"], list) and cell["app_args"]
        assert isinstance(cell["shape_point"], str) and cell["shape_point"]
        assert cell["binary_present"] is True
        if cell["roster_source"] == "sol_execbench":
            assert cell["binary_sha256"] is None
        else:
            assert len(cell["binary_sha256"]) == 64


def test_schema_rejects_runnable_cell_with_null_capture_identity():
    authority = _authority()
    cell = next(item for item in authority["cells"]
                if item["roster_source"] == "simbench_bw"
                and item["resolution_status"] == "runnable")
    cell["binary_sha256"] = None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(authority, _SCHEMA)


def test_simbench_resolution_consumes_durable_witness_not_binary_presence():
    authority = _authority()
    witness = json.loads((_REPO / bra.WITNESS_REL).read_text())
    witness_rows = {row["slug"]: row for row in witness["rows"]}
    projected = [cell for cell in authority["cells"]
                 if cell["roster_source"] == "simbench_bw"
                 and cell["resolution_status"] == "runnable"]
    assert len(projected) == 31
    for cell in projected:
        row = witness_rows[cell["slug"]]
        assert cell["binary_present"] is True
        assert cell["binary_sha256"] == row["binary_sha256"]
        assert cell["app_args"] == row["app_args"]
        assert cell["shape_point"] == row["shape_point"]
        assert cell["build_witness_row_sha256"] == row["row_seal_sha256"]


@pytest.mark.campaign_evidence
def test_committed_roster_authority_matches_regenerated():
    fresh = bra.build_authority(_REPO)
    assert _COMMITTED.read_bytes() == bra.canonical_json_bytes(fresh)


def test_every_excluded_cell_is_typed():
    excluded = [cell for cell in _authority()["cells"]
                if cell["resolution_status"] == "excluded"]
    assert len(excluded) == 222
    assert all(cell["exclusion_reason"] in bra.REASON_VOCABULARY
               for cell in excluded)
