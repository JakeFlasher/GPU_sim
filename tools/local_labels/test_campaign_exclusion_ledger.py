"""Tests for the typed-exclusion authority ledger + its coverage invariant."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jsonschema

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.local_labels import campaign_exclusion_ledger as cel  # noqa: E402

_SCHEMA = json.loads(
    (_REPO / "workloads" / "simbench_bw" / "schemas"
     / "campaign_exclusion_ledger.schema.json").read_text()
)


_FROZEN = (_REPO / "data/local_labels/campaign/"
           "capture_roster_authority_sm86.json")
_FROZEN_SHA256 = (
    "d02c817091b0a9f06c2106c0987c59a481910afcc3b78ab1de7a7d5bd5a6b4b4")
_HISTORICAL_LEDGER_PIN = (
    "7a6c5b8ba7dae431c24eb3da1acb58cbd34b264d25ed8570e3b6c6e0049e994d")


def _frozen_authority():
    payload = _FROZEN.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == _FROZEN_SHA256
    authority = json.loads(payload)
    assert authority["roster_id"] == "sm86_capture_roster_authority_v1"
    return authority


def _ledger():
    authority = _frozen_authority()
    return authority, cel.build_ledger(
        authority, _HISTORICAL_LEDGER_PIN, _REPO)


def test_schema_valid():
    _, ledger = _ledger()
    jsonschema.validate(ledger, _SCHEMA)


def test_coverage_identity_and_cross_check():
    authority, ledger = _ledger()
    cov = ledger["coverage"]
    assert cov["identity_holds"] is True
    assert cov["untyped_excluded_rows"] == []
    # Ledger total must equal the roster authority's own excluded tally.
    assert ledger["counts"]["total_exclusions"] == authority["counts"]["excluded"]
    # universe == runnable + pending_build + pending_reconstitution
    #            + pending_shrink + Σ(exclusions)
    assert cov["universe"] == (
        cov["runnable"] + cov["pending_build"]
        + cov["pending_reconstitution"] + cov.get("pending_shrink", 0)
        + cov["typed_exclusions"]
    )


def test_arch_gate_base_plus_sol_runnability_exclusions():
    authority, ledger = _ledger()
    by_reason = ledger["counts"]["by_reason"]
    # The arch/dtype base is fixed (simbench arch-gated 61 + SOL dtype-gated 34).
    assert by_reason["arch_gate"] == 95
    # Every taxonomy member is present as a key even when empty.
    assert set(by_reason) == set(cel.REASON_VOCABULARY)
    # Any exclusions beyond the arch base are the empirical SOL host-runnability
    # typed deferrals; they reconcile exactly to the authority's excluded tally
    # and route only to compile/correctness/host-capacity reasons.
    extra = sum(v for k, v in by_reason.items() if k != "arch_gate")
    assert extra == authority["counts"]["excluded"] - 95
    allowed_sol = {"calibration_inconclusive", "correctness_failed",
                   "host_capacity_deferred"}
    for reason, count in by_reason.items():
        if reason != "arch_gate" and count > 0:
            assert reason in allowed_sol, (reason, count)


def test_committed_ledger_matches_regenerated():
    """The committed ledger must not go stale relative to the current code +
    roster authority (by_reason + coverage identity + typed count)."""
    committed_path = (_REPO / "data" / "local_labels" / "campaign"
                      / "campaign_exclusion_ledger_sm86.json")
    if not committed_path.is_file():
        import pytest
        pytest.skip("no committed exclusion ledger artifact")
    committed = json.loads(committed_path.read_text())
    authority = _frozen_authority()
    fresh = cel.build_ledger(authority, _HISTORICAL_LEDGER_PIN, _REPO)
    assert committed["counts"]["by_reason"] == fresh["counts"]["by_reason"], \
        "committed ledger reason breakdown is stale"
    assert committed["coverage"]["typed_exclusions"] == fresh["coverage"]["typed_exclusions"]
    assert committed["coverage"]["identity_holds"] is True
    assert fresh["coverage"]["identity_holds"] is True


def test_untyped_excluded_row_breaks_identity():
    """An excluded cell whose reason is outside the taxonomy must fail-closed:
    it is not counted, it is listed as untyped, and the identity is False."""
    authority = {
        "arch": "sm_86", "sku": "rtx_3090_ti", "roster_id": "x",
        "counts": {"universe": 2, "runnable": 1, "pending_build": 0,
                   "pending_reconstitution": 0, "excluded": 1},
        "cells": [
            {"roster_source": "simbench_bw", "slug": "ok",
             "resolution_status": "runnable"},
            {"roster_source": "simbench_bw", "slug": "bad",
             "resolution_status": "excluded", "exclusion_reason": "not_a_reason"},
        ],
    }
    ledger = cel.build_ledger(authority, "0" * 64, _REPO)
    assert ledger["coverage"]["identity_holds"] is False
    assert ledger["coverage"]["untyped_excluded_rows"]
    assert ledger["counts"]["total_exclusions"] == 0
