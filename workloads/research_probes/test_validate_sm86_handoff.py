"""Regression checks for the generated SM86 handoff consistency contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workloads.research_probes import validate_sm86_handoff
from workloads.research_probes.validate_sm86_handoff import validate


ROOT = Path("results/captures/sm86_simulator_live_baseline_20260829_v2")


def test_current_handoff_matches_closure_and_manifest() -> None:
    validate()


def test_readme_residual_mutation_is_rejected(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    text = text.replace("| cp_async_canary | 21013 |", "| cp_async_canary | 21014 |", 1)
    readme.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="README residual table/cp_async_canary"):
        validate(readme_path=readme)


def test_summary_metric_mutation_is_rejected(tmp_path: Path) -> None:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    summary["cohort"][0]["ncu_gpc_cycles"] += 1
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="summary/cp_async_canary/ncu_gpc_cycles"):
        validate(summary_path=path)


def test_manifest_cp_async_limitation_mutation_is_rejected() -> None:
    manifest = json.loads((ROOT / "campaign_manifest.json").read_text(encoding="utf-8"))
    manifest["cells"][0]["limitations"][0] = (
        "Trace binary uses 64 groups while the NCU target uses 16384 groups."
    )
    with pytest.raises(ValueError, match="limitation text"):
        validate_sm86_handoff._validate_manifest(manifest)


def test_summary_correctness_group_pairing_mutation_is_rejected(tmp_path: Path) -> None:
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    summary["correctness_capture"]["matched_cp_async"]["group_count"] = 16_384
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="matched_cp_async"):
        validate(summary_path=path)
