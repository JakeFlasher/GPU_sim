from __future__ import annotations

import json
from pathlib import Path

import pytest

from capture_sm86_ncu_identity import capture_identity


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads((ROOT / "docs/sm86_ncu_accuracy_protocol_v1.json").read_text())


def _runner(*, compute_capability: str = "8.6", metric: bool = True):
    def run(command):
        command = tuple(command)
        if command[0] == "nvidia-smi":
            return 0, f"NVIDIA GeForce RTX 3090 Ti,GPU-test,{compute_capability},615.71.09,0,1350,10501,P0\n", ""
        if command[:2] == ("ncu", "--query-metrics"):
            return 0, ("gpc__cycles_elapsed Counter cycle cycles\n" if metric else ""), ""
        if command[:2] == ("ncu", "--version"):
            return 0, "Version 2026.1.0.0 (build test)\n", ""
        if command[0] == "nvcc":
            return 0, "Cuda compilation tools, release 13.4, V13.4.92\n", ""
        if command[0] == "rustc":
            return 0, "rustc 1.96.1 (fixture)\n", ""
        if command[:3] == ("git", "rev-parse", "HEAD"):
            return 0, "fixture-commit\n", ""
        if command[:2] == ("git", "status"):
            return 0, "", ""
        raise AssertionError(f"unexpected metadata command: {command}")

    return run


def _paths(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for field in ("simulator_binary", "trace", "control", "arch_card", "pchase_card"):
        path = tmp_path / field
        path.write_text("fixture")
        paths[field] = path
    return paths


def test_identity_capture_is_metadata_only_and_eligible_when_all_fields_match(tmp_path: Path):
    paths = _paths(tmp_path)
    result = capture_identity(
        _protocol(),
        **paths,
        nvbit_tracer_build="nvbit-fixture",
        simulator_build_profile="release",
        trace_schema_version="fixture-trace-v1",
        runner=_runner(),
    )
    assert result["status"] == "independently_verified"
    assert result["eligible_for_accuracy"] is True
    assert result["hardware"]["gpu_uuid"] == "GPU-test"
    assert result["measurement_contract"]["replay_mode"] == "application"
    assert result["measurement_contract"]["cache_policy"] == "all"
    assert result["metric_availability"]["available"] is True
    assert result["mismatches"] == []


def test_identity_mismatch_is_labeled_ineligible_without_collecting_data(tmp_path: Path):
    paths = _paths(tmp_path)
    result = capture_identity(
        _protocol(),
        **paths,
        nvbit_tracer_build="nvbit-fixture",
        simulator_build_profile="release",
        trace_schema_version="fixture-trace-v1",
        runner=_runner(compute_capability="7.5", metric=False),
    )
    assert result["status"] == "mismatch"
    assert result["eligible_for_accuracy"] is False
    assert any(reason.startswith("compute_capability:") for reason in result["mismatches"])
    assert "target_metric_unavailable_or_query_failed" in result["mismatches"]
