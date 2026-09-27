"""Regression tests for the fail-closed PhaseFuse scaling harness."""

from __future__ import annotations

import sys
from pathlib import Path

from workloads.research_probes.phasefuse_scaling import benchmark as benchmark_module
from workloads.research_probes.phasefuse_scaling.benchmark import (
    run_once,
    validate_contract,
)


def _expected() -> dict[str, object]:
    return {
        "case_key": "n4-r1-complete-sorted-seed7",
        "n": 4,
        "region_len": 1,
        "residency": "complete",
        "order": "sorted",
        "seed": 7,
    }


def _contract() -> dict[str, object]:
    return {
        "schema": "phasefuse_scale_v1",
        "case_key": "n4-r8-complete-sorted-seed7",
        "seed": 7,
        "n_records": 4,
        "program_count": 4,
        "region_len": 8,
        "residency": "complete",
        "input_order": "sorted",
        "build_input_ns": 1,
        "api_wall_ns": 1,
        "planner_time_ns": 1,
        "exact_execution_steps": 4,
        "causal_execution_steps": 4,
        "exact_final_cycle": 1,
        "causal_final_cycle": 1,
        "candidate_regions": 0,
        "collapsed_regions": 0,
        "reified_regions": 0,
        "candidate_instructions": 0,
        "collapsed_instructions": 0,
        "terminal_state_matches": True,
        "refusal_reasons": {},
    }


def test_contract_rejects_false_terminal_or_vacuous_refusals() -> None:
    payload = _contract()
    payload["terminal_state_matches"] = False
    payload["reified_regions"] = 1
    assert validate_contract(payload, _expected())


def test_contract_rejects_malformed_numeric_types_without_raising() -> None:
    payload = _contract()
    payload["n_records"] = "four"
    failures = validate_contract(payload, _expected())
    assert any("n_records" in failure for failure in failures)


def test_contract_missing_discriminator_is_reported_without_keyerror() -> None:
    payload = _contract()
    payload.pop("residency")
    failures = validate_contract(payload, _expected())
    assert any("missing fields" in failure for failure in failures)


def test_nonzero_verbose_child_is_a_failure_without_pipe_deadlock() -> None:
    row = run_once(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('x' * 100000); sys.exit(3)",
        ],
        timeout=2.0,
    )
    assert row["ok"] is False
    assert row["exit_code"] == 3
    assert row["stdout_truncated"] is True
    assert row["failures"]


def test_timeout_is_a_failure() -> None:
    row = run_once(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        timeout=0.05,
    )
    assert row["ok"] is False
    assert row["timed_out"] is True
    assert any("timeout" in failure for failure in row["failures"])


def test_zero_exit_without_contract_is_still_a_failure() -> None:
    row = run_once([sys.executable, "-c", "pass"], timeout=2.0, expected=_expected())
    assert row["ok"] is False
    assert any("JSON object" in failure for failure in row["failures"])


def test_main_returns_nonzero_for_a_failed_matrix_case(monkeypatch, tmp_path: Path) -> None:
    child = tmp_path / "failing-child"
    child.write_text("#!/bin/sh\nexit 4\n")
    child.chmod(0o755)
    monkeypatch.setattr(benchmark_module, "SIZES", (4,))
    monkeypatch.setattr(benchmark_module, "REGION_LENGTHS", (1,))
    monkeypatch.setattr(benchmark_module, "RESIDENCIES", ("complete",))
    monkeypatch.setattr(benchmark_module, "ORDERS", ("sorted",))
    monkeypatch.setattr(benchmark_module, "MATRIX_CASES", 1)
    output = tmp_path / "rows.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--binary",
            str(child),
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
    )
    assert benchmark_module.main() == 1
