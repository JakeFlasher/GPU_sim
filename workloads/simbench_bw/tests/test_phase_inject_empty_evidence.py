from __future__ import annotations

import sys
from pathlib import Path

import pytest


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

import phase_inject_orchestrator as phase  # noqa: E402


def test_phase_ledger_emission_rejects_zero_evidence(tmp_path: Path) -> None:
    with pytest.raises(
        phase.PhaseLedgerEvidenceMissing,
        match="phase_inject_ledger_evidence_missing",
    ):
        phase.emit_per_launch_phase_ledgers(
            traces_folder=tmp_path / "traces",
            ledger_output_dir=tmp_path / "ledgers",
            slug="fixture",
            shape_point="timing_shape",
            kernel_signature_sha256="1" * 64,
            binary_sha256="2" * 64,
            nvbit_tracer_build_hash="3" * 64,
            claim_scope="measured_local",
            ab_overhead_record={
                "t_wall_no_inject_ns": 100,
                "t_wall_with_inject_ns": 101,
                "overhead_pct": 1.0,
                "admission_threshold_pct": 5.0,
                "admitted": True,
            },
            dual_run_manifest_ref="dual_run_manifest.json",
        )


def test_phase_ab_rejects_unchecked_runtime_before_subprocess(tmp_path: Path) -> None:
    inputs = phase.AbOverheadInputs(
        executable=tmp_path / "target",
        args=[],
        tracer_so_path=tmp_path / "missing-tracer.so",
        traces_folder=tmp_path / "traces",
        repeat_count=1,
    )
    with pytest.raises(
        phase.PhaseInjectRuntimeUnavailable,
        match="phase_inject_runtime_unavailable",
    ):
        phase.run_ab_overhead_pair(inputs, admission_threshold_pct=5.0)
