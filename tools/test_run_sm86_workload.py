from __future__ import annotations

import json
import sys
from pathlib import Path

from run_sm86_workload import run_workload


def _python(*code: str) -> list[str]:
    return [sys.executable, "-c", *code]


def test_pass_records_bounded_outputs_and_status(tmp_path: Path):
    result = run_workload(
        _python("print('ok')"),
        workload_id="fixture_pass",
        output_dir=tmp_path / "pass",
        timeout_seconds=5,
        skip_gpu_preflight=True,
    )
    assert result["status"] == "passed"
    assert result["returncode"] == 0
    assert (tmp_path / "pass/stdout.txt").read_text().strip() == "ok"
    saved = json.loads((tmp_path / "pass/run.json").read_text())
    assert saved["schema"] == "sm86_workload_run_v1"


def test_cuda_oom_is_not_collapsed_into_runtime_error(tmp_path: Path):
    result = run_workload(
        _python("import sys; print('CUDA error: out of memory', file=sys.stderr); sys.exit(1)"),
        workload_id="fixture_oom",
        output_dir=tmp_path / "oom",
        timeout_seconds=5,
        skip_gpu_preflight=True,
    )
    assert result["status"] == "oom"
    assert result["failure_reason"] == "cuda_memory_allocation_failure"


def test_nvbit_memory_allocation_warning_does_not_fake_oom(tmp_path: Path):
    result = run_workload(
        _python("import sys; print(\"Do not call CUDA memory allocation in nvbit_at_ctx_init()\", file=sys.stderr)"),
        workload_id="fixture_nvbit_warning",
        output_dir=tmp_path / "warning",
        timeout_seconds=5,
        skip_gpu_preflight=True,
    )
    assert result["status"] == "passed"


def test_invalid_shape_is_classified_as_out_of_domain(tmp_path: Path):
    result = run_workload(
        _python("import sys; print('shape_override_tile_alignment_violation', file=sys.stderr); sys.exit(2)"),
        workload_id="fixture_ood",
        output_dir=tmp_path / "ood",
        timeout_seconds=5,
        skip_gpu_preflight=True,
    )
    assert result["status"] == "out_of_domain"


def test_timeout_terminates_the_process_group(tmp_path: Path):
    result = run_workload(
        _python("import time; time.sleep(30)"),
        workload_id="fixture_timeout",
        output_dir=tmp_path / "timeout",
        timeout_seconds=1,
        skip_gpu_preflight=True,
    )
    assert result["status"] == "timeout"
    assert result["timed_out"] is True
    assert result["process_group_cleanup"] is True
