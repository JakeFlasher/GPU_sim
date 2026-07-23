from __future__ import annotations

import sys
from pathlib import Path

import pytest


_SOL_KB_ROOT = Path(__file__).resolve().parent
if str(_SOL_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOL_KB_ROOT))

import generate_dossiers  # noqa: E402


def test_active_text_keeps_cutlass_roots_distinct() -> None:
    source = (
        "baselines/FIB/problem_001_fused_add_rmsnorm_h2048/cutlass/; "
        "sol-execbench/examples/cutlass/; "
        "cutlass/include/cutlass/gemm/"
    )

    assert generate_dossiers._active_text(source) == (
        "workloads/sol_execbench_baselines/FIB/"
        "problem_001_fused_add_rmsnorm_h2048/cutlass/; "
        "tools/sol_execbench/examples/cutlass/; "
        "third_party/cutlass/include/cutlass/gemm/"
    )


def test_dossier_cutlass_contract_fails_without_asserts(tmp_path: Path) -> None:
    (tmp_path / "problem_bad.md").write_text(
        "workloads/sol_execbench_baselines/FIB/problem_bad/"
        "third_party/cutlass/\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="solution-local CUTLASS"):
        generate_dossiers._assert_dossier_cutlass_path_contract(tmp_path)
