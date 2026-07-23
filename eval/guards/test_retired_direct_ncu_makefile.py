"""Fail-closed checks for the frozen direct-NCU Makefile."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = (
    REPO_ROOT
    / "workloads"
    / "sol_execbench_baselines"
    / "L1_direct_ncu_binaries"
    / "Makefile"
)


@pytest.mark.parametrize("target", ["probe.o", "probe"])
def test_retired_makefile_disables_builtin_compilation_rules(
    tmp_path: Path, target: str
) -> None:
    make = shutil.which("make")
    assert make is not None, "GNU Make is required by the repository contract"
    (tmp_path / "probe.cc").write_text("int main() { return 0; }\n", encoding="utf-8")

    result = subprocess.run(
        [make, "--no-print-directory", "-n", "-f", str(MAKEFILE), target],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "not an authorized historical inventory action" in combined
    assert not (tmp_path / "probe.o").exists()
    assert not (tmp_path / "probe").exists()


def test_retired_makefile_exposes_only_read_only_inventory_actions(
    tmp_path: Path,
) -> None:
    make = shutil.which("make")
    assert make is not None, "GNU Make is required by the repository contract"
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"preserve me\n")

    for target in ("help", "inventory", "list"):
        result = subprocess.run(
            [make, "--no-print-directory", "-f", str(MAKEFILE), target],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert sentinel.read_bytes() == b"preserve me\n"

    retired_targets = (
        "all",
        "build",
        "clean",
        "capture",
        "harvest",
        "merge",
        "problem_018_fused_rope_with_qk_norm_and_kv_cache_update",
        "problem_071_kv_cache_update_with_rope",
        "unknown-target",
    )
    for target in retired_targets:
        result = subprocess.run(
            [make, "--no-print-directory", "-f", str(MAKEFILE), target],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        combined = result.stdout + result.stderr
        assert result.returncode != 0, combined
        assert (
            "retired and non-executable" in combined
            or "not an authorized historical inventory action" in combined
        )
        assert sentinel.read_bytes() == b"preserve me\n"

    assert sorted(path.name for path in tmp_path.iterdir()) == ["sentinel"]
