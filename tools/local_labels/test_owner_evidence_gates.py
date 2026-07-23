"""Fail-closed contracts for machine-local evidence owner gates."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_GATE = REPO_ROOT / "tools/dev/campaign_evidence_gate.sh"


def _run_gate(*arguments: str, environment: dict[str, str] | None = None):
    return subprocess.run(
        ["/usr/bin/bash", os.fspath(CAMPAIGN_GATE), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )


def test_campaign_gate_refuses_pytest_argument_injection():
    result = _run_gate("--collect-only")

    assert result.returncode == 2
    assert "does not accept pytest arguments" in result.stderr


def test_campaign_gate_refuses_unadmitted_replica_root(tmp_path: Path):
    environment = os.environ.copy()
    environment["SM86_REFERENCE_CAPTURE_ROOT"] = os.fspath(tmp_path)
    result = _run_gate(environment=environment)

    assert result.returncode == 2
    assert "requires versioned alias authority" in result.stderr


def test_campaign_gate_pins_interpreter_and_exact_owner_tests():
    script = CAMPAIGN_GATE.read_text(encoding="utf-8")

    assert 'python="$repo_root/.venv/bin/python"' in script
    assert "${PYTHON" not in script
    assert '"$@"' not in script
    assert "-m campaign_evidence" not in script
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in script
    for node_id in (
        "test_archive_interrupted_sol_runnability.py::"
        "test_fixed_production_archive_is_still_verified",
        "test_campaign_done_manifest.py::"
        "test_only_exact_historical_bootstrap_receives_schema_compatibility",
    ):
        assert script.count(node_id) == 1
