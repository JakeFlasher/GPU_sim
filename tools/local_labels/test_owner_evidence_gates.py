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
    assert 'export SM86_REFERENCE_CAPTURE_ROOT="$archive_root"' in script
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in script

    expected_node_ids = (
        "eval/guards/test_legacy_ncu_calibration_frozen_evidence.py::"
        "test_frozen_evidence_manifest_and_historical_bytes_are_exact",
        "tools/local_labels/test_archive_interrupted_sol_runnability.py::"
        "test_fixed_production_archive_is_still_verified",
        "tools/local_labels/test_campaign_done_manifest.py::"
        "test_only_exact_historical_bootstrap_receives_schema_compatibility",
        "tools/local_labels/test_campaign_done_manifest.py::"
        "test_historical_bootstrap_strictly_loads_global_preparation_root",
        "tools/local_labels/test_campaign_done_manifest.py::"
        "test_exact_e387_definition_projection_verifies_with_local_git_blobs",
        "tools/local_labels/test_campaign_done_manifest.py::"
        "test_only_exact_bootstrap_relocates_former_checkout_to_sealed_git_bytes",
        "tools/local_labels/test_campaign_done_manifest.py::"
        "test_e387_legacy_definition_context_hashes_exact_git_blobs",
        "tools/local_labels/test_capture_roster_authority.py::"
        "test_committed_roster_authority_matches_regenerated",
        "tools/local_labels/test_simbench_build_witness.py::"
        "test_committed_witness_historical_admission",
        "tools/local_labels/test_preserved_binaries.py::"
        "test_committed_preserved_authorities_are_exact_deterministic_rebuild",
        "tools/local_labels/test_sm86_capture_chunk.py::"
        "test_consumer_rederives_external_matrix_and_rejects_resealed_invocation_tamper",
        "tools/local_labels/test_sm86_capture_matrix.py::"
        "test_committed_artifact_is_exact_deterministic_rebuild",
        "tools/local_labels/test_sm86_capture_matrix.py::"
        "test_external_authority_path_is_pinned_and_rederivable",
        "tools/local_labels/test_sm86_capture_matrix.py::"
        "test_sol_static_control_toolchain_is_host_verified",
        "tools/local_labels/test_sm86_capture_matrix.py::"
        "test_strided_batched_vendor_control_dispatches_post_trace",
        "tools/local_labels/test_sm86_capture_matrix.py::"
        "test_sol_logical_workload_hashes_and_counts_rederive",
        "tools/local_labels/test_sm86_capture_matrix.py::"
        "test_sol_bridge_rejects_wrong_definition_binding",
        "tools/local_labels/test_sm86_capture_matrix.py::"
        "test_sol_bridge_rejects_reduced_to_native_misroute",
    )
    observed_node_ids = tuple(
        line.rstrip().removesuffix("\\").strip()
        for line in script.splitlines()
        if "::" in line
    )
    assert observed_node_ids == expected_node_ids
