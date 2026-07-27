"""Owner-gated live matrix rederivation checks."""

from __future__ import annotations

import pytest

from tools.local_labels import test_sm86_capture_chunk as chunk_tests
from tools.local_labels import test_sm86_capture_matrix as matrix_tests


pytestmark = pytest.mark.campaign_evidence


def test_committed_artifact_is_exact_deterministic_rebuild() -> None:
    matrix_tests._campaign_committed_artifact_is_exact_deterministic_rebuild(
        matrix_tests._committed_matrix()
    )


def test_external_authority_path_is_pinned_and_rederivable(tmp_path) -> None:
    matrix_tests._campaign_external_authority_path_is_pinned_and_rederivable(
        tmp_path
    )


def test_sol_static_control_toolchain_is_host_verified() -> None:
    matrix_tests._campaign_sol_static_control_toolchain_is_host_verified()


def test_strided_batched_vendor_control_dispatches_post_trace() -> None:
    matrix_tests._campaign_strided_batched_vendor_control_dispatches_post_trace(
        matrix_tests._committed_matrix()
    )


def test_sol_logical_workload_hashes_and_counts_rederive() -> None:
    matrix_tests._campaign_sol_logical_workload_hashes_and_counts_rederive(
        matrix_tests._committed_matrix()
    )


def test_sol_bridge_rejects_wrong_definition_binding(monkeypatch) -> None:
    matrix_tests._campaign_sol_bridge_rejects_wrong_definition_binding(monkeypatch)


def test_sol_bridge_rejects_reduced_to_native_misroute() -> None:
    matrix_tests._campaign_sol_bridge_rejects_reduced_to_native_misroute()


def test_consumer_rederives_external_matrix_and_rejects_resealed_invocation_tamper(
    tmp_path,
) -> None:
    chunk_tests._campaign_consumer_rederives_external_matrix_and_rejects_resealed_invocation_tamper(
        tmp_path
    )
