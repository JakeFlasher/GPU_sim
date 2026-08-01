#!/usr/bin/env bash
# Run tests that validate immutable, machine-local campaign evidence.

set -euo pipefail

repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
python="$repo_root/.venv/bin/python"
archive_root=${SM86_REFERENCE_CAPTURE_ROOT:-/home/jakeshea/sm86_reference_capture_r27}
recorded_archive_root=/home/jakeshea/sm86_reference_capture_r27

if (($# != 0)); then
    echo "ERROR: campaign evidence gate does not accept pytest arguments" >&2
    exit 2
fi
if [[ ! -x "$python" ]]; then
    echo "ERROR: project Python is unavailable: $python" >&2
    exit 2
fi
if [[ "$archive_root" != "$recorded_archive_root" ]]; then
    echo "ERROR: relocated campaign evidence requires versioned alias authority: $archive_root" >&2
    exit 2
fi
if [[ ! -d "$archive_root" ]]; then
    echo "ERROR: campaign evidence archive is unavailable: $archive_root" >&2
    exit 2
fi

export SM86_REFERENCE_CAPTURE_ROOT="$archive_root"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$repo_root"
exec "$python" -m pytest -q -o addopts= \
    eval/guards/test_legacy_ncu_calibration_frozen_evidence.py::test_frozen_evidence_manifest_and_historical_bytes_are_exact \
    tools/local_labels/test_archive_interrupted_sol_runnability.py::test_fixed_production_archive_is_still_verified \
    tools/local_labels/test_campaign_done_manifest.py::test_only_exact_historical_bootstrap_receives_schema_compatibility \
    tools/local_labels/test_campaign_done_manifest.py::test_historical_bootstrap_strictly_loads_global_preparation_root \
    tools/local_labels/test_campaign_done_manifest.py::test_exact_e387_definition_projection_verifies_with_local_git_blobs \
    tools/local_labels/test_campaign_done_manifest.py::test_only_exact_bootstrap_relocates_former_checkout_to_sealed_git_bytes \
    tools/local_labels/test_campaign_done_manifest.py::test_e387_legacy_definition_context_hashes_exact_git_blobs \
    tools/local_labels/test_capture_roster_authority.py::test_committed_roster_authority_matches_regenerated \
    tools/local_labels/test_simbench_build_witness.py::test_committed_witness_historical_admission \
    tools/local_labels/test_preserved_binaries.py::test_committed_preserved_authorities_are_exact_deterministic_rebuild \
    tools/local_labels/test_sm120_cohort_frozen_at_b1646ef7.py::test_retained_sm120_projection_exactly_rederives_from_b1646ef7 \
    tools/local_labels/test_sm86_capture_matrix_campaign_evidence.py::test_consumer_rederives_external_matrix_and_rejects_resealed_invocation_tamper \
    tools/local_labels/test_sm86_live_capture_authority.py::test_live_authority_exactly_rederives_current_binaries \
    tools/local_labels/test_sm86_capture_matrix_campaign_evidence.py::test_committed_artifact_is_exact_deterministic_rebuild \
    tools/local_labels/test_sm86_capture_matrix_campaign_evidence.py::test_external_authority_path_is_pinned_and_rederivable \
    tools/local_labels/test_sm86_capture_matrix_campaign_evidence.py::test_sol_static_control_toolchain_is_host_verified \
    tools/local_labels/test_sm86_capture_matrix_campaign_evidence.py::test_strided_batched_vendor_control_dispatches_post_trace \
    tools/local_labels/test_sm86_capture_matrix_campaign_evidence.py::test_sol_logical_workload_hashes_and_counts_rederive \
    tools/local_labels/test_sm86_capture_matrix_campaign_evidence.py::test_sol_bridge_rejects_wrong_definition_binding \
    tools/local_labels/test_sm86_capture_matrix_campaign_evidence.py::test_sol_bridge_rejects_reduced_to_native_misroute \
    tools/trace_sampling/tests/test_golden_set_real.py::test_positive_without_ncu_label_fails_closed \
    tools/trace_sampling/tests/test_golden_set_real.py::test_positive_without_ncu_kernel_identity_fails_closed \
    tools/trace_sampling/tests/test_golden_set_real.py::test_positive_with_wrong_ncu_kernel_identity_fails_closed \
    tools/trace_sampling/tests/test_golden_set_real.py::test_authoritative_manifest_bytes_are_policy_pinned \
    tools/trace_sampling/tests/test_golden_set_real.py::test_real_golden_set_passes_all_gates
