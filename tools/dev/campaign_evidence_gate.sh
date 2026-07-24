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
    tools/local_labels/test_archive_interrupted_sol_runnability.py::test_fixed_production_archive_is_still_verified \
    tools/local_labels/test_campaign_done_manifest.py::test_only_exact_historical_bootstrap_receives_schema_compatibility \
    tools/local_labels/test_sm86_capture_matrix.py::test_committed_artifact_is_exact_deterministic_rebuild \
    tools/local_labels/test_sm86_capture_matrix.py::test_external_authority_path_is_pinned_and_rederivable \
    tools/local_labels/test_sm86_capture_matrix.py::test_sol_static_control_toolchain_is_host_verified \
    tools/local_labels/test_sm86_capture_matrix.py::test_strided_batched_vendor_control_dispatches_post_trace \
    tools/local_labels/test_sm86_capture_matrix.py::test_sol_logical_workload_hashes_and_counts_rederive \
    tools/local_labels/test_sm86_capture_matrix.py::test_sol_bridge_rejects_wrong_definition_binding \
    tools/local_labels/test_sm86_capture_matrix.py::test_sol_bridge_rejects_reduced_to_native_misroute
