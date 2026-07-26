"""Static regression checks for the Q816 wording-only correction.

These checks deliberately inspect only the active Q816 files. They protect the
current targets-never-features and evidence-namespace wording without treating
runtime guard identifiers as obsolete prose.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# These files had repository-wide-ban or retired local-exception prose. Keep the
# list explicit: calibration, historical, and harness documentation have
# different ownership and are not silently swept into this wording change.
_CONTRADICTION_FILES = (
    "tools/local_labels/label_driver.sh",
    "tools/local_labels/capture_count_labels.py",
    "tools/local_labels/probe_ncu_metric_availability.py",
    "tools/local_labels/run_headline_cohort_sweep.py",
    "tools/local_labels/activation_trace_orchestrator.py",
    "tools/sol_baselines_scaffold/validate.py",
    "tools/sm_86_predictor_accuracy_recovery_v2/generate_ac_compliance_report.py",
)

_RETIRED_REPOSITORY_BAN_PHRASES = (
    "cloud/server contract still forbids ncu",
    "ncu is invoked only",
    "ncu invoked only here",
    "ncu is forbidden on every device",
    "ncu quarantine + local-label exception",
    "per claude.md rule 1 local-label exception",
    "local-label exception scope",
)

_NAMESPACE_FILES = (
    "tools/local_labels/capture_count_labels.py",
    "tools/local_labels/capture_t2_stream_labels.py",
    "tools/local_labels/orchestrate_sm86_label_recapture.py",
    "tools/local_labels/run_observation_contract_microfixtures.py",
    "tools/local_labels/run_sm86_label_recapture_sweep.py",
    "tools/local_labels/sweep_repeatability_matrix.py",
)

_CLOUD_FILES = (
    "tools/cloud/container_bootstrap.sh",
    "tools/cloud/preflight_manifest.py",
)

_ATTRIBUTION_FILES = (
    "tools/cache_replay/l1_filtered_l2_eval.py",
    "tools/trace_sampling/counting_pass.py",
    "tools/local_labels/calibrate_nvml_reference_operating_point.py",
    "tools/local_labels/calibrate_nvml_short_burst_anchor.py",
    "tools/local_labels/canary/emit_canary_manifest.py",
    "tools/local_labels/expand_cohort_manifest_from_catalog.py",
    "tools/local_labels/fit_calibration_head.py",
    "tools/local_labels/inventory_cohort_candidates.py",
    "tools/local_labels/label_supersession.py",
    "tools/local_labels/live_sweep_completeness.py",
    "tools/local_labels/run_headline_cohort_predict_compare.py",
    "tools/local_labels/sol_reconstitution.py",
    "tools/local_labels/trace_register_cap_variant_cells.py",
    "tools/local_labels/validate_static_occupancy_agreement.py",
    "tools/sm_86_predictor_accuracy_recovery_v2/generate_ac_compliance_report.py",
)

_RETIRED_ATTRIBUTION_PHRASES = (
    "rule 1 (ncu quarantine)",
    "rule 1: no ncu metric",
    "per claude.md rule 1",
    "local-label exception boundary",
    "rule-1 clean",
    "no rule-1 surface",
)

_RETIRED_NAMESPACE_PHRASES = (
    "rule-1 quarantine",
    "critical rule 1 local-label exception",
    "quarantined .ncu-rep",
    "quarantined sub-prefix",
    "quarantine path directly",
    "sm_86 quarantine",
)

# Wording changes must not remove the adjacent fail-closed or authority guards.
_GUARD_TOKENS: dict[str, tuple[str, ...]] = {
    "tools/local_labels/label_driver.sh": (
        "EXPLICIT_NCU_LAUNCH_CONTROLS",
        "NCU_AUTHORITY_INVALID",
        "NCU_EXECUTABLE_SHA256",
    ),
    "tools/local_labels/capture_count_labels.py": (
        "QuarantinePathError",
        "assert_quarantine_path",
    ),
    "tools/local_labels/probe_ncu_metric_availability.py": (
        "assert_target_metric_available",
        "MetricNotProbeAvailableError",
    ),
    "tools/local_labels/capture_t2_stream_labels.py": (
        "QuarantinePathError",
        "capture_and_land",
    ),
    "tools/local_labels/orchestrate_sm86_label_recapture.py": (
        "_LABEL_DRIVER",
        "subprocess.run",
    ),
    "tools/local_labels/sweep_repeatability_matrix.py": (
        "_QUARANTINE_PREFIX",
        "path.resolve",
    ),
    "tools/local_labels/run_headline_cohort_sweep.py": (
        "sweep_repeatability_matrix.py",
        "preflight",
    ),
    "tools/local_labels/activation_trace_orchestrator.py": (
        "assert_write_allowed",
        "require_trace_admission",
    ),
    "tools/sol_baselines_scaffold/validate.py": (
        "_check_forbidden_surfaces",
        "nv-nsight-cu-cli",
    ),
    "tools/sm_86_predictor_accuracy_recovery_v2/generate_ac_compliance_report.py": (
        "static_no_root_or_ncu_guard",
        "target_test",
    ),
}


def _text(relative_path: str) -> str:
    # Keep this test rooted in the checkout; in particular, never walk or probe
    # unrelated workspace trees while checking the wording boundary.
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8").casefold()


def test_q816_files_drop_retired_repository_ban_phrases() -> None:
    for relative_path in _CONTRADICTION_FILES:
        text = _text(relative_path)
        found = [phrase for phrase in _RETIRED_REPOSITORY_BAN_PHRASES if phrase in text]
        assert not found, f"{relative_path} retains retired wording: {found}"


def test_q816_attribution_files_drop_retired_rule_phrases() -> None:
    for relative_path in _ATTRIBUTION_FILES:
        text = _text(relative_path)
        found = [phrase for phrase in _RETIRED_ATTRIBUTION_PHRASES if phrase in text]
        assert not found, f"{relative_path} retains retired attribution: {found}"


def test_q816_namespace_files_drop_retired_quarantine_attribution() -> None:
    for relative_path in _NAMESPACE_FILES:
        text = _text(relative_path)
        found = [phrase for phrase in _RETIRED_NAMESPACE_PHRASES if phrase in text]
        assert not found, f"{relative_path} retains retired namespace wording: {found}"


def test_q816_cloud_files_keep_capability_scope_explicit() -> None:
    bootstrap = _text(_CLOUD_FILES[0])
    preflight = _text(_CLOUD_FILES[1])
    assert "does not install or invoke ncu" in bootstrap
    assert "separately authorized" in preflight
    assert "host preflight" in preflight
    for text in (bootstrap, preflight):
        assert "ncu is denied in cloud containers" not in text
        assert "cloud gpu host denies hardware-counter access" not in text


def test_q816_direct_ncu_callers_do_not_claim_descriptor_dispatch() -> None:
    for relative_path in (
        "tools/local_labels/capture_count_labels.py",
        "tools/local_labels/capture_t2_stream_labels.py",
        "tools/local_labels/probe_ncu_metric_availability.py",
        "tools/local_labels/run_observation_contract_microfixtures.py",
    ):
        text = _text(relative_path)
        assert "subprocess.run" in text or "capture_and_land" in text
        assert "descriptor-bound label driver" not in text


def test_q816_files_retain_authority_and_fail_closed_guards() -> None:
    for relative_path, tokens in _GUARD_TOKENS.items():
        text = _text(relative_path)
        missing = [token for token in tokens if token.casefold() not in text]
        assert not missing, f"{relative_path} lost guard tokens: {missing}"


def test_q816_scope_excludes_forbidden_latex_tree() -> None:
    paths = (
        *_CONTRADICTION_FILES,
        *_NAMESPACE_FILES,
        *_CLOUD_FILES,
        *_ATTRIBUTION_FILES,
        *_GUARD_TOKENS,
    )
    assert all("IISWC_conference_latex" not in path for path in paths)
