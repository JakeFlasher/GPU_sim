from __future__ import annotations

from evaluate_reification_design import summarize_design_runs, summarize_selective_coverage


def _document(terminal: int, service_lower: int, predicted_upper: int | None = None):
    return {
        "launches": [{"timing": {
            "diagnostic_last_issue_cycles": 10,
            "last_processed_modeled_system_event_time": None,
            "terminal_modeled_event_time": terminal,
            "completion_frontier": {"lower": terminal, "upper": terminal},
            "readiness": {"status": "complete"},
        }}],
        "latent_envelope": {
            "launches": [{
                "frontiers": {"resource_service": {"lower": service_lower, "upper": predicted_upper}},
                "resource_service_projection": {
                    "lower_cycles": service_lower,
                    "upper_cycles": predicted_upper,
                },
                "ncu_projection": {
                    "simulator_lower_cycles": service_lower,
                    "simulator_upper_cycles": predicted_upper,
                    "predicted_lower_cycles": service_lower,
                    "predicted_upper_cycles": predicted_upper,
                    "role": "ncu_observer_elapsed_interval_not_hardware_endpoint_identity",
                },
            }]
        },
    }


def test_design_report_ranks_model_local_service_variants_and_checks_native_invariance():
    detailed = _document(785, 0)
    report = summarize_design_runs(
        detailed,
        [("slow", _document(785, 200, 250)), ("fast", _document(785, 100, 150))],
    )
    assert report["native_replay_terminal_invariant"] is True
    assert report["ranking"] == [["fast", "slow"]]
    assert report["hardware_comparison"] == "not_run_ncu_remains_independent_target"
    assert report["architectural_conclusion"] == "model_local_partial_order_only_declared_native_projection_unchanged"


def test_design_report_detects_native_change_instead_of_claiming_a_ranking():
    detailed = _document(785, 0)
    report = summarize_design_runs(detailed, [("changed", _document(900, 100, None))])
    assert report["native_replay_terminal_invariant"] is False
    assert report["architectural_conclusion"] == "native_replay_changed_reify_or_investigate"


def test_design_report_does_not_treat_missing_native_projection_as_invariant():
    detailed = {"launches": [{"timing": {}}]}
    report = summarize_design_runs(detailed, [("unknown", _document(785, 100, None))])
    assert report["native_projection_complete"] is False
    assert report["native_replay_terminal_invariant"] is False
    assert report["architectural_conclusion"] == "native_projection_unknown_exact_replay_required"


def test_selective_coverage_reports_zero_compression_without_fabricating_ratio():
    coverage = summarize_selective_coverage({
        "coverage": {
            "trace_records": 54,
            "candidate_records": 0,
            "summarized_records": 0,
            "reified_records": 0,
            "uncovered_records": 54,
        }
    })
    assert coverage["summarized_fraction"] == 0.0
    assert "compression_ratio" not in coverage
    assert coverage["step_reduction"] is None


def test_selective_coverage_separates_record_coverage_from_execution_step_reduction():
    coverage = summarize_selective_coverage({
        "exact_execution_steps": 16,
        "summarized_execution_steps": 1,
        "macro_dispatches": 1,
        "coverage": {
            "trace_records": 16,
            "candidate_records": 16,
            "summarized_records": 16,
            "reified_records": 0,
            "uncovered_records": 0,
        },
    })
    assert coverage["record_to_summary_ratio"] == 1.0
    assert coverage["step_reduction"] == 16.0

    assert coverage["macro_dispatches"] == 1
    assert coverage["macro_execution_observed"] is True
