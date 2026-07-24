from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path

import jsonschema
import pytest

from tools.local_labels import (
    build_cutlass_successor_m6_determinism_schedule as schedule_builder,
)
from tools.local_labels import cutlass_successor_m6_determinism_analysis as analysis
from tools.local_labels import cutlass_successor_m6_observation_journal as journal
from tools.local_labels import m6_determinism_campaign as m6
from tools.local_labels.test_cutlass_successor_m6_observation_journal import (
    GENERATED_AT,
    STATIC_FILE_HASHES,
    _observations,
    _rechain,
    _reseal_journal,
    _selected,
    _tier1_requests,
    full_journal,
    successor_contract,
)

Q710_FILE_HASHES = {
    "tools/local_labels/cutlass_successor_m6_observation_journal.py":
        "2133d6c8f05bb62bce44f5b353dd544b60e5130ed11b79823a522e0f9d3a014f",
    "workloads/simbench_bw/schemas/"
    "cutlass_successor_m6_observation_journal.schema.json":
        "85b5b9ddee4ed54300553bfeed93f4d94c538e52e5888b0775c7442d67e70b1c",
}


def _terminal_requests(schedule, policy, outcomes):
    requests = [{"record_kind": "epoch_open"}]
    slots = sorted(schedule["slots"], key=lambda row: row["ordinal"])
    for slot, outcome in zip(slots, outcomes):
        requests.append(
            {
                "record_kind": "attempt_started",
                "ordinal": slot["ordinal"],
                "retry_of_attempt_id": None,
            }
        )
        if outcome == "observed":
            requests.append(
                {
                    "record_kind": "attempt_terminal",
                    "outcome": "observed",
                    "ncu_replay_passes": 1,
                    "observations": _observations(
                        policy,
                        repeat_index=slot["repeat_index"],
                    ),
                }
            )
        else:
            requests.append(
                {
                    "record_kind": "attempt_terminal",
                    "outcome": "excluded_typed",
                    "ncu_replay_passes": 1,
                    "typed_exclusion": {
                        "reason": "profile_failed",
                        "detail": "synthetic Q712 contract fixture",
                    },
                }
            )
    requests.extend(
        [
            {"record_kind": "checkpoint"},
            {"record_kind": "epoch_closed"},
        ]
    )
    return requests


def _complete_no_trigger_requests(schedule, policy):
    requests = _tier1_requests(schedule, policy)
    requests.extend(
        {"record_kind": "trigger_decision", "cell_id": cell["cell_id"]}
        for cell in schedule["cells"]
    )
    requests.extend(
        [
            {"record_kind": "tier2_activated"},
            {"record_kind": "checkpoint"},
            {"record_kind": "epoch_closed"},
        ]
    )
    return requests


def _all_excluded_requests(schedule):
    requests = [{"record_kind": "epoch_open"}]
    for slot in sorted(schedule["slots"], key=lambda row: row["ordinal"]):
        requests.extend(
            [
                {
                    "record_kind": "attempt_started",
                    "ordinal": slot["ordinal"],
                    "retry_of_attempt_id": None,
                },
                {
                    "record_kind": "attempt_terminal",
                    "outcome": "excluded_typed",
                    "ncu_replay_passes": 1,
                    "typed_exclusion": {
                        "reason": "profile_failed",
                        "detail": "synthetic all-excluded Q712 fixture",
                    },
                },
            ]
        )
    requests.extend(
        [
            {"record_kind": "tier1_closed"},
            {"record_kind": "checkpoint"},
            {"record_kind": "epoch_closed"},
        ]
    )
    return requests


def _analyze(schedule, source_journal):
    """Exercise the approved API while preserving independent pre-amendment failures."""
    if "generated_at_utc" in inspect.signature(analysis.analyze).parameters:
        return analysis.analyze(
            schedule,
            source_journal,
            generated_at_utc=GENERATED_AT,
        )
    return analysis.analyze(schedule, source_journal)


def _reseal_analysis(document):
    document["analysis_sha256"] = m6.canonical_sha256(
        {key: value for key, value in document.items() if key != "analysis_sha256"}
    )


def _classification(document, *, cell_id, config_id, metric_key):
    return next(
        row
        for row in document["classifications"]
        if row["cell_id"] == cell_id
        and row["config_id"] == config_id
        and row["metric_key"] == metric_key
    )


@pytest.fixture(scope="module")
def empty_analysis(successor_contract):
    return _analyze(
        successor_contract["schedule"],
        successor_contract["empty"],
    )


@pytest.fixture(scope="module")
def partial_journal(successor_contract):
    return journal.build_journal(
        successor_contract["schedule"],
        _terminal_requests(
            successor_contract["schedule"],
            successor_contract["policy"],
            ["observed"],
        ),
        created_at_utc=GENERATED_AT,
    )


@pytest.fixture(scope="module")
def partial_analysis(successor_contract, partial_journal):
    return _analyze(successor_contract["schedule"], partial_journal)


@pytest.fixture(scope="module")
def mixed_journal(successor_contract):
    return journal.build_journal(
        successor_contract["schedule"],
        _terminal_requests(
            successor_contract["schedule"],
            successor_contract["policy"],
            ["observed", "excluded_typed"],
        ),
        created_at_utc=GENERATED_AT,
    )


@pytest.fixture(scope="module")
def mixed_analysis(successor_contract, mixed_journal):
    return _analyze(successor_contract["schedule"], mixed_journal)


@pytest.fixture(scope="module")
def no_trigger_journal(successor_contract):
    return journal.build_journal(
        successor_contract["schedule"],
        _complete_no_trigger_requests(
            successor_contract["schedule"],
            successor_contract["policy"],
        ),
        created_at_utc=GENERATED_AT,
    )


@pytest.fixture(scope="module")
def no_trigger_analysis(successor_contract, no_trigger_journal):
    return _analyze(successor_contract["schedule"], no_trigger_journal)


@pytest.fixture(scope="module")
def triggered_pending_analysis(successor_contract, full_journal):
    return _analyze(successor_contract["schedule"], full_journal)


@pytest.fixture(scope="module")
def all_excluded_analysis(successor_contract):
    excluded = journal.build_journal(
        successor_contract["schedule"],
        _all_excluded_requests(successor_contract["schedule"]),
        created_at_utc=GENERATED_AT,
    )
    return _analyze(successor_contract["schedule"], excluded)


@pytest.fixture(scope="module")
def oor_contract(successor_contract):
    schedule = successor_contract["schedule"]
    policy = successor_contract["policy"]
    metric = next(
        row
        for row in _selected(policy).values()
        if row["valid_max"] is not None
    )
    value = int(float(metric["valid_max"])) + 1
    target = next(
        slot
        for slot in sorted(schedule["slots"], key=lambda row: row["ordinal"])
        if slot["config_id"] == journal.REFERENCE_CONFIG
    )
    requests = [{"record_kind": "epoch_open"}]
    for slot in sorted(schedule["slots"], key=lambda row: row["ordinal"]):
        if slot["ordinal"] > target["ordinal"]:
            break
        requests.append(
            {
                "record_kind": "attempt_started",
                "ordinal": slot["ordinal"],
                "retry_of_attempt_id": None,
            }
        )
        if slot["ordinal"] == target["ordinal"]:
            requests.append(
                {
                    "record_kind": "attempt_terminal",
                    "outcome": "observed",
                    "ncu_replay_passes": 1,
                    "observations": _observations(
                        policy,
                        repeat_index=slot["repeat_index"],
                        override_key=metric["key"],
                        override_value=value,
                    ),
                }
            )
        else:
            requests.append(
                {
                    "record_kind": "attempt_terminal",
                    "outcome": "excluded_typed",
                    "ncu_replay_passes": 1,
                    "typed_exclusion": {
                        "reason": "profile_failed",
                        "detail": "synthetic prefix before OOR fixture",
                    },
                }
            )
    requests.extend(
        [
            {"record_kind": "checkpoint"},
            {"record_kind": "epoch_closed"},
        ]
    )
    source = journal.build_journal(
        schedule,
        requests,
        created_at_utc=GENERATED_AT,
    )
    return {
        "journal": source,
        "analysis": _analyze(schedule, source),
        "metric": metric,
        "target": target,
        "value": value,
    }


def test_schema_is_closed_and_authorities_remain_byte_exact(successor_contract):
    schema = json.loads((analysis.ROOT / analysis.SCHEMA_REL).read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    for definition in schema["$defs"].values():
        if definition.get("type") == "object":
            assert definition.get("additionalProperties") is False

    for relative, expected in {**STATIC_FILE_HASHES, **Q710_FILE_HASHES}.items():
        assert m6.sha256_bytes((analysis.ROOT / relative).read_bytes()) == expected
    assert successor_contract["schedule"]["execution_dispatch"][
        "schedule_is_executable"
    ] is False


def test_public_analyze_api_requires_explicit_generation_timestamp():
    parameters = inspect.signature(analysis.analyze).parameters
    assert list(parameters) == ["schedule", "journal", "generated_at_utc"]
    assert parameters["generated_at_utc"].kind is inspect.Parameter.KEYWORD_ONLY


def test_analysis_source_and_schema_bytes_are_bound(empty_analysis):
    assert empty_analysis["generated_at_utc"] == GENERATED_AT
    source_identity = empty_analysis["analysis_source_identity"]
    expected = {
        "analyzer": "tools/local_labels/cutlass_successor_m6_determinism_analysis.py",
        "schema": str(analysis.SCHEMA_REL),
    }
    assert set(source_identity) == set(expected)
    for key, relative in expected.items():
        reference = source_identity[key]
        payload = (analysis.ROOT / relative).read_bytes()
        assert reference == {
            "path": relative,
            "sha256": m6.sha256_bytes(payload),
        }
    assert empty_analysis["analysis_source_identity_sha256"] == m6.canonical_sha256(
        source_identity
    )


def test_empty_partial_and_mixed_journals_are_accounted(
    empty_analysis,
    partial_analysis,
    mixed_analysis,
):
    empty = empty_analysis["accounting"]
    assert empty["active_slots"] == 210
    assert empty["terminal_slots"] == 0
    assert empty["pending_slots"] == 210
    assert empty["expected_rows"] == 210 * 55
    assert empty["pending_raw_rows"] == 210 * 55
    assert empty_analysis["completion_state"] == "incomplete_evidence"

    partial = partial_analysis["accounting"]
    assert partial["terminal_slots"] == 1
    assert partial["observed_slots"] == 1
    assert partial["excluded_typed_slots"] == 0
    assert partial["pending_slots"] == 209
    assert partial["observed_raw_rows"] == 55

    mixed = mixed_analysis["accounting"]
    assert mixed["terminal_slots"] == 2
    assert mixed["observed_slots"] == 1
    assert mixed["excluded_typed_slots"] == 1
    assert mixed["pending_slots"] == 208
    assert mixed["observed_raw_rows"] == 55
    assert mixed["terminal_typed_excluded_rows"] == 55


def test_complete_no_trigger_and_triggered_tier2_pending_states(
    no_trigger_analysis,
    triggered_pending_analysis,
):
    no_trigger = no_trigger_analysis["accounting"]
    assert no_trigger["trigger_decisions_emitted"] == 3
    assert no_trigger["triggered_cells"] == 0
    assert no_trigger["active_tier2_slots"] == 0
    assert no_trigger["active_slots"] == 210
    assert no_trigger["terminal_slots"] == 210
    assert no_trigger["pending_slots"] == 0

    triggered = triggered_pending_analysis["accounting"]
    assert triggered["trigger_decisions_emitted"] == 3
    assert triggered["triggered_cells"] == 1
    assert triggered["tier2_contract_slots"] == 10
    assert triggered["active_tier2_slots"] == 10
    assert triggered["active_slots"] == 220
    assert triggered["terminal_slots"] == 210
    assert triggered["pending_slots"] == 10
    assert triggered["pending_raw_rows"] == 10 * 55
    assert triggered_pending_analysis["completion_state"] == "tier2_required"


def test_exact_cardinality_keys_and_accounting_identities(
    empty_analysis,
    mixed_analysis,
    no_trigger_analysis,
    triggered_pending_analysis,
):
    for document in (
        empty_analysis,
        mixed_analysis,
        no_trigger_analysis,
        triggered_pending_analysis,
    ):
        accounting = document["accounting"]
        assert len(document["classifications"]) == 990
        assert len(document["out_of_range_rates"]) == 330
        assert len(document["noise_floors"]) == 55
        assert len(document["denominators"]) == 165
        assert len(accounting["per_config_metric"]) == 330
        assert len(
            {
                (row["cell_id"], row["config_id"], row["metric_key"])
                for row in document["classifications"]
            }
        ) == 990
        assert len(
            {(row["config_id"], row["metric_key"]) for row in document["out_of_range_rates"]}
        ) == 330
        assert len({row["metric_key"] for row in document["noise_floors"]}) == 55
        assert len(
            {(row["cell_id"], row["metric_key"]) for row in document["denominators"]}
        ) == 165

        assert accounting["active_slots"] == (
            accounting["observed_slots"]
            + accounting["excluded_typed_slots"]
            + accounting["pending_slots"]
        )
        assert accounting["terminal_slots"] == (
            accounting["observed_slots"] + accounting["excluded_typed_slots"]
        )
        assert accounting["observed_raw_rows"] == (
            accounting["in_range_raw_rows"]
            + accounting["out_of_range_low_raw_rows"]
            + accounting["out_of_range_high_raw_rows"]
        )
        assert accounting["expected_rows"] == (
            accounting["observed_raw_rows"]
            + accounting["terminal_typed_excluded_rows"]
            + accounting["pending_raw_rows"]
        )
        assert accounting["observed_row_identity_holds"] is True
        assert accounting["terminal_slot_identity_holds"] is True
        assert accounting["expected_row_identity_holds"] is True
        assert accounting["bucket_identities_hold"] is True
        for row in document["classifications"]:
            assert row["expected_repeats"] == (
                row["observed_repeats"]
                + row["terminal_typed_excluded_repeats"]
                + row["pending_repeats"]
            )
            assert row["out_of_range_repeats"] == (
                row["out_of_range_low_repeats"]
                + row["out_of_range_high_repeats"]
            )
            assert row["observed_repeats"] == (
                row["in_range_repeats"]
                + row["out_of_range_low_repeats"]
                + row["out_of_range_high_repeats"]
            )
            assert row["observation_identity_holds"] is True
        for row in accounting["per_config_metric"]:
            assert row["expected_rows"] == (
                row["observed_raw_rows"]
                + row["terminal_typed_excluded_rows"]
                + row["pending_raw_rows"]
            )
            assert row["observed_raw_rows"] == (
                row["in_range_raw_rows"]
                + row["out_of_range_low_raw_rows"]
                + row["out_of_range_high_raw_rows"]
            )
            assert row["row_identity_holds"] is True


def test_all_excluded_evidence_never_becomes_a_pass(all_excluded_analysis):
    assert all_excluded_analysis["accounting"]["observed_slots"] == 0
    assert all_excluded_analysis["accounting"]["excluded_typed_slots"] == 210
    assert all(
        row["determinism_class"] == "insufficient_evidence"
        and row["would_have_been_admitted"] is None
        for row in all_excluded_analysis["classifications"]
    )
    assert all(
        row["denominator"] is None and row["exact_match"] is False
        for row in all_excluded_analysis["denominators"]
    )
    assert all(value is False for value in all_excluded_analysis["claims"].values() if isinstance(value, bool) and value not in {
        all_excluded_analysis["claims"]["analysis_is_diagnostic_only"],
        all_excluded_analysis["claims"]["hardware_values_are_targets_only"],
    })


def test_finite_oor_is_retained_uncensored_and_annotated(successor_contract, oor_contract):
    target = oor_contract["target"]
    metric = oor_contract["metric"]
    row = _classification(
        oor_contract["analysis"],
        cell_id=target["cell_id"],
        config_id=target["config_id"],
        metric_key=metric["key"],
    )
    assert row["determinism_class"] == "vendor_defect"
    assert row["in_range_repeats"] == 0
    assert row["out_of_range_repeats"] == 1
    assert row["out_of_range_low_repeats"] == 0
    assert row["out_of_range_high_repeats"] == 1
    assert row["out_of_range_values"] == [
        {
            "ordinal": target["ordinal"],
            "value": oor_contract["value"],
            "validity": "out_of_range_high",
        }
    ]
    assert isinstance(row["out_of_range_values"][0]["value"], int)
    assert "oor_floor_lower_bound" in row["annotations"]
    floor = next(
        item
        for item in oor_contract["analysis"]["noise_floors"]
        if item["metric_key"] == metric["key"]
    )
    assert floor["lower_bound"] is True
    assert oor_contract["analysis"]["analysis_policy"]["raw_values_clamped"] is False
    assert oor_contract["analysis"]["analysis_policy"][
        "finite_out_of_range_values_retained"
    ] is True
    analysis.validate_analysis(
        successor_contract["schedule"],
        oor_contract["journal"],
        oor_contract["analysis"],
    )


def test_finite_n_bounds_are_emitted_only_when_applicable(oor_contract):
    def assert_finite_n(row, *, observed, out_of_range):
        if observed == 0:
            assert row["event_resolution"] is None
        else:
            assert row["event_resolution"] == pytest.approx(
                min(1.0, 1.0 / observed)
            )
        if observed > 0 and out_of_range == 0:
            assert row["zero_event_upper_bound"] == pytest.approx(
                min(1.0, 3.0 / observed)
            )
        else:
            assert row["zero_event_upper_bound"] is None

    for row in oor_contract["analysis"]["classifications"]:
        assert_finite_n(
            row,
            observed=row["observed_repeats"],
            out_of_range=row["out_of_range_repeats"],
        )

    for row in oor_contract["analysis"]["out_of_range_rates"]:
        observed = row["observed_rows"]
        assert row["rate"] == (
            pytest.approx(row["out_of_range_rows"] / observed)
            if observed
            else None
        )
        assert_finite_n(
            row,
            observed=observed,
            out_of_range=row["out_of_range_rows"],
        )


def test_nonfinite_observations_are_rejected_upstream(successor_contract):
    observations = _observations(successor_contract["policy"])
    observations[0]["value"] = float("inf")
    observations[0]["validity"] = "out_of_range_high"
    with pytest.raises(journal.JournalError):
        journal.build_journal(
            successor_contract["schedule"],
            [
                {"record_kind": "epoch_open"},
                {
                    "record_kind": "attempt_started",
                    "ordinal": 0,
                    "retry_of_attempt_id": None,
                },
                {
                    "record_kind": "attempt_terminal",
                    "outcome": "observed",
                    "ncu_replay_passes": 1,
                    "observations": observations,
                },
            ],
            created_at_utc=GENERATED_AT,
        )


def test_three_cell_floors_are_null_insufficient_and_have_no_fallback(
    empty_analysis,
    no_trigger_analysis,
    triggered_pending_analysis,
):
    for document in (empty_analysis, no_trigger_analysis, triggered_pending_analysis):
        assert len(document["noise_floors"]) == 55
        for floor in document["noise_floors"]:
            assert floor["valid_cells"] <= 3
            assert floor["floor"] is None
            assert floor["cross_cell_p90"] is None
            assert floor["p90_upper_90"] is None
            assert floor["status"] == "insufficient_cells"
            assert floor["required_valid_cells"] == 30
            assert floor["fallback_authorized"] is False
        assert document["claims"]["floor_fallback_authorized"] is False
        assert document["claims"]["m6_done_created"] is False
        assert document["claims"]["m7_handoff_ready"] is False
        assert document["analysis_policy"]["floor_fallback_policy"] == (
            "forbidden_without_versioned_authority"
        )


def test_deterministic_resolution_and_pending_nondeterministic_denominators(
    no_trigger_analysis,
):
    deterministic = [
        row
        for row in no_trigger_analysis["denominators"]
        if row["determinism_class"] == "deterministic"
    ]
    nondeterministic = [
        row
        for row in no_trigger_analysis["denominators"]
        if row["determinism_class"] not in {"deterministic", "insufficient_evidence"}
    ]
    assert deterministic
    assert nondeterministic
    for row in deterministic:
        assert row["absolute_resolution"] > 0
        assert row["denominator"] == row["absolute_resolution"]
        assert row["denominator_kind"] == "absolute"
        assert row["gate_target"] == "exact_value"
        assert row["gate_scope"] == "diagnostic_only"
        assert row["exact_match"] is True
        assert row["status"] == "diagnostic_only_exact_resolution"
    for row in nondeterministic:
        assert row["pooled_floor"] is None
        assert row["denominator"] is None
        assert row["denominator_kind"] == "none"
        assert row["gate_target"] == "none"
        assert row["exact_match"] is False
        assert row["status"] == "pending_structural_floor"


def test_deterministic_bytes_negative_claims_and_external_pending_policies(
    successor_contract,
    no_trigger_journal,
    no_trigger_analysis,
):
    repeated = _analyze(successor_contract["schedule"], no_trigger_journal)
    encoded = json.dumps(
        no_trigger_analysis,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert encoded == json.dumps(
        repeated,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert no_trigger_analysis["analysis_sha256"] == m6.canonical_sha256(
        {
            key: value
            for key, value in no_trigger_analysis.items()
            if key != "analysis_sha256"
        }
    )
    assert no_trigger_analysis["claims"] == analysis.FIXED_CLAIMS
    multiplicity = no_trigger_analysis["analysis_policy"]["multiplicity_policy"]
    assert multiplicity["status"] == "external_pending"
    assert multiplicity["hypothesis_count"] == 0
    assert multiplicity["discoveries"] == []
    historical = no_trigger_analysis["analysis_policy"][
        "historical_admission_reconciliation"
    ]
    assert historical["status"] == "external_pending"
    assert historical["historical_admitted_set_sha256"] is None
    assert historical["exact_set_match"] is None


def test_forged_resealed_journal_summary_and_record_are_rejected(
    successor_contract,
    full_journal,
):
    forged_summary = copy.deepcopy(full_journal)
    forged_summary["replay_summary"]["cumulative_cost"]["fresh_processes"] -= 1
    _reseal_journal(forged_summary)
    with pytest.raises(analysis.AnalysisError, match="SUCCESSOR_M6_ANALYSIS_INPUT"):
        _analyze(successor_contract["schedule"], forged_summary)

    forged_record = copy.deepcopy(full_journal)
    records = copy.deepcopy(forged_record["records"])
    terminal = next(row for row in records if row["record_kind"] == "attempt_terminal")
    terminal["ordinal"] = 239
    forged_record["records"] = _rechain(successor_contract["schedule"], records)
    _reseal_journal(forged_record)
    with pytest.raises(analysis.AnalysisError, match="SUCCESSOR_M6_ANALYSIS_INPUT"):
        _analyze(successor_contract["schedule"], forged_record)


def test_schedule_transplant_is_rejected(successor_contract, full_journal):
    transplanted = schedule_builder.build_schedule(
        successor_contract["receipt_root"],
        seed=43,
        generated_at_utc="2026-07-23T01:00:00Z",
    )
    with pytest.raises(analysis.AnalysisError, match="SUCCESSOR_M6_ANALYSIS_INPUT"):
        _analyze(transplanted, full_journal)


@pytest.mark.parametrize(
    "mode", ["duplicate_key", "missing_row", "broken_accounting", "source_hash"]
)
def test_duplicate_missing_or_broken_diagnostics_are_rejected(
    successor_contract,
    no_trigger_journal,
    no_trigger_analysis,
    mode,
):
    forged = copy.deepcopy(no_trigger_analysis)
    if mode == "duplicate_key":
        first = forged["classifications"][0]
        last = forged["classifications"][-1]
        last["cell_id"] = first["cell_id"]
        last["cell_identity_sha256"] = first["cell_identity_sha256"]
        last["config_id"] = first["config_id"]
        last["config_sha256"] = first["config_sha256"]
        last["metric_key"] = first["metric_key"]
        last["metric_sha256"] = first["metric_sha256"]
        last["classification_sha256"] = m6.canonical_sha256(
            {key: value for key, value in last.items() if key != "classification_sha256"}
        )
    elif mode == "missing_row":
        forged["denominators"].pop()
    elif mode == "broken_accounting":
        forged["accounting"]["active_slots"] -= 1
    else:
        forged["analysis_source_identity"]["analyzer"]["sha256"] = "0" * 64
        forged["analysis_source_identity_sha256"] = m6.canonical_sha256(
            forged["analysis_source_identity"]
        )
    _reseal_analysis(forged)
    with pytest.raises(analysis.AnalysisError):
        analysis.validate_analysis(
            successor_contract["schedule"],
            no_trigger_journal,
            forged,
        )


@pytest.mark.parametrize("mode", ["removed", "clamped"])
def test_oor_removal_or_clamping_is_rejected(successor_contract, oor_contract, mode):
    forged = copy.deepcopy(oor_contract["analysis"])
    target = oor_contract["target"]
    metric = oor_contract["metric"]
    row = _classification(
        forged,
        cell_id=target["cell_id"],
        config_id=target["config_id"],
        metric_key=metric["key"],
    )
    if mode == "removed":
        row["out_of_range_values"] = []
    else:
        row["out_of_range_values"][0]["value"] = float(metric["valid_max"])
    row["classification_sha256"] = m6.canonical_sha256(
        {key: value for key, value in row.items() if key != "classification_sha256"}
    )
    _reseal_analysis(forged)
    with pytest.raises(analysis.AnalysisError):
        analysis.validate_analysis(
            successor_contract["schedule"],
            oor_contract["journal"],
            forged,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "nonnull_floor",
        "floor_fallback",
        "multiplicity_promoted",
        "history_promoted",
        "publication_authorized",
        "publication_claim",
        "canonical_m6_completed",
        "m6_done_created",
        "m7_handoff_ready",
        "m7_consumable",
    ],
)
def test_authority_promotions_are_rejected(
    successor_contract,
    no_trigger_journal,
    no_trigger_analysis,
    mutation,
):
    forged = copy.deepcopy(no_trigger_analysis)
    if mutation == "nonnull_floor":
        forged["noise_floors"][0]["floor"] = 0.01
    elif mutation == "floor_fallback":
        forged["noise_floors"][0]["fallback_authorized"] = True
        forged["claims"]["floor_fallback_authorized"] = True
    elif mutation == "multiplicity_promoted":
        forged["analysis_policy"]["multiplicity_policy"]["status"] = "complete"
    elif mutation == "history_promoted":
        forged["analysis_policy"]["historical_admission_reconciliation"][
            "status"
        ] = "complete"
    else:
        forged["claims"][mutation] = True
    _reseal_analysis(forged)
    with pytest.raises(analysis.AnalysisError):
        analysis.validate_analysis(
            successor_contract["schedule"],
            no_trigger_journal,
            forged,
        )


def test_no_execution_writer_publisher_fallback_input_or_synthetic_bypass_surface():
    source_path = analysis.ROOT / "tools/local_labels/cutlass_successor_m6_determinism_analysis.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    public_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    assert public_functions == {"analyze", "validate_analysis"}
    assert imports.isdisjoint({"argparse", "os", "subprocess"})
    for forbidden in (
        "allow-synthetic",
        "synthetic_bypass",
        "write_analysis",
        "publish_analysis",
        "collect_analysis",
        "CUDA_INJECTION64_PATH",
        "ncu ",
        "if __name__",
    ):
        assert forbidden not in source
    for function in (analysis.analyze, analysis.validate_analysis):
        assert not {
            "fallback",
            "fallback_authority",
            "synthetic",
            "execute",
            "publisher",
            "writer",
        }.intersection(inspect.signature(function).parameters)
