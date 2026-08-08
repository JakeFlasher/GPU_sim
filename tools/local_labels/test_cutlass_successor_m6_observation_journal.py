from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import jsonschema
import pytest

from tools.local_labels import admit_cutlass_successor_capture as admission
from tools.local_labels import build_cutlass_successor_m6_determinism_schedule as schedule_builder
from tools.local_labels import cutlass_successor_m6_observation_journal as journal
from tools.local_labels import m6_determinism_campaign as m6
from tools.local_labels.test_cutlass_successor_capture_admission import (
    _REPORT_IDENTITY,
    _all_pairs,
    _row_for_report_path,
    _row_report_csv,
)

GENERATED_AT = "2026-07-23T03:00:00Z"
STATIC_FILE_HASHES = {
    "tools/local_labels/build_cutlass_successor_m6_determinism_schedule.py":
        "3ec7994056b171d1ede12be4da65d73df9b5b3de09fef84427137d44d946c4a6",
    "tools/local_labels/test_cutlass_successor_m6_determinism_schedule.py":
        "b677c972aed47857fd8c9ec9fb664c5ecd92c5d7dae938f2901525f83dba1024",
    "workloads/simbench_bw/schemas/"
    "cutlass_successor_m6_determinism_schedule.v2.schema.json":
        "24f9d638ac50e9565e42d6e78b7923a6b5826841933add65905eb853a07a5888",
}


@pytest.fixture(scope="module")
def successor_contract():
    original_import = admission._import_ncu_report_csv
    original_sass = admission._sass_kernel_names
    admission._import_ncu_report_csv = lambda _identity, path: (
        _row_report_csv(_row_for_report_path(path)),
        admission.sha256_bytes(path.read_bytes()),
        None,
    )
    admission._sass_kernel_names = lambda _row: {
        _REPORT_IDENTITY["Kernel Name"]
    }
    try:
        with tempfile.TemporaryDirectory(
            prefix="successor-journal-pytest-", dir="/tmp",
        ) as temporary_root:
            root = Path(temporary_root) / "receipts"
            root.mkdir()
            _all_pairs(root)
            schedule = schedule_builder.build_schedule(
                root,
                seed=42,
                generated_at_utc="2026-07-23T01:00:00Z",
            )
            policy = journal.derive_metric_policy(schedule)
            empty = journal.build_journal(
                schedule,
                [],
                created_at_utc=GENERATED_AT,
            )
            yield {
                "schedule": schedule,
                "policy": policy,
                "empty": empty,
                "receipt_root": root,
            }
    finally:
        admission._import_ncu_report_csv = original_import
        admission._sass_kernel_names = original_sass


def _selected(policy):
    return {
        metric["key"]: metric
        for metric in policy["metrics"]
        if metric["capture_status"] == "selected"
    }


def _observations(
    policy,
    *,
    repeat_index=0,
    override_key=None,
    override_value=0.0,
):
    rows = []
    for key, metric in sorted(_selected(policy).items()):
        if metric["statistical_domain"] == "discrete":
            value = 0.0
        else:
            value = float(repeat_index // 2 + 1)
            if metric["valid_max"] == 1.0:
                value /= 10.0
        if key == override_key:
            value = override_value
        rows.append(
            {
                "metric_key": key,
                "metric_sha256": metric["metric_sha256"],
                "value": value,
                "validity": journal._validity(float(value), metric),
            }
        )
    return rows


def _trigger_metric(policy):
    return next(
        metric
        for metric in policy["metrics"]
        if metric["capture_status"] == "selected"
        and metric["statistical_domain"] == "discrete"
        and metric["valid_max"] is None
    )


def _tier1_requests(schedule, policy, *, trigger_cell=None, censor_reference=False):
    requests = [{"record_kind": "epoch_open"}]
    trigger_metric = _trigger_metric(policy)
    censored = False
    for slot in sorted(schedule["slots"], key=lambda row: row["ordinal"]):
        requests.append(
            {
                "record_kind": "attempt_started",
                "ordinal": slot["ordinal"],
                "retry_of_attempt_id": None,
            }
        )
        is_reference = slot["config_id"] == journal.REFERENCE_CONFIG
        if is_reference and not (censor_reference and not censored):
            override = None
            value = 0.0
            if slot["cell_id"] == trigger_cell:
                override = trigger_metric["key"]
                value = 1.0 if slot["repeat_index"] % 2 == 0 else 2.0
            requests.append(
                {
                    "record_kind": "attempt_terminal",
                    "outcome": "observed",
                    "ncu_replay_passes": 1,
                    "observations": _observations(
                        policy,
                        repeat_index=slot["repeat_index"],
                        override_key=override,
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
                        "detail": "synthetic contract fixture",
                    },
                }
            )
            if is_reference:
                censored = True
    requests.append({"record_kind": "tier1_closed"})
    return requests


def _full_requests(schedule, policy):
    trigger_cell = schedule["cells"][0]["cell_id"]
    requests = _tier1_requests(
        schedule,
        policy,
        trigger_cell=trigger_cell,
    )
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


@pytest.fixture(scope="module")
def full_journal(successor_contract):
    return journal.build_journal(
        successor_contract["schedule"],
        _full_requests(
            successor_contract["schedule"],
            successor_contract["policy"],
        ),
        created_at_utc=GENERATED_AT,
    )


def _rechain(schedule, records):
    normalized = []
    previous = journal.ZERO_SHA256
    for index, record in enumerate(records):
        body = journal._without(
            record,
            "sequence",
            "previous_record_sha256",
            "record_sha256",
        )
        sealed = m6._seal_causal_record(body, previous, index)
        normalized.append(sealed)
        previous = sealed["record_sha256"]
    return normalized


def _reseal_journal(document):
    document["records_sha256"] = m6.canonical_sha256(document["records"])
    document["journal_sha256"] = m6.canonical_sha256(
        journal._without(document, "journal_sha256")
    )


def test_schema_authority_and_static_schedule_are_closed(successor_contract):
    schema = json.loads((journal.ROOT / journal.SCHEMA_REL).read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    for name, definition in schema["$defs"].items():
        if definition.get("type") == "object" and name not in {
            "record_base",
            "attempt_fields",
        }:
            assert definition.get("additionalProperties") is False
        if name.endswith("_record"):
            assert definition["unevaluatedProperties"] is False

    assert schema["$defs"]["replay_summary"]["properties"][
        "interrupted_attempt_count"
    ]["maximum"] == 241
    assert successor_contract["empty"]["resource_policy"] == {
        "max_attempts_per_ordinal": 2,
        "max_fresh_processes": 480,
        "max_interrupted_attempts": 241,
        "max_exclusion_detail_bytes": 4096,
        "max_records": 4096,
    }
    policy = successor_contract["policy"]
    assert len(policy["metrics"]) == 61
    selected = [row for row in policy["metrics"] if row["capture_status"] == "selected"]
    assert len(selected) == 55
    assert all(
        row["model_io_role"] == "target" and row["deployability"] == "label_only"
        for row in selected
    )
    assert successor_contract["schedule"]["execution_dispatch"][
        "schedule_is_executable"
    ] is False
    assert successor_contract["empty"]["claims"] == journal.EXPECTED_CLAIMS
    assert "trigger_decisions" not in successor_contract["empty"]
    assert "cost_summary" not in successor_contract["empty"]
    assert "resume_epoch" not in successor_contract["empty"]

    for relative, expected in STATIC_FILE_HASHES.items():
        assert admission.sha256_bytes((journal.ROOT / relative).read_bytes()) == expected


def test_full_tier1_replay_derives_decisions_and_activation(
    successor_contract,
    full_journal,
):
    summary = journal.validate_journal(
        successor_contract["schedule"],
        full_journal,
    )
    assert summary["terminal_slots"] == 210
    assert summary["tier1_terminal_slots"] == 210
    assert summary["cumulative_cost"] == {
        "fresh_processes": 210,
        "ncu_replay_passes": 210,
    }
    assert summary["trigger_decisions_emitted"] == 3
    assert summary["triggered_cell_ids"] == [
        successor_contract["schedule"]["cells"][0]["cell_id"]
    ]
    expected_ordinals = [
        slot["ordinal"]
        for slot in successor_contract["schedule"]["tier2_candidates"]
        if slot["cell_id"] == summary["triggered_cell_ids"][0]
    ]
    assert len(expected_ordinals) == 10
    assert summary["activated_ordinals"] == expected_ordinals
    kinds = [record["record_kind"] for record in full_journal["records"]]
    assert kinds.count("trigger_decision") == 3
    assert kinds.count("tier2_activated") == 1


def test_open_crash_is_checkpointed_and_retried_later(successor_contract):
    schedule = successor_contract["schedule"]
    policy = successor_contract["policy"]
    crash = journal.build_journal(
        schedule,
        [
            {"record_kind": "epoch_open"},
            {
                "record_kind": "attempt_started",
                "ordinal": 0,
                "retry_of_attempt_id": None,
            },
        ],
        created_at_utc=GENERATED_AT,
    )
    first_attempt = crash["replay_summary"]["open_attempt_id"]
    assert first_attempt is not None

    interrupted = journal.extend_journal(
        schedule,
        crash,
        [
            {"record_kind": "checkpoint"},
            {"record_kind": "epoch_closed"},
        ],
    )
    assert interrupted["replay_summary"]["interrupted_attempt_id"] == first_attempt

    resumed = journal.extend_journal(
        schedule,
        interrupted,
        [
            {"record_kind": "epoch_open"},
            {
                "record_kind": "attempt_started",
                "ordinal": 0,
                "retry_of_attempt_id": first_attempt,
            },
            {
                "record_kind": "attempt_terminal",
                "outcome": "observed",
                "ncu_replay_passes": 1,
                "observations": _observations(policy),
            },
            {"record_kind": "checkpoint"},
            {"record_kind": "epoch_closed"},
        ],
    )
    summary = resumed["replay_summary"]
    assert summary["terminal_slots"] == 1
    assert summary["next_eligible_ordinal"] == 1
    assert summary["interrupted_attempt_id"] is None
    assert summary["interrupted_attempt_count"] == 1
    assert summary["cumulative_cost"] == {
        "fresh_processes": 2,
        "ncu_replay_passes": 1,
    }
    starts = [
        record
        for record in resumed["records"]
        if record["record_kind"] == "attempt_started"
    ]
    assert [record["attempt_index"] for record in starts] == [0, 1]
    assert starts[1]["retry_of_attempt_id"] == starts[0]["attempt_id"]


@pytest.mark.parametrize(
    "requests, code",
    [
        ([{"record_kind": "attempt_terminal"}], "EPOCH"),
        (
            [
                {"record_kind": "epoch_open"},
                {"record_kind": "attempt_started", "ordinal": 1},
            ],
            "ORDINAL",
        ),
        (
            [{"record_kind": "epoch_open"}, {"record_kind": "tier1_closed"}],
            "TIER1",
        ),
        (
            [
                {"record_kind": "epoch_open"},
                {"record_kind": "trigger_decision", "cell_id": "wrong"},
            ],
            "TRIGGER",
        ),
        (
            [{"record_kind": "epoch_open"}, {"record_kind": "tier2_activated"}],
            "ACTIVATION",
        ),
        ([{"record_kind": "epoch_closed"}], "EPOCH"),
        (
            [{"record_kind": "epoch_open"}, {"record_kind": "epoch_open"}],
            "EPOCH",
        ),
    ],
)
def test_invalid_causal_requests_fail_closed(successor_contract, requests, code):
    with pytest.raises(journal.JournalError, match=f"SUCCESSOR_M6_JOURNAL_{code}"):
        journal._replay(
            successor_contract["schedule"],
            successor_contract["policy"],
            requests,
            sealed=False,
        )


def test_retry_requires_later_epoch_exact_predecessor(successor_contract):
    schedule = successor_contract["schedule"]
    policy = successor_contract["policy"]
    first = journal._replay(
        schedule,
        policy,
        [
            {"record_kind": "epoch_open"},
            {"record_kind": "attempt_started", "ordinal": 0},
            {"record_kind": "checkpoint"},
            {"record_kind": "epoch_closed"},
        ],
        sealed=False,
    )[0]
    predecessor = next(
        record["attempt_id"]
        for record in first
        if record["record_kind"] == "attempt_started"
    )
    base = [journal._record_request(record) for record in first]
    base.append({"record_kind": "epoch_open"})
    for start in (
        {"record_kind": "attempt_started", "ordinal": 0},
        {
            "record_kind": "attempt_started",
            "ordinal": 0,
            "retry_of_attempt_id": "f" * 64,
        },
        {
            "record_kind": "attempt_started",
            "ordinal": 1,
            "retry_of_attempt_id": predecessor,
        },
    ):
        with pytest.raises(journal.JournalError, match="SUCCESSOR_M6_JOURNAL_(RETRY|ORDINAL)"):
            journal._replay(schedule, policy, base + [start], sealed=False)

    same_epoch = [journal._record_request(record) for record in first[:-1]]
    same_epoch.append(
        {
            "record_kind": "attempt_started",
            "ordinal": 0,
            "retry_of_attempt_id": predecessor,
        }
    )
    with pytest.raises(journal.JournalError, match="SUCCESSOR_M6_JOURNAL_EPOCH"):
        journal._replay(schedule, policy, same_epoch, sealed=False)


def test_per_ordinal_retry_resource_bound(successor_contract):
    schedule = successor_contract["schedule"]
    policy = successor_contract["policy"]
    requests = [
        {"record_kind": "epoch_open"},
        {"record_kind": "attempt_started", "ordinal": 0},
        {"record_kind": "checkpoint"},
        {"record_kind": "epoch_closed"},
    ]
    first_records, _ = journal._replay(schedule, policy, requests, sealed=False)
    first_attempt = next(
        record["attempt_id"]
        for record in first_records
        if record["record_kind"] == "attempt_started"
    )
    requests.extend(
        [
            {"record_kind": "epoch_open"},
            {
                "record_kind": "attempt_started",
                "ordinal": 0,
                "retry_of_attempt_id": first_attempt,
            },
            {"record_kind": "checkpoint"},
            {"record_kind": "epoch_closed"},
        ]
    )
    second_records, _ = journal._replay(schedule, policy, requests, sealed=False)
    second_attempt = [
        record["attempt_id"]
        for record in second_records
        if record["record_kind"] == "attempt_started"
    ][1]
    requests.extend(
        [
            {"record_kind": "epoch_open"},
            {
                "record_kind": "attempt_started",
                "ordinal": 0,
                "retry_of_attempt_id": second_attempt,
            },
        ]
    )
    with pytest.raises(journal.JournalError, match="SUCCESSOR_M6_JOURNAL_RETRY"):
        journal._replay(schedule, policy, requests, sealed=False)


def test_variable_payloads_are_resource_bounded_and_typed(successor_contract):
    class OversizedDetail(str):
        def encode(self, *args, **kwargs):
            pytest.fail("oversized detail must be rejected before UTF-8 allocation")

    schedule = successor_contract["schedule"]
    policy = successor_contract["policy"]
    prefix = [
        {"record_kind": "epoch_open"},
        {"record_kind": "attempt_started", "ordinal": 0},
    ]
    for detail in (OversizedDetail("x" * 4097), "\ud800"):
        with pytest.raises(
            journal.JournalError,
            match="SUCCESSOR_M6_JOURNAL_EXCLUSION",
        ):
            journal._replay(
                schedule,
                policy,
                prefix
                + [
                    {
                        "record_kind": "attempt_terminal",
                        "outcome": "excluded_typed",
                        "ncu_replay_passes": 1,
                        "typed_exclusion": {
                            "reason": "profile_failed",
                            "detail": detail,
                        },
                    }
                ],
                sealed=False,
            )

    observations = _observations(policy)
    observations[0]["value"] = 10**400
    with pytest.raises(
        journal.JournalError,
        match="SUCCESSOR_M6_JOURNAL_OBSERVATIONS",
    ):
        journal._replay(
            schedule,
            policy,
            prefix
            + [
                {
                    "record_kind": "attempt_terminal",
                    "outcome": "observed",
                    "ncu_replay_passes": 1,
                    "observations": observations,
                }
            ],
            sealed=False,
        )


def test_resealed_record_mutations_fail_replay(
    successor_contract,
    full_journal,
):
    schedule = successor_contract["schedule"]
    policy = successor_contract["policy"]
    mutations = []

    changed_terminal = copy.deepcopy(full_journal["records"])
    terminal = next(
        record for record in changed_terminal if record["record_kind"] == "attempt_terminal"
    )
    terminal["ordinal"] = 7
    mutations.append(changed_terminal)

    changed_checkpoint = copy.deepcopy(full_journal["records"])
    checkpoint = next(
        record for record in changed_checkpoint if record["record_kind"] == "checkpoint"
    )
    checkpoint["cumulative_cost"]["fresh_processes"] -= 1
    mutations.append(changed_checkpoint)

    changed_decision = copy.deepcopy(full_journal["records"])
    decision = next(
        record for record in changed_decision if record["record_kind"] == "trigger_decision"
    )
    decision["triggered"] = not decision["triggered"]
    mutations.append(changed_decision)

    changed_activation = copy.deepcopy(full_journal["records"])
    activation = next(
        record
        for record in changed_activation
        if record["record_kind"] == "tier2_activated"
    )
    activation["activated_ordinals"] = activation["activated_ordinals"][:-1]
    mutations.append(changed_activation)

    permuted = copy.deepcopy(full_journal["records"])
    permuted[0], permuted[1] = permuted[1], permuted[0]
    mutations.append(permuted)

    for records in mutations:
        with pytest.raises(journal.JournalError):
            journal._replay(
                schedule,
                policy,
                _rechain(schedule, records),
                sealed=True,
            )


def test_incomplete_reference_evidence_cannot_become_nontrigger(successor_contract):
    requests = _tier1_requests(
        successor_contract["schedule"],
        successor_contract["policy"],
        censor_reference=True,
    )
    requests.append(
        {
            "record_kind": "trigger_decision",
            "cell_id": successor_contract["schedule"]["cells"][0]["cell_id"],
        }
    )
    with pytest.raises(
        journal.JournalError,
        match="SUCCESSOR_M6_JOURNAL_TRIGGER_INCOMPLETE",
    ):
        journal._replay(
            successor_contract["schedule"],
            successor_contract["policy"],
            requests,
            sealed=False,
        )


def test_policy_summary_and_top_level_authority_tampering_refuse(
    successor_contract,
    full_journal,
):
    policy_mutation = copy.deepcopy(full_journal)
    selected = next(
        metric
        for metric in policy_mutation["metric_policy"]["metrics"]
        if metric["capture_status"] == "selected"
    )
    selected["model_io_role"] = "input"
    selected["metric_sha256"] = m6.canonical_sha256(
        journal._without(selected, "metric_sha256")
    )
    policy_mutation["metric_policy_sha256"] = m6.canonical_sha256(
        policy_mutation["metric_policy"]
    )
    _reseal_journal(policy_mutation)
    with pytest.raises(journal.JournalError, match="SUCCESSOR_M6_JOURNAL_METRICS"):
        journal.validate_journal(successor_contract["schedule"], policy_mutation)

    source_mutation = copy.deepcopy(full_journal)
    source_mutation["source_identity"]["statistics_module"]["sha256"] = "0" * 64
    source_mutation["source_identity_sha256"] = m6.canonical_sha256(
        source_mutation["source_identity"]
    )
    _reseal_journal(source_mutation)
    with pytest.raises(journal.JournalError, match="SUCCESSOR_M6_JOURNAL_SOURCE"):
        journal.validate_journal(successor_contract["schedule"], source_mutation)

    summary_mutation = copy.deepcopy(full_journal)
    summary_mutation["replay_summary"]["cumulative_cost"]["fresh_processes"] -= 1
    _reseal_journal(summary_mutation)
    with pytest.raises(journal.JournalError, match="SUCCESSOR_M6_JOURNAL_SUMMARY"):
        journal.validate_journal(successor_contract["schedule"], summary_mutation)

    extra_authority = copy.deepcopy(full_journal)
    extra_authority["trigger_decisions"] = []
    _reseal_journal(extra_authority)
    with pytest.raises(journal.JournalError, match="SUCCESSOR_M6_JOURNAL_SCHEMA"):
        journal.validate_journal(successor_contract["schedule"], extra_authority)


def test_schedule_transplant_is_rejected(successor_contract, full_journal):
    other = schedule_builder.build_schedule(
        successor_contract["receipt_root"],
        seed=43,
        generated_at_utc="2026-07-23T01:00:00Z",
    )
    with pytest.raises(journal.JournalError, match="SUCCESSOR_M6_JOURNAL_SCHEDULE"):
        journal.validate_journal(other, full_journal)


def test_no_execution_publication_or_synthetic_bypass_surface():
    source = (journal.ROOT / journal.MODULE_REL).read_text()
    for forbidden in (
        "allow-synthetic",
        "subprocess",
        "write_journal",
        "publish_journal",
        "CUDA_INJECTION64_PATH",
        "ncu ",
    ):
        assert forbidden not in source
