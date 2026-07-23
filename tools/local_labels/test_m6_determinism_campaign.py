from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import m6_determinism_campaign as campaign  # noqa: E402
from m6_determinism_stats import (  # noqa: E402
    d2_for_n,
    dispersion_gate,
    modality_result,
    sigma_rel,
    sigma_upper_factor,
)

FIXED_TIME = "2026-07-18T00:00:00Z"
OPERATOR_RECEIPT = "f" * 64


@pytest.fixture(scope="module")
def schedule() -> dict:
    return campaign.build_schedule(seed=0xC0FFEE, generated_at_utc=FIXED_TIME)


def test_canonical_matrix_uses_supported_frozen_validation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    original_validate = campaign.matrix_builder.validate_matrix

    def record_validate(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(
        campaign.matrix_builder,
        "validate_matrix",
        record_validate,
    )
    cache = campaign.InputSnapshotCache()
    matrix, matrix_payload, runnable = campaign._load_canonical_matrix(
        campaign.MATRIX_PATH,
        repo_root=campaign.REPO_ROOT,
        require_semantic_rederivation=True,
        cache=cache,
    )
    assert calls[0]["validation_mode"] == "frozen_v1"
    assert campaign.sha256_bytes(matrix_payload) == (
        campaign.matrix_builder.FROZEN_V1_MATRIX_SHA256
    )
    assert len(runnable) == campaign.EXPECTED_RUNNABLE
    registry_path = campaign.REGISTRY_PATH
    registry_document, registry_payload = campaign.load_json(
        registry_path,
        role="metric registry",
        cache=cache,
    )
    assert campaign.matrix_builder.validate_matrix(
        matrix,
        repo_root=campaign.REPO_ROOT,
        registry_document=registry_document,
        registry_payload=registry_payload,
        registry_sha256=campaign.sha256_bytes(registry_payload),
        validation_mode="frozen_v1",
    ) == []


def test_frozen_source_alias_resolves_exact_registry_and_sidecar_bytes() -> None:
    cache = campaign.InputSnapshotCache()
    aliases, alias_payload = campaign._load_frozen_source_alias(
        repo_root=campaign.REPO_ROOT,
        cache=cache,
    )
    assert campaign.sha256_bytes(alias_payload) == campaign.FROZEN_SOURCE_ALIAS_SHA256
    expected = {
        campaign.REGISTRY_PATH.relative_to(campaign.REPO_ROOT).as_posix(): (
            campaign.REGISTRY_PATH
        ),
        campaign.SIDECAR_SCHEMA_PATH.relative_to(campaign.REPO_ROOT).as_posix(): (
            campaign.SIDECAR_SCHEMA_PATH
        ),
    }
    assert set(aliases) == set(expected)
    for relative, path in expected.items():
        payload = campaign._frozen_source_payload(
            campaign.REPO_ROOT,
            campaign.FROZEN_V1_SOURCE_COMMIT,
            relative,
            role=f"test frozen source {relative}",
            cache=cache,
        )
        assert payload == path.read_bytes()
        assert campaign.sha256_bytes(payload) == aliases[relative]["sha256"]
        assert len(payload) == aliases[relative]["size_bytes"]


def test_matrix_authority_sources_do_not_invoke_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_subprocess(*_args, **_kwargs):  # noqa: ANN002,ANN003
        raise AssertionError("historical Git lookup attempted")

    monkeypatch.setattr(subprocess, "run", reject_subprocess)
    matrix = json.loads(campaign.MATRIX_PATH.read_text(encoding="utf-8"))
    sources = campaign._load_matrix_authority_sources(
        matrix,
        repo_root=campaign.REPO_ROOT,
        cache=campaign.InputSnapshotCache(),
    )
    assert sources["frozen_metric_registry"]["sha256"] == (
        matrix["input_pins"]["metric_registry"]["sha256"]
    )
    assert sources["frozen_sidecar_schema"]["sha256"] == (
        "49b436d60535c4e213d3841d65b2e6f1d58b1c45d132b5ee20b06dc5ed09ae94"
    )


def test_frozen_source_alias_rejects_manifest_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "FROZEN_SOURCE_ALIAS_SHA256", "0" * 64)
    with pytest.raises(campaign.M6Error) as error:
        campaign._load_frozen_source_alias(
            repo_root=campaign.REPO_ROOT,
            cache=campaign.InputSnapshotCache(),
        )
    assert error.value.code == "M6_SOURCE_IDENTITY_MISMATCH"


def test_frozen_source_alias_rejects_unknown_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_loads = campaign.loads_strict

    def load_version_drift(payload: bytes, *, role: str):
        value = original_loads(payload, role=role)
        if role == "frozen source alias table":
            value = copy.deepcopy(value)
            value["schema_version"] = 2
        return value

    monkeypatch.setattr(campaign, "loads_strict", load_version_drift)
    with pytest.raises(campaign.M6Error) as error:
        campaign._load_frozen_source_alias(
            repo_root=campaign.REPO_ROOT,
            cache=campaign.InputSnapshotCache(),
        )
    assert error.value.code == "M6_SOURCE_ALIAS_VERSION"


def test_frozen_source_alias_rejects_payload_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = campaign._load_frozen_source_alias

    def load_bad_identity(*, repo_root: Path, cache=None):
        aliases, payload = original_load(repo_root=repo_root, cache=cache)
        aliases = copy.deepcopy(aliases)
        registry = campaign.REGISTRY_PATH.relative_to(campaign.REPO_ROOT).as_posix()
        aliases[registry]["sha256"] = "0" * 64
        return aliases, payload

    monkeypatch.setattr(campaign, "_load_frozen_source_alias", load_bad_identity)
    with pytest.raises(campaign.M6Error) as error:
        campaign._frozen_source_payload(
            campaign.REPO_ROOT,
            campaign.FROZEN_V1_SOURCE_COMMIT,
            campaign.REGISTRY_PATH.relative_to(campaign.REPO_ROOT).as_posix(),
            role="drifted frozen registry",
            cache=campaign.InputSnapshotCache(),
        )
    assert error.value.code == "M6_SOURCE_IDENTITY_MISMATCH"


@pytest.mark.parametrize(
    ("source_commit", "relative_path", "expected_code"),
    [
        (
            "0" * 40,
            "tools/local_labels/metric_registry.json",
            "M6_SOURCE_ALIAS_COMMIT",
        ),
        (
            campaign.FROZEN_V1_SOURCE_COMMIT,
            "tools/local_labels/metric_registry.successor.json",
            "M6_SOURCE_ALIAS_MISSING",
        ),
    ],
)
def test_frozen_source_alias_rejects_unadmitted_fallbacks(
    source_commit: str,
    relative_path: str,
    expected_code: str,
) -> None:
    with pytest.raises(campaign.M6Error) as error:
        campaign._frozen_source_payload(
            campaign.REPO_ROOT,
            source_commit,
            relative_path,
            role="unadmitted frozen source",
            cache=campaign.InputSnapshotCache(),
        )
    assert error.value.code == expected_code


def _schema(kind: str) -> dict:
    return json.loads(campaign.SCHEMA_PATHS[kind].read_text())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _make_tree_readonly(root: Path) -> None:
    paths = list(root.rglob("*"))
    for path in paths:
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(
        (path for path in paths if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    root.chmod(0o555)


def _sidecar_for_slot(schedule: dict, slot: dict, *, value: object = 0.5) -> dict:
    cell = next(item for item in schedule["cells"] if item["cell_id"] == slot["cell_id"])
    config = next(item for item in schedule["configs"] if item["config_id"] == slot["config_id"])
    protocol = {
        "ncu_clock_control": config["clock_control"],
        "ncu_cache_control": config["cache_control"],
        "ncu_replay_mode": "kernel",
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_profile_from_start": False,
        "ncu_launch_skip": 0,
        "ncu_launch_count": 1,
        "host_persistence_mode": "Enabled",
        "host_sm_clock_mhz": 1350 if config["reference"] else 1700,
        "host_mem_clock_mhz": 10501,
        "host_temperature_c": 61,
    }
    return {
        "schema_version": 5,
        "device": "sm_86",
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "repeat_index": slot["repeat_index"],
        "launch_index": 0,
        "binary_sha256": cell["binary_sha256"],
        "profiler_family": "nsight_compute",
        "profiler_version": "2026.1.0",
        "cuda_version": "13.3",
        "driver_version": "610.43.02",
        "gpu_uuid_sha256": "1" * 64,
        "replay_mode": "kernel",
        "aggregation": "kernel",
        "warmup_count": 1,
        "cache_hit_rate_l2_sector": value,
        "confidence_interval_l2_sector": {
            "lower": value,
            "upper": value,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        },
        "measurement_protocol": protocol,
    }


def test_all_five_schemas_are_valid_and_closed_at_object_definitions() -> None:
    for kind in campaign.SCHEMA_PATHS:
        schema = _schema(kind)
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
        for definition in schema.get("$defs", {}).values():
            if definition.get("type") == "object":
                assert definition.get("additionalProperties") is False


def test_plan_has_exact_configs_counts_and_seeded_sweeps(schedule: dict) -> None:
    assert [(c["clock_control"], c["cache_control"]) for c in schedule["configs"]] == [
        ("none", "all"),
        ("none", "none"),
        ("base", "all"),
        ("base", "none"),
        ("boost", "all"),
        ("boost", "none"),
    ]
    assert len(schedule["cells"]) == 122
    assert len(schedule["slots"]) == 8_540
    assert len(schedule["tier2_candidates"]) == 1_220
    assert schedule["counts"]["maximum_slots"] == 9_760
    assert schedule["source_identity"]["stats_source"]["path"].endswith(
        "m6_determinism_stats.py"
    )
    assert schedule["source_identity"]["immutable_helper"]["path"].endswith(
        "immutable_artifact.py"
    )
    assert {item["path"] for item in schedule["source_identity"]["m6_schemas"]} == {
        path.relative_to(campaign.REPO_ROOT).as_posix()
        for path in campaign.SCHEMA_PATHS.values()
    }
    assert all(slot["process_isolation"] == "fresh_process" for slot in schedule["slots"])
    assert {slot["tier"] for slot in schedule["slots"]} == {1}
    assert {slot["tier"] for slot in schedule["tier2_candidates"]} == {2}
    assert {slot["config_id"] for slot in schedule["tier2_candidates"]} == {
        campaign.REFERENCE_CONFIG
    }
    first_sweep = [slot for slot in schedule["slots"] if slot["sweep_index"] == 0]
    assert len(first_sweep) == 122 * 6
    assert [slot["cell_id"] for slot in first_sweep] != sorted(
        slot["cell_id"] for slot in first_sweep
    )
    campaign.validate_schedule(schedule)


def test_plan_is_reproducible_and_seed_changes_order(schedule: dict) -> None:
    same = campaign.build_schedule(seed=0xC0FFEE, generated_at_utc=FIXED_TIME)
    other = campaign.build_schedule(seed=0xC0FFEF, generated_at_utc=FIXED_TIME)
    assert same == schedule
    assert same["slots_sha256"] != other["slots_sha256"]


def test_schedule_schema_rejects_boolean_numeric_fields(schedule: dict) -> None:
    schema = _schema("schedule")
    bad = copy.deepcopy(schedule)
    bad["seed"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(bad)
    bad = copy.deepcopy(schedule)
    bad["metrics"][0]["valid_min"] = False
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(bad)


def test_strict_json_rejects_unsafe_numbers_depth_and_surrogates() -> None:
    with pytest.raises(campaign.M6Error) as unsafe_integer:
        campaign.loads_strict(b'{"value":9007199254740992}', role="unsafe")
    assert unsafe_integer.value.code == "M6_JSON_UNSAFE_INTEGER"
    with pytest.raises(campaign.M6Error) as unsafe_float:
        campaign.loads_strict(b'{"value":1e20}', role="unsafe")
    assert unsafe_float.value.code == "M6_JSON_UNSAFE_FLOAT"
    with pytest.raises(campaign.M6Error) as surrogate:
        campaign.loads_strict(b'{"value":"\\ud800"}', role="unsafe")
    assert surrogate.value.code == "M6_JSON_SURROGATE"
    nested = b"[" * 300 + b"0" + b"]" * 300
    with pytest.raises(campaign.M6Error) as depth:
        campaign.loads_strict(nested, role="deep")
    assert depth.value.code in {"M6_JSON_DEPTH", "M6_JSON_PARSE"}


def test_observation_and_done_schemas_reject_boolean_numeric_fields(
    schedule: dict
) -> None:
    journal = campaign.build_journal(
        schedule,
        [
            campaign.build_excluded_event(
                schedule["slots"][0], reason="operator_stop", detail="fixture",
                operator_authority_receipt=OPERATOR_RECEIPT,
            )
        ],
        created_at_utc=FIXED_TIME,
    )
    bad_journal = copy.deepcopy(journal)
    bad_journal["events"][0]["cost"]["ncu_replay_passes"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema("observation_journal")).validate(bad_journal)

    artifact = {"artifact_sha256": "1" * 64, "size_bytes": 1}
    done = {
        "schema_version": 1,
        "artifact_kind": "m6_determinism_done",
        "status": "done",
        "generated_at_utc": FIXED_TIME,
        "schedule": artifact,
        "journal": artifact,
        "analysis": {
            "relative_path": "determinism_study.json",
            **artifact,
        },
        "schemas": [
            {
                "schema_id": schema_id,
                "path": path.relative_to(campaign.REPO_ROOT).as_posix(),
                "sha256": str(index + 2) * 64,
            }
            for index, (schema_id, path) in enumerate(
                campaign.SCHEMA_PATHS.items()
            )
        ],
        "source_identity_sha256": "6" * 64,
        "config_identity_sha256": "7" * 64,
        "metric_identity_sha256": "8" * 64,
        "m4_identity_sha256": "9" * 64,
        "m4_status": "complete",
        "counts": {
            "runnable_cells": 122,
            "configs": 6,
            "tier1_slots": 8540,
            "triggered_cells": 0,
            "tier2_slots": 0,
            "total_slots": 8540,
            "raw_rows_collected": 0,
            "classified_raw_rows": 0,
            "excluded_typed_raw_rows": 0,
            "not_collected_typed_rows": 469700,
        },
        "accounting_identity_holds": True,
        "denominator_identity_holds": True,
        "taxonomy": list(campaign.DETERMINISM_CLASSES),
        "collect_implementation": "disabled_offline_source",
        "done_sha256": "a" * 64,
    }
    Draft202012Validator(_schema("done")).validate(done)
    done["counts"]["total_slots"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema("done")).validate(done)


def test_completion_and_typed_exclusions_are_nonvacuous(schedule: dict) -> None:
    base = {
        "tier1_pending": 0,
        "trigger_state": "decided",
        "triggered_cells": 0,
        "tier2_pending": 0,
    }
    assert campaign._completion_state(
        **base,
        observed_slots=0,
        excluded_typed_slots=8_540,
        missing_selected_rows=469_700,
    ) == "incomplete_evidence"
    assert campaign._completion_state(
        **base,
        observed_slots=8_540,
        excluded_typed_slots=0,
        missing_selected_rows=1,
    ) == "incomplete_evidence"
    assert campaign._completion_state(
        **base,
        observed_slots=8_540,
        excluded_typed_slots=0,
        missing_selected_rows=0,
    ) == "complete"

    slot = schedule["slots"][0]
    with pytest.raises(campaign.M6Error) as stop_cost:
        campaign.build_excluded_event(
            slot,
            reason="operator_stop",
            detail="operator requested pause",
            fresh_processes=1,
            operator_authority_receipt=OPERATOR_RECEIPT,
        )
    assert stop_cost.value.code == "M6_TYPED_EXCLUSION_CONTRACT"
    with pytest.raises(campaign.M6Error) as missing_cost:
        campaign.build_excluded_event(
            slot,
            reason="sidecar_missing",
            detail="capture returned no sidecar",
        )
    assert missing_cost.value.code == "M6_TYPED_EXCLUSION_CONTRACT"

    with pytest.raises(campaign.M6Error) as missing_replay:
        campaign.build_excluded_event(
            slot,
            reason="profile_failed",
            detail="profiler exited before replay",
            fresh_processes=1,
            ncu_replay_passes=0,
        )
    assert missing_replay.value.code == "M6_TYPED_EXCLUSION_CONTRACT"

    invalid = campaign.build_excluded_event(
        slot,
        reason="profile_failed",
        detail="profiler exited nonzero",
        fresh_processes=1,
        ncu_replay_passes=1,
    )
    invalid["cost"]["fresh_processes"] = 0
    with pytest.raises(campaign.M6Error) as resealed_invalid:
        campaign.build_journal(schedule, [invalid], created_at_utc=FIXED_TIME)
    assert resealed_invalid.value.code == "M6_TYPED_EXCLUSION_CONTRACT"


def test_done_rejects_all_excluded_nonvacuous_forgery(
    schedule: dict,
    tmp_path: Path,
) -> None:
    forged_schedule = copy.deepcopy(schedule)
    forged_schedule["m4_identity"]["status"] = "complete"
    forged_journal = {"trigger_state": "decided"}
    forged_analysis = {
        "completion_state": "complete",
        "accounting": {
            "denominator_scope": "complete_m4_launch_set",
            "active_slots": 8_540,
            "observed_slots": 0,
            "terminal_slots": 8_540,
            "pending_slots": 0,
            "excluded_typed_slots": 8_540,
            "raw_rows_collected": 0,
            "expected_rows": 469_700,
            "not_collected_typed_rows": 469_700,
            "pending_expected_rows": 0,
            "classified_raw_rows": 0,
            "excluded_typed_raw_rows": 0,
            "per_config_metric": [],
        },
        "analysis_policy": {"reference_oor_zero_confirmed": True},
    }
    with pytest.raises(campaign.M6Error) as error:
        campaign.build_done(
            forged_schedule,
            forged_journal,
            forged_analysis,
            schedule_path=tmp_path / "schedule.json",
            journal_path=tmp_path / "journal.json",
            analysis_path=tmp_path / "analysis.json",
        )
    assert error.value.code == "M6_DONE_NONVACUOUS_EVIDENCE"


def test_dispersion_lockstep_scale_invariance_and_exact_d2() -> None:
    values = [10.0, 11.0, 9.0, 10.5, 9.5]
    assert dispersion_gate("stddev_abs", values) == pytest.approx(0.7905694150420949)
    assert dispersion_gate("coefficient_of_variation", values) == pytest.approx(
        dispersion_gate("coefficient_of_variation", [value * 100 for value in values])
    )
    assert sigma_rel("stddev_abs", values) == pytest.approx(
        sigma_rel("stddev_abs", [value * 100 for value in values])
    )
    assert sigma_rel("relative_tolerance", values) == pytest.approx(
        sigma_rel("relative_tolerance", [value * 100 for value in values])
    )
    assert [d2_for_n(n) for n in (5, 10, 20, 30)] == [2.326, 3.078, 3.735, 4.086]
    assert sigma_upper_factor(10) == pytest.approx(1.65, rel=0.04)
    assert sigma_upper_factor(20) == pytest.approx(1.37, rel=0.04)
    assert sigma_upper_factor(30) == pytest.approx(1.28, rel=0.04)
    with pytest.raises(campaign.M6Error):
        campaign._number(True, role="test")


def test_single_support_split_never_certifies_multimodal() -> None:
    result = modality_result(
        [0.877, 0.869, 0.851, 0.987, 0.878],
        discrete=False,
        absolute_resolution=0.0001,
        tier2=False,
    )
    assert result.outlier_flag is True
    assert result.certified is False


def test_continuous_multimodality_only_certifies_at_tier2() -> None:
    values = [1.0 + index * 0.0001 for index in range(15)] + [
        2.0 + index * 0.0001 for index in range(15)
    ]
    tier1 = modality_result(
        values[:20], discrete=False, absolute_resolution=0.0001, tier2=False
    )
    tier2 = modality_result(
        values, discrete=False, absolute_resolution=0.0001, tier2=True
    )
    assert tier1.certified is False
    assert tier2.flagged is True
    assert tier2.certified is True
    assert tier2.reason == "continuous_largest_gap_delta_bic"


def test_plan_cli_refuses_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "schedule.json"
    assert campaign.main([
        "plan", "--seed", "7", "--generated-at-utc", FIXED_TIME, "--out", str(out)
    ]) == 0
    before = out.read_bytes()
    assert out.stat().st_mode & 0o222 == 0
    assert not (tmp_path / ".schedule.json.m6-partial").exists()
    assert campaign.main([
        "plan", "--seed", "8", "--generated-at-utc", FIXED_TIME, "--out", str(out)
    ]) == 2
    assert out.read_bytes() == before


def test_anchored_input_snapshot_rejects_ancestor_symlink_and_same_size_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real"
    nested = real / "nested"
    nested.mkdir(parents=True)
    document = nested / "document.json"
    document.write_text('{"value":1}\n', encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(campaign.M6Error) as ancestor_error:
        campaign.load_json(alias / "nested" / "document.json", role="ancestor test")
    assert ancestor_error.value.code == "M6_INPUT_SNAPSHOT"

    hardlink_source = tmp_path / "hardlink-source.bin"
    hardlink_alias = tmp_path / "hardlink-alias.bin"
    hardlink_source.write_bytes(b"possibly unique")
    hardlink_alias.hardlink_to(hardlink_source)
    with pytest.raises(campaign.M6Error) as hardlink_error:
        campaign._read_regular(hardlink_source, role="hardlink test")
    assert hardlink_error.value.code == "M6_INPUT_SNAPSHOT"

    large = tmp_path / "large.bin"
    large.write_bytes(b"a" * ((1 << 20) + 1))
    original_read = campaign.immutable.os.read
    mutated = False

    def mutate_after_first_block(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        block = original_read(descriptor, size)
        if block and not mutated:
            mutated = True
            with large.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"b")
                stream.flush()
                campaign.os.fsync(stream.fileno())
        return block

    monkeypatch.setattr(campaign.immutable.os, "read", mutate_after_first_block)
    with pytest.raises(campaign.M6Error) as mutation_error:
        campaign._read_regular(large, role="same-size mutation")
    assert mutation_error.value.code == "M6_INPUT_SNAPSHOT"


def test_done_source_snapshot_replacement_between_checks_is_rejected(
    tmp_path: Path,
) -> None:
    source_snapshot = tmp_path / "source_snapshot.json"
    expected = b'{"witness":true}\n'
    source_snapshot.write_bytes(expected)
    cache = campaign.InputSnapshotCache()

    campaign._verify_existing_source_snapshot(
        source_snapshot,
        expected,
        cache=cache,
    )
    source_snapshot.unlink()
    source_snapshot.write_bytes(expected)

    with pytest.raises(campaign.M6Error) as error:
        campaign._verify_existing_source_snapshot(
            source_snapshot,
            expected,
            cache=cache,
        )
    assert error.value.code == "M6_INPUT_SNAPSHOT_MIXED"


def test_single_file_publication_handles_zero_write_and_preserves_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = campaign.os.write
    zero_out = tmp_path / "zero.json"
    zero_partial = tmp_path / ".zero.json.m6-partial"
    monkeypatch.setattr(campaign.os, "write", lambda _fd, _view: 0)
    with pytest.raises(campaign.M6Error) as zero_error:
        campaign._write_new(zero_out, {"value": 1})
    assert zero_error.value.code == "M6_OUTPUT_WRITE"
    assert zero_out.exists() is False
    assert zero_partial.exists() is True

    monkeypatch.setattr(campaign.os, "write", original_write)
    race_out = tmp_path / "race.json"
    race_partial = tmp_path / ".race.json.m6-partial"
    replaced = False

    def replace_partial_then_fail(descriptor: int, view: memoryview) -> int:
        nonlocal replaced
        if not replaced:
            replaced = True
            race_partial.unlink()
            race_partial.write_bytes(b"foreign evidence")
        return 0

    monkeypatch.setattr(campaign.os, "write", replace_partial_then_fail)
    with pytest.raises(campaign.M6Error) as replacement_error:
        campaign._write_new(race_out, {"value": 1})
    assert replacement_error.value.code == "M6_OUTPUT_WRITE"
    assert not race_out.exists()
    assert race_partial.read_bytes() == b"foreign evidence"


def test_single_file_publication_rejects_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    nested = real / "nested"
    nested.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    output = alias / "nested" / "schedule.json"
    with pytest.raises(campaign.M6Error) as error:
        campaign._write_new(output, {"value": 1})
    assert error.value.code == "M6_OUTPUT_PARENT"
    assert not (nested / "schedule.json").exists()


def test_event_transaction_rejects_generation_mixing(
    schedule: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slot = next(
        item for item in schedule["slots"]
        if item["config_id"] == campaign.REFERENCE_CONFIG
    )
    sidecar_path = tmp_path / "label.json"
    sentinel = tmp_path / "unreferenced-evidence.bin"
    _write_json(sidecar_path, _sidecar_for_slot(schedule, slot))
    sentinel.write_bytes(b"before")
    original = campaign._sidecar_ref_and_observations
    mutated = False

    def mutate_unreferenced_generation(*args, **kwargs):  # noqa: ANN002,ANN003
        nonlocal mutated
        result = original(*args, **kwargs)
        if not mutated:
            mutated = True
            sentinel.write_bytes(b"after!")
        return result

    monkeypatch.setattr(
        campaign,
        "_sidecar_ref_and_observations",
        mutate_unreferenced_generation,
    )
    with pytest.raises(campaign.M6Error) as error:
        campaign.build_observed_event(
            schedule,
            slot,
            [sidecar_path],
            journal_dir=tmp_path,
            observed_at_utc=FIXED_TIME,
            ncu_replay_pass_count=1,
        )
    assert error.value.code == "M6_TREE_CHANGED"


def _artifact(path: Path, root: Path, kind: str) -> dict:
    payload = path.read_bytes()
    return {
        "artifact_kind": kind,
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": campaign.sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _m4_sidecar(matrix: dict, matrix_cell: dict, schedule_cell: dict) -> dict:
    registry = json.loads(campaign._frozen_source_payload(
        campaign.REPO_ROOT,
        matrix["source_commit"],
        campaign.REGISTRY_PATH.relative_to(campaign.REPO_ROOT).as_posix(),
        role="fixture source registry",
    ))
    units = {row["ncu_metric"]: row["unit"] for row in registry["metrics"]}
    invocation = matrix_cell["invocation"]
    launch_index = invocation["capture_launch_contract"]["label_launch_index"]
    profile_from_start = invocation["shape_delivery"] == "application_args"
    return {
        "schema_version": 5,
        "device": "sm_86",
        "slug": schedule_cell["slug"],
        "shape_point": schedule_cell["shape_point"],
        "repeat_index": 0,
        "launch_index": launch_index,
        "binary_sha256": schedule_cell["binary_sha256"],
        "profiler_family": "nsight_compute",
        "profiler_version": "2026.1.0",
        "cuda_version": "13.2",
        "driver_version": "610.43.02",
        "gpu_uuid_sha256": "1" * 64,
        "replay_mode": "kernel",
        "aggregation": "kernel",
        "warmup_count": invocation["capture_launch_contract"][
            "label_driver_warmup_count"
        ],
        "cache_hit_rate_l2_sector": 0.5,
        "confidence_interval_l2_sector": {
            "lower": 0.5,
            "upper": 0.5,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        },
        "cache_hit_rate_l1tex_sector": 0.5,
        "confidence_interval_l1tex_sector": {
            "lower": 0.5,
            "upper": 0.5,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        },
        "unit_tagged_metrics": [
            {
                "ncu_metric": metric,
                "unit": units[metric],
                "value": 0.5 if units[metric] == "pct" else 1.0,
            }
            for metric in matrix["metric_registry"]["collection"][
                "selected_unit_metrics"
            ]
        ],
        "measurement_protocol": {
            "ncu_clock_control": "none",
            "ncu_cache_control": "all",
            "ncu_replay_mode": "kernel",
            "ncu_target_processes": "all",
            "ncu_filter_mode": "global",
            "ncu_profile_from_start": profile_from_start,
            "ncu_launch_skip": launch_index,
            "ncu_launch_count": 1,
            "host_persistence_mode": "Enabled",
            "host_sm_clock_mhz": 1350,
            "host_mem_clock_mhz": 10501,
            "host_temperature_c": 61,
        },
    }


def _fake_m4_archive(
    schedule: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, Path]:
    matrix = json.loads(campaign.MATRIX_PATH.read_text())
    matrix_cells = {
        (
            cell["identity"]["roster_source"],
            cell["identity"]["slug"],
        ): cell
        for cell in matrix["cells"]
    }
    schedule_cells = {cell["cell_id"]: cell for cell in schedule["cells"]}
    root = tmp_path / "m4"
    (root / "chunks").mkdir(parents=True)
    for chunk in matrix["chunks"]:
        chunk_root = root / "chunks" / chunk["chunk_id"]
        chunk_root.mkdir()
        rows = []
        for member in chunk["cells"]:
            source = member["roster_source"]
            slug = member["slug"]
            cell_id = f"{source}:{slug}"
            matrix_cell = matrix_cells[(source, slug)]
            schedule_cell = schedule_cells[cell_id]
            cell_root = chunk_root / "cells" / f"{source}__{slug}"
            cell_root.mkdir(parents=True)
            sidecar_path = cell_root / "label.json"
            run_path = cell_root / "run.json"
            report_path = cell_root / "report.ncu-rep"
            observation_path = cell_root / "launch.json"
            capture_path = cell_root / "capture.json"
            sidecar = _m4_sidecar(matrix, matrix_cell, schedule_cell)
            launch_index = sidecar["launch_index"]
            controls, _profile_from_start, shape_binding = (
                campaign._m4_expected_controls(matrix_cell, launch_index)
            )
            _write_json(sidecar_path, sidecar)
            signature = campaign.sha256_bytes(
                f"{slug}|{matrix_cell['invocation']['executable_path']}".encode()
            )
            run = {
                "schema_version": 1,
                "source_commit": matrix["source_commit"],
                "binary_sha256": schedule_cell["binary_sha256"],
                "kernel_signature_sha256": signature,
                "kernel_signature_source": "fallback_legacy",
                "arch": "sm_86",
                "slug": slug,
                "shape_point": str(
                    matrix_cell["invocation"]["logical_workload"]["value"]
                ),
                "repeat_index": 0,
                "launch_index": launch_index,
                "ncu_controls": controls,
            }
            _write_json(run_path, run)
            report_path.write_bytes(f"report:{cell_id}".encode())
            identity_columns = [
                "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
            ]
            observation = {
                "schema_version": 1,
                "kind": "ncu_details_csv_launch_observation",
                "kernel_name_base": "mangled",
                "requested_launch_skip": launch_index,
                "requested_launch_count": 1,
                "report_sha256": campaign.sha256_bytes(
                    report_path.read_bytes()
                ),
                "identity_columns": identity_columns,
                "identity": {
                    "ID": str(launch_index),
                    "Device": "NVIDIA GeForce RTX 3090 Ti (0)",
                    "Kernel Name": f"_Z_fixture_{slug}",
                    "Block Size": "(128, 1, 1)",
                    "Grid Size": "(1, 1, 1)",
                },
            }
            observation["observation_sha256"] = campaign.canonical_sha256(
                observation
            )
            _write_json(observation_path, observation)
            artifact_records = {
                "label_sidecar": _artifact(
                    sidecar_path, chunk_root, "label_sidecar"
                ),
                "label_run_manifest": _artifact(
                    run_path, chunk_root, "label_run_manifest"
                ),
                "ncu_report": _artifact(
                    report_path, chunk_root, "ncu_report"
                ),
                "ncu_launch_observation": _artifact(
                    observation_path, chunk_root, "ncu_launch_observation"
                ),
            }
            registry_payload = campaign._frozen_source_payload(
                campaign.REPO_ROOT,
                matrix["source_commit"],
                campaign.REGISTRY_PATH.relative_to(
                    campaign.REPO_ROOT
                ).as_posix(),
                role="fixture source registry",
            )
            capture = {
                "schema_version": 1,
                "identity": matrix_cell["identity"],
                "command": ["fixture-label-driver"],
                "binary_sha256": schedule_cell["binary_sha256"],
                "source_commit": matrix["source_commit"],
                "kernel_signature_sha256": signature,
                "kernel_signature_source": "fallback_legacy",
                "validated_artifact_receipt": {
                    "label_sidecar": {
                        "sha256": artifact_records["label_sidecar"]["sha256"],
                        "size_bytes": artifact_records["label_sidecar"][
                            "size_bytes"
                        ],
                    },
                    "label_run_manifest": {
                        "sha256": artifact_records["label_run_manifest"][
                            "sha256"
                        ],
                        "size_bytes": artifact_records["label_run_manifest"][
                            "size_bytes"
                        ],
                    },
                    "ncu_report": {
                        "sha256": artifact_records["ncu_report"]["sha256"],
                        "size_bytes": artifact_records["ncu_report"][
                            "size_bytes"
                        ],
                    },
                    "ncu_launch_observation": {
                        "sha256": artifact_records[
                            "ncu_launch_observation"
                        ]["sha256"],
                        "size_bytes": artifact_records[
                            "ncu_launch_observation"
                        ]["size_bytes"],
                        "observation_sha256": observation[
                            "observation_sha256"
                        ],
                    },
                    "metric_registry": {
                        "sha256": campaign.sha256_bytes(registry_payload),
                        "size_bytes": len(registry_payload),
                    },
                },
                "selected_unit_metrics": matrix["metric_registry"][
                    "collection"
                ]["selected_unit_metrics"],
                "selected_flat_metrics": matrix["metric_registry"][
                    "collection"
                ]["selected_flat_metrics"],
                "toolchain_contract": {
                    "expected_cuda_release": {"major": 13, "minor": 2},
                    "observed_cuda_release": {"major": 13, "minor": 2},
                    "expected_ncu_version_prefix": "2026.1",
                    "observed_ncu_version": "2026.1.0",
                },
                "publication_metric_scope": None,
                "logical_workload_sha256": campaign.canonical_sha256(
                    matrix_cell["invocation"]["logical_workload"]
                ),
                "capture_launch_contract": matrix_cell["invocation"][
                    "capture_launch_contract"
                ],
                "requested_launch_index": launch_index,
                "ncu_controls": controls,
                "shape_binding": shape_binding,
                "shape_forwarded_to_target": (
                    shape_binding == "application-args"
                ),
                "sol_phase_evidence": None,
                "elapsed_seconds": 1.0,
            }
            _write_json(capture_path, capture)
            artifact_records["label_capture_manifest"] = _artifact(
                capture_path, chunk_root, "label_capture_manifest"
            )
            rows.append({
                "roster_source": source,
                "slug": slug,
                "authority_cell_sha256": matrix_cell["authority_cell_sha256"],
                "disposition": "admitted",
                "typed_reason": None,
                "artifacts": list(artifact_records.values()),
            })
        done = {
            "schema_version": 2,
            "manifest_kind": "campaign_chunk_done",
            "chunk_id": chunk["chunk_id"],
            "output_root": str(chunk_root),
            "source_commit_pre": matrix["source_commit"],
            "source_commit_post": matrix["source_commit"],
            "source_head_at_seal": matrix["source_commit"],
            "source_commit_stable": True,
            "working_tree_clean": True,
            "protocol_version": "2.1",
            "arch": "sm_86",
            "capture_matrix_sha256": campaign.sha256_bytes(
                campaign.MATRIX_PATH.read_bytes()
            ),
            "matrix_seal_sha256": matrix["matrix_seal_sha256"],
            "roster_authority_sha256": matrix["authority"]["sha256"],
            "chunk_membership_sha256": chunk["membership_sha256"],
            "expected_cells": len(rows),
            "preflight_sha256": "e" * 64,
            "cell_results": rows,
        }
        _write_json(chunk_root / "DONE.json", done)
    _make_tree_readonly(root)

    def verified_done(*args, **kwargs):  # noqa: ANN002,ANN003
        assert len(args) == 1
        assert kwargs == {
            "repo_root": campaign.REPO_ROOT,
            "rehash_references": True,
            "require_current_head": True,
            "require_clean_tree": True,
        }
        return []

    monkeypatch.setattr(campaign.m4_done, "verify_seal", verified_done)
    monkeypatch.setattr(
        campaign.matrix_builder, "validate_matrix", lambda *args, **kwargs: []
    )
    return matrix, root


def _schedule_with_complete_m4(
    schedule: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, Path, dict]:
    matrix, root = _fake_m4_archive(schedule, tmp_path, monkeypatch)
    receipt = campaign.build_m4_binding_receipt(
        archive_root=root,
        generated_at_utc=FIXED_TIME,
    )
    complete = copy.deepcopy(schedule)
    complete["m4_identity"] = receipt
    complete["schedule_sha256"] = campaign.canonical_sha256(
        campaign._without(complete, "schedule_sha256")
    )
    campaign.validate_schedule(complete)
    return matrix, root, complete


def _repeat_evidence_bundle(
    schedule: dict,
    slot: dict,
    journal_dir: Path,
    *,
    kernel_name: str | None = None,
    kernel_signature_sha256: str | None = None,
) -> tuple[Path, dict[str, Path], dict]:
    journal_dir.mkdir(parents=True)
    cell = next(item for item in schedule["cells"] if item["cell_id"] == slot["cell_id"])
    config = next(
        item for item in schedule["configs"] if item["config_id"] == slot["config_id"]
    )
    m4_cell = next(
        item
        for item in schedule["m4_identity"]["cell_launch_identities"]
        if item["cell_id"] == slot["cell_id"]
    )
    assert len(m4_cell["launches"]) == 1
    m4_launch = m4_cell["launches"][0]
    launch_index = m4_launch["launch_index"]
    observed_kernel = kernel_name or m4_launch["kernel_name"]
    observed_signature = (
        kernel_signature_sha256 or m4_launch["kernel_signature_sha256"]
    )
    signature_source = m4_launch["kernel_signature_source"]

    sidecar_path = journal_dir / "repeat_label.json"
    run_path = journal_dir / "repeat_run.json"
    report_path = journal_dir / "repeat_report.ncu-rep"
    observation_path = journal_dir / "repeat_launch.json"
    capture_path = journal_dir / "repeat_capture.json"

    sidecar = _sidecar_for_slot(schedule, slot)
    sidecar["launch_index"] = launch_index
    sidecar["measurement_protocol"]["ncu_launch_skip"] = launch_index
    _write_json(sidecar_path, sidecar)
    controls = campaign._m6_repeat_controls(config, launch_index)
    run = {
        "schema_version": 1,
        "source_commit": schedule["source_identity"]["matrix_source_commit"],
        "binary_sha256": cell["binary_sha256"],
        "kernel_signature_sha256": observed_signature,
        "kernel_signature_source": signature_source,
        "arch": "sm_86",
        "slug": cell["slug"],
        "shape_point": cell["shape_point"],
        "repeat_index": slot["repeat_index"],
        "launch_index": launch_index,
        "ncu_controls": controls,
    }
    _write_json(run_path, run)
    report_path.write_bytes(f"repeat-report:{slot['slot_id']}".encode())
    identity_columns = [
        "ID", "Device", "Kernel Name", "Block Size", "Grid Size",
    ]
    observation = {
        "schema_version": 1,
        "kind": "ncu_details_csv_launch_observation",
        "kernel_name_base": "mangled",
        "requested_launch_skip": launch_index,
        "requested_launch_count": 1,
        "report_sha256": campaign.sha256_bytes(report_path.read_bytes()),
        "identity_columns": identity_columns,
        "identity": {
            "ID": str(launch_index),
            "Device": "NVIDIA GeForce RTX 3090 Ti (0)",
            "Kernel Name": observed_kernel,
            "Block Size": "(128, 1, 1)",
            "Grid Size": "(1, 1, 1)",
        },
    }
    observation["observation_sha256"] = campaign.canonical_sha256(observation)
    _write_json(observation_path, observation)

    def receipt(path: Path) -> dict:
        payload = path.read_bytes()
        return {
            "sha256": campaign.sha256_bytes(payload),
            "size_bytes": len(payload),
        }

    artifact_receipt = {
        "label_sidecar": receipt(sidecar_path),
        "label_run_manifest": receipt(run_path),
        "ncu_report": receipt(report_path),
        "ncu_launch_observation": {
            **receipt(observation_path),
            "observation_sha256": observation["observation_sha256"],
        },
    }
    capture = {
        "schema_version": 1,
        "artifact_kind": "m6_repeat_capture_manifest",
        "source_commit": schedule["source_identity"]["matrix_source_commit"],
        "schedule_sha256": schedule["schedule_sha256"],
        "slot_id": slot["slot_id"],
        "slot_sha256": slot["slot_sha256"],
        "cell_id": cell["cell_id"],
        "cell_identity_sha256": cell["cell_identity_sha256"],
        "config_id": config["config_id"],
        "config_sha256": config["config_sha256"],
        "repeat_index": slot["repeat_index"],
        "launch_index": launch_index,
        "binary_sha256": cell["binary_sha256"],
        "kernel_name": observed_kernel,
        "kernel_signature_sha256": observed_signature,
        "kernel_signature_source": signature_source,
        "ncu_controls": controls,
        "validated_artifact_receipt": artifact_receipt,
    }
    capture["capture_manifest_sha256"] = campaign.canonical_sha256(capture)
    _write_json(capture_path, capture)
    return sidecar_path, {
        "label_run_manifest": run_path,
        "ncu_report": report_path,
        "ncu_launch_observation": observation_path,
        "label_capture_manifest": capture_path,
    }, m4_launch


def test_complete_m4_receipt_is_rederived_from_verified_done_artifacts(
    schedule: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix, root = _fake_m4_archive(schedule, tmp_path, monkeypatch)
    receipt = campaign.build_m4_binding_receipt(
        archive_root=root,
        generated_at_utc=FIXED_TIME,
    )
    assert len(receipt["chunk_done_manifests"]) == 18
    assert len(receipt["cell_launch_identities"]) == 122


def test_snapshot_source_binding_deduplicates_frozen_alias_payloads(
    schedule: dict,
) -> None:
    authorities = campaign._source_authority_rows(schedule)
    frozen_pairs = {
        (
            schedule["source_identity"][key]["path"],
            schedule["source_identity"][key]["sha256"],
        )
        for key in ("frozen_metric_registry", "frozen_sidecar_schema")
    }
    authority_pairs = [(path, digest) for path, digest, _size in authorities]
    assert all(authority_pairs.count(pair) == 1 for pair in frozen_pairs)
    witness: list[dict] = []
    for relative, digest, size in authorities:
        path = campaign.REPO_ROOT / relative
        metadata = path.stat()
        identity = {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": metadata.st_mode,
            "nlink": metadata.st_nlink,
            "size_bytes": metadata.st_size,
        }
        witness.append({
            "canonical_path": str(path),
            "observed_sha256": digest,
            "observed_size_bytes": size,
            "pre_identity": identity,
            "post_identity": identity,
        })
    campaign._validate_snapshot_witness(witness)
    campaign._validate_snapshot_source_bindings(
        witness, schedule, repo_root=campaign.REPO_ROOT
    )
    witness[0]["observed_sha256"] = "0" * 64
    with pytest.raises(campaign.M6Error) as error:
        campaign._validate_snapshot_source_bindings(
            witness, schedule, repo_root=campaign.REPO_ROOT
        )
    assert error.value.code == "M6_SOURCE_SNAPSHOT_BINDING"


def test_relocated_done_requires_bundled_evidence_namespace(
    tmp_path: Path,
) -> None:
    with pytest.raises(campaign.M6Error) as error:
        campaign.verify_analysis_bundle(
            schedule={},
            journal={},
            analysis={},
            schedule_path=tmp_path / "schedule.json",
            journal_path=tmp_path / "journal.json",
            analysis_path=tmp_path / "analysis.json",
            done={},
            done_path=tmp_path / "bundle" / "M6_DONE.json",
            evidence_root=tmp_path / "external-evidence",
            bundle_root=tmp_path / "bundle",
        )
    assert error.value.code == "M6_EVIDENCE_NAMESPACE"


def test_recovery_receipt_admission_requires_sealed_owner_bindings() -> None:
    import m6_recover_staging as recovery

    schedule = {"schedule_sha256": "1" * 64}
    journal = {"journal_sha256": "2" * 64}
    analysis = {"analysis_sha256": "3" * 64}
    receipt = recovery._receipt(  # noqa: SLF001
        decision="EXACT_PREFIX_COMPLETE",
        staging_root=Path("/tmp/staging"),
        inventory=[{
            "relative_path": "value",
            "kind": "file",
            "sha256": "0" * 64,
            "size_bytes": 0,
            "device": 0,
            "inode": 1,
            "mode": 0o444,
            "nlink": 1,
        }],
        schedule_sha256=schedule["schedule_sha256"],
        journal_sha256=journal["journal_sha256"],
        analysis_sha256=analysis["analysis_sha256"],
        published_path="/tmp/published/artifact",
    )
    accepted = campaign._validate_recovery_receipt_payload(  # noqa: SLF001
        campaign.canonical_bytes(receipt),
        schedule=schedule,
        journal=journal,
        analysis=analysis,
        repo_root=campaign.REPO_ROOT,
        cache=campaign.InputSnapshotCache(),
    )
    assert accepted == receipt
    forged = copy.deepcopy(receipt)
    forged["schedule_sha256"] = "4" * 64
    forged["receipt_sha256"] = campaign.immutable.model_seal(
        forged, "receipt_sha256"
    )
    with pytest.raises(campaign.M6Error) as error:
        campaign._validate_recovery_receipt_value(
            forged,
            schema_value=json.loads(campaign.RECOVERY_SCHEMA_PATH.read_text()),
            schedule=schedule,
            journal=journal,
            analysis=analysis,
        )
    assert error.value.code == "M6_RECOVERY_IDENTITY_MISMATCH"
    campaign._validate_m4_identity(
        receipt,
        matrix=matrix,
        matrix_sha256=campaign.sha256_bytes(campaign.MATRIX_PATH.read_bytes()),
        cells=schedule["cells"],
        repo_root=campaign.REPO_ROOT,
        require_complete=True,
    )

    fake = copy.deepcopy(receipt)
    fake_launch = fake["cell_launch_identities"][0]["launches"][0]
    fake_launch["kernel_name"] += "_fake"
    fake_launch["launch_identity_sha256"] = campaign.canonical_sha256(
        campaign._without(fake_launch, "launch_identity_sha256")
    )
    fake["cell_launch_identities"][0]["launch_set_sha256"] = (
        campaign.canonical_sha256(
            fake["cell_launch_identities"][0]["launches"]
        )
    )
    fake["identity_sha256"] = campaign.canonical_sha256(
        campaign._without(fake, "identity_sha256")
    )
    with pytest.raises(campaign.M6Error) as error:
        campaign._validate_m4_identity(
            fake,
            matrix=matrix,
            matrix_sha256=campaign.sha256_bytes(
                campaign.MATRIX_PATH.read_bytes()
            ),
            cells=schedule["cells"],
            repo_root=campaign.REPO_ROOT,
            require_complete=True,
        )
    assert error.value.code == "M6_M4_RECEIPT_REDERIVATION"


def test_bind_m4_refuses_authoritative_done_verifier_failure(
    schedule: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _matrix, root = _fake_m4_archive(schedule, tmp_path, monkeypatch)
    monkeypatch.setattr(
        campaign.m4_done,
        "verify_seal",
        lambda *args, **kwargs: ["forged DONE"],
    )
    with pytest.raises(campaign.M6Error) as error:
        campaign.build_m4_binding_receipt(
            archive_root=root,
            generated_at_utc=FIXED_TIME,
        )
    assert error.value.code == "M6_M4_DONE_UNVERIFIED"


def test_m4_binding_rejects_cross_generation_tree(
    schedule: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _matrix, root = _fake_m4_archive(schedule, tmp_path, monkeypatch)
    sentinel = root / "unreferenced-evidence.bin"
    root.chmod(0o755)
    sentinel.write_bytes(b"before")
    sentinel.chmod(0o444)
    root.chmod(0o555)
    mutated = False

    def mutate_tree(*args, **kwargs):  # noqa: ANN002,ANN003
        nonlocal mutated
        if not mutated:
            mutated = True
            sentinel.chmod(0o644)
            sentinel.write_bytes(b"after!")
            sentinel.chmod(0o444)
        return []

    monkeypatch.setattr(campaign.m4_done, "verify_seal", mutate_tree)
    with pytest.raises(campaign.M6Error) as error:
        campaign.build_m4_binding_receipt(
            archive_root=root,
            generated_at_utc=FIXED_TIME,
        )
    assert error.value.code == "M6_TREE_CHANGED"


def test_bind_m4_refuses_cross_root_done_and_no_overwrite(
    schedule: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix, root = _fake_m4_archive(schedule, tmp_path, monkeypatch)
    chunk_id = matrix["chunks"][0]["chunk_id"]
    done_path = root / "chunks" / chunk_id / "DONE.json"
    done = json.loads(done_path.read_text())
    done["output_root"] = str(tmp_path / "different-root")
    done_path.chmod(0o644)
    _write_json(done_path, done)
    done_path.chmod(0o444)
    with pytest.raises(campaign.M6Error) as error:
        campaign.build_m4_binding_receipt(
            archive_root=root,
            generated_at_utc=FIXED_TIME,
        )
    assert error.value.code == "M6_M4_DONE_BINDING"

    _matrix, second_root = _fake_m4_archive(
        schedule, tmp_path / "second", monkeypatch,
    )
    out = tmp_path / "receipt.json"
    assert campaign.main([
        "bind-m4", "--m4-root", str(second_root),
        "--generated-at-utc", FIXED_TIME, "--out", str(out),
    ]) == 0
    before = out.read_bytes()
    assert campaign.main([
        "bind-m4", "--m4-root", str(second_root),
        "--generated-at-utc", FIXED_TIME, "--out", str(out),
    ]) == 2
    assert out.read_bytes() == before


def test_m4_binding_refuses_authority_overlap_partial_and_missing_capture(
    schedule: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix, root = _fake_m4_archive(schedule, tmp_path, monkeypatch)
    receipt = campaign.build_m4_binding_receipt(
        archive_root=root,
        generated_at_utc=FIXED_TIME,
    )
    writable_done = root / "chunks" / matrix["chunks"][0]["chunk_id"] / "DONE.json"
    writable_done.chmod(0o644)
    with pytest.raises(campaign.M6Error) as writable_error:
        campaign.build_m4_binding_receipt(
            archive_root=root,
            generated_at_utc=FIXED_TIME,
        )
    assert writable_error.value.code == "M6_TREE_SNAPSHOT"
    writable_done.chmod(0o444)
    matrix_sha256 = campaign.sha256_bytes(campaign.MATRIX_PATH.read_bytes())

    incomplete = copy.deepcopy(schedule["m4_identity"])
    incomplete["completed_cells"] = 121
    incomplete["identity_sha256"] = campaign.canonical_sha256(
        campaign._without(incomplete, "identity_sha256")
    )
    with pytest.raises(campaign.M6Error) as incomplete_error:
        campaign._validate_m4_identity(
            incomplete,
            matrix=matrix,
            matrix_sha256=matrix_sha256,
            cells=schedule["cells"],
            repo_root=campaign.REPO_ROOT,
            require_complete=False,
        )
    assert incomplete_error.value.code == "M6_M4_INCOMPLETE_IDENTITY"

    partial = copy.deepcopy(receipt)
    partial["completed_cells"] = 121
    partial["cell_launch_identities"].pop()
    partial["identity_sha256"] = campaign.canonical_sha256(
        campaign._without(partial, "identity_sha256")
    )
    with pytest.raises(campaign.M6Error) as partial_error:
        campaign._validate_m4_identity(
            partial,
            matrix=matrix,
            matrix_sha256=matrix_sha256,
            cells=schedule["cells"],
            repo_root=campaign.REPO_ROOT,
            require_complete=True,
        )
    assert partial_error.value.code == "M6_SCHEMA_VALIDATION"

    for output in (
        root / "overlapping-receipt.json",
        campaign.REPO_ROOT / ".m6-overlapping-receipt-test.json",
    ):
        assert not output.exists()
        with pytest.raises(campaign.M6Error) as overlap_error:
            campaign._authorize_m4_receipt_output(
                output,
                archive_root=root,
                repo_root=campaign.REPO_ROOT,
            )
        assert overlap_error.value.code == "M6_M4_RECEIPT_OUTPUT_OVERLAP"
        assert not output.exists()

    alternate_matrix = tmp_path / "alternate-matrix.json"
    alternate_matrix.write_bytes(campaign.MATRIX_PATH.read_bytes())
    with pytest.raises(campaign.M6Error) as matrix_error:
        campaign.build_m4_binding_receipt(
            archive_root=root,
            matrix_path=alternate_matrix,
            generated_at_utc=FIXED_TIME,
        )
    assert matrix_error.value.code == "M6_SOURCE_AUTHORITY"

    alternate_repo = tmp_path / "alternate-repository"
    alternate_repo.mkdir()
    with pytest.raises(campaign.M6Error) as repository_error:
        campaign.build_m4_binding_receipt(
            archive_root=root,
            repo_root=alternate_repo,
            generated_at_utc=FIXED_TIME,
        )
    assert repository_error.value.code == "M6_REPOSITORY_AUTHORITY"

    chunk_id = matrix["chunks"][0]["chunk_id"]
    done_path = root / "chunks" / chunk_id / "DONE.json"
    done = json.loads(done_path.read_text())
    done["cell_results"][0]["artifacts"] = [
        artifact
        for artifact in done["cell_results"][0]["artifacts"]
        if artifact["artifact_kind"] != "label_capture_manifest"
    ]
    done_path.chmod(0o644)
    _write_json(done_path, done)
    done_path.chmod(0o444)
    with pytest.raises(campaign.M6Error) as capture_error:
        campaign.build_m4_binding_receipt(
            archive_root=root,
            generated_at_utc=FIXED_TIME,
        )
    assert capture_error.value.code == "M6_M4_ARTIFACT_CARDINALITY"


def test_complete_m4_repeat_evidence_rejects_same_index_wrong_kernel(
    schedule: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _matrix, _root, complete = _schedule_with_complete_m4(
        schedule, tmp_path / "archive", monkeypatch,
    )
    slot = complete["slots"][0]
    valid_dir = tmp_path / "valid-repeat"
    sidecar_path, evidence_paths, m4_launch = _repeat_evidence_bundle(
        complete, slot, valid_dir,
    )
    event = campaign.build_observed_event(
        complete,
        slot,
        [sidecar_path],
        journal_dir=valid_dir,
        observed_at_utc=FIXED_TIME,
        ncu_replay_pass_count=2,
        launch_evidence_paths=[evidence_paths],
    )
    assert event["sidecars"][0]["m4_launch_identity_sha256"] == (
        m4_launch["launch_identity_sha256"]
    )
    assert event["sidecars"][0]["launch_evidence"]["kernel_name"] == (
        m4_launch["kernel_name"]
    )
    journal = campaign.build_journal(
        complete, [event], created_at_utc=FIXED_TIME,
    )
    journal_path = valid_dir / "journal.json"
    _write_json(journal_path, journal)
    campaign.validate_journal(complete, journal, journal_path=journal_path)

    missing_capture = dict(evidence_paths)
    missing_capture.pop("label_capture_manifest")
    with pytest.raises(campaign.M6Error) as missing_error:
        campaign.build_observed_event(
            complete,
            slot,
            [sidecar_path],
            journal_dir=valid_dir,
            observed_at_utc=FIXED_TIME,
            ncu_replay_pass_count=2,
            launch_evidence_paths=[missing_capture],
        )
    assert missing_error.value.code == "M6_REPEAT_EVIDENCE_PATHS"

    wrong_dir = tmp_path / "wrong-kernel-repeat"
    wrong_sidecar, wrong_evidence, _ = _repeat_evidence_bundle(
        complete,
        slot,
        wrong_dir,
        kernel_name=f"{m4_launch['kernel_name']}_different_kernel",
        kernel_signature_sha256="f" * 64,
    )
    with pytest.raises(campaign.M6Error) as wrong_kernel_error:
        campaign.build_observed_event(
            complete,
            slot,
            [wrong_sidecar],
            journal_dir=wrong_dir,
            observed_at_utc=FIXED_TIME,
            ncu_replay_pass_count=2,
            launch_evidence_paths=[wrong_evidence],
        )
    assert wrong_kernel_error.value.code == "M6_M4_KERNEL_IDENTITY"


def test_collect_requires_complete_m4_before_disabled(schedule: dict, tmp_path: Path, capsys) -> None:
    schedule_path = tmp_path / "schedule.json"
    _write_json(schedule_path, schedule)
    assert campaign.main(["collect", "--schedule", str(schedule_path)]) == 2
    assert "M6_M4_INCOMPLETE" in capsys.readouterr().err


def test_collect_is_unconditionally_disabled_after_prerequisite_gate(
    schedule: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    schedule_path = tmp_path / "schedule.json"
    _write_json(schedule_path, schedule)
    original = campaign._validate_m4_identity

    def allow_complete(*args, **kwargs):  # noqa: ANN002,ANN003
        if kwargs.get("require_complete"):
            return None
        return original(*args, **kwargs)

    monkeypatch.setattr(campaign, "_validate_m4_identity", allow_complete)
    assert campaign.main(["collect", "--schedule", str(schedule_path)]) == 2
    assert "M6_COLLECT_DISABLED_OFFLINE_SOURCE" in capsys.readouterr().err


def test_external_v5_oor_value_is_retained_uncensored(schedule: dict, tmp_path: Path) -> None:
    slot = next(slot for slot in schedule["slots"] if slot["config_id"] == campaign.REFERENCE_CONFIG)
    sidecar_path = tmp_path / "label.json"
    _write_json(sidecar_path, _sidecar_for_slot(schedule, slot, value=1.25))
    event = campaign.build_observed_event(
        schedule,
        slot,
        [sidecar_path],
        journal_dir=tmp_path,
        observed_at_utc=FIXED_TIME,
        ncu_replay_pass_count=3,
    )
    journal = campaign.build_journal(schedule, [event], created_at_utc=FIXED_TIME)
    journal_path = tmp_path / "journal.json"
    _write_json(journal_path, journal)
    analysis = campaign.analyze(
        schedule,
        journal,
        journal_path=journal_path,
        generated_at_utc=FIXED_TIME,
        bootstrap_replicates=10,
    )
    raw = next(
        row
        for row in analysis["row_accounting"]
        if row["metric_key"] == "cache_hit_rate_l2_sector"
        and row["value"] is not None
    )
    assert raw["value"] == 1.25
    assert raw["validity"] == "out_of_range_high"
    assert raw["typed_reason"] == "ncu_pct_overflow"
    classification = next(
        row
        for row in analysis["classifications"]
        if row["cell_id"] == slot["cell_id"]
        and row["config_id"] == campaign.REFERENCE_CONFIG
        and row["metric_key"] == "cache_hit_rate_l2_sector"
    )
    assert classification["determinism_class"] == "vendor_defect"
    assert analysis["analysis_policy"]["raw_values_clamped"] is False
    decision = next(
        row for row in analysis["trigger_decisions"] if row["cell_id"] == slot["cell_id"]
    )
    assert decision["triggered"] is True
    assert "out_of_range" in decision["reason_codes"]
    assert analysis["accounting"]["collected_identity_holds"] is True
    assert analysis["accounting"]["denominator_identity_holds"] is True


def test_parser_repair_diagnostic_is_replaced_by_uncensored_source_value(
    schedule: dict, tmp_path: Path
) -> None:
    slot = next(slot for slot in schedule["slots"] if slot["config_id"] == campaign.REFERENCE_CONFIG)
    sidecar = _sidecar_for_slot(schedule, slot, value=1.0)
    sidecar["parser_diagnostics"] = {
        "ncu_sector_hit_rate_clamped_to_100": [
            {
                "metric_label": "lts__t_sector_hit_rate.pct",
                "raw_value": 103.25,
            }
        ]
    }
    sidecar_path = tmp_path / "diagnostic.json"
    _write_json(sidecar_path, sidecar)
    event = campaign.build_observed_event(
        schedule,
        slot,
        [sidecar_path],
        journal_dir=tmp_path,
        observed_at_utc=FIXED_TIME,
        ncu_replay_pass_count=2,
    )
    observation = next(
        row
        for row in event["raw_observations"]
        if row["metric_key"] == "cache_hit_rate_l2_sector"
    )
    assert observation["value"] == pytest.approx(1.0325)
    assert observation["source_emitted_value"] == 103.25
    assert observation["source_emitted_unit"] == "percent"
    assert observation["source_pointer"].startswith("/parser_diagnostics/")


def test_boolean_raw_value_and_admitted_summary_fail_closed(schedule: dict, tmp_path: Path) -> None:
    slot = next(slot for slot in schedule["slots"] if slot["config_id"] == campaign.REFERENCE_CONFIG)
    bool_path = tmp_path / "bool.json"
    _write_json(bool_path, _sidecar_for_slot(schedule, slot, value=True))
    with pytest.raises(campaign.M6Error) as bool_error:
        campaign.build_observed_event(
            schedule,
            slot,
            [bool_path],
            journal_dir=tmp_path,
            observed_at_utc=FIXED_TIME,
            ncu_replay_pass_count=1,
        )
    assert bool_error.value.code in {"M6_SIDECAR_V5_STRUCTURE", "M6_NUMERIC_VALUE"}

    admitted_path = tmp_path / "label_admitted.json"
    _write_json(admitted_path, {"schema_version": 5, "cells": []})
    with pytest.raises(campaign.M6Error) as admitted_error:
        campaign.build_observed_event(
            schedule,
            slot,
            [admitted_path],
            journal_dir=tmp_path,
            observed_at_utc=FIXED_TIME,
            ncu_replay_pass_count=1,
        )
    assert admitted_error.value.code == "M6_SIDECAR_V5_STRUCTURE"


def test_journal_publication_is_exclusive_and_readonly(
    schedule: dict,
    tmp_path: Path,
) -> None:
    journal = campaign.build_journal(
        schedule,
        [
            campaign.build_excluded_event(
                schedule["slots"][0],
                reason="operator_stop",
                detail="fixture",
                operator_authority_receipt=OPERATOR_RECEIPT,
            )
        ],
        created_at_utc=FIXED_TIME,
    )
    output = tmp_path / "journal.json"
    campaign.publish_journal(schedule, journal, output=output)
    before = output.read_bytes()
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(campaign.M6Error) as overwrite:
        campaign.publish_journal(schedule, journal, output=output)
    assert overwrite.value.code == "M6_NO_OVERWRITE"
    assert output.read_bytes() == before


def test_journal_hash_chain_and_sidecar_immutability_are_verified(
    schedule: dict, tmp_path: Path
) -> None:
    slot = next(slot for slot in schedule["slots"] if slot["config_id"] == campaign.REFERENCE_CONFIG)
    sidecar_path = tmp_path / "label.json"
    _write_json(sidecar_path, _sidecar_for_slot(schedule, slot))
    event = campaign.build_observed_event(
        schedule,
        slot,
        [sidecar_path],
        journal_dir=tmp_path,
        observed_at_utc=FIXED_TIME,
        ncu_replay_pass_count=2,
    )
    journal = campaign.build_journal(schedule, [event], created_at_utc=FIXED_TIME)
    journal_path = tmp_path / "journal.json"
    _write_json(journal_path, journal)
    campaign.validate_journal(schedule, journal, journal_path=journal_path)

    tampered = copy.deepcopy(journal)
    tampered["events"][0]["raw_observations"][0]["value"] = 0.75
    tampered["events"][0]["event_sha256"] = campaign.canonical_sha256(
        campaign._without(tampered["events"][0], "event_sha256")
    )
    tampered["journal_sha256"] = campaign.canonical_sha256(
        campaign._without(tampered, "journal_sha256")
    )
    with pytest.raises(campaign.M6Error) as error:
        campaign.validate_journal(schedule, tampered, journal_path=journal_path)
    assert error.value.code == "M6_RAW_OBSERVATION_MISMATCH"

    sidecar = json.loads(sidecar_path.read_text())
    sidecar["cache_hit_rate_l2_sector"] = 0.6
    _write_json(sidecar_path, sidecar)
    with pytest.raises(campaign.M6Error) as changed:
        campaign.validate_journal(schedule, journal, journal_path=journal_path)
    assert changed.value.code == "M6_SIDECAR_REFERENCE"


def test_tier2_slot_is_rejected_without_trigger_decision(schedule: dict) -> None:
    event = campaign.build_excluded_event(
        schedule["tier2_candidates"][0],
        reason="operator_stop",
        detail="offline fixture",
        operator_authority_receipt=OPERATOR_RECEIPT,
    )
    with pytest.raises(campaign.M6Error) as error:
        campaign.build_journal(schedule, [event], created_at_utc=FIXED_TIME)
    assert error.value.code == "M6_JOURNAL_SLOT"


def test_require_done_failure_leaves_no_partial_output(
    schedule: dict, tmp_path: Path, capsys
) -> None:
    slot = schedule["slots"][0]
    journal = campaign.build_journal(
        schedule,
        [campaign.build_excluded_event(
            slot,
            reason="operator_stop",
            detail="fixture",
            operator_authority_receipt=OPERATOR_RECEIPT,
        )],
        created_at_utc=FIXED_TIME,
    )
    schedule_path = tmp_path / "schedule.json"
    journal_path = tmp_path / "journal.json"
    out_dir = tmp_path / "analysis"
    _write_json(schedule_path, schedule)
    _write_json(journal_path, journal)
    assert campaign.main(
        [
            "analyze",
            "--schedule",
            str(schedule_path),
            "--journal",
            str(journal_path),
            "--out-dir",
            str(out_dir),
            "--bootstrap-replicates",
            "2",
            "--generated-at-utc",
            FIXED_TIME,
            "--require-done",
        ]
    ) == 2
    assert "M6_DONE_NOT_EMITTED" in capsys.readouterr().err
    assert not out_dir.exists()


def test_analysis_publication_is_exact_readonly_and_relocatable(
    schedule: dict,
    tmp_path: Path,
) -> None:
    slot = schedule["slots"][0]
    journal = campaign.build_journal(
        schedule,
        [campaign.build_excluded_event(
            slot,
            reason="operator_stop",
            detail="fixture",
            operator_authority_receipt=OPERATOR_RECEIPT,
        )],
        created_at_utc=FIXED_TIME,
    )
    schedule_path = tmp_path / "schedule.json"
    journal_path = tmp_path / "journal.json"
    out_dir = tmp_path / "analysis"
    _write_json(schedule_path, schedule)
    _write_json(journal_path, journal)
    args = [
        "analyze",
        "--schedule", str(schedule_path),
        "--journal", str(journal_path),
        "--out-dir", str(out_dir),
        "--bootstrap-replicates", "2",
        "--generated-at-utc", FIXED_TIME,
    ]
    assert campaign.main(args) == 0
    assert {path.name for path in out_dir.iterdir()} == {"determinism_study.json"}
    assert out_dir.stat().st_mode & 0o222 == 0
    assert (out_dir / "determinism_study.json").stat().st_mode & 0o222 == 0
    before = (out_dir / "determinism_study.json").read_bytes()
    assert campaign.main(args) == 2
    assert (out_dir / "determinism_study.json").read_bytes() == before

    relocated = tmp_path / "relocated-analysis"
    out_dir.rename(relocated)
    assert campaign.main([
        "verify",
        "--schedule", str(schedule_path),
        "--journal", str(journal_path),
        "--analysis-root", str(relocated),
    ]) == 0


def test_analysis_publication_refuses_repository_overlap(tmp_path: Path) -> None:
    output = campaign.REPO_ROOT / ".m6-analysis-overlap-test"
    assert not output.exists()
    with pytest.raises(campaign.M6Error) as error:
        campaign._publish_analysis_directory(
            output,
            analysis_payload=b"{}\n",
            done_payload=None,
            repo_root=campaign.REPO_ROOT,
        )
    assert error.value.code == "M6_ANALYSIS_OUTPUT_OVERLAP"
    assert not output.exists()

    reserved = tmp_path / ".victim.staging"
    with pytest.raises(campaign.M6Error) as reserved_error:
        campaign._publish_analysis_directory(
            reserved,
            analysis_payload=b"{}\n",
            done_payload=None,
            repo_root=campaign.REPO_ROOT,
        )
    assert reserved_error.value.code == "M6_OUTPUT_PATH"
    assert not reserved.exists()


def test_analysis_schema_rejects_boolean_raw_value(schedule: dict, tmp_path: Path) -> None:
    slot = schedule["slots"][0]
    journal = campaign.build_journal(
        schedule,
        [campaign.build_excluded_event(
            slot,
            reason="operator_stop",
            detail="fixture",
            operator_authority_receipt=OPERATOR_RECEIPT,
        )],
        created_at_utc=FIXED_TIME,
    )
    journal_path = tmp_path / "journal.json"
    _write_json(journal_path, journal)
    analysis = campaign.analyze(
        schedule,
        journal,
        journal_path=journal_path,
        generated_at_utc=FIXED_TIME,
        bootstrap_replicates=2,
    )
    assert analysis["analysis_policy"]["reference_oor_zero_confirmed"] is False
    schema = _schema("analysis")
    bad = copy.deepcopy(analysis)
    bad["analysis_policy"]["bootstrap_seed"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(bad)
