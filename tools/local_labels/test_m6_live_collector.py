from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import m6_determinism_campaign as m6
from tools.local_labels import m6_live_collector as live


H = "a" * 64
H2 = "b" * 64
COMMIT = "1" * 40
NOW = "2026-08-08T12:00:00Z"


def _closed(schema: object) -> None:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
        for value in schema.values():
            _closed(value)
    elif isinstance(schema, list):
        for value in schema:
            _closed(value)


def _sealed(value: dict, field: str) -> dict:
    return immutable.seal_model(value, field)


def _operator(*, output_root: str, maximum: int, gpu: str = H) -> dict:
    return _sealed(
        {
            "kind": "m6_operator_hardware_authority_v1",
            "authority_id": "offline-test-no-hardware-credit",
            "authorized_at_utc": NOW,
            "source_commit": COMMIT,
            "gpu_uuid_sha256": gpu,
            "driver_version": live.GENERIC_DRIVER_VERSION,
            "output_root": output_root,
            "max_fresh_processes_per_resume": maximum,
            "scope": live.SCOPE,
            "allow_hardware_queries": True,
            "allow_hardware_spawns": True,
            "allow_privileged_commands": False,
        },
        "authority_sha256",
    )


def _session(*, maximum: int = 3, output_root: str = "/tmp/m6-live-test") -> dict:
    session_id = "m6-live-" + "c" * 24
    root = f"{output_root}/m6-live/{session_id}"
    file_ref = {"path": "x.json", "sha256": H, "size_bytes": 1}
    preflights = [
        {
            "phase": phase,
            "path": f"/tmp/{phase}.json",
            "sha256": character * 64,
            "size_bytes": 1,
            "generated_at_utc": NOW,
            "git_head": COMMIT,
            "pins_sha256": H,
            "matrix_sha256": H2,
            "checks_sha256": "d" * 64,
            "gpu_uuid_sha256": H,
            "driver_version": live.GENERIC_DRIVER_VERSION,
            "output_root": output_root,
            "capture_admitted": True,
        }
        for phase, character in (("p3", "3"), ("p4", "4"), ("session", "5"))
    ]
    unsigned = {
        "schema_version": 1,
        "kind": "m6_live_session_authority_v1",
        "trust_mode": live.FIXTURE_TRUST_MODE,
        "session_id": session_id,
        "issued_at_utc": NOW,
        "source": {
            "commit": COMMIT,
            "tree": "2" * 40,
            "working_tree_clean": True,
            "p1_gate_receipt_sha256": "6" * 64,
            "current_source_anchor_sha256": "7" * 64,
            "clean_s_p2_receipt_sha256": "8" * 64,
            "live_v1_authority_file_sha256": "9" * 64,
            "successor_matrix_file_sha256": "e" * 64,
        },
        "admission_prerequisites": {
            "trust_anchor": {
                "path": "/tmp/m6-trust-anchor.json",
                "sha256": H,
                "size_bytes": 1,
                "model_sha256": H2,
                "trust_anchor_id": "fixture-only",
            },
            "evidence": {
                name: {
                    "path": f"/tmp/{name}.json",
                    "sha256": H,
                    "size_bytes": 1,
                    **(
                        {}
                        if name in {"live_v1_authority", "successor_matrix"}
                        else {"model_sha256": H2}
                    ),
                }
                for name in live._EVIDENCE_NAMES
            },
            "campaign_lock_ownership": {
                "schema_version": 1,
                "kind": "m6_campaign_lock_ownership_v1",
                "status": "pass",
                "source_commit": COMMIT,
                "gpu_uuid_sha256": H,
                "output_root": output_root,
                "operator_authority_sha256": _operator(
                    output_root=output_root, maximum=maximum
                )["authority_sha256"],
                "lock_path": f"{output_root}/.m6-live-campaign.lock",
                "owner_pid": 123,
                "owner_start_ticks": 456,
                "acquired_at_utc": NOW,
                "expires_at_utc": "2026-08-08T12:10:00Z",
                "storage_fault_receipt_path": (
                    f"{output_root}/.m6-live-storage-fault.json"
                ),
                "receipt_sha256": H2,
            },
            "storage_fault_absent_at_utc": NOW,
        },
        "schedule": {
            "path": "/tmp/schedule.json",
            "sha256": "7" * 64,
            "size_bytes": 1,
            "schedule_sha256": H,
            "source_identity_sha256": "8" * 64,
            "config_identity_sha256": "9" * 64,
            "metric_identity_sha256": H2,
            "m4_identity_sha256": "e" * 64,
        },
        "inputs": {
            "matrix": {**file_ref, "path": "matrix.json", "sha256": H2},
            "metric_registry": {**file_ref, "path": "registry.json"},
            "sidecar_schema": {**file_ref, "path": "sidecar.json"},
            "campaign_protocol_pins": {**file_ref, "path": "pins.json"},
        },
        "schema_authorities": [
            {"name": name, "path": f"{name}.schema.json", "sha256": H, "size_bytes": 1}
            for name in sorted(live.SCHEMA_PATHS)
        ],
        "readiness_preflights": preflights,
        "gpu_uuid_sha256": H,
        "driver_version": live.GENERIC_DRIVER_VERSION,
        "ncu_identity": {
            "resolved_path": "/opt/ncu",
            "executable_sha256": H,
            "version_line": "Version 2026.1.0",
        },
        "tools": {
            **{
                name: {
                    "path": f"tools/local_labels/{name}.py",
                    "sha256": H,
                    "size_bytes": 1,
                }
                for name in (
                    "collector",
                    "campaign_preflight",
                    "immutable_artifact",
                    "ncu_action_executor",
                    "ncu_collection_receipt",
                )
            },
            "label_driver": {
                "path": "tools/local_labels/label_driver.sh",
                "sha256": H,
                "size_bytes": 1,
                "mode": 493,
            },
        },
        "namespace": {
            "contract": live.SESSION_NAMESPACE,
            "output_root": output_root,
            "session_root": root,
            "attempts_root": f"{root}/attempts",
            "receipts_root": f"{root}/receipts",
            "tier2_root": f"{root}/tier2",
        },
        "cost_ceiling": {
            "unit": "fresh_profile_process",
            "max_fresh_processes_per_resume": maximum,
            "source": "schedule.cost_gate.max_new_profiles_per_resume",
        },
        "operator_hardware_authority": _operator(
            output_root=output_root, maximum=maximum
        ),
        "scope": live.SCOPE,
    }
    value = _sealed(unsigned, "session_authority_sha256")
    live.validate_schema("session", value)
    return value


def _ledger_schedule() -> dict:
    slots = [
        {
            "ordinal": ordinal,
            "slot_id": f"m6-slot-{ordinal:016d}",
            "slot_sha256": f"{ordinal:064x}",
        }
        for ordinal in range(8540)
    ]
    return {
        "schedule_sha256": H,
        "cost_gate": {"max_new_profiles_per_resume": 3},
        "slots": slots,
        "tier2_candidates": [],
    }


def _plan_for(
    slot: dict,
    *,
    sequence: int,
    slot_attempt_index: int,
    session_sha256: str | None = None,
    resume_epoch: int = 0,
    attempt_root: Path = Path("/tmp/attempt"),
) -> dict:
    unit = [f"metric.unit.{index}" for index in range(53)]
    flat = ["metric.flat.0", "metric.flat.1"]
    unsigned = {
        "schema_version": 1,
        "kind": "m6_live_slot_plan_v1",
        "planned_at_utc": NOW,
        "expires_at_utc": "2026-08-08T12:15:00Z",
        "session_authority_sha256": (
            session_sha256 or _session()["session_authority_sha256"]
        ),
        "schedule_sha256": H,
        "tier2_activation_sha256": None,
        "resume_epoch": resume_epoch,
        "attempt_sequence": sequence,
        "slot_attempt_index": slot_attempt_index,
        "ordinal": slot["ordinal"],
        "slot_id": slot["slot_id"],
        "slot_sha256": slot["slot_sha256"],
        "tier": 1,
        "cell_id": "src:cell",
        "cell_identity_sha256": H,
        "config_id": "cfg=clock_none__cache_all",
        "config_sha256": H,
        "sweep_index": 0,
        "repeat_index": 0,
        "fresh_process": True,
        "launcher": {
            "executable": "/tmp/driver",
            "executable_sha256": H,
            "mode": 493,
        },
        "target": {
            "executable": "/tmp/target",
            "executable_sha256": H,
            "argv": ["/tmp/target"],
            "working_directory": "/tmp",
            "roster_source": "portable_lcd",
            "slug": "cell",
        },
        "shape": {
            "shape_point": "shape",
            "shape_binding": "application-args",
            "logical_workload_sha256": H,
            "application_argv": [],
        },
        "launch_selector": {
            "launch_index": 0,
            "kernel_name": "kernel",
            "kernel_signature_sha256": H,
            "m4_launch_identity_sha256": H,
        },
        "ncu_controls": {
            "target_processes": "all",
            "filter_mode": "global",
            "replay_mode": "kernel",
            "cache_control": "all",
            "clock_control": "none",
            "profile_from_start": False,
            "launch_skip": 0,
            "launch_count": 1,
        },
        "metric_request": {
            "selected_count": 55,
            "metric_identity_sha256": H2,
            "requested_ncu_metrics": flat + unit,
            "implicit_flat_metrics": flat,
            "explicit_unit_metrics": unit,
        },
        "output": {
            "attempt_root": str(attempt_root),
            "evidence_root": str(attempt_root / "evidence"),
            "sidecar_path": str(attempt_root / "evidence/label.json"),
            "attempt_receipt_path": f"/tmp/attempt-{sequence}.json",
        },
        "command": ["/tmp/driver", "--"],
    }
    return _sealed(unsigned, "plan_sha256")


@pytest.fixture(scope="module")
def incomplete_schedule() -> dict:
    return m6.build_schedule(seed=17, generated_at_utc=NOW)


def test_new_schemas_are_hash_pinned_valid_and_closed() -> None:
    assert set(live.SCHEMA_PATHS) == {"session", "plan", "attempt", "ledger", "tier2"}
    for kind, path in live.SCHEMA_PATHS.items():
        payload = path.read_bytes()
        assert immutable.sha256_bytes(payload) == live.SCHEMA_SHA256[kind]
        schema = live.load_schema(kind)
        Draft202012Validator.check_schema(schema)
        _closed(schema)


def test_collector_has_no_hardware_query_or_spawn_surface() -> None:
    source = inspect.getsource(live)
    assert "import subprocess" not in source
    assert "subprocess." not in source
    assert "probe_device(" not in source
    assert "execute_ncu_action(" not in source
    assert "execute_ncu_supervised_executable(" not in source
    assert not hasattr(live, "execute")
    assert "M6_COLLECT_DISABLED_OFFLINE_SOURCE" in inspect.getsource(m6._cmd_collect)


def test_complete_m4_is_required_before_live_admission(
    incomplete_schedule: dict, tmp_path: Path
) -> None:
    path = tmp_path / "schedule.json"
    path.write_bytes(immutable.pretty_json_bytes(incomplete_schedule))
    with pytest.raises(live.M6LiveError) as error:
        live.validate_campaign_inputs(incomplete_schedule, schedule_path=path)
    assert error.value.code == "M6_LIVE_M4_INCOMPLETE"


def _minimal_preflight(
    *, output_root: Path, generated: str = NOW, driver: str = "610.43.02"
) -> dict:
    checks = {
        name: {"status": "ok"} for name in live.campaign_preflight._BLOCKING_CHECKS
    }
    checks["driver"] = {"status": "ok", "expected": driver, "observed": driver}
    checks["device"] = {
        "status": "ok",
        "selection_policy": live.campaign_preflight.DEVICE_SELECTION_POLICY,
        "selected_host_gpu_index": 0,
        "nvidia_smi_device_count": 1,
        "gpu_uuid_sha256": H,
    }
    checks["trace_budget"] = {"status": "ok", "output_root": str(output_root)}
    checks["ncu"] = {
        "status": "ok",
        "resolved_path": "/opt/ncu",
        "executable_sha256": H,
        "version_line": "Version 2026.1.0",
    }
    return {
        "schema_version": 4,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "captured_at_utc": generated,
        "generated_at_utc": generated,
        "git_head": COMMIT,
        "working_tree_clean": True,
        "pins_sha256": H,
        "capture_matrix_sha256": H2,
        "protocol_version": "v2.1",
        "checks": checks,
        "blockers": [],
        "capture_admitted": True,
    }


def _permissive_preflight_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
    }


def test_pure_preflight_projection_binds_fresh_gpu_and_ncu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        live, "_load_preflight_schema", lambda _root: _permissive_preflight_schema()
    )
    root = tmp_path / "external"
    path = tmp_path / "preflight.json"
    document = _minimal_preflight(output_root=root)
    path.write_bytes(immutable.pretty_json_bytes(document))
    projection, detail = live._preflight_projection(
        "p3",
        path,
        source_commit=COMMIT,
        pins_identity={"sha256": H},
        matrix_identity={"sha256": H2},
        output_root=root,
        now=live._parse_time(NOW, role="test"),
        repo_root=live.REPO_ROOT,
    )
    assert projection["gpu_uuid_sha256"] == H
    assert projection["driver_version"] == "610.43.02"
    assert detail["ncu_identity"]["executable_sha256"] == H


@pytest.mark.parametrize(
    ("driver", "code"),
    [
        ("610.43.03", "M6_LIVE_TASK17_DRIVER_FORBIDDEN"),
        ("609.0", "M6_LIVE_PREFLIGHT_BINDING"),
    ],
)
def test_preflight_rejects_non_generic_driver_before_planning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    driver: str,
    code: str,
) -> None:
    monkeypatch.setattr(
        live, "_load_preflight_schema", lambda _root: _permissive_preflight_schema()
    )
    root = tmp_path / "external"
    path = tmp_path / "preflight.json"
    path.write_bytes(
        immutable.pretty_json_bytes(
            _minimal_preflight(output_root=root, driver=driver)
        )
    )
    with pytest.raises(live.M6LiveError) as error:
        live._preflight_projection(
            "p4",
            path,
            source_commit=COMMIT,
            pins_identity={"sha256": H},
            matrix_identity={"sha256": H2},
            output_root=root,
            now=live._parse_time(NOW, role="test"),
            repo_root=live.REPO_ROOT,
        )
    assert error.value.code == code


def test_preflight_freshness_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        live, "_load_preflight_schema", lambda _root: _permissive_preflight_schema()
    )
    root = tmp_path / "external"
    path = tmp_path / "preflight.json"
    path.write_bytes(
        immutable.pretty_json_bytes(
            _minimal_preflight(output_root=root, generated="2026-08-08T11:44:59Z")
        )
    )
    with pytest.raises(live.M6LiveError) as error:
        live._preflight_projection(
            "session",
            path,
            source_commit=COMMIT,
            pins_identity={"sha256": H},
            matrix_identity={"sha256": H2},
            output_root=root,
            now=live._parse_time(NOW, role="test"),
            repo_root=live.REPO_ROOT,
        )
    assert error.value.code == "M6_LIVE_PREFLIGHT_EXPIRED"


def _write_json(path: Path, document: dict) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = immutable.pretty_json_bytes(document)
    path.write_bytes(payload)
    return path, immutable.sha256_bytes(payload)


def _admission_fixture(tmp_path: Path) -> dict:
    output_root = tmp_path / "external-output"
    output_root.mkdir(parents=True)
    source_tree = "2" * 40
    p1 = _sealed(
        {
            "schema_version": 1,
            "kind": "m6_p1_gate_receipt_v1",
            "status": "pass",
            "source_commit": COMMIT,
            "source_tree": source_tree,
            "default_verification": "pass",
            "capture_focused_battery": "pass",
            "hardware_probe_deselected": (
                "tools/local_labels/test_campaign_preflight.py::"
                "test_environment_probes_ok_on_reference_host"
            ),
            "identity_closure": "pass",
        },
        "receipt_sha256",
    )
    current = _sealed(
        {
            "schema_version": 1,
            "kind": "m6_current_source_external_anchor_v1",
            "status": "pass",
            "source_commit": COMMIT,
            "source_tree": source_tree,
            "working_tree_clean": True,
            "p1_gate_receipt_sha256": p1["receipt_sha256"],
        },
        "anchor_sha256",
    )
    live_authority = {
        "schema_version": 1,
        "roster_id": "sm86_capture_roster_authority_v1",
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "source_commit": COMMIT,
        "accounting_identity_holds": True,
        "counts": {"universe": 344, "runnable": 122, "excluded": 222},
        "cells": [
            {
                "fixture_index": index,
                "resolution_status": "runnable" if index < 122 else "excluded",
            }
            for index in range(344)
        ],
    }
    live_path, live_sha = _write_json(tmp_path / "live-v1.json", live_authority)
    successor = {"authority": {"sha256": live_sha}, "cells": []}
    successor_path, successor_sha = _write_json(
        tmp_path / "successor.json", successor
    )
    clean = _sealed(
        {
            "schema_version": 1,
            "kind": "m6_clean_s_p2_receipt_v1",
            "status": "pass",
            "source_commit": COMMIT,
            "source_tree": source_tree,
            "working_tree_clean": True,
            "p1_source_anchor_sha256": current["anchor_sha256"],
            "producer_start_head": COMMIT,
            "authority_head": COMMIT,
            "producer_final_head": COMMIT,
            "live_v1_authority_sha256": live_sha,
            "successor_matrix_sha256": successor_sha,
            "verified_runnable_cells": 122,
        },
        "receipt_sha256",
    )
    operator = _operator(output_root=str(output_root), maximum=3)
    lock = _sealed(
        {
            "schema_version": 1,
            "kind": "m6_campaign_lock_ownership_v1",
            "status": "pass",
            "source_commit": COMMIT,
            "gpu_uuid_sha256": H,
            "output_root": str(output_root),
            "operator_authority_sha256": operator["authority_sha256"],
            "lock_path": str(output_root / ".m6-live-campaign.lock"),
            "owner_pid": 123,
            "owner_start_ticks": 456,
            "acquired_at_utc": NOW,
            "expires_at_utc": "2026-08-08T12:10:00Z",
            "storage_fault_receipt_path": str(
                output_root / ".m6-live-storage-fault.json"
            ),
        },
        "receipt_sha256",
    )
    documents = {
        "p1_gate_receipt": p1,
        "current_source_anchor": current,
        "clean_s_p2_receipt": clean,
        "live_v1_authority": live_authority,
        "successor_matrix": successor,
        "campaign_lock_ownership": lock,
        "operator_hardware_authority": operator,
    }
    paths: dict[str, Path] = {
        "live_v1_authority": live_path,
        "successor_matrix": successor_path,
    }
    hashes = {
        "live_v1_authority": live_sha,
        "successor_matrix": successor_sha,
    }
    for name, document in documents.items():
        if name in paths:
            continue
        path, digest = _write_json(tmp_path / f"{name}.json", document)
        paths[name] = path
        hashes[name] = digest
    anchor = _sealed(
        {
            "schema_version": 1,
            "kind": "m6_external_operator_trust_anchor_v1",
            "trust_mode": live.FIXTURE_TRUST_MODE,
            "trust_anchor_id": "fixture-only-no-production-credit",
            "status": "trusted",
            "source_commit": COMMIT,
            "source_tree": source_tree,
            "valid_from_utc": "2026-08-08T11:59:00Z",
            "valid_until_utc": "2026-08-08T12:10:00Z",
            "evidence_sha256": hashes,
        },
        "anchor_sha256",
    )
    anchor_path, anchor_file_sha = _write_json(tmp_path / "trust.json", anchor)
    return {
        "evidence_paths": paths,
        "trust_anchor_path": anchor_path,
        "trusted_anchor_sha256": anchor_file_sha,
        "trust_mode": live.FIXTURE_TRUST_MODE,
        "context": {
            "matrix": successor,
            "schedule_document": {
                "cost_gate": {"max_new_profiles_per_resume": 3}
            },
        },
        "output_root": output_root,
        "now": live._parse_time(NOW, role="fixture now"),
        "repo_root": live.REPO_ROOT,
    }


def test_public_admission_has_no_caller_clock_or_hex_authority(tmp_path: Path) -> None:
    for function in (
        live.build_session_authority,
        live.validate_session_authority,
        live.plan_next_slot,
    ):
        parameters = inspect.signature(function).parameters
        assert "admitted_at_utc" not in parameters
        assert "issued_at_utc" not in parameters
        assert "source_commit" not in parameters
        assert "operator_hardware_authority" not in parameters
    with pytest.raises(live.M6LiveError) as error:
        live.build_session_authority(
            schedule={},
            schedule_path=tmp_path / "schedule.json",
            admission_evidence_paths={},
            p3_preflight_path=tmp_path / "p3.json",
            p4_preflight_path=tmp_path / "p4.json",
            session_preflight_path=tmp_path / "session.json",
            output_root=tmp_path / "output",
        )
    assert error.value.code == "M6_LIVE_OPERATOR_TRUST_ANCHOR_UNAVAILABLE"


def test_detached_prerequisites_pass_only_in_explicit_fixture_mode(
    tmp_path: Path,
) -> None:
    kwargs = _admission_fixture(tmp_path)
    result = live._validate_admission_evidence(**kwargs)
    assert result["trust_mode"] == live.FIXTURE_TRUST_MODE
    assert result["source_commit"] == COMMIT


def test_detached_anchor_rejects_changed_p1_and_storage_fault(tmp_path: Path) -> None:
    kwargs = _admission_fixture(tmp_path)
    p1_path = kwargs["evidence_paths"]["p1_gate_receipt"]
    p1_path.write_bytes(p1_path.read_bytes() + b" ")
    with pytest.raises(live.M6LiveError) as error:
        live._validate_admission_evidence(**kwargs)
    assert error.value.code == "M6_LIVE_OPERATOR_TRUST_ANCHOR_MISMATCH"

    clean_kwargs = _admission_fixture(tmp_path / "storage-case")
    fault = clean_kwargs["output_root"] / ".m6-live-storage-fault.json"
    fault.write_text("fault", encoding="utf-8")
    with pytest.raises(live.M6LiveError) as error:
        live._validate_admission_evidence(**clean_kwargs)
    assert error.value.code == "M6_LIVE_STORAGE_FAULT_PRESENT"


def test_private_fixture_clock_rejects_stale_session_at_boundary() -> None:
    session = _session()
    schedule = {"cost_gate": {"max_new_profiles_per_resume": 3}}
    context = {
        "schedule": session["schedule"],
        "inputs": session["inputs"],
        "schema_authorities": session["schema_authorities"],
    }
    with pytest.raises(live.M6LiveError) as error:
        live._validate_session_bindings(
            session,
            schedule,
            context,
            now=live._parse_time("2026-08-08T12:20:00Z", role="fixture now"),
        )
    assert error.value.code == "M6_LIVE_PREFLIGHT_EXPIRED"


def _planner_case(
    tmp_path: Path, *, clock: str, cache: str, source: str
) -> tuple[dict, dict, dict, dict, dict]:
    executable = tmp_path / "target"
    executable.write_bytes(b"target")
    executable.chmod(0o755)
    binary_hash = immutable.sha256_bytes(b"target")
    driver = tmp_path / "driver"
    driver.write_bytes(b"driver")
    driver.chmod(0o755)
    driver_hash = immutable.sha256_bytes(b"driver")
    slug = f"slug_{source}"
    shape = "shape_0"
    cell_id = f"{source}:{slug}"
    cell = {
        "cell_id": cell_id,
        "roster_source": source,
        "slug": slug,
        "shape_point": shape,
        "authority_cell_sha256": "3" * 64,
        "binary_sha256": binary_hash,
        "cell_identity_sha256": "4" * 64,
    }
    config_id = f"cfg=clock_{clock}__cache_{cache}"
    config = {
        "config_id": config_id,
        "config_sha256": "5" * 64,
        "clock_control": clock,
        "cache_control": cache,
        "replay_mode": "kernel",
        "target_processes": "all",
        "filter_mode": "global",
        "profile_from_start": False,
        "fresh_process_per_profile": True,
    }
    slot = {
        "ordinal": 17,
        "slot_id": "m6-t1-" + "6" * 24,
        "slot_sha256": "7" * 64,
        "tier": 1,
        "cell_id": cell_id,
        "cell_identity_sha256": cell["cell_identity_sha256"],
        "config_id": config_id,
        "config_sha256": config["config_sha256"],
        "sweep_index": 3,
        "repeat_index": 3,
        "process_isolation": "fresh_process",
    }
    logical = {"kind": "shape_point", "value": shape}
    invocation = {
        "executable_path": executable.relative_to(tmp_path).as_posix(),
        "executable_sha256": binary_hash,
        "label_app_args": ["--shape", shape],
        "logical_workload": logical,
        "capture_launch_contract": {
            "label_launch_index": 2,
            "label_driver_warmup_count": 1,
        },
        "adapter_kind": "sol_execbench" if source == "sol_execbench" else "direct_binary",
        "shape_delivery": (
            "adapter_consumes_logical_shape" if source == "sol_execbench" else "application_args"
        ),
    }
    matrix_cell = {
        "identity": {"roster_source": source, "slug": slug},
        "disposition": "runnable",
        "authority_cell_sha256": cell["authority_cell_sha256"],
        "invocation": invocation,
    }
    launch = {
        "launch_index": 2,
        "kernel_name": "_Zexact_kernel",
        "kernel_signature_sha256": "8" * 64,
        "launch_identity_sha256": "9" * 64,
    }
    schedule = {
        "schedule_sha256": H,
        "metric_identity_sha256": H2,
        "slots": [slot],
        "tier2_candidates": [],
        "cells": [cell],
        "configs": [config],
        "m4_identity": {
            "cell_launch_identities": [
                {
                    "cell_id": cell_id,
                    "cell_identity_sha256": cell["cell_identity_sha256"],
                    "binary_sha256": binary_hash,
                    "launches": [launch],
                }
            ]
        },
    }
    flat = ["metric.flat.0", "metric.flat.1"]
    unit = [f"metric.unit.{index}" for index in range(53)]
    context = {
        "matrix": {"cells": [matrix_cell]},
        "implicit_flat_metrics": flat,
        "explicit_unit_metrics": unit,
        "requested_ncu_metrics": flat + unit,
    }
    session = {
        "session_authority_sha256": "a" * 64,
        "namespace": {
            "attempts_root": str(tmp_path / "out/attempts"),
            "receipts_root": str(tmp_path / "out/receipts"),
        },
        "ncu_identity": {
            "resolved_path": "/opt/ncu",
            "executable_sha256": "b" * 64,
            "version_line": "Version 2026.1.0",
        },
        "tools": {
            "label_driver": {
                "path": "driver",
                "sha256": driver_hash,
                "mode": 493,
            }
        },
    }
    ledger = {
        "resume_epoch": 0,
        "state": {
            "resume_epoch_fresh_processes_remaining": 1,
            "tier1_complete": False,
            "study_complete": False,
            "next_ordinal": 17,
        }
    }
    return session, schedule, context, ledger, slot


@pytest.mark.parametrize(
    ("clock", "cache"),
    [(clock, cache) for clock in ("none", "base", "boost") for cache in ("all", "none")],
)
def test_pure_planner_maps_all_six_configs_and_exact_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clock: str,
    cache: str,
) -> None:
    session, schedule, context, ledger, slot = _planner_case(
        tmp_path, clock=clock, cache=cache, source="portable_lcd"
    )
    monkeypatch.setattr(live, "validate_session_authority", lambda *args, **kwargs: context)
    monkeypatch.setattr(live, "validate_ledger_receipt", lambda *args, **kwargs: None)
    plan = live.plan_next_slot(
        session=session,
        schedule=schedule,
        schedule_path=tmp_path / "unused.json",
        preflight_paths=(),
        ledger=ledger,
        attempts=(),
        repo_root=tmp_path,
    )
    assert plan["ordinal"] == slot["ordinal"] == 17
    assert plan["fresh_process"] is True
    assert plan["ncu_controls"]["clock_control"] == clock
    assert plan["ncu_controls"]["cache_control"] == cache
    assert plan["ncu_controls"]["profile_from_start"] is False
    assert plan["metric_request"]["selected_count"] == 55
    assert len(plan["metric_request"]["requested_ncu_metrics"]) == 55
    assert plan["command"].count("--unit-metric") == 53
    assert "--ncu-exec-path" not in plan["command"]
    assert plan["command"][plan["command"].index("--kernel-name") + 1] == "_Zexact_kernel"
    assert plan["launch_selector"]["launch_index"] == 2


@pytest.mark.parametrize(
    ("source", "binding"),
    [
        ("portable_lcd", "application-args"),
        ("simbench_bw", "application-args"),
        ("sol_execbench", "logical-workload"),
    ],
)
def test_pure_planner_is_generic_across_full_cohort_adapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
    binding: str,
) -> None:
    session, schedule, context, ledger, _slot = _planner_case(
        tmp_path, clock="none", cache="all", source=source
    )
    monkeypatch.setattr(live, "validate_session_authority", lambda *args, **kwargs: context)
    monkeypatch.setattr(live, "validate_ledger_receipt", lambda *args, **kwargs: None)
    plan = live.plan_next_slot(
        session=session,
        schedule=schedule,
        schedule_path=tmp_path / "unused.json",
        preflight_paths=(),
        ledger=ledger,
        attempts=(),
        repo_root=tmp_path,
    )
    assert plan["target"]["roster_source"] == source
    assert plan["shape"]["shape_binding"] == binding
    assert plan["target"]["argv"][1:] == ["--shape", "shape_0"]


def test_planner_refuses_cost_exhaustion_before_target_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session, schedule, context, ledger, _slot = _planner_case(
        tmp_path, clock="none", cache="all", source="portable_lcd"
    )
    ledger["state"]["resume_epoch_fresh_processes_remaining"] = 0
    monkeypatch.setattr(live, "validate_session_authority", lambda *args, **kwargs: context)
    monkeypatch.setattr(live, "validate_ledger_receipt", lambda *args, **kwargs: None)
    with pytest.raises(live.M6LiveError) as error:
        live.plan_next_slot(
            session=session,
            schedule=schedule,
            schedule_path=tmp_path / "unused.json",
            preflight_paths=(),
                ledger=ledger,
            attempts=(),
            repo_root=tmp_path,
        )
    assert error.value.code == "M6_LIVE_COST_CEILING"


def test_planner_never_accepts_caller_tier2_without_activation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session, schedule, context, ledger, _slot = _planner_case(
        tmp_path, clock="none", cache="all", source="portable_lcd"
    )
    ledger["state"]["tier1_complete"] = True
    monkeypatch.setattr(live, "validate_session_authority", lambda *args, **kwargs: context)
    monkeypatch.setattr(live, "validate_ledger_receipt", lambda *args, **kwargs: None)
    with pytest.raises(live.M6LiveError) as error:
        live.plan_next_slot(
            session=session,
            schedule=schedule,
            schedule_path=tmp_path / "unused.json",
            preflight_paths=(),
                ledger=ledger,
            attempts=(),
            repo_root=tmp_path,
        )
    assert error.value.code == "M6_LIVE_TIER2_ACTIVATION_REQUIRED"


def test_planner_rejects_self_sealed_activation_without_tier1_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session()
    schedule = _ledger_schedule()
    ledger = live.build_initial_ledger_receipt(session, schedule)
    activation = _sealed(
        {"caller_selected_cell_ids": ["src:cell000"]}, "activation_sha256"
    )
    monkeypatch.setattr(live, "validate_session_authority", lambda *args, **kwargs: {})
    with pytest.raises(live.M6LiveError) as error:
        live.plan_next_slot(
            session=session,
            schedule=schedule,
            schedule_path=tmp_path / "unused.json",
            preflight_paths=(),
                ledger=ledger,
            attempts=(),
            activation=activation,
            repo_root=tmp_path,
        )
    assert error.value.code == "M6_LIVE_TIER2_CALLER_SUPPLIED"


def test_attempt_ledger_is_causal_retryable_and_distinguishes_failure_phase() -> None:
    session = _session()
    schedule = _ledger_schedule()
    ledger0 = live.build_initial_ledger_receipt(session, schedule)
    slot = schedule["slots"][0]
    plan0 = _plan_for(
        slot,
        sequence=0,
        slot_attempt_index=0,
        session_sha256=session["session_authority_sha256"],
    )
    pre = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger0,
        prior_attempts=(),
        plan=plan0,
        activation=None,
        outcome="pre_start_failure",
        finished_at_utc=NOW,
        failure={"code": "PREFLIGHT_RACE", "detail": "closed before start"},
    )
    ledger1 = live.append_attempt_to_ledger(
        previous_ledger=ledger0,
        session=session,
        schedule=schedule,
        prior_attempts=(),
        attempt=pre,
        plan=plan0,
        activation=None,
    )
    assert ledger1["state"]["next_ordinal"] == 0
    assert ledger1["state"]["pre_start_failures"] == 1

    plan1 = _plan_for(
        slot,
        sequence=1,
        slot_attempt_index=1,
        session_sha256=session["session_authority_sha256"],
    )
    process1 = live._build_fixture_process_identity(
        plan=plan1, pid=101, start_ticks=1001, started_at_utc=NOW
    )
    post = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger1,
        prior_attempts=(pre,),
        prior_plans=(plan0,),
        ledger_history=(ledger0,),
        plan=plan1,
        activation=None,
        outcome="post_start_failure",
        started_at_utc=NOW,
        process_identity=process1,
        finished_at_utc=NOW,
        failure={"code": "CHILD_FAILED", "detail": "process started"},
    )
    ledger2 = live.append_attempt_to_ledger(
        previous_ledger=ledger1,
        session=session,
        schedule=schedule,
        prior_attempts=(pre,),
        prior_plans=(plan0,),
        ledger_history=(ledger0,),
        attempt=post,
        plan=plan1,
        activation=None,
    )
    assert ledger2["state"]["next_ordinal"] == 0
    assert ledger2["state"]["post_start_failures"] == 1
    assert ledger2["state"]["lifetime_fresh_processes_started"] == 1

    plan2 = _plan_for(
        slot,
        sequence=2,
        slot_attempt_index=2,
        session_sha256=session["session_authority_sha256"],
    )
    terminal = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger2,
        prior_attempts=(pre, post),
        prior_plans=(plan0, plan1),
        ledger_history=(ledger0, ledger1),
        plan=plan2,
        activation=None,
        outcome="excluded_pre_start",
        finished_at_utc=NOW,
        failure={"code": "TYPED_EXCLUSION", "detail": "fixture terminal"},
        terminal_event_sha256=H2,
    )
    ledger3 = live.append_attempt_to_ledger(
        previous_ledger=ledger2,
        session=session,
        schedule=schedule,
        prior_attempts=(pre, post),
        prior_plans=(plan0, plan1),
        ledger_history=(ledger0, ledger1),
        attempt=terminal,
        plan=plan2,
        activation=None,
    )
    assert ledger3["state"]["terminal_prefix_slots"] == 1
    assert ledger3["state"]["next_ordinal"] == 1


def test_attempt_ledger_rejects_resealed_nonprefix_attempt() -> None:
    session = _session()
    schedule = _ledger_schedule()
    ledger = live.build_initial_ledger_receipt(session, schedule)
    plan = _plan_for(schedule["slots"][0], sequence=0, slot_attempt_index=0)
    attempt = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger,
        prior_attempts=(),
        plan=plan,
        activation=None,
        outcome="pre_start_failure",
        finished_at_utc=NOW,
        failure={"code": "PRE_START", "detail": "fixture"},
    )
    bad = dict(attempt)
    bad.pop("receipt_sha256")
    bad["ordinal"] = 1
    bad = _sealed(bad, "receipt_sha256")
    with pytest.raises(live.M6LiveError) as error:
        live.append_attempt_to_ledger(
            previous_ledger=ledger,
            session=session,
            schedule=schedule,
            prior_attempts=(),
            attempt=bad,
            plan=plan,
            activation=None,
        )
    assert error.value.code == "M6_LIVE_PLAN_BINDING"


def test_append_rejects_resealed_forged_attempt_id() -> None:
    session = _session()
    schedule = _ledger_schedule()
    ledger = live.build_initial_ledger_receipt(session, schedule)
    plan = _plan_for(
        schedule["slots"][0],
        sequence=0,
        slot_attempt_index=0,
        session_sha256=session["session_authority_sha256"],
    )
    attempt = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger,
        prior_attempts=(),
        plan=plan,
        activation=None,
        outcome="pre_start_failure",
        finished_at_utc=NOW,
        failure={"code": "PRE_START", "detail": "fixture"},
    )
    forged = dict(attempt)
    forged.pop("receipt_sha256")
    forged["attempt_id"] = "m6-attempt-" + "f" * 24
    forged = _sealed(forged, "receipt_sha256")
    with pytest.raises(live.M6LiveError) as error:
        live.append_attempt_to_ledger(
            previous_ledger=ledger,
            session=session,
            schedule=schedule,
            prior_attempts=(),
            attempt=forged,
            plan=plan,
            activation=None,
        )
    assert error.value.code == "M6_LIVE_ATTEMPT_ID"


def test_process_ceiling_is_per_causal_resume_epoch() -> None:
    session = _session(maximum=1)
    schedule = _ledger_schedule()
    schedule["cost_gate"]["max_new_profiles_per_resume"] = 1
    ledger0 = live.build_initial_ledger_receipt(session, schedule)
    slot = schedule["slots"][0]
    plan0 = _plan_for(
        slot,
        sequence=0,
        slot_attempt_index=0,
        session_sha256=session["session_authority_sha256"],
    )
    process0 = live._build_fixture_process_identity(
        plan=plan0, pid=201, start_ticks=2001, started_at_utc=NOW
    )
    first = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger0,
        prior_attempts=(),
        plan=plan0,
        activation=None,
        outcome="post_start_failure",
        started_at_utc=NOW,
        process_identity=process0,
        finished_at_utc=NOW,
        failure={"code": "CHILD_FAILED", "detail": "epoch zero"},
    )
    ledger1 = live.append_attempt_to_ledger(
        previous_ledger=ledger0,
        session=session,
        schedule=schedule,
        prior_attempts=(),
        attempt=first,
        plan=plan0,
        activation=None,
    )
    assert ledger1["state"]["resume_epoch_fresh_processes_remaining"] == 0
    ledger2 = live.begin_resume_epoch(
        previous_ledger=ledger1,
        session=session,
        schedule=schedule,
        attempts=(first,),
        plans=(plan0,),
        ledger_history=(ledger0,),
    )
    assert ledger2["resume_epoch"] == 1
    assert ledger2["state"]["resume_epoch_fresh_processes_remaining"] == 1
    assert ledger2["state"]["lifetime_fresh_processes_started"] == 1

    plan1 = _plan_for(
        slot,
        sequence=1,
        slot_attempt_index=1,
        session_sha256=session["session_authority_sha256"],
        resume_epoch=1,
    )
    process1 = live._build_fixture_process_identity(
        plan=plan1, pid=202, start_ticks=2002, started_at_utc=NOW
    )
    second = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger2,
        prior_attempts=(first,),
        prior_plans=(plan0,),
        ledger_history=(ledger0, ledger1),
        plan=plan1,
        activation=None,
        outcome="post_start_failure",
        started_at_utc=NOW,
        process_identity=process1,
        finished_at_utc=NOW,
        failure={"code": "CHILD_FAILED", "detail": "epoch one"},
    )
    ledger3 = live.append_attempt_to_ledger(
        previous_ledger=ledger2,
        session=session,
        schedule=schedule,
        prior_attempts=(first,),
        prior_plans=(plan0,),
        ledger_history=(ledger0, ledger1),
        attempt=second,
        plan=plan1,
        activation=None,
    )
    assert ledger3["state"]["lifetime_fresh_processes_started"] == 2
    assert ledger3["state"]["resume_epoch_fresh_processes_started"] == 1


def test_resume_epoch_cannot_advance_before_budget_is_exhausted() -> None:
    session = _session(maximum=1)
    schedule = _ledger_schedule()
    schedule["cost_gate"]["max_new_profiles_per_resume"] = 1
    ledger = live.build_initial_ledger_receipt(session, schedule)
    with pytest.raises(live.M6LiveError) as error:
        live.begin_resume_epoch(
            previous_ledger=ledger,
            session=session,
            schedule=schedule,
            attempts=(),
            plans=(),
            ledger_history=(),
        )
    assert error.value.code == "M6_LIVE_RESUME_EPOCH_EARLY"


def test_append_rejects_reused_process_start_identity() -> None:
    session = _session(maximum=2)
    schedule = _ledger_schedule()
    schedule["cost_gate"]["max_new_profiles_per_resume"] = 2
    ledger0 = live.build_initial_ledger_receipt(session, schedule)
    slot = schedule["slots"][0]
    plan0 = _plan_for(
        slot,
        sequence=0,
        slot_attempt_index=0,
        session_sha256=session["session_authority_sha256"],
    )
    process = live._build_fixture_process_identity(
        plan=plan0, pid=401, start_ticks=4001, started_at_utc=NOW
    )
    first = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger0,
        prior_attempts=(),
        plan=plan0,
        activation=None,
        outcome="post_start_failure",
        started_at_utc=NOW,
        process_identity=process,
        finished_at_utc=NOW,
        failure={"code": "CHILD_FAILED", "detail": "first"},
    )
    ledger1 = live.append_attempt_to_ledger(
        previous_ledger=ledger0,
        session=session,
        schedule=schedule,
        prior_attempts=(),
        attempt=first,
        plan=plan0,
        activation=None,
    )
    plan1 = _plan_for(
        slot,
        sequence=1,
        slot_attempt_index=1,
        session_sha256=session["session_authority_sha256"],
    )
    reused_process = live._build_fixture_process_identity(
        plan=plan1, pid=401, start_ticks=4001, started_at_utc=NOW
    )
    second = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger1,
        prior_attempts=(first,),
        prior_plans=(plan0,),
        ledger_history=(ledger0,),
        plan=plan1,
        activation=None,
        outcome="post_start_failure",
        started_at_utc=NOW,
        process_identity=reused_process,
        finished_at_utc=NOW,
        failure={"code": "CHILD_FAILED", "detail": "second"},
    )
    with pytest.raises(live.M6LiveError) as error:
        live.append_attempt_to_ledger(
            previous_ledger=ledger1,
            session=session,
            schedule=schedule,
            prior_attempts=(first,),
            prior_plans=(plan0,),
            ledger_history=(ledger0,),
            attempt=second,
            plan=plan1,
            activation=None,
        )
    assert error.value.code == "M6_LIVE_PROCESS_REUSE"


def _observed_attempt_fixture(tmp_path: Path, *, maximum: int = 3) -> dict:
    session = _session(output_root=str(tmp_path / "output"), maximum=maximum)
    schedule = _ledger_schedule()
    schedule["cost_gate"]["max_new_profiles_per_resume"] = maximum
    ledger = live.build_initial_ledger_receipt(session, schedule)
    attempt_root = tmp_path / "attempt"
    plan = _plan_for(
        schedule["slots"][0],
        sequence=0,
        slot_attempt_index=0,
        session_sha256=session["session_authority_sha256"],
        attempt_root=attempt_root,
    )
    artifact_payloads = {
        "ncu_report": b"report",
        "label_run_manifest": b"run-manifest",
        "ncu_launch_observation": b"launch-observation",
        "label_capture_manifest": b"capture-manifest",
    }
    artifacts = []
    for role, payload in artifact_payloads.items():
        relative = f"evidence/{role}.bin"
        path = attempt_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts.append(
            live.ncu_collection_receipt.artifact_binding(
                role=role,
                relative_path=relative,
                sha256=immutable.sha256_bytes(payload),
                size_bytes=len(payload),
            )
        )
    environment = live.ncu_action_executor.NcuCollectionEnvironment(
        {
            "LANG": "C",
            "LANGUAGE": "C",
            "LC_ALL": "C",
            "PATH": "/usr/local/cuda-13.2/bin:/usr/bin:/bin",
            "PWD": plan["target"]["working_directory"],
            "PYTHONNOUSERSITE": "1",
            "LABEL_DRIVER_GPU_INDEX": "0",
            "LABEL_DRIVER_PRESERVE_FAILED_EVIDENCE": "1",
        }
    )
    identity = live.ncu_action_executor.build_expected_ncu_identity(
        **session["ncu_identity"]
    )
    collection = live.ncu_collection_receipt.build_receipt(
        preflight_sha256=session["readiness_preflights"][-1]["sha256"],
        expected_ncu_identity=identity,
        command=plan["command"],
        environment=environment,
        working_directory=Path(plan["target"]["working_directory"]),
        exit_status=0,
        artifacts=artifacts,
        launch_observation_sha256=H2,
    )
    receipt_path = attempt_root / "evidence/collection-receipt.json"
    receipt_payload = immutable.pretty_json_bytes(collection)
    receipt_path.write_bytes(receipt_payload)
    collection_ref = {
        "relative_path": "evidence/collection-receipt.json",
        "sha256": immutable.sha256_bytes(receipt_payload),
        "size_bytes": len(receipt_payload),
        "receipt_seal_sha256": collection["receipt_seal_sha256"],
    }
    process = live._build_fixture_process_identity(
        plan=plan, pid=301, start_ticks=3001, started_at_utc=NOW
    )
    attempt = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger,
        prior_attempts=(),
        plan=plan,
        activation=None,
        outcome="observed",
        started_at_utc=NOW,
        process_identity=process,
        finished_at_utc=NOW,
        artifacts=artifacts,
        collection_receipt=collection_ref,
        terminal_event_sha256=H,
    )
    return {
        "session": session,
        "schedule": schedule,
        "ledger": ledger,
        "plan": plan,
        "attempt": attempt,
        "artifact_path": attempt_root / artifacts[0]["relative_path"],
    }


def test_append_rederives_plan_attempt_process_artifacts_and_collection(
    tmp_path: Path,
) -> None:
    case = _observed_attempt_fixture(tmp_path)
    ledger = live.append_attempt_to_ledger(
        previous_ledger=case["ledger"],
        session=case["session"],
        schedule=case["schedule"],
        prior_attempts=(),
        attempt=case["attempt"],
        plan=case["plan"],
        activation=None,
    )
    assert ledger["state"]["terminal_prefix_slots"] == 1
    assert case["attempt"]["q809_production_matched_label_authority"] is False
    assert case["attempt"]["trust_scope"].endswith("not_q809_label_authority")
    assert "sidecar_sha256" not in case["attempt"]
    assert {row["role"] for row in case["attempt"]["artifacts"]} == {
        "ncu_report",
        "label_run_manifest",
        "ncu_launch_observation",
        "label_capture_manifest",
    }


def test_append_rejects_changed_evidence_bytes(tmp_path: Path) -> None:
    case = _observed_attempt_fixture(tmp_path)
    case["artifact_path"].write_bytes(b"tampered")
    with pytest.raises(live.M6LiveError) as error:
        live.append_attempt_to_ledger(
            previous_ledger=case["ledger"],
            session=case["session"],
            schedule=case["schedule"],
            prior_attempts=(),
            attempt=case["attempt"],
            plan=case["plan"],
            activation=None,
        )
    assert error.value.code == "M6_LIVE_ATTEMPT_ARTIFACT"


def _tier2_fixture() -> tuple[dict, dict, list[dict]]:
    cells = [
        {"cell_id": f"src:cell{index:03d}", "cell_identity_sha256": f"{index + 1:064x}"}
        for index in range(122)
    ]
    slots = [
        {"ordinal": index, "slot_id": f"tier1-{index:016d}"}
        for index in range(8540)
    ]
    candidates = []
    ordinal = 8540
    for cell in cells:
        for _ in range(10):
            candidates.append(
                {
                    "ordinal": ordinal,
                    "slot_id": f"tier2-{ordinal:016d}",
                    "slot_sha256": f"{ordinal:064x}",
                    "cell_id": cell["cell_id"],
                }
            )
            ordinal += 1
    schedule = {
        "schedule_sha256": H,
        "slots": slots,
        "tier2_candidates": candidates,
        "cells": cells,
    }
    journal = {
        "schema_version": 2,
        "journal_sha256": H2,
        "causal_prefix_closed": True,
        "events": [
            {"slot_id": slot["slot_id"], "tier": 1} for slot in slots
        ],
    }
    decisions = []
    for index, cell in enumerate(cells):
        decision = {
            "cell_id": cell["cell_id"],
            "cell_identity_sha256": cell["cell_identity_sha256"],
            "triggered": index == 0,
            "reason_codes": ["boundary_proximity"] if index == 0 else [],
            "evidence_metric_keys": ["metric"] if index == 0 else [],
        }
        decisions.append(_sealed(decision, "decision_sha256"))
    return schedule, journal, decisions


def test_fixture_resume_epochs_reach_derived_tier2_after_tier1_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(maximum=1)
    base_slots = _ledger_schedule()["slots"][:2]
    candidate = {
        "ordinal": 2,
        "slot_id": "m6-tier2-" + "c" * 24,
        "slot_sha256": "c" * 64,
        "cell_id": "src:cell",
    }
    schedule = {
        "schedule_sha256": H,
        "cost_gate": {"max_new_profiles_per_resume": 1},
        "slots": base_slots,
        "tier2_candidates": [candidate],
    }
    monkeypatch.setattr(live, "validate_schema", lambda *args, **kwargs: None)
    monkeypatch.setattr(live, "validate_tier2_activation", lambda *args, **kwargs: None)
    ledger0 = live.build_initial_ledger_receipt(session, schedule)

    plan0 = _plan_for(
        base_slots[0],
        sequence=0,
        slot_attempt_index=0,
        session_sha256=session["session_authority_sha256"],
    )
    process0 = live._build_fixture_process_identity(
        plan=plan0, pid=501, start_ticks=5001, started_at_utc=NOW
    )
    attempt0 = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger0,
        prior_attempts=(),
        plan=plan0,
        activation=None,
        outcome="excluded_post_start",
        started_at_utc=NOW,
        process_identity=process0,
        finished_at_utc=NOW,
        failure={"code": "EXCLUDED", "detail": "tier1 zero"},
        terminal_event_sha256=H,
    )
    ledger1 = live.append_attempt_to_ledger(
        previous_ledger=ledger0,
        session=session,
        schedule=schedule,
        prior_attempts=(),
        attempt=attempt0,
        plan=plan0,
        activation=None,
    )
    ledger2 = live.begin_resume_epoch(
        previous_ledger=ledger1,
        session=session,
        schedule=schedule,
        attempts=(attempt0,),
        plans=(plan0,),
        ledger_history=(ledger0,),
    )
    plan1 = _plan_for(
        base_slots[1],
        sequence=1,
        slot_attempt_index=0,
        session_sha256=session["session_authority_sha256"],
        resume_epoch=1,
    )
    process1 = live._build_fixture_process_identity(
        plan=plan1, pid=502, start_ticks=5002, started_at_utc=NOW
    )
    attempt1 = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger2,
        prior_attempts=(attempt0,),
        prior_plans=(plan0,),
        ledger_history=(ledger0, ledger1),
        plan=plan1,
        activation=None,
        outcome="excluded_post_start",
        started_at_utc=NOW,
        process_identity=process1,
        finished_at_utc=NOW,
        failure={"code": "EXCLUDED", "detail": "tier1 one"},
        terminal_event_sha256=H2,
    )
    ledger3 = live.append_attempt_to_ledger(
        previous_ledger=ledger2,
        session=session,
        schedule=schedule,
        prior_attempts=(attempt0,),
        prior_plans=(plan0,),
        ledger_history=(ledger0, ledger1),
        attempt=attempt1,
        plan=plan1,
        activation=None,
    )
    activation = _sealed(
        {
            "schedule_sha256": H,
            "activated_cell_ids": ["src:cell"],
            "activated_ordinals": [2],
            "activated_candidate_slots_sha256": live._digest([candidate]),
        },
        "activation_sha256",
    )
    journal = {"events": [{"event_sha256": H}, {"event_sha256": H2}]}
    ledger4 = live.activate_tier2_ledger(
        previous_ledger=ledger3,
        session=session,
        schedule=schedule,
        attempts=(attempt0, attempt1),
        plans=(plan0, plan1),
        ledger_history=(ledger0, ledger1, ledger2),
        activation=activation,
        tier1_journal=journal,
        tier1_journal_path=Path("/tmp/fixture-journal.json"),
    )
    assert ledger4["state"]["tier1_complete"] is True
    assert ledger4["state"]["next_ordinal"] == 2
    ledger5 = live.begin_resume_epoch(
        previous_ledger=ledger4,
        session=session,
        schedule=schedule,
        attempts=(attempt0, attempt1),
        plans=(plan0, plan1),
        ledger_history=(ledger0, ledger1, ledger2, ledger3),
        activation=activation,
        tier1_journal=journal,
        tier1_journal_path=Path("/tmp/fixture-journal.json"),
    )
    assert ledger5["resume_epoch"] == 2
    assert ledger5["state"]["resume_epoch_fresh_processes_remaining"] == 1
    assert ledger5["state"]["next_ordinal"] == 2


def test_tier2_activation_is_derived_only_after_complete_tier1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schedule, journal, decisions = _tier2_fixture()
    monkeypatch.setattr(live.m6, "validate_schedule", lambda *args, **kwargs: None)
    monkeypatch.setattr(live.m6, "validate_journal", lambda *args, **kwargs: None)
    monkeypatch.setattr(live.m6, "_trigger_rows_from_events", lambda *args: [])
    monkeypatch.setattr(live.m6, "_trigger_decisions", lambda *args: decisions)
    activation = live.derive_tier2_activation(
        schedule=schedule,
        tier1_journal=journal,
        journal_path=tmp_path / "journal.json",
        rehash_sidecars=False,
    )
    assert activation["counts"] == {
        "tier1_slots": 8540,
        "decisions": 122,
        "triggered_cells": 1,
        "activated_tier2_slots": 10,
    }
    assert activation["activated_cell_ids"] == ["src:cell000"]
    assert activation["activated_ordinals"] == list(range(8540, 8550))

    incomplete = copy.deepcopy(journal)
    incomplete["events"].pop()
    with pytest.raises(live.M6LiveError) as error:
        live.derive_tier2_activation(
            schedule=schedule,
            tier1_journal=incomplete,
            journal_path=tmp_path / "journal.json",
            rehash_sidecars=False,
        )
    assert error.value.code == "M6_LIVE_TIER1_INCOMPLETE"


def test_tier2_validator_rejects_caller_nominated_cells(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schedule, journal, decisions = _tier2_fixture()
    monkeypatch.setattr(live.m6, "validate_schedule", lambda *args, **kwargs: None)
    monkeypatch.setattr(live.m6, "validate_journal", lambda *args, **kwargs: None)
    monkeypatch.setattr(live.m6, "_trigger_rows_from_events", lambda *args: [])
    monkeypatch.setattr(live.m6, "_trigger_decisions", lambda *args: decisions)
    activation = live.derive_tier2_activation(
        schedule=schedule,
        tier1_journal=journal,
        journal_path=tmp_path / "journal.json",
        rehash_sidecars=False,
    )
    forged = copy.deepcopy(activation)
    forged["activated_cell_ids"] = ["src:cell001"]
    forged["activation_sha256"] = immutable.model_seal(forged, "activation_sha256")
    with pytest.raises(live.M6LiveError) as error:
        live.validate_tier2_activation(
            forged,
            schedule=schedule,
            tier1_journal=journal,
            journal_path=tmp_path / "journal.json",
            rehash_sidecars=False,
        )
    assert error.value.code == "M6_LIVE_TIER2_CALLER_SUPPLIED"


def test_ledger_history_rejects_rollback_and_publication_fork(tmp_path: Path) -> None:
    session = _session(output_root=str(tmp_path / "output"))
    schedule = _ledger_schedule()
    ledger0 = live.build_initial_ledger_receipt(session, schedule)
    plan = _plan_for(
        schedule["slots"][0],
        sequence=0,
        slot_attempt_index=0,
        session_sha256=session["session_authority_sha256"],
    )
    attempt = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger0,
        prior_attempts=(),
        plan=plan,
        activation=None,
        outcome="pre_start_failure",
        finished_at_utc=NOW,
        failure={"code": "PRE_START", "detail": "fixture"},
    )
    ledger1 = live.append_attempt_to_ledger(
        previous_ledger=ledger0,
        session=session,
        schedule=schedule,
        prior_attempts=(),
        attempt=attempt,
        plan=plan,
        activation=None,
    )
    with pytest.raises(live.M6LiveError) as error:
        live.validate_ledger_receipt(
            ledger1,
            session=session,
            schedule=schedule,
            attempts=(attempt,),
            plans=(plan,),
            ledger_history=(),
        )
    assert error.value.code == "M6_LIVE_LEDGER_ROLLBACK"

    receipts = Path(session["namespace"]["receipts_root"])
    receipts.mkdir(parents=True)
    live._publish_fixture_ledger_receipt(
        receipts / "ledger-000000.json", ledger0, session=session, schedule=schedule
    )
    fork = dict(ledger1)
    fork.pop("ledger_receipt_sha256")
    fork["previous_ledger_receipt_sha256"] = H2
    fork["ledger_history_sha256s"] = [H2]
    fork = _sealed(fork, "ledger_receipt_sha256")
    with pytest.raises(live.M6LiveError) as error:
        live._publish_fixture_ledger_receipt(
            receipts / "ledger-000001.json",
            fork,
            session=session,
            schedule=schedule,
        )
    assert error.value.code == "M6_LIVE_LEDGER_FORK"
    with pytest.raises(live.M6LiveError) as error:
        live._publish_fixture_ledger_receipt(
            tmp_path / "outside.json",
            ledger1,
            session=session,
            schedule=schedule,
        )
    assert error.value.code == "M6_LIVE_PUBLICATION_PATH"


def test_immutable_ledger_publication_is_no_replace_and_read_only(tmp_path: Path) -> None:
    session = _session(output_root=str(tmp_path / "output"))
    schedule = _ledger_schedule()
    ledger = live.build_initial_ledger_receipt(session, schedule)
    receipts_root = Path(session["namespace"]["receipts_root"])
    receipts_root.mkdir(parents=True)
    path = receipts_root / "ledger-000000.json"
    assert (
        live._publish_fixture_ledger_receipt(
            path, ledger, session=session, schedule=schedule
        )
        == path
    )
    assert stat_mode(path) == 0o444
    with pytest.raises(live.M6LiveError) as error:
        live._publish_fixture_ledger_receipt(
            path, ledger, session=session, schedule=schedule
        )
    assert error.value.code == "M6_LIVE_NO_REPLACE"


def test_public_publishers_require_external_production_trust(tmp_path: Path) -> None:
    session = _session(output_root=str(tmp_path / "output"))
    forged = copy.deepcopy(session)
    forged["trust_mode"] = live.PRODUCTION_TRUST_MODE
    forged["session_authority_sha256"] = immutable.model_seal(
        forged, "session_authority_sha256"
    )
    path = Path(forged["namespace"]["session_root"]) / "session-authority.json"
    with pytest.raises(live.M6LiveError) as error:
        live.publish_session_authority(
            path,
            forged,
            schedule={},
            schedule_path=tmp_path / "schedule.json",
            preflight_paths=(),
        )
    assert error.value.code == "M6_LIVE_OPERATOR_TRUST_ANCHOR_UNAVAILABLE"


def test_fixture_publication_rejects_bad_ledger_seal(tmp_path: Path) -> None:
    session = _session(output_root=str(tmp_path / "output"))
    schedule = _ledger_schedule()
    ledger = live.build_initial_ledger_receipt(session, schedule)
    ledger["ledger_receipt_sha256"] = H2
    path = Path(session["namespace"]["receipts_root"]) / "ledger-000000.json"
    path.parent.mkdir(parents=True)
    with pytest.raises(live.M6LiveError) as error:
        live._publish_no_replace(path, ledger, kind="ledger")
    assert error.value.code == "M6_LIVE_PUBLICATION_UNVALIDATED"
    with pytest.raises(live.M6LiveError) as error:
        live._publish_fixture_ledger_receipt(
            path, ledger, session=session, schedule=schedule
        )
    assert error.value.code == "M6_LIVE_SEAL_INVALID"


def test_plan_semantics_rederive_and_plan_publication_is_immutable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(output_root=str(tmp_path / "output"))
    schedule = _ledger_schedule()
    ledger = live.build_initial_ledger_receipt(session, schedule)
    plan = _plan_for(
        schedule["slots"][0],
        sequence=0,
        slot_attempt_index=0,
        session_sha256=session["session_authority_sha256"],
    )
    monkeypatch.setattr(live, "_trusted_now", lambda: live._parse_time(NOW, role="now"))
    monkeypatch.setattr(live, "plan_next_slot", lambda **_kwargs: plan)
    forged = copy.deepcopy(plan)
    forged["command"] = ["/tmp/forged", "--"]
    forged["plan_sha256"] = immutable.model_seal(forged, "plan_sha256")
    with pytest.raises(live.M6LiveError) as error:
        live.validate_slot_plan(
            forged,
            session=session,
            schedule=schedule,
            schedule_path=tmp_path / "schedule.json",
            preflight_paths=(),
            ledger=ledger,
            attempts=(),
        )
    assert error.value.code == "M6_LIVE_PLAN_BINDING"

    receipts = Path(session["namespace"]["receipts_root"])
    receipts.mkdir(parents=True)
    plan_path = receipts / "plan-000000.json"
    assert live._publish_fixture_slot_plan(plan_path, plan, session=session) == plan_path
    with pytest.raises(live.M6LiveError) as error:
        live._publish_fixture_slot_plan(plan_path, plan, session=session)
    assert error.value.code == "M6_LIVE_NO_REPLACE"


def test_process_identity_requires_retained_receipt_and_plan_is_time_bounded(
    tmp_path: Path,
) -> None:
    attempt_root = tmp_path / "attempt"
    plan = _plan_for(
        _ledger_schedule()["slots"][0],
        sequence=0,
        slot_attempt_index=0,
        attempt_root=attempt_root,
    )
    parameters = inspect.signature(live.build_process_identity).parameters
    assert "pid" not in parameters
    assert "start_ticks" not in parameters
    with pytest.raises(live.M6LiveError) as error:
        live.build_process_identity(
            plan=plan,
            supervisor_receipt_path=tmp_path / "missing.json",
            session=_session(),
        )
    assert error.value.code == "M6_LIVE_PATH_UNAVAILABLE"

    session = _session()
    forged_receipt = {
        **live._supervisor_receipt_fields(plan, session),
        "pid": 900,
        "start_ticks": 9000,
        "started_at_utc": NOW,
        "authority_validated_at_utc": NOW,
        "storage_fault_absent_at_utc": NOW,
    }
    forged_receipt["target_argv"] = ["/tmp/forged"]
    forged_receipt = _sealed(forged_receipt, "receipt_sha256")
    forged_path = attempt_root / "forged-supervisor.json"
    forged_path.parent.mkdir(parents=True)
    forged_path.write_bytes(immutable.pretty_json_bytes(forged_receipt))
    forged_path.chmod(0o444)
    with pytest.raises(live.M6LiveError) as error:
        live.build_process_identity(
            plan=plan,
            supervisor_receipt_path=forged_path,
            session=session,
        )
    assert error.value.code == "M6_LIVE_SUPERVISOR_AUTHORITY_UNAVAILABLE"

    with pytest.raises(live.M6LiveError) as error:
        live._build_fixture_process_identity(
            plan=plan,
            pid=901,
            start_ticks=9001,
            started_at_utc="2026-08-08T12:15:01Z",
        )
    assert error.value.code == "M6_LIVE_PLAN_EXPIRED"

    session = _session()
    schedule = _ledger_schedule()
    ledger = live.build_initial_ledger_receipt(session, schedule)
    with pytest.raises(live.M6LiveError) as error:
        live.build_attempt_receipt(
            session=session,
            schedule=schedule,
            ledger=ledger,
            prior_attempts=(),
            plan=plan,
            activation=None,
            outcome="pre_start_failure",
            finished_at_utc="2026-08-08T12:15:01Z",
            failure={"code": "DELAYED", "detail": "expired before start"},
        )
    assert error.value.code == "M6_LIVE_PLAN_EXPIRED"


def test_supervisor_ed25519_authenticator_matches_rfc8032_vector() -> None:
    public_key = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    signature = (
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    )
    live._verify_ed25519(public_key, signature, b"")
    forged = signature[:-2] + "00"
    with pytest.raises(live.M6LiveError) as error:
        live._verify_ed25519(public_key, forged, b"")
    assert error.value.code == "M6_LIVE_SUPERVISOR_AUTHENTICATION"


def test_production_attempt_rejects_direct_self_sealed_process_model() -> None:
    session = _session()
    session["trust_mode"] = live.PRODUCTION_TRUST_MODE
    session["session_authority_sha256"] = immutable.model_seal(
        session, "session_authority_sha256"
    )
    schedule = _ledger_schedule()
    ledger = live.build_initial_ledger_receipt(session, schedule)
    plan = _plan_for(
        schedule["slots"][0],
        sequence=0,
        slot_attempt_index=0,
        session_sha256=session["session_authority_sha256"],
    )
    fixture_direct = live._build_fixture_process_identity(
        plan=plan, pid=999999, start_ticks=777777, started_at_utc=NOW
    )
    forged_direct = copy.deepcopy(fixture_direct)
    forged_direct["authority_mode"] = "retained_exact_child_receipt"
    forged_direct["supervisor_authority_id"] = "caller-forged"
    forged_direct["supervisor_registry_sha256"] = H
    forged_direct["producer_executable"] = "/tmp/caller-forged"
    forged_direct["producer_executable_sha256"] = H2
    forged_direct["supervisor_receipt"] = {
        "path": "/tmp/caller-forged.json",
        "sha256": H,
        "size_bytes": 1,
        "receipt_sha256": H2,
    }
    forged_direct["identity_sha256"] = immutable.model_seal(
        forged_direct, "identity_sha256"
    )
    for direct in (fixture_direct, forged_direct):
        with pytest.raises(live.M6LiveError) as error:
            live.build_attempt_receipt(
                session=session,
                schedule=schedule,
                ledger=ledger,
                prior_attempts=(),
                plan=plan,
                activation=None,
                outcome="post_start_failure",
                started_at_utc=NOW,
                process_identity=direct,
                finished_at_utc=NOW,
                failure={"code": "FORGED", "detail": "arbitrary caller PID"},
            )
        assert error.value.code == "M6_LIVE_PROCESS_CAPABILITY_REQUIRED"


def test_resume_rehashes_all_historical_artifacts(tmp_path: Path) -> None:
    case = _observed_attempt_fixture(tmp_path, maximum=1)
    ledger1 = live.append_attempt_to_ledger(
        previous_ledger=case["ledger"],
        session=case["session"],
        schedule=case["schedule"],
        prior_attempts=(),
        attempt=case["attempt"],
        plan=case["plan"],
        activation=None,
    )
    case["artifact_path"].write_bytes(b"mutated-after-append")
    with pytest.raises(live.M6LiveError) as error:
        live.begin_resume_epoch(
            previous_ledger=ledger1,
            session=case["session"],
            schedule=case["schedule"],
            attempts=(case["attempt"],),
            plans=(case["plan"],),
            ledger_history=(case["ledger"],),
        )
    assert error.value.code == "M6_LIVE_ATTEMPT_ARTIFACT"


def test_published_ledger_reconstructs_and_requires_attempt_inventory(
    tmp_path: Path,
) -> None:
    session = _session(output_root=str(tmp_path / "output"))
    schedule = _ledger_schedule()
    ledger0 = live.build_initial_ledger_receipt(session, schedule)
    plan = _plan_for(
        schedule["slots"][0],
        sequence=0,
        slot_attempt_index=0,
        session_sha256=session["session_authority_sha256"],
    )
    attempt = live.build_attempt_receipt(
        session=session,
        schedule=schedule,
        ledger=ledger0,
        prior_attempts=(),
        plan=plan,
        activation=None,
        outcome="pre_start_failure",
        finished_at_utc=NOW,
        failure={"code": "PRE_START", "detail": "fixture"},
    )
    ledger1 = live.append_attempt_to_ledger(
        previous_ledger=ledger0,
        session=session,
        schedule=schedule,
        prior_attempts=(),
        attempt=attempt,
        plan=plan,
        activation=None,
    )
    receipts = Path(session["namespace"]["receipts_root"])
    receipts.mkdir(parents=True)
    live._publish_fixture_ledger_receipt(
        receipts / "ledger-000000.json", ledger0, session=session, schedule=schedule
    )
    attempt_path = receipts / "attempt-000000.json"
    with pytest.raises(live.M6LiveError) as error:
        live._publish_fixture_attempt_receipt(
            attempt_path, attempt, session=session, plan=plan
        )
    assert error.value.code == "M6_LIVE_INPUT_INVALID"
    live._publish_fixture_slot_plan(receipts / "plan-000000.json", plan, session=session)
    live._publish_fixture_attempt_receipt(
        attempt_path, attempt, session=session, plan=plan
    )
    live._publish_fixture_ledger_receipt(
        receipts / "ledger-000001.json", ledger1, session=session, schedule=schedule
    )
    attempt_path.unlink()
    with pytest.raises(live.M6LiveError) as error:
        live.begin_resume_epoch(
            previous_ledger=ledger1,
            session=session,
            schedule=schedule,
            attempts=(attempt,),
            plans=(plan,),
            ledger_history=(ledger0,),
        )
    assert error.value.code == "M6_LIVE_INPUT_INVALID"


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777
