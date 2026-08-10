#!/usr/bin/env python3
"""Offline M6 live-owner contracts and pure per-slot command planning.

This module deliberately has no hardware probe or process-launch entry point.  It
binds a complete M4-backed schedule to fresh live-session authority, derives one
exact label-driver argv for the next causally eligible slot, and models immutable
attempt/ledger receipts.  Tier-2 activation is derived from a complete Tier-1
journal; callers cannot nominate activated cells or ordinals.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn

from jsonschema import Draft202012Validator, FormatChecker

try:
    from . import campaign_preflight
    from . import immutable_artifact as immutable
    from . import label_driver_authority
    from . import m6_determinism_campaign as m6
    from . import ncu_action_executor
    from . import ncu_collection_receipt
except ImportError:  # direct script/test import
    import campaign_preflight  # type: ignore[no-redef]
    import immutable_artifact as immutable  # type: ignore[no-redef]
    import label_driver_authority  # type: ignore[no-redef]
    import m6_determinism_campaign as m6  # type: ignore[no-redef]
    import ncu_action_executor  # type: ignore[no-redef]
    import ncu_collection_receipt  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "workloads/simbench_bw/schemas"
MATRIX_PATH = REPO_ROOT / "data/local_labels/campaign/sm86_reference_capture_matrix.json"
REGISTRY_PATH = REPO_ROOT / "tools/local_labels/metric_registry.json"
SIDECAR_SCHEMA_PATH = SCHEMA_DIR / "local_label_sidecar.schema.json"
PINS_PATH = REPO_ROOT / "tools/local_labels/campaign_protocol_pins.json"
PREFLIGHT_SCHEMA_PATH = SCHEMA_DIR / "campaign_preflight_manifest.schema.json"
LABEL_DRIVER_PATH = REPO_ROOT / label_driver_authority.LABEL_DRIVER_REL
COLLECTOR_PATH = Path(__file__).resolve()
CAMPAIGN_PREFLIGHT_PATH = COLLECTOR_PATH.with_name("campaign_preflight.py")
IMMUTABLE_ARTIFACT_PATH = COLLECTOR_PATH.with_name("immutable_artifact.py")
NCU_ACTION_EXECUTOR_PATH = COLLECTOR_PATH.with_name("ncu_action_executor.py")
NCU_COLLECTION_RECEIPT_PATH = COLLECTOR_PATH.with_name("ncu_collection_receipt.py")

GENERIC_DRIVER_VERSION = "610.43.02"
TASK17_ONLY_DRIVER_VERSION = "610.43.03"
EXPECTED_ARCH = "sm_86"
EXPECTED_SKU = "rtx_3090_ti"
EXPECTED_PROTOCOL = "v2.1"
EXPECTED_NCU_PREFIX = "2026.1.0"
ZERO_SHA256 = "0" * 64
SCOPE = "generic_full_cohort_tier1_then_derived_tier2_v1"
SESSION_NAMESPACE = "external_m6_live_session_no_replace_v1"
TIER2_DERIVATION = "m6_trigger_decisions_from_complete_uncensored_tier1_v1"
FRESHNESS_SECONDS = 900.0
FUTURE_SKEW_SECONDS = 2.0
EXPECTED_HARDWARE_PROBE = (
    "tools/local_labels/test_campaign_preflight.py::"
    "test_environment_probes_ok_on_reference_host"
)
FIXTURE_TRUST_MODE = "fixture_only_no_production_credit"
PRODUCTION_TRUST_MODE = "production_external_trust"
# No current detached production trust anchor has been admitted. Production
# remains a typed unavailable state until a reviewed code change pins one.
# The exact-child supervisor registry is independently empty. A future reviewed
# source revision must pin one absolute registry path and its byte SHA-256.
_STORAGE_FAULT_NAME = ".m6-live-storage-fault.json"
_EVIDENCE_NAMES = (
    "p1_gate_receipt",
    "current_source_anchor",
    "clean_s_p2_receipt",
    "live_v1_authority",
    "successor_matrix",
    "campaign_lock_ownership",
    "operator_hardware_authority",
)

SCHEMA_PATHS = {
    "session": SCHEMA_DIR / "m6_live_session_authority.schema.json",
    "plan": SCHEMA_DIR / "m6_live_slot_plan.schema.json",
    "attempt": SCHEMA_DIR / "m6_live_attempt_receipt.schema.json",
    "ledger": SCHEMA_DIR / "m6_live_ledger_receipt.schema.json",
    "tier2": SCHEMA_DIR / "m6_tier2_activation.schema.json",
}
SCHEMA_IDS = {
    "session": "simbench_bw/schemas/m6_live_session_authority.schema.json",
    "plan": "simbench_bw/schemas/m6_live_slot_plan.schema.json",
    "attempt": "simbench_bw/schemas/m6_live_attempt_receipt.schema.json",
    "ledger": "simbench_bw/schemas/m6_live_ledger_receipt.schema.json",
    "tier2": "simbench_bw/schemas/m6_tier2_activation.schema.json",
}
SCHEMA_SHA256 = {
    "session": "f7838fb9df1ccfee889b72c6f794e4a7a2c597963d73078aa5d72c17607ddd4e",
    "plan": "5c646e1a0262e2804faf3afdb7dfdb94c37c88787fdec0c58be354168ce2820f",
    "attempt": "0e4196b7aaee95a1cecb1aaf51e7c3d4c72c9c8a9a1eb0025eade632f965f2db",
    "ledger": "50f460130ef880a2c0fbc971c56a5abd36f24c5914084535bd5422cac9b50141",
    "tier2": "52444232ec9ae757505d7ce428180029bdcf610461656a50281b755d546934b3",
}

_OUTCOME_MODEL = {
    "pre_start_failure": {
        "failure_stage": "before_process_start",
        "process_started": False,
        "slot_terminal": False,
        "fresh_process_cost": 0,
    },
    "post_start_failure": {
        "failure_stage": "after_process_start",
        "process_started": True,
        "slot_terminal": False,
        "fresh_process_cost": 1,
    },
    "observed": {
        "failure_stage": "none",
        "process_started": True,
        "slot_terminal": True,
        "fresh_process_cost": 1,
    },
    "excluded_pre_start": {
        "failure_stage": "before_process_start",
        "process_started": False,
        "slot_terminal": True,
        "fresh_process_cost": 0,
    },
    "excluded_post_start": {
        "failure_stage": "after_process_start",
        "process_started": True,
        "slot_terminal": True,
        "fresh_process_cost": 1,
    },
}


class _VerifiedProcessIdentity:
    """Opaque capability minted only after live exact-child verification."""

    __slots__ = ("__model", "__token")

    def __init__(self, model: Mapping[str, Any], token: object):
        if token is not _VERIFIED_PROCESS_TOKEN:
            _fail("M6_LIVE_PROCESS_IDENTITY", "opaque capability constructor denied")
        self.__model = copy.deepcopy(dict(model))
        self.__token = token

    def _materialize(self, token: object) -> dict[str, Any]:
        if token is not _VERIFIED_PROCESS_TOKEN or self.__token is not token:
            _fail("M6_LIVE_PROCESS_IDENTITY", "opaque capability access denied")
        return copy.deepcopy(self.__model)


_VERIFIED_PROCESS_TOKEN = object()


class M6LiveError(ValueError):
    """Stable fail-closed live-owner contract error."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise M6LiveError(code, detail)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest(value: Any) -> str:
    try:
        return immutable.sha256_bytes(immutable.canonical_json_bytes(value))
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_LIVE_JSON_INVALID", str(exc))


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    try:
        return immutable.seal_model(value, field)
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_LIVE_SEAL_INVALID", str(exc))


def _verify_seal(value: Mapping[str, Any], field: str, *, role: str) -> None:
    try:
        immutable.verify_model_seal(value, field)
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_LIVE_SEAL_INVALID", f"{role}: {exc}")


def _parse_time(value: Any, *, role: str) -> datetime:
    if type(value) is not str or not value:
        _fail("M6_LIVE_TIME_INVALID", role)
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except (TypeError, ValueError) as exc:
        _fail("M6_LIVE_TIME_INVALID", f"{role}: {exc}")
    if parsed.tzinfo is None:
        _fail("M6_LIVE_TIME_INVALID", f"{role}: timezone required")
    return parsed.astimezone(timezone.utc)


def _trusted_now() -> datetime:
    """Return the process clock used by production admission boundaries."""
    return datetime.now(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _fresh_at(value: Any, *, role: str, now: datetime) -> datetime:
    parsed = _parse_time(value, role=role)
    age = (now - parsed).total_seconds()
    future = (parsed - now).total_seconds()
    if future > FUTURE_SKEW_SECONDS:
        _fail("M6_LIVE_PREFLIGHT_CLOCK_SKEW", role)
    if age > FRESHNESS_SECONDS:
        _fail("M6_LIVE_PREFLIGHT_EXPIRED", f"{role}: {age:.3f}s")
    return parsed


def _canonical_absolute(path: Path, *, role: str, must_exist: bool = False) -> Path:
    if not isinstance(path, Path):
        _fail("M6_LIVE_PATH_INVALID", f"{role}: Path required")
    text = os.fspath(path)
    if (
        not path.is_absolute()
        or text.startswith("//")
        or os.path.normpath(text) != text
        or "\x00" in text
    ):
        _fail("M6_LIVE_PATH_INVALID", f"{role}: canonical absolute path required")
    if must_exist:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            _fail("M6_LIVE_PATH_UNAVAILABLE", f"{role}: {exc}")
        if resolved != path:
            _fail("M6_LIVE_PATH_INVALID", f"{role}: symlink traversal is forbidden")
    return path


def _outside_repository(path: Path, *, repo_root: Path, role: str) -> None:
    candidate = _canonical_absolute(path, role=role)
    repository = _canonical_absolute(repo_root, role="repository root", must_exist=True)
    try:
        candidate.relative_to(repository)
    except ValueError:
        return
    _fail("M6_LIVE_NAMESPACE_OVERLAP", f"{role} overlaps the repository")


def _read_regular(path: Path, *, role: str) -> bytes:
    try:
        return immutable.read_regular(path)
    except (OSError, immutable.ImmutableArtifactError) as exc:
        _fail("M6_LIVE_INPUT_INVALID", f"{role}: {exc}")


def _load_json_path(path: Path, *, role: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(path, role=role)
    try:
        value = immutable.strict_json_loads(payload, label=role)
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_LIVE_INPUT_INVALID", str(exc))
    if type(value) is not dict:
        _fail("M6_LIVE_INPUT_INVALID", f"{role}: object required")
    return value, payload


def _file_identity(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = _read_regular(path, role=f"file identity {path}")
    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError:
        _fail("M6_LIVE_PATH_INVALID", f"authority path is outside repository: {path}")
    return {
        "path": relative,
        "sha256": immutable.sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def load_schema(kind: str) -> dict[str, Any]:
    path = SCHEMA_PATHS.get(kind)
    if path is None:
        _fail("M6_LIVE_SCHEMA_KIND", repr(kind))
    payload = _read_regular(path, role=f"{kind} schema")
    if immutable.sha256_bytes(payload) != SCHEMA_SHA256[kind]:
        _fail("M6_LIVE_SCHEMA_IDENTITY", f"{kind}: SHA-256 mismatch")
    try:
        schema = immutable.strict_json_loads(payload, label=f"{kind} schema")
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_LIVE_SCHEMA_INVALID", str(exc))
    if (
        type(schema) is not dict
        or schema.get("$id") != SCHEMA_IDS[kind]
        or schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        _fail("M6_LIVE_SCHEMA_IDENTITY", f"{kind}: shape mismatch")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        _fail("M6_LIVE_SCHEMA_INVALID", f"{kind}: {exc}")
    return schema


def validate_schema(kind: str, value: Any) -> None:
    errors = sorted(
        Draft202012Validator(
            load_schema(kind), format_checker=FormatChecker()
        ).iter_errors(value),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        error = errors[0]
        pointer = "/" + "/".join(str(part) for part in error.absolute_path)
        _fail(
            "M6_LIVE_SCHEMA_VALIDATION",
            f"{kind}{pointer}: {error.message}",
        )


def _schema_authorities(repo_root: Path) -> list[dict[str, Any]]:
    rows = []
    for kind in sorted(SCHEMA_PATHS):
        path = repo_root / SCHEMA_PATHS[kind].relative_to(REPO_ROOT)
        identity = _file_identity(path, repo_root=repo_root)
        if identity["sha256"] != SCHEMA_SHA256[kind]:
            _fail("M6_LIVE_SCHEMA_IDENTITY", kind)
        rows.append({"name": kind, **identity})
    return rows


def _schedule_path_identity(
    schedule: Mapping[str, Any], schedule_path: Path
) -> dict[str, Any]:
    path = _canonical_absolute(schedule_path, role="schedule", must_exist=True)
    payload = _read_regular(path, role="M6 schedule")
    try:
        parsed = immutable.strict_json_loads(payload, label="M6 schedule")
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_LIVE_SCHEDULE_INVALID", str(exc))
    if parsed != schedule:
        _fail("M6_LIVE_SCHEDULE_IDENTITY", "supplied schedule differs from path bytes")
    return {
        "path": str(path),
        "sha256": immutable.sha256_bytes(payload),
        "size_bytes": len(payload),
        "schedule_sha256": schedule.get("schedule_sha256"),
        "source_identity_sha256": schedule.get("source_identity_sha256"),
        "config_identity_sha256": schedule.get("config_identity_sha256"),
        "metric_identity_sha256": schedule.get("metric_identity_sha256"),
        "m4_identity_sha256": (schedule.get("m4_identity") or {}).get(
            "identity_sha256"
        ),
    }


def validate_campaign_inputs(
    schedule: Mapping[str, Any],
    *,
    schedule_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Revalidate the complete M4-backed schedule and every live input identity."""
    repository = _canonical_absolute(repo_root, role="repository root", must_exist=True)
    try:
        m6.validate_schedule(schedule, repo_root=repository, rehash_sources=True)
    except m6.M6Error as exc:
        _fail("M6_LIVE_SCHEDULE_INVALID", str(exc))
    if schedule.get("schema_version") != m6.ACTIVE_SCHEMA_VERSION:
        _fail("M6_LIVE_SCHEDULE_VERSION", "active schedule v2 is required")
    m4_identity = schedule.get("m4_identity")
    if (
        not isinstance(m4_identity, Mapping)
        or m4_identity.get("status") != "complete"
        or m4_identity.get("completed_cells") != m6.EXPECTED_RUNNABLE
        or m4_identity.get("expected_cells") != m6.EXPECTED_RUNNABLE
        or len(m4_identity.get("cell_launch_identities", ()))
        != m6.EXPECTED_RUNNABLE
    ):
        _fail("M6_LIVE_M4_INCOMPLETE", "complete sealed 122-cell M4 is required")
    if schedule.get("counts") != {
        "canonical_cells": 344,
        "runnable_cells": 122,
        "excluded_cells": 222,
        "configs": 6,
        "selected_metrics": 55,
        "registry_metrics": 61,
        "tier1_slots": 8540,
        "tier2_candidate_slots": 1220,
        "maximum_slots": 9760,
        "tier1_fresh_processes": 8540,
        "maximum_fresh_processes": 9760,
    }:
        _fail("M6_LIVE_SCHEDULE_COUNTS", "full-cohort counts differ")

    source = schedule.get("source_identity")
    if not isinstance(source, Mapping):
        _fail("M6_LIVE_SCHEDULE_IDENTITY", "source identity is absent")
    canonical = {
        "matrix": (MATRIX_PATH, "canonical_matrix"),
        "metric_registry": (REGISTRY_PATH, "metric_registry"),
        "sidecar_schema": (SIDECAR_SCHEMA_PATH, "sidecar_schema"),
    }
    values: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    for name, (canonical_path, source_name) in canonical.items():
        path = repository / canonical_path.relative_to(REPO_ROOT)
        document, payload = _load_json_path(path, role=name)
        expected = source.get(source_name)
        identity = _file_identity(path, repo_root=repository)
        if expected != identity:
            _fail("M6_LIVE_INPUT_IDENTITY", f"{name} differs from schedule authority")
        values[name] = document
        bindings[name] = identity

    matrix = values["matrix"]
    registry = values["metric_registry"]
    sidecar_schema = values["sidecar_schema"]
    if (
        matrix.get("matrix_seal_sha256") != source.get("matrix_seal_sha256")
        or matrix.get("authority", {}).get("sha256")
        != source.get("roster_authority_sha256")
        or matrix.get("chunking", {}).get("runnable_membership_sha256")
        != source.get("runnable_membership_sha256")
        or registry.get("registry_version") != 2
        or registry.get("label_schema_version") != 5
        or sidecar_schema.get("schema_version") != 5
    ):
        _fail("M6_LIVE_INPUT_IDENTITY", "matrix/registry/sidecar semantics differ")

    pins_path = repository / PINS_PATH.relative_to(REPO_ROOT)
    pins, _pins_payload = _load_json_path(pins_path, role="campaign protocol pins")
    pins_identity = _file_identity(pins_path, repo_root=repository)
    driver = (pins.get("toolchain") or {}).get("driver_version")
    if driver == TASK17_ONLY_DRIVER_VERSION:
        _fail(
            "M6_LIVE_TASK17_DRIVER_FORBIDDEN",
            "610.43.03 is not generic M6 authority",
        )
    if (
        pins.get("arch") != EXPECTED_ARCH
        or pins.get("sku") != EXPECTED_SKU
        or driver != GENERIC_DRIVER_VERSION
        or (pins.get("toolchain") or {}).get("ncu_version_prefix")
        != EXPECTED_NCU_PREFIX
        or (pins.get("measurement_protocol") or {}).get("version")
        != EXPECTED_PROTOCOL
    ):
        _fail("M6_LIVE_PINS_INVALID", "generic sm_86 campaign pins differ")

    selected = [
        row for row in schedule.get("metrics", ())
        if isinstance(row, Mapping) and row.get("capture_status") == "selected"
    ]
    collection = (matrix.get("metric_registry") or {}).get("collection") or {}
    flat = collection.get("selected_flat_metrics")
    unit = collection.get("selected_unit_metrics")
    if (
        len(selected) != 55
        or not isinstance(flat, list)
        or len(flat) != 2
        or not isinstance(unit, list)
        or len(unit) != 53
        or len(set(flat + unit)) != 55
        or {row.get("ncu_metric") for row in selected} != set(flat + unit)
    ):
        _fail("M6_LIVE_METRIC_SET", "exact 55-metric M6 selection is required")

    return {
        "schedule": _schedule_path_identity(schedule, schedule_path),
        "schedule_document": schedule,
        "matrix": matrix,
        "metric_registry": registry,
        "sidecar_schema": sidecar_schema,
        "pins": pins,
        "inputs": {
            **bindings,
            "campaign_protocol_pins": pins_identity,
        },
        "schema_authorities": _schema_authorities(repository),
        "implicit_flat_metrics": list(flat),
        "explicit_unit_metrics": list(unit),
        "requested_ncu_metrics": list(flat) + list(unit),
    }


def _load_preflight_schema(repo_root: Path) -> dict[str, Any]:
    path = repo_root / PREFLIGHT_SCHEMA_PATH.relative_to(REPO_ROOT)
    schema, _ = _load_json_path(path, role="campaign preflight schema")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        _fail("M6_LIVE_PREFLIGHT_SCHEMA", str(exc))
    return schema


def _preflight_projection(
    phase: str,
    path: Path,
    *,
    source_commit: str,
    pins_identity: Mapping[str, Any],
    matrix_identity: Mapping[str, Any],
    output_root: Path,
    now: datetime,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight_path = _canonical_absolute(path, role=f"{phase} preflight", must_exist=True)
    document, payload = _load_json_path(preflight_path, role=f"{phase} preflight")
    errors = sorted(
        Draft202012Validator(
            _load_preflight_schema(repo_root), format_checker=FormatChecker()
        ).iter_errors(document),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        _fail("M6_LIVE_PREFLIGHT_INVALID", f"{phase}: {errors[0].message}")
    if document.get("schema_version") != 4:
        _fail("M6_LIVE_PREFLIGHT_VERSION", f"{phase}: live v4 required")
    checks = document.get("checks")
    if not isinstance(checks, Mapping):
        _fail("M6_LIVE_PREFLIGHT_INVALID", f"{phase}: checks absent")
    if any(
        not isinstance(checks.get(name), Mapping)
        or checks[name].get("status") != campaign_preflight.STATUS_OK
        for name in campaign_preflight._BLOCKING_CHECKS
    ):
        _fail("M6_LIVE_READINESS_FALSE", f"{phase}: blocking check is not ok")
    driver = checks.get("driver") or {}
    observed_driver = driver.get("observed")
    if observed_driver == TASK17_ONLY_DRIVER_VERSION:
        _fail(
            "M6_LIVE_TASK17_DRIVER_FORBIDDEN",
            f"{phase}: 610.43.03 is Task-17-only",
        )
    device = checks.get("device") or {}
    trace_budget = checks.get("trace_budget") or {}
    ncu = checks.get("ncu") or {}
    generated_raw = document.get("generated_at_utc")
    generated = _fresh_at(
        generated_raw, role=f"{phase}.generated_at_utc", now=now
    )
    captured = _parse_time(document.get("captured_at_utc"), role=f"{phase}.captured_at_utc")
    try:
        budget_root = Path(trace_budget.get("output_root", ""))
    except TypeError:
        budget_root = Path("")
    gpu_uuid = device.get("gpu_uuid_sha256")
    if (
        generated != captured
        or document.get("git_head") != source_commit
        or document.get("working_tree_clean") is not True
        or document.get("pins_sha256") != pins_identity.get("sha256")
        or document.get("capture_matrix_sha256") != matrix_identity.get("sha256")
        or document.get("arch") != EXPECTED_ARCH
        or document.get("sku") != EXPECTED_SKU
        or document.get("protocol_version") != EXPECTED_PROTOCOL
        or document.get("blockers") != []
        or document.get("capture_admitted") is not True
        or driver.get("expected") != GENERIC_DRIVER_VERSION
        or observed_driver != GENERIC_DRIVER_VERSION
        or device.get("selection_policy")
        != campaign_preflight.DEVICE_SELECTION_POLICY
        or device.get("selected_host_gpu_index") != 0
        or device.get("nvidia_smi_device_count") != 1
        or not _is_sha256(gpu_uuid)
        or budget_root != output_root
    ):
        _fail("M6_LIVE_PREFLIGHT_BINDING", phase)
    try:
        ncu_identity = ncu_action_executor.build_expected_ncu_identity(
            resolved_path=ncu.get("resolved_path"),
            executable_sha256=ncu.get("executable_sha256"),
            version_line=ncu.get("version_line"),
        )
    except ncu_action_executor.NcuActionExecutorError as exc:
        _fail("M6_LIVE_NCU_AUTHORITY", f"{phase}: {exc}")
    if not ncu_identity.version_line.startswith(f"Version {EXPECTED_NCU_PREFIX}"):
        _fail("M6_LIVE_NCU_AUTHORITY", f"{phase}: version differs")
    projection = {
        "phase": phase,
        "path": str(preflight_path),
        "sha256": immutable.sha256_bytes(payload),
        "size_bytes": len(payload),
        "generated_at_utc": generated_raw,
        "git_head": source_commit,
        "pins_sha256": pins_identity["sha256"],
        "matrix_sha256": matrix_identity["sha256"],
        "checks_sha256": _digest(checks),
        "gpu_uuid_sha256": gpu_uuid,
        "driver_version": GENERIC_DRIVER_VERSION,
        "output_root": str(output_root),
        "capture_admitted": True,
    }
    return projection, {
        "document": document,
        "generated": generated,
        "ncu_identity": {
            "resolved_path": ncu_identity.resolved_path,
            "executable_sha256": ncu_identity.executable_sha256,
            "version_line": ncu_identity.version_line,
        },
    }


def _external_file_identity(path: Path, *, role: str) -> dict[str, Any]:
    absolute = _canonical_absolute(path, role=role, must_exist=True)
    payload = _read_regular(absolute, role=role)
    return {
        "path": str(absolute),
        "sha256": immutable.sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _load_sealed_external(
    path: Path,
    *,
    role: str,
    required_fields: set[str],
    seal_field: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    absolute = _canonical_absolute(path, role=role, must_exist=True)
    document, payload = _load_json_path(absolute, role=role)
    if set(document) != required_fields:
        _fail("M6_LIVE_EXTERNAL_AUTHORITY_INVALID", f"{role}: fields differ")
    _verify_seal(document, seal_field, role=role)
    return document, {
        "path": str(absolute),
        "sha256": immutable.sha256_bytes(payload),
        "size_bytes": len(payload),
        "model_sha256": document[seal_field],
    }


def _require_exact_status(document: Mapping[str, Any], *, role: str, kind: str) -> None:
    if (
        document.get("schema_version") != 1
        or document.get("kind") != kind
        or document.get("status") != "pass"
    ):
        _fail("M6_LIVE_EXTERNAL_AUTHORITY_INVALID", f"{role}: status differs")


def _validate_admission_evidence(
    *,
    evidence_paths: Mapping[str, Path],
    trust_anchor_path: Path,
    trusted_anchor_sha256: str,
    trust_mode: str,
    context: Mapping[str, Any],
    output_root: Path,
    now: datetime,
    repo_root: Path,
) -> dict[str, Any]:
    """Load every detached prerequisite under one out-of-band trust anchor."""
    if set(evidence_paths) != set(_EVIDENCE_NAMES):
        _fail("M6_LIVE_PREREQUISITE_SET", "exact admission evidence paths required")
    if not _is_sha256(trusted_anchor_sha256):
        _fail("M6_LIVE_OPERATOR_TRUST_ANCHOR_UNAVAILABLE", "trusted digest absent")
    anchor_path = _canonical_absolute(
        trust_anchor_path, role="operator trust anchor", must_exist=True
    )
    anchor, anchor_payload = _load_json_path(anchor_path, role="operator trust anchor")
    if immutable.sha256_bytes(anchor_payload) != trusted_anchor_sha256:
        _fail("M6_LIVE_OPERATOR_TRUST_ANCHOR_MISMATCH", "detached bytes differ")
    anchor_fields = {
        "schema_version", "kind", "trust_mode", "trust_anchor_id", "status",
        "source_commit", "source_tree", "valid_from_utc", "valid_until_utc",
        "evidence_sha256", "anchor_sha256",
    }
    if set(anchor) != anchor_fields:
        _fail("M6_LIVE_OPERATOR_TRUST_ANCHOR_INVALID", "fields differ")
    _verify_seal(anchor, "anchor_sha256", role="operator trust anchor")
    if (
        anchor.get("schema_version") != 1
        or anchor.get("kind") != "m6_external_operator_trust_anchor_v1"
        or anchor.get("trust_mode") != trust_mode
        or anchor.get("status") != "trusted"
        or not _is_commit(anchor.get("source_commit"))
        or not _is_commit(anchor.get("source_tree"))
    ):
        _fail("M6_LIVE_OPERATOR_TRUST_ANCHOR_INVALID", "semantics differ")
    valid_from = _parse_time(anchor.get("valid_from_utc"), role="trust valid_from")
    valid_until = _parse_time(anchor.get("valid_until_utc"), role="trust valid_until")
    if not valid_from <= now <= valid_until:
        _fail("M6_LIVE_OPERATOR_TRUST_ANCHOR_EXPIRED", "current time outside validity")

    loaded: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    expected_hashes = anchor.get("evidence_sha256")
    if type(expected_hashes) is not dict or set(expected_hashes) != set(_EVIDENCE_NAMES):
        _fail("M6_LIVE_OPERATOR_TRUST_ANCHOR_INVALID", "evidence set differs")
    seal_fields = {
        "p1_gate_receipt": "receipt_sha256",
        "current_source_anchor": "anchor_sha256",
        "clean_s_p2_receipt": "receipt_sha256",
        "live_v1_authority": None,
        "successor_matrix": None,
        "campaign_lock_ownership": "receipt_sha256",
        "operator_hardware_authority": "authority_sha256",
    }
    for name in _EVIDENCE_NAMES:
        path = _canonical_absolute(
            evidence_paths[name], role=name, must_exist=True
        )
        document, payload = _load_json_path(path, role=name)
        observed_sha = immutable.sha256_bytes(payload)
        if expected_hashes.get(name) != observed_sha:
            _fail("M6_LIVE_OPERATOR_TRUST_ANCHOR_MISMATCH", name)
        seal_field = seal_fields[name]
        if seal_field is not None:
            _verify_seal(document, seal_field, role=name)
        loaded[name] = document
        identity = {
            "path": str(path),
            "sha256": observed_sha,
            "size_bytes": len(payload),
        }
        if seal_field is not None:
            identity["model_sha256"] = document[seal_field]
        identities[name] = identity

    source_commit = anchor["source_commit"]
    source_tree = anchor["source_tree"]
    p1 = loaded["p1_gate_receipt"]
    p1_fields = {
        "schema_version", "kind", "status", "source_commit", "source_tree",
        "default_verification", "capture_focused_battery",
        "hardware_probe_deselected", "identity_closure", "receipt_sha256",
    }
    if set(p1) != p1_fields:
        _fail("M6_LIVE_P1_INVALID", "gate receipt fields differ")
    _require_exact_status(p1, role="P1 gate", kind="m6_p1_gate_receipt_v1")
    if (
        p1.get("source_commit") != source_commit
        or p1.get("source_tree") != source_tree
        or p1.get("default_verification") != "pass"
        or p1.get("capture_focused_battery") != "pass"
        or p1.get("hardware_probe_deselected") != EXPECTED_HARDWARE_PROBE
        or p1.get("identity_closure") != "pass"
    ):
        _fail("M6_LIVE_P1_INVALID", "byte-scoped gate differs")

    current = loaded["current_source_anchor"]
    current_fields = {
        "schema_version", "kind", "status", "source_commit", "source_tree",
        "working_tree_clean", "p1_gate_receipt_sha256", "anchor_sha256",
    }
    if set(current) != current_fields:
        _fail("M6_LIVE_CURRENT_SOURCE_INVALID", "anchor fields differ")
    _require_exact_status(
        current, role="current source", kind="m6_current_source_external_anchor_v1"
    )
    if (
        current.get("source_commit") != source_commit
        or current.get("source_tree") != source_tree
        or current.get("working_tree_clean") is not True
        or current.get("p1_gate_receipt_sha256") != p1["receipt_sha256"]
    ):
        _fail("M6_LIVE_CURRENT_SOURCE_INVALID", "source bindings differ")

    live = loaded["live_v1_authority"]
    live_counts = live.get("counts") if isinstance(live, Mapping) else None
    live_statuses = Counter(
        cell.get("resolution_status")
        for cell in live.get("cells", [])
        if isinstance(cell, Mapping)
    )
    if (
        live.get("schema_version") != 1
        or live.get("roster_id") != "sm86_capture_roster_authority_v1"
        or live.get("arch") != EXPECTED_ARCH
        or live.get("sku") != EXPECTED_SKU
        or live.get("source_commit") != source_commit
        or live.get("accounting_identity_holds") is not True
        or not isinstance(live.get("cells"), list)
        or len(live["cells"]) != 344
        or not isinstance(live_counts, Mapping)
        or live_counts.get("universe") != 344
        or live_counts.get("runnable") != 122
        or live_counts.get("excluded") != 222
        or live_statuses != Counter({"runnable": 122, "excluded": 222})
    ):
        _fail("M6_LIVE_P2_LIVE_V1_INVALID", "live-v1 authority differs")

    successor = loaded["successor_matrix"]
    if successor != context["matrix"]:
        _fail("M6_LIVE_P2_SUCCESSOR_INVALID", "successor matrix differs from schedule")
    if (successor.get("authority") or {}).get("sha256") != identities[
        "live_v1_authority"
    ]["sha256"]:
        _fail("M6_LIVE_P2_SUCCESSOR_INVALID", "live-v1 byte binding differs")
    if trust_mode == PRODUCTION_TRUST_MODE:
        for role, document, schema_name in (
            (
                "live-v1 authority",
                live,
                "capture_roster_authority.schema.json",
            ),
            ("successor matrix", successor, "sm86_capture_matrix.schema.json"),
        ):
            schema, _ = _load_json_path(
                repo_root / "workloads/simbench_bw/schemas" / schema_name,
                role=f"{role} schema",
            )
            errors = sorted(
                Draft202012Validator(schema).iter_errors(document),
                key=lambda error: [str(part) for part in error.absolute_path],
            )
            if errors:
                _fail(
                    "M6_LIVE_EXTERNAL_AUTHORITY_INVALID",
                    f"{role}: {errors[0].message}",
                )

    clean = loaded["clean_s_p2_receipt"]
    clean_fields = {
        "schema_version", "kind", "status", "source_commit", "source_tree",
        "working_tree_clean", "p1_source_anchor_sha256", "producer_start_head",
        "authority_head", "producer_final_head", "live_v1_authority_sha256",
        "successor_matrix_sha256", "verified_runnable_cells", "receipt_sha256",
    }
    if set(clean) != clean_fields:
        _fail("M6_LIVE_P2_INVALID", "clean-S receipt fields differ")
    _require_exact_status(clean, role="clean-S P2", kind="m6_clean_s_p2_receipt_v1")
    if (
        clean.get("source_commit") != source_commit
        or clean.get("source_tree") != source_tree
        or clean.get("working_tree_clean") is not True
        or clean.get("p1_source_anchor_sha256") != current["anchor_sha256"]
        or clean.get("producer_start_head") != source_commit
        or clean.get("authority_head") != source_commit
        or clean.get("producer_final_head") != source_commit
        or clean.get("live_v1_authority_sha256")
        != identities["live_v1_authority"]["sha256"]
        or clean.get("successor_matrix_sha256")
        != identities["successor_matrix"]["sha256"]
        or clean.get("verified_runnable_cells") != 122
    ):
        _fail("M6_LIVE_P2_INVALID", "clean-S causal bindings differ")

    operator = _validate_operator_authority(
        loaded["operator_hardware_authority"],
        source_commit=source_commit,
        gpu_uuid_sha256=loaded["campaign_lock_ownership"].get(
            "gpu_uuid_sha256", ""
        ),
        output_root=output_root,
        max_fresh_processes_per_resume=context["schedule_document"]["cost_gate"][
            "max_new_profiles_per_resume"
        ],
    )
    lock = loaded["campaign_lock_ownership"]
    lock_fields = {
        "schema_version", "kind", "status", "source_commit", "gpu_uuid_sha256",
        "output_root", "operator_authority_sha256", "lock_path", "owner_pid",
        "owner_start_ticks", "acquired_at_utc", "expires_at_utc",
        "storage_fault_receipt_path", "receipt_sha256",
    }
    if set(lock) != lock_fields:
        _fail("M6_LIVE_CAMPAIGN_LOCK_INVALID", "ownership fields differ")
    _require_exact_status(
        lock, role="campaign lock", kind="m6_campaign_lock_ownership_v1"
    )
    acquired = _fresh_at(lock.get("acquired_at_utc"), role="campaign lock", now=now)
    expires = _parse_time(lock.get("expires_at_utc"), role="campaign lock expiry")
    expected_lock = output_root / ".m6-live-campaign.lock"
    expected_fault = output_root / _STORAGE_FAULT_NAME
    if (
        lock.get("source_commit") != source_commit
        or lock.get("gpu_uuid_sha256") != operator["gpu_uuid_sha256"]
        or lock.get("output_root") != str(output_root)
        or lock.get("operator_authority_sha256") != operator["authority_sha256"]
        or lock.get("lock_path") != str(expected_lock)
        or type(lock.get("owner_pid")) is not int
        or lock["owner_pid"] < 1
        or type(lock.get("owner_start_ticks")) is not int
        or lock["owner_start_ticks"] < 1
        or expires <= acquired
        or now >= expires
        or lock.get("storage_fault_receipt_path") != str(expected_fault)
    ):
        _fail("M6_LIVE_CAMPAIGN_LOCK_INVALID", "ownership bindings differ")
    if expected_fault.exists() or expected_fault.is_symlink():
        _fail("M6_LIVE_STORAGE_FAULT_PRESENT", str(expected_fault))
    if trust_mode == PRODUCTION_TRUST_MODE:
        _validate_live_campaign_lock(lock)

    return {
        "trust_mode": trust_mode,
        "trust_anchor": {
            "path": str(anchor_path),
            "sha256": trusted_anchor_sha256,
            "size_bytes": len(anchor_payload),
            "model_sha256": anchor["anchor_sha256"],
            "trust_anchor_id": anchor["trust_anchor_id"],
        },
        "evidence": identities,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "operator": operator,
        "gpu_uuid_sha256": operator["gpu_uuid_sha256"],
        "campaign_lock": copy.deepcopy(lock),
    }


def _validate_live_campaign_lock(lock: Mapping[str, Any]) -> None:
    """Verify the production owner process still holds the exact advisory lock."""
    lock_path = Path(lock["lock_path"])
    try:
        metadata = lock_path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or lock_path.is_symlink():
            raise OSError("campaign lock is not a regular file")
        process_stat = Path(f"/proc/{lock['owner_pid']}/stat").read_text(
            encoding="utf-8"
        )
        closing = process_stat.rfind(")")
        fields = process_stat[closing + 2 :].split()
        observed_start_ticks = int(fields[19])
        locks = Path("/proc/locks").read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError, IndexError) as exc:
        _fail("M6_LIVE_CAMPAIGN_LOCK_NOT_OWNED", str(exc))
    device = (
        f"{os.major(metadata.st_dev):02x}:{os.minor(metadata.st_dev):02x}:"
        f"{metadata.st_ino}"
    )
    held = any(
        len(parts := line.split()) >= 6
        and parts[3] == "WRITE"
        and parts[4] == str(lock["owner_pid"])
        and parts[5].lower() == device.lower()
        for line in locks
    )
    if observed_start_ticks != lock["owner_start_ticks"] or not held:
        _fail(
            "M6_LIVE_CAMPAIGN_LOCK_NOT_OWNED",
            "PID/start identity does not own the exact lock inode",
        )


def _validate_operator_authority(
    value: Any,
    *,
    source_commit: str,
    gpu_uuid_sha256: str,
    output_root: Path,
    max_fresh_processes_per_resume: int,
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("M6_LIVE_OPERATOR_AUTHORITY", "exact object required")
    required = {
        "kind", "authority_id", "authorized_at_utc", "source_commit",
        "gpu_uuid_sha256", "driver_version", "output_root",
        "max_fresh_processes_per_resume", "scope", "allow_hardware_queries",
        "allow_hardware_spawns", "allow_privileged_commands", "authority_sha256",
    }
    if set(value) != required:
        _fail("M6_LIVE_OPERATOR_AUTHORITY", "fields differ")
    _verify_seal(value, "authority_sha256", role="operator hardware authority")
    if value.get("driver_version") == TASK17_ONLY_DRIVER_VERSION:
        _fail("M6_LIVE_TASK17_DRIVER_FORBIDDEN", "operator authority")
    fixed = {
        "kind": "m6_operator_hardware_authority_v1",
        "source_commit": source_commit,
        "gpu_uuid_sha256": gpu_uuid_sha256,
        "driver_version": GENERIC_DRIVER_VERSION,
        "output_root": str(output_root),
        "max_fresh_processes_per_resume": max_fresh_processes_per_resume,
        "scope": SCOPE,
        "allow_hardware_queries": True,
        "allow_hardware_spawns": True,
        "allow_privileged_commands": False,
    }
    if any(value.get(name) != expected for name, expected in fixed.items()):
        _fail("M6_LIVE_OPERATOR_AUTHORITY", "external bindings differ")
    _parse_time(value.get("authorized_at_utc"), role="operator authorized_at_utc")
    authority_id = value.get("authority_id")
    if type(authority_id) is not str or not authority_id or len(authority_id) > 128:
        _fail("M6_LIVE_OPERATOR_AUTHORITY", "authority_id is invalid")
    return copy.deepcopy(value)


def _label_driver_identity(repo_root: Path) -> dict[str, Any]:
    path = repo_root / LABEL_DRIVER_PATH.relative_to(REPO_ROOT)
    payload = _read_regular(path, role="label driver")
    try:
        metadata = path.lstat()
    except OSError as exc:
        _fail("M6_LIVE_TOOL_IDENTITY", f"label driver: {exc}")
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or stat.S_IMODE(metadata.st_mode) != label_driver_authority.LABEL_DRIVER_MODE
        or immutable.sha256_bytes(payload) != label_driver_authority.LABEL_DRIVER_SHA256
    ):
        _fail("M6_LIVE_TOOL_IDENTITY", "label driver authority differs")
    return {
        **_file_identity(path, repo_root=repo_root),
        "mode": label_driver_authority.LABEL_DRIVER_MODE,
    }


def _tool_identities(repo_root: Path) -> dict[str, Any]:
    source_paths = {
        "collector": COLLECTOR_PATH,
        "campaign_preflight": CAMPAIGN_PREFLIGHT_PATH,
        "immutable_artifact": IMMUTABLE_ARTIFACT_PATH,
        "ncu_action_executor": NCU_ACTION_EXECUTOR_PATH,
        "ncu_collection_receipt": NCU_COLLECTION_RECEIPT_PATH,
    }
    identities = {
        name: _file_identity(
            repo_root / path.relative_to(REPO_ROOT), repo_root=repo_root
        )
        for name, path in source_paths.items()
    }
    identities["label_driver"] = _label_driver_identity(repo_root)
    return identities


def _production_trust_configuration() -> tuple[Path, str]:
    _fail(
        "M6_LIVE_OPERATOR_TRUST_ANCHOR_UNAVAILABLE",
        "no reviewed production trust anchor is pinned in this source revision",
    )


def build_session_authority(
    *,
    schedule: Mapping[str, Any],
    schedule_path: Path,
    admission_evidence_paths: Mapping[str, Path],
    p3_preflight_path: Path,
    p4_preflight_path: Path,
    session_preflight_path: Path,
    output_root: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build only under a separately installed production trust anchor."""
    trust_path, trust_sha256 = _production_trust_configuration()
    return _build_session_authority(
        schedule=schedule,
        schedule_path=schedule_path,
        admission_evidence_paths=admission_evidence_paths,
        p3_preflight_path=p3_preflight_path,
        p4_preflight_path=p4_preflight_path,
        session_preflight_path=session_preflight_path,
        output_root=output_root,
        trust_anchor_path=trust_path,
        trusted_anchor_sha256=trust_sha256,
        trust_mode=PRODUCTION_TRUST_MODE,
        trusted_now=_trusted_now(),
        repo_root=repo_root,
    )


def _build_fixture_session_authority(
    *,
    schedule: Mapping[str, Any],
    schedule_path: Path,
    admission_evidence_paths: Mapping[str, Path],
    p3_preflight_path: Path,
    p4_preflight_path: Path,
    session_preflight_path: Path,
    output_root: Path,
    trust_anchor_path: Path,
    trusted_anchor_sha256: str,
    trusted_now: datetime,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Private fixture-only admission; it grants no production or hardware credit."""
    return _build_session_authority(
        schedule=schedule,
        schedule_path=schedule_path,
        admission_evidence_paths=admission_evidence_paths,
        p3_preflight_path=p3_preflight_path,
        p4_preflight_path=p4_preflight_path,
        session_preflight_path=session_preflight_path,
        output_root=output_root,
        trust_anchor_path=trust_anchor_path,
        trusted_anchor_sha256=trusted_anchor_sha256,
        trust_mode=FIXTURE_TRUST_MODE,
        trusted_now=trusted_now,
        repo_root=repo_root,
    )


def _build_session_authority(
    *,
    schedule: Mapping[str, Any],
    schedule_path: Path,
    admission_evidence_paths: Mapping[str, Path],
    p3_preflight_path: Path,
    p4_preflight_path: Path,
    session_preflight_path: Path,
    output_root: Path,
    trust_anchor_path: Path,
    trusted_anchor_sha256: str,
    trust_mode: str,
    trusted_now: datetime,
    repo_root: Path,
) -> dict[str, Any]:
    if trust_mode not in {FIXTURE_TRUST_MODE, PRODUCTION_TRUST_MODE}:
        _fail("M6_LIVE_OPERATOR_TRUST_ANCHOR_INVALID", "unknown trust mode")
    if trusted_now.tzinfo is None:
        _fail("M6_LIVE_TIME_INVALID", "trusted clock must be timezone-aware")
    now = trusted_now.astimezone(timezone.utc)
    repository = _canonical_absolute(repo_root, role="repository root", must_exist=True)
    root = _canonical_absolute(output_root, role="M6 output root")
    _outside_repository(root, repo_root=repository, role="M6 output root")
    context = validate_campaign_inputs(
        schedule, schedule_path=schedule_path, repo_root=repository
    )
    admission = _validate_admission_evidence(
        evidence_paths=admission_evidence_paths,
        trust_anchor_path=trust_anchor_path,
        trusted_anchor_sha256=trusted_anchor_sha256,
        trust_mode=trust_mode,
        context=context,
        output_root=root,
        now=now,
        repo_root=repository,
    )
    source_commit = admission["source_commit"]
    phases: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for phase, path in (
        ("p3", p3_preflight_path),
        ("p4", p4_preflight_path),
        ("session", session_preflight_path),
    ):
        projection, detail = _preflight_projection(
            phase,
            path,
            source_commit=source_commit,
            pins_identity=context["inputs"]["campaign_protocol_pins"],
            matrix_identity=context["inputs"]["matrix"],
            output_root=root,
            now=now,
            repo_root=repository,
        )
        phases.append(projection)
        details.append(detail)
    if any(
        left["generated"] > right["generated"]
        for left, right in zip(details, details[1:])
    ):
        _fail("M6_LIVE_READINESS_ORDER", "P3, P4, session order differs")
    uuids = {row["gpu_uuid_sha256"] for row in phases}
    ncu_identities = {json.dumps(row["ncu_identity"], sort_keys=True) for row in details}
    if uuids != {admission["gpu_uuid_sha256"]}:
        _fail("M6_LIVE_GPU_UUID_MISMATCH", "preflight and detached authority differ")
    if len(ncu_identities) != 1:
        _fail("M6_LIVE_NCU_AUTHORITY", "preflight NCU identities differ")
    maximum = schedule["cost_gate"]["max_new_profiles_per_resume"]
    source = {
        "commit": source_commit,
        "tree": admission["source_tree"],
        "working_tree_clean": True,
        "p1_gate_receipt_sha256": admission["evidence"]["p1_gate_receipt"][
            "model_sha256"
        ],
        "current_source_anchor_sha256": admission["evidence"][
            "current_source_anchor"
        ]["model_sha256"],
        "clean_s_p2_receipt_sha256": admission["evidence"][
            "clean_s_p2_receipt"
        ]["model_sha256"],
        "live_v1_authority_file_sha256": admission["evidence"][
            "live_v1_authority"
        ]["sha256"],
        "successor_matrix_file_sha256": admission["evidence"][
            "successor_matrix"
        ]["sha256"],
    }
    session_seed = {
        "source": source,
        "schedule_sha256": schedule["schedule_sha256"],
        "trust_anchor_sha256": trusted_anchor_sha256,
        "operator_authority_sha256": admission["operator"]["authority_sha256"],
        "session_preflight_sha256": phases[-1]["sha256"],
        "output_root": str(root),
    }
    session_id = f"m6-live-{_digest(session_seed)[:24]}"
    session_root = root / "m6-live" / session_id
    issued_at_utc = _format_time(now)
    unsigned = {
        "schema_version": 1,
        "kind": "m6_live_session_authority_v1",
        "trust_mode": trust_mode,
        "session_id": session_id,
        "issued_at_utc": issued_at_utc,
        "source": source,
        "admission_prerequisites": {
            "trust_anchor": admission["trust_anchor"],
            "evidence": admission["evidence"],
            "campaign_lock_ownership": admission["campaign_lock"],
            "storage_fault_absent_at_utc": issued_at_utc,
        },
        "schedule": context["schedule"],
        "inputs": context["inputs"],
        "schema_authorities": context["schema_authorities"],
        "readiness_preflights": phases,
        "gpu_uuid_sha256": admission["gpu_uuid_sha256"],
        "driver_version": GENERIC_DRIVER_VERSION,
        "ncu_identity": details[-1]["ncu_identity"],
        "tools": _tool_identities(repository),
        "namespace": {
            "contract": SESSION_NAMESPACE,
            "output_root": str(root),
            "session_root": str(session_root),
            "attempts_root": str(session_root / "attempts"),
            "receipts_root": str(session_root / "receipts"),
            "tier2_root": str(session_root / "tier2"),
        },
        "cost_ceiling": {
            "unit": "fresh_profile_process",
            "max_fresh_processes_per_resume": maximum,
            "source": "schedule.cost_gate.max_new_profiles_per_resume",
        },
        "operator_hardware_authority": admission["operator"],
        "scope": SCOPE,
    }
    authority = _seal(unsigned, "session_authority_sha256")
    validate_schema("session", authority)
    _validate_session_bindings(authority, schedule, context, now=now)
    return authority


def _validate_session_bindings(
    authority: Mapping[str, Any],
    schedule: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    now: datetime,
) -> None:
    _verify_seal(authority, "session_authority_sha256", role="session authority")
    _fresh_at(authority.get("issued_at_utc"), role="session issued_at_utc", now=now)
    if authority.get("schedule") != context["schedule"]:
        _fail("M6_LIVE_SESSION_BINDING", "schedule identity differs")
    if authority.get("inputs") != context["inputs"]:
        _fail("M6_LIVE_SESSION_BINDING", "input identities differ")
    if authority.get("schema_authorities") != context["schema_authorities"]:
        _fail("M6_LIVE_SESSION_BINDING", "schema identities differ")
    source = authority.get("source") or {}
    namespace = authority.get("namespace") or {}
    output_root = Path(namespace.get("output_root", ""))
    session_root = output_root / "m6-live" / str(authority.get("session_id"))
    expected_namespace = {
        "contract": SESSION_NAMESPACE,
        "output_root": str(output_root),
        "session_root": str(session_root),
        "attempts_root": str(session_root / "attempts"),
        "receipts_root": str(session_root / "receipts"),
        "tier2_root": str(session_root / "tier2"),
    }
    if namespace != expected_namespace:
        _fail("M6_LIVE_NAMESPACE_BINDING", "session namespace differs")
    if source.get("working_tree_clean") is not True:
        _fail("M6_LIVE_SOURCE_DIRTY", "source S is not clean")
    if authority.get("driver_version") != GENERIC_DRIVER_VERSION:
        if authority.get("driver_version") == TASK17_ONLY_DRIVER_VERSION:
            _fail("M6_LIVE_TASK17_DRIVER_FORBIDDEN", "session authority")
        _fail("M6_LIVE_DRIVER_AUTHORITY", "generic driver differs")
    maximum = schedule["cost_gate"]["max_new_profiles_per_resume"]
    if authority.get("cost_ceiling") != {
        "unit": "fresh_profile_process",
        "max_fresh_processes_per_resume": maximum,
        "source": "schedule.cost_gate.max_new_profiles_per_resume",
    }:
        _fail("M6_LIVE_COST_CEILING", "cost must derive exactly from schedule")
    preflights = authority.get("readiness_preflights") or []
    if [row.get("phase") for row in preflights] != ["p3", "p4", "session"]:
        _fail("M6_LIVE_READINESS_ORDER", "P3/P4/session evidence differs")
    times = [
        _fresh_at(
            row.get("generated_at_utc"),
            role=row.get("phase", "preflight"),
            now=now,
        )
        for row in preflights
    ]
    if any(left > right for left, right in zip(times, times[1:])):
        _fail("M6_LIVE_READINESS_ORDER", "P3/P4/session timestamps differ")
    for row in preflights:
        if (
            row.get("git_head") != source.get("commit")
            or row.get("gpu_uuid_sha256") != authority.get("gpu_uuid_sha256")
            or row.get("driver_version") != GENERIC_DRIVER_VERSION
            or row.get("output_root") != str(output_root)
            or row.get("capture_admitted") is not True
        ):
            _fail("M6_LIVE_PREFLIGHT_BINDING", row.get("phase", "unknown"))
    prerequisites = authority.get("admission_prerequisites") or {}
    lock = prerequisites.get("campaign_lock_ownership") or {}
    if authority.get("trust_mode") not in {
        FIXTURE_TRUST_MODE,
        PRODUCTION_TRUST_MODE,
    }:
        _fail("M6_LIVE_OPERATOR_TRUST_ANCHOR_INVALID", "session trust mode differs")
    expected_fault = output_root / _STORAGE_FAULT_NAME
    if (
        lock.get("storage_fault_receipt_path") != str(expected_fault)
        or lock.get("operator_authority_sha256")
        != (authority.get("operator_hardware_authority") or {}).get(
            "authority_sha256"
        )
    ):
        _fail("M6_LIVE_CAMPAIGN_LOCK_INVALID", "session lock projection differs")
    if expected_fault.exists() or expected_fault.is_symlink():
        _fail("M6_LIVE_STORAGE_FAULT_PRESENT", str(expected_fault))
    _validate_operator_authority(
        authority.get("operator_hardware_authority"),
        source_commit=source.get("commit"),
        gpu_uuid_sha256=authority.get("gpu_uuid_sha256"),
        output_root=output_root,
        max_fresh_processes_per_resume=maximum,
    )


def validate_session_authority(
    authority: Mapping[str, Any],
    *,
    schedule: Mapping[str, Any],
    schedule_path: Path,
    preflight_paths: Sequence[Path],
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Revalidate at trusted current time under the installed production anchor."""
    trust_path, trust_sha256 = _production_trust_configuration()
    return _validate_session_authority(
        authority,
        schedule=schedule,
        schedule_path=schedule_path,
        preflight_paths=preflight_paths,
        trust_anchor_path=trust_path,
        trusted_anchor_sha256=trust_sha256,
        trust_mode=PRODUCTION_TRUST_MODE,
        trusted_now=_trusted_now(),
        repo_root=repo_root,
    )


def _validate_fixture_session_authority(
    authority: Mapping[str, Any],
    *,
    schedule: Mapping[str, Any],
    schedule_path: Path,
    preflight_paths: Sequence[Path],
    trust_anchor_path: Path,
    trusted_anchor_sha256: str,
    trusted_now: datetime,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    return _validate_session_authority(
        authority,
        schedule=schedule,
        schedule_path=schedule_path,
        preflight_paths=preflight_paths,
        trust_anchor_path=trust_anchor_path,
        trusted_anchor_sha256=trusted_anchor_sha256,
        trust_mode=FIXTURE_TRUST_MODE,
        trusted_now=trusted_now,
        repo_root=repo_root,
    )


def _validate_session_authority(
    authority: Mapping[str, Any],
    *,
    schedule: Mapping[str, Any],
    schedule_path: Path,
    preflight_paths: Sequence[Path],
    trust_anchor_path: Path,
    trusted_anchor_sha256: str,
    trust_mode: str,
    trusted_now: datetime,
    repo_root: Path,
) -> dict[str, Any]:
    validate_schema("session", authority)
    if authority.get("trust_mode") != trust_mode:
        _fail("M6_LIVE_FIXTURE_ONLY", "session trust mode is not admissible")
    now = trusted_now.astimezone(timezone.utc)
    context = validate_campaign_inputs(
        schedule, schedule_path=schedule_path, repo_root=repo_root
    )
    if len(preflight_paths) != 3:
        _fail("M6_LIVE_PREFLIGHT_COUNT", "exactly P3, P4, session are required")
    prerequisites = authority.get("admission_prerequisites") or {}
    retained_anchor = prerequisites.get("trust_anchor") or {}
    if (
        retained_anchor.get("path") != str(trust_anchor_path)
        or retained_anchor.get("sha256") != trusted_anchor_sha256
    ):
        _fail("M6_LIVE_OPERATOR_TRUST_ANCHOR_MISMATCH", "session projection differs")
    retained_evidence = prerequisites.get("evidence") or {}
    if type(retained_evidence) is not dict or set(retained_evidence) != set(
        _EVIDENCE_NAMES
    ):
        _fail("M6_LIVE_PREREQUISITE_SET", "session evidence projection differs")
    evidence_paths = {
        name: Path(retained_evidence[name]["path"]) for name in _EVIDENCE_NAMES
    }
    output_root = Path((authority.get("namespace") or {}).get("output_root", ""))
    admission = _validate_admission_evidence(
        evidence_paths=evidence_paths,
        trust_anchor_path=trust_anchor_path,
        trusted_anchor_sha256=trusted_anchor_sha256,
        trust_mode=trust_mode,
        context=context,
        output_root=output_root,
        now=now,
        repo_root=repo_root,
    )
    expected_source = {
        "commit": admission["source_commit"],
        "tree": admission["source_tree"],
        "working_tree_clean": True,
        "p1_gate_receipt_sha256": admission["evidence"]["p1_gate_receipt"][
            "model_sha256"
        ],
        "current_source_anchor_sha256": admission["evidence"][
            "current_source_anchor"
        ]["model_sha256"],
        "clean_s_p2_receipt_sha256": admission["evidence"][
            "clean_s_p2_receipt"
        ]["model_sha256"],
        "live_v1_authority_file_sha256": admission["evidence"][
            "live_v1_authority"
        ]["sha256"],
        "successor_matrix_file_sha256": admission["evidence"][
            "successor_matrix"
        ]["sha256"],
    }
    if (
        prerequisites.get("trust_anchor") != admission["trust_anchor"]
        or retained_evidence != admission["evidence"]
        or prerequisites.get("campaign_lock_ownership")
        != admission["campaign_lock"]
        or authority.get("operator_hardware_authority") != admission["operator"]
        or authority.get("source") != expected_source
    ):
        _fail("M6_LIVE_PREREQUISITE_BINDING", "detached evidence changed")
    _fresh_at(
        prerequisites.get("storage_fault_absent_at_utc"),
        role="storage-fault absence check",
        now=now,
    )
    source_commit = admission["source_commit"]
    observed = []
    ncu_identities = []
    for phase, path in zip(("p3", "p4", "session"), preflight_paths, strict=True):
        projection, detail = _preflight_projection(
            phase,
            path,
            source_commit=source_commit,
            pins_identity=context["inputs"]["campaign_protocol_pins"],
            matrix_identity=context["inputs"]["matrix"],
            output_root=output_root,
            now=now,
            repo_root=repo_root,
        )
        observed.append(projection)
        ncu_identities.append(detail["ncu_identity"])
    if authority.get("readiness_preflights") != observed:
        _fail("M6_LIVE_PREFLIGHT_BINDING", "retained preflight bytes differ")
    if any(identity != authority.get("ncu_identity") for identity in ncu_identities):
        _fail("M6_LIVE_NCU_AUTHORITY", "retained preflight NCU differs")
    if authority.get("tools") != _tool_identities(repo_root):
        _fail("M6_LIVE_TOOL_IDENTITY", "session tool identity differs")
    _validate_session_bindings(authority, schedule, context, now=now)
    return context


def _active_slots(
    schedule: Mapping[str, Any], activation: Mapping[str, Any] | None
) -> list[Mapping[str, Any]]:
    slots = list(schedule["slots"])
    if activation is None:
        return slots
    validate_schema("tier2", activation)
    _verify_seal(activation, "activation_sha256", role="Tier-2 activation")
    if activation.get("schedule_sha256") != schedule.get("schedule_sha256"):
        _fail("M6_LIVE_TIER2_BINDING", "schedule differs")
    activated = set(activation.get("activated_cell_ids", ()))
    candidates = [
        slot for slot in schedule["tier2_candidates"]
        if slot["cell_id"] in activated
    ]
    if [slot["ordinal"] for slot in candidates] != activation.get("activated_ordinals"):
        _fail("M6_LIVE_TIER2_BINDING", "activated ordinal projection differs")
    if _digest(candidates) != activation.get("activated_candidate_slots_sha256"):
        _fail("M6_LIVE_TIER2_BINDING", "activated slot identity differs")
    return slots + candidates


def _validate_plan_seal(plan: Mapping[str, Any]) -> None:
    validate_schema("plan", plan)
    _verify_seal(plan, "plan_sha256", role="slot plan")
    planned = _parse_time(plan.get("planned_at_utc"), role="plan planned_at_utc")
    expires = _parse_time(plan.get("expires_at_utc"), role="plan expires_at_utc")
    if expires != planned + timedelta(seconds=FRESHNESS_SECONDS):
        _fail("M6_LIVE_PLAN_EXPIRY", "plan validity interval differs")


_ED_P = 2**255 - 19
_ED_L = 2**252 + 27742317777372353535851937790883648493
_ED_D = (-121665 * pow(121666, _ED_P - 2, _ED_P)) % _ED_P
_ED_I = pow(2, (_ED_P - 1) // 4, _ED_P)


def _ed_xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_ED_D * y * y + 1, _ED_P - 2, _ED_P) % _ED_P
    x = pow(xx, (_ED_P + 3) // 8, _ED_P)
    if (x * x - xx) % _ED_P:
        x = x * _ED_I % _ED_P
    if x & 1:
        x = _ED_P - x
    return x


_ED_BY = 4 * pow(5, _ED_P - 2, _ED_P) % _ED_P
_ED_BX = _ed_xrecover(_ED_BY)
_ED_B = (_ED_BX, _ED_BY, 1, _ED_BX * _ED_BY % _ED_P)
_ED_IDENTITY = (0, 1, 1, 0)


def _ed_add(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = left
    x2, y2, z2, t2 = right
    a = (y1 - x1) * (y2 - x2) % _ED_P
    b = (y1 + x1) * (y2 + x2) % _ED_P
    c = 2 * _ED_D * t1 * t2 % _ED_P
    d = 2 * z1 * z2 % _ED_P
    e = b - a
    f = d - c
    g = d + c
    h = b + a
    return e * f % _ED_P, g * h % _ED_P, f * g % _ED_P, e * h % _ED_P


def _ed_double(point: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, z, _t = point
    a = x * x % _ED_P
    b = y * y % _ED_P
    c = 2 * z * z % _ED_P
    d = -a
    e = (x + y) * (x + y) - a - b
    g = d + b
    f = g - c
    h = d - b
    return e * f % _ED_P, g * h % _ED_P, f * g % _ED_P, e * h % _ED_P


def _ed_scalar(
    point: tuple[int, int, int, int], scalar: int
) -> tuple[int, int, int, int]:
    result = _ED_IDENTITY
    addend = point
    while scalar:
        if scalar & 1:
            result = _ed_add(result, addend)
        addend = _ed_double(addend)
        scalar >>= 1
    return result


def _ed_equal(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> bool:
    return (
        (left[0] * right[2] - right[0] * left[2]) % _ED_P == 0
        and (left[1] * right[2] - right[1] * left[2]) % _ED_P == 0
    )


def _ed_decode(encoded: bytes) -> tuple[int, int, int, int]:
    if len(encoded) != 32:
        raise ValueError("encoded point length differs")
    value = int.from_bytes(encoded, "little")
    sign = value >> 255
    y = value & ((1 << 255) - 1)
    if y >= _ED_P:
        raise ValueError("noncanonical encoded point")
    x = _ed_xrecover(y)
    if (-x * x + y * y - 1 - _ED_D * x * x * y * y) % _ED_P:
        raise ValueError("point is not on the Ed25519 curve")
    if (x & 1) != sign:
        x = _ED_P - x
    point = (x, y, 1, x * y % _ED_P)
    if _ed_equal(point, _ED_IDENTITY) or not _ed_equal(
        _ed_scalar(point, _ED_L), _ED_IDENTITY
    ):
        raise ValueError("point is not in the prime-order subgroup")
    return point


def _verify_ed25519(public_key_hex: Any, signature_hex: Any, message: bytes) -> None:
    try:
        if type(public_key_hex) is not str or type(signature_hex) is not str:
            raise ValueError("hex strings required")
        public_key = bytes.fromhex(public_key_hex)
        signature = bytes.fromhex(signature_hex)
        if len(public_key) != 32 or len(signature) != 64:
            raise ValueError("key or signature length differs")
        public_point = _ed_decode(public_key)
        r_encoded = signature[:32]
        r_point = _ed_decode(r_encoded)
        scalar = int.from_bytes(signature[32:], "little")
        if scalar >= _ED_L:
            raise ValueError("noncanonical signature scalar")
        challenge = int.from_bytes(
            hashlib.sha512(r_encoded + public_key + message).digest(), "little"
        ) % _ED_L
        if not _ed_equal(
            _ed_scalar(_ED_B, scalar),
            _ed_add(r_point, _ed_scalar(public_point, challenge)),
        ):
            raise ValueError("signature equation differs")
    except ValueError as exc:
        _fail("M6_LIVE_SUPERVISOR_AUTHENTICATION", str(exc))


def _load_supervisor_registry() -> tuple[dict[str, Any], dict[str, Any]]:
    # Deliberately source-local: callers cannot install a production pin at runtime.
    pin: tuple[str, str] | None = None
    if pin is None:
        _fail(
            "M6_LIVE_SUPERVISOR_AUTHORITY_UNAVAILABLE",
            "no immutable external supervisor registry is pinned",
        )
    path_text, expected_sha256 = pin
    if not _is_sha256(expected_sha256):
        _fail("M6_LIVE_SUPERVISOR_AUTHORITY_INVALID", "registry pin differs")
    path = _canonical_absolute(
        Path(path_text), role="supervisor authority registry", must_exist=True
    )
    _outside_repository(
        path, repo_root=REPO_ROOT, role="supervisor authority registry"
    )
    registry, payload = _load_json_path(path, role="supervisor authority registry")
    if immutable.sha256_bytes(payload) != expected_sha256:
        _fail("M6_LIVE_SUPERVISOR_AUTHORITY_INVALID", "registry bytes differ")
    required = {
        "schema_version",
        "kind",
        "status",
        "authorities",
        "registry_sha256",
    }
    if set(registry) != required:
        _fail("M6_LIVE_SUPERVISOR_AUTHORITY_INVALID", "registry fields differ")
    _verify_seal(registry, "registry_sha256", role="supervisor authority registry")
    authorities = registry.get("authorities")
    if (
        registry.get("schema_version") != 1
        or registry.get("kind") != "m6_supervisor_authority_registry_v1"
        or registry.get("status") != "trusted"
        or not isinstance(authorities, list)
        or not authorities
    ):
        _fail("M6_LIVE_SUPERVISOR_AUTHORITY_INVALID", "registry semantics differ")
    return registry, {
        "path": str(path),
        "sha256": expected_sha256,
        "size_bytes": len(payload),
        "registry_sha256": registry["registry_sha256"],
    }


def _supervisor_authority(
    registry: Mapping[str, Any], authority_id: Any
) -> Mapping[str, Any]:
    fields = {
        "authority_id",
        "producer_path",
        "producer_sha256",
        "producer_size_bytes",
        "ed25519_public_key",
        "launcher_interpreter_path",
        "launcher_interpreter_sha256",
        "scope",
    }
    matches = [
        row
        for row in registry.get("authorities", ())
        if isinstance(row, Mapping) and row.get("authority_id") == authority_id
    ]
    if len(matches) != 1 or set(matches[0]) != fields:
        _fail("M6_LIVE_SUPERVISOR_AUTHORITY_INVALID", "authority entry differs")
    entry = matches[0]
    interpreter_path = entry.get("launcher_interpreter_path")
    interpreter_sha256 = entry.get("launcher_interpreter_sha256")
    interpreter_valid = (
        interpreter_path is None and interpreter_sha256 is None
    ) or (
        type(interpreter_path) is str
        and interpreter_path.startswith("/")
        and _is_sha256(interpreter_sha256)
    )
    if (
        entry.get("scope") != "m6_exact_direct_child_process_receipt_v1"
        or not _is_sha256(entry.get("producer_sha256"))
        or type(entry.get("producer_size_bytes")) is not int
        or entry["producer_size_bytes"] < 1
        or type(entry.get("ed25519_public_key")) is not str
        or len(entry["ed25519_public_key"]) != 64
        or not interpreter_valid
    ):
        _fail("M6_LIVE_SUPERVISOR_AUTHORITY_INVALID", "authority semantics differ")
    return entry


def _supervisor_receipt_fields(
    plan: Mapping[str, Any],
    session: Mapping[str, Any] | None = None,
    registry_identity: Mapping[str, Any] | None = None,
    authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version": 1,
        "kind": "m6_exact_child_supervisor_receipt_v1",
        "relationship": "exact_direct_child",
        "plan_sha256": plan["plan_sha256"],
        "launcher_executable": plan["launcher"]["executable"],
        "launcher_executable_sha256": plan["launcher"]["executable_sha256"],
        "launcher_argv": plan["command"],
        "target_executable": plan["target"]["executable"],
        "target_executable_sha256": plan["target"]["executable_sha256"],
        "target_argv": plan["target"]["argv"],
        "working_directory": plan["target"]["working_directory"],
    }
    if session is not None:
        prerequisites = session.get("admission_prerequisites") or {}
        lock = prerequisites.get("campaign_lock_ownership") or {}
        fields.update(
            {
                "session_authority_sha256": session.get("session_authority_sha256"),
                "session_preflight_sha256": (session.get("readiness_preflights") or [{}])[
                    -1
                ].get("sha256"),
                "trust_anchor_sha256": (prerequisites.get("trust_anchor") or {}).get(
                    "sha256"
                ),
                "campaign_lock_receipt_sha256": lock.get("receipt_sha256"),
            }
        )
    if authority is not None and registry_identity is not None:
        fields.update(
            {
                "supervisor_authority_id": authority.get("authority_id"),
                "supervisor_registry_sha256": registry_identity.get("sha256"),
                "producer_executable": authority.get("producer_path"),
                "producer_executable_sha256": authority.get("producer_sha256"),
            }
        )
    return fields


def _revalidate_production_session_at_start(
    session: Mapping[str, Any], *, started_at: datetime
) -> None:
    trust_path, trust_sha256 = _production_trust_configuration()
    schedule_path = Path((session.get("schedule") or {}).get("path", ""))
    schedule, _ = _load_json_path(schedule_path, role="retained M6 schedule")
    preflight_paths = [
        Path(row.get("path", "")) for row in session.get("readiness_preflights", ())
    ]
    _validate_session_authority(
        session,
        schedule=schedule,
        schedule_path=schedule_path,
        preflight_paths=preflight_paths,
        trust_anchor_path=trust_path,
        trusted_anchor_sha256=trust_sha256,
        trust_mode=PRODUCTION_TRUST_MODE,
        trusted_now=started_at,
        repo_root=REPO_ROOT,
    )


def _require_immutable_supervisor_receipt(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        _fail("M6_LIVE_SUPERVISOR_RECEIPT", str(exc))
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
    ):
        _fail("M6_LIVE_SUPERVISOR_RECEIPT", "immutable regular receipt required")


def _authenticate_supervisor_receipt(
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
    registry, registry_identity = _load_supervisor_registry()
    authority = _supervisor_authority(
        registry, receipt.get("supervisor_authority_id")
    )
    producer_path = _canonical_absolute(
        Path(authority["producer_path"]), role="supervisor producer", must_exist=True
    )
    producer = _read_regular(producer_path, role="supervisor producer")
    if (
        immutable.sha256_bytes(producer) != authority.get("producer_sha256")
        or len(producer) != authority.get("producer_size_bytes")
    ):
        _fail("M6_LIVE_SUPERVISOR_PRODUCER", "pinned producer bytes differ")
    authentication_model = dict(receipt)
    signature = authentication_model.pop("producer_signature_ed25519", None)
    authentication_model.pop("receipt_sha256", None)
    _verify_ed25519(
        authority.get("ed25519_public_key"),
        signature,
        immutable.canonical_json_bytes(authentication_model),
    )
    return registry_identity, authority, authentication_model


def _proc_process_identity(pid: int) -> tuple[int, int]:
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = payload.rfind(")")
        fields = payload[closing + 2 :].split()
        return int(fields[1]), int(fields[19])
    except (OSError, ValueError, IndexError) as exc:
        _fail("M6_LIVE_PROCESS_OS_IDENTITY", str(exc))


def _expected_live_child(
    plan: Mapping[str, Any], authority: Mapping[str, Any]
) -> tuple[str, list[str]]:
    launcher = _canonical_absolute(
        Path(plan["launcher"]["executable"]), role="planned launcher", must_exist=True
    )
    payload = _read_regular(launcher, role="planned launcher")
    if immutable.sha256_bytes(payload) != plan["launcher"]["executable_sha256"]:
        _fail("M6_LIVE_PROCESS_OS_IDENTITY", "launcher bytes differ")
    first_line = payload.splitlines()[0] if payload else b""
    interpreter_path = authority.get("launcher_interpreter_path")
    interpreter_sha256 = authority.get("launcher_interpreter_sha256")
    if not first_line.startswith(b"#!"):
        if interpreter_path is not None or interpreter_sha256 is not None:
            _fail("M6_LIVE_PROCESS_OS_IDENTITY", "unexpected interpreter authority")
        return str(launcher), list(plan["command"])
    try:
        declaration = os.fsdecode(first_line[2:]).strip().split(maxsplit=1)
        interpreter = _canonical_absolute(
            Path(interpreter_path), role="pinned launcher interpreter", must_exist=True
        )
        interpreter_payload = _read_regular(interpreter, role="launcher interpreter")
    except (IndexError, OSError, TypeError) as exc:
        _fail("M6_LIVE_PROCESS_OS_IDENTITY", f"launcher shebang: {exc}")
    if immutable.sha256_bytes(interpreter_payload) != interpreter_sha256:
        _fail("M6_LIVE_PROCESS_OS_IDENTITY", "interpreter bytes differ")
    declared = Path(declaration[0])
    if declared.name == "env" and len(declaration) == 2:
        if interpreter.name != declaration[1]:
            _fail("M6_LIVE_PROCESS_OS_IDENTITY", "env interpreter differs")
        argv = [declaration[1], str(launcher), *plan["command"][1:]]
    else:
        if interpreter != declared:
            _fail("M6_LIVE_PROCESS_OS_IDENTITY", "direct interpreter differs")
        argv = [str(interpreter)]
        if len(declaration) == 2:
            argv.append(declaration[1])
        argv.extend((str(launcher), *plan["command"][1:]))
    return str(interpreter), argv


def _verify_live_exact_child(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    supervisor_pid = receipt.get("supervisor_pid")
    child_pid = receipt.get("pid")
    if supervisor_pid != os.getpid() or type(child_pid) is not int:
        _fail("M6_LIVE_PROCESS_OS_IDENTITY", "supervisor/child PID differs")
    _supervisor_parent, supervisor_ticks = _proc_process_identity(supervisor_pid)
    if supervisor_ticks != receipt.get("supervisor_start_ticks"):
        _fail("M6_LIVE_PROCESS_OS_IDENTITY", "supervisor start identity differs")
    try:
        supervisor_executable = Path("/proc/self/exe").resolve(strict=True)
        supervisor_argv_bytes = Path("/proc/self/cmdline").read_bytes().split(b"\0")
    except OSError as exc:
        _fail("M6_LIVE_PROCESS_OS_IDENTITY", str(exc))
    supervisor_argv = [os.fsdecode(value) for value in supervisor_argv_bytes if value]
    if (
        str(supervisor_executable) != receipt.get("observed_supervisor_executable")
        or supervisor_argv != receipt.get("observed_supervisor_argv")
        or authority["producer_path"] not in supervisor_argv
    ):
        _fail("M6_LIVE_SUPERVISOR_PRODUCER", "live producer process differs")
    pidfd: int | None = None
    try:
        if hasattr(os, "pidfd_open"):
            pidfd = os.pidfd_open(child_pid)
        child_parent, child_ticks = _proc_process_identity(child_pid)
        if (
            child_parent != supervisor_pid
            or child_ticks != receipt.get("start_ticks")
        ):
            _fail("M6_LIVE_PROCESS_OS_IDENTITY", "exact child identity differs")
        executable = Path(f"/proc/{child_pid}/exe").resolve(strict=True)
        working_directory = Path(f"/proc/{child_pid}/cwd").resolve(strict=True)
        argv_bytes = Path(f"/proc/{child_pid}/cmdline").read_bytes().split(b"\0")
    except OSError as exc:
        _fail("M6_LIVE_PROCESS_OS_IDENTITY", str(exc))
    finally:
        if pidfd is not None:
            os.close(pidfd)
    argv = [os.fsdecode(value) for value in argv_bytes if value]
    expected_executable, expected_argv = _expected_live_child(plan, authority)
    if (
        str(executable) != receipt.get("observed_child_executable")
        or str(executable) != expected_executable
        or argv != receipt.get("observed_child_argv")
        or argv != expected_argv
        or str(working_directory) != plan["target"]["working_directory"]
    ):
        _fail("M6_LIVE_PROCESS_OS_IDENTITY", "executable/argv/cwd differs")


def _validate_process_identity(
    value: Any,
    *,
    plan: Mapping[str, Any],
    started_at_utc: str | None,
    session: Mapping[str, Any] | None = None,
    allow_fixture: bool = True,
) -> str | None:
    if value is None:
        if started_at_utc is not None:
            _fail("M6_LIVE_PROCESS_IDENTITY", "start time lacks process identity")
        return None
    if not isinstance(value, Mapping):
        _fail("M6_LIVE_PROCESS_IDENTITY", "exact process identity required")
    _verify_seal(value, "identity_sha256", role="process identity")
    expected = {
        "started_at_utc": started_at_utc,
        "plan_sha256": plan["plan_sha256"],
        "launcher_executable": plan["launcher"]["executable"],
        "launcher_executable_sha256": plan["launcher"]["executable_sha256"],
        "launcher_argv": plan["command"],
        "target_executable": plan["target"]["executable"],
        "target_executable_sha256": plan["target"]["executable_sha256"],
        "target_argv": plan["target"]["argv"],
        "working_directory": plan["target"]["working_directory"],
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        _fail("M6_LIVE_PROCESS_IDENTITY", "plan/process binding differs")
    started = _parse_time(started_at_utc, role="process started_at_utc")
    if not (
        _parse_time(plan["planned_at_utc"], role="plan planned_at_utc")
        <= started
        <= _parse_time(plan["expires_at_utc"], role="plan expires_at_utc")
    ):
        _fail("M6_LIVE_PLAN_EXPIRED", "process start falls outside plan validity")
    if type(value.get("pid")) is not int or value["pid"] < 1:
        _fail("M6_LIVE_PROCESS_IDENTITY", "PID is invalid")
    if type(value.get("start_ticks")) is not int or value["start_ticks"] < 1:
        _fail("M6_LIVE_PROCESS_IDENTITY", "start identity is invalid")
    mode = value.get("authority_mode")
    receipt_ref = value.get("supervisor_receipt")
    if mode == FIXTURE_TRUST_MODE and allow_fixture and receipt_ref is None:
        return value["identity_sha256"]
    if mode != "retained_exact_child_receipt" or not isinstance(receipt_ref, Mapping):
        _fail("M6_LIVE_PROCESS_IDENTITY", "retained supervisor receipt required")
    path = _canonical_absolute(
        Path(receipt_ref.get("path", "")), role="supervisor receipt", must_exist=True
    )
    _require_immutable_supervisor_receipt(path)
    root = _canonical_absolute(Path(plan["output"]["attempt_root"]), role="attempt root")
    try:
        path.relative_to(root)
    except ValueError:
        _fail("M6_LIVE_PROCESS_IDENTITY", "supervisor receipt is outside attempt root")
    receipt, payload = _load_json_path(path, role="supervisor receipt")
    if (
        immutable.sha256_bytes(payload) != receipt_ref.get("sha256")
        or len(payload) != receipt_ref.get("size_bytes")
        or receipt.get("receipt_sha256") != receipt_ref.get("receipt_sha256")
    ):
        _fail("M6_LIVE_PROCESS_IDENTITY", "supervisor receipt bytes differ")
    _verify_seal(receipt, "receipt_sha256", role="supervisor receipt")
    if session is None or session.get("trust_mode") != PRODUCTION_TRUST_MODE:
        _fail("M6_LIVE_PROCESS_IDENTITY", "production session authority required")
    registry_identity, authority, _authentication_model = (
        _authenticate_supervisor_receipt(receipt)
    )
    fixed = _supervisor_receipt_fields(
        plan,
        session,
        registry_identity=registry_identity,
        authority=authority,
    )
    dynamic = {
        "pid",
        "start_ticks",
        "started_at_utc",
        "supervisor_pid",
        "supervisor_start_ticks",
        "observed_supervisor_executable",
        "observed_supervisor_argv",
        "observed_child_executable",
        "observed_child_argv",
        "authority_validated_at_utc",
        "storage_fault_absent_at_utc",
        "producer_signature_ed25519",
        "receipt_sha256",
    }
    if set(receipt) != set(fixed) | dynamic:
        _fail("M6_LIVE_PROCESS_IDENTITY", "supervisor receipt fields differ")
    if any(receipt.get(name) != expected for name, expected in fixed.items()):
        _fail("M6_LIVE_PROCESS_IDENTITY", "supervisor receipt plan binding differs")
    identity_authority = {
        "supervisor_authority_id": authority["authority_id"],
        "supervisor_registry_sha256": registry_identity["sha256"],
        "producer_executable": authority["producer_path"],
        "producer_executable_sha256": authority["producer_sha256"],
    }
    if any(value.get(name) != expected for name, expected in identity_authority.items()):
        _fail("M6_LIVE_PROCESS_IDENTITY", "supervisor authority binding differs")
    child_fields = ("pid", "start_ticks", "started_at_utc")
    if any(receipt.get(name) != value.get(name) for name in child_fields):
        _fail("M6_LIVE_PROCESS_IDENTITY", "supervisor child identity differs")
    if any(
        receipt.get(name) != started_at_utc
        for name in ("authority_validated_at_utc", "storage_fault_absent_at_utc")
    ):
        _fail("M6_LIVE_PROCESS_IDENTITY", "process-start authority time differs")
    return value["identity_sha256"]


def _artifact_path(root: Path, relative_path: str, *, role: str) -> Path:
    try:
        safe = immutable.safe_relative_path(relative_path)
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_LIVE_ATTEMPT_ARTIFACT", f"{role}: {exc}")
    path = root / safe
    _canonical_absolute(path, role=role, must_exist=True)
    return path


def _validate_attempt_evidence(
    attempt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    session: Mapping[str, Any],
    rehash_artifacts: bool,
) -> None:
    validate_schema("attempt", attempt)
    _verify_seal(attempt, "receipt_sha256", role="attempt receipt")
    _validate_plan_seal(plan)
    if (
        attempt.get("plan_sha256") != plan.get("plan_sha256")
        or attempt.get("attempt_sequence") != plan.get("attempt_sequence")
        or attempt.get("resume_epoch") != plan.get("resume_epoch")
        or attempt.get("slot_attempt_index") != plan.get("slot_attempt_index")
        or attempt.get("ordinal") != plan.get("ordinal")
        or attempt.get("slot_id") != plan.get("slot_id")
        or attempt.get("slot_sha256") != plan.get("slot_sha256")
    ):
        _fail("M6_LIVE_PLAN_BINDING", "attempt does not rederive from plan")
    process_sha256 = _validate_process_identity(
        attempt.get("process_identity"),
        plan=plan,
        started_at_utc=attempt.get("started_at_utc"),
        session=session,
        allow_fixture=session.get("trust_mode") == FIXTURE_TRUST_MODE,
    )
    expected_attempt_id = "m6-attempt-" + _digest([
        session["session_authority_sha256"],
        attempt["resume_epoch"],
        attempt["attempt_sequence"],
        attempt["slot_id"],
        attempt["slot_attempt_index"],
        plan["plan_sha256"],
        process_sha256 or ZERO_SHA256,
    ])[:24]
    if attempt.get("attempt_id") != expected_attempt_id:
        _fail("M6_LIVE_ATTEMPT_ID", "attempt identity does not rederive")
    if not rehash_artifacts:
        return
    root = _canonical_absolute(
        Path(plan["output"]["attempt_root"]), role="attempt root"
    )
    artifacts = attempt.get("artifacts") or []
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    for artifact in artifacts:
        relative = artifact["relative_path"]
        if relative in seen_paths or artifact["role"] in seen_roles:
            _fail("M6_LIVE_ATTEMPT_ARTIFACT", "duplicate role or path")
        seen_paths.add(relative)
        seen_roles.add(artifact["role"])
        path = _artifact_path(root, relative, role=artifact["role"])
        payload = _read_regular(path, role=artifact["role"])
        if (
            immutable.sha256_bytes(payload) != artifact["sha256"]
            or len(payload) != artifact["size_bytes"]
        ):
            _fail("M6_LIVE_ATTEMPT_ARTIFACT", f"{artifact['role']}: bytes differ")
    receipt_ref = attempt.get("collection_receipt")
    if receipt_ref is None:
        return
    receipt_path = _artifact_path(
        root, receipt_ref["relative_path"], role="NCU collection receipt"
    )
    receipt_payload = _read_regular(receipt_path, role="NCU collection receipt")
    if (
        immutable.sha256_bytes(receipt_payload) != receipt_ref["sha256"]
        or len(receipt_payload) != receipt_ref["size_bytes"]
    ):
        _fail("M6_LIVE_COLLECTION_RECEIPT", "receipt bytes differ")
    try:
        receipt = immutable.strict_json_loads(
            receipt_payload, label="M6 NCU collection receipt"
        )
        ncu_collection_receipt.validate_schema(receipt)
        immutable.verify_model_seal(receipt, "receipt_seal_sha256")
    except (
        immutable.ImmutableArtifactError,
        ncu_collection_receipt.CollectionReceiptError,
    ) as exc:
        _fail("M6_LIVE_COLLECTION_RECEIPT", str(exc))
    expected = {
        "preflight_sha256": session["readiness_preflights"][-1]["sha256"],
        "ncu_identity": session["ncu_identity"],
        "command": plan["command"],
        "command_sha256": _digest(plan["command"]),
        "working_directory": plan["target"]["working_directory"],
        "exit_status": 0,
        "artifacts": artifacts,
    }
    if (
        receipt_ref["receipt_seal_sha256"] != receipt.get("receipt_seal_sha256")
        or any(receipt.get(name) != value for name, value in expected.items())
    ):
        _fail("M6_LIVE_COLLECTION_RECEIPT", "external bindings differ")


def _attempt_ref(document: Mapping[str, Any], relative_path: str) -> dict[str, Any]:
    try:
        immutable.safe_relative_path(relative_path)
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_LIVE_RECEIPT_PATH", str(exc))
    return {
        "attempt_sequence": document["attempt_sequence"],
        "resume_epoch": document["resume_epoch"],
        "relative_path": relative_path,
        "receipt_sha256": document["receipt_sha256"],
        "plan_relative_path": f"receipts/plan-{document['attempt_sequence']:06d}.json",
        "plan_sha256": document["plan_sha256"],
        "attempt_id": document["attempt_id"],
        "process_identity_sha256": (
            document["process_identity"]["identity_sha256"]
            if document["process_identity"] is not None
            else None
        ),
        "ordinal": document["ordinal"],
        "slot_id": document["slot_id"],
        "outcome": document["outcome"],
        "slot_terminal": document["slot_terminal"],
        "fresh_process_cost": document["fresh_process_cost"],
    }


def _derive_ledger_state(
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    activation: Mapping[str, Any] | None,
    resume_epoch: int,
) -> dict[str, Any]:
    active = _active_slots(schedule, activation)
    terminal_ids: list[str] = []
    slot_attempts: Counter[str] = Counter()
    epoch_costs: Counter[int] = Counter()
    process_keys: set[tuple[int, int]] = set()
    process_hashes: list[str] = []
    previous = ZERO_SHA256
    pre_failures = 0
    post_failures = 0
    terminal_index = 0
    previous_epoch = 0
    for sequence, attempt in enumerate(attempts):
        validate_schema("attempt", attempt)
        _verify_seal(attempt, "receipt_sha256", role=f"attempt {sequence}")
        attempt_epoch = attempt.get("resume_epoch")
        if (
            attempt.get("attempt_sequence") != sequence
            or type(attempt_epoch) is not int
            or attempt_epoch < previous_epoch
            or attempt_epoch > resume_epoch
            or attempt_epoch > previous_epoch + 1
        ):
            _fail("M6_LIVE_RESUME_EPOCH", f"attempt {sequence}")
        previous_epoch = attempt_epoch
        if attempt.get("previous_attempt_receipt_sha256") != previous:
            _fail("M6_LIVE_ATTEMPT_CHAIN", str(sequence))
        previous = attempt["receipt_sha256"]
        if terminal_index >= len(active):
            _fail("M6_LIVE_EXACT_PREFIX", "attempt follows completed active study")
        slot = active[terminal_index]
        expected_activation_sha256 = (
            None
            if terminal_index < len(schedule["slots"])
            else activation["activation_sha256"]
        )
        if (
            attempt.get("ordinal") != slot["ordinal"]
            or attempt.get("slot_id") != slot["slot_id"]
            or attempt.get("slot_sha256") != slot["slot_sha256"]
            or attempt.get("session_authority_sha256")
            != session.get("session_authority_sha256")
            or attempt.get("schedule_sha256") != schedule.get("schedule_sha256")
            or attempt.get("tier2_activation_sha256") != expected_activation_sha256
            or attempt.get("slot_attempt_index") != slot_attempts[slot["slot_id"]]
        ):
            _fail("M6_LIVE_EXACT_PREFIX", f"attempt {sequence} targets wrong slot")
        expected_model = _OUTCOME_MODEL.get(attempt.get("outcome"))
        if expected_model is None or any(
            attempt.get(name) != value for name, value in expected_model.items()
        ):
            _fail("M6_LIVE_ATTEMPT_OUTCOME", str(sequence))
        process = attempt.get("process_identity")
        if attempt["process_started"]:
            if not isinstance(process, Mapping):
                _fail("M6_LIVE_PROCESS_IDENTITY", f"attempt {sequence}")
            key = (process["pid"], process["start_ticks"])
            identity = process["identity_sha256"]
            if key in process_keys or identity in process_hashes:
                _fail("M6_LIVE_PROCESS_REUSE", f"attempt {sequence}")
            process_keys.add(key)
            process_hashes.append(identity)
        elif process is not None:
            _fail("M6_LIVE_PROCESS_IDENTITY", f"attempt {sequence}")
        slot_attempts[slot["slot_id"]] += 1
        epoch_costs[attempt_epoch] += attempt["fresh_process_cost"]
        if attempt["outcome"] == "pre_start_failure":
            pre_failures += 1
        elif attempt["outcome"] == "post_start_failure":
            post_failures += 1
        if attempt["slot_terminal"]:
            terminal_ids.append(slot["slot_id"])
            terminal_index += 1
    ceiling = session["cost_ceiling"]["max_fresh_processes_per_resume"]
    over = [epoch for epoch, count in epoch_costs.items() if count > ceiling]
    if over:
        _fail("M6_LIVE_COST_CEILING", f"resume epochs exceed ceiling: {over}")
    tier1_complete = terminal_index >= len(schedule["slots"])
    study_complete = terminal_index == len(active) and (
        activation is not None or not tier1_complete
    )
    next_ordinal = active[terminal_index]["ordinal"] if terminal_index < len(active) else None
    lifetime_fresh = sum(epoch_costs.values())
    current_fresh = epoch_costs[resume_epoch]
    return {
        "active_slots": len(active),
        "terminal_prefix_slots": terminal_index,
        "next_ordinal": next_ordinal,
        "tier1_complete": tier1_complete,
        "study_complete": study_complete,
        "terminal_slot_ids_sha256": _digest(terminal_ids),
        "lifetime_fresh_processes_started": lifetime_fresh,
        "resume_epoch_fresh_processes_started": current_fresh,
        "resume_epoch_fresh_processes_remaining": ceiling - current_fresh,
        "process_identities_sha256": _digest(process_hashes),
        "pre_start_failures": pre_failures,
        "post_start_failures": post_failures,
    }


def build_initial_ledger_receipt(
    session: Mapping[str, Any], schedule: Mapping[str, Any]
) -> dict[str, Any]:
    validate_schema("session", session)
    if session.get("schedule", {}).get("schedule_sha256") != schedule.get("schedule_sha256"):
        _fail("M6_LIVE_LEDGER_BINDING", "session and schedule differ")
    state = _derive_ledger_state(session, schedule, (), None, 0)
    unsigned = {
        "schema_version": 1,
        "kind": "m6_live_ledger_receipt_v1",
        "checkpoint_kind": "initial",
        "ledger_sequence": 0,
        "resume_epoch": 0,
        "previous_ledger_receipt_sha256": ZERO_SHA256,
        "ledger_history_sha256s": [],
        "session_authority_sha256": session["session_authority_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "tier2_activation_sha256": None,
        "attempt_receipts": [],
        "attempt_head_sha256": ZERO_SHA256,
        "state": state,
    }
    result = _seal(unsigned, "ledger_receipt_sha256")
    validate_schema("ledger", result)
    return result


def _ledger_transition_valid(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    previous_refs = previous.get("attempt_receipts", [])
    current_refs = current.get("attempt_receipts", [])
    if current_refs[: len(previous_refs)] != previous_refs:
        return False
    kind = current.get("checkpoint_kind")
    if kind == "attempt_append":
        return (
            len(current_refs) == len(previous_refs) + 1
            and current.get("resume_epoch") == previous.get("resume_epoch")
            and current.get("tier2_activation_sha256")
            == previous.get("tier2_activation_sha256")
        )
    if kind == "resume_epoch":
        return (
            current_refs == previous_refs
            and current.get("resume_epoch") == previous.get("resume_epoch") + 1
            and current.get("tier2_activation_sha256")
            == previous.get("tier2_activation_sha256")
            and previous["state"]["resume_epoch_fresh_processes_remaining"] == 0
        )
    if kind == "tier2_activation":
        return (
            current_refs == previous_refs
            and current.get("resume_epoch") == previous.get("resume_epoch")
            and previous.get("tier2_activation_sha256") is None
            and current.get("tier2_activation_sha256") is not None
        )
    return False


def validate_ledger_receipt(
    ledger: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]] = (),
    ledger_history: Sequence[Mapping[str, Any]] = (),
    activation: Mapping[str, Any] | None = None,
    tier1_journal: Mapping[str, Any] | None = None,
    tier1_journal_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    rehash_sidecars: bool = True,
) -> None:
    validate_schema("ledger", ledger)
    _verify_seal(ledger, "ledger_receipt_sha256", role="ledger receipt")
    receipts_root = Path((session.get("namespace") or {}).get("receipts_root", ""))
    retained_path = receipts_root / f"ledger-{ledger.get('ledger_sequence', -1):06d}.json"
    if retained_path.is_file() or retained_path.is_symlink():
        retained, _ = _load_json_path(retained_path, role="retained ledger checkpoint")
        if retained != ledger:
            _fail("M6_LIVE_LEDGER_ROLLBACK", "retained checkpoint bytes differ")
        retained_attempts, retained_plans = _load_published_attempt_inventory(
            ledger, session=session
        )
        if list(attempts) != retained_attempts or list(plans) != retained_plans:
            _fail("M6_LIVE_PLAN_INVENTORY", "restart inventory differs from disk")
    if len(plans) != len(attempts):
        _fail("M6_LIVE_PLAN_INVENTORY", "one exact plan per attempt is required")
    for attempt, plan in zip(attempts, plans, strict=True):
        _validate_attempt_evidence(
            attempt,
            plan=plan,
            session=session,
            rehash_artifacts=True,
        )
    if ledger.get("ledger_sequence") != len(ledger_history):
        _fail("M6_LIVE_LEDGER_ROLLBACK", "checkpoint history length differs")
    history_hashes: list[str] = []
    previous_hash = ZERO_SHA256
    previous_ledger: Mapping[str, Any] | None = None
    for sequence, checkpoint in enumerate(ledger_history):
        validate_schema("ledger", checkpoint)
        _verify_seal(
            checkpoint,
            "ledger_receipt_sha256",
            role=f"ledger history {sequence}",
        )
        if (
            checkpoint.get("ledger_sequence") != sequence
            or checkpoint.get("previous_ledger_receipt_sha256") != previous_hash
            or checkpoint.get("ledger_history_sha256s") != history_hashes
            or checkpoint.get("session_authority_sha256")
            != session.get("session_authority_sha256")
            or checkpoint.get("schedule_sha256") != schedule.get("schedule_sha256")
            or (
                previous_ledger is None
                and (
                    checkpoint.get("checkpoint_kind") != "initial"
                    or checkpoint.get("resume_epoch") != 0
                )
            )
            or (
                previous_ledger is not None
                and not _ledger_transition_valid(previous_ledger, checkpoint)
            )
        ):
            _fail("M6_LIVE_LEDGER_FORK", f"checkpoint {sequence}")
        previous_hash = checkpoint["ledger_receipt_sha256"]
        history_hashes.append(previous_hash)
        previous_ledger = checkpoint
    if (
        ledger.get("previous_ledger_receipt_sha256") != previous_hash
        or ledger.get("ledger_history_sha256s") != history_hashes
    ):
        _fail("M6_LIVE_LEDGER_FORK", "ledger does not extend exact retained head")
    if not ledger_history:
        if (
            ledger.get("checkpoint_kind") != "initial"
            or ledger.get("resume_epoch") != 0
        ):
            _fail("M6_LIVE_LEDGER_ROLLBACK", "genesis checkpoint differs")
    else:
        previous_ledger = ledger_history[-1]
        if not _ledger_transition_valid(previous_ledger, ledger):
            _fail("M6_LIVE_LEDGER_FORK", "checkpoint transition differs")
    if activation is not None:
        if tier1_journal is None or tier1_journal_path is None:
            _fail(
                "M6_LIVE_TIER2_CALLER_SUPPLIED",
                "ledger activation requires its complete Tier-1 journal",
            )
        validate_tier2_activation(
            activation,
            schedule=schedule,
            tier1_journal=tier1_journal,
            journal_path=tier1_journal_path,
            repo_root=repo_root,
            rehash_sidecars=rehash_sidecars,
        )
    if (
        ledger.get("session_authority_sha256")
        != session.get("session_authority_sha256")
        or ledger.get("schedule_sha256") != schedule.get("schedule_sha256")
        or ledger.get("tier2_activation_sha256")
        != (activation.get("activation_sha256") if activation is not None else None)
    ):
        _fail("M6_LIVE_LEDGER_BINDING", "external authority differs")
    expected_refs = [
        _attempt_ref(attempt, f"receipts/attempt-{index:06d}.json")
        for index, attempt in enumerate(attempts)
    ]
    if ledger.get("attempt_receipts") != expected_refs:
        _fail("M6_LIVE_LEDGER_BINDING", "attempt receipt inventory differs")
    expected_head = attempts[-1]["receipt_sha256"] if attempts else ZERO_SHA256
    if ledger.get("attempt_head_sha256") != expected_head:
        _fail("M6_LIVE_ATTEMPT_CHAIN", "ledger head differs")
    state = _derive_ledger_state(
        session, schedule, attempts, activation, ledger["resume_epoch"]
    )
    if activation is not None:
        terminal_event_hashes = [
            attempt.get("terminal_event_sha256")
            for attempt in attempts
            if attempt.get("slot_terminal") is True
        ][: len(schedule["slots"])]
        journal_event_hashes = [
            event.get("event_sha256") for event in tier1_journal["events"]
        ]
        if terminal_event_hashes != journal_event_hashes:
            _fail(
                "M6_LIVE_TIER1_ATTEMPT_JOURNAL_BINDING",
                "Tier-1 terminal attempt receipts do not bind the activation journal",
            )
    if ledger.get("state") != state:
        _fail("M6_LIVE_LEDGER_STATE", "derived state differs")


def build_process_identity(
    *,
    plan: Mapping[str, Any],
    supervisor_receipt_path: Path,
    session: Mapping[str, Any],
) -> _VerifiedProcessIdentity:
    """Mint an opaque identity only after authenticated live-child verification."""
    _validate_plan_seal(plan)
    path = _canonical_absolute(
        supervisor_receipt_path, role="supervisor receipt", must_exist=True
    )
    _require_immutable_supervisor_receipt(path)
    attempt_root = _canonical_absolute(
        Path(plan["output"]["attempt_root"]), role="attempt root"
    )
    try:
        path.relative_to(attempt_root)
    except ValueError:
        _fail("M6_LIVE_PROCESS_IDENTITY", "supervisor receipt is outside attempt root")
    receipt, payload = _load_json_path(path, role="supervisor receipt")
    _verify_seal(receipt, "receipt_sha256", role="supervisor receipt")
    registry_identity, authority, _authentication_model = (
        _authenticate_supervisor_receipt(receipt)
    )
    fixed = _supervisor_receipt_fields(
        plan,
        session,
        registry_identity=registry_identity,
        authority=authority,
    )
    dynamic = {
        "pid",
        "start_ticks",
        "started_at_utc",
        "supervisor_pid",
        "supervisor_start_ticks",
        "observed_supervisor_executable",
        "observed_supervisor_argv",
        "observed_child_executable",
        "observed_child_argv",
        "authority_validated_at_utc",
        "storage_fault_absent_at_utc",
        "producer_signature_ed25519",
        "receipt_sha256",
    }
    if set(receipt) != set(fixed) | dynamic:
        _fail("M6_LIVE_PROCESS_IDENTITY", "supervisor receipt fields differ")
    if any(receipt.get(name) != expected for name, expected in fixed.items()):
        _fail("M6_LIVE_PROCESS_IDENTITY", "supervisor receipt plan binding differs")
    started = _parse_time(receipt.get("started_at_utc"), role="process start")
    if any(
        receipt.get(name) != receipt.get("started_at_utc")
        for name in ("authority_validated_at_utc", "storage_fault_absent_at_utc")
    ):
        _fail("M6_LIVE_PROCESS_IDENTITY", "process-start authority time differs")
    _revalidate_production_session_at_start(session, started_at=started)
    _verify_live_exact_child(receipt, plan=plan, authority=authority)
    unsigned = {
        "authority_mode": "retained_exact_child_receipt",
        "pid": receipt["pid"],
        "start_ticks": receipt["start_ticks"],
        "started_at_utc": receipt["started_at_utc"],
        "plan_sha256": plan["plan_sha256"],
        "launcher_executable": plan["launcher"]["executable"],
        "launcher_executable_sha256": plan["launcher"]["executable_sha256"],
        "launcher_argv": copy.deepcopy(plan["command"]),
        "target_executable": plan["target"]["executable"],
        "target_executable_sha256": plan["target"]["executable_sha256"],
        "target_argv": copy.deepcopy(plan["target"]["argv"]),
        "working_directory": plan["target"]["working_directory"],
        "supervisor_authority_id": authority["authority_id"],
        "supervisor_registry_sha256": registry_identity["sha256"],
        "producer_executable": authority["producer_path"],
        "producer_executable_sha256": authority["producer_sha256"],
        "supervisor_receipt": {
            "path": str(path),
            "sha256": immutable.sha256_bytes(payload),
            "size_bytes": len(payload),
            "receipt_sha256": receipt["receipt_sha256"],
        },
    }
    identity = _seal(unsigned, "identity_sha256")
    _validate_process_identity(
        identity,
        plan=plan,
        started_at_utc=receipt["started_at_utc"],
        session=session,
        allow_fixture=False,
    )
    return _VerifiedProcessIdentity(identity, _VERIFIED_PROCESS_TOKEN)


def _build_fixture_process_identity(
    *, plan: Mapping[str, Any], pid: int, start_ticks: int, started_at_utc: str
) -> dict[str, Any]:
    """Private no-credit process fixture; production must load a retained receipt."""
    _validate_plan_seal(plan)
    unsigned = {
        "authority_mode": FIXTURE_TRUST_MODE,
        "pid": pid,
        "start_ticks": start_ticks,
        "started_at_utc": started_at_utc,
        "plan_sha256": plan["plan_sha256"],
        "launcher_executable": plan["launcher"]["executable"],
        "launcher_executable_sha256": plan["launcher"]["executable_sha256"],
        "launcher_argv": copy.deepcopy(plan["command"]),
        "target_executable": plan["target"]["executable"],
        "target_executable_sha256": plan["target"]["executable_sha256"],
        "target_argv": copy.deepcopy(plan["target"]["argv"]),
        "working_directory": plan["target"]["working_directory"],
        "supervisor_authority_id": None,
        "supervisor_registry_sha256": None,
        "producer_executable": None,
        "producer_executable_sha256": None,
        "supervisor_receipt": None,
    }
    identity = _seal(unsigned, "identity_sha256")
    _validate_process_identity(identity, plan=plan, started_at_utc=started_at_utc)
    return identity


def build_attempt_receipt(
    *,
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    ledger: Mapping[str, Any],
    prior_attempts: Sequence[Mapping[str, Any]],
    prior_plans: Sequence[Mapping[str, Any]] = (),
    ledger_history: Sequence[Mapping[str, Any]] = (),
    plan: Mapping[str, Any],
    activation: Mapping[str, Any] | None,
    outcome: str,
    finished_at_utc: str,
    started_at_utc: str | None = None,
    process_identity: Mapping[str, Any] | _VerifiedProcessIdentity | None = None,
    failure: Mapping[str, Any] | None = None,
    artifacts: Sequence[Mapping[str, Any]] = (),
    collection_receipt: Mapping[str, Any] | None = None,
    terminal_event_sha256: str | None = None,
    tier1_journal: Mapping[str, Any] | None = None,
    tier1_journal_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    rehash_sidecars: bool = True,
) -> dict[str, Any]:
    validate_ledger_receipt(
        ledger,
        session=session,
        schedule=schedule,
        attempts=prior_attempts,
        plans=prior_plans,
        ledger_history=ledger_history,
        activation=activation,
        tier1_journal=tier1_journal,
        tier1_journal_path=tier1_journal_path,
        repo_root=repo_root,
        rehash_sidecars=rehash_sidecars,
    )
    _validate_plan_seal(plan)
    model = _OUTCOME_MODEL.get(outcome)
    if model is None:
        _fail("M6_LIVE_ATTEMPT_OUTCOME", repr(outcome))
    sequence = len(prior_attempts)
    if (
        plan.get("attempt_sequence") != sequence
        or plan.get("resume_epoch") != ledger.get("resume_epoch")
        or plan.get("session_authority_sha256")
        != session.get("session_authority_sha256")
        or plan.get("schedule_sha256") != schedule.get("schedule_sha256")
        or plan.get("tier2_activation_sha256")
        != (activation.get("activation_sha256") if activation is not None else None)
        or plan.get("ordinal") != ledger["state"]["next_ordinal"]
    ):
        _fail("M6_LIVE_PLAN_BINDING", "attempt plan is not the next eligible slot")
    finished = _parse_time(finished_at_utc, role="attempt finished_at_utc")
    planned = _parse_time(plan["planned_at_utc"], role="plan planned_at_utc")
    expires = _parse_time(plan["expires_at_utc"], role="plan expires_at_utc")
    if not planned <= finished <= expires:
        _fail("M6_LIVE_PLAN_EXPIRED", "attempt falls outside plan validity")
    if model["process_started"]:
        if started_at_utc is None:
            _fail("M6_LIVE_ATTEMPT_TIME", "started timestamp required")
        started = _parse_time(started_at_utc, role="attempt started_at_utc")
        if started > finished:
            _fail("M6_LIVE_ATTEMPT_TIME", "start follows finish")
    elif started_at_utc is not None:
        _fail("M6_LIVE_ATTEMPT_TIME", "pre-start outcome has a start timestamp")
    trust_mode = session.get("trust_mode")
    if process_identity is None:
        process_model = None
    elif trust_mode == PRODUCTION_TRUST_MODE:
        if not isinstance(process_identity, _VerifiedProcessIdentity):
            _fail(
                "M6_LIVE_PROCESS_CAPABILITY_REQUIRED",
                "production attempts require an opaque verified-child capability",
            )
        process_model = process_identity._materialize(_VERIFIED_PROCESS_TOKEN)
    elif trust_mode == FIXTURE_TRUST_MODE:
        if not isinstance(process_identity, Mapping):
            _fail("M6_LIVE_FIXTURE_ONLY", "opaque production identity in fixture")
        process_model = dict(process_identity)
        if process_model.get("authority_mode") != FIXTURE_TRUST_MODE:
            _fail("M6_LIVE_FIXTURE_ONLY", "fixture process identity differs")
    else:
        _fail("M6_LIVE_PROCESS_IDENTITY", "session trust mode differs")
    if model["process_started"] != (process_model is not None):
        _fail("M6_LIVE_PROCESS_IDENTITY", "process-start phase differs")
    process_sha256 = _validate_process_identity(
        process_model,
        plan=plan,
        started_at_utc=started_at_utc,
        session=session,
        allow_fixture=session.get("trust_mode") == FIXTURE_TRUST_MODE,
    )
    failure_model = None if failure is None else dict(failure)
    if (outcome == "observed") != (failure_model is None):
        _fail("M6_LIVE_ATTEMPT_FAILURE", "failure shape differs from outcome")
    artifact_models = [dict(row) for row in artifacts]
    collection_model = (
        None if collection_receipt is None else dict(collection_receipt)
    )
    if outcome == "observed" and (
        len(artifact_models) != 4 or collection_model is None
    ):
        _fail(
            "M6_LIVE_ATTEMPT_ARTIFACT",
            "observed attempt requires four artifacts and collection receipt",
        )
    if not model["process_started"] and (
        artifact_models or collection_model is not None
    ):
        _fail(
            "M6_LIVE_ATTEMPT_ARTIFACT",
            "pre-start outcome cannot claim process artifacts",
        )
    if model["slot_terminal"]:
        if not _is_sha256(terminal_event_sha256):
            _fail("M6_LIVE_TERMINAL_EVENT", "terminal slot requires journal event seal")
    elif terminal_event_sha256 is not None:
        _fail("M6_LIVE_TERMINAL_EVENT", "nonterminal attempt cannot bind an event")
    previous = prior_attempts[-1]["receipt_sha256"] if prior_attempts else ZERO_SHA256
    attempt_id = "m6-attempt-" + _digest([
        session["session_authority_sha256"],
        ledger["resume_epoch"],
        sequence,
        plan["slot_id"],
        plan["slot_attempt_index"],
        plan["plan_sha256"],
        process_sha256 or ZERO_SHA256,
    ])[:24]
    unsigned = {
        "schema_version": 1,
        "kind": "m6_live_attempt_receipt_v1",
        "trust_scope": "m6_process_integrity_only_not_q809_label_authority",
        "q809_production_matched_label_authority": False,
        "resume_epoch": ledger["resume_epoch"],
        "attempt_sequence": sequence,
        "slot_attempt_index": plan["slot_attempt_index"],
        "previous_attempt_receipt_sha256": previous,
        "session_authority_sha256": session["session_authority_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "tier2_activation_sha256": (
            activation["activation_sha256"] if activation is not None else None
        ),
        "plan_sha256": plan["plan_sha256"],
        "ordinal": plan["ordinal"],
        "slot_id": plan["slot_id"],
        "slot_sha256": plan["slot_sha256"],
        "attempt_id": attempt_id,
        "outcome": outcome,
        **model,
        "process_identity": process_model,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "failure": failure_model,
        "artifacts": artifact_models,
        "collection_receipt": collection_model,
        "terminal_event_sha256": terminal_event_sha256,
    }
    receipt = _seal(unsigned, "receipt_sha256")
    validate_schema("attempt", receipt)
    _validate_attempt_evidence(
        receipt,
        plan=plan,
        session=session,
        rehash_artifacts=False,
    )
    return receipt


def append_attempt_to_ledger(
    *,
    previous_ledger: Mapping[str, Any],
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    prior_attempts: Sequence[Mapping[str, Any]],
    prior_plans: Sequence[Mapping[str, Any]] = (),
    ledger_history: Sequence[Mapping[str, Any]] = (),
    attempt: Mapping[str, Any],
    plan: Mapping[str, Any],
    activation: Mapping[str, Any] | None,
    tier1_journal: Mapping[str, Any] | None = None,
    tier1_journal_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    rehash_sidecars: bool = True,
) -> dict[str, Any]:
    validate_ledger_receipt(
        previous_ledger,
        session=session,
        schedule=schedule,
        attempts=prior_attempts,
        plans=prior_plans,
        ledger_history=ledger_history,
        activation=activation,
        tier1_journal=tier1_journal,
        tier1_journal_path=tier1_journal_path,
        repo_root=repo_root,
        rehash_sidecars=rehash_sidecars,
    )
    _validate_attempt_evidence(
        attempt,
        plan=plan,
        session=session,
        rehash_artifacts=True,
    )
    if (
        attempt.get("attempt_sequence") != len(prior_attempts)
        or attempt.get("resume_epoch") != previous_ledger.get("resume_epoch")
    ):
        _fail("M6_LIVE_ATTEMPT_SEQUENCE", "append does not target ledger head")
    attempts = [*prior_attempts, attempt]
    plans = [*prior_plans, plan]
    state = _derive_ledger_state(
        session,
        schedule,
        attempts,
        activation,
        previous_ledger["resume_epoch"],
    )
    refs = [
        _attempt_ref(row, f"receipts/attempt-{index:06d}.json")
        for index, row in enumerate(attempts)
    ]
    unsigned = {
        "schema_version": 1,
        "kind": "m6_live_ledger_receipt_v1",
        "checkpoint_kind": "attempt_append",
        "ledger_sequence": previous_ledger["ledger_sequence"] + 1,
        "resume_epoch": previous_ledger["resume_epoch"],
        "previous_ledger_receipt_sha256": previous_ledger[
            "ledger_receipt_sha256"
        ],
        "ledger_history_sha256s": [
            *previous_ledger["ledger_history_sha256s"],
            previous_ledger["ledger_receipt_sha256"],
        ],
        "session_authority_sha256": session["session_authority_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "tier2_activation_sha256": (
            activation["activation_sha256"] if activation is not None else None
        ),
        "attempt_receipts": refs,
        "attempt_head_sha256": attempt["receipt_sha256"],
        "state": state,
    }
    ledger = _seal(unsigned, "ledger_receipt_sha256")
    validate_ledger_receipt(
        ledger,
        session=session,
        schedule=schedule,
        attempts=attempts,
        plans=plans,
        ledger_history=[*ledger_history, previous_ledger],
        activation=activation,
        tier1_journal=tier1_journal,
        tier1_journal_path=tier1_journal_path,
        repo_root=repo_root,
        rehash_sidecars=rehash_sidecars,
    )
    return ledger


def begin_resume_epoch(
    *,
    previous_ledger: Mapping[str, Any],
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    ledger_history: Sequence[Mapping[str, Any]],
    activation: Mapping[str, Any] | None = None,
    tier1_journal: Mapping[str, Any] | None = None,
    tier1_journal_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    rehash_sidecars: bool = True,
) -> dict[str, Any]:
    """Advance one causal resume epoch only after the prior epoch is exhausted."""
    validate_ledger_receipt(
        previous_ledger,
        session=session,
        schedule=schedule,
        attempts=attempts,
        plans=plans,
        ledger_history=ledger_history,
        activation=activation,
        tier1_journal=tier1_journal,
        tier1_journal_path=tier1_journal_path,
        repo_root=repo_root,
        rehash_sidecars=rehash_sidecars,
    )
    if previous_ledger["state"]["resume_epoch_fresh_processes_remaining"] != 0:
        _fail("M6_LIVE_RESUME_EPOCH_EARLY", "current resume budget remains")
    resume_epoch = previous_ledger["resume_epoch"] + 1
    state = _derive_ledger_state(
        session, schedule, attempts, activation, resume_epoch
    )
    unsigned = {
        "schema_version": 1,
        "kind": "m6_live_ledger_receipt_v1",
        "checkpoint_kind": "resume_epoch",
        "ledger_sequence": previous_ledger["ledger_sequence"] + 1,
        "resume_epoch": resume_epoch,
        "previous_ledger_receipt_sha256": previous_ledger[
            "ledger_receipt_sha256"
        ],
        "ledger_history_sha256s": [
            *previous_ledger["ledger_history_sha256s"],
            previous_ledger["ledger_receipt_sha256"],
        ],
        "session_authority_sha256": session["session_authority_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "tier2_activation_sha256": (
            activation["activation_sha256"] if activation is not None else None
        ),
        "attempt_receipts": copy.deepcopy(previous_ledger["attempt_receipts"]),
        "attempt_head_sha256": previous_ledger["attempt_head_sha256"],
        "state": state,
    }
    ledger = _seal(unsigned, "ledger_receipt_sha256")
    validate_ledger_receipt(
        ledger,
        session=session,
        schedule=schedule,
        attempts=attempts,
        plans=plans,
        ledger_history=[*ledger_history, previous_ledger],
        activation=activation,
        tier1_journal=tier1_journal,
        tier1_journal_path=tier1_journal_path,
        repo_root=repo_root,
        rehash_sidecars=rehash_sidecars,
    )
    return ledger


def derive_tier2_activation(
    *,
    schedule: Mapping[str, Any],
    tier1_journal: Mapping[str, Any],
    journal_path: Path,
    repo_root: Path = REPO_ROOT,
    rehash_sidecars: bool = True,
) -> dict[str, Any]:
    """Derive activation only from a validated, complete Tier-1 journal."""
    try:
        m6.validate_schedule(schedule, repo_root=repo_root, rehash_sources=True)
        m6.validate_journal(
            schedule,
            tier1_journal,
            journal_path=journal_path,
            repo_root=repo_root,
            rehash_sidecars=rehash_sidecars,
        )
    except m6.M6Error as exc:
        _fail("M6_LIVE_TIER1_JOURNAL_INVALID", str(exc))
    if tier1_journal.get("schema_version") != m6.ACTIVE_SCHEMA_VERSION:
        _fail("M6_LIVE_TIER1_JOURNAL_INVALID", "active v2 journal required")
    events = tier1_journal.get("events")
    expected_slots = list(schedule["slots"])
    if (
        not isinstance(events, list)
        or len(events) != len(expected_slots)
        or len(expected_slots) != m6.TIER1_SLOTS
        or tier1_journal.get("causal_prefix_closed") is not True
        or [event.get("slot_id") for event in events]
        != [slot["slot_id"] for slot in expected_slots]
        or any(event.get("tier") != 1 for event in events)
    ):
        _fail(
            "M6_LIVE_TIER1_INCOMPLETE",
            "all 8540 Tier-1 terminals in shuffled ordinal order are required",
        )
    rows = m6._trigger_rows_from_events(schedule, events)
    decisions = m6._trigger_decisions(schedule, rows)
    if len(decisions) != m6.EXPECTED_RUNNABLE:
        _fail("M6_LIVE_TIER2_DERIVATION", "decision count differs")
    activated_cells = [
        decision["cell_id"] for decision in decisions if decision["triggered"]
    ]
    activated_set = set(activated_cells)
    candidates = [
        slot for slot in schedule["tier2_candidates"]
        if slot["cell_id"] in activated_set
    ]
    terminal_ids = [slot["slot_id"] for slot in expected_slots]
    unsigned = {
        "schema_version": 1,
        "kind": "m6_derived_tier2_activation_v1",
        "schedule_sha256": schedule["schedule_sha256"],
        "tier1_journal_sha256": tier1_journal["journal_sha256"],
        "derivation": TIER2_DERIVATION,
        "tier1_terminal_slots": m6.TIER1_SLOTS,
        "tier1_terminal_slot_ids_sha256": _digest(terminal_ids),
        "trigger_decisions": decisions,
        "trigger_decisions_sha256": _digest(decisions),
        "activated_cell_ids": activated_cells,
        "activated_ordinals": [slot["ordinal"] for slot in candidates],
        "activated_candidate_slots_sha256": _digest(candidates),
        "counts": {
            "tier1_slots": m6.TIER1_SLOTS,
            "decisions": m6.EXPECTED_RUNNABLE,
            "triggered_cells": len(activated_cells),
            "activated_tier2_slots": len(candidates),
        },
    }
    activation = _seal(unsigned, "activation_sha256")
    validate_schema("tier2", activation)
    return activation


def validate_tier2_activation(
    activation: Mapping[str, Any],
    *,
    schedule: Mapping[str, Any],
    tier1_journal: Mapping[str, Any],
    journal_path: Path,
    repo_root: Path = REPO_ROOT,
    rehash_sidecars: bool = True,
) -> None:
    expected = derive_tier2_activation(
        schedule=schedule,
        tier1_journal=tier1_journal,
        journal_path=journal_path,
        repo_root=repo_root,
        rehash_sidecars=rehash_sidecars,
    )
    if activation != expected:
        _fail("M6_LIVE_TIER2_CALLER_SUPPLIED", "activation does not rederive")


def activate_tier2_ledger(
    *,
    previous_ledger: Mapping[str, Any],
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    ledger_history: Sequence[Mapping[str, Any]],
    activation: Mapping[str, Any],
    tier1_journal: Mapping[str, Any],
    tier1_journal_path: Path,
    repo_root: Path = REPO_ROOT,
    rehash_sidecars: bool = True,
) -> dict[str, Any]:
    validate_ledger_receipt(
        previous_ledger,
        session=session,
        schedule=schedule,
        attempts=attempts,
        plans=plans,
        ledger_history=ledger_history,
        activation=None,
    )
    if previous_ledger["state"]["terminal_prefix_slots"] != len(schedule["slots"]):
        _fail("M6_LIVE_TIER1_INCOMPLETE", "ledger has not closed Tier-1")
    validate_tier2_activation(
        activation,
        schedule=schedule,
        tier1_journal=tier1_journal,
        journal_path=tier1_journal_path,
        repo_root=repo_root,
        rehash_sidecars=rehash_sidecars,
    )
    _active_slots(schedule, activation)
    state = _derive_ledger_state(
        session,
        schedule,
        attempts,
        activation,
        previous_ledger["resume_epoch"],
    )
    unsigned = {
        "schema_version": 1,
        "kind": "m6_live_ledger_receipt_v1",
        "checkpoint_kind": "tier2_activation",
        "ledger_sequence": previous_ledger["ledger_sequence"] + 1,
        "resume_epoch": previous_ledger["resume_epoch"],
        "previous_ledger_receipt_sha256": previous_ledger[
            "ledger_receipt_sha256"
        ],
        "ledger_history_sha256s": [
            *previous_ledger["ledger_history_sha256s"],
            previous_ledger["ledger_receipt_sha256"],
        ],
        "session_authority_sha256": session["session_authority_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "tier2_activation_sha256": activation["activation_sha256"],
        "attempt_receipts": copy.deepcopy(previous_ledger["attempt_receipts"]),
        "attempt_head_sha256": previous_ledger["attempt_head_sha256"],
        "state": state,
    }
    ledger = _seal(unsigned, "ledger_receipt_sha256")
    validate_ledger_receipt(
        ledger,
        session=session,
        schedule=schedule,
        attempts=attempts,
        plans=plans,
        ledger_history=[*ledger_history, previous_ledger],
        activation=activation,
        tier1_journal=tier1_journal,
        tier1_journal_path=tier1_journal_path,
        repo_root=repo_root,
        rehash_sidecars=rehash_sidecars,
    )
    return ledger


def _matrix_cell(
    matrix: Mapping[str, Any], schedule_cell: Mapping[str, Any]
) -> Mapping[str, Any]:
    matches = [
        cell for cell in matrix.get("cells", ())
        if isinstance(cell, Mapping)
        and cell.get("disposition") == "runnable"
        and (cell.get("identity") or {}).get("roster_source")
        == schedule_cell.get("roster_source")
        and (cell.get("identity") or {}).get("slug") == schedule_cell.get("slug")
    ]
    if len(matches) != 1:
        _fail("M6_LIVE_CELL_BINDING", schedule_cell.get("cell_id", "unknown"))
    cell = matches[0]
    invocation = cell.get("invocation") or {}
    if (
        cell.get("authority_cell_sha256")
        != schedule_cell.get("authority_cell_sha256")
        or invocation.get("executable_sha256")
        != schedule_cell.get("binary_sha256")
    ):
        _fail("M6_LIVE_CELL_BINDING", schedule_cell["cell_id"])
    return cell


def _m4_launch(
    schedule: Mapping[str, Any], schedule_cell: Mapping[str, Any]
) -> Mapping[str, Any]:
    rows = [
        row for row in schedule["m4_identity"]["cell_launch_identities"]
        if row.get("cell_id") == schedule_cell["cell_id"]
    ]
    if len(rows) != 1 or len(rows[0].get("launches", ())) != 1:
        _fail("M6_LIVE_M4_LAUNCH", schedule_cell["cell_id"])
    row = rows[0]
    if (
        row.get("cell_identity_sha256")
        != schedule_cell.get("cell_identity_sha256")
        or row.get("binary_sha256") != schedule_cell.get("binary_sha256")
    ):
        _fail("M6_LIVE_M4_LAUNCH", schedule_cell["cell_id"])
    return row["launches"][0]


def plan_next_slot(
    *,
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    schedule_path: Path,
    preflight_paths: Sequence[Path],
    ledger: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]] = (),
    ledger_history: Sequence[Mapping[str, Any]] = (),
    activation: Mapping[str, Any] | None = None,
    tier1_journal: Mapping[str, Any] | None = None,
    tier1_journal_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    rehash_sidecars: bool = True,
    verify_target_bytes: bool = True,
) -> dict[str, Any]:
    """Return the next exact command plan without querying or spawning hardware."""
    context = validate_session_authority(
        session,
        schedule=schedule,
        schedule_path=schedule_path,
        preflight_paths=preflight_paths,
        repo_root=repo_root,
    )
    validate_ledger_receipt(
        ledger,
        session=session,
        schedule=schedule,
        attempts=attempts,
        plans=plans,
        ledger_history=ledger_history,
        activation=activation,
        tier1_journal=tier1_journal,
        tier1_journal_path=tier1_journal_path,
        repo_root=repo_root,
        rehash_sidecars=rehash_sidecars,
    )
    state = ledger["state"]
    if state["resume_epoch_fresh_processes_remaining"] < 1:
        _fail("M6_LIVE_COST_CEILING", "resume epoch has no fresh process remaining")
    if state["tier1_complete"] and activation is None:
        _fail(
            "M6_LIVE_TIER2_ACTIVATION_REQUIRED",
            "Tier-2 can follow only a derived activation",
        )
    if state["study_complete"] or state["next_ordinal"] is None:
        _fail("M6_LIVE_STUDY_COMPLETE", "no active slot remains")
    active = _active_slots(schedule, activation)
    slot = next(
        (row for row in active if row["ordinal"] == state["next_ordinal"]),
        None,
    )
    if slot is None:
        _fail("M6_LIVE_EXACT_PREFIX", "next ordinal is not active")
    schedule_cell = next(
        (row for row in schedule["cells"] if row["cell_id"] == slot["cell_id"]),
        None,
    )
    config = next(
        (row for row in schedule["configs"] if row["config_id"] == slot["config_id"]),
        None,
    )
    if schedule_cell is None or config is None:
        _fail("M6_LIVE_SLOT_BINDING", slot["slot_id"])
    if (
        config.get("config_sha256") != slot.get("config_sha256")
        or config.get("fresh_process_per_profile") is not True
        or slot.get("process_isolation") != "fresh_process"
        or config.get("replay_mode") != "kernel"
        or config.get("target_processes") != "all"
        or config.get("filter_mode") != "global"
        or config.get("profile_from_start") is not False
    ):
        _fail("M6_LIVE_CONFIG_BINDING", slot["slot_id"])
    expected_config_id = (
        f"cfg=clock_{config['clock_control']}__cache_{config['cache_control']}"
    )
    if config["config_id"] != expected_config_id:
        _fail("M6_LIVE_CONFIG_BINDING", "clock/cache mapping differs")

    matrix_cell = _matrix_cell(context["matrix"], schedule_cell)
    invocation = matrix_cell["invocation"]
    executable_declared = repo_root / invocation["executable_path"]
    try:
        executable = executable_declared.resolve(strict=True)
    except OSError as exc:
        _fail("M6_LIVE_TARGET_IDENTITY", f"{slot['cell_id']}: {exc}")
    if not executable.is_file():
        _fail("M6_LIVE_TARGET_IDENTITY", f"{slot['cell_id']}: not regular")
    if verify_target_bytes:
        payload = _read_regular(executable, role=f"target {slot['cell_id']}")
        if immutable.sha256_bytes(payload) != invocation["executable_sha256"]:
            _fail("M6_LIVE_TARGET_IDENTITY", f"{slot['cell_id']}: SHA-256")
    app_args = invocation.get("label_app_args")
    logical = invocation.get("logical_workload")
    if (
        not isinstance(app_args, list)
        or any(type(argument) is not str or "\x00" in argument for argument in app_args)
        or not isinstance(logical, Mapping)
        or type(logical.get("value")) is not str
        or str(logical["value"]) != schedule_cell["shape_point"]
    ):
        _fail("M6_LIVE_SHAPE_BINDING", slot["cell_id"])
    shape_delivery = invocation.get("shape_delivery")
    source = schedule_cell["roster_source"]
    if shape_delivery == "application_args":
        shape_binding = "application-args"
    elif (
        shape_delivery == "adapter_consumes_logical_shape"
        and source == "sol_execbench"
        and invocation.get("adapter_kind") == "sol_execbench"
    ):
        shape_binding = "logical-workload"
    else:
        _fail("M6_LIVE_SHAPE_BINDING", slot["cell_id"])
    launch = _m4_launch(schedule, schedule_cell)
    launch_contract = invocation.get("capture_launch_contract") or {}
    if (
        launch.get("launch_index") != launch_contract.get("label_launch_index")
        or type(launch.get("kernel_name")) is not str
        or not launch["kernel_name"]
        or launch["kernel_name"].startswith("regex:")
    ):
        _fail("M6_LIVE_LAUNCH_SELECTOR", slot["cell_id"])
    launch_index = launch["launch_index"]
    warmup = launch_contract.get("label_driver_warmup_count")
    if type(warmup) is not int or warmup < 0:
        _fail("M6_LIVE_LAUNCH_SELECTOR", slot["cell_id"])

    slot_attempt_index = sum(
        attempt["slot_id"] == slot["slot_id"] for attempt in attempts
    )
    attempt_sequence = len(attempts)
    attempt_root = (
        Path(session["namespace"]["attempts_root"])
        / f"{slot['ordinal']:04d}-{slot['slot_id']}"
        / f"attempt-{slot_attempt_index:03d}"
    )
    evidence_root = attempt_root / "evidence"
    sidecar_path = (
        evidence_root / EXPECTED_ARCH / schedule_cell["slug"]
        / schedule_cell["shape_point"]
        / f"label_{slot['repeat_index']}_{launch_index}.json"
    )
    receipt_path = (
        Path(session["namespace"]["receipts_root"])
        / f"attempt-{attempt_sequence:06d}.json"
    )
    controls = {
        "target_processes": "all",
        "filter_mode": "global",
        "replay_mode": "kernel",
        "cache_control": config["cache_control"],
        "clock_control": config["clock_control"],
        "profile_from_start": False,
        "launch_skip": launch_index,
        "launch_count": 1,
    }
    ncu = session["ncu_identity"]
    launcher = (
        repo_root / session["tools"]["label_driver"]["path"]
    ).resolve(strict=True)
    command = [
        str(launcher),
        "--device", EXPECTED_ARCH,
        "--slug", schedule_cell["slug"],
        "--shape-point", schedule_cell["shape_point"],
        "--shape-binding", shape_binding,
        "--repeat", str(slot["repeat_index"]),
        "--launch", str(launch_index),
        "--binary", str(executable),
        "--warmup", str(warmup),
        "--ncu", ncu["resolved_path"],
        "--ncu-executable-sha256", ncu["executable_sha256"],
        "--ncu-version-line", ncu["version_line"],
        "--target-processes", controls["target_processes"],
        "--filter-mode", controls["filter_mode"],
        "--replay-mode", controls["replay_mode"],
        "--cache-control", controls["cache_control"],
        "--clock-control", controls["clock_control"],
        "--profile-from-start", "off",
        "--launch-skip", str(launch_index),
        "--launch-count", "1",
        "--out-dir", str(evidence_root),
        "--kernel-name", launch["kernel_name"],
    ]
    for metric in context["explicit_unit_metrics"]:
        command.extend(("--unit-metric", metric))
    command.extend(("--", *app_args))
    working_directory = repo_root if source == "sol_execbench" else executable.parent
    planned = _trusted_now()
    planned_at_utc = _format_time(planned)
    expires_at_utc = _format_time(planned + timedelta(seconds=FRESHNESS_SECONDS))
    unsigned = {
        "schema_version": 1,
        "kind": "m6_live_slot_plan_v1",
        "planned_at_utc": planned_at_utc,
        "expires_at_utc": expires_at_utc,
        "session_authority_sha256": session["session_authority_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "tier2_activation_sha256": (
            activation["activation_sha256"] if activation is not None else None
        ),
        "resume_epoch": ledger["resume_epoch"],
        "attempt_sequence": attempt_sequence,
        "slot_attempt_index": slot_attempt_index,
        "ordinal": slot["ordinal"],
        "slot_id": slot["slot_id"],
        "slot_sha256": slot["slot_sha256"],
        "tier": slot["tier"],
        "cell_id": slot["cell_id"],
        "cell_identity_sha256": slot["cell_identity_sha256"],
        "config_id": slot["config_id"],
        "config_sha256": slot["config_sha256"],
        "sweep_index": slot["sweep_index"],
        "repeat_index": slot["repeat_index"],
        "fresh_process": True,
        "launcher": {
            "executable": str(launcher),
            "executable_sha256": session["tools"]["label_driver"]["sha256"],
            "mode": session["tools"]["label_driver"]["mode"],
        },
        "target": {
            "executable": str(executable),
            "executable_sha256": invocation["executable_sha256"],
            "argv": [str(executable), *app_args],
            "working_directory": str(working_directory),
            "roster_source": source,
            "slug": schedule_cell["slug"],
        },
        "shape": {
            "shape_point": schedule_cell["shape_point"],
            "shape_binding": shape_binding,
            "logical_workload_sha256": _digest(logical),
            "application_argv": list(app_args),
        },
        "launch_selector": {
            "launch_index": launch_index,
            "kernel_name": launch["kernel_name"],
            "kernel_signature_sha256": launch["kernel_signature_sha256"],
            "m4_launch_identity_sha256": launch["launch_identity_sha256"],
        },
        "ncu_controls": controls,
        "metric_request": {
            "selected_count": 55,
            "metric_identity_sha256": schedule["metric_identity_sha256"],
            "requested_ncu_metrics": context["requested_ncu_metrics"],
            "implicit_flat_metrics": context["implicit_flat_metrics"],
            "explicit_unit_metrics": context["explicit_unit_metrics"],
        },
        "output": {
            "attempt_root": str(attempt_root),
            "evidence_root": str(evidence_root),
            "sidecar_path": str(sidecar_path),
            "attempt_receipt_path": str(receipt_path),
        },
        "command": command,
    }
    plan = _seal(unsigned, "plan_sha256")
    _validate_plan_seal(plan)
    return plan


def validate_slot_plan(
    plan: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    schedule_path: Path,
    preflight_paths: Sequence[Path],
    ledger: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]] = (),
    ledger_history: Sequence[Mapping[str, Any]] = (),
    activation: Mapping[str, Any] | None = None,
    tier1_journal: Mapping[str, Any] | None = None,
    tier1_journal_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Rederive every executable plan field from authenticated authorities."""
    _validate_plan_seal(plan)
    _fresh_at(plan["planned_at_utc"], role="plan publication", now=_trusted_now())
    expected = plan_next_slot(
        session=session,
        schedule=schedule,
        schedule_path=schedule_path,
        preflight_paths=preflight_paths,
        ledger=ledger,
        attempts=attempts,
        plans=plans,
        ledger_history=ledger_history,
        activation=activation,
        tier1_journal=tier1_journal,
        tier1_journal_path=tier1_journal_path,
        repo_root=repo_root,
    )
    ignored = {"planned_at_utc", "expires_at_utc", "plan_sha256"}
    retained = {key: value for key, value in plan.items() if key not in ignored}
    derived = {key: value for key, value in expected.items() if key not in ignored}
    if retained != derived:
        _fail("M6_LIVE_PLAN_BINDING", "plan does not rederive from authorities")


_PUBLICATION_VALIDATED = object()


def _publish_no_replace(
    path: Path,
    document: Mapping[str, Any],
    *,
    kind: str,
    validation_capability: object | None = None,
) -> Path:
    if validation_capability is not _PUBLICATION_VALIDATED:
        _fail("M6_LIVE_PUBLICATION_UNVALIDATED", kind)
    validate_schema(kind, document)
    absolute = _canonical_absolute(path, role=f"{kind} publication")
    try:
        result = immutable.publish_regular_file(
            absolute.parent,
            absolute.name,
            immutable.pretty_json_bytes(document),
            partial_suffix=".m6-partial",
        )
    except immutable.ImmutableArtifactError as exc:
        _fail("M6_LIVE_NO_REPLACE", f"{kind}: {exc}")
    if not isinstance(result, Path):
        _fail("M6_LIVE_PUBLICATION", f"{kind}: unexpected retained capability")
    return result


def _require_publication_path(path: Path, expected: Path, *, kind: str) -> Path:
    absolute = _canonical_absolute(path, role=f"{kind} publication")
    if absolute != expected:
        _fail(
            "M6_LIVE_PUBLICATION_PATH",
            f"{kind}: expected {expected}, observed {absolute}",
        )
    return absolute


def publish_session_authority(
    path: Path,
    document: Mapping[str, Any],
    *,
    schedule: Mapping[str, Any],
    schedule_path: Path,
    preflight_paths: Sequence[Path],
    repo_root: Path = REPO_ROOT,
) -> Path:
    validate_session_authority(
        document,
        schedule=schedule,
        schedule_path=schedule_path,
        preflight_paths=preflight_paths,
        repo_root=repo_root,
    )
    expected = Path(document["namespace"]["session_root"]) / "session-authority.json"
    absolute = _require_publication_path(path, expected, kind="session")
    return _publish_no_replace(
        absolute,
        document,
        kind="session",
        validation_capability=_PUBLICATION_VALIDATED,
    )


def _publish_fixture_session_authority(path: Path, document: Mapping[str, Any]) -> Path:
    validate_schema("session", document)
    _verify_seal(document, "session_authority_sha256", role="fixture session")
    if document.get("trust_mode") != FIXTURE_TRUST_MODE:
        _fail("M6_LIVE_FIXTURE_ONLY", "fixture publisher rejects production trust")
    expected = Path(document["namespace"]["session_root"]) / "session-authority.json"
    absolute = _require_publication_path(path, expected, kind="session")
    return _publish_no_replace(
        absolute,
        document,
        kind="session",
        validation_capability=_PUBLICATION_VALIDATED,
    )


def publish_slot_plan(
    path: Path,
    document: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    schedule_path: Path,
    preflight_paths: Sequence[Path],
    ledger: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]] = (),
    ledger_history: Sequence[Mapping[str, Any]] = (),
    repo_root: Path = REPO_ROOT,
) -> Path:
    validate_slot_plan(
        document,
        session=session,
        schedule=schedule,
        schedule_path=schedule_path,
        preflight_paths=preflight_paths,
        ledger=ledger,
        attempts=attempts,
        plans=plans,
        ledger_history=ledger_history,
        repo_root=repo_root,
    )
    expected = (
        Path(session["namespace"]["receipts_root"])
        / f"plan-{document['attempt_sequence']:06d}.json"
    )
    absolute = _require_publication_path(path, expected, kind="plan")
    return _publish_no_replace(
        absolute,
        document,
        kind="plan",
        validation_capability=_PUBLICATION_VALIDATED,
    )


def _publish_fixture_slot_plan(
    path: Path, document: Mapping[str, Any], *, session: Mapping[str, Any]
) -> Path:
    if session.get("trust_mode") != FIXTURE_TRUST_MODE:
        _fail("M6_LIVE_FIXTURE_ONLY", "fixture plan publisher")
    _validate_plan_seal(document)
    if document.get("session_authority_sha256") != session.get("session_authority_sha256"):
        _fail("M6_LIVE_PLAN_BINDING", "fixture plan session differs")
    expected = (
        Path(session["namespace"]["receipts_root"])
        / f"plan-{document['attempt_sequence']:06d}.json"
    )
    absolute = _require_publication_path(path, expected, kind="plan")
    return _publish_no_replace(
        absolute,
        document,
        kind="plan",
        validation_capability=_PUBLICATION_VALIDATED,
    )


def _require_published_plan(
    plan: Mapping[str, Any], *, session: Mapping[str, Any]
) -> None:
    path = (
        Path(session["namespace"]["receipts_root"])
        / f"plan-{plan['attempt_sequence']:06d}.json"
    )
    retained, _ = _load_json_path(path, role="published slot plan")
    _validate_plan_seal(retained)
    if retained != plan:
        _fail("M6_LIVE_PLAN_INVENTORY", "published plan bytes differ")


def publish_attempt_receipt(
    path: Path,
    document: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    schedule_path: Path,
    preflight_paths: Sequence[Path],
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    prior_attempts: Sequence[Mapping[str, Any]],
    prior_plans: Sequence[Mapping[str, Any]] = (),
    ledger_history: Sequence[Mapping[str, Any]] = (),
    repo_root: Path = REPO_ROOT,
) -> Path:
    validate_session_authority(
        session,
        schedule=schedule,
        schedule_path=schedule_path,
        preflight_paths=preflight_paths,
        repo_root=repo_root,
    )
    validate_slot_plan(
        plan,
        session=session,
        schedule=schedule,
        schedule_path=schedule_path,
        preflight_paths=preflight_paths,
        ledger=ledger,
        attempts=prior_attempts,
        plans=prior_plans,
        ledger_history=ledger_history,
        repo_root=repo_root,
    )
    _require_published_plan(plan, session=session)
    _validate_attempt_evidence(
        document, plan=plan, session=session, rehash_artifacts=True
    )
    if document.get("attempt_sequence") != len(prior_attempts):
        _fail("M6_LIVE_ATTEMPT_SEQUENCE", "publication is not next")
    expected = (
        Path(session["namespace"]["receipts_root"])
        / f"attempt-{document['attempt_sequence']:06d}.json"
    )
    absolute = _require_publication_path(path, expected, kind="attempt")
    return _publish_no_replace(
        absolute,
        document,
        kind="attempt",
        validation_capability=_PUBLICATION_VALIDATED,
    )


def _publish_fixture_attempt_receipt(
    path: Path,
    document: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Path:
    if session.get("trust_mode") != FIXTURE_TRUST_MODE:
        _fail("M6_LIVE_FIXTURE_ONLY", "fixture attempt publisher")
    validate_schema("attempt", document)
    _verify_seal(document, "receipt_sha256", role="fixture attempt")
    _require_published_plan(plan, session=session)
    _validate_attempt_evidence(
        document, plan=plan, session=session, rehash_artifacts=True
    )
    expected = (
        Path(session["namespace"]["receipts_root"])
        / f"attempt-{document['attempt_sequence']:06d}.json"
    )
    absolute = _require_publication_path(path, expected, kind="attempt")
    return _publish_no_replace(
        absolute,
        document,
        kind="attempt",
        validation_capability=_PUBLICATION_VALIDATED,
    )


def _load_published_attempt_inventory(
    ledger: Mapping[str, Any], *, session: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session_root = Path(session["namespace"]["session_root"])
    attempts: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    for sequence, reference in enumerate(ledger.get("attempt_receipts", ())):
        expected_attempt = f"receipts/attempt-{sequence:06d}.json"
        expected_plan = f"receipts/plan-{sequence:06d}.json"
        if (
            reference.get("relative_path") != expected_attempt
            or reference.get("plan_relative_path") != expected_plan
        ):
            _fail("M6_LIVE_PLAN_INVENTORY", f"published inventory {sequence}")
        attempt, _ = _load_json_path(
            session_root / expected_attempt, role=f"published attempt {sequence}"
        )
        plan, _ = _load_json_path(
            session_root / expected_plan, role=f"published plan {sequence}"
        )
        validate_schema("attempt", attempt)
        _verify_seal(attempt, "receipt_sha256", role=f"published attempt {sequence}")
        _validate_plan_seal(plan)
        if (
            attempt.get("receipt_sha256") != reference.get("receipt_sha256")
            or plan.get("plan_sha256") != reference.get("plan_sha256")
        ):
            _fail("M6_LIVE_PLAN_INVENTORY", f"published bytes {sequence}")
        _validate_attempt_evidence(
            attempt, plan=plan, session=session, rehash_artifacts=True
        )
        attempts.append(attempt)
        plans.append(plan)
    return attempts, plans


def _validate_published_ledger_prefix(
    ledger: Mapping[str, Any], *, session: Mapping[str, Any]
) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    receipts_root = Path(session["namespace"]["receipts_root"])
    sequence = ledger["ledger_sequence"]
    expected_hashes = ledger["ledger_history_sha256s"]
    observed_paths = sorted(receipts_root.glob("ledger-*.json"))
    expected_paths = [
        receipts_root / f"ledger-{index:06d}.json" for index in range(sequence)
    ]
    retained = [path for path in observed_paths if path.name != f"ledger-{sequence:06d}.json"]
    if retained != expected_paths:
        _fail("M6_LIVE_LEDGER_ROLLBACK", "published checkpoint inventory differs")
    previous = ZERO_SHA256
    previous_checkpoint: Mapping[str, Any] | None = None
    for index, (checkpoint_path, expected_hash) in enumerate(
        zip(expected_paths, expected_hashes, strict=True)
    ):
        checkpoint, _ = _load_json_path(
            checkpoint_path, role=f"published ledger {index}"
        )
        validate_schema("ledger", checkpoint)
        _verify_seal(
            checkpoint, "ledger_receipt_sha256", role=f"published ledger {index}"
        )
        if (
            checkpoint.get("ledger_sequence") != index
            or checkpoint.get("previous_ledger_receipt_sha256") != previous
            or checkpoint.get("ledger_receipt_sha256") != expected_hash
            or checkpoint.get("ledger_history_sha256s") != expected_hashes[:index]
            or checkpoint.get("session_authority_sha256")
            != session.get("session_authority_sha256")
            or checkpoint.get("schedule_sha256")
            != ledger.get("schedule_sha256")
            or (
                previous_checkpoint is None
                and checkpoint.get("checkpoint_kind") != "initial"
            )
            or (
                previous_checkpoint is not None
                and not _ledger_transition_valid(previous_checkpoint, checkpoint)
            )
        ):
            _fail("M6_LIVE_LEDGER_FORK", f"published checkpoint {index}")
        previous = expected_hash
        previous_checkpoint = checkpoint
        checkpoints.append(checkpoint)
    if ledger.get("previous_ledger_receipt_sha256") != previous:
        _fail("M6_LIVE_LEDGER_FORK", "new checkpoint does not extend published head")
    if (
        previous_checkpoint is None
        and ledger.get("checkpoint_kind") != "initial"
    ) or (
        previous_checkpoint is not None
        and not _ledger_transition_valid(previous_checkpoint, ledger)
    ):
        _fail("M6_LIVE_LEDGER_FORK", "new checkpoint transition differs")
    return checkpoints


def publish_ledger_receipt(
    path: Path,
    document: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    schedule_path: Path,
    preflight_paths: Sequence[Path],
    activation: Mapping[str, Any] | None = None,
    tier1_journal: Mapping[str, Any] | None = None,
    tier1_journal_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> Path:
    validate_session_authority(
        session,
        schedule=schedule,
        schedule_path=schedule_path,
        preflight_paths=preflight_paths,
        repo_root=repo_root,
    )
    expected = (
        Path(session["namespace"]["receipts_root"])
        / f"ledger-{document['ledger_sequence']:06d}.json"
    )
    absolute = _require_publication_path(path, expected, kind="ledger")
    history = _validate_published_ledger_prefix(document, session=session)
    attempts, plans = _load_published_attempt_inventory(document, session=session)
    checkpoints = [*history, document]
    for sequence, plan in enumerate(plans):
        append_index = next(
            (
                index
                for index, checkpoint in enumerate(checkpoints)
                if checkpoint.get("checkpoint_kind") == "attempt_append"
                and len(checkpoint.get("attempt_receipts", ())) == sequence + 1
            ),
            None,
        )
        if append_index is None or append_index < 1:
            _fail("M6_LIVE_PLAN_INVENTORY", f"no causal checkpoint for plan {sequence}")
        previous = checkpoints[append_index - 1]
        prior_activation = (
            activation if previous.get("tier2_activation_sha256") is not None else None
        )
        validate_slot_plan(
            plan,
            session=session,
            schedule=schedule,
            schedule_path=schedule_path,
            preflight_paths=preflight_paths,
            ledger=previous,
            attempts=attempts[:sequence],
            plans=plans[:sequence],
            ledger_history=checkpoints[: append_index - 1],
            activation=prior_activation,
            tier1_journal=tier1_journal,
            tier1_journal_path=tier1_journal_path,
            repo_root=repo_root,
        )
    validate_ledger_receipt(
        document,
        session=session,
        schedule=schedule,
        attempts=attempts,
        plans=plans,
        ledger_history=history,
        activation=activation,
        tier1_journal=tier1_journal,
        tier1_journal_path=tier1_journal_path,
        repo_root=repo_root,
    )
    return _publish_no_replace(
        absolute,
        document,
        kind="ledger",
        validation_capability=_PUBLICATION_VALIDATED,
    )


def _publish_fixture_ledger_receipt(
    path: Path,
    document: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> Path:
    if session.get("trust_mode") != FIXTURE_TRUST_MODE:
        _fail("M6_LIVE_FIXTURE_ONLY", "fixture ledger publisher")
    expected = (
        Path(session["namespace"]["receipts_root"])
        / f"ledger-{document['ledger_sequence']:06d}.json"
    )
    absolute = _require_publication_path(path, expected, kind="ledger")
    history = _validate_published_ledger_prefix(document, session=session)
    attempts, plans = _load_published_attempt_inventory(document, session=session)
    validate_ledger_receipt(
        document,
        session=session,
        schedule=schedule,
        attempts=attempts,
        plans=plans,
        ledger_history=history,
    )
    return _publish_no_replace(
        absolute,
        document,
        kind="ledger",
        validation_capability=_PUBLICATION_VALIDATED,
    )


def publish_tier2_activation(
    path: Path,
    document: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    schedule_path: Path,
    preflight_paths: Sequence[Path],
    tier1_journal: Mapping[str, Any],
    tier1_journal_path: Path,
    repo_root: Path = REPO_ROOT,
) -> Path:
    validate_session_authority(
        session,
        schedule=schedule,
        schedule_path=schedule_path,
        preflight_paths=preflight_paths,
        repo_root=repo_root,
    )
    validate_tier2_activation(
        document,
        schedule=schedule,
        tier1_journal=tier1_journal,
        journal_path=tier1_journal_path,
        repo_root=repo_root,
    )
    expected = Path(session["namespace"]["tier2_root"]) / "activation.json"
    absolute = _require_publication_path(path, expected, kind="tier2")
    return _publish_no_replace(
        absolute,
        document,
        kind="tier2",
        validation_capability=_PUBLICATION_VALIDATED,
    )


def _publish_fixture_tier2_activation(
    path: Path,
    document: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    schedule: Mapping[str, Any],
    tier1_journal: Mapping[str, Any],
    tier1_journal_path: Path,
) -> Path:
    if session.get("trust_mode") != FIXTURE_TRUST_MODE:
        _fail("M6_LIVE_FIXTURE_ONLY", "fixture Tier-2 publisher")
    validate_tier2_activation(
        document,
        schedule=schedule,
        tier1_journal=tier1_journal,
        journal_path=tier1_journal_path,
    )
    expected = Path(session["namespace"]["tier2_root"]) / "activation.json"
    absolute = _require_publication_path(path, expected, kind="tier2")
    return _publish_no_replace(
        absolute,
        document,
        kind="tier2",
        validation_capability=_PUBLICATION_VALIDATED,
    )


__all__ = [
    "M6LiveError",
    "SCHEMA_PATHS",
    "SCHEMA_SHA256",
    "GENERIC_DRIVER_VERSION",
    "TASK17_ONLY_DRIVER_VERSION",
    "build_session_authority",
    "validate_session_authority",
    "validate_campaign_inputs",
    "build_initial_ledger_receipt",
    "validate_ledger_receipt",
    "build_process_identity",
    "build_attempt_receipt",
    "append_attempt_to_ledger",
    "begin_resume_epoch",
    "derive_tier2_activation",
    "validate_tier2_activation",
    "activate_tier2_ledger",
    "plan_next_slot",
    "validate_slot_plan",
    "publish_slot_plan",
    "publish_session_authority",
    "publish_attempt_receipt",
    "publish_ledger_receipt",
    "publish_tier2_activation",
    "load_schema",
    "validate_schema",
]
