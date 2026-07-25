"""Pure contracts for persisted NCU launch-set inventory and metric models."""

from __future__ import annotations

import copy
import json
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_action_executor
from tools.local_labels import ncu_capture_contract
from tools.local_labels import ncu_details_export_receipt
from tools.local_labels import parse_ncu_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
INVENTORY_SCHEMA_PATH = SCHEMA_DIR / "ncu_launch_inventory.schema.json"
METRICS_SCHEMA_PATH = SCHEMA_DIR / "ncu_launch_metrics.schema.json"
EXECUTION_IDENTITY_SCHEMA_PATH = (
    SCHEMA_DIR / "ncu_launch_execution_identity.schema.json"
)
CAPTURE_MANIFEST_SCHEMA_PATH = (
    SCHEMA_DIR / "ncu_launch_set_capture_manifest.schema.json"
)
INVENTORY_SCHEMA_ID = "simbench_bw/schemas/ncu_launch_inventory.schema.json"
METRICS_SCHEMA_ID = "simbench_bw/schemas/ncu_launch_metrics.schema.json"
EXECUTION_IDENTITY_SCHEMA_ID = (
    "simbench_bw/schemas/ncu_launch_execution_identity.schema.json"
)
CAPTURE_MANIFEST_SCHEMA_ID = (
    "simbench_bw/schemas/ncu_launch_set_capture_manifest.schema.json"
)
_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
INVENTORY_SCHEMA_SHA256 = (
    "9ea8f9da467f3f8db5521cf6d31926b95576f33e17e750a559d2e7d06074d04d"
)
METRICS_SCHEMA_SHA256 = (
    "8357781adca524bb0eedc899c24cba42f91aeec83fd46b27ce10042b4c7ced08"
)
EXECUTION_IDENTITY_SCHEMA_SHA256 = (
    "0003ae56c43843ee637818f7132950dae220d291f4db6cd9e4c60a79aeaf5ef3"
)
CAPTURE_MANIFEST_SCHEMA_SHA256 = (
    "e55c8b65c3f006941d243fe9aaba59427e75a6587f9f90f2a49e78f1965d32e7"
)

EXECUTION_IDENTITY_KIND = "ncu_launch_set_execution_identity_v1"
PROCESS_CONTEXT_STREAM_POLICY = (
    "report_local_first_occurrence_equivalence_v1"
)

IDENTITY_COLUMNS = (
    "ID", "Process ID", "Context", "Stream", "Device", "Kernel Name",
    "Block Size", "Grid Size",
)
_EXPLICIT_PROTOCOL_FIELDS = frozenset({
    "ncu_target_processes",
    "ncu_filter_mode",
    "ncu_replay_mode",
    "ncu_cache_control",
    "ncu_clock_control",
    "ncu_profile_from_start",
    "ncu_launch_skip",
    "ncu_launch_count",
})
_INVENTORY_FIELDS = frozenset({
    "schema_version", "kind", "kernel_name_base", "requested_launch_skip",
    "requested_launch_count", "insertion_sentinel", "ncu_controls",
    "report_sha256", "identity_columns", "launch_count", "launches",
    "launch_inventory_sha256", "observation_sha256",
})
_LAUNCH_FIELDS = frozenset({
    "report_launch_ordinal", "identity", "identity_sha256",
})
_METRIC_ROW_FIELDS = frozenset({
    "report_launch_ordinal", "report_id", "fraction_values",
    "unit_tagged_metrics", "parser_diagnostics",
})
_EXECUTION_IDENTITY_FIELDS = frozenset({
    "schema_version", "kind", "executable_sha256",
    "runtime_module_sha256s", "process_context_stream_policy",
    "identity_sha256",
})
_CAPTURE_MANIFEST_FIELDS = frozenset({
    "command", "ncu_controls", "ncu_launch_observation_sha256",
    "ncu_execution_identity_sha256",
})
_DIAGNOSTIC_WINDOWS = {
    "ncu_sector_hit_rate_clamped_to_100": lambda value: 100.0 < value <= 105.0,
    "ncu_lts_pct_overcount_raw_counter_ratio_used": lambda value: value > 105.0,
}
_SCHEMA_SPECS = {
    "inventory": {
        "path": INVENTORY_SCHEMA_PATH,
        "id": INVENTORY_SCHEMA_ID,
        "sha256": INVENTORY_SCHEMA_SHA256,
        "root_type": "object",
    },
    "metrics": {
        "path": METRICS_SCHEMA_PATH,
        "id": METRICS_SCHEMA_ID,
        "sha256": METRICS_SCHEMA_SHA256,
        "root_type": "array",
    },
    "execution_identity": {
        "path": EXECUTION_IDENTITY_SCHEMA_PATH,
        "id": EXECUTION_IDENTITY_SCHEMA_ID,
        "sha256": EXECUTION_IDENTITY_SCHEMA_SHA256,
        "root_type": "object",
    },
    "capture_manifest": {
        "path": CAPTURE_MANIFEST_SCHEMA_PATH,
        "id": CAPTURE_MANIFEST_SCHEMA_ID,
        "sha256": CAPTURE_MANIFEST_SCHEMA_SHA256,
        "root_type": "object",
    },
}


@dataclass(frozen=True)
class DetailsExportProducts:
    """One pure, mutually bound Details export product set."""

    csv_bytes: bytes
    stderr_bytes: bytes
    inventory_document: dict[str, Any]
    inventory_bytes: bytes
    metrics_document: list[dict[str, Any]]
    metrics_bytes: bytes
    receipt_document: dict[str, Any]
    receipt_bytes: bytes


class LaunchDocumentError(ValueError):
    """A launch-set schema, model, seal, or cross-document contract failed."""


def _fail(message: str) -> None:
    raise LaunchDocumentError(message)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest(value: Any) -> str:
    try:
        return immutable.sha256_bytes(immutable.canonical_json_bytes(value))
    except immutable.ImmutableArtifactError as exc:
        raise LaunchDocumentError(str(exc)) from exc


def _require_canonical_json(value: Any) -> None:
    _digest(value)


def load_schema(
    kind: str, *, schema_path: Path | None = None,
    payload: bytes | None = None,
) -> dict[str, Any]:
    """Load one byte-pinned launch schema through its canonical authority."""
    spec = _SCHEMA_SPECS.get(kind)
    if spec is None:
        _fail(f"unsupported NCU launch schema kind: {kind!r}")
    expected_path = spec["path"]
    path = expected_path if schema_path is None else Path(schema_path)
    if path != expected_path:
        _fail(f"{kind} schema path is not canonical: {path}")
    if payload is None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise LaunchDocumentError(
                f"{kind} schema authority is unavailable: {exc}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            _fail(f"{kind} schema authority is not a regular file")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise LaunchDocumentError(
                f"{kind} schema authority is unreadable: {exc}"
            ) from exc
    if immutable.sha256_bytes(payload) != spec["sha256"]:
        _fail(f"{kind} schema SHA-256 mismatch")
    try:
        schema = immutable.strict_json_loads(
            payload, label=f"{kind} launch schema",
        )
    except immutable.ImmutableArtifactError as exc:
        raise LaunchDocumentError(str(exc)) from exc
    if type(schema) is not dict:
        _fail(f"{kind} schema root is not an object")
    if schema.get("$schema") != _SCHEMA_DRAFT:
        _fail(f"{kind} schema draft mismatch")
    if schema.get("$id") != spec["id"]:
        _fail(f"{kind} schema $id mismatch")
    if schema.get("type") != spec["root_type"]:
        _fail(f"{kind} schema root type mismatch")
    if kind == "inventory":
        properties = schema.get("properties")
        if (
            type(properties) is not dict
            or properties.get("schema_version", {}).get("const") != 3
            or properties.get("kind", {}).get("const")
            != "ncu_details_csv_launch_inventory_v3"
        ):
            _fail("inventory schema version or kind mismatch")
    elif kind == "execution_identity":
        properties = schema.get("properties")
        if (
            type(properties) is not dict
            or properties.get("schema_version", {}).get("const") != 1
            or properties.get("kind", {}).get("const")
            != EXECUTION_IDENTITY_KIND
            or properties.get("process_context_stream_policy", {}).get("const")
            != PROCESS_CONTEXT_STREAM_POLICY
        ):
            _fail("execution identity schema constants mismatch")
    elif kind == "capture_manifest":
        properties = schema.get("properties")
        definitions = schema.get("$defs")
        controls = (
            definitions.get("ncuControls")
            if type(definitions) is dict else None
        )
        control_properties = (
            controls.get("properties") if type(controls) is dict else None
        )
        command_schema = (
            properties.get("command") if type(properties) is dict else None
        )
        root_required = schema.get("required")
        controls_required = (
            controls.get("required") if type(controls) is dict else None
        )
        expected_controls = {
            "target_processes": "all",
            "filter_mode": "global",
            "replay_mode": "kernel",
            "cache_control": "all",
            "clock_control": "none",
            "profile_from_start": False,
        }
        if (
            type(properties) is not dict
            or set(properties) != _CAPTURE_MANIFEST_FIELDS
            or type(root_required) is not list
            or any(type(name) is not str for name in root_required)
            or set(root_required) != _CAPTURE_MANIFEST_FIELDS
            or schema.get("additionalProperties") is not False
            or type(command_schema) is not dict
            or command_schema.get("type") != "array"
            or command_schema.get("minItems") != 1
            or command_schema.get("maxItems") != 4096
            or command_schema.get("items") != {"type": "string"}
            or type(controls) is not dict
            or controls.get("additionalProperties") is not False
            or type(controls_required) is not list
            or any(type(name) is not str for name in controls_required)
            or set(controls_required) != set(expected_controls)
            or type(control_properties) is not dict
            or set(control_properties) != set(expected_controls)
            or any(
                type(control_properties[name]) is not dict
                or type(control_properties[name].get("const")) is not type(value)
                or control_properties[name].get("const") != value
                for name, value in expected_controls.items()
            )
        ):
            _fail("capture manifest schema contract mismatch")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise LaunchDocumentError(
            f"{kind} launch schema is invalid: {exc}"
        ) from exc
    return schema


def validate_schema(kind: str, value: Any) -> None:
    """Validate a model against one explicitly loaded byte-pinned schema."""
    validator = Draft202012Validator(load_schema(kind))
    errors = sorted(
        validator.iter_errors(value), key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/" + "/".join(
            str(component) for component in error.absolute_path
        )
        _fail(f"{kind} schema validation failed at {location}: {error.message}")


def _require_expected_launch_count(expected_launch_count: int) -> int:
    if (
        type(expected_launch_count) is not int
        or not 1 <= expected_launch_count < ncu_capture_contract.MAX_NCU_SELECTOR
    ):
        _fail("expected launch count is outside the Q809 selector range")
    return expected_launch_count


def _require_runtime_module_sha256s(
    values: Iterable[str], *, expected_launch_count: int,
) -> tuple[str, ...]:
    count = _require_expected_launch_count(expected_launch_count)
    if type(values) not in {list, tuple}:
        _fail("runtime module SHA-256 authority is not an ordered list or tuple")
    if len(values) != count:
        _fail("runtime module SHA-256 cardinality differs from launch count")
    hashes: list[str] = []
    for index, value in enumerate(values):
        if not _is_sha256(value):
            _fail(f"runtime module SHA-256 {index} is not lowercase 64-hex")
        hashes.append(value)
    return tuple(hashes)


def validate_launch_execution_identity_document(
    document: Mapping[str, Any], *, expected_launch_count: int,
    expected_target_executable_sha256: str,
    expected_runtime_module_sha256s: Iterable[str],
) -> dict[str, Any]:
    """Validate one target execution identity against independent authorities.

    ``executable_sha256`` identifies the profiled workload executable, never the
    NCU tool executable carried by :class:`ExpectedNcuIdentity`. This document
    identifies target code in launch order; a later cross-document validator
    must bind that order to one retained NCU inventory and report.
    """

    _require_canonical_json(document)
    count = _require_expected_launch_count(expected_launch_count)
    if not _is_sha256(expected_target_executable_sha256):
        _fail("target executable SHA-256 authority is not lowercase 64-hex")
    expected_modules = _require_runtime_module_sha256s(
        expected_runtime_module_sha256s,
        expected_launch_count=count,
    )
    if type(document) is not dict or set(document) != _EXECUTION_IDENTITY_FIELDS:
        _fail("execution identity root fields mismatch")
    runtime_modules = document.get("runtime_module_sha256s")
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or document.get("kind") != EXECUTION_IDENTITY_KIND
        or document.get("executable_sha256")
        != expected_target_executable_sha256
        or type(runtime_modules) is not list
        or len(runtime_modules) != count
        or runtime_modules != list(expected_modules)
        or document.get("process_context_stream_policy")
        != PROCESS_CONTEXT_STREAM_POLICY
    ):
        _fail("execution identity semantic contract mismatch")
    unsigned = dict(document)
    unsigned.pop("identity_sha256")
    if document.get("identity_sha256") != _digest(unsigned):
        _fail("execution identity_sha256 mismatch")
    return document


def build_launch_execution_identity_document(
    *, target_executable_sha256: str,
    runtime_module_sha256s: Iterable[str], expected_launch_count: int,
) -> dict[str, Any]:
    """Build one workload-target identity without filesystem or process access."""

    count = _require_expected_launch_count(expected_launch_count)
    if not _is_sha256(target_executable_sha256):
        _fail("target executable SHA-256 authority is not lowercase 64-hex")
    modules = _require_runtime_module_sha256s(
        runtime_module_sha256s,
        expected_launch_count=count,
    )
    document = {
        "schema_version": 1,
        "kind": EXECUTION_IDENTITY_KIND,
        "executable_sha256": target_executable_sha256,
        "runtime_module_sha256s": list(modules),
        "process_context_stream_policy": PROCESS_CONTEXT_STREAM_POLICY,
    }
    document["identity_sha256"] = _digest(document)
    validate_launch_execution_identity_document(
        document,
        expected_launch_count=count,
        expected_target_executable_sha256=target_executable_sha256,
        expected_runtime_module_sha256s=modules,
    )
    return document


def _require_capture_manifest_command(command: Iterable[str]) -> list[str]:
    if type(command) not in {list, tuple}:
        _fail("capture manifest command is not an ordered list or tuple")
    if not 1 <= len(command) <= 4096:
        _fail("capture manifest command cardinality is outside the executor bound")
    normalized: list[str] = []
    encoded_size = 0
    for index, argument in enumerate(command):
        if type(argument) is not str or "\x00" in argument:
            _fail(f"capture manifest command argument {index} is invalid")
        try:
            argument_size = len(argument.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as exc:
            raise LaunchDocumentError(
                f"capture manifest command argument {index} is not UTF-8: {exc}"
            ) from exc
        if argument_size > 131071:
            _fail(f"capture manifest command argument {index} exceeds the bound")
        encoded_size += argument_size + 1
        if encoded_size > 1024 * 1024:
            _fail("capture manifest command exceeds the encoded executor bound")
        normalized.append(argument)
    if not normalized[0]:
        _fail("capture manifest command executable is empty")
    return normalized


def validate_launch_set_capture_manifest_document(
    document: Mapping[str, Any], *, expected_command: Iterable[str],
    expected_launch_count: int, protocol: Mapping[str, Any],
    expected_launch_observation_sha256: str,
    expected_execution_identity_sha256: str,
) -> dict[str, Any]:
    """Validate the four-field manifest against independent Q809 authorities.

    The file's canonical presentation bytes and SHA-256 are its external seal;
    this object deliberately has no self-attested manifest seal.
    """

    _require_canonical_json(document)
    command = _require_capture_manifest_command(expected_command)
    controls = _require_q809_protocol(protocol, expected_launch_count)
    if not _is_sha256(expected_launch_observation_sha256):
        _fail("launch observation SHA-256 authority is not lowercase 64-hex")
    if not _is_sha256(expected_execution_identity_sha256):
        _fail("execution identity SHA-256 authority is not lowercase 64-hex")
    if type(document) is not dict or set(document) != _CAPTURE_MANIFEST_FIELDS:
        _fail("capture manifest root fields mismatch")
    observed_command = document.get("command")
    observed_controls = document.get("ncu_controls")
    if (
        type(observed_command) is not list
        or any(type(argument) is not str for argument in observed_command)
        or observed_command != command
        or type(observed_controls) is not dict
        or set(observed_controls) != set(controls)
        or any(
            type(observed_controls[name]) is not type(value)
            or observed_controls[name] != value
            for name, value in controls.items()
        )
        or not _is_sha256(document.get("ncu_launch_observation_sha256"))
        or document.get("ncu_launch_observation_sha256")
        != expected_launch_observation_sha256
        or not _is_sha256(document.get("ncu_execution_identity_sha256"))
        or document.get("ncu_execution_identity_sha256")
        != expected_execution_identity_sha256
    ):
        _fail("capture manifest semantic contract mismatch")
    return document


def build_launch_set_capture_manifest_document(
    *, expected_command: Iterable[str], expected_launch_count: int,
    protocol: Mapping[str, Any], launch_observation_sha256: str,
    execution_identity_sha256: str,
) -> dict[str, Any]:
    """Build one exact launch-set manifest without filesystem or process access."""

    command = _require_capture_manifest_command(expected_command)
    controls = _require_q809_protocol(protocol, expected_launch_count)
    document = {
        "command": command,
        "ncu_controls": controls,
        "ncu_launch_observation_sha256": launch_observation_sha256,
        "ncu_execution_identity_sha256": execution_identity_sha256,
    }
    validate_launch_set_capture_manifest_document(
        document,
        expected_command=command,
        expected_launch_count=expected_launch_count,
        protocol=protocol,
        expected_launch_observation_sha256=launch_observation_sha256,
        expected_execution_identity_sha256=execution_identity_sha256,
    )
    return document


def launch_set_capture_manifest_bytes(
    document: Mapping[str, Any], *, expected_command: Iterable[str],
    expected_launch_count: int, protocol: Mapping[str, Any],
    expected_launch_observation_sha256: str,
    expected_execution_identity_sha256: str,
) -> bytes:
    """Return the canonical presentation bytes required by DONE verification."""

    validated = validate_launch_set_capture_manifest_document(
        document,
        expected_command=expected_command,
        expected_launch_count=expected_launch_count,
        protocol=protocol,
        expected_launch_observation_sha256=expected_launch_observation_sha256,
        expected_execution_identity_sha256=expected_execution_identity_sha256,
    )
    try:
        payload = json.dumps(
            validated,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise LaunchDocumentError(
            f"capture manifest is not canonical presentation JSON: {exc}"
        ) from exc
    return (payload + "\n").encode("utf-8")


def _require_metric_request(
    values: Iterable[str], *, family: str,
) -> tuple[str, ...]:
    requested = tuple(values)
    if (
        not requested
        or any(type(name) is not str or not name for name in requested)
        or len(set(requested)) != len(requested)
    ):
        _fail(f"invalid or duplicate {family} metric request")
    return requested


def _require_q809_protocol(
    protocol: Mapping[str, Any], expected_launch_count: int,
) -> dict[str, Any]:
    if (
        type(expected_launch_count) is not int
        or not 1 <= expected_launch_count
        < ncu_capture_contract.MAX_NCU_SELECTOR
        or type(protocol) is not dict
        or set(protocol) != _EXPLICIT_PROTOCOL_FIELDS
    ):
        _fail("invalid Q809 launch-set protocol shape")
    try:
        ncu_capture_contract.validate_measurement_protocol(protocol)
    except ValueError as exc:
        raise LaunchDocumentError(str(exc)) from exc
    expected = {
        "ncu_target_processes": "all",
        "ncu_filter_mode": "global",
        "ncu_replay_mode": "kernel",
        "ncu_cache_control": "all",
        "ncu_clock_control": "none",
        "ncu_profile_from_start": False,
        "ncu_launch_skip": 0,
        "ncu_launch_count": expected_launch_count + 1,
    }
    if protocol != expected:
        _fail("Q809 launch-set protocol values mismatch")
    return {
        "target_processes": "all",
        "filter_mode": "global",
        "replay_mode": "kernel",
        "cache_control": "all",
        "clock_control": "none",
        "profile_from_start": False,
    }


def _protocol_from_inventory(document: Mapping[str, Any]) -> dict[str, Any]:
    controls = document.get("ncu_controls")
    if type(controls) is not dict:
        _fail("inventory ncu_controls is not an object")
    return {
        "ncu_target_processes": controls.get("target_processes"),
        "ncu_filter_mode": controls.get("filter_mode"),
        "ncu_replay_mode": controls.get("replay_mode"),
        "ncu_cache_control": controls.get("cache_control"),
        "ncu_clock_control": controls.get("clock_control"),
        "ncu_profile_from_start": controls.get("profile_from_start"),
        "ncu_launch_skip": document.get("requested_launch_skip"),
        "ncu_launch_count": document.get("requested_launch_count"),
    }


def validate_launch_inventory_document(
    document: Mapping[str, Any], *, expected_launch_count: int,
    report_sha256: str, protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate the full semantic and seal contract of one inventory v3."""
    _require_canonical_json(document)
    expected_controls = _require_q809_protocol(protocol, expected_launch_count)
    if not _is_sha256(report_sha256):
        _fail("report_sha256 is not lowercase 64-hex")
    if type(document) is not dict or set(document) != _INVENTORY_FIELDS:
        _fail("inventory root fields mismatch")
    launches = document.get("launches")
    sentinel = document.get("insertion_sentinel")
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 3
        or document.get("kind") != "ncu_details_csv_launch_inventory_v3"
        or document.get("kernel_name_base") != "mangled"
        or type(document.get("requested_launch_skip")) is not int
        or document["requested_launch_skip"] != 0
        or type(document.get("requested_launch_count")) is not int
        or document["requested_launch_count"] != expected_launch_count + 1
        or type(document.get("launch_count")) is not int
        or document["launch_count"] != expected_launch_count
        or document.get("report_sha256") != report_sha256
        or document.get("identity_columns") != list(IDENTITY_COLUMNS)
        or document.get("ncu_controls") != expected_controls
        or type(sentinel) is not dict
        or sentinel != {
            "policy": "request_expected_plus_one_reject_any_extra",
            "expected_launch_count": expected_launch_count,
            "requested_launch_count": expected_launch_count + 1,
        }
        or type(launches) is not list
        or len(launches) != expected_launch_count
    ):
        _fail("inventory semantic contract mismatch")
    if document.get("launch_inventory_sha256") != _digest(launches):
        _fail("launch_inventory_sha256 mismatch")
    unsigned = dict(document)
    unsigned.pop("observation_sha256")
    if document.get("observation_sha256") != _digest(unsigned):
        _fail("observation_sha256 mismatch")

    report_ids: list[str] = []
    for ordinal, row in enumerate(launches):
        if type(row) is not dict or set(row) != _LAUNCH_FIELDS:
            _fail(f"inventory launch row {ordinal} fields mismatch")
        identity = row.get("identity")
        if (
            type(row.get("report_launch_ordinal")) is not int
            or row["report_launch_ordinal"] != ordinal
            or type(identity) is not dict
            or set(identity) != set(IDENTITY_COLUMNS)
            or any(
                type(identity.get(name)) is not str
                or not identity[name]
                or identity[name] != identity[name].strip()
                for name in IDENTITY_COLUMNS
            )
        ):
            _fail(f"inventory launch row {ordinal} identity mismatch")
        row_unsigned = dict(row)
        row_unsigned.pop("identity_sha256")
        if row.get("identity_sha256") != _digest(row_unsigned):
            _fail(f"inventory launch row {ordinal} identity_sha256 mismatch")
        report_ids.append(identity["ID"])
    if len(set(report_ids)) != len(report_ids):
        _fail("inventory report IDs are not unique")
    return launches


def build_launch_inventory_document(
    *, csv_text: str, report_sha256: str, expected_launch_count: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one inventory model without reading or writing any artifact."""
    controls = _require_q809_protocol(protocol, expected_launch_count)
    if not _is_sha256(report_sha256):
        _fail("report_sha256 is not lowercase 64-hex")
    observation = parse_ncu_metrics.extract_launch_observations(
        csv_text,
        expected_launch_count=expected_launch_count,
        requested_launch_skip=protocol["ncu_launch_skip"],
        requested_launch_count=protocol["ncu_launch_count"],
    )
    launches: list[dict[str, Any]] = []
    for launch in observation["launches"]:
        row = {
            "report_launch_ordinal": launch["report_launch_ordinal"],
            "identity": copy.deepcopy(launch["identity"]),
        }
        row["identity_sha256"] = _digest(row)
        launches.append(row)
    document = {
        "schema_version": 3,
        "kind": "ncu_details_csv_launch_inventory_v3",
        "kernel_name_base": "mangled",
        "requested_launch_skip": 0,
        "requested_launch_count": expected_launch_count + 1,
        "insertion_sentinel": {
            "policy": "request_expected_plus_one_reject_any_extra",
            "expected_launch_count": expected_launch_count,
            "requested_launch_count": expected_launch_count + 1,
        },
        "ncu_controls": controls,
        "report_sha256": report_sha256,
        "identity_columns": list(IDENTITY_COLUMNS),
        "launch_count": expected_launch_count,
        "launches": launches,
        "launch_inventory_sha256": _digest(launches),
    }
    document["observation_sha256"] = _digest(document)
    validate_launch_inventory_document(
        document,
        expected_launch_count=expected_launch_count,
        report_sha256=report_sha256,
        protocol=protocol,
    )
    return document


def _validated_inventory_context(
    inventory: Mapping[str, Any],
) -> tuple[int, str, dict[str, Any], list[dict[str, Any]]]:
    if type(inventory) is not dict:
        _fail("inventory is not an object")
    count = inventory.get("launch_count")
    report_sha256 = inventory.get("report_sha256")
    protocol = _protocol_from_inventory(inventory)
    launches = validate_launch_inventory_document(
        inventory,
        expected_launch_count=count,
        report_sha256=report_sha256,
        protocol=protocol,
    )
    return count, report_sha256, protocol, launches


def _metric_entry(registry: Any, name: str) -> Any:
    try:
        entry = registry.get_by_ncu_metric(name)
    except Exception as exc:
        raise LaunchDocumentError(
            f"metric registry lookup failed for {name!r}: {exc}"
        ) from exc
    if entry is None:
        _fail(f"metric {name!r} is not registered")
    return entry


def _validated_metric_requests(
    registry: Any, percentages: tuple[str, ...], unit_metrics: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    percentage_entries = {
        name: _metric_entry(registry, name) for name in percentages
    }
    for name, entry in percentage_entries.items():
        if (
            not entry.is_flat_field
            or entry.parser_transform != "percent_to_fraction"
            or entry.unit != "pct"
        ):
            _fail(
                f"percentage metric {name!r} lacks flat percentage "
                "registry semantics"
            )
    unit_entries = {
        name: _metric_entry(registry, name) for name in unit_metrics
    }
    for name, entry in unit_entries.items():
        if not entry.is_unit_tagged:
            _fail(f"unit metric {name!r} lacks unit-tagged registry semantics")
        if entry.unit not in {"pct", "count", "ns", "cycles"}:
            _fail(f"unit metric {name!r} uses an unsupported registry unit")
    return percentage_entries, unit_entries


def _require_number(value: Any, *, label: str) -> float | int:
    if type(value) not in {int, float}:
        _fail(f"{label} is not a strict JSON number")
    _require_canonical_json(value)
    return value


def _require_registry_range(entry: Any, value: float | int, *, label: str) -> None:
    if value < entry.valid_min or (
        entry.valid_max is not None and value > entry.valid_max
    ):
        _fail(
            f"{label} is outside registry range "
            f"[{entry.valid_min}, {entry.valid_max}]"
        )


def _source_launches(
    csv_text: str, *, expected_count: int, protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        observation = parse_ncu_metrics.extract_launch_observations(
            csv_text,
            expected_launch_count=expected_count,
            requested_launch_skip=protocol["ncu_launch_skip"],
            requested_launch_count=protocol["ncu_launch_count"],
        )
    except ValueError as exc:
        raise LaunchDocumentError(
            f"Details CSV launch inventory is invalid: {exc}"
        ) from exc
    return observation["launches"]


def validate_launch_metrics_document(
    document: Any, *, csv_text: str, inventory: Mapping[str, Any], registry: Any,
    percentage_metric_names: Iterable[str], unit_metric_names: Iterable[str],
) -> list[dict[str, Any]]:
    """Validate ordered launch metrics against inventory and registry authority."""
    _require_canonical_json(document)
    expected_count, _report, protocol, launches = _validated_inventory_context(
        inventory,
    )
    percentages = _require_metric_request(
        percentage_metric_names, family="percentage",
    )
    unit_metrics = _require_metric_request(
        unit_metric_names, family="unit-tagged",
    )
    percentage_set = set(percentages)
    unit_set = set(unit_metrics)
    percentage_entries, unit_entries = _validated_metric_requests(
        registry, percentages, unit_metrics,
    )
    percentage_capable = set(percentages)
    percentage_capable.update(
        name for name, entry in unit_entries.items()
        if entry.parser_transform == "percent_to_fraction"
    )
    if type(document) is not list or len(document) != expected_count:
        _fail("launch metrics cardinality mismatch")

    source_launches = _source_launches(
        csv_text, expected_count=expected_count, protocol=protocol,
    )
    for ordinal, source_launch in enumerate(source_launches):
        if (
            source_launch.get("report_launch_ordinal") != ordinal
            or source_launch.get("identity") != launches[ordinal]["identity"]
        ):
            _fail(
                f"Details CSV launch {ordinal} identity does not match inventory"
            )

    for ordinal, row in enumerate(document):
        if type(row) is not dict or set(row) != _METRIC_ROW_FIELDS:
            _fail(f"launch metrics row {ordinal} fields mismatch")
        expected_id = launches[ordinal]["identity"]["ID"]
        if (
            type(row.get("report_launch_ordinal")) is not int
            or row["report_launch_ordinal"] != ordinal
            or type(row.get("report_id")) is not str
            or row["report_id"] != expected_id
        ):
            _fail(f"launch metrics row {ordinal} identity mismatch")
        fractions = row.get("fraction_values")
        if type(fractions) is not dict or set(fractions) != percentage_set:
            _fail(f"launch metrics row {ordinal} percentage keys mismatch")
        for name, value in fractions.items():
            number = _require_number(
                value, label=f"launch {ordinal} percentage {name}",
            )
            _require_registry_range(
                percentage_entries[name], number,
                label=f"launch {ordinal} percentage {name!r}",
            )

        tagged = row.get("unit_tagged_metrics")
        if type(tagged) is not dict or set(tagged) != unit_set:
            _fail(f"launch metrics row {ordinal} unit metric keys mismatch")
        for name, record in tagged.items():
            entry = unit_entries[name]
            if (
                type(record) is not dict
                or set(record) != {"unit", "value"}
                or record.get("unit") != entry.unit
            ):
                _fail(f"launch {ordinal} unit metric {name!r} shape or unit mismatch")
            number = _require_number(
                record["value"], label=f"launch {ordinal} unit metric {name}",
            )
            _require_registry_range(
                entry, number,
                label=f"launch {ordinal} unit metric {name!r}",
            )

        diagnostics = row.get("parser_diagnostics")
        if (
            type(diagnostics) is not dict
            or not set(diagnostics).issubset(_DIAGNOSTIC_WINDOWS)
        ):
            _fail(f"launch metrics row {ordinal} diagnostics shape mismatch")
        for diagnostic_name, records in diagnostics.items():
            if type(records) is not list or not records:
                _fail(f"launch {ordinal} diagnostic {diagnostic_name!r} is empty")
            seen: set[tuple[str, float | int]] = set()
            for record in records:
                if type(record) is not dict or set(record) != {
                    "metric_label", "raw_value",
                }:
                    _fail(f"launch {ordinal} diagnostic record shape mismatch")
                metric = record.get("metric_label")
                raw_value = _require_number(
                    record.get("raw_value"),
                    label=f"launch {ordinal} diagnostic raw value",
                )
                if type(metric) is not str:
                    _fail(f"launch {ordinal} diagnostic metric label is invalid")
                key = (metric, raw_value)
                if (
                    metric not in percentage_capable
                    or key in seen
                    or not _DIAGNOSTIC_WINDOWS[diagnostic_name](raw_value)
                ):
                    _fail(f"launch {ordinal} diagnostic record is invalid")
                normalized_values: list[float | int] = []
                if metric in fractions:
                    normalized_values.append(fractions[metric])
                if metric in tagged:
                    normalized_values.append(tagged[metric]["value"])
                if not normalized_values:
                    _fail(f"launch {ordinal} diagnostic metric is not retained")
                if (
                    diagnostic_name == "ncu_sector_hit_rate_clamped_to_100"
                    and any(value != 1.0 for value in normalized_values)
                ):
                    _fail(f"launch {ordinal} clamp diagnostic value mismatch")
                if (
                    diagnostic_name
                    == "ncu_lts_pct_overcount_raw_counter_ratio_used"
                    and metric not in parse_ncu_metrics._PCT_TO_RAW_COUNTER_PAIR
                ):
                    _fail(f"launch {ordinal} fallback diagnostic metric mismatch")
                seen.add(key)

    try:
        expected_document = parse_ncu_metrics.extract_launch_metric_observations(
            csv_text,
            registry,
            unit_metrics,
            expected_launch_count=expected_count,
            requested_launch_skip=protocol["ncu_launch_skip"],
            requested_launch_count=protocol["ncu_launch_count"],
            percentage_metric_names=percentages,
        )
    except ValueError as exc:
        raise LaunchDocumentError(
            f"Details CSV launch metrics are invalid: {exc}"
        ) from exc
    if _digest(document) != _digest(expected_document):
        _fail("launch metrics do not match Details CSV extraction")
    return document


def build_launch_metrics_document(
    *, csv_text: str, inventory: Mapping[str, Any], registry: Any,
    percentage_metric_names: Iterable[str], unit_metric_names: Iterable[str],
) -> list[dict[str, Any]]:
    """Build ordered launch metrics without reading or writing any artifact."""
    expected_count, _report, protocol, _launches = _validated_inventory_context(
        inventory,
    )
    percentages = _require_metric_request(
        percentage_metric_names, family="percentage",
    )
    unit_metrics = _require_metric_request(
        unit_metric_names, family="unit-tagged",
    )
    _validated_metric_requests(registry, percentages, unit_metrics)
    document = parse_ncu_metrics.extract_launch_metric_observations(
        csv_text,
        registry,
        unit_metrics,
        expected_launch_count=expected_count,
        requested_launch_skip=protocol["ncu_launch_skip"],
        requested_launch_count=protocol["ncu_launch_count"],
        percentage_metric_names=percentages,
    )
    validate_launch_metrics_document(
        document,
        csv_text=csv_text,
        inventory=inventory,
        registry=registry,
        percentage_metric_names=percentages,
        unit_metric_names=unit_metrics,
    )
    return document


def build_details_export_products(
    *,
    imported: parse_ncu_metrics.ImportedReportCsv,
    preflight_sha256: str,
    source_commit: str,
    capture_identity_sha256: str,
    expected_ncu_identity: ncu_action_executor.ExpectedNcuIdentity,
    report_relative_path: str,
    csv_relative_path: str,
    stderr_relative_path: str,
    inventory_relative_path: str,
    metrics_relative_path: str,
    expected_launch_count: int,
    protocol: Mapping[str, Any],
    registry: Any,
    percentage_metric_names: Iterable[str],
    unit_metric_names: Iterable[str],
) -> DetailsExportProducts:
    """Build one byte-exact Details product set without filesystem access."""

    if type(imported) is not parse_ncu_metrics.ImportedReportCsv:
        _fail("Details export import has the wrong authority type")
    if (
        type(imported.csv_bytes) is not bytes
        or not imported.csv_bytes
        or type(imported.text) is not str
    ):
        _fail("Details export CSV bytes are absent")
    try:
        decoded_csv = imported.csv_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise LaunchDocumentError(
            f"Details export CSV bytes are not strict UTF-8: {exc}"
        ) from exc
    if decoded_csv != imported.text:
        _fail("Details export CSV text differs from retained normalized bytes")
    if type(imported.stderr_bytes) is not bytes:
        _fail("Details export stderr is not exact bytes")
    if imported.ncu_identity != expected_ncu_identity:
        _fail("Details export NCU identity differs from independent authority")
    if imported.normalized_arguments != tuple(
        ncu_details_export_receipt.EXPECTED_IMPORT_ARGUMENTS
    ):
        _fail("Details export normalized arguments differ from authority")
    if imported.action_environment != tuple(
        ncu_details_export_receipt.EXPECTED_ACTION_ENVIRONMENT
    ):
        _fail("Details export action environment differs from authority")

    inventory = build_launch_inventory_document(
        csv_text=imported.text,
        report_sha256=imported.report_sha256,
        expected_launch_count=expected_launch_count,
        protocol=protocol,
    )
    metrics = build_launch_metrics_document(
        csv_text=imported.text,
        inventory=inventory,
        registry=registry,
        percentage_metric_names=percentage_metric_names,
        unit_metric_names=unit_metric_names,
    )
    try:
        inventory_bytes = immutable.pretty_json_bytes(inventory)
        metrics_bytes = immutable.pretty_json_bytes(metrics)
    except immutable.ImmutableArtifactError as exc:
        raise LaunchDocumentError(str(exc)) from exc
    try:
        products = [
            ncu_details_export_receipt.product_binding(
                role="ncu_report",
                relative_path=report_relative_path,
                sha256=imported.report_sha256,
                size_bytes=imported.report_size_bytes,
            ),
            ncu_details_export_receipt.product_from_bytes(
                role="details_csv",
                relative_path=csv_relative_path,
                payload=imported.csv_bytes,
            ),
            ncu_details_export_receipt.product_from_bytes(
                role="details_stderr",
                relative_path=stderr_relative_path,
                payload=imported.stderr_bytes,
            ),
            ncu_details_export_receipt.product_from_bytes(
                role="launch_inventory",
                relative_path=inventory_relative_path,
                payload=inventory_bytes,
            ),
            ncu_details_export_receipt.product_from_bytes(
                role="launch_metrics",
                relative_path=metrics_relative_path,
                payload=metrics_bytes,
            ),
        ]
        receipt = ncu_details_export_receipt.build_receipt(
            preflight_sha256=preflight_sha256,
            source_commit=source_commit,
            capture_identity_sha256=capture_identity_sha256,
            expected_ncu_identity=expected_ncu_identity,
            import_arguments=imported.normalized_arguments,
            environment=dict(imported.action_environment),
            working_directory=imported.working_directory,
            exit_status=imported.exit_status,
            products=products,
        )
        receipt_bytes = immutable.pretty_json_bytes(receipt)
    except (
        immutable.ImmutableArtifactError,
        ncu_details_export_receipt.DetailsExportReceiptError,
    ) as exc:
        raise LaunchDocumentError(str(exc)) from exc
    return DetailsExportProducts(
        csv_bytes=imported.csv_bytes,
        stderr_bytes=imported.stderr_bytes,
        inventory_document=inventory,
        inventory_bytes=inventory_bytes,
        metrics_document=metrics,
        metrics_bytes=metrics_bytes,
        receipt_document=receipt,
        receipt_bytes=receipt_bytes,
    )
