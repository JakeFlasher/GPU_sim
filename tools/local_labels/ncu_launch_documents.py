"""Pure contracts for persisted NCU launch-set inventory and metric models."""

from __future__ import annotations

import copy
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import ncu_capture_contract
from tools.local_labels import parse_ncu_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
INVENTORY_SCHEMA_PATH = SCHEMA_DIR / "ncu_launch_inventory.schema.json"
METRICS_SCHEMA_PATH = SCHEMA_DIR / "ncu_launch_metrics.schema.json"
INVENTORY_SCHEMA_ID = "simbench_bw/schemas/ncu_launch_inventory.schema.json"
METRICS_SCHEMA_ID = "simbench_bw/schemas/ncu_launch_metrics.schema.json"
_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
INVENTORY_SCHEMA_SHA256 = (
    "9ea8f9da467f3f8db5521cf6d31926b95576f33e17e750a559d2e7d06074d04d"
)
METRICS_SCHEMA_SHA256 = (
    "8357781adca524bb0eedc899c24cba42f91aeec83fd46b27ce10042b4c7ced08"
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
}


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
