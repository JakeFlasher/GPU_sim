"""Pure v5 sidecar composition for one validated Q809 NCU launch set."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import import_label
from tools.local_labels import ncu_capture_contract
from tools.local_labels import ncu_launch_documents


_PERCENTAGE_FIELDS = {
    "lts__t_sector_hit_rate.pct": (
        "cache_hit_rate_l2_sector",
        "confidence_interval_l2_sector",
    ),
    "l1tex__t_sector_hit_rate.pct": (
        "cache_hit_rate_l1tex_sector",
        "confidence_interval_l1tex_sector",
    ),
}


class LaunchSidecarError(ValueError):
    """One launch-set sidecar model failed its bound authority contract."""


@dataclass(frozen=True)
class LaunchSidecarProvenance:
    """Target and host provenance shared by every sidecar in one launch set."""

    device: str
    slug: str
    shape_point: str
    binary_sha256: str
    profiler_version: str
    cuda_version: str
    driver_version: str
    gpu_uuid_sha256: str
    warmup_count: int


def _fail(message: str) -> None:
    raise LaunchSidecarError(message)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return immutable.canonical_json_bytes(value)
    except immutable.ImmutableArtifactError as exc:
        raise LaunchSidecarError(str(exc)) from exc


def _validate_provenance(
    value: LaunchSidecarProvenance,
) -> LaunchSidecarProvenance:
    if type(value) is not LaunchSidecarProvenance:
        _fail("launch sidecar provenance has the wrong authority type")
    for field in (
        "device",
        "slug",
        "shape_point",
        "profiler_version",
        "cuda_version",
        "driver_version",
    ):
        observed = getattr(value, field)
        if type(observed) is not str or not observed or "\x00" in observed:
            _fail(f"launch sidecar provenance {field} is invalid")
    if not _is_sha256(value.binary_sha256):
        _fail("launch sidecar binary SHA-256 is invalid")
    if not _is_sha256(value.gpu_uuid_sha256):
        _fail("launch sidecar GPU UUID SHA-256 is invalid")
    if type(value.warmup_count) is not int or value.warmup_count < 0:
        _fail("launch sidecar warmup count is invalid")
    _canonical_bytes({
        "device": value.device,
        "slug": value.slug,
        "shape_point": value.shape_point,
        "binary_sha256": value.binary_sha256,
        "profiler_version": value.profiler_version,
        "cuda_version": value.cuda_version,
        "driver_version": value.driver_version,
        "gpu_uuid_sha256": value.gpu_uuid_sha256,
        "warmup_count": value.warmup_count,
    })
    return value


def _ordered_metric_names(
    values: Sequence[str], *, family: str,
) -> tuple[str, ...]:
    if type(values) not in {list, tuple}:
        _fail(f"{family} metric authority is not an ordered list or tuple")
    result = tuple(values)
    if (
        not result
        or any(type(name) is not str or not name for name in result)
        or len(set(result)) != len(result)
    ):
        _fail(f"{family} metric authority is empty, malformed, or duplicated")
    return result


def _validate_protocol(
    protocol: Mapping[str, Any], *, expected_launch_count: int,
) -> dict[str, Any]:
    if (
        type(expected_launch_count) is not int
        or not 1 <= expected_launch_count
        < ncu_capture_contract.MAX_NCU_SELECTOR
    ):
        _fail("launch sidecar count is outside the Q809 selector range")
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
    if (
        type(protocol) is not dict
        or set(protocol) != set(expected)
        or any(
            type(protocol[name]) is not type(expected_value)
            or protocol[name] != expected_value
            for name, expected_value in expected.items()
        )
    ):
        _fail("launch sidecar Q809 protocol authority differs")
    try:
        ncu_capture_contract.validate_measurement_protocol(protocol)
    except ValueError as exc:
        raise LaunchSidecarError(str(exc)) from exc
    return copy.deepcopy(protocol)


def _validate_bound_sidecar(
    document: dict[str, Any], *, ordinal: int,
    sidecar_schema: dict[str, Any], registry: Any,
) -> None:
    try:
        validation = import_label.validate_with_bound_authorities(
            document,
            schema=sidecar_schema,
            registry=registry,
        )
    except (TypeError, ValueError) as exc:
        raise LaunchSidecarError(
            f"launch sidecar {ordinal} authorities are invalid: {exc}"
        ) from exc
    if not validation.ok:
        _fail(
            f"launch sidecar {ordinal} is invalid: "
            f"{validation.typed_reason}: {validation.detail}"
        )


def _build_expected_sidecars(
    *, inventory: Mapping[str, Any], launch_metrics: Any,
    csv_text: str, report_sha256: str, expected_launch_count: int,
    protocol: Mapping[str, Any], registry: Any,
    percentage_metric_names: Sequence[str],
    unit_metric_names: Sequence[str], sidecar_schema: dict[str, Any],
    provenance: LaunchSidecarProvenance,
) -> tuple[dict[str, Any], ...]:
    provenance = _validate_provenance(provenance)
    percentages = _ordered_metric_names(
        percentage_metric_names, family="percentage",
    )
    units = _ordered_metric_names(unit_metric_names, family="unit-tagged")
    if set(percentages) != set(_PERCENTAGE_FIELDS):
        _fail("launch sidecars require the exact L2 and L1 percentage metrics")
    if type(csv_text) is not str or not csv_text:
        _fail("launch sidecar Details CSV authority is absent")
    if not _is_sha256(report_sha256):
        _fail("launch sidecar report SHA-256 authority is invalid")
    measurement_protocol = _validate_protocol(
        protocol, expected_launch_count=expected_launch_count,
    )
    try:
        ncu_launch_documents.validate_launch_inventory_document(
            inventory,
            expected_launch_count=expected_launch_count,
            report_sha256=report_sha256,
            protocol=protocol,
        )
        ncu_launch_documents.validate_launch_metrics_document(
            launch_metrics,
            csv_text=csv_text,
            inventory=inventory,
            registry=registry,
            percentage_metric_names=percentages,
            unit_metric_names=units,
        )
    except (TypeError, ValueError) as exc:
        raise LaunchSidecarError(
            f"launch sidecar source documents are invalid: {exc}"
        ) from exc

    documents: list[dict[str, Any]] = []
    for ordinal, metrics in enumerate(launch_metrics):
        fractions = metrics["fraction_values"]
        document: dict[str, Any] = {
            "schema_version": 5,
            "device": provenance.device,
            "slug": provenance.slug,
            "shape_point": provenance.shape_point,
            "repeat_index": 0,
            "launch_index": ordinal,
            "binary_sha256": provenance.binary_sha256,
            "profiler_family": "nsight_compute",
            "profiler_version": provenance.profiler_version,
            "cuda_version": provenance.cuda_version,
            "driver_version": provenance.driver_version,
            "gpu_uuid_sha256": provenance.gpu_uuid_sha256,
            "replay_mode": "kernel",
            "aggregation": "kernel",
            "warmup_count": provenance.warmup_count,
            "measurement_protocol": copy.deepcopy(measurement_protocol),
        }
        for metric_name, (flat_field, interval_field) in (
            _PERCENTAGE_FIELDS.items()
        ):
            value = fractions[metric_name]
            document[flat_field] = value
            document[interval_field] = {
                "lower": value,
                "upper": value,
                "confidence_level": 0.95,
                "units": "sector_hit_rate_fraction",
                "aggregation": "kernel",
            }
        document["unit_tagged_metrics"] = [
            {
                "ncu_metric": metric_name,
                "unit": metrics["unit_tagged_metrics"][metric_name]["unit"],
                "value": metrics["unit_tagged_metrics"][metric_name]["value"],
            }
            for metric_name in units
        ]
        diagnostics = metrics["parser_diagnostics"]
        if diagnostics:
            document["parser_diagnostics"] = copy.deepcopy(diagnostics)
        _canonical_bytes(document)
        _validate_bound_sidecar(
            document,
            ordinal=ordinal,
            sidecar_schema=sidecar_schema,
            registry=registry,
        )
        documents.append(document)
    return tuple(documents)


def build_launch_sidecar_documents(
    *, inventory: Mapping[str, Any], launch_metrics: Any,
    csv_text: str, report_sha256: str, expected_launch_count: int,
    protocol: Mapping[str, Any], registry: Any,
    percentage_metric_names: Sequence[str],
    unit_metric_names: Sequence[str], sidecar_schema: dict[str, Any],
    provenance: LaunchSidecarProvenance,
) -> tuple[dict[str, Any], ...]:
    """Build exactly one bound v5 sidecar per ordered launch, without I/O."""
    return _build_expected_sidecars(
        inventory=inventory,
        launch_metrics=launch_metrics,
        csv_text=csv_text,
        report_sha256=report_sha256,
        expected_launch_count=expected_launch_count,
        protocol=protocol,
        registry=registry,
        percentage_metric_names=percentage_metric_names,
        unit_metric_names=unit_metric_names,
        sidecar_schema=sidecar_schema,
        provenance=provenance,
    )


def validate_launch_sidecar_documents(
    documents: Sequence[Mapping[str, Any]], *,
    inventory: Mapping[str, Any], launch_metrics: Any,
    csv_text: str, report_sha256: str, expected_launch_count: int,
    protocol: Mapping[str, Any], registry: Any,
    percentage_metric_names: Sequence[str],
    unit_metric_names: Sequence[str], sidecar_schema: dict[str, Any],
    provenance: LaunchSidecarProvenance,
) -> tuple[dict[str, Any], ...]:
    """Validate persisted sidecars against independent launch-set authorities."""
    if type(documents) not in {list, tuple}:
        _fail("launch sidecars are not an ordered list or tuple")
    expected = _build_expected_sidecars(
        inventory=inventory,
        launch_metrics=launch_metrics,
        csv_text=csv_text,
        report_sha256=report_sha256,
        expected_launch_count=expected_launch_count,
        protocol=protocol,
        registry=registry,
        percentage_metric_names=percentage_metric_names,
        unit_metric_names=unit_metric_names,
        sidecar_schema=sidecar_schema,
        provenance=provenance,
    )
    if len(documents) != len(expected):
        _fail("launch sidecar cardinality differs from the launch inventory")
    validated: list[dict[str, Any]] = []
    for ordinal, (observed, expected_document) in enumerate(
        zip(documents, expected, strict=True)
    ):
        if type(observed) is not dict:
            _fail(f"launch sidecar {ordinal} root is not an exact object")
        if _canonical_bytes(observed) != _canonical_bytes(expected_document):
            _fail(f"launch sidecar {ordinal} differs from launch-set authority")
        _validate_bound_sidecar(
            observed,
            ordinal=ordinal,
            sidecar_schema=sidecar_schema,
            registry=registry,
        )
        validated.append(observed)
    return tuple(validated)
