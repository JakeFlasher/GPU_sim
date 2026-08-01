"""Strict matched-label admission with an explicit fixture-only authority lane."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from eval.contracts._common import (
    AuthorityBlockerCode,
    EvaluatorAuthorityUnavailable,
    authority_unavailable,
)
from eval.contracts.validation import (
    METRIC_KEYS,
    ContractValidationError,
    MatchedLabelAuthority,
    canonical_launch_key_token,
    canonical_sha256,
    loads_json_strict,
    validate_matched_labels,
)

_SYNTHETIC_AUTHORITY_VERSION = "synthetic_matched_label_authority.v1"
_SYNTHETIC_AUTHORITY_ROLE = "synthetic_fixture_only_not_production"
_AUTHORITY_MEMBERS = frozenset(
    {
        "schema_version",
        "authority_sha256",
        "authority_role",
        "fixture_id",
        "matched_labels_file_sha256",
        "matched_labels_sha256",
        "label_set_id",
        "label_set_sha256",
        "crosswalk_id",
        "crosswalk_sha256",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_AUTHORITY_BYTES = 64 * 1024
_MAX_MATCHED_LABEL_BYTES = 64 * 1024 * 1024
KNOWN_SYNTHETIC_AUTHORITY_FILE_SHA256 = MappingProxyType(
    {
        "evaluator-mvp-canary-v1": (
            "1c0f8257747384785581e8f7ddf83da086ad885ce0f662b67008838254e5d5a3"
        )
    }
)


@dataclass(frozen=True, slots=True)
class SyntheticAuthorityPin:
    """An authority identity supplied independently of fixture contents."""

    fixture_id: str
    authority_file_sha256: str

    def __post_init__(self) -> None:
        if type(self.fixture_id) is not str or not self.fixture_id:
            raise ValueError("synthetic fixture id must be a nonempty string")
        if not _is_sha256(self.authority_file_sha256):
            raise ValueError("synthetic authority file SHA-256 is invalid")


def known_synthetic_authority_pin(fixture_id: str) -> SyntheticAuthorityPin:
    """Return a code-pinned authority identity for a checked-in synthetic fixture."""

    try:
        authority_file_sha256 = KNOWN_SYNTHETIC_AUTHORITY_FILE_SHA256[fixture_id]
    except KeyError as exc:
        raise EvaluatorAuthorityUnavailable(
            AuthorityBlockerCode.LABEL_ADMISSION_AUTHORITY_UNAVAILABLE,
            "synthetic fixture authority is not code-pinned",
        ) from exc
    return SyntheticAuthorityPin(
        fixture_id=fixture_id,
        authority_file_sha256=authority_file_sha256,
    )


@dataclass(frozen=True, slots=True)
class Dim3:
    x: int
    y: int
    z: int


@dataclass(frozen=True, slots=True)
class ReplayLaunchIdentity:
    """The six launch fields authenticated by simulator trace evidence."""

    kernel_mangled_name: str
    grid: Dim3
    block: Dim3
    device_id: int
    stream_id: int
    binary_architecture: str


@dataclass(frozen=True, slots=True)
class CohortLaunchIdentity:
    """The three launch fields asserted by the independent crosswalk."""

    workload_id: str
    matched_execution_id: str
    process_launch_ordinal: int


@dataclass(frozen=True, slots=True)
class LaunchIdentity:
    replay: ReplayLaunchIdentity
    cohort: CohortLaunchIdentity
    canonical_token: str


@dataclass(frozen=True, slots=True)
class Dispersion:
    status: str
    rule: str
    scalar: int | float | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class LabelMetric:
    key: str
    ncu_metric: str
    unit: str
    outcome_status: str
    value: int | float | None
    unavailable_reason: str | None
    repeat_count: int
    aggregation: str
    dispersion: Dispersion


@dataclass(frozen=True, slots=True)
class MatchedLaunch:
    identity: LaunchIdentity
    kernel_family: str
    metrics: tuple[LabelMetric, ...]

    def metric_by_key(self, key: str) -> LabelMetric:
        for metric in self.metrics:
            if metric.key == key:
                return metric
        raise KeyError(key)


@dataclass(frozen=True, slots=True)
class MatchedLabels:
    matched_labels_sha256: str
    label_set_id: str
    label_set_sha256: str
    label_schema_sha256: str
    source_manifest_id: str
    source_manifest_sha256: str
    crosswalk_id: str
    crosswalk_sha256: str
    crosswalk_version: int
    binary_architecture: str
    gpu_sku: str
    launches: tuple[MatchedLaunch, ...]
    synthetic_fixture_id: str

    @property
    def authority(self) -> MatchedLabelAuthority:
        return MatchedLabelAuthority(
            matched_labels_sha256=self.matched_labels_sha256,
            label_set_id=self.label_set_id,
            label_set_sha256=self.label_set_sha256,
            crosswalk_id=self.crosswalk_id,
            crosswalk_sha256=self.crosswalk_sha256,
        )


def load_production_matched_labels(*_args: object, **_kwargs: object) -> MatchedLabels:
    """Refuse before opening any payload while production authority is absent."""

    raise authority_unavailable(
        AuthorityBlockerCode.LABEL_ADMISSION_AUTHORITY_UNAVAILABLE,
        "no production matched-label and crosswalk authority is installed",
    )


def load_synthetic_matched_labels(
    labels_path: str | Path,
    authority_path: str | Path,
    *,
    authority_pin: SyntheticAuthorityPin,
) -> MatchedLabels:
    """Load one independently pinned, explicitly non-production label fixture."""

    authority_bytes = _read_bounded(
        authority_path,
        limit=_MAX_AUTHORITY_BYTES,
        kind="synthetic fixture authority",
    )
    observed_authority_file_sha256 = hashlib.sha256(authority_bytes).hexdigest()
    if observed_authority_file_sha256 != authority_pin.authority_file_sha256:
        raise ContractValidationError(
            "FIXTURE_AUTHORITY_FILE_HASH_MISMATCH",
            "synthetic authority bytes do not match the independent pin",
            context={
                "expected_sha256": authority_pin.authority_file_sha256,
                "observed_sha256": observed_authority_file_sha256,
            },
        )
    authority_document = _decode_object_strict(
        authority_bytes, source=str(authority_path), kind="synthetic fixture authority"
    )
    authority = _validate_synthetic_authority(authority_document, authority_pin)

    labels_bytes = _read_bounded(
        labels_path,
        limit=_MAX_MATCHED_LABEL_BYTES,
        kind="matched-label fixture",
    )
    observed_labels_file_sha256 = hashlib.sha256(labels_bytes).hexdigest()
    if observed_labels_file_sha256 != authority_document["matched_labels_file_sha256"]:
        raise ContractValidationError(
            "FIXTURE_LABEL_FILE_HASH_MISMATCH",
            "matched-label fixture bytes do not match synthetic authority",
            context={
                "expected_sha256": authority_document["matched_labels_file_sha256"],
                "observed_sha256": observed_labels_file_sha256,
            },
        )
    labels_document = _decode_object_strict(
        labels_bytes, source=str(labels_path), kind="matched-label fixture"
    )
    validated = validate_matched_labels(labels_document, authority=authority)
    return _freeze_labels(validated, fixture_id=authority_pin.fixture_id)


def _validate_synthetic_authority(
    document: dict[str, Any], pin: SyntheticAuthorityPin
) -> MatchedLabelAuthority:
    if frozenset(document) != _AUTHORITY_MEMBERS:
        raise ContractValidationError(
            "FIXTURE_AUTHORITY_SCHEMA_INVALID",
            "synthetic authority members are not exact",
            context={"observed_members": sorted(document)},
        )
    if document["schema_version"] != _SYNTHETIC_AUTHORITY_VERSION:
        raise ContractValidationError(
            "FIXTURE_AUTHORITY_SCHEMA_INVALID",
            "synthetic authority schema version is unsupported",
        )
    if document["authority_role"] != _SYNTHETIC_AUTHORITY_ROLE:
        raise ContractValidationError(
            "FIXTURE_AUTHORITY_SCHEMA_INVALID",
            "synthetic authority role is not fixture-only",
        )
    if document["fixture_id"] != pin.fixture_id:
        raise ContractValidationError(
            "FIXTURE_AUTHORITY_PIN_MISMATCH",
            "synthetic authority fixture id does not match the independent pin",
        )

    hash_members = (
        "authority_sha256",
        "matched_labels_file_sha256",
        "matched_labels_sha256",
        "label_set_sha256",
        "crosswalk_sha256",
    )
    if any(not _is_sha256(document.get(member)) for member in hash_members):
        raise ContractValidationError(
            "FIXTURE_AUTHORITY_SCHEMA_INVALID",
            "synthetic authority contains an invalid SHA-256 identity",
        )
    for member in ("fixture_id", "label_set_id", "crosswalk_id"):
        if type(document.get(member)) is not str or not document[member]:
            raise ContractValidationError(
                "FIXTURE_AUTHORITY_SCHEMA_INVALID",
                "synthetic authority contains an invalid string identity",
                context={"member": member},
            )

    expected_seal = canonical_sha256(document, "authority_sha256")
    if document["authority_sha256"] != expected_seal:
        raise ContractValidationError(
            "FIXTURE_AUTHORITY_SEAL_MISMATCH",
            "synthetic fixture authority canonical seal does not match",
            context={
                "expected_sha256": expected_seal,
                "observed_sha256": document["authority_sha256"],
            },
        )

    return MatchedLabelAuthority(
        matched_labels_sha256=document["matched_labels_sha256"],
        label_set_id=document["label_set_id"],
        label_set_sha256=document["label_set_sha256"],
        crosswalk_id=document["crosswalk_id"],
        crosswalk_sha256=document["crosswalk_sha256"],
    )


def _freeze_labels(document: dict[str, Any], *, fixture_id: str) -> MatchedLabels:
    launches = tuple(_freeze_launch(launch) for launch in document["launches"])
    source = document["source_label_identity"]
    crosswalk = document["crosswalk_identity"]
    architecture = document["architecture"]
    return MatchedLabels(
        matched_labels_sha256=document["matched_labels_sha256"],
        label_set_id=source["label_set_id"],
        label_set_sha256=source["label_set_sha256"],
        label_schema_sha256=source["label_schema_sha256"],
        source_manifest_id=source["source_manifest_id"],
        source_manifest_sha256=source["source_manifest_sha256"],
        crosswalk_id=crosswalk["crosswalk_id"],
        crosswalk_sha256=crosswalk["crosswalk_sha256"],
        crosswalk_version=crosswalk["crosswalk_version"],
        binary_architecture=architecture["binary_architecture"],
        gpu_sku=architecture["gpu_sku"],
        launches=launches,
        synthetic_fixture_id=fixture_id,
    )


def _freeze_launch(launch: dict[str, Any]) -> MatchedLaunch:
    key = launch["launch_key"]
    identity = LaunchIdentity(
        replay=ReplayLaunchIdentity(
            kernel_mangled_name=key["kernel_mangled_name"],
            grid=_freeze_dim3(key["grid"]),
            block=_freeze_dim3(key["block"]),
            device_id=key["device_id"],
            stream_id=key["stream_id"],
            binary_architecture=key["binary_architecture"],
        ),
        cohort=CohortLaunchIdentity(
            workload_id=key["workload_id"],
            matched_execution_id=key["matched_execution_id"],
            process_launch_ordinal=key["process_launch_ordinal"],
        ),
        canonical_token=canonical_launch_key_token(key),
    )
    metrics = tuple(_freeze_metric(metric) for metric in launch["metrics"])
    if tuple(metric.key for metric in metrics) != METRIC_KEYS:
        metrics_by_key = {metric.key: metric for metric in metrics}
        metrics = tuple(metrics_by_key[key] for key in METRIC_KEYS)
    return MatchedLaunch(
        identity=identity,
        kernel_family=launch["kernel_family"],
        metrics=metrics,
    )


def _freeze_metric(metric: dict[str, Any]) -> LabelMetric:
    outcome = metric["outcome"]
    repeat = metric["repeat_metadata"]
    dispersion = repeat["dispersion"]
    return LabelMetric(
        key=metric["metric_key"],
        ncu_metric=metric["ncu_metric"],
        unit=metric["unit"],
        outcome_status=outcome["status"],
        value=outcome.get("value"),
        unavailable_reason=outcome.get("reason"),
        repeat_count=repeat["repeat_count"],
        aggregation=repeat["aggregation"],
        dispersion=Dispersion(
            status=dispersion["status"],
            rule=dispersion["rule"],
            scalar=dispersion.get("scalar"),
            reason=dispersion.get("reason"),
        ),
    )


def _freeze_dim3(value: dict[str, Any]) -> Dim3:
    return Dim3(x=value["x"], y=value["y"], z=value["z"])


def _read_bounded(path: str | Path, *, limit: int, kind: str) -> bytes:
    source = Path(path)
    try:
        with source.open("rb") as handle:
            material = handle.read(limit + 1)
    except OSError as exc:
        raise ContractValidationError(
            "FIXTURE_INPUT_READ_FAILED",
            f"could not read {kind}",
            context={"path": str(source), "error": str(exc)},
        ) from exc
    if len(material) > limit:
        raise ContractValidationError(
            "FIXTURE_INPUT_TOO_LARGE",
            f"{kind} exceeds the bounded fixture input size",
            context={"path": str(source), "max_bytes": limit},
        )
    return material


def _decode_object_strict(data: bytes, *, source: str, kind: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractValidationError(
            "FIXTURE_INPUT_UTF8_INVALID",
            f"{kind} is not valid UTF-8",
            context={"source": source},
        ) from exc
    value = loads_json_strict(text, source=source)
    if type(value) is not dict:
        raise ContractValidationError(
            "FIXTURE_INPUT_ROOT_INVALID",
            f"{kind} root must be an object",
            context={"source": source},
        )
    return value


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


__all__ = [
    "CohortLaunchIdentity",
    "Dim3",
    "Dispersion",
    "EvaluatorAuthorityUnavailable",
    "LabelMetric",
    "LaunchIdentity",
    "KNOWN_SYNTHETIC_AUTHORITY_FILE_SHA256",
    "MatchedLabels",
    "MatchedLaunch",
    "ReplayLaunchIdentity",
    "SyntheticAuthorityPin",
    "known_synthetic_authority_pin",
    "load_production_matched_labels",
    "load_synthetic_matched_labels",
]
