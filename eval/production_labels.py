"""Production-only construction and detached admission of matched NCU labels.

The simulator never imports this module. It accepts retained, independently
validated NCU launch documents only after a production launch-crosswalk v2 has
been fully verified. Label payloads are opened only after their detached
production authority bytes match an independent caller pin.
"""

from __future__ import annotations

import copy
import re
import statistics
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, ValidationError

from eval.contracts.launch_crosswalk import CompleteLaunchKey
from eval.contracts.model_seal import RetainedPayloadSnapshot
from eval.contracts.validation import (
    METRIC_KEYS,
    MatchedLabelAuthority,
    canonical_sha256,
    loads_json_strict,
    validate_matched_labels,
)
from eval.labels import MatchedLabels, _freeze_labels
from eval.registry import MetricDescriptor, MetricRegistry, load_metric_registry
from tools.local_labels import (
    metric_registry as source_metric_registry,
    ncu_action_executor,
    ncu_details_export_receipt,
    ncu_launch_documents,
    ncu_launch_sidecars,
)
from tools.local_labels import sidecar_schema_authority
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    canonical_json_bytes,
    safe_relative_path,
    sha256_bytes,
    strict_json_loads,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_SCHEMA_PATH = (
    ROOT / "eval/schemas/production_matched_label_authority.v1.schema.json"
)
REPEAT_MANIFEST_SCHEMA_PATH = (
    ROOT / "eval/schemas/production_ncu_repeat_manifest.v1.schema.json"
)
MATCHED_LABEL_SCHEMA_PATH = ROOT / "eval/schemas/matched_labels.v1.schema.json"
SOURCE_LABEL_SCHEMA_PATH = (
    ROOT / "workloads/simbench_bw/schemas/local_label_sidecar.schema.json"
)
NCU_INVENTORY_SCHEMA_PATH = ncu_launch_documents.INVENTORY_SCHEMA_PATH
NCU_METRICS_SCHEMA_PATH = ncu_launch_documents.METRICS_SCHEMA_PATH
NCU_INVENTORY_SCHEMA_ID = "simbench_bw/schemas/ncu_launch_inventory.schema.json"
NCU_METRICS_SCHEMA_ID = "simbench_bw/schemas/ncu_launch_metrics.schema.json"
NCU_INVENTORY_SCHEMA_SHA256 = (
    "9ea8f9da467f3f8db5521cf6d31926b95576f33e17e750a559d2e7d06074d04d"
)
NCU_METRICS_SCHEMA_SHA256 = (
    "8357781adca524bb0eedc899c24cba42f91aeec83fd46b27ce10042b4c7ced08"
)
AUTHORITY_SCHEMA_ID = (
    "https://gpu-simulator.local/eval/schemas/"
    "production_matched_label_authority.v1.schema.json"
)
REPEAT_MANIFEST_SCHEMA_ID = (
    "https://gpu-simulator.local/eval/schemas/"
    "production_ncu_repeat_manifest.v1.schema.json"
)
REPEAT_MANIFEST_SCHEMA_SHA256 = (
    "b1bb74e3e26dbaf1e5d10842b72930e769b33d88556348af2c140392cf97e190"
)
AUTHORITY_SCHEMA_SHA256 = (
    "fb22b9a53c0e6c40db5063167d2109cc6c2c022beda7c775297c05ec16693a52"
)
MATCHED_LABEL_SCHEMA_ID = (
    "https://gpu-simulator.local/eval/schemas/matched_labels.v1.schema.json"
)
MATCHED_LABEL_SCHEMA_SHA256 = (
    "c374994c8a84af4ecc18ff57086ad7bb61f41aa528425ef538f74e43dcea75bf"
)
AUTHORITY_SCHEMA_VERSION = "production_matched_label_authority.v1"
REPEAT_MANIFEST_SCHEMA_VERSION = "production_ncu_repeat_manifest.v1"
AUTHORITY_ROLE = "detached_production_matched_label_authority"
BUILDER_ID = "eval.production_labels.v1"
BUILDER_VERSION = 1
_MAX_AUTHORITY_BYTES = 1024 * 1024
_MAX_REPEAT_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_REPEAT_ARTIFACT_BYTES = 1024 * 1024 * 1024
_MAX_MATCHED_LABEL_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DIM3_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\Z")
_PRIVATE_CROSSWALK_TOKEN = object()
_PRIVATE_REPEAT_AUTHORITY_TOKEN = object()
_PRIVATE_REPEAT_TOKEN = object()
_PRIVATE_BUILD_TOKEN = object()
_PRIVATE_ARTIFACT_TOKEN = object()
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_PRODUCTION_CROSSWALK_REGISTRY_NAMES = (
    "CURRENT_M1_WORKLOAD_EXECUTION_AUTHORITIES",
    "PRODUCTION_WORKLOAD_EXECUTION_ADMISSION_AUTHORITIES",
    "PRODUCTION_PRELABEL_TRUST_ANCHORS",
    "PRODUCTION_NAMESPACE_ADAPTER_AUTHORITIES",
)

# No production repeat-manifest authority is admitted yet. Future entries must
# be installed by an independently reviewed producer/consumer authority slice;
# callers cannot supply manifest hashes or artifact identities directly.
PRODUCTION_REPEAT_MANIFEST_AUTHORITIES: Mapping[
    str, Mapping[str, Any]
] = MappingProxyType({})


class ProductionLabelError(ImmutableArtifactError):
    """A production matched-label authority or construction contract failed."""


def _fail(message: str) -> None:
    raise ProductionLabelError(message)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be lowercase 64-hex")
    return value


def _require_nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        _fail(f"{label} must be a nonempty string")
    return value


def _require_positive(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ProductionLabelSetAuthority:
    """Independent identities for one retained production label set."""

    label_set_id: str
    label_set_sha256: str
    source_manifest_id: str
    source_manifest_sha256: str
    expected_repeat_count: int
    binary_architecture: str
    gpu_sku: str
    producer_content_manifest_id: str
    producer_content_manifest_sha256: str
    label_schema_version: int = 5
    label_schema_sha256: str = sidecar_schema_authority.SCHEMA_SHA256

    def __post_init__(self) -> None:
        _require_nonempty(self.label_set_id, "label_set_id")
        _require_sha256(self.label_set_sha256, "label_set_sha256")
        _require_nonempty(self.source_manifest_id, "source_manifest_id")
        _require_sha256(self.source_manifest_sha256, "source_manifest_sha256")
        _require_positive(self.expected_repeat_count, "expected_repeat_count")
        _require_nonempty(self.binary_architecture, "binary_architecture")
        _require_nonempty(self.gpu_sku, "gpu_sku")
        _require_nonempty(
            self.producer_content_manifest_id, "producer_content_manifest_id"
        )
        _require_sha256(
            self.producer_content_manifest_sha256,
            "producer_content_manifest_sha256",
        )
        if self.label_schema_version != 5:
            _fail("production source label schema version must be 5")
        _require_sha256(self.label_schema_sha256, "label_schema_sha256")


@dataclass(frozen=True, slots=True)
class ProductionAuthorityPin:
    """Authority identity supplied independently of both detached files."""

    authority_file_sha256: str
    producer_content_manifest_sha256: str | None = None
    label_set_sha256: str | None = None
    source_manifest_sha256: str | None = None
    crosswalk_file_sha256: str | None = None
    builder_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.authority_file_sha256, "authority file SHA-256")
        for label, value in (
            (
                "producer content manifest SHA-256",
                self.producer_content_manifest_sha256,
            ),
            ("label-set SHA-256", self.label_set_sha256),
            ("source manifest SHA-256", self.source_manifest_sha256),
            ("crosswalk file SHA-256", self.crosswalk_file_sha256),
            ("builder SHA-256", self.builder_sha256),
        ):
            if value is not None:
                _require_sha256(value, label)


@dataclass(frozen=True, slots=True)
class _CrosswalkLaunch:
    launch_key: CompleteLaunchKey
    kernel_family: str
    ncu_native_launch_id: str
    ncu_record_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedProductionCrosswalk:
    """Opaque proof that launch_crosswalk v2 passed its full verifier."""

    payload_sha256: str
    model_seal_sha256: str
    crosswalk_id: str
    executable_sha256: str
    binary_architecture: str
    gpu_sku: str
    launches: tuple[_CrosswalkLaunch, ...]
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PRIVATE_CROSSWALK_TOKEN:
            _fail("verified production crosswalk must come from its authenticator")
        _require_sha256(self.payload_sha256, "crosswalk payload SHA-256")
        _require_sha256(self.model_seal_sha256, "crosswalk model seal")
        _require_nonempty(self.crosswalk_id, "crosswalk_id")
        _require_sha256(self.executable_sha256, "crosswalk executable SHA-256")
        if not self.launches:
            _fail("verified production crosswalk has no launches")


@dataclass(frozen=True, slots=True)
class _ValidatedLaunchMetrics:
    launch_key: CompleteLaunchKey
    metrics: tuple[tuple[str, str, int | float], ...]

    def by_name(self) -> dict[str, tuple[str, int | float]]:
        return {name: (unit, value) for name, unit, value in self.metrics}


@dataclass(frozen=True, slots=True)
class _RepeatArtifactIdentity:
    role: str
    absolute_path: str
    sha256: str
    size_bytes: int
    snapshot_identity_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty(self.role, "repeat artifact role")
        if (
            type(self.absolute_path) is not str
            or not Path(self.absolute_path).is_absolute()
            or "\x00" in self.absolute_path
        ):
            _fail("repeat artifact path is not an absolute path identity")
        _require_sha256(self.sha256, "repeat artifact SHA-256")
        _require_sha256(
            self.snapshot_identity_sha256,
            "repeat artifact snapshot identity SHA-256",
        )
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            _fail("repeat artifact size must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class _VerifiedRepeatManifestAuthority:
    authority_id: str
    manifest_file_sha256: str
    repeat_id: str
    cohort_repeat_ordinal: int
    source_manifest_sha256: str
    producer_content_manifest_sha256: str
    crosswalk_file_sha256: str
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PRIVATE_REPEAT_AUTHORITY_TOKEN:
            _fail(
                "production repeat-manifest authority must come from its "
                "authenticator"
            )
        _require_nonempty(self.authority_id, "repeat-manifest authority ID")
        _require_sha256(self.manifest_file_sha256, "repeat manifest file SHA-256")
        _require_nonempty(self.repeat_id, "repeat_id")
        if type(self.cohort_repeat_ordinal) is not int or (
            self.cohort_repeat_ordinal < 0
        ):
            _fail("cohort repeat ordinal must be a nonnegative integer")
        _require_sha256(self.source_manifest_sha256, "source manifest SHA-256")
        _require_sha256(
            self.producer_content_manifest_sha256,
            "producer content manifest SHA-256",
        )
        _require_sha256(self.crosswalk_file_sha256, "crosswalk file SHA-256")


@dataclass(frozen=True, slots=True)
class ValidatedNcuRepeat:
    """One complete retained NCU launch set, validated without target use."""

    repeat_index: int
    repeat_manifest_id: str
    repeat_manifest_sha256: str
    producer_content_manifest_sha256: str
    artifact_sha256s: tuple[str, ...]
    artifact_bundle_sha256: str
    artifact_identities: tuple[_RepeatArtifactIdentity, ...]
    launches: tuple[_ValidatedLaunchMetrics, ...]
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PRIVATE_REPEAT_TOKEN:
            _fail("NCU repeat evidence must come from retained validation")
        if type(self.repeat_index) is not int or self.repeat_index < 0:
            _fail("repeat_index must be a nonnegative integer")
        _require_nonempty(self.repeat_manifest_id, "repeat_manifest_id")
        _require_sha256(self.repeat_manifest_sha256, "repeat manifest SHA-256")
        _require_sha256(
            self.producer_content_manifest_sha256,
            "repeat producer content manifest SHA-256",
        )
        _require_sha256(self.artifact_bundle_sha256, "repeat artifact bundle SHA-256")
        if (
            not self.artifact_sha256s
            or not self.artifact_identities
            or not self.launches
            or len(self.artifact_sha256s) != len(self.artifact_identities)
            or any(
                type(item) is not _RepeatArtifactIdentity
                for item in self.artifact_identities
            )
        ):
            _fail("retained NCU repeat evidence is empty or inconsistent")
        if self.artifact_sha256s != tuple(
            item.sha256 for item in self.artifact_identities
        ):
            _fail("repeat artifact SHA-256 vector differs from retained identities")
        bundle_model = [
            {
                "role": item.role,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in self.artifact_identities
        ]
        expected_bundle_sha256 = sha256_bytes(canonical_json_bytes(bundle_model))
        if self.artifact_bundle_sha256 != expected_bundle_sha256:
            _fail("repeat artifact bundle SHA-256 mismatch")


@dataclass(frozen=True, slots=True)
class ProductionMatchedLabelBuild:
    document: dict[str, Any]
    payload: bytes
    crosswalk: VerifiedProductionCrosswalk
    source: ProductionLabelSetAuthority
    registry: MetricRegistry
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PRIVATE_BUILD_TOKEN:
            _fail("production label build must come from the builder")
        if canonical_json_bytes(self.document) != self.payload:
            _fail("production matched-label payload is not canonical JSON")


@dataclass(frozen=True, slots=True)
class ProductionMatchedLabelArtifacts:
    matched_labels_document: dict[str, Any]
    matched_labels_bytes: bytes
    authority_document: dict[str, Any]
    authority_bytes: bytes
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PRIVATE_ARTIFACT_TOKEN:
            _fail("production artifacts must come from the authority builder")
        if (
            canonical_json_bytes(self.matched_labels_document)
            != self.matched_labels_bytes
        ):
            _fail("matched-label artifact bytes are not canonical")
        if canonical_json_bytes(self.authority_document) != self.authority_bytes:
            _fail("detached authority artifact bytes are not canonical")


def _require_production_crosswalk_registries(
    crosswalk_contract: Any,
) -> None:
    namespace = vars(crosswalk_contract)
    for attribute in _PRODUCTION_CROSSWALK_REGISTRY_NAMES:
        if attribute not in namespace:
            _fail(
                "production crosswalk verifier authority is unavailable: "
                f"required immutable registry {attribute} is missing"
            )
        registry = namespace[attribute]
        if type(registry) is not _MAPPING_PROXY_TYPE:
            _fail(
                "production crosswalk verifier authority is unavailable: "
                f"registry {attribute} is not an immutable mapping proxy"
            )
        if not registry:
            _fail(
                "production crosswalk verifier authority is unavailable: "
                f"registry {attribute} is empty"
            )


def authenticate_production_crosswalk(
    crosswalk_path: str | Path,
    **verification_inputs: Any,
) -> VerifiedProductionCrosswalk:
    """Open a crosswalk only through the installed production verifier."""

    from eval.contracts import launch_crosswalk as crosswalk_contract

    _require_production_crosswalk_registries(crosswalk_contract)
    crosswalk_verifier = getattr(
        crosswalk_contract, "load_production_launch_crosswalk", None
    )
    if crosswalk_verifier is None:
        crosswalk_verifier = getattr(
            crosswalk_contract, "load_launch_crosswalk_v2", None
        )
    if not callable(crosswalk_verifier):
        _fail("production crosswalk verifier authority is unavailable")
    payload, snapshot = RetainedPayloadSnapshot.read_path(Path(crosswalk_path))
    snapshot.require_production_acquisition("production launch crosswalk")
    snapshot.verify_payload(payload)
    try:
        model = crosswalk_verifier(payload, **verification_inputs)
    except (TypeError, ValueError, ImmutableArtifactError) as exc:
        raise ProductionLabelError(
            f"production crosswalk verification failed: {exc}"
        ) from exc
    if type(model) is not dict or canonical_json_bytes(model) != payload:
        _fail("verified production crosswalk is not a canonical object")
    if model.get("schema_version") != 2:
        _fail("verified production crosswalk version is not v2")
    if model.get("contract_mode") != "TARGET_FREE_PRODUCTION":
        _fail("fixture or target-bearing crosswalk cannot authorize production labels")
    if model.get("outcome") != {"status": "ADMITTED", "reason_codes": []}:
        _fail("production crosswalk outcome is not admitted")
    architecture = model["architecture"]
    ncu_records = model["inventories"]["ncu"]["records"]
    if len(ncu_records) != len(model["bijection"]):
        _fail("production crosswalk NCU inventory is incomplete")
    launches: list[_CrosswalkLaunch] = []
    for row, ncu_record in zip(model["bijection"], ncu_records, strict=True):
        key = CompleteLaunchKey.from_model(row["launch_key"])
        if row["kernel_family"] != model["kernel_family"]:
            _fail("production crosswalk kernel family is not workload-authoritative")
        launches.append(
            _CrosswalkLaunch(
                launch_key=key,
                kernel_family=row["kernel_family"],
                ncu_native_launch_id=ncu_record["native_identity"]["native_launch_id"],
                ncu_record_sha256=ncu_record["record_sha256"],
            )
        )
    return VerifiedProductionCrosswalk(
        payload_sha256=snapshot.payload_sha256,
        model_seal_sha256=model["model_seal_sha256"],
        crosswalk_id=f"production-crosswalk-v2-{model['model_seal_sha256']}",
        executable_sha256=model["executable_sha256"],
        binary_architecture=architecture["binary_architecture"],
        gpu_sku=architecture["gpu_sku"],
        launches=tuple(launches),
        _construction_token=_PRIVATE_CROSSWALK_TOKEN,
    )


def _read_retained(
    path: str | Path, label: str
) -> tuple[bytes, RetainedPayloadSnapshot]:
    try:
        payload, snapshot = RetainedPayloadSnapshot.read_path(Path(path))
    except (OSError, ImmutableArtifactError) as exc:
        raise ProductionLabelError(f"could not retain {label}: {exc}") from exc
    snapshot.require_production_acquisition(label)
    snapshot.verify_payload(payload)
    return payload, snapshot


def _decode_json(payload: bytes, label: str) -> Any:
    try:
        return strict_json_loads(payload, label=label)
    except (UnicodeError, ImmutableArtifactError) as exc:
        raise ProductionLabelError(f"invalid {label}: {exc}") from exc


def _decode_schema_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = loads_json_strict(text, source=label)
    except (UnicodeError, ValueError) as exc:
        raise ProductionLabelError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict:
        _fail(f"{label} root is not an object")
    return value


def _parse_dim3(value: object, label: str) -> tuple[int, int, int]:
    if type(value) is not str:
        _fail(f"{label} is not a string")
    match = _DIM3_RE.fullmatch(value)
    if match is None:
        _fail(f"{label} is not a canonical NCU dimension triple")
    result = tuple(int(item) for item in match.groups())
    if any(item < 1 for item in result):
        _fail(f"{label} contains a nonpositive component")
    return result  # type: ignore[return-value]


def _parse_nonnegative_decimal(value: object, label: str) -> int:
    if type(value) is not str or not value.isascii() or not value.isdecimal():
        _fail(f"{label} is not a nonnegative decimal identity")
    return int(value)


def _require_inventory_identity(
    identity: Mapping[str, Any], expected: _CrosswalkLaunch, ordinal: int
) -> None:
    key = expected.launch_key
    observed = (
        identity.get("ID"),
        identity.get("Kernel Name"),
        _parse_dim3(identity.get("Grid Size"), f"launch {ordinal} grid"),
        _parse_dim3(identity.get("Block Size"), f"launch {ordinal} block"),
        _parse_nonnegative_decimal(identity.get("Device"), f"launch {ordinal} device"),
        _parse_nonnegative_decimal(identity.get("Stream"), f"launch {ordinal} stream"),
    )
    wanted = (
        expected.ncu_native_launch_id,
        key.kernel_mangled_name,
        key.grid,
        key.block,
        key.device_id,
        key.stream_id,
    )
    if observed != wanted:
        _fail(f"NCU launch {ordinal} identity differs from complete launch key")


def _load_ncu_document_schema(kind: str) -> dict[str, Any]:
    if kind == "inventory":
        configured_path = Path(NCU_INVENTORY_SCHEMA_PATH)
        canonical_path = (
            ROOT
            / "workloads/simbench_bw/schemas/ncu_launch_inventory.schema.json"
        )
        expected_sha256 = NCU_INVENTORY_SCHEMA_SHA256
    elif kind == "metrics":
        configured_path = Path(NCU_METRICS_SCHEMA_PATH)
        canonical_path = (
            ROOT / "workloads/simbench_bw/schemas/ncu_launch_metrics.schema.json"
        )
        expected_sha256 = NCU_METRICS_SCHEMA_SHA256
    else:
        _fail(f"unsupported governing NCU schema: {kind}")
    payload, snapshot = _read_retained(configured_path, f"NCU {kind} schema")
    if snapshot.payload_sha256 != expected_sha256:
        _fail(f"NCU {kind} schema SHA-256 mismatch")
    if configured_path != canonical_path:
        _fail(f"NCU {kind} schema path substitution is forbidden")
    try:
        return ncu_launch_documents.load_schema(
            kind, schema_path=canonical_path, payload=payload
        )
    except (TypeError, ValueError) as exc:
        raise ProductionLabelError(f"NCU {kind} schema is invalid: {exc}") from exc


def _load_governing_ncu_schemas() -> dict[str, dict[str, Any]]:
    return {
        "inventory": _load_ncu_document_schema("inventory"),
        "metrics": _load_ncu_document_schema("metrics"),
    }


def _load_repeat_manifest_schema() -> dict[str, Any]:
    canonical_path = (
        ROOT / "eval/schemas/production_ncu_repeat_manifest.v1.schema.json"
    )
    configured_path = Path(REPEAT_MANIFEST_SCHEMA_PATH)
    payload, snapshot = _read_retained(
        configured_path, "production NCU repeat-manifest schema"
    )
    if snapshot.payload_sha256 != REPEAT_MANIFEST_SCHEMA_SHA256:
        _fail("production NCU repeat-manifest schema SHA-256 mismatch")
    if configured_path != canonical_path:
        _fail("production NCU repeat-manifest schema path substitution is forbidden")
    schema = _decode_schema_json(payload, "production NCU repeat-manifest schema")
    if (
        schema.get("$id") != REPEAT_MANIFEST_SCHEMA_ID
        or schema.get("$schema")
        != "https://json-schema.org/draft/2020-12/schema"
    ):
        _fail("production NCU repeat-manifest schema identity is invalid")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ProductionLabelError(
            f"production NCU repeat-manifest schema is invalid: {exc}"
        ) from exc
    return schema


def _production_repeat_manifest_authority(
    authority_id: str,
    *,
    source: ProductionLabelSetAuthority,
    crosswalk: VerifiedProductionCrosswalk,
) -> _VerifiedRepeatManifestAuthority:
    if not isinstance(source, ProductionLabelSetAuthority):
        _fail("production label-set authority is unavailable")
    if not isinstance(crosswalk, VerifiedProductionCrosswalk):
        _fail("production crosswalk proof is unavailable")
    _require_nonempty(authority_id, "repeat-manifest authority ID")
    entry = PRODUCTION_REPEAT_MANIFEST_AUTHORITIES.get(authority_id)
    if entry is None:
        _fail(
            "PRODUCTION_REPEAT_MANIFEST_AUTHORITY_UNAVAILABLE: no independently "
            "admitted Q809 repeat-manifest authority is installed"
        )
    expected_fields = {
        "manifest_file_sha256",
        "repeat_id",
        "cohort_repeat_ordinal",
        "source_manifest_sha256",
        "producer_content_manifest_sha256",
        "crosswalk_file_sha256",
    }
    if type(entry) is not dict or set(entry) != expected_fields:
        _fail("installed production repeat-manifest authority is malformed")
    authority = _VerifiedRepeatManifestAuthority(
        authority_id=authority_id,
        manifest_file_sha256=entry["manifest_file_sha256"],
        repeat_id=entry["repeat_id"],
        cohort_repeat_ordinal=entry["cohort_repeat_ordinal"],
        source_manifest_sha256=entry["source_manifest_sha256"],
        producer_content_manifest_sha256=entry[
            "producer_content_manifest_sha256"
        ],
        crosswalk_file_sha256=entry["crosswalk_file_sha256"],
        _construction_token=_PRIVATE_REPEAT_AUTHORITY_TOKEN,
    )
    if (
        authority.source_manifest_sha256 != source.source_manifest_sha256
        or authority.producer_content_manifest_sha256
        != source.producer_content_manifest_sha256
        or authority.crosswalk_file_sha256 != crosswalk.payload_sha256
    ):
        _fail("installed repeat-manifest authority differs from source or crosswalk")
    return authority


def _validate_json_schema(
    value: Any, schema: Mapping[str, Any], *, label: str
) -> None:
    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError as exc:
        location = "/" + "/".join(str(item) for item in exc.absolute_path)
        raise ProductionLabelError(
            f"{label} schema validation failed at {location}: {exc.message}"
        ) from exc


def _artifact_binding(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        _fail(f"{label} binding fields differ")
    assert isinstance(value, dict)
    try:
        safe_relative_path(value["relative_path"])
    except ImmutableArtifactError as exc:
        raise ProductionLabelError(f"{label} path is invalid: {exc}") from exc
    _require_sha256(value["sha256"], f"{label} SHA-256")
    if (
        type(value["size_bytes"]) is not int
        or value["size_bytes"] < 0
        or value["size_bytes"] > _MAX_REPEAT_ARTIFACT_BYTES
    ):
        _fail(f"{label} size is outside the retained artifact bound")
    return value


def _read_bound_repeat_artifact(
    *,
    manifest_root: Path,
    manifest_path: Path,
    role: str,
    binding: Mapping[str, Any],
    require_nonempty: bool,
) -> tuple[bytes, _RepeatArtifactIdentity]:
    relative = safe_relative_path(binding["relative_path"])
    path = manifest_root.joinpath(*relative.parts)
    if path.absolute() == manifest_path.absolute():
        _fail(f"repeat manifest cannot also serve as {role}")
    payload, snapshot = _read_retained(path, f"repeat artifact {role}")
    if (
        snapshot.payload_sha256 != binding["sha256"]
        or snapshot.size_bytes != binding["size_bytes"]
    ):
        _fail(f"repeat artifact {role} differs from its authenticated manifest")
    if require_nonempty and not payload:
        _fail(f"repeat artifact {role} is empty")
    return payload, _RepeatArtifactIdentity(
        role=role,
        absolute_path=str(path.absolute()),
        sha256=snapshot.payload_sha256,
        size_bytes=snapshot.size_bytes,
        snapshot_identity_sha256=snapshot.snapshot_identity_sha256,
    )


def _require_repeat_manifest_bindings(
    manifest: Mapping[str, Any],
    *,
    authority: _VerifiedRepeatManifestAuthority,
    source: ProductionLabelSetAuthority,
    crosswalk: VerifiedProductionCrosswalk,
) -> tuple[dict[str, Any], tuple[CompleteLaunchKey, ...]]:
    repeat_identity = manifest["repeat_identity"]
    if repeat_identity != {
        "repeat_id": authority.repeat_id,
        "cohort_repeat_ordinal": authority.cohort_repeat_ordinal,
    }:
        _fail("repeat identity differs from its installed authority")
    if manifest["source_label_identity"] != {
        "label_set_id": source.label_set_id,
        "label_set_sha256": source.label_set_sha256,
        "source_manifest_id": source.source_manifest_id,
        "source_manifest_sha256": source.source_manifest_sha256,
    }:
        _fail("repeat manifest source identity differs from label-set authority")
    if manifest["crosswalk_identity"] != {
        "crosswalk_id": crosswalk.crosswalk_id,
        "crosswalk_file_sha256": crosswalk.payload_sha256,
        "crosswalk_model_seal_sha256": crosswalk.model_seal_sha256,
    }:
        _fail("repeat manifest crosswalk identity differs")
    if manifest["architecture"] != {
        "binary_architecture": crosswalk.binary_architecture,
        "gpu_sku": crosswalk.gpu_sku,
    }:
        _fail("repeat manifest architecture/SKU differs from crosswalk")
    if manifest["producer_content_manifest"] != {
        "manifest_id": source.producer_content_manifest_id,
        "content_manifest_sha256": source.producer_content_manifest_sha256,
    }:
        _fail("repeat manifest producer content anchor differs")

    capture = manifest["capture_contract"]
    protocol = capture["protocol"]
    if protocol["ncu_launch_count"] != len(crosswalk.launches) + 1:
        _fail("repeat manifest Q809 launch count differs from crosswalk")
    provenance = capture["sidecar_provenance"]
    if (
        provenance["device"] != crosswalk.binary_architecture
        or provenance["binary_sha256"] != crosswalk.executable_sha256
    ):
        _fail("repeat manifest sidecar provenance differs from crosswalk")

    bindings = manifest["launch_bindings"]
    if len(bindings) != len(crosswalk.launches):
        _fail("repeat manifest launch binding inventory is partial")
    launch_keys: list[CompleteLaunchKey] = []
    for ordinal, (binding, expected) in enumerate(
        zip(bindings, crosswalk.launches, strict=True)
    ):
        if (
            binding["cohort_repeat_ordinal"]
            != authority.cohort_repeat_ordinal
            or binding["report_launch_ordinal"] != ordinal
        ):
            _fail("repeat manifest cohort or launch ordinal is substituted")
        try:
            key = CompleteLaunchKey.from_model(binding["launch_key"])
        except (TypeError, ValueError, ImmutableArtifactError) as exc:
            raise ProductionLabelError(
                f"repeat manifest launch {ordinal} key is invalid: {exc}"
            ) from exc
        if binding["launch_key_sha256"] != key.sha256:
            _fail(f"repeat manifest launch {ordinal} key SHA-256 mismatch")
        if (
            key != expected.launch_key
            or binding["ncu_record_sha256"] != expected.ncu_record_sha256
        ):
            _fail(
                f"repeat manifest launch {ordinal} full key or NCU record differs "
                "from the target-free crosswalk"
            )
        launch_keys.append(key)
    return capture, tuple(launch_keys)


def _validate_authenticated_details_receipt(
    receipt_bytes: bytes,
    *,
    artifact_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = _decode_json(receipt_bytes, "NCU Details export receipt")
    if type(receipt) is not dict:
        _fail("NCU Details export receipt root is invalid")
    role_map = (
        ("ncu_report", "ncu_report"),
        ("details_csv", "details_csv"),
        ("details_stderr", "details_stderr"),
        ("launch_inventory", "launch_inventory"),
        ("launch_metrics", "launch_metrics"),
    )
    try:
        expected_products = [
            ncu_details_export_receipt.product_binding(
                role=receipt_role,
                relative_path=artifact_bindings[manifest_role]["relative_path"],
                sha256=artifact_bindings[manifest_role]["sha256"],
                size_bytes=artifact_bindings[manifest_role]["size_bytes"],
            )
            for receipt_role, manifest_role in role_map
        ]
        ncu_identity_model = receipt["ncu_identity"]
        expected_ncu_identity = ncu_action_executor.ExpectedNcuIdentity(
            resolved_path=ncu_identity_model["resolved_path"],
            executable_sha256=ncu_identity_model["executable_sha256"],
            version_line=ncu_identity_model["version_line"],
        )
        ncu_details_export_receipt.validate_receipt(
            receipt,
            expected_preflight_sha256=receipt["preflight_sha256"],
            expected_source_commit=receipt["source_commit"],
            expected_capture_identity_sha256=receipt["capture_identity_sha256"],
            expected_ncu_identity=expected_ncu_identity,
            expected_import_arguments=(
                ncu_details_export_receipt.EXPECTED_IMPORT_ARGUMENTS
            ),
            expected_environment=dict(
                ncu_details_export_receipt.EXPECTED_ACTION_ENVIRONMENT
            ),
            expected_working_directory="/",
            expected_exit_status=0,
            expected_products=expected_products,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionLabelError(
            f"NCU Details export receipt is invalid: {exc}"
        ) from exc
    return receipt


def _open_authenticated_repeat_artifacts(
    manifest: Mapping[str, Any], *, manifest_path: Path
) -> tuple[
    dict[str, bytes],
    tuple[_RepeatArtifactIdentity, ...],
    dict[str, Any],
]:
    artifacts = manifest["artifacts"]
    fixed_roles = (
        "ncu_report",
        "details_export_receipt",
        "details_csv",
        "details_stderr",
        "launch_inventory",
        "launch_metrics",
    )
    bindings: dict[str, Any] = {
        role: _artifact_binding(artifacts[role], label=role)
        for role in fixed_roles
    }
    sidecar_bindings: list[dict[str, Any]] = []
    for ordinal, value in enumerate(artifacts["launch_sidecars"]):
        if type(value) is not dict or set(value) != {
            "report_launch_ordinal",
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            _fail(f"launch sidecar {ordinal} binding fields differ")
        if value["report_launch_ordinal"] != ordinal:
            _fail("repeat manifest launch sidecar inventory is reordered")
        binding = _artifact_binding(
            {
                "relative_path": value["relative_path"],
                "sha256": value["sha256"],
                "size_bytes": value["size_bytes"],
            },
            label=f"launch_sidecar_{ordinal}",
        )
        sidecar_bindings.append(binding)
    if len(sidecar_bindings) != len(manifest["launch_bindings"]):
        _fail("repeat manifest launch sidecar inventory is partial")
    paths = [
        binding["relative_path"]
        for binding in (*bindings.values(), *sidecar_bindings)
    ]
    if len(paths) != len(set(paths)):
        _fail("repeat manifest artifact paths are duplicated")

    payloads: dict[str, bytes] = {}
    identities: list[_RepeatArtifactIdentity] = []
    root = manifest_path.parent
    for role in fixed_roles:
        payload, identity = _read_bound_repeat_artifact(
            manifest_root=root,
            manifest_path=manifest_path,
            role=role,
            binding=bindings[role],
            require_nonempty=role != "details_stderr",
        )
        payloads[role] = payload
        identities.append(identity)
    for ordinal, binding in enumerate(sidecar_bindings):
        role = f"launch_sidecar_{ordinal}"
        payload, identity = _read_bound_repeat_artifact(
            manifest_root=root,
            manifest_path=manifest_path,
            role=role,
            binding=binding,
            require_nonempty=True,
        )
        payloads[role] = payload
        identities.append(identity)
    bindings["launch_sidecars"] = sidecar_bindings
    return payloads, tuple(identities), bindings


def _decode_validate_repeat_documents(
    payloads: Mapping[str, bytes],
    *,
    schemas: Mapping[str, Mapping[str, Any]],
    capture: Mapping[str, Any],
    crosswalk: VerifiedProductionCrosswalk,
    artifact_bindings: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    receipt = _validate_authenticated_details_receipt(
        payloads["details_export_receipt"],
        artifact_bindings=artifact_bindings,
    )
    try:
        csv_text = payloads["details_csv"].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProductionLabelError("NCU Details CSV is not UTF-8") from exc
    inventory = _decode_json(payloads["launch_inventory"], "NCU launch inventory")
    metrics = _decode_json(payloads["launch_metrics"], "NCU launch metrics")
    if type(inventory) is not dict or type(metrics) is not list:
        _fail("NCU launch document roots are invalid")
    _validate_json_schema(
        inventory, schemas["inventory"], label="NCU launch inventory"
    )
    _validate_json_schema(metrics, schemas["metrics"], label="NCU launch metrics")

    protocol = capture["protocol"]
    percentage_metric_names = tuple(capture["percentage_metric_names"])
    unit_metric_names = tuple(capture["unit_metric_names"])
    registry = source_metric_registry.load_registry()
    report_sha256 = artifact_bindings["ncu_report"]["sha256"]
    try:
        inventory_rows = ncu_launch_documents.validate_launch_inventory_document(
            inventory,
            expected_launch_count=len(crosswalk.launches),
            report_sha256=report_sha256,
            protocol=protocol,
        )
        metric_rows = ncu_launch_documents.validate_launch_metrics_document(
            metrics,
            csv_text=csv_text,
            inventory=inventory,
            registry=registry,
            percentage_metric_names=percentage_metric_names,
            unit_metric_names=unit_metric_names,
        )
    except (TypeError, ValueError) as exc:
        raise ProductionLabelError(f"NCU launch documents are invalid: {exc}") from exc
    if len(inventory_rows) != len(crosswalk.launches):
        _fail("NCU repeat launch inventory is partial")

    sidecars: list[dict[str, Any]] = []
    for ordinal in range(len(crosswalk.launches)):
        sidecar = _decode_json(
            payloads[f"launch_sidecar_{ordinal}"],
            f"NCU launch sidecar {ordinal}",
        )
        if type(sidecar) is not dict:
            _fail(f"NCU launch sidecar {ordinal} root is invalid")
        sidecars.append(sidecar)
    provenance_model = capture["sidecar_provenance"]
    if (
        receipt["ncu_identity"]["version_line"]
        != provenance_model["profiler_version"]
    ):
        _fail("NCU receipt identity differs from sidecar profiler provenance")
    try:
        provenance = ncu_launch_sidecars.LaunchSidecarProvenance(
            device=provenance_model["device"],
            slug=provenance_model["slug"],
            shape_point=provenance_model["shape_point"],
            binary_sha256=provenance_model["binary_sha256"],
            profiler_version=provenance_model["profiler_version"],
            cuda_version=provenance_model["cuda_version"],
            driver_version=provenance_model["driver_version"],
            gpu_uuid_sha256=provenance_model["gpu_uuid_sha256"],
            warmup_count=provenance_model["warmup_count"],
        )
        _schema_bytes, sidecar_schema = (
            sidecar_schema_authority.load_authoritative_schema()
        )
        ncu_launch_sidecars.validate_launch_sidecar_documents(
            sidecars,
            inventory=inventory,
            launch_metrics=metrics,
            csv_text=csv_text,
            report_sha256=report_sha256,
            expected_launch_count=len(crosswalk.launches),
            protocol=protocol,
            registry=registry,
            percentage_metric_names=percentage_metric_names,
            unit_metric_names=unit_metric_names,
            sidecar_schema=sidecar_schema,
            provenance=provenance,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionLabelError(f"NCU launch sidecars are invalid: {exc}") from exc
    return inventory_rows, metric_rows


def _validate_retained_ncu_repeat_from_manifest(
    *,
    repeat_manifest_path: str | Path,
    authority: _VerifiedRepeatManifestAuthority,
    source: ProductionLabelSetAuthority,
    crosswalk: VerifiedProductionCrosswalk,
) -> ValidatedNcuRepeat:
    if type(authority) is not _VerifiedRepeatManifestAuthority:
        _fail("authenticated production repeat-manifest authority is unavailable")
    if not isinstance(source, ProductionLabelSetAuthority):
        _fail("production label-set authority is unavailable")
    if not isinstance(crosswalk, VerifiedProductionCrosswalk):
        _fail("production crosswalk proof is unavailable")
    if (
        authority.source_manifest_sha256 != source.source_manifest_sha256
        or authority.producer_content_manifest_sha256
        != source.producer_content_manifest_sha256
        or authority.crosswalk_file_sha256 != crosswalk.payload_sha256
    ):
        _fail("repeat-manifest authority differs from source or crosswalk")

    # Governing schemas are retained and hash-validated before any repeat
    # manifest or artifact payload is opened.
    ncu_schemas = _load_governing_ncu_schemas()
    repeat_schema = _load_repeat_manifest_schema()
    manifest_path = Path(repeat_manifest_path)
    manifest_bytes, manifest_snapshot = _read_retained(
        manifest_path, "production NCU repeat manifest"
    )
    if len(manifest_bytes) > _MAX_REPEAT_MANIFEST_BYTES:
        _fail("production NCU repeat manifest exceeds its byte bound")
    if manifest_snapshot.payload_sha256 != authority.manifest_file_sha256:
        _fail("production NCU repeat manifest file SHA-256 mismatch")
    manifest = _decode_json(manifest_bytes, "production NCU repeat manifest")
    if type(manifest) is not dict:
        _fail("production NCU repeat manifest root is invalid")
    if canonical_json_bytes(manifest) != manifest_bytes:
        _fail("production NCU repeat manifest is not canonical JSON")
    _validate_json_schema(
        manifest, repeat_schema, label="production NCU repeat manifest"
    )
    capture, launch_keys = _require_repeat_manifest_bindings(
        manifest, authority=authority, source=source, crosswalk=crosswalk
    )
    payloads, identities, artifact_bindings = (
        _open_authenticated_repeat_artifacts(
            manifest, manifest_path=manifest_path
        )
    )
    inventory_rows, metric_rows = _decode_validate_repeat_documents(
        payloads,
        schemas=ncu_schemas,
        capture=capture,
        crosswalk=crosswalk,
        artifact_bindings=artifact_bindings,
    )

    launches: list[_ValidatedLaunchMetrics] = []
    for ordinal, (inventory_row, metric_row, expected, launch_key) in enumerate(
        zip(
            inventory_rows,
            metric_rows,
            crosswalk.launches,
            launch_keys,
            strict=True,
        )
    ):
        if inventory_row["report_launch_ordinal"] != ordinal:
            _fail("NCU launch inventory is reordered")
        _require_inventory_identity(inventory_row["identity"], expected, ordinal)
        values: list[tuple[str, str, int | float]] = []
        for metric_name, value in metric_row["fraction_values"].items():
            values.append((metric_name, "pct", value))
        for metric_name, record in metric_row["unit_tagged_metrics"].items():
            values.append((metric_name, record["unit"], record["value"]))
        values.sort(key=lambda item: item[0])
        if len(values) != len({item[0] for item in values}):
            _fail(f"NCU launch {ordinal} repeats a metric")
        launches.append(_ValidatedLaunchMetrics(launch_key, tuple(values)))

    bundle_model = [
        {
            "role": identity.role,
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }
        for identity in identities
    ]
    return ValidatedNcuRepeat(
        repeat_index=authority.cohort_repeat_ordinal,
        repeat_manifest_id=authority.repeat_id,
        repeat_manifest_sha256=manifest_snapshot.payload_sha256,
        producer_content_manifest_sha256=(
            source.producer_content_manifest_sha256
        ),
        artifact_sha256s=tuple(item.sha256 for item in identities),
        artifact_bundle_sha256=sha256_bytes(canonical_json_bytes(bundle_model)),
        artifact_identities=identities,
        launches=tuple(launches),
        _construction_token=_PRIVATE_REPEAT_TOKEN,
    )


def validate_retained_ncu_repeat(
    *,
    repeat_manifest_path: str | Path,
    repeat_authority_id: str,
    source: ProductionLabelSetAuthority,
    crosswalk: VerifiedProductionCrosswalk,
) -> ValidatedNcuRepeat:
    """Authenticate one installed repeat authority, then open its exact manifest."""

    authority = _production_repeat_manifest_authority(
        repeat_authority_id, source=source, crosswalk=crosswalk
    )
    return _validate_retained_ncu_repeat_from_manifest(
        repeat_manifest_path=repeat_manifest_path,
        authority=authority,
        source=source,
        crosswalk=crosswalk,
    )


def _require_source_schema(source: ProductionLabelSetAuthority) -> None:
    payload, snapshot = _read_retained(SOURCE_LABEL_SCHEMA_PATH, "v5 label schema")
    if snapshot.payload_sha256 != source.label_schema_sha256:
        _fail("source label schema SHA-256 differs from independent label authority")
    try:
        sidecar_schema_authority.parse_authoritative_schema_bytes(payload)
    except ValueError as exc:
        raise ProductionLabelError(f"source label schema is invalid: {exc}") from exc


def _require_complete_repeats(
    repeats: Sequence[ValidatedNcuRepeat],
    *,
    source: ProductionLabelSetAuthority,
    crosswalk: VerifiedProductionCrosswalk,
) -> tuple[ValidatedNcuRepeat, ...]:
    if type(repeats) not in {list, tuple}:
        _fail("production NCU repeats must be an ordered list or tuple")
    material = tuple(repeats)
    if any(not isinstance(item, ValidatedNcuRepeat) for item in material):
        _fail("production NCU repeats contain unvalidated evidence")
    observed_indices = tuple(item.repeat_index for item in material)
    expected_indices = tuple(range(source.expected_repeat_count))
    if observed_indices != expected_indices:
        _fail("production NCU repeats are missing, duplicated, or reordered")
    manifest_ids = tuple(item.repeat_manifest_id for item in material)
    manifest_hashes = tuple(item.repeat_manifest_sha256 for item in material)
    if len(set(manifest_ids)) != len(manifest_ids) or len(set(manifest_hashes)) != len(
        manifest_hashes
    ):
        _fail("production NCU repeat manifests are duplicated")
    bundle_hashes = tuple(item.artifact_bundle_sha256 for item in material)
    if len(set(bundle_hashes)) != len(bundle_hashes):
        _fail("production NCU repeat artifact bundles are duplicated")
    seen_paths: set[str] = set()
    seen_nonempty_hashes: set[str] = set()
    for item in material:
        paths = {identity.absolute_path for identity in item.artifact_identities}
        nonempty_hashes = {
            identity.sha256
            for identity in item.artifact_identities
            if identity.size_bytes > 0
        }
        if seen_paths.intersection(paths) or seen_nonempty_hashes.intersection(
            nonempty_hashes
        ):
            _fail("production NCU artifacts are reused across repeats")
        seen_paths.update(paths)
        seen_nonempty_hashes.update(nonempty_hashes)
    expected_keys = tuple(item.launch_key for item in crosswalk.launches)
    for item in material:
        if (
            item.producer_content_manifest_sha256
            != source.producer_content_manifest_sha256
        ):
            _fail("NCU repeat does not belong to the external producer manifest")
        observed_keys = tuple(launch.launch_key for launch in item.launches)
        if observed_keys != expected_keys:
            _fail("NCU repeat launch inventory is missing, duplicated, or reordered")
    return material


def _dispersion(rule: str, values: Sequence[int | float]) -> float | None:
    if len(values) < 2:
        return None
    floats = [float(item) for item in values]
    mean = statistics.fmean(floats)
    if rule == "stddev_abs":
        return statistics.stdev(floats)
    if rule == "coefficient_of_variation":
        deviation = statistics.stdev(floats)
        if mean == 0:
            return 0.0 if deviation == 0 else None
        return deviation / abs(mean)
    if rule == "relative_tolerance":
        low, high = min(floats), max(floats)
        if high == low:
            return 0.0
        if mean == 0:
            return None
        return (high - low) / abs(mean)
    _fail(f"unsupported registry dispersion rule: {rule}")


def _unavailable_metric(
    descriptor: MetricDescriptor,
    *,
    repeat_count: int,
    reason: str,
    dispersion_scalar: float | None,
) -> dict[str, Any]:
    if dispersion_scalar is None:
        dispersion: dict[str, Any] = {
            "status": "unavailable",
            "rule": descriptor.repeatability_rule,
            "reason": "source_dispersion_null",
        }
    else:
        dispersion = {
            "status": "available",
            "rule": descriptor.repeatability_rule,
            "scalar": dispersion_scalar,
        }
    return {
        "metric_key": descriptor.key,
        "ncu_metric": descriptor.ncu_metric,
        "unit": descriptor.unit,
        "outcome": {"status": "unavailable", "value": None, "reason": reason},
        "repeat_metadata": {
            "repeat_count": repeat_count,
            "aggregation": (
                "mean_fraction" if descriptor.unit == "pct" else "mean_nonneg"
            ),
            "dispersion": dispersion,
        },
    }


def _aggregate_metric(
    descriptor: MetricDescriptor,
    *,
    per_repeat: Sequence[dict[str, tuple[str, int | float]]],
    registry: MetricRegistry,
) -> dict[str, Any]:
    observed = [row.get(descriptor.ncu_metric) for row in per_repeat]
    present = [item for item in observed if item is not None]
    repeat_count = len(per_repeat)
    if not present:
        return _unavailable_metric(
            descriptor,
            repeat_count=repeat_count,
            reason="metric_not_collected",
            dispersion_scalar=None,
        )
    if any(unit != descriptor.unit for unit, _value in present):
        _fail(f"unit mismatch for registry metric {descriptor.key}")
    values = [value for _unit, value in present]
    scalar = _dispersion(descriptor.repeatability_rule, values)
    if len(present) != repeat_count:
        return _unavailable_metric(
            descriptor,
            repeat_count=repeat_count,
            reason="typed_exclusion",
            dispersion_scalar=scalar,
        )
    if scalar is None:
        return _unavailable_metric(
            descriptor,
            repeat_count=repeat_count,
            reason="typed_exclusion",
            dispersion_scalar=None,
        )
    threshold = registry.repeatability_thresholds[descriptor.repeatability_rule]
    if scalar > threshold:
        return _unavailable_metric(
            descriptor,
            repeat_count=repeat_count,
            reason="typed_exclusion",
            dispersion_scalar=scalar,
        )
    value = statistics.fmean(float(item) for item in values)
    return {
        "metric_key": descriptor.key,
        "ncu_metric": descriptor.ncu_metric,
        "unit": descriptor.unit,
        "outcome": {"status": "available", "value": value},
        "repeat_metadata": {
            "repeat_count": repeat_count,
            "aggregation": (
                "mean_fraction" if descriptor.unit == "pct" else "mean_nonneg"
            ),
            "dispersion": {
                "status": "available",
                "rule": descriptor.repeatability_rule,
                "scalar": scalar,
            },
        },
    }


def build_production_matched_labels(
    *,
    source: ProductionLabelSetAuthority,
    crosswalk: VerifiedProductionCrosswalk,
    repeats: Sequence[ValidatedNcuRepeat],
) -> ProductionMatchedLabelBuild:
    """Aggregate exact launch-key/repeat cells into canonical matched_labels.v1."""

    if not isinstance(source, ProductionLabelSetAuthority):
        _fail("production label-set authority is unavailable")
    if not isinstance(crosswalk, VerifiedProductionCrosswalk):
        _fail("verified production crosswalk is unavailable")
    if (
        source.binary_architecture != crosswalk.binary_architecture
        or source.gpu_sku != crosswalk.gpu_sku
    ):
        _fail("production label source and crosswalk architecture/SKU mismatch")
    _load_repeat_manifest_schema()
    _load_governing_ncu_schemas()
    _require_source_schema(source)
    material = _require_complete_repeats(
        repeats, source=source, crosswalk=crosswalk
    )
    registry = load_metric_registry()
    if tuple(item.key for item in registry.metrics) != METRIC_KEYS:
        _fail("metric registry is not the exact retained 7+1 projection")

    launches: list[dict[str, Any]] = []
    for launch_index, crosswalk_launch in enumerate(crosswalk.launches):
        per_repeat = [
            repeat.launches[launch_index].by_name() for repeat in material
        ]
        launches.append(
            {
                "launch_key": crosswalk_launch.launch_key.to_model(),
                "kernel_family": crosswalk_launch.kernel_family,
                "metrics": [
                    _aggregate_metric(
                        descriptor, per_repeat=per_repeat, registry=registry
                    )
                    for descriptor in registry.metrics
                ],
            }
        )

    document: dict[str, Any] = {
        "schema_version": "matched_labels.v1",
        "source_label_identity": {
            "label_set_id": source.label_set_id,
            "label_set_sha256": source.label_set_sha256,
            "label_schema_version": source.label_schema_version,
            "label_schema_sha256": source.label_schema_sha256,
            "source_manifest_id": source.source_manifest_id,
            "source_manifest_sha256": source.source_manifest_sha256,
            "authority_role": "independent_label_source_not_result_derived",
        },
        "crosswalk_identity": {
            "crosswalk_id": crosswalk.crosswalk_id,
            "crosswalk_sha256": crosswalk.payload_sha256,
            "crosswalk_version": 2,
            "authority_role": "independently_matched_not_result_authorized",
        },
        "architecture": {
            "binary_architecture": source.binary_architecture,
            "gpu_sku": source.gpu_sku,
        },
        "launches": launches,
    }
    document["matched_labels_sha256"] = canonical_sha256(
        document, "matched_labels_sha256"
    )
    authority = MatchedLabelAuthority(
        matched_labels_sha256=document["matched_labels_sha256"],
        label_set_id=source.label_set_id,
        label_set_sha256=source.label_set_sha256,
        crosswalk_id=crosswalk.crosswalk_id,
        crosswalk_sha256=crosswalk.payload_sha256,
    )
    validate_matched_labels(document, authority=authority)
    payload = canonical_json_bytes(document)
    return ProductionMatchedLabelBuild(
        document=copy.deepcopy(document),
        payload=payload,
        crosswalk=crosswalk,
        source=source,
        registry=registry,
        _construction_token=_PRIVATE_BUILD_TOKEN,
    )


def _schema_identity(
    *, schema_id: str, schema_version: int | str, schema_sha256: str
) -> dict[str, Any]:
    return {
        "schema_id": schema_id,
        "schema_version": schema_version,
        "schema_sha256": schema_sha256,
    }


def _current_builder_sha256() -> str:
    payload, _snapshot = _read_retained(Path(__file__), "production label builder")
    return sha256_bytes(payload)


def _load_authority_schema() -> tuple[bytes, dict[str, Any]]:
    payload, snapshot = _read_retained(AUTHORITY_SCHEMA_PATH, "authority schema")
    if snapshot.payload_sha256 != AUTHORITY_SCHEMA_SHA256:
        _fail("production authority schema SHA-256 mismatch")
    schema = _decode_schema_json(payload, "production authority schema")
    if (
        type(schema) is not dict
        or schema.get("$id") != AUTHORITY_SCHEMA_ID
        or schema.get("$schema")
        != "https://json-schema.org/draft/2020-12/schema"
    ):
        _fail("production authority schema identity is invalid")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ProductionLabelError(
            f"production authority schema is invalid: {exc}"
        ) from exc
    return payload, schema


def _validate_authority_schema(document: dict[str, Any]) -> None:
    _payload, schema = _load_authority_schema()
    try:
        Draft202012Validator(schema).validate(document)
    except ValidationError as exc:
        location = "/" + "/".join(str(item) for item in exc.absolute_path)
        detail = (
            "production authority schema validation failed at "
            f"{location}: {exc.message}"
        )
        raise ProductionLabelError(detail) from exc


def create_production_matched_label_authority(
    build: ProductionMatchedLabelBuild,
) -> ProductionMatchedLabelArtifacts:
    """Create a canonical detached authority for one exact matched-label file."""

    if not isinstance(build, ProductionMatchedLabelBuild):
        _fail("production matched-label build proof is unavailable")
    source = build.source
    crosswalk = build.crosswalk
    _load_repeat_manifest_schema()
    _load_governing_ncu_schemas()
    builder_sha256 = _current_builder_sha256()
    _require_sha256(builder_sha256, "production builder SHA-256")
    _require_source_schema(source)
    matched_schema_bytes, matched_schema_snapshot = _read_retained(
        MATCHED_LABEL_SCHEMA_PATH, "matched-label schema"
    )
    if matched_schema_snapshot.payload_sha256 != MATCHED_LABEL_SCHEMA_SHA256:
        _fail("matched-label schema SHA-256 mismatch")
    matched_schema = _decode_schema_json(
        matched_schema_bytes, "matched-label schema"
    )
    if (
        type(matched_schema) is not dict
        or matched_schema.get("$id") != MATCHED_LABEL_SCHEMA_ID
    ):
        _fail("matched-label schema identity mismatch")

    registry = build.registry
    document: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "authority_role": AUTHORITY_ROLE,
        "matched_labels": {
            "schema_version": "matched_labels.v1",
            "file_sha256": sha256_bytes(build.payload),
            "size_bytes": len(build.payload),
            "canonical_seal_sha256": build.document["matched_labels_sha256"],
        },
        "source_label_identity": {
            "label_set_id": source.label_set_id,
            "label_set_sha256": source.label_set_sha256,
            "label_schema_version": source.label_schema_version,
            "label_schema_sha256": source.label_schema_sha256,
            "source_manifest_id": source.source_manifest_id,
            "source_manifest_sha256": source.source_manifest_sha256,
            "expected_repeat_count": source.expected_repeat_count,
            "authority_role": "independent_retained_ncu_label_source",
        },
        "crosswalk_identity": {
            "crosswalk_id": crosswalk.crosswalk_id,
            "crosswalk_file_sha256": crosswalk.payload_sha256,
            "crosswalk_model_seal_sha256": crosswalk.model_seal_sha256,
            "crosswalk_version": 2,
            "authority_role": "verified_production_bijection_not_result_authorized",
        },
        "schemas": {
            "authority": _schema_identity(
                schema_id=AUTHORITY_SCHEMA_ID,
                schema_version=AUTHORITY_SCHEMA_VERSION,
                schema_sha256=AUTHORITY_SCHEMA_SHA256,
            ),
            "matched_labels": _schema_identity(
                schema_id=MATCHED_LABEL_SCHEMA_ID,
                schema_version="matched_labels.v1",
                schema_sha256=MATCHED_LABEL_SCHEMA_SHA256,
            ),
            "production_ncu_repeat_manifest": _schema_identity(
                schema_id=REPEAT_MANIFEST_SCHEMA_ID,
                schema_version=REPEAT_MANIFEST_SCHEMA_VERSION,
                schema_sha256=REPEAT_MANIFEST_SCHEMA_SHA256,
            ),
            "source_label": _schema_identity(
                schema_id=sidecar_schema_authority.SCHEMA_ID,
                schema_version=5,
                schema_sha256=source.label_schema_sha256,
            ),
            "ncu_launch_inventory": _schema_identity(
                schema_id=NCU_INVENTORY_SCHEMA_ID,
                schema_version=3,
                schema_sha256=NCU_INVENTORY_SCHEMA_SHA256,
            ),
            "ncu_launch_metrics": _schema_identity(
                schema_id=NCU_METRICS_SCHEMA_ID,
                schema_version=1,
                schema_sha256=NCU_METRICS_SCHEMA_SHA256,
            ),
        },
        "registry_identity": {
            "projection_id": "metric_registry.7plus1.v1",
            "projection_sha256": registry.projection_sha256,
            "source_registry_version": registry.source_registry_version,
            "source_registry_sha256": registry.source_registry_sha256,
            "source_label_schema_version": registry.source_label_schema_version,
            "metric_keys": [item.key for item in registry.metrics],
        },
        "builder_identity": {
            "builder_id": BUILDER_ID,
            "builder_version": BUILDER_VERSION,
            "builder_sha256": builder_sha256,
        },
        "architecture": {
            "binary_architecture": source.binary_architecture,
            "gpu_sku": source.gpu_sku,
        },
        "producer_content_manifest": {
            "manifest_id": source.producer_content_manifest_id,
            "content_manifest_sha256": source.producer_content_manifest_sha256,
            "authority_role": "external_producer_content_manifest_anchor",
        },
    }
    document["authority_sha256"] = canonical_sha256(document, "authority_sha256")
    _validate_authority_schema(document)
    authority_bytes = canonical_json_bytes(document)
    return ProductionMatchedLabelArtifacts(
        matched_labels_document=copy.deepcopy(build.document),
        matched_labels_bytes=build.payload,
        authority_document=copy.deepcopy(document),
        authority_bytes=authority_bytes,
        _construction_token=_PRIVATE_ARTIFACT_TOKEN,
    )


def _read_authority_path(path: str | Path) -> bytes:
    payload, _snapshot = _read_retained(path, "detached production authority")
    if len(payload) > _MAX_AUTHORITY_BYTES:
        _fail("detached production authority exceeds its byte bound")
    return payload


def _read_label_path(path: str | Path) -> bytes:
    payload, _snapshot = _read_retained(path, "production matched-label payload")
    if len(payload) > _MAX_MATCHED_LABEL_BYTES:
        _fail("production matched-label payload exceeds its byte bound")
    return payload


def _require_schema_files(authority: Mapping[str, Any]) -> None:
    schemas = authority["schemas"]
    expected = {
        "authority": _schema_identity(
            schema_id=AUTHORITY_SCHEMA_ID,
            schema_version=AUTHORITY_SCHEMA_VERSION,
            schema_sha256=AUTHORITY_SCHEMA_SHA256,
        ),
        "matched_labels": _schema_identity(
            schema_id=MATCHED_LABEL_SCHEMA_ID,
            schema_version="matched_labels.v1",
            schema_sha256=MATCHED_LABEL_SCHEMA_SHA256,
        ),
        "production_ncu_repeat_manifest": _schema_identity(
            schema_id=REPEAT_MANIFEST_SCHEMA_ID,
            schema_version=REPEAT_MANIFEST_SCHEMA_VERSION,
            schema_sha256=REPEAT_MANIFEST_SCHEMA_SHA256,
        ),
        "source_label": _schema_identity(
            schema_id=sidecar_schema_authority.SCHEMA_ID,
            schema_version=5,
            schema_sha256=sidecar_schema_authority.SCHEMA_SHA256,
        ),
        "ncu_launch_inventory": _schema_identity(
            schema_id=NCU_INVENTORY_SCHEMA_ID,
            schema_version=3,
            schema_sha256=NCU_INVENTORY_SCHEMA_SHA256,
        ),
        "ncu_launch_metrics": _schema_identity(
            schema_id=NCU_METRICS_SCHEMA_ID,
            schema_version=1,
            schema_sha256=NCU_METRICS_SCHEMA_SHA256,
        ),
    }
    if schemas != expected:
        _fail("detached authority schema identities do not match retained contracts")
    _load_repeat_manifest_schema()
    _load_governing_ncu_schemas()
    source_schema, source_snapshot = _read_retained(
        SOURCE_LABEL_SCHEMA_PATH, "v5 source label schema"
    )
    if source_snapshot.payload_sha256 != sidecar_schema_authority.SCHEMA_SHA256:
        _fail("retained v5 source label schema hash drift")
    try:
        sidecar_schema_authority.parse_authoritative_schema_bytes(source_schema)
    except ValueError as exc:
        raise ProductionLabelError(
            f"retained v5 source schema is invalid: {exc}"
        ) from exc
    matched_schema, matched_snapshot = _read_retained(
        MATCHED_LABEL_SCHEMA_PATH, "matched-label schema"
    )
    if matched_snapshot.payload_sha256 != MATCHED_LABEL_SCHEMA_SHA256:
        _fail("retained matched-label schema hash drift")
    matched_model = _decode_schema_json(matched_schema, "matched-label schema")
    if (
        type(matched_model) is not dict
        or matched_model.get("$id") != MATCHED_LABEL_SCHEMA_ID
    ):
        _fail("retained matched-label schema identity drift")


def _require_registry_identity(authority: Mapping[str, Any]) -> MetricRegistry:
    registry = load_metric_registry()
    expected = {
        "projection_id": "metric_registry.7plus1.v1",
        "projection_sha256": registry.projection_sha256,
        "source_registry_version": registry.source_registry_version,
        "source_registry_sha256": registry.source_registry_sha256,
        "source_label_schema_version": registry.source_label_schema_version,
        "metric_keys": [item.key for item in registry.metrics],
    }
    if authority["registry_identity"] != expected:
        _fail("detached authority registry identity does not match retained 7+1")
    return registry


def _require_pin_matches(
    authority: Mapping[str, Any], pin: ProductionAuthorityPin
) -> None:
    expected_pairs = (
        (
            pin.producer_content_manifest_sha256,
            authority["producer_content_manifest"]["content_manifest_sha256"],
            "producer content manifest",
        ),
        (
            pin.label_set_sha256,
            authority["source_label_identity"]["label_set_sha256"],
            "label set",
        ),
        (
            pin.source_manifest_sha256,
            authority["source_label_identity"]["source_manifest_sha256"],
            "source manifest",
        ),
        (
            pin.crosswalk_file_sha256,
            authority["crosswalk_identity"]["crosswalk_file_sha256"],
            "crosswalk file",
        ),
        (
            pin.builder_sha256,
            authority["builder_identity"]["builder_sha256"],
            "builder",
        ),
    )
    for expected, observed, label in expected_pairs:
        if expected is not None and expected != observed:
            _fail(f"detached authority {label} differs from independent pin")


def authenticate_production_matched_label_authority(
    authority_path: str | Path,
    *,
    authority_pin: ProductionAuthorityPin,
) -> dict[str, Any]:
    """Authenticate detached authority bytes without opening label payload bytes."""

    if not isinstance(authority_pin, ProductionAuthorityPin):
        _fail("independent production authority pin is unavailable")
    authority_bytes = _read_authority_path(authority_path)
    observed_file_hash = sha256_bytes(authority_bytes)
    if observed_file_hash != authority_pin.authority_file_sha256:
        _fail("detached production authority file SHA-256 mismatch")
    authority = _decode_json(authority_bytes, "detached production authority")
    if type(authority) is not dict:
        _fail("detached production authority root is not an object")
    if canonical_json_bytes(authority) != authority_bytes:
        _fail("detached production authority is not canonical JSON")
    _validate_authority_schema(authority)
    expected_seal = canonical_sha256(authority, "authority_sha256")
    if authority["authority_sha256"] != expected_seal:
        _fail("detached production authority canonical seal mismatch")
    _require_pin_matches(authority, authority_pin)
    _require_schema_files(authority)
    _require_registry_identity(authority)
    if authority["source_label_identity"]["label_schema_sha256"] != (
        authority["schemas"]["source_label"]["schema_sha256"]
    ):
        _fail("detached authority source label schema binding mismatch")
    if authority["architecture"] != {
        "binary_architecture": "sm_86",
        "gpu_sku": "rtx_3090_ti",
    }:
        _fail("detached authority architecture/SKU is unsupported")
    return copy.deepcopy(authority)


def _require_payload_matches_authority(
    labels: Mapping[str, Any], authority: Mapping[str, Any]
) -> None:
    source = labels["source_label_identity"]
    authorized_source = authority["source_label_identity"]
    expected_source = {
        "label_set_id": authorized_source["label_set_id"],
        "label_set_sha256": authorized_source["label_set_sha256"],
        "label_schema_version": authorized_source["label_schema_version"],
        "label_schema_sha256": authorized_source["label_schema_sha256"],
        "source_manifest_id": authorized_source["source_manifest_id"],
        "source_manifest_sha256": authorized_source["source_manifest_sha256"],
        "authority_role": "independent_label_source_not_result_derived",
    }
    if source != expected_source:
        _fail("matched-label source identity differs from detached authority")
    authorized_crosswalk = authority["crosswalk_identity"]
    expected_crosswalk = {
        "crosswalk_id": authorized_crosswalk["crosswalk_id"],
        "crosswalk_sha256": authorized_crosswalk["crosswalk_file_sha256"],
        "crosswalk_version": authorized_crosswalk["crosswalk_version"],
        "authority_role": "independently_matched_not_result_authorized",
    }
    if labels["crosswalk_identity"] != expected_crosswalk:
        _fail("matched-label crosswalk identity differs from detached authority")
    if labels["architecture"] != authority["architecture"]:
        _fail("matched-label architecture/SKU differs from detached authority")
    repeat_count = authorized_source["expected_repeat_count"]
    for launch in labels["launches"]:
        for metric in launch["metrics"]:
            if metric["repeat_metadata"]["repeat_count"] != repeat_count:
                _fail(
                    "matched-label repeat count differs from source manifest authority"
                )


def load_authorized_production_matched_labels(
    labels_path: str | Path,
    authority_path: str | Path,
    *,
    authority_pin: ProductionAuthorityPin,
) -> MatchedLabels:
    """Authenticate authority first, then open and freeze exact production labels."""

    authority = authenticate_production_matched_label_authority(
        authority_path, authority_pin=authority_pin
    )
    # This call must remain after every detached-authority authentication step.
    labels_bytes = _read_label_path(labels_path)
    label_identity = authority["matched_labels"]
    if len(labels_bytes) != label_identity["size_bytes"]:
        _fail("production matched-label file size differs from detached authority")
    if sha256_bytes(labels_bytes) != label_identity["file_sha256"]:
        _fail("production matched-label file SHA-256 differs from detached authority")
    labels = _decode_json(labels_bytes, "production matched-label payload")
    if type(labels) is not dict:
        _fail("production matched-label root is not an object")
    if canonical_json_bytes(labels) != labels_bytes:
        _fail("production matched-label payload is not canonical JSON")
    matched_authority = MatchedLabelAuthority(
        matched_labels_sha256=label_identity["canonical_seal_sha256"],
        label_set_id=authority["source_label_identity"]["label_set_id"],
        label_set_sha256=authority["source_label_identity"]["label_set_sha256"],
        crosswalk_id=authority["crosswalk_identity"]["crosswalk_id"],
        crosswalk_sha256=authority["crosswalk_identity"]["crosswalk_file_sha256"],
    )
    validated = validate_matched_labels(labels, authority=matched_authority)
    if validated["matched_labels_sha256"] != label_identity["canonical_seal_sha256"]:
        _fail("matched-label canonical seal differs from detached authority")
    _require_payload_matches_authority(validated, authority)
    return _freeze_labels(validated, fixture_id=None)


def load_production_matched_labels(
    labels_path: str | Path,
    authority_path: str | Path,
    *,
    authority_pin: ProductionAuthorityPin,
) -> MatchedLabels:
    """Public production alias; detached authority remains mandatory."""

    return load_authorized_production_matched_labels(
        labels_path, authority_path, authority_pin=authority_pin
    )


def build_production_matched_label_artifacts(
    *,
    source: ProductionLabelSetAuthority,
    crosswalk: VerifiedProductionCrosswalk,
    repeats: Sequence[ValidatedNcuRepeat],
) -> ProductionMatchedLabelArtifacts:
    """Build matched labels and their detached authority as one exact pair."""

    build = build_production_matched_labels(
        source=source, crosswalk=crosswalk, repeats=repeats
    )
    return create_production_matched_label_authority(build)


__all__ = [
    "AUTHORITY_SCHEMA_ID",
    "AUTHORITY_SCHEMA_PATH",
    "AUTHORITY_SCHEMA_SHA256",
    "AUTHORITY_SCHEMA_VERSION",
    "BUILDER_ID",
    "BUILDER_VERSION",
    "PRODUCTION_REPEAT_MANIFEST_AUTHORITIES",
    "REPEAT_MANIFEST_SCHEMA_ID",
    "REPEAT_MANIFEST_SCHEMA_PATH",
    "REPEAT_MANIFEST_SCHEMA_SHA256",
    "REPEAT_MANIFEST_SCHEMA_VERSION",
    "ProductionAuthorityPin",
    "ProductionLabelError",
    "ProductionLabelSetAuthority",
    "ProductionMatchedLabelArtifacts",
    "ProductionMatchedLabelBuild",
    "ValidatedNcuRepeat",
    "VerifiedProductionCrosswalk",
    "authenticate_production_crosswalk",
    "authenticate_production_matched_label_authority",
    "build_production_matched_label_artifacts",
    "build_production_matched_labels",
    "create_production_matched_label_authority",
    "load_authorized_production_matched_labels",
    "load_production_matched_labels",
    "validate_retained_ncu_repeat",
]
