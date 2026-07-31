"""Canonical, duplicate-rejecting seals and retained-byte identities."""

from __future__ import annotations

import re
import stat
import unicodedata
from collections.abc import Callable, Hashable, Iterable
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Any, TypeVar

from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    MetadataSnapshot,
    PinnedRegularSnapshot,
    canonical_json_bytes,
    read_regular_snapshot,
    seal_model,
    sha256_bytes,
    strict_json_loads,
    verify_model_seal,
)

from ._common import (
    AuthorityBlockerCode,
    authority_unavailable,
)


MODEL_SEAL_SCHEMA_VERSION = 1
MODEL_SEAL_ARTIFACT_KIND = "EVALUATOR_MODEL_SEAL"
MODEL_SEAL_SCHEMA_ID = (
    "https://gpu-simulator.invalid/eval/schemas/model_seal.v1.schema.json"
)
MODEL_SEAL_SCHEMA_SHA256 = "696ab16e6ad547dab1c91f49f485856745fd5b6242f528c9925b7a389bdd60c7"
_MAX_SAFE_INTEGER = (1 << 53) - 1
PRODUCTION_SNAPSHOT_ACQUISITION = "PINNED_REGULAR_NOFOLLOW_V1"
FIXTURE_SNAPSHOT_ACQUISITION = "FIXTURE_CALLER_BYTES_V1"
_SNAPSHOT_ACQUISITIONS = frozenset(
    {PRODUCTION_SNAPSHOT_ACQUISITION, FIXTURE_SNAPSHOT_ACQUISITION}
)
_PRODUCTION_CONSTRUCTION_TOKEN = object()
_FIXTURE_CONSTRUCTION_TOKEN = object()
_ARTIFACT_KIND = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_T = TypeVar("_T")
_MODEL_SEAL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "schema_sha256",
        "sealed_schema_version",
        "sealed_artifact_kind",
        "sealed_schema_sha256",
        "sealed_model_sha256",
        "model_seal_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class RetainedPayloadSnapshot:
    """One payload identity with an explicit acquisition provenance class."""

    acquisition: str
    payload_sha256: str
    size_bytes: int
    metadata: MetadataSnapshot
    snapshot_identity_sha256: str
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        expected_token = (
            _PRODUCTION_CONSTRUCTION_TOKEN
            if self.acquisition == PRODUCTION_SNAPSHOT_ACQUISITION
            else _FIXTURE_CONSTRUCTION_TOKEN
        )
        if _construction_token is not expected_token:
            raise authority_unavailable(
                AuthorityBlockerCode.RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED,
                "retained snapshot construction did not pass through its governed "
                "acquisition API",
            )
        if self.acquisition not in _SNAPSHOT_ACQUISITIONS:
            raise ImmutableArtifactError("retained snapshot acquisition is invalid")
        require_sha256(self.payload_sha256, "payload_sha256")
        if type(self.size_bytes) is not int or not 0 <= self.size_bytes <= _MAX_SAFE_INTEGER:
            raise ImmutableArtifactError("snapshot size_bytes is invalid")
        metadata = _require_metadata(self.metadata)
        if not stat.S_ISREG(metadata[2]):
            raise ImmutableArtifactError("retained snapshot is not a regular file")
        if metadata[3] != 1:
            raise ImmutableArtifactError("retained snapshot must have one link")
        if metadata[4] != self.size_bytes:
            raise ImmutableArtifactError("retained snapshot size does not match metadata")
        require_sha256(self.snapshot_identity_sha256, "snapshot_identity_sha256")
        if self.snapshot_identity_sha256 != _snapshot_identity_sha256(
            self.acquisition, self.payload_sha256, self.size_bytes, metadata
        ):
            raise ImmutableArtifactError("retained snapshot identity SHA-256 mismatch")

    @classmethod
    def read_path(cls, path: Path) -> tuple[bytes, "RetainedPayloadSnapshot"]:
        """Read exact bytes through the shared descriptor/no-follow implementation."""

        pinned = read_regular_snapshot(Path(path))
        return pinned.payload, cls._from_exact_payload(
            pinned.payload,
            pinned.metadata,
            acquisition=PRODUCTION_SNAPSHOT_ACQUISITION,
            construction_token=_PRODUCTION_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def from_path(cls, path: Path) -> "RetainedPayloadSnapshot":
        """Acquire a production snapshot through an actual no-follow read."""

        _payload, snapshot = cls.read_path(path)
        return snapshot

    @classmethod
    def from_pinned_regular(
        cls, pinned: PinnedRegularSnapshot
    ) -> "RetainedPayloadSnapshot":
        """Reject a transferable bytes/metadata object as acquisition proof."""

        del pinned
        raise authority_unavailable(
            AuthorityBlockerCode.RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED,
            "PinnedRegularSnapshot is caller-constructible and cannot attest how bytes "
            "were acquired; use read_path for production acquisition",
        )

    @classmethod
    def from_fixture_payload(
        cls, payload: bytes, metadata: MetadataSnapshot
    ) -> "RetainedPayloadSnapshot":
        """Create a non-production identity for pure structural fixtures only."""

        return cls._from_exact_payload(
            payload,
            metadata,
            acquisition=FIXTURE_SNAPSHOT_ACQUISITION,
            construction_token=_FIXTURE_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def from_payload(
        cls, payload: bytes, metadata: MetadataSnapshot
    ) -> "RetainedPayloadSnapshot":
        """Reject the retired ambiguous bytes-plus-metadata constructor."""

        del payload, metadata
        raise authority_unavailable(
            AuthorityBlockerCode.RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED,
            "from_payload cannot prove a descriptor/no-follow acquisition; use "
            "read_path for production or from_fixture_payload for a fixture",
        )

    @classmethod
    def _from_exact_payload(
        cls,
        payload: bytes,
        metadata: MetadataSnapshot,
        *,
        acquisition: str,
        construction_token: object,
    ) -> "RetainedPayloadSnapshot":
        if type(payload) is not bytes:
            raise ImmutableArtifactError("retained payload must be exact bytes")
        metadata = _require_metadata(metadata)
        digest = sha256_bytes(payload)
        return cls(
            acquisition=acquisition,
            payload_sha256=digest,
            size_bytes=len(payload),
            metadata=metadata,
            snapshot_identity_sha256=_snapshot_identity_sha256(
                acquisition, digest, len(payload), metadata
            ),
            _construction_token=construction_token,
        )

    @property
    def is_production_acquisition(self) -> bool:
        return self.acquisition == PRODUCTION_SNAPSHOT_ACQUISITION

    @property
    def is_fixture_acquisition(self) -> bool:
        return self.acquisition == FIXTURE_SNAPSHOT_ACQUISITION

    def require_production_acquisition(self, label: str) -> None:
        if not self.is_production_acquisition:
            raise authority_unavailable(
                AuthorityBlockerCode.RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED,
                f"{label} was not read through a descriptor/no-follow acquisition",
            )

    def require_fixture_acquisition(self, label: str) -> None:
        if not self.is_fixture_acquisition:
            raise ImmutableArtifactError(
                f"{label} must use the explicit fixture-only acquisition"
            )

    def verify_payload(self, payload: bytes) -> None:
        if type(payload) is not bytes:
            raise ImmutableArtifactError("retained payload must be exact bytes")
        if len(payload) != self.size_bytes or sha256_bytes(payload) != self.payload_sha256:
            raise ImmutableArtifactError("retained payload does not match snapshot")

    def to_model(self) -> dict[str, Any]:
        return {
            "acquisition": self.acquisition,
            "payload_sha256": self.payload_sha256,
            "size_bytes": self.size_bytes,
            "metadata": {
                "device": str(self.metadata[0]),
                "inode": str(self.metadata[1]),
                "mode": str(self.metadata[2]),
                "link_count": str(self.metadata[3]),
                "size_bytes": str(self.metadata[4]),
                "mtime_ns": str(self.metadata[5]),
                "ctime_ns": str(self.metadata[6]),
            },
            "snapshot_identity_sha256": self.snapshot_identity_sha256,
        }

    @classmethod
    def from_model(cls, value: object) -> "RetainedPayloadSnapshot":
        require_exact_keys(
            value,
            {
                "acquisition",
                "payload_sha256",
                "size_bytes",
                "metadata",
                "snapshot_identity_sha256",
            },
            "retained payload snapshot",
        )
        assert isinstance(value, dict)
        if value["acquisition"] not in _SNAPSHOT_ACQUISITIONS:
            raise ImmutableArtifactError("retained snapshot acquisition is invalid")
        if value["acquisition"] == PRODUCTION_SNAPSHOT_ACQUISITION:
            raise authority_unavailable(
                AuthorityBlockerCode.RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED,
                "serialized snapshot metadata cannot attest a live descriptor/no-follow read",
            )
        metadata_model = value["metadata"]
        require_exact_keys(
            metadata_model,
            {
                "device",
                "inode",
                "mode",
                "link_count",
                "size_bytes",
                "mtime_ns",
                "ctime_ns",
            },
            "retained snapshot metadata",
        )
        assert isinstance(metadata_model, dict)
        metadata = (
            _parse_decimal(metadata_model["device"], "snapshot device"),
            _parse_decimal(metadata_model["inode"], "snapshot inode"),
            _parse_decimal(metadata_model["mode"], "snapshot mode"),
            _parse_decimal(metadata_model["link_count"], "snapshot link_count"),
            _parse_decimal(metadata_model["size_bytes"], "snapshot metadata size"),
            _parse_decimal(metadata_model["mtime_ns"], "snapshot mtime_ns"),
            _parse_decimal(metadata_model["ctime_ns"], "snapshot ctime_ns"),
        )
        return cls(
            acquisition=value["acquisition"],
            payload_sha256=value["payload_sha256"],
            size_bytes=value["size_bytes"],
            metadata=metadata,
            snapshot_identity_sha256=value["snapshot_identity_sha256"],
            _construction_token=_FIXTURE_CONSTRUCTION_TOKEN,
        )


def strict_json_object(payload: bytes, *, label: str = "evaluator JSON") -> dict[str, Any]:
    value = strict_json_loads(payload, label=label)
    if type(value) is not dict:
        raise ImmutableArtifactError(f"{label} must be an exact JSON object")
    return value


def canonical_model_bytes(model: dict[str, Any]) -> bytes:
    if type(model) is not dict:
        raise ImmutableArtifactError("evaluator model must be an exact JSON object")
    return canonical_json_bytes(model)


def seal_evaluator_model(unsigned: dict[str, Any]) -> dict[str, Any]:
    _require_model_header(unsigned)
    return seal_model(unsigned)


def verify_retained_schema(
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    *,
    expected_sha256: str,
    expected_id: str,
) -> dict[str, Any]:
    """Validate the exact pinned schema bytes used by a runtime decision."""

    if not isinstance(schema_snapshot, RetainedPayloadSnapshot):
        raise ImmutableArtifactError("schema snapshot is unavailable")
    schema_snapshot.require_production_acquisition("retained evaluator schema")
    schema_snapshot.verify_payload(schema_bytes)
    require_sha256(expected_sha256, "expected schema SHA-256")
    if schema_snapshot.payload_sha256 != expected_sha256:
        raise ImmutableArtifactError("retained schema SHA-256 mismatch")
    schema = strict_json_object(schema_bytes, label="retained evaluator schema")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ImmutableArtifactError("retained schema draft is invalid")
    if schema.get("$id") != expected_id:
        raise ImmutableArtifactError("retained schema identity mismatch")
    closed_object = (
        schema.get("type") == "object"
        and schema.get("additionalProperties") is False
    )
    closed_union = type(schema.get("oneOf")) is list and type(schema.get("$defs")) is dict
    if not (closed_object or closed_union):
        raise ImmutableArtifactError("retained evaluator schema is not closed")
    return schema


def verify_evaluator_model(
    model: dict[str, Any],
    *,
    expected_schema_version: int,
    expected_artifact_kind: str,
    expected_schema_sha256: str,
    expected_schema_id: str,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> None:
    verify_retained_schema(
        schema_bytes,
        schema_snapshot,
        expected_sha256=expected_schema_sha256,
        expected_id=expected_schema_id,
    )
    schema_version, artifact_kind, schema_sha256 = _require_model_header(model)
    if schema_version != expected_schema_version:
        raise ImmutableArtifactError("evaluator model schema_version mismatch")
    if artifact_kind != expected_artifact_kind:
        raise ImmutableArtifactError("evaluator model artifact_kind mismatch")
    if schema_sha256 != expected_schema_sha256:
        raise ImmutableArtifactError("evaluator model schema SHA-256 mismatch")
    verify_model_seal(model)


def create_detached_model_seal(
    model: dict[str, Any],
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    sealed_schema_bytes: bytes,
    sealed_schema_snapshot: RetainedPayloadSnapshot,
    sealed_schema_id: str,
) -> dict[str, Any]:
    verify_retained_schema(
        schema_bytes,
        schema_snapshot,
        expected_sha256=MODEL_SEAL_SCHEMA_SHA256,
        expected_id=MODEL_SEAL_SCHEMA_ID,
    )
    schema_version, artifact_kind, sealed_schema_sha256 = _require_model_header(model)
    verify_retained_schema(
        sealed_schema_bytes,
        sealed_schema_snapshot,
        expected_sha256=sealed_schema_sha256,
        expected_id=sealed_schema_id,
    )
    verify_model_seal(model)
    return seal_model(
        {
            "schema_version": MODEL_SEAL_SCHEMA_VERSION,
            "artifact_kind": MODEL_SEAL_ARTIFACT_KIND,
            "schema_sha256": MODEL_SEAL_SCHEMA_SHA256,
            "sealed_schema_version": schema_version,
            "sealed_artifact_kind": artifact_kind,
            "sealed_schema_sha256": sealed_schema_sha256,
            "sealed_model_sha256": sha256_bytes(canonical_json_bytes(model)),
        }
    )


def verify_detached_model_seal(
    model: dict[str, Any],
    seal: dict[str, Any],
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    sealed_schema_bytes: bytes,
    sealed_schema_snapshot: RetainedPayloadSnapshot,
    sealed_schema_id: str,
) -> None:
    loaded = _verify_detached_seal_structure(
        seal, schema_bytes=schema_bytes, schema_snapshot=schema_snapshot
    )
    schema_version, artifact_kind, sealed_schema_sha256 = _require_model_header(model)
    verify_retained_schema(
        sealed_schema_bytes,
        sealed_schema_snapshot,
        expected_sha256=sealed_schema_sha256,
        expected_id=sealed_schema_id,
    )
    verify_model_seal(model)
    if loaded["sealed_schema_version"] != schema_version:
        raise ImmutableArtifactError("sealed model schema identity mismatch")
    if loaded["sealed_artifact_kind"] != artifact_kind:
        raise ImmutableArtifactError("sealed model kind identity mismatch")
    if loaded["sealed_schema_sha256"] != sealed_schema_sha256:
        raise ImmutableArtifactError("sealed model schema SHA-256 mismatch")
    if loaded["sealed_model_sha256"] != sha256_bytes(canonical_json_bytes(model)):
        raise ImmutableArtifactError("sealed model SHA-256 mismatch")


def load_detached_model_seal(
    payload: bytes,
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    seal = strict_json_object(payload, label="detached evaluator model seal")
    return _verify_detached_seal_structure(
        seal, schema_bytes=schema_bytes, schema_snapshot=schema_snapshot
    )


def require_unique_identities(
    values: Iterable[_T],
    *,
    identity: Callable[[_T], Hashable],
    label: str,
) -> tuple[_T, ...]:
    material = tuple(values)
    observed: set[Hashable] = set()
    for value in material:
        key = identity(value)
        if key in observed:
            raise ImmutableArtifactError(f"duplicate {label} identity: {key!r}")
        observed.add(key)
    return material


def require_nfc(value: object, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not value and not allow_empty):
        raise ImmutableArtifactError(f"{label} must be a nonempty string")
    if unicodedata.normalize("NFC", value) != value:
        raise ImmutableArtifactError(f"{label} must be NFC-normalized")
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in value):
        raise ImmutableArtifactError(f"{label} contains a forbidden control or surrogate")
    return value


def require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ImmutableArtifactError(f"{label} must be lowercase 64-hex")
    return value


def require_exact_keys(value: object, keys: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != keys:
        raise ImmutableArtifactError(f"{label} fields are not exact")


def _verify_detached_seal_structure(
    seal: dict[str, Any],
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    verify_retained_schema(
        schema_bytes,
        schema_snapshot,
        expected_sha256=MODEL_SEAL_SCHEMA_SHA256,
        expected_id=MODEL_SEAL_SCHEMA_ID,
    )
    if type(seal) is not dict or frozenset(seal) != _MODEL_SEAL_KEYS:
        raise ImmutableArtifactError("detached model seal fields are not exact")
    version, artifact_kind, schema_sha256 = _require_model_header(seal)
    if version != MODEL_SEAL_SCHEMA_VERSION or artifact_kind != MODEL_SEAL_ARTIFACT_KIND:
        raise ImmutableArtifactError("detached model seal identity mismatch")
    if schema_sha256 != MODEL_SEAL_SCHEMA_SHA256:
        raise ImmutableArtifactError("detached model seal schema SHA-256 mismatch")
    if type(seal["sealed_schema_version"]) is not int or seal["sealed_schema_version"] < 1:
        raise ImmutableArtifactError("sealed_schema_version must be positive")
    _require_artifact_kind(seal["sealed_artifact_kind"])
    require_sha256(seal["sealed_schema_sha256"], "sealed_schema_sha256")
    require_sha256(seal["sealed_model_sha256"], "sealed_model_sha256")
    verify_model_seal(seal)
    return seal


def _require_model_header(model: object) -> tuple[int, str, str]:
    if type(model) is not dict:
        raise ImmutableArtifactError("evaluator model must be an exact JSON object")
    schema_version = model.get("schema_version")
    artifact_kind = model.get("artifact_kind")
    schema_sha256 = model.get("schema_sha256")
    if type(schema_version) is not int or schema_version < 1:
        raise ImmutableArtifactError("evaluator model schema_version must be positive")
    _require_artifact_kind(artifact_kind)
    require_sha256(schema_sha256, "schema_sha256")
    return schema_version, artifact_kind, schema_sha256


def _require_artifact_kind(value: object) -> str:
    if type(value) is not str or _ARTIFACT_KIND.fullmatch(value) is None:
        raise ImmutableArtifactError("artifact_kind is invalid")
    return value


def _require_metadata(value: object) -> MetadataSnapshot:
    if type(value) is not tuple or len(value) != 7:
        raise ImmutableArtifactError("retained metadata must be an exact seven-field tuple")
    if any(type(component) is not int or component < 0 for component in value):
        raise ImmutableArtifactError("retained metadata fields must be nonnegative integers")
    return value  # type: ignore[return-value]


def _parse_decimal(value: object, label: str) -> int:
    if (
        type(value) is not str
        or not value
        or (value != "0" and (value.startswith("0") or not value.isascii()))
        or not value.isdecimal()
    ):
        raise ImmutableArtifactError(f"{label} must be a canonical decimal string")
    return int(value)


def _snapshot_identity_sha256(
    acquisition: str,
    payload_sha256: str,
    size_bytes: int,
    metadata: MetadataSnapshot,
) -> str:
    if acquisition not in _SNAPSHOT_ACQUISITIONS:
        raise ImmutableArtifactError("retained snapshot acquisition is invalid")
    preimage = {
        "acquisition": acquisition,
        "payload_sha256": payload_sha256,
        "size_bytes": size_bytes,
        "metadata": [str(component) for component in metadata],
    }
    return sha256_bytes(canonical_json_bytes(preimage))
