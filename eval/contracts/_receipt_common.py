"""Pure primitives for the target-free evaluator receipt chain.

The chain uses canonical model hashes and explicit predecessor references rather
than wall-clock timestamps. All identifiers are ASCII tokens, never paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    canonical_json_bytes,
    sha256_bytes,
)

from .model_seal import (
    RetainedPayloadSnapshot,
    canonical_model_bytes,
    verify_evaluator_model,
)


IDENTITY_SCHEMA_VERSION = 1
CHAIN_SCHEMA_VERSION = 1
MAX_SAFE_INTEGER = (1 << 53) - 1
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*\Z")
_ARTIFACT_KIND = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")


class ReceiptPhase(str, Enum):
    PREREGISTRATION = "PREREGISTRATION_FIXTURE"
    PREDICTION_SEAL = "PREDICTION_SEAL_FIXTURE"
    LABEL_OPENING = "LABEL_OPENING_BLOCKED_FIXTURE"


_PHASE_ORDINAL = {
    ReceiptPhase.PREREGISTRATION: 0,
    ReceiptPhase.PREDICTION_SEAL: 1,
    ReceiptPhase.LABEL_OPENING: 2,
}
_PHASE_PREDECESSOR_KIND = {
    ReceiptPhase.PREREGISTRATION: None,
    ReceiptPhase.PREDICTION_SEAL: "EVALUATOR_PREREGISTRATION_FIXTURE",
    ReceiptPhase.LABEL_OPENING: "EVALUATOR_PREDICTION_SEAL_FIXTURE",
}
_PHASE_LIFETIMES = {
    ReceiptPhase.PREREGISTRATION: {
        "preregistration_bindings": "FIXTURE_SEALED_NONAUTHORITATIVE",
        "prediction_bindings": "NOT_CREATED",
        "label_bundle_binding": "BLOCKED_NOT_BOUND",
        "label_payload": "ABSENT",
        "scoring": "DISABLED",
    },
    ReceiptPhase.PREDICTION_SEAL: {
        "preregistration_bindings": "FIXTURE_SEALED_NONAUTHORITATIVE",
        "prediction_bindings": "FIXTURE_SEALED_NONAUTHORITATIVE",
        "label_bundle_binding": "BLOCKED_NOT_BOUND",
        "label_payload": "ABSENT",
        "scoring": "DISABLED",
    },
    ReceiptPhase.LABEL_OPENING: {
        "preregistration_bindings": "FIXTURE_SEALED_NONAUTHORITATIVE",
        "prediction_bindings": "FIXTURE_SEALED_NONAUTHORITATIVE",
        "label_bundle_binding": "FIXTURE_OPAQUE_DIGEST_UNADMITTED",
        "label_payload": "ABSENT",
        "scoring": "DISABLED",
    },
}


@dataclass(frozen=True, slots=True)
class RetainedSchemaInput:
    """Exact retained schema bytes supplied to otherwise pure receipt code."""

    schema_bytes: bytes
    schema_snapshot: RetainedPayloadSnapshot

    def __post_init__(self) -> None:
        if type(self.schema_bytes) is not bytes:
            raise ImmutableArtifactError("retained schema bytes are invalid")
        if not isinstance(self.schema_snapshot, RetainedPayloadSnapshot):
            raise ImmutableArtifactError("retained schema snapshot is invalid")
        self.schema_snapshot.require_production_acquisition("receipt schema")
        self.schema_snapshot.verify_payload(self.schema_bytes)


@dataclass(frozen=True, slots=True)
class ReceiptValidationSchemas:
    """Dependency-closed retained schemas for pure chain construction/validation."""

    preregistration: RetainedSchemaInput
    prediction_seal: RetainedSchemaInput
    label_opening: RetainedSchemaInput
    launch_crosswalk: RetainedSchemaInput
    launch_inventory: RetainedSchemaInput

    def __post_init__(self) -> None:
        for label, value in (
            ("preregistration", self.preregistration),
            ("prediction seal", self.prediction_seal),
            ("label opening", self.label_opening),
            ("launch crosswalk", self.launch_crosswalk),
            ("launch inventory", self.launch_inventory),
        ):
            if not isinstance(value, RetainedSchemaInput):
                raise ImmutableArtifactError(f"{label} retained schema input is invalid")


FIXTURE_ARTIFACT_AUTHORITY_STATUS = "FIXTURE_ONLY_OPAQUE_UNVALIDATED"


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """One explicitly fixture-only opaque identity with no authority claim."""

    artifact_kind: str
    artifact_schema_version: int
    artifact_id: str
    artifact_sha256: str
    authority_status: str = FIXTURE_ARTIFACT_AUTHORITY_STATUS
    identity_schema_version: int = IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.identity_schema_version) is not int or self.identity_schema_version != 1:
            raise ImmutableArtifactError("artifact identity schema version is unsupported")
        require_artifact_kind(self.artifact_kind, "artifact_kind")
        if type(self.artifact_schema_version) is not int or self.artifact_schema_version != 1:
            raise ImmutableArtifactError(
                "fixture artifact_schema_version must be exactly 1"
            )
        if self.authority_status != FIXTURE_ARTIFACT_AUTHORITY_STATUS:
            raise ImmutableArtifactError("fixture artifact authority status is invalid")
        require_ascii_token(self.artifact_id, "artifact_id")
        require_sha256(self.artifact_sha256, "artifact_sha256")

    def to_model(self) -> dict[str, Any]:
        return {
            "identity_schema_version": self.identity_schema_version,
            "artifact_kind": self.artifact_kind,
            "artifact_schema_version": self.artifact_schema_version,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "authority_status": self.authority_status,
        }

    @classmethod
    def from_model(
        cls, value: object, *, label: str = "artifact identity"
    ) -> "ArtifactIdentity":
        require_exact_keys(
            value,
            {
                "identity_schema_version",
                "artifact_kind",
                "artifact_schema_version",
                "artifact_id",
                "artifact_sha256",
                "authority_status",
            },
            label,
        )
        assert isinstance(value, dict)
        return cls(
            identity_schema_version=value["identity_schema_version"],
            artifact_kind=value["artifact_kind"],
            artifact_schema_version=value["artifact_schema_version"],
            artifact_id=value["artifact_id"],
            artifact_sha256=value["artifact_sha256"],
            authority_status=value["authority_status"],
        )


@dataclass(frozen=True, slots=True)
class EvaluatorModelIdentity:
    """Canonical identity for an already sealed evaluator model or receipt."""

    schema_version: int
    artifact_kind: str
    schema_sha256: str
    canonical_model_sha256: str
    model_seal_sha256: str
    identity_schema_version: int = IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.identity_schema_version) is not int or self.identity_schema_version != 1:
            raise ImmutableArtifactError("evaluator identity schema version is unsupported")
        require_safe_positive_integer(self.schema_version, "schema_version")
        require_artifact_kind(self.artifact_kind, "artifact_kind")
        require_sha256(self.schema_sha256, "schema_sha256")
        require_sha256(self.canonical_model_sha256, "canonical_model_sha256")
        require_sha256(self.model_seal_sha256, "model_seal_sha256")

    @classmethod
    def create(
        cls,
        model: dict[str, Any],
        *,
        expected_schema_version: int,
        expected_artifact_kind: str,
        expected_schema_sha256: str,
        expected_schema_id: str,
        retained_schema: RetainedSchemaInput,
    ) -> "EvaluatorModelIdentity":
        if not isinstance(retained_schema, RetainedSchemaInput):
            raise ImmutableArtifactError("retained evaluator schema input is invalid")
        verify_evaluator_model(
            model,
            expected_schema_version=expected_schema_version,
            expected_artifact_kind=expected_artifact_kind,
            expected_schema_sha256=expected_schema_sha256,
            expected_schema_id=expected_schema_id,
            schema_bytes=retained_schema.schema_bytes,
            schema_snapshot=retained_schema.schema_snapshot,
        )
        return cls(
            schema_version=expected_schema_version,
            artifact_kind=expected_artifact_kind,
            schema_sha256=expected_schema_sha256,
            canonical_model_sha256=sha256_bytes(canonical_model_bytes(model)),
            model_seal_sha256=model["model_seal_sha256"],
        )

    def to_model(self) -> dict[str, Any]:
        return {
            "identity_schema_version": self.identity_schema_version,
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "schema_sha256": self.schema_sha256,
            "canonical_model_sha256": self.canonical_model_sha256,
            "model_seal_sha256": self.model_seal_sha256,
        }

    @classmethod
    def from_model(
        cls, value: object, *, label: str = "evaluator model identity"
    ) -> "EvaluatorModelIdentity":
        require_exact_keys(
            value,
            {
                "identity_schema_version",
                "schema_version",
                "artifact_kind",
                "schema_sha256",
                "canonical_model_sha256",
                "model_seal_sha256",
            },
            label,
        )
        assert isinstance(value, dict)
        return cls(
            identity_schema_version=value["identity_schema_version"],
            schema_version=value["schema_version"],
            artifact_kind=value["artifact_kind"],
            schema_sha256=value["schema_sha256"],
            canonical_model_sha256=value["canonical_model_sha256"],
            model_seal_sha256=value["model_seal_sha256"],
        )


def phase_chain_model(
    *,
    phase: ReceiptPhase,
    chain_id_sha256: str,
    predecessor: EvaluatorModelIdentity | None,
) -> dict[str, Any]:
    """Return the one canonical phase/lifetime state for ``phase``."""

    if not isinstance(phase, ReceiptPhase):
        raise ImmutableArtifactError("receipt phase is invalid")
    require_sha256(chain_id_sha256, "chain_id_sha256")
    expected_kind = _PHASE_PREDECESSOR_KIND[phase]
    if expected_kind is None:
        if predecessor is not None:
            raise ImmutableArtifactError("preregistration phase cannot have a predecessor")
    else:
        if not isinstance(predecessor, EvaluatorModelIdentity):
            raise ImmutableArtifactError("receipt phase requires an exact predecessor")
        if predecessor.schema_version != 1 or predecessor.artifact_kind != expected_kind:
            raise ImmutableArtifactError("receipt predecessor phase identity mismatch")

    return {
        "chain_schema_version": CHAIN_SCHEMA_VERSION,
        "chain_mode": "FIXTURE_ONLY",
        "chain_id_sha256": chain_id_sha256,
        "phase": phase.value,
        "phase_ordinal": _PHASE_ORDINAL[phase],
        "ordering_basis": "PHASE_ORDINAL_AND_EXACT_PREDECESSOR_HASH",
        "predecessor_receipt": None if predecessor is None else predecessor.to_model(),
        "state_lifetime": dict(_PHASE_LIFETIMES[phase]),
    }


def require_exact_phase_chain(
    value: object,
    *,
    phase: ReceiptPhase,
    chain_id_sha256: str,
    predecessor: EvaluatorModelIdentity | None,
) -> None:
    expected = phase_chain_model(
        phase=phase,
        chain_id_sha256=chain_id_sha256,
        predecessor=predecessor,
    )
    if type(value) is not dict or canonical_json_bytes(value) != canonical_json_bytes(
        expected
    ):
        raise ImmutableArtifactError("receipt phase/state-lifetime chain mismatch")


def derive_domain_sha256(domain: str, value: dict[str, Any]) -> str:
    require_ascii_token(domain, "hash domain")
    if type(value) is not dict:
        raise ImmutableArtifactError("hash preimage must be an exact object")
    return sha256_bytes(canonical_json_bytes({"domain": domain, "value": value}))


def require_exact_artifact_kind(
    identity: ArtifactIdentity, expected_artifact_kind: str, *, label: str
) -> None:
    if identity.artifact_kind != expected_artifact_kind:
        raise ImmutableArtifactError(f"{label} artifact kind mismatch")


def require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ImmutableArtifactError(f"{label} must be lowercase 64-hex")
    return value


def require_safe_positive_integer(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_SAFE_INTEGER:
        raise ImmutableArtifactError(f"{label} must be a positive safe integer")
    return value


def require_safe_nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_SAFE_INTEGER:
        raise ImmutableArtifactError(f"{label} must be a nonnegative safe integer")
    return value


def require_ascii_token(value: object, label: str) -> str:
    if type(value) is not str or len(value) > 128 or _TOKEN.fullmatch(value) is None:
        raise ImmutableArtifactError(f"{label} must be a canonical ASCII token, not a path")
    return value


def require_artifact_kind(value: object, label: str) -> str:
    if type(value) is not str or _ARTIFACT_KIND.fullmatch(value) is None:
        raise ImmutableArtifactError(f"{label} is invalid")
    return value


def require_exact_keys(value: object, keys: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != keys:
        raise ImmutableArtifactError(f"{label} fields are not exact")
