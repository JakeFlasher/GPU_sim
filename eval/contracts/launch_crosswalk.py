"""Target-free, inventory-complete launch identity crosswalks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Sequence

from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    canonical_json_bytes,
    seal_model,
    sha256_bytes,
    verify_model_seal,
)

from ._common import (
    AuthorityBlockerCode,
    ContractOutcome,
    OutcomeStatus,
    aggregate_outcomes,
    authority_unavailable,
    evaluation_boundary_model,
    outcome_from_model,
    require_exact_evaluation_boundary,
)
from .model_seal import (
    RetainedPayloadSnapshot,
    require_exact_keys,
    require_nfc,
    require_sha256,
    seal_evaluator_model,
    strict_json_object,
    verify_evaluator_model,
    verify_retained_schema,
)


SCHEMA_VERSION = 1
ARTIFACT_KIND = "EVALUATOR_LAUNCH_CROSSWALK"
SCHEMA_ID = "https://gpu-simulator.invalid/eval/schemas/launch_crosswalk.v1.schema.json"
SCHEMA_SHA256 = "fb380c14ef978488b1255caaffc851ba5bd87456a19e6dd962bbd3b2545b25d5"
INVENTORY_SCHEMA_VERSION = 1
INVENTORY_ARTIFACT_KIND = "EVALUATOR_LAUNCH_INVENTORY"
INVENTORY_SCHEMA_ID = SCHEMA_ID
INVENTORY_SCHEMA_SHA256 = SCHEMA_SHA256
_MAX_SAFE_INTEGER = (1 << 53) - 1
_SAFE_ROLE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SAFE_KERNEL_NAME = re.compile(r"[A-Za-z0-9_.$@?<>-]{1,256}\Z")


class LaunchNamespace(str, Enum):
    CANONICAL = "CANONICAL"
    NCU = "NCU"
    NVBIT = "NVBIT"
    SIMULATOR = "SIMULATOR"


_SOURCE_KINDS = {
    LaunchNamespace.CANONICAL: "CANONICAL_LAUNCH_SOURCE",
    LaunchNamespace.NCU: "NCU_LAUNCH_SOURCE",
    LaunchNamespace.NVBIT: "NVBIT_LAUNCH_SOURCE",
    LaunchNamespace.SIMULATOR: "SIMULATOR_LAUNCH_SOURCE",
}
_SOURCE_NAMESPACES = (
    LaunchNamespace.NCU,
    LaunchNamespace.NVBIT,
    LaunchNamespace.SIMULATOR,
)
FIXTURE_CONTRACT_MODE = "FIXTURE_ONLY"
FIXTURE_SOURCE_SCHEMA_VERSION = 1
FIXTURE_SOURCE_ARTIFACT_KIND = "EVALUATOR_LAUNCH_SOURCE_FIXTURE"
FIXTURE_PARSER_STATUS = "FIXTURE_ONLY_NON_AUTHORITATIVE"
FIXTURE_BINDING_STATUS = "FIXTURE_ONLY_NON_AUTHORITATIVE"
# These registries are intentionally immutable and empty. Populating either one
# requires external parser/provenance or selector authority, not a code change.
PRODUCTION_LAUNCH_PARSER_AUTHORITIES = MappingProxyType({})
PRODUCTION_CROSS_NAMESPACE_PROVENANCE_AUTHORITIES = MappingProxyType({})
PRODUCTION_BINDING_AUTHORITIES = MappingProxyType({})


def _fixture_parser_authority_model(namespace: LaunchNamespace) -> dict[str, Any]:
    core = {
        "status": FIXTURE_PARSER_STATUS,
        "authority_id": f"FIXTURE_{namespace.value}_LAUNCH_PARSER_V1",
        "source_schema_id": (
            "https://gpu-simulator.invalid/eval/fixtures/"
            f"{namespace.value.lower()}_launch_source.v1.json"
        ),
    }
    return {
        **core,
        "authority_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


_FIXTURE_BINDING_AUTHORITY_CORE = {
    "status": FIXTURE_BINDING_STATUS,
    "authority_id": "FIXTURE_ORDINAL_ROLE_BINDING_V1",
    "mapping_policy": "EXACT_CANONICAL_TO_SOURCE_ROLE_DECLARATION",
}
FIXTURE_BINDING_AUTHORITY_SHA256 = sha256_bytes(
    canonical_json_bytes(_FIXTURE_BINDING_AUTHORITY_CORE)
)


@dataclass(frozen=True, slots=True)
class FixtureLaunchSpec:
    """One explicit non-production record in the closed fixture source format."""

    signature: "LaunchSignature"
    launch_ordinal: int
    launch_role: str

    def __post_init__(self) -> None:
        if not isinstance(self.signature, LaunchSignature):
            raise ImmutableArtifactError("fixture launch signature is invalid")
        _require_ordinal(self.launch_ordinal)
        _require_role(self.launch_role)

    def to_model(self) -> dict[str, Any]:
        return {
            "signature": self.signature.to_model(),
            "launch_ordinal": self.launch_ordinal,
            "launch_role": self.launch_role,
        }


@dataclass(frozen=True, slots=True)
class SourceArtifactDescriptor:
    namespace: LaunchNamespace
    artifact_kind: str
    artifact_sha256: str
    snapshot: RetainedPayloadSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, LaunchNamespace):
            raise ImmutableArtifactError("source namespace is invalid")
        if self.artifact_kind != _SOURCE_KINDS[self.namespace]:
            raise ImmutableArtifactError("source artifact kind does not match namespace")
        require_sha256(self.artifact_sha256, "source artifact SHA-256")
        if not isinstance(self.snapshot, RetainedPayloadSnapshot):
            raise ImmutableArtifactError("source snapshot is invalid")
        if self.snapshot.payload_sha256 != self.artifact_sha256:
            raise ImmutableArtifactError("source artifact hash does not match snapshot")

    @classmethod
    def from_fixture_payload(
        cls,
        *,
        namespace: LaunchNamespace,
        artifact_kind: str,
        payload: bytes,
        snapshot: RetainedPayloadSnapshot,
    ) -> "SourceArtifactDescriptor":
        """Create an explicitly non-production descriptor for fixture bytes."""

        if not isinstance(snapshot, RetainedPayloadSnapshot):
            raise ImmutableArtifactError("source snapshot is invalid")
        snapshot.require_fixture_acquisition("fixture launch source")
        snapshot.verify_payload(payload)
        return cls(namespace, artifact_kind, sha256_bytes(payload), snapshot)

    @classmethod
    def from_payload(
        cls,
        *,
        namespace: LaunchNamespace,
        artifact_kind: str,
        payload: bytes,
        snapshot: RetainedPayloadSnapshot,
    ) -> "SourceArtifactDescriptor":
        """Reject the retired authority-free production constructor."""

        del namespace, artifact_kind, payload, snapshot
        raise authority_unavailable(
            AuthorityBlockerCode.LAUNCH_PARSER_AUTHORITY_UNAVAILABLE,
            "source descriptors require an externally pinned namespace parser; "
            "use from_fixture_payload only for explicit fixtures",
        )

    def to_model(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace.value,
            "artifact_kind": self.artifact_kind,
            "artifact_sha256": self.artifact_sha256,
            "snapshot": self.snapshot.to_model(),
        }

    @classmethod
    def from_model(cls, value: object) -> "SourceArtifactDescriptor":
        require_exact_keys(
            value,
            {"namespace", "artifact_kind", "artifact_sha256", "snapshot"},
            "source artifact descriptor",
        )
        assert isinstance(value, dict)
        try:
            namespace = LaunchNamespace(value["namespace"])
        except (TypeError, ValueError) as exc:
            raise ImmutableArtifactError("source namespace is invalid") from exc
        return cls(
            namespace=namespace,
            artifact_kind=value["artifact_kind"],
            artifact_sha256=value["artifact_sha256"],
            snapshot=RetainedPayloadSnapshot.from_model(value["snapshot"]),
        )


@dataclass(frozen=True, slots=True)
class LaunchSignature:
    kernel_name: str
    grid_dim: tuple[int, int, int]
    block_dim: tuple[int, int, int]

    def __post_init__(self) -> None:
        require_nfc(self.kernel_name, "kernel_name")
        if _SAFE_KERNEL_NAME.fullmatch(self.kernel_name) is None:
            raise ImmutableArtifactError("kernel_name is outside the schema-safe alphabet")
        _require_dimension(self.grid_dim, "grid_dim")
        _require_dimension(self.block_dim, "block_dim")

    def to_model(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "grid_dim": list(self.grid_dim),
            "block_dim": list(self.block_dim),
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_model()))

    @classmethod
    def from_model(cls, value: object) -> "LaunchSignature":
        require_exact_keys(
            value, {"kernel_name", "grid_dim", "block_dim"}, "launch signature"
        )
        assert isinstance(value, dict)
        return cls(
            kernel_name=value["kernel_name"],
            grid_dim=_dimension_from_model(value["grid_dim"], "grid_dim"),
            block_dim=_dimension_from_model(value["block_dim"], "block_dim"),
        )


def fixture_launch_source_payload(
    namespace: LaunchNamespace, specs: Iterable[FixtureLaunchSpec]
) -> bytes:
    """Serialize the only supported fixture parser format canonically."""

    if not isinstance(namespace, LaunchNamespace):
        raise ImmutableArtifactError("fixture launch namespace is invalid")
    material = tuple(specs)
    if any(not isinstance(item, FixtureLaunchSpec) for item in material):
        raise ImmutableArtifactError("fixture launch source contains an invalid spec")
    model = {
        "fixture_schema_version": FIXTURE_SOURCE_SCHEMA_VERSION,
        "fixture_artifact_kind": FIXTURE_SOURCE_ARTIFACT_KIND,
        "namespace": namespace.value,
        "records": [item.to_model() for item in material],
    }
    return canonical_json_bytes(model)


def _fixture_specs_from_payload(
    source: SourceArtifactDescriptor, payload: bytes
) -> tuple[FixtureLaunchSpec, ...]:
    source.snapshot.require_fixture_acquisition("fixture launch source")
    source.snapshot.verify_payload(payload)
    model = strict_json_object(payload, label="fixture launch source")
    require_exact_keys(
        model,
        {
            "fixture_schema_version",
            "fixture_artifact_kind",
            "namespace",
            "records",
        },
        "fixture launch source",
    )
    if (
        type(model["fixture_schema_version"]) is not int
        or model["fixture_schema_version"] != FIXTURE_SOURCE_SCHEMA_VERSION
    ):
        raise ImmutableArtifactError("fixture launch source schema version is invalid")
    if model["fixture_artifact_kind"] != FIXTURE_SOURCE_ARTIFACT_KIND:
        raise ImmutableArtifactError("fixture launch source artifact kind is invalid")
    if model["namespace"] != source.namespace.value:
        raise ImmutableArtifactError("fixture launch source namespace mismatch")
    if canonical_json_bytes(model) != payload:
        raise ImmutableArtifactError("fixture launch source is not canonical JSON")
    raw_records = model["records"]
    if type(raw_records) is not list:
        raise ImmutableArtifactError("fixture launch records must be an array")
    specs: list[FixtureLaunchSpec] = []
    for value in raw_records:
        require_exact_keys(
            value,
            {"signature", "launch_ordinal", "launch_role"},
            "fixture launch record",
        )
        assert isinstance(value, dict)
        specs.append(
            FixtureLaunchSpec(
                signature=LaunchSignature.from_model(value["signature"]),
                launch_ordinal=value["launch_ordinal"],
                launch_role=value["launch_role"],
            )
        )
    records = tuple(
        LaunchRecord.create(
            source=source,
            signature=spec.signature,
            launch_ordinal=spec.launch_ordinal,
            launch_role=spec.launch_role,
        )
        for spec in specs
    )
    _canonical_candidates(records)
    return tuple(specs)


def _fixture_records_from_payload(
    source: SourceArtifactDescriptor, payload: bytes
) -> tuple["LaunchRecord", ...]:
    return _canonical_candidates(
        LaunchRecord.create(
            source=source,
            signature=spec.signature,
            launch_ordinal=spec.launch_ordinal,
            launch_role=spec.launch_role,
        )
        for spec in _fixture_specs_from_payload(source, payload)
    )


@dataclass(frozen=True, slots=True)
class LaunchIdentity:
    namespace: LaunchNamespace
    source_artifact_kind: str
    source_artifact_sha256: str
    source_snapshot_identity_sha256: str
    launch_ordinal: int
    launch_role: str
    qualified_launch_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, LaunchNamespace):
            raise ImmutableArtifactError("launch namespace is invalid")
        if self.source_artifact_kind != _SOURCE_KINDS[self.namespace]:
            raise ImmutableArtifactError("launch source kind does not match namespace")
        require_sha256(self.source_artifact_sha256, "source artifact SHA-256")
        require_sha256(
            self.source_snapshot_identity_sha256, "source snapshot identity"
        )
        require_sha256(self.qualified_launch_sha256, "qualified launch identity")
        _require_ordinal(self.launch_ordinal)
        _require_role(self.launch_role)

    @classmethod
    def create(
        cls,
        *,
        source: SourceArtifactDescriptor,
        signature: LaunchSignature,
        launch_ordinal: int,
        launch_role: str,
    ) -> "LaunchIdentity":
        _require_ordinal(launch_ordinal)
        _require_role(launch_role)
        digest = _qualified_launch_sha256(
            source, signature, launch_ordinal, launch_role
        )
        return cls(
            namespace=source.namespace,
            source_artifact_kind=source.artifact_kind,
            source_artifact_sha256=source.artifact_sha256,
            source_snapshot_identity_sha256=source.snapshot.snapshot_identity_sha256,
            launch_ordinal=launch_ordinal,
            launch_role=launch_role,
            qualified_launch_sha256=digest,
        )

    def to_model(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace.value,
            "source_artifact_kind": self.source_artifact_kind,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_snapshot_identity_sha256": self.source_snapshot_identity_sha256,
            "launch_ordinal": self.launch_ordinal,
            "launch_role": self.launch_role,
            "qualified_launch_sha256": self.qualified_launch_sha256,
        }

    @classmethod
    def from_model(cls, value: object) -> "LaunchIdentity":
        require_exact_keys(
            value,
            {
                "namespace",
                "source_artifact_kind",
                "source_artifact_sha256",
                "source_snapshot_identity_sha256",
                "launch_ordinal",
                "launch_role",
                "qualified_launch_sha256",
            },
            "launch identity",
        )
        assert isinstance(value, dict)
        try:
            namespace = LaunchNamespace(value["namespace"])
        except (TypeError, ValueError) as exc:
            raise ImmutableArtifactError("launch namespace is invalid") from exc
        _require_ordinal(value["launch_ordinal"])
        _require_role(value["launch_role"])
        require_sha256(value["source_artifact_sha256"], "source artifact SHA-256")
        require_sha256(
            value["source_snapshot_identity_sha256"], "source snapshot identity"
        )
        require_sha256(value["qualified_launch_sha256"], "qualified launch identity")
        return cls(
            namespace=namespace,
            source_artifact_kind=value["source_artifact_kind"],
            source_artifact_sha256=value["source_artifact_sha256"],
            source_snapshot_identity_sha256=value["source_snapshot_identity_sha256"],
            launch_ordinal=value["launch_ordinal"],
            launch_role=value["launch_role"],
            qualified_launch_sha256=value["qualified_launch_sha256"],
        )


@dataclass(frozen=True, slots=True)
class LaunchRecord:
    identity: LaunchIdentity
    signature: LaunchSignature

    def __post_init__(self) -> None:
        if not isinstance(self.identity, LaunchIdentity):
            raise ImmutableArtifactError("launch record identity is invalid")
        if not isinstance(self.signature, LaunchSignature):
            raise ImmutableArtifactError("launch record signature is invalid")
        expected = _qualified_launch_digest(
            self.identity.namespace,
            self.identity.source_artifact_kind,
            self.identity.source_artifact_sha256,
            self.identity.source_snapshot_identity_sha256,
            self.signature,
            self.identity.launch_ordinal,
            self.identity.launch_role,
        )
        if self.identity.qualified_launch_sha256 != expected:
            raise ImmutableArtifactError("qualified launch identity SHA-256 mismatch")

    @classmethod
    def create(
        cls,
        *,
        source: SourceArtifactDescriptor,
        signature: LaunchSignature,
        launch_ordinal: int,
        launch_role: str,
    ) -> "LaunchRecord":
        return cls(
            identity=LaunchIdentity.create(
                source=source,
                signature=signature,
                launch_ordinal=launch_ordinal,
                launch_role=launch_role,
            ),
            signature=signature,
        )

    def to_model(self) -> dict[str, Any]:
        return {"identity": self.identity.to_model(), "signature": self.signature.to_model()}

    @classmethod
    def from_model(cls, value: object) -> "LaunchRecord":
        require_exact_keys(value, {"identity", "signature"}, "launch record")
        assert isinstance(value, dict)
        return cls(
            identity=LaunchIdentity.from_model(value["identity"]),
            signature=LaunchSignature.from_model(value["signature"]),
        )


@dataclass(frozen=True, slots=True)
class IndependentLaunchBinding:
    namespace: LaunchNamespace
    inventory_model_seal_sha256: str
    source_artifact_kind: str
    source_artifact_sha256: str
    source_snapshot_identity_sha256: str
    launch_ordinal: int
    launch_role: str
    qualified_launch_sha256: str
    canonical_qualified_launch_sha256: str
    canonical_signature_sha256: str
    canonical_role: str
    source_role: str
    authority_status: str
    binding_authority_sha256: str
    model_seal_sha256: str

    def to_model(self) -> dict[str, Any]:
        return {
            "binding_version": 1,
            "binding_kind": "INDEPENDENT_INVENTORY_ORDINAL_ROLE",
            "namespace": self.namespace.value,
            "inventory_model_seal_sha256": self.inventory_model_seal_sha256,
            "source_artifact_kind": self.source_artifact_kind,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_snapshot_identity_sha256": self.source_snapshot_identity_sha256,
            "launch_ordinal": self.launch_ordinal,
            "launch_role": self.launch_role,
            "qualified_launch_sha256": self.qualified_launch_sha256,
            "canonical_qualified_launch_sha256": self.canonical_qualified_launch_sha256,
            "canonical_signature_sha256": self.canonical_signature_sha256,
            "role_mapping": {
                "mapping_kind": "EXACT_DECLARED_ROLE_MAPPING",
                "canonical_role": self.canonical_role,
                "source_role": self.source_role,
            },
            "authority_status": self.authority_status,
            "binding_authority_sha256": self.binding_authority_sha256,
            "model_seal_sha256": self.model_seal_sha256,
        }

    @classmethod
    def create(
        cls,
        *,
        inventory: dict[str, Any],
        canonical_launch: LaunchRecord,
        selected_launch: LaunchRecord,
        binding_authority_sha256: str,
        inventory_schema_bytes: bytes,
        inventory_schema_snapshot: RetainedPayloadSnapshot,
    ) -> "IndependentLaunchBinding":
        del (
            inventory,
            canonical_launch,
            selected_launch,
            binding_authority_sha256,
            inventory_schema_bytes,
            inventory_schema_snapshot,
        )
        raise authority_unavailable(
            AuthorityBlockerCode.BINDING_AUTHORITY_UNAVAILABLE,
            "the production ordinal/role binding authority registry is empty",
        )

    @classmethod
    def create_fixture(
        cls,
        *,
        inventory: dict[str, Any],
        canonical_launch: LaunchRecord,
        selected_launch: LaunchRecord,
        inventory_schema_bytes: bytes,
        inventory_schema_snapshot: RetainedPayloadSnapshot,
    ) -> "IndependentLaunchBinding":
        verify_launch_inventory_fixture_model(
            inventory,
            schema_bytes=inventory_schema_bytes,
            schema_snapshot=inventory_schema_snapshot,
        )
        namespace = LaunchNamespace(inventory["namespace"])
        if namespace not in _SOURCE_NAMESPACES or selected_launch.identity.namespace is not namespace:
            raise ImmutableArtifactError("binding source namespace is invalid")
        if selected_launch not in _inventory_records(inventory):
            raise ImmutableArtifactError("binding launch is absent from inventory")
        unsigned = {
            "binding_version": 1,
            "binding_kind": "INDEPENDENT_INVENTORY_ORDINAL_ROLE",
            "namespace": namespace.value,
            "inventory_model_seal_sha256": inventory["model_seal_sha256"],
            "source_artifact_kind": selected_launch.identity.source_artifact_kind,
            "source_artifact_sha256": selected_launch.identity.source_artifact_sha256,
            "source_snapshot_identity_sha256": (
                selected_launch.identity.source_snapshot_identity_sha256
            ),
            "launch_ordinal": selected_launch.identity.launch_ordinal,
            "launch_role": selected_launch.identity.launch_role,
            "qualified_launch_sha256": selected_launch.identity.qualified_launch_sha256,
            "canonical_qualified_launch_sha256": (
                canonical_launch.identity.qualified_launch_sha256
            ),
            "canonical_signature_sha256": canonical_launch.signature.sha256,
            "role_mapping": {
                "mapping_kind": "EXACT_DECLARED_ROLE_MAPPING",
                "canonical_role": canonical_launch.identity.launch_role,
                "source_role": selected_launch.identity.launch_role,
            },
            "authority_status": FIXTURE_BINDING_STATUS,
            "binding_authority_sha256": FIXTURE_BINDING_AUTHORITY_SHA256,
        }
        return cls.from_model(seal_model(unsigned))

    @classmethod
    def from_model(cls, value: object) -> "IndependentLaunchBinding":
        require_exact_keys(value, _BINDING_KEYS, "independent launch binding")
        assert isinstance(value, dict)
        if type(value["binding_version"]) is not int or value["binding_version"] != 1:
            raise ImmutableArtifactError("binding version is invalid")
        if value["binding_kind"] != "INDEPENDENT_INVENTORY_ORDINAL_ROLE":
            raise ImmutableArtifactError("binding kind is invalid")
        verify_model_seal(value)
        role_mapping = value["role_mapping"]
        require_exact_keys(
            role_mapping,
            {"mapping_kind", "canonical_role", "source_role"},
            "binding role mapping",
        )
        assert isinstance(role_mapping, dict)
        if role_mapping["mapping_kind"] != "EXACT_DECLARED_ROLE_MAPPING":
            raise ImmutableArtifactError("binding role mapping kind is invalid")
        try:
            namespace = LaunchNamespace(value["namespace"])
        except (TypeError, ValueError) as exc:
            raise ImmutableArtifactError("binding namespace is invalid") from exc
        if namespace not in _SOURCE_NAMESPACES:
            raise ImmutableArtifactError("binding namespace is not a source namespace")
        for field in (
            "inventory_model_seal_sha256",
            "source_artifact_sha256",
            "source_snapshot_identity_sha256",
            "qualified_launch_sha256",
            "canonical_qualified_launch_sha256",
            "canonical_signature_sha256",
            "binding_authority_sha256",
            "model_seal_sha256",
        ):
            require_sha256(value[field], field)
        _require_ordinal(value["launch_ordinal"])
        for role in (value["launch_role"], role_mapping["canonical_role"], role_mapping["source_role"]):
            _require_role(role)
        if value["source_artifact_kind"] != _SOURCE_KINDS[namespace]:
            raise ImmutableArtifactError("binding source kind does not match namespace")
        if value["authority_status"] != FIXTURE_BINDING_STATUS:
            raise ImmutableArtifactError("binding authority status is invalid")
        if value["binding_authority_sha256"] != FIXTURE_BINDING_AUTHORITY_SHA256:
            raise ImmutableArtifactError("binding authority SHA-256 is not pinned")
        return cls(
            namespace=namespace,
            inventory_model_seal_sha256=value["inventory_model_seal_sha256"],
            source_artifact_kind=value["source_artifact_kind"],
            source_artifact_sha256=value["source_artifact_sha256"],
            source_snapshot_identity_sha256=value["source_snapshot_identity_sha256"],
            launch_ordinal=value["launch_ordinal"],
            launch_role=value["launch_role"],
            qualified_launch_sha256=value["qualified_launch_sha256"],
            canonical_qualified_launch_sha256=value["canonical_qualified_launch_sha256"],
            canonical_signature_sha256=value["canonical_signature_sha256"],
            canonical_role=role_mapping["canonical_role"],
            source_role=role_mapping["source_role"],
            authority_status=value["authority_status"],
            binding_authority_sha256=value["binding_authority_sha256"],
            model_seal_sha256=value["model_seal_sha256"],
        )


_BINDING_KEYS = {
    "binding_version",
    "binding_kind",
    "namespace",
    "inventory_model_seal_sha256",
    "source_artifact_kind",
    "source_artifact_sha256",
    "source_snapshot_identity_sha256",
    "launch_ordinal",
    "launch_role",
    "qualified_launch_sha256",
    "canonical_qualified_launch_sha256",
    "canonical_signature_sha256",
    "role_mapping",
    "authority_status",
    "binding_authority_sha256",
    "model_seal_sha256",
}


def create_launch_inventory(
    *,
    source: SourceArtifactDescriptor,
    source_payload: bytes,
    candidates: Iterable[LaunchRecord],
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    del source_payload, candidates, schema_bytes, schema_snapshot
    if not isinstance(source, SourceArtifactDescriptor):
        raise ImmutableArtifactError("launch inventory source is invalid")
    source.snapshot.require_production_acquisition("fixture v2 launch source")
    raise authority_unavailable(
        AuthorityBlockerCode.LAUNCH_PARSER_AUTHORITY_UNAVAILABLE,
        "no namespace-specific production launch parser authority is installed",
    )


def create_launch_inventory_fixture(
    *,
    source: SourceArtifactDescriptor,
    source_payload: bytes,
    candidates: Iterable[LaunchRecord],
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    verify_retained_schema(
        schema_bytes,
        schema_snapshot,
        expected_sha256=INVENTORY_SCHEMA_SHA256,
        expected_id=INVENTORY_SCHEMA_ID,
    )
    source.snapshot.require_fixture_acquisition("fixture launch source")
    source.snapshot.verify_payload(source_payload)
    if sha256_bytes(source_payload) != source.artifact_sha256:
        raise ImmutableArtifactError("inventory source payload identity mismatch")
    supplied = _canonical_candidates(candidates)
    derived = _fixture_records_from_payload(source, source_payload)
    if supplied != derived:
        raise ImmutableArtifactError(
            "fixture candidate set differs from parser-derived complete inventory"
        )
    source_model = strict_json_object(source_payload, label="fixture launch source")
    unsigned = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "artifact_kind": INVENTORY_ARTIFACT_KIND,
        "schema_sha256": INVENTORY_SCHEMA_SHA256,
        "contract_mode": FIXTURE_CONTRACT_MODE,
        "parser_authority": _fixture_parser_authority_model(source.namespace),
        "namespace": source.namespace.value,
        "source": source.to_model(),
        "fixture_source_model": source_model,
        "candidate_count": len(derived),
        "qualified_candidate_ids": [
            record.identity.qualified_launch_sha256 for record in derived
        ],
        "candidate_records": [record.to_model() for record in derived],
    }
    return seal_evaluator_model(unsigned)


def verify_launch_inventory_model(
    model: dict[str, Any],
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> None:
    del model, schema_bytes, schema_snapshot
    raise authority_unavailable(
        AuthorityBlockerCode.LAUNCH_PARSER_AUTHORITY_UNAVAILABLE,
        "production inventory verification has no installed parser authority",
    )


def verify_launch_inventory_fixture_model(
    model: dict[str, Any],
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> None:
    verify_evaluator_model(
        model,
        expected_schema_version=INVENTORY_SCHEMA_VERSION,
        expected_artifact_kind=INVENTORY_ARTIFACT_KIND,
        expected_schema_sha256=INVENTORY_SCHEMA_SHA256,
        expected_schema_id=INVENTORY_SCHEMA_ID,
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
    )
    require_exact_keys(model, _INVENTORY_KEYS, "launch inventory")
    if model["contract_mode"] != FIXTURE_CONTRACT_MODE:
        raise ImmutableArtifactError("launch inventory is not fixture-only")
    source = SourceArtifactDescriptor.from_model(model["source"])
    source.snapshot.require_fixture_acquisition("fixture launch inventory source")
    if model["namespace"] != source.namespace.value:
        raise ImmutableArtifactError("inventory namespace does not match source")
    if model["parser_authority"] != _fixture_parser_authority_model(source.namespace):
        raise ImmutableArtifactError("fixture parser authority identity mismatch")
    fixture_source_model = model["fixture_source_model"]
    if type(fixture_source_model) is not dict:
        raise ImmutableArtifactError("fixture source model must be an object")
    fixture_source_payload = canonical_json_bytes(fixture_source_model)
    source.snapshot.verify_payload(fixture_source_payload)
    if sha256_bytes(fixture_source_payload) != source.artifact_sha256:
        raise ImmutableArtifactError("fixture source model identity mismatch")
    if type(model["candidate_records"]) is not list:
        raise ImmutableArtifactError("inventory candidate_records must be an array")
    records = tuple(LaunchRecord.from_model(item) for item in model["candidate_records"])
    canonical = _canonical_candidates(records)
    if records != canonical:
        raise ImmutableArtifactError("inventory candidates are not normalized and sorted")
    derived = _fixture_records_from_payload(source, fixture_source_payload)
    if records != derived:
        raise ImmutableArtifactError("inventory differs from parser-derived fixture records")
    _require_candidates_match_source(records, source)
    ids = [record.identity.qualified_launch_sha256 for record in records]
    candidate_count = _nonnegative_integer(
        model["candidate_count"], "inventory candidate_count"
    )
    if type(model["qualified_candidate_ids"]) is not list:
        raise ImmutableArtifactError("inventory qualified_candidate_ids must be an array")
    for candidate_id in model["qualified_candidate_ids"]:
        require_sha256(candidate_id, "inventory qualified candidate identity")
    if model["qualified_candidate_ids"] != ids or candidate_count != len(records):
        raise ImmutableArtifactError("inventory candidate closure mismatch")


_INVENTORY_KEYS = {
    "schema_version",
    "artifact_kind",
    "schema_sha256",
    "contract_mode",
    "parser_authority",
    "namespace",
    "source",
    "fixture_source_model",
    "candidate_count",
    "qualified_candidate_ids",
    "candidate_records",
    "model_seal_sha256",
}


@dataclass(frozen=True, slots=True)
class _NamespaceSelection:
    namespace: LaunchNamespace
    inventory: dict[str, Any]
    supplied_candidates: tuple[LaunchRecord, ...]
    name_dimension_match_count: int
    binding: IndependentLaunchBinding | None
    selected_launch: LaunchRecord | None
    outcome: ContractOutcome

    def to_model(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace.value,
            "inventory": self.inventory,
            "supplied_candidates": [item.to_model() for item in self.supplied_candidates],
            "name_dimension_match_count": self.name_dimension_match_count,
            "independent_binding": None if self.binding is None else self.binding.to_model(),
            "selected_launch": (
                None if self.selected_launch is None else self.selected_launch.to_model()
            ),
            "outcome": self.outcome.to_model(),
        }


def evaluate_launch_crosswalk(
    **_inputs: Any,
) -> dict[str, Any]:
    raise authority_unavailable(
        AuthorityBlockerCode.LAUNCH_PARSER_AUTHORITY_UNAVAILABLE,
        "fixture v2 crosswalk evaluation has no installed namespace parser authority",
    )


def evaluate_launch_crosswalk_fixture(
    *,
    canonical_inventory: dict[str, Any],
    ncu_inventory: dict[str, Any],
    nvbit_inventory: dict[str, Any],
    simulator_inventory: dict[str, Any],
    canonical_candidates: Sequence[LaunchRecord],
    ncu_candidates: Sequence[LaunchRecord],
    nvbit_candidates: Sequence[LaunchRecord],
    simulator_candidates: Sequence[LaunchRecord],
    ncu_binding: IndependentLaunchBinding | None,
    nvbit_binding: IndependentLaunchBinding | None,
    simulator_binding: IndependentLaunchBinding | None,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    inventory_schema_bytes: bytes,
    inventory_schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    """Create a sealed crosswalk only after complete inventory closure."""

    verify_retained_schema(
        schema_bytes,
        schema_snapshot,
        expected_sha256=SCHEMA_SHA256,
        expected_id=SCHEMA_ID,
    )
    inventory_models = {
        LaunchNamespace.CANONICAL: canonical_inventory,
        LaunchNamespace.NCU: ncu_inventory,
        LaunchNamespace.NVBIT: nvbit_inventory,
        LaunchNamespace.SIMULATOR: simulator_inventory,
    }
    supplied = {
        LaunchNamespace.CANONICAL: canonical_candidates,
        LaunchNamespace.NCU: ncu_candidates,
        LaunchNamespace.NVBIT: nvbit_candidates,
        LaunchNamespace.SIMULATOR: simulator_candidates,
    }
    for namespace, inventory in inventory_models.items():
        verify_launch_inventory_fixture_model(
            inventory,
            schema_bytes=inventory_schema_bytes,
            schema_snapshot=inventory_schema_snapshot,
        )
        if inventory["namespace"] != namespace.value:
            raise ImmutableArtifactError("inventory is installed under the wrong namespace")

    source_digests = [
        inventory_models[namespace]["source"]["artifact_sha256"]
        for namespace in LaunchNamespace
    ]
    source_snapshot_ids = [
        inventory_models[namespace]["source"]["snapshot"][
            "snapshot_identity_sha256"
        ]
        for namespace in LaunchNamespace
    ]
    if len(source_digests) != len(set(source_digests)) or len(
        source_snapshot_ids
    ) != len(set(source_snapshot_ids)):
        raise ImmutableArtifactError(
            "fixture launch sources alias across namespaces without provenance authority"
        )

    canonical_records = _inventory_records(canonical_inventory)
    if len(canonical_records) != 1:
        raise ImmutableArtifactError("canonical inventory must contain exactly one launch")
    canonical_launch = canonical_records[0]
    canonical_selection = _select_canonical(
        canonical_inventory, supplied[LaunchNamespace.CANONICAL]
    )
    bindings = {
        LaunchNamespace.NCU: ncu_binding,
        LaunchNamespace.NVBIT: nvbit_binding,
        LaunchNamespace.SIMULATOR: simulator_binding,
    }
    selections = [canonical_selection]
    for namespace in _SOURCE_NAMESPACES:
        selections.append(
            _select_source(
                namespace,
                inventory_models[namespace],
                supplied[namespace],
                canonical_launch,
                bindings[namespace],
            )
        )
    aggregate = aggregate_outcomes(item.outcome for item in selections)
    return seal_evaluator_model(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "schema_sha256": SCHEMA_SHA256,
            "contract_mode": FIXTURE_CONTRACT_MODE,
            "evaluation_boundary": evaluation_boundary_model(),
            "selection_signature": canonical_launch.signature.to_model(),
            "namespaces": {
                item.namespace.value.lower(): item.to_model() for item in selections
            },
            "outcome": aggregate.to_model(),
        }
    )


def verify_launch_crosswalk_model(
    model: dict[str, Any],
    **_inputs: Any,
) -> None:
    del model
    raise authority_unavailable(
        AuthorityBlockerCode.LAUNCH_PARSER_AUTHORITY_UNAVAILABLE,
        "fixture v2 crosswalk verification has no installed namespace parser authority",
    )


def verify_launch_crosswalk_fixture_model(
    model: dict[str, Any],
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    inventory_schema_bytes: bytes,
    inventory_schema_snapshot: RetainedPayloadSnapshot,
) -> None:
    verify_evaluator_model(
        model,
        expected_schema_version=SCHEMA_VERSION,
        expected_artifact_kind=ARTIFACT_KIND,
        expected_schema_sha256=SCHEMA_SHA256,
        expected_schema_id=SCHEMA_ID,
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
    )
    require_exact_keys(
        model,
        {
            "schema_version",
            "artifact_kind",
            "schema_sha256",
            "contract_mode",
            "evaluation_boundary",
            "selection_signature",
            "namespaces",
            "outcome",
            "model_seal_sha256",
        },
        "launch crosswalk",
    )
    if model["contract_mode"] != FIXTURE_CONTRACT_MODE:
        raise ImmutableArtifactError("launch crosswalk is not fixture-only")
    require_exact_evaluation_boundary(model["evaluation_boundary"])
    signature = LaunchSignature.from_model(model["selection_signature"])
    require_exact_keys(
        model["namespaces"],
        {"canonical", "ncu", "nvbit", "simulator"},
        "launch namespace map",
    )
    assert isinstance(model["namespaces"], dict)
    parsed = {
        namespace: _selection_from_model(
            model["namespaces"][namespace.value.lower()],
            namespace,
            inventory_schema_bytes,
            inventory_schema_snapshot,
        )
        for namespace in LaunchNamespace
    }
    source_digests = [
        parsed[namespace].inventory["source"]["artifact_sha256"]
        for namespace in LaunchNamespace
    ]
    source_snapshot_ids = [
        parsed[namespace].inventory["source"]["snapshot"][
            "snapshot_identity_sha256"
        ]
        for namespace in LaunchNamespace
    ]
    if len(source_digests) != len(set(source_digests)) or len(
        source_snapshot_ids
    ) != len(set(source_snapshot_ids)):
        raise ImmutableArtifactError(
            "fixture launch sources alias across namespaces without provenance authority"
        )
    canonical_records = _inventory_records(parsed[LaunchNamespace.CANONICAL].inventory)
    if len(canonical_records) != 1 or canonical_records[0].signature != signature:
        raise ImmutableArtifactError("canonical inventory/signature closure mismatch")
    canonical_launch = canonical_records[0]
    expected_canonical = _select_canonical(
        parsed[LaunchNamespace.CANONICAL].inventory,
        parsed[LaunchNamespace.CANONICAL].supplied_candidates,
    )
    if parsed[LaunchNamespace.CANONICAL] != expected_canonical:
        raise ImmutableArtifactError("canonical selection result mismatch")
    for namespace in _SOURCE_NAMESPACES:
        observed = parsed[namespace]
        expected = _select_source(
            namespace,
            observed.inventory,
            observed.supplied_candidates,
            canonical_launch,
            observed.binding,
        )
        if observed != expected:
            raise ImmutableArtifactError(f"{namespace.value} selection result mismatch")
    aggregate = aggregate_outcomes(item.outcome for item in parsed.values())
    root = outcome_from_model(
        model["outcome"],
        rejected_reasons=_ROOT_REJECTED_REASONS,
        unavailable_reasons=_ROOT_UNAVAILABLE_REASONS,
    )
    if root != aggregate:
        raise ImmutableArtifactError("launch crosswalk aggregate outcome mismatch")


def load_launch_crosswalk(
    payload: bytes,
    **_inputs: Any,
) -> dict[str, Any]:
    del payload
    raise authority_unavailable(
        AuthorityBlockerCode.LAUNCH_PARSER_AUTHORITY_UNAVAILABLE,
        "fixture v2 crosswalk loading has no installed namespace parser authority",
    )


def load_launch_crosswalk_fixture(
    payload: bytes,
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    inventory_schema_bytes: bytes,
    inventory_schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    model = strict_json_object(payload, label="launch crosswalk")
    verify_launch_crosswalk_fixture_model(
        model,
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
        inventory_schema_bytes=inventory_schema_bytes,
        inventory_schema_snapshot=inventory_schema_snapshot,
    )
    return model


def _select_canonical(
    inventory: dict[str, Any], supplied: Iterable[LaunchRecord]
) -> _NamespaceSelection:
    supplied_records = _canonical_candidates(supplied)
    inventory_records = _inventory_records(inventory)
    if supplied_records != inventory_records:
        outcome = ContractOutcome(
            OutcomeStatus.REJECTED,
            ("CANONICAL_INVENTORY_CANDIDATE_SET_MISMATCH",),
        )
        selected = None
    else:
        outcome = ContractOutcome(
            OutcomeStatus.UNAVAILABLE,
            ("FIXTURE_ONLY_CANONICAL_SELECTION",),
        )
        selected = inventory_records[0]
    return _NamespaceSelection(
        LaunchNamespace.CANONICAL,
        inventory,
        supplied_records,
        1,
        None,
        selected,
        outcome,
    )


def _select_source(
    namespace: LaunchNamespace,
    inventory: dict[str, Any],
    supplied: Iterable[LaunchRecord],
    canonical_launch: LaunchRecord,
    binding: IndependentLaunchBinding | None,
) -> _NamespaceSelection:
    supplied_records = _canonical_candidates(supplied)
    inventory_records = _inventory_records(inventory)
    prefix = namespace.value
    if binding is not None:
        if not isinstance(binding, IndependentLaunchBinding):
            raise ImmutableArtifactError("launch binding type is invalid")
        binding = IndependentLaunchBinding.from_model(binding.to_model())
    if supplied_records != inventory_records:
        return _NamespaceSelection(
            namespace,
            inventory,
            supplied_records,
            0,
            binding,
            None,
            ContractOutcome(
                OutcomeStatus.REJECTED,
                (f"{prefix}_INVENTORY_CANDIDATE_SET_MISMATCH",),
            ),
        )
    matches = tuple(
        item for item in inventory_records if item.signature == canonical_launch.signature
    )
    if not matches:
        return _NamespaceSelection(
            namespace,
            inventory,
            supplied_records,
            0,
            binding,
            None,
            ContractOutcome(
                OutcomeStatus.UNAVAILABLE, (f"{prefix}_SIGNATURE_UNAVAILABLE",)
            ),
        )
    if binding is None:
        return _NamespaceSelection(
            namespace,
            inventory,
            supplied_records,
            len(matches),
            None,
            None,
            ContractOutcome(
                OutcomeStatus.UNAVAILABLE,
                (f"{prefix}_BINDING_AUTHORITY_UNAVAILABLE",),
            ),
        )
    ordinal_role = tuple(
        item
        for item in matches
        if item.identity.launch_ordinal == binding.launch_ordinal
        and item.identity.launch_role == binding.launch_role
    )
    if not ordinal_role:
        reason = f"{prefix}_BINDING_NO_MATCH"
    elif len(ordinal_role) != 1:
        reason = f"{prefix}_BINDING_NOT_UNIQUE"
    elif not _binding_matches_context(
        binding, inventory, canonical_launch, ordinal_role[0]
    ):
        reason = f"{prefix}_BINDING_CONTEXT_MISMATCH"
    else:
        return _NamespaceSelection(
            namespace,
            inventory,
            supplied_records,
            len(matches),
            binding,
            ordinal_role[0],
            ContractOutcome(
                OutcomeStatus.UNAVAILABLE,
                (f"FIXTURE_ONLY_{prefix}_BINDING",),
            ),
        )
    return _NamespaceSelection(
        namespace,
        inventory,
        supplied_records,
        len(matches),
        binding,
        None,
        ContractOutcome(OutcomeStatus.REJECTED, (reason,)),
    )


def _binding_failure_selection(
    namespace: LaunchNamespace,
    inventory: dict[str, Any],
    supplied: tuple[LaunchRecord, ...],
    match_count: int,
    binding: IndependentLaunchBinding,
) -> _NamespaceSelection:
    return _NamespaceSelection(
        namespace,
        inventory,
        supplied,
        match_count,
        binding,
        None,
        ContractOutcome(
            OutcomeStatus.REJECTED,
            (f"{namespace.value}_BINDING_CONTEXT_MISMATCH",),
        ),
    )


def _binding_matches_context(
    binding: IndependentLaunchBinding,
    inventory: dict[str, Any],
    canonical_launch: LaunchRecord,
    selected: LaunchRecord,
) -> bool:
    identity = selected.identity
    return (
        binding.authority_status == FIXTURE_BINDING_STATUS
        and binding.binding_authority_sha256 == FIXTURE_BINDING_AUTHORITY_SHA256
        and binding.namespace is identity.namespace
        and binding.inventory_model_seal_sha256 == inventory["model_seal_sha256"]
        and binding.source_artifact_kind == identity.source_artifact_kind
        and binding.source_artifact_sha256 == identity.source_artifact_sha256
        and binding.source_snapshot_identity_sha256
        == identity.source_snapshot_identity_sha256
        and binding.launch_ordinal == identity.launch_ordinal
        and binding.launch_role == identity.launch_role
        and binding.qualified_launch_sha256 == identity.qualified_launch_sha256
        and binding.canonical_qualified_launch_sha256
        == canonical_launch.identity.qualified_launch_sha256
        and binding.canonical_signature_sha256 == canonical_launch.signature.sha256
        and binding.canonical_role == canonical_launch.identity.launch_role
        and binding.source_role == identity.launch_role
    )


def _selection_from_model(
    value: object,
    namespace: LaunchNamespace,
    inventory_schema_bytes: bytes,
    inventory_schema_snapshot: RetainedPayloadSnapshot,
) -> _NamespaceSelection:
    require_exact_keys(
        value,
        {
            "namespace",
            "inventory",
            "supplied_candidates",
            "name_dimension_match_count",
            "independent_binding",
            "selected_launch",
            "outcome",
        },
        "namespace selection",
    )
    assert isinstance(value, dict)
    if value["namespace"] != namespace.value:
        raise ImmutableArtifactError("namespace selection identity mismatch")
    inventory = value["inventory"]
    verify_launch_inventory_fixture_model(
        inventory,
        schema_bytes=inventory_schema_bytes,
        schema_snapshot=inventory_schema_snapshot,
    )
    if inventory["namespace"] != namespace.value:
        raise ImmutableArtifactError("selection inventory namespace mismatch")
    raw_candidates = value["supplied_candidates"]
    if type(raw_candidates) is not list:
        raise ImmutableArtifactError("supplied_candidates must be an array")
    supplied = tuple(LaunchRecord.from_model(item) for item in raw_candidates)
    if supplied != _canonical_candidates(supplied):
        raise ImmutableArtifactError("supplied candidates are not normalized and sorted")
    binding = (
        None
        if value["independent_binding"] is None
        else IndependentLaunchBinding.from_model(value["independent_binding"])
    )
    selected = (
        None
        if value["selected_launch"] is None
        else LaunchRecord.from_model(value["selected_launch"])
    )
    rejected, unavailable = _selection_reason_sets(namespace)
    return _NamespaceSelection(
        namespace=namespace,
        inventory=inventory,
        supplied_candidates=supplied,
        name_dimension_match_count=_nonnegative_integer(
            value["name_dimension_match_count"], "name_dimension_match_count"
        ),
        binding=binding,
        selected_launch=selected,
        outcome=outcome_from_model(
            value["outcome"],
            rejected_reasons=rejected,
            unavailable_reasons=unavailable,
        ),
    )


def _selection_reason_sets(
    namespace: LaunchNamespace,
) -> tuple[frozenset[str], frozenset[str]]:
    prefix = namespace.value
    if namespace is LaunchNamespace.CANONICAL:
        return (
            frozenset({"CANONICAL_INVENTORY_CANDIDATE_SET_MISMATCH"}),
            frozenset({"FIXTURE_ONLY_CANONICAL_SELECTION"}),
        )
    return (
        frozenset(
            {
                f"{prefix}_INVENTORY_CANDIDATE_SET_MISMATCH",
                f"{prefix}_BINDING_NO_MATCH",
                f"{prefix}_BINDING_NOT_UNIQUE",
                f"{prefix}_BINDING_CONTEXT_MISMATCH",
            }
        ),
        frozenset(
            {
                f"{prefix}_SIGNATURE_UNAVAILABLE",
                f"{prefix}_BINDING_AUTHORITY_UNAVAILABLE",
                f"FIXTURE_ONLY_{prefix}_BINDING",
            }
        ),
    )


_ROOT_REJECTED_REASONS = frozenset().union(
    *(_selection_reason_sets(namespace)[0] for namespace in LaunchNamespace)
)
_ROOT_UNAVAILABLE_REASONS = frozenset().union(
    *(_selection_reason_sets(namespace)[1] for namespace in LaunchNamespace)
)


def _inventory_records(model: dict[str, Any]) -> tuple[LaunchRecord, ...]:
    return tuple(LaunchRecord.from_model(item) for item in model["candidate_records"])


def _canonical_candidates(candidates: Iterable[LaunchRecord]) -> tuple[LaunchRecord, ...]:
    material = tuple(candidates)
    if any(not isinstance(item, LaunchRecord) for item in material):
        raise ImmutableArtifactError("candidate collection contains an invalid record")
    ordered = tuple(
        sorted(material, key=lambda item: item.identity.qualified_launch_sha256)
    )
    ids = tuple(item.identity.qualified_launch_sha256 for item in ordered)
    if len(ids) != len(set(ids)):
        raise ImmutableArtifactError("normalized qualified candidate identity collision")
    normalized_keys = tuple(
        (
            item.identity.namespace.value,
            item.identity.source_artifact_kind,
            item.identity.source_artifact_sha256,
            item.identity.launch_ordinal,
            item.identity.launch_role,
            item.signature.kernel_name,
            item.signature.grid_dim,
            item.signature.block_dim,
        )
        for item in ordered
    )
    if len(normalized_keys) != len(set(normalized_keys)):
        raise ImmutableArtifactError("normalized candidate semantic identity collision")
    return ordered


def _require_candidates_match_source(
    candidates: Iterable[LaunchRecord], source: SourceArtifactDescriptor
) -> None:
    for item in candidates:
        identity = item.identity
        if (
            identity.namespace is not source.namespace
            or identity.source_artifact_kind != source.artifact_kind
            or identity.source_artifact_sha256 != source.artifact_sha256
            or identity.source_snapshot_identity_sha256
            != source.snapshot.snapshot_identity_sha256
        ):
            raise ImmutableArtifactError("candidate does not belong to inventory source")


def _qualified_launch_sha256(
    source: SourceArtifactDescriptor,
    signature: LaunchSignature,
    launch_ordinal: int,
    launch_role: str,
) -> str:
    return _qualified_launch_digest(
        source.namespace,
        source.artifact_kind,
        source.artifact_sha256,
        source.snapshot.snapshot_identity_sha256,
        signature,
        launch_ordinal,
        launch_role,
    )


def _qualified_launch_digest(
    namespace: LaunchNamespace,
    source_artifact_kind: str,
    source_artifact_sha256: str,
    source_snapshot_identity_sha256: str,
    signature: LaunchSignature,
    launch_ordinal: int,
    launch_role: str,
) -> str:
    _require_ordinal(launch_ordinal)
    _require_role(launch_role)
    preimage = {
        "namespace": namespace.value,
        "source_artifact_kind": source_artifact_kind,
        "source_artifact_sha256": source_artifact_sha256,
        "source_snapshot_identity_sha256": source_snapshot_identity_sha256,
        "launch_ordinal": launch_ordinal,
        "launch_role": launch_role,
        "signature": signature.to_model(),
    }
    return sha256_bytes(canonical_json_bytes(preimage))


def _require_dimension(value: object, label: str) -> None:
    if type(value) is not tuple or len(value) != 3:
        raise ImmutableArtifactError(f"{label} must be an exact three-integer tuple")
    if any(
        type(component) is not int or not 1 <= component <= _MAX_SAFE_INTEGER
        for component in value
    ):
        raise ImmutableArtifactError(f"{label} components must be positive safe integers")


def _dimension_from_model(value: object, label: str) -> tuple[int, int, int]:
    if type(value) is not list or len(value) != 3:
        raise ImmutableArtifactError(f"{label} must contain exactly three integers")
    material = tuple(value)
    _require_dimension(material, label)
    return material  # type: ignore[return-value]


def _require_ordinal(value: object) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_SAFE_INTEGER:
        raise ImmutableArtifactError("launch_ordinal must be a nonnegative safe integer")


def _require_role(value: object) -> None:
    require_nfc(value, "launch_role")
    if type(value) is not str or _SAFE_ROLE.fullmatch(value) is None:
        raise ImmutableArtifactError("launch_role is invalid")


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_SAFE_INTEGER:
        raise ImmutableArtifactError(f"{label} must be a nonnegative safe integer")
    return value


# Additive production authority and crosswalk v2. The fixture-only v1 constants,
# wire models, and entry points above intentionally remain byte-for-byte stable.
WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_VERSION = 1
WORKLOAD_EXECUTION_AUTHORITY_ARTIFACT_KIND = (
    "EVALUATOR_WORKLOAD_EXECUTION_AUTHORITY"
)
WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_ID = (
    "https://gpu-simulator.invalid/eval/schemas/"
    "workload_execution_authority.v1.schema.json"
)
WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_SHA256 = (
    "5d8d9f752b94ac879e26684bd09d0149e10ff62cc2b29b07e40bc938767a696f"
)
CROSSWALK_V2_SCHEMA_VERSION = 2
CROSSWALK_V2_ARTIFACT_KIND = ARTIFACT_KIND
CROSSWALK_V2_SCHEMA_ID = (
    "https://gpu-simulator.invalid/eval/schemas/launch_crosswalk.v2.schema.json"
)
CROSSWALK_V2_SCHEMA_SHA256 = (
    "08d4bcfb54b27bc84e88aa7ab6b62de469aa335a825beeda162789ac64bad5a3"
)
PRODUCTION_SOURCE_ARTIFACT_KIND = "EVALUATOR_LAUNCH_SOURCE_V2"
PRODUCTION_INVENTORY_ARTIFACT_KIND = "EVALUATOR_LAUNCH_INVENTORY_V2"
PRODUCTION_CONTRACT_MODE = "TARGET_FREE_PRODUCTION"
V2_FIXTURE_CONTRACT_MODE = "FIXTURE_ONLY_V2_NON_AUTHORITATIVE"
V2_FIXTURE_CROSSWALK_REASON = "FIXTURE_ONLY_V2_CROSSWALK"
EXPLICIT_ORDINAL_BINDING = "EXPLICIT_WORKLOAD_EXECUTION_AUTHORITY_V1"
NATIVE_INDEX_NOT_ORDINAL = "NATIVE_INDEX_ONLY_NOT_PROCESS_ORDINAL"
NATIVE_INDEX_NOT_APPLICABLE = "NOT_APPLICABLE"
_WORKLOAD_ID_RULE = "SHA256_CANONICAL_WORKLOAD_V1"
_MATCHED_EXECUTION_ID_RULE = "SHA256_CANONICAL_MATCHED_EXECUTION_V1"
_WORKLOAD_ID = re.compile(r"workload-v1-[0-9a-f]{64}\Z")
_MATCHED_EXECUTION_ID = re.compile(r"matched-execution-v1-[0-9a-f]{64}\Z")
_CONTEXT_ID = re.compile(
    r"(?:canonical|ncu|nvbit|simulator)-context-v1-[0-9a-f]{64}\Z"
)
_FORBIDDEN_TARGET_FIELDS = frozenset(
    {
        "label",
        "labels",
        "metric",
        "metrics",
        "metric_name",
        "metric_names",
        "metric_value",
        "metric_values",
        "ncu_metric",
        "ncu_metrics",
        "observed_value",
        "predicted_value",
        "prediction",
        "predictions",
        "score",
        "scores",
        "target_value",
        "target_values",
        "unit_tagged_metrics",
    }
)
# No current M1 artifact is externally admitted, no pre-label trust anchor is
# installed, and no namespace adapter implementation is independently pinned.
# Every public production-v2 path checks these immutable registries before
# reading caller data. Structural fixture algebra uses separately named APIs.
CURRENT_M1_WORKLOAD_EXECUTION_AUTHORITIES = MappingProxyType({})
PRODUCTION_WORKLOAD_EXECUTION_ADMISSION_AUTHORITIES = MappingProxyType({})
PRODUCTION_PRELABEL_TRUST_ANCHORS = MappingProxyType({})
PRODUCTION_NAMESPACE_ADAPTER_AUTHORITIES = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ProductionParserAuthority:
    """One independently pinned identity-only namespace parser."""

    namespace: LaunchNamespace
    authority_id: str
    authority_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, LaunchNamespace):
            raise ImmutableArtifactError("parser authority namespace is invalid")
        _require_role(self.authority_id)
        require_sha256(self.authority_sha256, "parser authority SHA-256")

    def to_model(self) -> dict[str, Any]:
        return {
            "authority_id": self.authority_id,
            "authority_sha256": self.authority_sha256,
        }


@dataclass(frozen=True, slots=True)
class V2FixtureRetainedInput:
    """Exact retained bytes used only by non-authoritative v2 algebra tests."""

    payload: bytes
    snapshot: RetainedPayloadSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, RetainedPayloadSnapshot):
            raise ImmutableArtifactError("fixture retained input snapshot is invalid")
        self.snapshot.require_fixture_acquisition("fixture v2 retained input")
        self.snapshot.verify_payload(self.payload)

    def to_model(self) -> dict[str, Any]:
        return {
            "payload_sha256": self.snapshot.payload_sha256,
            "payload_snapshot": self.snapshot.to_model(),
        }


@dataclass(frozen=True, slots=True)
class V2FixtureAdapterEvidence:
    """Underlying evidence and parser implementation bound to one adapter."""

    underlying_evidence: V2FixtureRetainedInput
    parser_implementation: V2FixtureRetainedInput

    def __post_init__(self) -> None:
        if not isinstance(self.underlying_evidence, V2FixtureRetainedInput):
            raise ImmutableArtifactError("underlying fixture evidence is invalid")
        if not isinstance(self.parser_implementation, V2FixtureRetainedInput):
            raise ImmutableArtifactError("fixture parser implementation is invalid")

    def to_model(self) -> dict[str, Any]:
        return {
            "underlying_evidence": self.underlying_evidence.to_model(),
            "parser_implementation": self.parser_implementation.to_model(),
        }


@dataclass(frozen=True, slots=True)
class V2FixtureSourceInput:
    """Source projection and exact adapter inputs used to derive it."""

    source_projection: V2FixtureRetainedInput
    adapter_evidence: V2FixtureAdapterEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.source_projection, V2FixtureRetainedInput):
            raise ImmutableArtifactError("fixture source projection is invalid")
        if not isinstance(self.adapter_evidence, V2FixtureAdapterEvidence):
            raise ImmutableArtifactError("fixture adapter evidence is invalid")


@dataclass(frozen=True, slots=True)
class ExpectedProductionLaunch:
    """One pre-label launch-roster row, before cohort IDs are derived."""

    process_launch_ordinal: int
    kernel_mangled_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    device_id: int
    stream_id: int
    binary_architecture: str

    def __post_init__(self) -> None:
        _require_ordinal(self.process_launch_ordinal)
        _require_text(self.kernel_mangled_name, "kernel_mangled_name")
        _require_uint32_dimension(self.grid, "grid")
        _require_uint32_dimension(self.block, "block")
        _require_uint32(self.device_id, "device_id")
        _require_ordinal_value(self.stream_id, "stream_id")
        _require_role(self.binary_architecture)

    def to_identity_model(self) -> dict[str, Any]:
        return {
            "process_launch_ordinal": self.process_launch_ordinal,
            "kernel_mangled_name": self.kernel_mangled_name,
            "grid": _dim3_model(self.grid),
            "block": _dim3_model(self.block),
            "device_id": self.device_id,
            "stream_id": self.stream_id,
            "binary_architecture": self.binary_architecture,
        }


@dataclass(frozen=True, slots=True)
class CompleteLaunchKey:
    """The exact nine-field key used by fixture v2 crosswalk v2."""

    workload_id: str
    matched_execution_id: str
    process_launch_ordinal: int
    kernel_mangled_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    device_id: int
    stream_id: int
    binary_architecture: str

    def __post_init__(self) -> None:
        if type(self.workload_id) is not str or _WORKLOAD_ID.fullmatch(
            self.workload_id
        ) is None:
            raise ImmutableArtifactError("workload_id is invalid")
        if type(self.matched_execution_id) is not str or _MATCHED_EXECUTION_ID.fullmatch(
            self.matched_execution_id
        ) is None:
            raise ImmutableArtifactError("matched_execution_id is invalid")
        ExpectedProductionLaunch(
            process_launch_ordinal=self.process_launch_ordinal,
            kernel_mangled_name=self.kernel_mangled_name,
            grid=self.grid,
            block=self.block,
            device_id=self.device_id,
            stream_id=self.stream_id,
            binary_architecture=self.binary_architecture,
        )

    def to_model(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "matched_execution_id": self.matched_execution_id,
            "process_launch_ordinal": self.process_launch_ordinal,
            "kernel_mangled_name": self.kernel_mangled_name,
            "grid": _dim3_model(self.grid),
            "block": _dim3_model(self.block),
            "device_id": self.device_id,
            "stream_id": self.stream_id,
            "binary_architecture": self.binary_architecture,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_model()))

    @classmethod
    def from_model(cls, value: object) -> "CompleteLaunchKey":
        require_exact_keys(
            value,
            {
                "workload_id",
                "matched_execution_id",
                "process_launch_ordinal",
                "kernel_mangled_name",
                "grid",
                "block",
                "device_id",
                "stream_id",
                "binary_architecture",
            },
            "complete launch key",
        )
        assert isinstance(value, dict)
        return cls(
            workload_id=value["workload_id"],
            matched_execution_id=value["matched_execution_id"],
            process_launch_ordinal=value["process_launch_ordinal"],
            kernel_mangled_name=value["kernel_mangled_name"],
            grid=_dim3_from_model(value["grid"], "grid"),
            block=_dim3_from_model(value["block"], "block"),
            device_id=value["device_id"],
            stream_id=value["stream_id"],
            binary_architecture=value["binary_architecture"],
        )


@dataclass(frozen=True, slots=True)
class ProductionLaunchObservation:
    """One identity-only source row with an explicit pre-label ordinal binding."""

    launch_key: CompleteLaunchKey
    kernel_family: str
    native_context_id: str
    native_launch_id: str
    native_stream_id: int
    native_launch_index: int | None
    native_launch_index_role: str
    authority_model_seal_sha256: str
    binding_sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.launch_key, CompleteLaunchKey):
            raise ImmutableArtifactError("production observation launch key is invalid")
        _require_role(self.kernel_family)
        _require_context_id(self.native_context_id)
        _require_role(self.native_launch_id)
        _require_ordinal_value(self.native_stream_id, "native_stream_id")
        if self.native_stream_id != self.launch_key.stream_id:
            raise ImmutableArtifactError(
                "native stream does not equal the complete launch-key stream"
            )
        if self.native_launch_index is None:
            if self.native_launch_index_role != NATIVE_INDEX_NOT_APPLICABLE:
                raise ImmutableArtifactError("absent native launch index has an invalid role")
        else:
            _require_ordinal_value(self.native_launch_index, "native_launch_index")
            if self.native_launch_index_role != NATIVE_INDEX_NOT_ORDINAL:
                raise ImmutableArtifactError(
                    "native launch index may not be treated as process ordinal"
                )
        require_sha256(
            self.authority_model_seal_sha256,
            "ordinal-binding authority model seal",
        )
        require_sha256(self.binding_sha256, "ordinal binding SHA-256")
        require_sha256(self.record_sha256, "source record SHA-256")
        if self.binding_sha256 != _observation_binding_sha256(self):
            raise ImmutableArtifactError("ordinal binding SHA-256 mismatch")
        if self.record_sha256 != _observation_record_sha256(self):
            raise ImmutableArtifactError("source record SHA-256 mismatch")

    @classmethod
    def create(
        cls,
        *,
        launch_key: CompleteLaunchKey,
        kernel_family: str,
        native_context_id: str,
        native_launch_id: str,
        native_stream_id: int,
        authority_model_seal_sha256: str,
        native_launch_index: int | None = None,
    ) -> "ProductionLaunchObservation":
        role = (
            NATIVE_INDEX_NOT_APPLICABLE
            if native_launch_index is None
            else NATIVE_INDEX_NOT_ORDINAL
        )
        values = {
            "launch_key": launch_key,
            "kernel_family": kernel_family,
            "native_context_id": native_context_id,
            "native_launch_id": native_launch_id,
            "native_stream_id": native_stream_id,
            "native_launch_index": native_launch_index,
            "native_launch_index_role": role,
            "authority_model_seal_sha256": authority_model_seal_sha256,
        }
        binding_sha256 = _observation_binding_from_values(values)
        record_sha256 = _observation_record_from_values(values, binding_sha256)
        return cls(**values, binding_sha256=binding_sha256, record_sha256=record_sha256)

    def to_model(self) -> dict[str, Any]:
        return {
            "launch_key": self.launch_key.to_model(),
            "kernel_family": self.kernel_family,
            "native_identity": {
                "execution_context_id": self.native_context_id,
                "native_launch_id": self.native_launch_id,
                "native_stream_id": self.native_stream_id,
                "native_launch_index": self.native_launch_index,
            },
            "process_ordinal_binding": {
                "binding_kind": EXPLICIT_ORDINAL_BINDING,
                "authority_model_seal_sha256": self.authority_model_seal_sha256,
                "process_launch_ordinal": self.launch_key.process_launch_ordinal,
                "native_launch_index_role": self.native_launch_index_role,
                "binding_sha256": self.binding_sha256,
            },
            "record_sha256": self.record_sha256,
        }

    @classmethod
    def from_model(cls, value: object) -> "ProductionLaunchObservation":
        require_exact_keys(
            value,
            {
                "launch_key",
                "kernel_family",
                "native_identity",
                "process_ordinal_binding",
                "record_sha256",
            },
            "fixture v2 source record",
        )
        assert isinstance(value, dict)
        native = value["native_identity"]
        binding = value["process_ordinal_binding"]
        require_exact_keys(
            native,
            {
                "execution_context_id",
                "native_launch_id",
                "native_stream_id",
                "native_launch_index",
            },
            "native launch identity",
        )
        require_exact_keys(
            binding,
            {
                "binding_kind",
                "authority_model_seal_sha256",
                "process_launch_ordinal",
                "native_launch_index_role",
                "binding_sha256",
            },
            "process ordinal binding",
        )
        assert isinstance(native, dict) and isinstance(binding, dict)
        if binding["binding_kind"] != EXPLICIT_ORDINAL_BINDING:
            raise ImmutableArtifactError(
                "process ordinal lacks explicit workload-execution authority"
            )
        key = CompleteLaunchKey.from_model(value["launch_key"])
        if binding["process_launch_ordinal"] != key.process_launch_ordinal:
            raise ImmutableArtifactError("ordinal binding disagrees with launch key")
        return cls(
            launch_key=key,
            kernel_family=value["kernel_family"],
            native_context_id=native["execution_context_id"],
            native_launch_id=native["native_launch_id"],
            native_stream_id=native["native_stream_id"],
            native_launch_index=native["native_launch_index"],
            native_launch_index_role=binding["native_launch_index_role"],
            authority_model_seal_sha256=binding[
                "authority_model_seal_sha256"
            ],
            binding_sha256=binding["binding_sha256"],
            record_sha256=value["record_sha256"],
        )


# Explicit fixture aliases prevent structural algebra types from being mistaken
# for installed production authority.
FixtureV2ParserAuthority = ProductionParserAuthority
FixtureV2ExpectedLaunch = ExpectedProductionLaunch
FixtureV2LaunchObservation = ProductionLaunchObservation


def create_workload_execution_authority_fixture(
    *,
    executable_sha256: str,
    argv: Sequence[str],
    shape: Sequence[str],
    binary_architecture: str,
    gpu_sku: str,
    kernel_family: str,
    expected_launches: Sequence[ExpectedProductionLaunch],
    execution_nonce: str,
    parser_authorities: Mapping[LaunchNamespace, ProductionParserAuthority],
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    """Derive fixture-only IDs; this is not temporal or trust authority."""

    verify_retained_schema(
        schema_bytes,
        schema_snapshot,
        expected_sha256=WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_SHA256,
        expected_id=WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_ID,
    )
    require_sha256(executable_sha256, "workload executable SHA-256")
    normalized_argv = _text_sequence(argv, "argv", require_nonempty=True)
    normalized_shape = _text_sequence(shape, "shape", require_nonempty=False)
    _require_role(binary_architecture)
    _require_role(gpu_sku)
    _require_role(kernel_family)
    _require_role(execution_nonce)
    launches = _require_expected_launches(
        expected_launches, binary_architecture=binary_architecture
    )
    authorities = _require_parser_authorities(parser_authorities)
    workload_core = _workload_core(
        executable_sha256=executable_sha256,
        argv=normalized_argv,
        shape=normalized_shape,
        binary_architecture=binary_architecture,
        gpu_sku=gpu_sku,
        kernel_family=kernel_family,
        expected_launches=launches,
    )
    workload_id = f"workload-v1-{sha256_bytes(canonical_json_bytes(workload_core))}"
    execution_core = {
        "workload_id": workload_id,
        "execution_nonce": execution_nonce,
        "parser_authorities": {
            namespace.value.lower(): authorities[namespace].to_model()
            for namespace in LaunchNamespace
        },
    }
    matched_execution_id = (
        "matched-execution-v1-"
        f"{sha256_bytes(canonical_json_bytes(execution_core))}"
    )
    source_contracts = {
        namespace.value.lower(): {
            "namespace": namespace.value,
            "parser_authority_id": authorities[namespace].authority_id,
            "parser_authority_sha256": authorities[namespace].authority_sha256,
            "execution_context_id": _execution_context_id(
                matched_execution_id, namespace
            ),
            "ordinal_binding_policy": EXPLICIT_ORDINAL_BINDING,
        }
        for namespace in LaunchNamespace
    }
    roster = [
        {
            "launch_key": _complete_key_from_expected(
                launch,
                workload_id=workload_id,
                matched_execution_id=matched_execution_id,
            ).to_model(),
            "kernel_family": kernel_family,
        }
        for launch in launches
    ]
    model = seal_evaluator_model(
        {
            "schema_version": WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_VERSION,
            "artifact_kind": WORKLOAD_EXECUTION_AUTHORITY_ARTIFACT_KIND,
            "schema_sha256": WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_SHA256,
            "contract_mode": V2_FIXTURE_CONTRACT_MODE,
            "external_admission": {
                "status": "FIXTURE_ONLY_NON_AUTHORITATIVE",
                "detached_admission_sha256": None,
                "trust_anchor_id": None,
                "trust_anchor_sha256": None,
            },
            "issuance_claim": {
                "phase": "BEFORE_LABEL_COLLECTION",
                "input_policy": "TARGET_FREE_NO_LABEL_OR_PREDICTION_INPUTS",
                "execution_nonce": execution_nonce,
                "workload_id_rule": _WORKLOAD_ID_RULE,
                "matched_execution_id_rule": _MATCHED_EXECUTION_ID_RULE,
            },
            "workload_id": workload_id,
            "matched_execution_id": matched_execution_id,
            "executable_sha256": executable_sha256,
            "argv": list(normalized_argv),
            "shape": list(normalized_shape),
            "architecture": {
                "binary_architecture": binary_architecture,
                "gpu_sku": gpu_sku,
            },
            "kernel_family": kernel_family,
            "source_contracts": source_contracts,
            "expected_launch_count": len(roster),
            "expected_launch_roster": roster,
        }
    )
    _reject_target_surface(model, "workload execution authority")
    verify_workload_execution_authority_fixture_model(
        model, schema_bytes=schema_bytes, schema_snapshot=schema_snapshot
    )
    return model


def verify_workload_execution_authority_fixture_model(
    model: dict[str, Any],
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> None:
    verify_evaluator_model(
        model,
        expected_schema_version=WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_VERSION,
        expected_artifact_kind=WORKLOAD_EXECUTION_AUTHORITY_ARTIFACT_KIND,
        expected_schema_sha256=WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_SHA256,
        expected_schema_id=WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_ID,
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
    )
    require_exact_keys(model, _WORKLOAD_AUTHORITY_KEYS, "workload execution authority")
    if model["contract_mode"] != V2_FIXTURE_CONTRACT_MODE:
        raise ImmutableArtifactError("workload execution authority mode is invalid")
    if model["external_admission"] != {
        "status": "FIXTURE_ONLY_NON_AUTHORITATIVE",
        "detached_admission_sha256": None,
        "trust_anchor_id": None,
        "trust_anchor_sha256": None,
    }:
        raise ImmutableArtifactError(
            "fixture workload authority may not claim external admission"
        )
    issuance = model["issuance_claim"]
    require_exact_keys(
        issuance,
        {
            "phase",
            "input_policy",
            "execution_nonce",
            "workload_id_rule",
            "matched_execution_id_rule",
        },
        "workload execution issuance",
    )
    assert isinstance(issuance, dict)
    if issuance != {
        "phase": "BEFORE_LABEL_COLLECTION",
        "input_policy": "TARGET_FREE_NO_LABEL_OR_PREDICTION_INPUTS",
        "execution_nonce": issuance["execution_nonce"],
        "workload_id_rule": _WORKLOAD_ID_RULE,
        "matched_execution_id_rule": _MATCHED_EXECUTION_ID_RULE,
    }:
        raise ImmutableArtifactError("workload execution issuance policy is invalid")
    _require_role(issuance["execution_nonce"])
    require_sha256(model["executable_sha256"], "workload executable SHA-256")
    argv = _text_sequence(model["argv"], "argv", require_nonempty=True)
    shape = _text_sequence(model["shape"], "shape", require_nonempty=False)
    architecture = _architecture_from_model(model["architecture"])
    _require_role(model["kernel_family"])
    source_contracts = _parse_source_contracts(model["source_contracts"])
    raw_roster = model["expected_launch_roster"]
    if type(raw_roster) is not list or not raw_roster:
        raise ImmutableArtifactError("expected launch roster must be nonempty")
    keys: list[CompleteLaunchKey] = []
    for item in raw_roster:
        require_exact_keys(item, {"launch_key", "kernel_family"}, "expected launch")
        assert isinstance(item, dict)
        if item["kernel_family"] != model["kernel_family"]:
            raise ImmutableArtifactError("launch kernel family mismatch")
        keys.append(CompleteLaunchKey.from_model(item["launch_key"]))
    _require_complete_key_sequence(tuple(keys))
    if model["expected_launch_count"] != len(keys):
        raise ImmutableArtifactError("expected launch count mismatch")
    if any(
        key.workload_id != model["workload_id"]
        or key.matched_execution_id != model["matched_execution_id"]
        or key.binary_architecture != architecture[0]
        for key in keys
    ):
        raise ImmutableArtifactError("expected roster cohort or architecture mismatch")
    expected_launches = tuple(
        ExpectedProductionLaunch(
            process_launch_ordinal=key.process_launch_ordinal,
            kernel_mangled_name=key.kernel_mangled_name,
            grid=key.grid,
            block=key.block,
            device_id=key.device_id,
            stream_id=key.stream_id,
            binary_architecture=key.binary_architecture,
        )
        for key in keys
    )
    workload_core = _workload_core(
        executable_sha256=model["executable_sha256"],
        argv=argv,
        shape=shape,
        binary_architecture=architecture[0],
        gpu_sku=architecture[1],
        kernel_family=model["kernel_family"],
        expected_launches=expected_launches,
    )
    expected_workload_id = (
        "workload-v1-" f"{sha256_bytes(canonical_json_bytes(workload_core))}"
    )
    if model["workload_id"] != expected_workload_id:
        raise ImmutableArtifactError("workload_id derivation mismatch")
    parser_models = {
        namespace.value.lower(): source_contracts[namespace]["parser_authority"]
        for namespace in LaunchNamespace
    }
    execution_core = {
        "workload_id": expected_workload_id,
        "execution_nonce": issuance["execution_nonce"],
        "parser_authorities": parser_models,
    }
    expected_execution_id = (
        "matched-execution-v1-"
        f"{sha256_bytes(canonical_json_bytes(execution_core))}"
    )
    if model["matched_execution_id"] != expected_execution_id:
        raise ImmutableArtifactError("matched_execution_id derivation mismatch")
    contexts: list[str] = []
    authority_ids: list[str] = []
    for namespace in LaunchNamespace:
        contract = source_contracts[namespace]
        expected_context = _execution_context_id(expected_execution_id, namespace)
        if contract["execution_context_id"] != expected_context:
            raise ImmutableArtifactError("source execution context derivation mismatch")
        contexts.append(contract["execution_context_id"])
        authority_ids.append(contract["parser_authority"]["authority_id"])
    if len(set(contexts)) != len(contexts):
        raise ImmutableArtifactError("source execution contexts alias")
    if len(set(authority_ids)) != len(authority_ids):
        raise ImmutableArtifactError("parser authority identities alias across namespaces")
    _reject_target_surface(model, "workload execution authority")


_WORKLOAD_AUTHORITY_KEYS = {
    "schema_version",
    "artifact_kind",
    "schema_sha256",
    "contract_mode",
    "external_admission",
    "issuance_claim",
    "workload_id",
    "matched_execution_id",
    "executable_sha256",
    "argv",
    "shape",
    "architecture",
    "kernel_family",
    "source_contracts",
    "expected_launch_count",
    "expected_launch_roster",
    "model_seal_sha256",
}


def load_workload_execution_authority_fixture(
    payload: bytes,
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    model = strict_json_object(payload, label="workload execution authority")
    if canonical_json_bytes(model) != payload:
        raise ImmutableArtifactError("workload execution authority is not canonical JSON")
    verify_workload_execution_authority_fixture_model(
        model, schema_bytes=schema_bytes, schema_snapshot=schema_snapshot
    )
    return model


def require_current_m1_workload_execution_authority(authority_id: str) -> None:
    _require_role(authority_id)
    if authority_id not in CURRENT_M1_WORKLOAD_EXECUTION_AUTHORITIES:
        raise authority_unavailable(
            AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE,
            "current M1 evidence has no pre-label workload/execution authority",
        )
    raise ImmutableArtifactError("registered M1 authority loader is not implemented")


def create_launch_source_payload_v2_fixture(
    *,
    namespace: LaunchNamespace,
    workload_execution_authority: dict[str, Any],
    observations: Sequence[ProductionLaunchObservation],
    adapter_evidence: V2FixtureAdapterEvidence,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    authority_schema_bytes: bytes,
    authority_schema_snapshot: RetainedPayloadSnapshot,
) -> bytes:
    """Build a fixture identity projection bound to exact retained inputs."""

    verify_retained_schema(
        schema_bytes,
        schema_snapshot,
        expected_sha256=CROSSWALK_V2_SCHEMA_SHA256,
        expected_id=CROSSWALK_V2_SCHEMA_ID,
    )
    verify_workload_execution_authority_fixture_model(
        workload_execution_authority,
        schema_bytes=authority_schema_bytes,
        schema_snapshot=authority_schema_snapshot,
    )
    if not isinstance(namespace, LaunchNamespace):
        raise ImmutableArtifactError("fixture v2 source namespace is invalid")
    if not isinstance(adapter_evidence, V2FixtureAdapterEvidence):
        raise ImmutableArtifactError("fixture v2 adapter evidence is invalid")
    underlying_evidence = adapter_evidence.underlying_evidence.to_model()
    parser_implementation = adapter_evidence.parser_implementation.to_model()
    material = tuple(observations)
    _require_observations_against_authority(
        material, namespace, workload_execution_authority
    )
    contract = workload_execution_authority["source_contracts"][
        namespace.value.lower()
    ]
    if parser_implementation["payload_sha256"] != contract["parser_authority_sha256"]:
        raise ImmutableArtifactError(
            "retained parser implementation does not match predeclared parser identity"
        )
    if (
        namespace is LaunchNamespace.CANONICAL
        and adapter_evidence.underlying_evidence.payload
        != canonical_json_bytes(workload_execution_authority)
    ):
        raise ImmutableArtifactError(
            "canonical adapter evidence is not the exact fixture authority bytes"
        )
    model = seal_evaluator_model(
        {
            "schema_version": CROSSWALK_V2_SCHEMA_VERSION,
            "artifact_kind": PRODUCTION_SOURCE_ARTIFACT_KIND,
            "schema_sha256": CROSSWALK_V2_SCHEMA_SHA256,
            "contract_mode": V2_FIXTURE_CONTRACT_MODE,
            "namespace": namespace.value,
            "workload_execution_authority_model_seal_sha256": (
                workload_execution_authority["model_seal_sha256"]
            ),
            "parser_authority": {
                "authority_id": contract["parser_authority_id"],
                "authority_sha256": contract["parser_authority_sha256"],
            },
            "adapter_evidence": {
                "underlying_evidence": underlying_evidence,
                "parser_implementation": parser_implementation,
            },
            "execution_context_id": contract["execution_context_id"],
            "executable_sha256": workload_execution_authority[
                "executable_sha256"
            ],
            "architecture": workload_execution_authority["architecture"],
            "kernel_family": workload_execution_authority["kernel_family"],
            "record_count": len(material),
            "records": [observation.to_model() for observation in material],
        }
    )
    _reject_target_surface(model, "fixture v2 launch source")
    return canonical_json_bytes(model)


def create_launch_inventory_v2_fixture(
    *,
    source_input: V2FixtureSourceInput,
    workload_execution_authority: dict[str, Any],
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    authority_schema_bytes: bytes,
    authority_schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    """Build a fixture inventory from every retained projection record."""

    if not isinstance(source_input, V2FixtureSourceInput):
        raise ImmutableArtifactError("fixture v2 source input is invalid")
    source_model, namespace, observations = _load_launch_source_v2_fixture(
        source_input=source_input,
        workload_execution_authority=workload_execution_authority,
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
        authority_schema_bytes=authority_schema_bytes,
        authority_schema_snapshot=authority_schema_snapshot,
    )
    inventory = seal_evaluator_model(
        {
            "schema_version": CROSSWALK_V2_SCHEMA_VERSION,
            "artifact_kind": PRODUCTION_INVENTORY_ARTIFACT_KIND,
            "schema_sha256": CROSSWALK_V2_SCHEMA_SHA256,
            "contract_mode": V2_FIXTURE_CONTRACT_MODE,
            "namespace": namespace.value,
            "workload_execution_authority_model_seal_sha256": (
                workload_execution_authority["model_seal_sha256"]
            ),
            "source": {
                "payload_sha256": source_input.source_projection.snapshot.payload_sha256,
                "payload_snapshot": source_input.source_projection.snapshot.to_model(),
                "source_model_seal_sha256": source_model["model_seal_sha256"],
                "adapter_evidence": source_model["adapter_evidence"],
                "parser_authority": source_model["parser_authority"],
                "execution_context_id": source_model["execution_context_id"],
            },
            "executable_sha256": source_model["executable_sha256"],
            "architecture": source_model["architecture"],
            "kernel_family": source_model["kernel_family"],
            "launch_count": len(observations),
            "launch_key_sha256s": [item.launch_key.sha256 for item in observations],
            "records": [item.to_model() for item in observations],
        }
    )
    _reject_target_surface(inventory, "production launch inventory")
    return inventory


def verify_launch_inventory_v2_fixture_model(
    model: dict[str, Any],
    *,
    source_input: V2FixtureSourceInput,
    workload_execution_authority: dict[str, Any],
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    authority_schema_bytes: bytes,
    authority_schema_snapshot: RetainedPayloadSnapshot,
) -> None:
    verify_evaluator_model(
        model,
        expected_schema_version=CROSSWALK_V2_SCHEMA_VERSION,
        expected_artifact_kind=PRODUCTION_INVENTORY_ARTIFACT_KIND,
        expected_schema_sha256=CROSSWALK_V2_SCHEMA_SHA256,
        expected_schema_id=CROSSWALK_V2_SCHEMA_ID,
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
    )
    require_exact_keys(model, _V2_INVENTORY_KEYS, "production inventory")
    expected = create_launch_inventory_v2_fixture(
        source_input=source_input,
        workload_execution_authority=workload_execution_authority,
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
        authority_schema_bytes=authority_schema_bytes,
        authority_schema_snapshot=authority_schema_snapshot,
    )
    if canonical_json_bytes(model) != canonical_json_bytes(expected):
        raise ImmutableArtifactError(
            "production inventory does not match retained complete source"
        )


_V2_INVENTORY_KEYS = {
    "schema_version",
    "artifact_kind",
    "schema_sha256",
    "contract_mode",
    "namespace",
    "workload_execution_authority_model_seal_sha256",
    "source",
    "executable_sha256",
    "architecture",
    "kernel_family",
    "launch_count",
    "launch_key_sha256s",
    "records",
    "model_seal_sha256",
}


def evaluate_launch_crosswalk_v2_fixture(
    *,
    workload_execution_authority: dict[str, Any],
    inventories: Mapping[LaunchNamespace, dict[str, Any]],
    inventory_sources: Mapping[LaunchNamespace, V2FixtureSourceInput],
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    authority_schema_bytes: bytes,
    authority_schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    """Check fixture-only four-way bijection without admitting production."""

    verify_retained_schema(
        schema_bytes,
        schema_snapshot,
        expected_sha256=CROSSWALK_V2_SCHEMA_SHA256,
        expected_id=CROSSWALK_V2_SCHEMA_ID,
    )
    verify_workload_execution_authority_fixture_model(
        workload_execution_authority,
        schema_bytes=authority_schema_bytes,
        schema_snapshot=authority_schema_snapshot,
    )
    normalized_inventories = _require_namespace_mapping(inventories, "inventories")
    normalized_sources = _require_namespace_mapping(
        inventory_sources, "inventory_sources"
    )
    for namespace in LaunchNamespace:
        source = normalized_sources[namespace]
        if not isinstance(source, V2FixtureSourceInput):
            raise ImmutableArtifactError("fixture inventory source input is invalid")
        verify_launch_inventory_v2_fixture_model(
            normalized_inventories[namespace],
            source_input=source,
            workload_execution_authority=workload_execution_authority,
            schema_bytes=schema_bytes,
            schema_snapshot=schema_snapshot,
            authority_schema_bytes=authority_schema_bytes,
            authority_schema_snapshot=authority_schema_snapshot,
        )
        if normalized_inventories[namespace]["namespace"] != namespace.value:
            raise ImmutableArtifactError("inventory is installed under wrong namespace")
    _require_nonaliased_inventory_contexts(normalized_inventories)
    expected_roster = workload_execution_authority["expected_launch_roster"]
    expected_key_models = [item["launch_key"] for item in expected_roster]
    for namespace in LaunchNamespace:
        inventory = normalized_inventories[namespace]
        observed = [item["launch_key"] for item in inventory["records"]]
        if observed != expected_key_models:
            raise ImmutableArtifactError(
                f"{namespace.value} inventory is missing, duplicated, or reordered"
            )
        if inventory["launch_count"] != len(expected_key_models):
            raise ImmutableArtifactError(f"{namespace.value} launch count mismatch")
        if (
            inventory["executable_sha256"]
            != workload_execution_authority["executable_sha256"]
            or inventory["architecture"]
            != workload_execution_authority["architecture"]
        ):
            raise ImmutableArtifactError(
                f"{namespace.value} binary or architecture mismatch"
            )
    rows: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_roster):
        key = CompleteLaunchKey.from_model(expected["launch_key"])
        rows.append(
            {
                "launch_key": key.to_model(),
                "launch_key_sha256": key.sha256,
                "kernel_family": expected["kernel_family"],
                "members": {
                    namespace.value.lower(): {
                        "inventory_model_seal_sha256": normalized_inventories[
                            namespace
                        ]["model_seal_sha256"],
                        "record_sha256": normalized_inventories[namespace]["records"][
                            index
                        ]["record_sha256"],
                    }
                    for namespace in LaunchNamespace
                },
            }
        )
    crosswalk = seal_evaluator_model(
        {
            "schema_version": CROSSWALK_V2_SCHEMA_VERSION,
            "artifact_kind": CROSSWALK_V2_ARTIFACT_KIND,
            "schema_sha256": CROSSWALK_V2_SCHEMA_SHA256,
            "contract_mode": V2_FIXTURE_CONTRACT_MODE,
            "workload_execution_authority_model_seal_sha256": (
                workload_execution_authority["model_seal_sha256"]
            ),
            "workload_id": workload_execution_authority["workload_id"],
            "matched_execution_id": workload_execution_authority[
                "matched_execution_id"
            ],
            "executable_sha256": workload_execution_authority[
                "executable_sha256"
            ],
            "architecture": workload_execution_authority["architecture"],
            "kernel_family": workload_execution_authority["kernel_family"],
            "launch_count": len(rows),
            "inventories": {
                namespace.value.lower(): normalized_inventories[namespace]
                for namespace in LaunchNamespace
            },
            "bijection": rows,
            "bijection_sha256": sha256_bytes(canonical_json_bytes(rows)),
            "outcome": ContractOutcome(
                OutcomeStatus.UNAVAILABLE, (V2_FIXTURE_CROSSWALK_REASON,)
            ).to_model(),
        }
    )
    _reject_target_surface(crosswalk, "fixture v2 launch crosswalk")
    return crosswalk


def verify_launch_crosswalk_v2_fixture_model(
    model: dict[str, Any],
    *,
    workload_execution_authority: dict[str, Any],
    inventories: Mapping[LaunchNamespace, dict[str, Any]],
    inventory_sources: Mapping[LaunchNamespace, V2FixtureSourceInput],
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    authority_schema_bytes: bytes,
    authority_schema_snapshot: RetainedPayloadSnapshot,
) -> None:
    verify_evaluator_model(
        model,
        expected_schema_version=CROSSWALK_V2_SCHEMA_VERSION,
        expected_artifact_kind=CROSSWALK_V2_ARTIFACT_KIND,
        expected_schema_sha256=CROSSWALK_V2_SCHEMA_SHA256,
        expected_schema_id=CROSSWALK_V2_SCHEMA_ID,
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
    )
    require_exact_keys(model, _V2_CROSSWALK_KEYS, "fixture v2 crosswalk")
    outcome = outcome_from_model(
        model["outcome"],
        rejected_reasons=frozenset(),
        unavailable_reasons=frozenset({V2_FIXTURE_CROSSWALK_REASON}),
    )
    expected_outcome = ContractOutcome(
        OutcomeStatus.UNAVAILABLE, (V2_FIXTURE_CROSSWALK_REASON,)
    )
    if outcome != expected_outcome:
        raise ImmutableArtifactError("fixture v2 crosswalk claimed production admission")
    expected = evaluate_launch_crosswalk_v2_fixture(
        workload_execution_authority=workload_execution_authority,
        inventories=inventories,
        inventory_sources=inventory_sources,
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
        authority_schema_bytes=authority_schema_bytes,
        authority_schema_snapshot=authority_schema_snapshot,
    )
    if canonical_json_bytes(model) != canonical_json_bytes(expected):
        raise ImmutableArtifactError(
            "fixture v2 crosswalk does not match retained inventories"
        )


_V2_CROSSWALK_KEYS = {
    "schema_version",
    "artifact_kind",
    "schema_sha256",
    "contract_mode",
    "workload_execution_authority_model_seal_sha256",
    "workload_id",
    "matched_execution_id",
    "executable_sha256",
    "architecture",
    "kernel_family",
    "launch_count",
    "inventories",
    "bijection",
    "bijection_sha256",
    "outcome",
    "model_seal_sha256",
}


def load_launch_crosswalk_v2_fixture(
    payload: bytes,
    **verification_inputs: Any,
) -> dict[str, Any]:
    model = strict_json_object(payload, label="fixture v2 launch crosswalk")
    if canonical_json_bytes(model) != payload:
        raise ImmutableArtifactError("fixture v2 launch crosswalk is not canonical JSON")
    verify_launch_crosswalk_v2_fixture_model(model, **verification_inputs)
    return model


# Public production-v2 APIs remain fail-closed until independent registries bind
# a detached pre-label admission, its trust anchor, and retained namespace
# adapter implementations. They deliberately refuse before inspecting payloads.
def create_workload_execution_authority(**_inputs: Any) -> dict[str, Any]:
    _require_production_v2_authorities()
    raise ImmutableArtifactError("production v2 authority constructor is unavailable")


def verify_workload_execution_authority_model(
    _model: dict[str, Any], **_inputs: Any
) -> None:
    _require_production_v2_authorities()
    raise ImmutableArtifactError("production v2 authority verifier is unavailable")


def load_workload_execution_authority(
    _payload: bytes, **_inputs: Any
) -> dict[str, Any]:
    _require_production_v2_authorities()
    raise ImmutableArtifactError("production v2 authority loader is unavailable")


def create_production_launch_source_payload(**_inputs: Any) -> bytes:
    _require_production_v2_authorities()
    raise ImmutableArtifactError("production v2 source constructor is unavailable")


def create_production_launch_inventory(**_inputs: Any) -> dict[str, Any]:
    _require_production_v2_authorities()
    raise ImmutableArtifactError("production v2 inventory constructor is unavailable")


def verify_production_launch_inventory_model(
    _model: dict[str, Any], **_inputs: Any
) -> None:
    _require_production_v2_authorities()
    raise ImmutableArtifactError("production v2 inventory verifier is unavailable")


def evaluate_production_launch_crosswalk(**_inputs: Any) -> dict[str, Any]:
    _require_production_v2_authorities()
    raise ImmutableArtifactError("production v2 crosswalk evaluator is unavailable")


def verify_production_launch_crosswalk_model(
    _model: dict[str, Any], **_inputs: Any
) -> None:
    _require_production_v2_authorities()
    raise ImmutableArtifactError("production v2 crosswalk verifier is unavailable")


def load_production_launch_crosswalk(
    _payload: bytes, **_inputs: Any
) -> dict[str, Any]:
    _require_production_v2_authorities()
    raise ImmutableArtifactError("production v2 crosswalk loader is unavailable")


evaluate_launch_crosswalk_v2 = evaluate_production_launch_crosswalk
verify_launch_crosswalk_v2_model = verify_production_launch_crosswalk_model
load_launch_crosswalk_v2 = load_production_launch_crosswalk


def _load_launch_source_v2_fixture(
    *,
    source_input: V2FixtureSourceInput,
    workload_execution_authority: dict[str, Any],
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    authority_schema_bytes: bytes,
    authority_schema_snapshot: RetainedPayloadSnapshot,
) -> tuple[dict[str, Any], LaunchNamespace, tuple[ProductionLaunchObservation, ...]]:
    verify_retained_schema(
        schema_bytes,
        schema_snapshot,
        expected_sha256=CROSSWALK_V2_SCHEMA_SHA256,
        expected_id=CROSSWALK_V2_SCHEMA_ID,
    )
    verify_workload_execution_authority_fixture_model(
        workload_execution_authority,
        schema_bytes=authority_schema_bytes,
        schema_snapshot=authority_schema_snapshot,
    )
    if not isinstance(source_input, V2FixtureSourceInput):
        raise ImmutableArtifactError("fixture source input is invalid")
    source_payload = source_input.source_projection.payload
    source_snapshot = source_input.source_projection.snapshot
    source_model = strict_json_object(source_payload, label="fixture v2 launch source")
    if canonical_json_bytes(source_model) != source_payload:
        raise ImmutableArtifactError("fixture v2 launch source is not canonical JSON")
    verify_evaluator_model(
        source_model,
        expected_schema_version=CROSSWALK_V2_SCHEMA_VERSION,
        expected_artifact_kind=PRODUCTION_SOURCE_ARTIFACT_KIND,
        expected_schema_sha256=CROSSWALK_V2_SCHEMA_SHA256,
        expected_schema_id=CROSSWALK_V2_SCHEMA_ID,
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
    )
    require_exact_keys(source_model, _V2_SOURCE_KEYS, "fixture v2 source")
    if source_model["contract_mode"] != V2_FIXTURE_CONTRACT_MODE:
        raise ImmutableArtifactError("fixture v2 source mode is invalid")
    try:
        namespace = LaunchNamespace(source_model["namespace"])
    except (TypeError, ValueError) as exc:
        raise ImmutableArtifactError("fixture v2 source namespace is invalid") from exc
    contract = workload_execution_authority["source_contracts"][
        namespace.value.lower()
    ]
    expected_parser = {
        "authority_id": contract["parser_authority_id"],
        "authority_sha256": contract["parser_authority_sha256"],
    }
    if source_model["parser_authority"] != expected_parser:
        raise ImmutableArtifactError("fixture v2 source parser authority mismatch")
    if (
        source_model["workload_execution_authority_model_seal_sha256"]
        != workload_execution_authority["model_seal_sha256"]
        or source_model["execution_context_id"]
        != contract["execution_context_id"]
        or source_model["executable_sha256"]
        != workload_execution_authority["executable_sha256"]
        or source_model["architecture"]
        != workload_execution_authority["architecture"]
        or source_model["kernel_family"]
        != workload_execution_authority["kernel_family"]
    ):
        raise ImmutableArtifactError("fixture v2 source authority context mismatch")
    adapter_model = source_input.adapter_evidence.to_model()
    if source_model["adapter_evidence"] != adapter_model:
        raise ImmutableArtifactError("fixture adapter evidence binding mismatch")
    if (
        adapter_model["parser_implementation"]["payload_sha256"]
        != contract["parser_authority_sha256"]
    ):
        raise ImmutableArtifactError("fixture parser implementation identity mismatch")
    if (
        namespace is LaunchNamespace.CANONICAL
        and source_input.adapter_evidence.underlying_evidence.payload
        != canonical_json_bytes(workload_execution_authority)
    ):
        raise ImmutableArtifactError("canonical fixture evidence bytes mismatch")
    raw_records = source_model["records"]
    if type(raw_records) is not list:
        raise ImmutableArtifactError("fixture v2 source records must be an array")
    observations = tuple(
        ProductionLaunchObservation.from_model(item) for item in raw_records
    )
    if source_model["record_count"] != len(observations):
        raise ImmutableArtifactError("fixture v2 source record count mismatch")
    _require_observations_against_authority(
        observations, namespace, workload_execution_authority
    )
    _reject_target_surface(source_model, "fixture v2 launch source")
    return source_model, namespace, observations


_V2_SOURCE_KEYS = {
    "schema_version",
    "artifact_kind",
    "schema_sha256",
    "contract_mode",
    "namespace",
    "workload_execution_authority_model_seal_sha256",
    "parser_authority",
    "adapter_evidence",
    "execution_context_id",
    "executable_sha256",
    "architecture",
    "kernel_family",
    "record_count",
    "records",
    "model_seal_sha256",
}


def _require_production_v2_authorities() -> None:
    if (
        not CURRENT_M1_WORKLOAD_EXECUTION_AUTHORITIES
        or not PRODUCTION_WORKLOAD_EXECUTION_ADMISSION_AUTHORITIES
        or not PRODUCTION_PRELABEL_TRUST_ANCHORS
    ):
        raise authority_unavailable(
            AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE,
            "production v2 requires a detached externally anchored pre-label "
            "workload/execution admission; none is installed for current M1",
        )
    if not PRODUCTION_NAMESPACE_ADAPTER_AUTHORITIES:
        raise authority_unavailable(
            AuthorityBlockerCode.LAUNCH_PARSER_AUTHORITY_UNAVAILABLE,
            "production v2 namespace adapter authorities are not installed",
        )


def _require_expected_launches(
    launches: Sequence[ExpectedProductionLaunch], *, binary_architecture: str
) -> tuple[ExpectedProductionLaunch, ...]:
    if type(launches) not in {list, tuple} or not launches:
        raise ImmutableArtifactError("expected launch roster must be a nonempty sequence")
    material = tuple(launches)
    if any(not isinstance(item, ExpectedProductionLaunch) for item in material):
        raise ImmutableArtifactError("expected launch roster contains an invalid row")
    expected_ordinals = tuple(range(len(material)))
    observed_ordinals = tuple(item.process_launch_ordinal for item in material)
    if observed_ordinals != expected_ordinals:
        raise ImmutableArtifactError(
            "expected launch roster is missing, duplicated, or reordered"
        )
    if any(item.binary_architecture != binary_architecture for item in material):
        raise ImmutableArtifactError("expected launch architecture mismatch")
    return material


def _require_parser_authorities(
    values: Mapping[LaunchNamespace, ProductionParserAuthority],
) -> dict[LaunchNamespace, ProductionParserAuthority]:
    normalized = _require_namespace_mapping(values, "parser_authorities")
    result: dict[LaunchNamespace, ProductionParserAuthority] = {}
    for namespace in LaunchNamespace:
        authority = normalized[namespace]
        if (
            not isinstance(authority, ProductionParserAuthority)
            or authority.namespace is not namespace
        ):
            raise ImmutableArtifactError("parser authority is installed under wrong namespace")
        result[namespace] = authority
    ids = [item.authority_id for item in result.values()]
    if len(ids) != len(set(ids)):
        raise ImmutableArtifactError("parser authority identities alias across namespaces")
    return result


def _workload_core(
    *,
    executable_sha256: str,
    argv: tuple[str, ...],
    shape: tuple[str, ...],
    binary_architecture: str,
    gpu_sku: str,
    kernel_family: str,
    expected_launches: tuple[ExpectedProductionLaunch, ...],
) -> dict[str, Any]:
    return {
        "executable_sha256": executable_sha256,
        "argv": list(argv),
        "shape": list(shape),
        "architecture": {
            "binary_architecture": binary_architecture,
            "gpu_sku": gpu_sku,
        },
        "kernel_family": kernel_family,
        "expected_launch_roster": [item.to_identity_model() for item in expected_launches],
    }


def _complete_key_from_expected(
    launch: ExpectedProductionLaunch,
    *,
    workload_id: str,
    matched_execution_id: str,
) -> CompleteLaunchKey:
    return CompleteLaunchKey(
        workload_id=workload_id,
        matched_execution_id=matched_execution_id,
        process_launch_ordinal=launch.process_launch_ordinal,
        kernel_mangled_name=launch.kernel_mangled_name,
        grid=launch.grid,
        block=launch.block,
        device_id=launch.device_id,
        stream_id=launch.stream_id,
        binary_architecture=launch.binary_architecture,
    )


def _execution_context_id(
    matched_execution_id: str, namespace: LaunchNamespace
) -> str:
    digest = sha256_bytes(
        canonical_json_bytes(
            {
                "matched_execution_id": matched_execution_id,
                "namespace": namespace.value,
                "context_rule": "DISTINCT_NAMESPACE_EXECUTION_CONTEXT_V1",
            }
        )
    )
    return f"{namespace.value.lower()}-context-v1-{digest}"


def _parse_source_contracts(
    value: object,
) -> dict[LaunchNamespace, dict[str, Any]]:
    require_exact_keys(
        value,
        {namespace.value.lower() for namespace in LaunchNamespace},
        "source contracts",
    )
    assert isinstance(value, dict)
    result: dict[LaunchNamespace, dict[str, Any]] = {}
    for namespace in LaunchNamespace:
        contract = value[namespace.value.lower()]
        require_exact_keys(
            contract,
            {
                "namespace",
                "parser_authority_id",
                "parser_authority_sha256",
                "execution_context_id",
                "ordinal_binding_policy",
            },
            "source contract",
        )
        assert isinstance(contract, dict)
        if (
            contract["namespace"] != namespace.value
            or contract["ordinal_binding_policy"] != EXPLICIT_ORDINAL_BINDING
        ):
            raise ImmutableArtifactError("source contract namespace or policy mismatch")
        _require_role(contract["parser_authority_id"])
        require_sha256(contract["parser_authority_sha256"], "parser authority SHA-256")
        _require_context_id(contract["execution_context_id"])
        result[namespace] = {
            **contract,
            "parser_authority": {
                "authority_id": contract["parser_authority_id"],
                "authority_sha256": contract["parser_authority_sha256"],
            },
        }
    return result


def _require_observations_against_authority(
    observations: tuple[ProductionLaunchObservation, ...],
    namespace: LaunchNamespace,
    authority: dict[str, Any],
) -> None:
    if not observations:
        raise ImmutableArtifactError("fixture v2 source inventory must be nonempty")
    expected = authority["expected_launch_roster"]
    if len(observations) != len(expected):
        raise ImmutableArtifactError("fixture v2 source launch cardinality mismatch")
    context = authority["source_contracts"][namespace.value.lower()][
        "execution_context_id"
    ]
    expected_keys = [row["launch_key"] for row in expected]
    observed_keys = [item.launch_key.to_model() for item in observations]
    if observed_keys != expected_keys:
        raise ImmutableArtifactError(
            "fixture v2 source is missing, duplicated, or reordered"
        )
    native_ids: list[tuple[str, str]] = []
    native_indices: list[tuple[str, int]] = []
    for item, expected_row in zip(observations, expected, strict=True):
        if item.native_context_id != context:
            raise ImmutableArtifactError("source record execution context mismatch")
        if item.kernel_family != expected_row["kernel_family"]:
            raise ImmutableArtifactError("source record kernel family mismatch")
        if item.authority_model_seal_sha256 != authority["model_seal_sha256"]:
            raise ImmutableArtifactError("source ordinal binding authority mismatch")
        native_ids.append((item.native_context_id, item.native_launch_id))
        if item.native_launch_index is not None:
            native_indices.append((item.native_context_id, item.native_launch_index))
    if len(native_ids) != len(set(native_ids)):
        raise ImmutableArtifactError("duplicate or context-aliased native launch identity")
    if len(native_indices) != len(set(native_indices)):
        raise ImmutableArtifactError("duplicate native launch index is ambiguous")


def _require_nonaliased_inventory_contexts(
    inventories: Mapping[LaunchNamespace, dict[str, Any]],
) -> None:
    payloads = [inventories[item]["source"]["payload_sha256"] for item in LaunchNamespace]
    snapshots = [
        inventories[item]["source"]["payload_snapshot"]["snapshot_identity_sha256"]
        for item in LaunchNamespace
    ]
    evidence_payloads = [
        inventories[item]["source"]["adapter_evidence"]["underlying_evidence"][
            "payload_sha256"
        ]
        for item in LaunchNamespace
    ]
    evidence_snapshots = [
        inventories[item]["source"]["adapter_evidence"]["underlying_evidence"][
            "payload_snapshot"
        ]["snapshot_identity_sha256"]
        for item in LaunchNamespace
    ]
    contexts = [
        inventories[item]["source"]["execution_context_id"]
        for item in LaunchNamespace
    ]
    for values, label in (
        (payloads, "source payloads"),
        (snapshots, "source snapshots"),
        (evidence_payloads, "underlying evidence payloads"),
        (evidence_snapshots, "underlying evidence snapshots"),
        (contexts, "execution contexts"),
    ):
        if len(values) != len(set(values)):
            raise ImmutableArtifactError(f"cross-namespace {label} alias")


def _observation_binding_from_values(values: Mapping[str, Any]) -> str:
    launch_key = values["launch_key"]
    assert isinstance(launch_key, CompleteLaunchKey)
    preimage = {
        "binding_kind": EXPLICIT_ORDINAL_BINDING,
        "authority_model_seal_sha256": values["authority_model_seal_sha256"],
        "process_launch_ordinal": launch_key.process_launch_ordinal,
        "native_launch_index_role": values["native_launch_index_role"],
        "native_identity": {
            "execution_context_id": values["native_context_id"],
            "native_launch_id": values["native_launch_id"],
            "native_stream_id": values["native_stream_id"],
            "native_launch_index": values["native_launch_index"],
        },
    }
    return sha256_bytes(canonical_json_bytes(preimage))


def _observation_binding_sha256(value: ProductionLaunchObservation) -> str:
    return _observation_binding_from_values(
        {
            "launch_key": value.launch_key,
            "native_context_id": value.native_context_id,
            "native_launch_id": value.native_launch_id,
            "native_stream_id": value.native_stream_id,
            "native_launch_index": value.native_launch_index,
            "native_launch_index_role": value.native_launch_index_role,
            "authority_model_seal_sha256": value.authority_model_seal_sha256,
        }
    )


def _observation_record_from_values(
    values: Mapping[str, Any], binding_sha256: str
) -> str:
    launch_key = values["launch_key"]
    assert isinstance(launch_key, CompleteLaunchKey)
    preimage = {
        "launch_key": launch_key.to_model(),
        "kernel_family": values["kernel_family"],
        "native_identity": {
            "execution_context_id": values["native_context_id"],
            "native_launch_id": values["native_launch_id"],
            "native_stream_id": values["native_stream_id"],
            "native_launch_index": values["native_launch_index"],
        },
        "process_ordinal_binding": {
            "binding_kind": EXPLICIT_ORDINAL_BINDING,
            "authority_model_seal_sha256": values["authority_model_seal_sha256"],
            "process_launch_ordinal": launch_key.process_launch_ordinal,
            "native_launch_index_role": values["native_launch_index_role"],
            "binding_sha256": binding_sha256,
        },
    }
    return sha256_bytes(canonical_json_bytes(preimage))


def _observation_record_sha256(value: ProductionLaunchObservation) -> str:
    return _observation_record_from_values(
        {
            "launch_key": value.launch_key,
            "kernel_family": value.kernel_family,
            "native_context_id": value.native_context_id,
            "native_launch_id": value.native_launch_id,
            "native_stream_id": value.native_stream_id,
            "native_launch_index": value.native_launch_index,
            "native_launch_index_role": value.native_launch_index_role,
            "authority_model_seal_sha256": value.authority_model_seal_sha256,
        },
        value.binding_sha256,
    )


def _require_complete_key_sequence(keys: tuple[CompleteLaunchKey, ...]) -> None:
    ordinals = tuple(key.process_launch_ordinal for key in keys)
    if ordinals != tuple(range(len(keys))):
        raise ImmutableArtifactError("complete launch keys are missing, duplicated, or reordered")
    models = [canonical_json_bytes(key.to_model()) for key in keys]
    if len(models) != len(set(models)):
        raise ImmutableArtifactError("duplicate complete launch key")


def _require_namespace_mapping(value: object, label: str) -> dict[LaunchNamespace, Any]:
    if not isinstance(value, Mapping) or set(value) != set(LaunchNamespace):
        raise ImmutableArtifactError(f"{label} must contain exactly four namespace keys")
    return {namespace: value[namespace] for namespace in LaunchNamespace}


def _architecture_from_model(value: object) -> tuple[str, str]:
    require_exact_keys(value, {"binary_architecture", "gpu_sku"}, "architecture")
    assert isinstance(value, dict)
    _require_role(value["binary_architecture"])
    _require_role(value["gpu_sku"])
    return value["binary_architecture"], value["gpu_sku"]


def _dim3_model(value: tuple[int, int, int]) -> dict[str, int]:
    _require_uint32_dimension(value, "dim3")
    return {"x": value[0], "y": value[1], "z": value[2]}


def _dim3_from_model(value: object, label: str) -> tuple[int, int, int]:
    require_exact_keys(value, {"x", "y", "z"}, label)
    assert isinstance(value, dict)
    result = (value["x"], value["y"], value["z"])
    _require_uint32_dimension(result, label)
    return result


def _require_uint32_dimension(value: object, label: str) -> None:
    if type(value) is not tuple or len(value) != 3:
        raise ImmutableArtifactError(f"{label} must be an exact three-integer tuple")
    if any(
        type(component) is not int or not 1 <= component <= (1 << 32) - 1
        for component in value
    ):
        raise ImmutableArtifactError(
            f"{label} components must be positive uint32 integers"
        )


def _require_uint32(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= (1 << 32) - 1:
        raise ImmutableArtifactError(f"{label} must be a uint32")
    return value


def _require_ordinal_value(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_SAFE_INTEGER:
        raise ImmutableArtifactError(f"{label} must be a nonnegative safe integer")
    return value


def _require_context_id(value: object) -> str:
    require_nfc(value, "execution_context_id")
    if type(value) is not str or _CONTEXT_ID.fullmatch(value) is None:
        raise ImmutableArtifactError("execution_context_id is invalid")
    return value


def _require_text(value: object, label: str) -> str:
    require_nfc(value, label)
    assert isinstance(value, str)
    if len(value.encode("utf-8")) > 131071:
        raise ImmutableArtifactError(f"{label} exceeds the encoded size bound")
    return value


def _text_sequence(
    value: object, label: str, *, require_nonempty: bool
) -> tuple[str, ...]:
    if type(value) not in {list, tuple}:
        raise ImmutableArtifactError(f"{label} must be an ordered list or tuple")
    material = tuple(value)
    if require_nonempty and not material:
        raise ImmutableArtifactError(f"{label} must be nonempty")
    if len(material) > 4096:
        raise ImmutableArtifactError(f"{label} exceeds the cardinality bound")
    for index, item in enumerate(material):
        _require_text(item, f"{label}[{index}]")
    return material  # type: ignore[return-value]


def _reject_target_surface(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if type(key) is not str:
                raise ImmutableArtifactError(f"{label} contains a non-string field")
            if key.lower() in _FORBIDDEN_TARGET_FIELDS:
                raise ImmutableArtifactError(f"{label} contains forbidden target field {key}")
            _reject_target_surface(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_target_surface(child, label)
