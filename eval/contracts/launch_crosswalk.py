"""Target-free, inventory-complete launch identity crosswalks."""

from __future__ import annotations

import re
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
    source.snapshot.require_production_acquisition("production launch source")
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
        "production crosswalk evaluation has no installed namespace parser authority",
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
        "production crosswalk verification has no installed namespace parser authority",
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
        "production crosswalk loading has no installed namespace parser authority",
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
