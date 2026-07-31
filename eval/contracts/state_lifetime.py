"""Retained-byte state-lifetime classification for offline evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable

from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    canonical_json_bytes,
    sha256_bytes,
    verify_model_seal,
)

from ._common import (
    AuthorityBlockerCode,
    ContractOutcome,
    OutcomeStatus,
    authority_unavailable,
    evaluation_boundary_model,
    outcome_from_model,
    require_exact_evaluation_boundary,
)
from .launch_crosswalk import (
    INVENTORY_SCHEMA_ID,
    INVENTORY_SCHEMA_SHA256,
    SCHEMA_ID as CROSSWALK_SCHEMA_ID,
    SCHEMA_SHA256 as CROSSWALK_SCHEMA_SHA256,
    FIXTURE_CONTRACT_MODE as CROSSWALK_FIXTURE_CONTRACT_MODE,
    LaunchNamespace,
    verify_launch_crosswalk_fixture_model,
    verify_launch_crosswalk_model,
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
ARTIFACT_KIND = "EVALUATOR_STATE_LIFETIME_CLASSIFICATION"
SCHEMA_ID = "https://gpu-simulator.invalid/eval/schemas/state_lifetime.v1.schema.json"
SCHEMA_SHA256 = "261203e48f5369711209a7baf8aa89aee24919acb952eb91d30e5e0b63ffb3f7"
EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_ARTIFACT_KIND = "EVALUATOR_STATE_LIFETIME_EVIDENCE"
EVIDENCE_SCHEMA_ID = SCHEMA_ID
EVIDENCE_SCHEMA_SHA256 = SCHEMA_SHA256
_RECORD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
FIXTURE_CONTRACT_MODE = "FIXTURE_ONLY"
FIXTURE_EVIDENCE_AUTHORITY_STATUS = "FIXTURE_ONLY_CALLER_ASSERTIONS"
_FIXTURE_EVIDENCE_AUTHORITY_CORE = {
    "status": FIXTURE_EVIDENCE_AUTHORITY_STATUS,
    "authority_id": "FIXTURE_STATE_LIFETIME_ASSERTIONS_V1",
    "derivation": "CALLER_ENUM_ASSERTIONS_NOT_PRODUCTION_EVIDENCE",
}
FIXTURE_EVIDENCE_AUTHORITY_SHA256 = sha256_bytes(
    canonical_json_bytes(_FIXTURE_EVIDENCE_AUTHORITY_CORE)
)
# External state-evidence authority is deliberately absent.
PRODUCTION_STATE_EVIDENCE_AUTHORITIES = MappingProxyType({})


class EvidenceBasis(str, Enum):
    EXACT = "EXACT"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class StateOrigin(str, Enum):
    FRESH = "FRESH"
    PERSISTENT = "PERSISTENT"
    UNKNOWN = "UNKNOWN"


class CacheCondition(str, Enum):
    COLD = "COLD"
    EVICTED = "EVICTED"
    WARM = "WARM"
    UNKNOWN = "UNKNOWN"


class LaunchScope(str, Enum):
    SINGLE_LAUNCH = "SINGLE_LAUNCH"
    MULTI_LAUNCH = "MULTI_LAUNCH"
    UNKNOWN = "UNKNOWN"


class ExecutionRelation(str, Enum):
    INDEPENDENT = "INDEPENDENT"
    DEPENDENT = "DEPENDENT"
    CONCURRENT = "CONCURRENT"
    UNKNOWN = "UNKNOWN"


class IsolationState(str, Enum):
    ISOLATED = "ISOLATED"
    NON_ISOLATED = "NON_ISOLATED"
    UNKNOWN = "UNKNOWN"


class StreamScope(str, Enum):
    SINGLE_STREAM = "SINGLE_STREAM"
    MULTI_STREAM = "MULTI_STREAM"
    UNKNOWN = "UNKNOWN"


class ProcessScope(str, Enum):
    SINGLE_PROCESS = "SINGLE_PROCESS"
    MULTI_PROCESS = "MULTI_PROCESS"
    UNKNOWN = "UNKNOWN"


class DeviceScope(str, Enum):
    SINGLE_DEVICE = "SINGLE_DEVICE"
    MULTI_DEVICE = "MULTI_DEVICE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class StateLifetimeEvidence:
    evidence_record_id: str
    evidence_basis: EvidenceBasis
    state_origin: StateOrigin
    cache_condition: CacheCondition
    launch_scope: LaunchScope
    execution_relation: ExecutionRelation
    isolation_state: IsolationState
    stream_scope: StreamScope
    process_scope: ProcessScope
    device_scope: DeviceScope
    evidence_authority_status: str
    evidence_authority_sha256: str
    crosswalk_model_seal_sha256: str
    canonical_qualified_launch_sha256: str
    ncu_qualified_launch_sha256: str
    nvbit_qualified_launch_sha256: str
    simulator_qualified_launch_sha256: str
    record_identity_sha256: str

    def __post_init__(self) -> None:
        _require_record_id(self.evidence_record_id)
        for value, enum_type, label in _enum_fields(self):
            _require_enum(value, enum_type, label)
        if self.evidence_authority_status != FIXTURE_EVIDENCE_AUTHORITY_STATUS:
            raise ImmutableArtifactError("state evidence authority status is invalid")
        if self.evidence_authority_sha256 != FIXTURE_EVIDENCE_AUTHORITY_SHA256:
            raise ImmutableArtifactError("state evidence authority SHA-256 is not pinned")
        for field in (
            "evidence_authority_sha256",
            "crosswalk_model_seal_sha256",
            "canonical_qualified_launch_sha256",
            "ncu_qualified_launch_sha256",
            "nvbit_qualified_launch_sha256",
            "simulator_qualified_launch_sha256",
            "record_identity_sha256",
        ):
            require_sha256(getattr(self, field), field)
        if self.record_identity_sha256 != _record_identity_sha256(self):
            raise ImmutableArtifactError("state evidence record identity mismatch")

    @classmethod
    def create(cls, **_inputs: Any) -> "StateLifetimeEvidence":
        raise authority_unavailable(
            AuthorityBlockerCode.STATE_EVIDENCE_AUTHORITY_UNAVAILABLE,
            "the production state-evidence authority registry is empty",
        )

    @classmethod
    def create_fixture(
        cls,
        *,
        evidence_record_id: str,
        evidence_basis: EvidenceBasis,
        state_origin: StateOrigin,
        cache_condition: CacheCondition,
        launch_scope: LaunchScope,
        execution_relation: ExecutionRelation,
        isolation_state: IsolationState,
        stream_scope: StreamScope,
        process_scope: ProcessScope,
        device_scope: DeviceScope,
        crosswalk_model: dict[str, Any],
        crosswalk_schema_bytes: bytes,
        crosswalk_schema_snapshot: RetainedPayloadSnapshot,
        inventory_schema_bytes: bytes,
        inventory_schema_snapshot: RetainedPayloadSnapshot,
    ) -> "StateLifetimeEvidence":
        verify_launch_crosswalk_fixture_model(
            crosswalk_model,
            schema_bytes=crosswalk_schema_bytes,
            schema_snapshot=crosswalk_schema_snapshot,
            inventory_schema_bytes=inventory_schema_bytes,
            inventory_schema_snapshot=inventory_schema_snapshot,
        )
        selected = _selected_launch_ids(crosswalk_model)
        arguments = {
            "evidence_record_id": evidence_record_id,
            "evidence_basis": evidence_basis,
            "state_origin": state_origin,
            "cache_condition": cache_condition,
            "launch_scope": launch_scope,
            "execution_relation": execution_relation,
            "isolation_state": isolation_state,
            "stream_scope": stream_scope,
            "process_scope": process_scope,
            "device_scope": device_scope,
            "evidence_authority_status": FIXTURE_EVIDENCE_AUTHORITY_STATUS,
            "evidence_authority_sha256": FIXTURE_EVIDENCE_AUTHORITY_SHA256,
            "crosswalk_model_seal_sha256": crosswalk_model["model_seal_sha256"],
            "canonical_qualified_launch_sha256": selected["canonical"],
            "ncu_qualified_launch_sha256": selected["ncu"],
            "nvbit_qualified_launch_sha256": selected["nvbit"],
            "simulator_qualified_launch_sha256": selected["simulator"],
        }
        digest = _record_identity_from_arguments(arguments)
        return cls(**arguments, record_identity_sha256=digest)

    def to_model(self) -> dict[str, Any]:
        return {
            "evidence_record_id": self.evidence_record_id,
            "evidence_basis": self.evidence_basis.value,
            "state_origin": self.state_origin.value,
            "cache_condition": self.cache_condition.value,
            "launch_scope": self.launch_scope.value,
            "execution_relation": self.execution_relation.value,
            "isolation_state": self.isolation_state.value,
            "stream_scope": self.stream_scope.value,
            "process_scope": self.process_scope.value,
            "device_scope": self.device_scope.value,
            "evidence_authority_status": self.evidence_authority_status,
            "evidence_authority_sha256": self.evidence_authority_sha256,
            "crosswalk_model_seal_sha256": self.crosswalk_model_seal_sha256,
            "selected_qualified_launch_ids": {
                "canonical": self.canonical_qualified_launch_sha256,
                "ncu": self.ncu_qualified_launch_sha256,
                "nvbit": self.nvbit_qualified_launch_sha256,
                "simulator": self.simulator_qualified_launch_sha256,
            },
            "record_identity_sha256": self.record_identity_sha256,
        }

    @classmethod
    def from_model(cls, value: object) -> "StateLifetimeEvidence":
        require_exact_keys(value, _EVIDENCE_RECORD_KEYS, "state evidence record")
        assert isinstance(value, dict)
        selected = value["selected_qualified_launch_ids"]
        require_exact_keys(
            selected, {"canonical", "ncu", "nvbit", "simulator"}, "selected launch IDs"
        )
        assert isinstance(selected, dict)
        return cls(
            evidence_record_id=value["evidence_record_id"],
            evidence_basis=_parse_enum(EvidenceBasis, value["evidence_basis"], "evidence_basis"),
            state_origin=_parse_enum(StateOrigin, value["state_origin"], "state_origin"),
            cache_condition=_parse_enum(
                CacheCondition, value["cache_condition"], "cache_condition"
            ),
            launch_scope=_parse_enum(LaunchScope, value["launch_scope"], "launch_scope"),
            execution_relation=_parse_enum(
                ExecutionRelation, value["execution_relation"], "execution_relation"
            ),
            isolation_state=_parse_enum(
                IsolationState, value["isolation_state"], "isolation_state"
            ),
            stream_scope=_parse_enum(StreamScope, value["stream_scope"], "stream_scope"),
            process_scope=_parse_enum(
                ProcessScope, value["process_scope"], "process_scope"
            ),
            device_scope=_parse_enum(DeviceScope, value["device_scope"], "device_scope"),
            evidence_authority_status=value["evidence_authority_status"],
            evidence_authority_sha256=value["evidence_authority_sha256"],
            crosswalk_model_seal_sha256=value["crosswalk_model_seal_sha256"],
            canonical_qualified_launch_sha256=selected["canonical"],
            ncu_qualified_launch_sha256=selected["ncu"],
            nvbit_qualified_launch_sha256=selected["nvbit"],
            simulator_qualified_launch_sha256=selected["simulator"],
            record_identity_sha256=value["record_identity_sha256"],
        )


_EVIDENCE_RECORD_KEYS = {
    "evidence_record_id",
    "evidence_basis",
    "state_origin",
    "cache_condition",
    "launch_scope",
    "execution_relation",
    "isolation_state",
    "stream_scope",
    "process_scope",
    "device_scope",
    "evidence_authority_status",
    "evidence_authority_sha256",
    "crosswalk_model_seal_sha256",
    "selected_qualified_launch_ids",
    "record_identity_sha256",
}


def create_state_evidence_payload(
    records: Iterable[StateLifetimeEvidence],
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    del records, schema_bytes, schema_snapshot
    raise authority_unavailable(
        AuthorityBlockerCode.STATE_EVIDENCE_AUTHORITY_UNAVAILABLE,
        "production state evidence cannot be created without an external authority",
    )


def create_state_evidence_payload_fixture(
    records: Iterable[StateLifetimeEvidence],
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    verify_retained_schema(
        schema_bytes,
        schema_snapshot,
        expected_sha256=EVIDENCE_SCHEMA_SHA256,
        expected_id=EVIDENCE_SCHEMA_ID,
    )
    material = tuple(sorted(records, key=lambda item: item.evidence_record_id))
    if any(not isinstance(item, StateLifetimeEvidence) for item in material):
        raise ImmutableArtifactError("state evidence payload contains an invalid record")
    record_ids = [item.evidence_record_id for item in material]
    if len(record_ids) != len(set(record_ids)):
        raise ImmutableArtifactError("duplicate normalized state evidence record identity")
    return seal_evaluator_model(
        {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "artifact_kind": EVIDENCE_ARTIFACT_KIND,
            "schema_sha256": EVIDENCE_SCHEMA_SHA256,
            "contract_mode": FIXTURE_CONTRACT_MODE,
            "evidence_authority": {
                **_FIXTURE_EVIDENCE_AUTHORITY_CORE,
                "authority_sha256": FIXTURE_EVIDENCE_AUTHORITY_SHA256,
            },
            "records": [item.to_model() for item in material],
        }
    )


def verify_state_evidence_payload(
    model: dict[str, Any],
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> tuple[StateLifetimeEvidence, ...]:
    del model, schema_bytes, schema_snapshot
    raise authority_unavailable(
        AuthorityBlockerCode.STATE_EVIDENCE_AUTHORITY_UNAVAILABLE,
        "production state-evidence verification has no installed authority",
    )


def verify_state_evidence_fixture_payload(
    model: dict[str, Any],
    *,
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
) -> tuple[StateLifetimeEvidence, ...]:
    verify_evaluator_model(
        model,
        expected_schema_version=EVIDENCE_SCHEMA_VERSION,
        expected_artifact_kind=EVIDENCE_ARTIFACT_KIND,
        expected_schema_sha256=EVIDENCE_SCHEMA_SHA256,
        expected_schema_id=EVIDENCE_SCHEMA_ID,
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
            "evidence_authority",
            "records",
            "model_seal_sha256",
        },
        "state evidence payload",
    )
    if model["contract_mode"] != FIXTURE_CONTRACT_MODE:
        raise ImmutableArtifactError("state evidence payload is not fixture-only")
    expected_authority = {
        **_FIXTURE_EVIDENCE_AUTHORITY_CORE,
        "authority_sha256": FIXTURE_EVIDENCE_AUTHORITY_SHA256,
    }
    if model["evidence_authority"] != expected_authority:
        raise ImmutableArtifactError("fixture state evidence authority mismatch")
    if type(model["records"]) is not list:
        raise ImmutableArtifactError("state evidence records must be an array")
    records = tuple(StateLifetimeEvidence.from_model(item) for item in model["records"])
    if records != tuple(sorted(records, key=lambda item: item.evidence_record_id)):
        raise ImmutableArtifactError("state evidence records are not normalized and sorted")
    ids = [item.evidence_record_id for item in records]
    if len(ids) != len(set(ids)):
        raise ImmutableArtifactError("duplicate normalized state evidence record identity")
    return records


def classify_state_lifetime(**_inputs: Any) -> dict[str, Any]:
    raise authority_unavailable(
        AuthorityBlockerCode.STATE_EVIDENCE_AUTHORITY_UNAVAILABLE,
        "production state classification has no retained authority-bound evidence",
    )


def classify_state_lifetime_fixture(
    *,
    evidence_payload: bytes | None,
    evidence_snapshot: RetainedPayloadSnapshot | None,
    evidence_record_id: str,
    crosswalk_model: dict[str, Any],
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    evidence_schema_bytes: bytes,
    evidence_schema_snapshot: RetainedPayloadSnapshot,
    crosswalk_schema_bytes: bytes,
    crosswalk_schema_snapshot: RetainedPayloadSnapshot,
    inventory_schema_bytes: bytes,
    inventory_schema_snapshot: RetainedPayloadSnapshot,
) -> dict[str, Any]:
    """Classify retained fixture bytes tied to one fixture crosswalk record."""

    verify_retained_schema(
        schema_bytes,
        schema_snapshot,
        expected_sha256=SCHEMA_SHA256,
        expected_id=SCHEMA_ID,
    )
    _require_record_id(evidence_record_id)
    record, source, unavailable_reason = _resolve_evidence(
        evidence_payload=evidence_payload,
        evidence_snapshot=evidence_snapshot,
        evidence_record_id=evidence_record_id,
        crosswalk_model=crosswalk_model,
        evidence_schema_bytes=evidence_schema_bytes,
        evidence_schema_snapshot=evidence_schema_snapshot,
        crosswalk_schema_bytes=crosswalk_schema_bytes,
        crosswalk_schema_snapshot=crosswalk_schema_snapshot,
        inventory_schema_bytes=inventory_schema_bytes,
        inventory_schema_snapshot=inventory_schema_snapshot,
    )
    if unavailable_reason is None:
        assert record is not None
        outcome = _classify_outcome(record)
        if outcome.status is OutcomeStatus.ADMITTED:
            outcome = ContractOutcome(
                OutcomeStatus.UNAVAILABLE,
                ("FIXTURE_ONLY_STATE_EVIDENCE",),
            )
        evidence_model: dict[str, Any] | None = record.to_model()
    else:
        outcome = ContractOutcome(OutcomeStatus.UNAVAILABLE, (unavailable_reason,))
        evidence_model = None
    return seal_evaluator_model(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "schema_sha256": SCHEMA_SHA256,
            "contract_mode": FIXTURE_CONTRACT_MODE,
            "evaluation_boundary": evaluation_boundary_model(),
            "evidence_source": source,
            "evidence": evidence_model,
            "outcome": outcome.to_model(),
        }
    )


def verify_state_lifetime_model(
    model: dict[str, Any],
    **_inputs: Any,
) -> None:
    del model
    raise authority_unavailable(
        AuthorityBlockerCode.STATE_EVIDENCE_AUTHORITY_UNAVAILABLE,
        "production state-classification verification has no installed authority",
    )


def verify_state_lifetime_fixture_model(
    model: dict[str, Any],
    *,
    evidence_payload: bytes | None,
    evidence_snapshot: RetainedPayloadSnapshot | None,
    evidence_record_id: str,
    crosswalk_model: dict[str, Any],
    schema_bytes: bytes,
    schema_snapshot: RetainedPayloadSnapshot,
    evidence_schema_bytes: bytes,
    evidence_schema_snapshot: RetainedPayloadSnapshot,
    crosswalk_schema_bytes: bytes,
    crosswalk_schema_snapshot: RetainedPayloadSnapshot,
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
            "evidence_source",
            "evidence",
            "outcome",
            "model_seal_sha256",
        },
        "state lifetime classification",
    )
    if model["contract_mode"] != FIXTURE_CONTRACT_MODE:
        raise ImmutableArtifactError("state lifetime classification is not fixture-only")
    require_exact_evaluation_boundary(model["evaluation_boundary"])
    outcome_from_model(
        model["outcome"],
        rejected_reasons=_REJECTED_REASONS,
        unavailable_reasons=_UNAVAILABLE_REASONS,
    )
    expected = classify_state_lifetime_fixture(
        evidence_payload=evidence_payload,
        evidence_snapshot=evidence_snapshot,
        evidence_record_id=evidence_record_id,
        crosswalk_model=crosswalk_model,
        schema_bytes=schema_bytes,
        schema_snapshot=schema_snapshot,
        evidence_schema_bytes=evidence_schema_bytes,
        evidence_schema_snapshot=evidence_schema_snapshot,
        crosswalk_schema_bytes=crosswalk_schema_bytes,
        crosswalk_schema_snapshot=crosswalk_schema_snapshot,
        inventory_schema_bytes=inventory_schema_bytes,
        inventory_schema_snapshot=inventory_schema_snapshot,
    )
    if canonical_json_bytes(model) != canonical_json_bytes(expected):
        raise ImmutableArtifactError("state lifetime classification does not match retained bytes")


def load_state_lifetime_classification(
    payload: bytes,
    **_verification_inputs: Any,
) -> dict[str, Any]:
    del payload
    raise authority_unavailable(
        AuthorityBlockerCode.STATE_EVIDENCE_AUTHORITY_UNAVAILABLE,
        "production state-classification loading has no installed authority",
    )


def load_state_lifetime_fixture_classification(
    payload: bytes,
    **verification_inputs: Any,
) -> dict[str, Any]:
    model = strict_json_object(payload, label="state lifetime classification")
    verify_state_lifetime_fixture_model(model, **verification_inputs)
    return model


def _resolve_evidence(
    *,
    evidence_payload: bytes | None,
    evidence_snapshot: RetainedPayloadSnapshot | None,
    evidence_record_id: str,
    crosswalk_model: dict[str, Any],
    evidence_schema_bytes: bytes,
    evidence_schema_snapshot: RetainedPayloadSnapshot,
    crosswalk_schema_bytes: bytes,
    crosswalk_schema_snapshot: RetainedPayloadSnapshot,
    inventory_schema_bytes: bytes,
    inventory_schema_snapshot: RetainedPayloadSnapshot,
) -> tuple[StateLifetimeEvidence | None, dict[str, Any] | None, str | None]:
    try:
        verify_launch_crosswalk_fixture_model(
            crosswalk_model,
            schema_bytes=crosswalk_schema_bytes,
            schema_snapshot=crosswalk_schema_snapshot,
            inventory_schema_bytes=inventory_schema_bytes,
            inventory_schema_snapshot=inventory_schema_snapshot,
        )
        selected = _selected_launch_ids(crosswalk_model)
    except (ImmutableArtifactError, KeyError, TypeError, ValueError):
        return None, None, "CROSSWALK_UNAVAILABLE"
    if evidence_payload is None or evidence_snapshot is None:
        return None, None, "EXACT_EVIDENCE_PAYLOAD_UNAVAILABLE"
    if not isinstance(evidence_snapshot, RetainedPayloadSnapshot):
        return None, None, "EXACT_EVIDENCE_PAYLOAD_UNAVAILABLE"
    try:
        verify_retained_schema(
            evidence_schema_bytes,
            evidence_schema_snapshot,
            expected_sha256=EVIDENCE_SCHEMA_SHA256,
            expected_id=EVIDENCE_SCHEMA_ID,
        )
    except ImmutableArtifactError:
        return None, None, "EVIDENCE_SCHEMA_UNAVAILABLE"
    try:
        evidence_snapshot.require_fixture_acquisition("fixture state evidence payload")
        evidence_snapshot.verify_payload(evidence_payload)
        payload_model = strict_json_object(
            evidence_payload, label="state lifetime evidence payload"
        )
        records = verify_state_evidence_fixture_payload(
            payload_model,
            schema_bytes=evidence_schema_bytes,
            schema_snapshot=evidence_schema_snapshot,
        )
    except (ImmutableArtifactError, KeyError, TypeError, ValueError):
        return None, None, "EVIDENCE_PAYLOAD_INVALID"
    matches = tuple(
        item for item in records if item.evidence_record_id == evidence_record_id
    )
    if len(matches) != 1:
        return None, None, "EVIDENCE_RECORD_UNAVAILABLE"
    record = matches[0]
    if (
        record.crosswalk_model_seal_sha256 != crosswalk_model["model_seal_sha256"]
        or record.canonical_qualified_launch_sha256 != selected["canonical"]
        or record.ncu_qualified_launch_sha256 != selected["ncu"]
        or record.nvbit_qualified_launch_sha256 != selected["nvbit"]
        or record.simulator_qualified_launch_sha256 != selected["simulator"]
    ):
        return None, None, "EVIDENCE_CROSSWALK_BINDING_UNAVAILABLE"
    source = {
        "payload_snapshot": evidence_snapshot.to_model(),
        "payload_model_seal_sha256": payload_model["model_seal_sha256"],
        "evidence_record_id": record.evidence_record_id,
        "record_identity_sha256": record.record_identity_sha256,
        "crosswalk_model_seal_sha256": crosswalk_model["model_seal_sha256"],
        "selected_qualified_launch_ids": selected,
    }
    return record, source, None


def _classify_outcome(evidence: StateLifetimeEvidence) -> ContractOutcome:
    rejected: list[str] = []
    if evidence.cache_condition is CacheCondition.WARM:
        rejected.append("WARM_STATE_REJECTED")
    if evidence.state_origin is StateOrigin.PERSISTENT:
        rejected.append("PERSISTENT_STATE_REJECTED")
    if evidence.execution_relation is ExecutionRelation.CONCURRENT:
        rejected.append("CONCURRENT_EXECUTION_REJECTED")
    if evidence.execution_relation is ExecutionRelation.DEPENDENT:
        rejected.append("DEPENDENT_EXECUTION_REJECTED")
    if evidence.isolation_state is IsolationState.NON_ISOLATED:
        rejected.append("NON_ISOLATED_EXECUTION_REJECTED")
    if evidence.launch_scope is LaunchScope.MULTI_LAUNCH:
        rejected.append("MULTI_LAUNCH_REJECTED")
    if evidence.stream_scope is StreamScope.MULTI_STREAM:
        rejected.append("MULTI_STREAM_REJECTED")
    if evidence.process_scope is ProcessScope.MULTI_PROCESS:
        rejected.append("MULTI_PROCESS_REJECTED")
    if evidence.device_scope is DeviceScope.MULTI_DEVICE:
        rejected.append("MULTI_DEVICE_REJECTED")
    if rejected:
        return ContractOutcome(OutcomeStatus.REJECTED, tuple(rejected))

    unavailable: list[str] = []
    if evidence.evidence_basis is not EvidenceBasis.EXACT:
        unavailable.append("EXACT_EVIDENCE_UNAVAILABLE")
    if evidence.state_origin is StateOrigin.UNKNOWN:
        unavailable.append("FRESH_STATE_UNAVAILABLE")
    if evidence.cache_condition is CacheCondition.EVICTED:
        unavailable.append("EVICTED_IS_NOT_EXACT_COLD")
    elif evidence.cache_condition is CacheCondition.UNKNOWN:
        unavailable.append("COLD_STATE_UNAVAILABLE")
    if evidence.launch_scope is LaunchScope.UNKNOWN:
        unavailable.append("SINGLE_LAUNCH_UNAVAILABLE")
    if evidence.execution_relation is ExecutionRelation.UNKNOWN:
        unavailable.append("INDEPENDENCE_UNAVAILABLE")
    if evidence.isolation_state is IsolationState.UNKNOWN:
        unavailable.append("ISOLATION_UNAVAILABLE")
    if evidence.stream_scope is StreamScope.UNKNOWN:
        unavailable.append("SINGLE_STREAM_UNAVAILABLE")
    if evidence.process_scope is ProcessScope.UNKNOWN:
        unavailable.append("SINGLE_PROCESS_UNAVAILABLE")
    if evidence.device_scope is DeviceScope.UNKNOWN:
        unavailable.append("SINGLE_DEVICE_UNAVAILABLE")
    if unavailable:
        return ContractOutcome(OutcomeStatus.UNAVAILABLE, tuple(unavailable))

    exact = (
        evidence.evidence_basis is EvidenceBasis.EXACT
        and evidence.state_origin is StateOrigin.FRESH
        and evidence.cache_condition is CacheCondition.COLD
        and evidence.launch_scope is LaunchScope.SINGLE_LAUNCH
        and evidence.execution_relation is ExecutionRelation.INDEPENDENT
        and evidence.isolation_state is IsolationState.ISOLATED
        and evidence.stream_scope is StreamScope.SINGLE_STREAM
        and evidence.process_scope is ProcessScope.SINGLE_PROCESS
        and evidence.device_scope is DeviceScope.SINGLE_DEVICE
    )
    if not exact:
        raise ImmutableArtifactError("state lifetime enum space is not total")
    return ContractOutcome(OutcomeStatus.ADMITTED, ())


_REJECTED_REASONS = frozenset(
    {
        "WARM_STATE_REJECTED",
        "PERSISTENT_STATE_REJECTED",
        "CONCURRENT_EXECUTION_REJECTED",
        "DEPENDENT_EXECUTION_REJECTED",
        "NON_ISOLATED_EXECUTION_REJECTED",
        "MULTI_LAUNCH_REJECTED",
        "MULTI_STREAM_REJECTED",
        "MULTI_PROCESS_REJECTED",
        "MULTI_DEVICE_REJECTED",
    }
)
_UNAVAILABLE_REASONS = frozenset(
    {
        "EXACT_EVIDENCE_UNAVAILABLE",
        "FRESH_STATE_UNAVAILABLE",
        "EVICTED_IS_NOT_EXACT_COLD",
        "COLD_STATE_UNAVAILABLE",
        "SINGLE_LAUNCH_UNAVAILABLE",
        "INDEPENDENCE_UNAVAILABLE",
        "ISOLATION_UNAVAILABLE",
        "SINGLE_STREAM_UNAVAILABLE",
        "SINGLE_PROCESS_UNAVAILABLE",
        "SINGLE_DEVICE_UNAVAILABLE",
        "FIXTURE_ONLY_STATE_EVIDENCE",
        "CROSSWALK_UNAVAILABLE",
        "EXACT_EVIDENCE_PAYLOAD_UNAVAILABLE",
        "EVIDENCE_SCHEMA_UNAVAILABLE",
        "EVIDENCE_PAYLOAD_INVALID",
        "EVIDENCE_RECORD_UNAVAILABLE",
        "EVIDENCE_CROSSWALK_BINDING_UNAVAILABLE",
    }
)


def _selected_launch_ids(crosswalk: dict[str, Any]) -> dict[str, str]:
    if crosswalk.get("contract_mode") != CROSSWALK_FIXTURE_CONTRACT_MODE:
        raise ImmutableArtifactError("fixture state evidence requires a fixture crosswalk")
    if crosswalk.get("outcome", {}).get("status") != "UNAVAILABLE":
        raise ImmutableArtifactError("fixture crosswalk must remain production-unavailable")
    result: dict[str, str] = {}
    for namespace in LaunchNamespace:
        selected = crosswalk["namespaces"][namespace.value.lower()]["selected_launch"]
        if type(selected) is not dict:
            raise ImmutableArtifactError("crosswalk selected launch is unavailable")
        digest = selected["identity"]["qualified_launch_sha256"]
        require_sha256(digest, "selected qualified launch identity")
        result[namespace.value.lower()] = digest
    return result


def _record_identity_from_arguments(arguments: dict[str, Any]) -> str:
    selected = {
        "canonical": arguments["canonical_qualified_launch_sha256"],
        "ncu": arguments["ncu_qualified_launch_sha256"],
        "nvbit": arguments["nvbit_qualified_launch_sha256"],
        "simulator": arguments["simulator_qualified_launch_sha256"],
    }
    model = {
        "evidence_record_id": arguments["evidence_record_id"],
        "evidence_basis": arguments["evidence_basis"].value,
        "state_origin": arguments["state_origin"].value,
        "cache_condition": arguments["cache_condition"].value,
        "launch_scope": arguments["launch_scope"].value,
        "execution_relation": arguments["execution_relation"].value,
        "isolation_state": arguments["isolation_state"].value,
        "stream_scope": arguments["stream_scope"].value,
        "process_scope": arguments["process_scope"].value,
        "device_scope": arguments["device_scope"].value,
        "evidence_authority_status": arguments["evidence_authority_status"],
        "evidence_authority_sha256": arguments["evidence_authority_sha256"],
        "crosswalk_model_seal_sha256": arguments["crosswalk_model_seal_sha256"],
        "selected_qualified_launch_ids": selected,
    }
    return sha256_bytes(canonical_json_bytes(model))


def _record_identity_sha256(evidence: StateLifetimeEvidence) -> str:
    model = evidence.to_model()
    model.pop("record_identity_sha256")
    return sha256_bytes(canonical_json_bytes(model))


def _enum_fields(evidence: StateLifetimeEvidence) -> tuple[tuple[Any, type[Enum], str], ...]:
    return (
        (evidence.evidence_basis, EvidenceBasis, "evidence_basis"),
        (evidence.state_origin, StateOrigin, "state_origin"),
        (evidence.cache_condition, CacheCondition, "cache_condition"),
        (evidence.launch_scope, LaunchScope, "launch_scope"),
        (evidence.execution_relation, ExecutionRelation, "execution_relation"),
        (evidence.isolation_state, IsolationState, "isolation_state"),
        (evidence.stream_scope, StreamScope, "stream_scope"),
        (evidence.process_scope, ProcessScope, "process_scope"),
        (evidence.device_scope, DeviceScope, "device_scope"),
    )


def _parse_enum(enum_type: type[Enum], value: object, label: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ImmutableArtifactError(f"{label} is invalid") from exc


def _require_enum(value: object, enum_type: type[Enum], label: str) -> None:
    if not isinstance(value, enum_type):
        raise ImmutableArtifactError(f"{label} is invalid")


def _require_record_id(value: object) -> str:
    require_nfc(value, "evidence_record_id")
    if type(value) is not str or _RECORD_ID.fullmatch(value) is None:
        raise ImmutableArtifactError("evidence_record_id is invalid")
    return value
